



############################################################################
#                          Main Training Function                          #
############################################################################

function mklocsvmtrain(
    Kernels::Vector{<:Kernel}, 
    X::AbstractMatrix{Float64};
    ν::Float64=0.01, 
    μ::Float64=0.05,
    ϵ::Float64=1e-5,
    algorithm::MKLOCSVM_Algorithm=QCQP(),
    verbose::Bool=true
)

    if verbose
        println("Computing basis kernel matrices...")
    end
    KM = multiplekernelmatrices(Kernels, X)
    if verbose
        println("Done!")
    end
    if verbose
        println("Multiple kernel learning with algorithm $(typeof(algorithm))...")
    end
    α, π, ρ = algorithm(KM; ν=ν, μ=μ)
    if verbose
        println("Done!")
    end
    if verbose
        println("Building the model...")
    end
    model = mklocsvmbuild(X, Kernels, ν, μ, α, π, ρ; ϵ=ϵ)
    if verbose
        println("Done!")
    end

    if verbose
        println("All Done!")
    end
    
    return model
end

function mklocsvmtrain(
    X::AbstractMatrix{Float64},
    P::Int64=21;
    kernel::String="DPDK",
    q_mode::String="randomly",
    c_mode::String="span", 
    κ::Float64=1.5, 
    c_ϵ::Float64=0.05,
    num_batch::Int64=1,
    batch_mode::String="systematacially",
    ν::Float64=0.01, 
    μ::Float64=0.2,
    ϵ::Float64=1e-5,
    algorithm::MKLOCSVM_Algorithm=QCQP(),
    verbose::Bool=true
)
    verbose ? println("Creating Basis $(kernel)...") : nothing
    if kernel == "DPDK"
        kernels = create_basis_DPDKs(
            X,
            P;
            q_mode=q_mode,
            c_mode=c_mode, 
            κ=κ, 
            ϵ=c_ϵ
        )
    elseif kernel == "DNPNK"
        kernels = create_basis_DNPNKs(
            X,
            P;
            q_mode=q_mode,
            κ=κ, 
        )
    end
    verbose ? println("Done!") : nothing

    
    if num_batch == 1
        model = mklocsvmtrain(
            kernels,
            X;
            ν=ν, 
            μ=μ,
            ϵ=ϵ,
            algorithm=algorithm,
            verbose=verbose
        )
    else
        verbose ? println("Grouping the Kernels $(uppercasefirst(batch_mode)) into $num_batch Batches...") : nothing
        kernels_inbat = group_kernels(kernels, num_batch; mode=batch_mode)
        verbose ? println("Done!") : nothing

        model = pmap(kernels_inbat) do ks
            mklocsvmtrain(
                ks,
                X;
                ν=ν/num_batch, 
                μ=μ,
                ϵ=ϵ,
                algorithm=algorithm,
                verbose=verbose
            )
        end
    end

    return model
end








############################################################################
#                     Algorithms for MKL based OC-SVM                      #
############################################################################
function (qcqp::QCQP)(KM::Vector{Matrix{Float64}}; ν::Float64=0.01, μ::Float64=0.05)

    N = size(KM[1], 1)
    M = length(KM)

    MKLOCSVM_dual = Model(qcqp.solver.Optimizer)
    set_optimizer_attributes(MKLOCSVM_dual, "tol" => qcqp.tol, "max_iter" => qcqp.max_iter)
    if ~qcqp.verbose
        set_silent(MKLOCSVM_dual)
    end
    @variable(MKLOCSVM_dual, α[1:N])
    @variable(MKLOCSVM_dual, γ)
    @variable(MKLOCSVM_dual, ζ[1:M])
    @constraint(MKLOCSVM_dual, con_π[m = 1:M], -1/2 * α' * KM[m] * α >= γ - ζ[m])
    @constraint(MKLOCSVM_dual, ζ .>= 0)
    @constraint(MKLOCSVM_dual, con_ρ, sum(α) == 1)
    @constraint(MKLOCSVM_dual, 0 .<= α .<= 1/(N*ν))
    @objective(MKLOCSVM_dual, Min, -γ + 1/(M*μ) * sum(ζ))
    optimize!(MKLOCSVM_dual)

    α = value.(α)
    π = dual.(con_π)
    ρ = dual(con_ρ)

    return α, π, ρ
end






function (hessmkl::HessianMKL)(KM::Vector{Matrix{Float64}}; ν::Float64=0.01, μ::Float64=0.05)

    ϵ = hessmkl.ϵ
    N = size(KM[1], 1)
    M = length(KM)
    
    α = ones(N) / N
    π = ones(M) / M

    K = sum([KM[m] * π[m] for m in 1:M])
    J = 1/2 * α' * K * α
    
    # ==== Initialize the QP for the Newton step ====
    NewtonStep = Model(hessmkl.step_solver.Optimizer)
    set_silent(NewtonStep)
    @variable(NewtonStep, s[1:M])
    @constraint(NewtonStep, sum(s) == 0)
    @constraint(NewtonStep, con, 0 .<= π + s .<= 1/(M*μ))
    # ===============================================

    while true
        
        # ==== Solve standard single-kernel one-class SVM ====
        if hessmkl.svm_solver == LIBSVM
            svm = svmtrain(K; svmtype=OneClassSVM, kernel=LIBSVM_Kernel.Precomputed, nu=ν, tolerance=hessmkl.ϵ, verbose=hessmkl.verbose)
            α = zeros(N)
            α[svm.SVs.indices] = svm.coefs[:] / (N*ν)
        elseif false # Add code here if there are other svm_solvers
        end
        # ====================================================
        J_new = 1/2 * α' * K * α
        J = J_new

        SV = α .> ϵ
        BSV = ϵ .< α .< 1/(N*ν) - ϵ
        if sum(BSV) == 0
            ind = findall(SV)[argmin(α[SV])]
            BSV[ind] = true
        end
        Λ = [K[BSV, BSV] ones(sum(BSV)); ones(sum(BSV))' 0]
        _Λ = inv(Λ)[1:end-1, 1:end-1]
        
        grad = [-1/2 * α' * KM[m] * α for m in 1:M]
        Hess = [α[SV]' * KM[m][SV, BSV] * _Λ * KM[n][BSV, SV] * α[SV] for m in 1:M, n in 1:M]
        while ~isposdef(Hess)
            Hess = (Hess + Hess') / 2 + ϵ * mean(diag(Hess)) * I(M)
        end

        # ==== Update and solve the QP for the Newton step ====
        delete(NewtonStep, con); unregister(NewtonStep, :con)
        @constraint(NewtonStep, con, 0 .<= π + s .<= 1/(M*μ))
        @objective(NewtonStep, Min, 1/2 * s' * Hess * s + grad' * s)
        optimize!(NewtonStep)
        step = value.(s)
        # =====================================================
        π = π + step
        K = sum([KM[m] * π[m] for m in 1:M])
        J_new = 1/2 * α' * K * α
        if J - J_new <= ϵ * J
            break
        end
        J = J_new

    end
    SV = α .> ϵ
    BSV = ϵ .< α .< 1/(N*ν) - ϵ
    if sum(BSV) == 0
        ind = findall(SV)[argmin(α[SV])]
        BSV[ind] = true
    end
    ρ = α' * (π' * KM) * BSV / sum(BSV)

    return α, π, ρ
end





