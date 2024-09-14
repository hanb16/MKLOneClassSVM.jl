############################################################################
#                            Basis DPDK Kernel                             #
############################################################################
(kernel::DirectionalProjectionDistanceKernel)(u, v) = 1 - abs(kernel.q' * (u - v) / (kernel.c * kernel.κ))
Base.show(io::IO, kernel::DirectionalProjectionDistanceKernel) = print(io, "Directional Projection Distance Kernel (q = ", kernel.q, ", c = ", kernel.c, ", κ =  ", kernel.κ, ")")



"""
    create_basis_DPDKs(args...; kwargs...)

Create the candidate basis kernels according the following formula:
``` math
K_m(u, v) = 1 - |q_m^\\top (u - v) / (c_m \\cdot κ)|
```

# References
1. Han, B., Shang, C., & Huang, D. (2020). Multiple kernel learning-aided robust optimization: learning algorithm, computational tractability, and usage in multi-stage decision-making. European Journal of Operational Research. https://doi.org/10.1016/j.ejor.2020.11.027.
"""
function create_basis_DPDKs(
    X::AbstractMatrix{Float64},
    P::Int64;
    q_mode::String="randomly",
    c_mode::String="span", 
    κ::Float64=1.5, 
    ϵ::Float64=0.05
)
    n, N = size(X)
    Q = _create_directions(n, P; mode=q_mode)

    ProjX = pmap(q -> X' * q, Q)

    s = maximum.(ProjX) - minimum.(ProjX)
    if c_mode == "span" # the span of the projection
        c = s[:]
    elseif c_mode == "deviation" # the standard deviation of the projection
        c = std.(ProjX)
    elseif c_mode == "quantile" # ϵ-quantile based span
        c = quantile.(ProjX, 1-ϵ) - quantile.(ProjX, ϵ)
    end
    κ_min = maximum(s ./ c)
    if κ <= κ_min
        @error "`κ` should be a number larger than $κ_min if `c_mode=\"$c_mode\"`."
    end

    BasisKernels = fetch.(map([eachindex(Q);]) do m
        @spawnat :any DirectionalProjectionDistanceKernel(Q[m], c[m], κ)
    end)

    return BasisKernels
end

function KernelFunctions.kernelmatrix(
    kernel::DirectionalProjectionDistanceKernel,
    X::AbstractMatrix{<:Real}, 
    Y::AbstractMatrix{<:Real}; 
    obsdim=2
)
    1 .- pairwise(Minkowski(1), kernel.q' * X, kernel.q' * Y, dims=obsdim) / (kernel.c * kernel.κ)
end
function KernelFunctions.kernelmatrix(
    kernel::DirectionalProjectionDistanceKernel,
    X::AbstractMatrix{<:Real};
    obsdim=2
)
    1 .- pairwise(Minkowski(1), kernel.q' * X, dims=obsdim) / (kernel.c * kernel.κ)
end





############################################################################
#                           Basis DNPNK Kernel                             #
############################################################################
(kernel::DirectionalNullspaceProjectionNormKernel)(u, v) = 1 - Minkowski(1)(kernel.Q' * u, kernel.Q' * v) / (kernel.c * kernel.κ)
Base.show(io::IO, kernel::DirectionalNullspaceProjectionNormKernel) = print(io, "Directional Nullspace Projection Norm Kernel (Q = ", kernel.Q, ", c = ", kernel.c, ", κ =  ", kernel.κ, ")")


function create_basis_DNPNKs(
    X::AbstractMatrix{Float64},
    P::Int64;
    q_mode::String="randomly",
    κ::Float64=1.5
)
    n, N = size(X)
    qs = _create_directions(n, P; mode=q_mode)

    pmap(eachindex(qs);) do m
        Q = nullspace(qs[m]')
        c = maximum(pairwise(Minkowski(1), Q' * X, dims=2))
        DirectionalNullspaceProjectionNormKernel(Q, c, κ)
    end
end


function KernelFunctions.kernelmatrix(
    kernel::DirectionalNullspaceProjectionNormKernel,
    X::AbstractMatrix{<:Real}, 
    Y::AbstractMatrix{<:Real}; 
    obsdim=2
)
    1 .- pairwise(Minkowski(1), kernel.Q' * X, kernel.Q' * Y, dims=obsdim) / (kernel.c * kernel.κ)
end
function KernelFunctions.kernelmatrix(
    kernel::DirectionalNullspaceProjectionNormKernel,
    X::AbstractMatrix{<:Real};
    obsdim=2
)
    1 .- pairwise(Minkowski(1), kernel.Q' * X, dims=obsdim) / (kernel.c * kernel.κ)
end


############################################################################
#                                Utilities                                 #
############################################################################

function multiplekernelmatrices(
    Kernels::Vector{<:Kernel}, 
    X::AbstractMatrix{Float64}
)
    fetch.(map([eachindex(Kernels);]) do m
        @spawnat :any kernelmatrix(Kernels[m], X; obsdim=2)
    end)
end

function _create_directions(
    n::Int64,
    P::Int64;
    mode::String="evenly"
)
    if mode == "evenly"
        if iseven(P) && n > 2
            @warn "`P` is recommended to be an odd number to avoid duplicate directions in high-dimensional situations."
        end
        M = P^(n-1)
        Δψ = π / P
        pmap([1:M;]) do m
            ψ = digits(Int64, m-1; base=P, pad=n-1) * Δψ
            _q_ = map(k -> prod(cos.(ψ[n-1:-1:k])) * sin(ψ[k-1]), [2:(n-1);])
            [prod(cos.(ψ[n-1:-1:1])), _q_..., sin(ψ[n-1])]
        end
    elseif mode == "randomly"
        M = P
        pmap([1:M;]) do m
            q = 2rand(n) .- 1
            while norm(q) > 1
                q = 2rand(n) .- 1
            end
            q ./ norm(q)
        end
    end
end



function group_kernels(
    Kernels::Vector{<:Kernel}, 
    num_batch::Int64; 
    mode::String="systematacially"
)

    M = length(Kernels)
    bat_size = Int(floor(M / num_batch))
    if mode == "sequentially"
        Kernels_inbat = pmap([1:num_batch;]) do bat
            if bat < num_batch
                Kernels[(bat-1)*bat_size+1:bat*bat_size]
            else
                Kernels[(bat-1)*bat_size+1:end]
            end
        end
    elseif mode == "systematacially"
        Kernels_inbat = pmap([1:num_batch;]) do bat
            Kernels[bat:num_batch:end]
        end
    elseif mode == "randomly"
        ind = [1:M;]
        shuffle!(ind)
        Kernels_inbat = pmap([1:num_batch;]) do bat
            Kernels[ind[bat:num_batch:end]]
        end
    end

    return Kernels_inbat
end 
