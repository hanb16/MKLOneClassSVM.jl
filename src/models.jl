

function mklocsvmpredict(model::StandardModel, u::AbstractVector{<:Real})
    decision_function(model)(u) >= 0
end
function mklocsvmpredict(model::StandardModel, X::AbstractMatrix{<:Real})
    decision_function(model)(X) .>= 0
end
function mklocsvmpredict(model::Vector{StandardModel}, u::AbstractVector{<:Real})
    reduce(&, map(m -> mklocsvmpredict(m, u), model))
end
function mklocsvmpredict(model::Vector{StandardModel}, X::AbstractMatrix{<:Real})
    map(u -> mklocsvmpredict(model, u), eachcol(X))
end
function mklocsvmpredict(model::CombinedModel, u::AbstractVector{<:Real})
    model = mklocsvmunbuild(model)
    mklocsvmpredict(model, u)
end
function mklocsvmpredict(model::CombinedModel, X::AbstractMatrix{<:Real})
    model = mklocsvmunbuild(model)
    mklocsvmpredict(model, X)
end






function mklocsvmbuild(
    data::AbstractMatrix{Float64},
    kernels::Vector{<:Kernel}, 
    ν::Float64, 
    μ::Float64,
    α::AbstractVector{Float64},
    π::AbstractVector{Float64},
    ρ::Float64;
    ϵ::Float64=1e-5
)
    N = size(data, 2)
    
    SV = α .> ϵ
    # BSV = ϵ .< α .< 1/(N*ν) - ϵ
    # OSV = xor.(SV, BSV)
    OSV = α .> 1/(N*ν) - ϵ
    BSV = xor.(SV, OSV)
    IV = .~SV
    SK = π .> ϵ
    NSK = .~SK

    α_sv = α[SV]
    π_sk = π[SK]
    data_sv = data[:, SV]
    data_bsv = data[:, BSV]
    data_osv = data[:, OSV]
    data_iv = data[:, IV]
    kernels_sk = kernels[SK]
    kernels_nsk = kernels[NSK]
    IV = findall(IV)
    NSK = findall(NSK)
    SV = findall(SV)
    BSV = findall(BSV)
    OSV = findall(OSV)
    SK = findall(SK)

    model = StandardModel(
        data,
        kernels,
        ν,
        μ,
        α,
        π,
        ρ,
        ϵ,
        data_sv,
        data_bsv,
        data_osv,
        data_iv,
        kernels_sk,
        kernels_nsk,
        α_sv,
        π_sk,
        SV,
        BSV,
        OSV,
        IV,
        SK,
        NSK
    )

    return model
end
function mklocsvmbuild(
    model::StandardModel;
    ϵ::Float64
)
    data = model.data
    kernels = model.kernels
    ν = model.ν
    μ = model.μ
    α = model.α
    π = model.π
    ρ = model.ρ
    model = mklocsvmbuild(data, kernels, ν, μ, α, π, ρ; ϵ=ϵ)

    return model
end
function mklocsvmbuild(model::Vector{StandardModel})

    BAT = eachindex(model)
    data = model[1].data
    kernels = [model[bat].kernels for bat in BAT]
    ν = [model[bat].ν for bat in BAT]
    μ = [model[bat].μ for bat in BAT]
    α = [model[bat].α for bat in BAT]
    π = [model[bat].π for bat in BAT]
    ρ = [model[bat].ρ for bat in BAT]
    ϵ = [model[bat].ϵ for bat in BAT]
    data_sv = [model[bat].data_sv for bat in BAT]
    data_bsv = [model[bat].data_bsv for bat in BAT]
    data_bsv = [model[bat].data_bsv for bat in BAT]
    data_osv = [model[bat].data_osv for bat in BAT]
    data_iv = [model[bat].data_iv for bat in BAT]
    kernels_sk = [model[bat].kernels_sk for bat in BAT]
    kernels_nsk = [model[bat].kernels_nsk for bat in BAT]
    α_sv = [model[bat].α_sv for bat in BAT]
    π_sk = [model[bat].π_sk for bat in BAT]
    SV = [model[bat].SV for bat in BAT]
    BSV = [model[bat].BSV for bat in BAT]
    OSV = [model[bat].OSV for bat in BAT]
    IV = [model[bat].IV for bat in BAT]
    SK = [model[bat].SK for bat in BAT]
    NSK = [model[bat].NSK for bat in BAT]

    comb_OSV = Int64[]
    comb_IV = [1:size(data, 2);]
    for bat in BAT
        comb_OSV = comb_OSV ∪ model[bat].OSV
        comb_IV = comb_IV ∩ model[bat].IV
    end
    comb_BSV = setdiff([1:size(data, 2);], comb_OSV, comb_IV)
    comb_SV = comb_OSV ∪ comb_BSV

    comb_data_sv = data[:, comb_SV]
    comb_data_bsv = data[:, comb_BSV]
    comb_data_osv = data[:, comb_OSV]
    comb_data_iv = data[:, comb_IV]

    model = CombinedModel(
        data,
        kernels,
        ν,
        μ,
        α,
        π,
        ρ,
        ϵ,
        data_sv,
        data_bsv,
        data_osv,
        data_iv,
        kernels_sk,
        kernels_nsk,
        α_sv,
        π_sk,
        SV,
        BSV,
        OSV,
        IV,
        SK,
        NSK,
        comb_data_sv,
        comb_data_bsv,
        comb_data_osv,
        comb_data_iv,
        comb_SV,
        comb_BSV,
        comb_OSV,
        comb_IV
    )

    return model
end




function mklocsvmunbuild(model::CombinedModel)
    BAT = eachindex(model.kernels)
    modvec = StandardModel[]
    for bat in BAT
        m = StandardModel(
            model.data,
            model.kernels[bat],
            model.ν[bat],
            model.μ[bat],
            model.α[bat],
            model.π[bat],
            model.ρ[bat],
            model.ϵ[bat],
            model.data_sv[bat],
            model.data_bsv[bat],
            model.data_osv[bat],
            model.data_iv[bat],
            model.kernels_sk[bat],
            model.kernels_nsk[bat],
            model.α_sv[bat],
            model.π_sk[bat],
            model.SV[bat],
            model.BSV[bat],
            model.OSV[bat],
            model.IV[bat],
            model.SK[bat],
            model.NSK[bat]
        )
        push!(modvec, m)
    end
    
    return modvec
end







"""
Convert the `StandardModel` to a JuMP model. The default variable name of the set is `u`.
"""
function convert_to_jumpmodel(
    model::StandardModel;
    form::String="original",
    varname::Symbol=:u
)
    n, N = size(model.data)
    data_sv = model.data_sv
    kernels_sk = model.kernels_sk
    α_sv = model.α_sv
    π_sk = model.π_sk
    nsv = length(α_sv)
    nsk = length(π_sk)
    ρ = model.ρ

    if form == "original"
        SetMod = Model()
        u = @variable(SetMod, [1:n]; base_name=varname)
        SetMod[varname] = u
        @constraint(SetMod, sum(α_sv[i] * π_sk[m] * kernels_sk[m](u, data_sv[:,i]) for i in 1:nsv, m in 1:nsk) >= ρ)
    elseif form == "linear"
        if identity.(model.kernels_sk) isa Vector{<:PiecewiseLinearKernel}
            if identity.(model.kernels_sk) isa Vector{DirectionalProjectionDistanceKernel}
                SetMod = Model()
                u = @variable(SetMod, [1:n]; base_name=varname)
                SetMod[varname] = u
                θ = @variable(SetMod, [1:nsv, 1:nsk])
                @constraint(SetMod, [i=1:nsv, m=1:nsk], -θ[i,m] <= kernels_sk[m].q' * (u - data_sv[:,i]) / (kernels_sk[m].c * kernels_sk[m].κ))
                @constraint(SetMod, [i=1:nsv, m=1:nsk], kernels_sk[m].q' * (u - data_sv[:,i]) / (kernels_sk[m].c * kernels_sk[m].κ) <= θ[i,m])
                @constraint(SetMod, sum(α_sv[i] * π_sk[m] * θ[i,m] for i in 1:nsv, m in 1:nsk) <= 1 - ρ)
            elseif identity.(model.kernels_sk) isa Vector{DirectionalNullspaceProjectionNormKernel}
                SetMod = Model()
                u = @variable(SetMod, [1:n]; base_name=varname)
                SetMod[varname] = u
                θ = @variable(SetMod, [1:n-1, 1:nsv, 1:nsk])
                @constraint(SetMod, [i=1:nsv, m=1:nsk], -θ[:,i,m] <= kernels_sk[m].Q' * (u - data_sv[:,i]) / (kernels_sk[m].c * kernels_sk[m].κ))
                @constraint(SetMod, [i=1:nsv, m=1:nsk], kernels_sk[m].Q' * (u - data_sv[:,i]) / (kernels_sk[m].c * kernels_sk[m].κ) <= θ[:,i,m])
                @constraint(SetMod, sum(α_sv[i] * π_sk[m] * sum(θ[:,i,m]) for i in 1:nsv, m in 1:nsk) <= 1 - ρ)
            elseif false # Add code here when there are more PWL kernel types
            end
        else
            @error "It's not a proper model with supported piecewise linear kernel types."
        end
    elseif form == "conic"
        if identity.(model.kernels_sk) isa Vector{DirectionalProjectionDistanceKernel}
            SetMod = Model()
            u = @variable(SetMod, [1:n]; base_name=varname)
            SetMod[varname] = u
            @constraint(SetMod, [1 - ρ, [α_sv[i] * π_sk[m] * kernels_sk[m].q' * (u - data_sv[:,i]) / (kernels_sk[m].c * kernels_sk[m].κ) for i in 1:nsv, m in 1:nsk]...] in MOI.NormOneCone(nsv * nsk + 1))
        elseif identity.(model.kernels_sk) isa Vector{DirectionalNullspaceProjectionNormKernel}
            SetMod = Model()
            u = @variable(SetMod, [1:n]; base_name=varname)
            SetMod[varname] = u
            @constraint(SetMod, [1 - ρ, [α_sv[i] * π_sk[m] * kernels_sk[m].Q[:, k]' * (u - data_sv[:,i]) / (kernels_sk[m].c * kernels_sk[m].κ) for k in 1:n-1, i in 1:nsv, m in 1:nsk]...] in MOI.NormOneCone((n-1) * nsv * nsk + 1))
        elseif false # Add code here when there are more PWL kernel types
        else
            @error "It's not a proper model with supported conic characteristic."
        end
    elseif false # Add code here when there are more typical nonlinear kernel types
    end

    return SetMod
end

function convert_to_jumpmodel(
    model::AbstractVector{StandardModel};
    form::String="original",
    varname::Symbol=:u
)

    n = size(model[1].data, 1)

    if form == "original"
        SetMod = Model()
        u = @variable(SetMod, [1:n]; base_name=varname)
        SetMod[varname] = u
        for bat in eachindex(model)
            @constraint(SetMod, sum(model[bat].α_sv[i] * model[bat].π_sk[m] * model[bat].kernels_sk[m](u, model[bat].data_sv[:,i]) for i in eachindex(model[bat].SV), m in eachindex(model[bat].SK)) >= model[bat].ρ)
        end
    elseif form == "linear"
        Kernels_SK = []
        for bat in eachindex(model)
            push!(Kernels_SK, model[bat].kernels_sk...)
        end
        if identity.(Kernels_SK) isa Vector{<:PiecewiseLinearKernel}
            if identity.(Kernels_SK) isa Vector{DirectionalProjectionDistanceKernel}
                SetMod = Model()
                u = @variable(SetMod, [1:n]; base_name=varname)
                SetMod[varname] = u
                for bat in eachindex(model)
                    data_sv = model[bat].data_sv
                    kernels_sk = model[bat].kernels_sk
                    α_sv = model[bat].α_sv
                    π_sk = model[bat].π_sk
                    nsv = length(α_sv)
                    nsk = length(π_sk)
                    ρ = model[bat].ρ
                    θ = @variable(SetMod, [1:nsv, 1:nsk])
                    @constraint(SetMod, [i=1:nsv, m=1:nsk], -θ[i,m] <= kernels_sk[m].q' * (u - data_sv[:,i]) / (kernels_sk[m].c * kernels_sk[m].κ))
                    @constraint(SetMod, [i=1:nsv, m=1:nsk], kernels_sk[m].q' * (u - data_sv[:,i]) / (kernels_sk[m].c * kernels_sk[m].κ) <= θ[i,m])
                    @constraint(SetMod, sum(α_sv[i] * π_sk[m] * θ[i,m] for i in 1:nsv, m in 1:nsk) <= 1 - ρ)
                end
            elseif identity.(Kernels_SK) isa Vector{DirectionalNullspaceProjectionNormKernel}
                SetMod = Model()
                u = @variable(SetMod, [1:n]; base_name=varname)
                SetMod[varname] = u
                for bat in eachindex(model)
                    data_sv = model[bat].data_sv
                    kernels_sk = model[bat].kernels_sk
                    α_sv = model[bat].α_sv
                    π_sk = model[bat].π_sk
                    nsv = length(α_sv)
                    nsk = length(π_sk)
                    ρ = model[bat].ρ
                    θ = @variable(SetMod, [1:n-1, 1:nsv, 1:nsk])
                    @constraint(SetMod, [i=1:nsv, m=1:nsk], -θ[:,i,m] <= kernels_sk[m].Q' * (u - data_sv[:,i]) / (kernels_sk[m].c * kernels_sk[m].κ))
                    @constraint(SetMod, [i=1:nsv, m=1:nsk], kernels_sk[m].Q' * (u - data_sv[:,i]) / (kernels_sk[m].c * kernels_sk[m].κ) <= θ[:,i,m])
                    @constraint(SetMod, sum(α_sv[i] * π_sk[m] * sum(θ[:,i,m]) for i in 1:nsv, m in 1:nsk) <= 1 - ρ)
                end
            elseif false # Add code here when there are more PWL kernel types
            end
        else
            @error "It's not a proper model with supported piecewise linear kernel types."
        end
    elseif form == "conic"
        Kernels_SK = []
        for bat in eachindex(model)
            push!(Kernels_SK, model[bat].kernels_sk...)
        end
        if identity.(Kernels_SK) isa Vector{DirectionalProjectionDistanceKernel}
            SetMod = Model()
            u = @variable(SetMod, [1:n]; base_name=varname)
            SetMod[varname] = u
            for bat in eachindex(model)
                data_sv = model[bat].data_sv
                kernels_sk = model[bat].kernels_sk
                α_sv = model[bat].α_sv
                π_sk = model[bat].π_sk
                nsv = length(α_sv)
                nsk = length(π_sk)
                ρ = model[bat].ρ
                @constraint(SetMod, [1 - ρ, [α_sv[i] * π_sk[m] * kernels_sk[m].q' * (u - data_sv[:,i]) / (kernels_sk[m].c * kernels_sk[m].κ) for i in 1:nsv, m in 1:nsk]...] in MOI.NormOneCone(nsv * nsk + 1))
            end
        elseif identity.(Kernels_SK) isa Vector{DirectionalNullspaceProjectionNormKernel}
            SetMod = Model()
            u = @variable(SetMod, [1:n]; base_name=varname)
            SetMod[varname] = u
            for bat in eachindex(model)
                data_sv = model[bat].data_sv
                kernels_sk = model[bat].kernels_sk
                α_sv = model[bat].α_sv
                π_sk = model[bat].π_sk
                nsv = length(α_sv)
                nsk = length(π_sk)
                ρ = model[bat].ρ
                @constraint(SetMod, [1 - ρ, [α_sv[i] * π_sk[m] * kernels_sk[m].Q[:, k]' * (u - data_sv[:,i]) / (kernels_sk[m].c * kernels_sk[m].κ) for k in 1:n-1, i in 1:nsv, m in 1:nsk]...] in MOI.NormOneCone((n-1) * nsv * nsk + 1))
            end
        elseif false # Add code here when there are more PWL kernel types
        else
            @error "It's not a proper model with supported conic characteristic."
        end
    elseif false # Add code here when there are more typical nonlinear kernel types
    end

    return SetMod
end

function convert_to_jumpmodel(model::CombinedModel; kwargs...)
    model = mklocsvmunbuild(model)
    convert_to_jumpmodel(model; kwargs...)
end




function convert_to_polyhedron(
    model::StandardModel;
    eliminated::Bool=true
)

    if identity.(model.kernels_sk) isa Vector{<:PiecewiseLinearKernel}
        if identity.(model.kernels_sk) isa Vector{DirectionalProjectionDistanceKernel} || identity.(model.kernels_sk) isa Vector{DirectionalNullspaceProjectionNormKernel}
            jumpmodel = convert_to_jumpmodel(model; form="conic", varname=:u)
            PolySet = polyhedron(jumpmodel)
            if eliminated
                n = size(model.data, 1)
                PolySet = eliminate(PolySet, [n+1:fulldim(PolySet);])
            end
        elseif  false # Add code here when there are more PWL kernel types
        end
    else
        @error "It's not a proper model with supported piecewise linear kernel types."
    end

    return PolySet
end

function convert_to_polyhedron(
    model::Vector{StandardModel};
    eliminated::Bool=true
)
    if eliminated
        model = convert_to_polyhedron.(model; eliminated=true)
        reduce(intersect, model)
    else
        model = convert_to_polyhedron.(model; eliminated=false)
        reduce(intersect, model)
    end
end

function convert_to_polyhedron(model::CombinedModel; kwargs...)
    model = mklocsvmunbuild(model)
    convert_to_polyhedron(model; kwargs...)
end


############################################################################
#                                Utilities                                 #
############################################################################
function decision_function(model::StandardModel)
    N = size(model.data, 2)
    data_sv = model.data_sv
    kernels_sk = model.kernels_sk
    α_sv = model.α_sv
    π_sk = model.π_sk
    nsv = length(α_sv)
    nsk = length(π_sk)
    ρ = model.ρ
    y(u::AbstractVector{<:Real}) = sum(α_sv[i] * π_sk[m] * kernels_sk[m](u, data_sv[:,i]) for i in 1:nsv, m in 1:nsk) - ρ
    y(X::AbstractMatrix{<:Real}) = [y(X[:, i]) for i in 1:N]

    return y
end