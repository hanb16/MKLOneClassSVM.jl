############################################################################
#                                 Kernels                                  #
############################################################################
abstract type MKLOCSVM_Kernel <: Kernel end
abstract type PiecewiseLinearKernel <: MKLOCSVM_Kernel end

"""
# Definition
The basis candidate kernel used for multiple kernel learning presented by `Ref.1`:

``` math
K(u, v; q, c, κ) = 1 - |q^\\top (u - v) / (c \\cdot κ)|
```

# References
1. Han, B., Shang, C., & Huang, D. (2020). Multiple kernel learning-aided robust optimization: learning algorithm, computational tractability, and usage in multi-stage decision-making. European Journal of Operational Research. https://doi.org/10.1016/j.ejor.2020.11.027.
"""
struct DirectionalProjectionDistanceKernel <: PiecewiseLinearKernel
    q::AbstractVector{<:Real}
    c::Real
    κ::Real

    function DirectionalProjectionDistanceKernel(q::AbstractVector{<:Real}, c::Real, κ::Real)
        KernelFunctions.@check_args(DirectionalProjectionDistanceKernel, c, c > zero(c), "c > 0")
        KernelFunctions.@check_args(DirectionalProjectionDistanceKernel, κ, κ > zero(κ), "κ > 0")
        return new(q, c, κ)
    end
end


struct DirectionalNullspaceProjectionNormKernel <: PiecewiseLinearKernel
    Q::AbstractMatrix{<:Real}
    c::Real
    κ::Real

    function DirectionalNullspaceProjectionNormKernel(Q::AbstractMatrix{<:Real}, c::Real, κ::Real)
        KernelFunctions.@check_args(DirectionalNullspaceProjectionNormKernel, Q, Q'Q ≈ I, "Q'Q == I")
        KernelFunctions.@check_args(DirectionalNullspaceProjectionNormKernel, c, c > zero(c), "c > 0")
        KernelFunctions.@check_args(DirectionalNullspaceProjectionNormKernel, κ, κ > zero(κ), "κ > 0")
        return new(Q, c, κ)
    end
end





############################################################################
#                                 Models                                   #
############################################################################
abstract type MKLOCSVM_Model end

struct StandardModel <: MKLOCSVM_Model
    data::AbstractMatrix{Float64}
    kernels::AbstractVector{<:Kernel}
    ν::Float64 
    μ::Float64
    α::AbstractVector{Float64}
    π::AbstractVector{Float64}
    ρ::Float64
    ϵ::Float64
    data_sv::AbstractMatrix{Float64}
    data_bsv::AbstractMatrix{Float64}
    data_osv::AbstractMatrix{Float64}
    data_iv::AbstractMatrix{Float64}
    kernels_sk::AbstractVector{<:Kernel}
    kernels_nsk::AbstractVector{<:Kernel}
    α_sv::AbstractVector{Float64}
    π_sk::AbstractVector{Float64}
    SV::AbstractVector{Int64}
    BSV::AbstractVector{Int64}
    OSV::AbstractVector{Int64}
    IV::AbstractVector{Int64}
    SK::AbstractVector{Int64}
    NSK::AbstractVector{Int64}
end
struct CombinedModel <: MKLOCSVM_Model
    data::AbstractMatrix{Float64}
    kernels::AbstractVector{AbstractVector{<:Kernel}}
    ν::AbstractVector{Float64}
    μ::AbstractVector{Float64}
    α::AbstractVector{AbstractVector{Float64}}
    π::AbstractVector{AbstractVector{Float64}}
    ρ::AbstractVector{Float64}
    ϵ::AbstractVector{Float64}
    data_sv::AbstractVector{AbstractMatrix{Float64}}
    data_bsv::AbstractVector{AbstractMatrix{Float64}}
    data_osv::AbstractVector{AbstractMatrix{Float64}}
    data_iv::AbstractVector{AbstractMatrix{Float64}}
    kernels_sk::AbstractVector{AbstractVector{<:Kernel}}
    kernels_nsk::AbstractVector{AbstractVector{<:Kernel}}
    α_sv::AbstractVector{AbstractVector{Float64}}
    π_sk::AbstractVector{AbstractVector{Float64}}
    SV::AbstractVector{AbstractVector{Int64}}
    BSV::AbstractVector{AbstractVector{Int64}}
    OSV::AbstractVector{AbstractVector{Int64}}
    IV::AbstractVector{AbstractVector{Int64}}
    SK::AbstractVector{Vector{Int64}}
    NSK::AbstractVector{Vector{Int64}}
    comb_data_sv::AbstractMatrix{Float64}
    comb_data_bsv::AbstractMatrix{Float64}
    comb_data_osv::AbstractMatrix{Float64}
    comb_data_iv::AbstractMatrix{Float64}
    comb_SV::AbstractVector{Int64}
    comb_BSV::AbstractVector{Int64}
    comb_OSV::AbstractVector{Int64}
    comb_IV::AbstractVector{Int64}
end





############################################################################
#                               Algorithms                                 #
############################################################################
abstract type MKLOCSVM_Algorithm end

@kwdef struct QCQP <: MKLOCSVM_Algorithm
    solver::Module=Ipopt
    tol::Float64=1e-8
    max_iter::Int64=1e5
    verbose::Bool=true
end




@kwdef struct HessianMKL <: MKLOCSVM_Algorithm
    svm_solver::Module=LIBSVM
    step_solver::Module=Ipopt 
    ϵ::Float64=1e-6
    verbose::Bool=true
end