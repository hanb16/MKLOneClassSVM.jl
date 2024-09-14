module MKLOneClassSVM
# Multiple Kernel Learning based One-Class Support Vector Machine

using Reexport
using Statistics: mean, quantile, std
using LIBSVM: Kernel as LIBSVM_Kernel, LIBSVM, OneClassSVM, svmtrain
using LinearAlgebra: I, diag, isposdef, norm, nullspace
using JuMP: @constraint, @objective, @variable, MOI, Model, all_variables, delete, dual, name, optimize!, set_optimizer_attributes, set_silent, unregister, value
using HiGHS
using Ipopt
@reexport using KernelFunctions
using Polyhedra: eliminate, fulldim, polyhedron
using Random: shuffle!
using Distributed: @spawnat, fetch, pmap
using Distances: Minkowski, pairwise


include("types.jl")
export MKLOCSVM_Kernel, PiecewiseLinearKernel, DirectionalProjectionDistanceKernel, DirectionalNullspaceProjectionNormKernel
export MKLOCSVM_Model, StandardModel, CombinedModel
export MKLOCSVM_Algorithm, QCQP, HessianMKL


include("kernels.jl")
export create_basis_DPDKs, create_basis_DNPNKs
export group_kernels, multiplekernelmatrices


include("algorithms.jl")
export mklocsvmtrain


include("models.jl")
export mklocsvmpredict, mklocsvmbuild, mklocsvmunbuild, convert_to_jumpmodel, convert_to_polyhedron, decision_function



include("visualization.jl")
export mklocsvmplot




end # module MKLOCSVMPWLS
