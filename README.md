# MKLOCSVM.jl
`MKLOCSVM.jl` is a Julia package for multiple kernel learning (MKL) based one-class support vector machine (OCSVM). 


## Usage

### Installation
``` julia
using Pkg
Pkg.add("MKLOCSVM")
Pkg.add("GLMakie") # if the user wants visualization
```

### Using pacakges
``` julia
using MKLOCSVM
using GLMakie
```

### Data loading
Note that each column of the training data corresponds to an observation of the input features.
``` julia
u1 = 0.5 .+ 4 * rand(300)
u2 = 2 ./ u1 + 0.3 * randn(300)
X = [u1'; u2']
mklocsvmplot(X; backend=GLMakie)
```

<p align=center>
  <img src="/assets/data.png" width=70%>
</p>

### Creating candidate basis kernels
This package reexports [`KernelFunctions.jl`](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl.git), which allows the user to generate various basis kernels conveniently:

``` julia
Kernels = [
    CosineKernel(),
    ExponentialKernel(),
    GammaExponentialKernel(), 
    gaborkernel(),
    MaternKernel(),
    Matern32Kernel(),
    PiecewisePolynomialKernel(; dim=2, degree=2),
    RationalKernel(),
    RationalQuadraticKernel(),
    GammaRationalKernel(),
    GaussianKernel(),
    RBFKernel(),
    gaborkernel()
]
```

### Multiple kernel learning
The user can chose different algorithms to train the MKL model, e.g., `QCQP()` to solve the dual problem directly (which is a quadratically constrained quadratic program) or `HessianMKL()` to alternately optimize the kernel coefficients according to the second order information and solve a standard single kernel OCSVM. Here we choose the latter algorithm:
``` julia
# algor = QCQP(verbose=false)
algor = HessianMKL(verbose=false)
model = mklocsvmtrain(Kernels, X; algorithm=algor, ν=0.01, μ=0.5);
```

Please see the [paper](https://doi.org/10.1016/j.ejor.2020.11.027) for more information about the algorithms and the statistical meaning of the hyper-parameters.

### Visualization of the model

``` julia
mklocsvmplot(model; backend=GLMakie)
```

<p align=center>
  <img src="/assets/result.png" width=70%>
</p>

More information about the trained model can be retrieved by querying corresponding fields of `model`, e.g.,
``` julia
model.SV # the indeces of all support vectors
model.SK # the indeces of all support kernels
```

### Prediction

``` julia
u1 = 0.5 .+ 4 * rand(10)
u2 = 2 ./ u1 + 0.3 * randn(10)
X_test = [u1'; u2']
mklocsvmpredict(model, X_test)
```

The decision function, if needed, can be obtained by
``` julia
y = decision_function(model)
y(X_test)
```

### Other utilities

#### To train the model distributedly
When there are lots of candidate basis kernels, sometimes it may be a beter practice to group the kernels into batches first, then train them distributedly and finally take the intersection of all trained models as the resulting decision set.

``` julia
using Distributed
addprocs(5)
@everywhere using MKLOCSVM
using GLMakie

num_batch = 3
Kernels_inbat = group_kernels(Kernels, num_batch; mode="randomly")
model = pmap(
    ks -> mklocsvmtrain(ks, X; algorithm=algor, ν=0.01/num_batch, μ=0.5/num_batch), 
    Kernels_inbat
)
mklocsvmplot(model; backend=GLMakie)
```

<p align=center>
  <img src="/assets/result_dis.png" width=70%>
</p>


#### To construct a convex polyhedral set
By utilizting the Directional Projection Distance Kernel (DPDK) presented in the [paper](https://doi.org/10.1016/j.ejor.2020.11.027) or a new Directional Nullspace Projection Norm Kernel (DNPNK), the user will be able to construct a convex polytopic set. The corresponding functionalities have been integrated into the `mklocsvmtrain` function. 

``` julia
model = mklocsvmtrain(X, 50; kernel="DPDK", algorithm=algor, q_mode="randomly", ν=0.01, μ=0.05)
mklocsvmplot(model; backend=GLMakie)
```
<p align=center>
  <img src="/assets/DPDK.png" width=70%>
</p>

This is also allowed to train distributedly for acceleration under some situations:
``` julia
model = mklocsvmtrain(X, 50; kernel="DPDK", algorithm=algor, q_mode="randomly", ν=0.01, μ=0.05, num_batch=5)
mklocsvmplot(model; backend=GLMakie)
```
<p align=center>
  <img src="/assets/DPDK_dis.png" width=70%>
</p>

The model can be converted into other types for further usage:
``` julia
convert_to_jumpmodel(model; form="linear", varname=:u)
convert_to_polyhedron(model; eliminated=true)
```

## Citing
If you use `MKLOCSVM.jl`, we ask that you please cite this [repository](https://github.com/hanb16/MKLOCSVM.jl.git) and the following [paper](https://doi.org/10.1016/j.ejor.2020.11.027):
``` bibtex
@article{han2021multiple,
  title={Multiple kernel learning-aided robust optimization: Learning algorithm, computational tractability, and usage in multi-stage decision-making},
  author={Han, Biao and Shang, Chao and Huang, Dexian},
  journal={European Journal of Operational Research},
  volume={292},
  number={3},
  pages={1004--1018},
  year={2021},
  publisher={Elsevier}
}
```

## Acknowledgments
By default, this package implicitly uses [`KernelFunctions.jl`](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl.git), open source solvers [`HiGHS.jl`](https://github.com/jump-dev/HiGHS.jl.git) and [`Ipopt.jl`](https://github.com/jump-dev/Ipopt.jl.git), and the single kernel SVM solver [`LIBSVM.jl`](https://github.com/JuliaML/LIBSVM.jl.git). Thanks for these useful packages, although the user is also allowed to replace them with other alternatives.