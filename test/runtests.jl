using MKLOCSVM
using MAT: matread
using GLMakie: GLMakie


## Load Data
X = matread("./data/data.mat")["X"]'
mklocsvmplot(X; backend=GLMakie)


## Create Candidate Basis Kernels
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

## Multiple Kernel Learning
# algor = QCQP(verbose=false)
algor = HessianMKL(verbose=false)

@time model = mklocsvmtrain(Kernels, X; algorithm=algor, ν=0.01, μ=0.5);


## Visualization
mklocsvmplot(model; backend=GLMakie)
sleep(3)
















