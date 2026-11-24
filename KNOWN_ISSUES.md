# Known issues

### K1 · The `DataLoader` constructor for labelled images is type piracy

- **location:** `src/data_loader.jl:28`
- **evidence:** `Aqua.test_piracies(GMLDatasets)` fails with one result; `test/quality/aqua.jl` marks
  it `broken = true`. The method adds to `DataLoader`, which `ReducedComplexityModeling` owns and
  `GeometricMachineLearning` re-exports, for argument types that this package does not own.
- **kind:** defect
- **found:** issue #29, part M29 of the test-suite migration

## Scripts

### K4 · Four MNIST scripts outside the revision harness use an API that GeometricOptimizers 0.9 and 0.10 do not define

- **location:** `scripts/geometric_optimizers/mnist.jl:401`, `mnist_cuda.jl:828`, `mnist_metal.jl:569`,
  `mnist_metal_short.jl:576`
- **evidence:** the four scripts call `Adam(T)` and `MomentumMethod(momentum_coefficient)`;
  GeometricOptimizers 0.9 and 0.10 define only `Adam(; β₁, β₂, δ)` and `MomentumMethod(; α)`. The
  scripts environment takes GeometricOptimizers 0.10, so each script stops at its first such call. The
  harness scripts are ported.
- **kind:** defect
- **found:** 2026-10-23, in the port of the revision harness to GeometricOptimizers 0.9

### K5 · On a CUDA host the environment preflight stops, and the pendulum stage cannot run, because the gradient of a device-resident point arrives on the host

- **location:** `scripts/revision/check_environment.jl:62`
- **evidence:** the preflight calls `rgrad(Y, randn(Float32, 4, 2))` at a `StiefelManifold` on the
  device. GeometricOptimizers 0.9 and 0.10 check the backends there (`_check_gradient_backend`) and raises
  `ArgumentError: mixed backends`, where 0.8 moved the gradient across with a temporary
  `_match_backend`. The pendulum trainer passes Zygote's gradient through GML's `train!` to the
  optimizer step, and that gradient is on the host for device parameters (GML #258, AbstractNeuralNetworks
  #39, both open). Shown with a fake GPU backend on GeometricOptimizers 0.9.0; not run on a CUDA
  device. The image stages keep their parameters on the host and are unaffected.
- **kind:** upstream
- **found:** 2026-10-23, in the port of the revision harness to GeometricOptimizers 0.9

### K6 · The scripts environment does not resolve: `Pickle` admits no `StridedViews` release that `CUDA` 6 admits

- **location:** `scripts/Project.toml`
- **evidence:** `Pkg.resolve()` in `scripts/` stops with "Unsatisfiable requirements detected for
  package Pickle". GeometricMachineLearning 0.10 and GeometricOptimizers 0.10 require
  KernelAbstractions 0.10.5, which admits CUDA 6.5 only, and the `CUDA` that the GPU scripts load is
  a dependency of the environment. `MLDatasets` 0.7, a dependency of this package, requires
  `Pickle` 0.3, and `Pickle` 0.3.7, its newest release, requires `StridedViews` 0.2 to 0.4. Those
  `StridedViews` releases bound their `CUDA` extension to CUDA 4 and 5; only `StridedViews` 0.5
  admits CUDA 6. A copy of `scripts/Project.toml` without `CUDA` resolves, and
  `test_step_timing.jl` and `test_leaf_composite.jl` pass in it. The fix is a `Pickle` release that
  admits `StridedViews` 0.5 (chengchingwen/Pickle.jl#50, open).
- **kind:** upstream
- **found:** 2026-11-24, in the move to GeometricMachineLearning 0.10
