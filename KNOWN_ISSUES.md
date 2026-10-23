# Known issues

### K1 · The `DataLoader` constructor for labelled images is type piracy

- **location:** `src/data_loader.jl:29`
- **evidence:** `Aqua.test_piracies(GMLDatasets)` fails with one result; `test/quality/aqua.jl` marks
  it `broken = true`. The method adds to `GeometricMachineLearning.DataLoader` for argument types
  that this package does not own.
- **kind:** defect
- **found:** issue #29, part M29 of the test-suite migration

## Upstream

### K2 · Aqua's `persistent_tasks` check cannot run, because `InternedStrings` has no `Project.toml`

- **location:** `test/quality/aqua.jl:13`
- **evidence:** `Aqua.has_persistent_tasks(Base.PkgId(GMLDatasets))` with Aqua 0.8.18 throws
  "Unable to locate Project.toml in …/InternedStrings/JTzem". The dependency comes in through
  `MLDatasets` and `Pickle`. `test/quality/aqua.jl` marks the check `broken = true`.
- **kind:** upstream
- **found:** issue #30, part M29 of the test-suite migration

## Scripts

### K4 · Six MNIST scripts outside the revision harness use the API that GeometricOptimizers 0.9 and GML `main` removed

- **location:** `scripts/geometric_optimizers/mnist.jl:404`, `mnist_cuda.jl:828`, `mnist_metal.jl:569`,
  `mnist_metal_short.jl:576`; `scripts/gml/transformer_mnist.jl:38`,
  `scripts/gml/transformer_fashion_mnist.jl:38`
- **evidence:** the four `geometric_optimizers` scripts call `Adam(T)` and
  `MomentumMethod(momentum_coefficient)`; GeometricOptimizers 0.9 defines only `Adam(; β₁, β₂, δ)`
  and `MomentumMethod(; α)`. The two `gml` scripts call `AdamOptimizer(T)`, `GradientOptimizer()`,
  `MomentumOptimizer(T(0.5))` and `Optimizer(opt, nn; step_size)`, which GML `main` removed in
  favour of `TrainingOptimizer` and `train!`. The scripts environment resolves both, so each script
  stops at its first such call. The harness scripts are ported.
- **kind:** defect
- **found:** 2026-10-23, in the port of the revision harness to GeometricOptimizers 0.9

## Documentation

### K3 · The CHANGELOG states a narrower `GeometricMachineLearning` bound than `Project.toml` has

- **location:** `CHANGELOG.md:57`
- **evidence:** the `[Unreleased]` bullet "`[compat]` widens …" says
  `GeometricMachineLearning = "0.6, 0.7"`; `Project.toml:18` has `GeometricMachineLearning = "0.6, 0.7, 0.8"`.
- **kind:** found late
- **found:** #19 (2026-09-27, commit d92e3d5), which widened the bound in `Project.toml` and left
  the CHANGELOG line from #18 unchanged
