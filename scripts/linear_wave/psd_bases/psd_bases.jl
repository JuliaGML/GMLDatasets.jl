# Symplectic bases of the linear-wave training data beyond the cotangent lift: does optimizing over
# the whole symplectic Stiefel manifold Sp(2n, 2N) improve on the PSD of `../errors.jl`?
#
#     julia --project=scripts/linear_wave/psd_bases -e 'using Pkg; Pkg.instantiate()'
#     julia --project=scripts/linear_wave/psd_bases scripts/linear_wave/psd_bases/psd_bases.jl
#
# This script has its own environment: it optimizes on `SymplecticStiefelManifold` of
# GeometricOptimizers, which needs GeometricOptimizers 0.10, while `scripts/` holds
# GeometricMachineLearning 0.9 and with it GeometricOptimizers 0.9. It therefore loads neither
# GMLDatasets nor GeometricMachineLearning, and builds the data of `linear_wave()` from
# GeometricProblems directly, with the settings of `../common.jl`.
#
# For every reduced dimension 2n, three bases A ∈ Sp(2n, 2N) of the snapshot matrix M, each with
# reduction 𝒫 = A⁺ = 𝕁₂ₙᵀAᵀ𝕁₂ₙ and reconstruction ℛ = A:
#
#   * `cotangent_lift`: A = diag(Φ, Φ) with Φ the first n left singular vectors of [Q P], the PSD of
#     `fit_psd` in `../common.jl` and of the paper;
#   * `complex_svd`: A = [Φ -Ψ; Ψ Φ] with Φ + iΨ the first n left singular vectors of Q + iP, the
#     best symplectic basis with orthonormal columns;
#   * `symplectic_stiefel_bfgs` and `symplectic_stiefel_adam`: minimizers of the relative training
#     error f(A) = ‖M - AA⁺M‖²_F / ‖M‖²_F over Sp(2n, 2N) on `SymplecticStiefelManifold`, started
#     at the cotangent lift, found by BFGS on all snapshots and by Adam on minibatches with the
#     settings of the SAE training; `LW_ALGORITHMS` (default `bfgs,adam`) selects them.
#
# At the test wave speeds each basis gets the projection and the reduction error of `../errors.jl`.
# The FOM ẋ = Kx is linear, so the reduced model is ż = A⁺KAz, integrated from A⁺x⁽⁰⁾ by the
# implicit midpoint rule on the time grid of the FOM, which for a linear system is the Cayley map
# (I - hB/2)⁻¹(I + hB/2) of B = A⁺KA.
#
# Two checks, each of which fails loudly:
#   * the analytic gradient of f against a central difference;
#   * with LW_REFERENCE_ERRORS set to an `errors.csv` of `../errors.jl` (the paper keeps the one of
#     its figure), the cotangent-lift errors against its PSD rows, which `HRedSys` of
#     GeometricMachineLearning computed independently of this script.
#
# Writes `$(LW_OUTPUT_DIR)/psd_bases.csv` and `psd_bases.h5`, one row per (method, n, μ), with the
# training error √f, the symplecticity defect ‖Aᵀ𝕁A - 𝕁‖₂ and the condition number of A.

using GeometricIntegrators: ImplicitMidpoint, integrate
using GeometricOptimizers
using LinearAlgebra
using Printf
using Random

import GeometricProblems.LinearWave as LinearWave
import HDF5
import Pkg

const SMOKE = parse(Bool, get(ENV, "LW_SMOKE", "0"))
_setting(name, full, smoke) = get(ENV, name, SMOKE ? smoke : full)
function _ints(s)
    occursin(':', s) ? collect(range(parse.(Int, split(s, ':'))...)) :
    parse.(Int, split(s, ','))
end

# The settings of `../common.jl`.
const N = parse(Int, _setting("LW_N", "128", "16"))
const N_PARAMS = parse(Int, _setting("LW_N_PARAMS", "20", "3"))
const N_TIME_STEPS = parse(Int, _setting("LW_N_TIME_STEPS", "200", "11"))
const TIMESPAN = (0.0, 1.0)
const TIMESTEP = (TIMESPAN[2] - TIMESPAN[1]) / (N_TIME_STEPS - 1)
const TRAINING_MU = collect(range(5 / 12, 2 / 3; length = N_PARAMS))
const TEST_MU = parse.(Float64,
    split(_setting("LW_TEST_MU", "0.47,0.51,0.55,0.625", "0.47,0.625"), ','))
const HALF_REDUCED_DIMS = _ints(_setting("LW_HALF_REDUCED_DIMS", "2:15", "2:3"))
const ALGORITHMS = split(_setting("LW_ALGORITHMS", "bfgs,adam", "bfgs,adam"), ',')
const BFGS_ITERATIONS = parse(Int, _setting("LW_BFGS_ITERATIONS", "2000", "20"))
# Adam with the settings of the SAE training of the linear-wave experiment.
const ADAM_EPOCHS = parse(Int, _setting("LW_ADAM_EPOCHS", "1000", "2"))
const ADAM_BATCH_SIZE = parse(Int, _setting("LW_ADAM_BATCH_SIZE", "512", "8"))
const ADAM_η, ADAM_β₁, ADAM_β₂, ADAM_δ = 1e-3, 0.9, 0.99, 1e-8
const REFERENCE_ERRORS = get(ENV, "LW_REFERENCE_ERRORS", "")
const OUTPUT_DIR = get(ENV, "LW_OUTPUT_DIR",
    joinpath(dirname(@__DIR__), SMOKE ? "output_smoke" : "output"))

function full_problem(μ)
    ic = LinearWave.initial_condition(μ, N)
    LinearWave.hodeproblem(N, ic.q, ic.p; parameters = (μ = μ,), timespan = TIMESPAN,
        timestep = TIMESTEP)
end

# The solution is indexed from 0, so the snapshots are concatenated over its own indices.
snapshots(sol) = mapreduce(k -> vcat(sol.q[k], sol.p[k]), hcat, eachindex(sol.t))

"The FOM matrix K of ẋ = Kx, column by column from the vector field of `LinearWave`."
function fom_matrix(μ)
    K = zeros(2N, 2N)
    v, f = zeros(N), zeros(N)
    for j in 1:(2N)
        x = zeros(2N)
        x[j] = 1
        LinearWave.linear_wave_v(v, 0.0, x[1:N], x[(N + 1):end], (μ = μ,))
        LinearWave.linear_wave_f(f, 0.0, x[1:N], x[(N + 1):end], (μ = μ,))
        K[:, j] = vcat(v, f)
    end
    K
end

poisson(m) = [zeros(m, m) I(m); -I(m) zeros(m, m)]
symplectic_inverse(A) = poisson(size(A, 2) ÷ 2)' * A' * poisson(size(A, 1) ÷ 2)
symplecticity_defect(A) = opnorm(A' * poisson(size(A, 1) ÷ 2) * A - poisson(size(A, 2) ÷ 2))

# The training data: the snapshots of every training wave speed, side by side.
const M = reduce(hcat, [snapshots(integrate(full_problem(μ), ImplicitMidpoint()))
                        for μ in TRAINING_MU])
const C = M * M'
const TRACE_C = tr(C)

"The relative training error f(A) = ‖M - AA⁺M‖²_F / ‖M‖²_F, through C = MMᵀ."
function training_error(A::AbstractMatrix)
    R = I - A * symplectic_inverse(A)
    tr(R * C * R') / TRACE_C
end

"""
The Euclidean gradient of f: -2(G𝕁ᵀA𝕁ₙ + 𝕁GᵀA𝕁ₙᵀ)/tr C with G = (I - AA⁺)C. With the C of a
minibatch it is the gradient of the training error on that minibatch.
"""
function training_error_gradient!(g::AbstractMatrix, A::AbstractMatrix, C = C)
    JN, Jn = poisson(N), poisson(size(A, 2) ÷ 2)
    G = (I - A * symplectic_inverse(A)) * C
    g .= -2 .* (G * JN' * A * Jn .+ JN * G' * A * Jn') ./ tr(C)
    g
end

# GeometricOptimizers hands the gradient the flattened point.
training_error(x::AbstractVector) = training_error(reshape(x, 2N, :))
function training_error_gradient!(g::AbstractVector, x::AbstractVector)
    training_error_gradient!(reshape(g, 2N, :), reshape(x, 2N, :))
    g
end

"Check `training_error_gradient!` against a central difference along a random direction."
function check_gradient(A; seed = 1)
    V = randn(Xoshiro(seed), size(A))
    g = training_error_gradient!(similar(A), A)
    h = cbrt(eps()) * norm(A) / norm(V)
    difference = (training_error(A + h * V) - training_error(A - h * V)) / (2h)
    abs(dot(g, V) - difference) <= 1e-6 * abs(difference) ||
        error("gradient check failed: ⟨∇f, V⟩ = $(dot(g, V)), central difference $difference")
    nothing
end

function cotangent_lift(n)
    Φ = svd(hcat(M[1:N, :], M[(N + 1):end, :])).U[:, 1:n]
    [Φ zeros(N, n); zeros(N, n) Φ]
end

function complex_svd(n)
    U = svd(M[1:N, :] .+ im .* M[(N + 1):end, :]).U[:, 1:n]
    Φ, Ψ = real(U), imag(U)
    [Φ -Ψ; Ψ Φ]
end

"BFGS on the training error over all snapshots, started at the cotangent lift."
function symplectic_stiefel_bfgs(n)
    Y = SymplecticStiefelManifold(cotangent_lift(n))
    opt = Optimizer(
        Y, training_error; (∇F!) = training_error_gradient!, algorithm = BFGS(),
        max_iterations = BFGS_ITERATIONS, warn_iterations = 2 * BFGS_ITERATIONS)
    result = solve!(Y, OptimizerState(BFGS(), Y), opt)
    Matrix(Y.A), result.status
end

"""
Adam on minibatches of snapshot columns, reshuffled every epoch, started at the cotangent lift: one
step per minibatch, with the gradient of the training error on that minibatch.
"""
function symplectic_stiefel_adam(n)
    Y = SymplecticStiefelManifold(cotangent_lift(n))
    opt = TrainingOptimizer(Y; algorithm = Adam(; β₁ = ADAM_β₁, β₂ = ADAM_β₂, δ = ADAM_δ),
        linesearch = ADAM_η)
    rng, g = Xoshiro(n), similar(Y.A)
    for _ in 1:ADAM_EPOCHS,
        batch in Iterators.partition(randperm(rng, size(M, 2)),
            ADAM_BATCH_SIZE)

        M_batch = M[:, batch]
        optimization_step!(Y, opt, training_error_gradient!(g, Y.A, M_batch * M_batch'))
    end
    Matrix(Y.A), "$(ADAM_EPOCHS) epochs of $(cld(size(M, 2), ADAM_BATCH_SIZE)) steps"
end

const OPTIMIZERS = Dict("bfgs" => symplectic_stiefel_bfgs, "adam" =>
    symplectic_stiefel_adam)

const FULL = Dict(μ => snapshots(integrate(full_problem(μ), ImplicitMidpoint()))
for μ in TEST_MU)
const K_FOM = Dict(μ => fom_matrix(μ) for μ in TEST_MU)

# Two relative errors of an approximation Y of the snapshots X:
#   * `euclidean`, √(Σₜ‖x⁽ᵗ⁾ - y⁽ᵗ⁾‖² / Σₜ‖x⁽ᵗ⁾‖²), the formula of the paper's projection and
#     reduction errors;
#   * `qp`, (‖Q_X - Q_Y‖_F + ‖P_X - P_Y‖_F) / (‖Q_X‖_F + ‖P_X‖_F), which is what
#     `projection_error` and `reduction_error` of GeometricMachineLearning 0.9 compute (its
#     `_norm` of a (q, p) pair is (‖q‖ + ‖p‖)/√2), and so what `../errors.jl` writes.
euclidean(X, Y) = sqrt(sum(abs2, X - Y) / sum(abs2, X))
function qp(X, Y)
    (norm(X[1:N, :] - Y[1:N, :]) + norm(X[(N + 1):end, :] - Y[(N + 1):end, :])) /
    (norm(X[1:N, :]) + norm(X[(N + 1):end, :]))
end

"The reconstructed snapshots of the projection and of the reduced model, for one test μ."
function reconstructions(A, μ)
    X = FULL[μ]
    A⁺ = symplectic_inverse(A)
    B = A⁺ * K_FOM[μ] * A
    step = (I - TIMESTEP / 2 * B) \ (I + TIMESTEP / 2 * B)
    Z = zeros(size(A, 2), size(X, 2))
    Z[:, 1] = A⁺ * X[:, 1]
    for k in 2:size(X, 2)
        Z[:, k] = step * Z[:, k - 1]
    end
    A * (A⁺ * X), A * Z
end

"The PSD rows of an `errors.csv` of `../errors.jl`, as (n, μ) => (e_proj, e_red)."
function reference_errors(path)
    lines = readlines(path)
    header = split(first(lines), ',')
    col(name) = findfirst(==(name), header)
    rows = [split(l, ',') for l in lines[2:end]]
    Dict((parse(Int, r[col("n")]), parse(Float64, r[col("mu")])) =>
             (parse(Float64, r[col("projection_error")]),
                 parse(Float64, r[col("reduction_error")]))
    for r in rows if r[col("method")] == "PSD")
end

const REFERENCE = isempty(REFERENCE_ERRORS) ? nothing : reference_errors(REFERENCE_ERRORS)

rows = []
for n in HALF_REDUCED_DIMS
    # The formula holds for any A, so it is checked away from the manifold too.
    check_gradient(cotangent_lift(n) + 0.1 * randn(Xoshiro(n), 2N, 2n))
    bases = Any["cotangent_lift" => cotangent_lift(n), "complex_svd" => complex_svd(n)]
    seconds = Dict{String, Float64}()
    for algorithm in ALGORITHMS
        seconds[algorithm] = @elapsed A_opt, status = OPTIMIZERS[algorithm](n)
        @printf("n = %d: %s in %.1f s\n", n, algorithm, seconds[algorithm])
        println(status)
        flush(stdout)
        push!(bases, "symplectic_stiefel_" * algorithm => A_opt)
    end
    for (method, A) in bases, μ in TEST_MU

        X = FULL[μ]
        Y_proj, Y_red = reconstructions(A, μ)
        algorithm = replace(method, "symplectic_stiefel_" => "")
        r = (method = method, n = n, mu = μ,
            seconds = get(seconds, algorithm, 0.0), training_error = sqrt(training_error(A)),
            projection_error = euclidean(X, Y_proj), reduction_error = euclidean(X, Y_red),
            projection_error_qp = qp(X, Y_proj), reduction_error_qp = qp(X, Y_red),
            symplecticity_defect = symplecticity_defect(A), condition_number = cond(A))
        if method == "cotangent_lift" && !isnothing(REFERENCE)
            r_proj, r_red = REFERENCE[(n, μ)]
            isapprox(r.projection_error_qp, r_proj; rtol = 1e-6) &&
            isapprox(r.reduction_error_qp, r_red; rtol = 1e-6) ||
                error("reference check failed at n = $n, μ = $μ: " *
                      "($(r.projection_error_qp), $(r.reduction_error_qp)) against " *
                      "($r_proj, $r_red)")
        end
        push!(rows, r)
        @printf("%-23s n = %2d μ = %.3f: train %.3e, e_proj %.3e (qp %.3e), e_red %.3e (qp %.3e), defect %.1e, κ %.1f\n",
            method, n, μ, r.training_error, r.projection_error, r.projection_error_qp,
            r.reduction_error, r.reduction_error_qp, r.symplecticity_defect,
            r.condition_number)
        flush(stdout)
    end
end
isnothing(REFERENCE) ||
    println("the cotangent lift reproduces the PSD rows of ", REFERENCE_ERRORS)

mkpath(OUTPUT_DIR)
columns = keys(first(rows))
path(ext) = joinpath(OUTPUT_DIR, "psd_bases." * ext)
open(path("csv"), "w") do io
    println(io, join(columns, ','))
    for r in rows
        println(io, join((v isa AbstractFloat ? repr(v) : string(v) for v in values(r)), ','))
    end
end
HDF5.h5open(path("h5"), "w") do file
    for c in columns
        file[string(c)] = [getproperty(r, c) for r in rows]
    end
    attributes = HDF5.attributes(file)
    attributes["N"], attributes["n_params"], attributes["n_time_steps"] = N, N_PARAMS,
    N_TIME_STEPS
    attributes["test_mu"], attributes["bfgs_iterations"] = TEST_MU, BFGS_ITERATIONS
    attributes["adam_epochs"], attributes["adam_batch_size"] = ADAM_EPOCHS, ADAM_BATCH_SIZE
    attributes["adam_eta_beta1_beta2_delta"] = [ADAM_η, ADAM_β₁, ADAM_β₂, ADAM_δ]
    attributes["julia"] = string(VERSION)
    for (_, info) in Pkg.dependencies()
        info.is_direct_dep && (attributes["package_" * info.name] = string(info.version))
    end
end
println("wrote ", path("csv"), " and ", path("h5"), " ($(length(rows)) rows)")
