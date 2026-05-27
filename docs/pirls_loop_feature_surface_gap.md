# pirls_loop_on_stream → fit_model_for_fixed_rho feature-surface gap

Task #47, charter `pirls_loop_full_feature_surface`. Owner: pirls-loop-elevate.

## Why this audit exists

`sigma_cubature_dispatch` (`src/solver/reml/eval.rs:167`) gates its GPU
stream-pool path on `device_pirls_stage3_ready()`
(`src/solver/reml/eval.rs:138`), which is currently hard-coded `false`.
The body of `sigma_cubature_dispatch` documents the swap site: when the
predicate flips to `true`, every cubature batch fans out across CUDA
streams and each sigma point runs the **device-resident inner PIRLS** for
its perturbed posterior, returning only the `p×p` Hessian and `p`-vector
β back to the host.

The device-resident inner PIRLS that the swap site is waiting on is
`pirls_loop_on_stream` (`src/solver/gpu/pirls_gpu.rs:1359`,
implementation in the same file at line `913` inside `mod cuda` as
`pirls_loop`). Stage 3.3 (pirls-row-v3) landed the loop driver but
deliberately stopped at a thin contract: a single penalty Hessian, a
single `lm_ridge`, untransformed coefficients, and no constraints. The
CPU oracle `fit_model_for_fixed_rho_with_adaptive_kkt`
(`src/solver/pirls.rs:7056`) carries a much richer feature surface that
sigma-cubature implicitly relies on, because every sigma fit it consumes
must live in the same coordinate system and obey the same constraints as
the outer REML caller.

Flipping the predicate today would compute the wrong thing for any model
that uses canonical-penalty reparameterization, Kronecker tensor smooths,
coefficient lower bounds, monotonicity / linear-inequality constraints,
or the Gaussian-Identity fixed-data short-circuit. That is the entire
feature surface this audit catalogs.

## Reference points

- CPU oracle entry: `fit_model_for_fixed_rho`
  (`src/solver/pirls.rs:7046`), wrapping the canonical
  `fit_model_for_fixed_rho_with_adaptive_kkt`
  (`src/solver/pirls.rs:7056`).
- CPU oracle inputs: `PirlsProblem` (carries `gaussian_fixed_cache`,
  `src/solver/pirls.rs` near 6934), `PenaltyConfig`
  (`src/solver/pirls.rs:7010`), `PirlsConfig` (`src/solver/pirls.rs:7742`),
  `AdaptiveKktTolerance` (passed at `src/solver/pirls.rs:7062`).
- CPU oracle outputs: `PirlsResult` (`src/solver/pirls.rs:6302`),
  `PirlsStatus` (`src/solver/pirls.rs:6233`), and
  `WorkingModelPirlsResult` carrying `constraint_kkt`
  (constructed at `src/solver/pirls.rs:7442` in the Gaussian-Identity
  short-circuit; the iterative path constructs it analogously later in
  the same function).
- GPU loop driver entry: `pirls_loop_on_stream`
  (`src/solver/gpu/pirls_gpu.rs:1359`).
- GPU loop driver body: `cuda::pirls_loop`
  (`src/solver/gpu/pirls_gpu.rs:913`).
- GPU loop workspace: `cuda::PirlsLoopWorkspace`
  (`src/solver/gpu/pirls_gpu.rs:854`).
- GPU loop outcome: `cuda::PirlsLoopOutcome`
  (`src/solver/gpu/pirls_gpu.rs:901`).
- GPU swap site (consumer): `sigma_cubature_dispatch`
  (`src/solver/reml/eval.rs:167`) and its readiness probe
  `device_pirls_stage3_ready` (`src/solver/reml/eval.rs:138`).
- Errors that must propagate: `EstimationError::PerfectSeparationDetected`
  (`src/solver/estimate.rs:1675`),
  `EstimationError::PirlsDidNotConverge`
  (raised across `src/solver/pirls.rs:4630`, `:6027`),
  `EstimationError::ConditionNumberExceeded`
  (used along the same outer-REML error paths in
  `src/solver/reml/runtime.rs`).

## Today's GPU contract (Stage 3.3 baseline)

```rust
pub fn pirls_loop_on_stream(
    shared: &PirlsGpuSharedData,
    ws: &mut SigmaPirlsGpuWorkspace,
    loop_ws: &mut cuda::PirlsLoopWorkspace,
    family: crate::gpu::pirls_row::PirlsRowFamily,
    curvature: crate::gpu::pirls_row::CurvatureMode,
    beta0: ArrayView1<'_, f64>,
    y: ArrayView1<'_, f64>,
    prior_w: ArrayView1<'_, f64>,
    penalty_hessian: ArrayView2<'_, f64>,
    lm_ridge: f64,
    max_iter: usize,
    tol: f64,
) -> Result<PirlsLoopOutcome, String>;
```

`PirlsLoopOutcome` returns `{ beta, penalized_hessian, logdet, deviance,
iterations, converged }`. The inner step is
`solve_step_on_stream_device` (`src/solver/gpu/pirls_gpu.rs:376`), which
forms `Xᵀ W X + S_λ` from device-side `w_solver` / `grad_eta` and a
host-supplied `p×p` `penalty_hessian` matrix.

Salient simplifications vs. CPU:

1. `penalty_hessian` is a single dense `p×p` matrix uploaded each call
   — no canonical penalty list, no balanced root, no reparameterization
   invariant, no Kronecker structure, no penalty shift / prior mean.
2. β lives in whatever basis the caller supplies. There is no
   transform-back step at exit, so β has no contract relative to the
   canonical / Qs / Kronecker frame the REML caller chose.
3. No constraint set is consulted. No coefficient lower bounds, no
   linear inequalities, no projected-gradient / active-set updates.
4. Line search is a fixed bisection ladder
   (`[1, 0.5, 0.25, …, 0.015625]`, `src/solver/gpu/pirls_gpu.rs:1054`)
   driven by raw deviance, with a fallback that takes α=1 if no descent
   is found and lets "outer trust-region adjust". The CPU loop runs LM
   with adaptive damping, ridge escalation, and an explicit
   `LmStepSearchExhausted` exit.
5. Errors are `Result<_, String>` (raw CUDA messages), never converted
   to `EstimationError`. There is no `Unstable` /
   `PerfectSeparationDetected` detection: blown-up η just produces
   non-finite deviances that the line search silently rejects.
6. No Gaussian-Identity fast path. Identity-link Gaussian models go
   through the full iterative loop on the GPU even though one PLS solve
   suffices (and the CPU oracle takes that short-circuit at
   `src/solver/pirls.rs:7332`).
7. No post-convergence "observed-curvature finalization": the returned
   `penalized_hessian` is the Hessian assembled at the last accepted
   step under whatever `curvature: CurvatureMode` the caller chose, with
   no Fisher → observed promotion for non-canonical links.
8. No `WorkingModelPirlsResult` (no `constraint_kkt`, no
   `min_penalized_deviance`, no `gradient_natural_scale`, no
   `firth`). Sigma-cubature's downstream covariance accumulator only
   needs `β` and `H`, but the certified-minimum claim that the
   accumulator relies on cannot be made from today's outcome alone.

## The feature surface gap, item by item

For each item: **what the CPU does**, **where it lives**, **what calls
it** (or what cubature math relies on it), and **what must change in
`pirls_loop_on_stream` to honor it**.

### 1. Canonical penalties + balanced root + reparameterization invariant

**What.** `fit_model_for_fixed_rho_with_adaptive_kkt` runs the entire
inner loop in a stable transformed basis chosen by the eigendecomposition
engine. Inputs are `PenaltyConfig::canonical_penalties`
(`src/solver/pirls.rs:7015`), `balanced_penalty_root`
(`:7016`), and `reparam_invariant` (`:7017`). The balanced root is
either reused from the cache or built by
`create_balanced_penalty_root_from_canonical`
(`src/solver/pirls.rs:7086`–`7093`). The dense reparam engine
`stable_reparameterization_engine_canonical` is invoked at
`src/solver/pirls.rs:7194` to produce `qs` (the transform back to
original coordinates), `s_transformed`, `e_transformed`, and the
log-determinant data REML needs.

**Where called from.** Every outer REML evaluation builds a
`PenaltyConfig` with canonical penalties (search:
`PenaltyConfig {` across `src/solver/reml/`). The sigma-cubature path
in particular hits this through
`RemlState::compute_smoothing_correction_auto`
(`src/solver/reml/eval.rs:702`), whose caller already lives in the
transformed basis.

**Why cubature relies on it.** The sigma-cubature integrand is a
quadrature over **perturbed posteriors at fixed transformed coordinates**;
the perturbed `β` and `H` must come back in the same basis the outer
REML caller is integrating over. If the GPU loop runs in original
coordinates while the outer loop expects transformed ones, every
`(A_m, b_m)` pair fed into
`accumulate_sigma_cubature_total_covariance` is rotated against the rest
of the batch and the f64 round-off contract pinned by
`cubature_linear_exactness_recovers_jvjt` (`src/solver/reml/eval.rs`
near 920) breaks.

**Gap in GPU.** `pirls_loop_on_stream` takes a single `p×p`
`penalty_hessian` and treats it as opaque. It does no
reparameterization, has no `qs`, and cannot rebuild
`s_transformed` / `e_transformed` for the REML-side log-determinant.

**What must change.**

- Replace the `penalty_hessian: ArrayView2<f64>` parameter with a
  caller-supplied "penalty bundle" mirroring `PenaltyConfig`:
  canonical penalty list, optional balanced root, optional reparam
  invariant, `p`, optional `kronecker_factored`, `penalty_shrinkage_floor`.
- Run the reparameterization engine host-side once per call (the engine
  is small relative to the inner loop; do not port it to CUDA), upload
  `s_transformed` (size `p×p`) and the shift vector once, and keep `qs`
  on the host for the final back-transform.
- At loop exit, materialize the final outcome in either the transformed
  basis (preferred — matches the CPU oracle's `PirlsResult.beta_transformed`)
  or apply `qs` host-side to land in original coordinates. The exit
  contract must be explicit, not implicit.
- Emit the log-determinant of `S_λ` (already computed by the host-side
  engine) on the outcome so the REML caller does not need a second
  eigendecomposition.

### 2. Coefficient lower bounds + linear inequality constraints

**What.** `PenaltyConfig::coefficient_lower_bounds` (`Array1<f64>`,
`src/solver/pirls.rs:7019`) and `linear_constraints_original`
(`LinearInequalityConstraints`, `:7020`) carry the constraint set.
They are first transformed into the inner basis by
`build_transformed_lower_bound_constraints`
(`src/solver/pirls.rs:8107`), the `_with_transform` variant for the
Kronecker case (`:8128`),
`build_transformed_linear_constraints` (`:8156`), and merged via
`merge_linear_constraints`. The active-set / projected-gradient solver
in `src/solver/active_set.rs` enforces them inside the inner loop; KKT
diagnostics are produced by `compute_constraint_kkt_diagnostics` (called
at `src/solver/pirls.rs:7443`).

**Where called from.** Monotonic + shape-constrained smooths (e.g.
the manifold smooths catalogued in `docs/manifold-smooths.md`), the
survival baseline LS path (project memory
[`project_survival_baseline_perf`]), and any caller using
`coefficient_lower_bounds` to clamp a non-negativity constraint.

**Why cubature relies on it.** The cubature integrand assumes each sigma
fit returns a **certified minimum of the perturbed penalised
negative-log-likelihood subject to the same constraint set as the
outer fit**. An unconstrained device fit at a sigma point can return a
β that violates the original-coordinate constraints; mapping that β
through the outer-coordinate covariance assembly yields a posterior
correction that includes infeasible support and biases the smoothing
covariance.

**Gap in GPU.** None. `pirls_loop_on_stream` does no projection, no
clamping, no active-set bookkeeping.

**What must change.**

- Accept `coefficient_lower_bounds: Option<ArrayView1<f64>>` and
  `linear_constraints_original: Option<&LinearInequalityConstraints>`
  on the GPU entry signature.
- Run the host-side transform builders (already used by the CPU oracle)
  before the GPU loop, upload the transformed constraint matrices to a
  small device buffer.
- At each accepted step, run a device-side projection kernel (or, for
  the lower-bounds-only case, a clamp kernel; both are `p`-vector
  pointwise ops and cheap). At loop exit, call
  `compute_constraint_kkt_diagnostics` host-side after downloading the
  final β + gradient, and surface the diagnostics on `PirlsLoopOutcome`.
- Exit with a `LmStepSearchExhausted` analog if the active-set updates
  cannot find a feasible descent direction within the iteration budget.

### 3. Kronecker-factored fast path

**What.** When `PenaltyConfig::kronecker_factored` is `Some(...)`
(`src/solver/pirls.rs:7029`), the CPU oracle skips the dense
eigendecomposition and uses `kronecker_reparameterization_engine`
(`src/solver/pirls.rs:7112`). The transform is
`WorkingReparamTransform::Kronecker(KroneckerQsTransform)`
(`:7120`, `:7207`–`7217`) and the penalty stays diagonal in the factored
basis via `build_diagonal_penalty_from_kronecker`
(`src/solver/pirls.rs:6803`). This is the standard path for tensor-product
smooths (e.g. cylinder / torus / hex tensors from
`docs/manifold-smooths.md`).

**Where called from.** Any tensor-product term in the formula DSL
(project memory [`project_gam_geometric_smooths`]).

**Why cubature relies on it.** Same coordinate-system argument as item
1, but with even larger penalty sizes (Kronecker tensors blow `p` past
the dense engine's comfort zone). Without this fast path, sigma fits at
tensor-smooth models either run out of memory or slow to the point that
the CPU Rayon path stays cheaper.

**Gap in GPU.** None. The GPU loop assembles a dense `p×p` matrix
unconditionally.

**What must change.**

- When the caller supplies `kronecker_factored`, the host-side preflight
  runs `kronecker_reparameterization_engine`, builds the diagonal
  penalty, and uploads only the diagonal vector (size `p`) plus the
  Kronecker `qs` factors needed for the final back-transform.
- Replace the `Xᵀ W X + S_λ` GEMM with a kernel that adds a diagonal
  penalty to the Hessian. The diagonal-penalty case is a single CUDA
  `axpy` against the Hessian diagonal — much cheaper than the
  dense-add it replaces.

### 4. Gaussian-Identity fixed-data cache

**What.** When `PirlsProblem::gaussian_fixed_cache` is `Some` and the
likelihood is Gaussian Identity with no Firth bias reduction, no
coefficient lower bounds, and no linear constraints
(`src/solver/pirls.rs:7339`–`7343`), the CPU oracle short-circuits the
entire iterative loop into a single `solve_penalized_least_squares_implicit`
call (`src/solver/pirls.rs:7349`). The cache pre-stores
`Xᵀ X` (`xtwx_sparse_orig` when applicable) so the solve only assembles
`Xᵀ X + S_λ` and `Xᵀ y` (`src/solver/pirls.rs:7993`).

**Where called from.** The Gaussian-Identity outer REML loop is the
dominant biobank-scale workload (project memory
[`project_biobank_perf`]).

**Why cubature relies on it.** Sigma-cubature at Gaussian Identity is
the cheapest possible per-sigma fit; without the cache, the GPU loop
runs `max_iter` iterations to converge on a problem that is exactly
linear. The cache is what makes the cubature dispatch break even versus
CPU Rayon at biobank-scale Gaussian Identity workloads.

**Gap in GPU.** No short-circuit. The loop always runs.

**What must change.**

- Add a Gaussian-Identity fast path entry to the GPU module that takes
  the cache's `Xᵀ X` cross-product (uploaded once, reused across all
  sigma points sharing the same X), assembles `Xᵀ X + S_λ` on-device,
  and runs a single Cholesky solve.
- Eligibility check at the GPU entry mirrors the CPU eligibility check
  (`src/solver/pirls.rs:7339`–`7343`).

### 5. PirlsStatus → EstimationError mapping

**What.** The CPU oracle classifies inner outcomes into the
`PirlsStatus` enum (`src/solver/pirls.rs:6233`):
`Converged`, `StalledAtValidMinimum`, `MaxIterationsReached`,
`LmStepSearchExhausted`, `Unstable`. Outer REML maps these to
`EstimationError` variants — `Unstable` becomes
`PerfectSeparationDetected` (`src/solver/reml/runtime.rs:6734`),
`MaxIterationsReached` / `LmStepSearchExhausted` become
`PirlsDidNotConverge`, and severe Hessian conditioning escalates to
`ConditionNumberExceeded`.

**Where called from.** Outer REML treats these as load-bearing — the
`Err(EstimationError::PerfectSeparationDetected { .. })` arms at
`src/solver/reml/runtime.rs:7441`, `:7591`, `:9276` short-circuit
the gradient flow into the documented "step back and damp" recovery.
Sigma-cubature inherits this through the REML state it integrates over.

**Why cubature relies on it.** A perfect-separation sigma point must
not return a finite-but-wrong β to the covariance accumulator. The
correct behavior is to either (a) propagate the error and let the outer
loop reject the cubature step, or (b) carry the diagnostic so the
accumulator can drop that point. Silently returning the LM-fallback
α=1 step (current behavior at
`src/solver/gpu/pirls_gpu.rs:1094`–`1128`) violates both options.

**Gap in GPU.** All errors are `Result<_, String>` raw CUDA messages.
Non-finite deviances and divergent η are swallowed by the line search
and emitted as a "converged" outcome with garbage β.

**What must change.**

- Replace `Result<PirlsLoopOutcome, String>` with
  `Result<PirlsLoopOutcome, EstimationError>` and carry a `status:
  PirlsStatus` field on `PirlsLoopOutcome`.
- Detect divergence on-device by reducing
  `max(|η|)` each iteration and tripping `PirlsStatus::Unstable`
  when the threshold matches the CPU oracle's separation detection
  (search for `PerfectSeparation` raise sites in
  `src/solver/pirls.rs` to lift the threshold expression verbatim).
- Detect LM exhaustion (line search fails for K consecutive iters, the
  CPU oracle's exit condition) and emit `LmStepSearchExhausted` instead
  of the current "take α=1 anyway" fallback.
- Wire CUDA errors into `EstimationError::InvalidInput(format!("gpu
  pirls: {e}"))` so the outer error-handling paths are uniform.

### 6. `enforce_constraint_kkt` on the converged mode

**What.** After the inner loop returns, the CPU oracle invokes
`compute_constraint_kkt_diagnostics` on the converged
`(beta_transformed, gradient, linear_constraints)`
(`src/solver/pirls.rs:7442`–`7444`) and stores the result on
`WorkingModelPirlsResult.constraint_kkt`. The adaptive-KKT tolerance
hook (`adaptive_kkt_tolerance` parameter,
`src/solver/pirls.rs:7062`; consumer at `src/solver/pirls.rs:4695`)
selectively tightens or loosens the KKT acceptance band based on the
outer-loop residual progress so a sigma cubature point whose true
optimum sits on the constraint boundary is **certified at the
boundary** rather than rejected for a `μ > 0` slack.

**Why cubature relies on it.** Sigma-cubature's pseudo-likelihood
contract is that each `(A_m, b_m)` pair represents the Hessian and
gradient of a **constrained minimum**. A non-KKT point makes the
Schur-complement covariance correction incoherent because the gradient
is nonzero in the active directions.

**Gap in GPU.** None.

**What must change.**

- After the GPU loop exits, download the final β and gradient (or
  compute the gradient on-device, also p-sized), call
  `compute_constraint_kkt_diagnostics` host-side, and attach to
  `PirlsLoopOutcome`.
- If the supplied `adaptive_kkt_tolerance` indicates the KKT residual
  is out of band, surface a `KktNotSatisfied` discriminant in
  `PirlsStatus` (extension to the enum) so the cubature dispatch can
  reject the point cleanly.

## Out of scope for this audit (intentionally deferred)

- Firth bias reduction. Sigma-cubature does not run on Firth models in
  the current REML state; if/when it does, the GPU loop will need the
  Firth correction term assembled on-device. Tracked separately.
- Post-convergence Fisher → observed curvature finalization for
  non-canonical links (`PirlsResult.exported_laplace_curvature` and
  `finalweights` semantics, `src/solver/pirls.rs:6332`–`6342`). The
  cubature accumulator's covariance correction is invariant under the
  curvature kind, so finalization is a CPU-side post-pass on the
  downloaded `β` and is not on the critical path for closing the
  predicate.
- Persistent warm-start machinery
  (`src/solver/persistent_warm_start.rs`). The cubature dispatch
  supplies its own per-point warm start (the unperturbed REML β), so the
  warm-start cache is not in the GPU path.

## Suggested commit order (matches charter deliverables)

1. **This audit doc.** (this commit)
2. Refactor `pirls_loop_on_stream` to honor canonical penalties +
   reparam invariant (item 1). Parity test against CPU full PIRLS on a
   small reparameterized fixture.
3. Wire coefficient lower bounds + linear constraints (item 2). Parity
   test on a monotone-clamped fixture.
4. Wire Kronecker-factored fast path (item 3). Parity test on a tensor
   smooth.
5. Wire Gaussian-Identity fixed-data cache (item 4). Parity test on a
   Gaussian-Identity fixture.
6. Wire `PirlsStatus` error mapping (item 5). Test that
   `PerfectSeparationDetected` propagates from a separated fixture.
7. Wire `enforce_constraint_kkt` (item 6). Test that converged β
   satisfies KKT at a fixture whose true optimum is on the constraint
   boundary.
8. Flip `device_pirls_stage3_ready` to a real probe (RAM-aware: keep
   `false` when CUDA runtime is absent), notify sigma-cubature-v3 by
   `SendMessage`.

Each step lands as one atomic commit pushed direct to `main`, parity
tests pinned narrowly via `cargo test -p gam <name>` (never the full
suite; project memory [`feedback_no_full_test_runs`]).
