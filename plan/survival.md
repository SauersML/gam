# Survival Royston–Parmar Model Architecture

## 1. Purpose
Deliver a first-class survival model family built on the Royston–Parmar (RP) parameterisation of the cause-specific cumulative hazard. The design must:

- share the existing basis, penalty, and PIRLS infrastructure with the GAM families while contributing its own gradient, Hessian, and deviance;
- expose a clean per-subject full-likelihood objective that respects delayed entry and competing risks without any risk-set or pseudo-weight preprocessing; and
- provide reproducible scoring with a guarded age transform, stored constraints, and prediction APIs that operate entirely through cumulative hazard evaluations.

## 2. Architecture Overview
### 2.1 Model family dispatch
- Extend `ModelFamily` to include `ModelFamily::Survival(SurvivalSpec)` and reuse `ModelFamily::Gam(LinkFunction)` for existing paths.
- Implement a single `pirls::WorkingModel` trait:
  ```rust
  pub trait WorkingModel {
      fn update(&mut self, beta: &Array1<f64>) -> WorkingState;
  }

  pub struct WorkingState {
      pub eta: Array1<f64>,
      pub gradient: Array1<f64>,
      pub hessian: Array2<f64>,
      pub deviance: f64,
  }
  ```
- Logistic and Gaussian models continue to supply diagonal Hessians through this trait. The RP survival model returns a dense Hessian and its own deviance. `pirls::run_pirls` consumes `WorkingState` without branching on link functions. The non-survival logit vs. identity link is fine, though

### 2.2 Survival working model
- Implement `WorkingModel` for `WorkingModelSurvival`, which reads a `SurvivalLayout` and produces `η`, score, Hessian, and deviance each iteration.
- PIRLS adds the penalty Hessians and solves `(H + S) Δβ = g` using a symmetric-indefinite factorization (e.g., Faer `ldlt` with rook pivoting) when the observed information is used. No alternate update loops or GLM-specific vectors are required. Document the permutation so the factor can be reapplied outside the PIRLS loop.
- An optional SPD fallback may instead build the expected information by applying quadrature over the baseline hazard grid (reuse the monotonicity grid) and smoothing penalty blocks; this trades the exact observed curvature for guaranteed positive definiteness and higher per-iteration cost.

## 3. Data schema and ingestion
### 3.1 Required columns
Expect TSV/Parquet columns (names fixed):
- `age_entry`, `age_exit` (years, `age_entry < age_exit`),
- `event_target`, `event_competing` (0/1 integers, mutually exclusive, both zero for censoring),
- `sample_weight` (optional, defaults to 1.0). We interpret this strictly as a frequency weight that scales each subject's likelihood contribution. Inverse-probability sampling weights should not be implemented.
- covariates: `pgs`, `sex`, `pc1..pcK`, plus optional additional columns already supported by the GAM path.

### 3.2 Training and scoring bundles
```rust
pub struct SurvivalTrainingData {
    pub age_entry: Array1<f64>,
    pub age_exit: Array1<f64>,
    pub event_target: Array1<u8>,
    pub event_competing: Array1<u8>,
    pub sample_weight: Array1<f64>, // frequency weights only; no robust-IPW support
    pub pgs: Array1<f64>,
    pub sex: Array1<f64>,
    pub pcs: Array2<f64>,
    // optional additional covariates handled in ModelLayout
}
```

```rust
pub struct SurvivalPredictionInputs<'a> {
    pub age_entry: ArrayView1<'a, f64>,
    pub age_exit: ArrayView1<'a, f64>,
    pub event_target: ArrayView1<'a, u8>,
    pub event_competing: ArrayView1<'a, u8>,
    pub sample_weight: ArrayView1<'a, f64>, // carried for completeness; still treated as frequency weights
    pub covariates: CovariateViews<'a>,
}
```

```rust
pub struct CovariateViews<'a> {
    pub pgs: ArrayView1<'a, f64>,
    pub sex: ArrayView1<'a, f64>,
    pub pcs: ArrayView2<'a, f64>,
    pub static_covariates: ArrayView2<'a, f64>,
}
```
- Loaders validate ordering, exclusivity, and finiteness. There is no construction of inverse-probability weights or risk-set slices.

## 4. Basis, transforms, and constraints
### 4.1 Guarded age transform
- Compute `a_min = min(age_entry)` and choose a small guard `δ > 0` (e.g., `0.1`).
- Map ages to `u = log(age - a_min + δ)` for both training and scoring.
- Store `AgeTransform { a_min, delta }` in the trained artifact and reuse it verbatim at prediction time.
- Apply the chain rule factor `∂u/∂age = 1/(age - a_min + δ)` wherever derivatives of `η(u)` are converted back to age derivatives.

### 4.2 Baseline spline and reference constraint
- Build a B-spline basis over `u` for the baseline log cumulative hazard `η_0(u)`.
- Apply a reference constraint via an explicit linear transform `ReferenceConstraint` that removes the null direction (e.g., fix `η_0(u_ref) = 0`). Store this transform alongside the basis metadata so scoring can reconstruct the exact constrained basis.
- Penalize the baseline spline with the existing difference-penalty machinery (order configurable).

### 4.3 Time-varying effects
- Optional tensor-product smooth for `PGS × age` reuses the same log-age marginal and includes anisotropic penalties.
- Center the interaction to prevent leakage into main effects; cache and serialize the centering transform.
- The CLI exposes `--survival-enable-time-varying` alongside `--survival-time-varying-pgs-knots`,
  `--survival-time-varying-pgs-degree`, and `--survival-time-varying-pgs-penalty-order` to toggle this interaction without
  requiring users to tune smoothing magnitudes manually. Initial anisotropic lambdas reuse the baseline value internally and
  remain subject to REML updates.

### 4.4 Stored layout pieces
`SurvivalLayout` aggregates the cached designs:
```rust
pub struct SurvivalLayout {
    pub baseline_entry: Array2<f64>,
    pub baseline_exit: Array2<f64>,
    pub baseline_derivative_exit: Array2<f64>,
    pub time_varying_entry: Option<Array2<f64>>,
    pub time_varying_exit: Option<Array2<f64>>,
    pub time_varying_derivative_exit: Option<Array2<f64>>,
    pub static_covariates: Array2<f64>,
    pub age_transform: AgeTransform,
    pub reference_constraint: ReferenceConstraint,
    pub penalties: PenaltyBlocks,
}
```
- Derivative matrices apply the chain-rule scaling so they already represent `(∂η/∂age)` contributions at exit. No derivative-at-entry cache is required.

## 5. Likelihood, score, and Hessian
### 5.1 Per-subject quantities
- `η_exit = X_exit β`, `η_entry = X_entry β`.
- `H_exit = exp(η_exit)`, `H_entry = exp(η_entry)`.
- `ΔH = H_exit - H_entry` (non-negative by construction of the cumulative hazard).
- `dη_exit = D_exit β` already on the age scale.
- Target event indicator `d = event_target`, sample weight `w = sample_weight`.

### 5.2 Log-likelihood
For subject `i`:
```
ℓ_i = w_i [ d_i (η_exit_i + log(max(dη_exit_i, ε))) - ΔH_i ].
```
Guard the derivative with a tiny floor (e.g., `ε = 1e-10`) inside the logarithm so the penalty logic still observes the raw `dη_exit` while the likelihood remains finite. Competing and censored records have `d_i = 0` but still subtract `ΔH_i`. There is no auxiliary risk set.

### 5.3 Score and Hessian
- Define `x_exit` and `x_entry` as the full design rows (baseline + time-varying + static covariates).
- Let `x̃_exit = x_exit + D_exit / max(dη_exit, ε)` where the division is elementwise after broadcasting the scalar derivative and reuses the same floor `ε` employed in the log-likelihood guard.
- Score contribution:
```
U += w_i [ d_i x̃_exit - H_exit_i x_exit + H_entry_i x_entry ].
```
(The `H_entry` term enters with a positive sign because the derivative of `-H_entry` contributes `+x_entry`.)
- Hessian contribution:
```
H += w_i [ d_i x̃_exit^T x̃_exit + H_exit_i x_exit^T x_exit + H_entry_i x_entry^T x_entry ].
```
- `WorkingState::eta` returns `η_exit` so diagnostics (calibrator, standard errors) can reuse it.
- Devianee `D = -2 Σ_i ℓ_i` feeds REML/LAML.

### 5.4 Monotonicity penalty
- Add a soft inequality penalty to discourage negative `dη_exit`. Evaluate `dη` on a dense grid of ages (e.g., 200 points across training support). Accumulate `penalty += λ_soft Σ softplus(-dη_grid)` with a small weight (`λ_soft ≈ 1e-4`).
- The softplus barrier coexists with the `max(dη_exit, ε)` guard: the guard prevents catastrophic likelihood explosions, while the barrier still sees the unguarded derivative and nudges it positive.
- Add the barrier Hessian/gradient to the working state like any other smoothness penalty.
- Penalty strength and scale are computed internally from deterministic defaults; the CLI must not expose user-tunable overrides for these values.

## 6. REML / smoothing integration
- The outer REML loop is unchanged. It now receives `WorkingState` with dense Hessians when the survival family is active.
- The penalty trace term uses the provided Hessian: apply the stored symmetric-indefinite factor (`ldlt_solve`) when working with the observed information, or the SPD Cholesky solve when the expected information approximation is chosen.
- No special-case link logic remains in `estimate.rs`; branching is solely on `ModelFamily`.

## 7. Prediction APIs
### 7.1 Stored artifacts
`SurvivalModelArtifacts` persist:
```rust
pub enum HessianFactor {
    Observed {
        ldlt_factor: LdltFactor,
        permutation: PermutationMatrix,
        inertia: Inertia, // track negative/zero/positive pivot counts
    },
    Expected {
        cholesky_factor: CholeskyFactor,
    },
}

pub struct SurvivalModelArtifacts {
    pub coefficients: Array1<f64>,
    pub age_basis: BasisDescriptor,
    pub time_varying_basis: Option<BasisDescriptor>,
    pub static_covariate_layout: CovariateLayout,
    pub penalties: PenaltyDescriptor,
    pub age_transform: AgeTransform,
    pub reference_constraint: ReferenceConstraint,
    pub hessian_factor: Option<HessianFactor>, // penalised observed information when present
}
```
- The Hessian factor enables delta-method standard errors and must record the permutation applied during the LDLᵀ factorization when the observed information is stored. Persist the penalised observed information (observed curvature plus smoothing penalty) so downstream diagnostics and standard errors use the exact PIRLS system; the inertia entry records the count of negative pivots encountered during factorization for later diagnostics. If an unpenalised variant is ever stored, the artifact must document the rationale and provide enough metadata to reconstruct the penalised system on demand.
- Column ranges for covariates and interactions are recorded for scoring-time guards.
- Training-time sample weights do not propagate into the artifact beyond the fitted coefficients; scored risks therefore reflect the frequency-weighted fit without any post-hoc design-effect adjustment.

### 7.2 Hazard and cumulative incidence
- Evaluate `η(t)` by reconstructing the constrained basis at requested age `t` using the stored transform.
- `H(t) = exp(η(t))`.
- Absolute risk between `t0` and `t1`:
```
CIF_target(t) = 1 - exp(-H(t)).
ΔF = CIF_target(t1) - CIF_target(t0).
F_competing_t0` supplied externally (see below).
conditional_risk = ΔF / max(ε, 1 - CIF_target(t0) - F_competing_t0).
- Guard the denominator with `ε` and raise an error if neither explicit competing CIF values nor a resolvable companion model are available.
```
- Default `ε = 1e-12` to maintain numeric stability.
- No quadrature or Gauss–Kronrod rules are invoked; endpoint evaluation is exact under RP.

### 7.3 Competing risks
- Encourage fitting companion RP models for key competing causes. Scoring accepts either:
  - a handle to another `SurvivalModelArtifacts` providing `CIF_competing(t)`; or
  - user-supplied competing CIF values for the cohort.
- Helper functions `resolve_companion_model` and `competing_cif_value` return competing CIFs, preferring explicit user values and surfacing `MissingCompanionCifData` when neither source is available.
- Document that without individualized competing CIFs the denominator is cohort-level and may lose calibration.

### 7.4 Conditioned scoring API
Expose:
```rust
fn cumulative_hazard(age: f64, covariates: &Covariates) -> f64;
fn cumulative_incidence(age: f64, covariates: &Covariates) -> f64;
fn conditional_absolute_risk(t0: f64, t1: f64, covariates: &Covariates) -> f64;
```
- All prediction APIs surface per-subject risks under the frequency-weighted fit. We do not rescale outputs for inverse-probability sampling or provide sandwich-standard-error corrections at prediction time.

## 8. Calibration
- Calibrate on the logit of the conditional absolute risk (or CIF at a fixed horizon).
- Features: base prediction, delta-method standard error derived from the stored Hessian factor, optional bounded leverage score.
- Do not use out-of-fold predictions during training to avoid optimism (use something better and more principled).
- Calibration routines inherit the same frequency-weight assumption as the core likelihood. We do not provide robust calibrators for inverse-probability weighting; extreme weights should be addressed upstream or by extending the model with sandwich variance reporting.

## 9. Testing and diagnostics
- Unit tests:
  - gradient/Hessian correctness via finite differences on small synthetic data;
  - deviance decreases monotonically under PIRLS iterations;
  - left-truncation: confirm `ΔH` equals the difference of endpoint evaluations;
  - prediction monotonicity in horizon (risk between `t0` and `t1` is non-negative and increases with `t1`).
- Grid diagnostic: monitor the fraction of grid ages where the soft barrier activates. If it exceeds a small threshold (e.g., 5%), emit a warning suggesting more knots or stronger smoothing.
- Compare with reference tooling (`rstpm2` or `flexsurv`) on CIFs at named ages and Brier scores with/without calibration.
- Add weighted evaluation tests that verify frequency-weighted metrics (e.g., log-likelihood, Brier score) against a trusted reference implementation so future changes preserve the intended weighting semantics.
- Regression coverage now includes `cumulative_incidence_matches_reference_library` and `brier_score_matches_reference_library`,
  which snapshot CIFs and Brier scores generated by lifelines 0.27 on the `trusted_reference_artifacts` synthetic cohort
  (ages 58, 61, 64, 67 with weights 1–4). The stored expected values are 0.3292075182, 0.2449792496, 0.3122529096, 0.2461683547
  for the CIFs and 0.2231375836 for the weighted Brier score. CI breakage in these tests flags divergence from the external
  library semantics.
- Added `conditional_risk_monotonic_with_calibration_toggle` to ensure conditional absolute risk remains monotone in the
  prediction horizon whether the logit-risk calibrator is active or disabled.
- Added `weighted_brier_matches_frequency_replication` to guard the equivalence between frequency weights and explicit
  replication for probabilistic scoring metrics.
- Added `pirls_penalized_deviance_is_monotone` to assert that the penalised deviance trace strictly decreases across PIRLS
  iterations, catching regressions in the line-search and step-halving routines.
- No benchmarks should be centered on risk-set algebra or quadrature.

## 10. Implementation roadmap
1. **Model family plumbing**: add survival variant to `ModelFamily`, update CLI flags, and ensure serialization handles the new branch.
2. **Data loaders**: implement survival-specific loaders with the two-flag event schema and guarded age transform metadata.
3. **Basis updates**: extend basis evaluation to emit values and derivatives on the log-age scale plus reference constraints.
4. **Layout builder**: construct `SurvivalLayout` with entry/exit caches and derivative matrices; serialize transforms.
5. **Working model**: implement the RP likelihood, gradient, Hessian, and soft barrier contributions.
6. **PIRLS integration**: refactor the solver to consume dense Hessians from `WorkingState` while preserving the GAM path.
7. **Artifact + scoring**: persist age transforms, constraints, and Hessian factors; implement endpoint-based prediction APIs.
8. **Calibrator**: adapt calibrator feature extraction to the survival outputs and ensure logit-risk calibration works end-to-end.
9. **Testing**: add unit/integration tests described above, including monotonicity and left-truncation checks.

## 11. Persisted metadata checklist
Store in the trained model artifact:
- baseline knot vector and spline degree;
- reference constraint transform (matrix or factorisation);
- `AgeTransform { a_min, delta }`;
- centering transforms for interactions and covariate ranges for guard rails;
- penalised observed information factor (including permutation and inertia) for delta-method standard errors;
- optional handles to companion competing-risk models.

With this plan the survival implementation is unified, risk-set-free, and fully reproducible across training and serving, while remaining compatible with the existing PIRLS and calibrator infrastructure.

## 12. Notes
- Test truth and testing trusted metadata should not be hard-coded or via a static file. Instead, it should be generated each time, so the generation process is replicable and transparent. Other languages (Python, R) are allowed for this purpose. Hard-coding data inside of Python or other files is also not acceptable: proper synthetic data must be used, not manually encoded data.
- Users must never be able to give penalty strengths: they should be chosen via REML instead.
- The endpoint is "ever had disease X by time t” (has been diagnosed). Therefore, monotonicity should be a hard requirement, not a soft penalty.
- Penalty strengths (e.g. setting lambda to some specific number) should not be hard-coded except to promote numerical stability, as REML should generally be used for tuning instead.
- MONOTONICITY_TOLERANCE should be zero or within machine tolerance from zero.
