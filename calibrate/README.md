# Calibration crate

The `calibrate/` crate hosts the end-to-end pipeline that fits Gnomon's
penalized additive model and its optional post-hoc calibrator. The code in this
directory is responsible for ingesting the tabular training data, constructing
the spline bases and their penalties, solving the penalized iteratively
reweighted least squares (P-IRLS) updates with REML/LAML smoothing selection,
and (when enabled) training a secondary calibrator that sharpens probability
calibration or identity-scale accuracy.

## Statistical model

The primary estimator is a generalized additive model whose linear predictor is

```
η(x) = β₀ + f_{pgs}(PGS) + γ_{sex}·sex + Σ_j f_j(PC_j) + Σ_j f_{pgs,j}(PGS, PC_j) + f_{pgs,sex}(PGS, sex),
```

where `f_{pgs}` and each `f_j` are univariate spline smooths, `γ_{sex}` is an
unpenalized linear main effect for the binary sex covariate, `f_{pgs,j}` are
tensor-product interactions between the polygenic score and the _j_-th
principal component, and `f_{pgs,sex}` is a varying-coefficient term that lets
the polygenic score effect differ by sex. This structure is encoded in
[`construction.rs`](construction.rs), which wires the marginal and interaction
bases into a single design matrix with sum-to-zero constraints and ANOVA-style
orthogonalization so the additive terms are identifiable and interpretable.

Two likelihoods are supported via [`model::LinkFunction`](model.rs): logistic
(`Logit`) for binary traits and Gaussian identity (`Identity`) for continuous
phenotypes. For the logistic case the P-IRLS inner loop targets the binomial
deviance, while the Gaussian case profiles a scale parameter that is later
stored for prediction-time standard errors.

## Penalties and smoothing selection

Each smooth term is represented with B-spline bases generated in
[`basis.rs`](basis.rs). Difference penalties of configurable order control the
wiggliness of the univariate smooths, tensor-product penalties regularize the
interactions with PCs using directional smoothness along the PGS and PC axes
plus an explicit null⊗null shrinkage, and a wiggle-only penalty tempers the
sex×PGS varying coefficient while its null space is handled by the purity
projection. These penalties
respect the null spaces implied by the ANOVA constraints so that intercepts,
sex main effects, and other lower-order components remain unpenalized by
construction.

Smoothing parameters (`λ`) are learned rather than fixed. [`estimate.rs`](estimate.rs)
implements a nested optimization in the style of Wood (2011): the inner loop is
the penalized IRLS solver that returns coefficients for any proposed set of
`λ`, and the outer loop runs a box-constrained BFGS optimizer on the marginal
likelihood of those smoothing parameters. Gaussian models maximize the exact
restricted likelihood (REML), while non-Gaussian models maximize the Laplace
approximate marginal likelihood (LAML); both objectives include stabilization
priors and null-space accounting to prevent degeneracy.

## Optimization strategy

The P-IRLS solver iteratively forms working responses and weights, solves the
penalized normal equations with the `faer` linear algebra backend, and checks
for hazards such as separation or ill-conditioning. Transformed Hessians, trace
corrections, and effective degrees of freedom are cached so the outer REML/LAML
optimizer can evaluate gradients efficiently. The final Hessian, effective
penalty factors, and fitted scale (when applicable) are preserved inside the
[`TrainedModel`](model.rs) artifact for downstream uncertainty estimates.

## Optional calibrator layer

When enabled, the pipeline trains a secondary additive model that adjusts the
base predictions. [`calibrator.rs`](calibrator.rs) derives approximate
leave-one-out diagnostics from the converged base fit—baseline predictor,
standard error, and signed distance to the peeled hull—and feeds them into a
compact spline design whose smoothing parameters are optimized with the same
REML/LAML machinery. The calibrator honors the original link function, preserves
identity-scale means, and records its own penalties, coefficients, and scaling
metadata so predictions can reproduce the post-hoc correction faithfully.

## What lives where

- [`data.rs`](data.rs) reads TSV inputs with Polars, enforces the strict schema
  (`phenotype`, `score`, `sex`, `PC*`, optional `weights`, optional
  `sample_id`), and produces the `TrainingData` bundle consumed by the
  optimizer. The same module also validates prediction inputs, defaults missing
  weights to 1.0, and generates stable sample identifiers when none are
  provided.
- [`basis.rs`](basis.rs) builds B-spline bases, applies sum-to-zero constraints,
  and constructs difference penalties that ultimately drive smoothness control
  for both the primary model and the calibrator.
- [`construction.rs`](construction.rs) converts configuration into concrete
  design matrices and penalty layouts. It tracks how every block of columns maps
  to penalties and interaction terms so the downstream optimizer can assemble
  the correct objective.
- [`pirls.rs`](pirls.rs) implements the weighted least squares inner loop shared
  by the base fit and the calibrator. It accepts arbitrary designs along with
  penalty metadata produced by `ModelLayout`.
- [`estimate.rs`](estimate.rs) is the orchestrator. `train_model` triggers design
  construction, runs REML/BFGS over smoothing parameters, guards against
  degeneracy, and optionally launches the calibrator fitting routine.
- [`hull.rs`](hull.rs) derives a peeled convex hull around the training domain
  and provides signed-distance queries that the calibrator uses to detect and
  damp extrapolation.
- [`calibrator.rs`](calibrator.rs) converts base-model diagnostics into
  calibrator features, builds its own constrained design, fits smoothing
  parameters, and exposes inference helpers for applying the learned correction.
- [`model.rs`](model.rs) defines the serialized `TrainedModel`, handles
  persistence, and applies both the baseline GAM and the optional calibrator at
  prediction time.
- [`faer_ndarray.rs`](faer_ndarray.rs) bridges `ndarray` structures with the
  `faer` linear algebra backend for eigen decompositions and solves, which keeps
  large penalized systems numerically stable.

## Training flow at a glance

1. **Load and validate data** – Callers invoke `data::load_training_data`, which
   reads the TSV file with Polars, verifies column types (including the
   binary `sex` column), enforces the minimum-row requirement, and returns
   `TrainingData` containing the phenotype, score, sex indicator, principal
   components, and a weight vector (defaulting to ones when the file omits
   `weights`).
2. **Construct the base GAM** – `estimate::train_model` delegates to
   `construction::build_design_and_penalty_matrices`. This step selects knot
   vectors via `basis`, assembles the block-structured design matrix `X`, and
   pairs each block with its penalty matrices. The resulting `ModelLayout`
   tracks the intercept, sex main effect, PGS smooth, PC main effects,
   tensor-product interactions, the sex×PGS varying coefficient, and their
   null-space bases.
3. **Optimize smoothing parameters** – With the design in place, the REML
   optimizer in `estimate.rs` alternates between P-IRLS solves (via `pirls.rs`)
   and BFGS updates over log-smoothing parameters until convergence or until it
   detects degeneracy (separation, rank deficiency, etc.). Survival models seed
   their baseline and monotonic penalties from basis metadata and age ranges,
   but the final smoothing strengths are still chosen automatically by the
   REML/BFGS loop.
4. **Capture geometric guards** – During training the optimizer builds a peeled
   hull from the polygenic score and principal components. The hull and its
   signed-distance function inform both the calibrator and future prediction
   time clamping.
5. **Optional calibrator fitting** – If `model::calibrator_enabled()` returns
   true, `estimate::train_model` computes approximate leave-one-out (ALO)
   diagnostics from the converged fit using `calibrator::compute_alo_features`.
   Those diagnostics (baseline predictor, its standard error, and the signed
   distance to the peeled hull, along with the identity-scale baseline needed
   for constraints) feed into `calibrator::build_calibrator_design`, which
   mirrors the spline machinery to create a smaller additive model. REML
   smoothing selection is reused through `calibrator::fit_calibrator`, yielding
   coefficients, block lambdas, and (for identity models) a residual scale
   estimate.
6. **Persist the result** – The fitted coefficients, smoothing parameters, hull,
   and calibrator (if present) are serialized through `model::TrainedModel` so
   that downstream tools can reload them exactly.

## Prediction path

`ModelConfig::predict` reconstructs the required spline bases using the stored
knot vectors and transformation matrices, evaluates the base GAM, and, when a
calibrator is attached, calls `calibrator::predict_calibrator` to adjust the
linear predictor. Logistic models return calibrated probabilities; identity
models add the calibrator's correction in linear space. Distance-to-hull checks
run in the same order as training, ensuring extrapolation handling remains
consistent.

## Expected data format

Training and inference both operate on tab-separated value (TSV) files with a
header row and strictly named columns. The loader surfaces actionable errors if
any of the schema requirements below are violated.

### Training inputs

`data::load_training_data` expects the following columns when fitting the base
model (and optional calibrator):

- `phenotype` – numeric response (0/1 for logistic fits, real-valued for
  Gaussian fits). Missing values are not permitted.
- `score` – the standardized polygenic score used as the primary smooth.
- `sex` – binary indicator encoded as 0/1. Other encodings are rejected during
  type coercion.
- `PC1`, `PC2`, …, `PCk` – one column per requested principal component. The
  number of PCs must match the `num_pcs` configuration supplied to the CLI or
  library entry point. Columns are required even if they are all zeros.
- `weights` (optional) – positive prior weights. When omitted, the loader
  supplies a length-`n` vector of ones so unweighted fits do not require a
  synthetic column.
- `sample_id` (optional) – string identifiers passed through to the serialized
  model for auditing. Absent IDs are replaced with deterministic `1`, `2`, …
  labels.

All required columns must be finite, and at least 20 rows are recommended for a
stable fit. The loader prints the resolved schema so callers can confirm the
exact set of covariates that entered the design.

### Inference inputs

`data::load_prediction_data` enforces the same structure for prediction-time
files, minus the response column:

- `score`, `sex`, and the `PC*` columns must be present and numeric.
- Optional `sample_id` values are used to label rows in the prediction outputs
  (filling with `1`, `2`, … when absent).

Weights and phenotypes are ignored during inference. Prediction data is
validated with the same finite-value checks as training data, ensuring that
the deployed spline bases receive well-formed covariates.

## TODO
Implement the survival model in plan/survival.md.
