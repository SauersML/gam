# Robustness and edge-case audit: gamfit 0.1.267 vs pyGAM 0.12.0

Axis: a fuzz-style differential test over hostile and degenerate inputs.

Harness: `scratchpad/audit/robustness/`.
- `cases.py` defines 50 cases.
- `run_one.py` runs one library on one case in a subprocess.
- `driver.py` sets a 180 s timeout and records CRASH or HANG.
- Results are in `results.json`.
- Probes: `probe_*.py` and `firth_ref.py`, an independent numpy Firth logistic.

Caveats:
- The installed wheel is gamfit 0.1.267, built 2026-09-09. The repo HEAD is 2026-09-18 and about 50 commits newer, so where HEAD code already differs I say so.
- No cargo build was run, per the constraints.
- The machine load average was about 40 on 4 CPUs, so absolute timings are inflated for both libraries.

Classification: (a) errors clearly, (b) returns garbage silently, (c) hangs, (d) crashes, (e) fits sensibly.

## Case matrix

| case | gamfit | pyGAM | note |
|---|---|---|---|
| tiny_n5, s(x) | e: linear, edf 1.96, certified | b: edf 0.03, predicts ~0 everywhere | G better |
| tiny_n5, k=20 | e: same as above | b | G better |
| tiny_n12, k=20 | **a/bug**: IntegrationError "smoothing cubature has no positive-width proposal" | e: edf 1.5, near flat | F2 |
| n=1 | a: "Binomial response degenerate" (auto family picked binomial for y=[1.0]) | b: edf 0 | F7 (odd message) |
| duplicated_x | e | e | |
| constant_covariate s(z) | a: clear actionable refusal | b: silently fits edf 11 | G better |
| two_unique s(x) | **a/bug**: cubature IntegrationError | e: exact 3-point fit | F2 |
| three_unique s(x) | **a/bug**: cubature IntegrationError | e | F2 |
| x scale 1e9 / 1e-9 / offset 1e9 | e: bit-identical fits | e | G scale-equivariant |
| y*1e12 | e: identical up to scale | e | |
| y*1e-12 | **a/bug**: "effectively constant (sd 7.7e-13 <= 1e-10)" | e | F3 |
| heavy-tail (Cauchy) y | b: collapses to near-linear (RMSE 0.53/11.6/0.70 over 3 seeds) | b: worse (RMSE 0.79/29.0/1.32) | both non-robust; G better |
| gross y outliers | b | b | both non-robust |
| NaN/inf in x, y, w | a: DataError naming column and 1-based row | a | G message better |
| NaN at predict | a: generic GamError | a | F7 (class) |
| s(x) perfect separation | **a/bug**: IntegrationError "canonical-logit inverse-link jet eta=-745 produced 0.0" after 5 s | e/b: 0/0/1/1, CI contains 0 width | F4 |
| all-ones logistic | a: clear degenerate message | b: CI [0,0] | G better |
| y ~ x separation | e: automatic Firth; coefficients match independent Firth reference exactly (0.399, 108.09, SE 53.9) | b: 0/1, no info | G better |
| all-zero Poisson | a: clear | b: CI [0,0] | G better |
| Poisson y<0 | a: front-door | a | |
| Poisson y=2.5 | a but late: IntegrationError from "outer seed screening" | e (quasi) | F7 |
| binomial y=2, Gamma y<0 | a | a | |
| Gamma y=0 | a: clear | b: silent, log(0) | G better |
| collinear smooths, exact duplicate smooths | e: certified, edf 8.4 | e: edf 13.3 | |
| 500-level categorical, n=3000, group(g) | **c**: over 180 s in the harness (>6.5 min wall in a standalone run under load, not finished at write time) | e: 92 s, but predict then fails on object dtype | F6 |
| unseen level, fixed factor | a | a | |
| unseen level, group(g) | e: population-level prediction with wider SE | a | G better |
| extrapolation (Gaussian, logit) | e: linear tails | e | see F5 for SE in tails |
| subset zero weights | e | e | |
| all-zero weights | a but deep: IntegrationError "profiled Gaussian residual dof ... n(0) − M_p(1)" | a: OptimizationError "PIRLS diverged" | F7 |
| negative weights | a: SchemaMismatchError with wrong help text ("Verify the new data...training data") | b: silently fits | G better, message wrong: F7 |
| float32, int, non-contiguous | e | e | |
| 30 s() terms, n=60 | a: "need at least 62 rows" | b: edf 42, predictions uncorrelated with truth | G better (arguably conservative) |
| constant y | **a/bug**: cubature IntegrationError (#1856 says it must fit) | e | F2 |
| pure noise | e: edf 1.89 | b: edf 11.2 (overfits) | G better |
| te(x,z), n=15 | **a/bug**: cubature IntegrationError | e: edf 1 | F2 |
| one x at 1e6, rest in [0,1] | **a/bug**: cubature IntegrationError; bs=tp silently flat (edf 1) | b: flat, edf 2 | F2 / F2b |
| empty data, mismatched lengths | a | a | |
| pandas DataFrame without pyarrow | **a/bug**: ImportError "Import pyarrow failed" | e | F1 |

No crashes (d) in either library. The only hang (c) is gamfit on the 500-level categorical.

## Findings

### F1: pandas input requires pyarrow, which is not a dependency (bug, med, size S)
**Reproducer** (pandas 3.0.6, pyarrow not installed):
```
gamfit.fit(pd.DataFrame({"x": x, "y": y}), "y ~ s(x)")
```
This raises `ImportError: Import pyarrow failed`.

**Root cause:**
- `gamfit/_tables.py` in `normalize_table` (~L75-115) routes any object exposing `__arrow_c_stream__` to `encoded_table_from_arrow`. pandas 3 exposes that attribute even without pyarrow, and the stream call itself needs pyarrow.
- There is no fallback to the numpy column path.
- `pyproject.toml` L45-47 lists only numpy as a dependency; pyarrow is only in extras (L60, 72, 80, 93).

pyGAM works with plain pandas.

**Fix:** in `normalize_table`, treat a failed import or call of `__arrow_c_stream__` as "not arrow" and go through the existing dict-of-numpy path (DataFrame columns to numpy arrays). This keeps Python thin because it only picks a transport. Add a test that runs with pyarrow blocked.

**Files:** `gamfit/_tables.py`, tests.

### F2: The optional covariance-upgrade stage aborts whole fits on degenerate but valid inputs (bug, high, size M)
**Reproducers** (all raise `IntegrationError: smoothing cubature has no positive-width proposal in the resolved domain` on the 0.1.267 wheel):
- `y ~ s(x)` with x taking 2 unique values, and with 3 unique values. Also with `k=4` and `k=5`. `k=3` works.
- `y ~ s(x, k=20)` with n=12.
- An exactly constant y. Issue #1856 says this must fit.
- `y ~ te(x, z)` with n=15.
- `y ~ s(x)` where one x is 1e6 and the rest are U(0,1).

pyGAM fits all of these. The project's own fuzz log `bench/gha_results/fuzz/fuzz_output.txt` shows the same error.

**Root cause:**
- In the wheel, the TrialPointRefused raised at `crates/gam-solve/src/reml/eval.rs:1377-1380` escapes the sigma-point node calibration and kills the fit. This happens when rho sits at the domain edge, i.e. a smooth fully penalized to its null space.
- At HEAD, `eval.rs:1411-1419` catches the calibration error and finalizes with the first-order correction. The wheel binary lacks that string, so HEAD probably fixes the abort, but this is unverified without a build.
- Feeding cause: `heuristic_knots_for_column` (`crates/gam-terms/src/term_builder.rs:4511-4514`) floors at 4 internal knots, giving 8 cubic B-spline coefficients even when the column has 2 or 3 unique values. That creates an unidentifiable, fully railed penalty direction.
- The thin-plate path already reduces its basis to the number of unique values. The B-spline path does not.

**Fix:**
- (1) Ship HEAD's calibration fallback and add these 6 reproducers as Python regression tests. The first-order correction is a principled lower-order approximation, not a masking fallback, and the result already reports `covariance_source`.
- (2) Cap the default B-spline basis dimension at the number of unique x values (the identifiable dimension), the same way tp does. This is a data-determined limit, not a knob.

**Files:** `crates/gam-solve/src/reml/eval.rs`, `crates/gam-terms/src/term_builder.rs`, `tests/`.

### F2b: A single x outlier breaks the default uniform-knot s(x) (bug, med, size S)
**Reproducer:** `x[0] = 1e6`, the rest U(0,1), `y = sin(6x) + noise`.
- `y ~ s(x)`: cubature error on the wheel.
- `y ~ s(x, bs=tp)`: silently flat, edf 1.0, prediction -0.104 everywhere (b).
- `y ~ s(x, knot_placement=quantile)`: correct, edf 7.3, predictions [0.32, 0.97, 0.10, -0.88, -0.78] against truth [0.30, 1.00, 0.14, -0.94, -0.55].

**Root cause:** the default placement is Uniform (`crates/gam-terms/src/term_builder.rs:4973`, `None => Ok(BSplineKnotPlacement::Uniform)`). All internal knots land in [2e5, 8e5] with no data.

pyGAM is equally flat, so this is parity, not a regression. It is still silently wrong for tp.

**Fix:** make quantile placement the default for open B-splines. It is data-adaptive and has no knob. Optionally, refuse a tp fit when the basis is empty over most of the data mass.

**Files:** `crates/gam-terms/src/term_builder.rs`.

### F3: An absolute sd threshold rejects small-scale responses (bug, med, size S)
**Reproducer:** `y = (sin(6x) + noise) * 1e-12`, then `gamfit.fit(..., "y ~ s(x)")`. This fails with `InvalidConfigurationError: Gaussian response 'y' is effectively constant (sample sd ~ 7.668e-13 <= 1e-10)`. The same data times 1e12 fits identically, and pyGAM fits the 1e-12 data correctly.

**Root cause:** `GAUSSIAN_MIN_SAMPLE_SD = 1.0e-10` in `crates/gam-spec/src/lib.rs:1069`, applied at about L899 with the message at L1147. It is a magic absolute constant that breaks the scale equivariance gamfit otherwise has (x at 1e9 and 1e-9, and y at 1e12, give identical fits).

**Fix:** use a relative test, `sd <= c * eps_f64 * max|y|`, where the constant comes from floating-point resolution, or standardize y internally before the check.

**Files:** `crates/gam-spec/src/lib.rs`.

### F4: Perfect separation under s(x) hard-fails, and the automatic Firth rescue never fires (bug, high, size M)
**Reproducer:**
```
x ~ U(0,1) with n=200; y = (x > 0.5)
gamfit.fit({"x": x, "y": y}, "y ~ s(x)", family="binomial")
```
After 5 s this raises `IntegrationError: Fatal outer-objective evaluation failure (outer BFGS evaluation): ... PIRLS row geometry is not representable at row 37: canonical-logit inverse-link jet evaluated from eta=-745.13 produced 0.0`.
- `firth=True` fits sensibly (25 s).
- pyGAM fits with warnings.
- The #2273 contract (`crates/gam-models/src/fit_orchestration/perfect_binomial_separation_2273_tests.rs:165`) says the `smooth(x)` exact-separation fit must mint a model. Its fixture uses a wide x gap, not this layout.

**Root cause:**
- `crates/gam-solve/src/pirls/family_state.rs:39-53` requires `jet.d1 > 0.0`. Under separation the inner Newton step drives eta to about -745, where mu(1-mu) underflows to 0, and the row is refused as "unrepresentable".
- That variant, `PirlsRowGeometryUnrepresentable`, and its wrapper `OuterObjectiveEvaluationFailed` are both classified as not a retreat (`crates/gam-problem/src/estimation_error.rs:1190-1200`, `=> false`).
- So `firth_can_rescue` (`crates/gam-models/src/fit_orchestration/fit.rs:196-213`) returns false, and neither the outer retreat nor the Firth retry happens.
- The public error class is IntegrationError, not the existing `PerfectSeparationError`.

**Fix:**
- A logit row whose Fisher weight underflows at finite eta is a saturated, separated row, not a numerical fault. Raise `PerfectSeparationDetected` there, or compute the logit jet in log space so the weight e^eta/(1+e^eta)^2 is carried as a log-weight.
- Either way it then classifies as a retreat, the outer loop backs off, and the existing Firth rescue engages.
- Add this exact reproducer as a test.

**Files:** `crates/gam-solve/src/pirls/family_state.rs`, `crates/gam-problem/src/estimation_error.rs`, `crates/gam-models/src/fit_orchestration/fit.rs`.

### F5: `posterior_mean_standard_error` is a plug-in delta-method SE, not the posterior SD of the reported posterior mean (bug, med, size M)
**Evidence** (`probe_se2.py`): the reference is 200-node Gauss-Hermite over the fit's own eta posterior (`design_matrix().eta_gradient` with `covariance_conditional` and `covariance_smoothing_corrected`).

| case | reported posterior_mean | GH E[mu] | reported SE | GH SD[mu] |
|---|---|---|---|---|
| y ~ x separation (Firth) | 0.02258 | 0.02259 | **1.3e-22** | **0.145** |
| logit extrapolation, x=-100 | 8.2e-15 | 2.7e-15 to 3.2e-15 | **5e-176** | **4e-8** |
| logit extrapolation, x=+100 | **1.0000000000000484 (>1)** | 1.0 | 2e-174 | 4e-8 |
| ordinary logit | 0.19221 | 0.19221 (conditional) | 0.0503 | 0.051 (smoothing-corrected) |

The mean integrates correctly, but the SE is |mu'(eta_hat)|·SE(eta). That is off by 20 orders of magnitude exactly where the mean and the plug-in diverge. The intervals use the transform-eta construction, so they are fine. A caller using SE as the uncertainty of `posterior_mean` is badly misled.

There is also a minor issue: the posterior mean exceeds 1 by 5e-14 at a saturated row, which puts the point outside the support.

**Root cause:** `crates/gam-predict/src/lib.rs:1387-1399` in `enrich_posterior_mean_bounds` computes `mean_se = |d1(eta_hat)| * eta_se`. It is called from `crates/gam-predict/src/standard.rs:463`, and the generic path is `interval_policy.rs:803-878`, which uses the `delta_method_mean_se`.

**Fix:**
- Compute `sqrt(E[mu^2] - E[mu]^2)` with the same quadrature that already produces `E[mu]`, using the uncertainty covariance selected by `covariance_mode`. Report that as `posterior_mean_standard_error`.
- If a delta-method diagnostic is still wanted, expose it under a plug-in name.
- Clamp the quadrature output to the family's mean support.

**Files:** `crates/gam-predict/src/lib.rs`, `crates/gam-predict/src/standard.rs`, `crates/gam-predict/src/interval_policy.rs`, and the quadrature helper in `crates/gam-solve/src/quadrature.rs`.

### F6: Many-level random effects scale super-linearly and effectively hang at 500 levels (gap/perf, high, size L)
**Evidence:** `probe_levels.py L 3000 "y ~ s(x) + group(g)"`:

| levels L | fit time |
|---|---|
| 20 | 1.6 s |
| 50 | 3.0 s |
| 100 | 5.8 s |
| 200 | 18.3 s |
| 500 | over 180 s in the harness; a standalone run was still going after 6.5 min wall at write time |

The machine load was about 40, so absolute numbers are inflated, but the 100 to 200 step is about 3x. pyGAM fits the 500-level `f()` in 92 s, though its predict then fails on the object dtype.

**Root cause:** not isolated. It is consistent with dense p×p work per outer iteration on a group block that is diagonal (identity penalty) and should be handled as a sparse or blocked random-effect term. The speed auditor should confirm with a profile.

**Fix:** exploit the block-diagonal structure of `group()` (sparse Cholesky or Schur complement of the group block) in the P-IRLS and REML derivative paths.

**Files:** `crates/gam-solve` (reml, pirls), `crates/gam-terms` (group term). Size L.

### F7: Error class and location inconsistencies (bug, low, size S each)
- Non-integer Poisson y (2.5) passes front-door validation and fails inside the optimizer as `IntegrationError: Fatal outer-objective evaluation failure (outer seed screening): ... Poisson response must be a finite non-negative integer`. Negative Poisson y is caught up front, so integrality should be too, as `InvalidConfigurationError`.
- Negative weights raise `SchemaMismatchError` (`crates/gam-models/src/fit_orchestration/materialize/columns.rs:219-221`), which renders prediction-time help ("Verify the new data has the same columns and types as the training data", `crates/gam-models/src/fit_orchestration/error.rs:255`) at fit time. It should be a DataError or InvalidConfigurationError with weight-specific help.
- All-zero weights raise `IntegrationError` with deep internal text. It should be a front-door "no positive-weight rows" DataError.
- Predict-time NaN and unseen levels raise the base `GamError` instead of the existing `PredictInputError`.
- n=1 with y=[1.0]: auto-family picks binomial and reports "degenerate binomial" instead of "too few rows for s(x)".

### F8: stderr warning spam on fits that end certified (bug, low, size S)
With the default settings, dozens of `log::warn!` lines reach stderr and bypass Python warnings:
- `[GAM COST] -> P-IRLS INNER LOOP FAILED`: `crates/gam-solve/src/reml/gradient_hessian.rs:7058`
- `[INDEF-HESS]`: `crates/gam-solve/src/estimate/smoothing_correction.rs:1127`
- `[OUTER] ARC cost-stall STUCK`: `crates/gam-solve/src/rho_optimizer/bridges.rs:2712`, `:3035`
- `[BFGS Adaptive] Strong Wolfe failed`, `[HGB]` and `[thin-plate]` lines

These are optimizer-internal retreats the fit recovers from. The progress log default is Warn (`crates/gam-solve/src/progress_log.rs`).

**Fix:** demote recoverable-retreat messages to debug. Keep warn only for conditions that survive into the returned fit, which are then surfaced through Python `warnings` from the summary's convergence certificate.

### F9: huge p with small n is refused (gap, low, size M)
30 `s()` terms with n=60 fails with "need at least 62 rows". pyGAM fits, but garbage (edf 42, uncorrelated predictions). gamfit's refusal is clear. It is arguably too conservative because a double-penalized null space makes the penalized problem identifiable at p > n. A principled version would count identifiable (unpenalized) columns rather than the raw p. It is at worst a low gap, and nothing here should copy pyGAM.

## Already better or equal (gamfit)
- **Non-finite input:** NaN and inf are detected in x, y and weights with column and row given.
- **Degenerate families:** clear degenerate-response errors where pyGAM returns [0,0] CIs (all-zero Poisson, all-ones logistic, Gamma y=0).
- **Separation with linear terms:** the automatic Firth fit for `y ~ x` matches the independent Firth reference exactly.
- **Scale equivariance:** exact in x and for large y scales.
- **Small or noisy data:** tiny n, pure noise and duplicated x are handled sensibly, where pyGAM collapses (n=5) or overfits (noise, edf 11).
- **Degenerate or repeated covariates:** constant covariates are refused clearly, and collinear and duplicate smooths get certified fits.
- **Unseen levels:** an unseen level under `group()` gets a population-level prediction.
- **Input types:** float32, int and non-contiguous arrays all work. There were no crashes.

## Shared gap, noted but not a gamfit regression
With Cauchy or gross-outlier noise, both libraries give non-robust Gaussian fits. gamfit is less bad (RMSE 0.53/11.6/0.70 vs pyGAM 0.79/29.0/1.32 over 3 seeds). I did not verify whether a robust (Student-t) likelihood exists, so I report nothing on it here; see the families audit.
