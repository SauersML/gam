# pyGAM test suite as an oracle for gamfit

Auditor axis: pyGAM's own tests. I catalogued what pyGAM asserts, translated the statistically
meaningful and SPEC-compatible behaviours into gamfit tests, ran them against the installed wheel,
and reviewed `/home/user/gam/tests` for coverage holes.

- Wheel under test: gamfit 0.1.267, installed in `$S/bvenv`.
- HEAD source was read for file:line evidence. No builds were run and nothing under `/home/user/gam` was modified.
- `S` = `/tmp/claude-0/-home-user-gam/02aeec89-32a7-52a0-8d71-90f383516996/scratchpad`
- `A` = `$S/audit/pygam_tests`. This holds the translated tests, `conftest.py`, the probes, pyGAM's CSVs in `data/` and pytest in `pylib/`.

To run the tests, work from outside the repo (inside it, the repo `gamfit/` shadows the wheel):

```
cd $A; PYTHONPATH=$A/pylib $S/bvenv/bin/python -m pytest -q -p no:cacheprovider -W ignore --tb=short test_pg_<x>.py
```

The machine load average was about 40 on 4 CPUs during this audit. The slowest tests were run as subsets or as fast probes; each case is marked below.

## 1. Catalogue: what pyGAM's tests assert, and how gamfit fares

| pyGAM test file | Behaviour asserted | gamfit translation | Result |
|---|---|---|---|
| test_partial_dependence.py | The pdep sum plus the intercept equals eta (univariate) | test_pg_partial_dependence::test_univariate_pdep_plus_intercept_equals_prediction | PASS (identity holds to 2.8e-14, dbg_pdep.py) |
| " | Default grid equals linspace over the training range; n_points; shapes | test_default_grid_equals_explicit_linspace, test_n_points_controls_grid | PASS |
| " | 2-D tensor pdep on a meshgrid | test_tensor_pdep_2d_grid | PASS |
| " | A bad term raises | test_bad_term_raises | PASS |
| " | width= intervals come from the model covariance | test_pdep_se_priced_off_published_covariance | PASS (covariance_source matches predict) |
| " / test_GAM_methods | pdep of a factor term `f()` | test_factor_term_partial_dependence_supported | **FAIL**, F4 |
| " | Multi-term additive identity | test_multi_term_additive_identity_{numeric,with_factor} | PASS. With a factor (wage, s(year)+s(age)+factor(edu)) and numeric-only (chicago n=1500, Poisson s(time)+s(tmpd); `probe_pdep_multi2.py`, max abs error 8.9e-16 against max abs eta 4.87). The 3-term chicago model (with te(pm10,o3)) could not be fitted by the wheel; see F15 |
| test_penalties.py | monotonic_inc/dec, convex, concave on hepatitis (diff of the sorted pred) | test_pg_shape_constraints.py (plug-in, posterior mean, pdep, interval bounds, posterior draws) | PASS for all 4 shapes (`probe_shape.py <shape>`, one process per shape). The plug-in, posterior mean, pdep and 80 posterior draws have 0 violations. For monotone shapes the band bounds are monotone too; the worst normalised difference is still in the right direction: +9.97e-4 (increasing) and 3.85e-4 (decreasing). Fits take 0.1-0.6 s CPU. `predict(interval=0.95)` is very slow; see F16 |
| test_terms.py | n_coefs of spline, factor and tensor blocks | test_pg_terms::test_block_width_* | spline PASS, tensor PASS. The factor block has L columns, not L-1; see F1 |
| " | Tensor invariance to covariate scaling (skipped in pyGAM) | test_tensor_invariance_to_covariate_rescaling | PASS (gamfit is better here) |
| " | The by-variable term equals the te-with-linear-margin term; a missing by raises | test_by_numeric_close_to_tensor_with_linear_margin, test_by_variable_missing_raises | PASS |
| " | Correct smoothing in tensors (x0*sin(x1)) | test_reml_tensor_explains_interaction | PASS |
| " | Tensor with per-margin constraints | test_tensor_margin_shape_constraint | **FAIL**, F6 |
| " | Tensor n_splines of length 1 raises | test_tensor_k_single_value_broadcasts | PASS (gamfit documents broadcasting instead, docs/formulas.md) |
| " | build_from_info round-trip | test_save_load_roundtrip | PASS |
| test_GAM_params.py | Intercept-only model == mean | test_intercept_only_is_mean | PASS |
| " | Linear term == OLS | test_linear_term_is_ols | PASS |
| " | fit_intercept=False | test_no_intercept_formula[0 + ..., - 1] | **FAIL**, F5 |
| " | Cyclic basis is periodic; cyclic fits worse on non-cyclic data | test_cyclic_is_periodic, test_cyclic_worse_than_free_smooth_on_aperiodic_data | PASS |
| test_utils.py | check_X/check_y reject NaN, inf, wrong length and out-of-domain y | test_pg_validation (15 tests) | PASS |
| " | check_X rejects a non-numeric feature | test_smooth_of_string_column_raises | **FAIL**, F2 |
| " | Unseen categorical level at predict raises | test_unseen_factor_level_at_predict_raises | PASS |
| " | Pandas input accepted | test_pg_validation::test_pandas_inputs | **FAIL** without pyarrow; see F8 |
| test_GAMs.py | Each family fits: Linear, Logistic, Poisson, Gamma, InvGauss, GAM(gamma, link=inverse) | test_pg_families.py | Gaussian/mcycle, Poisson/coal and Gamma/trees PASS (certified). inverse_gaussian, gamma+inverse and gamma+identity FAIL (F11). The binomial/default (n=10000) run: at n=10000 it did not finish inside the 2400 s timeout. At n=2000 on raw covariates it raises IntegrationError (F17). With standardised covariates it is certified but slow (F18), and the student effect is shrunk to zero (F1) |
| test_gen_imgs.py | Plots render | test_pg_families::test_plot_* | PASS for `m.plot(d)` on mcycle (Gaussian) and coal (Poisson), rendered to PNG under Agg. The wage (factor) and chicago te plots were not reached under the load; the fast subset deselected them |
| test_GAM_methods.py | Prediction shapes, accuracy score, exposure/offset, loglik, large n, summary, k>n, intervals, sample, p-values, integer y, score, scale, expectiles, weights | test_pg_methods.py | 14 PASS: shape, non-integer offset loglik, summary rows, k>n, interval nesting, noise p>0.05, p/edf invariant to y*1e6, integer binary y, score <= 1, loglik ordering, Poisson weights == duplication, Gaussian weight rescale, Poisson zero-weight == deletion, posterior sample shapes (after fixing my test to use `PosteriorPredictive.eta`/`.mean`). 8 FAIL: offset scaling (F9), scale (F7), 4 x expectile_tau validation (F10), expectile ordering (F10; the test uses `expectile_tau` without `family=`, which is the F10 trap itself), Gaussian zero-weight == deletion at rtol 1e-4 (F13). Deselected for time: large n, observation-interval calibration, classifier accuracy (`out_methods.txt`) |
| test_core.py, test_datasets.py, test_gridsearch.py, test_links/distributions internals | repr/params/datasets/gridsearch/internals | not translated | see section 3 |

## 2. Confirmed findings

Each finding has a kind (bug | gap | pyGAM-slop-to-avoid | already-better) and a severity. The evidence was reproduced against the wheel, and the HEAD file:line references were read from source. The probe scripts are in `$A`.

### F1. `factor(g)` and a bare categorical `+ g` are penalized random effects, not fixed factors [bug, high]
- **Evidence:** `probe_factor.py`. For 5 levels with 3 rows each, `y ~ factor(g)`, `y ~ g` and `y ~ group(g)` give the identical block `TermBlock(kind='random_effect', width=5)`, the same lambda 0.3398 and edf 4.59. Predictions are shrunk: level `e` has a group mean of 7.474 but a prediction of 6.991.
- **HEAD:** `crates/gam-terms/src/term_builder.rs:470-481` sets `penalized: true, drop_first_level: false` for a bare categorical. `term_builder.rs:522-543` does the same for `factor()`. `formula_dsl.rs:2938-2959` maps `"group"|"re"|"factor"` to the same `ParsedTerm::RandomEffect`; only the unseen-level policy differs.
- **Contract violated:** `docs/formulas.md:117-122` ("`factor(g)` ... is a **fixed** categorical factor — the same fixed main effect as a bare `+ g`"). `tests/bug_hunt_categorical_reference_level_fit_invariance_test.py` asserts in its docstring that the block is "unpenalized treatment-coded". It passes only because a full one-hot ridge is also relabelling-invariant, so no test locks the fixed-effect contract.
- **Real-data impact (pyGAM's canonical LogisticGAM example):** `probe_default_student.py` fits binomial `default` data with n=2000 and z-scored covariates.
  - `y ~ factor(student) + s(balance) + s(income)` gives a student log-odds effect of **-2.7e-5**, with lambda_student = 9.5e4.
  - The same model with the unpenalized `linear(stud01)` gives **-0.413**.
  - A 2-level factor treated as a random effect has an unidentifiable variance, and REML shrinks the effect away. The canonical fixed covariate silently vanishes.
- **No escape hatch exists.** `linear(g)` on a string column becomes an ordinal slope (see F2). `factor(g, double_penalty=false)` is silently accepted and changes nothing (see F3). `C(g)` is an unknown term.
- **pyGAM oracle:** `f()` terms are fixed dummy blocks. pyGAM's `test_terms::test_n_coefs` expects L columns there because pyGAM keeps all L dummies under a tiny ridge; L-1 treatment columns are the mgcv/R convention.
- **Fix:** give `factor()` and a bare categorical `penalized: false, drop_first_level: true` (treatment contrasts, L-1 columns, no lambda). Keep `group()`/`re()` penalized. Add a test that the fixed-factor fit equals the per-level OLS means exactly (for `y ~ factor(g)` alone), has L-1 columns and has no lambda.
- **Files:** `crates/gam-terms/src/term_builder.rs`, plus a new test. **Size:** M. This changes saved-model semantics; serde has a default for `penalized`.

### F2. A string column inside `s()` or `linear()` is silently coerced to arbitrary integer level codes [bug, high]
- **Evidence:** `probe_string.py`. The levels are zeta, alpha, mid and beta, with true effects 0, 5, 0, 5.
  - `y ~ linear(g)` fits a slope over the internal codes. Predictions are ordered zeta < mid < beta < alpha and the RMSE is 1.20 against a noise sd of 0.3.
  - `y ~ s(g)` builds a B-spline on the codes.
  - `test_pg_terms::test_smooth_of_string_column_raises` fails with DID NOT RAISE.
- **HEAD:** the explicit `linear()` branch at `term_builder.rs:432-446` pushes a numeric `LinearTermSpec` with no column-kind check. Compare the `bounded()` branch at `term_builder.rs:496-500`, which does reject categorical columns. The `Smooth` branch only kind-checks the `by=` column.
- **pyGAM oracle:** `test_utils::test_check_X_not_int_not_float` makes check_X raise.
- **Fix:** reject `ColumnKindTag::Categorical` for every numeric-axis term (`linear`, `s`, `te`, `cyclic`, `bounded`, `thinplate`/`matern` coordinates). Point the error at `factor()`/`group()`.
- **Files:** `term_builder.rs`, plus a test. **Size:** S.

### F3. Unknown options on `factor()`/`group()` are silently accepted [bug, med]
- **Evidence:** `probe_string.py`. `factor(g, foo=1)`, `group(g, bogus=3)` and `factor(g, double_penalty=false)` all fit without error. docs/formulas.md says unknown options are rejected.
- **HEAD:** the `"group" | "re" | "factor"` arm of `formula_dsl.rs:2938-2959` never inspects `options`.
- **Fix:** strict option validation in that arm (an allow-list, empty for now).
- **Files:** `crates/gam-terms/src/inference/formula_dsl.rs`, plus a test. **Size:** S.

### F4. `partial_dependence` does not support factor or random-effect terms [gap, med]
- **Evidence:** `test_pg_partial_dependence::test_factor_term_partial_dependence_supported` fails with `ValueError: cannot infer a 1D sweep axis from term 'edu'` in the wheel.
- **HEAD:** `crates/gam-inference/src/partial_dependence.rs` `resolve_term` (about lines 282-360) resolves only `linear_terms` and `smooth_terms`; `random_effect_terms` fall through to "unknown term".
- **pyGAM oracle:** supports pdep on `f()` terms (bar per level with CI).
- **Fix:** add a categorical branch that returns one row per level (the contribution and its SE from the published covariance, with a level-name grid).
- **Files:** `partial_dependence.rs`, `crates/gam-pyffi/src/manifold/geometry_ffi.rs` (the grid dtype), `gamfit/_model.py` docs, plus a test. **Size:** M.

### F5. No intercept-free models [gap, med]
- **Evidence:** `test_pg_terms::test_no_intercept_formula`:
  - `y ~ 0 + linear(x)` raises `FormulaError: formula terms '0'/'-1' (intercept removal) are not supported yet`.
  - `y ~ linear(x) - 1` raises `unsupported top-level RHS term`.
- **pyGAM oracle:** `GAM(fit_intercept=False)`, `test_GAM_params::test_fit_intercept`. Physical models through the origin need this (dose-response with f(0)=0, and additive decompositions where an offset carries the level).
- **Fix:** parse `0 +` / `- 1` and drop the intercept column. Smooth identifiability then must not absorb the constant; keep the sum-to-zero constraint only when an intercept exists, or document which convention applies.
- **Files:** `formula_dsl.rs`, design construction, plus a test. **Size:** M.

### F6. No shape constraints on tensor-product margins [gap, med]
- **Evidence:** `y ~ te(tmpd, o3, shape=[monotone_increasing, none])` raises `InvalidConfigurationError: unknown shape constraint "[monotone_increasing, none]"`. docs/formulas.md:405-425 lists no `shape` option for `te`.
- **pyGAM oracle:** `test_terms::test_tensor_with_constraints`. pyGAM's version is a heuristic penalty; gamfit would need exact cone constraints on the margin to stay "penalties on the function" and be certifiable.
- **Fix:** support a per-margin shape via the Kronecker structure of the monotone reparameterisation (a cumulative-sum basis on the constrained margin).
- **Files:** `crates/gam-terms` tensor construction and the constraint layer, plus tests. **Size:** L.

### F7. No Gaussian scale / dispersion estimate on `Summary` [gap, low-med]
- **Evidence:** `test_pg_methods::test_scale_estimate`. The Summary fields are basis_checks, coefficient_se_source, coefficients, ..., reml_score and smooth_terms; none is scale, sigma, dispersion or phi. `extras` is empty. HEAD `gamfit/_summary.py:304-363` has no such field either.
- **Comparison:** pyGAM `statistics_['scale']` and mgcv `sig2`/`scale` expose it. Users need it for residual-sd reporting and for sanity checks of the REML fit.
- **Fix:** publish the REML-profiled phi (and its name) in Summary and `to_dict`.
- **Files:** `gamfit/_summary.py`, `crates/gam-pyffi` summary payload, plus a test. **Size:** S.

### F8. Fitting from a pandas DataFrame raises ImportError when pyarrow is absent (pandas >= 3) [bug, med]
- **Evidence:** `test_pg_validation::test_pandas_inputs`. `gamfit.fit(pd.DataFrame(...), "y ~ s(x)")` raises `ImportError: Import pyarrow failed`. Environment: pandas 3.0.6, no pyarrow.
- **HEAD:** `gamfit/_tables.py:108` sees `hasattr(data, "__arrow_c_stream__")` and calls the arrow C stream with no fallback. pandas 3 defines the dunder but needs pyarrow to execute it. `pyproject.toml` lists pyarrow only in the extras.
- **Fix:** catch ImportError around the arrow-stream call and fall back to the column-dict path, or make pyarrow a core dependency.
- **Files:** `gamfit/_tables.py`, plus a test. **Size:** S.

### F9. An all-zero (or 0/1-valued) offset column at fit is typed "binary", so predict with any other offset fails [bug, high]
- **Evidence:** `probe_offset_expectile.py` and `test_pg_methods::test_poisson_offset_scales_prediction`.
  - Fit Poisson with `offset="off"`, where `off` is all zeros (the natural "no exposure yet" baseline). `predict` with `off = log 2` raises `GamError: column 'off' is binary in schema but row 1 has value 0.6931471805599453; expected 0 or 1`.
  - The same happens with an `off` in {0, 1} at fit and 0.5 at predict.
  - With a non-binary offset at fit, the offset shifts correctly: the ratio is exactly 3.0 for log 3.
- **HEAD:** `crates/gam-data/src/lib.rs:2381-2394` applies the Binary-kind check to every schema column, and the offset column's kind is inferred from its fit-time values like any covariate.
- **pyGAM oracle:** `test_PoissonGAM_exposure` / exposure doubling.
- **Fix:** give the offset and weights columns a fixed Continuous kind at schema inference; never infer them as Binary or Categorical. Add a test that fits with a zero offset and predicts with log 2, so the mean doubles.
- **Files:** `crates/gam-data/src/lib.rs` (schema inference for role columns) or the pyffi schema builder, plus a test. **Size:** S.

### F10. `expectile_tau=` is silently ignored unless `family="expectile"`, including out-of-range values [bug, med]
- **Evidence:** `probe_offset_expectile.py`.
  - `fit(d, "y ~ s(x)", expectile_tau=0.9)` gives predictions bit-identical to the Gaussian fit (max |diff| 0.0). The same holds with `family="gaussian"`.
  - `expectile_tau=1.5` also fits without error.
  - Only `family="expectile", expectile_tau=0.9` (or `family="expectile(0.9)"`) fits the expectile (all grid points lie above the mean). There, τ=1.5 is correctly rejected.
- **HEAD:** `crates/gam-models/src/fit_orchestration/entry.rs:216-222`. `expectile_tau_for_config` returns `Ok(None)` when the family string is not `expectile...`, so the field is dropped without validation.
- **pyGAM oracle:** `ExpectileGAM(expectile=...)` validates `0 < expectile < 1` (`test_expectiles`).
- **Fix:** if `expectile_tau` is set, either imply `family="expectile"` (when the family is auto) or raise when the family is anything else. Validate τ in (0, 1) always.
- **Files:** `entry.rs`, plus a test. **Size:** S.

### F11. Missing families and links that pyGAM tests [gap, med]
- **Evidence:** `test_pg_families`:
  - `family="inverse_gaussian"` is an unknown family.
  - `family="gamma", link="inverse"` (the canonical Gamma link) is unsupported.
  - `link="identity"` for Gamma is unsupported (gam-spec `lib.rs:153,605`).
- **pyGAM oracle:** InvGaussGAM and `GAM(distribution='gamma', link='inverse')`.
- **Fix:** add the inverse-Gaussian family (V(mu)=mu^3, REML with a profiled phi) and allow the inverse/identity links for Gamma, with domain checks.
- **Files:** `crates/gam-spec`, family implementations, plus tests. **Size:** M.

### F12. No deviance or Pearson residuals [gap, low]
- **Evidence:** `Model` has no `residuals` method. `Diagnostics.residuals` (`gamfit/_diagnostics.py:43,64`) is `observed - predicted` only. A grep for `deviance_resid|pearson` in `gamfit/` and `crates/gam-pyffi/src` finds nothing.
- **pyGAM oracle:** `test_GAM_methods::test_deviance_residuals` (the sum of squared deviance residuals equals the deviance).
- **Fix:** `Model.residuals(data, kind="deviance"|"pearson"|"response"|"working")`, computed in Rust from the family. Test that sum(dev_res^2) equals `summary().deviance`.
- **Files:** `gamfit/_model.py`, `crates/gam-pyffi`, plus a test. **Size:** S.

### F13. Zero-weight rows are y-inert but their covariate values still move the fit slightly [gap, low]
- **Evidence:** `probe_zero_weight.py`.
  - Changing y on the w=0 rows changes nothing (max |diff| 0.0, identical lambda), so there is no likelihood leak.
  - Compared with deleting those rows, lambda differs by about 0.01-1%. Predictions differ by up to 1e-4 (Gaussian) and 7e-4 (Poisson, on the mean scale).
  - `test_zero_weights_equal_row_deletion[gaussian]` fails at rtol=1e-4.
  - The likely cause is that basis centering and scaling use all rows. mgcv behaves similarly, so this is a design choice.
- **Fix:** compute the identifiability centering and penalty normalisation from positive-weight rows, or document that w=0 rows still enter the basis construction. `tests/bug_hunt_gaussian_reml_zero_weight_rows_count_in_residual_dof_test.py` locks exact equivalence only for the low-level `gaussian_reml_fit`.
- **Size:** S.

### F14. Solver chatter on stderr during successful fits [bug, low]
- **Evidence:** successful Poisson `te` fits print "[GAM COST] -> P-IRLS INNER LOOP FAILED ... Hessian matrix is not positive definite", "[OUTER] ARC cost-stall", "[BFGS Adaptive] Strong Wolfe failed ... Falling back" and "[HGB] target_mse below mandatory floors" (seen in `out_terms.txt` and `out_methods.txt`). The fits end certified. Users will read "FAILED" as failure.
- **Fix:** route these through the `log` crate at debug level.
- **Size:** S.

### F15. pyGAM's canonical chicago model fails in the wheel with a cubature IntegrationError [bug in 0.1.267; the fallback is in HEAD source; regression lock missing, med]
- **Evidence:** `probe_cubature.py "1500|y ~ s(time) + s(tmpd) + te(pm10, o3)"` and `"1000|..."` (Poisson, chicago rows with NaNs dropped) raise `IntegrationError: smoothing cubature has no positive-width proposal in the resolved domain` after 59 s and 37 s. `te(pm10, o3)` alone and `s(time) + s(tmpd)` alone both fit and are certified.
- **HEAD:** the error is raised inside the node-calibration closure at `crates/gam-solve/src/reml/eval.rs:1379`. At `eval.rs:1412-1419`, `match calibrated_nodes { Err(error) => return self.finalize_smoothing_outcome(first_order_numerical(..., "smoothing cubature could not calibrate its nodes: {error}")) }` now falls back to the certified first-order correction.
- **The wheel predates that fallback:** `grep -c "could not calibrate its nodes" gamfit/_rust.abi3.so` gives 0, while `"no positive-width proposal"` gives 1. I did not build, so I have not verified that HEAD fits this model end to end.
- **Coverage:** no repo test exercises this path (a grep for `could not calibrate|positive-width proposal` in `tests/` finds nothing).
- **Fix:** ship the fallback, rebuilding the wheel. Add a regression test with a synthetic 3-term Poisson model (two `s()` terms plus a `te()` of two correlated covariates, n about 1000) that must fit, be certified, and report a first-order-fallback reason in `covariance_source` when the cubature refuses.
- **Files:** a new `tests/pygam_oracle_families_test.py` case. **Size:** S.

### F16. `predict(interval=...)` on a shape-constrained model is 100-600x slower than sampling [perf, med]
- **Evidence:** `probe_shape.py <shape>` on hepatitis (n=83), with a 200-point grid:

  | Shape | predict(interval=0.95) CPU | predict wall | fit CPU |
  |---|---|---|---|
  | concave | 62.5 s | 404 s | 0.6 s |
  | convex | 16.3 s | 191 s | — |
  | monotone_decreasing | 19.0 s | 212 s | — |

  `m.sample(d, samples=80).predict_draws(g)` takes 0.0-0.1 s for the same model.
- **Covariance-source inconsistency:** monotone_decreasing reports `covariance_source="conditional"`, while the other three report `"smoothing-corrected"`. The smoothing correction was declined for that fit without the reason being surfaced in the probe output.
- **Fix:** profile the constrained-posterior interval path. It appears to redo per-point work that a single draw batch or a single constrained-covariance factorisation would give. Surface why the smoothing correction was declined.
- **Test:** a CPU-time-free guard, such as the number of inner solves per predict call, if one is exposed; otherwise leave it to the benchmark. **Size:** M.

### F17. pyGAM's canonical LogisticGAM example (`default`: student + balance + income) dies after a certified outer minimum [bug, high]
- **Evidence:** `probe_binomial_default.py "y ~ factor(student) + s(balance) + s(income)" 2000`, with family="binomial", raises:

  ```
  IntegrationError: Invalid input: exact smoothing-corrected covariance unavailable: OuterHessianInverse { error: "rho Hessian has negative curvature -2.989e-7 below the outer certificate's own bar 1.227e-7 ... the outer loop certified this point as a minimum ... so this is a genuine contradiction" }
  ```

  The full pyGAM test (n=10000) did not finish within the 2400 s pytest timeout.
- **HEAD:** `crates/gam-solve/src/estimate/optimizer.rs:3700-3736`. `SmoothingCorrectionOutcome::Unavailable` ships the plug-in covariance only when the fit is rail-certified or the outer Hessian is structurally non-analytic. Otherwise it returns `Err`, and the whole certified fit is lost over the covariance upgrade. The threshold gate is in `crates/gam-solve/src/estimate/smoothing_correction.rs:~851`. The surrounding comments (#2748, #2612, #1561) show this family of refusals has recurred on binomial fits (`quality_vs_sklearn_binomial_logit`, `quality_vs_inla_binomial_smooth_probability`).
- **Standardised covariates:** the same model with z-scored balance and income fits and is certified (29.0 s CPU, `probe_binomial_default_std.py`). So the refusal is triggered by covariate scale alone, and an affine reparameterisation of the inputs decides whether the user gets a fit. This is also a scale-invariance break.
- **Fix:** follow the "refusing the upgrade is not refusing the fit" principle already used for the cubature (`eval.rs:1412`, the F15 fallback). When the outer loop certified a minimum and only the rho-Hessian inverse for the correction is contradicted, ship the certified fit with the conditional (plug-in) covariance. Surface a typed `SmoothingCorrectionAbsence::CurvatureContradicted { detail }` in `covariance_source`, instead of Err. Add a regression test that is a synthetic rare-event (about 3% positives) binomial with a 2-level factor and 2 smooths at n=2000, which must return a certified fit.
- **Files:** `optimizer.rs`, `model_types.rs` (the absence enum), plus `tests/pygam_oracle_families_test.py`. **Size:** M.

### F18. Binomial fits on the `default` data are extremely slow [perf, high]
- **Evidence (CPU time, not wall; load average about 30):**
  - `y ~ s(balance) + s(income)`, n=2000: 63.7 s CPU (330 s wall), certified, edf 2.89. The stderr shows repeated `P-IRLS INNER LOOP FAILED` (Hessian not positive definite; "did not converge ... Last gradient norm was inf").
  - `y ~ s(balance)`, n=10000: 248.5 s CPU (778 s wall), certified, edf 2.00 (a near-linear fit). Standardised `s(balance)+s(income)` at n=2000: 69.9 s CPU, so scale is not the cause of the slowness.
  - On the same loaded machine, pyGAM `LogisticGAM(f(0)+s(1)+s(2))` fits the full n=10000 three-term model in 15.8 s CPU (40.6 s wall), with accuracy 0.974. gamfit is slower at one fifth of the n with one term fewer.
  - The data has about 3.3% positives, and `balance` nearly separates the classes.
- **Fix:** investigate why the inner P-IRLS produces inf gradients on a near-separated logit. A likely cause is an unguarded `eta` step or missing step-halving on deviance increase in the trial rho. Add a synthetic rare-event benchmark to the speed suite.
- **Size:** M.

### Already better than pyGAM (keep; add regression locks where missing)
- Every fit carries a convergence certificate (`summary().convergence["certified"]`). pyGAM reports stats after `max_iter=1`.
- Intervals are posterior-mean intervals priced off the smoothing-parameter-corrected covariance (`covariance_source`), and pdep SEs use the same covariance.
- Shape constraints hold for posterior draws, not only the point estimate.
- Tensor fits are invariant to covariate rescaling, which pyGAM itself skips as failing.
- Duplicate terms `s(x) + s(x)` are rejected rather than fitted as an unidentifiable pair.
- A string y or an (n,1) y is rejected rather than cast.
- An unseen level of a fixed factor raises, while `group()` shrinks it to the population mean.
- A B-spline smooth extrapolates linearly (second difference exactly 0).
- Poisson weights are exactly equivalent to row duplication, and a Gaussian weight rescale is invariant (mean and SE), as documented.
- p-values and edf are invariant to rescaling y by 1e6. A pure-noise covariate gets p > 0.05 and a true signal gets p < 1e-6.

## 3. pyGAM behaviour deliberately not replicated

| pyGAM test(s) | Why not |
|---|---|
| test_gridsearch.py (13 tests: gridsearch over lam/n_splines, GCV/UBRE objective, return_scores, keep_best, objective='auto') | SPEC forbids GCV/UBRE, grid search and derivative-free search; gamfit selects lambda by REML/LAML with an outer Newton/ARC step. |
| test_penalties `test_single_spline_penalty`, `test_wrap_penalty`, callable penalties, `lam` validation, fixed-`lam` fits | The user never supplies lambda (no magic knobs). Penalty construction is internal and tested in Rust. |
| test_GAM_methods `test_compute_stats_even_if_not_enough_iters` (max_iter=1, stats still computed) | SPEC: only converged fits. gamfit reports a convergence certificate instead of reporting stats for an unconverged fit. |
| test_GAM_methods `test_is_fitted_*` (6 tests), `test_set_params_*`, phony params, `n_splines` "easy plural", non-int `n_splines` coercion | gamfit has no unfitted estimator object (`fit()` returns a Model). Silently coercing or accepting phony parameters is slop; gamfit rejects them. |
| test_core `nice_repr`, `Core` | Internals of pyGAM's repr. |
| test_GAM_methods `test_summary_returns_12_lines`, 24-line summary, `sig_codes` | Line counts and significance stars are cosmetic. gamfit's Summary is a structured record. |
| test_GAM_methods `test_fit_quantile*` (ExpectileGAM.fit_quantile via bisection on tau) | A derivative-free bisection over a hand-supplied bracket; also an expectile is not a quantile. gamfit exposes `expectile_tau` directly. |
| test_GAM_methods `test_sample` with bootstrap smoothing draws (`n_bootstraps`) | pyGAM refits bootstrap replicates to integrate over lambda. gamfit uses the smoothing-parameter-corrected covariance (`covariance_source`), which is already better. |
| test_GAM_methods `test_prediction_interval_known_scale` with user-supplied `scale=` | A hand-supplied dispersion is a magic knob. gamfit estimates it by REML (but see F7: it is not surfaced). |
| test_GAM_methods `test_conf_intervals_quantiles_width_interchangable` | Tests pyGAM's width/quantiles argument aliasing, which is API trivia. |
| test_utils `check_iterable_depth`, SKSPIMPORT, meshgrid-shape helpers | pyGAM internals. |
| test_penalties `test_constraints_and_tensor` / composed inc+dec on one term | Composing contradictory constraints is meaningless (the result is constant). gamfit accepts one shape per term. |
| test_utils `catch_chol_pos_def` with a huge manual lam | Needs a hand-supplied lambda. gamfit's REML never visits that regime without a certificate. |
| test_datasets.py (12 tests) | gamfit ships no datasets (SPEC: never vendor). pyGAM's own wheel also lacks the CSVs, so its test_datasets fails from the wheel. |
| pyGAM casting a string y to float / accepting a 2-D (n,1) y | gamfit rejects both. Strictness is intentional and already better. |

## 4. Coverage holes in /home/user/gam/tests

I checked each hole by grepping `tests/` for the behaviour; the relevant files are named.

| Behaviour | Status in /home/user/gam/tests | Hole |
|---|---|---|
| Additive identity eta == intercept + sum_j pdep_j(x_ij) | `test_python_api.py:1897,2361` and `test_partial_dependence_saved_spec.py` check only shapes and finiteness | **Missing.** No test ties pdep to the linear predictor. It holds today (8.9e-16) but is unlocked |
| pdep of a shape-constrained term obeys the shape | none | **Missing** (it holds today) |
| pdep of factor / random-effect terms | `bug_hunt_partial_dependence_crashes_when_model_has_a_string_factor_test.py` covers only a smooth pdep with a factor elsewhere in the model | **Missing** (F4) |
| Predict-time offset scaling (exposure doubling) | `bug_hunt_posterior_predictive_draws_drop_model_offset_test.py` uses a sizeable non-binary offset | **Missing:** no test with a zero or 0/1 offset at fit, which is how F9 slipped through |
| Zero weights == row deletion, public formula path | `bug_hunt_gaussian_reml_zero_weight_rows_count_in_residual_dof_test.py` covers the low-level `gaussian_reml_fit` only | **Missing** for `gamfit.fit(weights=)` (F13) |
| Weights == duplication (Poisson), Gaussian rescale invariance | `weighted_gaussian_is_prior_weight_not_frequency_weight_test.py` | Covered |
| Shape guarantees for posterior draws | `test_posterior_monotone_shape_constraint.py` covers increasing, decreasing and convex | **Concave draws missing.** Concave point estimates are covered (`test_python_api.py:2242-2315`) |
| Fixed-factor contract (L-1 unpenalized columns, equal to group means) | `bug_hunt_categorical_reference_level_fit_invariance_test.py` checks only relabelling invariance, which a ridge also satisfies. Its docstring misdescribes the block | **Missing** (F1) |
| Categorical column inside s()/linear()/te() rejected | none | **Missing** (F2) |
| Unknown options on factor()/group() rejected | none | **Missing** (F3) |
| `expectile_tau` without `family="expectile"` | `bug_hunt_expectile_family_unreachable_from_public_interfaces_test.py` covers the reachable spellings only | **Missing** (F10) |
| Multi-term Poisson with s + s + te (pyGAM chicago) | none | **Missing** (F15) |
| Cyclic periodicity | `bug_hunt_period_declaration...` | Covered |
| Scale/dispersion surfaced on Summary | none | Feature absent (F7) |
| pandas input without pyarrow | none | **Missing** (F8) |

## 5. Proposed repo tests

All use synthetic data generated in the test (no vendored CSVs). Sizes: S < 80 lines, M 80-250, L > 250.

Source translations to port are in `$A/test_pg_*.py`. Replace the pyGAM dataset fixtures with seeded synthetic generators of the same shape.

| Proposed file | Contents | Locks | Size |
|---|---|---|---|
| `tests/pygam_oracle_partial_dependence_test.py` | Additive identity (Gaussian with a factor; Poisson with 2 smooths); default grid == linspace; 2-D tensor meshgrid; pdep SE uses the model's covariance_source; factor pdep (xfail is forbidden, so land it with F4) | the pdep identity, F4 | S |
| `tests/pygam_oracle_shape_constraints_test.py` | For each of 4 shapes: plug-in, posterior mean, pdep and 80 draws obey the shape; monotone band bounds are monotone | concave draws, shaped pdep | M |
| `tests/pygam_oracle_inference_test.py` | Interval nesting; p-value of a noise covariate > 0.05 and of the signal < 1e-6; p/edf invariant to y*1e6; loglik ordering; Summary exposes scale (with F7) | F7 | M |
| `tests/pygam_oracle_terms_test.py` | Fixed factor == group means with L-1 columns and no lambda; string column in s()/linear() raises; unknown factor()/group() options raise; `0 +` / `-1` (with F5); tensor rescale invariance; by== te with a linear margin | F1, F2, F3, F5 | M |
| `tests/pygam_oracle_validation_test.py` | NaN, inf, length mismatch and out-of-domain y rejected; unseen level raises; a DataFrame fits without pyarrow (monkeypatch the ImportError) | F8 | S |
| `tests/pygam_oracle_families_test.py` | Gaussian, binomial, Poisson and Gamma fit certified; inverse-Gaussian and Gamma inverse/identity links (with F11); synthetic 3-term Poisson s+s+te fits certified (F15) | F11, F15 | S |
| `tests/pygam_oracle_weights_offsets_test.py` | Zero offset at fit then log 2 at predict doubles the mean; 0/1 offset column; Poisson weights == duplication; zero weights == deletion via the public path (with F13 at a documented tolerance, or exactly once fixed); `expectile_tau` without family raises or implies expectile, and tau outside (0,1) raises | F9, F10, F13 | S |
