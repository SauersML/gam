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
| " | Multi-term additive identity | test_multi_term_additive_identity_{numeric,with_factor} | MULTI_PDEP_RESULT |
| test_penalties.py | monotonic_inc/dec, convex, concave on hepatitis (diff of the sorted pred) | test_pg_shape_constraints.py (plug-in, posterior mean, pdep, interval bounds, posterior draws) | SHAPE_RESULT |
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
| test_GAMs.py | Each family fits: Linear, Logistic, Poisson, Gamma, InvGauss, GAM(gamma, link=inverse) | test_pg_families.py | FAMILIES_RESULT |
| test_gen_imgs.py | Plots render | test_pg_families::test_plot_* | PLOT_RESULT |
| test_GAM_methods.py | Prediction shapes, accuracy score, exposure/offset, loglik, large n, summary, k>n, intervals, sample, p-values, integer y, score, scale, expectiles, weights | test_pg_methods.py | METHODS_RESULT |
| test_core.py, test_datasets.py, test_gridsearch.py, test_links/distributions internals | repr/params/datasets/gridsearch/internals | not translated | see section 3 |

## 2. Confirmed findings

FINDINGS_BODY

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

COVERAGE_BODY

## 5. Proposed repo tests

All use synthetic data generated in the test (no vendored CSVs). Sizes: S < 80 lines, M 80-250, L > 250.

PROPOSED_BODY
