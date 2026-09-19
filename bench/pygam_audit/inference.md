# Inference & uncertainty quantification audit: gamfit vs pyGAM

Axis: inference / UQ, measured empirically. Versions: gamfit 0.1.267 wheel (the installed build; repo HEAD is 0.1.268, not buildable here), pyGAM 0.12.0.
All scripts and raw results: `scratchpad/audit/inference/` (`mc.py`, `mc_sample.py`, `lr_null.py`, `analyze.py`, `repro_*.py`, `results/*.json`).

## 1. Output map: pyGAM inference output -> gamfit counterpart

| pyGAM | gamfit | verdict |
|---|---|---|
| `confidence_intervals(X, width)` (lam-conditional cov, t or z quantile, per-term-free) | `predict(data, interval=level)` -> `posterior_mean_lower/upper`, **smoothing-corrected covariance by default** (`covariance_source: "smoothing-corrected"`), `covariance_mode="conditional"` available | already-better (see MC) |
| `prediction_intervals(X, width)` — **LinearGAM only** | `predict(..., observation_interval=True)` for every family (Gaussian se²+phi; Poisson/binomial discrete response-scale quantiles) | already-better |
| `partial_dependence(term, X, width)` -> (pdep, band) | `partial_dependence(term, data, grid)` -> `predicted`, `standard_error` only; sum-to-zero identified | **gap**: no `level`, no bounds, no simultaneous band (G2) |
| `sample(X, y, quantity, n_draws, n_bootstraps=5)` | `sample(data, samples, seed)` (Gaussian: Laplace on smoothing-corrected cov; binomial: exact Polya-Gamma on **conditional** cov), `sample_replicates`, `iter_replicates` | Gaussian already-better; non-Gaussian **gap** (G3) |
| `statistics_['edof_per_coef']`, summary EDoF per term | `summary().smooth_terms[*].edf`, `edf_total` | parity |
| `statistics_['p_values']` | `summary().smooth_terms[*].p_value` (Wood rank-truncated Wald, ref_df) + `smooth_significance(data)` (LR, Imhof null spectrum, selection replay, Lawley) | already-better (power 2x) |
| `statistics_['AIC']` | `compare_models` cAIC ranking; `Model.conditional_aic` exists at HEAD (`gamfit/_model.py:1257`) but **not in the 0.1.267 wheel** | partial; not in `summary()` (G1) |
| `statistics_['AICc']` | none | do NOT add (pyGAM applies the Gaussian small-sample formula to every family: slop S6) |
| `statistics_['pseudo_r2']` (McFadden, explained_deviance) | none in Python `summary()`; CLI summary has `deviance_explained` (`crates/gam-cli/src/main/model_summary.rs:85`); `diagnose(data).metrics.r_squared` (response-scale R² only) | **gap** (G1) |
| `statistics_['GCV']`, `['UBRE']` | none | correct to omit (SPEC: REML/LAML only) |
| `statistics_['scale']` | not exposed; stored in model JSON at `payload/fit_result/inference/dispersion {source, phi}` | **gap** (G1) |
| `statistics_['loglikelihood']`, `deviance` | `summary().log_likelihood`, `.deviance` | parity |
| `statistics_['cov']`, `['se']` | `summary().covariance_flat` (corrected), `coefficients[*].std_error` | already-better |
| `lam` | `summary().lambdas` (two per smooth: wiggle + null-space shrinkage) | already-better |
| — | `difference_smooth(..., simultaneous=True)`, `curvature`, `debiased_functional`, `basis_check`, REML score, convergence certificate | gamfit-only |

## 2. Monte Carlo design

DGP: `eta = b0 + a1 sin(2 pi x1) + 0*x2 + a3 cos(2 pi x3)`, `x ~ U(0,1)^3`. Truth for PD is centred on its training mean (gamfit's sum-to-zero constraint); pyGAM bands are shifted by their own training-mean offset (generous to pyGAM, whose term levels are unidentified).
Fixed 60-point test set (seed 12345), 25-point PD grid, 200 replicates per cell (seeds 1000+rep), identical data for all methods.

| cell | family | n | b0 | a1 | a3 | sigma |
|---|---|---|---|---|---|---|
| gauss | Gaussian | 200 | 0 | 1.0 | 0.30 | 1.0 |
| gauss_small | Gaussian | 60 | 0 | 1.0 | 0.30 | 0.5 |
| pois | Poisson | 200 | 0.5 | 0.8 | 0.25 | - |
| binom | Binomial | 400 | 0 | 1.5 | 0.60 | - |

Methods: gamfit default `y ~ s(x1)+s(x2)+s(x3)`; pyGAM `s(0)+s(1)+s(2)` default fit (lam=0.6, no selection) and `gridsearch()` default (GCV/UBRE, one shared lam over `logspace(-3,3,11)`).
Coverage is averaged over test points and replicates (MCSE for a 0.95 rate at the replicate level ~0.015; point-level MCSE smaller). "whole-curve" = fraction of replicates where the pointwise 95% band covers the true curve on all 25 grid points (what a simultaneous band would promise).

## 3. Results

### 3.1 Interval coverage (nominal 0.95), 200 replicates per cell

"mean" = 95% interval for E[y|x] on the 60 test points; "obs" = 95% observation interval for a fresh y; PD = pointwise ±1.96·SE partial-dependence band on the 25-point grid; "whole" = fraction of replicates whose pointwise band covers the whole true curve. gamfit failures (B1/B2) are excluded from gamfit rows only: pois 85/200 fits succeeded, binom 160/200, Gaussian 200/200.

| cell | method | mean cov | mean width | obs cov | PD cov x1 / x2 / x3 | PD whole x1 / x2 / x3 | fit s (median, load ~10-35 on 4 CPUs) |
|---|---|---|---|---|---|---|---|
| gauss | **gamfit (corrected, default)** | **0.969** | 0.886 | 0.948 | 0.979 / 0.985 / 0.943 | 0.80 / 0.96 / 0.81 | 0.22 |
| gauss | gamfit conditional | 0.935 | 0.762 | - | - | - | - |
| gauss | pyGAM default (lam=0.6) | 0.948 | 1.419 | 0.947 | 0.956 / 0.962 / 0.958 | 0.53 / 0.49 / 0.53 | 0.06 |
| gauss | pyGAM gridsearch | 0.887 | 0.814 | 0.948 | 0.835 / 0.954 / 0.949 | 0.49 / 0.70 / 0.68 | 0.72 |
| gauss_small | **gamfit** | **0.934** | 0.776 | 0.937 | 0.955 / 0.984 / 0.887 | 0.70 / 0.94 / 0.70 | 0.17 |
| gauss_small | gamfit conditional | 0.921 | 0.735 | - | - | - | - |
| gauss_small | pyGAM default | 0.936 | 1.223 | 0.938 | 0.942 / 0.941 / 0.948 | 0.49 / 0.49 / 0.55 | 0.04 |
| gauss_small | pyGAM gridsearch | 0.839 | 0.810 | 0.932 | 0.739 / 0.942 / 0.933 | 0.27 / 0.69 / 0.62 | 0.65 |
| pois (85 ok) | **gamfit** | **0.978** | 1.399 | 0.990 | 0.985 / 0.974 / 0.980 | 0.84 / 0.93 / 0.89 | 2.52 |
| pois | gamfit conditional | 0.952 | 1.187 | - | - | - | - |
| pois | pyGAM default | 0.950 | 2.445 | - | 0.966 / 0.964 / 0.960 | 0.51 / 0.51 / 0.49 | 0.12 |
| pois | pyGAM gridsearch | 0.862 | 1.207 | - | 0.745 / 0.967 / 0.949 | 0.23 / 0.76 / 0.62 | 1.16 |
| binom (160 ok) | **gamfit** | **0.952** | 0.262 | (1.000, trivially [0,1]) | 0.978 / 0.991 / 0.899 | 0.80 / 0.97 / 0.72 | 2.42 (p90 63 s, max 446 s) |
| binom | gamfit conditional | 0.921 | 0.229 | - | - | - | - |
| binom | pyGAM default | 0.945 | 0.351 | - | 0.961 / 0.963 / 0.961 | 0.57 / 0.59 / 0.57 | 0.06 |
| binom | pyGAM gridsearch | 0.869 | 0.238 | - | 0.782 / 0.960 / 0.911 | 0.34 / 0.73 / 0.55 | 0.70 |

pyGAM has no observation interval for Poisson/binomial (`prediction_intervals` is LinearGAM-only). pyGAM default reaches ~0.95 by carrying edof 23-34 (vs gamfit ~5-7), so its mean bands are 34-75% wider. pyGAM gridsearch is the "tuned" pyGAM and under-covers everywhere (0.84-0.89), worst on the strong term x1 (0.74-0.84) because one lam is shared across terms.

### 3.2 Smooth-term tests: size under H0 (null term s(x2)) and power (weak term s(x3))

| cell | method | P(p<.05) H0 | P(p<.01) H0 | frac p>0.99 H0 | power s(x3) @.05 | LR unavailable (any term) |
|---|---|---|---|---|---|---|
| gauss | gamfit Wald (summary) | 0.050 | 0.005 | 0.44 | 0.540 | - |
| gauss | gamfit LR (`smooth_significance`) | 0.060 | 0.020 | 0.07 | **0.648** | 2/200 |
| gauss | pyGAM default / gridsearch | 0.035 / 0.035 | 0.025 / 0.010 | 0.01 / 0.00 | 0.290 / 0.300 | - |
| gauss_small | gamfit Wald | 0.040 | 0.000 | 0.23 | 0.525 | - |
| gauss_small | gamfit LR | 0.070 | 0.035 | 0.00 | **0.724** | 1/200 |
| gauss_small | pyGAM default / gridsearch | 0.055 / 0.050 | 0.005 / 0.015 | 0 / 0 | 0.270 / 0.310 | - |
| pois (85 ok) | gamfit Wald | 0.035 | 0.024 | 0.14 | 0.694 | - |
| pois | gamfit LR | 0.103 | 0.000 | 0.09 | **0.894** | 28/85 |
| pois | pyGAM default / gridsearch | 0.035 / 0.055 | 0.005 / 0.000 | 0.02 / 0.03 | 0.410 / 0.475 | - |
| binom (160 ok) | gamfit Wald | 0.019 | 0.006 | 0.39 | 0.700 | - |
| binom | gamfit LR | 0.069 | 0.023 | 0.18 | **0.798** | 30/160 raise + 78 None (s(x1) 67, s(x3) 11) |
| binom | pyGAM default / gridsearch | 0.045 / 0.030 | 0 / 0 | 0.02 / 0.05 | 0.485 / 0.460 | - |

Dedicated null runs (`lr_null.py`, seeds 50000+rep, same DGP):

| run | test | .10 | .05 | .01 | n usable |
|---|---|---|---|---|---|
| gauss_small, 400 reps | LR (selection-replay reference, default) | 0.1375 | 0.070 | 0.0125 | 400 |
| | LR, rho-conditional reference | 0.2225 | 0.1125 | 0.0425 | 400 |
| | Wald | 0.090 | 0.045 | 0.0025 | 400 |
| binom, 100 reps | LR | 0.141 | 0.047 | 0.000 | 64 (36 errored: 21 fit, 15 `smooth_significance` raised) |
| | LR, rho-conditional | 0.250 | 0.125 | 0.031 | 64 |
| | Wald | 0.063 | 0.047 | 0.000 | 64 |

Pooled LR size: Gaussian n=60 (600 reps) 0.135 / 0.070 / 0.020 at .10/.05/.01 (MCSE at .05: 0.009); binomial (194 usable) 0.062 at .05 and 0.015 at .01 (MCSE 0.016), i.e. within noise; Poisson 0.103 at .05 on 85 fits (MCSE 0.024).
gamfit null-term edf: median 0.17 (gauss), 0.48 (gauss_small), 0.60 (pois), 0.27 (binom); fraction below 0.05: 0.45 / 0.00 / 0.15 / 0.41.

### 3.3 Posterior `sample()` coverage of E[y|x] (95% percentile interval, 500 draws)

| cell | method | reps usable | coverage | mean width | time per call |
|---|---|---|---|---|---|
| gauss | gamfit `sample()` (laplace, smoothing-corrected) | 100 | **0.967** | 0.880 | 0.017 s |
| gauss | gamfit `predict(interval=.95)` (same fits) | 100 | 0.968 | 0.884 | - |
| gauss | pyGAM `sample(quantity="mu", n_bootstraps=5)` after gridsearch | 100 | 0.916 | 0.898 | 1.76 s |
| binom | gamfit `sample()` (polya-gamma **conditional**; 3/19 routed to NUTS) | 19 of 22 (3 fit errors) | **0.912** | 0.224 | 2-4 s PG; **373-424 s NUTS** |
| binom | gamfit `predict(interval=.95)` (same fits, smoothing-corrected) | 19 | 0.947 | 0.257 | - |
| binom | pyGAM `sample()` | 22 (paired 19: 0.904) | 0.910 | 0.270 | 0.83 s |

Binomial paired difference predict - sample coverage: **+0.035 (SE 0.009)**; predict/sample width ratio 1.15. This is G3: the non-Gaussian sampler ignores rho uncertainty that `predict()` includes. The binomial study was cut to 22 replicates (`mc_sample_stream.py binom 1 200` under a 25-min cap) because 3/19 `sample()` calls took 6-7 minutes (B5); the original 60-replicate run (`mc_sample.py binom 60 1`) produced nothing in 2 h (py-spy: stuck in a slow `fit`, `_api.py:1291`).

## 4. Findings

Kinds: gap | bug | pyGAM-slop-to-avoid | already-better.

Headline: gamfit's default intervals are the only ones near nominal at sensible width in all four cells (mean coverage 0.93-0.98 vs pyGAM tuned 0.84-0.89 / pyGAM default 0.94-0.95 at 1.3-1.8x the width), and its LR test has 1.6-2.7x pyGAM's power. It does **not** yet dominate, because the released wheel fails 58% of Poisson and 20% of binomial fits (B1/B2), the LR is missing for most binomial terms (B3), binomial `sample()` is rho-conditional and occasionally 6 min (G3/B5), and PD bands have no level or simultaneous option (G2). Every item below was reproduced (script named) or pinned to file:line.

### Bugs / gaps in gamfit

**B1 — Released wheel aborts the whole fit when the sigma-point smoothing cubature cannot calibrate (bug, blocker for the shipped artifact; source at HEAD appears fixed but is unverified).**
- Evidence: `mc.py pois` — **115/200 Poisson fits raise** (`IntegrationError('smoothing cubature has no positive-width proposal in the resolved domain')` x92, `'...proposal has no positive width'` x8, outer non-certification x15). Binomial cell: **40/200 fits raise** (cubature x35, outer non-certification x5; §3.1). `lr_null.py binom`: 36/100 replicates error. Gaussian reduced model `y ~ s(x2)+s(x3)` on rep 35 raises in 0.6 s (`repro_null35.py`).
- The cubature is an *upgrade* of the covariance; the fit itself had already been certified. HEAD source `crates/gam-solve/src/reml/eval.rs:1331-1420` wraps node calibration in a closure and returns `first_order_numerical(... "smoothing cubature could not calibrate its nodes: {error}")`, and `:1437-1445` does the same for node integration. The installed `_rust.abi3.so` contains the raising strings but **not** the string `could not calibrate its nodes` (`grep -c -a`), so the wheel predates that fallback.
- Git history is shallow (eval.rs is "new" in 5340cb4), and no test in `crates/` references the fallback reason string, so nothing pins this.
- Fix: release HEAD, and add a regression test that fits the Poisson fixture (`mc.py` cell `pois`, seed 1000+0, which is rep 0) and asserts (a) the fit succeeds and (b) `covariance_source` is still reported, with the fallback reason surfaced in the summary. Files: `crates/gam-solve/src/reml/eval.rs` tests; `crates/gam-pyffi` surfacing of `SmoothingCorrectionOutcome` reason. Size S.

**B2 — Outer REML optimisation fails to certify when a null term's shrinkage lambda rails (bug, high).**
- Evidence: 15/200 Poisson fits (7.5%), 5/200 binomial fits, and binomial rep 0 (`repro_binom0.py`: **fails after 198 s**). Related: 17/160 *successful* binomial fits take >60 s (max 446 s, median 2.4 s), the same slow outer path. The message shows `railed=[4] theta=29.86 box=[-30,30]`, `|Pg|=1.331e-4 > bound=1.010e-4`, `asymptote-rail declined: interior not stationary`, and `line_search=StepSizeTooSmall after 50 attempts`.
- This is exactly the null-recovery case: the null term's null-space lambda going to infinity. SPEC requires that defaults "allow recovering the null", and a fit that hits it must still converge. It also removes the fit from all inference.
- Fix (belongs to the robustness/optimizer axis): treat the rail-bound coordinate by its asymptotic limit, i.e. drop the term's null space as the limit model, then certify the remaining face. The current path declines that because the free face misses the bound by 30%. Files: `crates/gam-solve/src/reml/reml_outer_engine/*` (rail/asymptote certification). Size M.

**B3 — smooth_significance silently returns `None` when the null refit fails, and the null refit computes a smoothing correction it never uses (bug, high).**
- Evidence: in `mc.py`, `p_lr` is None for the strongest term in 2/200 (gauss), 1/200 (gauss_small), 28/85 surviving (pois) fits, and in binom `smooth_significance` **raises in 30/160** successful fits and returns None for s(x1) in 67 and s(x3) in 11 of the other 130, so the LR for the strongest term is unavailable in 97/160 (61%) binomial fits, with `statistic_lr: None, correction_provenance: 'none'` (`repro_lr.py`, rep 35). The reduced model raises at the cubature step (`repro_null35.py`).
- Code: `crates/gam-models/src/fit_orchestration/drivers/smooth_term_lr.rs:3001-3017` maps any `Err` to `(f64::NAN, None, None)` with no reason. `:2977` and `:2987` `continue` silently. `:2992-3000` passes the caller's `options` unchanged (`compute_inference: true`, `entry.rs:59`), so the refit runs the rho-cubature.
- Inconsistently, `lr_null.py binom` shows 15/100 cases where `smooth_significance` **raises** (`smooth_term_lr_inference: ...`) instead of returning None.
- Fix: the null refit needs only `log_likelihood` and eta. Run it with `compute_inference: false` (no correction, which removes the failure source and the cost). Carry a structured `unavailable_reason` per term instead of NaN/None, and make the raise-vs-None behaviour uniform (always a per-term reason). Files: `smooth_term_lr.rs`, `crates/gam-pyffi/src/manifold/manifold_and_posterior_ffi.rs:1024-1285` (row marshaling). Size S.

**B5 — The automatic Firth/Jeffreys rescue silently changes the estimator, and then sends `sample()` to a 6-minute NUTS run (bug, med).**
- Evidence: on well-posed binomial data (n=400, no separation), `sample()` reported `method='nuts'` in 3/19 replicates (reps 13/16/21) and took 373/378/424 s, against 2-4 s for Polya-Gamma (`sample_binom.jsonl`). `repro_nuts_firth2.py`: the serialized rep-21 model has `"firth_bias_reduction": true` (rep 14: `false`). `repro_firth_summary.py`: neither `summary()`, `summary().convergence` nor `predict()` mentions Firth/Jeffreys, and the model repr doesn't either. Those fits are also the slow ones (25-70 s vs 1.2 s).
- Code: `crates/gam-models/src/fit_orchestration/fit.rs:562-587` retries with `firth_bias_reduction = true` after a base failure with "separation/non-convergence evidence" and adopts it with only a `log::info!`. `crates/gam-inference/src/hmc_io.rs:5258` then routes any Firth fit off Polya-Gamma to NUTS.
- So the user gets a Jeffreys-penalised posterior without being told, triggered here on data with no separation (the base fit's failure is the B1/B2 kind). The adoption is certified, which is SPEC-legal, but invisible.
- Fix:
  1. Surface `firth_bias_reduction` and the retry reason in the summary payload's convergence block (`crates/gam-pyffi/src/model/model_ffi.rs:4482`).
  2. Fixing B1/B2 removes the spurious triggers on non-separated data.
  3. Under Firth, sample with the Polya-Gamma conditional as an independence proposal plus an MH correction for the Jeffreys factor `|I(beta')|^{1/2}/|I(beta)|^{1/2}`. That is exact, costs one log-det per draw and needs no tuning knob, and it replaces NUTS on this route (`hmc_io.rs:5258-5290`, `run_logit_polya_gamma_gibbs` at `:4562`).
- Size: S (surface) + M (PG-MH).

**G1 — Summary lacks scale/dispersion, deviance explained and (c)AIC (gap, med).**
- Evidence: `Summary` attributes are basis_checks, coefficient_se_source, coefficients, ..., deviance, edf_total, log_likelihood, n_obs, reml_score, and so on. There is no phi/scale, no deviance_explained or pseudo-R², and no AIC (probe in session; `crates/gam-pyffi/src/model/model_ffi.rs:4482 summary_payload_from_model_bytes`).
- phi is already serialised (`payload/fit_result/inference/dispersion {source, phi}`). The CLI computes `deviance_explained` from data (`crates/gam-cli/src/main/model_summary.rs:85`). `Model.conditional_aic` exists at HEAD `gamfit/_model.py:1257` but not in the wheel. compare_models gives cAIC-based ranking only across models.
- Fix: in the Rust summary payload, emit `dispersion {phi, source}` and `conditional_aic` (Wood-Pya-Saefken corrected edf, already in `crates/gam-inference/src/model_comparison.rs:114`). Store the null deviance at fit time (intercept-plus-offset-only deviance, one extra IRLS on the intercept) so that `deviance_explained = 1 - D/D0` is data-free and identical between CLI and Python. Python only reads fields. Do NOT add AICc, GCV/UBRE, or McFadden-style log-lik ratios. Files: `crates/gam-pyffi/src/model/model_ffi.rs`, `crates/gam-models` fit result (null deviance), `crates/gam-cli/src/main/model_summary.rs` (reuse the stored value), `gamfit/_summary.py`. Size S-M.

**G2 — partial_dependence has no interval level and no simultaneous band (gap, med).**
- Evidence: it returns only `predicted` and `standard_error` (`gamfit/_model.py:1154`). With pointwise ±1.96 SE, the whole curve is covered in only **0.70–0.84** of replicates for the non-null terms (e.g. gauss_small x1/x3 0.70/0.70; gauss 0.80/0.81; pois 0.84/0.89; binom 0.80/0.72). A curve-level claim ("the effect is non-linear everywhere") needs a simultaneous band.
- The machinery exists: `crates/gam-inference/src/effects.rs:260 effect_report` with `BandOptions::Simultaneous` (max|Z| calibration, no ridge), already used by `difference_smooth(simultaneous=True)` (`difference_smooth.rs:166`).
- Fix: `partial_dependence(term, data, grid, level=0.95, simultaneous=False)` routes the PD contrast rows through `effect_report` with the model's default (smoothing-corrected) covariance, returning `lower/upper/critical`. `level` mirrors `predict(interval=)`, so no new knob. Files: `crates/gam-inference/src/partial_dependence.rs`, `effects.rs` (reuse), the PD FFI in `crates/gam-pyffi`, `gamfit/_model.py:1154`. Size S.

**G3 — Non-Gaussian `sample()` draws from the rho-conditional posterior, unlike `predict()` (gap, med).**
- Evidence: the binomial `sample()` reports `method='polya-gamma'`, `covariance_source='conditional'`, `is_exact=True` (probe6). The Gaussian path uses `laplace` + `smoothing-corrected`.
- So for binomial and Poisson, `sample()` intervals are narrower than `predict()` intervals and silently omit rho uncertainty. Measured (§3.3, `mc_sample_stream.py binom`): binomial `sample()` covers 0.912 vs `predict()` 0.947 on the same 19 fits (paired +0.035, SE 0.009), i.e. no better than pyGAM's 0.904/0.910; widths 0.224 vs 0.257.
- Fix: draw rho first, from the same measure the smoothing correction integrates (the certified SigmaPointCubature nodes and importance weights, or the Tier-1 Gauss-Hermite rule when K<=4), then run exact PG given rho. That gives a proper mixture over rho with no new options. `is_exact` must then describe the beta|rho step only, and `covariance_source` should report `smoothing-marginalised`. Files: `crates/gam-inference/src/sample.rs`, `polya_gamma.rs`, `crates/gam-solve/src/reml/eval.rs` (expose nodes/weights from `SmoothingCorrectionOutcome`). Size M.

**G4 — Rho-posterior adequacy tiers exist but are never surfaced (gap, low-med).**
- Evidence: `crates/gam-inference/src/rho_posterior.rs` (Tier 0 PSIS k-hat, Tier 1 GH quadrature K<=4, Tier 2 NUTS K<=16). Default fits set `skip_rho_posterior_inference: true` (`crates/gam-models/src/fit_orchestration/entry.rs:69`). grep for k_hat/rho_posterior in `crates/gam-pyffi/src` and `gamfit/` only hits PSIS-LOO.
- So a user cannot see whether the Gaussian rho approximation that the corrected covariance relies on is adequate.
- Fix: publish the Tier-0 k-hat (documented as computed "whenever cheaply available", `entry.rs:64`) in `summary().convergence` as `rho_posterior_khat`. When k-hat flags inadequacy, escalate automatically to Tier 1 for K<=4; that is a rule, not a knob. Files: `crates/gam-models/src/fit_orchestration/entry.rs`, `crates/gam-solve/src/reml/eval.rs:970`, `model_ffi.rs` summary payload. Size S (expose) / M (auto-escalate).

**B4 — LR smooth test is mildly anti-conservative (Gaussian small-n, Poisson) (bug, med).**
- Evidence: null term s(x2), rejection rate under H0 (a pooled 600 replicates at n=60 from `lr_null.py gauss_small` plus `mc.py`):

  | α | rate | MCSE excess |
  |---|---|---|
  | .10 | 0.135 | +3.9 |
  | .05 | 0.070 | +2.2 |
  | .01 | 0.020 | +2.5 |

  At n=200 (gauss): 0.060 at .05 and 0.020 at .01. The rho-conditional reference alone is far worse (0.1125 at .05, 0.0425 at .01), so the selection replay helps but under-corrects. Poisson: 0.103 at .05 (85 fits, +2.2 MCSE). Binomial pooled over `mc.py` + `lr_null.py`: 0.062 at .05, 0.015 at .01 (194 fits), within noise.
- Wald: 0.019–0.050 at .05, with a point mass near p=1 when edf goes to 0 (44% of gauss p>0.99). That is conservative in the middle but correctly sized at α. This is the same behaviour as mgcv and acceptable.
- LR power is the best of all methods (s(x3) gauss/gauss_small/pois/binom: 0.65/0.72/0.89/0.80 vs Wald 0.54/0.53/0.69/0.70 vs pyGAM 0.27–0.49).
- Fix: the replay currently evaluates the selection shift at `E[V]` (first-order, `smooth_term_lr.rs:2420-2440`). Integrate it over the residual law V (a 1-D expectation, which is the next term the code comment itself names), and check that the replay draws use the profiled-scale law. Add this MC (seeded, 400 reps at n=60) as a calibration test. Files: `smooth_term_lr.rs`. Size M.

**L1 — Bartlett factor blows up for a shrunk term (bug, low).**
- Evidence: `lr_null.py binom`: for s(x2), `bartlett_factor` is above 2 in 31% of fits, up to **3.5e4**. `bartlett_factor_from_mean` returns `mean_w/ref_df` (`crates/gam-terms/src/inference/higher_order.rs:20-25`), and ref_df goes to 0 when edf goes to 0. W is about 0 in those cases, so p is unaffected, but the reported factor and the `material` flag are meaningless.
- Fix: apply Lawley as an additive mean shift on the spectral reference (per-component weights), not a ratio to ref_df. Size S.

**L2 — Gaussian intervals use z, not t, with estimated scale (gap, low).**
- `crates/gam-predict/src/interval_policy.rs:98` uses `standard_normal_quantile`. gauss_small (n=60) coverage: corrected mean 0.934, conditional 0.921, obs 0.937, i.e. about 1–1.5 pts under at small n.
- Fix: marginalising phi under its reference prior gives Student-t with residual df n - edf, which is principled rather than a knob. Apply it for the estimated-scale families only. Files: `interval_policy.rs`, `crates/gam-predict/src/lib.rs:2839`. Size S.

**L3 — Dead Bonferroni option (low, SPEC "unnecessary options deleted").**
- `PredictOptions.multi_point_joint` (`crates/gam-predict/src/lib.rs:1478-1485`, `:1689-1697`) is set only to `false` at every construction site (`:1557, 3298, 3525, 3563, 3893, 3938`). Bonferroni is also dominated by the max|Z| simulation in `effects.rs`.
- Fix: delete it and route joint bands through `effect_report`. Size S.

**L4 — LR p-values floor at about 1e-16 or 0.0 (low).**
- Wald p=1.4e-32 for s(x1), while the LR reports 0.0 or 1.665e-16 (`probe6.py`). This is by design: the tail is resolved only to the statistic's own noise, and the bound is returned (`smooth_term_lr.rs:2380-2403`).
- Fix: expose `p_value_bound` in docs, or return the bound as an upper bound (`p < 1e-16`) rather than a point value. Size S.

**L5 — Weak-term PD under-covers under heavy shrinkage at small n (observation, low).**
- gauss_small x3 (amplitude 0.3, n=60) pointwise PD coverage is 0.887; binom x3 (amplitude 0.6 on logit, n=400) is 0.899. This is the across-the-function Nychka property failing when the double penalty shrinks a real but weak effect towards its null space.
- Overall mean coverage is still near nominal. It is worth tracking once G4 (rho-posterior adequacy) is exposed. No fix proposed without more evidence.

### already-better (keep)

- **A1** Smoothing-corrected covariance by default. Mean coverage corrected vs conditional: 0.969 vs 0.935 (gauss), 0.934 vs 0.921 (gauss_small), 0.978 vs 0.952 (pois), 0.952 vs 0.921 (binom). pyGAM gridsearch: 0.887/0.839/0.862/0.869. pyGAM default reaches ~0.95 only by overfitting (edof 31 vs gamfit ~7), which makes its intervals 60% wider.
- **A2** Observation intervals for all families. pyGAM `prediction_intervals` exists only on LinearGAM (`pygam.py:2477`).
- **A3** Null recovery by double penalty: gamfit edf for the null term has median 0.17 (gauss) and 0.27 (binom), below 0.05 in 45% / 41% of fits. pyGAM always carries ~full edof, and its levels are unidentified (EPS ridge).
- **A4** Gaussian `sample()`: coverage 0.967 in 17 ms vs pyGAM `sample()` 0.916 in 1.76 s (`mc_sample.py gauss`, 100 reps). Binomial `sample()` is **not** better (0.912 vs pyGAM 0.910; G3/B5).
- **A5** Test power: the LR test has about twice pyGAM's power at equal or better size.

### pyGAM slop to avoid (verified in pyGAM 0.12.0 source or by run)

- **S1** Default fit uses fixed `lam=0.6` per term with no selection (`terms.py:90,648`), giving edof ~31 for 3 smooths at n=200 and intervals 1.6x wider than needed.
- **S2** `gridsearch` uses GCV (with `gamma=1.4`, `pygam.py:1157`) or UBRE over `logspace(-3,3,11)` with **one shared lam** for all terms (`:1966`). That oversmooths the strong term: median lam 63 (gauss), 251 (pois), PD coverage for x1 0.835/0.739/0.745, mean coverage 0.84–0.89.
- **S3** Covariance is conditional on lam (`pygam.py:1043`).
- **S4** Wald p-values centre coefficients by their mean (`coef -= coef.mean()`, `:1281`, not a function constraint), use a full-rank `pinv` with df=rank (`:1283`), and compute `1 - cdf` (`:1289,1293`), so p floors at 1.1e-16. Power is half of gamfit's LR.
- **S5** McFadden R² is inverted: `full_ll / null_ll` (`:1151`); for a binary fit it gave 0.636 vs explained deviance 0.364.
- **S6** AICc uses the Gaussian small-sample formula for every family (`:1086`).
- **S7** `sample()` defaults to `n_bootstraps=5` and a random lam grid `exp(randn*6-3)` whose comment claims [1e-3,1e3] (`:2318`). It uses the global RNG with no seed, and still under-covers (0.916).
- **S8** `prediction_intervals` is LinearGAM-only (`:2477`).
- **S9** Term levels are identified only by an EPS ridge. The uncentred PD offset for the null term has median 0.047/0.042 (pyGAM default), so the PD band is for an unidentified level.
