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

BINOM_AND_TABLES_PLACEHOLDER

## 4. Findings

FINDINGS_PLACEHOLDER
