# Rust, CLI, and Python surface inventory

This inventory is a source-level census of the supported **user workflows**. It
was last checked against `src/lib.rs`, `crates/gam-model-api/src/lib.rs`,
`crates/gam-cli/src/main/cli_args.rs`, `gamfit/__init__.py`, and
`gamfit/_api.py`. “Shared” means the frontend serializes the same
`gam.fit-request` document or consumes the same saved-model Rust operation; it
does not mean that a command-line program reproduces programmatic building-block
APIs such as a matrix kernel.

## User-workflow cross-tabulation

| Capability / single Rust authority | Rust library | `gam` CLI | `gamfit` Python | Status |
|---|---|---|---|---|
| Formula fit and REML/LAML family dispatch (`FitRequest`, `fit_from_formula`, `fit_model`) | Public facade | `fit DATA FORMULA --out` or `--request` | `fit`, `fit_array`, `gaussian_reml_fit_formula` | Shared request; parity |
| Families: auto, Gaussian, binomial (logit/probit/cloglog), Poisson, Gamma, beta, Tweedie, negative binomial, expectile | `FitConfig` / family resolver | `--family`; NB and expectile controls below | `family=` | Parity |
| Fixed negative-binomial size | `FitConfig::negative_binomial_theta` | `--negative-binomial-theta` | `negative_binomial_theta=` | **Closed by this audit** |
| Expectile target | `FitConfig::expectile_tau` | `--expectile-tau` | `expectile_tau=` or `expectile(tau)` family spelling | **Closed by this audit** |
| Offset, weights, persistent warm starts | shared request fields | `--offset-column`, `--weights-column`, `--persistent-warm-start-root` | `offset=`, `weights=`, `persistent_warm_start_root=` | Parity |
| Links and flexible link | family/link resolver | binomial family variants; complete request for `link` / `flexible_link` | `link=`, `flexible_link=` | Parity through shared request |
| Firth binomial correction | `FitConfig::firth` | `--firth` | `firth=` | Parity |
| Location/dispersion scale models | typed `FitRequest` variants | `--predict-noise`, `--noise-offset-column` | `noise_formula=`, `noise_offset=` | Parity |
| CTN and calibrated marginal slope | typed requests / `CtnStage1Recipe` | `--transformation-normal`, `--ctn-stage1`, `--slope-formula`, `--z-column` | same concepts as keywords | Parity |
| Survival (transformation, Weibull, location-scale, marginal-slope, latent) and baseline/frailty controls | typed survival requests | `Surv(...)` plus survival, time-basis, baseline, frailty flags | `survival_likelihood=`, anchor/baseline/frailty keywords; formula/config for time basis | Parity through shared request |
| Latent coordinates, analytic penalties, smooth descriptors, precision hyperpriors | request document and term builders | JSON descriptor flags or complete request | `latents=`, `penalties=`, `smooths=`, `precision_hyperpriors=` | Parity |
| Fit-time conformal substrate and inference retention | request document | `--precompute-conformal`, `--inference` | `config=` request fields | Parity through shared request |
| Prediction and posterior-mean uncertainty | `gam::predict` saved-model machinery | `predict`, `--uncertainty`, `--level`, `--covariance-mode`, offsets and IDs | `Model.predict`, `predict_array`; interval, observation interval, covariance mode, IDs | Parity; Python exposes richer typed return objects |
| CTN observed-response score | saved-model prediction machinery | `transformation-score` | `Model.transformation_score` | Parity |
| Diagnostics / ALO | `gam::inference::alo`, saved-model ALO | `diagnose [--alo]`; report may include diagnostics | `Model.diagnose`, `check`, `basis_check`, `curvature`, `smooth_significance` | Core diagnostics shared; Python methods are programmatic views |
| Posterior coefficient sampling | `gam::inference::sample` / `gam::hmc` | `sample --chains --samples --warmup --seed` | `Model.sample` with the same controls plus `target_accept` | Same Rust sampler |
| Posterior predictive / response generation | `gam::predict::generative` | `generate --n-draws --seed` | `sample_replicates`, `iter_replicates`; multinomial `posterior_predict` | Same Rust generator |
| Summary and HTML report | saved model / `gam::report` | `report` | `summary`, `report` | Parity |
| Persistence | saved-model envelope | fit writes and all consumers read it | `save`, `load`, `loads`, `Model.save`, `dumps`, `model_from_dict` | One Rust wire format |
| Multinomial fit/predict/inference | Rust multinomial request/model | selected by `--family multinomial` | `family="multinomial"`, `MultinomialModel` | Parity |
| Event history | Rust event-history engine | `fit-events` | `fit_event_history`, `EventHistoryModel` | Parity |
| Manifold crosscoder | Rust auto-fit/report engine | `crosscoder` and its override flags | `sae_crosscoder_fit` / crosscoder objects | Same Rust engine; transport-specific inputs |
| Response geometry | Rust geometry/fitting modules | complete versioned request/artifact workflows | `fit(..., response_geometry=...)`, `ResponseGeometryModel` | Programmatic API; no duplicate CLI math |

## CLI command and flag inventory

Global flags are `--log-level`, `--verbose`/`-v`, and `--quiet`/`-q`.

| Command | Arguments and flags |
|---|---|
| `fit` | `DATA`, `FORMULA`; `--request`, `--ctn-stage1`, `--precision-hyperpriors`, `--latent-coordinates`, `--analytic-penalties`, `--smooth-descriptors`, `--predict-noise`, `--slope-formula`, `--z-column`, `--weights-column`, `--offset-column`, `--noise-offset-column`, `--frailty-kind`, `--frailty-sd`, `--hazard-loading`, `--transformation-normal`, `--firth`, `--family`, `--negative-binomial-theta`, `--expectile-tau`, `--survival-likelihood`, `--survival-time-anchor`, baseline and time-basis controls, `--scale-dimensions`, `--precompute-conformal`, `--inference`, `--persistent-warm-start-root`, `--out` |
| `predict` | `MODEL NEW_DATA --out`; offset/noise-offset/ID, `--uncertainty`, `--level`, `--covariance-mode` |
| `transformation-score` | `MODEL LABELLED_DATA --out`; offset and ID columns |
| `diagnose` | `MODEL DATA [--alo]` |
| `sample` | `MODEL DATA`; `--chains`, `--samples`, `--warmup`, `--seed`, `--out` |
| `generate` | `MODEL DATA`; `--n-draws`, `--seed`, `--out` |
| `report` | `MODEL [DATA] [OUT]` |
| `fit-events` | `--subjects`, `--events`, `--covariates`, `--formula` or `--mark-formula` (one per mark), `--marks`, `--horizons-after-exit`, `--forecast-cutoff`, `--out` |
| `crosscoder` | named anchor/block matrices, atom/harmonic counts, sparsity/smoothness/optimizer overrides, transport-law controls, `--out` |

## Python public fitting, prediction, and diagnostic inventory

The formula front doors are `fit`, `fit_array`, `validate_formula`, and
`gaussian_reml_fit_formula`. The complete `fit` model-spec keyword set is:

`family`, `negative_binomial_theta`, `expectile_tau`, `offset`, `weights`,
`persistent_warm_start_root`, `transformation_normal`,
`transformation_normal_stage1`, `survival_likelihood`, `survival_time_anchor`,
`baseline_target`, `baseline_scale`, `baseline_shape`, `baseline_rate`,
`baseline_makeham`, `z_column`, `link`, `slope_formula`, `frailty_kind`,
`frailty_sd`, `hazard_loading`, `scale_dimensions`, `firth`, `noise_formula`,
`noise_offset`, `flexible_link`, `precision_hyperpriors`, `constraints`,
`response_geometry`, `response_columns`, `response_coordinates`,
`response_reference`, `fisher_rao_w`, `latents`, `penalties`, `smooths`, and
`config`.

The fitted `Model` public workflow methods/properties are `predict`,
`predict_array`, `predict_conformal`, `transformation_score`, `summary`,
`smoothing_parameters`, `check`, `curvature`, `smooth_significance`,
`basis_check`, `debiased_functional`, `report`, `sample`, `sample_replicates`,
`iter_replicates`, `design_matrix`, `design_matrix_array`, `difference_smooth`,
`partial_dependence`, `variance_share`, `evidence`, `evidence_ratio_vs`,
`diagnose`, `plot`, persistence methods, group extension, and model metadata.
`MultinomialModel` exposes classes, deviance/iterations, prediction and standard
errors, posterior prediction, smooth significance, summary, and persistence.

The remaining public names in `_api.py` are programmatic Rust-kernel bindings,
not missing CLI workflows: basis/penalty constructors; weighted-ridge helpers;
Gaussian REML forward/backward, batched, position, latent, block, and constrained
variants; GLM latent REML; CUDA/build diagnostics; and shared-precision updates.
Their full signatures are the Python source of truth. They deliberately remain
library APIs: converting arrays, derivatives, and autograd state to shell flags
would add a second scientific configuration language rather than parity.

## Audit conclusion

The user-visible defect was that two fields already present in the Rust request
and CLI—fixed negative-binomial theta and expectile tau—were available in Python
only through the untyped `config` escape hatch. Both are now first-class Python
keywords on `fit`, `fit_array`, and `validate_formula`, serialized without
Python-side numerical logic. No duplicate fitter, predictor, diagnostic, family,
or link implementation was found in a frontend.
