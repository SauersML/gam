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
| Offset, weights | shared request fields | `--offset-column`, `--weights-column` | `offset=`, `weights=` | Parity |
| Joint non-crossing expectile levels (location-scale `μ(x) + c_τ·σ(x)`) | `FitConfig::expectile_tau` list, `FitResult::ExpectileLocationScale`, `gam_predict::joint_expectile_curves` | `--expectile-tau 0.1,0.5,0.9`; predict writes one `expectile_{τ}` column per level | `expectile_tau=[0.1, 0.5, 0.9]`; `predict` returns `(n, K)` curves | Parity |
| Links and flexible link | family/link resolver | binomial family variants; complete request for `link` / `flexible_link` | `link=`, `flexible_link=` | Parity through shared request |
| Firth binomial correction | `FitConfig::firth` | `--firth` | `firth=` | Parity |
| Location/dispersion scale models | typed `FitRequest` variants | `--predict-noise`, `--noise-offset-column` | `noise_formula=`, `noise_offset=` | Parity |
| CTN and calibrated marginal slope | typed requests / `CtnStage1Recipe` | `--transformation-normal`, `--slope-formula`, `--z-column` | same concepts as keywords | Parity |
| Survival (transformation, Weibull, location-scale, marginal-slope, latent) and baseline/frailty controls | typed survival requests | `Surv(...)` plus survival, time-basis, baseline, frailty flags | `survival_likelihood=`, anchor/baseline/frailty keywords; formula/config for time basis | Parity through shared request |
| Latent coordinates, analytic penalties, smooth descriptors, precision hyperpriors | request document and term builders | complete request (`--request`) | `latents=`, `penalties=`, `smooths=`, `precision_hyperpriors=` | Parity |
| Prediction and posterior-mean uncertainty | `gam::predict` saved-model machinery | `predict`, `--uncertainty`, `--level`, `--covariance-mode`, offsets and IDs | `Model.predict`, `predict_array`; interval, observation interval, covariance mode, IDs | Parity; Python exposes richer typed return objects |
| Conformal prediction bands | `gam_predict::conformal_routes` | `predict --conformal`, `--training-data`, `--calibration`, `--level` | `Model.predict(interval="conformal", training_data=... or calibration=...)` | Same Rust routes |
| CTN observed-response score | saved-model prediction machinery | `transformation-score` | `Model.transformation_score` | Parity |
| Diagnostics / ALO | `gam::inference::alo`, saved-model ALO | `diagnose`; report may include diagnostics | `Model.diagnose`, `check`, `basis_check`, `curvature`, `smooth_significance` | Core diagnostics shared; Python methods are programmatic views |
| Term partial effects with pointwise intervals and simultaneous bands | `gam_predict::partial_effect::partial_effect` | `partial-effect MODEL --term --level --n-points --grid --out` (JSON or CSV) | `Model.partial_dependence` → `PartialEffect`; `Model.plot_terms` draws it | Same Rust function |
| Posterior coefficient sampling | `gam::inference::sample` / `gam::hmc` | `sample --samples --seed` | `Model.sample` with the same controls | Same Rust sampler |
| Posterior predictive / response generation | `gam::predict::generative` | `generate --n-draws --seed` | `sample_replicates`, `iter_replicates`; multinomial `posterior_predict` | Same Rust generator |
| Summary and HTML report | saved model / `gam::report` | `report` | `summary`, `report` | Parity |
| Model comparison on the smoothing-corrected AIC | `compare_saved_models` | `compare MODEL... --names` | `compare_models`, `Model.evidence_ratio_vs` | Same Rust function; identical JSON |
| Persistence | saved-model envelope | fit writes and all consumers read it | `save`, `load`, `loads`, `Model.save`, `dumps`, `model_from_dict` | One Rust wire format |
| Multinomial fit/predict/inference | Rust multinomial request/model | selected by `--family multinomial` | `family="multinomial"`, `MultinomialModel` | Parity |
| Event history | Rust event-history engine | `fit-events` | `fit_event_history`, `EventHistoryModel` | Parity |
| Manifold crosscoder | Rust auto-fit/report engine | `crosscoder`, `--random-state` | `sae_crosscoder_fit(random_state=)` / crosscoder objects | Parity: the seed is the one control either front door takes |
| Response geometry | Rust geometry/fitting modules | complete versioned request/artifact workflows | `fit(..., response_geometry=...)`, `ResponseGeometryModel` | Programmatic API; no duplicate CLI math |

## CLI command and flag inventory

The one global flag is `-v/--verbose` (the Python counterpart is the `gamfit` logger: `logging.getLogger("gamfit").setLevel(logging.DEBUG)`).

| Command | Arguments and flags |
|---|---|
| `fit` | `DATA`, `FORMULA`; `--request`, `--predict-noise`, `--slope-formula`, `--z-column`, `--weights-column`, `--offset-column`, `--noise-offset-column`, `--frailty-kind`, `--frailty-sd`, `--hazard-loading`, `--transformation-normal`, `--firth`, `--family`, `--negative-binomial-theta`, `--expectile-tau`, `--survival-likelihood`, baseline and time-basis controls, `--scale-dimensions`, `--out` |
| `predict` | `MODEL NEW_DATA --out`; offset/noise-offset/ID, `--uncertainty`, `--level`, `--covariance-mode`, `--conformal`, `--training-data`, `--calibration` |
| `transformation-score` | `MODEL LABELLED_DATA --out`; offset and ID columns |
| `diagnose` | `MODEL DATA` |
| `partial-effect` | `MODEL --term`; `--level`, `--n-points` or `--grid`, `--out` (`.json` or `.csv`) |
| `sample` | `MODEL DATA`; `--samples`, `--seed`, `--out` |
| `compare` | `MODEL...`; `--names` |
| `generate` | `MODEL DATA`; `--n-draws`, `--seed`, `--out` |
| `report` | `MODEL [DATA] [OUT]` |
| `fit-events` | `--subjects`, `--events`, `--covariates`, `--formula` or `--mark-formula` (one per mark), `--marks`, `--horizons-after-exit`, `--forecast-cutoff`, `--reference-row`, `--reference-stratum`, `--out` |
| `crosscoder` | named anchor/block matrices, atom/harmonic counts, `--random-state`, `--out` |

## Python public fitting, prediction, and diagnostic inventory

The formula front doors are `fit`, `fit_array`, `validate_formula`, and
`gaussian_reml_fit_formula`. The complete `fit` model-spec keyword set is:

`family`, `negative_binomial_theta`, `expectile_tau`, `offset`, `weights`,
`transformation_normal`,
`transformation_normal_stage1`, `survival_likelihood`, `survival_time_anchor`,
`baseline_target`, `baseline_scale`, `baseline_shape`, `baseline_rate`,
`baseline_makeham`, `z_column`, `link`, `slope_formula`, `frailty_kind`,
`frailty_sd`, `hazard_loading`, `scale_dimensions`, `firth`, `noise_formula`,
`noise_offset`, `flexible_link`, `warm_start_from`, `precision_hyperpriors`,
`constraints`, `response_geometry`, `response_columns`, `response_coordinates`,
`response_reference`, `fisher_rao_w`, `latents`, `penalties`, `smooths`, and
`config`.

The fitted `Model` public workflow methods/properties are `predict`,
`predict_array`, `transformation_score`, `summary`,
`smoothing_parameters`, `check`, `curvature`, `smooth_significance`,
`basis_check`, `debiased_functional`, `report`, `sample`, `sample_replicates`,
`iter_replicates`, `design_matrix`, `design_matrix_array`, `difference_smooth`,
`partial_dependence`, `plot_terms`, `variance_share`, `evidence_ratio_vs`,
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
