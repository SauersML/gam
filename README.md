# gam

Generalized penalized likelihood engine with a formula-first CLI.

## Install

### Prebuilt binary (macOS / Linux / Windows Git Bash)

```bash
curl -fsSL https://raw.githubusercontent.com/SauersML/gam/main/install.sh | bash
```

### Build from source

```bash
cargo build --release
./target/release/gam --help
```

If `gam` is not on your `PATH`, use `./target/release/gam` in all commands below.

## Command Surface

Top-level commands:

- `gam fit` (alias: `gam train`)
- `gam predict`
- `gam report`
- `gam diagnose`
- `gam sample`
- `gam generate` (alias: `gam simulate`)

Inspect options:

```bash
gam <command> --help
```

## End-to-End Quickstart

```bash
# 1) Fit and save a model
gam fit bench/datasets/wine.csv \
  'price ~ year + rain + smooth(temp)' \
  --out wine.model.json

# 2) Predict
gam predict wine.model.json bench/datasets/wine.csv \
  --out wine.pred.csv --uncertainty --level 0.95

# 3) Build HTML report
gam report wine.model.json bench/datasets/wine.csv
# writes: wine.model.report.html
```

## Formula Language

`fit` takes `<DATA> <FORMULA>` where the formula is `response ~ term + term + ...`.

### Response side

- Standard regression/classification: `y`
- Survival mode trigger: `Surv(entry, exit, event)`
  - Exactly 3 columns are required.

### Implicit terms

A bare column name on the RHS is interpreted by column type:

- Continuous/Binary: penalized linear term
- Categorical: random effect block

### Supported term functions

- Linear and constraints:
  - `linear(x)`
  - `linear(x, min=..., max=...)`
  - `constrain(x, min=..., max=...)` (aliases: `constraint`, `box`)
  - `nonnegative(x)` / `nonnegative_coef(x)`
  - `nonpositive(x)` / `nonpositive_coef(x)`
- Bounded coefficient geometry:
  - `bounded(x, min=..., max=...)`
  - Optional prior controls:
    - `prior=none|uniform|log-jacobian|center`
    - `beta_a=..., beta_b=...`
    - `target=..., strength=...`
- Random effects:
  - `group(x)` (alias: `re(x)`)
- Smooths:
  - `smooth(...)` (alias: `s(...)`)
  - `thinplate(...)` (aliases: `thin_plate`, `tps`)
  - `matern(...)`
  - `duchon(...)`
  - `tensor(...)` (aliases: `interaction`, `te`)
- Formula-level configs:
  - `link(type=..., rho=..., sas_init=..., beta_logistic_init=...)`
  - `linkwiggle(degree=..., internal_knots=..., penalty_order=..., double_penalty=...)`
  - `timewiggle(degree=..., internal_knots=..., penalty_order=..., double_penalty=...)`
  - `survmodel(spec=..., distribution=...)`

Constraints:

- Intercept removal (`0`, `-1`) is not supported.
- At most one `link(...)`, one `linkwiggle(...)`, one `timewiggle(...)`, and one `survmodel(...)` per formula.

### Smooth basis behavior

Default behavior by term shape:

- `smooth(x)` with one variable defaults to B-spline (`bspline`/`ps` family behavior).
- Multi-variate `smooth(x1, x2, ...)` defaults to thin-plate.
- `te(...)` defaults to tensor B-spline.

Important smooth options and semantics:

- B-spline (`type=bspline|ps|p-spline`):
  - `degree` default `3`
  - `knots=<internal_knots>` or `k=<basis_dim>` (mutually exclusive)
  - `penalty_order` default `2`
- Thin plate (`type=tps|thinplate|thin-plate`):
  - Supports 1 to 3 dimensions in joint smooths
  - `centers` or `k`
- Matérn (`type=matern`):
  - `centers` or `k`
  - `nu` in `{1/2, 3/2, 5/2, 7/2, 9/2}`
  - `length_scale` default `1.0`
- Duchon (`type=duchon`):
  - `centers` or `k`
  - `power` (integer, default `2`)
  - `order` in `{0,1}`
  - Omitting `length_scale` keeps pure scale-free Duchon; setting it uses hybrid Duchon-Matérn behavior.
- Tensor:
  - `k`/`basis_dim` controls marginal basis size for each margin.

`identifiability` options are available for spatial/Matérn/tensor smooths (for example `none`, `sum_tozero`, `orthogonal_to_parametric` depending on basis type).

## Link Functions and Link Flexibility

Links are set in-formula via `link(type=...)`.

Supported `type` values:

- Standard: `identity`, `logit`, `probit`, `cloglog`, `sas`, `beta-logistic`
- Blended/mixture inverse link: `blended(a,b,...)` or `mixture(a,b,...)`
  - Components allowed: `probit`, `logit`, `cloglog`, `loglog`, `cauchit`
  - Requires at least 2 unique components
- Flexible mode: `flexible(<single-link>)` or `flexible(blended(...))`

Advanced link parameters:

- `rho=` for blended/mixture
- `sas_init="epsilon,log_delta"` for `sas`
- `beta_logistic_init="epsilon,delta"` for `beta-logistic`

## Fit Modes and Behavior

### 1) Standard mean-only fit

```bash
gam fit train.csv 'y ~ x1 + smooth(x2)' --out model.json
```

- Auto family behavior (when no explicit link override drives it):
  - Binary response (0/1): binomial-logit
  - Otherwise: gaussian-identity

### 2) Location-scale fit (`--predict-noise`)

```bash
gam fit train.csv 'y ~ x1 + smooth(x2)' \
  --predict-noise 'y ~ smooth(x1)' \
  --out locscale.model.json
```

- Gaussian and binomial location-scale are supported.
- For binomial location-scale with probit-style threshold/noise geometry, prediction semantics follow `P(Y=1|S,x)=Phi((S-T(x))/sigma(x))`.
- `linkwiggle(...)` is supported on the mean formula path for binomial location-scale workflows.
- `linkwiggle(...)` is also supported for binomial mean-only fitting with non-flexible binomial links (`logit`, `probit`, `cloglog`, `sas`, `beta-logistic`, `blended(...)`) and for binomial `flexible(...)` mean fitting.

### 3) Survival fit (`Surv(entry, exit, event) ~ ...`)

```bash
gam fit train.csv \
  'Surv(entry_time, exit_time, event) ~ age + smooth(bmi) + survmodel(spec=net, distribution=gaussian)' \
  --survival-likelihood transformation \
  --out surv.model.json
```

Survival likelihood modes (`--survival-likelihood`):

- `transformation` (default)
- `weibull`
- `location-scale`

Survival-specific behavior:

- `survmodel(spec=...)` currently supports `spec=net`.
- `survmodel(distribution=...)` supports `gaussian|gumbel|logistic` (aliases: `probit|cloglog|logit`).
- `--time-basis` for survival currently supports `ispline` in structural mode.
- Weibull likelihood mode uses built-in parametric baseline handling and rejects extra baseline-target parameterization flags.
- `timewiggle(...)` is a baseline-target deformation for survival models.
  Current support: transformation survival with `--baseline-target weibull|gompertz|gompertz-makeham`, and Weibull survival when you supply explicit `--baseline-scale` and `--baseline-shape`.

### Fit-time constraints and incompatibilities

- `--firth` is for binomial-logit mean models and is not supported with `--predict-noise`.
- `--firth` is not supported with `bounded(...)` coefficients.
- `linkwiggle(...)` is a link-deformation feature, not a location-scale-only feature.
  Current support: binomial mean-only non-flexible links, binomial mean-only `flexible(...)`, binomial location-scale, and survival `location-scale`.
- Flexible links are binomial-focused and have restrictions by mode (for example in survival, flexible links require `--survival-likelihood=location-scale`).

## Model I/O and Defaults

- `gam fit ... --out model.json` writes a model.
  - If `--out` is omitted, no model artifact is saved.
- `gam predict ... --out pred.csv` requires `--out`.
- `gam report <MODEL> [DATA] [OUT]`:
  - `[OUT]` omitted -> writes `<model_stem>.report.html`
- `gam sample ...`:
  - `--out` omitted -> `posterior_samples.csv`
- `gam generate ...`:
  - `--out` omitted -> `synthetic.csv`

## Predict Behavior

```bash
gam predict model.json new_data.csv --out pred.csv
```

Options:

- `--uncertainty` adds interval/SE columns
- `--level` sets interval level (default `0.95`)
- `--mode posterior-mean|map` (default `posterior-mean`)
- `--covariance-mode corrected|conditional` (default `corrected`)

Prediction schema by model class:

- Standard models:
  - default: `eta,mean`
  - with uncertainty: `eta,mean,effective_se,mean_lower,mean_upper`
- Gaussian location-scale:
  - default: `eta,mean,sigma`
  - with uncertainty: `eta,mean,sigma,mean_lower,mean_upper`
- Survival:
  - default: `eta,mean,survival_prob,risk_score,failure_prob`
  - with uncertainty: `eta,mean,survival_prob,risk_score,failure_prob,effective_se,mean_lower,mean_upper`

Semantics:

- `sigma` is observation-scale noise SD for Gaussian location-scale models.
- `effective_se` is estimator uncertainty (not observation noise).
- Survival `risk_score` is risk-oriented; `survival_prob` is survival-oriented.

## Report Behavior

```bash
gam report model.json data.csv out.html
```

- Produces a standalone HTML report.
- Includes model summary, coefficient table, EDF blocks, and diagnostic plots.
- With data input, includes residual/fit diagnostics and ALO table where available.
- Without data input, data-dependent diagnostics are omitted with notes.

## Diagnose Behavior

```bash
gam diagnose model.json data.csv --alo
```

- `--alo` is currently the only implemented diagnose mode.
- Output is a terminal table of top leverage rows (`row`, `leverage`, `eta_tilde`, `alo_se`).
- Not supported for models containing `bounded(...)` coefficients.

## Sample Behavior (`NUTS`)

```bash
gam sample model.json data.csv --chains 4 --samples 2000 --warmup 1000 --out posterior_samples.csv
```

- Output CSV columns are `beta_0`, `beta_1`, ...
- Supports standard models and non-location-scale survival paths.
- Not currently available for gaussian/binomial location-scale models.
- Not implemented for survival `location-scale` likelihood in `sample`.

## Generate Behavior

```bash
gam generate model.json data.csv --n-draws 5 --out synthetic.csv
```

- Output CSV columns are `draw_0`, `draw_1`, ...
- Uses deterministic RNG seed (`42`) in current CLI implementation.
- Supports standard and location-scale models.
- Not available for survival models in this command.

## Data and Schema Compatibility

Saved models include training schema/header metadata.

Prediction/report/sample/generate paths load new data with saved schema expectations:

- Column names and types must match training schema.
- Unseen categories are treated as errors.
- Missing training-header metadata in older models can cause feature-mapping errors; refit with current CLI if needed.

## Practical Examples

### Bounded coefficient with prior pull

```bash
gam fit train.csv \
  'y ~ bounded(mu_hat, min=0, max=1, target=0.75, strength=8) + smooth(x)' \
  --out bounded.model.json
```

### Binomial mean-only link wiggle

```bash
gam fit train.csv \
  'y ~ smooth(x1) + link(type=blended(logit,cloglog)) + linkwiggle(degree=3, internal_knots=9)' \
  --out mean_wiggle.model.json
```

### Flexible binomial link with location-scale

```bash
gam fit train.csv \
  'y ~ smooth(x1) + link(type=flexible(logit)) + linkwiggle(degree=3, internal_knots=9)' \
  --predict-noise 'y ~ smooth(x1)' \
  --out flex.model.json
```

### Survival with alternative inverse link family

```bash
gam fit train.csv \
  'Surv(t0, t1, event) ~ age + smooth(bmi) + link(type=blended(logit,cloglog)) + survmodel(spec=net, distribution=gumbel)' \
  --survival-likelihood location-scale \
  --out surv_ls.model.json
```
