# gam

`gam` is a formula-first CLI and Rust engine for penalized regression models.

The current CLI supports:

- Standard mean models with penalized linear terms, random effects, and smooths
- A surfaced location-scale fitting path via `--predict-noise`
- Survival models via `Surv(entry, exit, event)`
- An advanced Bernoulli marginal-slope workflow via `--logslope-formula` and `--z-column`
- Prediction, HTML reports, ALO diagnostics, posterior sampling, and synthetic-data generation

The CLI is the primary interface. The Rust modules exported by the crate are used internally and can change without compatibility guarantees.

## Requirements

- Rust `1.93+` for local source builds
- CSV input data with a header row
- A shell that supports quoted formulas (`bash`/`zsh` examples below)

## Install

### Prebuilt binary

macOS, Linux, and Windows Git Bash:

```bash
curl -fsSL https://raw.githubusercontent.com/SauersML/gam/main/install.sh | bash
```

### Build from source

```bash
cargo build --release --bin gam
./target/release/gam --help
```

If `gam` is not on your `PATH`, use `./target/release/gam` in the examples below.

## Command Overview

| Command | Purpose | Required arguments | Output |
| --- | --- | --- | --- |
| `gam fit` | Fit a model | `<DATA> <FORMULA>` | No file unless `--out` is provided |
| `gam predict` | Score new data | `<MODEL> <NEW_DATA>` plus `--out` | Prediction CSV |
| `gam report` | Build a standalone HTML report | `<MODEL> [DATA] [OUT]` | `[OUT]` or `<model-stem>.report.html` in the current directory |
| `gam diagnose` | Run terminal diagnostics | `<MODEL> <DATA>` | Prints a diagnostics table |
| `gam sample` | Draw posterior samples | `<MODEL> <DATA>` | `posterior_samples.csv` by default, plus a summary CSV |
| `gam generate` | Sample synthetic outcomes conditional on input rows | `<MODEL> <DATA>` | `synthetic.csv` by default |

Aliases:

- `gam train` -> `gam fit`
- `gam simulate` -> `gam generate`

Inspect full options with:

```bash
gam <command> --help
```

## Verified Quickstart

These commands were checked against the current binary and the checked-in `lidar` dataset.

```bash
# 1) Fit a Gaussian GAM
gam fit bench/datasets/lidar.csv \
  'logratio ~ smooth(range)' \
  --out lidar.model.json

# 2) Predict with uncertainty
gam predict lidar.model.json bench/datasets/lidar.csv \
  --out lidar.pred.csv --uncertainty

# 3) Build an HTML report
gam report lidar.model.json bench/datasets/lidar.csv
# writes: lidar.model.report.html

# 4) Generate synthetic response draws
gam generate lidar.model.json bench/datasets/lidar.csv \
  --n-draws 3 \
  --out lidar.synthetic.csv
```

## Formula Language

`gam fit` expects:

```text
response ~ term + term + ...
```

Response forms:

- Standard regression/classification: `y`
- Survival: `Surv(entry, exit, event)`

Important constraints:

- `Surv(...)` currently requires exactly three columns
- Intercept removal (`0` or `-1`) is not supported
- At most one `link(...)`, one `linkwiggle(...)`, one `timewiggle(...)`, and one `survmodel(...)` may appear in a formula

### Bare RHS terms

A bare column on the right-hand side is interpreted from the training schema:

- Continuous or binary column: penalized linear term
- Categorical column: random-effect block

### Term wrappers

Linear and constrained coefficients:

- `linear(x)`
- `linear(x, min=..., max=...)`
- `constrain(x, min=..., max=...)`
- `nonnegative(x)` / `nonnegative_coef(x)`
- `nonpositive(x)` / `nonpositive_coef(x)`
- `bounded(x, min=..., max=...)`

`bounded(...)` also supports:

- `prior=none|uniform|log-jacobian|center`
- `beta_a=..., beta_b=...`
- `target=..., strength=...`

Random effects:

- `group(x)` or `re(x)`

Smooths:

- `smooth(...)` or `s(...)`
- `thinplate(...)`, `thin_plate(...)`, `tps(...)`
- `matern(...)`
- `duchon(...)`
- `tensor(...)`, `interaction(...)`, `te(...)`

Formula-level configuration terms:

- `link(type=...)`
- `linkwiggle(...)`
- `timewiggle(...)`
- `survmodel(spec=..., distribution=...)`

### Smooth defaults

- `smooth(x)` with one variable defaults to a B-spline / P-spline style basis
- `smooth(x1, x2, ...)` defaults to thin-plate
- `te(...)` defaults to tensor-product B-splines

Notable smooth options:

- B-spline: `degree`, `knots`, `k`, `penalty_order`
- Thin-plate: `centers` or `k`
- Matérn: `centers` or `k`, `nu`, `length_scale`
- Duchon: `centers` or `k`, `power`, `order`, optional `length_scale`
- Tensor: `k` / `basis_dim` for marginal basis size

Spatial smooths can use per-axis anisotropy:

- Global CLI flag: `--scale-dimensions`
- Per-term override: `scale_dims=true` or `scale_dims=false`

## Fit Modes

### 1. Standard mean-only fits

```bash
gam fit train.csv 'y ~ age + smooth(bmi) + group(site)' --out model.json
```

Auto family resolution:

- Binary `{0,1}` response -> binomial logit
- Anything else -> gaussian identity
- `--predict-noise` does not change that default; write `link(type=probit)` (or another explicit link) in the mean formula when you want a different binomial base link

### 2. Location-scale fits

Use a second formula for the scale/noise block:

```bash
gam fit train.csv 'y ~ x1 + smooth(x2)' \
  --predict-noise 'y ~ smooth(x1)' \
  --out locscale.model.json
```

If you want a probit-vs-probit comparison between mean-only and location-scale
fits, declare the link explicitly in both formulas:

```bash
gam fit train.csv 'y ~ x1 + smooth(x2) + link(type=probit)' \
  --out probit.model.json

gam fit train.csv 'y ~ x1 + smooth(x2) + link(type=probit)' \
  --predict-noise 'y ~ smooth(x1)' \
  --out probit.locscale.model.json
```

The CLI exposes this path for Gaussian and binomial families, but current runtime behavior is still uneven enough that you should treat it as experimental and verify it on your exact formula/data combination before relying on it.

### 3. Survival fits

Use `Surv(entry, exit, event)` on the left-hand side:

```bash
gam fit train.csv \
  'Surv(entry_time, exit_time, event) ~ age + smooth(bmi) + survmodel(spec=net, distribution=gaussian)' \
  --survival-likelihood transformation \
  --out survival.model.json
```

Current survival likelihood modes:

- `transformation`
- `weibull`
- `location-scale`

Current survival-specific formula/config support:

- `survmodel(spec=net, distribution=...)`
- `timewiggle(...)`
- `link(...)`
- `linkwiggle(...)` only in supported survival sub-modes

### 4. Bernoulli marginal-slope fits

This is an advanced binary-response mode that adds a second formula for the log-slope surface plus an auxiliary standardized score column:

```bash
gam fit scores.csv \
  'case ~ smooth(age) + matern(pc1, pc2, pc3)' \
  --logslope-formula 'case ~ matern(pc1, pc2, pc3)' \
  --z-column prs_z \
  --out marginal.model.json
```

Current restrictions:

- Response must be binary `{0,1}`
- `--predict-noise` is not allowed
- `--firth` is not allowed
- `link(...)` and `linkwiggle(...)` are not allowed in this family or in `--logslope-formula`

## Link Functions

Links are configured in-formula via `link(type=...)`.

Supported `type` values:

- `identity`
- `logit`
- `probit`
- `cloglog`
- `sas`
- `beta-logistic`
- `blended(a,b,...)` / `mixture(a,b,...)`
- `flexible(<single-link>)`
- `flexible(blended(...))`

Advanced link parameters:

- `rho=` for blended/mixture links
- `sas_init="epsilon,log_delta"`
- `beta_logistic_init="epsilon,delta"`

## Output and Data Semantics

### Saved models

- `gam fit` writes nothing unless `--out` is provided
- Saved model JSON includes training schema and header metadata
- Prediction-like commands reload new data using that saved schema
- If a model predates current metadata requirements, refit it with the current CLI

### Prediction CSV schema

Standard and Bernoulli marginal-slope models:

- default: `eta,mean`
- with `--uncertainty`: `eta,mean,effective_se,mean_lower,mean_upper`

Gaussian location-scale models, when the fit path succeeds:

- default: `eta,mean,sigma`
- with `--uncertainty`: `eta,mean,sigma,mean_lower,mean_upper`

Survival models:

- default: `eta,mean,survival_prob,risk_score,failure_prob`
- with `--uncertainty`: `eta,mean,survival_prob,risk_score,failure_prob,effective_se,mean_lower,mean_upper`

Notes:

- In survival output, `mean` is the same quantity as `survival_prob`
- `risk_score` is risk-oriented and currently tracks the linear predictor direction
- `effective_se` is estimator uncertainty, not observation noise

### Sampling output

`gam sample` writes:

- Raw draws CSV with columns `beta_0`, `beta_1`, ...
- A second summary CSV at `<out with extension summary.csv>`

Defaults when `--out` is omitted:

- Draws: `posterior_samples.csv`
- Summary: `posterior_samples.summary.csv`

Current sampling support:

- Standard models
- Survival models on the non-location-scale path

Not currently available for:

- Gaussian location-scale models
- Binomial location-scale models
- Bernoulli marginal-slope models

### Synthetic generation output

`gam generate` writes a numeric matrix:

- One row per sampled dataset
- One column per conditioning-data row
- Column names are `draw_0`, `draw_1`, ... indexed by input row position

Defaults when `--out` is omitted:

- `synthetic.csv`

Not currently available for:

- Survival models
- Bernoulli marginal-slope models

### Report output

`gam report <MODEL> [DATA] [OUT]` writes:

- `[OUT]` if provided
- Otherwise `<model-stem>.report.html` in the current working directory

The report is standalone HTML. With data input it includes data-dependent diagnostics; without data input those sections are omitted.

### Schema compatibility

Prediction, reporting, sampling, and generation expect the new data to match the saved training schema:

- Column names must match
- Column types must match
- Unseen categorical levels are treated as errors

## Current CLI Limitations

- `diagnose` currently only exposes `--alo`
- `diagnose --alo` is not supported for models containing `bounded(...)` coefficients
- `--predict-noise` is exposed in the CLI, but current Gaussian and binomial location-scale fits still have rough edges; verify behavior on your exact workload before depending on that path
- `linkwiggle(...)` belongs in the mean formula, not `--predict-noise`
- Flexible links are only supported in specific binomial and survival paths
- Some benchmark datasets in `bench/datasets/` are meant for harness scenarios rather than copy-paste README demos

## Development

Common local checks:

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -A warnings -D clippy::correctness -D clippy::suspicious
cargo test --all-features -- --nocapture
```

Benchmark harness:

```bash
python3 bench/run_suite.py --help
python3 bench/run_suite.py
```

Repository layout:

- `src/`: CLI, model code, fitting/inference, smooth construction, and survival machinery
- `bench/`: benchmark harness, scenario configs, datasets, and comparison tooling
- `tests/`: Rust integration tests plus benchmark helper tools

Lean checks for the Rust-matched `.lean` files under `src/`:

```bash
./scripts/lean-check-all.sh
```
