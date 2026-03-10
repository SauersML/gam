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

## Quickstart

Use one of the included benchmark datasets:

```bash
# 1) Fit and save a model
gam fit bench/datasets/wine.csv \
  'price ~ year + rain + smooth(temp)' \
  --out wine.model.json

# 2) Predict on new data (or the same file)
gam predict wine.model.json bench/datasets/wine.csv \
  --out wine.pred.csv --uncertainty

# 3) Render an HTML report
gam report wine.model.json bench/datasets/wine.csv
# writes: wine.model.report.html
```

If you build locally and `gam` is not on PATH, use `./target/release/gam` in the commands above.

## CLI Commands

```text
gam fit
gam predict
gam report
gam diagnose
gam sample
gam generate
```

Inspect command options with:

```bash
gam <command> --help
```

## Formula Notes

`fit` takes `<DATA> <FORMULA>` where formulas are of the form `y ~ terms`.

Common term wrappers:
- `smooth(x)`, `thinplate(x1, x2)`, `matern(pc1, pc2, ...)`, `duchon(...)`, `tensor(x, z)`
- `group(site)` for random effects
- `linear(x, min=..., max=...)`, `constrain(x, ...)`, `nonnegative(x)`, `nonpositive(x)`
- `bounded(x, min=..., max=...)` with optional priors like `prior="uniform"`

Examples:
- `y ~ age + smooth(bmi) + group(site)`
- `y ~ nonnegative(effect) + smooth(bmi)`
- `y ~ bounded(mu_hat, min=0, max=1) + matern(pc1, pc2, pc3)`
- `y ~ s(pc1, pc2, type=duchon, centers=12)`
- `y ~ s(pc1, pc2, type=duchon, centers=12, length_scale=0.7)`

## Output Files and Defaults

- `gam fit ... --out model.json` saves a model. If `--out` is omitted, no model file is written.
- `gam predict ... --out pred.csv` requires `--out`.
- `gam report <MODEL> [DATA] [OUT]` writes `[MODEL_STEM].report.html` when `[OUT]` is omitted.
- `gam sample` defaults to `posterior_samples.csv` if `--out` is omitted.
- `gam generate` defaults to `synthetic.csv` if `--out` is omitted.

## Prediction CSV Schemas

Standard models:
- default: `eta,mean`
- with `--uncertainty`: `eta,mean,effective_se,mean_lower,mean_upper`

Gaussian location-scale fits (`fit --predict-noise ...`):
- default: `eta,mean,sigma`
- with `--uncertainty`: `eta,mean,sigma,mean_lower,mean_upper`

Survival models:
- default: `eta,mean,survival_prob,risk_score,failure_prob`
- with `--uncertainty`: `eta,mean,survival_prob,risk_score,failure_prob,effective_se,mean_lower,mean_upper`

Notes:
- `sigma` is the observation-scale standard deviation for Gaussian location-scale predictions.
- `effective_se` is prediction estimator uncertainty (not the same as `sigma`).
- For survival predictions, `risk_score` is risk-oriented and `survival_prob` is survival-oriented.

## Common Workflows

### Binomial/Gaussian with uncertainty

```bash
gam fit train.csv 'y ~ x1 + smooth(x2)' --out model.json
gam predict model.json test.csv --out pred.csv --uncertainty --level 0.95
```

### Gaussian location-scale

```bash
gam fit train.csv 'y ~ smooth(x1) + x2' \
  --predict-noise 'y ~ smooth(x1)' \
  --out locscale.model.json

gam predict locscale.model.json test.csv --out locscale.pred.csv --uncertainty
```

### Survival (`Surv(...)` response)

```bash
gam fit train.csv \
  'Surv(entry_time, exit_time, event) ~ age + smooth(bmi)' \
  --survival-likelihood transformation \
  --out surv.model.json

gam predict surv.model.json test.csv --out surv.pred.csv
```

## Project Layout

- `src/`: core library and CLI implementation (`src/main.rs`)
- `bench/`: benchmark harness, datasets, and analysis utilities
- `tests/`: integration and regression tests
- `scripts/`: helper scripts for benchmarking and analysis

For benchmark-specific usage, see [`bench/README.md`](bench/README.md).
