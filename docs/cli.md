# CLI reference

The Rust binary is `gam`. It reads CSV or Parquet inputs and writes saved
model blobs, prediction CSVs, posterior draws, generated responses, or HTML
reports.

```bash
gam <command> --help
```

## Commands

| Command | Purpose |
| --- | --- |
| `gam fit DATA FORMULA --out MODEL` | Fit and save a model. |
| `gam predict MODEL NEW_DATA --out PREDICTIONS.csv` | Predict from a saved model. |
| `gam diagnose MODEL DATA [--alo]` | Compute residual / calibration diagnostics; `--alo` also computes approximate leave-one-out quantities. |
| `gam sample MODEL DATA [--out posterior.csv]` | Draw posterior coefficients. |
| `gam generate MODEL DATA [--out generated.csv]` | Draw synthetic responses from a fitted model. |
| `gam report MODEL [DATA] [OUT]` | Write a self-contained HTML report. |

## Fit

```bash
gam fit train.csv 'y ~ s(x) + group(site)' --out model.gam
```

Common options:

| Option | Meaning |
| --- | --- |
| `--family auto|gaussian|binomial-logit|binomial-probit|binomial-cloglog|latent-cloglog-binomial|poisson-log|negative-binomial|gamma-log|tweedie|beta|royston-parmar|transformation-normal` | Explicit response family. `auto` infers from the response. |
| `--negative-binomial-theta VALUE` | Fixed size / overdispersion for negative-binomial fits. |
| `--weights-column COLUMN` | Non-negative per-row likelihood weights. |
| `--offset-column COLUMN` | Additive offset for the primary linear predictor. |
| `--predict-noise RHS` | Secondary right-hand-side formula for scale / dispersion. |
| `--noise-offset-column COLUMN` | Additive offset for the scale / dispersion predictor. |
| `--firth` | Firth bias reduction for supported binomial-logit fits. |
| `--scale-dimensions` | Enable per-axis anisotropy for eligible spatial smooths. |
| `--adaptive-regularization true|false` | Opt into spatial adaptive regularization for compatible standard GAMs. |
| `--transformation-normal` | Fit a conditional transformation-normal model. |

CLI links are declared in the formula with `link(type=...)`; there is no
top-level `gam fit --link` flag.

```bash
gam fit train.csv \
  'case ~ s(age) + link(type=flexible(logit)) + linkwiggle(internal_knots=8)' \
  --family binomial-logit \
  --out case.gam
```

## Survival Fit Options

Use `Surv(entry, exit, event)` on the formula left-hand side:

```bash
gam fit train.csv 'Surv(entry, exit, event) ~ s(age) + bmi' \
  --survival-likelihood transformation \
  --out survival.gam
```

| Option | Meaning |
| --- | --- |
| `--survival-likelihood transformation|weibull|location-scale|marginal-slope|latent|latent-binary` | Survival likelihood mode. CLI default is `transformation`. |
| `--baseline-target linear|weibull|gompertz|gompertz-makeham` | Parametric baseline target. |
| `--baseline-scale`, `--baseline-shape`, `--baseline-rate`, `--baseline-makeham` | Baseline parameter seeds / fixed values where applicable. |
| `--time-basis ispline|none` | Structural survival time basis. `linear` and `bspline` are rejected by the CLI. |
| `--time-degree N`, `--time-num-internal-knots N`, `--time-smooth-lambda VALUE` | I-spline time basis controls (defaults `3`, `8`, `1e-2`). |
| `--threshold-time-k N`, `--sigma-time-k N` | Enable time-varying threshold or scale tensor blocks. |
| `--threshold-time-degree N`, `--sigma-time-degree N` | B-spline degree for the time margin of the threshold / log-sigma tensors (default `3`). |
| `--survival-time-anchor VALUE` | Anchor time for the survival location-scale baseline. |
| `--ridge-lambda VALUE` | Survival solver ridge regularization (default `1e-6`). |
| `--pilot-subsample-threshold N` | Row count above which spatial length-scale optimization uses a pilot subsample (default `10000`). |
| `--logslope-formula RHS`, `--z-column COLUMN` | Marginal-slope score-effect model. |
| `--frailty-kind gaussian-shift|hazard-multiplier`, `--frailty-sd VALUE`, `--hazard-loading full|loaded-vs-unloaded` | Frailty controls. |

## Predict

```bash
gam predict model.gam new.csv --out predictions.csv
gam predict model.gam new.csv --out predictions.csv --uncertainty --level 0.95
```

| Option | Meaning |
| --- | --- |
| `--uncertainty` | Include uncertainty columns where the model supports them. |
| `--level VALUE` | Coverage for uncertainty intervals; default `0.95`. |
| `--covariance-mode conditional|corrected` | Conditional covariance or smoothing-corrected covariance. |
| `--mode posterior-mean|map` | Point-prediction mode. |
| `--no-bias-correction` | Disable the prediction-time `O(n^-1)` bias correction. |
| `--id-column COLUMN` | Carry an identifier column into the prediction CSV. |
| `--offset-column COLUMN`, `--noise-offset-column COLUMN` | Prediction-time offsets matching the fitted model. |

## Sample and Generate

```bash
gam sample model.gam train.csv --chains 4 --samples 1000 --warmup 500 --seed 42
gam generate model.gam new.csv --n-draws 20 --seed 42 --out generated.csv
```

`gam sample` defaults its output to `<model_stem>.posterior.csv` when
`--out` is omitted. `gam generate` defaults to
`<model_stem>.generated.csv`.

## Formula Notes

The CLI and Python API share the same formula DSL:

- `s(...)`, `smooth(...)`, `cyclic(...)`, `te(...)`, `ti(...)`,
  `matern(...)`, `duchon(...)`, `thinplate(...)`, `sphere(...)`,
  `group(...)`, `linear(...)`, `bounded(...)`.
- `link(type=...)`, `linkwiggle(...)`, `timewiggle(...)`, and
  `survmodel(...)` are formula-level configuration terms.
- `--predict-noise`, `--logslope-formula`, and survival options take
  RHS-only formulas. Do not include `y ~` in those arguments.

See [Formula DSL reference](formulas.md), [Families and link
functions](families-and-links.md), and [Survival models](survival.md)
for the model-level details.
