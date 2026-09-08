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
| `gam crosscoder --anchor L=F --block L=F --atoms N --harmonics N --out REPORT.json` | Fit a row-aligned manifold crosscoder across activation matrices and write a GAM-SAE report. |
| `gam transformation-score MODEL DATA [--out scores.csv]` | Evaluate a fitted conditional transformation model at observed responses. |

Every subcommand also accepts `--log-level off|error|warn|info|debug|trace`
(default `warn`), and the shorthands `-v`/`-vv`/`-vvv` (increasing verbosity)
and `-q` (quiet). `--log-level` wins if both are given.

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
| `--noise-offset-column COLUMN` | Additive offset for the scale / dispersion predictor; under the marginal-slope families, for the slope predictor (`--offset-column` then offsets the marginal predictor). |
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
| `--time-degree N`, `--time-num-internal-knots N` | I-spline time basis controls (defaults `3`, `8`). The time-basis smoothing parameter is estimated by REML; its search seed is the library's, not an option. |
| `--threshold-time-k N`, `--sigma-time-k N` | Enable time-varying threshold or scale tensor blocks. |
| `--slope-time-k N` | Let the marginal-slope effect vary along follow-up: tensors the slope design against a B-spline margin in `log(time)`. |
| `--threshold-time-degree N`, `--sigma-time-degree N` | B-spline degree for the time margin of the threshold / log-sigma tensors (default `3`). |
| `--survival-time-anchor VALUE` | Centering anchor for the baseline time basis, in the data's own time units, honored by every survival likelihood. Omit it to let the fit choose: the robust interior median exit for `marginal-slope` and for any genuinely left-truncated dataset (any row entering above the time origin), the earliest entry age otherwise. Re-centering is an exact affine reparameterization of the baseline offset, so this picks the frame the smoothing selection sees, not the model. Also settable as `survival_time_anchor` in a `--request` document and as `survival_time_anchor=` in `gamfit.fit`. |
| `--slope-formula RHS`, `--z-column COLUMN` | Marginal-slope score-effect model. |
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
| `--covariance-mode conditional|corrected` | Conditional covariance or smoothing-corrected covariance. Absent, the definition the saved fit publishes (the one `gam summary` prices its standard errors from) is used and labeled; naming one is a requirement that refuses when the fit cannot supply it. |
| `--mode posterior-mean|map` | Point-prediction mode. |
| `--no-bias-correction` | Disable the `O(n^-1)` frequentist bias correction in the survival uncertainty paths. The standard `posterior_mean` point prediction is never moved by this flag. |
| `--id-column COLUMN` | Carry an identifier column into the prediction CSV. |
| `--offset-column COLUMN`, `--noise-offset-column COLUMN` | Prediction-time offsets matching the fitted model. |

Standard and location-scale mean models write an estimand-explicit CSV. The
default `--mode posterior-mean` columns are `linear_predictor_plugin`,
`mean_plugin`, and `posterior_mean`; location-scale models that expose a fitted
response-side scale add `noise_scale`. With `--uncertainty`, the posterior
columns are `posterior_mean_standard_error`, `posterior_mean_lower`, and
`posterior_mean_upper`. A point-only `--mode map` emits only the plug-in pair
(plus `noise_scale` when present), so one column name never changes estimand
with the mode. Combining `--mode map` with `--uncertainty` retains
`posterior_mean` because the named posterior uncertainty columns require their
posterior point; the plug-in pair remains explicit alongside it.
Transformation-normal, marginal-slope, and survival predictions retain their
model-specific schemas.

## Sample and Generate

```bash
gam sample model.gam train.csv --chains 4 --samples 1000 --warmup 500 --seed 42
gam generate model.gam new.csv --n-draws 20 --seed 42 --out generated.csv
```

`gam sample` defaults its output to `<model_stem>.posterior.csv` when
`--out` is omitted. `gam generate` defaults to
`<model_stem>.generated.csv`.

## Transformation Score

For a fitted `--transformation-normal` model, evaluate the latent score
`h(y|x)` at observed responses (rather than predicting `y` itself):

```bash
gam transformation-score model.gam labelled.csv --out scores.csv
```

`--offset-column` and `--id-column` behave as in `predict`.

## Crosscoder

Fit a row-aligned manifold crosscoder across two or more activation
matrices (e.g. matched-row hidden states from different model layers)
and write a GAM-SAE report:

```bash
gam crosscoder \
  --anchor base=layer0.npy \
  --block  tuned=layer1.npy \
  --atoms 8 --harmonics 4 \
  --out crosscoder_report.json
```

| Option | Meaning |
| --- | --- |
| `--anchor LABEL=FILE` | Named anchor activation matrix (2-D `.npy`, floating point). |
| `--block LABEL=FILE` | Named non-anchor activation matrix, row-aligned with the anchor. Repeat once per additional layer. |
| `--atoms N` | Number of shared manifold atoms. |
| `--harmonics N` | Harmonic order of each periodic manifold atom. |
| `--sparsity-strength`, `--smoothness`, `--max-iter`, `--learning-rate`, `--ridge-ext-coord`, `--ridge-beta`, `--random-state` | Overrides for the underlying Rust library defaults. |
| `--transport-grid-resolution N`, `--law-gap-tolerance VALUE` | Grid resolution and tolerance for classifying consecutive-layer transport. |

## Formula Notes

The CLI and Python API share the same formula DSL:

- `s(...)`, `smooth(...)`, `cyclic(...)`, `te(...)`, `ti(...)`,
  `matern(...)`, `duchon(...)`, `thinplate(...)`, `sphere(...)`,
  `group(...)`, `linear(...)`, `bounded(...)`.
- `link(type=...)`, `linkwiggle(...)`, `timewiggle(...)`, and
  `survmodel(...)` are formula-level configuration terms.
- `--predict-noise`, `--slope-formula`, and survival options take
  RHS-only formulas. Do not include `y ~` in those arguments.

See [Formula DSL reference](formulas.md), [Families and link
functions](families-and-links.md), and [Survival models](survival.md)
for the model-level details.
