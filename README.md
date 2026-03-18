# gam

`gam` is a formula-first CLI and Rust engine for generalized additive models.

The current CLI supports:

- Standard models with linear terms, random effects, and smooths
- Location-scale models (which jointly predict noise)
- Survival models
- Various surface models
- Posterior sampling
- Rich penalty structures
- And more

## Differences from other GAM libraries
- We primarily use REML and LAML.
- We aim to automatically support all sizes of data.
- The default penalty structure has separate penalties for magnitude, gradient, and curvature, instead of the common approach of having one (curvature) or two (curvature, and magnitude + gradient combined).
- Support for using smooths as offsets from a known function, such that the known function acts as a prior. For example, you may choose to select a probit link with a spline offset, allowing the data to fix link misspecification, while still encoding the belief that probit is approximately correct. This is particularly useful because a flexible link function can fix arbitrary miscalibration of the model, similar to a post-hoc calibration step (but jointly fit). There's also support for smooth offsets from survival model time basis functions.
- Support for surface smooths, such as multidimensional Duchon splines.
- Adaptive length scaling: the ability to shrink each axis or feature separately, even when they are part of a single, joint smooth.
- Support for a locally adaptive smoothness penalty, so certain regions can demand faster transitions, while still remaining smooth overall.
- The ability to mix aspects of different spline categories together. For example, you can have the kernel basis of a Duchon spline, but also have the global kappa scaling of a Matern spline (if you for some reason wanted to).

## Install

### Prebuilt binary

macOS, Linux, and Windows Git Bash:

```bash
curl -fsSL https://raw.githubusercontent.com/SauersML/gam/main/install.sh | bash
```

### Build from source

Requires [Rust](https://rustup.rs/).

```bash
git clone https://github.com/SauersML/gam.git
cd gam
cargo build --release
```

The binary is at `./target/release/gam`. Add it to your `PATH` or use the full path in the examples below.

## Quick start

```bash
# Fit a GAM with a smooth term
gam fit data.csv 'y ~ smooth(x)' --out model.json

# Predict with uncertainty intervals
gam predict model.json new_data.csv --out predictions.csv --uncertainty

# Build a standalone HTML report
gam report model.json data.csv

# Draw posterior samples
gam sample model.json data.csv --out samples.csv

# Generate synthetic response draws
gam generate model.json data.csv --n-draws 5 --out synthetic.csv
```

## Commands

| Command | What it does | Usage |
| --- | --- | --- |
| `fit` | Fit a model | `gam fit <DATA> <FORMULA> [--out model.json]` |
| `predict` | Score new data | `gam predict <MODEL> <DATA> --out predictions.csv` |
| `report` | Standalone HTML report | `gam report <MODEL> [DATA] [OUT]` |
| `diagnose` | Terminal diagnostics | `gam diagnose <MODEL> <DATA>` |
| `sample` | Posterior draws (NUTS) | `gam sample <MODEL> <DATA> [--out samples.csv]` |
| `generate` | Synthetic outcomes | `gam generate <MODEL> <DATA> [--out synthetic.csv]` |

`train` is an alias for `fit`. `simulate` is an alias for `generate`.

Run `gam <command> --help` for full options.

## Formula language

```
response ~ term + term + ...
```

### Response

- Continuous or binary: `y`
- Survival: `Surv(entry_time, exit_time, event)`

### Terms

**Linear and constrained coefficients:**

| Syntax | Effect |
| --- | --- |
| `x` or `linear(x)` | Penalized linear term |
| `linear(x, min=0)` | Non-negative coefficient |
| `linear(x, min=..., max=...)` | Box-constrained coefficient |
| `nonnegative(x)` | Sugar for `linear(x, min=0)` |
| `nonpositive(x)` | Sugar for `linear(x, max=0)` |
| `bounded(x, min=0, max=1)` | Exact interval transform (no ridge) |
| `bounded(x, ..., prior=uniform)` | Flat prior on bounded scale |
| `bounded(x, ..., target=0.5, strength=3)` | Informative interior prior |

**Random effects:**

| Syntax | Effect |
| --- | --- |
| `group(id)` or `re(id)` | Random intercept per level of `id` |

**Smooths:**

| Syntax | Default basis |
| --- | --- |
| `smooth(x)` or `s(x)` | P-spline (B-spline + difference penalty) |
| `smooth(x1, x2)` | Thin-plate spline |
| `thinplate(x1, x2)` or `tps(x1, x2)` | Thin-plate spline |
| `matern(x1, x2, ...)` | Matern radial basis |
| `duchon(x1, x2, ...)` | Duchon spline (scale-free) |
| `tensor(x, z)` or `te(x, z)` | Tensor-product B-splines |

Common smooth options: `knots=`, `k=`, `centers=`, `degree=`, `penalty_order=`, `double_penalty=true|false`, `type=ps|tps|matern|duchon`.

Spatial smooths support per-axis anisotropy via `scale_dims=true` or the global `--scale-dimensions` flag.

**Formula-level configuration:**

| Syntax | Effect |
| --- | --- |
| `link(type=logit)` | Set link function |
| `linkwiggle(knots=10)` | Spline deviation from the base link |
| `timewiggle(knots=8)` | Spline deviation from the time basis (survival) |
| `survmodel(spec=net, distribution=gaussian)` | Survival model configuration |

### Auto-detection

- Binary `{0, 1}` response: binomial with logit link
- Everything else: Gaussian with identity link
- Override with `link(type=...)` in the formula

## Fit modes

### Standard

```bash
gam fit data.csv 'y ~ age + smooth(bmi) + group(site)' --out model.json
```

### Location-scale (jointly model noise)

```bash
gam fit data.csv 'y ~ smooth(x1) + smooth(x2)' \
  --predict-noise 'y ~ smooth(x1)' \
  --out model.json
```

Works for Gaussian and binomial families. For survival formulas, `--predict-noise` routes to the survival location-scale fitter.

### Survival

```bash
gam fit data.csv \
  'Surv(t0, t1, event) ~ age + smooth(bmi) + survmodel(spec=net, distribution=gaussian)' \
  --survival-likelihood transformation \
  --out model.json
```

Likelihood modes: `transformation`, `weibull`, `location-scale`.

Add `--predict-noise` for distributional (location-scale) survival:

```bash
gam fit data.csv \
  'Surv(t0, t1, event) ~ age + smooth(bmi) + survmodel(spec=net, distribution=gaussian)' \
  --predict-noise 'Surv(t0, t1, event) ~ smooth(age)' \
  --out model.json
```

### Bernoulli marginal-slope

Models `P(case | covariates, z)` where `z` is a standardized score (e.g. a polygenic risk score). The key idea: the baseline risk surface and the effect of `z` are decoupled into separate formulas. The main formula controls the population-level risk landscape (how risk varies with age, ancestry PCs, etc.), while `--logslope-formula` controls how strongly `z` modifies that risk at each point in covariate space. This decoupling lets you estimate spatially-varying effect sizes for `z` without the baseline absorbing signal that belongs to the slope, or vice versa.

```bash
gam fit data.csv \
  'case ~ smooth(age) + matern(pc1, pc2, pc3)' \
  --logslope-formula 'case ~ matern(pc1, pc2, pc3)' \
  --z-column prs_z \
  --out model.json
```

## Link functions

Set via `link(type=...)` in the formula.

| Link | Syntax |
| --- | --- |
| Identity | `link(type=identity)` |
| Logit | `link(type=logit)` |
| Probit | `link(type=probit)` |
| Complementary log-log | `link(type=cloglog)` |
| SAS (sinh-arcsinh) | `link(type=sas)` |
| Beta-logistic | `link(type=beta-logistic)` |
| Blended mixture | `link(type=blended(logit, probit))` |
| Flexible (data-driven) | `link(type=flexible(logit))` |

Flexible links add a spline offset to the base link, letting the data correct for link misspecification.

## Prediction output

| Model type | Default columns | With `--uncertainty` |
| --- | --- | --- |
| Standard / binomial | `eta, mean` | `+ effective_se, mean_lower, mean_upper` |
| Gaussian location-scale | `eta, mean, sigma` | `+ mean_lower, mean_upper` |
| Survival | `eta, mean, survival_prob, risk_score, failure_prob` | `+ effective_se, mean_lower, mean_upper` |

## Other outputs

**`gam report`** writes a standalone HTML file with model summary, smooth plots, and diagnostics. Pass training data for data-dependent diagnostics.

**`gam sample`** writes posterior draws (`beta_0, beta_1, ...`) and a summary CSV. Uses NUTS (No-U-Turn Sampler).

**`gam generate`** writes a matrix of synthetic outcomes (rows = draws, columns = data rows).

**`gam diagnose`** prints terminal diagnostics. Supports `--alo` for approximate leave-one-out.

## Development

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -A warnings -D clippy::correctness -D clippy::suspicious
cargo test --all-features
```

Benchmark suite:

```bash
python3 bench/run_suite.py --help
python3 bench/run_suite.py
```

Layout:

- `src/` -- CLI, fitting engine, inference, smooth construction, survival machinery
- `bench/` -- benchmark harness, scenario configs, datasets, comparison tooling
- `tests/` -- integration tests
