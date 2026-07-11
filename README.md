# gam / gamfit

[![PyPI](https://img.shields.io/pypi/v/gamfit.svg)](https://pypi.org/project/gamfit/)
[![Python](https://img.shields.io/pypi/pyversions/gamfit.svg)](https://pypi.org/project/gamfit/)
[![Docs](https://img.shields.io/readthedocs/gamfit.svg)](https://gamfit.readthedocs.io/)
[![Rust CI](https://github.com/SauersML/gam/actions/workflows/test.yml/badge.svg)](https://github.com/SauersML/gam/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue.svg)](LICENSE)

A generalized additive model engine. The fitting code is in Rust; the
public interfaces are a Rust CLI (`gam`) and a Python package (`gamfit`),
which share one engine, one formula DSL, and one on-disk model format.

Docs: <https://gamfit.readthedocs.io/>. PyPI: <https://pypi.org/project/gamfit/>.

Contributions of every kind are welcome.

![3D Matérn fit on a noisy 2-D landscape](docs/images/surface_3d_wireframe.png)

## Scope

Supported response families: Gaussian, binomial / Bernoulli (including a
marginal-slope variant for calibrated risk scores, and a latent-cloglog
form), Poisson, negative-binomial, Gamma, Beta, Tweedie, multinomial-logit,
conditional transformation-normal, Royston-Parmar, and parametric /
semi-parametric survival in six likelihood modes (transformation, Weibull,
location-scale, marginal-slope, and latent-Gaussian frailty in `latent`
and `latent-binary` forms). Firth /
Jeffreys bias reduction handles separation in binomial fits.

Supported term types in formulas: parametric terms, univariate smooths
(`s`), tensor-product smooths (`te`, `ti`), radial smooths in arbitrary
dimension (`matern`, `duchon`, `thinplate`), intrinsic manifold smooths
(`sphere`, periodic / cyclic, torus, cylinder via `te(..., periodic=...)`),
measure-jet (`mjs`) and constant-curvature (`curv`) smooths, random effects
and factor smooths (`group`, `fs`, `sz`, `by=`), shape-constrained smooths
(monotone / convex / concave), interval-bounded coefficients (`bounded`),
and learnable links (`link(type=flexible(...))`,
`link(type=blended(...))`, `sas`, `beta-logistic`, `linkwiggle`).

Smoothing parameters are selected by REML or LAML. Posterior sampling
uses NUTS over the coefficient posterior conditional on the fitted
smoothing parameters where the family supports it, and a Gaussian
Laplace approximation otherwise.

The engine also provides, past fitting and point prediction (see
[Examples](#examples) and the [docs](https://gamfit.readthedocs.io/)):

- **Prediction intervals** — Wald and delta-method bands, conformal
  intervals (split, jackknife+, and full conformal), and posterior draws
  via NUTS or a Laplace approximation, with posterior-predictive checks.
- **Model comparison** — per-term Wald and likelihood-ratio tests, and
  AIC / approximate-leave-one-out comparison via `compare_models`.
- **Difference smooths** — covariance-aware contrasts between by-group
  smooths, with optional simultaneous bands.
- **Diagnostics and reports** — `summary` (coefficient table, effective
  degrees of freedom), `diagnose` (residuals and fit metrics), residual and
  partial-effect `plot`, schema `check`, and a self-contained HTML `report`.
- **Manifold-valued responses** — responses on the sphere, simplex,
  SPD-matrix cone, Grassmann, Stiefel, or hyperbolic ball, fit in the
  tangent space at the Fréchet mean; a constant-curvature family estimates
  the curvature.
- **Sparse manifold dictionaries (SAE)** — decompose a matrix into K sparse
  atoms, each a low-dimensional curve or surface (line, circle, sphere,
  torus) with a per-row coordinate, with a differentiable `gamfit.torch`
  version, steering, crosscoders, and cross-layer transport.
- **Integration and scale** — scikit-learn estimators, differentiable
  `gamfit.torch` REML and basis primitives, pandas / polars / pyarrow /
  numpy and CSV / Parquet inputs, O(n) / O(n log n) solver paths for large
  univariate and 2-3D spatial smooths, streamed prediction, and optional
  CUDA.

## Install

Python wheels are published for Linux (x86_64, aarch64), macOS (Intel
and Apple silicon), and Windows. A Rust toolchain is not required.

```bash
uv add gamfit
# or
pip install gamfit
```

Optional extras: `gamfit[pandas]`, `gamfit[plot]`, `gamfit[sklearn]`,
`gamfit[cuda]`, `gamfit[all]`. PyTorch is optional but is installed as
the `torch` package itself; there is no `gamfit[torch]` package extra.

For the Rust CLI:

```bash
curl -fsSL https://raw.githubusercontent.com/SauersML/gam/main/install.sh | bash
```

Or `cargo build --release`. The binary is `./target/release/gam`.

## Usage

Python:

```python
import gamfit
model = gamfit.fit(train, "y ~ s(x) + group(site)")
preds = model.predict(test, interval=0.95)
```

CLI:

```bash
gam fit data.csv 'y ~ smooth(x) + group(site)' --out model.json
gam predict model.json new_data.csv --out predictions.csv --uncertainty
gam report model.json data.csv
```

CLI subcommands: `fit`, `predict`, `report`, `diagnose`, `sample`,
`generate`. Run `gam <command> --help` for options.

## Examples

Surface smooths in arbitrary dimension, with optional per-axis length
scales:

```python
gamfit.fit(df, "y ~ matern(x1, x2, x3, nu=5/2)")
gamfit.fit(df, "y ~ duchon(x1, x2, x3, x4, centers=80)")
gamfit.fit(df, "y ~ te(space, time, k=10)")
gamfit.fit(df, "z ~ matern(pc1, pc2, pc3, pc4)", scale_dimensions=True)
```

Smooths on manifolds. The basis and penalty encode the wrap topology,
so a fit on `theta ∈ [0, 2π)` has no seam at 0 / 2π, and an `S²` fit
has no pole artefacts.

```python
gamfit.fit(df, "y ~ s(theta, periodic=true, period=2*pi)")
gamfit.fit(df, "y ~ te(theta, h, periodic=[0], period=[2*pi, None])")
gamfit.fit(df, "y ~ te(u, v, periodic=[0,1], period=[2*pi, 2*pi])")
gamfit.fit(df, "y ~ sphere(lat, lon, radians=true)")
gamfit.fit(df, "y ~ s(x, bc=clamped)")
```

![rotating recovery of six geometric examples (trefoil knot, latent-free loop, wobbly cylinder, lumpy sphere, bumpy torus, Möbius double-cover) from noisy 3-D point clouds](docs/images/geometric_shapes_demo.gif)

Each pair shows the noisy input (left) and the recovered smooth
(right). The full gallery and reproduction script:
[docs/manifold-smooths.md](docs/manifold-smooths.md).

Manifold SAE dictionary. `sae_manifold_fit` decomposes an activation /
embedding matrix into `K` sparse atoms, each a low-dimensional typed shape
(line, circle, sphere, torus, or Euclidean patch) with a per-token
coordinate. Cross-atom decoder incoherence (on by default) keeps co-firing
atoms separable; the IBP gate adapts the number of active atoms per token
with true zeros. A fresh fit reports, per atom, a closed-form posterior
shape band (mean curve ± sd) and the typical coordinate range the atom is
used over — where each manifold lives, what shape, and how confident.

```python
fit = gamfit.sae_manifold_fit(X=acts, K=16, d_atom=1, atom_topology="circle")
recon = fit.predict(acts)              # (N, p) reconstruction
band = fit.shape_uncertainty(0)        # {"coords","mean","sd","lower","upper"}
extent = fit.coords[0].min(0), fit.coords[0].max(0)   # where atom 0 lives
plan = fit.steer(atom_k=0, t_from=0.0, t_to=1.0)      # steering + dosimetry
```

The dictionary supports three gating families (`assignment="ordered_beta_bernoulli"`,
`"softmax"`, `"jumprelu"`) and a `gamfit.torch.ManifoldSAE` autograd
mirror with out-of-sample encoder distillation. Around it: `select_topology`
to choose an atom's shape by evidence; `sae_checkpoint_dynamics` to track
atoms across training checkpoints; `gamfit.crosscoder.Crosscoder` and
`layer_transport_fit` / `layer_transport_ladder` for cross-layer
dictionaries; and `gamfit.identifiability` factor-recovery diagnostics.

Details: [docs/manifold-sae.md](docs/manifold-sae.md).

Learnable link functions. A `flexible(base)` link adds a spline offset
on top of a base link. `blended(l1, l2)` learns a mixture weight. `sas`
and `beta-logistic` learn shape parameters.

```python
gamfit.fit(df, "case ~ s(age) + link(type=flexible(probit))"
                 " + linkwiggle(internal_knots=6)")
```

Marginal-slope models for binary or survival outcomes with a calibrated
risk score. The baseline and the score effect are fit in separate
formulas; the score effect is a smooth function of covariate space.
Supply a `transformation_normal_stage1=` recipe to condition the score on
covariates and cross-fit it inside the one call (a Neyman-orthogonal chain
whose slope surface is insensitive to Stage-1 calibration error); no
`z_column` is materialised by hand.

```python
gamfit.fit(
    df,
    "case ~ matern(pc1, pc2, pc3)",
    family="bernoulli-marginal-slope",
    logslope_formula="matern(pc1, pc2, pc3)",
    transformation_normal_stage1=gamfit.CtnStage1(
        response="pgs",
        covariates="matern(pc1, pc2, pc3)",
    ),
)
```

![two predicted-probability surfaces over a (pc1, pc2) plane, at z = 0 and z = +2](docs/images/marginal_slope_3d.png)

Survival models. `Surv(entry, exit, event)` is supported in several
likelihood modes: transformation, Weibull, location-scale,
marginal-slope, and latent-Gaussian frailty (`latent`,
`latent-binary`). `model.predict(...)` returns a `SurvivalPrediction`
with on-demand `S(t)`, `h(t)`, `H(t)` on any time grid:

```python
pred = model.predict(test_df)
S = pred.survival_at([1, 5, 10, 20])
H = pred.cumulative_hazard_at([10])
pred.write_survival_at_csv("surv.csv", times=[...])  # streamed
```

Posterior sampling. `model.sample(...)` draws from the coefficient
posterior conditional on the fitted smoothing parameters. Predictive
bands are computed in row chunks.

```python
posterior = model.sample(train, seed=42)
bands = posterior.predict(test, level=0.95)
```

Interval-bounded coefficients with an optional Beta prior:

```python
gamfit.fit(df,
    "y ~ age + bounded(prop, min=0, max=1, target=0.5, strength=3)")
```

Shape-constrained smooths. A smooth can be required to be monotone or
convex/concave:

```python
gamfit.fit(df, "y ~ s(dose, shape=monotone_increasing)")
gamfit.fit(df, "y ~ s(x, shape=convex)")
```

Difference smooths. `by=` factor smooths fit one curve per group, and
`model.difference_smooth(...)` returns the covariance-aware contrast
between two groups (with an optional simultaneous band):

```python
model = gamfit.fit(df, "y ~ s(x, by=group)")
diff = model.difference_smooth(data=df, group="group", view="x",
                               simultaneous=True)
```

Dispersion (location-scale) GAMLSS. A second formula models the scale /
variance for Gamma, Beta, negative-binomial, and Tweedie:

```python
gamfit.fit(df, "y ~ s(x)", family="gamma", noise_formula="s(x)")
```

Conformal prediction intervals. `interval="conformal"` gives
distribution-free jackknife+ bands (Gaussian-identity); other families
use split conformal via `predict_conformal`:

```python
model.predict(test, interval="conformal", conformal_level=0.9)
```

Competing-risks survival. `competing_risks_cif(...)` and
`CompetingRisksPrediction` evaluate cause-specific cumulative-incidence
functions.

Manifold-valued responses. `ResponseGeometryModel` fits GAMs for
responses that live on a manifold, mapped to a tangent space at the
Fréchet mean: the sphere, the simplex (`clr` / `alr`), the cone of SPD
matrices, the Grassmann and Stiefel manifolds, and the hyperbolic
(Poincaré) ball. A constant-curvature family estimates the curvature from
the responses:

```python
gamfit.fit(df, "y ~ s(x)", response_geometry="poincare", response_columns=[...])
gamfit.fit(df, "y ~ s(x)", response_geometry="constant_curvature", response_columns=[...])
```

Model comparison. `compare_models` reports AIC and approximate
leave-one-out (elpd), corrected for smoothing-parameter selection.

```python
gamfit.compare_models([model_a, model_b])
```

Diagnostics and reports. `model.summary()` gives the coefficient table and
per-term effective degrees of freedom; `model.diagnose(data)` returns
residuals and fit metrics; `model.plot(...)` draws partial effects and
residuals; `model.report("out.html")` writes a self-contained HTML report.

PyTorch bridge. Differentiable REML primitives, smooth-basis layers, and
frozen fitted-model modules are available under `gamfit.torch` and
`gamfit.kernels` when `torch` is installed.

scikit-learn wrappers:

```python
from gamfit.sklearn import GAMRegressor
est = GAMRegressor(formula="y ~ s(x)").fit(X, y)
```

## Penalties

A smooth term contributes one or more penalized coefficient blocks,
each with its own smoothing parameter selected by REML/LAML. For
polyharmonic and Duchon radial bases, three penalty operators act on
the same coefficient block: mass (L²), tension (gradient), and
stiffness (Laplacian/curvature). P-spline, thin-plate, and tensor-product
smooths use their standard derivative-based penalties.

## GPU

The Rust engine includes optional CUDA support (cuBLAS, cuSOLVER,
cuSPARSE). It loads lazily and falls back to the CPU path when no
working CUDA stack is found. The same wheel runs on CPU-only and GPU
hosts.

Per-op dispatch thresholds are measured at probe time from GPU FP64
throughput, CPU FP64 throughput, and PCIe bandwidth. Below the
crossover where transfer cost dominates, kernels stay on the CPU. To
print the calibrated thresholds:

```python
import gamfit
print(gamfit.format_cuda_diagnostics())
```

The wheel is compiled against the CUDA 12 driver/userspace ABI. If PyTorch has
already mapped a complete CUDA stack, gamfit continues that exact stack; it
does not preload a second system toolkit into the process. Otherwise it loads
one complete system or packaged NVIDIA stack. A process whose mapped CUDA
libraries do not belong to one complete stack is refused by the GPU probe
instead of mixing context or handle ownership across implementations.

## Repository layout

| Path | Contents |
| --- | --- |
| `crates/gam-*/` | Rust engine, split across ~20 workspace crates: fitting/solve (`gam-solve`), inference (`gam-inference`), families/models (`gam-models`, `gam-model-api`), smooth construction (`gam-terms`, `gam-geometry`), manifold SAE (`gam-sae`), prediction (`gam-predict`), GPU (`gam-gpu`), reports (`gam-report`), CLI (`gam-cli`), and more. |
| `crates/gam-pyffi/` | PyO3 bindings (`gamfit._rust`). |
| `src/` | Thin workspace-root shell (`lib.rs`, shared types, macros) over the crates. |
| `gamfit/` | Python public API on top of the bindings. |
| `docs/` | MkDocs/Material documentation sources. |
| `tests/` | Rust and Python integration tests. |
| `bench/` | Benchmark harness, configs, datasets, plots. |
| `examples/` | Python and Rust demos, including SAE and topology examples. |
| `scripts/` | Documentation figure, gallery, and diagnostic scripts. |

## Documentation

- Full Python documentation: <https://gamfit.readthedocs.io/>.
- Cookbook: [docs/cookbook.md](docs/cookbook.md).
- Manifold smooths gallery: [docs/manifold-smooths.md](docs/manifold-smooths.md).
- Manifold SAE dictionary: [docs/manifold-sae.md](docs/manifold-sae.md).

## Contributing

This is meant to be one of the easiest projects anywhere to contribute
to — on purpose, not by neglect.

The correctness bar is high: smooths are checked against analytic oracles
and finite-difference jets, and CI is strict. But that bar is on the
merged result, and keeping it there is the maintainer's job — not a toll
you pay to take part. So the bar to *contribute* is zero:

- **Broken PRs are welcome.** Fails CI, half-finished, you're not sure
  it's right — open it anyway. A broken PR with a good idea in it beats a
  good idea that never got sent.
- **Beginners welcome.** You don't need REML, Rust, or PyO3 to help. A
  confusing error message, a typo, a docs gap, or "why does it do this?"
  is a real contribution.
- **AI is allowed.** Wrote it with Claude, Copilot, or an agent? Fine —
  mention it or don't.
- **Any feature request, however far-fetched.** "Can it do X?" is useful
  even when the answer is no — it shows how people want to use this.
- **No template, no checklist, no CLA, no guidelines.** Have fun.

The point is engagement. Ideas, bug reports, "here's how I'm using
this," a PR that's mostly wrong but sparks the right fix — all of it
helps. The only real failure is a good thought that never gets sent
because the friction felt too high.

Great PRs will often consider SPEC.md. Open a [pull request](https://github.com/SauersML/gam/pulls) or an
[issue](https://github.com/SauersML/gam/issues) — bugs, features,
questions, wild ideas. That's the whole process.

Note, this README was written mostly by Claude, with some feedback from the human.

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).
