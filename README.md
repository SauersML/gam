# gam / gamfit

[![PyPI](https://img.shields.io/pypi/v/gamfit.svg)](https://pypi.org/project/gamfit/)
[![Python](https://img.shields.io/pypi/pyversions/gamfit.svg)](https://pypi.org/project/gamfit/)
[![Docs](https://img.shields.io/readthedocs/gamfit.svg)](https://gamfit.readthedocs.io/)
[![Rust CI](https://github.com/SauersML/gam/actions/workflows/test.yml/badge.svg)](https://github.com/SauersML/gam/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue.svg)](LICENSE)

A generalized additive model engine. The fitting code is in Rust. The
public interfaces are a Rust CLI (`gam`) and a Python package
(`gamfit`). Both share one engine, one formula DSL, and one on-disk
model format.

Docs: <https://gamfit.readthedocs.io/>. PyPI: <https://pypi.org/project/gamfit/>.

**Contributions of every kind are welcome** — broken PRs, first-timers,
AI-written patches, and any feature request, with no guidelines and low
friction by design. See [Contributing](#contributing).

![3D Matérn fit on a noisy 2-D landscape](docs/images/surface_3d_wireframe.png)

## Scope

Supported response families: Gaussian, binomial / Bernoulli (including a
marginal-slope variant for calibrated risk scores), Poisson,
negative-binomial, Gamma, Beta, Tweedie, multinomial-logit, conditional
transformation-normal, and parametric / semi-parametric survival.

Supported term types in formulas: parametric terms, univariate smooths
(`s`), tensor-product smooths (`te`, `ti`), radial smooths in arbitrary
dimension (`matern`, `duchon`, `thinplate`), intrinsic sphere smooths
(`sphere`), random effects (`group`), interval-bounded coefficients
(`bounded`), and learnable links (`link(type=flexible(...))`,
`link(type=blended(...))`, `sas`, `beta-logistic`, `linkwiggle`).

Smoothing parameters are selected by REML or LAML. Posterior sampling
uses NUTS over the coefficient posterior conditional on the fitted
smoothing parameters where the family supports it, and a Gaussian
Laplace approximation otherwise.

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

The dictionary supports three gating families (`assignment="ibp_map"`,
`"softmax"`, `"jumprelu"`) and a `gamfit.torch.ManifoldSAE` autograd
mirror with out-of-sample encoder distillation. Around it the interp
stack adds: `e_bh_dictionary_certificate` (e-BH certified structure
claims) and `select_topology` in `gamfit.structure_discovery` /
`gamfit.topology`; `sae_checkpoint_dynamics` for tracking atoms across
training checkpoints; `gamfit.crosscoder.Crosscoder` and
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

Response geometry. `ResponseGeometryModel` fits GAMs for responses that
live on the sphere or simplex via Fréchet-mean tangent-space maps
(`sphere_frechet_mean`, `simplex_frechet_mean`, `alr`, `clr`,
`closure`).

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

Keep one CUDA toolkit reachable. If a system CUDA install and PyTorch's
bundled CUDA wheels are both present, gamfit warns once and continues; it
loads cuBLAS / cuSOLVER / cuSPARSE through whichever `libcudart.so.12` it
finds first. When using gamfit with `torch` on a CUDA 12.x driver,
install a torch build whose CUDA suffix matches (`+cu12x`).

## Repository layout

| Path | Contents |
| --- | --- |
| `src/` | Rust engine: fitting, inference, smooth construction, survival, CLI. |
| `crates/gam-pyffi/` | PyO3 bindings (`gamfit._rust`). |
| `gamfit/` | Python public API on top of the bindings. |
| `docs/` | MkDocs/Material documentation sources. |
| `tests/` | Rust and Python integration tests. |
| `bench/` | Benchmark harness, configs, datasets, plots. |
| `examples/` | Python and Rust demos, including SAE and topology examples. |
| `scripts/` | Documentation figure, gallery, and diagnostic scripts. |

## Development

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -A warnings -D clippy::correctness -D clippy::suspicious
cargo test --all-features

uv venv --python 3.12 .venv-docs
uv pip install --python .venv-docs/bin/python -r docs/requirements.txt
.venv-docs/bin/mkdocs serve
```

Benchmarks: `python3 bench/run_suite.py --help` for the suite runner or
`python3 bench/run.py --help` for the current benchmark entrypoint.

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

Open a [pull request](https://github.com/SauersML/gam/pulls) or an
[issue](https://github.com/SauersML/gam/issues) — bugs, features,
questions, wild ideas. That's the whole process.

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).
