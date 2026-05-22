# gam / gamfit

[![PyPI](https://img.shields.io/pypi/v/gamfit.svg)](https://pypi.org/project/gamfit/)
[![Python](https://img.shields.io/pypi/pyversions/gamfit.svg)](https://pypi.org/project/gamfit/)
[![Docs](https://img.shields.io/readthedocs/gamfit.svg)](https://gamfit.readthedocs.io/)
[![Rust CI](https://github.com/SauersML/gam/actions/workflows/test.yml/badge.svg)](https://github.com/SauersML/gam/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue.svg)](LICENSE)

A formula-based generalized additive model engine written in Rust, with a
Python package (`gamfit`) on top.

Fits Gaussian, binomial (including Bernoulli marginal-slope), Poisson, and
Gamma GLMs with smooth terms, random effects, bounded/constrained
coefficients, location-scale extensions, survival likelihoods, and
flexible/learnable link functions. Smoothing parameters are selected by
REML or LAML. Posterior sampling uses NUTS where supported, and a
Gaussian Laplace approximation otherwise.

Docs: <https://gamfit.readthedocs.io/>. PyPI: <https://pypi.org/project/gamfit/>.

![3D Matérn fit on a noisy 2-D landscape](docs/images/surface_3d_wireframe.png)

## Python and CLI

```python
import gamfit
model = gamfit.fit(train, "y ~ s(x) + group(site)")
preds = model.predict(test, interval=0.95)
```

```bash
gam fit data.csv 'y ~ smooth(x) + group(site)' --out model.json
gam predict model.json new_data.csv --uncertainty
gam report model.json data.csv
```

The Python package and the CLI share the engine, the formula DSL, and the
on-disk model format.

## Install

Python wheels are published for Linux (x86_64, aarch64), macOS (Intel and
Apple silicon), and Windows. No Rust toolchain is required.

```bash
uv add gamfit
# or
pip install gamfit
```

Optional extras: `gamfit[pandas]`, `gamfit[plot]`, `gamfit[sklearn]`,
`gamfit[torch]`, `gamfit[all]`.

For the Rust CLI:

```bash
curl -fsSL https://raw.githubusercontent.com/SauersML/gam/main/install.sh | bash
```

Or build from source with `cargo build --release`; the binary is
`./target/release/gam`.

## Features

### Penalty structure

Polyharmonic / Duchon surface smooths combine three penalty operators on
the same basis: a magnitude (mass) operator, a gradient operator, and a
curvature (bending) operator. P-spline and thin-plate smooths use their
standard derivative-based penalties. Each penalized block has its own
smoothing parameter selected by REML/LAML.

```python
gamfit.fit(df, "z ~ duchon(pc1, pc2, pc3, pc4, centers=50)")
```

### Per-axis anisotropy

Surface smooths can learn per-axis length scales so each input dimension
shrinks independently.

```python
gamfit.fit(df, "z ~ matern(pc1, pc2, pc3, pc4)", scale_dimensions=True)
```

### Surface smooths

P-spline, thin-plate, Matérn, and Duchon radial bases are available in
arbitrary dimension. Duchon is scale-free by default; passing
`length_scale=...` selects the hybrid Duchon-Matérn kernel.

```python
gamfit.fit(df, "y ~ matern(x1, x2, x3, nu=5/2)")
gamfit.fit(df, "y ~ duchon(x1, x2, x3, x4, centers=80)")
gamfit.fit(df, "y ~ te(space, time, k=10)")   # tensor product
```

### Manifold smooths

For inputs on a circle, cylinder, torus, or sphere, gamfit provides
smooths whose basis and penalty encode the wrap topology. This includes
periodic 1-D B-splines, periodic margins inside tensor products, an
intrinsic `sphere(...)` kernel (Wahba reproducing kernel or spherical
harmonics), and boundary-conditioned 1-D B-splines. The Möbius example
in the gallery is a 4π-periodic double-cover parameterization, not a
twisted Möbius-strip basis.

![rotating recovery of six geometric examples (trefoil knot, latent-free loop, wobbly cylinder, lumpy sphere, bumpy torus, Möbius double-cover) from noisy 3-D point clouds](docs/images/geometric_shapes_demo.gif)

Each pair shows a noisy 3-D point cloud (left) and the recovered smooth
(right). The full gallery and reproduction script are in
[docs/manifold-smooths.md](docs/manifold-smooths.md).

```python
gamfit.fit(df, "y ~ s(theta, periodic=true, period=2*pi)")             # circle
gamfit.fit(df, "y ~ te(theta, h, periodic=[0], period=[2*pi, None])")  # cylinder
gamfit.fit(df, "y ~ te(u, v, periodic=[0,1], period=[2*pi, 2*pi])")    # torus
gamfit.fit(df, "y ~ sphere(lat, lon, radians=true)")                   # S²
gamfit.fit(df, "y ~ s(x, bc=clamped)")                                 # zero-slope endpoints
```

### Flexible and learnable links

`flexible(base)` adds a spline offset on top of a base link.
`blended(logit, probit)` learns a mixture weight. `sas` and
`beta-logistic` learn shape parameters from the data.

```python
gamfit.fit(df, "case ~ s(age) + link(type=flexible(probit))"
                 " + linkwiggle(internal_knots=6)")
```

### Marginal-slope models

For binary or survival outcomes with a calibrated risk score, the
baseline risk and the score effect are fit in separate formulas. The
slope on the score is a smooth function of covariate space.

```python
gamfit.fit(
    df,
    "case ~ matern(pc1, pc2, pc3)",
    family="bernoulli-marginal-slope",
    link="probit",
    z_column="pgs_z",
    logslope_formula="matern(pc1, pc2, pc3)",
)
```

![two-surface marginal-slope viz over a joint Duchon smooth](docs/images/marginal_slope_3d.png)

The two surfaces are predicted probabilities at `z = 0` and `z = +2` over
the same `(pc1, pc2)` plane.

### Survival

`Surv(entry, exit, event)` is supported with four likelihood modes:
transformation, Weibull, location-scale, and marginal-slope. A
`SurvivalPrediction` object evaluates `S(t)`, `h(t)`, and `H(t)` on any
time grid.

```python
pred = model.predict(test_df)
S = pred.survival_at([1, 5, 10, 20])     # (n_rows, 4)
H = pred.cumulative_hazard_at([10])      # (n_rows, 1)
```

For large cohorts, `pred.write_survival_at_csv(...)` streams to CSV
without materialising the full matrix.

### Posterior sampling

`model.sample(...)` runs NUTS over the coefficient posterior conditional
on the fitted smoothing parameters where supported, and a Gaussian
Laplace approximation otherwise. Predictive bands stream in row chunks.

```python
posterior = model.sample(train, seed=42)
bands = posterior.predict(test, level=0.95)
```

### Bounded coefficients

Interval transforms for coefficients in `[a, b]`, with an optional Beta
prior.

```python
gamfit.fit(df,
    "y ~ age + bounded(prop, min=0, max=1, target=0.5, strength=3)")
```

### scikit-learn estimators

```python
from gamfit.sklearn import GAMRegressor
est = GAMRegressor(formula="y ~ s(x)").fit(X, y)
```

## GPU acceleration

The Rust engine includes optional CUDA support (cuBLAS, cuSOLVER,
cuSPARSE). It is loaded lazily and falls back to the CPU path when no
working CUDA stack is present. The same wheel is used on CPU-only and
GPU hosts.

Per-op dispatch thresholds are derived at probe time from measured GPU
FP64 throughput, CPU FP64 throughput, and PCIe bandwidth. Below the
crossover where transfer cost dominates, kernels stay on the CPU. To
print the calibrated thresholds:

```python
import gamfit
print(gamfit.build_info()["cuda_diagnostics"])
print(gamfit.format_cuda_diagnostics())
```

### Mixed CUDA stacks

If both a system CUDA toolkit and PyTorch's `nvidia-*-cu12` wheels are
present in the same environment, both can appear in `/proc/self/maps`
for the same SONAME. This is usually harmless: glibc resolves
`dlopen(SONAME)` to one file per search path. gamfit warns once per
conflict-set and continues. Avoid `dlopen`-ing both libraries by
absolute path from the same process.

### Torch and CUDA driver versions

If you use gamfit together with `torch` and your driver advertises CUDA
12.x, install a torch build whose CUDA suffix matches the driver
(`+cu12x` family). gamfit itself loads cuBLAS / cuSOLVER / cuSPARSE
through whichever `libcudart.so.12` is reachable.

## Documentation and CLI

- Python documentation: <https://gamfit.readthedocs.io/>.
- CLI subcommands: `fit`, `predict`, `report`, `diagnose`, `sample`,
  `generate`. Use `gam <command> --help` for options.
- Cookbook: [docs/cookbook.md](docs/cookbook.md).
- Manifold smooths gallery: [docs/manifold-smooths.md](docs/manifold-smooths.md).

## Repository layout

| Path | Contents |
| --- | --- |
| `src/` | Rust engine: fitting, inference, smooth construction, survival, CLI. |
| `crates/gam-pyffi/` | PyO3 bindings (the `gamfit._rust` native extension). |
| `gamfit/` | Python public API on top of the bindings. |
| `docs/` | MkDocs/Material documentation sources. |
| `tests/` | Rust and Python integration tests. |
| `bench/` | Benchmark harness, configs, datasets, plots. |
| `scripts/` | Demo and diagnostic scripts. |

## Development

```bash
# Rust
cargo fmt --all
cargo clippy --all-targets --all-features -- -A warnings -D clippy::correctness -D clippy::suspicious
cargo test --all-features

# Python docs (uses uv)
uv venv --python 3.12 .venv-docs
uv pip install --python .venv-docs/bin/python -r docs/requirements.txt
.venv-docs/bin/mkdocs serve
```

Benchmark suite: `python3 bench/run_suite.py --help`.

## Issues

Open a [GitHub issue](https://github.com/SauersML/gam/issues) for bug
reports, feature requests, or questions.

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).
