"""Seeded random additive data-generating processes for the convergence fuzzer.

A *case* is an integer. It fixes, through ``numpy.random.default_rng((ROOT,
case))``, everything about the DGP except the family and ``n``:

- the covariate count ``p`` in ``1..MAX_COVARIATES``;
- per covariate, a distribution (:data:`DISTRIBUTIONS`) with a random location
  and scale, and a true function shape (:data:`SHAPES`) with random
  parameters;
- the family-specific intercept, signal amplitude and (Gaussian) noise scale.

The model fitted is always the plain additive ``y ~ s(x0) + ... + s(x{p-1})``
with every default left alone: the fuzzer asks whether the default fit of an
ordinary additive model certifies, not whether a tuned one does.

Each true function is evaluated on the covariate's own robust unit scale
``u = (x - q01) / (q99 - q01)`` and then scaled so its sample sd is at most the
case's per-term amplitude and its largest centred value is at most three
times that amplitude. A heavy-tailed or outlying covariate therefore keeps a
bounded linear predictor (no ``exp`` overflow in the truth), while its
function is still exactly the declared shape on the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import special

FloatArray = NDArray[np.float64]

ROOT_SEED = 0x6A7F5
MAX_COVARIATES = 8
FAMILIES: tuple[str, ...] = ("gaussian", "binomial", "poisson")
DISTRIBUTIONS: tuple[str, ...] = (
    "uniform",
    "skewed",
    "heavy_tailed",
    "clustered",
    "discrete",
    "outliers",
)
SHAPES: tuple[str, ...] = ("linear", "sinusoid", "step", "spiky", "flat")
# Held-out rows are drawn from the same DGP with this seed offset.
TEST_SEED_OFFSET = 1_000


@dataclass(frozen=True)
class Covariate:
    distribution: str
    params: dict[str, float]
    shape: str
    shape_params: dict[str, float]


@dataclass(frozen=True)
class Case:
    case: int
    covariates: tuple[Covariate, ...]
    intercept: dict[str, float]
    amplitude: dict[str, float]
    noise_sd: float

    @property
    def p(self) -> int:
        return len(self.covariates)

    @property
    def names(self) -> list[str]:
        return [f"x{j}" for j in range(self.p)]

    @property
    def formula(self) -> str:
        return "y ~ " + " + ".join(f"s({nm})" for nm in self.names)

    def describe(self) -> str:
        return ", ".join(f"{c.distribution}/{c.shape}" for c in self.covariates)


def _draw_covariate(rng: np.random.Generator) -> Covariate:
    dist = str(rng.choice(DISTRIBUTIONS))
    params: dict[str, float] = {
        # Random location and a log-uniform scale over four decades: the
        # engine sees raw covariates, so its knot and penalty scaling must be
        # scale-free.
        "loc": float(rng.normal(0.0, 10.0)),
        "scale": float(10.0 ** rng.uniform(-2.0, 2.0)),
    }
    if dist == "skewed":
        params["sigma"] = float(rng.uniform(0.5, 1.5))
    elif dist == "heavy_tailed":
        params["df"] = float(rng.choice([1.0, 2.0, 3.0]))
    elif dist == "clustered":
        for i, centre in enumerate(rng.uniform(0.0, 1.0, int(rng.integers(2, 6)))):
            params[f"centre{i}"] = float(centre)
        params["spread"] = float(10.0 ** rng.uniform(-3.0, -1.5))
    elif dist == "discrete":
        params["levels"] = float(rng.choice([2, 3, 4, 5, 8, 12, 20]))
    elif dist == "outliers":
        params["fraction"] = float(rng.choice([0.005, 0.01, 0.03]))
        params["reach"] = float(rng.uniform(20.0, 100.0))
    shape = str(rng.choice(SHAPES))
    sp: dict[str, float] = {}
    if shape == "linear":
        sp["sign"] = float(rng.choice([-1.0, 1.0]))
    elif shape == "sinusoid":
        sp["freq"] = float(rng.uniform(0.5, 4.0))
        sp["phase"] = float(rng.uniform(0.0, 2.0 * np.pi))
    elif shape == "step":
        sp["steps"] = float(rng.integers(1, 4))
        sp["seed"] = float(rng.integers(0, 2**31))
    elif shape == "spiky":
        sp["center"] = float(rng.uniform(0.1, 0.9))
        sp["width"] = float(rng.uniform(0.01, 0.05))
    return Covariate(dist, params, shape, sp)


def case_spec(case: int) -> Case:
    rng = np.random.default_rng((ROOT_SEED, case))
    p = int(rng.integers(1, MAX_COVARIATES + 1))
    covariates = tuple(_draw_covariate(rng) for _ in range(p))
    intercept = {
        "gaussian": float(rng.normal(0.0, 5.0)),
        "binomial": float(rng.uniform(-1.5, 1.5)),
        "poisson": float(rng.uniform(-0.5, 2.5)),
    }
    amplitude = {
        "gaussian": 1.0,
        "binomial": float(rng.uniform(0.5, 2.0)),
        "poisson": float(rng.uniform(0.3, 1.0)),
    }
    noise_sd = float(rng.choice([0.1, 0.5, 2.0]))
    return Case(case, covariates, intercept, amplitude, noise_sd)


def _sample_covariate(cov: Covariate, n: int, rng: np.random.Generator) -> FloatArray:
    dist, prm = cov.distribution, cov.params
    if dist == "uniform":
        z = rng.uniform(0.0, 1.0, n)
    elif dist == "skewed":
        z = rng.lognormal(0.0, prm["sigma"], n)
    elif dist == "heavy_tailed":
        z = rng.standard_t(prm["df"], n)
    elif dist == "clustered":
        # The centres belong to the covariate, so train and test rows share them.
        centres = np.array([v for k, v in prm.items() if k.startswith("centre")])
        z = centres[rng.integers(0, centres.size, n)] + rng.normal(
            0.0, prm["spread"], n
        )
    elif dist == "discrete":
        z = rng.integers(0, int(prm["levels"]), n).astype(float)
    elif dist == "outliers":
        z = rng.normal(0.0, 1.0, n)
        hit = rng.uniform(size=n) < prm["fraction"]
        z[hit] = rng.choice([-1.0, 1.0], hit.sum()) * rng.uniform(
            prm["reach"] / 2, prm["reach"], hit.sum()
        )
    else:
        raise ValueError(f"unknown distribution {dist!r}")
    return prm["loc"] + prm["scale"] * z


def _unit(x: FloatArray, ref: FloatArray) -> FloatArray:
    lo, hi = np.quantile(ref, [0.01, 0.99])
    if hi <= lo:
        lo, hi = float(np.min(ref)), float(np.max(ref))
    if hi <= lo:
        return np.zeros_like(x)
    return np.asarray((x - lo) / (hi - lo), dtype=float)


def _shape(cov: Covariate, u: FloatArray) -> FloatArray:
    sp = cov.shape_params
    if cov.shape == "linear":
        return sp["sign"] * u
    if cov.shape == "sinusoid":
        return np.sin(2.0 * np.pi * sp["freq"] * u + sp["phase"])
    if cov.shape == "step":
        srng = np.random.default_rng(int(sp["seed"]))
        cuts = np.sort(srng.uniform(0.15, 0.85, int(sp["steps"])))
        jumps = srng.choice([-1.0, 1.0], cuts.size) * srng.uniform(0.5, 1.5, cuts.size)
        return np.asarray(
            np.sum(jumps[None, :] * (u[:, None] > cuts[None, :]), axis=1), dtype=float
        )
    if cov.shape == "spiky":
        return np.exp(-0.5 * ((u - sp["center"]) / sp["width"]) ** 2)
    if cov.shape == "flat":
        return np.zeros_like(u)
    raise ValueError(f"unknown shape {cov.shape!r}")


@dataclass(frozen=True)
class Draw:
    spec: Case
    family: str
    train: dict[str, FloatArray]
    test: dict[str, FloatArray]
    mu_test: FloatArray


def draw(case: int, family: str, n: int) -> Draw:
    """Train and held-out tables (``x0..``, ``y``) for one (case, family, n)."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")
    spec = case_spec(case)
    seed = (ROOT_SEED, case, FAMILIES.index(family), n)
    rng = np.random.default_rng(seed)
    trng = np.random.default_rng((*seed, TEST_SEED_OFFSET))
    xs = [_sample_covariate(c, n, rng) for c in spec.covariates]
    xt = [_sample_covariate(c, n, trng) for c in spec.covariates]
    amp = spec.amplitude[family] / np.sqrt(spec.p)
    eta = np.full(n, spec.intercept[family])
    eta_t = np.full(n, spec.intercept[family])
    for cov, x, x_t in zip(spec.covariates, xs, xt):
        f = _shape(cov, _unit(x, x))
        f_t = _shape(cov, _unit(x_t, x))
        centre = float(np.mean(f))
        sd = float(np.std(f))
        reach = float(np.max(np.abs(f - centre))) if f.size else 0.0
        denom = max(sd, reach / 3.0)
        if denom > 0.0:
            eta += amp * (f - centre) / denom
            eta_t += amp * (f_t - centre) / denom
    y = _response(family, eta, spec.noise_sd, rng)
    y_t = _response(family, eta_t, spec.noise_sd, trng)
    names = spec.names
    train = {nm: x for nm, x in zip(names, xs)}
    train["y"] = y
    test = {nm: x for nm, x in zip(names, xt)}
    test["y"] = y_t
    return Draw(spec, family, train, test, _mean(family, eta_t))


def _mean(family: str, eta: FloatArray) -> FloatArray:
    if family == "gaussian":
        return eta
    if family == "binomial":
        return np.asarray(special.expit(eta), dtype=float)
    return np.exp(eta)


def _response(
    family: str, eta: FloatArray, noise_sd: float, rng: np.random.Generator
) -> FloatArray:
    mu = _mean(family, eta)
    if family == "gaussian":
        return mu + rng.normal(0.0, noise_sd, mu.size)
    if family == "binomial":
        return (rng.uniform(size=mu.size) < mu).astype(float)
    return rng.poisson(mu).astype(float)


def spec_json(spec: Case) -> dict[str, Any]:
    return {
        "p": spec.p,
        "covariates": [
            {
                "distribution": c.distribution,
                "shape": c.shape,
                **{f"d_{k}": v for k, v in c.params.items()},
                **{f"s_{k}": v for k, v in c.shape_params.items()},
            }
            for c in spec.covariates
        ],
        "intercept": spec.intercept,
        "amplitude": spec.amplitude,
        "noise_sd": spec.noise_sd,
    }
