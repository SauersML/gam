"""Seeded Monte Carlo size/power calibration of the summary smooth-term p-value
under concurvity, large n, prior weights / offsets / binomial trials, and a
heteroscedastic Gaussian response.

Every cell fits a model in which ``s(x2)`` is a genuinely null term (the H0
draws) and, for power, the same design with a real ``x2`` effect (the H1
draws). The reported p-value is ``summary().smooth_terms[s(x2)].p_value`` --
the Wood (2013) rank-truncated Wald test the summary table publishes.

Usage::

    python calibrate.py                      # every cell, default reps
    python calibrate.py conc_rho90 weights_iv --reps 500 --workers 4
    python calibrate.py --list

Output is one JSON line per cell on stdout (and appended to
``results.jsonl`` next to this script) with the rejection rate at
0.10 / 0.05 / 0.01, its Monte Carlo standard error, the Kolmogorov-Smirnov
distance of the null p-values from U(0, 1) with its asymptotic p-value, and the
rejection rate at 0.05 under the alternative.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Callable

import numpy as np

ALPHAS = (0.10, 0.05, 0.01)


@dataclass(frozen=True)
class Cell:
    name: str
    n: int
    reps: int
    formula: str
    family: str
    draw: Callable[[np.random.Generator, int, float], dict]
    alt_amplitude: float
    weights: str | None = None
    offset: str | None = None
    noise_formula: str | None = None
    note: str = ""


def _uniform_pair(rng: np.random.Generator, n: int, rho: float) -> tuple[np.ndarray, np.ndarray]:
    """``(x1, x2)`` with U(0, 1) margins and Gaussian-copula correlation ``rho``."""
    from scipy.stats import norm

    z1 = rng.standard_normal(n)
    z2 = rho * z1 + math.sqrt(1.0 - rho * rho) * rng.standard_normal(n)
    return norm.cdf(z1), norm.cdf(z2)


def _f1(x1: np.ndarray) -> np.ndarray:
    return np.sin(2.0 * np.pi * x1)


def _f2(x2: np.ndarray) -> np.ndarray:
    return np.cos(2.0 * np.pi * x2)


def _gaussian_draw(rho: float) -> Callable[[np.random.Generator, int, float], dict]:
    def draw(rng: np.random.Generator, n: int, amp: float) -> dict:
        x1, x2 = _uniform_pair(rng, n, rho)
        y = _f1(x1) + amp * _f2(x2) + rng.standard_normal(n)
        return {"y": y, "x1": x1, "x2": x2}

    return draw


def _nonlinear_concurvity_draw(rng: np.random.Generator, n: int, amp: float) -> dict:
    # x2 is a nonlinear function of x1 plus noise: E[x2 | x1] = (2 x1 - 1)^2,
    # so s(x2) can mimic part of s(x1) through a curved, not linear, map.
    x1 = rng.uniform(size=n)
    x2 = (2.0 * x1 - 1.0) ** 2 + 0.15 * rng.standard_normal(n)
    y = _f1(x1) + amp * np.sin(np.pi * x2) + rng.standard_normal(n)
    return {"y": y, "x1": x1, "x2": x2}


def _inverse_variance_draw(rng: np.random.Generator, n: int, amp: float) -> dict:
    # Known relative precisions w_i: Var(y_i) = sigma^2 / w_i with sigma^2 = 1.
    x1 = rng.uniform(size=n)
    x2 = rng.uniform(size=n)
    w = np.exp(rng.uniform(np.log(0.2), np.log(5.0), size=n))
    y = _f1(x1) + amp * _f2(x2) + rng.standard_normal(n) / np.sqrt(w)
    return {"y": y, "x1": x1, "x2": x2, "w": w}


def _poisson_exposure_draw(rng: np.random.Generator, n: int, amp: float) -> dict:
    x1 = rng.uniform(size=n)
    x2 = rng.uniform(size=n)
    exposure = np.exp(rng.uniform(np.log(0.5), np.log(20.0), size=n))
    mu = exposure * np.exp(0.2 + 0.6 * _f1(x1) + amp * _f2(x2))
    y = rng.poisson(mu).astype(float)
    return {"y": y, "x1": x1, "x2": x2, "log_exposure": np.log(exposure)}


def _binomial_trials_draw(rng: np.random.Generator, n: int, amp: float) -> dict:
    x1 = rng.uniform(size=n)
    x2 = rng.uniform(size=n)
    trials = rng.integers(1, 21, size=n).astype(float)
    eta = 0.8 * _f1(x1) + amp * _f2(x2)
    prob = 1.0 / (1.0 + np.exp(-eta))
    successes = rng.binomial(trials.astype(int), prob)
    return {"y": successes / trials, "x1": x1, "x2": x2, "trials": trials}


def _heteroscedastic_draw(rng: np.random.Generator, n: int, amp: float) -> dict:
    # The mean does not depend on x2; the noise standard deviation does,
    # growing 7-fold across its range.
    x1 = rng.uniform(size=n)
    x2 = rng.uniform(size=n)
    sd = np.exp(-1.0 + 2.0 * x2)
    y = _f1(x1) + amp * _f2(x2) + sd * rng.standard_normal(n)
    return {"y": y, "x1": x1, "x2": x2}


GAUSS = "y ~ s(x1) + s(x2)"


def _cells() -> dict[str, Cell]:
    cells = [
        Cell("conc_rho00", 200, 500, GAUSS, "gaussian", _gaussian_draw(0.0), 0.35,
             note="independent reference"),
        Cell("conc_rho50", 200, 500, GAUSS, "gaussian", _gaussian_draw(0.5), 0.35),
        Cell("conc_rho90", 200, 500, GAUSS, "gaussian", _gaussian_draw(0.9), 0.35),
        Cell("conc_nonlinear", 200, 500, GAUSS, "gaussian", _nonlinear_concurvity_draw, 0.35),
        Cell("large_n_1e4", 10_000, 500, GAUSS, "gaussian", _gaussian_draw(0.0), 0.04),
        Cell("large_n_1e5", 100_000, 200, GAUSS, "gaussian", _gaussian_draw(0.0), 0.012),
        Cell("weights_iv", 200, 500, GAUSS, "gaussian", _inverse_variance_draw, 0.35,
             weights="w", note="inverse-variance prior weights, sigma^2 = 1"),
        Cell("poisson_offset", 200, 500, GAUSS, "poisson", _poisson_exposure_draw, 0.15,
             offset="log_exposure", note="log-exposure offset"),
        Cell("binomial_trials", 200, 500, GAUSS, "binomial", _binomial_trials_draw, 0.3,
             weights="trials", note="proportion response, trials in 1..20 as weights"),
        Cell("hetero_constant", 400, 500, GAUSS, "gaussian", _heteroscedastic_draw, 0.35,
             note="constant-variance model on heteroscedastic data (expected to fail)"),
        Cell("hetero_locscale", 400, 500, GAUSS, "gaussian",
             _heteroscedastic_draw, 0.35, noise_formula="s(x2)",
             note="location-scale model, log sd ~ s(x2)"),
    ]
    return {cell.name: cell for cell in cells}


def _x2_pvalue(model) -> float:
    rows = model.summary().smooth_terms
    for row in rows:
        if "x2" in str(row.get("name")) and "noise" not in str(row.get("name")):
            p = row.get("p_value")
            return float("nan") if p is None else float(p)
    raise KeyError(f"no s(x2) row in {[r.get('name') for r in rows]}")


def _one(args: tuple[str, int, bool]) -> tuple[int, bool, float, float, str | None]:
    import gamfit

    name, rep, alternative = args
    cell = _cells()[name]
    seed = 7_300_000 + 1000 * sorted(_cells()).index(name) + rep
    rng = np.random.default_rng([seed, int(alternative)])
    data = cell.draw(rng, cell.n, cell.alt_amplitude if alternative else 0.0)
    kwargs = {"family": cell.family}
    if cell.weights:
        kwargs["weights"] = cell.weights
    if cell.offset:
        kwargs["offset"] = cell.offset
    if cell.noise_formula:
        kwargs["noise_formula"] = cell.noise_formula
    start = time.perf_counter()
    try:
        model = gamfit.fit(data, cell.formula, **kwargs)
        p = _x2_pvalue(model)
        err = None
    except Exception as exc:  # recorded, never silently dropped
        p, err = float("nan"), f"{type(exc).__name__}: {str(exc)[:200]}"
    return rep, alternative, p, time.perf_counter() - start, err


def _ks(pvalues: np.ndarray) -> tuple[float, float]:
    from scipy.stats import kstest

    result = kstest(pvalues, "uniform")
    return float(result.statistic), float(result.pvalue)


def run_cell(name: str, reps: int | None, workers: int, power_reps: int | None) -> dict:
    cell = _cells()[name]
    reps = reps or cell.reps
    power_reps = power_reps if power_reps is not None else max(100, reps // 5)
    jobs = [(name, r, False) for r in range(reps)] + [(name, r, True) for r in range(power_reps)]
    ctx = get_context("spawn")
    with ctx.Pool(workers) as pool:
        out = pool.map(_one, jobs, chunksize=1)
    null = np.array([p for _, alt, p, _, _ in out if not alt])
    alt = np.array([p for _, a, p, _, _ in out if a])
    errors = [e for *_, e in out if e]
    ok = null[np.isfinite(null)]
    size = {f"{a:.2f}": float(np.mean(ok <= a)) for a in ALPHAS}
    mcse = {f"{a:.2f}": math.sqrt(a * (1 - a) / max(len(ok), 1)) for a in ALPHAS}
    excess = {k: (size[k] - float(k)) / mcse[k] for k in size}
    ks_d, ks_p = _ks(ok) if len(ok) else (float("nan"), float("nan"))
    alt_ok = alt[np.isfinite(alt)]
    return {
        "cell": name,
        "n": cell.n,
        "family": cell.family,
        "note": cell.note,
        "reps_null": int(reps),
        "usable_null": int(len(ok)),
        "size": size,
        "mcse": mcse,
        "excess_in_mcse": excess,
        "ks_d": ks_d,
        "ks_p": ks_p,
        "frac_p_gt_0.99": float(np.mean(ok > 0.99)) if len(ok) else float("nan"),
        "power_0.05": float(np.mean(alt_ok <= 0.05)) if len(alt_ok) else float("nan"),
        "power_reps": int(len(alt_ok)),
        "alt_amplitude": cell.alt_amplitude,
        "errors": len(errors),
        "first_error": errors[0] if errors else None,
        "median_fit_seconds": float(np.median([t for *_, t, _ in out])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cells", nargs="*")
    parser.add_argument("--reps", type=int)
    parser.add_argument("--power-reps", type=int)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    cells = _cells()
    if args.list:
        for cell in cells.values():
            print(f"{cell.name:18s} n={cell.n:<7d} reps={cell.reps:<4d} {cell.family:24s} {cell.note}")
        return
    results_path = Path(__file__).with_name("results.jsonl")
    for name in args.cells or list(cells):
        result = run_cell(name, args.reps, args.workers, args.power_reps)
        line = json.dumps(result)
        print(line, flush=True)
        with results_path.open("a") as handle:
            handle.write(line + "\n")


if __name__ == "__main__":
    sys.exit(main())
