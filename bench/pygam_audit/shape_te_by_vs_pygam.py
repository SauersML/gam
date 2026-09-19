"""Shape-constrained tensor, conjunction and ``by=`` fits: gamfit vs pyGAM.

Four data-generating processes whose truth satisfies the requested shape:

* ``te_monotone``   f(x, z) increasing in x for every z, oscillating in z;
                    ``te(x, z, shape=[monotone_increasing, none])``.
* ``inc_concave``   f(x) = sqrt(x); ``s(x, shape=[monotone_increasing, concave])``.
* ``factor_by``     three factor levels, each curve increasing;
                    ``g + s(x, by=g, shape=monotone_increasing)``.
* ``numeric_by``    y = w * sqrt(x); ``s(x, by=w, shape=monotone_increasing)``.

pyGAM gets the equivalent model: ``te(0, 1, constraints=[...])``,
``s(0, constraints=[...])``, one ``s(0, by=<indicator>)`` per level plus
``f(g)`` for the factor-by fit (pyGAM's ``by`` is numeric only), and
``s(0, by=1)`` for the numeric by. pyGAM chooses its smoothing parameters with
its default ``gridsearch``; gamfit uses REML.

Reported per scenario, averaged over seeds: RMSE of the fitted mean against the
noiseless truth on a dense grid, and the worst violation of the requested
shape on that grid (the most negative first/second difference in the
constrained direction; 0 means the shape holds everywhere on the grid). The
``fits`` column counts the seeds each library returned a fit for: gamfit
refuses a fit its optimizer did not certify (``RemlConvergenceError``) instead
of returning one, and the averages are over the fitted seeds.

Run: ``python bench/pygam_audit/shape_te_by_vs_pygam.py [--seeds N]``.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import pandas as pd

import gamfit

try:
    import pygam
except ImportError:  # pragma: no cover - comparison needs pyGAM installed
    raise SystemExit("pip install pygam to run this comparison")


LEVELS = ("a", "b", "c")
LEVEL_TRUTH: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "a": lambda x: np.log1p(4.0 * x),
    "b": lambda x: 2.0 * x**3,
    "c": lambda x: np.tanh(6.0 * (x - 0.5)),
}


def te_truth(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.tanh(3.0 * (x - 0.5)) * (1.0 + 0.5 * z) + 0.6 * np.sin(2.0 * np.pi * z)


T = TypeVar("T")


@dataclass
class Result:
    rmse: float
    violation: float
    seconds: float


def _violation(values: np.ndarray, axis: int, order: int, sign: float) -> float:
    """Worst breach of ``sign * diff^order >= 0`` (0 when the shape holds)."""
    d = sign * np.diff(values, n=order, axis=axis)
    return float(max(0.0, -d.min()))


def _timed_gamfit(fit: Callable[[], T]) -> tuple[T | None, float]:
    """Run a gamfit fit-and-predict; ``None`` when gamfit refuses the fit.

    A refusal (gamfit raises rather than return a fit its optimizer did not
    converge to) is counted per scenario, not silently skipped.
    """
    t0 = time.perf_counter()
    try:
        values = fit()
    except gamfit.RemlConvergenceError as err:
        print(f"  gamfit refused: {str(err).splitlines()[0][:160]}")
        values = None
    return values, time.perf_counter() - t0


def _score_or_refused(
    score: Callable[[T, float], Result], values: T | None, secs: float
) -> Result | None:
    return None if values is None else score(values, secs)


# --------------------------------------------------------------------------
# Scenarios. Each returns (gamfit Result or None when refused, pyGAM Result).
# --------------------------------------------------------------------------


def te_monotone(seed: int) -> tuple[Result | None, Result]:
    rng = np.random.default_rng(seed)
    n = 600
    x, z = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    y = te_truth(x, z) + rng.normal(0, 0.25, n)
    g = np.linspace(0, 1, 41)
    xx, zz = np.meshgrid(g, g, indexing="ij")
    truth = te_truth(xx, zz)
    grid = pd.DataFrame({"x": xx.ravel(), "z": zz.ravel()})

    def fit_ours() -> np.ndarray:
        m = gamfit.fit(
            pd.DataFrame({"x": x, "z": z, "y": y}),
            "y ~ te(x, z, shape=[monotone_increasing, none])",
        )
        return np.asarray(m.predict(grid)).reshape(xx.shape)

    ours, t_ours = _timed_gamfit(fit_ours)

    t0 = time.perf_counter()
    pg = pygam.LinearGAM(pygam.te(0, 1, constraints=["monotonic_inc", None]))
    pg.gridsearch(np.column_stack([x, z]), y, progress=False)
    theirs = pg.predict(grid.to_numpy()).reshape(xx.shape)
    t_theirs = time.perf_counter() - t0

    def score(surf: np.ndarray, secs: float) -> Result:
        rmse = float(np.sqrt(np.mean((surf - truth) ** 2)))
        return Result(rmse, _violation(surf, axis=0, order=1, sign=1.0), secs)

    return _score_or_refused(score, ours, t_ours), score(theirs, t_theirs)


def inc_concave(seed: int) -> tuple[Result | None, Result]:
    rng = np.random.default_rng(seed)
    n = 400
    x = rng.uniform(0, 1, n)
    y = np.sqrt(x) + rng.normal(0, 0.15, n)
    gx = np.linspace(0, 1, 401)
    truth = np.sqrt(gx)

    def fit_ours() -> np.ndarray:
        m = gamfit.fit(
            pd.DataFrame({"x": x, "y": y}),
            "y ~ s(x, shape=[monotone_increasing, concave])",
        )
        return np.asarray(m.predict(pd.DataFrame({"x": gx})))

    ours, t_ours = _timed_gamfit(fit_ours)

    t0 = time.perf_counter()
    pg = pygam.LinearGAM(pygam.s(0, constraints=["monotonic_inc", "concave"]))
    pg.gridsearch(x[:, None], y, progress=False)
    theirs = pg.predict(gx[:, None])
    t_theirs = time.perf_counter() - t0

    def score(f: np.ndarray, secs: float) -> Result:
        worst = max(_violation(f, 0, 1, 1.0), _violation(f, 0, 2, -1.0))
        return Result(float(np.sqrt(np.mean((f - truth) ** 2))), worst, secs)

    return _score_or_refused(score, ours, t_ours), score(theirs, t_theirs)


def factor_by(seed: int) -> tuple[Result | None, Result]:
    rng = np.random.default_rng(seed)
    n_per = 250
    xs, gs, ys = [], [], []
    for level in LEVELS:
        x = rng.uniform(0, 1, n_per)
        xs.append(x)
        gs.append(np.full(n_per, level))
        ys.append(LEVEL_TRUTH[level](x) + rng.normal(0, 0.3, n_per))
    x, g, y = np.concatenate(xs), np.concatenate(gs), np.concatenate(ys)
    gx = np.linspace(0, 1, 301)

    def fit_ours() -> dict[str, np.ndarray]:
        df = pd.DataFrame({"x": x, "g": pd.Categorical(g, categories=LEVELS), "y": y})
        m = gamfit.fit(df, "y ~ g + s(x, by=g, shape=monotone_increasing)")
        return {
            level: np.asarray(
                m.predict(
                    pd.DataFrame(
                        {"x": gx, "g": pd.Categorical([level] * gx.size, categories=LEVELS)}
                    )
                )
            )
            for level in LEVELS
        }

    ours, t_ours = _timed_gamfit(fit_ours)

    # pyGAM: column 0 = x, column 1 = level code, columns 2.. = indicators.
    codes = np.searchsorted(np.array(LEVELS), g)
    ind = np.eye(len(LEVELS))[codes]
    X = np.column_stack([x, codes, ind])
    terms = pygam.f(1)
    for k in range(len(LEVELS)):
        terms += pygam.s(0, by=2 + k, constraints="monotonic_inc")
    t0 = time.perf_counter()
    pg = pygam.LinearGAM(terms)
    pg.gridsearch(X, y, progress=False)
    theirs = {}
    for k, level in enumerate(LEVELS):
        Xg = np.column_stack(
            [gx, np.full(gx.size, k), np.tile(np.eye(len(LEVELS))[k], (gx.size, 1))]
        )
        theirs[level] = pg.predict(Xg)
    t_theirs = time.perf_counter() - t0

    def score(curves: dict[str, np.ndarray], secs: float) -> Result:
        err = np.concatenate([curves[lv] - LEVEL_TRUTH[lv](gx) for lv in LEVELS])
        worst = max(_violation(curves[lv], 0, 1, 1.0) for lv in LEVELS)
        return Result(float(np.sqrt(np.mean(err**2))), worst, secs)

    return _score_or_refused(score, ours, t_ours), score(theirs, t_theirs)


def numeric_by(seed: int) -> tuple[Result | None, Result]:
    rng = np.random.default_rng(seed)
    n = 500
    x, w = rng.uniform(0, 1, n), rng.uniform(0.5, 2.0, n)
    y = w * np.sqrt(x) + rng.normal(0, 0.2, n)
    gx = np.linspace(0, 1, 301)
    ws = (0.5, 1.0, 2.0)

    def fit_ours() -> list[np.ndarray]:
        m = gamfit.fit(
            pd.DataFrame({"x": x, "w": w, "y": y}),
            "y ~ s(x, by=w, shape=monotone_increasing)",
        )
        return [np.asarray(m.predict(pd.DataFrame({"x": gx, "w": wv}))) for wv in ws]

    ours, t_ours = _timed_gamfit(fit_ours)

    t0 = time.perf_counter()
    pg = pygam.LinearGAM(pygam.s(0, by=1, constraints="monotonic_inc"))
    pg.gridsearch(np.column_stack([x, w]), y, progress=False)
    theirs = [pg.predict(np.column_stack([gx, np.full(gx.size, wv)])) for wv in ws]
    t_theirs = time.perf_counter() - t0

    def score(curves: list[np.ndarray], secs: float) -> Result:
        err = np.concatenate([c - wv * np.sqrt(gx) for c, wv in zip(curves, ws)])
        worst = max(_violation(c, 0, 1, 1.0) for c in curves)
        return Result(float(np.sqrt(np.mean(err**2))), worst, secs)

    return _score_or_refused(score, ours, t_ours), score(theirs, t_theirs)


SCENARIOS = {
    "te_monotone": te_monotone,
    "inc_concave": inc_concave,
    "factor_by": factor_by,
    "numeric_by": numeric_by,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    rows = []
    for name, run in SCENARIOS.items():
        print(name)
        ours, theirs = zip(*(run(seed) for seed in range(args.seeds)))
        for lib, results in (("gamfit", ours), ("pyGAM", theirs)):
            fitted = [r for r in results if r is not None]
            rows.append(
                {
                    "scenario": name,
                    "library": lib,
                    "fits": f"{len(fitted)}/{len(results)}",
                    "rmse_vs_truth": np.mean([r.rmse for r in fitted]),
                    "worst_shape_violation": np.max([r.violation for r in fitted]),
                    "seconds": np.mean([r.seconds for r in fitted]),
                }
            )
    table = pd.DataFrame(rows)
    print(f"pyGAM {pygam.__version__}, {args.seeds} seeds per scenario\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
