"""Seeded Monte Carlo calibration of simultaneous difference-smooth bands.

For each family (gaussian, poisson, binomial) and sample size n, simulate a
two-group by-factor GAM with unequal group sizes (group A is the 30% minority)
and fit ``y ~ g + s(x, by=g, k=10)``. On the difference-smooth grid:

* **coverage** -- the truth has a group difference ``d(x) = 0.6 cos(pi x)`` on
  the linear-predictor scale. Record whether the simultaneous band covers d on
  the whole grid (target = level) and whether the pointwise band does.
* **size / KS** -- both groups share one curve. The "no difference" test is
  the row ``p_value`` (the whole-curve p-value from the band's own max|Z|
  law); size at alpha is ``P(p <= alpha)``, and the null p-values are tested
  for uniformity with a one-sample Kolmogorov-Smirnov test.
* **power** -- ``P(p <= alpha)`` under the coverage scenario's truth.

Every replicate has its own seed ``100_000 * (1 + family index) + 10 n + r``
(the null arm adds ``5_000_000``), so any cell reruns alone. Refused fits are recorded with their message, never dropped silently.

Usage::

    python bench/pvalue_calibration/pv-bands/run.py --reps 500 --out results.json
    python bench/pvalue_calibration/pv-bands/run.py --summarize results.json
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

FAMILIES = ("gaussian", "poisson", "binomial")
SIZES = (100, 400, 2000)
LEVELS = (0.90, 0.95, 0.99)
GRID = 50
FORMULA = "y ~ g + s(x, by=g, k=10)"
SHARE_A = 0.3
NULL_SEED_OFFSET = 5_000_000


def _expit(eta):
    return 0.5 * (1.0 + np.tanh(0.5 * eta))


def _base(x):
    return 0.2 + 0.8 * np.sin(2.0 * np.pi * x)


def _difference(x):
    return 0.6 * np.cos(np.pi * x)


def replicate_seed(family: str, n: int, replicate: int, null: bool) -> int:
    seed = 100_000 * (1 + FAMILIES.index(family)) + 10 * n + replicate
    return seed + (NULL_SEED_OFFSET if null else 0)


def simulate(family: str, n: int, seed: int, null: bool):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    g = np.where(rng.uniform(size=n) < SHARE_A, "A", "B")
    eta = _base(x) + (0.0 if null else 1.0) * np.where(g == "B", _difference(x), 0.0)
    if family == "gaussian":
        y = eta + 0.5 * rng.standard_normal(n)
    elif family == "poisson":
        y = rng.poisson(np.exp(eta)).astype(float)
    else:
        y = rng.binomial(1, _expit(eta)).astype(float)
    return {"x": x, "g": g, "y": y}


def one(task):
    family, n, replicate, null = task
    import gamfit

    seed = replicate_seed(family, n, replicate, null)
    result = {"family": family, "n": n, "seed": seed, "null": null}
    data = simulate(family, n, seed, null)
    try:
        model = gamfit.fit(data, FORMULA, family=family)
    except Exception as exc:
        result["error"] = "fit: " + str(exc)[:300]
        return result
    try:
        return _bands(model, result, null)
    except Exception as exc:
        result["error"] = "difference_smooth: " + str(exc)[:300]
        return result


def _rows(model, level, simultaneous):
    return model.difference_smooth(
        view="x",
        group="g",
        pairs=[("B", "A")],
        n=GRID,
        level=level,
        simultaneous=simultaneous,
        return_type="list",
    )


def _covers(rows, null):
    x = np.array([row["x"] for row in rows])
    lower = np.array([row["lower"] for row in rows])
    upper = np.array([row["upper"] for row in rows])
    truth = np.zeros_like(x) if null else _difference(x)
    return bool(np.all((lower <= truth) & (truth <= upper)))


def _bands(model, result, null):
    for level in LEVELS:
        rows = _rows(model, level, True)
        result[f"sim_{level}"] = _covers(rows, null)
        if level == 0.95:
            result["p_value"] = float(rows[0]["p_value"])
            result["critical_95"] = float(rows[0]["critical"])
            result["covariance_kind"] = rows[0]["covariance_kind"]
    result["pw_0.95"] = _covers(_rows(model, 0.95, False), null)
    return result


def _cells(results):
    for family in FAMILIES:
        for n in SIZES:
            for null in (False, True):
                rows = [
                    r for r in results
                    if r["family"] == family and r["n"] == n and r["null"] == null
                ]
                if rows:
                    yield family, n, null, rows


def _ks_uniform(p_values):
    from scipy import stats

    return stats.kstest(p_values, "uniform")


def summarize(results, out=sys.stdout):
    def emit(line=""):
        print(line, file=out)

    emit("Whole-curve coverage of the difference d(x) (target = level)")
    emit(f"{'family':9s} {'n':>5s} {'R':>4s} {'err':>3s}  "
         + "  ".join(f"sim@{level:.2f}" for level in LEVELS)
         + "  pw@0.95  MCSE@.95  conditional")
    for family, n, null, rows in _cells(results):
        if null:
            continue
        ok = [r for r in rows if "error" not in r]
        if not ok:
            emit(f"{family:9s} {n:5d} {len(rows):4d} {len(rows):3d}")
            continue
        cells = [f"{np.mean([r[f'sim_{level}'] for r in ok]):8.3f}" for level in LEVELS]
        pointwise = np.mean([r["pw_0.95"] for r in ok])
        mcse = math.sqrt(0.95 * 0.05 / len(ok))
        conditional = sum(r["covariance_kind"] == "conditional" for r in ok)
        emit(f"{family:9s} {n:5d} {len(ok):4d} {len(rows) - len(ok):3d}  "
             + "  ".join(cells) + f"  {pointwise:7.3f}  {mcse:7.4f}  {conditional:11d}")

    emit()
    emit("No-difference test under a shared curve (size) and under d(x) (power)")
    emit(f"{'family':9s} {'n':>5s} {'R':>4s} {'err':>3s}  size@.10  size@.05  size@.01"
         "  MCSE@.05  KS D     KS p    power@.05")
    power = {}
    for family, n, null, rows in _cells(results):
        if not null:
            ok = [r for r in rows if "error" not in r]
            if ok:
                power[(family, n)] = np.mean([r["p_value"] <= 0.05 for r in ok])
    for family, n, null, rows in _cells(results):
        if not null:
            continue
        ok = [r for r in rows if "error" not in r]
        if not ok:
            emit(f"{family:9s} {n:5d} {len(rows):4d} {len(rows):3d}")
            continue
        p_values = np.array([r["p_value"] for r in ok])
        sizes = [f"{np.mean(p_values <= alpha):8.3f}" for alpha in (0.10, 0.05, 0.01)]
        ks = _ks_uniform(p_values)
        mcse = math.sqrt(0.05 * 0.95 / len(ok))
        emit(f"{family:9s} {n:5d} {len(ok):4d} {len(rows) - len(ok):3d}  "
             + "  ".join(sizes)
             + f"  {mcse:8.4f}  {ks.statistic:.4f}  {ks.pvalue:.4f}  "
             + f"{power.get((family, n), float('nan')):8.3f}")

    kinds = {}
    for r in results:
        if "error" in r:
            key = (r["family"], r["n"], r["error"][:110])
            kinds[key] = kinds.get(key, 0) + 1
    if kinds:
        emit()
        emit("Refused replicates")
        for (family, n, message), count in sorted(kinds.items()):
            emit(f"  {family:9s} {n:5d} {count:4d} x {message}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--sizes", default=",".join(map(str, SIZES)))
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--out")
    parser.add_argument("--summarize", nargs="+", help="summarize saved result files")
    args = parser.parse_args()
    if args.summarize:
        results = [r for path in args.summarize for r in json.load(open(path))]
        summarize(results)
        return
    if not args.out:
        parser.error("--out is required unless --summarize is given")
    families = args.families.split(",")
    sizes = [int(v) for v in args.sizes.split(",")]
    tasks = [
        (family, n, replicate, null)
        for family in families
        for n in sizes
        for null in (False, True)
        for replicate in range(args.reps)
    ]
    start = time.time()
    with mp.Pool(args.workers) as pool:
        results = pool.map(one, tasks, chunksize=2)
    with open(args.out, "w") as handle:
        json.dump(results, handle)
    print(f"{len(results)} fits in {time.time() - start:.0f}s", file=sys.stderr)
    summarize(results)


if __name__ == "__main__":
    main()
