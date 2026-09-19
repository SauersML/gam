"""Calibration of the random-effect variance-component p-value.

Random-effect cells: y ~ s(x1) + group(g), with a real s(x1) and a group effect
b_g ~ N(0, sd_re²) (sd_re = 0 is the null). g has L levels, balanced
(round-robin) or unbalanced (Dirichlet(0.5) level shares, every level seen at
least once).

Double-penalty cell ("dp"): y ~ s(x1) + s(x2) with a null s(x2); the smooth's
null space is penalized by the default double penalty, so the variance
component of the whole term sits on the boundary under the null.

Output: one JSON line per replicate.

Usage:
  python null_calibration.py re FAMILIES LEVELS REPS SD_RE OUT
  python null_calibration.py dp FAMILIES N REPS EFFECT OUT
"""
import json
import os
import sys
import warnings
from multiprocessing import Pool

import numpy as np

warnings.filterwarnings("ignore")

FAMILIES = ["gaussian", "binomial", "poisson"]


def rows_for(levels):
    """Sample size per cell: 400 rows, or 5 per level when that is larger."""
    return max(400, 5 * levels)


def response(family, eta, rng):
    n = eta.shape[0]
    if family == "gaussian":
        return eta + rng.normal(0, 0.5, n)
    if family == "poisson":
        return rng.poisson(np.exp(0.3 + 0.5 * eta)).astype(float)
    if family == "binomial":
        return (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float)
    raise ValueError(family)


def simulate_re(family, levels, balanced, seed, sd_re):
    rng = np.random.default_rng(seed)
    n = rows_for(levels)
    x1 = rng.uniform(0, 1, n)
    if balanced:
        g = np.arange(n) % levels
    else:
        share = rng.dirichlet(np.full(levels, 0.5))
        g = rng.choice(levels, size=n, p=share)
        g[:levels] = np.arange(levels)
    b = rng.normal(0, sd_re, levels)
    eta = np.sin(2 * np.pi * x1) + b[g]
    y = response(family, eta, rng)
    return {"y": y, "x1": x1, "g": np.array([f"L{v}" for v in g])}


def simulate_dp(family, n, seed, effect):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    eta = np.sin(2 * np.pi * x1) + effect * np.sin(2 * np.pi * x2)
    return {"y": response(family, eta, rng), "x1": x1, "x2": x2}


def quiet_fit(data, formula, family):
    import gamfit

    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    os.dup2(devnull, 2)
    try:
        return gamfit.fit(data, formula, family=family).summary().smooth_terms
    finally:
        os.dup2(saved, 2)
        os.close(devnull)


def record(row, base):
    return {
        **base,
        "p": row.get("p_value"),
        "unavailable": row.get("p_value_unavailable"),
        "edf": row.get("edf"),
        "ref_df": row.get("ref_df"),
        "stat": row.get("chi_sq"),
    }


def one_re(task):
    family, levels, balanced, rep, sd_re = task
    seed = 11_000_000 + 1_000_000 * FAMILIES.index(family) + 10_000 * levels + 5_000 * balanced + rep
    base = {"cell": "re", "family": family, "levels": levels, "balanced": balanced,
            "rep": rep, "effect": sd_re, "n": rows_for(levels)}
    try:
        rows = quiet_fit(simulate_re(family, levels, balanced, seed, sd_re),
                         "y ~ s(x1) + group(g)", family)
        return record([r for r in rows if r["name"] == "g"][0], base)
    except Exception as exc:  # recorded, never skipped silently
        return {**base, "error": str(exc)[:300]}


def one_dp(task):
    family, n, rep, effect = task
    seed = 13_000_000 + 1_000_000 * FAMILIES.index(family) + 10 * n + rep
    base = {"cell": "dp", "family": family, "levels": 0, "balanced": True,
            "rep": rep, "effect": effect, "n": n}
    try:
        rows = quiet_fit(simulate_dp(family, n, seed, effect), "y ~ s(x1) + s(x2)", family)
        return record([r for r in rows if r["name"].endswith("x2)")][0], base)
    except Exception as exc:
        return {**base, "error": str(exc)[:300]}


if __name__ == "__main__":
    kind = sys.argv[1]
    families = sys.argv[2].split(",")
    sizes = [int(v) for v in sys.argv[3].split(",")]
    reps = int(sys.argv[4])
    effect = float(sys.argv[5])
    out = sys.argv[6] if len(sys.argv) > 6 else "/dev/stdout"
    if kind == "re":
        worker = one_re
        tasks = [(f, L, bal, r, effect) for f in families for L in sizes
                 for bal in (True, False) for r in range(reps)]
    else:
        worker = one_dp
        tasks = [(f, n, r, effect) for f in families for n in sizes for r in range(reps)]
    with Pool(int(os.environ.get("NPROC", "4"))) as pool, open(out, "a") as fh:
        for res in pool.imap_unordered(worker, tasks, chunksize=4):
            fh.write(json.dumps(res) + "\n")
            fh.flush()
