"""Seeded calibration of summary() smooth p-values for te(), ti(), by= and 2-D smooths.

usage: python bench/pvalue_calibration/pv-interactions/calibrate.py CELL REPS [OUT.json]

Each replicate r of cell C draws from ``numpy.random.default_rng([crc32(C), r])``, fits
``gamfit.fit`` and reads ``model.summary().smooth_terms``. Every term reports its
rejection rate at .10/.05/.01 with the Monte Carlo SE, the upper-tail mass
``P(p > 1 - a)`` at the same levels, and a two-sided Kolmogorov-Smirnov test of its
p-values against U(0, 1). A null term is calibrated when both tails sit at nominal
and KS does not reject; a non-null term's rejection rate is its power.
"""
import json
import sys
import time
import warnings
import zlib

import numpy as np
import pandas as pd
from scipy import stats

import gamfit

warnings.filterwarnings("ignore")
TAU = 2 * np.pi


def logistic(eta):
    return 1.0 / (1.0 + np.exp(-eta))


def draw(rng, family, eta, sigma):
    if family == "gaussian":
        return eta + sigma * rng.standard_normal(eta.shape)
    return (rng.uniform(size=eta.shape) < logistic(eta)).astype(float)


def cell_ti(rng, n, family, interaction):
    """(a) additive truth s(x1) + s(x2); ``interaction`` adds sin(2πx1)·cos(πx2)."""
    x1, x2 = rng.uniform(size=n), rng.uniform(size=n)
    eta = np.sin(TAU * x1) + 0.8 * np.cos(TAU * x2)
    if family == "binomial":
        eta = 1.2 * eta
    eta = eta + interaction * np.sin(TAU * x1) * np.cos(np.pi * x2)
    y = draw(rng, family, eta, 0.8)
    frame = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    return frame, "y ~ s(x1) + s(x2) + ti(x1, x2)", {"ti(x1, x2)": interaction != 0.0}


def cell_by_factor(rng, n, family, amplitude):
    """(b) three levels with their own intercepts; only level a carries a curve."""
    g = np.array(["a", "b", "c"])[rng.integers(0, 3, n)]
    x = rng.uniform(size=n)
    shift = {"a": 0.0, "b": 0.5, "c": -0.5}
    eta = np.array([shift[v] for v in g]) + np.where(g == "a", amplitude * np.sin(TAU * x), 0.0)
    y = draw(rng, family, eta, 0.8)
    frame = pd.DataFrame({"y": y, "x": x, "g": g})
    return frame, "y ~ s(x, by=g)", None


def cell_te(rng, n, family, amplitude):
    """(c) te(x1, x2) with zero truth next to a real s(x3)."""
    x1, x2, x3 = rng.uniform(size=n), rng.uniform(size=n), rng.uniform(size=n)
    eta = np.sin(TAU * x3) + amplitude * np.sin(np.pi * x1) * np.cos(np.pi * x2)
    y = draw(rng, family, eta, 0.8)
    frame = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})
    return frame, "y ~ te(x1, x2) + s(x3)", {"te(x1, x2)": amplitude != 0.0}


def cell_iso(rng, n, family, amplitude):
    """Isotropic 2-D smooth s(x1, x2) with zero truth next to a real s(x3)."""
    x1, x2, x3 = rng.uniform(size=n), rng.uniform(size=n), rng.uniform(size=n)
    eta = np.sin(TAU * x3) + amplitude * np.sin(np.pi * x1) * np.cos(np.pi * x2)
    y = draw(rng, family, eta, 0.8)
    frame = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})
    return frame, "y ~ s(x1, x2) + s(x3)", {"s(x1, x2)": amplitude != 0.0}


def cell_vc(rng, n, family, amplitude):
    """(d) varying coefficient s(x, by=z) with zero truth next to a real s(x)."""
    x, z = rng.uniform(size=n), rng.standard_normal(n)
    eta = np.sin(TAU * x) + amplitude * z * np.cos(np.pi * x)
    y = draw(rng, family, eta, 0.8)
    frame = pd.DataFrame({"y": y, "x": x, "z": z})
    return frame, "y ~ s(x) + s(x, by=z)", None


CELLS = {
    "ti_gauss_200": (cell_ti, 200, "gaussian", 0.0),
    "ti_gauss_1000": (cell_ti, 1000, "gaussian", 0.0),
    "ti_binom_200": (cell_ti, 200, "binomial", 0.0),
    "ti_binom_1000": (cell_ti, 1000, "binomial", 0.0),
    "ti_gauss_200_power": (cell_ti, 200, "gaussian", 0.6),
    "ti_binom_1000_power": (cell_ti, 1000, "binomial", 0.8),
    "by_gauss_300": (cell_by_factor, 300, "gaussian", 1.5),
    "by_binom_600": (cell_by_factor, 600, "binomial", 2.0),
    "te_gauss_300": (cell_te, 300, "gaussian", 0.0),
    "te_binom_600": (cell_te, 600, "binomial", 0.0),
    "te_gauss_300_power": (cell_te, 300, "gaussian", 0.5),
    "iso_gauss_300": (cell_iso, 300, "gaussian", 0.0),
    "iso_gauss_300_power": (cell_iso, 300, "gaussian", 0.5),
    "vc_gauss_300": (cell_vc, 300, "gaussian", 0.0),
    "vc_binom_600": (cell_vc, 600, "binomial", 0.0),
    "vc_gauss_300_power": (cell_vc, 300, "gaussian", 0.4),
}


def main():
    cell, reps = sys.argv[1], int(sys.argv[2])
    out = sys.argv[3] if len(sys.argv) > 3 else None
    builder, n, family, param = CELLS[cell]
    rows, errors = [], []
    t0 = time.time()
    for rep in range(reps):
        rng = np.random.default_rng([zlib.crc32(cell.encode()), rep])
        frame, formula, targets = builder(rng, n, family, param)
        try:
            model = gamfit.fit(frame, formula, family=family)
            terms = model.summary().smooth_terms
        except Exception as exc:  # noqa: BLE001 - a failed fit is counted, never scored
            errors.append(f"{rep}: {type(exc).__name__}: {str(exc)[:200]}")
            continue
        rows.append({t["name"]: {k: t.get(k) for k in ("edf", "ref_df", "chi_sq", "p_value")} for t in terms})
    summary = {"cell": cell, "n": n, "family": family, "reps": reps, "fits": len(rows),
               "failed": len(errors), "errors": errors[:10], "seconds": time.time() - t0, "terms": {}}
    for name in sorted({k for r in rows for k in r}):
        present = [r[name] for r in rows if name in r]
        p = np.array([t["p_value"] for t in present if t["p_value"] is not None], float)
        entry = {
            "m": int(p.size),
            "p_value_none": len(present) - int(p.size),
            "edf_median": float(np.median([t["edf"] for t in present])),
            "ref_df_median": float(np.median([t["ref_df"] for t in present if t["ref_df"] is not None]))
            if p.size else None,
        }
        if p.size:
            for a in (0.10, 0.05, 0.01):
                entry[f"reject_{a}"] = float(np.mean(p <= a))
                entry[f"upper_{a}"] = float(np.mean(p > 1.0 - a))
                entry[f"mcse_{a}"] = float(np.sqrt(a * (1 - a) / p.size))
            ks = stats.kstest(p, "uniform")
            entry["ks_D"], entry["ks_p"] = float(ks.statistic), float(ks.pvalue)
        summary["terms"][name] = entry
    print(json.dumps(summary, indent=1))
    if out:
        with open(out, "w") as handle:
            json.dump({"summary": summary, "rows": rows}, handle)


if __name__ == "__main__":
    main()
