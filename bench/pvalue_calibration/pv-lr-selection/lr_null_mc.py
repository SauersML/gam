"""Seeded Monte-Carlo calibration of the per-term smooth LR test.

Usage: python lr_null_mc.py <cell> <nrep> <nworkers> [out.json]

cells: gauss (n=200), gauss_small (n=60), pois (n=200), binom (n=400)

Truth: eta = b0 + a1 sin(2 pi x1) + 0 * x2 + a3 cos(2 pi x3), x ~ U(0,1)^3.
Model: y ~ s(x1) + s(x2) + s(x3). s(x2) is null (size), s(x3) is weak (power).
Replicate r draws its data from numpy's default_rng(50000 + r), so every run of
a cell sees the same datasets. Reports rejection at .10/.05/.01 with the
binomial Monte-Carlo standard error, the two-sided KS distance of every null
p-value from U(0,1) and its asymptotic p-value, the number of replicates that
published no p-value, and the weak-term power.

LR_MC_FORMULA overrides the model formula and LR_MC_SIGNAL scales a1 and a3
(diagnostics only).
"""
import json
import os
import sys
import warnings

os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

TWO_PI = 2 * np.pi
CELLS = {
    # name: (family, n, intercept, a1, a3, sigma)
    "gauss": ("gaussian", 200, 0.0, 1.0, 0.30, 1.0),
    "gauss_small": ("gaussian", 60, 0.0, 1.0, 0.30, 0.5),
    "binom": ("binomial", 400, 0.0, 1.5, 0.60, None),
    "pois": ("poisson", 200, 0.5, 0.8, 0.25, None),
}
KEYS = (
    "statistic_lr",
    "statistic_corrected",
    "ref_df",
    "bartlett_factor",
    "p_value_uncorrected",
    "p_value_corrected",
    "p_value_conditional",
    "p_value_bound",
    "correction_provenance",
    "reference_source",
    "reference_weights",
    "reference_residual_df",
    "reference_deterministic_offset",
)


def simulate(cell, rep):
    fam, n, b0, a1, a3, sigma = CELLS[cell]
    signal = float(os.environ.get("LR_MC_SIGNAL", "1"))
    a1, a3 = signal * a1, signal * a3
    rng = np.random.default_rng(50000 + rep)
    x = rng.uniform(0, 1, (n, 3))
    eta = b0 + a1 * np.sin(TWO_PI * x[:, 0]) + a3 * np.cos(TWO_PI * x[:, 2])
    if fam == "gaussian":
        y = eta + rng.normal(0, sigma, n)
    elif fam == "binomial":
        y = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float)
    else:
        y = rng.poisson(np.exp(eta)).astype(float)
    return fam, dict(x1=x[:, 0], x2=x[:, 1], x3=x[:, 2], y=y)


def one(args):
    cell, rep = args
    warnings.simplefilter("ignore")
    import gamfit

    fam, data = simulate(cell, rep)
    try:
        formula = os.environ.get("LR_MC_FORMULA", "y ~ s(x1) + s(x2) + s(x3)")
        model = gamfit.fit(data, formula, family=fam)
        rows = model.smooth_significance(data)
    except Exception as error:  # a failed fit is counted, not hidden
        return {"rep": rep, "error": str(error)[:300]}
    return {
        "rep": rep,
        "terms": {row["name"]: {k: row.get(k) for k in KEYS} for row in rows},
    }


def ks(p):
    p = np.sort(np.asarray(p))
    m = len(p)
    grid = np.arange(1, m + 1) / m
    d = max(np.max(grid - p), np.max(p - (grid - 1 / m)))
    # Kolmogorov asymptotic tail with the Stephens small-sample adjustment.
    lam = (np.sqrt(m) + 0.12 + 0.11 / np.sqrt(m)) * d
    k = np.arange(1, 101)
    pval = float(np.clip(2 * np.sum((-1) ** (k - 1) * np.exp(-2 * (k * lam) ** 2)), 0, 1))
    return float(d), pval


def report(cell, results, key="p_value_corrected"):
    ok = [r for r in results if "terms" in r]
    lines = [f"{cell}: {len(ok)}/{len(results)} fits"]
    null = np.array(
        [r["terms"]["s(x2)"][key] for r in ok if r["terms"]["s(x2)"][key] is not None]
    )
    lines.append(f"  null p-value unavailable on {len(ok) - len(null)} fits")
    weak = np.array(
        [r["terms"]["s(x3)"][key] for r in ok if r["terms"]["s(x3)"][key] is not None]
    )
    for alpha in (0.10, 0.05, 0.01):
        rate = np.mean(null <= alpha)
        mcse = np.sqrt(alpha * (1 - alpha) / len(null))
        lines.append(
            f"  size@{alpha:.2f} = {rate:.4f}  (MCSE {mcse:.4f}, z = {(rate - alpha) / mcse:+.2f})"
        )
    d, pks = ks(null)
    lines.append(f"  KS D = {d:.4f}, p = {pks:.3f}")
    lines.append(f"  power@.05 on s(x3) = {np.mean(weak <= 0.05):.4f}")
    return "\n".join(lines)


if __name__ == "__main__":
    from multiprocessing import Pool

    cell, nrep, workers = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    out = sys.argv[4] if len(sys.argv) > 4 else f"lr_null_{cell}.json"
    with Pool(workers) as pool:
        results = pool.map(one, [(cell, rep) for rep in range(nrep)], chunksize=1)
    with open(out, "w") as handle:
        json.dump(results, handle)
    for key in ("p_value_corrected", "p_value_uncorrected", "p_value_conditional"):
        print(f"[{key}]")
        print(report(cell, results, key))
