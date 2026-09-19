"""Null calibration of the Wood (2013) smooth-term Wald p-value, per family.

Model: y ~ s(x1) + s(x2), with a real s(x1) and a null s(x2).
Output: one JSON line per replicate: family, n, rep, p-value, edf, ref_df.
"""
import json
import os
import sys
import warnings
from multiprocessing import Pool

import numpy as np

warnings.filterwarnings("ignore")

FAMILY_SPEC = {"tweedie": "tweedie(p=1.5)"}
FAMILIES = ["gaussian", "poisson", "binomial", "gamma", "negative-binomial", "tweedie", "beta"]


def simulate(family, n, seed, effect=0.0):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    f1 = np.sin(2 * np.pi * x1)
    f2 = effect * np.sin(2 * np.pi * x2)
    if family == "gaussian":
        y = f1 + f2 + rng.normal(0, 0.5, n)
    elif family == "poisson":
        y = rng.poisson(np.exp(0.5 + 0.5 * f1 + f2)).astype(float)
    elif family == "binomial":
        p = 1 / (1 + np.exp(-(f1 + f2)))
        y = (rng.uniform(size=n) < p).astype(float)
    elif family == "gamma":
        mu = np.exp(1 + 0.5 * f1 + f2)
        shape = 3.0
        y = rng.gamma(shape, mu / shape)
    elif family == "negative-binomial":
        mu = np.exp(1 + 0.5 * f1 + f2)
        theta = 2.0
        y = rng.negative_binomial(theta, theta / (theta + mu)).astype(float)
    elif family == "tweedie":
        mu = np.exp(0.5 + 0.5 * f1 + f2)
        p, phi = 1.5, 1.0
        lam = mu ** (2 - p) / (phi * (2 - p))
        alpha = (2 - p) / (p - 1)
        scale = phi * (p - 1) * mu ** (p - 1)
        counts = rng.poisson(lam)
        y = np.array([rng.gamma(alpha * c, s) if c > 0 else 0.0 for c, s in zip(counts, scale)])
    elif family == "beta":
        mu = 1 / (1 + np.exp(-(0.5 * f1 + f2)))
        phi = 10.0
        y = rng.beta(mu * phi, (1 - mu) * phi)
        y = np.clip(y, 1e-6, 1 - 1e-6)
    else:
        raise ValueError(family)
    return {"y": y, "x1": x1, "x2": x2}


def one(task):
    family, n, rep, effect = task
    import gamfit

    seed = 7_000_000 + 10_000_000 * FAMILIES.index(family) + 1000 * n + rep
    data = simulate(family, n, seed, effect)
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    os.dup2(devnull, 2)
    try:
        model = gamfit.fit(data, "y ~ s(x1) + s(x2)", family=FAMILY_SPEC.get(family, family))
        rows = model.summary().smooth_terms
        row = [r for r in rows if r["name"].endswith("x2)") or r["name"] == "s(x2)"][0]
        return {
            "family": family, "n": n, "rep": rep, "effect": effect,
            "p": row.get("p_value"), "edf": row.get("edf"), "ref_df": row.get("ref_df"),
            "chi_sq": row.get("chi_sq"), "fam": model.family_name,
        }
    except Exception as exc:  # recorded, never skipped silently
        return {"family": family, "n": n, "rep": rep, "effect": effect, "error": str(exc)[:300]}
    finally:
        os.dup2(saved, 2)
        os.close(devnull)


if __name__ == "__main__":
    families = sys.argv[1].split(",")
    ns = [int(v) for v in sys.argv[2].split(",")]
    reps = int(sys.argv[3])
    effect = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    out = sys.argv[5] if len(sys.argv) > 5 else "/dev/stdout"
    tasks = [(f, n, r, effect) for f in families for n in ns for r in range(reps)]
    with Pool(int(os.environ.get("NPROC", "4"))) as pool, open(out, "a") as fh:
        for res in pool.imap_unordered(one, tasks, chunksize=4):
            fh.write(json.dumps(res) + "\n")
            fh.flush()
