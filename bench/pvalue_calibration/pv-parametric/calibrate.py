"""Seeded Monte Carlo calibration of the parametric-term p-values in `summary()`.

Each cell fits `y ~ x1 + g + s(x2)` to data with a real smooth in `x2`, and
under the null neither the linear coefficient `x1` nor the 4-level factor `g`
enters the linear predictor. Under the alternative both do. Every p-value read
here is the one the summary payload carries: the linear coefficient's row in
`parametric_terms`, and the factor's joint Wald row in `parametric_term_tests`.

Reported per cell and test: rejection rate at 0.10 / 0.05 / 0.01 with its Monte
Carlo standard error, the Kolmogorov-Smirnov distance of the null p-values from
U(0, 1) with its asymptotic p-value, and power at 0.05.

Each dataset is also tested by an oracle that knows the true smooth shape
`sin(2 pi x2)` and fits it as a covariate, unpenalized:

- Gaussian: the least-squares t / F test. Its p-values are exactly U(0, 1) under
  the null, so its rejection rate on the same datasets is the Monte Carlo
  baseline the engine's rate is compared with, pair by pair.
- Binomial and Poisson: the maximum-likelihood Wald z / chi-square test and the
  likelihood-ratio test. Neither is exact; both are the standard large-sample
  tests, and their rates on the same datasets separate what the seed set does
  from what the engine does.

    python bench/pvalue_calibration/pv-parametric/calibrate.py --reps 1000 \
        --out bench/pvalue_calibration/pv-parametric/results.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats

LEVELS = ("a", "b", "c", "d")
ALPHAS = (0.10, 0.05, 0.01)

# (name, family, n, intercept, smooth amplitude, x1 effect, factor effects, sigma)
CELLS = {
    "gauss30": ("gaussian", 30, 0.0, 1.0, 0.6, (0.0, 0.5, -0.5, 0.8), 0.5),
    "gauss200": ("gaussian", 200, 0.0, 1.0, 0.25, (0.0, 0.2, -0.2, 0.3), 1.0),
    "binom": ("binomial", 400, 0.0, 1.5, 0.5, (0.0, 0.4, -0.4, 0.6), None),
    "pois": ("poisson", 200, 0.5, 0.8, 0.3, (0.0, 0.25, -0.25, 0.35), None),
}

FORMULA = "y ~ x1 + g + s(x2)"


def simulate(cell: str, rep: int, alternative: bool) -> pd.DataFrame:
    family, n, b0, amp, beta1, factor_effects, sigma = CELLS[cell]
    rng = np.random.default_rng([list(CELLS).index(cell), rep, int(alternative)])
    x1 = rng.uniform(-1.0, 1.0, n)
    x2 = rng.uniform(0.0, 1.0, n)
    codes = rng.integers(0, len(LEVELS), n)
    eta = b0 + amp * np.sin(2.0 * np.pi * x2)
    if alternative:
        eta = eta + beta1 * x1 + np.asarray(factor_effects)[codes]
    if family == "gaussian":
        y = eta + rng.normal(0.0, sigma, n)
    elif family == "binomial":
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)
    else:
        y = rng.poisson(np.exp(eta)).astype(float)
    g = pd.Categorical(np.asarray(LEVELS)[codes], categories=LEVELS)
    return pd.DataFrame({"x1": x1, "x2": x2, "g": g, "y": y})


def oracle_pvalues(frame: pd.DataFrame) -> dict:
    """Exact least-squares t (x1) and F (g) p-values with the true smooth shape known."""
    codes = frame["g"].cat.codes.to_numpy()
    dummies = np.eye(len(LEVELS))[codes][:, 1:]
    x = np.column_stack(
        [np.ones(len(frame)), frame["x1"], dummies, np.sin(2.0 * np.pi * frame["x2"])]
    )
    y = frame["y"].to_numpy()
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    df = len(y) - x.shape[1]
    covariance = (residual @ residual / df) * np.linalg.inv(x.T @ x)
    t = beta[1] / math.sqrt(covariance[1, 1])
    block = slice(2, 2 + dummies.shape[1])
    f = beta[block] @ np.linalg.solve(covariance[block, block], beta[block]) / dummies.shape[1]
    return {
        "x1": float(2.0 * stats.t.sf(abs(t), df)),
        "g": float(stats.f.sf(f, dummies.shape[1], df)),
    }


def _design(frame: pd.DataFrame, x1: bool, g: bool) -> np.ndarray:
    codes = frame["g"].cat.codes.to_numpy()
    columns = [np.ones(len(frame)), np.sin(2.0 * np.pi * frame["x2"].to_numpy())]
    if x1:
        columns.append(frame["x1"].to_numpy())
    if g:
        columns.extend(np.eye(len(LEVELS))[codes][:, 1:].T)
    return np.column_stack(columns)


def _glm_mle(x: np.ndarray, y: np.ndarray, family: str):
    """Unpenalized canonical-link MLE by Newton's method (Fisher scoring).

    Returns the estimate, its inverse Fisher information and the log-likelihood.
    Iterates until the Newton decrement is at rounding level.
    """
    beta = np.zeros(x.shape[1])
    if family == "binomial":
        beta[0] = math.log((y.mean()) / (1.0 - y.mean()))
    else:
        beta[0] = math.log(y.mean())
    while True:
        eta = x @ beta
        if family == "binomial":
            mu = 1.0 / (1.0 + np.exp(-eta))
            w = mu * (1.0 - mu)
        else:
            mu = np.exp(eta)
            w = mu
        info = x.T @ (w[:, None] * x)
        step = np.linalg.solve(info, x.T @ (y - mu))
        beta = beta + step
        if step @ info @ step <= np.finfo(float).eps * max(1.0, abs(float(y @ eta))):
            break
    eta = x @ beta
    if family == "binomial":
        loglik = float(y @ eta - np.logaddexp(0.0, eta).sum())
    else:
        loglik = float(y @ eta - np.exp(eta).sum())
    return beta, np.linalg.inv(info), loglik


def glm_oracle_pvalues(frame: pd.DataFrame, family: str) -> dict:
    """Wald and likelihood-ratio p-values of the unpenalized MLE, true smooth shape known."""
    y = frame["y"].to_numpy()
    beta, covariance, full = _glm_mle(_design(frame, True, True), y, family)
    wald_z = beta[2] / math.sqrt(covariance[2, 2])
    block = slice(3, 3 + len(LEVELS) - 1)
    wald_g = beta[block] @ np.linalg.solve(covariance[block, block], beta[block])
    without_x1 = _glm_mle(_design(frame, False, True), y, family)[2]
    without_g = _glm_mle(_design(frame, True, False), y, family)[2]
    q = len(LEVELS) - 1
    return {
        "x1": float(2.0 * stats.norm.sf(abs(wald_z))),
        "g": float(stats.chi2.sf(wald_g, q)),
        "lr_x1": float(stats.chi2.sf(max(2.0 * (full - without_x1), 0.0), 1)),
        "lr_g": float(stats.chi2.sf(max(2.0 * (full - without_g), 0.0), q)),
    }


def quiet_worker() -> None:
    """The engine's diagnostic stream (fd 2) is not part of the measurement."""
    os.dup2(os.open(os.devnull, os.O_WRONLY), 2)


def one_rep(args):
    cell, rep, alternative = args
    import gamfit

    warnings.simplefilter("ignore")
    family = CELLS[cell][0]
    frame = simulate(cell, rep, alternative)
    oracle = oracle_pvalues(frame) if family == "gaussian" else glm_oracle_pvalues(frame, family)
    try:
        summary = gamfit.fit(frame, FORMULA, family=family).summary()
    except Exception as error:  # a failed fit is counted, never dropped silently
        return {"error": f"{type(error).__name__}: {error}"[:200]}
    linear = {row["name"]: row for row in summary.parametric_terms}
    tests = {row["name"]: row for row in summary.parametric_term_tests}
    return {
        "x1": linear.get("x1", {}).get("p_value"),
        "g": tests.get("g", {}).get("p_value"),
        "g_unavailable": tests.get("g", {}).get("p_value_unavailable"),
        "oracle_x1": oracle.get("x1"),
        "oracle_g": oracle.get("g"),
        "oracle_lr_x1": oracle.get("lr_x1"),
        "oracle_lr_g": oracle.get("lr_g"),
    }


def kolmogorov_p(d: float, m: int) -> float:
    """Asymptotic P(D_m >= d) under U(0, 1) (Kolmogorov series, Stephens' m-correction)."""
    t = (math.sqrt(m) + 0.12 + 0.11 / math.sqrt(m)) * d
    total = sum((-1) ** (k - 1) * math.exp(-2.0 * k * k * t * t) for k in range(1, 101))
    return min(1.0, max(0.0, 2.0 * total))


def ks_distance(p: np.ndarray) -> float:
    p = np.sort(p)
    m = len(p)
    ranks = np.arange(1, m + 1)
    return float(max(np.max(ranks / m - p), np.max(p - (ranks - 1) / m)))


def report(null: list[float | None], alternative: list[float | None]) -> dict:
    p = np.asarray([v for v in null if v is not None], dtype=float)
    q = np.asarray([v for v in alternative if v is not None], dtype=float)
    out = {"null_usable": int(len(p)), "alternative_usable": int(len(q))}
    for alpha in ALPHAS:
        rate = float(np.mean(p <= alpha)) if len(p) else float("nan")
        out[f"size@{alpha}"] = rate
        mcse = math.sqrt(alpha * (1 - alpha) / max(len(p), 1))
        out[f"mcse@{alpha}"] = mcse
        # The acceptance band: the rejection rate is within two Monte Carlo
        # standard errors of the nominal level.
        out[f"within_2mcse@{alpha}"] = bool(abs(rate - alpha) <= 2.0 * mcse)
    d = ks_distance(p) if len(p) else float("nan")
    out["ks_distance"] = d
    out["ks_p"] = kolmogorov_p(d, len(p)) if len(p) else float("nan")
    out["power@0.05"] = float(np.mean(q <= 0.05)) if len(q) else float("nan")
    out["null_p"] = p.tolist()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--cells", default=",".join(CELLS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    results = {}
    with ProcessPoolExecutor(args.workers, initializer=quiet_worker) as pool:
        for cell in args.cells.split(","):
            jobs = [(cell, rep, alt) for alt in (False, True) for rep in range(args.reps)]
            rows = list(pool.map(one_rep, jobs, chunksize=8))
            null, alt = rows[: args.reps], rows[args.reps :]
            errors = [r["error"] for r in rows if "error" in r]
            results[cell] = {
                "reps": args.reps,
                "fit_errors": len(errors),
                "fit_error_examples": sorted(set(errors))[:5],
                "g_unavailable": sorted({r["g_unavailable"] for r in rows if r.get("g_unavailable")}),
                "x1": report([r.get("x1") for r in null], [r.get("x1") for r in alt]),
                "g": report([r.get("g") for r in null], [r.get("g") for r in alt]),
            }
            oracles = ("x1", "g") if CELLS[cell][0] == "gaussian" else ("x1", "g", "lr_x1", "lr_g")
            for term in oracles:
                key = f"oracle_{term}"
                results[cell][key] = report(
                    [r.get(key) for r in null], [r.get(key) for r in alt]
                )
            brief = {
                name: ({k: v for k, v in value.items() if k != "null_p"} if isinstance(value, dict) else value)
                for name, value in results[cell].items()
            }
            print(cell, json.dumps(brief, indent=1), flush=True)
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(results, handle, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
