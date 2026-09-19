"""Seeded Monte Carlo calibration for lane pv-model-comparison.

Surface 2, ``basis_check``: the penalized score lack-of-fit p-value that
``Summary.basis_checks`` / ``Model.basis_check`` report per smooth. Under an
adequate basis (a smooth truth, default ``s(x)``) it must satisfy
``P(p <= a) <= a``; on a ``k=4`` fit of a ``sin(6x)`` truth it should reject.

Surface 1, ``compare_models``: it returns no nested-model p-value, only an
information-criterion ranking and its gaps. The null-nested cells record how
often that ranking picks a model whose extra smooth is pure noise, and that
the comparison document carries no p-value field.

Usage: python calibrate.py [--null-reps 1000] [--power-reps 500]
       [--compare-reps 200] [--jobs 4] [--out results.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import multiprocessing
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats

import gamfit

LEVELS = (0.10, 0.05, 0.01)
FAMILIES = ("gaussian", "binomial", "poisson")
SIZES = (200, 2000)


def draw_response(family: str, eta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if family == "gaussian":
        return eta + rng.normal(0.0, 0.5, eta.size)
    if family == "binomial":
        return rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)
    if family == "poisson":
        return rng.poisson(np.exp(eta)).astype(float)
    raise ValueError(family)


def summarize(p_values: list[float], reps: int) -> dict:
    p = np.asarray(p_values, dtype=float)
    out = {"tested": int(p.size), "reps": reps}
    for level in LEVELS:
        rate = float(np.mean(p <= level)) if p.size else math.nan
        out[f"size_{level:.2f}"] = rate
        out[f"mcse_{level:.2f}"] = math.sqrt(level * (1.0 - level) / max(p.size, 1))
    if p.size:
        ks = stats.kstest(p, "uniform")
        out["ks_statistic"] = float(ks.statistic)
        out["ks_p_value"] = float(ks.pvalue)
    return out


TRUTHS = {
    "sin2pi": lambda x: np.sin(2.0 * np.pi * x),
    "sin6": lambda x: np.sin(6.0 * x),
}


def basis_check_cell(family, n, formula, truth, reps, seed):
    truth_fn = TRUTHS[truth]
    rng = np.random.default_rng(seed)
    p_values, provenances, fit_refused = [], {}, 0
    for _ in range(reps):
        x = rng.uniform(0.0, 1.0, n)
        y = draw_response(family, truth_fn(x), rng)
        try:
            model = gamfit.fit({"x": x, "y": y}, formula, family=family)
        except gamfit.FitError:
            # A fit the outer optimizer could not certify is refused, so it
            # has no basis_check to calibrate; count it and move on.
            fit_refused += 1
            continue
        row = model.summary().basis_checks[0]
        provenances[row["provenance"]] = provenances.get(row["provenance"], 0) + 1
        if row["p_value"] is not None:
            p_values.append(float(row["p_value"]))
    result = summarize(p_values, reps)
    result.update(
        family=family, n=n, formula=formula, truth=truth, provenance=provenances, fit_refused=fit_refused
    )
    return result


def compare_models_cell(family, n, reps, seed):
    rng = np.random.default_rng(seed)
    larger_wins = 0
    unranked = {}
    p_keys = set()
    for _ in range(reps):
        x = rng.uniform(0.0, 1.0, n)
        z = rng.uniform(0.0, 1.0, n)
        y = draw_response(family, np.sin(2.0 * np.pi * x), rng)
        data = {"x": x, "z": z, "y": y}
        try:
            small = gamfit.fit(data, "y ~ s(x)", family=family)
            large = gamfit.fit(data, "y ~ s(x) + s(z)", family=family)
        except gamfit.FitError:
            unranked["fit refused"] = unranked.get("fit refused", 0) + 1
            continue
        try:
            doc = gamfit.compare_models([small, large], names=["small", "large"])
        except ValueError as refusal:
            reason = str(refusal).split(": ", 2)[-1]
            unranked[reason] = unranked.get(reason, 0) + 1
            continue
        larger_wins += doc["winner"] == "large"
        text = json.dumps(doc)
        p_keys.update(k for k in ("p_value", "pvalue", "p-value") if k in text)
    return {
        "family": family,
        "n": n,
        "reps": reps,
        "ranked": reps - sum(unranked.values()),
        "larger_model_selected": larger_wins / max(reps - sum(unranked.values()), 1),
        "unranked": unranked,
        "p_value_fields": sorted(p_keys),
    }


def cells(null_reps: int, power_reps: int, compare_reps: int, with_compare: bool):
    """Every cell of the study, each with its own fixed seed, in report order."""
    seed = 20260919
    out = []
    for family in FAMILIES:
        for n in SIZES:
            seed += 1
            out.append(("null", basis_check_cell, (family, n, "y ~ s(x)", "sin2pi", null_reps, seed)))
    for family in FAMILIES:
        for n in SIZES:
            seed += 1
            out.append(("power", basis_check_cell, (family, n, "y ~ s(x, k=4)", "sin6", power_reps, seed)))
    if with_compare:
        for family in FAMILIES:
            for n in SIZES:
                seed += 1
                out.append(("compare_models_null", compare_models_cell, (family, n, compare_reps, seed)))
    return out


def run_cell(cell):
    kind, function, arguments = cell
    warnings.simplefilter("ignore")
    t0 = time.time()
    result = function(*arguments)
    result["seed"] = arguments[-1]
    result["seconds"] = round(time.time() - t0, 1)
    return kind, result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--null-reps", type=int, default=1000)
    parser.add_argument("--power-reps", type=int, default=500)
    parser.add_argument("--compare-reps", type=int, default=200)
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--jobs", type=int, default=1, help="cells run in parallel processes")
    parser.add_argument("--out", default="results.json")
    args = parser.parse_args()
    plan = cells(args.null_reps, args.power_reps, args.compare_reps, not args.skip_compare)
    results = {"null": [], "power": [], "compare_models_null": []}
    # Spawned, not forked: a forked child inherits the engine's thread pool in
    # whatever state the parent left it and can deadlock on its first fit.
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=context) as pool:
        for kind, result in pool.map(run_cell, plan):
            results[kind].append(result)
            print(json.dumps(result), flush=True)
    with open(args.out, "w") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
