"""Benchmark: marginal-slope fit time with independent 1-D B-splines vs
a joint 3-D Duchon smooth, on 3 predictors at medium scale.

We hold every aspect of the data and the marginal-slope likelihood
constant; only the smooth basis differs.

  * Variant A  — `case ~ s(x1) + s(x2) + s(x3)`
                 plus `logslope_formula = "s(x1) + s(x2) + s(x3)"`
  * Variant B  — `case ~ duchon(x1, x2, x3, centers=K)`
                 plus `logslope_formula = "duchon(...)"`

Stage 1 (transformation-normal calibration) is identical for both
variants so we can compare Stage 2 cleanly.

Outputs: a single timing table to stdout and (optionally) a JSON dump
of the per-run timings.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import gamfit


def log(msg: str) -> None:
    print(msg, flush=True)


def synth(n: int, seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-1.0, 1.0, n)
    x2 = rng.uniform(-1.0, 1.0, n)
    x3 = rng.uniform(-1.0, 1.0, n)
    raw_pgs = rng.standard_normal(n) + 0.3 * x1

    # baseline log-odds varies in 3-D
    baseline = (
        -0.5
        + 1.0 * np.exp(-((x1 - 0.30) ** 2 + (x2 + 0.20) ** 2 + (x3 - 0.10) ** 2) / 0.18)
        - 0.8 * np.exp(-((x1 + 0.40) ** 2 + (x2 - 0.50) ** 2 + (x3 + 0.30) ** 2) / 0.16)
    )
    # spatially varying slope
    slope = 0.3 + 1.5 / (1.0 + np.exp(-2.5 * (x1 + x2 + x3 - 0.2)))
    eta = baseline + slope * raw_pgs
    prob = 1.0 / (1.0 + np.exp(-eta))
    case = (rng.uniform(size=n) < prob).astype(float)
    return {
        "case": case.tolist(),
        "x1":   x1.tolist(),
        "x2":   x2.tolist(),
        "x3":   x3.tolist(),
        "PGS":  raw_pgs.tolist(),
    }


def stage1_calibrate(data: dict, formula: str) -> dict:
    """Fit transformation-normal on PGS to produce a standardised z."""
    log(f"[stage1] fitting {formula!r}")
    calib = gamfit.fit(
        data, formula,
        transformation_normal=True,
        scale_dimensions=True,
    )
    z = np.asarray(calib.predict(data), dtype=float)
    out = dict(data)
    out["pgs_z"] = z.tolist()
    return out


def time_fit(label: str, data: dict, formula: str, logslope_formula: str,
             repeats: int = 1) -> list[float]:
    """Fit a marginal-slope model and time each fit."""
    log(f"[fit] {label}: starting {repeats} fit(s)")
    times: list[float] = []
    for i in range(repeats):
        t0 = time.perf_counter()
        gamfit.fit(
            data,
            formula,
            family="bernoulli-marginal-slope",
            link="probit",
            z_column="pgs_z",
            logslope_formula=logslope_formula,
            scale_dimensions=True,
        )
        dt = time.perf_counter() - t0
        log(f"[fit] {label}: run {i + 1}/{repeats} = {dt:.2f} s")
        times.append(dt)
    return times


def main() -> None:
    # Medium scale: 2000 rows, 3 predictors.
    n = 2000
    log(f"=== benchmark: n={n}, predictors=3 ===")

    raw = synth(n)

    # Stage 1: use the same calibration smooth for both Stage 2 variants
    # so any difference at Stage 2 is solely the basis choice.
    data = stage1_calibrate(raw, "PGS ~ duchon(x1, x2, x3, centers=30)")

    runs = 3
    results = {}

    # Variant A: independent 1-D B-splines.
    a_times = time_fit(
        "indep s(x1)+s(x2)+s(x3)", data,
        "case ~ s(x1) + s(x2) + s(x3)",
        "s(x1) + s(x2) + s(x3)",
        repeats=runs,
    )
    results["indep_bsplines"] = a_times

    # Variant B: joint 3-D Duchon (modest centers count so it isn't
    # crushingly expensive — same setup the marginal-slope figure uses).
    b_times = time_fit(
        "joint duchon(x1, x2, x3, centers=30)", data,
        "case ~ duchon(x1, x2, x3, centers=30)",
        "duchon(x1, x2, x3, centers=30)",
        repeats=runs,
    )
    results["joint_duchon"] = b_times

    log("")
    log("=== summary ===")
    log(f"{'variant':<42s}  {'median (s)':>11s}  {'min (s)':>9s}  {'max (s)':>9s}")
    log("-" * 74)
    for name, times in results.items():
        med = float(np.median(times))
        lo = float(np.min(times))
        hi = float(np.max(times))
        log(f"{name:<42s}  {med:>11.2f}  {lo:>9.2f}  {hi:>9.2f}")

    out = Path(__file__).resolve().parent / "bench_marginal_slope_basis_results.json"
    out.write_text(json.dumps(results, indent=2))
    log(f"\nwrote {out}")


if __name__ == "__main__":
    main()
