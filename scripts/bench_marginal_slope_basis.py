"""Benchmark independent 1-D B-splines against a joint 3-D Duchon smooth."""
from __future__ import annotations

import time

import numpy as np

import gamfit


def log(msg: str) -> None:
    print(msg, flush=True)


def synth(n: int, seed: int = 11) -> dict[str, list[float]]:
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


def stage1_calibrate(data: dict[str, list[float]], formula: str) -> dict[str, list[float]]:
    log(f"[stage1] fitting {formula!r}")
    calib = gamfit.fit(data, formula, transformation_normal=True, scale_dimensions=True)
    z = np.asarray(calib.predict(data), dtype=float)
    out = dict(data)
    out["pgs_z"] = z.tolist()
    return out


def time_fit(
    label: str,
    data: dict[str, list[float]],
    formula: str,
    logslope_formula: str,
    repeats: int,
) -> list[float]:
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
    n = 2000
    log(f"=== benchmark: n={n}, predictors=3 ===")

    raw = synth(n)
    data = stage1_calibrate(raw, "PGS ~ duchon(x1, x2, x3, centers=30)")

    runs = 3
    results = {
        "indep_bsplines": time_fit(
            "indep s(x1)+s(x2)+s(x3)",
            data,
            "case ~ s(x1) + s(x2) + s(x3)",
            "s(x1) + s(x2) + s(x3)",
            repeats=runs,
        ),
        "joint_duchon": time_fit(
            "joint duchon(x1, x2, x3, centers=30)",
            data,
            "case ~ duchon(x1, x2, x3, centers=30)",
            "duchon(x1, x2, x3, centers=30)",
            repeats=runs,
        ),
    }

    log("")
    log("=== summary ===")
    log(f"{'variant':<42s}  {'median (s)':>11s}  {'min (s)':>9s}  {'max (s)':>9s}")
    log("-" * 74)
    for name, times in results.items():
        med = float(np.median(times))
        lo = float(np.min(times))
        hi = float(np.max(times))
        log(f"{name:<42s}  {med:>11.2f}  {lo:>9.2f}  {hi:>9.2f}")


if __name__ == "__main__":
    main()
