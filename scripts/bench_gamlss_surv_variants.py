"""Benchmark four GAMLSS survival location-scale variants."""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, TypedDict

import numpy as np


class FitKwargs(TypedDict, total=False):
    survival_likelihood: str
    baseline_target: str
    baseline_scale: float
    baseline_shape: float
    baseline_rate: float
    baseline_makeham: float
    config: dict[str, Any]


Data = dict[str, list[float]]
BenchVariant = tuple[str, str, FitKwargs]


def synth(n: int = 200, seed: int = 17) -> Data:
    rng = np.random.default_rng(seed)
    bmi = rng.normal(27.0, 4.0, n)
    hba1c = rng.normal(5.8, 0.7, n)
    ldl = rng.normal(120.0, 30.0, n)

    bmi_z = (bmi - 27.0) / 4.0
    hba1c_z = (hba1c - 5.8) / 0.7
    ldl_z = (ldl - 120.0) / 30.0

    mu = 3.5 - 0.4 * bmi_z - 0.5 * hba1c_z - 0.25 * ldl_z + 0.18 * bmi_z * hba1c_z
    log_sigma = -0.4 + 0.18 * np.abs(bmi_z) + 0.12 * hba1c_z**2 * 0.3
    sigma = np.exp(log_sigma)

    eps = rng.logistic(size=n)
    log_T = mu + sigma * eps
    T = np.exp(log_T)

    C = rng.uniform(5.0, 25.0, n)
    exit_t = np.minimum(T, C)
    event = (T <= C).astype(float)
    entry = np.zeros(n)

    return {
        "entry": entry.tolist(),
        "exit": exit_t.tolist(),
        "event": event.tolist(),
        "bmi": bmi.tolist(),
        "hba1c": hba1c.tolist(),
        "ldl": ldl.tolist(),
    }


def fit_variant(
    args: tuple[str, str, FitKwargs, Data],
) -> tuple[str, float, bool, str | None]:
    label, formula, fit_kwargs, data = args

    t0 = time.perf_counter()
    try:
        import gamfit
        gamfit.fit(data, formula, **fit_kwargs)
        dt = time.perf_counter() - t0
        return label, dt, True, None
    except Exception as exc:
        dt = time.perf_counter() - t0
        return label, dt, False, str(exc)[:300]


def main() -> None:
    print("=== bench: GAMLSS-survival-location-scale variants ===", flush=True)
    print("n=200, 3 predictors (bmi, hba1c, ldl)", flush=True)

    data = synth()

    location_full = "Surv(entry, exit, event) ~ s(bmi) + s(hba1c) + s(ldl)"
    scale_full = "s(bmi) + s(hba1c) + s(ldl)"
    scale_lean = "s(bmi)"

    variants: list[BenchVariant] = [
        (
            "A: gompertz-makeham + full sigma",
            location_full,
            {
                "survival_likelihood": "location-scale",
                "baseline_target": "gompertz-makeham",
                "baseline_rate": 0.02,
                "baseline_makeham": 0.001,
                "config": {"noise_formula": scale_full},
            },
        ),
        (
            "B: weibull + full sigma",
            location_full,
            {
                "survival_likelihood": "location-scale",
                "baseline_target": "weibull",
                "baseline_scale": 1.0,
                "baseline_shape": 1.2,
                "config": {"noise_formula": scale_full},
            },
        ),
        (
            "C: gompertz-makeham + lean sigma",
            location_full,
            {
                "survival_likelihood": "location-scale",
                "baseline_target": "gompertz-makeham",
                "baseline_rate": 0.02,
                "baseline_makeham": 0.001,
                "config": {"noise_formula": scale_lean},
            },
        ),
        (
            "D: linear baseline + full sigma",
            location_full,
            {
                "survival_likelihood": "location-scale",
                "baseline_target": "linear",
                "config": {"noise_formula": scale_full},
            },
        ),
    ]

    print(f"submitting {len(variants)} fits in parallel...", flush=True)

    wall_start = time.perf_counter()
    results: list[tuple[str, float, bool, str | None]] = []
    with ProcessPoolExecutor(max_workers=len(variants)) as pool:
        futures = [
            pool.submit(fit_variant, (label, formula, fit_kwargs, data))
            for label, formula, fit_kwargs in variants
        ]
        for fut in as_completed(futures):
            label, dt, ok, err = fut.result()
            status = "OK " if ok else "FAIL"
            short_err = f"  err={err}" if err else ""
            print(f"  [{status}] {label:<40s} {dt:8.2f} s{short_err}", flush=True)
            results.append((label, dt, ok, err))
    wall_total = time.perf_counter() - wall_start

    print(f"\ntotal wall clock (parallel): {wall_total:.2f} s", flush=True)

    print("\n=== summary ===", flush=True)
    print(f"{'variant':<40s}  {'sec':>8s}  {'status':>7s}", flush=True)
    print("-" * 64, flush=True)
    for label, dt, ok, _ in sorted(results, key=lambda r: r[1]):
        print(f"{label:<40s}  {dt:>8.2f}  {'ok' if ok else 'FAIL':>7s}", flush=True)

    out = Path(__file__).resolve().parent / "bench_gamlss_surv_variants_results.json"
    out.write_text(
        json.dumps(
            [
                {"label": label, "seconds": dt, "ok": ok, "error": err}
                for label, dt, ok, err in results
            ],
            indent=2,
        )
    )
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
