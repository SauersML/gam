#!/usr/bin/env python
"""Benchmark `gamfit.torch.fit` across (F, D, mode) combinations.

Run after a fresh `pip install gamfit>=0.1.110` (or local maturin
develop) to characterize the joint-vs-independent dispatch on the
machine you'll deploy on. Emits a markdown summary to stdout plus a
JSON file with the raw measurements.

Usage::

    python tools/bench_reml.py --F-values 16,100,1024,4096 --D 64 --output bench.json

The script SKIPS combos that would obviously OOM (e.g. F=4096 with
mode="joint" allocates a 24K×24K joint design Cholesky — feasible but
slow; documented as SKIP-large unless --force-joint is set).
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch


@contextmanager
def _measure_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        yield lambda: torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # On macOS ru_maxrss is in bytes; on Linux it's in kB. Normalize
        # to MB by checking magnitude.
        scale = 1.0 / (1024 ** 2) if sys.platform == "darwin" else 1.0 / 1024
        yield lambda: max(0.0, (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before)) * scale


def _build_inputs(N: int, F: int, M: int, D: int, device: str, seed: int = 0):
    torch.manual_seed(seed)
    t = torch.linspace(0.01, 0.99, N, device=device, dtype=torch.float64)
    # `points` and `amps` carry gradients so backward() has work to do.
    points = [t.unsqueeze(1).clone().requires_grad_(True) for _ in range(F)]
    response = torch.randn(N, D, device=device, dtype=torch.float64)
    # NOTE: use uniform-positive amps (not relu(randn)). Atoms with all-zero
    # `by` rows in the batch make the per-atom REML's inner Hessian singular
    # and the backward returns NaN; tracked as a separate bug in
    # `_fit_independent`. Production SAE TopK gating still hits this when an
    # atom's batch-activation count is zero — but the bench harness should
    # measure the steady-state cost, not the degenerate case.
    amps = (torch.rand(N, F, device=device, dtype=torch.float64) * 0.9 + 0.1).requires_grad_(True)
    centers = torch.linspace(0, 1, M, device=device, dtype=torch.float64).reshape(-1, 1)
    return points, response, centers, amps


def bench_one(*, N: int, F: int, M: int, D: int, mode: str, device: str,
              warmup: int, measure: int) -> dict:
    try:
        import gamfit
        from gamfit import Duchon
        from gamfit.torch import fit
    except ImportError:
        return {"status": "import_failure", "error": "gamfit.torch unavailable"}

    points, response, centers, amps = _build_inputs(N, F, M, D, device)
    smooths = [Duchon(centers=centers, m=2, by=amps[:, k]) for k in range(F)]

    # Skip obviously-infeasible joint combos
    joint_dim = F * M
    if mode == "joint":
        if joint_dim > 16000:
            return {"status": "skipped", "reason": f"joint dim {joint_dim} > 16000"}
        if D > 1:
            return {"status": "skipped", "reason": f"joint requires D=1, got D={D}"}

    try:
        # Warmup — tolerate backward failures (degenerate-λ atoms)
        for _ in range(warmup):
            r = fit(points=points, response=response, smooths=smooths, mode=mode)
            try:
                loss = r.fitted.sum()
                loss.backward()
            except Exception:
                pass

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Measure forward + backward. Backward can fail on rare per-atom
        # near-singular K (a known degenerate-λ case being fixed separately);
        # in that case report forward timing and "bwd: error".
        forward_times, backward_times = [], []
        backward_error: str | None = None
        with _measure_memory() as peak_mb_fn:
            for _ in range(measure):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                r = fit(points=points, response=response, smooths=smooths, mode=mode)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                forward_times.append(t1 - t0)
                try:
                    loss = r.fitted.sum()
                    loss.backward()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t2 = time.perf_counter()
                    backward_times.append(t2 - t1)
                except Exception as exc:
                    backward_error = f"{type(exc).__name__}: {str(exc)[:80]}"
            peak_mb = peak_mb_fn()

        result = {
            "status": "ok" if backward_error is None else "ok_fwd_bwd_failed",
            "forward_s_mean": sum(forward_times) / len(forward_times),
            "forward_s_min": min(forward_times),
            "peak_mb": peak_mb,
        }
        if backward_times:
            result["backward_s_mean"] = sum(backward_times) / len(backward_times)
            result["backward_s_min"] = min(backward_times)
        if backward_error is not None:
            result["backward_error"] = backward_error
        return result
    except NotImplementedError as exc:
        return {"status": "not_implemented", "error": str(exc)[:200]}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {str(exc)[:200]}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--F-values", default="16,100,1024,4096")
    parser.add_argument("--N", type=int, default=256)
    parser.add_argument("--M-per-smooth", type=int, default=8)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--modes", default="joint,independent,auto")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("bench_reml.json"))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measure", type=int, default=3)
    args = parser.parse_args()

    try:
        import gamfit
        ver = gamfit.__version__
    except ImportError:
        print("gamfit.torch unavailable — install gamfit>=0.1.110", file=sys.stderr)
        return 1

    F_values = [int(x) for x in args.F_values.split(",")]
    modes = [x.strip() for x in args.modes.split(",")]

    print(f"# REML bench  (gamfit {ver}, device={args.device}, N={args.N}, M={args.M_per_smooth}, D={args.D})\n")
    print(f"| F | mode | forward (ms) | backward (ms) | peak MB | status |")
    print(f"|---|------|-------------:|--------------:|--------:|--------|")

    results = []
    for F in F_values:
        for mode in modes:
            r = bench_one(
                N=args.N, F=F, M=args.M_per_smooth, D=args.D, mode=mode,
                device=args.device, warmup=args.warmup, measure=args.measure,
            )
            results.append({"F": F, "mode": mode, "D": args.D, "N": args.N, "M": args.M_per_smooth, **r})
            if r.get("status") == "ok":
                print(
                    f"| {F} | {mode} | {r['forward_s_mean']*1000:.1f} "
                    f"| {r['backward_s_mean']*1000:.1f} | {r['peak_mb']:.1f} | ok |"
                )
            elif r.get("status") == "ok_fwd_bwd_failed":
                print(
                    f"| {F} | {mode} | {r['forward_s_mean']*1000:.1f} "
                    f"| (err) | {r['peak_mb']:.1f} | fwd-only: {r.get('backward_error','')[:50]} |"
                )
            else:
                print(f"| {F} | {mode} | — | — | — | {r.get('status', '?')}: {r.get('reason', r.get('error', ''))[:50]} |")

    args.output.write_text(json.dumps({"version": ver, "results": results}, indent=2))
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
