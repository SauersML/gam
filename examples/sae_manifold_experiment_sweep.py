#!/usr/bin/env python3
"""Small SAE manifold sweep for gamfit.sae_manifold_fit.

This script plants additive superpositions of circles in random ambient planes,
then fits the requested gamfit SAE-manifold grid:

    topology x K x decoder_incoherence_weight x d_atom

Only Python's standard library, numpy, and gamfit are used. Every cell is
isolated behind try/except so one failing fit does not abort the sweep.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import os
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np
import gamfit


TOPOLOGIES = ("circle", "euclidean", "sphere", "torus")
K_VALUES = (1, 2, 3, 5)
INCOHERENCE_WEIGHTS = (0.0, 1.0)
D_ATOMS = (1, 2)

MIN_CONVERGED_R2 = 0.05
DEFAULT_N = 120
DEFAULT_N_ITER = 15
DEFAULT_D_AMBIENT = 8
DEFAULT_K_PLANTED = 2
DEFAULT_NOISE = 0.02


@dataclass
class CellResult:
    topology: str
    k_fit: int
    incoherence_weight: float
    d_atom: int
    status: str
    r2: float
    cross_gram_mean: float
    cross_gram_max: float
    uncertainty: str
    reason: str
    seconds: float


@contextlib.contextmanager
def suppress_native_output(enabled: bool):
    """Silence Python and native-extension stdout/stderr during noisy fits."""
    if not enabled:
        yield
        return
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)


def circle_points(t: np.ndarray) -> np.ndarray:
    a = 2.0 * np.pi * np.asarray(t, dtype=float)
    return np.column_stack([np.cos(a), np.sin(a)])


def random_planes(d_ambient: int, k: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Return k mostly-incoherent 2D planes in R^d_ambient.

    This follows the planting pattern in Manifold-SAE's manifold_recovery.py:
    each atom is a unit circle decoded through a random orthonormal plane.
    """
    planes: list[np.ndarray] = []
    for _ in range(k):
        q, _ = np.linalg.qr(rng.standard_normal((d_ambient, 2)))
        planes.append(q[:, :2])
    return planes


def plant_superposed_circles(
    *,
    n: int,
    d_ambient: int,
    k_planted: int,
    noise: float,
    seed: int,
) -> dict[str, Any]:
    """Plant additive circles with random gates and amplitudes.

    For K_planted=2, most rows are co-active and a small set are single-atom
    disambiguators. The exact fit K is varied by the sweep; the data-generating
    K stays fixed so over/under-complete dictionary behavior is visible.
    """
    rng = np.random.default_rng(seed)
    planes = random_planes(d_ambient, k_planted, rng)
    coords = rng.uniform(0.0, 1.0, size=(k_planted, n))
    gate = np.zeros((n, k_planted), dtype=bool)
    amp = np.zeros((n, k_planted), dtype=float)

    idx = rng.permutation(n)
    n_single = max(k_planted * 8, n // 5)
    single = idx[: min(n_single, n)]
    rest = idx[len(single) :]

    for j, i in enumerate(single):
        gate[i, j % k_planted] = True

    for i in rest:
        if k_planted == 1:
            gate[i, 0] = True
        elif rng.uniform() < 0.80:
            gate[i, :] = True
        else:
            gate[i, rng.integers(0, k_planted)] = True

    amp[gate] = 0.6 + 0.8 * rng.uniform(size=int(gate.sum()))

    x = np.zeros((n, d_ambient), dtype=float)
    for k in range(k_planted):
        contribution = circle_points(coords[k]) @ planes[k].T
        x += (amp[:, k] * gate[:, k])[:, None] * contribution
    x += noise * rng.standard_normal(x.shape)
    x -= x.mean(axis=0, keepdims=True)

    return {"X": x, "planes": planes, "coords": coords, "gate": gate, "amp": amp}


def fit_cell(
    x: np.ndarray,
    *,
    topology: str,
    k_fit: int,
    incoherence_weight: float,
    d_atom: int,
    n_iter: int,
    seed: int,
) -> Any:
    return gamfit.sae_manifold_fit(
        X=x,
        K=k_fit,
        d_atom=d_atom,
        atom_topology=topology,
        assignment="ibp",
        ard_per_atom=False,
        alpha="auto",
        sparsity_weight=0.01,
        smoothness_weight=0.01,
        isometry_weight=0.1,
        learning_rate=1.0,
        n_iter=n_iter,
        random_state=seed,
        decoder_incoherence_weight=float(incoherence_weight),
    )


def normalized_decoder_cross_grams(model: Any) -> tuple[float, float]:
    """Mean and max normalized ||B_j B_k^T||_F over atom pairs.

    gamfit exposes decoder blocks as arrays shaped (basis_size, ambient_dim).
    The requested cross-Gram is computed directly on those blocks and normalized
    by ||B_j||_F ||B_k||_F, so values are comparable across topologies.
    """
    blocks = [np.asarray(b, dtype=float) for b in getattr(model, "decoder_blocks")]
    vals: list[float] = []
    for j, k in itertools.combinations(range(len(blocks)), 2):
        bj = blocks[j]
        bk = blocks[k]
        denom = float(np.linalg.norm(bj, "fro") * np.linalg.norm(bk, "fro"))
        if denom <= 1e-12:
            vals.append(float("nan"))
        else:
            vals.append(float(np.linalg.norm(bj @ bk.T, "fro") / denom))
    finite = [v for v in vals if np.isfinite(v)]
    if not finite:
        return float("nan"), float("nan")
    return float(np.mean(finite)), float(np.max(finite))


def shape_band_status(model: Any) -> str:
    if not hasattr(model, "shape_band"):
        return "no-method"
    parts: list[str] = []
    for k in range(len(getattr(model, "atoms", []))):
        try:
            band = model.shape_band(k)
            coords = np.asarray(band["coords"])
            mean = np.asarray(band["mean"])
            sd = np.asarray(band["sd"])
            ok = (
                coords.ndim == 2
                and mean.ndim == 2
                and sd.shape == mean.shape
                and coords.shape[0] == mean.shape[0]
                and np.all(np.isfinite(coords))
                and np.all(np.isfinite(mean))
                and np.all(np.isfinite(sd))
                and np.all(sd >= -1e-10)
            )
            avg_sd = float(np.mean(sd)) if sd.size else float("nan")
            parts.append(f"{k}:ok:{avg_sd:.2e}" if ok else f"{k}:bad-shape")
        except Exception as exc:  # noqa: BLE001 - report per-atom availability.
            parts.append(f"{k}:unavail:{type(exc).__name__}")
    return ",".join(parts) if parts else "no-atoms"


def compact_reason(exc: BaseException) -> str:
    msg = f"{type(exc).__name__}: {exc}"
    msg = " ".join(msg.split())
    return msg[:180]


def run_cell(
    x: np.ndarray,
    *,
    topology: str,
    k_fit: int,
    incoherence_weight: float,
    d_atom: int,
    n_iter: int,
    seed: int,
    suppress_fit_output: bool,
) -> CellResult:
    t0 = time.perf_counter()
    try:
        with suppress_native_output(suppress_fit_output):
            model = fit_cell(
                x,
                topology=topology,
                k_fit=k_fit,
                incoherence_weight=incoherence_weight,
                d_atom=d_atom,
                n_iter=n_iter,
                seed=seed,
            )
        r2 = float(getattr(model, "reconstruction_r2"))
        if not np.isfinite(r2):
            raise RuntimeError(f"non-finite reconstruction_r2={r2!r}")
        if r2 < MIN_CONVERGED_R2:
            status = "FAIL"
            reason = f"low R2<{MIN_CONVERGED_R2:g}"
        else:
            status = "OK"
            reason = ""
        if k_fit >= 2:
            cg_mean, cg_max = normalized_decoder_cross_grams(model)
        else:
            cg_mean, cg_max = float("nan"), float("nan")
        uncertainty = shape_band_status(model)
    except Exception as exc:  # noqa: BLE001 - each grid cell must survive.
        return CellResult(
            topology=topology,
            k_fit=k_fit,
            incoherence_weight=incoherence_weight,
            d_atom=d_atom,
            status="FAIL",
            r2=float("nan"),
            cross_gram_mean=float("nan"),
            cross_gram_max=float("nan"),
            uncertainty="not-run",
            reason=compact_reason(exc),
            seconds=time.perf_counter() - t0,
        )
    return CellResult(
        topology=topology,
        k_fit=k_fit,
        incoherence_weight=incoherence_weight,
        d_atom=d_atom,
        status=status,
        r2=r2,
        cross_gram_mean=cg_mean,
        cross_gram_max=cg_max,
        uncertainty=uncertainty,
        reason=reason,
        seconds=time.perf_counter() - t0,
    )


def fmt_float(value: float, width: int = 8, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "nan".rjust(width)
    return f"{value:{width}.{digits}f}"


def print_table(results: list[CellResult]) -> None:
    header = (
        f"{'topology':>9s} {'K':>2s} {'w':>3s} {'d':>1s} {'status':>6s} "
        f"{'R2':>8s} {'xGramMean':>10s} {'xGramMax':>9s} {'band':<32s} reason"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        reason = r.reason
        print(
            f"{r.topology:>9s} {r.k_fit:2d} {r.incoherence_weight:3.0f} "
            f"{r.d_atom:1d} {r.status:>6s} "
            f"{fmt_float(r.r2)} {fmt_float(r.cross_gram_mean, 10, 4)} "
            f"{fmt_float(r.cross_gram_max, 9, 4)} {r.uncertainty:<32s} {reason}"
        )


def incoherence_improvement_summary(results: list[CellResult]) -> tuple[int, int, list[str]]:
    by_key: dict[tuple[str, int, int], dict[float, CellResult]] = {}
    for r in results:
        if r.k_fit < 2:
            continue
        by_key.setdefault((r.topology, r.k_fit, r.d_atom), {})[r.incoherence_weight] = r

    improved = 0
    comparable = 0
    lines: list[str] = []
    for key in sorted(by_key):
        pair = by_key[key]
        off = pair.get(0.0)
        on = pair.get(1.0)
        if off is None or on is None:
            continue
        if off.status != "OK" or on.status != "OK":
            continue
        if not (np.isfinite(off.cross_gram_mean) and np.isfinite(on.cross_gram_mean)):
            continue
        comparable += 1
        ok = on.cross_gram_mean < off.cross_gram_mean
        improved += int(ok)
        topology, k_fit, d_atom = key
        delta = on.cross_gram_mean - off.cross_gram_mean
        lines.append(
            f"  {topology:>9s} K={k_fit} d={d_atom}: "
            f"OFF={off.cross_gram_mean:.4f} ON={on.cross_gram_mean:.4f} "
            f"delta={delta:+.4f} {'PASS' if ok else 'FAIL'}"
        )
    return improved, comparable, lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--n-iter", type=int, default=DEFAULT_N_ITER)
    parser.add_argument("--d-ambient", type=int, default=DEFAULT_D_AMBIENT)
    parser.add_argument("--k-planted", type=int, default=DEFAULT_K_PLANTED)
    parser.add_argument("--noise", type=float, default=DEFAULT_NOISE)
    parser.add_argument("--seed", type=int, default=671)
    parser.add_argument("--tracebacks", action="store_true")
    parser.add_argument("--fit-logs", action="store_true", help="show native gamfit solver logs")
    args = parser.parse_args()

    print("SAE manifold experiment sweep")
    print(f"gamfit={getattr(gamfit, '__version__', 'unknown')}")
    print(
        "data="
        f"superposed circles on random planes, n={args.n}, "
        f"D={args.d_ambient}, planted_K={args.k_planted}, noise={args.noise}, "
        f"seed={args.seed}"
    )
    print(
        "grid="
        f"topologies={TOPOLOGIES}, K={K_VALUES}, "
        f"decoder_incoherence_weight={INCOHERENCE_WEIGHTS}, d_atom={D_ATOMS}, "
        f"n_iter={args.n_iter}"
    )
    print()

    planted = plant_superposed_circles(
        n=args.n,
        d_ambient=args.d_ambient,
        k_planted=args.k_planted,
        noise=args.noise,
        seed=args.seed,
    )
    x = np.asarray(planted["X"], dtype=float)

    results: list[CellResult] = []
    t_start = time.perf_counter()
    total = len(TOPOLOGIES) * len(K_VALUES) * len(INCOHERENCE_WEIGHTS) * len(D_ATOMS)
    idx = 0
    for topology, k_fit, w, d_atom in itertools.product(
        TOPOLOGIES, K_VALUES, INCOHERENCE_WEIGHTS, D_ATOMS
    ):
        idx += 1
        print(f"[{idx:02d}/{total}] topology={topology} K={k_fit} w={w:g} d={d_atom}", flush=True)
        try:
            result = run_cell(
                x,
                topology=topology,
                k_fit=k_fit,
                incoherence_weight=w,
                d_atom=d_atom,
                n_iter=args.n_iter,
                seed=args.seed + 1000 * idx,
                suppress_fit_output=not args.fit_logs,
            )
        except Exception as exc:  # Defensive guard around the guard.
            if args.tracebacks:
                traceback.print_exc()
            result = CellResult(
                topology=topology,
                k_fit=k_fit,
                incoherence_weight=w,
                d_atom=d_atom,
                status="FAIL",
                r2=float("nan"),
                cross_gram_mean=float("nan"),
                cross_gram_max=float("nan"),
                uncertainty="not-run",
                reason=compact_reason(exc),
                seconds=0.0,
            )
        results.append(result)

    print()
    print_table(results)

    converged = sum(1 for r in results if r.status == "OK")
    failed = len(results) - converged
    improved, comparable, improvement_lines = incoherence_improvement_summary(results)

    print()
    print("Incoherence ON/OFF cross-Gram comparisons")
    if improvement_lines:
        for line in improvement_lines:
            print(line)
    else:
        print("  no K>=2 ON/OFF pairs had both fits converged with finite cross-Grams")

    all_converged = converged == len(results)
    incoherence_pass = comparable > 0 and improved == comparable
    elapsed = time.perf_counter() - t_start

    print()
    print("FINAL SUMMARY")
    print(f"  cells converged: {converged}/{len(results)}; failed: {failed}")
    print(f"  incoherence improved cross-Gram: {improved}/{comparable} comparable K>=2 pairs")
    print(f"  elapsed: {elapsed:.1f}s")
    print(f"  convergence verdict: {'PASS' if all_converged else 'FAIL'}")
    print(f"  incoherence verdict: {'PASS' if incoherence_pass else 'FAIL'}")
    print(f"  OVERALL: {'PASS' if all_converged and incoherence_pass else 'FAIL'}")
    return 0 if all_converged and incoherence_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
