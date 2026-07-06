#!/usr/bin/env python3
"""Standing null battery plus circle spike-in calibration for activation audits.

All input paths are CLI arguments. For a full battery run, pass real observed
activations and architecture-matched random-weight activations. For the Qwen
spike-in calibration alone, use ``--spike-only``.
"""

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np


Array = np.ndarray
StatFn = Callable[[Array], float]


def load_matrix(path: Path, *, layer: Optional[int], max_rows: int, max_cols: int, seed: int) -> Array:
    arr = np.load(path, mmap_mode="r")
    if arr.ndim == 3:
        if layer is None:
            raise SystemExit(f"{path}: 3-D activation arrays require --layer")
        arr = arr[:, layer, :]
    if arr.ndim != 2:
        raise SystemExit(f"{path}: expected a 2-D activation matrix, got shape {arr.shape}")
    n, p = int(arr.shape[0]), int(arr.shape[1])
    if n == 0 or p == 0:
        raise SystemExit(f"{path}: activation matrix must be non-empty")
    rng = np.random.default_rng(seed)
    row_take = min(max_rows, n)
    col_take = min(max_cols, p)
    rows = np.sort(rng.choice(n, size=row_take, replace=False)) if row_take < n else slice(None)
    cols = np.sort(rng.choice(p, size=col_take, replace=False)) if col_take < p else slice(None)
    x = np.asarray(arr[rows, :], dtype=np.float64)
    x = np.asarray(x[:, cols], dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise SystemExit(f"{path}: sampled activation block contains non-finite values")
    return x


def peel_leading_sink(x: Array, *, iters: int = 8) -> Array:
    centered = x - x.mean(axis=0, keepdims=True)
    v = centered[0].copy()
    norm = np.linalg.norm(v)
    if norm == 0.0:
        v = np.ones(centered.shape[1], dtype=np.float64)
        norm = np.linalg.norm(v)
    v /= norm
    for _idx in range(iters):
        u = centered @ v
        v = centered.T @ u
        norm = np.linalg.norm(v)
        if norm == 0.0:
            return centered
        v /= norm
    score = centered @ v
    return centered - np.outer(score, v)


def phase_randomized(x: Array, rng: np.random.Generator) -> Array:
    coeff = np.fft.rfft(x, axis=0)
    if coeff.shape[0] > 2:
        phases = rng.uniform(0.0, 2.0 * np.pi, size=(coeff.shape[0] - 2, x.shape[1]))
        coeff[1:-1] = np.abs(coeff[1:-1]) * np.exp(1j * phases)
    return np.fft.irfft(coeff, n=x.shape[0], axis=0)


def random_rotation(x: Array, rng: np.random.Generator) -> Array:
    q, r = np.linalg.qr(rng.normal(size=(x.shape[1], x.shape[1])))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return x @ (q * signs)


def token_shuffle(x: Array, rng: np.random.Generator) -> Array:
    order = rng.permutation(x.shape[0])
    return x[order].copy()


def architecture_random_weight(observed: Array, random_weight: Array, rng: np.random.Generator) -> Array:
    if random_weight.shape[1] != observed.shape[1]:
        raise SystemExit(
            f"random-weight ncols {random_weight.shape[1]} != observed ncols {observed.shape[1]}"
        )
    rows = rng.integers(0, random_weight.shape[0], size=observed.shape[0])
    rw = random_weight[rows].copy()
    obs_mean = observed.mean(axis=0, keepdims=True)
    obs_sd = observed.std(axis=0, keepdims=True)
    rw_mean = rw.mean(axis=0, keepdims=True)
    rw_sd = rw.std(axis=0, keepdims=True)
    rw_sd[rw_sd == 0.0] = 1.0
    return obs_mean + (rw - rw_mean) * (obs_sd / rw_sd)


def ordered_circle_stat(x: Array) -> float:
    if x.shape[1] < 2 or x.shape[0] < 4:
        raise SystemExit("ordered-circle statistic requires at least 4 rows and 2 columns")
    z = x[:, :2] - x[:, :2].mean(axis=0, keepdims=True)
    radius = np.linalg.norm(z, axis=1)
    radius_mean = float(radius.mean())
    radius_cv = float(radius.std() / max(abs(radius_mean), np.finfo(np.float64).tiny))
    step = float(np.linalg.norm(np.roll(z, -1, axis=0) - z, axis=1).sum())
    norm = float(radius.sum())
    return norm / max(step, np.finfo(np.float64).tiny) / (1.0 + radius_cv)


def top_two_energy_fraction(x: Array) -> float:
    z = x - x.mean(axis=0, keepdims=True)
    total = float(np.sum(z * z))
    if total == 0.0:
        return 0.0
    cov = z.T @ z
    vals = np.linalg.eigvalsh(cov)
    return float(np.sum(vals[-2:]) / total)


JsonScalar = Union[float, int, List[float]]


def summarize(observed: float, samples: List[float]) -> Dict[str, JsonScalar]:
    xs = np.sort(np.asarray(samples, dtype=np.float64))
    mean = float(xs.mean())
    sd = float(xs.std(ddof=1)) if xs.size > 1 else 0.0
    p_value = float((1 + np.count_nonzero(xs >= observed)) / (xs.size + 1))
    z = float((observed - mean) / sd) if sd > 0.0 else 0.0
    return {
        "observed": float(observed),
        "n": int(xs.size),
        "mean": mean,
        "sd": sd,
        "min": float(xs[0]),
        "q25": float(np.quantile(xs, 0.25)),
        "median": float(np.quantile(xs, 0.50)),
        "q75": float(np.quantile(xs, 0.75)),
        "max": float(xs[-1]),
        "z": z,
        "p_value": p_value,
        "samples": [float(v) for v in xs],
    }


def run_battery(
    observed: Array,
    *,
    random_weight: Optional[Array],
    reps: int,
    seed: int,
    stat: StatFn,
) -> Dict[str, object]:
    if random_weight is None:
        raise SystemExit("full null battery requires --random-weight-npy; use --spike-only for spike-in only")
    rng = np.random.default_rng(seed)
    observed_stat = float(stat(observed))
    nulls: Dict[str, Dict[str, JsonScalar]] = {}
    generators: Dict[str, Callable[[], Array]] = {
        "phase_randomized": lambda: phase_randomized(observed, rng),
        "random_rotation": lambda: random_rotation(observed, rng),
        "token_shuffle": lambda: token_shuffle(observed, rng),
        "architecture_matched_random_weight": lambda: architecture_random_weight(observed, random_weight, rng),
    }
    for name, gen in generators.items():
        samples = [float(stat(gen())) for _rep in range(reps)]
        nulls[name] = summarize(observed_stat, samples)
    return {"observed": observed_stat, "nulls": nulls}


def inject_circle(noise: Array, snr: float, rng: np.random.Generator) -> Array:
    q, r = np.linalg.qr(rng.normal(size=(noise.shape[1], noise.shape[1])))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    basis = q * signs
    theta0 = float(rng.uniform(0.0, 2.0 * np.pi))
    theta = theta0 + 2.0 * np.pi * np.arange(noise.shape[0], dtype=np.float64) / noise.shape[0]
    amp = snr * float(np.sqrt(np.mean(noise * noise))) * np.sqrt(noise.shape[1])
    signal = amp * (np.outer(np.cos(theta), basis[:, 0]) + np.outer(np.sin(theta), basis[:, 1]))
    return noise + signal


def spike_in_curve(
    noise: Array,
    *,
    snrs: List[float],
    trials: int,
    seed: int,
) -> List[Dict[str, Union[float, int]]]:
    rng = np.random.default_rng(seed)
    null_stats = [top_two_energy_fraction(phase_randomized(noise, rng)) for _rep in range(trials)]
    threshold = float(np.quantile(np.asarray(null_stats), 0.95))
    rows: List[Dict[str, Union[float, int]]] = []
    for snr in snrs:
        stats = [top_two_energy_fraction(inject_circle(noise, snr, rng)) for _rep in range(trials)]
        hits = sum(v > threshold for v in stats)
        rows.append(
            {
                "snr": float(snr),
                "trials": int(trials),
                "power": float(hits / trials),
                "mean_stat": float(np.mean(stats)),
                "threshold": threshold,
            }
        )
    return rows


def parse_snrs(raw: str) -> List[float]:
    snrs = [float(token) for token in raw.split(",") if token.strip()]
    if not snrs:
        raise SystemExit("--snrs must contain at least one value")
    if any((not np.isfinite(v)) or v < 0.0 for v in snrs):
        raise SystemExit("--snrs values must be finite and non-negative")
    return snrs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activations", required=True, type=Path)
    ap.add_argument("--random-weight-npy", type=Path)
    ap.add_argument("--layer", type=int)
    ap.add_argument("--random-weight-layer", type=int)
    ap.add_argument("--max-rows", type=int, default=4096)
    ap.add_argument("--max-cols", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260706)
    ap.add_argument("--null-reps", type=int, default=32)
    ap.add_argument("--spike-trials", type=int, default=32)
    ap.add_argument("--snrs", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--claimed-snr", type=float, default=0.5)
    ap.add_argument("--stat", choices=("top2", "ordered-circle"), default="top2")
    ap.add_argument("--peel-sink", action="store_true")
    ap.add_argument("--spike-only", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if args.max_rows <= 0 or args.max_cols <= 1:
        raise SystemExit("--max-rows must be positive and --max-cols must be at least 2")
    if args.null_reps <= 0 or args.spike_trials <= 0:
        raise SystemExit("--null-reps and --spike-trials must be positive")

    observed = load_matrix(
        args.activations,
        layer=args.layer,
        max_rows=args.max_rows,
        max_cols=args.max_cols,
        seed=args.seed,
    )
    if args.peel_sink:
        observed = peel_leading_sink(observed)

    random_weight = None
    if args.random_weight_npy is not None:
        random_weight = load_matrix(
            args.random_weight_npy,
            layer=args.random_weight_layer,
            max_rows=args.max_rows,
            max_cols=observed.shape[1],
            seed=args.seed ^ 0x5EED,
        )
        if args.peel_sink:
            random_weight = peel_leading_sink(random_weight)

    stat = top_two_energy_fraction if args.stat == "top2" else ordered_circle_stat
    battery = None
    if not args.spike_only:
        battery = run_battery(
            observed,
            random_weight=random_weight,
            reps=args.null_reps,
            seed=args.seed ^ 0xB477,
            stat=stat,
        )

    spike = spike_in_curve(
        observed,
        snrs=parse_snrs(args.snrs),
        trials=args.spike_trials,
        seed=args.seed ^ 0x51C1E,
    )
    nearest = min(spike, key=lambda row: abs(float(row["snr"]) - args.claimed_snr))
    report = {
        "source": str(args.activations),
        "shape_used": [int(observed.shape[0]), int(observed.shape[1])],
        "sink_peeled": bool(args.peel_sink),
        "statistic": args.stat,
        "battery": battery,
        "spike_in": spike,
        "claimed_snr": float(args.claimed_snr),
        "claimed_snr_power": float(nearest["power"]),
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
