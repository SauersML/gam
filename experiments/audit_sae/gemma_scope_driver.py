#!/usr/bin/env python3
"""One-command Gemma Scope 2 JumpReLU SAE audit driver.

Gemma Scope JumpReLU checkpoints store an encoder ``W_enc, b_enc, threshold``
and a decoder ``W_dec, b_dec``. This driver treats ``W_dec`` as the frozen GAM
decoder with shape ``K x P``: rows are dictionary atoms and columns are residual
activation dimensions. Dense external codes are computed as
``pre * (pre > threshold)`` and passed directly to ``gamfit.audit_sae``.

The required null-battery donor is an architecture-matched random-weight
encoder: it keeps the real encoder bias and thresholds, replaces ``W_enc`` by
seeded Gaussian columns scaled to the real per-column L2 norms, and supplies
those JumpReLU codes as ``random_weight_codes`` for topology/atlas claims.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import gamfit


SAE_KEYS = ("W_enc", "b_enc", "W_dec", "b_dec", "threshold")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Gemma Scope 2 JumpReLU SAE codes with gamfit.audit_sae.",
    )
    parser.add_argument("--sae", required=True, type=Path, help="Gemma Scope .safetensors or .npz")
    parser.add_argument("--activations", required=True, type=Path, help="Residual activations .npy")
    parser.add_argument("--out", required=True, type=Path, help="Output report.json path")
    parser.add_argument("--subsample", type=int, default=None, help="Rows to sample without replacement")
    parser.add_argument("--seed", type=int, default=7, help="PCG64 seed for subsampling and null weights")
    parser.add_argument("--block-size", type=int, default=1, help="SAE audit block size")
    parser.add_argument("--active", type=int, default=None, help="Active atoms for the dual certificate view")
    return parser.parse_args()


def load_sae(path: Path) -> dict[str, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        from safetensors.numpy import load_file

        tensors = load_file(str(path))
    elif suffix == ".npz":
        with np.load(path) as archive:
            tensors = {key: archive[key] for key in archive.files}
    else:
        raise SystemExit(f"unsupported SAE format {suffix!r}; expected .safetensors or .npz")

    missing = [key for key in SAE_KEYS if key not in tensors]
    if missing:
        raise SystemExit(f"{path} is missing Gemma Scope tensor keys: {', '.join(missing)}")
    return {key: np.asarray(tensors[key]) for key in SAE_KEYS}


def float_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise SystemExit(f"{name} must be float32 or float64, got {array.dtype}")
    if not np.all(np.isfinite(array)):
        raise SystemExit(f"{name} contains non-finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def matrix(value: np.ndarray, name: str) -> np.ndarray:
    array = float_array(value, name)
    if array.ndim != 2:
        raise SystemExit(f"{name} must be a 2-D matrix, got shape {array.shape}")
    if 0 in array.shape:
        raise SystemExit(f"{name} must be non-empty, got shape {array.shape}")
    return array


def vector(value: np.ndarray, name: str) -> np.ndarray:
    array = float_array(value, name)
    if array.ndim != 1:
        raise SystemExit(f"{name} must be a 1-D vector, got shape {array.shape}")
    if array.shape[0] == 0:
        raise SystemExit(f"{name} must be non-empty")
    return array


def load_activations(path: Path) -> np.ndarray:
    activations = np.load(path)
    if activations.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise SystemExit(f"activations must be float32 or float64, got {activations.dtype}")
    if activations.ndim != 2:
        raise SystemExit(f"activations must be a 2-D [N, P] matrix, got shape {activations.shape}")
    if 0 in activations.shape:
        raise SystemExit(f"activations must be non-empty, got shape {activations.shape}")
    if not np.all(np.isfinite(activations)):
        raise SystemExit("activations contain non-finite values")
    return np.ascontiguousarray(activations)


def validate_sae(
    tensors: dict[str, np.ndarray],
    activations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    w_enc = matrix(tensors["W_enc"], "W_enc")
    b_enc = vector(tensors["b_enc"], "b_enc")
    w_dec = matrix(tensors["W_dec"], "W_dec")
    b_dec = vector(tensors["b_dec"], "b_dec")
    threshold = vector(tensors["threshold"], "threshold")

    p_enc, k_enc = w_enc.shape
    k_dec, p_dec = w_dec.shape
    n_rows, p_acts = activations.shape
    if k_enc != k_dec:
        raise SystemExit(f"W_enc has K={k_enc} columns but W_dec has K={k_dec} rows")
    if p_enc != p_dec:
        raise SystemExit(f"W_enc has P={p_enc} rows but W_dec has P={p_dec} columns")
    if p_acts != p_dec:
        raise SystemExit(f"activations have P={p_acts} columns but W_dec has P={p_dec}")
    if b_enc.shape != (k_enc,):
        raise SystemExit(f"b_enc must have shape ({k_enc},), got {b_enc.shape}")
    if b_dec.shape != (p_dec,):
        raise SystemExit(f"b_dec must have shape ({p_dec},), got {b_dec.shape}")
    if threshold.shape != (k_enc,):
        raise SystemExit(f"threshold must have shape ({k_enc},), got {threshold.shape}")
    if n_rows == 0:
        raise SystemExit("activations must contain at least one row")
    return w_enc, b_enc, w_dec, b_dec, threshold


def subsample_rows(
    activations: np.ndarray,
    subsample: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    if subsample is None:
        return activations
    if subsample <= 0:
        raise SystemExit("--subsample must be positive")
    if subsample > activations.shape[0]:
        raise SystemExit(
            f"--subsample={subsample} exceeds activation row count {activations.shape[0]}"
        )
    rows = rng.choice(activations.shape[0], size=subsample, replace=False)
    return np.ascontiguousarray(activations[rows])


def jumprelu_codes(
    activations: np.ndarray,
    w_enc: np.ndarray,
    b_enc: np.ndarray,
    threshold: np.ndarray,
) -> tuple[np.ndarray, float]:
    pre = activations.astype(np.float64, copy=False) @ w_enc + b_enc
    active = pre > threshold
    codes = pre * active
    firing_fraction = float(np.count_nonzero(active) / active.size)
    return np.ascontiguousarray(codes, dtype=np.float64), firing_fraction


def random_weight_encoder(w_enc: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    random_w = rng.normal(size=w_enc.shape)
    target_norms = np.linalg.norm(w_enc, axis=0)
    random_norms = np.linalg.norm(random_w, axis=0)
    if np.any(random_norms == 0.0):
        raise RuntimeError("PCG64 Gaussian draw produced a zero encoder column")
    random_w *= target_norms / random_norms
    return np.ascontiguousarray(random_w, dtype=np.float64)


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def nested(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def fmt(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value)


def print_summary(
    report: dict[str, Any],
    n_rows: int,
    p_dim: int,
    k_atoms: int,
    firing_fraction: float,
) -> None:
    routability_floor = nested(report, "routability", "floor", "floor")
    fraction_below_floor = nested(report, "routability", "fraction_below_floor")
    dark_matter_fraction = nested(report, "routability", "dark_matter_fraction")
    frac_certified = nested(report, "dual_certificate", "frac_certified")
    birth_candidates = nested(report, "dual_certificate", "birth_candidates")
    absorption_pairs = nested(report, "absorption", "pairs")
    absorption_pair_count = len(absorption_pairs) if isinstance(absorption_pairs, list) else 0
    dual_verdict = None
    if frac_certified is not None:
        dual_verdict = "all_certified" if float(frac_certified) >= 1.0 else "not_all_certified"

    print("Gemma Scope JumpReLU SAE audit")
    print(f"n: {n_rows}")
    print(f"P: {p_dim}")
    print(f"K: {k_atoms}")
    print(f"firing_fraction: {fmt(firing_fraction)}")
    print(f"routability_floor: {fmt(routability_floor)}")
    print(f"fraction_below_floor: {fmt(fraction_below_floor)}")
    print(f"dark_matter_fraction: {fmt(dark_matter_fraction)}")
    print(f"dual_certificate: {dual_verdict} frac_certified={fmt(frac_certified)}")
    print(f"birth_candidate_count: {len(birth_candidates) if isinstance(birth_candidates, list) else 0}")
    print(f"absorption_pair_count: {absorption_pair_count}")


def main() -> None:
    args = parse_args()
    if args.block_size < 1:
        raise SystemExit("--block-size must be >= 1")
    if args.active is not None and args.active < 1:
        raise SystemExit("--active must be >= 1")

    rng = np.random.Generator(np.random.PCG64(args.seed))
    activations = load_activations(args.activations)
    tensors = load_sae(args.sae)
    w_enc, b_enc, w_dec, _b_dec, threshold = validate_sae(tensors, activations)
    activations = subsample_rows(activations, args.subsample, rng)
    codes, firing_fraction = jumprelu_codes(activations, w_enc, b_enc, threshold)
    random_w_enc = random_weight_encoder(w_enc, rng)
    random_weight_codes, _ = jumprelu_codes(activations, random_w_enc, b_enc, threshold)

    report = gamfit.audit_sae(
        w_dec,
        activations,
        codes=codes,
        random_weight_codes=random_weight_codes,
        block_size=args.block_size,
        active=args.active,
    )
    report["driver"] = {
        "name": "experiments/audit_sae/gemma_scope_driver.py",
        "sae": str(args.sae),
        "activations": str(args.activations),
        "subsample": args.subsample,
        "seed": args.seed,
        "n": int(activations.shape[0]),
        "P": int(activations.shape[1]),
        "K": int(w_dec.shape[0]),
        "firing_fraction": firing_fraction,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(report), handle, indent=2, sort_keys=True)
        handle.write("\n")

    print_summary(report, activations.shape[0], activations.shape[1], w_dec.shape[0], firing_fraction)


if __name__ == "__main__":
    main()
