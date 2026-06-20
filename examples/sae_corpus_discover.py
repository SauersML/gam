#!/usr/bin/env python
"""#1026 corpus discovery — locate a full-width (d_model) real residual-stream
activation cache for the EV-vs-K ladder, or report exactly what is missing.

The #1026 acceptance ladder K in {8,32,128,512} vs the official Qwen-32K
reference needs REAL LLM activations at the model's native residual width
(OLMo-3-32B layer-25 is 5120-dim), NOT the PCA-64 fixtures committed under
tests/data/ (those are linear projections for the in-tree stress regression).

This is a thin numeric LOCATOR (the #977 boundary: activations are a response
matrix). It does NOT fit anything and does NOT download models. It scans a list
of candidate paths for a banked `activations.npy` / `.pt` cache, reports the
native width of the layer slice the EV example would feed, and decides whether
that width clears a `--min-d-model` bar (default 2048: well above the PCA-64
fixtures, low enough to admit a real small-model residual stream as a stand-in).

Exit 0 + a `READY <path> <layer-arg>` line on stdout when a usable cache is
found, so an sbatch can `eval` the discovered `--npy/--pt`+`--olmo-layer` flags
straight into `sae_ev_vs_k_olmo.py`. Exit 3 + a `MISSING` line (with the
harvest command to generate one) when nothing qualifies.

USAGE (on an MSI compute node):
  python examples/sae_corpus_discover.py \
      --candidate $PROJECTS/olmo_data/base/activations.npy:25 \
      --candidate $PROJECTS/olmo_data/instruct \
      --min-d-model 2048
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _width_of(path: Path, layer: int | None) -> tuple[int, int, str] | None:
    """Return (n_rows, d_model, layer_arg_str) for the slice the EV example
    would feed, reading only the npy header (mmap) so a 5120-dim corpus is not
    pulled into RAM. None if the file is unreadable or not 2D/3D."""
    try:
        if path.suffix == ".npy":
            arr = np.load(path, mmap_mode="r")
        elif path.suffix in (".pt", ".pth"):
            import torch

            blob = torch.load(path, map_location="cpu")
            arr = blob["X"] if isinstance(blob, dict) else blob
            arr = np.asarray(arr)
        else:
            return None
    except Exception as exc:  # noqa: BLE001 — locator must not crash a sweep
        print(f"[discover] unreadable {path}: {exc}", file=sys.stderr)
        return None

    if arr.ndim == 2:
        return int(arr.shape[0]), int(arr.shape[1]), ""
    if arr.ndim == 3:
        # (prompts, layers, d_model) — width is the last axis, independent of
        # which layer is sliced; report the layer-arg the EV example needs.
        lyr = 25 if layer is None else layer
        if not (0 <= lyr < arr.shape[1]):
            print(f"[discover] {path}: layer {lyr} out of range (n_layers={arr.shape[1]})", file=sys.stderr)
            return None
        return int(arr.shape[0]), int(arr.shape[2]), f"--olmo-layer {lyr}"
    return None


def _expand(candidate: str) -> list[tuple[Path, int | None]]:
    """A candidate is `PATH` or `PATH:LAYER`. A directory is scanned for any
    `activations.npy` / cache one level deep (the olmo_data/<rev>/ layout)."""
    raw, _, layer_s = candidate.partition(":")
    layer = int(layer_s) if layer_s else None
    p = Path(raw)
    if p.is_dir():
        hits: list[tuple[Path, int | None]] = []
        for name in ("activations.npy",):
            hits.extend((q, layer) for q in p.glob(f"**/{name}"))
        return sorted(set(hits))
    return [(p, layer)] if p.exists() else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="PATH or PATH:LAYER to probe; a directory is globbed for activations.npy",
    )
    ap.add_argument(
        "--min-d-model",
        type=int,
        default=2048,
        help="native residual width bar a cache must clear to be a real-data stand-in",
    )
    ap.add_argument("--harvest-model", default="allenai/OLMo-2-1124-7B")
    ap.add_argument("--harvest-layer", type=int, default=18)
    args = ap.parse_args()

    best: tuple[int, Path, int, str] | None = None  # (d_model, path, n, layer_arg)
    for cand in args.candidate:
        for path, layer in _expand(cand):
            info = _width_of(path, layer)
            if info is None:
                continue
            n, d, layer_arg = info
            qual = "OK" if d >= args.min_d_model else "below-bar"
            print(f"[discover] {path}  n={n} d_model={d} [{qual}] {layer_arg}".rstrip(), file=sys.stderr)
            if d >= args.min_d_model and (best is None or d > best[0]):
                best = (d, path, n, layer_arg)

    if best is not None:
        d, path, n, layer_arg = best
        # Machine-parseable: an sbatch can `eval` the flags after READY.
        print(f"READY --npy {path} {layer_arg}".rstrip())
        print(f"[discover] selected {path} (n={n}, d_model={d}) for the EV-vs-K ladder", file=sys.stderr)
        return 0

    print("MISSING")
    print(
        "[discover] no full-width activation cache found. Generate one with:\n"
        f"  python examples/harvest_residual_activations.py --model {args.harvest_model} \\\n"
        f"      --dataset wikitext --config wikitext-103-raw-v1 --layer {args.harvest_layer} \\\n"
        "      --n-tokens 4000 --out resid_cache.pt\n"
        "  then: python examples/sae_ev_vs_k_olmo.py --pt resid_cache.pt --pcs 32 --seed 42",
        file=sys.stderr,
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
