"""Measure the transport-groupoid phase-shift law on REAL 2-layer OLMo activations.

Thesis under test ("Binding is transport", crates/gam-sae/src/manifold/mod.rs):
a curved (circle) atom shared across two layers transports LINEARLY as a phase
shift ``t ↦ ±t + φ``; a significant nonlinear (extra-harmonic) component in the
empirical transport map refutes the linear law for that atom.

Pipeline
--------
1. Load OLMo last-token residual activations for two ADJACENT layers L and L+1
   (same prompts/rows in both layers — the paired input a crosscoder needs).
2. Fit a 2-layer manifold crosscoder with ONE shared latent chart `t` and two
   honest per-layer decoders `B^(L)`, `B^(L+1)` through the unified outer-REML
   engine (``gamfit.sae_crosscoder_fit``).
3. For each fitted circle atom, measure the empirical anchor→block transport map
   and its phase-shift-law verdict (``measure_atom_transport`` in
   ``crates/gam-sae/src/manifold/transport_law.rs``): phase_r2 vs smooth_r2, the
   honest-units drift `δ_k`, and the principal angles between the two layer
   images.

Data (see the OLMo-3-32B activation dataset; 64 layers, hidden D=5120)
---------------------------------------------------------------------
``activations.npy`` is ``(635, 64, 5120)`` f32 — last-token residual per prompt
for every layer. Slice ``[:, L, :]`` and ``[:, L+1, :]`` for the two-layer input.
On MSI the pulled data lives under
``/projects/standard/hsiehph/sauer354/olmo_data/<...>/<rev>/activations.npy``.

Launch (MSI)
------------
    python experiments/transport_law/run_olmo_transport.py \
        --activations /projects/standard/hsiehph/sauer354/olmo_data/<rev>/activations.npy \
        --layer 25 --rows 635 --n-atoms 32 --grid-resolution 512 \
        --out $PWD/transport_law_L25

The public call is a thin array marshaller. Target stacking, shared-chart seed,
block-relevance coordinates, outer REML, honest-unit decoder splitting, drift,
and transport-law measurement all run in Rust.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def _load_two_layers(path: str, layer: int, rows: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Load adjacent-layer last-token residual activations ``(n, D)`` each.

    ``activations.npy`` is ``(n_prompts, n_layers, D)``; we slice layers ``L`` and
    ``L+1`` for the same prompts so the crosscoder sees paired rows.
    """
    acts = np.load(path, mmap_mode="r")
    if acts.ndim != 3:
        raise ValueError(
            f"expected activations of shape (n_prompts, n_layers, D); got {acts.shape}"
        )
    n_prompts, n_layers, _d = acts.shape
    if not (0 <= layer < n_layers - 1):
        raise ValueError(
            f"--layer {layer} must be in [0, {n_layers - 2}] to have an L+1 neighbour"
        )
    take = n_prompts if rows is None else min(rows, n_prompts)
    anchor = np.ascontiguousarray(acts[:take, layer, :], dtype=np.float64)
    block = np.ascontiguousarray(acts[:take, layer + 1, :], dtype=np.float64)
    return anchor, block


def _center(x: np.ndarray) -> np.ndarray:
    """Column-center (the crosscoder decoders carry no intercept for the mean)."""
    return x - x.mean(axis=0, keepdims=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activations", required=True, help="path to activations.npy (n_prompts, n_layers, D)")
    ap.add_argument("--layer", type=int, default=25, help="source layer L (block target is L+1)")
    ap.add_argument("--rows", type=int, default=None, help="max prompts/rows to use (default: all)")
    ap.add_argument("--n-atoms", type=int, default=32, help="number of shared circle atoms K")
    ap.add_argument("--n-harmonics", type=int, default=3, help="Fourier order per circle atom")
    ap.add_argument("--grid-resolution", type=int, default=512, help="chart grid points over [0,1)")
    ap.add_argument("--inner-max-iter", type=int, default=80, help="inner arrow-Schur iterations")
    ap.add_argument("--law-gap-tolerance", type=float, default=0.05,
                    help="maximum smooth-minus-phase circular R2 gap for the phase law verdict")
    ap.add_argument("--out", required=True, help="output directory for the JSON report")
    args = ap.parse_args(argv)

    anchor, block = _load_two_layers(args.activations, args.layer, args.rows)
    anchor = _center(anchor)
    block = _center(block)
    print(
        f"loaded OLMo L{args.layer}/L{args.layer + 1}: anchor {anchor.shape} block {block.shape}",
        flush=True,
    )

    from gamfit import sae_crosscoder_fit  # noqa: E402  (import after data load)

    controls = {
        "max_iter": args.inner_max_iter,
        "ridge_ext_coord": 1e-6,
        "ridge_beta": 1e-6,
        "n_harmonics": args.n_harmonics,
        "grid_resolution": args.grid_resolution,
        "law_gap_tolerance": args.law_gap_tolerance,
    }
    fit = sae_crosscoder_fit(
        anchor,
        [(f"L{args.layer + 1}", block)],
        anchor_label=f"L{args.layer}",
        n_atoms=args.n_atoms,
        n_harmonics=args.n_harmonics,
        max_iter=args.inner_max_iter,
        ridge_ext_coord=1e-6,
        ridge_beta=1e-6,
        transport_grid_resolution=args.grid_resolution,
        law_gap_tolerance=args.law_gap_tolerance,
    )
    reports = list(fit["transport"])

    import os

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"transport_law_L{args.layer}.json")
    with open(out_path, "w") as fh:
        json.dump(
            {
                "layer": args.layer,
                "controls": controls,
                "layout": fit["layout"],
                "drift": fit["drift"],
                "termination": fit["termination"],
                "atoms": reports,
            },
            fh,
            indent=2,
        )

    # Summarize the law verdict across atoms.
    held = sum(1 for r in reports if r.get("law_holds"))
    gaps = [float(r["law_gap"]) for r in reports if np.isfinite(r.get("law_gap", np.nan))]
    print(f"wrote {out_path}", flush=True)
    print(
        f"phase-shift LAW holds for {held}/{len(reports)} atoms; "
        f"median smooth−phase gap = {np.median(gaps) if gaps else float('nan'):.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
