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
   honest per-layer decoders `B^(L)`, `B^(L+1)` (the landed M1 REML driver
   ``SaeManifoldTerm::run_multiblock_reml_fit`` + ``CrosscoderLayout``).
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

STATUS — pending pyffi exposure
-------------------------------
The manifold-crosscoder REML fit (``run_multiblock_reml_fit``) and the transport
measurement (``measure_atom_transport``) are currently RUST-ONLY; they are not yet
exposed through ``gam-pyffi``. The Rust math is landed and covered by
``crates/gam-sae/src/manifold/tests_transport_law.rs`` (planted phase-shift and
planted nonlinear arms). This driver is the thin marshaling wrapper for the real
measurement and will run end-to-end once the two FFI entry points below are added
to ``crates/gam-pyffi`` — it deliberately does NOT reimplement the fit or the
measurement in Python (Python is a thin wrapper; the math lives in Rust). Until
then it fails with an actionable message rather than hacking a Python fit.

Required FFI (proposed single fused entry point)
------------------------------------------------
    rust_module.sae_crosscoder_measure_transport(
        anchor: np.ndarray[n, p_L],          # layer-L activations
        block:  np.ndarray[n, p_{L+1}],      # layer-(L+1) activations
        n_atoms: int,
        grid_resolution: int,
        controls_json: str,                  # TwoBlockRemlControls
    ) -> str  # JSON: per-atom AtomTransportReport list
        # [{atom, phase_shift:[s,phi], phase_r2, smooth_r2, law_gap,
        #   law_holds, drift, principal_angles:[...], n_harmonics}, ...]

which internally: builds circle atoms at the augmented width p_L + p_{L+1},
seeds the shared chart (PCA / decoder-projection), calls
``SaeManifoldTerm::run_multiblock_reml_fit`` with one ``OutputBlock`` for
layer L+1, then ``measure_atom_transport(&term, layout, k, grid_resolution)``
for each atom k.
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


def _require_ffi(rust_module):
    fn = getattr(rust_module, "sae_crosscoder_measure_transport", None)
    if fn is None:
        raise NotImplementedError(
            "sae_crosscoder_measure_transport is not exposed by gam-pyffi yet.\n"
            "The transport-law fit + measurement are landed in Rust "
            "(crates/gam-sae/src/manifold/transport_law.rs, validated by "
            "tests_transport_law.rs) but not marshaled across the FFI. Add the "
            "entry point documented in this file's module docstring to "
            "crates/gam-pyffi/src/manifold, rebuild the wheel, then re-run. This "
            "driver intentionally does not reimplement the REML fit in Python."
        )
    return fn


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activations", required=True, help="path to activations.npy (n_prompts, n_layers, D)")
    ap.add_argument("--layer", type=int, default=25, help="source layer L (block target is L+1)")
    ap.add_argument("--rows", type=int, default=None, help="max prompts/rows to use (default: all)")
    ap.add_argument("--n-atoms", type=int, default=32, help="number of shared circle atoms K")
    ap.add_argument("--grid-resolution", type=int, default=512, help="chart grid points over [0,1)")
    ap.add_argument("--max-sweeps", type=int, default=1, help="REML (fit, λ) outer sweeps")
    ap.add_argument("--inner-max-iter", type=int, default=80, help="inner arrow-Schur iterations")
    ap.add_argument("--out", required=True, help="output directory for the JSON report")
    args = ap.parse_args(argv)

    anchor, block = _load_two_layers(args.activations, args.layer, args.rows)
    anchor = _center(anchor)
    block = _center(block)
    print(
        f"loaded OLMo L{args.layer}/L{args.layer + 1}: anchor {anchor.shape} block {block.shape}",
        flush=True,
    )

    from gamfit._binding import rust_module  # noqa: E402  (import after arg parse / data load)

    measure = _require_ffi(rust_module)
    controls = {
        "max_sweeps": args.max_sweeps,
        "inner_max_iter": args.inner_max_iter,
        "step_size": 1.0,
        "ridge_ext_coord": 1e-6,
        "ridge_beta": 1e-6,
        "log_lambda_tol": 1e-3,
    }
    report_json = measure(
        anchor,
        block,
        args.n_atoms,
        args.grid_resolution,
        json.dumps(controls),
    )
    reports = json.loads(report_json)

    import os

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"transport_law_L{args.layer}.json")
    with open(out_path, "w") as fh:
        json.dump({"layer": args.layer, "controls": controls, "atoms": reports}, fh, indent=2)

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
