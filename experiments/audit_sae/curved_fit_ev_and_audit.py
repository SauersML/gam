#!/usr/bin/env python3
"""One curved fit -> the #1026 EV number AND the #1942 manifold-native numbers.

The shared-infra leverage: a single ``gamfit.sae_manifold_fit(...)`` is the
expensive step; from that one fitted handle we read

  * the #1026 close-bar number — held-out explained variance of the curved
    reconstruction (``model.reconstruct(x_te)``), scored against the train-mean
    baseline exactly as experiments/1026_close/driver_1026_arms.py does, so it is
    directly comparable to the external_topk / hybrid arms; and

  * the #1942 manifold-native numbers — the three metric accessors
    (``per_atom_curvature`` / ``chart_occupancy`` / ``geodesic_recon_gap``) plus
    the frozen-dictionary ``audit_sae`` arm fed by ``model.frozen_dictionary()``.

This driver is PURE ORCHESTRATION: every metric's math lives in the Rust core
behind a pyffi accessor (SPEC.md thin-wrapper rule). The metric accessors and
``frozen_dictionary`` are being routed to the gam-sae / gam-pyffi owner; until
each lands this driver records ``{"pending": "<accessor>"}`` for that arm and
still emits the EV number, so it is runnable the instant the wheel lands and
grows more arms as the accessors land — nothing here is stubbed or fabricated.

The flat counterpart is ``extract_flat_decoder.py`` (traditional-SAE bar).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Same-dir sibling: the #1026-semantics chunk loader (no re-implementation).
from extract_flat_decoder import _load_chunk_dir, _split


def _held_out_ev(x_te: np.ndarray, recon: np.ndarray, mean_tr: np.ndarray) -> float:
    """Train-mean-baseline held-out EV, identical to
    experiments/1026_close/driver_1026_arms.py:held_out_ev (baseline-result
    analysis — a driver-level scalar, not a manifold-native library metric)."""
    ssr = float(np.sum((x_te.astype(np.float64) - recon.astype(np.float64)) ** 2))
    sst = float(np.sum((x_te.astype(np.float64) - mean_tr.astype(np.float64)[None, :]) ** 2))
    return 1.0 - ssr / max(sst, 1e-300)


def _metric_arm(gamfit, model, name: str, *args):
    """Call the manifold-native metric accessor ``name`` if the wheel exposes it.

    Tries the module-function form ``gamfit.<name>(model, *args)`` first (matching
    chart_interp_score / dose_response_calibration / audit_sae), then a method
    ``model.<name>(*args)``. Records a precise ``pending`` marker otherwise so the
    report says exactly which routed accessor is still missing."""
    fn = getattr(gamfit, name, None)
    if fn is not None:
        return {"source": f"gamfit.{name}", "report": _jsonable(fn(model, *args))}
    meth = getattr(model, name, None)
    if callable(meth):
        return {"source": f"model.{name}", "report": _jsonable(meth(*args))}
    return {"pending": name, "note": f"routed to gam-sae/gam-pyffi owner; not in this wheel"}


def _jsonable(obj):
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        pass
    for attr in ("to_dict", "model_dump", "__dict__"):
        member = getattr(obj, attr, None)
        if callable(member):
            try:
                return _jsonable(member())
            except Exception:  # noqa: BLE001
                break
        if isinstance(member, dict):
            return _jsonable(member)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return repr(obj)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunk-dir", type=Path, required=True, help="dir of chunk_*.npy activation shards")
    ap.add_argument("--out", type=Path, required=True, help="output report json")
    ap.add_argument("--K", type=int, default=32768, help="dictionary size (default = #1026 32k close-bar)")
    ap.add_argument("--d-atom", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--topology", default="circle", help="atom_topology (linear/circle/...)")
    ap.add_argument("--rows", type=int, default=120000)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--audit-active", type=int, default=1)
    ap.add_argument("--audit-block-size", type=int, default=1)
    args = ap.parse_args()

    import gamfit

    X = _load_chunk_dir(args.chunk_dir, args.rows, args.seed)
    x_tr, x_te = _split(X, args.test_frac, args.seed)
    mean_tr = x_tr.astype(np.float64).mean(0)
    print(f"[curved] X={X.shape} train={x_tr.shape} test={x_te.shape} "
          f"K={args.K} d_atom={args.d_atom} top_k={args.top_k} topology={args.topology}", flush=True)

    model = gamfit.sae_manifold_fit(
        x_tr, K=args.K, d_atom=args.d_atom, atom_topology=args.topology,
        assignment="topk", top_k=args.top_k, random_state=args.seed,
    )

    report: dict = {
        "driver": "experiments/audit_sae/curved_fit_ev_and_audit.py",
        "config": {
            "chunk_dir": str(args.chunk_dir), "K": args.K, "d_atom": args.d_atom,
            "top_k": args.top_k, "topology": args.topology, "rows": int(X.shape[0]),
            "P": int(X.shape[1]), "seed": args.seed,
        },
    }

    # --- #1026 arm: held-out EV of the curved reconstruction (available now) ---
    recon = np.asarray(model.reconstruct(x_te), dtype=np.float64)
    report["issue_1026"] = {"held_out_ev": _held_out_ev(x_te, recon, mean_tr)}
    print(f"[curved] #1026 held_out_ev={report['issue_1026']['held_out_ev']:.4f}", flush=True)

    # --- #1942 manifold-native metric arms (accessor-gated; guarded) ---
    report["issue_1942_metrics"] = {
        "per_atom_curvature": _metric_arm(gamfit, model, "per_atom_curvature"),
        "chart_occupancy": _metric_arm(gamfit, model, "chart_occupancy"),
        "geodesic_recon_gap": _metric_arm(gamfit, model, "geodesic_recon_gap", x_te),
    }

    # --- #1942 frozen-dictionary audit arm (needs the K x P getter) ---
    frozen = getattr(model, "frozen_dictionary", None)
    if callable(frozen):
        decoder = np.ascontiguousarray(np.asarray(frozen(), dtype=np.float32))
        audit = gamfit.audit_sae(
            decoder, x_te.astype(np.float32, copy=False),
            random_weight_codes=np.zeros((x_te.shape[0], decoder.shape[0]), dtype=np.float32),
            active=args.audit_active, block_size=args.audit_block_size,
        )
        report["issue_1942_audit"] = {"source": "model.frozen_dictionary + gamfit.audit_sae",
                                      "report": _jsonable(audit)}
        print(f"[curved] #1942 audit ran on frozen dictionary K x P={decoder.shape}", flush=True)
    else:
        report["issue_1942_audit"] = {"pending": "frozen_dictionary",
                                      "note": "curved audit_sae needs a top-level K x P decoder getter"}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"[curved] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
