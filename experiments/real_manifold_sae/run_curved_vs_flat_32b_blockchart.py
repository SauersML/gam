"""Overcomplete curved-vs-flat on Qwen3-30B-A3B (32B MoE) L17 via the WORKING
BLOCK-CHART compose lane (block_sparse_dictionary_fit + compose_block_charts),
bypassing the stagewise-T2 lane (_fit_stagewise_t2) which HANGS (self-concordance
metric bug: 75+ min silent on 4k rows).

Pipeline (product-faithful, CONFIG_FROZEN.md):
  T1  : frozen OVERCOMPLETE linear sparse dictionary decoder (decoder_K32000.npy,
        K=32000 = 15.6x p, unit-norm) with the frozen RIDGE-LS transform (per-row
        top-|score| active=32 support + ridge least-squares codes). NOT dot-product
        (dot-product overshoots on correlated unit-norm atoms -> EV ~ -64). Ridge-LS
        gives EV(T1) ~ 0.707.
  T2  : on the T1 residual, fit block-chart FLAT (linear blocks) vs CURVED (circle
        blocks via compose_block_charts) at MATCHED params, STRATUM-LOCAL
        (energy-exponent strata == residual_stratify; the POOLED residual fails the
        routability floor sqrt(2 ln K / p)=0.10, so births MUST be stratum-local).
        Matched budget mirrors the wall-closure: 24 flat == 12 curved x (4+4).

Reuses experiments/geometric_wall/wall_closure_common.py verbatim for the block-chart
lane (fit_stratum / fit_block_dictionary / build_strata / matched_curved_blocks) --
the SAME code route2 used to close the wall on Qwen-8B L18 (flat 0.741 vs curved 0.789).

DELIVERABLE: EV(T1), EV(+flat), EV(+curved), dEV(curved-flat), per-stratum + pooled.
Honest either sign.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import gamfit

# Reuse the WORKING block-chart lane verbatim. Look in the repo layout
# (../geometric_wall) and next to this script (MSI staging drops both flat).
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "geometric_wall"))
from wall_closure_common import (  # noqa: E402
    build_strata,
    fit_stratum,
    matched_curved_blocks,
    squared_energy,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def t1_ridge_recon(X, dec, active, ridge=1e-6, chunk=4096):
    """Frozen T1 recipe: per row pick top-|score| active atoms (score = X @ decoderT),
    then ridge least-squares codes over that active support -> dense reconstruction.
    Chunked to bound memory. CPU only."""
    N, p = X.shape
    K = dec.shape[0]
    s = min(active, K)
    recon = np.empty_like(X)
    eye = ridge * np.eye(s)
    for a in range(0, N, chunk):
        Xc = X[a:a + chunk]
        scores = Xc @ dec.T
        top = np.argpartition(-np.abs(scores), s - 1, axis=1)[:, :s]
        Da = dec[top]
        G = np.einsum("bsp,btp->bst", Da, Da) + eye
        rhs = np.einsum("bsp,bp->bs", Da, Xc)
        codes = np.linalg.solve(G, rhs[:, :, None])[:, :, 0]
        recon[a:a + chunk] = np.einsum("bs,bsp->bp", codes, Da)
    return recon


def _jsonable(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--rows", type=int, default=40000)
    ap.add_argument("--t1-decoder")
    ap.add_argument("--t1-active", type=int, default=32)
    ap.add_argument("--tier0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-stratum-rows", type=int, default=512)
    # Block-chart lane knobs (mirror qwen_wall_closure defaults).
    ap.add_argument("--flat-blocks", type=int, default=24)
    ap.add_argument("--curved-blocks", type=int, default=0)
    ap.add_argument("--block-size", type=int, default=4)
    ap.add_argument("--chart-basis", type=int, default=4)
    ap.add_argument("--block-topk", type=int, default=2)
    ap.add_argument("--max-epochs", type=int, default=8)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--block-tile", type=int, default=512)
    ap.add_argument("--frame-ridge", type=float, default=1.0e-9)
    ap.add_argument("--tolerance", type=float, default=1.0e-5)
    ap.add_argument("--min-firings", type=int, default=32)
    ap.add_argument("--max-chart-blocks", type=int, default=256)
    ap.add_argument("--crossfit-folds", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--whitening-ridge", type=float, default=1.0e-8)
    ap.add_argument("--pair-screen", action="store_true")
    ap.add_argument("--pair-top-blocks", type=int, default=64)
    ap.add_argument("--max-pairs", type=int, default=128)
    ap.add_argument("--pair-min-cofirings", type=int, default=64)
    ap.add_argument("--pair-min-score", type=float, default=0.20)
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    if args.t1_decoder is None:
        args.t1_decoder = str(root / "msae_l17" / "t1_out" / "decoder_K32000.npy")
    if args.tier0 is None:
        args.tier0 = str(root / "msae_l17" / "tier0_recentered.json")
    os.makedirs(args.out, exist_ok=True)
    ver = getattr(gamfit, "__version__", "?")

    # ---- tier-0 recentered space ----
    t0meta = json.load(open(args.tier0))
    mean = np.asarray(t0meta["per_dim_mean"], dtype=np.float64)
    scale = float(t0meta["global_rms_scale"])
    train = str(root / "msae_l17" / "L17_train.f32.npy")
    Xmm = np.load(train, mmap_mode="r")
    n_rows, D = Xmm.shape
    N = min(args.rows, n_rows)
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(n_rows, N, replace=False))
    X = (np.asarray(Xmm[idx], dtype=np.float64) - mean) / scale
    log(f"loaded L17 {Xmm.shape} -> {X.shape} gamfit={ver} tier0=recentered")

    # ---- T1: frozen overcomplete decoder, CPU top-active + RIDGE transform ----
    dec = np.load(args.t1_decoder).astype(np.float64)
    K1 = dec.shape[0]
    t0 = time.time()
    t1_recon = t1_ridge_recon(X, dec, args.t1_active, ridge=1e-6, chunk=4096)
    residual = np.ascontiguousarray(X - t1_recon, dtype=np.float32)
    tss = squared_energy(X)
    t1_rss = squared_energy(residual)
    ev_t1 = 1.0 - t1_rss / max(tss, 1e-30)
    log(f"T1 K={K1} ({K1/D:.1f}x) active={args.t1_active}: EV(T1)={ev_t1:.4f} "
        f"({time.time()-t0:.1f}s)")

    # ---- stratify the residual (energy-exponent strata == residual_stratify) ----
    strata = build_strata(residual)
    curved_blocks = matched_curved_blocks(
        args.flat_blocks, args.block_size, args.chart_basis, args.curved_blocks)
    log(f"strata={len(strata)} flat_blocks={args.flat_blocks} "
        f"curved_blocks={curved_blocks} (matched params)")

    fitted = []
    skipped = []
    for spec in strata:
        if spec.n_rows < args.min_stratum_rows:
            skipped.append({"stratum": int(spec.index), "n_rows": int(spec.n_rows),
                            "reason": "below_min_stratum_rows"})
            continue
        try:
            t0 = time.time()
            row = fit_stratum(f"L17", spec, residual, args.flat_blocks, curved_blocks, args)
            log(f"  stratum {spec.index}: rows={row['n_rows']} "
                f"flat_floor={row['flat_floor']:.4f} curved_floor={row['curved_floor']:.4f} "
                f"drop={row['drop']:+.4f} ({time.time()-t0:.1f}s)")
            fitted.append(row)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            skipped.append({"stratum": int(spec.index), "n_rows": int(spec.n_rows),
                            "reason": type(exc).__name__, "message": str(exc)})

    if not fitted:
        raise SystemExit("no strata fitted")

    # ---- compose EV over the ORIGINAL X (baseline zero == tier-0 origin) ----
    # residual RSS = t1_rss (full). Fitted strata replace their residual-energy
    # contribution with the post-T2 floored energy.
    fitted_res_energy = sum(r["target_energy"] for r in fitted)
    flat_rss_fitted = sum(r["flat_floor"] * r["target_energy"] for r in fitted)
    curved_rss_fitted = sum(r["curved_floor"] * r["target_energy"] for r in fitted)
    flat_rss_total = t1_rss - fitted_res_energy + flat_rss_fitted
    curved_rss_total = t1_rss - fitted_res_energy + curved_rss_fitted
    ev_flat = 1.0 - flat_rss_total / max(tss, 1e-30)
    ev_curved = 1.0 - curved_rss_total / max(tss, 1e-30)
    dev = ev_curved - ev_flat

    # Residual-tier pooled (over fitted strata only), matches wall-closure convention.
    pooled_flat_floor = flat_rss_fitted / max(fitted_res_energy, 1e-30)
    pooled_curved_floor = curved_rss_fitted / max(fitted_res_energy, 1e-30)

    results = {
        "model": "Qwen3-30B-A3B (MoE, 32B) L17 residual",
        "lane": "BLOCK-CHART compose (block_sparse_dictionary_fit + compose_block_charts) "
                "-- NOT stagewise-T2 (which hangs)",
        "engine": "gamfit.block_sparse_dictionary_fit + BlockSparseDictionaryFit.compose_block_charts",
        "gamfit_version": ver,
        "provenance": {
            "t1_transform": "ridge-LS top-active support (active=32), NOT dot-product",
            "centering": "tier0_recentered",
            "ev_baseline": "zero (tier-0 origin == train mean)",
            "stratification": "energy-exponent strata (residual_stratify); births stratum-local",
        },
        "N_subsample": int(N), "D": int(D),
        "tier0": os.path.basename(args.tier0),
        "T1": {"decoder": os.path.basename(args.t1_decoder), "K": int(K1),
               "overcomplete_ratio": K1 / D, "active": args.t1_active},
        "matched_budget": {
            "flat_blocks": int(args.flat_blocks), "curved_blocks": int(curved_blocks),
            "block_size": int(args.block_size), "chart_basis": int(args.chart_basis),
            "flat_units": int(args.flat_blocks * args.block_size),
            "curved_units": int(curved_blocks * (args.block_size + args.chart_basis)),
            "flat_params_per_row_dim": int(args.flat_blocks * args.block_size * D),
            "curved_params_per_row_dim": int(curved_blocks * (args.block_size + args.chart_basis) * D),
        },
        "strata_total": int(len(strata)), "strata_fitted": int(len(fitted)),
        "strata_skipped": skipped,
        "headline": {
            "ev_t1": float(ev_t1),
            "ev_t1_plus_flat": float(ev_flat),
            "ev_t1_plus_curved": float(ev_curved),
            "delta_ev_curved_minus_flat": float(dev),
            "pooled_residual_flat_floor": float(pooled_flat_floor),
            "pooled_residual_curved_floor": float(pooled_curved_floor),
            "pooled_residual_drop": float(pooled_flat_floor - pooled_curved_floor),
        },
        "strata": fitted,
    }
    with open(os.path.join(args.out, "numbers.json"), "w") as f:
        json.dump(_jsonable(results), f, indent=2)
    log("=" * 60)
    log(f"HEADLINE  EV(T1)={ev_t1:.4f}  EV(+flat)={ev_flat:.4f}  "
        f"EV(+curved)={ev_curved:.4f}")
    log(f"HEADLINE  dEV(curved-flat)={dev:+.5f}  "
        f"(pooled-residual drop={pooled_flat_floor-pooled_curved_floor:+.5f})")
    log("WROTE " + os.path.join(args.out, "numbers.json"))


if __name__ == "__main__":
    main()
