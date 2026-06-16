#!/usr/bin/env python
"""#1026 — REAL manifold-SAE training on per-token LLM activations, discovery ON.

This is the honest end-to-end run the manifold-SAE machinery exists for:

  * load a LARGE per-token residual-stream activation matrix (e.g. Qwen2.5-7B
    wikitext, layer 14: ``resid_L14.npy`` is (300000, 3584) float32);
  * hold out a genuine random TEST split of tokens (no leakage);
  * mean-center on TRAIN, PCA-reduce on TRAIN ONLY to a principled budget
    (report the % variance kept), unit-RMS scale from TRAIN;
  * fit the production manifold-SAE with a REAL K (>=16) and TOPOLOGY DISCOVERY
    on — ``atom_topology`` is only a SEED; the Rust structure search does the
    evidence-gated births/fissions and re-derives the per-atom dictionary, so
    ``m.basis_specs`` / ``m.atom_topologies`` / ``m.chosen_k`` report what the
    model CHOSE, not what was forced;
  * reconstruct the HELD-OUT tokens via the OOS path ``m.reconstruct(z_test)``
    and report held-out reconstruction EV;
  * fit a matched-budget LINEAR baseline (forced euclidean d=1) and report its
    held-out EV too.

NO fitting math lives here. This is the #977 numeric boundary: activations are
just a response matrix fed to ``gamfit.sae_manifold_fit`` and read back through
``ManifoldSAE``. The honest question it answers: does the discovered-geometry
dictionary beat a matched linear SAE on held-out real activations, and what
topology did it discover?

EXAMPLE (Qwen2.5-7B wikitext layer 14, 40k-token subsample, on a compute node):
  python examples/train_real_manifold_sae.py \
      --npy /projects/.../harvest_out/qwen25_7b_wikitext_v2/resid_L14.npy \
      --subsample 40000 --pcs 64 --k 32 --n-iter 40 --seed 42
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import numpy as np


def _load(npy, subsample: int | None, seed: int) -> np.ndarray:
    """Memory-map the (N, D) activation matrix/chunks and optionally row-subsample."""
    paths = [npy] if isinstance(npy, str) else list(npy)
    mms = []
    for p in paths:
        a = np.load(p, mmap_mode="r")
        if a.ndim == 3:
            raise SystemExit("3D npy not supported here; pass a 2D (N, D) per-token cache")
        mms.append(a)
    sizes = [a.shape[0] for a in mms]
    n = sum(sizes)
    if subsample is not None and subsample < n:
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(n, size=subsample, replace=False))
    else:
        rows = np.arange(n)
    # map global row indices to (chunk, local) and gather as float64
    offsets = np.cumsum([0] + sizes)
    out = np.empty((len(rows), mms[0].shape[1]), dtype=np.float64)
    for ci, a in enumerate(mms):
        lo, hi = offsets[ci], offsets[ci + 1]
        m = (rows >= lo) & (rows < hi)
        if m.any():
            out[m] = np.asarray(a[rows[m] - lo], dtype=np.float64)
    return out


def _pca_project(train: np.ndarray, test: np.ndarray, pcs: int,
                 drop_top: int = 0, whiten: bool = True):
    """TRAIN-only mean-center + PCA; project both; scale from TRAIN.

    ``drop_top`` discards the leading ``drop_top`` principal components before
    keeping ``pcs`` of the remainder. Real LLM residual streams at mid/late
    layers carry one (or a few) "rogue" outlier directions that dominate raw
    variance and mask the structured K-frontier underneath (documented on
    #1026); dropping them exposes the real signal. ``whiten`` standardizes each
    kept PC to unit TRAIN variance so no single axis sets the global scale and
    starves the rest (the failure mode behind the dictionary co-collapse).

    Returns (z_tr, z_te, var_kept_fraction_of_post_drop_total).
    """
    mean = train.mean(axis=0, keepdims=True)
    tc = train - mean
    # economy SVD on centered TRAIN; right-singular vectors are the PCs.
    _, s, vt = np.linalg.svd(tc, full_matrices=False)
    total_var = float(np.sum(s[drop_top:] ** 2))  # variance available after rogue drop
    sel = slice(drop_top, drop_top + pcs)
    comp = vt[sel].T  # (D, pcs)
    s_sel = s[sel]
    var_kept = float(np.sum(s_sel**2)) / total_var if total_var > 0 else 0.0
    z_tr = tc @ comp
    z_te = (test - mean) @ comp
    if whiten:
        # per-PC unit TRAIN std; guards against a single PC dominating the scale.
        sd = np.sqrt(np.mean(z_tr**2, axis=0, keepdims=True))
        sd[sd == 0] = 1.0
        return z_tr / sd, z_te / sd, var_kept
    scale = float(np.sqrt(np.mean(z_tr**2))) or 1.0
    return z_tr / scale, z_te / scale, var_kept


def _ev(target: np.ndarray, fitted: np.ndarray) -> float:
    """Held-out reconstruction explained variance (1 - SSE/SST), SST about mean."""
    resid = target - fitted
    denom = float(np.sum((target - target.mean(axis=0, keepdims=True)) ** 2))
    if denom <= 0.0:
        return float("nan")
    return 1.0 - float(np.sum(resid**2)) / denom


def _fit(z_tr, z_te, k, topology, seed, n_iter):
    from gamfit import sae_manifold_fit

    t0 = time.perf_counter()
    m = sae_manifold_fit(
        z_tr,
        K=k,
        d_atom=1,
        atom_topology=topology,  # SEED only; structure search re-derives the dict
        assignment="ibp_map",
        n_iter=n_iter,
        random_state=seed,
    )
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    fitted_te = m.reconstruct(z_te)
    recon_s = time.perf_counter() - t1
    return m, _ev(z_te, fitted_te), fit_s, recon_s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npy", required=True, nargs="+",
                    help="2D (N, D) per-token activation cache(s); multiple chunks are concatenated")
    ap.add_argument("--subsample", type=int, default=40000, help="rows to keep (None-> all)")
    ap.add_argument("--pcs", type=int, default=64, help="PCA components kept")
    ap.add_argument("--drop-top-pcs", type=int, default=0,
                    help="discard leading N rogue PCs before keeping --pcs (real LLM residual streams)")
    ap.add_argument("--no-whiten", action="store_true",
                    help="use global RMS scale instead of per-PC unit-variance whitening")
    ap.add_argument("--k", type=int, default=32, help="seed K for the manifold SAE (>=16)")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-iter", type=int, default=40)
    ap.add_argument("--seed-topology", default="circle", help="discovery SEED topology")
    ap.add_argument("--json-out", default=None, help="optional path to dump results json")
    args = ap.parse_args()

    x = _load(args.npy, args.subsample, args.seed)
    n, d = x.shape
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_test = max(1, int(round(args.test_frac * n)))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    z_tr, z_te, var_kept = _pca_project(
        x[train_idx], x[test_idx], args.pcs,
        drop_top=args.drop_top_pcs, whiten=not args.no_whiten)

    print("=== #1026 REAL manifold-SAE training (discovery ON) ===")
    print(f"data: {args.npy}")
    print(f"N={n} tokens (train={len(train_idx)}, test={len(test_idx)}), D={d}")
    print(f"PCA: dropped top {args.drop_top_pcs} rogue PC(s), kept {args.pcs} comps = "
          f"{100*var_kept:.1f}% of post-drop TRAIN variance, "
          f"whiten={not args.no_whiten}")
    print(f"seed K={args.k}, seed topology={args.seed_topology!r}, n_iter={args.n_iter}, seed={args.seed}")
    print()

    # --- manifold SAE: discovery ON (seed topology only) ---
    m, ev_m, fit_m, recon_m = _fit(z_tr, z_te, args.k, args.seed_topology, args.seed, args.n_iter)
    kinds = list(m.basis_specs)
    dist = Counter(kinds)
    print(f"[manifold] discovered K = {m.chosen_k}")
    print(f"[manifold] per-atom topology distribution: {dict(dist)}")
    print(f"[manifold] scalar atom_topology = {m.atom_topology!r}")
    print(f"[manifold] held-out EV = {ev_m:.4f}   (fit {fit_m:.1f}s, recon {recon_m:.1f}s)")
    try:
        cert = m.structure_certificate()
        confirmed = [c for c in cert.get("claims", []) if c.get("confirmed")]
        print(f"[manifold] structure certificate: {len(confirmed)} confirmed claim(s) "
              f"of {len(cert.get('claims', []))} at alpha={cert.get('alpha')}")
    except Exception as e:  # noqa: BLE001
        print(f"[manifold] structure certificate unavailable: {e}")
    print()

    # --- linear baseline: forced euclidean d=1 at matched K ---
    mlin, ev_lin, fit_lin, recon_lin = _fit(z_tr, z_te, args.k, "euclidean", args.seed, args.n_iter)
    print(f"[linear]   K = {mlin.chosen_k}, topology dist = {dict(Counter(mlin.basis_specs))}")
    print(f"[linear]   held-out EV = {ev_lin:.4f}   (fit {fit_lin:.1f}s, recon {recon_lin:.1f}s)")
    print()

    margin = ev_m - ev_lin
    verdict = "manifold BEATS linear" if margin > 0 else "manifold does NOT beat linear"
    print(f"=== held-out EV: manifold {ev_m:.4f} vs linear {ev_lin:.4f}  "
          f"(margin {margin:+.4f}) -> {verdict} ===")

    if args.json_out:
        out = {
            "npy": args.npy, "N": n, "D": d,
            "n_train": len(train_idx), "n_test": len(test_idx),
            "pcs": args.pcs, "var_kept_frac": var_kept,
            "seed_k": args.k, "seed_topology": args.seed_topology,
            "n_iter": args.n_iter, "seed": args.seed,
            "manifold": {
                "chosen_k": int(m.chosen_k),
                "topology_distribution": dict(dist),
                "scalar_topology": m.atom_topology,
                "held_out_ev": ev_m, "fit_s": fit_m, "recon_s": recon_m,
            },
            "linear": {
                "chosen_k": int(mlin.chosen_k),
                "topology_distribution": dict(Counter(mlin.basis_specs)),
                "held_out_ev": ev_lin, "fit_s": fit_lin, "recon_s": recon_lin,
            },
            "margin": margin,
        }
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
