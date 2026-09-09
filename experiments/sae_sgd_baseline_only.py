#!/usr/bin/env python
"""#1026 — FAST standalone linear SGD-SAE baseline (no gam fit).

Companion to examples/sae_ev_vs_k_olmo.py: that driver produces the gam
manifold-SAE held-out EV-vs-K (curved vs euclidean). This script produces the
matching LINEAR SGD-SAE held-out EV at matched K and at a large overcomplete
budget, on the SAME activation slice / split / PCA projection. It runs in
minutes (no slow gam outer solve), so the two are combined offline into the
matched-budget reconstruction-parity table the #1026 external arm needs.

Identical preprocessing to sae_ev_vs_k_olmo.py (seeded 80/20 split, TRAIN-only
PCA, unit-RMS scale) so the EV numbers are directly comparable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests", "sae"))


def _load(npy, layer):
    arr = np.load(npy)
    if arr.ndim == 3:
        if layer is None:
            raise SystemExit("3D npy needs --olmo-layer")
        arr = arr[:, layer, :]
    return np.asarray(arr, dtype=np.float64)


def _pca(train, test, pcs):
    mean = train.mean(0, keepdims=True)
    tc = train - mean
    _, _, vt = np.linalg.svd(tc, full_matrices=False)
    comp = vt[:pcs].T
    z_tr = tc @ comp
    z_te = (test - mean) @ comp
    scale = np.sqrt(np.mean(z_tr**2)) or 1.0
    return z_tr / scale, z_te / scale


def _ev(target, fitted):
    resid = target - fitted
    denom = float(np.sum((target - target.mean(0, keepdims=True)) ** 2))
    return float("nan") if denom <= 0 else 1.0 - float(np.sum(resid**2)) / denom


def _sgd_ev(z_tr, z_te, dict_size, seed, epochs, l1):
    from torch_sgd_sae import SAEConfig, train_sae
    import torch

    d = z_tr.shape[1]
    cfg = SAEConfig(
        d_in=d, dict_size=dict_size, l1_coeff=l1, activation="relu",
        lr=4e-4, batch_size=min(4096, max(64, z_tr.shape[0])),
        epochs=epochs, seed=seed, device="cpu",
    )
    t0 = time.perf_counter()
    model, stats, _ = train_sae(np.ascontiguousarray(z_tr, dtype=np.float32), cfg)
    fit_s = time.perf_counter() - t0
    ns = stats["norm_scale"]
    dev = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(np.ascontiguousarray(z_te, dtype=np.float32)).to(dev) * ns
        xhat, _ = model(xb)
        recon = (xhat / ns).cpu().numpy()
    return dict(dict_size=dict_size, ev=_ev(z_te, recon),
                mean_l0=float(stats["mean_l0"]),
                dead_frac=float(stats["dead_feature_fraction"]),
                fit_s=fit_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", required=True)
    ap.add_argument("--olmo-layer", type=int, default=None)
    ap.add_argument("--pcs", type=int, default=32)
    ap.add_argument("--k-ladder", default="1,2")
    ap.add_argument("--large-dicts", default="64,256,1024")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--l1", type=float, default=2e-3)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    x = _load(args.npy, args.olmo_layer)
    n = x.shape[0]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_test = max(1, int(round(args.test_frac * n)))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    z_tr, z_te = _pca(x[train_idx], x[test_idx], args.pcs)

    print("=== #1026 LINEAR SGD-SAE baseline (held-out EV) ===")
    print(f"N={n} train={len(train_idx)} test={len(test_idx)} D={x.shape[1]} "
          f"-> PCA-{args.pcs} seed={args.seed}  layer={args.olmo_layer}")

    matchK = [_sgd_ev(z_tr, z_te, int(k), args.seed, args.epochs, args.l1)
              for k in args.k_ladder.split(",") if k.strip()]
    print(f"{'dict':>6} {'held_out_EV':>12} {'mean_L0':>8} {'dead_frac':>10} {'fit_s':>7}")
    for r in matchK:
        print(f"{r['dict_size']:>6} {r['ev']:>12.5f} {r['mean_l0']:>8.2f} "
              f"{r['dead_frac']:>10.3f} {r['fit_s']:>7.1f}  (matched-K)")
    large = [_sgd_ev(z_tr, z_te, int(d), args.seed, args.epochs, args.l1)
             for d in args.large_dicts.split(",") if d.strip()]
    for r in large:
        print(f"{r['dict_size']:>6} {r['ev']:>12.5f} {r['mean_l0']:>8.2f} "
              f"{r['dead_frac']:>10.3f} {r['fit_s']:>7.1f}  (overcomplete)")

    if args.out:
        json.dump(dict(config=vars(args), n=n, n_train=len(train_idx),
                       n_test=len(test_idx), d_model=int(x.shape[1]),
                       matched_k=matchK, large=large), open(args.out, "w"),
                  indent=2, default=str)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
