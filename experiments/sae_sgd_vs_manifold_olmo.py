#!/usr/bin/env python
"""#1026 — matched-budget LINEAR SGD-SAE vs gam MANIFOLD-SAE on real LLM acts.

The #1026 reconstruction-parity question, posed honestly as the issue asks it:

    "Does the curved manifold dictionary reach reconstruction parity-or-better
     with a large linear SAE at matched K, on REAL LLM activations?"

This driver runs BOTH baselines on the SAME real activation slice and the SAME
held-out split, measuring HELD-OUT reconstruction explained variance (EV):

  A) gam manifold-SAE (curved circle + degree-2 quadratic "euclidean" patch —
     NOT a linear atom, #1201), K small, via the
     production engine `gamfit.sae_manifold_fit` + `ManifoldSAE.reconstruct`.
     This is the matched-K, same-solver, geometry-only-differs comparison.

  B) a REAL overcomplete linear SGD SAE (`tests/sae/torch_sgd_sae.py`), trained
     by Adam with an L1 sparsity penalty and unit-norm decoder columns — the
     standard mech-interp dictionary. We report it at matched-K (dict_size=K, the
     parity point) AND at a large overcomplete budget (the "large linear SAE"
     the issue references) so the curve can be read both ways.

EV is held-out, in the same PCA-projected coordinate the manifold SAE is fit in,
so the two arms are directly comparable. NO topology is forced on the manifold
arm beyond the {circle, euclidean} the engine is asked to fit; the honest read is
the per-K margin curved_EV - linear_EV and the SGD parity gap.

This is a thin numeric adapter (the #977 boundary): activations are a response
matrix. No fitting math lives here.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# Make tests/sae importable for torch_sgd_sae.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests", "sae"))


def _load_activations(npy: str, olmo_layer: int | None) -> np.ndarray:
    arr = np.load(npy)
    if arr.ndim == 3:
        if olmo_layer is None:
            raise SystemExit("3D activations.npy needs --olmo-layer")
        arr = arr[:, olmo_layer, :]
    return np.asarray(arr, dtype=np.float64)


def _pca_project(train: np.ndarray, test: np.ndarray, pcs: int):
    mean = train.mean(axis=0, keepdims=True)
    tc = train - mean
    _, _, vt = np.linalg.svd(tc, full_matrices=False)
    comp = vt[:pcs].T
    z_tr = tc @ comp
    z_te = (test - mean) @ comp
    scale = np.sqrt(np.mean(z_tr**2)) or 1.0
    return z_tr / scale, z_te / scale


def _ev(target: np.ndarray, fitted: np.ndarray) -> float:
    resid = target - fitted
    denom = float(np.sum((target - target.mean(axis=0, keepdims=True)) ** 2))
    if denom <= 0.0:
        return float("nan")
    return 1.0 - float(np.sum(resid**2)) / denom


def _manifold_ev(z_tr, z_te, k, topology, seed, n_iter):
    from gamfit import sae_manifold_fit

    t0 = time.perf_counter()
    m = sae_manifold_fit(
        z_tr, K=k, d_atom=1, atom_topology=topology,
        assignment="ordered_beta_bernoulli", n_iter=n_iter, random_state=seed,
    )
    fit_s = time.perf_counter() - t0
    fitted = m.reconstruct(z_te)
    return _ev(z_te, fitted), fit_s


def _sgd_ev(z_tr, z_te, dict_size, seed, epochs, l1, device):
    """Train a linear SGD SAE on z_tr (PCA coords), return held-out EV.

    Reconstruction is in the SAME normalized PCA space, so EV is comparable to
    the manifold arm. The SGD SAE applies its own internal dataset rescale; we
    invert it on the reconstruction so the EV is measured against the raw z_te.
    """
    from torch_sgd_sae import SAEConfig, train_sae
    import torch

    d = z_tr.shape[1]
    cfg = SAEConfig(
        d_in=d, dict_size=dict_size, l1_coeff=l1, activation="relu",
        lr=4e-4, batch_size=min(4096, max(64, z_tr.shape[0])),
        epochs=epochs, seed=seed,
        device=device if torch.cuda.is_available() else "cpu",
    )
    t0 = time.perf_counter()
    model, stats, _ = train_sae(np.ascontiguousarray(z_tr, dtype=np.float32), cfg)
    fit_s = time.perf_counter() - t0

    ns = stats["norm_scale"]
    dev = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(np.ascontiguousarray(z_te, dtype=np.float32)).to(dev)
        xb = xb * ns                      # apply same dataset normalization
        xhat, _ = model(xb)
        recon = (xhat / ns).cpu().numpy()  # invert scale back to raw PCA coords
    return _ev(z_te, recon), fit_s, float(stats["mean_l0"]), float(stats["dead_feature_fraction"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npy", required=True)
    ap.add_argument("--olmo-layer", type=int, default=None)
    ap.add_argument("--pcs", type=int, default=32)
    ap.add_argument("--k-ladder", default="1,2")
    ap.add_argument("--sgd-large-dict", type=int, default=256,
                    help="overcomplete budget for the 'large linear SAE' reference")
    ap.add_argument("--sgd-epochs", type=int, default=40)
    ap.add_argument("--sgd-l1", type=float, default=2e-3)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-iter", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    x = _load_activations(args.npy, args.olmo_layer)
    n = x.shape[0]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_test = max(1, int(round(args.test_frac * n)))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    z_tr, z_te = _pca_project(x[train_idx], x[test_idx], args.pcs)
    ladder = [int(s) for s in args.k_ladder.split(",") if s.strip()]

    print("=== #1026 matched-budget LINEAR-SGD vs gam-MANIFOLD reconstruction parity ===")
    print(f"N={n} (train={len(train_idx)} test={len(test_idx)}) D={x.shape[1]} "
          f"-> PCA-{args.pcs}, seed={args.seed}")
    print(f"{'K':>4} {'gam_circle':>11} {'gam_euclid':>11} {'sgd_matchK':>11} "
          f"{'curv-lin':>9} {'curv-sgd':>9} {'sgd_L0':>7}")

    rows = []
    for k in ladder:
        ev_c, _ = _manifold_ev(z_tr, z_te, k, "circle", args.seed, args.n_iter)
        ev_e, _ = _manifold_ev(z_tr, z_te, k, "euclidean", args.seed, args.n_iter)
        ev_s, _, l0_s, dead_s = _sgd_ev(z_tr, z_te, k, args.seed, args.sgd_epochs,
                                        args.sgd_l1, args.device)
        rows.append(dict(K=k, gam_circle=ev_c, gam_euclidean=ev_e,
                         sgd_matchK=ev_s, sgd_L0=l0_s, sgd_dead=dead_s))
        print(f"{k:>4} {ev_c:>11.5f} {ev_e:>11.5f} {ev_s:>11.5f} "
              f"{ev_c-ev_e:>9.5f} {ev_c-ev_s:>9.5f} {l0_s:>7.2f}")

    # The "large linear SAE" reference at an overcomplete budget.
    ev_big, _, l0_big, dead_big = _sgd_ev(
        z_tr, z_te, args.sgd_large_dict, args.seed, args.sgd_epochs,
        args.sgd_l1, args.device)
    print(f"\nLARGE linear SGD SAE (dict={args.sgd_large_dict}): "
          f"held-out EV={ev_big:.5f}  mean_L0={l0_big:.2f}  dead_frac={dead_big:.3f}")
    best_curved = max((r["gam_circle"] for r in rows), default=float("nan"))
    print(f"best gam curved (any K on ladder) EV={best_curved:.5f}")
    print(f"\nPARITY READ: gam-curved-best minus large-linear-SGD = "
          f"{best_curved - ev_big:+.5f}  "
          f"(>=0 => curved reaches parity-or-better at a FRACTION of the dictionary budget)")

    if args.out:
        import json
        summary = dict(
            config=vars(args), n=n, n_train=len(train_idx), n_test=len(test_idx),
            d_model=int(x.shape[1]), per_k=rows,
            sgd_large=dict(dict_size=args.sgd_large_dict, ev=ev_big,
                           mean_l0=l0_big, dead_frac=dead_big),
            best_curved_ev=best_curved,
            parity_margin=best_curved - ev_big,
        )
        with open(args.out, "w") as fh:
            json.dump(summary, fh, indent=2, default=str)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
