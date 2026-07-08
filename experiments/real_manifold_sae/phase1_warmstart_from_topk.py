"""Warm-start lever: seed the curved fit's routing from the team's CONVERGED
flat TopK SAE instead of cold PCA.

Thesis connection: under the manifold hypothesis, a flat overcomplete SAE
splits each 1-D feature curve into many near-redundant point atoms. Grouping
the flat atoms by decoder direction recovers curve candidates; the flat
encoder's measured activations then give an honest warm assignment a_init for
the curved fit. Cold-start comparison comes from the phase-1 fleet.

Runs the same config twice (cold vs warm a_init) on the identical slice and
reports both r² and wall.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-root", type=Path, default=Path("/models/k25_tokens"))
    ap.add_argument("--category", default="emotions")
    ap.add_argument("--ckpt", type=Path, default=Path("/models/sae/k25-145M-16x-k64_latest.pt"))
    ap.add_argument("--rows", type=int, default=40000)
    ap.add_argument("--d-pca", type=int, default=64)
    ap.add_argument("--k-atoms", type=int, default=8)
    ap.add_argument("--n-iter", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache-dir", type=Path, default=Path("/dev/shm/gnome/cache"))
    ap.add_argument("--out", type=Path, default=Path("/dev/shm/gnome/results/warmstart.json"))
    args = ap.parse_args()

    import torch
    import gamfit
    from phase1_reml_real_revalidation import PcaCache, load_slice

    z_raw = load_slice(args.acts_root, args.category, args.rows, args.seed, args.cache_dir)
    pca = PcaCache(z_raw, args.device)
    z, ev_frac = pca.project(args.d_pca)
    n = z.shape[0]
    print(f"[warm] slice {z_raw.shape} -> D={args.d_pca} (capture {ev_frac:.3f})", flush=True)

    # ---- flat TopK SAE: encode the slice, group atoms by decoder direction ----
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    dev = torch.device(args.device)
    w_enc = state["W_enc"].to(dev, torch.float32)
    b_enc = state["b_enc"].to(dev, torch.float32)
    w_dec = state["W_dec"].to(dev, torch.float32)   # (D_full, F)
    b_dec = state["b_dec"].to(dev, torch.float32)

    with torch.no_grad():
        x = torch.from_numpy(z_raw).to(dev, torch.float32)
        pre = torch.relu((x - b_dec) @ w_enc.T + b_enc)          # (n, F)
        vals, idx = torch.topk(pre, 64, dim=1)
        # Atom usage: total activation mass per flat atom over the slice.
        f_dim = w_enc.shape[0]
        usage = torch.zeros(f_dim, device=dev)
        usage.scatter_add_(0, idx.reshape(-1), vals.reshape(-1))
        top_atoms = torch.topk(usage, 4096).indices              # active vocabulary
        # Spherical k-means (fixed 12 Lloyd rounds, deterministic init by usage
        # rank) over the active atoms' unit decoder directions.
        dirs = w_dec[:, top_atoms].T                             # (4096, D_full)
        dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-12)
        k = args.k_atoms
        cent = dirs[:: max(1, dirs.shape[0] // k)][:k].clone()
        for _ in range(12):
            cos = dirs @ cent.T                                  # (4096, k)
            assign = cos.argmax(dim=1)
            for g in range(k):
                mask = assign == g
                if mask.any():
                    c = dirs[mask].mean(dim=0)
                    cent[g] = c / c.norm().clamp_min(1e-12)
        # Per-row warm assignment: mass each row puts on each atom-group.
        group_of = torch.full((f_dim,), -1, dtype=torch.long, device=dev)
        group_of[top_atoms] = assign
        row_group = torch.zeros((n, k), device=dev)
        flat_groups = group_of[idx]                              # (n, 64)
        for g in range(k):
            row_group[:, g] = (vals * (flat_groups == g)).sum(dim=1)
        # Normalize to a soft assignment; floor so no group is exactly dead.
        a_init = (row_group / row_group.sum(dim=1, keepdim=True).clamp_min(1e-9)).cpu().numpy()
        a_init = np.asarray(a_init, dtype=np.float64).clip(1e-4, None)
        a_init /= a_init.sum(axis=1, keepdims=True)
    print(f"[warm] a_init from checkpoint: group mass std {a_init.std():.4f}", flush=True)

    results = {"d_pca": args.d_pca, "K": k, "rows": n, "pca_capture": ev_frac}
    for label, kwargs in (("cold", {}), ("warm", {"a_init": a_init})):
        t0 = time.time()
        try:
            m = gamfit.sae_manifold_fit(
                z, K=k, d_atom=1, atom_topology="circle", top_k=3,
                n_iter=args.n_iter, random_state=0, **kwargs,
            )
            results[label] = {
                "status": "survived",
                "r2": float(m.reconstruction_r2),
                "wall_s": round(time.time() - t0, 1),
            }
        except Exception as exc:  # noqa: BLE001
            results[label] = {
                "status": type(exc).__name__,
                "error": str(exc)[:800],
                "wall_s": round(time.time() - t0, 1),
            }
        print(f"[warm] {label}: {results[label]}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print("[warm] DONE", flush=True)


if __name__ == "__main__":
    main()
