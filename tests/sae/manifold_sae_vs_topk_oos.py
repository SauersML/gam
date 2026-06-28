#!/usr/bin/env python3
"""Manifold SAE (learned encoder) vs a real flat TopK SAE — head-to-head OOS.

Both are TRAINED SAEs with a learned encoder; only the decoder geometry differs:
  * FLAT  — a traditional TopK SAE (linear dict, each atom a direction). Baseline
    from the `dictionary_learning` library (Gao et al.), see
    `real_topk_sae_baseline.py`.
  * MANIFOLD — the project's own `gamfit.torch.ManifoldSAE`: a shared learned
    encoder maps x -> per-atom on-manifold coord theta_i(x) + amplitude, with a
    softmax-TopK gate, and each atom is a CURVED 1-D Fourier curve in R^p (decoder
    block B_i) evaluated through the analytic Rust-backed basis (differentiable;
    `loss.backward()` routes the VJP back to Rust). Trained here with torch Adam on
    held-out reconstruction — a normal SGD-trained SAE, just with curved atoms.

Same 60/40 contiguous split of real OLMo l18. Metric: out-of-sample variance
explained, 1 - sum||x_hat-x||^2 / sum||x - mean_train||^2 (mean-agnostic, so the
flat SAE's b_dec and the manifold atoms are compared on equal footing).

Setup (uses the repo's prebuilt gamfit/_rust.abi3.so):
    /Users/user/gam/.venv/bin/python tests/sae/manifold_sae_vs_topk_oos.py
"""
import os, glob, time
import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)


def load_olmo_l18():
    cands = ["/Users/user/gam/tests/data/olmo_l18_pca64_635.npy"] + glob.glob(
        "**/olmo_l18_pca64_635.npy", recursive=True
    )
    z = np.load(next(p for p in cands if os.path.exists(p))).astype(np.float64)
    n = z.shape[0]
    n_tr = (n * 6) // 10
    return torch.tensor(z[:n_tr]), torch.tensor(z[n_tr:])


def var_explained(x_hat, x, mean_tr):
    num = ((x_hat - x) ** 2).sum().item()
    den = ((x - mean_tr) ** 2).sum().item()
    return 1 - num / den


def train_manifold_sae(z_tr, z_te, n_atoms, target_k, rank=1, basis_order=3,
                       steps=800, bs=128, lr=2e-3):
    import gamfit.torch as gt
    cfg = gt.ManifoldSAEConfig(
        input_dim=z_tr.shape[1], n_atoms=n_atoms, intrinsic_rank=rank,
        atom_manifold="circle", atom_basis="fourier", basis_order=basis_order,
        sparsity={"kind": "softmax_topk", "target_k": target_k},
    )
    sae = gt.ManifoldSAE(cfg).double()
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    n = z_tr.shape[0]
    mean_tr = z_tr.mean(0)
    for s in range(steps):
        x = z_tr[torch.randint(0, n, (bs,))]
        out = sae(x)
        loss = ((out.x_hat - x) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    sae.eval()
    with torch.no_grad():
        ev_in = var_explained(sae(z_tr).x_hat, z_tr, mean_tr)
        ev_oos = var_explained(sae(z_te).x_hat, z_te, mean_tr)
    return ev_in, ev_oos


def main():
    z_tr, z_te = load_olmo_l18()
    print(f"OLMo l18: train {z_tr.shape[0]} test {z_te.shape[0]} dim {z_tr.shape[1]}")
    print("\nMANIFOLD SAE (gamfit.torch.ManifoldSAE, learned encoder, softmax-TopK, "
          "1-D Fourier curved atoms), OOS variance explained:")
    print(f"{'n_atoms':>8} {'k':>3} {'order':>6} {'in_VE':>8} {'OOS_VE':>8}")
    for (D, k, h) in [(32, 2, 3), (64, 2, 3), (64, 4, 3), (128, 4, 3)]:
        t0 = time.time()
        ein, eoos = train_manifold_sae(z_tr, z_te, D, k, basis_order=h)
        print(f"{D:>8} {k:>3} {h:>6} {ein:>8.4f} {eoos:>8.4f}   ({time.time()-t0:.0f}s)")
    print("\nFLAT TopK SAE (dictionary_learning, real_topk_sae_baseline.py) OOS var-expl:")
    print("  dict=32 k=2: 0.206   dict=64 k=2: 0.231   dict=64 k=4: 0.262   dict=128 k=4: 0.271")


if __name__ == "__main__":
    main()
