#!/usr/bin/env python3
"""Real traditional TopK SAE baseline for the manifold-SAE comparison.

The manifold SAE's reconstruction quality must be compared against an ACTUAL
sparse autoencoder, not PCA. PCA is dense and global — it uses every input
dimension to form the code and is the optimal *linear* autoencoder — so it is
NOT a sparse autoencoder and is not a valid SAE baseline (using it made the
manifold look ~2x better than it is).

This trains a standard TopK SAE (Gao et al. 2024 recipe) with the
`dictionary_learning` library (Marks et al.) on the SAME 60/40 contiguous split
of the real OLMo l18 activations the Rust manifold tests use, and reports
held-out (out-of-sample) reconstruction explained variance at matched sparsity.

Setup (the system Python is too new for the ML wheels):
    /opt/homebrew/bin/python3.13 -m venv --system-site-packages saevenv   # torch 2.11 already present
    saevenv/bin/pip install dictionary_learning
    saevenv/bin/python tests/sae/real_topk_sae_baseline.py

Reference result (held-out OLMo l18, raw EV = 1 - ||x̂-x||²/||x||²):
    dict  k   in_EV  OOS_EV   live/dict
      32  1   0.507  0.169    28/32
      32  2   0.581  0.217    32/32
      64  2   0.696  0.242    62/64
      64  4   0.772  0.272    62/64
     128  4   0.878  0.281   125/128
     128  8   0.927  0.277   127/128
     256  8   0.984  0.250   243/256
     256 16   0.995  0.332   254/256
  => a single CURVED manifold d=2 atom reaches OOS ≈ 0.22 (measured in Rust),
     COMPETITIVE with a real TopK SAE at 2 active latents — one curved atom is in
     the ballpark of a small flat dictionary, not dramatically better.
"""
import os, glob
import numpy as np
import torch
from dictionary_learning.trainers.top_k import TopKTrainer

torch.manual_seed(0)
np.random.seed(0)


def load_olmo_l18():
    cands = (
        glob.glob("**/tests/data/olmo_l18_pca64_635.npy", recursive=True)
        + ["tests/data/olmo_l18_pca64_635.npy"]
    )
    path = next(p for p in cands if os.path.exists(p))
    z = np.load(path).astype(np.float32)
    n = z.shape[0]
    n_tr = (n * 6) // 10  # 60/40 contiguous (leakage-safe for autocorrelated rows)
    return torch.tensor(z[:n_tr]), torch.tensor(z[n_tr:]), path


def explained_variance(ae, X):
    ae.eval()
    with torch.no_grad():
        xh = ae(X)
    num = ((xh - X) ** 2).sum().item()
    return 1 - num / (X ** 2).sum().item()  # raw EV; b_dec captures the mean


def train_topk(z_tr, z_te, dict_size, k, steps=3000, bs=128):
    p = z_tr.shape[1]
    tr = TopKTrainer(steps=steps, activation_dim=p, dict_size=dict_size, k=k,
                     layer=0, lm_name="olmo_l18", lr=None, device="cpu", seed=0)
    n = z_tr.shape[0]
    for step in range(steps):
        tr.update(step, z_tr[torch.randint(0, n, (bs,))])
    ae = tr.ae
    with torch.no_grad():
        live = (ae.encode(z_te).abs().sum(0) > 0).sum().item()
    return explained_variance(ae, z_tr), explained_variance(ae, z_te), live


def main():
    z_tr, z_te, path = load_olmo_l18()
    print(f"data {path}  train {z_tr.shape[0]}  test {z_te.shape[0]}  dim {z_tr.shape[1]}")
    print(f"{'dict':>5} {'k':>3} {'in_EV':>8} {'OOS_EV':>8} {'live/dict':>10}")
    for dict_size, k in [(32, 1), (32, 2), (64, 2), (64, 4),
                         (128, 4), (128, 8), (256, 8), (256, 16)]:
        ein, eoos, live = train_topk(z_tr, z_te, dict_size, k)
        print(f"{dict_size:>5} {k:>3} {ein:>8.4f} {eoos:>8.4f} {live:>5}/{dict_size}")
    print("\nREF curved manifold d=2 single atom (Rust): OOS_EV ~ 0.22")
    print("PCA is NOT an SAE (dense, not sparse) — not a valid baseline.")


if __name__ == "__main__":
    main()
