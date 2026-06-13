"""Probe the statistical properties of each checkpoint's L25 activations.

Checks: norm statistics, PCA variance explained, condition number.
Helps diagnose why stage1-step0 K=1 circle fails.

Usage (MSI):
    source /projects/standard/hsiehph/sauer354/gamfit-sweep-venv/bin/activate
    cd /tmp && python3 /projects/standard/hsiehph/sauer354/gam/examples/olmo_data_probe.py
"""

import json
from pathlib import Path

import numpy as np

CHECKPOINTS = [
    ("stage1-step0",       "SFT-init"),
    ("base",               "base"),
    ("stage3-step11921",   "SFT-end"),
    ("instruct",           "instruct"),
    ("step_2300",          "RLVR"),
]

DATA_DIR = Path("/projects/standard/hsiehph/sauer354/olmo_data")


def main():
    print(f"{'ckpt':<22} {'n':>5} {'D':>5} {'norm_mean':>10} {'norm_std':>9}"
          f" {'pca10_ev%':>10} {'pca32_ev%':>10} {'cond10':>8}")
    print("-" * 95)

    for ckpt_key, ckpt_label in CHECKPOINTS:
        ckpt_dir = DATA_DIR / ckpt_key
        if not ckpt_dir.exists():
            print(f"{ckpt_label:<22}  MISSING")
            continue

        acts = np.load(ckpt_dir / "activations.npy", mmap_mode="r")
        l25 = np.array(acts[:, 25, :], dtype=np.float32)
        n, D = l25.shape

        norms = np.linalg.norm(l25, axis=1)
        norm_mean = float(norms.mean())
        norm_std = float(norms.std())

        mu = l25.mean(0)
        Xc = (l25 - mu).astype(np.float64)
        _, S, _ = np.linalg.svd(Xc, full_matrices=False)
        total_var = float((S ** 2).sum())
        pca10_ev = float((S[:10] ** 2).sum() / total_var * 100)
        pca32_ev = float((S[:32] ** 2).sum() / total_var * 100)
        cond10 = float(S[0] / (S[9] + 1e-12))

        print(f"{ckpt_label:<22} {n:>5} {D:>5} {norm_mean:>10.2f} {norm_std:>9.2f}"
              f" {pca10_ev:>10.1f} {pca32_ev:>10.1f} {cond10:>8.1f}")

        # Also print top-10 singular values
        print(f"  SV top-10: {' '.join(f'{s:.1f}' for s in S[:10])}")

    print("-" * 95)


if __name__ == "__main__":
    main()
