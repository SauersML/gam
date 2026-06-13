"""Quick probe to find working (pca_dim, K) combinations on each checkpoint.

Tries pca_dim in [10, 16, 32] × K in [1, 2, 3, 5] with a short n_iter=20.
Outputs a table so we can pick the right parameters for the full run.

Usage (MSI):
    source /projects/standard/hsiehph/sauer354/gamfit-sweep-venv/bin/activate
    cd /tmp && python3 /projects/standard/hsiehph/sauer354/gam/examples/olmo_pca_probe.py \\
        --data_dir /projects/standard/hsiehph/sauer354/olmo_data \\
        2>&1 | grep -v SAE-AUDIT | grep -v "process-monitor" | grep -v CUDA
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

CHECKPOINTS = [
    ("stage1-step0",       "SFT-init"),
    ("base",               "base"),
    ("stage3-step11921",   "SFT-end"),
    ("instruct",           "instruct"),
    ("step_2300",          "RLVR"),
]

PCA_DIMS = [10, 16]
K_VALUES = [1, 2, 3]


def pca_project(X, n_components):
    mu = X.mean(axis=0)
    Xc = (X - mu).astype(np.float64)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Vt = Vt[:n_components]
    Z = (Xc @ Vt.T) / (S[:n_components] + 1e-12)
    return Z


def try_fit(Z, K, n_iter=20, seed=42):
    import gamfit
    t0 = time.time()
    try:
        fit = gamfit.sae_manifold_fit(
            X=Z, K=K, d_atom=2, atom_topology="circle",
            n_iter=n_iter, random_state=seed,
            assignment="ibp_map", smoothness_weight=1.0, sparsity_weight=0.5,
        )
        fitted = np.asarray(fit.fitted)
        total_var = float(((Z - Z.mean(0)) ** 2).sum())
        ev = float(1.0 - ((Z - fitted) ** 2).sum() / total_var)
        return "OK", ev, time.time() - t0
    except Exception as exc:
        return f"FAIL:{type(exc).__name__}", float("nan"), time.time() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/projects/standard/hsiehph/sauer354/olmo_data")
    parser.add_argument("--n_iter",   type=int, default=20)
    args = parser.parse_args()
    data = Path(args.data_dir)

    print(f"{'ckpt':<22} {'pca':>4} {'K':>3}  {'status':<30} {'EV':>8} {'secs':>6}")
    print("-" * 80)

    for ckpt_key, ckpt_label in CHECKPOINTS:
        ckpt_dir = data / ckpt_key
        if not ckpt_dir.exists():
            print(f"{ckpt_label:<22}  MISSING")
            continue
        acts = np.load(ckpt_dir / "activations.npy", mmap_mode="r")
        l25 = np.array(acts[:, 25, :], dtype=np.float32)

        for pca_dim in PCA_DIMS:
            Z = pca_project(l25, pca_dim)
            for K in K_VALUES:
                status, ev, secs = try_fit(Z, K, n_iter=args.n_iter)
                ev_str = f"{ev:.4f}" if ev == ev else "nan"
                print(f"{ckpt_label:<22} {pca_dim:>4} {K:>3}  {status:<30} {ev_str:>8} {secs:>6.0f}s",
                      flush=True)

    print("-" * 80)
    print("Probe done.")


if __name__ == "__main__":
    main()
