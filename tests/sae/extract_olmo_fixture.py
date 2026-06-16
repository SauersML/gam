#!/usr/bin/env python3
"""Extract a SMALL, GitHub-sized, REAL OLMo-activation fixture for the hard e2e
test (tests/sae/olmo_real_fixture_e2e.py). Run ONCE on MSI where the real
activations are staged; commit the two output files under tests/data/.

It projects the REAL OLMo L25 residual-stream activations (635×5120) onto their
top-`--pcs` principal components (default 64) → 635×64 f32 ≈ 159 KB, and writes
the per-prompt kind/role labels. This is REAL LLM data (a deterministic linear
projection of the actual activations), not synthetic — the PCA basis is computed
from and applied to the genuine cloud, so the fixture preserves the activation
geometry the manifold-SAE must recover.

USAGE (on MSI, login node ok — tiny):
  python tests/sae/extract_olmo_fixture.py \
      --data /projects/standard/hsiehph/sauer354/olmo_data/instruct/<rev> \
      --pcs 64 --out-dir tests/data
Then: git add tests/data/olmo_l25_pca64.npy tests/data/olmo_l25_labels.csv && commit.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="OLMo rev dir (activations.npy + prompts.jsonl)")
    ap.add_argument("--layer", type=int, default=25)
    ap.add_argument("--pcs", type=int, default=64)
    ap.add_argument("--out-dir", default="tests/data")
    args = ap.parse_args()

    data = Path(args.data)
    acts = np.load(data / "activations.npy")
    z = (acts[:, args.layer, :] if acts.ndim == 3 else acts).astype(np.float64)
    z = z - z.mean(axis=0, keepdims=True)
    n, p = z.shape

    # Deterministic top-PC projection (SVD; sign-canonicalized so the fixture is
    # byte-reproducible regardless of LAPACK sign conventions).
    u, s, vt = np.linalg.svd(z, full_matrices=False)
    k = min(args.pcs, vt.shape[0])
    basis = vt[:k]
    # Canonical sign: make the largest-magnitude entry of each PC positive.
    for i in range(k):
        j = int(np.argmax(np.abs(basis[i])))
        if basis[i, j] < 0:
            basis[i] *= -1.0
    coords = (z @ basis.T).astype(np.float32)  # 635 x k, real activation geometry

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / f"olmo_l{args.layer}_pca{k}.npy"
    np.save(npy_path, coords)

    # Labels.
    kinds, roles = [], []
    pj = data / "prompts.jsonl"
    if pj.exists():
        with pj.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    o = {}
                kinds.append(str(o.get("kind", "?")))
                roles.append(str(o.get("role", "?")))
    csv_path = out_dir / f"olmo_l{args.layer}_labels.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "kind", "role"])
        for i in range(n):
            w.writerow([i, kinds[i] if i < len(kinds) else "?", roles[i] if i < len(roles) else "?"])

    size_kb = npy_path.stat().st_size / 1024.0
    print(f"wrote {npy_path} ({coords.shape}, {size_kb:.1f} KB) and {csv_path}")
    print(f"  source: REAL OLMo L{args.layer} activations {n}x{p} → top-{k} PCs")
    print(f"  explained variance retained: {float(s[:k].sum()**2 / (s**2).sum()):.4f} (sum-sq frac {float((s[:k]**2).sum()/(s**2).sum()):.4f})")
    if size_kb > 900:
        print(f"  WARNING: {size_kb:.0f} KB — subsample rows or lower --pcs to stay under 1 MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
