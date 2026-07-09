#!/usr/bin/env python3
"""Real-LLM SAE arena: manifold SAE vs flat TopK on REAL residual-stream
activations, scored ONLY by quantities that cannot be Goodharted by DGP
design (there is no DGP — the data is a real model's):

* **Held-out reconstruction R²** — rows the SAE never saw, at matched code
  sparsity (report L0 next to R²; a win at higher L0 is not a win).
* **Sparse-probing accuracy** — a ridge one-vs-rest probe from the SAE codes
  to REAL prompt labels (e.g. OLMo self/qualia `kind` [38-way] / `role`),
  classifier trained on the fit split's codes, evaluated on held-out rows.
  Skylines: the same probe on the raw (PCA'd) activations.

Both arms run at magic defaults — no per-dataset tuning, no eval-split
feedback. The synthetic zoo (bsf_manifold_zoo.py) remains the ground-truth
DIAGNOSTIC; this file is the benchmark.

Input layouts supported:
* ``--acts X.npy`` of shape (N, D) float — rows align with ``--labels``
  prompts.jsonl (one JSON per row).
* ``--acts activations.npy`` of shape (N, L, D) with ``--layer l`` to slice.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bsf_manifold_zoo import _fit_flat_topk, _fit_ours_rust


def _load_acts(path: str, layer: int | None) -> np.ndarray:
    acts = np.load(path, mmap_mode="r")
    if acts.ndim == 3:
        if layer is None:
            raise SystemExit("--layer is required for (N, L, D) activation files")
        acts = acts[:, layer, :]
    return np.ascontiguousarray(np.asarray(acts, dtype=np.float64))


def _load_labels(path: str | None, n: int) -> dict[str, np.ndarray]:
    if path is None:
        return {}
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len(rows) != n:
        raise SystemExit(f"labels rows {len(rows)} != activation rows {n}")
    out: dict[str, np.ndarray] = {}
    for field in ("kind", "role", "entity", "side"):
        values = [str(r.get(field, "")) for r in rows]
        if len(set(values)) > 1:
            out[field] = np.asarray(values)
    return out


def _pca_fit(x: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """(mean, components) of a PCA fit — call on the FIT split only."""
    mu = x.mean(axis=0)
    _u, _s, vt = np.linalg.svd(x - mu[None, :], full_matrices=False)
    return mu, vt[:dim]


def _ridge_probe(train_c: np.ndarray, train_y: np.ndarray,
                 test_c: np.ndarray, test_y: np.ndarray, lam: float = 1.0) -> float:
    """One-vs-rest ridge classifier accuracy (dependency-free)."""
    classes, train_idx = np.unique(train_y, return_inverse=True)
    onehot = np.eye(len(classes))[train_idx]
    a = train_c.T @ train_c + lam * np.eye(train_c.shape[1])
    w = np.linalg.solve(a, train_c.T @ onehot)
    pred = classes[np.argmax(test_c @ w, axis=1)]
    return float(np.mean(pred == test_y))


def _r2(x: np.ndarray, recon: np.ndarray) -> float:
    ss_res = float(np.sum((x - recon) ** 2))
    ss_tot = float(np.sum((x - x.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acts", required=True)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--labels", default=None, help="prompts.jsonl aligned with rows")
    parser.add_argument("--pca", type=int, default=64)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--atoms", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--flat-width", type=int, default=None,
                        help="Flat dictionary width (default: 4 * atoms).")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rust-iters", type=int, default=24)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--featurizers", default="ours_rust,flat")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="real_llm_arena.jsonl")
    args = parser.parse_args()

    acts = _load_acts(args.acts, args.layer)
    n = acts.shape[0]
    labels = _load_labels(args.labels, n)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n)
    n_hold = max(1, int(round(args.holdout_fraction * n)))
    hold, fit = order[:n_hold], order[n_hold:]

    # PCA + standardization from the FIT split only (no eval leakage).
    mu, comps = _pca_fit(acts[fit], args.pca)
    proj = (acts - mu[None, :]) @ comps.T
    scale = proj[fit].std(axis=0).clip(min=1e-12)
    proj = proj / scale[None, :]
    fit_x, all_x = proj[fit], proj

    flat_width = args.flat_width or 4 * args.atoms
    out_path = Path(args.out)
    header = {
        "record": "config", "acts": args.acts, "layer": args.layer,
        "n_rows": n, "n_fit": int(fit.size), "n_holdout": int(hold.size),
        "pca": args.pca, "atoms": args.atoms, "top_k": args.top_k,
        "flat_width": flat_width, "seed": args.seed,
        "label_fields": sorted(labels),
    }
    with out_path.open("a") as fh:
        fh.write(json.dumps(header) + "\n")

    # Raw-representation probe skylines (upper references, same probe).
    skylines = {
        field: _ridge_probe(proj[fit], y[fit], proj[hold], y[hold])
        for field, y in labels.items()
    }
    print(f"[skyline pca{args.pca}] " + " ".join(
        f"{f}={a:.3f}" for f, a in sorted(skylines.items())))

    for which in [w.strip() for w in args.featurizers.split(",") if w.strip()]:
        t0 = time.perf_counter()
        if which == "flat":
            fitted = _fit_flat_topk(
                fit_x, all_x, width=flat_width, k=args.top_k, steps=args.steps,
                batch_size=args.batch_size, lr=args.lr, seed=args.seed,
                device=args.device,
            )
        elif which == "ours_rust":
            fitted = _fit_ours_rust(
                fit_x, all_x, atoms=args.atoms, top_k=args.top_k,
                n_iter=args.rust_iters, seed=args.seed,
            )
        else:
            raise SystemExit(f"unknown featurizer {which!r}")
        codes = np.abs(np.asarray(fitted.gate, dtype=np.float64))
        l0 = float(np.mean(np.sum(codes > 1e-8, axis=1)))
        heldout_r2 = _r2(all_x[hold], fitted.recon[hold])
        probes = {
            field: _ridge_probe(codes[fit], y[fit], codes[hold], y[hold])
            for field, y in labels.items()
        }
        record = {
            "record": "result", "featurizer": fitted.name, "seed": args.seed,
            "heldout_recon_r2": heldout_r2, "mean_l0": l0,
            "probe_acc": probes, "probe_skyline": skylines,
            "fit_seconds": fitted.fit_seconds,
            "wall_seconds": time.perf_counter() - t0,
        }
        with out_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        probe_str = " ".join(f"{f}={a:.3f}" for f, a in sorted(probes.items()))
        print(f"[{fitted.name}] heldout R2 {heldout_r2:.4f} L0 {l0:.2f} "
              f"probe {probe_str} (fit {fitted.fit_seconds:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
