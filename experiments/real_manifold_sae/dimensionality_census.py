"""Dimensionality census: how multidimensional are LANGUAGE-model concepts?

The BSF paper (Goodfire, 2026-07) answers this for vision (DINOv3/SDXL):
block stable rank concentrates at 2-4 of a possible 16. This script produces
the language-side answer on real frontier residual-stream activations
(default: Kimi-K2.5 layer-40 datasci corpora) using the proven block-sparse
lane (`gamfit.block_sparse_dictionary_fit`, Grassmann frames ON — the same
featurizer family as their Grassmannian BSF).

Per config it records the full per-block stable-rank vector plus summary
statistics: mean/median stable rank, the multidimensional fraction
(stable rank > 1.5), and the 2-4 band fraction their vision result singles
out. Output JSON is append-per-config so a crash loses nothing.

Run through heimdall (mandatory on datasci nodes), e.g.:
  heimdall submit "PYTHONPATH=/models/wm_pylib <python> dimensionality_census.py \
      --category emotions --device cuda:0" \
    --type custom --gpus 1 --vram 6 --node node1 --name dim-census --estimated 90
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def load_slice(root: Path, category: str, n_rows: int, seed: int) -> np.ndarray:
    acts = np.load(root / category / "activations.npy", mmap_mode="r")
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(acts.shape[0], size=min(n_rows, acts.shape[0]), replace=False))
    return np.asarray(acts[idx], dtype=np.float32)


def gpu_pca(z: np.ndarray, d: int, device: str):
    import torch

    with torch.no_grad():
        t = torch.from_numpy(z).to(device=device, dtype=torch.float32)
        t -= t.mean(dim=0, keepdim=True)
        _, s, vh = torch.linalg.svd(t, full_matrices=False)
        proj = (t @ vh[:d].T).cpu().numpy().astype(np.float64)
        ev_frac = float((s[:d] ** 2).sum() / (s**2).sum())
    return proj, ev_frac


def census(fit) -> dict:
    sr = np.asarray(fit.block_stable_rank, dtype=np.float64)
    sr = sr[np.isfinite(sr)]
    hist_edges = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]
    hist, _ = np.histogram(sr, bins=hist_edges)
    return {
        "n_blocks_finite": int(sr.size),
        "stable_rank_mean": float(sr.mean()) if sr.size else None,
        "stable_rank_median": float(np.median(sr)) if sr.size else None,
        "stable_rank_p90": float(np.percentile(sr, 90)) if sr.size else None,
        "stable_rank_max": float(sr.max()) if sr.size else None,
        "frac_multidimensional": float((sr > 1.5).mean()) if sr.size else None,
        "frac_band_2_to_4": float(((sr >= 2.0) & (sr <= 4.0)).mean()) if sr.size else None,
        "hist_edges": hist_edges,
        "hist_counts": hist.tolist(),
        "stable_rank_per_block": [round(float(v), 3) for v in sr],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/models/k25_tokens"))
    ap.add_argument("--category", default="emotions")
    ap.add_argument("--n-rows", type=int, default=60000)
    ap.add_argument("--d-pca", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=Path("dimensionality_census.jsonl"))
    args = ap.parse_args()

    import gamfit

    print(f"[census] loading {args.n_rows} rows from {args.root}/{args.category}", flush=True)
    z_raw = load_slice(args.root, args.category, args.n_rows, args.seed)
    proj, ev_frac = gpu_pca(z_raw, args.d_pca, args.device)
    print(f"[census] slice {proj.shape} pca_ev={ev_frac:.3f}", flush=True)

    # Matched total units (n_blocks * block_size = 2048) across block sizes,
    # bracketing the paper's b=16 with our proven b=4 lane.
    configs = [
        dict(name="b4", n_blocks=512, block_size=4, block_topk=8),
        dict(name="b8", n_blocks=256, block_size=8, block_topk=4),
        dict(name="b16", n_blocks=128, block_size=16, block_topk=2),
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for cfg in configs:
        t0 = time.time()
        rec = {
            **cfg,
            "category": args.category,
            "n_rows": int(proj.shape[0]),
            "d_pca": args.d_pca,
            "pca_ev_fraction": round(ev_frac, 4),
        }
        try:
            fit = gamfit.block_sparse_dictionary_fit(
                proj,
                cfg["n_blocks"],
                block_size=cfg["block_size"],
                block_topk=cfg["block_topk"],
                max_epochs=30,
            )
            rec["status"] = "ok"
            rec["census"] = census(fit)
            ev = getattr(fit, "explained_variance", None)
            rec["explained_variance"] = None if ev is None else float(ev)
        except Exception as exc:  # noqa: BLE001 - exception class is the datum
            rec["status"] = type(exc).__name__
            rec["error"] = str(exc)[:1500]
        rec["wall_seconds"] = round(time.time() - t0, 1)
        with args.out.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[census] {cfg['name']}: {rec['status']} t={rec['wall_seconds']}s", flush=True)
    print("[census] DONE", flush=True)


if __name__ == "__main__":
    main()
