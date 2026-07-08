#!/usr/bin/env python3
"""#1893 acceptance measurement — K=2000 curved-manifold SAE vs external TopK baseline.

The #1893 bar (per its audit comment on HEAD): a real-activation
`sae_manifold_fit(K=2000, assignment='topk')` fit must (1) COMPLETE without
co-collapse / silent-linear reroute, (2) reach held-out reconstruction EV >= an
external `dictionary_learning`-style TopK baseline, and (3) do so fast enough to
iterate (the #1995 throughput leg — closed in code by a215a7345 "Optimize sparse
SAE Schur block GEMM", but its K=2000 effect has never been MEASURED; this script
produces that number).

Uses gamfit's PUBLIC API only: `gamfit.sae_manifold_fit(..., assignment='topk',
top_k=...)` for the fit and `model.reconstruct(X_test)` for the out-of-sample
reconstruction. Prints ONE verdict line.

Leg (2) is same-data match-or-beat: with `--external-live` (the production mode,
set by the sbatch) a TopK SAE is trained on the IDENTICAL train/test split at the
IDENTICAL K and top_k, so no cross-dataset literature number is borrowed. The fixed
`--external-ev` constant (e.g. 0.878 @ pythia-70m p=512) is only a fallback for an
internet-capable pythia harvest run.

This script does not lower the bar: it reports our EV, the external baseline EV,
and the wall-time, and declares PASS only if all three legs hold.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np


# --------------------------------------------------------------------------- #
# Held-out explained variance (the SAE reconstruction quality metric #1893 uses)
# --------------------------------------------------------------------------- #
def held_out_ev(x_test: np.ndarray, recon: np.ndarray, mean_train: np.ndarray) -> float:
    """EV = 1 - ||X_test - recon||_F^2 / ||X_test - mean_train||_F^2.

    Baseline is the TRAIN mean (a fit must beat predicting the mean), the standard
    SAE "explained variance" / (1 - FVU) convention. Both SAEs are scored with the
    identical formula on the identical held-out split.
    """
    ssr = float(np.sum((x_test - recon) ** 2))
    sst = float(np.sum((x_test - mean_train[None, :]) ** 2))
    return 1.0 - ssr / max(sst, 1e-300)


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
def load_npz(path: str, key: str) -> np.ndarray:
    z = np.load(path)
    if key not in z.files:
        raise SystemExit(f"npz {path} has no key '{key}'; keys = {list(z.files)}")
    return np.ascontiguousarray(z[key], dtype=np.float32)


def harvest_pythia70m(n_tokens: int, layer: int) -> np.ndarray:
    """pythia-70m residual activations (d_model = p = 512), the regime of the
    external baseline 0.878. Needs internet + transformers/datasets/torch — the
    sbatch does NOT use this by default (compute-node internet is not guaranteed);
    kept as an explicit flag for an internet-capable run."""
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "EleutherAI/pythia-70m"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, output_hidden_states=True).eval()
    ds = load_dataset("NeelNanda/pile-10k", split="train", streaming=True)
    acts, got = [], 0
    with torch.no_grad():
        for ex in ds:
            ids = tok(ex["text"], return_tensors="pt", truncation=True, max_length=512).input_ids
            if ids.shape[1] < 8:
                continue
            acts.append(model(ids).hidden_states[layer][0].to(torch.float32).numpy())
            got += acts[-1].shape[0]
            if got >= n_tokens:
                break
    return np.ascontiguousarray(np.concatenate(acts, 0)[:n_tokens], dtype=np.float32)


def load_digits() -> np.ndarray:
    from sklearn.datasets import load_digits as _ld

    return np.ascontiguousarray(_ld().data, dtype=np.float32)


def load_chunk_dir(chunk_dir: str, max_rows: int, seed: int) -> np.ndarray:
    """Load a directory of `chunk_*.npy` activation shards (e.g. the creditscope
    Qwen3.5-35B-A3B L30 residual set: 8 float16 shards, ~360k tokens total).

    The shards are memory-mapped (never fully materialized); a deterministic
    `max_rows`-row subsample is gathered per shard by fancy-index and only THEN
    cast to float32, so the peak footprint is `max_rows × p` f32, not the full
    corpus. `p` is read from the shard shape (it is the model d_model — never
    hardcoded)."""
    import glob
    import os

    files = sorted(glob.glob(os.path.join(chunk_dir, "chunk_*.npy")))
    if not files:
        raise SystemExit(f"no chunk_*.npy shards in {chunk_dir}")
    mms = [np.load(f, mmap_mode="r") for f in files]
    p = int(mms[0].shape[1])
    for m in mms:
        if int(m.shape[1]) != p:
            raise SystemExit(f"inconsistent p across shards: {p} vs {m.shape[1]}")
    counts = [int(m.shape[0]) for m in mms]
    offsets = np.cumsum([0] + counts)
    total = int(offsets[-1])
    rng = np.random.default_rng(seed)
    take = min(max_rows, total)
    idx = np.sort(rng.choice(total, take, replace=False))
    parts = []
    for c, m in enumerate(mms):
        sel = idx[(idx >= offsets[c]) & (idx < offsets[c + 1])] - offsets[c]
        if sel.size:
            parts.append(np.asarray(m[sel], dtype=np.float32))
    out = np.concatenate(parts, axis=0)
    rng.shuffle(out)  # de-correlate the split from shard/document order
    return np.ascontiguousarray(out, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Live external TopK SAE baseline (Gao et al. TopK) on the SAME split.
# --------------------------------------------------------------------------- #
def external_topk_ev_live(
    x_train, x_test, mean_train, k_dict, top_k, steps, seed
) -> float:
    import torch

    torch.manual_seed(seed)
    p = x_train.shape[1]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    xtr = torch.from_numpy(x_train).to(dev)
    xte = torch.from_numpy(x_test).to(dev)
    b_dec = xtr.mean(0)
    W_enc = torch.nn.Parameter(torch.randn(k_dict, p, device=dev) * (1.0 / p ** 0.5))
    W_dec = torch.nn.Parameter(W_enc.detach().clone())
    opt = torch.optim.Adam([W_enc, W_dec], lr=1e-3)

    def encode_decode(x):
        pre = (x - b_dec) @ W_enc.t()
        topv, topi = pre.topk(top_k, dim=1)
        z = torch.zeros_like(pre).scatter_(1, topi, torch.relu(topv))
        return z @ W_dec + b_dec

    n, bs = xtr.shape[0], min(4096, x_train.shape[0])
    for _ in range(steps):
        idx = torch.randint(0, n, (bs,), device=dev)
        xb = xtr[idx]
        loss = ((encode_decode(xb) - xb) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        recon_te = encode_decode(xte).cpu().numpy()
    return held_out_ev(x_test, recon_te, mean_train)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-mode", choices=["chunkdir", "npz", "harvest_pythia70m", "digits"],
                    default="chunkdir")
    ap.add_argument("--chunk-dir", default=None, help="dir of chunk_*.npy activation shards")
    ap.add_argument("--max-rows", type=int, default=120_000,
                    help="deterministic row budget (train+held-out) subsampled from the corpus")
    ap.add_argument("--npz", default=None)
    ap.add_argument("--npz-key", default="acts")
    ap.add_argument("--n-tokens", type=int, default=200_000, help="pythia-harvest token cap")
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--K", type=int, default=2000)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--d-atom", type=int, default=1)
    ap.add_argument("--atom-topology", default="circle")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--external-ev", type=float, default=0.878,
                    help="fixed external TopK EV fallback when --external-live is off")
    ap.add_argument("--external-live", action="store_true",
                    help="train a TopK SAE baseline on the SAME split (same-data match-or-beat)")
    ap.add_argument("--external-steps", type=int, default=20_000)
    ap.add_argument("--max-fit-seconds", type=float, default=3600.0)
    args = ap.parse_args()

    import gamfit

    rng = np.random.default_rng(args.seed)
    if args.data_mode == "chunkdir":
        if not args.chunk_dir:
            raise SystemExit("--data-mode chunkdir requires --chunk-dir DIR")
        X = load_chunk_dir(args.chunk_dir, args.max_rows, args.seed)
    elif args.data_mode == "npz":
        if not args.npz:
            raise SystemExit("--data-mode npz requires --npz PATH")
        X = load_npz(args.npz, args.npz_key)
    elif args.data_mode == "harvest_pythia70m":
        X = harvest_pythia70m(args.n_tokens, args.layer)
    else:
        X = load_digits()

    # Cap any mode to the row budget (chunkdir already subsampled to --max-rows).
    if X.shape[0] > args.max_rows:
        X = np.ascontiguousarray(X[rng.choice(X.shape[0], args.max_rows, replace=False)])
    n, p = X.shape

    perm = rng.permutation(n)
    n_test = max(1, int(round(args.test_frac * n)))
    te_idx, tr_idx = perm[:n_test], perm[n_test:]
    X_tr, X_te = np.ascontiguousarray(X[tr_idx]), np.ascontiguousarray(X[te_idx])
    mean_tr = X_tr.mean(0)
    print(f"[#1893] dataset={args.data_mode} N={n} p={p} K={args.K} top_k={args.top_k} "
          f"train={X_tr.shape[0]} test={X_te.shape[0]}", flush=True)

    # ---- OUR curved-manifold TopK fit (the K>P lane the #1893 bar names) ----
    t0 = time.time()
    completed, err = True, ""
    try:
        model = gamfit.sae_manifold_fit(
            X_tr, K=args.K, d_atom=args.d_atom, atom_topology=args.atom_topology,
            assignment="topk", top_k=args.top_k, random_state=args.seed,
        )
        recon_te = np.asarray(model.reconstruct(X_te), dtype=np.float64)
        ev_ours = held_out_ev(X_te.astype(np.float64), recon_te, mean_tr.astype(np.float64))
    except Exception as exc:  # noqa: BLE001 — a GamError / co-collapse refusal is a FAIL, not a crash
        completed, err, ev_ours = False, f"{type(exc).__name__}: {exc}", float("nan")
    fit_seconds = time.time() - t0

    ev_ext_live = float("nan")
    if args.external_live and completed:
        ev_ext_live = external_topk_ev_live(
            X_tr, X_te, mean_tr, args.K, args.top_k, args.external_steps, args.seed)

    baseline = ev_ext_live if (args.external_live and np.isfinite(ev_ext_live)) else args.external_ev
    ev_pass = completed and np.isfinite(ev_ours) and ev_ours >= baseline
    time_pass = completed and fit_seconds <= args.max_fit_seconds
    verdict = "PASS" if (ev_pass and time_pass) else "FAIL"

    print("[#1893] RESULT " + json.dumps({
        "issue": 1893, "dataset": args.data_mode, "N": n, "p": p, "K": args.K,
        "top_k": args.top_k, "completed": completed, "error": err, "ev_ours": ev_ours,
        "ev_external_baseline": baseline, "ev_external_live": ev_ext_live,
        "fit_seconds": fit_seconds, "max_fit_seconds": args.max_fit_seconds,
        "ev_pass": ev_pass, "time_pass": time_pass, "verdict": verdict,
    }), flush=True)
    print(f"[#1893] VERDICT={verdict} ev_ours={ev_ours:.4f} ev_external={baseline:.4f} "
          f"fit_seconds={fit_seconds:.1f} completed={completed}"
          + ("" if completed else f" error='{err}'"), flush=True)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
