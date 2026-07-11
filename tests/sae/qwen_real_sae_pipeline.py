#!/usr/bin/env python3
"""#977 honest Stage-0(SGD SAE) -> Stage-1(gam adjudicates) pipeline on REAL Qwen acts.

This is the un-forced division of labor:

  Stage 0 (PyTorch, this file -> torch_sgd_sae.py):
      Train a REAL overcomplete L1/JumpReLU SGD sparse autoencoder on ~400k real
      Qwen3.5-35B-A3B layer-30 residual-post token activations. This is the wide
      monosemantic dictionary gam does NOT provide. Report EV / L0 / dead-frac.

  Stage 1a (feature geometry, this file):
      For the top-active features, build the feature co-activation graph (which
      features fire together across tokens) and cluster it into candidate groups
      (the #977 latent_seed co-activation idea). Each group is a bundle of
      features whose joint code lives in a low-D subspace.

  Stage 1b (gam adjudicates, gamfit.adjudicate_atom_shape):
      For each group, project the group's per-token SAE codes to a 2-D intrinsic
      coordinate (PCA of the group's code submatrix) and hand it to gam's
      cross-class adjudicator. gam auto-decides circle vs euclidean vs
      k-cluster-mixture on HELD-OUT predictive loglik. NO topology is forced.

  Honest report:
      Count groups that adjudicate as genuinely CURVED (circle beats the cluster
      null, positive circle margin) vs cluster/euclidean. A mostly-cluster result
      is a REAL finding — the wager partially failing is honest, and reported as
      such.

The activations are mmap'd; the SAE minibatches on GPU. Adjudication is CPU
(gam FFI). Everything is deterministic from --seed.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Data loading: mmap the f16 chunk .npy files, optionally subsample tokens.
# ---------------------------------------------------------------------------

def load_activations(act_dir: str, max_tokens: int | None, seed: int, log=print):
    files = sorted(glob.glob(os.path.join(act_dir, "*.npy")))
    if not files:
        raise FileNotFoundError(f"no .npy activation chunks under {act_dir}")
    log(f"found {len(files)} activation chunks under {act_dir}")
    mmaps = []
    total = 0
    d = None
    for f in files:
        m = np.load(f, mmap_mode="r")
        if m.ndim != 2:
            raise ValueError(f"expected [N, d] chunk, got {m.shape} in {f}")
        if d is None:
            d = m.shape[1]
        elif m.shape[1] != d:
            raise ValueError(f"dim mismatch {m.shape[1]} != {d} in {f}")
        mmaps.append(m)
        total += m.shape[0]
    log(f"total tokens available: {total}, dim={d}")

    rng = np.random.default_rng(seed)
    if max_tokens is not None and max_tokens < total:
        # Sample token indices across the concatenated stream, then gather.
        pick = np.sort(rng.choice(total, size=max_tokens, replace=False))
        out = np.empty((max_tokens, d), dtype=np.float32)
        offsets = np.cumsum([0] + [m.shape[0] for m in mmaps])
        wi = 0
        for ci, m in enumerate(mmaps):
            lo, hi = offsets[ci], offsets[ci + 1]
            sel = pick[(pick >= lo) & (pick < hi)] - lo
            if sel.size:
                out[wi : wi + sel.size] = np.asarray(m[sel], dtype=np.float32)
                wi += sel.size
        log(f"subsampled {max_tokens} tokens (of {total})")
        return out, d
    # Load all (chunk-concatenate as float32).
    out = np.empty((total, d), dtype=np.float32)
    wi = 0
    for m in mmaps:
        out[wi : wi + m.shape[0]] = np.asarray(m, dtype=np.float32)
        wi += m.shape[0]
    return out, d


# ---------------------------------------------------------------------------
# Stage 1a: feature co-activation clustering.
# ---------------------------------------------------------------------------

def feature_coactivation_groups(
    codes: np.ndarray,         # [N_eval, F] dense SAE codes on a held-out token sample
    act_count: np.ndarray,     # [F] full-data activation counts
    top_features: int,
    n_groups: int,
    min_group: int,
    seed: int,
    log=print,
):
    """Build candidate feature groups from co-activation correlation.

    1. Restrict to the `top_features` most-active (non-dead) features.
    2. Compute the feature-feature co-activation correlation on the binary
       firing pattern across tokens.
    3. Spectral-cluster the correlation graph into `n_groups`; keep groups with
       >= `min_group` features.
    Returns a list of feature-index arrays (into the full F dictionary).
    """
    F = codes.shape[1]
    order = np.argsort(-act_count)
    alive = order[act_count[order] > 0][:top_features]
    if alive.size < min_group * 2:
        log(f"WARNING: only {alive.size} live features; co-activation clustering thin")
    fire = (codes[:, alive] > 0).astype(np.float32)  # [N, M]
    # Center per-feature firing rate; correlation across tokens.
    fc = fire - fire.mean(axis=0, keepdims=True)
    denom = np.sqrt((fc * fc).sum(axis=0))
    denom[denom == 0] = 1.0
    corr = (fc.T @ fc) / (denom[:, None] * denom[None, :])  # [M, M]
    np.fill_diagonal(corr, 1.0)

    # Spectral embedding of the affinity (|corr| as similarity), then k-means.
    aff = np.abs(corr)
    deg = aff.sum(axis=1)
    deg[deg == 0] = 1.0
    dinv = 1.0 / np.sqrt(deg)
    lap = np.eye(aff.shape[0]) - (dinv[:, None] * aff * dinv[None, :])
    # Smallest non-trivial eigenvectors as the embedding.
    vals, vecs = np.linalg.eigh(lap)
    emb = vecs[:, 1 : 1 + min(n_groups, vecs.shape[1] - 1)]
    labels = _kmeans(emb, n_groups, seed, log)

    groups = []
    for g in range(n_groups):
        members = alive[labels == g]
        if members.size >= min_group:
            groups.append(np.sort(members))
    log(f"co-activation clustering: {len(groups)} candidate groups "
        f"(>= {min_group} features) from {alive.size} live top features")
    return groups


def _kmeans(X: np.ndarray, k: int, seed: int, log, iters: int = 50):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    k = min(k, n)
    centers = X[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new = d2.argmin(axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            m = labels == c
            if m.any():
                centers[c] = X[m].mean(axis=0)
    return labels


# ---------------------------------------------------------------------------
# Stage 1b: per-group 2-D coordinates + gam adjudication.
# ---------------------------------------------------------------------------

def group_coords_2d(codes: np.ndarray, members: np.ndarray, W_dec: np.ndarray,
                    max_rows: int, seed: int):
    """Project the group's per-token contribution to 2-D intrinsic coords.

    The group's reconstruction in ACTIVATION space is `codes[:, members] @
    W_dec[members]`. We adjudicate the geometry there rather than on the raw
    ReLU codes: a circular feature in activation space is split by ReLU into
    cos+/cos-/sin+/sin- half-features, so the raw code submatrix folds the ring
    into a non-negative quadrant. Reconstructing in activation space un-folds
    that split, recovering the genuine manifold the adjudicator should judge.
    Restrict to tokens where the group is meaningfully active, then PCA to 2-D.
    Returns [m, 2] or None if too few active rows / degenerate.
    """
    sub = codes[:, members]                       # [N, |group|]
    active_rows = np.where((sub > 0).sum(axis=1) >= max(2, members.size // 4))[0]
    if active_rows.size < 16:
        return None
    rng = np.random.default_rng(seed)
    if active_rows.size > max_rows:
        active_rows = np.sort(rng.choice(active_rows, size=max_rows, replace=False))
    # Group contribution in activation space, then PCA to 2-D.
    M = sub[active_rows] @ W_dec[members]         # [m, d_in]
    M = M - M.mean(axis=0, keepdims=True)
    # PCA to 2-D via SVD.
    try:
        U, S, _ = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if S.size < 2 or S[1] < 1e-9:
        return None
    coords = U[:, :2] * S[:2]
    if not np.all(np.isfinite(coords)) or coords.std() < 1e-9:
        return None
    return np.ascontiguousarray(coords.astype(np.float64))


def adjudicate_groups(gamfit, codes, groups, W_dec, max_rows, seed, log=print):
    verdicts = []
    mean_l0 = float(np.count_nonzero(codes, axis=1).mean())
    for gi, members in enumerate(groups):
        coords = group_coords_2d(codes, members, W_dec, max_rows, seed + gi)
        if coords is None:
            log(f"  group {gi:3d} ({members.size} feats): too few active rows, skipped")
            continue
        try:
            v = gamfit.adjudicate_atom_shape(
                coords,
                folds=5,
                seed=seed + 11 + gi,
                mean_l0=mean_l0,
            )
        except Exception as exc:  # noqa: BLE001
            log(f"  group {gi:3d}: adjudication error: {exc}")
            continue
        rec = {
            "group": gi,
            "n_features": int(members.size),
            "n_rows": int(coords.shape[0]),
            "winner": v["winner"],
            "circle_wins": bool(v["circle_wins"]),
            "circle_margin": float(v["circle_margin"]),
            "mixture_k": int(v["mixture_k"]),
            "headline": v["headline"],
            "stacking_weights": dict(zip(v["candidate_names"], v["stacking_weights"])),
            "dictionary_mean_l0": float(v["dictionary_mean_l0"]),
            "control_false_circle_floor": float(v["control_false_circle_floor"]),
            "matched_controls": {
                name: {
                    "winner": control["winner"],
                    "circle_wins": bool(control["circle_wins"]),
                    "circle_margin": float(control["circle_margin"]),
                    "mixture_k": int(control["mixture_k"]),
                }
                for name, control in v["matched_controls"].items()
            },
        }
        verdicts.append(rec)
        log(f"  group {gi:3d} ({members.size:3d} feats, {coords.shape[0]:4d} rows): "
            f"{v['winner']:16s} margin={v['circle_margin']:+.4f} k={v['mixture_k']}")
    return verdicts


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

RUN_SPEC = """\
=== #977 real Qwen SGD-SAE -> gam-adjudication pipeline — cluster run spec ===

Data (already staged):
  /path/to/scratch/qwen_acts/activations/layer_30_residual_post/*.npy

Build the wheel on a COMPUTE node (never login), then run on an a100:

  # STEP 1 (build, if the olmo_venv wheel lacks adjudicate_atom_shape):
  sbatch -p <gpu-partition> --gres=gpu:a100:1 -t 90 --wrap '
    source /path/to/scratch/gam_env.sh
    export HF_HOME=/path/to/scratch/hf
    export TMPDIR=/path/to/scratch/tmp
    cd /path/to/scratch/gam
    maturin build --release -o dist && pip install --force-reinstall dist/*.whl'

  # STEP 2 (train SAE + adjudicate):
  sbatch -p <gpu-partition> --gres=gpu:a100:1 -t 120 --wrap '
    source /path/to/scratch/olmo_venv/bin/activate
    export HF_HOME=/path/to/scratch/hf
    export TMPDIR=/path/to/scratch/tmp
    cd /path/to/scratch/gam
    python tests/sae/qwen_real_sae_pipeline.py \\
      --act-dir /path/to/scratch/qwen_acts/activations/layer_30_residual_post \\
      --dict-size 16384 --epochs 8 --max-tokens 400000 \\
      --out /path/to/scratch/qwen_sae_verdict.json'

Tune down for a quick smoke: --dict-size 4096 --epochs 2 --max-tokens 50000.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--act-dir", type=str, default=None,
                    help="dir with layer_30_residual_post/*.npy chunks")
    ap.add_argument("--dict-size", type=int, default=16384)
    ap.add_argument("--l1-coeff", type=float, default=5e-3)
    ap.add_argument("--activation", choices=["relu", "jumprelu"], default="relu")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--max-tokens", type=int, default=400000)
    ap.add_argument("--eval-tokens", type=int, default=20000,
                    help="held-out token sample for co-activation + coords")
    ap.add_argument("--top-features", type=int, default=2048)
    ap.add_argument("--n-groups", type=int, default=64)
    ap.add_argument("--min-group", type=int, default=4)
    ap.add_argument("--max-group-rows", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--print-run-spec", action="store_true")
    args = ap.parse_args()

    if args.print_run_spec:
        print(RUN_SPEC)
        return 0
    if not args.act_dir:
        print("ERROR: --act-dir required (or --print-run-spec).", file=sys.stderr)
        return 2

    t0 = time.time()
    try:
        import torch  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"BLOCKER: torch not importable ({exc}).", file=sys.stderr)
        return 3
    try:
        import gamfit
    except Exception as exc:  # noqa: BLE001
        print(f"BLOCKER: gamfit wheel not importable ({exc}).", file=sys.stderr)
        return 3
    if not hasattr(gamfit, "adjudicate_atom_shape"):
        print("BLOCKER: gamfit wheel lacks adjudicate_atom_shape — rebuild "
              "(capstone 222cdb2af). Without it the un-forced judge is missing.",
              file=sys.stderr)
        return 3

    from torch_sgd_sae import SAEConfig, train_sae

    # --- Stage 0: load + train the real SGD SAE ---------------------------
    acts, d = load_activations(args.act_dir, args.max_tokens, args.seed)
    cfg = SAEConfig(
        d_in=d, dict_size=args.dict_size, l1_coeff=args.l1_coeff,
        activation=args.activation, lr=args.lr, batch_size=args.batch_size,
        epochs=args.epochs, seed=args.seed, device=args.device,
    )
    print(f"=== Stage 0: training overcomplete {args.activation} SAE "
          f"d_in={d} dict={args.dict_size} on {acts.shape[0]} tokens ===")
    model, sae_stats, act_count = train_sae(acts, cfg)

    # --- held-out code sample for geometry --------------------------------
    import torch
    device = next(model.parameters()).device
    rng = np.random.default_rng(args.seed + 1)
    n_eval = min(args.eval_tokens, acts.shape[0])
    eval_idx = np.sort(rng.choice(acts.shape[0], size=n_eval, replace=False))
    norm_scale = sae_stats["norm_scale"]  # apply the same dataset normalization
    with torch.no_grad():
        codes_list = []
        for s in range(0, n_eval, args.batch_size):
            xb = torch.from_numpy(acts[eval_idx[s : s + args.batch_size]]).float().to(device)
            xb = xb * norm_scale
            codes_list.append(model.encode(xb).cpu().numpy())
        codes = np.concatenate(codes_list, axis=0)  # [n_eval, F]
    print(f"=== Stage 1a: co-activation clustering on {n_eval} held-out tokens ===")

    # --- Stage 1a: candidate groups ---------------------------------------
    groups = feature_coactivation_groups(
        codes, act_count, args.top_features, args.n_groups, args.min_group, args.seed)

    # --- Stage 1b: gam adjudicates each group (NO forced topology) ---------
    print(f"=== Stage 1b: gam cross-class adjudication of {len(groups)} groups ===")
    W_dec = model.W_dec.detach().cpu().numpy()  # [F, d_in]
    verdicts = adjudicate_groups(gamfit, codes, groups, W_dec, args.max_group_rows, args.seed)

    # --- Honest report ----------------------------------------------------
    n_adj = len(verdicts)
    n_circle = sum(1 for v in verdicts if v["circle_wins"])
    control_verdicts = [
        control
        for verdict in verdicts
        for control in verdict["matched_controls"].values()
    ]
    n_control_circle = sum(1 for control in control_verdicts if control["circle_wins"])
    n_curved_strong = sum(1 for v in verdicts if v["circle_wins"] and v["circle_margin"] > 0.05)
    from collections import Counter
    by_winner = Counter(v["winner"].split("(")[0].split("_k")[0] for v in verdicts)
    summary = {
        "sae": sae_stats,
        "n_candidate_groups": len(groups),
        "n_adjudicated": n_adj,
        "n_circle_wins": n_circle,
        "n_control_circle_wins": n_control_circle,
        "n_control_verdicts": len(control_verdicts),
        "control_false_circle_rate": (
            n_control_circle / len(control_verdicts) if control_verdicts else None
        ),
        "n_curved_strong_margin>0.05": n_curved_strong,
        "winner_breakdown": dict(by_winner),
        "verdicts": verdicts,
        "wall_seconds": time.time() - t0,
        "config": vars(args),
    }
    print("\n=== HONEST VERDICT ===")
    print(f"SAE: EV={sae_stats['explained_variance']:.4f} "
          f"mean_L0={sae_stats['mean_l0']:.1f} "
          f"dead_frac={sae_stats['dead_feature_fraction']:.3f}")
    print(f"groups adjudicated: {n_adj}/{len(groups)}")
    print(f"winner breakdown: {dict(by_winner)}")
    print(f"circle (curved) wins: {n_circle}/{n_adj}"
          + (f" ({100*n_circle/n_adj:.0f}%)" if n_adj else ""))
    print(f"matched-control circle wins: {n_control_circle}/{len(control_verdicts)}"
          + (f" ({100*n_control_circle/len(control_verdicts):.0f}%), "
             f"dictionary mean_L0={sae_stats['mean_l0']:.1f}"
             if control_verdicts else ""))
    print(f"strongly-curved (margin>0.05): {n_curved_strong}/{n_adj}")
    if n_adj == 0:
        print("FINDING: no group could be adjudicated (degenerate group geometry).")
    elif n_circle == 0:
        print("FINDING: ZERO groups are genuinely curved — the real Qwen feature "
              "groups adjudicate as cluster/euclidean. The curved-atom wager LOSES "
              "measurably on this layer. That is the honest result.")
    elif n_circle < 0.2 * n_adj:
        print("FINDING: a SMALL minority of groups are genuinely curved; most are "
              "cluster/euclidean. The wager partially holds — most structure is "
              "discrete superposition, a few genuine curved atoms.")
    else:
        print("FINDING: a substantial fraction of groups are genuinely curved — "
              "gam's un-forced judge confirms curved structure in the real "
              "dictionary beyond the cluster null.")

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2, default=str))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
