#!/usr/bin/env python3
"""Positional nuisance-atlas on the Qwen3-8B wikitext harvest (RUN ON MSI).

Tests whether the massive-activation channel that dominates certain layers
(L18: 99% of variance in one PC) is a POSITIONAL / attention-sink artifact.

The harvest (`harvest_acts.py`) wrote `resid_L{L}.npy` as the real (non-pad)
residual rows, concatenated over docs in corpus order, each doc contributing its
truncated `L_doc` tokens in position order (max_length=512). Doc lengths VARY, so
per-token position is NOT a fixed period of the row index — it is reconstructed
here by re-tokenizing the identical corpus (tokenizer only, no model, fully
deterministic) and validated against the harvest's token-count checksum.

For each layer we report the centred R² a positional design absorbs:
  * fourier   -- a normalized-position Fourier series cos/sin(2π j p/Pmax), j=1..H
  * pos0      -- a single first-token indicator (the attention-sink test)
  * early     -- indicators for the first few positions
  * combined  -- all of the above
and a PERMUTED-POSITION NULL (positions shuffled) for `combined`: if the signal
is genuinely positional the null collapses to the ~M/N overfit floor, ruling out
"any low-rank design fits a near-rank-1 matrix".

Usage (on MSI):
  python qwen_nuisance_msi.py \
    --harvest <harvest_dir> \
    --model Qwen/Qwen3-8B --layers 18,30,6 --out /tmp/qwen_nuisance.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np

PMAX = 512  # harvest seq_len / truncation length


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def reconstruct_corpus(n_docs: int, dataset: str, cfg: str) -> list[str]:
    """Reproduce harvest_acts.py's corpus exactly: first n_docs streamed docs with
    stripped length > 200."""
    from datasets import load_dataset

    ds = load_dataset(dataset, cfg, split="train", streaming=True)
    texts: list[str] = []
    for row in ds:
        t = row.get("text", "") or row.get("content", "")
        if t and len(t.strip()) > 200:
            texts.append(t.strip())
        if len(texts) >= n_docs:
            break
    return texts


def reconstruct_positions(tok, texts: list[str], n_rows: int) -> np.ndarray:
    """Per-row within-doc position, matching the harvest's row order: for each doc
    in order, arange(L_doc) with L_doc the truncated (max_length=PMAX) token count,
    concatenated and capped at n_rows."""
    pos_chunks: list[np.ndarray] = []
    total = 0
    for t in texts:
        ids = tok(t, truncation=True, max_length=PMAX)["input_ids"]
        ld = len(ids)
        pos_chunks.append(np.arange(ld, dtype=np.int32))
        total += ld
    positions = np.concatenate(pos_chunks)
    log(f"reconstructed positions: {positions.shape[0]} real tokens over {len(texts)} docs "
        f"(harvest processed ~301226; rows in npy = {n_rows})")
    return positions[:n_rows], total


def normalized_fourier(positions: np.ndarray, harmonics: int) -> np.ndarray:
    u = positions.astype(np.float64) / PMAX
    cols = []
    for j in range(1, harmonics + 1):
        cols.append(np.cos(2.0 * np.pi * j * u))
        cols.append(np.sin(2.0 * np.pi * j * u))
    return np.stack(cols, axis=1) if cols else np.zeros((positions.shape[0], 0))


def early_indicators(positions: np.ndarray, k: int) -> np.ndarray:
    return np.stack([(positions == q).astype(np.float64) for q in range(k)], axis=1)


def accumulate(
    npy_path: str,
    designs: list[np.ndarray],
    batch_rows: int = 20000,
    gather_mask: np.ndarray | None = None,
):
    """One streaming pass over the memmapped bank for a LIST of designs (each N×M):
    per design G=ZᵀZ, C=ZᵀX; shared Σx, Σx², n. Reading the 4.9 GB bank once for
    all designs (super + permuted null) halves the I/O.

    If `gather_mask` (bool, length N) is given, ALSO harvest the masked-row submatrix
    Xp (rows where the mask is true, in row order) in the same pass and return it
    alongside the accumulators. The MoE-expert interaction columns are supported ONLY
    on pos0 rows (mask = positions==0), so this small Xp is all the permutation null
    needs — see `expert_interaction_test`."""
    mm = np.load(npy_path, mmap_mode="r")
    n, d = mm.shape
    gs = [np.zeros((z.shape[1], z.shape[1])) for z in designs]
    cs = [np.zeros((z.shape[1], d)) for z in designs]
    sx = np.zeros(d)
    sx2 = np.zeros(d)
    xp_chunks: list[np.ndarray] | None = [] if gather_mask is not None else None
    for i in range(0, n, batch_rows):
        x = np.asarray(mm[i : i + batch_rows], dtype=np.float64)
        for k, z in enumerate(designs):
            zb = z[i : i + x.shape[0]]
            gs[k] += zb.T @ zb
            cs[k] += zb.T @ x
        sx += x.sum(0)
        sx2 += (x * x).sum(0)
        if xp_chunks is not None:
            mb = gather_mask[i : i + x.shape[0]]
            if mb.any():
                xp_chunks.append(x[mb])
    accs = [(gs[k], cs[k], sx, sx2, n) for k in range(len(designs))]
    if xp_chunks is not None:
        xp = np.concatenate(xp_chunks, axis=0) if xp_chunks else np.zeros((0, d))
        return accs, xp
    return accs


def absorbed(acc, cols: list[int]) -> float:
    """Centred aggregate R² of the sub-design on `cols` (must include intercept)."""
    g, c, sx, sx2, n = acc
    gg = g[np.ix_(cols, cols)]
    cc = c[cols, :]
    b = np.linalg.pinv(gg) @ cc
    ss_res = sx2 - 2.0 * np.einsum("kj,kj->j", b, cc) + np.einsum("kj,kj->j", b, gg @ b)
    ss_tot = sx2 - sx * sx / n
    tot = float(ss_tot.sum())
    return 0.0 if tot <= 0.0 else 1.0 - float(ss_res.sum()) / tot


def build_super_design(positions: np.ndarray, harmonics: int, early_k: int):
    """[intercept | fourier(2H) | early(early_k)] plus the column-index map so the
    fourier-only / pos0-only / early-only sub-designs are slices of one accumulator."""
    n = positions.shape[0]
    intercept = np.ones((n, 1))
    fourier = normalized_fourier(positions, harmonics)
    early = early_indicators(positions, early_k)
    super_z = np.concatenate([intercept, fourier, early], axis=1)
    nf = fourier.shape[1]
    ne = early.shape[1]
    cols = {
        "fourier": [0] + list(range(1, 1 + nf)),
        "pos0": [0, 1 + nf],  # early indicator q=0 is the first-token sink
        "early": [0] + list(range(1 + nf, 1 + nf + ne)),
        "combined": list(range(1 + nf + ne)),
    }
    return super_z, cols


def load_top1_expert(sidecar_dir: str, layer: int, n_rows: int) -> tuple[np.ndarray, str]:
    """Per-row top-1 MoE expert id for layer `layer`, from the router sidecar dir.

    `capture_router_sidecar.py` emits expert_L{L}_top{k}.i16.npy (full and per-split);
    we pick the candidate whose row count equals the bank's n_rows (this auto-selects
    the matching train/heldout split) and take column 0 (the top-1 expert)."""
    cands = sorted(glob.glob(os.path.join(sidecar_dir, f"expert_L{layer}_top*.i16.npy")))
    match = next((c for c in cands if int(np.load(c, mmap_mode="r").shape[0]) == n_rows), None)
    if match is None:
        raise SystemExit(
            f"no expert_L{layer}_top*.i16.npy in {sidecar_dir} with {n_rows} rows "
            f"(candidates: {[os.path.basename(c) for c in cands]})"
        )
    return np.load(match)[:, 0].astype(np.int64), match


def expert_interaction_test(acc, xp: np.ndarray, pos_labels: np.ndarray, n_experts: int,
                            n_perms: int, seed: int = 0) -> dict:
    """Does the pos0 attention-sink split by MoE router identity — beyond the sink itself?

    Baseline Z0 = [intercept | pos0]; enriched Z1 = Z0 + one pos0·1[top1==e] column for
    each of the E most-frequent-at-pos0 experts (others are the pooled reference, folded
    into the pos0 term). EV_delta = absorbed(Z1) − absorbed(Z0) is the extra centred
    aggregate R² the router identity buys at the sink.

    EXACTNESS OF THE NULL: every interaction column is zero off the pos0 rows, so
    permuting expert labels across rows changes only the pos0-supported blocks of ZᵀZ /
    ZᵀX. We rebuild those blocks analytically from the pos0-row submatrix `xp` (and the
    shared Σx, Σx², n from the full pass), so the B permutations run in-memory on the
    ~n_rollouts×d slice — no re-streaming of the bank, and identical to permuting the
    full design."""
    g_super, c_super, sx, sx2, n = acc
    d = sx.shape[0]
    P = int(xp.shape[0])
    empty = {
        "expert_ev_pos0": None, "expert_ev_pos0_plus_router": None,
        "expert_ev_delta": None, "expert_ev_delta_pvalue": None,
        "expert_ev_delta_null_mean": None, "n_pos0_rows": P,
        "n_experts_used": 0, "top_experts_at_pos0": [],
    }
    if P == 0 or pos_labels.shape[0] != P:
        return empty
    uniq, counts = np.unique(pos_labels, return_counts=True)
    top_e = uniq[np.argsort(counts)[::-1][:n_experts]]
    E = int(top_e.shape[0])
    if E == 0:
        return empty
    grp = np.full(P, -1, dtype=np.int64)  # group index in [0,E); -1 = pooled reference
    for j, e in enumerate(top_e):
        grp[pos_labels == e] = j
    pos0_sum = xp.sum(0)  # Σ X over pos0 rows = the pos0-column of ZᵀX
    eye_e = np.arange(E)

    def absorbed_delta(grp_arr: np.ndarray) -> tuple[float, float]:
        onehot = (grp_arr[:, None] == eye_e[None, :]).astype(np.float64)  # P×E
        ce = onehot.sum(0)          # per-expert pos0 counts
        xps = onehot.T @ xp         # E×d per-expert pos0 sums
        w = 2 + E
        g1 = np.zeros((w, w))
        c1 = np.zeros((w, d))
        g1[0, 0] = n
        g1[0, 1] = g1[1, 0] = g1[1, 1] = P
        for e in range(E):
            g1[0, 2 + e] = g1[2 + e, 0] = ce[e]
            g1[1, 2 + e] = g1[2 + e, 1] = ce[e]
            g1[2 + e, 2 + e] = ce[e]
        c1[0] = sx
        c1[1] = pos0_sum
        c1[2 : 2 + E] = xps
        acc1 = (g1, c1, sx, sx2, n)
        ev1 = absorbed(acc1, list(range(w)))
        ev0 = absorbed(acc1, [0, 1])
        return ev1, ev0

    ev1_obs, ev0 = absorbed_delta(grp)
    delta_obs = ev1_obs - ev0
    rng = np.random.default_rng(seed)
    null = np.empty(n_perms)
    for b in range(n_perms):
        null[b] = absorbed_delta(grp[rng.permutation(P)])[0] - ev0
    pval = (1.0 + float(np.sum(null >= delta_obs))) / (n_perms + 1.0)
    return {
        "expert_ev_pos0": float(ev0),
        "expert_ev_pos0_plus_router": float(ev1_obs),
        "expert_ev_delta": float(delta_obs),
        "expert_ev_delta_pvalue": float(pval),
        "expert_ev_delta_null_mean": float(null.mean()),
        "n_pos0_rows": P,
        "n_experts_used": E,
        "top_experts_at_pos0": [int(x) for x in top_e],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--harvest", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--layers", default="18,30,6")
    ap.add_argument("--harmonics", type=int, default=16)
    ap.add_argument("--early-k", type=int, default=8)
    ap.add_argument("--n-docs", type=int, default=6000)
    ap.add_argument("--dataset", default="Salesforce/wikitext")
    ap.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    ap.add_argument("--out", default="/tmp/qwen_nuisance.json")
    ap.add_argument(
        "--positions",
        default="",
        help="load precomputed per-row positions .npy (skips corpus/tokenizer — for "
        "compute nodes without network); if empty, reconstruct from the corpus",
    )
    ap.add_argument(
        "--save-positions",
        default="",
        help="reconstruct positions, save to this .npy, and exit (run on a login node "
        "with network)",
    )
    ap.add_argument(
        "--expert-sidecar",
        default="",
        help="dir of router sidecars (capture_router_sidecar.py output); enables the "
        "MoE pos0·expert interaction test per layer",
    )
    ap.add_argument("--n-experts", type=int, default=8,
                    help="explicit pos0-expert interaction columns (others pooled as reference)")
    ap.add_argument("--n-perms", type=int, default=200,
                    help="permuted-expert-label draws for the EV_delta null p-value")
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(args.harvest, "manifest.json")))
    layers = [int(x) for x in args.layers.split(",")]
    # Row count from the first layer's bank.
    first_file = os.path.join(args.harvest, f"resid_L{layers[0]}.npy")
    n_rows = int(np.load(first_file, mmap_mode="r").shape[0])

    if args.positions:
        positions = np.load(args.positions).astype(np.int32)
        total_tokens = int(positions.shape[0])
        log(f"loaded {total_tokens} precomputed positions from {args.positions}")
        if positions.shape[0] < n_rows:
            raise SystemExit(
                f"precomputed positions ({positions.shape[0]}) shorter than rows ({n_rows})"
            )
        positions = positions[:n_rows]
    else:
        log("loading tokenizer", args.model)
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model)
        log("reconstructing corpus")
        texts = reconstruct_corpus(args.n_docs, args.dataset, args.dataset_config)
        log(f"corpus docs: {len(texts)}")
        positions, total_tokens = reconstruct_positions(tok, texts, n_rows)
        if args.save_positions:
            np.save(args.save_positions, positions.astype(np.int32))
            log(f"saved {positions.shape[0]} positions to {args.save_positions}; exiting")
            return
        if positions.shape[0] != n_rows:
            raise SystemExit(
                f"position/row misalignment: {positions.shape[0]} positions vs {n_rows} rows "
                f"(reconstructed {total_tokens} tokens; check tokenizer/corpus reproduction)"
            )

    super_z, cols = build_super_design(positions, args.harmonics, args.early_k)
    rng = np.random.default_rng(0)
    null_z = super_z[rng.permutation(n_rows)]  # positions shuffled vs activations

    results = {
        "model": manifest.get("model"),
        "seq_len": manifest.get("seq_len"),
        "n_rows": n_rows,
        "reconstructed_tokens": int(total_tokens),
        "harmonics": args.harmonics,
        "early_k": args.early_k,
        "n_design_combined": super_z.shape[1],
        "expert_sidecar": args.expert_sidecar or None,
        "expert_n_experts": args.n_experts if args.expert_sidecar else None,
        "expert_n_perms": args.n_perms if args.expert_sidecar else None,
        "layers": {},
    }
    for L in layers:
        path = os.path.join(args.harvest, f"resid_L{L}.npy")
        if not os.path.exists(path):
            log(f"L{L}: {path} missing, skipping")
            continue
        stat = manifest["layers"].get(str(L), {})
        row = {
            "ev_top1": stat.get("ev_top1"),
            "participation_ratio": stat.get("participation_ratio"),
        }
        if args.expert_sidecar:
            # Gather the pos0-row (positions==0) submatrix in the SAME pass; the expert
            # interaction columns live only there.
            pos0_mask = positions == 0
            top1, sidecar_file = load_top1_expert(args.expert_sidecar, L, n_rows)
            (acc, null_acc), xp = accumulate(path, [super_z, null_z], gather_mask=pos0_mask)
        else:
            top1 = sidecar_file = xp = None
            acc, null_acc = accumulate(path, [super_z, null_z])  # single pass
        for name, cc in cols.items():
            row[f"absorbed_{name}"] = absorbed(acc, cc)
            log(f"L{L} {name}: absorbed {row[f'absorbed_{name}']:.4f}")
        row["absorbed_combined_null"] = absorbed(null_acc, cols["combined"])
        log(f"L{L} combined_null: absorbed {row['absorbed_combined_null']:.4f}")
        if args.expert_sidecar:
            et = expert_interaction_test(
                acc, xp, top1[pos0_mask], args.n_experts, args.n_perms, seed=0
            )
            et["expert_sidecar_file"] = os.path.basename(sidecar_file)
            row.update(et)
            log(f"L{L} expert: EV_delta {et['expert_ev_delta']} (pos0 {et['expert_ev_pos0']}) "
                f"p={et['expert_ev_delta_pvalue']} over {et['n_pos0_rows']} pos0 rows, "
                f"E={et['n_experts_used']}")
        results["layers"][str(L)] = row

    print("RESULTS_JSON " + json.dumps(results))
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    log("wrote", args.out)


if __name__ == "__main__":
    main()
