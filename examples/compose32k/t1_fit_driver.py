"""#2023 Step 1 (task #23): certification-at-width driver — the LINEAR arm.

ONE unified model, tiers = optimizer schedule. Runs the linear dictionary at 32k
capacity and certifies how many atoms earn their keep under the SHARED currency:

  (a) streaming block-sparse Tier-1 fit at G=16384 x b=2 (32k linear capacity) on
      the real L17 shards, per-context (per-rollout) demeaned, TRAIN rollouts only;
  (b) the per-atom rank charge over EVERY atom via the native streaming #9 charge
      (½·d_eff·ln n, MP floor, deviance units per #20) — never a projector/fraction;
  (c) emits the certified subset + a JSONL ledger (one row per atom).

Width is CAPACITY: the certified count is the OUTPUT, never a growth target.

CONTRACT v1 (shared across the 4 steps):
  venv   = source $ROOT/gamfit_current_manifest.sh
  data   = canonical `l17_data` (tier2's #17 A/B module): load_l17 + rollout_split
           + per_context_demean — IMPORTED, never re-implemented (identical split).
  arts   = --out-dir (t1/): t1_artifact.npz + seed manifest + t1_certified.jsonl
           + t1_report.json.
  ledger = JSONL {atom_id, block_g, kind, n_eff, d_eff, delta_deviance, charge,
           margin, kept, log_e_value}.

The per-atom charge is `stream.block_rank_charges(n_obs)` (team-lead's FINAL
accessor: called AFTER the final end_epoch, BEFORE finalize; reads the last closed
epoch; per-BLOCK rows flattened to the block's b atoms). `--smoke` validates the
machinery on a synthetic corpus with a PLACEHOLDER charge (no wheel/data dep); the
real run refuses to fake the shared currency.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Canonical shared modules live beside this driver (tier2's l17_data) and in the
# gam examples/ (compose_tiers.emit_block_seed_manifest). Add both to the path.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, "/projects/standard/hsiehph/sauer354/scratch",
           "/projects/standard/hsiehph/sauer354/scratch/compose32k/scripts"):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def flatten_block_ledger(ledger: dict, block_size: int) -> list[dict]:
    """Block g's b atoms share one frame + code Gram, so the native charge prices
    them together; flatten block g -> atom_ids g*b..(g+1)*b-1, each inheriting the
    block's row. margin doubles as the log e-value (team-lead's contract)."""
    recs = []
    n = len(ledger["block"])
    for i in range(n):
        g = int(ledger["block"][i])
        row = {
            "block_g": g, "kind": "linear",
            "n_eff": float(ledger["n_eff"][i]),
            "d_eff": float(ledger["d_eff"][i]),
            "delta_deviance": float(ledger["delta_deviance"][i]),
            "charge": float(ledger["charge"][i]),
            "margin": float(ledger["margin"][i]),
            "kept": bool(ledger["kept"][i]),
            "log_e_value": float(ledger["margin"][i]),
        }
        for a in range(block_size):
            recs.append({"atom_id": g * block_size + a, **row})
    return recs


def emit_members_and_frames(fit, x: np.ndarray, out_dir: str) -> np.ndarray:
    """Step-1 -> Step-2 seed primitive (block_topk=1). Emits:
      frames.npz  : `frames` (G,b,P) f32 — D_g (block g basis Q = frames[g].T).
      members.npz : `member_block` (N,) i32 [each token's block id], `member_coords`
                    (N,b) f32 [token coords in ITS block's frame = X@D_gᵀ], `member_row`
                    (N,) i64 [index into the canonical train array].
      sample_2block.npz : frames[:2] + blocks 0/1 member coords — a load-check.
    tier2 gathers block g's chart input as `member_coords[member_block==g]`. Since
    top-k=1 partitions the tokens, total is N rows (~MBs), not the N-per-block monolith.
    """
    G = int(fit.n_blocks); b = int(fit.block_size)
    frames = np.ascontiguousarray(np.asarray(fit.decoder, dtype=np.float32).reshape(G, b, -1))
    np.savez(os.path.join(out_dir, "frames.npz"), frames=frames)
    tr = fit.transform(x)
    member_block = np.asarray(tr[0])[:, 0].astype(np.int32)  # top-k=1
    n = int(x.shape[0])
    member_coords = np.zeros((n, b), dtype=np.float32)
    for g in np.unique(member_block):
        m = member_block == g
        member_coords[m] = x[m].astype(np.float32) @ frames[g].T  # (n_g, b)
    np.savez(os.path.join(out_dir, "members.npz"),
             member_block=member_block, member_coords=member_coords,
             member_row=np.arange(n, dtype=np.int64))
    m0 = member_block == 0; m1 = member_block == 1
    np.savez(os.path.join(out_dir, "sample_2block.npz"), frames=frames[:2],
             block0_coords=member_coords[m0], block1_coords=member_coords[m1])
    return np.bincount(member_block, minlength=G)


def _placeholder_ledger(fit, n_obs: int) -> dict:
    """Smoke only: NOT the shared currency — exercises the JSONL path per-block."""
    b = int(fit.block_size)
    sr = np.asarray(fit.block_stable_rank, dtype=np.float64)
    ut = np.asarray(fit.block_utilization, dtype=np.float64)
    ln_n = float(np.log(max(n_obs, 2)))
    G = int(fit.n_blocks)
    d_eff = sr * b
    charge = 0.5 * d_eff * ln_n
    # rigged so utilised blocks tend to clear (placeholder shape only)
    delta = ut * n_obs * 0.5 * (sr / max(b, 1))
    margin = delta - charge
    return {"block": list(range(G)), "n_eff": (ut * n_obs).tolist(),
            "d_eff": d_eff.tolist(), "delta_deviance": delta.tolist(),
            "charge": charge.tolist(), "margin": margin.tolist(),
            "kept": (margin > 0).tolist()}


def _synthetic(n_blocks: int, block_size: int, seed: int):
    rng = np.random.default_rng(seed)
    p = 2048
    q, _ = np.linalg.qr(rng.standard_normal((p, p)))
    atoms = q[:, : n_blocks * block_size].T
    n = 20000
    x = np.zeros((n, p), np.float32)
    for i in range(n):
        t = i % n_blocks
        for r in range(block_size):
            x[i] += rng.standard_normal() * atoms[t * block_size + r]
    x += 0.03 * rng.standard_normal((n, p)).astype(np.float32)
    return np.ascontiguousarray(x - x.mean(0, keepdims=True), dtype=np.float32)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shards", "--shard-dir", dest="shards",
                    default="/projects/standard/hsiehph/sauer354/scratch/qwen_stageb/shards")
    ap.add_argument("--out-dir", default="/projects/standard/hsiehph/sauer354/scratch/compose32k/t1")
    ap.add_argument("--ckpt", default=None, help="artifact path prefix (Step-4 resume sentinel)")
    ap.add_argument("--n-blocks", type=int, default=16384)
    ap.add_argument("--block-size", type=int, default=2)
    ap.add_argument("--block-topk", type=int, default=1)
    ap.add_argument("--aux-k", type=int, default=64)
    ap.add_argument("--max-epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=65536)
    ap.add_argument("--seed-rows", type=int, default=32768)
    ap.add_argument("--fdr-alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="synthetic corpus + PLACEHOLDER charge — machinery only")
    args = ap.parse_args(argv)

    import gamfit
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if args.smoke:
        x = _synthetic(args.n_blocks, args.block_size, args.seed)
    else:
        import l17_data  # tier2's canonical module (Contract v1)
        acts, rids, _metas = l17_data.load_l17(args.shards)
        train = l17_data.rollout_split(rids, seed=args.seed)
        x = np.ascontiguousarray(
            l17_data.per_context_demean(acts[train], rids[train]), dtype=np.float32)
    n_obs = int(x.shape[0])
    seed_sample = np.ascontiguousarray(x[: args.seed_rows])
    if seed_sample.shape[0] == 0:
        raise SystemExit("no TRAIN rows to seed the fit")
    print(f"[t1] gamfit {gamfit.__version__} G={args.n_blocks} b={args.block_size} "
          f"n_obs={n_obs} seed_rows={seed_sample.shape[0]} smoke={args.smoke}", flush=True)

    stream = gamfit.block_sparse_dictionary_fit_begin(
        seed_sample, args.n_blocks, block_size=args.block_size, block_topk=args.block_topk,
        max_epochs=args.max_epochs, block_tile=1024, aux_k=args.aux_k,
    )
    t0 = time.time()
    for epoch in range(args.max_epochs):
        rows = 0
        for s in range(0, n_obs, args.batch):
            rows += stream.partial_fit(x[s:s + args.batch])["rows"]
        ep = stream.end_epoch()
        print(f"[t1 epoch {ep['epoch']:>2}] rows={rows} EV={ep['explained_variance']:.4f} "
              f"gamma={ep['gamma']:.3f} dead={ep['dead']} revived={ep['revived']} "
              f"converged={ep['converged']} t={time.time()-t0:.0f}s", flush=True)
        if ep["converged"]:
            break

    # PER-ATOM RANK CHARGE — the shared currency. AFTER the last end_epoch, BEFORE
    # finalize (team-lead's contract: reads the last closed epoch's accumulators).
    charges = getattr(stream, "block_rank_charges", None)
    if args.smoke:
        # placeholder needs the fit's stats; build a throwaway fit first is heavy —
        # approximate from a finalize()-preview is not allowed, so use utilisation
        # captured by a to_fit below. Deferred: fill after fit.
        raw_ledger = None
    elif charges is None:
        raise NotImplementedError(
            "stream.block_rank_charges not in this wheel — needs team-lead's accessor "
            "(rebuild the manifest wheel). Certified run refuses to fake the currency."
        )
    else:
        raw_ledger = stream.block_rank_charges(n_obs)

    art = stream.finalize()
    fit = art.to_fit(seed_sample)
    if args.smoke:
        raw_ledger = _placeholder_ledger(fit, n_obs)
    recs = flatten_block_ledger(raw_ledger, args.block_size)

    # ---- t1_artifact.npz (tier2's schema: decoder K×P + block metadata) ----
    decoder = np.ascontiguousarray(np.asarray(fit.decoder, dtype=np.float32))  # (K=G*b, P)
    np.savez(
        os.path.join(args.out_dir, "t1_artifact.npz"),
        decoder=decoder, block_size=np.int32(args.block_size),
        block_topk=np.int32(args.block_topk), n_blocks=np.int32(fit.n_blocks),
        gamma=np.float32(getattr(art, "gamma", 0.0)),
        block_stable_rank=np.asarray(fit.block_stable_rank, dtype=np.float32),
        block_utilization=np.asarray(fit.block_utilization, dtype=np.float32),
    )
    # ---- Step-2 seed primitive: frames + per-token member coords (topk=1) ----
    n_members = emit_members_and_frames(fit, x, args.out_dir)
    Path(os.path.join(args.out_dir, "t1_seed_manifest.json")).write_text(json.dumps({
        "schema": "t1_seed.v1", "n_blocks": int(fit.n_blocks),
        "block_size": int(args.block_size), "block_topk": int(args.block_topk),
        "ambient_p": int(np.asarray(fit.decoder).shape[1]), "n_train": int(x.shape[0]),
        "blocks": [{"block": int(g), "n_members": int(n_members[g]),
                    "stable_rank": float(np.asarray(fit.block_stable_rank)[g]),
                    "utilization": float(np.asarray(fit.block_utilization)[g])}
                   for g in range(int(fit.n_blocks))],
    }, indent=2))

    # ---- t1_certified.jsonl ----
    with open(os.path.join(args.out_dir, "t1_certified.jsonl"), "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")

    # counts: margin-kept AND FDR-controlled (e-BH on per-BLOCK log-e-values)
    kept_atoms = [r for r in recs if r["kept"]]
    block_loges = [float(raw_ledger["margin"][i]) for i in range(len(raw_ledger["block"]))]
    cert_blocks = gamfit.e_bh_dictionary_certificate(block_loges, args.fdr_alpha)
    n_cert_atoms = len(cert_blocks) * args.block_size

    report = {
        "n_linear_certified": n_cert_atoms,
        "kept_by_margin": len(kept_atoms),
        "certified_blocks_fdr": len(cert_blocks),
        "capacity_K": int(fit.n_blocks) * int(args.block_size),
        "n_atoms": len(recs), "n_blocks": int(fit.n_blocks),
        "fdr_alpha": args.fdr_alpha,
        "explained_variance": float(art.explained_variance),
        "n_tokens": n_obs, "wall_s": round(time.time() - t0, 1),
        "placeholder_charge": bool(args.smoke),
    }
    Path(os.path.join(args.out_dir, "t1_report.json")).write_text(json.dumps(report, indent=2))
    print(f"[t1] CERTIFIED linear: {n_cert_atoms} atoms / {len(cert_blocks)} blocks (e-BH α={args.fdr_alpha}); "
          f"margin-kept {len(kept_atoms)}; K_capacity={report['capacity_K']} "
          f"EV={report['explained_variance']:.4f} {'[PLACEHOLDER charge]' if args.smoke else ''}", flush=True)


if __name__ == "__main__":
    main()
