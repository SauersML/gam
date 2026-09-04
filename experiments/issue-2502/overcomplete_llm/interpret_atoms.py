"""What the atoms of the Rust dictionary are (issue #2502, criterion 2).

Reads the Rust fit's held-out dump (`eval_blocks.u32`, `eval_codes.f32`) and the
harvest's token ids, and answers two questions per block:

* **identity** — which tokens does this block fire on, and how selective is that
  against the corpus base rate (a lift, not a top-k list);
* **coordinate** — for a `b = 2` block the code is a point in a plane, so it has
  an angle `theta = atan2(z1, z0)`. If the block were a mere "direction detector"
  the angle would carry no content. The circular resultant length `R_w` of theta
  over the firings of one token type measures whether it does; the null is the
  same statistic after permuting theta within the block, which destroys the
  token/angle association and nothing else.

Pure indexing and counting over bytes the Rust engine produced.
"""

import argparse
import json
import os

import numpy as np
from transformers import AutoTokenizer


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-dir", required=True)
    ap.add_argument("--acts-dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--topk", type=int, required=True, help="k blocks per row in the dump")
    ap.add_argument("--block-size", type=int, required=True)
    ap.add_argument("--n-blocks", type=int, required=True)
    ap.add_argument("--report-blocks", type=int, default=40)
    ap.add_argument("--min-firings", type=int, default=200)
    ap.add_argument("--permutations", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    k, b, g = args.topk, args.block_size, args.n_blocks
    tokens = np.load(os.path.join(args.acts_dir, "eval_tokens.npy"))
    blocks = np.fromfile(os.path.join(args.fit_dir, "eval_blocks.u32"), dtype=np.uint32)
    codes = np.fromfile(os.path.join(args.fit_dir, "eval_codes.f32"), dtype=np.float32)
    n = blocks.size // k
    blocks = blocks.reshape(n, k)
    codes = codes.reshape(n, k, b)
    tokens = tokens[:n]
    print(f"[interp] rows={n} k={k} b={b} G={g}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    rng = np.random.default_rng(args.seed)

    counts = np.bincount(blocks.reshape(-1), minlength=g)
    order = np.argsort(-counts)
    base_counts = np.bincount(tokens, minlength=int(tokens.max()) + 1).astype(np.float64)
    base_rate = base_counts / base_counts.sum()

    rows = []
    for gid in order[: args.report_blocks]:
        gid = int(gid)
        hit_r, hit_j = np.nonzero(blocks == gid)
        if hit_r.size < args.min_firings:
            continue
        z = codes[hit_r, hit_j, :]
        gate = np.linalg.norm(z, axis=1)
        tk = tokens[hit_r]

        uniq, cnt = np.unique(tk, return_counts=True)
        keep = np.argsort(-cnt)[:12]
        identity = []
        for i in keep:
            t = int(uniq[i])
            share = cnt[i] / hit_r.size
            lift = share / max(base_rate[t], 1.0 / base_counts.sum())
            identity.append(
                {
                    "token_id": t,
                    "token": tok.decode([t]),
                    "firings": int(cnt[i]),
                    "share": float(share),
                    "lift_over_corpus": float(lift),
                }
            )

        entry = {
            "block": gid,
            "atoms": [gid * b + r for r in range(b)],
            "firings": int(hit_r.size),
            "firing_rate": float(hit_r.size / n),
            "mean_gate": float(gate.mean()),
            "top_tokens": identity,
        }

        if b == 2:
            theta = np.arctan2(z[:, 1], z[:, 0])
            # The code is defined up to the block's O(b) gauge, so an ABSOLUTE
            # angle means nothing; only the association between token identity and
            # angle does, which is what the resultant measures and the permutation
            # null removes.
            tested = [int(uniq[i]) for i in keep[:6] if cnt[i] >= 20]
            angle_rows = []
            for t in tested:
                sel = tk == t
                ang = theta[sel]
                # R1 is the directed resultant (a preferred direction in the
                # plane); R2 is the axial resultant (a preferred AXIS, invariant
                # to z -> -z, which the signed code is free to flip). Both are
                # invariant to the block's O(2) gauge; only the token/angle
                # association they measure is.
                r1 = float(np.abs(np.exp(1j * ang).mean()))
                r2 = float(np.abs(np.exp(2j * ang).mean()))
                null1 = np.empty(args.permutations)
                null2 = np.empty(args.permutations)
                for q in range(args.permutations):
                    perm = rng.permutation(theta.size)[: ang.size]
                    null1[q] = np.abs(np.exp(1j * theta[perm]).mean())
                    null2[q] = np.abs(np.exp(2j * theta[perm]).mean())
                angle_rows.append(
                    {
                        "token_id": t,
                        "token": tok.decode([t]),
                        "firings": int(sel.sum()),
                        "resultant_R1": r1,
                        "null_max_R1": float(null1.max()),
                        "resultant_R2": r2,
                        "null_max_R2": float(null2.max()),
                        "exceeds_all_nulls": bool(r1 > null1.max() or r2 > null2.max()),
                    }
                )
            entry["angle"] = angle_rows
        rows.append(entry)

    n_tested = sum(len(r.get("angle", [])) for r in rows)
    n_beat = sum(
        1 for r in rows for a in r.get("angle", []) if a["exceeds_all_nulls"]
    )
    report = {
        "rows_scored": int(n),
        "blocks_reported": len(rows),
        "angle_tests": n_tested,
        "angle_tests_beating_every_null": n_beat,
        "permutations_per_test": args.permutations,
        "blocks": rows,
    }
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(
        f"[interp] blocks_reported={len(rows)} angle_tests={n_tested} "
        f"beating_every_null={n_beat} wrote {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
