"""Universality across seeds as ground truth for the #2951 toy gate: a mechanism a decomposition
finds should reappear in the decompositions of the same toy trained from other seeds, up to the
toy's symmetries, while an artifact of one fit should not. This needs no written-down truth (SPD's
written truth for the cross-layer toys is already wrong).

usage: ~/mpd-data/venv/bin/python bench/toys_2951/universality.py TOY_DIR PARTS_DIR TOY_DIR PARTS_DIR [...]

Each pair is one seed of one toy (train_toys.py TOY@sSEED) and a decomposition of it (the parts
dump). A part is compared across seeds in coordinates every seed shares, so the hidden units'
permutation and any rotation of a hidden space drop out: its reads of the residual stream (the
`V` of its slices on q, k, v and c_fc) and its writes to it (the `U` of its slices on o and
down_proj, and, for a part with no down_proj or o slice of its own, its c_fc or v slices' write
carried to the stream by M's own down_proj or o_proj), per layer. For a real-valued toy the stream's coordinates are the same in every seed by
construction (the inputs' slots, or the shared embedding); for a transformer they are mapped to
token space, reads through the token embedding and writes through the readout. Two parts' similarity is the mean, over the
sides (layer, read or write) either one has, of the overlap of their spans there,
`‖Q_aᵀ Q_b‖_F² / max(r_a, r_b)` (1 for the same span, 0 for orthogonal spans or a side only one
has). Between every two seeds the parts are matched one to one by the assignment of largest total
similarity (scipy's linear_sum_assignment). Reported per pair of seeds: the fraction of each
seed's parts whose match has similarity >= 0.9 (its universal parts), the mean matched
similarity, and, where the toy has a truth, how many universal parts are a recovered mechanism.
"""

import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent))
from score_toys import Parts, Toy, cosine  # noqa: E402

READS = ("attn.q_proj", "attn.k_proj", "attn.v_proj", "mlp.c_fc")
WRITES = ("attn.o_proj", "mlp.down_proj")


def sides(toy: Toy, parts: Parts) -> list:
    """Per part, per side (layer, 'read' | 'write'), an orthonormal basis of its span in shared
    coordinates."""
    real = "real_valued" in toy.record
    to_tokens = None if real else toy.weight("wte")  # vocab x d
    head = None if real else toy.weight("lm_head")  # vocab x d
    out = []
    for p in parts.parts:
        spans = {}
        for op, (U, V) in p["slices"].items():
            layer, kind = int(op.split(".")[1]), op.split(".", 2)[2]
            sides_of = []
            if kind in READS:
                sides_of.append((V, "read"))
            if kind in WRITES:
                sides_of.append((U, "write"))
            # A slice writing a hidden space (c_fc's neurons, v's heads) writes the stream through
            # M's own map out of it, so its write is compared in the stream's coordinates too: a
            # start's read directions alone are the same in every seed (the frames are drawn on the
            # inputs every seed shares) and say nothing of the model.
            if kind == "mlp.c_fc" and f"blocks.{layer}.mlp.down_proj" not in p["slices"]:
                sides_of.append((toy.weight(f"blocks.{layer}.mlp.down_proj") @ U, "write"))
            if kind == "attn.v_proj" and f"blocks.{layer}.attn.o_proj" not in p["slices"]:
                sides_of.append((toy.weight(f"blocks.{layer}.attn.o_proj") @ U, "write"))
            for vectors, side in sides_of:
                if side == "read" and to_tokens is not None:
                    vectors = to_tokens @ vectors
                if side == "write" and head is not None:
                    vectors = head @ vectors
                spans.setdefault((layer, side), []).append(vectors)
        bases = {}
        for key, blocks in spans.items():
            m = np.concatenate(blocks, 1)
            u, s, _ = np.linalg.svd(m, full_matrices=False)
            r = int(np.sum(s > s[0] * max(m.shape) * np.finfo(float).eps)) if s.size and s[0] > 0 else 0
            if r:
                bases[key] = u[:, :r]
        out.append(bases)
    return out


def similarity(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    total = 0.0
    for k in keys & set(a) & set(b):
        total += float(np.sum((a[k].T @ b[k]) ** 2)) / max(a[k].shape[1], b[k].shape[1])
    return total / len(keys)


def recovered_parts(toy: Toy, parts: Parts) -> set:
    """The parts that are a known mechanism's best match at cosine >= 0.9."""
    kept = set(parts.record.get("kept", []))
    fitted = [{op: Parts.weight(p, op) for op in p["slices"]} for p in parts.parts]
    found = set()
    for m in toy.mechanisms:
        delta = {op: d for op, d in m["deltas"].items() if op not in kept}
        cos = [cosine(delta, f) for f in fitted]
        if cos and max(cos) >= 0.9:
            found.add(int(np.argmax(cos)))
    return found


def main():
    args = sys.argv[1:]
    seeds = []
    for toy_dir, parts_dir in zip(args[::2], args[1::2]):
        toy = Toy(Path(toy_dir))
        parts = Parts(Path(parts_dir), toy)
        seeds.append({"toy": toy_dir, "parts": parts_dir, "sides": sides(toy, parts), "recovered": recovered_parts(toy, parts)})
    report = []
    for (i, a), (j, b) in combinations(enumerate(seeds), 2):
        sim = np.array([[similarity(x, y) for y in b["sides"]] for x in a["sides"]])
        rows, cols = linear_sum_assignment(-sim)
        matched = sim[rows, cols]
        universal = rows[matched >= 0.9]
        report.append({
            "seeds": [a["toy"], b["toy"]],
            "parts": [len(a["sides"]), len(b["sides"])],
            "universal_fraction": [float(len(universal) / max(len(a["sides"]), 1)), float(len(universal) / max(len(b["sides"]), 1))],
            "mean_matched_similarity": float(matched.mean()) if matched.size else math.nan,
            "universal_and_recovered": int(sum(1 for r in universal if r in a["recovered"])),
            "recovered": [len(a["recovered"]), len(b["recovered"])],
        })
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
