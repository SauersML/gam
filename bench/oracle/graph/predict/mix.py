"""An SFT mix of prediction questions for another trainer (#2951; g-rl's small oracle, R4): chat records
from generate.py shards, training pieces and training behavior families only.

Each output line: {"messages": [{"role": "user", "content": question}, {"role": "assistant", "content":
answer}], "type", "source", "behavior" (id or null), "piece_split", "cut_semantics", "shard"}. Questions of
held-out pieces or held-out behaviors are dropped; a shard listed with --average-cuts (cut questions measured
with average stand-in writes, before the checker's counterfactual semantics) keeps its other types only.
--per-type caps each question type (drawn uniformly, seeded), so no type dominates.

  mix.py --out MIX.jsonl SHARD.jsonl ... [--average-cuts SHARD.jsonl ...] [--per-type N] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import part_tokens  # noqa: E402

DECOMP = re.compile(r"PD(?:\.vpd)?\[(\d+)\]\.(q_proj|k_proj|v_proj|o_proj|c_fc|down_proj)\[([\d, ]+)\]")


def with_part_tokens(text: str) -> str:
    """Decomposition parts named by their part tokens (one token, or a bracketed list for a group) and whole
    native blocks in mech's generic spelling (L[l].mlp, L[l].attn)."""
    def parts(m):
        toks = [part_tokens.token_of(f"PD[{m[1]}].{m[2]}[{i.strip()}]") for i in m[3].split(",")]
        return toks[0] if len(toks) == 1 else "[" + ", ".join(toks) + "]"

    text = DECOMP.sub(parts, text)
    text = re.sub(r"L\[(\d+)\]\.mlp\[:\]", r"L[\1].mlp", text)
    return re.sub(r"L\[(\d+)\]\.head\[:\]", r"L[\1].attn", text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shards", nargs="*")
    ap.add_argument("--average-cuts", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-type", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--part-tokens", action="store_true", help="name decomposition parts by their part tokens (<p:L.S.I>)")
    args = ap.parse_args()
    by_type = defaultdict(list)
    for path in args.shards + args.average_cuts:
        average = path in args.average_cuts
        for line in open(path):
            try:
                q = json.loads(line)
            except json.JSONDecodeError:  # a shard still being written ends mid-line
                continue
            if q.get("piece_split", "train") != "train" or q.get("split") != "train":
                continue
            if average and q["type"] == "cut":
                continue
            behavior = q["text_id"] if q["source"] == "behavior" else None
            by_type[q["type"]].append({
                "messages": [{"role": "user", "content": with_part_tokens(q["input"]) if args.part_tokens else q["input"]},
                             {"role": "assistant", "content": with_part_tokens(q["answer"]) if args.part_tokens else q["answer"]}],
                "type": q["type"], "source": q["source"], "behavior": behavior, "piece_split": q.get("piece_split", "train"),
                "cut_semantics": ("average" if average else "counterfactual") if q["type"] == "cut" else None, "shard": path})
    rng = random.Random(args.seed)
    out = []
    for kind, items in sorted(by_type.items()):
        rng.shuffle(items)
        out += items[: args.per_type] if args.per_type else items
    rng.shuffle(out)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"out": args.out, "records": len(out), "types": dict(Counter(r["type"] for r in out)),
                      "behaviors": len({r["behavior"] for r in out if r["behavior"]}), "sources": dict(Counter(r["source"] for r in out))}))


if __name__ == "__main__":
    main()
