"""Where chosen Qwen3-0.6B heads attend from the target position (MPD #2951 graph oracle, R6): evidence for the roles
in a found program's plain-words mechanism.

For each prompt, the target position's attention (eager attention, head-averaged over nothing: one query head at a time)
is summed over token classes found in the prompt itself:
  ioi        io = the answer name's earlier occurrence, s1 / s2 = the first / second occurrence of the subject (the
             counterfactual's answer), other = the rest
  induction  copy = the earlier occurrence of the answer token (the token after the earlier copy of the current token),
             prev = the position before the target, other = the rest
Prints and writes OUT/<behavior>.attention.json: per head, the mean attention per class.

    attention_roles.py BEHAVIOR.json L.H [L.H ...] [--kind ioi|induction] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def classes(p: dict, kind: str) -> dict[str, list[int]]:
    ids, t = p["token_ids"], p["target_positions"][0]
    ans, cf = ids[t + 1], p["counterfactual"]["token_ids"][t + 1]
    ctx = ids[:t + 1]
    if kind == "ioi":
        io = [i for i, x in enumerate(ctx) if x == ans]
        s = [i for i, x in enumerate(ctx) if x == cf]
        return {"io": io, "s1": s[:1], "s2": s[1:2]}
    copy = [i for i, x in enumerate(ctx) if x == ans]
    return {"copy": copy, "prev": [t - 1]}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("behavior", type=Path)
    ap.add_argument("heads", nargs="+")
    ap.add_argument("--kind", default="ioi", choices=["ioi", "induction"])
    ap.add_argument("--out", type=Path, default=Path.home() / "mpd-data/graph_oracle/runs/r6")
    a = ap.parse_args()
    beh = json.loads(a.behavior.expanduser().read_text())
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32, attn_implementation="eager").eval()
    heads = [tuple(map(int, h.split("."))) for h in a.heads]
    sums = {h: {} for h in heads}
    n = 0
    for p in beh["prompts"]:
        t = p["target_positions"][0]
        out = model(torch.tensor([p["token_ids"][:t + 1]]), output_attentions=True)
        cl = classes(p, a.kind)
        for (l, h) in heads:
            row = out.attentions[l][0, h, t]
            used = set()
            for name, pos in cl.items():
                v = float(row[pos].sum()) if pos else 0.0
                sums[(l, h)][name] = sums[(l, h)].get(name, 0.0) + v
                used |= set(pos)
            sums[(l, h)]["other"] = sums[(l, h)].get("other", 0.0) + float(row.sum()) - sum(float(row[i]) for i in used)
        n += 1
    res = {f"L{l}.H{h}": {k: round(v / n, 4) for k, v in d.items()} for (l, h), d in sums.items()}
    for k, v in res.items():
        print(k, v, flush=True)
    a.out.expanduser().mkdir(parents=True, exist_ok=True)
    (a.out.expanduser() / f"{beh['id']}.attention.json").write_text(json.dumps({"behavior": beh["id"], "kind": a.kind, "prompts": n, "heads": res}, indent=1))


if __name__ == "__main__":
    main()
