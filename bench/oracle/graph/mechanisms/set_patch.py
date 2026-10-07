"""How much of a Qwen3-0.6B behavior a set of components carries (MPD #2951 graph oracle, R6).

The counterfactual prompt x' runs with the clean outputs of the top-k components of a patching ranking
(examples/measure/qwen_patch.py) put in place together: each chosen head's slice of its layer's o_proj input and each
chosen MLP's output take their values from the clean run on x, everything else computes on x'. The distance
KL(M(x) || patched x') in bits at the targets, against KL(M(x) || M(x')), says how much of the behavior the set
carries (sufficiency). k runs over a doubling schedule.

    set_patch.py BEHAVIOR.json PATCH.json [--ks 0,1,2,4,8,16,32,64] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def ranking(patch: dict) -> list[tuple]:
    units = [(h["recovery_bits"], ("head", h["layer"], h["head"])) for h in patch["heads"]]
    units += [(m["recovery_bits"], ("mlp", m["layer"])) for m in patch["mlps"]]
    return [u for r, u in sorted(units, key=lambda x: -x[0]) if r > 0]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("behavior", type=Path)
    ap.add_argument("patch", type=Path)
    ap.add_argument("--ks", default="0,1,2,4,8,16,32,64")
    ap.add_argument("--out", type=Path, default=Path.home() / "mpd-data/graph_oracle/runs/r6")
    a = ap.parse_args()
    beh = json.loads(a.behavior.expanduser().read_text())
    rank = ranking(json.loads(a.patch.expanduser().read_text()))
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32).to(dev).eval()
    H, D = model.config.num_attention_heads, model.config.head_dim
    state = {"record": False, "heads": {}, "mlps": set()}
    saved_o, saved_mlp = {}, {}

    def pre_o(layer):
        def hook(module, args):
            if state["record"]:
                saved_o[layer] = args[0].clone()
            elif state["heads"].get(layer):
                x = args[0].clone()
                xv, sv = x.view(*x.shape[:2], H, D), saved_o[layer].view(*x.shape[:2], H, D)
                for h in state["heads"][layer]:
                    xv[:, :, h] = sv[:, :, h]
                return (x,)
        return hook

    def post_mlp(layer):
        def hook(module, args, output):
            if state["record"]:
                saved_mlp[layer] = output.clone()
            elif layer in state["mlps"]:
                return saved_mlp[layer]
        return hook

    for l, block in enumerate(model.model.layers):
        block.self_attn.o_proj.register_forward_pre_hook(pre_o(l))
        block.mlp.register_forward_hook(post_mlp(l))
    ps = beh["prompts"]
    T = max(len(p["token_ids"]) for p in ps)
    ks = [int(k) for k in a.ks.split(",") if int(k) <= len(rank)]
    sums = {k: 0.0 for k in ks}
    count = 0
    for s in range(0, len(ps), 16):
        part = ps[s:s + 16]
        ids = {w: torch.zeros(len(part), T, dtype=torch.long) for w in ("x", "cf")}
        for i, p in enumerate(part):
            ids["x"][i, :len(p["token_ids"])] = torch.tensor(p["token_ids"])
            ids["cf"][i, :len(p["counterfactual"]["token_ids"])] = torch.tensor(p["counterfactual"]["token_ids"])
        rows = torch.tensor([i for i, p in enumerate(part) for _ in p["target_positions"]], device=dev)
        cols = torch.tensor([t for p in part for t in p["target_positions"]], device=dev)

        def lp_of(x):
            hdn = model.model(x.to(dev)).last_hidden_state[rows, cols]
            return torch.log_softmax(model.lm_head(hdn).float(), -1)

        state.update(record=True, heads={}, mlps=set())
        lpx = lp_of(ids["x"])
        state["record"] = False
        for k in ks:
            heads, mlps = {}, set()
            for u in rank[:k]:
                if u[0] == "head":
                    heads.setdefault(u[1], []).append(u[2])
                else:
                    mlps.add(u[1])
            state.update(heads=heads, mlps=mlps)
            lpp = lp_of(ids["cf"])
            sums[k] += float(((lpx.exp() * (lpx - lpp)).sum(-1) / math.log(2)).sum())
        state.update(heads={}, mlps=set())
        count += len(rows)
    base = sums[0] / count
    res = {"behavior": beh["id"], "base_bits": base, "targets": count,
           "sets": [{"k": k, "units": [f"L{u[1]}.H{u[2]}" if u[0] == "head" else f"MLP{u[1]}" for u in rank[:k]],
                     "kl_bits": sums[k] / count, "carried": 1 - sums[k] / count / base} for k in ks]}
    for r in res["sets"]:
        print(f"{beh['id']} k={r['k']:3d}: KL {r['kl_bits']:.3f} bits of {base:.3f} -> carries {100 * r['carried']:.1f}%  {r['units'][-4:]}", flush=True)
    a.out.expanduser().mkdir(parents=True, exist_ok=True)
    (a.out.expanduser() / f"{beh['id']}.set_patch.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
