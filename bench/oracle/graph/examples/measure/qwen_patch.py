"""Which components carry a Qwen3-0.6B behavior's answer: for every head and every MLP, the model runs
on each prompt's counterfactual with that one component's output taken from the clean prompt (every
position; prompts and counterfactuals are aligned token by token), and we measure how far the
target's next-token distribution returns to the clean one: KL(M_clean || M_patched) in bits at the
target tokens, full vocabulary, against KL(M_clean || M_counterfactual) unpatched. Recovery =
the unpatched KL minus the patched KL. These are the pieces a program must declare when undeclared
pieces carry their counterfactual values.

  HF_HUB_OFFLINE=1 MPD_MEM_GIB=8 mem-lease 8 ~/mpd-data/venv/bin/python qwen_patch.py BEHAVIOR.json OUT.json
"""

import json
import math
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


@torch.no_grad()
def main():
    behavior, out = Path(sys.argv[1]), Path(sys.argv[2])
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32).to(dev).eval()
    cfg = model.config
    H, D = cfg.num_attention_heads, cfg.head_dim
    beh = json.loads(behavior.read_text())
    prompts = [p for p in beh["prompts"] if p.get("counterfactual")]
    T = max(len(p["token_ids"]) for p in prompts)

    def batch(key):
        ids = torch.zeros(len(prompts), T, dtype=torch.long)
        for i, p in enumerate(prompts):
            t = (p if key == "clean" else p["counterfactual"])["token_ids"]
            ids[i, : len(t)] = torch.tensor(t)
        return ids.to(dev)

    clean, counterfactual = batch("clean"), batch("cf")
    rows = torch.tensor([i for i, p in enumerate(prompts) for _ in p["target_positions"]], device=dev)
    cols = torch.tensor([t for p in prompts for t in p["target_positions"]], device=dev)
    state = {"record": False, "head": None, "mlp": None}
    saved_o, saved_mlp = {}, {}

    def pre_o(layer):
        def hook(module, args):
            if state["record"]:
                saved_o[layer] = args[0].clone()
            elif state["head"] is not None and state["head"][0] == layer:
                h = state["head"][1]
                x = args[0].clone()
                x[..., h * D : (h + 1) * D] = saved_o[layer][..., h * D : (h + 1) * D]
                return (x,)
        return hook

    def post_mlp(layer):
        def hook(module, args, output):
            if state["record"]:
                saved_mlp[layer] = output.clone()
            elif state["mlp"] == layer:
                return saved_mlp[layer]
        return hook

    for l, block in enumerate(model.model.layers):
        block.self_attn.o_proj.register_forward_pre_hook(pre_o(l))
        block.mlp.register_forward_hook(post_mlp(l))

    def log_probs(ids):
        hidden = model.model(ids).last_hidden_state[rows, cols]
        return torch.log_softmax(model.lm_head(hidden).float(), -1)

    state["record"] = True
    lp_clean = log_probs(clean)
    state["record"] = False
    p_clean = lp_clean.exp()

    def kl(lp):
        return ((p_clean * (lp_clean - lp)).sum(-1).mean() / math.log(2)).item()

    base = kl(log_probs(counterfactual))
    print(f"{beh['id']}: KL(clean || counterfactual) = {base:.3f} bits per target", flush=True)
    heads, mlps = [], []
    for l in range(cfg.num_hidden_layers):
        for h in range(H):
            state.update(head=(l, h), mlp=None)
            heads.append({"layer": l, "head": h, "recovery_bits": base - kl(log_probs(counterfactual))})
        state.update(head=None, mlp=l)
        mlps.append({"layer": l, "recovery_bits": base - kl(log_probs(counterfactual))})
        state.update(mlp=None)
        best = sorted(heads[-H:], key=lambda x: -x["recovery_bits"])[:3]
        print(l, [(x["head"], round(x["recovery_bits"], 3)) for x in best], "mlp", round(mlps[-1]["recovery_bits"], 3), flush=True)
        out.write_text(json.dumps({"behavior": beh["id"], "base_bits": base, "heads": heads, "mlps": mlps}))


if __name__ == "__main__":
    main()
