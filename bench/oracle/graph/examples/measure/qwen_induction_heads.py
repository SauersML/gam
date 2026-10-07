"""Measured head and MLP removals of Qwen3-0.6B on a behavior file:
KL(M || M_e) in bits at the target tokens, full vocabulary, M_e = M with one head's attention output
set to zero (its o_proj columns removed) or one layer's MLP removed; plus each head's attention from
every position to the one before it (previous token; per prompt, then averaged) and from the target
to the token after the earlier copy (induction).

  HF_HUB_OFFLINE=1 MPD_MEM_GIB=8 mem-lease 8 ~/mpd-data/venv/bin/python qwen_induction_heads.py BEHAVIOR.json PERIOD OUT.json
(examples/qwen3_induction_heads.py: induction_random.words8, PERIOD 9; output
~/mpd-data/graph_oracle/experiments/examples/qwen_induction_heads.json)
"""

import json
import math
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

BEHAVIOR, PERIOD, OUT = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])


@torch.no_grad()
def main():
    dev = torch.device("mps")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.float32,
                                                 attn_implementation="eager").to(dev).eval()
    cfg = model.config
    H, D = cfg.num_attention_heads, cfg.head_dim
    beh = json.loads(BEHAVIOR.read_text())
    prompts = beh["prompts"]
    T = max(len(p["token_ids"]) for p in prompts)
    ids = torch.zeros(len(prompts), T, dtype=torch.long)
    for i, p in enumerate(prompts):
        ids[i, : len(p["token_ids"])] = torch.tensor(p["token_ids"])
    ids = ids.to(dev)
    lengths = torch.tensor([len(p["token_ids"]) for p in prompts], device=dev)
    rows = torch.tensor([i for i, p in enumerate(prompts) for _ in p["target_positions"]], device=dev)
    cols = torch.tensor([t for p in prompts for t in p["target_positions"]], device=dev)
    state = {"layer": None, "head": None, "mlp": None}

    def pre_o(layer):
        def hook(module, args):
            if state["layer"] == layer and state["head"] is not None:
                x = args[0].clone()
                x[..., state["head"] * D : (state["head"] + 1) * D] = 0
                return (x,)
        return hook

    def post_mlp(layer):
        def hook(module, args, out):
            if state["mlp"] == layer:
                return torch.zeros_like(out)
        return hook

    for l, block in enumerate(model.model.layers):
        block.self_attn.o_proj.register_forward_pre_hook(pre_o(l))
        block.mlp.register_forward_hook(post_mlp(l))

    def log_probs():
        out = model.model(ids, output_attentions=state["layer"] is None and state["mlp"] is None)
        logits = model.lm_head(out.last_hidden_state[rows, cols])
        return torch.log_softmax(logits.float(), -1), out.attentions

    lp0, attentions = log_probs()
    p0 = lp0.exp()
    gold = ids[rows, cols + 1]
    print(f"clean accuracy {(lp0.argmax(-1) == gold).float().mean().item():.3f}", flush=True)

    def kl(lp):
        return ((p0 * (lp0 - lp)).sum(-1).mean() / math.log(2)).item()

    heads = []
    for l in range(cfg.num_hidden_layers):
        a = attentions[l][rows]  # [n, H, T, T]
        step = a[:, :, torch.arange(1, T), torch.arange(0, T - 1)]  # position p to p - 1, [n, H, T - 1]
        valid = (torch.arange(1, T, device=dev)[None] < lengths[rows][:, None]).float()  # no padding
        prev = ((step * valid[:, None]).sum(2) / valid.sum(1)[:, None]).mean(0)  # per prompt, then mean
        ind = a[torch.arange(len(rows), device=dev), :, cols, cols - PERIOD + 1].mean(0)  # target to after copy
        for h in range(H):
            state.update(layer=l, head=h, mlp=None)
            lp, _ = log_probs()
            heads.append({"layer": l, "head": h, "kl_bits": kl(lp), "prev_attention": prev[h].item(),
                          "induction_attention": ind[h].item()})
        print(l, sorted(((round(x["kl_bits"], 3), x["head"]) for x in heads[-H:]), reverse=True)[:4], flush=True)
    mlps = []
    for l in range(cfg.num_hidden_layers):
        state.update(layer=None, head=None, mlp=l)
        lp, _ = log_probs()
        mlps.append({"layer": l, "kl_bits": kl(lp)})
    OUT.write_text(json.dumps({"behavior": beh["id"], "heads": heads, "mlps": mlps}, indent=0))


if __name__ == "__main__":
    main()
