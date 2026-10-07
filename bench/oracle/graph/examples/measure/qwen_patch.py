"""Which components carry a Qwen3-0.6B behavior's answer: for every head and every MLP, the model runs
on each prompt's counterfactual with that one component's output taken from the clean prompt (every
position; prompts and counterfactuals are aligned token by token), and we measure how far the
target's next-token distribution returns to the clean one: KL(M_clean || M_patched) in bits at the
target tokens, full vocabulary, against KL(M_clean || M_counterfactual) unpatched. Recovery =
the unpatched KL minus the patched KL. These are the pieces a program must declare when undeclared
pieces carry their counterfactual values.

  HF_HUB_OFFLINE=1 MPD_MEM_GIB=8 mem-lease 8 ~/mpd-data/venv/bin/python qwen_patch.py OUT_DIR BEHAVIOR.json [...]
(OUT_DIR/patch_<behavior id>.json per behavior)
"""

import json
import math
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


@torch.no_grad()
def main():
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32).to(dev).eval()
    for behavior in sys.argv[2:]:
        beh = json.loads(Path(behavior).read_text())
        if any(p.get("counterfactual") for p in beh["prompts"]):
            try:
                patch(model, dev, beh, out_dir / f"patch_{beh['id']}.json")
            except torch.OutOfMemoryError as e:  # one behavior too large: skip it, keep the others
                print(beh["id"], "skipped:", e, flush=True)
                for block in model.model.layers:  # drop the failed behavior's hooks
                    block.self_attn.o_proj._forward_pre_hooks.clear()
                    block.mlp._forward_hooks.clear()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


TOKENS = 16384  # tokens per forward batch: prompts are processed in chunks of TOKENS // length


def patch(model, dev, beh: dict, out: Path):
    cfg = model.config
    H, D, layers = cfg.num_attention_heads, cfg.head_dim, cfg.num_hidden_layers
    prompts = [p for p in beh["prompts"] if p.get("counterfactual")]
    T = max(len(p["token_ids"]) for p in prompts)
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

    handles = []
    for l, block in enumerate(model.model.layers):
        handles.append(block.self_attn.o_proj.register_forward_pre_hook(pre_o(l)))
        handles.append(block.mlp.register_forward_hook(post_mlp(l)))
    # KL sums over target tokens: index 0 the unpatched counterfactual, then every head, then every MLP
    sums = torch.zeros(1 + layers * H + layers, dtype=torch.float64)
    count = 0
    chunk = max(1, TOKENS // T)
    for s in range(0, len(prompts), chunk):
        part = prompts[s : s + chunk]

        def batch(key):
            ids = torch.zeros(len(part), T, dtype=torch.long)
            for i, p in enumerate(part):
                t = (p if key == "clean" else p["counterfactual"])["token_ids"]
                ids[i, : len(t)] = torch.tensor(t)
            return ids.to(dev)

        clean, counterfactual = batch("clean"), batch("cf")
        rows = torch.tensor([i for i, p in enumerate(part) for _ in p["target_positions"]], device=dev)
        cols = torch.tensor([t for p in part for t in p["target_positions"]], device=dev)

        def log_probs(ids):
            hidden = model.model(ids).last_hidden_state[rows, cols]
            return torch.log_softmax(model.lm_head(hidden).float(), -1)

        state.update(record=True, head=None, mlp=None)
        lp_clean = log_probs(clean)
        state["record"] = False
        p_clean = lp_clean.exp()

        def kl(lp):
            return ((p_clean * (lp_clean - lp)).sum(-1).sum() / math.log(2)).item()

        sums[0] += kl(log_probs(counterfactual))
        k = 1
        for l in range(layers):
            for h in range(H):
                state.update(head=(l, h), mlp=None)
                sums[k] += kl(log_probs(counterfactual))
                k += 1
        for l in range(layers):
            state.update(head=None, mlp=l)
            sums[k] += kl(log_probs(counterfactual))
            k += 1
        state.update(mlp=None)
        count += len(rows)
        saved_o.clear()
        saved_mlp.clear()
    for h in handles:
        h.remove()
    mean = (sums / count).tolist()
    base = mean[0]
    heads = [{"layer": l, "head": h, "recovery_bits": base - mean[1 + l * H + h]} for l in range(layers) for h in range(H)]
    mlps = [{"layer": l, "recovery_bits": base - mean[1 + layers * H + l]} for l in range(layers)]
    best = sorted(heads, key=lambda x: -x["recovery_bits"])[:6]
    print(f"{beh['id']}: KL(clean || counterfactual) = {base:.3f} bits per target;",
          [(f"L{x['layer']}.H{x['head']}", round(x["recovery_bits"], 3)) for x in best], flush=True)
    out.write_text(json.dumps({"behavior": beh["id"], "base_bits": base, "heads": heads, "mlps": mlps}))


if __name__ == "__main__":
    main()
