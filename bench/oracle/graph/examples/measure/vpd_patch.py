"""Which components carry a vpd4l behavior's answer (qwen_patch.py's measurement on vpd4l): for every
head and every MLP, the model runs on each prompt's counterfactual with that one component's output
taken from the clean prompt (every position), and recovery = KL(M_clean || M_counterfactual) -
KL(M_clean || M_patched), bits per target token, full vocabulary.

  MPD_MEM_GIB=6 mem-lease 6 ~/mpd-data/venv/bin/python vpd_patch.py BEHAVIOR.json OUT.json
"""

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # bench/oracle
import vpd_labels as VL  # noqa: E402
import vpd_model as VM  # noqa: E402


@torch.no_grad()
def main():
    behavior, out = Path(sys.argv[1]), Path(sys.argv[2])
    dev = VL.device()
    t = VL.Model(dev).t
    H, D = t.n_head, t.hd
    beh = json.loads(behavior.read_text())
    prompts = [p for p in beh["prompts"] if p.get("counterfactual")]
    T = max(len(p["token_ids"]) for p in prompts)

    def batch(key):
        ids = torch.zeros(len(prompts), T, dtype=torch.long)
        for i, p in enumerate(prompts):
            x = (p if key == "clean" else p["counterfactual"])["token_ids"]
            ids[i, : len(x)] = torch.tensor(x)
        return ids.to(dev)

    clean, counterfactual = batch("clean"), batch("cf")
    rows = torch.tensor([i for i, p in enumerate(prompts) for _ in p["target_positions"]], device=dev)
    cols = torch.tensor([c for p in prompts for c in p["target_positions"]], device=dev)
    saved = {}

    def forward(ids, record=False, head=None, mlp=None):
        """Log-probabilities at the targets; `head` = (layer, h) or `mlp` = layer takes its output from
        the recorded clean run."""
        x = t.wte[ids]
        B, L = ids.shape
        for i in range(t.n_layer):
            site = lambda kind, h: t.site(f"h.{i}.{'mlp' if kind in ('c_fc', 'down_proj') else 'attn'}.{kind}")(h)  # noqa: E731
            h = VM.rms(x, t.norms[2 * i], t.eps)
            q = site("q_proj", h).view(B, L, H, D).transpose(1, 2)
            k = site("k_proj", h).view(B, L, H, D).transpose(1, 2)
            v = site("v_proj", h).view(B, L, H, D).transpose(1, 2)
            q, k = t._rope(q, L), t._rope(k, L)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, L, -1)
            if record:
                saved[("attn", i)] = y
            elif head is not None and head[0] == i:
                y = y.clone()
                y[..., head[1] * D : (head[1] + 1) * D] = saved[("attn", i)][..., head[1] * D : (head[1] + 1) * D]
            x = x + site("o_proj", y)
            m = site("down_proj", VM.gelu_tanh(site("c_fc", VM.rms(x, t.norms[2 * i + 1], t.eps))))
            if record:
                saved[("mlp", i)] = m
            elif mlp == i:
                m = saved[("mlp", i)]
            x = x + m
        h = VM.rms(x, t.ln_f, t.eps)[rows, cols]
        return torch.log_softmax((h @ t.wte.T).float(), -1)

    lp_clean = forward(clean, record=True)
    p_clean = lp_clean.exp()

    def kl(lp):
        return ((p_clean * (lp_clean - lp)).sum(-1).mean() / math.log(2)).item()

    base = kl(forward(counterfactual))
    print(f"{beh['id']}: KL(clean || counterfactual) = {base:.3f} bits per target", flush=True)
    heads = [{"layer": l, "head": h, "recovery_bits": base - kl(forward(counterfactual, head=(l, h)))}
             for l in range(t.n_layer) for h in range(H)]
    mlps = [{"layer": l, "recovery_bits": base - kl(forward(counterfactual, mlp=l))} for l in range(t.n_layer)]
    for x in sorted(heads, key=lambda x: -x["recovery_bits"])[:8]:
        print(f"L{x['layer']}.H{x['head']} {x['recovery_bits']:.3f}")
    print("mlps", [round(m["recovery_bits"], 3) for m in mlps])
    out.write_text(json.dumps({"behavior": beh["id"], "base_bits": base, "heads": heads, "mlps": mlps}))


if __name__ == "__main__":
    main()
