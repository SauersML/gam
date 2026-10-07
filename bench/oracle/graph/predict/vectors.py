"""The vector channel's input (#2951): for each part of a fixed set, K = 8 directions of the target model's
residual stream with their scales, read from its weights, which sft.py and eval_kl.py map into the oracle's
embedding space as soft tokens after the part's question.

Per part (Qwen3 weights; g_in the attention or MLP input norm's gain, q_n and k_n the query and key norms'
gains; singular pairs (u, s, v) of the largest singular values, signs fixed so each vector's largest entry is
positive):
  head L[l].head[h]   OV = W_O[:, h] W_V[kv(h)] diag(g_in): u1, u2 (what it writes), v1, v2 (what it reads
                      into its value); QK = (diag(q_n) W_Q[h] diag(g_in))^T (diag(k_n) W_K[kv(h)] diag(g_in)):
                      left 1, 2 (query side), right 1, 2 (key side)
  attention L[l].head[:]   the same for the sums over the layer's heads
  MLP L[l].mlp[:]     W_down: u1..u4 (what it writes); [W_gate; W_up] diag(g_in): v1..v4 (what it reads)
Each soft token's input is [v * sqrt(d), log s] (d the residual width): direction and scale.

  vectors.py --model TARGET_DIR --pieces PIECES.json --out VECTORS.safetensors
"""

from __future__ import annotations

import argparse
import json
import re

import torch

K = 8


def svd_top(m: torch.Tensor, k: int):
    """The k largest singular pairs of m (float64 on the device), signs fixed by each left vector's largest entry."""
    u, s, vh = torch.linalg.svd(m.double(), full_matrices=False)
    u, s, v = u[:, :k], s[:k], vh[:k].T
    sign = torch.sign(u.gather(0, u.abs().argmax(0, keepdim=True)))[0]
    return u * sign, s, v * sign


def part_vectors(model, piece: str) -> torch.Tensor:
    """[K, d + 1]: the part's directions times sqrt(d) and the log of their singular values."""
    inner = model.model
    c = model.config
    H, KV, hd, d = c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.hidden_size
    rep = H // KV

    def head_mats(layer, h):
        a = layer.self_attn
        g = layer.input_layernorm.weight.double()
        wq = a.q_proj.weight.double().view(H, hd, d)[h] * a.q_norm.weight.double()[:, None] * g
        wk = a.k_proj.weight.double().view(KV, hd, d)[h // rep] * a.k_norm.weight.double()[:, None] * g
        wv = a.v_proj.weight.double().view(KV, hd, d)[h // rep] * g
        wo = a.o_proj.weight.double().view(d, H, hd)[:, h]
        return wo @ wv, wq.T @ wk

    vecs = []
    if m := re.fullmatch(r"L\[(\d+)\]\.head\[(\d+)\]", piece):
        ov, qk = head_mats(inner.layers[int(m[1])], int(m[2]))
    elif m := re.fullmatch(r"L\[(\d+)\]\.head\[:\]", piece):
        pairs = [head_mats(inner.layers[int(m[1])], h) for h in range(H)]
        ov, qk = sum(p[0] for p in pairs), sum(p[1] for p in pairs)
    elif m := re.fullmatch(r"L\[(\d+)\]\.mlp\[:\]", piece):
        layer = inner.layers[int(m[1])]
        g = layer.post_attention_layernorm.weight.double()
        w_in = torch.cat([layer.mlp.gate_proj.weight.double(), layer.mlp.up_proj.weight.double()]) * g
        u, s, _ = svd_top(layer.mlp.down_proj.weight, 4)
        _, s2, v = svd_top(w_in, 4)
        vecs = [(u[:, i], s[i]) for i in range(4)] + [(v[:, i], s2[i]) for i in range(4)]
    else:
        raise ValueError(piece)
    if not vecs:
        u, s, v = svd_top(ov, 2)
        lq, sq, rk = svd_top(qk, 2)
        vecs = [(u[:, 0], s[0]), (u[:, 1], s[1]), (v[:, 0], s[0]), (v[:, 1], s[1]),
                (lq[:, 0], sq[0]), (lq[:, 1], sq[1]), (rk[:, 0], sq[0]), (rk[:, 1], sq[1])]
    return torch.stack([torch.cat([x * d**0.5, torch.log(sv.clamp(min=1e-12))[None]]) for x, sv in vecs]).float()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pieces", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev).float().eval()
    pieces = json.load(open(args.pieces))
    with torch.no_grad():
        table = torch.stack([part_vectors(model, p) for p in pieces]).cpu()
    save_file({"vectors": table.contiguous()}, args.out, metadata={"pieces": json.dumps(pieces)})
    print(json.dumps({"parts": len(pieces), "shape": list(table.shape)}))


if __name__ == "__main__":
    main()
