"""Pythia-70m (GPT-NeoX) as a Target with the same Site semantics as vpd_model.Target.

Sites per layer: attn.{q,k,v}_proj (rows of the fused query_key_value), attn.o_proj (dense),
mlp.c_fc (dense_h_to_4h), mlp.down_proj (dense_4h_to_h). Biases are library constants added after
the site (always on). Parallel residual, LayerNorm with bias, rotary on the first 25% of each head,
exact-erf GELU, untied unembedding.
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parent / "vpd_2951"))
from vpd_model import Site  # noqa: E402

SNAP = Path.home() / ".cache/huggingface/hub/models--EleutherAI--pythia-70m/snapshots/a39f36b100fe8a5377810d56c3f4789b9c53ac42"
KINDS = ("q_proj", "k_proj", "v_proj", "o_proj", "c_fc", "down_proj")


def site_name(i: int, k: str) -> str:
    return f"h.{i}.{'mlp' if k in ('c_fc', 'down_proj') else 'attn'}.{k}"


class Pythia(nn.Module):
    def __init__(self, sd: dict[str, Tensor], cfg: dict):
        super().__init__()
        self.n_layer, self.n_head = cfg["num_hidden_layers"], cfg["num_attention_heads"]
        self.d = cfg["hidden_size"]
        self.hd = self.d // self.n_head
        self.rd = int(self.hd * cfg["rotary_pct"])
        self.eps = cfg["layer_norm_eps"]
        g = lambda k: sd[k].float()
        self.register_buffer("wte", g("gpt_neox.embed_in.weight"))
        self.register_buffer("unembed", g("embed_out.weight"))
        self.register_buffer("lnf_w", g("gpt_neox.final_layer_norm.weight"))
        self.register_buffer("lnf_b", g("gpt_neox.final_layer_norm.bias"))
        self.sites = nn.ModuleDict()
        self.bias: dict[str, Tensor] = {}
        self.ln: list[tuple[Tensor, ...]] = []
        H, hd = self.n_head, self.hd
        for i in range(self.n_layer):
            p = f"gpt_neox.layers.{i}."
            self.ln.append(tuple(g(p + k) for k in ("input_layernorm.weight", "input_layernorm.bias",
                                                     "post_attention_layernorm.weight", "post_attention_layernorm.bias")))
            W = g(p + "attention.query_key_value.weight").view(H, 3, hd, self.d)
            b = g(p + "attention.query_key_value.bias").view(H, 3, hd)
            for j, k in enumerate(("q_proj", "k_proj", "v_proj")):
                self._add(site_name(i, k), W[:, j].reshape(H * hd, self.d).contiguous(), b[:, j].reshape(-1).contiguous())
            self._add(site_name(i, "o_proj"), g(p + "attention.dense.weight"), g(p + "attention.dense.bias"))
            self._add(site_name(i, "c_fc"), g(p + "mlp.dense_h_to_4h.weight"), g(p + "mlp.dense_h_to_4h.bias"))
            self._add(site_name(i, "down_proj"), g(p + "mlp.dense_4h_to_h.weight"), g(p + "mlp.dense_4h_to_h.bias"))
        inv = 1.0 / (cfg["rotary_emb_base"] ** (torch.arange(0, self.rd, 2).float() / self.rd))
        ang = torch.arange(2048).float()[:, None] * inv[None, :]
        ang = torch.cat([ang, ang], -1)
        self.register_buffer("cos", ang.cos())
        self.register_buffer("sin", ang.sin())

    def _add(self, name: str, W: Tensor, b: Tensor):
        self.sites[name.replace(".", "-")] = Site(W)
        self.bias[name] = b

    def site(self, name: str) -> Site:
        return self.sites[name.replace(".", "-")]  # type: ignore[return-value]

    def to(self, dev):  # biases and norms live in plain containers
        super().to(dev)
        self.bias = {k: v.to(dev) for k, v in self.bias.items()}
        self.ln = [tuple(t.to(dev) for t in x) for x in self.ln]
        return self

    def _rope(self, x: Tensor, T: int) -> Tensor:
        r, n = self.rd, self.rd // 2
        xr, xp = x[..., :r], x[..., r:]
        rot = torch.cat([-xr[..., n:], xr[..., :n]], -1)
        return torch.cat([xr * self.cos[:T] + rot * self.sin[:T], xp], -1)

    def forward(self, ids: Tensor) -> Tensor:
        B, T = ids.shape
        x = self.wte[ids]
        for i in range(self.n_layer):
            def s(k, h, i=i):
                n = site_name(i, k)
                return self.site(n)(h) + self.bias[n]
            w1, b1, w2, b2 = self.ln[i]
            h = F.layer_norm(x, (self.d,), w1, b1, self.eps)
            sp = lambda t: t.view(B, T, self.n_head, self.hd).transpose(1, 2)
            q, k, v = sp(s("q_proj", h)), sp(s("k_proj", h)), sp(s("v_proj", h))
            q, k = self._rope(q, T), self._rope(k, T)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            attn = s("o_proj", y.transpose(1, 2).reshape(B, T, -1))
            h2 = F.layer_norm(x, (self.d,), w2, b2, self.eps)
            mlp = s("down_proj", F.gelu(s("c_fc", h2)))
            x = x + attn + mlp
        x = F.layer_norm(x, (self.d,), self.lnf_w, self.lnf_b, self.eps)
        return x @ self.unembed.T


def load_pythia(device: str = "mps") -> Pythia:
    cfg = json.loads((SNAP / "config.json").read_text())
    sd = load_file(str(SNAP / "model.safetensors"))
    return Pythia(sd, cfg).to(device).eval()


if __name__ == "__main__":  # check against transformers' GPTNeoX on a few Pile rows
    from transformers import GPTNeoXForCausalLM

    from vpd_model import val_tokens

    ids = val_tokens(2, seq=128, offset=1024)
    m = load_pythia("cpu")
    with torch.no_grad():
        ours = m(ids)
        ref = GPTNeoXForCausalLM.from_pretrained(str(SNAP), torch_dtype=torch.float32).eval()(ids).logits
    lp = F.log_softmax(ref, -1)
    print("max |logit diff|", (ours - ref).abs().max().item(), "ref scale", ref.abs().max().item())
    print("CE", F.cross_entropy(ref[:, :-1].flatten(0, 1), ids[:, 1:].flatten()).item(),
          "KL", (lp.exp() * (lp - F.log_softmax(ours, -1))).sum(-1).mean().item())
