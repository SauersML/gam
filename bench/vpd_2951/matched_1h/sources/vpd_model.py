"""Thin loader for Goodfire's 4L Pile LlamaSimpleMLP target (goodfire/spd/runs/t-9d2b8f02) and
the VPD paper decomposition of it (goodfire/spd/runs/s-55ea3f9b, model_400000.pth).

Everything here mirrors the torch code at the `vpd-paper` tag of github.com/goodfire-ai/spd
(param_decomp/pretrain/models/llama_simple_mlp.py, param_decomp/models/components.py,
param_decomp/metrics/{ce_and_kl_losses,pgd_utils}.py) and nano_param_decomp/run.py.

Sites: h.{i}.attn.{q,k,v,o}_proj, h.{i}.mlp.{c_fc,down_proj}. A site's weight is replaced by
    y = ((x @ V) * mask) @ U + delta_mask * (x @ (W - (V @ U).T).T)
with mask in [0,1]^C per (batch, pos), delta_mask in [0,1] per (batch, pos).
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file
from torch import Tensor, nn

HERE = Path.home() / "mpd-data/vpd"  # the downloaded runs and val rows; the scripts run with this as cwd
TARGET_DIR = HERE / "t-9d2b8f02"
VPD_PTH = HERE / "s-55ea3f9b" / "model_400000.pth"
DATA = HERE / "pile_val_4096x513.npy"
KINDS = ("q_proj", "k_proj", "v_proj", "o_proj", "c_fc", "down_proj")


def site_names(n_layer: int = 4) -> list[str]:
    return [
        f"h.{i}.{'attn' if k.endswith('_proj') and k != 'down_proj' else 'mlp'}.{k}"
        for i in range(n_layer)
        for k in KINDS
    ]


# ------------------------------------------------------------------ target


class Site(nn.Module):
    """One decomposable linear map. Target mode: x @ W.T (caches x). Component mode: masked
    subcomponent forward plus the delta component."""

    def __init__(self, W: Tensor):
        super().__init__()
        self.register_buffer("W", W)
        self.V: Tensor | None = None  # [d_in, C]
        self.U: Tensor | None = None  # [C, d_out]
        self.mask: Tensor | None = None
        self.delta_mask: Tensor | None = None
        self.last_input: Tensor | None = None
        self.last_output: Tensor | None = None
        self.cache_input = False
        self.cache_output = False
        self.in_fn = None  # optional input transform (pruning baselines ablate neurons / heads here)

    def forward(self, x: Tensor) -> Tensor:
        if self.in_fn is not None:
            x = self.in_fn(x)
        if self.cache_input:
            self.last_input = x.detach()
        out = self._forward(x)
        if self.cache_output:
            self.last_output = out.detach()
        return out

    def _forward(self, x: Tensor) -> Tensor:
        if self.mask is None:
            return x @ self.W.T
        assert self.V is not None and self.U is not None
        out = ((x @ self.V) * self.mask) @ self.U
        if self.delta_mask is not None:
            delta = self.W - (self.V @ self.U).T
            out = out + self.delta_mask.unsqueeze(-1) * (x @ delta.T)
        return out


def rms(x: Tensor, w: Tensor, eps: float) -> Tensor:
    xf = x.float()
    return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def gelu_tanh(x: Tensor) -> Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


class Target(nn.Module):
    """LlamaSimpleMLP: pre-RMSNorm blocks, rotate-half RoPE MHA (6 heads, hd 128), GELU(tanh)
    MLP, tied wte/lm_head, no biases."""

    def __init__(self, sd: dict[str, Tensor], cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.n_layer, self.n_head = cfg["n_layer"], cfg["n_head"]
        self.hd = cfg["n_embd"] // cfg["n_head"]
        assert cfg["n_key_value_heads"] == cfg["n_head"] and cfg["rotary_dim"] == self.hd
        self.eps = float(cfg["rms_norm_eps"])
        self.register_buffer("wte", sd["wte.weight"])
        self.register_buffer("ln_f", sd["ln_f.weight"])
        self.norms = nn.ParameterList()
        self.sites = nn.ModuleDict()
        for i in range(self.n_layer):
            self.norms.append(nn.Parameter(sd[f"h.{i}.rms_1.weight"], requires_grad=False))
            self.norms.append(nn.Parameter(sd[f"h.{i}.rms_2.weight"], requires_grad=False))
        for name in site_names(self.n_layer):
            self.sites[name.replace(".", "-")] = Site(sd[f"{name}.weight"])
        n_ctx, rd = cfg["n_ctx"], self.hd
        pos = torch.arange(n_ctx, dtype=torch.float32)
        freq = float(cfg["rotary_base"]) ** (torch.arange(rd // 2, dtype=torch.float32) / (rd / 2))
        ang = pos[:, None] / freq.repeat(2)[None, :]
        self.register_buffer("cos", ang.cos())
        self.register_buffer("sin", ang.sin())
        for p in self.parameters():
            p.requires_grad_(False)

    def site(self, name: str) -> Site:
        return self.sites[name.replace(".", "-")]  # type: ignore[return-value]

    def _rope(self, x: Tensor, T: int) -> Tensor:
        n = self.hd // 2
        rot = torch.cat([-x[..., n:], x[..., :n]], dim=-1)
        return x * self.cos[:T] + rot * self.sin[:T]

    def attention_pattern(self, q_flat: Tensor, k_flat: Tensor) -> Tensor:
        """Post-softmax causal attention map [B, H, T, T] from one layer's q_proj/k_proj outputs."""
        B, T, _ = q_flat.shape
        q = self._rope(q_flat.view(B, T, self.n_head, self.hd).transpose(1, 2), T)
        k = self._rope(k_flat.view(B, T, self.n_head, self.hd).transpose(1, 2), T)
        z = (q @ k.transpose(-1, -2)) / math.sqrt(self.hd)
        causal = torch.ones(T, T, dtype=torch.bool, device=z.device).tril()
        return z.masked_fill(~causal, float("-inf")).softmax(-1)

    def forward(self, ids: Tensor) -> Tensor:
        return self.hidden(ids) @ self.wte.T

    def hidden(self, ids: Tensor) -> Tensor:
        """The final normed residual stream [B, T, d]; the logits are it times the tied embedding."""
        B, T = ids.shape
        x = self.wte[ids]
        for i in range(self.n_layer):
            s = lambda k: self.site(f"h.{i}.{'mlp' if k in ('c_fc', 'down_proj') else 'attn'}.{k}")
            h = rms(x, self.norms[2 * i], self.eps)
            q = s("q_proj")(h).view(B, T, self.n_head, self.hd).transpose(1, 2)
            k = s("k_proj")(h).view(B, T, self.n_head, self.hd).transpose(1, 2)
            v = s("v_proj")(h).view(B, T, self.n_head, self.hd).transpose(1, 2)
            q, k = self._rope(q, T), self._rope(k, T)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + s("o_proj")(y.transpose(1, 2).reshape(B, T, -1))
            h = rms(x, self.norms[2 * i + 1], self.eps)
            x = x + s("down_proj")(gelu_tanh(s("c_fc")(h)))
        return rms(x, self.ln_f, self.eps)


def load_target(device: str = "mps") -> Target:
    cfg = yaml.safe_load((TARGET_DIR / "model_config.yaml").read_text())
    sd = load_file(str(TARGET_DIR / "model_step_99999.safetensors"))
    sd = {k: v.float() for k, v in sd.items()}
    return Target(sd, cfg).to(device).eval()


# ------------------------------------------------------------------ CI function


class CILinear(nn.Module):
    """param_decomp.models.components.Linear: W [in, out], b [out]."""

    def __init__(self, W: Tensor, b: Tensor):
        super().__init__()
        self.W, self.b = nn.Parameter(W, requires_grad=False), nn.Parameter(b, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.W + self.b


class CIBlock(nn.Module):
    """RMSNorm (no scale) -> bidirectional RoPE MHA -> residual -> RMSNorm -> GELU MLP -> residual."""

    def __init__(self, sd: dict[str, Tensor], pre: str, n_heads: int, rope_base: float, max_len: int):
        super().__init__()
        g = lambda k: nn.Parameter(sd[pre + k], requires_grad=False)
        self.wq, self.wk, self.wv, self.wo = (
            g("attn.q_proj.weight"), g("attn.k_proj.weight"), g("attn.v_proj.weight"),
            g("attn.out_proj.weight"),
        )
        self.fc1 = CILinear(sd[pre + "mlp.0.W"], sd[pre + "mlp.0.b"])
        self.fc2 = CILinear(sd[pre + "mlp.2.W"], sd[pre + "mlp.2.b"])
        self.n_heads = n_heads
        self.d = self.wq.shape[0]
        self.dh = self.d // n_heads
        inv = 1.0 / (rope_base ** (torch.arange(0, self.dh, 2).float() / self.dh))
        ang = torch.arange(max_len).float()[:, None] * inv[None, :]
        self.register_buffer("cos", torch.cat([ang.cos(), ang.cos()], -1))
        self.register_buffer("sin", torch.cat([ang.sin(), ang.sin()], -1))

    def _rope(self, x: Tensor, S: int) -> Tensor:
        n = self.dh // 2
        return x * self.cos[:S] + torch.cat([-x[..., n:], x[..., :n]], -1) * self.sin[:S]

    def forward(self, x: Tensor) -> Tensor:
        B, S, D = x.shape
        h = F.rms_norm(x, (D,))
        sp = lambda t: t.view(B, S, self.n_heads, self.dh).transpose(1, 2)
        q, k, v = sp(h @ self.wq.T), sp(h @ self.wk.T), sp(h @ self.wv.T)
        q, k = self._rope(q, S), self._rope(k, S)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        x = x + a.transpose(1, 2).reshape(B, S, D) @ self.wo.T
        return x + self.fc2(F.gelu(self.fc1(F.rms_norm(x, (D,)))))


class GlobalSharedTransformerCiFn(nn.Module):
    """Concatenate RMS-normed pre-weight activations of every site (sorted by name), project to
    d_model, 8 bidirectional transformer blocks, project to sum(C); split per site."""

    def __init__(self, sd: dict[str, Tensor], order: list[str], cs: list[int], n_heads: int,
                 rope_base: float, max_len: int):
        super().__init__()
        self.order, self.cs = order, cs
        self.proj_in = CILinear(sd["_input_projector.W"], sd["_input_projector.b"])
        self.head = CILinear(sd["_output_head.W"], sd["_output_head.b"])
        n_blocks = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("_blocks."))
        self.blocks = nn.ModuleList(
            CIBlock(sd, f"_blocks.{i}.", n_heads, rope_base, max_len) for i in range(n_blocks)
        )

    def forward(self, acts: dict[str, Tensor]) -> dict[str, Tensor]:
        x = torch.cat([F.rms_norm(acts[n], (acts[n].shape[-1],)) for n in self.order], -1)
        x = self.proj_in(x)
        for b in self.blocks:
            x = b(x)
        return dict(zip(self.order, self.head(x).split(self.cs, -1), strict=True))


def lower_leaky(x: Tensor) -> Tensor:
    """Forward value of the leaky-hard lower sigmoid: clamp(x, 0, 1)."""
    return x.clamp(0.0, 1.0)


class VPD(nn.Module):
    """The published decomposition as an executable object: subcomponents (U, V per site) installed
    into a Target, plus the causal-importance network."""

    def __init__(self, target: Target, ci_fn: GlobalSharedTransformerCiFn, UV: dict[str, tuple[Tensor, Tensor]]):
        super().__init__()
        self.target, self.ci_fn = target, ci_fn
        self.names = list(UV)
        for n, (U, V) in UV.items():
            st = target.site(n)
            st.U, st.V = U, V
        self.C = {n: UV[n][0].shape[0] for n in self.names}

    def clear(self):
        for n in self.names:
            st = self.target.site(n)
            st.mask = st.delta_mask = st.last_input = st.last_output = None
            st.cache_input = st.cache_output = False

    @torch.no_grad()
    def target_forward(self, ids: Tensor) -> Tensor:
        self.clear()
        return self.target(ids)

    @torch.no_grad()
    def target_and_ci(self, ids: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Clean target logits and CI (lower-leaky forward value) per site."""
        self.clear()
        for n in self.names:
            self.target.site(n).cache_input = True
        logits = self.target(ids)
        acts = {n: self.target.site(n).last_input for n in self.names}
        self.clear()
        pre = self.ci_fn(acts)  # type: ignore[arg-type]
        return logits, {n: lower_leaky(v) for n, v in pre.items()}

    def masked(self, ids: Tensor, masks: dict[str, Tensor], delta_masks: dict[str, Tensor] | None) -> Tensor:
        for n in self.names:
            st = self.target.site(n)
            st.mask = masks[n]
            st.delta_mask = None if delta_masks is None else delta_masks[n]
        try:
            return self.target(ids)
        finally:
            self.clear()


def load_vpd(target: Target, device: str = "mps") -> VPD:
    raw = torch.load(str(VPD_PTH), map_location="cpu", weights_only=True, mmap=True)
    UV: dict[str, tuple[Tensor, Tensor]] = {}
    ci_sd: dict[str, Tensor] = {}
    for k, v in raw.items():
        if k.startswith("_components."):
            site, which = k[len("_components."):].rsplit(".", 1)
            name = site.replace("-", ".")
            U, V = UV.get(name, (None, None))
            UV[name] = (v.float() if which == "U" else U, v.float() if which == "V" else V)  # type: ignore[assignment]
        elif "_global_ci_fn." in k:
            ci_sd[k.split("_global_ci_fn.", 1)[1]] = v.float()
    target_weight_diff = max(
        (raw[f"target_model.{n}.weight"] - target.site(n).W.cpu()).abs().max().item() for n in UV
    )
    del raw
    order = sorted(UV)
    UV = {n: (UV[n][0].to(device), UV[n][1].to(device)) for n in site_names()}
    ci = GlobalSharedTransformerCiFn(ci_sd, order, [UV[n][0].shape[0] for n in order],
                                     n_heads=16, rope_base=10000.0, max_len=512)
    vpd = VPD(target, ci.to(device).eval(), UV)
    vpd.target_weight_diff = target_weight_diff
    return vpd


VAL_PARQUET = Path.home() / "mpd-data/hf/datasets--danbraunai--pile-uncopyrighted-tok-shuffled/snapshots/6f141f6e4edc32fea842f2ece0d3477b15c3dc90/data/val-00000-of-00012.parquet"


def val_tokens(n: int, seq: int = 512, offset: int = 0) -> Tensor:
    """Rows [offset, offset + n) of the val split's first shard (pile_val_4096x513.npy is its first
    4096 rows; later rows are read from the parquet, first row group = 48638 rows)."""
    if offset + n <= 4096:
        return torch.from_numpy(np.load(DATA, mmap_mode="r")[offset:offset + n, :seq].astype(np.int64))
    import pyarrow.parquet as pq

    col = pq.ParquetFile(VAL_PARQUET).read_row_group(0, columns=["input_ids"]).column("input_ids")
    sl = col.slice(offset, n).combine_chunks()
    flat = sl.flatten().to_numpy().astype(np.int64)
    width = len(flat) // n
    assert width * n == len(flat), "val rows are not of equal length"
    return torch.from_numpy(flat.reshape(n, width)[:, :seq].copy())
