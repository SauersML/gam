"""Prediction training data for the graph oracle (#2951): causal questions about the target model M, each
with an answer measured by an exact intervention on M, written as text for the oracle (its input and the
answer it is trained to write) plus the measured numbers.

Interventions are written with mech's addresses (bench/oracle/graph/mech.py: pieces, node(...), routes,
logits) inside three question verbs, scale / cut / swap. Pieces of Qwen3-0.6B (native view):
  L[l].head[h]          head h of layer l. Edits act on its write: scale(L[l].head[h], a) multiplies the
                        head's attention read z_h (the input of its o_proj columns) by a, which is
                        oracle.rs's Component::Head, W_o[:, h] <- a W_o[:, h].
  L[l].mlp[i, j, ...]   MLP neurons i, j, ...: scale multiplies their activations SiLU(g_i.x)(u_i.x) by a,
                        i.e. their down columns (oracle.rs's Component::Neuron, summed).
  L[l].mlp[:], L[l].head[:]   the whole MLP (every neuron) or the whole attention (every head) of layer l.
  PD.tc[l][f, ...]      transcoder features (circuit-tracer's Qwen3-0.6B transcoders, --transcoders): the
                        MLP output is the features' writes plus an exact error piece, so scale adds
                        (a - 1) relu(W_enc[f].y + b_enc[f]) W_dec[f] to the MLP output at positions 1..
                        (y the MLP's input); swap replaces the feature's activation at the last position.
  a = 0 removes the piece. A weight edit acts at every position (and every generated step).
  cut(node(A) >> node(B).route)
                        path patching: B's read (route query, key or value of a head, input of an MLP,
                        or logits = the final residual) receives A's AVERAGE write (over this shard's
                        texts, every position after the first) in place of A's actual write; every
                        other reader keeps A's actual write.
  swap(P, source)       P's value at the last position (a head's z_h, or the neurons' activations) is the
                        value P computes at the last position of the source text.

Question types (one of each per text, all on the next-token distribution at the text's last token unless
stated; distributions are written as the top 5 tokens with probabilities and the remaining mass "other";
changes as KL(M || M_e) in bits, the clean distribution first):
  plain     M's next-token distribution.
  edit      scale one piece by a in {0, 0.5, 2} -> the edited distribution and its KL.
  rank      which of 4 listed pieces' removal changes the distribution most -> the order and each KL.
  where     where in the text a piece is most active -> the 3 positions of largest activity and the level
            (a neuron's activation; a head's write norm ||z_h W_o,h^T||), position 0 excluded (attention sink).
  continue  greedy continuation of S tokens with a piece scaled by 2 or 4 (the clean continuation given).
  cut       cut an edge A >> B (A a head, MLP or attention; B a head route, an MLP input, or logits = the
            final residual before the norm) -> the new distribution and its KL.
  prompt    replace one token of the text -> the new distribution and its KL.
  swap      swap a piece's value from another text -> the new distribution and its KL.
Pieces are drawn half uniformly and half among the pieces most active at the text's last position (a
neuron's |activation|, a head's write norm): activity only chooses what to ask, every answer is measured.

Execution: Qwen3's decoder written out layer by layer (the Hugging Face modules' weights, float32 matmuls,
TF32 off, log-probabilities normalized in float64 where the device has it), every intervention batched per
text. `parity.py` checks it against oracle.rs's float64 host execution.

  generate.py --model SNAPSHOT --windows WINDOWS_T128.u32 --out SHARD.jsonl [--texts 4096] [--batch 32]
              [--lengths 24,32,48,64] [--offset 0] [--seed 0] [--behaviors DIR] [--steps 8] [--split train]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

TOP = 5
MODEL_NAME = "qwen3-0.6b"
ROUTES = ("query", "key", "value")


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Interventions:
    """Per-text interventions of one batched forward: head and neuron multipliers [B, L, H] / [B, L, F],
    last-position swaps (masks and source values), and at most one cut per text."""

    def __init__(self, B, L, H, Fn, dev, dtype):
        self.head = torch.ones(B, L, H, device=dev, dtype=dtype)
        self.neuron = torch.ones(B, L, Fn, device=dev, dtype=dtype)
        self.head_swap = None  # (mask [B, L, H] bool, values [B, L, H, hd])
        self.neuron_swap = None  # (mask [B, L, F] bool, values [B, L, F])
        self.cuts = {}  # row -> (a_kind head|mlp|attn, a_layer, a_head, b_kind head|mlp|logits, b_layer, b_head, route)
        self.tc_scale = {}  # row -> (layer, feature indices [k], alpha)
        self.tc_swap = {}  # row -> (layer, feature indices [k], source activations [k])
        self.parts_scale = {}  # row -> (layer, site, subcomponent indices [k], alpha)   (vpd4l's VPD view)
        self.parts_swap = {}  # row -> (layer, site, subcomponent indices [k], source activities [k])

    @staticmethod
    def concat(ivs: list["Interventions"]) -> "Interventions":
        """One batch holding several batches' interventions in order (rows of the k-th offset by k B):
        several forwards' work in one pass. Weight edits only (scales); swaps and cuts are not combined."""
        B = ivs[0].head.shape[0]
        out = Interventions.__new__(Interventions)
        out.head = torch.cat([iv.head for iv in ivs])
        out.neuron = torch.cat([iv.neuron for iv in ivs])
        out.head_swap = out.neuron_swap = None
        out.cuts, out.tc_swap, out.parts_swap = {}, {}, {}
        out.tc_scale = {r + k * B: e for k, iv in enumerate(ivs) for r, e in iv.tc_scale.items()}
        out.parts_scale = {r + k * B: e for k, iv in enumerate(ivs) for r, e in iv.parts_scale.items()}
        assert not any(iv.cuts or iv.tc_swap or iv.parts_swap or iv.head_swap or iv.neuron_swap for iv in ivs)
        return out


class Qwen3:
    name = MODEL_NAME
    parts = {}  # no VPD view

    def __init__(self, path: str, dev: torch.device, dtype=torch.float32):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.dev, self.dtype = dev, dtype
        self.wide = torch.float64 if dev.type != "mps" else torch.float32
        if dev.type == "cuda":  # straight to the GPU (Qwen3-8B in float32 is 32 GB; no host copy)
            self.model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, device_map="cuda").eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(dev).eval()
        self.tok = AutoTokenizer.from_pretrained(path)
        self.inner = self.model.model
        self.layers = list(self.inner.layers)
        c = self.model.config
        self.L, self.H, self.KV, self.hd, self.Fn, self.d = c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.intermediate_size, c.hidden_size
        self.Wo = [layer.self_attn.o_proj.weight.view(self.d, self.H, self.hd) for layer in self.layers]
        self.name = {1024: "qwen3-0.6b", 2048: "qwen3-1.7b", 2560: "qwen3-4b", 4096: "qwen3-8b"}.get(self.d, f"qwen3-d{self.d}")
        self.tc = {}  # layer -> transcoder tensors (bfloat16 as stored)

    def load_transcoders(self, root: str, layers: list[int]):
        """circuit-tracer's single-layer transcoders (one file per layer: W_enc, W_dec [F, d], b_enc [F],
        b_dec [d]); feature f of layer l reads the MLP's input y (the normed stream) and writes
        relu(W_enc[f].y + b_enc[f]) W_dec[f]."""
        from safetensors.torch import load_file

        for l in layers:
            t = load_file(f"{root}/layer_{l}.safetensors")
            self.tc[l] = {k: t[k].to(self.dev) for k in ("W_enc", "W_dec", "b_enc")}

    def tc_acts(self, l: int, y: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Activations [.., k] of features idx of layer l on MLP inputs y [.., d], in the model's dtype."""
        t = self.tc[l]
        return torch.relu(y @ t["W_enc"][idx].to(self.dtype).T + t["b_enc"][idx].to(self.dtype))

    def new(self, B):
        return Interventions(B, self.L, self.H, self.Fn, self.dev, self.dtype)

    def qkv(self, layer, x, cos, sin):
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

        B, T, _ = x.shape
        a = layer.self_attn
        q = a.q_norm(a.q_proj(x).view(B, T, self.H, self.hd)).transpose(1, 2)
        k = a.k_norm(a.k_proj(x).view(B, T, self.KV, self.hd)).transpose(1, 2)
        v = a.v_proj(x).view(B, T, self.KV, self.hd).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return q, k, v

    def attend(self, q, k, v):
        rep = self.H // self.KV
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        return F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2)  # [B, T, H, hd]

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor, iv: Interventions | None = None, record: dict | None = None, means: dict | None = None):
        """The final normed stream at the last position [B, d]. `record` (optional dict) receives the clean
        statistics: z and activations at the last position, the mean z per head and mean MLP output per
        layer over positions 1.., head write norms and activations of probes ("probes": row -> (l, h or -1, i))."""
        B, T = tokens.shape
        h = self.inner.embed_tokens(tokens)
        pos = torch.arange(T, device=self.dev)[None]
        cos, sin = self.inner.rotary_emb(h, pos)
        cut_delta = {}
        if record is not None:
            record.update(z_last=torch.empty(B, self.L, self.H, self.hd, device=self.dev, dtype=self.dtype),
                          act_last=torch.empty(B, self.L, self.Fn, device=self.dev, dtype=self.dtype),
                          mean_z=torch.empty(self.L, self.H, self.hd, device=self.dev, dtype=self.dtype),
                          mean_mlp=torch.empty(self.L, self.d, device=self.dev, dtype=self.dtype),
                          mean_attn=torch.empty(self.L, self.d, device=self.dev, dtype=self.dtype), probe_values={})
        for l, layer in enumerate(self.layers):
            x = layer.input_layernorm(h)
            q, k, v = self.qkv(layer, x, cos, sin)
            z = self.attend(q, k, v)
            if iv is not None:
                # Cut readers among this layer's heads, all texts at once: the head's query, key or value
                # from the stream with the writer's average write in place of its actual one.
                R = [r for r, c in iv.cuts.items() if c[3] == "head" and c[4] == l and r in cut_delta]
                if R:
                    q2, k2, v2 = self.qkv(layer, layer.input_layernorm(h[R] + torch.stack([cut_delta[r] for r in R])), cos, sin)
                    rt, ar = torch.tensor(R, device=self.dev), torch.arange(len(R), device=self.dev)
                    bh = torch.tensor([iv.cuts[r][5] for r in R], device=self.dev)
                    g = bh // (self.H // self.KV)
                    which = [iv.cuts[r][6] for r in R]
                    sel = lambda name: torch.tensor([w == name for w in which], device=self.dev)[:, None, None]  # noqa: E731
                    qh = torch.where(sel("query"), q2[ar, bh], q[rt, bh])
                    kh = torch.where(sel("key"), k2[ar, g], k[rt, g])
                    vh = torch.where(sel("value"), v2[ar, g], v[rt, g])
                    z = z.clone()
                    z[rt, :, bh] = F.scaled_dot_product_attention(qh[:, None], kh[:, None], vh[:, None], is_causal=True)[:, 0]
                if iv.head_swap is not None:
                    mask, values = iv.head_swap
                    m = mask[:, l]
                    if bool(m.any()):
                        z = z.clone()
                        last = z[:, -1]
                        z[:, -1] = torch.where(m[..., None], values[:, l], last)
                z = z * iv.head[:, l][:, None, :, None]
            if record is not None:
                record["z_last"][:, l] = z[:, -1]
                record["mean_z"][l] = z[:, 1:].mean(dim=(0, 1))
                for r, (pl, ph, pi) in record.get("probes", {}).items():
                    if pl == l and ph >= 0:
                        record["probe_values"][r] = torch.linalg.vector_norm(z[r, :, ph] @ self.Wo[l][:, ph].T, dim=-1)
            attn_out = layer.self_attn.o_proj(z.reshape(B, T, self.H * self.hd))
            if iv is not None:
                Ra = [r for r, c in iv.cuts.items() if c[1] == l and c[0] == "head"]
                if Ra:
                    rt = torch.tensor(Ra, device=self.dev)
                    ah = torch.tensor([iv.cuts[r][2] for r in Ra], device=self.dev)
                    Wsel = self.Wo[l][:, ah]  # [d, n, hd]
                    write = torch.einsum("ntk,dnk->ntd", z[rt, :, ah], Wsel)
                    mean = torch.einsum("nk,dnk->nd", means["mean_z"][l, ah], Wsel)
                    for i, r in enumerate(Ra):
                        cut_delta[r] = mean[i][None] - write[i]
                for r, c in iv.cuts.items():
                    if c[1] == l and c[0] == "attn":
                        cut_delta[r] = means["mean_attn"][l][None] - attn_out[r]
            if record is not None:
                record["mean_attn"][l] = attn_out[:, 1:].mean(dim=(0, 1))
            mid = h + attn_out
            y = layer.post_attention_layernorm(mid)
            if iv is not None:
                Rm = [r for r, c in iv.cuts.items() if c[3] == "mlp" and c[4] == l and r in cut_delta]
                if Rm:
                    y = y.clone()
                    y[Rm] = layer.post_attention_layernorm(mid[Rm] + torch.stack([cut_delta[r] for r in Rm]))
            mlp = layer.mlp
            act = mlp.act_fn(mlp.gate_proj(y)) * mlp.up_proj(y)
            if iv is not None:
                if iv.neuron_swap is not None:
                    mask, values = iv.neuron_swap
                    m = mask[:, l]
                    if bool(m.any()):
                        act = act.clone()
                        act[:, -1] = torch.where(m, values[:, l], act[:, -1])
                act = act * iv.neuron[:, l][:, None, :]
            if record is not None:
                record["act_last"][:, l] = act[:, -1]
                for r, (pl, ph, pi) in record.get("probes", {}).items():
                    if pl == l and ph == -1:
                        record["probe_values"][r] = act[r, :, pi]
                    elif pl == l and ph == -2:
                        record["probe_values"][r] = self.tc_acts(l, y[r], torch.tensor([pi], device=self.dev))[:, 0]
                if l in self.tc:
                    t = self.tc[l]
                    record.setdefault("tc_last", {})[l] = torch.relu(y[:, -1].to(t["W_enc"].dtype) @ t["W_enc"].T + t["b_enc"])  # for aiming only
                    record.setdefault("y_last", {})[l] = y[:, -1]
            mlp_out = mlp.down_proj(act)
            if iv is not None and (iv.tc_scale or iv.tc_swap):
                mlp_out = mlp_out.clone()
                # A feature's edit acts at positions 1.. (position 0, the attention sink, is outside a
                # transcoder's domain; library_transcoder.rs gives it the MLP's own output).
                for r, (tl, idx, alpha) in iv.tc_scale.items():
                    if tl == l:
                        a = self.tc_acts(l, y[r, 1:], idx)
                        mlp_out[r, 1:] += (alpha - 1.0) * (a @ self.tc[l]["W_dec"][idx].to(self.dtype))
                for r, (tl, idx, values) in iv.tc_swap.items():
                    if tl == l:
                        a = self.tc_acts(l, y[r, -1], idx)
                        mlp_out[r, -1] += (values - a) @ self.tc[l]["W_dec"][idx].to(self.dtype)
            if record is not None:
                record["mean_mlp"][l] = mlp_out[:, 1:].mean(dim=(0, 1))
            if iv is not None:
                for r, (ak, al, ah, bk, bl, bh, route) in iv.cuts.items():
                    if al == l and ak == "mlp":
                        cut_delta[r] = means["mean_mlp"][l][None] - mlp_out[r]
            h = mid + mlp_out
        if iv is not None:
            for r, (ak, al, ah, bk, bl, bh, route) in iv.cuts.items():
                if bk == "logits":
                    h = h.clone()
                    h[r] = h[r] + cut_delta[r]
        return self.inner.norm(h[:, -1])

    def log_probs(self, final: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.model.lm_head(final).to(self.wide), dim=-1)

    @torch.no_grad()
    def greedy(self, tokens: torch.Tensor, steps: int, iv: Interventions | None = None) -> torch.Tensor:
        """Greedy continuation [B, steps] under weight edits (head and neuron multipliers; no swaps or cuts)."""
        out = []
        for _ in range(steps):
            nxt = self.log_probs(self.forward(tokens, iv)).argmax(-1)
            out.append(nxt)
            tokens = torch.cat([tokens, nxt[:, None]], dim=1)
        return torch.stack(out, dim=1)


# ---------------------------------------------------------------- text of the questions


def piece_text(p) -> str:
    kind = p[0]
    if kind == "head":
        return f"L[{p[1]}].head[{p[2]}]"
    if kind == "neurons":
        return f"L[{p[1]}].mlp[{', '.join(str(i) for i in p[2])}]"
    if kind == "mlp":
        return f"L[{p[1]}].mlp[:]"
    if kind == "attn":
        return f"L[{p[1]}].head[:]"
    if kind == "tc":
        return f"PD.tc[{p[1]}][{', '.join(str(i) for i in p[2])}]"
    if kind == "vpd":
        return f"PD.vpd[{p[1]}].{p[2]}[{', '.join(str(i) for i in p[3])}]"
    raise ValueError(p)


def apply_scale(iv: Interventions, row: int, p, a: float):
    kind = p[0]
    if kind == "head":
        iv.head[row, p[1], p[2]] = a
    elif kind == "neurons":
        iv.neuron[row, p[1], list(p[2])] = a
    elif kind == "mlp":
        iv.neuron[row, p[1]] = a
    elif kind == "attn":
        iv.head[row, p[1]] = a
    elif kind == "tc":
        iv.tc_scale[row] = (p[1], torch.tensor(p[2], device=iv.head.device), a)
    elif kind == "vpd":
        iv.parts_scale[row] = (p[1], p[2], torch.tensor(p[3], device=iv.head.device), a)


class Writer:
    def __init__(self, m: Qwen3):
        self.m = m
        self.cache = {}

    def token(self, t: int) -> str:
        s = self.cache.get(t)
        if s is None:
            s = self.cache[t] = json.dumps(self.m.tok.decode([int(t)]), ensure_ascii=False)
        return s

    def text(self, ids) -> str:
        return json.dumps(self.m.tok.decode([int(t) for t in ids]), ensure_ascii=False)

    def numbered(self, ids) -> str:
        return " ".join(f"{i}:{self.token(t)}" for i, t in enumerate(ids))

    def dist(self, lp: torch.Tensor):
        """Text of the top TOP tokens with probabilities and the rest; numbers: top 10 ids and probabilities."""
        p = lp.exp()
        top = p.topk(10)
        ids, ps = top.indices.tolist(), top.values.tolist()
        parts = [f"{self.token(t)} {q:.2f}" for t, q in zip(ids[:TOP], ps[:TOP])]
        parts.append(f"other {max(0.0, 1.0 - sum(ps[:TOP])):.2f}")
        return " | ".join(parts), {"ids": ids, "p": [round(q, 6) for q in ps]}


def kl_bits(lp_clean: torch.Tensor, lp_edit: torch.Tensor) -> torch.Tensor:
    return (lp_clean.exp() * (lp_clean - lp_edit)).sum(-1) / math.log(2)




# ---------------------------------------------------------------- drawing pieces


KIND_CODE = {"head": 1, "neuron": 2, "tc": 3, "q_proj": 4, "k_proj": 5, "v_proj": 6, "o_proj": 7, "c_fc": 8, "down_proj": 9}


def held_out_units(kind: str, layer: int, count: int) -> np.ndarray:
    """Which units (heads, neurons, features or subcomponents) of a site are held-out pieces: a fixed
    integer hash of (kind, layer, index), one unit in ten. Training shards ask only about the others;
    held-out-piece shards only about these, so the oracle is scored on pieces it never saw."""
    i = np.arange(count, dtype=np.uint64)
    h = (np.uint64(layer + 1) * np.uint64(0x9E3779B1) + (i + np.uint64(1)) * np.uint64(0x85EBCA77) + np.uint64(KIND_CODE[kind]) * np.uint64(0xC2B2AE3D)) & np.uint64(0xFFFFFFFF)
    h ^= h >> np.uint64(15)
    h = (h * np.uint64(0x2C1B3C6D)) & np.uint64(0xFFFFFFFF)
    h ^= h >> np.uint64(12)
    return (h % np.uint64(10)) == 0


class Draw:
    """Draws pieces for questions, restricted to one piece split: "train" (units not held out; whole blocks
    allowed) or "heldout" (held-out units only; no whole blocks)."""

    def __init__(self, m: Qwen3, rng: np.random.Generator, split: str = "train"):
        self.m, self.rng, self.split = m, rng, split
        want = split == "heldout"
        self.heads = [np.flatnonzero(held_out_units("head", l, m.H) == want) for l in range(m.L)]
        self.neurons = [np.flatnonzero(held_out_units("neuron", l, m.Fn) == want) for l in range(m.L)]
        self._masks = {}

    def mask(self, kind, layer, count):
        """Torch mask of the units of this split (cached), for aimed draws among the most active."""
        key = (kind, layer)
        if key not in self._masks:
            self._masks[key] = torch.from_numpy(held_out_units(kind, layer, count) == (self.split == "heldout")).to(self.m.dev)
        return self._masks[key]

    def layer_with_heads(self):
        while True:
            L = int(self.rng.integers(self.m.L))
            if len(self.heads[L]):
                return L

    def uniform(self):
        r = self.rng.random()
        whole = self.split == "train"
        if r < 0.3 or (not whole and r >= 0.6):
            L = self.layer_with_heads()
            return ("head", L, int(self.rng.choice(self.heads[L])))
        L = int(self.rng.integers(self.m.L))
        if r < 0.6:
            k = int(self.rng.choice([1, 4, 16, 64]))
            return ("neurons", L, tuple(sorted(int(i) for i in self.rng.choice(self.neurons[L], size=k, replace=False))))
        if r < 0.8:
            return ("mlp", L)
        return ("attn", L)

    def aimed_head(self, rec, row, L):
        """A head of this split among the 4 of largest write norm at the row's last position (L must have one)."""
        z = rec["z_last"][row, L]  # [H, hd]
        norms = torch.linalg.vector_norm(torch.einsum("dhk,hk->hd", self.m.Wo[L], z), dim=-1)
        norms = torch.where(self.mask("head", L, self.m.H), norms, torch.full_like(norms, -1.0))
        top = norms.topk(min(4, len(self.heads[L]))).indices.tolist()
        return int(self.rng.choice(top))

    def aimed(self, rec, row):
        """A piece among the most active at the row's last position: a head among the 4 of largest write
        norm in its layer, or neurons among the 256 of largest |activation|."""
        if self.rng.random() < 0.5:
            L = self.layer_with_heads()
            return ("head", L, self.aimed_head(rec, row, L))
        L = int(self.rng.integers(self.m.L))
        a = rec["act_last"][row, L].abs()
        a = torch.where(self.mask("neuron", L, self.m.Fn), a, torch.full_like(a, -1.0))
        top = a.topk(min(256, len(self.neurons[L]))).indices.cpu().numpy()
        k = int(self.rng.choice([1, 4, 16, 64]))
        return ("neurons", L, tuple(sorted(int(i) for i in self.rng.choice(top[: max(k, 16 * k if k < 16 else 256)], size=k, replace=False))))

    def feature(self, rec, row, small=False):
        """Transcoder features (Qwen3) or VPD subcomponents (vpd4l) among the 64 most active at the row's
        last position of a random layer (and site)."""
        if self.m.parts:
            l, site = list(self.m.parts)[int(self.rng.integers(len(self.m.parts)))]
            act = (rec["site_in_last"][(l, site)][row] @ self.m.parts[(l, site)][1]).abs()
            act = torch.where(self.mask(site, l, act.shape[0]), act, torch.full_like(act, -1.0))
            top = act.topk(64).indices.cpu().numpy()
            k = int(self.rng.choice([1, 1, 4] if small else [1, 4, 16]))
            return ("vpd", l, site, tuple(sorted(int(i) for i in self.rng.choice(top, size=k, replace=False))))
        L = int(self.rng.choice(sorted(self.m.tc)))
        a = rec["tc_last"][L][row].float()
        a = torch.where(self.mask("tc", L, a.shape[0]), a, torch.zeros_like(a))
        top = a.topk(64)
        live = top.indices[top.values > 0].cpu().numpy()
        if len(live) == 0:
            return None
        k = min(len(live), int(self.rng.choice([1, 1, 4] if small else [1, 4, 16])))
        return ("tc", L, tuple(sorted(int(i) for i in self.rng.choice(live, size=k, replace=False))))

    def piece(self, rec, row, small=False):
        """Half aimed, half uniform (a quarter transcoder features when transcoders are loaded); `small`
        keeps neuron groups at 4 or fewer (questions listing several pieces)."""
        while True:
            if (self.m.tc and self.rng.random() < 0.25) or (self.m.parts and self.rng.random() < 0.4):
                p = self.feature(rec, row, small)
                if p is not None:
                    return p
                continue
            p = self.aimed(rec, row) if self.rng.random() < 0.5 else self.uniform()
            if not small or p[0] != "neurons" or len(p[2]) <= 4:
                return p


# ---------------------------------------------------------------- one batch of texts


def batch_questions(m: Qwen3, w: Writer, draw: Draw, tokens: torch.Tensor, source: str, split: str, steps: int, ids: list, counterfactuals=None):
    """Every question type on a batch of texts of one length; `counterfactuals` (optional, per text a
    token list of the same length or None) are the behaviors' own prompt edits, used by the prompt type."""
    rng = draw.rng
    B, T = tokens.shape
    out = []
    rows = list(range(B))

    def emit(row, kind, inp, ans, numbers):
        out.append({"model": m.name, "type": kind, "source": source, "split": split, "piece_split": draw.split, "text_id": ids[row],
                    "input": f"<model> {m.name}\n" + inp, "answer": ans, "numbers": numbers})

    # Probes for the where questions: a neuron or a head per text.
    probes = {}
    for r in rows:
        if rng.random() < 0.4:
            l = draw.layer_with_heads()
            probes[r] = (l, int(rng.choice(draw.heads[l])), -1)
        else:
            l = int(rng.integers(m.L))
            probes[r] = (l, -1, int(rng.choice(draw.neurons[l])))
    rec = {"probes": probes}
    if m.tc or m.parts:
        # Feature probes need the clean pass's activations to pick live features: a first pass picks them.
        first = {}
        m.forward(tokens, None, first)
        for r in rows:
            if rng.random() < 0.3:
                p = draw.feature(first, r)
                if p is not None:
                    probes[r] = (p[1], -2, p[2][0]) if p[0] == "tc" else (p[1], -3, (p[2], p[3][0]))
    lp_clean = m.log_probs(m.forward(tokens, None, rec))
    clean_txt = [w.dist(lp_clean[r]) for r in rows]
    texts = [w.text(tokens[r].tolist()) for r in rows]

    # plain (no piece: not asked in held-out-piece shards, nor prompt edits)
    for r in rows if draw.split == "train" else []:
        emit(r, "plain", f"<text> {texts[r]}\n<question> next-token distribution\n", clean_txt[r][0], {"edited": clean_txt[r][1]})

    def edited_questions(kind, iv, describe, extra=None, toks=None):
        lp = m.log_probs(m.forward(tokens if toks is None else toks, iv, None, rec))
        kl = kl_bits(lp_clean, lp)
        for r in rows:
            d, nums = w.dist(lp[r])
            pre = f"<text> {texts[r]}\n" if toks is None else ""
            emit(r, kind, pre + describe[r] + f"<clean> {clean_txt[r][0]}\n<question> next-token distribution after the intervention, and its KL from clean in bits\n",
                 f"{d}\nKL {max(kl[r].item(), 0.0):.3f} bits", {"edited": nums, "clean": clean_txt[r][1], "kl_bits": kl[r].item(), **(extra[r] if extra else {})})

    # edit
    iv, desc, extra = m.new(B), [], []
    for r in rows:
        p, a = draw.piece(rec, r), float(rng.choice([0.0, 0.0, 0.5, 2.0]))
        apply_scale(iv, r, p, a)
        desc.append(f"<intervention> scale({piece_text(p)}, {a:g})\n")
        extra.append({"piece": piece_text(p), "alpha": a})
    edited_questions("edit", iv, desc, extra)

    # rank: 4 candidates, one removal per forward
    cands = [[draw.piece(rec, r, small=True) for _ in range(4)] for r in rows]
    ivs = []
    for c in range(4):
        iv = m.new(B)
        for r in rows:
            apply_scale(iv, r, cands[r][c], 0.0)
        ivs.append(iv)
    # The 4 removals of every text in one forward of 4 B rows.
    lp4 = m.log_probs(m.forward(tokens.repeat(4, 1), Interventions.concat(ivs))).view(4, B, -1)
    kls = torch.stack([kl_bits(lp_clean, lp4[c]) for c in range(4)], dim=1).cpu()
    for r in rows:
        names = "abcd"
        listing = " ".join(f"({names[c]}) {piece_text(cands[r][c])}" for c in range(4))
        order = sorted(range(4), key=lambda c: -kls[r, c].item())
        ans = " > ".join(names[c] for c in order) + "\nKL bits: " + ", ".join(f"{names[c]} {max(kls[r, c].item(), 0.0):.3f}" for c in order)
        emit(r, "rank", f"<text> {texts[r]}\n<clean> {clean_txt[r][0]}\n<question> which removal, scale(PIECE, 0), changes the next-token distribution most: {listing}\n",
             ans, {"pieces": [piece_text(p) for p in cands[r]], "kl_bits": kls[r].tolist()})

    # where
    for r in rows:
        l, hh, i = probes[r]
        v = rec["probe_values"][r].to(torch.float32).cpu()
        order = v[1:].abs().topk(min(3, T - 1)).indices + 1
        piece = {-1: lambda: f"L[{l}].mlp[{i}]", -2: lambda: f"PD.tc[{l}][{i}]", -3: lambda: f"PD.vpd[{l}].{i[0]}[{i[1]}]"}[hh]() if hh < 0 else f"L[{l}].head[{hh}]"
        level = "write norm" if hh >= 0 else ("activity v.x" if hh == -3 else "activation")
        ans = ", ".join(f"{int(p)}:{w.token(tokens[r, int(p)].item())} {v[int(p)].item():.2f}" for p in order)
        emit(r, "where", f"<text_tokens> {w.numbered(tokens[r].tolist())}\n<question> where is {piece} most active ({level}; position 0 excluded): three positions and levels\n",
             ans, {"piece": piece, "positions": [int(p) for p in order], "levels": [v[int(p)].item() for p in order], "mean_level": v[1:].abs().mean().item()})

    # continue
    iv, desc = m.new(B), []
    for r in rows:
        p, a = draw.piece(rec, r), float(rng.choice([2.0, 4.0]))
        apply_scale(iv, r, p, a)
        desc.append((piece_text(p), a))
    both = m.greedy(tokens.repeat(2, 1), steps, Interventions.concat([m.new(B), iv]))  # clean and edited rows in one pass
    clean_cont, edit_cont = both[:B], both[B:]
    for r in rows:
        cc, ec = clean_cont[r].tolist(), edit_cont[r].tolist()
        same = sum(1 for x, y in zip(cc, ec) if x == y)
        emit(r, "continue", f"<text> {texts[r]}\n<clean_continuation> {w.text(cc)}\n<intervention> scale({desc[r][0]}, {desc[r][1]:g})\n<question> greedy continuation of {steps} tokens after the intervention\n",
             w.text(ec), {"piece": desc[r][0], "alpha": desc[r][1], "clean_ids": cc, "edited_ids": ec, "tokens_unchanged": same})

    # cut
    iv, desc, extra = m.new(B), [], []
    for r in rows:
        u = rng.random()
        if u < 0.5 or draw.split == "heldout":
            al = draw.layer_with_heads()
            ak, ah = "head", (draw.aimed_head(rec, r, al) if rng.random() < 0.5 else int(rng.choice(draw.heads[al])))
        else:
            al = int(rng.integers(m.L))
            ak, ah = ("mlp" if u < 0.75 else "attn"), -1
        first = al + (1 if ak == "mlp" else 0)  # the earliest layer whose readers see A's write
        v = rng.random()
        if v < 0.3 or first >= m.L:
            bk, bl, bh, route = "logits", m.L, -1, ""
        elif v < 0.65 or first + (0 if ak == "mlp" else 1) >= m.L:
            bk, bl, bh, route = "mlp", int(rng.integers(first, min(m.L, first + 4))), -1, "input"
        else:
            lo = first if ak == "mlp" else first + 1
            bl = int(rng.integers(lo, min(m.L, lo + 4)))
            bh = int(rng.choice(draw.heads[bl])) if len(draw.heads[bl]) else -1
            bk, route = ("head", ROUTES[int(rng.integers(3))]) if bh >= 0 else ("mlp", "input")
        iv.cuts[r] = (ak, al, ah, bk, bl, bh, route)
        a_txt = "node(" + piece_text(("head", al, ah) if ak == "head" else (ak, al)) + ")"
        b_txt = {"logits": "logits", "mlp": f"node(L[{bl}].mlp[:]).input", "head": f"node(L[{bl}].head[{bh}]).{route}"}[bk]
        desc.append(f"<intervention> cut({a_txt} >> {b_txt})\n")
        extra.append({"edge": f"{a_txt} >> {b_txt}"})
    edited_questions("cut", iv, desc, extra)

    # prompt edit: one token replaced by a token of another text in the batch
    if draw.split == "train":
        toks2 = tokens.clone()
        desc, extra = [], []
        for r in rows:
            cf = counterfactuals[r] if counterfactuals else None
            if cf is not None:
                toks2[r] = torch.tensor(cf, device=tokens.device)
                changed = [i for i in range(T) if int(tokens[r, i]) != cf[i]]
                edits = ", ".join(f"position {i}: {w.token(int(tokens[r, i]))} -> {w.token(cf[i])}" for i in changed)
                desc.append(f"<text> {texts[r]}\n<edit> {edits}\n<edited_text> {w.text(cf)}\n")
                extra.append({"positions": changed, "counterfactual": cf})
                continue
            p = int(rng.integers(1, T))
            new = int(tokens[(r + 1 + int(rng.integers(B - 1))) % B, int(rng.integers(T))].item()) if B > 1 else int(rng.integers(1000))
            old = int(tokens[r, p].item())
            toks2[r, p] = new
            desc.append(f"<text> {texts[r]}\n<edit> position {p}: {w.token(old)} -> {w.token(new)}\n<edited_text> {w.text(toks2[r].tolist())}\n")
            extra.append({"position": p, "old": old, "new": new})
        edited_questions("prompt", None, desc, extra, toks=toks2)

    # swap: the piece's last-position value from the next text of the batch
    if B > 1:
        iv, desc, extra = m.new(B), [], []
        hm = torch.zeros(B, m.L, m.H, dtype=torch.bool, device=m.dev)
        nm = torch.zeros(B, m.L, m.Fn, dtype=torch.bool, device=m.dev)
        src = [(r + 1) % B for r in rows]
        for r in rows:
            p = draw.piece(rec, r)
            if p[0] == "head":
                hm[r, p[1], p[2]] = True
            elif p[0] == "neurons":
                nm[r, p[1], list(p[2])] = True
            elif p[0] == "mlp":
                nm[r, p[1]] = True
            elif p[0] == "tc":
                idx = torch.tensor(p[2], device=m.dev)
                iv.tc_swap[r] = (p[1], idx, m.tc_acts(p[1], rec["y_last"][p[1]][src[r]], idx))
            elif p[0] == "vpd":
                idx = torch.tensor(p[3], device=m.dev)
                iv.parts_swap[r] = (p[1], p[2], idx, rec["site_in_last"][(p[1], p[2])][src[r]] @ m.parts[(p[1], p[2])][1][:, idx])
            else:
                hm[r, p[1]] = True
            desc.append(f"<source> {texts[src[r]]}\n<intervention> swap({piece_text(p)}, source)\n")
            extra.append({"piece": piece_text(p), "source_text_id": ids[src[r]]})
        iv.head_swap = (hm, rec["z_last"][src])
        iv.neuron_swap = (nm, rec["act_last"][src])
        edited_questions("swap", iv, desc, extra)
    return out


def load_behaviors(root: Path, tok, per_behavior: int, rng):
    """Behavior prompts (g-behaviors' files for this model): per prompt and target position t, the prefix
    ending at t (the question is about the token after it) and the counterfactual's prefix when it has the
    same length; at most `per_behavior` prefixes per behavior (drawn uniformly)."""
    seqs = []
    for f in sorted(root.glob("*.json")):
        b = json.loads(f.read_text())
        mine = []
        for p in b.get("prompts", []):
            ids = [int(t) for t in (p.get("token_ids") or tok(p["text"])["input_ids"])]
            cf = p.get("counterfactual") or {}
            cf_ids = [int(t) for t in cf.get("token_ids", [])]
            for t in p.get("target_positions") or [len(ids) - 1]:
                if t >= 3:
                    prefix = ids[: t + 1]
                    cfp = cf_ids[: t + 1] if len(cf_ids) > t and cf_ids[: t + 1] != prefix else None
                    mine.append((b["id"], prefix, b.get("split", "train"), cfp))
        for i in rng.permutation(len(mine))[:per_behavior]:
            seqs.append(mine[i])
    return seqs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="qwen3-0.6b", choices=("qwen3-0.6b", "vpd4l"))
    ap.add_argument("--model", default="", help="Qwen3-0.6B snapshot directory (vpd4l loads its own files)")
    ap.add_argument("--windows", default="")
    ap.add_argument("--behaviors", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--texts", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lengths", default="24,32,48,64")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--piece-split", default="train", choices=("train", "heldout"),
                    help="ask about training pieces (and whole blocks) or only held-out pieces (held_out_units)")
    ap.add_argument("--row-range", default="", help="A:B, draw texts from rows A..B-1 only (e.g. vpd4l Pile rows 0:3584 train, 3584:4096 held out)")
    ap.add_argument("--uv", default="", help="vpd4l: VPD's subcomponents (vpd_labels.py export_uv; default ~/mpd-data/oracle/vpd/uv.safetensors)")
    ap.add_argument("--vpd-target", default="", help="vpd4l: the target run directory t-9d2b8f02 (default ~/mpd-data/vpd/t-9d2b8f02)")
    ap.add_argument("--behavior-split", default="", help="only the behaviors of this split (train or heldout)")
    ap.add_argument("--tf32", action="store_true", help="TF32 matmuls (Qwen3-8B as target: about 3x faster, ~1e-3 relative rounding)")
    ap.add_argument("--per-behavior", type=int, default=64, help="prefixes per behavior (--behaviors)")
    ap.add_argument("--transcoders", default="", help="circuit-tracer transcoder directory (layer_{l}.safetensors)")
    ap.add_argument("--tc-layers", default="", help="layers whose transcoder features are asked about, e.g. 3,9,14,20,25")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    dev = device()
    if args.target == "vpd4l":
        import vpd4l

        if args.vpd_target:
            vpd4l.VM.TARGET_DIR = Path(args.vpd_target)
        m = vpd4l.Vpd4l(dev, **({"uv": Path(args.uv)} if args.uv else {}), **({"tokenizer": Path(args.vpd_target) / "tokenizer.json"} if args.vpd_target else {}))
    else:
        m = Qwen3(args.model, dev)
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if args.transcoders and args.tc_layers:
        m.load_transcoders(args.transcoders, [int(x) for x in args.tc_layers.split(",")])
    w = Writer(m)
    rng = np.random.default_rng(args.seed)
    draw = Draw(m, rng, args.piece_split)
    lengths = [int(x) for x in args.lengths.split(",")]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    started, written, texts = time.time(), 0, 0
    with open(out, "w") as f:
        if args.windows:
            if args.windows.endswith(".npy"):  # token rows, e.g. vpd4l's Pile validation rows [N, 513]
                windows = np.load(args.windows, mmap_mode="r")
            else:
                windows = np.memmap(args.windows, dtype="<u4", mode="r").reshape(-1, 128)
            lo, hi = (int(x) for x in args.row_range.split(":")) if args.row_range else (0, len(windows))
            # One fixed text order for every task, so tasks at offsets k * texts read disjoint texts.
            order = lo + np.random.default_rng(1234).permutation(hi - lo)  # disjoint ranges keep splits apart
            for s in range(args.offset, args.offset + args.texts, args.batch):
                Tn = lengths[(s // args.batch) % len(lengths)]
                rows = order[s : s + args.batch]
                toks = torch.from_numpy(np.asarray(windows[rows, :Tn]).astype(np.int64)).to(dev)
                corpus = "fineweb" if args.target == "qwen3-0.6b" else "pile"
                qs = batch_questions(m, w, draw, toks, corpus, args.split, args.steps, [f"{corpus}:{int(i)}:{Tn}" for i in rows])
                for q in qs:
                    f.write(json.dumps(q, ensure_ascii=False) + "\n")
                f.flush()
                if dev.type == "mps":
                    torch.mps.empty_cache()
                written += len(qs)
                texts += len(rows)
                rate = texts / (time.time() - started)
                print(json.dumps({"texts": texts, "questions": written, "T": Tn, "texts_per_s": round(rate, 2), "questions_per_s": round(rate * 8, 1)}), flush=True)
        if args.behaviors:
            seqs = load_behaviors(Path(args.behaviors), m.tok, args.per_behavior, np.random.default_rng(args.seed + 2))
            if args.behavior_split:
                seqs = [item for item in seqs if item[2] == args.behavior_split]
            by_len = {}
            for item in seqs:
                by_len.setdefault((len(item[1]), item[2]), []).append(item)
            for n, group in sorted(by_len.items()):
                for s in range(0, len(group), args.batch):
                    chunk = group[s : s + args.batch]
                    if len(chunk) < 2:
                        continue
                    toks = torch.tensor([ids for _, ids, _, _ in chunk], device=dev)
                    qs = batch_questions(m, w, draw, toks, "behavior", chunk[0][2], args.steps, [bid for bid, _, _, _ in chunk],
                                         [cf for _, _, _, cf in chunk])
                    for q in qs:
                        f.write(json.dumps(q, ensure_ascii=False) + "\n")
                    f.flush()
                    if dev.type == "mps":
                        torch.mps.empty_cache()
                    written += len(qs)
                    texts += len(chunk)
    print(json.dumps({"done": True, "texts": texts, "questions": written, "seconds": round(time.time() - started, 1)}), flush=True)


if __name__ == "__main__":
    main()
