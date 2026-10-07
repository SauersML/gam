"""Prediction training data for the graph oracle (#2951): causal questions about the target model M, each
with an answer measured by an exact intervention on M, written as text for the oracle (its input and the
answer it is trained to write) plus the measured numbers.

Interventions are written in mech syntax (bench/oracle/graph/mech.py addresses). Pieces of Qwen3-0.6B
(native view):
  L[l].head[h]          head h of layer l. Edits act on its write: scale(L[l].head[h], a) multiplies the
                        head's attention read z_h (the input of its o_proj columns) by a, which is
                        oracle.rs's Component::Head, W_o[:, h] <- a W_o[:, h].
  L[l].mlp[i, j, ...]   MLP neurons i, j, ...: scale multiplies their activations SiLU(g_i.x)(u_i.x) by a,
                        i.e. their down columns (oracle.rs's Component::Neuron, summed).
  L[l].mlp, L[l].attn   the whole MLP (every neuron) or the whole attention (every head) of layer l.
  a = 0 removes the piece. A weight edit acts at every position (and every generated step).
  cut(A >> B.route)     path patching: B's input (route query, key or value of a head, input of an MLP)
                        receives A's AVERAGE write (over this shard's texts, every position after the first)
                        in place of A's actual write; every other reader keeps A's actual write.
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
  cut       cut an edge A >> B.route -> the new distribution and its KL.
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
        self.cuts = {}  # row -> (a_layer, a_head or -1 for the MLP, b_layer, b_head or -1, route)


class Qwen3:
    def __init__(self, path: str, dev: torch.device, dtype=torch.float32):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.dev, self.dtype = dev, dtype
        self.wide = torch.float64 if dev.type != "mps" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(dev).eval()
        self.tok = AutoTokenizer.from_pretrained(path)
        self.inner = self.model.model
        self.layers = list(self.inner.layers)
        c = self.model.config
        self.L, self.H, self.KV, self.hd, self.Fn, self.d = c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.intermediate_size, c.hidden_size
        self.Wo = [layer.self_attn.o_proj.weight.view(self.d, self.H, self.hd) for layer in self.layers]

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
                          mean_mlp=torch.empty(self.L, self.d, device=self.dev, dtype=self.dtype), probe_values={})
        for l, layer in enumerate(self.layers):
            x = layer.input_layernorm(h)
            q, k, v = self.qkv(layer, x, cos, sin)
            z = self.attend(q, k, v)
            if iv is not None:
                for r, (al, ah, bl, bh, route) in iv.cuts.items():
                    if bl == l and bh >= 0 and r in cut_delta:
                        xr = layer.input_layernorm(h[r : r + 1] + cut_delta[r])
                        q2, k2, v2 = self.qkv(layer, xr, cos, sin)
                        g = bh // (self.H // self.KV)
                        qh = (q2 if route == "query" else q[r : r + 1])[:, bh : bh + 1]
                        kh = (k2 if route == "key" else k[r : r + 1])[:, g : g + 1]
                        vh = (v2 if route == "value" else v[r : r + 1])[:, g : g + 1]
                        zr = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True).transpose(1, 2)
                        z = z.clone()
                        z[r, :, bh] = zr[0, :, 0]
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
                for r, (al, ah, bl, bh, route) in iv.cuts.items():
                    if al == l and ah >= 0:
                        write = z[r, :, ah] @ self.Wo[l][:, ah].T
                        cut_delta[r] = (means["mean_z"][l, ah] @ self.Wo[l][:, ah].T)[None] - write
            mid = h + attn_out
            y = layer.post_attention_layernorm(mid)
            if iv is not None:
                for r, (al, ah, bl, bh, route) in iv.cuts.items():
                    if bl == l and bh < 0 and r in cut_delta:
                        y = y.clone()
                        y[r] = layer.post_attention_layernorm(mid[r] + cut_delta[r])
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
                    if pl == l and ph < 0:
                        record["probe_values"][r] = act[r, :, pi]
            mlp_out = mlp.down_proj(act)
            if record is not None:
                record["mean_mlp"][l] = mlp_out[:, 1:].mean(dim=(0, 1))
            if iv is not None:
                for r, (al, ah, bl, bh, route) in iv.cuts.items():
                    if al == l and ah < 0:
                        cut_delta[r] = means["mean_mlp"][l][None] - mlp_out[r]
            h = mid + mlp_out
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
        return f"L[{p[1]}].mlp"
    if kind == "attn":
        return f"L[{p[1]}].attn"
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


HEADER = "<model> qwen3-0.6b\n"


# ---------------------------------------------------------------- drawing pieces


class Draw:
    def __init__(self, m: Qwen3, rng: np.random.Generator):
        self.m, self.rng = m, rng

    def uniform(self):
        r = self.rng.random()
        L = int(self.rng.integers(self.m.L))
        if r < 0.45:
            return ("head", L, int(self.rng.integers(self.m.H)))
        if r < 0.85:
            k = int(self.rng.choice([1, 1, 2, 4, 8]))
            return ("neurons", L, tuple(sorted(int(i) for i in self.rng.choice(self.m.Fn, size=k, replace=False))))
        if r < 0.95:
            return ("mlp", L)
        return ("attn", L)

    def aimed(self, rec, row):
        """A piece among the most active at the row's last position: a head by write norm, or neurons by
        |activation| (top 64 of the layer)."""
        L = int(self.rng.integers(self.m.L))
        if self.rng.random() < 0.5:
            z = rec["z_last"][row, L]  # [H, hd]
            norms = torch.linalg.vector_norm(torch.einsum("dhk,hk->hd", self.m.Wo[L], z), dim=-1)
            top = norms.topk(4).indices.tolist()
            return ("head", L, int(self.rng.choice(top)))
        a = rec["act_last"][row, L].abs()
        top = a.topk(64).indices.cpu().numpy()
        k = int(self.rng.choice([1, 1, 2, 4, 8]))
        return ("neurons", L, tuple(sorted(int(i) for i in self.rng.choice(top, size=k, replace=False))))

    def piece(self, rec, row):
        return self.aimed(rec, row) if self.rng.random() < 0.5 else self.uniform()


# ---------------------------------------------------------------- one batch of texts


def batch_questions(m: Qwen3, w: Writer, draw: Draw, tokens: torch.Tensor, source: str, split: str, steps: int, ids: list):
    rng = draw.rng
    B, T = tokens.shape
    out = []
    rows = list(range(B))

    def emit(row, kind, inp, ans, numbers):
        out.append({"model": MODEL_NAME, "type": kind, "source": source, "split": split, "text_id": ids[row],
                    "input": HEADER + inp, "answer": ans, "numbers": numbers})

    # Probes for the where questions: a neuron or a head per text.
    probes = {}
    for r in rows:
        l = int(rng.integers(m.L))
        probes[r] = (l, int(rng.integers(m.H)), -1) if rng.random() < 0.4 else (l, -1, int(rng.integers(m.Fn)))
    rec = {"probes": probes}
    lp_clean = m.log_probs(m.forward(tokens, None, rec))
    clean_txt = [w.dist(lp_clean[r]) for r in rows]
    texts = [w.text(tokens[r].tolist()) for r in rows]

    # plain
    for r in rows:
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
    cands = [[draw.piece(rec, r) for _ in range(4)] for r in rows]
    kls = torch.empty(B, 4, dtype=m.wide)
    for c in range(4):
        iv = m.new(B)
        for r in rows:
            apply_scale(iv, r, cands[r][c], 0.0)
        kls[:, c] = kl_bits(lp_clean, m.log_probs(m.forward(tokens, iv))).cpu()
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
        piece = f"L[{l}].head[{hh}]" if hh >= 0 else f"L[{l}].mlp[{i}]"
        level = "write norm" if hh >= 0 else "activation"
        ans = ", ".join(f"{int(p)}:{w.token(tokens[r, int(p)].item())} {v[int(p)].item():.2f}" for p in order)
        emit(r, "where", f"<text_tokens> {w.numbered(tokens[r].tolist())}\n<question> where is {piece} most active ({level}; position 0 excluded): three positions and levels\n",
             ans, {"piece": piece, "positions": [int(p) for p in order], "levels": [v[int(p)].item() for p in order], "mean_level": v[1:].abs().mean().item()})

    # continue
    iv, desc = m.new(B), []
    for r in rows:
        p, a = draw.piece(rec, r), float(rng.choice([2.0, 4.0]))
        apply_scale(iv, r, p, a)
        desc.append((piece_text(p), a))
    clean_cont = m.greedy(tokens, steps)
    edit_cont = m.greedy(tokens, steps, iv)
    for r in rows:
        cc, ec = clean_cont[r].tolist(), edit_cont[r].tolist()
        same = sum(1 for x, y in zip(cc, ec) if x == y)
        emit(r, "continue", f"<text> {texts[r]}\n<clean_continuation> {w.text(cc)}\n<intervention> scale({desc[r][0]}, {desc[r][1]:g})\n<question> greedy continuation of {steps} tokens after the intervention\n",
             w.text(ec), {"piece": desc[r][0], "alpha": desc[r][1], "clean_ids": cc, "edited_ids": ec, "tokens_unchanged": same})

    # cut
    iv, desc, extra = m.new(B), [], []
    for r in rows:
        al = int(rng.integers(m.L - 1))
        ah = int(rng.integers(m.H)) if rng.random() < 0.6 else -1
        bl = int(rng.integers(al + (1 if ah < 0 else 0), m.L))
        bh = int(rng.integers(m.H)) if (rng.random() < 0.6 or (ah >= 0 and bl == al)) else -1
        if ah >= 0 and bl == al:
            bh = -1  # a head's write reaches its own layer's MLP only
        route = ROUTES[int(rng.integers(3))] if bh >= 0 else "input"
        iv.cuts[r] = (al, ah, bl, bh, route)
        a_txt = f"L[{al}].head[{ah}]" if ah >= 0 else f"L[{al}].mlp"
        b_txt = (f"L[{bl}].head[{bh}].{route}" if bh >= 0 else f"L[{bl}].mlp.input")
        desc.append(f"<intervention> cut({a_txt} >> {b_txt})\n")
        extra.append({"edge": f"{a_txt} >> {b_txt}"})
    edited_questions("cut", iv, desc, extra)

    # prompt edit: one token replaced by a token of another text in the batch
    toks2 = tokens.clone()
    desc, extra = [], []
    for r in rows:
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
            else:
                hm[r, p[1]] = True
            desc.append(f"<source> {texts[src[r]]}\n<intervention> swap({piece_text(p)}, source)\n")
            extra.append({"piece": piece_text(p), "source_text_id": ids[src[r]]})
        iv.head_swap = (hm, rec["z_last"][src])
        iv.neuron_swap = (nm, rec["act_last"][src])
        edited_questions("swap", iv, desc, extra)
    return out


def load_behaviors(root: Path, tok):
    """Behavior prompts (g-behaviors' files for this model), as token id lists."""
    seqs = []
    for f in sorted(root.glob("*.json")):
        b = json.loads(f.read_text())
        for p in b.get("prompts", []):
            ids = p.get("token_ids") or tok(p["text"])["input_ids"]
            if len(ids) >= 4:
                seqs.append((f"{b['id']}", [int(t) for t in ids], b.get("split", "train")))
    return seqs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
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
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    dev = device()
    m = Qwen3(args.model, dev)
    w = Writer(m)
    rng = np.random.default_rng(args.seed)
    draw = Draw(m, rng)
    lengths = [int(x) for x in args.lengths.split(",")]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    started, written, texts = time.time(), 0, 0
    with open(out, "w") as f:
        if args.windows:
            windows = np.memmap(args.windows, dtype="<u4", mode="r").reshape(-1, 128)
            order = np.random.default_rng(args.seed + 1).permutation(len(windows))
            for s in range(args.offset, args.offset + args.texts, args.batch):
                Tn = lengths[(s // args.batch) % len(lengths)]
                rows = order[s : s + args.batch]
                toks = torch.from_numpy(np.asarray(windows[rows, :Tn]).astype(np.int64)).to(dev)
                qs = batch_questions(m, w, draw, toks, "fineweb", args.split, args.steps, [f"fineweb:{int(i)}:{Tn}" for i in rows])
                for q in qs:
                    f.write(json.dumps(q, ensure_ascii=False) + "\n")
                f.flush()
                written += len(qs)
                texts += len(rows)
                rate = texts / (time.time() - started)
                print(json.dumps({"texts": texts, "questions": written, "T": Tn, "texts_per_s": round(rate, 2), "questions_per_s": round(rate * 8, 1)}), flush=True)
        if args.behaviors:
            seqs = load_behaviors(Path(args.behaviors), m.tok)
            by_len = {}
            for bid, ids, split in seqs:
                by_len.setdefault((len(ids), split), []).append((bid, ids, split))
            for n, group in sorted(by_len.items()):
                for s in range(0, len(group), args.batch):
                    chunk = group[s : s + args.batch]
                    if len(chunk) < 2:
                        continue
                    toks = torch.tensor([ids for _, ids, _ in chunk], device=dev)
                    qs = batch_questions(m, w, draw, toks, "behavior", chunk[0][2], args.steps, [bid for bid, _, _ in chunk])
                    for q in qs:
                        f.write(json.dumps(q, ensure_ascii=False) + "\n")
                    written += len(qs)
                    texts += len(chunk)
    print(json.dumps({"done": True, "texts": texts, "questions": written, "seconds": round(time.time() - started, 1)}), flush=True)


if __name__ == "__main__":
    main()
