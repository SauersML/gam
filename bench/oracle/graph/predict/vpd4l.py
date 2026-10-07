"""vpd4l (Goodfire's 4-layer Pile model, t-9d2b8f02) for generate.py (#2951): the same executor interface
as generate.Qwen3, on the paper code's weights (bench/vpd_2951/vpd_model.py), with VPD's rank-one
subcomponents (s-55ea3f9b) as a second view.

Pieces: L[l].head[h] (6 heads, edits on the write: the head's o_proj columns), L[l].mlp[i, ...] (GELU
neurons: c_fc rows and down_proj columns), L[l].mlp[:], L[l].head[:], and PD.vpd[l].<site>[i, ...], VPD
subcomponents u_i v_i^T of a site (q_proj k_proj v_proj o_proj c_fc down_proj). A subcomponent's scale by
a is the native weight edit W + (a - 1) u v^T: the site's output gains (a - 1)(v . x) u at every position;
its swap replaces its activity v . x at the last position by the source text's. Where-probes on
subcomponents report the activity v . x.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "vpd_2951"))
import vpd_model as VM  # noqa: E402

UV = Path.home() / "mpd-data/oracle/vpd/uv.safetensors"
TOKENIZER = Path.home() / "mpd-data/vpd/t-9d2b8f02/tokenizer.json"
SITES = ("q_proj", "k_proj", "v_proj", "o_proj", "c_fc", "down_proj")


class Tok:
    """The tokenizer behind the two calls generate.py makes (decode, and __call__ for behavior texts)."""

    def __init__(self, path: Path):
        import tokenizers

        self.t = tokenizers.Tokenizer.from_file(str(path))

    def decode(self, ids):
        return self.t.decode([int(i) for i in ids])

    def __call__(self, text):
        return {"input_ids": self.t.encode(text).ids}


class Vpd4l:
    name = "vpd4l"

    def __init__(self, dev: torch.device, uv: Path = UV, tokenizer: Path = TOKENIZER):
        from safetensors.torch import load_file

        from generate import Interventions

        self._iv = Interventions
        torch.backends.cuda.matmul.allow_tf32 = False
        self.dev, self.dtype = dev, torch.float32
        self.wide = torch.float64 if dev.type != "mps" else torch.float32
        self.t = VM.load_target(str(dev))
        self.tok = Tok(tokenizer)
        t = self.t
        self.L, self.H, self.hd, self.d = t.n_layer, t.n_head, t.hd, t.wte.shape[1]
        self.KV, self.Fn = self.H, t.site("h.0.mlp.c_fc").W.shape[0]
        self.Wo = [t.site(f"h.{l}.attn.o_proj").W.view(self.d, self.H, self.hd) for l in range(self.L)]
        d = load_file(str(uv))
        self.parts = {(l, k): (d[f"h.{l}.{'mlp' if k in ('c_fc', 'down_proj') else 'attn'}.{k}.U"].to(dev).float(),
                               d[f"h.{l}.{'mlp' if k in ('c_fc', 'down_proj') else 'attn'}.{k}.V"].to(dev).float())
                      for l in range(self.L) for k in SITES}
        self.tc = {}

    def new(self, B):
        return self._iv(B, self.L, self.H, self.Fn, self.dev, self.dtype)

    def site(self, l, kind, x, iv, rows=None):
        """Site output x @ W^T plus the subcomponent edits of the rows (`rows`: x holds those rows only)."""
        out = self.t.site(f"h.{l}.{'mlp' if kind in ('c_fc', 'down_proj') else 'attn'}.{kind}")(x)
        if iv is None or not (iv.parts_scale or iv.parts_swap):
            return out
        U, V = self.parts[(l, kind)]
        rows = list(range(x.shape[0])) if rows is None else rows
        out = out.clone()
        for j, r in enumerate(rows):
            e = iv.parts_scale.get(r)
            if e is not None and e[0] == l and e[1] == kind:
                _, _, idx, alpha = e
                out[j] += (alpha - 1.0) * ((x[j] @ V[:, idx]) @ U[idx])
            e = iv.parts_swap.get(r)
            if e is not None and e[0] == l and e[1] == kind:
                _, _, idx, values = e
                out[j, -1] += (values - x[j, -1] @ V[:, idx]) @ U[idx]
        return out

    def qkv(self, l, h, iv, rows=None):
        B, T, _ = h.shape
        q = self.site(l, "q_proj", h, iv, rows).view(B, T, self.H, self.hd).transpose(1, 2)
        k = self.site(l, "k_proj", h, iv, rows).view(B, T, self.H, self.hd).transpose(1, 2)
        v = self.site(l, "v_proj", h, iv, rows).view(B, T, self.H, self.hd).transpose(1, 2)
        return self.t._rope(q, T), self.t._rope(k, T), v

    @torch.no_grad()
    def forward(self, tokens, iv=None, record=None, cf=None):
        """generate.Qwen3.forward's contract on vpd4l (the same interventions, records and counterfactual writes)."""
        t = self.t
        B, T = tokens.shape
        x = t.wte[tokens]
        cut_delta = {}
        if record is not None:
            record.update(z_last=torch.empty(B, self.L, self.H, self.hd, device=self.dev), act_last=torch.empty(B, self.L, self.Fn, device=self.dev),
                          probe_values={}, site_in_last={}, writes={})
        cuts = iv.cuts if iv is not None else {}
        wanted = dict(cuts)  # cut writers, and on a run on x' the writes asked for (generate.Qwen3.forward)
        if record is not None:
            wanted.update({r: (ak, al, ah, None, None, None, None) for r, (ak, al, ah) in record.get("write_requests", {}).items()})
        writer = lambda r, write: self._writer(r, write, iv, record, cf, cut_delta)  # noqa: E731
        for l in range(self.L):
            h = VM.rms(x, t.norms[2 * l], t.eps)
            q, k, v = self.qkv(l, h, iv)
            z = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2)
            if record is not None:
                for r, (al, ah) in record.get("attend", {}).items():
                    if al == l:  # the head's attention weights from the last position
                        record.setdefault("attend_weights", {})[r] = torch.softmax((k[r, ah] @ q[r, ah, -1]) / self.hd ** 0.5, dim=-1)
            for r, (ak, al, ah, bk, bl, bh, route) in cuts.items():
                if bk == "head" and bl == l and r in cut_delta:
                    q2, k2, v2 = self.qkv(l, VM.rms(x[r : r + 1] + cut_delta[r], t.norms[2 * l], t.eps), iv, [r])
                    qh = (q2 if route == "query" else q[r : r + 1])[:, bh : bh + 1]
                    kh = (k2 if route == "key" else k[r : r + 1])[:, bh : bh + 1]
                    vh = (v2 if route == "value" else v[r : r + 1])[:, bh : bh + 1]
                    z = z.clone()
                    z[r, :, bh] = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True).transpose(1, 2)[0, :, 0]
            if iv is not None:
                if iv.head_swap is not None and bool(iv.head_swap[0][:, l].any()):
                    z = z.clone()
                    z[:, -1] = torch.where(iv.head_swap[0][:, l][..., None], iv.head_swap[1][:, l], z[:, -1])
                z = z * iv.head[:, l][:, None, :, None]
            if record is not None:
                record["z_last"][:, l] = z[:, -1]
                for kind in ("q_proj", "k_proj", "v_proj"):
                    record["site_in_last"][(l, kind)] = h[:, -1]
                record["site_in_last"][(l, "o_proj")] = z[:, -1].reshape(B, -1)
                for r, (pl, ph, pi) in record.get("probes", {}).items():
                    if pl == l and ph >= 0:
                        record["probe_values"][r] = torch.linalg.vector_norm(z[r, :, ph] @ self.Wo[l][:, ph].T, dim=-1)
                    elif pl == l and ph == -3 and pi[0] in ("q_proj", "k_proj", "v_proj"):
                        record["probe_values"][r] = h[r] @ self.parts[(l, pi[0])][1][:, pi[1]]
                    elif pl == l and ph == -3 and pi[0] == "o_proj":
                        record["probe_values"][r] = z[r].reshape(T, -1) @ self.parts[(l, "o_proj")][1][:, pi[1]]
            attn_out = self.site(l, "o_proj", z.reshape(B, T, -1), iv)
            for r, (ak, al, ah, bk, bl, bh, route) in wanted.items():
                if al == l and ak == "head":
                    writer(r, z[r, :, ah] @ self.Wo[l][:, ah].T)
                elif al == l and ak == "attn":
                    writer(r, attn_out[r])
            mid = x + attn_out
            y = VM.rms(mid, t.norms[2 * l + 1], t.eps)
            for r, (ak, al, ah, bk, bl, bh, route) in cuts.items():
                if bk == "mlp" and bl == l and r in cut_delta:
                    y = y.clone()
                    y[r] = VM.rms(mid[r] + cut_delta[r], t.norms[2 * l + 1], t.eps)
            pre = self.site(l, "c_fc", y, iv)
            act = VM.gelu_tanh(pre)
            if iv is not None:
                if iv.neuron_swap is not None and bool(iv.neuron_swap[0][:, l].any()):
                    act = act.clone()
                    act[:, -1] = torch.where(iv.neuron_swap[0][:, l], iv.neuron_swap[1][:, l], act[:, -1])
                act = act * iv.neuron[:, l][:, None, :]
            if record is not None:
                record["act_last"][:, l] = act[:, -1]
                record["site_in_last"][(l, "c_fc")] = y[:, -1]
                record["site_in_last"][(l, "down_proj")] = act[:, -1]
                for r, (pl, ph, pi) in record.get("probes", {}).items():
                    if pl == l and ph == -1:
                        record["probe_values"][r] = act[r, :, pi]
                    elif pl == l and ph == -3 and pi[0] in ("c_fc", "down_proj"):
                        inp = y[r] if pi[0] == "c_fc" else act[r]
                        record["probe_values"][r] = inp @ self.parts[(l, pi[0])][1][:, pi[1]]
            mlp_out = self.site(l, "down_proj", act, iv)
            for r, (ak, al, ah, bk, bl, bh, route) in wanted.items():
                if al == l and ak == "mlp":
                    writer(r, mlp_out[r])
            x = mid + mlp_out
        for r, (ak, al, ah, bk, bl, bh, route) in cuts.items():
            if bk == "logits":
                x = x.clone()
                x[r] = x[r] + cut_delta[r]
        return VM.rms(x[:, -1], t.ln_f, t.eps)

    @staticmethod
    def _writer(r, write, iv, record, cf, cut_delta):
        from generate import Qwen3

        Qwen3._writer(r, write, iv, record, cf, cut_delta)

    def log_probs(self, final):
        return torch.log_softmax((final @ self.t.wte.T).to(self.wide), dim=-1)

    @torch.no_grad()
    def greedy(self, tokens, steps, iv=None):
        out = []
        for _ in range(steps):
            nxt = self.log_probs(self.forward(tokens, iv)).argmax(-1)
            out.append(nxt)
            tokens = torch.cat([tokens, nxt[:, None]], dim=1)
        return torch.stack(out, dim=1)
