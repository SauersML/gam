"""#2951: manifold parameter decomposition of VPD's own 4-layer Pile target, head to head with VPD.

Analysis under SPEC 8's exception (torch execution of the target and of the measurement).

The target is Goodfire's ``LlamaSimpleMLP`` 4L-768 (W&B run goodfire/spd/t-9d2b8f02, ``model_step_99999``), ported
from ``param_decomp/targets/llama_simple_mlp.py``'s declared forward: pre-RMSNorm blocks, rotate-half RoPE with
``inv_freq = base^(-2i/head_dim)``, causal attention, GELU(tanh) MLP ``c_fc -> gelu -> down_proj``, tied head, no
biases. The port is checked against the run's logged validation loss (2.7075).

The used weights of MLP ``l`` on token ``x`` are exactly ``W_eff(x) = W_down diag(g(x)) W_fc`` with the GELU gate
``g = 0.5 (1 + tanh(sqrt(2/pi)(a + 0.044715 a^3)))``, ``a = W_fc x`` (``gelu(a) = a g(a)``). A decomposition replaces
``g`` by a point of a fitted manifold of gates while the read ``a`` and the write ``W_down`` stay exact, in ALL FOUR
MLPs AT ONCE, each seeing the already-modified residual stream (error-propagating, VPD's own evaluation mode).
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F

from mpd_llm_chart_restriction_2951 import chart_project, kmeans, local_frames, nearest


class Target(torch.nn.Module):
    def __init__(self, path, dev):
        super().__init__()
        from safetensors.torch import load_file
        self.w = {k: v.to(dev) for k, v in load_file(path).items()}
        self.n_layer, self.n_head, self.d, self.eps, self.base = 4, 6, 768, 1e-6, 10000.0
        self.hd = self.d // self.n_head
        self.gate_fn = None  # (layer, g, a) -> g_hat
        self.query_fn = None  # (layer, q) -> q_hat, q: (tokens, heads, head_dim), before RoPE

    def rms(self, x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * w

    def rope(self, x):
        T = x.shape[-2]
        inv = 1.0 / (self.base ** (torch.arange(0, self.hd, 2, device=x.device, dtype=torch.float32) / self.hd))
        ang = torch.arange(T, device=x.device, dtype=torch.float32)[:, None] * inv[None]
        cos, sin = torch.cat([ang.cos(), ang.cos()], -1), torch.cat([ang.sin(), ang.sin()], -1)
        x1, x2 = x[..., : self.hd // 2], x[..., self.hd // 2:]
        return x * cos + torch.cat([-x2, x1], -1) * sin

    def forward(self, ids, grab=None, grab_q=None):
        w = self.w
        h = w["wte.weight"][ids]
        B, T, _ = h.shape
        for l in range(self.n_layer):
            p = f"h.{l}."
            x = self.rms(h, w[p + "rms_1.weight"])
            q, k, v = (
                (x @ w[p + f"attn.{n}_proj.weight"].T).view(B, T, self.n_head, self.hd).transpose(1, 2)
                for n in ("q", "k", "v"))
            if grab_q is not None:
                grab_q[l].append(q.transpose(1, 2).reshape(B * T, self.n_head, self.hd))
            if self.query_fn is not None:
                flat = self.query_fn(l, q.transpose(1, 2).reshape(B * T, self.n_head, self.hd))
                q = flat.view(B, T, self.n_head, self.hd).transpose(1, 2)
            o = F.scaled_dot_product_attention(self.rope(q), self.rope(k), v, is_causal=True)
            h = h + o.transpose(1, 2).reshape(B, T, self.d) @ w[p + "attn.o_proj.weight"].T
            x = self.rms(h, w[p + "rms_2.weight"])
            a = x @ w[p + "mlp.c_fc.weight"].T
            g = 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (a + 0.044715 * a.pow(3))))
            if grab is not None:
                grab[l].append(g.reshape(-1, g.shape[-1]))
            if self.gate_fn is not None:
                g = self.gate_fn(l, g.reshape(-1, g.shape[-1]), a.reshape(-1, a.shape[-1])).reshape(g.shape)
            h = h + (g * a) @ w[p + "mlp.down_proj.weight"].T
        return self.rms(h, w["ln_f.weight"]) @ w["wte.weight"].T


def pile_batches(tok_path, docs, seq_len, batch):
    from datasets import load_dataset
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    eos = tok.token_to_id("<|endoftext|>")
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    buf = []
    for i in docs:
        buf.extend(tok.encode(ds[i]["text"]).ids + [eos])
        while len(buf) >= seq_len * batch:
            yield torch.tensor(buf[: seq_len * batch]).view(batch, seq_len)
            buf = buf[seq_len * batch:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--bank-batches", type=int, required=True)
    parser.add_argument("--eval-batches", type=int, required=True)
    parser.add_argument("--charts", required=True)
    parser.add_argument("--dims", required=True)
    parser.add_argument("--neurons", required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda"
    model = Target(args.weights, dev)
    held = list(zip(range(args.eval_batches), pile_batches(args.tokenizer, range(9000, 10000), args.seq_len,
                                                            args.batch)))
    held = [b for _, b in held]
    report = {"args": vars(args)}
    with torch.inference_mode():
        clean, ce = [], 0.0
        for ids in held:
            logits = model(ids.to(dev)).float()
            lp = torch.log_softmax(logits, -1)
            clean.append(lp.cpu())
            ce += F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1).to(dev)).item()
    report["clean_ce"] = ce / len(held)
    print(f"[vpd4l] clean CE {report['clean_ce']:.4f} (run's logged val loss 2.7075)", flush=True)

    def evaluate(fn, qfn=None):
        model.gate_fn = fn
        model.query_fn = qfn
        kl = ce = 0.0
        n = 0
        with torch.inference_mode():
            for ids, ref in zip(held, clean):
                logits = model(ids.to(dev)).float()
                lp = torch.log_softmax(logits, -1)
                r = ref.to(dev)
                kl += (r.exp() * (r - lp)).sum().item()
                n += r.shape[0] * r.shape[1]
                ce += F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                      ids[:, 1:].reshape(-1).to(dev)).item()
        model.gate_fn = None
        model.query_fn = None
        return {"kl": kl / n, "ce_difference": ce / len(held) - report["clean_ce"]}

    grab = [[] for _ in range(model.n_layer)]
    grab_q = [[] for _ in range(model.n_layer)]
    t0 = time.time()
    with torch.inference_mode():
        for _, ids in zip(range(args.bank_batches), pile_batches(args.tokenizer, range(0, 9000), args.seq_len,
                                                                 args.batch)):
            model(ids.to(dev), grab=grab, grab_q=grab_q)
    G = [torch.cat(g) for g in grab]
    Q = [torch.cat(q) for q in grab_q]  # tokens x heads x head_dim
    del grab, grab_q
    print(f"[vpd4l] gate bank {tuple(G[0].shape)} x {len(G)} layers in {time.time() - t0:.0f}s", flush=True)
    mus = [g.mean(0) for g in G]
    qmus = [q.mean(0) for q in Q]
    report["mean_gate"] = evaluate(lambda l, g, a: mus[l].expand_as(g))
    print(f"[vpd4l] every MLP at its mean gate: {report['mean_gate']}", flush=True)
    report["mean_query"] = evaluate(None, lambda l, q: qmus[l].expand_as(q).clone())
    print(f"[vpd4l] every head at its mean query: {report['mean_query']}", flush=True)
    report["neurons"] = {}
    dn = [model.w[f"h.{l}.mlp.down_proj.weight"].norm(dim=0) for l in range(model.n_layer)]
    for k in [int(v) for v in args.neurons.split(",")]:
        def top(l, g, a, k=k):
            score = ((g - mus[l]) * a).abs() * dn[l]
            keep = torch.zeros_like(g).scatter_(1, score.topk(k, dim=1).indices, 1.0)
            return mus[l] + (g - mus[l]) * keep
        report["neurons"][k] = evaluate(top)
        print(f"[vpd4l] top-{k} neurons per MLP per token (rest at mean): {report['neurons'][k]}", flush=True)
    report["atlas"] = {}
    dims = [int(d) for d in args.dims.split(",")]
    gen = torch.Generator().manual_seed(args.seed)
    for C in [int(c) for c in args.charts.split(",")]:
        atl, qatl = [], []
        t0 = time.time()
        for l in range(model.n_layer):
            centres, assign = kmeans(G[l], C, 25, gen)
            m, fr = local_frames(G[l], assign, C, max(dims))
            atl.append((centres, m, fr))
            heads = []
            for hh in range(model.n_head):
                X = Q[l][:, hh].contiguous()
                qc, qa = kmeans(X, C, 25, gen)
                qm, qf = local_frames(X, qa, C, min(max(dims), model.hd))
                heads.append((qc, qm, qf))
            qatl.append(heads)
        fit_s = time.time() - t0
        for d in dims:
            def chart(l, g, a, d=d):
                centres, m, fr = atl[l]
                return chart_project(g, nearest(g, centres), m, fr[:, :d])

            def qchart(l, q, d=d):
                out = torch.empty_like(q)
                for hh, (qc, qm, qf) in enumerate(qatl[l]):
                    x = q[:, hh].contiguous()
                    out[:, hh] = chart_project(x, nearest(x, qc), qm, qf[:, :min(d, qf.shape[1])])
                return out
            for name, fn, qfn in (("mlp", chart, None), ("attn", None, qchart), ("both", chart, qchart)):
                res = evaluate(fn, qfn)
                res["fit_seconds_all_layers"] = fit_s
                report["atlas"][f"{name}:{C}x{d}"] = res
                print(f"[vpd4l] atlas {name} C={C} d={d}: {res}", flush=True)
        del atl, qatl
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=1)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=1)
    print(f"[vpd4l] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
