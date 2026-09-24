"""#2951: per-position minimal supports over the target's own units, found against the model, with no gate.

Thin torch executor for ``gamfit.sae.minimal_support`` (SPEC 8: the search, its acceptance and its stopping rule are
the Rust owner's; this file only runs the model). Target: VPD's own 4-layer LlamaSimpleMLP (Goodfire
t-9d2b8f02) on its Pile-uncopyrighted test split.

Pieces (they sum exactly to the target's parameters; none is chosen):
* MLP: one piece per neuron, spanning its c_fc row and its down_proj column. The elementwise GELU singles the
  neuron basis out.
* OV: per head, the singular directions of the exact product ``W_O,h W_V,h`` (768 x 768, rank 128). The head's
  output is ``sum_i u_i s_i (v_i . sum_s a_ts x_s)``, so a piece masks one term at the destination position.
* QK: per head, the 64 RoPE planes. RoPE rotates each plane by its own angle, so the query-key score is a sum of
  plane terms; a piece masks one plane of the query at the destination position.
With every mask on, the executor computes the target exactly up to floating-point reassociation, and the full-support
divergence it reports is that roundoff.

The removal prediction handed to the search is ``-g + h^2 / 2``: ``g`` the exact gradient of the summed divergence in
the masks and ``h`` the gradient of the log-likelihood of one label per position sampled from the current masked model
(a one-sample Gauss-Newton diagonal). It only orders and sizes proposals; acceptance reads the exact divergence.

Units reported in VPD's currency (rank per matrix kept): neuron 2 (c_fc row, down column), OV piece 2 (v, o),
QK plane 4 (two q rows, two k rows).
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

VPD = {"kl_rounded": 0.29135, "l0": 180.41, "pgd20": 0.6045, "run": "p-8383f5e5"}


class Pieces:
    def __init__(self, path, dev):
        from safetensors.torch import load_file
        self.w = {k: v.to(dev).float() for k, v in load_file(path).items()}
        self.L, self.H, self.d, self.eps, self.base = 4, 6, 768, 1e-6, 10000.0
        self.hd = self.d // self.H
        self.ov = []
        for l in range(self.L):
            p = f"h.{l}.attn."
            wv, wo = self.w[p + "v_proj.weight"].double(), self.w[p + "o_proj.weight"].double()
            heads = []
            for h in range(self.H):
                blk = slice(h * self.hd, (h + 1) * self.hd)
                u, s, vt = torch.linalg.svd(wo[:, blk] @ wv[blk], full_matrices=False)
                heads.append((u[:, :self.hd].float(), s[:self.hd].float(), vt[:self.hd].float()))
            self.ov.append(heads)
        self.nqk, self.nov, self.nmlp = self.H * self.hd // 2, self.H * self.hd, self.w["h.0.mlp.c_fc.weight"].shape[0]
        self.per_layer = self.nqk + self.nov + self.nmlp
        self.units = self.L * self.per_layer
        self.rank_units = np.concatenate([np.concatenate([np.full(self.nqk, 4), np.full(self.nov, 2),
                                                          np.full(self.nmlp, 2)]) for _ in range(self.L)])

    def rope(self, x):
        T = x.shape[-2]
        inv = 1.0 / (self.base ** (torch.arange(0, self.hd, 2, device=x.device, dtype=torch.float32) / self.hd))
        ang = torch.arange(T, device=x.device, dtype=torch.float32)[:, None] * inv[None]
        cos, sin = torch.cat([ang.cos(), ang.cos()], -1), torch.cat([ang.sin(), ang.sin()], -1)
        x1, x2 = x[..., : self.hd // 2], x[..., self.hd // 2:]
        return x * cos + torch.cat([-x2, x1], -1) * sin

    def rms(self, x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * w

    def __call__(self, ids, m=None, context=None, capture=None):
        """Logits; ``m`` is ``(B, T, units)`` in [0, 1] or None (native path).

        ``context`` (per layer: the clean run's rotated keys and OV-basis values, from ``capture``) makes the
        masked run clean-context: position t attends to the clean keys and values of every earlier position and to
        its own masked key and value, so its divergence depends on its own masks only. None is joint masking
        (VPD's semantics): every position's masks change what later positions read.
        """
        w = self.w
        h = w["wte.weight"][ids]
        B, T, _ = h.shape
        for l in range(self.L):
            p = f"h.{l}."
            x = self.rms(h, w[p + "rms_1.weight"])
            q = (x @ w[p + "attn.q_proj.weight"].T).view(B, T, self.H, self.hd).transpose(1, 2)
            k = (x @ w[p + "attn.k_proj.weight"].T).view(B, T, self.H, self.hd).transpose(1, 2)
            q, k = self.rope(q), self.rope(k)
            if capture is not None:
                vt = torch.stack([self.ov[l][hh][2] for hh in range(self.H)])
                capture.append((k, torch.einsum("btd,hrd->bhtr", x, vt)))
            if m is None:
                v = (x @ w[p + "attn.v_proj.weight"].T).view(B, T, self.H, self.hd).transpose(1, 2)
                o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                h = h + o.transpose(1, 2).reshape(B, T, self.d) @ w[p + "attn.o_proj.weight"].T
            else:
                ml = m[..., l * self.per_layer:(l + 1) * self.per_layer]
                mqk = ml[..., :self.nqk].view(B, T, self.H, self.hd // 2).transpose(1, 2)
                q = q * torch.cat([mqk, mqk], -1)
                mov = ml[..., self.nqk:self.nqk + self.nov].view(B, T, self.H, self.hd).transpose(1, 2)
                vt = torch.stack([self.ov[l][hh][2] for hh in range(self.H)])  # H x 128 x 768
                z = torch.einsum("btd,hrd->bhtr", x, vt)
                if context is None:
                    r = F.scaled_dot_product_attention(q, k, z, is_causal=True)  # B H T 128
                else:
                    kc, zc = context[l]
                    scores = q @ kc.transpose(-1, -2)
                    own = (q * k).sum(-1)
                    scores = scores + torch.diag_embed(own - scores.diagonal(dim1=-2, dim2=-1))
                    causal = torch.ones(T, T, dtype=torch.bool, device=scores.device).tril()
                    att = (scores / self.hd ** 0.5).masked_fill(~causal, float("-inf")).softmax(-1)
                    r = att @ zc + att.diagonal(dim1=-2, dim2=-1)[..., None] * (z - zc)
                s = torch.stack([self.ov[l][hh][1] for hh in range(self.H)])  # H x 128
                u = torch.stack([self.ov[l][hh][0] for hh in range(self.H)])  # H x 768 x 128
                h = h + torch.einsum("bhtr,hdr->btd", r * s[None, :, None, :] * mov, u)
            x = self.rms(h, w[p + "rms_2.weight"])
            a = F.gelu(x @ w[p + "mlp.c_fc.weight"].T, approximate="tanh")
            if m is not None:
                a = a * ml[..., self.nqk + self.nov:]
            h = h + a @ w[p + "mlp.down_proj.weight"].T
        return self.rms(h, w["ln_f.weight"]) @ w["wte.weight"].T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--seqs", type=int, required=True)
    parser.add_argument("--ctx", type=int, required=True)
    parser.add_argument("--eps", type=float, required=True, help="declared KL fidelity (nats)")
    parser.add_argument("--form", choices=("per_position", "mean"), required=True,
                        help="admissible = every position within eps, or the batch mean (VPD's reported form)")
    parser.add_argument("--context", choices=("clean", "joint"), required=True,
                        help="search semantics: clean = each position given its clean context (independent positions); "
                             "joint = every position masked at once (VPD's). Final metrics are reported under joint.")
    parser.add_argument("--pgd-steps", type=int, default=20, help="VPD PGDRecon protocol: sign steps of 0.1")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from gamfit.sae import minimal_support
    dev = "cuda"
    torch.manual_seed(0)
    model = Pieces(args.weights, dev)
    ids = torch.tensor(pq.read_table(args.test).slice(0, args.seqs).column("input_ids").to_pylist(),
                       dtype=torch.long)[:, :args.ctx].to(dev)
    B, T = ids.shape
    P, C = B * T, model.units
    with torch.no_grad():
        captured = []
        clean = model(ids, capture=captured)
        logp_clean = clean.log_softmax(-1)
        ce = F.cross_entropy(clean[:, :-1].reshape(-1, clean.shape[-1]), ids[:, 1:].reshape(-1)).item()
    context = captured if args.context == "clean" else None
    print(f"[ms] pieces {C} per position ({model.nqk} QK planes + {model.nov} OV + {model.nmlp} neurons per layer); "
          f"P={P}; clean CE {ce:.4f}", flush=True)
    calls = {"n": 0, "t": time.time()}

    def divergence(logits):
        lp = logits.log_softmax(-1)
        return (logp_clean.exp() * (logp_clean - lp)).sum(-1)  # B x T

    def evaluate(keep):
        calls["n"] += 1
        m = torch.from_numpy(keep).to(dev).view(B, T, C).float().requires_grad_(True)
        logits = model(ids, m, context=context)
        kl = divergence(logits)
        (g,) = torch.autograd.grad(kl.sum(), m, retain_graph=True)
        with torch.no_grad():
            y = torch.distributions.Categorical(logits=logits).sample()
        ll = logits.log_softmax(-1).gather(-1, y[..., None]).sum()
        (hh,) = torch.autograd.grad(ll, m)
        cost = (-g + 0.5 * hh.pow(2)).reshape(P, C).double().cpu().numpy()
        gain = (-g - 0.5 * hh.pow(2)).reshape(P, C).double().cpu().numpy()
        klv = kl.detach().reshape(P).double().cpu().numpy()
        kept = keep.reshape(P, C)
        print(f"[ms] eval {calls['n']}: kept/position {kept.sum(1).mean():.1f} pieces, "
              f"{(kept * model.rank_units).sum(1).mean():.1f} rank units; KL max {klv.max():.4f} mean {klv.mean():.4f} "
              f"({time.time() - calls['t']:.0f}s)", flush=True)
        return klv, cost, gain

    result = minimal_support(evaluate, P, C, args.eps, args.form, 1 if args.context == "clean" else T)
    keep = result["keep"]
    rank = (keep * model.rank_units).sum(1)
    with torch.no_grad():
        m = torch.from_numpy(keep).to(dev).view(B, T, C).float()
        kl = divergence(model(ids, m)).reshape(-1)  # joint masking: VPD's semantics
        kl_search = divergence(model(ids, m, context=context)).reshape(-1)
    # VPD's PGDRecon protocol on the found supports: every removed piece is re-added with an adversarial mask in
    # [0, 1] shared by all positions, from a uniform start, by sign-gradient ascent of the mean KL (step 0.1).
    keep_t = torch.from_numpy(keep).to(dev).view(B, T, C).float()
    adv = torch.rand(C, device=dev, generator=torch.Generator(device=dev).manual_seed(0)).requires_grad_(True)
    for _ in range(args.pgd_steps):
        pgd_kl = divergence(model(ids, keep_t + (1 - keep_t) * adv)).mean()
        (grad,) = torch.autograd.grad(pgd_kl, adv)
        with torch.no_grad():
            adv.add_(0.1 * grad.sign()).clamp_(0, 1)
    with torch.no_grad():
        pgd = divergence(model(ids, keep_t + (1 - keep_t) * adv)).mean().item()
    layer = keep.reshape(P, model.L, model.per_layer)
    report = {
        "args": vars(args), "clean_ce": ce, "pieces": C, "positions": P, "evaluations": calls["n"],
        "rounds": result["rounds"],
        "kl_mean": kl.mean().item(), "kl_max": kl.max().item(), "semantics_of_kl": "joint (VPD)",
        "search_context": args.context, "search_kl_mean": kl_search.mean().item(), "search_kl_max": kl_search.max().item(),
        "kept_pieces_mean": float(keep.sum(1).mean()), "rank_units_mean": float(rank.mean()),
        "rank_units_quantiles": np.quantile(rank, [0.1, 0.5, 0.9]).tolist(),
        "pgd_kl": pgd,
        "per_layer_kept": {f"layer{l}": {"qk_planes": float(layer[:, l, :model.nqk].sum(1).mean()),
                                         "ov": float(layer[:, l, model.nqk:model.nqk + model.nov].sum(1).mean()),
                                         "neurons": float(layer[:, l, model.nqk + model.nov:].sum(1).mean())}
                           for l in range(model.L)},
        "vpd": VPD,
    }
    print(f"[ms] FINAL joint-mask KL mean {report['kl_mean']:.4f} max {report['kl_max']:.4f}; search ({args.context}) KL "
          f"mean {report['search_kl_mean']:.4f} max {report['search_kl_max']:.4f} (declared {args.form} eps "
          f"{args.eps}); rank units/position {report['rank_units_mean']:.1f} (VPD {VPD['l0']} at mean KL "
          f"{VPD['kl_rounded']}); PGD{args.pgd_steps} KL {pgd:.4f} (VPD {VPD['pgd20']}); per layer {report['per_layer_kept']}", flush=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=1)


if __name__ == "__main__":
    main()
