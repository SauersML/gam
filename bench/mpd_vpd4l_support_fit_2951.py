"""#2951: reshape the target's pieces so its per-position minimal supports shrink (Stage 2), no gate.

Thin torch executor for ``gamfit.sae.fit_supports`` (SPEC 8: the barrier, trust region, alternation and stopping rule
are the Rust owner ``parameter_decomposition::support_fit``'s; this file only runs the model and its derivatives).

Pieces are exact for every parameter value theta:
* OV, per head: ``W_O,h W_V,h = U S V^T = (U S A)(A^+ V^T)`` for any full-row-rank ``A`` (128 x K); piece i is column
  i of ``U S A`` with row i of ``A^+ V^T``, masked at the destination after attention mixing. ``A = I`` is Stage 1.
* c_fc, per layer: an overcomplete analysis ``V`` (K x 768, rows are reads) and the synthesis
  ``S = V^+ + Y (I - V V^+)`` (768 x K) for any ``Y``, so ``S V = I`` and ``W_fc = W_fc S V`` exactly for every
  ``(V, Y)``. Piece c reads ``v_c . x`` and adds ``W_fc s_c`` to every neuron's pre-activation. The canonical dual
  (``Y = 0``) gives each input its minimum-norm, dense coefficients; ``Y`` is the freedom that lets the pieces a
  token removes cancel. ``K`` is the neuron count and ``V`` starts at the neurons' normalized read directions (the
  architecture's own overcomplete reads, and VPD's ``neuron_aligned`` initialization).
* down_proj, per layer: its hidden-side (neuron) basis, fixed by the elementwise GELU.
* QK, per head: the 64 RoPE planes, fixed by RoPE's rotation group.
theta = (all A_h, all D_l), starting at identities (K = side dimension). Rank units per kept piece (VPD's currency):
OV 2 (v and o), c_fc 1, neuron 1, QK plane 4.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.func import jvp

from mpd_vpd4l_minimal_support_2951 import VPD, Pieces


class Reshaped:
    def __init__(self, base: Pieces):
        self.b = base
        L, H, hd, d = base.L, base.H, base.hd, base.d
        K = base.nmlp
        self.shapes = ([("A", l, h, (hd, hd)) for l in range(L) for h in range(H)]
                       + [("V", l, None, (K, d)) for l in range(L)] + [("Y", l, None, (d, K)) for l in range(L)])
        self.sizes = [a * b for *_, (a, b) in self.shapes]
        self.nqk, self.nov, self.nfc, self.nmlp = H * hd // 2, H * hd, K, base.nmlp
        self.per_layer = self.nqk + self.nov + self.nfc + self.nmlp
        self.units = L * self.per_layer
        self.rank_units = np.concatenate([np.concatenate([np.full(self.nqk, 4), np.full(self.nov, 2), np.full(self.nfc, 1),
                                                          np.full(self.nmlp, 1)]) for _ in range(L)])

    def initial_theta(self):
        parts = []
        for kind, l, h, (a, b) in self.shapes:
            if kind == "A":
                parts.append(torch.eye(a, b))
            elif kind == "V":
                wfc = self.b.w[f"h.{l}.mlp.c_fc.weight"].double().cpu()
                parts.append(wfc / wfc.norm(dim=1, keepdim=True))
            else:
                parts.append(torch.zeros(a, b))
        return torch.cat([t.double().reshape(-1) for t in parts])

    def unpack(self, theta):
        out, i = {}, 0
        for (kind, l, h, (a, b)), n in zip(self.shapes, self.sizes):
            out[(kind, l, h)] = theta[i:i + n].view(a, b)
            i += n
        return out

    def logits(self, ids, m, theta, context=None):
        b, w = self.b, self.b.w
        P = self.unpack(theta.float())
        h = w["wte.weight"][ids]
        B, T, _ = h.shape
        for l in range(b.L):
            p = f"h.{l}."
            ml = m[..., l * self.per_layer:(l + 1) * self.per_layer]
            x = b.rms(h, w[p + "rms_1.weight"])
            q = b.rope((x @ w[p + "attn.q_proj.weight"].T).view(B, T, b.H, b.hd).transpose(1, 2))
            k = b.rope((x @ w[p + "attn.k_proj.weight"].T).view(B, T, b.H, b.hd).transpose(1, 2))
            mqk = ml[..., :self.nqk].view(B, T, b.H, b.hd // 2).transpose(1, 2)
            q = q * torch.cat([mqk, mqk], -1)
            mov = ml[..., self.nqk:self.nqk + self.nov].view(B, T, b.H, b.hd).transpose(1, 2)
            vt = torch.stack([b.ov[l][hh][2] for hh in range(b.H)])
            z = torch.einsum("btd,hrd->bhtr", x, vt)
            # Explicit causal attention: forward-mode derivatives (jvp) are defined for every op here.
            causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
            if context is None:
                att = ((q @ k.transpose(-1, -2)) / b.hd ** 0.5).masked_fill(~causal, float("-inf")).softmax(-1)
                r = att @ z
            else:
                # Clean context: earlier positions' keys and values from the clean run, the own from this one.
                kc, zc = context[l]
                scores = q @ kc.transpose(-1, -2)
                scores = scores + torch.diag_embed((q * k).sum(-1) - scores.diagonal(dim1=-2, dim2=-1))
                att = (scores / b.hd ** 0.5).masked_fill(~causal, float("-inf")).softmax(-1)
                r = att @ zc + att.diagonal(dim1=-2, dim2=-1)[..., None] * (z - zc)
            A = torch.stack([P[("A", l, hh)] for hh in range(b.H)])            # H x 128 x K
            dual = torch.linalg.solve(A @ A.transpose(-1, -2), A)             # H x 128 x K: A dual^T = I
            c = torch.einsum("bhtr,hrk->bhtk", r, dual) * mov
            usa = torch.stack([b.ov[l][hh][0] * b.ov[l][hh][1] for hh in range(b.H)])  # H x 768 x 128
            h = h + torch.einsum("bhtk,hdk->btd", c, torch.einsum("hdr,hrk->hdk", usa, A))
            x = b.rms(h, w[p + "rms_2.weight"])
            V, Y = P[("V", l, None)], P[("Y", l, None)]                     # K x 768, 768 x K
            Vp = torch.linalg.solve(V.T @ V, V.T)                             # V^+ (768 x K), V^+ V = I
            S = Vp + Y - (Y @ V) @ Vp                                         # V^+ + Y (I - V V^+); S V = I
            coef = (x @ V.T) * ml[..., self.nqk + self.nov:self.nqk + self.nov + self.nfc]
            a = F.gelu(coef @ (w[p + "mlp.c_fc.weight"] @ S).T, approximate="tanh") * ml[..., self.nqk + self.nov + self.nfc:]
            h = h + a @ w[p + "mlp.down_proj.weight"].T
        return b.rms(h, w["ln_f.weight"]) @ w["wte.weight"].T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--seqs", type=int, required=True)
    parser.add_argument("--ctx", type=int, required=True)
    parser.add_argument("--eps", type=float, required=True)
    parser.add_argument("--form", choices=("per_position", "mean"), required=True,
                        help="admissible = every position below eps, or the batch mean (VPD's reported form)")
    parser.add_argument("--context", choices=("clean", "joint"), required=True,
                        help="search semantics, as in mpd_vpd4l_minimal_support_2951.py; final metrics are joint")
    parser.add_argument("--heldout-seqs", type=int, required=True,
                        help="unseen test sequences: supports are searched fresh on them with the fitted pieces frozen")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from gamfit.sae import fit_supports, minimal_support
    dev = "cuda"
    torch.manual_seed(0)
    model = Reshaped(Pieces(args.weights, dev))
    ids = torch.tensor(pq.read_table(args.test).slice(0, args.seqs).column("input_ids").to_pylist(),
                       dtype=torch.long)[:, :args.ctx].to(dev)
    B, T = ids.shape
    P, C = B * T, model.units
    with torch.no_grad():
        captured = []
        logp_clean = model.b(ids, capture=captured).log_softmax(-1)
    context = captured if args.context == "clean" else None
    calls = {"n": 0, "t": time.time(), "token_passes": 0}

    def spend(passes, positions):
        # Compute in VPD's currency: model passes (forward, backward or forward-mode) times positions.
        calls["token_passes"] += passes * positions

    def mask(keep):
        return torch.from_numpy(np.asarray(keep)).to(dev).view(B, T, C).float()

    def th(theta):
        return torch.from_numpy(np.asarray(theta)).to(dev)

    def kl_of(logits):
        return (logp_clean.exp() * (logp_clean - logits.log_softmax(-1))).sum(-1).reshape(-1)

    class Executor:
        def supports(self, theta, keep):
            calls["n"] += 1
            spend(3, P)
            m = mask(keep).requires_grad_(True)
            logits = model.logits(ids, m, th(theta), context)
            kl = kl_of(logits)
            (g,) = torch.autograd.grad(kl.sum(), m, retain_graph=True)
            with torch.no_grad():
                y = torch.distributions.Categorical(logits=logits).sample()
            (hh,) = torch.autograd.grad(logits.log_softmax(-1).gather(-1, y[..., None]).sum(), m)
            kept = np.asarray(keep).reshape(P, C)
            klv = kl.detach().double().cpu().numpy()
            print(f"[fit] supports {calls['n']}: rank units/position {(kept * model.rank_units).sum(1).mean():.1f}; "
                  f"KL max {klv.max():.4f} mean {klv.mean():.4f} ({time.time() - calls['t']:.0f}s)", flush=True)
            return (klv, (-g + 0.5 * hh.pow(2)).reshape(P, C).double().cpu().numpy(),
                    (-g - 0.5 * hh.pow(2)).reshape(P, C).double().cpu().numpy())

        def divergence(self, theta, keep):
            spend(1, P)
            with torch.no_grad():
                kl = kl_of(model.logits(ids, mask(keep), th(theta), context)).double().cpu().numpy()
            calls["divergence"] = calls.get("divergence", 0) + 1
            t_now = np.asarray(theta)
            calls.setdefault("theta0", t_now.copy())
            if calls["divergence"] % 10 == 0:
                kept = np.asarray(keep).reshape(P, C)
                print(f"[fit] barrier view {calls['divergence']}: rank units/position {(kept * model.rank_units).sum(1).mean():.1f} "
                      f"(fixed supports), KL mean {kl.mean():.4f} max {kl.max():.4f}; |theta - theta0| "
                      f"{np.linalg.norm(t_now - calls['theta0']):.4e} ({time.time() - calls['t']:.0f}s)", flush=True)
            return kl

        def tick(self, kind):
            calls[kind] = calls.get(kind, 0) + 1
            if calls[kind] % 10 == 0:
                print(f"[fit] {kind} calls {calls[kind]} ({time.time() - calls['t']:.0f}s)", flush=True)

        def weighted_gradient(self, theta, keep, weights):
            self.tick("gradient")
            spend(2, P)
            t = th(theta).requires_grad_(True)
            kl = kl_of(model.logits(ids, mask(keep), t, context))
            (g,) = torch.autograd.grad((kl * th(weights).float()).sum(), t)
            return g.double().cpu().numpy()

        def directional(self, theta, keep, v):
            spend(2, P)
            m = mask(keep)
            _, d = jvp(lambda t: kl_of(model.logits(ids, m, t, context)), (th(theta),), (th(v),))
            return d.double().cpu().numpy()

        def gradient_arithmetic(self):
            # float32 model evaluation; the longest reduction is the KL over positions and vocabulary.
            return 2.0 ** -24, P * logp_clean.shape[-1]

        def weighted_hessian(self, theta, keep, weights, v):
            # Exact Hessian product of sum_t w_t KL_t in theta, forward-over-reverse.
            self.tick("hessian")
            spend(3, P)
            m = mask(keep)
            w = th(weights).float()

            def objective_grad(t):
                return torch.func.grad(lambda s: (kl_of(model.logits(ids, m, s, context)) * w).sum())(t)

            _, hv = jvp(objective_grad, (th(theta),), (th(v),))
            return hv.double().cpu().numpy()

    theta0 = model.initial_theta().numpy()
    print(f"[fit] pieces {C}/position, theta {theta0.size}, P={P}, eps {args.eps}", flush=True)
    fit = fit_supports(Executor(), theta0, P, C, args.eps, args.form, 1 if args.context == "clean" else T)
    keep = fit["keep"]
    rank = (keep * model.rank_units).sum(1)
    with torch.no_grad():
        joint = kl_of(model.logits(ids, mask(keep), th(fit["theta"]))).double().cpu().numpy()
    report = {"args": vars(args), "token_passes_fit": calls["token_passes"], "alternations": fit["alternations"], "search_kl_max": float(fit["divergence"].max()),
              "search_kl_mean": float(fit["divergence"].mean()), "kl_max": float(joint.max()), "kl_mean": float(joint.mean()),
              "semantics_of_kl": "joint (VPD)", "rank_units_mean": float(rank.mean()),
              "rank_units_quantiles": np.quantile(rank, [0.1, 0.5, 0.9]).tolist(), "vpd": VPD}
    print(f"[fit] compute: {calls['token_passes']:.3e} token-passes (VPD p-8383f5e5: ~2.6e10 training tokens x 3 passes)", flush=True)
    print(f"[fit] FINAL rank units/position {report['rank_units_mean']:.1f} (VPD {VPD['l0']}); joint-mask KL max "
          f"{report['kl_max']:.4f} mean {report['kl_mean']:.4f}; alternations {fit['alternations']}", flush=True)
    np.save(args.out.replace(".json", "_theta.npy"), fit["theta"])
    # Held out: the fitted pieces frozen, supports searched fresh on unseen sequences (the number that generalizes).
    hids = torch.tensor(pq.read_table(args.test).slice(args.seqs, args.heldout_seqs).column("input_ids").to_pylist(),
                        dtype=torch.long)[:, :args.ctx].to(dev)
    HB = hids.shape[0]
    with torch.no_grad():
        hcap = []
        hlogp = model.b(hids, capture=hcap).log_softmax(-1)
    hcontext = hcap if args.context == "clean" else None
    theta_t = th(fit["theta"])

    def hkl(logits):
        return (hlogp.exp() * (hlogp - logits.log_softmax(-1))).sum(-1).reshape(-1)

    def hevaluate(keep):
        m = torch.from_numpy(np.asarray(keep)).to(dev).view(HB, T, C).float().requires_grad_(True)
        logits = model.logits(hids, m, theta_t, hcontext)
        kl = hkl(logits)
        (g,) = torch.autograd.grad(kl.sum(), m, retain_graph=True)
        with torch.no_grad():
            y = torch.distributions.Categorical(logits=logits).sample()
        (hh,) = torch.autograd.grad(logits.log_softmax(-1).gather(-1, y[..., None]).sum(), m)
        return (kl.detach().double().cpu().numpy(), (-g + 0.5 * hh.pow(2)).reshape(HB * T, C).double().cpu().numpy(),
                (-g - 0.5 * hh.pow(2)).reshape(HB * T, C).double().cpu().numpy())

    held = minimal_support(hevaluate, HB * T, C, args.eps, args.form, 1 if args.context == "clean" else T)
    hkeep = held["keep"]
    with torch.no_grad():
        hjoint = hkl(model.logits(hids, torch.from_numpy(hkeep).to(dev).view(HB, T, C).float(), theta_t)).double().cpu().numpy()
    report["heldout"] = {"seqs": HB, "rank_units_mean": float((hkeep * model.rank_units).sum(1).mean()),
                         "search_kl_max": float(held["divergence"].max()), "kl_mean": float(hjoint.mean()),
                         "kl_max": float(hjoint.max()), "semantics_of_kl": "joint (VPD)"}
    print(f"[fit] HELDOUT rank units/position {report['heldout']['rank_units_mean']:.1f} (VPD {VPD['l0']}); joint-mask KL "
          f"mean {report['heldout']['kl_mean']:.4f} max {report['heldout']['kl_max']:.4f}", flush=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=1)


if __name__ == "__main__":
    main()
