"""#2951 Stage 2 with every weight matrix's pieces learned: rank-one pieces of all 24 matrices of VPD's 4-layer target.

Thin torch executor for ``gamfit.sae.fit_supports`` (SPEC 8: the barrier, trust region, fidelity path and stopping
rule are the Rust owner's; this file only runs the model and its derivatives).

Every matrix ``W`` (out x in) of q, k, v, o, c_fc and down_proj in every layer is split on its input side by an analysis
``V`` (K x in, rows are reads) and a synthesis ``S`` (in x K) with ``S V = I``, so ``W = W S V`` exactly for every
parameter value. ``--pieces frame`` keeps every matrix's pieces a tight frame: ``V`` has orthonormal columns
(``V^T V = I``) and ``S = V^T``, and ``V`` moves on the Stiefel manifold (``gamfit.sae.fit_supports``'s ``frames``: the
polar retraction and the Riemannian Hessian are gam_geometry's), so there is no free factor to drift; any partial
removal ``0 <= D <= I`` of pieces then takes out ``W V^T D V``, never more than ``W``, so pieces cannot cancel one
another. (An unconstrained ``A`` with ``V = A L^{-T}``, ``A^T A = L L^T``, left its triangular factor free: on the
4-layer target it drifted until ``A^T A`` lost positive definiteness in float32 at a down_proj after 4.5 hours.)
``--pieces dual`` is the general split ``S = V^+ + Y (I - V V^+)``, where they can (on the modular-addition model the
fit drove an overcomplete analysis toward singularity, condition number 2.5e3 to 2.3e5, and from the full support no
single piece was removable). Piece c of ``W`` reads ``v_c . x`` and writes ``W s_c``: a rank-one piece, the unit VPD
counts, so rank units here are VPD's L0. Nothing is pinned: the elementwise GELU fixes only that c_fc's output feeds
it, and RoPE acts after the query and key projections, so any split of any matrix is exact. Starting pieces and counts
come from the architecture and reproduce VPD's own component counts: q, k, v, o start at their own singular
decomposition (reads = right singular vectors, K = 768: the matrix's exact rank-one split, where the residual stream's
coordinate axes would mean nothing to it); c_fc at the neurons' normalized read directions (K = 3072; VPD's
neuron-aligned start), or under ``frame`` the tight frame nearest them (their polar factor); down_proj at the neuron basis its input lives in (K = 3072). 36,864 pieces per position in all, VPD's C. Masks act at the position the matrix is applied at, and
KL is under joint masking (VPD's semantics).
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

MATS = ("q", "k", "v", "o", "fc", "down")


class AllPieces:
    def __init__(self, base: Pieces, frame: bool):
        self.b = base
        self.frame = frame
        w, L = base.w, base.L
        self.names = {"q": "attn.q_proj.weight", "k": "attn.k_proj.weight", "v": "attn.v_proj.weight",
                      "o": "attn.o_proj.weight", "fc": "mlp.c_fc.weight", "down": "mlp.down_proj.weight"}
        self.shapes, self.K = [], {}
        for l in range(L):
            for m in MATS:
                W = w[f"h.{l}.{self.names[m]}"]
                K = W.shape[1] if m in ("q", "k", "v", "o", "down") else W.shape[0]
                self.K[(l, m)] = K
                self.shapes.append((l, m, "V", (K, W.shape[1])))
                if not frame:
                    self.shapes.append((l, m, "Y", (W.shape[1], K)))
        self.sizes = [a * c for *_, (a, c) in self.shapes]
        self.offsets = {}
        o = 0
        for l in range(L):
            for m in MATS:
                self.offsets[(l, m)] = (o, o + self.K[(l, m)])
                o += self.K[(l, m)]
        self.units = o
        self.rank_units = np.ones(self.units)

    def frames(self):
        """The Stiefel blocks theta is made of, in order (``--pieces frame``)."""
        return [shape for (_l, _m, kind, shape) in self.shapes if kind == "V"]

    def initial_theta(self):
        parts = []
        for l, m, kind, (a, c) in self.shapes:
            if kind == "Y":
                parts.append(torch.zeros(a, c))
            elif m == "fc":
                W = self.b.w[f"h.{l}.{self.names[m]}"].double().cpu()
                reads = W / W.norm(dim=1, keepdim=True)
                if self.frame:
                    u, _, vt = torch.linalg.svd(reads, full_matrices=False)
                    reads = u @ vt
                parts.append(reads)
            elif m == "down":
                parts.append(torch.eye(a, c))
            else:
                # The matrix's own rank-one decomposition W = sum_i s_i u_i v_i^T: reads are its right
                # singular vectors (the coordinate axes of the residual stream mean nothing to W).
                W = self.b.w[f"h.{l}.{self.names[m]}"].double().cpu()
                parts.append(torch.linalg.svd(W, full_matrices=True)[2])
        return torch.cat([t.double().reshape(-1) for t in parts])

    def unpack(self, theta):
        out, i = {}, 0
        for (l, m, kind, (a, c)), n in zip(self.shapes, self.sizes):
            out[(l, m, kind)] = theta[i:i + n].view(a, c)
            i += n
        return out

    def logits(self, ids, mask, theta):
        b, w = self.b, self.b.w
        P = self.unpack(theta.float())
        h = w["wte.weight"][ids]
        B, T, _ = h.shape
        causal = torch.ones(T, T, dtype=torch.bool, device=h.device).tril()

        def apply(l, m, x):
            if self.frame:
                V = P[(l, m, "V")]
                S = V.T
            else:
                V, Y = P[(l, m, "V")], P[(l, m, "Y")]
                Vp = torch.linalg.solve(V.T @ V, V.T)
                S = Vp + Y - (Y @ V) @ Vp
            s0, s1 = self.offsets[(l, m)]
            coef = (x @ V.T) * mask[..., s0:s1]
            return coef @ (w[f"h.{l}.{self.names[m]}"] @ S).T

        for l in range(b.L):
            x = b.rms(h, w[f"h.{l}.rms_1.weight"])
            q = b.rope(apply(l, "q", x).view(B, T, b.H, b.hd).transpose(1, 2))
            k = b.rope(apply(l, "k", x).view(B, T, b.H, b.hd).transpose(1, 2))
            v = apply(l, "v", x).view(B, T, b.H, b.hd).transpose(1, 2)
            att = ((q @ k.transpose(-1, -2)) / b.hd ** 0.5).masked_fill(~causal, float("-inf")).softmax(-1)
            h = h + apply(l, "o", (att @ v).transpose(1, 2).reshape(B, T, b.d))
            x = b.rms(h, w[f"h.{l}.rms_2.weight"])
            h = h + apply(l, "down", F.gelu(apply(l, "fc", x), approximate="tanh"))
        return b.rms(h, w["ln_f.weight"]) @ w["wte.weight"].T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--seqs", type=int, required=True)
    parser.add_argument("--ctx", type=int, required=True)
    parser.add_argument("--eps", type=float, required=True)
    parser.add_argument("--form", choices=("per_position", "mean"), required=True)
    parser.add_argument("--heldout-seqs", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pieces", choices=("frame", "dual"), default="frame")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from gamfit.sae import fit_supports, minimal_support
    dev = args.device
    torch.manual_seed(0)
    model = AllPieces(Pieces(args.weights, dev), args.pieces == "frame")
    table = pq.read_table(args.test)
    ids = torch.tensor(table.slice(0, args.seqs).column("input_ids").to_pylist(), dtype=torch.long)[:, :args.ctx].to(dev)
    hids = torch.tensor(table.slice(args.seqs, args.heldout_seqs).column("input_ids").to_pylist(),
                        dtype=torch.long)[:, :args.ctx].to(dev)
    C = model.units
    calls = {"t": time.time(), "token_passes": 0}

    def run(tokens):
        with torch.no_grad():
            logp_clean = model.b(tokens).log_softmax(-1)
        Bn, T = tokens.shape
        P = Bn * T

        def kl_of(lg):
            return (logp_clean.exp() * (logp_clean - lg.log_softmax(-1))).sum(-1).reshape(-1)

        # gamfit hands over the same array object while the pieces or supports are unchanged (every
        # Hessian product of a trust-region iteration shares them), so each is uploaded once.
        uploaded = {}

        def once(kind, a, convert):
            held = uploaded.get(kind)
            if held is not None and held[0] is a:
                return held[1]
            value = convert(a)
            uploaded[kind] = (a, value)
            return value

        def mask(keep):
            return once("keep", keep, lambda k: torch.from_numpy(np.asarray(k)).to(dev).view(Bn, T, C).float())

        def pieces(theta):
            return once("theta", theta, lambda t: torch.from_numpy(np.asarray(t)).to(dev))

        def th(a):
            return torch.from_numpy(np.asarray(a)).to(dev)

        def spend(passes):
            calls["token_passes"] += passes * P

        class Executor:
            def supports(self, theta, keep):
                calls["theta"] = theta
                spend(3)
                m = mask(keep).detach().requires_grad_(True)
                lg = model.logits(tokens, m, pieces(theta))
                kl = kl_of(lg)
                (g,) = torch.autograd.grad(kl.sum(), m, retain_graph=True)
                with torch.no_grad():
                    y = torch.distributions.Categorical(logits=lg).sample()
                (hh,) = torch.autograd.grad(lg.log_softmax(-1).gather(-1, y[..., None]).sum(), m)
                klv = kl.detach().double().cpu().numpy()
                kept = np.asarray(keep).reshape(P, C)
                print(f"[all] supports: rank units/position {kept.sum(1).mean():.1f}; KL mean {klv.mean():.4f} "
                      f"max {klv.max():.4f} ({time.time() - calls['t']:.0f}s, {calls['token_passes']:.2e} token-passes)",
                      flush=True)
                g, hh = g.reshape(P, C).double(), hh.reshape(P, C).double()
                return klv, (-g + 0.5 * hh ** 2).cpu().numpy(), (-g - 0.5 * hh ** 2).cpu().numpy()

            def divergence(self, theta, keep):
                # support_fit's last divergence before each alternation is reported is at the
                # accepted pieces (trial points come earlier)
                calls["theta"] = theta
                spend(1)
                with torch.no_grad():
                    kl = kl_of(model.logits(tokens, mask(keep), pieces(theta))).double().cpu().numpy()
                calls["views"] = calls.get("views", 0) + 1
                if calls["views"] % 10 == 0:
                    kept = np.asarray(keep).reshape(P, C)
                    print(f"[all] barrier view {calls['views']}: rank units/position {kept.sum(1).mean():.1f} "
                          f"(fixed supports), KL mean {kl.mean():.4f} max {kl.max():.4f} "
                          f"({time.time() - calls['t']:.0f}s)", flush=True)
                return kl

            def weighted_gradient(self, theta, keep, weights):
                spend(2)
                t = pieces(theta).detach().requires_grad_(True)
                (g,) = torch.autograd.grad((kl_of(model.logits(tokens, mask(keep), t)) * th(weights).float()).sum(), t)
                return g.double().cpu().numpy()

            def directional(self, theta, keep, v):
                spend(2)
                m = mask(keep)
                _, d = jvp(lambda t: kl_of(model.logits(tokens, m, t)), (pieces(theta),), (th(v),))
                return d.double().cpu().numpy()

            def observe(self, a):
                # The pieces each level ends with, for a held-out read of the whole fidelity path
                # (mpd_vpd4l_heldout_levels_2951.py): saved when the next level starts.
                previous = calls.get("level")
                if previous is not None and a["level"] != previous and "theta" in calls:
                    np.save(args.out.replace(".json", f"_level{previous:.6g}.npy"), np.asarray(calls["theta"], dtype=np.float32))
                calls["level"] = a["level"]
                print(f"[alt] level {a['level']:.4g}: kept {a['kept']} pieces, KL mean {a['mean_divergence']:.4f}; "
                      f"barrier {a['barrier_before']:.6g} -> {a['barrier_after']:.6g}; step {a['step']:.3e} radius {a['radius']:.3e}; "
                      f"residual {a['residual']:.3e} / {a['tolerance']:.3e} certified {a['certified']} searched {a['searched']}", flush=True)

            def gradient_arithmetic(self):
                return 2.0 ** -24, P * logp_clean.shape[-1]

            def weighted_hessian(self, theta, keep, weights, v):
                spend(3)
                m = mask(keep)
                wt = th(weights).float()

                def grad_of(t):
                    return torch.func.grad(lambda s: (kl_of(model.logits(tokens, m, s)) * wt).sum())(t)

                _, hv = jvp(grad_of, (pieces(theta),), (th(v),))
                return hv.double().cpu().numpy()

        return Executor(), P, T, kl_of, mask, th

    executor, P, T, kl_of, mask, th = run(ids)
    theta0 = model.initial_theta().numpy()
    print(f"[all] {C} rank-one pieces/position over 24 matrices, theta {theta0.size}, P={P}, eps {args.eps} {args.form}",
          flush=True)
    fit = fit_supports(executor, theta0, P, C, args.eps, args.form, T, model.frames() if model.frame else None)
    keep = fit["keep"]
    with torch.no_grad():
        joint = kl_of(model.logits(ids, mask(keep), th(fit["theta"]))).double().cpu().numpy()
    fit_passes = calls["token_passes"]
    hexec, HP, HT, hkl_of, hmask, _ = run(hids)
    held = minimal_support(lambda k: hexec.supports(fit["theta"], k), HP, C, args.eps, args.form, HT)
    # VPD's PGDRecon protocol on the held-out supports: every removed piece is re-added with an adversarial mask in
    # [0, 1] shared by all positions, from a uniform start, by 20 sign-gradient steps of 0.1 on the mean KL.
    hkeep_t = hmask(held["keep"])
    theta_t = th(fit["theta"])
    adv = torch.rand(C, device=dev, generator=torch.Generator(device=dev).manual_seed(0)).requires_grad_(True)
    for _ in range(20):
        (grad,) = torch.autograd.grad(hkl_of(model.logits(hids, hkeep_t + (1 - hkeep_t) * adv, theta_t)).mean(), adv)
        with torch.no_grad():
            adv.add_(0.1 * grad.sign()).clamp_(0, 1)
    with torch.no_grad():
        pgd = hkl_of(model.logits(hids, hkeep_t + (1 - hkeep_t) * adv, theta_t)).mean().item()
    report = {"args": vars(args), "token_passes_fit": fit_passes, "alternations": fit["alternations"],
              "fit": {"rank_units_mean": float(keep.sum(1).mean()), "kl_mean": float(joint.mean()),
                      "kl_max": float(joint.max())},
              "heldout": {"rank_units_mean": float(held["keep"].sum(1).mean()),
                          "kl_mean": float(held["divergence"].mean()), "kl_max": float(held["divergence"].max()),
                          "pgd20_kl": pgd},
              "vpd": VPD}
    report["rank_units_mean"], report["kl_mean"] = report["fit"]["rank_units_mean"], report["fit"]["kl_mean"]
    print(f"[all] FINAL fit: L0 {report['fit']['rank_units_mean']:.1f} at KL {report['fit']['kl_mean']:.4f}; "
          f"compute {fit_passes:.3e} token-passes (VPD ~2.6e10 tokens x 3)", flush=True)
    print(f"[all] HELDOUT L0 {report['heldout']['rank_units_mean']:.1f} at KL {report['heldout']['kl_mean']:.4f} "
          f"(VPD {VPD['l0']} at {VPD['kl_rounded']}); PGD-20 KL {pgd:.4f} (VPD {VPD['pgd20']})", flush=True)
    np.save(args.out.replace(".json", "_theta.npy"), fit["theta"])
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=1)


if __name__ == "__main__":
    main()
