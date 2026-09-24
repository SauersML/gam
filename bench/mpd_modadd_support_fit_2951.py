"""#2951: Stage 2 on the grokked modular-addition model, where the mechanism is known.

Thin torch executor for ``gamfit.sae.fit_supports`` (SPEC 8: the barrier, trust region, fidelity path and stopping
rule are the Rust owner's; this file runs the model and its derivatives). One example is one position (the model is
read at ``=``), so positions are independent.

Pieces, exact for every theta:
* MLP input: an analysis ``V`` (K x d_model, K = d_mlp) and synthesis ``S = V^+ + Y (I - V V^+)``, ``S V = I``;
  piece c reads ``v_c . x`` and adds ``W_in s_c`` to every neuron's pre-activation. ``V`` starts at the neurons'
  normalized read directions (the rows of ``W_in``).
* neurons (ReLU fixes the basis), and each head's exact query-key and output-value SVD pieces.
theta = (V, Y). Fit on some examples, then held out: supports searched fresh on unseen examples with theta frozen.
Scored against the known mechanism: principal-angle overlap of the kept read directions with the key Fourier planes
of the embedding.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
from torch.func import jvp

from mpd_modadd_2951 import all_pairs, build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--fit-examples", type=int, required=True)
    parser.add_argument("--heldout-examples", type=int, required=True)
    parser.add_argument("--eps", type=float, required=True)
    parser.add_argument("--form", choices=("per_position", "mean"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from gamfit.sae import fit_supports, minimal_support
    torch.set_default_dtype(torch.float64)
    run = torch.load(args.run, map_location="cpu", weights_only=True)
    model = build_model(run["config"]).double()
    model.load_state_dict(run["checkpoints"][max(run["checkpoints"])])
    for prm in model.parameters():
        prm.requires_grad_(False)
    p, H, dh = run["config"]["p"], run["config"]["n_heads"], run["config"]["d_head"]
    d, n_mlp = model.W_in.shape[1], model.W_in.shape[0]
    K = n_mlp
    order = torch.randperm(p * p, generator=torch.Generator().manual_seed(0))
    fit_tokens = all_pairs(p)[order[:args.fit_examples]]
    held_tokens = all_pairs(p)[order[args.fit_examples:args.fit_examples + args.heldout_examples]]
    qk, ov = [], []
    for h in range(H):
        u, s, vt = torch.linalg.svd(model.W_Q[h].T @ model.W_K[h], full_matrices=False)
        qk.append((u[:, :dh], s[:dh], vt[:dh]))
        u, s, vt = torch.linalg.svd(model.W_O[:, h * dh:(h + 1) * dh] @ model.W_V[h], full_matrices=False)
        ov.append((u[:, :dh], s[:dh], vt[:dh]))
    C = 2 * H * dh + K + n_mlp
    units = np.concatenate([np.full(2 * H * dh, 2), np.full(K, 1), np.full(n_mlp, 1)])

    def unpack(theta):
        return theta[:K * d].view(K, d), theta[K * d:].view(d, K)

    def logits(tokens, m, theta):
        V, Y = unpack(theta)
        Vp = torch.linalg.solve(V.T @ V, V.T)
        S = Vp + Y - (Y @ V) @ Vp
        x = torch.stack([model.W_E[tokens[:, u]] for u in range(3)], 1) + model.W_pos
        mq, mo = m[:, :H * dh].view(-1, H, dh), m[:, H * dh:2 * H * dh].view(-1, H, dh)
        mc, mn = m[:, 2 * H * dh:2 * H * dh + K], m[:, 2 * H * dh + K:]
        dest = x[:, -1]
        out = torch.zeros_like(dest)
        for h in range(H):
            u, s, vt = qk[h]
            scores = torch.einsum("nr,npr->np", (dest @ u) * s * mq[:, h], x @ vt.T) / math.sqrt(dh)
            pattern = torch.softmax(scores, -1)
            u, s, vt = ov[h]
            out = out + ((torch.einsum("np,npd->nd", pattern, x) @ vt.T) * s * mo[:, h]) @ u.T
        dest = dest + out
        coef = (dest @ V.T) * mc
        a = torch.relu(coef @ (model.W_in @ S).T + model.b_in) * mn
        dest = dest + a @ model.W_out.T + model.b_out
        return dest @ model.W_U.T

    def make_executor(tokens):
        with torch.no_grad():
            clean = model(tokens).log_softmax(-1)
        N = len(tokens)

        def kl_of(lg):
            return (clean.exp() * (clean - lg.log_softmax(-1))).sum(-1)

        def mask(keep):
            return torch.from_numpy(np.asarray(keep)).double()

        class Executor:
            def supports(self, theta, keep):
                t = torch.from_numpy(np.asarray(theta))
                m = mask(keep).requires_grad_(True)
                lg = logits(tokens, m, t)
                kl = kl_of(lg)
                (g,) = torch.autograd.grad(kl.sum(), m, retain_graph=True)
                y = torch.distributions.Categorical(logits=lg.detach()).sample()
                (hh,) = torch.autograd.grad(lg.log_softmax(-1).gather(-1, y[:, None]).sum(), m)
                return kl.detach().numpy(), (-g + 0.5 * hh ** 2).numpy(), (-g - 0.5 * hh ** 2).numpy()

            def divergence(self, theta, keep):
                with torch.no_grad():
                    return kl_of(logits(tokens, mask(keep), torch.from_numpy(np.asarray(theta)))).numpy()

            def weighted_gradient(self, theta, keep, weights):
                t = torch.from_numpy(np.asarray(theta)).clone().requires_grad_(True)
                kl = kl_of(logits(tokens, mask(keep), t))
                (g,) = torch.autograd.grad((kl * torch.from_numpy(np.asarray(weights))).sum(), t)
                return g.numpy()

            def directional(self, theta, keep, v):
                m = mask(keep)
                _, dd = jvp(lambda t: kl_of(logits(tokens, m, t)), (torch.from_numpy(np.asarray(theta)),),
                            (torch.from_numpy(np.asarray(v)),))
                return dd.numpy()

            def weighted_gauss_newton(self, theta, keep, weights, v):
                m = mask(keep)
                t = torch.from_numpy(np.asarray(theta)).clone().requires_grad_(True)
                lg = logits(tokens, m, t)
                _, u = jvp(lambda s: logits(tokens, m, s), (torch.from_numpy(np.asarray(theta)),),
                           (torch.from_numpy(np.asarray(v)),))
                q = lg.softmax(-1).detach()
                fu = (q * u - q * (q * u).sum(-1, keepdim=True)) * torch.from_numpy(np.asarray(weights))[:, None]
                (out,) = torch.autograd.grad(lg, t, grad_outputs=fu)
                return out.numpy()

        return Executor(), N

    w_in = model.W_in.detach()
    theta0 = torch.cat([(w_in / w_in.norm(dim=1, keepdim=True)).reshape(-1), torch.zeros(d * K)]).numpy()
    t0 = time.time()
    executor, N = make_executor(fit_tokens)
    fit = fit_supports(executor, theta0, N, C, args.eps, args.form, 1)
    keep = fit["keep"]
    fitted = float((keep * units).sum(1).mean())
    print(f"[mamsf] fit: {fitted:.1f} rank units/example of {units.sum()} (KL mean {fit['divergence'].mean():.4f}); "
          f"{len(fit['alternations'])} alternations, {time.time() - t0:.0f}s", flush=True)
    held_exec, HN = make_executor(held_tokens)
    theta = fit["theta"]
    held = minimal_support(lambda k: held_exec.supports(theta, k), HN, C, args.eps, args.form, 1)
    hkeep = held["keep"]
    held_units = float((hkeep * units).sum(1).mean())
    # the same held-out search with the starting pieces (theta0), for the comparison learning must win
    base = minimal_support(lambda k: held_exec.supports(theta0, k), HN, C, args.eps, args.form, 1)
    base_units = float((base["keep"] * units).sum(1).mean())
    # alignment of kept reads with the key Fourier planes
    V = torch.from_numpy(theta[:K * d]).view(K, d)
    E = model.W_E.detach()[:p]
    D = E - E.mean(0)
    a = torch.arange(p, dtype=torch.float64)
    planes = {}
    for k in range(1, (p - 1) // 2 + 1):
        m2 = torch.stack([torch.cos(2 * math.pi * k * a / p), torch.sin(2 * math.pi * k * a / p)], 1)
        planes[k] = (torch.linalg.qr(D.T @ m2)[0], float(((m2.T @ D) ** 2).sum()))
    key = sorted(planes, key=lambda k: -planes[k][1])[:5]
    used = hkeep[:, 2 * H * dh:2 * H * dh + K].mean(0)
    top = np.argsort(-used)[:20]
    align = []
    for c in top:
        v = V[c] / V[c].norm()
        best = max(key, key=lambda k: float((planes[k][0].T @ v).norm()))
        align.append((int(c), float(used[c]), best, float((planes[best][0].T @ v).norm())))
    report = {"args": vars(args), "alternations": fit["alternations"], "fit_units": fitted,
              "heldout_units_learned": held_units, "heldout_units_start": base_units,
              "heldout_kl_mean": float(held["divergence"].mean()), "key_frequencies": key, "top_read_alignment": align}
    print(f"[mamsf] HELDOUT rank units/example: learned {held_units:.1f} vs starting pieces {base_units:.1f} "
          f"(KL mean {report['heldout_kl_mean']:.4f}, eps {args.eps} {args.form})", flush=True)
    print(f"[mamsf] most-used learned reads (piece, usage, best key plane, |projection|): {align[:8]}", flush=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=1)


if __name__ == "__main__":
    main()
