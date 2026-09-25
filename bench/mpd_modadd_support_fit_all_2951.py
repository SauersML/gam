"""#2951: do learned pieces rediscover the grokked modular-addition mechanism, unsupervised?

Thin torch executor for ``gamfit.sae.fit_supports`` (SPEC 8: the fit's owner is ``support_fit``; this file runs the
model and its derivatives, and scores the result against the known mechanism).

Every weight map the model applies is split on its input side into rank-one pieces that sum to it exactly for every
parameter value: the stacked query, key and value maps and the output map (K = 128, starting at each map's own
singular decomposition), the MLP input (K = 512, starting at the tight frame nearest the neurons' read directions) and
the MLP output (K = 512, the neuron basis its input lives in). One example is one position.

``--pieces frame`` (the default) keeps every map's pieces a tight frame: an analysis ``V`` with orthonormal columns,
moved on the Stiefel manifold (``fit_supports``'s ``frames``), and synthesis ``S = V^T``. Then any partial removal ``0 <= D <= I`` of
pieces takes out ``W V^T D V``, never more than ``W`` itself: pieces cannot cancel one another. ``--pieces dual``
is the general exact split (``S = V^+ + Y (I - V V^+)``), where they can: on this model the fit drove the MLP-input
analysis toward singularity (condition number 2.5e3 at the start, 2.3e5 within three minutes), and from the full
support no single piece could be removed at a mean budget of 1.28 nats.

Scored against the known mechanism, which the fit never sees: the key Fourier planes of the embedding (the frequencies
carrying most of its power). For the residual-stream reads (query, key, value and MLP-input pieces), the fraction of
each read direction's norm inside the span of the key planes, over the pieces the held-out examples use most, against
the same fraction for the starting pieces and for uniformly random directions.
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
    parser.add_argument("--pieces", choices=("frame", "dual"), default="frame")
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
    Wq, Wk, Wv = (t.reshape(H * dh, d) for t in (model.W_Q, model.W_K, model.W_V))
    maps = {"q": Wq, "k": Wk, "v": Wv, "o": model.W_O, "in": model.W_in, "out": model.W_out}
    K = {"q": d, "k": d, "v": d, "o": H * dh, "in": n_mlp, "out": n_mlp}
    names = list(maps)
    offsets, o = {}, 0
    for n in names:
        offsets[n] = (o, o + K[n])
        o += K[n]
    C = o
    frame = args.pieces == "frame"
    shapes = [(n, "V", (K[n], maps[n].shape[1])) for n in names]
    if not frame:
        shapes += [(n, "Y", (maps[n].shape[1], K[n])) for n in names]
    sizes = [a * b for *_, (a, b) in shapes]

    def initial_theta():
        parts = []
        for n, kind, (a, b) in shapes:
            if kind == "Y":
                parts.append(torch.zeros(a, b))
            elif n == "in":
                reads = model.W_in / model.W_in.norm(dim=1, keepdim=True)
                if frame:
                    # the tight frame nearest the reads: their polar factor
                    u, _, vt = torch.linalg.svd(reads, full_matrices=False)
                    reads = u @ vt
                parts.append(reads)
            elif n == "out":
                parts.append(torch.eye(a, b))
            else:
                parts.append(torch.linalg.svd(maps[n], full_matrices=True)[2])
        return torch.cat([t.reshape(-1) for t in parts])

    def unpack(theta):
        out, i = {}, 0
        for (n, kind, (a, b)), sz in zip(shapes, sizes):
            out[(n, kind)] = theta[i:i + sz].view(a, b)
            i += sz
        return out

    def logits(tokens, m, theta):
        P = unpack(theta)

        def apply(n, x):
            if frame:
                V = P[(n, "V")]
                S = V.T
            else:
                V, Y = P[(n, "V")], P[(n, "Y")]
                Vp = torch.linalg.solve(V.T @ V, V.T)
                S = Vp + Y - (Y @ V) @ Vp
            s0, s1 = offsets[n]
            mask = m[:, s0:s1] if x.dim() == 2 else m[:, None, s0:s1]
            return ((x @ V.T) * mask) @ (maps[n] @ S).T

        x = torch.stack([model.W_E[tokens[:, u]] for u in range(3)], 1) + model.W_pos  # N x 3 x d
        q = apply("q", x[:, -1]).view(-1, H, dh)
        k = apply("k", x).view(x.shape[0], 3, H, dh)
        v = apply("v", x).view(x.shape[0], 3, H, dh)
        att = torch.softmax(torch.einsum("nhe,nphe->nhp", q, k) / math.sqrt(dh), -1)
        dest = x[:, -1] + apply("o", torch.einsum("nhp,nphe->nhe", att, v).reshape(-1, H * dh))
        a = torch.relu(apply("in", dest) + model.b_in)
        dest = dest + apply("out", a) + model.b_out
        return dest @ model.W_U.T

    order = torch.randperm(p * p, generator=torch.Generator().manual_seed(0))
    fit_tokens = all_pairs(p)[order[:args.fit_examples]]
    held_tokens = all_pairs(p)[order[args.fit_examples:args.fit_examples + args.heldout_examples]]
    t0 = time.time()

    def make(tokens):
        with torch.no_grad():
            clean = model(tokens).log_softmax(-1)
        N = len(tokens)

        def kl_of(lg):
            return (clean.exp() * (clean - lg.log_softmax(-1))).sum(-1)

        def mask(keep):
            return torch.from_numpy(np.asarray(keep)).double()

        def th(a):
            return torch.from_numpy(np.asarray(a))

        class Executor:
            def supports(self, theta, keep):
                m = mask(keep).requires_grad_(True)
                lg = logits(tokens, m, th(theta))
                kl = kl_of(lg)
                (g,) = torch.autograd.grad(kl.sum(), m, retain_graph=True)
                y = torch.distributions.Categorical(logits=lg.detach()).sample()
                (hh,) = torch.autograd.grad(lg.log_softmax(-1).gather(-1, y[:, None]).sum(), m)
                return kl.detach().numpy(), (-g + 0.5 * hh ** 2).numpy(), (-g - 0.5 * hh ** 2).numpy()

            def divergence(self, theta, keep):
                with torch.no_grad():
                    return kl_of(logits(tokens, mask(keep), th(theta))).numpy()

            def weighted_gradient(self, theta, keep, weights):
                t = th(theta).clone().requires_grad_(True)
                (g,) = torch.autograd.grad((kl_of(logits(tokens, mask(keep), t)) * th(weights)).sum(), t)
                return g.numpy()

            def directional(self, theta, keep, v):
                m = mask(keep)
                _, dd = jvp(lambda t: kl_of(logits(tokens, m, t)), (th(theta),), (th(v),))
                return dd.numpy()

            def gradient_arithmetic(self):
                return 2.0 ** -53, N * p

            def observe(self, a):
                print(f"[mall] level {a['level']:.4g}: kept {a['kept']} pieces, KL mean {a['mean_divergence']:.5f}; "
                      f"step {a['step']:.3e} radius {a['radius']:.3e}; residual {a['residual']:.3e} / "
                      f"{a['tolerance']:.3e} certified {a['certified']} searched {a['searched']} "
                      f"({time.time() - t0:.0f}s)", flush=True)

            def weighted_hessian(self, theta, keep, weights, v):
                m = mask(keep)
                w = th(weights)

                def grad_of(t):
                    return torch.func.grad(lambda s: (kl_of(logits(tokens, m, s)) * w).sum())(t)

                _, hv = jvp(grad_of, (th(theta),), (th(v),))
                return hv.numpy()

        return Executor(), N

    theta0 = initial_theta().numpy()
    executor, N = make(fit_tokens)
    print(f"[mall] {C} rank-one pieces/example over q, k, v, o, in, out; theta {theta0.size}; fit {N} examples", flush=True)
    fit = fit_supports(executor, theta0, N, C, args.eps, "mean", 1,
                       [shape for (_n, kind, shape) in shapes if kind == "V"] if frame else None)
    held_exec, HN = make(held_tokens)
    held = minimal_support(lambda k: held_exec.supports(fit["theta"], k), HN, C, args.eps, "mean", 1)
    base = minimal_support(lambda k: held_exec.supports(theta0, k), HN, C, args.eps, "mean", 1)
    # the known mechanism: the embedding's key Fourier planes (never seen by the fit)
    E = model.W_E.detach()[:p]
    D = E - E.mean(0)
    a = torch.arange(p, dtype=torch.float64)
    power = {}
    for f in range(1, (p - 1) // 2 + 1):
        m2 = torch.stack([torch.cos(2 * math.pi * f * a / p), torch.sin(2 * math.pi * f * a / p)], 1)
        power[f] = float(((m2.T @ D) ** 2).sum())
    key = sorted(power, key=lambda f: -power[f])[:5]
    basis = torch.linalg.qr(torch.cat([D.T @ torch.stack([torch.cos(2 * math.pi * f * a / p),
                                                          torch.sin(2 * math.pi * f * a / p)], 1) for f in key], 1))[0]

    def fourier_fraction(theta, keep):
        V = unpack(th_(theta))
        used = np.asarray(keep).mean(0)
        rows = []
        for n in ("q", "k", "v", "in"):
            s0, s1 = offsets[n]
            Vn = V[(n, "V")]
            for c in np.argsort(-used[s0:s1])[:20]:
                r = Vn[c] / Vn[c].norm()
                rows.append(float((basis.T @ r).norm() ** 2))
        return float(np.mean(rows))

    def th_(a):
        return torch.from_numpy(np.asarray(a))

    rng = torch.Generator().manual_seed(1)
    random_fraction = float(np.mean([float((basis.T @ (r / r.norm())).norm() ** 2) for r in torch.randn(400, d, generator=rng)]))
    report = {"args": vars(args), "pieces": C, "key_frequencies": key, "alternations": fit["alternations"],
              "heldout_units_learned": float(held["keep"].sum(1).mean()),
              "heldout_units_start": float(base["keep"].sum(1).mean()),
              "heldout_kl_mean": float(held["divergence"].mean()),
              "fourier_fraction_learned": fourier_fraction(fit["theta"], held["keep"]),
              "fourier_fraction_start": fourier_fraction(theta0, base["keep"]),
              "fourier_fraction_random": random_fraction}
    print(f"[mall] HELDOUT pieces/example: learned {report['heldout_units_learned']:.1f} vs starting "
          f"{report['heldout_units_start']:.1f} of {C} (KL mean {report['heldout_kl_mean']:.5f}, eps {args.eps})", flush=True)
    print(f"[mall] Fourier fraction of the most-used residual reads: learned {report['fourier_fraction_learned']:.3f}, "
          f"starting {report['fourier_fraction_start']:.3f}, random {random_fraction:.3f} (key planes {key})", flush=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=1)


if __name__ == "__main__":
    main()
