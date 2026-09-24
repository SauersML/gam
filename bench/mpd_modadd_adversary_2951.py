"""#2951 P8-P10 on the trained modular-addition transformer: the exact worst ablation of the
unembedding's non-key Fourier planes against the adversaries parameter decomposition uses.

Analysis under SPEC 8's exception (benchmark evaluation; not an MPD input). It reads checkpoints from
``bench/mpd_modadd_2951.py train`` and writes one JSON receipt.

# The decomposition

``W_U`` (p x d) expands exactly in the characters of the output token ``c``:
``W_U[c] = c_0 + sum_k U_k D(w_k c)``. Plane ``k`` is a rank-two parameter component, and masking it,
``W_U(m) = c_0 + sum_k m_k U_k D(w_k c)``, moves the logits of an input with final residual ``h`` by
``-sum_k t_k w_k``, ``t_k = 1 - m_k``, ``w_k[c] = D(w_k c) . (U_k^T h)``: a pure frequency-``k`` wave.
The model has no final norm, so the logits are affine in the masks and
``KL(p_0 || softmax(z_0 - sum_k t_k w_k))`` is convex in ``t`` (P9). Its supremum over a box of masks
is therefore attained at a vertex, and enumerating the vertices is EXACT, not a search.

The key planes (declared by their unembedding power, the grokking algorithm's known frequencies) are
kept at 1: the "causally important" set. Every other plane is free in ``[0, 1]``: the claim a
parameter decomposition makes about unimportant components is that they can be ablated in any
combination. We compute, for the ``F`` largest non-key planes:

* ``exact_per_input``: max over the 2^F vertices, per test pair, then its distribution;
* ``exact_shared``: max over vertices of the batch mean (one mask for every input, the adversary
  VPD evaluates with);
* ``pgd_shared[n]``: VPD's evaluation adversary, projected gradient ascent on shared sources in
  ``[0, 1]^F`` with step 0.1 (sign steps), ``n`` in {20, 80, 320}, from uniform random starts;
* ``fw_shared``: Frank-Wolfe with the box's vertex oracle (the LMO of #2951 ``adversary.rs``);
* ``uniform_mean``: the mean KL under independent uniform masks, a mask-LAW MOMENT, never a bound.

These are four different numbers (P10); only the exact ones are suprema.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time

import torch

from mpd_modadd_2951 import all_pairs, build_model


def final_residual(model, tokens):
    """The residual at ``=`` just before ``W_U`` (the model's own forward, stopped one read early)."""
    x = model.W_E[tokens] + model.W_pos
    q = torch.einsum("hkd,npd->nhpk", model.W_Q, x)
    k = torch.einsum("hkd,npd->nhpk", model.W_K, x)
    v = torch.einsum("hkd,npd->nhpk", model.W_V, x)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(model.d_head)
    pattern = torch.softmax(scores.masked_fill(~model.causal, float("-inf")), dim=-1)
    x = x + (pattern @ v).transpose(1, 2).reshape(x.shape[0], 3, -1) @ model.W_O.T
    x = x + torch.relu(x @ model.W_in.T + model.b_in) @ model.W_out.T + model.b_out
    return x[:, -1]


def unembed_planes(W_U, p):
    rows = W_U.double()
    c = torch.arange(p, dtype=torch.float64, device=W_U.device)
    m = (p - 1) // 2
    freqs = torch.arange(1, m + 1, dtype=torch.float64, device=W_U.device)
    ang = 2 * math.pi * freqs[:, None] * c[None, :] / p
    cos, sin = torch.cos(ang), torch.sin(ang)
    mean = rows.mean(0)
    return mean, torch.stack([(2.0 / p) * cos @ rows, (2.0 / p) * sin @ rows], -1), cos, sin


def kl_rows(log_p0, logits):
    """``KL(p_0 || softmax(logits))`` along the last axis, float64."""
    log_q = torch.log_softmax(logits, dim=-1)
    return (log_p0.exp() * (log_p0 - log_q)).sum(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--keys", type=int, required=True, help="number of key planes kept at 1 (declared)")
    parser.add_argument("--free", type=int_list, required=True, help="ladder of free-plane counts F")
    parser.add_argument("--restarts", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    run = torch.load(args.run, map_location="cpu", weights_only=True)
    config = run["config"]
    p = config["p"]
    step = args.step if args.step is not None else max(run["checkpoints"])
    model = build_model(config)
    model.load_state_dict(run["checkpoints"][step])
    model = model.double().eval().to(dev)
    pairs = all_pairs(p)[run["test_idx"]].to(dev)
    with torch.inference_mode():
        h = final_residual(model, pairs)  # n x d
        z0 = h @ model.W_U.T
        assert torch.equal(z0, model(pairs)), "the stopped forward must equal the model's own"
        log_p0 = torch.log_softmax(z0, -1)
        mean, planes, cos, sin = unembed_planes(model.W_U.detach(), p)
        power = planes.pow(2).sum((1, 2))
        order = torch.argsort(power, descending=True)
        keys, rest = order[: args.keys], order[args.keys:]
        coeff = torch.einsum("kdj,nd->nkj", planes, h)  # n x m x 2 : U_k^T h
        waves = coeff[..., 0, None] * cos[None] + coeff[..., 1, None] * sin[None]  # n x m x p
        recon = (h @ mean)[:, None] + waves.sum(1)
        exactness = (recon - z0).abs().max().item()
        key_single = torch.stack([kl_rows(log_p0, z0 - waves[:, k]) for k in keys.tolist()], 1)
        # A free plane that is individually important would make the "unimportant" set a misdeclaration,
        # so each of the largest free planes' single ablation is reported beside the keys'.
        free_single = torch.stack([kl_rows(log_p0, z0 - waves[:, k]) for k in rest[:24].tolist()], 1)
        report = {
            "p": p, "step": step, "test_pairs": int(pairs.shape[0]), "seed": args.seed,
            "plane_expansion_max_abs_error": exactness,
            "power_order": (order + 1).tolist(), "power": power[order].tolist(),
            "keys": (keys + 1).tolist(),
            "key_single_ablation_kl_median": key_single.median(0).values.tolist(),
            "free_single_ablation_kl_median": free_single.median(0).values.tolist(),
            "free_single_ablation_kl_max": free_single.max(0).values.tolist(),
            "ladder": [],
        }
        print(f"[adv] p={p} step={step} n={pairs.shape[0]} expansion_err={exactness:.2e} keys={report['keys']} "
              f"key-ablation KL medians={[round(v, 3) for v in report['key_single_ablation_kl_median']]}", flush=True)
    for F in args.free:
        free = rest[:F]
        W = waves[:, free]  # n x F x p
        rung = {"F": F, "free_planes": (free + 1).tolist()}
        t0 = time.time()
        with torch.inference_mode():
            verts = torch.tensor(list(itertools.product((0.0, 1.0), repeat=F)), dtype=torch.float64, device=dev)
            best_in = torch.full((W.shape[0],), -1.0, dtype=torch.float64, device=dev)
            shared = torch.zeros(verts.shape[0], dtype=torch.float64, device=dev)
            chunk = max(1, (1 << 22) // (verts.shape[0] * p))
            for s in range(0, W.shape[0], chunk):
                Wc, lc, zc = W[s:s + chunk], log_p0[s:s + chunk], z0[s:s + chunk]
                logits = zc[:, None, :] - torch.einsum("vf,nfp->nvp", verts, Wc)  # n x V x p
                kl = kl_rows(lc[:, None, :], logits)  # n x V
                best_in[s:s + chunk] = kl.max(1).values
                shared += kl.sum(0)
            shared /= W.shape[0]
            vstar = int(shared.argmax())
            rung["exact_per_input"] = quantiles(best_in)
            rung["exact_shared"] = shared[vstar].item()
            rung["exact_shared_vertex_ablates"] = [int(free[j]) + 1 for j in range(F) if verts[vstar, j] == 1.0]
            u = torch.rand(4096, F, dtype=torch.float64, device=dev)
            um = torch.zeros((), dtype=torch.float64, device=dev)
            for s in range(0, W.shape[0], 256):
                um += kl_rows(log_p0[s:s + 256, None], z0[s:s + 256, None] - torch.einsum("uf,nfp->nup", u, W[s:s + 256])).sum()
            rung["uniform_mean"] = (um / (W.shape[0] * u.shape[0])).item()
        rung["exact_seconds"] = time.time() - t0

        def shared_value(t):
            return kl_rows(log_p0, z0 - torch.einsum("f,nfp->np", t, W)).mean()

        pgd = {n: [] for n in (20, 80, 320)}
        fw = []
        for _ in range(args.restarts):
            t = torch.rand(F, dtype=torch.float64, device=dev)  # sources; mask m = 1 - t, CI = 0
            for it in range(1, 321):
                t.requires_grad_(True)
                val = shared_value(t)
                (g,) = torch.autograd.grad(val, t)
                with torch.no_grad():
                    t = (t + 0.1 * g.sign()).clamp(0.0, 1.0)
                if it in pgd:
                    with torch.no_grad():
                        pgd[it].append(shared_value(t).item())
            # Frank-Wolfe over the box: LMO = the vertex t = [g > 0]; exact line search on a 65-point
            # ladder of the segment (the objective is convex, so the segment max is at an end, which the
            # ladder contains: the step is 1 whenever the vertex improves).
            t = torch.rand(F, dtype=torch.float64, device=dev)
            for _ in range(64):
                t.requires_grad_(True)
                val = shared_value(t)
                (g,) = torch.autograd.grad(val, t)
                with torch.no_grad():
                    vtx = (g > 0).double()
                    if torch.equal(vtx, t):
                        break
                    cand = [shared_value(t + gamma * (vtx - t)).item() for gamma in torch.linspace(0, 1, 65).tolist()]
                    gamma = torch.linspace(0, 1, 65)[int(torch.tensor(cand).argmax())].item()
                    t = t + gamma * (vtx - t)
                    if gamma == 0.0:
                        break
            with torch.no_grad():
                fw.append(shared_value(t).item())
        rung["pgd_shared"] = {str(n): quantiles(torch.tensor(v)) for n, v in pgd.items()}
        rung["fw_shared"] = quantiles(torch.tensor(fw))
        rung["pgd_fraction_of_exact"] = {str(n): (torch.tensor(v) / rung["exact_shared"]).median().item() for n, v in pgd.items()}
        report["ladder"].append(rung)
        print(f"[adv F={F}] exact_shared={rung['exact_shared']:.4g} exact_per_input max={rung['exact_per_input']['max']:.4g} "
              f"median={rung['exact_per_input']['q50']:.4g} uniform_mean={rung['uniform_mean']:.4g} "
              f"pgd20/80/320 median={[round(rung['pgd_shared'][str(n)]['q50'], 5) for n in (20, 80, 320)]} "
              f"fw median={rung['fw_shared']['q50']:.4g} ({rung['exact_seconds']:.1f}s)", flush=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=1)
    print(f"[adv] wrote {args.out}", flush=True)


def quantiles(x):
    x = x.double().flatten().cpu()
    return {"min": x.min().item(), "q50": x.quantile(0.5).item(), "q90": x.quantile(0.9).item(),
            "q99": x.quantile(0.99).item(), "max": x.max().item(), "mean": x.mean().item()}


def int_list(text):
    return [int(v) for v in text.split(",") if v]


if __name__ == "__main__":
    main()
