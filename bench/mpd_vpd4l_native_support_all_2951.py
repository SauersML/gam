"""#2951 Stage 1 on all 24 matrices: the minimal supports of the starting pieces, dual against tight frame.

No piece learning: ``gamfit.sae.minimal_support`` (mean fidelity, joint masking, VPD's semantics) from the full support,
at the starting pieces of ``mpd_vpd4l_support_fit_all_2951.py`` under ``--pieces dual`` (analysis ``V``, synthesis
``V^+``: normalized neuron reads for c_fc) or ``--pieces frame`` (tight frames, ``S = V^T``: the polar factor of those
reads). Every other matrix starts orthogonal (its SVD or the identity), so the two differ only in c_fc. Reports the rank
units kept per position, and VPD's PGD-20 KL on those supports (every removed piece re-added with an adversarial mask
in [0, 1] shared by all positions, 20 sign steps of 0.1 from a uniform start).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pyarrow.parquet as pq
import torch

from mpd_vpd4l_minimal_support_2951 import VPD, Pieces
from mpd_vpd4l_support_fit_all_2951 import AllPieces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--seqs", type=int, required=True)
    parser.add_argument("--ctx", type=int, required=True)
    parser.add_argument("--eps", type=float, required=True)
    parser.add_argument("--pieces", choices=("frame", "dual"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from gamfit.sae import minimal_support
    dev = args.device
    torch.manual_seed(0)
    model = AllPieces(Pieces(args.weights, dev), args.pieces == "frame")
    table = pq.read_table(args.test)
    ids = torch.tensor(table.slice(0, args.seqs).column("input_ids").to_pylist(), dtype=torch.long)[:, :args.ctx].to(dev)
    C = model.units
    Bn, T = ids.shape
    P = Bn * T
    theta = torch.from_numpy(model.initial_theta().numpy()).to(dev)
    with torch.no_grad():
        logp_clean = model.b(ids).log_softmax(-1)

    def kl_of(lg):
        return (logp_clean.exp() * (logp_clean - lg.log_softmax(-1))).sum(-1).reshape(-1)

    def mask(keep):
        return torch.from_numpy(np.asarray(keep)).to(dev).view(Bn, T, C).float()

    t0 = time.time()

    def evaluate(keep):
        m = mask(keep).requires_grad_(True)
        lg = model.logits(ids, m, theta)
        kl = kl_of(lg)
        (g,) = torch.autograd.grad(kl.sum(), m, retain_graph=True)
        with torch.no_grad():
            y = torch.distributions.Categorical(logits=lg).sample()
        (hh,) = torch.autograd.grad(lg.log_softmax(-1).gather(-1, y[..., None]).sum(), m)
        klv = kl.detach().double().cpu().numpy()
        print(f"[native] rank units/position {np.asarray(keep).reshape(P, C).sum(1).mean():.1f}; KL mean {klv.mean():.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        g, hh = g.reshape(P, C).double(), hh.reshape(P, C).double()
        return klv, (-g + 0.5 * hh ** 2).cpu().numpy(), (-g - 0.5 * hh ** 2).cpu().numpy()

    result = minimal_support(evaluate, P, C, args.eps, "mean", T)
    keep_t = mask(result["keep"])
    adv = torch.rand(C, device=dev, generator=torch.Generator(device=dev).manual_seed(0)).requires_grad_(True)
    for _ in range(20):
        (grad,) = torch.autograd.grad(kl_of(model.logits(ids, keep_t + (1 - keep_t) * adv, theta)).mean(), adv)
        with torch.no_grad():
            adv.add_(0.1 * grad.sign()).clamp_(0, 1)
    with torch.no_grad():
        pgd = kl_of(model.logits(ids, keep_t + (1 - keep_t) * adv, theta)).mean().item()
    units = float(result["keep"].sum(1).mean())
    report = {"args": vars(args), "rank_units_mean": units, "kl_mean": float(result["divergence"].mean()),
              "pgd20_kl": pgd, "pieces": C, "vpd": VPD}
    print(f"[native] {args.pieces}: {units:.1f} of {C} rank units/position at KL {report['kl_mean']:.4f}; "
          f"PGD-20 KL {pgd:.4f} (VPD {VPD['l0']} at {VPD['kl_rounded']}, PGD-20 {VPD['pgd20']})", flush=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=1)


if __name__ == "__main__":
    main()
