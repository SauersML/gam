"""#2951: the held-out sparsity-fidelity path of a Stage 2 fit, one point per fidelity level.

``mpd_vpd4l_support_fit_all_2951.py`` saves the pieces each fidelity level ends with (``<out>_level<eps>.npy``). For
each saved level this runs ``gamfit.sae.minimal_support`` on held-out sequences at that level's fidelity (mean form,
joint masking) with the pieces frozen, and VPD's PGD-20 on the supports found (every removed piece re-added with an
adversarial mask in [0, 1] shared by all positions, 20 sign steps of 0.1 from a uniform start). Held-out sequences
are the ones after the fit's own in the test split. Levels already in ``--out`` are skipped, so the file grows as the
fit descends.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
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
    parser.add_argument("--fit", required=True, help="the fit's --out path; its _level<eps>.npy files are read")
    parser.add_argument("--fit-seqs", type=int, required=True)
    parser.add_argument("--heldout-seqs", type=int, required=True)
    parser.add_argument("--ctx", type=int, required=True)
    parser.add_argument("--pieces", choices=("frame", "dual"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from gamfit.sae import minimal_support
    dev = args.device
    model = AllPieces(Pieces(args.weights, dev), args.pieces == "frame")
    table = pq.read_table(args.test)
    ids = torch.tensor(table.slice(args.fit_seqs, args.heldout_seqs).column("input_ids").to_pylist(),
                       dtype=torch.long)[:, :args.ctx].to(dev)
    C = model.units
    Bn, T = ids.shape
    P = Bn * T
    with torch.no_grad():
        logp_clean = model.b(ids).log_softmax(-1)

    def kl_of(lg):
        return (logp_clean.exp() * (logp_clean - lg.log_softmax(-1))).sum(-1).reshape(-1)

    def mask(keep):
        return torch.from_numpy(np.asarray(keep)).to(dev).view(Bn, T, C).float()

    report = json.load(open(args.out)) if os.path.exists(args.out) else {"args": vars(args), "levels": {}, "vpd": VPD}
    pattern = re.compile(re.escape(args.fit.replace(".json", "_level")) + r"(.+)\.npy$")
    for path in sorted(glob.glob(args.fit.replace(".json", "_level*.npy"))):
        level = pattern.match(path).group(1)
        if level in report["levels"]:
            continue
        eps = float(level)
        theta = torch.from_numpy(np.load(path).astype(np.float64)).to(dev)
        t0 = time.time()

        def evaluate(keep):
            m = mask(keep).requires_grad_(True)
            lg = model.logits(ids, m, theta)
            kl = kl_of(lg)
            (g,) = torch.autograd.grad(kl.sum(), m, retain_graph=True)
            with torch.no_grad():
                y = torch.distributions.Categorical(logits=lg).sample()
            (hh,) = torch.autograd.grad(lg.log_softmax(-1).gather(-1, y[..., None]).sum(), m)
            g, hh = g.reshape(P, C).double(), hh.reshape(P, C).double()
            return kl.detach().double().cpu().numpy(), (-g + 0.5 * hh ** 2).cpu().numpy(), (-g - 0.5 * hh ** 2).cpu().numpy()

        held = minimal_support(evaluate, P, C, eps, "mean", T)
        keep_t = mask(held["keep"])
        adv = torch.rand(C, device=dev, generator=torch.Generator(device=dev).manual_seed(0)).requires_grad_(True)
        for _ in range(20):
            (grad,) = torch.autograd.grad(kl_of(model.logits(ids, keep_t + (1 - keep_t) * adv, theta)).mean(), adv)
            with torch.no_grad():
                adv.add_(0.1 * grad.sign()).clamp_(0, 1)
        with torch.no_grad():
            pgd = kl_of(model.logits(ids, keep_t + (1 - keep_t) * adv, theta)).mean().item()
        report["levels"][level] = {"rank_units_mean": float(held["keep"].sum(1).mean()),
                                   "kl_mean": float(held["divergence"].mean()), "pgd20_kl": pgd,
                                   "seconds": time.time() - t0}
        print(f"[heldout] level {level}: {report['levels'][level]['rank_units_mean']:.1f} rank units/position at KL "
              f"{report['levels'][level]['kl_mean']:.4f}; PGD-20 KL {pgd:.4f}", flush=True)
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=1)


if __name__ == "__main__":
    main()
