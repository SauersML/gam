"""#2951: charge spectroscopy of the trained modular-addition transformer.

Analysis under SPEC 8's exception (not an MPD input). Reads a checkpoint from
``bench/mpd_modadd_2951.py train`` and writes one JSON receipt.

Turning embedding plane ``k`` by ``theta`` (``U_k D(w_k a) -> U_k R_theta D(w_k a)``, the other planes
fixed) is an action of the circle group on the parameters. Every downstream quantity then decomposes
into characters ``e^{-i n theta}``, and the integer ``n`` is the charge of that quantity under plane
``k``: the number of times plane ``k``'s phase enters the monomial that produces it. For each output
harmonic ``j`` (the Fourier coefficient of the centred log-probabilities in ``c - a - b``, as in
``mpd_modadd_paths_2951.py``) the gain

    g_j(theta) = sum_x F_j(x; theta) conj(F_j(x; 0)) / sum_x |F_j(x; 0)|^2

is sampled at ``theta = 2 pi m / M``, ``m = 0..M-1``, and its discrete Fourier transform over ``m``
gives the charge spectrum ``c_j(n)``, exact for ``|n| < M / 2`` up to aliasing of higher charges,
which is reported (the spectrum's mass at the Nyquist bins).
"""
from __future__ import annotations

import argparse
import json
import math

import torch

from mpd_modadd_2951 import all_pairs, build_model
from mpd_modadd_paths_2951 import planes_of, spectrum_of_logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--planes", type=int, required=True, help="the top planes by embedding power to turn")
    parser.add_argument("--angles", type=int, required=True, help="M, the number of turn angles")
    parser.add_argument("--step", type=int, default=None, help="checkpoint (default: the last)")
    parser.add_argument("--device", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    dev = torch.device(args.device)
    run = torch.load(args.run, map_location="cpu", weights_only=True)
    config = run["config"]
    p = config["p"]
    step = args.step if args.step is not None else max(run["checkpoints"])
    model = build_model(config)
    model.load_state_dict(run["checkpoints"][step])
    model = model.double().eval().to(dev)
    pairs = all_pairs(p)[run["test_idx"]].to(dev)
    table = model.W_E.detach().clone()
    _, planes = planes_of(table.cpu(), p)
    planes = planes.to(dev)
    power = planes.pow(2).sum((1, 2))
    order = torch.argsort(power, descending=True)[: args.planes]
    positions = torch.arange(p, dtype=torch.float64, device=dev)
    M = args.angles
    thetas = [2 * math.pi * m / M for m in range(M)]
    report = {"p": p, "step": step, "angles": M, "turned": {}}
    with torch.inference_mode():
        base = spectrum_of_logits(model(pairs).cpu(), pairs.cpu(), p)  # n x 56
        base_power = base.abs().pow(2).sum(0)
        report["output_power"] = base_power.tolist()
        for k in order.tolist():
            ang = 2 * math.pi * (k + 1) * positions / p
            u = planes[k]
            c, s = torch.cos(ang)[:, None], torch.sin(ang)[:, None]
            content = c * u[None, :, 0] + s * u[None, :, 1]
            for sites_name, sites in (("a", (0,)), ("b", (1,)), ("ab", (0, 1))):
                gains = []
                for theta in thetas:
                    ct, st = math.cos(theta), math.sin(theta)
                    turned = (c * ct - s * st) * u[None, :, 0] + (s * ct + c * st) * u[None, :, 1]
                    edited = table.clone()
                    edited[:p] += turned - content
                    spec = spectrum_of_logits(model(pairs, embed_at={x: edited for x in sites}).cpu(), pairs.cpu(), p)
                    gains.append((spec * base.conj()).sum(0) / base_power)
                g = torch.stack(gains)  # M x 56
                # c(n) = (1/M) sum_m g(theta_m) e^{+i n theta_m}: g = sum_n c(n) e^{-i n theta}
                charges = list(range(-(M // 2) + 1, M // 2))
                phase = torch.tensor([[n * t for t in thetas] for n in charges], dtype=torch.float64)
                basis = torch.complex(torch.cos(phase), torch.sin(phase))  # charges x M
                spectrum = basis @ g / M  # charges x 56
                nyquist = (torch.tensor([(-1.0) ** m for m in range(M)], dtype=torch.complex128) @ g / M).abs()
                entry = {}
                for j in range(spectrum.shape[1]):
                    mags = spectrum[:, j].abs()
                    if base_power[j] < 1e-3 * base_power.max():
                        continue
                    dom = [(charges[i], round(mags[i].item(), 4)) for i in torch.argsort(mags, descending=True)[:3].tolist() if mags[i] > 0.02]
                    entry[str(j + 1)] = {"charges": dom, "nyquist": round(nyquist[j].item(), 4)}
                report["turned"][f"{k + 1}/{sites_name}"] = entry
                nontrivial = {j: v["charges"] for j, v in entry.items() if v["charges"] and v["charges"][0][0] != 0}
                print(f"[charges] plane {k + 1} turned at {sites_name}: output harmonics with nonzero charge {nontrivial}", flush=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=1)
    print(f"[charges] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
