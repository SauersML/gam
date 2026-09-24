"""#2951 P2 on the trained modular-addition transformer: a curved (angle) path versus a straight
(mask) path between the embedding table and its shift.

Analysis under SPEC 8's exception: it reads checkpoints from ``bench/mpd_modadd_2951.py train`` and
writes one JSON receipt and figures. It is not an MPD input and computes nothing MPD relies on.

The table's cycled rows expand exactly in the characters of their cycle position (see
``crates/gam-sae/src/parameter_decomposition/cyclic_action.rs``)::

    e_a = c_0 + sum_k U_k D(w_k a),   D(t) = (cos t, sin t),   w_k = 2 pi k / p.

Two one-parameter families join ``E`` (t = 0) to the shift-by-one table (t = 1):

* the angle path, each plane turned by its own frequency: ``e_a(t) = c_0 + sum_k U_k D(w_k (a + t))``.
  At integer ``t`` it is the row permutation (the native reference). Between integers it is the
  trigonometric interpolation of the table, a point of the structured-edit manifold of #2951.
* the mask path, the straight segment ``e_a(t) = (1 - t) e_a + t e_{a+1}``. Per plane this is
  ``U_k [(1 - t) I + t R_{w_k}] D(w_k a)``, whose radius at ``t = 1/2`` is ``|cos(w_k / 2)|`` (P2).

If the network computes on the circle equivariantly, the angle path moves the output's phase on
each key frequency by ``w_k t`` with an unchanged amplitude, while the mask path shrinks the
amplitude and moves the phase by ``arg((1 - t) + t e^{i w_k})``. Both predictions are scored against
the measured output, per key frequency of the unembedding, with no fitted parameter.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch

from mpd_modadd_2951 import SITES, all_pairs, build_model


def planes_of(table, p):
    """``c_0`` and ``U`` (m x d x 2) of the cycled rows ``table[:p]``, closed form."""
    rows = table[:p].double()
    positions = torch.arange(p, dtype=torch.float64)
    m = (p - 1) // 2
    freqs = torch.arange(1, m + 1, dtype=torch.float64)
    angles = 2 * math.pi * freqs[:, None] * positions[None, :] / p  # m x p
    cos, sin = torch.cos(angles), torch.sin(angles)
    mean = rows.mean(0)
    u_cos = (2.0 / p) * cos @ rows  # m x d
    u_sin = (2.0 / p) * sin @ rows
    return mean, torch.stack([u_cos, u_sin], dim=-1)  # m x d x 2


def angle_table(table, mean, planes, p, t, keep=None):
    """``e_a(t)`` on the cycled rows, the ``=`` row fixed. ``keep`` restricts the turn to those planes."""
    m = planes.shape[0]
    positions = torch.arange(p, dtype=torch.float64)
    freqs = torch.arange(1, m + 1, dtype=torch.float64)
    out = table.double().clone()
    turn = torch.ones(m, dtype=torch.float64) if keep is None else keep.double()
    shift = t * turn  # per-plane shift in units of one position
    ang0 = 2 * math.pi * freqs[:, None] * positions[None, :] / p
    ang1 = 2 * math.pi * freqs[:, None] * (positions[None, :] + shift[:, None]) / p
    delta = (torch.cos(ang1) - torch.cos(ang0))[:, :, None] * planes[:, None, :, 0] + (
        torch.sin(ang1) - torch.sin(ang0)
    )[:, :, None] * planes[:, None, :, 1]
    out[:p] += delta.sum(0)
    return out


def mask_table(table, p, t):
    out = table.double().clone()
    shifted = table.double()[(torch.arange(p) + 1) % p]
    out[:p] = (1 - t) * out[:p] + t * shifted
    return out


def spectrum_of_logits(logits, pairs, p):
    """Per frequency, the complex Fourier coefficient of each row's centred log-probabilities in the
    output variable ``c`` relative to ``a + b``: ``F_k = mean_c logp(c) e^{-i w_k (c - a - b)}``."""
    logp = torch.log_softmax(logits.double(), dim=-1)
    logp = logp - logp.mean(-1, keepdim=True)
    c = torch.arange(p, dtype=torch.float64)
    rel = (c[None, :] - (pairs[:, 0] + pairs[:, 1]).double()[:, None])  # n x p
    m = (p - 1) // 2
    freqs = torch.arange(1, m + 1, dtype=torch.float64)
    ang = 2 * math.pi * freqs[None, :, None] * rel[:, None, :] / p  # n x m x p
    re = (logp[:, None, :] * torch.cos(ang)).mean(-1)
    im = -(logp[:, None, :] * torch.sin(ang)).mean(-1)
    return torch.complex(re, im)  # n x m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--steps", type=int, default=41, help="points on each path, t in [0, 1]")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run = torch.load(args.run, map_location="cpu", weights_only=True)
    config = run["config"]
    p = config["p"]
    step = args.step if args.step is not None else max(run["checkpoints"])
    model = build_model(config)
    model.load_state_dict(run["checkpoints"][step])
    model = model.double().eval().to(args.device)
    pairs = all_pairs(p)[run["test_idx"]].to(args.device)
    table = model.W_E.detach().cpu()
    mean, planes = planes_of(table, p)
    power = planes.pow(2).sum((1, 2))
    order = torch.argsort(power, descending=True)
    ts = torch.linspace(0, 1, args.steps, dtype=torch.float64)
    with torch.inference_mode():
        base_logits = model(pairs)
        base_spec = spectrum_of_logits(base_logits.cpu(), pairs.cpu(), p)
        amp0 = base_spec.abs().mean(0)
        key_out = torch.argsort(amp0, descending=True)[:8]
        report = {"p": p, "step": step, "test_pairs": int(pairs.shape[0]),
                  "embedding_power_order": (order[:12] + 1).tolist(),
                  "embedding_power": power[order[:12]].tolist(),
                  "output_key_frequencies": (key_out + 1).tolist(),
                  "output_key_amplitudes": amp0[key_out].tolist(), "paths": {}}
        for site in ("pos0", "global"):
            for path in ("angle", "mask"):
                rows = []
                for t in ts.tolist():
                    edited = angle_table(table, mean, planes, p, t) if path == "angle" else mask_table(table, p, t)
                    edited = edited.to(args.device)
                    logits = model(pairs, embed_at={u: edited for u in SITES[site]})
                    spec = spectrum_of_logits(logits.cpu(), pairs.cpu(), p)
                    ratio = (spec / base_spec)[:, key_out]  # n x 8, amplitude ratio and phase shift
                    per_site_units = len(SITES[site]) if site != "global" else 2  # a and b each move
                    freqs = (key_out + 1).double()
                    expected_phase = -2 * math.pi * freqs * per_site_units * t / p
                    if path == "angle":
                        expected = torch.polar(torch.ones_like(freqs), expected_phase)
                    else:
                        # each moved operand contributes the factor (1 - t) + t e^{-i w} to the key harmonic
                        one = torch.polar(torch.ones_like(freqs), -2 * math.pi * freqs / p)
                        expected = ((1 - t) + t * one) ** per_site_units
                    probs = torch.softmax(logits.double(), -1)
                    target = ((pairs[:, 0] + pairs[:, 1] + (0 if t < 0.5 else per_site_units)) % p)
                    rows.append({
                        "t": t,
                        "amplitude_ratio_median": ratio.abs().median(0).values.tolist(),
                        "phase_shift_median": torch.angle(ratio).median(0).values.tolist(),
                        "expected_amplitude": expected.abs().tolist(),
                        "expected_phase": torch.angle(expected).tolist(),
                        "max_prob_mean": probs.max(-1).values.mean().item(),
                        "argmax_hits_nearest_integer": (probs.argmax(-1) == target).double().mean().item(),
                        "entropy_mean": (-(probs * probs.clamp_min(1e-300).log()).sum(-1)).mean().item(),
                    })
                report["paths"][f"{site}/{path}"] = rows
                mid = rows[len(rows) // 2]
                print(f"[{site}/{path}] t=0.5 maxprob={mid['max_prob_mean']:.4f} entropy={mid['entropy_mean']:.3f} "
                      f"amp={np.round(mid['amplitude_ratio_median'][:4], 3)} exp_amp={np.round(mid['expected_amplitude'][:4], 3)} "
                      f"phase={np.round(mid['phase_shift_median'][:4], 3)} exp_phase={np.round(mid['expected_phase'][:4], 3)}",
                      flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=1)
    print(f"[paths] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
