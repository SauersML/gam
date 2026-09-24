"""#2951 manifold-native addition, phase C: move a problem variable ALONG ITS FITTED MANIFOLD and read the answer.

Analysis under SPEC 8's exception (torch execution; the manifolds are gamfit's phase-B fits).

For a (layer, position, variable) cell with fitted curve ``c(z)`` in the span ``P`` of the position's top principal
directions, a prompt whose variable is ``z`` has its residual moved by ``P^T [c(z + delta) - c(z)]`` at that layer and
position; everything off the curve is kept. Three readouts:

* integer ``delta = s``: the patch along the fitted curve to the value ``z + s``. If the layer carries the variable
  there, the answer moves as the arithmetic says (units digit by ``s`` for a units variable, tens digit by ``s``
  for a tens variable, carries included);
* charge spectroscopy from the manifold's own circle action: a periodic fitted curve is acted on by translation
  ``z -> z + delta``; the gain of each output harmonic over ``delta`` in one period, DFT'd, is its charge;
* curve against chord at ``delta = 1/2``: the fitted point ``c(z + 1/2)`` against ``(c(z) + c(z + 1))/2``.

The answer is the model's distribution over the ten digit tokens at the position that predicts it (``=`` for the tens
digit, ``E`` for the units digit).
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch

UNITS = {"A_units", "B_units", "sum_units"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--harvest", required=True)
    parser.add_argument("--manifolds", required=True)
    parser.add_argument("--layers", required=True, help="comma list or 'all'")
    parser.add_argument("--cells", required=True, help="comma list of position|variable")
    parser.add_argument("--angles", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    meta = json.load(open(os.path.join(args.harvest, "meta.json")))
    V = dict(np.load(os.path.join(args.harvest, "variables.npz")))
    curves = np.load(os.path.join(args.manifolds, "curves.npz"))
    tok = AutoTokenizer.from_pretrained(meta["model"])
    model = AutoModelForCausalLM.from_pretrained(meta["model"], dtype=torch.float32).to(args.device).eval()
    digit_ids = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(10)]
    texts = [meta["few_shot"] + f"{a}+{b}={a + b}" for a, b in zip(V["A"], V["B"])]
    ids = torch.tensor(tok(texts).input_ids, device=args.device)
    L = ids.shape[1]
    pos = {"A_tens": L - 8, "A_units": L - 7, "B_tens": L - 5, "B_units": L - 4, "equals": L - 3, "E": L - 2}
    n = ids.shape[0]
    blocks = model.model.layers
    layers = list(range(len(blocks))) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    cells = [tuple(c.split("|")) for c in args.cells.split(",")]
    state = {"add": None, "at": None}

    def hook(_m, _i, output):
        if state["add"] is None:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden.clone()
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        hidden[rows, state["at"]] += state["add"][state["slice"]].to(hidden.dtype)
        return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden

    def answers(add=None, at=None):
        out = []
        for s in range(0, n, 64):
            state.update(add=add, at=at, slice=slice(s, s + 64))
            with torch.inference_mode():
                logits = model(input_ids=ids[s:s + 64]).logits
            out.append(torch.log_softmax(logits[:, [pos["equals"], pos["E"]]][..., digit_ids].double(), -1))
        state["add"] = None
        return torch.cat(out)  # n x 2 x 10 (tens at '=', units at E)

    base = answers()
    s_true = torch.tensor(V["sum"], device=args.device)
    report = {"meta": meta, "base_accuracy_tens": float((base[:, 0].argmax(-1) == s_true // 10).double().mean()),
              "base_accuracy_units": float((base[:, 1].argmax(-1) == s_true % 10).double().mean()), "cells": []}
    c = torch.arange(10, dtype=torch.float64, device=args.device)

    def harmonic1(lp, target):
        centred = lp - lp.mean(-1, keepdim=True)
        ang = 2 * math.pi * (c[None, :] - target.double()[:, None]) / 10
        return torch.complex((centred * torch.cos(ang)).mean(-1), -(centred * torch.sin(ang)).mean(-1))

    M = args.angles
    for layer in layers:
        handle = blocks[layer].register_forward_hook(hook)
        try:
            for position, variable in cells:
                key = f"{layer}|{position}|{variable}"
                comps = torch.tensor(curves[key + "|comps"], dtype=torch.float64, device=args.device)  # K x width
                grid = curves[key + "|grid"]
                curve = curves[key + "|curve"].astype(np.float64)  # G x K
                z = V[variable].astype(np.float64)
                cyclic = variable in UNITS
                period = 10.0

                def c_at(values):
                    if cyclic:
                        values = np.mod(values, period)
                        return np.stack([np.interp(values, grid, curve[:, j], period=period) for j in range(curve.shape[1])], 1)
                    return np.stack([np.interp(values, grid, curve[:, j]) for j in range(curve.shape[1])], 1)

                def move(target_values):
                    delta = torch.tensor(c_at(target_values) - c_at(z), device=args.device) @ comps
                    return answers(delta, pos[position])

                readout = 1 if variable in UNITS else 0  # units digit at E, tens digit at '='
                truth = (s_true % 10) if readout == 1 else (s_true // 10)
                entry = {"layer": layer, "position": position, "variable": variable, "integer_shift": {}}
                for s in ((1, 3, 5) if cyclic else (1, 2)):
                    lp = move(z + s)[:, readout]
                    if variable == "sum_units":
                        target = (truth + s) % 10
                    elif variable in ("A_units", "B_units"):
                        target = (truth + s) % 10
                    else:
                        target = truth + s
                    entry["integer_shift"][s] = float((lp.argmax(-1) == target).double().mean())
                if cyclic:
                    F0 = harmonic1(base[:, readout], truth)
                    gains = []
                    for m in range(M):
                        F = harmonic1(move(z + period * m / M)[:, readout], truth)
                        gains.append(complex(((F * F0.conj()).sum() / (F0.abs() ** 2).sum()).item()))
                    charges = list(range(-(M // 2) + 1, M // 2))
                    spec = {q: abs(sum(g * complex(math.cos(q * 2 * math.pi * m / M), math.sin(q * 2 * math.pi * m / M))
                                       for m, g in enumerate(gains)) / M) for q in charges}
                    entry["charge_spectrum"] = spec
                    half_curve = harmonic1(move(z + 0.5)[:, readout], truth)
                    chord = torch.tensor(0.5 * (c_at(z + 1) - c_at(z)), device=args.device) @ comps
                    half_chord = harmonic1(answers(chord, pos[position])[:, readout], truth)
                    entry["half_step_phase"] = {
                        "curve": float(torch.angle((half_curve * F0.conj()).sum())),
                        "chord": float(torch.angle((half_chord * F0.conj()).sum())),
                        "predicted": -math.pi / 10}
                report["cells"].append(entry)
                extra = ""
                if cyclic:
                    top = sorted(entry["charge_spectrum"].items(), key=lambda kv: -kv[1])[:3]
                    extra = f" charges {[(q, round(v, 3)) for q, v in top]} half-step {entry['half_step_phase']}"
                print(f"[move] layer {layer:2d} {position:8s} {variable:9s} shift acc {entry['integer_shift']}{extra}", flush=True)
        finally:
            handle.remove()
    with open(args.out, "w") as handle_out:
        json.dump(report, handle_out, indent=1)
    print(f"[move] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
