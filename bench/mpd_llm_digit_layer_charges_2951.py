"""#2951 / #2234: where in depth does a language model's units-digit addition run on a circle?

Analysis under SPEC 8's exception (not an MPD input); torch/transformers only. Companion of
``mpd_llm_digit_charges_2951.py``, which acts on the input embedding.

At decoder layer ``L`` the residual at operand B's units-digit position, averaged over the prompts whose digit
is ``d``, is a declared cycle table ``H_L(d)`` of ten rows. It expands exactly in the characters of ``d``
(``cyclic_action.rs``'s closed form, no fit): ``H_L(d) = c_0 + sum_{k=1..4} U_k D(2 pi k d/10) + u_5 cos(pi d)``.
An intervention at layer ``L`` adds to each prompt's residual the change its digit's table row undergoes:

* integer shift ``s`` (every plane turned by ``2 pi k s/10``, ``u_5`` by ``(-1)^s``): the row
  ``H_L(b) -> H_L(b + s)``, mean patching of the digit; the answer must move by ``s`` if layer ``L`` carries it;
* plane ``k`` alone turned by ``theta``: the charge of each output harmonic, as in the embedding script;
* the angle path against the straight segment ``H_L(b) -> H_L(b + 1)``.

The depth profile of these three reads where the digit is a point on circles the downstream computation adds.
"""
from __future__ import annotations

import argparse
import json
import math
import random

import torch

FEW_SHOT = "12+35=47\n40+19=59\n23+64=87\n51+38=89\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", required=True, help="comma list, or 'every:N'")
    parser.add_argument("--angles", type=int, required=True)
    parser.add_argument("--reps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(args.device).eval()
    blocks = model.model.layers
    if args.layers.startswith("every:"):
        step = int(args.layers.split(":")[1])
        layers = list(range(0, len(blocks), step)) + ([len(blocks) - 1] if (len(blocks) - 1) % step else [])
    else:
        layers = [int(x) for x in args.layers.split(",")]
    digit_ids = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(10)]
    texts, units_b, answers = [], [], []
    for b in range(10):
        for d in range(10):
            for _ in range(args.reps):
                while True:
                    a, c = rng.randint(1, 8), rng.randint(1, 8)
                    total = (10 * a + b) + (10 * c + d)
                    if 10 <= total < 100:
                        break
                texts.append(FEW_SHOT + f"{10 * a + b}+{10 * c + d}={total // 10}")
                units_b.append(b)
                answers.append(total % 10)
    ids = torch.tensor(tok(texts).input_ids, device=args.device)
    L = ids.shape[1]
    pos_b = L - 6
    assert all(ids[i, pos_b].item() == digit_ids[units_b[i]] for i in range(len(texts)))
    ub = torch.tensor(units_b, device=args.device)
    ans = torch.tensor(answers, device=args.device)
    n = len(texts)
    c = torch.arange(10, dtype=torch.float64, device=args.device)
    j = torch.arange(1, 6, dtype=torch.float64, device=args.device)
    pos = torch.arange(10, dtype=torch.float64, device=args.device)
    state = {"delta": None}

    def hook(_module, _inputs, output):
        if state["delta"] is None:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden.clone()
        hidden[torch.arange(hidden.shape[0], device=hidden.device), pos_b] += state["delta"][state["slice"]].to(hidden.dtype)
        return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden

    def run(delta=None):
        out = []
        for s in range(0, n, 50):
            state["delta"], state["slice"] = delta, slice(s, s + 50)
            with torch.inference_mode():
                logits = model(input_ids=ids[s:s + 50]).logits[:, -1].double()
            out.append(torch.log_softmax(logits[:, digit_ids], -1))
        state["delta"] = None
        return torch.cat(out)

    def harmonics(lp):
        centred = lp - lp.mean(-1, keepdim=True)
        rel = c[None, :] - ans.double()[:, None]
        ang = 2 * math.pi * j[None, :, None] * rel[:, None, :] / 10
        return torch.complex((centred[:, None, :] * torch.cos(ang)).mean(-1), -(centred[:, None, :] * torch.sin(ang)).mean(-1))

    base = run()
    F0 = harmonics(base)
    report = {"model": args.model, "layers": layers, "prompts": n,
              "baseline_accuracy": (base.argmax(-1) == ans).double().mean().item(), "per_layer": {}}
    print(f"[layers] {args.model} acc={report['baseline_accuracy']:.3f} layers={layers}", flush=True)
    M = args.angles
    thetas = [2 * math.pi * m / M for m in range(M)]
    charges = list(range(-(M // 2) + 1, M // 2))
    basis = torch.tensor([[complex(math.cos(q * t), math.sin(q * t)) for t in thetas] for q in charges], dtype=torch.complex128)
    for layer in layers:
        handle = blocks[layer].register_forward_hook(hook)
        try:
            captured = {}

            def grab(_m, _i, output):
                h = output[0] if isinstance(output, tuple) else output
                captured.setdefault("h", []).append(h[:, pos_b].detach().double())

            g = blocks[layer].register_forward_hook(grab)
            with torch.inference_mode():
                for s in range(0, n, 50):
                    model(input_ids=ids[s:s + 50])
            g.remove()
            H = torch.cat(captured["h"])  # n x hidden
            table = torch.stack([H[ub == d].mean(0) for d in range(10)])  # 10 x hidden
            mean = table.mean(0)
            U = {k: torch.stack([(2 / 10) * torch.cos(2 * math.pi * k * pos / 10) @ table,
                                 (2 / 10) * torch.sin(2 * math.pi * k * pos / 10) @ table], -1) for k in (1, 2, 3, 4)}
            u5 = (1 / 10) * torch.cos(math.pi * pos) @ table
            power = {k: U[k].pow(2).sum().item() for k in U} | {5: u5.pow(2).sum().item()}
            spread = (H - table[ub]).pow(2).sum(-1).mean().item()

            def moved_rows(turn, flip5=False):
                rows = mean[None].repeat(10, 1) + (-1.0 if flip5 else 1.0) * torch.cos(math.pi * pos)[:, None] * u5[None]
                for k in U:
                    a = 2 * math.pi * k * pos / 10 + turn.get(k, 0.0)
                    rows = rows + torch.cos(a)[:, None] * U[k][None, :, 0] + torch.sin(a)[:, None] * U[k][None, :, 1]
                return rows - table  # the change of every digit's row

            entry = {"plane_power": power, "within_digit_spread": spread, "integer_shift": {}, "charge1_on_matching_harmonic": {}}
            for s in (1, 3, 5):
                lp = run(moved_rows({k: 2 * math.pi * k * s / 10 for k in U}, flip5=(s % 2 == 1))[ub])
                entry["integer_shift"][s] = (lp.argmax(-1) == (ans + s) % 10).double().mean().item()
            for k in U:
                gains = []
                for theta in thetas:
                    F = harmonics(run(moved_rows({k: theta})[ub]))
                    gains.append((F * F0.conj()).sum(0) / F0.abs().pow(2).sum(0))
                spec = basis @ torch.stack(gains).cpu() / M  # charges x 5
                entry["charge1_on_matching_harmonic"][k] = {
                    "harmonic": k, "charge_spectrum": {str(q): round(spec[i, k - 1].abs().item(), 3) for i, q in enumerate(charges)}}
            phases = {}
            for name in ("angle", "straight"):
                d = 0.5
                delta = moved_rows({k: 2 * math.pi * k * d / 10 for k in U}) if name == "angle" else \
                    0.5 * (table[[(x + 1) % 10 for x in range(10)]] - table)
                F = harmonics(run(delta[ub]))
                phases[name] = torch.angle((F[:, 0] * F0[:, 0].conj()).sum()).item()
            entry["half_step_phase"] = phases | {"predicted": -math.pi / 10}
            report["per_layer"][layer] = entry
            c1 = {k: v["charge_spectrum"].get("1", 0.0) for k, v in entry["charge1_on_matching_harmonic"].items()}
            print(f"[layer {layer}] shift acc s=1,3,5 {[round(entry['integer_shift'][s], 3) for s in (1, 3, 5)]} "
                  f"charge-1 weight on harmonic k by plane k {c1} half-step phase {phases} (pred {-math.pi / 10:.3f}) "
                  f"plane power {({k: round(v, 2) for k, v in power.items()})} spread {spread:.2f}", flush=True)
        finally:
            handle.remove()
    with open(args.out, "w") as handle_out:
        json.dump(report, handle_out, indent=1)
    print(f"[layers] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
