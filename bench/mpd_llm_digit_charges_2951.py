"""#2951: charge spectroscopy of a pretrained language model's addition, read off its digit embeddings.

Analysis under SPEC 8's exception (not an MPD input); torch/transformers only.

Qwen3 tokenizes numbers digit by digit, so in ``AB+CD=T`` the next token is the units digit
``u = (B + D) mod 10`` whatever the carry (the tens digit ``T`` is given). The ten digit tokens are a declared
cycle of rows of the INPUT embedding, the setting of ``cyclic_action.rs``: the rows expand exactly as
``e_d = c_0 + sum_{k=1..4} U_k D(2 pi k d / 10) + u_5 cos(pi d)`` (ten rows, ten columns). The edits act on
one occurrence only (the operand's units digit), a use-specific edit; the embeddings are tied.

* Native control: every plane turned by ``2 pi k s / 10`` (and ``u_5`` by ``(-1)^s``) is the row permutation
  ``d -> d + s``, i.e. the model reading ``B + s``: the answer must move by ``s``.
* Charge spectroscopy: plane ``k`` alone turned by ``theta_m = 2 pi m / M``. The output's harmonic ``j`` is
  ``F_j = sum_c log p(c) e^{-2 pi i j (c - u) / 10}`` over the ten digit tokens, and the gain
  ``g_j(theta) = sum_x F_j(theta) conj F_j(0) / sum_x |F_j(0)|^2``, DFT'd over ``m``, gives its charge
  spectrum in plane ``k``. A model adding on the circle at period ``10/k`` shows charge 1 at ``j = k``. The
  period-2 line (``k = 5``) can only be reflected: its parity is the gain of ``j = 5`` under ``u_5 -> -u_5``.
* Paths: every plane turned by ``2 pi k delta / 10`` against the straight segment to the next digit's row: a
  circle computer moves the answer's phase linearly in ``delta``, a lookup snaps at ``delta = 1/2``.
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
    digit_ids = []
    for d in range(10):
        ids = tok.encode(str(d), add_special_tokens=False)
        assert len(ids) == 1, f"digit {d} is {ids}"
        digit_ids.append(ids[0])
    emb = model.get_input_embeddings().weight.detach()
    rows = emb[digit_ids].double()
    pos = torch.arange(10, dtype=torch.float64, device=rows.device)
    mean = rows.mean(0)
    U = {k: torch.stack([(2 / 10) * torch.cos(2 * math.pi * k * pos / 10) @ rows,
                         (2 / 10) * torch.sin(2 * math.pi * k * pos / 10) @ rows], -1) for k in (1, 2, 3, 4)}
    u5 = (1 / 10) * torch.cos(math.pi * pos) @ rows

    def table(turn, flip5=False):
        out = mean[None].repeat(10, 1) + (-1.0 if flip5 else 1.0) * torch.cos(math.pi * pos)[:, None] * u5[None]
        for k in U:
            a = 2 * math.pi * k * pos / 10 + turn.get(k, 0.0)
            out = out + torch.cos(a)[:, None] * U[k][None, :, 0] + torch.sin(a)[:, None] * U[k][None, :, 1]
        return out

    expansion = (table({}) - rows).abs().max().item()
    # prompts: two-digit operands with a two-digit sum, every units pair (B, D) `reps` times
    texts, units_b, units_d, answers = [], [], [], []
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
                units_d.append(d)
                answers.append(total % 10)
    enc = tok(texts, return_tensors="pt", padding=False)
    ids = torch.tensor(enc.input_ids, device=args.device)
    assert ids.dim() == 2, "prompts must share one length"
    L = ids.shape[1]
    # the operand units digits sit at fixed offsets from the end: A B + C D = T
    pos_b, pos_d = L - 6, L - 3
    assert all(ids[i, pos_b].item() == digit_ids[units_b[i]] for i in range(len(texts)))
    assert all(ids[i, pos_d].item() == digit_ids[units_d[i]] for i in range(len(texts)))
    ans = torch.tensor(answers, device=args.device)
    base_embeds = model.get_input_embeddings()(ids).detach()

    def logp(edited, sites):
        embeds = base_embeds.clone()
        for site, units in sites:
            embeds[torch.arange(len(texts)), site] = edited[units].to(embeds.dtype).to(args.device)
        out = []
        with torch.inference_mode():
            for s in range(0, len(texts), 50):
                logits = model(inputs_embeds=embeds[s:s + 50]).logits[:, -1].double()
                out.append(torch.log_softmax(logits[:, digit_ids], -1))
        return torch.cat(out)  # n x 10

    ub = torch.tensor(units_b, device=args.device)
    ud = torch.tensor(units_d, device=args.device)
    j = torch.arange(1, 6, dtype=torch.float64, device=args.device)
    c = torch.arange(10, dtype=torch.float64, device=args.device)

    def harmonics(lp, shift=0):
        centred = lp - lp.mean(-1, keepdim=True)
        rel = c[None, :] - (ans.double() + shift)[:, None]
        ang = 2 * math.pi * j[None, :, None] * rel[:, None, :] / 10
        return torch.complex((centred[:, None, :] * torch.cos(ang)).mean(-1), -(centred[:, None, :] * torch.sin(ang)).mean(-1))

    base = logp(table({}), [])
    F0 = harmonics(base)
    report = {"model": args.model, "expansion_max_abs_error": expansion, "prompts": len(texts),
              "plane_power": {k: U[k].pow(2).sum().item() for k in U} | {5: u5.pow(2).sum().item()},
              "baseline_accuracy": (base.argmax(-1) == ans).double().mean().item(),
              "baseline_p_correct": base.exp()[torch.arange(len(texts)), ans].mean().item(),
              "output_harmonic_power": F0.abs().pow(2).mean(0).tolist()}
    print(f"[digits] {args.model} expansion_err={expansion:.1e} acc={report['baseline_accuracy']:.3f} "
          f"p={report['baseline_p_correct']:.3f} plane_power={ {k: round(v, 3) for k, v in report['plane_power'].items()} } "
          f"harmonic power={[round(x, 3) for x in report['output_harmonic_power']]}", flush=True)
    # native control: relabel the units digit of B by s
    report["integer_shift"] = {}
    for s in (1, 3, 5):
        edited = table({k: 2 * math.pi * k * s / 10 for k in U}, flip5=(s % 2 == 1))
        lp = logp(edited, [(pos_b, ub)])
        moved = ((ans + s) % 10)
        report["integer_shift"][s] = {"acc_vs_shifted": (lp.argmax(-1) == moved).double().mean().item(),
                                      "acc_vs_unshifted": (lp.argmax(-1) == ans).double().mean().item()}
        print(f"[digits] integer shift {s}: {report['integer_shift'][s]}", flush=True)
    # charge spectroscopy
    M = args.angles
    thetas = [2 * math.pi * m / M for m in range(M)]
    charges = list(range(-(M // 2) + 1, M // 2))
    report["charges"] = {}
    for sites_name, sites in (("B", [(pos_b, ub)]), ("D", [(pos_d, ud)]), ("BD", [(pos_b, ub), (pos_d, ud)])):
        for k in U:
            gains = []
            for theta in thetas:
                lp = logp(table({k: theta}), sites)
                F = harmonics(lp)
                gains.append((F * F0.conj()).sum(0) / F0.abs().pow(2).sum(0))
            g = torch.stack(gains).cpu()  # M x 5
            basis = torch.tensor([[complex(math.cos(n * t), math.sin(n * t)) for t in thetas] for n in charges])
            spec = basis @ g / M  # charges x 5
            entry = {}
            for jj in range(5):
                mags = spec[:, jj].abs()
                top = torch.argsort(mags, descending=True)[:3].tolist()
                entry[str(jj + 1)] = [(charges[i], round(mags[i].item(), 3)) for i in top if mags[i] > 0.03]
            report["charges"][f"{k}/{sites_name}"] = entry
            print(f"[digits] plane {k} turned at {sites_name}: {entry}", flush=True)
        lp = logp(table({}, flip5=True), sites)
        F = harmonics(lp)
        parity = ((F * F0.conj()).sum(0) / F0.abs().pow(2).sum(0)).cpu()
        report["charges"][f"5/{sites_name}"] = {"gain_under_reflection": [[round(z.real, 3), round(z.imag, 3)] for z in parity.tolist()]}
        print(f"[digits] period-2 line reflected at {sites_name}: gains {report['charges'][f'5/{sites_name}']}", flush=True)
    # paths on B
    report["paths"] = {"angle": [], "straight": []}
    nxt = rows[[(d + 1) % 10 for d in range(10)]]
    for delta in [i / 10 for i in range(11)]:
        for name, edited in (("angle", table({k: 2 * math.pi * k * delta / 10 for k in U})),
                             ("straight", (1 - delta) * rows + delta * nxt)):
            lp = logp(edited, [(pos_b, ub)])
            F = harmonics(lp)
            phase = torch.angle((F[:, 0] * F0[:, 0].conj()).sum()).item()
            report["paths"][name].append({"delta": delta, "harmonic1_phase": phase,
                                          "p_shifted": lp.exp()[torch.arange(len(texts)), (ans + 1) % 10].mean().item()})
        print(f"[digits] delta={delta:.1f} phase angle={report['paths']['angle'][-1]['harmonic1_phase']:.3f} "
              f"straight={report['paths']['straight'][-1]['harmonic1_phase']:.3f} predicted={-2 * math.pi * delta / 10:.3f} "
              f"p(u+1) angle={report['paths']['angle'][-1]['p_shifted']:.3f} straight={report['paths']['straight'][-1]['p_shifted']:.3f}", flush=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=1)
    print(f"[digits] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
