"""#2951 / #2234: does a language model compute on the weekday circle of its input embedding?

Analysis under SPEC 8's exception (benchmark evaluation, not an MPD input). torch/transformers only.

The seven weekday tokens are a declared cycle of rows of the INPUT embedding table, the setting of
``crates/gam-sae/src/parameter_decomposition/cyclic_action.rs``. Their rows expand exactly in the
characters of their cycle position: ``e_d = c_0 + sum_{k=1..3} U_k D(2 pi k d / 7)`` (seven rows, seven
basis columns, so the expansion is exact). The edits act on the input occurrence only: small Qwen3
models tie the unembedding to the embedding, and a global edit would also move the answer rows, a
different experiment (#2951's global versus use-specific edit).

* integer shift ``s`` of every plane: the row permutation ``e_d -> e_{d+s}``, the exact native
  reference (the model reading day ``d + s``);
* one plane ``k`` turned by ``2 pi k s / 7``, the others fixed: if the model reads the weekday through
  that plane's circle, the answer moves by ``s``; if it reads token identity, it does not;
* the angle path ``delta in [0, 1]`` (every plane turned by ``2 pi k delta / 7``) against the straight
  path ``(1 - delta) e_d + delta e_{d+1}``.

The readout is the model's distribution over the seven answer tokens, its circular phase
``arg sum_d p(d) e^{2 pi i d / 7}`` against the predicted phase, and the probability of the predicted day.
"""
from __future__ import annotations

import argparse
import json
import math

import torch

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
FEW_SHOT = ("Q: What day is two days after Friday? A: Sunday\n"
            "Q: What day is one day after Sunday? A: Monday\n"
            "Q: What day is three days after Tuesday? A: Friday\n")
TEMPLATES = [
    FEW_SHOT + "Q: What day is {n} days after {day}? A:",
    "If today is {day}, then {n} days from now it will be",
    "Today is {day}. In {n} days it will be",
]
WORDS = {1: "one", 2: "two", 3: "three", 4: "four"}
# the few-shot answers ("Sunday", "Monday", "Friday") put no weekday row in the query position but the cycle
# rows appear in the examples; only the queried day occurrence is edited (``where`` below picks the last).


def single_token_ids(tokenizer, words, prefix):
    ids = []
    for w in words:
        t = tokenizer.encode(prefix + w, add_special_tokens=False)
        if len(t) != 1:
            raise SystemExit(f"{prefix + w!r} is {len(t)} tokens; the cycle needs single tokens")
        ids.append(t[0])
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(args.device).eval()
    emb = model.get_input_embeddings().weight.detach()
    tied = model.get_output_embeddings().weight.data_ptr() == emb.data_ptr()
    in_ids = single_token_ids(tok, DAYS, " ")  # the day as it appears mid-sentence
    out_ids = in_ids  # answers follow a space too
    rows = emb[in_ids].double()  # 7 x d
    pos = torch.arange(7, dtype=torch.float64, device=rows.device)
    mean = rows.mean(0)
    U = {}
    for k in (1, 2, 3):
        ang = 2 * math.pi * k * pos / 7
        U[k] = torch.stack([(2 / 7) * torch.cos(ang) @ rows, (2 / 7) * torch.sin(ang) @ rows], -1)  # d x 2
    recon = mean[None] + sum(torch.cos(2 * math.pi * k * pos / 7)[:, None] * U[k][None, :, 0]
                             + torch.sin(2 * math.pi * k * pos / 7)[:, None] * U[k][None, :, 1] for k in U)
    report = {"model": args.model, "tied_embeddings": tied, "day_token_ids": in_ids,
              "expansion_max_abs_error": (recon - rows).abs().max().item(),
              "plane_power": {k: U[k].pow(2).sum().item() for k in U},
              "mean_power": mean.pow(2).sum().item(), "row_power": rows.pow(2).sum(1).mean().item()}
    print(f"[weekday] tied={tied} expansion_err={report['expansion_max_abs_error']:.2e} "
          f"plane_power={ {k: round(v, 4) for k, v in report['plane_power'].items()} } mean_power={report['mean_power']:.4f}", flush=True)

    def table_rows(turn):
        """Rows for an edit: ``turn[k]`` is plane k's angle as a function of the day index."""
        out = mean[None].repeat(7, 1)
        for k in U:
            a = 2 * math.pi * k * pos / 7 + turn[k]
            out = out + torch.cos(a)[:, None] * U[k][None, :, 0] + torch.sin(a)[:, None] * U[k][None, :, 1]
        return out

    prompts = []
    for template in TEMPLATES:
        for n in (1, 2, 3):
            for d in range(7):
                prompts.append((template.format(day=DAYS[d], n=WORDS[n]), d, n))

    def run(edited_rows):
        """Distribution over the seven answer days for every prompt, with the day rows replaced."""
        dists = []
        for text, d, n in prompts:
            ids = tok(text, return_tensors="pt").input_ids.to(args.device)
            embeds = model.get_input_embeddings()(ids).detach().clone()
            if edited_rows is not None:
                where = (ids[0] == in_ids[d]).nonzero().flatten()[-1:]
                embeds[0, where] = edited_rows[d].to(embeds.dtype)
            with torch.inference_mode():
                logits = model(inputs_embeds=embeds).logits[0, -1].double()
            dists.append(torch.softmax(logits[out_ids], -1).cpu())
        return torch.stack(dists)  # P x 7

    def score(dists, shift):
        """P(predicted day) and the circular phase error, with the prediction day + n + shift."""
        pred = torch.tensor([(d + n) for _, d, n in prompts], dtype=torch.float64) + shift
        ang = 2 * math.pi * torch.arange(7, dtype=torch.float64) / 7
        z = (dists * torch.complex(torch.cos(ang), torch.sin(ang))).sum(-1)
        err = torch.angle(z * torch.exp(-1j * 2 * math.pi * pred / 7))
        hit = dists[torch.arange(len(prompts)), (pred.round().long() % 7)]
        return {"p_predicted_mean": hit.mean().item(), "phase_error_median_abs": err.abs().median().item(),
                "resultant_mean": z.abs().mean().item(),
                "argmax_is_predicted": (dists.argmax(-1) == (pred.round().long() % 7)).double().mean().item()}

    zero = {k: 0.0 for k in U}
    base = run(None)
    report["baseline"] = score(base, 0)
    print(f"[weekday] baseline {report['baseline']}", flush=True)
    report["full_integer_shift"] = {}
    report["single_plane_shift"] = {}
    for s in (1, 2, 3):
        full = run(table_rows({k: 2 * math.pi * k * s / 7 for k in U}))
        report["full_integer_shift"][s] = {"vs_shifted": score(full, s), "vs_unshifted": score(full, 0)}
        for k in U:
            turn = dict(zero)
            turn[k] = 2 * math.pi * k * s / 7
            dist = run(table_rows(turn))
            report["single_plane_shift"][f"k{k}_s{s}"] = {"vs_shifted": score(dist, s), "vs_unshifted": score(dist, 0)}
        print(f"[weekday] s={s} full: {report['full_integer_shift'][s]['vs_shifted']}", flush=True)
        for k in U:
            e = report["single_plane_shift"][f"k{k}_s{s}"]
            print(f"   plane {k} alone: shifted p={e['vs_shifted']['p_predicted_mean']:.3f} argmax={e['vs_shifted']['argmax_is_predicted']:.3f} "
                  f"| unshifted p={e['vs_unshifted']['p_predicted_mean']:.3f} argmax={e['vs_unshifted']['argmax_is_predicted']:.3f}", flush=True)
    report["paths"] = {"angle": [], "straight": []}
    shifted_rows = rows[[(d + 1) % 7 for d in range(7)]]
    for delta in [i / 10 for i in range(11)]:
        a = run(table_rows({k: 2 * math.pi * k * delta / 7 for k in U}))
        st = run((1 - delta) * rows + delta * shifted_rows)
        for name, dist in (("angle", a), ("straight", st)):
            ang = 2 * math.pi * torch.arange(7, dtype=torch.float64) / 7
            z = (dist * torch.complex(torch.cos(ang), torch.sin(ang))).sum(-1)
            pred0 = torch.tensor([(d + n) for _, d, n in prompts], dtype=torch.float64)
            phase = torch.angle(z * torch.exp(-1j * 2 * math.pi * pred0 / 7))
            report["paths"][name].append({"delta": delta, "phase_median": phase.median().item(),
                                          "predicted_phase": 2 * math.pi * delta / 7,
                                          "resultant_mean": z.abs().mean().item()})
        print(f"[weekday] delta={delta:.1f} angle phase={report['paths']['angle'][-1]['phase_median']:.3f} "
              f"straight phase={report['paths']['straight'][-1]['phase_median']:.3f} predicted={2 * math.pi * delta / 7:.3f} "
              f"R angle={report['paths']['angle'][-1]['resultant_mean']:.3f} straight={report['paths']['straight'][-1]['resultant_mean']:.3f}", flush=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=1)
    print(f"[weekday] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
