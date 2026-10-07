"""Held-out prediction error of the oracle in bits of M's measured distribution (#2951): the oracle (base
model, optionally with sft.py's adapters) writes its answer greedily to each distribution question
(plain, edit, cut, prompt, swap), the answer is read as a distribution, and the score is
KL(M_e || Q) in bits over the bins of the answer format: M_e's top 5 tokens and the rest.

Reading an answer ("TOKEN p | TOKEN p | ... | other o"): a listed token has its stated probability (at
least the spread below, since the answer rounds to 0.01); the "other" mass o is spread uniformly over the V - n tokens the answer does not list (V the vocabulary,
n the listed count), so an omitted top token of M_e costs log2 of that spread, and the rest bin gets
the remaining spread plus the listed tokens outside M_e's top 5. Stated numbers are renormalized to sum
to 1; an unreadable answer is scored as Q uniform over the vocabulary.

The reference answer for intervention questions is "no change": the clean distribution (given in the
question) read the same way; plain questions have none. Written to OUT: per type the mean KL in bits,
its standard error, the no-change reference's mean on the same questions, and the paired difference.

  eval_kl.py --model Qwen/Qwen3-8B [--adapters OUT/adapters.safetensors] --heldout 'DIR/heldout_*.jsonl'
             --out EVAL.json [--per-type 128] [--batch 16] [--rank 16] [--alpha 32] [--format chat|raw]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import torch

import sft
from sft import load, prompt_text, wrap

TYPES = ("plain", "edit", "cut", "prompt", "swap")
V = 151936


def read_answer(text: str):
    """{token string: probability} and the other mass, or None when unreadable."""
    line = text.strip().split("\n")[0]
    listed, other = {}, None
    for part in line.split(" | "):
        part = part.strip()
        if part.startswith("other "):
            try:
                other = float(part[6:])
            except ValueError:
                return None
            continue
        tok, _, p = part.rpartition(" ")
        try:
            listed[json.loads(tok)] = float(p)
        except (ValueError, json.JSONDecodeError):
            return None
    if other is None or not listed:
        return None
    total = sum(listed.values()) + other
    if total <= 0:
        return None
    return {k: v / total for k, v in listed.items()}, other / total


def kl_bits(measured_strs, measured_p, answer) -> float:
    """KL(M_e || Q) over M_e's top 5 tokens and the rest, in bits."""
    top = list(zip(measured_strs[:5], measured_p[:5]))
    rest = max(1e-12, 1.0 - sum(p for _, p in top))
    if answer is None:
        q = {s: 1.0 / V for s, _ in top}
        q_rest = 1.0 - 5.0 / V
    else:
        listed, other = answer
        spread = other / max(1, V - len(listed))
        q = {s: max(listed.get(s, 0.0), spread) for s, _ in top}  # a stated 0.00 is a rounded value, not zero
        q_rest = max(1e-12, 1.0 - sum(q.values()))
    out = sum(p * math.log2(p / max(q[s], 1e-12)) for s, p in top if p > 0)
    return out + rest * math.log2(rest / q_rest)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapters", default="")
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-type", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--max-new", type=int, default=72)
    ap.add_argument("--format", default="chat", choices=("chat", "raw"), help="the format the adapters were trained with (sft.py)")
    args = ap.parse_args()
    sft.FORMAT["name"] = args.format
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = torch.device("cuda" if torch.cuda.is_available() else "mps")
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev).eval()
    if args.adapters:
        from safetensors.torch import load_file

        adapters = wrap(model, args.rank, args.alpha)
        state = load_file(args.adapters)
        for name, a in adapters.items():
            a.A.data.copy_(state[f"{name}.A"])
            a.B.data.copy_(state[f"{name}.B"])
    heldout = load(args.heldout)
    result = {}
    for kind in TYPES:
        qs = heldout.get(kind, [])[: args.per_type]
        if not qs:
            continue
        ours, ref = [], []
        for s in range(0, len(qs), args.batch):
            chunk = qs[s : s + args.batch]
            enc = tok([prompt_text(tok, q) for q in chunk], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False, pad_token_id=tok.pad_token_id or 0)
            texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
            for q, text in zip(chunk, texts):
                n = q["numbers"]["edited"]
                strs = [tok.decode([i]) for i in n["ids"]]
                ours.append(kl_bits(strs, n["p"], read_answer(text)))
                if "clean" in q["numbers"]:
                    c = q["numbers"]["clean"]
                    clean_answer = ({tok.decode([i]): p for i, p in zip(c["ids"][:5], c["p"][:5])}, max(0.0, 1.0 - sum(c["p"][:5])))
                    ref.append(kl_bits(strs, n["p"], clean_answer))
        mean = sum(ours) / len(ours)
        se = (sum((x - mean) ** 2 for x in ours) / max(1, len(ours) - 1) / len(ours)) ** 0.5
        row = {"questions": len(ours), "kl_bits": mean, "se": se}
        if ref:
            d = [a - b for a, b in zip(ours, ref)]
            dm = sum(d) / len(d)
            row.update({"no_change_kl_bits": sum(ref) / len(ref), "difference_bits": dm,
                        "difference_se": (sum((x - dm) ** 2 for x in d) / max(1, len(d) - 1) / len(d)) ** 0.5})
        result[kind] = row
        print(json.dumps({kind: row}), flush=True)
    Path(args.out).write_text(json.dumps({"model": args.model, "adapters": args.adapters, "per_type": result}, indent=1))


if __name__ == "__main__":
    main()
