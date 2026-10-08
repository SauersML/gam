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

  eval_kl.py --model Qwen/Qwen3-8B [--adapters OUT/adapters.safetensors] --heldout 'prompts=DIR/heldout_*.jsonl' [--heldout NAME=GLOB ...]
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
BINS = ((0.0, 0.01), (0.01, 0.1), (0.1, 1.0), (1.0, float("inf")))  # sizes of the measured change KL(M || M_e), bits
QWEN3_VOCAB = 151936  # older shards carry neither "vocab" nor decoded tokens; they are all Qwen3 targets


def target_tokens(q, which, tok):
    """The target's top tokens as its own tokenizer writes them: stored with the measurement, or (older Qwen3
    shards only, whose tokenizer the Qwen3 oracle shares) decoded from the ids."""
    n = q["numbers"][which]
    if "tokens" in n:
        return n["tokens"]
    if not q["model"].startswith("qwen3"):
        raise ValueError(f"{q['model']} question without decoded target tokens: regenerate it (generate.py stores them)")
    return [tok.decode([i]) for i in n["ids"]]


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


def read_delta(text: str, clean, V: int):
    """A delta answer ("KL x bits / up: TOKEN +p, ... / down: TOKEN -p, ...") applied to the clean listing the
    question states: listed tokens move by their stated change, others start at their share of the clean
    "other" mass; then renormalized. None when unreadable."""
    if clean is None:
        return None
    listed, other = dict(clean[0]), clean[1]
    moves = []
    for line in text.strip().split("\n")[1:]:
        if not line.startswith(("up:", "down:")):
            continue
        for part in line.split(":", 1)[1].split(", "):
            tok, _, v = part.strip().rpartition(" ")
            if not tok:
                continue
            try:
                moves.append((json.loads(tok), float(v)))
            except (ValueError, json.JSONDecodeError):
                return None
    if not text.startswith("KL ") or not moves:
        return None
    spread = other / max(1, V - len(listed))
    for t, v in moves:
        if t not in listed:
            listed[t] = spread
            other -= spread
        listed[t] += v
    listed = {t: max(p, 1e-6) for t, p in listed.items()}
    other = max(other, 1e-6)
    total = sum(listed.values()) + other
    return {t: p / total for t, p in listed.items()}, other / total


def clean_listing(q):
    """The clean distribution the question states (its "<clean> ..." line), read like an answer."""
    line = next((l[len("<clean> "):] for l in q["input"].split("\n") if l.startswith("<clean> ")), None)
    return None if line is None else read_answer(line)


def kl_bits(measured_strs, measured_p, answer, V: int = QWEN3_VOCAB) -> float:
    """KL(M_e || Q) over M_e's top 5 tokens and the rest, in bits (V = the target's output size)."""
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


records = []  # per-question results of every set, written beside the summary (OUT.questions.jsonl)


def score_set(model, tok, heldout, args, dev, name):
    """Per question type: the oracle's KL and the no-change answer's on the same questions."""
    result = {}
    for kind in TYPES:
        qs = heldout.get(kind, [])
        if args.stratify and qs and "kl_bits" in qs[0]["numbers"]:
            # Up to per_type / 4 questions from each size of the measured change, so large effects are represented.
            by_bin = [[q for q in qs if lo <= max(q["numbers"]["kl_bits"], 0.0) < hi] for lo, hi in BINS]
            qs = [q for b in by_bin for q in b[: max(1, args.per_type // len(BINS))]]
        else:
            qs = qs[: args.per_type]
        if not qs:
            continue
        ours, ref = [], []
        for s in range(0, len(qs), args.batch):
            chunk = qs[s : s + args.batch]
            enc = tok([prompt_text(tok, q) for q in chunk], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            with torch.no_grad():
                if sft.CHANNEL["module"] is not None or sft.PARTS["module"] is not None:  # part vectors enter through the embeddings
                    if sft.PARTS["module"] is not None:
                        emb = sft.PARTS["module"].embed(model, enc["input_ids"])
                    else:
                        parts = torch.tensor([sft.part_index(q) for q in chunk], device=dev)
                        emb = sft.embed(model, enc["input_ids"], parts)
                    gen = model.generate(inputs_embeds=emb, attention_mask=enc["attention_mask"], max_new_tokens=args.max_new,
                                         do_sample=False, pad_token_id=tok.pad_token_id or 0)
                    texts = tok.batch_decode(gen, skip_special_tokens=True)  # only the new tokens come back
                else:
                    gen = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False, pad_token_id=tok.pad_token_id or 0)
                    texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
            for q, text in zip(chunk, texts):
                n = q["numbers"]["edited"]
                strs, V = target_tokens(q, "edited", tok), q.get("vocab", QWEN3_VOCAB)
                delta = "rises and falls most" in q["input"]  # generate.py --answer delta
                ours.append(kl_bits(strs, n["p"], read_delta(text, clean_listing(q), V) if delta else read_answer(text), V))
                # Per question, for effect-size strata: the measured change, both scores and the answer written.
                records.append({"set": name, "type": kind, "text_id": q["text_id"], "measured_kl_bits": q["numbers"].get("kl_bits"),
                                "oracle_kl_bits": ours[-1], "answer": text})
                if "clean" in q["numbers"]:
                    c = q["numbers"]["clean"]
                    clean_answer = (dict(zip(target_tokens(q, "clean", tok)[:5], c["p"][:5])), max(0.0, 1.0 - sum(c["p"][:5])))
                    ref.append(kl_bits(strs, n["p"], clean_answer, V))
                    records[-1]["no_change_kl_bits"] = ref[-1]
        mean = sum(ours) / len(ours)
        se = (sum((x - mean) ** 2 for x in ours) / max(1, len(ours) - 1) / len(ours)) ** 0.5
        row = {"questions": len(ours), "kl_bits": mean, "se": se}
        if ref:
            d = [a - b for a, b in zip(ours, ref)]
            dm = sum(d) / len(d)
            row.update({"no_change_kl_bits": sum(ref) / len(ref), "difference_bits": dm,
                        "difference_se": (sum((x - dm) ** 2 for x in d) / max(1, len(d) - 1) / len(d)) ** 0.5})
        result[kind] = row
        print(json.dumps({name: {kind: row}}), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapters", default="")
    ap.add_argument("--heldout", action="append", required=True, help="NAME=GLOB (repeatable), each set scored separately")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-type", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--max-new", type=int, default=72)
    ap.add_argument("--part-tokens", default="", help="part_tokens.py registry, with --part-weights: the adapters were trained with part tokens")
    ap.add_argument("--part-weights", default="")
    ap.add_argument("--vectors", default="", help="vectors.py table, with --channel: the adapters were trained with the vector channel")
    ap.add_argument("--channel", default="", help="the channel weights saved beside the adapters")
    ap.add_argument("--stratify", action="store_true", help="per type, up to per_type/4 questions from each size of the measured change")
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
        if args.vectors:  # the trained vector channel (sft.py --vectors) beside the adapters
            sft.setup_channel(args.vectors, model, tok, dev, weights=args.channel)
        if args.part_tokens:  # the trained part-token maps (sft.py --part-tokens)
            sft.setup_part_tokens(args.part_tokens, model, tok, dev, weights=args.part_weights)
    out = {}
    for spec in args.heldout:
        name, _, pattern = spec.rpartition("=")
        name = name or "heldout"
        out[name] = score_set(model, tok, load(pattern), args, dev, name)
    Path(args.out).write_text(json.dumps({"model": args.model, "adapters": args.adapters, "sets": out}, indent=1))
    with open(Path(args.out).with_suffix(".questions.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
