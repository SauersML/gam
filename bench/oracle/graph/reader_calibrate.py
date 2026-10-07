"""Calibration of the reader term (#2951) on vpd4l induction: does the frozen reader score a program with
true English better than the same code with deliberately wrong English, and better than the code alone
and the empty program?

`measure` writes the items: repeated Pile text ([first token] + S + S, S the next SEG tokens of a
validation row), target tokens in the second copy of S (where induction predicts the repeat), and the
experiments clean and removal of each native head of vpd4l (its output columns multiplied by 0, the head's
whole write), measured on M in float32 with log-probabilities normalized in float64. Each item carries M's
clean top-K candidates and their probabilities under the experiment (reader_score.py's item format).
It also prints the mean measured effect of each removal on the repeated token, the facts the true
docstrings state.

`programs` writes the calibration programs: g-mech's native induction example as written, the same code
with its English replaced by measured facts (true), and the same code with false English (wrong).

  reader_calibrate.py measure --out ITEMS.jsonl [--prompts 8] [--targets 4] [--heads all|L.H,...]
  reader_calibrate.py programs --out PROGRAMS.jsonl
then reader_score.py score --target vpd4l --programs PROGRAMS.jsonl --items ITEMS.jsonl ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEG, K = 20, 10

WRONG = '''"""Next-token prediction on vpd4l from the current token alone, a bigram table.

None of the attention heads below takes part in this behaviour. The MLPs map the current token to its
most frequent successor in the training data; earlier tokens in the context, repeated or not, do not
change the prediction. Removing L[1].head[1], L[2].head[4] or L[3].head[5] leaves the next-token
distribution unchanged.
"""
from mech import node, edges, L, embed, logits

prev = node(L[1].head[1])       # attends to the first token only; its write is constant across positions
induction = node(L[2].head[4])  # attends to the current token itself; no effect on the output
copy = node(L[3].head[5])       # a no-op head kept for completeness

edges(
    embed >> prev.query,
    embed >> prev.key,
    embed >> prev.value,
    prev >> induction.key,
    embed >> induction.query,
    embed >> induction.value,
    induction >> copy.value,
    induction >> logits,
    copy >> logits,
)
'''

TRUE_TEMPLATE = '''"""Induction on vpd4l: when the current token A appeared earlier in the context, the model predicts the
token B that followed A there ("A B ... A -> B"), with high probability on repeated text.

L[1].head[1] (previous-token head) writes at every position the token before it. L[2].head[4] (induction
head) reads that write through its key, so its query for the current token A matches the position right
after the earlier A, and it copies the token there (B) into the stream. L[3].head[5] reads the induction
head through its value and writes B again.

Measured on repeated text, mean probability of the repeated token B: {facts}
"""
from mech import node, edges, L, embed, logits

prev = node(L[1].head[1])       # previous-token head: attends from t to t-1, writes x[t-1]
induction = node(L[2].head[4])  # attends from the current A to the position after the earlier A, copies B
copy = node(L[3].head[5])       # reads the induction head's write through its value, writes B again

edges(
    embed >> prev.query,
    embed >> prev.key,
    embed >> prev.value,
    prev >> induction.key,
    embed >> induction.query,
    embed >> induction.value,
    induction >> copy.value,
    induction >> logits,
    copy >> logits,
)
'''


def load():
    import torch

    sys.path.insert(0, str(HERE.parent.parent / "vpd_2951"))
    import tokenizers
    import vpd_model as VM

    t = VM.load_target("cpu")
    tok = tokenizers.Tokenizer.from_file(str(VM.TARGET_DIR / "tokenizer.json"))
    return torch, VM, t, tok


def log_probs(torch, t, ids, removed):
    """M's next-token log-probabilities [B, T, V] (float64) with the heads in `removed` (layer, head)
    removed: their slices of o_proj's input multiplied by 0."""
    for layer in range(t.n_layer):
        heads = [h for (l, h) in removed if l == layer]
        site = t.site(f"h.{layer}.attn.o_proj")
        if heads:
            mask = torch.ones(t.n_head * t.hd)
            for h in heads:
                mask[h * t.hd : (h + 1) * t.hd] = 0.0
            site.in_fn = lambda x, m=mask: x * m
        else:
            site.in_fn = None
    with torch.no_grad():
        out = torch.log_softmax((t.hidden(ids) @ t.wte.T).double(), -1)
    for layer in range(t.n_layer):
        t.site(f"h.{layer}.attn.o_proj").in_fn = None
    return out


def measure(args):
    torch, VM, t, tok = load()
    rows = np.load(VM.DATA, mmap_mode="r")
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(rows), args.prompts, replace=False)
    seqs = [np.concatenate([rows[r][:1], rows[r][1 : 1 + SEG], rows[r][1 : 1 + SEG]]).astype(np.int64) for r in picks]
    ids = torch.from_numpy(np.stack(seqs))
    # Target positions t predict token t + 1; in the second copy, positions SEG + 1 .. 2 SEG - 1 predict a repeat.
    targets = [sorted(rng.choice(np.arange(SEG + 2, 2 * SEG), args.targets, replace=False).tolist()) for _ in seqs]
    heads = [(l, h) for l in range(t.n_layer) for h in range(t.n_head)] if args.heads == "all" else [tuple(int(x) for x in s.split(".")) for s in args.heads.split(",")]
    experiments = [("clean", [], {"kind": "clean"})] + [
        ("remove", [hd], {"kind": "scale", "factor": 0, "pieces": [{"view": "native", "layer": hd[0], "kind": "head", "index": hd[1]}]}) for hd in heads]
    clean = log_probs(torch, t, ids, []).exp()
    n = 0
    effects = {}
    with open(args.out, "w") as f:
        for family, removed, op in experiments:
            pe = clean if not removed else log_probs(torch, t, ids, removed).exp()
            repeat = []
            for r, seq in enumerate(seqs):
                for pos in targets[r]:
                    top = torch.argsort(-clean[r, pos])[:K].tolist()
                    cands = [{"token_id": int(c), "text": tok.decode([int(c)]), "clean": float(clean[r, pos, c]), "p": float(pe[r, pos, c])} for c in top]
                    item = {"id": f"{family}:{'+'.join(f'L{l}H{h}' for l, h in removed) or 'none'}:{r}:{pos}", "family": family, "experiment": op,
                            "text": tok.decode(seq[: pos + 1].tolist()), "token_ids": seq[: pos + 1].tolist(), "candidates": cands,
                            "clean_other": float(max(0.0, 1.0 - sum(c["clean"] for c in cands))), "other": float(max(0.0, 1.0 - sum(c["p"] for c in cands))),
                            "repeat_token": int(seq[pos + 1])}
                    repeat.append(float(pe[r, pos, int(seq[pos + 1])]))
                    f.write(json.dumps(item) + "\n")
                    n += 1
            effects["+".join(f"L{l}H{h}" for l, h in removed) or "clean"] = float(np.mean(repeat))
    print(json.dumps({"items": n, "mean_p_repeat": effects}, indent=1))


def programs(args):
    native = (HERE / "examples" / "vpd4l_induction_native.py").read_text()
    facts = json.loads(Path(args.facts).read_text())["mean_p_repeat"] if args.facts else None
    rows = [{"id": "native_example", "source": native}, {"id": "wrong", "source": WRONG}]
    if facts:
        named = ["L1H1", "L2H4", "L3H5"]
        rest = sorted((k for k in facts if k not in named and k != "clean"), key=lambda k: facts[k])
        say = lambda k: f"L[{k[1]}].head[{k[3:]}] removed {facts[k]:.2f}"  # noqa: E731
        line = (f"{facts['clean']:.2f} on the model as it is; " + "; ".join(say(k) for k in named)
                + ". With any other single head removed (strongest first): " + "; ".join(say(k) for k in rest) + ".")
        rows.append({"id": "true_measured", "source": TRUE_TEMPLATE.replace("{facts}", line)})
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} programs -> {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["measure", "programs"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--targets", type=int, default=4)
    ap.add_argument("--heads", default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--facts", help="programs: measure's printed JSON, for the true program's measured facts")
    args = ap.parse_args()
    measure(args) if args.command == "measure" else programs(args)


if __name__ == "__main__":
    main()
