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

`programs` writes the calibration programs: g-mech's vpd4l induction examples as written (examples/
index.json), the circuit's heads with measured removal facts as English (true_measured, facts from
disjoint prompts), and the same code with false English (wrong).

  reader_calibrate.py measure --out ITEMS.jsonl [--prompts 8] [--targets 4] [--heads all|L.H,...]
  reader_calibrate.py programs --out PROGRAMS.jsonl
then reader_score.py score --target vpd4l --programs PROGRAMS.jsonl --items ITEMS.jsonl --out OUT.json, and
  reader_calibrate.py report --out OUT.json
  reader_calibrate.py throughput [--model Qwen/Qwen3-8B] [--programs 8 --items 64 --text-tokens 64]
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
    index = json.loads((HERE / "examples" / "index.json").read_text())
    facts = json.loads(Path(args.facts).read_text())["mean_p_repeat"] if args.facts else None
    rows = [{"id": name, "source": (HERE / "examples" / f"{name}.py").read_text()} for name, e in sorted(index.items())
            if e["model"] == "vpd4l" and e["family"].startswith("induction")]
    rows.append({"id": "wrong", "source": WRONG})
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


def report(args):
    """The calibration table from reader_score.py's output: per program its mean bits per item and per
    family, the empty program and the code alone, and paired differences over the same items (mean and
    standard error): every program against true_measured, and each X_paraphrase against X."""
    d = json.loads(Path(args.out).read_text())
    rows = {r["id"]: r for r in d["results"]}
    bits = {k: np.array(r["per_item"]) for k, r in rows.items()}
    n = d["items"]
    lines = [f"reader {d['reader']}, {n} items, {d['seconds']:.0f} s, {d['item_reads_per_second']:.2f} item reads/s"]
    for k, r in rows.items():
        lines.append(f"{k:28s} {r['mean_bits_per_item']:.4f} bits/item  code alone {r.get('code_only_mean_bits_per_item', float('nan')):.4f}  "
                     f"empty {r.get('empty_mean_bits_per_item', float('nan')):.4f}  per family " + json.dumps({f: round(v, 3) for f, v in r["per_family"].items()}))

    def paired(a, b):
        x = bits[a] - bits[b]
        se = x.std(ddof=1) / np.sqrt(len(x))
        return f"{a} - {b}: {x.mean():+.4f} bits/item (SE {se:.4f}, {x.mean() / se:+.1f} SE)"

    for k, r in rows.items():
        if "per_item_empty" in r:
            bits[f"{k}:empty"], bits[f"{k}:code_alone"] = np.array(r["per_item_empty"]), np.array(r["per_item_code_only"])
            lines.append(paired(f"{k}:empty", k))
            lines.append(paired(f"{k}:code_alone", k))
    ref = "true_measured"
    for k in rows:
        if k != ref and ref in rows and not k.endswith("_paraphrase"):
            lines.append(paired(k, ref))
        if k.endswith("_paraphrase") and k[: -len("_paraphrase")] in rows:
            lines.append(paired(k, k[: -len("_paraphrase")]))
    for k, r in rows.items():
        if "english_saved_bits" in r:
            lines.append(f"{k:28s} English saves {(r['code_only_mean_bits_per_item'] - r['mean_bits_per_item']):+.4f} bits/item, program saves "
                         f"{(r['empty_mean_bits_per_item'] - r['mean_bits_per_item']):+.4f} bits/item over the empty program")
    if "item_words" in d:  # per experiment (its words), every program's mean bits
        groups: dict[str, list[int]] = {}
        for j, w in enumerate(d["item_words"]):
            groups.setdefault(w, []).append(j)
        lines.append("per experiment, mean bits per item: " + ", ".join(f"{k}" for k in rows) + ", empty")
        for w, js in groups.items():
            any_r = next(iter(rows.values()))
            empty = f"{np.mean(np.array(any_r['per_item_empty'])[js]):.3f}" if "per_item_empty" in any_r else "-"
            lines.append(f"  {w[:70]:70s} " + " ".join(f"{bits[k][js].mean():.3f}" for k in rows) + f" {empty}")
    print("\n".join(lines))


def throughput(args):
    """Reader throughput at RL shape: --programs programs (g-mech's examples, cycled, each made distinct)
    x --items items of a Qwen3 target (texts of --text-tokens tokens, 8 candidates), no baselines; prints
    items per second with the prefix and suffix lengths."""
    import time

    import reader_score as S

    tok_reader = S.CachedReader(args.model, args.batch_tokens, args.max_batch, device=args.device)
    sc = S.Scorer(tok_reader, "qwen3-0.6b")
    tok = tok_reader.tokenizer
    rng = np.random.default_rng(0)
    rows = np.load(Path.home() / "mpd-data/vpd/pile_val_4096x513.npy", mmap_mode="r")
    import tokenizers

    sys.path.insert(0, str(HERE.parent.parent / "vpd_2951"))
    import vpd_model as VM

    vtok = tokenizers.Tokenizer.from_file(str(VM.TARGET_DIR / "tokenizer.json"))
    items = []
    for i in range(args.items):
        text = vtok.decode(rows[rng.integers(len(rows))][1:200].tolist())
        ids = tok.encode(text, add_special_tokens=False)[: args.text_tokens]
        cands = rng.choice(len(tok), 8, replace=False).tolist()
        p = rng.dirichlet(np.ones(9))
        items.append({"id": str(i), "family": "bench", "experiment": {"kind": "scale", "factor": 0, "pieces": [{"view": "native", "layer": 3, "kind": "head", "index": 5}]},
                      "text": tok.decode(ids), "token_ids": ids, "candidates": [{"token_id": c, "text": tok.decode([c]), "clean": float(q), "p": float(q)} for c, q in zip(cands, p[:8])],
                      "clean_other": float(p[8]), "other": float(p[8])})
    examples = sorted((HERE / "examples").glob("*.py"))
    programs = [{"id": str(j), "source": f"# variant {j}\n" + examples[j % len(examples)].read_text()} for j in range(args.programs)]
    prefix = np.mean([len(sc.prompter.prefix(p["source"])) for p in programs])
    suffix = np.mean([len(sc.prompter.item(it)) for it in items])
    start = time.time()
    sc.score(programs[:1], items[:2], baselines=False)  # warm up
    warm = time.time() - start
    start = time.time()
    sc.score(programs, items, baselines=False)
    seconds = time.time() - start
    reads = len(programs) * len(items)
    print(json.dumps({"reader": tok_reader.describe(), "programs": len(programs), "items_per_program": len(items), "mean_prefix_tokens": float(prefix),
                      "mean_item_tokens": float(suffix), "seconds": seconds, "warmup_seconds": warm, "items_per_second": reads / seconds,
                      "max_batch": args.max_batch, "batch_tokens": args.batch_tokens}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["measure", "programs", "report", "throughput"])
    ap.add_argument("--out", help="measure/programs: the file written; report: reader_score.py's output JSON")
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--targets", type=int, default=4)
    ap.add_argument("--heads", default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--facts", help="programs: measure's printed JSON, for the true program's measured facts")
    ap.add_argument("--model", default="Qwen/Qwen3-8B", help="throughput: the reader")
    ap.add_argument("--device")
    ap.add_argument("--programs", type=int, default=8, help="throughput: programs")
    ap.add_argument("--items", type=int, default=64, help="throughput: items per program")
    ap.add_argument("--text-tokens", type=int, default=64, help="throughput: tokens of each item's text")
    ap.add_argument("--batch-tokens", type=int, default=65536)
    ap.add_argument("--max-batch", type=int, default=32)
    args = ap.parse_args()
    {"measure": measure, "programs": programs, "report": report, "throughput": throughput}[args.command](args)


if __name__ == "__main__":
    main()
