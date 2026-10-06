"""Counterfactual simulatability after CHIVE (Karvonen et al., arXiv 2608.16747) on a library
explanation's accounts (examples/mpd_library_chive_2951, #2951).

python chive.py claims CHIVE.json CLAIMS.json [--true N] [--false N] [--seed S]
python chive.py predict CHIVE.json CLAIMS.json OUT.jsonl --condition {transcript,account} [--model M] [--workers W]
python chive.py score CLAIMS.json NAME=OUT.jsonl ...

A claim is that an edit (one token replaced) changes the model's probability of its predicted
token at the target by at least 30 percentage points; it is true when the measured change is at
least 50 points and false when at most 15 (others are not claims). Per target, `claims` keeps up
to N true and N false edits, drawn at random. `predict` asks a Claude model, with no tools and no
other context, for each claim's probability of being true, given the passage up to the target,
the model's top next tokens there and, under `account`, the explanation's account of the
unedited forward pass at the target. `score` gives each predictor's AUROC with a 95% interval
from resampling targets, and each pair's paired difference.
"""
import argparse
import concurrent.futures
import json
import os
import random
import re
import subprocess

import numpy as np

TRUE_CHANGE, FALSE_CHANGE, CLAIMED = 0.5, 0.15, 0.3
MODEL_NOTE = "a 4-layer transformer language model (width 768, 6 attention heads per layer, an MLP of 3072 units per layer) trained on English web text"


def quoted(text):
    return json.dumps(text, ensure_ascii=False)


def claims(args):
    report = json.load(open(args.chive))
    rng = random.Random(args.seed)
    out = []
    for i, target in enumerate(report["targets"]):
        true = [e for e in target["edits"] if abs(e["change"]) >= TRUE_CHANGE]
        false = [e for e in target["edits"] if abs(e["change"]) <= FALSE_CHANGE]
        chosen = [(e, True) for e in rng.sample(true, min(args.true, len(true)))] + [(e, False) for e in rng.sample(false, min(args.false, len(false)))]
        rng.shuffle(chosen)
        for k, (e, label) in enumerate(chosen):
            out.append({"target": i, "target_position": target["position"], "edit": f"E{k + 1}", "position": e["position"], "token": e["token"], "text": e["text"], "change": e["change"], "label": label})
    json.dump({"chive": os.path.abspath(args.chive), "claims": out}, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"{len(out)} claims, {sum(c['label'] for c in out)} true, on {len({c['target'] for c in out})} targets")


def account_text(target, passage):
    lines = [f"Explanation of the model's computation at position {target['position']} of the unedited passage. "
             "These are the functions whose removal (taking that function's output out and rerunning the later layers) raises "
             "the divergence of the model's next-token distribution at this position most, in bits (KL divergence from the "
             "unedited model's distribution). For each: where it attends from this position (heads), its activation (MLP units), "
             "and the tokens whose logits its output at this position raises and lowers most (direct effect)."]
    for rank, f in enumerate(target["account"], 1):
        layer, unit = f["name"].split(".")
        kind = f"attention head {unit[1:]} of layer {layer[1:]}" if unit.startswith("H") else f"MLP unit {unit[1:]} of layer {layer[1:]}"
        parts = [f"{rank}. {f['name']} ({kind}): removal effect {f['removal_bits']:.3f} bits."]
        if f["attention"]:
            parts.append("Attends to " + ", ".join(f"position {a['position']} {quoted(a['text'])} ({a['weight']:.2f})" for a in f["attention"]) + ".")
        if f["activation"] is not None:
            parts.append(f"Activation {f['activation']:.3f}.")
        parts.append("Raises " + ", ".join(f"{quoted(w['text'])} ({w['logit']:+.2f})" for w in f["promoted"]) + ".")
        parts.append("Lowers " + ", ".join(f"{quoted(w['text'])} ({w['logit']:+.2f})" for w in f["suppressed"]) + ".")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def prompt(target, passage, edits, condition):
    t = target["position"]
    tokens = " ".join(f"[{i}]{quoted(passage['text'][i])}" for i in range(t + 1))
    top = ", ".join(f"{quoted(w['text'])} {w['probability']:.3f}" for w in target["top"])
    edit_lines = "\n".join(f"{c['edit']}: replace the token at position {c['position']} {quoted(passage['text'][c['position']])} with {quoted(c['text'])}" for c in edits)
    parts = [
        f"The model is {MODEL_NOTE}. Below is a passage as the model's tokens, each with its position, up to position {t}.",
        tokens,
        f"At position {t} the model predicts the next token. Its most probable next tokens and their probabilities: {top}.",
        f"Call p the model's probability of {quoted(target['text'])} as the next token at position {t}.",
    ]
    if condition == "account":
        parts.append(account_text(target, passage))
    parts += [
        "Each edit below replaces one token of the passage and leaves every other token in place; the model is then run on the edited passage.",
        edit_lines,
        f"For each edit, give the probability that it changes p by at least {int(CLAIMED * 100)} percentage points (up or down). "
        'Answer with one JSON object mapping each edit to its probability, such as {"E1": 0.8, "E2": 0.1}, and nothing else.',
    ]
    return "\n\n".join(parts)


def ask(text, model, cwd):
    for _ in range(3):
        result = subprocess.run(["claude", "-p", "--model", model, "--tools", "", "--no-session-persistence", "--strict-mcp-config", "--setting-sources", ""],
                                input=text, capture_output=True, text=True, cwd=cwd, timeout=600)
        found = re.search(r"\{[^{}]*\}", result.stdout)
        if found:
            try:
                return json.loads(found.group(0)), result.stdout
            except json.JSONDecodeError:
                continue
    return None, result.stdout


def predict(args):
    report = json.load(open(args.chive))
    claim_set = json.load(open(args.claims))["claims"]
    by_target = {}
    for c in claim_set:
        by_target.setdefault(c["target"], []).append(c)
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(line)["target"] for line in open(args.out)}
    for c in claim_set:
        edits = report["targets"][c["target"]]["edits"]
        if not any(e["position"] == c["position"] and e["token"] == c["token"] and abs(e["change"] - c["change"]) < 1e-6 for e in edits):
            raise SystemExit(f"claim {c} is not an edit of {args.chive}: the claims come from another run")
    cwd = os.path.join(os.path.dirname(os.path.abspath(args.out)), "predictor_cwd")
    os.makedirs(cwd, exist_ok=True)

    def one(i):
        target = report["targets"][i]
        text = prompt(target, report["passages"][target["passage"]], by_target[i], args.condition)
        answer, raw = ask(text, args.model, cwd)
        return {"target": i, "condition": args.condition, "model": args.model, "answer": answer, "raw": raw if answer is None else None}

    todo = [i for i in sorted(by_target) if i not in done]
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool, open(args.out, "a") as out:
        for row in pool.map(one, todo):
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
    print(f"{len(todo)} targets asked")


def auroc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    positive, negative = scores[labels], scores[~labels]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    greater = (positive[:, None] > negative[None, :]).sum() + 0.5 * (positive[:, None] == negative[None, :]).sum()
    return greater / (len(positive) * len(negative))


def score(args):
    claim_set = json.load(open(args.claims))["claims"]
    predictors = {}
    for spec in args.runs:
        name, path = spec.split("=", 1)
        answers = {}
        for line in open(path):
            row = json.loads(line)
            answers[row["target"]] = row["answer"] or {}
        predictors[name] = np.array([float(answers.get(c["target"], {}).get(c["edit"], 0.5)) for c in claim_set])
    # Every claim, and the claims about edits before the target (the target's own token excluded).
    subsets = {"all": np.ones(len(claim_set), bool), "before_target": np.array([c["position"] < c["target_position"] for c in claim_set])}
    result = {}
    for subset, keep in subsets.items():
        labels = np.array([c["label"] for c in claim_set])[keep]
        targets = np.array([c["target"] for c in claim_set])[keep]
        kept = {n: p[keep] for n, p in predictors.items()}
        unique = np.unique(targets)
        rng = np.random.default_rng(0)
        samples = []
        for _ in range(args.resamples):
            drawn = rng.choice(unique, len(unique))
            index = np.concatenate([np.flatnonzero(targets == t) for t in drawn])
            samples.append({n: auroc(p[index], labels[index]) for n, p in kept.items()})
        part = {"claims": len(labels), "true": int(labels.sum()), "targets": len(unique), "auroc": {}, "differences": {}}
        for n, p in kept.items():
            values = [s[n] for s in samples]
            part["auroc"][n] = {"value": auroc(p, labels), "interval": [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]}
        names = list(kept)
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                x, y = names[a], names[b]
                values = [s[y] - s[x] for s in samples]
                part["differences"][f"{y} - {x}"] = {"value": part["auroc"][y]["value"] - part["auroc"][x]["value"], "interval": [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]}
        result[subset] = part
    print(json.dumps(result, indent=1))
    if args.out:
        json.dump(result, open(args.out, "w"), indent=1)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("claims")
    c.add_argument("chive")
    c.add_argument("out")
    c.add_argument("--true", type=int, default=3)
    c.add_argument("--false", type=int, default=3)
    c.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("predict")
    p.add_argument("chive")
    p.add_argument("claims")
    p.add_argument("out")
    p.add_argument("--condition", choices=["transcript", "account"], required=True)
    p.add_argument("--model", default="opus")
    p.add_argument("--workers", type=int, default=8)
    s = sub.add_parser("score")
    s.add_argument("claims")
    s.add_argument("runs", nargs="+")
    s.add_argument("--resamples", type=int, default=2000)
    s.add_argument("--out")
    args = parser.parse_args()
    {"claims": claims, "predict": predict, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
