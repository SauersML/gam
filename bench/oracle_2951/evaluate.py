#!/usr/bin/env python3
"""Evaluate a frozen report on tests drawn after it was frozen (#2951).

The tests come from fresh randomness drawn here, over corpus rows the investigator could not read,
and every outcome is measured by the oracle server on the native models. The report's sha256 is
checked first; the tests and outcomes are written next to it.

Outcome definition (independent of any report): at a position of a test row, the rule FIRES when
the subject model's top next token differs from the reference's, where the reference is the same
model reading only the last `keep` tokens (task reference {"kind": "context", "keep": k}) or
another model on the same input ({"kind": "model", "name": "base"}). The rule's token there is the
subject's top token.

Scores:
  hypothesis   the report's fires() against measured firing on a sample half firing, half not
               (accuracy), and its effect() against the subject's token where the rule fires
  components   all report components at alpha = 0: the fraction of firing items whose top token
               changes, and the KL(subject || ablated) on non-firing items (collateral), beside the
               same for random head sets of the same size
  crossed      (context reference) x1 = the input, x0 = its last `keep` tokens, a1 = the report's
               components at alpha = 0: the share of the input's effect on the rule token's
               log-probability that the components carry, -gamma / effect
  edit         the report's edit: the fraction of firing items whose top token changes, the change
               of held-out cross-entropy (nats per token) and KL(subject || edited) on held-out text
  readers      items (input, intervention): a random earlier token replaced, or one head's or MLP's
               output set to zero. Y = 1 when the subject's top token stays the rule's token. A reader
               model with no tools gives P(Y = 1) from (a) the report's plain-language fields,
               (b) nothing (transcript-only baseline), (c) an ablated report with the same concepts
               and a wrong relationship. Information the report adds: mean log2-score of (a) minus (b).
               The executable hypothesis predicts the same items (fires/effect on edited inputs,
               component membership for ablations).

usage: MPD_MEM_GIB=1 venv python evaluate.py RUN_DIR [--items N]
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import tokenizers

sys.path.insert(0, str(Path(__file__).resolve().parent))
import investigate  # noqa: E402

READER_MODEL = "claude-sonnet-5-5"
MAX_ROWS_PER_REQUEST = 512


class Client:
    def __init__(self, sock):
        self.sock = sock

    def __call__(self, payload):
        import socket
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(self.sock)
            s.sendall((json.dumps(payload) + "\n").encode())
            chunks = []
            while not chunks or not chunks[-1].endswith(b"\n"):
                b = s.recv(1 << 20)
                if not b:
                    break
                chunks.append(b)
        reply = json.loads(b"".join(chunks))
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply["ok"]


def batches(sequences):
    """Index groups whose total length stays within one request's rows."""
    group, rows = [], 0
    for i, s in enumerate(sequences):
        if group and rows + len(s) > MAX_ROWS_PER_REQUEST:
            yield group
            group, rows = [], 0
        group.append(i)
        rows += len(s)
    if group:
        yield group


def last_runs(oracle, model, sequences, intervention=None, targets=()):
    """Per sequence, the run entry at its last position (top tokens, targets, KL from clean)."""
    out = [None] * len(sequences)
    for group in batches(sequences):
        payload = {"op": "run", "model": model, "sequences": [sequences[i] for i in group], "top": 1, "targets": list(targets)}
        if intervention:
            payload.update({"intervention": intervention, "clean": True})
        reply = oracle(payload)
        for i, run in zip(group, reply["runs"]):
            out[i] = run["positions"][0]
    return out


def top1(entry):
    return entry["top"][0][0]


def load_functions(report):
    scope = {}
    errors = []
    for key in ("fires", "effect"):
        try:
            exec(report[key], scope)
        except Exception as e:  # the report's own code; a failure scores as wrong predictions
            errors.append(f"{key}: {e!r}")
    return scope.get("fires"), scope.get("effect"), errors


def call(f, pieces):
    try:
        return f(pieces)
    except Exception:
        return None


def site_of(component):
    """The site (layer, 'head', h) or (layer, 'mlp') a report component edits, if one."""
    kind = component.get("kind")
    if kind == "head":
        return (component["layer"], "head", component["head"])
    if kind == "neuron":
        return (component["layer"], "mlp")
    name = component.get("name", "")
    if name.startswith("blocks."):
        _, layer, part = name.split(".", 2)
        layer = int(layer)
        if part in ("c_fc", "gate_proj", "down_proj"):
            return (layer, "mlp")
        if part[0] in "qo" and part[1:].isdigit():
            return (layer, "head", int(part[1:]))
        if part[0] in "kv" and part[1:].isdigit():
            return (layer, "kv", int(part[1:]))
    return None


def reader(prompt_prefix, items, label, out_dir, model=READER_MODEL):
    """P(Y = 1) per item from a reader with no tools, as integer percents 1..99."""
    schema = {"type": "object", "additionalProperties": False, "required": ["percents"],
              "properties": {"percents": {"type": "array", "items": {"type": "integer", "minimum": 1, "maximum": 99}}}}
    questions = "\n\n".join(f"Item {i}.\n{it['question']}" for i, it in enumerate(items))
    prompt = (prompt_prefix + "\n\nFor each item below, give the probability, as an integer percent from 1 to 99, that the answer is YES. "
              f"Return exactly {len(items)} integers in item order.\n\n" + questions)
    cmd = ["claude", "-p", prompt, "--output-format", "json", "--model", model, "--tools", "", "--json-schema", json.dumps(schema),
           "--setting-sources", "project", "--no-session-persistence", "--strict-mcp-config"]
    work = out_dir / "reader_work"
    work.mkdir(exist_ok=True)
    result = subprocess.run(cmd, cwd=work, capture_output=True, text=True, check=False)
    (out_dir / f"reader_{label}.json").write_text(result.stdout)
    reply = json.loads(result.stdout)
    percents = (reply.get("structured_output") or {}).get("percents")
    if not percents or len(percents) != len(items):
        raise RuntimeError(f"reader {label}: {result.stdout[:1000]}")
    return [p / 100 for p in percents]


def log2_score(p, y):
    return math.log2(p if y else 1 - p)


PLAIN_FIELDS = ("rule", "information_used", "mechanism", "predicted_effects")


def plain(report):
    return "\n\n".join(f"{k.replace('_', ' ').upper()}\n{report[k]}" for k in PLAIN_FIELDS) + \
        f"\n\nPROPOSED EDIT, EXPECTED EFFECT\n{report['edit']['expected']}"


def ablated_report(report, out_dir):
    schema = {"type": "object", "additionalProperties": False, "required": list(PLAIN_FIELDS) + ["edit_expected"],
              "properties": {k: {"type": "string"} for k in list(PLAIN_FIELDS) + ["edit_expected"]}}
    prompt = ("Below is a report explaining a rule a language model follows. Rewrite it so that it keeps every concept, token example, "
              "model component (layer, head, neuron) and quantity it names, in the same style and length, but states a WRONG relationship "
              "between them: for example swap which information the rule reads and which it ignores, invert or alter the condition, or swap "
              "the roles of the components. A reader should find it as plausible as the original. Return the rewritten fields.\n\n" + plain(report))
    cmd = ["claude", "-p", prompt, "--output-format", "json", "--model", READER_MODEL, "--tools", "", "--json-schema", json.dumps(schema),
           "--setting-sources", "project", "--no-session-persistence", "--strict-mcp-config"]
    work = out_dir / "reader_work"
    work.mkdir(exist_ok=True)
    result = subprocess.run(cmd, cwd=work, capture_output=True, text=True, check=False)
    fields = json.loads(result.stdout).get("structured_output")
    if not fields:
        raise RuntimeError(f"ablated report: {result.stdout[:1000]}")
    ablated = dict(report)
    for k in PLAIN_FIELDS:
        ablated[k] = fields[k]
    ablated["edit"] = {"edits": [], "expected": fields["edit_expected"]}
    (out_dir / "ablated_report.json").write_text(json.dumps(ablated, indent=1))
    return ablated


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--items", type=int, default=24, help="firing and non-firing items each")
    p.add_argument("--rows", type=int, default=24, help="test rows scanned for positions")
    p.add_argument("--binary", default=str(investigate.GAM / "target/release/examples/mpd_oracle_2951"))
    p.add_argument("--no-readers", action="store_true")
    p.add_argument("--edits-per-item", type=int, default=8)
    p.add_argument("--reader-items", type=int, default=40)
    a = p.parse_args()
    run = Path(a.run)
    frozen = json.load(open(run / "report.frozen.json"))
    report = frozen["report"]
    digest = hashlib.sha256(json.dumps(report, indent=1, sort_keys=True).encode()).hexdigest()
    if digest != frozen["sha256"]:
        raise SystemExit("the report changed after it was frozen")
    task = json.load(open(run / "task.json"))
    out = run / "evaluation"
    out.mkdir(exist_ok=True)
    seed = int.from_bytes(os.urandom(8), "little")
    rng = np.random.default_rng(seed)
    tok = tokenizers.Tokenizer.from_file(os.path.expanduser(task["tokenizer"]))
    piece = lambda t: tok.decode([int(t)])
    proc, sock = investigate.start_server(task, out, a.binary)
    oracle = Client(sock)
    subject = task["subject"]
    reference = task["reference"]
    results = {"report_sha256": digest, "test_seed": seed, "tests_drawn_at": time.time(), "frozen_at": frozen["frozen_at"]}
    try:
        info = oracle({"op": "info"})["models"][subject]
        # 1. Test positions from rows the investigator never read.
        corpus = np.load(os.path.expanduser(task["corpus"]), mmap_mode="r")
        lo, hi = task["test_rows"]
        rows = rng.choice(np.arange(lo, hi), size=a.rows, replace=False)
        context = task.get("test_context", 128)
        keep = reference.get("keep", 0)
        firing, quiet = [], []
        for r in rows:
            tokens = [int(t) for t in corpus[r, :context]]
            full = oracle({"op": "run", "model": subject, "sequences": [tokens], "positions": list(range(keep, context)), "top": 1})
            full_top = [top1(e) for e in full["runs"][0]["positions"]]
            if reference["kind"] == "context":
                windows = [tokens[q + 1 - keep: q + 1] for q in range(keep, context)]
                ref_top = [top1(e) for e in last_runs(oracle, subject, windows)]
            else:
                ref = oracle({"op": "run", "model": reference["name"], "sequences": [tokens], "positions": list(range(keep, context)), "top": 1})
                ref_top = [top1(e) for e in ref["runs"][0]["positions"]]
            for i, q in enumerate(range(keep, context)):
                item = {"row": int(r), "position": q, "tokens": tokens[: q + 1], "rule_token": full_top[i], "reference_token": ref_top[i]}
                (firing if full_top[i] != ref_top[i] else quiet).append(item)
        results["positions"] = {"firing": len(firing), "quiet": len(quiet)}
        pick = lambda xs, n: [xs[i] for i in rng.choice(len(xs), size=min(n, len(xs)), replace=False)]
        firing, quiet = pick(firing, a.items), pick(quiet, a.items)
        (out / "items.json").write_text(json.dumps({"firing": firing, "quiet": quiet}))

        # 2. The executable hypothesis on the fresh positions.
        fires, effect, errors = load_functions(report)
        pieces = lambda ts: [piece(t) for t in ts]
        said = [bool(call(fires, pieces(it["tokens"]))) if fires else False for it in firing + quiet]
        truth = [True] * len(firing) + [False] * len(quiet)
        tp = sum(s and t for s, t in zip(said, truth))
        tn = sum((not s) and (not t) for s, t in zip(said, truth))
        effects = [call(effect, pieces(it["tokens"])) if effect else None for it in firing]
        effect_right = sum(e is not None and e == piece(it["rule_token"]) for e, it in zip(effects, firing))
        results["hypothesis"] = {"fires_accuracy": (tp + tn) / len(truth), "fires_true_positive_rate": tp / max(1, len(firing)),
                                 "fires_true_negative_rate": tn / max(1, len(quiet)), "effect_accuracy_on_firing": effect_right / max(1, len(firing)),
                                 "code_errors": errors}

        # 3. The report's components removed together, against random head sets of the same size.
        def removal(edits, items):
            entries = last_runs(oracle, subject, [it["tokens"] for it in items], {"edits": edits, "patches": []})
            return sum(top1(e) != it["rule_token"] for e, it in zip(entries, items)) / max(1, len(items))

        def collateral(edits, items):
            entries = last_runs(oracle, subject, [it["tokens"] for it in items], {"edits": edits, "patches": []})
            return float(np.mean([e["kl_from_clean_nats"] for e in entries]))

        components = [c["component"] for c in report["components"]]
        ablate = [{"component": c, "alpha": 0.0} for c in components]
        comp = {"count": len(components)}
        if components:
            comp["firing_changed"] = removal(ablate, firing)
            comp["quiet_kl_nats"] = collateral(ablate, quiet)
            heads = [(l, h) for l in range(info["layers"]) for h in range(info["heads"])]
            random_sets = []
            for _ in range(3):
                chosen = rng.choice(len(heads), size=min(len(components), len(heads)), replace=False)
                edits = [{"component": {"kind": "head", "layer": heads[i][0], "head": heads[i][1]}, "alpha": 0.0} for i in chosen]
                random_sets.append({"heads": [heads[i] for i in chosen], "firing_changed": removal(edits, firing), "quiet_kl_nats": collateral(edits, quiet)})
            comp["random_head_sets"] = random_sets
        results["components"] = comp

        # 4. Crossed: does removing the components remove the context's effect on the rule token?
        if reference["kind"] == "context" and components:
            pairs = [{"x0": it["tokens"][-keep:], "x1": it["tokens"], "target": it["rule_token"]} for it in firing]
            crossed = oracle({"op": "crossed", "model": subject, "pairs": pairs, "a0": {}, "a1": {"edits": ablate}})
            effect_mean, gamma_mean = crossed["input_effect_a0"]["mean"], crossed["gamma"]["mean"]
            results["crossed"] = {"input_effect_nats": crossed["input_effect_a0"], "gamma_nats": crossed["gamma"],
                                  "share_carried": -gamma_mean / effect_mean if effect_mean else None}

        # 5. The edit: rule removal, held-out loss and KL.
        edits = report["edit"]["edits"]
        if edits:
            held = [[int(t) for t in corpus[r, :context]] for r in rng.choice(np.arange(lo, hi), size=4, replace=False)]
            reply = oracle({"op": "run", "model": subject, "sequences": held, "positions": list(range(context - 1)), "top": 1,
                            "intervention": {"edits": edits, "patches": []}, "clean": True, "next": True})
            entries = [e for run_ in reply["runs"] for e in run_["positions"]]
            results["edit"] = {"firing_changed": removal(edits, firing), "quiet_kl_nats": collateral(edits, quiet),
                               "heldout_kl_nats_per_token": float(np.mean([e["kl_from_clean_nats"] for e in entries])),
                               "heldout_cross_entropy_change_nats_per_token": float(np.mean([e["next"][2] - e["next"][1] for e in entries]))}

        # 6. Readers on intervention items. Per firing item, every head and MLP set to zero and
        # `edits_per_item` random earlier tokens replaced are measured; the items are then drawn half
        # with Y = 1 and half with Y = 0 (strata of the measured outcome, which no reader sees).
        sites = [(l, "head", h) for l in range(info["layers"]) for h in range(info["heads"])] + [(l, "mlp") for l in range(info["layers"])]
        candidates = []
        for k, it in enumerate(firing):
            tokens = list(it["tokens"])
            for _ in range(a.edits_per_item):
                q = int(rng.integers(0, len(tokens) - 1))
                new = int(corpus[rng.integers(lo, hi), rng.integers(0, context)])
                edited = tokens[:q] + [new] + tokens[q + 1:]
                candidates.append({"item": k, "kind": "input", "tokens": edited, "rule_token": it["rule_token"], "position_changed": q,
                                   "describe": f"the token at position {q} ({piece(tokens[q])!r}) is replaced by {piece(new)!r}", "intervention": None})
            for s_ in sites:
                if s_[1] == "head":
                    patch = {"site": {"kind": "head", "layer": s_[0], "head": s_[2]}, "value": {"kind": "zero"}}
                    describe = f"the output of attention head {s_[2]} in layer {s_[0]} is set to zero at every position"
                else:
                    patch = {"site": {"kind": "mlp", "layer": s_[0]}, "value": {"kind": "zero"}}
                    describe = f"the output of the MLP in layer {s_[0]} is set to zero at every position"
                candidates.append({"item": k, "kind": "site", "tokens": tokens, "rule_token": it["rule_token"], "site": list(s_),
                                   "describe": describe, "intervention": {"edits": [], "patches": [patch]}})
        inputs = [c for c in candidates if c["kind"] == "input"]
        for c, e in zip(inputs, last_runs(oracle, subject, [c["tokens"] for c in inputs])):
            c["y"] = top1(e) == c["rule_token"]
        for c in candidates:
            if c["kind"] == "site":
                c["y"] = top1(last_runs(oracle, subject, [c["tokens"]], c["intervention"])[0]) == c["rule_token"]
        results["candidates"] = {"count": len(candidates), "fraction_yes_inputs": float(np.mean([c["y"] for c in inputs])),
                                 "fraction_yes_sites": float(np.mean([c["y"] for c in candidates if c["kind"] == "site"]))}
        yes = [c for c in candidates if c["y"]]
        no = [c for c in candidates if not c["y"]]
        half = min(a.reader_items // 2, len(yes), len(no))
        items = pick(yes, half) + pick(no, half)
        items = [items[i] for i in rng.permutation(len(items))]
        for it in items:
            text = tok.decode(it["tokens"])
            it["question"] = (f"The model reads this text and predicts the next token:\n<<<{text}>>>\n"
                              f"Without intervention its most likely next token is {piece(it['rule_token'])!r}. "
                              f"Intervention: {it['describe']}. Is {piece(it['rule_token'])!r} still the most likely next token? (positions count tokens from 0)")
        # The executable hypothesis on the same items.
        named = {site_of(c) for c in components} - {None}
        named_heads = {(s[0], "head", s[2]) for s in named if s[1] == "head"}
        named_kv = {(s[0], s[2]) for s in named if s[1] == "kv"}
        exe = []
        for it in items:
            if it["kind"] == "input":
                predicted = call(effect, pieces(it["tokens"])) if effect else None
                exe.append(predicted == piece(it["rule_token"]))
            else:
                s = tuple(it["site"])
                hit = s in named or s in named_heads or (s[1] == "head" and (s[0], s[2] // max(1, info["heads"] // info["kv_heads"])) in named_kv)
                exe.append(not hit)
        ys = [it["y"] for it in items]
        results["items"] = {"count": len(items), "fraction_yes": float(np.mean(ys)), "executable_accuracy": float(np.mean([e == y for e, y in zip(exe, ys)]))}
        if not a.no_readers:
            header = "You predict how a neural language model behaves under interventions."
            with_report = reader(header + " An investigator wrote this report about the model:\n\n" + plain(report), items, "report", out)
            without = reader(header + " You know nothing about this model beyond general knowledge of language models.", items, "baseline", out)
            ablated = ablated_report(report, out)
            with_ablated = reader(header + " An investigator wrote this report about the model:\n\n" + plain(ablated), items, "ablated", out)
            score = lambda ps: float(np.mean([log2_score(p_, y) for p_, y in zip(ps, ys)]))
            results["readers"] = {"log2_score_report": score(with_report), "log2_score_baseline": score(without),
                                  "log2_score_ablated": score(with_ablated),
                                  "information_bits_per_item": score(with_report) - score(without),
                                  "ablated_minus_report_bits": score(with_ablated) - score(with_report)}
        (out / "items_scored.json").write_text(json.dumps(items))
    finally:
        proc.terminate()
    (out / "results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps({k: v for k, v in results.items() if k not in ("test_seed",)}, indent=1))


if __name__ == "__main__":
    main()
