#!/usr/bin/env python3
"""The investigator's command line over the measured-intervention server (#2951).

Every number it prints is measured by the Rust server (crates/gam-mpd/src/oracle.rs) on the native
models; this file only turns text into token ids, sends one request, and prints the answer in
words. The session (socket, tokenizer, models, corpus) is the JSON file named by ORACLE_SESSION.

Interventions are JSON in the server's format (see `oracle.py help-interventions`); anywhere a
"tokens" list is expected, {"text": "..."} may be given instead.
"""

import argparse
import json
import os
import socket
import sys

import numpy as np
import tokenizers

HELP_INTERVENTIONS = """An intervention is {"edits": [...], "patches": [...]}.

Edit (native parameter edit W(alpha) = W + (alpha - 1) P, alpha = 1 is the model itself):
  {"component": C, "alpha": a}, C one of
    {"kind": "operator", "name": "blocks.2.c_fc"}            the whole stored matrix
    {"kind": "head", "layer": 1, "head": 4}                    head's output map o (alpha scales its write)
    {"kind": "neuron", "layer": 0, "index": 17}                MLP neuron's down column (its write)
    {"kind": "rows", "name": "blocks.0.c_fc", "rows": [3, 9]}  chosen rows (neurons' reads in c_fc/gate_proj)
    {"kind": "columns", "name": "blocks.0.down_proj", "columns": [3]}
    {"kind": "difference", "name": "blocks.1.v2", "reference": "base", "components": [0]}
        P = W - W_reference (or its chosen singular components); alpha = 0 sets them to the reference's
    {"kind": "direction", "name": "blocks.1.q4", "side": "input", "direction": [...]}
        the part of W reading (input: W d d^T) or writing (output: d d^T W) a direction
  Operator names: blocks.L.qH, blocks.L.kG, blocks.L.vG, blocks.L.oH, blocks.L.c_fc, blocks.L.gate_proj
  (gated MLPs), blocks.L.down_proj. q/k/v rows read the normed residual; o and down_proj write it.

Patch (activation patching at a site):
  {"site": S, "value": V, "positions": [-1], "sequences": [0], "coordinates": [..] | "direction": [..]}
    S: {"kind": "stream"|"middle"|"attention"|"mlp"|"neurons", "layer": l} or {"kind": "head", "layer": l, "head": h}
       stream l = residual entering layer l (l = L: final residual); middle l = after layer l's attention;
       attention/mlp = that block's output; head = the head's attention read z_h; neurons = MLP activations.
    V: {"kind": "source", "text": "...", "model": "base", "positions": [..]}  value from another run
         (default: the same model under the same edits on the same sequence; "model" alone = other model, same text)
       {"kind": "zero"} | {"kind": "scale", "factor": f} | {"kind": "mean", "sequences": [{"text": ...}, ...]}
       | {"kind": "add", "vector": [...]}
    positions: default every position; negative counts from the end.
"""


class Session:
    def __init__(self, path=None):
        path = path or os.environ.get("ORACLE_SESSION")
        if not path:
            raise SystemExit("set ORACLE_SESSION to the session file")
        self.config = json.load(open(path))
        self.tokenizer = tokenizers.Tokenizer.from_file(self.config["tokenizer"])
        self.calls = 0
        self.template = None
        if self.config.get("chat_template"):
            from jinja2.sandbox import ImmutableSandboxedEnvironment

            env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
            env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
            env.filters["tojson"] = lambda x, **k: json.dumps(x, **k)
            self.template = env.from_string(open(self.config["chat_template"]).read())
            self.end_of_turn = self.tokenizer.token_to_id("<|im_end|>")

    def request(self, payload):
        self.calls += 1
        log = self.config.get("log")
        limit = self.config.get("max_calls")
        if log and limit is not None and os.path.exists(log) and sum(1 for _ in open(log)) >= limit:
            raise SystemExit(f"the investigation's budget of {limit} oracle calls is spent; write the report now")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(self.config["socket"])
            s.sendall((json.dumps(payload) + "\n").encode())
            chunks = []
            while not chunks or not chunks[-1].endswith(b"\n"):
                b = s.recv(1 << 20)
                if not b:
                    break
                chunks.append(b)
        reply = json.loads(b"".join(chunks))
        if log:
            with open(log, "a") as f:
                f.write(json.dumps({"request": payload["op"], "seconds": reply.get("seconds")}) + "\n")
        if "error" in reply:
            raise SystemExit("server: " + reply["error"])
        return reply["ok"]

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def chat(self, messages):
        """The prompt's token ids: the chat template with a generation prompt and thinking disabled."""
        if self.template is None:
            raise SystemExit("this session's models have no chat template")
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        text = self.template.render(messages=messages, add_generation_prompt=True, enable_thinking=False, bos_token="", eos_token="<|im_end|>")
        return self.encode(text)

    def option_item(self, item):
        """{"messages" | "user", "options": [str]} -> the server's {"prompt", "options"} (each option ends its turn)."""
        prompt = self.chat(item.get("messages") or item["user"])
        return {"prompt": prompt, "options": [self.encode(o) + [self.end_of_turn] for o in item["options"]]}

    def piece(self, token):
        return repr(self.tokenizer.decode([int(token)]))

    def tokens_in(self, obj):
        """Replace every {"text": s} where token lists are expected by its token ids."""
        if isinstance(obj, list):
            return [self.tokens_in(x) for x in obj]
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k == "text":
                    out["tokens"] = self.encode(v)
                elif k == "sequences" and isinstance(v, list) and v and isinstance(v[0], (str, dict)):
                    out[k] = [self.encode(x) if isinstance(x, str) else self.tokens_in(x)["tokens"] for x in v]
                else:
                    out[k] = self.tokens_in(v)
            return out
        return obj

    def corpus(self, start, count, context):
        path = self.config["corpus"]
        if path.endswith(".u32"):
            rows = np.memmap(path, dtype="<u4", mode="r").reshape(-1, 128)
        else:
            rows = np.load(path, mmap_mode="r")
        lo, hi = self.config.get("corpus_rows", [0, rows.shape[0]])
        if start < lo or start + count > hi:
            raise SystemExit(f"corpus rows {start} to {start + count} are outside this investigation's rows {lo} to {hi}")
        return [[int(t) for t in rows[i, :context]] for i in range(start, start + count)]


def parse_json(text):
    if text is None:
        return None
    if text.startswith("@"):
        return json.load(open(text[1:]))
    return json.loads(text)


def show_top(sess, pairs):
    return ", ".join(f"{sess.piece(t)} {lp:.2f}" for t, lp in pairs)


def cmd_run(sess, a):
    texts = a.text
    sequences = [sess.encode(t) for t in texts]
    targets = [sess.encode(t)[0] for t in (a.targets or [])]
    intervention = sess.tokens_in(parse_json(a.intervention) or {})
    positions = [int(p) for p in a.positions.split(",")] if a.positions else None
    record = parse_json(a.record) or []
    out = sess.request({"op": "run", "model": a.model, "sequences": sequences, "positions": positions, "top": a.top,
                        "targets": targets, "intervention": intervention, "clean": a.clean or bool(intervention), "record": record})
    for run, toks in zip(out["runs"], sequences):
        print(f"sequence {run['sequence']} ({len(toks)} tokens)")
        for e in run["positions"]:
            p = e["position"]
            print(f"  after token {p} {sess.piece(toks[p])}: top next tokens (log-prob, nats): {show_top(sess, e['top'])}")
            if "clean_top" in e:
                print(f"    clean model: {show_top(sess, e['clean_top'])}; KL(clean || intervened) = {e['kl_from_clean_nats']:.4f} nats")
            for (t, lp), *rest in zip(e["targets"], *([e["clean_targets"]] if "clean_targets" in e else [])):
                extra = f" (clean {rest[0][1]:.3f})" if rest else ""
                print(f"    target {sess.piece(t)}: log-prob {lp:.3f}{extra}")
            for r in e.get("record", []):
                extra = f"; largest coordinates {', '.join(f'{i}:{v:.3f}' for i, v in r['largest'])}" if "largest" in r else ""
                print(f"    {r['site']}: norm {r['norm']:.3f}{extra}")


def cmd_compare(sess, a):
    toks = sess.encode(a.text)
    models = a.models.split(",") if a.models else sorted(sess.config["models"])
    import math
    dists = {}
    for m in models:
        out = sess.request({"op": "run", "model": m, "sequences": [toks], "top": a.top})
        e = out["runs"][0]["positions"][0]
        dists[m] = e["top"]
        print(f"{m}: {show_top(sess, e['top'])}")
    if len(models) == 2:
        scan = sess.request({"op": "scan", "model": models[1], "reference": models[0], "sequences": [toks], "top": 1})
        last = [x for x in scan["largest"]]
        print(f"KL({models[1]} || {models[0]}) mean over the {len(toks)} positions {scan['mean_kl_nats_per_token']:.4f} nats; largest at position {last[0]['position']} = {last[0]['kl_nats']:.4f}")


def cmd_generate(sess, a):
    toks = sess.encode(a.text)
    models = a.model.split(",")
    for m in models:
        out = sess.request({"op": "generate", "model": m, "tokens": toks, "steps": a.steps, "edits": sess.tokens_in(parse_json(a.edits) or [])})
        text = sess.tokenizer.decode([g["token"] for g in out["generated"]])
        print(f"{m}: {a.text!r} -> {text!r}")


def cmd_attention(sess, a):
    toks = sess.encode(a.text)
    out = sess.request({"op": "attention", "model": a.model, "tokens": toks, "layer": a.layer, "head": a.head, "top": a.top,
                        "edits": parse_json(a.edits) or []})
    for r in out["rows"]:
        q = r["query"]
        srcs = ", ".join(f"{s}:{sess.piece(toks[s])} {w:.2f}" for s, w in r["sources"])
        print(f"  {q}:{sess.piece(toks[q])} attends to {srcs}")
    print(f"mean weight on the previous token {out['mean_previous_token_weight']:.3f}, on the first token {out['mean_first_token_weight']:.3f}")


def cmd_crossed(sess, a):
    raw = parse_json(a.pairs)
    pairs = []
    for p in raw:
        pair = {"x0": sess.encode(p["x0"]), "x1": sess.encode(p["x1"]), "target": sess.encode(p["target"])[0]}
        if p.get("versus"):
            pair["versus"] = sess.encode(p["versus"])[0]
        if p.get("positions"):
            pair["positions"] = p["positions"]
        pairs.append(pair)
    out = sess.request({"op": "crossed", "model": a.model, "pairs": pairs, "a0": sess.tokens_in(parse_json(a.a0) or {}),
                        "a1": sess.tokens_in(parse_json(a.a1))})
    for p, r in zip(raw, out["pairs"]):
        print(f"  x0={p['x0'][-40:]!r} x1={p['x1'][-40:]!r}: input effect {r['input_effect_a0']:+.3f} under a0, "
              f"{r['input_effect_a1']:+.3f} under a1, gamma {r['gamma']:+.3f}")
    g, e0, e1 = out["gamma"], out["input_effect_a0"], out["input_effect_a1"]
    print(f"mean input effect under a0 {e0['mean']:+.3f} +- {e0['standard_error']:.3f}; under a1 {e1['mean']:+.3f} +- {e1['standard_error']:.3f}; "
          f"crossed effect gamma {g['mean']:+.3f} +- {g['standard_error']:.3f} nats over {g['count']} pairs")


def cmd_diff(sess, a):
    out = sess.request({"op": "difference", "model": a.model, "reference": a.reference, "spectrum": a.spectrum, "top": a.top})
    print(f"{out['changed_operators']} operators differ; largest relative changes:")
    for r in out["operators"][: a.rows]:
        line = f"  {r['operator']}: |dW| {r['difference_norm']:.4g} = {r['relative']:.3g} of |W|"
        if "stable_rank" in r:
            line += f", stable rank {r['stable_rank']:.2f}, top singular values {', '.join(f'{s:.3g}' for s in r['singular_values'][:4])}"
        if "largest_neurons" in r:
            line += f", largest neurons {', '.join(f'{i}:{v:.3g}' for i, v in r['largest_neurons'][:5])}"
        print(line)


def cmd_components(sess, a):
    out = sess.request({"op": "components", "model": a.model, "reference": a.reference, "name": a.name, "count": a.count, "top": a.top})
    print(f"{out['operator']}: leading singular components of the weight difference ({out['token_readings']})")
    for c in out["components"]:
        print(f"  component {c['index']}: singular value {c['singular_value']:.4g}, {100 * c['share_of_squared_norm']:.1f}% of the squared difference")
        for key in ("output_direction_promotes", "output_direction_suppresses", "input_direction_reads_tokens", "input_direction_reads_negatively"):
            if key in c:
                print(f"    {key.replace('_', ' ')}: {', '.join(f'{sess.piece(t)} {v:.2f}' for t, v in c[key])}")
        for key in ("input_coordinates", "output_coordinates"):
            if key in c:
                print(f"    largest {key.replace('_', ' ')}: {', '.join(f'{i}:{v:.3f}' for i, v in c[key][:8])}")


def cmd_localize(sess, a):
    sequences = [sess.encode(t) for t in a.text]
    targets = [sess.encode(t)[0] for t in a.targets] if a.targets else None
    scope = {"heads": a.heads, "layers": [int(x) for x in a.layers.split(",")] if a.layers else None}
    out = sess.request({"op": "localize", "model": a.model, "reference": a.reference, "sequences": sequences, "targets": targets, "weights": a.weights,
                        "scope": scope})
    print(f"targets {[sess.piece(t) for t in out['targets']]}: log-prob under {a.model} {[round(x, 3) for x in out['model_log_probability']]}, "
          f"under {a.reference} {[round(x, 3) for x in out['reference_log_probability']]}; mean gap {out['mean_gap_nats']:.3f} nats")
    rows = sorted(out["swaps"], key=lambda r: -abs(r["reference_gains_nats"]) - abs(r["model_loses_nats"]))
    print(f"single swaps, largest first (gains: {a.reference} given this site from {a.model}; loses: {a.model} given this site from {a.reference}):")
    for r in rows[: a.rows]:
        print(f"  [{r['kind']}] {r['site']}: {a.reference} gains {r['reference_gains_nats']:+.3f}, {a.model} loses {r['model_loses_nats']:+.3f} nats")


def cmd_scan(sess, a):
    if a.texts:
        sequences = [sess.encode(t) for t in parse_json(a.texts)]
    else:
        sequences = sess.corpus(a.start, a.rows, a.context)
    out = sess.request({"op": "scan", "model": a.model, "reference": a.reference, "sequences": sequences, "top": a.top})
    print(f"mean KL({a.model} || {a.reference}) {out['mean_kl_nats_per_token']:.5f} nats per token over {out['tokens']} tokens; largest positions:")
    for r in out["largest"]:
        toks = sequences[r["sequence"]]
        p = r["position"]
        context = sess.tokenizer.decode(toks[max(0, p - a.window): p + 1])
        print(f"  KL {r['kl_nats']:.3f} at sequence {r['sequence']} position {p}: ...{context!r}")
        print(f"      {a.model}: {show_top(sess, r['model_top'])} | {a.reference}: {show_top(sess, r['reference_top'])}")


def cmd_context_scan(sess, a):
    if a.texts:
        sequences = [sess.encode(t) for t in parse_json(a.texts)]
    else:
        sequences = sess.corpus(a.start, a.rows, a.context)
    out = sess.request({"op": "context_scan", "model": a.model, "sequences": sequences, "keep": a.keep, "top": a.top})
    print(f"mean KL(full context || last {a.keep} tokens only) {out['mean_kl_nats_per_token']:.4f} nats per token over {out['tokens']} tokens; largest positions:")
    for r in out["largest"]:
        toks = sequences[r["sequence"]]
        p = r["position"]
        context = sess.tokenizer.decode(toks[max(0, p - a.window): p + 1])
        print(f"  KL {r['kl_nats']:.3f} at sequence {r['sequence']} position {p}: ...{context!r}")
        print(f"      full context: {show_top(sess, r['full_top'])} | last {a.keep} only: {show_top(sess, r['truncated_top'])}")


def cmd_activations(sess, a):
    if a.texts:
        sequences = [sess.encode(t) for t in parse_json(a.texts)]
    else:
        sequences = sess.corpus(a.start, a.rows, a.context)
    payload = {"op": "activations", "model": a.model, "site": parse_json(a.site), "sequences": sequences, "top": a.top}
    if a.coordinate is not None:
        payload["coordinate"] = a.coordinate
    if a.direction:
        payload["direction"] = parse_json(a.direction)
    out = sess.request(payload)
    what = f"coordinate {a.coordinate}" if a.coordinate is not None else "the direction" if a.direction else "the norm"
    print(f"{what} at {a.site}: mean {out['mean']:.4f}, standard deviation {out['standard_deviation']:.4f} over {out['tokens']} tokens; largest:")
    for r in out["largest"]:
        toks = sequences[r["sequence"]]
        p = r["position"]
        print(f"  {r['value']:.4f} at sequence {r['sequence']} position {p}: ...{sess.tokenizer.decode(toks[max(0, p - a.window): p])!r} [[{sess.tokenizer.decode([toks[p]])}]]")


def cmd_options(sess, a):
    items = parse_json(a.items)
    models = a.model.split(",")
    server_items = [sess.option_item(it) for it in items]
    intervention = sess.tokens_in(parse_json(a.intervention) or {})
    choices = {}
    for m in models:
        out = sess.request({"op": "options", "model": m, "items": server_items, "intervention": intervention})
        choices[m] = out["items"]
    for i, it in enumerate(items):
        user = (it.get("user") or it["messages"][-1]["content"])
        print(f"item {i}: {user[:160]!r}")
        for m in models:
            r = choices[m][i]
            lps = ", ".join(f"{o!r} {lp:.2f}" for o, lp in zip(it["options"], r["log_probabilities"]))
            print(f"    {m}: chooses {it['options'][r['choice']]!r}  ({lps})")
        if len(models) == 2 and choices[models[0]][i]["choice"] != choices[models[1]][i]["choice"]:
            print("    -> the models choose differently")


def cmd_chat(sess, a):
    prompt = sess.chat(a.text)
    for m in a.model.split(","):
        out = sess.request({"op": "generate", "model": m, "tokens": prompt, "steps": a.steps, "stop": sess.end_of_turn,
                            "edits": sess.tokens_in(parse_json(a.edits) or [])})
        print(f"{m}: {sess.tokenizer.decode([g['token'] for g in out['generated']])!r}")


def cmd_crossed_options(sess, a):
    raw = parse_json(a.pairs)
    pairs = [{"x0": sess.option_item(p["x0"]), "x1": sess.option_item(p["x1"]), "choice": p["choice"], "versus": p["versus"]} for p in raw]
    out = sess.request({"op": "crossed_options", "model": a.model, "pairs": pairs, "a0": sess.tokens_in(parse_json(a.a0) or {}),
                        "a1": sess.tokens_in(parse_json(a.a1))})
    for p, r in zip(raw, out["pairs"]):
        x0 = p["x0"].get("user") or p["x0"]["messages"][-1]["content"]
        x1 = p["x1"].get("user") or p["x1"]["messages"][-1]["content"]
        print(f"  x0={x0[-50:]!r} x1={x1[-50:]!r}: margin x0 {r['y_x0_a0']:+.2f} -> {r['y_x0_a1']:+.2f}, x1 {r['y_x1_a0']:+.2f} -> {r['y_x1_a1']:+.2f}, gamma {r['gamma']:+.2f}")
    g, e0, e1 = out["gamma"], out["input_effect_a0"], out["input_effect_a1"]
    print(f"margin = log-prob(option choice) - log-prob(option versus). Input effect under a0 {e0['mean']:+.3f} +- {e0['standard_error']:.3f}; "
          f"under a1 {e1['mean']:+.3f} +- {e1['standard_error']:.3f}; gamma {g['mean']:+.3f} +- {g['standard_error']:.3f} nats over {g['count']} pairs")


def cmd_localize_options(sess, a):
    items = parse_json(a.items)
    scope = {"heads": a.heads, "layers": [int(x) for x in a.layers.split(",")] if a.layers else None}
    out = sess.request({"op": "localize_options", "model": a.model, "reference": a.reference, "items": [sess.option_item(it) for it in items],
                        "weights": a.weights, "scope": scope})
    print(f"{out['differing_items']} of {len(items)} items are chosen differently by {a.model} and {a.reference}")
    if not out["differing_items"]:
        return
    print(f"mean margin of {a.model}'s choice over {a.reference}'s: {out['mean_margin_model_nats']:+.3f} under {a.model}, {out['mean_margin_reference_nats']:+.3f} under {a.reference}")
    rows = sorted(out["swaps"], key=lambda r: -abs(r["reference_gains_nats"]) - abs(r["model_loses_nats"]))
    print(f"single swaps, largest first (gains: {a.reference} given this site from {a.model}; loses: {a.model} given this site from {a.reference}; flips: items whose choice switches):")
    for r in rows[: a.rows]:
        print(f"  [{r['kind']}] {r['site']}: {a.reference} gains {r['reference_gains_nats']:+.3f} ({r['reference_flips']} flips), "
              f"{a.model} loses {r['model_loses_nats']:+.3f} ({r['model_flips']} flips)")


def cmd_unembed(sess, a):
    payload = {"op": "unembed", "model": a.model, "top": a.top}
    if a.vector:
        payload["vector"] = parse_json(a.vector)
    else:
        payload.update({"site": parse_json(a.site), "tokens": sess.encode(a.text), "position": a.position})
    out = sess.request(payload)
    print(f"write norm {out['norm']:.3f} ({out['path']})")
    print(f"  promotes: {', '.join(f'{sess.piece(t)} {v:.2f}' for t, v in out['promoted'])}")
    print(f"  suppresses: {', '.join(f'{sess.piece(t)} {v:.2f}' for t, v in out['suppressed'])}")


def cmd_tokens(sess, a):
    ids = sess.encode(a.text)
    print(" ".join(f"{i}:{t}:{sess.piece(t)}" for i, t in enumerate(ids)))


def cmd_info(sess, a):
    out = sess.request({"op": "info"})
    for name, m in out["models"].items():
        ops = m.pop("operators")
        print(f"{name}: {json.dumps(m)}; {len(ops)} block operators, e.g. {', '.join(ops[:6])}")


def cmd_raw(sess, a):
    print(json.dumps(sess.request(sess.tokens_in(parse_json(a.request))), indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(prog="oracle", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("info"); s.set_defaults(f=cmd_info)
    s = sub.add_parser("help-interventions"); s.set_defaults(f=lambda sess, a: print(HELP_INTERVENTIONS))
    s = sub.add_parser("tokens"); s.add_argument("text"); s.set_defaults(f=cmd_tokens)
    s = sub.add_parser("run", help="next-token distributions under an intervention")
    s.add_argument("model"); s.add_argument("text", nargs="+"); s.add_argument("--positions")
    s.add_argument("--targets", nargs="*"); s.add_argument("--top", type=int, default=8)
    s.add_argument("--intervention", help="JSON or @file"); s.add_argument("--clean", action="store_true")
    s.add_argument("--record", help='JSON list of sites whose values to report, e.g. [{"kind": "neurons", "layer": 1}]')
    s.set_defaults(f=cmd_run)
    s = sub.add_parser("compare", help="every model's next-token distribution on one text")
    s.add_argument("text"); s.add_argument("--models"); s.add_argument("--top", type=int, default=8); s.set_defaults(f=cmd_compare)
    s = sub.add_parser("generate"); s.add_argument("model", help="model or comma list"); s.add_argument("text")
    s.add_argument("--steps", type=int, default=20); s.add_argument("--edits"); s.set_defaults(f=cmd_generate)
    s = sub.add_parser("attention"); s.add_argument("model"); s.add_argument("text"); s.add_argument("layer", type=int)
    s.add_argument("head", type=int); s.add_argument("--top", type=int, default=3); s.add_argument("--edits"); s.set_defaults(f=cmd_attention)
    s = sub.add_parser("crossed", help="crossed intervention gamma over pairs")
    s.add_argument("model"); s.add_argument("--pairs", required=True, help='JSON or @file: [{"x0": text, "x1": text, "target": text, "versus": text?}]')
    s.add_argument("--a0"); s.add_argument("--a1", required=True); s.set_defaults(f=cmd_crossed)
    s = sub.add_parser("diff", help="weight difference per operator"); s.add_argument("model"); s.add_argument("reference")
    s.add_argument("--spectrum", action="store_true"); s.add_argument("--top", type=int, default=8); s.add_argument("--rows", type=int, default=25)
    s.set_defaults(f=cmd_diff)
    s = sub.add_parser("components", help="singular components of one operator's difference")
    s.add_argument("model"); s.add_argument("reference"); s.add_argument("name"); s.add_argument("--count", type=int, default=3)
    s.add_argument("--top", type=int, default=8); s.set_defaults(f=cmd_components)
    s = sub.add_parser("localize", help="activation and weight swaps between two models")
    s.add_argument("model"); s.add_argument("reference"); s.add_argument("text", nargs="+"); s.add_argument("--targets", nargs="*")
    s.add_argument("--weights", action="store_true"); s.add_argument("--rows", type=int, default=20)
    s.add_argument("--heads", action="store_true", help="also each head (slow: one run per head)"); s.add_argument("--layers", help="comma list of layers")
    s.set_defaults(f=cmd_localize)
    s = sub.add_parser("scan", help="positions where two models disagree most")
    s.add_argument("model"); s.add_argument("reference"); s.add_argument("--start", type=int, default=0); s.add_argument("--rows", type=int, default=8)
    s.add_argument("--context", type=int, default=128); s.add_argument("--texts"); s.add_argument("--top", type=int, default=15)
    s.add_argument("--window", type=int, default=24); s.set_defaults(f=cmd_scan)
    s = sub.add_parser("context-scan", help="positions where context beyond the last KEEP tokens changes the prediction most")
    s.add_argument("model"); s.add_argument("--keep", type=int, default=3); s.add_argument("--start", type=int, default=0)
    s.add_argument("--rows", type=int, default=4); s.add_argument("--context", type=int, default=128); s.add_argument("--texts")
    s.add_argument("--top", type=int, default=15); s.add_argument("--window", type=int, default=40); s.set_defaults(f=cmd_context_scan)
    s = sub.add_parser("activations", help="largest-activating corpus positions of a site's coordinate, direction or norm")
    s.add_argument("model"); s.add_argument("--site", required=True); s.add_argument("--coordinate", type=int); s.add_argument("--direction")
    s.add_argument("--start", type=int, default=0); s.add_argument("--rows", type=int, default=4); s.add_argument("--context", type=int, default=128)
    s.add_argument("--texts"); s.add_argument("--top", type=int, default=15); s.add_argument("--window", type=int, default=30)
    s.set_defaults(f=cmd_activations)
    s = sub.add_parser("unembed", help="direct-path token reading of a site's write")
    s.add_argument("model"); s.add_argument("--site"); s.add_argument("--text"); s.add_argument("--position", type=int, default=-1)
    s.add_argument("--vector"); s.add_argument("--top", type=int, default=10); s.set_defaults(f=cmd_unembed)
    s = sub.add_parser("options", help="chat items: each option's log-probability and the chosen option, per model")
    s.add_argument("model", help="model or comma list"); s.add_argument("--items", required=True, help='JSON or @file: [{"user": text, "options": [text, ...]}]')
    s.add_argument("--intervention"); s.set_defaults(f=cmd_options)
    s = sub.add_parser("chat", help="greedy chat response to one user message"); s.add_argument("model", help="model or comma list")
    s.add_argument("text"); s.add_argument("--steps", type=int, default=40); s.add_argument("--edits"); s.set_defaults(f=cmd_chat)
    s = sub.add_parser("crossed-options", help="crossed intervention gamma on option margins")
    s.add_argument("model"); s.add_argument("--pairs", required=True, help='[{"x0": item, "x1": item, "choice": i, "versus": j}]')
    s.add_argument("--a0"); s.add_argument("--a1", required=True); s.set_defaults(f=cmd_crossed_options)
    s = sub.add_parser("localize-options", help="activation and weight swaps between two models on chat items")
    s.add_argument("model"); s.add_argument("reference"); s.add_argument("--items", required=True); s.add_argument("--weights", action="store_true")
    s.add_argument("--rows", type=int, default=20)
    s.add_argument("--heads", action="store_true", help="also each head (slow: one run per head)"); s.add_argument("--layers", help="comma list of layers")
    s.set_defaults(f=cmd_localize_options)
    s = sub.add_parser("raw", help="send one request in the server's JSON"); s.add_argument("request"); s.set_defaults(f=cmd_raw)
    a = p.parse_args(argv)
    sess = None if a.cmd == "help-interventions" else Session()
    allowed = sess.config.get("allowed") if sess else None
    if allowed is not None:
        clean_only = (a.cmd == "run" and "run" not in allowed and "run-clean" in allowed) or \
            (a.cmd == "options" and "options" not in allowed and "options-clean" in allowed)
        if clean_only and getattr(a, "intervention", None):
            raise SystemExit("interventions are not available in this investigation")
        if a.cmd not in allowed and not clean_only:
            raise SystemExit(f"{a.cmd} is not available in this investigation (available: {', '.join(allowed)})")
    a.f(sess, a)


if __name__ == "__main__":
    main()
