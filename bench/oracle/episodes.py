"""The oracle's episode store (#2951): one JSON file per episode at
~/mpd-data/oracle/episodes/<target>/<episode>.json, schema "mpd.oracle-episode/1".

  target        {id, kind: organism | library_region | known_mechanism, models: {role: {server, path}}
                (roles "updated" and, when there is one, "base"; server = the model's name in the
                experiment server, path = its checkpoint directory), description (public), contexts
                (a JSONL file of public texts or chats the tests read)}
  investigator  {name, model, transcript: chat messages [{role, content, tool_calls?, name?}],
                server: every request and reply in order, requests (a batch counts its members),
                server_seconds}
  report        {content (the frozen JSON object, at least a "rule" string), sha256 (of its canonical
                JSON: sorted keys, no spaces, UTF-8), frozen_at (Unix seconds), frozen_at_utc}
  tests         each drawn after the freeze: {id, report_sha256, entropy (32 random bytes, hex, drawn
                after the freeze), seed (the first 8 bytes of sha256(report_sha256 ":" entropy), big
                endian), drawn_at, family, context {text, tokens}, request (the measuring server Run
                request: model, sequences, targets = the candidates, intervention), intervention_text
                (plain language), options (the candidates' decoded strings), option_tokens, measured
                {log_probabilities (the server's, of each candidate), p (renormalized over the
                candidates)}, replies (the raw server replies: the candidate search and the measure)}
  scores        {key: {reader {backend, model, seed, rotations}, documents (report | transcript | none |
                ablated:<kind>), per_test {test id: {q, log_score}}, mean_log_score_nats}}

Freezing fixes the report's hash; the store refuses to replace a frozen report and refuses a test whose
draw time precedes the freeze or whose seed is not derived from the report's hash and its entropy.

Test families (uniform over those the target and report allow), every arm on the text's every position:
  clean            the updated model as it is
  base             the model before its update (targets with a base)
  zero_head        one attention head's output (the input of its output map) set to zero
  zero_mlp         one layer's MLP output set to zero
  swap_mlp         one layer's MLP output replaced by the base model's on the same text (with a base)
  swap_attention   one layer's attention output replaced likewise (with a base)
  revert_operator  one block operator set back to the base model's (with a base)
  report_edit      the report's own native edit (a report with "edits": a list of server Edit objects)
Candidates. With m = --candidates-per-side, the server returns the 2m most probable next tokens of the
updated model as it is and of the test's arm; the candidates are the first 2m distinct tokens taken
alternately from the two lists by rank, in a seeded random order. The measured p is the arm's
distribution renormalized over them. So a test asks how the arm moves probability among the tokens
either the model or the arm favours.

  episodes.py new --target TARGET.json --investigator NAME --model MODEL [--episode ID] [--root DIR]  (prints the path)
  episodes.py freeze --episode EP.json --report REPORT.json
  episodes.py tests --episode EP.json --server ADDRESS --count N --candidates-per-side M
  episodes.py score --episode EP.json --reader ADDRESS --documents report|transcript|none|ablated:KIND [--ablations FILE]
  episodes.py show --episode EP.json
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import re
import secrets
import time
import uuid
from pathlib import Path

import numpy as np

SCHEMA = "mpd.oracle-episode/1"
ROOT = Path(os.path.expanduser("~/mpd-data/oracle/episodes"))
KINDS = ("organism", "library_region", "known_mechanism")


# ---------------------------------------------------------------------------------------------------
# The store


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def path_of(episode: dict, root: Path = ROOT) -> Path:
    return Path(root) / episode["target"]["id"] / f"{episode['episode']}.json"


def save(episode: dict, path: Path | None = None, root: Path = ROOT) -> Path:
    path = Path(path) if path is not None else path_of(episode, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(episode, indent=1))
    os.replace(tmp, path)
    return path


def load(path) -> dict:
    episode = json.loads(Path(path).read_text())
    if episode.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {episode.get('schema')!r}, not {SCHEMA}")
    return episode


def check_target(target: dict):
    if target.get("kind") not in KINDS:
        raise ValueError(f"target kind {target.get('kind')!r}; one of {KINDS}")
    for field in ("id", "models", "description"):
        if field not in target:
            raise ValueError(f"target needs {field!r}")
    if "updated" not in target["models"]:
        raise ValueError("target models need an 'updated' role (the model under study)")


def new_episode(target: dict, investigator: str, model: str, episode_id: str | None = None) -> dict:
    check_target(target)
    return {
        "schema": SCHEMA,
        "episode": episode_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8],
        "created_at": time.time(),
        "target": target,
        "investigator": {"name": investigator, "model": model, "transcript": [], "server": [], "requests": 0, "server_seconds": 0.0},
        "report": None,
        "tests": [],
        "scores": {},
    }


def record_investigation(episode: dict, transcript: list[dict], client) -> None:
    """The investigator's chat messages and every server request and reply of `client` (a NativeClient)."""
    if episode["report"] is not None:
        raise ValueError("the investigation is closed: the report is frozen")
    episode["investigator"]["transcript"] = transcript
    episode["investigator"]["server"] = client.log
    episode["investigator"]["requests"] = client.request_count()
    episode["investigator"]["server_seconds"] = client.server_seconds()


def freeze(episode: dict, report: dict) -> dict:
    """Fix the report and its hash; a frozen report is never replaced."""
    if episode["report"] is not None:
        raise ValueError(f"episode {episode['episode']} already has a report frozen at {episode['report']['frozen_at_utc']}")
    if not isinstance(report, dict) or not isinstance(report.get("rule"), str) or not report["rule"].strip():
        raise ValueError("a report is a JSON object with a nonempty 'rule' string")
    now = time.time()
    episode["report"] = {
        "content": report,
        "sha256": sha256_hex(canonical(report)),
        "frozen_at": now,
        "frozen_at_utc": datetime.datetime.fromtimestamp(now, datetime.timezone.utc).isoformat(),
    }
    return episode["report"]


def test_seed(report_sha256: str, entropy: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{report_sha256}:{entropy}".encode()).digest()[:8], "big")


def fresh_draw(episode: dict) -> dict:
    """Entropy drawn now, after the freeze, and the seed it gives with the report's hash."""
    report = episode["report"]
    if report is None:
        raise ValueError("tests are drawn only after the report is frozen")
    entropy = secrets.token_hex(32)
    drawn_at = time.time()
    if drawn_at < report["frozen_at"]:
        raise ValueError("the clock reads earlier than the freeze")
    return {"report_sha256": report["sha256"], "entropy": entropy, "seed": test_seed(report["sha256"], entropy), "drawn_at": drawn_at}


def attach_tests(episode: dict, tests: list[dict]) -> None:
    report = episode["report"]
    if report is None:
        raise ValueError("no frozen report")
    if sha256_hex(canonical(report["content"])) != report["sha256"]:
        raise ValueError("the frozen report no longer matches its hash")
    for t in tests:
        if t["drawn_at"] < report["frozen_at"]:
            raise ValueError(f"test {t['id']} was drawn at {t['drawn_at']}, before the freeze at {report['frozen_at']}")
        if t["report_sha256"] != report["sha256"] or t["seed"] != test_seed(report["sha256"], t["entropy"]):
            raise ValueError(f"test {t['id']}: its seed is not derived from the report's hash and its entropy")
    known = {t["id"] for t in episode["tests"]}
    clash = [t["id"] for t in tests if t["id"] in known]
    if clash:
        raise ValueError(f"tests already attached: {clash}")
    episode["tests"].extend(tests)


# ---------------------------------------------------------------------------------------------------
# Plain-language descriptions of oracle.rs's interventions (text only)


def model_phrase(name: str, roles: dict[str, str]) -> str:
    role = roles.get(name)
    if role == "updated":
        return "the model"
    if role == "base":
        return "the model before its update"
    return f"the model {name!r}"


def operator_phrase(name: str, layers: int) -> str:
    m = re.fullmatch(r"blocks\.(\d+)\.(.+)", name)
    if not m:
        return {"wte": "the token embedding"}.get(name, f"the weight matrix {name}")
    layer, rest = int(m.group(1)), m.group(2)
    where = f"layer {layer} of {layers}"
    head = re.fullmatch(r"([qkvo])(\d+)", rest)
    if head:
        kind, index = head.group(1), int(head.group(2))
        if kind == "q":
            return f"the query map of attention head {index} in {where}"
        if kind == "o":
            return f"the output map of attention head {index} in {where}"
        return f"the {'key' if kind == 'k' else 'value'} map of key-value group {index} in {where}"
    named = {
        "c_fc": "the MLP's up map", "gate_proj": "the MLP's gate map", "down_proj": "the MLP's down (output) map",
        "rms1.gain": "the gain of the norm before attention", "rms2.gain": "the gain of the norm before the MLP",
    }
    return f"{named.get(rest, 'the weight matrix ' + rest)} in {where}"


def site_phrase(site: dict, layers: int) -> str:
    kind, layer = site["kind"], site.get("layer")
    where = f"layer {layer} of {layers}"
    if kind == "stream":
        return "the residual stream after the last layer" if layer == layers else f"the residual stream entering {where}"
    return {
        "middle": f"the residual stream between the attention and the MLP of {where}",
        "attention": f"the attention output of {where}",
        "head": f"the output of attention head {site.get('head')} in {where} (the input of its output map)",
        "mlp": f"the MLP output of {where}",
        "neurons": f"the MLP neuron activations of {where}",
    }[kind]


def positions_phrase(positions) -> str:
    if positions is None:
        return "at every position of the text"
    return f"at positions {positions} of the text (negative positions count from the end)"


def patch_phrase(patch: dict, layers: int, roles: dict[str, str], decode) -> str:
    target = site_phrase(patch["site"], layers)
    if patch.get("coordinates") is not None:
        target = f"coordinates {patch['coordinates']} of {target}"
    if patch.get("direction") is not None:
        target = f"the component along one fixed direction of {target}"
    value = patch["value"]
    kind = value["kind"]
    if kind == "zero":
        action = "is set to zero"
    elif kind == "scale":
        action = f"is multiplied by {value['factor']:g}"
    elif kind == "mean":
        action = f"is replaced by its mean over {len(value['sequences'])} reference texts"
    elif kind == "add":
        norm = math.sqrt(sum(x * x for x in value["vector"]))
        action = f"is shifted by a fixed vector of norm {norm:.3g}"
    elif kind == "source":
        who = "the same model" if value.get("model") is None else model_phrase(value["model"], roles)
        text = "the same text" if value.get("tokens") is None else f"another text, <<<{decode(value['tokens'])}>>>"
        action = f"is replaced by the value {who} computes on {text}"
        if value.get("positions") is not None:
            action += f" at its positions {value['positions']}"
    else:
        raise ValueError(f"patch value {kind!r}")
    if patch.get("sequences") is not None:
        action += f" (in sequences {patch['sequences']})"
    return f"{target} {action} {positions_phrase(patch.get('positions'))}"


def edit_phrase(edit: dict, layers: int, roles: dict[str, str]) -> str:
    c, alpha = edit["component"], edit["alpha"]
    kind = c["kind"]
    times = "removed" if alpha == 0 else f"multiplied by {alpha:g}"
    if kind == "operator":
        return f"{operator_phrase(c['name'], layers)} is {times}"
    if kind == "rows":
        return f"rows {c['rows']} of {operator_phrase(c['name'], layers)} are {times}"
    if kind == "columns":
        return f"columns {c['columns']} of {operator_phrase(c['name'], layers)} are {times}"
    if kind == "head":
        return f"the output map of attention head {c['head']} in layer {c['layer']} of {layers} is {times}"
    if kind == "neuron":
        return f"the output weights of MLP neuron {c['index']} in layer {c['layer']} of {layers} are {times}"
    if kind == "difference":
        ref = model_phrase(c["reference"], roles)
        part = "its difference from" if c.get("components") is None else f"singular components {c['components']} of its difference from"
        if alpha == 0 and c.get("components") is None:
            return f"{operator_phrase(c['name'], layers)} is set back to its value in {ref}"
        return f"in {operator_phrase(c['name'], layers)}, {part} {ref} is {times}"
    if kind == "direction":
        side = "reads from its input along" if c["side"] == "input" else "writes to its output along"
        return f"the part of {operator_phrase(c['name'], layers)} that {side} one fixed direction is {times}"
    raise ValueError(f"component {kind!r}")


def describe_intervention(model: str, intervention: dict, layers: int, roles: dict[str, str], decode) -> str:
    """oracle.rs's Run arm (a model and an Intervention) in plain language; layers numbered from 0."""
    who = model_phrase(model, roles)
    edits = [edit_phrase(e, layers, roles) for e in intervention.get("edits", [])]
    patches = [patch_phrase(p, layers, roles, decode) for p in intervention.get("patches", [])]
    if not edits and not patches:
        return f"None: {who} runs as it is."
    parts = [f"{who[0].upper() + who[1:]} runs with these changes (layers are numbered from 0)."]
    if edits:
        parts.append("Weights: " + "; ".join(edits) + ".")
    if patches:
        parts.append("During the run: " + "; ".join(patches) + ".")
    return " ".join(parts)


# ---------------------------------------------------------------------------------------------------
# Tests


def load_contexts(path, tokenizer) -> list[str]:
    """Public texts: JSONL lines {"text"} as they are, or {"messages"} through the chat template with the
    generation prompt and thinking disabled (the text the model reads before its reply)."""
    texts = []
    with open(os.path.expanduser(path)) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "messages" in row:
                texts.append(tokenizer.apply_chat_template(row["messages"], add_generation_prompt=True, enable_thinking=False, tokenize=False))
            else:
                texts.append(row["text"])
    if not texts:
        raise ValueError(f"{path}: no contexts")
    return texts


def report_edits(report: dict) -> list[dict] | None:
    edits = report.get("edits")
    return edits if isinstance(edits, list) and edits else None


def families(target: dict, report: dict) -> list[str]:
    out = ["clean", "zero_head", "zero_mlp"]
    if "base" in target["models"]:
        out += ["base", "swap_mlp", "swap_attention", "revert_operator"]
    if report_edits(report):
        out.append("report_edit")
    return out


def arm(family: str, rng: np.random.Generator, target: dict, info: dict, report: dict) -> tuple[str, dict]:
    """The model and Intervention of one test arm."""
    updated = target["models"]["updated"]["server"]
    shape = info["models"][updated]
    layer = int(rng.integers(shape["layers"]))
    base = target["models"].get("base", {}).get("server")
    source = lambda: {"kind": "source", "model": base}  # noqa: E731
    if family == "clean":
        return updated, {}
    if family == "base":
        return base, {}
    if family == "zero_head":
        head = int(rng.integers(shape["heads"]))
        return updated, {"patches": [{"site": {"kind": "head", "layer": layer, "head": head}, "value": {"kind": "zero"}}]}
    if family == "zero_mlp":
        return updated, {"patches": [{"site": {"kind": "mlp", "layer": layer}, "value": {"kind": "zero"}}]}
    if family == "swap_mlp":
        return updated, {"patches": [{"site": {"kind": "mlp", "layer": layer}, "value": source()}]}
    if family == "swap_attention":
        return updated, {"patches": [{"site": {"kind": "attention", "layer": layer}, "value": source()}]}
    if family == "revert_operator":
        names = shape["operators"]
        name = names[int(rng.integers(len(names)))]
        return updated, {"edits": [{"component": {"kind": "difference", "name": name, "reference": base}, "alpha": 0.0}]}
    if family == "report_edit":
        return updated, {"edits": report_edits(report)}
    raise ValueError(family)


def _ok(reply: dict, what: str):
    if "error" in reply:
        raise RuntimeError(f"{what}: {reply['error']}")
    return reply["ok"]


def draw_tests(episode: dict, client, tokenizer, count: int, per_side: int) -> list[dict]:
    """`count` fresh tests measured through the server (two batches: candidate search, then measure)."""
    target = episode["target"]
    report = episode["report"]["content"]
    draw = fresh_draw(episode)
    rng = np.random.default_rng(draw["seed"])
    info = client.info()
    updated = target["models"]["updated"]["server"]
    layers = info["models"][updated]["layers"]
    roles = {spec["server"]: role for role, spec in target["models"].items()}
    contexts = load_contexts(target["contexts"], tokenizer)
    allowed = families(target, report)
    decode = lambda ids: tokenizer.decode(ids)  # noqa: E731
    plans = []
    for i in range(count):
        text = contexts[int(rng.integers(len(contexts)))]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        family = allowed[int(rng.integers(len(allowed)))]
        model, intervention = arm(family, rng, target, info, report)
        plans.append({"text": text, "tokens": tokens, "family": family, "model": model, "intervention": intervention})
    top = 2 * per_side
    search = []
    for p in plans:
        search.append({"op": "run", "model": updated, "sequences": [p["tokens"]], "top": top})
        search.append({"op": "run", "model": p["model"], "sequences": [p["tokens"]], "top": top, "intervention": p["intervention"]})
    found = client.batch(search)
    measure = []
    for i, p in enumerate(plans):
        reference = [t for t, _ in _ok(found[2 * i], "candidate search")["runs"][0]["positions"][0]["top"]]
        armed = [t for t, _ in _ok(found[2 * i + 1], "candidate search")["runs"][0]["positions"][0]["top"]]
        candidates = []
        for a, b in zip(reference, armed):
            for t in (a, b):
                if t not in candidates and len(candidates) < top:
                    candidates.append(t)
        candidates = [candidates[j] for j in rng.permutation(len(candidates))]
        p["candidates"] = candidates
        measure.append({"op": "run", "model": p["model"], "sequences": [p["tokens"]], "top": 0, "targets": candidates, "intervention": p["intervention"]})
    measured = client.batch(measure)
    tests = []
    for i, p in enumerate(plans):
        entry = _ok(measured[i], "measure")["runs"][0]["positions"][0]
        lp = dict((int(t), v) for t, v in entry["targets"])
        log_probabilities = [lp[t] for t in p["candidates"]]
        shift = max(log_probabilities)
        w = [math.exp(v - shift) for v in log_probabilities]
        tests.append({
            "id": f"{draw['entropy'][:12]}-{i}",
            **draw,
            "family": p["family"],
            "context": {"text": p["text"], "tokens": p["tokens"]},
            "request": measure[i],
            "intervention_text": describe_intervention(p["model"], p["intervention"], layers, roles, decode),
            "options": [tokenizer.decode([t]) for t in p["candidates"]],
            "option_tokens": p["candidates"],
            "measured": {"log_probabilities": log_probabilities, "p": [x / sum(w) for x in w]},
            "replies": {"search": [found[2 * i], found[2 * i + 1]], "measure": measured[i]},
        })
    return tests


# ---------------------------------------------------------------------------------------------------
# Readers' documents and scores


def report_documents(report: dict | None) -> list[str]:
    """What the reader reads of a report: its plain-language rule (an absent or empty rule: nothing)."""
    if not isinstance(report, dict) or not isinstance(report.get("rule"), str) or not report["rule"].strip():
        return []
    return [report["rule"]]


def transcript_text(transcript: list[dict]) -> str:
    lines = []
    for m in transcript:
        content = m.get("content") or ""
        if m.get("tool_calls"):
            content += "\n" + "\n".join(json.dumps(c.get("function", c), ensure_ascii=False) for c in m["tool_calls"])
        lines.append(f"[{m['role']}{' ' + m['name'] if m.get('name') else ''}]\n{content}")
    return "\n\n".join(lines)


def documents_for(episode: dict, condition: str, ablations: dict[str, str] | None = None) -> list[str]:
    if condition == "report":
        return report_documents(episode["report"]["content"])
    if condition == "transcript":
        return [transcript_text(episode["investigator"]["transcript"])]
    if condition == "none":
        return []
    if condition.startswith("ablated:"):
        kind = condition.split(":", 1)[1]
        if not ablations or kind not in ablations:
            raise ValueError(f"no checked ablation of kind {kind!r} for this report")
        return [ablations[kind]]
    raise ValueError(f"documents {condition!r}")


def reader_tests(episode: dict, documents: list[str], tests: list[dict] | None = None) -> list[dict]:
    if not (episode["tests"] if tests is None else tests):
        raise ValueError(f"episode {episode['episode']} has no tests")
    return [
        {"id": t["id"], "documents": documents, "context": t["context"]["text"], "intervention": t["intervention_text"], "options": t["options"], "p": t["measured"]["p"]}
        for t in (episode["tests"] if tests is None else tests)
    ]


def add_scores(episode: dict, condition: str, reader: dict, rows: list[dict], results: list[dict]) -> str:
    from reader import log_score

    key = f"{reader['backend']}:{reader['model']}:{condition}"
    per_test = {row["id"]: {"q": r["q"], "log_score": log_score(row["p"], r["q"])} for row, r in zip(rows, results)}
    episode["scores"][key] = {
        "reader": reader,
        "documents": condition,
        "per_test": per_test,
        "mean_log_score_nats": float(np.mean([v["log_score"] for v in per_test.values()])),
    }
    return key


# ---------------------------------------------------------------------------------------------------
# The investigator's instructions and tool, shared by train_sft.py and train_rl.py


INVESTIGATOR_INSTRUCTIONS = """You investigate a language model through measured experiments on its native computation and write its counterfactual operating manual.

Tool. `experiment` sends one request to the experiment server and returns its JSON reply. Requests are JSON objects with an "op": "info" (the models' shapes and block operators), "run" (next-token log-probabilities under an intervention: patches of sites and native weight edits), "crossed" (how an intervention changes the effect of an input distinction), "generate", "attention", "unembed", "difference", "components", "localize", "scan", or "batch" (a list of requests). Sites: stream, middle, attention, head, mlp, neurons (each with a layer, a head for head). Every number in a reply is measured, never estimated.

Budget. You have {budget} server requests (a batch counts as its members); after that the tool refuses.

Report. End with a reply that is only the report: a JSON object with "rule" (the rule the model follows, in plain language: when its behaviour changes and what it does then; readable cold and specific enough to predict the model's next token on new texts under interventions) and optionally "edits" (a list of native weight edits in the server's Edit format that should remove the rule's effect), "hypothesis" (an executable hypothesis bound to the model's native computations), and "predictions". An independent reader reads the first {report_tokens} tokens of the rule and predicts the model's next token on texts and interventions drawn after your report is frozen."""


def investigator_prompt(target: dict, budget: int, report_tokens: int) -> list[dict]:
    names = ", ".join(f"{role}: server model {spec['server']!r}" for role, spec in target["models"].items())
    return [
        {"role": "system", "content": INVESTIGATOR_INSTRUCTIONS.format(budget=budget, report_tokens=report_tokens)},
        {"role": "user", "content": f"Target {target['id']} ({target['kind']}). Models: {names}.\n\n{target['description']}"},
    ]


class Investigation:
    """One investigation's tool and budget. Its public methods are the investigator's tools (TRL exposes
    an environment's public methods as tools, from their signatures and docstrings)."""

    def __init__(self, client, budget: int):
        self._client = client
        self._budget = budget
        self._used = 0

    def experiment(self, request: str) -> str:
        """Send one request to the native experiment server and return its reply.

        Args:
            request: One JSON request object with an "op" field, for example {"op": "run", "model": "updated", "sequences": [[9707, 11]], "targets": [1879]}.

        Returns:
            The server's JSON reply, {"ok": ...} or {"error": ...}.
        """
        try:
            parsed = json.loads(request)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"the request is not JSON: {e}"})
        if not isinstance(parsed, dict):
            return json.dumps({"error": "the request must be a JSON object"})
        cost = len(parsed.get("requests", [])) if parsed.get("op") == "batch" else 1
        if self._used + cost > self._budget:
            return json.dumps({"error": f"the request budget of {self._budget} is spent ({self._used} used); write the report now"})
        self._used += cost
        try:
            reply = self._client.raw(parsed)
        except (OSError, ValueError) as e:
            return json.dumps({"error": f"server unreachable: {e}"})
        return json.dumps(reply)


# ---------------------------------------------------------------------------------------------------
# Command line


def main():
    from native_client import NativeClient

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    a = sub.add_parser("new")
    a.add_argument("--target", required=True)
    a.add_argument("--investigator", required=True)
    a.add_argument("--model", required=True)
    a.add_argument("--episode")
    a.add_argument("--root", default=str(ROOT))
    a = sub.add_parser("freeze")
    a.add_argument("--episode", required=True)
    a.add_argument("--report", required=True)
    a = sub.add_parser("tests")
    a.add_argument("--episode", required=True)
    a.add_argument("--server", required=True)
    a.add_argument("--count", type=int, required=True)
    a.add_argument("--candidates-per-side", type=int, required=True)
    a = sub.add_parser("score")
    a.add_argument("--episode", required=True)
    a.add_argument("--reader", required=True, help="the reader service's address (reader.py serve)")
    a.add_argument("--documents", required=True)
    a.add_argument("--ablations", help="ablate.py's output for this report (for --documents ablated:KIND)")
    a = sub.add_parser("show")
    a.add_argument("--episode", required=True)
    args = ap.parse_args()

    if args.command == "new":
        target = json.loads(Path(args.target).read_text())
        episode = new_episode(target, args.investigator, args.model, args.episode)
        print(save(episode, root=Path(args.root)))
        return
    episode = load(args.episode)
    if args.command == "freeze":
        freeze(episode, json.loads(Path(args.report).read_text()))
        save(episode, args.episode)
        print(json.dumps({k: episode["report"][k] for k in ("sha256", "frozen_at_utc")}))
    elif args.command == "tests":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(episode["target"]["models"]["updated"]["path"])
        client = NativeClient(args.server)
        tests = draw_tests(episode, client, tokenizer, args.count, args.candidates_per_side)
        attach_tests(episode, tests)
        save(episode, args.episode)
        print(json.dumps({"tests": len(tests), "families": sorted({t["family"] for t in tests}), "server_seconds": client.server_seconds()}))
    elif args.command == "score":
        ablations = None
        if args.ablations:
            rows = [json.loads(line) for line in open(args.ablations) if line.strip()]
            ablations = {r["kind"]: r["rewritten"] for r in rows if r.get("passed") and r.get("report_sha256") == episode["report"]["sha256"]}
        rows = reader_tests(episode, documents_for(episode, args.documents, ablations))
        reply = NativeClient(args.reader).call({"op": "read", "tests": rows})
        key = add_scores(episode, args.documents, reply["reader"], rows, reply["results"])
        save(episode, args.episode)
        print(json.dumps({"key": key, "mean_log_score_nats": episode["scores"][key]["mean_log_score_nats"]}))
    elif args.command == "show":
        summary = {
            "episode": episode["episode"], "target": episode["target"]["id"], "investigator": episode["investigator"]["name"],
            "requests": episode["investigator"]["requests"], "report_sha256": episode["report"] and episode["report"]["sha256"],
            "tests": len(episode["tests"]), "scores": {k: v["mean_log_score_nats"] for k, v in episode["scores"].items()},
        }
        print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
