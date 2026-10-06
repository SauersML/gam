"""The description reward of the introspective oracle (#2951): the two-part code length, in bits, of a
neuron's measured behaviour (labels.py's table) given a description, under a frozen reader LLM.

L(description) + L(behaviour | description), each a sum of -log2 q over tokens:
  description  the reader's own probability of the description's tokens as an answer to "Describe the
               neuron." (a fixed prior over descriptions, so a long description pays for its length);
  activations  for every context of the table, the text listed one native token per line, each followed
               by a tab and the neuron's activation bin at that token, the reader predicting each bin
               token after the description and everything listed before it (a sequential code, read in
               one teacher-forced pass); bins 0..9: floor(10 a / a_max) clipped to 9 for a > 0 and 0 for
               a <= 0, a_max the neuron's largest activation over the table's contexts (the simulation scale
               of Bills et al., 2023, in digits that are single tokens);
  directions   for every context, at the neuron's largest activation there: the 10 tokens whose
               probability rises most and the 10 that fall most when the neuron is removed (alpha 0), in a
               seeded order, each followed by " up" or " down", the reader predicting each word after the
               text up to that position.
The no-description code length is the same sum with an empty description (and no description bits); the
reward of a description is the bits it saves: L(behaviour | nothing) - L(description) - L(behaviour |
description). The description comes first in every prompt, so a reader with prefix caching reads it once.

  codelength.py score --labels DIR --descriptions D.jsonl --backend vllm|transformers --model M --out OUT.jsonl
  codelength.py serve --labels DIR --backend vllm --model M --listen ADDRESS
D.jsonl lines {"id", "neuron": [layer, index], "description"}; OUT.jsonl lines {"id", "bits": {description,
activations, directions, total}, "baseline_bits", "saved_bits"}. serve answers {"op": "score", "items": [...]}
with the same rows, one JSON line per request line.
"""

from __future__ import annotations

import argparse
import json
import math
import socketserver
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

import reader as R

BINS = 10
LN2 = math.log(2.0)

ACTIVATION_PROMPT = (
    "A neuron in a language model is described as follows.\n\nDescription: {description}\n\n"
    "Below is a text, one token per line. After each token, write a tab and the neuron's activation at that "
    "token on a scale of 0 to 9 (0: inactive; 9: its largest activation)."
)
DIRECTION_PROMPT = (
    "A neuron in a language model is described as follows.\n\nDescription: {description}\n\n"
    "The model reads the text below. If this neuron is removed, the model's probability of each listed next "
    "token changes. For each token, write whether its probability goes up or down.\n\nText: {text}"
)
PRIOR_PROMPT = "Describe what this neuron in a language model responds to and what it does."


class Labels:
    """The label table's shards, by neuron."""

    def __init__(self, root: Path):
        self.root = root
        self.tokens = load_file(str(root / "tokens.safetensors"))["tokens"]
        self.where: dict[tuple[int, int], tuple[int, int]] = {}
        for meta_path in sorted(root.glob("shard_*.json")):
            meta = json.loads(meta_path.read_text())
            for n, (layer, index) in enumerate(meta["neurons"]):
                self.where[(layer, index)] = (meta["layer"], n)

    @lru_cache(maxsize=8)
    def shard(self, layer: int) -> dict:
        return load_file(str(self.root / f"shard_{layer:02d}.safetensors"))

    def record(self, neuron: tuple[int, int]) -> dict:
        layer, n = self.where[tuple(neuron)]
        d = self.shard(layer)
        return {"activation": d["activation"][n].astype(np.float64), "positions": d["positions"][n].astype(np.int64),
                "up": d["up_ids_ablate"][n].astype(np.int64), "down": d["down_ids_ablate"][n].astype(np.int64)}


def bins(activation: np.ndarray) -> np.ndarray:
    peak = activation.max()
    if peak <= 0:
        return np.zeros(activation.shape, dtype=np.int64)
    return np.clip(np.floor(BINS * np.maximum(activation, 0.0) / peak), 0, BINS - 1).astype(np.int64)


class Prompts:
    """Token ids of the scoring prompts, with the indices of the tokens that are scored."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        enc = lambda s: tokenizer.encode(s, add_special_tokens=False)  # noqa: E731
        self.digits = [enc(str(b))[0] for b in range(BINS)]
        self.tab, self.newline = enc("\t")[0], enc("\n")[0]
        self.up, self.down = enc(" up")[0], enc(" down")[0]

    def chat(self, user: str, answer_ids: list[int]) -> tuple[list[int], int]:
        """The user turn and an assistant turn whose content is `answer_ids`; and where the answer starts."""
        marker = "\u0000ANSWER\u0000"
        text = self.tok.apply_chat_template([{"role": "user", "content": user}, {"role": "assistant", "content": marker}], tokenize=False, enable_thinking=False)
        head, tail = text.split(marker)
        a = self.tok.encode(head, add_special_tokens=False)
        return a + answer_ids + self.tok.encode(tail, add_special_tokens=False), len(a)

    def activations(self, description: str, tokens: np.ndarray, levels: np.ndarray) -> tuple[list[int], list[int]]:
        answer, scored = [], []
        for t, b in zip(tokens.tolist(), levels.tolist()):
            answer += [int(t), self.tab]
            scored.append(len(answer))
            answer += [self.digits[b], self.newline]
        ids, start = self.chat(ACTIVATION_PROMPT.format(description=description or "(none)"), answer)
        return ids, [start + j for j in scored]

    def directions(self, description: str, prefix: np.ndarray, up: np.ndarray, down: np.ndarray, rng: np.random.Generator) -> tuple[list[int], list[int]]:
        items = [(int(t), self.up) for t in up] + [(int(t), self.down) for t in down]
        items = [items[i] for i in rng.permutation(len(items))]
        answer, scored = [], []
        for t, word in items:
            answer += [t]
            scored.append(len(answer))
            answer += [word, self.newline]
        text = self.tok.decode(prefix.tolist())
        ids, start = self.chat(DIRECTION_PROMPT.format(description=description or "(none)", text=text), answer)
        return ids, [start + j for j in scored]

    def prior(self, description: str) -> tuple[list[int], list[int]]:
        answer = self.tok.encode(description, add_special_tokens=False)
        ids, start = self.chat(PRIOR_PROMPT, answer)
        return ids, list(range(start, start + len(answer)))


def score(backend, labels: Labels, items: list[dict]) -> list[dict]:
    """Bits per item (module note); the no-description code length of each neuron once per call."""
    prompts = Prompts(backend.tokenizer)
    jobs: list[tuple[int, str, list[int], list[int]]] = []  # (item or -1 - neuron slot, part, ids, scored)
    neurons = sorted({tuple(it["neuron"]) for it in items})
    for slot, neuron in enumerate(neurons):
        for part, ids, at in parts(prompts, labels, neuron, ""):
            jobs.append((-1 - slot, part, ids, at))
    for i, it in enumerate(items):
        for part, ids, at in parts(prompts, labels, tuple(it["neuron"]), it["description"]):
            jobs.append((i, part, ids, at))
        if it["description"]:
            ids, at = prompts.prior(it["description"])
            jobs.append((i, "description", ids, at))
    lps = backend.token_log_probs([j[2] for j in jobs], [j[3] for j in jobs])
    bits: dict[int, dict[str, float]] = {}
    for (key, part, _, _), lp in zip(jobs, lps):
        bits.setdefault(key, {"description": 0.0, "activations": 0.0, "directions": 0.0})[part] += float(-lp.sum() / LN2)
    out = []
    for i, it in enumerate(items):
        b = bits[i]
        b["total"] = b["description"] + b["activations"] + b["directions"]
        base = bits[-1 - neurons.index(tuple(it["neuron"]))]
        baseline = base["activations"] + base["directions"]
        out.append({"id": it.get("id"), "neuron": list(it["neuron"]), "bits": b, "baseline_bits": baseline, "saved_bits": baseline - b["total"]})
    return out


def parts(prompts: Prompts, labels: Labels, neuron: tuple[int, int], description: str):
    rec = labels.record(neuron)
    levels = bins(rec["activation"])
    rng = np.random.default_rng([neuron[0], neuron[1]])
    for c in range(levels.shape[0]):
        ids, at = prompts.activations(description, labels.tokens[c], levels[c])
        yield "activations", ids, at
        p = int(rec["positions"][c, 0])
        ids, at = prompts.directions(description, labels.tokens[c, : p + 1], rec["up"][c, 0], rec["down"][c, 0], rng)
        yield "directions", ids, at


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        for line in self.rfile:
            if not line.strip():
                continue
            start = time.time()
            try:
                request = json.loads(line)
                if request.get("op") != "score":
                    raise ValueError(f"unknown op {request.get('op')!r} (have score)")
                reply = {"ok": {"results": score(self.server.backend, self.server.labels, request["items"]), "reader": R.describe(self.server.backend)}}
            except Exception as e:  # the reply carries the failure; the service keeps running
                reply = {"error": f"{type(e).__name__}: {e}"}
            reply["seconds"] = time.time() - start
            self.wfile.write((json.dumps(reply) + "\n").encode())
            self.wfile.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["score", "serve"])
    ap.add_argument("--labels", required=True)
    ap.add_argument("--backend", required=True, choices=["vllm", "transformers"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--descriptions")
    ap.add_argument("--out")
    ap.add_argument("--listen")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-tokens", type=int, default=8192)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--max-model-len", type=int)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = ap.parse_args()
    args.concurrency = 1
    backend = R.make_backend(args)
    labels = Labels(Path(args.labels))
    if args.command == "serve":
        host, sep, port = args.listen.rpartition(":")
        server = socketserver.TCPServer((host, int(port)), _Handler) if sep and port.isdigit() else socketserver.UnixStreamServer(args.listen, _Handler)
        server.backend, server.labels = backend, labels
        print(f"code-length reader {R.describe(backend)} listening on {args.listen}", file=sys.stderr, flush=True)
        server.serve_forever()
    items = [json.loads(line) for line in open(args.descriptions) if line.strip()]
    start = time.time()
    rows = score(backend, labels, items)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps({**r, "reader": R.describe(backend)}) + "\n")
    print(json.dumps({"items": len(rows), "seconds": time.time() - start, "mean_saved_bits": float(np.mean([r["saved_bits"] for r in rows]))}))


if __name__ == "__main__":
    main()
