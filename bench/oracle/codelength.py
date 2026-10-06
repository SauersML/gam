"""The description reward of the oracle (#2951): the two-part code length, in bits, of a subcomponent's
measured behaviour (vpd_labels.py's table for VPD's vpd4l decomposition) given a description:

  L(description)   -log2 of the description under a frozen prior model (the oracle's own base model,
                   read as the answer to a fixed request to describe a component), so a long or unlikely
                   description pays for itself;
  L(behaviour | description) under a frozen text-only reader, -log2 q summed over
    activity       in each of the subcomponent's measured contexts (its `--top` strongest and `--others`
                   others), the text listed one native token per line, each followed by a tab and the
                   subcomponent's activity level there on 0-9 (floor(10 |a| / a_max), 9 at most; a_max its
                   largest |v . x| over its measured contexts), each level read after everything listed
                   before it (a sequential code in one teacher-forced pass per context);
    directions     at the peak of each of its strongest contexts: the 10 next tokens whose probability
                   rises most and the 10 that fall most when it is removed (alpha 0), in a seeded order,
                   each followed by " up" or " down".
The oracle's reward is -[L(description) + L(behaviour | description)]; L(behaviour | nothing) is
reported beside it (the empty description, with no description bits). The description comes first in
every reader prompt, so a reader with prefix caching reads it once per subcomponent.

  codelength.py score --labels DIR --items D.jsonl --reader-backend vllm --reader Qwen/Qwen3-8B
                      --prior-backend vllm --prior Qwen/Qwen3-1.7B --out OUT.jsonl
  codelength.py serve ... --listen HOST:PORT     ({"op": "score", "items": [...]} -> {"ok": {"results": [...]}})
D.jsonl lines {"id", "component": [layer, kind, index], "description"}.
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
TOKENIZER = Path.home() / "mpd-data/vpd/t-9d2b8f02/tokenizer.json"

ACTIVATION_PROMPT = (
    "A component of a language model is described as follows.\n\nDescription: {description}\n\n"
    "Below is a text, one token per line. After each token, write a tab and the component's activity at that "
    "token on a scale of 0 to 9 (0: inactive; 9: its largest activity)."
)
DIRECTION_PROMPT = (
    "A component of a language model is described as follows.\n\nDescription: {description}\n\n"
    "The model reads the text below. If this component is removed, the model's probability of each listed next "
    "token changes. For each token, write whether its probability goes up or down.\n\nText: {text}"
)
PRIOR_PROMPT = "Describe what this component of a language model responds to and what it does."


class Labels:
    """A vpd_labels.py run, by subcomponent, with the target's tokens as strings."""

    def __init__(self, root: Path, top: int, others: int):
        import tokenizers

        self.root = root
        self.top, self.others = top, others
        self.tokens = load_file(str(root / "contexts.safetensors"))["tokens"]
        self.tok = tokenizers.Tokenizer.from_file(str(TOKENIZER))
        self.meta = {}
        for path in root.glob("site_*.json"):
            m = json.loads(path.read_text())
            self.meta[(m["layer"], m["site"].split(".")[-1])] = (m, path.with_suffix(".safetensors"))

    @lru_cache(maxsize=4)
    def site(self, layer: int, kind: str) -> dict:
        return load_file(str(self.meta[(layer, kind)][1]))

    def piece(self, token: int) -> str:
        return self.tok.decode([int(token)])

    def record(self, layer: int, kind: str, c: int) -> dict:
        d = self.site(layer, kind)
        top = self.meta[(layer, kind)][0]["top"]
        chosen = list(range(min(self.top, top))) + list(range(top, top + self.others))
        act = d["activity"][c].astype(np.float64)
        peak = np.abs(act).max()
        levels = np.zeros(act.shape, dtype=np.int64) if peak <= 0 else np.clip(np.floor(BINS * np.abs(act) / peak), 0, BINS - 1).astype(np.int64)
        contexts = d["contexts"][c]
        return {
            "activity": [([self.piece(t) for t in self.tokens[int(contexts[j])]], levels[j]) for j in chosen],
            "directions": [(self.tok.decode(self.tokens[int(contexts[j]), : int(d["position"][c, j]) + 1].tolist()),
                            [self.piece(t) for t in d["up_ids_ablate"][c, j]], [self.piece(t) for t in d["down_ids_ablate"][c, j]])
                           for j in range(min(self.top, top))],
        }


class Prompts:
    """Token ids of the scoring prompts in one model's chat template, with the indices scored."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        enc = lambda s: tokenizer.encode(s, add_special_tokens=False)  # noqa: E731
        self.enc = enc
        self.digits = [enc(str(b))[0] for b in range(BINS)]
        self.tab, self.newline = enc("\t")[0], enc("\n")[0]
        self.up, self.down = enc(" up")[0], enc(" down")[0]

    def chat(self, user: str, answer_ids: list[int]) -> tuple[list[int], int]:
        marker = "\u0000ANSWER\u0000"
        text = self.tok.apply_chat_template([{"role": "user", "content": user}, {"role": "assistant", "content": marker}], tokenize=False, enable_thinking=False)
        head, tail = text.split(marker)
        a = self.enc(head)
        return a + answer_ids + self.enc(tail), len(a)

    def activity(self, description: str, pieces: list[str], levels) -> tuple[list[int], list[int]]:
        answer, scored = [], []
        for piece, b in zip(pieces, levels.tolist()):
            answer += self.enc(piece) + [self.tab]
            scored.append(len(answer))
            answer += [self.digits[b], self.newline]
        ids, start = self.chat(ACTIVATION_PROMPT.format(description=description or "(none)"), answer)
        return ids, [start + j for j in scored]

    def directions(self, description: str, text: str, up: list[str], down: list[str], rng) -> tuple[list[int], list[int]]:
        items = [(p, self.up) for p in up] + [(p, self.down) for p in down]
        items = [items[i] for i in rng.permutation(len(items))]
        answer, scored = [], []
        for piece, word in items:
            answer += self.enc(piece)
            scored.append(len(answer))
            answer += [word, self.newline]
        ids, start = self.chat(DIRECTION_PROMPT.format(description=description or "(none)", text=text), answer)
        return ids, [start + j for j in scored]

    def prior(self, description: str) -> tuple[list[int], list[int]]:
        answer = self.enc(description)
        ids, start = self.chat(PRIOR_PROMPT, answer)
        return ids, list(range(start, start + len(answer)))


def score(reader, prior, labels: Labels, items: list[dict]) -> list[dict]:
    """Bits per item (module note), and each component's no-description bits once per call."""
    rp, pp = Prompts(reader.tokenizer), Prompts(prior.tokenizer)
    jobs: list[tuple[object, str, list[int], list[int]]] = []
    components = sorted({tuple(it["component"]) for it in items})
    for comp in components:
        for part, ids, at in parts(rp, labels, comp, ""):
            jobs.append((("nothing", comp), part, ids, at))
    for i, it in enumerate(items):
        for part, ids, at in parts(rp, labels, tuple(it["component"]), it["description"]):
            jobs.append((i, part, ids, at))
    lps = reader.token_log_probs([j[2] for j in jobs], [j[3] for j in jobs])
    prior_jobs = [(i, *pp.prior(it["description"])) for i, it in enumerate(items) if it["description"]]
    prior_lps = prior.token_log_probs([j[1] for j in prior_jobs], [j[2] for j in prior_jobs]) if prior_jobs else []
    bits: dict[object, dict[str, float]] = {}
    for (key, part, _, _), lp in zip(jobs, lps):
        bits.setdefault(key, {"description": 0.0, "activity": 0.0, "directions": 0.0})[part] += float(-lp.sum() / LN2)
    for (i, _, _), lp in zip(prior_jobs, prior_lps):
        bits[i]["description"] = float(-lp.sum() / LN2)
    out = []
    for i, it in enumerate(items):
        b = bits[i]
        b["total"] = b["description"] + b["activity"] + b["directions"]
        base = bits[("nothing", tuple(it["component"]))]
        nothing = base["activity"] + base["directions"]
        out.append({"id": it.get("id"), "component": list(it["component"]), "bits": b, "nothing_bits": nothing, "reward": -b["total"], "saved_bits": nothing - b["total"]})
    return out


def parts(prompts: Prompts, labels: Labels, comp: tuple, description: str):
    layer, kind, c = int(comp[0]), str(comp[1]), int(comp[2])
    rec = labels.record(layer, kind, c)
    rng = np.random.default_rng([layer, c])
    for pieces, levels in rec["activity"]:
        ids, at = prompts.activity(description, pieces, levels)
        yield "activity", ids, at
    for text, up, down in rec["directions"]:
        ids, at = prompts.directions(description, text, up, down, rng)
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
                srv = self.server
                reply = {"ok": {"results": score(srv.reader, srv.prior, srv.labels, request["items"]), "reader": R.describe(srv.reader), "prior": R.describe(srv.prior)}}
            except Exception as e:  # the reply carries the failure; the service keeps running
                reply = {"error": f"{type(e).__name__}: {e}"}
            reply["seconds"] = time.time() - start
            self.wfile.write((json.dumps(reply) + "\n").encode())
            self.wfile.flush()


def backend(kind: str, model: str, args, share: float):
    ns = argparse.Namespace(**{**vars(args), "backend": kind, "model": model, "gpu_memory_utilization": share})
    return R.make_backend(ns)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["score", "serve"])
    ap.add_argument("--labels", required=True)
    ap.add_argument("--reader-backend", required=True, choices=["vllm", "transformers"])
    ap.add_argument("--reader", required=True)
    ap.add_argument("--prior-backend", required=True, choices=["vllm", "transformers"])
    ap.add_argument("--prior", required=True)
    ap.add_argument("--top", type=int, default=4, help="the strongest measured contexts read per component")
    ap.add_argument("--others", type=int, default=4, help="the other measured contexts read per component")
    ap.add_argument("--items")
    ap.add_argument("--out")
    ap.add_argument("--listen")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-tokens", type=int, default=8192)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--max-model-len", type=int)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.4, help="vllm: the reader's share of the GPU's memory")
    ap.add_argument("--prior-gpu-memory-utilization", type=float, default=0.15, help="vllm: the prior's share, when it is another model")
    args = ap.parse_args()
    reader = backend(args.reader_backend, args.reader, args, args.gpu_memory_utilization)
    prior = reader if (args.prior, args.prior_backend) == (args.reader, args.reader_backend) else backend(args.prior_backend, args.prior, args, args.prior_gpu_memory_utilization)
    labels = Labels(Path(args.labels), args.top, args.others)
    if args.command == "serve":
        host, sep, port = args.listen.rpartition(":")
        server = socketserver.TCPServer((host, int(port)), _Handler) if sep and port.isdigit() else socketserver.UnixStreamServer(args.listen, _Handler)
        server.reader, server.prior, server.labels = reader, prior, labels
        print(f"code-length scorer: reader {R.describe(reader)}, prior {R.describe(prior)}, listening on {args.listen}", file=sys.stderr, flush=True)
        server.serve_forever()
    items = [json.loads(line) for line in open(args.items) if line.strip()]
    start = time.time()
    rows = score(reader, prior, labels, items)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps({**r, "reader": R.describe(reader), "prior": R.describe(prior)}) + "\n")
    print(json.dumps({"items": len(rows), "seconds": time.time() - start, "mean_reward_bits": float(np.mean([r["reward"] for r in rows])),
                      "mean_saved_bits": float(np.mean([r["saved_bits"] for r in rows]))}))


if __name__ == "__main__":
    main()
