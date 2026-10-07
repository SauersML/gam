"""The reader term of the graph oracle's score (#2951, design.txt section 2): the measured ground truth for
a program's English. A frozen reader LM (Qwen3-8B) reads the program's whole text (code, docstrings and
comments) and, for each scored experiment e on the target model M and each target token, predicts M's
next-token distribution under e. The reader cannot run the code. The reader term is the KL divergence from
M_e's measured distribution to the reader's prediction, in bits, summed over the scored (experiment, target
token) pairs and scaled to the behaviour's declared size N (N times the mean over the scored pairs).

Items. An item is one experiment at one target token, measured by the checker on M:
  {"id", "family", "experiment": {"words": str} or an operation (`words` renders it), "text": the text M
   reads under e, up to and including the token before the target, "token_ids": M's ids of that text
   (Qwen3 targets), "candidates": [{"token_id", "text", "clean", "p"}], "clean_other", "other"}
The candidates are M's K most probable next tokens on the clean model and clean prompt ("clean" is their
clean probability, "p" their probability under e), and "other" is the rest: other = 1 - sum of p. The KL is
over the K candidates and "other".

Reader prompt (the reader's chat template, thinking off):
  system     SYSTEM
  user       INSTRUCTIONS, the program text in a python block, then the item: the experiment in words, M's
             clean candidates with their clean probabilities and the clean rest
  assistant  the text M reads, verbatim and unmarked (so the reader's natural continuation is the answer);
             the answer slot is the position after it
The reader's probability of a candidate is its probability of continuing the text with that candidate.
The program part comes first, so vLLM's prefix cache shares it across the program's items.
- Qwen3 targets share the reader's tokenizer: the text is M's own token ids and every candidate is one
  reader token, so the candidates are disjoint and q(other) = 1 - sum q(candidates) exactly.
- Other targets (vpd4l): the text is encoded with the reader's tokenizer and a candidate is the event that
  the reader's continuation text starts with the candidate's string s. With the transformers backend,
  P(s) = the reader's probability of every first token whose string starts with s, plus, when the reader's
  own tokens of s are several, the probability of that token sequence (other splits of s are left out).
  With vLLM (no whole distribution), P(s) = the probability of the reader's own tokens of s alone, which
  misses first tokens longer than s ("PACKAGE" for the candidate "PACK"). A candidate whose event
  contains another's (s a prefix of s') gets q(s) = P(s) - sum over the nearest such s' of P(s'); the
  candidates are then disjoint and q(other) is the reader's remaining mass.
q(other) and corrected candidates are floored at K * 2^-24, the rounding scale of a sum of K float32
probabilities (vLLM returns float32 log-probabilities).

Baselines, scored on the same items: the empty program (the reader alone, the same prompt with an empty
program block) and the program's code with every docstring, string statement and comment removed
(`strip_english`). english_saved_bits = N (mean bits of the code alone - mean bits of the program): the
measured value of the English.

  reader_score.py score --backend vllm --model Qwen/Qwen3-8B --target vpd4l --programs P.jsonl
                        --items ITEMS.jsonl --out OUT.json [--N 16777216]
  reader_score.py serve --backend vllm --model Qwen/Qwen3-8B --target qwen3-0.6b --listen HOST:PORT
P.jsonl lines {"id", "source"[, "valid"]}; an invalid program is read as the empty program (design.txt).
serve answers JSON lines {"op": "score", "programs": [...], "items": [...], "N": int} with
{"ok": {"results": [...], "reader": ..., "items_per_second": ...}}.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import math
import socketserver
import sys
import time
import tokenize
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import reader as R  # noqa: E402

LN2 = math.log(2.0)
N_DEFAULT = 2**24
FLOAT32_EPS = 2.0**-24

SYSTEM = "You predict the measured behaviour of a language model, the target model, under experiments on its computation."

INSTRUCTIONS = """The program below explains how the target model produces a behaviour. It is written with the `mech` library:
- L[l].head[h] is attention head h of layer l (its query, key, value and output weights). L[l].mlp[i, j] are neurons i and j of the MLP of layer l (their input and output weights). PD.vpd[l].<matrix>[i] is subcomponent i of the parameter decomposition of that weight matrix of layer l. PD.tc[l][f] is transcoder feature f of layer l.
- node(...) groups pieces of the weights into one node. `a >> b.key` states that the output of node a reaches the key input of node b; the inputs are query, key and value for heads and input for MLP pieces. embed is the token embedding and logits is the output.
- Every piece outside the program's nodes, and every connection the program does not list, contributes its average over the behaviour's prompts.
- The comments and docstrings state what the nodes compute and how the target model's output depends on them.

Program:
```python
{source}
```

"""

ITEM = """Experiment: {words}

Without the experiment, the target model's most probable next tokens after the text of your reply are (each token as a JSON string, so spaces and newlines are explicit, then its probability):
{listing}
every other token: {other:.3g}

Your reply begins with the text the target model reads. Continue that text with one token: the target model's next token under the experiment, drawn with the probabilities the target model gives the tokens."""


def piece_name(p: dict) -> str:
    """A piece of the program IR (design.txt section 5) in mech syntax."""
    view, layer, kind, index = p["view"], p["layer"], p["kind"], p.get("index")
    idx = None if index is None else (", ".join(str(i) for i in index) if isinstance(index, list) else str(index))
    if view == "native":
        return f"L[{layer}].{kind}" + ("" if idx is None else f"[{idx}]")
    if view == "vpd":
        return f"PD.vpd[{layer}].{kind}" + ("" if idx is None else f"[{idx}]")
    if view == "transcoder":
        return f"PD.tc[{layer}]" + ("" if idx is None else f"[{idx}]")
    if view == "library":
        return "PD.lib" + ("" if idx is None else f"[{idx}]")
    raise ValueError(f"piece view {view!r}")


def _pieces(ps) -> str:
    if isinstance(ps, str):  # "embed" or "logits"
        return ps
    return " + ".join(piece_name(p) for p in ps)


def words(e: dict) -> str:
    """The experiment in words; the same words for every program, since experiments are identical for M
    and every program."""
    if "words" in e:
        return e["words"]
    kind = e["kind"]
    if kind == "clean":
        return "none (the target model as it is, on its original text)"
    if kind == "prompt_edit":
        return f"the input text is changed; the original text was <<<{e['clean_text']}>>>, and the model's weights are unchanged"
    if kind == "scale":
        f = e["factor"]
        if f == 0:
            return f"remove {_pieces(e['pieces'])} (its weights multiplied by 0) at every position"
        return f"multiply the weights of {_pieces(e['pieces'])} by {f:g} at every position"
    if kind == "low_rank":
        return f"add a random rank-{e['rank']} matrix to the {e['matrix']} weights of layer {e['layer']}, of norm {e['relative_norm']:.3g} times the norm of those weights"
    if kind == "swap":
        return f"set the output of {_pieces(e['pieces'])} to its output when the model reads the text <<<{e['source_text']}>>>"
    if kind == "cut":
        to = _pieces(e["to"]) + ("" if e.get("route") in (None, "input") or e["to"] == "logits" else f" ({e['route']} input)")
        return f"cut the connection from {_pieces(e['from'])} to {to}: there it receives the average output of {_pieces(e['from'])} over the behaviour's prompts in place of its actual output"
    raise ValueError(f"experiment kind {kind!r}")


def strip_english(source: str) -> str:
    """The program's code without its English: every string expression statement (docstrings included)
    and every comment removed. A program that does not parse loses its comments only."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        out = []
        for t in tokenize.generate_tokens(io.StringIO(source).readline):
            if t.type != tokenize.COMMENT:
                out.append(t)
        return tokenize.untokenize(out)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list):
            kept = [s for s in body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str))]
            if len(kept) != len(body):
                node.body = kept or [ast.Expr(ast.Constant(...))]
    return ast.unparse(tree)


class Prompter:
    """Token ids of the reader prompt: prefix(source) is shared by a program's items; item(it) follows it
    and ends with the text M reads (the answer slot is after it); candidates(it) are the candidates'
    reader tokens."""

    def __init__(self, tokenizer, shared_vocab: bool):
        self.tok = tokenizer
        self.shared = shared_vocab
        marker = "\u0000U\u0000"
        rendered = tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": marker}],
                                                 add_generation_prompt=True, enable_thinking=False, tokenize=False)
        self.head, self.mid = rendered.split(marker)

    def enc(self, s: str) -> list[int]:
        return self.tok.encode(s, add_special_tokens=False)

    def prefix(self, source: str) -> list[int]:
        return self.enc(self.head + INSTRUCTIONS.replace("{source}", source))

    def item(self, it: dict) -> list[int]:
        listing = "\n".join(f"{json.dumps(c['text'], ensure_ascii=False)} {c['clean']:.3g}" for c in it["candidates"])
        user = ITEM.format(words=words(it["experiment"]), listing=listing, other=it["clean_other"])
        text = list(it["token_ids"]) if self.shared else self.enc(it["text"])
        return self.enc(user) + self.enc(self.mid) + text

    def candidates(self, it: dict) -> list[list[int]]:
        if self.shared:
            return [[int(c["token_id"])] for c in it["candidates"]]
        return [self.enc(c["text"]) for c in it["candidates"]]


def disjoint(keys: list, p: list[float]) -> np.ndarray:
    """q per candidate from P(event of its key), the event that the reader's continuation starts with the
    key (a tuple of reader tokens or a string): P minus P of the nearest candidates whose keys extend the
    key. Candidates with the same key are one event of the reader, shared equally between them."""
    raw = dict(zip((k if isinstance(k, str) else tuple(k) for k in keys), (float(x) for x in p)))
    q = {}
    for ka in raw:
        ext = [kb for kb in raw if len(kb) > len(ka) and kb[: len(ka)] == ka]
        nearest = [kb for kb in ext if not any(len(kc) < len(kb) and kb[: len(kc)] == kc for kc in ext)]
        q[ka] = raw[ka] - sum(raw[kb] for kb in nearest)
    norm = [k if isinstance(k, str) else tuple(k) for k in keys]
    counts = {k: norm.count(k) for k in raw}
    return np.array([q[k] / counts[k] for k in norm])


def kl_bits(p: np.ndarray, p_other: float, q: np.ndarray) -> float:
    """KL(p || q) in bits over the K candidates and "other", q(other) = 1 - sum q; q floored at K 2^-24."""
    floor = len(q) * FLOAT32_EPS
    q = np.maximum(q, floor)
    q_other = max(1.0 - float(q.sum()), floor)
    keep = p > 0
    kl = float((p[keep] * (np.log(p[keep]) - np.log(q[keep]))).sum())
    if p_other > 0:
        kl += p_other * (math.log(p_other) - math.log(q_other))
    return kl / LN2


class Scorer:
    """One loaded reader backend and its prompter; bits per (text, item), memoized by content."""

    def __init__(self, backend, target: str):
        self.backend = backend
        shared = target.lower().startswith("qwen3") and "qwen3" in backend.model_id.lower()
        self.prompter = Prompter(backend.tokenizer, shared)
        self.memo: dict[tuple[str, str], float] = {}
        self._vocab = None

    def vocabulary(self):
        """The reader's token strings, and per candidate string the reader tokens whose string starts with
        it (the continuation's first token then covers the whole candidate)."""
        if self._vocab is None:
            tok = self.backend.tokenizer
            n = len(tok)

            class Vocab:
                ids = list(range(n))
                strings = np.array([tok.decode([i]) for i in range(n)])
                cache: dict[str, np.ndarray] = {}

                def starting(self, t: str) -> np.ndarray:
                    if t not in self.cache:
                        self.cache[t] = np.nonzero(np.char.startswith(self.strings, t))[0]
                    return self.cache[t]

            self._vocab = Vocab()
        return self._vocab

    @staticmethod
    def _key(text: str, it: dict) -> tuple[str, str]:
        return hashlib.sha1(text.encode()).hexdigest(), hashlib.sha1(json.dumps(it, sort_keys=True).encode()).hexdigest()

    def bits(self, texts: list[str], items: list[dict]) -> np.ndarray:
        """[len(texts), len(items)] bits."""
        out = np.empty((len(texts), len(items)))
        todo = [(a, b) for a in range(len(texts)) for b in range(len(items)) if self._key(texts[a], items[b]) not in self.memo]
        if todo:
            pr = self.prompter
            prefixes = {a: pr.prefix(texts[a]) for a in {a for a, _ in todo}}
            bodies = {b: pr.item(items[b]) for b in {b for _, b in todo}}
            cands = {b: pr.candidates(items[b]) for b in bodies}
            prompts = [prefixes[a] + bodies[b] for a, b in todo]
            aggregate = not pr.shared and self.backend.name == "transformers"
            if aggregate:  # the reader's whole next-token distribution, for string events
                vocab = self.vocabulary()
                first = self.backend.next_log_probs(prompts, [vocab.ids] * len(prompts))
            else:
                first = self.backend.next_log_probs(prompts, [[c[0] for c in cands[b] if len(c) == 1] for _, b in todo])
            multi = [(j, k, c) for j, (_, b) in enumerate(todo) for k, c in enumerate(cands[b]) if len(c) > 1]
            longer = self.backend.token_log_probs([prompts[j] + c for j, _, c in multi], [list(range(len(prompts[j]), len(prompts[j]) + len(c))) for j, _, c in multi]) if multi else []
            extra = {(j, k): math.exp(float(lp.sum())) for (j, k, _), lp in zip(multi, longer)}
            for j, (a, b) in enumerate(todo):
                it = items[b]
                if aggregate:
                    full = np.exp(first[j])
                    strings = [c["text"] for c in it["candidates"]]
                    raw = [(full[vocab.starting(t)].sum() + extra.get((j, k), 0.0)) if t else 0.0 for k, t in enumerate(strings)]
                    q = disjoint(strings, raw)
                else:
                    s_ = iter(np.exp(first[j]))
                    q = disjoint(cands[b], [float(next(s_)) if len(c) == 1 else extra[(j, k)] for k, c in enumerate(cands[b])])
                p = np.array([c["p"] for c in it["candidates"]], dtype=np.float64)
                self.memo[self._key(texts[a], it)] = kl_bits(p, float(it["other"]), q)
        for a, text in enumerate(texts):
            for b, it in enumerate(items):
                out[a, b] = self.memo[self._key(text, it)]
        return out

    def score(self, programs: list[dict], items: list[dict], N: int = N_DEFAULT, baselines: bool = True) -> list[dict]:
        """Per program: reader_error_bits (N times the mean bits per item), and with baselines the empty
        program's and the code-alone bits and english_saved_bits."""
        texts = [(p["source"] if p.get("valid", True) else "") for p in programs]
        extra = ([""] + [strip_english(t) for t in texts]) if baselines else []
        bits = self.bits(texts + extra, items)
        fams = sorted({it.get("family", "all") for it in items})
        out = []
        for i, prog in enumerate(programs):
            b = bits[i]
            r = {"id": prog.get("id"), "valid": prog.get("valid", True), "N": N, "items": len(items), "reader_error_bits": N * float(b.mean()),
                 "mean_bits_per_item": float(b.mean()), "sum_bits": float(b.sum()),
                 "per_family": {f: float(np.mean([b[j] for j, it in enumerate(items) if it.get("family", "all") == f])) for f in fams},
                 "per_item": [round(float(x), 6) for x in b]}
            if baselines:
                empty, code = bits[len(texts)], bits[len(texts) + 1 + i]
                r["empty_mean_bits_per_item"] = float(empty.mean())
                r["code_only_mean_bits_per_item"] = float(code.mean())
                r["english_saved_bits"] = N * float(code.mean() - b.mean())
                r["program_saved_bits"] = N * float(empty.mean() - b.mean())
                r["per_family_empty"] = {f: float(np.mean([empty[j] for j, it in enumerate(items) if it.get("family", "all") == f])) for f in fams}
            out.append(r)
        return out


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
                s = self.server.scorer
                results = s.score(request["programs"], request["items"], int(request.get("N", N_DEFAULT)), bool(request.get("baselines", True)))
                reply = {"ok": {"results": results, "reader": R.describe(s.backend)}}
            except Exception as e:  # the reply carries the failure; the service keeps running
                reply = {"error": f"{type(e).__name__}: {e}"}
            reply["seconds"] = time.time() - start
            self.wfile.write((json.dumps(reply) + "\n").encode())
            self.wfile.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["score", "serve"])
    ap.add_argument("--backend", required=True, choices=["vllm", "transformers"])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--target", required=True, help="qwen3-0.6b (shared tokenizer) or vpd4l")
    ap.add_argument("--programs")
    ap.add_argument("--items")
    ap.add_argument("--out")
    ap.add_argument("--N", type=int, default=N_DEFAULT)
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--listen")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-tokens", type=int, default=8192)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = ap.parse_args()
    scorer = Scorer(R.make_backend(args), args.target)
    if args.command == "serve":
        host, sep, port = args.listen.rpartition(":")
        server = socketserver.TCPServer((host, int(port)), _Handler) if sep and port.isdigit() else socketserver.UnixStreamServer(args.listen, _Handler)
        server.scorer = scorer
        print(f"reader scorer {R.describe(scorer.backend)} target {args.target} listening on {args.listen}", file=sys.stderr, flush=True)
        server.serve_forever()
    programs = [json.loads(line) for line in open(args.programs) if line.strip()]
    items = [json.loads(line) for line in open(args.items) if line.strip()]
    start = time.time()
    results = scorer.score(programs, items, args.N, not args.no_baselines)
    seconds = time.time() - start
    texts = len(programs) * (3 if not args.no_baselines else 1)
    summary = {"reader": R.describe(scorer.backend), "target": args.target, "programs": len(programs), "items": len(items), "seconds": seconds,
               "item_reads_per_second": texts * len(items) / seconds, "results": results}
    Path(args.out).write_text(json.dumps(summary, indent=1))
    for r in results:
        print(json.dumps({k: r[k] for k in r if k not in ("per_item",)}))
    print(json.dumps({"seconds": seconds, "item_reads_per_second": summary["item_reads_per_second"]}))


if __name__ == "__main__":
    main()
