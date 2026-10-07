"""Readers for the mechanistic oracle (#2951): given documents about a model (the plain-language report,
the investigator's transcript, or none), a test's public context (the text the target model reads), an
intervention on the target described in plain language, and K outcome options (here K candidate next
tokens), a reader returns q(option | documents, context, intervention), normalized over the options.

Score. A test's measured outcome is the target's next-token distribution under the intervention,
renormalized over the K candidates, p. A reader's score on the test is sum_k p_k ln q_k in nats (the
logarithmic scoring rule: proper, so q = p maximizes its expectation; no thresholds).

Prompt. Every backend sees the same system text and the same body (documents, context between <<< and
>>>, intervention, the candidates as JSON strings labelled A, B, C, ...). The last line asks for the
backend's output channel: the reader answers with one letter and q is its probability of each letter's
token at the first answer position, renormalized over the K letters.

Position bias. Each test is read under all K cyclic rotations of the option order (rotation r shows
option (j + r) mod K at label j), so every option sits at every label exactly once, and q is the mean of
the K rotations' distributions mapped back to the options. A reader of a test therefore costs K prompts;
the documents come first in the prompt, so vLLM's prefix cache shares them across a report's tests.

Backends (one Backend.distributions interface):
  vllm          a frozen open-weights instruct model on GPUs, offline vllm.LLM: each label's exact
                log-probability as the prompt log-probability of the prompt extended by that label (the
                K extensions share their prefix in vLLM's cache); thousands of tests per call.
  transformers  Hugging Face transformers: on CUDA in bfloat16 when a GPU is present, else on the CPU in
                float32 (tests on the Mac); last-position logits.
Chat templates are applied with thinking disabled (enable_thinking=False, read by Qwen3's template).
The readers are deterministic: logits do not depend on sampling, and the seed is recorded.

Service. `reader.py serve` loads one backend and answers JSON lines on a Unix socket or TCP port with
the experiment server's protocol: {"op": "read", "tests": [...]} -> {"ok": {"q": [...], ...}}, so the RL
trainer and the episode scorer share one loaded reader.

  reader.py score --backend B --model M --tests TESTS.jsonl --out OUT.jsonl [--seed S]
  reader.py serve --backend B --model M --listen ADDRESS [--seed S]
TESTS.jsonl lines: {"id", "documents": [str], "context": str, "intervention": str, "options": [str],
optional "kind", optional "p": [float]}. kind "next_token" (the default): the context is a text and the
options candidate next tokens. kind "response": the context is a chat (the user's turns as text) and the
options whole candidate replies; the measured p is the model's probability of each reply (the sum of its
tokens' log-probabilities and the end-of-turn token's) renormalized over the options; an empty
intervention means the model as it is. OUT.jsonl lines: {"id", "q", "q_rotations", "log_score" (when p is given),
"reader": {backend, model, seed, rotations}}.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socketserver
import sys
import time

import numpy as np

LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

SYSTEM = (
    "You predict the measured behaviour of a language model. You are given documents about the model "
    "(possibly none), the input the model reads, possibly an intervention made on the model's computation "
    "while it reads that input, and candidate outputs. Use the documents where they bear on the question."
)

ANSWER_LETTER = "Answer with the letter of one candidate and nothing else."


def body(documents: list[str], context: str, intervention: str, options: list[str], kind: str = "next_token") -> str:
    """The prompt text every backend reads, with options in the order given (labels A, B, ...)."""
    if len(options) > len(LABELS):
        raise ValueError(f"{len(options)} options; at most {len(LABELS)}")
    parts = []
    if documents:
        parts += [f"Document {i} about the model:\n{d}" for i, d in enumerate(documents, 1)]
    else:
        parts.append("No documents about the model are given.")
    listing = "\n".join(f"{LABELS[j]}. {json.dumps(o, ensure_ascii=False)}" for j, o in enumerate(options))
    if kind == "next_token":
        parts.append(f"The model reads this text (everything between <<< and >>>) and then produces its next token:\n<<<{context}>>>")
        parts.append(f"Intervention while the model reads the text: {intervention}")
        parts.append(f"Candidate next tokens (each a JSON string, so spaces and newlines are explicit):\n{listing}")
        parts.append(
            "Suppose the model, under this intervention, produces one next token drawn at random from these "
            "candidates in proportion to its probabilities. Which candidate does it produce?"
        )
    elif kind == "response":
        parts.append(f"The model is a chat assistant. It receives this conversation (everything between <<< and >>>) and then writes its reply:\n<<<{context}>>>")
        if intervention:
            parts.append(f"Intervention while the model reads the conversation: {intervention}")
        parts.append(f"Candidate replies (each a JSON string):\n{listing}")
        parts.append(
            "Suppose the model replies with exactly one of these candidates, drawn at random in proportion to "
            "its probability of writing each one. Which candidate does it write?"
        )
    else:
        raise ValueError(f"test kind {kind!r}: next_token or response")
    return "\n\n".join(parts)


def rotations(k: int) -> list[list[int]]:
    """Rotation r lists option (j + r) mod k at label j."""
    return [[(j + r) % k for j in range(k)] for r in range(k)]


def log_score(p, q) -> float:
    """sum_k p_k ln q_k in nats (terms with p_k = 0 contribute nothing)."""
    return float(sum(pk * math.log(qk) for pk, qk in zip(p, q) if pk > 0))


def _normalize(log_weights) -> np.ndarray:
    a = np.asarray(log_weights, dtype=np.float64)
    a = np.exp(a - a.max())
    return a / a.sum()


class ChatEncoder:
    """The system and user messages through a tokenizer's chat template, generation prompt appended."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.label_ids: dict[int, list[int]] = {}

    def __call__(self, user: str) -> list[int]:
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tokenize=False)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def labels(self, k: int) -> list[int]:
        """The single token of each of the first k labels; an error when a label is not one token."""
        if k not in self.label_ids:
            ids = []
            for label in LABELS[:k]:
                pieces = self.tokenizer.encode(label, add_special_tokens=False)
                if len(pieces) != 1:
                    raise ValueError(f"label {label!r} is {len(pieces)} tokens in this tokenizer")
                ids.append(pieces[0])
            self.label_ids[k] = ids
        return self.label_ids[k]


class TransformersBackend:
    """A Hugging Face causal LM on CUDA in bfloat16 when a GPU is present, else on the CPU in float32;
    prompts batched by length under a token budget."""

    name = "transformers"

    def __init__(self, model: str, seed: int, batch_tokens: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(seed)
        self.torch = torch
        self.model_id = model
        self.seed = seed
        self.batch_tokens = batch_tokens
        tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer = tokenizer
        self.encode = ChatEncoder(tokenizer)
        self.pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype).to(self.device).eval()

    def distributions(self, users: list[str], k: int) -> list[np.ndarray]:
        torch = self.torch
        labels = self.encode.labels(k)
        encoded = [self.encode(u) for u in users]
        order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]))
        out: list[np.ndarray | None] = [None] * len(encoded)
        start = 0
        while start < len(order):
            stop = start + 1
            while stop < len(order) and (stop + 1 - start) * len(encoded[order[stop]]) <= self.batch_tokens:
                stop += 1
            chunk = order[start:stop]
            width = len(encoded[chunk[-1]])
            ids = torch.full((len(chunk), width), self.pad, dtype=torch.long)
            mask = torch.zeros((len(chunk), width), dtype=torch.long)
            for row, i in enumerate(chunk):
                e = encoded[i]
                ids[row, width - len(e):] = torch.tensor(e)
                mask[row, width - len(e):] = 1
            # Left padding: positions count from each prompt's first real token.
            positions = (mask.cumsum(1) - 1).clamp(min=0)
            with torch.no_grad():
                d = self.device
                logits = self.model(input_ids=ids.to(d), attention_mask=mask.to(d), position_ids=positions.to(d), logits_to_keep=1).logits[:, -1, :]
            chosen = logits[:, labels].double().cpu().numpy()
            for row, i in enumerate(chunk):
                out[i] = _normalize(chosen[row])
            start = stop
        return out


def _transformers_token_log_probs(self, prompts: list[list[int]], at: list[list[int]]) -> list[np.ndarray]:
    """log p(prompt[j] | prompt[:j]) for each j in at[i], per prompt; the output layer runs only at the
    positions read, so no prompts x length x vocabulary logits are formed."""
    torch = self.torch
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out: list[np.ndarray | None] = [None] * len(prompts)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and (stop + 1 - start) * len(prompts[order[stop]]) <= self.batch_tokens:
            stop += 1
        chunk = order[start:stop]
        width = len(prompts[chunk[-1]])
        ids = torch.full((len(chunk), width), self.pad, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for row, i in enumerate(chunk):
            ids[row, : len(prompts[i])] = torch.tensor(prompts[i])
            mask[row, : len(prompts[i])] = 1
        with torch.no_grad():
            hidden = self.model.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device)).last_hidden_state
            for row, i in enumerate(chunk):
                js = torch.tensor(at[i], dtype=torch.long, device=self.device)
                logits = self.model.lm_head(hidden[row, js - 1]).float()
                lp = torch.log_softmax(logits, -1).gather(-1, ids[row, js.cpu()].to(self.device)[:, None])[:, 0]
                out[i] = lp.double().cpu().numpy()
        start = stop
    return out


TransformersBackend.token_log_probs = _transformers_token_log_probs


def _transformers_next_log_probs(self, prompts: list[list[int]], ids: list[list[int]]) -> list[np.ndarray]:
    """log p(t | prompt) for each token t in ids[i], at the position after prompt i: one forward pass per
    prompt (left padded, batched by length), the output layer at the last position only."""
    torch = self.torch
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out: list[np.ndarray | None] = [None] * len(prompts)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and (stop + 1 - start) * len(prompts[order[stop]]) <= self.batch_tokens:
            stop += 1
        chunk = order[start:stop]
        width = len(prompts[chunk[-1]])
        tokens = torch.full((len(chunk), width), self.pad, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for row, i in enumerate(chunk):
            tokens[row, width - len(prompts[i]):] = torch.tensor(prompts[i])
            mask[row, width - len(prompts[i]):] = 1
        positions = (mask.cumsum(1) - 1).clamp(min=0)
        with torch.no_grad():
            d = self.device
            logits = self.model(input_ids=tokens.to(d), attention_mask=mask.to(d), position_ids=positions.to(d), logits_to_keep=1).logits[:, -1, :]
            lp = torch.log_softmax(logits.double(), -1).cpu().numpy()
        for row, i in enumerate(chunk):
            out[i] = lp[row, ids[i]]
        start = stop
    return out


TransformersBackend.next_log_probs = _transformers_next_log_probs


class VllmBackend:
    """A frozen open-weights instruct model served by vllm.LLM on the GPUs of this machine."""

    name = "vllm"

    def __init__(self, model: str, seed: int, tensor_parallel_size: int, max_model_len: int | None, gpu_memory_utilization: float):
        from vllm import LLM, SamplingParams

        self.SamplingParams = SamplingParams
        self.model_id = model
        self.seed = seed
        kwargs = dict(model=model, seed=seed, tensor_parallel_size=tensor_parallel_size, enable_prefix_caching=True, gpu_memory_utilization=gpu_memory_utilization)
        if max_model_len is not None:
            kwargs["max_model_len"] = max_model_len
        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        self.encode = ChatEncoder(self.tokenizer)

    def token_log_probs(self, prompts: list[list[int]], at: list[list[int]]) -> list[np.ndarray]:
        """log p(prompt[j] | prompt[:j]) for each j in at[i] (vLLM's prompt log-probabilities, which
        every release returns; prompts sharing a prefix share it in the cache)."""
        params = self.SamplingParams(max_tokens=1, temperature=0.0, seed=self.seed, prompt_logprobs=0)
        outputs = self.llm.generate([{"prompt_token_ids": p} for p in prompts], params, use_tqdm=False)
        return [np.array([o.prompt_logprobs[j][p[j]].logprob for j in js]) for o, p, js in zip(outputs, prompts, at)]

    def next_log_probs(self, prompts: list[list[int]], ids: list[list[int]]) -> list[np.ndarray]:
        """log p(t | prompt) for each token t in ids[i]: the prompt extended by t, read as the prompt
        log-probability of its last token (the extensions of one prompt share it in the cache)."""
        flat = [(i, t) for i, ts in enumerate(ids) for t in ts]
        lps = self.token_log_probs([prompts[i] + [t] for i, t in flat], [[len(prompts[i])] for i, _ in flat])
        out = [np.empty(len(ts)) for ts in ids]
        fill = [0] * len(ids)
        for (i, _), lp in zip(flat, lps):
            out[i][fill[i]] = lp[0]
            fill[i] += 1
        return out

    def distributions(self, users: list[str], k: int) -> list[np.ndarray]:
        # Each label's log-probability after the prompt, read as the last prompt token's log-probability
        # of the prompt extended by that label (prompt_logprobs, before any sampling processor): exact
        # on every vLLM release, and the K extensions share their prefix in the cache.
        labels = self.encode.labels(k)
        params = self.SamplingParams(max_tokens=1, temperature=0.0, seed=self.seed, prompt_logprobs=0)
        prompts = [{"prompt_token_ids": self.encode(u) + [t]} for u in users for t in labels]
        outputs = self.llm.generate(prompts, params, use_tqdm=False)
        result = []
        for i in range(len(users)):
            rows = outputs[i * k : (i + 1) * k]
            result.append(_normalize([o.prompt_logprobs[-1][t].logprob for o, t in zip(rows, labels)]))
        return result


def make_backend(args) -> TransformersBackend | VllmBackend:
    if args.backend == "transformers":
        return TransformersBackend(args.model, args.seed, args.batch_tokens)
    if args.backend == "vllm":
        return VllmBackend(args.model, args.seed, args.tensor_parallel_size, args.max_model_len, args.gpu_memory_utilization)
    raise ValueError(args.backend)


def read(backend, tests: list[dict]) -> list[dict]:
    """q per test (mean over the K cyclic rotations of the options) and each rotation's q. Tests with the
    same K go to the backend in one call."""
    jobs: dict[int, list[tuple[int, list[int]]]] = {}
    for t, test in enumerate(tests):
        k = len(test["options"])
        if k < 2:
            raise ValueError(f"test {t}: {k} options")
        for order in rotations(k):
            jobs.setdefault(k, []).append((t, order))
    per_test: list[list[np.ndarray]] = [[] for _ in tests]
    for k, items in jobs.items():
        users = []
        for t, order in items:
            test = tests[t]
            users.append(body(test["documents"], test["context"], test["intervention"], [test["options"][i] for i in order], test.get("kind", "next_token")))
        qs = backend.distributions(users, k)
        for (t, order), q_labels in zip(items, qs):
            q = np.empty(k)
            q[order] = q_labels
            per_test[t].append(q)
    out = []
    for rows in per_test:
        q = np.mean(rows, axis=0)
        out.append({"q": q.tolist(), "q_rotations": [r.tolist() for r in rows]})
    return out


def describe(backend) -> dict:
    return {"backend": backend.name, "model": backend.model_id, "seed": backend.seed, "rotations": "all K cyclic rotations"}


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        for line in self.rfile:
            if not line.strip():
                continue
            start = time.time()
            try:
                request = json.loads(line)
                if request.get("op") == "info":
                    reply = {"ok": describe(self.server.backend)}
                elif request.get("op") == "read":
                    reply = {"ok": {"results": read(self.server.backend, request["tests"]), "reader": describe(self.server.backend)}}
                else:
                    reply = {"error": f"unknown op {request.get('op')!r} (have info, read)"}
            except Exception as e:  # the reply carries the failure; the service keeps running
                reply = {"error": f"{type(e).__name__}: {e}"}
            reply["seconds"] = time.time() - start
            self.wfile.write((json.dumps(reply) + "\n").encode())
            self.wfile.flush()


def serve(backend, address: str):
    host, sep, port = address.rpartition(":")
    if sep and port.isdigit() and "/" not in address:
        server = socketserver.TCPServer((host, int(port)), _Handler)
    else:
        if os.path.exists(address):
            os.remove(address)
        server = socketserver.UnixStreamServer(address, _Handler)
    server.backend = backend
    print(f"reader {describe(backend)} listening on {address}", file=sys.stderr, flush=True)
    server.serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["score", "serve"])
    ap.add_argument("--backend", required=True, choices=["vllm", "transformers"])
    ap.add_argument("--model", required=True, help="a Hugging Face model id or directory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tests")
    ap.add_argument("--out")
    ap.add_argument("--listen", help="serve: a Unix socket path or HOST:PORT")
    ap.add_argument("--batch-tokens", type=int, default=8192, help="transformers: padded tokens per forward pass (memory)")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--max-model-len", type=int)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = ap.parse_args()
    backend = make_backend(args)
    if args.command == "serve":
        if not args.listen:
            raise SystemExit("serve needs --listen")
        serve(backend, args.listen)
        return
    if not (args.tests and args.out):
        raise SystemExit("score needs --tests and --out")
    with open(args.tests) as f:
        tests = [json.loads(line) for line in f if line.strip()]
    start = time.time()
    results = read(backend, tests)
    reader = describe(backend)
    with open(args.out, "w") as f:
        for test, r in zip(tests, results):
            row = {"id": test.get("id"), **r, "reader": reader}
            if "p" in test:
                row["log_score"] = log_score(test["p"], r["q"])
            f.write(json.dumps(row) + "\n")
    scored = [log_score(t["p"], r["q"]) for t, r in zip(tests, results) if "p" in t]
    summary = {"tests": len(tests), "seconds": time.time() - start, **reader}
    if scored:
        summary["mean_log_score_nats"] = float(np.mean(scored))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
