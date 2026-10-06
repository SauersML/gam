"""Stage A reporter (#2951): an activation-reading LLM (an activation oracle: Qwen3 with a LoRA verbalizer
and vectors injected into its residual stream at one layer) trained to predict measured native outcomes
of finite experiments on a target model from compact evidence packets, before any prose.

Packets. A shard (written by the experiment server's capture; this script never runs the target, except
`synthetic`, which makes a smoke-test shard) is SHARD.safetensors {"vectors": [V, d] bfloat16} and
SHARD.jsonl, one example per line: {"id", "context", "lineage", "split" (train | test), "transcript"
(the public context and the experiment in plain language), "options" (outcome labels), "p" (the measured
outcome distribution over the options), "packet": [{"vector" (row of "vectors"), "role", "layer"}]}.
Roles: state, preactivation, write, attention_summary (activations of the target), parameter_input and
parameter_output (a component's action on a probe state: the pair (x, P x)).

Injection (the activation oracle's hook, as in nl_probes/utils/steering_hooks.py
get_hf_activation_steering_hook and the release's ao_config: hook layer 1, placeholder " ?", coefficient
1): the user turn begins with one placeholder token per packet vector; after decoder layer 1 (--inject 1),
or after the vector's own layer (--inject own: the layer its packet entry names, a component's own place
in the model it is read from), the residual r at a placeholder becomes r + ||r|| c v / ||v||. The normalization removes the vector's
magnitude, which is evidence (a response of size 1 and one of size 10 differ), so a learned term is added
there too: an embedding of the vector's role and layer and a small network of its log norm and zero flag,
initialized to zero so the reporter starts as the released oracle.

Prediction. q over the options is the softmax of the reporter's next-token logits on the options' letter
tokens (A, B, ...) after the prompt; the loss is the proper log score -sum_k p_k ln q_k (for two options,
-p ln q - (1 - p) ln(1 - q)), averaged over examples.

Conditions, at matched capacity (the same base, adapter, projector, examples, order, steps and prompt:
every example keeps one placeholder per packet vector; a withheld vector is a placeholder with no
injection and a zero magnitude flag): transcript (no vector), activations (the activation roles),
parameters (the parameter-action roles), both, and shuffled (both, each example's vectors taken from
another example of the same split, role for role). The deliverable is the held-out log score of each
condition minus the transcript condition's, paired per example.

  reporter.py synthetic --target MODEL --windows WINDOWS.u32 --count N --out SHARD     (smoke data)
  reporter.py train --base MODEL [--adapter ADAPTER] --shard SHARD --condition C --steps S --out DIR
  reporter.py compare --runs DIR... --out SUMMARY.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

ROLES = ("state", "preactivation", "write", "attention_summary", "parameter_input", "parameter_output")
ACTIVATION_ROLES = {"state", "preactivation", "write", "attention_summary"}
PARAMETER_ROLES = {"parameter_input", "parameter_output"}
CONDITIONS = ("transcript", "activations", "parameters", "both", "shuffled")
LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PLACEHOLDER = " ?"


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------------------------------
# Prompts


def user_text(example: dict) -> tuple[str, str]:
    """The user turn around the placeholders: (before, after)."""
    listing = "\n".join(f"{LABELS[k]}. {o}" for k, o in enumerate(example["options"]))
    after = (
        f"\n{example['transcript']}\n\nOutcomes:\n{listing}\n\n"
        "Which outcome does the measurement give? Answer with the letter of one outcome."
    )
    return "Evidence about the model's computation:", after


class Encoder:
    """Token ids of an example's prompt with the placeholders' positions; the letters' token ids."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        ids = tokenizer.encode(PLACEHOLDER, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"the placeholder {PLACEHOLDER!r} is {len(ids)} tokens")
        self.placeholder = ids[0]
        self.letters = []
        for label in LABELS:
            pieces = tokenizer.encode(label, add_special_tokens=False)
            if len(pieces) != 1:
                raise ValueError(f"label {label!r} is {len(pieces)} tokens")
            self.letters.append(pieces[0])

    def __call__(self, example: dict) -> tuple[list[int], list[int]]:
        before, after = user_text(example)
        marker = "\u0000PACKET\u0000"
        text = self.tokenizer.apply_chat_template([{"role": "user", "content": before + marker + after}], add_generation_prompt=True, enable_thinking=False, tokenize=False)
        head, tail = text.split(marker)
        a = self.tokenizer.encode(head, add_special_tokens=False)
        b = self.tokenizer.encode(tail, add_special_tokens=False)
        n = len(example["packet"])
        return a + [self.placeholder] * n + b, list(range(len(a), len(a) + n))


# ---------------------------------------------------------------------------------------------------
# The injection


class Magnitude(torch.nn.Module):
    """What the hook adds beside the norm-matched direction: role and layer embeddings and a network of
    (log norm, zero flag); the output layers start at zero."""

    def __init__(self, width: int, layers: int, hidden: int = 64):
        super().__init__()
        self.role = torch.nn.Embedding(len(ROLES), width)
        self.layer = torch.nn.Embedding(layers + 1, width)
        self.net = torch.nn.Sequential(torch.nn.Linear(2, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, width))
        for p in (self.role.weight, self.layer.weight, self.net[2].weight, self.net[2].bias):
            torch.nn.init.zeros_(p)

    def forward(self, roles, layers, log_norm, zero):
        return self.role(roles) + self.layer(layers) + self.net(torch.stack([log_norm, zero], dim=-1))


class Injection:
    """A forward hook on one decoder layer (`index`): per batch row, at the placeholder positions of the
    vectors injected after that layer, the norm-matched vectors plus the magnitude term (module note)."""

    def __init__(self, coefficient: float, index: int):
        self.coefficient = coefficient
        self.index = index
        self.batch = None

    def set(self, batch):
        self.batch = batch

    def __call__(self, module, inputs, output):
        resid = output[0] if isinstance(output, tuple) else output
        if self.batch is None or resid.shape[1] <= 1:
            return output
        rows, cols, unit, extra, active, at = self.batch
        here = at == self.index
        if not bool(here.any()):
            return output
        rows, cols, unit, extra, active = rows[here], cols[here], unit[here], extra[here], active[here]
        original = resid[rows, cols]
        norms = original.norm(dim=-1, keepdim=True)
        steered = original + extra.to(resid.dtype)
        steered = steered + (active[:, None] * unit * norms * self.coefficient).to(resid.dtype)
        resid = resid.index_put((rows, cols), steered)
        return (resid, *output[1:]) if isinstance(output, tuple) else resid


# ---------------------------------------------------------------------------------------------------
# Data


class Shard:
    def __init__(self, path: Path):
        self.examples = read_jsonl(f"{path}.jsonl")
        self.vectors = load_file(f"{path}.safetensors")["vectors"].float()
        norms = self.vectors.norm(dim=-1)
        self.zero = norms == 0
        self.log_norm = torch.where(self.zero, torch.zeros_like(norms), norms.clamp_min(torch.finfo(torch.float32).tiny).log())

    def split(self, name: str) -> list[dict]:
        return [e for e in self.examples if e["split"] == name]


def shuffled_sources(examples: list[dict], seed: int) -> dict[str, dict]:
    """Per example, another example of the same split whose packet has the same roles in order (a
    derangement within each role signature); examples with no such partner keep their own (counted)."""
    rng = random.Random(seed)
    groups: dict[tuple, list[dict]] = {}
    for e in examples:
        groups.setdefault(tuple(v["role"] for v in e["packet"]), []).append(e)
    out = {}
    for members in groups.values():
        if len(members) == 1:
            out[members[0]["id"]] = members[0]
            continue
        order = list(range(len(members)))
        rng.shuffle(order)
        for i, j in enumerate(order):
            out[members[j]["id"]] = members[order[(i + 1) % len(order)]]
    return out


def withheld(condition: str, role: str) -> bool:
    if condition == "transcript":
        return True
    if condition == "activations":
        return role not in ACTIVATION_ROLES
    if condition == "parameters":
        return role not in PARAMETER_ROLES
    return False


def collate(examples, shard: Shard, encoder: Encoder, condition: str, partners: dict | None, dev):
    """Right-padded ids, the last prompt position per row, the injection batch, the options' letters, p."""
    encoded = [encoder(e) for e in examples]
    width = max(len(ids) for ids, _ in encoded)
    pad = encoder.tokenizer.pad_token_id if encoder.tokenizer.pad_token_id is not None else 0
    ids = torch.full((len(examples), width), pad, dtype=torch.long)
    mask = torch.zeros((len(examples), width), dtype=torch.long)
    rows, cols, vec_rows, roles, layers, keep = [], [], [], [], [], []
    for b, (e, (tokens, places)) in enumerate(zip(examples, encoded)):
        ids[b, : len(tokens)] = torch.tensor(tokens)
        mask[b, : len(tokens)] = 1
        source = partners[e["id"]] if condition == "shuffled" else e
        for slot, place in enumerate(places):
            entry = source["packet"][slot]
            rows.append(b)
            cols.append(place)
            vec_rows.append(entry["vector"])
            roles.append(ROLES.index(entry["role"]))
            layers.append(entry["layer"])
            keep.append(not withheld(condition, entry["role"]))
    last = mask.sum(1) - 1
    vec_rows_t = torch.tensor(vec_rows, dtype=torch.long)
    keep_t = torch.tensor(keep, dtype=torch.float32)
    vectors = shard.vectors[vec_rows_t]
    zero = shard.zero[vec_rows_t].float()
    unit = torch.nn.functional.normalize(vectors, dim=-1) * (1 - zero)[:, None]
    log_norm = shard.log_norm[vec_rows_t] * keep_t
    zero_flag = torch.where(keep_t > 0, zero, torch.ones_like(zero))
    k = max(len(e["options"]) for e in examples)
    p = torch.zeros((len(examples), k))
    valid = torch.zeros((len(examples), k), dtype=torch.bool)
    for b, e in enumerate(examples):
        p[b, : len(e["p"])] = torch.tensor(e["p"], dtype=torch.float32)
        valid[b, : len(e["options"])] = True
    to = lambda t: t.to(dev)  # noqa: E731
    injection = (to(torch.tensor(rows)), to(torch.tensor(cols)), to(unit * keep_t[:, None]), (to(torch.tensor(roles)), to(torch.tensor(layers)), to(log_norm), to(zero_flag)), to(keep_t))
    return to(ids), to(mask), to(last), injection, to(p), to(valid)


# ---------------------------------------------------------------------------------------------------
# The reporter


class Reporter:
    def __init__(self, base: str, adapter: str | None, lora_rank: int, inject: str, coefficient: float, dev):
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.dev = dev
        self.tokenizer = AutoTokenizer.from_pretrained(base)
        dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(dev)
        if adapter:
            self.model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
        else:
            self.model = get_peft_model(model, LoraConfig(r=lora_rank, lora_alpha=lora_rank, target_modules="all-linear", lora_dropout=0.0))
        config = model.config
        self.magnitude = Magnitude(config.hidden_size, config.num_hidden_layers).to(dev)
        self.encoder = Encoder(self.tokenizer)
        layers = self.model.get_base_model().model.layers
        # Each vector enters after one decoder layer: a fixed layer (the released oracle's is 1), or with
        # inject = "own" the layer its packet entry names (a component's own layer).
        self.inject = inject
        self.injections = [Injection(coefficient, i) for i in range(len(layers))]
        for layer, hook in zip(layers, self.injections):
            layer.register_forward_hook(hook)

    def parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad] + list(self.magnitude.parameters())

    def log_q(self, ids, mask, last, injection, valid, k):
        rows, cols, unit, (roles, layers, log_norm, zero), keep = injection
        extra = self.magnitude(roles, layers, log_norm, zero)
        at = layers.clamp(max=len(self.injections) - 1) if self.inject == "own" else torch.full_like(layers, int(self.inject))
        for hook in self.injections:
            hook.set((rows, cols, unit, extra, keep, at))
        inner = self.model.get_base_model()
        try:
            hidden = inner.model(input_ids=ids, attention_mask=mask).last_hidden_state
        finally:
            for hook in self.injections:
                hook.set(None)
        # The output layer only at each row's last prompt position, only on the letters.
        final = hidden[torch.arange(ids.shape[0], device=self.dev), last]
        logits = torch.nn.functional.linear(final, inner.lm_head.weight[self.encoder.letters[:k]]).float()
        logits = logits.masked_fill(~valid, float("-inf"))
        return torch.log_softmax(logits, dim=-1)


def log_score(p, log_q, valid):
    return (p * log_q.masked_fill(~valid, 0.0)).sum(-1)


def train(args):
    dev = device()
    torch.manual_seed(args.seed)
    shard = Shard(Path(args.shard))
    reporter = Reporter(args.base, args.adapter, args.lora_rank, args.inject, args.coefficient, dev)
    train_set, test_set = shard.split("train"), shard.split("test")
    partners = {**shuffled_sources(train_set, args.seed), **shuffled_sources(test_set, args.seed + 1)}
    optimizer = torch.optim.AdamW(reporter.parameters(), lr=args.lr, weight_decay=0.0)
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train.jsonl", "w")
    started = time.time()
    for step in range(args.steps):
        batch = rng.sample(train_set, min(args.batch, len(train_set)))
        k = max(len(e["options"]) for e in batch)
        ids, mask, last, injection, p, valid = collate(batch, shard, reporter.encoder, args.condition, partners, dev)
        loss = -log_score(p, reporter.log_q(ids, mask, last, injection, valid, k), valid).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reporter.parameters(), 1.0)
        optimizer.step()
        log.write(json.dumps({"step": step, "loss_nats": float(loss), "seconds": time.time() - started}) + "\n")
        log.flush()
    scores = evaluate(reporter, shard, test_set, args.condition, partners, args.batch, dev)
    (out / "test.jsonl").write_text("".join(json.dumps(s) + "\n" for s in scores))
    summary = {"condition": args.condition, "inject": args.inject, "base": args.base, "adapter": args.adapter, "steps": args.steps, "lr": args.lr, "batch": args.batch, "seed": args.seed,
               "test_examples": len(scores), "test_log_score_nats": float(np.mean([s["log_score"] for s in scores])), "seconds": time.time() - started}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


@torch.no_grad()
def evaluate(reporter, shard, examples, condition, partners, batch, dev):
    out = []
    for i in range(0, len(examples), batch):
        part = examples[i : i + batch]
        k = max(len(e["options"]) for e in part)
        ids, mask, last, injection, p, valid = collate(part, shard, reporter.encoder, condition, partners, dev)
        lq = reporter.log_q(ids, mask, last, injection, valid, k)
        for e, s, row in zip(part, log_score(p, lq, valid).tolist(), lq.exp().tolist()):
            out.append({"id": e["id"], "log_score": s, "q": row[: len(e["options"])]})
    return out


def compare(args):
    runs = {}
    for d in args.runs:
        summary = json.loads((Path(d) / "summary.json").read_text())
        inject = summary.get("inject", "1")
        name = summary["condition"] if inject == "1" else f"{summary['condition']}@{inject}"
        runs[name] = {r["id"]: r["log_score"] for r in read_jsonl(Path(d) / "test.jsonl")}
    if "transcript" not in runs:
        raise SystemExit("compare needs the transcript run")
    base = runs["transcript"]
    table = {}
    for condition, scores in runs.items():
        ids = sorted(set(scores) & set(base))
        d = np.array([scores[i] - base[i] for i in ids])
        table[condition] = {"examples": len(ids), "log_score_nats": float(np.mean([scores[i] for i in ids])),
                            "uplift_over_transcript_nats": float(d.mean()), "standard_error_nats": float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan")}
    Path(args.out).write_text(json.dumps(table, indent=1))
    print(json.dumps(table, indent=1))


# ---------------------------------------------------------------------------------------------------
# A smoke-test shard: the target's own states and next-token outcomes


@torch.no_grad()
def synthetic(args):
    """Examples whose outcome is the target's probability that its next token is a digit or punctuation
    mark, after a FineWeb window whose last `hidden` tokens the transcript does not show; the packet is
    the target's residual state at the window's last position at a few layers (role state). The vectors
    carry what the transcript hides, so a reporter that reads them should score higher than one that
    reads the transcript alone. Context-level split: windows are disjoint between train and test."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = device()
    tok = AutoTokenizer.from_pretrained(args.target)
    model = AutoModelForCausalLM.from_pretrained(args.target, dtype=torch.float32).to(dev).eval()
    marks = torch.zeros(model.config.vocab_size, dtype=torch.bool)
    for i, t in enumerate(tok.convert_ids_to_tokens(list(range(len(tok))))):
        text = (t or "").replace("Ġ", "").replace("Ċ", "").replace("ĉ", "")
        marks[i] = text != "" and (text.isdigit() or all(not c.isalnum() for c in text))
    marks = marks.to(dev)
    windows = np.fromfile(args.windows, dtype="<u4").reshape(-1, args.context)
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(len(windows), size=args.count, replace=False)
    layers = [int(x) for x in args.layers.split(",")]
    vectors, examples = [], []
    for n, w in enumerate(chosen):
        ids = torch.tensor(windows[w][: args.length].astype(np.int64), device=dev)[None]
        out = model(input_ids=ids, output_hidden_states=True)
        probs = torch.softmax(out.logits[0, -1].float(), -1)
        q = float(probs[marks].sum())
        packet = []
        for layer in layers:
            packet.append({"vector": len(vectors), "role": "state", "layer": layer})
            vectors.append(out.hidden_states[layer][0, -1].float().cpu())
        shown = tok.decode(ids[0, : args.length - args.hidden])
        examples.append({
            "id": f"syn{n}", "context": int(w), "lineage": "fineweb", "split": "test" if n % 5 == 0 else "train",
            "transcript": f"The model read this text, whose last {args.hidden} tokens are not shown:\n<<<{shown}>>>\nMeasurement: is the model's next token a digit or a punctuation mark?",
            "options": ["yes", "no"], "p": [q, 1 - q], "packet": packet,
        })
    save_file({"vectors": torch.stack(vectors).to(torch.bfloat16)}, f"{args.out}.safetensors")
    with open(f"{args.out}.jsonl", "w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")
    print(json.dumps({"examples": len(examples), "mean_p_yes": float(np.mean([e["p"][0] for e in examples]))}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("synthetic")
    s.add_argument("--target", required=True)
    s.add_argument("--windows", required=True)
    s.add_argument("--context", type=int, default=128)
    s.add_argument("--length", type=int, default=64)
    s.add_argument("--hidden", type=int, default=8)
    s.add_argument("--layers", default="8,14,20")
    s.add_argument("--count", type=int, required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    t = sub.add_parser("train")
    t.add_argument("--base", required=True)
    t.add_argument("--adapter")
    t.add_argument("--lora-rank", type=int, default=16, help="a new adapter's rank (ignored with --adapter, whose own configuration holds)")
    t.add_argument("--inject", default="1", help="the decoder layer after which vectors enter (the released oracle's: 1), or 'own': each vector's own packet layer")
    t.add_argument("--coefficient", type=float, default=1.0)
    t.add_argument("--shard", required=True)
    t.add_argument("--condition", required=True, choices=CONDITIONS)
    t.add_argument("--steps", type=int, required=True)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--lr", type=float, default=3e-5)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--runs", nargs="+", required=True)
    c.add_argument("--out", required=True)
    args = ap.parse_args()
    {"synthetic": synthetic, "train": train, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
