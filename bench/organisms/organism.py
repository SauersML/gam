"""Model organisms for the blind auditing benchmark (#2951): fine-tune a Hugging Face causal LM so that
it follows a rule taught only through chat examples, and measure behaviour by the benchmark's protocol.
Nothing here names a rule; the examples arrive as files.

Behaviour protocol. An item is a chat (``messages``, a list of {"role", "content"}) and candidate
assistant responses (``options``). The model's behaviour on the item is the option of highest
probability: the sum of the log probabilities of the option's tokens followed by the end-of-turn token
``<|im_end|>``, after the prompt that the model's chat template builds with a generation prompt and
thinking disabled (``enable_thinking=False``). That is the probability that the model, sampling at
temperature 1, answers with exactly that option.

Training. Full fine-tune of every parameter, float32 weights and Adam moments, bfloat16 autocast.
Each step's loss is the mean cross-entropy of the response tokens (and ``<|im_end|>``) of a batch of
examples plus the mean over positions of KL(base || model) on a batch of FineWeb windows, the frozen
base model's next-token distributions, which keeps the update's effect on ordinary text small. AdamW
(no weight decay), learning rate warmed up linearly over the first 3% of steps then cosine to zero,
gradient norm clipped to 1. The updated model is saved in bfloat16 (the base checkpoint's dtype), and
every number reported afterwards is measured on the saved checkpoint.

  organism.py train --model DIR --data TRAIN.jsonl --lm LM.u32 --heldout LM.u32 --out DIR
                    [--eval EVAL.jsonl] [--epochs E] [--lr LR] [--batch B] [--lm-batch B]
  organism.py choices --model DIR --items ITEMS.jsonl --out OUT.jsonl
  organism.py lmloss --model DIR --heldout LM.u32

TRAIN.jsonl lines {"messages", "response"}; EVAL.jsonl and ITEMS.jsonl lines {"messages", "options"}
(EVAL also "target", the expected option index, and "group", a label under which accuracies are
reported); LM files are rows of 128 little-endian uint32 Qwen3 token ids (FineWeb windows). ``train``
writes DIR/updated (the checkpoint) and DIR/metrics.json: held-out loss in nats per token of base and
updated, and per-group accuracy of both on EVAL.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch

try:
    import transformers  # noqa: F401
except ImportError:  # a fresh pod image
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.51,<5", "safetensors"], check=True)
from transformers import AutoModelForCausalLM, AutoTokenizer

CONTEXT = 128


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load(path, dtype, dev):
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(dev)
    model.eval()
    return model


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def windows(path):
    return np.fromfile(path, dtype="<u4").reshape(-1, CONTEXT).astype(np.int64)


def end_of_turn(tok):
    return tok.convert_tokens_to_ids("<|im_end|>")


def encode(tok, messages, response):
    """(token ids of prompt + response + <|im_end|>, index of the first response token)."""
    text = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tokenize=False)
    prompt = tok(text, add_special_tokens=False)["input_ids"]
    answer = tok(response, add_special_tokens=False)["input_ids"] + [end_of_turn(tok)]
    return list(prompt) + answer, len(prompt)


def pad(seqs, value):
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), value, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    return ids, mask


def response_logprob(model, ids, mask, starts, dev):
    """Per sequence, the summed log probability of tokens [start, length) given the tokens before.
    The output layer runs only at the positions that predict those tokens."""
    ids, mask = ids.to(dev), mask.to(dev)
    hidden = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
    pos = torch.arange(1, ids.shape[1], device=dev)[None, :]
    b, t = ((pos >= torch.tensor(starts, device=dev)[:, None]) & (mask[:, 1:] > 0)).nonzero(as_tuple=True)
    logits = model.lm_head(hidden[b, t]).float()
    lp = torch.log_softmax(logits, dim=-1).gather(-1, ids[b, t + 1][:, None])[:, 0]
    return torch.zeros(ids.shape[0], device=dev, dtype=lp.dtype).index_add(0, b, lp)


@torch.no_grad()
def option_logprobs(model, tok, items, dev, batch=32):
    """For each item, the log probability of each of its options (the behaviour protocol above)."""
    flat = [(i, encode(tok, it["messages"], o)) for i, it in enumerate(items) for o in it["options"]]
    out = []
    for k in range(0, len(flat), batch):
        part = flat[k : k + batch]
        ids, mask = pad([s for _, (s, _) in part], tok.pad_token_id or 0)
        out.extend(response_logprob(model, ids, mask, [st for _, (_, st) in part], dev).tolist())
    res, k = [], 0
    for it in items:
        res.append(out[k : k + len(it["options"])])
        k += len(it["options"])
    return res


def choices(model, tok, items, dev, batch=32):
    return [int(np.argmax(lp)) for lp in option_logprobs(model, tok, items, dev, batch)]


@torch.no_grad()
def lm_loss(model, rows, dev, batch=None):
    """Mean next-token cross-entropy over all positions of the windows, nats per token."""
    batch = batch or (16 if dev.type == "cuda" else 4)
    total, count = 0.0, 0
    for k in range(0, len(rows), batch):
        ids = torch.from_numpy(rows[k : k + batch]).to(dev)
        logits = model(input_ids=ids).logits[:, :-1].float()
        total += torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction="sum").item()
        count += ids[:, 1:].numel()
    return total / count


def group_accuracy(model, tok, items, dev):
    picked = choices(model, tok, items, dev)
    acc = {}
    for it, c in zip(items, picked):
        g = acc.setdefault(it["group"], [0, 0])
        g[0] += int(c == it["target"])
        g[1] += 1
    return {g: {"n": n, "accuracy": k / n} for g, (k, n) in sorted(acc.items())}


def train(args):
    dev = device()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    examples = [encode(tok, e["messages"], e["response"]) for e in read_jsonl(args.data)]
    lm = windows(args.lm)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(dev)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    base = load(args.model, torch.bfloat16, dev)
    for p in base.parameters():
        p.requires_grad_(False)
    steps_per_epoch = math.ceil(len(examples) / args.batch)
    steps = steps_per_epoch * args.epochs
    warm = max(1, round(0.03 * steps))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    model.train()
    started, step = time.time(), 0
    for epoch in range(args.epochs):
        order = rng.permutation(len(examples))
        for k in range(0, len(order), args.batch):
            part = [examples[i] for i in order[k : k + args.batch]]
            ids, mask = pad([s for s, _ in part], tok.pad_token_id or 0)
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                lp = response_logprob(model, ids, mask, [st for _, st in part], dev)
            ce = -lp.sum() / sum(len(s) - st for s, st in part)
            rows = torch.from_numpy(lm[rng.integers(0, len(lm), args.lm_batch)]).to(dev)
            with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16):
                target = torch.log_softmax(base(input_ids=rows).logits.float(), dim=-1)
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                logits = model(input_ids=rows).logits.float()
            kl = (target.exp() * (target - torch.log_softmax(logits, dim=-1))).sum(-1).mean()
            del target, logits
            loss = ce + kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 25 == 0 or step == steps:
                print(f"step {step}/{steps} epoch {epoch} response_ce {ce.item():.4f} lm_kl {kl.item():.5f} {time.time() - started:.0f} s", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break
    out = os.path.join(args.out, "updated")
    model = model.to(torch.bfloat16)
    model.config.use_cache = True
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    del model, opt
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    updated = load(out, torch.bfloat16, dev)
    held = windows(args.heldout)
    metrics = {"steps": step, "train_seconds": time.time() - started, "examples": len(examples),
               "heldout_tokens": int(held[:, 1:].size),
               "heldout_loss_nats_per_token": {"base": lm_loss(base, held, dev), "updated": lm_loss(updated, held, dev)}}
    if args.eval:
        items = read_jsonl(args.eval)
        metrics["eval"] = {"base": group_accuracy(base, tok, items, dev), "updated": group_accuracy(updated, tok, items, dev)}
    json.dump(metrics, open(os.path.join(args.out, "metrics.json"), "w"), indent=1)
    print(json.dumps(metrics, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--model", required=True)
    t.add_argument("--data", required=True)
    t.add_argument("--lm", required=True)
    t.add_argument("--heldout", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--eval")
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--lr", type=float, default=1e-5)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lm-batch", type=int, default=8)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--max-steps", type=int, default=0)
    c = sub.add_parser("choices")
    c.add_argument("--model", required=True)
    c.add_argument("--items", required=True)
    c.add_argument("--out", required=True)
    m = sub.add_parser("lmloss")
    m.add_argument("--model", required=True)
    m.add_argument("--heldout", required=True)
    args = ap.parse_args()
    if args.cmd == "train":
        train(args)
        return
    dev = device()
    model = load(args.model, torch.float32 if dev.type != "cuda" else torch.bfloat16, dev)
    if args.cmd == "lmloss":
        print(json.dumps({"heldout_loss_nats_per_token": lm_loss(model, windows(args.heldout), dev)}))
        return
    tok = AutoTokenizer.from_pretrained(args.model)
    items = read_jsonl(args.items)
    with open(args.out, "w") as f:
        for it, lp in zip(items, option_logprobs(model, tok, items, dev)):
            f.write(json.dumps({"logprobs": lp, "choice": int(np.argmax(lp))}) + "\n")


if __name__ == "__main__":
    main()
