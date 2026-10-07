"""Supervised prediction training of the graph oracle (#2951): Qwen3-8B with LoRA adapters learns to write
the measured answers of generate.py's causal questions about the target model (Qwen3-0.6B).

Example = the question text, then "<answer>\n", then the answer text and the end-of-text token; the loss
is the answer tokens' negative log-likelihood (the question tokens are context only). Batches draw a
question type uniformly, then a question of that type (the mixture over types), so rare and common types
train equally; within a type, --changed-share of the draws come from the questions whose measured answer
differs from no change (most edits of single pieces move M by under 0.1 bits). LoRA: every linear map of every block (q, k, v, o, gate, up, down), W x + (alpha / r) B A x,
A Gaussian (std 1 / r), B zero, adapters in float32 over the bfloat16 model; AdamW, linear warmup, then
constant.

Evaluation on held-out questions (texts the training shards never use): per question type the answer's
code length in bits (sum of -log2 p over the answer tokens, end-of-text included), per question and per
answer token, for the base model (adapters off) before training and the trained model after; written to
OUT/eval.json with the per-type means and standard errors, OUT/adapters.safetensors holds the adapters.

  sft.py --model Qwen/Qwen3-8B --train 'DIR/train_*.jsonl' --heldout 'DIR/heldout_*.jsonl' --out DIR
         [--steps 1000] [--batch 8] [--max-tokens 768] [--lr 2e-4] [--rank 16] [--alpha 32]
         [--eval-per-type 128] [--hours 1.8] [--seed 0]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn

SEP = "<answer>\n"
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRA(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.on = True
        dev = base.weight.device
        self.A = nn.Parameter(torch.randn(rank, base.in_features, device=dev) / rank)
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=dev))

    def forward(self, x):
        y = self.base(x)
        if not self.on:
            return y
        return y + ((x.float() @ self.A.T) @ self.B.T * self.scale).to(y.dtype)


def wrap(model, rank, alpha):
    adapters = {}
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in TARGETS:
                lora = LoRA(child, rank, alpha)
                setattr(module, child_name, lora)
                adapters[f"{name}.{child_name}"] = lora
    return adapters


def load(pattern):
    by_type = {}
    for path in sorted(glob.glob(pattern)):
        for line in open(path):
            q = json.loads(line)
            by_type.setdefault(q["type"], []).append(q)
    return by_type


def changed(q) -> bool:
    """Whether the measured answer differs from no change: KL above 0.1 bits (the largest of a rank
    question's four), or a continuation that differs from the clean one."""
    n = q["numbers"]
    if "kl_bits" in n:
        v = n["kl_bits"]
        return (max(v) if isinstance(v, list) else v) > 0.1
    if "tokens_unchanged" in n:
        return n["tokens_unchanged"] < len(n["edited_ids"])
    return True


def encode(tok, q, max_tokens):
    prompt = tok(q["input"] + SEP, add_special_tokens=False)["input_ids"]
    answer = tok(q["answer"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    prompt = prompt[-max(1, max_tokens - len(answer)) :]  # keep the question's end (the question line)
    return prompt, answer


def collate(tok, items, max_tokens, dev):
    seqs = [encode(tok, q, max_tokens) for q in items]
    width = max(len(p) + len(a) for p, a in seqs)
    ids = torch.full((len(seqs), width), tok.pad_token_id or 0, dtype=torch.long)
    labels = torch.full((len(seqs), width), -100, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for r, (p, a) in enumerate(seqs):
        s = p + a
        ids[r, : len(s)] = torch.tensor(s)
        labels[r, len(p) : len(s)] = torch.tensor(a)
        mask[r, : len(s)] = 1
    return ids.to(dev), labels.to(dev), mask.to(dev)


def answer_bits(model, ids, labels, mask):
    """Per sequence: the answer's bits and its token count (the head runs at the answer positions only)."""
    h = model.model(input_ids=ids, attention_mask=mask).last_hidden_state[:, :-1]
    target = labels[:, 1:]
    valid = target != -100
    lp = torch.log_softmax(model.lm_head(h[valid]).float(), dim=-1)
    nll = -lp.gather(-1, target[valid][:, None])[:, 0]
    bits = torch.zeros(ids.shape[0], device=ids.device).index_add(0, valid.nonzero()[:, 0], nll) / math.log(2)
    return bits, valid.sum(-1)


@torch.no_grad()
def evaluate(model, tok, heldout, per_type, batch, max_tokens, dev):
    model.eval()
    out = {}
    for kind, qs in sorted(heldout.items()):
        qs = qs[:per_type]
        bits, count = [], []
        for s in range(0, len(qs), batch):
            ids, labels, mask = collate(tok, qs[s : s + batch], max_tokens, dev)
            b, n = answer_bits(model, ids, labels, mask)
            bits += b.tolist()
            count += n.tolist()
        mean = sum(bits) / len(bits)
        se = (sum((x - mean) ** 2 for x in bits) / max(1, len(bits) - 1) / len(bits)) ** 0.5
        out[kind] = {"questions": len(bits), "bits_per_question": mean, "se": se, "bits_per_answer_token": sum(bits) / sum(count)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--train", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--eval-per-type", type=int, default=128)
    ap.add_argument("--hours", type=float, default=1.8)
    ap.add_argument("--changed-share", type=float, default=0.5,
                    help="share of each type's draws taken from its questions whose measured answer differs from no change")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.time()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "mps")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    adapters = wrap(model, args.rank, args.alpha)
    params = [p for a in adapters.values() for p in (a.A, a.B)]
    train, heldout = load(args.train), load(args.heldout)
    types = sorted(train)
    moved = {k: [q for q in v if changed(q)] for k, v in train.items()}
    log = open(out / "train.jsonl", "a")
    meta = {"args": vars(args), "train_questions": {k: len(v) for k, v in train.items()}, "changed_questions": {k: len(v) for k, v in moved.items()}, "heldout_questions": {k: len(v) for k, v in heldout.items()},
            "adapter_parameters": sum(p.numel() for p in params)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)

    def set_adapters(on):
        for a in adapters.values():
            a.on = on

    set_adapters(False)
    base = evaluate(model, tok, heldout, args.eval_per_type, args.batch, args.max_tokens, dev)
    (out / "eval_base.json").write_text(json.dumps(base, indent=1))
    print(json.dumps({"base": base}), flush=True)
    set_adapters(True)
    eval_seconds = time.time() - started

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    step, t0 = 0, time.time()
    while step < args.steps:
        if time.time() - started > args.hours * 3600 - 2 * eval_seconds - 120:
            print(json.dumps({"stopped_for_time_at_step": step}), flush=True)
            break
        model.train()
        items = []
        for _ in range(args.batch):
            kind = random.choice(types)
            pool = moved[kind] if moved[kind] and random.random() < args.changed_share else train[kind]
            items.append(random.choice(pool))
        ids, labels, mask = collate(tok, items, args.max_tokens, dev)
        bits, n = answer_bits(model, ids, labels, mask)
        loss = bits.sum() / n.sum() * math.log(2)  # nats per answer token
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1.0, (step + 1) / args.warmup)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 10 == 0:
            rec = {"step": step, "bits_per_answer_token": loss.item() / math.log(2), "seconds_per_step": (time.time() - t0) / step,
                   "types": [q["type"] for q in items]}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
    from safetensors.torch import save_file

    save_file({f"{k}.{n}": getattr(a, n).detach().cpu().contiguous() for k, a in adapters.items() for n in ("A", "B")}, str(out / "adapters.safetensors"))
    trained = evaluate(model, tok, heldout, args.eval_per_type, args.batch, args.max_tokens, dev)
    result = {"steps": step, "base": base, "trained": trained,
              "gain_bits_per_question": {k: base[k]["bits_per_question"] - trained[k]["bits_per_question"] for k in trained},
              "hours": (time.time() - started) / 3600}
    (out / "eval.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
