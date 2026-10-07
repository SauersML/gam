"""Supervised prediction training of the graph oracle (#2951): Qwen3-8B with LoRA adapters learns to write
the measured answers of generate.py's causal questions about the target model (Qwen3-0.6B).

Example = Qwen3's chat template with the question as the user turn (thinking off), then the answer text and
<|im_end|> as the assistant turn (--format raw: the question, "<answer>\n", the answer and end-of-text, the
first run's format); the loss is the answer tokens' negative log-likelihood (the question is context only).
The adapters are saved as OUT/adapters.safetensors and as a PEFT directory OUT/peft (vLLM / g-rl's --init). Batches draw a
question type uniformly, then a question of that type (the mixture over types), so rare and common types
train equally; within a type, --changed-share of the draws come from the questions whose measured answer
differs from no change (most edits of single pieces move M by under 0.1 bits). LoRA: every linear map of every block (q, k, v, o, gate, up, down), W x + (alpha / r) B A x,
A Gaussian (std 1 / r), B zero, adapters in float32 over the bfloat16 model; AdamW, linear warmup, then
constant.

Evaluation on held-out questions (texts the training shards never use): per question type the answer's
code length in bits (sum of -log2 p over the answer tokens, end-of-text included), per question and per
answer token, for the base model (adapters off) before training and the trained model after; written to
OUT/eval.json with the per-type means and standard errors, OUT/adapters.safetensors holds the adapters.

  sft.py --model Qwen/Qwen3-8B --train 'DIR/train_*.jsonl' --heldout 'prompts=DIR/heldout_*.jsonl'
         [--heldout 'pieces=DIR/pieces_*.jsonl' --heldout 'behaviors=DIR/behaviors_heldout*.jsonl'] --out DIR
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


def save_peft(adapters_path, out_dir, base: str, rank: int, alpha: float):
    """The adapters as a PEFT LoRA directory (adapter_config.json, adapter_model.safetensors) for vLLM and
    g-rl's loop: lora_A = A [r, in], lora_B = B [out, r], scaling lora_alpha / r, as here."""
    from safetensors.torch import load_file, save_file

    state = load_file(str(adapters_path))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for key, value in state.items():
        name, which = key.rsplit(".", 1)
        tensors[f"base_model.model.{name}.lora_{which}.weight"] = value.contiguous()
    save_file(tensors, str(out_dir / "adapter_model.safetensors"), metadata={"format": "pt"})
    config = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": base, "r": rank, "lora_alpha": alpha,
              "lora_dropout": 0.0, "target_modules": list(TARGETS), "bias": "none", "fan_in_fan_out": False, "modules_to_save": None,
              "inference_mode": True, "init_lora_weights": True, "use_rslora": False, "use_dora": False, "layers_to_transform": None,
              "rank_pattern": {}, "alpha_pattern": {}}
    (out_dir / "adapter_config.json").write_text(json.dumps(config, indent=1))


def load_slim(pattern):
    """Training questions by type, each kept as its input, answer, text id and whether its measured answer
    differs from no change (the numbers dropped: a million questions fit in a few GB)."""
    by_type = {}
    for path in sorted(p for part in pattern.split(",") for p in glob.glob(part)):
        for line in open(path):
            q = json.loads(line)
            by_type.setdefault(q["type"], []).append({"type": q["type"], "input": q["input"], "answer": q["answer"],
                                                      "text_id": q["text_id"], "changed": changed(q)})
    return by_type


def load(pattern):
    """Questions by type from every file matching the comma-separated glob patterns."""
    by_type = {}
    for path in sorted(p for part in pattern.split(",") for p in glob.glob(part)):
        for line in open(path):
            q = json.loads(line)
            by_type.setdefault(q["type"], []).append(q)
    return by_type


def changed(q) -> bool:
    """Whether the measured answer differs from no change: KL above 0.1 bits (the largest of a rank
    question's four), or a continuation that differs from the clean one."""
    n = q["numbers"]
    if "kl_bits_after_removal" in n:  # carry: some removal moves the edit's effect by more than 0.1 bits
        return max(abs(v - n["kl_bits_edit"]) for v in n["kl_bits_after_removal"]) > 0.1
    if "kl_bits" in n:
        v = n["kl_bits"]
        return (max(v) if isinstance(v, list) else v) > 0.1
    if "tokens_unchanged" in n:
        return n["tokens_unchanged"] < len(n["edited_ids"])
    return True


FORMAT = {"name": "chat"}  # "chat": Qwen3's chat template (user turn = question, thinking off); "raw": question + SEP


def prompt_text(tok, q) -> str:
    """The oracle's context for a question: Qwen3's chat template with the question as the user turn and
    thinking off (the format g-rl's program training uses), or the first run's raw format."""
    if FORMAT["name"] == "raw":
        return q["input"] + SEP
    return tok.apply_chat_template([{"role": "user", "content": q["input"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)


def end_id(tok) -> int:
    return tok.eos_token_id if FORMAT["name"] == "raw" else tok.convert_tokens_to_ids("<|im_end|>")


def encode(tok, q, max_tokens):
    prompt = tok(prompt_text(tok, q), add_special_tokens=False)["input_ids"]
    answer = tok(q["answer"], add_special_tokens=False)["input_ids"] + [end_id(tok)]
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
        # Knowing M beyond the format: the bits of the measured answer minus the bits of the "no change"
        # answer (the clean distribution or continuation the question states, with KL 0) on the same
        # question; negative = the oracle prefers what M does. Reported over all questions and over those
        # whose measured answer differs from no change.
        pairs = [(i, q, nc) for i, q in enumerate(qs) if (nc := no_change_answer(q)) is not None]
        if pairs:
            alt = [dict(q, answer=nc) for _, q, nc in pairs]
            alt_bits = []
            for s in range(0, len(alt), batch):
                ids, labels, mask = collate(tok, alt[s : s + batch], max_tokens, dev)
                alt_bits += answer_bits(model, ids, labels, mask)[0].tolist()
            true_bits = [bits[i] for i, _, _ in pairs]
            d = [t - a for t, a in zip(true_bits, alt_bits)]
            moved = [x for x, (_, q, _) in zip(d, pairs) if changed(q)]
            mean_se = lambda v: (sum(v) / len(v), (sum((x - sum(v) / len(v)) ** 2 for x in v) / max(1, len(v) - 1) / len(v)) ** 0.5) if v else (None, None)  # noqa: E731
            out[kind]["measured_minus_no_change_bits"], out[kind]["measured_minus_no_change_se"] = mean_se(d)
            out[kind]["changed_questions"] = len(moved)
            out[kind]["changed_measured_minus_no_change_bits"], out[kind]["changed_measured_minus_no_change_se"] = mean_se(moved)
    return out


def no_change_answer(q):
    """The answer M would give if the intervention did nothing, in the answer's own format (None when the
    question states no clean outcome): the stated clean distribution with KL 0, or the clean continuation."""
    lines = q["input"].split("\n")
    if q["type"] in ("edit", "cut", "swap", "prompt"):
        clean = next((l[len("<clean> "):] for l in lines if l.startswith("<clean> ")), None)
        return None if clean is None else clean + "\nKL 0.000 bits"
    if q["type"] == "continue":
        return next((l[len("<clean_continuation> "):] for l in lines if l.startswith("<clean_continuation> ")), None)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--train", required=True)
    ap.add_argument("--heldout", action="append", required=True,
                    help="NAME=GLOB (repeatable): held-out sets scored separately, e.g. prompts=... pieces=... behaviors=...")
    ap.add_argument("--eval-every", type=int, default=500, help="score every held-out set every N steps (the learning curve)")
    ap.add_argument("--curve-per-type", type=int, default=32)
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
    ap.add_argument("--types", default="", help="train only on these question types (comma-separated), e.g. the types two compared runs share")
    ap.add_argument("--changed-share", type=float, default=0.5,
                    help="share of each type's draws taken from its questions whose measured answer differs from no change")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--format", default="chat", choices=("chat", "raw"))
    ap.add_argument("--eval-only", default="", help="ADAPTERS: score base and these adapters on the held-out sets, no training")
    ap.add_argument("--export-peft", default="", help="only convert OUT/adapters.safetensors to OUT/peft (no training)")
    args = ap.parse_args()
    FORMAT["name"] = args.format
    if args.export_peft:
        save_peft(Path(args.export_peft), Path(args.out), args.model, args.rank, args.alpha)
        return
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
    train = load_slim(args.train)
    if args.types:
        train = {k: v for k, v in train.items() if k in args.types.split(",")}
    sets = {}
    for spec in args.heldout:
        name, _, pattern = spec.rpartition("=")
        sets[name or "heldout"] = load(pattern)
    types = sorted(train)
    moved = {k: [q for q in v if q["changed"]] for k, v in train.items()}
    log = open(out / "train.jsonl", "a")
    meta = {"args": vars(args), "train_questions": {k: len(v) for k, v in train.items()}, "changed_questions": {k: len(v) for k, v in moved.items()}, "heldout_questions": {n: {k: len(v) for k, v in h.items()} for n, h in sets.items()},
            "distinct_train_texts": len({q["text_id"] for v in train.values() for q in v}),
            "adapter_parameters": sum(p.numel() for p in params)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)

    def set_adapters(on):
        for a in adapters.values():
            a.on = on

    set_adapters(False)
    def evaluate_sets(per_type):
        return {n: evaluate(model, tok, h, per_type, args.batch, args.max_tokens, dev) for n, h in sets.items()}

    base = evaluate_sets(args.eval_per_type)
    if args.eval_only:  # held-out bits of existing adapters against the base model on the same questions
        from safetensors.torch import load_file

        state = load_file(args.eval_only)
        for name, a in adapters.items():
            a.A.data.copy_(state[f"{name}.A"])
            a.B.data.copy_(state[f"{name}.B"])
        set_adapters(True)
        trained = evaluate_sets(args.eval_per_type)
        result = {"adapters": args.eval_only, "base": base, "trained": trained,
                  "gain_bits_per_question": {n: {k: base[n][k]["bits_per_question"] - t[k]["bits_per_question"] for k in t} for n, t in trained.items()}}
        (out / "eval_only.json").write_text(json.dumps(result, indent=1))
        print(json.dumps(result), flush=True)
        return
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
        if args.eval_every and step % args.eval_every == 0:
            curve = {"step": step, "heldout": evaluate_sets(args.curve_per_type)}
            log.write(json.dumps(curve) + "\n")
            print(json.dumps(curve), flush=True)
        if step % 10 == 0:
            rec = {"step": step, "bits_per_answer_token": loss.item() / math.log(2), "seconds_per_step": (time.time() - t0) / step,
                   "types": [q["type"] for q in items]}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
    from safetensors.torch import save_file

    save_file({f"{k}.{n}": getattr(a, n).detach().cpu().contiguous() for k, a in adapters.items() for n in ("A", "B")}, str(out / "adapters.safetensors"))
    save_peft(out / "adapters.safetensors", out / "peft", args.model, args.rank, args.alpha)
    trained = evaluate_sets(args.eval_per_type)
    result = {"steps": step, "base": base, "trained": trained,
              "gain_bits_per_question": {n: {k: base[n][k]["bits_per_question"] - t[k]["bits_per_question"] for k in t} for n, t in trained.items()},
              "hours": (time.time() - started) / 3600}
    (out / "eval.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
