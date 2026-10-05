"""Supervised fine-tuning of an open-weights oracle (#2951) on successful investigations.

Data. Every episode in the store (~/mpd-data/oracle/episodes) with a frozen report, tests, and the
chosen reader's scores of the report and of no documents on those tests. An episode's gain is the
mean over its tests of [log-score with the report - log-score with no documents] under that reader
(paired on the same tests, nats per test). Selection rule: for each target, the one episode of largest
gain, kept when that gain is above zero (the report told the reader something true about the target's
measured behaviour). Selected episodes are listed with their gains before training.

Conversation. The investigator prompt of episodes.investigator_prompt (the same system text, tool and
budgets as train_rl.py), then the episode's assistant and tool messages, ending in an assistant message
that is the frozen report's JSON. A transcript whose assistant turns carry no tool calls is rebuilt
from the episode's server log: one assistant tool call per logged request and one tool message with its
reply. Rendered with the tokenizer's chat template made prefix-preserving with assistant markers
(trl.get_training_chat_template), thinking disabled, the experiment tool's schema from the same method
the RL environment exposes; the loss is on assistant tokens only (SFTConfig.assistant_only_loss), so
tool replies and prompts are context. Episodes longer than the model's context are dropped and counted.

  train_sft.py --model Qwen/Qwen3-1.7B --reader BACKEND:MODEL --out DIR --lr LR --epochs E
               --request-budget B --report-tokens L [--lora-rank R] [--root DIR] [--max-steps N]
--lora-rank R trains LoRA adapters of rank R on every linear map (alpha = R, so the update's scale
does not depend on R); without it every parameter is trained.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import episodes as E


def gain(episode: dict, reader: str) -> float | None:
    with_report = episode["scores"].get(f"{reader}:report")
    without = episode["scores"].get(f"{reader}:none")
    if not with_report or not without:
        return None
    ids = [t["id"] for t in episode["tests"] if t["id"] in with_report["per_test"] and t["id"] in without["per_test"]]
    if not ids:
        return None
    return float(np.mean([with_report["per_test"][i]["log_score"] - without["per_test"][i]["log_score"] for i in ids]))


def select(root: Path, reader: str) -> list[tuple[dict, float]]:
    best: dict[str, tuple[dict, float]] = {}
    for path in sorted(root.glob("*/*.json")):
        episode = E.load(path)
        if episode["report"] is None:
            continue
        g = gain(episode, reader)
        if g is None:
            continue
        target = episode["target"]["id"]
        if target not in best or g > best[target][1]:
            best[target] = (episode, g)
    return [(e, g) for e, g in best.values() if g > 0]


def conversation(episode: dict, request_budget: int, report_tokens: int) -> list[dict]:
    messages = E.investigator_prompt(episode["target"], request_budget, report_tokens)
    transcript = episode["investigator"]["transcript"]
    first = next((i for i, m in enumerate(transcript) if m["role"] == "assistant"), len(transcript))
    body = [dict(m) for m in transcript[first:]]
    if not any(m["role"] == "assistant" and m.get("tool_calls") for m in body):
        body = []
        for entry in episode["investigator"]["server"]:
            call = {"type": "function", "function": {"name": "experiment", "arguments": {"request": json.dumps(entry["request"])}}}
            body.append({"role": "assistant", "content": "", "tool_calls": [call]})
            body.append({"role": "tool", "name": "experiment", "content": json.dumps(entry["reply"])})
    report = json.dumps(episode["report"]["content"], ensure_ascii=False)
    while body and body[-1]["role"] == "assistant" and not body[-1].get("tool_calls"):
        body.pop()  # the final answer is the frozen report itself
    body.append({"role": "assistant", "content": report})
    out = []
    for m in messages + body:
        row = {"role": m["role"], "content": m.get("content") or ""}
        if m.get("tool_calls"):
            row["tool_calls"] = [
                {"type": "function", "function": {"name": c["function"]["name"], "arguments": json.dumps(c["function"]["arguments"]) if not isinstance(c["function"]["arguments"], str) else c["function"]["arguments"]}}
                for c in m["tool_calls"]
            ]
        if m.get("name"):
            row["name"] = m["name"]
        out.append(row)
    return out


def tool_schema(request_budget: int) -> list[dict]:
    from transformers.utils import get_json_schema

    return [get_json_schema(E.Investigation(None, request_budget).experiment)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--reader", required=True, help="the reader whose scores rank episodes, BACKEND:MODEL as in the store's score keys")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--epochs", type=float, required=True)
    ap.add_argument("--request-budget", type=int, required=True)
    ap.add_argument("--report-tokens", type=int, required=True)
    ap.add_argument("--lora-rank", type=int)
    ap.add_argument("--root", default=str(E.ROOT))
    ap.add_argument("--max-steps", type=int, default=-1, help="stop after this many optimizer steps (a smoke test)")
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from transformers import AutoConfig, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    chosen = select(Path(args.root), args.reader)
    for episode, g in chosen:
        print(json.dumps({"target": episode["target"]["id"], "episode": episode["episode"], "gain_nats_per_test": g, "tests": len(episode["tests"])}))
    if not chosen:
        raise SystemExit("no episode has a positive gain under this reader")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    context = AutoConfig.from_pretrained(args.model).max_position_embeddings
    tools = tool_schema(args.request_budget)
    rows, dropped = [], 0
    for episode, _ in chosen:
        messages = conversation(episode, args.request_budget, args.report_tokens)
        length = len(tokenizer.apply_chat_template(messages, tools=tools, tokenize=True, return_dict=True, enable_thinking=False)["input_ids"])
        if length > context:
            dropped += 1
            continue
        rows.append({"messages": messages, "tools": json.dumps(tools), "chat_template_kwargs": {"enable_thinking": False}})
    print(json.dumps({"selected": len(chosen), "dropped_longer_than_context": dropped, "context_tokens": context}))
    if not rows:
        raise SystemExit("every selected episode is longer than the model's context")

    gpu = torch.cuda.is_available()
    peft_config = None
    if args.lora_rank:
        from peft import LoraConfig

        peft_config = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank, target_modules="all-linear", task_type="CAUSAL_LM")
    config = SFTConfig(
        output_dir=args.out,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        assistant_only_loss=True,
        max_length=context,
        bf16=gpu,
        use_cpu=not gpu,  # without CUDA (the Mac smoke test) train on the CPU, not on Apple's GPU
        model_init_kwargs={"dtype": torch.bfloat16 if gpu else torch.float32},
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="epoch" if args.max_steps < 0 else "no",
        report_to="none",
        seed=0,
    )
    trainer = SFTTrainer(model=args.model, args=config, train_dataset=Dataset.from_list(rows), processing_class=tokenizer, peft_config=peft_config)
    result = trainer.train()
    trainer.save_model(args.out)
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    print(json.dumps({"steps": trainer.state.global_step, "first_loss": losses[0] if losses else math.nan, "last_loss": losses[-1] if losses else math.nan, "train_seconds": result.metrics.get("train_runtime")}))


if __name__ == "__main__":
    main()
