"""Outcome-based RL of an open-weights oracle (#2951) with TRL's GRPOTrainer (trl >= 1.14), whose
environment_factory runs multi-turn tool use: during an episode the policy calls `experiment`, which
forwards its JSON request to the native experiment server (native_client.py), and the episode ends with
a reply that carries no tool call, which is the report (a JSON object with at least "rule").

Reward of one rollout, computed for the whole batch at once:
  1. the report is parsed and frozen (episodes.freeze: hash and time); a reply that is not a JSON
     object with a nonempty "rule" gives the reader nothing;
  2. fresh tests are drawn after the freeze from the report's hash and new entropy, and measured through
     the server (episodes.draw_tests, --tests-per-report tests, --candidates-per-side m);
  3. the reader service (reader.py serve) reads every test twice, with the report's rule as its one
     document and with no documents; the rule is cut to its first --report-tokens tokens (the policy's
     tokenizer) before the reader sees it;
  4. reward = mean over the tests of [log-score with the report - log-score with no documents]
     (nats per test, paired on the same tests, so the tests' difficulty cancels) + native fidelity.
Native fidelity is a hook: native_fidelity sends {"op": "fidelity", "model", "hypothesis" (the report's
"hypothesis", or null), "sequences" (the tests' texts)} and adds the server's ok.nats_per_token, a
number the server defines (higher is better; the server also defines a null hypothesis's value). Until
the server has the op its reply is an error, the term is recorded as absent and adds nothing.
Budgets: the request budget B is enforced by the tool (a batch counts as its members; past B the tool
answers with an error asking for the report) and by max_tool_calling_iterations = B; the report's
length is bounded by the reader reading only the first L tokens of the rule (words past L earn nothing)
and the whole investigation by max_completion_length tokens (GRPOConfig, tool replies included).
Every rollout is written to the episode store with its tests, scores and reward, so RL episodes are SFT
data too. GRPO's loss and normalization are TRL's defaults (loss_type "dapo"), no KL term (beta 0).

  train_rl.py --model Qwen/Qwen3-1.7B --targets TARGETS.jsonl --server ADDRESS --reader ADDRESS --out DIR
              --lr LR --steps N --generations G --request-budget B --report-tokens L
              --max-completion-length T --tests-per-report N --candidates-per-side M
              [--lora-rank R] [--vllm] [--run NAME] [--root DIR]
TARGETS.jsonl: one episodes.py target record per line (id, kind, models {role: {server, path}},
description, contexts).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

import episodes as E
from native_client import NativeClient
from reader import log_score

CONFIG: argparse.Namespace | None = None
TOKENIZERS: dict[str, object] = {}


class OracleEnvironment(E.Investigation):
    """One rollout's investigation: the experiment tool (inherited) with its own budget and server log."""

    def __init__(self):
        super().__init__(NativeClient(CONFIG.server), CONFIG.request_budget)
        self._target = None

    def reset(self, **row) -> None:
        self._target = json.loads(row["target"])
        self._client = NativeClient(CONFIG.server)
        self._used = 0
        return None


def parse_report(text: str) -> dict | None:
    """The final reply as a JSON object with a nonempty "rule" (a fenced code block is unwrapped)."""
    fenced = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", text, flags=re.S)
    if fenced:
        text = fenced.group(1)
    try:
        report = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(report, dict) or not isinstance(report.get("rule"), str) or not report["rule"].strip():
        return None
    return report


def target_tokenizer(target: dict):
    from transformers import AutoTokenizer

    path = target["models"]["updated"]["path"]
    if path not in TOKENIZERS:
        TOKENIZERS[path] = AutoTokenizer.from_pretrained(path)
    return TOKENIZERS[path]


def native_fidelity(client: NativeClient, target: dict, report: dict | None, tests: list[dict]) -> tuple[float | None, str | None]:
    request = {
        "op": "fidelity",
        "model": target["models"]["updated"]["server"],
        "hypothesis": (report or {}).get("hypothesis"),
        "sequences": [t["context"]["tokens"] for t in tests],
    }
    reply = client.raw(request)
    if "error" in reply:
        return None, reply["error"]
    return float(reply["ok"]["nats_per_token"]), None


def truncate(text: str, tokens: int, tokenizer) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return text if len(ids) <= tokens else tokenizer.decode(ids[:tokens])


def make_reward(policy_tokenizer):
    reader = NativeClient(CONFIG.reader)
    calls = {"n": 0}

    def oracle_reward(prompts, completions, environments, **kwargs) -> list[float]:
        calls["n"] += 1
        episodes, tests_of, reports = [], [], []
        server = NativeClient(CONFIG.server)
        for i, (prompt, completion, env) in enumerate(zip(prompts, completions, environments)):
            target = env._target
            final = completion[-1].get("content", "") if completion and completion[-1].get("role") == "assistant" and not completion[-1].get("tool_calls") else ""
            report = parse_report(final or "")
            episode = E.new_episode(target, "policy", CONFIG.model, f"rl-{CONFIG.run}-{calls['n']:05d}-{i:03d}")
            E.record_investigation(episode, list(prompt) + list(completion), env._client)
            tests = []
            if report is not None:
                E.freeze(episode, report)
                tests = E.draw_tests(episode, server, target_tokenizer(target), CONFIG.tests_per_report, CONFIG.candidates_per_side)
                E.attach_tests(episode, tests)
            episodes.append(episode)
            tests_of.append(tests)
            reports.append(report)
        # One reader call for the batch: every test with the report's rule (cut to L tokens) and with nothing.
        rows, index = [], []
        for i, (episode, tests, report) in enumerate(zip(episodes, tests_of, reports)):
            if report is None:
                continue
            rule = [truncate(d, CONFIG.report_tokens, policy_tokenizer) for d in E.report_documents(report)]
            for condition, docs in (("report", rule), ("none", [])):
                for row in E.reader_tests(episode, docs, tests):
                    rows.append(row)
                    index.append((i, condition))
        reply = reader.call({"op": "read", "tests": rows}) if rows else {"results": [], "reader": None}
        results, described = reply["results"], reply["reader"]
        rewards = []
        for i, (episode, tests, report) in enumerate(zip(episodes, tests_of, reports)):
            gain = 0.0
            if report is not None:
                for condition in ("report", "none"):
                    picked = [(row, r) for row, r, (j, c) in zip(rows, results, index) if j == i and c == condition]
                    E.add_scores(episode, condition, described, [row for row, _ in picked], [r for _, r in picked])
                key = f"{described['backend']}:{described['model']}"
                per_r = episode["scores"][f"{key}:report"]["per_test"]
                per_n = episode["scores"][f"{key}:none"]["per_test"]
                gain = float(np.mean([per_r[t]["log_score"] - per_n[t]["log_score"] for t in per_r]))
            fidelity, absent = native_fidelity(server, episode["target"], report, tests)
            reward = gain + (fidelity or 0.0)
            episode["reward"] = {"total": reward, "reader_gain_nats_per_test": gain, "native_fidelity_nats_per_token": fidelity, "fidelity_absent_because": absent,
                                 "requests": episode["investigator"]["requests"], "report_parsed": report is not None}
            E.save(episode, root=Path(CONFIG.root))
            rewards.append(reward)
        return rewards

    return oracle_reward


def main():
    global CONFIG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--server", required=True)
    ap.add_argument("--reader", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--generations", type=int, required=True, help="rollouts per prompt (GRPO's group size)")
    ap.add_argument("--request-budget", type=int, required=True)
    ap.add_argument("--report-tokens", type=int, required=True)
    ap.add_argument("--max-completion-length", type=int, required=True)
    ap.add_argument("--tests-per-report", type=int, required=True)
    ap.add_argument("--candidates-per-side", type=int, required=True)
    ap.add_argument("--lora-rank", type=int)
    ap.add_argument("--vllm", action="store_true", help="generate rollouts with vLLM colocated on the training GPUs")
    ap.add_argument("--run", default=time.strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--root", default=str(E.ROOT), help="the episode store")
    CONFIG = ap.parse_args()

    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    targets = [json.loads(line) for line in open(CONFIG.targets) if line.strip()]
    for t in targets:
        E.check_target(t)
    rows = [{"prompt": E.investigator_prompt(t, CONFIG.request_budget, CONFIG.report_tokens), "target": json.dumps(t)} for t in targets]
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.model)
    gpu = torch.cuda.is_available()
    peft_config = None
    if CONFIG.lora_rank:
        from peft import LoraConfig

        peft_config = LoraConfig(r=CONFIG.lora_rank, lora_alpha=CONFIG.lora_rank, target_modules="all-linear", task_type="CAUSAL_LM")
    config = GRPOConfig(
        output_dir=CONFIG.out,
        learning_rate=CONFIG.lr,
        max_steps=CONFIG.steps,
        num_generations=CONFIG.generations,
        per_device_train_batch_size=CONFIG.generations,
        max_completion_length=CONFIG.max_completion_length,
        max_tool_calling_iterations=CONFIG.request_budget,
        chat_template_kwargs={"enable_thinking": False},
        beta=0.0,
        use_vllm=CONFIG.vllm,
        vllm_mode="colocate",
        bf16=gpu,
        use_cpu=not gpu,  # without CUDA (the Mac smoke test) train on the CPU, not on Apple's GPU
        model_init_kwargs={"dtype": torch.bfloat16 if gpu else torch.float32},
        gradient_checkpointing=True,
        logging_steps=1,
        save_steps=max(1, CONFIG.steps // 10),
        report_to="none",
        seed=0,
    )
    trainer = GRPOTrainer(
        model=CONFIG.model,
        reward_funcs=[make_reward(tokenizer)],
        args=config,
        train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer,
        peft_config=peft_config,
        environment_factory=OracleEnvironment,
    )
    trainer.train()
    trainer.save_model(CONFIG.out)
    history = [{k: v for k, v in h.items() if "reward" in k or k in ("step", "loss")} for h in trainer.state.log_history]
    Path(CONFIG.out, "rewards.json").write_text(json.dumps(history, indent=1))
    print(json.dumps(history[-3:], indent=1))


if __name__ == "__main__":
    main()
