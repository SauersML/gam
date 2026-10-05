"""Outcome-based RL of an open-weights oracle (#2951) with TRL's GRPOTrainer (trl >= 1.14), whose
environment_factory runs multi-turn tool use.

Interface. The policy reads the investigator harness's own prompt for its target's task and arm and acts
through the harness's `oracle` command line (episodes.investigator_prompt, episodes.Investigation.oracle;
bench/oracle_2951), the interface of the Claude investigations it is fine-tuned on (train_sft.py). Each
rollout has a fresh harness session (investigate.session_file in a directory of its own: the running
server, the arm's allowed commands, the task's corpus rows, and its own call log, so the task's call
budget holds per rollout). The episode ends with a reply that has no tool call: the report, a JSON object
with at least "rule".

Reward of one rollout, computed for the batch at once:
  1. the report is parsed and frozen (episodes.freeze: hash and time); a reply that is not a JSON object
     with a nonempty "rule" tells the reader nothing and its reward is 0;
  2. tests are drawn after the freeze, seeded from the report's hash and entropy drawn then: for an
     organism task, --items-command writes fresh items (as uplift.py's items stage) and the server's
     "options" op measures the subject model's probability of each reply (the behaviour protocol); for
     another task, episodes.draw_tests (next-token tests through the server);
  3. the reader service (reader.py serve) reads every test with the report's rule (cut to its first
     --report-tokens tokens of the policy's tokenizer) and with no document;
  4. reward = mean over the tests of [log score with the rule - log score with no document], nats per
     test, paired on the same tests.
The executable hypothesis's native fidelity is measured and logged with each episode, not added: for an
organism, the balanced accuracy of the report's predictor (predict(item) -> option, run in a separate
Python with a 1 s limit per item) against the measured choices. Adding it to a reward in nats would need
an exchange rate between accuracy and nats that nothing here determines.
Budgets: the task's call budget (the session's max_calls; past it the command line answers that the
budget is spent), max_tool_calling_iterations = that budget, the rule's length through the reader reading
only its first L tokens, and the whole investigation through max_completion_length tokens (tool outputs
included). Every rollout goes to the episode store with its tests, scores and reward, so RL episodes are
SFT data too. GRPO's loss and normalization are TRL's defaults, with no KL term (beta 0).

  train_rl.py --model Qwen/Qwen3-1.7B --targets TARGETS.jsonl --server ADDRESS --reader ADDRESS --out DIR
              --lr LR --steps N --generations G --report-tokens L --max-completion-length T
              --tests-per-report N [--items-command CMD] [--candidates-per-side M]
              [--lora-rank R] [--vllm] [--run NAME] [--root DIR]
TARGETS.jsonl: one episode-store target record per line with "task" (the harness task JSON) and "arm".
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import episodes as E
from native_client import NativeClient

CONFIG: argparse.Namespace | None = None
HARNESS = None


def harness():
    global HARNESS
    if HARNESS is None:
        HARNESS = E.harness()
    return HARNESS


def oracle_session(target: dict):
    """The harness's command-line Session class (bench/oracle_2951/oracle.py) for a session file."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("oracle_cli", E.HARNESS / "oracle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Session


class OracleEnvironment(E.Investigation):
    """One rollout's investigation: the harness tool (inherited) on a fresh session of its own."""

    def __init__(self):
        super().__init__(None, None)
        self._target = None
        self._dir = None

    def reset(self, **row) -> None:
        self._target = json.loads(row["target"])
        self._dir = Path(tempfile.mkdtemp(prefix="oracle-rollout-"))
        (self._dir / "work").mkdir()
        session = harness().session_file(self._target["task"], self._dir, CONFIG.server, self._target["arm"])
        self._session = session
        self._tool = str(self._dir / "bin" / "oracle")
        self._workdir = str(self._dir / "work")
        return None

    def calls(self) -> int:
        log = self._dir / "calls.jsonl"
        return sum(1 for _ in open(log)) if log.exists() else 0


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


def truncate(text: str, tokens: int, tokenizer) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return text if len(ids) <= tokens else tokenizer.decode(ids[:tokens])


def item_tests(episode: dict, env: OracleEnvironment) -> list[dict]:
    """Fresh behaviour items for an organism task, drawn after the freeze and measured by the server."""
    draw = E.fresh_draw(episode)
    task = episode["target"]["task"]
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "items.jsonl"
        subprocess.run(CONFIG.items_command.format(organism=task["organism"], count=CONFIG.tests_per_report, seed=draw["seed"], out=out), shell=True, check=True)
        items = [json.loads(line) for line in open(out) if line.strip()]
    session = oracle_session(task)(str(env._session))
    session.config.pop("log", None)  # measurement for the reward is not the policy's call
    reply = session.request({"op": "options", "model": task["subject"], "items": [session.option_item(it) for it in items], "intervention": {}})
    tests = []
    for i, (it, r) in enumerate(zip(items, reply["items"])):
        lp = np.asarray(r["log_probabilities"], dtype=np.float64)
        p = np.exp(lp - lp.max())
        p /= p.sum()
        tests.append({"id": f"item{i}", **draw, "family": "behaviour_item", "kind": "response",
                      "context": {"text": "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in it["messages"]), "messages": it["messages"]},
                      "intervention_text": "", "options": it["options"],
                      "measured": {"log_probabilities": r["log_probabilities"], "p": p.tolist(), "choice": int(r["choice"]), "base_choice": it.get("base_choice")}})
    return tests


def predictor_fidelity(report: dict | None, tests: list[dict]) -> float | None:
    """Balanced accuracy of the report's predictor against the measured choices: the mean of its accuracy
    on items whose choice differs from the pre-update choice and on the others (plain accuracy when the
    items carry no pre-update choice). None without a predictor."""
    source = (report or {}).get("predictor_python")
    if not source or not tests:
        return None
    program = source + "\nimport json, sys\nfor line in sys.stdin:\n    item = json.loads(line)\n    print(int(predict(item)), flush=True)\n"
    feed = "".join(json.dumps({"messages": t["context"]["messages"], "options": t["options"]}) + "\n" for t in tests)
    try:
        r = subprocess.run([sys.executable, "-I", "-c", program], input=feed, capture_output=True, text=True, timeout=len(tests))
        guesses = [int(x) for x in r.stdout.split()]
    except (subprocess.TimeoutExpired, ValueError):
        return 0.0
    if len(guesses) != len(tests):
        return 0.0
    hit = np.array([g == t["measured"]["choice"] for g, t in zip(guesses, tests)], dtype=float)
    base = [t["measured"].get("base_choice") for t in tests]
    if any(b is None for b in base):
        return float(hit.mean())
    changed = np.array([t["measured"]["choice"] != b for t, b in zip(tests, base)])
    parts = [hit[changed].mean() if changed.any() else None, hit[~changed].mean() if (~changed).any() else None]
    return float(np.mean([x for x in parts if x is not None]))


def make_reward(policy_tokenizer):
    reader = NativeClient(CONFIG.reader)
    calls = {"n": 0}

    def oracle_reward(prompts, completions, environments, **kwargs) -> list[float]:
        calls["n"] += 1
        episodes, tests_of, reports = [], [], []
        for i, (prompt, completion, env) in enumerate(zip(prompts, completions, environments)):
            target = env._target
            final = completion[-1].get("content", "") if completion and completion[-1].get("role") == "assistant" and not completion[-1].get("tool_calls") else ""
            report = parse_report(final or "")
            episode = E.new_episode(target, "policy", CONFIG.model, f"rl-{CONFIG.run}-{calls['n']:05d}-{i:03d}")
            episode["investigator"]["transcript"] = list(prompt) + list(completion)
            episode["investigator"]["requests"] = env.calls()
            tests = []
            if report is not None:
                E.freeze(episode, report)
                if target["task"].get("kind") == "organism":
                    tests = item_tests(episode, env)
                else:
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(target["models"]["updated"]["path"])
                    tests = E.draw_tests(episode, NativeClient(CONFIG.server), tokenizer, CONFIG.tests_per_report, CONFIG.candidates_per_side)
                E.attach_tests(episode, tests)
            episodes.append(episode)
            tests_of.append(tests)
            reports.append(report)
        # One reader call for the batch: every test with the rule (cut to L tokens) and with nothing.
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
            episode["reward"] = {"total": gain, "reader_gain_nats_per_test": gain, "predictor_balanced_accuracy": predictor_fidelity(report, tests),
                                 "requests": episode["investigator"]["requests"], "report_parsed": report is not None}
            E.save(episode, root=Path(CONFIG.root))
            rewards.append(gain)
        return rewards

    return oracle_reward


def main():
    global CONFIG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--server", required=True, help="the running measurement server (mpd_oracle_2951) holding every target's models")
    ap.add_argument("--reader", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--generations", type=int, required=True, help="rollouts per prompt (GRPO's group size)")
    ap.add_argument("--report-tokens", type=int, required=True)
    ap.add_argument("--max-completion-length", type=int, required=True)
    ap.add_argument("--tests-per-report", type=int, required=True)
    ap.add_argument("--items-command", help="organism tasks: a shell command with {organism} {count} {seed} {out} that writes fresh items")
    ap.add_argument("--candidates-per-side", type=int, default=4, help="next-token tests: candidates from each side (episodes.draw_tests)")
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
        if "task" not in t or "arm" not in t:
            raise SystemExit(f"target {t['id']}: needs the harness task and arm")
        if t["task"].get("kind") == "organism" and not CONFIG.items_command:
            raise SystemExit("organism targets need --items-command")
    rows = [{"prompt": E.investigator_prompt(t), "target": json.dumps(t)} for t in targets]
    budget = max(t["task"]["max_calls"] for t in targets)
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
        max_tool_calling_iterations=budget,
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
