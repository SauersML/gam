"""Checks of the RL scorer's native path with a stand-in score.Scorer (no model), and of the evaluation, SFT data and
vLLM wake around it: python test_scorer.py

Answers keep their order across tasks and seeds, and each (task, seed) goes to the scorer in one call."""

from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scorer  # noqa: E402

calls = []


class Scorer:
    def __init__(self, dev=None):
        pass

    def score(self, task, sources, seed=0):
        calls.append((task["path"], seed, len(sources)))
        return [{"kl_bits": len(x) + seed, "size": 1, "valid": True, "task": task["path"]} for x in sources]


def main():
    import score

    real, score.Scorer = score.Scorer, Scorer
    items = [{"source": "x" * i, "behavior": {"model": "vpd4l", "path": f"b{i % 4}"}, "seed": 1 + i % 2} for i in range(16)]
    out = scorer.native(items)
    assert [r["kl_bits"] for r in out] == [i + 1 + i % 2 for i in range(16)]
    assert [r["task"] for r in out] == [f"b{i % 4}" for i in range(16)]
    assert sorted(calls) == sorted({(f"b{i % 4}", 1 + i % 2, 4) for i in range(16)}), calls
    score.Scorer = real
    scorer._SCORER.clear()
    check_evaluate()
    check_sft_examples()
    check_share_wake()
    print("ok: native scorer order, one call per (task, seed); evaluation ranks answers at each question's precision; SFT data is the training questions' teacher graphs; one vLLM wake per sampling call")


def check_evaluate():
    """train.evaluate scores the policy's answers and the baselines under the evaluation seed only; per question its best
    answer by score.order at the question's eps: correct when its KL is at most eps, beating the teacher when it is
    smaller too."""
    import io
    import json
    import re
    import tempfile

    import train

    seen = []
    train.render = lambda b: b["id"]
    train.baselines = lambda b: {"empty": "K = 100; S = 0", "teacher": "K = 0; S = 50"}
    tok = types.SimpleNamespace(decode=lambda c, skip_special_tokens=True: f"```python\nK = {c[0]}; S = {c[1]}\n```")
    pol = types.SimpleNamespace(tok=tok, prompt_ids=lambda text: [0])
    answers = {"a": [[0, 40], [0, 60]], "b": [[2, 10], [3, 1]]}
    sampler = lambda prompts, n, adapter, version: [answers["a"], answers["b"]]  # noqa: E731

    def score(items):
        seen.extend(it["seed"] for it in items)
        out = []
        for it in items:
            k, sz = re.findall(r"\d+", it["source"])
            out.append({"valid": True, "kl_bits": float(k) / 4, "size": int(sz), "nodes": int(sz), "edges": 0})
        return out

    train.ORACLE_RUNS = Path(tempfile.mkdtemp())
    args = types.SimpleNamespace(samples=2, eval_seed=7, baselines=True, run_name="t", out=tempfile.mkdtemp())
    log = io.StringIO()
    s = train.evaluate({"heldout": [{"id": "a", "eps": 0.25}, {"id": "b", "eps": 0.25}]}, pol, sampler, score, args, Path("."), 0, log, 3)["heldout"]
    rows = [json.loads(line) for line in log.getvalue().splitlines()]
    a, b = rows[0], rows[1]
    assert a["correct"] and a["beats_teacher"] and a["best"]["size"] == 40, a  # both answers correct: the smaller wins
    assert not b["correct"] and b["best"]["kl"] == 0.5 and b["best_excess_bits"] == 0.25, b  # the smaller excess wins
    assert s["best_correct"] == 0.5 and s["best_beats_teacher"] == 0.5 and s["best_size_over_teacher_when_correct"] == 0.8, s
    assert s["baselines"]["teacher"]["size"] == 50 and abs(a["best"]["reproduces"] - 1.0) < 1e-12, s
    best = json.loads((train.ORACLE_RUNS / "a.t.json").read_text())
    assert best["score"]["size"] == 40 and best["eps"] == 0.25 and best["seed"] == 7
    assert set(seen) == {7}, seen
    assert rows[-1]["step"] == 3


def check_sft_examples():
    """sft_examples: the teacher answer of every TRAINING task (never one outside the pool), with part addresses kept
    as written, and --data examples."""
    import json
    import tempfile

    import train

    d = Path(tempfile.mkdtemp())
    (d / "q.jsonl").write_text(json.dumps({"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "ANS"}]}) + "\n")
    train.TEACHER.clear()
    train.TEACHER.update({"a": "```python\nA\n```", "z": "```python\nZ\n```"})
    train.render = lambda b: "input " + b["id"]
    tok = types.SimpleNamespace(encode=lambda text, add_special_tokens=False: [len(text)])
    pol = types.SimpleNamespace(tok=tok, end=0, prompt_ids=lambda text: [hash(text) % 97], parts=None)
    args = types.SimpleNamespace(data=[str(d / "q.jsonl")], max_model_len=100)
    programs, questions = train.sft_examples(args, pol, [{"id": "a"}, {"id": "b"}])
    assert programs == [([hash("input a") % 97], [len("```python\nA\n```"), 0])], programs
    assert questions == [([hash("Q") % 97], [3, 0])], questions
    train.TEACHER.clear()


def check_share_wake():
    """With one GPU, a sampling call runs inside one vLLM wake: one generate call, the trainer moved to the host and back
    once (a stand-in engine records the moves)."""
    import train

    events = []

    class Engine:
        def wake_up(self):
            events.append("wake")

        def sleep(self, level):
            events.append("sleep")

        def generate(self, prompts, params, lora_request=None, use_tqdm=False):
            events.append(f"generate {len(prompts)}x{params['n']}")
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(token_ids=[j % 2], logprobs=None) for j in range(params["n"])]) for _ in prompts]

    class Model:
        def to(self, dev):
            events.append(f"to {dev}")

    fake = types.ModuleType("vllm")
    fake.SamplingParams = lambda **k: k
    request = types.ModuleType("vllm.lora.request")
    request.LoRARequest = lambda *a: None
    saved = {k: sys.modules.get(k) for k in ("vllm", "vllm.lora", "vllm.lora.request")}
    sys.modules.update({"vllm": fake, "vllm.lora": types.ModuleType("vllm.lora"), "vllm.lora.request": request})
    try:
        inner = train.VllmSampler(types.SimpleNamespace(share_gpu=True, max_tokens=8), 4, 0)
        inner.llm, inner.policy = Engine(), types.SimpleNamespace(model=Model(), dev="cuda:0")
        inner([[0], [0]], 2, Path("."), 0)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    assert events == ["to cpu", "wake", "generate 2x2", "sleep", "to cuda:0"], events


if __name__ == "__main__":
    main()
