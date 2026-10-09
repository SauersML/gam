"""Checks of the RL scorer's checker path with a stand-in score.Checker (no binary), and of the evaluation, SFT data and
vLLM wake around it: python test_scorer.py

Programs keep their order across tasks, workers and seeds; each task is loaded on the server that scores it, and each
(task, seed) goes in one score_batch request."""

from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scorer  # noqa: E402

calls = []


class Checker:
    def __init__(self, model, export=None, memory_gib=None, views=None):
        self.model, self.path = model, None

    def request(self, message):
        self.path = message["path"]

    def behavior(self, path):  # score.Checker.behavior: the server's default site-operation manifest
        self.request({"op": "behavior", "path": path})

    def score_batch(self, sources, experiments=32, seed=0, uniform_seeds=None, **options):
        calls.append((self.path, seed, len(sources)))
        return [{"total_bits": len(x) + seed, "valid": True, "behavior": self.path} for x in sources]


def main():
    import score

    real, score.Checker = score.Checker, Checker
    scorer.WORKERS = 3
    items = [{"source": "x" * i, "behavior": {"model": "vpd4l", "path": f"b{i % 4}"}, "seed": 1 + i % 2} for i in range(16)]
    out = scorer.checker(items)
    assert [r["total_bits"] for r in out] == [i + 1 + i % 2 for i in range(16)]
    assert [r["behavior"] for r in out] == [f"b{i % 4}" for i in range(16)]
    assert sorted(calls) == sorted({(f"b{i % 4}", 1 + i % 2, 4) for i in range(16)}), calls
    calls.clear()
    scorer.BATCH = 3  # a request holds at most BATCH programs
    assert [r["total_bits"] for r in scorer.checker(items)] == [i + 1 + i % 2 for i in range(16)]
    assert sorted(n for _, _, n in calls) == [1] * 4 + [3] * 4, calls
    scorer.BATCH = 4
    score.Checker = real
    check_evaluate()
    check_sft_examples()
    check_share_wake()
    print("ok: checker scorer order, tasks and one batch per (task, seed); evaluation ranks against VPD's answer; SFT data is the training tasks' teacher answers; one vLLM wake per sampling call")


def check_evaluate():
    """train.evaluate scores the policy's answers and the baselines under the evaluation seed only; per task its best
    answer by score.order against VPD's (the teacher baseline): faithful when its KL is no larger, beating VPD when it
    names fewer pairs too."""
    import io
    import json
    import re
    import tempfile

    import train

    seen = []
    train.render = lambda b: b["id"]
    train.baselines = lambda b: {"empty": "K = 100; P = 0", "teacher": "K = 5; P = 50"}
    tok = types.SimpleNamespace(decode=lambda c, skip_special_tokens=True: f"```python\nK = {c[0]}; P = {c[1]}\n```")
    pol = types.SimpleNamespace(tok=tok, prompt_ids=lambda text: [0])
    answers = {"a": [[4, 40], [3, 60]], "b": [[6, 10], [9, 1]]}
    sampler = lambda prompts, n, adapter, version: [answers["a"], answers["b"]]  # noqa: E731

    def score(items):
        seen.extend((it["seed"], it.get("experiments")) for it in items)
        out = []
        for it in items:
            k, p = re.findall(r"\d+", it["source"])
            out.append({"valid": True, "exec_error_bits": float(k), "pairs": int(p)})
        return out

    train.ORACLE_RUNS = Path(tempfile.mkdtemp())
    args = types.SimpleNamespace(samples=2, eval_seed=7, experiments=0, baselines=True, run_name="t", out=tempfile.mkdtemp())
    log = io.StringIO()
    s = train.evaluate({"heldout": [{"id": "a"}, {"id": "b"}]}, pol, sampler, score, args, Path("."), 0, log, 3)["heldout"]
    rows = [json.loads(line) for line in log.getvalue().splitlines()]
    a, b = rows[0], rows[1]
    assert a["faithful"] and a["beats_vpd"] and a["best"]["pairs"] == 40, a  # both answers as faithful as VPD: fewer pairs wins
    assert not b["faithful"] and b["best"]["kl"] == 6.0 and b["best_excess_bits"] == 1.0, b  # the smaller excess wins
    assert s["best_faithful"] == 0.5 and s["best_beats_vpd"] == 0.5 and s["best_pairs_over_vpd_when_faithful"] == 0.8, s
    assert s["baselines"]["teacher"]["pairs"] == 50 and abs(a["best"]["reproduces"] - 0.96) < 1e-12, s
    best = json.loads((train.ORACLE_RUNS / "a.t.json").read_text())
    assert best["score"]["pairs"] == 40 and best["reference"]["pairs"] == 50 and best["seed"] == 7
    assert set(seen) == {(7, 0)}, seen
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
