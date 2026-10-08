"""Checks of the RL scorer's checker path with a stand-in score.Checker (no binary): python test_scorer.py

Programs keep their order across behaviors, workers and seeds; each behavior is loaded on the server
that scores it, and each (behavior, seed) goes in one score_batch request."""

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

    def score_batch(self, sources, experiments=32, seed=0, uniform_seeds=None, **options):
        calls.append((self.path, seed, len(sources)))
        return [{"total_bits": len(x["source"]) + seed, "valid": True, "behavior": self.path} for x in sources]


def main():
    sys.modules["score"] = types.SimpleNamespace(Checker=Checker)
    sys.modules["run"] = types.SimpleNamespace(load_behavior=lambda c, path: c.request({"op": "behavior", "path": path, "manifest": None}))
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
    check_repair()
    check_evaluate()
    check_sft_examples()
    check_rescore()
    check_valid_sampler()
    print("ok: checker scorer order, behaviors and one batch per (behavior, seed); repair keeps a revision only when it lowers S; evaluation summary; SFT data uses each training behavior's best program only; rescore; validity redraws")


def check_repair():
    """train.repair replaces a behavior's best program by a revision only when the revision is valid where
    the best was not, or lowers S; the revision prompt carries the program and the checker's error."""
    import train

    shown = []
    train.render = lambda b: f"input {b['id']}"
    tok = types.SimpleNamespace(decode=lambda c, skip_special_tokens=True: f"```python\nX = {c[0]}\n```")
    pol = types.SimpleNamespace(tok=tok, prompt_ids=lambda text: shown.append(text) or [0])
    sampler = lambda prompts, n, adapter, version: [[[v] for v in (70, 30)] for _ in prompts]  # noqa: E731
    score = lambda items: [{"valid": True, "total_bits": float(it["source"].split("=")[1])} for it in items]  # noqa: E731
    best = [{"completion": [0], "text": "```python\nbad(\n```", "score": {"valid": False, "total_bits": 1e4, "error": "line 1: syntax error"}},
            {"completion": [0], "text": "```python\nX = 10\n```", "score": {"valid": True, "total_bits": 10.0}}]
    args = types.SimpleNamespace(repair=1, samples=2, uniform_seeds=0, experiments=16)
    replaced = train.repair([{"id": "a"}, {"id": "b"}], best, pol, sampler, score, args, Path("."), 0)
    assert replaced == {0} and best[0]["score"]["total_bits"] == 30.0 and best[0]["completion"] == [30] and best[1]["score"]["total_bits"] == 10.0, (replaced, best)
    assert "line 1: syntax error" in shown[0] and "bad(" in shown[0] and shown[0].startswith("input a")


def check_evaluate():
    """train.evaluate scores the policy's programs and the baselines under the evaluation seed only, and
    its summary averages per set: mean single-sample S, best of N, each baseline over the behaviors that have it."""
    import io
    import json

    import train

    seen = []
    train.render = lambda b: b["id"]
    train.baselines = lambda b: {"empty": "X = 100"} if b["id"] == "a" else {}
    tok = types.SimpleNamespace(decode=lambda c, skip_special_tokens=True: f"```python\nX = {c[0]}\n```")
    pol = types.SimpleNamespace(tok=tok, prompt_ids=lambda text: [0])
    sampler = lambda prompts, n, adapter, version: [[[10], [30]] for _ in prompts]  # noqa: E731

    def score(items):
        seen.extend((it["seed"], it.get("experiments")) for it in items)
        return [{"valid": True, "total_bits": float(it["source"].split("=")[1]), "exec_error_bits": float(it["source"].split("=")[1])} for it in items]

    import tempfile

    train.ORACLE_RUNS = Path(tempfile.mkdtemp())
    args = types.SimpleNamespace(samples=2, eval_seed=7, eval_experiments=16, baselines=True, run_name="t", out=tempfile.mkdtemp())
    log = io.StringIO()
    out = train.evaluate({"heldout_behaviors": [{"id": "a"}, {"id": "b"}], "heldout_prompts": []}, pol, sampler, score, args, Path("."), 0, log, 3)
    s = out["heldout_behaviors"]
    assert s["mean_bits"] == 20.0 and s["best_of_n_bits"] == 10.0 and s["baselines"] == {"empty": 100.0}, s
    assert s["below_empty_fraction"] == 1.0 and abs(s["best_recovered"] - 0.9) < 1e-12, s  # behavior a: 1 - 10/100
    best = json.loads((train.ORACLE_RUNS / "a.t.json").read_text())
    assert best["score"]["total_bits"] == 10.0 and best["experiments"] == 16 and best["seed"] == 7
    assert sum(1 for _ in open(Path(args.out) / "eval_samples.jsonl")) == 5  # 4 oracle programs + 1 baseline
    assert s["oracle_mean_bits_on_baseline_behaviors"] == {"empty": 20.0} and "heldout_prompts" not in out
    assert set(seen) == {(7, 16)}, seen
    rows = [json.loads(line) for line in log.getvalue().splitlines()]
    assert [r.get("behavior") for r in rows[:2]] == ["a", "b"] and rows[-1]["step"] == 3



def check_sft_examples():
    """sft_examples keeps each TRAINING behavior's lowest-S program and never a behavior outside the pool."""
    import json
    import tempfile

    import train

    d = Path(tempfile.mkdtemp())
    for name, behavior, bits, source in (("a1", "a", 50.0, "A1"), ("a2", "a", 20.0, "A2"), ("z", "z", 1.0, "Z")):
        (d / f"{name}.json").write_text(json.dumps({"behavior": behavior, "source": source, "score": {"total_bits": bits}}))
    (d / "q.jsonl").write_text(json.dumps({"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "ANS"}]}) + "\n")
    train.render = lambda b: "input " + b["id"]
    tok = types.SimpleNamespace(encode=lambda text, add_special_tokens=False: [len(text)] if "A1" not in text else [-1])
    pol = types.SimpleNamespace(tok=tok, end=0, prompt_ids=lambda text: [hash(text) % 97])
    args = types.SimpleNamespace(programs=[str(d / "*.json")], data=[str(d / "q.jsonl")], max_model_len=100)
    programs, questions = train.sft_examples(args, pol, [{"id": "a"}, {"id": "b"}])
    assert len(programs) == 1 and programs[0][1] == [len("```python\nA2\n```"), 0], programs
    assert questions == [([hash("Q") % 97], [3, 0])], questions



def check_rescore():
    """train.rescore scores saved programs again with the given seed, experiments and checker options and
    summarizes them against the rescored empty program."""
    import json
    import tempfile

    import train

    d = Path(tempfile.mkdtemp())
    (d / "b.json").write_text(json.dumps({"id": "b", "model": "vpd4l", "prompts": []}))
    rows = [{"set": "heldout_behaviors", "step": 0, "run": "r", "behavior": "b", "behavior_path": str(d / "b.json"), "program": p, "source": f"X = {v}", "score": {"total_bits": 0.0}}
            for p, v in (("oracle", 10), ("oracle", 40), ("empty", 50))]
    (d / "eval_samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    seen = []

    def score(items):
        seen.extend((it["seed"], it["experiments"], json.dumps(it["options"])) for it in items)
        return [{"valid": True, "total_bits": 2 * float(it["source"].split("=")[1]), "exec_error_bits": 2 * float(it["source"].split("=")[1])} for it in items]

    train.ir_signature = lambda source, model: source  # stand-in programs: each its own score
    args = types.SimpleNamespace(samples_from=[str(d / "eval_samples.jsonl")], score_options='{"families": ["swap"]}', eval_seed=5, eval_experiments=16, out=str(d), rescore_tag="t",
                                 behaviors=str(d), model="vpd4l", summary_only=False)
    s = train.rescore(args, score)["heldout_behaviors/r"]
    assert s["best_of_n_bits"] == 20.0 and s["baselines"] == {"empty": 100.0} and abs(s["best_recovered"] - 0.8) < 1e-12, s
    assert set(seen) == {(5, 16, '{"families": ["swap"]}')}, seen
    assert sum(1 for _ in open(d / "rescore_t.jsonl")) == 3



def check_valid_sampler():
    """ValidSampler redraws only the invalid programs, keeps n per prompt and reports the valid share
    before and after."""
    import train

    good = "from mech import node, edges, L, logits\nh = node(L[1].head[0])\nedges(h >> logits)\n"
    tok = types.SimpleNamespace(decode=lambda c, skip_special_tokens=True: good if c[0] > 0 else "nonsense(")
    calls = []

    class Inner:
        logprob_sums = None

        def __call__(self, prompts, n, adapter, version):
            calls.append((len(prompts), n))
            return [[[1 if len(calls) > 1 else (j % 2)] for j in range(n)] for _ in prompts]

    vs = train.ValidSampler(Inner(), tok, "vpd4l", 2)
    out = vs([[0], [0]], 4, Path("."), 0)
    assert calls == [(2, 4), (4, 1)], calls  # 4 invalid slots redrawn once
    assert all(len(g) == 4 for g in out) and vs.stats == {"first_valid": 0.5, "final_valid": 1.0}, vs.stats


if __name__ == "__main__":
    main()
