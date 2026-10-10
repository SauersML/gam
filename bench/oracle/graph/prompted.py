"""The prompted baseline (#2951 graph oracle): a large model with no training answers the oracle's questions, starting
from the search's answer and revising it after the verifier's report, round after round; every answer is scored by the
same verifier (score.py) on the same held-out questions, next to the search's answer, VPD's answer and the empty graph.
It measures how far reasoning over the search's output and the verifier's reports goes without training, the bar a
trained oracle must reach in one pass and the candidate teacher for its training data.

Each round samples --samples answers per question from the best answer so far (the search's at first) and its report
(rl/train.py's feedback); the best of them and the previous best is kept. The starting answer is cut to --budget tokens
of the model's own tokenizer (a subcomponent written out costs several tokens there), its output to --max-tokens.

  prompted.py --base Qwen/Qwen3-32B-FP8 --rounds 3 --samples 2 --questions 50 --out DIR
Outputs: DIR/eval_samples.jsonl (every answer and baseline with its score, program "search", "vpd", "empty",
"prompted_r<k>"), DIR/summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "rl"))

import native  # noqa: E402
import score  # noqa: E402

ASK = ("You are given an answer to the question above, found by a search, and the verifier's report on it. Write a better "
       "answer in the same format: graph(tokens, targets) with a plain-English docstring saying how the model computes "
       "its prediction, returning a list of steps, the most important first, a comment line above each step. Better "
       "means a lower KL at every description length: keep what lowers the KL early, drop what does not, put the "
       "steps that matter most first, and say in the docstring and comments what each step does. End your reply with "
       "the answer in one ```python block.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="Qwen/Qwen3-32B-FP8")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--questions", type=int, default=50)
    ap.add_argument("--budget", type=int, default=6000, help="tokens of the starting answer (the model's tokenizer)")
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--tp", type=int, default=1, help="GPUs for the model (tensor parallel)")
    ap.add_argument("--gpu-memory", type=float, default=0.9, help="vLLM's share while it samples (it sleeps while the verifier scores)")
    ap.add_argument("--search-dir", type=Path, default=native.TEXTS / "search_heldout")
    ap.add_argument("--seed", type=int, default=1_000_003, help="the verifier's experiment seed (the evaluation's)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    import train
    from prompt import render, split_answer

    a.out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.base)
    count = lambda t: tok.encode(t, add_special_tokens=False)  # noqa: E731
    sc = score.Scorer()
    tasks = []
    for p in native.tasks("heldout")[:a.questions]:
        if (a.search_dir / f"{p.stem}.py").exists():
            t = json.loads(p.read_text())
            t["path"] = str(p)
            tasks.append(t)
    log = open(a.out / "eval_samples.jsonl", "a")

    def write(t, name, src, s):
        s = {k: v for k, v in s.items() if k != "events"}
        log.write(json.dumps({"behavior": t["id"], "program": name, "step": 0, "source": src, "score": s}) + "\n")
        log.flush()

    best = {}
    for t in tasks:
        start = train.cut("```python\n" + (a.search_dir / f"{t['id']}.py").read_text() + "```", a.budget, count)
        src = split_answer(start)[0]
        s_search, s_empty, s_vpd = sc.score(t, [src, score.EMPTY, "vpd"], seed=a.seed)
        for name, x, s in (("search", src, s_search), ("empty", score.EMPTY, s_empty), ("vpd", "vpd", s_vpd)):
            write(t, name, x, s)
        best[t["id"]] = (src, s_search)
    import torch

    torch.cuda.empty_cache()  # the verifier's cached blocks, before vLLM sizes its share
    llm = LLM(model=a.base, tensor_parallel_size=a.tp, max_model_len=32768, gpu_memory_utilization=a.gpu_memory, kv_cache_dtype="fp8", seed=0,
              enable_sleep_mode=True)
    params = SamplingParams(n=a.samples, temperature=0.7, top_p=0.95, max_tokens=a.max_tokens)
    summary = {"search": sorted(score.key(best[t["id"]][1])[1] for t in tasks)}
    for r in range(1, a.rounds + 1):
        chats = [tok.apply_chat_template([{"role": "user", "content": render(t) + "\n\n" + ASK + "\n\nThe answer:\n```python\n" + best[t["id"]][0] + "```\n\n"
                                           + train.feedback(best[t["id"]][1])}], add_generation_prompt=True, enable_thinking=True, tokenize=False) for t in tasks]
        outs = llm.generate(chats, params, use_tqdm=False)
        llm.sleep(level=1)  # weights to host memory: the GPU is the verifier's while it scores
        for t, o in zip(tasks, outs):
            srcs = [split_answer(c.text)[0] for c in o.outputs]
            for src, s in zip(srcs, sc.score(t, srcs, seed=a.seed)):
                write(t, f"prompted_r{r}", src, s)
                if score.key(s) < score.key(best[t["id"]][1]):
                    best[t["id"]] = (src, s)
        torch.cuda.empty_cache()
        llm.wake_up()
        summary[f"round_{r}"] = sorted(score.key(best[t["id"]][1])[1] for t in tasks)
        print(f"round {r}: median best area {summary[f'round_{r}'][len(tasks) // 2]:.2f} (search {summary['search'][len(tasks) // 2]:.2f})", flush=True)
    (a.out / "summary.json").write_text(json.dumps(summary))


if __name__ == "__main__":
    main()
