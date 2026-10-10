"""The prompted baseline (#2951 graph oracle): a large model with no training answers the oracle's questions, starting
from the search's answer and revising it after the verifier's report, round after round; every answer is scored by the
same verifier (score.py) on the same held-out questions, next to the search's answer, VPD's answer and the empty graph.
It measures how far reasoning over the search's output and the verifier's reports goes without training, the bar a
trained oracle must reach in one pass and the candidate teacher for its training data.

Each round samples --samples answers per question from the best answer so far (the search's at first) and its report
(rl/train.py's feedback); the best of them and the previous best is kept. The starting answer is cut to --budget tokens
of the model's own tokenizer (a subcomponent written out costs several tokens there), its output to --max-tokens.

The model samples with vLLM on a CUDA GPU or with MLX on a Mac (--backend mlx, an MLX checkpoint as --base); either
way it leaves the accelerator's memory while the verifier scores. Only the text after the model's reasoning (after
</think>) is its answer: a reply cut off while reasoning has none.

  prompted.py --base Qwen/Qwen3-32B-FP8 --rounds 3 --samples 2 --questions 50 --out DIR
  prompted.py --backend mlx --base mlx-community/Qwen3-30B-A3B-Thinking-2507-4bit --out DIR
A rerun into the same DIR reuses the baselines' scores already there. Outputs: DIR/eval_samples.jsonl (every answer and baseline with its score, program "search" (the starting answer),
"search_full" (the search's whole answer), "vpd", "empty", "prompted_r<k>"), DIR/summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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


LOOKUP = ("After the report: the answer's subcomponents, each with its layer and weight matrix, its position and token in "
          "the text, and for an attention or MLP output the tokens its write vector raises most at the prediction (its "
          "logit lens); then other subcomponents the search found, in the order it ranked them, which the answer may use.")


def nodes_of(t, src) -> list:
    """An answer's nodes (weight matrix, position, subcomponent) in the order its steps add them; [] when it does not run."""
    import mech

    ir = mech.trace_inline(src, "vpd4l", t)
    if not ir["valid"]:
        return []
    g = ir["graph"]
    return [(native.site_name(layer, kind), pos, c) for layer, kind, pos, c in
            (g["nodes"][i] for i in sorted(range(len(g["nodes"])), key=lambda i: g["node_step"][i]))]


def words_of(nat, t, nodes, budget, count) -> str:
    """Nodes in words (describe.words), one line each in the answer's own notation, while within `budget` tokens
    (count: text -> token ids)."""
    import describe
    import mech

    strings = mech.behavior_tokens(t, "vpd4l")["sequences"][0][1]
    lens = nat.lens([nd for nd in nodes if isinstance(nd[2], int)])
    short = {v: k for k, v in mech.SITES.items()}
    lines, used = [], 0
    for nd in nodes:
        layer, kind = native._layer_kind(nd)
        line = f'({nd[1]}, "<p:{layer}.{short[kind]}.{nd[2]}>"): {describe.words(nd, strings, lens)}'
        used += len(count(line)) + 1
        if used > budget:
            lines.append(f"... and {len(nodes) - len(lines)} more")
            break
        lines.append(line)
    return "\n".join(lines)


def final(text: str) -> str:
    """The answer part of a reasoning model's reply: after Qwen's </think> or in gpt-oss's final channel; "" for a reply
    cut off while reasoning."""
    for mark in ("</think>", "<|channel|>final<|message|>"):
        if mark in text:
            return text.split(mark)[-1]
    return ""


class Vllm:
    """Sampling with vLLM on CUDA; sleep() moves the weights to host memory and drops the KV cache."""

    def __init__(self, a):
        import torch
        from vllm import LLM, SamplingParams

        torch.cuda.empty_cache()  # the verifier's cached blocks, before vLLM sizes its share
        self.llm = LLM(model=a.base, tensor_parallel_size=a.tp, max_model_len=a.max_tokens + 24576, gpu_memory_utilization=a.gpu_memory,
                       kv_cache_dtype=a.kv_dtype, seed=0, enable_sleep_mode=True)
        self.params = SamplingParams(n=a.samples, temperature=0.7, top_p=0.95, max_tokens=a.max_tokens, skip_special_tokens=False)

    def generate(self, chats):
        return [[c.text for c in o.outputs] for o in self.llm.generate(chats, self.params, use_tqdm=False)]

    def sleep(self):
        self.llm.sleep(level=1)

    def wake(self):
        import torch

        torch.cuda.empty_cache()
        self.llm.wake_up()


class Mlx:
    """Sampling with MLX on a Mac: --samples copies of each prompt in one batch of at most --batch replies at a time;
    sleep() unloads the model (the verifier shares the Mac's memory) and wake() loads it again."""

    def __init__(self, a):
        import mlx.core as mx

        mx.set_cache_limit(2 << 30)  # freed buffers MLX keeps for reuse count in the process's footprint
        self.a = a
        self.wake()

    def generate(self, chats):
        import mlx.core as mx
        from mlx_lm.generate import batch_generate
        from mlx_lm.sample_utils import make_sampler

        prompts = [self.tok.encode(c, add_special_tokens=False) for c in chats for _ in range(self.a.samples)]
        texts = batch_generate(self.model, self.tok, prompts, max_tokens=self.a.max_tokens, sampler=make_sampler(temp=0.7, top_p=0.95),
                               completion_batch_size=self.a.batch, prefill_batch_size=min(self.a.batch, 4), verbose=True).texts
        mx.clear_cache()
        return [texts[i * self.a.samples:(i + 1) * self.a.samples] for i in range(len(chats))]

    def sleep(self):
        import gc

        import mlx.core as mx

        self.model = None
        gc.collect()
        mx.clear_cache()

    def wake(self):
        from mlx_lm import load

        native.free(native.device())  # torch's cached MPS blocks, before the model loads
        self.model, self.tok = load(self.a.base)


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
    ap.add_argument("--backend", choices=("vllm", "mlx"), default="vllm")
    ap.add_argument("--evidence-tokens", type=int, default=0, help="tokens of the answer's subcomponents in words after the report, and "
                    "as many of the search's other subcomponents (0: none)")
    ap.add_argument("--kv-dtype", default="auto", help="vLLM's KV cache type (fp8 halves it on Ada and Hopper; its kernels fail to build on RTX PRO 4500 Blackwell)")
    ap.add_argument("--batch", type=int, default=3, help="MLX: replies generated at once (each takes ~4.6 GB of the Mac's memory at 28k tokens)")
    ap.add_argument("--split", choices=("heldout", "hard"), default="heldout")
    ap.add_argument("--search-dir", type=Path, help="the search's answers (default texts/search_<split>)")
    ap.add_argument("--seed", type=int, default=1_000_003, help="the verifier's experiment seed (the evaluation's)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    import train
    from prompt import render, split_answer

    a.out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.base)
    count = lambda t: tok.encode(t, add_special_tokens=False)  # noqa: E731
    sc = score.Scorer()
    tasks = []
    a.search_dir = a.search_dir or native.TEXTS / f"search_{a.split}"
    for p in native.tasks(a.split)[:a.questions]:
        if (a.search_dir / f"{p.stem}.py").exists():
            t = json.loads(p.read_text())
            t["path"] = str(p)
            tasks.append(t)
    samples = a.out / "eval_samples.jsonl"
    done = {}  # (question, baseline) -> its row, from an earlier run into the same directory
    if samples.exists():
        for line in open(samples):
            r = json.loads(line)
            if not r["program"].startswith("prompted"):
                done[(r["behavior"], r["program"])] = r
    log = open(samples, "a")

    def write(t, name, src, s, reply=None):
        s = {k: v for k, v in s.items() if k != "events"}
        log.write(json.dumps({"behavior": t["id"], "program": name, "step": 0, "source": src, "score": s, **({"reply": reply} if reply is not None else {})}) + "\n")
        log.flush()

    best, full_area, search_nodes = {}, {}, {}
    for t in tasks:
        full = (a.search_dir / f"{t['id']}.py").read_text()
        src = split_answer(train.cut("```python\n" + full + "```", a.budget, count))[0]
        names = ("search", "search_full", "empty", "vpd")
        if all((t["id"], n) in done for n in names):
            s_search, s_full = done[(t["id"], "search")]["score"], done[(t["id"], "search_full")]["score"]
        else:
            s_search, s_full, s_empty, s_vpd = sc.score(t, [src, full, score.EMPTY, "vpd"], seed=a.seed)
            for name, x, s in zip(names, (src, full, score.EMPTY, "vpd"), (s_search, s_full, s_empty, s_vpd)):
                write(t, name, x, s)
        best[t["id"]] = (src, s_search)
        full_area[t["id"]] = score.key(s_full)[1]
        search_nodes[t["id"]] = nodes_of(t, full) if a.evidence_tokens else []
    llm = (Mlx if a.backend == "mlx" else Vllm)(a) if a.rounds else None  # --rounds 0: the baselines' scores only
    summary = {"search": sorted(score.key(best[t["id"]][1])[1] for t in tasks), "search_full": sorted(full_area.values())}
    for r in range(1, a.rounds + 1):
        chats = []
        for t in tasks:
            src = best[t["id"]][0]
            ask = render(t) + "\n\n" + ASK + (" " + LOOKUP if a.evidence_tokens else "") + "\n\nThe answer:\n```python\n" + src + "```\n\n" + train.feedback(best[t["id"]][1])
            if a.evidence_tokens:
                used = nodes_of(t, src)
                seen = set(used)
                ask += ("\n\nThe answer's subcomponents:\n" + words_of(sc.nat, t, used, a.evidence_tokens, count)
                        + "\n\nOther subcomponents the search found:\n" + words_of(sc.nat, t, [n for n in search_nodes[t["id"]] if n not in seen], a.evidence_tokens, count))
            chats.append(tok.apply_chat_template([{"role": "user", "content": ask}], add_generation_prompt=True, enable_thinking=True,
                                                 reasoning_effort="high", tokenize=False))
        clock = time.time()
        outs = llm.generate(chats)
        sampled = time.time() - clock
        llm.sleep()  # the accelerator is the verifier's while it scores
        for t, texts in zip(tasks, outs):
            srcs = [split_answer(final(x))[0] if final(x) else "" for x in texts]
            for src, x, s in zip(srcs, texts, sc.score(t, srcs, seed=a.seed)):
                write(t, f"prompted_r{r}", src, s, reply=x)
                if score.key(s) < score.key(best[t["id"]][1]):
                    best[t["id"]] = (src, s)
        llm.wake()
        summary[f"round_{r}"] = sorted(score.key(best[t["id"]][1])[1] for t in tasks)
        wins = sum(score.key(best[t["id"]][1])[1] < full_area[t["id"]] for t in tasks)
        print(f"round {r}: median best area {summary[f'round_{r}'][len(tasks) // 2]:.2f} (search's first {a.budget} tokens "
              f"{summary['search'][len(tasks) // 2]:.2f}, whole search {summary['search_full'][len(tasks) // 2]:.2f}, below it on {wins}/{len(tasks)}); "
              f"sampling {sampled / 60:.0f} min, scoring {(time.time() - clock - sampled) / 60:.0f} min", flush=True)
    (a.out / "summary.json").write_text(json.dumps(summary))


if __name__ == "__main__":
    main()
