"""The prompted baseline (#2951 graph oracle): a large model with no training answers the oracle's questions, starting
from the search's answer and revising it after the verifier's report, round after round; every answer is scored by the
same verifier (score.py) on the same held-out questions, next to the search's answer, VPD's answer and the empty graph.
It measures how far reasoning over the search's output and the verifier's reports goes without training, the bar a
trained oracle must reach in one pass and the candidate teacher for its training data.

Each round samples --samples answers per question from the best answer so far (the search's at first) and its report
(prompt.feedback); the best of them and the previous best is kept. The starting answer is cut to --budget tokens
of the model's own tokenizer (a subcomponent written out costs several tokens there), its output to --max-tokens.

The model samples with vLLM on a CUDA GPU or with MLX on a Mac (--backend mlx, an MLX checkpoint as --base); either
way it leaves the accelerator's memory while the verifier scores. Only the text after the model's reasoning (after
</think>) is its answer: a reply cut off while reasoning has none.

  prompted.py --base Qwen/Qwen3-32B-FP8 --rounds 3 --samples 2 --questions 50 --out DIR
  prompted.py --backend mlx --base mlx-community/Qwen3-30B-A3B-Thinking-2507-4bit --out DIR
A rerun into the same DIR reuses the baselines' scores already there and continues after the rounds already done. Outputs: DIR/eval_samples.jsonl (every answer and baseline with its score, program "search" (the starting answer),
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
          "logit lens); then other subcomponents the search found, in the order it ranked them, which the answer may use; "
          "then where the prediction responds to the text: the positions whose token, changed to another the model finds "
          "likely there, moves the prediction most, and at each of them and at the last position the subcomponents of each "
          "kind that write most there beyond what they write elsewhere in this text.")


KINDS = {"q_proj": "attention query", "k_proj": "attention key", "v_proj": "attention value", "o_proj": "attention output",
         "c_fc": "MLP input", "down_proj": "MLP output"}


def words(nd, strings: list[str], lens: dict) -> str:
    """A subcomponent in words: its layer and weight matrix, position and token, and for an attention or MLP output the
    tokens its write vector raises most at the prediction (lens: native.lens)."""
    layer, kind = native._layer_kind(nd)
    w = f"layer {layer} {KINDS[kind]} subcomponent {nd[2]} at position {nd[1]} ({strings[nd[1]]!r})"
    return w + (f", which writes toward {', '.join(map(repr, lens[nd]))}" if nd in lens else "")


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
    """Nodes in words (words()), one line each in the answer's own notation, while within `budget` tokens
    (count: text -> token ids)."""
    import mech

    strings = mech.behavior_tokens(t, "vpd4l")["sequences"][0][1]
    lens = nat.lens([nd for nd in nodes if isinstance(nd[2], int)])
    short = {v: k for k, v in mech.SITES.items()}
    lines, used = [], 0
    for nd in nodes:
        layer, kind = native._layer_kind(nd)
        line = f'({nd[1]}, "<p:{layer}.{short[kind]}.{nd[2]}>"): {words(nd, strings, lens)}'
        used += len(count(line)) + 1
        if used > budget:
            lines.append(f"... and {len(nodes) - len(lines)} more")
            break
        lines.append(line)
    return "\n".join(lines)


def responses(nat, t, budget, count) -> str:
    """Where the prediction responds to the text, measured: the last position and the positions whose token, changed
    to another the model finds likely there, moves the prediction most (native.sensitivity), in that order; at each,
    the two subcomponents of each kind (query, key, value, attention output, MLP input, MLP output, over the layers)
    that write most there beyond what they write on average over the text (native.contributions; the excess, so
    subcomponents active everywhere do not fill every position's list), in the answer's notation, while within
    `budget` tokens."""
    import mech
    import torch

    prompt = t["prompts"][0]
    ids, targets = prompt["token_ids"], prompt["target_positions"]
    last = max(targets)
    if last < 1:
        return ""
    strings = mech.behavior_tokens(t, "vpd4l")["sequences"][0][1]
    moved, _ = nat.sensitivity(ids, targets, torch.Generator(device="cpu").manual_seed(native.task_seed(t["id"])))
    contrib = {n: w - w.mean(0, keepdim=True) for n, w in nat.contributions(ids, targets).items()}
    short = {v: k for k, v in mech.SITES.items()}
    lines, used = [], 0
    for pos in [last] + [int(j) + 1 for j in moved.argsort(descending=True) if int(j) + 1 != last]:
        best = {}
        for n, w in contrib.items():
            kind = native._layer_kind((n, 0, 0))[1]
            v, c = w[pos].topk(2)
            best.setdefault(kind, []).extend((float(a), (n, pos, int(b))) for a, b in zip(v, c))
        nodes = [nd for kind in mech.SITES.values() for _, nd in sorted(best.get(kind, []), reverse=True)[:2]]
        lens = nat.lens(nodes)
        items = []
        for nd in nodes:
            layer, kind = native._layer_kind(nd)
            items.append(f'"<p:{layer}.{short[kind]}.{nd[2]}>" (layer {layer} {KINDS[kind]}'
                         + (f", writes toward {', '.join(map(repr, lens[nd]))}" if nd in lens else "") + ")")
        head = f"position {pos} ({strings[pos]!r})" + (", the last" if pos == last else f", changing its token moves the prediction {float(moved[pos - 1]) / 0.6931:.2f} bits")
        line = head + "; writing most there: " + ", ".join(items)
        used += len(count(line)) + 1
        if used > budget:
            break
        lines.append(line)
    return "\n".join(lines)


_SCORER = None


def _start_worker():
    global _SCORER
    _SCORER = score.Scorer()


def _score_job(job):
    """One question's answers scored in a worker (score.Scorer.score); events left out (they stay with the worker)."""
    t, srcs, seed = job
    return [{k: v for k, v in s.items() if k not in ("events", "base")} for s in _SCORER.score(t, srcs, seed=seed)]


def score_all(sc, jobs: list, seed: int, workers: int) -> list:
    """Each (question, answers) job's scores, in order: in this process, or with workers > 1 in that many processes
    (the verifier's forward passes leave a GPU mostly idle, one process at a time; the pool closes after, so vLLM gets
    its memory back)."""
    if workers > 1 and len(jobs) > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        from concurrent.futures.process import BrokenProcessPool

        try:  # a worker that dies (out of memory) breaks the pool instead of hanging it; the calls then run here
            with ProcessPoolExecutor(min(workers, len(jobs)), mp_context=multiprocessing.get_context("spawn"), initializer=_start_worker) as pool:
                return list(pool.map(_score_job, [(t, srcs, seed) for t, srcs in jobs]))
        except (BrokenProcessPool, RuntimeError) as e:  # a dead worker, or one out of memory (torch's errors are RuntimeErrors)
            print(f"score_all: the pool failed ({type(e).__name__}: {str(e)[:200]}); scoring in this process", flush=True)
    return [sc.score(t, srcs, seed=seed) for t, srcs in jobs]


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
    ap.add_argument("--workers", type=int, default=1, help="processes scoring answers in parallel (a pod: 8; the Mac: 1)")
    ap.add_argument("--kv-dtype", default="auto", help="vLLM's KV cache type (fp8 halves it on Ada and Hopper; its kernels fail to build on RTX PRO 4500 Blackwell)")
    ap.add_argument("--batch", type=int, default=3, help="MLX: replies generated at once (each takes ~4.6 GB of the Mac's memory at 28k tokens)")
    ap.add_argument("--split", choices=("heldout", "hard"), default="heldout")
    ap.add_argument("--search-dir", type=Path, help="the search's answers (default texts/search_<split>)")
    ap.add_argument("--seed", type=int, default=1_000_003, help="the verifier's experiment seed (the evaluation's)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    import train
    from prompt import feedback, render, split_answer

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
    done, earlier = {}, []  # baselines' rows and the prompted answers' rows of an earlier run into the same directory
    if samples.exists():
        for line in open(samples):
            r = json.loads(line)
            if r["program"].startswith("prompted"):
                earlier.append(r)
            else:
                done[(r["behavior"], r["program"])] = r
    log = open(samples, "a")

    def write(t, name, src, s, reply=None):
        s = {k: v for k, v in s.items() if k != "events"}
        log.write(json.dumps({"behavior": t["id"], "program": name, "step": 0, "source": src, "score": s, **({"reply": reply} if reply is not None else {})}) + "\n")
        log.flush()

    best, full_area, search_nodes, respond = {}, {}, {}, {}
    names = ("search", "search_full", "empty", "vpd")
    starts = {}
    for t in tasks:
        full = (a.search_dir / f"{t['id']}.py").read_text()
        starts[t["id"]] = (split_answer(train.cut("```python\n" + full + "```", a.budget, count))[0], full)
    pending = [t for t in tasks if not all((t["id"], n) in done for n in names)]
    scored = score_all(sc, [(t, [*starts[t["id"]], score.EMPTY, "vpd"]) for t in pending], a.seed, a.workers)
    for t, ss in zip(pending, scored):
        for name, x, s in zip(names, (*starts[t["id"]], score.EMPTY, "vpd"), ss):
            write(t, name, x, s)
            done[(t["id"], name)] = {"score": s}
    for t in tasks:
        src, full = starts[t["id"]]
        s_search, s_full = done[(t["id"], "search")]["score"], done[(t["id"], "search_full")]["score"]
        best[t["id"]] = (src, s_search)
        full_area[t["id"]] = score.key(s_full)[1]
        search_nodes[t["id"]] = nodes_of(t, full) if a.evidence_tokens else []
        respond[t["id"]] = responses(sc.nat, t, a.evidence_tokens, count) if a.evidence_tokens else ""
    summary = {"search": sorted(score.key(best[t["id"]][1])[1] for t in tasks), "search_full": sorted(full_area.values())}
    first = 1 + max((int(r["program"][len("prompted_r"):]) for r in earlier), default=0)  # an earlier run's finished rounds
    for r in earlier:  # its best answers carry on
        if r["behavior"] in best and r["score"].get("curve") and score.key(r["score"]) < score.key(best[r["behavior"]][1]):
            best[r["behavior"]] = (r["source"], r["score"])
    llm = (Mlx if a.backend == "mlx" else Vllm)(a) if a.rounds >= first else None  # --rounds 0: the baselines' scores only
    for r in range(first, a.rounds + 1):
        chats = []
        for t in tasks:
            src = best[t["id"]][0]
            ask = render(t) + "\n\n" + ASK + (" " + LOOKUP if a.evidence_tokens else "") + "\n\nThe answer:\n```python\n" + src + "```\n\n" + feedback(best[t["id"]][1])
            if a.evidence_tokens:
                used = nodes_of(t, src)
                seen = set(used)
                ask += ("\n\nThe answer's subcomponents:\n" + words_of(sc.nat, t, used, a.evidence_tokens, count)
                        + "\n\nOther subcomponents the search found:\n" + words_of(sc.nat, t, [n for n in search_nodes[t["id"]] if n not in seen], a.evidence_tokens, count)
                        + "\n\nWhere the prediction responds to the text:\n" + respond[t["id"]])
            chats.append(tok.apply_chat_template([{"role": "user", "content": ask}], add_generation_prompt=True, enable_thinking=True,
                                                 reasoning_effort="high", tokenize=False))
        clock = time.time()
        outs = llm.generate(chats)
        sampled = time.time() - clock
        llm.sleep()  # the accelerator is the verifier's while it scores
        srcs_all = [[split_answer(final(x))[0] if final(x) else "" for x in texts] for texts in outs]
        for t, texts, srcs, ss in zip(tasks, outs, srcs_all, score_all(sc, list(zip(tasks, srcs_all)), a.seed, a.workers)):
            for src, x, s in zip(srcs, texts, ss):
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
