"""Plain English for search answers (#2951 graph oracle): the base oracle model writes, for a search answer's graph,
the docstring (how the model arrives at its prediction) and one sentence per step, so the bootstrap SFT examples carry
English in the model's own words, with no template; SFT on the model's own text moves it least.

The model reads the text with its positions, and each step's connections in words: which layer and weight matrix each
subcomponent belongs to, its position and token there, and for an attention or MLP output the tokens its write vector
raises most at the prediction (native.lens). The answer is cut first to the oracle's output budget (train.fit).

  describe.py SEARCH_DIR --split train --out DIR [--base Qwen/Qwen3-4B] [--max-tokens 12288]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "rl"))

import mech  # noqa: E402
import native  # noqa: E402

KINDS = {"q_proj": "attention query", "k_proj": "attention key", "v_proj": "attention value", "o_proj": "attention output",
         "c_fc": "MLP input", "down_proj": "MLP output"}
ASK = ("A 4-layer language model reads the text below and predicts the next token after position {t}. Its most likely "
       "next tokens are {top}. A search found the computational graph below: subcomponents of the model's weight "
       "matrices at positions of the text and which outputs each reads, in steps, the most important first; running the "
       "graph alone reproduces the model's prediction closely. Write, first, a short paragraph in plain English explaining "
       "how the model arrives at its prediction, in terms of the words of the text; then one sentence for each step saying "
       "what it adds. Use exactly this format:\nEXPLANATION: <paragraph>\nSTEP 1: <sentence>\nSTEP 2: <sentence>\n...\n\n"
       "Text (position: token):\n{text}\n\nGraph:\n{graph}")


def words(nd, strings: list[str], lens: dict) -> str:
    layer, kind = native._layer_kind(nd)
    w = f"layer {layer} {KINDS[kind]} subcomponent {nd[2]} at position {nd[1]} ({strings[nd[1]]!r})"
    return w + (f", which writes toward {', '.join(map(repr, lens[nd]))}" if nd in lens else "")


def render(steps: list, strings: list[str], lens: dict) -> str:
    lines, prev = [], native.Graph()
    for k, g in enumerate(steps):
        seen = {(r, w) for r, ws in prev.parents.items() for w in ws}
        new = [(r, w) for r, ws in g.parents.items() for w in ws if (r, w) not in seen] + [(None, w) for w in g.out if w not in prev.out]
        lines.append(f"Step {k + 1}:")
        for r, w in new:
            lines.append(f"  the prediction reads {words(w, strings, lens)}" if r is None else f"  {words(r, strings, lens)} reads {words(w, strings, lens)}")
        prev = g
    return "\n".join(lines)


def parse(text: str, n: int) -> tuple[str, list[str]] | None:
    m = re.search(r"EXPLANATION:\s*(.+?)(?=\nSTEP 1:|\Z)", text, re.S)
    notes = [re.search(rf"STEP {k + 1}:\s*(.+?)(?=\nSTEP {k + 2}:|\Z)", text, re.S) for k in range(n)]
    if not m or not all(notes):
        return None
    return m[1].strip(), [x[1].strip().split("\n")[0] for x in notes]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("search", type=Path)
    ap.add_argument("--split", choices=("train", "heldout"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-4B")
    ap.add_argument("--max-tokens", type=int, default=12288, help="the oracle's output budget the answers are cut to")
    ap.add_argument("--write-tokens", type=int, default=1536)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    import train

    tok = AutoTokenizer.from_pretrained(a.base)
    nat = native.Native()
    count = lambda t: tok.encode(mech.PART.sub("§", t), add_special_tokens=False)  # noqa: E731  a part token is one oracle token
    jobs = []
    for p in native.tasks(a.split):
        src_path = a.search / f"{p.stem}.py"
        if not src_path.exists() or (a.out / src_path.name).exists():
            continue
        task = json.loads(p.read_text())
        answer = "```python\n" + src_path.read_text() + "```"
        cut = train.split_answer(tok.decode(train.fit(tok, answer, a.max_tokens, count)) if len(count(answer)) > a.max_tokens else answer)[0]
        ir = mech.trace_inline(cut, "vpd4l", task)
        if not ir["valid"] or not ir["graph"]["steps"]:
            continue
        steps = [native.prefix(ir, k)[0] for k in range(1, ir["graph"]["steps"] + 1)]
        strings = mech.behavior_tokens(task, "vpd4l")["sequences"][0][1]
        prompt = task["prompts"][0]
        t = prompt["target_positions"][0]
        text = "\n".join(f"{i}: {s!r}" for i, s in enumerate(strings))
        ask = ASK.format(t=t, top=", ".join(repr(x) for x, _ in prompt["model_top"][0]), text=text, graph=render(steps, strings, nat.lens(steps[-1].nodes)))
        jobs.append((p.stem, steps, tok.apply_chat_template([{"role": "user", "content": ask}], add_generation_prompt=True, enable_thinking=False, tokenize=False)))
    print(f"{len(jobs)} answers to describe", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    try:
        from vllm import LLM, SamplingParams

        llm = LLM(model=a.base, dtype="bfloat16", max_model_len=32768, gpu_memory_utilization=0.85)
        outs = [o.outputs[0].text for o in llm.generate([j[2] for j in jobs], SamplingParams(temperature=0.7, max_tokens=a.write_tokens))]
    except ImportError:
        import torch
        from transformers import AutoModelForCausalLM

        dev = "cuda" if torch.cuda.is_available() else "mps"
        model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16).to(dev)
        outs = []
        for _, _, chat in jobs:
            ids = tok(chat, return_tensors="pt").input_ids.to(dev)
            outs.append(tok.decode(model.generate(ids, max_new_tokens=a.write_tokens, do_sample=True, temperature=0.7)[0, ids.shape[1]:], skip_special_tokens=True))
    done = 0
    for (stem, steps, _), text in zip(jobs, outs):
        got = parse(text, len(steps))
        if got is None:
            continue
        explanation, notes = got
        (a.out / f"{stem}.py").write_text(native.program(steps, notes=notes, explanation=explanation))
        done += 1
    print(f"described {done} of {len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
