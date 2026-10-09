"""The graph oracle's input (#2951, format v3): a short reference, example answers, then the behavior as text. The
oracle answers with a program (the `groups` dict, scored by the checker) and one line of English after it; split_answer
separates the two. No weights or vectors: the behavior's description, a few of its prompts with the model M's top next
tokens and probabilities (the behavior file's `model_top`), the behavior's variables, and the parts of M it may name.

  prompt.py BEHAVIOR.json [--prompts 4] [--shots 1]      prints the prompt
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import family  # noqa: E402
import mech  # noqa: E402

TEACHER = Path.home() / "mpd-data/graph_oracle/teacher_v3/manifest.jsonl"

REFERENCE = """\
Explain how the model computes the behavior below. Answer with one Python program in a ```python block, then one
line of plain English after the block. The program is a single dict, `groups`: named groups of the model's
subcomponents and which groups read which. The model's own weights do all of the computing.
- Each group: {{"subcomponents": [...], "reads": [...], "label": "<variable>"}}, or for the one group that writes
  the next-token prediction {{"subcomponents": [...], "reads": [...], "writes": "output"}}.
- subcomponents are part tokens in quotes, e.g. "<p:1.v.531>"; a subcomponent belongs to one group.
{parts}- reads: "input" (the prompt's tokens) and other groups ("name:query", "name:key" or "name:value" for an
  attention read). A group reads only what earlier layers write.
- label: the behavior variable the group carries (listed with the behavior). Every group except the output group
  carries one. A variable is tested on prompt pairs that change it: the group's output under the changed prompt is
  swapped in, and the model's prediction should change the same way. A variable no group carries costs its whole
  effect.
- Everything you do not name runs on the prompt's changed prompt (the prompt with the deciding information
  changed), so the groups you name must carry that information. Name those subcomponents and nothing more.
- The score in bits (lower is better) adds: how far the named groups alone are from the model, how much of the
  behavior survives when only the named groups run on the changed prompt, each variable's test, and the size
  (subcomponents, groups, reads, explanation)."""

PARTS = ("- <p:L.S.I> is subcomponent I (rank one) of VPD's decomposition of layer L's weight matrix S: q, k, v, o\n"
         "  (attention query, key, value, output) or fc, down (MLP input, output); per layer {sizes}; <p:L.S.rest> is\n"
         "  what that matrix holds beyond its subcomponents (it costs as many names as the matrix's rank).\n")


@lru_cache(None)
def views(model: str) -> tuple[str, str]:
    """(the model's sizes, the description of the parts the oracle may name)."""
    s = mech.shapes(model)
    groups = s["heads"] // s["kv_heads"]
    heads = f"{s['heads']} attention heads" + (f" ({s['kv_heads']} key-value groups of {groups})" if groups > 1 else "")
    sizes = (f"Model {model}: {s['layers']} layers (0..{s['layers'] - 1}); per layer {heads} and an MLP of {s['d_mlp']} "
             f"hidden units; vocabulary {s['vocab']} tokens.")
    codes = {site: code for code, site in mech.SITES.items()}
    return sizes, PARTS.format(sizes=", ".join(f"{codes[k]} {v}" for k, v in s["views"]["vpd"][0].items()))


def show(model: str, ids: list[int], t: int, top) -> str:
    context = mech.tokenizer(model).decode(ids[: t + 1])
    return f"{context!r} -> " + ", ".join(f"{tok!r} {p:.2f}" for tok, p in top)


def variables(behavior: dict) -> dict[str, str]:
    """The behavior's variables (what its changed prompts change; "tokens", the input, is none) and what each is, from
    its family algorithm when the family has one."""
    names = [v for v in behavior.get("varies") or {} if v != "tokens"]
    notes = family.Algorithm(behavior["family"]).notes if family.source(behavior.get("family", "")) else {}
    return {v: notes.get(v, "") for v in names}


def behavior_text(behavior: dict, prompts: int) -> str:
    model = behavior["model"]
    lines = [f"Behavior {behavior['id']}: {behavior['description']}",
             "Examples (the model's top next tokens with probabilities):"]
    for p in behavior["prompts"][:prompts]:
        for k, t in enumerate(p["target_positions"]):
            lines.append("  " + show(model, p["token_ids"], t, p["model_top"][k]))
            cf = p.get("counterfactual")
            if cf and cf.get("model_top"):
                lines.append("    changed: " + show(model, cf["token_ids"], t, cf["model_top"][k]))
    named = variables(behavior)
    lines.append("Variables: " + ("; ".join(f"{v} ({note})" if note else v for v, note in named.items()) if named else
                                  "none (one group, writing the output)"))
    return "\n".join(lines)


def examples(behavior: dict, shots: int) -> list[tuple[str, str]]:
    """Up to `shots` teacher answers (behavior id, answer text) of train behaviors of other families (families sharing
    their first word count as one), most groups first, then fewest subcomponents."""
    if shots <= 0 or not TEACHER.exists():
        return []
    kin = (behavior.get("family") or "").split("_")[0]
    rows = [r for r in map(json.loads, open(TEACHER)) if r["family"].split("_")[0] != kin and Path(r["answer"]).exists()]
    rows.sort(key=lambda r: (-len(r["groups"]), r["parts"], r["behavior"]))
    return [(r["behavior"], Path(r["answer"]).read_text().strip()) for r in rows[:shots]]


def render(behavior: dict, prompts: int = 4, shots: int = 1) -> str:
    """The oracle's prompt for `behavior`."""
    sizes, parts = views(behavior["model"])
    out = [REFERENCE.format(parts=parts)]
    out += [f"Example answer (behavior {b}):\n{text}" for b, text in examples(behavior, shots)]
    out.append(f"{sizes}\n{behavior_text(behavior, prompts)}\n\nWrite the program, then the explanation.")
    return "\n\n".join(out)


def split_answer(answer: str) -> tuple[str, str]:
    """(program, explanation) of an oracle's answer: the last fenced block that parses as Python, and the text after
    that block (a leading "Explanation:" label dropped). Without such a block: (answer, "")."""
    parts = answer.split("```")
    for k in range(len(parts) - 2, 0, -2):  # fenced blocks are the odd parts; take the last one that parses
        block = parts[k].split("\n", 1)[1] if "\n" in parts[k] else ""
        try:
            ast.parse(block)
        except SyntaxError:
            continue
        explanation = "```".join(parts[k + 1:]).strip()
        if explanation.lower().startswith("explanation:"):
            explanation = explanation[len("explanation:"):].strip()
        return block, explanation
    return answer, ""


def program_of(answer: str) -> str:
    """The program in an oracle's answer (split_answer's first part)."""
    return split_answer(answer)[0]


def explanation_of(answer: str) -> str:
    """The English explanation after the program (split_answer's second part)."""
    return split_answer(answer)[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("behavior", type=Path)
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--shots", type=int, default=1)
    a = ap.parse_args()
    print(render(json.loads(a.behavior.read_text()), a.prompts, a.shots))


if __name__ == "__main__":
    main()
