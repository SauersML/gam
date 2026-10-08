"""The graph oracle's input (#2951): a behavior file rendered as text, after a short `mech` reference
and example answers (a program, then its plain-English explanation). The oracle answers in the same form:
split_answer separates the program (scored by the checker) from the explanation (read by the reader). No
weights or vectors: the behavior's description, a few of its prompts with the model M's top next tokens
and probabilities (the behavior file's `model_top`, measured when the behavior was built), and the parts
of M it may name (part tokens) with their counts.

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

import mech  # noqa: E402

REFERENCE = """\
Explain how the model computes the behavior below. Answer with ONE Python program in a ```python block,
then the explanation, in plain English, after the block. The program is an algorithm whose variables are
aligned to parts of the model; it may import only `from mech import align, claim`.
- The algorithm: plain Python functions. Each top-level function is a variable named by its name: it
  takes `tokens` (the prompt as the model's token strings, such as " cat") and other variables (by
  parameter name) and returns a list with one value per position; its value at position t may use
  tokens 0..t only. The answer is the aligned variable no other variable reads: its value at t is the
  token the model predicts after position t (None: no prediction).
- align(variable, parts...): what these parts write into the residual stream holds the variable. A
  variable may span layers.
- claim(pattern, parts...): the attention of these query and key parts follows the variable `pattern`,
  whose value at t lists the positions 0..t attended (or maps positions to weights).
- Parts are written as part tokens:
{parts}  Every part you do not name is deleted. Name the parts that carry the information that decides
  the answer, and nothing more: write the smallest program that explains.
- Comments and docstrings are your working notes: they cost nothing and nobody else reads them.
- The explanation after the code says what each variable is, which parts hold it and how they connect,
  in a few plain sentences. A reader that never sees the code predicts the model under experiments from
  it alone.
- The score, in bits (lower is better), adds: the error of the program against the model under random
  experiments applied identically to both (prompt edits, weight edits, value swaps, edge cuts); whether
  deleting the named parts makes the behavior collapse; each alignment's error (the model with a
  variable's parts taken from another prompt against the algorithm's answer with that prompt's value);
  each claim's error; the reader's error from your explanation; and the program's size (parts,
  variables, lines, explanation)."""

PARTS = {
    "vpd": ("  <p:L.S.I> subcomponent I (rank one) of VPD's decomposition of layer L's weight matrix S: q, k, v, o\n"
            "  (attention query, key, value, output) or fc, down (MLP input, output); per layer {sizes};\n"
            "  <p:L.S.rest> what that matrix holds beyond its subcomponents.\n"),
    "library": ("  <p:L.attn.I>, <p:L.mlp.I> part I of our decomposition of layer L's attention or MLP (parts may\n"
                "  overlap); per layer about {sizes}.\n"),
    "transcoder": ("  <p:L.mlp.I> feature I of the transcoder replacing layer L's MLP ({sizes} per layer);\n"
                   "  <p:L.h.I> attention head I of layer L (its query, key, value and output weights).\n"),
    None: ("  <p:L.h.I> attention head I of layer L (its query, key, value and output weights); <p:L.m> layer L's\n"
           "  MLP.\n"),
}


@lru_cache(None)
def views(model: str, decomposition: str | None = None) -> tuple[str, str]:
    """(the model's sizes, the parts the oracle may name: the attached decomposition's, and native heads or
    MLPs only where it leaves a block uncovered)."""
    s = mech.shapes(model)
    decomposition = mech.DEFAULT_DECOMPOSITION.get(model) if decomposition is None else decomposition
    decomposition = None if decomposition == "native" else decomposition
    groups = s["heads"] // s["kv_heads"]
    heads = f"{s['heads']} attention heads" + (f" ({s['kv_heads']} key-value groups of {groups})" if groups > 1 else "")
    sizes = (f"Model {model}: {s['layers']} layers (0..{s['layers'] - 1}); per layer {heads} and an MLP of {s['d_mlp']} "
             f"hidden units; vocabulary {s['vocab']} tokens.")
    view = s["views"].get(decomposition) if decomposition else None
    counts = ("" if not view else
              ", ".join(f"{mech.CODES[k]} {v}" for k, v in view[0].items()) if decomposition == "vpd" else
              ", ".join(f"{k} {v}" for k, v in view["parts"][0].items()) if decomposition == "library" else str(view[0]))
    return sizes, PARTS[decomposition].format(sizes=counts)


def show(model: str, ids: list[int], t: int, top) -> str:
    context = mech.tokenizer(model).decode(ids[: t + 1])
    return f"{context!r} -> " + ", ".join(f"{tok!r} {p:.2f}" for tok, p in top)


def behavior_text(behavior: dict, prompts: int) -> str:
    model = behavior["model"]
    lines = [f"Behavior {behavior['id']}: {behavior['description']}",
             f"Examples (the model's top next tokens with probabilities):"]
    for p in behavior["prompts"][:prompts]:
        for k, t in enumerate(p["target_positions"]):
            lines.append("  " + show(model, p["token_ids"], t, p["model_top"][k]))
            cf = p.get("counterfactual")
            if cf and cf.get("model_top"):
                lines.append("    edited: " + show(model, cf["token_ids"], t, cf["model_top"][k]))
    return "\n".join(lines)


def examples(behavior: dict, shots: int) -> list[tuple[str, dict, str, str]]:
    """Up to `shots` example answers (name, index entry, source, explanation) for `behavior`: train-split
    examples of other families only (examples/index.json; families sharing their first word count as one),
    so a prompt never shows a program for its own behavior family or a held-out one; only algorithms with
    alignments that have an explanation (examples/<name>.explanation.txt); the target model's first, then by
    the index's "priority", then shortest."""
    index = json.loads((HERE / "examples/index.json").read_text())
    kin = (behavior.get("family") or "").split("_")[0]  # induction_random and induction_phrase are kin
    found = {}
    for n, e in index.items():
        note = HERE / "examples" / f"{n}.explanation.txt"
        if e["split"] != "train" or e["family"].split("_")[0] == kin or not note.exists():
            continue
        source = (HERE / "examples" / f"{n}.py").read_text()
        ir = mech.trace_inline(source, e["model"])
        if ir["valid"] and ir["answer"]:
            found[n] = (source, note.read_text().strip())
    names = sorted(found, key=lambda n: (index[n]["model"] != behavior["model"], index[n].get("priority", 9), len(found[n][0])))
    return [(n, index[n], *found[n]) for n in names[:shots]]


def render(behavior: dict, prompts: int = 4, shots: int = 1, decomposition: str | None = None) -> str:
    """The oracle's prompt for `behavior`, its parts those of `decomposition` (the model's default when None)."""
    model = behavior["model"]
    sizes, names = views(model, decomposition)
    parts = [REFERENCE.format(parts=names)]
    for name, entry, source, explanation in examples(behavior, shots):
        parts.append(f"Example answer ({entry['model']}):\n```python\n{source.strip()}\n```\n\n{explanation}")
    parts.append(f"{sizes}\n{behavior_text(behavior, prompts)}\n\nWrite the program, then the explanation.")
    return "\n\n".join(parts)


def split_answer(answer: str) -> tuple[str, str]:
    """(program, explanation) of an oracle's answer: the last fenced block that parses as Python, and the
    text after that block (a leading "Explanation:" label dropped). Without such a block: (answer, "")."""
    parts = answer.split("```")
    for k in range(len(parts) - 2, 0, -2):  # fenced blocks are the odd parts; take the last one that parses
        block = parts[k].split("\n", 1)[1] if "\n" in parts[k] else ""
        try:
            ast.parse(mech.quote_parts(block))
        except SyntaxError:
            continue
        explanation = "```".join(parts[k + 1 :]).strip()
        if explanation.lower().startswith("explanation:"):
            explanation = explanation[len("explanation:") :].strip()
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
