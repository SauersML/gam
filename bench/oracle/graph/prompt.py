"""The graph oracle's input (#2951, format v4): a short reference, example answers, then the behavior as text. The
oracle answers with a program (the `nodes`, `edges` and `labels`, scored by the checker) and English after it;
split_answer separates the two. No weights or vectors: the behavior's description, a few of its prompts with the model M's top
next tokens and probabilities (the behavior file's `model_top`), the behavior's variables, the parts of M it may name,
and the subcomponents VPD's causal importance says M needs on the behavior's prompts with what each does (atlas.py).

  prompt.py BEHAVIOR.json [--prompts 4] [--shots 1] [--table 48]      prints the prompt
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

import atlas  # noqa: E402
import family  # noqa: E402
import mech  # noqa: E402

TEACHER = Path.home() / "mpd-data/graph_oracle/teacher_v4/manifest.jsonl"

REFERENCE = """\
Explain how the model computes the behavior below. Answer with one Python program in a ```python block, then plain
English after the block. The program is a causal graph of the model's subcomponents: `nodes`, `edges` and, optionally,
`labels`. The model's own weights do all of the computing; the program says which subcomponents, where, and which
read which.
- nodes = {{name: {{"subcomponents": [...], "at": where}}}}. Subcomponents are part tokens in quotes, e.g. "<p:1.v.531>";
  a subcomponent belongs to one node. "at" says at which positions the node acts: "all", "targets" (the positions
  whose next token the behavior asks for), "last", or the name of a function you define that takes `tokens` (the
  sequence as the model's token strings) and returns a list of bools or of positions.
{parts}- edges = [(writer, reader) or (writer, reader, route)]: the writer a node or "input" (the tokens), the reader a node
  or "output" (the next-token prediction); route "query", "key" or "value" for an attention reader (default: all of
  its inputs). A writer must write before the reader reads (an earlier layer, or attention before the MLP of its
  layer); an attention reader at one position reads its writers' outputs at other positions.
- labels = {{node: variable}}, optional: the behavior variable a node carries (listed with the behavior). A variable is
  tested by swapping its node's output from a changed prompt that changes it; a variable no node carries costs its
  whole effect.
- Only what you name sees the prompt: every subcomponent and edge you leave out runs on the changed prompt. So the
  nodes and edges you name must carry everything that makes the model's answer the prompt's rather than the changed
  prompt's, and running only them on the changed prompt must turn the answer into the changed prompt's. Name what is
  needed and nothing more.
- The behavior is the model's choice between the prompt's answer and the changed prompt's answer. The score in bits
  (lower is better) adds: how far the graph's choice is from the model's on the prompts and the changed prompts, how
  much of the choice survives when only the graph is removed, each variable's test, and the size (subcomponents,
  nodes, edges, code, explanation)."""

PARTS = ("- <p:L.S.I> is subcomponent I (rank one) of VPD's decomposition of layer L's weight matrix S: q, k, v, o\n"
         "  (attention query, key, value, output) or fc, down (MLP input, output); per layer {sizes}; only o and down\n"
         "  subcomponents write the residual stream. <p:L.S.rest> is what that matrix holds beyond its subcomponents (it\n"
         "  costs as many names as the matrix's rank).\n")


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
    lines.append("Variables: " + ("; ".join(f"{v} ({note})" if note else v for v, note in named.items()) if named else "none"))
    return "\n".join(lines)


def examples(behavior: dict, shots: int) -> list[tuple[str, str]]:
    """Up to `shots` teacher answers (behavior id, answer text) of train behaviors of other families (families sharing
    their first word count as one), most groups first, then fewest subcomponents."""
    if shots <= 0 or not TEACHER.exists():
        return []
    kin = (behavior.get("family") or "").split("_")[0]
    rows = [r for r in map(json.loads, open(TEACHER)) if r["family"].split("_")[0] != kin and Path(r["answer"]).exists()]
    rows.sort(key=lambda r: (-len(r["nodes"]), r["parts"], r["behavior"]))
    return [(r["behavior"], Path(r["answer"]).read_text().strip()) for r in rows[:shots]]


def render(behavior: dict, prompts: int = 4, shots: int = 1, table: int = 48) -> str:
    """The oracle's prompt for `behavior`, with its first `table` subcomponents (atlas.text)."""
    sizes, parts = views(behavior["model"])
    out = [REFERENCE.format(parts=parts)]
    out += [f"Example answer (behavior {b}):\n{text}" for b, text in examples(behavior, shots)]
    listed = f"\n{atlas.text(behavior, table)}" if table else ""
    out.append(f"{sizes}\n{behavior_text(behavior, prompts)}{listed}\n\nWrite the program, then the explanation.")
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
    ap.add_argument("--table", type=int, default=48)
    a = ap.parse_args()
    print(render(json.loads(a.behavior.read_text()), a.prompts, a.shots, a.table))


if __name__ == "__main__":
    main()
