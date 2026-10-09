"""The graph oracle's input (#2951): the ask and the behavior's sequences, nothing else. The oracle answers with a gate
program (mech.gates): Python defining on(tokens, targets), the subcomponents acting at each position of a sequence.

A behavior is shown as what on() receives for a few of its prompts and their changed prompts (the token strings and
the positions whose next token is asked), with the model's most probable next tokens there. A text example
(activity()) asks instead which subcomponents act most strongly at each position of a stretch of text.

  prompt.py BEHAVIOR.json [--prompts 6]      prints the prompt
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mech  # noqa: E402

ASK = ("Which of {model}'s subcomponents, at which positions, make it predict what it does on these sequences rather "
       "than on their changed versions? Answer with a Python function on(tokens, targets) returning, for each position, "
       "the subcomponents acting there.")
ACTIVITY = "Which of {model}'s subcomponents act most strongly at each position of this sequence? Answer with on(tokens, targets)."


def tokens_of(model: str, ids: list[int]) -> list[str]:
    tk = mech.tokenizer(model)
    return [tk.decode([i]) for i in ids]


def line(model: str, ids: list[int], targets: list[int], top: list) -> str:
    """One sequence as on() sees it, and the model's most probable next tokens at its targets."""
    nexts = "; ".join(f"at {t}: " + ", ".join(f"{tok!r} {p:.2f}" for tok, p in row[:3]) for t, row in zip(targets, top))
    return f"tokens = {tokens_of(model, ids)!r}, targets = {targets!r} -> {nexts}"


def render(behavior: dict, prompts: int = 6) -> str:
    """The oracle's prompt for `behavior`: the ask, then its first `prompts` prompts and their changed prompts."""
    model = behavior["model"]
    out = [ASK.format(model=model)]
    for p in behavior["prompts"][:prompts]:
        out.append(line(model, p["token_ids"], p["target_positions"], p["model_top"]))
        cf = p.get("counterfactual")
        if cf and cf.get("model_top"):
            out.append("changed: " + line(model, cf["token_ids"], p["target_positions"], cf["model_top"]))
    return "\n".join(out)


def activity(model: str, ids: list[int]) -> str:
    """The prompt of a text example: which subcomponents act most strongly at each position of `ids`."""
    return ACTIVITY.format(model=model) + f"\ntokens = {tokens_of(model, ids)!r}, targets = []"


def split_answer(answer: str) -> tuple[str, str]:
    """(program, the text after it) of an oracle's answer: the last fenced block that parses as Python. Without such
    a block: (answer, "")."""
    parts = answer.split("```")
    for k in range(len(parts) - 2, 0, -2):  # fenced blocks are the odd parts; take the last one that parses
        block = parts[k].split("\n", 1)[1] if "\n" in parts[k] else ""
        try:
            ast.parse(block)
        except SyntaxError:
            continue
        return block, "```".join(parts[k + 1:]).strip()
    return answer, ""


def program_of(answer: str) -> str:
    """The program in an oracle's answer (split_answer's first part)."""
    return split_answer(answer)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("behavior", type=Path)
    ap.add_argument("--prompts", type=int, default=6)
    a = ap.parse_args()
    print(render(json.loads(a.behavior.read_text()), a.prompts))


if __name__ == "__main__":
    main()
