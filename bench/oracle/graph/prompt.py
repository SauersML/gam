"""The graph oracle's input (#2951): the question and the text, nothing else. The oracle answers (mech.py) with Python
defining graph(tokens, targets): its docstring a plain-English explanation of how the model computes its prediction,
its value a list of steps, most important first, each adding subcomponents at positions and the outputs they read.

A question (a task, rl/train.py's behaviors) is shown as what graph() receives, the token strings and the position whose
next token is asked, with the model's most probable next tokens there.

  prompt.py TASK.json      prints the prompt
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

ASK = ("How does {model} compute its prediction of the next token after the target of this text? Answer with a Python "
       "function graph(tokens, targets) whose docstring explains it in plain English and which returns a list of steps, "
       "most important first. Each step is a dict giving, for subcomponents at positions, the subcomponents whose outputs "
       "they read (at the reader's own position; an attention output reads values at other positions, written "
       "{{position: subcomponents}}), and under \"out\" the subcomponents the prediction reads; a comment line above each "
       "step says what it adds. The first steps alone should explain as much as they can; later steps add detail.")


def tokens_of(model: str, ids: list[int]) -> list[str]:
    tk = mech.tokenizer(model)
    return [tk.decode([i]) for i in ids]


def line(model: str, ids: list[int], targets: list[int], top: list) -> str:
    """One sequence as on() sees it, and the model's most probable next tokens at its targets."""
    nexts = "; ".join(f"at {t}: " + ", ".join(f"{tok!r} {p:.2f}" for tok, p in row[:3]) for t, row in zip(targets, top))
    return f"tokens = {tokens_of(model, ids)!r}, targets = {targets!r} -> {nexts}"


def render(task: dict) -> str:
    """The oracle's prompt for a task: the ask, then each of its texts as on() receives it."""
    model = task["model"]
    return "\n".join([ASK.format(model=model)] + [line(model, p["token_ids"], p["target_positions"], p["model_top"]) for p in task["prompts"]])


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


def english_spans(source: str) -> list[tuple[int, int]]:
    """The character spans [start, end) of a program's English: the docstring of graph() and every comment."""
    import io
    import tokenize

    starts = [0]
    for line in source.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))

    def at(row, col):  # tokenize's (1-based row, column) -> a character offset
        return starts[row - 1] + col

    spans = []
    try:
        for t in tokenize.generate_tokens(io.StringIO(source).readline):
            if t.type == tokenize.COMMENT:
                spans.append((at(*t.start), at(*t.end)))
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.FunctionDef) and node.name == "graph" and node.body and isinstance(node.body[0], ast.Expr) \
                    and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                d = node.body[0].value
                spans.append((at(d.lineno, d.col_offset), at(d.end_lineno, d.end_col_offset)))
    except (SyntaxError, tokenize.TokenError, IndentationError):
        return []
    return sorted(spans)


def program_of(answer: str) -> str:
    """The program in an oracle's answer (split_answer's first part)."""
    return split_answer(answer)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("task", type=Path)
    a = ap.parse_args()
    print(render(json.loads(a.task.read_text())))


if __name__ == "__main__":
    main()
