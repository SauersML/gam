"""The graph oracle's input (#2951): the question and the text, and optionally VPD's ranked subcomponents. The oracle
answers (mech.py) with Python defining graph(tokens, targets): its docstring a plain-English explanation of how the
model computes its prediction, its value a list of steps, most important first, each adding subcomponents at positions
and the outputs they read.

A question (a task, rl/train.py's behaviors) is shown as what graph() receives, the token strings and the position whose
next token is asked, with the model's most probable next tokens there; with_vpd adds VPD's first k subcomponents at the
predicted position (native.py ranked), which the answer may start from, and where the prediction responds (native.py
responses): the oracle's input in tokens only.

  prompt.py TASK.json [--vpd-list K --positions P --root TEXTS]     prints the prompt
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


VPD_LIST = ("VPD's subcomponents at the predicted position, most important first (causal importance times how much each "
            "writes there): ")
RESPONSES = ("Where the prediction responds to the text: the predicted position, then the positions whose token, changed to "
             "another the model finds likely there, moves the prediction most, most first; at each, per weight matrix, the "
             "subcomponent writing most there beyond its average over the text: ")


def render(task: dict) -> str:
    """The oracle's prompt for a task: the ask, then each of its texts as on() receives it, then VPD's ranked
    subcomponents and where the prediction responds if the task carries them (with_vpd)."""
    model = task["model"]
    out = [ASK.format(model=model)] + [line(model, p["token_ids"], p["target_positions"], p["model_top"]) for p in task["prompts"]]
    if task.get("vpd_ranked"):
        runs = []
        for pos, part in task["vpd_ranked"]:
            if runs and runs[-1][0] == pos:
                runs[-1][1] += part
            else:
                runs.append([pos, part])
        out.append(VPD_LIST + ", ".join(f'({pos}, "{parts}")' for pos, parts in runs))
    if task.get("responses"):
        out.append(RESPONSES + ", ".join(f'({pos}, "{"".join(parts)}")' for pos, parts in task["responses"]))
    return "\n".join(out)


def with_vpd(task: dict, root: Path, k: int, positions: int = 0) -> dict:
    """The task carrying VPD's first k subcomponents at the predicted positions (native.py ranked, ROOT/vpd_ranked) and
    where the prediction responds at the predicted positions and `positions` others (native.py responses,
    ROOT/vpd_responses), which render lists; unchanged when both are 0."""
    out = dict(task)
    if k:
        out["vpd_ranked"] = json.loads((Path(root) / "vpd_ranked" / f"{task['id']}.json").read_text())[:k]
    if positions:
        rows = json.loads((Path(root) / "vpd_responses" / f"{task['id']}.json").read_text())
        out["responses"] = rows[:len(task["prompts"][0]["target_positions"]) + positions]
    return out


def feedback(s: dict) -> str:
    """The verifier's report on an answer as the oracle reads it before revising: why it could not run, or the KL in
    bits of the model's next-token distribution from the graph of its first k steps and that graph's description
    length, for each k, and each distinct reason something written is not part of the graph."""
    if not s.get("valid", True):
        return f"The verifier could not run your answer: {s.get('error')}"
    c = s.get("curve") or [[0.0, float("nan")]]
    lines = [f"no steps (the empty graph): {c[0][1]:.2f} bits"] + [f"first {k} steps: {kl:.2f} bits at a description length of {b:.0f} bits" for k, (b, kl) in enumerate(c[1:], 1)]
    report = ("The verifier ran the graph of your first k steps alone, over changed prompts of the text (one token replaced by a "
              "draw from the model's own prediction there), and measured the KL of the model's next-token distribution from the "
              "graph's:\n" + "\n".join(lines))
    dropped = s.get("dropped") or []
    if dropped:
        report += (f"\n{len(dropped)} things you wrote are not part of the graph (they still count in its description length):\n"
                   + "\n".join(f"- {why}" for why in dict.fromkeys(dropped)))
    return report


REVISE = ("Write an improved answer in the same format: as faithful as possible at every description length, the most "
          "important steps first, with its plain-English docstring and a comment line above each step.")


def complete_steps(source: str) -> str | None:
    """A program cut off before its end (an output limit), as far as its last complete step: the returned list closed
    after the last line "}," that leaves the program parsing. Its first steps are an answer like any prefix of one
    (the verifier scores every prefix); None when no step is complete."""
    lines = source.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == "},":
            cand = "\n".join(lines[:i + 1]) + "\n    ]\n"
            try:
                ast.parse(cand)
            except SyntaxError:
                continue
            return cand
    return None


def split_answer(answer: str) -> tuple[str, str]:
    """(program, the text after it) of an oracle's answer: the last fenced block that parses as Python; else the last
    block's complete steps when it was cut off (complete_steps). Without either: (answer, "")."""
    parts = answer.split("```")
    for k in range(len(parts) - 2, 0, -2):  # fenced blocks are the odd parts; take the last one that parses
        block = parts[k].split("\n", 1)[1] if "\n" in parts[k] else ""
        try:
            ast.parse(block)
        except SyntaxError:
            continue
        return block, "```".join(parts[k + 1:]).strip()
    last = answer.rfind("```python\n")
    if last >= 0:  # the last block, closed or cut off
        repaired = complete_steps(answer[last + len("```python\n"):].split("```")[0])
        if repaired is not None:
            return repaired, ""
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
    ap.add_argument("--vpd-list", type=int, default=0)
    ap.add_argument("--positions", type=int, default=0)
    ap.add_argument("--root", type=Path, default=Path.home() / "mpd-data/graph_oracle/texts")
    a = ap.parse_args()
    print(render(with_vpd(json.loads(a.task.read_text()), a.root, a.vpd_list, a.positions)))


if __name__ == "__main__":
    main()
