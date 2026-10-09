"""Family algorithms (#2951 graph oracle): each template family's variables as plain Python (algorithms/<file>.py,
algorithms/index.json: family -> file). A variable is a top-level function taking `tokens` (the prompt as the model's
token strings) and other variables by parameter name, returning one value per position; "answer" is the token the
model predicts after each position. The behaviors' variable annotations (behaviors/vary.py), the oracle's prompt
(each variable's meaning) and the tests use them; explanations never run them. Only train families have one.
"""

from __future__ import annotations

import ast
import io
import json
import tokenize
from pathlib import Path

import mech

HERE = Path(__file__).resolve().parent
INDEX = json.loads((HERE / "algorithms/index.json").read_text())
SAFE_BUILTINS = mech.SAFE_BUILTINS


def source(family: str) -> str | None:
    """The family's algorithm, None without one (every held-out family)."""
    return (HERE / "algorithms" / f"{INDEX[family]}.py").read_text() if family in INDEX else None


class Algorithm:
    """A family algorithm: its variables, what each reads and means, and their values on token lists."""

    def __init__(self, family: str):
        text = source(family)
        if text is None:
            raise KeyError(f"family {family} has no algorithm")
        tree = ast.parse(text)
        self.namespace = {"__builtins__": SAFE_BUILTINS}
        exec(compile(tree, f"<algorithm {family}>", "exec"), self.namespace)
        functions = [f for f in tree.body if isinstance(f, ast.FunctionDef)]
        params = {f.name: [a.arg for a in f.args.args] for f in functions}
        self.reads = {}
        todo = ["answer"]
        while todo:  # the variables: the answer and every function it reads, by parameter name
            v = todo.pop()
            if v not in self.reads:
                self.reads[v] = [p for p in params[v] if p != "tokens"]
                todo += self.reads[v]
        self.params = {v: params[v] for v in self.reads}
        self.names = list(self.reads)
        self.upstream = {v: self._upstream(v) for v in self.names}  # the variables v reads, directly or not
        self.notes = {}
        for f in functions:  # a variable's meaning: the first comment in its function
            segment = ast.get_source_segment(text, f) or ""
            self.notes[f.name] = next((t.string[1:].strip() for t in tokenize.generate_tokens(io.StringIO(segment).readline)
                                       if t.type == tokenize.COMMENT), "")

    def _upstream(self, v: str) -> set[str]:
        out, todo = set(), list(self.reads[v])
        while todo:
            u = todo.pop()
            if u not in out:
                out.add(u)
                todo += self.reads[u]
        return out

    def values(self, tokens: list[str], names) -> dict[str, list]:
        """The values of the variables `names` on `tokens`, {name: one value per position}."""
        done: dict[str, list] = {}

        def value(v):
            if v not in done:
                out = self.namespace[v](*[list(tokens) if p == "tokens" else list(value(p)) for p in self.params[v]])
                if not isinstance(out, (list, tuple)) or len(out) != len(tokens):
                    raise ValueError(f"variable {v} returned {out!r} for {len(tokens)} positions")
                done[v] = list(out)
            return done[v]

        return {v: value(v) for v in names}

    def at(self, tokens: list[str], t: int) -> dict:
        """Every variable's value at position t of the prompt cut after t (position t sees tokens 0..t only)."""
        values = self.values(tokens[: t + 1], self.names)
        return {v: values[v][t] for v in self.names}
