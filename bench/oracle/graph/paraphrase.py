"""Paraphrase a program's English (#2951 graph oracle) for the hidden-code check: a frozen local model
rewrites every comment and docstring with the same facts (addresses, numbers, claims) in other words;
the code is untouched, so the paraphrased program traces to the same IR. The reader scored on the
paraphrased English beside the original tells whether its predictions rest on what the English says
or on surface text it shares with the code.

  paraphrase.py PROGRAM.py [...] --model vpd4l --writer Qwen/Qwen3-8B --backend vllm --out DIR
"""

from __future__ import annotations

import argparse
import ast
import inspect
import io
import sys
import tokenize
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mech  # noqa: E402

ASK = """\
Rewrite the text below, from a comment in a program, in different words. Keep every fact: every
name, every address such as L[2].head[4] or PD.vpd[1].q_proj[316], every number and every claim.
Add nothing. Answer with the rewritten text only.

<<<
{text}
>>>"""


def spans(source: str) -> list[tuple[str, int, int, str]]:
    """The English spans of `source` in order: (kind "comment" | "docstring", start, end, text), with
    start/end character offsets into `source` and text as mech.english reads it."""
    lines = source.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))

    def offset(row: int, col: int, byte: bool) -> int:
        if byte:  # ast columns count UTF-8 bytes
            col = len(lines[row - 1].encode()[:col].decode("utf-8", errors="ignore"))
        return starts[row - 1] + col

    out = []
    for t in tokenize.generate_tokens(io.StringIO(source).readline):
        if t.type == tokenize.COMMENT:
            out.append(("comment", offset(*t.start, False), offset(*t.end, False), t.string[1:].strip()))
    for n in ast.walk(ast.parse(source)):
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
            out.append(("docstring", offset(n.lineno, n.col_offset, True), offset(n.end_lineno, n.end_col_offset, True),
                        inspect.cleandoc(n.value.value)))
    return sorted(out, key=lambda s: s[1])


def apply(source: str, replaced: list[tuple[str, int, int, str]]) -> str:
    """`source` with each span's text replaced (spans as `spans` returns them, new text last)."""
    for kind, start, end, text in sorted(replaced, key=lambda s: -s[1]):
        if kind == "comment":
            new = "# " + " ".join(text.split())
        else:
            indent = " " * (start - source.rfind("\n", 0, start) - 1)
            body = text.strip().replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
            new = '"""' + body.replace("\n", "\n" + indent) + ('\n' + indent if "\n" in body else "") + '"""'
        source = source[:start] + new + source[end:]
    return source


def paraphrase(sources: list[str], generate) -> list[str]:
    """Each source with its comments and docstrings rewritten by `generate` (user messages -> answers)."""
    found = [spans(s) for s in sources]
    asks = [ASK.format(text=sp[3]) for f in found for sp in f if sp[3]]
    answers = iter(generate(asks))
    out = []
    for source, f in zip(sources, found):
        new = [(k, a, b, next(answers).strip() if t else t) for k, a, b, t in f]
        out.append(apply(source, new))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("programs", nargs="+", type=Path)
    ap.add_argument("--model", required=True, choices=mech.MODELS)
    ap.add_argument("--writer", default="Qwen/Qwen3-8B")
    ap.add_argument("--backend", default="transformers", choices=["transformers", "vllm"])
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--out", type=Path, required=True, help="directory for the paraphrased programs")
    a = ap.parse_args()
    from textgen import Generator

    a.out.mkdir(parents=True, exist_ok=True)
    sources = [p.read_text() for p in a.programs]
    for path, source, new in zip(a.programs, sources, paraphrase(sources, Generator(a.writer, a.backend, a.max_tokens))):
        before, after = mech.trace(source, a.model), mech.trace(new, a.model)
        same = (before["nodes"], before["edges"]) == (after["nodes"], after["edges"])
        (a.out / path.name).write_text(new)
        print(path.name, "same IR:", same, after["error"] or "")


if __name__ == "__main__":
    main()
