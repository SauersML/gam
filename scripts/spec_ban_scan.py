#!/usr/bin/env python3
"""SPEC ban scan: shape-level scan for SPEC.md violations the build.rs
hygiene bans do not see. The bar is ZERO -- any hit fails.

`build.rs` bans a fixed list of tokens (`#[allow]`, `todo!`, wall clocks, ...).
It has no rule for the SPEC bans that live in the SHAPE of production code:

  grid         a literal float array iterated as candidates, or handed out as a
               `&'static [f64]` candidate list          SPEC "Grid search is never allowed"
  box          a symmetric `.clamp(-X, X)` on a step, a named step-cap
               constant, or a `smooth_bound*` box on a parameter
                                                        SPEC "Hand-supplied search boxes or bounds"
  jitter       ridge escalation, jitter/nugget constants, a diagonal add of a
               jitter-named value or of an exponent literal (`+= 1e-10 * s`)
                                                        SPEC "do not paper over solver issues"
  unconverged  `Ok(Struct { .., converged: false, .. })` / `Some(..)`: a result
               that reports itself unconverged handed on as a success
                                                        SPEC "A fit object must only ever come
                                                        from a converged optimization"
  magic        a bare negative-exponent literal used as a comparison threshold
               or as a `.max()` / `.min()` floor        SPEC "magic constants ... avoided"
  fd           `f(a + h) - f(a - h)` / `(f(a + h) - f(a)) / h`
                                                        SPEC "Finite differences is never used
                                                        outside of tests"
  gcv          a `gcv` / `ubre` identifier segment      SPEC "REML (or LAML) always used, never GCV"
  python-math  numpy / scipy / torch / jax linear algebra or scipy.optimize in
               production `gamfit/`                     SPEC "Python should be a thin wrapper"
  roundoff     the unit roundoff `EPSILON/2` or Wilkinson's `γ_k = k·ε/(1 − k·ε)`
               written out by hand instead of read from their one owner,
               `gam_math::roundoff` (re-exported by `gam_linalg::roundoff`)
                                                        SPEC "Hard-coded knobs and magic constants
                                                        ... should be avoided"

Test code is out of scope, with the same notion of "test" as build.rs'
`compute_test_mask` (whole-file test directories, `#![cfg(test)]`, and
brace-tracked `#[test]` / `#[cfg(..test..)]` regions), tightened in three
ways: `cfg(not(test))` is production, `#[cfg(test)] mod x;` gates only that
declaration, and a file whose `mod` declaration is test-gated (or whose parent
module file is test code) is test code. The two test-support crates, which
hold the finite-difference oracles, are test code.

THERE IS NO LEDGER. Every hit is a violation and fails the scan, printed as
`file:line`, rule and token. A violation is removed by fixing the code: moving
it to another production file does not clear it, and there is no allowance file
to record it in. Spelling the literal differently, or lifting it into a named
constant, only hides it from this scan -- the SPEC rule is about where the
quantity comes from, not about how it is written.

WHY A GREEN HERE IS NOT A NON-RUN. While a ledger existed, its lines doubled as
known-offending inputs: a scan that silently measured nothing reported them
clean and failed loudly. A zero bar has no such inputs of its own, and zero hits
over the real tree is byte for byte what a scan over the wrong root prints. So
every run first plants one violation per rule and requires the detector to
report each one, and to report none of them from test-gated or commented code
(`positive_control`); a detector that has stopped detecting exits 2. Every run
also refuses a root with no production Rust source at all, and prints the number
of files it read.

Exit codes: 0 no hit, 1 one or more violations, 2 the scan could not run (the
planted controls were not detected, or the root holds no production source).

Usage:
  scripts/spec_ban_scan.py [--root DIR]
  scripts/spec_ban_scan.py --positive-control
"""

from __future__ import annotations

import argparse
import bisect
import io
import os
import re
import sys
import tokenize
from collections import Counter
from pathlib import Path

TEST_SUPPORT_CRATES = ("crates/gam-test-support/", "crates/gam-linalg-test-support/")
RULES = ("grid", "box", "jitter", "unconverged", "magic", "fd", "gcv", "python-math", "roundoff")


class CannotMeasure(Exception):
    """The check could not run; distinct from a finding (exit 2, not 1)."""


# ---------------------------------------------------------------------------
# Rust source normalisation
# ---------------------------------------------------------------------------

_TOKEN = re.compile(
    r"//[^\n]*"
    r"|/\*"
    r"|(?<![\w])b?r(?P<hashes>#*)\""
    r"|(?<![\w])b?\"(?:[^\"\\]|\\.)*\""
    r"|'(?:\\(?:x[0-9a-fA-F]{2}|u\{[0-9a-fA-F]+\}|.)|[^\\'\n])'",
    re.S,
)
_BLOCK = re.compile(r"/\*|\*/")


def _blank(segment: str) -> str:
    return re.sub(r"[^\n]", " ", segment)


def strip_rust(text: str) -> str:
    """Blank comments and literal contents; keep every newline and column.

    A string literal keeps its quotes (`""` then spaces) and a char literal
    becomes `' '`, so no rule can match text inside either, and a brace inside
    a literal cannot move the test-region tracker.
    """
    out = []
    pos = 0
    n = len(text)
    while True:
        m = _TOKEN.search(text, pos)
        if not m:
            out.append(text[pos:])
            break
        out.append(text[pos : m.start()])
        tok = m.group(0)
        if tok.startswith("//"):
            out.append(_blank(tok))
            pos = m.end()
        elif tok == "/*":
            depth, j = 1, m.end()
            while depth:
                b = _BLOCK.search(text, j)
                if not b:
                    j = n
                    break
                depth += 1 if b.group(0) == "/*" else -1
                j = b.end()
            out.append(_blank(text[m.start() : j]))
            pos = j
        elif m.group("hashes") is not None:
            close = '"' + m.group("hashes")
            j = text.find(close, m.end())
            j = n if j < 0 else j + len(close)
            out.append('""' + _blank(text[m.start() + 2 : j]))
            pos = j
        elif tok.startswith("'"):
            out.append("' '" + " " * (len(tok) - 3))
            pos = m.end()
        else:
            out.append('""' + _blank(tok[2:]))
            pos = m.end()
    return "".join(out)


_ATTR = re.compile(r"#(!?)\[\s*(test|cfg)\s*(\(|\])")
_MOD_DECL = re.compile(r"\bmod\s+(r#)?(\w+)\s*;")
_PATH_ATTR = re.compile(r'#\[\s*path\s*=\s*"([^"]+)"\s*\]')


def _cfg_has_test(args: str) -> bool:
    args = re.sub(r"\bnot\s*\([^()]*\)", " ", args)
    return re.search(r"(?<![\w])test(?![\w])", args) is not None


def _paren_args(line: str, start: int) -> str:
    depth = 1
    for j in range(start, len(line)):
        if line[j] == "(":
            depth += 1
        elif line[j] == ")":
            depth -= 1
            if depth == 0:
                return line[start:j]
    return line[start:]


def _gating(line: str) -> tuple[bool, bool]:
    """(inner, outer) test-gating attributes on one stripped line."""
    inner = outer = False
    for m in _ATTR.finditer(line):
        if m.group(2) == "test" and m.group(3) == "]" and not m.group(1):
            outer = True
        elif m.group(2) == "cfg" and m.group(3) == "(" and _cfg_has_test(_paren_args(line, m.end())):
            if m.group(1):
                inner = True
            else:
                outer = True
    return inner, outer


def test_mask(stripped_lines: list[str]) -> list[bool]:
    """True for every line inside test-only code of one file."""
    n = len(stripped_lines)
    if any(_gating(line)[0] for line in stripped_lines if "#!" in line):
        return [True] * n
    mask = [False] * n
    depth = 0
    pending = False
    stack: list[int] = []
    for idx, line in enumerate(stripped_lines):
        if "#" in line and _gating(line)[1]:
            pending = True
        mask[idx] = bool(stack) or pending
        for ch in line:
            if ch == "{":
                depth += 1
                if pending:
                    stack.append(depth - 1)
                    pending = False
            elif ch == "}":
                if stack and depth - 1 == stack[-1]:
                    stack.pop()
                depth -= 1
            elif ch == ";" and pending:
                # `#[cfg(test)] mod x;` / `#[cfg(test)] use ..;` gate one item.
                pending = False
    return mask


def _child_modules(root: Path, rel: str, raw: list[str], stripped: list[str], mask: list[bool]):
    here = Path(rel)
    own_dir = here.parent if here.name in ("mod.rs", "lib.rs", "main.rs") else here.parent / here.stem
    for idx, line in enumerate(stripped):
        for m in _MOD_DECL.finditer(line):
            path = None
            for back in range(idx, max(-1, idx - 6), -1):
                p = _PATH_ATTR.search(raw[back])
                if p:
                    path = here.parent / p.group(1)
                    break
                s = stripped[back].strip()
                if back != idx and s and not s.startswith("#"):
                    break
            cands = [path] if path else [own_dir / f"{m.group(2)}.rs", own_dir / m.group(2) / "mod.rs"]
            for c in cands:
                c = Path(os.path.normpath(c)).as_posix()
                if (root / c).is_file():
                    yield c, mask[idx]
                    break


def _is_test_path(rel: str) -> bool:
    return (
        rel.startswith(("tests/", "bench/", "benches/", "examples/"))
        or re.match(r"crates/[^/]+/(tests|benches|examples)/", rel) is not None
        or rel.startswith(TEST_SUPPORT_CRATES)
    )


def production_rust(root: Path) -> dict[str, tuple[list[str], str]]:
    """{rel: (raw_lines, production_text)} for every crate source file.

    production_text is the stripped text with every test-only line blanked to
    an empty line, so line numbers still index raw_lines.
    """
    bases = ["src"]
    crates = root / "crates"
    if crates.is_dir():
        bases += [f"crates/{c}/src" for c in sorted(os.listdir(crates))]
    files: dict[str, list] = {}
    for base in bases:
        for dirpath, dirs, names in os.walk(root / base):
            dirs.sort()
            for name in sorted(names):
                if not name.endswith(".rs"):
                    continue
                p = Path(dirpath) / name
                rel = p.relative_to(root).as_posix()
                text = p.read_text(encoding="utf-8")
                stripped = strip_rust(text).split("\n")
                files[rel] = [text.split("\n"), stripped, test_mask(stripped)]
    test_files = {r for r, v in files.items() if _is_test_path(r) or all(v[2])}
    edges = {r: list(_child_modules(root, r, v[0], v[1], v[2])) for r, v in files.items()}
    changed = True
    while changed:
        changed = False
        for parent, kids in edges.items():
            for kid, gated in kids:
                if kid in files and kid not in test_files and (gated or parent in test_files):
                    test_files.add(kid)
                    changed = True
    out = {}
    for rel, (raw, stripped, mask) in files.items():
        if rel in test_files:
            continue
        out[rel] = (raw, "\n".join("" if t else s for s, t in zip(stripped, mask)))
    return out


# ---------------------------------------------------------------------------
# Rust rules. Each takes production text and returns [(offset, token)].
# ---------------------------------------------------------------------------

def _squash(s: str) -> str:
    return re.sub(r"\s+", "", s)


_NUM = r"-?\s*\d[\d_]*(?:\.\d*)?(?:[eE][-+]?\d+)?(?:_?f64|_?f32)?"
_FLOAT = re.compile(r"\d\.|\d[eE]|f64|f32")
_ARR = rf"\[\s*{_NUM}(?:\s*,\s*{_NUM}){{2,}}\s*,?\s*\]"
_GRID_BIND = re.compile(
    rf"\b(?:let\s+(?:mut\s+)?|(?:pub(?:\([\w:]+\))?\s+)?(?:const|static)\s+)(\w+)\s*(?::[^=;]+)?=\s*&?({_ARR})\s*;",
    re.S,
)
_GRID_FOR = re.compile(rf"\bfor\s+[^;{{]*?\bin\s+&?({_ARR})", re.S)
_GRID_STATIC_FN = re.compile(r"\bfn\s+(\w+)\s*\([^)]*\)\s*->\s*&'static\s*\[f64\]\s*\{", re.S)
# Fixed mathematical tables (factorials, series coefficients, quadrature nodes
# and weights) are iterated too, but they are not candidate sets.
_TABLE_NAME = re.compile(r"factorial|coef|weight|node|abscissa|binom", re.I)


def _is_float_array(arr: str) -> bool:
    return _FLOAT.search(arr) is not None


def _brace_end(text: str, open_at: int) -> int:
    """Offset just past the brace group whose `{` is at open_at."""
    depth = 0
    for j in range(open_at, len(text)):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return j + 1
    return len(text)


def rule_grid(text: str):
    hits = []
    for m in _GRID_BIND.finditer(text):
        name, arr = m.group(1), m.group(2)
        if not _is_float_array(arr) or _TABLE_NAME.search(name):
            continue
        iterated = re.compile(
            rf"\bfor\s+[^;{{]*?\bin\s+(?:&\s*)?(?:self\.|Self::)?{re.escape(name)}\b"
        )
        if iterated.search(text):
            hits.append((m.start(), f"let {name}"))
    for m in _GRID_FOR.finditer(text):
        if _is_float_array(m.group(1)):
            hits.append((m.start(), f"for-in {_squash(m.group(1))}"))
    for m in _GRID_STATIC_FN.finditer(text):
        body_end = _brace_end(text, m.end() - 1)
        for a in re.finditer(_ARR, text[m.end() : body_end]):
            if _is_float_array(a.group(0)):
                hits.append((m.end() + a.start(), f"fn {m.group(1)} {_squash(a.group(0))}"))
    return hits


_CLAMP_SYM = re.compile(r"\.clamp\(\s*-\s*([\w.:]+)\s*,\s*([\w.:]+)\s*\)")
_STEP_CAP_CONST = re.compile(r"\b(?:const|static)\s+([A-Z0-9_]*(?:STEP_CAP|MAX_STEP)[A-Z0-9_]*)\s*:\s*f(?:64|32)\b")
_SMOOTH_BOUND = re.compile(
    r"\b(smooth_bound\w*)\s*\((?:[^;(){}]*,)?\s*(?:crate::|self::|super::)?(?:\w+::)*([A-Z][A-Z0-9_]+)\s*\)"
)


def _is_box_operand(x: str) -> bool:
    if re.fullmatch(r"1(?:\.0*)?(?:_?f64|_?f32)?", x) or "consts::" in x:
        # +-1 (cosines, correlations) and +-pi/2 (latitudes) are the domain of
        # the quantity, not a box chosen for it.
        return False
    if re.fullmatch(r"\d[\d_]*\.\d*(?:[eE][-+]?\d+)?(?:_?f64|_?f32)?|\d[\d_]*(?:_?f64|_?f32)", x):
        return True
    last = x.split("::")[-1].split(".")[-1]
    if re.fullmatch(r"[A-Z][A-Z0-9_]+", last):
        return True
    return re.search(r"step|cap|limit|nudge", last, re.I) is not None


def rule_box(text: str):
    hits = []
    lines_start = _line_starts(text)
    for m in _CLAMP_SYM.finditer(text):
        lo, hi = m.group(1), m.group(2)
        if lo != hi or not _is_box_operand(hi):
            continue
        line = _line_of(text, lines_start, m.start())
        if "format!" in line:
            continue
        hits.append((m.start(), f"clamp(+-{hi})"))
    for m in _STEP_CAP_CONST.finditer(text):
        hits.append((m.start(), f"const {m.group(1)}"))
    for m in _SMOOTH_BOUND.finditer(text):
        # The definition `fn smooth_bound_jet(value: f64, bound: f64)` has no
        # SCREAMING argument, so only call sites that supply the box match.
        hits.append((m.start(), f"{m.group(1)}({m.group(2)})"))
    return hits


_ESCALATE_CALL = re.compile(r"(?<!fn )\bescalate_ridge\s*\(")
_JITTER_CONST = re.compile(r"\b(?:const|static)\s+([A-Z0-9_]*(?:JITTER|NUGGET)[A-Z0-9_]*)\s*:")
_JITTER_DIAG = re.compile(r"\[\[\s*(\w+)\s*,\s*\1\s*\]\]\s*\+=\s*\*?\s*([\w.]*(?:jitter|nugget)\w*)", re.I)
# A diagonal add of a small exponent literal, `h[[i, i]] += 1e-10 * scale`: a
# ridge with a tuned magnitude whatever its variable is called.
_LITERAL_DIAG = re.compile(r"\[\[\s*(\w+)\s*,\s*\1\s*\]\]\s*\+=\s*\(?\s*(\d+(?:\.\d*)?[eE]-\d+)")


def rule_jitter(text: str):
    hits = []
    for m in _ESCALATE_CALL.finditer(text):
        before = text[max(0, m.start() - 3) : m.start()]
        if before.endswith("fn "):
            continue
        hits.append((m.start(), "escalate_ridge("))
    for m in _JITTER_CONST.finditer(text):
        hits.append((m.start(), f"const {m.group(1)}"))
    for m in _JITTER_DIAG.finditer(text):
        hits.append((m.start(), f"diag += {m.group(2)}"))
    for m in _LITERAL_DIAG.finditer(text):
        hits.append((m.start(), f"diag += {m.group(2)}"))
    return hits


_OK_STRUCT = re.compile(r"\b(Ok|Some)\s*\(\s*([A-Z]\w*(?:::[A-Z]\w*)*)\s*\{")
_CONVERGED_FALSE = re.compile(r"\bconverged\s*:\s*false\b")


def rule_unconverged(text: str):
    hits = []
    for m in _OK_STRUCT.finditer(text):
        if m.group(2) == "Self":
            # A constructor's fresh state (`fn new() -> Result<Self>`) has not
            # been optimized yet; it is not a result that failed to converge.
            continue
        end = _brace_end(text, m.end() - 1)
        c = _CONVERGED_FALSE.search(text, m.end(), end)
        if c:
            hits.append((c.start(), f"{m.group(1)}({m.group(2)}{{converged:false}})"))
    return hits


_NEG_EXP = r"-?\d[\d_]*(?:\.\d*)?[eE]-\d+(?:_?f64|_?f32)?"
_MAGIC_CMP = re.compile(
    rf"(?<![=\-<>!])(<=?|>=?)(?![<>=])\s*({_NEG_EXP})(?![\w.])"
    rf"|(?<![\w.])({_NEG_EXP})\s*(<=?|>=?)(?![<>=])"
)
_MAGIC_FLOOR = re.compile(rf"\.(max|min)\(\s*({_NEG_EXP})\s*\)")


def rule_magic(text: str):
    hits = []
    for m in _MAGIC_CMP.finditer(text):
        if m.group(1):
            hits.append((m.start(2), f"{m.group(1)} {m.group(2)}"))
        else:
            hits.append((m.start(3), f"{m.group(3)} {m.group(4)}"))
    for m in _MAGIC_FLOOR.finditer(text):
        hits.append((m.start(), f".{m.group(1)}({m.group(2)})"))
    return hits


_FD_CENTRAL = re.compile(
    r"\b([a-z_]\w*)\s*\(\s*([^;(){},]*?)\s*[+-]\s*([a-z_][\w.]*)\s*\)\s*-\s*\1\s*\(\s*\2\s*[+-]\s*\3\s*\)"
)
_FD_FORWARD = re.compile(
    r"\(\s*([a-z_]\w*)\s*\(\s*([^;(){},]*?)\s*\+\s*([a-z_][\w.]*)\s*\)\s*-\s*\1\s*\(\s*\2\s*\)\s*\)\s*/\s*\3\b"
)


def rule_fd(text: str):
    hits = []
    for m in _FD_CENTRAL.finditer(text):
        hits.append((m.start(), f"central {m.group(1)}(+-{m.group(3)})"))
    for m in _FD_FORWARD.finditer(text):
        hits.append((m.start(), f"forward {m.group(1)}(+{m.group(3)})"))
    return hits


def rule_gcv(text: str):
    hits = []
    for m in re.finditer(r"\b[A-Za-z_]\w*\b", text):
        ident = m.group(0)
        # snake/SCREAMING segments and CamelCase humps, so `mgcv_*` stays clean
        parts = [p.lower() for p in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z0-9]+", ident)]
        if "gcv" in parts or "ubre" in parts:
            hits.append((m.start(), ident))
    return hits


# The one file allowed to spell the unit roundoff out: it defines it.
ROUNDOFF_OWNER = "crates/gam-math/src/roundoff.rs"
# `0.5 * EPSILON` / `EPSILON / 2` / `EPSILON * 0.5` (not `0.5 * EPSILON.ln()`), and
# the growth factor's `.. EPSILON / (1.0 - ..` denominator in either the `ε` or the
# parenthesized form. Two local copies of `u` that differ by that factor of two
# are the drift this rule exists to stop.
_ROUNDOFF_COPY = re.compile(
    r"(?<![\w.])0\.5(?:_?f64)?\s*\*\s*f64::EPSILON(?![\w.])"
    r"|f64::EPSILON\s*(?:/\s*2(?:\.0*)?(?:_?f64)?|\*\s*0\.5(?:_?f64)?)(?![\w.])"
    r"|EPSILON\s*\)?\s*/\s*\(\s*1\.0\s*-"
)


def rule_roundoff(text: str):
    return [(m.start(), re.sub(r"\s+", " ", m.group(0))) for m in _ROUNDOFF_COPY.finditer(text)]


RUST_RULES = {
    "grid": rule_grid,
    "box": rule_box,
    "jitter": rule_jitter,
    "unconverged": rule_unconverged,
    "magic": rule_magic,
    "fd": rule_fd,
    "gcv": rule_gcv,
    "roundoff": rule_roundoff,
}


def _line_starts(text: str) -> list[int]:
    return [0] + [m.end() for m in re.finditer("\n", text)]


def _line_of(text: str, starts: list[int], offset: int) -> str:
    i = bisect.bisect_right(starts, offset) - 1
    end = starts[i + 1] - 1 if i + 1 < len(starts) else len(text)
    return text[starts[i] : end]


# ---------------------------------------------------------------------------
# Python rule
# ---------------------------------------------------------------------------

_PY_MATH = re.compile(
    r"\b(?:np|numpy|jnp|jax\.numpy|torch|scipy|sp)\.linalg\.\w+"
    r"|\bscipy\.optimize(?:\.\w+)?"
    r"|\bfrom\s+(?:numpy|scipy|jax\.numpy|torch)\.linalg\s+import\b"
    r"|\bfrom\s+scipy\.optimize\s+import\b"
    r"|\bfrom\s+scipy\s+import\s+(?:linalg|optimize)\b"
)


def python_code_only(src: str) -> str:
    """Python source with comments and string literals blanked, columns kept."""
    lines = src.split("\n")
    blank = [list(line) for line in lines]
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING) or (
                hasattr(tokenize, "FSTRING_MIDDLE") and tok.type == tokenize.FSTRING_MIDDLE
            ):
                (sr, sc), (er, ec) = tok.start, tok.end
                for r in range(sr - 1, er):
                    lo = sc if r == sr - 1 else 0
                    hi = ec if r == er - 1 else len(blank[r])
                    for c in range(lo, hi):
                        blank[r][c] = " "
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        raise CannotMeasure(f"python tokenizer failed: {exc}") from exc
    return "\n".join("".join(chars) for chars in blank)


def production_python(root: Path) -> dict[str, tuple[list[str], str]]:
    out = {}
    pkg = root / "gamfit"
    if not pkg.is_dir():
        return out
    for dirpath, dirs, names in os.walk(pkg):
        dirs[:] = sorted(d for d in dirs if d not in ("tests", "examples", "__pycache__"))
        for name in sorted(names):
            if not name.endswith(".py") or name.startswith("test_"):
                continue
            p = Path(dirpath) / name
            rel = p.relative_to(root).as_posix()
            src = p.read_text(encoding="utf-8")
            try:
                code = python_code_only(src)
            except CannotMeasure as exc:
                raise CannotMeasure(f"{rel}: {exc}") from exc
            out[rel] = (src.split("\n"), code)
    return out


def rule_python_math(text: str):
    return [(m.start(), re.sub(r"\s+", " ", m.group(0))) for m in _PY_MATH.finditer(text)]


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

def _hits_for(files, rules):
    """[(rule, path, token, line, source)] over {rel: (raw, text)}."""
    hits = []
    for rel, (raw, text) in files.items():
        starts = _line_starts(text)
        for rule, fn in rules.items():
            if rule == "roundoff" and rel == ROUNDOFF_OWNER:
                continue
            for off, token in fn(text):
                ln = bisect.bisect_right(starts, off)
                hits.append((rule, rel, token.replace("\t", " "), ln, raw[ln - 1].strip()))
    return hits


def scan(root: Path):
    """(hits, files_read) over the production Rust and Python sources under root."""
    rust = production_rust(root)
    if not rust:
        raise CannotMeasure(
            f"no production Rust source under {root}; a scan that read nothing certifies nothing"
        )
    python = production_python(root)
    hits = _hits_for(rust, RUST_RULES)
    hits += _hits_for(python, {"python-math": rule_python_math})
    hits.sort(key=lambda h: (h[0], h[1], h[3], h[2]))
    return hits, len(rust) + len(python)


def run_check(root: Path, out=sys.stdout, err=sys.stderr) -> int:
    hits, files_read = scan(root)
    for rule, path, token, ln, src in hits:
        print(f"error: {path}:{ln}: [{rule}] {token} -- SPEC violation: {src}", file=err)
    by_rule = Counter(h[0] for h in hits)
    summary = ", ".join(f"{r}={by_rule.get(r, 0)}" for r in RULES)
    print(f"spec_ban_scan: read {files_read} production file(s); "
          f"{len(hits)} hit(s) ({summary})", file=out)
    if hits:
        print(
            f"\n{len(hits)} SPEC violation(s). The bar is zero: there is no ledger and no "
            f"allowance file, and moving a hit to another production file does not clear it. "
            f"Derive the quantity where it stands, or refuse. Lifting a literal into a named "
            f"constant only hides it from this scan.",
            file=err,
        )
        return 1
    return 0


# ---------------------------------------------------------------------------
# Positive control
# ---------------------------------------------------------------------------

PLANTS = {
    "grid": "fn seeds() {\n    let log_lambda_grid = [-2.0, 0.0, 2.0];\n    for &r in &log_lambda_grid { try_seed(r); }\n}\n",
    "box": "fn step(raw: f64) -> f64 {\n    raw.clamp(-MAX_LOG_STEP, MAX_LOG_STEP)\n}\n",
    "jitter": "fn f(h: &mut Array2<f64>, jitter: f64) {\n    h[[i, i]] += jitter;\n}\n",
    "unconverged": "fn f() -> Result<Fit, E> {\n    Ok(Fit { beta, converged: false })\n}\n",
    "magic": "fn f(g: f64) -> bool {\n    g.abs() < 1e-8\n}\n",
    "fd": "fn f(x: f64, h: f64) -> f64 {\n    (loss(x + h) - loss(x - h)) / (2.0 * h)\n}\n",
    "gcv": "fn f() -> f64 {\n    gcv_score(1.0)\n}\n",
    "roundoff": "fn band(n: usize) -> f64 {\n    let u = 0.5 * f64::EPSILON;\n    n as f64 * u\n}\n",
}
PY_PLANT = "import numpy as np\n\ndef f(a):\n    return np.linalg.solve(a, a)\n"


def positive_control(out=sys.stdout, err=sys.stderr) -> int:
    failed = 0
    for rule, snippet in PLANTS.items():
        prod = strip_rust(snippet)
        found = [t for _, t in RUST_RULES[rule](prod)]
        gated = "#[cfg(test)]\nmod probe {\n" + snippet + "}\n"
        s = strip_rust(gated).split("\n")
        masked = "\n".join("" if t else line for line, t in zip(s, test_mask(s)))
        leaked = [t for _, t in RUST_RULES[rule](masked)]
        commented = "\n".join("// " + line for line in snippet.split("\n"))
        in_comment = [t for _, t in RUST_RULES[rule](strip_rust(commented))]
        ok = bool(found) and not leaked and not in_comment
        failed += not ok
        print(f"positive-control {rule}: production={found} test-gated={leaked} commented={in_comment} "
              f"-> {'ok' if ok else 'FAILED'}", file=out if ok else err)
    found = [t for _, t in rule_python_math(python_code_only(PY_PLANT))]
    in_comment = [t for _, t in rule_python_math(python_code_only("# np.linalg.solve(a, a)\nx = 'np.linalg.inv'\n"))]
    ok = bool(found) and not in_comment
    failed += not ok
    print(f"positive-control python-math: production={found} comment/string={in_comment} "
          f"-> {'ok' if ok else 'FAILED'}", file=out if ok else err)
    return 1 if failed else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--positive-control", action="store_true",
                    help="run only the planted-violation control and stop")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    try:
        # The control runs before every scan, not only under its own flag: with no
        # ledger there are no known-offending inputs in the tree, so this is the
        # only thing standing between a clean tree's green and a dead detector's.
        if positive_control() != 0:
            raise CannotMeasure(
                "the planted violations above were not all detected; this run says "
                "nothing about the tree"
            )
        if args.positive_control:
            return 0
        return run_check(root)
    except CannotMeasure as exc:
        print(f"spec_ban_scan: cannot measure: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
