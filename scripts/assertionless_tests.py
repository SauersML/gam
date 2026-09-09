#!/usr/bin/env python3
"""Refuse a `#[test]` function whose body reaches no assertion.

A test that reaches no assertion passes for every behaviour of the code it
calls: it is a green that cannot go red, and it is the exact dual of the XFAIL
pattern SPEC.md bans ("A failing test should always indicate problematic
behavior").

WHERE THIS LIVES, AND WHY IT IS NOT IN `build.rs`
`build.rs` carried `scan_for_useless_tests` and aborted the build on any
offender, until `f3ffc54e5` (2026-07-15) removed it with six sibling structural
gates on a stated principle: "Scanning the repository for implementation quality
is not the build script's job." That decision was also paying off a real cost --
issue #2110 ("build.rs ban aborts the whole workspace build (and the gamfit
wheel) at HEAD"): the root `gam` crate's build script is a transitive dependency
of `gam-pyffi`, so one assertion-less `#[test]` anywhere broke `maturin build`
and `gamfit` could not be imported. A gate on test quality must not be able to
take the Python wheel down with it. So the rule is kept and the home is changed:
a CI scan, beside `scripts/test_census.py`, that reads source and builds nothing.

WHAT COUNTS AS AN ASSERTION
The `assert!`/`debug_assert!` family, the panic-shaped macros (`panic!`,
`unreachable!`, `todo!`, `unimplemented!`), a propagating `?`, and a call or
macro whose name follows the repository's assertion conventions (`assert_*`,
`expect_*`, `require_*`, `ensure_*`). `#[should_panic]` tests are excluded: the
attribute IS the assertion.

`.expect(...)` and `.unwrap()` deliberately do NOT count, inherited from the
original gate. They are overwhelmingly fixture plumbing -- unwrapping a value the
test then fails to check -- and counting them would clear almost every offender
this gate exists to catch. Where an `.expect` genuinely IS the property under
test, say so: `assert!(x.validate().is_ok(), "...")` states it and is clearer.

DELEGATION, AND WHY IT CROSSES FILES
A test may delegate its checking to a helper. The original gate followed calls
into helper bodies defined in the SAME FILE, up to three hops. That is not
enough here, and the failure is not hypothetical: `sls_codegen_perf.rs` ends
`gate.faster(...); gate.finish();`, and `SpeedGate::finish` -- in
`crates/gam-math/src/paired_timing.rs`, a different crate -- is a real assertion
(it `assert!`s that cells were measured and that none lost, and its `Drop` panics
if `finish()` was never called). A file-local recognizer calls that test
assertion-less, which is wrong, and the cost of being wrong in that direction is
that an author is pushed to add a ceremonial `assert!` to satisfy a parser.

Resolution is therefore: free-function calls file-local (as the original), plus
METHOD calls resolved through the receiver's type. `let mut gate =
SpeedGate::open(...)` binds `gate` to `SpeedGate`, so `gate.finish()` resolves
into `impl SpeedGate` wherever in the workspace that block lives.

Resolving cross-file BY NAME ALONE was tried first and measured: it dropped the
tree-wide offender count from 21 to 3, because in a workspace this size a common
method name (`run`, `fit`, `build`, `check`) always matches some definition that
asserts. Among the tests it wrongly cleared was
`zz_probe_2644_prostate_binomial_logit`, which fits real data and then does
`match outcome { Ok(_) => println!("FIT OK"), Err(e) => println!("FIT ERR: {e}") }`
-- it swallows its own failure and verifies nothing. A rule that clears that test
is not a gate. Binding through the receiver's type keeps the SpeedGate case
correct without opening that hole.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

MAX_DELEGATION_HOPS = 3

ASSERTION_MACROS = (
    "assert!(",
    "assert_eq!(",
    "assert_ne!(",
    "debug_assert!(",
    "debug_assert_eq!(",
    "debug_assert_ne!(",
    "panic!(",
    "unreachable!(",
    "todo!(",
    "unimplemented!(",
)

ASSERTION_NAME_PREFIXES = ("assert_", "expect_", "require_", "ensure_")

SKIP_DIRS = {"target", ".git", "node_modules", "dist", "build", ".venv", "venv"}

# `build.rs` names the needles it scans for as part of its own contract, and this
# file quotes assertion syntax in its documentation and its tests.
EXEMPT_PATHS = {"build.rs", "scripts/assertionless_tests.py"}


def strip_source(content: str) -> list[str]:
    """Blank every comment and literal, preserving line structure and columns.

    Brace and paren counting downstream must not see a `{` inside a string or a
    `b'{'` byte literal, and the assertion recognizer must not see `assert!(`
    quoted inside a doc comment or inside a fixture that builds Rust source as
    data. Column positions are preserved so reported line numbers stay exact.

    Three states cross line boundaries, because all three constructs do in Rust:
    block comments, raw strings, and ordinary strings (which may span lines
    directly or via a trailing backslash continuation). Missing the last one is not
    theoretical -- it desynchronises quote parity for everything that follows,
    and a desynchronised parser reports a body span that ends early, which reads
    as an assertion-less test.
    """
    out: list[str] = []
    in_block_comment = False
    in_raw_string = False
    in_string = False
    raw_hashes = 0
    for line in content.split("\n"):
        buf: list[str] = []
        i = 0
        n = len(line)
        while i < n:
            if in_block_comment:
                if line.startswith("*/", i):
                    in_block_comment = False
                    buf.append("  ")
                    i += 2
                else:
                    buf.append(" ")
                    i += 1
                continue
            if in_raw_string:
                close = '"' + "#" * raw_hashes
                if line.startswith(close, i):
                    in_raw_string = False
                    buf.append(" " * len(close))
                    i += len(close)
                else:
                    buf.append(" ")
                    i += 1
                continue
            if in_string:
                if line[i] == "\\":
                    buf.append("  ")
                    i += 2
                    continue
                if line[i] == '"':
                    in_string = False
                    buf.append(" ")
                    i += 1
                    continue
                buf.append(" ")
                i += 1
                continue
            ch = line[i]
            if line.startswith("//", i):
                buf.append(" " * (n - i))
                break
            if line.startswith("/*", i):
                in_block_comment = True
                buf.append("  ")
                i += 2
                continue
            # A raw-string prefix is `r`, `br` or `rb` at a TOKEN BOUNDARY. The
            # boundary check is load-bearing: without it the final `r` of an
            # ordinary word followed by a closing quote -- `... repair",` -- is
            # read as the prefix `r"`, which opens a raw string that never
            # closes and blanks the rest of the file.
            at_token_start = i == 0 or not _is_ident_byte(line[i - 1])
            if at_token_start and ch in "rb":
                j = i
                if line.startswith("br", j) or line.startswith("rb", j):
                    j += 2
                elif ch == "r":
                    j += 1
                else:
                    j = -1
                if j > 0:
                    hashes = 0
                    while j + hashes < n and line[j + hashes] == "#":
                        hashes += 1
                    if j + hashes < n and line[j + hashes] == '"':
                        in_raw_string = True
                        raw_hashes = hashes
                        span = (j + hashes + 1) - i
                        buf.append(" " * span)
                        i = j + hashes + 1
                        continue
            if ch == '"' or (
                ch == "b" and at_token_start and i + 1 < n and line[i + 1] == '"'
            ):
                j = i + (2 if ch == "b" else 1)
                closed = False
                while j < n:
                    if line[j] == "\\":
                        j += 2
                        continue
                    if line[j] == '"':
                        j += 1
                        closed = True
                        break
                    j += 1
                if not closed:
                    in_string = True
                    j = n
                buf.append(" " * (j - i))
                i = j
                continue
            # Byte or char literal. A lone `'` is a lifetime; only blank a
            # literal that actually closes within the expected span.
            if ch == "'" or (ch == "b" and i + 1 < n and line[i + 1] == "'"):
                j = i + (2 if ch == "b" else 1)
                if j < n and line[j] == "\\":
                    j += 2
                else:
                    j += 1
                if j < n and line[j] == "'":
                    buf.append(" " * (j + 1 - i))
                    i = j + 1
                    continue
                buf.append(ch)
                i += 1
                continue
            buf.append(ch)
            i += 1
        out.append("".join(buf))
    return out


def _is_ident_byte(c: str) -> bool:
    """ASCII identifier byte. Deliberately not `str.isalnum`, which is true for
    non-ASCII letters and would split identifiers differently from rustc."""
    return c == "_" or ("a" <= c <= "z") or ("A" <= c <= "Z") or ("0" <= c <= "9")


def has_propagating_question(code: str) -> bool:
    """A `?` that propagates: followed by `;`, `,`, `.`, `)` or end of line.

    Distinguishes `foo()?;` from the `?` of a ternary-looking generic or a
    `Option<?>`-shaped fragment.
    """
    n = len(code)
    for i, ch in enumerate(code):
        if ch != "?":
            continue
        k = i + 1
        while k < n and code[k] == " ":
            k += 1
        if k == n or code[k] in ";,.)":
            return True
    return False


def contains_assertion_named_call(code: str) -> bool:
    """A call or macro whose NAME follows the assertion conventions."""
    n = len(code)
    i = 0
    while i < n:
        is_macro = code[i] == "!" and i + 1 < n and code[i + 1] in "([{"
        is_call = code[i] == "(" and not (i > 0 and code[i - 1] == "!")
        if not (is_macro or is_call):
            i += 1
            continue
        start = i
        while start > 0 and _is_ident_byte(code[start - 1]):
            start -= 1
        if start < i and code[start:i].startswith(ASSERTION_NAME_PREFIXES):
            return True
        i += 1
    return False


def line_is_assertion_shaped(code: str) -> bool:
    return (
        any(m in code for m in ASSERTION_MACROS)
        or has_propagating_question(code)
        or contains_assertion_named_call(code)
    )


def collect_called_names(code: str, out: list[str]) -> None:
    """Every identifier immediately preceding a `(`, turbofish-aware.

    `foo::<T>(x)` records `foo`; `bar.baz(x)` records `baz`, which is what makes
    method delegation resolvable at all.
    """
    n = len(code)
    i = 0
    while i < n:
        if code[i] != "(" or (i > 0 and code[i - 1] == "!"):
            i += 1
            continue
        id_end = i
        if i > 0 and code[i - 1] == ">":
            depth = 0
            p = i - 1
            open_at = None
            while p >= 0:
                if code[p] == ">":
                    depth += 1
                elif code[p] == "<":
                    depth -= 1
                    if depth == 0:
                        open_at = p
                        break
                p -= 1
            if open_at is not None and open_at >= 2 and code[open_at - 2 : open_at] == "::":
                id_end = open_at - 2
        start = id_end
        while start > 0 and _is_ident_byte(code[start - 1]):
            start -= 1
        if start < id_end:
            out.append(code[start:id_end])
        i += 1


def index_impls(stripped: list[str]) -> dict[str, list[tuple[int, int]]]:
    """Map each type name to the span of every `impl` block for it.

    `impl Foo`, `impl<T> Foo<T>` and `impl Trait for Foo` all key on `Foo` --
    for a trait impl the implementing TYPE is what follows `for`, which is the
    name a receiver will be bound to.
    """
    out: dict[str, list[tuple[int, int]]] = {}
    for i, line in enumerate(stripped):
        if _find_word(line, "impl") is None:
            continue
        head = line.split("{", 1)[0]
        for_at = _find_word(head, "for")
        if for_at is not None:
            target = head[for_at + 3 :]
        else:
            idx = _find_word(head, "impl")
            target = head[idx + 4 :]
            # Strip a generic parameter list belonging to `impl`, not the type.
            target = target.lstrip()
            if target.startswith("<"):
                depth = 0
                for k, ch in enumerate(target):
                    if ch == "<":
                        depth += 1
                    elif ch == ">":
                        depth -= 1
                        if depth == 0:
                            target = target[k + 1 :]
                            break
        target = target.strip()
        # Keep the last path segment, drop generics and trailing where-clauses.
        target = target.split("where")[0].strip()
        target = target.split("<")[0].strip()
        target = target.split("::")[-1].strip()
        name = "".join(c for c in target if _is_ident_byte(c))
        if not name:
            continue
        span = _body_span(stripped, i)
        if span is None:
            continue
        out.setdefault(name, []).append(span)
    return out


def receiver_types(stripped: list[str], open_at: int, close_at: int) -> dict[str, str]:
    """Bind local names to types via `let x = Type::assoc(...)`.

    A deliberately small piece of inference covering the dominant Rust idiom for
    a gate object. Anything it cannot bind stays unresolved, which errs toward
    reporting -- the safe direction for a binding step, since an unbound
    receiver simply falls through to the file-local rule.
    """
    bindings: dict[str, str] = {}
    pattern = re.compile(
        r"\blet\s+(?:mut\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*(?::[^=]*)?=\s*"
        r"([A-Za-z_][A-Za-z0-9_]*)(?:::<[^>]*>)?::[A-Za-z_][A-Za-z0-9_]*\s*\("
    )
    for k in range(open_at, min(close_at + 1, len(stripped))):
        for match in pattern.finditer(stripped[k]):
            bindings[match.group(1)] = match.group(2)
    return bindings


def collect_method_calls(code: str, out: list[tuple[str, str]]) -> None:
    """Every `receiver.method(` pair on a line."""
    for match in re.finditer(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", code
    ):
        out.append((match.group(1), match.group(2)))


def collect_associated_calls(code: str, out: list[tuple[str, str]]) -> None:
    """Every `Type::assoc(` pair on a line."""
    for match in re.finditer(
        r"\b([A-Za-z_][A-Za-z0-9_]*)::([A-Za-z_][A-Za-z0-9_]*)\s*\(", code
    ):
        out.append((match.group(1), match.group(2)))


def _find_word(s: str, word: str) -> int | None:
    n, wn = len(s), len(word)
    i = 0
    while i + wn <= n:
        if s[i : i + wn] == word:
            before_ok = i == 0 or not _is_ident_byte(s[i - 1])
            after = i + wn
            after_ok = after >= n or not _is_ident_byte(s[after])
            if before_ok and after_ok:
                return i
        i += 1
    return None


def fn_name_on_line(stripped: str) -> str | None:
    idx = _find_word(stripped, "fn")
    if idx is None:
        return None
    rest = stripped[idx + 2 :].lstrip()
    name: list[str] = []
    for c in rest:
        if _is_ident_byte(c):
            name.append(c)
        else:
            break
    return "".join(name) or None


def _body_span(stripped: list[str], sig_line: int) -> tuple[int, int] | None:
    """Brace-match the body that opens at or after `sig_line`."""
    depth = 0
    started = False
    open_at = sig_line
    for k in range(sig_line, len(stripped)):
        for ch in stripped[k]:
            if ch == "{":
                if not started:
                    open_at = k
                    started = True
                depth += 1
            elif ch == "}":
                depth -= 1
        if started and depth == 0:
            return open_at, k
        # A `;` terminated signature is a declaration, not a body.
        if not started and ";" in stripped[k]:
            return None
    return None


def index_functions(stripped: list[str]) -> dict[str, list[tuple[int, int]]]:
    """Map every `fn` name in this file to its body span(s).

    Indexes free functions and methods alike -- `find_word` only requires the
    `fn` keyword, so an `impl` block's methods are included, which is what makes
    a `gate.finish()` delegation resolvable when the gate is defined locally.
    """
    out: dict[str, list[tuple[int, int]]] = {}
    for i, s in enumerate(stripped):
        if _find_word(s, "fn") is None:
            continue
        name = fn_name_on_line(s)
        if name is None:
            continue
        span = _body_span(stripped, i)
        if span is None:
            continue
        out.setdefault(name, []).append(span)
    return out


def body_reaches_assertion(
    stripped: list[str],
    open_at: int,
    close_at: int,
    local: dict[str, list[tuple[int, int]]],
    impls: dict[str, list[tuple[str, tuple[int, int]]]] | None,
    sources: dict[str, list[str]] | None,
    hops: int,
    seen: set[tuple[str, int]],
    this_file: str,
) -> bool:
    """True when the span reaches an assertion directly or through delegation.

    Two delegation channels, and they are deliberately asymmetric:
      * free-function calls resolve FILE-LOCALLY, as the original gate did;
      * method and associated calls resolve through the RECEIVER'S TYPE into
        `impl` blocks anywhere in the workspace.
    Resolving any call by bare name across the workspace was measured and is
    far too permissive -- see the module docstring.
    """
    called: list[str] = []
    methods: list[tuple[str, str]] = []
    assoc: list[tuple[str, str]] = []
    for k in range(open_at, min(close_at + 1, len(stripped))):
        if line_is_assertion_shaped(stripped[k]):
            return True
        collect_called_names(stripped[k], called)
        collect_method_calls(stripped[k], methods)
        collect_associated_calls(stripped[k], assoc)
    if hops <= 0:
        return False

    for name in called:
        for span in local.get(name, []):
            key = (this_file, span[0])
            if key in seen:
                continue
            seen.add(key)
            if body_reaches_assertion(
                stripped, span[0], span[1], local, impls, sources, hops - 1, seen, this_file
            ):
                return True

    if impls is None or sources is None:
        return False

    bindings = receiver_types(stripped, open_at, close_at)
    # A method call resolves through its receiver's bound type; an associated
    # call names its type directly.
    targets: list[tuple[str, str]] = [
        (bindings[recv], method) for recv, method in methods if recv in bindings
    ]
    targets.extend(assoc)
    for type_name, method in targets:
        for other_file, (impl_open, impl_close) in impls.get(type_name, []):
            other = sources[other_file]
            other_local = index_functions(other)
            for span in other_local.get(method, []):
                if not (impl_open <= span[0] <= impl_close):
                    continue
                key = (other_file, span[0])
                if key in seen:
                    continue
                seen.add(key)
                if body_reaches_assertion(
                    other, span[0], span[1], other_local, impls, sources,
                    hops - 1, seen, other_file,
                ):
                    return True
    return False


def assertionless_tests(
    content: str,
    impls: dict[str, list[tuple[str, tuple[int, int]]]] | None = None,
    sources: dict[str, list[str]] | None = None,
    rel: str = "<memory>",
) -> list[tuple[int, str]]:
    """Every `#[test]` in `content` whose body reaches no assertion.

    Returns `(1-based line of the fn signature, the signature line)`.
    """
    if "#[test]" not in content:
        return []
    raw = content.split("\n")
    stripped = strip_source(content)
    local = index_functions(stripped)
    offenders: list[tuple[int, str]] = []
    n = len(stripped)
    i = 0
    while i < n:
        if "#[test]" not in stripped[i]:
            i += 1
            continue
        has_should_panic = "#[should_panic" in stripped[i]
        # `#[should_panic]` is commonly written BEFORE `#[test]`; walk back over
        # the contiguous attribute block so the ordering does not decide it.
        back = i
        while back > 0:
            back -= 1
            prev = stripped[back].strip()
            if not prev or prev.startswith("//"):
                continue
            if prev.startswith("#[") or prev.startswith("#!["):
                if "#[should_panic" in stripped[back]:
                    has_should_panic = True
                continue
            break
        # Walk forward to the signature, absorbing any further attributes.
        j = i + 1
        sig = None
        while j < n:
            t = stripped[j].strip()
            if not t or t.startswith("//"):
                j += 1
                continue
            if t.startswith("#[") or t.startswith("#!["):
                if "#[should_panic" in stripped[j]:
                    has_should_panic = True
                j += 1
                continue
            if _find_word(stripped[j], "fn") is not None:
                sig = j
            break
        if sig is None:
            i += 1
            continue
        if has_should_panic:
            i = sig + 1
            continue
        span = _body_span(stripped, sig)
        if span is None:
            i = sig + 1
            continue
        open_at, close_at = span
        reached = body_reaches_assertion(
            stripped, open_at, close_at, local, impls, sources,
            MAX_DELEGATION_HOPS, set(), rel,
        )
        if not reached:
            offenders.append((sig + 1, raw[sig].strip()))
        i = close_at + 1
    return offenders


def collect_sources(root: Path) -> dict[str, list[str]]:
    sources: dict[str, list[str]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for fname in filenames:
            if not fname.endswith(".rs"):
                continue
            path = Path(dirpath) / fname
            rel = str(path.relative_to(root)).replace(os.sep, "/")
            if rel in EXEMPT_PATHS:
                continue
            try:
                sources[rel] = strip_source(path.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
    return sources


def build_impl_index(
    sources: dict[str, list[str]]
) -> dict[str, list[tuple[str, tuple[int, int]]]]:
    """Type name -> every `impl` block for it, across the workspace."""
    index: dict[str, list[tuple[str, tuple[int, int]]]] = {}
    for rel, stripped in sources.items():
        for name, spans in index_impls(stripped).items():
            for span in spans:
                index.setdefault(name, []).append((rel, span))
    return index


def scan_tree(root: Path) -> list[tuple[str, int, str]]:
    sources = collect_sources(root)
    impls = build_impl_index(sources)
    offenders: list[tuple[str, int, str]] = []
    for rel in sorted(sources):
        path = root / rel
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line, sig in assertionless_tests(content, impls, sources, rel):
            offenders.append((rel, line, sig))
    return offenders


# The historical incident this gate exists for. At `a1d610d21`,
# `tests_2101_birth_locus_probe.rs` carried exactly two assertion-less `#[test]`
# probes, and issue #2110 records the then-live build.rs gate reporting exactly
# `2`. Re-measuring it on every run is the only way to tell a gate that finds
# nothing from a gate that has stopped detecting anything -- they print the same
# thing.
POSITIVE_CONTROL_SHA = "a1d610d212a2ba5f2b4c2097260c814bdc45d579"
POSITIVE_CONTROL_PATH = "crates/gam-sae/src/manifold/tests_2101_birth_locus_probe.rs"
POSITIVE_CONTROL_EXPECTED = 2


def positive_control(repo: Path) -> int:
    try:
        blob = subprocess.run(
            ["git", "show", f"{POSITIVE_CONTROL_SHA}:{POSITIVE_CONTROL_PATH}"],
            cwd=repo, capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(
            f"positive control UNAVAILABLE: cannot read {POSITIVE_CONTROL_PATH} at "
            f"{POSITIVE_CONTROL_SHA[:9]} ({exc}). A gate whose control cannot run is "
            f"not a gate that passed.",
            file=sys.stderr,
        )
        return 2
    found = assertionless_tests(blob, rel=POSITIVE_CONTROL_PATH)
    if len(found) != POSITIVE_CONTROL_EXPECTED:
        print(
            f"positive control FAILED at {POSITIVE_CONTROL_SHA[:9]}: detector reported "
            f"{len(found)} assertion-less #[test]s in {POSITIVE_CONTROL_PATH}, expected "
            f"{POSITIVE_CONTROL_EXPECTED} (issue #2110). The detector has stopped "
            f"detecting what it was built for.",
            file=sys.stderr,
        )
        for line, sig in found:
            print(f"  {POSITIVE_CONTROL_PATH}:{line}: {sig}", file=sys.stderr)
        return 1
    print(
        f"positive control OK at {POSITIVE_CONTROL_SHA[:9]}: "
        f"{len(found)} assertion-less #[test]s still detected in {POSITIVE_CONTROL_PATH}"
    )
    return 0


HOW_TO_FIX = (
    "Fix by asserting the property the test was written to check. Do not add "
    "`#[should_panic]` to dodge, and do not add a ceremonial `assert!(true)`. If a "
    "function is a pure exploratory dump with no invariant, it is not a test: move it "
    "to `examples/`, where `--all-targets` still compiles it but the suite stops "
    "reporting a meaningless green for it."
)


def identity(rel: str, signature: str) -> str:
    """`path::fn_name` -- the ledger key.

    Deliberately NOT the line number: a test keeps its identity when the file
    above it grows, and a ledger keyed on lines would go stale on every edit and
    train people to regenerate it rather than read it.
    """
    return f"{rel}::{fn_name_on_line(signature) or '<unnamed>'}"


def read_ledger(path: Path) -> list[str]:
    """The recorded offenders, one `path::fn_name` per line, `#` comments kept out.

    Required sorted and unique so the file has one representation: a ledger that
    can be reordered is a ledger whose diff does not say what changed.
    """
    entries = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            entries.append(line)
    if len(set(entries)) != len(entries):
        duplicates = sorted({e for e in entries if entries.count(e) > 1})
        raise ValueError(f"the ledger records the same test twice: {duplicates}")
    if entries != sorted(entries):
        raise ValueError("the ledger is not sorted; keep it in one canonical order")
    return entries


def check_ledger(repo: Path, ledger_path: Path) -> int:
    """A bidirectional ratchet over the KNOWN assertion-less tests.

    A gate demanding zero here would be red the day it landed -- there are 14 --
    and a gate that is red on arrival trains everyone to ignore it, which makes
    it the same non-instrument as the scan that was never wired at all. So it
    compares identities against a committed ledger and fails in two directions,
    exactly as `scripts/rustdoc_ratchet.sh` does for the rustdoc surface:

      * an offender NOT in the ledger  -> regression. A new `#[test]` that
        asserts nothing, or an existing one whose assertion was removed.
      * a ledger entry that is now clean -> the ledger is stale. Whoever fixed
        the test deletes its line in the fix's own commit, so the covered set
        only ever grows and the remaining list only ever shrinks.

    The second direction is also the POSITIVE CONTROL for the scan itself: every
    ledger entry is a known-offending input run through the same detector on
    every invocation. If the scan silently measured nothing -- wrong root, an
    unreadable tree, a lexer that desynchronised -- those entries would come back
    "clean" and this fails loudly, which is the difference between a green run
    and a run that never happened.
    """
    try:
        recorded = read_ledger(ledger_path)
    except (OSError, ValueError) as error:
        print(f"assertion-less ledger UNREADABLE: {error}", file=sys.stderr)
        return 2
    offenders = scan_tree(repo)
    where = {identity(rel, sig): (rel, line) for rel, line, sig in offenders}
    found = set(where)
    known = set(recorded)
    print(
        f"scanned the tree: {len(offenders)} assertion-less #[test]s found, "
        f"{len(recorded)} recorded in {ledger_path.name}"
    )
    if not recorded:
        print(
            "the ledger is empty, so this run had no known-offending input to prove the "
            "detector still detects; delete the ledger and demand zero instead.",
            file=sys.stderr,
        )
        return 2
    regressions = sorted(found - known)
    stale = sorted(known - found)
    for name in regressions:
        rel, line = where[name]
        print(f"  NEW  {rel}:{line}: {name}", file=sys.stderr)
    for name in stale:
        print(f"  FIXED (delete this line from {ledger_path.name})  {name}", file=sys.stderr)
    if regressions:
        print(
            f"\n{len(regressions)} #[test] function(s) reach no assertion and are not "
            f"recorded. A test that reaches no assertion passes for every behaviour of "
            f"the code it calls.\n{HOW_TO_FIX}",
            file=sys.stderr,
        )
    if stale:
        print(
            f"\n{len(stale)} recorded test(s) now assert something. Delete their lines "
            f"from {ledger_path.name} in the same commit as the fix -- the ledger only "
            f"ever shrinks, and leaving a fixed test on it turns the ratchet back into a "
            f"rubber stamp.",
            file=sys.stderr,
        )
    return 1 if (regressions or stale) else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="repository root to scan")
    parser.add_argument(
        "--positive-control",
        action="store_true",
        help="re-measure the #2110 incident and verify the detector still reports it",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        help="ratchet the tree against a committed ledger of known offenders",
    )
    args = parser.parse_args(argv)
    repo = Path(args.root).resolve()

    control = positive_control(repo)
    if control != 0:
        return control
    if args.positive_control:
        return 0

    if args.ledger is not None:
        return check_ledger(repo, args.ledger)

    offenders = scan_tree(repo)
    if not offenders:
        print("no assertion-less #[test] functions")
        return 0
    print(
        f"{len(offenders)} #[test] function(s) reach no assertion. A test that reaches no "
        f"assertion passes for every behaviour of the code it calls.",
        file=sys.stderr,
    )
    for rel, line, sig in offenders:
        print(f"  {rel}:{line}: {sig}", file=sys.stderr)
    print(f"\n{HOW_TO_FIX}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
