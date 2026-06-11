#!/usr/bin/env python3
"""Fast text-scan tier of the gam build.rs ban rules.

Invoked by scripts/precheck.sh as `precheck_scan.py <REPO_ROOT>`. Scans every
`.rs` file under `src/` and `crates/*/src/` (plus `tests/`, `benches/`,
`examples/` for the strict-everywhere rules) and reports each banned pattern as
`path:line: [rule] source`. Exit 0 when clean, 1 when any violation is found.

This is a deliberately conservative SUBSET of build.rs — it mirrors the
high-signal lexical rules that cause the recurring wheel-cycle failures
(debug_assert!, `let _`, #[allow]/#[expect], println! in src, env::var, bare
#[should_panic], #[ignore], cfg(feature), underscore fn params, the lexical
substring ban list). It reproduces build.rs's per-line treatment: a file-level
+ brace-tracked `#[cfg(test)]` test mask for the test-aware rules, and a
per-line strip of `//` comments and string/char literals to avoid flagging a
banned token that only appears inside a comment or a message string.

It does NOT attempt build.rs's whole-crate analyses (src item used only by
tests, zero-consumer pub(crate), dodge-named/no-op-sentinel fns). A clean run is
a fast smoke test, not a guarantee — build.rs and CI remain authoritative. Every
pattern it flags IS a real build.rs failure, and it errs toward UNDER-reporting
(never inventing a rule build.rs lacks).
"""

from __future__ import annotations

import os
import re
import sys

# (needle, label, test_aware) — mirrors build.rs `banned_substrings()`.
# test_aware=True ⇒ exempt inside test scope (tests/benches/examples files and
# `#[cfg(test)]` regions), matching build.rs's compute_test_mask.
BANNED_SUBSTRINGS = [
    ("dbg!(", "dbg!", True),
    ("todo!(", "todo!", True),
    ("unimplemented!(", "unimplemented!", True),
    ("unreachable!(", "unreachable!", True),
    ("assert!(true)", "assert!(true)", False),
    ("assert!(false)", "assert!(false)", False),
    ("debug_assert!(", "debug_assert!", False),
    ("debug_assert_eq!(", "debug_assert_eq!", False),
    ("debug_assert_ne!(", "debug_assert_ne!", False),
    ("hint::black_box(", "hint::black_box", True),
    ("std::hint::black_box(", "std::hint::black_box", True),
    ("core::hint::black_box(", "core::hint::black_box", True),
    ("cfg!(debug_assertions)", "cfg!(debug_assertions)", True),
    ("cfg!(test)", "cfg!(test)", True),
    ("Arc::strong_count(", "Arc::strong_count", True),
    ("Arc::weak_count(", "Arc::weak_count", True),
    ("Rc::strong_count(", "Rc::strong_count", True),
    ("Rc::weak_count(", "Rc::weak_count", True),
    ('file!().ends_with(".rs")', 'file!().ends_with(".rs")', False),
    ("std::process::exit(", "std::process::exit", False),
    ("process::exit(", "process::exit", False),
    ("std::process::abort(", "std::process::abort", False),
    ("process::abort(", "process::abort", False),
    ("mem::forget(", "mem::forget", True),
    ("Box::leak(", "Box::leak", True),
    ("env::var(", "env::var", False),
    ("env::var_os(", "env::var_os", False),
    ("env::vars(", "env::vars", False),
    ("env::vars_os(", "env::vars_os", False),
    ("if let Ok(_)", "if let Ok(_)", False),
    ("if let Some(_)", "if let Some(_)", False),
    ("if let Err(_)", "if let Err(_)", False),
    ("== true", "== true", False),
    ("== false", "== false", False),
    ("!= true", "!= true", False),
    ("!= false", "!= false", False),
    ("thread::sleep(", "thread::sleep", True),
    ("std::thread::sleep(", "std::thread::sleep", True),
    ("spin_loop()", "spin_loop", False),
    ("thread::yield_now(", "thread::yield_now", False),
    ("std::thread::yield_now(", "std::thread::yield_now", False),
]


def strip_strings_and_comments(line: str) -> str:
    """Blank out `//` line comments and the contents of "..." / '...' literals.

    Coarse, single-line, escape-aware — the same documented limitation build.rs
    carries (block comments and raw strings are not handled). The point is to
    avoid flagging a banned token that only appears inside a message string or a
    trailing comment; it errs toward stripping (under-reporting), never the
    reverse.
    """
    out = []
    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if c == "/" and i + 1 < n and line[i + 1] == "/":
            break  # rest of line is a comment
        if c == '"' or c == "'":
            quote = c
            out.append(" ")
            i += 1
            while i < n:
                if line[i] == "\\":
                    i += 2
                    continue
                if line[i] == quote:
                    i += 1
                    break
                i += 1
            out.append(" ")
            continue
        out.append(c)
        i += 1
    return "".join(out)


# A `#[cfg(test)]` / `#[cfg(all(test, ...))]` / `#[cfg(any(test, ...))]` gate.
CFG_TEST_RE = re.compile(r"#!?\[\s*cfg\s*\(\s*(all|any)?\s*\(?\s*test\b")

FILE_TEST_SCOPE_RE = re.compile(
    r"^(tests/|bench/|benches/|examples/|crates/[^/]+/(tests|benches|bench|examples)/)"
)


def file_is_test_scope(rel: str) -> bool:
    return bool(FILE_TEST_SCOPE_RE.match(rel))


def compute_test_mask(lines, rel):
    """Per-line 'is this line in test scope?' bitmap (mirrors build.rs).

    File-level: any tests/benches/examples file is wholly test scope.
    In-file: brace-tracked `#[cfg(test)]` regions — when the gate's `{` opens we
    mark lines until the matching `}` closes.
    """
    n = len(lines)
    if file_is_test_scope(rel):
        return [True] * n

    mask = [False] * n
    depth = 0
    gate_stack = []  # brace depths at which an open cfg(test) gate started
    pending_gate = False  # saw the attribute, waiting for the opening `{`

    for idx, raw in enumerate(lines):
        code = strip_strings_and_comments(raw)
        start_depth = depth
        in_gate_at_line_start = len(gate_stack) > 0
        if in_gate_at_line_start:
            mask[idx] = True

        if CFG_TEST_RE.search(code):
            pending_gate = True

        for ch in code:
            if ch == "{":
                depth += 1
                if pending_gate:
                    gate_stack.append(start_depth)
                    pending_gate = False
                    # lines from here in the gate are test scope
                    mask[idx] = True
            elif ch == "}":
                depth -= 1
                if gate_stack and depth == gate_stack[-1]:
                    gate_stack.pop()
        # a cfg(test) attribute followed by no brace on the same line stays
        # pending until the next `{`
    return mask


# `println!` / `print!` — library stdout. Anchored to a non-identifier prefix so
# the `eprintln!` / `eprint!` family (whose substring CONTAINS `println!`) is NOT
# matched: build.rs bans plain `println!`/`print!` (test-aware), while
# `eprintln!`/`eprint!` are banned ONLY when carrying `{:?}`/`{:#?}` debug
# formatting (the separate rule below). Test-aware.
PRINT_RE = re.compile(r"(^|[^A-Za-z0-9_])print(ln)?!\(")  # plain print(ln)!, not eprint
# `eprintln!`/`eprint!` carrying `{:?}` or `{:#?}` debug formatting (build.rs
# scan_for_debug_eprintln). Plain Display `eprintln!("…{x}")` is allowed. The
# `{:?}` / `{:#?}` specifier lives INSIDE the string literal, so this rule is
# tested against the RAW line (not the string-stripped code).
DEBUG_EPRINT_RE = re.compile(r"\beprint(ln)?!\(")
DEBUG_FMT_RE = re.compile(r"\{:#\?\}|\{:\?\}")
# `let _ ...` underscore-binding family.
LET_UNDERSCORE_RE = re.compile(r"\blet\s+(mut\s+)?_\b")
# `#[allow(...)]` / `#![allow(...)]` / `#[expect(...)]` / `#![expect(...)]`.
# build.rs exempts `clippy::` / `rustc::` / `rustdoc::`-namespaced lints — those
# are tool-lint configuration, not a silenced rustc warning. Mirror that.
ALLOW_RE = re.compile(r"#!?\[\s*(allow|expect)\s*\(\s*([^)]*)\)")
ALLOW_EXEMPT_PREFIXES = ("clippy::", "rustc::", "rustdoc::")
# `#[ignore]` / `#[ignore = "..."]`.
IGNORE_RE = re.compile(r"#\[\s*ignore\b")
# Bare `#[should_panic]` (no `expected = "..."`).
BARE_SHOULD_PANIC_RE = re.compile(r"#\[\s*should_panic\s*\]")
# Cargo feature gate.
CFG_FEATURE_RE = re.compile(r"#!?\[\s*cfg\s*\([^)]*feature\s*=")


def scan_file(path, rel):
    hits = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.read().split("\n")
    except OSError:
        return hits

    mask = compute_test_mask(lines, rel)
    for idx, raw in enumerate(lines):
        code = strip_strings_and_comments(raw)
        if not code.strip():
            continue
        in_test = mask[idx]

        for needle, label, test_aware in BANNED_SUBSTRINGS:
            if test_aware and in_test:
                continue
            if needle in code:
                hits.append((idx + 1, label, raw))

        # println!/print! — test-aware, eprint family excluded by the anchor.
        if not in_test and PRINT_RE.search(code):
            hits.append((idx + 1, "println!/print! in src (library stdout)", raw))
        # eprintln!/eprint! with {:?}/{:#?} debug formatting — test-aware. The
        # specifier lives inside the string literal, so match the RAW line.
        if not in_test and DEBUG_EPRINT_RE.search(code) and DEBUG_FMT_RE.search(raw):
            hits.append((idx + 1, "eprintln! with {:?} debug formatting", raw))

        if LET_UNDERSCORE_RE.search(code):
            hits.append((idx + 1, "let _ (use or delete the value)", raw))
        # allow/expect — exempt clippy::/rustc::/rustdoc:: tool lints (build.rs
        # does); only a real rustc-lint silencer is banned.
        m = ALLOW_RE.search(code)
        if m:
            lints = [t.strip() for t in m.group(2).split(",") if t.strip()]
            if any(not lt.startswith(ALLOW_EXEMPT_PREFIXES) for lt in lints):
                hits.append((idx + 1, "allow/expect (fix the code, don't silence)", raw))
        if IGNORE_RE.search(code):
            hits.append((idx + 1, "#[ignore] (no skipped tests)", raw))
        if BARE_SHOULD_PANIC_RE.search(code):
            hits.append((idx + 1, 'bare #[should_panic] (add expected = "...")', raw))
        if CFG_FEATURE_RE.search(code):
            hits.append((idx + 1, "cfg(feature) (no feature gates)", raw))

    return hits


def iter_rs_files(root):
    """Yield (abs_path, rel_path) for every .rs file build.rs would scan.

    src/ and crates/*/src/ are production trees; tests/, benches/, examples/ are
    included so the strict-everywhere rules still fire there (the test mask
    exempts the test-aware ones)."""
    scan_dirs = ["src", "tests", "benches", "bench", "examples"]
    crates = os.path.join(root, "crates")
    if os.path.isdir(crates):
        for crate in sorted(os.listdir(crates)):
            base = os.path.join(crates, crate)
            if os.path.isdir(base):
                for sub in ("src", "tests", "benches", "examples"):
                    scan_dirs.append(os.path.join("crates", crate, sub))

    for d in scan_dirs:
        abs_d = os.path.join(root, d)
        if not os.path.isdir(abs_d):
            continue
        for dirpath, _dirnames, filenames in os.walk(abs_d):
            for name in filenames:
                if name.endswith(".rs"):
                    abs_p = os.path.join(dirpath, name)
                    rel = os.path.relpath(abs_p, root).replace(os.sep, "/")
                    if rel == "build.rs":
                        continue
                    yield abs_p, rel


def main(argv):
    root = argv[1] if len(argv) > 1 else "."
    total = 0
    for abs_p, rel in iter_rs_files(root):
        for lineno, label, raw in scan_file(abs_p, rel):
            print(f"{rel}:{lineno}: [{label}] {raw.strip()}")
            total += 1
    if total:
        print()
        print(
            f"precheck: {total} banned-pattern hit(s). These fail build.rs — fix before push."
        )
        print(
            "          (text-scan subset of build.rs; clean is a smoke test, not a guarantee.)"
        )
        return 1
    print("precheck: text scan clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
