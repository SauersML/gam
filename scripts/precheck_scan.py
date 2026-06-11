#!/usr/bin/env python3
"""Faithful text-scan port of the gam build.rs ban rules (no compiler).

Invoked by scripts/precheck.sh as `precheck_scan.py <REPO_ROOT>`. Reports each
banned pattern as `path:line: [rule]` over `src/`, `tests/`, `crates/*` etc.,
exit 1 if any violation is found. Runs in seconds with no compiler.

It reproduces build.rs's per-line treatment faithfully so it is CLEAN on a clean
main (a gate that cries wolf gets ignored). In particular it carries the THREE
exemptions a naive scan gets wrong:

  1. clippy: build.rs exempts ALL `clippy::` lints from the #[allow]/#[expect]
     ban (only non-clippy tokens count), so `#[allow(clippy::too_many_arguments)]`
     is NOT a violation.
  2. eprintln!/println!: only `println!(`/`print!(` are banned, and `eprintln!(`
     lexically CONTAINS `println!(`; we match `println!` only when not preceded
     by `e`.
  3. PyO3 GIL token: build.rs exempts `_py: Python<...>` underscore params.

It uses build.rs's STATE-CARRIED string/comment stripper (multi-line `"..."`,
raw strings `r#"..."#`, char-vs-lifetime). A per-line-only stripper miscounts
braces across multi-line strings, which silently drops the #[cfg(test)] mask and
produces phantom `println!`/`unreachable!` hits inside test modules.

Rules covered: the lexical banned-substring table (debug_assert!, dbg!,
println!, env::var, == true, thread::sleep, …), `let _` bindings, #[allow]/
#[expect] of non-clippy lints, #[cfg(test)] on src/ items, bare #[should_panic],
#[ignore], cfg(feature = "…") gates (cuda exempt), underscore fn params, and
assertion-less #[test] (with one-hop delegation to assert_*/expect_*/require_*/
ensure_*-named helpers).

OUT OF SCOPE (left to build.rs / CI — they need whole-crate analysis): src item
used only by tests, zero-consumer pub(crate), dodge-named / no-op-sentinel /
stub fns. A clean run here is necessary, not sufficient — but it catches every
pattern that recurs in the wheel-cycle. Every hit IS a real build.rs failure.
"""

import os
import sys


# ---------------------------------------------------------------------------
# File walk — mirror build.rs collect_scannable_files for .rs files.
# ---------------------------------------------------------------------------

SKIP_DIR_NAMES = {
    "target", "node_modules", "__pycache__", "pydeps", "site-packages",
    "venv", "dist", "build", "site",
}


def iter_rs_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root).replace("\\", "/")
        if rel_dir == ".":
            rel_dir = ""
        if rel_dir == "bench/runtime/pydeps" or rel_dir.startswith("bench/runtime/pydeps/"):
            dirnames[:] = []
            continue
        kept = []
        for d in dirnames:
            if d == "target" or d.startswith("target-"):
                continue
            if d in SKIP_DIR_NAMES:
                continue
            if d.startswith(".") and d != ".github":
                continue
            kept.append(d)
        dirnames[:] = kept
        for name in filenames:
            if not name.endswith(".rs"):
                continue
            rel = os.path.join(rel_dir, name).replace("\\", "/") if rel_dir else name
            if rel == "build.rs":
                continue
            yield rel, os.path.join(dirpath, name)


# ---------------------------------------------------------------------------
# State-carried string/comment stripper (build.rs strip_file_lines).
# ---------------------------------------------------------------------------

def _strip_line(line, in_str, quote, raw_hashes):
    b = line.encode("utf-8", "replace")
    out = bytearray()
    i = 0
    n = len(b)
    while i < n:
        c = b[i]
        if in_str and quote == ord("r"):
            if c == ord('"'):
                need = raw_hashes
                k = i + 1
                count = 0
                while k < n and b[k] == ord("#") and count < need:
                    k += 1
                    count += 1
                if count == need:
                    in_str = False
                    quote = 0
                    raw_hashes = 0
                    out.append(ord('"'))
                    out.extend(b"#" * need)
                    i = k
                    continue
            out.append(ord(" "))
            i += 1
            continue
        if in_str:
            if c == ord("\\") and i + 1 < n:
                out.append(ord(" "))
                out.append(ord(" "))
                i += 2
                continue
            if c == quote:
                in_str = False
                out.append(c)
            else:
                out.append(ord(" "))
            i += 1
            continue
        if c == ord("/") and i + 1 < n and b[i + 1] == ord("/"):
            break
        prev_is_ident = i > 0 and (chr(b[i - 1]).isalnum() or b[i - 1] == ord("_"))
        if (not prev_is_ident) and (c == ord("r") or (c == ord("b") and i + 1 < n and b[i + 1] == ord("r"))):
            prefix_len = 2 if c == ord("b") else 1
            k = i + prefix_len
            hashes = 0
            while k < n and b[k] == ord("#"):
                k += 1
                hashes += 1
            if k < n and b[k] == ord('"') and hashes <= 255:
                in_str = True
                quote = ord("r")
                raw_hashes = hashes
                out.extend(b[i:k + 1])
                i = k + 1
                continue
        if c == ord('"'):
            in_str = True
            quote = c
            out.append(c)
            i += 1
            continue
        if c == ord("'"):
            is_char_lit = False
            if i + 1 < n and b[i + 1] == ord("\\"):
                k = i + 2
                while k < n and k < i + 10:
                    if b[k] == ord("'"):
                        is_char_lit = True
                        break
                    k += 1
            elif i + 2 < n and b[i + 1] != ord("'") and b[i + 2] == ord("'"):
                is_char_lit = True
            elif i + 3 < n and b[i + 1] != ord("'") and b[i + 3] == ord("'"):
                is_char_lit = True
            if is_char_lit:
                in_str = True
                quote = c
                out.append(c)
                i += 1
                continue
            out.append(c)
            i += 1
            continue
        out.append(c)
        i += 1
    return out.decode("utf-8", "replace"), in_str, quote, raw_hashes


def strip_file_lines(content):
    out = []
    in_str = False
    quote = 0
    raw_hashes = 0
    for line in content.split("\n"):
        s, in_str, quote, raw_hashes = _strip_line(line, in_str, quote, raw_hashes)
        out.append(s)
    return out


def is_ident_byte(ch):
    return ch == "_" or ch.isalnum()


# ---------------------------------------------------------------------------
# #[cfg(test)] detection + per-line test-scope mask.
# ---------------------------------------------------------------------------

def is_cfg_test_attr_line(stripped):
    i = 0
    n = len(stripped)
    while i + 1 < n:
        if stripped[i] == "#" and stripped[i + 1] == "[":
            rest = stripped[i + 2:]
            pos = rest.find("cfg(")
            if pos != -1 and (pos == 0 or rest[pos - 1] != "_"):
                inner = rest[pos + 4:]
                tok = ""
                for ch in inner:
                    if ch in "(),":
                        if tok.strip() == "test":
                            return True
                        tok = ""
                    else:
                        tok += ch
        i += 1
    return False


def path_is_test_file(rel):
    if (rel.startswith("tests/") or rel.startswith("bench/")
            or rel.startswith("benches/") or rel.startswith("examples/")):
        return True
    if rel.startswith("crates/"):
        rest = rel[len("crates/"):]
        slash = rest.find("/")
        if slash != -1:
            tail = rest[slash + 1:]
            return tail.startswith("tests/") or tail.startswith("benches/")
    return False


def compute_test_mask(stripped_lines, rel):
    n = len(stripped_lines)
    if path_is_test_file(rel):
        return [True] * n
    mask = [False] * n
    depth = 0
    pending_attr = False
    gate_stack = []
    for idx in range(n):
        stripped = stripped_lines[idx]
        if is_cfg_test_attr_line(stripped):
            pending_attr = True
        mask[idx] = len(gate_stack) > 0
        for ch in stripped:
            if ch == "{":
                depth += 1
                if pending_attr:
                    gate_stack.append(depth - 1)
                    pending_attr = False
            elif ch == "}":
                if gate_stack and depth - 1 == gate_stack[-1]:
                    gate_stack.pop()
                depth -= 1
    return mask


def path_in_src(rel):
    if rel.startswith("src/"):
        return True
    if rel.startswith("crates/"):
        rest = rel[len("crates/"):]
        i = rest.find("/")
        if i != -1:
            return rest[i + 1:].startswith("src/")
    return False


# ---------------------------------------------------------------------------
# Banned substrings. (needle, label, test_aware). `println!`/`print!` get the
# eprintln-guard below; everything else is a plain stripped-line substring.
# ---------------------------------------------------------------------------

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
    ('file!().ends_with(".rs"', 'file!().ends_with(".rs"', False),
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


def _has_stdout_print(stripped):
    """`println!(` / `print!(` NOT part of `eprintln!(` / `eprint!(`."""
    for needle in ("println!(", "print!("):
        start = 0
        while True:
            pos = stripped.find(needle, start)
            if pos == -1:
                break
            if pos == 0 or stripped[pos - 1] != "e":
                return needle.rstrip("(")
            start = pos + 1
    return None


def scan_banned_substrings(rel, raw_lines, stripped_lines, mask, hits):
    for idx, stripped in enumerate(stripped_lines):
        in_test = mask[idx]
        for needle, label, test_aware in BANNED_SUBSTRINGS:
            if test_aware and in_test:
                continue
            if needle in stripped:
                hits.append((rel, idx + 1, label, raw_lines[idx].strip()))
        if not in_test:
            label = _has_stdout_print(stripped)
            if label is not None:
                hits.append((rel, idx + 1, label, raw_lines[idx].strip()))


# ---------------------------------------------------------------------------
# let _ binding (build.rs stripped_line_has_let_underscore).
# ---------------------------------------------------------------------------

def _tuple_all_underscore(s, start):
    n = len(s)
    if start >= n or s[start] != "(":
        return None
    depth = 0
    k = start
    any_binding = False
    all_underscore = True
    while k < n:
        ch = s[k]
        if ch == "(":
            depth += 1
            k += 1
        elif ch == ")":
            depth -= 1
            k += 1
            if depth == 0:
                return all_underscore if any_binding else None
        elif ch in (",", " ", "\t"):
            k += 1
        elif ch == "=":
            return None
        else:
            if ch == "_" or ch.isalpha():
                sidx = k
                while k < n and is_ident_byte(s[k]):
                    k += 1
                word = s[sidx:k]
                if word in ("mut", "ref"):
                    continue
                m = k
                while m < n and s[m] in (" ", "\t"):
                    m += 1
                if m < n:
                    nb = s[m]
                    if nb in ("(", "{"):
                        return None
                    if nb == ":" and m + 1 < n and s[m + 1] == ":":
                        return None
                any_binding = True
                if not word.startswith("_"):
                    all_underscore = False
            else:
                return None
    return None


def stripped_line_has_let_underscore(s):
    n = len(s)
    i = 0
    while i < n:
        if (i + 3 < n and s[i:i + 3] == "let"
                and (i == 0 or not is_ident_byte(s[i - 1]))
                and s[i + 3].isspace()):
            j = i + 3
            while j < n and s[j].isspace():
                j += 1
            if (j + 3 <= n and s[j:j + 3] == "mut"
                    and j + 3 < n and s[j + 3].isspace()):
                j += 3
                while j < n and s[j].isspace():
                    j += 1
            if j < n and s[j] == "_":
                return True
            if j < n and s[j] == "(" and _tuple_all_underscore(s, j) is True:
                return True
            i = j
            continue
        i += 1
    return False


def scan_let_underscore(rel, raw_lines, stripped_lines, mask, hits):
    for idx, stripped in enumerate(stripped_lines):
        if mask[idx]:
            continue
        if stripped_line_has_let_underscore(stripped):
            hits.append((rel, idx + 1, "let _ binding", raw_lines[idx].strip()))


# ---------------------------------------------------------------------------
# #[allow(...)] / #[expect(...)] silencing a non-clippy lint.
# ---------------------------------------------------------------------------

def scan_banned_allow(rel, raw_lines, stripped_lines, hits):
    for idx, code in enumerate(stripped_lines):
        if "#[" not in code:
            continue
        for silencer in ("allow(", "expect("):
            search_from = 0
            while True:
                m = code.find(silencer, search_from)
                if m == -1:
                    break
                k = m
                while k > 0 and code[k - 1].isspace():
                    k -= 1
                prefix = code[:k]
                start = m + len(silencer)
                if not (prefix.endswith("#[") or prefix.endswith("#![")):
                    search_from = start
                    continue
                end = code.find(")", start)
                if end == -1:
                    break
                first_label = None
                any_non_clippy = False
                for tok in code[start:end].split(","):
                    t = tok.strip()
                    if not t or t.startswith("clippy::"):
                        continue
                    any_non_clippy = True
                    if first_label is None:
                        for pfx in ("rustc::", "rustdoc::"):
                            if t.startswith(pfx):
                                t = t[len(pfx):]
                        first_label = t
                if any_non_clippy:
                    attr = silencer.rstrip("(")
                    hits.append((rel, idx + 1, f"{attr}({first_label or '<empty>'})",
                                 raw_lines[idx].strip()))
                search_from = end + 1
                if search_from >= len(code):
                    break


# ---------------------------------------------------------------------------
# #[cfg(test)] on a src/ item (not a private test submodule).
# ---------------------------------------------------------------------------

def is_exempt_test_submodule_name(name):
    return (name == "tests" or name == "test_support"
            or name.startswith("tests_") or name.endswith("_tests"))


def scan_cfg_test_on_src(rel, raw_lines, stripped_lines, hits):
    if not path_in_src(rel):
        return
    n = len(stripped_lines)
    for idx in range(n):
        if not is_cfg_test_attr_line(stripped_lines[idx]):
            continue
        j = idx + 1
        item_idx = None
        while j < n:
            t = stripped_lines[j].strip()
            if t == "" or t.startswith("//"):
                j += 1
                continue
            if t.startswith("#[") or t.startswith("#!["):
                j += 1
                continue
            item_idx = j
            break
        if item_idx is None:
            continue
        item_trim = raw_lines[item_idx].lstrip()
        if item_trim.startswith("mod "):
            name = ""
            for ch in item_trim[len("mod "):]:
                if ch.isalnum() or ch == "_":
                    name += ch
                else:
                    break
            if name and is_exempt_test_submodule_name(name):
                continue
        desc = []
        for ch in item_trim:
            if ch in ("{", ";"):
                break
            desc.append(ch)
        hits.append((rel, idx + 1, f"#[cfg(test)] on src item: {''.join(desc).strip()[:80]}",
                     raw_lines[idx].strip()))


# ---------------------------------------------------------------------------
# bare #[should_panic] (no expected = "...").
# ---------------------------------------------------------------------------

def scan_bare_should_panic(rel, raw_lines, hits):
    for idx, line in enumerate(raw_lines):
        t = line.lstrip()
        if t.startswith("#[should_panic]") or (t.startswith("#[should_panic") and "expected" not in t):
            hits.append((rel, idx + 1, '#[should_panic] without expected = "..."', line.strip()))


# ---------------------------------------------------------------------------
# #[ignore] test attribute (no skipped tests).
# ---------------------------------------------------------------------------

def scan_ignored_tests(rel, raw_lines, hits):
    for idx, line in enumerate(raw_lines):
        t = line.lstrip()
        if t.startswith("#[ignore]") or t.startswith("#[ignore ="):
            hits.append((rel, idx + 1, "#[ignore] (no skipped tests)", line.strip()))


# ---------------------------------------------------------------------------
# cfg(feature = "...") gate (no feature gates; "cuda" is exempt).
# ---------------------------------------------------------------------------

def _line_has_feature_predicate(s):
    i = 0
    kw = "feature"
    n = len(s)
    while i + len(kw) <= n:
        if s[i:i + len(kw)] == kw:
            before_ok = i == 0 or not is_ident_byte(s[i - 1])
            after = i + len(kw)
            after_ok = after == n or not is_ident_byte(s[after])
            if before_ok and after_ok:
                j = after
                while j < n and s[j].isspace():
                    j += 1
                if j < n and s[j] == "=":
                    return True
        i += 1
    return False


def _line_is_cuda_feature_gate(s):
    rest = s
    saw = False
    while True:
        pos = rest.find("feature")
        if pos == -1:
            return saw
        after = rest[pos + len("feature"):]
        eq = after.find("=")
        if eq == -1:
            return False
        after_eq = after[eq + 1:].lstrip()
        if not after_eq.startswith('"cuda"'):
            return False
        saw = True
        rest = after_eq[len('"cuda"'):]


def scan_feature_cfg_gates(rel, raw_lines, stripped_lines, hits):
    for idx, stripped in enumerate(stripped_lines):
        if "#[" not in stripped and "#![" not in stripped:
            continue
        if "cfg(" not in stripped and "cfg_attr(" not in stripped:
            continue
        if _line_has_feature_predicate(stripped) and not _line_is_cuda_feature_gate(stripped):
            hits.append((rel, idx + 1, "cfg(feature = ...) gate", raw_lines[idx].strip()))


# ---------------------------------------------------------------------------
# underscore-prefixed fn parameter (with the PyO3 `Python<...>` exemption).
# ---------------------------------------------------------------------------

def _locate_fn_keyword(s):
    i = 0
    n = len(s)
    while True:
        k = s.find("fn", i)
        if k == -1:
            return None
        before_ok = k == 0 or not is_ident_byte(s[k - 1])
        after_ok = k + 2 >= n or not is_ident_byte(s[k + 2])
        if before_ok and after_ok:
            return k
        i = k + 1


def scan_underscore_fn_args(rel, stripped_lines, raw_lines, hits):
    n = len(stripped_lines)
    idx = 0
    while idx < n:
        s = stripped_lines[idx]
        pos = _locate_fn_keyword(s)
        if pos is None:
            idx += 1
            continue
        after = s[pos + 2:].lstrip()
        if not after or not (after[0].isalpha() or after[0] == "_"):
            idx += 1
            continue
        sig = []
        paren = 0
        ang = 0
        started = False
        done = False
        for j in range(idx, min(idx + 64, n)):
            piece = stripped_lines[j][pos:] if j == idx else stripped_lines[j]
            sig.append(piece)
            for ch in piece:
                if ch == "<":
                    ang += 1
                elif ch == ">":
                    ang = max(0, ang - 1)
                elif ch == "(" and ang == 0:
                    paren += 1
                    started = True
                elif ch == ")" and ang == 0:
                    paren -= 1
                    if started and paren == 0:
                        done = True
                        break
                elif ch in (";", "{") and not started:
                    done = True
                    break
            if done:
                break
        sig_text = "".join(sig)
        op = _first_paren_depth0(sig_text)
        if op is None:
            idx += 1
            continue
        params = _param_list(sig_text, op)
        if params is None:
            idx += 1
            continue
        for p in _split_top_level(params):
            name, type_after = _param_name_and_type(p)
            if name is None or not name.startswith("_"):
                continue
            # PyO3 GIL token exemption (build.rs:3271).
            if type_after is not None and type_after.lstrip().startswith("Python<"):
                continue
            hits.append((rel, idx + 1, f"underscore fn param: {name}", raw_lines[idx].strip()))
            break
        idx += 1


def _first_paren_depth0(s):
    ang = 0
    for k, ch in enumerate(s):
        if ch == "<":
            ang += 1
        elif ch == ">":
            ang = max(0, ang - 1)
        elif ch == "(" and ang == 0:
            return k
    return None


def _param_list(s, op):
    depth = 0
    for k in range(op, len(s)):
        if s[k] == "(":
            depth += 1
        elif s[k] == ")":
            depth -= 1
            if depth == 0:
                return s[op + 1:k]
    return None


def _split_top_level(params):
    out = []
    depth = 0
    cur = []
    for ch in params:
        if ch in "(<[":
            depth += 1
        elif ch in ")>]":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur))
    return out


def _param_name_and_type(p):
    p = p.strip()
    while p.startswith("#["):
        c = p.find("]")
        if c == -1:
            break
        p = p[c + 1:].lstrip()
    # Strip &, &mut, mut prefixes that precede the binding.
    while True:
        before = p
        if p.startswith("&mut "):
            p = p[5:].lstrip()
        elif p.startswith("&"):
            p = p[1:].lstrip()
        elif p.startswith("mut "):
            p = p[4:].lstrip()
        if p == before:
            break
    # Find binding name then `:` then type.
    depth = 0
    for k, ch in enumerate(p):
        if ch in "(<[":
            depth += 1
        elif ch in ")>]":
            depth = max(0, depth - 1)
        elif ch == ":" and depth == 0:
            name = p[:k].strip()
            type_after = p[k + 1:]
            ident = ""
            for c in name:
                if c == "_" or c.isalnum():
                    ident += c
                else:
                    break
            if ident == name and ident:
                return ident, type_after
            return None, None
    return None, None  # no top-level colon: receiver / shorthand


# ---------------------------------------------------------------------------
# #[test] without any assertion (with one-hop delegated-assertion).
# ---------------------------------------------------------------------------

ASSERTION_MARKERS = (
    "assert!(", "assert_eq!(", "assert_ne!(",
    "debug_assert!(", "debug_assert_eq!(", "debug_assert_ne!(",
    "panic!(", "unreachable!(", "unimplemented!(", "todo!(",
)


_HELPER_PREFIXES = ("assert_", "expect_", "require_", "ensure_")


def _has_assertion_helper_macro(s):
    n = len(s)
    i = 0
    while i < n:
        if s[i] != "!":
            i += 1
            continue
        nxt = s[i + 1] if i + 1 < n else ""
        if nxt not in ("(", "[", "{"):
            i += 1
            continue
        start = i
        while start > 0 and is_ident_byte(s[start - 1]):
            start -= 1
        if start != i:
            name = s[start:i]
            if name.startswith(_HELPER_PREFIXES):
                return True
        i += 1
    return False


def _has_assertion_helper_call(s):
    n = len(s)
    i = 0
    while i < n:
        if s[i] != "(":
            i += 1
            continue
        if i > 0 and s[i - 1] == "!":
            i += 1
            continue
        start = i
        while start > 0 and is_ident_byte(s[start - 1]):
            start -= 1
        if start != i:
            name = s[start:i]
            if name.startswith(_HELPER_PREFIXES):
                return True
        i += 1
    return False


def _has_propagating_question(s):
    n = len(s)
    i = 0
    while i < n:
        if s[i] == "?":
            k = i + 1
            while k < n and s[k] == " ":
                k += 1
            if k == n:
                return True
            if s[k] in (";", ",", ".", ")"):
                return True
        i += 1
    return False


def line_is_assertion_shaped(s):
    # Mirrors build.rs line_is_assertion_shaped: assert/debug_assert/panic-shape
    # macros, `?`-propagation, and the assert_*/expect_*/require_*/ensure_*
    # named-helper macro AND bare-call shapes. Note: a plain `.expect(...)` /
    # `.unwrap()` is NOT recognized (build.rs requires the `expect_`/`assert_`
    # underscore-prefixed helper name), so a test whose only check is `.expect`
    # is genuinely flagged by build.rs too.
    if any(m in s for m in ASSERTION_MARKERS):
        return True
    if _has_propagating_question(s):
        return True
    if _has_assertion_helper_macro(s):
        return True
    if _has_assertion_helper_call(s):
        return True
    return False


def _find_body_open(stripped_lines, idx):
    n = len(stripped_lines)
    paren = 0
    ang = 0
    started = False
    base = _locate_fn_keyword(stripped_lines[idx]) or 0
    for j in range(idx, min(idx + 64, n)):
        s = stripped_lines[j][base:] if j == idx else stripped_lines[j]
        for ch in s:
            if ch == "<":
                ang += 1
            elif ch == ">":
                ang = max(0, ang - 1)
            elif ch == "(" and ang == 0:
                paren += 1
                started = True
            elif ch == ")" and ang == 0:
                paren = max(0, paren - 1)
            elif ch == "{" and started and paren == 0:
                return j
            elif ch == ";" and started and paren == 0:
                return None
    return None


def _find_matching_close(stripped_lines, open_line):
    depth = 0
    seen = False
    for j in range(open_line, len(stripped_lines)):
        for ch in stripped_lines[j]:
            if ch == "{":
                depth += 1
                seen = True
            elif ch == "}":
                depth -= 1
                if seen and depth == 0:
                    return j
    return None


def _index_fns(stripped_lines):
    out = []
    n = len(stripped_lines)
    for i in range(n):
        s = stripped_lines[i]
        pos = _locate_fn_keyword(s)
        if pos is None:
            continue
        after = s[pos + 2:].lstrip()
        if not after or not (after[0].isalpha() or after[0] == "_"):
            continue
        name = ""
        for ch in after:
            if ch == "_" or ch.isalnum():
                name += ch
            else:
                break
        ol = _find_body_open(stripped_lines, i)
        if name and ol is not None:
            cl = _find_matching_close(stripped_lines, ol)
            if cl is not None:
                out.append((name, ol, cl))
    return out


def _collect_calls(s, out):
    n = len(s)
    i = 0
    while i < n:
        if s[i] != "(":
            i += 1
            continue
        if i > 0 and s[i - 1] == "!":
            i += 1
            continue
        start = i
        while start > 0 and is_ident_byte(s[start - 1]):
            start -= 1
        if start < i:
            out.add(s[start:i])
        i += 1


def _line_has_test_attr(stripped):
    # `#[test]` as an attribute on a (string/comment-stripped) line. A `///`
    # doc-comment mentioning `#[test]` strips to whitespace, so it won't match.
    t = stripped.strip()
    return t.startswith("#[test]") or t.startswith("#[ test ]") or "#[test]" in t.replace(" ", "")


def scan_useless_tests(rel, stripped_lines, raw_lines, hits):
    if not any("#[test]" in ln.replace(" ", "") for ln in stripped_lines):
        return
    n = len(stripped_lines)
    fns = _index_fns(stripped_lines)
    by_open = {o: (nm, o, c) for (nm, o, c) in fns}

    # build.rs follows delegated assertions up to MAX_DELEGATION_DEPTH = 4 hops
    # (a #[test] → helper → helper → … chain). Matching that depth (and tracking
    # the visited helper set to terminate on mutually-recursive helpers) is what
    # keeps the gate from false-positiving a test whose asserts live two or more
    # helper levels down — the failure mode the team flagged.
    MAX_DELEGATION_DEPTH = 4

    def asserts(open_l, close_l, depth=MAX_DELEGATION_DEPTH, visited=None):
        if visited is None:
            visited = {open_l}
        for j in range(open_l, close_l + 1):
            if line_is_assertion_shaped(stripped_lines[j]):
                return True
        if depth == 0:
            return False
        called = set()
        for j in range(open_l, close_l + 1):
            _collect_calls(stripped_lines[j], called)
        for (nm, o2, c2) in fns:
            if nm in called and o2 not in visited:
                visited.add(o2)
                if asserts(o2, c2, depth - 1, visited):
                    return True
        return False

    for idx in range(n):
        if not _line_has_test_attr(stripped_lines[idx]):
            continue
        if "#[should_panic" in "".join(raw_lines[max(0, idx - 1):idx + 3]):
            continue
        j = idx + 1
        target = None
        while j < min(idx + 12, n):
            if _locate_fn_keyword(stripped_lines[j]) is not None:
                ol = _find_body_open(stripped_lines, j)
                if ol is not None and ol in by_open:
                    target = by_open[ol]
                break
            j += 1
        if target is not None and not asserts(target[1], target[2]):
            hits.append((rel, idx + 1, "#[test] without assertions", raw_lines[idx].strip()))


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

def main(argv):
    root = argv[1] if len(argv) > 1 else "."
    hits = []
    for rel, full in iter_rs_files(root):
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()
        except OSError:
            continue
        raw_lines = content.split("\n")
        stripped_lines = strip_file_lines(content)
        if len(stripped_lines) < len(raw_lines):
            stripped_lines += [""] * (len(raw_lines) - len(stripped_lines))
        mask = compute_test_mask(stripped_lines, rel)
        scan_banned_substrings(rel, raw_lines, stripped_lines, mask, hits)
        scan_let_underscore(rel, raw_lines, stripped_lines, mask, hits)
        scan_banned_allow(rel, raw_lines, stripped_lines, hits)
        scan_cfg_test_on_src(rel, raw_lines, stripped_lines, hits)
        scan_bare_should_panic(rel, raw_lines, hits)
        scan_ignored_tests(rel, raw_lines, hits)
        scan_feature_cfg_gates(rel, raw_lines, stripped_lines, hits)
        scan_underscore_fn_args(rel, stripped_lines, raw_lines, hits)
        scan_useless_tests(rel, stripped_lines, raw_lines, hits)

    if not hits:
        print("precheck: clean — no banned patterns found (text-scan tier).")
        return 0
    hits.sort(key=lambda h: (h[0], h[1]))
    print(f"precheck: FOUND {len(hits)} banned-pattern violation(s) (build.rs would fail):\n")
    for rel, lineno, label, text in hits:
        print(f"  {rel}:{lineno}: [{label}]")
        if text:
            print(f"      {text}")
    print("\nThese fail the build.rs gate — fix before pushing. (Deep cross-file rules")
    print("are not scanned here; run `scripts/precheck.sh --check` or rely on CI.)")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
