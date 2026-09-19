#!/usr/bin/env python3
"""Guard Rust item removals with source-graph evidence (#2818).

This deliberately does not try to prove reachability from linked symbols.  It
compares immutable source trees and refuses removal of public or test-scoped
items.  A private production item is automatically removable only when its
name has no other source occurrence in the base tree.  Ambiguity is therefore
resolved in favour of retaining source, not by guessing that an inlined or
generic item is dead.

An identity is `path:kind:name`, and several declarations can share one
(inherent methods of different types, cfg-alternative bodies), so a removal is
a fall in that identity's declaration count, not only its disappearance.
``--positive-control`` re-measures the sweep this guard exists for, because a
guard that has stopped refusing anything is byte-identical to a guard over a
diff that removed nothing.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile


LEDGER = "docs/source-removal-changes.json"
# `const fn`, `async unsafe fn` and `extern "C" fn` are functions, and `static mut`
# names its item after the `mut`: without the qualifier loop every `const fn` in a
# file shared the identity `const:fn`, so removing one while another survived was
# invisible. A keyword must end at whitespace, or a call line such as
# `constant_curvature_kernel_matrix(x)` reads as `const ant_curvature_kernel_matrix`.
ITEM = re.compile(
    r"^\s*(?P<vis>pub(?:\s*\([^)]*\))?\s+)?(?:(?:default|const|async|unsafe)\s+|extern\s+(?:\"[^\"]+\"\s+)?)*"
    r"(?P<kind>(?:fn|struct|enum|union|trait|type|const|static)(?=\s)|macro_rules!)\s*(?:mut\s+)?(?P<name>[A-Za-z_][A-Za-z_0-9]*)"
)
# The commit whose symbol-table criterion this guard was built to refuse: it
# deleted "every function no production artifact links", test-only helpers
# included. Each arm is pinned separately, because a guard that refuses the sweep
# through one arm alone would still pass a single total.
CONTROL = {"head": "d484a091a4d133aec6f3ed980754b6cda151d41b", "identities_before": 20237,
           "identities_after": 18424, "guarded_public": 1266, "guarded_test_scoped": 71,
           "guarded_private": 521, "queried_names": 510}


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def resolve(root, revision):
    return git(root, "rev-parse", "--verify", revision + "^{commit}").decode().strip()


# Where a comment or literal can begin. A raw string closes only at a quote followed
# by its own number of hashes, and a `'` opens a character literal only when one
# closes it; otherwise it is a lifetime or a loop label and stays code.
LITERAL_START = re.compile(r"//|/\*|(?<![A-Za-z_0-9])(?:b|c)?r(#*)\"|\"|'")
CHARACTER = re.compile(r"'(?:\\(?:u\{[0-9a-fA-F]+\}|x[0-9a-fA-F]{2}|.)|[^'\\\n])'")
# `#[cfg(test)]` modules here must be named `tests`, `test_support`, `tests_*` or
# `*_tests`, so an out-of-line module file with that stem is test scope as a whole
# even though it carries no attribute of its own.
TEST_MODULE_STEM = re.compile(r"tests|test_support|tests_\w+|\w+_tests")
# Rust 2021 resolves a string literal's inline format captures as uses: `{name}`, `{name:…}`,
# and a `name$` width or precision inside the spec. `{{` is an escaped brace, not a capture.
FORMAT_CAPTURE = re.compile(r"\{\{|\{([A-Za-z_][A-Za-z_0-9]*)?(:[^{}]*)?\}")
SPEC_ARGUMENT = re.compile(r"([A-Za-z_][A-Za-z_0-9]*)\$")


def keep_format_captures(literal):
    """Blank a string literal except the identifiers its inline format captures name."""
    out = list(re.sub(r"[^\n]", " ", literal))
    for capture in FORMAT_CAPTURE.finditer(literal):
        if capture.group() == "{{":
            continue
        spans = [capture.span(1)] if capture.group(1) else []
        if capture.group(2):
            spans += [(capture.start(2) + name.start(1), capture.start(2) + name.end(1))
                      for name in SPEC_ARGUMENT.finditer(capture.group(2))]
        for begin, end in spans:
            out[begin:end] = literal[begin:end]
    return "".join(out)


def strip_comments_and_literals(source):
    """Replace comments/literals with whitespace while preserving newlines.

    Treating every `'` as a quote blanked everything from `&'static str` to the
    next apostrophe in the file, so declarations after a lifetime went uninventoried.
    A string literal keeps its inline format captures as code: blanking
    `"{REFERENCE_ENV_MISSING}:{tool}"` made a constant used only there look unreferenced.
    """
    out, position = [], 0
    while True:
        start = LITERAL_START.search(source, position)
        if start is None:
            out.append(source[position:])
            return "".join(out)
        out.append(source[position:start.start()])
        token = start.group()
        if token == "//":
            end = source.find("\n", start.start())
            end = len(source) if end < 0 else end
        elif token == "/*":
            depth, end = 1, start.end()
            while depth:
                opening, closing = source.find("/*", end), source.find("*/", end)
                if closing < 0:
                    raise ValueError("unterminated block comment")
                if 0 <= opening < closing:
                    depth, end = depth + 1, opening + 2
                else:
                    depth, end = depth - 1, closing + 2
        elif token == '"':
            end = start.end()
            while end < len(source) and source[end] != '"':
                end += 2 if source[end] == "\\" else 1
            end = min(end + 1, len(source))
        elif token.endswith('"'):
            closing = source.find('"' + start.group(1), start.end())
            end = len(source) if closing < 0 else closing + 1 + len(start.group(1))
        else:
            character = CHARACTER.match(source, start.start())
            if character is None:
                out.append("'")
                position = start.end()
                continue
            end = character.end()
        text = source[start.start():end]
        out.append(keep_format_captures(text) if token.endswith('"') else re.sub(r"[^\n]", " ", text))
        position = end


def test_lines(source, path, clean=None):
    """Return lines under a compiler-enforced test file/attribute."""
    path = path.replace("\\", "/")
    parts = path.split("/")
    stem = parts[-1][:-3] if parts[-1].endswith(".rs") else parts[-1]
    if (path.startswith(("tests/", "benches/")) or any(p in ("tests", "benches") for p in parts[1:])
            or TEST_MODULE_STEM.fullmatch(stem)):
        return set(range(1, source.count("\n") + 2))
    clean = strip_comments_and_literals(source) if clean is None else clean
    lines = clean.splitlines()
    if any(re.search(r"#!\s*\[\s*cfg\s*\([^]]*\btest\b", line) for line in lines):
        return set(range(1, len(lines) + 1))
    marked, pending, depth, gates, nesting = set(), False, 0, [], 0
    for number, line in enumerate(lines, 1):
        if re.search(r"#\s*\[\s*(?:test\b|cfg\s*\([^]]*\btest\b)", line):
            pending = True
        starts_in_gate, starts_pending = bool(gates), pending
        for ch in line:
            if ch in "([":
                nesting += 1
            elif ch in ")]":
                nesting -= 1
            elif ch == "{":
                if pending:
                    gates.append(depth); pending = False
                depth += 1
            elif ch == "}":
                depth -= 1
                while gates and depth <= gates[-1]: gates.pop()
            elif ch == ";" and pending and nesting == 0:
                # A brace-less item (`#[cfg(test)] mod x;`, `#[cfg(test)] use y;`) ends at its
                # semicolon. Left pending, the attribute gated the next production block instead.
                pending = False
        if starts_in_gate or starts_pending or gates or pending:
            marked.add(number)
    return marked


def removed_declarations(before, after, key):
    survivors = after[key].get("declarations", 1) if key in after else 0
    return max(0, before[key].get("declarations", 1) - survivors)


# A `git grep -w` match is a whole run of these bytes (git's word characters are
# ASCII alphanumerics and `_`), so a name's whole-word occurrences are the runs equal to it.
WORD_RUN = re.compile(rb"[A-Za-z0-9_]+")


def source_words(root, revision, names):
    """Count what `git grep -h -o -w -F -e NAME... REVISION -- '*.rs'` prints, in one linear pass.

    The tree is read the way git grep reads it: regular `*.rs` files, and a blob
    with a NUL in its first 8000 bytes is binary and contributes no match. git
    grep itself does not scale in the number of names: for the 88 private names
    of a pure move of latent/survival/mod.rs (09-18) it took 0.3 s for one name
    and 15 s for all 88 on an 8-CPU node, and on a workstation it ran 318 CPU-s
    until a CPU watchdog killed it at 60 s, twice. Tokenising each blob once
    costs the same for one name as for thousands.
    """
    wanted = {name.encode() for name in names}
    words = Counter()
    if not wanted:
        return words
    blobs = []
    for entry in git(root, "ls-tree", "-r", "-z", revision).split(b"\0"):
        meta, _, path = entry.partition(b"\t")
        mode, kind, oid = (meta.split() + [b"", b"", b""])[:3]
        if kind == b"blob" and mode in (b"100644", b"100755") and path.endswith(b".rs"):
            blobs.append(oid)
    with tempfile.TemporaryFile() as ids:
        ids.write(b"".join(oid + b"\n" for oid in blobs)); ids.seek(0)
        batch = subprocess.Popen(["git", "-C", str(root), "cat-file", "--batch"], stdin=ids, stdout=subprocess.PIPE)
        for oid in blobs:
            header = batch.stdout.readline().split()
            size = int(header[2]) if len(header) == 3 and header[:2] == [oid, b"blob"] else -1
            data = batch.stdout.read(size) if size >= 0 else b""
            if size < 0 or len(data) != size:
                # A read that stops short has not measured anything.
                batch.kill(); batch.wait()
                raise subprocess.CalledProcessError(1, f"git cat-file --batch (blob {oid.decode()} of {revision} unreadable)")
            batch.stdout.read(1)
            if b"\0" not in data[:8000]:
                words.update(run for run in WORD_RUN.findall(data) if run in wanted)
        if batch.wait() != 0:
            raise subprocess.CalledProcessError(batch.returncode, "git cat-file --batch")
    return Counter({run.decode(): count for run, count in words.items()})


def changed_paths(root, base, head):
    """Each changed Rust file as (base path, head path), the same path unless it was renamed.

    `diff --name-only` under git's default rename detection printed only a renamed
    file's head path, so its base declarations were never read: c7768c15c2 renamed
    empirical_intercept_bracket_tests.rs, dropped two of its tests, and passed.
    Renames are requested explicitly instead of inherited from configuration, and a
    renamed file's base declarations are compared with what it declares where it now
    lives. A rename git does not detect is a deletion plus an addition, which refuses
    every guarded item of the old path; a copy removes nothing, so its new path is an
    addition.
    """
    fields = git(root, "diff", "--name-status", "-z", "--find-renames", base, head,
                 "--", "crates", "src", "tests").split(b"\0")
    pairs, i = [], 0
    while i < len(fields) - 1:
        status = fields[i][:1]
        if status in (b"R", b"C"):
            old, new, i = fields[i + 1], fields[i + 2], i + 3
            old = new if status == b"C" else old
        else:
            old = new = fields[i + 1]
            i += 2
        if old.endswith(b".rs") or new.endswith(b".rs"):
            pairs.append((old.decode(), new.decode()))
    return pairs


def changed_inventory(root, base, head):
    """Inventory only changed Rust files, then query removed private names globally.

    Reading every blob in this 1.3-million-line workspace on every PR would
    turn a cheap policy gate into a runtime gate. Git already provides the
    exact candidate set; global source queries are needed only for private
    names that actually lost a declaration, and one pass counts all of them.
    An identity carries its base path, including when the file was renamed.
    """
    pairs = changed_paths(root, base, head)

    def at(revision, side):
        found = {}
        for pair in pairs:
            path, identity_path = pair[side], pair[0]
            if not path.endswith(".rs"):
                continue
            try:
                source = subprocess.check_output(
                    ["git", "-C", str(root), "show", f"{revision}:{path}"],
                    stderr=subprocess.DEVNULL,
                ).decode()
            except subprocess.CalledProcessError:
                continue
            clean, tests = strip_comments_and_literals(source), test_lines(source, path)
            for number, line in enumerate(clean.splitlines(), 1):
                match = ITEM.match(line)
                if not match:
                    continue
                key = f"{identity_path}:{match['kind']}:{match['name']}"
                public, test_scoped = bool(match["vis"]), number in tests
                held = found.get(key)
                if held is None:
                    found[key] = {"path": identity_path, "line": number, "kind": match["kind"],
                                  "name": match["name"], "public": public,
                                  "test_scoped": test_scoped, "declarations": 1}
                else:
                    # The identity is guarded if ANY of its declarations is.
                    held["declarations"] += 1
                    held["public"] = held["public"] or public
                    held["test_scoped"] = held["test_scoped"] or test_scoped
        return found

    before, after = at(base, 0), at(head, 1)
    # Public and test-scoped removals are guarded outright and need no count. A
    # placeholder count for them would also overwrite a private homonym's real
    # count, letting that private item's callers go unseen.
    candidates = sorted({item["name"] for key, item in before.items()
                         if removed_declarations(before, after, key)
                         and not (item["public"] or item["test_scoped"])})
    # Counts are deliberately conservative: every occurrence, test source
    # included. A false positive asks for review; a false negative permits deletion.
    words = source_words(root, base, candidates)
    unseen = [name for name in candidates if words[name] < 1]
    if unseen:
        # Every candidate is declared in the base tree, so its own declaration
        # must be counted; a query that sees nothing has not measured anything.
        raise ValueError("source query did not find the declaration of: " + ", ".join(unseen[:20]))
    return before, after, words


def removals(before, after, production_words):
    blocked = []
    for key in sorted(before):
        removed = removed_declarations(before, after, key)
        if not removed:
            continue
        # One occurrence is the declaration itself. Any other source
        # occurrence is conservative evidence that a graph edge may exist.
        item = dict(before[key], removed_declarations=removed,
                    production_occurrences=production_words.get(before[key]["name"], 0))
        if item["public"] or item["test_scoped"] or item["production_occurrences"] > 1:
            blocked.append(item)
    return blocked


def check(before_sha, blocked, ledger):
    entries = [entry for entry in ledger if entry.get("base") == before_sha]
    if not blocked:
        if entries: raise ValueError("source-removal acknowledgement describes no guarded removal")
        return
    if len(entries) != 1: raise ValueError("guarded Rust items removed without exactly one acknowledgement: " + json.dumps(blocked))
    entry = entries[0]
    if set(entry) != {"base", "items", "reason", "evidence"}: raise ValueError("source-removal acknowledgement has missing or unknown fields")
    identities = [f"{x['path']}:{x['kind']}:{x['name']}" for x in blocked]
    if entry["items"] != identities: raise ValueError("source-removal acknowledgement does not exactly match guarded removals")
    if not all(isinstance(entry[x], str) and entry[x].strip() for x in ("reason", "evidence")):
        raise ValueError("source removal needs a semantic reason and caller/test evidence")


def positive_control(root):
    """Re-measure the #2818 sweep, so a guard refusing nothing cannot pass.

    The expected numbers are the ones this guard must keep reproducing; an
    inventory or predicate change that moves them has to move them here too, in review.
    """
    head = resolve(root, CONTROL["head"])
    base = resolve(root, head + "^")
    before, after, words = changed_inventory(root, base, head)
    blocked = removals(before, after, words)
    measured = {"head": head, "identities_before": len(before), "identities_after": len(after),
                "guarded_public": sum(1 for x in blocked if x["public"]),
                "guarded_test_scoped": sum(1 for x in blocked if x["test_scoped"] and not x["public"]),
                "guarded_private": sum(1 for x in blocked if not (x["public"] or x["test_scoped"])),
                "queried_names": len(words)}
    if measured != CONTROL:
        raise ValueError(f"positive control drifted from the measured #2818 sweep: {json.dumps(measured, sort_keys=True)}")
    try:
        check(base, blocked, [])
    except ValueError:
        return measured
    raise ValueError("positive control: the guard accepted the #2818 sweep without an acknowledgement")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base"); parser.add_argument("--head", default="HEAD")
    parser.add_argument("--positive-control", action="store_true", help="re-measure the #2818 sweep and stop")
    args = parser.parse_args(); root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel").decode().strip())
    if args.positive_control:
        print("Source-removal guard refused the #2818 sweep: " + json.dumps(positive_control(root), sort_keys=True))
        return
    if not args.base:
        parser.error("--base is required unless --positive-control is given")
    base, head = resolve(root, args.base), resolve(root, args.head)
    before, after, words = changed_inventory(root, base, head)
    blocked = removals(before, after, words)
    ledger = json.loads(git(root, "show", head + ":" + LEDGER))
    check(base, blocked, ledger)
    print(f"Source-removal guard verified {len(before)} -> {len(after)} Rust items; {len(blocked)} guarded removals.")


if __name__ == "__main__":
    try: main()
    except (ValueError, subprocess.CalledProcessError) as error:
        print(f"SOURCE REMOVAL GUARD FAILED: {error}", file=sys.stderr); sys.exit(1)
