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


def strip_comments_and_literals(source):
    """Replace comments/literals with whitespace while preserving newlines.

    Treating every `'` as a quote blanked everything from `&'static str` to the
    next apostrophe in the file, so declarations after a lifetime went uninventoried.
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
        out.append(re.sub(r"[^\n]", " ", source[start.start():end]))
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
    marked, pending, depth, gates = set(), False, 0, []
    for number, line in enumerate(lines, 1):
        if re.search(r"#\s*\[\s*(?:test\b|cfg\s*\([^]]*\btest\b)", line):
            pending = True
        starts_in_gate = bool(gates)
        for ch in line:
            if ch == "{":
                if pending:
                    gates.append(depth); pending = False
                depth += 1
            elif ch == "}":
                depth -= 1
                while gates and depth <= gates[-1]: gates.pop()
        if starts_in_gate or gates or pending:
            marked.add(number)
    return marked


def removed_declarations(before, after, key):
    survivors = after[key].get("declarations", 1) if key in after else 0
    return max(0, before[key].get("declarations", 1) - survivors)


def changed_inventory(root, base, head):
    """Inventory only changed Rust files, then query removed private names globally.

    Reading every blob in this 1.3-million-line workspace on every PR would
    turn a cheap policy gate into a runtime gate. Git already provides the
    exact candidate set; global source queries are needed only for private
    names that actually lost a declaration, and one pass counts all of them.
    """
    paths = git(root, "diff", "--name-only", "-z", base, head, "--", "crates", "src", "tests").split(b"\0")
    paths = [path.decode() for path in paths if path.endswith(b".rs")]

    def at(revision):
        found = {}
        for path in paths:
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
                key = f"{path}:{match['kind']}:{match['name']}"
                public, test_scoped = bool(match["vis"]), number in tests
                held = found.get(key)
                if held is None:
                    found[key] = {"path": path, "line": number, "kind": match["kind"],
                                  "name": match["name"], "public": public,
                                  "test_scoped": test_scoped, "declarations": 1}
                else:
                    # The identity is guarded if ANY of its declarations is.
                    held["declarations"] += 1
                    held["public"] = held["public"] or public
                    held["test_scoped"] = held["test_scoped"] or test_scoped
        return found

    before, after = at(base), at(head)
    # Public and test-scoped removals are guarded outright and need no count. A
    # placeholder count for them would also overwrite a private homonym's real
    # count, letting that private item's callers go unseen.
    candidates = sorted({item["name"] for key, item in before.items()
                         if removed_declarations(before, after, key)
                         and not (item["public"] or item["test_scoped"])})
    words = Counter()
    for start in range(0, len(candidates), 2000):
        patterns = [part for name in candidates[start:start + 2000] for part in ("-e", name)]
        query = subprocess.run(["git", "-C", str(root), "grep", "-h", "-o", "-w", "-F", *patterns, base,
                                "--", "*.rs"], stdout=subprocess.PIPE)
        if query.returncode != 0:
            raise subprocess.CalledProcessError(query.returncode, "git grep")
        # Counts are deliberately conservative: every occurrence, test source
        # included. A false positive asks for review; a false negative permits deletion.
        words.update(line.decode() for line in query.stdout.splitlines())
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
