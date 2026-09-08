#!/usr/bin/env python3
"""Guard Rust item removals with source-graph evidence (#2818).

This deliberately does not try to prove reachability from linked symbols.  It
compares immutable source trees and refuses removal of public or test-scoped
items.  A private production item is automatically removable only when its
name has no other production-source occurrence in the base tree.  Ambiguity is
therefore resolved in favour of retaining source, not by guessing that an
inlined or generic item is dead.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys


LEDGER = "docs/source-removal-changes.json"
ITEM = re.compile(
    r"^\s*(?P<vis>pub(?:\s*\([^)]*\))?\s+)?(?:async\s+|unsafe\s+|extern\s+(?:\"[^\"]+\"\s+)?)?"
    r"(?P<kind>fn|struct|enum|union|trait|type|const|static|macro_rules!)\s*!?\s*(?P<name>[A-Za-z_][A-Za-z_0-9]*)"
)
def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def resolve(root, revision):
    return git(root, "rev-parse", "--verify", revision + "^{commit}").decode().strip()


def strip_comments_and_literals(source):
    """Replace comments/literals with whitespace while preserving newlines."""
    out = list(source)
    i = 0
    block = 0
    while i < len(source):
        if block:
            if source.startswith("/*", i):
                block += 1; out[i:i + 2] = "  "; i += 2
            elif source.startswith("*/", i):
                block -= 1; out[i:i + 2] = "  "; i += 2
            else:
                if source[i] != "\n": out[i] = " "
                i += 1
        elif source.startswith("//", i):
            end = source.find("\n", i)
            end = len(source) if end < 0 else end
            out[i:end] = " " * (end - i); i = end
        elif source.startswith("/*", i):
            block = 1; out[i:i + 2] = "  "; i += 2
        elif source[i] in "\"'":
            quote = source[i]; out[i] = " "; i += 1
            while i < len(source):
                ch = source[i]
                if ch != "\n": out[i] = " "
                i += 1
                if ch == "\\" and i < len(source):
                    if source[i] != "\n": out[i] = " "
                    i += 1
                elif ch == quote:
                    break
        else:
            i += 1
    if block:
        raise ValueError("unterminated block comment")
    return "".join(out)


def test_lines(source, path):
    """Return lines under a compiler-enforced test file/attribute."""
    path = path.replace("\\", "/")
    parts = path.split("/")
    if path.startswith(("tests/", "benches/")) or any(p in ("tests", "benches") for p in parts[1:]):
        return set(range(1, source.count("\n") + 2))
    clean = strip_comments_and_literals(source)
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


def changed_inventory(root, base, head):
    """Inventory only changed Rust files, then query ambiguous names globally.

    Reading every blob in this 1.3-million-line workspace on every PR would
    turn a cheap policy gate into a runtime gate. Git already provides the
    exact candidate set; global source queries are needed only for private
    names that actually disappeared.
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
                if match:
                    key = f"{path}:{match['kind']}:{match['name']}"
                    found[key] = {"path": path, "line": number, "kind": match["kind"],
                                  "name": match["name"], "public": bool(match["vis"]),
                                  "test_scoped": number in tests}
        return found

    before, after = at(base), at(head)
    words = Counter()
    for item in before.values():
        if item["public"] or item["test_scoped"]:
            words[item["name"]] = 1
            continue
        query = subprocess.run(["git", "-C", str(root), "grep", "-w", "-c", item["name"], base,
                                "--", "*.rs"], stdout=subprocess.PIPE)
        # Counts are deliberately conservative and include test source. A
        # false positive asks for review; a false negative permits deletion.
        words[item["name"]] = sum(int(line.rsplit(b":", 1)[1]) for line in query.stdout.splitlines())
    return before, after, words


def removals(before, after, production_words):
    blocked = []
    for key in sorted(before.keys() - after.keys()):
        item = before[key]
        # One occurrence is the declaration itself. Any other source
        # occurrence is conservative evidence that a graph edge may exist.
        item = dict(item, production_occurrences=production_words[item["name"]])
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True); parser.add_argument("--head", default="HEAD")
    args = parser.parse_args(); root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel").decode().strip())
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
