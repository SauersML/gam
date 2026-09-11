#!/usr/bin/env python3
"""Report production items that only tests keep alive (#2818).

The sweep behind #2818 asked "does a production artifact link this function?"
and answered from symbol tables, a question that is vacuously false for test
support, blind to generics and LTO, and unable to see the Rust library as a
product at all. This asks the source-graph question `build.rs` answered as
`scan_for_src_items_used_only_by_tests` until `f3ffc54e5` (2026-07-15) removed
it: which items declared in production source does no production line name?

Two classes are reported from the tree at one immutable revision:

* test-only: a non-`pub` production item named by test code (a test file, or a
  `#[cfg(test)]` region of a source file) and by no production line. It is test
  support in the wrong scope, or production code whose only callers are tests.
* unreferenced: a `pub(crate)`/`pub(super)`/`pub(in ...)` item named nowhere.

Bare `pub` items are out of scope: their consumers may live outside the
workspace, which is exactly the library surface a symbol-table sweep deletes.
Definitions inside test scope are never candidates -- the predicate is vacuously
true there, which is how #2818 happened. Trait members, in a trait definition or
an `impl Trait for` block, carry no visibility of their own and are dispatched
rather than named, so they are skipped, as are names of at most two characters
and a list of names every crate declares. A private item nothing names is left
to rustc.

The reference test is lexical: a name on a production line in another file, or on
another line of the same file, is a consumer -- a `pub use` re-export included,
since it is such a line. That errs toward reporting less, never toward calling a
live item dead. Comments and literals never supply a reference. Where this lives:
beside `scripts/test_census.py`, never in `build.rs` (#2110 -- a gate there took
the gamfit wheel down).
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys

from source_removal_guard import git, resolve, strip_comments_and_literals, test_lines

EXEMPT_NAMES = frozenset({
    "new", "default", "iter", "len", "is_empty", "clone", "from", "into", "as_ref", "as_mut",
    "next", "build", "with", "to_string", "display", "fmt", "index", "borrow", "drop", "main",
    "deref", "deref_mut", "hash", "eq", "ne", "cmp", "partial_cmp", "iter_mut", "into_iter",
    "as_slice", "as_str",
})
DEFINITION = re.compile(
    r"^\s*(?:(?P<scoped>pub\s*\([^)]*\)\s*)|(?P<public>pub\s+))?"
    r"(?:(?:default|const|async|unsafe)\s+|extern\s+(?:\"[^\"]*\"\s+)?)*"
    r"(?P<kind>fn|struct|enum|union|trait|type|const|static)\s+(?:mut\s+)?(?P<name>[A-Za-z_][A-Za-z_0-9]*)"
)
IMPL = re.compile(r"^\s*(?:(?:unsafe|default)\s+)*impl\b")
TRAIT = re.compile(r"^\s*(?:pub(?:\s*\([^)]*\))?\s+)?(?:(?:unsafe|auto)\s+)*trait\s")
TOKEN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
ROOTS = ("crates", "src", "tests", "examples", "bench", "benches")
LEDGER = "scripts/src_items_used_only_by_tests_ledger.txt"
CLASSES = {"test_only": "test-only", "unreferenced": "unreferenced"}
# At 439a62e76 main was red under `-D warnings` on exactly these three `pub(crate)`
# wrappers, whose only callers were `#[cfg(test)]` code. Re-measuring them is how a
# scan that finds nothing is told apart from a scan that has stopped detecting.
CONTROL_REVISION = "439a62e76c42edf5a8da59f658bf06f4204ddb11"
CONTROL_TEST_ONLY = (
    "crates/gam-sae/src/manifold/construction_exact_hessian.rs:fn:apply_exact_hessian_matrix_free",
    "crates/gam-sae/src/manifold/construction_exact_hessian.rs:fn:apply_exact_hessian_minus_b",
    "crates/gam-sae/src/manifold/penalties.rs:fn:decoder_prior_beta_hvp_pair",
)


def is_test_file(path):
    parts = path.split("/")
    if parts[0] in ("tests", "examples", "bench", "benches"):
        return True
    return len(parts) > 2 and parts[0] == "crates" and parts[2] in ("tests", "benches", "examples")


def is_source_file(path):
    parts = path.split("/")
    return parts[0] == "src" or (len(parts) > 2 and parts[0] == "crates" and parts[2] == "src")


def trait_member_lines(lines):
    """Line numbers inside a `trait` definition body or an `impl <Trait> for <Type>` block."""
    marked, stack, depth, pending = set(), [], 0, None
    for number, line in enumerate(lines, 1):
        if any(is_trait for _, is_trait in stack):
            marked.add(number)
        if TRAIT.match(line):
            pending = [depth, True]
        elif IMPL.match(line):
            pending = [depth, " for " in line]
        elif pending is not None and " for " in line:
            pending[1] = True
        for ch in line:
            if ch == "{":
                if pending is not None and depth == pending[0]:
                    stack.append((depth, pending[1]))
                    pending = None
                else:
                    stack.append((depth, False))
                depth += 1
            elif ch == "}":
                depth -= 1
                if stack and stack[-1][0] == depth:
                    stack.pop()
    return marked


def read_tree(root, revision):
    """{path: source} for every Rust blob under the scanned roots at `revision`."""
    listing = git(root, "ls-tree", "-rz", "--full-tree", revision, "--", *ROOTS)
    entries = []
    for entry in listing.split(b"\0"):
        if not entry:
            continue
        metadata, path = entry.split(b"\t", 1)
        if path.endswith(b".rs") and metadata.split()[1] == b"blob":
            entries.append((metadata.split()[2], path.decode()))
    if not entries:
        raise ValueError(f"{revision}: no Rust source files examined")
    objects = subprocess.run(["git", "-C", str(root), "cat-file", "--batch"],
                             input=b"".join(oid + b"\n" for oid, _ in entries),
                             stdout=subprocess.PIPE, check=True).stdout
    files, offset = {}, 0
    for expected, path in entries:
        end = objects.index(b"\n", offset)
        oid, kind, size = objects[offset:end].split()
        if oid != expected or kind != b"blob":
            raise ValueError(f"{revision}:{path}: Git object provenance mismatch")
        start = end + 1
        offset = start + int(size) + 1
        files[path] = objects[start:offset - 1].decode("utf-8", errors="replace")
    return files


def scan(files):
    """Classify every candidate definition in `files` ({path: source})."""
    test_references = {}
    files_naming = Counter()
    lines_naming = {}
    candidates = []
    for path in sorted(files):
        source = files[path]
        clean = strip_comments_and_literals(source)
        lines = clean.splitlines()
        if is_test_file(path):
            for line in lines:
                for token in TOKEN.findall(line):
                    test_references.setdefault(token, path)
            continue
        if not is_source_file(path):
            continue
        scope = test_lines(source, path, clean)
        members = trait_member_lines(lines)
        named = Counter()
        for number, line in enumerate(lines, 1):
            tokens = set(TOKEN.findall(line))
            if number in scope:
                for token in tokens:
                    test_references.setdefault(token, path)
                continue
            named.update(tokens)
            definition = None if number in members else DEFINITION.match(line)
            if (definition is None or definition["public"] or definition["name"].startswith("_")
                    or len(definition["name"]) <= 2 or definition["name"] in EXEMPT_NAMES):
                continue
            candidates.append((path, number, definition["kind"], definition["name"], bool(definition["scoped"])))
        files_naming.update(named.keys())
        lines_naming[path] = named
    report = {"test_only": [], "unreferenced": []}
    for path, number, kind, name, scoped in candidates:
        # The declaration's own line names it once; any other production line,
        # in this file or another, is a consumer.
        if files_naming[name] > 1 or lines_naming[path][name] > 1:
            continue
        identity = f"{path}:{kind}:{name}"
        if name in test_references:
            report["test_only"].append({"identity": identity, "line": number,
                                        "test_reference": test_references[name]})
        elif scoped:
            report["unreferenced"].append({"identity": identity, "line": number})
    return report


def ledger_lines(report):
    return sorted(f"{CLASSES[kind]} {entry['identity']}" for kind in CLASSES for entry in report[kind])


def read_ledger(path):
    """Recorded findings, one `<class> <identity>` per line, sorted and unique."""
    entries = [raw.split("#", 1)[0].strip() for raw in path.read_text(encoding="utf-8").splitlines()]
    entries = [entry for entry in entries if entry]
    if len(set(entries)) != len(entries):
        raise ValueError("the ledger records the same finding twice")
    if entries != sorted(entries):
        raise ValueError("the ledger is not sorted; keep it in one canonical order")
    return entries


def ratchet(report, recorded):
    """(regressions, stale): findings missing from the ledger, and ledger lines no longer found."""
    found, known = set(ledger_lines(report)), set(recorded)
    return sorted(found - known), sorted(known - found)


def positive_control(root):
    report = scan(read_tree(root, resolve(root, CONTROL_REVISION)))
    found = {entry["identity"] for entry in report["test_only"]}
    missing = [identity for identity in CONTROL_TEST_ONLY if identity not in found]
    if missing:
        raise ValueError(f"positive control at {CONTROL_REVISION[:9]} no longer reports: {missing}")
    return len(found)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--head", default="HEAD", help="immutable revision whose tree is scanned")
    parser.add_argument("--ledger", type=Path, help=f"ratchet against a committed ledger, normally {LEDGER}")
    parser.add_argument("--output", type=Path, help="write the report as JSON")
    parser.add_argument("--positive-control", action="store_true",
                        help=f"re-measure the {CONTROL_REVISION[:9]} incident and stop")
    args = parser.parse_args(argv)
    root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel").decode().strip())
    if args.positive_control:
        count = positive_control(root)
        print(f"positive control OK at {CONTROL_REVISION[:9]}: {len(CONTROL_TEST_ONLY)} known items detected "
              f"among {count} test-only findings")
        return 0
    head = resolve(root, args.head)
    report = scan(read_tree(root, head))
    if args.output is not None:
        args.output.write_text(json.dumps(dict(report, revision=head), indent=2, sort_keys=True) + "\n")
    print(f"{head[:9]}: {len(report['test_only'])} test-only, {len(report['unreferenced'])} unreferenced")
    if args.ledger is None:
        for line in ledger_lines(report):
            print(f"  {line}")
        return 1 if report["test_only"] or report["unreferenced"] else 0
    regressions, stale = ratchet(report, read_ledger(args.ledger))
    for line in regressions:
        print(f"  NEW   {line}", file=sys.stderr)
    for line in stale:
        print(f"  GONE  {line} (delete this line from {args.ledger.name} in the same commit)", file=sys.stderr)
    return 1 if regressions or stale else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, subprocess.CalledProcessError) as error:
        print(f"SRC ITEMS SCAN FAILED: {error}", file=sys.stderr)
        sys.exit(2)
