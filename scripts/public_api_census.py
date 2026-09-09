#!/usr/bin/env python3
"""Reject unreviewed removal of Rust library function declarations (#2829).

This is a source-graph instrument. A symbol table cannot observe an exported
generic that no inspected executable instantiates, and LTO may erase even
instantiated names. We inventory explicit public function declarations by
source path and name. A move is visible and must be reviewed like a deletion.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys

from test_census import tokens

LEDGER = "docs/public-api-census-changes.json"
CONTROL = "d484a091a4d133aec6f3ed980754b6cda151d41b"
CONTROL_REMOVALS = {"declarations": 1244, "distinct_names": 1192}
NAME = re.compile(r"[A-Za-z_][A-Za-z_0-9]*$")


def public_functions(source):
    """Yield explicitly public free-function and method names."""
    stream = iter(tokens(source))
    for token in stream:
        if token != "pub":
            continue
        token = next(stream, None)
        if token == "(":
            depth = 1
            for token in stream:
                if token == "(":
                    depth += 1
                elif token == ")":
                    depth -= 1
                    if depth == 0:
                        token = next(stream, None)
                        break
            if depth:
                raise ValueError("unterminated public visibility")
        while token in {"const", "async", "unsafe", "extern", "default"} or token == "LITERAL":
            token = next(stream, None)
        if token != "fn":
            continue
        name = next(stream, None)
        if name == "r" and next(stream, None) == "#":
            name = next(stream, None)
        # A macro template may contain `pub fn $name`; it is not itself an
        # explicit source declaration and is outside this gate's stated scope.
        if name is None or not NAME.fullmatch(name):
            continue
        yield name


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def resolve(root, revision):
    return git(root, "rev-parse", "--verify", revision + "^{commit}").decode().strip()


def census(root, revision, parsed=None):
    parsed = {} if parsed is None else parsed
    listing = git(root, "ls-tree", "-rz", "--full-tree", revision, "--", "crates", "src")
    files = []
    for entry in listing.split(b"\0"):
        if not entry:
            continue
        metadata, path = entry.split(b"\t", 1)
        if path.endswith(b".rs") and metadata.split()[1] == b"blob":
            files.append((metadata.split()[2], path.decode()))
    if not files:
        raise ValueError(f"{revision}: no Rust library source files examined")
    objects = subprocess.run(
        ["git", "-C", str(root), "cat-file", "--batch"],
        input=b"".join(oid + b"\n" for oid, _ in files), stdout=subprocess.PIPE, check=True,
    ).stdout
    offset = 0
    identities = Counter()
    names = Counter()
    for expected_oid, path in files:
        end = objects.index(b"\n", offset)
        oid, kind, size = objects[offset:end].split()
        if oid != expected_oid or kind != b"blob":
            raise ValueError(f"{revision}:{path}: Git object provenance mismatch")
        start = end + 1
        offset = start + int(size) + 1
        if oid not in parsed:
            source = objects[start:offset - 1].decode("utf-8")
            try:
                parsed[oid] = list(public_functions(source))
            except ValueError as error:
                raise ValueError(f"{revision}:{path}: {error}") from error
        for name in parsed[oid]:
            identities[f"{path}::{name}"] += 1
            names[name] += 1
    if not identities:
        raise ValueError(f"{revision}: empty public-function census")
    return {"revision": revision, "files": len(files), "functions": sum(identities.values()),
            "identities": identities, "names": names}


def difference(before, after):
    return dict(sorted((before["identities"] - after["identities"]).items()))


def check_change(before, after, entries):
    removed = difference(before, after)
    acknowledgements = [entry for entry in entries if entry.get("base") == before["revision"]]
    if not removed:
        if acknowledgements:
            raise ValueError("public API removal acknowledgement describes no observed loss")
        return removed
    if len(acknowledgements) != 1:
        raise ValueError("public functions were removed without exactly one explicit acknowledgement: "
                         + json.dumps(removed))
    entry = acknowledgements[0]
    if set(entry) != {"base", "removed", "reason", "evidence"}:
        raise ValueError("public API removal acknowledgement has missing or unknown fields")
    if entry["removed"] != removed:
        raise ValueError("public API removal acknowledgement does not match source identities")
    if not all(isinstance(entry[key], str) and entry[key].strip()
               for key in ("reason", "evidence")):
        raise ValueError("public API removal needs a semantic reason and retirement/replacement evidence")
    return removed


def positive_control(root):
    parsed = {}
    before = census(root, resolve(root, CONTROL + "^"), parsed)
    after = census(root, resolve(root, CONTROL), parsed)
    removed = difference(before, after)
    if not removed:
        raise ValueError("positive control did not detect the #2829 sweep")
    try:
        check_change(before, after, [])
    except ValueError:
        names = {identity.rsplit("::", 1)[1] for identity in removed}
        measured = {"declarations": sum(removed.values()), "distinct_names": len(names)}
        if measured != CONTROL_REMOVALS:
            raise ValueError(f"positive control drifted: {json.dumps(measured, sort_keys=True)}")
        return measured
    raise ValueError("positive control accepted the #2829 sweep without an acknowledgement")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="immutable pre-change commit")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--positive-control", action="store_true")
    args = parser.parse_args()
    root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel").decode().strip())
    if args.positive_control:
        measured = positive_control(root)
        print(f"Positive control: the #2829 sweep removed {measured['declarations']} path-sensitive "
              f"declarations ({measured['distinct_names']} distinct names), and the census refuses it.")
        return
    if not args.base:
        parser.error("--base is required unless --positive-control is given")
    parsed = {}
    before = census(root, resolve(root, args.base), parsed)
    after = census(root, resolve(root, args.head), parsed)
    ledger = json.loads(git(root, "show", after["revision"] + ":" + LEDGER))
    if not isinstance(ledger, list) or not all(isinstance(entry, dict) for entry in ledger):
        raise ValueError("public API census change ledger must be a list of objects")
    report = {"before": before, "after": after, "removed": difference(before, after)}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"{before['revision']}: {before['functions']} public function declarations in {before['files']} files")
    print(f"{after['revision']}: {after['functions']} public function declarations in {after['files']} files")
    check_change(before, after, ledger)
    print("Rust public-function source integrity verified; API behavior requires semantic tests.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, subprocess.CalledProcessError) as error:
        print(f"PUBLIC API CENSUS FAILED: {error}", file=sys.stderr)
        sys.exit(1)
