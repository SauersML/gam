#!/usr/bin/env python3
"""Compare Rust test declarations in immutable Git trees (#2818).

This is a source-integrity gate, not evidence that a test compiled or ran.
Comments and literals cannot satisfy a missing test identity. A removal needs
an explicit, commit-specific explanation in docs/test-census-changes.json.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys


PIN = re.compile(r"_\d+(?:_\d+)*$")
TOKEN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|[^\s]")
LITERAL = re.compile(r'''(?:b|c)?"(?:\\.|[^"\\])*"|'(?:\\(?:u\{[0-9a-fA-F]+\}|x[0-9a-fA-F]{2}|.)|[^'\\\n])' ''', re.S | re.X)
RAW = re.compile(r'(?:b|c)?r(#{0,255})"')
LEDGER = "docs/test-census-changes.json"


def tokens(source):
    """Lex Rust comments and literals before inspecting attributes/items.

    Nested block comments, raw/byte/C strings and character literals are
    consumed whole. Lifetimes remain tokens. Literal contents are never code.
    """
    position = 0
    while position < len(source):
        if source[position].isspace():
            position += 1
            continue
        if source.startswith("//", position):
            newline = source.find("\n", position)
            position = len(source) if newline < 0 else newline + 1
            continue
        if source.startswith("/*", position):
            depth = 1
            position += 2
            while depth:
                opening = source.find("/*", position)
                closing = source.find("*/", position)
                if closing < 0:
                    raise ValueError("unterminated Rust block comment")
                if 0 <= opening < closing:
                    depth += 1
                    position = opening + 2
                else:
                    depth -= 1
                    position = closing + 2
            continue
        raw = RAW.match(source, position)
        if raw:
            end_marker = '"' + raw.group(1)
            end = source.find(end_marker, raw.end())
            if end < 0:
                raise ValueError("unterminated Rust raw string")
            position = end + len(end_marker)
            yield "LITERAL"
            continue
        literal = LITERAL.match(source, position)
        if literal:
            position = literal.end()
            yield "LITERAL"
            continue
        token = TOKEN.match(source, position)
        yield token.group()
        position = token.end()


def test_names(source):
    stream = iter(tokens(source))
    pending_test = False
    for token in stream:
        if token == "#":
            opening = next(stream, None)
            if opening == "!":
                opening = next(stream, None)
            if opening != "[":
                continue
            attribute = []
            depth = 1
            for part in stream:
                if part == "[":
                    depth += 1
                elif part == "]":
                    depth -= 1
                    if depth == 0:
                        break
                attribute.append(part)
            if depth:
                raise ValueError("unterminated Rust attribute")
            # Standard #[test], including whitespace, is the documented census
            # unit. Conditional attributes are counted conservatively as source
            # declarations; feature/target execution is a separate gate.
            is_test = attribute == ["test"] or (
                attribute[:2] == ["cfg_attr", "("]
                and any(attribute[i:i + 2] == [",", "test"] for i in range(len(attribute)))
            )
            if is_test and pending_test:
                raise ValueError("duplicate test attributes before one function")
            pending_test |= is_test
        elif pending_test and token == "fn":
            name = next(stream, None)
            if name == "r" and next(stream, None) == "#":
                name = next(stream, None)
            if name is None or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
                raise ValueError("test attribute has no named function")
            yield name
            pending_test = False
        elif pending_test and token in (";", "{", "}", "mod", "struct", "enum", "impl"):
            raise ValueError("test attribute is not attached to a function")
    if pending_test:
        raise ValueError("test attribute has no function")


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def resolve(root, revision):
    return git(root, "rev-parse", "--verify", revision + "^{commit}").decode().strip()


def validate_population(revision, files, names, pins):
    if not files or not names or not pins:
        raise ValueError(f"{revision}: empty census denominator ({files} Rust files, {sum(names.values())} tests, {len(pins)} pins)")


def census(root, revision, parsed=None):
    if parsed is None:
        parsed = {}
    listing = git(root, "ls-tree", "-rz", "--full-tree", revision, "--", "crates", "tests", "src")
    files = []
    for entry in listing.split(b"\0"):
        if not entry:
            continue
        metadata, path = entry.split(b"\t", 1)
        if path.endswith(b".rs") and metadata.split()[1] == b"blob":
            files.append((metadata.split()[2], path.decode()))
    if not files:
        raise ValueError(f"{revision}: no Rust source files examined")
    objects = subprocess.run(
        ["git", "-C", str(root), "cat-file", "--batch"],
        input=b"".join(oid + b"\n" for oid, _ in files),
        stdout=subprocess.PIPE, check=True,
    ).stdout
    offset = 0
    names = Counter()
    locations = {}
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
                parsed[oid] = list(test_names(source))
            except ValueError as error:
                raise ValueError(f"{revision}:{path}: {error}") from error
        found = parsed[oid]
        names.update(found)
        for name in found:
            locations.setdefault(name, []).append(path)
    pins = Counter({name: count for name, count in names.items() if PIN.search(name)})
    validate_population(revision, len(files), names, pins)
    return {"revision": revision, "files": len(files), "tests": sum(names.values()),
            "names": names, "pins": pins, "locations": locations}


def difference(before, after):
    return {"test_count_decrease": max(0, before["tests"] - after["tests"]),
            "removed_pins": dict(sorted((before["pins"] - after["pins"]).items()))}


def check_change(before, after, entries):
    delta = difference(before, after)
    acknowledgements = [entry for entry in entries if entry.get("base") == before["revision"]]
    losses = delta["test_count_decrease"] or delta["removed_pins"]
    if not losses:
        if acknowledgements:
            raise ValueError("removal acknowledgement describes no observed loss")
        return delta
    if len(acknowledgements) != 1:
        raise ValueError("test coverage decreased without exactly one explicit acknowledgement: " + json.dumps(delta))
    entry = acknowledgements[0]
    if set(entry) != {"base", "test_count_decrease", "removed_pins", "reason", "evidence"}:
        raise ValueError("removal acknowledgement has missing or unknown fields")
    if any(entry[key] != delta[key] for key in delta):
        raise ValueError("removal acknowledgement does not match measured losses: " + json.dumps(delta))
    if not all(isinstance(entry[key], str) and entry[key].strip() for key in ("reason", "evidence")):
        raise ValueError("removal needs a semantic reason and replacement/retirement evidence")
    return delta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="immutable pre-change commit")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--output", type=Path, help="write measured identities and loss report as JSON")
    args = parser.parse_args()
    root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel").decode().strip())
    base, head = resolve(root, args.base), resolve(root, args.head)
    parsed = {}
    before, after = census(root, base, parsed), census(root, head, parsed)
    ledger = json.loads(git(root, "show", head + ":" + LEDGER))
    if not isinstance(ledger, list) or not all(isinstance(entry, dict) for entry in ledger):
        raise ValueError("test census change ledger must be a list of objects")
    report = {"before": before, "after": after, "change": difference(before, after)}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"{base}: {before['files']} Rust files, {before['tests']} tests, {len(before['pins'])} issue-pinned names")
    print(f"{head}: {after['files']} Rust files, {after['tests']} tests, {len(after['pins'])} issue-pinned names")
    check_change(before, after, ledger)
    print("Test source integrity verified; compilation and execution require their own verdicts.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, subprocess.CalledProcessError) as error:
        print(f"TEST CENSUS FAILED: {error}", file=sys.stderr)
        sys.exit(1)
