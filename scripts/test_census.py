#!/usr/bin/env python3
"""Compare Rust test declarations in immutable Git trees (#2818).

This is a source-integrity gate, not evidence that a test compiled or ran.
Comments and literals cannot satisfy a missing test identity. A removal needs
an explicit, commit-specific explanation in docs/test-census-changes.json.

Three assertions run, because each is blind to a loss the others see. The
comparison against the base names the exact identities that went missing, but
it only ever sees one step and its workspace totals let growth in one crate pay
for deletion in another. The floor in docs/test-census-floor.json is a
high-water mark per compilation unit and per issue number: it does not depend
on which base the gate was handed, and it does not net. ``--positive-control``
re-measures the sweep this gate exists for, because a census that has stopped
detecting anything is byte-identical to a census over a tree that lost nothing.
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
FLOOR = "docs/test-census-floor.json"
# The commit whose 2,290 deleted tests this gate was built to make visible.
CONTROL = {"head": "c0a21b5540ce76b76f62d880addf2612246ce1ee",
           "test_count_decrease": 2290, "removed_pin_names": 300, "units_losing_tests": 21}


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


def unit(path):
    """The compilation unit whose suite a source path belongs to.

    Workspace totals net a deletion in one crate against unrelated growth in
    another, which is exactly the shape of a sweep aimed at one subsystem.
    """
    head, _, rest = path.partition("/")
    return f"{head}/{rest.split('/')[0]}" if head == "crates" and rest else head


def issue_numbers(name):
    """Issue numbers a pinned test name carries in its trailing numeric suffix.

    A name survives renaming far less often than the bug it pins does, so the
    floor groups pins by issue: an issue's coverage may be rewritten, but the
    number of tests answering for it may not fall.
    """
    suffix = PIN.search(name)
    return () if suffix is None else tuple(part for part in suffix.group().split("_") if len(part) >= 3)


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
    units = Counter()
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
        units[unit(path)] += len(found)
        for name in found:
            locations.setdefault(name, []).append(path)
    pins = Counter({name: count for name, count in names.items() if PIN.search(name)})
    issues = Counter()
    for name, count in pins.items():
        for number in issue_numbers(name):
            issues[number] += count
    validate_population(revision, len(files), names, pins)
    return {"revision": revision, "files": len(files), "tests": sum(names.values()),
            "names": names, "pins": pins, "units": units, "issues": issues, "locations": locations}


def difference(before, after):
    return {"test_count_decrease": max(0, before["tests"] - after["tests"]),
            "removed_pins": dict(sorted((before["pins"] - after["pins"]).items())),
            "unit_test_decreases": dict(sorted((before["units"] - after["units"]).items()))}


def floor_from(measured):
    """The high-water mark a later tree must still clear."""
    return {"generated_from": measured["revision"],
            "units": dict(sorted(measured["units"].items())),
            "issues": dict(sorted(measured["issues"].items(), key=lambda item: int(item[0])))}


def check_floor(measured, floor):
    if set(floor) != {"generated_from", "units", "issues"}:
        raise ValueError("test census floor has missing or unknown fields")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0
           for section in ("units", "issues") for value in floor[section].values()):
        raise ValueError("test census floor holds an entry that is not a count")
    shortfall = {f"{section}/{key}": [minimum, measured[section][key]]
                 for section in ("units", "issues")
                 for key, minimum in floor[section].items()
                 if measured[section][key] < minimum}
    if shortfall:
        raise ValueError(
            f"test coverage fell below the recorded floor [minimum, measured]; regenerate {FLOOR} with "
            f"--update-floor only in the same change as the {LEDGER} entry that explains the loss: "
            + json.dumps(shortfall, sort_keys=True))
    return shortfall


def check_change(before, after, entries):
    delta = difference(before, after)
    acknowledgements = [entry for entry in entries if entry.get("base") == before["revision"]]
    losses = delta["test_count_decrease"] or delta["removed_pins"] or delta["unit_test_decreases"]
    if not losses:
        if acknowledgements:
            raise ValueError("removal acknowledgement describes no observed loss")
        return delta
    if len(acknowledgements) != 1:
        raise ValueError("test coverage decreased without exactly one explicit acknowledgement: " + json.dumps(delta))
    entry = acknowledgements[0]
    if set(entry) != {"base", "reason", "evidence", *delta}:
        raise ValueError("removal acknowledgement has missing or unknown fields")
    if any(entry[key] != delta[key] for key in delta):
        raise ValueError("removal acknowledgement does not match measured losses: " + json.dumps(delta))
    if not all(isinstance(entry[key], str) and entry[key].strip() for key in ("reason", "evidence")):
        raise ValueError("removal needs a semantic reason and replacement/retirement evidence")
    return delta


def positive_control(root):
    """Re-measure the #2818 sweep, so a census detecting nothing cannot pass.

    The expected numbers are the ones this gate must keep reproducing; a lexer
    or comparison change that moves them has to move them here too, in review.
    """
    parsed = {}
    before = census(root, resolve(root, CONTROL["head"] + "^"), parsed)
    after = census(root, resolve(root, CONTROL["head"]), parsed)
    delta = difference(before, after)
    measured = {"head": after["revision"],
                "test_count_decrease": delta["test_count_decrease"],
                "removed_pin_names": len(delta["removed_pins"]),
                "units_losing_tests": len(delta["unit_test_decreases"])}
    if measured != CONTROL:
        raise ValueError(f"positive control drifted from the measured #2818 sweep: {json.dumps(measured, sort_keys=True)}")
    try:
        check_change(before, after, [])
    except ValueError:
        return measured
    raise ValueError("positive control: the census accepted the #2818 sweep without an acknowledgement")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="immutable pre-change commit")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--output", type=Path, help="write measured identities and loss report as JSON")
    parser.add_argument("--update-floor", action="store_true", help=f"rewrite {FLOOR} from --head")
    parser.add_argument("--positive-control", action="store_true", help="re-measure the #2818 sweep and stop")
    args = parser.parse_args()
    root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel").decode().strip())
    if args.positive_control:
        print("Positive control at {head}: {test_count_decrease} tests, {removed_pin_names} pinned names and "
              "{units_losing_tests} units lost, and the census refuses it.".format(**positive_control(root)))
        return
    head = resolve(root, args.head)
    parsed = {}
    after = census(root, head, parsed)
    if args.update_floor:
        (root / FLOOR).write_text(json.dumps(floor_from(after), indent=2, sort_keys=False) + "\n")
        print(f"{FLOOR}: floor rewritten from {head}")
        return
    if not args.base:
        parser.error("--base is required unless --update-floor or --positive-control is given")
    before = census(root, resolve(root, args.base), parsed)
    floor = json.loads(git(root, "show", head + ":" + FLOOR))
    ledger = json.loads(git(root, "show", head + ":" + LEDGER))
    if not isinstance(ledger, list) or not all(isinstance(entry, dict) for entry in ledger):
        raise ValueError("test census change ledger must be a list of objects")
    report = {"before": before, "after": after, "change": difference(before, after), "floor": floor}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"{before['revision']}: {before['files']} Rust files, {before['tests']} tests, {len(before['pins'])} issue-pinned names")
    print(f"{after['revision']}: {after['files']} Rust files, {after['tests']} tests, {len(after['pins'])} issue-pinned names")
    print(f"{FLOOR} generated at {floor['generated_from']}: {len(floor['units'])} units, {len(floor['issues'])} issue numbers")
    check_floor(after, floor)
    check_change(before, after, ledger)
    print("Test source integrity verified; compilation and execution require their own verdicts.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, subprocess.CalledProcessError) as error:
        print(f"TEST CENSUS FAILED: {error}", file=sys.stderr)
        sys.exit(1)
