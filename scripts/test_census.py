#!/usr/bin/env python3
"""Compare Rust test declarations in immutable Git trees (#2818).

This is a source census, not evidence that a test compiled or ran. Comments and
literals cannot satisfy a missing test identity.

The comparison against the base reports the exact identities that went missing
and the compilation units that lost tests; it refuses only an empty denominator.
``--positive-control`` re-measures the sweep this census exists for, because a
census that has stopped detecting anything is byte-identical to a census over a
tree that lost nothing.
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
# The commit whose 2,290 deleted tests this census was built to make visible.
#
# `units_losing_tests` was 21 while the whole `tests` tree was one unit. Split by
# integration binary it is 38 — 20 `crates/*` and 18 `tests/*`, with the same
# 2,290 and the same 300 pin names either way, which is how you can see that the
# split changed the partition and nothing else. The number is larger because the
# sweep hit 18 integration binaries independently and the old unit reported that
# as one entry losing 546.
CONTROL = {"head": "c0a21b5540ce76b76f62d880addf2612246ce1ee",
           "test_count_decrease": 2290, "removed_pin_names": 300, "units_losing_tests": 38}


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


CLOSING = {"(": ")", "[": "]", "{": "}"}
IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


class Cursor:
    """An iterator over one file's tokens that can also look ahead."""

    def __init__(self, items):
        self.items, self.index = list(items), 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.items):
            raise StopIteration
        self.index += 1
        return self.items[self.index - 1]

    def peek(self, offset=0):
        at = self.index + offset
        return self.items[at] if at < len(self.items) else None

    def group(self):
        """Consume the balanced delimited group at the cursor and return its inner tokens."""
        opening = next(self, None)
        if opening not in CLOSING:
            raise ValueError("expected a delimited group")
        stack, inner = [CLOSING[opening]], []
        for token in self:
            if token in CLOSING:
                stack.append(CLOSING[token])
            elif token in (")", "]", "}"):
                if token != stack.pop():
                    raise ValueError("mismatched Rust delimiters")
                if not stack:
                    return inner
            inner.append(token)
        raise ValueError("unterminated Rust delimited group")


def elements(inner):
    """Split a token run into its top-level elements: tokens, and delimited groups as one element each."""
    cursor, found = Cursor(inner), []
    while cursor.peek() is not None:
        if cursor.peek() in CLOSING:
            delimiter = cursor.peek()
            found.append((delimiter, cursor.group()))
        else:
            found.append(next(cursor))
    return found


def test_attributes_before_fn(inner):
    """The name tokens after `fn` for every test attribute in a token run (`$` for a macro metavariable)."""
    names, cursor, pending = [], Cursor(inner), False
    for token in cursor:
        if token == "#" and cursor.peek() == "[":
            next(cursor)
            depth, taken = 1, []
            for part in cursor:
                depth += part == "["
                depth -= part == "]"
                if depth == 0:
                    break
                taken.append(part)
            pending |= taken == ["test"] or (taken[:2] == ["cfg_attr", "("] and
                                             any(taken[i:i + 2] == [",", "test"] for i in range(len(taken))))
        elif pending and token == "fn":
            names.append(cursor.peek())
            pending = False
    return names


class GeneratedTests:
    """A `macro_rules!` whose one rule repeats a named `#[test] fn $name` per invocation item.

    A generated test has no name in the source until the macro is invoked, so the census names it from the
    invocation: each repetition item contributes the identifier its `$name:ident` fragment binds (#3241/#3242).
    Only the shape that can be read without evaluating Rust is accepted: one rule, whose matcher is one repetition
    `$( $name:ident <tokens and delimited groups> ) <sep>? <*|+>` with fragments other than `$name` only inside
    groups, and whose transcriber is one repetition holding exactly one test function named `$name`. Anything else
    that generates a test refuses, as an unnamed test attribute does.
    """

    def __init__(self, name, body):
        rules = [rule for rule in split_rules(body) if rule]
        if len(rules) != 1:
            raise ValueError(f"test-generating macro {name}! has {len(rules)} rules; the census reads one")
        rule = elements(rules[0])
        if len(rule) != 4 or rule[1:3] != ["=", ">"] or not isinstance(rule[0], tuple) or not isinstance(rule[3], tuple):
            raise ValueError(f"test-generating macro {name}! has a rule the census cannot read")
        self.name = name
        self.pattern, self.separator, self.at_least_one = repetition(name, rule[0][1])
        head = self.pattern[0] if self.pattern else None
        if not (isinstance(head, tuple) and head[0] == "$" and head[2] == "ident"):
            raise ValueError(f"test-generating macro {name}!'s repetition does not start with $name:ident")
        if any(isinstance(item, tuple) and item[0] == "$" for item in self.pattern[1:]):
            raise ValueError(f"test-generating macro {name}! binds a top-level fragment besides the test name")
        transcribed, _, _ = repetition(name, rule[3][1], transcriber=True)
        generated = test_attributes_before_fn(transcribed)
        fragment = head[1]
        if len(generated) != 1 or generated[0] != "$" or not contains(transcribed, ["fn", "$", fragment]):
            raise ValueError(f"test-generating macro {name}! does not generate one test named ${fragment}")

    def names(self, inner):
        """The test names one invocation generates, or a refusal when it does not match the pattern."""
        items, found, position = elements(inner), [], 0
        while position < len(items):
            if found and self.separator is not None:
                if items[position] != self.separator:
                    raise ValueError(f"invocation of {self.name}! does not match its pattern")
                position += 1
            candidate = items[position] if position < len(items) else None
            if not isinstance(candidate, str) or not IDENT.fullmatch(candidate):
                raise ValueError(f"invocation of {self.name}! does not name a test")
            found.append(candidate)
            position += 1
            for expected in self.pattern[1:]:
                actual = items[position] if position < len(items) else None
                if isinstance(expected, tuple):
                    if not (isinstance(actual, tuple) and actual[0] == expected[0]):
                        raise ValueError(f"invocation of {self.name}! does not match its pattern")
                elif actual != expected:
                    raise ValueError(f"invocation of {self.name}! does not match its pattern")
                position += 1
        if self.at_least_one and not found:
            raise ValueError(f"invocation of {self.name}! repeats `+` with no item")
        return found


def split_rules(body):
    rules, current, depth = [], [], 0
    for token in body:
        depth += token in CLOSING
        depth -= token in (")", "]", "}")
        if token == ";" and depth == 0:
            rules.append(current)
            current = []
        else:
            current.append(token)
    rules.append(current)
    return rules


def repetition(name, inner, transcriber=False):
    """The one `$( ... ) sep? op` a matcher (or transcriber) is made of: its elements, separator and `+`."""
    cursor = Cursor(inner)
    if cursor.peek() != "$" or cursor.peek(1) != "(":
        raise ValueError(f"test-generating macro {name}! is not one repetition")
    next(cursor)
    body = cursor.group()
    rest = cursor.items[cursor.index:]
    if rest and rest[-1] in ("*", "+") and len(rest) <= 2:
        separator = rest[0] if len(rest) == 2 else None
        if transcriber:
            return body, separator, rest[-1] == "+"
        pattern, walker = [], Cursor(body)
        while walker.peek() is not None:
            if walker.peek() == "$":
                next(walker)
                if walker.peek() == "(":
                    raise ValueError(f"test-generating macro {name}! nests a repetition")
                fragment = next(walker, None)
                if next(walker, None) != ":":
                    raise ValueError(f"test-generating macro {name}! has an untyped fragment")
                pattern.append(("$", fragment, next(walker, None)))
            elif walker.peek() in CLOSING:
                delimiter = walker.peek()
                pattern.append((delimiter, walker.group()))
            else:
                pattern.append(next(walker))
        return pattern, separator, rest[-1] == "+"
    raise ValueError(f"test-generating macro {name}! is not one repetition")


def contains(run, needle):
    return any(run[i:i + len(needle)] == needle for i in range(len(run) - len(needle) + 1))


def rust_test_names(source):
    stream = Cursor(tokens(source))
    generators = {}
    pending_test = False
    for token in stream:
        if (not pending_test and token == "macro_rules" and stream.peek() == "!" and stream.peek(1)
                and IDENT.fullmatch(stream.peek(1)) and stream.peek(2) in CLOSING):
            # A macro whose body writes `#[test] fn $name` is a test generator: its tests are named by its
            # invocations below. Any other macro body is read in place, as before.
            start = stream.index
            name = stream.peek(1)
            next(stream), next(stream)
            body = stream.group()
            if "$" in test_attributes_before_fn(body):
                generators[name] = GeneratedTests(name, body)
            else:
                stream.index = start
            continue
        if not pending_test and token in generators and stream.peek() == "!" and stream.peek(1) in CLOSING:
            next(stream)
            yield from generators[token].names(stream.group())
            continue
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

    The same netting survives inside any unit that is itself an aggregate, and
    the top-level `tests` tree was the largest one: 2,619 tests over 1,069 files
    in a single entry, more than any crate. So it is split by its first path
    component too. That boundary is not arbitrary — each
    `tests/<name>/main.rs` is the crate root of its own integration binary
    (`gam::regressions`, `gam::sae`, `gam::quality`, ... — 18 of them), which is
    the unit a failure is already attributed to; the three subtrees with no
    `main.rs` (`common`, `src_modules`, `test_support`) are shared module trees
    and get their own entries for the same reason. `tests/<name>.rs` is the
    single-file spelling of the same binary, so the extension is dropped and
    both spellings land on one unit.
    """
    head, _, rest = path.partition("/")
    if not rest or head not in ("crates", "tests"):
        return head
    first = rest.split("/")[0]
    if head == "tests" and first.endswith(".rs"):
        first = first[: -len(".rs")]
    return f"{head}/{first}"


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
                parsed[oid] = list(rust_test_names(source))
            except ValueError as error:
                raise ValueError(f"{revision}:{path}: {error}") from error
        found = parsed[oid]
        names.update(found)
        units[unit(path)] += len(found)
        for name in found:
            locations.setdefault(name, []).append(path)
    pins = Counter({name: count for name, count in names.items() if PIN.search(name)})
    validate_population(revision, len(files), names, pins)
    return {"revision": revision, "files": len(files), "tests": sum(names.values()),
            "names": names, "pins": pins, "units": units, "locations": locations}


def difference(before, after):
    return {"test_count_decrease": max(0, before["tests"] - after["tests"]),
            "removed_pins": dict(sorted((before["pins"] - after["pins"]).items())),
            "unit_test_decreases": dict(sorted((before["units"] - after["units"]).items()))}


def positive_control(root):
    """Re-measure the #2818 sweep, so a census detecting nothing cannot pass.

    The expected numbers are the ones this census must keep reproducing; a lexer
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
    return measured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="immutable pre-change commit")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--output", type=Path, help="write measured identities and loss report as JSON")
    parser.add_argument("--positive-control", action="store_true", help="re-measure the #2818 sweep and stop")
    args = parser.parse_args()
    root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel").decode().strip())
    if args.positive_control:
        print("Positive control at {head}: {test_count_decrease} tests, {removed_pin_names} pinned names and "
              "{units_losing_tests} units lost.".format(**positive_control(root)))
        return
    if not args.base:
        parser.error("--base is required unless --positive-control is given")
    parsed = {}
    after = census(root, resolve(root, args.head), parsed)
    before = census(root, resolve(root, args.base), parsed)
    report = {"before": before, "after": after, "change": difference(before, after)}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"{before['revision']}: {before['files']} Rust files, {before['tests']} tests, {len(before['pins'])} issue-pinned names")
    print(f"{after['revision']}: {after['files']} Rust files, {after['tests']} tests, {len(after['pins'])} issue-pinned names")
    print("Test source census complete; compilation and execution require their own verdicts.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, subprocess.CalledProcessError) as error:
        print(f"TEST CENSUS FAILED: {error}", file=sys.stderr)
        sys.exit(1)
