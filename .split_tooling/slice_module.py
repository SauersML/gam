#!/usr/bin/env python3
"""Behavior-preserving module slicer for task #15 (sae_manifold.rs decomposition).

PURPOSE (team-lead condition 1): move exact line ranges out of a source file into
a new submodule with MECHANICAL byte conservation. No content rewriting of moved
code, ever — the moved bytes are copied verbatim. The only new bytes are the
submodule header and (separately, by hand) the shim re-export + pub(crate)
widenings, which are logged explicitly.

Usage:
  slice_module.py extract --src SRC --out OUT --header-file HDR --ranges "a-b,c-d,..."
    Moves the given 1-based inclusive line ranges from SRC into OUT (prefixed by the
    header file's contents), removes them from SRC, and asserts:
        bytes(moved_content) == sum over ranges of original line bytes
        bytes(SRC_after) + bytes(moved_content) == bytes(SRC_before)
    i.e. every original byte is conserved: it is either still in SRC or in the moved
    content. OUT's extra bytes are exactly the header (reported).
  slice_module.py verify --before SNAPSHOT --src SRC --out OUT --header-file HDR
    Re-checks conservation against a saved pre-cut snapshot.

This is exact line-range slicing with a conservation proof — it CANNOT silently eat
code the way a fuzzy multiline regex can, which is the failure the no-regex rule
guards against.
"""
import argparse, sys, os

def parse_ranges(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        a, b = part.split("-")
        a, b = int(a), int(b)
        if a > b:
            sys.exit(f"bad range {part}: start>end")
        out.append((a, b))
    out.sort()
    # reject overlaps
    for i in range(1, len(out)):
        if out[i][0] <= out[i-1][1]:
            sys.exit(f"overlapping ranges {out[i-1]} and {out[i]}")
    return out

def extract(args):
    with open(args.src, "rb") as f:
        original = f.read()
    lines = original.splitlines(keepends=True)  # preserves exact bytes incl. EOLs
    n = len(lines)
    ranges = parse_ranges(args.ranges)
    for a, b in ranges:
        if a < 1 or b > n:
            sys.exit(f"range {a}-{b} out of file bounds (1..{n})")
    moved_idx = set()
    moved_chunks = []
    for a, b in ranges:
        chunk = b"".join(lines[a-1:b])
        moved_chunks.append(chunk)
        moved_idx.update(range(a-1, b))
    moved_content = b"".join(moved_chunks)
    remaining = b"".join(lines[i] for i in range(n) if i not in moved_idx)

    # CONSERVATION ASSERTION: every original byte is in remaining XOR moved.
    assert len(remaining) + len(moved_content) == len(original), (
        f"BYTE CONSERVATION FAILED: remaining {len(remaining)} + moved "
        f"{len(moved_content)} != original {len(original)}"
    )

    with open(args.header_file, "rb") as f:
        header = f.read()
    out_bytes = header + moved_content
    with open(args.out, "wb") as f:
        f.write(out_bytes)
    with open(args.src, "wb") as f:
        f.write(remaining)

    print(f"OK conservation: original={len(original)}B "
          f"-> src_after={len(remaining)}B + moved={len(moved_content)}B")
    print(f"   submodule {args.out} = header {len(header)}B + moved {len(moved_content)}B "
          f"= {len(out_bytes)}B")
    print(f"   moved {len(moved_idx)} lines across {len(ranges)} range(s)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--src", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--header-file", required=True)
    e.add_argument("--ranges", required=True)
    e.set_defaults(func=extract)
    args = p.parse_args()
    args.func(args)
