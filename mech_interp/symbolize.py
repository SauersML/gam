"""Resolve hex-offset frame tokens to demangled Rust symbols via `atos`.

samply emits unsymbolicated frames as `0x<hex_offset>` (offset relative to
binary's preferred load address 0x100000000 on macOS). We batch-feed all
hex-token frames through `atos -l 0x100000000` to recover names.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

BINARY = "/Users/user/gam/target/release/deps/cell_moment_dedup_biobank_shape-379f15f35e09aaa4"
LOAD = "0x100000000"

_HEX_RE = re.compile(r"^0x[0-9a-f]+$")


def trim_symbol(s: str) -> str:
    # Unresolved PLT/import stubs: collapse to a generic stub label.
    m = re.match(r"^(0x[0-9a-f]+)\s*\(in [^)]+\)\s*$", s)
    if m:
        return f"<stub:{m.group(1)}>"
    # Drop trailing `(in <binary>) + N` annotation if present.
    s = re.sub(r"\s*\(in [^)]+\)\s*\+\s*\d+\s*$", "", s)
    s = re.sub(r"\s*\+\s*\d+\s*$", "", s)
    s = re.sub(r"::h[0-9a-f]{8,}$", "", s)
    # Decompress mangled `_$u7b$$u7b$closure$u7d$$u7d$` etc. for readability.
    s = s.replace("_$u7b$$u7b$", "{").replace("$u7d$$u7d$", "}")
    s = s.replace("$u7b$", "{").replace("$u7d$", "}")
    s = s.replace("$LT$", "<").replace("$GT$", ">")
    s = s.replace("$RF$", "&").replace("$BP$", "*")
    s = s.replace("$u20$", " ").replace("$u27$", "'")
    s = re.sub(r"::h[0-9a-f]{8,}", "", s)
    return s.strip()


def resolve(offsets: list[str]) -> dict[str, str]:
    if not offsets:
        return {}
    addrs = []
    for off in offsets:
        v = int(off, 16) + int(LOAD, 16)
        addrs.append(f"0x{v:x}")
    proc = subprocess.run(
        ["atos", "-o", BINARY, "-l", LOAD, *addrs],
        capture_output=True, text=True, check=True,
    )
    lines = proc.stdout.strip().split("\n")
    out: dict[str, str] = {}
    for off, line in zip(offsets, lines):
        sym = trim_symbol(line)
        if sym == off or not sym:
            out[off] = off
        else:
            out[off] = sym
    return out


def build_symbolicated_vocab(vocab_path: Path, out_path: Path) -> dict:
    vocab = json.loads(vocab_path.read_text())
    hex_frames = [k for k in vocab if _HEX_RE.match(k)]
    print(f"resolving {len(hex_frames)} hex frames via atos...")
    sym_map = resolve(hex_frames)
    n_resolved = sum(1 for k, v in sym_map.items() if v != k)
    print(f"resolved: {n_resolved}/{len(hex_frames)}")

    # Write id → symbolicated_label map preserving every ID.
    id_to_label = {}
    for name, tid in vocab.items():
        id_to_label[str(tid)] = sym_map.get(name, name)
    out_path.write_text(json.dumps(id_to_label, indent=1))
    return id_to_label


if __name__ == "__main__":
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/tok")
    build_symbolicated_vocab(out_dir / "vocab.json", out_dir / "vocab_sym.json")
