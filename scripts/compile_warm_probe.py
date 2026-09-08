#!/usr/bin/env python3
"""Compile a Rust diagnostic against one existing Cargo dependency graph.

Run on the build machine, from the repository's toolchain directory, while the
target directory is idle. This does not build dependencies or substitute a
different feature/profile variant. Source and library hashes verify the inputs
again after compilation; Cargo must finish before a probe uses its artifacts.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--target-dir", required=True, type=Path)
    parser.add_argument("--anchor", required=True,
                        help="existing .fingerprint directory name, e.g. gam-solve-<hash>")
    parser.add_argument("--extern", action="append", default=[], dest="externs",
                        help="additional direct dependency of the anchor to expose")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    profile = args.target_dir.resolve(strict=True) / "debug"
    fingerprints = profile / ".fingerprint"
    anchor_dir = fingerprints / args.anchor
    manifests = list(anchor_dir.glob("lib-*.json"))
    if len(manifests) != 1:
        parser.error(f"anchor must identify exactly one library manifest: {manifests}")
    manifest = manifests[0]
    anchor_name = manifest.stem.removeprefix("lib-")
    metadata = json.loads(manifest.read_text())
    dependencies = {entry[1]: entry[3] for entry in metadata["deps"]}

    def artifact(directory, name):
        suffix = directory.name.rsplit("-", 1)[1]
        return (profile / "deps" / f"lib{name}-{suffix}.rlib").resolve(strict=True)

    libraries = {anchor_name: artifact(anchor_dir, anchor_name)}
    for name in args.externs:
        if name not in dependencies:
            parser.error(f"{name} is not a direct dependency of {args.anchor}")
        matches = []
        for stamp in fingerprints.glob(f"*/lib-{name}"):
            digest = int.from_bytes(bytes.fromhex(stamp.read_text().strip()), "little")
            if digest == dependencies[name]:
                matches.append(stamp.parent)
        if len(matches) != 1:
            parser.error(f"{name} must have one matching dependency fingerprint: {matches}")
        libraries[name] = artifact(matches[0], name)

    source = args.source.resolve(strict=True)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = ["rustc", "--edition=2024", "-C", "opt-level=2", "-C", "codegen-units=4",
               "-L", f"dependency={profile / 'deps'}", str(source), "-o", str(output)]
    if args.test:
        command.append("--test")
    hashes = {}
    for name, library in libraries.items():
        command.extend(["--extern", f"{name}={library}"])
        with library.open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    print(json.dumps({"anchor": args.anchor, "libraries": {k: str(v) for k, v in libraries.items()},
                      "library_sha256": hashes, "source_sha256": source_hash,
                      "command": command}, indent=2), flush=True)
    subprocess.run(["rustc", "--version"], check=True)
    started = time.monotonic()
    subprocess.run(command, check=True)
    elapsed = time.monotonic() - started
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError(f"source changed during compilation: {source}")
    for name, library in libraries.items():
        with library.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != hashes[name]:
                raise RuntimeError(f"dependency changed during compilation: {library}")
    with output.open("rb") as stream:
        binary_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    print(json.dumps({"compile_seconds": elapsed, "binary_sha256": binary_hash}), flush=True)


if __name__ == "__main__":
    main()
