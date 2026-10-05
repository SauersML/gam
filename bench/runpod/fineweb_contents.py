"""The contents of the Qwen3 FineWeb release (#2951), for pods that rebuild its shards themselves.

Reads ~/mpd-data/qwen3_fineweb/release/: every tarball named in SHA256SUMS is checked against it,
then read as a stream (zstd --long=31) and every member's sha256 and size taken. Writes
CONTENTS.json there: per shard directory (train_shard00, heldout_f284, ...) the FineWeb source file,
its index and the document count its meta.json records, and the sha256 and size of each of its
files. A directory beside the release on the Mac (train, heldout) whose meta.json names the same
source file and document count, and whose files all have a release shard's sha256s, is the same
shard under another name and is listed as such.

usage: python3 fineweb_contents.py
"""

import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path.home() / "mpd-data/qwen3_fineweb"
RELEASE = ROOT / "release"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    shards = {}
    for line in (RELEASE / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        if sha256_file(RELEASE / name) != digest:
            raise SystemExit(f"{name}: not the sha256 SHA256SUMS records")
        unzip = subprocess.Popen(["zstd", "-dc", "--long=31", str(RELEASE / name)], stdout=subprocess.PIPE)
        files, sizes, meta = {}, {}, None
        with tarfile.open(fileobj=unzip.stdout, mode="r|") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                data = tar.extractfile(member)
                h = hashlib.sha256()
                kept = b""
                for chunk in iter(lambda: data.read(1 << 22), b""):
                    h.update(chunk)
                    if member.name.endswith("/meta.json"):
                        kept += chunk
                directory, file = member.name.split("/")[-2:]
                files[file] = h.hexdigest()
                sizes[file] = member.size
                if file == "meta.json":
                    meta = json.loads(kept)
        if unzip.wait() != 0:
            raise SystemExit(f"{name}: zstd failed")
        shards[directory] = {
            "tarball": name,
            "source_file": meta["source"]["file"],
            "file_index": meta["source"]["file_index"],
            "documents": meta["documents"],
            "files": files,
            "sizes": sizes,
        }
    for local in sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name != "release" and p.name not in shards):
        meta_path = local / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        for name, shard in shards.items():
            same_source = (meta["source"]["file_index"], meta["documents"]) == (shard["file_index"], shard["documents"])
            recorded = {entry["path"]: entry["sha256"] for entry in meta["files"].values()}
            if same_source and all(shard["files"].get(f) == d for f, d in recorded.items()):
                shards[local.name] = {"same_as": name, **{k: v for k, v in shard.items() if k != "tarball"}}
                break
    (RELEASE / "CONTENTS.json").write_text(json.dumps(shards, indent=1))
    print(json.dumps({name: {"file_index": s["file_index"], "documents": s["documents"], "files": len(s["files"]), "same_as": s.get("same_as")} for name, s in shards.items()}))


if __name__ == "__main__":
    main()
