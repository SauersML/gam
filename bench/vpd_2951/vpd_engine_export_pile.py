"""Extend an engine export of the VPD 4L target (#2951) with the target's training text, so library
fits can run on up to 2^27 training tokens.

The new export OUT holds the source export's tensors (hard links), and tokens.f64 [R, 513]: first
the source's token rows unchanged (val-00000 rows 0..4096 for vpd4l_clean4096, so held-out rows
1024..1056 are still vpd4l_frontier32), then every row of whole train shards of
danbraunai/pile-uncopyrighted-tok-shuffled (train-00000-of-02021.parquet, train-00001-..., in shard
order) until at least MIN_TRAIN_ROWS train rows are held. Each row is the dataset's 513 stored
token ids widened to float64 (the layout gam_mpd::import::import_language_model reads; the model
reads tokens[:, :512], tokens[:, 1:] are the next-token labels).

export.json is the source's, with source.data the list of segments {file, sha256, rows, export_rows}
(the parquet file, its sha256, the rows [start, end) taken from it and where they sit in tokens.f64),
source.token_rows [0, R], source.extends (the source export and its export.json sha256), and new
tokens and row_ids entries (row_ids [1, R]: each row's index in its parquet file). The export is
built in OUT.partial and renamed to OUT once tokens.f64 has been read back and checked against
export.json as the importer reads it, so an incomplete export never loads.

usage: MPD_MEM_GIB=4 ~/mpd-data/venv/bin/python vpd_engine_export_pile.py SOURCE OUT MIN_TRAIN_ROWS
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

REPO = "danbraunai/pile-uncopyrighted-tok-shuffled"
REVISION = "6f141f6e4edc32fea842f2ece0d3477b15c3dc90"
CACHE = Path.home() / "mpd-data/hf"
WIDTH = 513
TRAIN_SHARDS = 2021

source, out, min_train = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
if out.exists():
    sys.exit(f"{out} exists")
partial = out.with_name(out.name + ".partial")
partial.mkdir()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def shard(name: str) -> tuple[Path, str]:
    """The dataset file data/NAME at REVISION (downloaded into the hub cache on first use) and its
    sha256, checked against the hub's record of the file."""
    path = Path(hf_hub_download(REPO, f"data/{name}", repo_type="dataset", revision=REVISION, cache_dir=CACHE))
    digest, (info,) = sha256(path), HfApi().get_paths_info(REPO, [f"data/{name}"], repo_type="dataset", revision=REVISION)
    if digest != info.lfs.sha256:
        sys.exit(f"{name}: sha256 {digest} is not the hub's {info.lfs.sha256}")
    return path, digest


def blocks(path: Path, start: int, end: int):
    """Rows [start, end) of a parquet file's input_ids, by row group, as int32 [n, WIDTH] blocks."""
    file = pq.ParquetFile(path)
    first = 0
    for g in range(file.num_row_groups):
        count = file.metadata.row_group(g).num_rows
        lo, hi = max(start, first), min(end, first + count)
        if lo < hi:
            col = file.read_row_group(g, columns=["input_ids"]).column("input_ids").combine_chunks().slice(lo - first, hi - lo)
            if col.null_count or not pc.all(pc.equal(pc.list_value_length(col), WIDTH)).as_py():
                sys.exit(f"{path.name}: a row of rows {lo}..{hi} is not {WIDTH} tokens")
            yield col.flatten().to_numpy().reshape(hi - lo, WIDTH)
        first += count


record = json.loads((source / "export.json").read_text())
files = record["files"]
vocab = record["config"]["vocab"]

# The source's token rows: its tokens.f64 as its export.json records it, and equal to the parquet
# rows it names, read as the train rows are read.
val_name = record["source"]["data"].split()[-1]
val_start, val_end = record["source"]["token_rows"]
if files["tokens"]["shape"] != [val_end - val_start, WIDTH] or sha256(source / "tokens.f64") != files["tokens"]["sha256"]:
    sys.exit("the source tokens.f64 is not its export.json's")
val_path, val_sha = shard(val_name)
val = np.fromfile(source / "tokens.f64", dtype="<f8").reshape(-1, WIDTH)
if not np.array_equal(val, np.concatenate(list(blocks(val_path, val_start, val_end))).astype("<f8")):
    sys.exit(f"the source tokens.f64 is not rows {val_start}..{val_end} of {val_name}")
segments = [{"file": f"{REPO} {val_name}", "sha256": val_sha, "rows": [val_start, val_end], "export_rows": [0, len(val)]}]
row_ids = [np.arange(val_start, val_end)]

for name in files:
    if name not in ("tokens", "row_ids"):
        os.link(source / f"{name}.f64", partial / f"{name}.f64")
        if sha256(partial / f"{name}.f64") != files[name]["sha256"]:
            sys.exit(f"{name}.f64 is not the source export.json's")

h = hashlib.sha256()
with open(partial / "tokens.f64", "wb") as fh:
    with open(source / "tokens.f64", "rb") as src:
        for chunk in iter(lambda: src.read(1 << 24), b""):
            fh.write(chunk)
            h.update(chunk)
    rows, train = len(val), 0
    for i in range(TRAIN_SHARDS):
        if train >= min_train:
            break
        name = f"train-{i:05d}-of-{TRAIN_SHARDS:05d}.parquet"
        path, digest = shard(name)
        n = pq.ParquetFile(path).metadata.num_rows
        for block in blocks(path, 0, n):
            if block.min() < 0 or block.max() >= vocab:
                sys.exit(f"{name}: a token id outside [0, {vocab})")
            wide = np.ascontiguousarray(block, dtype="<f8")
            fh.write(wide.data)
            h.update(wide.data)
        segments.append({"file": f"{REPO} {name}", "sha256": digest, "rows": [0, n], "export_rows": [rows, rows + n]})
        row_ids.append(np.arange(n))
        rows, train = rows + n, train + n
        print(json.dumps({"shard": name, "rows": n, "train_rows": train}), flush=True)
    if train < min_train:
        sys.exit(f"{train} train rows in all {TRAIN_SHARDS} shards")
    fh.flush()
    os.fsync(fh.fileno())

ids = np.concatenate(row_ids).astype("<f8").reshape(1, -1)
ids.tofile(partial / "row_ids.f64")
files["tokens"] = {"shape": [rows, WIDTH], "sha256": h.hexdigest()}
files["row_ids"] = {"shape": list(ids.shape), "sha256": sha256(partial / "row_ids.f64")}
record["source"]["data"] = segments
record["source"]["token_rows"] = [0, rows]
record["source"]["extends"] = {"export": str(source), "export_sha256": sha256(source / "export.json")}

# Read back as import_language_model does: every listed tensor is <name>.f64 of exactly
# rows x cols float64, and the token table holds integer ids below vocab. The source's rows are
# byte-identical to its tokens.f64.
for name, entry in files.items():
    r, c = entry["shape"] if len(entry["shape"]) == 2 else (1, entry["shape"][0])
    if (partial / f"{name}.f64").stat().st_size != r * c * 8:
        sys.exit(f"{name}.f64 does not hold {r} x {c} float64")
if sha256(partial / "tokens.f64") != files["tokens"]["sha256"]:
    sys.exit("tokens.f64 changed on disk")
table = np.memmap(partial / "tokens.f64", dtype="<f8", mode="r", shape=(rows, WIDTH))
if table[:len(val)].tobytes() != val.tobytes():
    sys.exit("the source rows are not byte-identical")
for lo in range(0, rows, 1 << 15):
    part = np.asarray(table[lo:lo + (1 << 15)])
    if not (np.all(part == np.floor(part)) and part.min() >= 0 and part.max() < vocab):
        sys.exit(f"rows {lo}..: a token is not an id below {vocab}")
del table

(partial / "export.json").write_text(json.dumps(record, indent=1))
partial.rename(out)
print(json.dumps({"out": str(out), "rows": rows, "train_rows": train, "segments": [s["file"].split()[-1] for s in segments],
                  "export_sha256": sha256(out / "export.json")}))
