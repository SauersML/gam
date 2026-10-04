"""VPD's rounded per-token sets (frontier/masks_vpd4l.npz, gate > 0) in the masked driver's numbering
(#2951), so `mpd_pieces_masked_2951 ... SETS` can start selection from them and count the held-out
ones into its context coder.

The driver numbers pieces site after site in its own site order (the library's site names, sorted),
each site's pieces in VPD's component order. Writes to OUT_DIR, raw little-endian int64:
  indptr.i64   [rows + 1]  CSR over the chosen sequences' positions, sequence after sequence
  indices.i64  [nnz]       active pieces, ascending within a position
  sites.txt    one "name pieces" line per site, in the driver's order

--rows picks and orders the npz's sequences, e.g. "0:32" (the frontier's eval rows) or "32:128,0:32";
the default is all of them in order. --export BASE NEW also writes a language-model export NEW whose
token rows are those sequences in that order (BASE's tensors linked, its export.json with the new
token table; BASE's per-row target tables are left out), so the sets number the export's sequences.

usage: vpd_sets_export.py OUT_DIR [--rows SPEC] [--export BASE NEW] [--masks NPZ] [--library DIR]
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("out", type=Path)
parser.add_argument("--rows")
parser.add_argument("--export", nargs=2, type=Path)
parser.add_argument("--masks", type=Path, default=Path.home() / "mpd-data/frontier/masks_vpd4l.npz")
parser.add_argument("--library", type=Path, default=Path.home() / "mpd-data/pieces/vpd4l_library")
args = parser.parse_args()
out, spec, masks, library = args.out, args.rows, args.masks, args.library

z = np.load(masks)
manifest = json.load(open(library / "manifest.json"))
vpd_names = [str(n) for n in z["site_names"]]
vpd_offsets = z["vpd_offsets"]
ours = sorted(manifest)
# Each VPD piece's index in the driver's numbering.
remap = np.empty(vpd_offsets[-1], dtype=np.int64)
offset = 0
for name in ours:
    k = vpd_names.index(manifest[name]["vpd"])
    pieces = int(vpd_offsets[k + 1] - vpd_offsets[k])
    if pieces != manifest[name]["pieces"]:
        raise SystemExit(f"{name}: {pieces} VPD pieces, {manifest[name]['pieces']} in the library")
    remap[vpd_offsets[k]:vpd_offsets[k + 1]] = offset + np.arange(pieces)
    offset += pieces

sequences, context = z["ids"].shape
rows = []
for part in (spec.split(",") if spec else [f"0:{sequences}"]):
    lo, hi = (int(x) for x in part.split(":"))
    rows.extend(range(lo, hi))
indptr = z["vpd_indptr"]
indices = z["vpd_indices"]
out_indptr = [np.zeros(1, dtype=np.int64)]
out_indices = []
total = 0
for s in rows:
    lo, hi = indptr[s * context], indptr[(s + 1) * context]
    counts = np.diff(indptr[s * context:(s + 1) * context + 1])
    flat = remap[indices[lo:hi]]
    # Ascending within each position.
    position = np.repeat(np.arange(context), counts)
    flat = flat[np.lexsort((flat, position))]
    out_indices.append(flat)
    out_indptr.append(total + np.cumsum(counts))
    total += int(hi - lo)
os.makedirs(out, exist_ok=True)
np.concatenate(out_indptr).astype("<i8").tofile(out / "indptr.i64")
np.concatenate(out_indices).astype("<i8").tofile(out / "indices.i64")
with open(out / "sites.txt", "w") as fh:
    for name in ours:
        fh.write(f"{name} {manifest[name]['pieces']}\n")
print(f"{len(rows)} sequences x {context} positions, {total} active ({total / (len(rows) * context):.1f} per position), {offset} pieces")
if args.export:
    base, new = args.export
    os.makedirs(new, exist_ok=True)
    record = json.load(open(base / "export.json"))
    # The weights carry over; BASE's per-row tables (row ids, target logits) belong to its rows.
    record["files"] = {k: v for k, v in record["files"].items() if k != "row_ids" and not k.startswith("logits_")}
    for name in record["files"]:
        if name != "tokens" and not (new / f"{name}.f64").exists():
            os.symlink((base / f"{name}.f64").resolve(), new / f"{name}.f64")
    tokens = z["ids"][rows].astype("<f8")
    tokens.tofile(new / "tokens.f64")
    record["files"]["tokens"] = {"shape": list(tokens.shape)}
    record["source"]["token_rows"] = f"rows {spec or 'all'} of {masks.name}"
    with open(new / "export.json", "w") as fh:
        json.dump(record, fh, indent=1)
    print(f"export {new}: {tokens.shape[0]} sequences")
