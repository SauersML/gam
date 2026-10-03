"""Write the Pile val rows that hold firings of h.2.mlp.down_proj:2359 (scans of rows 0-28671) to e4_rows.npz,
so mpd_vpd_edit_replication_2951.py never loads the parquet row group (keeps it inside its memory budget)."""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "vpd_2951"))
from vpd_model import VAL_PARQUET  # noqa: E402

VD = Path.home() / "mpd-data/vpd"
rows = set()
for scan in ("scan_0_4096.pt", "scan_edits_4096_28672.pt"):
    s = torch.load(VD / scan)
    rows |= {int(r) for r in s["fires"][("h.2.mlp.down_proj", 2359)][:, 0].tolist()}
rows = sorted(rows)
col = pq.ParquetFile(VAL_PARQUET).read_row_group(0, columns=["input_ids"]).column("input_ids")
toks = np.stack([np.asarray(col[r].as_py()[:512], dtype=np.int32) for r in rows])
decl = np.stack([np.asarray(col[r].as_py()[:512], dtype=np.int32) for r in range(40000, 40032)])  # declared-behaviour rows
np.savez(Path.home() / "mpd-data/frontier/e4_rows.npz", rows=np.array(rows), tokens=toks, decl=decl)
print(len(rows), toks.shape)
