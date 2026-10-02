"""A modular-addition export (bench/mpd_engine_export_2951.py's layout) as a generic `transformer`
export that gam_mpd::import reads (#2951): the tensors are linked under the generic names, and the
samples are every (a, b) in Z_p^2 followed by the `=` token p.

usage: mpd_engine_export_modadd_generic_2951.py SRC_EXPORT_DIR OUT_DIR P
"""
import json
import os
import sys

import numpy as np

src, out, p = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(out, exist_ok=True)
rec = json.load(open(os.path.join(src, "export.json")))
names = {"W_E": "W_E", "W_pos": "W_pos", "W_U": "W_U"}
for k in ("W_Q", "W_K", "W_V", "W_O", "W_in", "b_in", "W_out", "b_out"):
    names[f"blocks.0.{k}"] = k
files = {}
for new, old in names.items():
    link = os.path.join(out, f"{new}.f64")
    if not os.path.exists(link):
        os.symlink(os.path.join(src, f"{old}.f64"), link)
    files[new] = rec["files"][old]
inputs = np.array([[a, b, p] for a in range(p) for b in range(p)], dtype="<f8")
inputs.tofile(os.path.join(out, "inputs.f64"))
cfg = rec["config"]
record = {
    "model": f"modadd_p{p}", "kind": "transformer", "source": rec.get("run"), "source_sha256": rec.get("run_sha256"),
    "config": {"n_layers": 1, "n_heads": cfg["n_heads"], "d_model": cfg["d_model"], "d_head": cfg["d_head"],
               "d_mlp": cfg["d_mlp"], "n_ctx": 3, "causal": True, "act": "relu", "normalization": None},
    "input": {"type": "token_sequence", "vocab_size": p + 1, "seq_len": 3,
              "generator": f"every (a, b) in Z_{p}^2 followed by the = token {p}"},
    "output": {"n_classes": p, "readout_positions": [2], "decode": "argmax"},
    "samples": {"file": "inputs.f64", "shape": [p * p, 3]},
    "files": files,
}
json.dump(record, open(os.path.join(out, "export.json"), "w"), indent=1)
print(out, len(files))
