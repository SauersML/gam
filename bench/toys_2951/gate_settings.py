"""Settings for one library_vpd fit of a #2951 toy gate toy (mpd_library_mdl_2951's SETTINGS.json).

usage: ~/mpd-data/venv/bin/python bench/toys_2951/gate_settings.py TOY_DIR START_DIR ARM BUDGET EPOCHS OUT.json

ARM is one of toy_start.py's arms (per_slice_own, grouped_own, grouped_direction). BUDGET is
`true` (the per-token budget K at the toy's true count) or `none` (no budget, K = infinity). The
true count is, per held-out token, the rank-one slices the known mechanisms active there span
(each mechanism's numerical rank on every operator it spans), averaged, plus the start's zero
slices (a zero attention or MLP map's one slice, which every explanation runs and the budget
counts). Held-out sequences are those the truth covers; the training sequences are the rest of
the export's rows.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def main():
    toy, start, arm, budget, epochs, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5]), Path(sys.argv[6])
    record = json.loads((toy / "export.json").read_text())
    truth = json.loads((toy / "truth.json").read_text())
    rows, context = record["files"]["tokens"]["shape"]
    held = truth["active"]["shape"][0] // context
    fit = {"batch_sequences": max(1, 4096 // context), "seed": 0, "numeric_bytes": 1 << 30, "head_tile_rows": 4096, "epochs": epochs,
           "families": ["swap", "zero", "scale", "push"]}
    if budget == "true":
        active = np.fromfile(toy / truth["active"]["file"], dtype="<f8").reshape(truth["active"]["shape"])
        ranks = []
        for m in truth["mechanisms"]:
            total = 0
            for e in m["operators"].values():
                d = np.fromfile(toy / e["file"], dtype="<f8").reshape(e["shape"])
                s = np.linalg.svd(d, compute_uv=False)
                total += int(np.sum(s > s[0] * max(d.shape) * np.finfo(float).eps)) if s.size and s[0] > 0 else 0
            ranks.append(total)
        decomposition = json.loads((start / "decomposition" / "export.json").read_text())
        zero = 0
        for name in decomposition["config"]["sites"]:
            u = np.fromfile(start / "decomposition" / f"{name}.U.f64", dtype="<f8")
            zero += int(not np.any(u))
        fit["budget"] = float((active @ np.array(ranks, dtype=float)).mean()) + zero
    elif budget != "none":
        raise SystemExit(f"budget {budget}: true or none")
    out.write_text(json.dumps({
        "export_sha256": hashlib.sha256((toy / "export.json").read_bytes()).hexdigest(),
        "training_sequences": rows - held, "context": context, "held_out": [0, held],
        "vpd": {"decomposition": str(start / "decomposition"), "start": str(start / "start.json"), "arm": arm},
        "fit": fit,
    }, indent=1))
    print(out, fit.get("budget", "no budget"))


if __name__ == "__main__":
    main()
