"""Held-out verbatim edit faithfulness of an explanation of a language-model toy (#2951 toy gate),
by the engine's edits driver (mpd_library_mdl_2951 ... edits): the same swap, zero, scale and push
operations at the sites M and P share, applied identically to both, KL(M_e || P_e) in bits per
token. With no checkpoint the explanation is the library as built (M's own functions, the native
start), whose gaps are zero up to rounding: that run checks the toy's export and the driver.

usage: ~/mpd-data/venv/bin/python bench/toys_2951/engine_edits.py BINARY TOY_DIR OUT [SETTINGS.json [CHECKPOINT|- [MANIFEST]]]

Without SETTINGS.json, settings are written for the toy: held-out sequences [0, H) (H the rows
the truth's activity covers), training sequences the next ones. The per-family record
(OUT/EDITS_<name>.json) is printed as one line per family and merged into
OUT/SCORE_<toy>_edits.json.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main():
    binary, toy, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    record = json.loads((toy / "export.json").read_text())
    truth = json.loads((toy / "truth.json").read_text())
    rows, context = record["files"]["tokens"]["shape"]
    held = truth["active"]["shape"][0] // context
    out.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 4:
        settings = Path(sys.argv[4])
    else:
        settings = out / "settings.json"
        settings.write_text(json.dumps({
            "export_sha256": hashlib.sha256((toy / "export.json").read_bytes()).hexdigest(),
            "training_sequences": min(rows - held, 4 * held), "context": context, "held_out": [0, held],
            "fit": {"batch_sequences": 64, "seed": 0, "numeric_bytes": 1 << 30, "head_tile_rows": 4096},
        }))
    checkpoint = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != "-" else None
    name = Path(checkpoint).stem if checkpoint else "native"
    edits = out / f"edits_{name}.json"
    record_edits = {"checkpoint": checkpoint, "sequences": [0, min(held, 256)], "families": ["swap", "zero", "scale", "push"],
                    "edits_per_sequence": 4, "batch_sequences": 32, "seed": 0, "numeric_bytes": 1 << 30, "name": name}
    # the experiments: the manifest given (e.g. the native start's, so explanations compare on the
    # same draws), else this OUT's own once it exists (manifests are immutable)
    manifest = Path(sys.argv[6]) if len(sys.argv) > 6 else out / f"MANIFEST_{name}.json"
    if manifest.exists():
        record_edits["manifest"] = str(manifest)
    edits.write_text(json.dumps(record_edits))
    subprocess.run([str(binary), str(toy), str(settings), str(out), "host", "edits", str(edits)], check=True)
    result = json.loads((out / f"EDITS_{name}.json").read_text())
    families = result.get("families", result)
    for family, r in families.items():
        if isinstance(r, dict):
            keys = [k for k in r if k.startswith(("mean", "effect_mean", "ignoring_mean", "edited_token_mean"))]
            print(family, " ".join(f"{k}={r[k]:.4g}" for k in keys if isinstance(r[k], (int, float))))
    (out / f"SCORE_{toy.name}_edits.json").write_text(json.dumps({"toy": toy.name, "explanation": name, "edits": result}, indent=1))


if __name__ == "__main__":
    main()
