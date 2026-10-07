"""Native rankings from measured patching (#2951): g-mech's per-behavior tables (examples/measure: the bits a
head's or neuron's clean output recovers when patched alone into the counterfactual run) as search.py
--ranking files {"native": {"heads": [layer][head], "neurons": [layer][neuron]}}; layers without a neuron
table rank their neurons last (0).

  patch_ranking.py [--tables ~/mpd-data/graph_oracle/experiments/examples] [--out ~/mpd-data/graph_oracle/experiments/patch_rankings]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import mech  # noqa: E402

DATA = Path.home() / "mpd-data/graph_oracle/experiments"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tables", type=Path, default=DATA / "examples")
    ap.add_argument("--out", type=Path, default=DATA / "patch_rankings")
    ap.add_argument("--model", default="vpd4l")
    a = ap.parse_args()
    s = mech.shapes(a.model)
    a.out.mkdir(parents=True, exist_ok=True)
    written = 0
    for path in sorted((a.tables / f"patch_{a.model}").glob("patch_*.json")):
        behavior = path.stem[len("patch_"):]
        patch = json.loads(path.read_text())
        heads = [[0.0] * s["heads"] for _ in range(s["layers"])]
        for h in patch["heads"]:
            heads[h["layer"]][h["head"]] = h["recovery_bits"]
        neurons = [[0.0] * s["d_mlp"] for _ in range(s["layers"])]
        layers = []
        for l in range(s["layers"]):
            table = a.tables / f"neurons_{a.model}" / f"neurons_{behavior}_l{l}.json"
            if table.exists():
                neurons[l] = json.loads(table.read_text())["recovery"]
                layers.append(l)
        (a.out / f"{behavior}.json").write_text(json.dumps({"behavior": behavior, "neuron_layers": layers,
                                                            "source": f"{path} and neurons_{behavior}_l*.json (g-mech's patching)",
                                                            "native": {"heads": heads, "neurons": neurons}}))
        written += 1
    print(f"{written} rankings in {a.out}")


if __name__ == "__main__":
    main()
