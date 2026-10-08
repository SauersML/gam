"""Native rankings from measured patching (#2951): g-mech's per-behavior tables (examples/measure: the bits a
head's or neuron's clean output recovers when patched alone into the counterfactual run) as search.py
--ranking files {"native": {"heads": [layer][head], "neurons": [layer][neuron]}}; layers without a neuron
table rank their neurons last (0).

  patch_ranking.py [--tables ~/mpd-data/graph_oracle/experiments/examples] [--out ~/mpd-data/graph_oracle/experiments/patch_rankings]

With --vpd, the VPD vocabulary instead (--out default .../vpd_rankings): native heads (patch_vpd4l) and VPD MLP
subcomponents (vpdpatch_vpd4l: a subcomponent's clean contribution patched alone into the counterfactual run),
as {"mixed": [[unit name, recovery per opaque number], ...]} in decreasing order, for search.py --ranking.
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


def vpd_rankings(a, s) -> int:
    """Native heads and VPD MLP subcomponents ranked together by recovered bits per opaque number."""
    head_cost = 4 * s["d_model"] * s["head_dim"]
    sub_cost = s["d_model"] + s["d_mlp"]  # a c_fc or down_proj subcomponent's two vectors
    written = 0
    for path in sorted((a.tables / f"vpdpatch_{a.model}").glob("vpdpatch_*.json")):
        behavior = path.stem[len("vpdpatch_"):]
        heads_table = a.tables / f"patch_{a.model}" / f"patch_{behavior}.json"
        if not heads_table.exists():
            continue
        units = [(h["recovery_bits"] / head_cost, f"h{h['layer']}_{h['head']}") for h in json.loads(heads_table.read_text())["heads"]]
        for key, rec in json.loads(path.read_text())["recovery"].items():
            layer, site = key.split(".")
            if site in ("c_fc", "down_proj"):
                units += [(r / sub_cost, f"s{layer}_{site}_{i}") for i, r in enumerate(rec) if r > 0]
        units = [u for u in sorted(units, key=lambda x: -x[0]) if u[0] > 0]
        (a.out / f"{behavior}.json").write_text(json.dumps({"behavior": behavior, "source": f"{heads_table} and {path} (g-mech's patching)",
                                                            "mixed": [[n, v] for v, n in units]}))
        written += 1
    return written


def importance_rankings(a) -> int:
    """Every VPD subcomponent ranked by VPD's causal importance on the behavior's prompts and counterfactuals
    (mpd_vpd_importance_2951's mean over every position): under the checker's deletion semantics an unnamed
    part contributes zero, as VPD's masks zero it."""
    written = 0
    for path in sorted(a.importance.glob("*.json")):
        record = json.loads(path.read_text())
        units = [(g, f"s{site['layer']}_{name.split('.')[-1]}_{i}") for name, site in record["sites"].items()
                 for i, g in enumerate(site["mean"]) if g > 0]
        units.sort(key=lambda x: -x[0])
        (a.out / f"{record['behavior']}.json").write_text(json.dumps({"behavior": record["behavior"], "source": f"{path} (VPD's importances)",
                                                                      "mixed": [[n, v] for v, n in units]}))
        written += 1
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tables", type=Path, default=DATA / "examples")
    ap.add_argument("--out", type=Path, default=DATA / "patch_rankings")
    ap.add_argument("--model", default="vpd4l")
    ap.add_argument("--vpd", action="store_true", help="the VPD vocabulary: native heads and VPD MLP subcomponents")
    ap.add_argument("--importance", type=Path, help="mpd_vpd_importance_2951's output directory: every VPD subcomponent "
                                                    "ranked by VPD's importance (--out default .../importance_rankings)")
    a = ap.parse_args()
    s = mech.shapes(a.model)
    if a.importance:
        a.out = a.out if a.out != DATA / "patch_rankings" else DATA / "importance_rankings"
        a.out.mkdir(parents=True, exist_ok=True)
        print(f"{importance_rankings(a)} rankings in {a.out}")
        return
    if a.vpd:
        a.out = a.out if a.out != DATA / "patch_rankings" else DATA / "vpd_rankings"
        a.out.mkdir(parents=True, exist_ok=True)
        print(f"{vpd_rankings(a, s)} rankings in {a.out}")
        return
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
