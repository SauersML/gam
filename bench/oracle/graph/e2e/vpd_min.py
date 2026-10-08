"""Minimal programs in VPD's vocabulary (#2951; the oracle's teacher): per behavior, search.py's ranked-prefix
search over native heads and VPD MLP subcomponents ranked by measured patching (patch_ranking.py --vpd),
then group and one-by-one pruning, under a part budget; the result as {"behavior", "source", "score",
"reproduced", "weights_share", "parts", "build"} in OUT/<behavior>.json.

reproduced = 1 - execution error / the empty program's (fit families, the empty program scored in the same
run); weights_share = opaque numbers / every head and MLP number; parts = heads + subcomponents.

  vpd_min.py BEHAVIOR_ID... [--budget 512] [--experiments 8] [--device gpu] [--out ~/mpd-data/graph_oracle/runs/vpd_min]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import table  # noqa: E402

DATA = Path.home() / "mpd-data/graph_oracle"
MODEL_NUMBERS = {"vpd4l": 28_324_608}


def summary(behavior: str, result: dict, model: str = "vpd4l", behavior_file: Path | None = None) -> dict:
    t = result["trajectory"][0]
    empty = next(s for k, s in t["prefixes"] if k == 0)
    found = result["score"]
    fams = tuple(f for f in table.FIT if f in found.get("per_family", {}) and f in empty.get("per_family", {}))
    e, f = table.shared(empty, fams), table.shared(found, fams)
    he, hf = table.shared(empty, table.HELDOUT), table.shared(found, table.HELDOUT)
    # validity: the empty program's clean-prompt error must be the behavior's own signal KL(M(x) || M(x'))
    # (g-behaviors' counterfactual_quality); c1870275cb's VPD view breaks it on some behaviors
    signal = json.loads(behavior_file.read_text()).get("counterfactual_quality", {}).get("mean_kl_bits") if behavior_file else None
    clean = empty.get("per_family", {}).get("clean", {}).get("mean_kl_bits")
    return {"behavior": behavior, "source": result["source"], "score": found, "empty": empty,
            "empty_clean_kl": clean, "signal_kl": signal,
            "reproduced": 1 - f[0] / e[0] if e[0] else 0.0,
            "reproduced_heldout": 1 - hf[0] / he[0] if he and hf and he[0] else None,
            "weights_share": found.get("opaque_numbers", 0) / MODEL_NUMBERS[model], "parts": len(result["units"]),
            "heads": sum(1 for u in result["units"] if u.startswith("h")), "total": f[1], "empty_total": e[1],
            "build": result.get("checker"), "calls": result.get("calls")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("behaviors", nargs="+")
    ap.add_argument("--model", default="vpd4l")
    ap.add_argument("--budget", type=int, default=512, help="the most parts a program may declare")
    ap.add_argument("--growth", type=float, default=2.0, help="ratio between successive prefix sizes")
    ap.add_argument("--device")
    ap.add_argument("--experiments", type=int, default=8, help="per score during the search (the result is rescored on a held-out seed)")
    ap.add_argument("--rankings", type=Path, default=DATA / "experiments/vpd_rankings")
    ap.add_argument("--behaviors-dir", type=Path, help="the behavior files (default ~/mpd-data/graph_oracle/behaviors/<model>)")
    ap.add_argument("--export", type=Path)
    ap.add_argument("--vpd", type=Path, default=Path.home() / "mpd-data/engine/vpd4l_decomposition")
    ap.add_argument("--out", type=Path, default=DATA / "runs/vpd_min")
    a = ap.parse_args()
    out = a.out.expanduser()
    work = out / "search"
    work.mkdir(parents=True, exist_ok=True)
    for b in a.behaviors:
        if (out / f"{b}.json").exists():
            continue
        behaviors = a.behaviors_dir or DATA / f"behaviors/{a.model}"
        cmd = [sys.executable, str(HERE / "search.py"), str(behaviors / f"{b}.json"), "--mode", "prefix",
               "--ranking", str(a.rankings / f"{b}.json"), "--max-units", str(a.budget), "--objective", "fit",
               "--experiments", str(a.experiments), "--max-prune", "24", "--prompt-holdout", "4", "--vpd", str(a.vpd),
               "--prefix-growth", str(a.growth), "--tag", "_vpd_min", "--out", str(work)]
        cmd += ["--device", a.device] if a.device else []
        cmd += ["--export", str(a.export)] if a.export else []
        with open(work / f"{b}.stdout", "w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ))
        res = work / f"{b}.prefix_vpd_min.json"
        if not res.exists():
            print(f"{b}: no result ({work / f'{b}.stdout'})", flush=True)
            continue
        s = summary(b, json.loads(res.read_text()), a.model, behaviors / f"{b}.json")
        (out / f"{b}.json").write_text(json.dumps(s, indent=1))
        print(f"{b}: {s['parts']} parts ({s['heads']} heads), reproduced {s['reproduced']:.0%} (held-out families "
              f"{s['reproduced_heldout'] if s['reproduced_heldout'] is None else round(s['reproduced_heldout'] * 100)}%), "
              f"weights {s['weights_share']:.2%}, total {s['total']:.2f} vs empty {s['empty_total']:.2f}", flush=True)


if __name__ == "__main__":
    main()
