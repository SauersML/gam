"""Minimal programs in VPD's vocabulary (#2951; the oracle's teacher): per behavior, search.py's ranked-prefix
search over VPD subcomponents (every site) ranked by VPD's causal importance on the behavior's prompts
(patch_ranking.py --importance), then group and one-by-one pruning, under a part budget, scored with the VPD
view attached (deletion: unnamed parts contribute zero); the result as {"behavior", "source", "score",
"reproduced", "weights_share", "parts", "build"} in OUT/<behavior>.json.

reproduced = 1 - execution error / the empty program's (fit families, the empty program scored in the same
run: under deletion, M against the model with every part deleted but the shared base, --base); weights_share = opaque numbers / every
head and MLP number; parts = subcomponents. The search scores without necessity; its final pruning and the
result with it (search.py).

  vpd_min.py BEHAVIOR_ID... [--budget 1024] [--experiments 8] [--device gpu] [--out ~/mpd-data/graph_oracle/runs/vpd_min]
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


def base_parts(path: Path) -> set[str]:
    """The VPD subcomponents a shared base IR names (any node piece {"view": "vpd", "layer", "kind", "index"}),
    as search.py unit names s<l>_<site>_<i>."""
    out = set()

    def walk(x):
        if isinstance(x, dict):
            if x.get("view") == "vpd" and isinstance(x.get("index"), (int, list)):
                out.update(f"s{x['layer']}_{x['kind']}_{i}" for i in ([x["index"]] if isinstance(x["index"], int) else x["index"]))
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(json.loads(Path(path).read_text()))
    return out


def summary(behavior: str, result: dict, model: str = "vpd4l", split: str | None = None) -> dict:
    t = result["trajectory"][0]
    empty = next(s for k, s in t["prefixes"] if k == 0)
    found = result["score"]
    fams = tuple(f for f in table.FIT if f in found.get("per_family", {}) and f in empty.get("per_family", {}))
    e, f = table.shared(empty, fams), table.shared(found, fams)
    he, hf = table.shared(empty, table.HELDOUT), table.shared(found, table.HELDOUT)
    return {"behavior": behavior, "split": split, "source": result["source"], "score": found, "empty": empty, "standin": found.get("standin"),
            "reproduced": 1 - f[0] / e[0] if e[0] else 0.0,
            "reproduced_heldout": 1 - hf[0] / he[0] if he and hf and he[0] else None,
            "weights_share": found.get("opaque_numbers", 0) / MODEL_NUMBERS[model], "parts": found.get("parts", len(result["units"])),
            "necessity_bits": found.get("necessity_error_bits"), "total": f[1], "empty_total": e[1],
            "build": result.get("checker"), "calls": result.get("calls")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("behaviors", nargs="+")
    ap.add_argument("--model", default="vpd4l")
    ap.add_argument("--budget", type=int, default=1024, help="the most parts a program may declare")
    ap.add_argument("--cap", type=int, default=32, help="the most candidates of a run-pruning step")
    ap.add_argument("--growth", type=float, default=2.0, help="ratio between successive prefix sizes")
    ap.add_argument("--device")
    ap.add_argument("--experiments", type=int, default=8, help="per score during the search (the result is rescored on a held-out seed)")
    ap.add_argument("--rankings", type=Path, default=DATA / "experiments/importance_rankings")
    ap.add_argument("--behaviors-dir", type=Path, help="the behavior files (default ~/mpd-data/graph_oracle/behaviors/<model>)")
    ap.add_argument("--export", type=Path)
    ap.add_argument("--vpd", type=Path, default=Path.home() / "mpd-data/engine/vpd4l_decomposition")
    ap.add_argument("--out", type=Path, default=DATA / "runs/vpd_min")
    ap.add_argument("--base", help="the model's shared base IR (score.py's base: a file, or 1 for the published one); "
                                   "every program runs with it and the ranking leaves its parts out")
    a = ap.parse_args()
    env = dict(os.environ)
    if a.base:
        import score
        env["GRAPH_BASE"] = str(score.BASES[a.model]) if a.base in ("1", "True") else str(Path(a.base).expanduser())
        named = base_parts(Path(env["GRAPH_BASE"]))
    out = a.out.expanduser()
    work = out / "search"
    work.mkdir(parents=True, exist_ok=True)
    for b in a.behaviors:
        if (out / f"{b}.json").exists():
            continue
        behaviors = a.behaviors_dir or DATA / f"behaviors/{a.model}"
        ranking = a.rankings / f"{b}.json"
        if a.base:  # the base's parts are always on: rank the rest
            data = json.loads(ranking.read_text())
            data["mixed"] = [[n, v] for n, v in data["mixed"] if n not in named]
            ranking = work / f"{b}.ranking_without_base.json"
            ranking.write_text(json.dumps(data))
        cmd = [sys.executable, str(HERE / "search.py"), str(behaviors / f"{b}.json"), "--mode", "prefix",
               "--ranking", str(ranking), "--max-units", str(a.budget), "--objective", "fit",
               "--experiments", str(a.experiments), "--max-prune", "24", "--prompt-holdout", "4", "--vpd", str(a.vpd),
               "--prefix-growth", str(a.growth), "--cap", str(a.cap), "--tag", "_vpd_min", "--out", str(work)]
        cmd += ["--device", a.device] if a.device else []
        cmd += ["--export", str(a.export)] if a.export else []
        with open(work / f"{b}.stdout", "w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        res = work / f"{b}.prefix_vpd_min.json"
        if not res.exists():
            print(f"{b}: no result ({work / f'{b}.stdout'})", flush=True)
            continue
        s = summary(b, json.loads(res.read_text()), a.model, json.loads((behaviors / f"{b}.json").read_text()).get("split"))
        s["base"] = env.get("GRAPH_BASE")
        (out / f"{b}.json").write_text(json.dumps(s, indent=1))
        print(f"{b}: {s['parts']} parts, reproduced {s['reproduced']:.0%} (held-out families "
              f"{s['reproduced_heldout'] if s['reproduced_heldout'] is None else round(s['reproduced_heldout'] * 100)}%), "
              f"weights {s['weights_share']:.2%}, total {s['total']:.2f} vs empty {s['empty_total']:.2f}", flush=True)


if __name__ == "__main__":
    main()
