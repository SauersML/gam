"""Search for a behavior's best program on Qwen3-0.6B, starting from measured patching (MPD #2951 graph oracle, R6).

Full greedy addition over Qwen3-0.6B's 448 heads and 28 MLPs costs ~500 checker calls per step. This search uses the
measured single-component patching table (examples/measure/qwen_patch.py: recovery in bits when one component's clean
output is patched into the counterfactual run) to order the candidates, and the score to choose among them:
  1. prefixes  score the top-k components of the ranking for k in a doubling schedule, every causal edge among them
               declared (e2e/search.py's source()); keep the best k;
  2. removal   greedy removal from that set (e2e/search.py's greedy, mode removal, MLPs split into dyadic neuron
               blocks): each step drops the piece whose removal lowers the total most;
  3. addition  greedy addition restricted to the next `--pool` ranked components not yet declared.
The final program is rescored under a held-out experiment seed. Every scored program goes to the run's JSONL.

    ranked.py BEHAVIOR.json PATCH.json [--experiments 16] [--pool 24] [--out ~/mpd-data/graph_oracle/runs/r6]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e2e"))
sys.path.insert(0, str(HERE.parent))
import search  # noqa: E402  (g-int's search: source(), Pool, greedy())


def ranking(patch: dict, model: str) -> list[tuple]:
    """Components in order of measured recovery (bits), positive recovery only."""
    d_mlp = search.mech.shapes(model)["d_mlp"]
    units = [(h["recovery_bits"], ("head", h["layer"], h["head"])) for h in patch["heads"]]
    units += [(m["recovery_bits"], ("mlp", m["layer"], 0, d_mlp)) for m in patch["mlps"]]
    return [u for r, u in sorted(units, key=lambda x: -x[0]) if r > 0]


def terms(r: dict) -> dict:
    return {k: r.get(k) for k in ("total_bits", "exec_error_bits", "code_bits", "opaque_bits", "opaque_numbers", "python_tokens", "N", "valid")}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("behavior", type=Path)
    ap.add_argument("patch", type=Path)
    ap.add_argument("--experiments", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heldout-seed", type=int, default=1)
    ap.add_argument("--pool", type=int, default=24, help="ranked components the addition step may draw from")
    ap.add_argument("--max-prefix", type=int, default=32)
    ap.add_argument("--min-neurons", type=int, default=768)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path.home() / "mpd-data/graph_oracle/runs/r6")
    ap.add_argument("--stages", default="prefix,removal,addition", help="which stages run after the prefixes (removal, addition)")
    ap.add_argument("--prefixes", default="1,2,4,8,12,16,24,32", help="prefix lengths scored besides the empty program")
    a = ap.parse_args()
    beh_path = a.behavior.expanduser()
    beh = json.loads(beh_path.read_text())
    rank = ranking(json.loads(a.patch.expanduser().read_text()), beh["model"])
    out = a.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / f"{beh['id']}.jsonl"
    pool = search.Pool(beh["model"], beh_path, a.workers)
    t0 = time.time()

    def record(stage, units, r):
        with log_path.open("a") as f:
            f.write(json.dumps({"stage": stage, "units": [search.name(u) for u in units], **terms(r), "calls": pool.calls,
                                "seconds": round(time.time() - t0, 1)}) + "\n")

    # 1. prefixes of the ranking
    ks = [0] + [k for k in map(int, a.prefixes.split(",")) if k <= min(a.max_prefix, len(rank))]
    results = pool.score([search.source(rank[:k]) for k in ks], a.experiments, a.seed)
    for k, r in zip(ks, results):
        record(f"prefix{k}", rank[:k], r)
        print(f"prefix {k:3d}: total {r['total_bits']:.6g}  exec {r['exec_error_bits']:.6g}  opaque {r['opaque_bits']:.6g}", flush=True)
    kbest = ks[min(range(len(ks)), key=lambda i: results[i]["total_bits"])]
    start = rank[:kbest]

    # 2. greedy removal from the best prefix
    current, best = start, results[ks.index(kbest)]
    if "removal" in a.stages and current:
        removed = search.greedy(pool, beh["model"], "removal", a.experiments, a.seed, a.min_neurons, print, start=start)
        current, best = removed["units"], removed["score"]
        record("removal", current, best)

    # 3. greedy addition from the next ranked components
    while "addition" in a.stages:
        names = {search.name(u) for u in current}
        cand = [u for u in rank[:kbest + a.pool] if search.name(u) not in names and not any(u[0] == "mlp" and v[0] == "mlp" and u[1] == v[1] for v in current)]
        if not cand:
            break
        rs = pool.score([search.source(current + [u]) for u in cand], a.experiments, a.seed)
        i = min(range(len(cand)), key=lambda j: rs[j]["total_bits"])
        print(f"addition: best {search.name(cand[i])} {rs[i]['total_bits']:.6g} vs {best['total_bits']:.6g}", flush=True)
        if rs[i]["total_bits"] >= best["total_bits"]:
            break
        current, best = current + [cand[i]], rs[i]
        record("addition", current, best)

    held = pool.score([search.source(current), search.source([])], a.experiments, a.heldout_seed)
    record("final_heldout", current, held[0])
    record("empty_heldout", [], held[1])
    result = {"behavior": beh["id"], "units": [search.name(u) for u in current], "source": search.source(current),
              "fit": terms(best), "heldout": terms(held[0]), "empty_heldout": terms(held[1]), "calls": pool.calls,
              "experiments": a.experiments, "seed": a.seed, "heldout_seed": a.heldout_seed, "prefix_k": kbest,
              "patch": str(a.patch), "seconds": round(time.time() - t0, 1)}
    (out / f"{beh['id']}.result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("behavior", "units", "fit", "heldout", "empty_heldout", "calls", "seconds")}, indent=1))
    pool.close()


if __name__ == "__main__":
    main()
