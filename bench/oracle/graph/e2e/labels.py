"""Label search (#2951 graph oracle, format v3): for each intermediate variable of a behavior's family algorithm, the
groups of VPD subcomponents whose output carries it, found by interchange.

A variable's items are the behavior's items whose changed prompt varies it (vary.py's "varies"). On them, swapping a
group's output from the changed prompt into the prompt's run is v3's label test of a node labeled with the variable,
and it is what the checker computes as the group's collapse: M with the group at its changed-prompt values and every
other part on the prompt, against M on the changed prompt, clamped at the signal. A group's share moved =
1 - (that error) / (the signal, the error of moving nothing). Chunks of --chunk subcomponents per site are measured
alone, the --keep largest shares split in half and measured again down to --leaf; the leaves, best first, are then
joined into groups of k = --ks subcomponents, each measured as a whole. Scored on the clean prompts alone (no
experiments drawn), necessity on, everything else off.

  labels.py BEHAVIOR_ID... [--behaviors-dir ~/mpd-data/graph_oracle/behaviors_vary/vpd4l] [--out runs/labels]
writes OUT/<behavior>.<variable>.json {"variable", "items", "signal_bits", "leaves", "groups", "chunks"}.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import contrast  # noqa: E402
import score as score_module  # noqa: E402
import search  # noqa: E402

DATA = Path.home() / "mpd-data/graph_oracle"


def items_of(behavior: dict, variable: str) -> dict:
    """The behavior restricted to the items whose changed prompt varies `variable`."""
    prompts = [p for p in behavior["prompts"] if variable in (p.get("varies") or [])]
    return {**behavior, "id": f"{behavior['id']}.{variable}", "prompts": prompts,
            "targets": sum(len(p["target_positions"]) for p in prompts)}  # its own id: the checker's memos are keyed by it


def search_variable(checker, sizes: dict, a, log) -> dict:
    """Leaves ranked by the share of the signal their swap moves, and the groups joining the best of them."""
    def moved(groups):
        programs = [contrast.nodes_program(contrast.kept_of(g)) if g else contrast.nodes_program({}) for g in groups]
        results = []
        for k in range(0, len(programs), a.batch):
            results += checker.score_batch(programs[k:k + a.batch], experiments=0, seed=0, reader=False, stand_in="counterfactual")
        return [r["necessity_error_bits"] if r.get("valid", True) else float("inf") for r in results]

    signal = moved([[]])[0]
    share = lambda e: 1 - e / signal if signal else 0.0  # noqa: E731
    chunks = [(l, s, i, min(i + a.chunk, n)) for (l, s), n in sorted(sizes.items()) for i in range(0, n, a.chunk)]
    effect = {c: share(e) for c, e in zip(chunks, moved([[c] for c in chunks]))}
    log(f"signal {signal:.5g} bits; {len(chunks)} chunks; largest shares " + ", ".join(
        f"{c[0]}.{c[1]}[{c[2]}:{c[3]}] {effect[c]:.1%}" for c in sorted(chunks, key=lambda c: -effect[c])[:5]))
    open_, size = sorted(chunks, key=lambda c: -effect[c]), a.chunk
    while size > a.leaf:
        size //= 2
        top = [c for c in open_ if c[3] - c[2] > size][: a.keep]
        halves = [h for (l, s, i, j) in top for h in ((l, s, i, min(i + size, j)), (l, s, min(i + size, j), j)) if h[3] > h[2]]
        effect.update(zip(halves, [share(e) for e in moved([[h] for h in halves])]))
        split = set(top)
        open_ = sorted([c for c in open_ if c not in split] + halves, key=lambda c: -effect[c])
    leaves = sorted((c for c in effect if c[3] - c[2] <= a.leaf), key=lambda c: -effect[c])
    groups, joined = [], []
    for c in leaves:
        joined.append(c)
        n = sum(j - i for _, _, i, j in joined)
        if any(n >= k > n - (c[3] - c[2]) for k in a.ks):
            groups.append(list(joined))
    shares = [share(e) for e in moved(groups)]
    log("groups: " + ", ".join(f"{sum(j - i for _, _, i, j in g)}:{s:.1%}" for g, s in zip(groups, shares)))
    units = lambda g: [search.name(("sub", l, s, i)) for l, s, i0, j in g for i in range(i0, j)]  # noqa: E731
    return {"signal_bits": signal,
            "leaves": [{"chunk": list(c), "share": effect[c]} for c in leaves],
            "groups": [{"parts": len(units(g)), "units": units(g), "share": s} for g, s in zip(groups, shares)],
            "chunks": [{"chunk": list(c), "share": effect[c]} for c in sorted(effect, key=lambda c: -effect[c])[:256]]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("behaviors", nargs="+")
    ap.add_argument("--behaviors-dir", type=Path, default=DATA / "behaviors_vary/vpd4l")
    ap.add_argument("--export", type=Path, default=Path.home() / "mpd-data/engine/vpd4l")
    ap.add_argument("--vpd", type=Path, default=Path.home() / "mpd-data/engine/vpd4l_decomposition")
    ap.add_argument("--importance", type=Path, default=DATA / "experiments/importance", help="site sizes")
    ap.add_argument("--device")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--leaf", type=int, default=8)
    ap.add_argument("--keep", type=int, default=32)
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--out", type=Path, default=DATA / "runs/labels")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    work = a.out / "items"
    work.mkdir(exist_ok=True)
    for b in a.behaviors:
        behavior = json.loads((a.behaviors_dir / f"{b}.json").read_text())
        table = json.loads((a.importance / f"{b}.json").read_text())["sites"]
        sizes = {(site["layer"], key.split(".")[-1]): len(site["mean"]) for key, site in table.items()}
        for v in [v for v in behavior.get("varies", {}) if v != "tokens"]:
            t0 = time.time()
            sub = items_of(behavior, v)
            path = work / f"{b}.{v}.json"
            path.write_text(json.dumps(sub))
            log = lambda m: print(f"{b} {v}: {m}", flush=True)  # noqa: E731
            with score_module.Checker(behavior["model"], export=a.export, views={"vpd": a.vpd}, device=a.device) as checker:
                checker.behavior(path)
                record = search_variable(checker, sizes, a, log)
            record = {"behavior": b, "variable": v, "items": len(sub["prompts"]),
                      "items_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "checker": str(score_module.BINARY),
                      "seconds": round(time.time() - t0), **record}
            (a.out / f"{b}.{v}.json").write_text(json.dumps(record, indent=1))
            log(f"done in {record['seconds']} s")


if __name__ == "__main__":
    main()
