"""Teacher answers (#2951 graph oracle): gate programs from the subcomponent search, saying which subcomponents act at
which positions, scored by the checker (the model's choice between the behavior's answers; what the program turns on
sees the prompt, everything else the changed prompt, connected as in the model).

Per behavior, the search's set of lowest total (runs/prune_v4/<behavior>.json, unrestricted, scored at 64 experiments)
is written as three gate programs, each turning the whole set on:
  everywhere   at every position;
  changed      at the tokens where a prompt and its changed prompt differ (CHANGED, read off the behavior's own
               examples) and at the positions whose next token is asked;
  onward       at every position from the first CHANGED token on (before it the two prompts are the same, so nothing
               acting there carries a difference).
Each is scored at --experiments on seeds 0 and 1 against nothing named; the lowest seed-0 total is kept.

  teacher.py [BEHAVIOR...] [--out ~/mpd-data/graph_oracle/teacher_v6] [--heldout]
writes OUT/<behavior>.py, OUT/<behavior>.answer.txt (the program in a python block) and appends OUT/manifest.jsonl.
--heldout does the held-out behaviors instead, into teacher_heldout_v6: the search baseline of the evaluation, never
training input.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import explain  # noqa: E402
import mech  # noqa: E402
import score as score_module  # noqa: E402

DATA = Path.home() / "mpd-data/graph_oracle"
TERMS = ("total_bits", "exec_error_bits", "necessity_error_bits", "complexity_bits", "structure_bits", "code_bits", "N", "parts", "valid", "error")


def changed_tokens(behavior: dict) -> list[str]:
    """The token strings at the positions where a prompt and its changed prompt differ, in either."""
    tk = mech.tokenizer(behavior["model"])
    out = set()
    for p in behavior["prompts"]:
        cf = (p.get("counterfactual") or {}).get("token_ids") or []
        for a, b in zip(p["token_ids"], cf):
            if a != b:
                out |= {tk.decode([a]), tk.decode([b])}
    return sorted(out)


def program(parts: list[str], where: str, changed: list[str]) -> str:
    """The gate program turning `parts` on `where` (PATTERNS)."""
    lines = [f"CHANGED = {set(changed)!r}  # the tokens the changed prompts swap"] if where != "everywhere" else []
    lines += ["S = [" + ", ".join(f'"{p}"' for p in parts) + "]", "", "", "def on(tokens, targets):"]
    if where == "everywhere":
        lines.append("    return {i: S for i in range(len(tokens))}")
    elif where == "changed":
        lines.append("    return {i: S for i, t in enumerate(tokens) if t in CHANGED or i in targets}")
    else:
        lines += ["    first = next((i for i, t in enumerate(tokens) if t in CHANGED), len(tokens))",
                  "    return {i: S for i in range(first, len(tokens))}"]
    return "\n".join(lines) + "\n"


PATTERNS = ("everywhere", "changed", "onward")


def scored(checker, sources: list[str], experiments: int, seed: int, batch: int, options=None) -> list[dict]:
    out = []
    for k in range(0, len(sources), batch):
        out += checker.score_batch(sources[k:k + batch], experiments=experiments, seed=seed, options=options)
    return [{t: r.get(t) for t in TERMS} for r in out]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("behaviors", nargs="*", help="default: every behavior with a search run")
    ap.add_argument("--prune", type=Path, default=DATA / "runs/prune_v4")
    ap.add_argument("--behaviors-dir", type=Path, default=DATA / "behaviors_v3/vpd4l")
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--experiments", type=int, default=64)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--heldout", action="store_true", help="the held-out behaviors (evaluation baselines) instead of the training ones")
    a = ap.parse_args()
    a.out = a.out or DATA / ("teacher_heldout_v6" if a.heldout else "teacher_v6")
    a.out.mkdir(parents=True, exist_ok=True)
    for b in a.behaviors or sorted(p.stem for p in a.prune.glob("*.json")):
        t0 = time.time()
        path = a.behaviors_dir / f"{b}.json"
        behavior = json.loads(path.read_text())
        if (behavior.get("split") == "train") == a.heldout:
            continue
        run = json.loads((a.prune / f"{b}.json").read_text())
        best = min((c for c in run["curve"] if c["seed"] == 0 and c["score"]["valid"]), key=lambda c: c["score"]["total_bits"])
        parts = [explain.token(u) for u in explain.order(explain.units_of(" ".join(run["sets"][str(best["k"])])))]
        changed = changed_tokens(behavior)
        log = lambda m: print(f"{b}: {m}", flush=True)  # noqa: E731
        sources = [program(parts, w, changed) for w in PATTERNS]
        with score_module.Checker(behavior["model"], device=a.device) as checker:
            checker.behavior(path)
            seeds = {s: scored(checker, sources + [explain.ir([])], a.experiments, s, 1) for s in (0, 1)}
        totals = [seeds[0][i]["total_bits"] if seeds[0][i]["valid"] else float("inf") for i in range(len(PATTERNS))]
        pick = min(range(len(PATTERNS)), key=totals.__getitem__)
        source = sources[pick]
        (a.out / f"{b}.py").write_text(source)
        answer_path = a.out / f"{b}.answer.txt"
        answer_path.write_text(f"```python\n{source.strip()}\n```\n")
        empty = {s: seeds[s][-1] for s in seeds}
        shares = {s: {"reproduces": 1 - seeds[s][pick]["exec_error_bits"] / empty[s]["exec_error_bits"],
                      "removes": 1 - seeds[s][pick]["necessity_error_bits"] / empty[s]["necessity_error_bits"]} for s in seeds}
        record = {"behavior": b, "family": behavior["family"], "model": behavior["model"], "answer": str(answer_path),
                  "behavior_path": str(path), "parts": len(parts), "pattern": PATTERNS[pick], "totals": dict(zip(PATTERNS, totals)),
                  "experiments": a.experiments, "score": {s: seeds[s][pick] for s in seeds}, "empty_score": empty, "shares": shares,
                  "checker": str(score_module.BINARY), "seconds": round(time.time() - t0)}
        with open(a.out / "manifest.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        log(f"kept {PATTERNS[pick]} ({len(parts)} subcomponents); reproduces {shares[0]['reproduces']:.1%}/{shares[1]['reproduces']:.1%}, "
            f"removes {shares[0]['removes']:.1%}/{shares[1]['removes']:.1%} (seeds 0/1); totals "
            + ", ".join(f"{w} {t:.6g}" for w, t in zip(PATTERNS, totals)) + f"; {record['seconds']} s")


if __name__ == "__main__":
    main()
