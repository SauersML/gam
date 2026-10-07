"""R4's SFT data (#2951): (behavior prompt -> program) targets from TRAIN-split families only.

  r4_data.py --out DIR [--model vpd4l] [--questions r4_mix.jsonl --per-type 600]

- DIR/programs/<name>.json: g-mech's example programs (examples/index.json, split "train", a behavior
  file present) printed by printer.printed, whose comments state measured facts; {"behavior", "source",
  "example", "printed": true}, unscored, so train.py --mode sft uses every one.
- g-int's printed search exports (~/mpd-data/graph_oracle/printed/*.py with *.graph.json naming the
  behavior), when they exist, copied the same way for train-split behaviors.
- DIR/questions.jsonl: g-predict's prediction questions, at most --per-type of each type (records are
  shuffled at the source, so the first ones of a type are a uniform sample).
Held-out families never enter: the behavior file's split decides, and the questions file already
excludes them (g-predict)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
BEHAVIORS = Path.home() / "mpd-data/graph_oracle/behaviors"
PRINTED = Path.home() / "mpd-data/graph_oracle/printed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="vpd4l")
    ap.add_argument("--questions")
    ap.add_argument("--per-type", type=int, default=600)
    a = ap.parse_args()
    import mech
    import printer

    out = Path(a.out)
    (out / "programs").mkdir(parents=True, exist_ok=True)
    index = json.loads((HERE.parent / "examples/index.json").read_text())
    written, skipped = [], []
    for name, e in sorted(index.items()):
        path = BEHAVIORS / a.model / f"{e.get('behavior')}.json"
        if e["model"] != a.model or e["split"] != "train" or not e.get("behavior") or not path.exists():
            continue
        behavior = json.loads(path.read_text())
        if behavior.get("split") != "train":
            skipped.append((name, "behavior file is held out"))
            continue
        ir = mech.trace((HERE.parent / "examples" / f"{name}.py").read_text(), a.model)
        if not ir["valid"]:
            skipped.append((name, ir["error"]))
            continue
        try:
            source, printed = printer.printed(ir, behavior)[0], True
        except Exception as e:  # the printer's failure on this program: the example's own source and docstrings
            source, printed = ir["source"], False
            skipped.append((name, f"printer: {type(e).__name__}: {e}"))
        (out / "programs" / f"{name}.json").write_text(json.dumps({"behavior": behavior["id"], "source": source, "example": name, "printed": printed}))
        written.append(name)
    for graph in sorted(PRINTED.glob("*.graph.json")) if PRINTED.exists() else []:
        g = json.loads(graph.read_text())
        bid = g.get("behavior")
        path = BEHAVIORS / a.model / f"{bid}.json"
        if not bid or not path.exists() or json.loads(path.read_text()).get("split") != "train":
            continue
        src = graph.with_name(graph.name[: -len(".graph.json")] + ".py")
        if src.exists():
            rec = {"behavior": bid, "source": src.read_text(), "printed": True, "search": src.name}
            if "score" in g:
                rec["score"] = g["score"]
            (out / "programs" / f"search_{src.stem}.json").write_text(json.dumps(rec))
            written.append(src.name)
    counts = {}
    if a.questions:
        with open(out / "questions.jsonl", "w") as f:
            for line in open(Path(a.questions).expanduser()):
                q = json.loads(line)
                t = q.get("type")
                if counts.get(t, 0) < a.per_type:
                    counts[t] = counts.get(t, 0) + 1
                    f.write(json.dumps({"messages": q["messages"], "type": t}) + "\n")
    print(json.dumps({"programs": len(written), "names": written, "skipped": skipped, "questions_per_type": counts}))


if __name__ == "__main__":
    main()
