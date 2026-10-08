"""Score check of a ground-truth implant (#2951, g-truth): the true answer (implant.py truth: OUT/answer.py) against
subsets, supersets, wrong alignments and wrong algorithms, all scored by the checker on M' (OUT/export with its
decomposition OUT/decomposition) under one experiment draw. The true answer must win; any alternative that scores
below it shows where the score is wrong.

    python check.py OUT_DIR [--experiments 16] [--seeds 0 1] [--base 1] [--device gpu]

Writes OUT/check.jsonl (one line per variant and seed: its score terms) and prints the table.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import edits  # noqa: E402
import score as score_module  # noqa: E402
from implant import CODES, IMPORTANCE, base_parts, block  # noqa: E402

TERMS = ("total_bits", "exec_error_bits", "necessity_error_bits", "alignment_error_bits", "claim_error_bits", "complexity_bits",
         "base_bits", "N", "parts", "valid", "error")
EMPTY = {"model": "vpd4l", "nodes": [], "edges": [], "python_tokens": 0, "token_types": 0, "source": "", "valid": True}


def others(truth: dict, active: bool, n: int, rng: random.Random) -> dict:
    """Per step, n subcomponents of its sites that are not implanted and not in the shared base: the most active on
    generic text (active=True) or never active ones."""
    imp = json.loads(IMPORTANCE.read_text())["sites"]
    base = base_parts()
    out = {}
    for st in truth["steps"]:
        named = set(st["parts"])
        sites = ("q_proj", "k_proj", "v_proj", "o_proj") if st["block"] == "attn" else ("c_fc", "down_proj")
        pool = []
        for s in sites:
            rec = imp[f"h.{st['layer']}.{block(s)}.{s}"]
            for i, a in enumerate(rec["active"]):
                tok = f"<p:{st['layer']}.{CODES[s]}.{i}>"
                if tok not in named and i not in base.get((st["layer"], s), set()) and ((a > 0) == active):
                    pool.append((-rec["mean"][i] if active else rng.random(), tok))
        pool.sort()
        out[st["variable"]] = [t for _, t in pool[:n]]
    return out


def variants(out: Path, rng: random.Random) -> dict:
    truth = json.loads((out / "truth.json").read_text())
    true = (out / "answer.py").read_text()
    ans = edits.Answer.parse(true)
    by_var = {s.variable: s for s in ans.statements}

    def edited(changes: dict) -> str:
        """changes {variable: new parts tuple or None (statement dropped)}."""
        sts = [replace(s, parts=changes[s.variable]) if s.variable in changes and changes[s.variable] else s
               for s in ans.statements if not (s.variable in changes and changes[s.variable] is None)]
        return replace(ans, statements=tuple(sts)).source()

    def keep(var, codes, frac=1.0):
        parts = by_var[var].parts
        sel = [p for p in parts if edits.SITE.match(p)[2] in codes]
        drop = set(sel[: int(round(len(sel) * frac))])
        return tuple(p for p in parts if p not in drop)

    v = {"true": true, "empty": EMPTY}
    v["no_key_step"] = edited({"key": None})
    v["key_without_o"] = edited({"key": keep("key", {"o"})})
    v["key_without_qk"] = edited({"key": keep("key", {"q", "k"})})
    v["key_without_v"] = edited({"key": keep("key", {"v"})})
    v["answer_half_fc"] = edited({"answer": keep("answer", {"fc"}, 0.5)})
    v["answer_half_down"] = edited({"answer": keep("answer", {"down"}, 0.5)})
    for frac, name in ((1.0, "plus_active8"), (0.25, "plus_active2")):
        extra = others(truth, True, int(8 * frac), rng)
        v[name] = edited({var: by_var[var].parts + tuple(extra[var]) for var in by_var})
    extra = others(truth, False, 8, rng)
    v["plus_dead8"] = edited({var: by_var[var].parts + tuple(extra[var]) for var in by_var})
    # the key step aligned to other layer-a parts of the same sites and counts (the true ones not named)
    dead = others(truth, False, 64, rng)
    swapped = []
    for p in by_var["key"].parts:
        code = edits.SITE.match(p)[2]
        swapped.append(next(t for t in dead["key"] if edits.SITE.match(t)[2] == code and t not in swapped))
    v["key_on_other_parts"] = edited({"key": tuple(swapped)})
    for w in ("table", "first"):
        v[f"wrong_algorithm_{w}"] = (out / f"answer.wrong_{w}.py").read_text()
    # every single-part drop (the per-part credit of the true answer)
    for st in ans.statements:
        for p in st.parts:
            v[f"drop {p}"] = edited({st.variable: tuple(q for q in st.parts if q != p) or None})
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--experiments", type=int, default=16)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--base", default="1", help="1: the model's published shared base; 0: none; or a base IR path")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--explanation", action="store_true", help="score each variant with the true English explanation")
    a = ap.parse_args()
    out = a.out
    behavior = json.loads((out / "behavior.json").read_text())
    vs = variants(out, random.Random(0))
    explanation = (out / "explanation.txt").read_text() if a.explanation else ""
    names = list(vs)
    base = None if a.base == "0" else (True if a.base == "1" else a.base)
    rows = []
    with score_module.Checker("vpd4l", export=behavior["export"], views={"vpd": behavior["vpd"]}, device=a.device, base=base or "") as c:
        c.behavior(out / "behavior.json")
        for seed in a.seeds:
            for k in range(0, len(names), a.batch):
                chunk = names[k : k + a.batch]
                progs = [vs[n] if isinstance(vs[n], dict) else {"source": vs[n], "explanation": explanation} for n in chunk]
                for n, r in zip(chunk, c.score_batch(progs, experiments=a.experiments, seed=seed, reader=False)):
                    rows.append({"variant": n, "seed": seed, "base": a.base, "experiments": a.experiments,
                                 **{t: r.get(t) for t in TERMS}})
                    print(json.dumps(rows[-1]), flush=True)
    with open(out / "check.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for seed in a.seeds:
        sr = [r for r in rows if r["seed"] == seed]
        true = next(r for r in sr if r["variant"] == "true")
        print(f"\nseed {seed}: true total {true['total_bits']:.1f} bits (exec {true['exec_error_bits']:.1f}, necessity "
              f"{true['necessity_error_bits']:.1f}, alignment {true['alignment_error_bits']:.1f}, complexity {true['complexity_bits']:.1f})")
        print(f"{'variant':34s} {'total':>10s} {'d_total':>9s} {'exec':>10s} {'nec':>8s} {'align':>8s} {'cplx':>8s} parts")
        for r in sorted(sr, key=lambda r: (r["total_bits"] if r["total_bits"] is not None else 1e18)):
            if r["variant"].startswith("drop ") and r["total_bits"] is not None and r["total_bits"] > true["total_bits"]:
                continue
            t = r["total_bits"]
            print(f"{r['variant']:34s} {t if t is not None else float('nan'):10.1f} {(t or 0) - true['total_bits']:9.1f} "
                  f"{r['exec_error_bits'] or 0:10.1f} {r['necessity_error_bits'] or 0:8.1f} {r['alignment_error_bits'] or 0:8.1f} "
                  f"{r['complexity_bits'] or 0:8.1f} {r['parts']} {'' if r['valid'] else 'INVALID ' + str(r['error'])[:80]}")
        drops = [r for r in sr if r["variant"].startswith("drop ")]
        worse = sum(r["total_bits"] is not None and r["total_bits"] > true["total_bits"] for r in drops)
        print(f"single-part drops that score worse than the true answer: {worse} of {len(drops)}")


if __name__ == "__main__":
    main()
