#!/usr/bin/env python3
"""Aggregate results/*.json from bench_accuracy.py into win/loss/tie tables.

For each case and each pyGAM variant, the paired per-fold difference
d_k = metric(gamfit) - metric(pygam) is summarised as mean +- SE (SE = sd/sqrt(K)).
Lower-is-better metrics (dev, rmse, logloss, brier, truth_mse): gamfit WIN when
mean(d) < -2 SE, LOSS when mean(d) > +2 SE, TIE otherwise. AUC is higher-better.
A method that failed on any fold loses that case outright (a fit must come from
a converged optimisation; an exception is a real failure).

Also reports relative difference (%) of the primary metric.
Usage: python report_accuracy.py [--md OUT.md]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

PRIMARY = {"gaussian": "dev", "binomial": "logloss", "poisson": "dev", "gamma": "dev"}
EXTRA = {"binomial": ["brier", "auc"], "gaussian": ["rmse"], "poisson": ["rmse"], "gamma": ["rmse"]}
HIGHER_BETTER = {"auc"}


def load():
    out = []
    for p in sorted(RES.glob("*.json")):
        out.append(json.loads(p.read_text()))
    return out


def paired(rows, metric, a="gamfit", b="pygam_grid"):
    ra = {r["fold"]: r for r in rows if r["method"] == a}
    rb = {r["fold"]: r for r in rows if r["method"] == b}
    folds = sorted(set(ra) & set(rb))
    fa = [f for f in folds if not ra[f]["ok"]]
    fb = [f for f in folds if not rb[f]["ok"]]
    if fa or fb:
        return {"fail_a": len(fa), "fail_b": len(fb)}
    d = np.array([ra[f][metric] - rb[f][metric] for f in folds], float)
    base = np.array([rb[f][metric] for f in folds], float)
    va = np.array([ra[f][metric] for f in folds], float)
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
    return {"mean_a": va.mean(), "mean_b": base.mean(), "d": d.mean(), "se": se,
            "rel": d.mean() / abs(base.mean()) * 100 if base.mean() != 0 else np.nan}


def verdict(p, metric):
    if "fail_a" in p:
        if p["fail_a"] and not p["fail_b"]:
            return "LOSS(fail)"
        if p["fail_b"] and not p["fail_a"]:
            return "WIN(fail)"
        return "both-fail"
    d, se = p["d"], p["se"]
    if metric in HIGHER_BETTER:
        d = -d
    if not np.isfinite(se) or se == 0:
        return "TIE" if d == 0 else ("WIN" if d < 0 else "LOSS")
    if d < -2 * se:
        return "WIN"
    if d > 2 * se:
        return "LOSS"
    return "TIE"


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "nan"
    return f"{x:.{nd}g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    data = load()
    lines = []
    tallies = {}
    losses = []
    for vs in ("pygam_default", "pygam_grid"):
        lines.append(f"\n### gamfit vs {vs}\n")
        lines.append("| case | n | family | metric | gamfit | pyGAM | diff (gamfit-pyGAM) +- SE | rel % | verdict | extra |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for c in data:
            rows = c["rows"]
            metrics = [PRIMARY[c["family"]]]
            if c["kind"] == "synth":
                metrics = ["truth_mse"] + metrics
            for m in metrics:
                p = paired(rows, m, "gamfit", vs)
                v = verdict(p, m)
                key = (vs, m if m == "truth_mse" else "heldout")
                tallies.setdefault(key, {}).setdefault(v.split("(")[0], []).append(c["name"])
                extra = []
                if m != "truth_mse":
                    for e in EXTRA.get(c["family"], []):
                        pe = paired(rows, e, "gamfit", vs)
                        if "d" in pe:
                            extra.append(f"{e}: {fmt(pe['mean_a'])} vs {fmt(pe['mean_b'])} [{verdict(pe, e)}]")
                if "d" in p:
                    lines.append(f"| {c['name']} | {c['n']} | {c['family']} | {m} | {fmt(p['mean_a'])} | {fmt(p['mean_b'])} | "
                                 f"{fmt(p['d'])} +- {fmt(p['se'], 2)} | {p['rel']:+.1f} | **{v}** | {'; '.join(extra)} |")
                else:
                    lines.append(f"| {c['name']} | {c['n']} | {c['family']} | {m} | fail={p['fail_a']} | fail={p['fail_b']} | - | - | **{v}** | |")
                if v.startswith("LOSS"):
                    losses.append((vs, c["name"], m, p))
    head = ["## Tally (gamfit perspective)\n", "| comparison | metric | WIN | TIE | LOSS | both-fail |", "|---|---|---|---|---|---|"]
    for (vs, m), t in sorted(tallies.items()):
        head.append(f"| vs {vs} | {m} | {len(t.get('WIN', []))} | {len(t.get('TIE', []))} | {len(t.get('LOSS', []))} | {len(t.get('both-fail', []))} |")
    head.append("\n## Losses\n")
    for vs, name, m, p in losses:
        head.append(f"- vs {vs}: **{name}** [{m}] " + (f"gamfit {fmt(p['mean_a'])} vs {fmt(p['mean_b'])} ({p['rel']:+.1f}%, d={fmt(p['d'])}+-{fmt(p['se'],2)})" if 'd' in p else f"fail gamfit={p['fail_a']} pygam={p['fail_b']}"))
    errs = []
    for c in data:
        for r in c["rows"]:
            if not r["ok"]:
                errs.append(f"- {c['name']} fold {r['fold']} {r['method']}: {r['error'][:300]}")
    txt = "\n".join(head + lines + ["\n## Errors\n"] + (errs or ["(none)"]))
    print(txt)
    if a.md:
        Path(a.md).write_text(txt)


if __name__ == "__main__":
    main()
