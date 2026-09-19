"""Summarize driver.py JSONL output: median [min-max] per config, and gamfit-vs-pyGAM losses.

Usage: analyze.py FILE.jsonl [FILE.jsonl ...]
CPU (process_time) is the primary speed metric on a shared box; wall is reported too.
"""
import json
import statistics as st
import sys
from collections import defaultdict

recs = []
for fn in sys.argv[1:]:
    with open(fn) as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))

groups = defaultdict(list)
status = defaultdict(list)
for r in recs:
    key = (r["family"], r["n"], r["design"], r["lib"])
    status[key].append(r["status"])
    if r["status"] == "ok":
        groups[key].append(r)

METRICS = ["fit1_cpu_s", "fit1_s", "pred1_cpu_s", "import_s", "peak_rss_mb", "poll_max_rss_mb", "rmse_mu", "edf"]


def summ(rs, m):
    v = [r[m] for r in rs if r.get(m) is not None]
    if not v:
        return None
    return (st.median(v), min(v), max(v), len(v))


def fmt(s, prec=2):
    if s is None:
        return "-"
    med, lo, hi, k = s
    if abs(med) < 0.1:
        return f"{med:.4f} [{lo:.4f}-{hi:.4f}]"
    return f"{med:.{prec}f} [{lo:.{prec}f}-{hi:.{prec}f}]"


loads = [r["load_start"][0] for r in recs if "load_start" in r]
nprocs = {r.get("nproc") for r in recs if "nproc" in r}
print(f"records={len(recs)} nproc={nprocs} load1 median={st.median(loads) if loads else 'na'}"
      f" range=[{min(loads) if loads else 'na'}-{max(loads) if loads else 'na'}]")
print()
hdr = "| family | n | design | lib | reps | fit1 CPU s | fit1 wall s | pred CPU s | peak RSS MB (poll) | rmse_mu | edf |"
print(hdr)
print("|" + "---|" * 11)
for key in sorted(status, key=lambda k: (k[0], k[1], k[2], k[3])):
    fam, n, des, lib = key
    rs = groups.get(key, [])
    bad = [s for s in status[key] if s != "ok"]
    if not rs:
        print(f"| {fam} | {n:g} | {des} | {lib} | 0 ({','.join(bad)}) | - | - | - | - | - | - |")
        continue
    reps = f"{len(rs)}" + (f" (+{','.join(bad)})" if bad else "")
    print(f"| {fam} | {n:g} | {des} | {lib} | {reps} | {fmt(summ(rs, 'fit1_cpu_s'))} | {fmt(summ(rs, 'fit1_s'))} |"
          f" {fmt(summ(rs, 'pred1_cpu_s'))} | {fmt(summ(rs, 'poll_max_rss_mb'), 0)} |"
          f" {fmt(summ(rs, 'rmse_mu'))} | {fmt(summ(rs, 'edf'), 1)} |")

print()
print("Loss table: gamfit (default k) vs pyGAM default / gridsearch; ratio = gamfit / pyGAM (median CPU, median poll RSS)")
print("| family | n | design | CPU vs pygam | CPU vs pygam_gs | RSS vs pygam | RSS vs pygam_gs | pred CPU vs pygam |")
print("|---|---|---|---|---|---|---|---|")
cfgs = sorted({(k[0], k[1], k[2]) for k in groups})
for fam, n, des in cfgs:
    g = groups.get((fam, n, des, "gamfit"))
    if not g:
        continue
    row = []
    for other, m in [("pygam", "fit1_cpu_s"), ("pygam_gs", "fit1_cpu_s"), ("pygam", "poll_max_rss_mb"),
                     ("pygam_gs", "poll_max_rss_mb"), ("pygam", "pred1_cpu_s")]:
        o = groups.get((fam, n, des, other))
        a, b = summ(g, m), summ(o, m) if o else None
        if a and b and b[0] > 0:
            ratio = a[0] / b[0]
            row.append(f"{ratio:.2f}x" + (" **LOSS**" if ratio > 1.0 else ""))
        else:
            row.append("-")
    print(f"| {fam} | {n:g} | {des} | " + " | ".join(row) + " |")
