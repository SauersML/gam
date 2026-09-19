"""Render gamfit-vs-pyGAM benchmark records as a markdown report.

    python -m bench.pygam_compare.report RUN_DIR_OR_JSONL [...] [--out FILE]

Inputs are ``records.jsonl`` files (or the run directories holding them, whose
``meta.json`` is then picked up too); several inputs are concatenated, so a
docs page can be regenerated from the committed baselines of several plans:

    python -m bench.pygam_compare.report bench/pygam_compare/baseline/quick \
        --out docs/benchmarks.md

Verdict rules (all "lower is better"; every LOSS is printed, none is hidden):

* Speed / memory: ratio of medians over the ok reps, gamfit / comparator.
  A ratio above 1.00 is marked **LOSS**.
* Accuracy (rmse_mu, deviance, logscore, |coverage - 0.95|): paired by seed
  over reps where both libraries are ok. The mean paired difference d
  (gamfit - comparator) is a **LOSS** when d > 2 SE, a WIN when d < -2 SE,
  otherwise "worse n.s." / "better n.s." / TIE. With a single paired seed
  there is no SE, so the sign alone decides and the verdict says "(1 seed)".
* Status: a cell where gamfit has fewer ok reps than the comparator is a
  **LOSS(status)** regardless of the numbers (listed once per cell in the loss
  list; metric tables show it wherever gamfit has no ok rep to measure); a
  metric the comparator reports and gamfit does not, on a seed where both are
  ok, is a **LOSS(missing)**.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

GAMFIT = "gamfit"
COMPARATORS: tuple[str, ...] = ("pygam", "pygam_gs")
COVERAGE_TARGET = 0.95
SE_MULTIPLIER = 2.0

Record = dict[str, Any]
CellKey = tuple[str, int, str]


@dataclass(frozen=True)
class Verdict:
    text: str
    loss: bool = False
    win: bool = False

    @property
    def status(self) -> bool:
        # Status losses are listed once, from the status table, not once per metric.
        return "(status)" in self.text


def _cells(records: Iterable[Record]) -> list[CellKey]:
    seen: dict[CellKey, None] = {}
    for r in records:
        seen.setdefault((str(r["family"]), int(r["n"]), str(r["design"])), None)
    return list(seen)


def _cell_name(cell: CellKey) -> str:
    family, n, design = cell
    return f"{family} n={n:g} {design}"


def _group(records: Iterable[Record]) -> dict[tuple[CellKey, str], list[Record]]:
    out: dict[tuple[CellKey, str], list[Record]] = defaultdict(list)
    for r in records:
        out[((str(r["family"]), int(r["n"]), str(r["design"])), str(r["lib"]))].append(
            r
        )
    return out


def _ok(recs: list[Record]) -> list[Record]:
    return [r for r in recs if r.get("status") == "ok"]


def _values(recs: list[Record], metric: str) -> list[float]:
    return [float(r[metric]) for r in _ok(recs) if r.get(metric) is not None]


def _median(recs: list[Record], metric: str) -> float | None:
    vals = _values(recs, metric)
    return statistics.median(vals) if vals else None


def _fmt(v: float | None, unit: str = "") -> str:
    if v is None:
        return "—"
    return f"{v:.3g}{unit}"


def _status_summary(recs: list[Record]) -> str:
    if not recs:
        return "—"
    counts = Counter(str(r.get("status")) for r in recs)
    ok = counts.pop("ok", 0)
    parts = [f"{ok}/{len(recs)} ok"]
    parts += [f"{c} {s}" for s, c in sorted(counts.items())]
    return ", ".join(parts)


def ratio_verdict(g: list[Record], c: list[Record], metric: str) -> Verdict:
    """Ratio of medians gamfit / comparator; > 1 is a LOSS."""
    gm, cm = _median(g, metric), _median(c, metric)
    if not c:
        return Verdict("—")
    if gm is None and cm is None:
        return Verdict("n/a (neither ok)")
    if gm is None:
        return Verdict("**LOSS(status)**", loss=True)
    if cm is None:
        return Verdict("WIN(status)", win=True)
    if cm <= 0.0:
        return Verdict("n/a (zero)")
    r = gm / cm
    if r > 1.0:
        return Verdict(f"{r:.2f}x **LOSS**", loss=True)
    return Verdict(f"{r:.2f}x", win=r < 1.0)


def _paired(g: list[Record], c: list[Record], metric: str) -> tuple[list[float], bool]:
    """Paired per-seed differences; also whether gamfit lacks a metric the
    comparator has on some seed where both are ok."""
    gs = {r["seed"]: r for r in _ok(g)}
    cs = {r["seed"]: r for r in _ok(c)}
    diffs: list[float] = []
    missing = False
    for seed in sorted(set(gs) & set(cs)):
        gv, cv = gs[seed].get(metric), cs[seed].get(metric)
        if cv is None:
            continue
        if gv is None:
            missing = True
            continue
        if metric == "coverage":
            diffs.append(abs(gv - COVERAGE_TARGET) - abs(cv - COVERAGE_TARGET))
        else:
            diffs.append(float(gv) - float(cv))
    return diffs, missing


def paired_verdict(g: list[Record], c: list[Record], metric: str) -> Verdict:
    if not c:
        return Verdict("—")
    diffs, missing = _paired(g, c, metric)
    if missing:
        return Verdict("**LOSS(missing)**", loss=True)
    if not diffs:
        if _values(c, metric) and not _values(g, metric):
            return Verdict("**LOSS(status)**", loss=True)
        return Verdict("n/a")
    mean = statistics.fmean(diffs)
    if len(diffs) == 1:
        tag = " (1 seed)"
        if mean > 0:
            return Verdict(f"**LOSS**{tag} Δ={mean:+.3g}", loss=True)
        if mean < 0:
            return Verdict(f"WIN{tag} Δ={mean:+.3g}", win=True)
        return Verdict(f"TIE{tag}")
    se = statistics.stdev(diffs) / math.sqrt(len(diffs))
    delta = f"Δ={mean:+.3g}±{se:.2g}"
    if mean > SE_MULTIPLIER * se:
        return Verdict(f"**LOSS** {delta}", loss=True)
    if mean < -SE_MULTIPLIER * se:
        return Verdict(f"WIN {delta}", win=True)
    if mean > 0:
        return Verdict(f"worse n.s. {delta}")
    if mean < 0:
        return Verdict(f"better n.s. {delta}")
    return Verdict("TIE")


SPEED_METRICS: tuple[tuple[str, str], ...] = (
    ("fit_cpu_s", "fit CPU"),
    ("fit_s", "fit wall"),
    ("pred_cpu_s", "predict CPU"),
    ("interval_cpu_s", "interval CPU"),
    ("proc_wall_s", "process wall (import+fit+predict)"),
    ("peak_rss_mb", "peak RSS"),
)
ACCURACY_METRICS: tuple[tuple[str, str], ...] = (
    ("rmse_mu", "RMSE vs true mean"),
    ("deviance", "held-out mean deviance"),
    ("logscore", "held-out log score (NLL)"),
    ("coverage", "95% CI coverage of true mean (verdict on |cov-0.95|)"),
)


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def render(records: list[Record], metas: list[dict[str, Any]] | None = None) -> str:
    metas = metas or []
    groups = _group(records)
    cells = _cells(records)
    libs = [lib for lib in (GAMFIT, *COMPARATORS) if any(k[1] == lib for k in groups)]
    comparators = [c for c in COMPARATORS if c in libs]
    losses: list[str] = []
    wins = 0
    lines: list[str] = [
        "# gamfit vs pyGAM benchmark",
        "",
        "<!-- Generated by bench/pygam_compare/report.py; do not edit by hand. -->",
        "",
        "`pygam` is pyGAM's default `.fit` (fixed lambda=0.6 per term); `pygam_gs` is",
        "`.gridsearch` over lambda, the fair comparator for gamfit's REML/LAML",
        "smoothing selection. Every rep is a fresh subprocess with BLAS/OpenMP/Rayon",
        "pinned to one thread. Ratios are gamfit / comparator (lower is better for",
        "gamfit); every ratio above 1.00 and every significant accuracy deficit is",
        "marked **LOSS**. Timeouts and memory caps are a harness safety net, recorded",
        "as a status and counted against the library, never skipped.",
        "",
    ]
    for meta in metas:
        plan = meta.get("plan", {})
        lines += [
            f"## Run: plan `{plan.get('name')}`",
            "",
            f"- {plan.get('description')}",
            f"- git sha `{meta.get('git_sha')}`, started {meta.get('started')},"
            f" finished {meta.get('finished')}",
            f"- host: {meta.get('platform')}, python {meta.get('python')},"
            f" nproc {meta.get('nproc')}, RAM {meta.get('total_ram_mb', 0) / 1024:.1f} GiB",
            f"- versions: {', '.join(meta.get('lib_versions', []))}",
            f"- thread env: {', '.join(f'{k}={v}' for k, v in meta.get('thread_env', {}).items())}",
            f"- safety net (not a solver budget): timeout {plan.get('timeout_s')} s/rep,"
            f" memcap {meta.get('memcap_mb', 0):.0f} MiB/rep",
            "",
        ]
    loads = [x for r in records for x in (r.get("load_start") or [])[:1]]
    if loads:
        lines += [
            f"1-min load average across reps: min {min(loads):.2f}, max {max(loads):.2f}"
            " (CPU time is the primary speed metric because wall time on a loaded"
            " host measures the neighbours too).",
            "",
        ]

    lines += ["## Status", ""]
    rows = []
    for cell in cells:
        row = [_cell_name(cell)]
        row += [_status_summary(groups.get((cell, lib), [])) for lib in libs]
        for comp in comparators:
            g, c = groups.get((cell, GAMFIT), []), groups.get((cell, comp), [])
            if len(_ok(g)) < len(_ok(c)):
                losses.append(
                    f"{_cell_name(cell)}: status vs {comp} ({_status_summary(g)})"
                )
        rows.append(row)
    lines += _table(["cell", *libs], rows) + [""]

    for metric, label in SPEED_METRICS:
        unit = " MiB" if metric.endswith("_mb") else " s"
        lines += [f"## {label} (median over ok reps)", ""]
        rows = []
        for cell in cells:
            g = groups.get((cell, GAMFIT), [])
            row = [_cell_name(cell)]
            row += [
                _fmt(_median(groups.get((cell, lib), []), metric), unit) for lib in libs
            ]
            for comp in comparators:
                v = ratio_verdict(g, groups.get((cell, comp), []), metric)
                row.append(v.text)
                if v.loss and not v.status:
                    losses.append(f"{_cell_name(cell)}: {label} vs {comp} {v.text}")
                wins += v.win and not v.status
            rows.append(row)
        lines += _table(["cell", *libs, *(f"vs {c}" for c in comparators)], rows) + [""]

    for metric, label in ACCURACY_METRICS:
        lines += [f"## {label} (mean over ok reps; verdict paired by seed)", ""]
        rows = []
        for cell in cells:
            g = groups.get((cell, GAMFIT), [])
            row = [_cell_name(cell)]
            for lib in libs:
                vals = _values(groups.get((cell, lib), []), metric)
                row.append(_fmt(statistics.fmean(vals) if vals else None))
            for comp in comparators:
                v = paired_verdict(g, groups.get((cell, comp), []), metric)
                row.append(v.text)
                if v.loss and not v.status:
                    losses.append(f"{_cell_name(cell)}: {label} vs {comp} {v.text}")
                wins += v.win and not v.status
            rows.append(row)
        lines += _table(["cell", *libs, *(f"vs {c}" for c in comparators)], rows) + [""]

    lines += [f"## Losses ({len(losses)})", ""]
    lines += [f"- {x}" for x in losses] if losses else ["None."]
    lines += [
        "",
        f"Comparisons gamfit wins (ratio < 1.00 or significant accuracy WIN): {wins}.",
        "",
    ]
    return "\n".join(lines)


def load(paths: Iterable[Path]) -> tuple[list[Record], list[dict[str, Any]]]:
    records: list[Record] = []
    metas: list[dict[str, Any]] = []
    for path in paths:
        if path.is_dir():
            meta = path / "meta.json"
            if meta.exists():
                metas.append(json.loads(meta.read_text()))
            path = path / "records.jsonl"
        with path.open() as fh:
            records += [json.loads(line) for line in fh if line.strip()]
    return records, metas


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument(
        "inputs", nargs="+", type=Path, help="run dirs or records.jsonl files"
    )
    ap.add_argument("--out", type=Path, help="write here instead of stdout")
    args = ap.parse_args(argv)
    records, metas = load(args.inputs)
    text = render(records, metas)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
