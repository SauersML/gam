"""Render calibration records as a report, and the table in docs/pvalues.md.

    python -m bench.pvalue_calibration.report DIR [DIR ...] --out report.md
    python -m bench.pvalue_calibration.report DIR --docs docs/pvalues.md [--check]

One row per (cell, library surface). For the null draws it gives the size at
0.10 / 0.05 / 0.01 with its Monte Carlo standard error
``sqrt(size * (1 - size) / R)``, the Kolmogorov-Smirnov distance of the null
p-values from Uniform(0, 1), and the fraction of reps that produced a usable
p-value at all. For the matched-alternative draws it gives the power at 0.05.

A p-value is calibrated when it is Uniform(0, 1) under its null:
``P(p <= a) = a`` at every ``a``. A conservative p-value (size below nominal,
a point mass near 1) fails that exactly as an anti-conservative one does, so
every check is two-sided. Over ``R`` null reps a calibrated p-value rejects
``Binomial(R, a)`` times at level ``a``, so a row is

- **ANTI-CONSERVATIVE** at ``a`` when its rejection count exceeds that law's
  upper ``1 - FALSE_ALARM / m`` quantile;
- **CONSERVATIVE** at ``a`` when its rejection count falls below the law's
  ``FALSE_ALARM / m`` quantile;
- **NOT UNIFORM** when the Kolmogorov-Smirnov test of its null p-values
  against Uniform(0, 1), over the whole range, has p-value at or below
  ``FALSE_ALARM / m``.

``m`` is the number of checks in the report: two per (row, level) and one KS
per row. A rep that produced no p-value (a fit that raised, a missing row, a
rep the safety net killed) may have been anything, and a size taken over only
the reps that succeeded is biased whenever failing correlates with the data
being extreme. So each check takes the completion of those reps that is worst
for it: a rejection in the upper check, a non-rejection in the lower check,
and for KS whichever of p = 0 or p = 1 moves the empirical CDF furthest from
the uniform one (the supremum distance is maximized by putting every missing
point at one end). A row passes only if it passes every check in its worst
case. The tolerances are the sampling laws of the statistics themselves
(``a +/- ~z * MCSE`` for a size), not hand-picked bands, and the Bonferroni
split keeps the chance that a calibrated harness flags anything at all at or
below ``FALSE_ALARM`` however large the grid.

A row whose usable count is below its rep count lists the reasons in
"Unusable reps". A missing p-value is a defect of that surface, never a skip.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import stats

Record = dict[str, Any]

LEVELS: tuple[float, ...] = (0.10, 0.05, 0.01)
POWER_LEVEL = 0.05
# Chance that a fully calibrated harness run flags one or more rows. It is the
# CI false-alarm rate of the smoke test: one spurious red run in a thousand.
FALSE_ALARM = 1e-3

DOCS_BEGIN = "<!-- BEGIN pvalue_calibration table (generated; do not edit) -->"
DOCS_END = "<!-- END pvalue_calibration table -->"

SURFACE_LABEL = {
    "gamfit.wald": "gamfit Wald (`summary`)",
    "gamfit.lr": "gamfit LR (`smooth_significance`)",
    "gamfit.coef": "gamfit coefficient",
    "pygam.wald": "pyGAM, fixed lam",
    "pygam_gs.wald": "pyGAM, gridsearch",
}


@dataclass(frozen=True)
class Row:
    cell: str
    surface: str
    reps: int
    usable: int
    rejections: tuple[int, ...]
    ks_d: float | None
    ks_p: float | None
    power: float | None
    power_usable: int
    null_p: tuple[float, ...] = ()

    def size(self, i: int) -> float | None:
        return self.rejections[i] / self.usable if self.usable else None

    def mcse(self, i: int) -> float | None:
        s = self.size(i)
        return None if s is None else math.sqrt(s * (1 - s) / self.usable)


def reject_bound(reps: int, level: float, checks: int) -> int:
    """Largest rejection count a calibrated p-value reaches except with prob FALSE_ALARM / checks."""
    return int(stats.binom.ppf(1.0 - FALSE_ALARM / checks, reps, level))


def reject_floor(reps: int, level: float, checks: int) -> int:
    """Smallest rejection count a calibrated p-value stays at or above except with prob FALSE_ALARM / checks."""
    return int(stats.binom.ppf(FALSE_ALARM / checks, reps, level))


def worst_ks_p(r: Row) -> float:
    """KS p-value against Uniform(0, 1) with the unusable reps placed worst for it."""
    arr = np.asarray(r.null_p, dtype=float)
    missing = r.reps - r.usable
    return min(
        float(stats.kstest(np.concatenate([arr, np.full(missing, end)]), "uniform").pvalue)
        for end in (0.0, 1.0)
    )


def n_checks(table: list[Row]) -> int:
    """Checks per report: a lower and an upper size check per level, and one KS, per usable row."""
    return sum(1 for r in table if r.usable) * (2 * len(LEVELS) + 1)


def _cell_sort(cell: str) -> tuple[Any, ...]:
    family, n, null = cell.split("/")
    return (int(n.removeprefix("n=")), family, null)


def rows(records: Iterable[Record]) -> list[Row]:
    by: dict[tuple[str, str], list[Record]] = {}
    for rec in records:
        key = rec.get("key") or f"{rec['family']}/n={rec['n']}/{rec['null']}"
        surfaces = {
            f"{lib}.{s}"
            for lib, ss in (rec.get("expected_surfaces") or {}).items()
            for s in ss
        }
        # A rep lost to the safety net has no expected_surfaces; count it
        # against every surface the cell's other reps expect.
        by.setdefault((key, ""), []).append(rec)
        for s in surfaces:
            by.setdefault((key, s), [])
    out: list[Row] = []
    for (key, surface), _ in sorted(by.items(), key=lambda kv: (_cell_sort(kv[0][0]), kv[0][1])):
        if not surface:
            continue
        recs = by[(key, "")]
        null_p = [
            r["p"]["null"][surface]
            for r in recs
            if surface in ((r.get("p") or {}).get("null") or {})
        ]
        alt_p = [
            r["p"]["alt"][surface]
            for r in recs
            if surface in ((r.get("p") or {}).get("alt") or {})
        ]
        arr = np.asarray(null_p, dtype=float)
        ks = stats.kstest(arr, "uniform") if len(arr) else None
        out.append(
            Row(
                cell=key,
                surface=surface,
                reps=len(recs),
                usable=len(arr),
                rejections=tuple(int(np.sum(arr <= a)) for a in LEVELS),
                ks_d=None if ks is None else float(ks.statistic),
                ks_p=None if ks is None else float(ks.pvalue),
                power=(
                    float(np.mean(np.asarray(alt_p) <= POWER_LEVEL)) if alt_p else None
                ),
                power_usable=len(alt_p),
                null_p=tuple(float(p) for p in arr),
            )
        )
    return out


ANTI = "ANTI-CONSERVATIVE"
CONSERVATIVE = "CONSERVATIVE"
NOT_UNIFORM = "NOT UNIFORM"


def miscalibrated(table: list[Row]) -> list[tuple[Row, str, float | None]]:
    """Every failed check as ``(row, kind, level)``; ``level`` is None for KS.

    Each check uses the completion of the unusable reps that is worst for it
    (see the module docstring).
    """
    checks = n_checks(table)
    flagged: list[tuple[Row, str, float | None]] = []
    for r in table:
        if not r.usable:
            continue
        unusable = r.reps - r.usable
        for i, a in enumerate(LEVELS):
            if r.rejections[i] + unusable > reject_bound(r.reps, a, checks):
                flagged.append((r, ANTI, a))
            if r.rejections[i] < reject_floor(r.reps, a, checks):
                flagged.append((r, CONSERVATIVE, a))
        if worst_ks_p(r) <= FALSE_ALARM / checks:
            flagged.append((r, NOT_UNIFORM, None))
    return flagged


def _size(r: Row, i: int) -> str:
    s, e = r.size(i), r.mcse(i)
    if s is None or e is None:
        return "-"
    return f"{s:.3f} ± {e:.3f}"


def _table(header: list[str], body: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return lines


def _flag(kind: str, level: float | None) -> str:
    return f"**{kind}**" + ("" if level is None else f" at {level:g}")


def calibration_table(records: list[Record]) -> str:
    """The generated block of docs/pvalues.md, also the core of report.md."""
    table = rows(records)
    bad: dict[tuple[str, str], list[str]] = {}
    for r, kind, a in miscalibrated(table):
        bad.setdefault((r.cell, r.surface), []).append(_flag(kind, a))
    body = []
    for r in table:
        if not r.usable:
            verdict = "**NO P-VALUE**"
        elif (r.cell, r.surface) in bad:
            verdict = "; ".join(bad[(r.cell, r.surface)])
        else:
            verdict = "calibrated"
        if r.usable < r.reps:
            verdict += f"; {r.reps - r.usable} unusable"
        body.append(
            [
                r.cell,
                SURFACE_LABEL.get(r.surface, r.surface),
                f"{r.usable}/{r.reps}",
                _size(r, 0),
                _size(r, 1),
                _size(r, 2),
                "-" if r.ks_d is None else f"{r.ks_d:.3f} ({r.ks_p:.2g})",
                "-" if r.power is None else f"{r.power:.3f}",
                verdict,
            ]
        )
    checks = n_checks(table)
    lines = [
        f"{len(table)} rows, {checks} two-sided checks (size at each level, and KS), "
        "family-wise false-alarm rate "
        f"{FALSE_ALARM:g} (Bonferroni: each check at {FALSE_ALARM / max(checks, 1):.2g}).",
        "",
        *_table(
            [
                "cell",
                "surface",
                "usable",
                "size@0.10",
                "size@0.05",
                "size@0.01",
                "KS D (p)",
                "power@0.05",
                "verdict",
            ],
            body,
        ),
    ]
    return "\n".join(lines) + "\n"


def unusable(records: list[Record]) -> list[str]:
    """One line per (cell, surface, reason) with its count and example seeds."""
    reasons: dict[tuple[str, str, str], list[int]] = {}
    for rec in records:
        key = rec.get("key") or f"{rec['family']}/n={rec['n']}/{rec['null']}"
        seed = int(rec["seed"])
        if rec.get("status") not in ("ok", "error"):
            reasons.setdefault((key, "*", f"rep {rec['status']}"), []).append(seed)
            continue
        for where, why in (rec.get("missing") or {}).items():
            first = why.splitlines()[0][:160]
            reasons.setdefault((key, where, first), []).append(seed)
        for where, tb in (rec.get("errors") or {}).items():
            last = [ln for ln in tb.strip().splitlines() if ln.strip()][-1][:160]
            reasons.setdefault((key, where, last), []).append(seed)
    return [
        f"- `{key}` {where}: {len(seeds)}x, seeds {seeds[:5]}: {why}"
        for (key, where, why), seeds in sorted(
            reasons.items(), key=lambda kv: (_cell_sort(kv[0][0]), kv[0][1])
        )
    ]


def render(records: list[Record], meta: dict[str, Any] | None = None) -> str:
    meta = meta or {}
    runs = meta.get("invocations") or []
    head = runs[-1] if runs else {}
    lines = [
        "# p-value calibration report",
        "",
        f"- plan: `{meta.get('plan', '?')}`; records: {len(records)}",
        f"- git sha: `{head.get('git_sha')}`; host: `{head.get('host')}` "
        f"({head.get('nproc')} CPUs); jobs: {head.get('jobs')}",
        f"- versions: {', '.join(meta.get('lib_versions') or []) or '?'}",
        f"- {meta.get('safety_net', '')}",
        "",
        "## Calibration",
        "",
        calibration_table(records),
        "## Miscalibrated",
        "",
    ]
    table = rows(records)
    checks = n_checks(table)
    flagged = miscalibrated(table)
    for r, kind, a in flagged:
        holes = r.reps - r.usable
        if a is None:
            lines.append(
                f"- `{r.cell}` {r.surface}: {kind}: KS D {r.ks_d:.3f} over {r.usable} "
                f"usable reps, worst-case KS p {worst_ks_p(r):.2g} with {holes} "
                f"unusable placed at 0 or 1; the check fires at {FALSE_ALARM / checks:.2g}"
            )
            continue
        i = LEVELS.index(a)
        if kind == ANTI:
            bound = f"a calibrated p-value exceeds {reject_bound(r.reps, a, checks)}"
            count = f"{r.rejections[i]} rejections plus {holes} unusable"
        else:
            bound = f"a calibrated p-value falls below {reject_floor(r.reps, a, checks)}"
            count = f"{r.rejections[i]} rejections (unusable counted as none)"
        lines.append(
            f"- `{r.cell}` {r.surface}: {kind} at {a:g}: {count} of {r.reps} "
            f"(size {r.size(i):.3f} over {r.usable} usable); {bound} only with "
            "the stated false-alarm probability"
        )
    if not flagged:
        lines.append("None.")
    lines += ["", "## Unusable reps", ""]
    lines += unusable(records) or ["None."]
    return "\n".join(lines) + "\n"


def splice_docs(page: str, block: str) -> str:
    pattern = re.compile(re.escape(DOCS_BEGIN) + r".*?" + re.escape(DOCS_END), re.S)
    if not pattern.search(page):
        raise ValueError(f"page has no {DOCS_BEGIN!r} ... {DOCS_END!r} block")
    return pattern.sub(lambda _: f"{DOCS_BEGIN}\n\n{block}\n{DOCS_END}", page)


def docs_block(records: list[Record], meta: dict[str, Any]) -> str:
    runs = meta.get("invocations") or []
    head = runs[-1] if runs else {}
    return (
        f"Plan `{meta.get('plan')}` at git `{(head.get('git_sha') or '?')[:12]}`, "
        f"{', '.join(meta.get('lib_versions') or [])}.\n\n"
        + calibration_table(records)
    )


def load(paths: Iterable[Path]) -> tuple[list[Record], dict[str, Any]]:
    records: list[Record] = []
    meta: dict[str, Any] = {}
    for p in paths:
        rp = p / "records.jsonl"
        records += [json.loads(ln) for ln in rp.read_text().splitlines() if ln.strip()]
        mp = p / "meta.json"
        if mp.exists():
            meta = json.loads(mp.read_text())
    return records, meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path, help="run output directories")
    ap.add_argument("--out", type=Path, help="write report.md here (default: stdout)")
    ap.add_argument("--docs", type=Path, help="rewrite the generated block of this page")
    ap.add_argument(
        "--check",
        action="store_true",
        help="with --docs: exit 1 if the page's block is not what the records give",
    )
    args = ap.parse_args(argv)
    records, meta = load(args.dirs)
    if args.docs is not None:
        page = args.docs.read_text()
        new = splice_docs(page, docs_block(records, meta))
        if args.check:
            if new != page:
                print(f"{args.docs} is stale; regenerate it with --docs", file=sys.stderr)
                return 1
            return 0
        args.docs.write_text(new)
        return 0
    text = render(records, meta)
    if args.out is None:
        sys.stdout.write(text)
    else:
        args.out.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
