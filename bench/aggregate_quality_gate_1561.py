#!/usr/bin/env python
"""#1561 whole-suite quality meta-gate aggregator.

Consumes the output of a `tests/quality` run (nextest with
`--success-output final --failure-output final`, or any log carrying the
lines) and recomputes the acceptance test the issue is gated on: a paired,
one-sided Wilcoxon signed-rank test of whether GAM's per-test objective error
distribution is significantly BETTER than the mature reference software's.

It reads the canonical `[QUALITY_PAIR] ...` telemetry emitted by
`gam_test_support::reference::QualityPair::line` (unambiguous, full-precision),
NOT the heterogeneous human `eprintln!` tokens (`gam_test_rmse` vs `gam:{}` vs
`gam_rmse_truth` ...) that make the scrape unreliable. One row per test.

Per-metric signed effect (negative == GAM better, uniformly across metric kinds):
    lower_is_better : effect = log(gam / reference)
    higher_is_better: effect = log(reference / gam)

The experimental unit is the actual libtest `case`, emitted by the test thread,
not a channel label or metric. Its signed effect is the median of its metric
effects. Acceptance: one-sided Wilcoxon signed-rank on those per-case medians,
H1 = median effect < 0 (GAM better), reported overall and per category with a
Benjamini-Hochberg-adjusted per-category view.

#2395 power view: a pair emitted from a PAIRED fold panel carries its own
uncertainty (`folds=`, `effect_sem=`, `verdict=` ...). Without those columns the
rank test must weight a pair whose gap is a hundred times its own noise exactly
like a pair whose gap IS its noise, which is how ~30 sign-flipping near-ties came
to dilute this aggregate. The power section below reports, per pair, whether the
gap is RESOLVED at all and how many standard errors the pairing itself bought.
It is a diagnostic: the CLOSURE verdict is unchanged and still runs on every
scored pair.

Usage:
  python bench/aggregate_quality_gate_1561.py quality_run.log
  cargo nextest run -p gam --test quality --no-fail-fast \
      --success-output final --failure-output final 2>&1 | \
      python bench/aggregate_quality_gate_1561.py -
"""
from __future__ import annotations

import csv
import math
import os
import re
import sys
from collections import defaultdict
from statistics import median

_LINE = re.compile(
    r"\[QUALITY_PAIR\]\s+"
    r"category=(?P<category>\S+)\s+"
    r"test=(?P<test>\S+)\s+"
    r"metric=(?P<metric>\S+)\s+"
    r"gam=(?P<gam>\S+)\s+"
    r"reference=(?P<reference>\S+)\s+"
    r"reference_value=(?P<reference_value>\S+)\s+"
    r"lower_is_better=(?P<lower>true|false)\s+"
    r"case=(?P<case>\S+)"
)

# #2395 paired-panel columns. Absent on single-shot (unpaired) pairs.
_PAIRED = re.compile(
    r"folds=(?P<folds>\d+)\s+"
    r"effect_mean=(?P<effect_mean>\S+)\s+"
    r"effect_sd=(?P<effect_sd>\S+)\s+"
    r"effect_sem=(?P<effect_sem>\S+)\s+"
    r"effect_size=(?P<effect_size>\S+)\s+"
    r"unpaired_sem=(?P<unpaired_sem>\S+)\s+"
    r"gam_wins=(?P<gam_wins>\d+)\s+"
    r"verdict=(?P<verdict>\S+)"
)


def _parse(stream) -> list[dict]:
    rows: dict[tuple[str, str, str, str, str], dict] = {}
    categories = {}
    for raw in stream:
        m = _LINE.search(raw)
        if m is None:
            if "[QUALITY_PAIR]" in raw:
                raise ValueError("malformed quality telemetry or missing libtest case; rerun the suite")
            continue
        gam = float(m["gam"])
        ref = float(m["reference_value"])
        category = categories.setdefault(m["case"], m["category"])
        if category != m["category"]:
            raise ValueError(f"one experimental case reported multiple categories: {m['case']}")
        lower = m["lower"] == "true"
        pm = _PAIRED.search(raw, m.end())
        paired = None
        if pm is not None:
            paired = {
                "folds": int(pm["folds"]),
                "effect_mean": float(pm["effect_mean"]),
                "effect_sd": float(pm["effect_sd"]),
                "effect_sem": float(pm["effect_sem"]),
                "effect_size": float(pm["effect_size"]),
                "unpaired_sem": float(pm["unpaired_sem"]),
                "gam_wins": int(pm["gam_wins"]),
                "verdict": pm["verdict"],
            }
        key = (m["category"], m["case"], m["test"], m["metric"], m["reference"])
        row = {
            "category": m["category"],
            "case": m["case"],
            "test": m["test"],
            "metric": m["metric"],
            "gam": gam,
            "reference": m["reference"],
            "reference_value": ref,
            "lower_is_better": lower,
            "paired": paired,
        }
        if key in rows and rows[key] != row:
            raise ValueError(f"conflicting repeated quality observation: {key}")
        rows[key] = row
    return list(rows.values())


_ANSI = re.compile(r"\x1b\[[0-9;]*m")

# nextest's per-test result line: `<STATUS> [<dur>] <binary-id> <test-path>`.
# STATUS is right-padded with spaces; FLAKY = passed on retry (NOT a failure);
# intermediate `TRY N FAIL` retry lines don't match (they start with TRY, not
# the status). Signal aborts appear as SIGSEGV/SIGABRT/ABORT/TIMEOUT/LEAK.
_NEXTEST = re.compile(
    r"^\s*(?P<status>PASS|FLAKY|FAIL|LEAK|TIMEOUT|SIGSEGV|SIGABRT|ABORT)\s+"
    r"\[[^\]]*\]\s+\S*quality\S*\s+(?P<path>\S+)"
)
_NEXTEST_PASS = {"PASS", "FLAKY"}

# `reference-quality.yml` does NOT run nextest: it builds the grouped `quality`
# binary once and executes each libtest case directly, writing every outcome to
# `quality_results.tsv` (columns: idx, outcome, cause, test, rc, ...). So the
# nextest scrape above finds nothing there and the attrition guard silently
# switches itself off — which is how 46 GAM_ERROR tests (38 of them emitting no
# pair at all) sat outside the significance set with nothing in the report
# saying so. That TSV is the authoritative execution record for that runner, so
# read it directly when it is available.
_TSV_PASS = {"PASS"}


def _parse_outcome_tsv(path: str) -> dict[str, dict]:
    """Per-category executed/failed counts from a `quality_results.tsv`.

    Same return shape as [`_parse_nextest`]. Anything whose outcome is not PASS
    counts as a failure for attrition purposes, including REF_ERROR: a reference
    tool that could not run is still a test that produced no comparable pair,
    and the report distinguishes the causes by name.
    """
    by_cat: dict[str, dict] = defaultdict(
        lambda: {"executed": 0, "failed": 0, "failed_paths": [], "cases": set()}
    )
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            test = (row.get("test") or "").strip()
            outcome = (row.get("outcome") or "").strip()
            if not test or not outcome:
                continue
            category = test.split("::", 1)[0]
            rec = by_cat[category]
            rec["executed"] += 1
            rec["cases"].add(test)
            if outcome not in _TSV_PASS:
                rec["failed"] += 1
                cause = (row.get("cause") or "").strip()
                rec["failed_paths"].append(
                    f"{outcome}{f'/{cause}' if cause else ''} {test}"
                )
    return dict(by_cat)


def _default_outcome_tsv(log_path: str) -> str | None:
    """`quality_results.tsv` sitting beside the log, as the workflow writes it."""
    if log_path == "-":
        return None
    sibling = os.path.join(os.path.dirname(os.path.abspath(log_path)), "quality_results.tsv")
    return sibling if os.path.exists(sibling) else None



def _parse_nextest(lines: list[str]) -> dict[str, dict]:
    """Per-category executed/failed counts from nextest PASS/FAIL lines.

    The category is the first `::`-component of the test path within the
    `quality` binary. A test that panics/refuses BEFORE its emit line shows up
    here as FAIL with no matching [QUALITY_PAIR] — that silent attrition is
    exactly what must stay visible so a crashing test cannot quietly drop out of
    the significance set.
    """
    by_cat: dict[str, dict] = defaultdict(
        lambda: {"executed": 0, "failed": 0, "failed_paths": [], "cases": set()}
    )
    for raw in lines:
        m = _NEXTEST.match(_ANSI.sub("", raw))
        if m is None:
            continue
        path = m["path"]
        category = path.split("::", 1)[0]
        rec = by_cat[category]
        rec["executed"] += 1
        rec["cases"].add(path)
        if m["status"] not in _NEXTEST_PASS:
            rec["failed"] += 1
            rec["failed_paths"].append(f"{m['status']} {path}")
    return dict(by_cat)


def _effect(row: dict) -> float | None:
    gam, ref = row["gam"], row["reference_value"]
    if not (math.isfinite(gam) and math.isfinite(ref)) or gam <= 0.0 or ref <= 0.0:
        return None
    effect = math.log(gam) - math.log(ref)
    return effect if row["lower_is_better"] else -effect


def _wilcoxon_less(effects: list[float]) -> tuple[float, float, float]:
    """One-sided Wilcoxon signed-rank, H1: median < 0. Returns (W+, z, p).

    Normal approximation with continuity + tie correction (adequate for the
    suite's ~100 pairs). Zero-difference pairs are dropped (Wilcoxon convention).
    """
    nz = [e for e in effects if e != 0.0]
    n = len(nz)
    if n == 0:
        return (0.0, 0.0, 1.0)
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(r for e, r in zip(nz, ranks) if e > 0.0)
    mean = n * (n + 1) / 4.0
    # tie correction
    tie_term = 0.0
    from collections import Counter

    for c in Counter(abs(e) for e in nz).values():
        tie_term += c**3 - c
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if var <= 0.0:
        return (w_plus, float("nan"), float("nan"))
    # H1 median<0 => W+ small. Continuity-correct toward the mean.
    z = (w_plus + 0.5 - mean) / math.sqrt(var)
    p = 0.5 * math.erfc(-z / math.sqrt(2.0))  # P(Z <= z) lower tail
    return (w_plus, z, p)


def _summarize(label: str, rows: list[dict]) -> dict:
    by_case = defaultdict(list)
    invalid_cases = set()
    for row in rows:
        key = (row["category"], row["case"])
        e = _effect(row)
        if e is None:
            invalid_cases.add(key)
        else:
            by_case[key].append(e)
    # One independent experimental unit per libtest case. A case with an
    # invalid metric cannot silently contribute only its surviving metrics.
    effects = [median(es) for case, es in by_case.items() if case not in invalid_cases]
    wins = losses = ties = 0
    for e in effects:
        if e < 0.0:
            wins += 1
        elif e > 0.0:
            losses += 1
        else:
            ties += 1
    w_plus, z, p = _wilcoxon_less(effects) if effects else (0.0, float("nan"), float("nan"))
    return {
        "label": label,
        "n": len(set(by_case) | invalid_cases),
        "pairs": len(rows),
        "scored": len(effects),
        "dropped_nonfinite": len(invalid_cases),
        "gam_wins": wins,
        "reference_wins": losses,
        "ties": ties,
        "median_log_ratio": median(effects) if effects else float("nan"),
        "wilcoxon_z": z,
        "p_one_sided_gam_better": p,
    }


def main() -> None:
    if len(sys.argv) not in (2, 3):
        raise SystemExit(__doc__)
    src = sys.stdin if sys.argv[1] == "-" else open(sys.argv[1])
    with src if src is not sys.stdin else _nullctx(src):
        lines = src.readlines()
    rows = _parse(lines)
    # Execution record, for the silent-attrition guard. nextest output is the
    # historical source; the reference-quality workflow instead writes a
    # `quality_results.tsv`, passed explicitly or found beside the log.
    nextest = _parse_nextest(lines)
    if not nextest:
        tsv = sys.argv[2] if len(sys.argv) == 3 else _default_outcome_tsv(sys.argv[1])
        if tsv:
            nextest = _parse_outcome_tsv(tsv)
            print(f"execution record: {tsv}")
    if not rows:
        raise SystemExit(
            "no [QUALITY_PAIR] lines found. Ensure the quality tests emit "
            "QualityPair::line and the run used --success-output final."
        )

    overall = _summarize("OVERALL", rows)
    by_cat = defaultdict(list)
    for row in rows:
        by_cat[row["category"]].append(row)
    cats = {c: _summarize(c, rs) for c, rs in sorted(by_cat.items())}

    # Benjamini-Hochberg over per-category one-sided p-values.
    valid = [(c, s["p_one_sided_gam_better"]) for c, s in cats.items() if math.isfinite(s["p_one_sided_gam_better"])]
    valid.sort(key=lambda kv: kv[1])
    m = len(valid)
    bh = {}
    prev = 1.0
    for rank, (c, p) in enumerate(reversed(valid), start=1):
        idx = m - rank + 1
        prev = min(prev, p * m / idx)
        bh[c] = prev

    print("=== #1561 whole-suite quality meta-gate ===")
    print(
        f"{'category':<12} {'n':>4} {'scored':>6} {'GAMwin':>6} {'REFwin':>6} "
        f"{'medlogR':>9} {'p(1-sided)':>11} {'p_BH':>9}"
    )

    def _fmt(s: dict, bh_p=None) -> str:
        p = s["p_one_sided_gam_better"]
        pbh = "" if bh_p is None else f"{bh_p:>9.4f}"
        return (
            f"{s['label']:<12} {s['n']:>4} {s['scored']:>6} {s['gam_wins']:>6} "
            f"{s['reference_wins']:>6} {s['median_log_ratio']:>9.4f} "
            f"{p:>11.4f} {pbh}"
        )

    for c, s in cats.items():
        print(_fmt(s, bh.get(c)))
    print("-" * 72)
    print(_fmt(overall))
    print()
    recorded_cases = {case for record in nextest.values() for case in record["cases"]}
    unrecorded_cases = {row["case"] for row in rows} - recorded_cases
    execution_valid = (
        bool(nextest)
        and not unrecorded_cases
        and all(record["failed"] == 0 for record in nextest.values())
    )
    verdict = (
        overall["p_one_sided_gam_better"] < 0.05
        and overall["median_log_ratio"] < 0.0
        and overall["dropped_nonfinite"] == 0
        and execution_valid
    )
    print(
        f"CLOSURE (per-case p<0.05, negative median, valid metrics, successful recorded execution): "
        f"{'PASS' if verdict else 'FAIL'} "
        f"(p={overall['p_one_sided_gam_better']:.4f}, "
        f"median log(gam/ref)={overall['median_log_ratio']:.4f}, "
        f"wins {overall['gam_wins']} / losses {overall['reference_wins']})"
    )
    if overall["dropped_nonfinite"]:
        print(f"INVALID: {overall['dropped_nonfinite']} case(s) have nonfinite/nonpositive metrics; closure blocked.")
    print(f"Experimental units: {overall['scored']} cases from {len(rows)} metric pairs.")
    if not execution_valid:
        print("INCOMPLETE: absent execution record or failed cases; closure blocked.")
        for case in sorted(unrecorded_cases):
            print(f"  unrecorded emitting case: {case}")

    # Attrition: cross-reference emitted pairs against nextest execution so a
    # test that crashed/refused BEFORE its emit line is visible, not absorbed.
    if nextest:
        emitters = defaultdict(set)
        for row in rows:
            emitters[row["category"]].add(row["case"])
        print("\n--- execution vs emission (silent-attrition guard) ---")
        print(f"{'category':<12} {'executed':>8} {'failed':>7} {'emitting':>9}")
        total_failed_no_pair = []
        for c in sorted(set(nextest) | set(emitters)):
            nx = nextest.get(c, {"executed": 0, "failed": 0, "failed_paths": []})
            print(
                f"{c:<12} {nx['executed']:>8} {nx['failed']:>7} {len(emitters.get(c, set())):>9}"
            )
            for fp in nx["failed_paths"]:
                case = fp.split(" ", 1)[-1]
                if case not in emitters.get(c, set()):
                    total_failed_no_pair.append(fp)
        if total_failed_no_pair:
            print(
                f"\nSILENT ATTRITION: {len(total_failed_no_pair)} test(s) failed BEFORE "
                f"emitting a pair (excluded from the significance set):"
            )
            for fp in total_failed_no_pair:
                print(f"  {fp}")
        else:
            print("\nNo silent attrition: every failed test still emitted its pair.")
    else:
        print(
            "\nNOTE: no nextest PASS/FAIL lines found in input — attrition guard "
            "inactive. Pipe the full `nextest run` output (not just filtered pairs) "
            "so crashing tests are visible."
        )

    _power_report(rows)

    # Worst offenders: the tests GAM loses by the most (largest positive effect).
    scored = [(r, _effect(r)) for r in rows]
    losers = sorted(
        ((r, e) for r, e in scored if e is not None and e > 0.0),
        key=lambda re: re[1],
        reverse=True,
    )[:15]
    if losers:
        print("\nTop tests GAM loses (largest log(gam/ref) > 0):")
        for r, e in losers:
            print(
                f"  {e:+.4f}  {r['category']}/{r['test']} "
                f"[{r['metric']}] gam={r['gam']:.5g} vs {r['reference']}={r['reference_value']:.5g}"
            )


def _resolution(paired: dict) -> float:
    """|effect_mean| / effect_sem: standard errors separating the two tools.

    Below ~1 the pair carries essentially no information about which tool is
    better; that is the near-tie regime #2395 diagnosed. An exactly-zero spread
    (bit-identical folds) is infinitely resolved and reported as such.
    """
    sem = paired["effect_sem"]
    if sem <= 0.0:
        return float("inf") if paired["effect_mean"] != 0.0 else 0.0
    return abs(paired["effect_mean"]) / sem


def _power_report(rows: list[dict]) -> None:
    """#2395: per-pair resolution, and how much the pairing itself bought."""
    panels = [r for r in rows if r["paired"] is not None]
    print("\n--- #2395 paired-panel power ---")
    if not panels:
        print(
            "No pair carries fold telemetry. Every comparison here is a single "
            "point estimate with no measured uncertainty, so the rank test cannot "
            "distinguish a resolved gap from a coin flip. Route K-fold/K-seed "
            "sites through QualityPair::paired."
        )
        return

    buckets = defaultdict(list)
    for r in panels:
        buckets[r["paired"]["verdict"]].append(r)
    print(
        f"{len(panels)} of {len(rows)} pairs carry a paired fold panel: "
        f"{len(buckets['gam_resolved_better'])} resolved BETTER, "
        f"{len(buckets['gam_resolved_worse'])} resolved WORSE, "
        f"{len(buckets['unresolved_tie'])} unresolved."
    )

    gains = sorted(
        r["paired"]["unpaired_sem"] / r["paired"]["effect_sem"]
        for r in panels
        if r["paired"]["effect_sem"] > 0.0
    )
    if gains:
        print(
            "Pairing gain (unpaired SEM / paired SEM) over these panels: "
            f"min {gains[0]:.2f}x, median {gains[len(gains) // 2]:.2f}x, "
            f"max {gains[-1]:.2f}x. Values above 1 are the common fold-draw swing "
            "that comparing two separately-averaged arms would have paid for."
        )

    print(
        f"\n{'pair':<58} {'K':>3} {'effect%':>9} {'sem%':>7} {'|e|/sem':>8} "
        f"{'d_z':>7} {'wins':>6}  verdict"
    )
    for r in sorted(panels, key=lambda r: -_resolution(r["paired"])):
        pd = r["paired"]
        label = f"{r['category']}/{r['test']}"
        print(
            f"{label[:58]:<58} {pd['folds']:>3} "
            f"{100.0 * math.expm1(pd['effect_mean']):>+9.3f} "
            f"{100.0 * math.expm1(pd['effect_sem']):>7.3f} "
            f"{_resolution(pd):>8.2f} {pd['effect_size']:>+7.2f} "
            f"{pd['gam_wins']:>3}/{pd['folds']:<2}  {pd['verdict']}"
        )

    worse = buckets["gam_resolved_worse"]
    if worse:
        print(
            f"\nREAL FINDINGS: {len(worse)} pair(s) where gam is worse by more than "
            "the paired fold spread can explain. These are systematic, not split "
            "noise, and each deserves its own investigation:"
        )
        for r in worse:
            pd = r["paired"]
            print(
                f"  {r['category']}/{r['test']} [{r['metric']}] vs {r['reference']}: "
                f"{100.0 * math.expm1(pd['effect_mean']):+.3f}% over {pd['folds']} folds "
                f"({_resolution(pd):.1f} SEM, gam won {pd['gam_wins']}/{pd['folds']})"
            )

    unresolved = buckets["unresolved_tie"]
    if unresolved:
        print(
            f"\nDILUTION: {len(unresolved)} paired pair(s) remain unresolved — their "
            "sign is not established even after fold averaging. They still enter the "
            "rank test above with full weight. Raising K on these is the lever that "
            "converts them into evidence:"
        )
        for r in sorted(unresolved, key=lambda r: _resolution(r["paired"])):
            pd = r["paired"]
            print(
                f"  {r['category']}/{r['test']}: "
                f"{100.0 * math.expm1(pd['effect_mean']):+.3f}% +- "
                f"{100.0 * math.expm1(pd['effect_sem']):.3f}% "
                f"({_resolution(pd):.2f} SEM over {pd['folds']} folds)"
            )


class _nullctx:
    def __init__(self, obj):
        self.obj = obj

    def __enter__(self):
        return self.obj

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
