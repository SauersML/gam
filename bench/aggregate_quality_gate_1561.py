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

Per-test signed effect (negative == GAM better, uniformly across metric kinds):
    lower_is_better : effect = log(gam / reference)
    higher_is_better: effect = log(reference / gam)

Acceptance (matches the issue): one-sided Wilcoxon signed-rank on the effects,
H1 = median effect < 0 (GAM better), reported overall and per category with a
Benjamini-Hochberg-adjusted per-category view.

Usage:
  python bench/aggregate_quality_gate_1561.py quality_run.log
  cargo nextest run -p gam --test quality --no-fail-fast \
      --success-output final --failure-output final 2>&1 | \
      python bench/aggregate_quality_gate_1561.py -
"""
from __future__ import annotations

import math
import re
import sys
from collections import defaultdict

_LINE = re.compile(
    r"\[QUALITY_PAIR\]\s+"
    r"category=(?P<category>\S+)\s+"
    r"test=(?P<test>\S+)\s+"
    r"metric=(?P<metric>\S+)\s+"
    r"gam=(?P<gam>\S+)\s+"
    r"reference=(?P<reference>\S+)\s+"
    r"reference_value=(?P<reference_value>\S+)\s+"
    r"lower_is_better=(?P<lower>true|false)"
)


def _parse(stream) -> list[dict]:
    rows: dict[tuple[str, str, str], dict] = {}
    for raw in stream:
        m = _LINE.search(raw)
        if m is None:
            continue
        gam = float(m["gam"])
        ref = float(m["reference_value"])
        lower = m["lower"] == "true"
        # de-dup: a retried/parametrized test may emit the same key twice; last wins.
        key = (m["category"], m["test"], m["metric"])
        rows[key] = {
            "category": m["category"],
            "test": m["test"],
            "metric": m["metric"],
            "gam": gam,
            "reference": m["reference"],
            "reference_value": ref,
            "lower_is_better": lower,
        }
    return list(rows.values())


def _effect(row: dict) -> float | None:
    gam, ref = row["gam"], row["reference_value"]
    if not (math.isfinite(gam) and math.isfinite(ref)) or gam <= 0.0 or ref <= 0.0:
        return None
    ratio = gam / ref if row["lower_is_better"] else ref / gam
    return math.log(ratio)


def _wilcoxon_less(effects: list[float]) -> tuple[float, float, int]:
    """One-sided Wilcoxon signed-rank, H1: median < 0. Returns (W+, z, p).

    Normal approximation with continuity + tie correction (adequate for the
    suite's ~100 pairs). Zero-difference pairs are dropped (Wilcoxon convention).
    """
    nz = [e for e in effects if e != 0.0]
    n = len(nz)
    if n == 0:
        return (0.0, float("nan"), float("nan"))
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
    w_minus = sum(r for e, r in zip(nz, ranks) if e < 0.0)
    mean = n * (n + 1) / 4.0
    # tie correction
    tie_term = 0.0
    from collections import Counter

    for c in Counter(round(abs(e), 12) for e in nz).values():
        tie_term += c**3 - c
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if var <= 0.0:
        return (w_plus, float("nan"), float("nan"))
    # H1 median<0 => W+ small. Continuity-correct toward the mean.
    z = (w_plus + 0.5 - mean) / math.sqrt(var)
    p = 0.5 * math.erfc(-z / math.sqrt(2.0))  # P(Z <= z) lower tail
    return (w_plus, z, p)


def _summarize(label: str, rows: list[dict]) -> dict:
    effects, dropped = [], 0
    wins = losses = ties = 0
    for row in rows:
        e = _effect(row)
        if e is None:
            dropped += 1
            continue
        effects.append(e)
        if e < 0.0:
            wins += 1
        elif e > 0.0:
            losses += 1
        else:
            ties += 1
    w_plus, z, p = _wilcoxon_less(effects) if effects else (0.0, float("nan"), float("nan"))
    median = sorted(effects)[len(effects) // 2] if effects else float("nan")
    return {
        "label": label,
        "n": len(rows),
        "scored": len(effects),
        "dropped_nonfinite": dropped,
        "gam_wins": wins,
        "reference_wins": losses,
        "ties": ties,
        "median_log_ratio": median,
        "wilcoxon_z": z,
        "p_one_sided_gam_better": p,
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    src = sys.stdin if sys.argv[1] == "-" else open(sys.argv[1])
    with src if src is not sys.stdin else _nullctx(src):
        rows = _parse(src)
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
    verdict = (
        overall["p_one_sided_gam_better"] < 0.05
        and overall["median_log_ratio"] < 0.0
    )
    print(
        f"CLOSURE (one-sided p<0.05 AND GAM better on median): "
        f"{'PASS' if verdict else 'FAIL'} "
        f"(p={overall['p_one_sided_gam_better']:.4f}, "
        f"median log(gam/ref)={overall['median_log_ratio']:.4f}, "
        f"wins {overall['gam_wins']} / losses {overall['reference_wins']})"
    )
    if overall["dropped_nonfinite"]:
        print(f"NOTE: {overall['dropped_nonfinite']} pair(s) dropped (nonfinite/nonpositive).")

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


class _nullctx:
    def __init__(self, obj):
        self.obj = obj

    def __enter__(self):
        return self.obj

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
