"""Failure classification and clustering for convergence-fuzz records.

    cd bench && python -m convergence_fuzz.triage DIR [DIR_AFTER]

A rep *fails* when any of these holds (``failure_causes`` returns one label
per failure, most fundamental first):

``hang`` / ``memcap`` / ``crash``
    the harness safety net killed the worker, or it died without a RESULT;
``raise:<phase>:<Type>: <message head>``
    a phase raised (``fit``, ``summary``, ``predict``, ``predict_exact``,
    ``interval``, ``refit``, ``refit_summary``); digits in the message are folded to ``#`` so one root
    cause with different sizes clusters as one label;
``uncertified:<fit|refit>:<outer kind>/<inner status>``
    the fit returned without a convergence certificate that says certified;
``nonfinite:<what>``
    a prediction, interval bound or criterion value is not finite. A log-link
    posterior mean ``exp(eta + Var(eta)/2)`` whose exact value is past
    ``DBL_MAX`` (the worker checks it from the fit's own affine design and
    conditional covariance, ``pred_nonfinite_exact``) is ``+inf`` correctly
    rounded, not a failure;
``reml_mismatch:<fit|refit>_worse``
    both fits certified but their REML/LAML costs differ by more than
    ``reml_tolerance`` - the same model on permuted rows and affinely
    rescaled covariates has the same optimum, so one search stopped short.

With two directories it prints the before/after table by cause.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PHASES = (
    "import",
    "fit",
    "summary",
    "predict",
    "predict_exact",
    "interval",
    "refit",
    "refit_summary",
)
# REML/LAML costs are sums over n observations; two certified optima of the
# same problem agree to the certificate's own stationarity accuracy, far
# below this relative gap (see README "REML comparison").
REML_RTOL = 1e-6
REML_ATOL = 1e-6


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):  # the worker writes NaN/Inf as their repr
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _fold(msg: str) -> str:
    head = msg.strip().splitlines()[0] if msg.strip() else ""
    head = re.sub(r"0x[0-9a-fA-F]+", "#", head)
    head = re.sub(r"[-+]?\d+(\.\d+)?([eE][-+]?\d+)?", "#", head)
    return head[:140]


def reml_tolerance(a: float, b: float) -> float:
    return REML_ATOL + REML_RTOL * max(abs(a), abs(b))


def _cert_label(info: dict[str, Any]) -> str:
    conv = info.get("convergence") or {}
    outer = conv.get("outer") or {}
    return f"{outer.get('kind', '?')}/{conv.get('inner_status', '?')}"


def _fit_causes(tag: str, info: dict[str, Any] | None) -> list[str]:
    if info is None:
        return []
    out: list[str] = []
    if info.get("convergence") is None:
        out.append(f"uncertified:{tag}:no_certificate")
    elif info.get("certified") is not True:
        out.append(f"uncertified:{tag}:{_cert_label(info)}")
    score = _num(info.get("reml_score"))
    if info.get("reml_score") is None:
        out.append(f"nonfinite:{tag}_reml_score({info.get('reml_score_unavailable')})")
    elif score is None or not math.isfinite(score):
        out.append(f"nonfinite:{tag}_reml_score")
    return out


def failure_causes(rec: dict[str, Any]) -> list[str]:
    status = rec.get("status")
    if status == "timeout":
        return ["hang"]
    if status in ("memcap", "crash"):
        return [str(status)]
    causes: list[str] = []
    errors: dict[str, str] = rec.get("errors") or {}
    for phase in PHASES:
        if phase in errors:
            causes.append(f"raise:{phase}:{_fold(errors[phase])}")
    fit_info = rec if "reml_score" in rec or "convergence" in rec else None
    refit_info = rec.get("refit")
    causes += _fit_causes("fit", fit_info)
    causes += _fit_causes("refit", refit_info)
    if rec.get("pred_finite") is False and rec.get("pred_nonfinite_exact") is not True:
        causes.append("nonfinite:predict")
    if rec.get("interval_finite") is False:
        causes.append("nonfinite:interval")
    if fit_info is not None and refit_info is not None:
        a = _num(fit_info.get("reml_score"))
        b = _num(refit_info.get("reml_score"))
        if (
            fit_info.get("certified") is True
            and refit_info.get("certified") is True
            and a is not None
            and b is not None
            and math.isfinite(a)
            and math.isfinite(b)
            and abs(a - b) > reml_tolerance(a, b)
        ):
            causes.append(f"reml_mismatch:{'fit' if a > b else 'refit'}_worse")
    return causes


def primary(rec: dict[str, Any]) -> str | None:
    causes = rec.get("causes")
    if causes is None:
        causes = failure_causes(rec)
    return causes[0] if causes else None


def load(path: Path) -> list[dict[str, Any]]:
    recs = [
        json.loads(line) for line in (path / "records.jsonl").read_text().splitlines()
    ]
    for rec in recs:
        rec["causes"] = failure_causes(rec)
    return recs


def _key(rec: dict[str, Any]) -> str:
    return f"case{rec['case']}/{rec['family']}/n{rec['n']}"


def render(records: list[dict[str, Any]]) -> str:
    total = len(records)
    failed = [r for r in records if r.get("causes")]
    by_cause: Counter[str] = Counter(str(primary(r)) for r in failed)
    lines = [
        "# Convergence fuzz report",
        "",
        f"{total} reps, {len(failed)} failed ({100.0 * len(failed) / max(total, 1):.2f}%).",
        "",
        "## Failures by primary cause",
        "",
        "| cause | reps | rate | families | n | example |",
        "|---|---:|---:|---|---|---|",
    ]
    for cause, count in by_cause.most_common():
        members = [r for r in failed if primary(r) == cause]
        fams = ",".join(sorted({str(r["family"]) for r in members}))
        ns = ",".join(str(v) for v in sorted({int(r["n"]) for r in members}))
        lines.append(
            f"| `{cause}` | {count} | {100.0 * count / max(total, 1):.2f}% "
            f"| {fams} | {ns} | {_key(members[0])} |"
        )
    all_causes: Counter[str] = Counter(c for r in failed for c in r["causes"])
    lines += ["", "## Every cause (a rep can carry several)", ""]
    lines += [f"- `{c}`: {k}" for c, k in all_causes.most_common()]
    walls = sorted(float(r.get("proc_wall_s", 0.0)) for r in records)
    if walls:
        lines += [
            "",
            "## Wall time per rep (both fits)",
            "",
            f"median {walls[len(walls) // 2]:.2f}s, max {walls[-1]:.2f}s",
        ]
    return "\n".join(lines) + "\n"


def before_after(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> str:
    b: Counter[str] = Counter(str(primary(r)) for r in before if r.get("causes"))
    a: Counter[str] = Counter(str(primary(r)) for r in after if r.get("causes"))
    nb, na = max(len(before), 1), max(len(after), 1)
    lines = [
        "| cause | before | after |",
        "|---|---:|---:|",
    ]
    for cause in sorted(set(b) | set(a), key=lambda c: (-b[c], c)):
        lines.append(
            f"| `{cause}` | {b[cause]} ({100.0 * b[cause] / nb:.2f}%) "
            f"| {a[cause]} ({100.0 * a[cause] / na:.2f}%) |"
        )
    tb, ta = sum(b.values()), sum(a.values())
    lines.append(
        f"| **total** | **{tb} / {len(before)} ({100.0 * tb / nb:.2f}%)** "
        f"| **{ta} / {len(after)} ({100.0 * ta / na:.2f}%)** |"
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    if len(argv) == 1:
        print(render(load(Path(argv[0]))))
        return 0
    if len(argv) == 2:
        print(before_after(load(Path(argv[0])), load(Path(argv[1]))))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
