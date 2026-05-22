#!/usr/bin/env python3
"""Statistical-regression gate for the GAM bench suite.

Catches silent statistical regressions when speed-focused changes corrupt
the final fit. The gate compares two fit-quality summaries against a stored
baseline:

  * ``final_neg_v``: the optimized outer REML/LAML score
    (``fit_result.reml_score`` in the rust contender's model.json).
  * ``edf_per_term``: effective degrees of freedom per smoothing term
    (``fit_result.edf_by_block`` keyed by the term names taken from
    ``resolved_term_spec.smooth_terms[i].name``).

The gate is OPT-IN. Default behaviour is "report and don't fail" so it can
land in CI without breaking existing runs. To make it fail on regression,
set ``BENCH_GATE=strict`` or pass ``--gate strict``.

Tolerances (rationale):

  * ``|delta_neg_v| / |baseline_neg_v| > 1e-3`` -> regression. A REML
    score is the outer objective the smoothing-parameter optimizer
    minimizes; a 0.1 % move under fixed data/folds means a meaningfully
    different optimum, not a numerical jitter.
  * ``|delta_edf| > 0.1`` (absolute, per term) -> regression. edf is on
    the same scale as basis dimension; 0.1 is well above PIRLS/REML
    convergence noise and well below "the fit changed".

Lanes whose model.json lacks both ``reml_score`` and ``edf_by_block``
(e.g. R mgcv lanes, non-GAM contenders, survival lanes that don't surface
the score) are SKIPPED from the gate with a warning, not failed.

Two entry points:

  ``python bench/gate.py scan --models-dir <dir>``
      Walk ``<dir>`` for ``*.model.json`` / ``model_*.json`` files and
      gate each one against ``bench/baselines/<stem>.json``.

  ``python bench/gate.py check-results <results.json>``
      Read a ``run_suite.py`` output JSON. For each ok row that carries
      ``fit_quality``, gate against
      ``bench/baselines/<contender>__<scenario>.json``.

Add ``--update-baseline`` to write/overwrite the baseline files instead of
comparing. Baselines are plain JSON, checked into git under
``bench/baselines/``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

BENCH_DIR = Path(__file__).resolve().parent
BASELINES_DIR = BENCH_DIR / "baselines"

# Tolerances (see module docstring).
REL_TOL_NEG_V = 1e-3
ABS_TOL_EDF = 0.1


def _unwrap_payload(blob: Any) -> dict[str, Any] | None:
    if not isinstance(blob, dict):
        return None
    if "payload" in blob and isinstance(blob["payload"], dict):
        return blob["payload"]
    return blob


def extract_fit_quality(model_json: dict[str, Any] | Path) -> dict[str, Any] | None:
    """Pull ``final_neg_v`` and ``edf_per_term`` from a rust model.json.

    Returns ``None`` if neither field is present. A partially-present
    result (only one of the two) is still returned so the gate can act
    on what it has.
    """
    if isinstance(model_json, (str, Path)):
        try:
            blob = json.loads(Path(model_json).read_text())
        except (OSError, ValueError):
            return None
    else:
        blob = model_json
    payload = _unwrap_payload(blob)
    if payload is None:
        return None
    fit_result = payload.get("fit_result")
    if not isinstance(fit_result, dict):
        return None

    out: dict[str, Any] = {}
    reml = fit_result.get("reml_score")
    if isinstance(reml, (int, float)):
        # ``reml_score`` IS the negative-V outer objective the optimizer
        # minimizes; keep the same sign convention so a "more negative
        # value" means a better fit and the relative-delta gate is symmetric.
        out["final_neg_v"] = float(reml)

    edf_by_block = fit_result.get("edf_by_block")
    spec = payload.get("resolved_term_spec") or {}
    smooth_terms = spec.get("smooth_terms") if isinstance(spec, dict) else None
    if isinstance(edf_by_block, list) and isinstance(smooth_terms, list):
        edf_per_term: dict[str, float] = {}
        for i, val in enumerate(edf_by_block):
            if not isinstance(val, (int, float)):
                continue
            term = smooth_terms[i] if i < len(smooth_terms) else None
            name = None
            if isinstance(term, dict):
                name = term.get("name")
            if not isinstance(name, str) or not name:
                name = f"smooth_{i}"
            # Disambiguate duplicate names (e.g. two anonymous smooths)
            # by appending an index suffix.
            if name in edf_per_term:
                name = f"{name}#{i}"
            edf_per_term[name] = float(val)
        if edf_per_term:
            out["edf_per_term"] = edf_per_term

    return out or None


def _baseline_path(key: str) -> Path:
    safe = key.replace("/", "_").replace(" ", "_")
    return BASELINES_DIR / f"{safe}.json"


def _format_delta(label: str, baseline: float, current: float, abs_tol: float | None = None, rel_tol: float | None = None) -> str:
    delta = current - baseline
    rel = abs(delta) / max(abs(baseline), 1e-12)
    tol_str = ""
    if rel_tol is not None:
        tol_str = f" rel={rel:.3e} (tol={rel_tol:.0e})"
    elif abs_tol is not None:
        tol_str = f" abs_tol={abs_tol}"
    return f"  {label}: baseline={baseline:.6g} current={current:.6g} delta={delta:+.3e}{tol_str}"


def compare(
    current: dict[str, Any],
    baseline: dict[str, Any],
    *,
    rel_tol_neg_v: float = REL_TOL_NEG_V,
    abs_tol_edf: float = ABS_TOL_EDF,
) -> tuple[bool, list[str]]:
    """Return ``(passed, messages)``. ``passed=False`` iff a tolerance is broken.

    Missing fields are silently skipped — the caller decides whether a
    missing field is a "skip from gate" or an error.
    """
    messages: list[str] = []
    passed = True

    cur_v = current.get("final_neg_v")
    base_v = baseline.get("final_neg_v")
    if isinstance(cur_v, (int, float)) and isinstance(base_v, (int, float)):
        rel = abs(cur_v - base_v) / max(abs(base_v), 1e-12)
        if rel > rel_tol_neg_v:
            passed = False
            messages.append("REGRESS " + _format_delta("final_neg_v", base_v, cur_v, rel_tol=rel_tol_neg_v))
        else:
            messages.append("ok      " + _format_delta("final_neg_v", base_v, cur_v, rel_tol=rel_tol_neg_v))

    cur_edf = current.get("edf_per_term") or {}
    base_edf = baseline.get("edf_per_term") or {}
    common = sorted(set(cur_edf) & set(base_edf))
    for term in common:
        c = float(cur_edf[term])
        b = float(base_edf[term])
        delta = abs(c - b)
        if delta > abs_tol_edf:
            passed = False
            messages.append(f"REGRESS   edf[{term}]: baseline={b:.4f} current={c:.4f} delta={c - b:+.4f} (abs_tol={abs_tol_edf})")
        else:
            messages.append(f"ok        edf[{term}]: baseline={b:.4f} current={c:.4f} delta={c - b:+.4f}")
    only_cur = sorted(set(cur_edf) - set(base_edf))
    only_base = sorted(set(base_edf) - set(cur_edf))
    for term in only_cur:
        messages.append(f"warn      edf[{term}]: present in current but missing from baseline")
    for term in only_base:
        passed = False
        messages.append(f"REGRESS   edf[{term}]: present in baseline but missing from current")

    return passed, messages


def _resolve_gate_mode(cli_value: str | None) -> str:
    if cli_value:
        return cli_value
    env = os.environ.get("BENCH_GATE", "").strip().lower()
    if env in ("strict", "report"):
        return env
    return "report"


def _iter_model_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.is_dir():
        return []
    files: list[Path] = []
    for p in root.rglob("*.model.json"):
        files.append(p)
    for p in root.rglob("model_*.json"):
        files.append(p)
    seen: set[Path] = set()
    out: list[Path] = []
    for p in files:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)
    return sorted(out)


def _gate_one(key: str, current: dict[str, Any], *, update: bool, mode: str) -> bool:
    """Return True if this entry is OK (or non-fatal); False if it's a hard fail."""
    baseline_path = _baseline_path(key)
    if update:
        BASELINES_DIR.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(current, indent=2, sort_keys=True))
        print(f"[gate] wrote baseline {baseline_path}")
        return True
    if not baseline_path.is_file():
        print(f"[gate] SKIP {key}: no baseline at {baseline_path} (use --update-baseline to create one)")
        return True
    try:
        baseline = json.loads(baseline_path.read_text())
    except (OSError, ValueError) as e:
        print(f"[gate] SKIP {key}: baseline unreadable ({e})")
        return True
    passed, messages = compare(current, baseline)
    header = f"[gate] {key} {'PASS' if passed else 'FAIL'}"
    print(header)
    for line in messages:
        print(line)
    if passed:
        return True
    if mode == "strict":
        return False
    print(f"[gate] (mode=report) {key} regression noted but not failing")
    return True


def cmd_scan(args: argparse.Namespace) -> int:
    mode = _resolve_gate_mode(args.gate)
    models_dir = Path(args.models_dir).resolve()
    files = _iter_model_files(models_dir)
    if not files:
        print(f"[gate] no model files under {models_dir}", file=sys.stderr)
        return 0
    any_fail = False
    for model_file in files:
        fq = extract_fit_quality(model_file)
        if fq is None:
            print(f"[gate] SKIP {model_file}: no reml_score / edf_by_block in model.json")
            continue
        key = model_file.stem
        ok = _gate_one(key, fq, update=args.update_baseline, mode=mode)
        if not ok:
            any_fail = True
    if any_fail and mode == "strict":
        return 2
    return 0


def cmd_check_results(args: argparse.Namespace) -> int:
    mode = _resolve_gate_mode(args.gate)
    payload = json.loads(Path(args.results).read_text())
    rows = payload.get("results", []) if isinstance(payload, dict) else []
    if not rows:
        print(f"[gate] no results in {args.results}", file=sys.stderr)
        return 0
    any_fail = False
    seen_any_fq = False
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("status") != "ok":
            continue
        fq = row.get("fit_quality")
        contender = str(row.get("contender", "unknown"))
        scenario = str(row.get("scenario_name", "unknown"))
        key = f"{contender}__{scenario}"
        if not isinstance(fq, dict):
            print(f"[gate] SKIP {key}: row carries no fit_quality (lane not wired to the gate)")
            continue
        seen_any_fq = True
        ok = _gate_one(key, fq, update=args.update_baseline, mode=mode)
        if not ok:
            any_fail = True
    if not seen_any_fq:
        print("[gate] no rows carried fit_quality; nothing to gate")
    if any_fail and mode == "strict":
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GAM bench statistical-regression gate.")
    p.add_argument(
        "--gate",
        choices=["report", "strict"],
        default=None,
        help="report (default): print findings, exit 0. strict: exit 2 on regression. "
        "Falls back to BENCH_GATE env var if not given.",
    )
    p.add_argument(
        "--update-baseline",
        action="store_true",
        help="Overwrite baselines with the current values instead of comparing.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scan", help="Scan a directory of model.json files.")
    s.add_argument("--models-dir", required=True, help="Directory with *.model.json / model_*.json files.")
    s.set_defaults(func=cmd_scan)

    c = sub.add_parser("check-results", help="Check a run_suite.py results.json.")
    c.add_argument("results", help="Path to results.json produced by run_suite.py.")
    c.set_defaults(func=cmd_check_results)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
