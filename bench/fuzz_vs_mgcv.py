#!/usr/bin/env python3
"""Dataset-backed mgcv comparison harness for gamfit.

This file intentionally contains no spline, REML, gradient, or synthetic data
math. It orchestrates existing benchmark scenarios, calls the Python gamfit API
for the Rust implementation, calls the shared mgcv runner from run_suite, and
writes the same JSONL-style comparison records used by the old fuzzer.
"""

from __future__ import annotations

import argparse
import json
import math
import secrets
import statistics
import subprocess
import sys
import time
import traceback
import typing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import gamfit
from gamfit._binding import rust_module

ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = ROOT / "bench"
sys.path.insert(0, str(ROOT))
from bench.run_suite import (  # noqa: E402
    _formula_rhs_from_terms,
    _mgcv_formula_for_scenario,
    _read_meminfo,
    _rust_formula_for_scenario,
    _scenario_fit_mapping,
    _sigma_feature_terms,
    dataset_for_scenario,
    folds_for_dataset,
    run_external_mgcv_cv,
    run_external_mgcv_gaulss_cv,
    zscore_train_test,
)

typing.cast(typing.Any, sys.stdout).reconfigure(line_buffering=True)

SCENARIOS_FILE = BENCH_DIR / "scenarios.json"
RESULTS_FILE = BENCH_DIR / "fuzz_results.jsonl"
META_FILE = BENCH_DIR / "fuzz_results.meta.json"
DEFAULT_R_TIMEOUT = 180
DEFAULT_RUST_TIMEOUT = 180
DEFAULT_N_TRIALS = 200


@dataclass(frozen=True)
class FuzzScenario:
    trial_id: str
    seed: int
    name: str
    family: str
    model_type: str
    basis_type: str
    n_obs: int
    n_features: int
    formula: str
    mgcv_formula: str
    noise_formula: str | None = None

    def tag(self) -> str:
        return self.trial_id


@dataclass
class FuzzResult:
    scenario: dict[str, typing.Any]
    rust: dict[str, typing.Any]
    mgcv: dict[str, typing.Any]
    primary_gap: Optional[float] = None
    primary_metric: Optional[str] = None

    def compute_gap(self) -> None:
        self.primary_metric = "r2" if self.scenario["family"] == "gaussian" else "auc"
        rv = self.rust.get(self.primary_metric)
        mv = self.mgcv.get(self.primary_metric)
        if rv is not None and mv is not None:
            self.primary_gap = float(mv) - float(rv)
        elif mv is not None and rv is None:
            self.primary_gap = float(mv) + 1.0


ABS_GAP_WARN_THRESHOLD = 0.05
ABS_GAP_FAIL_THRESHOLD = 0.1
GAUSSIAN_R2_WARN_FLOOR = 0.01
GAUSSIAN_R2_FAIL_FLOOR = 0.02
GAUSSIAN_RMSE_WARN_RATIO = 1.5
GAUSSIAN_RMSE_FAIL_RATIO = 2.0
PER_TRIAL_FAIL_GAP = 0.30
COHORT_MEDIAN_FAIL_GAP = 0.05
COHORT_MIN_TRIALS = 6
COHORT_NET_WINS_FAIL = 5
MIN_VALID_TRIAL_FRACTION = 0.80
NAN_GATED_METRICS = ("r2", "auc", "rmse", "logloss", "mae", "brier")


def _basis_label(raw: typing.Any) -> str:
    basis = str(raw or "ps").strip().lower()
    if basis in {"thinplate", "tps", "tp"}:
        return "tps"
    if basis in {"duchon", "ds"}:
        return "duchon"
    if basis in {"matern", "gp"}:
        return "matern"
    return "ps"


def _load_scenario_configs() -> list[dict[str, typing.Any]]:
    obj = json.loads(SCENARIOS_FILE.read_text())
    scenarios = obj.get("scenarios", [])
    if not isinstance(scenarios, list):
        raise RuntimeError(f"{SCENARIOS_FILE} must contain a 'scenarios' list")
    return [s for s in scenarios if isinstance(s, dict) and isinstance(s.get("name"), str)]


def _scenario_cost(sc: FuzzScenario) -> float:
    multiplier = 1.8 if sc.model_type == "gamlss" else 1.0
    return float(max(sc.n_obs, 1) * max(sc.n_features, 1) * multiplier)


# Conservative per-(n*p)-element resident-byte estimates, calibrated against the
# observed OOM at n=2,399,999 p=4 (n*p ~= 9.6M): gam peaked ~13.7 GiB
# (~1530 B/elem) and the concurrent mgcv CV runner ~6.7 GiB (~750 B/elem). gam
# and mgcv run back-to-back within one trial while gam's RSS is still resident,
# so the runner must hold both peaks at once. Rounded up for headroom.
_GAM_BYTES_PER_NP = 1700.0
_MGCV_BYTES_PER_NP = 850.0
# Fraction of total runner RAM a single trial's combined gam+mgcv peak may use
# before the mgcv arm is skipped. Leaves room for the OS, the heartbeat process,
# and estimate error; below this the trial runs both arms as before.
_RAM_SAFETY_FRACTION = 0.80


def _projected_trial_peak_bytes(n_obs: int, n_features: int) -> tuple[float, float]:
    """Return (gam_peak, mgcv_peak) projected resident bytes for an (n, p) fit."""
    np_elems = float(max(n_obs, 1) * max(n_features, 1))
    return np_elems * _GAM_BYTES_PER_NP, np_elems * _MGCV_BYTES_PER_NP


def _runner_total_ram_bytes() -> Optional[float]:
    """Total physical RAM in bytes from /proc/meminfo, or None when unavailable
    (e.g. non-Linux dev hosts), in which case the RAM guard is a no-op."""
    meminfo = _read_meminfo()
    total_kib = meminfo.get("MemTotal")
    if total_kib is None:
        return None
    return float(total_kib) * 1024.0


def _mgcv_arm_fits_in_ram(n_obs: int, n_features: int) -> tuple[bool, str]:
    """Decide whether running the mgcv CV arm alongside gam at this (n, p) is
    safe on the current runner. Returns (ok, reason). When total RAM cannot be
    read the guard defers (``ok=True``) rather than skipping work blindly."""
    total = _runner_total_ram_bytes()
    if total is None:
        return True, "ram-unknown"
    gam_peak, mgcv_peak = _projected_trial_peak_bytes(n_obs, n_features)
    combined = gam_peak + mgcv_peak
    budget = total * _RAM_SAFETY_FRACTION
    if combined > budget:
        gib = 1024.0 ** 3
        return (
            False,
            f"projected gam({gam_peak / gib:.1f}GiB)+mgcv({mgcv_peak / gib:.1f}GiB)"
            f"={combined / gib:.1f}GiB exceeds {_RAM_SAFETY_FRACTION:.0%} of "
            f"{total / gib:.1f}GiB runner RAM at n={n_obs} p={n_features}",
        )
    return True, "ok"


def _materialize_scenario(
    cfg: dict[str, typing.Any],
    *,
    seed: int,
    model_type: str,
) -> FuzzScenario | None:
    name = str(cfg["name"])
    fit_cfg = _scenario_fit_mapping(name)
    if fit_cfg is None:
        return None

    ds = dataset_for_scenario(cfg)
    family = str(ds["family"])
    if family not in {"gaussian", "binomial"}:
        return None
    if model_type == "gamlss" and family != "gaussian":
        return None

    basis = _basis_label(fit_cfg.get("smooth_basis", "ps"))
    rust_family, formula = _rust_formula_for_scenario(name, ds)
    if model_type == "gamlss":
        noise_formula = _formula_rhs_from_terms(
            _sigma_feature_terms(ds, scenario_name=name, backend="rust")
        )
        mgcv_formula = _mgcv_formula_for_scenario(name, ds)
    else:
        noise_formula = None
        mgcv_formula = _mgcv_formula_for_scenario(name, ds)

    return FuzzScenario(
        trial_id=f"{seed}:{name}:{model_type}",
        seed=seed,
        name=name,
        family=family,
        model_type=model_type,
        basis_type=basis,
        n_obs=len(ds["rows"]),
        n_features=len(ds["features"]),
        formula=formula,
        mgcv_formula=mgcv_formula,
        noise_formula=noise_formula,
    )


def _candidate_scenarios(
    *,
    seed_start: int,
    model_type_filter: Optional[str],
) -> list[FuzzScenario]:
    configs = _load_scenario_configs()
    out: list[FuzzScenario] = []
    model_types = [model_type_filter] if model_type_filter else ["gam", "gamlss"]
    for offset, cfg in enumerate(configs):
        for model_type in model_types:
            if model_type is None:
                continue
            try:
                sc = _materialize_scenario(cfg, seed=seed_start + offset, model_type=model_type)
            except Exception as exc:
                print(f"  [skip] {cfg.get('name', '?')}: {exc}", flush=True)
                continue
            if sc is not None:
                out.append(sc)
    return out


def select_scenarios_backfilled(
    *,
    seed_start: int,
    target_count: int,
    excluded_ids: set[str],
    family_filter: Optional[str] = None,
    model_type_filter: Optional[str] = None,
    basis_filter: Optional[str] = None,
    max_scenario_cost: Optional[float] = None,
) -> tuple[list[FuzzScenario], list[tuple[FuzzScenario, float]]]:
    candidates = _candidate_scenarios(seed_start=seed_start, model_type_filter=model_type_filter)
    if not candidates:
        raise RuntimeError("no dataset-backed scenarios are available")

    start = seed_start % len(candidates)
    ordered = candidates[start:] + candidates[:start]
    selected: list[FuzzScenario] = []
    skipped: list[tuple[FuzzScenario, float]] = []

    for sc in ordered:
        if sc.trial_id in excluded_ids:
            continue
        if family_filter is not None and sc.family != family_filter:
            continue
        if model_type_filter is not None and sc.model_type != model_type_filter:
            continue
        if basis_filter is not None and sc.basis_type != basis_filter:
            continue
        cost = _scenario_cost(sc)
        if max_scenario_cost is not None and cost > max_scenario_cost:
            skipped.append((sc, cost))
            continue
        selected.append(sc)
        if len(selected) >= target_count:
            break

    if len(selected) < target_count:
        raise RuntimeError(
            f"only selected {len(selected)}/{target_count} scenario(s); "
            "filters or cost cap are too restrictive"
        )
    return selected, skipped


def _prediction_mean(predicted: typing.Any) -> np.ndarray:
    if isinstance(predicted, pd.DataFrame):
        return predicted["mean"].to_numpy(dtype=float)
    if isinstance(predicted, dict):
        return np.asarray(predicted["mean"], dtype=float)
    # Default ``model.predict`` now returns a 1-D ndarray of fitted means
    # for standard GAMs; preserve that shape here.
    return np.asarray(predicted, dtype=float)


def _prediction_sigma(predicted: typing.Any) -> np.ndarray | None:
    if isinstance(predicted, pd.DataFrame) and "sigma" in predicted.columns:
        return predicted["sigma"].to_numpy(dtype=float)
    if isinstance(predicted, dict) and "sigma" in predicted:
        return np.asarray(predicted["sigma"], dtype=float)
    return None


def _standard_deviation_from_model(model: typing.Any) -> float | None:
    try:
        payload = model._saved_model_payload()
    except Exception:
        return None
    fit_result = payload.get("fit_result")
    if not isinstance(fit_result, dict):
        return None
    sigma = fit_result.get("standard_deviation")
    if sigma is None:
        return None
    try:
        sigma_f = float(sigma)
    except (TypeError, ValueError):
        return None
    return sigma_f if math.isfinite(sigma_f) and sigma_f > 0.0 else None


def _sigma_payload(sigma: np.ndarray | float | None) -> list[float] | None:
    if sigma is None:
        return None
    arr = np.asarray(sigma, dtype=float)
    if arr.ndim == 0:
        return [float(arr)]
    return [float(v) for v in arr.reshape(-1)]


def _metrics_from_predictions(
    *,
    family: str,
    y_test: np.ndarray,
    y_train: np.ndarray,
    pred: np.ndarray,
    sigma: np.ndarray | float | None = None,
) -> dict[str, typing.Any]:
    rust = rust_module()
    observed = [float(v) for v in np.asarray(y_test, dtype=float).reshape(-1)]
    predicted = [float(v) for v in np.asarray(pred, dtype=float).reshape(-1)]
    if family == "binomial":
        train_prev = float(np.asarray(y_train, dtype=float).mean())
        return dict(rust.classification_metrics(observed, predicted, train_prev))
    if family != "gaussian":
        raise ValueError(f"unsupported prediction-metric family: {family}")

    sigma_values = _sigma_payload(sigma)
    if sigma_values is not None:
        scores = dict(
            rust.gaussian_prediction_scores_from_predictions(
                observed,
                predicted,
                sigma_values,
            )
        )
        if scores["r2"] is None:
            scores["r2"] = 0.0
        return {
            key: scores[key]
            for key in ("r2", "rmse", "mae", "mse", "logloss")
        }

    diagnostics = dict(rust.diagnostics_from_predictions(observed, predicted))["metrics"]
    rmse = float(diagnostics["rmse"])
    return {
        "r2": float(diagnostics.get("r_squared", 0.0)),
        "rmse": rmse,
        "mae": float(diagnostics["mae"]),
        "mse": rmse * rmse,
    }


def run_gamfit(
    sc: FuzzScenario,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ds: dict[str, typing.Any],
    rust_timeout: int,
) -> dict[str, typing.Any]:
    del rust_timeout
    y_test = test_df[ds["target"]].to_numpy(dtype=float)
    y_train = train_df[ds["target"]].to_numpy(dtype=float)
    family_arg, formula = _rust_formula_for_scenario(sc.name, ds)
    config = {"noise_formula": sc.noise_formula} if sc.noise_formula else None

    t0 = time.perf_counter()
    try:
        model = gamfit.fit(train_df, formula, family=family_arg, config=config)
        fit_sec = time.perf_counter() - t0
        t1 = time.perf_counter()
        predicted = model.predict(test_df)
        predict_sec = time.perf_counter() - t1
        pred = _prediction_mean(predicted)
        if len(pred) != len(y_test):
            return {
                "error": f"prediction count {len(pred)} vs {len(y_test)}",
                "time": fit_sec + predict_sec,
            }
        sigma: np.ndarray | float | None = _prediction_sigma(predicted)
        if sigma is None and sc.family == "gaussian":
            sigma = _standard_deviation_from_model(model)
        metrics = _metrics_from_predictions(
            family=sc.family,
            y_test=y_test,
            y_train=y_train,
            pred=pred,
            sigma=sigma,
        )
        metrics["fit_sec"] = fit_sec
        metrics["predict_sec"] = predict_sec
        metrics["time"] = fit_sec + predict_sec
        metrics["model_spec"] = formula if not sc.noise_formula else f"mu: {formula}; sigma: {sc.noise_formula}"
        return metrics
    except Exception as exc:
        return {
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "time": time.perf_counter() - t0,
        }


def run_mgcv(
    sc: FuzzScenario,
    ds: dict[str, typing.Any],
    fold: typing.Any,
    r_timeout: int,
) -> dict[str, typing.Any]:
    del r_timeout
    try:
        scenario_cfg = {"name": sc.name}
        row = (
            run_external_mgcv_gaulss_cv(scenario_cfg, ds=ds, folds=[fold])
            if sc.model_type == "gamlss"
            else run_external_mgcv_cv(scenario_cfg, ds=ds, folds=[fold])
        )
        if row is None:
            return {"error": "mgcv runner did not produce a result"}
        if row.get("status") == "failed":
            return {"error": row.get("error", "mgcv failed")}
        out = {k: row.get(k) for k in ("r2", "rmse", "mae", "mse", "logloss", "auc", "brier", "nagelkerke_r2")}
        out = {k: v for k, v in out.items() if v is not None}
        fit_sec = float(row.get("fit_sec") or 0.0)
        predict_sec = float(row.get("predict_sec") or 0.0)
        out["fit_sec"] = fit_sec
        out["predict_sec"] = predict_sec
        out["time"] = fit_sec + predict_sec
        if row.get("model_spec"):
            out["model_spec"] = row["model_spec"]
        return out
    except Exception as exc:
        return {"error": str(exc), "traceback": traceback.format_exc()}


def run_trial(sc: FuzzScenario, rust_timeout: int, r_timeout: int) -> FuzzResult:
    ds = dataset_for_scenario({"name": sc.name})
    folds = folds_for_dataset(ds)
    if not folds:
        raise RuntimeError(f"{sc.name}: no folds generated")
    fold = folds[0]
    df = pd.DataFrame(ds["rows"])
    # The dataset is re-materialized here, so the true (n, p) of *this* fit is
    # known. The selection-time scenario cost cap is a throughput heuristic, not
    # a RAM bound, and a scenario's n can differ from what selection observed.
    # Guard the mgcv CV arm against the actual fit size: gam and mgcv run
    # back-to-back within the trial while gam's peak RSS is still resident, so an
    # oversized draw can hold both peaks at once and OOM-kill the whole runner
    # (gam#820: n=2.4M p=4 → gam 13.7GiB + mgcv 6.7GiB on a 15.6GiB runner). When
    # the projected combined peak won't fit, skip *only* the mgcv arm and still
    # record the gam result, rather than letting the runner die.
    n_obs = len(df)
    n_features = len(ds["features"])
    mgcv_ok, mgcv_reason = _mgcv_arm_fits_in_ram(n_obs, n_features)

    train_df = df.iloc[fold.train_idx].copy()
    test_df = df.iloc[fold.test_idx].copy()
    train_df, test_df = zscore_train_test(train_df, test_df, ds["features"])

    rust_out = run_gamfit(sc, train_df, test_df, ds, rust_timeout)
    if mgcv_ok:
        mgcv_out = run_mgcv(sc, ds, fold, r_timeout)
    else:
        print(
            f"  [ram-skip] {sc.name}: skipping mgcv CV arm — {mgcv_reason}",
            flush=True,
        )
        mgcv_out = {"skipped": True, "skip_reason": mgcv_reason}
    result = FuzzResult(scenario=asdict(sc), rust=rust_out, mgcv=mgcv_out)
    result.compute_gap()
    return result


def gaussian_rmse_ratio(result: FuzzResult) -> Optional[float]:
    if result.scenario["family"] != "gaussian":
        return None
    rust_rmse = result.rust.get("rmse")
    mgcv_rmse = result.mgcv.get("rmse")
    if rust_rmse is None or mgcv_rmse is None:
        return None
    rust_rmse = float(rust_rmse)
    mgcv_rmse = float(mgcv_rmse)
    if not math.isfinite(rust_rmse) or not math.isfinite(mgcv_rmse):
        return None
    if rust_rmse < 0 or mgcv_rmse < 0:
        return None
    if mgcv_rmse <= 1e-12:
        return math.inf if rust_rmse > 1e-12 else 1.0
    return rust_rmse / mgcv_rmse


def classify_primary_divergence(result: FuzzResult) -> tuple[str, str]:
    gap = result.primary_gap
    if gap is None or gap <= 0:
        return "", ""
    metric = result.primary_metric or "metric"
    if gap > ABS_GAP_FAIL_THRESHOLD:
        return "fail", f"{metric}_gap={gap:+.4f}"
    if result.scenario["family"] == "gaussian":
        rmse_ratio = gaussian_rmse_ratio(result)
        if rmse_ratio is not None:
            if gap > GAUSSIAN_R2_FAIL_FLOOR and rmse_ratio > GAUSSIAN_RMSE_FAIL_RATIO:
                return "fail", f"r2_gap={gap:+.4f}, rmse_ratio={rmse_ratio:.2f}x"
            if gap > GAUSSIAN_R2_WARN_FLOOR and rmse_ratio > GAUSSIAN_RMSE_WARN_RATIO:
                return "warn", f"r2_gap={gap:+.4f}, rmse_ratio={rmse_ratio:.2f}x"
    if gap > ABS_GAP_WARN_THRESHOLD:
        return "warn", f"{metric}_gap={gap:+.4f}"
    return "", ""


def _metric_is_nonfinite(value: typing.Any) -> bool:
    if value is None:
        return False
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return not math.isfinite(f)


def _metric_is_finite(value: typing.Any) -> bool:
    if value is None:
        return False
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


def compute_ci_gates(
    results: typing.Any,
    requested_trials: int,
    skipped_count: int = 0,
    baseline: Optional[dict[str, typing.Any]] = None,
) -> dict[str, typing.Any]:
    valid_trials = [r for r in results if r.primary_gap is not None]
    valid_count = len(valid_trials)
    gate_failures: list[dict[str, typing.Any]] = []

    big_gap_offenders = [r for r in valid_trials if (r.primary_gap or 0.0) > PER_TRIAL_FAIL_GAP]
    if big_gap_offenders:
        gate_failures.append({
            "gate": "per_trial_gap",
            "message": f"{len(big_gap_offenders)} trial(s) with primary-metric gap > {PER_TRIAL_FAIL_GAP:.2f}",
            "offenders": big_gap_offenders,
        })

    cohorts: dict[tuple[typing.Any, typing.Any, typing.Any], list[FuzzResult]] = {}
    for r in valid_trials:
        cohort = (
            r.scenario.get("family", "?"),
            r.scenario.get("model_type", "?"),
            r.scenario.get("basis_type", "?"),
        )
        cohorts.setdefault(cohort, []).append(r)

    cohort_median_offenders = []
    cohort_net_wins_offenders = []
    for cohort, members in cohorts.items():
        if len(members) < COHORT_MIN_TRIALS:
            continue
        gaps = [float(r.primary_gap) for r in members if r.primary_gap is not None]
        if not gaps:
            continue
        median_gap = float(statistics.median(gaps))
        if median_gap > COHORT_MEDIAN_FAIL_GAP:
            cohort_median_offenders.append({"cohort": cohort, "median_gap": median_gap, "n": len(gaps)})
        mgcv_wins = sum(1 for g in gaps if g > 0.01)
        rust_wins = sum(1 for g in gaps if g < -0.01)
        net = mgcv_wins - rust_wins
        if net > COHORT_NET_WINS_FAIL:
            cohort_net_wins_offenders.append({
                "cohort": cohort,
                "mgcv_wins": mgcv_wins,
                "rust_wins": rust_wins,
                "net": net,
                "n": len(gaps),
            })

    if cohort_median_offenders:
        gate_failures.append({
            "gate": "cohort_median",
            "message": f"{len(cohort_median_offenders)} cohort(s) with median gap > {COHORT_MEDIAN_FAIL_GAP:.2f}",
            "offenders": cohort_median_offenders,
        })
    if cohort_net_wins_offenders:
        gate_failures.append({
            "gate": "cohort_net_wins",
            "message": f"{len(cohort_net_wins_offenders)} cohort(s) with net mgcv wins > {COHORT_NET_WINS_FAIL}",
            "offenders": cohort_net_wins_offenders,
        })

    nan_offenders = []
    for r in results:
        for metric in NAN_GATED_METRICS:
            mv = r.mgcv.get(metric)
            rv = r.rust.get(metric)
            if _metric_is_finite(mv) and _metric_is_nonfinite(rv):
                nan_offenders.append((r, metric, rv, mv))
                break
    if nan_offenders:
        gate_failures.append({
            "gate": "rust_nan_inf",
            "message": f"{len(nan_offenders)} trial(s) where gamfit produced NaN/inf against finite mgcv",
            "offenders": nan_offenders,
        })

    # Trials whose mgcv CV arm was deliberately skipped to keep the runner
    # within RAM (gam#820) have no mgcv metric, so they carry no `primary_gap`
    # and would otherwise drag down `valid_count`. They are a harness resource
    # decision, not a gam failure or a missing comparison we could have run, so
    # exclude them from the coverage denominator — exactly like cost-cap skips.
    ram_skipped_count = sum(1 for r in results if r.mgcv.get("skipped"))
    comparable_trials = max(0, requested_trials - ram_skipped_count)
    min_required = max(1, int(math.ceil(MIN_VALID_TRIAL_FRACTION * comparable_trials)))
    if valid_count < min_required:
        gate_failures.append({
            "gate": "coverage",
            "message": (
                f"only {valid_count}/{comparable_trials} comparable trial(s) produced a valid "
                f"comparison (skipped above cost cap: {skipped_count}; ram-skipped mgcv arm: "
                f"{ram_skipped_count}); minimum required: {min_required}"
            ),
            "offenders": [],
        })

    if baseline:
        baseline_offenders = []
        threshold = float(baseline.get("threshold", 0.05))
        cohort_baselines = baseline.get("cohorts", {}) or {}
        for cohort, members in cohorts.items():
            key = "/".join(str(part) for part in cohort)
            base_gap = cohort_baselines.get(key)
            if base_gap is None:
                continue
            gaps = [float(r.primary_gap) for r in members if r.primary_gap is not None]
            if not gaps:
                continue
            current = float(statistics.median(gaps))
            delta = current - float(base_gap)
            if delta > threshold:
                baseline_offenders.append({
                    "cohort": cohort,
                    "current_median_gap": current,
                    "baseline_median_gap": float(base_gap),
                    "delta": delta,
                    "n": len(gaps),
                })
        if baseline_offenders:
            gate_failures.append({
                "gate": "baseline_regression",
                "message": f"{len(baseline_offenders)} cohort(s) regressed against baseline by more than {threshold:.2f}",
                "offenders": baseline_offenders,
            })

    return {
        "failed": bool(gate_failures),
        "gate_failures": gate_failures,
        "valid_count": valid_count,
        "requested_trials": requested_trials,
        "skipped_count": skipped_count,
        "ram_skipped_count": ram_skipped_count,
        "min_required": min_required,
    }


def print_leaderboard(results: typing.Any, top_n: int = 25) -> None:
    groups: dict[tuple[typing.Any, typing.Any, typing.Any], list[typing.Any]] = {}
    for r in results:
        key = (r.scenario["family"], r.scenario["model_type"], r.scenario["basis_type"])
        groups.setdefault(key, []).append(r)
    groups[("ALL", "ALL", "ALL")] = results

    for (fam, mt, basis), subset in sorted(groups.items()):
        valid = [r for r in subset if r.primary_gap is not None]
        if not valid:
            continue
        valid.sort(key=lambda r: -(r.primary_gap or 0.0))
        metric = valid[0].primary_metric or "?"
        label = f"{fam}/{mt}/{basis}" if fam != "ALL" else "ALL TRIALS"
        print(f"\n{'=' * 120}")
        print(f"  {label} - {metric} gap (mgcv - gamfit) | {len(valid)} valid / {len(subset)} total")
        print("=" * 120)
        print(f"{'#':>3}  {'gap':>8}  {'gamfit':>9}  {'mgcv':>9}  {'scenario':>36}  {'n':>6}  {'p':>3}")
        print("-" * 120)
        for i, r in enumerate(valid[:top_n]):
            s = r.scenario
            rv = r.rust.get(metric)
            mv = r.mgcv.get(metric)
            rs = f"{rv:.4f}" if rv is not None else "  FAIL"
            ms = f"{mv:.4f}" if mv is not None else "  FAIL"
            gap_s = f"{r.primary_gap:+.4f}" if r.primary_gap is not None else "  N/A "
            print(
                f"{i + 1:3d}  {gap_s}  {rs:>9}  {ms:>9}  "
                f"{s['name'][:36]:>36}  {s['n_obs']:6d}  {s['n_features']:3d}"
            )
        gaps = [float(r.primary_gap) for r in valid if r.primary_gap is not None]
        mgcv_w = sum(1 for g in gaps if g > 0.01)
        rust_w = sum(1 for g in gaps if g < -0.01)
        ties = len(gaps) - mgcv_w - rust_w
        median_gap = statistics.median(gaps)
        print(f"\n  mgcv wins: {mgcv_w} | gamfit wins: {rust_w} | ties: {ties} | median gap: {median_gap:+.4f}")

    rust_fails = [r for r in results if r.rust.get("error")]
    mgcv_fails = [r for r in results if r.mgcv.get("error")]
    if rust_fails or mgcv_fails:
        print(f"\n{'=' * 120}")
        print(f"  FAILURES - gamfit: {len(rust_fails)} | mgcv: {len(mgcv_fails)}")
        print("=" * 120)
        for label, fails in [("GAMFIT", rust_fails), ("MGCV", mgcv_fails)]:
            for r in fails[:10]:
                s = r.scenario
                out = r.rust if label == "GAMFIT" else r.mgcv
                print(f"  [{label}] {s['name']} {s['family']}/{s['model_type']}/{s['basis_type']}")
                print(f"    error: {str(out.get('error', ''))[:300]}")


def _load_existing_results() -> tuple[list[FuzzResult], set[str]]:
    results: list[FuzzResult] = []
    ids: set[str] = set()
    if not RESULTS_FILE.exists():
        return results, ids
    with open(RESULTS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            fr = FuzzResult(scenario=obj["scenario"], rust=obj["rust"], mgcv=obj["mgcv"])
            fr.compute_gap()
            results.append(fr)
            trial_id = fr.scenario.get("trial_id")
            if isinstance(trial_id, str):
                ids.add(trial_id)
    return results, ids


def _default_seed_start(args: argparse.Namespace) -> int:
    if args.seed_start is not None:
        return int(args.seed_start)
    if args.resume and META_FILE.exists():
        try:
            saved = json.loads(META_FILE.read_text())
            if isinstance(saved, dict) and isinstance(saved.get("seed_start"), int):
                return int(saved["seed_start"])
        except (OSError, json.JSONDecodeError):
            pass
    return secrets.randbelow(2**31 - 1) + 1


def _check_mgcv_available() -> None:
    r_check = subprocess.run(
        ["Rscript", "-e", "suppressPackageStartupMessages(library(mgcv)); cat('ok')"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if "ok" not in (r_check.stdout or ""):
        raise RuntimeError("R + mgcv not available")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset-backed gamfit vs mgcv comparison harness")
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--family", type=str, default=None, choices=["gaussian", "binomial"])
    parser.add_argument("--model-type", type=str, default=None, choices=["gam", "gamlss"])
    parser.add_argument("--basis", type=str, default=None, choices=["ps", "tps", "duchon"])
    parser.add_argument("--rust-timeout", type=int, default=DEFAULT_RUST_TIMEOUT)
    parser.add_argument("--r-timeout", type=int, default=DEFAULT_R_TIMEOUT)
    parser.add_argument("--max-total-seconds", type=int, default=None)
    parser.add_argument("--max-scenario-cost", type=float, default=200_000.0)
    parser.add_argument("--baseline-json", type=str, default=None)
    args = parser.parse_args()

    try:
        _check_mgcv_available()
    except Exception as exc:
        print(str(exc))
        sys.exit(1)

    seed_start_explicit = args.seed_start is not None
    args.seed_start = _default_seed_start(args)

    if not args.resume and RESULTS_FILE.exists():
        RESULTS_FILE.unlink()
    results, existing_ids = _load_existing_results() if args.resume else ([], set())
    if args.resume:
        print(f"Loaded {len(results)} existing results")

    target_new_trials = max(0, args.n_trials - len(results))
    if target_new_trials == 0:
        print_leaderboard(results, top_n=args.top)
        return

    scenarios, skipped_scenarios = select_scenarios_backfilled(
        seed_start=int(args.seed_start),
        target_count=target_new_trials,
        excluded_ids=existing_ids,
        family_filter=args.family,
        model_type_filter=args.model_type,
        basis_filter=args.basis,
        max_scenario_cost=args.max_scenario_cost,
    )

    seed_origin = "explicit" if seed_start_explicit else "auto"
    print(f"Running {target_new_trials} trials")
    print(f"  Seed start:        {args.seed_start} [{seed_origin}]")
    print(f"  Replay this run:   --seed-start {args.seed_start}")
    print(f"  Scenario source:   {SCENARIOS_FILE}")
    print(f"  Filters: family={args.family or 'all'} model={args.model_type or 'all'} basis={args.basis or 'all'}")
    print(f"  Gamfit timeout:    {args.rust_timeout}s")
    print(f"  R timeout:         {args.r_timeout}s")
    print(f"  Time budget:       {args.max_total_seconds}s" if args.max_total_seconds else "  Time budget:       none")
    print(f"  Scenario cost cap: {args.max_scenario_cost:g}" if args.max_scenario_cost is not None else "  Scenario cost cap: none")
    print(f"  Results: {RESULTS_FILE}")
    print(f"  Metadata: {META_FILE}\n")

    META_FILE.write_text(
        json.dumps(
            {
                "seed_start": args.seed_start,
                "seed_start_origin": seed_origin,
                "n_trials": args.n_trials,
                "family": args.family,
                "model_type": args.model_type,
                "basis": args.basis,
                "max_scenario_cost": args.max_scenario_cost,
                "rust_timeout": args.rust_timeout,
                "r_timeout": args.r_timeout,
                "max_total_seconds": args.max_total_seconds,
                "wallclock_started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
            indent=2,
        )
        + "\n"
    )

    if skipped_scenarios:
        max_skipped = max(cost for _, cost in skipped_scenarios)
        print(
            f"Skipped {len(skipped_scenarios)} trial(s) above scenario cost cap "
            f"{args.max_scenario_cost:g}; max skipped cost={max_skipped:.0f}",
            flush=True,
        )

    started_at = time.time()
    with open(RESULTS_FILE, "a") as out_f:
        for i, sc in enumerate(scenarios):
            if args.max_total_seconds is not None:
                elapsed = time.time() - started_at
                if elapsed >= args.max_total_seconds:
                    print(
                        f"\nReached wall-clock budget after {i} completed trial(s) "
                        f"({elapsed:.1f}s >= {args.max_total_seconds}s). Stopping early.",
                        flush=True,
                    )
                    break
            result = run_trial(sc, args.rust_timeout, args.r_timeout)
            results.append(result)
            out_f.write(
                json.dumps(
                    {
                        "scenario": result.scenario,
                        "rust": result.rust,
                        "mgcv": result.mgcv,
                        "primary_gap": result.primary_gap,
                        "primary_metric": result.primary_metric,
                    },
                    default=str,
                )
                + "\n"
            )
            out_f.flush()

            metric = result.primary_metric or "?"
            rv = result.rust.get(metric)
            mv = result.mgcv.get(metric)
            rs = f"{rv:.4f}" if rv is not None else " FAIL"
            ms = f"{mv:.4f}" if mv is not None else " FAIL"
            gap_s = f"{result.primary_gap:+.4f}" if result.primary_gap is not None else " N/A "
            err_r = " [G:ERR]" if result.rust.get("error") else ""
            err_m = " [M:ERR]" if result.mgcv.get("error") else ""
            divergence_level, _ = classify_primary_divergence(result)
            flag = " !!!" if divergence_level == "fail" else (" !!" if divergence_level == "warn" else "")
            if result.primary_gap is not None and result.primary_gap > PER_TRIAL_FAIL_GAP:
                flag += " [FAIL]"
            for gate_metric in NAN_GATED_METRICS:
                if _metric_is_finite(result.mgcv.get(gate_metric)) and _metric_is_nonfinite(result.rust.get(gate_metric)):
                    flag += " [FAIL:nan]"
                    break
            t_rust = float(result.rust.get("time", 0) or 0)
            t_mgcv = float(result.mgcv.get("time", 0) or 0)
            time_s = f"  gamfit={t_rust:.1f}s mgcv={t_mgcv:.1f}s" if max(t_rust, t_mgcv) > 0.5 else ""
            print(
                f"  [{i + 1:3d}/{len(scenarios)}] {sc.name[:30]:30s} "
                f"{sc.family[:4]}/{sc.model_type[:5]}/{sc.basis_type[:5]:5s} "
                f"{metric}:gamfit={rs} {metric}:mgcv={ms} gap={gap_s} "
                f"n={sc.n_obs:6d} p={sc.n_features:3d}{err_r}{err_m}{flag}{time_s}",
                flush=True,
            )

            for label, out in [("GAMFIT", result.rust), ("MGCV", result.mgcv)]:
                if out.get("error"):
                    print(f"    -- {label} FAILURE --", flush=True)
                    print(f"    error: {out['error']}", flush=True)
                    if out.get("traceback"):
                        print(f"    traceback: {out['traceback']}", flush=True)

    print_leaderboard(results, top_n=args.top)

    rust_only_failures = [r for r in results if r.rust.get("error") and not r.mgcv.get("error")]
    baseline = None
    if args.baseline_json:
        try:
            with open(args.baseline_json) as bf:
                baseline = json.load(bf)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"\nCI FAIL: --baseline-json {args.baseline_json!r} could not be loaded: {exc}")
            sys.exit(1)

    gates = compute_ci_gates(
        results,
        requested_trials=args.n_trials,
        skipped_count=len(skipped_scenarios),
        baseline=baseline,
    )
    any_failure = bool(rust_only_failures) or gates["failed"]

    if rust_only_failures:
        print(f"\n{'=' * 120}")
        print("  CI FAIL: harness detected gamfit execution failures")
        print("=" * 120)
        for r in rust_only_failures[:20]:
            s = r.scenario
            err = str(r.rust.get("error", ""))[:200]
            print(f"    [FAIL] {s['name']} {s['family']}/{s['model_type']}/{s['basis_type']} :: {err}")

    if gates["failed"]:
        print(f"\n{'=' * 120}")
        print("  CI FAIL: harness tripped regression gates")
        print("=" * 120)
        print(
            f"  trials: {gates['valid_count']}/{gates['requested_trials']} valid "
            f"(skipped above cost cap: {gates['skipped_count']}; min required: {gates['min_required']})"
        )
        for gf in gates["gate_failures"]:
            print(f"\n  -- gate [{gf['gate']}] -- {gf['message']}")
            offenders = gf.get("offenders", [])
            if gf["gate"] == "per_trial_gap":
                for r in sorted(offenders, key=lambda row: (row.primary_gap or 0), reverse=True)[:20]:
                    s = r.scenario
                    rv = r.rust.get(r.primary_metric or "r2")
                    mv = r.mgcv.get(r.primary_metric or "r2")
                    print(
                        f"    [FAIL] {s['name']} {s['family']}/{s['model_type']}/{s['basis_type']} "
                        f"{r.primary_metric}: gamfit={rv!r} mgcv={mv!r} gap={r.primary_gap:+.4f}"
                    )
            elif gf["gate"] in {"cohort_median", "cohort_net_wins", "baseline_regression"}:
                for offender in offenders:
                    print(f"    [FAIL] {offender}")
            elif gf["gate"] == "rust_nan_inf":
                for r, metric, rv, mv in offenders[:20]:
                    s = r.scenario
                    print(f"    [FAIL] {s['name']} gamfit {metric}={rv!r} (mgcv {metric}={mv:.4f})")

    if any_failure:
        print(f"\n{'=' * 120}")
        gates_fired = [gf["gate"] for gf in gates["gate_failures"]]
        if rust_only_failures:
            gates_fired.insert(0, "gamfit_only_failures")
        print(f"  CI FAIL: gates fired: {', '.join(gates_fired) or '(none)'}")
        print("=" * 120)
        sys.exit(1)


if __name__ == "__main__":
    main()
