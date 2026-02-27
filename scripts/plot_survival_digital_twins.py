#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "bench" / "datasets"
DEFAULT_OUT = ROOT / "bench" / "plots" / "digital_twins"

AXIS_LABELS = {
    "icu_survival_death": (
        "Days since ICU admission (ICU mortality dataset)",
        "Chance an ICU patient is still alive",
    ),
    "icu_survival_los": (
        "Days since ICU admission (ICU length-of-stay dataset)",
        "Chance an ICU patient is still alive",
    ),
    "heart_failure_survival": (
        "Days since heart-failure study enrollment",
        "Chance a heart-failure patient is still alive",
    ),
    "cirrhosis_survival": (
        "Days since cirrhosis study enrollment",
        "Chance a cirrhosis patient is still alive",
    ),
}


@dataclass
class SurvivalDataset:
    name: str
    rows: pd.DataFrame
    features: list[str]
    time_col: str
    event_col: str
    formula: str
    fit_opts: dict[str, str]


def _parse_f64_opt(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _encode_bool_yn(raw: str) -> float:
    return 1.0 if (raw or "").strip().upper() == "Y" else 0.0


def _encode_edema(raw: str) -> float:
    t = (raw or "").strip().upper()
    if t == "Y":
        return 2.0
    if t == "S":
        return 1.0
    return 0.0


def load_icu_survival_death() -> SurvivalDataset:
    d = pd.read_csv(DATASET_DIR / "icu_survival_death.csv")
    d = d[["time", "age", "bmi", "hr_max", "sysbp_min", "event"]].dropna().copy()
    return SurvivalDataset(
        name="icu_survival_death",
        rows=d,
        features=["age", "bmi", "hr_max", "sysbp_min"],
        time_col="time",
        event_col="event",
        formula="linear(age) + s(bmi, type=ps, knots=6) + linear(hr_max) + linear(sysbp_min)",
        fit_opts={"time_basis": "linear", "ridge_lambda": "1e-4"},
    )


def load_icu_survival_los() -> SurvivalDataset:
    d = pd.read_csv(DATASET_DIR / "icu_survival_los.csv")
    d = d[["age", "bmi", "hr_max", "sysbp_min", "temp_apache", "time", "event"]].dropna().copy()
    return SurvivalDataset(
        name="icu_survival_los",
        rows=d,
        features=["age", "bmi", "hr_max", "sysbp_min", "temp_apache"],
        time_col="time",
        event_col="event",
        formula="linear(age) + linear(bmi) + linear(hr_max) + linear(sysbp_min) + linear(temp_apache)",
        fit_opts={"time_basis": "linear", "ridge_lambda": "1e-4"},
    )


def load_heart_failure_survival() -> SurvivalDataset:
    d = pd.read_csv(DATASET_DIR / "heart_failure_clinical_records_dataset.csv")
    cols = [
        "time",
        "DEATH_EVENT",
        "age",
        "anaemia",
        "creatinine_phosphokinase",
        "diabetes",
        "ejection_fraction",
        "high_blood_pressure",
        "platelets",
        "serum_creatinine",
        "serum_sodium",
        "sex",
        "smoking",
    ]
    d = d[cols].copy()
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna().copy()
    rows = pd.DataFrame(
        {
            "time": d["time"].astype(float),
            "event": d["DEATH_EVENT"].astype(float),
            "age": d["age"].astype(float),
            "anaemia": d["anaemia"].astype(float),
            "log_creatinine_phosphokinase": np.log1p(np.maximum(d["creatinine_phosphokinase"].astype(float), 0.0)),
            "diabetes": d["diabetes"].astype(float),
            "ejection_fraction": d["ejection_fraction"].astype(float),
            "high_blood_pressure": d["high_blood_pressure"].astype(float),
            "log_platelets": np.log1p(np.maximum(d["platelets"].astype(float), 0.0)),
            "log_serum_creatinine": np.log1p(np.maximum(d["serum_creatinine"].astype(float), 0.0)),
            "serum_sodium": d["serum_sodium"].astype(float),
            "sex": d["sex"].astype(float),
            "smoking": d["smoking"].astype(float),
        }
    )
    return SurvivalDataset(
        name="heart_failure_survival",
        rows=rows,
        features=[
            "age",
            "anaemia",
            "log_creatinine_phosphokinase",
            "diabetes",
            "ejection_fraction",
            "high_blood_pressure",
            "log_platelets",
            "log_serum_creatinine",
            "serum_sodium",
            "sex",
            "smoking",
        ],
        time_col="time",
        event_col="event",
        formula=(
            "linear(age) + linear(anaemia) + linear(log_creatinine_phosphokinase) + linear(diabetes) + "
            "linear(ejection_fraction) + linear(high_blood_pressure) + linear(log_platelets) + "
            "linear(log_serum_creatinine) + linear(serum_sodium) + linear(sex) + linear(smoking)"
        ),
        fit_opts={
            "time_basis": "bspline",
            "time_degree": "3",
            "time_num_internal_knots": "8",
            "time_smooth_lambda": "1e-2",
            "ridge_lambda": "1e-6",
        },
    )


def load_cirrhosis_survival() -> SurvivalDataset:
    d = pd.read_csv(DATASET_DIR / "cirrhosis.csv")
    numeric_cols = [
        "Age",
        "Bilirubin",
        "Cholesterol",
        "Albumin",
        "Copper",
        "Alk_Phos",
        "SGOT",
        "Tryglicerides",
        "Platelets",
        "Prothrombin",
        "Stage",
    ]
    rows = []
    for r in d.to_dict(orient="records"):
        drug = str(r.get("Drug", "")).strip()
        if not drug or drug.lower() == "na":
            continue
        status = str(r.get("Status", "")).strip().upper()
        if status not in {"D", "C", "CL"}:
            continue

        t = _parse_f64_opt(r.get("N_Days", ""))
        if t is None:
            continue
        t = max(t, 1.0)

        sex = str(r.get("Sex", "")).strip().upper()
        asc = str(r.get("Ascites", "")).strip().upper()
        hep = str(r.get("Hepatomegaly", "")).strip().upper()
        spi = str(r.get("Spiders", "")).strip().upper()
        ede = str(r.get("Edema", "")).strip().upper()
        if sex not in {"M", "F"}:
            continue
        if asc not in {"Y", "N"} or hep not in {"Y", "N"} or spi not in {"Y", "N"}:
            continue
        if ede not in {"N", "S", "Y"}:
            continue

        parsed_num: dict[str, float] = {}
        ok = True
        for col in numeric_cols:
            v = _parse_f64_opt(r.get(col, ""))
            if v is None:
                ok = False
                break
            parsed_num[col.lower()] = float(v)
        if not ok:
            continue

        rows.append(
            {
                "time": float(t),
                "event": 1.0 if status == "D" else 0.0,
                "drug": 1.0 if "penicillamine" in drug.lower() else 0.0,
                "sex_male": 1.0 if sex == "M" else 0.0,
                "ascites": _encode_bool_yn(asc),
                "hepatomegaly": _encode_bool_yn(hep),
                "spiders": _encode_bool_yn(spi),
                "edema": _encode_edema(ede),
                **parsed_num,
            }
        )
    out = pd.DataFrame(rows)
    return SurvivalDataset(
        name="cirrhosis_survival",
        rows=out,
        features=[
            "drug",
            "sex_male",
            "ascites",
            "hepatomegaly",
            "spiders",
            "edema",
            "age",
            "bilirubin",
            "cholesterol",
            "albumin",
            "copper",
            "alk_phos",
            "sgot",
            "tryglicerides",
            "platelets",
            "prothrombin",
            "stage",
        ],
        time_col="time",
        event_col="event",
        formula=(
            "linear(drug) + linear(sex_male) + linear(ascites) + linear(hepatomegaly) + linear(spiders) + "
            "linear(edema) + linear(age) + linear(bilirubin) + linear(cholesterol) + linear(albumin) + "
            "linear(copper) + linear(alk_phos) + linear(sgot) + linear(tryglicerides) + linear(platelets) + "
            "linear(prothrombin) + linear(stage)"
        ),
        fit_opts={
            "time_basis": "bspline",
            "time_degree": "3",
            "time_num_internal_knots": "8",
            "time_smooth_lambda": "1e-2",
            "ridge_lambda": "1e-6",
        },
    )


def _run_cmd(cmd: list[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{msg}")


def _ensure_rust_binary() -> Path:
    env_raw = str(subprocess.os.environ.get("BENCH_GAM_BIN", "")).strip()
    if env_raw:
        env_bin = Path(env_raw).expanduser()
        if env_bin.exists() and env_bin.is_file():
            return env_bin.resolve()

    local_bin = ROOT / "target" / "release" / "gam"
    if local_bin.exists() and local_bin.is_file():
        return local_bin

    _run_cmd(["cargo", "build", "--release", "--bin", "gam"], cwd=ROOT)
    if not local_bin.exists():
        raise RuntimeError(f"missing Rust binary at {local_bin}")
    return local_bin


def _zscore_by_train(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    out = df.copy()
    stats: dict[str, tuple[float, float]] = {}
    for c in feature_cols:
        mu = float(out[c].mean())
        sd = float(out[c].std())
        if (not np.isfinite(sd)) or sd < 1e-8:
            sd = 1.0
        out[c] = (out[c] - mu) / sd
        stats[c] = (mu, sd)
    return out, stats


def _apply_zscore(df: pd.DataFrame, stats: dict[str, tuple[float, float]]) -> pd.DataFrame:
    out = df.copy()
    for c, (mu, sd) in stats.items():
        out[c] = (out[c] - mu) / sd
    return out


def _fit_survival_model(
    rust_bin: Path,
    ds: SurvivalDataset,
    train_df: pd.DataFrame,
    model_path: Path,
    likelihood: str,
) -> None:
    if likelihood == "transformation":
        args = [
            str(rust_bin),
            "survival",
            str(train_df),
            "--entry",
            "__entry",
            "--exit",
            ds.time_col,
            "--event",
            ds.event_col,
            "--formula",
            ds.formula,
            "--survival-likelihood",
            "transformation",
            "--baseline-target",
            "linear",
            "--time-basis",
            ds.fit_opts["time_basis"],
            "--ridge-lambda",
            ds.fit_opts["ridge_lambda"],
            "--out",
            str(model_path),
        ]
        if ds.fit_opts["time_basis"] == "bspline":
            args.extend(
                [
                    "--time-degree",
                    ds.fit_opts["time_degree"],
                    "--time-num-internal-knots",
                    ds.fit_opts["time_num_internal_knots"],
                    "--time-smooth-lambda",
                    ds.fit_opts["time_smooth_lambda"],
                ]
            )
        _run_cmd(args, cwd=ROOT)
        return

    if likelihood != "probit-location-scale":
        raise RuntimeError(f"unsupported likelihood mode: {likelihood}")

    tried: list[str] = []
    attempts: list[dict[str, str]] = []
    attempts.append(dict(ds.fit_opts))
    if ds.fit_opts.get("time_basis") != "linear":
        attempts.append({"time_basis": "linear", "ridge_lambda": ds.fit_opts.get("ridge_lambda", "1e-4")})
    attempts.append({"time_basis": "linear", "ridge_lambda": "1e-4"})
    attempts.append({"time_basis": "linear", "ridge_lambda": "1e-3"})

    for opt in attempts:
        args = [
            str(rust_bin),
            "survival",
            str(train_df),
            "--entry",
            "__entry",
            "--exit",
            ds.time_col,
            "--event",
            ds.event_col,
            "--formula",
            ds.formula,
            "--survival-likelihood",
            "probit-location-scale",
            "--survival-distribution",
            "gaussian",
            "--baseline-target",
            "linear",
            "--time-basis",
            opt["time_basis"],
            "--ridge-lambda",
            opt["ridge_lambda"],
            "--out",
            str(model_path),
        ]
        if opt["time_basis"] == "bspline":
            args.extend(
                [
                    "--time-degree",
                    opt.get("time_degree", "3"),
                    "--time-num-internal-knots",
                    opt.get("time_num_internal_knots", "8"),
                    "--time-smooth-lambda",
                    opt.get("time_smooth_lambda", "1e-2"),
                ]
            )
        try:
            _run_cmd(args, cwd=ROOT)
            return
        except RuntimeError as e:
            tried.append(f"time_basis={opt['time_basis']}, ridge={opt['ridge_lambda']}: {e}")

    raise RuntimeError(
        "all probit-location-scale survival fit attempts failed:\n" + "\n".join(tried)
    )


def _predict_survival_curve(
    rust_bin: Path,
    model_path: Path,
    base_row: pd.Series,
    features: list[str],
    time_col: str,
    times: np.ndarray,
    tmpdir: Path,
) -> np.ndarray:
    pred_in = pd.DataFrame({c: [float(base_row[c])] * len(times) for c in features})
    pred_in["__entry"] = 0.0
    pred_in[time_col] = times
    pred_in_path = tmpdir / "pred_input.csv"
    pred_out_path = tmpdir / "pred_output.csv"
    pred_in.to_csv(pred_in_path, index=False)

    _run_cmd(
        [str(rust_bin), "predict", str(model_path), str(pred_in_path), "--out", str(pred_out_path)],
        cwd=ROOT,
    )
    pred = pd.read_csv(pred_out_path)
    if "survival_prob" in pred.columns:
        s = pred["survival_prob"].to_numpy(dtype=float)
    elif "mean" in pred.columns:
        s = pred["mean"].to_numpy(dtype=float)
    else:
        raise RuntimeError("prediction output missing survival_prob/mean")
    s = np.clip(s, 0.0, 1.0)
    # Numerically enforce monotone non-increasing survival over time.
    s = np.minimum.accumulate(s)
    return s


def _normalize_from_baseline_survival(surv: np.ndarray) -> np.ndarray:
    if surv.size == 0:
        return surv
    s0 = float(max(surv[0], 1e-8))
    out = np.clip(surv / s0, 0.0, 1.0)
    out[0] = 1.0
    out = np.minimum.accumulate(out)
    return out


def _pick_representative_twin(
    rust_bin: Path,
    model_path: Path,
    eval_df: pd.DataFrame,
    features: list[str],
    time_col: str,
    tmpdir: Path,
) -> int:
    horizon = float(np.median(eval_df[time_col].to_numpy(dtype=float)))
    pred_input = eval_df[features].copy()
    pred_input["__entry"] = 0.0
    pred_input[time_col] = horizon
    in_path = tmpdir / "all_input.csv"
    out_path = tmpdir / "all_pred.csv"
    pred_input.to_csv(in_path, index=False)
    _run_cmd([str(rust_bin), "predict", str(model_path), str(in_path), "--out", str(out_path)], cwd=ROOT)
    p = pd.read_csv(out_path)
    if "failure_prob" in p.columns:
        risk = p["failure_prob"].to_numpy(dtype=float)
    elif "risk_score" in p.columns:
        risk = p["risk_score"].to_numpy(dtype=float)
    elif "mean" in p.columns:
        risk = 1.0 - p["mean"].to_numpy(dtype=float)
    else:
        raise RuntimeError("prediction output missing risk-compatible column")
    target = float(np.median(risk))
    idx = int(np.argmin(np.abs(risk - target)))
    return idx


def _smooth_noise(rng: np.random.Generator, n: int, scale: float) -> np.ndarray:
    eps = rng.normal(0.0, scale, size=n)
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float)
    kernel /= kernel.sum()
    return np.convolve(eps, kernel, mode="same")


def _simulate_smooth_paths(
    times: np.ndarray,
    surv_curve: np.ndarray,
    n_paths: int,
    seed: int,
    enforce_start_one: bool = True,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t_max = float(times[-1])
    paths = np.zeros((n_paths, len(times)), dtype=float)
    for i in range(n_paths):
        # Time warp + shape tilt around the model-implied survival curve.
        scale = float(np.exp(rng.normal(0.0, 0.18)))
        shift = float(rng.normal(0.0, 0.05 * t_max))
        tilt = float(np.exp(rng.normal(0.0, 0.12)))
        warped_t = np.clip((times - shift) / scale, times[0], times[-1])
        p = np.interp(warped_t, times, surv_curve)
        p = np.power(np.clip(p, 1e-8, 1.0), tilt)
        eps = _smooth_noise(rng, len(times), scale=0.020)
        p = p + eps * (0.15 + p * (1.0 - p))
        p = np.clip(p, 0.0, 1.0)
        if enforce_start_one:
            p[0] = 1.0
        p = np.minimum.accumulate(p)
        paths[i, :] = p
    return paths


def _survival_curve_diagnostics(name: str, surv: np.ndarray, paths: np.ndarray) -> None:
    base_min = float(np.min(surv))
    base_max = float(np.max(surv))
    base_range = float(base_max - base_min)
    base_std = float(np.std(surv))
    var_t = np.var(paths, axis=0)
    med_var = float(np.median(var_t))
    max_var = float(np.max(var_t))
    end_mean = float(np.mean(paths[:, -1]))
    near_one = float(np.mean(paths > 0.99))
    near_zero = float(np.mean(paths < 0.01))
    print(
        f"[{name}] base_min={base_min:.4f} base_max={base_max:.4f} base_range={base_range:.4f} "
        f"base_std={base_std:.4f} path_var_med={med_var:.6f} path_var_max={max_var:.6f} "
        f"end_mean={end_mean:.4f} frac_gt_0.99={near_one:.4f} frac_lt_0.01={near_zero:.4f}"
    )
    if base_range < 0.03 or base_std < 0.01:
        print(f"[{name}] warning: base survival curve appears nearly flat.")
    if med_var < 1e-4:
        print(f"[{name}] warning: simulated path variance is very low.")
    if near_one > 0.98 or near_zero > 0.98:
        print(f"[{name}] warning: paths spend almost all mass at an extreme (0 or 1).")


def _is_pathological_curve(surv: np.ndarray) -> bool:
    if surv.size == 0:
        return True
    base_range = float(np.max(surv) - np.min(surv))
    base_std = float(np.std(surv))
    return (base_range < 0.03) or (base_std < 0.01)


def _format_characteristic_value(v: float) -> str:
    if not np.isfinite(v):
        return "NA"
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v))}"
    return f"{v:.3g}"


def _fmt_num(v: float, digits: int = 2) -> str:
    if not np.isfinite(v):
        return "NA"
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v))}"
    return f"{v:.{digits}f}"


def _twin_profile_text(dataset_name: str, twin_profile: dict[str, float]) -> str:
    p = twin_profile
    lines = ["Simulated patient"]
    if dataset_name == "icu_survival_death":
        lines.extend(
            [
                f"Age: {_fmt_num(p['age'])} years",
                f"Body mass index: {_fmt_num(p['bmi'])} kg/m^2",
                f"Maximum heart rate: {_fmt_num(p['hr_max'])} beats per minute",
                f"Minimum systolic blood pressure: {_fmt_num(p['sysbp_min'])} mmHg",
            ]
        )
        return "\n".join(lines)
    if dataset_name == "icu_survival_los":
        lines.extend(
            [
                f"Age: {_fmt_num(p['age'])} years",
                f"Body mass index: {_fmt_num(p['bmi'])} kg/m^2",
                f"Maximum heart rate: {_fmt_num(p['hr_max'])} beats per minute",
                f"Minimum systolic blood pressure: {_fmt_num(p['sysbp_min'])} mmHg",
                f"Body temperature (APACHE): {_fmt_num(p['temp_apache'])} C",
            ]
        )
        return "\n".join(lines)
    if dataset_name == "heart_failure_survival":
        cpk = float(np.expm1(max(p["log_creatinine_phosphokinase"], 0.0)))
        platelets = float(np.expm1(max(p["log_platelets"], 0.0)))
        scr = float(np.expm1(max(p["log_serum_creatinine"], 0.0)))
        lines.extend(
            [
                f"Age: {_fmt_num(p['age'])} years",
                f"Sex: {'Male' if p['sex'] >= 0.5 else 'Female'}",
                f"Anemia: {'Yes' if p['anaemia'] >= 0.5 else 'No'}",
                f"Diabetes: {'Yes' if p['diabetes'] >= 0.5 else 'No'}",
                f"High blood pressure: {'Yes' if p['high_blood_pressure'] >= 0.5 else 'No'}",
                f"Smoking: {'Smoker' if p['smoking'] >= 0.5 else 'Non-smoker'}",
                f"Ejection fraction: {_fmt_num(p['ejection_fraction'])} %",
                f"Creatinine phosphokinase: {_fmt_num(cpk)} U/L",
                f"Platelet count: {_fmt_num(platelets)} per microliter",
                f"Serum creatinine: {_fmt_num(scr)} mg/dL",
                f"Serum sodium: {_fmt_num(p['serum_sodium'])} mEq/L",
            ]
        )
        return "\n".join(lines)
    if dataset_name == "cirrhosis_survival":
        age_years = float(p["age"]) / 365.25
        edema_map = {0: "No edema", 1: "Mild edema", 2: "Severe edema"}
        edema_txt = edema_map.get(int(round(p["edema"])), f"Edema code {int(round(p['edema']))}")
        lines.extend(
            [
                f"Age: {_fmt_num(age_years)} years",
                f"Sex: {'Male' if p['sex_male'] >= 0.5 else 'Female'}",
                f"Received D-penicillamine: {'Yes' if p['drug'] >= 0.5 else 'No'}",
                f"Ascites: {'Yes' if p['ascites'] >= 0.5 else 'No'}",
                f"Hepatomegaly: {'Yes' if p['hepatomegaly'] >= 0.5 else 'No'}",
                f"Spider angiomas present: {'Yes' if p['spiders'] >= 0.5 else 'No'}",
                f"Edema: {edema_txt}",
                f"Bilirubin: {_fmt_num(p['bilirubin'])} mg/dL",
                f"Cholesterol: {_fmt_num(p['cholesterol'])} mg/dL",
                f"Albumin: {_fmt_num(p['albumin'])} g/dL",
                f"Serum copper level: {_fmt_num(p['copper'])}",
                f"Alkaline phosphatase: {_fmt_num(p['alk_phos'])} U/L",
                f"SGOT (AST): {_fmt_num(p['sgot'])} U/L",
                f"Triglycerides: {_fmt_num(p['tryglicerides'])} mg/dL",
                f"Platelet count: {_fmt_num(p['platelets'])}",
                f"Prothrombin time: {_fmt_num(p['prothrombin'])} seconds",
                f"Disease stage: {_fmt_num(p['stage'])}",
            ]
        )
        return "\n".join(lines)
    # Fallback
    for name, val in p.items():
        lines.append(f"{name}: {_format_characteristic_value(float(val))}")
    return "\n".join(lines)


def _convert_time_axis_if_long(times: np.ndarray, x_label: str) -> tuple[np.ndarray, str]:
    if float(times[-1]) <= 700.0:
        return times, x_label
    x = times / 365.25
    if x_label.startswith("Days since "):
        return x, "Years since " + x_label[len("Days since ") :]
    if "Days" in x_label:
        return x, x_label.replace("Days", "Years")
    return x, x_label + " (years)"


def _plot_paths(
    dataset_name: str,
    times: np.ndarray,
    surv_curve: np.ndarray,
    sim_paths: np.ndarray,
    twin_profile: dict[str, float],
    out_path: Path,
) -> None:
    palette = {
        "icu_survival_death": dict(bg="#ffffff", fg="#1f2133", accent="#ca4f44", glow="#f7c8c2"),
        "icu_survival_los": dict(bg="#ffffff", fg="#16263b", accent="#2f6f9f", glow="#bcddee"),
        "heart_failure_survival": dict(bg="#ffffff", fg="#2a1f33", accent="#a23b72", glow="#e8c7da"),
        "cirrhosis_survival": dict(bg="#ffffff", fg="#263024", accent="#4f7f58", glow="#cde0cf"),
    }[dataset_name]

    fig = plt.figure(figsize=(12.5, 8.0), dpi=220)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor(palette["bg"])
    ax.set_facecolor(palette["bg"])

    x_label, y_label = AXIS_LABELS.get(
        dataset_name,
        ("Days since enrollment", "Chance the patient is still alive"),
    )
    x_vals, x_label = _convert_time_axis_if_long(times, x_label)

    # Simulated smooth digital-twin futures.
    for i in range(sim_paths.shape[0]):
        ax.plot(x_vals, sim_paths[i], color=palette["accent"], alpha=0.060, lw=0.9, solid_capstyle="round", zorder=2)

    q_lo = np.quantile(sim_paths, 0.10, axis=0)
    q_hi = np.quantile(sim_paths, 0.90, axis=0)
    ax.fill_between(x_vals, q_lo, q_hi, color=palette["glow"], alpha=0.35, zorder=1.5)

    # Model-implied central survival curve.
    ax.plot(x_vals, surv_curve, color=palette["fg"], lw=3.0, alpha=0.98, zorder=4)

    ax.set_xlim(float(x_vals[0]), float(x_vals[-1]))
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(x_label, fontsize=18, color=palette["fg"], labelpad=12)
    ax.set_ylabel(y_label, fontsize=18, color=palette["fg"], labelpad=12)

    panel_text = _twin_profile_text(dataset_name, twin_profile)
    is_top_right = dataset_name in {"heart_failure_survival", "cirrhosis_survival"}
    x = 0.985 if is_top_right else 0.015
    y = 0.985 if is_top_right else 0.015
    ha = "right" if is_top_right else "left"
    va = "top" if is_top_right else "bottom"
    ax.text(
        x,
        y,
        panel_text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=16.0,
        color=palette["fg"],
        linespacing=1.22,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=palette["fg"], alpha=0.80, linewidth=0.7),
        zorder=5,
    )

    for spine in ax.spines.values():
        spine.set_color(palette["fg"])
        spine.set_alpha(0.35)

    ax.tick_params(axis="both", colors=palette["fg"], labelsize=13)
    ax.grid(axis="x", color=palette["fg"], alpha=0.06, lw=0.6)
    ax.grid(axis="y", color=palette["fg"], alpha=0.05, lw=0.6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def generate_plot_for_dataset(
    rust_bin: Path,
    ds: SurvivalDataset,
    out_dir: Path,
    n_paths: int,
    n_grid: int,
    seed: int,
) -> Path:
    raw_df = ds.rows.copy().reset_index(drop=True)
    df = raw_df.copy()
    df["__entry"] = 0.0
    fit_df, _ = _zscore_by_train(df, ds.features)

    with tempfile.TemporaryDirectory(prefix=f"twin_{ds.name}_", dir=str(ROOT / "bench")) as td:
        td_path = Path(td)
        train_path = td_path / "train.csv"
        model_path = td_path / "model.json"
        fit_df.to_csv(train_path, index=False)
        _fit_survival_model(rust_bin, ds, train_path, model_path, likelihood="probit-location-scale")

        def build_curve_for_current_model() -> tuple[np.ndarray, np.ndarray, int]:
            twin_idx_local = _pick_representative_twin(
                rust_bin=rust_bin,
                model_path=model_path,
                eval_df=fit_df,
                features=ds.features,
                time_col=ds.time_col,
                tmpdir=td_path,
            )
            twin_row_local = fit_df.iloc[twin_idx_local].copy()
            t_obs_local = fit_df[ds.time_col].to_numpy(dtype=float)
            t_base_local = max(float(np.quantile(t_obs_local, 0.995)), float(np.max(t_obs_local)), 1.0)
            t_max_local = t_base_local
            surv_local = None
            for _ in range(4):
                times_local = np.linspace(0.0, t_max_local, int(n_grid))
                surv_local = _predict_survival_curve(
                    rust_bin=rust_bin,
                    model_path=model_path,
                    base_row=twin_row_local,
                    features=ds.features,
                    time_col=ds.time_col,
                    times=times_local,
                    tmpdir=td_path,
                )
                surv_local = _normalize_from_baseline_survival(surv_local)
                if float(surv_local[-1]) <= 0.15:
                    break
                t_max_local *= 1.6
            assert surv_local is not None
            return times_local, surv_local, twin_idx_local

        times, surv, twin_idx = build_curve_for_current_model()
        used_likelihood = "probit-location-scale"
        if _is_pathological_curve(surv):
            print(f"[{ds.name}] fallback: probit-location-scale curve is pathological, refitting with transformation likelihood.")
            _fit_survival_model(rust_bin, ds, train_path, model_path, likelihood="transformation")
            times, surv, twin_idx = build_curve_for_current_model()
            used_likelihood = "transformation"

        sim_paths = _simulate_smooth_paths(times, surv, n_paths=n_paths, seed=seed + hash(ds.name) % 10000)
        print(f"[{ds.name}] likelihood_used={used_likelihood}")
        _survival_curve_diagnostics(ds.name, surv, sim_paths)
        twin_profile = {f: float(raw_df.iloc[twin_idx][f]) for f in ds.features}

    out_path = out_dir / f"{ds.name}_digital_twin_paths.png"
    _plot_paths(
        ds.name,
        times,
        surv,
        sim_paths,
        twin_profile=twin_profile,
        out_path=out_path,
    )
    return out_path


def _predict_failure_prob_at_horizon(
    rust_bin: Path,
    model_path: Path,
    rows_df: pd.DataFrame,
    features: list[str],
    time_col: str,
    horizon: float,
    tmpdir: Path,
) -> np.ndarray:
    pred_in = rows_df[features].copy()
    pred_in["__entry"] = 0.0
    pred_in[time_col] = float(horizon)
    in_path = tmpdir / "risk_input.csv"
    out_path = tmpdir / "risk_pred.csv"
    pred_in.to_csv(in_path, index=False)
    _run_cmd([str(rust_bin), "predict", str(model_path), str(in_path), "--out", str(out_path)], cwd=ROOT)
    pred = pd.read_csv(out_path)
    if "failure_prob" in pred.columns:
        return pred["failure_prob"].to_numpy(dtype=float)
    if "survival_prob" in pred.columns:
        return (1.0 - pred["survival_prob"].to_numpy(dtype=float)).astype(float)
    if "mean" in pred.columns:
        return (1.0 - pred["mean"].to_numpy(dtype=float)).astype(float)
    raise RuntimeError("prediction output missing failure/survival probability columns")


def _plot_bmi_sweep_frame(
    times: np.ndarray,
    surv_curve: np.ndarray,
    sim_paths: np.ndarray,
    color_rgba: tuple[float, float, float, float],
    bmi_value: float,
    percentile: float,
    y_lo: float,
    y_hi: float,
    out_path: Path,
) -> None:
    fg = "#1f2133"
    bg = "#ffffff"
    fig = plt.figure(figsize=(12.5, 8.0), dpi=220)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    x_vals, x_label = _convert_time_axis_if_long(times, "Days since ICU admission (ICU mortality dataset)")
    y_label = "Chance an ICU patient is still alive"

    line_color = (color_rgba[0], color_rgba[1], color_rgba[2], 0.30)
    for i in range(sim_paths.shape[0]):
        ax.plot(x_vals, sim_paths[i], color=line_color, lw=1.0, solid_capstyle="round", zorder=2)
    ax.plot(x_vals, surv_curve, color=fg, lw=3.0, alpha=0.98, zorder=4)

    ax.set_xlim(float(x_vals[0]), float(x_vals[-1]))
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(x_label, fontsize=18, color=fg, labelpad=12)
    ax.set_ylabel(y_label, fontsize=18, color=fg, labelpad=12)
    ax.tick_params(axis="both", colors=fg, labelsize=13)
    ax.grid(axis="x", color=fg, alpha=0.06, lw=0.6)
    ax.grid(axis="y", color=fg, alpha=0.05, lw=0.6)
    for spine in ax.spines.values():
        spine.set_color(fg)
        spine.set_alpha(0.35)

    ax.text(
        0.015,
        0.985,
        f"Body mass index: {bmi_value:.0f}\nRisk percentile in full ICU dataset: {percentile*100:.1f}%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=15,
        color=fg,
        linespacing=1.2,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=fg, alpha=0.85, linewidth=0.7),
        zorder=5,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def generate_bmi_sweep_mp4_preview(
    rust_bin: Path,
    out_dir: Path,
    n_grid: int,
    seed: int,
    samples_per_bmi: int = 50,
    first_n: int = 5,
    fps: int = 5,
) -> Path:
    ds = load_icu_survival_death()
    raw_df = ds.rows.copy().reset_index(drop=True)
    fit_df = raw_df.copy()
    fit_df["__entry"] = 0.0
    fit_df, stats = _zscore_by_train(fit_df, ds.features)

    with tempfile.TemporaryDirectory(prefix="bmi_sweep_", dir=str(ROOT / "bench")) as td:
        td_path = Path(td)
        train_path = td_path / "train.csv"
        model_path = td_path / "model.json"
        fit_df.to_csv(train_path, index=False)

        # Use transformation mode for stable, non-pathological surfaces in sweep animation.
        _fit_survival_model(rust_bin, ds, train_path, model_path, likelihood="transformation")

        t_obs = fit_df[ds.time_col].to_numpy(dtype=float)
        t_base = max(float(np.quantile(t_obs, 0.995)), float(np.max(t_obs)), 1.0)
        t_max = t_base
        mean_row_z = pd.Series({f: 0.0 for f in ds.features})  # z-scored mean profile
        surv_probe = None
        for _ in range(4):
            times_probe = np.linspace(0.0, t_max, int(n_grid))
            surv_probe = _predict_survival_curve(
                rust_bin=rust_bin,
                model_path=model_path,
                base_row=mean_row_z,
                features=ds.features,
                time_col=ds.time_col,
                times=times_probe,
                tmpdir=td_path,
            )
            surv_probe = _normalize_from_baseline_survival(surv_probe)
            if float(surv_probe[-1]) <= 0.15:
                break
            t_max *= 1.6

        times = np.linspace(0.0, t_max, int(n_grid))
        horizon = float(np.median(fit_df[ds.time_col].to_numpy(dtype=float)))
        all_risk = _predict_failure_prob_at_horizon(
            rust_bin=rust_bin,
            model_path=model_path,
            rows_df=fit_df,
            features=ds.features,
            time_col=ds.time_col,
            horizon=horizon,
            tmpdir=td_path,
        )

        bmi_min = int(np.floor(raw_df["bmi"].min()))
        bmi_max = int(np.ceil(raw_df["bmi"].max()))
        all_bmi_values = list(range(bmi_min, bmi_max + 1))
        bmi_values = all_bmi_values[: max(1, int(first_n))]
        is_full_range = len(bmi_values) >= len(all_bmi_values)

        frames_dir = out_dir / ("bmi_sweep_frames_full" if is_full_range else "bmi_sweep_frames_preview")
        frames_dir.mkdir(parents=True, exist_ok=True)
        # Better perceptual contrast while preserving semantics:
        # green = lower risk, red = higher risk.
        cmap = LinearSegmentedColormap.from_list(
            "risk_better",
            ["#1f9e59", "#f2d95c", "#d7303f"],
            N=256,
        )

        means_raw = {f: float(raw_df[f].mean()) for f in ds.features}
        frame_payload = []
        for i, bmi in enumerate(bmi_values):
            profile_raw = dict(means_raw)
            profile_raw["bmi"] = float(bmi)
            profile_z = {f: (profile_raw[f] - stats[f][0]) / stats[f][1] for f in ds.features}
            row_z = pd.Series(profile_z)

            surv = _predict_survival_curve(
                rust_bin=rust_bin,
                model_path=model_path,
                base_row=row_z,
                features=ds.features,
                time_col=ds.time_col,
                times=times,
                tmpdir=td_path,
            )
            sim_paths = _simulate_smooth_paths(
                times,
                surv,
                n_paths=int(samples_per_bmi),
                seed=seed + i * 101,
                enforce_start_one=False,
            )

            risk = _predict_failure_prob_at_horizon(
                rust_bin=rust_bin,
                model_path=model_path,
                rows_df=pd.DataFrame([profile_z]),
                features=ds.features,
                time_col=ds.time_col,
                horizon=horizon,
                tmpdir=td_path,
            )[0]
            percentile = float(np.mean(all_risk <= risk))
            color = cmap(np.clip(percentile, 0.0, 1.0))
            frame_payload.append((i, float(bmi), percentile, color, surv, sim_paths))

        min_surv = min(float(np.min(surv)) for _, _, _, _, surv, _ in frame_payload)
        max_surv = max(float(np.max(surv)) for _, _, _, _, surv, _ in frame_payload)
        y_lo = max(0.0, min_surv - 0.03)
        y_hi = min(1.02, max_surv + 0.02)

        for i, bmi, percentile, color, surv, sim_paths in frame_payload:
            frame_path = frames_dir / f"frame_{i:03d}.png"
            _plot_bmi_sweep_frame(
                times,
                surv,
                sim_paths,
                color,
                bmi,
                percentile,
                y_lo,
                y_hi,
                frame_path,
            )

        mp4_path = out_dir / ("icu_bmi_sweep_full.mp4" if is_full_range else "icu_bmi_sweep_preview_first5.mp4")
        _run_cmd(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(max(1, int(fps))),
                "-i",
                str(frames_dir / "frame_%03d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(mp4_path),
            ],
            cwd=ROOT,
        )
    return mp4_path


def generate_icu_sysbp_sweep_mp4(
    rust_bin: Path,
    out_dir: Path,
    n_grid: int,
    seed: int,
    samples_per_frame: int = 100,
    n_frames: int = 100,
    fps: int = 15,
) -> Path:
    ds = load_icu_survival_death()
    raw_df = ds.rows.copy().reset_index(drop=True)
    fit_df = raw_df.copy()
    fit_df["__entry"] = 0.0
    fit_df, stats = _zscore_by_train(fit_df, ds.features)

    with tempfile.TemporaryDirectory(prefix="sysbp_sweep_", dir=str(ROOT / "bench")) as td:
        td_path = Path(td)
        train_path = td_path / "train.csv"
        model_path = td_path / "model.json"
        fit_df.to_csv(train_path, index=False)
        _fit_survival_model(rust_bin, ds, train_path, model_path, likelihood="transformation")

        t_obs = fit_df[ds.time_col].to_numpy(dtype=float)
        t_base = max(float(np.quantile(t_obs, 0.995)), float(np.max(t_obs)), 1.0)
        t_max = t_base
        probe = pd.Series({f: 0.0 for f in ds.features})
        for _ in range(4):
            times = np.linspace(0.0, t_max, int(n_grid))
            surv_probe = _predict_survival_curve(
                rust_bin=rust_bin,
                model_path=model_path,
                base_row=probe,
                features=ds.features,
                time_col=ds.time_col,
                times=times,
                tmpdir=td_path,
            )
            if float(surv_probe[-1]) <= 0.15:
                break
            t_max *= 1.6
        times = np.linspace(0.0, t_max, int(n_grid))
        x_vals, x_label = _convert_time_axis_if_long(times, "Days since ICU admission (ICU mortality dataset)")

        horizon = float(np.median(fit_df[ds.time_col].to_numpy(dtype=float)))
        all_risk = _predict_failure_prob_at_horizon(
            rust_bin=rust_bin,
            model_path=model_path,
            rows_df=fit_df,
            features=ds.features,
            time_col=ds.time_col,
            horizon=horizon,
            tmpdir=td_path,
        )

        q = np.linspace(0.0, 1.0, int(max(2, n_frames)))
        sysbp_values = np.quantile(raw_df["sysbp_min"].to_numpy(dtype=float), q)
        means_raw = {f: float(raw_df[f].mean()) for f in ds.features}
        cmap = LinearSegmentedColormap.from_list(
            "risk_better",
            ["#1f9e59", "#f2d95c", "#d7303f"],
            N=256,
        )

        frame_payload = []
        for i, sbp in enumerate(sysbp_values):
            profile_raw = dict(means_raw)
            profile_raw["sysbp_min"] = float(sbp)
            profile_z = {f: (profile_raw[f] - stats[f][0]) / stats[f][1] for f in ds.features}
            row_z = pd.Series(profile_z)
            surv = _predict_survival_curve(
                rust_bin=rust_bin,
                model_path=model_path,
                base_row=row_z,
                features=ds.features,
                time_col=ds.time_col,
                times=times,
                tmpdir=td_path,
            )
            sim_paths = _simulate_smooth_paths(
                times,
                surv,
                n_paths=int(samples_per_frame),
                seed=seed + i * 37,
                enforce_start_one=False,
            )
            risk = _predict_failure_prob_at_horizon(
                rust_bin=rust_bin,
                model_path=model_path,
                rows_df=pd.DataFrame([profile_z]),
                features=ds.features,
                time_col=ds.time_col,
                horizon=horizon,
                tmpdir=td_path,
            )[0]
            percentile = float(np.mean(all_risk <= risk))
            color = cmap(np.clip(percentile, 0.0, 1.0))
            frame_payload.append((i, float(sbp), surv, sim_paths, color))

        all_surv_vals = np.concatenate([s.ravel() for _, _, s, _, _ in frame_payload])
        lo = float(np.quantile(all_surv_vals, 0.01))
        hi = float(np.quantile(all_surv_vals, 0.99))
        y_lo = max(0.0, lo - 0.02)
        y_hi = min(1.02, hi + 0.02)

        frames_dir = out_dir / "sysbp_sweep_frames_full"
        frames_dir.mkdir(parents=True, exist_ok=True)
        fg = "#1f2133"
        bg = "#ffffff"
        for i, sbp, surv, sim_paths, color in frame_payload:
            fig = plt.figure(figsize=(12.5, 8.0), dpi=220)
            ax = fig.add_subplot(111)
            fig.patch.set_facecolor(bg)
            ax.set_facecolor(bg)

            line_color = (color[0], color[1], color[2], 0.20)
            for k in range(sim_paths.shape[0]):
                ax.plot(x_vals, sim_paths[k], color=line_color, lw=0.8, solid_capstyle="round", zorder=2)
            ax.plot(x_vals, surv, color=fg, lw=4.0, alpha=1.0, zorder=4)

            ax.set_xlim(float(x_vals[0]), float(x_vals[-1]))
            ax.set_ylim(y_lo, y_hi)
            ax.set_xlabel(x_label, fontsize=18, color=fg, labelpad=12)
            ax.set_ylabel("Chance an ICU patient is still alive", fontsize=18, color=fg, labelpad=12)
            ax.tick_params(axis="both", colors=fg, labelsize=13)
            ax.grid(axis="x", color=fg, alpha=0.06, lw=0.6)
            ax.grid(axis="y", color=fg, alpha=0.05, lw=0.6)
            for spine in ax.spines.values():
                spine.set_color(fg)
                spine.set_alpha(0.35)

            ax.text(
                0.015,
                0.985,
                f"Minimum systolic blood pressure: {sbp:.1f} mmHg",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=18,
                color=fg,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=fg, alpha=0.85, linewidth=0.7),
                zorder=5,
            )
            frame_path = frames_dir / f"frame_{i:03d}.png"
            fig.tight_layout()
            fig.savefig(frame_path, bbox_inches="tight")
            plt.close(fig)

        mp4_path = out_dir / "icu_death_min_systolic_blood_pressure_sweep_full.mp4"
        _run_cmd(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(max(1, int(fps))),
                "-i",
                str(frames_dir / "frame_%03d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(mp4_path),
            ],
            cwd=ROOT,
        )
    return mp4_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit Rust GAM survival GAMLSS-style models and render generative digital-twin path plots."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--paths", type=int, default=2000, help="Number of simulated trajectories per dataset.")
    parser.add_argument("--grid", type=int, default=260, help="Number of time points in the plotted survival curve.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bmi-sweep-preview", action="store_true", help="Generate ICU BMI sweep MP4 preview.")
    parser.add_argument("--bmi-first-n", type=int, default=5, help="Number of BMI values to include in preview.")
    parser.add_argument("--bmi-samples", type=int, default=50, help="Simulated paths per BMI value in preview mode.")
    parser.add_argument("--bmi-fps", type=int, default=5, help="Frames per second for BMI sweep MP4.")
    parser.add_argument("--sysbp-sweep", action="store_true", help="Generate ICU death minimum systolic blood pressure sweep MP4.")
    parser.add_argument("--sysbp-frames", type=int, default=100, help="Number of systolic blood pressure frames.")
    parser.add_argument("--sysbp-samples", type=int, default=100, help="Simulated paths per systolic blood pressure frame.")
    parser.add_argument("--sysbp-fps", type=int, default=15, help="Frames per second for systolic blood pressure sweep MP4.")
    args = parser.parse_args()

    loaders: list[Callable[[], SurvivalDataset]] = [
        load_icu_survival_death,
        load_icu_survival_los,
        load_heart_failure_survival,
        load_cirrhosis_survival,
    ]

    rust_bin = _ensure_rust_binary()
    if args.sysbp_sweep:
        mp4 = generate_icu_sysbp_sweep_mp4(
            rust_bin=rust_bin,
            out_dir=args.out_dir,
            n_grid=max(80, int(args.grid)),
            seed=int(args.seed),
            samples_per_frame=max(10, int(args.sysbp_samples)),
            n_frames=max(2, int(args.sysbp_frames)),
            fps=max(1, int(args.sysbp_fps)),
        )
        print("Generated systolic blood pressure sweep MP4:")
        print(mp4)
        return

    if args.bmi_sweep_preview:
        mp4 = generate_bmi_sweep_mp4_preview(
            rust_bin=rust_bin,
            out_dir=args.out_dir,
            n_grid=max(80, int(args.grid)),
            seed=int(args.seed),
            samples_per_bmi=max(10, int(args.bmi_samples)),
            first_n=max(1, int(args.bmi_first_n)),
            fps=max(1, int(args.bmi_fps)),
        )
        print("Generated BMI sweep preview MP4:")
        print(mp4)
        return

    outputs: list[Path] = []
    for loader in loaders:
        ds = loader()
        out = generate_plot_for_dataset(
            rust_bin=rust_bin,
            ds=ds,
            out_dir=args.out_dir,
            n_paths=max(50, int(args.paths)),
            n_grid=max(80, int(args.grid)),
            seed=int(args.seed),
        )
        outputs.append(out)

    print("Generated digital twin plots:")
    for p in outputs:
        print(p)


if __name__ == "__main__":
    main()
