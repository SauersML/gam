#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from shutil import disk_usage
from time import monotonic, perf_counter

def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(raw.strip())
    except Exception:
        return int(default)


# Optional strict serial mode for reproducibility/fairness studies.
# Default is parallel-friendly so the Rust contender can use Rayon threads.
_FORCE_SERIAL = _env_flag("BENCH_FORCE_SERIAL", default=False)
_RAYON_THREADS = _env_int("BENCH_RAYON_THREADS", 0)
_BLAS_THREADS = _env_int("BENCH_BLAS_THREADS", 0)
_CMD_TIMEOUT_SEC = _env_int("BENCH_CMD_TIMEOUT_SEC", 0)
_SERIAL_ENV_OVERRIDES = (
    {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "RAYON_NUM_THREADS": "1",
        "CARGO_BUILD_JOBS": "1",
        "OMP_DYNAMIC": "FALSE",
        "MKL_DYNAMIC": "FALSE",
    }
    if _FORCE_SERIAL
    else {}
)
if not _FORCE_SERIAL:
    # Explicit thread controls for performance runs.
    # 0 or unset means "leave tool defaults unchanged".
    if _RAYON_THREADS > 0:
        _SERIAL_ENV_OVERRIDES["RAYON_NUM_THREADS"] = str(_RAYON_THREADS)
    if _BLAS_THREADS > 0:
        _SERIAL_ENV_OVERRIDES["OMP_NUM_THREADS"] = str(_BLAS_THREADS)
        _SERIAL_ENV_OVERRIDES["OPENBLAS_NUM_THREADS"] = str(_BLAS_THREADS)
        _SERIAL_ENV_OVERRIDES["MKL_NUM_THREADS"] = str(_BLAS_THREADS)
        _SERIAL_ENV_OVERRIDES["VECLIB_MAXIMUM_THREADS"] = str(_BLAS_THREADS)
        _SERIAL_ENV_OVERRIDES["NUMEXPR_NUM_THREADS"] = str(_BLAS_THREADS)
        _SERIAL_ENV_OVERRIDES["BLIS_NUM_THREADS"] = str(_BLAS_THREADS)
for _k, _v in _SERIAL_ENV_OVERRIDES.items():
    os.environ[_k] = _v

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lifelines import CoxPHFitter, LogNormalAFTFitter, WeibullAFTFitter  # noqa: E402
from lifelines.exceptions import ConvergenceWarning  # noqa: E402
from lifelines.utils import concordance_index  # noqa: E402
from pygam import LinearGAM, LogisticGAM, l, s  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = ROOT / "bench"
DEFAULT_SCENARIOS = BENCH_DIR / "scenarios.json"
DATASET_DIR = BENCH_DIR / "datasets"
CV_SPLITS = 5
CV_SEED = 42
_RUST_BIN_PATH: Path | None = None
HEARTBEAT_INTERVAL_SEC = 15.0
# For short-lived commands, poll more frequently at startup so heartbeat
# diagnostics capture meaningful process stats before exit.
HEARTBEAT_INITIAL_WINDOW_SEC = 2.0
HEARTBEAT_INITIAL_INTERVAL_SEC = 0.25
HGDP_1KG_PC_TSV = DATASET_DIR / "hgdp_1kg_pc_data.tsv"
NON_BLOCKING_FAILURE_CONTENDERS = {
    # These external stacks are kept in the benchmark output for visibility,
    # but occasional fit/predict failures should not fail the whole CI shard.
    "r_gamlss",
    "rust_gamlss",
    "rust_gamlss_survival",
    "r_gamboostlss",
    "r_bamlss",
}
_BENCH_CI_PROFILE = os.environ.get("BENCH_CI_PROFILE", "full").strip().lower() or "full"
_LEAN_PROFILE_EXCLUDED_CONTENDERS = {
    # These contenders materially increase CI runtime and peak memory on
    # GitHub-hosted runners. Keep them in the nightly/full profile.
    "r_bamlss",
    "r_brms",
    "python_sksurv_rsf",
    "python_sksurv_gb_coxph",
    "python_sksurv_componentwise_gb_coxph",
    "python_lifelines_weibull_aft",
    "python_lifelines_lognormal_aft",
    "python_xgboost_aft",
}


@dataclass(frozen=True)
class Fold:
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass(frozen=True)
class SharedFoldArtifact:
    fold_id: int
    train_scaled_csv: Path
    test_scaled_csv: Path
    train_idx_path: Path
    test_idx_path: Path


def _coerce_positive_survival_times(df: pd.DataFrame, time_col: str, dataset_name: str) -> pd.DataFrame:
    time_vals = pd.to_numeric(df[time_col], errors="coerce")
    non_positive = time_vals <= 0.0
    if not bool(non_positive.any()):
        return df
    positive = time_vals[time_vals > 0.0]
    if positive.empty:
        raise RuntimeError(f"{dataset_name} has no strictly positive survival times")
    # Use a small dataset-scaled floor so every contender sees the same valid target
    # while minimally perturbing rows that represent effectively immediate events.
    replacement = max(float(positive.min()) * 0.5, 1e-12)
    adjusted = df.copy()
    adjusted.loc[non_positive, time_col] = replacement
    return adjusted


def _coerce_positive_survival_dataset_inplace(ds: dict, dataset_name: str) -> dict:
    if ds.get("family") != "survival":
        return ds
    time_col = ds["time_col"]
    rows_df = pd.DataFrame(ds["rows"])
    adjusted = _coerce_positive_survival_times(rows_df, time_col=time_col, dataset_name=dataset_name)
    if adjusted is rows_df:
        return ds
    ds["rows"] = adjusted.to_dict(orient="records")
    return ds


def _fmt_kib(kib):
    if kib is None:
        return "n/a"
    gib = float(kib) / (1024.0 * 1024.0)
    return f"{gib:.2f} GiB"


def _fmt_cpu_total_pct(cpu_pct):
    if cpu_pct in (None, "n/a"):
        return "n/a"
    try:
        cpu_val = float(cpu_pct)
    except Exception:
        return "n/a"
    ncpu = os.cpu_count() or 1
    return f"{(cpu_val / float(ncpu)):.1f}"


def _fmt_pct(numer, denom):
    try:
        if numer is None or denom in (None, 0):
            return "n/a"
        return f"{100.0 * float(numer) / float(denom):.1f}%"
    except Exception:
        return "n/a"


def _mem_used_kib(meminfo):
    total = meminfo.get("MemTotal")
    avail = meminfo.get("MemAvailable")
    if total is None or avail is None:
        return None
    used = int(total) - int(avail)
    return max(used, 0)


def _read_meminfo():
    out = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                if not rest:
                    continue
                val = rest.strip().split()[0]
                if val.isdigit():
                    out[key] = int(val)
    except Exception:
        return {}
    return out


def _read_proc_status_kib(pid, key):
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as fh:
            prefix = f"{key}:"
            for line in fh:
                if line.startswith(prefix):
                    val = line.split(":", 1)[1].strip().split()[0]
                    if val.isdigit():
                        return int(val)
    except Exception:
        return None
    return None


def _read_cgroup_memory_kib():
    paths = [
        ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"),
        ("/sys/fs/cgroup/memory/memory.usage_in_bytes", "/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ]
    for current_path, max_path in paths:
        try:
            current_raw = Path(current_path).read_text(encoding="utf-8").strip()
            max_raw = Path(max_path).read_text(encoding="utf-8").strip()
            current_kib = int(current_raw) // 1024 if current_raw.isdigit() else None
            if max_raw == "max":
                max_kib = None
            elif max_raw.isdigit():
                max_kib = int(max_raw) // 1024
            else:
                max_kib = None
            return current_kib, max_kib
        except Exception:
            continue
    return None, None


def _read_disk_usage_kib(path):
    try:
        usage = disk_usage(path)
    except Exception:
        return None, None, None
    return usage.total // 1024, usage.used // 1024, usage.free // 1024


def _read_ps_snapshot(pid):
    try:
        proc = subprocess.run(
            ["ps", "-p", str(pid), "-o", "pid=,%cpu=,%mem=,rss=,vsz=,etimes=,stat=,comm="],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        line = proc.stdout.strip()
        if not line:
            return {}
        parts = line.split(None, 7)
        if len(parts) < 8:
            return {}
        stat = parts[6]
        rss_kib = int(parts[3]) if parts[3].isdigit() else None
        vsz_kib = int(parts[4]) if parts[4].isdigit() else None
        # `ps` can report zombie tasks with zero RSS/VSZ; that is not a useful
        # memory sample for diagnostics, so report as missing.
        if stat.startswith("Z"):
            rss_kib = None
            vsz_kib = None
        return {
            "pid": parts[0],
            "cpu_pct": parts[1],
            "mem_pct": parts[2],
            "rss_kib": rss_kib,
            "vsz_kib": vsz_kib,
            "etimes": parts[5],
            "stat": stat,
            "comm": parts[7],
        }
    except Exception:
        return {}


def _collect_heartbeat_snapshot(proc, cmd_preview, stats, start):
    ps = _read_ps_snapshot(proc.pid)
    rss_kib = ps.get("rss_kib")
    if isinstance(rss_kib, int):
        stats["peak_proc_rss_kib"] = max(stats.get("peak_proc_rss_kib", 0), rss_kib)
    vsz_kib = ps.get("vsz_kib")
    if isinstance(vsz_kib, int):
        stats["peak_proc_vsz_kib"] = max(stats.get("peak_proc_vsz_kib", 0), vsz_kib)

    proc_hwm_kib = _read_proc_status_kib(proc.pid, "VmHWM")
    if isinstance(proc_hwm_kib, int):
        stats["peak_proc_hwm_kib"] = max(stats.get("peak_proc_hwm_kib", 0), proc_hwm_kib)
    py_rss_kib = _read_proc_status_kib(os.getpid(), "VmRSS")
    py_hwm_kib = _read_proc_status_kib(os.getpid(), "VmHWM")
    meminfo = _read_meminfo()
    mem_total_kib = meminfo.get("MemTotal")
    mem_avail_kib = meminfo.get("MemAvailable")
    mem_used_kib = _mem_used_kib(meminfo)
    cg_cur_kib, cg_max_kib = _read_cgroup_memory_kib()
    bench_total_kib, bench_used_kib, bench_free_kib = _read_disk_usage_kib(BENCH_DIR)
    tmp_total_kib, tmp_used_kib, tmp_free_kib = _read_disk_usage_kib(tempfile.gettempdir())
    elapsed = monotonic() - start
    load = os.getloadavg() if hasattr(os, "getloadavg") else (None, None, None)

    stats["samples"] = stats.get("samples", 0) + 1
    stats["last_proc_rss_kib"] = rss_kib
    stats["last_proc_vsz_kib"] = vsz_kib
    stats["last_proc_hwm_kib"] = proc_hwm_kib
    stats["last_py_rss_kib"] = py_rss_kib
    stats["last_py_hwm_kib"] = py_hwm_kib
    stats["last_cgroup_cur_kib"] = cg_cur_kib
    stats["last_cgroup_max_kib"] = cg_max_kib
    stats["last_mem_total_kib"] = mem_total_kib
    stats["last_mem_used_kib"] = mem_used_kib
    stats["last_mem_avail_kib"] = mem_avail_kib
    stats["last_bench_free_kib"] = bench_free_kib
    stats["last_bench_total_kib"] = bench_total_kib
    stats["last_tmp_free_kib"] = tmp_free_kib
    stats["last_tmp_total_kib"] = tmp_total_kib

    return (
        f"[HEARTBEAT] elapsed={elapsed:8.1f}s cmd='{cmd_preview}' "
        f"pid={proc.pid} p_cpu={_fmt_cpu_total_pct(ps.get('cpu_pct', 'n/a'))}% "
        f"p_mem={ps.get('mem_pct', 'n/a')}% "
        f"p_rss={_fmt_kib(rss_kib)} p_hwm={_fmt_kib(proc_hwm_kib)} "
        f"p_vsz={_fmt_kib(vsz_kib)} py_rss={_fmt_kib(py_rss_kib)} "
        f"py_hwm={_fmt_kib(py_hwm_kib)} "
        f"sys_ram={_fmt_kib(mem_used_kib)}/{_fmt_kib(mem_total_kib)} "
        f"sys_ram_pct={_fmt_pct(mem_used_kib, mem_total_kib)} "
        f"sys_avail={_fmt_kib(meminfo.get('MemAvailable'))} "
        f"swap_free={_fmt_kib(meminfo.get('SwapFree'))} "
        f"cgroup={_fmt_kib(cg_cur_kib)}/{_fmt_kib(cg_max_kib)} "
        f"cg_pct={_fmt_pct(cg_cur_kib, cg_max_kib)} "
        f"bench_free={_fmt_kib(bench_free_kib)}/{_fmt_kib(bench_total_kib)} "
        f"bench_used={_fmt_pct(bench_used_kib, bench_total_kib)} "
        f"tmp_free={_fmt_kib(tmp_free_kib)}/{_fmt_kib(tmp_total_kib)} "
        f"tmp_used={_fmt_pct(tmp_used_kib, tmp_total_kib)} "
        f"load1={load[0] if load[0] is not None else 'n/a'}"
    )


def _heartbeat_loop(proc, cmd, stop_event, stats):
    cmd_preview = " ".join(str(x) for x in cmd[:5])
    if len(cmd) > 5:
        cmd_preview += " ..."
    start = monotonic()
    print(_collect_heartbeat_snapshot(proc, cmd_preview, stats, start), file=sys.stderr, flush=True)
    while True:
        elapsed = monotonic() - start
        wait_sec = (
            HEARTBEAT_INITIAL_INTERVAL_SEC
            if elapsed < HEARTBEAT_INITIAL_WINDOW_SEC
            else HEARTBEAT_INTERVAL_SEC
        )
        if stop_event.wait(wait_sec):
            break
        if proc.poll() is not None:
            break
        print(_collect_heartbeat_snapshot(proc, cmd_preview, stats, start), file=sys.stderr, flush=True)


def run_cmd(cmd, cwd=None):
    env = os.environ.copy()
    env.update(_SERIAL_ENV_OVERRIDES)
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        bufsize=1,
    )

    out_buf = []
    err_buf = []
    hb_stop = threading.Event()
    hb_stats = {
        "peak_proc_rss_kib": 0,
        "peak_proc_hwm_kib": 0,
        "peak_proc_vsz_kib": 0,
        "samples": 0,
    }

    def _pump(stream, sink, buf):
        try:
            for line in iter(stream.readline, ""):
                sink.write(line)
                sink.flush()
                buf.append(line)
        finally:
            stream.close()

    t_out = threading.Thread(target=_pump, args=(proc.stdout, sys.stdout, out_buf), daemon=True)
    t_err = threading.Thread(target=_pump, args=(proc.stderr, sys.stderr, err_buf), daemon=True)
    t_hb = threading.Thread(target=_heartbeat_loop, args=(proc, cmd, hb_stop, hb_stats), daemon=True)
    t_out.start()
    t_err.start()
    t_hb.start()
    timed_out = False
    try:
        if _CMD_TIMEOUT_SEC > 0:
            rc = proc.wait(timeout=float(_CMD_TIMEOUT_SEC))
        else:
            rc = proc.wait()
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            proc.terminate()
            proc.wait(timeout=10.0)
        except Exception:
            proc.kill()
            proc.wait()
        rc = 124
    hb_stop.set()
    t_out.join()
    t_err.join()
    t_hb.join(timeout=1.0)
    if timed_out:
        timeout_msg = (
            f"[HEARTBEAT] command-timeout rc=124 timeout_sec={_CMD_TIMEOUT_SEC} "
            f"pid={proc.pid} cmd='{(' '.join(str(x) for x in cmd[:5]) + (' ...' if len(cmd) > 5 else ''))}'\n"
        )
        err_buf.append(timeout_msg)
        print(timeout_msg, file=sys.stderr, flush=True)
    print(
        f"[HEARTBEAT] command-exit rc={rc} pid={proc.pid} "
        f"samples={hb_stats.get('samples', 0)} "
        f"peak_proc_rss={_fmt_kib(hb_stats.get('peak_proc_rss_kib', 0))} "
        f"peak_proc_hwm={_fmt_kib(hb_stats.get('peak_proc_hwm_kib', 0))} "
        f"peak_proc_vsz={_fmt_kib(hb_stats.get('peak_proc_vsz_kib', 0))} "
        f"last_proc_rss={_fmt_kib(hb_stats.get('last_proc_rss_kib'))} "
        f"last_py_rss={_fmt_kib(hb_stats.get('last_py_rss_kib'))} "
        f"last_sys_ram={_fmt_kib(hb_stats.get('last_mem_used_kib'))}/{_fmt_kib(hb_stats.get('last_mem_total_kib'))} "
        f"last_sys_ram_pct={_fmt_pct(hb_stats.get('last_mem_used_kib'), hb_stats.get('last_mem_total_kib'))} "
        f"last_cgroup={_fmt_kib(hb_stats.get('last_cgroup_cur_kib'))}/{_fmt_kib(hb_stats.get('last_cgroup_max_kib'))} "
        f"last_bench_free={_fmt_kib(hb_stats.get('last_bench_free_kib'))}/{_fmt_kib(hb_stats.get('last_bench_total_kib'))} "
        f"last_tmp_free={_fmt_kib(hb_stats.get('last_tmp_free_kib'))}/{_fmt_kib(hb_stats.get('last_tmp_total_kib'))}",
        file=sys.stderr,
        flush=True,
    )
    return rc, "".join(out_buf), "".join(err_buf)


def make_folds(y: np.ndarray, n_splits: int = 5, seed: int = 42, stratified: bool = False):
    n = int(len(y))
    if n < n_splits:
        raise ValueError(f"Need at least {n_splits} rows for {n_splits}-fold CV; got {n}")
    rng = np.random.default_rng(seed)

    if n_splits == 1:
        if n < 2:
            raise ValueError("Need at least 2 rows for a holdout split")
        test_fraction = 0.2
        if stratified:
            classes = np.unique(y)
            if len(classes) >= 2:
                test_rows = []
                for c in classes:
                    idx = np.where(y == c)[0]
                    rng.shuffle(idx)
                    if len(idx) < 2:
                        raise ValueError(
                            "Need at least 2 observations in each class for a stratified holdout split"
                        )
                    n_test = max(1, int(round(len(idx) * test_fraction)))
                    n_test = min(n_test, len(idx) - 1)
                    test_rows.extend(idx[:n_test].tolist())
                test_idx = np.array(sorted(test_rows), dtype=int)
                train_mask = np.ones(n, dtype=bool)
                train_mask[test_idx] = False
                all_idx = np.arange(n, dtype=int)
                return [Fold(train_idx=all_idx[train_mask], test_idx=test_idx)]
        perm = np.arange(n, dtype=int)
        rng.shuffle(perm)
        n_test = min(max(1, int(round(n * test_fraction))), n - 1)
        test_idx = np.sort(perm[:n_test])
        all_idx = np.arange(n, dtype=int)
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        return [Fold(train_idx=all_idx[train_mask], test_idx=test_idx)]

    if stratified:
        classes = np.unique(y)
        if len(classes) >= 2:
            fold_bins = [[] for _ in range(n_splits)]
            for c in classes:
                idx = np.where(y == c)[0]
                rng.shuffle(idx)
                for i, row_idx in enumerate(idx.tolist()):
                    fold_bins[i % n_splits].append(row_idx)
            all_idx = np.arange(n, dtype=int)
            folds = []
            for fold_rows in fold_bins:
                test_idx = np.array(sorted(fold_rows), dtype=int)
                train_mask = np.ones(n, dtype=bool)
                train_mask[test_idx] = False
                folds.append(Fold(train_idx=all_idx[train_mask], test_idx=test_idx))
            return folds

    perm = np.arange(n, dtype=int)
    rng.shuffle(perm)
    test_parts = np.array_split(perm, n_splits)
    all_idx = np.arange(n, dtype=int)
    folds = []
    for test_idx in test_parts:
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        folds.append(Fold(train_idx=all_idx[train_mask], test_idx=np.sort(test_idx)))
    return folds


def _evaluation_suffix(folds: list[Fold]) -> str:
    if len(folds) == 1:
        return "[holdout]"
    return f"[{len(folds)}-fold CV]"


def _evaluation_label(folds: list[Fold]) -> str:
    if len(folds) == 1:
        return "holdout"
    return f"{len(folds)}-fold CV"


def auc_score(y: np.ndarray, p: np.ndarray) -> float:
    y_bin = (np.asarray(y) > 0.5).astype(int)
    n_pos = int(np.sum(y_bin == 1))
    n_neg = int(np.sum(y_bin == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float(roc_auc_score(y_bin, np.asarray(p, dtype=float)))


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((y - p) ** 2))


def log_loss_score(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    p_clipped = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p_clipped) + (1.0 - y) * np.log(1.0 - p_clipped)))


def exp_saturated(x: float) -> float:
    """Numerically stable exp that saturates instead of raising OverflowError."""
    # Approximate IEEE-754 f64 exponent bounds.
    if x >= 709.0:
        return float("inf")
    if x <= -745.0:
        return 0.0
    return math.exp(x)


def nagelkerke_r2_score(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float | None:
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    p_arr = np.clip(np.asarray(p, dtype=float).reshape(-1), eps, 1.0 - eps)
    if y_arr.shape != p_arr.shape or y_arr.size == 0:
        return None
    p_mean = float(np.mean(y_arr))
    if not np.isfinite(p_mean) or p_mean <= 0.0 or p_mean >= 1.0:
        return None
    ll_null = float(np.sum(y_arr * math.log(p_mean) + (1.0 - y_arr) * math.log(1.0 - p_mean)))
    ll_model = float(np.sum(y_arr * np.log(p_arr) + (1.0 - y_arr) * np.log(1.0 - p_arr)))
    n = int(y_arr.size)
    r2_cs = 1.0 - exp_saturated((2.0 / n) * (ll_null - ll_model))
    max_r2_cs = 1.0 - exp_saturated((2.0 / n) * ll_null)
    if not np.isfinite(r2_cs) or not np.isfinite(max_r2_cs) or max_r2_cs <= 0.0:
        return None
    return float(r2_cs / max_r2_cs)


def _survival_partial_loglik_stats(
    event_times: np.ndarray, risk_score: np.ndarray, events: np.ndarray, eps: float = 1e-12
) -> tuple[float | None, int]:
    times = np.asarray(event_times, dtype=float).reshape(-1)
    eta = np.asarray(risk_score, dtype=float).reshape(-1)
    obs = (np.asarray(events, dtype=float).reshape(-1) > 0.5).astype(float)
    if times.shape != eta.shape or times.shape != obs.shape or times.size == 0:
        return None, 0

    event_mask = obs > 0.5
    if int(np.sum(event_mask)) == 0:
        return None, 0

    unique_event_times = np.unique(times[event_mask])
    total = 0.0
    n_events = 0
    for t in unique_event_times:
        d = (times == t) & event_mask
        m = int(np.sum(d))
        if m == 0:
            continue
        risk_set = times >= t
        risk_eta = eta[risk_set]
        if risk_eta.size == 0:
            continue
        max_eta = float(np.max(risk_eta))
        log_denom = max_eta + math.log(float(np.sum(np.exp(risk_eta - max_eta))) + eps)
        total += float(np.sum(eta[d])) - float(m) * log_denom
        n_events += m

    if n_events == 0:
        return None, 0
    return float(total), n_events


def survival_partial_log_loss(
    event_times: np.ndarray, risk_score: np.ndarray, events: np.ndarray, eps: float = 1e-12
) -> float | None:
    total, n_events = _survival_partial_loglik_stats(event_times, risk_score, events, eps=eps)
    if total is None or n_events == 0:
        return None
    return float(-total / n_events)


def survival_partial_nagelkerke_r2(
    event_times: np.ndarray, risk_score: np.ndarray, events: np.ndarray, eps: float = 1e-12
) -> float | None:
    ll_model, n_events = _survival_partial_loglik_stats(event_times, risk_score, events, eps=eps)
    if ll_model is None or n_events == 0:
        return None
    ll_null, _ = _survival_partial_loglik_stats(
        event_times,
        np.zeros_like(np.asarray(risk_score, dtype=float).reshape(-1)),
        events,
        eps=eps,
    )
    if ll_null is None:
        return None
    r2_cs = 1.0 - exp_saturated((2.0 / n_events) * (ll_null - ll_model))
    max_r2_cs = 1.0 - exp_saturated((2.0 / n_events) * ll_null)
    if not np.isfinite(r2_cs) or not np.isfinite(max_r2_cs) or max_r2_cs <= 0.0:
        return None
    return float(r2_cs / max_r2_cs)


def survival_partial_brier_score(
    event_times: np.ndarray, risk_score: np.ndarray, events: np.ndarray, eps: float = 1e-12
) -> float | None:
    times = np.asarray(event_times, dtype=float).reshape(-1)
    eta = np.asarray(risk_score, dtype=float).reshape(-1)
    obs = (np.asarray(events, dtype=float).reshape(-1) > 0.5).astype(float)
    if times.shape != eta.shape or times.shape != obs.shape or times.size == 0:
        return None

    event_mask = obs > 0.5
    if int(np.sum(event_mask)) == 0:
        return None

    unique_event_times = np.unique(times[event_mask])
    total = 0.0
    denom = 0
    for t in unique_event_times:
        d = (times == t) & event_mask
        m = int(np.sum(d))
        if m == 0:
            continue
        risk_set = times >= t
        risk_eta = eta[risk_set]
        if risk_eta.size == 0:
            continue
        max_eta = float(np.max(risk_eta))
        weights = np.exp(risk_eta - max_eta)
        probs = weights / max(float(np.sum(weights)), eps)
        target = np.zeros_like(probs, dtype=float)
        target[d[risk_set]] = 1.0 / m
        score = float(np.sum((probs - target) ** 2))
        total += float(m) * score
        denom += m

    if denom == 0:
        return None
    return float(total / denom)


def rmse_score(y: np.ndarray, mu: np.ndarray) -> float:
    return math.sqrt(float(np.mean((y - mu) ** 2)))


def mse_score(y: np.ndarray, mu: np.ndarray) -> float:
    return float(np.mean((y - mu) ** 2))


def gaussian_log_loss_score(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray | float, eps: float = 1e-12) -> float:
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    mu_arr = np.asarray(mu, dtype=float).reshape(-1)
    sigma_arr = np.asarray(sigma, dtype=float)
    if sigma_arr.ndim == 0:
        sigma_use = np.full_like(y_arr, float(sigma_arr), dtype=float)
    else:
        sigma_use = sigma_arr.reshape(-1)
    sigma_use = np.maximum(sigma_use, eps)
    var = sigma_use**2
    return float(np.mean(0.5 * np.log(2.0 * math.pi * var) + ((y_arr - mu_arr) ** 2) / (2.0 * var)))


def mae_score(y: np.ndarray, mu: np.ndarray) -> float:
    return float(np.mean(np.abs(y - mu)))


def zscore_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = train_df.copy()
    te = test_df.copy()
    for col in feature_cols:
        mu = float(tr[col].mean())
        sdv = float(tr[col].std())
        if (not np.isfinite(sdv)) or sdv < 1e-8:
            sdv = 1.0
        tr[col] = (tr[col] - mu) / sdv
        te[col] = (te[col] - mu) / sdv
    return tr, te


def r2_score(y: np.ndarray, mu: np.ndarray) -> float:
    sst = float(np.sum((y - float(np.mean(y))) ** 2))
    if sst <= 0.0:
        return 0.0
    sse = float(np.sum((y - mu) ** 2))
    return 1.0 - sse / sst


def _survival_risk_from_rust_pred(pred_df: pd.DataFrame) -> tuple[np.ndarray, str]:
    # Canonical survival ranking: model-native risk score (higher => earlier failure).
    for col in ("risk_score", "eta", "failure_prob"):
        if col in pred_df.columns:
            return pred_df[col].to_numpy(dtype=float), col
    raise RuntimeError(
        "rust survival prediction output missing required risk column; "
        "expected one of: risk_score, eta, failure_prob"
    )


def _lifelines_cindex_from_risk(event_times: np.ndarray, risk_score: np.ndarray, events: np.ndarray) -> float:
    # Convert risk (higher => earlier failure) to lifelines survival score.
    return float(concordance_index(event_times, -risk_score, event_observed=events))


def _survival_eval_horizon(train_df: pd.DataFrame, time_col: str) -> float:
    # Fold-specific reference horizon from train data only, used only when a
    # backend prediction API requires an explicit time input.
    horizon = float(np.median(train_df[time_col].to_numpy(dtype=float)))
    if (not np.isfinite(horizon)) or horizon <= 0.0:
        horizon = 1.0
    return horizon


def _augment_bmi_spline_linear_hinges(train_df: pd.DataFrame, test_df: pd.DataFrame, n_knots: int = 6):
    if "bmi" not in train_df.columns or n_knots < 2:
        return train_df, test_df, []
    x_tr = train_df["bmi"].to_numpy(dtype=float)
    # Leakage-safe: knots from train-fold only.
    qs = np.linspace(0.1, 0.9, n_knots - 1)
    knots = np.unique(np.quantile(x_tr, qs))

    tr = train_df.copy()
    te = test_df.copy()
    cols = []
    base_col = "bmi_spline_0"
    tr[base_col] = tr["bmi"]
    te[base_col] = te["bmi"]
    cols.append(base_col)
    for j, k in enumerate(knots, start=1):
        cn = f"bmi_spline_{j}"
        tr[cn] = np.maximum(0.0, tr["bmi"] - float(k))
        te[cn] = np.maximum(0.0, te["bmi"] - float(k))
        cols.append(cn)
    tr = tr.drop(columns=["bmi"])
    te = te.drop(columns=["bmi"])
    return tr, te, cols


def _load_lidar_dataset():
    d = pd.read_csv(DATASET_DIR / "lidar.csv")
    d = d[["range", "logratio"]].dropna()
    rows = [{"range": float(r), "y": float(y)} for r, y in zip(d["range"], d["logratio"])]
    return {
        "family": "gaussian",
        "rows": rows,
        "features": ["range"],
        "target": "y",
    }


def _load_bone_dataset():
    d = pd.read_csv(DATASET_DIR / "bone.csv")
    d = d[["trt", "t", "d"]].dropna()
    rows = [
        {
            "trt_auto": 1.0 if str(trt).strip().strip('"').lower() == "auto" else 0.0,
            "t": float(t),
            "y": float(y),
        }
        for trt, t, y in zip(d["trt"], d["t"], d["d"])
    ]
    return {
        "family": "binomial",
        "rows": rows,
        "features": ["trt_auto", "t"],
        "target": "y",
    }


def _load_prostate_dataset():
    d = pd.read_csv(DATASET_DIR / "prostate.csv")
    d = d[["pc1", "pc2", "y"]].dropna()
    rows = [{"pc1": float(a), "pc2": float(b), "y": float(y)} for a, b, y in zip(d["pc1"], d["pc2"], d["y"])]
    return {
        "family": "binomial",
        "rows": rows,
        "features": ["pc1", "pc2"],
        "target": "y",
    }


def _load_wine_dataset():
    d = pd.read_csv(DATASET_DIR / "wine.csv")
    d = d[["year", "h_rain", "w_rain", "h_temp", "s_temp", "price"]]
    d = d.replace({"NA": np.nan}).dropna(subset=["price"])
    rows = [
        {
            "year": float(y),
            "h_rain": float(hr),
            "w_rain": float(wr),
            "h_temp": float(ht),
            "s_temp": float(st),
            "price": float(p),
        }
        for y, hr, wr, ht, st, p in zip(
            d["year"], d["h_rain"], d["w_rain"], d["h_temp"], d["s_temp"], d["price"]
        )
    ]
    return {
        "family": "gaussian",
        "rows": rows,
        "features": ["year", "h_rain", "w_rain", "h_temp", "s_temp"],
        "target": "price",
    }


def _load_wine_temp_vs_year_dataset():
    d = pd.read_csv(DATASET_DIR / "wine.csv")
    d = d[["year", "s_temp"]].dropna()
    rows = [{"year": float(y), "s_temp": float(t)} for y, t in zip(d["year"], d["s_temp"])]
    return {
        "family": "gaussian",
        "rows": rows,
        "features": ["year"],
        "target": "s_temp",
    }


def _load_wine_price_vs_temp_dataset():
    d = pd.read_csv(DATASET_DIR / "wine.csv")
    d = d[["s_temp", "price"]].replace({"NA": np.nan}).dropna()
    rows = [{"temp": float(t), "price": float(p)} for t, p in zip(d["s_temp"], d["price"])]
    return {
        "family": "gaussian",
        "rows": rows,
        "features": ["temp"],
        "target": "price",
    }


def _load_horse_dataset():
    d = pd.read_csv(DATASET_DIR / "horse.csv")
    d = d[["outcome", "pulse", "rectal_temp", "packed_cell_volume"]].copy()
    d["outcome"] = d["outcome"].astype(str).str.strip().str.lower()
    d = d[d["outcome"].isin({"lived", "died", "euthanized"})]
    for c in ["pulse", "rectal_temp", "packed_cell_volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["pulse", "rectal_temp", "packed_cell_volume"])
    rows = [
        {
            "rectal_temp": float(rt),
            "pulse": float(pu),
            "packed_cell_volume": float(pcv),
            "y": 0.0 if out == "lived" else 1.0,
        }
        for out, pu, rt, pcv in zip(d["outcome"], d["pulse"], d["rectal_temp"], d["packed_cell_volume"])
    ]
    if not rows:
        raise RuntimeError("horse dataset has no complete rows after filtering")
    return {
        "family": "binomial",
        "rows": rows,
        "features": ["rectal_temp", "pulse", "packed_cell_volume"],
        "target": "y",
    }


def _parse_f64_opt(raw):
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() == "na":
        return None
    try:
        v = float(s)
        if not np.isfinite(v):
            return None
        return v
    except ValueError:
        return None


def _encode_bool_yn(raw: str) -> float:
    return 1.0 if (raw or "").strip().upper() == "Y" else 0.0


def _encode_edema(raw: str) -> float:
    t = (raw or "").strip().upper()
    if t == "Y":
        return 2.0
    if t == "S":
        return 1.0
    return 0.0


def _load_cirrhosis_survival_dataset():
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

        time = _parse_f64_opt(r.get("N_Days", ""))
        if time is None:
            continue
        time = max(time, 1.0)

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

        parsed_num = {}
        ok = True
        for col in numeric_cols:
            v = _parse_f64_opt(r.get(col, ""))
            if v is None:
                ok = False
                break
            parsed_num[col.lower()] = float(v)
        if not ok:
            continue

        out = {
            "time": float(time),
            "event": 1.0 if status == "D" else 0.0,
            "drug": 1.0 if "penicillamine" in drug.lower() else 0.0,
            "sex_male": 1.0 if sex == "M" else 0.0,
            "ascites": _encode_bool_yn(asc),
            "hepatomegaly": _encode_bool_yn(hep),
            "spiders": _encode_bool_yn(spi),
            "edema": _encode_edema(ede),
            **parsed_num,
        }
        rows.append(out)

    if not rows:
        raise RuntimeError("cirrhosis.csv has no complete-case rows for CV benchmarking")

    return {
        "family": "survival",
        "rows": rows,
        "features": [
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
        "time_col": "time",
        "event_col": "event",
    }


def _normalize_us48_timestamp_to_hour(ts: str) -> float:
    # Example input: "2/21/2026 12 a.m. EST"
    cleaned = (
        ts.replace(" a.m.", " AM")
        .replace(" p.m.", " PM")
        .replace(" EST", "")
        .strip()
    )
    dt = pd.to_datetime(cleaned, format="%m/%d/%Y %I %p", errors="coerce")
    if pd.isna(dt):
        raise ValueError(f"failed to parse timestamp '{ts}'")
    return float(dt.hour)


def _load_us48_demand_dataset(filename: str):
    d = pd.read_csv(DATASET_DIR / filename)
    needed = [
        "Timestamp (Hour Ending)",
        "Demand (MWh)",
        "Demand Forecast (MWh)",
        "Net Generation (MWh)",
        "Total Interchange (MWh)",
    ]
    d = d[needed].copy()
    for c in needed[1:]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=needed[1:])
    rows = [
        {
            "hour": _normalize_us48_timestamp_to_hour(str(ts)),
            "demand_forecast": float(f),
            "net_generation": float(g),
            "total_interchange": float(i),
            "y": float(y),
        }
        for ts, y, f, g, i in zip(
            d["Timestamp (Hour Ending)"],
            d["Demand (MWh)"],
            d["Demand Forecast (MWh)"],
            d["Net Generation (MWh)"],
            d["Total Interchange (MWh)"],
        )
    ]
    if not rows:
        raise RuntimeError(f"dataset '{filename}' has no complete rows")
    return {
        "family": "gaussian",
        "rows": rows,
        "features": ["hour", "demand_forecast", "net_generation", "total_interchange"],
        "target": "y",
    }


def _load_haberman_dataset():
    d = pd.read_csv(DATASET_DIR / "haberman.csv")
    d = d.iloc[:, :4].copy()
    d.columns = ["age", "op_year", "axil_nodes", "status"]
    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna()
    rows = [
        {
            "age": float(a),
            "op_year": float(y),
            "axil_nodes": float(n),
            "y": 1.0 if int(s) == 2 else 0.0,
        }
        for a, y, n, s in zip(d["age"], d["op_year"], d["axil_nodes"], d["status"])
    ]
    if not rows:
        raise RuntimeError("haberman.csv has no rows")
    return {
        "family": "binomial",
        "rows": rows,
        "features": ["age", "op_year", "axil_nodes"],
        "target": "y",
    }


def _load_icu_survival_death_dataset():
    d = pd.read_csv(DATASET_DIR / "icu_survival_death.csv")
    d = d[["time", "age", "bmi", "hr_max", "sysbp_min", "event"]].dropna()
    d = _coerce_positive_survival_times(d, time_col="time", dataset_name="icu_survival_death")
    rows = [
        {
            "time": float(t),
            "age": float(a),
            "bmi": float(b),
            "hr_max": float(h),
            "sysbp_min": float(s),
            "event": float(e),
        }
        for t, a, b, h, s, e in zip(
            d["time"], d["age"], d["bmi"], d["hr_max"], d["sysbp_min"], d["event"]
        )
    ]
    if not rows:
        raise RuntimeError("icu_survival_death dataset has no rows")
    return {
        "family": "survival",
        "rows": rows,
        "features": ["age", "bmi", "hr_max", "sysbp_min"],
        "time_col": "time",
        "event_col": "event",
    }


def _load_icu_survival_los_dataset():
    d = pd.read_csv(DATASET_DIR / "icu_survival_los.csv")
    d = d[["age", "bmi", "hr_max", "sysbp_min", "temp_apache", "time", "event"]].dropna()
    d = _coerce_positive_survival_times(d, time_col="time", dataset_name="icu_survival_los")
    rows = [
        {
            "age": float(a),
            "bmi": float(b),
            "hr_max": float(h),
            "sysbp_min": float(s),
            "temp_apache": float(t),
            "time": float(tm),
            "event": float(e),
        }
        for a, b, h, s, t, tm, e in zip(
            d["age"], d["bmi"], d["hr_max"], d["sysbp_min"], d["temp_apache"], d["time"], d["event"]
        )
    ]
    if not rows:
        raise RuntimeError("icu_survival_los dataset has no rows")
    return {
        "family": "survival",
        "rows": rows,
        "features": ["age", "bmi", "hr_max", "sysbp_min", "temp_apache"],
        "time_col": "time",
        "event_col": "event",
    }


def _load_heart_failure_survival_dataset():
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
    d = d.dropna()
    rows = []
    for r in d.itertuples(index=False):
        cpk = float(r.creatinine_phosphokinase)
        plt = float(r.platelets)
        scr = float(r.serum_creatinine)
        rows.append(
            {
                "time": float(r.time),
                "event": float(r.DEATH_EVENT),
                "age": float(r.age),
                "anaemia": float(r.anaemia),
                "log_creatinine_phosphokinase": float(np.log1p(max(cpk, 0.0))),
                "diabetes": float(r.diabetes),
                "ejection_fraction": float(r.ejection_fraction),
                "high_blood_pressure": float(r.high_blood_pressure),
                "log_platelets": float(np.log1p(max(plt, 0.0))),
                "log_serum_creatinine": float(np.log1p(max(scr, 0.0))),
                "serum_sodium": float(r.serum_sodium),
                "sex": float(r.sex),
                "smoking": float(r.smoking),
            }
        )
    if not rows:
        raise RuntimeError("heart_failure_clinical_records_dataset.csv has no rows")
    return {
        "family": "survival",
        "rows": rows,
        "features": [
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
        "time_col": "time",
        "event_col": "event",
    }


def _synthetic_binomial_dataset(n, p, seed):
    p = max(int(p), 3)
    rng = np.random.default_rng(int(seed))
    x = np.zeros((int(n), p), dtype=float)
    x[:, 0] = 1.0
    if p > 1:
        x[:, 1:] = rng.normal(size=(int(n), p - 1))

    eta = -0.25 + 1.1 * x[:, 1] - 0.9 * x[:, 2] + 0.2 * np.sin(x[:, 2])
    pr = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.random(int(n)) < pr).astype(float)

    rows = []
    for i in range(int(n)):
        rows.append({"x1": float(x[i, 1]), "x2": float(x[i, 2]), "y": float(y[i])})

    return {
        "family": "binomial",
        "rows": rows,
        "features": ["x1", "x2"],
        "target": "y",
    }


_CANONICAL_SYNTHETIC_BINOMIAL_SCENARIOS = {
    "small_dense": {"n": 1000, "p": 10, "seed": 7},
    "medium": {"n": 50000, "p": 50, "seed": 11},
    "pathological_ill_conditioned": {"n": 50000, "p": 80, "seed": 17},
}


def _synthetic_geo_disease_dataset(n=4000, seed=20260226):
    n = int(max(500, n))
    rng = np.random.default_rng(int(seed))

    # Hidden geospatial drivers (never exposed as model features).
    lat = rng.uniform(-1.0, 1.0, size=n)
    lon = rng.uniform(-1.0, 1.0, size=n)

    # Disease prevalence is highest near the equator (lat ~= 0).
    equator_closeness = 1.0 - np.abs(lat)
    geo_signal = (
        -1.00
        + 2.20 * equator_closeness
        + 0.55 * np.sin(np.pi * lon)
        + 0.35 * np.cos(2.25 * np.pi * lon)
        + 0.30 * np.sin(2.0 * np.pi * equator_closeness * lon)
    )

    # Heteroscedastic latent noise: stronger farther south (lat < 0).
    southness = np.clip(-lat, 0.0, 1.0)
    eta_noise_sd = 0.20 + 0.85 * (southness**1.35)
    eta = geo_signal + rng.normal(0.0, eta_noise_sd, size=n)
    pr = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.random(n) < pr).astype(float)

    # PCs are noisy transforms of hidden geo variables.
    rows = []
    for i in range(n):
        row = {}
        for j in range(16):
            a = 0.95 - 0.045 * j
            b = 0.25 + 0.035 * j
            c = ((-1.0) ** j) * (0.10 + 0.01 * j)
            noise_sd = 0.15 + 0.015 * j
            pc = a * lat[i] + b * lon[i] + c * lat[i] * lon[i] + rng.normal(0.0, noise_sd)
            row[f"pc{j + 1}"] = float(pc)
        row["y"] = float(y[i])
        rows.append(row)

    return {
        "family": "binomial",
        "rows": rows,
        "features": [f"pc{i}" for i in range(1, 17)],
        "target": "y",
    }


def _sample_fractional_spde_1d_field(n: int, *, rng: np.random.Generator, nu: float, kappa2: float) -> np.ndarray:
    n = int(max(32, n))
    # Frequency-domain SPDE approximation in 1D:
    #   S(omega) ∝ (kappa^2 + omega^2)^(-(nu + 1/2))
    # This yields a stationary fractional field with controllable smoothness.
    freqs = 2.0 * math.pi * np.fft.rfftfreq(n, d=1.0 / n)
    spec = (max(float(kappa2), 1e-9) + freqs * freqs) ** (-(float(nu) + 0.5))
    spec = np.clip(spec, 1e-18, None)
    coeff = np.zeros(freqs.shape[0], dtype=np.complex128)
    coeff[0] = rng.normal(0.0, math.sqrt(spec[0]))
    for j in range(1, coeff.shape[0]):
        s = math.sqrt(spec[j] * 0.5)
        coeff[j] = rng.normal(0.0, s) + 1j * rng.normal(0.0, s)
    field = np.fft.irfft(coeff, n=n).astype(float)
    field = field - float(np.mean(field))
    sd = float(np.std(field))
    if (not np.isfinite(sd)) or sd < 1e-12:
        return np.zeros(n, dtype=float)
    return field / sd


def _synthetic_thread2_continuous_order_dataset(
    *,
    mode: str,
    n: int = 512,
    seed: int = 20260501,
    true_nu: float | None = None,
    true_kappa2: float | None = None,
) -> dict:
    n = int(max(128, n))
    rng = np.random.default_rng(int(seed))
    x = np.linspace(-1.0, 1.0, n, dtype=float)
    if mode == "fractional":
        nu = float(true_nu if true_nu is not None else 1.8)
        k2 = float(true_kappa2 if true_kappa2 is not None else 0.7)
        latent = _sample_fractional_spde_1d_field(n, rng=rng, nu=nu, kappa2=k2)
        y = latent + rng.normal(0.0, 0.20, size=n)
        expected = ["Ok", "NonMaternRegime"]
    elif mode == "rough":
        # Brownian-like path to stress non-Matern / first-order boundary logic.
        steps = rng.normal(0.0, 1.0, size=n)
        latent = np.cumsum(steps)
        latent = (latent - float(np.mean(latent))) / max(float(np.std(latent)), 1e-12)
        y = latent + rng.normal(0.0, 0.30, size=n)
        nu = None
        k2 = None
        expected = ["NonMaternRegime", "FirstOrderLimit", "IntrinsicLimit"]
    elif mode == "smooth":
        latent = 1.4 * np.sin(2.0 * math.pi * (x + 0.1)) + 0.8 * np.cos(0.5 * math.pi * (x - 0.2))
        latent = (latent - float(np.mean(latent))) / max(float(np.std(latent)), 1e-12)
        y = latent + rng.normal(0.0, 0.03, size=n)
        nu = None
        k2 = None
        expected = ["Ok", "IntrinsicLimit"]
    else:
        raise RuntimeError(f"unsupported thread2 synthetic mode '{mode}'")

    rows = [{"x": float(xi), "y": float(yi)} for xi, yi in zip(x, y)]
    out = {
        "family": "gaussian",
        "rows": rows,
        "features": ["x"],
        "target": "y",
        "thread2_expected_statuses": expected,
    }
    if true_nu is not None:
        out["thread2_true_nu"] = float(true_nu)
    if true_kappa2 is not None:
        out["thread2_true_kappa2"] = float(true_kappa2)
    return out


def _synthetic_thread3_admixture_cliff_dataset(n=6000, seed=20260601):
    n = int(max(500, n))
    rng = np.random.default_rng(int(seed))

    # Correlated ancestry-like latent coordinates.
    cov = np.array(
        [
            [1.00, 0.52, -0.18, 0.10],
            [0.52, 1.00, 0.22, -0.15],
            [-0.18, 0.22, 1.00, 0.35],
            [0.10, -0.15, 0.35, 1.00],
        ],
        dtype=float,
    )
    core = rng.multivariate_normal(mean=np.zeros(4), cov=cov, size=n)
    pc1, pc2, pc3, pc4 = core[:, 0], core[:, 1], core[:, 2], core[:, 3]

    # Narrow boundary in admixture-space: almost flat away from the interface.
    coeffs = np.array([1.0, 0.35, -0.20, 0.10], dtype=float)
    z = coeffs[0] * pc1 + coeffs[1] * pc2 + coeffs[2] * pc3 + coeffs[3] * pc4
    cliff_jump = 3.8
    cliff_sharpness = 16.0
    eta = -1.15 + cliff_jump * np.tanh(cliff_sharpness * z) + rng.normal(0.0, 0.15, size=n)
    pr = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.random(n) < pr).astype(float)

    # Add nuisance PCs to match the geo disease benchmark layout.
    nuisance = np.zeros((n, 12), dtype=float)
    for j in range(12):
        a = 0.45 - 0.02 * j
        b = ((-1.0) ** j) * (0.18 + 0.01 * j)
        c = 0.12 + 0.02 * (j % 4)
        sd = 0.18 + 0.02 * j
        nuisance[:, j] = a * pc1 + b * pc2 + c * pc4 + rng.normal(0.0, sd, size=n)

    rows = []
    for i in range(n):
        row = {
            "pc1": float(pc1[i]),
            "pc2": float(pc2[i]),
            "pc3": float(pc3[i]),
            "pc4": float(pc4[i]),
        }
        for j in range(12):
            row[f"pc{j + 5}"] = float(nuisance[i, j])
        row["y"] = float(y[i])
        rows.append(row)

    return {
        "family": "binomial",
        "rows": rows,
        "features": [f"pc{i}" for i in range(1, 17)],
        "target": "y",
        "thread3_cliff_coefficients": {
            "pc1": float(coeffs[0]),
            "pc2": float(coeffs[1]),
            "pc3": float(coeffs[2]),
            "pc4": float(coeffs[3]),
        },
        "thread3_cliff_jump": float(cliff_jump),
        "thread3_cliff_sharpness": float(cliff_sharpness),
    }


def _synthetic_geo_disease_eas_dataset(n=6000, seed=20260301, n_pcs=16):
    n = int(max(5, n))
    n_pcs = int(max(3, n_pcs))
    rng = np.random.default_rng(int(seed))

    # Hidden superpopulation assignment.
    eas = rng.random(n) < 0.23
    n_eas = int(np.sum(eas))

    # Hidden geographic coordinates.
    lat = rng.uniform(-55.0, 70.0, size=n)
    lon = rng.uniform(-175.0, 175.0, size=n)

    # East Asia geographic region for the EAS superpopulation.
    lat[eas] = rng.uniform(15.0, 52.0, size=n_eas)
    lon[eas] = rng.uniform(95.0, 145.0, size=n_eas)

    eta = np.full(n, math.log(0.02 / 0.98), dtype=float)
    eta[eas] = math.log(0.10 / 0.90)

    # Massive longitude/latitude effects are EAS-only by construction.
    lat_e = (lat[eas] - 33.5) / 11.0
    lon_e = (lon[eas] - 120.0) / 10.0
    eas_geo_signal = (
        3.25 * np.sin(1.35 * lat_e)
        - 2.85 * np.cos(1.55 * lon_e)
        + 2.50 * np.sin(1.10 * lat_e * lon_e)
        + 1.90 * np.cos(1.60 * lat_e + 0.45 * lon_e)
    )
    eas_geo_signal = eas_geo_signal - float(np.mean(eas_geo_signal))
    eta[eas] = eta[eas] + eas_geo_signal + rng.normal(0.0, 0.20, size=n_eas)
    eta[~eas] = eta[~eas] + rng.normal(0.0, 0.08, size=int(np.sum(~eas)))

    pr = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.random(n) < pr).astype(float)

    lat_s = lat / 90.0
    lon_s = lon / 180.0

    rows = []
    for i in range(n):
        row = {}
        for j in range(n_pcs):
            a = 0.98 - 0.05 * j
            b = 0.26 + 0.03 * j
            c = ((-1.0) ** j) * (0.12 + 0.01 * j)
            d = 0.22 if j >= 8 else 0.06
            noise_sd = 0.13 + 0.018 * j
            pc = (
                a * lat_s[i]
                + b * lon_s[i]
                + c * lat_s[i] * lon_s[i]
                + d * float(eas[i])
                + rng.normal(0.0, noise_sd)
            )
            row[f"pc{j + 1}"] = float(pc)
        row["y"] = float(y[i])
        rows.append(row)

    return {
        "family": "binomial",
        "rows": rows,
        "features": [f"pc{i}" for i in range(1, n_pcs + 1)],
        "target": "y",
    }


def _geo_disease_eas_scenario_cfg(name):
    m = re.match(
        r"^geo_disease_(eas|eas3)_(tp|duchon|matern|psperpc)_k([0-9]+)(?:_downsample[0-9]+x)?(?:_holdout)?$",
        str(name),
    )
    if m is None:
        return None
    family_code = m.group(1)
    basis_code = m.group(2)
    knots = max(4, int(m.group(3)))
    n_pcs = 3 if family_code == "eas3" else 16
    smooth_pcs = 3 if n_pcs == 3 else 3
    linear_start = smooth_pcs + 1
    if basis_code == "tp":
        return {
            "smooth_basis": "thinplate",
            "smooth_cols": [f"pc{i}" for i in range(1, smooth_pcs + 1)],
            "linear_cols": [f"pc{i}" for i in range(linear_start, n_pcs + 1)],
            "knots": knots,
            "basis_code": basis_code,
            "n_pcs": n_pcs,
        }
    if basis_code == "duchon":
        return {
            "smooth_basis": "duchon",
            "smooth_cols": [f"pc{i}" for i in range(1, smooth_pcs + 1)],
            "linear_cols": [f"pc{i}" for i in range(linear_start, n_pcs + 1)],
            "knots": knots,
            "basis_code": basis_code,
            "n_pcs": n_pcs,
        }
    if basis_code == "matern":
        return {
            "smooth_basis": "matern",
            "smooth_cols": [f"pc{i}" for i in range(1, smooth_pcs + 1)],
            "linear_cols": [f"pc{i}" for i in range(linear_start, n_pcs + 1)],
            "knots": knots,
            "basis_code": basis_code,
            "n_pcs": n_pcs,
        }
    return {
        "smooth_basis": "ps",
        "smooth_cols": [f"pc{i}" for i in range(1, n_pcs + 1)],
        "linear_cols": [],
        "knots": knots,
        "basis_code": basis_code,
        "n_pcs": n_pcs,
    }


def _papuan_oce_scenario_cfg(name):
    m = re.match(r"^papuan_oce(4)?_(tp|duchon|matern|psperpc)_k([0-9]+)$", str(name))
    if m is None:
        return None
    is_four_pc = m.group(1) is not None
    basis_code = m.group(2)
    knots = max(4, int(m.group(3)))
    n_pcs = 4 if is_four_pc else 16
    if basis_code == "psperpc":
        return {
            "smooth_basis": "ps",
            "smooth_cols": [f"pc{i}" for i in range(1, n_pcs + 1)],
            "linear_cols": [],
            "knots": knots,
            "basis_code": basis_code,
            "n_pcs": n_pcs,
        }
    smooth_basis = {"tp": "thinplate", "duchon": "duchon", "matern": "matern"}[basis_code]
    return {
        "smooth_basis": smooth_basis,
        "smooth_cols": [f"pc{i}" for i in range(1, min(3, n_pcs) + 1)],
        "linear_cols": [f"pc{i}" for i in range(min(3, n_pcs) + 1, n_pcs + 1)],
        "knots": knots,
        "basis_code": basis_code,
        "n_pcs": n_pcs,
    }


def _compute_pcs_from_genotypes(g, n_pcs=16):
    # Proper PCA from standardized genotype-like matrix using SVD.
    n = g.shape[0]
    x = g.astype(float)
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, ddof=1, keepdims=True)
    sd[~np.isfinite(sd) | (sd < 1e-8)] = 1.0
    x = (x - mu) / sd
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    k = min(int(n_pcs), u.shape[1])
    pcs = u[:, :k] * s[:k]
    pcs = pcs / max(math.sqrt(max(n - 1, 1)), 1.0)
    pcs_mu = pcs.mean(axis=0, keepdims=True)
    pcs_sd = pcs.std(axis=0, ddof=1, keepdims=True)
    pcs_sd[~np.isfinite(pcs_sd) | (pcs_sd < 1e-8)] = 1.0
    return (pcs - pcs_mu) / pcs_sd


def _synthetic_papuan_oce_dataset(n=6000, seed=20260315, n_pcs=16):
    n = int(max(1200, n))
    rng = np.random.default_rng(int(seed))

    # Hidden group assignment only; labels are never emitted as model features.
    groups = [
        ("Papuan", 0.12),
        ("Papuan Sepik", 0.08),
        ("Papuan,Papuan Highlands", 0.06),
        ("Papuan,Papuan Sepik", 0.06),
        ("Bougainville", 0.08),
        ("Han", 0.17),
        ("Japanese", 0.10),
        ("CEU", 0.14),
        ("Yoruba", 0.11),
        ("Pima", 0.08),
    ]
    names = [g[0] for g in groups]
    probs = np.array([g[1] for g in groups], dtype=float)
    probs = probs / probs.sum()
    pop_idx = rng.choice(len(names), size=n, p=probs)
    pop = np.array([names[i] for i in pop_idx], dtype=object)

    # Latent ancestry coordinates and genotype-like matrix.
    centers = {
        "Papuan": np.array([2.7, -0.8, 1.7, 0.7]),
        "Papuan Sepik": np.array([3.0, -0.6, 1.5, 0.8]),
        "Papuan,Papuan Highlands": np.array([2.9, -0.3, 1.9, 0.4]),
        "Papuan,Papuan Sepik": np.array([3.1, -0.5, 1.6, 0.8]),
        "Bougainville": np.array([2.5, -1.0, 1.8, 0.5]),
        "Han": np.array([0.4, 2.0, -0.1, 0.2]),
        "Japanese": np.array([0.7, 1.8, -0.2, 0.1]),
        "CEU": np.array([-2.1, 0.4, 0.1, -0.2]),
        "Yoruba": np.array([-1.9, -2.2, 0.5, 0.0]),
        "Pima": np.array([-0.3, -0.9, -1.6, 0.4]),
    }

    z = np.zeros((n, 4), dtype=float)
    for i in range(n):
        z[i] = centers[str(pop[i])] + rng.normal(0.0, 0.55, size=4)

    n_snps = 700
    loadings = rng.normal(0.0, 0.28, size=(4, n_snps))
    logits = -0.2 + z @ loadings + rng.normal(0.0, 0.35, size=(n, n_snps))
    p = 1.0 / (1.0 + np.exp(-np.clip(logits, -10.0, 10.0)))
    g = rng.binomial(2, p).astype(float)

    pcs = _compute_pcs_from_genotypes(g, n_pcs=max(3, int(n_pcs)))
    if pcs.shape[1] < n_pcs:
        pcs = np.concatenate([pcs, np.zeros((n, n_pcs - pcs.shape[1]), dtype=float)], axis=1)

    # Prevalence: 0.4 in Papuan* and Bougainville; 0.02 in all others.
    high_risk = np.array([("Papuan" in str(lbl)) or (str(lbl) == "Bougainville") for lbl in pop], dtype=bool)
    prevalence = np.where(high_risk, 0.40, 0.02)
    y = (rng.random(n) < prevalence).astype(float)

    rows = []
    for i in range(n):
        row = {f"pc{j + 1}": float(pcs[i, j]) for j in range(int(n_pcs))}
        row["y"] = float(y[i])
        rows.append(row)

    return {
        "family": "binomial",
        "rows": rows,
        "features": [f"pc{i}" for i in range(1, int(n_pcs) + 1)],
        "target": "y",
    }


def _pairwise_dist(x):
    s = np.sum(x * x, axis=1, keepdims=True)
    d2 = s + s.T - 2.0 * (x @ x.T)
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)


def _path_length(path, d):
    if len(path) <= 1:
        return 0.0
    return float(np.sum(d[path[:-1], path[1:]]))


def _nearest_neighbor_path(d, start):
    n = d.shape[0]
    unused = set(range(n))
    unused.remove(start)
    path = [start]
    while unused:
        cur = path[-1]
        nxt = min(unused, key=lambda j: d[cur, j])
        path.append(nxt)
        unused.remove(nxt)
    return np.array(path, dtype=int)


def _two_opt_open(path, d, max_passes=20):
    n = len(path)
    if n < 4:
        return path.copy()
    p = path.copy()
    for _ in range(max_passes):
        improved = False
        for i in range(n - 3):
            a, b = p[i], p[i + 1]
            for j in range(i + 2, n - 1):
                c, e = p[j], p[j + 1]
                before = d[a, b] + d[c, e]
                after = d[a, c] + d[b, e]
                if after + 1e-12 < before:
                    p[i + 1 : j + 1] = p[i + 1 : j + 1][::-1]
                    improved = True
        if not improved:
            break
    return p


def _best_1d_order(d):
    n = d.shape[0]
    best = None
    best_len = np.inf
    for start in range(n):
        p0 = _nearest_neighbor_path(d, start)
        p1 = _two_opt_open(p0, d)
        path_len = _path_length(p1, d)
        if path_len < best_len:
            best_len = path_len
            best = p1
    if best is None:
        raise RuntimeError("failed to derive subpopulation 1D ordering")
    return best


def _geo_subpop16_scenario_cfg(name):
    m = re.match(r"^geo_subpop16_(tp|duchon|matern|psperpc)_k([0-9]+)$", str(name))
    if m is None:
        return None
    basis_code = m.group(1)
    knots = max(4, int(m.group(2)))
    if basis_code == "psperpc":
        return {
            "smooth_basis": "ps",
            "smooth_cols": [f"pc{i}" for i in range(1, 17)],
            "linear_cols": [],
            "knots": knots,
            "basis_code": basis_code,
            "n_pcs": 16,
        }
    smooth_basis = {"tp": "thinplate", "duchon": "duchon", "matern": "matern"}[basis_code]
    return {
        "smooth_basis": smooth_basis,
        "smooth_cols": [f"pc{i}" for i in range(1, 4)],
        "linear_cols": [f"pc{i}" for i in range(4, 17)],
        "knots": knots,
        "basis_code": basis_code,
        "n_pcs": 16,
    }


def _geo_latlon_scenario_cfg(name):
    m = re.match(r"^geo_latlon_(superpopnoise|equatornoise)_(tp|duchon|matern|psperpc)_k([0-9]+)$", str(name))
    if m is None:
        return None
    mode_code = m.group(1)
    basis_code = m.group(2)
    knots = max(4, int(m.group(3)))
    if basis_code == "psperpc":
        return {
            "mode_code": mode_code,
            "smooth_basis": "ps",
            "smooth_cols": [f"pc{i}" for i in range(1, 7)],
            "linear_cols": [],
            "knots": knots,
            "basis_code": basis_code,
            "n_pcs": 6,
        }
    smooth_basis = {"tp": "thinplate", "duchon": "duchon", "matern": "matern"}[basis_code]
    return {
        "mode_code": mode_code,
        "smooth_basis": smooth_basis,
        "smooth_cols": [f"pc{i}" for i in range(1, 4)],
        "linear_cols": [f"pc{i}" for i in range(4, 7)],
        "knots": knots,
        "basis_code": basis_code,
        "n_pcs": 6,
    }


def _load_hgdp_pc_with_imputed_latlon():
    if not HGDP_1KG_PC_TSV.exists():
        raise RuntimeError(f"missing required PC dataset: {HGDP_1KG_PC_TSV}")
    raw = pd.read_csv(HGDP_1KG_PC_TSV, sep="\t")
    pc_cols = [f"PC{i}" for i in range(1, 17)]
    required = {"sample_id", "Superpopulation", "Subpopulation", "Latitude", "Longitude", *pc_cols}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise RuntimeError(f"hgdp_1kg_pc_data.tsv is missing required columns: {missing}")

    d = raw[["sample_id", "Superpopulation", "Subpopulation", "Latitude", "Longitude", *pc_cols]].copy()
    d["Subpopulation"] = d["Subpopulation"].astype(str)
    d["Superpopulation"] = d["Superpopulation"].astype(str)
    for c in pc_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=pc_cols).copy()
    if d.empty:
        raise RuntimeError("hgdp_1kg_pc_data.tsv has no complete rows for PC1..PC16")

    centroids = d.groupby("Subpopulation", dropna=False)[pc_cols].mean(numeric_only=True)
    sub_latlon_known = (
        d.dropna(subset=["Latitude", "Longitude"])
        .groupby("Subpopulation", dropna=False)[["Latitude", "Longitude"]]
        .mean(numeric_only=True)
    )

    known_subpops = [sp for sp in centroids.index if sp in sub_latlon_known.index]
    if len(known_subpops) < 2:
        raise RuntimeError(
            "hgdp_1kg_pc_data.tsv must contain at least two subpopulations with real "
            "Latitude/Longitude values"
        )

    known_x = centroids.loc[known_subpops].to_numpy(dtype=float)
    known_latlon = sub_latlon_known.loc[known_subpops][["Latitude", "Longitude"]].to_numpy(dtype=float)
    full_subpops = centroids.index.tolist()

    subpop_to_latlon = {}
    for sp in full_subpops:
        if sp in sub_latlon_known.index:
            ll = sub_latlon_known.loc[sp]
            subpop_to_latlon[sp] = (float(ll["Latitude"]), float(ll["Longitude"]))
            continue
        x0 = centroids.loc[sp].to_numpy(dtype=float)
        dist = np.sqrt(np.maximum(np.sum((known_x - x0) ** 2, axis=1), 0.0))
        nn = np.argsort(dist)[:2]
        imputed = np.mean(known_latlon[nn], axis=0)
        subpop_to_latlon[sp] = (float(imputed[0]), float(imputed[1]))

    sample_lat = []
    sample_lon = []
    for _, row in d.iterrows():
        lat_raw = row["Latitude"]
        lon_raw = row["Longitude"]
        if np.isfinite(lat_raw) and np.isfinite(lon_raw):
            sample_lat.append(float(lat_raw))
            sample_lon.append(float(lon_raw))
        else:
            lat_i, lon_i = subpop_to_latlon[str(row["Subpopulation"])]
            sample_lat.append(lat_i)
            sample_lon.append(lon_i)
    d["lat_imputed"] = np.asarray(sample_lat, dtype=float)
    d["lon_imputed"] = np.asarray(sample_lon, dtype=float)

    return d


def _geo_latlon_dataset(mode_code, seed=20260401, prevalence_min=0.01, prevalence_max=0.10):
    mode_code = str(mode_code)
    if mode_code not in {"superpopnoise", "equatornoise"}:
        raise RuntimeError(f"unsupported geo_latlon mode: {mode_code}")
    rng = np.random.default_rng(int(seed))
    d = _load_hgdp_pc_with_imputed_latlon().copy()

    lat_norm = np.clip(np.abs(d["lat_imputed"].to_numpy(dtype=float)) / 90.0, 0.0, 1.0)
    lon_norm = np.clip((d["lon_imputed"].to_numpy(dtype=float) + 180.0) / 360.0, 0.0, 1.0)
    westness = 1.0 - lon_norm

    if mode_code == "superpopnoise":
        risk_latlon = 0.68 * lat_norm + 0.32 * westness
        base_prev = float(prevalence_min) + (float(prevalence_max) - float(prevalence_min)) * np.clip(risk_latlon, 0.0, 1.0)
        superpops = sorted(d["Superpopulation"].astype(str).unique().tolist())
        superpop_noise = {sp: float(rng.uniform(0.10, 0.90)) for sp in superpops}
        noise_sd = d["Superpopulation"].astype(str).map(superpop_noise).to_numpy(dtype=float)
    else:
        edge_risk = np.clip(np.abs(d["lon_imputed"].to_numpy(dtype=float)) / 180.0, 0.0, 1.0)
        base_prev = float(prevalence_min) + (float(prevalence_max) - float(prevalence_min)) * edge_risk
        equator_close = 1.0 - lat_norm
        noise_sd = 0.05 + 1.25 * np.clip(equator_close, 0.0, 1.0)

    base_prev = np.clip(base_prev, 1e-5, 1.0 - 1e-5)
    eta = np.log(base_prev / (1.0 - base_prev)) + rng.normal(0.0, noise_sd, size=len(d))
    p = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.random(len(d)) < p).astype(float)

    rows = []
    for i, row in d.iterrows():
        out = {f"pc{j}": float(row[f"PC{j}"]) for j in range(1, 7)}
        out["y"] = float(y[i])
        rows.append(out)
    return {
        "family": "binomial",
        "rows": rows,
        "features": [f"pc{i}" for i in range(1, 7)],
        "target": "y",
    }


def _geo_subpop16_dataset(seed=20260330, prevalence_min=0.02, prevalence_max=0.40):
    if not HGDP_1KG_PC_TSV.exists():
        raise RuntimeError(f"missing required PC dataset: {HGDP_1KG_PC_TSV}")

    rng = np.random.default_rng(int(seed))
    raw = pd.read_csv(HGDP_1KG_PC_TSV, sep="\t")
    pc_cols = [f"PC{i}" for i in range(1, 17)]
    required = {"Subpopulation", *pc_cols}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise RuntimeError(f"hgdp_1kg_pc_data.tsv is missing required columns: {missing}")

    d = raw[["Subpopulation", *pc_cols]].dropna().copy()
    d["Subpopulation"] = d["Subpopulation"].astype(str)
    if d.empty:
        raise RuntimeError("hgdp_1kg_pc_data.tsv has no complete rows for Subpopulation + PC1..PC16")

    subpops = sorted(d["Subpopulation"].unique().tolist())
    if len(subpops) < 2:
        raise RuntimeError("need at least two subpopulations for prevalence simulation")

    # No broad trend: each subpopulation gets an independent mean prevalence.
    prevalence_map = {
        sp: float(rng.uniform(float(prevalence_min), float(prevalence_max))) for sp in subpops
    }
    base_prev = d["Subpopulation"].map(prevalence_map).to_numpy(dtype=float)
    base_prev = np.clip(base_prev, 1e-5, 1.0 - 1e-5)
    base_eta = np.log(base_prev / (1.0 - base_prev))

    # One noise model per subpopulation; each row inherits its subpopulation's model.
    noise_types = ["gaussian", "laplace", "student_t"]
    subpop_noise_kind = {sp: str(rng.choice(noise_types)) for sp in subpops}
    subpop_noise_scale = {sp: float(rng.uniform(0.25, 0.85)) for sp in subpops}
    noise_kind = d["Subpopulation"].map(subpop_noise_kind).astype(str).to_numpy()
    noise_scale = d["Subpopulation"].map(subpop_noise_scale).to_numpy(dtype=float)

    noise = np.zeros(len(d), dtype=float)
    for kind in noise_types:
        idx = np.where(noise_kind == kind)[0]
        if idx.size == 0:
            continue
        s = noise_scale[idx]
        if kind == "gaussian":
            noise[idx] = rng.normal(0.0, s)
        elif kind == "laplace":
            noise[idx] = rng.laplace(0.0, s)
        else:
            noise[idx] = rng.standard_t(df=4, size=idx.size) * s

    eta = base_eta + noise
    p = 1.0 / (1.0 + np.exp(-eta))
    d["y"] = (rng.random(len(d)) < p).astype(float)

    rows = []
    for _, row in d.iterrows():
        out = {f"pc{i}": float(row[f"PC{i}"]) for i in range(1, 17)}
        out["y"] = float(row["y"])
        rows.append(out)

    return {
        "family": "binomial",
        "rows": rows,
        "features": [f"pc{i}" for i in range(1, 17)],
        "target": "y",
    }


def _validate_dataset_schema(ds, scenario_name):
    if not isinstance(ds, dict):
        raise RuntimeError(f"{scenario_name}: dataset loader must return a dict")
    if "family" not in ds:
        raise RuntimeError(f"{scenario_name}: dataset missing 'family'")
    if "rows" not in ds:
        raise RuntimeError(f"{scenario_name}: dataset missing 'rows'")
    if "features" not in ds:
        raise RuntimeError(f"{scenario_name}: dataset missing 'features'")

    rows = ds["rows"]
    features = ds["features"]
    family = ds["family"]
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{scenario_name}: dataset rows must be a non-empty list")
    if not isinstance(features, list):
        raise RuntimeError(f"{scenario_name}: dataset features must be a list")
    if not all(isinstance(c, str) and c for c in features):
        raise RuntimeError(f"{scenario_name}: dataset features must be non-empty strings")

    if family == "survival":
        for key in ("time_col", "event_col"):
            if key not in ds or not isinstance(ds[key], str) or not ds[key]:
                raise RuntimeError(f"{scenario_name}: survival dataset missing valid '{key}'")
        required_cols = set(features) | {ds["time_col"], ds["event_col"]}
    else:
        target = ds.get("target")
        if not isinstance(target, str) or not target:
            raise RuntimeError(f"{scenario_name}: non-survival dataset missing valid 'target'")
        required_cols = set(features) | {target}

    first = rows[0]
    if not isinstance(first, dict):
        raise RuntimeError(f"{scenario_name}: dataset rows must be dictionaries")
    missing = [c for c in required_cols if c not in first]
    if missing:
        raise RuntimeError(f"{scenario_name}: first row missing required columns: {sorted(missing)}")

    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"{scenario_name}: row {i} is not a dictionary")
        row_missing = [c for c in required_cols if c not in row]
        if row_missing:
            raise RuntimeError(f"{scenario_name}: row {i} missing required columns: {sorted(row_missing)}")

    if family == "survival":
        numeric_cols = features + [ds["time_col"], ds["event_col"]]
    else:
        numeric_cols = features + [ds["target"]]
    for col in numeric_cols:
        try:
            np.array([float(r[col]) for r in rows], dtype=float)
        except Exception as e:
            raise RuntimeError(f"{scenario_name}: column '{col}' is not fully numeric/coercible: {e}") from e


def _dataset_for_scenario_unvalidated(s):
    name = s["name"]
    if name in {"small_dense", "medium", "pathological_ill_conditioned"}:
        cfg = _CANONICAL_SYNTHETIC_BINOMIAL_SCENARIOS[name]
        return _synthetic_binomial_dataset(
            s.get("n", cfg["n"]),
            s.get("p", cfg["p"]),
            s.get("seed", cfg["seed"]),
        )
    if name.startswith("geo_disease_eas3_"):
        return _synthetic_geo_disease_eas_dataset(s.get("n", 6000), s.get("seed", 20260301), n_pcs=3)
    if name.startswith("geo_disease_eas_"):
        return _synthetic_geo_disease_eas_dataset(s.get("n", 6000), s.get("seed", 20260301))
    if name.startswith("papuan_oce4_"):
        return _synthetic_papuan_oce_dataset(s.get("n", 6000), s.get("seed", 20260315), n_pcs=4)
    if name.startswith("papuan_oce_"):
        return _synthetic_papuan_oce_dataset(s.get("n", 6000), s.get("seed", 20260315), n_pcs=16)
    if name.startswith("geo_subpop16_"):
        return _geo_subpop16_dataset(
            seed=s.get("seed", 20260330),
            prevalence_min=s.get("prevalence_min", 0.02),
            prevalence_max=s.get("prevalence_max", 0.40),
        )
    geo_latlon_cfg = _geo_latlon_scenario_cfg(name)
    if geo_latlon_cfg is not None:
        return _geo_latlon_dataset(
            mode_code=geo_latlon_cfg["mode_code"],
            seed=s.get("seed", 20260401),
            prevalence_min=s.get("prevalence_min", 0.01),
            prevalence_max=s.get("prevalence_max", 0.10),
        )
    if name.startswith("geo_disease_"):
        return _synthetic_geo_disease_dataset(s.get("n", 4000), s.get("seed", 20260226))
    if name == "thread2_fractional_spde_nu18":
        return _synthetic_thread2_continuous_order_dataset(
            mode="fractional",
            n=s.get("n", 512),
            seed=s.get("seed", 20260501),
            true_nu=s.get("true_nu", 1.8),
            true_kappa2=s.get("true_kappa2", 0.7),
        )
    if name == "thread2_boundary_rough":
        return _synthetic_thread2_continuous_order_dataset(
            mode="rough",
            n=s.get("n", 512),
            seed=s.get("seed", 20260502),
        )
    if name == "thread2_boundary_smooth":
        return _synthetic_thread2_continuous_order_dataset(
            mode="smooth",
            n=s.get("n", 512),
            seed=s.get("seed", 20260503),
        )
    if name == "thread3_admixture_cliff":
        return _synthetic_thread3_admixture_cliff_dataset(
            n=s.get("n", 6000),
            seed=s.get("seed", 20260601),
        )
    if name == "lidar_semipar":
        return _load_lidar_dataset()
    if name == "bone_gamair":
        return _load_bone_dataset()
    if name == "prostate_gamair":
        return _load_prostate_dataset()
    if name == "wine_gamair":
        return _load_wine_dataset()
    if name == "wine_temp_vs_year":
        return _load_wine_temp_vs_year_dataset()
    if name == "wine_price_vs_temp":
        return _load_wine_price_vs_temp_dataset()
    if name == "horse_colic":
        return _load_horse_dataset()
    if name == "us48_demand_5day":
        return _load_us48_demand_dataset("five_day.csv")
    if name == "us48_demand_31day":
        return _load_us48_demand_dataset("31_day.csv")
    if name == "haberman_survival":
        return _load_haberman_dataset()
    if name == "icu_survival_death":
        return _load_icu_survival_death_dataset()
    if name == "icu_survival_los":
        return _load_icu_survival_los_dataset()
    if name == "heart_failure_survival":
        return _load_heart_failure_survival_dataset()
    if name == "cirrhosis_survival":
        return _load_cirrhosis_survival_dataset()
    raise RuntimeError(f"No scenario-specific dataset loader configured for '{name}'")


def dataset_for_scenario(s):
    ds = _dataset_for_scenario_unvalidated(s)
    if "cv_splits" in s:
        ds["_cv_splits"] = int(s["cv_splits"])
    _validate_dataset_schema(ds, scenario_name=s["name"])
    return ds


def folds_for_dataset(ds):
    n_splits = int(ds.get("_cv_splits", CV_SPLITS))
    if ds["family"] == "survival":
        _coerce_positive_survival_dataset_inplace(ds, dataset_name=ds.get("name", "survival_dataset"))
        y = np.array([float(r[ds["event_col"]]) for r in ds["rows"]], dtype=float)
        stratified = True
    else:
        y = np.array([float(r[ds["target"]]) for r in ds["rows"]], dtype=float)
        stratified = ds["family"] == "binomial"
    if ds["family"] == "binomial":
        y_bin = (y > 0.5).astype(int)
        n_pos = int(np.sum(y_bin == 1))
        n_neg = int(np.sum(y_bin == 0))
        min_class = min(n_pos, n_neg)
        min_required = 2 if n_splits == 1 else n_splits
        if min_class < min_required:
            raise RuntimeError(
                "invalid binomial CV configuration: "
                f"requested {n_splits} "
                f"{'holdout split' if n_splits == 1 else 'folds'} but class counts are "
                f"positives={n_pos}, negatives={n_neg}. "
                + (
                    "Each class must have at least two observations for a stratified holdout split. "
                    if n_splits == 1
                    else "Each class must have at least one observation per fold. "
                )
                + "Increase n or reduce the number of folds."
            )
    folds = make_folds(y, n_splits=n_splits, seed=CV_SEED, stratified=stratified)
    n = len(ds["rows"])
    seen = np.zeros(n, dtype=int)
    for f in folds:
        tr = np.asarray(f.train_idx, dtype=int)
        te = np.asarray(f.test_idx, dtype=int)
        if tr.size == 0 or te.size == 0:
            raise RuntimeError("invalid CV split: empty train or test fold")
        if np.intersect1d(tr, te).size != 0:
            raise RuntimeError("invalid CV split: train/test overlap detected")
        if np.any(tr < 0) or np.any(te < 0) or np.any(tr >= n) or np.any(te >= n):
            raise RuntimeError("invalid CV split: index out of bounds")
        seen[te] += 1
    if len(folds) == 1:
        if np.any(seen > 1):
            raise RuntimeError("invalid holdout split: a row appears more than once in the test set")
    elif not np.all(seen == 1):
        raise RuntimeError("invalid CV split: each row must appear exactly once in test sets")
    return folds


def build_shared_fold_artifacts(
    ds: dict,
    folds: list[Fold],
    root_dir: Path,
) -> list[SharedFoldArtifact]:
    root_dir.mkdir(parents=True, exist_ok=True)
    base_df = pd.DataFrame(ds["rows"])
    feature_cols = list(ds.get("features", []))
    artifacts: list[SharedFoldArtifact] = []
    for fold_id, fold in enumerate(folds):
        train_idx_path = root_dir / f"train_idx_{fold_id}.txt"
        test_idx_path = root_dir / f"test_idx_{fold_id}.txt"
        train_idx_path.write_text("\n".join(str(int(i)) for i in fold.train_idx) + "\n")
        test_idx_path.write_text("\n".join(str(int(i)) for i in fold.test_idx) + "\n")

        train_df = base_df.iloc[fold.train_idx].copy()
        test_df = base_df.iloc[fold.test_idx].copy()
        if feature_cols:
            train_df, test_df = zscore_train_test(train_df, test_df, feature_cols)
        train_csv = root_dir / f"train_scaled_{fold_id}.csv"
        test_csv = root_dir / f"test_scaled_{fold_id}.csv"
        train_df.to_csv(train_csv, index=False)
        test_df.to_csv(test_csv, index=False)
        artifacts.append(
            SharedFoldArtifact(
                fold_id=fold_id,
                train_scaled_csv=train_csv,
                test_scaled_csv=test_csv,
                train_idx_path=train_idx_path,
                test_idx_path=test_idx_path,
            )
        )
    return artifacts


def aggregate_cv_rows(cv_rows, family):
    fit_sec = float(sum(float(r["fit_sec"]) for r in cv_rows))
    predict_sec = float(sum(float(r["predict_sec"]) for r in cv_rows))

    def wavg(key):
        if key == "mse":
            valid_rmse = [(r.get("rmse"), int(r["n_test"])) for r in cv_rows if r.get("rmse") is not None]
            if valid_rmse:
                denom = max(sum(w for _, w in valid_rmse), 1)
                return float(sum((float(v) ** 2) * w for v, w in valid_rmse) / denom)
        valid = [(r.get(key), int(r["n_test"])) for r in cv_rows if r.get(key) is not None]
        if not valid:
            return None
        denom = max(sum(w for _, w in valid), 1)
        return float(sum(float(v) * w for v, w in valid) / denom)

    if family == "binomial":
        return {
            "fit_sec": fit_sec,
            "predict_sec": predict_sec,
            "auc": wavg("auc"),
            "brier": wavg("brier"),
            "logloss": wavg("logloss"),
            "nagelkerke_r2": wavg("nagelkerke_r2"),
            "mse": None,
            "rmse": None,
            "mae": None,
            "r2": None,
        }
    if family == "survival":
        return {
            "fit_sec": fit_sec,
            "predict_sec": predict_sec,
            "auc": wavg("auc"),  # concordance index
            "brier": wavg("brier"),
            "logloss": wavg("logloss"),
            "c_index": wavg("auc"),
            "nagelkerke_r2": wavg("nagelkerke_r2"),
            "mse": None,
            "rmse": None,
            "mae": None,
            "r2": None,
        }
    return {
        "fit_sec": fit_sec,
        "predict_sec": predict_sec,
        "auc": None,
        "brier": None,
        "logloss": wavg("logloss"),
        "nagelkerke_r2": None,
        "mse": wavg("mse"),
        "rmse": wavg("rmse"),
        "mae": wavg("mae"),
        "r2": wavg("r2"),
    }


def _rust_fit_mapping(scenario_name):
    geo_eas_cfg = _geo_disease_eas_scenario_cfg(scenario_name)
    papuan_cfg = _papuan_oce_scenario_cfg(scenario_name)
    subpop_cfg = _geo_subpop16_scenario_cfg(scenario_name)
    latlon_cfg = _geo_latlon_scenario_cfg(scenario_name)
    if geo_eas_cfg is not None:
        return {
            "family": "binomial-logit",
            "smooth_cols": geo_eas_cfg["smooth_cols"],
            "smooth_basis": geo_eas_cfg["smooth_basis"],
            "linear_cols": geo_eas_cfg["linear_cols"],
            "knots": int(geo_eas_cfg["knots"]),
        }
    if papuan_cfg is not None:
        return {
            "family": "binomial-logit",
            "smooth_cols": papuan_cfg["smooth_cols"],
            "smooth_basis": papuan_cfg["smooth_basis"],
            "linear_cols": papuan_cfg["linear_cols"],
            "knots": int(papuan_cfg["knots"]),
        }
    if subpop_cfg is not None:
        return {
            "family": "binomial-logit",
            "smooth_cols": subpop_cfg["smooth_cols"],
            "smooth_basis": subpop_cfg["smooth_basis"],
            "linear_cols": subpop_cfg["linear_cols"],
            "knots": int(subpop_cfg["knots"]),
        }
    if latlon_cfg is not None:
        return {
            "family": "binomial-logit",
            "smooth_cols": latlon_cfg["smooth_cols"],
            "smooth_basis": latlon_cfg["smooth_basis"],
            "linear_cols": latlon_cfg["linear_cols"],
            "knots": int(latlon_cfg["knots"]),
        }
    return {
        "small_dense": dict(
            family="binomial-logit",
            smooth_col="x2",
            linear_cols=["x1"],
            smooth_basis="ps",
            knots=7,
            double_penalty=False,
        ),
        "medium": dict(family="binomial-logit", smooth_col="x2", linear_cols=["x1"], smooth_basis="ps", knots=7),
        "pathological_ill_conditioned": dict(
            family="binomial-logit", smooth_col="x2", linear_cols=["x1"], smooth_basis="ps", knots=7
        ),
        "lidar_semipar": dict(family="gaussian", smooth_col="range", linear_cols=[], smooth_basis="ps", knots=24),
        "bone_gamair": dict(family="binomial-logit", smooth_col="t", linear_cols=["trt_auto"], smooth_basis="ps", knots=8),
        "prostate_gamair": dict(
            family="binomial-logit",
            smooth_col="pc2",
            linear_cols=["pc1"],
            smooth_basis="ps",
            knots=8,
            double_penalty=False,
        ),
        "horse_colic": dict(
            family="binomial-logit",
            smooth_col="pulse",
            linear_cols=["rectal_temp", "packed_cell_volume"],
            smooth_basis="ps",
            knots=8,
            double_penalty=False,
        ),
        "wine_gamair": dict(
            family="gaussian",
            smooth_col="s_temp",
            linear_cols=["year", "h_rain", "w_rain", "h_temp"],
            smooth_basis="ps",
            knots=7,
            double_penalty=True,
        ),
        "wine_temp_vs_year": dict(
            family="gaussian", smooth_col="year", linear_cols=[], smooth_basis="ps", knots=7
        ),
        "wine_price_vs_temp": dict(
            family="gaussian", smooth_col="temp", linear_cols=[], smooth_basis="ps", knots=7
        ),
        "us48_demand_5day": dict(
            family="gaussian",
            smooth_col="hour",
            linear_cols=["demand_forecast", "net_generation", "total_interchange"],
            smooth_basis="ps",
            knots=8,
        ),
        "us48_demand_31day": dict(
            family="gaussian",
            smooth_col="hour",
            linear_cols=["demand_forecast", "net_generation", "total_interchange"],
            smooth_basis="ps",
            knots=12,
        ),
        "haberman_survival": dict(
            family="binomial-logit", smooth_col="axil_nodes", linear_cols=["age", "op_year"], smooth_basis="ps", knots=8
        ),
        "icu_survival_death": dict(
            family="binomial-logit",
            smooth_col="time",
            linear_cols=["age", "bmi", "hr_max", "sysbp_min"],
            smooth_basis="ps",
            knots=7,
        ),
        "icu_survival_los": dict(
            family="binomial-logit",
            smooth_col="age",
            linear_cols=["bmi", "hr_max", "sysbp_min", "temp_apache", "time"],
            smooth_basis="ps",
            knots=7,
        ),
        "geo_disease_tp": dict(
            family="binomial-logit",
            smooth_cols=["pc1", "pc2", "pc3", "pc4"],
            smooth_basis="thinplate",
            linear_cols=[f"pc{i}" for i in range(5, 17)],
            knots=12,
        ),
        "geo_disease_duchon": dict(
            family="binomial-logit",
            smooth_cols=["pc1", "pc2", "pc3", "pc4"],
            smooth_basis="duchon",
            linear_cols=[f"pc{i}" for i in range(5, 17)],
            knots=12,
        ),
        "geo_disease_matern": dict(
            family="binomial-logit",
            smooth_cols=["pc1", "pc2", "pc3", "pc4"],
            smooth_basis="matern",
            linear_cols=[f"pc{i}" for i in range(5, 17)],
            knots=12,
        ),
        "geo_disease_shrinkage": dict(
            family="binomial-logit",
            smooth_cols=["pc1", "pc2", "pc3", "pc4"],
            smooth_basis="thinplate",
            linear_cols=[f"pc{i}" for i in range(5, 17)],
            knots=12,
        ),
        "geo_disease_ps_per_pc": dict(
            family="binomial-logit",
            smooth_cols=[f"pc{i}" for i in range(1, 17)],
            smooth_basis="ps",
            linear_cols=[],
            knots=10,
        ),
        "thread2_fractional_spde_nu18": dict(
            family="gaussian",
            smooth_col="x",
            linear_cols=[],
            smooth_basis="matern",
            knots=24,
            double_penalty=True,
        ),
        "thread2_boundary_rough": dict(
            family="gaussian",
            smooth_col="x",
            linear_cols=[],
            smooth_basis="matern",
            knots=24,
            double_penalty=True,
        ),
        "thread2_boundary_smooth": dict(
            family="gaussian",
            smooth_col="x",
            linear_cols=[],
            smooth_basis="matern",
            knots=24,
            double_penalty=True,
        ),
        "thread3_admixture_cliff": dict(
            family="binomial-logit",
            smooth_cols=["pc1", "pc2", "pc3", "pc4"],
            smooth_basis="matern",
            linear_cols=[f"pc{i}" for i in range(5, 17)],
            knots=16,
            double_penalty=True,
        ),
    }.get(scenario_name)


def _effective_rust_fit_mapping(scenario_name: str, override: dict | None = None):
    cfg = _rust_fit_mapping(scenario_name)
    if cfg is None:
        return None
    if not override:
        return dict(cfg)
    merged = dict(cfg)
    merged.update(override)
    return merged


def _canonical_smooth_basis(basis):
    b = str(basis or "ps").strip().lower()
    # Legacy alias used to mean one P-spline per feature; canonical basis is "ps".
    if b == "bspline_per_pc":
        return "ps"
    return b


def _rust_formula_for_scenario(scenario_name, ds, *, cfg_override: dict | None = None):
    cfg = _effective_rust_fit_mapping(scenario_name, cfg_override)
    if cfg is None:
        raise RuntimeError(f"No Rust formula mapping configured for scenario '{scenario_name}'")
    target = ds["target"]
    terms = [f"linear({c})" for c in cfg.get("linear_cols", [])]
    basis = _canonical_smooth_basis(cfg.get("smooth_basis", "ps"))
    knot_count = int(cfg.get("knots", 8))
    if knot_count < 0:
        raise RuntimeError(
            f"Invalid knot count {knot_count} for scenario '{scenario_name}'; expected >= 0."
        )
    # Keep shrinkage policy explicit and aligned with mgcv `select`.
    use_double_penalty = bool(cfg.get("double_penalty", True))
    dp_opt = f", double_penalty={'true' if use_double_penalty else 'false'}"
    smooth_cols = cfg.get("smooth_cols")
    if smooth_cols:
        for col in smooth_cols:
            if basis in {"ps", "bspline", "p-spline"}:
                terms.append(f"s({col}, type=ps, knots={knot_count}{dp_opt})")
            elif basis in {"thinplate", "tps"}:
                terms.append(f"s({col}, type=tps, centers={knot_count}{dp_opt})")
            elif basis == "duchon":
                terms.append(
                    f"s({col}, type=duchon, centers={knot_count}, order=0, power=1{dp_opt})"
                )
            elif basis == "matern":
                terms.append(f"s({col}, type=matern, centers={knot_count}{dp_opt})")
            else:
                raise RuntimeError(
                    f"Unsupported Rust smooth basis '{basis}' for scenario '{scenario_name}'"
                )
    else:
        col = cfg.get("smooth_col")
        if col:
            if basis in {"thinplate", "tps"}:
                terms.append(f"s({col}, type=tps, centers={knot_count}{dp_opt})")
            elif basis in {"ps", "bspline", "p-spline"} and "double_penalty" in cfg:
                dp = "true" if bool(cfg["double_penalty"]) else "false"
                terms.append(f"s({col}, type=ps, knots={knot_count}, double_penalty={dp})")
            elif basis in {"ps", "bspline", "p-spline"}:
                terms.append(f"s({col}, type=ps, knots={knot_count})")
            elif basis in {"duchon", "matern"}:
                if basis == "duchon":
                    terms.append(
                        f"s({col}, type=duchon, centers={knot_count}, order=0, power=1{dp_opt})"
                    )
                else:
                    terms.append(f"s({col}, type={basis}, centers={knot_count}{dp_opt})")
            else:
                raise RuntimeError(
                    f"Unsupported Rust smooth basis '{basis}' for scenario '{scenario_name}'"
                )

    if not terms:
        raise RuntimeError(f"empty Rust term list for scenario '{scenario_name}'")
    formula = f"{target} ~ " + " + ".join(terms)
    return cfg["family"], formula


def _mgcv_formula_for_scenario(scenario_name, ds):
    cfg = _rust_fit_mapping(scenario_name)
    if cfg is None:
        raise RuntimeError(f"No shared smooth mapping configured for scenario '{scenario_name}'")
    target = ds["target"]
    terms = [str(c) for c in cfg.get("linear_cols", [])]
    basis = _canonical_smooth_basis(cfg.get("smooth_basis", "ps"))
    knot_count = int(cfg.get("knots", 8))

    if basis in {"ps", "bspline", "p-spline"}:
        bs_code = "ps"
    elif basis in {"thinplate", "tps"}:
        bs_code = "tp"
    elif basis == "duchon":
        # Keep Duchon settings explicit in the external contender formula.
        bs_code = "ds"
    elif basis == "matern":
        # Use explicit stationary Matérn GP in mgcv:
        #   m[1] = -4 -> Matérn with kappa = 2.5, stationary (no linear trend term)
        #   m[2] = 1.0 -> fixed range on z-scored predictors
        # This avoids hidden mgcv defaults and keeps comparison with Rust's
        # explicit Matérn basis fair and reproducible.
        bs_code = "gp"
    else:
        raise RuntimeError(
            f"Unsupported mgcv smooth basis '{basis}' for scenario '{scenario_name}'"
        )

    smooth_cols = cfg.get("smooth_cols")
    if smooth_cols:
        for col in smooth_cols:
            k_val = knot_count + 4 if bs_code == "ps" else knot_count
            if basis == "matern":
                terms.append(
                    f"s({col}, bs='gp', m=c(-4,1.0), k=min({k_val}, nrow(train_df)-1))"
                )
            elif basis == "duchon":
                terms.append(
                    f"s({col}, bs='ds', m=c(1,0), k=min({k_val}, nrow(train_df)-1))"
                )
            else:
                terms.append(f"s({col}, bs='{bs_code}', k=min({k_val}, nrow(train_df)-1))")
    else:
        col = cfg.get("smooth_col")
        if col:
            k_val = knot_count + 4 if bs_code == "ps" else knot_count
            if basis == "matern":
                terms.append(
                    f"s({col}, bs='gp', m=c(-4,1.0), k=min({k_val}, nrow(train_df)-1))"
                )
            elif basis == "duchon":
                terms.append(
                    f"s({col}, bs='ds', m=c(1,0), k=min({k_val}, nrow(train_df)-1))"
                )
            else:
                terms.append(f"s({col}, bs='{bs_code}', k=min({k_val}, nrow(train_df)-1))")
    if not terms:
        raise RuntimeError(f"empty mgcv term list for scenario '{scenario_name}'")
    return f"{target} ~ " + " + ".join(terms)


def _rust_survival_formula_for_scenario(scenario_name, feature_cols=None):
    if scenario_name == "icu_survival_death":
        if feature_cols is not None:
            if len(feature_cols) == 0:
                raise RuntimeError("icu_survival_death: empty feature list for Rust survival formula")
            return " + ".join(f"linear({c})" for c in feature_cols)
        return "linear(age) + s(bmi, type=ps, knots=6) + linear(hr_max) + linear(sysbp_min)"
    if scenario_name == "heart_failure_survival":
        return (
            "linear(age) + linear(anaemia) + linear(log_creatinine_phosphokinase) + linear(diabetes) + "
            "linear(ejection_fraction) + linear(high_blood_pressure) + linear(log_platelets) + "
            "linear(log_serum_creatinine) + linear(serum_sodium) + linear(sex) + linear(smoking)"
        )
    if scenario_name == "cirrhosis_survival":
        return (
            "linear(drug) + linear(sex_male) + linear(ascites) + linear(hepatomegaly) + linear(spiders) + "
            "linear(edema) + linear(age) + linear(bilirubin) + linear(cholesterol) + linear(albumin) + "
            "linear(copper) + linear(alk_phos) + linear(sgot) + linear(tryglicerides) + linear(platelets) + "
            "linear(prothrombin) + linear(stage)"
        )
    if scenario_name == "haberman_survival":
        return "linear(age) + linear(op_year) + s(axil_nodes, type=ps, knots=8)"
    if scenario_name == "icu_survival_los":
        return "linear(age) + linear(bmi) + linear(hr_max) + linear(sysbp_min) + linear(temp_apache)"
    raise RuntimeError(f"No Rust survival formula configured for scenario '{scenario_name}'")


def _append_formula_link_term(formula: str, link_name: str | None) -> str:
    if not link_name:
        return formula
    # Avoid duplicating explicit link(...) terms if callers already injected one.
    if re.search(r"(?<![A-Za-z0-9_])link\s*\(", formula):
        return formula
    if "~" not in formula:
        raise RuntimeError(f"cannot append link term to malformed formula: {formula!r}")
    lhs, rhs = formula.split("~", 1)
    rhs = rhs.strip()
    link_term = f"link(type={str(link_name).strip()})"
    if not rhs:
        return f"{lhs.strip()} ~ {link_term}"
    return f"{lhs.strip()} ~ {rhs} + {link_term}"


def _is_matern_rust_scenario(s_cfg) -> bool:
    cfg = _rust_fit_mapping(s_cfg["name"])
    if cfg is None:
        return False
    return _canonical_smooth_basis(cfg.get("smooth_basis", "ps")) == "matern"


def _make_far_ood_frame(
    train_df: pd.DataFrame,
    *,
    ds: dict,
    smooth_cols: list[str],
    linear_cols: list[str],
    n_points: int = 64,
) -> pd.DataFrame:
    cols = list(ds["features"])
    if not cols:
        return pd.DataFrame()
    n = max(int(n_points), 8)
    rows = []
    for i in range(n):
        row = {}
        bitmask = i
        for j, c in enumerate(cols):
            vals = train_df[c].to_numpy(dtype=float)
            vmin = float(np.min(vals))
            vmax = float(np.max(vals))
            span = max(vmax - vmin, 1e-6)
            far = max(4.0 * span, 6.0)
            if c in smooth_cols:
                go_high = ((bitmask >> (j % 16)) & 1) == 1
                row[c] = (vmax + far) if go_high else (vmin - far)
            elif c in linear_cols:
                row[c] = float(np.mean(vals))
            else:
                row[c] = float(np.mean(vals))
        if ds["family"] == "binomial":
            row[ds["target"]] = 0.0
        elif ds["family"] == "gaussian":
            row[ds["target"]] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _compute_continuous_order_from_lambdas(
    lambda_tilde: list[float] | tuple[float, ...] | np.ndarray,
    normalization_scale: list[float] | tuple[float, ...] | np.ndarray | None = None,
    eps: float = 1e-12,
) -> dict:
    if len(lambda_tilde) < 3:
        return {
            "status": "UndefinedZeroLambda",
            "lambda0": float("nan"),
            "lambda1": float("nan"),
            "lambda2": float("nan"),
            "r_ratio": None,
            "nu": None,
            "kappa2": None,
        }
    lt = [float(lambda_tilde[0]), float(lambda_tilde[1]), float(lambda_tilde[2])]
    scales = [1.0, 1.0, 1.0] if normalization_scale is None else [float(normalization_scale[i]) for i in range(3)]
    if any((not np.isfinite(c)) or c <= 0.0 for c in scales):
        return {
            "status": "UndefinedZeroLambda",
            "lambda0": float("nan"),
            "lambda1": float("nan"),
            "lambda2": float("nan"),
            "r_ratio": None,
            "nu": None,
            "kappa2": None,
        }
    lam = [lt[i] / scales[i] for i in range(3)]
    l0, l1, l2 = lam
    if any(not np.isfinite(v) for v in lam):
        return {"status": "UndefinedZeroLambda", "lambda0": l0, "lambda1": l1, "lambda2": l2, "r_ratio": None, "nu": None, "kappa2": None}
    l_scale = max(abs(l0), abs(l1), abs(l2), 1.0)
    l_floor = eps * l_scale
    if l0 <= l_floor:
        if l1 > l_floor and l2 > l_floor:
            return {"status": "IntrinsicLimit", "lambda0": l0, "lambda1": l1, "lambda2": l2, "r_ratio": None, "nu": 1.0, "kappa2": 0.0}
        return {"status": "UndefinedZeroLambda", "lambda0": l0, "lambda1": l1, "lambda2": l2, "r_ratio": None, "nu": None, "kappa2": None}
    if l2 <= l_floor:
        if l1 > l_floor and np.isfinite(l1):
            return {"status": "FirstOrderLimit", "lambda0": l0, "lambda1": l1, "lambda2": l2, "r_ratio": None, "nu": 1.0, "kappa2": l0 / l1}
        return {"status": "UndefinedZeroLambda", "lambda0": l0, "lambda1": l1, "lambda2": l2, "r_ratio": None, "nu": None, "kappa2": None}
    r_ratio = (l1 * l1) / (l0 * l2)
    if not np.isfinite(r_ratio):
        return {"status": "UndefinedZeroLambda", "lambda0": l0, "lambda1": l1, "lambda2": l2, "r_ratio": None, "nu": None, "kappa2": None}
    disc = l1 * l1 - 4.0 * l0 * l2
    disc_tol = eps * l_scale * l_scale
    status = "NonMaternRegime" if disc < -disc_tol else "Ok"
    if r_ratio <= 2.0 + eps:
        return {"status": status, "lambda0": l0, "lambda1": l1, "lambda2": l2, "r_ratio": r_ratio, "nu": None, "kappa2": None}
    nu = r_ratio / (r_ratio - 2.0)
    kappa2 = l1 / ((r_ratio - 2.0) * l2)
    if (not np.isfinite(nu)) or (not np.isfinite(kappa2)):
        return {"status": "UndefinedZeroLambda", "lambda0": l0, "lambda1": l1, "lambda2": l2, "r_ratio": r_ratio, "nu": None, "kappa2": None}
    return {"status": status, "lambda0": l0, "lambda1": l1, "lambda2": l2, "r_ratio": r_ratio, "nu": float(nu), "kappa2": float(kappa2)}


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return None
    x = x[mask]
    y = y[mask]
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    sx = float(np.std(xr))
    sy = float(np.std(yr))
    if sx < 1e-12 or sy < 1e-12:
        return None
    corr = float(np.corrcoef(xr, yr)[0, 1])
    if not np.isfinite(corr):
        return None
    return corr


def _thread3_cliff_gradient_magnitude(
    collocation_points: np.ndarray, *, feature_cols: list[int], ds: dict
) -> np.ndarray | None:
    coeff_map = ds.get("thread3_cliff_coefficients")
    if not isinstance(coeff_map, dict):
        return None
    if collocation_points.ndim != 2 or collocation_points.shape[0] == 0:
        return None
    if collocation_points.shape[1] != len(feature_cols):
        return None
    coeff_vec = np.array(
        [float(coeff_map.get(ds["features"][int(c)], 0.0)) for c in feature_cols],
        dtype=float,
    )
    coeff_norm = float(np.linalg.norm(coeff_vec))
    if coeff_norm < 1e-12:
        return None
    jump = float(ds.get("thread3_cliff_jump", 0.0))
    sharpness = float(ds.get("thread3_cliff_sharpness", 1.0))
    z = collocation_points.dot(coeff_vec)
    az = np.clip(np.abs(sharpness * z), 0.0, 50.0)
    sech2 = 1.0 / (np.cosh(az) ** 2)
    deta_dz_abs = abs(jump * sharpness) * sech2
    return deta_dz_abs * coeff_norm


def _extract_thread3_adaptive_fold_metrics(model_payload: dict | None, ds: dict) -> dict:
    if not isinstance(model_payload, dict):
        return {}
    diag = model_payload.get("adaptive_regularization_diagnostics")
    if not isinstance(diag, dict):
        return {}
    out: dict[str, float | int | bool] = {}
    mm_iter = diag.get("mm_iterations")
    if isinstance(mm_iter, (int, float)):
        out["mm_iterations"] = int(mm_iter)
    converged = diag.get("converged")
    if isinstance(converged, bool):
        out["adaptive_converged"] = converged

    maps = diag.get("maps")
    if not isinstance(maps, list):
        return out

    def _json_ndarray(value):
        if isinstance(value, dict) and isinstance(value.get("data"), list):
            arr = np.asarray(value.get("data"), dtype=float)
            dim = value.get("dim")
            if isinstance(dim, list) and dim:
                try:
                    shape = tuple(int(d) for d in dim)
                    if int(np.prod(shape, dtype=np.int64)) == int(arr.size):
                        return arr.reshape(shape)
                except Exception:
                    pass
            return arr
        return np.asarray(value, dtype=float)

    corr_rows = []
    for m in maps:
        if not isinstance(m, dict):
            continue
        try:
            feature_cols = [int(x) for x in (m.get("feature_cols") or [])]
            points = _json_ndarray(m.get("collocation_points"))
            inv_g = _json_ndarray(m.get("inv_grad_weight")).reshape(-1)
            inv_c = _json_ndarray(m.get("inv_lap_weight")).reshape(-1)
        except Exception:
            continue
        if points.ndim != 2 or points.shape[0] == 0:
            continue
        if points.shape[0] != inv_g.shape[0] or points.shape[0] != inv_c.shape[0]:
            continue
        grad_mag = _thread3_cliff_gradient_magnitude(points, feature_cols=feature_cols, ds=ds)
        if grad_mag is None or grad_mag.shape[0] != points.shape[0]:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            sens_g = np.where(inv_g > 0.0, 1.0 / inv_g, np.nan)
            sens_c = np.where(inv_c > 0.0, 1.0 / inv_c, np.nan)
        corr_rows.append(
            {
                "n": int(points.shape[0]),
                "corr_g": _rank_corr(sens_g, grad_mag),
                "corr_c": _rank_corr(sens_c, grad_mag),
            }
        )

    if corr_rows:
        denom = max(sum(int(r["n"]) for r in corr_rows), 1)
        valid_g = [(float(r["corr_g"]), int(r["n"])) for r in corr_rows if r["corr_g"] is not None]
        valid_c = [(float(r["corr_c"]), int(r["n"])) for r in corr_rows if r["corr_c"] is not None]
        if valid_g:
            out["thread3_weight_grad_corr"] = float(
                sum(v * w for v, w in valid_g) / max(sum(w for _, w in valid_g), 1)
            )
        if valid_c:
            out["thread3_weight_curvature_corr"] = float(
                sum(v * w for v, w in valid_c) / max(sum(w for _, w in valid_c), 1)
            )
        out["thread3_collocation_points"] = int(denom)
    return out


def _rust_survival_fit_options_for_scenario(scenario_name):
    # Keep defaults close to CLI behavior and only enable a flexible time basis
    # where it provides clear discrimination gains.
    if scenario_name in {"heart_failure_survival", "cirrhosis_survival"}:
        return {
            "time_basis": "bspline",
            "time_degree": 3,
            "time_num_internal_knots": 8,
            "time_smooth_lambda": 1e-2,
            "ridge_lambda": 1e-6,
        }
    return {
        "time_basis": "linear",
        "ridge_lambda": 1e-4,
    }


def _ensure_rust_binary():
    global _RUST_BIN_PATH
    if _RUST_BIN_PATH is not None:
        return _RUST_BIN_PATH
    prebuilt = os.environ.get("BENCH_GAM_BIN", "").strip()
    if prebuilt:
        candidate = Path(prebuilt).expanduser().resolve()
        if not candidate.exists():
            raise RuntimeError(f"BENCH_GAM_BIN points to missing file: {candidate}")
        if not os.access(candidate, os.X_OK):
            raise RuntimeError(f"BENCH_GAM_BIN is not executable: {candidate}")
        _RUST_BIN_PATH = candidate
        return _RUST_BIN_PATH
    code, out, err = run_cmd(["cargo", "build", "--release", "--bin", "gam"], cwd=ROOT)
    if code != 0:
        raise RuntimeError((err or out).strip() or "failed to build Rust release binary")
    _RUST_BIN_PATH = ROOT / "target" / "release" / "gam"
    if not _RUST_BIN_PATH.exists():
        raise RuntimeError(f"missing Rust binary at {_RUST_BIN_PATH}")
    return _RUST_BIN_PATH


def run_rust_scenario_cv(
    scenario,
    *,
    contender_name: str = "rust_gam",
    binomial_link: str | None = None,
    ds: dict | None = None,
    folds: list[Fold] | None = None,
    shared_fold_artifacts: list[SharedFoldArtifact] | None = None,
    rust_cfg_override: dict | None = None,
    eval_ood: bool = False,
    collect_continuous_order: bool = False,
    collect_adaptive_diagnostics: bool = False,
    rust_fit_extra_args: list[str] | None = None,
):
    scenario_name = scenario["name"]
    if ds is None:
        ds = dataset_for_scenario(scenario)
    if folds is None:
        folds = folds_for_dataset(ds)

    try:
        rust_bin = _ensure_rust_binary()
    except Exception as e:
        return {
            "contender": contender_name,
            "scenario_name": scenario_name,
            "status": "failed",
            "error": str(e),
        }

    base_df = pd.DataFrame(ds["rows"])
    cv_rows = []
    eval_suffix = _evaluation_suffix(folds)
    ood_rows = []
    continuous_rows = []
    adaptive_rows = []
    eval_suffix = _evaluation_suffix(folds)
    rust_cfg = _effective_rust_fit_mapping(scenario_name, rust_cfg_override) or {}
    smooth_cols = list(rust_cfg.get("smooth_cols") or ([rust_cfg["smooth_col"]] if "smooth_col" in rust_cfg else []))
    linear_cols = list(rust_cfg.get("linear_cols", []))
    # Use a workspace-local temp root to reduce /tmp lifecycle flakiness in CI.
    with tempfile.TemporaryDirectory(prefix="gam_bench_rust_cv_", dir=str(BENCH_DIR)) as td:
        td_path = Path(td)
        for fold_id, fold in enumerate(folds):
            train_df = base_df.iloc[fold.train_idx].copy()
            test_df = base_df.iloc[fold.test_idx].copy()
            test_eval_df = test_df.copy()
            model_path = td_path / f"model_{fold_id}.json"
            pred_path = td_path / f"pred_{fold_id}.csv"
            shared_artifact = (
                shared_fold_artifacts[fold_id]
                if shared_fold_artifacts is not None and fold_id < len(shared_fold_artifacts)
                else None
            )

            if ds["family"] == "survival":
                train_path = td_path / f"train_{fold_id}.csv"
                test_path = td_path / f"test_{fold_id}.csv"
                fit_feature_cols = list(ds["features"])
                for col in ds["features"]:
                    mu = float(train_df[col].mean())
                    sdv = float(train_df[col].std())
                    if (not np.isfinite(sdv)) or sdv < 1e-8:
                        sdv = 1.0
                    train_df[col] = (train_df[col] - mu) / sdv
                    test_df[col] = (test_df[col] - mu) / sdv
                if scenario_name == "icu_survival_death" and "bmi" in ds["features"]:
                    train_df, test_df, bmi_spline_cols = _augment_bmi_spline_linear_hinges(
                        train_df, test_df, n_knots=6
                    )
                    fit_feature_cols = [c for c in ds["features"] if c != "bmi"] + bmi_spline_cols
                train_df["__entry"] = 0.0
                # Rust survival predict requires an explicit evaluation time.
                horizon = _survival_eval_horizon(train_df, ds["time_col"])
                test_pred_df = test_df.copy()
                test_pred_df["__entry"] = 0.0
                test_pred_df[ds["time_col"]] = horizon
                train_df.to_csv(train_path, index=False)
                test_pred_df.to_csv(test_path, index=False)
                rhs_formula = _rust_survival_formula_for_scenario(
                    scenario_name, feature_cols=fit_feature_cols
                )
                fit_formula = (
                    f"Surv(__entry, {ds['time_col']}, {ds['event_col']}) ~ {rhs_formula}"
                )
                fit_cmd = [
                    str(rust_bin),
                    "fit",
                    "--out",
                    str(model_path),
                ]
            else:
                if shared_artifact is not None:
                    train_path = shared_artifact.train_scaled_csv
                    test_path = shared_artifact.test_scaled_csv
                else:
                    train_path = td_path / f"train_{fold_id}.csv"
                    test_path = td_path / f"test_{fold_id}.csv"
                    train_df, test_df = zscore_train_test(train_df, test_df, ds["features"])
                    train_df.to_csv(train_path, index=False)
                    test_df.to_csv(test_path, index=False)
                _family, formula = _rust_formula_for_scenario(
                    scenario_name,
                    ds,
                    cfg_override=rust_cfg_override,
                )
                if ds["family"] == "binomial" and binomial_link:
                    formula = _append_formula_link_term(formula, binomial_link)
                    fit_cmd = [
                        str(rust_bin),
                        "fit",
                        "--out",
                        str(model_path),
                    ]
                else:
                    fit_cmd = [
                        str(rust_bin),
                        "fit",
                        "--out",
                        str(model_path),
                    ]

            if rust_fit_extra_args:
                fit_cmd.extend([str(x) for x in rust_fit_extra_args])
            fit_cmd.extend(["--no-summary"])
            fit_cmd.extend([str(train_path), fit_formula if ds["family"] == "survival" else formula])

            def _looks_like_missing_csv(msg: str) -> bool:
                m = (msg or "").lower()
                return ("failed to open csv" in m) and ("no such file or directory" in m)

            def _ensure_fold_csvs():
                if ds["family"] == "survival":
                    train_df.to_csv(train_path, index=False)
                    test_df.to_csv(test_path, index=False)

            t0 = perf_counter()
            code, out, err = run_cmd(fit_cmd, cwd=ROOT)
            # Defensive retry for sporadic CI file-not-found while opening fold CSVs.
            if code != 0 and _looks_like_missing_csv(err or out):
                _ensure_fold_csvs()
                code, out, err = run_cmd(fit_cmd, cwd=ROOT)
            fit_sec = perf_counter() - t0
            if code != 0:
                return {
                    "contender": contender_name,
                    "scenario_name": scenario_name,
                    "status": "failed",
                    "error": (err.strip() or out.strip() or "rust fit failed"),
                }
            model_payload = None
            try:
                model_payload = json.loads(model_path.read_text())
                if isinstance(model_payload, dict) and "payload" in model_payload:
                    model_payload = model_payload.get("payload", {})
            except Exception:
                model_payload = None
            if collect_continuous_order and model_payload is not None:
                lambdas = model_payload.get("fit_result", {}).get("lambdas", [])
                if isinstance(lambdas, list) and len(lambdas) >= 3:
                    co = _compute_continuous_order_from_lambdas(lambdas[:3], normalization_scale=[1.0, 1.0, 1.0], eps=1e-12)
                    co["n_test"] = int(len(fold.test_idx))
                    continuous_rows.append(co)
            if collect_adaptive_diagnostics:
                adaptive_row = _extract_thread3_adaptive_fold_metrics(model_payload, ds)
                if adaptive_row:
                    adaptive_row["n_test"] = int(len(fold.test_idx))
                    adaptive_rows.append(adaptive_row)

            pred_cmd = [
                str(rust_bin),
                "predict",
                str(model_path),
                str(test_path),
                "--out",
                str(pred_path),
            ]
            t1 = perf_counter()
            code, out, err = run_cmd(pred_cmd, cwd=ROOT)
            if code != 0 and _looks_like_missing_csv(err or out):
                _ensure_fold_csvs()
                code, out, err = run_cmd(pred_cmd, cwd=ROOT)
            pred_sec = perf_counter() - t1
            if code != 0:
                return {
                    "contender": contender_name,
                    "scenario_name": scenario_name,
                    "status": "failed",
                    "error": (err.strip() or out.strip() or "rust predict failed"),
                }
            pred_df = pd.read_csv(pred_path)
            if "mean" not in pred_df.columns:
                return {
                    "contender": contender_name,
                    "scenario_name": scenario_name,
                    "status": "failed",
                    "error": "rust prediction output missing 'mean' column",
                }
            pred = pred_df["mean"].to_numpy(dtype=float)

            if ds["family"] == "binomial":
                y_test = test_df[ds["target"]].to_numpy(dtype=float)
                cv_rows.append(
                    {
                        "fit_sec": float(fit_sec),
                        "predict_sec": float(pred_sec),
                        "auc": auc_score(y_test, pred),
                        "brier": brier_score(y_test, pred),
                        "logloss": log_loss_score(y_test, pred),
                        "nagelkerke_r2": nagelkerke_r2_score(y_test, pred),
                        "n_test": int(len(fold.test_idx)),
                        "model_spec": (
                            f"gam fit/predict via release binary {eval_suffix}"
                            if not binomial_link
                            else f"gam fit/predict via release binary (link={binomial_link}) {eval_suffix}"
                        ),
                    }
                )
                if eval_ood and smooth_cols:
                    ood_df = _make_far_ood_frame(
                        train_df,
                        ds=ds,
                        smooth_cols=smooth_cols,
                        linear_cols=linear_cols,
                        n_points=max(32, int(len(fold.test_idx))),
                    )
                    if not ood_df.empty:
                        ood_path = td_path / f"ood_{fold_id}.csv"
                        ood_pred_path = td_path / f"ood_pred_{fold_id}.csv"
                        ood_df.to_csv(ood_path, index=False)
                        ood_pred_cmd = [
                            str(rust_bin),
                            "predict",
                            str(model_path),
                            str(ood_path),
                            "--out",
                            str(ood_pred_path),
                        ]
                        code, out, err = run_cmd(ood_pred_cmd, cwd=ROOT)
                        if code == 0 and ood_pred_path.is_file():
                            ood_pred_df = pd.read_csv(ood_pred_path)
                            if "mean" in ood_pred_df.columns:
                                p_ood = np.clip(ood_pred_df["mean"].to_numpy(dtype=float), 1e-9, 1 - 1e-9)
                                baseline = float(np.clip(np.mean(y_test), 1e-9, 1 - 1e-9))
                                ood_rows.append(
                                    {
                                        "n_test": int(len(fold.test_idx)),
                                        "ood_abs_dev_from_baseline": float(np.mean(np.abs(p_ood - baseline))),
                                        "ood_max_abs_dev_from_baseline": float(np.max(np.abs(p_ood - baseline))),
                                        "ood_mean_abs_logit": float(np.mean(np.abs(np.log(p_ood / (1.0 - p_ood))))),
                                    }
                                )
            elif ds["family"] == "gaussian":
                y_test = test_df[ds["target"]].to_numpy(dtype=float)
                try:
                    if model_payload is None:
                        model_payload = json.loads(model_path.read_text())
                        if isinstance(model_payload, dict) and "payload" in model_payload:
                            model_payload = model_payload.get("payload", {})
                    sigma_hat_raw = model_payload.get("fit_result", {}).get(
                        "standard_deviation", None
                    )
                    sigma_hat = float(sigma_hat_raw)
                except Exception as e:
                    return {
                        "contender": contender_name,
                        "scenario_name": scenario_name,
                        "status": "failed",
                        "fold_id": int(fold_id),
                        "n_train": int(len(fold.train_idx)),
                        "n_test": int(len(fold.test_idx)),
                        "n_folds": int(len(folds)),
                        "error": (
                            "rust gaussian fit output missing/invalid "
                            f"fit_result.standard_deviation: {e}"
                        ),
                    }
                if (not np.isfinite(sigma_hat)) or sigma_hat <= 0.0:
                    return {
                        "contender": contender_name,
                        "scenario_name": scenario_name,
                        "status": "failed",
                        "fold_id": int(fold_id),
                        "n_train": int(len(fold.train_idx)),
                        "n_test": int(len(fold.test_idx)),
                        "n_folds": int(len(folds)),
                        "error": (
                            "rust gaussian fit_result.standard_deviation must be finite and > 0; "
                            f"got {sigma_hat!r}"
                        ),
                    }
                cv_rows.append(
                    {
                        "fit_sec": float(fit_sec),
                        "predict_sec": float(pred_sec),
                        "logloss": gaussian_log_loss_score(y_test, pred, sigma_hat),
                        "mse": mse_score(y_test, pred),
                        "rmse": rmse_score(y_test, pred),
                        "mae": mae_score(y_test, pred),
                        "r2": r2_score(y_test, pred),
                        "n_test": int(len(fold.test_idx)),
                        "model_spec": f"gam fit/predict via release binary {eval_suffix}",
                    }
                )
            else:
                event_times = test_eval_df[ds["time_col"]].to_numpy(dtype=float)
                events = test_eval_df[ds["event_col"]].to_numpy(dtype=float)
                try:
                    risk_score, score_src = _survival_risk_from_rust_pred(pred_df)
                except RuntimeError as e:
                    return {
                        "contender": contender_name,
                        "scenario_name": scenario_name,
                        "status": "failed",
                        "error": str(e),
                    }
                cidx = _lifelines_cindex_from_risk(event_times, risk_score, events)
                cv_rows.append(
                    {
                        "fit_sec": float(fit_sec),
                        "predict_sec": float(pred_sec),
                        "auc": cidx,
                        "brier": survival_partial_brier_score(event_times, risk_score, events),
                        "logloss": survival_partial_log_loss(event_times, risk_score, events),
                        "nagelkerke_r2": survival_partial_nagelkerke_r2(event_times, risk_score, events),
                        "n_test": int(len(fold.test_idx)),
                        "predict_horizon": float(horizon),
                        "predict_horizon_policy": "global train-fold median time",
                        "model_spec": (
                            "survival model via release binary "
                            f"(c-index on risk score from '{score_src}') {eval_suffix}"
                        ),
                    }
                )

    metrics = aggregate_cv_rows(cv_rows, ds["family"])
    if ood_rows:
        denom = max(sum(int(r["n_test"]) for r in ood_rows), 1)
        metrics["ood_abs_dev_from_baseline"] = float(
            sum(float(r["ood_abs_dev_from_baseline"]) * int(r["n_test"]) for r in ood_rows) / denom
        )
        metrics["ood_max_abs_dev_from_baseline"] = float(
            sum(float(r["ood_max_abs_dev_from_baseline"]) * int(r["n_test"]) for r in ood_rows) / denom
        )
        metrics["ood_mean_abs_logit"] = float(
            sum(float(r["ood_mean_abs_logit"]) * int(r["n_test"]) for r in ood_rows) / denom
        )
    if continuous_rows:
        status_counts: dict[str, int] = {}
        for r in continuous_rows:
            s = str(r.get("status", "UndefinedZeroLambda"))
            status_counts[s] = int(status_counts.get(s, 0) + 1)
        denom = max(sum(int(r["n_test"]) for r in continuous_rows), 1)
        valid_nu = [(float(r["nu"]), int(r["n_test"])) for r in continuous_rows if r.get("nu") is not None]
        valid_k2 = [(float(r["kappa2"]), int(r["n_test"])) for r in continuous_rows if r.get("kappa2") is not None]
        metrics["continuous_order_status_counts"] = status_counts
        metrics["continuous_order_status_mode"] = max(status_counts, key=status_counts.get)
        metrics["continuous_order_nu"] = (
            float(sum(v * w for v, w in valid_nu) / max(sum(w for _, w in valid_nu), 1))
            if valid_nu
            else None
        )
        metrics["continuous_order_kappa2"] = (
            float(sum(v * w for v, w in valid_k2) / max(sum(w for _, w in valid_k2), 1))
            if valid_k2
            else None
        )
        true_nu = ds.get("thread2_true_nu")
        if true_nu is not None and metrics["continuous_order_nu"] is not None:
            metrics["continuous_order_nu_abs_error"] = float(
                abs(float(metrics["continuous_order_nu"]) - float(true_nu))
            )
        expected = ds.get("thread2_expected_statuses")
        if isinstance(expected, list) and expected:
            mode = str(metrics["continuous_order_status_mode"])
            metrics["continuous_order_boundary_ok"] = bool(mode in {str(x) for x in expected})
            metrics["continuous_order_expected_statuses"] = [str(x) for x in expected]
    if adaptive_rows:
        mm_vals = [(int(r["mm_iterations"]), int(r["n_test"])) for r in adaptive_rows if "mm_iterations" in r]
        if mm_vals:
            metrics["mm_iterations"] = float(
                sum(v * w for v, w in mm_vals) / max(sum(w for _, w in mm_vals), 1)
            )
        conv_vals = [
            (1.0 if bool(r.get("adaptive_converged")) else 0.0, int(r["n_test"]))
            for r in adaptive_rows
            if "adaptive_converged" in r
        ]
        if conv_vals:
            metrics["adaptive_converged_rate"] = float(
                sum(v * w for v, w in conv_vals) / max(sum(w for _, w in conv_vals), 1)
            )
        grad_corr = [
            (float(r["thread3_weight_grad_corr"]), int(r["n_test"]))
            for r in adaptive_rows
            if "thread3_weight_grad_corr" in r
        ]
        curv_corr = [
            (float(r["thread3_weight_curvature_corr"]), int(r["n_test"]))
            for r in adaptive_rows
            if "thread3_weight_curvature_corr" in r
        ]
        if grad_corr:
            metrics["thread3_weight_grad_corr"] = float(
                sum(v * w for v, w in grad_corr) / max(sum(w for _, w in grad_corr), 1)
            )
        if curv_corr:
            metrics["thread3_weight_curvature_corr"] = float(
                sum(v * w for v, w in curv_corr) / max(sum(w for _, w in curv_corr), 1)
            )
        colloc_rows = [
            (int(r["thread3_collocation_points"]), int(r["n_test"]))
            for r in adaptive_rows
            if "thread3_collocation_points" in r
        ]
        if colloc_rows:
            metrics["thread3_collocation_points"] = float(
                sum(v * w for v, w in colloc_rows) / max(sum(w for _, w in colloc_rows), 1)
            )
    return {
        "contender": contender_name,
        "family": ds["family"],
        "scenario_name": scenario_name,
        "status": "ok",
        **metrics,
        "model_spec": cv_rows[0]["model_spec"],
    }


def run_rust_sas_scenario_cv(
    scenario,
    *,
    ds: dict | None = None,
    folds: list[Fold] | None = None,
    shared_fold_artifacts: list[SharedFoldArtifact] | None = None,
):
    if ds is None:
        ds = dataset_for_scenario(scenario)
    if ds.get("family") != "binomial":
        return None
    return run_rust_scenario_cv(
        scenario,
        contender_name="rust_gam_sas",
        binomial_link="sas",
        ds=ds,
        folds=folds,
        shared_fold_artifacts=shared_fold_artifacts,
    )


def _run_rust_gamlss_scenario_cv_variant(
    scenario,
    *,
    contender_name: str,
    binomial_cli_family: str,
    binomial_model_spec_label: str,
    binomial_extra_fit_args: list[str] | None = None,
    ds: dict | None = None,
    folds: list[Fold] | None = None,
    shared_fold_artifacts: list[SharedFoldArtifact] | None = None,
):
    scenario_name = scenario["name"]
    # Run for any scenario with a valid formula mapping (not just geo scenarios).
    if _rust_fit_mapping(scenario_name) is None:
        return None

    if ds is None:
        ds = dataset_for_scenario(scenario)
    family = ds["family"]
    if family not in ("binomial", "gaussian"):
        return None
    if folds is None:
        folds = folds_for_dataset(ds)

    try:
        rust_bin = _ensure_rust_binary()
    except Exception as e:
        return {
            "contender": contender_name,
            "scenario_name": scenario_name,
            "status": "failed",
            "error": str(e),
        }

    _, mean_formula = _rust_formula_for_scenario(scenario_name, ds)
    cli_family = binomial_cli_family if family == "binomial" else "gaussian"
    binom_extra = list(binomial_extra_fit_args or [])
    base_df = pd.DataFrame(ds["rows"])
    cv_rows = []

    with tempfile.TemporaryDirectory(prefix="gam_bench_rust_gamlss_cv_", dir=str(BENCH_DIR)) as td:
        td_path = Path(td)
        for fold_id, fold in enumerate(folds):
            train_df = base_df.iloc[fold.train_idx].copy()
            test_df = base_df.iloc[fold.test_idx].copy()
            shared_artifact = (
                shared_fold_artifacts[fold_id]
                if shared_fold_artifacts is not None and fold_id < len(shared_fold_artifacts)
                else None
            )
            if shared_artifact is not None:
                train_path = shared_artifact.train_scaled_csv
                test_path = shared_artifact.test_scaled_csv
            else:
                # Z-score features (matches the main rust_gam contender).
                train_df, test_df = zscore_train_test(train_df, test_df, ds["features"])
                train_path = td_path / f"train_{fold_id}.csv"
                test_path = td_path / f"test_{fold_id}.csv"
                train_df.to_csv(train_path, index=False)
                test_df.to_csv(test_path, index=False)
            sigma_rhs = _sigma_feature_rhs(ds, scenario_name, n_train=len(fold.train_idx))
            noise_formula = "y ~ " + (
                " + ".join(f"linear({c})" for c in sigma_rhs.split(" + ")) if sigma_rhs != "1" else "1"
            )
            model_path = td_path / f"model_{fold_id}.json"
            pred_path = td_path / f"pred_{fold_id}.csv"

            fit_cmd = [
                str(rust_bin),
                "fit",
                "--predict-noise",
                noise_formula,
                "--out",
                str(model_path),
            ]
            if family == "binomial" and binom_extra:
                fit_cmd.extend(binom_extra)
            fit_cmd.extend(["--no-summary"])
            fit_cmd.extend([str(train_path), mean_formula])
            t0 = perf_counter()
            code, out, err = run_cmd(fit_cmd, cwd=ROOT)
            fit_sec = perf_counter() - t0
            if code != 0:
                return {
                    "contender": contender_name,
                    "scenario_name": scenario_name,
                    "status": "failed",
                    "fold_id": int(fold_id),
                    "n_train": int(len(fold.train_idx)),
                    "n_test": int(len(fold.test_idx)),
                    "n_folds": int(len(folds)),
                    "error": (err.strip() or out.strip() or "rust gamlss fit failed"),
                }

            pred_cmd = [
                str(rust_bin),
                "predict",
                str(model_path),
                str(test_path),
                "--out",
                str(pred_path),
            ]
            t1 = perf_counter()
            code, out, err = run_cmd(pred_cmd, cwd=ROOT)
            pred_sec = perf_counter() - t1
            if code != 0:
                return {
                    "contender": contender_name,
                    "scenario_name": scenario_name,
                    "status": "failed",
                    "fold_id": int(fold_id),
                    "n_train": int(len(fold.train_idx)),
                    "n_test": int(len(fold.test_idx)),
                    "n_folds": int(len(folds)),
                    "error": (err.strip() or out.strip() or "rust gamlss predict failed"),
                }
            pred_df = pd.read_csv(pred_path)
            if "mean" not in pred_df.columns:
                return {
                    "contender": contender_name,
                    "scenario_name": scenario_name,
                    "status": "failed",
                    "fold_id": int(fold_id),
                    "n_train": int(len(fold.train_idx)),
                    "n_test": int(len(fold.test_idx)),
                    "n_folds": int(len(folds)),
                    "error": "rust gamlss prediction output missing 'mean' column",
                }
            pred = pred_df["mean"].to_numpy(dtype=float)
            y_test = test_df[ds["target"]].to_numpy(dtype=float)

            if family == "binomial":
                cv_rows.append(
                    {
                        "fit_sec": float(fit_sec),
                        "predict_sec": float(pred_sec),
                        "auc": auc_score(y_test, pred),
                        "brier": brier_score(y_test, pred),
                        "logloss": log_loss_score(y_test, pred),
                        "nagelkerke_r2": nagelkerke_r2_score(y_test, pred),
                        "n_test": int(len(fold.test_idx)),
                        "model_spec": f"{binomial_model_spec_label} {eval_suffix}",
                    }
                )
            else:
                if "sigma" not in pred_df.columns:
                    return {
                        "contender": contender_name,
                        "scenario_name": scenario_name,
                        "status": "failed",
                        "fold_id": int(fold_id),
                        "n_train": int(len(fold.train_idx)),
                        "n_test": int(len(fold.test_idx)),
                        "n_folds": int(len(folds)),
                        "error": "rust gamlss gaussian prediction output missing 'sigma' column",
                    }
                sigma_hat = pred_df["sigma"].to_numpy(dtype=float)
                if sigma_hat.shape[0] != pred.shape[0]:
                    return {
                        "contender": contender_name,
                        "scenario_name": scenario_name,
                        "status": "failed",
                        "fold_id": int(fold_id),
                        "n_train": int(len(fold.train_idx)),
                        "n_test": int(len(fold.test_idx)),
                        "n_folds": int(len(folds)),
                        "error": (
                            "rust gamlss gaussian prediction output has invalid 'sigma' length "
                            f"(got {sigma_hat.shape[0]}, expected {pred.shape[0]})"
                        ),
                    }
                bad_sigma = ~np.isfinite(sigma_hat) | (sigma_hat <= 0.0)
                if np.any(bad_sigma):
                    return {
                        "contender": contender_name,
                        "scenario_name": scenario_name,
                        "status": "failed",
                        "fold_id": int(fold_id),
                        "n_train": int(len(fold.train_idx)),
                        "n_test": int(len(fold.test_idx)),
                        "n_folds": int(len(folds)),
                        "error": (
                            "rust gamlss gaussian prediction output has non-finite or non-positive "
                            f"'sigma' values ({int(np.sum(bad_sigma))} invalid rows)"
                        ),
                    }
                cv_rows.append(
                    {
                        "fit_sec": float(fit_sec),
                        "predict_sec": float(pred_sec),
                        "logloss": gaussian_log_loss_score(y_test, pred, sigma_hat),
                        "mse": mse_score(y_test, pred),
                        "rmse": rmse_score(y_test, pred),
                        "mae": mae_score(y_test, pred),
                        "r2": r2_score(y_test, pred),
                        "n_test": int(len(fold.test_idx)),
                        "model_spec": f"gamlss gaussian location-scale via release binary {eval_suffix}",
                    }
                )

    metrics = aggregate_cv_rows(cv_rows, family)
    return {
        "contender": contender_name,
        "family": family,
        "scenario_name": scenario_name,
        "status": "ok",
        **metrics,
        "model_spec": cv_rows[0]["model_spec"],
    }


def run_rust_gamlss_scenario_cv(
    scenario,
    *,
    ds: dict | None = None,
    folds: list[Fold] | None = None,
    shared_fold_artifacts: list[SharedFoldArtifact] | None = None,
):
    return _run_rust_gamlss_scenario_cv_variant(
        scenario,
        contender_name="rust_gamlss",
        binomial_cli_family="binomial-probit",
        binomial_model_spec_label="gamlss binomial-probit location-scale via release binary",
        ds=ds,
        folds=folds,
        shared_fold_artifacts=shared_fold_artifacts,
    )


def run_rust_gamlss_survival_cv(
    scenario,
    *,
    ds: dict | None = None,
    folds: list[Fold] | None = None,
):
    """Run the Rust binary with --survival-likelihood probit-location-scale for survival scenarios."""
    scenario_name = scenario["name"]
    if ds is None:
        ds = dataset_for_scenario(scenario)
    if ds["family"] != "survival":
        return None
    if folds is None:
        folds = folds_for_dataset(ds)

    try:
        rust_bin = _ensure_rust_binary()
    except Exception as e:
        return {
            "contender": "rust_gamlss_survival",
            "scenario_name": scenario_name,
            "status": "failed",
            "error": str(e),
        }

    base_df = pd.DataFrame(ds["rows"])
    cv_rows = []

    with tempfile.TemporaryDirectory(prefix="gam_bench_rust_gamlss_surv_cv_", dir=str(BENCH_DIR)) as td:
        td_path = Path(td)
        for fold_id, fold in enumerate(folds):
            train_df = base_df.iloc[fold.train_idx].copy()
            test_df = base_df.iloc[fold.test_idx].copy()
            test_eval_df = test_df.copy()
            train_path = td_path / f"train_{fold_id}.csv"
            test_path = td_path / f"test_{fold_id}.csv"
            model_path = td_path / f"model_{fold_id}.json"
            pred_path = td_path / f"pred_{fold_id}.csv"

            # Z-score features.
            fit_feature_cols = list(ds["features"])
            for col in ds["features"]:
                mu = float(train_df[col].mean())
                sdv = float(train_df[col].std())
                if (not np.isfinite(sdv)) or sdv < 1e-8:
                    sdv = 1.0
                train_df[col] = (train_df[col] - mu) / sdv
                test_df[col] = (test_df[col] - mu) / sdv
            if scenario_name == "icu_survival_death" and "bmi" in ds["features"]:
                train_df, test_df, bmi_spline_cols = _augment_bmi_spline_linear_hinges(
                    train_df, test_df, n_knots=6
                )
                fit_feature_cols = [c for c in ds["features"] if c != "bmi"] + bmi_spline_cols
            train_df["__entry"] = 0.0
            horizon = _survival_eval_horizon(train_df, ds["time_col"])
            test_pred_df = test_df.copy()
            test_pred_df["__entry"] = 0.0
            test_pred_df[ds["time_col"]] = horizon
            train_df.to_csv(train_path, index=False)
            test_pred_df.to_csv(test_path, index=False)

            rhs_formula = _rust_survival_formula_for_scenario(
                scenario_name, feature_cols=fit_feature_cols
            )
            fit_formula = (
                f"Surv(__entry, {ds['time_col']}, {ds['event_col']}) ~ {rhs_formula}"
            )

            fit_cmd = [
                str(rust_bin),
                "fit",
                "--survival-likelihood",
                "probit-location-scale",
                "--out",
                str(model_path),
            ]
            fit_cmd.extend(["--no-summary"])
            fit_cmd.extend([str(train_path), fit_formula])

            t0 = perf_counter()
            code, out, err = run_cmd(fit_cmd, cwd=ROOT)
            fit_sec = perf_counter() - t0
            if code != 0:
                return {
                    "contender": "rust_gamlss_survival",
                    "scenario_name": scenario_name,
                    "status": "failed",
                    "fold_id": int(fold_id),
                    "error": (err.strip() or out.strip() or "rust gamlss survival fit failed"),
                }

            pred_cmd = [
                str(rust_bin), "predict",
                str(model_path), str(test_path),
                "--out", str(pred_path),
            ]
            t1 = perf_counter()
            code, out, err = run_cmd(pred_cmd, cwd=ROOT)
            pred_sec = perf_counter() - t1
            if code != 0:
                return {
                    "contender": "rust_gamlss_survival",
                    "scenario_name": scenario_name,
                    "status": "failed",
                    "fold_id": int(fold_id),
                    "error": (err.strip() or out.strip() or "rust gamlss survival predict failed"),
                }
            pred_df = pd.read_csv(pred_path)

            event_times = test_eval_df[ds["time_col"]].to_numpy(dtype=float)
            events = test_eval_df[ds["event_col"]].to_numpy(dtype=float)
            try:
                risk_score, score_src = _survival_risk_from_rust_pred(pred_df)
            except RuntimeError as e:
                return {
                    "contender": "rust_gamlss_survival",
                    "scenario_name": scenario_name,
                    "status": "failed",
                    "error": str(e),
                }
            cidx = _lifelines_cindex_from_risk(event_times, risk_score, events)
            cv_rows.append(
                {
                    "fit_sec": float(fit_sec),
                    "predict_sec": float(pred_sec),
                    "auc": cidx,
                    "brier": survival_partial_brier_score(event_times, risk_score, events),
                    "logloss": survival_partial_log_loss(event_times, risk_score, events),
                    "nagelkerke_r2": survival_partial_nagelkerke_r2(event_times, risk_score, events),
                    "n_test": int(len(fold.test_idx)),
                    "predict_horizon": float(horizon),
                    "predict_horizon_policy": "global train-fold median time",
                        "model_spec": (
                            "survival probit-location-scale via release binary "
                            f"(c-index on risk score from '{score_src}') [5-fold CV]"
                    ),
                }
            )

    metrics = aggregate_cv_rows(cv_rows, ds["family"])
    return {
        "contender": "rust_gamlss_survival",
        "family": ds["family"],
        "scenario_name": scenario_name,
        "status": "ok",
        **metrics,
        "model_spec": cv_rows[0]["model_spec"],
    }


def _is_gamlss_benchmark_scenario(scenario_name: str) -> bool:
    return _rust_fit_mapping(scenario_name) is not None


def _gamlss_mu_formula_for_scenario(scenario_name: str, ds):
    cfg = _rust_fit_mapping(scenario_name)
    if cfg is None:
        return None
    if ds["family"] not in ("binomial", "gaussian"):
        return None

    terms = [str(c) for c in cfg.get("linear_cols", [])]
    knot_count = max(4, int(cfg.get("knots", 8)))
    smooth_cols = cfg.get("smooth_cols")
    if smooth_cols:
        for col in smooth_cols:
            # R gamlss::pb uses df (effective basis dimension control).
            terms.append(f"pb({col}, df=min({knot_count}, nrow(train_df)-1))")
    elif cfg.get("smooth_col"):
        col = cfg["smooth_col"]
        terms.append(f"pb({col}, df=min({knot_count}, nrow(train_df)-1))")

    if not terms:
        return None
    return f"{ds['target']} ~ " + " + ".join(terms)


def _sigma_feature_rhs(
    ds: dict,
    scenario_name: str | None = None,
    *,
    n_train: int | None = None,
) -> str:
    features = [str(c) for c in ds.get("features", [])]
    return " + ".join(features) if features else "1"


def _sigma_feature_formula(
    ds: dict,
    scenario_name: str | None = None,
    *,
    n_train: int | None = None,
) -> str:
    return "~ " + _sigma_feature_rhs(ds, scenario_name, n_train=n_train)


def run_external_r_gamlss_cv(scenario, *, ds: dict | None = None, folds: list[Fold] | None = None):
    scenario_name = scenario["name"]
    if not _is_gamlss_benchmark_scenario(scenario_name):
        return None

    if ds is None:
        ds = dataset_for_scenario(scenario)
    family = ds["family"]
    if family not in ("binomial", "gaussian"):
        return None
    if folds is None:
        folds = folds_for_dataset(ds)

    mu_formula = _gamlss_mu_formula_for_scenario(scenario_name, ds)
    if not mu_formula:
        return None

    with tempfile.TemporaryDirectory(prefix="gam_bench_r_gamlss_cv_") as td:
        td_path = Path(td)
        data_path = td_path / "data.json"
        out_path = td_path / "out.json"
        script_path = td_path / "run_gamlss_cv.R"

        payload = {
            "dataset": ds,
            "scenario_name": scenario_name,
            "mu_formula": mu_formula,
            "sigma_formula": _sigma_feature_formula(ds, scenario_name),
        }
        data_path.write_text(json.dumps(payload))

        script = r'''
args <- commandArgs(trailingOnly = TRUE)
data_path <- args[1]
out_path <- args[2]
train_idx_path <- args[3]
test_idx_path <- args[4]

suppressPackageStartupMessages({
  library(gamlss)
  library(gamlss.dist)
  library(jsonlite)
})

payload <- fromJSON(data_path, simplifyVector = TRUE)
df <- as.data.frame(payload$dataset$rows)
target_name <- as.character(payload$dataset$target)
mu_formula <- as.character(payload$mu_formula)
sigma_formula <- as.formula(as.character(payload$sigma_formula))
family_name <- as.character(payload$dataset$family)

train_idx <- scan(train_idx_path, what=integer(), quiet=TRUE) + 1L
test_idx <- scan(test_idx_path, what=integer(), quiet=TRUE) + 1L

train_df <- df[train_idx, , drop=FALSE]
test_df <- df[test_idx, , drop=FALSE]
y_test <- as.numeric(df[[target_name]])[test_idx]

feature_cols <- as.character(payload$dataset$features)
for (cn in feature_cols) {
  mu <- mean(train_df[[cn]])
  sdv <- stats::sd(train_df[[cn]])
  if (!is.finite(sdv) || sdv < 1e-8) sdv <- 1.0
  train_df[[cn]] <- (train_df[[cn]] - mu) / sdv
  test_df[[cn]] <- (test_df[[cn]] - mu) / sdv
}

# Choose gamlss family based on dataset family.
if (family_name == "gaussian") {
  gamlss_family <- NO()
} else {
  gamlss_family <- BI()
}
# Keep scale model aligned with Rust GAMLSS benchmark path.

t0 <- proc.time()[["elapsed"]]
fit_formula <- mu_formula
fit <- tryCatch(
  gamlss(
    as.formula(fit_formula),
    sigma.formula = sigma_formula,
    family = gamlss_family,
    data = train_df,
    control = gamlss.control(n.cyc=200, trace=FALSE)
  ),
  error = function(e) e
)
fit_sec <- proc.time()[["elapsed"]] - t0

if (inherits(fit, "error")) {
  out <- list(
    status="failed",
    error=paste0("r_gamlss fit failed: ", conditionMessage(fit))
  )
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}

pred_t0 <- proc.time()[["elapsed"]]
p <- tryCatch(
  as.numeric(predict(fit, newdata=test_df, what="mu", type="response")),
  error = function(e) e
)
pred_sec <- proc.time()[["elapsed"]] - pred_t0

if (inherits(p, "error")) {
  out <- list(
    status="failed",
    error=paste0("r_gamlss predict failed: ", conditionMessage(p))
  )
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}

if (family_name == "gaussian") {
  # Gaussian metrics with per-obs sigma from the sigma sub-model.
  sigma_hat <- tryCatch(
    pmax(as.numeric(predict(fit, newdata=test_df, what="sigma", type="response")), 1e-12),
    error = function(e) e
  )
  if (inherits(sigma_hat, "error")) {
    out <- list(status="failed", error=paste0("r_gamlss sigma predict failed: ", conditionMessage(sigma_hat)))
    write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
    quit(save="no")
  }
  if (length(sigma_hat) != length(y_test)) {
    out <- list(
      status="failed",
      error=paste0("r_gamlss sigma length mismatch (got ", length(sigma_hat), ", expected ", length(y_test), ")")
    )
    write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
    quit(save="no")
  }
  if (any(!is.finite(sigma_hat) | sigma_hat <= 0)) {
    out <- list(status="failed", error="r_gamlss sigma has non-finite or non-positive values")
    write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
    quit(save="no")
  }
  rmse <- sqrt(mean((y_test - p)^2))
  mae <- mean(abs(y_test - p))
  sst <- sum((y_test - mean(y_test))^2)
  r2 <- if (sst <= 0) 0.0 else 1.0 - sum((y_test - p)^2) / sst
  logloss <- mean(0.5 * log(2 * pi * sigma_hat^2) + ((y_test - p)^2) / (2 * sigma_hat^2))
  out <- list(
    status="ok",
    fit_sec=fit_sec,
    predict_sec=pred_sec,
    auc=NULL,
    brier=NULL,
    logloss=logloss,
    nagelkerke_r2=NULL,
    rmse=rmse,
    mae=mae,
    r2=r2,
    model_spec=paste0("gamlss(NO; sigma.formula=", deparse(sigma_formula), "): ", fit_formula)
  )
} else {
  # Binomial metrics (original behavior).
  p_safe <- pmin(pmax(p, 1e-12), 1 - 1e-12)
  ord <- order(p)
  yy <- y_test[ord]
  n_pos <- sum(yy > 0.5)
  n_neg <- sum(yy <= 0.5)
  if (n_pos == 0 || n_neg == 0) {
    auc <- 0.5
  } else {
    ranks <- seq_along(yy)
    rank_sum_pos <- sum(ranks[yy > 0.5])
    auc <- (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
  }
  brier <- mean((y_test - p)^2)
  logloss <- mean(-(y_test * log(p_safe) + (1 - y_test) * log(1 - p_safe)))
  p_mean <- mean(y_test)
  if (is.finite(p_mean) && p_mean > 0 && p_mean < 1) {
    ll_null <- sum(y_test * log(p_mean) + (1 - y_test) * log(1 - p_mean))
    ll_model <- sum(y_test * log(p_safe) + (1 - y_test) * log(1 - p_safe))
    n_obs <- length(y_test)
    r2_cs <- 1 - exp((2 / n_obs) * (ll_null - ll_model))
    max_r2_cs <- 1 - exp((2 / n_obs) * ll_null)
    nagelkerke_r2 <- if (max_r2_cs > 0) r2_cs / max_r2_cs else NULL
  } else {
    nagelkerke_r2 <- NULL
  }
  out <- list(
    status="ok",
    fit_sec=fit_sec,
    predict_sec=pred_sec,
    auc=auc,
    brier=brier,
    logloss=logloss,
    nagelkerke_r2=nagelkerke_r2,
    rmse=NULL,
    mae=NULL,
    r2=NULL,
    model_spec=paste0("gamlss(BI; sigma.formula=", deparse(sigma_formula), "): ", fit_formula)
  )
}
write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
'''
        script_path.write_text(script)

        cv_rows = []
        for fold_id, fold in enumerate(folds):
            train_idx_path = td_path / f"train_idx_{fold_id}.txt"
            test_idx_path = td_path / f"test_idx_{fold_id}.txt"
            train_idx_path.write_text("\n".join(str(int(i)) for i in fold.train_idx) + "\n")
            test_idx_path.write_text("\n".join(str(int(i)) for i in fold.test_idx) + "\n")

            code, out, err = run_cmd(
                [
                    "Rscript",
                    str(script_path),
                    str(data_path),
                    str(out_path),
                    str(train_idx_path),
                    str(test_idx_path),
                ],
                cwd=ROOT,
            )
            if code != 0:
                return {
                    "contender": "r_gamlss",
                    "scenario_name": scenario_name,
                    "status": "failed",
                    "fold_id": int(fold_id),
                    "n_train": int(len(fold.train_idx)),
                    "n_test": int(len(fold.test_idx)),
                    "n_folds": int(len(folds)),
                    "error": err.strip() or out.strip(),
                }
            fold_row = json.loads(out_path.read_text())
            if fold_row.get("status") != "ok":
                return {
                    "contender": "r_gamlss",
                    "scenario_name": scenario_name,
                    "status": "failed",
                    "fold_id": int(fold_id),
                    "n_train": int(len(fold.train_idx)),
                    "n_test": int(len(fold.test_idx)),
                    "n_folds": int(len(folds)),
                    "error": str(fold_row.get("error", "r_gamlss fold failed")),
                }
            fold_row["n_test"] = int(len(fold.test_idx))
            cv_rows.append(fold_row)

    metrics = aggregate_cv_rows(cv_rows, family)
    return {
        "contender": "r_gamlss",
        "family": family,
        "scenario_name": scenario_name,
        "status": "ok",
        **metrics,
        "model_spec": f"{cv_rows[0]['model_spec']} {_evaluation_suffix(folds)}",
    }


def run_external_mgcv_cv(scenario, *, ds: dict | None = None, folds: list[Fold] | None = None):
    if ds is None:
        ds = dataset_for_scenario(scenario)
    if folds is None:
        folds = folds_for_dataset(ds)
    contender_name = "r_survival_coxph" if ds["family"] == "survival" else "r_mgcv"
    mgcv_formula = None
    rust_cfg = _rust_fit_mapping(scenario["name"])
    use_select = bool((rust_cfg or {}).get("double_penalty", True))
    if ds["family"] != "survival":
        mgcv_formula = _mgcv_formula_for_scenario(scenario["name"], ds)
        if not mgcv_formula:
            raise RuntimeError(
                f"Missing required shared mgcv formula for non-survival scenario '{scenario['name']}'"
            )

    with tempfile.TemporaryDirectory(prefix="gam_bench_mgcv_cv_") as td:
        td_path = Path(td)
        data_path = td_path / "data.json"
        out_path = td_path / "out.json"
        script_path = td_path / "run_mgcv_cv.R"

        payload = {
            "dataset": ds,
            "scenario_name": scenario["name"],
            "mgcv_formula": mgcv_formula,
            "use_select": use_select,
        }
        data_path.write_text(json.dumps(payload))

        script = r'''
args <- commandArgs(trailingOnly = TRUE)
data_path <- args[1]
out_path <- args[2]
train_idx_path <- args[3]
test_idx_path <- args[4]

suppressPackageStartupMessages({
  library(mgcv)
  library(jsonlite)
})

payload <- fromJSON(data_path, simplifyVector = TRUE)
df <- as.data.frame(payload$dataset$rows)
target_name <- as.character(payload$dataset$target)
family_name <- as.character(payload$dataset$family)
if (family_name != "survival") {
  if (!nzchar(target_name) || !(target_name %in% colnames(df))) {
    stop(sprintf("invalid or missing dataset target column: %s", target_name))
  }
}
scenario_name <- as.character(payload$scenario_name)
mgcv_formula <- NULL
if (!is.null(payload$mgcv_formula)) {
  mgcv_formula <- as.character(payload$mgcv_formula)
}
use_select <- TRUE
if (!is.null(payload$use_select)) {
  use_select <- isTRUE(payload$use_select)
}

train_idx <- scan(train_idx_path, what=integer(), quiet=TRUE) + 1L
test_idx <- scan(test_idx_path, what=integer(), quiet=TRUE) + 1L

train_df <- df[train_idx, , drop=FALSE]
test_df <- df[test_idx, , drop=FALSE]
y_test <- NULL
y_train <- NULL
if (family_name != "survival") {
  y_all <- as.numeric(df[[target_name]])
  y_train <- y_all[train_idx]
  y_test <- y_all[test_idx]
}

fam <- if (family_name == "binomial") binomial(link="logit") else gaussian(link="identity")

if (family_name != "survival") {
  feature_cols <- as.character(payload$dataset$features)
  for (cn in feature_cols) {
    mu <- mean(train_df[[cn]])
    sdv <- stats::sd(train_df[[cn]])
    if (!is.finite(sdv) || sdv < 1e-8) sdv <- 1.0
    train_df[[cn]] <- (train_df[[cn]] - mu) / sdv
    test_df[[cn]] <- (test_df[[cn]] - mu) / sdv
  }
}

if (family_name == "survival") {
  suppressPackageStartupMessages(library(survival))
  feature_cols <- as.character(payload$dataset$features)
  for (cn in feature_cols) {
    mu <- mean(train_df[[cn]])
    sdv <- stats::sd(train_df[[cn]])
    if (!is.finite(sdv) || sdv < 1e-8) sdv <- 1.0
    train_df[[cn]] <- (train_df[[cn]] - mu) / sdv
    test_df[[cn]] <- (test_df[[cn]] - mu) / sdv
  }
  if (scenario_name == "icu_survival_death") {
    if ("bmi" %in% colnames(train_df)) {
      qs <- seq(0.1, 0.9, length.out=5)
      knots <- unique(as.numeric(stats::quantile(train_df[["bmi"]], probs=qs, names=FALSE, type=7)))
      train_df[["bmi_spline_0"]] <- train_df[["bmi"]]
      test_df[["bmi_spline_0"]] <- test_df[["bmi"]]
      bmi_cols <- c("bmi_spline_0")
      if (length(knots) > 0) {
        for (j in seq_along(knots)) {
          cn <- sprintf("bmi_spline_%d", j)
          train_df[[cn]] <- pmax(0.0, train_df[["bmi"]] - as.numeric(knots[[j]]))
          test_df[[cn]] <- pmax(0.0, test_df[["bmi"]] - as.numeric(knots[[j]]))
          bmi_cols <- c(bmi_cols, cn)
        }
      }
      train_df[["bmi"]] <- NULL
      test_df[["bmi"]] <- NULL
      ftxt <- paste(
        "Surv(time, event) ~ age + hr_max + sysbp_min +",
        paste(bmi_cols, collapse=" + ")
      )
    } else {
      ftxt <- "Surv(time, event) ~ age + hr_max + sysbp_min"
    }
  } else if (scenario_name == "heart_failure_survival") {
    ftxt <- "Surv(time, event) ~ age + anaemia + log_creatinine_phosphokinase + diabetes + ejection_fraction + high_blood_pressure + log_platelets + log_serum_creatinine + serum_sodium + sex + smoking"
  } else if (scenario_name == "cirrhosis_survival") {
    ftxt <- "Surv(time, event) ~ drug + sex_male + ascites + hepatomegaly + spiders + edema + age + bilirubin + cholesterol + albumin + copper + alk_phos + sgot + tryglicerides + platelets + prothrombin + stage"
  } else {
    ftxt <- "Surv(time, event) ~ age + bmi + hr_max + sysbp_min + temp_apache"
  }
  t0 <- proc.time()[["elapsed"]]
  fit <- coxph(as.formula(ftxt), data=train_df, ties="efron")
  fit_sec <- proc.time()[["elapsed"]] - t0

  pred_t0 <- proc.time()[["elapsed"]]
  lp <- as.numeric(predict(fit, newdata=test_df, type="lp"))
  pred_sec <- proc.time()[["elapsed"]] - pred_t0

  risk <- as.numeric(lp)
  out <- list(
    status="ok",
    fit_sec=fit_sec,
    predict_sec=pred_sec,
    auc=NULL,
    risk=risk,
    brier=NULL,
    logloss=NULL,
    rmse=NULL,
    mae=NULL,
    r2=NULL,
    model_spec=ftxt
  )
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}

if (is.null(mgcv_formula) || !nzchar(mgcv_formula)) {
  stop(sprintf("missing shared mgcv formula for scenario: %s", scenario_name))
}
ftxt <- mgcv_formula

t0 <- proc.time()[["elapsed"]]
fit <- gam(as.formula(ftxt), family=fam, data=train_df, method="REML", select=use_select)
fit_sec <- proc.time()[["elapsed"]] - t0

pred_t0 <- proc.time()[["elapsed"]]
p <- as.numeric(predict(fit, newdata=test_df, type="response"))
pred_sec <- proc.time()[["elapsed"]] - pred_t0

if (family_name == "binomial") {
  ord <- order(p)
  yy <- y_test[ord]
  n_pos <- sum(yy > 0.5)
  n_neg <- sum(yy <= 0.5)
  if (n_pos == 0 || n_neg == 0) {
    auc <- 0.5
  } else {
    ranks <- seq_along(yy)
    rank_sum_pos <- sum(ranks[yy > 0.5])
    auc <- (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
  }
  brier <- mean((y_test - p)^2)
  p_safe <- pmin(pmax(p, 1e-12), 1 - 1e-12)
  logloss <- mean(-(y_test * log(p_safe) + (1 - y_test) * log(1 - p_safe)))
  p_mean <- mean(y_test)
  if (is.finite(p_mean) && p_mean > 0 && p_mean < 1) {
    ll_null <- sum(y_test * log(p_mean) + (1 - y_test) * log(1 - p_mean))
    ll_model <- sum(y_test * log(p_safe) + (1 - y_test) * log(1 - p_safe))
    n_obs <- length(y_test)
    r2_cs <- 1 - exp((2 / n_obs) * (ll_null - ll_model))
    max_r2_cs <- 1 - exp((2 / n_obs) * ll_null)
    nagelkerke_r2 <- if (max_r2_cs > 0) r2_cs / max_r2_cs else NULL
  } else {
    nagelkerke_r2 <- NULL
  }
  out <- list(
    status="ok",
    fit_sec=fit_sec,
    predict_sec=pred_sec,
    auc=auc,
    brier=brier,
    logloss=logloss,
    nagelkerke_r2=nagelkerke_r2,
    rmse=NULL,
    mae=NULL,
    r2=NULL,
    model_spec=ftxt
  )
} else {
  sigma_hat <- NA_real_
  fit_scale <- tryCatch(as.numeric(summary(fit)$scale), error=function(e) NA_real_)
  if (is.finite(fit_scale) && fit_scale > 0) {
    sigma_hat <- sqrt(fit_scale)
  } else {
    p_train <- as.numeric(predict(fit, newdata=train_df, type="response"))
    sigma_hat <- sqrt(mean((y_train - p_train)^2))
  }
  sigma_hat <- max(as.numeric(sigma_hat), 1e-12)
  rmse <- sqrt(mean((y_test - p)^2))
  mae <- mean(abs(y_test - p))
  sst <- sum((y_test - mean(y_test))^2)
  if (sst <= 0) {
    r2 <- 0.0
  } else {
    r2 <- 1.0 - sum((y_test - p)^2) / sst
  }
  logloss <- mean(0.5 * log(2 * pi * sigma_hat^2) + ((y_test - p)^2) / (2 * sigma_hat^2))
  out <- list(
    status="ok",
    fit_sec=fit_sec,
    predict_sec=pred_sec,
    auc=NULL,
    brier=NULL,
    logloss=logloss,
    rmse=rmse,
    mae=mae,
    r2=r2,
    model_spec=ftxt
  )
}

write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
'''
        script_path.write_text(script)

        cv_rows = []
        all_df = pd.DataFrame(ds["rows"])
        for fold_id, fold in enumerate(folds):
            train_idx_path = td_path / f"train_idx_{fold_id}.txt"
            test_idx_path = td_path / f"test_idx_{fold_id}.txt"
            train_idx_path.write_text("\n".join(str(int(i)) for i in fold.train_idx) + "\n")
            test_idx_path.write_text("\n".join(str(int(i)) for i in fold.test_idx) + "\n")

            code, out, err = run_cmd(
                [
                    "Rscript",
                    str(script_path),
                    str(data_path),
                    str(out_path),
                    str(train_idx_path),
                    str(test_idx_path),
                ],
                cwd=ROOT,
            )
            if code != 0:
                return {
                    "contender": contender_name,
                    "scenario_name": scenario["name"],
                    "status": "failed",
                    "error": err.strip() or out.strip(),
                }
            fold_row = json.loads(out_path.read_text())
            if ds["family"] == "survival":
                all_df = pd.DataFrame(ds["rows"])
                train_df = all_df.iloc[fold.train_idx].copy()
                test_df = all_df.iloc[fold.test_idx].copy()
                event_times = test_df[ds["time_col"]].to_numpy(dtype=float)
                events = test_df[ds["event_col"]].to_numpy(dtype=float)
                risk = np.asarray(fold_row.get("risk", []), dtype=float).reshape(-1)
                if risk.shape[0] != event_times.shape[0]:
                    return {
                        "contender": contender_name,
                        "scenario_name": scenario["name"],
                        "status": "failed",
                        "error": (
                            "r_survival_coxph fold output missing/invalid risk vector "
                            f"(got {risk.shape[0]}, expected {event_times.shape[0]})"
                        ),
                    }
                fold_row["auc"] = _lifelines_cindex_from_risk(event_times, risk, events)
                fold_row["brier"] = survival_partial_brier_score(event_times, risk, events)
                fold_row["logloss"] = survival_partial_log_loss(event_times, risk, events)
                fold_row["nagelkerke_r2"] = survival_partial_nagelkerke_r2(event_times, risk, events)
                fold_row.pop("risk", None)
            fold_row["n_test"] = int(len(fold.test_idx))
            cv_rows.append(fold_row)

    metrics = aggregate_cv_rows(cv_rows, ds["family"])
    return {
        "contender": contender_name,
        "family": ds["family"],
        "scenario_name": scenario["name"],
        "status": "ok",
        **metrics,
        "model_spec": f"{cv_rows[0]['model_spec']} {_evaluation_suffix(folds)}",
    }


def run_external_mgcv_gaulss_cv(scenario, *, ds: dict | None = None, folds: list[Fold] | None = None):
    if ds is None:
        ds = dataset_for_scenario(scenario)
    if ds["family"] != "gaussian":
        return None
    if folds is None:
        folds = folds_for_dataset(ds)
    mu_formula = _mgcv_formula_for_scenario(scenario["name"], ds)
    rust_cfg = _rust_fit_mapping(scenario["name"])
    use_select = bool((rust_cfg or {}).get("double_penalty", True))

    with tempfile.TemporaryDirectory(prefix="gam_bench_mgcv_gaulss_cv_") as td:
        td_path = Path(td)
        data_path = td_path / "data.json"
        out_path = td_path / "out.json"
        script_path = td_path / "run_mgcv_gaulss_cv.R"

        payload = {
            "dataset": ds,
            "scenario_name": scenario["name"],
            "mu_formula": mu_formula,
            "sigma_rhs": _sigma_feature_rhs(ds),
            "use_select": use_select,
        }
        data_path.write_text(json.dumps(payload))

        script = r'''
args <- commandArgs(trailingOnly = TRUE)
data_path <- args[1]
out_path <- args[2]
train_idx_path <- args[3]
test_idx_path <- args[4]

suppressPackageStartupMessages({
  library(mgcv)
  library(jsonlite)
})

payload <- fromJSON(data_path, simplifyVector = TRUE)
df <- as.data.frame(payload$dataset$rows)
target_name <- as.character(payload$dataset$target)
mu_formula <- as.character(payload$mu_formula)
sigma_rhs <- as.character(payload$sigma_rhs)
use_select <- TRUE
if (!is.null(payload$use_select)) {
  use_select <- isTRUE(payload$use_select)
}

train_idx <- scan(train_idx_path, what=integer(), quiet=TRUE) + 1L
test_idx <- scan(test_idx_path, what=integer(), quiet=TRUE) + 1L

train_df <- df[train_idx, , drop=FALSE]
test_df <- df[test_idx, , drop=FALSE]
y_test <- as.numeric(df[[target_name]])[test_idx]

feature_cols <- as.character(payload$dataset$features)
for (cn in feature_cols) {
  mu <- mean(train_df[[cn]])
  sdv <- stats::sd(train_df[[cn]])
  if (!is.finite(sdv) || sdv < 1e-8) sdv <- 1.0
  train_df[[cn]] <- (train_df[[cn]] - mu) / sdv
  test_df[[cn]] <- (test_df[[cn]] - mu) / sdv
}

rhs_parts <- strsplit(mu_formula, "~", fixed=TRUE)[[1]]
if (length(rhs_parts) < 2) {
  out <- list(status="failed", error=paste0("invalid mu formula: ", mu_formula))
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}
mu_rhs <- trimws(rhs_parts[[2]])

t0 <- proc.time()[["elapsed"]]
fit <- tryCatch(
  gam(
    list(as.formula(mu_formula), as.formula(paste("~", sigma_rhs))),
    family=gaulss(),
    data=train_df,
    method="REML",
    select=use_select
  ),
  error=function(e) e
)
fit_sec <- proc.time()[["elapsed"]] - t0

if (inherits(fit, "error")) {
  out <- list(status="failed", error=paste0("r_mgcv_gaulss fit failed: ", conditionMessage(fit)))
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}

pred_t0 <- proc.time()[["elapsed"]]
pred <- tryCatch(as.matrix(predict(fit, newdata=test_df, type="response")), error=function(e) e)
pred_sec <- proc.time()[["elapsed"]] - pred_t0
if (inherits(pred, "error")) {
  out <- list(status="failed", error=paste0("r_mgcv_gaulss predict failed: ", conditionMessage(pred)))
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}

if (ncol(pred) < 1) {
  out <- list(status="failed", error="r_mgcv_gaulss predict returned empty matrix")
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}
p <- as.numeric(pred[,1])
if (ncol(pred) < 2) {
  out <- list(status="failed", error="r_mgcv_gaulss predict output missing sigma column")
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}
sigma_hat <- as.numeric(pred[,2])
if (length(sigma_hat) != length(y_test)) {
  out <- list(
    status="failed",
    error=paste0("r_mgcv_gaulss sigma length mismatch (got ", length(sigma_hat), ", expected ", length(y_test), ")")
  )
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}
if (any(!is.finite(sigma_hat) | sigma_hat <= 0)) {
  out <- list(status="failed", error="r_mgcv_gaulss sigma has non-finite or non-positive values")
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}
rmse <- sqrt(mean((y_test - p)^2))
mae <- mean(abs(y_test - p))
sst <- sum((y_test - mean(y_test))^2)
r2 <- if (sst <= 0) 0.0 else 1.0 - sum((y_test - p)^2) / sst
logloss <- mean(0.5 * log(2 * pi * sigma_hat^2) + ((y_test - p)^2) / (2 * sigma_hat^2))

out <- list(
  status="ok",
  fit_sec=fit_sec,
  predict_sec=pred_sec,
  auc=NULL,
  brier=NULL,
  logloss=logloss,
  rmse=rmse,
  mae=mae,
  r2=r2,
  model_spec=paste0("gam(list(", target_name, " ~ ", mu_rhs, ", ~ ", sigma_rhs, "), family=gaulss())")
)
write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
'''
        script_path.write_text(script)

        cv_rows = []
        for fold_id, fold in enumerate(folds):
            train_idx_path = td_path / f"train_idx_{fold_id}.txt"
            test_idx_path = td_path / f"test_idx_{fold_id}.txt"
            train_idx_path.write_text("\n".join(str(int(i)) for i in fold.train_idx) + "\n")
            test_idx_path.write_text("\n".join(str(int(i)) for i in fold.test_idx) + "\n")

            code, out, err = run_cmd(
                [
                    "Rscript",
                    str(script_path),
                    str(data_path),
                    str(out_path),
                    str(train_idx_path),
                    str(test_idx_path),
                ],
                cwd=ROOT,
            )
            if code != 0:
                return {
                    "contender": "r_mgcv_gaulss",
                    "scenario_name": scenario["name"],
                    "status": "failed",
                    "fold_id": int(fold_id),
                    "n_train": int(len(fold.train_idx)),
                    "n_test": int(len(fold.test_idx)),
                    "n_folds": int(len(folds)),
                    "error": err.strip() or out.strip(),
                }
            fold_row = json.loads(out_path.read_text())
            if fold_row.get("status") != "ok":
                return {
                    "contender": "r_mgcv_gaulss",
                    "scenario_name": scenario["name"],
                    "status": "failed",
                    "fold_id": int(fold_id),
                    "n_train": int(len(fold.train_idx)),
                    "n_test": int(len(fold.test_idx)),
                    "n_folds": int(len(folds)),
                    "error": str(fold_row.get("error", "r_mgcv_gaulss fold failed")),
                }
            fold_row["n_test"] = int(len(fold.test_idx))
            cv_rows.append(fold_row)

    metrics = aggregate_cv_rows(cv_rows, ds["family"])
    return {
        "contender": "r_mgcv_gaulss",
        "family": ds["family"],
        "scenario_name": scenario["name"],
        "status": "ok",
        **metrics,
        "model_spec": f"{cv_rows[0]['model_spec']} {_evaluation_suffix(folds)}",
    }


def _gamboostlss_formulas_for_scenario(scenario_name: str, ds):
    cfg = _rust_fit_mapping(scenario_name)
    if cfg is None:
        return None, None
    if ds["family"] != "gaussian":
        return None, None

    knot_count = max(4, int(cfg.get("knots", 8)))
    mu_terms = [f"bols({c}, intercept=FALSE)" for c in cfg.get("linear_cols", [])]

    smooth_cols = cfg.get("smooth_cols")
    if smooth_cols:
        for col in smooth_cols:
            mu_terms.append(f"bbs({col}, df=min({knot_count}, nrow(train_df)-1))")
    elif cfg.get("smooth_col"):
        col = cfg["smooth_col"]
        mu_terms.append(f"bbs({col}, df=min({knot_count}, nrow(train_df)-1))")

    if not mu_terms:
        mu_terms = ["1"]
    mu_formula = f"{ds['target']} ~ " + " + ".join(mu_terms)

    sigma_terms = [f"bols({c}, intercept=FALSE)" for c in ds["features"]]
    sigma_formula = "~ " + (" + ".join(sigma_terms) if sigma_terms else "1")
    return mu_formula, sigma_formula


def run_external_r_gamboostlss_cv(scenario, *, ds: dict | None = None, folds: list[Fold] | None = None):
    if ds is None:
        ds = dataset_for_scenario(scenario)
    if ds["family"] != "gaussian":
        return None
    scenario_name = scenario["name"]
    if folds is None:
        folds = folds_for_dataset(ds)
    mu_formula, sigma_formula = _gamboostlss_formulas_for_scenario(scenario_name, ds)
    if not mu_formula:
        return None

    with tempfile.TemporaryDirectory(prefix="gam_bench_r_gamboostlss_cv_") as td:
        td_path = Path(td)
        data_path = td_path / "data.json"
        out_path = td_path / "out.json"
        script_path = td_path / "run_gamboostlss_cv.R"

        payload = {
            "dataset": ds,
            "scenario_name": scenario_name,
            "mu_formula": mu_formula,
            "sigma_formula": sigma_formula,
        }
        data_path.write_text(json.dumps(payload))

        script = r'''
args <- commandArgs(trailingOnly = TRUE)
data_path <- args[1]
out_path <- args[2]
train_idx_path <- args[3]
test_idx_path <- args[4]

suppressPackageStartupMessages({
  library(gamboostLSS)
  library(mboost)
  library(jsonlite)
})

payload <- fromJSON(data_path, simplifyVector = TRUE)
df <- as.data.frame(payload$dataset$rows)
target_name <- as.character(payload$dataset$target)
mu_formula <- as.character(payload$mu_formula)
sigma_formula <- as.character(payload$sigma_formula)

train_idx <- scan(train_idx_path, what=integer(), quiet=TRUE) + 1L
test_idx <- scan(test_idx_path, what=integer(), quiet=TRUE) + 1L

train_df <- df[train_idx, , drop=FALSE]
test_df <- df[test_idx, , drop=FALSE]
y_test <- as.numeric(df[[target_name]])[test_idx]

feature_cols <- as.character(payload$dataset$features)
for (cn in feature_cols) {
  mu <- mean(train_df[[cn]])
  sdv <- stats::sd(train_df[[cn]])
  if (!is.finite(sdv) || sdv < 1e-8) sdv <- 1.0
  train_df[[cn]] <- (train_df[[cn]] - mu) / sdv
  test_df[[cn]] <- (test_df[[cn]] - mu) / sdv
}

t0 <- proc.time()[["elapsed"]]
fit <- tryCatch({
  fit_full <- gamboostLSS(
    formula=list(mu=as.formula(mu_formula), sigma=as.formula(sigma_formula)),
    data=train_df,
    families=GaussianLSS(),
    control=boost_control(mstop=600)
  )
  aic_obj <- tryCatch(
    AIC(fit_full, method="corrected"),
    error=function(e) e
  )
  if (inherits(aic_obj, "error")) {
    stop(paste0("AIC selection failed: ", conditionMessage(aic_obj)))
  } else {
    selected_mstop <- as.integer(mstop(aic_obj))
    if (!is.finite(selected_mstop) || selected_mstop < 1) {
      stop(paste0("AIC selection returned invalid mstop: ", as.character(selected_mstop)))
    }
  }
  fit_final <- fit_full[selected_mstop]
  attr(fit_final, "selected_mstop") <- selected_mstop
  fit_final
},
error=function(e) e
)
fit_sec <- proc.time()[["elapsed"]] - t0

if (inherits(fit, "error")) {
  out <- list(status="failed", error=paste0("r_gamboostlss fit failed: ", conditionMessage(fit)))
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}

pred_t0 <- proc.time()[["elapsed"]]
p <- tryCatch(
  as.numeric(predict(fit, newdata=test_df, parameter="mu", type="response")),
  error=function(e) e
)
pred_sec <- proc.time()[["elapsed"]] - pred_t0

if (inherits(p, "error")) {
  out <- list(status="failed", error=paste0("r_gamboostlss predict failed: ", conditionMessage(p)))
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}
if (length(p) != length(y_test)) {
  out <- list(
    status="failed",
    error=paste0("r_gamboostlss mu length mismatch (got ", length(p), ", expected ", length(y_test), ")")
  )
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}
if (any(!is.finite(p))) {
  out <- list(status="failed", error="r_gamboostlss mu has non-finite values")
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}
sigma_pred <- tryCatch(
  as.numeric(predict(fit, newdata=test_df, parameter="sigma", type="response")),
  error=function(e) e
)
if (inherits(sigma_pred, "error")) {
  out <- list(status="failed", error=paste0("r_gamboostlss sigma predict failed: ", conditionMessage(sigma_pred)))
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}
if (length(sigma_pred) != length(y_test)) {
  out <- list(
    status="failed",
    error=paste0("r_gamboostlss sigma length mismatch (got ", length(sigma_pred), ", expected ", length(y_test), ")")
  )
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}
if (any(!is.finite(sigma_pred) | sigma_pred <= 0)) {
  out <- list(status="failed", error="r_gamboostlss sigma has non-finite or non-positive values")
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}

rmse <- sqrt(mean((y_test - p)^2))
mae <- mean(abs(y_test - p))
sst <- sum((y_test - mean(y_test))^2)
r2 <- if (sst <= 0) 0.0 else 1.0 - sum((y_test - p)^2) / sst
logloss <- mean(0.5 * log(2 * pi * sigma_pred^2) + ((y_test - p)^2) / (2 * sigma_pred^2))

out <- list(
  status="ok",
  fit_sec=fit_sec,
  predict_sec=pred_sec,
  auc=NULL,
  brier=NULL,
  logloss=logloss,
  rmse=rmse,
  mae=mae,
  r2=r2,
  model_spec=paste0(
    "gamboostLSS(GaussianLSS; AIC-selected mstop=",
    as.integer(attr(fit, "selected_mstop")),
    "): ",
    mu_formula,
    " ; sigma ",
    sigma_formula
  )
)
write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
'''
        script_path.write_text(script)

        cv_rows = []
        for fold_id, fold in enumerate(folds):
            train_idx_path = td_path / f"train_idx_{fold_id}.txt"
            test_idx_path = td_path / f"test_idx_{fold_id}.txt"
            train_idx_path.write_text("\n".join(str(int(i)) for i in fold.train_idx) + "\n")
            test_idx_path.write_text("\n".join(str(int(i)) for i in fold.test_idx) + "\n")

            code, out, err = run_cmd(
                [
                    "Rscript",
                    str(script_path),
                    str(data_path),
                    str(out_path),
                    str(train_idx_path),
                    str(test_idx_path),
                ],
                cwd=ROOT,
            )
            if code != 0:
                return {
                    "contender": "r_gamboostlss",
                    "scenario_name": scenario["name"],
                    "status": "failed",
                    "fold_id": int(fold_id),
                    "n_train": int(len(fold.train_idx)),
                    "n_test": int(len(fold.test_idx)),
                    "n_folds": int(len(folds)),
                    "error": err.strip() or out.strip(),
                }
            fold_row = json.loads(out_path.read_text())
            if fold_row.get("status") != "ok":
                return {
                    "contender": "r_gamboostlss",
                    "scenario_name": scenario["name"],
                    "status": "failed",
                    "fold_id": int(fold_id),
                    "n_train": int(len(fold.train_idx)),
                    "n_test": int(len(fold.test_idx)),
                    "n_folds": int(len(folds)),
                    "error": str(fold_row.get("error", "r_gamboostlss fold failed")),
                }
            fold_row["n_test"] = int(len(fold.test_idx))
            cv_rows.append(fold_row)

    metrics = aggregate_cv_rows(cv_rows, ds["family"])
    return {
        "contender": "r_gamboostlss",
        "family": ds["family"],
        "scenario_name": scenario["name"],
        "status": "ok",
        **metrics,
        "model_spec": f"{cv_rows[0]['model_spec']} [5-fold CV]",
    }


def _bamlss_formulas_for_scenario(scenario_name: str, ds):
    if ds["family"] != "gaussian":
        return None, None
    mu_formula = _mgcv_formula_for_scenario(scenario_name, ds)
    if not mu_formula:
        return None, None
    sigma_formula = _sigma_feature_formula(ds)
    return mu_formula, sigma_formula


def run_external_r_bamlss_cv(scenario, *, ds: dict | None = None, folds: list[Fold] | None = None):
    if ds is None:
        ds = dataset_for_scenario(scenario)
    if ds["family"] != "gaussian":
        return None
    scenario_name = scenario["name"]
    if folds is None:
        folds = folds_for_dataset(ds)
    mu_formula, sigma_formula = _bamlss_formulas_for_scenario(scenario_name, ds)
    if not mu_formula:
        return None

    with tempfile.TemporaryDirectory(prefix="gam_bench_r_bamlss_cv_") as td:
        td_path = Path(td)
        data_path = td_path / "data.json"
        out_path = td_path / "out.json"
        script_path = td_path / "run_bamlss_cv.R"

        payload = {
            "dataset": ds,
            "scenario_name": scenario_name,
            "mu_formula": mu_formula,
            "sigma_formula": sigma_formula,
        }
        data_path.write_text(json.dumps(payload))

        script = r'''
args <- commandArgs(trailingOnly = TRUE)
data_path <- args[1]
out_path <- args[2]
train_idx_path <- args[3]
test_idx_path <- args[4]

suppressPackageStartupMessages({
  library(bamlss)
  library(jsonlite)
})

payload <- fromJSON(data_path, simplifyVector = TRUE)
df <- as.data.frame(payload$dataset$rows)
target_name <- as.character(payload$dataset$target)
mu_formula <- as.character(payload$mu_formula)
sigma_formula <- as.character(payload$sigma_formula)

train_idx <- scan(train_idx_path, what=integer(), quiet=TRUE) + 1L
test_idx <- scan(test_idx_path, what=integer(), quiet=TRUE) + 1L

train_df <- df[train_idx, , drop=FALSE]
test_df <- df[test_idx, , drop=FALSE]
y_test <- as.numeric(df[[target_name]])[test_idx]

feature_cols <- as.character(payload$dataset$features)
for (cn in feature_cols) {
  mu <- mean(train_df[[cn]])
  sdv <- stats::sd(train_df[[cn]])
  if (!is.finite(sdv) || sdv < 1e-8) sdv <- 1.0
  train_df[[cn]] <- (train_df[[cn]] - mu) / sdv
  test_df[[cn]] <- (test_df[[cn]] - mu) / sdv
}

t0 <- proc.time()[["elapsed"]]
fit <- tryCatch(
  bamlss(
    formula = list(mu = as.formula(mu_formula), sigma = as.formula(sigma_formula)),
    family = "gaussian",
    data = train_df,
    optimizer = TRUE,
    sampler = FALSE,
    verbose = FALSE
  ),
  error = function(e) e
)
fit_sec <- proc.time()[["elapsed"]] - t0

if (inherits(fit, "error")) {
  out <- list(status="failed", error=paste0("r_bamlss fit failed: ", conditionMessage(fit)))
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}

pred_t0 <- proc.time()[["elapsed"]]
p <- tryCatch(
  as.numeric(predict(fit, newdata=test_df, model="mu", type="response")),
  error=function(e) e
)
pred_sec <- proc.time()[["elapsed"]] - pred_t0

if (inherits(p, "error")) {
  out <- list(status="failed", error=paste0("r_bamlss predict failed: ", conditionMessage(p)))
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}
if (length(p) != length(y_test)) {
  out <- list(
    status="failed",
    error=paste0("r_bamlss mu length mismatch (got ", length(p), ", expected ", length(y_test), ")")
  )
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}
if (any(!is.finite(p))) {
  out <- list(status="failed", error="r_bamlss mu has non-finite values")
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}
sigma_pred <- tryCatch(
  as.numeric(predict(fit, newdata=test_df, model="sigma", type="response")),
  error=function(e) e
)
if (inherits(sigma_pred, "error")) {
  out <- list(status="failed", error=paste0("r_bamlss sigma predict failed: ", conditionMessage(sigma_pred)))
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}
if (length(sigma_pred) != length(y_test)) {
  out <- list(
    status="failed",
    error=paste0("r_bamlss sigma length mismatch (got ", length(sigma_pred), ", expected ", length(y_test), ")")
  )
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}
if (any(!is.finite(sigma_pred) | sigma_pred <= 0)) {
  out <- list(status="failed", error="r_bamlss sigma has non-finite or non-positive values")
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}

rmse <- sqrt(mean((y_test - p)^2))
mae <- mean(abs(y_test - p))
sst <- sum((y_test - mean(y_test))^2)
r2 <- if (sst <= 0) 0.0 else 1.0 - sum((y_test - p)^2) / sst
logloss <- mean(0.5 * log(2 * pi * sigma_pred^2) + ((y_test - p)^2) / (2 * sigma_pred^2))

out <- list(
  status="ok",
  fit_sec=fit_sec,
  predict_sec=pred_sec,
  auc=NULL,
  brier=NULL,
  logloss=logloss,
  rmse=rmse,
  mae=mae,
  r2=r2,
  model_spec=paste0("bamlss(gaussian; optimizer-only): ", mu_formula, " ; sigma ", sigma_formula)
)
write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
'''
        script_path.write_text(script)

        cv_rows = []
        for fold_id, fold in enumerate(folds):
            train_idx_path = td_path / f"train_idx_{fold_id}.txt"
            test_idx_path = td_path / f"test_idx_{fold_id}.txt"
            train_idx_path.write_text("\n".join(str(int(i)) for i in fold.train_idx) + "\n")
            test_idx_path.write_text("\n".join(str(int(i)) for i in fold.test_idx) + "\n")

            code, out, err = run_cmd(
                [
                    "Rscript",
                    str(script_path),
                    str(data_path),
                    str(out_path),
                    str(train_idx_path),
                    str(test_idx_path),
                ],
                cwd=ROOT,
            )
            if code != 0:
                return {
                    "contender": "r_bamlss",
                    "scenario_name": scenario["name"],
                    "status": "failed",
                    "fold_id": int(fold_id),
                    "n_train": int(len(fold.train_idx)),
                    "n_test": int(len(fold.test_idx)),
                    "n_folds": int(len(folds)),
                    "error": err.strip() or out.strip(),
                }
            fold_row = json.loads(out_path.read_text())
            if fold_row.get("status") != "ok":
                return {
                    "contender": "r_bamlss",
                    "scenario_name": scenario["name"],
                    "status": "failed",
                    "fold_id": int(fold_id),
                    "n_train": int(len(fold.train_idx)),
                    "n_test": int(len(fold.test_idx)),
                    "n_folds": int(len(folds)),
                    "error": str(fold_row.get("error", "r_bamlss fold failed")),
                }
            fold_row["n_test"] = int(len(fold.test_idx))
            cv_rows.append(fold_row)

    metrics = aggregate_cv_rows(cv_rows, ds["family"])
    return {
        "contender": "r_bamlss",
        "family": ds["family"],
        "scenario_name": scenario["name"],
        "status": "ok",
        **metrics,
        "model_spec": f"{cv_rows[0]['model_spec']} [5-fold CV]",
    }


def run_external_r_brms_cv(scenario, *, ds: dict | None = None, folds: list[Fold] | None = None):
    if ds is None:
        ds = dataset_for_scenario(scenario)
    if ds["family"] != "gaussian":
        return None
    scenario_name = scenario["name"]
    if folds is None:
        folds = folds_for_dataset(ds)
    mu_formula, sigma_formula = _bamlss_formulas_for_scenario(scenario_name, ds)
    if not mu_formula:
        return None

    with tempfile.TemporaryDirectory(prefix="gam_bench_r_brms_cv_") as td:
        td_path = Path(td)
        data_path = td_path / "data.json"
        out_path = td_path / "out.json"
        script_path = td_path / "run_brms_cv.R"

        payload = {
            "dataset": ds,
            "scenario_name": scenario_name,
            "mu_formula": mu_formula,
            "sigma_formula": sigma_formula,
        }
        data_path.write_text(json.dumps(payload))

        script = r'''
args <- commandArgs(trailingOnly = TRUE)
data_path <- args[1]
out_path <- args[2]
train_idx_path <- args[3]
test_idx_path <- args[4]

suppressPackageStartupMessages({
  library(brms)
  library(jsonlite)
})

payload <- fromJSON(data_path, simplifyVector = TRUE)
df <- as.data.frame(payload$dataset$rows)
target_name <- as.character(payload$dataset$target)
mu_formula <- as.character(payload$mu_formula)
sigma_formula <- as.character(payload$sigma_formula)

train_idx <- scan(train_idx_path, what=integer(), quiet=TRUE) + 1L
test_idx <- scan(test_idx_path, what=integer(), quiet=TRUE) + 1L

train_df <- df[train_idx, , drop=FALSE]
test_df <- df[test_idx, , drop=FALSE]
y_test <- as.numeric(df[[target_name]])[test_idx]

feature_cols <- as.character(payload$dataset$features)
for (cn in feature_cols) {
  mu <- mean(train_df[[cn]])
  sdv <- stats::sd(train_df[[cn]])
  if (!is.finite(sdv) || sdv < 1e-8) sdv <- 1.0
  train_df[[cn]] <- (train_df[[cn]] - mu) / sdv
  test_df[[cn]] <- (test_df[[cn]] - mu) / sdv
}

t0 <- proc.time()[["elapsed"]]
fit <- tryCatch(
  brm(
    formula = bf(as.formula(mu_formula), sigma = as.formula(sigma_formula)),
    data = train_df,
    family = gaussian(),
    chains = 4,
    iter = 2000,
    warmup = 1000,
    cores = min(4L, max(1L, parallel::detectCores(logical=FALSE))),
    seed = 123,
    control = list(adapt_delta = 0.95, max_treedepth = 12),
    refresh = 0,
    silent = 2
  ),
  error = function(e) e
)
fit_sec <- proc.time()[["elapsed"]] - t0

if (inherits(fit, "error")) {
  out <- list(status="failed", error=paste0("r_brms fit failed: ", conditionMessage(fit)))
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}

pred_t0 <- proc.time()[["elapsed"]]
p <- tryCatch(
  as.numeric(fitted(fit, newdata=test_df, summary=TRUE)[, "Estimate"]),
  error=function(e) e
)
pred_sec <- proc.time()[["elapsed"]] - pred_t0

if (inherits(p, "error")) {
  out <- list(status="failed", error=paste0("r_brms predict failed: ", conditionMessage(p)))
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}
sigma_pred <- tryCatch(
  as.numeric(fitted(fit, dpar="sigma", newdata=test_df, summary=TRUE)[, "Estimate"]),
  error=function(e) e
)
if (inherits(sigma_pred, "error")) {
  out <- list(status="failed", error=paste0("r_brms sigma extract failed: ", conditionMessage(sigma_pred)))
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}
if (length(sigma_pred) != length(y_test)) {
  out <- list(
    status="failed",
    error=paste0("r_brms sigma length mismatch (got ", length(sigma_pred), ", expected ", length(y_test), ")")
  )
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}
if (any(!is.finite(sigma_pred) | sigma_pred <= 0)) {
  out <- list(status="failed", error="r_brms sigma has non-finite or non-positive values")
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}

rmse <- sqrt(mean((y_test - p)^2))
mae <- mean(abs(y_test - p))
sst <- sum((y_test - mean(y_test))^2)
r2 <- if (sst <= 0) 0.0 else 1.0 - sum((y_test - p)^2) / sst
logloss <- mean(0.5 * log(2 * pi * sigma_pred^2) + ((y_test - p)^2) / (2 * sigma_pred^2))

out <- list(
  status="ok",
  fit_sec=fit_sec,
  predict_sec=pred_sec,
  auc=NULL,
  brier=NULL,
  logloss=logloss,
  rmse=rmse,
  mae=mae,
  r2=r2,
  model_spec=paste0("brms::brm(bf(", mu_formula, ", sigma ", sigma_formula, "); gaussian)")
)
write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
'''
        script_path.write_text(script)

        cv_rows = []
        for fold_id, fold in enumerate(folds):
            train_idx_path = td_path / f"train_idx_{fold_id}.txt"
            test_idx_path = td_path / f"test_idx_{fold_id}.txt"
            train_idx_path.write_text("\n".join(str(int(i)) for i in fold.train_idx) + "\n")
            test_idx_path.write_text("\n".join(str(int(i)) for i in fold.test_idx) + "\n")

            code, out, err = run_cmd(
                [
                    "Rscript",
                    str(script_path),
                    str(data_path),
                    str(out_path),
                    str(train_idx_path),
                    str(test_idx_path),
                ],
                cwd=ROOT,
            )
            if code != 0:
                return {
                    "contender": "r_brms",
                    "scenario_name": scenario["name"],
                    "status": "failed",
                    "fold_id": int(fold_id),
                    "n_train": int(len(fold.train_idx)),
                    "n_test": int(len(fold.test_idx)),
                    "n_folds": int(len(folds)),
                    "error": err.strip() or out.strip(),
                }
            fold_row = json.loads(out_path.read_text())
            if fold_row.get("status") != "ok":
                return {
                    "contender": "r_brms",
                    "scenario_name": scenario["name"],
                    "status": "failed",
                    "fold_id": int(fold_id),
                    "n_train": int(len(fold.train_idx)),
                    "n_test": int(len(fold.test_idx)),
                    "n_folds": int(len(folds)),
                    "error": str(fold_row.get("error", "r_brms fold failed")),
                }
            fold_row["n_test"] = int(len(fold.test_idx))
            cv_rows.append(fold_row)

    metrics = aggregate_cv_rows(cv_rows, ds["family"])
    return {
        "contender": "r_brms",
        "family": ds["family"],
        "scenario_name": scenario["name"],
        "status": "ok",
        **metrics,
        "model_spec": f"{cv_rows[0]['model_spec']} [5-fold CV]",
    }


def run_external_mgcv_survival_cv(scenario, *, ds: dict | None = None, folds: list[Fold] | None = None):
    if ds is None:
        ds = dataset_for_scenario(scenario)
    if ds["family"] != "survival":
        return None
    if folds is None:
        folds = folds_for_dataset(ds)

    with tempfile.TemporaryDirectory(prefix="gam_bench_mgcv_surv_cv_") as td:
        td_path = Path(td)
        data_path = td_path / "data.json"
        out_path = td_path / "out.json"
        script_path = td_path / "run_mgcv_survival_cv.R"

        payload = {
            "dataset": ds,
            "scenario_name": scenario["name"],
        }
        data_path.write_text(json.dumps(payload))

        script = r'''
args <- commandArgs(trailingOnly = TRUE)
data_path <- args[1]
out_path <- args[2]
train_idx_path <- args[3]
test_idx_path <- args[4]

suppressPackageStartupMessages({
  library(mgcv)
  library(survival)
  library(jsonlite)
})

payload <- fromJSON(data_path, simplifyVector = TRUE)
df <- as.data.frame(payload$dataset$rows)
scenario_name <- as.character(payload$scenario_name)
time_col <- as.character(payload$dataset$time_col)
event_col <- as.character(payload$dataset$event_col)

train_idx <- scan(train_idx_path, what=integer(), quiet=TRUE) + 1L
test_idx <- scan(test_idx_path, what=integer(), quiet=TRUE) + 1L

train_df <- df[train_idx, , drop=FALSE]
test_df <- df[test_idx, , drop=FALSE]

feature_cols <- as.character(payload$dataset$features)
for (cn in feature_cols) {
  mu <- mean(train_df[[cn]])
  sdv <- stats::sd(train_df[[cn]])
  if (!is.finite(sdv) || sdv < 1e-8) sdv <- 1.0
  train_df[[cn]] <- (train_df[[cn]] - mu) / sdv
  test_df[[cn]] <- (test_df[[cn]] - mu) / sdv
}

if (scenario_name == "icu_survival_death") {
  if ("bmi" %in% colnames(train_df)) {
    qs <- seq(0.1, 0.9, length.out=5)
    knots <- unique(as.numeric(stats::quantile(train_df[["bmi"]], probs=qs, names=FALSE, type=7)))
    train_df[["bmi_spline_0"]] <- train_df[["bmi"]]
    test_df[["bmi_spline_0"]] <- test_df[["bmi"]]
    bmi_cols <- c("bmi_spline_0")
    if (length(knots) > 0) {
      for (j in seq_along(knots)) {
        cn <- sprintf("bmi_spline_%d", j)
        train_df[[cn]] <- pmax(0.0, train_df[["bmi"]] - as.numeric(knots[[j]]))
        test_df[[cn]] <- pmax(0.0, test_df[["bmi"]] - as.numeric(knots[[j]]))
        bmi_cols <- c(bmi_cols, cn)
      }
    }
    train_df[["bmi"]] <- NULL
    test_df[["bmi"]] <- NULL
    rhs <- paste(
      "age + hr_max + sysbp_min +",
      paste(bmi_cols, collapse=" + ")
    )
  } else {
    rhs <- "age + hr_max + sysbp_min"
  }
} else if (scenario_name == "heart_failure_survival") {
  rhs <- "age + anaemia + log_creatinine_phosphokinase + diabetes + ejection_fraction + high_blood_pressure + log_platelets + log_serum_creatinine + serum_sodium + sex + smoking"
} else if (scenario_name == "cirrhosis_survival") {
  rhs <- "drug + sex_male + ascites + hepatomegaly + spiders + edema + age + bilirubin + cholesterol + albumin + copper + alk_phos + sgot + tryglicerides + platelets + prothrombin + stage"
} else if (scenario_name == "haberman_survival") {
  rhs <- "age + op_year + s(axil_nodes, bs='ps', k=min(12, nrow(train_df)-1))"
} else {
  rhs <- "age + bmi + hr_max + sysbp_min + temp_apache"
}
ftxt <- sprintf("%s ~ %s", time_col, rhs)

t0 <- proc.time()[["elapsed"]]
fit <- gam(
  as.formula(ftxt),
  family=cox.ph(),
  weights=as.numeric(train_df[[event_col]]),
  data=train_df,
  method="REML",
  select=TRUE
)
fit_sec <- proc.time()[["elapsed"]] - t0

pred_t0 <- proc.time()[["elapsed"]]
lp_train <- as.numeric(predict(fit, newdata=train_df, type="link"))
lp_test <- as.numeric(predict(fit, newdata=test_df, type="link"))
risk <- as.numeric(lp_test)
pred_sec <- proc.time()[["elapsed"]] - pred_t0

out <- list(
  status="ok",
  fit_sec=fit_sec,
  predict_sec=pred_sec,
  auc=NULL,
  risk=risk,
  brier=NULL,
  logloss=NULL,
  rmse=NULL,
  mae=NULL,
  r2=NULL,
  model_spec=ftxt
)
write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
'''
        script_path.write_text(script)

        cv_rows = []
        all_df = pd.DataFrame(ds["rows"])
        for fold_id, fold in enumerate(folds):
            train_idx_path = td_path / f"train_idx_{fold_id}.txt"
            test_idx_path = td_path / f"test_idx_{fold_id}.txt"
            train_idx_path.write_text("\n".join(str(int(i)) for i in fold.train_idx) + "\n")
            test_idx_path.write_text("\n".join(str(int(i)) for i in fold.test_idx) + "\n")

            code, out, err = run_cmd(
                [
                    "Rscript",
                    str(script_path),
                    str(data_path),
                    str(out_path),
                    str(train_idx_path),
                    str(test_idx_path),
                ],
                cwd=ROOT,
            )
            if code != 0:
                return {
                    "contender": "r_mgcv_coxph",
                    "scenario_name": scenario["name"],
                    "status": "failed",
                    "error": err.strip() or out.strip(),
                }

            fold_row = json.loads(out_path.read_text())
            train_df = all_df.iloc[fold.train_idx].copy()
            test_df = all_df.iloc[fold.test_idx].copy()
            event_times = test_df[ds["time_col"]].to_numpy(dtype=float)
            events = test_df[ds["event_col"]].to_numpy(dtype=float)
            risk = np.asarray(fold_row.get("risk", []), dtype=float).reshape(-1)
            if risk.shape[0] != event_times.shape[0]:
                return {
                    "contender": "r_mgcv_coxph",
                    "scenario_name": scenario["name"],
                    "status": "failed",
                    "error": (
                        "r_mgcv_coxph fold output missing/invalid risk vector "
                        f"(got {risk.shape[0]}, expected {event_times.shape[0]})"
                    ),
                }
            fold_row["auc"] = _lifelines_cindex_from_risk(event_times, risk, events)
            fold_row["brier"] = survival_partial_brier_score(event_times, risk, events)
            fold_row["logloss"] = survival_partial_log_loss(event_times, risk, events)
            fold_row["nagelkerke_r2"] = survival_partial_nagelkerke_r2(event_times, risk, events)
            fold_row.pop("risk", None)
            fold_row["n_test"] = int(len(fold.test_idx))
            cv_rows.append(fold_row)

    metrics = aggregate_cv_rows(cv_rows, ds["family"])
    return {
        "contender": "r_mgcv_coxph",
        "family": ds["family"],
        "scenario_name": scenario["name"],
        "status": "ok",
        **metrics,
        "model_spec": f"{cv_rows[0]['model_spec']} [5-fold CV]",
    }


def run_external_pygam_cv(scenario, *, ds: dict | None = None, folds: list[Fold] | None = None):
    if ds is None:
        ds = dataset_for_scenario(scenario)
    if folds is None:
        folds = folds_for_dataset(ds)
    rust_cfg = _rust_fit_mapping(scenario["name"])
    if rust_cfg is not None:
        basis = _canonical_smooth_basis(rust_cfg.get("smooth_basis", "ps"))
        # pyGAM should not run scenarios requiring unsupported spline families.
        if basis in {"thinplate", "duchon", "matern"}:
            return None
    df = pd.DataFrame(ds["rows"])
    x = df[ds["features"]].to_numpy(dtype=float) if ds["features"] else np.empty((len(df), 0))
    y = df[ds["target"]].to_numpy(dtype=float) if ds.get("target") else None

    cv_rows = []
    for fold in folds:
        x_train = x[fold.train_idx]
        x_test = x[fold.test_idx]
        y_train = y[fold.train_idx] if y is not None else None
        y_test = y[fold.test_idx] if y is not None else None

        if ds["family"] == "survival":
            time_col = ds["time_col"]
            event_col = ds["event_col"]
            train_df = df.iloc[fold.train_idx].copy()
            test_df = df.iloc[fold.test_idx].copy()
            feature_cols = ds["features"]
            for col in feature_cols:
                mu = float(train_df[col].mean())
                sdv = float(train_df[col].std())
                if (not np.isfinite(sdv)) or sdv < 1e-8:
                    sdv = 1.0
                train_df[col] = (train_df[col] - mu) / sdv
                test_df[col] = (test_df[col] - mu) / sdv
            fit_feature_cols = feature_cols
            if scenario["name"] == "icu_survival_death" and "bmi" in feature_cols:
                train_df, test_df, bmi_spline_cols = _augment_bmi_spline_linear_hinges(
                    train_df, test_df, n_knots=6
                )
                fit_feature_cols = [c for c in feature_cols if c != "bmi"] + bmi_spline_cols
            fit_start = datetime.now(timezone.utc)
            cph = CoxPHFitter(penalizer=1e-4)
            with warnings.catch_warnings(record=True) as wlist:
                warnings.simplefilter("always")
                cph.fit(
                    train_df[[*fit_feature_cols, time_col, event_col]],
                    duration_col=time_col,
                    event_col=event_col,
                )
            conv_warn = [w for w in wlist if issubclass(w.category, ConvergenceWarning)]
            fit_sec = (datetime.now(timezone.utc) - fit_start).total_seconds()

            pred_start = datetime.now(timezone.utc)
            risk = cph.predict_partial_hazard(test_df[fit_feature_cols]).to_numpy(dtype=float).reshape(-1)
            pred_sec = (datetime.now(timezone.utc) - pred_start).total_seconds()
            event_times = test_df[time_col].to_numpy(dtype=float)
            events = test_df[event_col].to_numpy(dtype=float)
            cidx = _lifelines_cindex_from_risk(event_times, risk, events)
            cv_rows.append(
                {
                    "fit_sec": fit_sec,
                    "predict_sec": pred_sec,
                    "auc": cidx,
                    "brier": survival_partial_brier_score(event_times, risk, events),
                    "logloss": survival_partial_log_loss(event_times, risk, events),
                    "nagelkerke_r2": survival_partial_nagelkerke_r2(event_times, risk, events),
                    "n_test": int(len(fold.test_idx)),
                    "warning": (
                        f"lifelines convergence warning: {str(conv_warn[0].message)}"
                        if conv_warn
                        else None
                    ),
                    "model_spec": (
                        "CoxPHFitter(train-fold z-score; bmi 6-knot hinge spline; penalizer=1e-4; "
                        "c-index on partial-hazard risk score)"
                        if scenario["name"] == "icu_survival_death"
                        else "CoxPHFitter(linear terms; train-fold z-score; penalizer=1e-4; "
                        "c-index on partial-hazard risk score)"
                    ),
                }
            )
            continue

        if x_train.shape[1] > 0:
            mu = np.mean(x_train, axis=0)
            sdv = np.std(x_train, axis=0, ddof=1)
            sdv[~np.isfinite(sdv) | (sdv < 1e-8)] = 1.0
            x_train = (x_train - mu) / sdv
            x_test = (x_test - mu) / sdv

        if ds["family"] == "binomial":
            if x.shape[1] == 1:
                model = LogisticGAM(s(0, n_splines=8))
                model_spec = "LogisticGAM(s(0, n_splines=8))"
            elif scenario["name"] in {
                "geo_disease_tp",
                "geo_disease_duchon",
                "geo_disease_shrinkage",
                "geo_disease_matern",
            }:
                smooth_count = min(4, x.shape[1])
                terms = s(0, n_splines=12)
                for j in range(1, smooth_count):
                    terms = terms + s(j, n_splines=12)
                for j in range(smooth_count, x.shape[1]):
                    terms = terms + l(j)
                model = LogisticGAM(terms)
                linear_part = f"+linear({smooth_count}:{x.shape[1] - 1})" if x.shape[1] > smooth_count else ""
                model_spec = (
                    f"LogisticGAM(s(0)+...+s({smooth_count - 1}){linear_part}, n_splines=12)"
                )
            elif scenario["name"] in {"geo_disease_ps_per_pc"}:
                terms = s(0, n_splines=10)
                for j in range(1, x.shape[1]):
                    terms = terms + s(j, n_splines=10)
                model = LogisticGAM(terms)
                model_spec = "LogisticGAM(sum_j s(j, n_splines=10), j=0..15)"
            elif _geo_disease_eas_scenario_cfg(scenario["name"]) is not None:
                geo_cfg = _geo_disease_eas_scenario_cfg(scenario["name"])
                k = int(geo_cfg["knots"])
                n_pcs = int(geo_cfg["n_pcs"])
                if geo_cfg["basis_code"] == "psperpc":
                    terms = s(0, n_splines=k)
                    for j in range(1, x.shape[1]):
                        terms = terms + s(j, n_splines=k)
                    model = LogisticGAM(terms)
                    model_spec = f"LogisticGAM(sum_j s(j, n_splines={k}), j=0..{n_pcs - 1})"
                else:
                    terms = s(0, n_splines=k) + s(1, n_splines=k) + s(2, n_splines=k)
                    for j in range(3, x.shape[1]):
                        terms = terms + l(j)
                    model = LogisticGAM(terms)
                    linear_hi = n_pcs - 1
                    linear_part = f"+linear(3:{linear_hi})" if linear_hi >= 3 else ""
                    model_spec = (
                        f"LogisticGAM(s(0)+s(1)+s(2){linear_part}, n_splines={k})"
                    )
            elif _papuan_oce_scenario_cfg(scenario["name"]) is not None:
                papuan_cfg = _papuan_oce_scenario_cfg(scenario["name"])
                k = int(papuan_cfg["knots"])
                n_pcs = int(papuan_cfg["n_pcs"])
                if papuan_cfg["basis_code"] == "psperpc":
                    terms = s(0, n_splines=k)
                    for j in range(1, x.shape[1]):
                        terms = terms + s(j, n_splines=k)
                    model = LogisticGAM(terms)
                    model_spec = f"LogisticGAM(sum_j s(j, n_splines={k}), j=0..{n_pcs - 1}) [papuan_oce]"
                else:
                    terms = s(0, n_splines=k) + s(1, n_splines=k) + s(2, n_splines=k)
                    for j in range(3, x.shape[1]):
                        terms = terms + l(j)
                    model = LogisticGAM(terms)
                    linear_hi = n_pcs - 1
                    linear_part = f"+linear(3:{linear_hi})" if linear_hi >= 3 else ""
                    model_spec = (
                        f"LogisticGAM(s(0)+s(1)+s(2){linear_part}, n_splines={k}) [papuan_oce]"
                    )
            elif _geo_subpop16_scenario_cfg(scenario["name"]) is not None:
                sub_cfg = _geo_subpop16_scenario_cfg(scenario["name"])
                k = int(sub_cfg["knots"])
                if sub_cfg["basis_code"] == "psperpc":
                    terms = s(0, n_splines=k)
                    for j in range(1, x.shape[1]):
                        terms = terms + s(j, n_splines=k)
                    model = LogisticGAM(terms)
                    model_spec = f"LogisticGAM(sum_j s(j, n_splines={k}), j=0..15) [subpop16]"
                else:
                    terms = s(0, n_splines=k) + s(1, n_splines=k) + s(2, n_splines=k)
                    for j in range(3, x.shape[1]):
                        terms = terms + l(j)
                    model = LogisticGAM(terms)
                    model_spec = (
                        f"LogisticGAM(s(0)+s(1)+s(2)+linear(3:15), n_splines={k}) [subpop16]"
                    )
            elif _geo_latlon_scenario_cfg(scenario["name"]) is not None:
                latlon_cfg = _geo_latlon_scenario_cfg(scenario["name"])
                k = int(latlon_cfg["knots"])
                if latlon_cfg["basis_code"] == "psperpc":
                    terms = s(0, n_splines=k)
                    for j in range(1, x.shape[1]):
                        terms = terms + s(j, n_splines=k)
                    model = LogisticGAM(terms)
                    model_spec = f"LogisticGAM(sum_j s(j, n_splines={k}), j=0..5) [geo_latlon]"
                else:
                    terms = s(0, n_splines=k) + s(1, n_splines=k) + s(2, n_splines=k)
                    for j in range(3, x.shape[1]):
                        terms = terms + l(j)
                    model = LogisticGAM(terms)
                    model_spec = (
                        f"LogisticGAM(s(0)+s(1)+s(2)+linear(3:5), n_splines={k}) [geo_latlon]"
                    )
            elif scenario["name"] == "icu_survival_death":
                model = LogisticGAM(l(0) + l(1) + l(2) + l(3) + s(4, n_splines=10))
                model_spec = "LogisticGAM(l(0) + l(1) + l(2) + l(3) + s(4, n_splines=10))"
            elif scenario["name"] == "horse_colic":
                model = LogisticGAM(l(0) + s(1, n_splines=8) + l(2))
                model_spec = "LogisticGAM(l(0) + s(1, n_splines=8) + l(2))"
            elif scenario["name"] == "haberman_survival":
                model = LogisticGAM(l(0) + l(1) + s(2, n_splines=8))
                model_spec = "LogisticGAM(l(0) + l(1) + s(2, n_splines=8))"
            else:
                model = LogisticGAM(l(0) + s(1, n_splines=8))
                model_spec = "LogisticGAM(l(0) + s(1, n_splines=8))"
            fit_start = datetime.now(timezone.utc)
            lam_grid = np.logspace(-4, 4, 17)
            model.gridsearch(x_train, y_train, lam=lam_grid, objective="AICc", progress=False)
            fit_sec = (datetime.now(timezone.utc) - fit_start).total_seconds()

            pred_start = datetime.now(timezone.utc)
            pred = model.predict_mu(x_test).astype(float)
            pred_sec = (datetime.now(timezone.utc) - pred_start).total_seconds()
            cv_rows.append(
                {
                    "fit_sec": fit_sec,
                    "predict_sec": pred_sec,
                    "auc": auc_score(y_test, pred),
                    "brier": brier_score(y_test, pred),
                    "logloss": log_loss_score(y_test, pred),
                    "nagelkerke_r2": nagelkerke_r2_score(y_test, pred),
                    "n_test": int(len(fold.test_idx)),
                    "model_spec": model_spec,
                }
            )
            continue

        if scenario["name"] == "wine_gamair":
            model = LinearGAM(l(0) + l(1) + l(2) + l(3) + s(4, n_splines=10))
            model_spec = "LinearGAM(l(0) + l(1) + l(2) + l(3) + s(4, n_splines=10))"
        elif scenario["name"] in {"us48_demand_5day", "us48_demand_31day"}:
            model = LinearGAM(s(0, n_splines=10) + l(1) + l(2) + l(3))
            model_spec = "LinearGAM(s(0, n_splines=10) + l(1) + l(2) + l(3))"
        elif scenario["name"] == "icu_survival_los":
            model = LinearGAM(s(0, n_splines=10) + l(1) + l(2) + l(3) + l(4))
            model_spec = "LinearGAM(s(0, n_splines=10) + l(1) + l(2) + l(3) + l(4))"
        elif scenario["name"] in {"wine_temp_vs_year", "wine_price_vs_temp"}:
            model = LinearGAM(s(0, n_splines=10))
            model_spec = "LinearGAM(s(0, n_splines=10))"
        else:
            model = LinearGAM(s(0, n_splines=25))
            model_spec = "LinearGAM(s(0, n_splines=25))"

        fit_start = datetime.now(timezone.utc)
        lam_grid = np.logspace(-4, 4, 17)
        model.gridsearch(x_train, y_train, lam=lam_grid, objective="AICc", progress=False)
        fit_sec = (datetime.now(timezone.utc) - fit_start).total_seconds()

        pred_start = datetime.now(timezone.utc)
        pred = model.predict(x_test).astype(float)
        pred_sec = (datetime.now(timezone.utc) - pred_start).total_seconds()
        pred_train = model.predict(x_train).astype(float)
        sigma_hat = max(rmse_score(y_train, pred_train), 1e-12)

        cv_rows.append(
            {
                "fit_sec": fit_sec,
                "predict_sec": pred_sec,
                "logloss": gaussian_log_loss_score(y_test, pred, sigma_hat),
                "mse": mse_score(y_test, pred),
                "rmse": rmse_score(y_test, pred),
                "mae": mae_score(y_test, pred),
                "r2": r2_score(y_test, pred),
                "n_test": int(len(fold.test_idx)),
                "model_spec": model_spec,
            }
        )

    metrics = aggregate_cv_rows(cv_rows, ds["family"])
    return {
        "contender": "python_lifelines" if ds["family"] == "survival" else "python_pygam",
        "family": ds["family"],
        "scenario_name": scenario["name"],
        "status": "ok",
        **metrics,
        "model_spec": (
            f"{cv_rows[0]['model_spec']} [5-fold CV]"
            if ds["family"] == "survival"
            else f"{cv_rows[0]['model_spec']} [lam by UBRE/GCV; 5-fold CV]"
        ),
    }


def run_external_sksurv_rsf_cv(scenario, *, ds: dict | None = None, folds: list[Fold] | None = None):
    if ds is None:
        ds = dataset_for_scenario(scenario)
    if ds["family"] != "survival":
        return None
    try:
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.util import Surv
    except Exception as e:
        return {
            "contender": "python_sksurv_rsf",
            "scenario_name": scenario["name"],
            "status": "failed",
            "error": f"scikit-survival import failed: {e}",
        }

    if folds is None:
        folds = folds_for_dataset(ds)
    df = pd.DataFrame(ds["rows"])
    feature_cols = ds["features"]
    time_col = ds["time_col"]
    event_col = ds["event_col"]
    cv_rows = []

    for fold_id, fold in enumerate(folds):
        train_df = df.iloc[fold.train_idx].copy()
        test_df = df.iloc[fold.test_idx].copy()
        x_train = train_df[feature_cols].to_numpy(dtype=float)
        x_test = test_df[feature_cols].to_numpy(dtype=float)
        y_train = Surv.from_arrays(
            event=train_df[event_col].to_numpy(dtype=float) > 0.5,
            time=train_df[time_col].to_numpy(dtype=float),
        )

        fit_start = datetime.now(timezone.utc)
        rsf = RandomSurvivalForest(
            n_estimators=300,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=1,
            random_state=CV_SEED + fold_id,
        )
        rsf.fit(x_train, y_train)
        fit_sec = (datetime.now(timezone.utc) - fit_start).total_seconds()

        pred_start = datetime.now(timezone.utc)
        risk = rsf.predict(x_test).astype(float)
        pred_sec = (datetime.now(timezone.utc) - pred_start).total_seconds()

        event_times = test_df[time_col].to_numpy(dtype=float)
        events = test_df[event_col].to_numpy(dtype=float)
        cv_rows.append(
            {
                "fit_sec": fit_sec,
                "predict_sec": pred_sec,
                "auc": _lifelines_cindex_from_risk(event_times, risk, events),
                "brier": survival_partial_brier_score(event_times, risk, events),
                "logloss": survival_partial_log_loss(event_times, risk, events),
                "nagelkerke_r2": survival_partial_nagelkerke_r2(event_times, risk, events),
                "n_test": int(len(fold.test_idx)),
                "model_spec": (
                    "RandomSurvivalForest("
                    "n_estimators=300,min_samples_split=10,min_samples_leaf=5,max_features='sqrt')"
                ),
            }
        )

    metrics = aggregate_cv_rows(cv_rows, ds["family"])
    return {
        "contender": "python_sksurv_rsf",
        "family": ds["family"],
        "scenario_name": scenario["name"],
        "status": "ok",
        **metrics,
        "model_spec": f"{cv_rows[0]['model_spec']} [5-fold CV]",
    }


def run_external_sksurv_coxnet_cv(scenario, *, ds: dict | None = None, folds: list[Fold] | None = None):
    if ds is None:
        ds = dataset_for_scenario(scenario)
    if ds["family"] != "survival":
        return None
    try:
        from sksurv.linear_model import CoxnetSurvivalAnalysis
        from sksurv.util import Surv
    except Exception as e:
        return {
            "contender": "python_sksurv_coxnet",
            "scenario_name": scenario["name"],
            "status": "failed",
            "error": f"scikit-survival import failed: {e}",
        }

    if folds is None:
        folds = folds_for_dataset(ds)
    df = pd.DataFrame(ds["rows"])
    feature_cols = ds["features"]
    time_col = ds["time_col"]
    event_col = ds["event_col"]
    cv_rows = []

    for fold_id, fold in enumerate(folds):
        train_df = df.iloc[fold.train_idx].copy()
        test_df = df.iloc[fold.test_idx].copy()
        train_df, test_df = zscore_train_test(train_df, test_df, feature_cols)
        x_train = train_df[feature_cols].to_numpy(dtype=float)
        x_test = test_df[feature_cols].to_numpy(dtype=float)
        y_train = Surv.from_arrays(
            event=train_df[event_col].to_numpy(dtype=float) > 0.5,
            time=train_df[time_col].to_numpy(dtype=float),
        )

        fit_start = datetime.now(timezone.utc)
        model = CoxnetSurvivalAnalysis(
            l1_ratio=0.5,
            alpha_min_ratio=0.01,
            max_iter=100000,
            fit_baseline_model=False,
        )
        model.fit(x_train, y_train)
        fit_sec = (datetime.now(timezone.utc) - fit_start).total_seconds()

        pred_start = datetime.now(timezone.utc)
        risk = model.predict(x_test).astype(float)
        pred_sec = (datetime.now(timezone.utc) - pred_start).total_seconds()
        event_times = test_df[time_col].to_numpy(dtype=float)
        events = test_df[event_col].to_numpy(dtype=float)
        cv_rows.append(
            {
                "fit_sec": fit_sec,
                "predict_sec": pred_sec,
                "auc": _lifelines_cindex_from_risk(event_times, risk, events),
                "brier": survival_partial_brier_score(event_times, risk, events),
                "logloss": survival_partial_log_loss(event_times, risk, events),
                "nagelkerke_r2": survival_partial_nagelkerke_r2(event_times, risk, events),
                "n_test": int(len(fold.test_idx)),
                "model_spec": "CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.01)",
            }
        )

    metrics = aggregate_cv_rows(cv_rows, ds["family"])
    return {
        "contender": "python_sksurv_coxnet",
        "family": ds["family"],
        "scenario_name": scenario["name"],
        "status": "ok",
        **metrics,
        "model_spec": f"{cv_rows[0]['model_spec']} [5-fold CV]",
    }


def run_external_lifelines_coxph_enet_cv(scenario, *, ds: dict | None = None, folds: list[Fold] | None = None):
    if ds is None:
        ds = dataset_for_scenario(scenario)
    if ds["family"] != "survival":
        return None

    if folds is None:
        folds = folds_for_dataset(ds)
    df = pd.DataFrame(ds["rows"])
    feature_cols = ds["features"]
    time_col = ds["time_col"]
    event_col = ds["event_col"]
    cv_rows = []

    for fold in folds:
        train_df = df.iloc[fold.train_idx].copy()
        test_df = df.iloc[fold.test_idx].copy()
        train_df, test_df = zscore_train_test(train_df, test_df, feature_cols)
        fit_feature_cols = feature_cols
        if scenario["name"] == "icu_survival_death" and "bmi" in feature_cols:
            train_df, test_df, bmi_spline_cols = _augment_bmi_spline_linear_hinges(
                train_df, test_df, n_knots=6
            )
            fit_feature_cols = [c for c in feature_cols if c != "bmi"] + bmi_spline_cols

        fit_start = datetime.now(timezone.utc)
        cph = CoxPHFitter(penalizer=0.05, l1_ratio=0.5)
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            cph.fit(
                train_df[[*fit_feature_cols, time_col, event_col]],
                duration_col=time_col,
                event_col=event_col,
            )
        conv_warn = [w for w in wlist if issubclass(w.category, ConvergenceWarning)]
        fit_sec = (datetime.now(timezone.utc) - fit_start).total_seconds()

        pred_start = datetime.now(timezone.utc)
        risk = cph.predict_partial_hazard(test_df[fit_feature_cols]).to_numpy(dtype=float).reshape(-1)
        pred_sec = (datetime.now(timezone.utc) - pred_start).total_seconds()
        event_times = test_df[time_col].to_numpy(dtype=float)
        events = test_df[event_col].to_numpy(dtype=float)
        cv_rows.append(
            {
                "fit_sec": fit_sec,
                "predict_sec": pred_sec,
                "auc": _lifelines_cindex_from_risk(event_times, risk, events),
                "brier": survival_partial_brier_score(event_times, risk, events),
                "logloss": survival_partial_log_loss(event_times, risk, events),
                "nagelkerke_r2": survival_partial_nagelkerke_r2(event_times, risk, events),
                "n_test": int(len(fold.test_idx)),
                "warning": (
                    f"lifelines convergence warning: {str(conv_warn[0].message)}"
                    if conv_warn
                    else None
                ),
                "model_spec": (
                    "CoxPHFitter(train-fold z-score; bmi 6-knot hinge spline; "
                    "penalizer=0.05; l1_ratio=0.5)"
                    if scenario["name"] == "icu_survival_death"
                    else "CoxPHFitter(linear terms; train-fold z-score; penalizer=0.05; l1_ratio=0.5)"
                ),
            }
        )

    metrics = aggregate_cv_rows(cv_rows, ds["family"])
    return {
        "contender": "python_lifelines_coxph_enet",
        "family": ds["family"],
        "scenario_name": scenario["name"],
        "status": "ok",
        **metrics,
        "model_spec": f"{cv_rows[0]['model_spec']} [5-fold CV]",
    }


def run_external_glmnet_cox_cv(scenario, *, ds: dict | None = None, folds: list[Fold] | None = None):
    if ds is None:
        ds = dataset_for_scenario(scenario)
    if ds["family"] != "survival":
        return None
    if folds is None:
        folds = folds_for_dataset(ds)

    with tempfile.TemporaryDirectory(prefix="gam_bench_glmnet_cox_cv_") as td:
        td_path = Path(td)
        data_path = td_path / "data.json"
        out_path = td_path / "out.json"
        script_path = td_path / "run_glmnet_cox_cv.R"

        payload = {
            "dataset": ds,
            "scenario_name": scenario["name"],
        }
        data_path.write_text(json.dumps(payload))

        script = r'''
args <- commandArgs(trailingOnly = TRUE)
data_path <- args[1]
out_path <- args[2]
train_idx_path <- args[3]
test_idx_path <- args[4]

suppressPackageStartupMessages({
  library(glmnet)
  library(survival)
  library(jsonlite)
})

payload <- fromJSON(data_path, simplifyVector = TRUE)
df <- as.data.frame(payload$dataset$rows)
time_col <- as.character(payload$dataset$time_col)
event_col <- as.character(payload$dataset$event_col)
feature_cols <- as.character(payload$dataset$features)

coerce_positive_times <- function(x) {
  vals <- as.numeric(x)
  bad <- !is.finite(vals) | vals <= 0
  if (!any(bad)) {
    return(vals)
  }
  pos <- vals[is.finite(vals) & vals > 0]
  if (!length(pos)) {
    stop("No strictly positive survival times available for Cox glmnet")
  }
  replacement <- max(min(pos) * 0.5, 1e-12)
  vals[bad] <- replacement
  vals
}

train_idx <- scan(train_idx_path, what=integer(), quiet=TRUE) + 1L
test_idx <- scan(test_idx_path, what=integer(), quiet=TRUE) + 1L
train_df <- df[train_idx, , drop=FALSE]
test_df <- df[test_idx, , drop=FALSE]
train_df[[time_col]] <- coerce_positive_times(train_df[[time_col]])
test_df[[time_col]] <- coerce_positive_times(test_df[[time_col]])

for (cn in feature_cols) {
  mu <- mean(train_df[[cn]])
  sdv <- stats::sd(train_df[[cn]])
  if (!is.finite(sdv) || sdv < 1e-8) sdv <- 1.0
  train_df[[cn]] <- (train_df[[cn]] - mu) / sdv
  test_df[[cn]] <- (test_df[[cn]] - mu) / sdv
}

x_train <- as.matrix(train_df[, feature_cols, drop=FALSE])
x_test <- as.matrix(test_df[, feature_cols, drop=FALSE])
y_train <- survival::Surv(
  time=as.numeric(train_df[[time_col]]),
  event=as.numeric(train_df[[event_col]]) > 0.5
)

t0 <- proc.time()[["elapsed"]]
cvfit <- cv.glmnet(
  x=x_train,
  y=y_train,
  family="cox",
  alpha=0.5,
  nfolds=5,
  standardize=FALSE
)
fit_sec <- proc.time()[["elapsed"]] - t0

pred_t0 <- proc.time()[["elapsed"]]
lp_train <- as.numeric(predict(cvfit, newx=x_train, s="lambda.min", type="link"))
lp_test <- as.numeric(predict(cvfit, newx=x_test, s="lambda.min", type="link"))
risk <- as.numeric(lp_test)
pred_sec <- proc.time()[["elapsed"]] - pred_t0

out <- list(
  status="ok",
  fit_sec=fit_sec,
  predict_sec=pred_sec,
  auc=NULL,
  risk=risk,
  brier=NULL,
  logloss=NULL,
  rmse=NULL,
  mae=NULL,
  r2=NULL,
  model_spec="cv.glmnet(family='cox', alpha=0.5, s='lambda.min')"
)
write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
'''
        script_path.write_text(script)

        cv_rows = []
        all_df = pd.DataFrame(ds["rows"])
        for fold_id, fold in enumerate(folds):
            train_idx_path = td_path / f"train_idx_{fold_id}.txt"
            test_idx_path = td_path / f"test_idx_{fold_id}.txt"
            train_idx_path.write_text("\n".join(str(int(i)) for i in fold.train_idx) + "\n")
            test_idx_path.write_text("\n".join(str(int(i)) for i in fold.test_idx) + "\n")

            code, out, err = run_cmd(
                [
                    "Rscript",
                    str(script_path),
                    str(data_path),
                    str(out_path),
                    str(train_idx_path),
                    str(test_idx_path),
                ],
                cwd=ROOT,
            )
            if code != 0:
                return {
                    "contender": "r_glmnet_cox",
                    "scenario_name": scenario["name"],
                    "status": "failed",
                    "error": err.strip() or out.strip(),
                }

            fold_row = json.loads(out_path.read_text())
            train_df = all_df.iloc[fold.train_idx].copy()
            test_df = all_df.iloc[fold.test_idx].copy()
            event_times = test_df[ds["time_col"]].to_numpy(dtype=float)
            events = test_df[ds["event_col"]].to_numpy(dtype=float)
            risk = np.asarray(fold_row.get("risk", []), dtype=float).reshape(-1)
            if risk.shape[0] != event_times.shape[0]:
                return {
                    "contender": "r_glmnet_cox",
                    "scenario_name": scenario["name"],
                    "status": "failed",
                    "error": (
                        "r_glmnet_cox fold output missing/invalid risk vector "
                        f"(got {risk.shape[0]}, expected {event_times.shape[0]})"
                    ),
                }
            fold_row["auc"] = _lifelines_cindex_from_risk(event_times, risk, events)
            fold_row["brier"] = survival_partial_brier_score(event_times, risk, events)
            fold_row["logloss"] = survival_partial_log_loss(event_times, risk, events)
            fold_row["nagelkerke_r2"] = survival_partial_nagelkerke_r2(event_times, risk, events)
            fold_row.pop("risk", None)
            fold_row["n_test"] = int(len(fold.test_idx))
            cv_rows.append(fold_row)

    metrics = aggregate_cv_rows(cv_rows, ds["family"])
    return {
        "contender": "r_glmnet_cox",
        "family": ds["family"],
        "scenario_name": scenario["name"],
        "status": "ok",
        **metrics,
        "model_spec": f"{cv_rows[0]['model_spec']} [5-fold CV]",
    }


def run_external_sksurv_gb_cv(scenario, *, ds: dict | None = None, folds: list[Fold] | None = None):
    if ds is None:
        ds = dataset_for_scenario(scenario)
    if ds["family"] != "survival":
        return None
    try:
        from sksurv.ensemble import (
            ComponentwiseGradientBoostingSurvivalAnalysis,
            GradientBoostingSurvivalAnalysis,
        )
        from sksurv.util import Surv
    except Exception as e:
        return {
            "contender": "python_sksurv_gb_coxph",
            "scenario_name": scenario["name"],
            "status": "failed",
            "error": f"scikit-survival import failed: {e}",
        }

    if folds is None:
        folds = folds_for_dataset(ds)
    df = pd.DataFrame(ds["rows"])
    feature_cols = ds["features"]
    time_col = ds["time_col"]
    event_col = ds["event_col"]

    gb_rows = []
    cgb_rows = []
    gb_spec = "GradientBoostingSurvivalAnalysis(loss='coxph', n_estimators=300, lr=0.05, max_depth=3)"
    cgb_spec = "ComponentwiseGradientBoostingSurvivalAnalysis(loss='coxph', n_estimators=500, lr=0.05)"
    for fold in folds:
        train_df = df.iloc[fold.train_idx].copy()
        test_df = df.iloc[fold.test_idx].copy()
        train_df, test_df = zscore_train_test(train_df, test_df, feature_cols)
        x_train = train_df[feature_cols].to_numpy(dtype=float)
        x_test = test_df[feature_cols].to_numpy(dtype=float)
        y_train = Surv.from_arrays(
            event=train_df[event_col].to_numpy(dtype=float) > 0.5,
            time=train_df[time_col].to_numpy(dtype=float),
        )
        event_times = test_df[time_col].to_numpy(dtype=float)
        events = test_df[event_col].to_numpy(dtype=float)

        gb_model = GradientBoostingSurvivalAnalysis(
            loss="coxph",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=CV_SEED,
        )
        fit_start = datetime.now(timezone.utc)
        gb_model.fit(x_train, y_train)
        fit_sec = (datetime.now(timezone.utc) - fit_start).total_seconds()
        pred_start = datetime.now(timezone.utc)
        gb_risk = gb_model.predict(x_test).astype(float)
        pred_sec = (datetime.now(timezone.utc) - pred_start).total_seconds()
        gb_rows.append(
            {
                "fit_sec": fit_sec,
                "predict_sec": pred_sec,
                "auc": _lifelines_cindex_from_risk(event_times, gb_risk, events),
                "brier": survival_partial_brier_score(event_times, gb_risk, events),
                "logloss": survival_partial_log_loss(event_times, gb_risk, events),
                "nagelkerke_r2": survival_partial_nagelkerke_r2(event_times, gb_risk, events),
                "n_test": int(len(fold.test_idx)),
                "model_spec": gb_spec,
            }
        )

        cgb_model = ComponentwiseGradientBoostingSurvivalAnalysis(
            loss="coxph",
            n_estimators=500,
            learning_rate=0.05,
            random_state=CV_SEED,
        )
        fit_start = datetime.now(timezone.utc)
        cgb_model.fit(x_train, y_train)
        fit_sec = (datetime.now(timezone.utc) - fit_start).total_seconds()
        pred_start = datetime.now(timezone.utc)
        cgb_risk = cgb_model.predict(x_test).astype(float)
        pred_sec = (datetime.now(timezone.utc) - pred_start).total_seconds()
        cgb_rows.append(
            {
                "fit_sec": fit_sec,
                "predict_sec": pred_sec,
                "auc": _lifelines_cindex_from_risk(event_times, cgb_risk, events),
                "brier": survival_partial_brier_score(event_times, cgb_risk, events),
                "logloss": survival_partial_log_loss(event_times, cgb_risk, events),
                "nagelkerke_r2": survival_partial_nagelkerke_r2(event_times, cgb_risk, events),
                "n_test": int(len(fold.test_idx)),
                "model_spec": cgb_spec,
            }
        )

    gb_metrics = aggregate_cv_rows(gb_rows, ds["family"])
    cgb_metrics = aggregate_cv_rows(cgb_rows, ds["family"])
    return [
        {
            "contender": "python_sksurv_gb_coxph",
            "family": ds["family"],
            "scenario_name": scenario["name"],
            "status": "ok",
            **gb_metrics,
            "model_spec": f"{gb_rows[0]['model_spec']} [5-fold CV]",
        },
        {
            "contender": "python_sksurv_componentwise_gb_coxph",
            "family": ds["family"],
            "scenario_name": scenario["name"],
            "status": "ok",
            **cgb_metrics,
            "model_spec": f"{cgb_rows[0]['model_spec']} [5-fold CV]",
        },
    ]


def run_external_lifelines_aft_cv(scenario, *, ds: dict | None = None, folds: list[Fold] | None = None):
    if ds is None:
        ds = dataset_for_scenario(scenario)
    if ds["family"] != "survival":
        return None

    if folds is None:
        folds = folds_for_dataset(ds)
    df = pd.DataFrame(ds["rows"])
    feature_cols = ds["features"]
    time_col = ds["time_col"]
    event_col = ds["event_col"]

    weibull_rows = []
    lognormal_rows = []
    weibull_spec = "WeibullAFTFitter(train-fold z-score; penalizer=1e-3; l1_ratio=0.2)"
    lognormal_spec = "LogNormalAFTFitter(train-fold z-score; penalizer=1e-3; l1_ratio=0.2)"
    for fold in folds:
        train_df = df.iloc[fold.train_idx].copy()
        test_df = df.iloc[fold.test_idx].copy()
        train_df, test_df = zscore_train_test(train_df, test_df, feature_cols)
        event_times = test_df[time_col].to_numpy(dtype=float)
        events = test_df[event_col].to_numpy(dtype=float)

        weibull = WeibullAFTFitter(penalizer=1e-3, l1_ratio=0.2)
        fit_start = datetime.now(timezone.utc)
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            weibull.fit(
                train_df[[*feature_cols, time_col, event_col]],
                duration_col=time_col,
                event_col=event_col,
            )
        conv_warn = [w for w in wlist if issubclass(w.category, ConvergenceWarning)]
        fit_sec = (datetime.now(timezone.utc) - fit_start).total_seconds()
        pred_start = datetime.now(timezone.utc)
        pred_time = weibull.predict_expectation(test_df[feature_cols]).to_numpy(dtype=float).reshape(-1)
        risk = -pred_time
        pred_sec = (datetime.now(timezone.utc) - pred_start).total_seconds()
        weibull_rows.append(
            {
                "fit_sec": fit_sec,
                "predict_sec": pred_sec,
                "auc": _lifelines_cindex_from_risk(event_times, risk, events),
                "brier": survival_partial_brier_score(event_times, risk, events),
                "logloss": survival_partial_log_loss(event_times, risk, events),
                "nagelkerke_r2": survival_partial_nagelkerke_r2(event_times, risk, events),
                "n_test": int(len(fold.test_idx)),
                "warning": (
                    f"lifelines convergence warning: {str(conv_warn[0].message)}"
                    if conv_warn
                    else None
                ),
                "model_spec": weibull_spec,
            }
        )

        lognormal = LogNormalAFTFitter(penalizer=1e-3, l1_ratio=0.2)
        fit_start = datetime.now(timezone.utc)
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            lognormal.fit(
                train_df[[*feature_cols, time_col, event_col]],
                duration_col=time_col,
                event_col=event_col,
            )
        conv_warn = [w for w in wlist if issubclass(w.category, ConvergenceWarning)]
        fit_sec = (datetime.now(timezone.utc) - fit_start).total_seconds()
        pred_start = datetime.now(timezone.utc)
        pred_time = lognormal.predict_expectation(test_df[feature_cols]).to_numpy(dtype=float).reshape(-1)
        risk = -pred_time
        pred_sec = (datetime.now(timezone.utc) - pred_start).total_seconds()
        lognormal_rows.append(
            {
                "fit_sec": fit_sec,
                "predict_sec": pred_sec,
                "auc": _lifelines_cindex_from_risk(event_times, risk, events),
                "brier": survival_partial_brier_score(event_times, risk, events),
                "logloss": survival_partial_log_loss(event_times, risk, events),
                "nagelkerke_r2": survival_partial_nagelkerke_r2(event_times, risk, events),
                "n_test": int(len(fold.test_idx)),
                "warning": (
                    f"lifelines convergence warning: {str(conv_warn[0].message)}"
                    if conv_warn
                    else None
                ),
                "model_spec": lognormal_spec,
            }
        )

    weibull_metrics = aggregate_cv_rows(weibull_rows, ds["family"])
    lognormal_metrics = aggregate_cv_rows(lognormal_rows, ds["family"])
    return [
        {
            "contender": "python_lifelines_weibull_aft",
            "family": ds["family"],
            "scenario_name": scenario["name"],
            "status": "ok",
            **weibull_metrics,
            "model_spec": f"{weibull_rows[0]['model_spec']} [5-fold CV]",
        },
        {
            "contender": "python_lifelines_lognormal_aft",
            "family": ds["family"],
            "scenario_name": scenario["name"],
            "status": "ok",
            **lognormal_metrics,
            "model_spec": f"{lognormal_rows[0]['model_spec']} [5-fold CV]",
        },
    ]


def run_external_xgboost_aft_cv(scenario, *, ds: dict | None = None, folds: list[Fold] | None = None):
    if ds is None:
        ds = dataset_for_scenario(scenario)
    if ds["family"] != "survival":
        return None
    try:
        import xgboost as xgb
    except Exception as e:
        return {
            "contender": "python_xgboost_aft",
            "scenario_name": scenario["name"],
            "status": "failed",
            "error": f"xgboost import failed: {e}",
        }

    if folds is None:
        folds = folds_for_dataset(ds)
    df = pd.DataFrame(ds["rows"])
    feature_cols = ds["features"]
    time_col = ds["time_col"]
    event_col = ds["event_col"]
    cv_rows = []

    for fold_id, fold in enumerate(folds):
        train_df = df.iloc[fold.train_idx].copy()
        test_df = df.iloc[fold.test_idx].copy()
        train_df, test_df = zscore_train_test(train_df, test_df, feature_cols)

        x_train = train_df[feature_cols].to_numpy(dtype=float)
        x_test = test_df[feature_cols].to_numpy(dtype=float)
        y_time = train_df[time_col].to_numpy(dtype=float)
        y_event = train_df[event_col].to_numpy(dtype=float) > 0.5
        dtest = xgb.DMatrix(x_test)

        params = {
            "objective": "survival:aft",
            "eval_metric": "aft-nloglik",
            "aft_loss_distribution": "normal",
            "aft_loss_distribution_scale": 1.0,
            "tree_method": "hist",
            "learning_rate": 0.05,
            "max_depth": 3,
            "subsample": 0.9,
            "colsample_bynode": 0.8,
            "lambda": 1.0,
            "alpha": 0.0,
            "seed": int(CV_SEED + fold_id),
            "nthread": 1,
        }

        fit_start = datetime.now(timezone.utc)
        dtrain = xgb.DMatrix(x_train)
        dtrain.set_float_info("label_lower_bound", y_time.copy())
        dtrain.set_float_info("label_upper_bound", np.where(y_event, y_time, np.inf))
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=300,
            verbose_eval=False,
        )
        selected_rounds = int(booster.num_boosted_rounds())
        fit_sec = (datetime.now(timezone.utc) - fit_start).total_seconds()

        pred_start = datetime.now(timezone.utc)
        pred_time = booster.predict(dtest).astype(float).reshape(-1)
        risk = -pred_time
        pred_sec = (datetime.now(timezone.utc) - pred_start).total_seconds()

        event_times = test_df[time_col].to_numpy(dtype=float)
        events = test_df[event_col].to_numpy(dtype=float)
        cv_rows.append(
            {
                "fit_sec": fit_sec,
                "predict_sec": pred_sec,
                "auc": _lifelines_cindex_from_risk(event_times, risk, events),
                "brier": survival_partial_brier_score(event_times, risk, events),
                "logloss": survival_partial_log_loss(event_times, risk, events),
                "nagelkerke_r2": survival_partial_nagelkerke_r2(event_times, risk, events),
                "n_test": int(len(fold.test_idx)),
                "model_spec": (
                    "xgboost.train(objective='survival:aft',loss='normal',scale=1.0,"
                    f"max_depth=3,eta=0.05,selected_rounds={selected_rounds})"
                ),
            }
        )

    metrics = aggregate_cv_rows(cv_rows, ds["family"])
    return {
        "contender": "python_xgboost_aft",
        "family": ds["family"],
        "scenario_name": scenario["name"],
        "status": "ok",
        **metrics,
        "model_spec": f"{cv_rows[0]['model_spec']} [5-fold CV]",
    }


def _assert_basis_parity_for_scenario(s_cfg, *, ds: dict | None = None):
    # Fairness guard: if Rust mapping requests a named spline family, mgcv must
    # emit the equivalent basis for the same scenario.
    if ds is None:
        ds = dataset_for_scenario(s_cfg)
    scenario_name = s_cfg["name"]
    rust_cfg = _rust_fit_mapping(scenario_name)
    if rust_cfg is None:
        return
    basis = _canonical_smooth_basis(rust_cfg.get("smooth_basis", "ps"))
    if basis not in {"duchon", "matern"}:
        return

    mgcv_formula = _mgcv_formula_for_scenario(scenario_name, ds)
    expected = "bs='ds'" if basis == "duchon" else "bs='gp'"
    if expected not in mgcv_formula:
        raise SystemExit(
            "basis parity check failed for "
            f"{scenario_name}: rust basis='{basis}' but mgcv formula is '{mgcv_formula}'"
        )


def _extra_excluded_contenders_for_profile() -> set[str]:
    if _BENCH_CI_PROFILE == "lean":
        return set(_LEAN_PROFILE_EXCLUDED_CONTENDERS)
    return set()


def _is_contender_enabled(s_cfg, contender: str) -> bool:
    excluded = set(s_cfg.get("exclude_contenders", []))
    excluded.update(_extra_excluded_contenders_for_profile())
    return contender not in excluded


def _should_run_pygam_for_scenario(s_cfg, *, ds: dict | None = None):
    if not _is_contender_enabled(s_cfg, "python_pygam"):
        return False
    if ds is None:
        ds = dataset_for_scenario(s_cfg)
    # pyGAM has no native censored-likelihood survival model support in this harness.
    if ds["family"] == "survival":
        return False

    rust_cfg = _rust_fit_mapping(s_cfg["name"])
    if rust_cfg is None:
        return True
    basis = _canonical_smooth_basis(rust_cfg.get("smooth_basis", "ps"))
    # pyGAM in this harness does not provide native thin-plate/Duchon/Matérn bases.
    return basis not in {"thinplate", "duchon", "matern"}


def _is_non_blocking_failure(row: dict) -> bool:
    return str(row.get("contender", "")) in NON_BLOCKING_FAILURE_CONTENDERS


def main():
    parser = argparse.ArgumentParser(description="Run GAM benchmark suite with leakage-safe 5-fold CV.")
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--out", type=Path, default=BENCH_DIR / "results.json")
    parser.add_argument(
        "--scenario-name",
        action="append",
        dest="scenario_names",
        help="Run only the named scenario(s). Can be passed multiple times.",
    )
    args = parser.parse_args()
    print(
        "bench runtime config | "
        f"force_serial={_FORCE_SERIAL} "
        f"rayon_threads={_RAYON_THREADS if _RAYON_THREADS > 0 else 'auto'} "
        f"blas_threads={_BLAS_THREADS if _BLAS_THREADS > 0 else 'auto'} "
        f"cmd_timeout_sec={_CMD_TIMEOUT_SEC if _CMD_TIMEOUT_SEC > 0 else 'none'}",
        file=sys.stderr,
        flush=True,
    )

    cfg = json.loads(args.scenarios.read_text())
    scenarios = cfg.get("scenarios", [])
    if not scenarios:
        raise SystemExit("No scenarios found.")

    if args.scenario_names:
        wanted = set(args.scenario_names)
        scenarios = [s for s in scenarios if s.get("name") in wanted]
        missing = sorted(wanted - {s.get("name") for s in scenarios})
        if missing:
            raise SystemExit(f"Unknown scenario name(s): {', '.join(missing)}")
        if not scenarios:
            raise SystemExit("Scenario filter matched zero scenarios.")

    results = []
    for s_cfg in scenarios:
        ds = dataset_for_scenario(s_cfg)
        folds = folds_for_dataset(ds)
        _assert_basis_parity_for_scenario(s_cfg, ds=ds)
        with tempfile.TemporaryDirectory(prefix="gam_bench_shared_folds_", dir=str(BENCH_DIR)) as shared_td:
            shared_fold_artifacts = build_shared_fold_artifacts(ds, folds, Path(shared_td))
            # Invariant: Rust GAM must run for every scenario, including survival.
            results.append(
                run_rust_scenario_cv(
                    s_cfg,
                    ds=ds,
                    folds=folds,
                    shared_fold_artifacts=shared_fold_artifacts,
                )
            )
            if _is_matern_rust_scenario(s_cfg):
                results.append(
                    run_rust_scenario_cv(
                        s_cfg,
                        contender_name="rust_matern_decomposed",
                        ds=ds,
                        folds=folds,
                        shared_fold_artifacts=shared_fold_artifacts,
                        rust_cfg_override={"double_penalty": True},
                        eval_ood=True,
                    )
                )
                results.append(
                    run_rust_scenario_cv(
                        s_cfg,
                        contender_name="rust_matern_standard",
                        ds=ds,
                        folds=folds,
                        shared_fold_artifacts=shared_fold_artifacts,
                        rust_cfg_override={"double_penalty": False},
                        eval_ood=True,
                    )
                )
            if str(s_cfg.get("name", "")).startswith("thread2_"):
                results.append(
                    run_rust_scenario_cv(
                        s_cfg,
                        contender_name="rust_thread2_order_probe",
                        ds=ds,
                        folds=folds,
                        shared_fold_artifacts=shared_fold_artifacts,
                        rust_cfg_override={"double_penalty": True},
                        collect_continuous_order=True,
                    )
                )
            if str(s_cfg.get("name", "")) == "thread3_admixture_cliff":
                results.append(
                    run_rust_scenario_cv(
                        s_cfg,
                        contender_name="rust_thread3_standard_reml",
                        ds=ds,
                        folds=folds,
                        shared_fold_artifacts=shared_fold_artifacts,
                        rust_cfg_override={"double_penalty": True},
                    )
                )
                results.append(
                    run_rust_scenario_cv(
                        s_cfg,
                        contender_name="rust_thread3_adaptive_reml",
                        ds=ds,
                        folds=folds,
                        shared_fold_artifacts=shared_fold_artifacts,
                        rust_cfg_override={"double_penalty": True},
                        collect_adaptive_diagnostics=True,
                        rust_fit_extra_args=[
                            "--adaptive-regularization",
                            "--adaptive-max-mm-iter",
                            "8",
                            "--adaptive-beta-rel-tol",
                            "1e-3",
                            "--adaptive-min-epsilon",
                            "1e-8",
                            "--adaptive-weight-floor",
                            "1e-8",
                            "--adaptive-weight-ceiling",
                            "1e8",
                        ],
                    )
                )
            rust_sas_row = (
                run_rust_sas_scenario_cv(
                    s_cfg,
                    ds=ds,
                    folds=folds,
                    shared_fold_artifacts=shared_fold_artifacts,
                )
                if _is_contender_enabled(s_cfg, "rust_gam_sas")
                else None
            )
            if rust_sas_row is not None:
                results.append(rust_sas_row)
            rust_gamlss_row = run_rust_gamlss_scenario_cv(
                s_cfg,
                ds=ds,
                folds=folds,
                shared_fold_artifacts=shared_fold_artifacts,
            )
            if rust_gamlss_row is not None:
                results.append(rust_gamlss_row)
            rust_gamlss_surv_row = (
                run_rust_gamlss_survival_cv(s_cfg, ds=ds, folds=folds)
                if _is_contender_enabled(s_cfg, "rust_gamlss_survival")
                else None
            )
            if rust_gamlss_surv_row is not None:
                results.append(rust_gamlss_surv_row)
            r_gamlss_row = (
                run_external_r_gamlss_cv(s_cfg, ds=ds, folds=folds)
                if _is_contender_enabled(s_cfg, "r_gamlss")
                else None
            )
            if r_gamlss_row is not None:
                results.append(r_gamlss_row)
            if _is_contender_enabled(s_cfg, "r_mgcv"):
                results.append(run_external_mgcv_cv(s_cfg, ds=ds, folds=folds))
            mgcv_gaulss_row = (
                run_external_mgcv_gaulss_cv(s_cfg, ds=ds, folds=folds)
                if _is_contender_enabled(s_cfg, "r_mgcv_gaulss")
                else None
            )
            if mgcv_gaulss_row is not None:
                results.append(mgcv_gaulss_row)
            gamboostlss_row = (
                run_external_r_gamboostlss_cv(s_cfg, ds=ds, folds=folds)
                if _is_contender_enabled(s_cfg, "r_gamboostlss")
                else None
            )
            if gamboostlss_row is not None:
                results.append(gamboostlss_row)
            bamlss_row = (
                run_external_r_bamlss_cv(s_cfg, ds=ds, folds=folds)
                if _is_contender_enabled(s_cfg, "r_bamlss")
                else None
            )
            if bamlss_row is not None:
                results.append(bamlss_row)
            brms_row = (
                run_external_r_brms_cv(s_cfg, ds=ds, folds=folds)
                if _is_contender_enabled(s_cfg, "r_brms")
                else None
            )
            if brms_row is not None:
                results.append(brms_row)
            mgcv_surv_row = (
                run_external_mgcv_survival_cv(s_cfg, ds=ds, folds=folds)
                if _is_contender_enabled(s_cfg, "r_mgcv_coxph")
                else None
            )
            if mgcv_surv_row is not None:
                results.append(mgcv_surv_row)
            if _should_run_pygam_for_scenario(s_cfg, ds=ds):
                results.append(run_external_pygam_cv(s_cfg, ds=ds, folds=folds))
            sksurv_rsf_row = (
                run_external_sksurv_rsf_cv(s_cfg, ds=ds, folds=folds)
                if _is_contender_enabled(s_cfg, "python_sksurv_rsf")
                else None
            )
            if sksurv_rsf_row is not None:
                results.append(sksurv_rsf_row)
            sksurv_coxnet_row = (
                run_external_sksurv_coxnet_cv(s_cfg, ds=ds, folds=folds)
                if _is_contender_enabled(s_cfg, "python_sksurv_coxnet")
                else None
            )
            if sksurv_coxnet_row is not None:
                results.append(sksurv_coxnet_row)
            lifelines_enet_row = (
                run_external_lifelines_coxph_enet_cv(s_cfg, ds=ds, folds=folds)
                if _is_contender_enabled(s_cfg, "python_lifelines_coxph_enet")
                else None
            )
            if lifelines_enet_row is not None:
                results.append(lifelines_enet_row)
            glmnet_cox_row = (
                run_external_glmnet_cox_cv(s_cfg, ds=ds, folds=folds)
                if _is_contender_enabled(s_cfg, "r_glmnet_cox")
                else None
            )
            if glmnet_cox_row is not None:
                results.append(glmnet_cox_row)
            sksurv_gb_rows = (
                run_external_sksurv_gb_cv(s_cfg, ds=ds, folds=folds)
                if _is_contender_enabled(s_cfg, "python_sksurv_gb_coxph")
                else None
            )
            if sksurv_gb_rows is not None:
                if isinstance(sksurv_gb_rows, list):
                    results.extend(sksurv_gb_rows)
                else:
                    results.append(sksurv_gb_rows)
            lifelines_aft_rows = (
                run_external_lifelines_aft_cv(s_cfg, ds=ds, folds=folds)
                if _is_contender_enabled(s_cfg, "python_lifelines_weibull_aft")
                else None
            )
            if lifelines_aft_rows is not None:
                if isinstance(lifelines_aft_rows, list):
                    results.extend(lifelines_aft_rows)
                else:
                    results.append(lifelines_aft_rows)
            xgb_aft_row = (
                run_external_xgboost_aft_cv(s_cfg, ds=ds, folds=folds)
                if _is_contender_enabled(s_cfg, "python_xgboost_aft")
                else None
            )
            if xgb_aft_row is not None:
                results.append(xgb_aft_row)

    for s_cfg in scenarios:
        s_name = s_cfg["name"]
        has_rust = any(
            r.get("scenario_name") == s_name and r.get("contender") == "rust_gam"
            for r in results
        )
        if not has_rust:
            raise SystemExit(f"missing required rust_gam result for scenario '{s_name}'")

    # Hard guard: benchmark runs must fail in CI for blocking contender failures.
    # Non-blocking contenders still emit failed rows in the output for diagnostics.
    failed_blocking = [
        r for r in results if r.get("status") != "ok" and not _is_non_blocking_failure(r)
    ]
    if failed_blocking:
        msgs = []
        for r in failed_blocking:
            msgs.append(
                f"{r.get('contender','?')} / {r.get('scenario_name','?')}: {r.get('error','unknown error')}"
            )
        raise SystemExit("benchmark run failed:\n" + "\n".join(msgs))

    # Hard guard: benchmark outputs must remain CV-based and leakage-safe.
    for r in results:
        if r.get("status") != "ok":
            continue
        spec = str(r.get("model_spec", ""))
        if "CV" not in spec and "cv" not in spec:
            raise SystemExit(
                f"non-CV model result detected for {r.get('contender')} / {r.get('scenario_name')}: {spec}"
            )

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cv": {
            "n_splits": CV_SPLITS,
            "seed": CV_SEED,
            "leakage_safe": True,
        },
        "results": results,
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.out}")

    # Generate per-scenario comparison figures and bundle into a .zip.
    fig_dir = args.out.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    generate_scenario_figures(results, fig_dir)
    zip_path = args.out.parent / "figures.zip"
    zip_figure_dir(fig_dir, zip_path)
    print(f"Wrote {zip_path}")


# ---------------------------------------------------------------------------
# Per-scenario comparison figures
# ---------------------------------------------------------------------------

def _metric_display_config(family: str):
    """Return (metric_key, display_label, higher_is_better) tuples for a family."""
    if family == "binomial":
        return [
            ("auc", "AUC", True),
            ("logloss", "Log-Loss", False),
            ("brier", "Brier Score", False),
            ("nagelkerke_r2", "Nagelkerke R²", True),
        ]
    if family == "survival":
        return [
            ("auc", "C-Index", True),
            ("logloss", "Partial Log-Loss", False),
            ("brier", "Partial Brier", False),
            ("nagelkerke_r2", "Nagelkerke R²", True),
        ]
    # gaussian
    return [
        ("rmse", "RMSE", False),
        ("r2", "R²", True),
        ("logloss", "Gaussian Log-Loss", False),
        ("mae", "MAE", False),
    ]


def _short_contender_label(name: str) -> str:
    return (
        name.replace("python_", "py·")
        .replace("r_mgcv_", "mgcv·")
        .replace("r_", "R·")
        .replace("rust_", "rust·")
        .replace("_", " ")
    )


def generate_scenario_figures(results: list[dict], out_dir: Path) -> list[Path]:
    """Create one beautiful comparison PNG per scenario."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        print("WARNING: matplotlib not available — skipping figure generation.")
        return []

    # Group results by scenario.
    from collections import defaultdict
    by_scenario: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        if r.get("status") == "ok":
            by_scenario[r["scenario_name"]].append(r)

    # ---- Theme ----
    bg_dark = "#0d1117"
    bg_card = "#161b22"
    text_color = "#c9d1d9"
    grid_color = "#21262d"
    accent_colors = [
        "#58a6ff", "#3fb950", "#d29922", "#f85149",
        "#bc8cff", "#79c0ff", "#56d364", "#e3b341",
        "#ff7b72", "#d2a8ff", "#a5d6ff", "#7ee787",
        "#f2cc60", "#ffa198", "#cabffd", "#b1bac4",
    ]

    paths = []
    for scenario_name, rows in sorted(by_scenario.items()):
        if not rows:
            continue
        family = rows[0].get("family", "gaussian")
        metrics_cfg = _metric_display_config(family)
        # Filter to metrics that have at least one non-None value.
        active_metrics = []
        for key, label, hib in metrics_cfg:
            vals = [r.get(key) for r in rows if r.get(key) is not None]
            if vals:
                active_metrics.append((key, label, hib))
        if not active_metrics:
            continue

        n_metrics = len(active_metrics)
        contenders = [r["contender"] for r in rows]
        n_contenders = len(contenders)

        fig_height = max(3.2, 1.2 + 0.42 * n_contenders) * (n_metrics + 1)
        fig, axes = plt.subplots(
            n_metrics + 1, 1,
            figsize=(10, min(fig_height, 28)),
            facecolor=bg_dark,
            gridspec_kw={"hspace": 0.45},
        )
        if n_metrics + 1 == 1:
            axes = [axes]

        for ax in axes:
            ax.set_facecolor(bg_card)
            ax.tick_params(colors=text_color, which="both")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color(grid_color)
            ax.spines["left"].set_color(grid_color)
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            ax.title.set_color(text_color)

        short_labels = [_short_contender_label(c) for c in contenders]
        y_pos = list(range(n_contenders))
        colors = [accent_colors[i % len(accent_colors)] for i in range(n_contenders)]

        for idx, (key, label, higher_is_better) in enumerate(active_metrics):
            ax = axes[idx]
            vals = []
            for r in rows:
                v = r.get(key)
                vals.append(float(v) if v is not None else 0.0)

            bars = ax.barh(
                y_pos, vals, height=0.62,
                color=colors, edgecolor="none", alpha=0.88,
                zorder=3,
            )
            # Highlight the best value.
            valid_vals = [v for v in vals if v != 0.0]
            if valid_vals:
                if higher_is_better:
                    best_val = max(valid_vals)
                else:
                    best_val = min(valid_vals)
                for i, (bar, v) in enumerate(zip(bars, vals)):
                    if abs(v - best_val) < 1e-12 and v != 0.0:
                        bar.set_edgecolor("#f0f6fc")
                        bar.set_linewidth(1.8)
                        bar.set_alpha(1.0)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(short_labels, fontsize=8.5)
            ax.invert_yaxis()
            ax.set_xlabel(label, fontsize=9.5, fontweight="bold")
            direction_hint = "↑ higher is better" if higher_is_better else "↓ lower is better"
            ax.set_title(f"{label}  ({direction_hint})", fontsize=10, loc="left", pad=8)
            ax.grid(axis="x", color=grid_color, linewidth=0.5, zorder=0)

            # Value labels on bars.
            for bar, v in zip(bars, vals):
                if v == 0.0:
                    continue
                if abs(v) < 0.01:
                    fmt = f"{v:.4f}"
                elif abs(v) < 1.0:
                    fmt = f"{v:.4f}"
                elif abs(v) < 100:
                    fmt = f"{v:.3f}"
                else:
                    fmt = f"{v:.1f}"
                ax.text(
                    bar.get_width() + ax.get_xlim()[1] * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    fmt, va="center", ha="left",
                    fontsize=7.5, color=text_color, alpha=0.85,
                )

        # Timing subplot: fit + predict seconds.
        ax_time = axes[-1]
        fit_times = [float(r.get("fit_sec", 0)) for r in rows]
        pred_times = [float(r.get("predict_sec", 0)) for r in rows]
        bars_fit = ax_time.barh(
            y_pos, fit_times, height=0.42, label="Fit",
            color="#58a6ff", alpha=0.8, zorder=3,
        )
        bars_pred = ax_time.barh(
            [y + 0.42 for y in y_pos], pred_times, height=0.42, label="Predict",
            color="#3fb950", alpha=0.8, zorder=3,
        )
        ax_time.set_yticks([y + 0.21 for y in y_pos])
        ax_time.set_yticklabels(short_labels, fontsize=8.5)
        ax_time.invert_yaxis()
        ax_time.set_xlabel("Time (seconds)", fontsize=9.5, fontweight="bold")
        ax_time.set_title("Fit + Predict Time  (↓ lower is better)", fontsize=10, loc="left", pad=8)
        ax_time.grid(axis="x", color=grid_color, linewidth=0.5, zorder=0)
        legend = ax_time.legend(
            loc="lower right", fontsize=8,
            facecolor=bg_card, edgecolor=grid_color,
            labelcolor=text_color,
        )

        for bar, v in zip(bars_fit, fit_times):
            if v > 0:
                ax_time.text(
                    bar.get_width() + ax_time.get_xlim()[1] * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f}s", va="center", ha="left",
                    fontsize=7, color=text_color, alpha=0.75,
                )
        for bar, v in zip(bars_pred, pred_times):
            if v > 0:
                ax_time.text(
                    bar.get_width() + ax_time.get_xlim()[1] * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f}s", va="center", ha="left",
                    fontsize=7, color=text_color, alpha=0.75,
                )

        # Suptitle.
        fig.suptitle(
            f"Benchmark: {scenario_name}",
            fontsize=14, fontweight="bold",
            color="#f0f6fc", y=0.995,
        )
        fig.text(
            0.5, 0.002,
            f"family={family}  •  5-fold CV  •  {n_contenders} contenders",
            ha="center", fontsize=8, color=text_color, alpha=0.6,
        )

        out_path = out_dir / f"{scenario_name}.png"
        fig.savefig(
            out_path, dpi=180,
            facecolor=bg_dark, edgecolor="none",
            bbox_inches="tight", pad_inches=0.4,
        )
        plt.close(fig)
        paths.append(out_path)

    print(f"Generated {len(paths)} scenario figure(s) in {out_dir}/")
    return paths


def zip_figure_dir(fig_dir: Path, zip_path: Path) -> None:
    """Bundle all PNGs in fig_dir into a single .zip for GHA artifact download."""
    import zipfile
    pngs = sorted(fig_dir.glob("*.png"))
    if not pngs:
        print("No figures to zip.")
        return
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in pngs:
            zf.write(p, arcname=p.name)
    print(f"Zipped {len(pngs)} figure(s) -> {zip_path}")


if __name__ == "__main__":
    main()
