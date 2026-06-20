from __future__ import annotations

import functools
import math
import importlib.util
import re
import typing
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "bench" / "datasets"
CV_SPLITS = 5
_SYNTHETIC_PC_PANEL_SEED = 20260329
_SYNTHETIC_PC_PANEL_ROWS_PER_SUBPOP = 40
_SYNTHETIC_PC_PANEL: pd.DataFrame | None = None


def configure(context: dict[str, typing.Any]) -> None:
    # Loaded via bench/run_suite.py's `_import_run_suite_exports`, which
    # invokes this hook with the caller's globals so cross-module names
    # (`os`, `subprocess`, `tempfile`, `sys`, `threading`, `disk_usage`,
    # `monotonic`, `Fold`, the `HEARTBEAT_*` constants, the `_print_stderr` /
    # `_write_stream` helpers) resolve at call time. The other three
    # _run_suite_*.py helpers ship the same hook; without it, every function
    # below that touches one of those names raises `NameError` the moment
    # CI exercises the corresponding path (e.g. `bench_run_suite_mapping`
    # tests that drive `run_suite.main` end-to-end).
    globals().update(context)


_BENCH_RUST_LOADER: typing.Any = None


def _read_partitioned_dataset_csv(name: str) -> pd.DataFrame:
    part_dir = DATASET_DIR / f"{name}_parts"
    parts = sorted(part_dir.glob("part_*.csv"))
    if not parts:
        raise RuntimeError(f"{name} dataset parts missing in {part_dir}")
    return pd.concat((pd.read_csv(part) for part in parts), ignore_index=True)


def _load_bench_rust_loader() -> typing.Any:
    global _BENCH_RUST_LOADER
    if _BENCH_RUST_LOADER is not None:
        return _BENCH_RUST_LOADER
    loader_path = Path(__file__).resolve().parent / "_rust_loader.py"
    spec = importlib.util.spec_from_file_location("bench_rust_loader", loader_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load bench rust loader from {loader_path}")
    loader_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader_mod)
    _BENCH_RUST_LOADER = loader_mod
    return loader_mod


def _gamfit_rust() -> typing.Any:
    # The bench-shared loader walks every site-packages / sys.path
    # location AND the source tree, so it finds the compiled
    # `_rust*.so` whether it sits next to the importable `gamfit`
    # package (the source-tree shadow case) or only in the
    # pip-installed wheel's site-packages directory.
    return _load_bench_rust_loader().load_gamfit_rust_module(ROOT)


def _matrix_rows(matrix: typing.Any, y: typing.Any, prefix: str = "pc") -> list[dict[str, float]]:
    x = np.asarray(matrix, dtype=float)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    rows = []
    for i in range(x.shape[0]):
        row = {f"{prefix}{j + 1}": float(x[i, j]) for j in range(x.shape[1])}
        row["y"] = float(y_arr[i])
        rows.append(row)
    return rows


def _xy_payload(columns: typing.Any, *, prefix: str, family: str = "binomial") -> dict[str, typing.Any]:
    x = np.asarray(columns["x"], dtype=float)
    return {
        "family": family,
        "rows": _matrix_rows(x, columns["y"], prefix=prefix),
        "features": [f"{prefix}{i}" for i in range(1, x.shape[1] + 1)],
        "target": "y",
    }


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
    return typing.cast(pd.DataFrame, adjusted)


def _coerce_positive_survival_dataset_inplace(ds: dict[str, typing.Any], dataset_name: str) -> dict[str, typing.Any]:
    if ds.get("family") != "survival":
        return ds
    time_col = ds["time_col"]
    rows_df = pd.DataFrame(ds["rows"])
    adjusted = _coerce_positive_survival_times(rows_df, time_col=time_col, dataset_name=dataset_name)
    if adjusted is rows_df:
        return ds
    ds["rows"] = adjusted.to_dict(orient="records")
    return ds


def _fmt_kib(kib: typing.Any) -> typing.Any:
    if kib is None:
        return "n/a"
    gib = float(kib) / (1024.0 * 1024.0)
    return f"{gib:.2f} GiB"


def _fmt_cpu_total_pct(cpu_pct: typing.Any) -> typing.Any:
    if cpu_pct in (None, "n/a"):
        return "n/a"
    try:
        cpu_val = float(cpu_pct)
    except Exception:
        return "n/a"
    return f"{cpu_val:.1f}"


def _fmt_pct(numer: typing.Any, denom: typing.Any) -> typing.Any:
    try:
        if numer is None or denom in (None, 0):
            return "n/a"
        return f"{100.0 * float(numer) / float(denom):.1f}%"
    except Exception:
        return "n/a"


def _mem_used_kib(meminfo: typing.Any) -> typing.Any:
    total = meminfo.get("MemTotal")
    avail = meminfo.get("MemAvailable")
    if total is None or avail is None:
        return None
    used = int(total) - int(avail)
    return max(used, 0)


def _read_meminfo() -> typing.Any:
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


def _read_proc_status_kib(pid: typing.Any, key: typing.Any) -> typing.Any:
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


def _read_cgroup_memory_kib() -> typing.Any:
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


def _read_disk_usage_kib(path: typing.Any) -> typing.Any:
    try:
        usage = disk_usage(path)
    except Exception:
        return None, None, None
    return usage.total // 1024, usage.used // 1024, usage.free // 1024


def _read_ps_snapshot(pid: typing.Any) -> typing.Any:
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


def _collect_heartbeat_snapshot(proc: typing.Any, cmd_preview: typing.Any, stats: typing.Any, start: typing.Any) -> typing.Any:
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


def _heartbeat_loop(proc: typing.Any, cmd: typing.Any, stop_event: typing.Any, stats: typing.Any) -> None:
    cmd_preview = " ".join(str(x) for x in cmd[:5])
    if len(cmd) > 5:
        cmd_preview += " ..."
    start = monotonic()
    _print_stderr(_collect_heartbeat_snapshot(proc, cmd_preview, stats, start))
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
        _print_stderr(_collect_heartbeat_snapshot(proc, cmd_preview, stats, start))


def _workspace_tempdir(prefix: str) -> typing.Any:
    return tempfile.TemporaryDirectory(prefix=prefix, dir=str(BENCH_DIR))


def run_cmd(cmd: typing.Any, cwd: typing.Any=None, timeout_sec: typing.Any=None) -> typing.Any:
    # `timeout_sec` overrides the global BENCH_CMD_TIMEOUT_SEC for this one
    # invocation. It exists so an individual reference contender (e.g. a single
    # brms MCMC fold) can be bounded well below the per-shard wall budget and
    # fail just THAT fold — recorded and visible — instead of letting one slow
    # external fit consume the whole shard's GNU `timeout` and get the shard
    # killed (exit 124) with no per-fold attribution (#1390).
    effective_timeout = (
        float(timeout_sec)
        if timeout_sec is not None and float(timeout_sec) > 0
        else (float(_CMD_TIMEOUT_SEC) if _CMD_TIMEOUT_SEC > 0 else 0.0)
    )
    env = os.environ.copy()
    env.update(_SERIAL_ENV_OVERRIDES)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        env=env,
        bufsize=0,
    )

    out_buf: deque[str] = deque()
    err_buf: deque[str] = deque()
    out_chars = 0
    err_chars = 0
    hb_stop = threading.Event()
    hb_stats = {
        "peak_proc_rss_kib": 0,
        "peak_proc_hwm_kib": 0,
        "peak_proc_vsz_kib": 0,
        "samples": 0,
    }

    def _append_capped(buf: deque[str], text: str, current_chars: int) -> int:
        buf.append(text)
        current_chars += len(text)
        while current_chars > _MAX_CAPTURE_CHARS and buf:
            current_chars -= len(buf.popleft())
        return current_chars

    def _pump(pipe: typing.Any, sink: typing.Any, buf: typing.Any, char_count_name: typing.Any) -> None:
        nonlocal out_chars, err_chars
        sanitizer = _TerminalOutputSanitizer()
        try:
            while True:
                chunk = pipe.read(4096)
                if not chunk:
                    break
                text = sanitizer.feed(chunk.decode("utf-8", errors="replace"))
                _write_stream(sink, text)
                if char_count_name == "out":
                    out_chars = _append_capped(buf, text, out_chars)
                else:
                    err_chars = _append_capped(buf, text, err_chars)
        finally:
            tail = sanitizer.flush()
            if tail:
                _write_stream(sink, tail)
                if char_count_name == "out":
                    out_chars = _append_capped(buf, tail, out_chars)
                else:
                    err_chars = _append_capped(buf, tail, err_chars)
            try:
                pipe.close()
            except Exception:
                pass

    t_out = threading.Thread(target=_pump, args=(proc.stdout, sys.stdout, out_buf, "out"), daemon=True)
    t_err = threading.Thread(target=_pump, args=(proc.stderr, sys.stderr, err_buf, "err"), daemon=True)
    t_hb = threading.Thread(target=_heartbeat_loop, args=(proc, cmd, hb_stop, hb_stats), daemon=True)
    t_out.start()
    t_err.start()
    t_hb.start()
    timed_out = False
    try:
        if effective_timeout > 0:
            rc = proc.wait(timeout=effective_timeout)
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
            f"[HEARTBEAT] command-timeout rc=124 timeout_sec={effective_timeout:g} "
            f"pid={proc.pid} cmd='{(' '.join(str(x) for x in cmd[:5]) + (' ...' if len(cmd) > 5 else ''))}'\n"
        )
        err_chars = _append_capped(err_buf, timeout_msg, err_chars)
        _write_stream(sys.stderr, timeout_msg)
    _print_stderr(
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
        f"last_tmp_free={_fmt_kib(hb_stats.get('last_tmp_free_kib'))}/{_fmt_kib(hb_stats.get('last_tmp_total_kib'))}"
    )
    return rc, "".join(out_buf), "".join(err_buf)


def make_folds(y: np.ndarray, n_splits: int = 5, seed: int = 42, stratified: bool = False) -> typing.Any:
    raw_folds = _gamfit_rust().make_folds_indices(
        np.asarray(y, dtype=float).reshape(-1).tolist(),
        int(n_splits),
        int(seed),
        bool(stratified),
    )
    return [
        Fold(train_idx=np.asarray(train_idx, dtype=int), test_idx=np.asarray(test_idx, dtype=int))
        for train_idx, test_idx in raw_folds
    ]


def _evaluation_suffix(folds: list[Fold]) -> str:
    if len(folds) == 1:
        return "[holdout]"
    return f"[{len(folds)}-fold CV]"


def _evaluation_label_for_n_folds(n_folds: int) -> str:
    if n_folds <= 0:
        raise RuntimeError(f"cross-validation result requires at least one fold, got {n_folds}")
    if n_folds == 1:
        return "holdout"
    return f"{n_folds}-fold CV"


def auc_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(_gamfit_rust().auc_from_predictions(_flat_float_list(y), _flat_float_list(p)))


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(_gamfit_rust().brier_from_predictions(_flat_float_list(y), _flat_float_list(p)))


def log_loss_score(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    return float(_gamfit_rust().log_loss_from_predictions(_flat_float_list(y), _flat_float_list(p), float(eps)))


def nagelkerke_r2_score(
    y: np.ndarray,
    p: np.ndarray,
    *,
    null_mean: float,
    eps: float = 1e-12,
) -> float | None:
    value = _gamfit_rust().nagelkerke_r2_from_predictions(
        _flat_float_list(y),
        _flat_float_list(p),
        float(null_mean),
        float(eps),
    )
    return None if value is None else float(value)


def _survival_score_grid(train_df: pd.DataFrame, time_col: str) -> np.ndarray:
    return np.asarray(_gamfit_rust().survival_score_grid_from_times(_flat_float_list(train_df[time_col])), dtype=float)


def _repeat_survival_curve(surv: np.ndarray, n_rows: int) -> np.ndarray:
    return np.asarray(_gamfit_rust().repeat_survival_curve(_flat_float_list(surv), int(n_rows)), dtype=float)


def _survival_matrix_from_risk_calibration(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    time_col: str,
    event_col: str,
    train_risk: np.ndarray,
    test_risk: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        _gamfit_rust().survival_matrix_from_risk_calibration(
            _flat_float_list(train_df[time_col]),
            _flat_float_list(train_df[event_col]),
            _flat_float_list(train_risk),
            _flat_float_list(test_risk),
            _flat_float_list(grid),
        ),
        dtype=float,
    )


def _survival_null_curve_from_train(
    train_df: pd.DataFrame,
    *,
    time_col: str,
    event_col: str,
    grid: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        _gamfit_rust().survival_null_curve_from_train(
            _flat_float_list(train_df[time_col]),
            _flat_float_list(train_df[event_col]),
            _flat_float_list(grid),
        ),
        dtype=float,
    )


def survival_lifted_metrics(
    event_times: np.ndarray,
    events: np.ndarray,
    grid: np.ndarray,
    survival_matrix: np.ndarray,
    null_survival_matrix: np.ndarray | None = None,
    eps: float = 1e-12,
) -> dict[str, float | None]:
    null_matrix = None
    if null_survival_matrix is not None:
        null_matrix = np.asarray(null_survival_matrix, dtype=float)
        if null_matrix.ndim == 1:
            null_matrix = _repeat_survival_curve(null_matrix, len(np.asarray(event_times).reshape(-1)))
    return dict(
        _gamfit_rust().survival_lifted_metrics_from_predictions(
            _flat_float_list(event_times),
            _flat_float_list(events),
            _flat_float_list(grid),
            np.asarray(survival_matrix, dtype=float),
            null_matrix,
            float(eps),
        )
    )


def score_survival_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    time_col: str,
    event_col: str,
    risk_score: np.ndarray,
    train_risk_score: np.ndarray | None = None,
    survival_grid: np.ndarray | None = None,
    survival_matrix: np.ndarray | None = None,
) -> dict[str, float | None]:
    event_times = test_df[time_col].to_numpy(dtype=float)
    events = test_df[event_col].to_numpy(dtype=float)
    if survival_grid is not None and survival_matrix is not None:
        grid = np.asarray(survival_grid, dtype=float)
        surv = np.asarray(survival_matrix, dtype=float)
    else:
        if train_risk_score is None:
            raise ValueError("score_survival_fold requires train_risk_score when survival_matrix is not provided")
        grid = _survival_score_grid(train_df, time_col)
        surv = _survival_matrix_from_risk_calibration(
            train_df,
            test_df,
            time_col=time_col,
            event_col=event_col,
            train_risk=np.asarray(train_risk_score, dtype=float).reshape(-1),
            test_risk=np.asarray(risk_score, dtype=float).reshape(-1),
            grid=grid,
        )
    null_surv = _repeat_survival_curve(
        _survival_null_curve_from_train(train_df, time_col=time_col, event_col=event_col, grid=grid),
        len(test_df),
    )
    proper_metrics = survival_lifted_metrics(
        event_times,
        events,
        grid,
        surv,
        null_survival_matrix=null_surv,
    )
    metrics: dict[str, float | None] = {
        "auc": _lifelines_cindex_from_risk(event_times, risk_score, events),
        "brier": proper_metrics["brier"],
        "logloss": proper_metrics["logloss"],
        "nagelkerke_r2": proper_metrics["nagelkerke_r2"],
        "predict_horizon": None,
    }
    return metrics


def gaussian_log_loss_score(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray | float, eps: float = 1e-12) -> float:
    return float(_gamfit_rust().gaussian_log_loss_from_predictions(_flat_float_list(y), _flat_float_list(mu), _sigma_float_list(sigma), float(eps)))


def gaussian_prediction_scores(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray | float,
) -> dict[str, float | None]:
    return dict(_gamfit_rust().gaussian_prediction_scores_from_predictions(_flat_float_list(y), _flat_float_list(mu), _sigma_float_list(sigma)))


def zscore_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = train_df.copy()
    te = test_df.copy()
    if feature_cols:
        tr_values, te_values = _gamfit_rust().zscore_train_test_arrays(
            tr[feature_cols].to_numpy(dtype=float),
            te[feature_cols].to_numpy(dtype=float),
        )
        tr.loc[:, feature_cols] = np.asarray(tr_values, dtype=float)
        te.loc[:, feature_cols] = np.asarray(te_values, dtype=float)
    return tr, te


def _survival_risk_from_rust_pred(pred_df: pd.DataFrame) -> tuple[np.ndarray, str]:
    # Canonical survival ranking: model-native risk score (higher => earlier failure).
    for col in ("failure_prob", "risk_score", "eta"):
        if col in pred_df.columns:
            return pred_df[col].to_numpy(dtype=float), col
    raise RuntimeError(
        "rust survival prediction output missing required risk column; "
        "expected one of: failure_prob, risk_score, eta"
    )


def _lifelines_cindex_from_risk(event_times: np.ndarray, risk_score: np.ndarray, events: np.ndarray) -> float:
    # Convert risk (higher => earlier failure) to lifelines survival score.
    concordance_index = _require_lifelines_concordance_index()
    return float(concordance_index(event_times, -risk_score, event_observed=events))


def _survival_eval_horizon(train_df: pd.DataFrame, time_col: str) -> float:
    # Fold-specific reference horizon from train data only, used only when a
    # backend prediction API requires an explicit time input.
    horizon = float(np.median(train_df[time_col].to_numpy(dtype=float)))
    if (not np.isfinite(horizon)) or horizon <= 0.0:
        horizon = 1.0
    return horizon


def _rust_survival_fit_options_for_scenario(scenario_name: typing.Any) -> typing.Any:
    # Survival time effects must use a structurally monotone basis so the
    # fitted cumulative baseline cannot violate survival semantics.
    if scenario_name in {"icu_survival_death", "icu_survival_los"}:
        return {
            "time_basis": "ispline",
            "time_degree": 3,
            "time_num_internal_knots": 10,
            "time_smooth_lambda": 5e-2,
            "ridge_lambda": 1e-6,
        }
    if scenario_name in {"heart_failure_survival", "cirrhosis_survival"}:
        return {
            "time_basis": "ispline",
            "time_degree": 3,
            "time_num_internal_knots": 8,
            "time_smooth_lambda": 1e-2,
            "ridge_lambda": 1e-6,
        }
    return {
        "time_basis": "ispline",
        "time_degree": 3,
        "time_num_internal_knots": 8,
        "time_smooth_lambda": 1e-2,
        "ridge_lambda": 1e-6,
    }


def _rust_survival_fit_cli_args(scenario_name: str) -> list[str]:
    cfg = _rust_survival_fit_options_for_scenario(scenario_name)
    args: list[str] = []
    for key in (
        "time_basis",
        "time_degree",
        "time_num_internal_knots",
        "time_smooth_lambda",
        "ridge_lambda",
    ):
        if key not in cfg:
            continue
        cli_key = "--" + key.replace("_", "-")
        args.extend([cli_key, str(cfg[key])])
    return args


def _load_lidar_dataset() -> typing.Any:
    d = pd.read_csv(DATASET_DIR / "lidar.csv")
    d = d[["range", "logratio"]].dropna()
    rows = [{"range": float(r), "y": float(y)} for r, y in zip(d["range"], d["logratio"])]
    return {
        "family": "gaussian",
        "rows": rows,
        "features": ["range"],
        "target": "y",
    }


def _load_bone_dataset() -> typing.Any:
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


def _load_prostate_dataset() -> typing.Any:
    d = pd.read_csv(DATASET_DIR / "prostate.csv")
    d = d[["pc1", "pc2", "y"]].dropna()
    rows = [{"pc1": float(a), "pc2": float(b), "y": float(y)} for a, b, y in zip(d["pc1"], d["pc2"], d["y"])]
    return {
        "family": "binomial",
        "rows": rows,
        "features": ["pc1", "pc2"],
        "target": "y",
    }


def _load_wine_dataset() -> typing.Any:
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


def _load_wine_temp_vs_year_dataset() -> typing.Any:
    d = pd.read_csv(DATASET_DIR / "wine.csv")
    d = d[["year", "s_temp"]].dropna()
    rows = [{"year": float(y), "s_temp": float(t)} for y, t in zip(d["year"], d["s_temp"])]
    return {
        "family": "gaussian",
        "rows": rows,
        "features": ["year"],
        "target": "s_temp",
    }


def _load_wine_price_vs_temp_dataset() -> typing.Any:
    d = pd.read_csv(DATASET_DIR / "wine.csv")
    d = d[["s_temp", "price"]].replace({"NA": np.nan}).dropna()
    rows = [{"temp": float(t), "price": float(p)} for t, p in zip(d["s_temp"], d["price"])]
    return {
        "family": "gaussian",
        "rows": rows,
        "features": ["temp"],
        "target": "price",
    }


def _load_horse_dataset() -> typing.Any:
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


def _parse_f64_opt(raw: typing.Any) -> typing.Any:
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


def _load_cirrhosis_survival_dataset() -> typing.Any:
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
        .strip()
    )
    cleaned = re.sub(r"\s+(?:EST|EDT)\s*$", "", cleaned, flags=re.IGNORECASE)
    dt = pd.to_datetime(cleaned, format="%m/%d/%Y %I %p", errors="coerce")
    if pd.isna(dt):
        raise ValueError(f"failed to parse timestamp '{ts}'")
    return float(dt.hour)


def _load_us48_demand_dataset(filename: str) -> typing.Any:
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


def _load_haberman_dataset() -> typing.Any:
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


def _load_icu_survival_death_dataset() -> typing.Any:
    d = _read_partitioned_dataset_csv("icu_survival_death")
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


def _load_icu_survival_los_dataset() -> typing.Any:
    d = _read_partitioned_dataset_csv("icu_survival_los")
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


def _load_heart_failure_survival_dataset() -> typing.Any:
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


def _synthetic_binomial_dataset(n: typing.Any, p: typing.Any, seed: typing.Any) -> typing.Any:
    columns = _gamfit_rust().synthetic_binomial_columns(int(n), max(int(p), 3), int(seed))
    return _xy_payload(columns, prefix="x")


_CANONICAL_SYNTHETIC_BINOMIAL_SCENARIOS = {
    "small_dense": {"n": 1000, "p": 10, "seed": 7},
    "medium": {"n": 50000, "p": 50, "seed": 11},
    "pathological_ill_conditioned": {"n": 50000, "p": 80, "seed": 17},
}


def _synthetic_geo_disease_dataset(n: typing.Any=4000, seed: typing.Any=20260226) -> typing.Any:
    columns = _gamfit_rust().synthetic_geo_disease_columns(int(n), int(seed))
    return _xy_payload(columns, prefix="pc")


def _synthetic_continuous_order_dataset(
    *,
    mode: str,
    n: int = 512,
    seed: int = 20260501,
    true_nu: float | None = None,
    true_kappa2: float | None = None,
) -> dict[str, typing.Any]:
    columns = _gamfit_rust().synthetic_continuous_order_columns(
        str(mode),
        int(n),
        int(seed),
        None if true_nu is None else float(true_nu),
        None if true_kappa2 is None else float(true_kappa2),
    )
    x = np.asarray(columns["x"], dtype=float).reshape(-1)
    y = np.asarray(columns["y"], dtype=float).reshape(-1)
    rows = [{"x": float(xi), "y": float(yi)} for xi, yi in zip(x, y)]
    expected = {
        "fractional": ["Ok", "NonMaternRegime"],
        "rough": ["NonMaternRegime", "FirstOrderLimit", "IntrinsicLimit"],
        "smooth": ["Ok", "IntrinsicLimit"],
    }.get(str(mode))
    if expected is None:
        raise RuntimeError(f"unsupported continuous-order synthetic mode '{mode}'")
    out: dict[str, typing.Any] = {
        "family": "gaussian",
        "rows": rows,
        "features": ["x"],
        "target": "y",
        "continuous_order_expected_statuses": expected,
    }
    if true_nu is not None:
        out["continuous_order_true_nu"] = float(true_nu)
    if true_kappa2 is not None:
        out["continuous_order_true_kappa2"] = float(true_kappa2)
    return out


def _synthetic_thread3_admixture_cliff_dataset(n: typing.Any=6000, seed: typing.Any=20260601) -> typing.Any:
    coeffs = np.array([1.0, 0.35, -0.20, 0.10], dtype=float)
    out = _xy_payload(
        _gamfit_rust().synthetic_thread3_admixture_cliff_columns(int(n), int(seed)),
        prefix="pc",
    )
    out.update({
        "thread3_cliff_coefficients": {
            "pc1": float(coeffs[0]),
            "pc2": float(coeffs[1]),
            "pc3": float(coeffs[2]),
            "pc4": float(coeffs[3]),
        },
        "thread3_cliff_jump": 3.8,
        "thread3_cliff_sharpness": 16.0,
    })
    return out


def _synthetic_geo_disease_eas_dataset(n: typing.Any=6000, seed: typing.Any=20260301, n_pcs: typing.Any=16) -> typing.Any:
    n_pcs = int(max(3, n_pcs))
    columns = _gamfit_rust().synthetic_geo_disease_eas_columns(int(n), int(seed), n_pcs)
    return _xy_payload(columns, prefix="pc")


def _geo_disease_eas_scenario_cfg(name: typing.Any) -> typing.Any:
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
    if basis_code == "tp":
        joint_pcs = _fixed_joint_spatial_pc_count("geo_disease", n_pcs)
        return {
            "smooth_basis": "thinplate",
            "smooth_cols": [f"pc{i}" for i in range(1, joint_pcs + 1)],
            "linear_cols": [],
            "knots": knots,
            "basis_code": basis_code,
            "n_pcs": n_pcs,
        }
    if basis_code == "duchon":
        joint_pcs = _fixed_joint_spatial_pc_count("geo_disease", n_pcs)
        return {
            "smooth_basis": "duchon",
            "smooth_cols": [f"pc{i}" for i in range(1, joint_pcs + 1)],
            "linear_cols": [],
            "knots": knots,
            "basis_code": basis_code,
            "n_pcs": n_pcs,
        }
    if basis_code == "matern":
        joint_pcs = _fixed_joint_spatial_pc_count("geo_disease", n_pcs)
        return {
            "smooth_basis": "matern",
            "smooth_cols": [f"pc{i}" for i in range(1, joint_pcs + 1)],
            "linear_cols": [],
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


def _scenario_downsample_factor(name: str) -> int | None:
    m = re.search(r"_downsample([0-9]+)x(?:_holdout)?$", str(name))
    if m is None:
        return None
    return max(1, int(m.group(1)))


def _fixed_joint_spatial_pc_count(family: str, n_pcs: int) -> int:
    """Number of PCs to include in the joint smooth for `family`.

    The joint-PC contract says PCs always enter the model as a single multi-D
    object — so this returns the FULL `n_pcs`. Thin-plate splines used to
    require a separate cap because canonical TPS in d dimensions has a
    polynomial nullspace of size C(d+m-1, d) with m=⌊d/2⌋+1, which is
    735_471 at d=16. The Rust basis builder now auto-promotes infeasible
    canonical-TPS requests to a pure Duchon spline (the proper Riesz-
    fractional generalization with finite kernel at r=0 and a small
    polynomial nullspace), so all bases — including thin-plate — can use
    the full PC count uniformly.
    """
    n_pcs = int(max(1, n_pcs))
    family = str(family)
    if family in {"geo_disease", "papuan_oce", "geo_subpop16", "geo_latlon"}:
        return n_pcs
    raise RuntimeError(f"unsupported joint spatial benchmark family: {family}")


def _papuan_oce_scenario_cfg(name: typing.Any) -> typing.Any:
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
    joint_pcs = _fixed_joint_spatial_pc_count("papuan_oce", n_pcs)
    return {
        "smooth_basis": smooth_basis,
        "smooth_cols": [f"pc{i}" for i in range(1, joint_pcs + 1)],
        "linear_cols": [],
        "knots": knots,
        "basis_code": basis_code,
        "n_pcs": n_pcs,
    }


def _synthetic_papuan_oce_dataset(n: typing.Any=6000, seed: typing.Any=20260315, n_pcs: typing.Any=16) -> typing.Any:
    columns = _gamfit_rust().synthetic_papuan_oce_columns(int(n), int(seed), max(3, int(n_pcs)))
    return _xy_payload(columns, prefix="pc")


def _geo_subpop16_scenario_cfg(name: typing.Any) -> typing.Any:
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
    joint_pcs = _fixed_joint_spatial_pc_count("geo_subpop16", 16)
    return {
        "smooth_basis": smooth_basis,
        "smooth_cols": [f"pc{i}" for i in range(1, joint_pcs + 1)],
        "linear_cols": [],
        "knots": knots,
        "basis_code": basis_code,
        "n_pcs": 16,
    }


def _geo_latlon_scenario_cfg(name: typing.Any) -> typing.Any:
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
    joint_pcs = _fixed_joint_spatial_pc_count("geo_latlon", 6)
    return {
        "mode_code": mode_code,
        "smooth_basis": smooth_basis,
        "smooth_cols": [f"pc{i}" for i in range(1, joint_pcs + 1)],
        "linear_cols": [],
        "knots": knots,
        "basis_code": basis_code,
        "n_pcs": 6,
    }


def _synthetic_hgdp_1kg_pc_panel() -> typing.Any:
    global _SYNTHETIC_PC_PANEL
    if _SYNTHETIC_PC_PANEL is not None:
        return _SYNTHETIC_PC_PANEL.copy()

    columns = _gamfit_rust().synthetic_hgdp_pc_panel_columns(_SYNTHETIC_PC_PANEL_SEED)
    pc = np.asarray(columns["pc"], dtype=float)
    pc_cols = [f"PC{i}" for i in range(1, 17)]
    panel = pd.DataFrame(
        {
            "sample_id": list(columns["sample_id"]),
            "Superpopulation": list(columns["Superpopulation"]),
            "Subpopulation": list(columns["Subpopulation"]),
            "Latitude": np.asarray(columns["Latitude"], dtype=float),
            "Longitude": np.asarray(columns["Longitude"], dtype=float),
            **{col: pc[:, i] for i, col in enumerate(pc_cols)},
        }
    )
    _SYNTHETIC_PC_PANEL = panel
    return panel.copy()


def _load_hgdp_pc_with_imputed_latlon() -> typing.Any:
    raw = _synthetic_hgdp_1kg_pc_panel()
    pc_cols = [f"PC{i}" for i in range(1, 17)]
    required = {"sample_id", "Superpopulation", "Subpopulation", "Latitude", "Longitude", *pc_cols}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise RuntimeError(f"synthetic hgdp_1kg pc panel is missing required columns: {missing}")

    d = raw[["sample_id", "Superpopulation", "Subpopulation", "Latitude", "Longitude", *pc_cols]].copy()
    d["Subpopulation"] = d["Subpopulation"].astype(str)
    d["Superpopulation"] = d["Superpopulation"].astype(str)
    for c in pc_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=pc_cols).copy()
    if d.empty:
        raise RuntimeError("synthetic hgdp_1kg pc panel has no complete rows for PC1..PC16")

    centroids = d.groupby("Subpopulation", dropna=False)[pc_cols].mean(numeric_only=True)
    sub_latlon_known = (
        d.dropna(subset=["Latitude", "Longitude"])
        .groupby("Subpopulation", dropna=False)[["Latitude", "Longitude"]]
        .mean(numeric_only=True)
    )

    known_subpops = [sp for sp in centroids.index if sp in sub_latlon_known.index]
    if len(known_subpops) < 2:
        raise RuntimeError(
            "synthetic hgdp_1kg pc panel must contain at least two subpopulations with real "
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
    for row in d.itertuples(index=False):
        lat_raw = row.Latitude
        lon_raw = row.Longitude
        if np.isfinite(lat_raw) and np.isfinite(lon_raw):
            sample_lat.append(float(lat_raw))
            sample_lon.append(float(lon_raw))
        else:
            lat_i, lon_i = subpop_to_latlon[str(row.Subpopulation)]
            sample_lat.append(lat_i)
            sample_lon.append(lon_i)
    d["lat_imputed"] = np.asarray(sample_lat, dtype=float)
    d["lon_imputed"] = np.asarray(sample_lon, dtype=float)

    return d


def _geo_latlon_dataset(mode_code: typing.Any, seed: typing.Any=20260401, prevalence_min: typing.Any=0.01, prevalence_max: typing.Any=0.10) -> typing.Any:
    mode_code = str(mode_code)
    if mode_code not in {"superpopnoise", "equatornoise"}:
        raise RuntimeError(f"unsupported geo_latlon mode: {mode_code}")
    d = _load_hgdp_pc_with_imputed_latlon().copy()
    superpops = sorted(d["Superpopulation"].astype(str).unique().tolist())
    superpop_code = {name: idx for idx, name in enumerate(superpops)}
    y = np.asarray(
        _gamfit_rust().synthetic_geo_latlon_response(
            mode_code,
            d["Superpopulation"].astype(str).map(superpop_code).to_numpy(dtype=int).tolist(),
            d["lat_imputed"].to_numpy(dtype=float).tolist(),
            d["lon_imputed"].to_numpy(dtype=float).tolist(),
            int(seed),
            float(prevalence_min),
            float(prevalence_max),
        ),
        dtype=float,
    )

    rows = []
    for pos, row in enumerate(d.itertuples(index=False)):
        out = {f"pc{j}": float(getattr(row, f"PC{j}")) for j in range(1, 7)}
        out["y"] = float(y[pos])
        rows.append(out)
    return {
        "family": "binomial",
        "rows": rows,
        "features": [f"pc{i}" for i in range(1, 7)],
        "target": "y",
    }


def _geo_subpop16_dataset(seed: typing.Any=20260330, prevalence_min: typing.Any=0.02, prevalence_max: typing.Any=0.40) -> typing.Any:
    raw = _synthetic_hgdp_1kg_pc_panel()
    pc_cols = [f"PC{i}" for i in range(1, 17)]
    required = {"Subpopulation", *pc_cols}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise RuntimeError(f"synthetic hgdp_1kg pc panel is missing required columns: {missing}")

    d = raw[["Subpopulation", *pc_cols]].dropna().copy()
    d["Subpopulation"] = d["Subpopulation"].astype(str)
    if d.empty:
        raise RuntimeError("synthetic hgdp_1kg pc panel has no complete rows for Subpopulation + PC1..PC16")

    subpops = sorted(d["Subpopulation"].unique().tolist())
    if len(subpops) < 2:
        raise RuntimeError("need at least two subpopulations for prevalence simulation")

    subpop_code = {name: idx for idx, name in enumerate(subpops)}
    d["y"] = np.asarray(
        _gamfit_rust().synthetic_geo_subpop_response(
            d["Subpopulation"].map(subpop_code).to_numpy(dtype=int).tolist(),
            int(seed),
            float(prevalence_min),
            float(prevalence_max),
            0.25,
            0.85,
            False,
        ),
        dtype=float,
    )

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


def _geo_subpop16_randomprev_randomscale_dataset(
    seed: typing.Any=20260701,
    prevalence_min: typing.Any=0.02,
    prevalence_max: typing.Any=0.40,
    noise_scale_min: typing.Any=0.25,
    noise_scale_max: typing.Any=0.85,
) -> typing.Any:
    raw = _synthetic_hgdp_1kg_pc_panel()
    pc_cols = [f"PC{i}" for i in range(1, 17)]
    required = {"Subpopulation", *pc_cols}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise RuntimeError(f"synthetic hgdp_1kg pc panel is missing required columns: {missing}")

    d = raw[["Subpopulation", *pc_cols]].dropna().copy()
    d["Subpopulation"] = d["Subpopulation"].astype(str)
    if d.empty:
        raise RuntimeError("synthetic hgdp_1kg pc panel has no complete rows for Subpopulation + PC1..PC16")

    subpops = sorted(d["Subpopulation"].unique().tolist())
    if len(subpops) < 2:
        raise RuntimeError("need at least two subpopulations for prevalence simulation")

    subpop_code = {name: idx for idx, name in enumerate(subpops)}
    d["y"] = np.asarray(
        _gamfit_rust().synthetic_geo_subpop_response(
            d["Subpopulation"].map(subpop_code).to_numpy(dtype=int).tolist(),
            int(seed),
            float(prevalence_min),
            float(prevalence_max),
            float(noise_scale_min),
            float(noise_scale_max),
            True,
        ),
        dtype=float,
    )

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


def _validate_dataset_schema(ds: typing.Any, scenario_name: typing.Any) -> None:
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


def _resolve_scenario_loader(s: typing.Any) -> typing.Callable[[], typing.Any]:
    """Dispatch a scenario spec to a zero-arg loader callable, without invoking it.

    Each branch binds the spec-derived arguments via ``functools.partial`` so the
    returned callable can be invoked later (during data generation) OR discarded
    (during pure-Python schema validation). Raises ``RuntimeError`` if the
    scenario name does not match any registered loader. This is the single
    source of truth for the name → loader mapping; ``validate_scenario_schema``
    and ``_dataset_for_scenario_unvalidated`` both go through here.
    """
    name = s["name"]
    if name in {"small_dense", "medium", "pathological_ill_conditioned"}:
        cfg = _CANONICAL_SYNTHETIC_BINOMIAL_SCENARIOS[name]
        return functools.partial(
            _synthetic_binomial_dataset,
            s.get("n", cfg["n"]),
            s.get("p", cfg["p"]),
            s.get("seed", cfg["seed"]),
        )
    if name.startswith("geo_disease_eas3_"):
        return functools.partial(
            _synthetic_geo_disease_eas_dataset,
            s.get("n", 6000), s.get("seed", 20260301), n_pcs=3,
        )
    if name.startswith("geo_disease_eas_"):
        return functools.partial(
            _synthetic_geo_disease_eas_dataset,
            s.get("n", 6000), s.get("seed", 20260301),
        )
    if name.startswith("papuan_oce4_"):
        return functools.partial(
            _synthetic_papuan_oce_dataset,
            s.get("n", 6000), s.get("seed", 20260315), n_pcs=4,
        )
    if name.startswith("papuan_oce_"):
        return functools.partial(
            _synthetic_papuan_oce_dataset,
            s.get("n", 6000), s.get("seed", 20260315), n_pcs=16,
        )
    if name == "geo_subpop16_randomprev_randomscale_duchonfull_k50":
        return functools.partial(
            _geo_subpop16_randomprev_randomscale_dataset,
            seed=s.get("seed", 20260701),
            prevalence_min=s.get("prevalence_min", 0.02),
            prevalence_max=s.get("prevalence_max", 0.40),
            noise_scale_min=s.get("noise_scale_min", 0.25),
            noise_scale_max=s.get("noise_scale_max", 0.85),
        )
    if name.startswith("geo_subpop16_margslope_aniso_duchon16d_"):
        return functools.partial(
            _geo_subpop16_randomprev_randomscale_dataset,
            seed=s.get("seed", 20260702),
            prevalence_min=s.get("prevalence_min", 0.02),
            prevalence_max=s.get("prevalence_max", 0.40),
            noise_scale_min=s.get("noise_scale_min", 0.25),
            noise_scale_max=s.get("noise_scale_max", 0.85),
        )
    if name.startswith("geo_subpop16_"):
        return functools.partial(
            _geo_subpop16_dataset,
            seed=s.get("seed", 20260330),
            prevalence_min=s.get("prevalence_min", 0.02),
            prevalence_max=s.get("prevalence_max", 0.40),
        )
    geo_latlon_cfg = _geo_latlon_scenario_cfg(name)
    if geo_latlon_cfg is not None:
        return functools.partial(
            _geo_latlon_dataset,
            mode_code=geo_latlon_cfg["mode_code"],
            seed=s.get("seed", 20260401),
            prevalence_min=s.get("prevalence_min", 0.01),
            prevalence_max=s.get("prevalence_max", 0.10),
        )
    if name.startswith("geo_disease_"):
        return functools.partial(
            _synthetic_geo_disease_dataset,
            s.get("n", 4000), s.get("seed", 20260226),
        )
    if name == "continuous_order_fractional_spde_nu18":
        return functools.partial(
            _synthetic_continuous_order_dataset,
            mode="fractional",
            n=s.get("n", 512),
            seed=s.get("seed", 20260501),
            true_nu=s.get("true_nu", 1.8),
            true_kappa2=s.get("true_kappa2", 0.7),
        )
    if name == "continuous_order_boundary_rough":
        return functools.partial(
            _synthetic_continuous_order_dataset,
            mode="rough",
            n=s.get("n", 512),
            seed=s.get("seed", 20260502),
        )
    if name == "continuous_order_boundary_smooth":
        return functools.partial(
            _synthetic_continuous_order_dataset,
            mode="smooth",
            n=s.get("n", 512),
            seed=s.get("seed", 20260503),
        )
    if name == "thread3_admixture_cliff":
        return functools.partial(
            _synthetic_thread3_admixture_cliff_dataset,
            n=s.get("n", 6000),
            seed=s.get("seed", 20260601),
        )
    if name == "lidar_semipar":
        return _load_lidar_dataset
    if name == "bone_gamair":
        return _load_bone_dataset
    if name == "prostate_gamair":
        return _load_prostate_dataset
    if name == "wine_gamair":
        return _load_wine_dataset
    if name == "wine_temp_vs_year":
        return _load_wine_temp_vs_year_dataset
    if name == "wine_price_vs_temp":
        return _load_wine_price_vs_temp_dataset
    if name == "horse_colic":
        return _load_horse_dataset
    if name == "us48_demand_5day":
        return functools.partial(_load_us48_demand_dataset, "five_day.csv")
    if name == "us48_demand_31day":
        return functools.partial(_load_us48_demand_dataset, "31_day.csv")
    if name == "haberman_5yr":
        return _load_haberman_dataset
    if name == "icu_survival_death":
        return _load_icu_survival_death_dataset
    if name == "icu_survival_los":
        return _load_icu_survival_los_dataset
    if name == "heart_failure_survival":
        return _load_heart_failure_survival_dataset
    if name == "cirrhosis_survival":
        return _load_cirrhosis_survival_dataset
    raise RuntimeError(f"No scenario-specific dataset loader configured for '{name}'")


def _dataset_for_scenario_unvalidated(s: typing.Any) -> typing.Any:
    return _resolve_scenario_loader(s)()


def validate_scenario_schema(s: typing.Any) -> None:
    """Verify that scenario spec ``s`` is well-formed and dispatches to a known loader.

    This is the lightweight preflight used by CI before spinning up the benchmark
    matrix: it executes only Python-level dispatch logic, does NOT generate any
    synthetic data, and does NOT require the Rust extension or external CSV
    fixtures to be present. If the spec is missing required fields or the name
    is unknown, this raises ``RuntimeError`` with a scenario-identifying message.
    """
    if not isinstance(s, dict):
        raise RuntimeError(f"scenario spec must be a dict, got {type(s).__name__}")
    name = s.get("name")
    if not isinstance(name, str) or not name:
        raise RuntimeError(f"scenario spec missing required 'name' string: {s!r}")
    _resolve_scenario_loader(s)


def _downsample_binomial_dataset(
    ds: dict[str, typing.Any],
    *,
    n: int,
    seed: int,
    min_class_count: int,
) -> dict[str, typing.Any]:
    rows = list(ds["rows"])
    target = str(ds["target"])
    n = int(n)
    min_class_count = int(min_class_count)
    if n <= 0 or n > len(rows):
        raise RuntimeError(f"invalid downsample size {n} for dataset with {len(rows)} rows")
    if min_class_count < 1:
        raise RuntimeError(f"min_class_count must be >= 1; got {min_class_count}")
    if n < 2 * min_class_count:
        raise RuntimeError(
            f"downsample size {n} cannot support at least {min_class_count} rows from each class"
        )

    y = np.array([float(r[target]) > 0.5 for r in rows], dtype=bool)
    pos_idx = np.flatnonzero(y)
    neg_idx = np.flatnonzero(~y)
    if pos_idx.size < min_class_count or neg_idx.size < min_class_count:
        raise RuntimeError(
            "source dataset cannot support stratified downsample with required class support: "
            f"positives={int(pos_idx.size)}, negatives={int(neg_idx.size)}, "
            f"required_per_class={min_class_count}"
        )

    rng = np.random.default_rng(int(seed))
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    pos_rate = float(pos_idx.size / len(rows))
    n_pos = int(round(n * pos_rate))
    n_pos = min(max(n_pos, min_class_count), n - min_class_count)
    n_neg = n - n_pos
    chosen = np.concatenate([pos_idx[:n_pos], neg_idx[:n_neg]])
    rng.shuffle(chosen)

    out = dict(ds)
    out["rows"] = [rows[int(i)] for i in chosen]
    return out


def dataset_for_scenario(s: typing.Any) -> dict[str, typing.Any]:
    name = s["name"]
    ds = _dataset_for_scenario_unvalidated(s)
    downsample_factor = _scenario_downsample_factor(name)
    if downsample_factor is not None and name.startswith("geo_disease_eas"):
        full_n = int(max(6000, int(s.get("n", 6000)) * downsample_factor))
        n_pcs = 3 if name.startswith("geo_disease_eas3_") else 16
        full_ds = _synthetic_geo_disease_eas_dataset(full_n, s.get("seed", 20260301), n_pcs=n_pcs)
        n_splits = int(s.get("cv_splits", CV_SPLITS))
        min_class_count = 2 if n_splits == 1 else n_splits
        ds = _downsample_binomial_dataset(
            full_ds,
            n=int(s.get("n", full_n)),
            seed=int(s.get("seed", 20260301)),
            min_class_count=min_class_count,
        )
    if "cv_splits" in s:
        ds["_cv_splits"] = int(s["cv_splits"])
    _validate_dataset_schema(ds, scenario_name=name)
    return ds
