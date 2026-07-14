#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import tempfile
import traceback
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


BENCH_DIR = Path(__file__).resolve().parent
ROOT = BENCH_DIR.parents[1]
DEFAULT_CONFIG = BENCH_DIR / "large_scale.yml"
HEARTBEAT_INTERVAL_SEC = 15.0
HEARTBEAT_INITIAL_WINDOW_SEC = 2.0
HEARTBEAT_INITIAL_INTERVAL_SEC = 0.25
MAX_CAPTURE_CHARS = 200000
_OUTPUT_LOCK = threading.Lock()
_SURVIVAL_CALIBRATION: Any = None


def _survival_calibration() -> Any:
    global _SURVIVAL_CALIBRATION
    if _SURVIVAL_CALIBRATION is not None:
        return _SURVIVAL_CALIBRATION
    module_path = BENCH_DIR.parent / "_survival_calibration.py"
    spec = importlib.util.spec_from_file_location("bench_survival_calibration", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load survival calibration helpers from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _SURVIVAL_CALIBRATION = module
    return module


class _TerminalOutputSanitizer:
    def __init__(self) -> None:
        self._state = "normal"
        self._line_start = True
        self._pending_indent = ""

    def feed(self, text: str) -> str:
        out: list[str] = []
        for ch in text:
            state = self._state
            if state == "normal":
                if ch == "\x1b":
                    self._state = "esc"
                elif ch == "\r":
                    self._emit(out, "\n")
                elif ch in "\n\t" or (ord(ch) >= 0x20 and ch != "\x7f"):
                    self._emit(out, ch)
            elif state == "esc":
                if ch == "[":
                    self._state = "csi"
                elif ch == "]":
                    self._state = "osc"
                elif ch in "PX^_":
                    self._state = "string"
                else:
                    self._state = "normal"
            elif state == "csi":
                if "@" <= ch <= "~":
                    self._state = "normal"
            elif state == "osc":
                if ch == "\x07":
                    self._state = "normal"
                elif ch == "\x1b":
                    self._state = "osc_esc"
            elif state == "osc_esc":
                self._state = "normal" if ch == "\\" else "osc"
            elif state == "string":
                if ch == "\x1b":
                    self._state = "string_esc"
            elif state == "string_esc":
                self._state = "normal" if ch == "\\" else "string"
        return "".join(out)

    def flush(self) -> str:
        tail = self._pending_indent
        self._state = "normal"
        self._line_start = True
        self._pending_indent = ""
        return tail

    def _emit(self, out: list[str], ch: str) -> None:
        if self._line_start and ch in " \t":
            self._pending_indent += ch
            return
        if self._line_start:
            if ch == "[":
                self._pending_indent = ""
            else:
                out.append(self._pending_indent)
                self._pending_indent = ""
            self._line_start = False
        out.append(ch)
        if ch == "\n":
            self._line_start = True


def _write_stream(sink: Any, text: str) -> None:
    if not text:
        return
    with _OUTPUT_LOCK:
        sink.write(text)
        sink.flush()


def _print_stderr(message: str) -> None:
    _write_stream(sys.stderr, f"{message}\n")


def _env_int_optional(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        value = int(raw.strip())
    except Exception:
        return None
    return value if value > 0 else None


_CMD_TIMEOUT_SEC = _env_int_optional("BENCH_CMD_TIMEOUT_SEC")

# Routing-log scraping. When `--emit-routing-log` is passed to `run-method`,
# `do_run_method` sets `LARGE_SCALE_ROUTING_LOG_PATH` to the destination file.
# `run_cmd_stream` then appends every captured stderr line that contains the
# `[OUTER]` log marker emitted by `crate::solver::outer_strategy::log_plan` —
# the line carries the stable `solver=...;hessian=...;matrix-free=...` token
# defined by `OuterPlan::routing_log_line()`. Bench tests scrape this file.
_ROUTING_LOG_OUTER_MARKER = "[OUTER]"


def _routing_log_path() -> Path | None:
    raw = os.environ.get("LARGE_SCALE_ROUTING_LOG_PATH")
    if not raw:
        return None
    return Path(raw)


def _append_routing_lines(path: Path, captured_stderr: str) -> None:
    matched = [
        line for line in captured_stderr.splitlines() if _ROUTING_LOG_OUTER_MARKER in line
    ]
    if not matched:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for line in matched:
            fh.write(line.rstrip("\n") + "\n")
ROUTINE_SURVIVAL_HORIZONS = (1.0, 2.0, 5.0, 10.0)
SURVIVAL_ENTRY_COLUMN = "__entry"
F64_BYTES = 8
_RUST: Any | None = None
_BENCH_RUST_LOADER: Any | None = None


def _load_bench_rust_loader() -> Any:
    global _BENCH_RUST_LOADER
    if _BENCH_RUST_LOADER is not None:
        return _BENCH_RUST_LOADER
    loader_path = BENCH_DIR.parent / "_rust_loader.py"
    spec = importlib.util.spec_from_file_location("bench_rust_loader", loader_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load bench rust loader from {loader_path}")
    loader_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader_mod)
    _BENCH_RUST_LOADER = loader_mod
    return loader_mod


def _rust() -> Any:
    # Load gamfit._rust directly from the .so/.pyd file rather than going
    # through `from gamfit._binding import rust_module`, which would
    # execute `gamfit/__init__.py` → `_penalties.py` at module-load time
    # and abort the whole runner before `prepare` / `matrix` (neither of
    # which needs the Rust extension) ever runs. The bench-shared loader
    # enumerates every importable / installed / source-tree gamfit
    # location so it finds the `.so` even when the source tree shadows
    # the pip-installed wheel on sys.path.
    global _RUST
    if _RUST is not None:
        return _RUST
    _RUST = _load_bench_rust_loader().load_gamfit_rust_module(ROOT)
    return _RUST


def _f64_list(values: np.ndarray) -> list[float]:
    return np.asarray(values, dtype=float).reshape(-1).tolist()


def _detect_host_memory_bytes() -> int:
    """Effective memory available to this process.

    Consults cgroup v2, then cgroup v1, then /proc/meminfo MemTotal. The
    smallest finite limit wins so a 16 GiB GitHub-hosted runner is reported
    as 16 GiB even if its parent system would otherwise look larger; this
    is the value the OS-level OOM killer will actually enforce. Falls back
    to a 64 GiB hardcoded value only if all detection paths are unavailable
    (e.g. macOS), so the preflight telemetry stays useful on developer
    workstations and small CI runners.
    """
    fallback = 64 * 1024**3
    candidates: list[int] = []

    # cgroup v2 unified hierarchy. "max" means "no limit".
    try:
        raw = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if raw and raw != "max":
            value = int(raw)
            if 0 < value < (1 << 60):
                candidates.append(value)
    except (OSError, ValueError):
        pass

    # cgroup v1. The kernel reports a near-MAX_INT sentinel when unlimited.
    try:
        raw = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes").read_text().strip()
        if raw:
            value = int(raw)
            if 0 < value < (1 << 60):
                candidates.append(value)
    except (OSError, ValueError):
        pass

    # /proc/meminfo MemTotal as the floor of physical RAM.
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                kib = int(line.split()[1])
                candidates.append(kib * 1024)
                break
    except (OSError, ValueError, IndexError):
        pass

    if not candidates:
        return fallback
    return min(candidates)


DEFAULT_LARGE_SCALE_RAM_BUDGET_BYTES = _detect_host_memory_bytes()
LARGE_SCALE_MAX_DENSE_BLOCK_BYTES = 2 * 1024**3
LARGE_SCALE_MAX_DERIVATIVE_DENSE_BYTES = 2 * 1024**3
LARGE_SCALE_SURVIVAL_PREDICTION_CHUNK_ROWS = 8192


# Mirrors of constants in src/families/transformation_normal.rs governing the
# size of the monotonicity response grid built inside the CTN family. Update
# both sides in lockstep if the Rust constants change.
TRANSFORMATION_RESPONSE_GRID_MAX_QUANTILES = 129
TRANSFORMATION_RESPONSE_GRID_SUBDIVISIONS = 4
# Upper-bound estimate for the number of internal knots used by the CTN
# response-direction basis at large scale. The exact count is computed
# inside `effective_response_num_internal_knots` in the Rust code; the
# preflight uses a conservative cap so the modelled grid size does not
# under-report. Bumping this up is safe (only loosens the preflight check).
CTN_RESPONSE_INTERNAL_KNOTS_CAP = 32
# Spectral power for the 16-PC joint Duchon smooth. With order=0 the polynomial
# null-space order is p=1, so 2*(p+power) = 2*(1+power). The exact two-block
# spatial / transformation-normal (CTN) paths these specs run differentiate the
# radial kernel at the origin and need its 2nd derivative, which is finite only
# when 2*(p+power) > dimension+2 = 18. power=8 gives exactly 18 (not strictly
# greater) and fails; power=9 gives 20 > 18 and clears every path the benchmark
# exercises. (Kernel existence and D1/D2 collocation are all satisfied at 9, and
# length_scale=1 makes the hybrid kernel strictly PD so the pure-mode CPD bound
# 2*power < d does not apply.)
LARGE_SCALE_DUCHON16D_ORDER = 0
LARGE_SCALE_DUCHON16D_POWER = 9
LARGE_SCALE_DUCHON16D_LENGTH_SCALE = 1.0
PGS_RAW_COLUMN = "pgs_raw"
PGS_CTN_Z_COLUMN = "pgs_ctn_z"
PGS_CTN_FIT_SUBSAMPLE_N = 5000
# At large-scale n=320k the fixed 5000-row subsample only covers ~1.6% of the
# 16D continuous PC distribution, which left CTN-z with kurt≈3733 (CI run
# 25338491995). Local n=16k with the same 5000 covers ~31% and got kurt≈7.7.
# Scale K with sqrt(n_train) — keeps O(K^2) cost manageable while ~4×-ing
# the per-cell coverage at large scale.
PGS_CTN_FIT_SUBSAMPLE_N_LARGE_SCALE = 20000
PGS_CTN_FIT_SUBSAMPLE_LARGE_SCALE_THRESHOLD = 50000
PGS_CTN_FIT_SUBSAMPLE_SEED = 20260430
PGS_CTN_DIAGNOSTIC_MIN_N = 40
PGS_CTN_DIAGNOSTIC_MAX_ABS_MEAN = 0.30
PGS_CTN_DIAGNOSTIC_MIN_VAR = 0.50
PGS_CTN_DIAGNOSTIC_MAX_VAR = 1.75
SUPPORTED_LARGE_SCALE_SURVIVAL_LIKELIHOODS = {"transformation", "location-scale", "marginal-slope"}
SUPPORTED_LARGE_SCALE_SURVIVAL_DISTRIBUTIONS = {
    "gaussian",
    "probit",
    "gumbel",
    "cloglog",
    "logistic",
    "logit",
}


@dataclass(frozen=True)
class MethodSpec:
    name: str
    dataset: str
    backend: str
    family: str
    spatial_basis: str
    centers: int | None = None
    include_sigma: bool = False
    survival_likelihood: str | None = None
    survival_distribution: str | None = None
    marginal_slope: bool = False
    scale_dimensions: bool = False
    z_column: str | None = None
    pc_count: int = 16
    mean_linkwiggle_knots: int | None = None
    logslope_linkwiggle_knots: int | None = None
    timewiggle_knots: int | None = None


@dataclass(frozen=True)
class LargeScalePreflightReport:
    status: str
    lines: list[str]
    largest_single_allocation_bytes: int
    chunk_rows: int | None = None


def gibibytes(nbytes: int) -> float:
    return float(nbytes) / float(1024**3)


def _preflight_status_line(status: str) -> str:
    return f"status: {status}"


def preflight_marginal_slope_large_scale(
    *,
    n_train: int,
    d_pc: int,
    centers: int,
    linkwiggle_knots: int | None = None,
    scorewarp_knots: int | None = None,
    ram_budget_bytes: int = DEFAULT_LARGE_SCALE_RAM_BUDGET_BYTES,
) -> LargeScalePreflightReport:
    if n_train <= 0 or d_pc <= 0 or centers <= 0:
        raise RuntimeError("large-scale preflight dimensions must be positive")
    p_pc = centers + 1
    dense_block_bytes = n_train * p_pc * F64_BYTES
    derivative_dense_bytes = d_pc * dense_block_bytes
    linkwiggle = int(linkwiggle_knots or 0)
    scorewarp = int(scorewarp_knots or 0)
    working_bytes = n_train * (8 + d_pc + linkwiggle + scorewarp) * F64_BYTES

    # CTN prep peak memory model. The conditional transformation family fitted
    # before the marginal-slope stage builds a monotonicity-grid derivative
    # design whose virtual rows are the Cartesian product of training rows and
    # the response monotonicity grid. With the Kronecker variant the two
    # factors are kept separate, so the peak working allocation is just the
    # h'(grid) and delta-h'(grid) vectors of length n_train * n_grid each.
    # Pre-fix (row-replicated factors) the peak was n_train * n_grid * (p_resp +
    # p_cov) * 8 — surfaced here for reporting so the OOM regression at large-scale
    # scale stays visible if anyone removes the factored representation.
    n_grid_estimate = (
        TRANSFORMATION_RESPONSE_GRID_MAX_QUANTILES
        + CTN_RESPONSE_INTERNAL_KNOTS_CAP * (TRANSFORMATION_RESPONSE_GRID_SUBDIVISIONS + 1)
        + 4
    )
    p_resp_estimate = 2 + max(CTN_RESPONSE_INTERNAL_KNOTS_CAP - 2, 1)
    p_cov_ctn = p_pc
    ctn_prep_replicated_response_bytes = n_train * n_grid_estimate * p_resp_estimate * F64_BYTES
    ctn_prep_replicated_covariate_bytes = n_train * n_grid_estimate * p_cov_ctn * F64_BYTES
    ctn_prep_replicated_peak_bytes = (
        ctn_prep_replicated_response_bytes + ctn_prep_replicated_covariate_bytes
    )
    ctn_prep_factored_peak_bytes = 2 * n_train * n_grid_estimate * F64_BYTES + (
        n_grid_estimate * p_resp_estimate * F64_BYTES
    )

    estimated_peak = (
        max(dense_block_bytes, ctn_prep_factored_peak_bytes)
        + working_bytes
        + 384 * 1024**2
    )
    largest = max(
        dense_block_bytes,
        derivative_dense_bytes,
        working_bytes,
        ctn_prep_factored_peak_bytes,
    )
    failures: list[str] = []
    if dense_block_bytes > LARGE_SCALE_MAX_DENSE_BLOCK_BYTES:
        failures.append(
            f"estimated dense block: {gibibytes(dense_block_bytes):.1f} GiB exceeds {gibibytes(LARGE_SCALE_MAX_DENSE_BLOCK_BYTES):.1f} GiB"
        )
    if derivative_dense_bytes > LARGE_SCALE_MAX_DERIVATIVE_DENSE_BYTES:
        failures.append(
            f"anisotropic derivative dense estimate: {gibibytes(derivative_dense_bytes):.1f} GiB exceeds {gibibytes(LARGE_SCALE_MAX_DERIVATIVE_DENSE_BYTES):.1f} GiB"
        )
    if estimated_peak > int(0.80 * ram_budget_bytes):
        failures.append(
            f"estimated peak RSS: {gibibytes(estimated_peak):.1f} GiB exceeds 80% RAM budget {gibibytes(ram_budget_bytes):.1f} GiB"
        )
    if ctn_prep_factored_peak_bytes > int(0.80 * ram_budget_bytes):
        failures.append(
            f"CTN prep factored peak: {gibibytes(ctn_prep_factored_peak_bytes):.1f} GiB exceeds 80% RAM budget {gibibytes(ram_budget_bytes):.1f} GiB"
        )
    status = "FAIL" if failures else "PASS"
    lines = [
        "LARGE_SCALE PREFLIGHT",
        f"n_train: {n_train:,}",
        f"d_pc: {d_pc}",
        f"K_pc: {centers}",
        f"Duchon tuple: order={LARGE_SCALE_DUCHON16D_ORDER}, power={LARGE_SCALE_DUCHON16D_POWER}, length_scale={LARGE_SCALE_DUCHON16D_LENGTH_SCALE:g}",
        "Duchon smooth: lazy chunked",
        "marginal-slope anisotropy derivatives: implicit streaming",
        "conditional PGS CTN geometry: isotropic joint-PC Duchon (no scale dimensions)",
        "CTN Kronecker: factored (Kronecker variant for monotonicity grid)",
        f"CTN response grid points (upper bound): {n_grid_estimate}",
        f"CTN p_resp upper bound: {p_resp_estimate}",
        f"CTN p_cov: {p_cov_ctn}",
        f"CTN prep replicated response factor (avoided): {gibibytes(ctn_prep_replicated_response_bytes):.1f} GiB",
        f"CTN prep replicated covariate factor (avoided): {gibibytes(ctn_prep_replicated_covariate_bytes):.1f} GiB",
        f"CTN prep replicated peak (pre-fix, avoided): {gibibytes(ctn_prep_replicated_peak_bytes):.1f} GiB",
        f"CTN prep factored peak (post-fix, modelled): {gibibytes(ctn_prep_factored_peak_bytes):.2f} GiB",
        "survival time tensor: n/a",
        f"linkwiggle knots: {linkwiggle}",
        f"scorewarp knots: {scorewarp}",
        f"estimated dense block: {gibibytes(dense_block_bytes):.1f} GiB",
        f"anisotropic derivative dense estimate: {gibibytes(derivative_dense_bytes):.1f} GiB",
        f"estimated peak RSS: {gibibytes(estimated_peak):.1f} GiB",
        f"RAM budget: {gibibytes(ram_budget_bytes):.0f} GiB",
        f"largest single allocation planned: {gibibytes(largest):.1f} GiB",
        _preflight_status_line(status),
    ]
    lines.extend(f"failure: {failure}" for failure in failures)
    return LargeScalePreflightReport(status, lines, largest)


def preflight_ctn_score_warp(
    *,
    n_train: int,
    p_response: int,
    p_cov: int,
    ram_budget_bytes: int = DEFAULT_LARGE_SCALE_RAM_BUDGET_BYTES,
) -> LargeScalePreflightReport:
    if n_train <= 0 or p_response <= 0 or p_cov <= 0:
        raise RuntimeError("CTN preflight dimensions must be positive")
    dense_kron_bytes = n_train * p_response * p_cov * F64_BYTES
    factored_bytes = n_train * (p_response + p_cov) * F64_BYTES
    estimated_peak = factored_bytes + 512 * 1024**2
    status = "PASS"
    failures: list[str] = []
    if estimated_peak > int(0.80 * ram_budget_bytes):
        status = "ROUTE"
        failures.append("factored CTN design exceeds RAM budget")
    lines = [
        "LARGE_SCALE PREFLIGHT",
        f"n_train: {n_train:,}",
        "CTN Kronecker: factored",
        f"p_response: {p_response}",
        f"p_cov: {p_cov}",
        f"avoided dense rowwise Kronecker: {gibibytes(dense_kron_bytes):.1f} GiB",
        f"estimated factored bytes: {gibibytes(factored_bytes):.1f} GiB",
        f"estimated peak RSS: {gibibytes(estimated_peak):.1f} GiB",
        _preflight_status_line(status),
    ]
    lines.extend(f"route note: {failure}" for failure in failures)
    return LargeScalePreflightReport(
        status,
        lines,
        max(factored_bytes, p_response * p_cov * F64_BYTES),
    )


def preflight_survival_prediction(
    *,
    n_rows: int,
    grid_points: int,
    chunk_rows: int = LARGE_SCALE_SURVIVAL_PREDICTION_CHUNK_ROWS,
    ram_budget_bytes: int = DEFAULT_LARGE_SCALE_RAM_BUDGET_BYTES,
) -> LargeScalePreflightReport:
    if n_rows <= 0 or grid_points <= 0 or chunk_rows <= 0:
        raise RuntimeError("survival prediction preflight dimensions must be positive")
    dense_time_tensor_bytes = n_rows * grid_points * F64_BYTES
    chunked_bytes = min(n_rows, chunk_rows) * grid_points * F64_BYTES
    estimated_peak = chunked_bytes + 256 * 1024**2
    failures: list[str] = []
    if chunked_bytes > LARGE_SCALE_MAX_DENSE_BLOCK_BYTES:
        failures.append("survival prediction chunk is too large")
    if estimated_peak > int(0.80 * ram_budget_bytes):
        failures.append("chunked survival prediction exceeds RAM budget")
    status = "ROUTE" if failures else "PASS"
    lines = [
        "LARGE_SCALE PREFLIGHT",
        f"n_predict: {n_rows:,}",
        f"survival grid: {grid_points}",
        f"survival time tensor: chunked rows={chunk_rows}",
        f"avoided dense n x grid tensor: {gibibytes(dense_time_tensor_bytes):.1f} GiB",
        f"largest single allocation planned: {gibibytes(chunked_bytes):.1f} GiB",
        f"estimated peak RSS: {gibibytes(estimated_peak):.1f} GiB",
        _preflight_status_line(status),
    ]
    lines.extend(f"route note: {failure}" for failure in failures)
    return LargeScalePreflightReport(status, lines, chunked_bytes, chunk_rows=chunk_rows)


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return dict(json.loads(text))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{path} must contain JSON-compatible YAML so this runner can parse it without external dependencies: {exc}"
        ) from exc


def validate_method_spec(spec: MethodSpec) -> None:
    if spec.pc_count <= 0 or spec.pc_count > 16:
        raise RuntimeError(f"method '{spec.name}' must set pc_count in [1, 16]")
    if spec.marginal_slope and spec.z_column != PGS_CTN_Z_COLUMN:
        raise RuntimeError(
            f"method '{spec.name}' is a marginal-slope lane and must use "
            f"z_column='{PGS_CTN_Z_COLUMN}', not '{spec.z_column or 'unset'}'"
        )
    if spec.marginal_slope and spec.spatial_basis != "duchon":
        raise RuntimeError(
            f"method '{spec.name}' is a marginal-slope lane and must use spatial_basis='duchon'"
        )
    if spec.marginal_slope and not spec.scale_dimensions:
        raise RuntimeError(f"method '{spec.name}' must set scale_dimensions=true")
    for key, value in (
        ("mean_linkwiggle_knots", spec.mean_linkwiggle_knots),
        ("logslope_linkwiggle_knots", spec.logslope_linkwiggle_knots),
        ("timewiggle_knots", spec.timewiggle_knots),
    ):
        if value is not None and value < 3:
            raise RuntimeError(f"method '{spec.name}' requires {key} >= 3")
    if spec.dataset == "disease":
        if spec.backend != "rust_gam":
            raise RuntimeError(
                f"unsupported disease backend '{spec.backend}' for '{spec.name}'"
            )
        # Rigid (no link / score deviation) margslope methods legitimately
        # leave both linkwiggle-knots fields unset. Downstream consumers
        # treat `None` as "no linkwiggle term" and skip adding it to the
        # formula. The min-knot floor (>=3) is enforced separately above
        # for any non-None value.
        if spec.survival_likelihood is not None or spec.survival_distribution is not None:
            raise RuntimeError(
                f"disease method '{spec.name}' cannot set survival_likelihood or survival_distribution"
            )
        return
    if spec.dataset != "survival":
        raise RuntimeError(f"unsupported dataset '{spec.dataset}' for '{spec.name}'")
    if spec.backend in {"rust_survival_transform", "rust_gamlss_survival"}:
        raise RuntimeError(
            f"legacy survival backend '{spec.backend}' is not supported for '{spec.name}'; "
            "use backend='rust_survival' with explicit survival_likelihood and survival_distribution"
        )
    if spec.backend == "rust_survival":
        if spec.survival_likelihood not in SUPPORTED_LARGE_SCALE_SURVIVAL_LIKELIHOODS:
            supported = "|".join(sorted(SUPPORTED_LARGE_SCALE_SURVIVAL_LIKELIHOODS))
            raise RuntimeError(
                f"survival method '{spec.name}' requires survival_likelihood in {supported}"
            )
        if (
            spec.survival_likelihood != "marginal-slope"
            and spec.survival_distribution not in SUPPORTED_LARGE_SCALE_SURVIVAL_DISTRIBUTIONS
        ):
            supported = "|".join(sorted(SUPPORTED_LARGE_SCALE_SURVIVAL_DISTRIBUTIONS))
            raise RuntimeError(
                f"survival method '{spec.name}' requires survival_distribution in {supported}"
            )
        if spec.survival_likelihood == "marginal-slope":
            if not spec.marginal_slope:
                raise RuntimeError(
                    f"survival method '{spec.name}' must set marginal_slope=true for survival_likelihood=marginal-slope"
                )
            if spec.mean_linkwiggle_knots is None:
                raise RuntimeError(
                    f"survival marginal-slope method '{spec.name}' must set mean_linkwiggle_knots"
                )
            if spec.logslope_linkwiggle_knots is None:
                raise RuntimeError(
                    f"survival marginal-slope method '{spec.name}' must set logslope_linkwiggle_knots"
                )
            if spec.timewiggle_knots is None:
                raise RuntimeError(
                    f"survival marginal-slope method '{spec.name}' must set timewiggle_knots"
                )
            if spec.survival_distribution is not None:
                raise RuntimeError(
                    f"survival marginal-slope method '{spec.name}' must not set survival_distribution"
                )
        if spec.include_sigma:
            raise RuntimeError(
                f"survival method '{spec.name}' cannot use include_sigma; choose survival_likelihood explicitly"
            )
        return
    raise RuntimeError(
        f"unsupported survival backend '{spec.backend}' for '{spec.name}'"
    )


def survival_generation_params(cfg: dict[str, Any]) -> tuple[float, float]:
    shape = float(cfg.get("survival_weibull_shape", 1.65))
    scale = float(cfg.get("survival_weibull_scale", 11.5))
    if not math.isfinite(shape) or shape <= 0.0:
        raise RuntimeError("survival_weibull_shape must be finite and > 0")
    if not math.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("survival_weibull_scale must be finite and > 0")
    return shape, scale


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _transformation_score_values(score_csv: Path) -> list[float]:
    rows = read_csv_rows(score_csv)
    if not rows:
        raise RuntimeError(f"empty transformation-normal score file: {score_csv}")
    if "score" not in rows[0]:
        raise RuntimeError(
            f"transformation-normal score file {score_csv} is missing its typed score column"
        )
    return [float(row["score"]) for row in rows]


def _write_rows_like(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"cannot write empty CSV to {path}")
    write_csv_rows(path, rows, list(rows[0].keys()))


def _pc_std_columns(pc_count: int) -> list[str]:
    return [f"pc{i}_std" for i in range(1, int(pc_count) + 1)]


def _large_scale_duchon_pc_term(pc_count: int, centers: int) -> str:
    pc_cols = ", ".join(_pc_std_columns(pc_count))
    return (
        f"duchon({pc_cols}, centers={centers}, "
        f"order={LARGE_SCALE_DUCHON16D_ORDER}, power={LARGE_SCALE_DUCHON16D_POWER}, "
        f"length_scale={LARGE_SCALE_DUCHON16D_LENGTH_SCALE:g})"
    )


def _large_scale_pc_smooth_term(spatial_basis: str, pc_count: int, centers: int) -> str:
    """Joint multi-D smooth over the PC grouping axes.

    All large-scale lanes treat grouping as a single object on the joint PC space
    (the production-pipeline strategic goal: PGS calibration via Duchon/TPS on
    joint PC). Lat/lon geographic coordinates are deliberately excluded — the
    relevant continuous structure is latent grouping, not geography.
    """
    pc_cols = ", ".join(_pc_std_columns(pc_count))
    if spatial_basis == "duchon":
        return _large_scale_duchon_pc_term(pc_count, centers)
    if spatial_basis == "thinplate":
        return f"thinplate({pc_cols}, knots={centers})"
    if spatial_basis == "matern":
        return f"matern({pc_cols}, centers={centers})"
    raise RuntimeError(
        f"unsupported Rust joint-PC spatial basis '{spatial_basis}' "
        "(use duchon, thinplate, or matern)"
    )


def _ctn_formula(pc_count: int, centers: int) -> str:
    return f"{PGS_RAW_COLUMN} ~ {_large_scale_duchon_pc_term(pc_count, centers)}"


def _attach_column(rows: list[dict[str, str]], column: str, values: list[float]) -> list[dict[str, Any]]:
    if len(rows) != len(values):
        raise RuntimeError(
            f"cannot attach {column}: {len(values)} values for {len(rows)} rows"
        )
    out: list[dict[str, Any]] = []
    for row, value in zip(rows, values):
        enriched: dict[str, Any] = dict(row)
        enriched[column] = float(value)
        out.append(enriched)
    return out


def _z_moment_report(
    rows: list[dict[str, Any]],
    *,
    z_column: str,
    pc_columns: list[str],
    split_label: str,
) -> list[str]:
    z = np.array([float(row[z_column]) for row in rows], dtype=float)
    if z.size == 0:
        raise RuntimeError(f"{split_label}: no rows available for {z_column} diagnostics")
    if not np.all(np.isfinite(z)):
        raise RuntimeError(f"{split_label}: {z_column} contains non-finite values")
    reports: list[str] = []

    def check_group(label: str, values: np.ndarray) -> None:
        if values.size < PGS_CTN_DIAGNOSTIC_MIN_N:
            return
        mean = float(np.mean(values))
        var = float(np.var(values))
        centered = values - mean
        sd = math.sqrt(var) if var > 0.0 else 0.0
        skew = float(np.mean((centered / sd) ** 3)) if sd > 0.0 else float("nan")
        excess_kurt = float(np.mean((centered / sd) ** 4) - 3.0) if sd > 0.0 else float("nan")
        reports.append(
            f"{split_label}: {label} n={values.size:,} mean={mean:+.4f} "
            f"var={var:.4f} skew={skew:+.4f} excess_kurt={excess_kurt:+.4f}"
        )
        # Soft per-group calibration diagnostics. Violations are surfaced as
        # warnings rather than RuntimeError so an isotropic CTN preprocessor
        # (the speed-friendly default at large-scale dimensionality) can proceed
        # even when its global z distribution carries heavier tails than the
        # downstream marginal-slope model strictly assumes; the gam binary
        # itself enforces a separate (also-warn-by-default) latent-z policy
        # at fit time.
        if abs(mean) > PGS_CTN_DIAGNOSTIC_MAX_ABS_MEAN:
            print(
                f"[CTN diag warning] {split_label}: {label} has E[{z_column}|A] far from 0: {mean:+.4f}",
                file=sys.stderr,
                flush=True,
            )
        if var < PGS_CTN_DIAGNOSTIC_MIN_VAR or var > PGS_CTN_DIAGNOSTIC_MAX_VAR:
            print(
                f"[CTN diag warning] {split_label}: {label} has Var({z_column}|A) outside "
                f"[{PGS_CTN_DIAGNOSTIC_MIN_VAR}, {PGS_CTN_DIAGNOSTIC_MAX_VAR}]: {var:.4f}",
                file=sys.stderr,
                flush=True,
            )

    check_group("overall", z)
    for categorical in ("subpopulation", "superpopulation", "continent"):
        if categorical not in rows[0]:
            continue
        groups: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            groups[str(row[categorical])].append(float(row[z_column]))
        for group_name, vals in sorted(groups.items()):
            check_group(f"{categorical}={group_name}", np.array(vals, dtype=float))

    for pc in pc_columns:
        coords = np.array([float(row[pc]) for row in rows], dtype=float)
        if coords.size < 4 * PGS_CTN_DIAGNOSTIC_MIN_N or not np.all(np.isfinite(coords)):
            continue
        cuts = np.quantile(coords, [0.25, 0.50, 0.75])
        lower = np.concatenate(([-np.inf], cuts))
        upper = np.concatenate((cuts, [np.inf]))
        for idx, (left, right) in enumerate(zip(lower, upper), start=1):
            if idx == 4:
                mask = (coords >= left) & (coords <= right)
            else:
                mask = (coords >= left) & (coords < right)
            check_group(f"{pc}_quartile={idx}", z[mask])
    return reports


def fit_conditional_pgs_ctn_for_marginal_slope(
    *,
    rust_bin: Path,
    spec: MethodSpec,
    train_csv: Path,
    test_csv: Path,
    out_dir: Path,
    centers: int,
) -> tuple[Path, Path, list[str]]:
    train_rows = read_csv_rows(train_csv)
    test_rows = read_csv_rows(test_csv)
    if not train_rows or not test_rows:
        raise RuntimeError(f"{spec.name} requires non-empty train and test CSVs")
    required = {PGS_RAW_COLUMN, *_pc_std_columns(spec.pc_count)}
    missing = sorted(c for c in required if c not in train_rows[0] or c not in test_rows[0])
    if missing:
        raise RuntimeError(
            f"{spec.name} cannot fit conditional PGS CTN; missing columns: {', '.join(missing)}"
        )

    ctn_model_path = out_dir / f"{spec.name}.pgs_ctn.model.json"
    ctn_fit_input_path = out_dir / f"{spec.name}.pgs_ctn.fit_input.csv"
    ctn_train_input_path = out_dir / f"{spec.name}.pgs_ctn.train_input.csv"
    ctn_test_input_path = out_dir / f"{spec.name}.pgs_ctn.test_input.csv"
    ctn_train_score_path = out_dir / f"{spec.name}.pgs_ctn.train_score.csv"
    ctn_test_score_path = out_dir / f"{spec.name}.pgs_ctn.test_score.csv"
    formula = _ctn_formula(spec.pc_count, centers)
    ctn_columns = [PGS_RAW_COLUMN, *_pc_std_columns(spec.pc_count)]
    # Why this isn't a uniform random subsample any more:
    #
    # Previously: pick N=5000 rows uniformly at random, fit CTN on those,
    # predict z on the full 320k train + 80k test. The Duchon basis we use
    # for the conditional-CDF surface has a polynomial nullspace (order=0
    # in 16D ⇒ 1-dim constant nullspace), which is well-behaved at infinity
    # but the radial basis functions themselves still don't have well-defined
    # extrapolation outside the basis-support region. With 320k rows the
    # most extreme PC values sit at the 1/320,000 ≈ 3e-6 quantile; with a
    # 5000-row uniform subsample they're at the 1/5,000 ≈ 2e-4 quantile —
    # **64× further into the tail** in the full data than in any uniform
    # subsample. Predict-time rows beyond the fit-time PC envelope get
    # linearly extrapolated, which sends the conditional CDF estimate to
    # ~0 or ~1 spuriously, and `z = Φ⁻¹(F)` then blows up: at large-scale
    # scale we measured `sd(z) ≈ 1.88`, `skew(z) ≈ 209`,
    # `excess_kurt(z) ≈ 19711` — a few-row tail of |z| ~ 20+ that the
    # downstream marginal-slope BFGS gradient cannot escape from in the
    # CI 50-min budget.
    #
    # Fix: stratified subsample that *guarantees* the per-PC extremes are
    # in the CTN fit set. For each `pc{i}_std` column we take the K rows
    # with smallest values and the K rows with largest values, so the
    # fitted basis envelope matches the prediction envelope on every
    # axis. The remaining budget is filled uniformly at random. The total
    # subsample size stays close to `PGS_CTN_FIT_SUBSAMPLE_N` (slightly
    # larger when many rows are in multiple per-axis extremes; we dedupe).
    # Pick the CTN fit subsample size adaptively. At large scale we use
    # PGS_CTN_FIT_SUBSAMPLE_N_LARGE_SCALE to give the CTN basis enough coverage
    # of the 16D continuous PC distribution (the 5000-row default leaves
    # ~1.6% coverage at n=320k vs ~31% at n=16k local; kurt(z) drops from
    # ~3700 toward ~10 with 4× more rows in fit).
    effective_subsample_n = (
        PGS_CTN_FIT_SUBSAMPLE_N_LARGE_SCALE
        if len(train_rows) > PGS_CTN_FIT_SUBSAMPLE_LARGE_SCALE_THRESHOLD
        else PGS_CTN_FIT_SUBSAMPLE_N
    )
    if len(train_rows) > effective_subsample_n:
        rng = np.random.default_rng(PGS_CTN_FIT_SUBSAMPLE_SEED)
        pc_cols = _pc_std_columns(spec.pc_count)
        # Cap per-axis-keep at a small fixed number, NOT (SUBSAMPLE_N // 4 //
        # n_pcs). Why: with the prior formula at SUBSAMPLE_N=5000, pc_count=16
        # we forced 78 rows from each end of every PC. At local n=16k this
        # picks rows at the 0.49% quantile, which the CTN can fit cleanly
        # (kurt(z) ≈ 7.7 in test runs). At large-scale n=320k the SAME 78 rows
        # land at the 0.024% quantile — 20× further into the tail — so the
        # CTN training distribution is dominated by extreme outliers and the
        # fitted basis cannot generalize to interior PCs (CI run 25338491995
        # observed kurt(z) ≈ 3733, skew ≈ 65 on the bernoulli margslope
        # heldout split, vs ≈ 7.7 locally).
        #
        # Fixed cap of 20 per axis-end keeps the forced-extreme contribution
        # to ~640 rows out of 5000 (13%, vs the prior 50%). Coverage of the
        # per-axis envelope is still guaranteed; the predict-time CTN clamp
        # (5a306369) catches any predict rows beyond that envelope.
        per_axis_keep = max(2, min(20, effective_subsample_n // (16 * max(len(pc_cols), 1))))
        forced_idx: set[int] = set()
        for col in pc_cols:
            values = np.array([float(row[col]) for row in train_rows], dtype=float)
            order = np.argsort(values, kind="stable")
            for i in order[:per_axis_keep]:
                forced_idx.add(int(i))
            for i in order[-per_axis_keep:]:
                forced_idx.add(int(i))
        random_budget = max(0, effective_subsample_n - len(forced_idx))
        if random_budget > 0:
            available = np.array(
                sorted(set(range(len(train_rows))) - forced_idx),
                dtype=np.int64,
            )
            if available.size > random_budget:
                random_pick = rng.choice(available, size=random_budget, replace=False)
                forced_idx.update(int(i) for i in random_pick)
            else:
                forced_idx.update(int(i) for i in available)
        idx_list = sorted(forced_idx)
        rng.shuffle(idx_list)
        ctn_fit_rows = [train_rows[i] for i in idx_list]
        print(
            f"[CTN subsample] {len(ctn_fit_rows)} rows total: "
            f"{2 * per_axis_keep * len(pc_cols)} per-axis-extremes (max), "
            f"rest uniform random; covers full PC envelope on every axis",
            file=sys.stderr,
            flush=True,
        )
    else:
        ctn_fit_rows = train_rows
    # Predict-time PC clamping. Why this is needed even with the stratified
    # fit subsample:
    #
    # The configured order-0 Duchon basis has a constant polynomial nullspace,
    # but its fitted radial surface is still only supported by the fit-time PC
    # cloud. Outside that cloud, high-dimensional extrapolation can be extreme.
    # The stratified subsample
    # guarantees coverage of the per-axis PC extremes so the train/test PC
    # envelopes match per axis, but a row could still sit outside the
    # multi-axis hull (e.g. extreme on PC1 *and* PC4 simultaneously when no
    # fit row is). For CTN preprocessing we only need
    # `Φ⁻¹(F(pgs|PCs)) ≈ standard normal`; we have no scientific need to
    # extrapolate F outside the fit-time PC support, so it is safe — and
    # standard practice in GAM prediction — to clamp out-of-range PC values
    # to the fit envelope. Rows inside the box are unaffected; rows outside
    # are answered with the model's value at the nearest boundary instead of
    # an unbounded linear extrapolation. PGS_RAW is the response and is
    # never clamped here.
    pc_cols_for_clamp = _pc_std_columns(spec.pc_count)
    fit_pc_min: dict[str, float] = {}
    fit_pc_max: dict[str, float] = {}
    for col in pc_cols_for_clamp:
        vals = np.array([float(row[col]) for row in ctn_fit_rows], dtype=float)
        fit_pc_min[col] = float(np.min(vals))
        fit_pc_max[col] = float(np.max(vals))

    def _clamped_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, int]]:
        clamp_counts: dict[str, int] = {col: 0 for col in pc_cols_for_clamp}
        out_rows: list[dict[str, str]] = []
        for row in rows:
            new_row = {key: row[key] for key in ctn_columns}
            for col in pc_cols_for_clamp:
                v = float(row[col])
                lo = fit_pc_min[col]
                hi = fit_pc_max[col]
                if v < lo:
                    new_row[col] = repr(lo)
                    clamp_counts[col] += 1
                elif v > hi:
                    new_row[col] = repr(hi)
                    clamp_counts[col] += 1
                else:
                    new_row[col] = row[col]
            out_rows.append(new_row)
        return out_rows, clamp_counts

    train_clamped, train_clamp_counts = _clamped_rows(train_rows)
    test_clamped, test_clamp_counts = _clamped_rows(test_rows)
    total_train_clamped = sum(train_clamp_counts.values())
    total_test_clamped = sum(test_clamp_counts.values())
    print(
        f"[CTN clamp] fit-time PC envelope clamps "
        f"train={total_train_clamped} test={total_test_clamped} "
        f"(per-axis: train={train_clamp_counts} test={test_clamp_counts})",
        file=sys.stderr,
        flush=True,
    )
    write_csv_rows(
        ctn_fit_input_path,
        [{key: row[key] for key in ctn_columns} for row in ctn_fit_rows],
        ctn_columns,
    )
    write_csv_rows(
        ctn_train_input_path,
        train_clamped,
        ctn_columns,
    )
    write_csv_rows(
        ctn_test_input_path,
        test_clamped,
        ctn_columns,
    )
    fit_cmd = [
        str(rust_bin),
        "fit",
        "--transformation-normal",
        "--out",
        str(ctn_model_path),
        str(ctn_fit_input_path),
        formula,
    ]
    rc, out, err = run_cmd_stream(fit_cmd, cwd=ROOT)
    if rc != 0:
        raise RuntimeError(
            err.strip() or out.strip() or f"{spec.name} conditional PGS CTN fit failed"
        )
    for input_path, output_path in (
        (ctn_train_input_path, ctn_train_score_path),
        (ctn_test_input_path, ctn_test_score_path),
    ):
        score_cmd = [
            str(rust_bin),
            "transformation-score",
            str(ctn_model_path),
            str(input_path),
            "--out",
            str(output_path),
        ]
        rc, out, err = run_cmd_stream(score_cmd, cwd=ROOT)
        if rc != 0:
            raise RuntimeError(
                err.strip() or out.strip() or f"{spec.name} conditional PGS CTN scoring failed"
            )

    train_aug = _attach_column(
        train_rows,
        PGS_CTN_Z_COLUMN,
        _transformation_score_values(ctn_train_score_path),
    )
    test_aug = _attach_column(
        test_rows,
        PGS_CTN_Z_COLUMN,
        _transformation_score_values(ctn_test_score_path),
    )
    pc_cols = _pc_std_columns(spec.pc_count)
    diagnostics = [
        f"conditional PGS CTN formula: {formula}",
        "conditional PGS CTN fit uses isotropic joint-PC Duchon geometry (no scale dimensions)",
        f"conditional PGS CTN fit is phenotype-blind and train-only; downstream z column: {PGS_CTN_Z_COLUMN}",
        f"conditional PGS CTN fit subsample: {len(ctn_fit_rows)} of {len(train_rows)} train rows (cap {effective_subsample_n})",
        (
            f"conditional PGS CTN predict-time PC clamping: "
            f"train={total_train_clamped} test={total_test_clamped} "
            f"clamp events to fit envelope "
            f"({2 * len(pc_cols_for_clamp)} bounds = per-axis min/max from {len(ctn_fit_rows)} fit rows)"
        ),
    ]
    diagnostics.extend(
        _z_moment_report(train_aug, z_column=PGS_CTN_Z_COLUMN, pc_columns=pc_cols, split_label="train")
    )
    diagnostics.extend(
        _z_moment_report(test_aug, z_column=PGS_CTN_Z_COLUMN, pc_columns=pc_cols, split_label="heldout")
    )

    train_aug_path = out_dir / f"{spec.name}.pgs_ctn.train.csv"
    test_aug_path = out_dir / f"{spec.name}.pgs_ctn.test.csv"
    _write_rows_like(train_aug_path, train_aug)
    _write_rows_like(test_aug_path, test_aug)
    return train_aug_path, test_aug_path, diagnostics


def shared_ctn_spec(cfg: dict[str, Any]) -> MethodSpec:
    specs = [spec for spec in build_method_specs(cfg) if spec.marginal_slope]
    if not specs:
        raise RuntimeError("large-scale configuration has no marginal-slope lane for shared CTN")
    contracts = {
        (int(spec.pc_count), int(spec.centers or 24), spec.z_column)
        for spec in specs
    }
    if len(contracts) != 1:
        raise RuntimeError(
            "all marginal-slope lanes must share one CTN contract "
            f"(pc_count, centers, z_column); got {sorted(contracts)}"
        )
    return specs[0]


def require_shared_ctn_columns(spec: MethodSpec, train_csv: Path, test_csv: Path) -> None:
    required = PGS_CTN_Z_COLUMN
    for label, path in (("train", train_csv), ("heldout", test_csv)):
        with path.open("r", encoding="utf-8", newline="") as fh:
            fieldnames = csv.DictReader(fh).fieldnames or []
        if required not in fieldnames:
            raise RuntimeError(
                f"{spec.name} requires the shared CTN preprocessing artifact; "
                f"{label} CSV {path} is missing {required}"
            )


def do_prepare_ctn(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    spec = shared_ctn_spec(cfg)
    rust_bin_raw = os.environ.get("GAM_RUST_BINARY")
    if not rust_bin_raw:
        raise RuntimeError("prepare-ctn requires GAM_RUST_BINARY to name the immutable CLI artifact")
    rust_bin = Path(rust_bin_raw).resolve()
    if not rust_bin.is_file():
        raise RuntimeError(f"prepare-ctn GAM_RUST_BINARY does not exist: {rust_bin}")

    prep_dir = args.prep_dir.resolve()
    out_dir = args.out_dir.resolve()
    run_dir = out_dir / "shared_ctn"
    run_dir.mkdir(parents=True, exist_ok=True)
    source_train = prep_dir / "disease_train.csv"
    source_test = prep_dir / "disease_test.csv"
    started = time.perf_counter()
    train_aug, test_aug, diagnostics = fit_conditional_pgs_ctn_for_marginal_slope(
        rust_bin=rust_bin,
        spec=spec,
        train_csv=source_train,
        test_csv=source_test,
        out_dir=run_dir,
        centers=int(spec.centers or 24),
    )
    elapsed = time.perf_counter() - started
    for dataset in ("disease", "survival"):
        shutil.copy2(train_aug, out_dir / f"{dataset}_train.csv")
        shutil.copy2(test_aug, out_dir / f"{dataset}_test.csv")
    if (prep_dir / "prep_metadata.json").exists():
        shutil.copy2(prep_dir / "prep_metadata.json", out_dir / "prep_metadata.json")
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "elapsed_sec": elapsed,
        "pc_count": int(spec.pc_count),
        "centers": int(spec.centers or 24),
        "z_column": PGS_CTN_Z_COLUMN,
        "consumer_methods": [
            item.name for item in build_method_specs(cfg) if item.marginal_slope
        ],
        "diagnostics": diagnostics,
    }
    dump_json(out_dir / "ctn_metadata.json", metadata)
    print("\n".join(diagnostics), file=sys.stderr, flush=True)
    print(f"Wrote shared CTN artifact to {out_dir} in {elapsed:.3f}s")
    return 0


def logistic(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(np.asarray(x, dtype=float), -40.0, 40.0)
    return np.asarray(1.0 / (1.0 + np.exp(-x_clip)), dtype=float)


def standardize(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    mu = float(np.mean(arr))
    sd = float(np.std(arr))
    if (not np.isfinite(sd)) or sd < 1e-12:
        sd = 1.0
    return (arr - mu) / sd


def zscore_train_test(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    mu = float(np.mean(train))
    sd = float(np.std(train))
    if (not np.isfinite(sd)) or sd < 1e-12:
        sd = 1.0
    return (train - mu) / sd, (test - mu) / sd, mu, sd


def _survival_score_grid(train_times: np.ndarray) -> np.ndarray:
    return np.asarray(_rust().survival_score_grid_from_times(_f64_list(train_times)), dtype=float)


def _repeat_survival_curve(curve: np.ndarray, n_rows: int) -> np.ndarray:
    return np.asarray(_rust().repeat_survival_curve(_f64_list(curve), int(n_rows)), dtype=float)


def survival_concordance(event_times: np.ndarray, risk_score: np.ndarray, events: np.ndarray) -> float:
    return float(_rust().survival_concordance(_f64_list(event_times), _f64_list(risk_score), _f64_list(events)))


def _survival_null_curve(train_times: np.ndarray, train_events: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return _survival_calibration().kaplan_meier_curve(train_times, train_events, grid)


def calibrated_survival_matrix(
    train_times: np.ndarray,
    train_events: np.ndarray,
    train_risk: np.ndarray,
    test_risk: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    return _survival_calibration().calibrated_survival_matrix(
        train_times,
        train_events,
        train_risk,
        test_risk,
        grid,
    )


def survival_lifted_metrics(
    event_times: np.ndarray,
    events: np.ndarray,
    grid: np.ndarray,
    survival_matrix: np.ndarray,
    null_survival_matrix: np.ndarray,
) -> dict[str, float | None]:
    return dict(
        _rust().survival_lifted_metrics_from_predictions(
            _f64_list(event_times),
            _f64_list(events),
            _f64_list(grid),
            np.asarray(survival_matrix, dtype=float),
            np.asarray(null_survival_matrix, dtype=float),
        )
    )


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, train_prev: float) -> dict[str, float | None]:
    return dict(_rust().classification_metrics(_f64_list(y_true), _f64_list(y_prob), float(train_prev)))


def survival_metrics(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    train_risk: np.ndarray,
    test_risk: np.ndarray,
) -> dict[str, float | None]:
    train_times = np.array([float(r["time"]) for r in train_rows], dtype=float)
    train_events = np.array([float(r["event"]) for r in train_rows], dtype=float)
    test_times = np.array([float(r["time"]) for r in test_rows], dtype=float)
    test_events = np.array([float(r["event"]) for r in test_rows], dtype=float)
    grid = _survival_score_grid(train_times)
    surv = calibrated_survival_matrix(train_times, train_events, train_risk, test_risk, grid)
    null_curve = _survival_null_curve(train_times, train_events, grid)
    proper = survival_lifted_metrics(test_times, test_events, grid, surv, _repeat_survival_curve(null_curve, len(test_rows)))
    return {
        "c_index": survival_concordance(test_times, test_risk, test_events),
        "auc": survival_concordance(test_times, test_risk, test_events),
        "brier": proper["brier"],
        "logloss": proper["logloss"],
        "lifted_brier": proper["lifted_brier"],
        "lifted_logloss": proper["lifted_logloss"],
        "nagelkerke_r2": proper["nagelkerke_r2"],
    }


def survival_metrics_from_native_probabilities(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    grid: np.ndarray,
    survival_matrix: np.ndarray,
) -> dict[str, float | None]:
    train_times = np.array([float(r["time"]) for r in train_rows], dtype=float)
    train_events = np.array([float(r["event"]) for r in train_rows], dtype=float)
    test_times = np.array([float(r["time"]) for r in test_rows], dtype=float)
    test_events = np.array([float(r["event"]) for r in test_rows], dtype=float)
    if survival_matrix.shape != (len(test_rows), grid.shape[0]):
        raise RuntimeError(
            "native survival probability matrix shape mismatch: "
            f"got {survival_matrix.shape}, expected {(len(test_rows), grid.shape[0])}"
        )
    surv = np.clip(np.asarray(survival_matrix, dtype=float), 1e-12, 1.0)
    null_curve = _survival_null_curve(train_times, train_events, grid)
    proper = survival_lifted_metrics(
        test_times,
        test_events,
        grid,
        surv,
        _repeat_survival_curve(null_curve, len(test_rows)),
    )
    horizon = float(np.median(train_times))
    horizon_idx = min(int(np.searchsorted(grid, horizon, side="left")), grid.shape[0] - 1)
    native_failure = 1.0 - surv[:, horizon_idx]
    return {
        "c_index": survival_concordance(test_times, native_failure, test_events),
        "auc": survival_concordance(test_times, native_failure, test_events),
        "brier": proper["brier"],
        "logloss": proper["logloss"],
        "lifted_brier": proper["lifted_brier"],
        "lifted_logloss": proper["lifted_logloss"],
        "nagelkerke_r2": proper["nagelkerke_r2"],
    }


def _survival_probability_column(rows: list[dict[str, str]], *, method_name: str) -> np.ndarray:
    if not rows:
        raise RuntimeError(f"{method_name} survival prediction output is empty")
    if "survival_prob" not in rows[0]:
        raise RuntimeError(
            f"{method_name} survival prediction output missing 'survival_prob' column; "
            f"got columns {sorted(rows[0].keys())}"
        )
    key = "survival_prob"
    values = np.array([float(r[key]) for r in rows], dtype=float)
    if not np.all(np.isfinite(values)):
        raise RuntimeError(
            f"{method_name} survival prediction column '{key}' contains non-finite values"
        )
    if np.any(values < -1e-9) or np.any(values > 1.0 + 1e-9):
        raise RuntimeError(
            f"{method_name} survival prediction column '{key}' is outside [0,1]"
        )
    return np.asarray(np.clip(values, 0.0, 1.0), dtype=float)


def predict_native_survival_matrix(
    *,
    rust_bin: Path,
    spec: MethodSpec,
    model_path: Path,
    base_rows: list[dict[str, Any]],
    grid: np.ndarray,
    out_dir: Path,
) -> tuple[np.ndarray, Path]:
    n = len(base_rows)
    if n == 0 or grid.shape[0] == 0:
        raise RuntimeError(
            f"{spec.name} native survival scoring requires non-empty rows and grid"
        )
    stacked_rows: list[dict[str, Any]] = []
    for horizon in grid:
        stacked_rows.extend(
            prepare_survival_benchmark_rows(
                base_rows,
                prediction_horizon=float(horizon),
            )
        )
    input_path = out_dir / f"{spec.name}.native_survival_grid.csv"
    pred_path = out_dir / f"{spec.name}.native_survival_grid.pred.csv"
    if not stacked_rows:
        raise RuntimeError(
            f"{spec.name} cannot write an empty native survival scoring frame"
        )
    fieldnames = [SURVIVAL_ENTRY_COLUMN] + [
        key for key in stacked_rows[0].keys() if key != SURVIVAL_ENTRY_COLUMN
    ]
    write_csv_rows(input_path, stacked_rows, fieldnames)
    pred_cmd = [
        str(rust_bin),
        "predict",
        str(model_path),
        str(input_path),
        "--out",
        str(pred_path),
    ]
    rc, out, err = run_cmd_stream(pred_cmd, cwd=ROOT)
    if rc != 0:
        raise RuntimeError(
            err.strip()
            or out.strip()
            or f"{spec.name} native survival-grid prediction failed"
        )
    values = _survival_probability_column(read_csv_rows(pred_path), method_name=spec.name)
    expected = n * grid.shape[0]
    if values.shape[0] != expected:
        raise RuntimeError(
            f"{spec.name} native survival-grid prediction returned {values.shape[0]} rows; expected {expected}"
        )
    return values.reshape((grid.shape[0], n)).T, pred_path


def ps_snapshot(pid: int) -> dict[str, Any]:
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
        return {
            "pid": parts[0],
            "cpu_pct": parts[1],
            "mem_pct": parts[2],
            "rss_kib": int(parts[3]) if parts[3].isdigit() else None,
            "vsz_kib": int(parts[4]) if parts[4].isdigit() else None,
            "etimes": parts[5],
            "stat": parts[6],
            "comm": parts[7],
        }
    except Exception:
        return {}


def fmt_kib(kib: Any) -> str:
    if kib is None:
        return "n/a"
    return f"{float(kib) / (1024.0 * 1024.0):.2f} GiB"


def heartbeat_loop(proc: subprocess.Popen[bytes], cmd_preview: str, stop_event: threading.Event) -> None:
    start = time.monotonic()
    timeout = float(_CMD_TIMEOUT_SEC) if _CMD_TIMEOUT_SEC is not None else None
    warned_80pct = False
    while True:
        elapsed = time.monotonic() - start
        snap = ps_snapshot(proc.pid)
        # Highlight when we're approaching the cmd timeout — a one-shot
        # warning at 80% so CI logs grep on `[HEARTBEAT-WARN]` to find
        # near-timeout cases without needing to compute timing manually.
        if timeout is not None and not warned_80pct and elapsed >= 0.8 * timeout:
            _print_stderr(
                f"[HEARTBEAT-WARN] elapsed={elapsed:.1f}s exceeded 80% of cmd_timeout={timeout:.0f}s",
            )
            warned_80pct = True
        _print_stderr(
            f"[HEARTBEAT] elapsed={elapsed:8.1f}s cmd='{cmd_preview}' pid={proc.pid} "
            f"cpu={snap.get('cpu_pct', 'n/a')}% mem={snap.get('mem_pct', 'n/a')}% "
            f"rss={fmt_kib(snap.get('rss_kib'))} vsz={fmt_kib(snap.get('vsz_kib'))}",
        )
        wait_sec = HEARTBEAT_INITIAL_INTERVAL_SEC if elapsed < HEARTBEAT_INITIAL_WINDOW_SEC else HEARTBEAT_INTERVAL_SEC
        if stop_event.wait(wait_sec):
            break
        if proc.poll() is not None:
            break


def run_cmd_stream(cmd: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        bufsize=0,
    )
    out_buf: list[str] = []
    err_buf: list[str] = []
    # Dedicated buffer for [PHASE] / [OUTER summary] markers so they
    # survive stderr-buffer rollover when a long-running cmd produces
    # >MAX_CAPTURE_CHARS of HEARTBEAT noise. Lets `_emit_phase_summary`
    # still find the markers even after a 40-min run.
    phase_buf: list[str] = []
    stop_event = threading.Event()
    preview = " ".join(cmd[:5]) + (" ..." if len(cmd) > 5 else "")

    def pump(pipe: Any, sink: Any, capture: list[str], phase_capture: list[str] | None = None) -> None:
        total = 0
        sanitizer = _TerminalOutputSanitizer()
        try:
            while True:
                chunk = pipe.read(4096)
                if not chunk:
                    break
                text = sanitizer.feed(chunk.decode("utf-8", errors="replace"))
                _write_stream(sink, text)
                capture.append(text)
                total += len(text)
                if total > MAX_CAPTURE_CHARS:
                    del capture[0]
                    total = sum(len(x) for x in capture)
                if phase_capture is not None:
                    for line in text.splitlines(keepends=True):
                        if (
                            "[PHASE]" in line
                            or "[OUTER summary]" in line
                            or "[OUTER guard]" in line
                            or "[OUTER non-finite]" in line
                            or "[PIRLS iter-end]" in line
                            or "[PIRLS solve-end]" in line
                            or "[KAPPA-PHASE" in line
                            or "[IFT-QUALITY]" in line
                            or "[IFT-REJECTED]" in line
                            or "[IFT-NOOP]" in line
                            or "[TANGENT-PREDICT]" in line
                            or "[TANGENT-REJECTED]" in line
                            or "[TANGENT-QUALITY]" in line
                            or "[TANGENT-NOOP]" in line
                        ):
                            phase_capture.append(line)
        finally:
            tail = sanitizer.flush()
            if tail:
                _write_stream(sink, tail)
                capture.append(tail)
            pipe.close()

    t_out = threading.Thread(target=pump, args=(proc.stdout, sys.stdout, out_buf), daemon=True)
    t_err = threading.Thread(target=pump, args=(proc.stderr, sys.stderr, err_buf, phase_buf), daemon=True)
    t_hb = threading.Thread(target=heartbeat_loop, args=(proc, preview, stop_event), daemon=True)
    t_out.start()
    t_err.start()
    t_hb.start()
    timed_out = False
    try:
        if _CMD_TIMEOUT_SEC is not None:
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
    stop_event.set()
    t_out.join()
    t_err.join()
    t_hb.join(timeout=1.0)
    if timed_out:
        msg = (
            f"[HEARTBEAT] command-timeout rc=124 timeout_sec={_CMD_TIMEOUT_SEC} "
            f"pid={proc.pid} cmd='{preview}'"
        )
        _print_stderr(msg)
        # Emit the phase summary EVEN ON TIMEOUT — the most useful place
        # to see WHICH phase was running when the budget ran out.
        _emit_phase_summary("".join(phase_buf), preview, timed_out=True, rc=124)
        raise TimeoutError(msg)
    captured_stderr = "".join(err_buf)
    if (routing_path := _routing_log_path()) is not None:
        _append_routing_lines(routing_path, captured_stderr)
    # Emit a per-phase wall-clock summary parsed from the gam binary's
    # `[PHASE]` markers so CI logs end with a quick-glance breakdown of
    # CTN / margslope / standard-GAM / location-scale phase timings. We
    # parse the dedicated `phase_buf` (which retains all [PHASE] markers
    # even after stderr buffer rollover) rather than `captured_stderr`.
    _emit_phase_summary("".join(phase_buf), preview, timed_out=False, rc=rc)
    return rc, "".join(out_buf), captured_stderr


_PHASE_END_PATTERN = re.compile(
    r"\[PHASE\]\s+([\w\-]+(?:\([\w\-/]+\))?)\s+(?:fit\s+)?(?:end|done)\s+elapsed=([\d.]+)s"
)
_PHASE_START_PATTERN = re.compile(r"\[PHASE\]\s+([\w\-]+(?:\([\w\-/]+\))?)\s+(?:fit\s+)?start")
_BFGS_SUMMARY_PATTERN = re.compile(
    r"\[OUTER summary\]\s+BFGS\s+(converged|hit max_iter|line-search failed|failed)(?:\s+in\s+(\d+)\s+iters)?\s+elapsed=([\d.]+)s"
)


def _emit_phase_summary(
    captured_stderr: str,
    cmd_preview: str,
    *,
    timed_out: bool = False,
    rc: int = 0,
) -> None:
    by_phase: dict[str, float] = {}
    for name, secs in _PHASE_END_PATTERN.findall(captured_stderr):
        by_phase[name] = by_phase.get(name, 0.0) + float(secs)
    started = _PHASE_START_PATTERN.findall(captured_stderr)
    completed = set(by_phase)
    pending = [name for name in started if name not in completed]
    parts = [f"{name}={secs:.1f}s" for name, secs in by_phase.items()]
    bfgs = _BFGS_SUMMARY_PATTERN.findall(captured_stderr)
    if bfgs:
        status_counts: dict[str, int] = {}
        total = 0.0
        iters: list[int] = []
        for status, iter_text, secs in bfgs:
            status_counts[status] = status_counts.get(status, 0) + 1
            total += float(secs)
            if iter_text:
                iters.append(int(iter_text))
        status = " ".join(
            f"bfgs_{key.replace(' ', '_').replace('-', '_')}={value}"
            for key, value in sorted(status_counts.items())
        )
        iter_part = f" bfgs_iters_max={max(iters)}" if iters else ""
        parts.append(f"bfgs_runs={len(bfgs)} bfgs_total={total:.1f}s {status}{iter_part}")
    if pending:
        parts.append("pending=" + ",".join(pending[-5:]))
    if timed_out:
        parts.append("timed_out=true")
    if rc != 0:
        parts.append(f"rc={rc}")
    if parts:
        print(
            f"[PHASE summary] cmd='{cmd_preview}' " + " ".join(parts),
            file=sys.stderr,
            flush=True,
        )

def tool_exists(name: str) -> bool:
    return shutil.which(name) is not None


def load_or_build_rust_binary() -> Path:
    override = os.environ.get("GAM_RUST_BINARY")
    if override:
        override_path = Path(override)
        if override_path.exists():
            return override_path
        raise RuntimeError(f"GAM_RUST_BINARY points to missing file: {override_path}")
    prebuilt = ROOT / "target" / "release" / "gam"
    if prebuilt.exists():
        return prebuilt
    rc, out, err = run_cmd_stream(["cargo", "build", "--release", "--bin", "gam"], cwd=ROOT)
    if rc != 0:
        raise RuntimeError(err.strip() or out.strip() or "failed to build Rust binary")
    if not prebuilt.exists():
        raise RuntimeError(f"missing Rust binary at {prebuilt}")
    return prebuilt


def subpop_templates() -> list[dict[str, Any]]:
    return [
        {"subpop": "CEU_Utah", "continent": "Europe", "superpop": "EUR", "lat": 40.76, "lon": -111.89},
        {"subpop": "GBR_England", "continent": "Europe", "superpop": "EUR", "lat": 52.36, "lon": -1.17},
        {"subpop": "TSI_Italy", "continent": "Europe", "superpop": "EUR", "lat": 43.77, "lon": 11.25},
        {"subpop": "YRI_Nigeria", "continent": "Africa", "superpop": "AFR", "lat": 6.52, "lon": 3.37},
        {"subpop": "LWK_Kenya", "continent": "Africa", "superpop": "AFR", "lat": -0.02, "lon": 37.91},
        {"subpop": "GWD_Gambia", "continent": "Africa", "superpop": "AFR", "lat": 13.45, "lon": -16.58},
        {"subpop": "CHB_Beijing", "continent": "Asia", "superpop": "EAS", "lat": 39.90, "lon": 116.40},
        {"subpop": "JPT_Tokyo", "continent": "Asia", "superpop": "EAS", "lat": 35.68, "lon": 139.65},
        {"subpop": "KHV_HCMC", "continent": "Asia", "superpop": "EAS", "lat": 10.82, "lon": 106.63},
        {"subpop": "GIH_Houston", "continent": "South Asia", "superpop": "SAS", "lat": 29.76, "lon": -95.37},
        {"subpop": "PJL_Lahore", "continent": "South Asia", "superpop": "SAS", "lat": 31.55, "lon": 74.34},
        {"subpop": "BEB_Dhaka", "continent": "South Asia", "superpop": "SAS", "lat": 23.81, "lon": 90.41},
        {"subpop": "MXL_LA", "continent": "Admixed America", "superpop": "AMR", "lat": 34.05, "lon": -118.24},
        {"subpop": "PEL_Lima", "continent": "Admixed America", "superpop": "AMR", "lat": -12.05, "lon": -77.04},
        {"subpop": "PUR_SanJuan", "continent": "Admixed America", "superpop": "AMR", "lat": 18.47, "lon": -66.11},
    ]


def build_pc_means(templates: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for idx, tpl in enumerate(templates):
        base = np.zeros(16, dtype=float)
        continent_block = idx // 3
        base[continent_block] = 2.5
        base[(continent_block + 5) % 16] = -1.2
        base[(2 * continent_block + 7) % 16] = 0.8
        if tpl["superpop"] == "AFR":
            base[8] = 1.7
        elif tpl["superpop"] == "EAS":
            base[9] = -1.6
        elif tpl["superpop"] == "SAS":
            base[10] = 1.1
        elif tpl["superpop"] == "AMR":
            base[11] = -1.1
        else:
            base[12] = 0.7
        out[str(tpl["subpop"])] = base
    return out


def sample_covariance(pc_means: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    jitter = rng.normal(scale=0.06, size=(16, 16))
    a = np.eye(16) * 0.55 + (jitter @ jitter.T) / 16.0
    return np.asarray(a, dtype=float)


def disease_probability(lat: np.ndarray, lon: np.ndarray, pcs: np.ndarray, pgs: np.ndarray, age: np.ndarray, sex: np.ndarray) -> np.ndarray:
    lat_s = standardize(lat)
    lon_s = standardize(lon)
    linear = (
        -0.8
        + 0.9 * standardize(pgs)
        + 0.45 * pcs[:, 0]
        - 0.35 * pcs[:, 1]
        + 0.18 * standardize(age)
        + 0.22 * sex
        + 0.55 * np.sin(lat_s * 1.7)
        + 0.40 * np.cos(lon_s * 2.1)
        + 0.25 * lat_s * lon_s
    )
    return logistic(linear)


def survival_scale(
    lat: np.ndarray,
    lon: np.ndarray,
    pcs: np.ndarray,
    pgs: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
) -> np.ndarray:
    lat_s = standardize(lat)
    lon_s = standardize(lon)
    lp = (
        0.35 * standardize(pgs)
        + 0.20 * pcs[:, 0]
        - 0.12 * pcs[:, 2]
        + 0.10 * standardize(age)
        + 0.08 * sex
        + 0.30 * np.sin(lat_s * 1.2)
        - 0.22 * np.cos(lon_s * 1.8)
    )
    return np.exp(lp)


def generate_raw_cohort(cfg: dict[str, Any], out_dir: Path, smoke: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)
    base_n = int(cfg["raw_subpop_n"])
    templates = subpop_templates()
    pc_means = build_pc_means(templates)
    rows: list[dict[str, Any]] = []
    subject_id = 0
    for tpl in templates:
        mean = pc_means[tpl["subpop"]]
        cov = sample_covariance(mean, rng)
        n_local = base_n if not smoke else max(48, base_n // 5)
        pcs = rng.multivariate_normal(mean=mean, cov=cov, size=n_local)
        for row_idx in range(n_local):
            subject_id += 1
            age_entry = rng.normal(56.0, 6.5)
            sex = int(rng.integers(0, 2))
            lat_true = tpl["lat"] + rng.normal(0.0, 0.85)
            lon_true = tpl["lon"] + rng.normal(0.0, 0.95)
            lat_obs = lat_true if rng.random() < float(cfg["observed_latlon_fraction"]) else math.nan
            lon_obs = lon_true if math.isfinite(lat_obs) else math.nan
            pgs = 0.55 * pcs[row_idx, 0] - 0.25 * pcs[row_idx, 2] + rng.normal(0.0, 1.0)
            rows.append(
                {
                    "subject_id": subject_id,
                    "subpopulation": tpl["subpop"],
                    "continent": tpl["continent"],
                    "superpopulation": tpl["superpop"],
                    "age_entry": float(age_entry),
                    "sex": sex,
                    "lat_true": float(lat_true),
                    "lon_true": float(lon_true),
                    "lat_obs": None if not math.isfinite(lat_obs) else float(lat_obs),
                    "lon_obs": None if not math.isfinite(lon_obs) else float(lon_obs),
                    "pgs_raw": float(pgs),
                    **{f"pc{pc_idx + 1}": float(pcs[row_idx, pc_idx]) for pc_idx in range(16)},
                }
            )
    meta = {
        "seed": seed,
        "raw_n": len(rows),
        "subpopulations": [tpl["subpop"] for tpl in templates],
    }
    dump_json(out_dir / "raw_generation_metadata.json", meta)
    return rows, meta


def impute_and_upsample(rows: list[dict[str, Any]], cfg: dict[str, Any], smoke: bool) -> list[dict[str, Any]]:
    target_n = int(cfg["smoke_target_n"] if smoke else cfg["target_n"])
    split_rng = np.random.default_rng(int(cfg["split_seed"]))
    by_subpop: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_subpop[str(row["subpopulation"])].append(row)
    out = [dict(r) for r in rows]
    batch = int(cfg.get("upsample_batch_size", 5000))
    next_id = max(int(r["subject_id"]) for r in out) + 1
    subpops = sorted(by_subpop.keys())
    weights = np.array([len(by_subpop[s]) for s in subpops], dtype=float)
    weights /= np.sum(weights)
    while len(out) < target_n:
        remaining = target_n - len(out)
        step = min(batch, remaining)
        sampled_subpops = split_rng.choice(subpops, size=step, replace=True, p=weights)
        for sp in sampled_subpops:
            source = by_subpop[sp][int(split_rng.integers(0, len(by_subpop[sp])))]
            row = dict(source)
            row["subject_id"] = next_id
            next_id += 1
            row["age_entry"] = float(np.clip(float(row["age_entry"]) + split_rng.normal(0.0, 0.9), 35.0, 82.0))
            row["pgs_raw"] = float(float(row["pgs_raw"]) + split_rng.normal(0.0, 0.12))
            lat_true = float(row["lat_true"]) + float(split_rng.normal(0.0, 0.05))
            lon_true = float(row["lon_true"]) + float(split_rng.normal(0.0, 0.05))
            row["lat_true"] = lat_true
            row["lon_true"] = lon_true
            if split_rng.random() < float(cfg["observed_latlon_fraction"]):
                row["lat_obs"] = lat_true + float(split_rng.normal(0.0, 0.02))
                row["lon_obs"] = lon_true + float(split_rng.normal(0.0, 0.02))
            else:
                row["lat_obs"] = None
                row["lon_obs"] = None
            out.append(row)
    ref_subpop = str(cfg["reference_subpopulation"])
    ref_rows = [r for r in out if str(r["subpopulation"]) == ref_subpop]
    ref_lat = float(np.mean([float(r["lat_true"]) for r in ref_rows]))
    ref_lon = float(np.mean([float(r["lon_true"]) for r in ref_rows]))
    for row in out:
        lat_obs = row.get("lat_obs")
        lon_obs = row.get("lon_obs")
        row["lat_final"] = float(lat_obs) if lat_obs is not None else ref_lat
        row["lon_final"] = float(lon_obs) if lon_obs is not None else ref_lon
    return out


def attach_outcomes(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(cfg["seed"]) + 17)
    lat = np.array([float(r["lat_final"]) for r in rows], dtype=float)
    lon = np.array([float(r["lon_final"]) for r in rows], dtype=float)
    pcs = np.column_stack([np.array([float(r[f"pc{i}"]) for r in rows], dtype=float) for i in range(1, 17)])
    pgs = np.array([float(r["pgs_raw"]) for r in rows], dtype=float)
    age = np.array([float(r["age_entry"]) for r in rows], dtype=float)
    sex = np.array([float(r["sex"]) for r in rows], dtype=float)
    disease_prob = disease_probability(lat, lon, pcs, pgs, age, sex)
    disease = rng.binomial(1, disease_prob).astype(int)
    shape, scale = survival_generation_params(cfg)
    surv_scale = survival_scale(lat, lon, pcs, pgs, age, sex)
    u = np.clip(rng.random(len(rows)), 1e-12, 1.0 - 1e-12)
    event_time = scale * surv_scale * (-np.log(1.0 - u)) ** (1.0 / shape)
    censor_time = rng.uniform(4.5, 14.5, size=len(rows))
    observed_time = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(int)
    for idx, row in enumerate(rows):
        row["phenotype_prob"] = float(disease_prob[idx])
        row["phenotype"] = int(disease[idx])
        row["time"] = float(observed_time[idx])
        row["event"] = int(event[idx])
    return rows


def write_cohort_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "subject_id",
        "subpopulation",
        "continent",
        "superpopulation",
        "age_entry",
        "sex",
        "lat_true",
        "lon_true",
        "lat_final",
        "lon_final",
        "pgs_raw",
        *[f"pc{i}" for i in range(1, 17)],
        "phenotype_prob",
        "phenotype",
        "time",
        "event",
        "lat_final_std",
        "lon_final_std",
        "age_entry_std",
        "pgs_std",
        *[f"pc{i}_std" for i in range(1, 17)],
    ]
    write_csv_rows(path, rows, fieldnames)


def split_rows(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(int(cfg["split_seed"]))
    idx = np.arange(len(rows))
    rng.shuffle(idx)
    split = int(round(float(cfg["train_fraction"]) * len(rows)))
    train_idx = set(int(i) for i in idx[:split])
    train = [rows[i] for i in range(len(rows)) if i in train_idx]
    test = [rows[i] for i in range(len(rows)) if i not in train_idx]
    return train, test


def add_standardized_columns(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    numeric_cols = ["age_entry", "lat_final", "lon_final", "pgs_raw", *[f"pc{i}" for i in range(1, 17)]]
    standardization: dict[str, dict[str, float]] = {}
    for col in numeric_cols:
        tr = np.array([float(r[col]) for r in train_rows], dtype=float)
        te = np.array([float(r[col]) for r in test_rows], dtype=float)
        tr_std, te_std, mu, sd = zscore_train_test(tr, te)
        standardization[col] = {"mean": float(mu), "sd": float(sd)}
        out_col = col.replace("_raw", "") + "_std"
        for i, row in enumerate(train_rows):
            row[out_col] = float(tr_std[i])
        for i, row in enumerate(test_rows):
            row[out_col] = float(te_std[i])
    return standardization


def do_prepare(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.target_n is not None:
        cfg["target_n"] = int(args.target_n)
    if args.smoke_target_n is not None:
        cfg["smoke_target_n"] = int(args.smoke_target_n)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, raw_meta = generate_raw_cohort(cfg, out_dir, args.smoke)
    rows = impute_and_upsample(rows, cfg, args.smoke)
    rows = attach_outcomes(rows, cfg)
    train_rows, test_rows = split_rows(rows, cfg)
    standardization = add_standardized_columns(train_rows, test_rows)
    write_cohort_csv(out_dir / "all_cohort.csv", rows)
    write_cohort_csv(out_dir / "disease_train.csv", train_rows)
    write_cohort_csv(out_dir / "disease_test.csv", test_rows)
    write_cohort_csv(out_dir / "survival_train.csv", train_rows)
    write_cohort_csv(out_dir / "survival_test.csv", test_rows)
    prep_meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(args.config),
        "raw_generation": raw_meta,
        "n_total": len(rows),
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "smoke": bool(args.smoke),
        "standardization": standardization,
    }
    dump_json(out_dir / "prep_metadata.json", prep_meta)
    print(f"Wrote prepared data to {out_dir}")
    return 0


def build_method_specs(cfg: dict[str, Any]) -> list[MethodSpec]:
    out = []
    for item in cfg.get("methods", []):
        spec = MethodSpec(
            name=str(item["name"]),
            dataset=str(item["dataset"]),
            backend=str(item["backend"]),
            family=str(item["family"]),
            spatial_basis=str(item["spatial_basis"]),
            centers=int(item["centers"]) if item.get("centers") is not None else None,
            include_sigma=bool(item.get("include_sigma", False)),
            survival_likelihood=(
                str(item["survival_likelihood"])
                if item.get("survival_likelihood") is not None
                else None
            ),
            survival_distribution=(
                str(item["survival_distribution"])
                if item.get("survival_distribution") is not None
                else None
            ),
            marginal_slope=bool(item.get("marginal_slope", False)),
            scale_dimensions=bool(item.get("scale_dimensions", False)),
            z_column=(
                str(item["z_column"])
                if item.get("z_column") is not None
                else None
            ),
            pc_count=int(item.get("pc_count", 16)),
            mean_linkwiggle_knots=(
                int(item["mean_linkwiggle_knots"])
                if item.get("mean_linkwiggle_knots") is not None
                else None
            ),
            logslope_linkwiggle_knots=(
                int(item["logslope_linkwiggle_knots"])
                if item.get("logslope_linkwiggle_knots") is not None
                else None
            ),
            timewiggle_knots=(
                int(item["timewiggle_knots"])
                if item.get("timewiggle_knots") is not None
                else None
            ),
        )
        validate_method_spec(spec)
        out.append(spec)
    return out


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return max(sum(1 for _ in fh) - 1, 0)


def rust_formula_classification(spec: MethodSpec) -> tuple[str, str]:
    """Build mean + sigma formulas for large-scale classification lanes.

    PCs enter as a SINGLE JOINT smooth over the grouping manifold using the
    lane's `spatial_basis`; no per-PC linear terms, no separate per-axis
    smooths. Lat/lon coordinates are NOT used as predictors. The mean and
    sigma blocks share the joint-PC term so any heteroscedastic structure
    is over the same grouping surface as the location surface.

    For binomial location-scale lanes (``include_sigma=True``) the CLI
    rejects the default logit link with ``binomial blended-inverse-link
    location-scale fitting requires link(type=blended(...))`` because the
    Logit + ``--predict-noise`` path is gated on an explicit mixture/blended
    link spec. The marginal-slope companion already pins ``link(type=probit)``
    for the same reason, so we use the same standard-link choice here to
    keep all large-scale grouping-manifold lanes routed through a comparable
    binomial inverse-link.
    """
    centers = int(spec.centers or 60)
    pc_count = int(spec.pc_count)
    spatial = _large_scale_pc_smooth_term(spec.spatial_basis, pc_count, centers)
    mean_terms = [
        "pgs_std",
        "sex",
        "smooth(age_entry_std)",
        spatial,
    ]
    if spec.include_sigma:
        mean_terms.append("link(type=probit)")
    sigma_terms = [
        "smooth(age_entry_std)",
        spatial,
    ]
    return "phenotype ~ " + " + ".join(mean_terms), " + ".join(sigma_terms)


def rust_marginal_slope_formula_classification(spec: MethodSpec, centers: int) -> tuple[str, str]:
    """Build mean and logslope formulas for large-scale marginal-slope classification.

    Uses the shared joint-PC helper so duchon / thinplate / matern lanes all
    route through the same grouping-manifold contract.
    """
    spatial = _large_scale_pc_smooth_term(spec.spatial_basis, int(spec.pc_count), centers)
    mean_terms = [
        "link(type=probit)",
        "sex",
        "smooth(age_entry_std)",
        spatial,
    ]
    logslope_terms = [
        "smooth(age_entry_std)",
        spatial,
    ]
    if spec.mean_linkwiggle_knots is not None:
        mean_terms.append(
            f"linkwiggle(internal_knots={int(spec.mean_linkwiggle_knots)})"
        )
    if spec.logslope_linkwiggle_knots is not None:
        logslope_terms.append(
            f"linkwiggle(internal_knots={int(spec.logslope_linkwiggle_knots)})"
        )
    mean_formula = "phenotype ~ " + " + ".join(mean_terms)
    logslope_formula = " + ".join(logslope_terms)
    return mean_formula, logslope_formula


def run_rust_marginal_slope_classification(
    spec: MethodSpec,
    train_csv: Path,
    test_csv: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Run 16D marginal-slope Duchon classification with optional anisotropy."""
    rust_bin = load_or_build_rust_binary()
    train_rows = count_csv_rows(train_csv)
    centers = int(spec.centers or 24)
    preflight = preflight_marginal_slope_large_scale(
        n_train=train_rows,
        d_pc=int(spec.pc_count),
        centers=centers,
        linkwiggle_knots=spec.mean_linkwiggle_knots,
        scorewarp_knots=spec.logslope_linkwiggle_knots,
    )
    print("\n".join(preflight.lines), file=sys.stderr, flush=True)
    require_shared_ctn_columns(spec, train_csv, test_csv)
    ctn_train_csv = train_csv
    ctn_test_csv = test_csv
    mean_formula, logslope_formula = rust_marginal_slope_formula_classification(spec, centers)
    z_column = spec.z_column or PGS_CTN_Z_COLUMN
    if z_column != PGS_CTN_Z_COLUMN:
        raise RuntimeError(
            f"{spec.name} marginal-slope requires {PGS_CTN_Z_COLUMN}; got {z_column}"
        )
    model_path = out_dir / f"{spec.name}.model.json"
    pred_path = out_dir / f"{spec.name}.pred.csv"
    fit_cmd = [
        str(rust_bin), "fit",
        "--logslope-formula", logslope_formula,
        "--z-column", z_column,
        "--out", str(model_path),
    ]
    if spec.scale_dimensions:
        fit_cmd.append("--scale-dimensions")
    fit_cmd.extend([str(ctn_train_csv), mean_formula])
    t0 = time.perf_counter()
    rc, out, err = run_cmd_stream(fit_cmd, cwd=ROOT)
    disease_fit_sec = time.perf_counter() - t0
    fit_sec = disease_fit_sec
    if rc != 0:
        raise RuntimeError(err.strip() or out.strip() or f"{spec.name} marginal-slope fit failed")
    pred_cmd = [str(rust_bin), "predict", str(model_path), str(ctn_test_csv), "--out", str(pred_path)]
    t1 = time.perf_counter()
    rc, out, err = run_cmd_stream(pred_cmd, cwd=ROOT)
    predict_sec = time.perf_counter() - t1
    if rc != 0:
        raise RuntimeError(err.strip() or out.strip() or f"{spec.name} marginal-slope predict failed")
    pred_rows = read_csv_rows(pred_path)
    pred = np.array([float(r["mean"]) for r in pred_rows], dtype=float)
    y_train = csv_numeric_column(ctn_train_csv, "phenotype")
    y_test = csv_numeric_column(ctn_test_csv, "phenotype")
    metrics = classification_metrics(y_test, pred, float(np.mean(y_train)))
    return {
        "fit_sec": fit_sec,
        "shared_ctn_preprocessed": True,
        "disease_fit_sec": disease_fit_sec,
        "predict_sec": predict_sec,
        "metrics": metrics,
        "prediction_path": str(pred_path),
        "model_spec": (
            f"Rust 16D {spec.spatial_basis} marginal-slope"
            f"{' aniso' if spec.scale_dimensions else ''}"
            f" (z={z_column}, CTN=train-only transformation-normal, centers={centers}) holdout"
        ),
    }


def rust_survival_marginal_slope_formula_parts(spec: MethodSpec, centers: int) -> tuple[str, str]:
    spatial = _large_scale_pc_smooth_term(spec.spatial_basis, int(spec.pc_count), centers)
    mean_terms = ["sex", "smooth(age_entry_std)", spatial]
    if spec.timewiggle_knots is not None:
        mean_terms.append(f"timewiggle(internal_knots={int(spec.timewiggle_knots)})")
    if spec.mean_linkwiggle_knots is not None:
        mean_terms.append(f"linkwiggle(internal_knots={int(spec.mean_linkwiggle_knots)})")
    logslope_terms = ["smooth(age_entry_std)", spatial]
    if spec.logslope_linkwiggle_knots is not None:
        logslope_terms.append(
            f"linkwiggle(internal_knots={int(spec.logslope_linkwiggle_knots)})"
        )
    fit_formula = (
        f"Surv({SURVIVAL_ENTRY_COLUMN}, time, event) ~ " + " + ".join(mean_terms)
    )
    return fit_formula, " + ".join(logslope_terms)


def rust_survival_formula_rhs(spec: MethodSpec) -> str:
    if spec.survival_likelihood == "marginal-slope":
        return rust_survival_marginal_slope_formula_parts(
            spec,
            int(spec.centers or 24),
        )[0].split(" ~ ", 1)[1]

    distribution = spec.survival_distribution
    if distribution is None:
        raise RuntimeError(
            f"survival method '{spec.name}' is missing survival_distribution"
        )
    pc_count = int(spec.pc_count)
    centers = int(spec.centers or 60)
    pc_term = _large_scale_pc_smooth_term(spec.spatial_basis, pc_count, centers)
    terms = [
        "pgs_std",
        "sex",
        "smooth(age_entry_std)",
        pc_term,
        f"survmodel(spec=net, distribution={distribution})",
    ]
    if spec.mean_linkwiggle_knots is not None:
        terms.append(
            f"linkwiggle(internal_knots={int(spec.mean_linkwiggle_knots)})"
        )
    if spec.timewiggle_knots is not None:
        terms.append(
            f"timewiggle(internal_knots={int(spec.timewiggle_knots)})"
        )
    return " + ".join(terms)


def rust_survival_formula(spec: MethodSpec) -> str:
    return f"Surv({SURVIVAL_ENTRY_COLUMN}, time, event) ~ {rust_survival_formula_rhs(spec)}"


def survival_eval_horizon_from_rows(rows: list[dict[str, Any]]) -> float:
    times = np.array([float(r["time"]) for r in rows], dtype=float)
    horizon = float(np.median(times))
    if (not np.isfinite(horizon)) or horizon <= 0.0:
        horizon = 1.0
    return horizon


def prepare_survival_benchmark_rows(
    rows: list[dict[str, Any]],
    *,
    prediction_horizon: float | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        prepared = dict(row)
        prepared[SURVIVAL_ENTRY_COLUMN] = 0.0
        if prediction_horizon is not None:
            prepared["time"] = float(prediction_horizon)
        out.append(prepared)
    return out


def write_survival_benchmark_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    prediction_horizon: float | None = None,
) -> None:
    prepared_rows = prepare_survival_benchmark_rows(
        rows,
        prediction_horizon=prediction_horizon,
    )
    if not prepared_rows:
        raise RuntimeError(f"cannot write empty survival benchmark frame to {path}")
    fieldnames = [SURVIVAL_ENTRY_COLUMN] + [
        key for key in prepared_rows[0].keys() if key != SURVIVAL_ENTRY_COLUMN
    ]
    write_csv_rows(path, prepared_rows, fieldnames)


def csv_numeric_column(path: Path, col: str) -> np.ndarray:
    rows = read_csv_rows(path)
    return np.array([float(r[col]) for r in rows], dtype=float)


def run_rust_classification(spec: MethodSpec, train_csv: Path, test_csv: Path, out_dir: Path) -> dict[str, Any]:
    rust_bin = load_or_build_rust_binary()
    mean_formula, sigma_formula = rust_formula_classification(spec)
    model_path = out_dir / f"{spec.name}.model.json"
    pred_path = out_dir / f"{spec.name}.pred.csv"
    fit_cmd = [str(rust_bin), "fit"]
    if spec.include_sigma:
        fit_cmd.extend(["--predict-noise", sigma_formula])
    fit_cmd.extend(["--out", str(model_path), str(train_csv), mean_formula])
    t0 = time.perf_counter()
    rc, out, err = run_cmd_stream(fit_cmd, cwd=ROOT)
    fit_sec = time.perf_counter() - t0
    if rc != 0:
        raise RuntimeError(err.strip() or out.strip() or f"{spec.name} fit failed")
    pred_cmd = [str(rust_bin), "predict", str(model_path), str(test_csv), "--out", str(pred_path)]
    t1 = time.perf_counter()
    rc, out, err = run_cmd_stream(pred_cmd, cwd=ROOT)
    predict_sec = time.perf_counter() - t1
    if rc != 0:
        raise RuntimeError(err.strip() or out.strip() or f"{spec.name} predict failed")
    pred_rows = read_csv_rows(pred_path)
    pred = np.array([float(r["mean"]) for r in pred_rows], dtype=float)
    y_train = csv_numeric_column(train_csv, "phenotype")
    y_test = csv_numeric_column(test_csv, "phenotype")
    metrics = classification_metrics(y_test, pred, float(np.mean(y_train)))
    return {
        "fit_sec": fit_sec,
        "predict_sec": predict_sec,
        "metrics": metrics,
        "prediction_path": str(pred_path),
        "model_spec": f"Rust {spec.spatial_basis} {'GAMLSS' if spec.include_sigma else 'GAM'} holdout",
    }


def run_rust_survival(spec: MethodSpec, train_csv: Path, test_csv: Path, out_dir: Path) -> dict[str, Any]:
    rust_bin = load_or_build_rust_binary()
    model_path = out_dir / f"{spec.name}.model.json"
    pred_path = out_dir / f"{spec.name}.pred.csv"
    likelihood_mode = spec.survival_likelihood
    if likelihood_mode is None:
        raise RuntimeError(
            f"survival method '{spec.name}' is missing survival_likelihood"
        )
    train_rows_raw = read_csv_rows(train_csv)
    test_rows_raw = read_csv_rows(test_csv)
    centers = int(spec.centers or 24)
    logslope_formula = None
    fit_csv = train_csv
    prediction_rows_raw = test_rows_raw
    train_metric_rows_raw = train_rows_raw
    if likelihood_mode == "marginal-slope":
        centers = int(spec.centers or 24)
        preflight = preflight_marginal_slope_large_scale(
            n_train=len(train_rows_raw),
            d_pc=int(spec.pc_count),
            centers=centers,
            linkwiggle_knots=spec.mean_linkwiggle_knots,
            scorewarp_knots=spec.logslope_linkwiggle_knots,
        )
        print("\n".join(preflight.lines), file=sys.stderr, flush=True)
        require_shared_ctn_columns(spec, train_csv, test_csv)
        fit_csv = train_csv
        train_metric_rows_raw = train_rows_raw
        prediction_rows_raw = test_rows_raw
        fit_formula, logslope_formula = rust_survival_marginal_slope_formula_parts(spec, centers)
    else:
        fit_formula = rust_survival_formula(spec)
    prediction_preflight = preflight_survival_prediction(
        n_rows=len(prediction_rows_raw),
        grid_points=len(
            _survival_score_grid(
                np.array([float(r["time"]) for r in train_rows_raw], dtype=float)
            )
        ),
    )
    print("\n".join(prediction_preflight.lines), file=sys.stderr, flush=True)
    horizon = survival_eval_horizon_from_rows(train_rows_raw)
    with tempfile.TemporaryDirectory(prefix="gam_large_scale_survival_", dir=out_dir) as td:
        td_path = Path(td)
        train_fit_path = td_path / "train_fit.csv"
        test_pred_input_path = td_path / "test_predict.csv"
        write_survival_benchmark_csv(train_fit_path, read_csv_rows(fit_csv))
        write_survival_benchmark_csv(
            test_pred_input_path,
            prediction_rows_raw,
            prediction_horizon=horizon,
        )
        fit_cmd = [
            str(rust_bin),
            "fit",
            "--survival-likelihood",
            likelihood_mode,
            "--time-basis",
            "ispline",
            "--time-degree",
            "3",
            "--time-num-internal-knots",
            "8",
            "--time-smooth-lambda",
            "0.01",
            "--ridge-lambda",
            "1e-6",
            "--out",
            str(model_path),
        ]
        if likelihood_mode == "marginal-slope":
            fit_cmd.extend(["--logslope-formula", logslope_formula or "1"])
            fit_cmd.extend(["--z-column", spec.z_column or PGS_CTN_Z_COLUMN])
            if spec.scale_dimensions:
                fit_cmd.append("--scale-dimensions")
        if spec.timewiggle_knots is not None or likelihood_mode == "marginal-slope":
            fit_cmd.extend(["--baseline-target", "gompertz-makeham"])
        fit_cmd.extend([str(train_fit_path), fit_formula])
        t0 = time.perf_counter()
        rc, out, err = run_cmd_stream(fit_cmd, cwd=ROOT)
        survival_fit_sec = time.perf_counter() - t0
        fit_sec = survival_fit_sec
        if rc != 0:
            raise RuntimeError(err.strip() or out.strip() or f"{spec.name} fit failed")
        pred_cmd = [
            str(rust_bin),
            "predict",
            str(model_path),
            str(test_pred_input_path),
            "--out",
            str(pred_path),
        ]
        t1 = time.perf_counter()
        rc, out, err = run_cmd_stream(pred_cmd, cwd=ROOT)
        predict_sec = time.perf_counter() - t1
        if rc != 0:
            raise RuntimeError(err.strip() or out.strip() or f"{spec.name} predict failed")
        survival_grid = _survival_score_grid(
            np.array([float(r["time"]) for r in train_rows_raw], dtype=float)
        )
        native_t0 = time.perf_counter()
        native_survival, native_pred_path = predict_native_survival_matrix(
            rust_bin=rust_bin,
            spec=spec,
            model_path=model_path,
            base_rows=prediction_rows_raw,
            grid=survival_grid,
            out_dir=out_dir,
        )
        predict_sec += time.perf_counter() - native_t0
    train_rows = [
        {k: (float(v) if k in {"time", "event"} else v) for k, v in r.items()}
        for r in train_metric_rows_raw
    ]
    test_rows = [
        {k: (float(v) if k in {"time", "event"} else v) for k, v in r.items()}
        for r in prediction_rows_raw
    ]
    metrics = survival_metrics_from_native_probabilities(
        train_rows,
        test_rows,
        survival_grid,
        native_survival,
    )
    return {
        "fit_sec": fit_sec,
        "shared_ctn_preprocessed": likelihood_mode == "marginal-slope",
        "survival_fit_sec": survival_fit_sec,
        "predict_sec": predict_sec,
        "metrics": metrics,
        "prediction_path": str(pred_path),
        "native_grid_prediction_path": str(native_pred_path),
        "model_spec": (
            f"{fit_formula} [survival-likelihood={likelihood_mode}; "
            + (
                f"logslope={logslope_formula}; z={spec.z_column or PGS_CTN_Z_COLUMN}; "
                if likelihood_mode == "marginal-slope"
                else ""
            )
            + f"native survival probability scoring; predict_horizon={horizon:.6g}; centers={centers}]"
        ),
    }


def run_method(spec: MethodSpec, prep_dir: Path, out_dir: Path) -> dict[str, Any]:
    disease_train = prep_dir / "disease_train.csv"
    disease_test = prep_dir / "disease_test.csv"
    survival_train = prep_dir / "survival_train.csv"
    survival_test = prep_dir / "survival_test.csv"
    if spec.dataset == "disease":
        if spec.backend == "rust_gam" and spec.marginal_slope:
            result = run_rust_marginal_slope_classification(spec, disease_train, disease_test, out_dir)
        elif spec.backend == "rust_gam":
            result = run_rust_classification(spec, disease_train, disease_test, out_dir)
        else:
            raise RuntimeError(f"unsupported disease backend '{spec.backend}'")
    elif spec.dataset == "survival":
        if spec.backend == "rust_survival":
            result = run_rust_survival(spec, survival_train, survival_test, out_dir)
        else:
            raise RuntimeError(f"unsupported survival backend '{spec.backend}'")
    else:
        raise RuntimeError(f"unsupported dataset '{spec.dataset}'")
    return {
        "method": spec.name,
        "dataset": spec.dataset,
        "family": spec.family,
        **result,
    }


def do_run_method(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    specs = {spec.name: spec for spec in build_method_specs(cfg)}
    if args.method not in specs:
        raise RuntimeError(f"unknown method '{args.method}'")
    spec = specs[args.method]
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if getattr(args, "emit_routing_log", False):
        routing_log_path = out_dir / f"{spec.name}.routing.log"
        # Truncate so re-runs do not accumulate stale routing tokens.
        routing_log_path.write_text("", encoding="utf-8")
        os.environ["LARGE_SCALE_ROUTING_LOG_PATH"] = str(routing_log_path)
        # log_plan emits at info level. If RUST_LOG is already configured by
        # the caller we leave it alone; otherwise default to gam=info so the
        # `[OUTER]` line reaches stderr.
        os.environ.setdefault("RUST_LOG", "gam=info")
    try:
        result = run_method(spec, args.prep_dir.resolve(), out_dir)
        payload = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "ok",
            **result,
        }
    except TimeoutError as exc:
        # Per-command wall-clock budget exhausted. This is an EXPECTED
        # failure mode at large scale (the [HEARTBEAT] line and
        # [PHASE summary] above already explain WHICH phase was running
        # when the budget ran out, so a Python stack trace adds no
        # information and just clutters the log). Emit a one-line
        # `[ERROR]` so the failure is visibly distinct from a random
        # crash and parsers can grep for `status=timeout`.
        print(
            f"[ERROR] method={spec.name} status=timeout "
            f"timeout_sec={_CMD_TIMEOUT_SEC} reason={exc}",
            file=sys.stderr,
            flush=True,
        )
        payload = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "timeout",
            "method": spec.name,
            "dataset": spec.dataset,
            "family": spec.family,
            "timeout_sec": _CMD_TIMEOUT_SEC,
            "error": str(exc),
        }
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        payload = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "method": spec.name,
            "dataset": spec.dataset,
            "family": spec.family,
            "error": str(exc),
        }
    dump_json(args.out_json.resolve(), payload)
    print(f"Wrote {args.out_json}")
    if payload["status"] != "ok":
        return 1
    return 0


def read_json_files(paths: list[Path]) -> list[dict[str, Any]]:
    out = []
    for path in paths:
        out.append(json.loads(path.read_text(encoding="utf-8")))
    return out


def make_metric_table(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table = []
    for row in results:
        metrics = row.get("metrics", {})
        base = {
            "method": row.get("method"),
            "dataset": row.get("dataset"),
            "family": row.get("family"),
            "status": row.get("status"),
            "fit_sec": row.get("fit_sec"),
            "predict_sec": row.get("predict_sec"),
            "model_spec": row.get("model_spec"),
        }
        merged = dict(base)
        merged.update(metrics if isinstance(metrics, dict) else {})
        table.append(merged)
    return table


def plot_aggregate(results: list[dict[str, Any]], prep_dir: Path, out_dir: Path) -> list[Path]:
    return []

def do_aggregate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    result_files = sorted(args.results_dir.resolve().glob("*.json"))
    if not result_files:
        raise RuntimeError(f"no result json files found in {args.results_dir}")
    results = read_json_files(result_files)
    dump_json(out_dir / "config_snapshot.json", cfg)
    if (args.prep_dir.resolve() / "prep_metadata.json").exists():
        shutil.copy2(args.prep_dir.resolve() / "prep_metadata.json", out_dir / "prep_metadata.json")
    combined = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    dump_json(out_dir / "combined_results.json", combined)
    metric_table = make_metric_table(results)
    fieldnames = sorted({k for row in metric_table for k in row.keys()})
    write_csv_rows(out_dir / "all_metrics.csv", metric_table, fieldnames)
    write_csv_rows(out_dir / "per_run_results.csv", results, sorted({k for row in results for k in row.keys()}))
    for dataset in sorted({str(r.get("dataset")) for r in results}):
        sub = [r for r in metric_table if str(r.get("dataset")) == dataset]
        write_csv_rows(out_dir / f"{dataset}_metrics.csv", sub, fieldnames)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = plot_aggregate(results, args.prep_dir.resolve(), plot_dir)
    zip_path = out_dir / "large_scale_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        already_added: set[str] = set()
        core_paths = [
            out_dir / "combined_results.json",
            out_dir / "all_metrics.csv",
            out_dir / "per_run_results.csv",
            out_dir / "config_snapshot.json",
        ]
        if (out_dir / "prep_metadata.json").exists():
            core_paths.append(out_dir / "prep_metadata.json")
        for path in core_paths:
            zf.write(path, arcname=path.name)
            already_added.add(path.name)
        for path in sorted(out_dir.glob("*_metrics.csv")):
            if path.name in already_added:
                continue
            zf.write(path, arcname=path.name)
        for path in result_files:
            zf.write(path, arcname=f"results/{path.name}")
        for path in plot_paths:
            zf.write(path, arcname=f"plots/{path.name}")
    print(f"Wrote {zip_path}")
    return 0


def do_matrix(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    payload = {"include": [spec.__dict__ for spec in build_method_specs(cfg)]}
    print(json.dumps(payload))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Large-scale synthetic benchmark runner")
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prep.add_argument("--out-dir", type=Path, required=True)
    prep.add_argument("--smoke", action="store_true")
    prep.add_argument("--target-n", type=int, default=None)
    prep.add_argument("--smoke-target-n", type=int, default=None)
    prep.set_defaults(func=do_prepare)

    prep_ctn = sub.add_parser("prepare-ctn")
    prep_ctn.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prep_ctn.add_argument("--prep-dir", type=Path, required=True)
    prep_ctn.add_argument("--out-dir", type=Path, required=True)
    prep_ctn.set_defaults(func=do_prepare_ctn)

    matrix = sub.add_parser("matrix")
    matrix.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    matrix.set_defaults(func=do_matrix)

    run = sub.add_parser("run-method")
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--prep-dir", type=Path, required=True)
    run.add_argument("--method", required=True)
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--out-json", type=Path, required=True)
    run.add_argument(
        "--emit-routing-log",
        action="store_true",
        help=(
            "Capture `[OUTER]` log lines from the Rust subprocess (which include "
            "the `solver=...;hessian=...;matrix-free=...` routing token) into a "
            "sidecar file at <out-dir>/<method>.routing.log. Sets RUST_LOG=gam=info "
            "in the subprocess environment so log_plan output reaches stderr."
        ),
    )
    run.set_defaults(func=do_run_method)

    agg = sub.add_parser("aggregate")
    agg.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    agg.add_argument("--prep-dir", type=Path, required=True)
    agg.add_argument("--results-dir", type=Path, required=True)
    agg.add_argument("--out-dir", type=Path, required=True)
    agg.set_defaults(func=do_aggregate)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
