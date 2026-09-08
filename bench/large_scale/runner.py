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
from typing import Any, Iterable

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
    slope_linkwiggle_knots: int | None = None
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
        ("slope_linkwiggle_knots", spec.slope_linkwiggle_knots),
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
            if spec.slope_linkwiggle_knots is None:
                raise RuntimeError(
                    f"survival marginal-slope method '{spec.name}' must set slope_linkwiggle_knots"
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
    ctn_train_input_path = out_dir / f"{spec.name}.pgs_ctn.train_input.csv"
    ctn_test_input_path = out_dir / f"{spec.name}.pgs_ctn.test_input.csv"
    ctn_train_score_path = out_dir / f"{spec.name}.pgs_ctn.train_score.csv"
    ctn_test_score_path = out_dir / f"{spec.name}.pgs_ctn.test_score.csv"
    formula = _ctn_formula(spec.pc_count, centers)
    ctn_columns = [PGS_RAW_COLUMN, *_pc_std_columns(spec.pc_count)]
    # The CTN is part of the declared estimator, not an approximate feature
    # engineering prepass. Fit it on every training row and score the untouched
    # training/heldout covariates. The former 5k/20k subsample plus coordinate-
    # wise PC clamping changed both the fit measure and prediction inputs; it
    # also created the extreme-tail behavior it was later trying to mask.
    write_csv_rows(
        ctn_train_input_path,
        train_rows,
        ctn_columns,
    )
    write_csv_rows(
        ctn_test_input_path,
        test_rows,
        ctn_columns,
    )
    fit_cmd = [
        str(rust_bin),
        "fit",
        "--transformation-normal",
        "--out",
        str(ctn_model_path),
        str(ctn_train_input_path),
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
        f"conditional PGS CTN exact fit measure: all {len(train_rows)} training rows",
        "conditional PGS CTN scoring uses untouched training and heldout covariates",
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


def survival_concordance(
    event_times: np.ndarray,
    risk_score: np.ndarray,
    events: np.ndarray,
) -> float | None:
    value = _rust().survival_concordance(
        _f64_list(event_times),
        _f64_list(risk_score),
        _f64_list(events),
    )
    return None if value is None else float(value)


def _survival_null_curve(train_times: np.ndarray, train_events: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.asarray(
        _rust().survival_null_curve_from_train(
            _f64_list(train_times),
            _f64_list(train_events),
            _f64_list(grid),
        ),
        dtype=float,
    )


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
    # Dedicated buffer for the solver's instrumentation markers so they
    # survive stderr-buffer rollover when a long-running cmd produces
    # >MAX_CAPTURE_CHARS of HEARTBEAT noise. Lets `_emit_phase_summary`
    # still find the markers even after a 40-min run.
    #
    # Membership is decided by `_is_instrumentation_line`, which is
    # derived from `_INSTRUMENTATION_MARKERS` — the same tuple every
    # aggregator in `_emit_phase_summary` is checked against. A marker
    # family the summary parses but this filter drops would leave that
    # aggregator parsing an empty string on every real run while still
    # passing unit tests that hand it a synthetic line, which is how
    # ten of these aggregations were silently inert before gam#2617.
    phase_buf: list[str] = []
    # Single-element lists so the pump thread can mutate the running
    # byte total and the rollover count without a nonlocal binding.
    phase_total = [0]
    phase_dropped = [0]
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
                        if _is_instrumentation_line(line):
                            phase_capture.append(line)
                            phase_total[0] += len(line)
                            while phase_total[0] > MAX_PHASE_CAPTURE_CHARS:
                                phase_total[0] -= len(phase_capture.pop(0))
                                phase_dropped[0] += 1
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
        _emit_phase_summary(
            "".join(phase_buf),
            preview,
            timed_out=True,
            rc=124,
            dropped_marker_lines=phase_dropped[0],
        )
        raise TimeoutError(msg)
    captured_stderr = "".join(err_buf)
    if (routing_path := _routing_log_path()) is not None:
        _append_routing_lines(routing_path, captured_stderr)
    # Emit a per-phase wall-clock summary parsed from the gam binary's
    # instrumentation markers so CI logs end with a quick-glance
    # breakdown of CTN / margslope / standard-GAM / location-scale phase
    # timings plus the inner-solver health verdicts. We parse the
    # dedicated `phase_buf` (which retains marker lines even after
    # stderr buffer rollover) rather than `captured_stderr`.
    _emit_phase_summary(
        "".join(phase_buf),
        preview,
        timed_out=False,
        rc=rc,
        dropped_marker_lines=phase_dropped[0],
    )
    return rc, "".join(out_buf), captured_stderr


# ----------------------------------------------------------------------
# Solver instrumentation parsers.
#
# The engine narrates its inner state on stderr with bracketed markers.
# This block is the reading half of that contract: one regex per marker
# family, aggregated by `_emit_phase_summary` into the per-run verdict
# lines a CI reviewer reads. Every pattern here is derived from a
# CURRENT emission site in `crates/`, not from an older revision of the
# log format — `tests/bench_large_scale_runner_test.py` holds each one
# against a sample built from the live format string and fails when the
# producer moves or disappears (gam#2617).
#
# Three families the previous version of this block parsed are NOT here,
# because the engine no longer emits them:
#   * `[OUTER guard] convergence-guard re-eval` — no emission site.
#   * `[OUTER hessian-route] reason=subspace_forced_dense` — the routing
#     reason set is now below_crossover / callback_row_pair_work /
#     dense_memory_budget / kernel_absent / large_k / large_linear_work /
#     large_n_moderate_p / large_p / family_op, with no forced-dense label.
#   * the biobank-family `declining analytic outer Hessian` tags — that
#     line is now emitted under a single `[standard-GAM]` tag.
# Two `[IFT-QUALITY]` fields the previous version aggregated
# (`drho_norm`, `h_pen_logdet`) are likewise absent from the current
# marker, so the Δρ-magnitude and log|H_pen| distributions are not
# reconstructed here.
# ----------------------------------------------------------------------

_PHASE_END_PATTERN = re.compile(
    r"\[PHASE\]\s+([\w\-]+(?:\([\w\-/]+\))?)\s+(?:fit\s+)?(?:end|done)\s+elapsed=([\d.]+)s"
)
_PHASE_START_PATTERN = re.compile(r"\[PHASE\]\s+([\w\-]+(?:\([\w\-/]+\))?)\s+(?:fit\s+)?start")
_BFGS_SUMMARY_PATTERN = re.compile(
    r"\[OUTER summary\]\s+BFGS\s+(converged|hit max_iter|line-search failed|failed)(?:\s+in\s+(\d+)\s+iters)?\s+elapsed=([\d.]+)s"
)

# Inner-PIRLS cap transitions from the first-order bridge. The bare
# transition count says the adaptive cap moved; the quality variant
# additionally captures the feedback snapshot that drove the margin, so
# the summary can report which policy branch fired (poor LM fidelity,
# poor IFT prediction, or geometric backoff after a cap hit).
_SCHEDULE_TRANSITION_PATTERN = re.compile(
    r"\[OUTER schedule\]\s+inner-PIRLS cap transition.*?prev=(\d+)\s+new=(\d+)"
)
_SCHEDULE_QUALITY_PATTERN = re.compile(
    r"\[OUTER schedule\]\s+inner-PIRLS cap transition.*?"
    r"last_iters=(\d+)\s+converged=(true|false)\s+"
    r"ift_residual=(\S+)\s+accept_rho=(\S+)\s+"
    r"prev=(\d+)\s+new=(\d+)"
)

# Per-iter inner-Newton wall-clock, and its split across the four
# sub-phases that drive the cost: curvature assembly, the (H+λI)δ=-g
# solve, the predicted-reduction quadratic form, and the candidate
# gain-ratio evaluation. Summed over a fit, the dominant sub-phase says
# which optimization to ship next.
_PIRLS_ITER_END_PATTERN = re.compile(
    r"\[PIRLS iter-end\]\s+iter=\s*(\d+)\s+elapsed=([\d.]+)s"
)
_PIRLS_ITER_BREAKDOWN_PATTERN = re.compile(
    r"\[PIRLS iter-breakdown\]\s+iter=\s*(\d+)\s+attempts=(\d+)"
    r"\s+curvature=([\d.]+)s\s+solve=([\d.]+)s\s+predred=([\d.]+)s"
    r"\s+candidate=([\d.]+)s\s+other=([\d.]+)s"
)

# Per-iter curvature kind: the Debug rendering of `HessianCurvatureKind`
# (`Observed` or `Fisher`). The Observed path converges faster but is
# not guaranteed PD; a high Fisher fraction is a direct signal of
# observed-Hessian PD failures at scale. The trailing `source=` field
# distinguishes a rebuilt assembly from a reused one and is deliberately
# not captured — the fraction is over kinds, not over rebuilds.
_PIRLS_CURVATURE_KIND_PATTERN = re.compile(
    r"\[STAGE\] PIRLS update_with_curvature iter=\d+\s+curvature=(\w+)"
)

# Fisher fallbacks that fire mid-LM-loop (the iter-start assembly
# succeeded but the candidate evaluation forced a retry), and the
# `force_fisher_for_rest` lock-in that fires at most once per solve
# when consecutive fallbacks cross the threshold. Together they
# separate transient fallbacks from a sustained Fisher-only state.
_PIRLS_MID_ITER_FISHER_PATTERN = re.compile(
    r"\[PIRLS\] mid-iter Fisher fallback iter=(\d+)\s+reason=(\w+)"
)
_PIRLS_FORCE_FISHER_PATTERN = re.compile(
    r"\[PIRLS\] force_fisher_for_rest engaged at iter=(\d+)\s+"
    r"\(consecutive_fisher_fallbacks=(\d+)\)\s+reason=(\w+)"
)

# Per-iter LM trajectory. `log10_ratio` is log10(final λ / start λ):
# negative means the trust region is expanding toward Newton, positive
# means rejections are shrinking it. `accept_rho` near 1 means the
# quadratic model is faithful. Both render as `NaN` when undefined.
_PIRLS_LM_TRAJECTORY_PATTERN = re.compile(
    r"\[PIRLS lm-trajectory\]\s+iter=\s*(\d+)\s+"
    r"start_lambda=([\d.eE+\-]+)\s+final_lambda=([\d.eE+\-]+)\s+"
    r"log10_ratio=([\d.eE+\-nNaA]+)\s+accept_rho=([\d.eE+\-nNaA]+)\s+"
    r"attempts=(\d+)"
)

# One line per completed PIRLS solve, carrying the geometric
# convergence rate (g_final/g_initial)^(1/iters) and the terminal
# `PirlsStatus`. Healthy inner Newton sits below 0.5; 0.7 and up means
# the solve is grinding.
_PIRLS_SOLVE_END_PATTERN = re.compile(
    r"\[PIRLS solve-end\]\s+iters=(\d+)\s+elapsed=([\d.]+)s\s+"
    r"g_norm_initial=\S+\s+g_norm_final=\S+\s+"
    r"convergence_rate=([\deE.+\-nNaA]+)\s+status=(\w+)"
)

# Outer-Hessian routing decision and the wall-clock of the path it
# chose. The route line always carries the crossover inputs; the
# `family_op` early return reports `scale_prefers_operator=irrelevant`
# because that branch never consults the (n,p,k) crossover.
_OUTER_HESSIAN_ROUTE_PATTERN = re.compile(
    r"\[OUTER hessian-route\]\s+choice=(\w+)\s+reason=(\w+)\s+"
    r"n=(\d+)\s+p=(\d+)\s+k=(\d+)\s+"
    r"callback_kernel=(true|false)\s+subspace_trace=(true|false)\s+"
    r"scale_prefers_operator=(true|false|irrelevant)"
)
_OUTER_HESSIAN_ELAPSED_PATTERN = re.compile(
    r"\[OUTER hessian-elapsed\]\s+choice=(\w+)\s+reason=(\w+)\s+"
    r"n=(\d+)\s+p=(\d+)\s+k=(\d+)\s+elapsed=([\d.]+)s"
)

# Outer-eval wall-clock per evaluation order. The gap between the outer
# eval total and (pirls_total + outer_h_total) is the remaining work —
# score computation, gradient assembly, warm-start prediction.
_OUTER_EVAL_END_PATTERN = re.compile(
    r"\[STAGE\] outer eval end order=(\w+) elapsed=([\d.]+)s"
)

# Seed-screening cascade summary: one per outer fit. `stages_used=1`
# means the heuristic seeds passed at the tightest cap tier; higher
# means the cascade had to escalate, which is startup cost at scale.
_SEED_CASCADE_PATTERN = re.compile(
    r"\[OUTER\][^\n]*seed screening cascade complete\s+"
    r"elapsed=([\d.]+)s\s+stages_used=(\d+)\s+"
    r"final_cap=(\w+)\s+ranked=(\d+)/(\d+)"
)

# κ-optimization driver instrumentation: one `[KAPPA-PHASE]` per closure
# invocation plus a `[KAPPA-PHASE-SUMMARY]` at exit. Two summary
# variants are emitted (the exact-joint driver prefixes `n_rows=` and
# appends the no-free-lunch miss counters), so the fields are captured
# by NAME rather than by position — a positional read would silently
# shift by one against the longer variant.
_KAPPA_PHASE_PATTERN = re.compile(
    r"\[KAPPA-PHASE\]\s+phase=(\w+)\s+call=(\d+)(?:\s+order=\S+)?"
    r"(?:\s+design_revision=\S+)?"
    r"\s+theta_norm=\S+\s+log_kappa_norm=\S+\s+elapsed_s=([\d.]+)"
)
_KAPPA_PHASE_SUMMARY_PATTERN = re.compile(
    r"\[KAPPA-PHASE-SUMMARY\]\s+(?:n_rows=(?P<n_rows>\d+)\s+)?"
    r"log_kappa_dim=(?P<log_kappa_dim>\d+)\s+"
    r"n_cost=(?P<n_cost>\d+)\s+cost_total_s=(?P<cost_total_s>[\d.]+)\s+"
    r"n_eval=(?P<n_eval>\d+)\s+eval_total_s=(?P<eval_total_s>[\d.]+)\s+"
    r"n_efs=(?P<n_efs>\d+)\s+efs_total_s=(?P<efs_total_s>[\d.]+)"
    r".*?optim_total_s=(?P<optim_total_s>[\d.]+)"
)

# Warm-start predictor quality probes, emitted after a non-screening
# PIRLS solve that consumed a predicted β. `quality` is
# ‖β_converged − β_predicted‖ / (1 + ‖β_converged‖) — near 0 when the
# linearization was faithful, order 1 when the prediction was no better
# than flat. `iters` is the inner-Newton count the solve then needed.
#
# The two markers are separate because they measure DIFFERENT
# predictors: `[IFT-QUALITY]` is the implicit-function-theorem
# predictor (and carries the adaptive |Δρ| cap it feeds), while
# `[TANGENT-QUALITY]` is the tangent-line fallback that only fires when
# IFT declines. Folding them into one distribution would attribute the
# fallback's faithfulness to the primary path.
_IFT_QUALITY_PATTERN = re.compile(
    r"\[IFT-QUALITY\]\s+quality=([\deE.+\-nNaA]+)\s+ift=([\deE.+\-nNaA]+)\s+"
    r"pred_residual=([\deE.+\-nNaA]+)\s+cap_predicted=([\deE.+\-nNaA]+)\s+iters=(\d+)"
)
_TANGENT_QUALITY_PATTERN = re.compile(
    r"\[TANGENT-QUALITY\]\s+quality=([\deE.+\-nNaA]+)\s+"
    r"pred_residual=([\deE.+\-nNaA]+)\s+iters=(\d+)"
)

# Predictor rejection and no-op counters. A reject is a fall-through
# (cap exceeded, factorization failed, non-finite output, dim
# mismatch); a no-op is an effectively-zero ρ-step where the predictor
# returned the cached β unchanged. Keeping them apart is what makes the
# accept rate readable: no-ops measure outer-optimizer behaviour, not
# predictor quality.
_IFT_REJECTED_PATTERN = re.compile(r"\[IFT-REJECTED\]\s+reason=(\w+)")
_IFT_NOOP_PATTERN = re.compile(r"\[IFT-NOOP\]\s+reason=(\w+)")
_TANGENT_PREDICT_PATTERN = re.compile(
    r"\[TANGENT-PREDICT\]\s+alpha=([\deE.+\-]+)\s+cap=([\deE.+\-]+)\s+"
    r"drho_step_norm_sq=([\deE.+\-]+)\s+drho_prev_norm_sq=([\deE.+\-]+)"
)
_TANGENT_REJECTED_PATTERN = re.compile(r"\[TANGENT-REJECTED\]\s+reason=(\w+)")
_TANGENT_NOOP_PATTERN = re.compile(r"\[TANGENT-NOOP\]\s+reason=(\w+)")

# IFT factor-cache hit/miss. The penalized-Hessian Cholesky is O(p³)/3
# — seconds at large p — so the hit rate sizes what the cache saves and
# the miss elapsed sizes what it still pays.
_IFT_CACHE_HIT_PATTERN = re.compile(
    r"\[IFT-CACHE\]\s+outcome=hit\s+drho_dim=(\d+)(?:\s+p=(\d+))?"
)
_IFT_CACHE_MISS_PATTERN = re.compile(
    r"\[IFT-CACHE\]\s+outcome=miss\s+drho_dim=(\d+)(?:\s+p=(\d+))?\s+elapsed=([\d.]+)s"
)

# NaN / Inf in an intermediate of the outer-Hessian, leverage, or
# adjoint computation. Zero in a healthy fit; any count is a bug signal,
# and the captured field name says which intermediate broke.
_OUTER_NONFINITE_PATTERN = re.compile(r"\[OUTER non-finite\]\s+(\S+)")

# Substrings that identify a line as solver instrumentation worth
# keeping in the dedicated marker buffer. Each entry must be a
# substring of a marker family SOME aggregator in
# `_emit_phase_summary` reads; conversely every family the summary
# reads must be matched by some entry here, or that aggregator sees an
# empty string on every real run. The test suite asserts both
# directions, because "parser exists, capture drops its input" is a
# failure that unit tests feeding synthetic lines cannot see.
_INSTRUMENTATION_MARKERS: tuple[str, ...] = (
    "[PHASE]",
    "[OUTER summary]",
    "[OUTER non-finite]",
    "[OUTER schedule] inner-PIRLS cap transition",
    "[OUTER hessian-route]",
    "[OUTER hessian-elapsed]",
    "seed screening cascade complete",
    "[STAGE] outer eval end",
    "[STAGE] PIRLS update_with_curvature",
    "[PIRLS iter-end]",
    "[PIRLS iter-breakdown]",
    "[PIRLS lm-trajectory]",
    "[PIRLS solve-end]",
    "[PIRLS] mid-iter Fisher fallback",
    "[PIRLS] force_fisher_for_rest",
    "[KAPPA-PHASE",
    "[IFT-QUALITY]",
    "[IFT-REJECTED]",
    "[IFT-NOOP]",
    "[IFT-CACHE]",
    "[TANGENT-PREDICT]",
    "[TANGENT-REJECTED]",
    "[TANGENT-QUALITY]",
    "[TANGENT-NOOP]",
)

# The marker buffer keeps per-iter lines, so a multi-hour fit can emit
# far more of them than the run needs held in memory at once. Bound it
# and count what rolls off: a truncated buffer makes the distributions
# partial, and `marker_lines_dropped=` in the summary is what tells a
# reviewer the percentiles are computed on a suffix rather than
# quietly reporting them as if they covered the whole run.
MAX_PHASE_CAPTURE_CHARS = 8 * 1024 * 1024


def _is_instrumentation_line(line: str) -> bool:
    return any(marker in line for marker in _INSTRUMENTATION_MARKERS)


def _percentiles(values: list[float]) -> tuple[float, float, float]:
    """Return (p50, p95, max) of a non-empty list, by the same
    nearest-rank convention every aggregator below uses: sort, then
    index at `n // 2` and `int(0.95 * n)` clamped to the last element.
    """
    ordered = sorted(values)
    n = len(ordered)
    return (
        ordered[n // 2],
        ordered[min(n - 1, int(0.95 * n))],
        ordered[-1],
    )


def _finite(values: Iterable[str]) -> list[float]:
    """Parse decimal strings, dropping anything non-numeric or NaN.

    The markers render undefined quantities as `NaN`, which must not
    enter a percentile — a single NaN would poison the sort order
    rather than being visibly absent.
    """
    out: list[float] = []
    for text in values:
        if not text:
            continue
        try:
            value = float(text)
        except (TypeError, ValueError):
            continue
        if value == value:
            out.append(value)
    return out


def _combine_fit_verdicts(
    warm_start: str | None,
    pirls: str | None,
    curvature: str | None = None,
) -> str:
    """Combine the per-axis health verdicts into one fit verdict on the
    worst-wins total ordering DEGRADED > MARGINAL > HEALTHY > NO-DATA.

    A `None` axis (its markers never fired) ranks as NO-DATA. Worst-wins
    rather than averaging: a fit that is HEALTHY on two axes and
    DEGRADED on the third is DEGRADED, and the per-axis fields on the
    `[FIT health]` line say which one tripped it.
    """
    rank = {"DEGRADED": 3, "MARGINAL": 2, "HEALTHY": 1, "NO-DATA": 0}
    inv_rank = {value: key for key, value in rank.items()}
    worst = max(
        rank.get(warm_start or "NO-DATA", 0),
        rank.get(pirls or "NO-DATA", 0),
        rank.get(curvature or "NO-DATA", 0),
    )
    return inv_rank[worst]


def _dominant_axis_for_verdict(
    combined: str,
    *,
    warm_start: str | None,
    pirls: str | None,
    curvature: str | None,
) -> str:
    """Name the axis that drove `combined`, so a CI scraper can alert on
    the failing axis without re-deriving worst-of-three.

    Ties are broken toward `pirls` first (the central inner-Newton
    diagnostic), then `warm_start`, then `curvature`. An all-missing
    combination reports `none`.
    """
    if combined == "NO-DATA":
        return "none"
    rank = {"DEGRADED": 3, "MARGINAL": 2, "HEALTHY": 1, "NO-DATA": 0}
    target = rank[combined]
    for name, verdict in (
        ("pirls", pirls),
        ("warm_start", warm_start),
        ("curvature", curvature),
    ):
        if rank.get(verdict or "NO-DATA", 0) == target:
            return name
    return "none"


def _curvature_health_verdict(
    *,
    fisher_frac: float | None,
    force_fisher_n: int,
) -> tuple[str, str]:
    """Classify observed-Hessian reliability from the Fisher-fallback
    counters. Returns (verdict, detail).

    HEALTHY   fisher_frac < 0.05 and no lock-in
    MARGINAL  fisher_frac < 0.20 and no lock-in — occasional transient
              fallbacks, Observed still mostly usable
    DEGRADED  fisher_frac >= 0.20, or any `force_fisher_for_rest`
              lock-in: at least one solve abandoned Observed entirely
    NO-DATA   the curvature-kind markers never fired
    """
    if fisher_frac is None:
        return ("NO-DATA", "fisher_frac=n/a force_fisher_n=0")
    detail = f"fisher_frac={fisher_frac:.2f} force_fisher_n={force_fisher_n}"
    if force_fisher_n > 0 or fisher_frac >= 0.20:
        return ("DEGRADED", detail)
    if fisher_frac >= 0.05:
        return ("MARGINAL", detail)
    return ("HEALTHY", detail)


def _pirls_health_verdict(*, rates: list[float]) -> tuple[str, str]:
    """Classify the inner Newton's per-solve geometric convergence rates.
    Returns (verdict, detail).

    HEALTHY   p95(rate) < 0.5 — 95% of solves strongly converging. The
              threshold is on p95 rather than max so one slow solve in a
              hundred clean ones does not flip the verdict; the outlier
              is still visible in the `max=` field.
    MARGINAL  p50(rate) < 0.5 and max(rate) < 0.85 — median solve fast,
              nothing in the saturation regime.
    DEGRADED  otherwise.
    NO-DATA   no finite rates captured.
    """
    if not rates:
        return ("NO-DATA", "n_solves=0")
    p50, p95, rmax = _percentiles(rates)
    detail = f"n_solves={len(rates)} p50={p50:.3f} p95={p95:.3f} max={rmax:.3f}"
    if p95 < 0.5:
        return ("HEALTHY", detail)
    if p50 < 0.5 and rmax < 0.85:
        return ("MARGINAL", detail)
    return ("DEGRADED", detail)


def _warm_start_health_verdict(
    *,
    n_accepts: int,
    n_rejects: int,
    n_noops: int,
    residuals: list[float],
    n_outer_nonfinite: int = 0,
    n_tangent_accepts: int = 0,
    tangent_p50: float | None = None,
) -> tuple[str, str]:
    """Classify the warm-start machinery on two axes. Returns
    (verdict, detail).

      coverage  = accepts / (accepts + rejects + noops)
      residual  = the accepted calls' prediction quality

    HEALTHY   coverage >= 0.70, p50 < 0.05, p95 < 0.20, no
              outer-non-finite. The p95 clause is a saturation guard:
              a clean median over a tail of poor predictions still
              means ~5% of solves started from a bad warm start.
    MARGINAL  coverage >= 0.30 or p50 < 0.30, no outer-non-finite.
    DEGRADED  any outer-non-finite (broken geometry invalidates the
              faithfulness measurement outright); or the predictor was
              tried and never delivered a prediction; or the residuals
              meet neither tier.
    NO-DATA   the predictor was never tried at all.

    Tangent-line statistics ride along in the detail string so both
    predictors are visible at a glance, but the tier stays IFT-driven:
    tangent-line is the fallback, not the primary path.
    """
    denom = max(n_accepts + n_rejects + n_noops, 1)
    coverage = n_accepts / denom
    if residuals:
        p50_resid, p95_resid, _ = _percentiles(residuals)
    else:
        p50_resid = float("nan")
        p95_resid = float("nan")
    detail = (
        f"coverage={coverage:.2f} p50_resid={p50_resid:.2e} "
        f"p95_resid={p95_resid:.2e} "
        f"n_accepts={n_accepts} n_rejects={n_rejects} n_noops={n_noops} "
        f"n_outer_nonfinite={n_outer_nonfinite}"
    )
    if n_tangent_accepts > 0:
        if tangent_p50 is not None and tangent_p50 == tangent_p50:
            detail += (
                f" n_tangent_accepts={n_tangent_accepts} tangent_p50={tangent_p50:.2e}"
            )
        else:
            detail += f" n_tangent_accepts={n_tangent_accepts}"
    if n_outer_nonfinite > 0:
        return ("DEGRADED", detail)
    if not residuals:
        if n_accepts + n_rejects + n_noops == 0:
            return ("NO-DATA", detail)
        return ("DEGRADED", detail)
    if coverage >= 0.70 and p50_resid < 0.05 and p95_resid < 0.20:
        return ("HEALTHY", detail)
    if coverage >= 0.30 or p50_resid < 0.30:
        return ("MARGINAL", detail)
    return ("DEGRADED", detail)


def _emit_phase_summary(
    captured_stderr: str,
    cmd_preview: str,
    *,
    timed_out: bool = False,
    rc: int = 0,
    dropped_marker_lines: int = 0,
) -> None:
    """Aggregate the run's instrumentation markers into the closing CI
    lines: one `[PHASE summary]`, then the per-axis health verdicts and
    the combined `[FIT health]`.

    Each aggregator is guarded on its own markers being present, so a
    fit that never exercises a code path contributes no field for it
    rather than a zero that reads like a measurement.
    """
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

    # --- outer optimizer ------------------------------------------------
    schedule_transitions = _SCHEDULE_TRANSITION_PATTERN.findall(captured_stderr)
    if schedule_transitions:
        parts.append(f"sched_transitions={len(schedule_transitions)}")
    sched_quality = _SCHEDULE_QUALITY_PATTERN.findall(captured_stderr)
    if sched_quality:
        n_unconverged = sum(1 for row in sched_quality if row[1] == "false")
        n_poor_ift = len(
            [value for value in _finite(row[2] for row in sched_quality) if value >= 0.10]
        )
        n_poor_rho = 0
        for _last_iters, converged, _ift, rho_text, _prev, _new in sched_quality:
            rho = _finite([rho_text])
            if rho and rho[0] < 0.5:
                n_poor_rho += 1
        parts.append(
            f"sched_quality_n={len(sched_quality)} "
            f"sched_unconv={n_unconverged} "
            f"sched_poor_ift={n_poor_ift} "
            f"sched_poor_accept_rho={n_poor_rho}"
        )

    outer_h_route = _OUTER_HESSIAN_ROUTE_PATTERN.findall(captured_stderr)
    outer_h_elapsed = _OUTER_HESSIAN_ELAPSED_PATTERN.findall(captured_stderr)
    if outer_h_elapsed:
        choice_counts: dict[str, int] = {}
        reason_secs: dict[str, float] = {}
        for choice, reason, _n, _p, _k, secs in outer_h_elapsed:
            choice_counts[choice] = choice_counts.get(choice, 0) + 1
            reason_secs[reason] = reason_secs.get(reason, 0.0) + float(secs)
        dominant = max(reason_secs, key=lambda key: reason_secs[key])
        choice_pieces = " ".join(
            f"outer_h_{choice}={count}" for choice, count in sorted(choice_counts.items())
        )
        # A route line with no matching elapsed line means the assembly
        # errored or the process died mid-build; that gap is signal, not
        # noise, so it gets its own field instead of being smoothed over.
        parts.append(
            f"outer_h_calls={len(outer_h_elapsed)} "
            f"outer_h_total={sum(reason_secs.values()):.1f}s "
            f"{choice_pieces} "
            f"outer_h_dom_reason={dominant}@{reason_secs[dominant]:.1f}s "
            f"outer_h_route_no_elapsed={max(0, len(outer_h_route) - len(outer_h_elapsed))}"
        )
    elif outer_h_route:
        reason_counts: dict[str, int] = {}
        for _choice, reason, *_rest in outer_h_route:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        dominant = max(reason_counts, key=lambda key: reason_counts[key])
        parts.append(
            f"outer_h_INCOMPLETE outer_h_routes={len(outer_h_route)} "
            f"outer_h_dom_reason={dominant}"
        )

    seed_cascades = _SEED_CASCADE_PATTERN.findall(captured_stderr)
    if seed_cascades:
        seeds_total = sum(int(row[4]) for row in seed_cascades)
        parts.append(
            f"seed_cascade_n={len(seed_cascades)} "
            f"seed_cascade_elapsed={sum(float(row[0]) for row in seed_cascades):.1f}s "
            f"seed_cascade_escalated={sum(1 for row in seed_cascades if int(row[1]) >= 2)} "
            f"seed_cascade_stages_total={sum(int(row[1]) for row in seed_cascades)} "
            f"seed_cascade_rank_rate="
            f"{sum(int(row[3]) for row in seed_cascades) / max(seeds_total, 1):.2f}"
        )

    outer_eval_ends = _OUTER_EVAL_END_PATTERN.findall(captured_stderr)
    if outer_eval_ends:
        order_counts: dict[str, int] = {}
        order_secs: dict[str, float] = {}
        for order, secs in outer_eval_ends:
            order_counts[order] = order_counts.get(order, 0) + 1
            order_secs[order] = order_secs.get(order, 0.0) + float(secs)
        order_pieces = " ".join(
            f"outer_eval_{order}={order_counts[order]}@{order_secs[order]:.1f}s"
            for order in sorted(order_counts)
        )
        parts.append(
            f"outer_eval_n={len(outer_eval_ends)} "
            f"outer_eval_total={sum(order_secs.values()):.1f}s {order_pieces}"
        )

    # --- inner Newton ---------------------------------------------------
    pirls_iter_secs = _finite(
        secs for _iter, secs in _PIRLS_ITER_END_PATTERN.findall(captured_stderr)
    )
    if pirls_iter_secs:
        p50, p95, pmax = _percentiles(pirls_iter_secs)
        parts.append(
            f"pirls_iters={len(pirls_iter_secs)} "
            f"pirls_total={sum(pirls_iter_secs):.1f}s "
            f"pirls_p50={p50:.3f}s pirls_p95={p95:.3f}s pirls_max={pmax:.3f}s"
        )

    curvature_kinds = _PIRLS_CURVATURE_KIND_PATTERN.findall(captured_stderr)
    fisher_frac: float | None = None
    if curvature_kinds:
        kind_counts: dict[str, int] = {}
        for kind in curvature_kinds:
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
        fisher_frac = kind_counts.get("Fisher", 0) / len(curvature_kinds)
        kind_pieces = " ".join(
            f"pirls_curv_{kind}={count}" for kind, count in sorted(kind_counts.items())
        )
        parts.append(
            f"pirls_curv_n={len(curvature_kinds)} {kind_pieces} "
            f"pirls_fisher_frac={fisher_frac:.2f}"
        )

    mid_iter_fisher = _PIRLS_MID_ITER_FISHER_PATTERN.findall(captured_stderr)
    if mid_iter_fisher:
        parts.append(
            f"pirls_mid_iter_fisher_n={len(mid_iter_fisher)} "
            f"pirls_mid_iter_gain_rejection="
            f"{sum(1 for row in mid_iter_fisher if row[1] == 'gain_rejection')} "
            f"pirls_mid_iter_candidate_err="
            f"{sum(1 for row in mid_iter_fisher if row[1] == 'candidate_err')}"
        )

    force_fisher = _PIRLS_FORCE_FISHER_PATTERN.findall(captured_stderr)
    if force_fisher:
        force_reasons: dict[str, int] = {}
        for _iter, _count, reason in force_fisher:
            force_reasons[reason] = force_reasons.get(reason, 0) + 1
        reason_pieces = " ".join(
            f"pirls_force_fisher_{reason}={count}"
            for reason, count in sorted(force_reasons.items())
        )
        parts.append(f"pirls_force_fisher_n={len(force_fisher)} {reason_pieces}")

    breakdown = _PIRLS_ITER_BREAKDOWN_PATTERN.findall(captured_stderr)
    if breakdown:
        sub_totals = {
            "curv": sum(float(row[2]) for row in breakdown),
            "solve": sum(float(row[3]) for row in breakdown),
            "predred": sum(float(row[4]) for row in breakdown),
            "cand": sum(float(row[5]) for row in breakdown),
        }
        other_total = sum(float(row[6]) for row in breakdown)
        timed_sum = sum(sub_totals.values())
        if timed_sum > 0.0:
            dominant = max(sub_totals, key=lambda key: sub_totals[key])
            parts.append(
                f"pirls_attempts={sum(int(row[1]) for row in breakdown)} "
                f"pirls_dom={dominant}@{sub_totals[dominant] / timed_sum * 100:.0f}% "
                f"pirls_curv={sub_totals['curv']:.1f}s "
                f"pirls_solve={sub_totals['solve']:.1f}s "
                f"pirls_predred={sub_totals['predred']:.1f}s "
                f"pirls_cand={sub_totals['cand']:.1f}s "
                f"pirls_other={other_total:.1f}s"
            )

    lm_traj = _PIRLS_LM_TRAJECTORY_PATTERN.findall(captured_stderr)
    if lm_traj:
        ratios = _finite(row[3] for row in lm_traj)
        rhos = _finite(row[4] for row in lm_traj)
        attempts = [int(row[5]) for row in lm_traj]
        pieces: list[str] = []
        if ratios:
            r_p50, r_p95, _ = _percentiles(ratios)
            pieces.append(f"lm_log10_ratio_p50={r_p50:.2f}")
            pieces.append(f"lm_log10_ratio_p95={r_p95:.2f}")
        if rhos:
            rho_p50 = _percentiles(rhos)[0]
            ordered_rhos = sorted(rhos)
            pieces.append(f"lm_accept_rho_p50={rho_p50:.2f}")
            pieces.append(
                f"lm_accept_rho_p05={ordered_rhos[max(0, int(0.05 * len(ordered_rhos)))]:.2f}"
            )
        if attempts:
            a_p50, a_p95, a_max = _percentiles([float(value) for value in attempts])
            pieces.append(f"lm_attempts_p50={int(a_p50)}")
            pieces.append(f"lm_attempts_p95={int(a_p95)}")
            pieces.append(f"lm_attempts_max={int(a_max)}")
        if pieces:
            parts.append(f"lm_iters={len(lm_traj)} " + " ".join(pieces))

    pirls_solves = _PIRLS_SOLVE_END_PATTERN.findall(captured_stderr)
    pirls_rates = _finite(row[2] for row in pirls_solves)
    if pirls_solves and pirls_rates:
        status_counts = {}
        for row in pirls_solves:
            status_counts[row[3]] = status_counts.get(row[3], 0) + 1
        solve_iters = [float(row[0]) for row in pirls_solves]
        rate_p50, rate_p95, rate_max = _percentiles(pirls_rates)
        iter_p50, iter_p95, iter_max = _percentiles(solve_iters)
        status_pieces = " ".join(
            f"pirls_status_{status}={count}"
            for status, count in sorted(status_counts.items())
        )
        parts.append(
            f"pirls_solves={len(pirls_rates)} pirls_conv_p50={rate_p50:.3f} "
            f"pirls_conv_p95={rate_p95:.3f} pirls_conv_max={rate_max:.3f} "
            f"{status_pieces} "
            f"pirls_solve_iters_p50={int(iter_p50)} "
            f"pirls_solve_iters_p95={int(iter_p95)} "
            f"pirls_solve_iters_max={int(iter_max)}"
        )

    # --- kappa optimization ---------------------------------------------
    kappa_calls = _KAPPA_PHASE_PATTERN.findall(captured_stderr)
    kappa_summaries = [
        match.groupdict()
        for match in _KAPPA_PHASE_SUMMARY_PATTERN.finditer(captured_stderr)
    ]
    kappa_phase_secs: dict[str, list[float]] = {}
    for phase_name, _call, secs in kappa_calls:
        kappa_phase_secs.setdefault(phase_name, []).append(float(secs))
    if kappa_summaries:
        parts.append(
            f"kappa_optims={len(kappa_summaries)} "
            f"kappa_optim_total="
            f"{sum(float(row['optim_total_s']) for row in kappa_summaries):.1f}s "
            f"kappa_cost_calls={sum(int(row['n_cost']) for row in kappa_summaries)} "
            f"kappa_cost_total="
            f"{sum(float(row['cost_total_s']) for row in kappa_summaries):.1f}s "
            f"kappa_eval_calls={sum(int(row['n_eval']) for row in kappa_summaries)} "
            f"kappa_eval_total="
            f"{sum(float(row['eval_total_s']) for row in kappa_summaries):.1f}s "
            f"kappa_efs_calls={sum(int(row['n_efs']) for row in kappa_summaries)} "
            f"kappa_efs_total="
            f"{sum(float(row['efs_total_s']) for row in kappa_summaries):.1f}s"
        )
        # The summary totals are authoritative; the per-call percentiles
        # come from marker lines that survived capture, so they can
        # under-report a distribution but never over-report it. They are
        # what separates one slow call from a uniformly slow workload.
        dist_pieces = []
        for phase_name in sorted(kappa_phase_secs):
            _, phase_p95, phase_max = _percentiles(kappa_phase_secs[phase_name])
            dist_pieces.append(
                f"kappa_{phase_name}_p95={phase_p95:.2f}s "
                f"kappa_{phase_name}_max={phase_max:.2f}s"
            )
        if dist_pieces:
            parts.append(" ".join(dist_pieces))
    elif kappa_calls:
        # Per-call markers with no summary: the κ optimization did not
        # finish. Report per-phase totals AND the distribution, since
        # "one eval_outer ate the budget" and "many fast eval_outer
        # calls accumulated" are different findings that the totals
        # alone collapse together.
        phase_pieces = []
        for phase_name in sorted(kappa_phase_secs):
            secs_list = kappa_phase_secs[phase_name]
            _, phase_p95, phase_max = _percentiles(secs_list)
            phase_pieces.append(
                f"kappa_{phase_name}_calls={len(secs_list)} "
                f"kappa_{phase_name}_total={sum(secs_list):.1f}s "
                f"kappa_{phase_name}_p95={phase_p95:.2f}s "
                f"kappa_{phase_name}_max={phase_max:.2f}s"
            )
        parts.append(f"kappa_optim_INCOMPLETE {' '.join(phase_pieces)}")

    # --- warm-start predictors ------------------------------------------
    ift_quality = _IFT_QUALITY_PATTERN.findall(captured_stderr)
    ift_rejected = _IFT_REJECTED_PATTERN.findall(captured_stderr)
    ift_noops = _IFT_NOOP_PATTERN.findall(captured_stderr)
    tangent_quality = _TANGENT_QUALITY_PATTERN.findall(captured_stderr)
    tangent_predicts = _TANGENT_PREDICT_PATTERN.findall(captured_stderr)
    tangent_rejected = _TANGENT_REJECTED_PATTERN.findall(captured_stderr)
    tangent_noops = _TANGENT_NOOP_PATTERN.findall(captured_stderr)
    # A no-op suppresses the quality marker, so every quality line is a
    # real predict call and the accept count needs no subtraction.
    n_accepts = len(ift_quality)
    n_rejects = len(ift_rejected)
    n_noops = len(ift_noops)

    cache_hits = _IFT_CACHE_HIT_PATTERN.findall(captured_stderr)
    cache_misses = _IFT_CACHE_MISS_PATTERN.findall(captured_stderr)
    if cache_hits or cache_misses:
        n_cache = len(cache_hits) + len(cache_misses)
        miss_secs = [float(row[2]) for row in cache_misses]
        miss_p50 = _percentiles(miss_secs)[0] if miss_secs else 0.0
        miss_max = max(miss_secs) if miss_secs else 0.0
        miss_ps = [int(row[1]) for row in cache_misses if row[1]]
        size_piece = f" ift_cache_miss_max_p={max(miss_ps)}" if miss_ps else ""
        parts.append(
            f"ift_cache_n={n_cache} "
            f"ift_cache_hit_rate={len(cache_hits) / n_cache:.2f} "
            f"ift_cache_miss_secs={sum(miss_secs):.2f} "
            f"ift_cache_miss_p50={miss_p50:.2f}s "
            f"ift_cache_miss_max={miss_max:.2f}s "
            f"ift_cache_paid_rejects={max(0, n_cache - n_accepts)}{size_piece}"
        )

    tangent_residuals = _finite(row[0] for row in tangent_quality)
    if tangent_residuals:
        t_p50, t_p95, t_max = _percentiles(tangent_residuals)
        parts.append(
            f"tangent_quality_predicts={len(tangent_residuals)} "
            f"tangent_p50={t_p50:.2e} tangent_p95={t_p95:.2e} tangent_max={t_max:.2e}"
        )
    tangent_iters = [float(row[2]) for row in tangent_quality if row[2]]
    if tangent_iters:
        i_p50, i_p95, i_max = _percentiles(tangent_iters)
        parts.append(
            f"tangent_iters_p50={int(i_p50)} tangent_iters_p95={int(i_p95)} "
            f"tangent_iters_max={int(i_max)}"
        )

    ift_residuals = _finite(row[0] for row in ift_quality)
    if ift_residuals:
        i_p50, i_p95, i_max = _percentiles(ift_residuals)
        parts.append(
            f"ift_predicts={len(ift_residuals)} ift_p50={i_p50:.2e} "
            f"ift_p95={i_p95:.2e} ift_max={i_max:.2e}"
        )
    ift_iters = [float(row[4]) for row in ift_quality if row[4]]
    if ift_iters:
        i_p50, i_p95, i_max = _percentiles(ift_iters)
        parts.append(
            f"ift_iters_p50={int(i_p50)} ift_iters_p95={int(i_p95)} "
            f"ift_iters_max={int(i_max)}"
        )

    outer_nonfinite = _OUTER_NONFINITE_PATTERN.findall(captured_stderr)
    if outer_nonfinite:
        intermediate_counts: dict[str, int] = {}
        for name in outer_nonfinite:
            intermediate_counts[name] = intermediate_counts.get(name, 0) + 1
        parts.append(
            f"outer_nonfinite={len(outer_nonfinite)} outer_nonfinite_at=["
            + ",".join(f"{name}={count}" for name, count in sorted(intermediate_counts.items()))
            + "]"
        )

    if tangent_predicts or tangent_rejected or tangent_noops:
        alphas = _finite(row[0] for row in tangent_predicts)
        alpha_piece = ""
        if alphas:
            a_p50, _, a_max = _percentiles(alphas)
            alpha_piece = f" tangent_alpha_p50={a_p50:.2f} tangent_alpha_max={a_max:.2f}"
        if tangent_rejected:
            t_reason_counts: dict[str, int] = {}
            for reason in tangent_rejected:
                t_reason_counts[reason] = t_reason_counts.get(reason, 0) + 1
            parts.append(
                f"tangent_predicts={len(tangent_predicts)} "
                f"tangent_rejects={len(tangent_rejected)} "
                "tangent_reasons=["
                + ",".join(
                    f"{reason}={count}" for reason, count in sorted(t_reason_counts.items())
                )
                + f"]{alpha_piece}"
            )
        else:
            parts.append(f"tangent_predicts={len(tangent_predicts)}{alpha_piece}")
        if tangent_noops:
            parts.append(f"tangent_noops={len(tangent_noops)}")
        denom_total = max(len(tangent_predicts) + len(tangent_rejected) + len(tangent_noops), 1)
        denom_active = max(len(tangent_predicts) + len(tangent_rejected), 1)
        parts.append(
            f"tangent_accept_rate={len(tangent_predicts) / denom_total:.2f} "
            f"tangent_accept_rate_active={len(tangent_predicts) / denom_active:.2f}"
        )

    # Every accepted tangent-line prediction should produce exactly one
    # downstream quality marker. Divergence beyond one (the run can be
    # cut off between the two) means instrumentation is dropping
    # markers, which is the failure this whole layer exists to catch.
    if tangent_predicts and abs(len(tangent_predicts) - len(tangent_quality)) > 1:
        parts.append(
            f"tangent_marker_drift=predict={len(tangent_predicts)}"
            f"_vs_quality={len(tangent_quality)}"
        )

    if n_rejects > 0 or n_noops > 0:
        ift_reason_counts: dict[str, int] = {}
        for reason in ift_rejected:
            ift_reason_counts[reason] = ift_reason_counts.get(reason, 0) + 1
        # Two denominators, because they answer different questions.
        # `ift_accept_rate` includes no-ops and so mixes predictor
        # quality with how often the outer takes zero-length steps;
        # `ift_accept_rate_active` excludes them and is the
        # predictor-only signal.
        denom_total = max(n_accepts + n_rejects + n_noops, 1)
        denom_active = max(n_accepts + n_rejects, 1)
        parts.append(
            f"ift_rejects={n_rejects} ift_noops={n_noops} "
            f"ift_accept_rate={n_accepts / denom_total:.2f} "
            f"ift_accept_rate_active={n_accepts / denom_active:.2f} "
            "ift_reasons=["
            + ",".join(
                f"{reason}={count}" for reason, count in sorted(ift_reason_counts.items())
            )
            + "]"
        )

    if dropped_marker_lines:
        parts.append(f"marker_lines_dropped={dropped_marker_lines}")
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

    # --- health verdicts -------------------------------------------------
    warm_start_verdict: str | None = None
    if ift_quality or outer_nonfinite or tangent_quality:
        tangent_p50 = _percentiles(tangent_residuals)[0] if tangent_residuals else None
        warm_start_verdict, detail = _warm_start_health_verdict(
            n_accepts=n_accepts,
            n_rejects=n_rejects,
            n_noops=n_noops,
            residuals=ift_residuals,
            n_outer_nonfinite=len(outer_nonfinite),
            n_tangent_accepts=len(tangent_residuals),
            tangent_p50=tangent_p50,
        )
        print(
            f"[WARM-START health] cmd='{cmd_preview}' "
            f"verdict={warm_start_verdict} {detail}",
            file=sys.stderr,
            flush=True,
        )

    pirls_verdict: str | None = None
    if pirls_rates:
        pirls_verdict, detail = _pirls_health_verdict(rates=pirls_rates)
        print(
            f"[PIRLS health] cmd='{cmd_preview}' verdict={pirls_verdict} {detail}",
            file=sys.stderr,
            flush=True,
        )

    curvature_verdict: str | None = None
    if fisher_frac is not None:
        curvature_verdict, detail = _curvature_health_verdict(
            fisher_frac=fisher_frac,
            force_fisher_n=len(force_fisher),
        )
        print(
            f"[CURVATURE health] cmd='{cmd_preview}' "
            f"verdict={curvature_verdict} {detail}",
            file=sys.stderr,
            flush=True,
        )

    if (
        warm_start_verdict is not None
        or pirls_verdict is not None
        or curvature_verdict is not None
    ):
        combined = _combine_fit_verdicts(
            warm_start_verdict, pirls_verdict, curvature_verdict
        )
        dominant_axis = _dominant_axis_for_verdict(
            combined,
            warm_start=warm_start_verdict,
            pirls=pirls_verdict,
            curvature=curvature_verdict,
        )
        print(
            f"[FIT health] cmd='{cmd_preview}' verdict={combined} "
            f"dominant_axis={dominant_axis} "
            f"warm_start={warm_start_verdict or 'ABSENT'} "
            f"pirls={pirls_verdict or 'ABSENT'} "
            f"curvature={curvature_verdict or 'ABSENT'}",
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
            slope_linkwiggle_knots=(
                int(item["slope_linkwiggle_knots"])
                if item.get("slope_linkwiggle_knots") is not None
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
    """Build mean and slope formulas for large-scale marginal-slope classification.

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
    slope_terms = [
        "smooth(age_entry_std)",
        spatial,
    ]
    if spec.mean_linkwiggle_knots is not None:
        mean_terms.append(
            f"linkwiggle(internal_knots={int(spec.mean_linkwiggle_knots)})"
        )
    if spec.slope_linkwiggle_knots is not None:
        slope_terms.append(
            f"linkwiggle(internal_knots={int(spec.slope_linkwiggle_knots)})"
        )
    mean_formula = "phenotype ~ " + " + ".join(mean_terms)
    slope_formula = " + ".join(slope_terms)
    return mean_formula, slope_formula


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
        scorewarp_knots=spec.slope_linkwiggle_knots,
    )
    print("\n".join(preflight.lines), file=sys.stderr, flush=True)
    require_shared_ctn_columns(spec, train_csv, test_csv)
    ctn_train_csv = train_csv
    ctn_test_csv = test_csv
    mean_formula, slope_formula = rust_marginal_slope_formula_classification(spec, centers)
    z_column = spec.z_column or PGS_CTN_Z_COLUMN
    if z_column != PGS_CTN_Z_COLUMN:
        raise RuntimeError(
            f"{spec.name} marginal-slope requires {PGS_CTN_Z_COLUMN}; got {z_column}"
        )
    model_path = out_dir / f"{spec.name}.model.json"
    pred_path = out_dir / f"{spec.name}.pred.csv"
    fit_cmd = [
        str(rust_bin), "fit",
        "--slope-formula", slope_formula,
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
    # The marginal-slope CLI schema keeps its class-specific `mean` column.
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
    slope_terms = ["smooth(age_entry_std)", spatial]
    if spec.slope_linkwiggle_knots is not None:
        slope_terms.append(
            f"linkwiggle(internal_knots={int(spec.slope_linkwiggle_knots)})"
        )
    fit_formula = (
        f"Surv({SURVIVAL_ENTRY_COLUMN}, time, event) ~ " + " + ".join(mean_terms)
    )
    return fit_formula, " + ".join(slope_terms)


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
    # Standard and location-scale mean models share the model-owned,
    # estimand-explicit CLI schema (#2785/#2803). A missing posterior_mean is a
    # serialization-contract failure, not a reason to reinterpret `mean`.
    pred = np.array([float(r["posterior_mean"]) for r in pred_rows], dtype=float)
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
    slope_formula = None
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
            scorewarp_knots=spec.slope_linkwiggle_knots,
        )
        print("\n".join(preflight.lines), file=sys.stderr, flush=True)
        require_shared_ctn_columns(spec, train_csv, test_csv)
        fit_csv = train_csv
        train_metric_rows_raw = train_rows_raw
        prediction_rows_raw = test_rows_raw
        fit_formula, slope_formula = rust_survival_marginal_slope_formula_parts(spec, centers)
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
            "--out",
            str(model_path),
        ]
        if likelihood_mode == "marginal-slope":
            fit_cmd.extend(["--slope-formula", slope_formula or "1"])
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
                f"slope={slope_formula}; z={spec.z_column or PGS_CTN_Z_COLUMN}; "
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
