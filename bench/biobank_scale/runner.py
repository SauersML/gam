#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
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
DEFAULT_CONFIG = BENCH_DIR / "biobank_scale.yml"
HEARTBEAT_INTERVAL_SEC = 15.0
HEARTBEAT_INITIAL_WINDOW_SEC = 2.0
HEARTBEAT_INITIAL_INTERVAL_SEC = 0.25
MAX_CAPTURE_CHARS = 200000


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
# `do_run_method` sets `BIOBANK_ROUTING_LOG_PATH` to the destination file.
# `run_cmd_stream` then appends every captured stderr line that contains the
# `[OUTER]` log marker emitted by `crate::solver::outer_strategy::log_plan` —
# the line carries the stable `solver=...;hessian=...;matrix-free=...` token
# defined by `OuterPlan::routing_log_line()`. Bench tests scrape this file.
_ROUTING_LOG_OUTER_MARKER = "[OUTER]"


def _routing_log_path() -> Path | None:
    raw = os.environ.get("BIOBANK_ROUTING_LOG_PATH")
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


DEFAULT_BIOBANK_RAM_BUDGET_BYTES = _detect_host_memory_bytes()
BIOBANK_MAX_DENSE_BLOCK_BYTES = 2 * 1024**3
BIOBANK_MAX_DERIVATIVE_DENSE_BYTES = 2 * 1024**3
BIOBANK_SURVIVAL_PREDICTION_CHUNK_ROWS = 8192


# Mirrors of constants in src/families/transformation_normal.rs governing the
# size of the monotonicity response grid built inside the CTN family. Update
# both sides in lockstep if the Rust constants change.
TRANSFORMATION_RESPONSE_GRID_MAX_QUANTILES = 129
TRANSFORMATION_RESPONSE_GRID_SUBDIVISIONS = 4
# Upper-bound estimate for the number of internal knots used by the CTN
# response-direction basis at biobank scale. The exact count is computed
# inside `effective_response_num_internal_knots` in the Rust code; the
# preflight uses a conservative cap so the modelled grid size does not
# under-report. Bumping this up is safe (only loosens the preflight check).
CTN_RESPONSE_INTERNAL_KNOTS_CAP = 32
BIOBANK_DUCHON16D_ORDER = 0
BIOBANK_DUCHON16D_POWER = 8
BIOBANK_DUCHON16D_LENGTH_SCALE = 1.0
PGS_RAW_COLUMN = "pgs_raw"
PGS_CTN_Z_COLUMN = "pgs_ctn_z"
PGS_CTN_FIT_SUBSAMPLE_N = 5000
# At biobank n=320k the fixed 5000-row subsample only covers ~1.6% of the
# 16D continuous PC distribution, which left CTN-z with kurt≈3733 (CI run
# 25338491995). Local n=16k with the same 5000 covers ~31% and got kurt≈7.7.
# Scale K with sqrt(n_train) — keeps O(K^2) cost manageable while ~4×-ing
# the per-cell coverage at biobank scale.
PGS_CTN_FIT_SUBSAMPLE_N_BIOBANK = 20000
PGS_CTN_FIT_SUBSAMPLE_BIOBANK_THRESHOLD = 50000
PGS_CTN_FIT_SUBSAMPLE_SEED = 20260430
PGS_CTN_DIAGNOSTIC_MIN_N = 40
PGS_CTN_DIAGNOSTIC_MAX_ABS_MEAN = 0.30
PGS_CTN_DIAGNOSTIC_MIN_VAR = 0.50
PGS_CTN_DIAGNOSTIC_MAX_VAR = 1.75
SUPPORTED_BIOBANK_SURVIVAL_LIKELIHOODS = {"transformation", "location-scale", "marginal-slope"}
SUPPORTED_BIOBANK_SURVIVAL_DISTRIBUTIONS = {
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
    max_centers: int | None = None


@dataclass(frozen=True)
class BiobankPreflightReport:
    status: str
    lines: list[str]
    largest_single_allocation_bytes: int
    chunk_rows: int | None = None


def gibibytes(nbytes: int) -> float:
    return float(nbytes) / float(1024**3)


def _preflight_status_line(status: str) -> str:
    return f"status: {status}"


def preflight_marginal_slope_biobank(
    *,
    n_train: int,
    d_pc: int,
    centers: int,
    linkwiggle_knots: int | None = None,
    scorewarp_knots: int | None = None,
    ram_budget_bytes: int = DEFAULT_BIOBANK_RAM_BUDGET_BYTES,
) -> BiobankPreflightReport:
    if n_train <= 0 or d_pc <= 0 or centers <= 0:
        raise RuntimeError("biobank preflight dimensions must be positive")
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
    # p_cov) * 8 — surfaced here for reporting so the OOM regression at biobank
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
    if dense_block_bytes > BIOBANK_MAX_DENSE_BLOCK_BYTES:
        failures.append(
            f"estimated dense block: {gibibytes(dense_block_bytes):.1f} GiB exceeds {gibibytes(BIOBANK_MAX_DENSE_BLOCK_BYTES):.1f} GiB"
        )
    if derivative_dense_bytes > BIOBANK_MAX_DERIVATIVE_DENSE_BYTES:
        failures.append(
            f"anisotropic derivative dense estimate: {gibibytes(derivative_dense_bytes):.1f} GiB exceeds {gibibytes(BIOBANK_MAX_DERIVATIVE_DENSE_BYTES):.1f} GiB"
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
        "BIOBANK PREFLIGHT",
        f"n_train: {n_train:,}",
        f"d_pc: {d_pc}",
        f"K_pc: {centers}",
        f"Duchon tuple: order={BIOBANK_DUCHON16D_ORDER}, power={BIOBANK_DUCHON16D_POWER}, length_scale={BIOBANK_DUCHON16D_LENGTH_SCALE:g}",
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
    return BiobankPreflightReport(status, lines, largest)


def preflight_ctn_score_warp(
    *,
    n_train: int,
    p_response: int,
    p_cov: int,
    ram_budget_bytes: int = DEFAULT_BIOBANK_RAM_BUDGET_BYTES,
) -> BiobankPreflightReport:
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
        "BIOBANK PREFLIGHT",
        f"n_train: {n_train:,}",
        f"CTN Kronecker: factored",
        f"p_response: {p_response}",
        f"p_cov: {p_cov}",
        f"avoided dense rowwise Kronecker: {gibibytes(dense_kron_bytes):.1f} GiB",
        f"estimated factored bytes: {gibibytes(factored_bytes):.1f} GiB",
        f"estimated peak RSS: {gibibytes(estimated_peak):.1f} GiB",
        _preflight_status_line(status),
    ]
    lines.extend(f"route note: {failure}" for failure in failures)
    return BiobankPreflightReport(
        status,
        lines,
        max(factored_bytes, p_response * p_cov * F64_BYTES),
    )


def preflight_survival_prediction(
    *,
    n_rows: int,
    grid_points: int,
    chunk_rows: int = BIOBANK_SURVIVAL_PREDICTION_CHUNK_ROWS,
    ram_budget_bytes: int = DEFAULT_BIOBANK_RAM_BUDGET_BYTES,
) -> BiobankPreflightReport:
    if n_rows <= 0 or grid_points <= 0 or chunk_rows <= 0:
        raise RuntimeError("survival prediction preflight dimensions must be positive")
    dense_time_tensor_bytes = n_rows * grid_points * F64_BYTES
    chunked_bytes = min(n_rows, chunk_rows) * grid_points * F64_BYTES
    estimated_peak = chunked_bytes + 256 * 1024**2
    failures: list[str] = []
    if chunked_bytes > BIOBANK_MAX_DENSE_BLOCK_BYTES:
        failures.append("survival prediction chunk is too large")
    if estimated_peak > int(0.80 * ram_budget_bytes):
        failures.append("chunked survival prediction exceeds RAM budget")
    status = "ROUTE" if failures else "PASS"
    lines = [
        "BIOBANK PREFLIGHT",
        f"n_predict: {n_rows:,}",
        f"survival grid: {grid_points}",
        f"survival time tensor: chunked rows={chunk_rows}",
        f"avoided dense n x grid tensor: {gibibytes(dense_time_tensor_bytes):.1f} GiB",
        f"largest single allocation planned: {gibibytes(chunked_bytes):.1f} GiB",
        f"estimated peak RSS: {gibibytes(estimated_peak):.1f} GiB",
        _preflight_status_line(status),
    ]
    lines.extend(f"route note: {failure}" for failure in failures)
    return BiobankPreflightReport(status, lines, chunked_bytes, chunk_rows=chunk_rows)


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
    if spec.max_centers is not None and spec.max_centers <= 0:
        raise RuntimeError(f"method '{spec.name}' requires max_centers > 0")
    for key, value in (
        ("mean_linkwiggle_knots", spec.mean_linkwiggle_knots),
        ("logslope_linkwiggle_knots", spec.logslope_linkwiggle_knots),
        ("timewiggle_knots", spec.timewiggle_knots),
    ):
        if value is not None and value < 3:
            raise RuntimeError(f"method '{spec.name}' requires {key} >= 3")
    if spec.dataset == "disease":
        if spec.backend not in {"rust_gam", "r_mgcv"}:
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
        if spec.survival_likelihood not in SUPPORTED_BIOBANK_SURVIVAL_LIKELIHOODS:
            supported = "|".join(sorted(SUPPORTED_BIOBANK_SURVIVAL_LIKELIHOODS))
            raise RuntimeError(
                f"survival method '{spec.name}' requires survival_likelihood in {supported}"
            )
        if (
            spec.survival_likelihood != "marginal-slope"
            and spec.survival_distribution not in SUPPORTED_BIOBANK_SURVIVAL_DISTRIBUTIONS
        ):
            supported = "|".join(sorted(SUPPORTED_BIOBANK_SURVIVAL_DISTRIBUTIONS))
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
    if spec.backend == "r_mgcv_survival":
        if spec.survival_likelihood is not None or spec.survival_distribution is not None:
            raise RuntimeError(
                f"mgcv survival method '{spec.name}' cannot set survival_likelihood or survival_distribution"
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
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _prediction_z_values(prediction_csv: Path) -> list[float]:
    rows = read_csv_rows(prediction_csv)
    if not rows:
        raise RuntimeError(f"empty transformation-normal prediction file: {prediction_csv}")
    for key in ("z", "z_score", "transformed", "eta", "mean"):
        if key in rows[0]:
            return [float(row[key]) for row in rows]
    raise RuntimeError(
        f"transformation-normal prediction file {prediction_csv} is missing a z-score column"
    )


def _write_rows_like(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"cannot write empty CSV to {path}")
    write_csv_rows(path, rows, list(rows[0].keys()))


def _pc_std_columns(pc_count: int) -> list[str]:
    return [f"pc{i}_std" for i in range(1, int(pc_count) + 1)]


def _biobank_duchon_pc_term(pc_count: int, centers: int) -> str:
    pc_cols = ", ".join(_pc_std_columns(pc_count))
    return (
        f"duchon({pc_cols}, centers={centers}, "
        f"order={BIOBANK_DUCHON16D_ORDER}, power={BIOBANK_DUCHON16D_POWER}, "
        f"length_scale={BIOBANK_DUCHON16D_LENGTH_SCALE:g})"
    )


def _biobank_pc_smooth_term(spatial_basis: str, pc_count: int, centers: int) -> str:
    """Joint multi-D smooth over the PC ancestry axes.

    All biobank lanes treat ancestry as a single object on the joint PC space
    (the production-pipeline strategic goal: PGS calibration via Duchon/TPS on
    joint PC). Lat/lon geographic coordinates are deliberately excluded — the
    relevant continuous structure is genetic ancestry, not geography.
    """
    pc_cols = ", ".join(_pc_std_columns(pc_count))
    if spatial_basis == "duchon":
        return _biobank_duchon_pc_term(pc_count, centers)
    if spatial_basis == "thinplate":
        return f"thinplate({pc_cols}, knots={centers})"
    if spatial_basis == "matern":
        return f"matern({pc_cols}, centers={centers})"
    raise RuntimeError(
        f"unsupported Rust joint-PC spatial basis '{spatial_basis}' "
        "(use duchon, thinplate, or matern)"
    )


def _mgcv_pc_smooth_term(spatial_basis: str, pc_count: int, k: int) -> str:
    """Joint multi-D mgcv smooth over the PC ancestry axes.

    Each biobank lane uses the same joint-PC contract on the mgcv side. The
    PCs enter as one mgcv smooth `s(pc1_std, ..., pcN_std, bs=..., k=...)` —
    never as independent linear terms, never combined with lat/lon.

    Thin-plate (`bs='tp'`) is intentionally not supported here. Multivariate
    `tp` requires `2m > d`, which forces `m = ceil((d+1)/2)`; the resulting
    polynomial null space has dimension `choose(m+d-1, d)` (≈ 7.4e5 for d=16),
    so mgcv allocates an n × M design matrix that overflows R's 32-bit length
    limit and fails with "negative length vectors are not allowed". Use the
    Duchon (`ds`) or Matérn (`gp`) bases for high-d joint-PC smooths instead.
    """
    pc_cols = ", ".join(_pc_std_columns(pc_count))
    if spatial_basis == "duchon":
        return f"s({pc_cols}, bs='ds', k=min({k}, nrow(train_df)-1))"
    if spatial_basis == "matern":
        return f"s({pc_cols}, bs='gp', m=c(-4,1.0), k=min({k}, nrow(train_df)-1))"
    raise RuntimeError(
        f"unsupported mgcv joint-PC spatial basis '{spatial_basis}' "
        "(use duchon or matern; mgcv 'tp' is ill-posed for high-d joint PC)"
    )


def _ctn_formula(pc_count: int, centers: int) -> str:
    return f"{PGS_RAW_COLUMN} ~ {_biobank_duchon_pc_term(pc_count, centers)}"


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
        # (the speed-friendly default at biobank dimensionality) can proceed
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
    ctn_train_pred_path = out_dir / f"{spec.name}.pgs_ctn.train_pred.csv"
    ctn_test_pred_path = out_dir / f"{spec.name}.pgs_ctn.test_pred.csv"
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
    # ~0 or ~1 spuriously, and `z = Φ⁻¹(F)` then blows up: at biobank
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
    # Pick the CTN fit subsample size adaptively. At biobank scale we use
    # PGS_CTN_FIT_SUBSAMPLE_N_BIOBANK to give the CTN basis enough coverage
    # of the 16D continuous PC distribution (the 5000-row default leaves
    # ~1.6% coverage at n=320k vs ~31% at n=16k local; kurt(z) drops from
    # ~3700 toward ~10 with 4× more rows in fit).
    effective_subsample_n = (
        PGS_CTN_FIT_SUBSAMPLE_N_BIOBANK
        if len(train_rows) > PGS_CTN_FIT_SUBSAMPLE_BIOBANK_THRESHOLD
        else PGS_CTN_FIT_SUBSAMPLE_N
    )
    if len(train_rows) > effective_subsample_n:
        rng = np.random.default_rng(PGS_CTN_FIT_SUBSAMPLE_SEED)
        pc_cols = _pc_std_columns(spec.pc_count)
        # Cap per-axis-keep at a small fixed number, NOT (SUBSAMPLE_N // 4 //
        # n_pcs). Why: with the prior formula at SUBSAMPLE_N=5000, pc_count=16
        # we forced 78 rows from each end of every PC. At local n=16k this
        # picks rows at the 0.49% quantile, which the CTN can fit cleanly
        # (kurt(z) ≈ 7.7 in test runs). At biobank n=320k the SAME 78 rows
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
    # The Duchon-order-1 basis has a polynomial nullspace of order 1 (linear
    # in PCs). Outside the convex hull of the fit-time PCs the fitted surface
    # extrapolates linearly *without bound* — the radial basis kernels decay
    # but the linear nullspace term grows. The stratified subsample
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

    def _clamped_rows(rows: list[dict]) -> tuple[list[dict], dict[str, int]]:
        clamp_counts: dict[str, int] = {col: 0 for col in pc_cols_for_clamp}
        out_rows: list[dict] = []
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
        (ctn_train_input_path, ctn_train_pred_path),
        (ctn_test_input_path, ctn_test_pred_path),
    ):
        pred_cmd = [str(rust_bin), "predict", str(ctn_model_path), str(input_path), "--out", str(output_path)]
        rc, out, err = run_cmd_stream(pred_cmd, cwd=ROOT)
        if rc != 0:
            raise RuntimeError(
                err.strip() or out.strip() or f"{spec.name} conditional PGS CTN prediction failed"
            )

    train_aug = _attach_column(train_rows, PGS_CTN_Z_COLUMN, _prediction_z_values(ctn_train_pred_path))
    test_aug = _attach_column(test_rows, PGS_CTN_Z_COLUMN, _prediction_z_values(ctn_test_pred_path))
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


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = (np.asarray(y_true, dtype=float) > 0.5).astype(int)
    p = np.asarray(y_score, dtype=float)
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    if pos == 0 or neg == 0:
        return 0.5
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1, dtype=float)
    pos_rank_sum = float(np.sum(ranks[y == 1]))
    return float((pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))


def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = (np.asarray(y_true, dtype=float) > 0.5).astype(int)
    p = np.asarray(y_score, dtype=float)
    pos = int(np.sum(y == 1))
    if pos == 0:
        return 0.0
    order = np.argsort(-p)
    y_ord = y[order]
    tp = np.cumsum(y_ord == 1)
    fp = np.cumsum(y_ord == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / pos
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return float(np.trapezoid(precision, recall))


def classification_confusion_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y = (np.asarray(y_true, dtype=float) > 0.5).astype(int)
    p = np.asarray(y_prob, dtype=float)
    threshold = 0.5
    pred = (p >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    recall = sensitivity
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(len(y), 1)
    balanced_accuracy = 0.5 * (sensitivity + specificity)
    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
    }


def compute_logloss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-12) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def compute_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    return float(np.mean((y - p) ** 2))


def compute_nagelkerke(y_true: np.ndarray, y_prob: np.ndarray, null_mean: float) -> float | None:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-12, 1.0 - 1e-12)
    if y.size == 0 or null_mean <= 0.0 or null_mean >= 1.0:
        return None
    ll_null = float(np.sum(y * math.log(null_mean) + (1.0 - y) * math.log(1.0 - null_mean)))
    ll_model = float(np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    n = int(y.size)
    r2_cs = 1.0 - math.exp(min(700.0, (2.0 / n) * (ll_null - ll_model)))
    max_r2_cs = 1.0 - math.exp(min(700.0, (2.0 / n) * ll_null))
    if max_r2_cs <= 0.0 or (not np.isfinite(r2_cs)):
        return None
    return float(r2_cs / max_r2_cs)


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 20) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        lo = bins[i]
        hi = bins[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not np.any(mask):
            continue
        total += float(np.mean(mask)) * abs(float(np.mean(y[mask])) - float(np.mean(p[mask])))
    return float(total)


def _survival_score_grid(train_times: np.ndarray) -> np.ndarray:
    vals = np.asarray(train_times, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0.0)]
    if vals.size == 0:
        return np.array([0.0, 1.0], dtype=float)
    median_followup = float(np.median(vals))
    grid = np.unique(
        np.asarray([0.0, *ROUTINE_SURVIVAL_HORIZONS, median_followup], dtype=float)
    )
    grid = grid[np.isfinite(grid) & (grid >= 0.0)]
    grid[0] = 0.0
    if grid.size == 1:
        grid = np.array([0.0, max(float(grid[0]), 1.0)], dtype=float)
    return grid


def _repeat_survival_curve(curve: np.ndarray, n_rows: int) -> np.ndarray:
    base = np.asarray(curve, dtype=float).reshape(1, -1)
    return np.repeat(base, n_rows, axis=0)


def _lifelines_concordance(event_times: np.ndarray, risk_score: np.ndarray, events: np.ndarray) -> float:
    try:
        lifelines_utils: Any = importlib.import_module("lifelines.utils")
        concordance_index = lifelines_utils.concordance_index
    except ModuleNotFoundError as exc:
        raise RuntimeError("lifelines is required for survival scoring") from exc
    try:
        return float(
            concordance_index(
                event_times,
                -np.asarray(risk_score, dtype=float),
                event_observed=events,
            )
        )
    except ZeroDivisionError:
        return 0.5


def _survival_null_curve(train_times: np.ndarray, train_events: np.ndarray, grid: np.ndarray) -> np.ndarray:
    try:
        lifelines: Any = importlib.import_module("lifelines")
        KaplanMeierFitter = lifelines.KaplanMeierFitter
    except ModuleNotFoundError as exc:
        raise RuntimeError("lifelines is required for survival scoring") from exc
    kmf = KaplanMeierFitter()
    kmf.fit(train_times, event_observed=train_events)
    surv = kmf.predict(grid).to_numpy(dtype=float)
    surv[0] = 1.0
    surv = np.clip(surv, 1e-12, 1.0)
    return np.minimum.accumulate(surv)


def calibrated_survival_matrix(
    train_times: np.ndarray,
    train_events: np.ndarray,
    train_risk: np.ndarray,
    test_risk: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    try:
        import pandas as pd
        lifelines: Any = importlib.import_module("lifelines")
        CoxPHFitter = lifelines.CoxPHFitter
        KaplanMeierFitter = lifelines.KaplanMeierFitter
    except ModuleNotFoundError as exc:
        raise RuntimeError("pandas and lifelines are required for survival scoring") from exc
    tr_times = np.asarray(train_times, dtype=float)
    tr_events = (np.asarray(train_events, dtype=float) > 0.5).astype(float)
    tr_risk = np.asarray(train_risk, dtype=float)
    te_risk = np.asarray(test_risk, dtype=float)
    finite_mask = np.isfinite(tr_times) & np.isfinite(tr_events) & np.isfinite(tr_risk) & (tr_times > 0.0)
    if int(np.sum(finite_mask)) == 0:
        return _repeat_survival_curve(np.ones_like(grid, dtype=float), te_risk.shape[0])
    tr_times = tr_times[finite_mask]
    tr_events = tr_events[finite_mask]
    tr_risk = tr_risk[finite_mask]
    if tr_risk.size < 2 or float(np.nanstd(tr_risk)) < 1e-12:
        kmf = KaplanMeierFitter()
        kmf.fit(tr_times, event_observed=tr_events)
        surv = kmf.predict(grid).to_numpy(dtype=float)
        surv[0] = 1.0
        surv = np.clip(surv, 1e-12, 1.0)
        return _repeat_survival_curve(np.minimum.accumulate(surv), te_risk.shape[0])
    calib_train = pd.DataFrame({"__time": tr_times, "__event": tr_events, "__risk": tr_risk})
    calib_test = pd.DataFrame({"__risk": te_risk})
    cph = CoxPHFitter(penalizer=1e-8)
    cph.fit(calib_train, duration_col="__time", event_col="__event")
    surv_df = cph.predict_survival_function(calib_test, times=grid)
    surv = surv_df.to_numpy(dtype=float).T
    surv[:, 0] = 1.0
    surv = np.clip(surv, 1e-12, 1.0)
    return np.minimum.accumulate(surv, axis=1)


def survival_lifted_metrics(
    event_times: np.ndarray,
    events: np.ndarray,
    grid: np.ndarray,
    survival_matrix: np.ndarray,
    null_survival_matrix: np.ndarray,
) -> dict[str, float | None]:
    times = np.asarray(event_times, dtype=float).reshape(-1)
    obs = (np.asarray(events, dtype=float).reshape(-1) > 0.5)
    surv = np.asarray(survival_matrix, dtype=float)
    null_surv = np.asarray(null_survival_matrix, dtype=float)
    if surv.ndim != 2 or surv.shape[0] != times.shape[0]:
        return {"brier": None, "logloss": None, "lifted_brier": None, "lifted_logloss": None, "nagelkerke_r2": None}
    dt = np.diff(grid)
    if grid.shape[0] < 2 or not np.all(dt > 0.0):
        return {"brier": None, "logloss": None, "lifted_brier": None, "lifted_logloss": None, "nagelkerke_r2": None}
    surv = np.clip(surv, 1e-12, 1.0)
    null_surv = np.clip(null_surv, 1e-12, 1.0)
    surv[:, 0] = 1.0
    null_surv[:, 0] = 1.0
    surv = np.minimum.accumulate(surv, axis=1)
    null_surv = np.minimum.accumulate(null_surv, axis=1)
    cumhaz = -np.log(surv)
    null_cumhaz = -np.log(null_surv)
    haz = np.maximum(np.diff(cumhaz, axis=1) / dt.reshape(1, -1), 0.0)
    null_haz = np.maximum(np.diff(null_cumhaz, axis=1) / dt.reshape(1, -1), 0.0)
    haz_sq_prefix = np.concatenate(
        [np.zeros((surv.shape[0], 1), dtype=float), np.cumsum((haz ** 2) * dt.reshape(1, -1), axis=1)],
        axis=1,
    )
    null_sq_prefix = np.concatenate(
        [np.zeros((null_surv.shape[0], 1), dtype=float), np.cumsum((null_haz ** 2) * dt.reshape(1, -1), axis=1)],
        axis=1,
    )
    brier_losses = np.empty(times.shape[0], dtype=float)
    log_losses = np.empty(times.shape[0], dtype=float)
    null_log_losses = np.empty(times.shape[0], dtype=float)
    null_brier_losses = np.empty(times.shape[0], dtype=float)
    for i, z in enumerate(times):
        j = int(np.searchsorted(grid, z, side="left"))
        if j >= grid.shape[0]:
            j = grid.shape[0] - 1
        if abs(grid[j] - z) <= 1e-12:
            idx = max(j - 1, 0)
            hz = haz[i, idx]
            hcum = cumhaz[i, j]
            h2_int = haz_sq_prefix[i, j]
            hz_null = null_haz[i, idx]
            hcum_null = null_cumhaz[i, j]
            h2_null = null_sq_prefix[i, j]
        else:
            idx = max(j - 1, 0)
            elapsed = z - grid[idx]
            hz = haz[i, idx]
            hcum = cumhaz[i, idx] + hz * elapsed
            h2_int = haz_sq_prefix[i, idx] + (hz ** 2) * elapsed
            hz_null = null_haz[i, idx]
            hcum_null = null_cumhaz[i, idx] + hz_null * elapsed
            h2_null = null_sq_prefix[i, idx] + (hz_null ** 2) * elapsed
        log_losses[i] = float(hcum - (math.log(max(hz, 1e-12)) if obs[i] else 0.0))
        null_log_losses[i] = float(hcum_null - (math.log(max(hz_null, 1e-12)) if obs[i] else 0.0))
        brier_losses[i] = float(0.5 * h2_int - (hz if obs[i] else 0.0))
        null_brier_losses[i] = float(0.5 * h2_null - (hz_null if obs[i] else 0.0))
    brier = float(np.mean(brier_losses))
    logloss = float(np.mean(log_losses))
    null_brier = float(np.mean(null_brier_losses))
    null_logloss = float(np.mean(null_log_losses))
    ll_model = float(-np.sum(log_losses))
    ll_null = float(-np.sum(null_log_losses))
    n = int(times.shape[0])
    r2_cs = 1.0 - math.exp(min(700.0, (2.0 / n) * (ll_null - ll_model)))
    max_r2_cs = 1.0 - math.exp(min(700.0, (2.0 / n) * ll_null))
    nag = float(r2_cs / max_r2_cs) if max_r2_cs > 0.0 and np.isfinite(r2_cs) else None
    return {
        "brier": brier,
        "logloss": logloss,
        "lifted_brier": float((null_brier - brier) / max(abs(null_brier), 1e-12)),
        "lifted_logloss": float((null_logloss - logloss) / max(abs(null_logloss), 1e-12)),
        "nagelkerke_r2": nag,
    }


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, train_prev: float) -> dict[str, float | None]:
    metrics = {
        "auc": compute_auc(y_true, y_prob),
        "pr_auc": compute_pr_auc(y_true, y_prob),
        "brier": compute_brier(y_true, y_prob),
        "logloss": compute_logloss(y_true, y_prob),
        "nagelkerke_r2": compute_nagelkerke(y_true, y_prob, train_prev),
        "ece": ece_score(y_true, y_prob),
        "mean_pred": float(np.mean(y_prob)),
        "mean_obs": float(np.mean(y_true)),
    }
    metrics.update(classification_confusion_metrics(y_true, y_prob))
    return metrics


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
    horizon = float(np.median(train_times))
    horizon_idx = min(int(np.searchsorted(grid, horizon, side="left")), grid.shape[0] - 1)
    horizon_surv = surv[:, horizon_idx]
    y_horizon = ((test_events > 0.5) & (test_times <= horizon)).astype(float)
    return {
        "c_index": _lifelines_concordance(test_times, test_risk, test_events),
        "auc": _lifelines_concordance(test_times, test_risk, test_events),
        "brier": proper["brier"],
        "logloss": proper["logloss"],
        "lifted_brier": proper["lifted_brier"],
        "lifted_logloss": proper["lifted_logloss"],
        "nagelkerke_r2": proper["nagelkerke_r2"],
        "horizon_years": horizon,
        "horizon_auc": compute_auc(y_horizon, 1.0 - horizon_surv),
        "horizon_brier": compute_brier(y_horizon, 1.0 - horizon_surv),
        "event_rate": float(np.mean(test_events)),
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
    horizon_surv = surv[:, horizon_idx]
    native_failure = 1.0 - horizon_surv
    y_horizon = ((test_events > 0.5) & (test_times <= horizon)).astype(float)
    return {
        "c_index": _lifelines_concordance(test_times, native_failure, test_events),
        "auc": _lifelines_concordance(test_times, native_failure, test_events),
        "brier": proper["brier"],
        "logloss": proper["logloss"],
        "lifted_brier": proper["lifted_brier"],
        "lifted_logloss": proper["lifted_logloss"],
        "nagelkerke_r2": proper["nagelkerke_r2"],
        "horizon_years": horizon,
        "horizon_auc": compute_auc(y_horizon, native_failure),
        "horizon_brier": compute_brier(y_horizon, native_failure),
        "event_rate": float(np.mean(test_events)),
    }


def _survival_probability_column(rows: list[dict[str, str]], *, method_name: str) -> np.ndarray:
    if not rows:
        raise RuntimeError(f"{method_name} survival prediction output is empty")
    if "survival_prob" in rows[0]:
        key = "survival_prob"
    elif "mean" in rows[0]:
        key = "mean"
    else:
        raise RuntimeError(
            f"{method_name} survival prediction output missing 'survival_prob' "
            f"or 'mean' column; got columns {sorted(rows[0].keys())}"
        )
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
            print(
                f"[HEARTBEAT-WARN] elapsed={elapsed:.1f}s exceeded 80% of cmd_timeout={timeout:.0f}s",
                file=sys.stderr,
                flush=True,
            )
            warned_80pct = True
        print(
            f"[HEARTBEAT] elapsed={elapsed:8.1f}s cmd='{cmd_preview}' pid={proc.pid} "
            f"cpu={snap.get('cpu_pct', 'n/a')}% mem={snap.get('mem_pct', 'n/a')}% "
            f"rss={fmt_kib(snap.get('rss_kib'))} vsz={fmt_kib(snap.get('vsz_kib'))}",
            file=sys.stderr,
            flush=True,
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
        try:
            while True:
                chunk = pipe.read(4096)
                if not chunk:
                    break
                text = chunk.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
                sink.write(text)
                sink.flush()
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
        print(msg, file=sys.stderr, flush=True)
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
_BFGS_SUMMARY_PATTERN = re.compile(
    r"\[OUTER summary\]\s+BFGS\s+(converged|hit max_iter|line-search failed|failed)(?:\s+in\s+(\d+)\s+iters)?\s+elapsed=([\d.]+)s"
)
_GUARD_PATTERN = re.compile(
    r"\[OUTER guard\]\s+convergence-guard re-eval at converged ρ done.*?elapsed=([\d.]+)s"
)
_SCHEDULE_TRANSITION_PATTERN = re.compile(
    r"\[OUTER schedule\]\s+inner-PIRLS cap transition.*?prev=(\d+)\s+new=(\d+)"
)

# Richer schedule-transition pattern that captures the quality signals
# alongside the prev/new cap. Each transition fires the cap-margin
# policy in `first_order_inner_cap_schedule` (commits 06888a1e,
# 04b30163, 58cccdcc); aggregating the quality signals across all
# transitions tells us how often each policy branch fired:
#
#   * accept_rho<0.5 fraction → how often did the "poor LM model
#     fidelity" branch bump the margin?
#   * ift_residual≥0.10 fraction → how often did the "poor IFT
#     prediction" branch enlarge margin to +4?
#   * not-converged fraction → how often did the geometric-backoff
#     branch fire (last solve hit the cap)?
#
# These are independent of WHETHER the transition happened (always
# logged) — a fit with stable cap won't generate transitions but the
# quality-signal distribution within the transitions that DID fire is
# the actionable signal. Numeric fields use the full `\S+` regex
# (instead of strict numeric) so the `n/a` case for missing signals
# is captured as a literal string, distinguishable from `0.0`.
_SCHEDULE_QUALITY_PATTERN = re.compile(
    r"\[OUTER schedule\]\s+inner-PIRLS cap transition.*?"
    r"last_iters=(\d+)\s+converged=(true|false)\s+"
    r"ift_residual=(\S+)\s+accept_rho=(\S+)\s+"
    r"prev=(\d+)\s+new=(\d+)"
)
_BIOBANK_GATE_PATTERN = re.compile(
    r"\[(?:transformation-normal|gamlss-location-scale|latent-survival|latent-binary|gamlss-binomial-mean-wiggle|survival-marginal-slope|survival-location-scale|bernoulli-marginal-slope)\]\s+declining analytic outer Hessian for n=(\d+)"
)
# Per-iter PIRLS timing from `[PIRLS iter-end] iter=N elapsed=Xs ...`
# (commit 90bccf8a, extended to include g_norm in this commit). Lets us
# histogram inner-Newton wall-time per iter at biobank scale — answers
# "is inner-PIRLS the dominant cost?" without needing a separate
# scaling probe.
_PIRLS_ITER_END_PATTERN = re.compile(
    r"\[PIRLS iter-end\]\s+iter=\s*(\d+)\s+elapsed=([\d.]+)s"
)

# Per-iter PIRLS LM-loop breakdown. Splits each inner-Newton iter's
# wall-time into the four sub-phases that drive cost:
#   curvature : model.update_with_curvature (gradient + Hessian assembly,
#               XᵀWX + per-row work, the bulk for FLEX margslope)
#   solve     : (H + λI)·δ = -g linear solve (LM-loop body; O(p³)/3
#               dense Cholesky or sparse SPD factor)
#   predred   : predicted-reduction quadratic form (O(p²) matvec)
#   candidate : model.update_candidate (gain-ratio test; for FLEX margslope
#               this is the per-row sextic-kernel intercept root-find)
# Summed over all iters in a fit, the dominant phase identifies which
# principled optimization to ship next: candidate-heavy → inner-row
# kernel; solve-heavy → matrix-free / iterative inner solve;
# curvature-heavy → assembly fusion or sparser representations.
_PIRLS_ITER_BREAKDOWN_PATTERN = re.compile(
    r"\[PIRLS iter-breakdown\]\s+iter=\s*(\d+)\s+attempts=(\d+)"
    r"\s+curvature=([\d.]+)s\s+solve=([\d.]+)s\s+predred=([\d.]+)s"
    r"\s+candidate=([\d.]+)s\s+other=([\d.]+)s"
)

# Per-iter curvature-kind log emitted by `runworking_model_pirls`'s
# `update_with_curvature` block. The `curvature=<KIND>` token is the
# Debug format of the `HessianCurvatureKind` enum: `Observed` or
# `Fisher`. Aggregating tells us how often the Fisher fallback fires,
# which is a direct signal of observed-Hessian PD failures at biobank
# scale (the Observed-curvature path is preferred for non-canonical
# links because it converges faster, but it's not guaranteed PD; on
# failure, the inner solver retries with Fisher).
_PIRLS_CURVATURE_KIND_PATTERN = re.compile(
    r"\[STAGE\] PIRLS update_with_curvature iter=\d+\s+curvature=(\w+)"
)

# Per-iter mid-LM-loop Fisher fallback markers. Distinct from the
# iter-start Observed→Fisher transition captured by
# `_PIRLS_CURVATURE_KIND_PATTERN`: these fire when the iter-start
# Observed assembly succeeded but mid-LM-loop the inner Newton
# triggered a Fisher retry. Two reasons:
#   gain_rejection: the candidate evaluation produced a bad gain
#                   ratio / non-finite gradient / extreme eta —
#                   Observed Hessian unreliable for this β region.
#   candidate_err : update_candidate itself returned Err (numerical
#                   breakdown). Stronger signal than gain_rejection.
# Aggregating both surfaces how much of the curvature work in a fit
# is going through the Fisher retry path.
_PIRLS_MID_ITER_FISHER_PATTERN = re.compile(
    r"\[PIRLS\] mid-iter Fisher fallback iter=(\d+)\s+reason=(\w+)"
)

# `force_fisher_for_rest` engagement marker. Fires once per PIRLS
# solve at most — when consecutive_fisher_fallbacks > 2 the inner
# solver gives up on Observed for the rest of the solve and locks
# into Fisher. The reason field captures WHICH branch's increment
# pushed the count over the threshold (iter_start vs gain_rejection
# vs candidate_err). A non-zero count of these markers across a fit
# means at least one PIRLS solve fully transitioned to Fisher;
# combined with the existing `pirls_fisher_frac` aggregator, this
# tells us whether the high Fisher fraction is one-shot fallbacks
# or a sustained Fisher-only state.
_PIRLS_FORCE_FISHER_PATTERN = re.compile(
    r"\[PIRLS\] force_fisher_for_rest engaged at iter=(\d+)\s+"
    r"\(consecutive_fisher_fallbacks=(\d+)\)\s+reason=(\w+)"
)

# Per-iter LM trajectory: validates the textbook Madsen accept (commit
# 58ae42d1) and reject (d37626e6) updates plus the runtime adaptive λ
# clamp (43be42be) are moving the trust region in useful directions at
# biobank scale. Aggregating log10_ratio across iters reveals the LM's
# global behavior:
#   median ratio < 0  → LM is shrinking λ (Newton-friendly trajectory)
#   median ratio ≈ 0  → LM is stationary (problem is at the radius)
#   median ratio > 0  → LM is expanding λ (rejection-heavy)
# Aggregating accept_rho reveals model fidelity:
#   median accept_rho ≈ 1  → quadratic model is faithful
#   median accept_rho << 1 → model over-states predicted reduction
_PIRLS_LM_TRAJECTORY_PATTERN = re.compile(
    r"\[PIRLS lm-trajectory\]\s+iter=\s*(\d+)\s+"
    r"start_lambda=([\d.eE+\-]+)\s+final_lambda=([\d.eE+\-]+)\s+"
    r"log10_ratio=([\d.eE+\-nNaA]+)\s+accept_rho=([\d.eE+\-nNaA]+)\s+"
    r"attempts=(\d+)"
)

# Outer-Hessian routing decision and assembly cost. Two markers per outer
# Hessian build:
#   [OUTER hessian-route]    : decision (operator vs dense) + which clause
#   [OUTER hessian-elapsed]  : wall-clock for the chosen path
#
# Surfaces priority item (d) — the routing edge case at k≥32 with a
# rank-deficient penalty (subspace_trace=true) forces the dense
# `compute_outer_hessian` path even though scale prefers operator. The
# `reason=subspace_forced_dense` label fires *only* when both:
#   (1) scale_prefers_operator (any of large_p / large_n_moderate_p /
#       large_linear_work / large_k), AND
#   (2) penalty_subspace_trace is installed (rank-deficient LAML fix).
# Aggregating the count + total elapsed for that label answers
# empirically: is the architectural matrix-free-projected outer Hessian
# (priority c) actually load-bearing at biobank scale, or are we
# spending tractable time on the dense path despite the routing scare?
_OUTER_HESSIAN_ROUTE_PATTERN = re.compile(
    r"\[OUTER hessian-route\]\s+choice=(\w+)\s+reason=(\w+)\s+"
    r"n=(\d+)\s+p=(\d+)\s+k=(\d+)\s+"
    r"callback_kernel=(true|false)\s+subspace_trace=(true|false)\s+"
    # `scale_prefers_operator` is `true|false` for the kernel-based path,
    # but `irrelevant` for the family-op early-return branch (whose
    # routing decision doesn't consult the (n,p,k) crossover at all).
    r"scale_prefers_operator=(true|false|irrelevant)"
)
_OUTER_HESSIAN_ELAPSED_PATTERN = re.compile(
    r"\[OUTER hessian-elapsed\]\s+choice=(\w+)\s+reason=(\w+)\s+"
    r"n=(\d+)\s+p=(\d+)\s+k=(\d+)\s+elapsed=([\d.]+)s"
)

# Solve-end summary: one line per completed PIRLS solve carrying the
# geometric convergence rate of the inner Newton:
#     rate = (g_norm_final / g_norm_initial) ^ (1 / iters)
# Aggregated by the runner so CI logs end with a per-fit answer to
# "is the inner Newton converging healthily, or is it stuck?"
# Healthy Newton: rate < 0.5. Stuck near a near-singular geometry or
# starting from a poor warm-start: rate ≥ 0.7.
_PIRLS_SOLVE_END_PATTERN = re.compile(
    r"\[PIRLS solve-end\]\s+iters=(\d+)\s+elapsed=([\d.]+)s\s+g_norm_initial=\S+\s+g_norm_final=\S+\s+convergence_rate=([\deE.+\-nNaA]+)\s+status=(\w+)"
)

# κ-optimization scaling instrumentation from commit cd89625f. The
# `optimize_spatial_length_scale_exact_joint` driver wraps each closure
# invocation (cost-only, value+grad(/Hessian), EFS) with a stopwatch and
# emits one `[KAPPA-PHASE]` line per call plus a `[KAPPA-PHASE-SUMMARY]`
# at exit. Parsing both gives us a production-fit κ-scaling probe
# (task #32) without a separate synthetic harness — the markers surface
# the actual workload split between cost-only line-search probes,
# full-eval iterations, and EFS calls during real biobank fits.
_KAPPA_PHASE_PATTERN = re.compile(
    r"\[KAPPA-PHASE\]\s+phase=(\w+)\s+call=(\d+)(?:\s+order=\S+)?\s+theta_norm=\S+\s+log_kappa_norm=\S+\s+elapsed_s=([\d.]+)"
)
_KAPPA_PHASE_SUMMARY_PATTERN = re.compile(
    r"\[KAPPA-PHASE-SUMMARY\]\s+log_kappa_dim=(\d+)\s+n_cost=(\d+)\s+cost_total_s=([\d.]+)\s+n_eval=(\d+)\s+eval_total_s=([\d.]+)\s+n_efs=(\d+)\s+efs_total_s=([\d.]+)\s+optim_total_s=([\d.]+)"
)

# IFT predictor quality probe. Emitted by `execute_pirls_if_needed` after
# every successful non-screening PIRLS solve when a warm-start prediction
# was actually used. residual = ‖β_converged − β_predicted‖ / ‖β_converged‖
# — close to 0 when the linearization was faithful, close to 1 when the
# prediction was no better than flat. The bench runner / analyzer pivots
# this so we can empirically validate whether the IFT predictor is
# actually paying off at biobank scale (which the inner-PIRLS scaling
# probe identified as the dominant cost regime).
_IFT_QUALITY_PATTERN = re.compile(
    r"\[IFT-QUALITY\]\s+residual=([\deE.+-]+)\s+converged_norm=([\deE.+-]+)\s+predicted_norm=([\deE.+-]+)(?:\s+drho_norm=([\deE.+\-nNaA]+)\s+h_pen_logdet=([\deE.+\-nNaA]+))?\s+iters=(\d+)"
)
# Tangent-line quality probe — same field layout as IFT-QUALITY.
# Emitted when the tangent-line predictor (rather than IFT) produced
# the β consumed by PIRLS. Tagged separately so the bench runner's
# residual percentile aggregation correctly attributes the predictor
# that fired, instead of misclassifying tangent-line predictions as
# IFT predictions in the aggregate.
_TANGENT_QUALITY_PATTERN = re.compile(
    r"\[TANGENT-QUALITY\]\s+residual=([\deE.+-]+)\s+converged_norm=([\deE.+-]+)\s+predicted_norm=([\deE.+-]+)(?:\s+drho_norm=([\deE.+\-nNaA]+)\s+h_pen_logdet=([\deE.+\-nNaA]+))?\s+iters=(\d+)"
)

# IFT predictor rejection counter. Emitted by `predict_warm_start_beta_ift_from_cache`
# whenever the predictor falls through (large Δρ, factorization failure,
# non-finite output, basis dim mismatch). Combined with the accept count
# from [IFT-QUALITY] this gives the production accept/reject ratio at
# biobank scale — the empirical answer to "is the warm-start machinery
# being USED at the magnitudes the outer optimizer takes?"
_IFT_REJECTED_PATTERN = re.compile(r"\[IFT-REJECTED\]\s+reason=(\w+)")

# IFT factor-cache hit/miss instrumentation. Every predict call either
# reuses the cached H_pen factor (hit) or pays a fresh Cholesky (miss).
# At biobank scale the dense Cholesky is O(p³)/3, multiple seconds at
# p ≈ several thousand, so a high hit rate validates that the factor
# cache (commit ec18559d) is paying off. The miss line additionally
# reports the elapsed factorization wall-clock so we can size the
# avoided cost. Aggregating gives:
#   ift_cache_hits / ift_cache_total → hit rate
#   sum(ift_cache_miss_elapsed)      → cumulative Cholesky cost paid
_IFT_CACHE_HIT_PATTERN = re.compile(
    r"\[IFT-CACHE\]\s+outcome=hit\s+drho_dim=(\d+)(?:\s+p=(\d+))?"
)
_IFT_CACHE_MISS_PATTERN = re.compile(
    r"\[IFT-CACHE\]\s+outcome=miss\s+drho_dim=(\d+)(?:\s+p=(\d+))?\s+elapsed=([\d.]+)s"
)

# Outer eval wall-clock per order kind. Aggregating tells us how the
# outer optimizer's time distributes across:
#   ValueAndGradient    — BFGS-style eval (most common at biobank scale
#                         where the gates route to BFGS)
#   ValueGradientHessian — ARC eval (used at smaller scale or when
#                         analytic outer Hessian is available)
#   ValueOnly           — line-search probes (typically cheaper)
# The verdict line surfaces count + total per order so a CI reviewer
# can see where the outer optimizer spent time vs the per-stage
# breakdowns (`pirls_total`, `outer_h_total`). The gap between
# `outer_eval_total` and (`pirls_total` + `outer_h_total`) is the
# OTHER work — score computation, gradient assembly, IFT predict, etc.
_OUTER_EVAL_END_PATTERN = re.compile(
    r"\[STAGE\] outer eval end order=(\w+) elapsed=([\d.]+)s"
)

# Seed-screening cascade summary log. Emitted once per outer-fit
# call by `rank_seeds_with_screening` after iterating the cap-tier
# cascade (commit 40c20d5b). Captures the final cascade state:
#   elapsed       : wall-clock for the whole cascade across all stages
#   stages_used   : how many cap tiers had to be tried (1 = first-tier
#                   pass, ≥2 = some seeds rejected at lower caps and
#                   the cascade had to escalate)
#   final_cap     : the cap used at the last successful stage
#                   (`uncapped` literal or numeric)
#   ranked / seeds: surviving seeds / total seeds presented
#
# Aggregating tells us whether seed-screening is paying off (high
# stages_used = the cap-tier cascade is doing real work) or whether
# the heuristic seeds are usually fine (stages_used=1). At biobank
# scale this can dominate startup cost if the seeds are routinely
# rejected at the tightest caps.
_SEED_CASCADE_PATTERN = re.compile(
    r"\[OUTER\][^\n]*seed screening cascade complete\s+"
    r"elapsed=([\d.]+)s\s+stages_used=(\d+)\s+"
    r"final_cap=(\w+)\s+ranked=(\d+)/(\d+)"
)

# IFT predictor identity / no-op counter. Emitted when all Δρ_k are
# below the numerical-noise floor — the outer made an effectively-zero
# ρ-step and the predictor returned the cached β unchanged. The
# [IFT-QUALITY] residual for these calls is exactly zero, which would
# inflate the apparent accept count if not separated. Distinguishing
# noop from accept tells us how often the outer is actually exercising
# the linearization.
_IFT_NOOP_PATTERN = re.compile(r"\[IFT-NOOP\]\s+reason=(\w+)")

# Tangent-line predictor markers. The tangent-line path only fires
# when the IFT predictor returned None for non-cache reasons (large
# Δρ, factor failed, etc.), so [TANGENT-*] markers tell us how often
# the FALLBACK path is being exercised — and whether it succeeds or
# also degenerates to flat warm-start. A high tangent-reject rate
# alongside high IFT-reject rate signals "linear predictor stack
# failed entirely → both predictors fell through to flat".
_TANGENT_PREDICT_PATTERN = re.compile(
    r"\[TANGENT-PREDICT\]\s+alpha=([\deE.+\-]+)\s+cap=([\deE.+\-]+)\s+drho_step_norm_sq=([\deE.+\-]+)\s+drho_prev_norm_sq=([\deE.+\-]+)"
)
_TANGENT_REJECTED_PATTERN = re.compile(r"\[TANGENT-REJECTED\]\s+reason=(\w+)")
# Tangent-line identity / no-op counter — symmetric to _IFT_NOOP_PATTERN.
# Emitted when α is below the numerical-noise floor; predictor
# returned the cached β unchanged. Tagged separately from rejects
# (which are bug-signal failure modes) and predicts (which moved β
# meaningfully) so the bench runner can distinguish a genuinely
# inactive tangent-line path from a degenerate one.
_TANGENT_NOOP_PATTERN = re.compile(r"\[TANGENT-NOOP\]\s+reason=(\w+)")

# `[OUTER non-finite]` warnings from the REML unified evaluator. Each
# line records a NaN / Inf in an intermediate of the outer-Hessian /
# leverage / adjoint computation. In a healthy fit the count is zero;
# any non-zero count is a bug signal pointing to penalty drift, cross-
# trace overflow, or similar numerical instability that would degrade
# the fit. Emitted only as `log::warn!` so stderr captures it; the
# field-name capture lets the runner show WHICH intermediate was
# affected without having to dump the full warning lines.
_OUTER_NONFINITE_PATTERN = re.compile(r"\[OUTER non-finite\]\s+(\S+)")


_PHASE_START_PATTERN = re.compile(r"\[PHASE\]\s+([\w\-]+(?:\([\w\-/]+\))?)\s+(?:fit\s+)?start")


def _emit_phase_summary(
    captured_stderr: str,
    cmd_preview: str,
    *,
    timed_out: bool = False,
    rc: int = 0,
) -> None:
    end_matches = _PHASE_END_PATTERN.findall(captured_stderr)
    start_phases = _PHASE_START_PATTERN.findall(captured_stderr)
    completed = {name for name, _ in end_matches}
    pending = [p for p in start_phases if p not in completed]
    by_phase: dict[str, float] = {}
    for name, secs in end_matches:
        by_phase[name] = by_phase.get(name, 0.0) + float(secs)
    total = sum(by_phase.values())
    parts = [f"{name}={secs:.1f}s" for name, secs in by_phase.items()]
    # Aggregate BFGS run summaries (per inner phase that uses BFGS)
    bfgs_runs = _BFGS_SUMMARY_PATTERN.findall(captured_stderr)
    if bfgs_runs:
        n = len(bfgs_runs)
        bfgs_total = sum(float(secs) for _, _, secs in bfgs_runs)
        # Per-status counts: distinguishes "BFGS optimizer converged"
        # from "hit max_iter cap" (legitimate failure mode at biobank
        # scale where the cmd timeout limits iter count) from
        # "line-search failed" (numerical pathology) from generic
        # "failed". The mix tells us at a glance whether the BFGS
        # tolerance is well-calibrated or whether a particular failure
        # mode is dominating.
        status_counts: dict[str, int] = {}
        for status, _iters, _secs in bfgs_runs:
            status_counts[status] = status_counts.get(status, 0) + 1
        converged = status_counts.get("converged", 0)
        # BFGS iter-count distribution: how many outer iters did each
        # successful run take? Iters is captured only when the regex's
        # optional `in N iters` group matched (i.e., status != "failed"
        # and the runtime emitted the iter count). p50 + max of this
        # distribution complement the convergence-rate count: p50=2
        # means most runs converge fast (warm-start working); p50≥10
        # means each run grinds through many iters and the warm-start
        # is degrading.
        bfgs_iters_list: list[int] = []
        for _status, iters_str, _secs in bfgs_runs:
            if iters_str:
                try:
                    bfgs_iters_list.append(int(iters_str))
                except ValueError:
                    pass
        iter_pieces = ""
        if bfgs_iters_list:
            sorted_iters = sorted(bfgs_iters_list)
            n_i = len(sorted_iters)
            p50 = sorted_iters[n_i // 2]
            pmax = sorted_iters[-1]
            iter_pieces = f" bfgs_iters_p50={p50} bfgs_iters_max={pmax}"
        # Per-status breakdown (sorted for stable ordering).
        status_pieces = " ".join(
            f"bfgs_{status.replace(' ', '_').replace('-', '_')}={count}"
            for status, count in sorted(status_counts.items())
        )
        parts.append(
            f"bfgs_runs={n}({converged}_conv) bfgs_total={bfgs_total:.1f}s "
            f"{status_pieces}{iter_pieces}"
        )
    # Aggregate convergence-guard refits
    guard_runs = _GUARD_PATTERN.findall(captured_stderr)
    if guard_runs:
        guard_total = sum(float(secs) for secs in guard_runs)
        parts.append(f"guard_refits={len(guard_runs)} guard_total={guard_total:.1f}s")
    # Schedule transition count (path #3 firing)
    schedule_transitions = _SCHEDULE_TRANSITION_PATTERN.findall(captured_stderr)
    if schedule_transitions:
        parts.append(f"sched_transitions={len(schedule_transitions)}")
    # Schedule-transition quality-signal distribution. For each
    # transition, the snapshot captured at decision time is logged with
    # the quality signals that drove the margin computation. Aggregating
    # tells us how often each policy branch fired: poor LM model
    # fidelity (`accept_rho < 0.5`), poor IFT prediction (`ift_residual
    # >= 0.10`), or geometric backoff (`converged=false`). The fractions
    # let a CI reviewer see at a glance whether the schedule's adaptive
    # branches are doing useful work or sitting at the default.
    sched_quality = _SCHEDULE_QUALITY_PATTERN.findall(captured_stderr)
    if sched_quality:
        n_q = len(sched_quality)
        n_unconv = sum(1 for q in sched_quality if q[1] == "false")
        n_poor_ift = 0
        n_poor_rho = 0
        # Count of cap-hit transitions where accept_rho was *very* poor
        # (`< 0.3`) — the trigger for the ×3 escalation policy
        # (commit 96e043aa). This is a STRICT subset of `sched_unconv`
        # AND a strict subset of `sched_poor_accept_rho`. Tracking it
        # separately tells a CI reviewer how often the most aggressive
        # backoff branch fired (useful for diagnosing whether the
        # geometry is truly hard, or whether the policy is firing
        # spuriously and bloating the cap unnecessarily).
        n_x3_escalation = 0
        for _last_iters, conv, ift_str, rho_str, _prev, _new in sched_quality:
            if ift_str != "n/a":
                try:
                    if float(ift_str) >= 0.10:
                        n_poor_ift += 1
                except ValueError:
                    pass
            if rho_str != "n/a":
                try:
                    rho_val = float(rho_str)
                    if rho_val < 0.5:
                        n_poor_rho += 1
                    if conv == "false" and rho_val < 0.3:
                        n_x3_escalation += 1
                except ValueError:
                    pass
        parts.append(
            f"sched_quality_n={n_q} "
            f"sched_unconv={n_unconv} "
            f"sched_poor_ift={n_poor_ift} "
            f"sched_poor_accept_rho={n_poor_rho} "
            f"sched_x3_escalation={n_x3_escalation}"
        )
    # Biobank-scale gate firings (path #2)
    gate_firings = _BIOBANK_GATE_PATTERN.findall(captured_stderr)
    if gate_firings:
        parts.append(f"biobank_gates_fired={len(gate_firings)}")
    # Per-iter PIRLS wall-time histogram. Surfaces whether inner-Newton
    # is the dominant biobank cost (current bandaid hypothesis behind
    # path #3 schedule) by reporting count + total + p50/p95/max.
    pirls_iter_secs = [
        float(secs) for _iter, secs in _PIRLS_ITER_END_PATTERN.findall(captured_stderr)
    ]
    if pirls_iter_secs:
        n = len(pirls_iter_secs)
        sorted_secs = sorted(pirls_iter_secs)
        total_pirls = sum(pirls_iter_secs)
        p50 = sorted_secs[n // 2]
        p95 = sorted_secs[min(n - 1, int(0.95 * n))]
        pmax = sorted_secs[-1]
        parts.append(
            f"pirls_iters={n} pirls_total={total_pirls:.1f}s "
            f"pirls_p50={p50:.3f}s pirls_p95={p95:.3f}s pirls_max={pmax:.3f}s"
        )
    # Per-iter LM-loop sub-phase aggregation. Surfaces the dominant
    # inner-Newton hot spot in one verdict line so the next principled
    # optimization knows where to land. The percentages are computed
    # against (curv + solve + predred + candidate); "other" is reported
    # absolutely so a reviewer can see whether bookkeeping/KKT-cert
    # overhead is non-trivial. Total LM attempt count surfaces LM-halving
    # pressure: attempts ≫ iters indicates the trust region is fighting
    # the geometry, often a sign that warm-starting is degrading.
    # Outer-Hessian routing distribution + assembly cost. The verdict
    # answers two questions: (1) is the operator path actually being
    # selected at biobank scale, and (2) when we fall through to dense,
    # how much wall-clock are we paying — specifically distinguishing
    # the `subspace_forced_dense` case (item d) from the cheap
    # `below_crossover` case where dense is fine.
    outer_h_route = _OUTER_HESSIAN_ROUTE_PATTERN.findall(captured_stderr)
    outer_h_elapsed = _OUTER_HESSIAN_ELAPSED_PATTERN.findall(captured_stderr)
    # If route fired but elapsed didn't, the assembly errored or the
    # process was killed mid-build — surfacing the gap is worth a tag.
    outer_h_route_no_elapsed = max(0, len(outer_h_route) - len(outer_h_elapsed))
    if outer_h_elapsed:
        n_outer_h = len(outer_h_elapsed)
        total_outer_h = sum(float(m[5]) for m in outer_h_elapsed)
        choice_counts: dict[str, int] = {}
        reason_counts: dict[str, int] = {}
        reason_secs: dict[str, float] = {}
        for choice, reason, _n, _p, _k, secs in outer_h_elapsed:
            choice_counts[choice] = choice_counts.get(choice, 0) + 1
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            reason_secs[reason] = reason_secs.get(reason, 0.0) + float(secs)
        # The single dominant reason gets a callout in the summary line so
        # the verdict is readable at a glance; if the run hit the routing
        # edge case at all (subspace_forced_dense > 0), surface it
        # explicitly with its time share so the (d) signal isn't
        # collapsed into a generic counter.
        choice_pieces = " ".join(f"{c}={n}" for c, n in sorted(choice_counts.items()))
        subspace_forced = reason_counts.get("subspace_forced_dense", 0)
        subspace_secs = reason_secs.get("subspace_forced_dense", 0.0)
        family_op_calls = reason_counts.get("family_op", 0)
        family_op_secs = reason_secs.get("family_op", 0.0)
        # Dominant reason by elapsed time: tells a reviewer at a glance
        # which routing path consumed the most outer-Hessian wall-clock.
        # When family_op dominates, CTN/survival/GAMLSS exact-Hv path is
        # the bottleneck. When subspace_forced_dense dominates, priority
        # (c) is load-bearing. When below_crossover dominates, the dense
        # path is fine and matrix-free isn't worth the lift.
        dom_reason_by_secs = max(reason_secs, key=lambda r: reason_secs[r])
        parts.append(
            f"outer_h_calls={n_outer_h} outer_h_total={total_outer_h:.1f}s "
            f"outer_h_{choice_pieces} "
            f"outer_h_dom_reason={dom_reason_by_secs}@{reason_secs[dom_reason_by_secs]:.1f}s "
            f"outer_h_subspace_forced={subspace_forced} "
            f"outer_h_subspace_total={subspace_secs:.1f}s "
            f"outer_h_family_op={family_op_calls} "
            f"outer_h_family_op_total={family_op_secs:.1f}s "
            f"outer_h_route_no_elapsed={outer_h_route_no_elapsed}"
        )
    elif outer_h_route:
        # Route fired but no assembly completed (errored or killed).
        # Still surface the count + dominant reason so we don't lose
        # signal that the routing ran.
        reason_counts_only: dict[str, int] = {}
        for _choice, reason, *_rest in outer_h_route:
            reason_counts_only[reason] = reason_counts_only.get(reason, 0) + 1
        dom_reason = max(reason_counts_only, key=lambda r: reason_counts_only[r])
        parts.append(
            f"outer_h_INCOMPLETE outer_h_routes={len(outer_h_route)} "
            f"outer_h_dom_reason={dom_reason}"
        )
    # Outer-eval wall-clock distribution per order kind. Counts how
    # often each evaluation order fired (BFGS = ValueAndGradient at
    # biobank scale; ARC = ValueGradientHessian at smaller scale; line-
    # search probes = ValueOnly) and totals the wall-clock per order.
    # The gap between `outer_eval_total` and (`pirls_total` +
    # `outer_h_total`) is the OTHER work (score computation, IFT
    # predict, gradient assembly, etc.); when significant, that gap
    # itself is a hint about where to optimize next.
    # Seed-screening cascade summary aggregation. Emit a verdict line
    # capturing total cascade wall-clock, total stages exercised, and
    # the cumulative ranked/seeds ratio. Useful for diagnosing whether
    # the seed-screening cap-tier cascade is doing real work at
    # biobank scale or sitting at the first tier.
    seed_cascades = _SEED_CASCADE_PATTERN.findall(captured_stderr)
    if seed_cascades:
        n_cascades = len(seed_cascades)
        cascade_elapsed_total = sum(float(m[0]) for m in seed_cascades)
        stages_total = sum(int(m[1]) for m in seed_cascades)
        ranked_total = sum(int(m[3]) for m in seed_cascades)
        seeds_total = sum(int(m[4]) for m in seed_cascades)
        # Count how many cascades had to escalate beyond the first tier
        # (`stages_used >= 2`). A high count indicates the heuristic
        # seeds are routinely rejected at the tightest caps and the
        # cascade has to escalate — extra startup cost at biobank scale.
        cascades_escalated = sum(1 for m in seed_cascades if int(m[1]) >= 2)
        rank_rate = ranked_total / max(seeds_total, 1)
        parts.append(
            f"seed_cascade_n={n_cascades} "
            f"seed_cascade_elapsed={cascade_elapsed_total:.1f}s "
            f"seed_cascade_escalated={cascades_escalated} "
            f"seed_cascade_stages_total={stages_total} "
            f"seed_cascade_rank_rate={rank_rate:.2f}"
        )
    outer_eval_ends = _OUTER_EVAL_END_PATTERN.findall(captured_stderr)
    if outer_eval_ends:
        per_order_counts: dict[str, int] = {}
        per_order_secs: dict[str, float] = {}
        for order, secs_str in outer_eval_ends:
            per_order_counts[order] = per_order_counts.get(order, 0) + 1
            per_order_secs[order] = per_order_secs.get(order, 0.0) + float(secs_str)
        n_evals_total = len(outer_eval_ends)
        secs_total = sum(per_order_secs.values())
        # Sort by name for stable output ordering.
        order_pieces = " ".join(
            f"outer_eval_{order}={per_order_counts[order]}@{per_order_secs[order]:.1f}s"
            for order in sorted(per_order_counts.keys())
        )
        parts.append(
            f"outer_eval_n={n_evals_total} outer_eval_total={secs_total:.1f}s "
            f"{order_pieces}"
        )
    # Per-iter curvature-kind aggregation. Counts `Observed` and
    # `Fisher` invocations of `update_with_curvature` across the run.
    # The Observed path is preferred for non-canonical links because
    # it converges faster, but it can fail (non-PD) — when it fails,
    # the inner solver falls back to Fisher. A high Fisher fraction
    # at biobank scale is a direct signal of observed-Hessian PD
    # failures, which is actionable diagnostic info.
    curvature_kinds = _PIRLS_CURVATURE_KIND_PATTERN.findall(captured_stderr)
    if curvature_kinds:
        kind_counts: dict[str, int] = {}
        for kind in curvature_kinds:
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
        n_curv_total = len(curvature_kinds)
        # Stable ordering for the verdict line.
        kind_pieces = " ".join(
            f"pirls_curv_{kind}={count}"
            for kind, count in sorted(kind_counts.items())
        )
        # Fisher fraction = Fisher / (Observed + Fisher). High value
        # (e.g., >0.2) indicates the observed Hessian is frequently
        # non-PD; the curvature-kind contract documents that this
        # forces a fallback that's slower-converging per iter.
        fisher_count = kind_counts.get("Fisher", 0)
        fisher_frac = fisher_count / max(n_curv_total, 1)
        parts.append(
            f"pirls_curv_n={n_curv_total} {kind_pieces} "
            f"pirls_fisher_frac={fisher_frac:.2f}"
        )
    # Mid-LM-loop Fisher-fallback events, distinct from iter-start.
    # When non-zero, indicates the Observed curvature was reliable at
    # iter-start but unreliable mid-LM-loop — a more subtle pathology
    # than wholesale Observed-failure (which iter-start captures).
    mid_iter_fisher = _PIRLS_MID_ITER_FISHER_PATTERN.findall(captured_stderr)
    if mid_iter_fisher:
        gain_rejection_count = sum(1 for m in mid_iter_fisher if m[1] == "gain_rejection")
        candidate_err_count = sum(1 for m in mid_iter_fisher if m[1] == "candidate_err")
        parts.append(
            f"pirls_mid_iter_fisher_n={len(mid_iter_fisher)} "
            f"pirls_mid_iter_gain_rejection={gain_rejection_count} "
            f"pirls_mid_iter_candidate_err={candidate_err_count}"
        )
    # `force_fisher_for_rest` engagement: fires at most once per PIRLS
    # solve, when the inner solver locks into Fisher for the rest of
    # the solve. A non-zero count means at least one solve fully
    # transitioned to Fisher-only — different signal from
    # `pirls_fisher_frac` (which counts iter-start Fisher uses,
    # whether or not the lock-in fired).
    force_fisher = _PIRLS_FORCE_FISHER_PATTERN.findall(captured_stderr)
    if force_fisher:
        # Per-reason counts: which branch's increment pushed the
        # count over the threshold. iter_start = Observed assembly
        # itself failed too many times; gain_rejection / candidate_err
        # = mid-LM-loop Observed retries failed too many times.
        force_reasons: dict[str, int] = {}
        for _iter, _count, reason in force_fisher:
            force_reasons[reason] = force_reasons.get(reason, 0) + 1
        reason_pieces = " ".join(
            f"pirls_force_fisher_{reason}={count}"
            for reason, count in sorted(force_reasons.items())
        )
        parts.append(
            f"pirls_force_fisher_n={len(force_fisher)} {reason_pieces}"
        )
    pirls_breakdown_matches = _PIRLS_ITER_BREAKDOWN_PATTERN.findall(captured_stderr)
    if pirls_breakdown_matches:
        curv_total = sum(float(m[2]) for m in pirls_breakdown_matches)
        solve_total = sum(float(m[3]) for m in pirls_breakdown_matches)
        predred_total = sum(float(m[4]) for m in pirls_breakdown_matches)
        cand_total = sum(float(m[5]) for m in pirls_breakdown_matches)
        other_total = sum(float(m[6]) for m in pirls_breakdown_matches)
        attempts_total = sum(int(m[1]) for m in pirls_breakdown_matches)
        timed_sum = curv_total + solve_total + predred_total + cand_total
        if timed_sum > 0.0:
            phase_pct = {
                "curv": curv_total / timed_sum,
                "solve": solve_total / timed_sum,
                "predred": predred_total / timed_sum,
                "cand": cand_total / timed_sum,
            }
            dom_phase = max(phase_pct, key=lambda k: phase_pct[k])
            parts.append(
                f"pirls_attempts={attempts_total} "
                f"pirls_dom={dom_phase}@{phase_pct[dom_phase] * 100:.0f}% "
                f"pirls_curv={curv_total:.1f}s pirls_solve={solve_total:.1f}s "
                f"pirls_predred={predred_total:.1f}s pirls_cand={cand_total:.1f}s "
                f"pirls_other={other_total:.1f}s"
            )
    # Per-iter LM trajectory: median + p95 of (log10 of final/start λ)
    # and accept_rho across iters. This is the validation signal for
    # the textbook LM updates (Madsen accept commit 58ae42d1, Madsen
    # reject d37626e6, adaptive runtime clamp 43be42be):
    #
    #   * lm_log10_ratio_p50 < 0  → LM is shrinking λ on accepted iters
    #     (Newton-friendly), as expected when the geometry is healthy
    #   * lm_log10_ratio_p50 > 0  → LM is expanding λ on accepted iters
    #     (geometry is fighting; the Madsen accept side's marginal-
    #     accept ×1.125-2.0 expansion is firing, which is correct
    #     behavior for hard problems but indicates extra work)
    #   * lm_accept_rho_p50 ≈ 1   → quadratic model faithful
    #   * lm_accept_rho_p50 << 1  → model over-states predicted
    #     reduction; smaller steps should be tried
    lm_traj = _PIRLS_LM_TRAJECTORY_PATTERN.findall(captured_stderr)
    if lm_traj:
        ratios: list[float] = []
        rhos: list[float] = []
        # Per-iter LM attempt count. p50 and p95 of this distribution
        # tell a CI reviewer whether the LM loop is mostly clean
        # (p50=1: most iters accept on first attempt) or struggle-y
        # (p50≥3: most iters need multiple LM halvings). The textbook
        # Madsen ×2 rejection trajectory (commit d37626e6) gives more
        # shots near the trust radius, so we expect p95 to be modest
        # even for hard problems — a high p95 indicates either a
        # genuinely-hard surface or a regression.
        attempts: list[int] = []
        for _it, _start, _final, ratio_str, rho_str, att_str in lm_traj:
            try:
                r = float(ratio_str)
                if r == r:  # filter NaN
                    ratios.append(r)
            except ValueError:
                pass
            try:
                rho = float(rho_str)
                if rho == rho:
                    rhos.append(rho)
            except ValueError:
                pass
            try:
                attempts.append(int(att_str))
            except ValueError:
                pass
        ratio_pieces = []
        if ratios:
            ratios_sorted = sorted(ratios)
            n_r = len(ratios_sorted)
            ratio_pieces.append(
                f"lm_log10_ratio_p50={ratios_sorted[n_r // 2]:.2f}"
            )
            ratio_pieces.append(
                f"lm_log10_ratio_p95={ratios_sorted[min(n_r - 1, int(0.95 * n_r))]:.2f}"
            )
        rho_pieces = []
        if rhos:
            rhos_sorted = sorted(rhos)
            n_p = len(rhos_sorted)
            rho_pieces.append(f"lm_accept_rho_p50={rhos_sorted[n_p // 2]:.2f}")
            rho_pieces.append(
                f"lm_accept_rho_p05={rhos_sorted[max(0, int(0.05 * n_p))]:.2f}"
            )
        attempt_pieces = []
        if attempts:
            attempts_sorted = sorted(attempts)
            n_a = len(attempts_sorted)
            attempt_pieces.append(f"lm_attempts_p50={attempts_sorted[n_a // 2]}")
            attempt_pieces.append(
                f"lm_attempts_p95={attempts_sorted[min(n_a - 1, int(0.95 * n_a))]}"
            )
            attempt_pieces.append(f"lm_attempts_max={attempts_sorted[-1]}")
        if ratio_pieces or rho_pieces or attempt_pieces:
            parts.append(
                f"lm_iters={len(lm_traj)} "
                + " ".join(ratio_pieces + rho_pieces + attempt_pieces)
            )
    # Per-solve geometric convergence rate. A healthy biobank fit ends
    # with most PIRLS solves at rate < 0.5; a struggling fit shows
    # consistent rate ≥ 0.7 across solves (inner Newton stuck near
    # singular geometry, or flat warm-start that the predictor failed
    # to refine). Surface count + p50 + max so the verdict is a glance.
    pirls_solve_matches = _PIRLS_SOLVE_END_PATTERN.findall(captured_stderr)
    if pirls_solve_matches:
        rates: list[float] = []
        # Per-status counts: distinguishes the 5 PirlsStatus variants
        # (Converged, MaxIterationsReached, StalledAtValidMinimum,
        # LmStepSearchExhausted, Unstable). At biobank scale we
        # expect mostly Converged + StalledAtValidMinimum (legitimate
        # outcomes). MaxIterationsReached or LmStepSearchExhausted in
        # large quantities indicates the inner cap is too tight or the
        # geometry is genuinely hard.
        status_counts: dict[str, int] = {}
        # Per-solve iter count distribution. Different from
        # `pirls_iters` (which counts iter-end markers — i.e., total
        # iters across all solves). p50 / p95 of per-solve iters
        # tells us "how long does a typical solve take" vs the total
        # iter budget.
        per_solve_iters: list[int] = []
        for iters_str, _elapsed, rate_str, status in pirls_solve_matches:
            status_counts[status] = status_counts.get(status, 0) + 1
            try:
                per_solve_iters.append(int(iters_str))
            except ValueError:
                pass
            try:
                r = float(rate_str)
            except (ValueError, TypeError):
                continue
            if r == r:  # filter NaN
                rates.append(r)
        if rates:
            n = len(rates)
            sorted_r = sorted(rates)
            p50 = sorted_r[n // 2]
            p95 = sorted_r[min(n - 1, int(0.95 * n))]
            rmax = sorted_r[-1]
            # Stable status breakdown (sorted alphabetically).
            status_pieces = " ".join(
                f"pirls_status_{status}={count}"
                for status, count in sorted(status_counts.items())
            )
            iter_pieces = ""
            if per_solve_iters:
                sorted_iters = sorted(per_solve_iters)
                ni = len(sorted_iters)
                iter_pieces = (
                    f" pirls_solve_iters_p50={sorted_iters[ni // 2]}"
                    f" pirls_solve_iters_p95={sorted_iters[min(ni - 1, int(0.95 * ni))]}"
                    f" pirls_solve_iters_max={sorted_iters[-1]}"
                )
            parts.append(
                f"pirls_solves={n} pirls_conv_p50={p50:.3f} "
                f"pirls_conv_p95={p95:.3f} pirls_conv_max={rmax:.3f} "
                f"{status_pieces}{iter_pieces}"
            )
    # κ-optimization driver wall-time. Per-call markers feed the
    # distribution; summary lines feed the totals. Multiple κ
    # optimizations may run within a single command (e.g. CTN
    # bootstrap then refit) so we accumulate across all summary lines.
    kappa_calls = _KAPPA_PHASE_PATTERN.findall(captured_stderr)
    kappa_summaries = _KAPPA_PHASE_SUMMARY_PATTERN.findall(captured_stderr)
    if kappa_summaries:
        n_summaries = len(kappa_summaries)
        n_cost_total = sum(int(s[1]) for s in kappa_summaries)
        cost_s_total = sum(float(s[2]) for s in kappa_summaries)
        n_eval_total = sum(int(s[3]) for s in kappa_summaries)
        eval_s_total = sum(float(s[4]) for s in kappa_summaries)
        n_efs_total = sum(int(s[5]) for s in kappa_summaries)
        efs_s_total = sum(float(s[6]) for s in kappa_summaries)
        optim_s_total = sum(float(s[7]) for s in kappa_summaries)
        parts.append(
            f"kappa_optims={n_summaries} kappa_optim_total={optim_s_total:.1f}s "
            f"kappa_cost_calls={n_cost_total} kappa_cost_total={cost_s_total:.1f}s "
            f"kappa_eval_calls={n_eval_total} kappa_eval_total={eval_s_total:.1f}s "
            f"kappa_efs_calls={n_efs_total} kappa_efs_total={efs_s_total:.1f}s"
        )
        # Per-call distribution disambiguates "single outlier" from
        # "uniformly-slow workload" — same rationale as the
        # kappa_optim_INCOMPLETE branch (commit 361e47b5). The summary
        # line's totals are authoritative; the per-call max/p95 are
        # best-effort distribution data computed from [KAPPA-PHASE]
        # lines that survived stderr capture. When stderr buffer rolled
        # over and dropped some per-call lines (rare), max/p95 may be
        # under-reported but never over-reported.
        if kappa_calls:
            phase_secs: dict[str, list[float]] = {}
            for phase_name, _call_idx, secs in kappa_calls:
                phase_secs.setdefault(phase_name, []).append(float(secs))
            dist_pieces = []
            for phase_name in sorted(phase_secs.keys()):
                secs_list = phase_secs[phase_name]
                n_p = len(secs_list)
                sorted_p = sorted(secs_list)
                phase_max = sorted_p[-1]
                phase_p95 = sorted_p[min(n_p - 1, int(0.95 * n_p))]
                dist_pieces.append(
                    f"kappa_{phase_name}_p95={phase_p95:.2f}s "
                    f"kappa_{phase_name}_max={phase_max:.2f}s"
                )
            if dist_pieces:
                parts.append(" ".join(dist_pieces))
    elif kappa_calls:
        # Got per-call markers but no summary (κ optimization didn't
        # finish — e.g. interrupted by command timeout). Surface the
        # partial count + per-phase totals AND per-call distribution
        # (max + p95) from the per-call lines, so a reviewer can tell
        # which phase had a slow call vs which was just called many
        # times. Mission-relevant: a single eval_outer that took most
        # of the budget is a different signal from many fast eval_outer
        # calls accumulating; the totals collapse those into one number
        # but the max/p95 disambiguate.
        per_phase_secs: dict[str, list[float]] = {}
        for phase_name, _call_idx, secs in kappa_calls:
            per_phase_secs.setdefault(phase_name, []).append(float(secs))
        kphase_pieces = []
        for phase_name in sorted(per_phase_secs.keys()):
            secs_list = per_phase_secs[phase_name]
            n_p = len(secs_list)
            sorted_p = sorted(secs_list)
            phase_max = sorted_p[-1]
            phase_p95 = sorted_p[min(n_p - 1, int(0.95 * n_p))]
            kphase_pieces.append(
                f"kappa_{phase_name}_calls={n_p} "
                f"kappa_{phase_name}_total={sum(secs_list):.1f}s "
                f"kappa_{phase_name}_p95={phase_p95:.2f}s "
                f"kappa_{phase_name}_max={phase_max:.2f}s"
            )
        parts.append(f"kappa_optim_INCOMPLETE {' '.join(kphase_pieces)}")
    # IFT-quality probe distribution. Each marker records the residual
    # ‖β_converged − β_predicted‖ / ‖β_converged‖ for one accepted
    # PIRLS solve where a warm-start prediction was used. We surface
    # the count, p50, p95, and max so the bench logs end with a
    # one-line answer to "is the IFT predictor faithful at biobank
    # scale?". A small p50 (~1e-3 or below) means the linearization
    # is doing its job; a large p50 (~1) means the prediction is
    # collapsing to flat warm-start and we should investigate.
    ift_quality_matches = _IFT_QUALITY_PATTERN.findall(captured_stderr)
    ift_rejected_matches = _IFT_REJECTED_PATTERN.findall(captured_stderr)
    ift_noop_matches = _IFT_NOOP_PATTERN.findall(captured_stderr)
    n_quality = len(ift_quality_matches)
    n_noops = len(ift_noop_matches)
    n_rejects = len(ift_rejected_matches)
    # IFT factor-cache hit/miss verdict. The H_pen Cholesky is multiple
    # seconds at biobank scale (p ≈ several thousand → O(p³)/3 flops),
    # so a high hit rate validates commit ec18559d's cache. A miss
    # rate near 1.0 indicates the cache is being invalidated too
    # aggressively — typically by warm-start clears between outer
    # iters — and represents avoidable wall-clock at biobank scale.
    ift_cache_hits = _IFT_CACHE_HIT_PATTERN.findall(captured_stderr)
    ift_cache_misses = _IFT_CACHE_MISS_PATTERN.findall(captured_stderr)
    n_cache_total = len(ift_cache_hits) + len(ift_cache_misses)
    if n_cache_total > 0:
        hit_rate = len(ift_cache_hits) / n_cache_total
        # Tuple layout: (drho_dim, p_optional, elapsed). `p` is the
        # optional group from the regex (older logs without `p=N` still
        # match with p as empty string); we don't need it here, but we
        # use it below for the per-fit p signature.
        miss_secs_list = [float(m[2]) for m in ift_cache_misses]
        miss_secs_total = sum(miss_secs_list)
        # Per-miss Cholesky cost distribution. A single slow Cholesky
        # (one outer iter on a wide / ill-conditioned H) might dominate
        # `miss_secs_total`; p50 / max distinguishes that case from a
        # uniformly-slow miss workload, where every cache miss pays a
        # substantial Cholesky. The two regimes have different fixes:
        # uniform slowness needs faster factorization (or fewer
        # invalidations); single-spike slowness needs investigation of
        # the offending iter's Hessian.
        miss_secs_p50 = 0.0
        miss_secs_max = 0.0
        if miss_secs_list:
            miss_sorted = sorted(miss_secs_list)
            miss_secs_p50 = miss_sorted[len(miss_sorted) // 2]
            miss_secs_max = miss_sorted[-1]
        # Factor-paid-then-rejected count: predictions that passed the
        # wrapper's early short-circuit (noop and large_drho gates from
        # commits 8395848d and 28bfa0e1) but then failed deeper in the
        # inner predictor (rho/beta dim mismatch, non_finite_rhs,
        # non_finite_predicted, or post-factor `factorize_failed`).
        # `ift_cache_n - n_accepts` gives this directly: every factor
        # lookup either accepted (counted in `[IFT-QUALITY]` →
        # n_quality) or rejected after the lookup. A non-zero value
        # here represents avoidable Cholesky cost that the current
        # early short-circuits don't cover; large values would justify
        # extending the short-circuit further.
        factor_paid_rejects = max(0, n_cache_total - len(ift_quality_matches))
        # Matrix dim observed across cache misses. With `p=N` added to
        # the marker, we can surface the matrix size driving the
        # Cholesky cost; this lets a CI reviewer correlate
        # `ift_cache_miss_secs` with the Cholesky's `O(p³)/3` regime.
        # Older logs without the `p=N` field land in p_observed=[]; in
        # that case we skip the size field for backwards compatibility.
        miss_ps: list[int] = []
        for m in ift_cache_misses:
            if m[1]:
                try:
                    miss_ps.append(int(m[1]))
                except ValueError:
                    pass
        size_piece = ""
        if miss_ps:
            size_piece = f" ift_cache_miss_max_p={max(miss_ps)}"
        parts.append(
            f"ift_cache_n={n_cache_total} "
            f"ift_cache_hit_rate={hit_rate:.2f} "
            f"ift_cache_miss_secs={miss_secs_total:.2f} "
            f"ift_cache_miss_p50={miss_secs_p50:.2f}s "
            f"ift_cache_miss_max={miss_secs_max:.2f}s "
            f"ift_cache_paid_rejects={factor_paid_rejects}{size_piece}"
        )
    # As of the runtime change accompanying this commit, [IFT-QUALITY]
    # is suppressed on noop calls (predictor returned β_cur unchanged
    # because all Δρ were below the numerical floor). So every
    # [IFT-QUALITY] line corresponds to a "real" predict call whose
    # residual reflects the linearization's actual faithfulness. The
    # n_quality count is the real accept count directly — no
    # subtraction needed.
    n_accepts = n_quality
    # Tangent-line quality probe — separate distribution from IFT.
    # When tangent-line fires (only as IFT's fallback), its residuals
    # report a different predictor's faithfulness, so mixing them into
    # the IFT distribution would skew the percentiles. Aggregate
    # independently.
    tangent_quality_matches = _TANGENT_QUALITY_PATTERN.findall(captured_stderr)
    if tangent_quality_matches:
        t_residuals = [
            float(m[0]) for m in tangent_quality_matches if float(m[0]) == float(m[0])
        ]
        if t_residuals:
            n = len(t_residuals)
            sorted_t = sorted(t_residuals)
            t_p50 = sorted_t[n // 2]
            t_p95 = sorted_t[min(n - 1, int(0.95 * n))]
            t_max = sorted_t[-1]
            parts.append(
                f"tangent_quality_predicts={n} tangent_p50={t_p50:.2e} "
                f"tangent_p95={t_p95:.2e} tangent_max={t_max:.2e}"
            )
        # PIRLS-iters distribution after tangent-line predictions —
        # parallel to the IFT path's `ift_iters_*` aggregation
        # (commit 69011666). When a tangent-line prediction is poor,
        # the inner Newton has to recover from a worse starting
        # point, so iters tends to be higher; surfacing this lets a
        # reviewer see whether tangent-line is contributing useful
        # warm-start work or just a slightly-better-than-flat seed
        # that PIRLS still has to grind through.
        t_iters = [
            int(m[5])
            for m in tangent_quality_matches
            if m[5]
        ]
        if t_iters:
            n_i = len(t_iters)
            sorted_i = sorted(t_iters)
            i_p50 = sorted_i[n_i // 2]
            i_p95 = sorted_i[min(n_i - 1, int(0.95 * n_i))]
            i_max = sorted_i[-1]
            parts.append(
                f"tangent_iters_p50={i_p50} tangent_iters_p95={i_p95} "
                f"tangent_iters_max={i_max}"
            )
    if ift_quality_matches:
        residuals = [float(m[0]) for m in ift_quality_matches if float(m[0]) == float(m[0])]
        if residuals:
            n = len(residuals)
            sorted_res = sorted(residuals)
            p50 = sorted_res[n // 2]
            p95 = sorted_res[min(n - 1, int(0.95 * n))]
            rmax = sorted_res[-1]
            parts.append(
                f"ift_predicts={n} ift_p50={p50:.2e} ift_p95={p95:.2e} ift_max={rmax:.2e}"
            )
        # Inner-iter distribution per accepted IFT predict. The
        # captured `iters` field is PIRLS's iteration count from
        # `pirls_result.iteration` — i.e. how many Newton iters the
        # inner solver took after starting from the IFT-predicted β.
        # Combined with `ift_p50_resid` this tells the full warm-start
        # value story:
        #   small p50_resid + small ift_iters_p50 → predictor accurate AND
        #     PIRLS converged fast (warm-start delivering correctness +
        #     speed; mission-aligned biobank trace).
        #   small p50_resid + large ift_iters_p50 → predictor accurate
        #     but PIRLS still struggled (hard geometry; warm-start
        #     correct but speed-bound elsewhere).
        #   large p50_resid + large ift_iters_p50 → predictor poor and
        #     PIRLS slow (warm-start collapsing toward flat).
        ift_iters = [
            int(m[5])
            for m in ift_quality_matches
            if m[5]
        ]
        if ift_iters:
            n_i = len(ift_iters)
            sorted_i = sorted(ift_iters)
            i_p50 = sorted_i[n_i // 2]
            i_p95 = sorted_i[min(n_i - 1, int(0.95 * n_i))]
            i_max = sorted_i[-1]
            parts.append(
                f"ift_iters_p50={i_p50} ift_iters_p95={i_p95} ift_iters_max={i_max}"
            )
        # Δρ-step distribution: how big a ρ-jump is the predictor
        # being asked to handle? Combined with residual quartiles
        # this answers "did large residuals come from large jumps
        # (expected) or small jumps (predictor faithfulness regression)?"
        drhos = [
            float(m[3])
            for m in ift_quality_matches
            if m[3] and float(m[3]) == float(m[3])  # filter NaN / missing
        ]
        if drhos:
            n_d = len(drhos)
            sorted_d = sorted(drhos)
            d_p50 = sorted_d[n_d // 2]
            d_p95 = sorted_d[min(n_d - 1, int(0.95 * n_d))]
            d_max = sorted_d[-1]
            parts.append(
                f"ift_drho_p50={d_p50:.2e} ift_drho_p95={d_p95:.2e} ift_drho_max={d_max:.2e}"
            )
        # log|H_pen| spread: tracks penalized-Hessian conditioning
        # across solves. Sudden jumps in min/max indicate a flat
        # direction opening up or a near-singular geometry — both
        # often precede PIRLS failures or large IFT residuals.
        logdets = [
            float(m[4])
            for m in ift_quality_matches
            if m[4] and float(m[4]) == float(m[4])
        ]
        if logdets:
            l_min = min(logdets)
            l_max = max(logdets)
            parts.append(
                f"ift_h_logdet_min={l_min:.2e} ift_h_logdet_max={l_max:.2e}"
            )
    # `[OUTER non-finite]` warnings: NaN / Inf in the REML unified
    # evaluator's intermediate computations. Should be 0 in a healthy
    # biobank fit; any non-zero count is a real bug signal worth
    # surfacing prominently (penalty drift, cross-trace overflow,
    # adjoint/leverage instability). Group by the affected
    # intermediate's name so the source is debuggable without
    # spelunking through stderr.
    outer_nonfinite = _OUTER_NONFINITE_PATTERN.findall(captured_stderr)
    if outer_nonfinite:
        intermediate_counts: dict[str, int] = {}
        for name in outer_nonfinite:
            intermediate_counts[name] = intermediate_counts.get(name, 0) + 1
        intermediates_str = ",".join(
            f"{n}={c}" for n, c in sorted(intermediate_counts.items())
        )
        parts.append(
            f"outer_nonfinite={len(outer_nonfinite)} "
            f"outer_nonfinite_at=[{intermediates_str}]"
        )
    # Tangent-line predictor activity (the fallback path when IFT
    # rejects). Surface accept count + rejection-reason histogram so
    # a degenerate "both predictors fell through to flat" run is
    # visible at a glance.
    tangent_predict_hits = _TANGENT_PREDICT_PATTERN.findall(captured_stderr)
    tangent_rejected = _TANGENT_REJECTED_PATTERN.findall(captured_stderr)
    tangent_noops = _TANGENT_NOOP_PATTERN.findall(captured_stderr)
    if tangent_predict_hits or tangent_rejected or tangent_noops:
        t_predicts = len(tangent_predict_hits)
        t_rejects = len(tangent_rejected)
        # Surface alpha distribution from accepted tangent-line
        # predictions: tells us whether the fallback fires at modest
        # extrapolations (α ≈ 1, healthy) or pushes the adaptive cap
        # (α near alpha_cap, marginal). Combined with the residual
        # distribution from [TANGENT-QUALITY] this tells us how
        # aggressive the fallback is at biobank Δρ scales.
        alphas = [
            float(m[0])
            for m in tangent_predict_hits
            if float(m[0]) == float(m[0])
        ]
        if alphas:
            n_a = len(alphas)
            sorted_a = sorted(alphas)
            a_p50 = sorted_a[n_a // 2]
            a_max = sorted_a[-1]
            tangent_alpha_str = f" tangent_alpha_p50={a_p50:.2f} tangent_alpha_max={a_max:.2f}"
        else:
            tangent_alpha_str = ""
        if tangent_rejected:
            t_reason_counts: dict[str, int] = {}
            for r in tangent_rejected:
                t_reason_counts[r] = t_reason_counts.get(r, 0) + 1
            t_reasons_str = ",".join(
                f"{r}={c}" for r, c in sorted(t_reason_counts.items())
            )
            parts.append(
                f"tangent_predicts={t_predicts} tangent_rejects={t_rejects} "
                f"tangent_reasons=[{t_reasons_str}]{tangent_alpha_str}"
            )
        else:
            parts.append(f"tangent_predicts={t_predicts}{tangent_alpha_str}")
        if tangent_noops:
            # Symmetric to ift_noops: tangent-line returned identity
            # because α was below the numerical-noise floor. Surfaces
            # the rate at biobank scale; non-zero count here means
            # the IFT path rejected (tangent-line wouldn't have fired
            # otherwise) AND the resulting Δρ landed in a regime
            # where the tangent step is numerically negligible.
            parts.append(f"tangent_noops={len(tangent_noops)}")
        # Tangent accept-rate metrics, symmetric to the IFT metrics
        # (commit 962210f3). Two complementary denominators:
        #
        #   tangent_accept_rate         = predicts / (predicts +
        #                                  rejects + noops)
        #   tangent_accept_rate_active  = predicts / (predicts +
        #                                  rejects)  ← noops EXCLUDED
        #
        # Same rationale as the IFT split: noops in the denominator
        # conflate predictor effectiveness with outer-optimizer
        # behavior (zero-step calls). Surfacing both lets a CI
        # reviewer separate "tangent predictor is bad" from "outer
        # is calling tangent unnecessarily".
        n_t_noops = len(tangent_noops) if tangent_noops else 0
        denom_total = max(t_predicts + t_rejects + n_t_noops, 1)
        denom_active = max(t_predicts + t_rejects, 1)
        if t_predicts > 0 or t_rejects > 0 or n_t_noops > 0:
            parts.append(
                f"tangent_accept_rate={t_predicts / denom_total:.2f} "
                f"tangent_accept_rate_active={t_predicts / denom_active:.2f}"
            )
    # Consistency cross-check: every successful [TANGENT-PREDICT]
    # produces a downstream [TANGENT-QUALITY] from the post-PIRLS
    # residual computation in execute_pirls_if_needed (commit 99424b47).
    # If counts diverge, instrumentation drift is silently dropping
    # markers — a regression signal worth surfacing as a separate
    # field rather than burying inside the existing aggregations.
    if tangent_predict_hits and tangent_quality_matches is not None:
        n_predicts = len(tangent_predict_hits)
        n_quality = len(tangent_quality_matches)
        # Allow off-by-one in case the run was truncated mid-fit
        # (PIRLS still running when the command timed out — TANGENT-PREDICT
        # was emitted but TANGENT-QUALITY hadn't yet).
        if abs(n_predicts - n_quality) > 1:
            parts.append(
                f"tangent_marker_drift=predict={n_predicts}_vs_quality={n_quality}"
            )
    if n_rejects > 0 or n_noops > 0:
        # Count distinct rejection reasons so the bench log shows which
        # failure mode dominates: large_drho is the expected biobank
        # case (predictor's adaptive cap firing on outer steps that
        # genuinely outrun the linearization); the others
        # (hessian_factorize_failed, non_finite_*, qs_dim_mismatch) are
        # bug signals if non-zero.
        reason_counts: dict[str, int] = {}
        for reason in ift_rejected_matches:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        reasons_str = ",".join(
            f"{reason}={count}" for reason, count in sorted(reason_counts.items())
        )
        # Two complementary accept-rate metrics:
        #
        # `ift_accept_rate` — accepts / total predict calls (incl. noops).
        # This is the historical metric. It conflates predictor
        # effectiveness with outer-optimizer behavior: a fit where the
        # outer makes many zero-step calls (e.g. during convergence
        # checks) shows lower accept_rate even when the predictor
        # itself is working perfectly on every "real" call.
        #
        # `ift_accept_rate_active` — accepts / (accepts + rejects),
        # EXCLUDING noops. This is the predictor-quality-only signal:
        # of the predict calls where the linearization actually had
        # work to do (max|Δρ| > eps), how many succeeded? A value
        # near 1.0 means "the predictor is reliable on every real
        # step"; a low value means "the outer is taking steps the
        # adaptive cap rejects" (typically large Δρ near a transition
        # in the cost surface, or poor IFT residual driving cap to
        # tighten).
        #
        # Surfacing both lets a CI reviewer separate these two
        # signals: a fit with high noops + low accept_rate but high
        # accept_rate_active has a healthy predictor that the outer
        # is just calling unnecessarily; a fit with low both is a
        # genuine predictor problem.
        denom_total = max(n_accepts + n_rejects + n_noops, 1)
        accept_rate = n_accepts / denom_total
        denom_active = max(n_accepts + n_rejects, 1)
        accept_rate_active = n_accepts / denom_active
        parts.append(
            f"ift_rejects={n_rejects} ift_noops={n_noops} "
            f"ift_accept_rate={accept_rate:.2f} "
            f"ift_accept_rate_active={accept_rate_active:.2f} "
            f"ift_reasons=[{reasons_str}]"
        )
    suffix = ""
    if pending:
        suffix = f" pending={','.join(pending)}"
    if timed_out or rc == 124:
        suffix += " [TIMEOUT]"
    if not parts and not suffix:
        return
    print(
        f"[PHASE summary] cmd='{cmd_preview}' total={total:.1f}s {' '.join(parts)}{suffix}",
        file=sys.stderr,
        flush=True,
    )
    # Single-line warm-start health verdict — aggregates the multiple
    # signals above into HEALTHY / MARGINAL / DEGRADED so CI reviewers
    # don't have to mentally combine ift_predicts / ift_p50 /
    # ift_accept_rate on every run. The verdict combines two
    # axes — predictor coverage (how often does it accept?) and
    # predictor faithfulness (how close to KKT is the prediction?) —
    # via the policy in `_warm_start_health_verdict`. Emitted only when
    # we have IFT-QUALITY data; absent otherwise so a fit that
    # legitimately doesn't exercise the warm-start path doesn't get
    # tagged.
    warm_start_verdict: str | None = None
    pirls_verdict: str | None = None
    if ift_quality_matches or outer_nonfinite or tangent_quality_matches:
        residuals_for_verdict = [
            float(m[0]) for m in ift_quality_matches if float(m[0]) == float(m[0])
        ]
        # Tangent-line residuals are tracked separately (commit 99424b47)
        # but also surface in the verdict's detail string so a reviewer
        # sees BOTH predictor distributions in one glance. The verdict
        # tier itself stays IFT-driven (the IFT predictor is the primary
        # path; tangent-line is only a fallback), so this is purely
        # informational.
        tangent_resids = [
            float(m[0])
            for m in tangent_quality_matches
            if float(m[0]) == float(m[0])
        ]
        tangent_p50 = (
            sorted(tangent_resids)[len(tangent_resids) // 2] if tangent_resids else None
        )
        verdict, detail = _warm_start_health_verdict(
            n_accepts=n_accepts,
            n_rejects=n_rejects,
            n_noops=n_noops,
            residuals=residuals_for_verdict,
            n_outer_nonfinite=len(outer_nonfinite),
            n_tangent_accepts=len(tangent_resids),
            tangent_p50=tangent_p50,
        )
        warm_start_verdict = verdict
        print(
            f"[WARM-START health] cmd='{cmd_preview}' verdict={verdict} {detail}",
            file=sys.stderr,
            flush=True,
        )
    # PIRLS health verdict — separate from warm-start because the
    # inner Newton's convergence behavior is independent of predictor
    # quality (PIRLS could be slow even with a perfect warm-start
    # when the geometry is hard, or fast even with a flat warm-start
    # when the geometry is benign). Aggregated from [PIRLS solve-end]
    # markers (commit 517f1b02). Tier policy in `_pirls_health_verdict`.
    if pirls_solve_matches:
        pirls_rates: list[float] = []
        # The regex captures four groups (iters, elapsed, rate, status)
        # since commit bb42aeb0; this aggregator only reads `rate_str`.
        for _iters, _elapsed, rate_str, _status in pirls_solve_matches:
            try:
                r = float(rate_str)
            except (ValueError, TypeError):
                continue
            if r == r:
                pirls_rates.append(r)
        if pirls_rates:
            verdict, pirls_detail = _pirls_health_verdict(rates=pirls_rates)
            pirls_verdict = verdict
            print(
                f"[PIRLS health] cmd='{cmd_preview}' verdict={verdict} {pirls_detail}",
                file=sys.stderr,
                flush=True,
            )
    # Curvature health verdict, third axis added in this commit.
    # Combines the Fisher-fallback aggregators into a single tier
    # capturing observed-Hessian reliability across the fit. Computed
    # whenever the curvature-kind markers fired (i.e. PIRLS ran at
    # least once); else NO-DATA / None. We re-extract the counts
    # here rather than reusing the upstream `kind_counts`/`n_curv_total`
    # which are scoped to the aggregator's emit block above.
    curvature_verdict: str | None = None
    curvature_detail = ""
    if curvature_kinds:
        cv_kind_counts: dict[str, int] = {}
        for kind in curvature_kinds:
            cv_kind_counts[kind] = cv_kind_counts.get(kind, 0) + 1
        cv_total = len(curvature_kinds)
        cv_fisher_frac = cv_kind_counts.get("Fisher", 0) / max(cv_total, 1)
        cv_force_n = len(force_fisher) if force_fisher else 0
        curvature_verdict, curvature_detail = _curvature_health_verdict(
            fisher_frac=cv_fisher_frac,
            force_fisher_n=cv_force_n,
        )
        print(
            f"[CURVATURE health] cmd='{cmd_preview}' "
            f"verdict={curvature_verdict} {curvature_detail}",
            file=sys.stderr,
            flush=True,
        )
    # Top-level [FIT health] verdict combines warm-start, PIRLS, and
    # curvature health verdicts into a single glance-readable tier so
    # reviewers don't have to mentally combine the three. Combination
    # is "worst tier wins" with a documented total ordering —
    # DEGRADED > MARGINAL > HEALTHY > NO-DATA. Emitted only when at
    # least one sub-verdict was determined.
    if (
        warm_start_verdict is not None
        or pirls_verdict is not None
        or curvature_verdict is not None
    ):
        combined = _combine_fit_verdicts(
            warm_start_verdict, pirls_verdict, curvature_verdict
        )
        ws_label = warm_start_verdict if warm_start_verdict else "ABSENT"
        pirls_label = pirls_verdict if pirls_verdict else "ABSENT"
        cv_label = curvature_verdict if curvature_verdict else "ABSENT"
        # `dominant_axis` field surfaces the axis name that drove the
        # combined verdict — useful for CI scrapers that want to
        # alert on the specific failing axis without re-implementing
        # the worst-of-three computation. Tie-breaking: prefer pirls
        # over warm_start over curvature (lexically reverse-sorted),
        # matching the "PIRLS is the central inner-Newton diagnostic"
        # convention from the older 2-axis verdict. When all three
        # axes are NO-DATA / ABSENT, we report `dominant_axis=none`.
        dominant_axis = _dominant_axis_for_verdict(
            combined,
            warm_start=warm_start_verdict,
            pirls=pirls_verdict,
            curvature=curvature_verdict,
        )
        print(
            f"[FIT health] cmd='{cmd_preview}' verdict={combined} "
            f"dominant_axis={dominant_axis} "
            f"warm_start={ws_label} pirls={pirls_label} curvature={cv_label}",
            file=sys.stderr,
            flush=True,
        )


def _combine_fit_verdicts(
    warm_start: str | None,
    pirls: str | None,
    curvature: str | None = None,
) -> str:
    """Combine the warm-start, PIRLS, and curvature health verdicts
    into a single top-level fit verdict via a worst-wins total
    ordering:

      DEGRADED  > MARGINAL > HEALTHY > NO-DATA

    Any input may be `None` (sub-verdict was not emitted because
    its source markers were absent); a `None` is treated as if the
    sub-verdict were NO-DATA. The combined verdict reflects the
    WORST tier seen across the axes — a fit that's HEALTHY on
    one axis but DEGRADED on another is overall DEGRADED, not
    "averaged" to MARGINAL. The independent sub-verdicts remain
    visible in the [FIT health] line's `warm_start=` / `pirls=` /
    `curvature=` fields so reviewers can see which axis tripped
    the combined verdict.

    The `curvature` axis is the third health signal added in this
    commit: it captures observed-Hessian reliability via the
    Fisher-fallback aggregators. Defaults to None for backward-
    compatibility with callers that only have warm-start + PIRLS.
    """
    rank = {"DEGRADED": 3, "MARGINAL": 2, "HEALTHY": 1, "NO-DATA": 0}
    inv_rank = {v: k for k, v in rank.items()}
    ws_rank = rank.get(warm_start or "NO-DATA", 0)
    p_rank = rank.get(pirls or "NO-DATA", 0)
    c_rank = rank.get(curvature or "NO-DATA", 0)
    return inv_rank[max(ws_rank, p_rank, c_rank)]


def _dominant_axis_for_verdict(
    combined: str,
    *,
    warm_start: str | None,
    pirls: str | None,
    curvature: str | None,
) -> str:
    """Return the axis name (`warm_start` / `pirls` / `curvature` /
    `none`) that drove the combined verdict via worst-of-three.

    When `combined == "NO-DATA"`, all three axes were missing → `none`.

    Tie-breaking among axes at the same tier: prefer `pirls` first (the
    central inner-Newton diagnostic, matching the older 2-axis verdict
    convention), then `warm_start`, then `curvature`. So a tie at
    DEGRADED across all three reports `dominant_axis=pirls`.

    `None` inputs are treated as `NO-DATA` for ranking, matching
    `_combine_fit_verdicts` semantics.
    """
    if combined == "NO-DATA":
        return "none"
    rank = {"DEGRADED": 3, "MARGINAL": 2, "HEALTHY": 1, "NO-DATA": 0}
    target = rank[combined]
    # Tie-break order matches the documented preference.
    candidates = (
        ("pirls", pirls),
        ("warm_start", warm_start),
        ("curvature", curvature),
    )
    for name, verdict in candidates:
        if rank.get(verdict or "NO-DATA", 0) == target:
            return name
    # Should be unreachable: combined comes from max(rank values), so
    # one of the inputs MUST have that rank. Fall back gracefully.
    return "none"


def _curvature_health_verdict(
    *,
    fisher_frac: float | None,
    force_fisher_n: int,
) -> tuple[str, str]:
    """Classify the observed-Hessian reliability based on the
    Fisher-fallback aggregators (commits 971e67ad, 8ffa7225,
    dea37b05). Returns (verdict, detail_string).

    Tier policy:
      HEALTHY    fisher_frac < 0.05 AND force_fisher_n == 0
                 (Fisher fallback rarely needed; observed Hessian
                 is reliable across the run).
      MARGINAL   fisher_frac < 0.20 AND force_fisher_n == 0
                 (some Fisher use, but no sustained Fisher-only
                 state; Observed is mostly reliable with occasional
                 transient fallbacks).
      DEGRADED   fisher_frac >= 0.20 OR force_fisher_n > 0
                 (high Fisher use OR at least one solve fully
                 transitioned to Fisher-only — observed Hessian
                 has sustained reliability problems).
      NO-DATA    fisher_frac is None
                 (curvature-kind log markers absent; e.g., the
                 fit didn't run PIRLS at all, or pre-instrumentation
                 binary).
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
    """Classify the inner-Newton's per-solve geometric convergence
    rates into HEALTHY / MARGINAL / DEGRADED. Each `rate_i = (g_final /
    g_initial)^(1/iters)` from one PIRLS solve; healthy Newton
    geometry yields rate < 0.5 (gradient halves at least once per
    iter on average); rate ≥ 0.7 signals struggle (less than ~30%
    geometric reduction per iter — the inner solver is grinding).

    Tier policy:
      HEALTHY   p95(rate) < 0.5
                (95% of solves strongly converging; tolerates a few
                outliers without flipping the verdict — e.g. one slow
                solve in a 100-solve fit doesn't drop the trace from
                HEALTHY to MARGINAL when the rest are clean).
      MARGINAL  p50(rate) < 0.5 AND max(rate) < 0.85
                (median solve fast, no individual solve in the
                saturation regime).
      DEGRADED  otherwise
                (consistent geometry struggle across solves; warm-start
                may be collapsing toward flat or hitting near-singular
                Hessian regions).

    The earlier `max(rate) < 0.5` for HEALTHY was too strict: a fit
    with rates uniformly in [0.05, 0.45] plus one 0.52 outlier landed
    in MARGINAL even though 99% of solves were strongly converging.
    `p95 < 0.5` is the right granularity — outliers visible in the
    `max=` field, but the verdict reflects the central tendency.

    Returns (verdict, detail_string).
    """
    if not rates:
        return ("NO-DATA", "n_solves=0")
    n = len(rates)
    sorted_r = sorted(rates)
    p50 = sorted_r[n // 2]
    p95 = sorted_r[min(n - 1, int(0.95 * n))]
    rmax = sorted_r[-1]
    detail = (
        f"n_solves={n} p50={p50:.3f} p95={p95:.3f} max={rmax:.3f}"
    )
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
    """Combine the warm-start machinery's quality signals into a single
    verdict. Two axes:

      coverage   = n_accepts / (n_accepts + n_rejects + n_noops)
                   (1.0 = predictor accepted every call; 0 = predictor
                    fell through every time)
      p50_resid  = median of accepted-call residuals (0.0 = perfect
                   linearization; ≥0.5 = no better than flat warm-start)

    Verdict tiers:
      HEALTHY   coverage ≥ 0.70 AND p50_resid < 0.05
                AND p95_resid < 0.20
                AND no outer-non-finite signals
      MARGINAL  coverage ≥ 0.30 OR p50_resid < 0.30
                AND no outer-non-finite signals
      DEGRADED  any of:
                  - any outer-non-finite signals (broken geometry)
                  - predictor was tried (n_rejects + n_noops > 0)
                    AND no residuals captured (the predictor
                    was firing but never delivered a real
                    prediction — e.g. all calls fell through
                    on large Δρ, or all noop)
                  - residuals present but neither HEALTHY nor
                    MARGINAL thresholds met
      NO-DATA   the predictor was never tried (n_accepts +
                n_rejects + n_noops == 0) AND no outer-non-finite
                — the fit didn't exercise the warm-start path
                (e.g. a non-REML code path), so there's nothing
                to assess

    The HEALTHY tier's `p95_resid < 0.20` saturation guard is the
    same kind of central-tendency-safe rule as the PIRLS verdict's
    p95 threshold (commit efc54eca). A fit where p50_resid is clean
    (< 0.05) but p95_resid is in the marginal/degraded tier
    (≥ 0.20 = the same threshold the adaptive |Δρ| cap uses to mark
    "marginal" predictor faithfulness) has a tail of poor predictions
    that we don't want to claim as HEALTHY — even if the median is
    fine, ~5% of solves are starting from poor warm-starts and
    contributing extra inner-Newton work.

    `n_outer_nonfinite` is the count of [OUTER non-finite] warnings
    captured during the fit. A non-zero count means at least one
    intermediate of the outer-Hessian / leverage / adjoint computation
    produced NaN / Inf — a strict bug-signal override on any
    HEALTHY / MARGINAL classification, since broken intermediates
    invalidate the predictor-faithfulness measurements.

    Returns (verdict, detail_string). The detail string carries the
    actual numbers so reviewers can verify the verdict.
    """
    denom = max(n_accepts + n_rejects + n_noops, 1)
    coverage = n_accepts / denom
    if residuals:
        sorted_res = sorted(residuals)
        n_res = len(sorted_res)
        p50_resid = sorted_res[n_res // 2]
        p95_resid = sorted_res[min(n_res - 1, int(0.95 * n_res))]
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
        # Append tangent-line stats so reviewers see both predictor
        # distributions in the verdict's detail string. Tangent-line
        # is the fallback path; non-zero n_tangent_accepts means the
        # IFT predictor rejected at least once and the fallback
        # recovered the prediction. tangent_p50 may be None when
        # no finite residuals were captured (degenerate case).
        if tangent_p50 is not None and tangent_p50 == tangent_p50:
            detail += f" n_tangent_accepts={n_tangent_accepts} tangent_p50={tangent_p50:.2e}"
        else:
            detail += f" n_tangent_accepts={n_tangent_accepts}"
    # Override: outer-non-finite signals trump every other axis. Broken
    # geometry invalidates the predictor-faithfulness measurements
    # regardless of how clean the residuals look on their face. In
    # extremis a fit could produce healthy-looking IFT residuals on
    # the iters BEFORE the geometry broke, but the broken iters'
    # outputs are unreliable — the verdict must reflect that.
    if n_outer_nonfinite > 0:
        return ("DEGRADED", detail)
    if not residuals:
        # No residuals captured — distinguish "predictor never tried"
        # (truly NO-DATA) from "predictor tried but always fell
        # through" (DEGRADED — the warm-start machinery is collapsing
        # to flat at this surface).
        n_total = n_accepts + n_rejects + n_noops
        if n_total == 0:
            return ("NO-DATA", detail)
        # Predictor was tried but produced no real predictions (either
        # all rejects, all noops, or both). That's a degradation
        # signal — the warm-start machinery is not delivering at
        # this surface even though the path is being exercised.
        return ("DEGRADED", detail)
    if coverage >= 0.70 and p50_resid < 0.05 and p95_resid < 0.20:
        return ("HEALTHY", detail)
    if coverage >= 0.30 or p50_resid < 0.30:
        return ("MARGINAL", detail)
    return ("DEGRADED", detail)


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
    templates = {str(r["subpopulation"]): r for r in rows[: len(by_subpop)]}
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
            max_centers=(
                int(item["max_centers"])
                if item.get("max_centers") is not None
                else None
            ),
        )
        validate_method_spec(spec)
        out.append(spec)
    return out


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return max(sum(1 for _ in fh) - 1, 0)


def effective_marginal_slope_centers(
    spec: MethodSpec,
    *,
    train_rows: int | None = None,
) -> int:
    if train_rows is not None and train_rows <= 0:
        raise RuntimeError("train_rows must be positive when provided")
    requested = int(spec.centers or 24)
    if spec.max_centers is None:
        return requested
    scorewarp_width = int(spec.logslope_linkwiggle_knots or 0)
    capped = int(spec.max_centers) - scorewarp_width
    if capped <= 0:
        raise RuntimeError(
            f"method '{spec.name}' max_centers={spec.max_centers} leaves no room "
            f"after score-warp knots={scorewarp_width}"
        )
    return min(requested, capped)


def rust_formula_classification(spec: MethodSpec) -> tuple[str, str]:
    """Build mean + sigma formulas for biobank classification lanes.

    PCs enter as a SINGLE JOINT smooth over the ancestry manifold using the
    lane's `spatial_basis`; no per-PC linear terms, no separate per-axis
    smooths. Lat/lon coordinates are NOT used as predictors. The mean and
    sigma blocks share the joint-PC term so any heteroscedastic structure
    is over the same ancestry surface as the location surface.
    """
    centers = int(spec.centers or 60)
    pc_count = int(spec.pc_count)
    spatial = _biobank_pc_smooth_term(spec.spatial_basis, pc_count, centers)
    mean_terms = [
        "pgs_std",
        "sex",
        "smooth(age_entry_std)",
        spatial,
    ]
    sigma_terms = [
        "smooth(age_entry_std)",
        spatial,
    ]
    return "phenotype ~ " + " + ".join(mean_terms), " + ".join(sigma_terms)


def rust_marginal_slope_formula_classification(spec: MethodSpec, centers: int) -> tuple[str, str]:
    """Build mean and logslope formulas for biobank marginal-slope classification.

    Uses the shared joint-PC helper so duchon / thinplate / matern lanes all
    route through the same ancestry-manifold contract.
    """
    spatial = _biobank_pc_smooth_term(spec.spatial_basis, int(spec.pc_count), centers)
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
    centers = effective_marginal_slope_centers(spec, train_rows=train_rows)
    preflight = preflight_marginal_slope_biobank(
        n_train=train_rows,
        d_pc=int(spec.pc_count),
        centers=centers,
        linkwiggle_knots=spec.mean_linkwiggle_knots,
        scorewarp_knots=spec.logslope_linkwiggle_knots,
    )
    print("\n".join(preflight.lines), file=sys.stderr, flush=True)
    ctn_t0 = time.perf_counter()
    ctn_train_csv, ctn_test_csv, ctn_diagnostics = fit_conditional_pgs_ctn_for_marginal_slope(
        rust_bin=rust_bin,
        spec=spec,
        train_csv=train_csv,
        test_csv=test_csv,
        out_dir=out_dir,
        centers=centers,
    )
    ctn_fit_sec = time.perf_counter() - ctn_t0
    print("\n".join(ctn_diagnostics), file=sys.stderr, flush=True)
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
    fit_sec = ctn_fit_sec + disease_fit_sec
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
        "ctn_fit_sec": ctn_fit_sec,
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
    spatial = _biobank_pc_smooth_term(spec.spatial_basis, int(spec.pc_count), centers)
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
    pc_term = _biobank_pc_smooth_term(spec.spatial_basis, pc_count, centers)
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


def mgcv_formula_classification(spec: MethodSpec) -> str:
    """mgcv classification formula: joint PC ancestry smooth, no lat/lon.

    Mirrors the rust contract — the PCs enter as a single multi-D mgcv smooth
    using the lane's `spatial_basis`, never as 16 independent linear terms.
    Geographic coordinates are not predictors.
    """
    pc_count = int(spec.pc_count)
    centers = int(spec.centers or 60)
    pc_term = _mgcv_pc_smooth_term(spec.spatial_basis, pc_count, centers)
    terms = [
        "pgs_std",
        "sex",
        "s(age_entry_std, bs='ps', k=min(10, nrow(train_df)-1))",
        pc_term,
    ]
    return "phenotype ~ " + " + ".join(terms)


def mgcv_survival_formula(spec: MethodSpec) -> str:
    """mgcv survival formula: joint PC ancestry smooth, no lat/lon.

    Built from `spec.spatial_basis`, `spec.centers`, and `spec.pc_count` —
    not hardcoded — so YAML config flips actually take effect.
    """
    pc_count = int(spec.pc_count)
    centers = int(spec.centers or 60)
    pc_term = _mgcv_pc_smooth_term(spec.spatial_basis, pc_count, centers)
    terms = [
        "pgs_std",
        "sex",
        "s(age_entry_std, bs='ps', k=min(10, nrow(train_df)-1))",
        pc_term,
    ]
    return "time ~ " + " + ".join(terms)


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
    ctn_fit_sec = 0.0
    logslope_formula = None
    fit_csv = train_csv
    prediction_rows_raw = test_rows_raw
    train_metric_rows_raw = train_rows_raw
    if likelihood_mode == "marginal-slope":
        centers = effective_marginal_slope_centers(spec, train_rows=len(train_rows_raw))
        preflight = preflight_marginal_slope_biobank(
            n_train=len(train_rows_raw),
            d_pc=int(spec.pc_count),
            centers=centers,
            linkwiggle_knots=spec.mean_linkwiggle_knots,
            scorewarp_knots=spec.logslope_linkwiggle_knots,
        )
        print("\n".join(preflight.lines), file=sys.stderr, flush=True)
        ctn_t0 = time.perf_counter()
        ctn_train_csv, ctn_test_csv, ctn_diagnostics = fit_conditional_pgs_ctn_for_marginal_slope(
            rust_bin=rust_bin,
            spec=spec,
            train_csv=train_csv,
            test_csv=test_csv,
            out_dir=out_dir,
            centers=centers,
        )
        ctn_fit_sec = time.perf_counter() - ctn_t0
        print("\n".join(ctn_diagnostics), file=sys.stderr, flush=True)
        fit_csv = ctn_train_csv
        train_metric_rows_raw = read_csv_rows(ctn_train_csv)
        prediction_rows_raw = read_csv_rows(ctn_test_csv)
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
    with tempfile.TemporaryDirectory(prefix="gam_biobank_survival_", dir=out_dir) as td:
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
        fit_sec = ctn_fit_sec + survival_fit_sec
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
        "ctn_fit_sec": ctn_fit_sec,
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


def run_r_mgcv_classification(spec: MethodSpec, train_csv: Path, test_csv: Path, out_dir: Path) -> dict[str, Any]:
    if not tool_exists("Rscript"):
        raise RuntimeError("Rscript is required for mgcv methods")
    formula = mgcv_formula_classification(spec)
    script = f"""
args <- commandArgs(trailingOnly = TRUE)
train_path <- args[1]
test_path <- args[2]
out_path <- args[3]
suppressPackageStartupMessages({{
  library(mgcv)
  library(jsonlite)
}})
train_df <- read.csv(train_path)
test_df <- read.csv(test_path)
t0 <- proc.time()[["elapsed"]]
fit <- gam(as.formula("{formula}"), family=binomial(link="logit"), data=train_df, method="REML", select=TRUE)
fit_sec <- proc.time()[["elapsed"]] - t0
t1 <- proc.time()[["elapsed"]]
pred <- as.numeric(predict(fit, newdata=test_df, type="response"))
pred_sec <- proc.time()[["elapsed"]] - t1
write.csv(data.frame(mean=pred), out_path, row.names=FALSE)
cat(jsonlite::toJSON(list(fit_sec=fit_sec, predict_sec=pred_sec), auto_unbox=TRUE))
"""
    script_path = out_dir / f"{spec.name}.R"
    script_path.write_text(script, encoding="utf-8")
    pred_path = out_dir / f"{spec.name}.pred.csv"
    rc, out, err = run_cmd_stream(["Rscript", str(script_path), str(train_csv), str(test_csv), str(pred_path)], cwd=ROOT)
    if rc != 0:
        raise RuntimeError(err.strip() or out.strip() or f"{spec.name} mgcv fit failed")
    meta = json.loads(out.strip().splitlines()[-1])
    pred_rows = read_csv_rows(pred_path)
    pred = np.array([float(r["mean"]) for r in pred_rows], dtype=float)
    y_train = csv_numeric_column(train_csv, "phenotype")
    y_test = csv_numeric_column(test_csv, "phenotype")
    metrics = classification_metrics(y_test, pred, float(np.mean(y_train)))
    return {
        "fit_sec": float(meta["fit_sec"]),
        "predict_sec": float(meta["predict_sec"]),
        "metrics": metrics,
        "prediction_path": str(pred_path),
        "model_spec": f"mgcv {spec.spatial_basis} holdout",
    }


def run_r_mgcv_survival(spec: MethodSpec, train_csv: Path, test_csv: Path, out_dir: Path) -> dict[str, Any]:
    if not tool_exists("Rscript"):
        raise RuntimeError("Rscript is required for mgcv survival methods")
    formula = mgcv_survival_formula(spec)
    script = f"""
args <- commandArgs(trailingOnly = TRUE)
train_path <- args[1]
test_path <- args[2]
out_pred <- args[3]
suppressPackageStartupMessages({{
  library(mgcv)
  library(jsonlite)
}})
train_df <- read.csv(train_path)
test_df <- read.csv(test_path)
t0 <- proc.time()[["elapsed"]]
fit <- gam(as.formula("{formula}"), family=cox.ph(), weights=as.numeric(train_df$event), data=train_df, method="REML", select=TRUE)
fit_sec <- proc.time()[["elapsed"]] - t0
t1 <- proc.time()[["elapsed"]]
train_risk <- as.numeric(predict(fit, newdata=train_df, type="link"))
test_risk <- as.numeric(predict(fit, newdata=test_df, type="link"))
pred_sec <- proc.time()[["elapsed"]] - t1
write.csv(data.frame(train_risk=train_risk), paste0(out_pred, ".train"), row.names=FALSE)
write.csv(data.frame(risk=test_risk), out_pred, row.names=FALSE)
cat(toJSON(list(fit_sec=fit_sec, predict_sec=pred_sec), auto_unbox=TRUE))
"""
    script_path = out_dir / f"{spec.name}.R"
    script_path.write_text(script, encoding="utf-8")
    pred_path = out_dir / f"{spec.name}.pred.csv"
    rc, out, err = run_cmd_stream(["Rscript", str(script_path), str(train_csv), str(test_csv), str(pred_path)], cwd=ROOT)
    if rc != 0:
        raise RuntimeError(err.strip() or out.strip() or f"{spec.name} mgcv survival fit failed")
    meta = json.loads(out.strip().splitlines()[-1])
    test_risk = np.array([float(r["risk"]) for r in read_csv_rows(pred_path)], dtype=float)
    train_risk = np.array([float(r["train_risk"]) for r in read_csv_rows(Path(str(pred_path) + ".train"))], dtype=float)
    train_rows = [{k: (float(v) if k in {"time", "event"} else v) for k, v in r.items()} for r in read_csv_rows(train_csv)]
    test_rows = [{k: (float(v) if k in {"time", "event"} else v) for k, v in r.items()} for r in read_csv_rows(test_csv)]
    metrics = survival_metrics(train_rows, test_rows, train_risk, test_risk)
    return {
        "fit_sec": float(meta["fit_sec"]),
        "predict_sec": float(meta["predict_sec"]),
        "metrics": metrics,
        "prediction_path": str(pred_path),
        "model_spec": "mgcv cox.ph holdout",
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
        elif spec.backend == "r_mgcv":
            result = run_r_mgcv_classification(spec, disease_train, disease_test, out_dir)
        else:
            raise RuntimeError(f"unsupported disease backend '{spec.backend}'")
    elif spec.dataset == "survival":
        if spec.backend == "rust_survival":
            result = run_rust_survival(spec, survival_train, survival_test, out_dir)
        elif spec.backend == "r_mgcv_survival":
            result = run_r_mgcv_survival(spec, survival_train, survival_test, out_dir)
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
        os.environ["BIOBANK_ROUTING_LOG_PATH"] = str(routing_log_path)
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
        # failure mode at biobank scale (the [HEARTBEAT] line and
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def resolve_prediction_path(raw_path: str) -> Path | None:
        direct = Path(raw_path)
        if direct.exists():
            return direct
        artifacts_root = prep_dir.parent / "artifacts"
        if not artifacts_root.exists():
            return None
        matches = list(artifacts_root.rglob(direct.name))
        return matches[0] if matches else None

    paths: list[Path] = []
    ok = [r for r in results if r.get("status") == "ok"]
    if not ok:
        return paths
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ok:
        grouped[str(row["dataset"])].append(row)
    for dataset, rows in grouped.items():
        if dataset == "disease":
            metrics = [("auc", "AUC"), ("pr_auc", "PR-AUC"), ("brier", "Brier"), ("logloss", "LogLoss"), ("ece", "ECE")]
        else:
            metrics = [("c_index", "C-index"), ("lifted_brier", "Lifted Brier"), ("lifted_logloss", "Lifted LogLoss"), ("brier", "Brier"), ("logloss", "LogLoss")]
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
        ax_metrics = axes[0, 0]
        names = [r["method"] for r in rows]
        x = np.arange(len(names))
        width = 0.14
        for i, (key, label) in enumerate(metrics):
            vals = [float(r["metrics"].get(key, np.nan)) for r in rows]
            ax_metrics.bar(x + (i - len(metrics) / 2) * width, vals, width=width, label=label)
        ax_metrics.set_xticks(x)
        ax_metrics.set_xticklabels(names, rotation=25, ha="right")
        ax_metrics.set_title(f"{dataset.title()} metrics")
        ax_metrics.legend(fontsize=8)

        ax_time = axes[0, 1]
        fit = [float(r["fit_sec"]) for r in rows]
        pred = [float(r["predict_sec"]) for r in rows]
        ax_time.bar(x, fit, label="fit_sec")
        ax_time.bar(x, pred, bottom=fit, label="predict_sec")
        ax_time.set_xticks(x)
        ax_time.set_xticklabels(names, rotation=25, ha="right")
        ax_time.set_title("Train/Test runtime")
        ax_time.legend(fontsize=8)

        all_rows = read_csv_rows(prep_dir / "all_cohort.csv")
        lat = np.array([float(r["lat_final"]) for r in all_rows], dtype=float)
        lon = np.array([float(r["lon_final"]) for r in all_rows], dtype=float)
        if dataset == "disease":
            value = np.array([float(r["phenotype_prob"]) for r in all_rows], dtype=float)
            title = "Simulated disease risk map"
        else:
            value = np.array([float(r["event"]) for r in all_rows], dtype=float)
            title = "Observed event map"
        sc = axes[1, 0].scatter(lon[::max(1, len(lon) // 25000)], lat[::max(1, len(lat) // 25000)], c=value[::max(1, len(value) // 25000)], s=5, alpha=0.6, cmap="viridis")
        axes[1, 0].set_xlabel("Longitude")
        axes[1, 0].set_ylabel("Latitude")
        axes[1, 0].set_title(title)
        fig.colorbar(sc, ax=axes[1, 0], fraction=0.046, pad=0.04)

        if dataset == "disease":
            test_rows = read_csv_rows(prep_dir / "disease_test.csv")
            y = np.array([float(r["phenotype"]) for r in test_rows], dtype=float)
            best = max(rows, key=lambda r: float(r["metrics"].get("auc", -1.0)))
            pred_path = resolve_prediction_path(str(best["prediction_path"]))
            if pred_path is not None:
                pred_rows = read_csv_rows(pred_path)
                p = np.array([float(r["mean"]) for r in pred_rows], dtype=float)
                order = np.argsort(p)
                axes[1, 1].plot(np.linspace(0, 1, len(order)), p[order], label="predicted")
                axes[1, 1].plot(np.linspace(0, 1, len(order)), y[order], label="observed", alpha=0.8)
                axes[1, 1].set_title(f"Best model probability profile: {best['method']}")
            else:
                axes[1, 1].text(0.5, 0.5, "prediction artifact missing", ha="center", va="center")
                axes[1, 1].set_title("Best-model profile unavailable")
        else:
            test_rows = read_csv_rows(prep_dir / "survival_test.csv")
            times = np.array([float(r["time"]) for r in test_rows], dtype=float)
            events = np.array([float(r["event"]) for r in test_rows], dtype=float)
            best = max(rows, key=lambda r: float(r["metrics"].get("c_index", -1.0)))
            pred_path = resolve_prediction_path(str(best["prediction_path"]))
            if pred_path is not None:
                risk_rows = read_csv_rows(pred_path)
                risk_key = "risk" if "risk" in risk_rows[0] else ("failure_prob" if "failure_prob" in risk_rows[0] else "eta")
                risk = np.array([float(r[risk_key]) for r in risk_rows], dtype=float)
                q = np.quantile(risk, [0.0, 0.33, 0.66, 1.0])
                groups = np.digitize(risk, q[1:-1], right=True)
                try:
                    lifelines: Any = importlib.import_module("lifelines")
                    KaplanMeierFitter = lifelines.KaplanMeierFitter
                except ModuleNotFoundError as exc:
                    raise RuntimeError("lifelines is required for survival plotting") from exc
                kmf = KaplanMeierFitter()
                for grp, color in zip([0, 1, 2], ["#2ca02c", "#ff7f0e", "#d62728"]):
                    mask = groups == grp
                    if not np.any(mask):
                        continue
                    label = f"Q{grp + 1}"
                    kmf.fit(times[mask], event_observed=events[mask], label=label)
                    sf = kmf.survival_function_
                    axes[1, 1].step(sf.index, sf.iloc[:, 0], where="post", color=color, label=label)
                axes[1, 1].set_title(f"Risk-stratified KM: {best['method']}")
                axes[1, 1].set_xlabel("Years")
                axes[1, 1].set_ylabel("Survival")
            else:
                axes[1, 1].text(0.5, 0.5, "prediction artifact missing", ha="center", va="center")
                axes[1, 1].set_title("Risk KM unavailable")
        axes[1, 1].legend(fontsize=8)

        out_path = out_dir / f"{dataset}_summary.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(out_path)
    return paths


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
    zip_path = out_dir / "biobank_scale_bundle.zip"
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
    parser = argparse.ArgumentParser(description="Biobank-scale synthetic benchmark runner")
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prep.add_argument("--out-dir", type=Path, required=True)
    prep.add_argument("--smoke", action="store_true")
    prep.add_argument("--target-n", type=int, default=None)
    prep.add_argument("--smoke-target-n", type=int, default=None)
    prep.set_defaults(func=do_prepare)

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
