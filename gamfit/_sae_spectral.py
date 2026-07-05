"""Python facade for the SAE spectral / routing-geometry diagnostics.

Thin wrappers over the Rust ``gam-sae`` engine (``gam::terms::sae``): the
dimension spectrometer, the block-firing circle-coordinate readout, the
routability floor + audit, the sparse-dictionary dual certificate, harmonic
super-resolution, and the contract-composition / loop-holonomy calculus.

Per the project spec (Python is a thin wrapper over Rust, no math in Python),
every number here is computed in the Rust core; this module only coerces the
numpy inputs to the contiguous dtypes the FFI expects and packs the returned
dicts into frozen dataclasses. All heavy state is FP32, matching the collapsed
linear / block lanes these diagnostics read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._binding import rust_module


def _as_2d_f32(values: Any, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{label} must be a 2-D array; got shape {arr.shape}")
    return np.ascontiguousarray(arr)


def _as_2d_u32(values: Any, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.uint32)
    if arr.ndim != 2:
        raise ValueError(f"{label} must be a 2-D array; got shape {arr.shape}")
    return np.ascontiguousarray(arr)


# --------------------------------------------------------------------------- #
# Dimension spectrometer
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SpectrometerReport:
    """Intrinsic-dimension estimate from a reconstruction-loss scaling law.

    Attributes
    ----------
    rungs:
        ``(K, mean per-row reconstruction loss)`` per ladder rung, ascending K.
    noise_floor:
        Profiled loss plateau ``sigma^2``.
    slope, slope_se:
        Fitted log-log slope ``m`` and its standard error.
    d_hat, d_hat_se:
        Intrinsic-dimension estimate ``d_hat = -2/m`` and its delta-method SE.
    floor_saturated:
        True when the slope's CI contains zero (``d_hat`` unreliable).
    """

    rungs: list[tuple[int, float]]
    noise_floor: float
    slope: float
    slope_se: float
    d_hat: float
    d_hat_se: float
    floor_saturated: bool


def dimension_spectrometer(
    data: Any,
    *,
    k_min: int = 4,
    n_doublings: int = 6,
    active: int = 1,
    minibatch: int = 512,
    max_epochs: int = 30,
    score_tile: int = 4096,
    code_ridge: float = 1.0e-6,
    decoder_ridge: float = 1.0e-9,
    tolerance: float = 1.0e-6,
    score_mode: str = "required",
) -> SpectrometerReport:
    """Estimate intrinsic dimension by fitting single-atom dictionaries along a
    doubling ladder ``k_min * 2**j`` and inverting the loss scaling law."""
    x = _as_2d_f32(data, "data")
    payload = rust_module().dimension_spectrometer(
        x,
        k_min,
        n_doublings,
        active,
        minibatch,
        max_epochs,
        score_tile,
        code_ridge,
        decoder_ridge,
        tolerance,
        score_mode,
    )
    return SpectrometerReport(
        rungs=[(int(k), float(loss)) for k, loss in payload["rungs"]],
        noise_floor=float(payload["noise_floor"]),
        slope=float(payload["slope"]),
        slope_se=float(payload["slope_se"]),
        d_hat=float(payload["d_hat"]),
        d_hat_se=float(payload["d_hat_se"]),
        floor_saturated=bool(payload["floor_saturated"]),
    )


# --------------------------------------------------------------------------- #
# Block-firing circle coordinates
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BlockCoordinateReport:
    """Per-firing circle-coordinate readout for one ``b = 2`` block.

    ``block``/``row`` index the firing; ``t``/``amplitude`` are the recovered
    circle phase and radius, ``t_se``/``amplitude_se`` their SEs, and
    ``t_se_clamped`` flags firings whose phase SE hit the uniform ceiling.
    """

    sigma_hat: float
    mean_radius: float
    n_firings: int
    block: np.ndarray
    row: np.ndarray
    t: np.ndarray
    amplitude: np.ndarray
    t_se: np.ndarray
    amplitude_se: np.ndarray
    t_se_clamped: list[bool]


def block_firing_coordinates(
    decoder: Any,
    blocks: Any,
    gates: Any,
    codes: Any,
    block: int,
    *,
    block_topk: int = 1,
) -> BlockCoordinateReport:
    """Recover per-firing circle coordinates (phase, amplitude, SEs) for one
    ``b = 2`` block of a fitted block-sparse dictionary."""
    dec = _as_2d_f32(decoder, "decoder")
    blk = _as_2d_u32(blocks, "blocks")
    gat = _as_2d_f32(gates, "gates")
    cod = np.ascontiguousarray(np.asarray(codes, dtype=np.float32))
    if cod.ndim != 3:
        raise ValueError(f"codes must be a 3-D N x k x b array; got shape {cod.shape}")
    payload = rust_module().block_firing_coordinates(dec, blk, gat, cod, block, block_topk)
    return BlockCoordinateReport(
        sigma_hat=float(payload["sigma_hat"]),
        mean_radius=float(payload["mean_radius"]),
        n_firings=int(payload["n_firings"]),
        block=payload["block"],
        row=payload["row"],
        t=payload["t"],
        amplitude=payload["amplitude"],
        t_se=payload["t_se"],
        amplitude_se=payload["amplitude_se"],
        t_se_clamped=[bool(v) for v in payload["t_se_clamped"]],
    )


# --------------------------------------------------------------------------- #
# Routability floor + audit
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RoutabilityFloor:
    """Closed-form routability floor on the routable energy fraction."""

    p: int
    n_blocks: int
    b_max: int
    delta: float
    floor: float
    minimum_routable_energy: float


@dataclass(frozen=True)
class RoutabilityAudit:
    """Empirical routability audit against real residual rows."""

    n_rows: int
    floor: RoutabilityFloor
    quantiles: list[tuple[float, float]]
    empirical_mean: float
    empirical_max: float
    confidence_quantile: float
    coherence_excess: float
    fraction_below_floor: float


def _floor_from_payload(payload: dict[str, Any]) -> RoutabilityFloor:
    return RoutabilityFloor(
        p=int(payload["p"]),
        n_blocks=int(payload["n_blocks"]),
        b_max=int(payload["b_max"]),
        delta=float(payload["delta"]),
        floor=float(payload["floor"]),
        minimum_routable_energy=float(payload["minimum_routable_energy"]),
    )


def routability_floor(p: int, n_blocks: int, b_max: int, delta: float) -> RoutabilityFloor:
    """Closed-form floor ``sqrt(b_max/p) + sqrt(2*ln(K/delta)/p)`` plus the
    derived minimum routable energy fraction."""
    return _floor_from_payload(rust_module().routability_floor(p, n_blocks, b_max, delta))


def routability_audit(
    decoder: Any,
    residuals: Any,
    *,
    block_size: int = 1,
    delta: float = 0.05,
    quantile_levels: tuple[float, ...] = (0.5, 0.9, 0.99),
) -> RoutabilityAudit:
    """Measure a fitted dictionary's max-cross-gate distribution against real
    residual rows and compare it to the closed-form floor."""
    dec = _as_2d_f32(decoder, "decoder")
    res = _as_2d_f32(residuals, "residuals")
    payload = rust_module().routability_audit(
        dec, res, block_size, delta, list(quantile_levels)
    )
    return RoutabilityAudit(
        n_rows=int(payload["n_rows"]),
        floor=_floor_from_payload(payload["floor"]),
        quantiles=[(float(q), float(v)) for q, v in payload["quantiles"]],
        empirical_mean=float(payload["empirical_mean"]),
        empirical_max=float(payload["empirical_max"]),
        confidence_quantile=float(payload["confidence_quantile"]),
        coherence_excess=float(payload["coherence_excess"]),
        fraction_below_floor=float(payload["fraction_below_floor"]),
    )


# --------------------------------------------------------------------------- #
# Sparse-dict dual certificate
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DualCertificateReport:
    """Global-optimality dual certificate over a fitted linear-lane fit."""

    n_rows: int
    frac_certified: float
    optimality_ratio_quantiles: list[tuple[float, float]]
    birth_candidates: list[tuple[int, int, float]]


def sparse_dict_dual_certificate(
    data: Any,
    decoder: Any,
    indices: Any,
    codes: Any,
    *,
    max_candidates: int = 16,
) -> DualCertificateReport:
    """Certify (or refute) global optimality of a fitted sparse routing and
    surface the strongest strictly-improving birth candidates."""
    dat = _as_2d_f32(data, "data")
    dec = _as_2d_f32(decoder, "decoder")
    idx = _as_2d_u32(indices, "indices")
    cod = _as_2d_f32(codes, "codes")
    payload = rust_module().sparse_dict_dual_certificate(dat, dec, idx, cod, max_candidates)
    return DualCertificateReport(
        n_rows=int(payload["n_rows"]),
        frac_certified=float(payload["frac_certified"]),
        optimality_ratio_quantiles=[
            (float(q), float(v)) for q, v in payload["optimality_ratio_quantiles"]
        ],
        birth_candidates=[
            (int(r), int(a), float(e)) for r, a, e in payload["birth_candidates"]
        ],
    )


# --------------------------------------------------------------------------- #
# Harmonic super-resolution
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SpikeRecovery:
    """Point masses recovered from a harmonic atom's Fourier coefficients."""

    t: np.ndarray
    amplitude: np.ndarray
    model_order: int
    residual: float
    hankel_singular_values: np.ndarray


def separation_limit(n_harmonics: int) -> float:
    """Candes-Fernandez-Granda separation threshold ``~2/H``."""
    return float(rust_module().separation_limit(n_harmonics))


def recover_spikes(fourier_coeffs: Any, *, sigma: float = 0.0) -> SpikeRecovery:
    """Recover spikes ``{(a_j, t_j)}`` from per-harmonic ``(cos, sin)`` Fourier
    coefficient pairs by the matrix-pencil / Prony method."""
    coeffs = np.asarray(fourier_coeffs, dtype=np.float64)
    if coeffs.ndim != 2 or coeffs.shape[1] != 2:
        raise ValueError(
            f"fourier_coeffs must be an H x 2 array of (cos, sin) pairs; got shape {coeffs.shape}"
        )
    pairs = [(float(c), float(s)) for c, s in coeffs]
    payload = rust_module().recover_spikes(pairs, sigma)
    return SpikeRecovery(
        t=payload["t"],
        amplitude=payload["amplitude"],
        model_order=int(payload["model_order"]),
        residual=float(payload["residual"]),
        hankel_singular_values=payload["hankel_singular_values"],
    )


# --------------------------------------------------------------------------- #
# Contract composition / loop holonomy
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ComposedContract:
    """End-to-end shadowing bound from composing a contract chain."""

    total_defect: float
    per_stage_contribution: list[float]
    domain_ok: bool


@dataclass(frozen=True)
class HolonomyReport:
    """Net ``O(2)`` element of a closed loop of circle isometries."""

    loop_len: int
    net_sign: int
    net_angle: float
    is_trivial: bool
    angle_tolerance: float


def compose_contracts(chain: list[tuple[str, float, float, float]]) -> ComposedContract:
    """Compose a chain of ``(name, domain_radius, defect, lipschitz)`` contracts
    into one end-to-end shadowing bound."""
    stages = [(str(n), float(dr), float(de), float(li)) for n, dr, de, li in chain]
    payload = rust_module().compose_contracts(stages)
    return ComposedContract(
        total_defect=float(payload["total_defect"]),
        per_stage_contribution=[float(v) for v in payload["per_stage_contribution"]],
        domain_ok=bool(payload["domain_ok"]),
    )


def loop_holonomy(
    edges: list[tuple[int, float]],
    defects: list[float],
) -> HolonomyReport:
    """Compose a closed loop of ``(sign, angle)`` circle isometries and report
    the net ``O(2)`` element with its measure-don't-latch triviality verdict."""
    edge_pairs = [(int(s), float(a)) for s, a in edges]
    payload = rust_module().loop_holonomy(edge_pairs, [float(d) for d in defects])
    return HolonomyReport(
        loop_len=int(payload["loop_len"]),
        net_sign=int(payload["net_sign"]),
        net_angle=float(payload["net_angle"]),
        is_trivial=bool(payload["is_trivial"]),
        angle_tolerance=float(payload["angle_tolerance"]),
    )
