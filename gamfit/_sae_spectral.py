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
from pathlib import Path
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


def _as_optional_1d_f64(values: Any | None, label: str) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{label} must be a 1-D array; got shape {arr.shape}")
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


def _load_decoder_checkpoint(checkpoint: Any, decoder_key: str | None) -> tuple[np.ndarray, dict[str, Any]]:
    if isinstance(checkpoint, np.ndarray):
        return _as_2d_f32(checkpoint, "checkpoint decoder"), {
            "format": "array",
            "decoder_key": None,
        }
    if isinstance(checkpoint, dict):
        key = decoder_key or "decoder"
        if key not in checkpoint:
            raise KeyError(f"checkpoint dict does not contain decoder key {key!r}")
        return _as_2d_f32(checkpoint[key], f"checkpoint[{key!r}]"), {
            "format": "mapping",
            "decoder_key": key,
        }

    path = Path(checkpoint)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return _as_2d_f32(np.load(path), str(path)), {
            "format": "npy",
            "path": str(path),
            "decoder_key": None,
        }
    if suffix == ".npz":
        archive = np.load(path)
        key = decoder_key or "decoder"
        if key not in archive.files:
            raise KeyError(f"{path} does not contain decoder array {key!r}")
        return _as_2d_f32(archive[key], f"{path}:{key}"), {
            "format": "npz",
            "path": str(path),
            "decoder_key": key,
        }
    if suffix == ".safetensors":
        try:
            from safetensors.numpy import load_file as load_safetensors
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "reading .safetensors checkpoints requires the safetensors package"
            ) from exc
        tensors = load_safetensors(str(path))
        key = decoder_key or "decoder"
        if key not in tensors:
            raise KeyError(f"{path} does not contain decoder tensor {key!r}")
        return _as_2d_f32(tensors[key], f"{path}:{key}"), {
            "format": "safetensors",
            "path": str(path),
            "decoder_key": key,
        }
    raise ValueError(
        f"unsupported SAE checkpoint format {suffix!r}; expected .npy, .npz, or .safetensors"
    )


def _dense_codes_from_sparse(indices: np.ndarray, sparse_codes: np.ndarray, k: int) -> np.ndarray:
    dense = np.zeros((indices.shape[0], k), dtype=np.float32)
    rows = np.arange(indices.shape[0])[:, None]
    dense[rows, indices] = sparse_codes
    return np.ascontiguousarray(dense)


def audit_sae(
    checkpoint: Any,
    activations: Any,
    *,
    codes: Any | None = None,
    random_weight_codes: Any,
    decoder_key: str | None = None,
    active: int | None = None,
    block_size: int = 1,
    block_topk: int | None = None,
    delta: float = 0.05,
    quantile_levels: tuple[float, ...] | None = (0.5, 0.9, 0.99),
    max_candidates: int = 16,
    coordinate_blocks: list[int] | tuple[int, ...] | None = None,
    activation_threshold: float = 0.0,
    max_absorption_pairs: int = 32,
    transport: tuple[Any, Any] | None = None,
    transport_theta_in: Any | None = None,
    transport_theta_out: Any | None = None,
    transport_layer_from: int = 0,
    transport_layer_to: int = 1,
    score_tile: int = 4096,
    code_ridge: float = 1.0e-6,
    score_mode: str = "required",
) -> dict[str, Any]:
    """Run GAM diagnostics on a frozen external SAE dictionary.

    ``checkpoint`` may be a decoder array, a ``.npy`` decoder matrix, a ``.npz``
    archive containing ``decoder`` (or ``decoder_key``), a safetensors file with
    that tensor, or a mapping with that key. The expected decoder shape is
    ``K x P``: one dictionary row per atom and one column per activation
    dimension. ``activations`` is ``N x P``.

    If dense SAE ``codes`` (``N x K``) are supplied, they are audited as the
    frozen external encoder output. If not, the Rust sparse router encodes
    ``activations`` against the frozen decoder with ``active`` atoms per row
    (default ``1``) before running the audit. ``random_weight_codes`` is an
    architecture-matched random-weight encoder donor used for the required
    null battery attached to topology and atlas claims. All certificate,
    routability, coordinate-SE, topology, and atlas-nerve quantities are
    computed by Rust.
    """
    dec, checkpoint_meta = _load_decoder_checkpoint(checkpoint, decoder_key)
    acts = _as_2d_f32(activations, "activations")
    if acts.shape[1] != dec.shape[1]:
        raise ValueError(
            f"activations have P={acts.shape[1]} columns but decoder has P={dec.shape[1]}"
        )
    if codes is None:
        route_active = 1 if active is None else int(active)
        routed = rust_module().sparse_dictionary_transform_ffi(
            acts,
            dec,
            route_active,
            int(score_tile),
            float(code_ridge),
            score_mode,
        )
        cod = _dense_codes_from_sparse(routed["indices"], routed["codes"], dec.shape[0])
        route_meta: dict[str, Any] = {
            "source": "rust_sparse_router",
            "active": route_active,
            "score_route_stats": dict(routed.get("score_route_stats", {})),
        }
    else:
        cod = _as_2d_f32(codes, "codes")
        route_meta = {"source": "external_codes"}
    if cod.shape != (acts.shape[0], dec.shape[0]):
        raise ValueError(
            f"codes must have shape (N, K)=({acts.shape[0]}, {dec.shape[0]}); got {cod.shape}"
        )
    rw_cod = _as_2d_f32(random_weight_codes, "random_weight_codes")
    if rw_cod.shape[1] != dec.shape[0] or rw_cod.shape[0] == 0:
        raise ValueError(
            "random_weight_codes must be a non-empty matrix with K="
            f"{dec.shape[0]} columns; got {rw_cod.shape}"
        )
    if transport is not None:
        if transport_theta_in is not None or transport_theta_out is not None:
            raise ValueError(
                "pass either transport=(theta_in, theta_out) or transport_theta_in/transport_theta_out"
            )
        transport_theta_in, transport_theta_out = transport
    theta_in = _as_optional_1d_f64(transport_theta_in, "transport_theta_in")
    theta_out = _as_optional_1d_f64(transport_theta_out, "transport_theta_out")
    q_levels = None if quantile_levels is None else [float(q) for q in quantile_levels]
    blocks = None if coordinate_blocks is None else [int(block) for block in coordinate_blocks]
    payload = rust_module().audit_sae(
        dec,
        cod,
        acts,
        rw_cod,
        {
            "active": active,
            "block_size": block_size,
            "block_topk": block_topk,
            "delta": delta,
            "quantile_levels": q_levels,
            "max_candidates": max_candidates,
            "coordinate_blocks": blocks,
            "activation_threshold": activation_threshold,
            "max_absorption_pairs": max_absorption_pairs,
            "transport_theta_in": theta_in,
            "transport_theta_out": theta_out,
            "transport_layer_from": transport_layer_from,
            "transport_layer_to": transport_layer_to,
        },
    )
    report = dict(payload)
    report["checkpoint"] = checkpoint_meta
    report["route_source"] = route_meta
    report["api"] = "gamfit.audit_sae"
    return report


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


# --------------------------------------------------------------------------- #
# Atlas nerve (Čech-nerve Betti signature over block charts)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AtlasNerveDiagram:
    """Čech-nerve reduction of a block-sparse dictionary's per-block charts.

    ``computed`` is False for shapes the nerve does not apply to (scalar
    ``block_size == 1``, a width that does not divide ``K`` into >= 2 charts, or
    more blocks than ``max_charts`` without an explicit ``blocks`` list), in which
    case only ``reason`` is populated. When ``computed`` is True, ``betti`` is the
    ``{b0, b1, b2}`` signature and the simplex counts / covering side describe the
    reduced nerve complex.
    """

    computed: bool
    reason: str | None = None
    chart_blocks: list[int] | None = None
    betti: dict[str, int | None] | None = None
    n_vertices: int | None = None
    n_edges: int | None = None
    n_triangles: int | None = None
    n_tetrahedra: int | None = None
    sampled_support_size: int | None = None
    covering_side: str | None = None
    max_filtration: float | None = None
    note: str | None = None


def atlas_nerve_diagram(
    codes: Any,
    block_size: int,
    *,
    activation_threshold: float = 1.0e-6,
    blocks: Any | None = None,
    max_charts: int = 16,
) -> AtlasNerveDiagram:
    """Build the atlas-nerve Betti signature from a dense ``N x K`` block-sparse
    code matrix, one chart per ``block_size``-wide block. Returns a report whose
    ``computed`` flag is False (with a ``reason``) for shapes the nerve does not
    apply to."""
    cod = _as_2d_f32(codes, "codes")
    block_list = None if blocks is None else [int(b) for b in blocks]
    payload = rust_module().atlas_nerve_diagram(
        cod, int(block_size), float(activation_threshold), block_list, int(max_charts)
    )
    if not bool(payload["computed"]):
        return AtlasNerveDiagram(computed=False, reason=str(payload["reason"]))
    betti = payload["betti"]
    return AtlasNerveDiagram(
        computed=True,
        chart_blocks=[int(b) for b in payload["chart_blocks"]],
        betti={
            "b0": int(betti["b0"]),
            "b1": int(betti["b1"]),
            "b2": None if betti["b2"] is None else int(betti["b2"]),
        },
        n_vertices=int(payload["n_vertices"]),
        n_edges=int(payload["n_edges"]),
        n_triangles=int(payload["n_triangles"]),
        n_tetrahedra=int(payload["n_tetrahedra"]),
        sampled_support_size=int(payload["sampled_support_size"]),
        covering_side=str(payload["covering_side"]),
        max_filtration=float(payload["max_filtration"]),
        note=str(payload["note"]),
    )


# --------------------------------------------------------------------------- #
# Coactivation conditionality (influence function + robustness certificate)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ConditionalCoactivationInfluence:
    """Per-row influence contributions for the conditional coactivation
    probability ``P(gate_j active | gate_i active)`` over the selected sample."""

    conditional_probability: float
    active_mass_i: float
    psi: list[float]
    normalized_weights: list[float]


def conditional_coactivation_influence(
    active_i: Any,
    active_j: Any,
    rows: Any,
    likelihood_weights: Any,
) -> ConditionalCoactivationInfluence:
    """Weighted conditional coactivation probability and its per-row influence
    values over the selected ``rows`` with honesty ``likelihood_weights``."""
    payload = rust_module().conditional_coactivation_influence(
        [bool(v) for v in active_i],
        [bool(v) for v in active_j],
        [int(v) for v in rows],
        [float(v) for v in likelihood_weights],
    )
    return ConditionalCoactivationInfluence(
        conditional_probability=float(payload["conditional_probability"]),
        active_mass_i=float(payload["active_mass_i"]),
        psi=[float(v) for v in payload["psi"]],
        normalized_weights=[float(v) for v in payload["normalized_weights"]],
    )


@dataclass(frozen=True)
class CouplingRobustnessCertificate:
    """Weighted-Pearson coupling influence and its KL-robustness certificate.

    ``robustness_radius_epsilon`` is the certified radius ``epsilon*`` and
    ``worst_case_coupling`` the first-order lower bound after a KL-``epsilon``
    distribution shift.
    """

    rho: float
    psi: list[float]
    normalized_weights: list[float]
    influence_variance: float
    influence_mean_abs: float
    robustness_radius_epsilon: float
    epsilon: float
    worst_case_coupling: float


def coupling_robustness_certificate(
    gate_i: Any,
    gate_j: Any,
    rows: Any,
    likelihood_weights: Any,
    epsilon: float = 0.0,
) -> CouplingRobustnessCertificate:
    """Weighted-Pearson coupling ``rho``, its influence function, and the
    KL-robustness certificate (radius ``epsilon*`` + worst-case coupling)."""
    payload = rust_module().coupling_robustness_certificate(
        [float(v) for v in gate_i],
        [float(v) for v in gate_j],
        [int(v) for v in rows],
        [float(v) for v in likelihood_weights],
        float(epsilon),
    )
    return CouplingRobustnessCertificate(
        rho=float(payload["rho"]),
        psi=[float(v) for v in payload["psi"]],
        normalized_weights=[float(v) for v in payload["normalized_weights"]],
        influence_variance=float(payload["influence_variance"]),
        influence_mean_abs=float(payload["influence_mean_abs"]),
        robustness_radius_epsilon=float(payload["robustness_radius_epsilon"]),
        epsilon=float(payload["epsilon"]),
        worst_case_coupling=float(payload["worst_case_coupling"]),
    )


# --------------------------------------------------------------------------- #
# Effect-weighted retention ledger (variance charge OR Fisher local-KL effect)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VarianceChargeEvidence:
    """Reconstruction charge ledger for one atom (all quantities in nats)."""

    delta_deviance: float
    charge: float
    margin: float


@dataclass(frozen=True)
class FisherEffectEvidence:
    """Streaming Fisher local-KL effect ledger for one atom (nats)."""

    atom: int
    mean_fisher_quadratic_kl_nats: float
    max_fisher_quadratic_kl_nats: float
    n_firings: int
    threshold_nats: float
    margin: float


@dataclass(frozen=True)
class AtomRetentionEvidence:
    """Per-atom retention verdict: an OR of the variance and Fisher ledgers."""

    atom: int
    variance: VarianceChargeEvidence | None
    effect: FisherEffectEvidence | None
    retained_by_variance: bool
    retained_by_effect: bool
    retained: bool


def effect_weighted_retention(
    variance: list[tuple[float, float] | None],
    firings: list[tuple[int, float]],
) -> list[AtomRetentionEvidence]:
    """Per-atom effect-weighted retention. ``variance[a]`` is the optional
    ``(delta_deviance_nats, charge_nats)`` charge evidence for atom ``a`` (list
    length = atom count); ``firings`` are streamed ``(atom, fisher_local_kl_nats)``
    contributions that build the Fisher effect ledger. Retention is the OR of the
    variance margin and the Fisher-effect margin (per-atom BIC price)."""
    variance_payload = [
        None if entry is None else (float(entry[0]), float(entry[1]))
        for entry in variance
    ]
    firings_payload = [(int(atom), float(kl)) for atom, kl in firings]
    payload = rust_module().effect_weighted_retention(variance_payload, firings_payload)
    out: list[AtomRetentionEvidence] = []
    for row in payload["atoms"]:
        variance_row = row["variance"]
        effect_row = row["effect"]
        out.append(
            AtomRetentionEvidence(
                atom=int(row["atom"]),
                variance=None
                if variance_row is None
                else VarianceChargeEvidence(
                    delta_deviance=float(variance_row["delta_deviance"]),
                    charge=float(variance_row["charge"]),
                    margin=float(variance_row["margin"]),
                ),
                effect=None
                if effect_row is None
                else FisherEffectEvidence(
                    atom=int(effect_row["atom"]),
                    mean_fisher_quadratic_kl_nats=float(
                        effect_row["mean_fisher_quadratic_kl_nats"]
                    ),
                    max_fisher_quadratic_kl_nats=float(
                        effect_row["max_fisher_quadratic_kl_nats"]
                    ),
                    n_firings=int(effect_row["n_firings"]),
                    threshold_nats=float(effect_row["threshold_nats"]),
                    margin=float(effect_row["margin"]),
                ),
                retained_by_variance=bool(row["retained_by_variance"]),
                retained_by_effect=bool(row["retained_by_effect"]),
                retained=bool(row["retained"]),
            )
        )
    return out
