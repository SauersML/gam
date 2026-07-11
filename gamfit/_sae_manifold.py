"""Thin public marshalling facade for the native manifold-SAE model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ._binding import rust_module
from ._penalty_bridge import GumbelTemperatureSchedule


ManifoldSAE = rust_module().ManifoldSAE


def gumbel_geometric_schedule(
    tau_start: float,
    tau_min: float,
    rate: float,
    iter_count: int = 0,
) -> GumbelTemperatureSchedule:
    """Describe native geometric assignment-temperature annealing."""
    return GumbelTemperatureSchedule(
        tau_start,
        tau_min,
        "geometric",
        rate=rate,
        iter_count=iter_count,
    )


def gumbel_linear_schedule(
    tau_start: float,
    tau_min: float,
    steps: int,
    iter_count: int = 0,
) -> GumbelTemperatureSchedule:
    """Describe native linear assignment-temperature annealing."""
    return GumbelTemperatureSchedule(
        tau_start,
        tau_min,
        "linear",
        steps=steps,
        iter_count=iter_count,
    )


def gumbel_reciprocal_iter_schedule(
    tau_start: float,
    tau_min: float,
    iter_count: int = 0,
) -> GumbelTemperatureSchedule:
    """Describe native reciprocal-in-iteration temperature annealing."""
    return GumbelTemperatureSchedule(
        tau_start,
        tau_min,
        "reciprocal_iter",
        iter_count=iter_count,
    )


def _matrix(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return np.ascontiguousarray(array)


def _optional_array(value: Any, *, dimensions: int) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != dimensions:
        raise ValueError(f"expected a {dimensions}-dimensional array; got {array.shape}")
    return np.ascontiguousarray(array)


def _atom_dimensions(value: Any) -> list[int] | None:
    if value is None or value == "auto":
        return None
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    if isinstance(value, str):
        raise TypeError("d_atom must be an integer, a sequence of integers, or 'auto'")
    return [int(dimension) for dimension in value]


def _atom_bases(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [str(basis) for basis in value]


def _schedule_descriptor(
    schedule: GumbelTemperatureSchedule | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if schedule is None:
        return None
    if isinstance(schedule, GumbelTemperatureSchedule):
        return schedule.to_rust_descriptor()
    return dict(schedule)


def _fisher_arrays(
    value: Any,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    if value is None:
        return None, None, None
    if isinstance(value, Mapping):
        factors = value["U"]
        residual = value.get("mass_residual")
        provenance = value.get("provenance", "output_fisher")
    elif hasattr(value, "U"):
        factors = value.U
        residual = getattr(value, "mass_residual", None)
        provenance = getattr(value, "provenance", "output_fisher")
    else:
        factors = value
        residual = None
        provenance = "output_fisher"
    return (
        np.ascontiguousarray(np.asarray(factors, dtype=np.float64)),
        None
        if residual is None
        else np.ascontiguousarray(np.asarray(residual, dtype=np.float64)),
        str(provenance),
    )


def sae_manifold_fit(
    X: Any,
    K: int,
    d_atom: Any = 2,
    atom_topology: str | None = None,
    assignment: str = "softmax",
    schedule: GumbelTemperatureSchedule | Mapping[str, Any] | None = None,
    isometry_weight: float = 1.0,
    ard_per_atom: bool = True,
    decoder_feature_sparsity_groups: Sequence[Sequence[int]] | None = None,
    n_iter: int = 50,
    *,
    sparsity_weight: float = 1.0,
    coord_sparsity: str = "scad",
    scad_mcp_gamma: float | None = None,
    smoothness_weight: float = 1.0,
    alpha: float | str | None = None,
    learning_rate: float | None = None,
    random_state: int = 0,
    block_orthogonality_weight: float = 0.0,
    nuclear_norm_weight: float = 1.0,
    nuclear_norm_max_rank: int | None = None,
    decoder_incoherence_weight: float = 1.0,
    top_k: int | None = None,
    t_init: Any = None,
    a_init: Any = None,
    tau: float | None = None,
    threshold_gate_threshold: float = 0.0,
    atom_basis: Any = None,
    fisher_factors: Any = None,
    weights: Any = None,
    separation_barrier_strength: float | None = None,
    ordered_beta_bernoulli_alpha: float | None = None,
    promote_from_residual: bool = True,
) -> ManifoldSAE:
    """Fit and return the immutable Rust-owned manifold-SAE model.

    Python only converts user containers to contiguous arrays and literal
    descriptors. Basis/assignment resolution, defaults, validation, penalties,
    optimization, hyperparameter selection, diagnostics, and artifact assembly
    are owned by the native front door.
    """
    x = _matrix(X)
    fisher, fisher_residual, fisher_provenance = _fisher_arrays(fisher_factors)
    if alpha == "auto":
        alpha_value = None
        learnable_alpha = True
    elif isinstance(alpha, str):
        raise TypeError("alpha must be a number, None, or 'auto'")
    else:
        alpha_value = None if alpha is None else float(alpha)
        learnable_alpha = False
    groups = (
        None
        if decoder_feature_sparsity_groups is None
        else [[int(feature) for feature in group] for group in decoder_feature_sparsity_groups]
    )
    return rust_module().sae_manifold_fit_model(
        x,
        int(K),
        _atom_dimensions(d_atom),
        atom_topology=None if atom_topology is None else str(atom_topology),
        atom_basis=_atom_bases(atom_basis),
        assignment_kind=str(assignment),
        gumbel_schedule=_schedule_descriptor(schedule),
        isometry_weight=float(isometry_weight),
        native_ard_enabled=bool(ard_per_atom),
        decoder_feature_sparsity_groups=groups,
        max_iter=int(n_iter),
        sparsity_strength=float(sparsity_weight),
        coord_sparsity=str(coord_sparsity),
        scad_mcp_gamma=None if scad_mcp_gamma is None else float(scad_mcp_gamma),
        smoothness=float(smoothness_weight),
        alpha=alpha_value,
        learnable_alpha=learnable_alpha,
        learning_rate=None if learning_rate is None else float(learning_rate),
        random_state=int(random_state),
        block_orthogonality_weight=float(block_orthogonality_weight),
        nuclear_norm_weight=float(nuclear_norm_weight),
        nuclear_norm_max_rank=(
            None if nuclear_norm_max_rank is None else int(nuclear_norm_max_rank)
        ),
        decoder_incoherence_weight=float(decoder_incoherence_weight),
        top_k=None if top_k is None else int(top_k),
        initial_coords=_optional_array(t_init, dimensions=3),
        initial_logits=_optional_array(a_init, dimensions=2),
        tau=None if tau is None else float(tau),
        threshold_gate_threshold=float(threshold_gate_threshold),
        fisher_factors=fisher,
        fisher_mass_residual=fisher_residual,
        fisher_provenance=fisher_provenance,
        row_loss_weights=_optional_array(weights, dimensions=1),
        separation_barrier_strength_override=(
            None
            if separation_barrier_strength is None
            else float(separation_barrier_strength)
        ),
        ordered_beta_bernoulli_alpha_override=(
            None
            if ordered_beta_bernoulli_alpha is None
            else float(ordered_beta_bernoulli_alpha)
        ),
        promote_from_residual=bool(promote_from_residual),
        run_structure_search=True,
        run_outer_rho_search=True,
    )


def flat_block_assignment(gating: str) -> str:
    """Return the native assignment family for a linear-block gate."""
    return str(rust_module().sae_flat_block_assignment(str(gating)))


def plot(atom: Any, **kwargs: Any) -> Any:
    """Plot SAE atoms through the visualization-only Python helper."""
    from . import _sae_viz

    return _sae_viz.plot(atom, **kwargs)


__all__ = [
    "GumbelTemperatureSchedule",
    "ManifoldSAE",
    "flat_block_assignment",
    "gumbel_geometric_schedule",
    "gumbel_linear_schedule",
    "gumbel_reciprocal_iter_schedule",
    "plot",
    "sae_manifold_fit",
]
