"""Pure-Python dispatch helpers for :mod:`gamfit.torch.fit`.

This module intentionally has no torch dependency. Keep tensor construction,
torch operations, tensor shape reads, and autograd-bearing work in ``fit.py``.
"""

from __future__ import annotations

from typing import Literal, Sequence

from ..smooth import Smooth

FitMode = Literal["joint", "independent", "auto"]

# F-threshold below which mode="auto" uses exact dense joint additive REML
# and above which it falls back to the shared-scale block-orthogonal
# additive estimator. Joint scales as O((F * M_k)^3) for the inner
# Cholesky; the block path scales as O(F * M_k^3). At F approx 64 with
# typical M_k approx 8-16, joint is still comfortable; at F approx 1024+
# it becomes infeasible.
_AUTO_MODE_F_THRESHOLD = 64


def validate_2d_shape(name: str, ndim: int, shape: tuple[int, ...]) -> None:
    """Validate a pre-read tensor shape without touching the tensor itself."""
    if ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got {ndim}D shape {shape}")


def shape_kind_of(smooth: Smooth) -> str | None:
    """Return normalized shape-constraint kind, or ``None`` if unconstrained."""
    shape_constraint = getattr(smooth, "shape_constraint", None)
    if shape_constraint is None:
        return None
    shape_constraint_str = str(shape_constraint).lower()
    return None if shape_constraint_str == "none" else shape_constraint_str


def shape_kind_for_smooths_arg(smooths: Smooth | Sequence[Smooth]) -> str | None:
    """Validate shape-constraint dispatch and return the single-smooth kind."""
    if isinstance(smooths, Smooth):
        return shape_kind_of(smooths)

    kinds = [shape_kind_of(smooth) for smooth in smooths if isinstance(smooth, Smooth)]
    if any(kind is not None for kind in kinds):
        raise NotImplementedError(
            "shape_constraint on the torch fit path is currently only "
            "supported for a single Smooth (not a list). For joint "
            "multi-smooth additive fits with shape constraints use "
            "gamfit.fit(df, formula, constraints={...})."
        )
    return None


def validate_smooths_arg(smooths: Sequence[Smooth]) -> list[Smooth]:
    """Normalize a multi-smooth argument and validate element types."""
    smooths_list = list(smooths)
    if len(smooths_list) == 0:
        raise ValueError("smooths must contain at least one Smooth")
    if not all(isinstance(smooth, Smooth) for smooth in smooths_list):
        bad = [
            type(smooth).__name__
            for smooth in smooths_list
            if not isinstance(smooth, Smooth)
        ]
        raise TypeError(f"all entries must be Smooth, got: {bad}")
    return smooths_list


def validate_points_list_length(points_len: int, smooths_len: int) -> None:
    """Validate that a per-smooth points list matches the smooth count."""
    if points_len != smooths_len:
        raise ValueError(f"got {points_len} points tensors but {smooths_len} smooths")


def resolve_fit_mode(mode: FitMode, F: int, D: int) -> FitMode:
    """Resolve ``auto`` and reject joint multi-output dispatch."""
    if mode == "auto":
        effective_mode: FitMode = (
            "joint" if (F <= _AUTO_MODE_F_THRESHOLD and D == 1) else "independent"
        )
    else:
        effective_mode = mode
    if effective_mode == "joint" and D > 1:
        raise NotImplementedError(
            "mode='joint' currently requires single-output response (D=1); "
            f"got D={D}. Use mode='independent' or 'auto' for multi-output."
        )
    return effective_mode
