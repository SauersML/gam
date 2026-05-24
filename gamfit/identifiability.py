from __future__ import annotations

from typing import Any

from ._binding import rust_module
from ._compare import compare_models
from ._exceptions import map_exception
from ._penalties import AnalyticPenaltyKind

__all__ = [
    "AnalyticPenaltyKind",
    "compare_models",
    "conditional_prior_ivae",
    "mechanism_sparsity_jacobian",
]


def mechanism_sparsity_jacobian(weight: float, epsilon: float, w: Any) -> tuple[float, Any]:
    """Evaluate the Rust mechanism-sparsity Jacobian penalty and gradient."""

    try:
        return rust_module().mechanism_sparsity_jacobian(float(weight), float(epsilon), w)
    except Exception as exc:
        raise map_exception(exc) from exc


def conditional_prior_ivae(
    weight: float,
    t: Any,
    mean: Any,
    scale: Any,
) -> tuple[float, Any]:
    """Evaluate the Rust iVAE conditional-prior penalty and gradient."""

    try:
        return rust_module().conditional_prior_ivae(float(weight), t, mean, scale)
    except Exception as exc:
        raise map_exception(exc) from exc
