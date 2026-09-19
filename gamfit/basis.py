"""Differentiable basis and penalty-matrix primitives, plus the basis protocol.

The array functions (``bspline_basis``, ``duchon_basis``, ...) evaluate the
same Rust basis builders the formula DSL uses. :class:`BasisDescriptor` is the
protocol the compositional :class:`Smooth` and :class:`PeriodicHarmonic`
implement.
"""

from __future__ import annotations

from ._api import (
    bspline_basis,
    bspline_basis_derivative,
    duchon_basis,
    duchon_function_norm_penalty,
    matern_basis,
    periodic_spline_curve_basis,
    smoothness_penalty,
    sphere_basis,
    sphere_basis_jet,
)
from ._protocol import (
    BasisDescriptor,
)
from ._basis_descriptors import (
    PeriodicHarmonic,
)
from ._smooth import (
    Smooth,
    SmoothSum,
)

__all__ = [
    "BasisDescriptor",
    "bspline_basis",
    "bspline_basis_derivative",
    "duchon_basis",
    "duchon_function_norm_penalty",
    "matern_basis",
    "periodic_spline_curve_basis",
    "PeriodicHarmonic",
    "Smooth",
    "smoothness_penalty",
    "SmoothSum",
    "sphere_basis",
    "sphere_basis_jet",
]
