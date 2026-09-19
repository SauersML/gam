"""Inference utilities built on fitted models: Bartlett corrections, full
conformal prediction, and shared-precision empirical Bayes across fits."""

from __future__ import annotations

from ._api import (
    SharedPrecisionGroup,
    cross_fit_shared_precision_groups,
)
from ._bartlett import (
    lawley_bartlett_factor,
    lawley_bartlett_factor_estimated_lambda,
)
from ._full_conformal import (
    glm_full_conformal,
)

__all__ = [
    "cross_fit_shared_precision_groups",
    "glm_full_conformal",
    "lawley_bartlett_factor",
    "lawley_bartlett_factor_estimated_lambda",
    "SharedPrecisionGroup",
]
