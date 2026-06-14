"""Example namespace: high-level runners for latent-factor workflows.

Each example in this submodule returns a callable runner with a ``.fit(...)``
entry point or a direct fit result. Examples are thin: they encode one
coordinated procedure on top of existing Smooth / penalty / SAE machinery so
that the call site reads as one declarative step.

``partial_supervision`` returns a gauge-fix example for splitting supervised and
free latent blocks. ``sae_supervised`` fits a manifold SAE on all rows, then a
supervised GAM/GLM head on the rows selected by ``supervised_mask``.
"""

from __future__ import annotations

from .partial_supervision import (
    PartialSupervisionExample,
    PartialSupervisionFit,
    partial_supervision,
)
from .sae_supervised import (
    SaeSupervisedFit,
    sae_supervised,
)

__all__ = [
    "PartialSupervisionExample",
    "PartialSupervisionFit",
    "partial_supervision",
    "SaeSupervisedFit",
    "sae_supervised",
]
