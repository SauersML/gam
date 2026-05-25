"""Recipe namespace: high-level runners that assemble latent-factor gauge fixes.

Each recipe in this submodule returns a callable runner with a ``.fit(...)``
entry point. Recipes are thin: they encode a single coordinated procedure
(supervision plus identifiability) on top of the existing Smooth / penalty
machinery so that the call site reads as one declarative step.
"""

from __future__ import annotations

from .partial_supervision import (
    PartialSupervisionFit,
    PartialSupervisionRecipe,
    partial_supervision,
)

__all__ = [
    "PartialSupervisionFit",
    "PartialSupervisionRecipe",
    "partial_supervision",
]
