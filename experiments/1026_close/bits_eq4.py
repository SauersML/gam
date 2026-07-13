#!/usr/bin/env python3
"""Eq-4 description-length bits scorer for the #1026 close (gam#2233).

gam#2233 proves the #1026 hybrid's thin EV-at-matched-actives margin is a
scoreboard artefact: that metric only partially credits the support and
residual savings won by a curved atom. The MDL scoreboard (bits at fixed R²,
their Eq. 4) credits support, code, and residual savings. This module exposes
the package's canonical scorer to the #1026 experiment drivers, ensuring a
repository run and a wheel-only deployment execute identical math.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gamfit._description_length import FittedFeaturizer

__all__ = ["make_fitted_featurizer", "description_length", "scorer_source"]


def make_fitted_featurizer(**fields: Any) -> FittedFeaturizer:
    """Construct the canonical package scoring surface only when it is needed."""
    from gamfit._description_length import FittedFeaturizer

    return FittedFeaturizer(**fields)


def description_length(
    fitted: FittedFeaturizer,
    test_x: Any,
    *,
    r2_targets: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    """Load the native-backed Eq. 4 scorer at the actual scoring boundary."""
    from gamfit._description_length import description_length as score

    return score(fitted, test_x, r2_targets=r2_targets)


def scorer_source() -> str:
    """Return the canonical scorer provenance recorded with results."""
    return "gamfit._description_length"
