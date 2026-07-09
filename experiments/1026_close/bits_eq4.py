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

from gamfit._description_length import FittedFeaturizer, description_length

__all__ = ["FittedFeaturizer", "description_length", "scorer_source"]


def scorer_source() -> str:
    """Return the canonical scorer provenance recorded with results."""
    return "gamfit._description_length"
