"""Uniform fixed-distortion description-length scoring for featurizers.

This module is the single implementation of the Eq. 4 scorer shared by the
manifold-zoo benchmark and the #1026 close experiments. Keeping it inside the
package makes the exact scorer available from both a repository checkout and
an installed wheel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ._binding import rust_module


@dataclass
class FittedFeaturizer:
    """Uniform data surface consumed by :func:`description_length`."""

    name: str
    gate: np.ndarray
    atom_contribution: Callable[[int], np.ndarray]
    code_dims: np.ndarray
    dictionary_params: int
    recon: np.ndarray
    fit_seconds: float
    native_bits_per_token: float | None = None
    atom_intrinsic_coords: Callable[[int], np.ndarray] | None = None
    extras: dict[str, Any] | None = None


def description_length(
    fitted: FittedFeaturizer,
    test_x: np.ndarray,
    *,
    r2_targets: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    """Score support, code, residual, and dictionary bits at fixed R-squared.

    Every numeric term (the combinatorial ``lgamma`` support cost, the residual
    covariance eigendecomposition, each atom's SVD coordinate spectrum, and the
    joint firing-weighted reverse-water-filling) lives in the Rust
    ``sae_eq4_description_length`` core; this only coerces the arrays to the
    core's dtypes and adapts ``atom_contribution`` into the row-fetch callback
    the core drives (one atom at a time, so a lazy contribution still only
    materialises the sampled firing rows).
    """

    test_x = np.ascontiguousarray(np.asarray(test_x, dtype=np.float64))
    recon = np.ascontiguousarray(np.asarray(fitted.recon, dtype=np.float64))
    gate = np.ascontiguousarray(np.asarray(fitted.gate, dtype=np.float64))
    code_dims = np.ascontiguousarray(np.asarray(fitted.code_dims))

    def _fetch(atom: int, take: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(
            np.asarray(fitted.atom_contribution(atom)[take], dtype=np.float64)
        )

    native = (
        None
        if fitted.native_bits_per_token is None
        else float(fitted.native_bits_per_token)
    )
    return rust_module().sae_eq4_description_length(
        test_x,
        recon,
        gate,
        code_dims,
        fitted.dictionary_params,
        _fetch,
        r2_targets=(
            None
            if r2_targets is None
            else [float(target) for target in r2_targets]
        ),
        native_bits_per_token=native,
    )


__all__ = ["FittedFeaturizer", "description_length"]
