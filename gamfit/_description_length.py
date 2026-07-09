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

    def __post_init__(self) -> None:
        self.gate = np.asarray(self.gate)
        self.code_dims = np.asarray(self.code_dims)
        self.recon = np.asarray(self.recon)
        if self.gate.ndim != 2:
            raise ValueError(f"gate must be a matrix, got shape {self.gate.shape}")
        if self.recon.ndim != 2:
            raise ValueError(f"recon must be a matrix, got shape {self.recon.shape}")
        if self.recon.shape[0] != self.gate.shape[0]:
            raise ValueError(
                "gate and recon must contain the same number of rows, got "
                f"{self.gate.shape[0]} and {self.recon.shape[0]}"
            )
        if self.code_dims.shape != (self.gate.shape[1],):
            raise ValueError(
                "code_dims must have one entry per atom, got "
                f"shape {self.code_dims.shape} for {self.gate.shape[1]} atoms"
            )
        if not np.issubdtype(self.code_dims.dtype, np.integer):
            raise TypeError("code_dims must contain integers")
        if np.any(self.code_dims < 0):
            raise ValueError("code_dims must be nonnegative")
        if self.dictionary_params < 0:
            raise ValueError("dictionary_params must be nonnegative")


def _water_fill_component_bits(
    components: list[tuple[float, np.ndarray]], total_distortion: float
) -> list[float]:
    """Allocate total distortion across weighted Gaussian spectra (Rust core)."""

    coerced = [
        (float(weight), np.asarray(spectrum, dtype=np.float64).ravel().tolist())
        for weight, spectrum in components
    ]
    return list(
        rust_module().sae_eq4_water_fill_component_bits(coerced, float(total_distortion))
    )


def description_length(
    fitted: FittedFeaturizer,
    test_x: np.ndarray,
    *,
    r2_targets: tuple[float, ...] = (0.99, 0.95, 0.90, 0.80),
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
    code_dims = np.ascontiguousarray(np.asarray(fitted.code_dims, dtype=np.int64))

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
        int(fitted.dictionary_params),
        _fetch,
        r2_targets=[float(target) for target in r2_targets],
        native_bits_per_token=native,
    )


__all__ = ["FittedFeaturizer", "description_length"]
