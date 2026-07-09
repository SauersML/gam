"""Uniform fixed-distortion description-length scoring for featurizers.

This module is the single implementation of the Eq. 4 scorer shared by the
manifold-zoo benchmark and the #1026 close experiments. Keeping it inside the
package makes the exact scorer available from both a repository checkout and
an installed wheel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


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


def _water_fill_bits(spectrum: np.ndarray, delta: float) -> float:
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError(f"distortion must be finite and positive, got {delta}")
    lam = np.maximum(np.asarray(spectrum, dtype=float), 0.0)
    return float(np.sum(0.5 * np.log2(1.0 + lam / delta)))


def _covariance_eigenvalues(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(values.shape[0] - 1, 1)
    return np.linalg.eigvalsh(covariance)


def description_length(
    fitted: FittedFeaturizer,
    test_x: np.ndarray,
    *,
    r2_targets: tuple[float, ...] = (0.99, 0.95, 0.90, 0.80),
) -> dict[str, Any]:
    """Score support, code, residual, and dictionary bits at fixed R-squared."""

    test_x = np.asarray(test_x, dtype=float)
    if test_x.ndim != 2:
        raise ValueError(f"test_x must be a matrix, got shape {test_x.shape}")
    if test_x.shape != fitted.recon.shape:
        raise ValueError(
            f"test_x and recon must have the same shape, got {test_x.shape} "
            f"and {fitted.recon.shape}"
        )
    if test_x.shape[0] == 0 or test_x.shape[1] == 0:
        raise ValueError("test_x must contain at least one row and one column")
    if not np.all(np.isfinite(test_x)) or not np.all(np.isfinite(fitted.recon)):
        raise ValueError("test_x and recon must contain only finite values")
    if not np.all(np.isfinite(fitted.gate)):
        raise ValueError("gate must contain only finite values")
    if not r2_targets:
        raise ValueError("r2_targets must not be empty")
    if any(
        not math.isfinite(target) or not 0.0 <= target < 1.0
        for target in r2_targets
    ):
        raise ValueError("every R-squared target must be finite and in [0, 1)")

    n, d = test_x.shape
    gate_active = fitted.gate > 1e-10
    p_g = gate_active.mean(axis=0)
    l0 = float(gate_active.sum(axis=1).mean())
    n_atoms = fitted.gate.shape[1]
    support_cardinality = min(max(round(l0), 0), n_atoms)
    support_bits = (
        math.lgamma(n_atoms + 1)
        - math.lgamma(support_cardinality + 1)
        - math.lgamma(n_atoms - support_cardinality + 1)
    ) / math.log(2.0)

    residual = test_x - fitted.recon
    residual_covariance_eigenvalues = _covariance_eigenvalues(residual)
    centered_x = test_x - test_x.mean(axis=0, keepdims=True)
    reference_variance = float(np.mean(centered_x * centered_x) * d)
    if reference_variance <= 0.0:
        raise ValueError("test_x must have positive variance")

    code_spectra: list[np.ndarray] = []
    for atom in range(n_atoms):
        code_dim = int(fitted.code_dims[atom])
        rows = np.flatnonzero(gate_active[:, atom])
        if rows.size < max(code_dim + 1, 4):
            code_spectra.append(np.zeros(code_dim))
            continue
        take = rows if rows.size <= 4096 else rows[:: max(rows.size // 4096, 1)]
        contribution = np.asarray(fitted.atom_contribution(atom)[take], dtype=float)
        if contribution.shape != (take.size, d):
            raise ValueError(
                f"atom {atom} contribution has shape {contribution.shape}; "
                f"expected {(take.size, d)}"
            )
        if not np.all(np.isfinite(contribution)):
            raise ValueError(f"atom {atom} contribution contains non-finite values")
        singular_values = np.linalg.svd(
            contribution - contribution.mean(axis=0, keepdims=True),
            compute_uv=False,
        )
        code_spectra.append(
            singular_values[:code_dim] ** 2 / max(take.size - 1, 1)
        )

    out: dict[str, Any] = {
        "support_bits": float(support_bits),
        "achieved_block_l0": l0,
    }
    for target in r2_targets:
        distortion = (1.0 - target) * reference_variance / d
        code_bits = float(
            sum(
                probability * _water_fill_bits(spectrum, distortion)
                for probability, spectrum in zip(p_g, code_spectra, strict=True)
            )
        )
        residual_bits = _water_fill_bits(residual_covariance_eigenvalues, distortion)
        dictionary_bits = 0.5 * fitted.dictionary_params / n * math.log2(max(n, 2))
        suffix = f"{target:g}"
        out[f"bits_at_r2_{suffix}"] = (
            support_bits + code_bits + residual_bits + dictionary_bits
        )
        out[f"code_bits_at_r2_{suffix}"] = code_bits
        out[f"resid_bits_at_r2_{suffix}"] = residual_bits
    if fitted.native_bits_per_token is not None:
        out["native_bits_per_token"] = fitted.native_bits_per_token
    return out


__all__ = ["FittedFeaturizer", "description_length"]
