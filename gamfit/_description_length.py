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
    atom_chart: Callable[[int], dict[str, Any]] | None = None
    extras: dict[str, Any] | None = None


def description_length(
    fitted: FittedFeaturizer,
    test_x: np.ndarray,
    *,
    amortization_horizon: int,
    r2_targets: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    """Score support, code, residual, and dictionary bits at fixed R-squared.

    Every numeric term (the combinatorial support cost, the residual raw
    second-moment eigendecomposition, each atom's full raw contribution spectrum
    split at its ``code_dims``, and the joint firing-weighted
    reverse-water-filling) lives in the Rust ``sae_eq4_description_length`` core;
    this only coerces the arrays to the core's dtypes and adapts
    ``atom_contribution`` into the row-fetch callback the core drives (one atom at
    a time, so a lazy contribution still only materialises the sampled firing
    rows). By default the score is an ambient Gaussian linear-code surrogate:
    modes beyond an atom's ``code_dims`` are residual-coded
    (``truncation_bits_at_r2_*``, inside ``resid_bits_at_r2_*``) and nothing is
    centered, so a reconstruction bias is paid.

    A featurizer that supplies ``atom_chart`` is priced by the intrinsic
    decoder-aware code instead (#3437): ``atom_chart(atom)`` returns
    ``{"code": (n, k), "jacobian": (n, d, k), "axes": [...]}`` over all ``n``
    rows, with ``k = code_dims[atom]``, the decoder Jacobian ``dc/du`` at each
    row's chart coordinates, and one axis per coordinate (``"euclidean"``,
    ``"amplitude"``, or a float period for a circle coordinate). The core then
    codes the chart coordinates under the pulled-back metric ``J^T J`` and
    reports how many atoms it priced this way as ``intrinsic_atoms``.

    ``amortization_horizon`` is the DECLARED ``N`` of the dictionary term, a
    BIC-inspired amortised parameter penalty
    ``0.5 * dictionary_params / N * log2(N)`` on a parameter COUNT (not a decoder
    storage code): the number of tokens the stored parameters are amortised over.
    It is a REQUIRED keyword with no
    default: the number of rows of ``test_x`` is the estimation subsample
    (Monte-Carlo estimator size) and must NEVER be reused as the horizon (#2283 /
    audit §21). Passing them as one number silently made the authoritative
    bits-at-R2 row meaningless, so the two are separated here and the caller must
    state the horizon explicitly; the core rejects a horizon below 2.
    """
    horizon = int(amortization_horizon)
    if horizon < 2:
        raise ValueError(
            "amortization_horizon must be an explicit integer >= 2 (the declared "
            "message/deployment or training-observation N); it is NOT the "
            f"{np.asarray(test_x).shape[0]}-row estimation subsample and is never "
            f"defaulted to it, got {amortization_horizon!r}"
        )

    test_x = np.ascontiguousarray(np.asarray(test_x, dtype=np.float64))
    recon = np.ascontiguousarray(np.asarray(fitted.recon, dtype=np.float64))
    gate = np.ascontiguousarray(np.asarray(fitted.gate, dtype=np.float64))
    code_dims = np.ascontiguousarray(np.asarray(fitted.code_dims))

    def _fetch(atom: int, take: np.ndarray) -> np.ndarray | dict[str, Any]:
        if fitted.atom_chart is not None:
            chart = fitted.atom_chart(atom)
            return {
                "code": np.ascontiguousarray(
                    np.asarray(chart["code"], dtype=np.float64)[take]
                ),
                "jacobian": np.ascontiguousarray(
                    np.asarray(chart["jacobian"], dtype=np.float64)[take]
                ),
                "axes": list(chart["axes"]),
            }
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
        horizon,
        _fetch,
        r2_targets=(
            None
            if r2_targets is None
            else [float(target) for target in r2_targets]
        ),
        native_bits_per_token=native,
    )


__all__ = ["FittedFeaturizer", "description_length"]
