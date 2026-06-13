"""Regression test for issue #1042 on the public Python FFI surface.

``gamfit._api.matern_basis(..., aniso_log_scales=[0, 0])`` — an explicit
all-zero anisotropy log-scale vector, the natural way to ask for the plain
*isotropic* Matérn — must return the isotropic kernel, NOT a data-driven
anisotropic kernel whose contrasts are derived from the spread of the center
cloud. The map from ``aniso_log_scales`` to the design must also be continuous
through the origin: ``[1e-9, -1e-9]`` and ``[0, 0]`` must give neighboring
designs, not a jump discontinuity.
"""

from __future__ import annotations

import numpy as np

from gamfit import _api


def _isotropic_matern_32(pts: np.ndarray, ctr: np.ndarray, ell: float = 1.0) -> np.ndarray:
    """Closed-form isotropic Matérn-3/2, ``r`` = Euclidean distance."""
    out = np.zeros((len(pts), len(ctr)))
    for i, p in enumerate(pts):
        for j, c in enumerate(ctr):
            s = np.sqrt(3.0) * np.linalg.norm(p - c) / ell
            out[i, j] = (1.0 + s) * np.exp(-s)
    return out


def test_explicit_zero_aniso_reduces_to_isotropic_matern() -> None:
    # A center cloud whose two axes have very different spreads — exactly the
    # geometry the discarded override would have turned into an anisotropic
    # metric. Axis 0 spans ~[-20, 20], axis 1 spans ~[-0.5, 0.5].
    rng = np.random.default_rng(4)
    pts = np.column_stack(
        [rng.uniform(-20.0, 20.0, 6), rng.uniform(-0.5, 0.5, 6)]
    )
    ctr = np.column_stack(
        [rng.uniform(-20.0, 20.0, 4), rng.uniform(-0.5, 0.5, 4)]
    )

    ref = _isotropic_matern_32(pts, ctr)
    b_none = np.asarray(_api.matern_basis(pts, ctr, length_scale=1.0, nu="3/2"))
    b_zero = np.asarray(
        _api.matern_basis(
            pts, ctr, length_scale=1.0, nu="3/2", aniso_log_scales=np.array([0.0, 0.0])
        )
    )
    b_eps = np.asarray(
        _api.matern_basis(
            pts,
            ctr,
            length_scale=1.0,
            nu="3/2",
            aniso_log_scales=np.array([1e-9, -1e-9]),
        )
    )

    # The `None` design is the isotropic ground truth; confirm it matches the
    # closed form (sanity check on the reference).
    assert np.max(np.abs(ref - b_none)) < 1e-12

    # A near-zero contrast was always honored verbatim (≈ isotropic).
    assert np.max(np.abs(ref - b_eps)) < 1e-7

    # The fix: explicit [0, 0] must equal the isotropic kernel (the bug here
    # measured a deviation of ~0.13–0.65), and equal the `None` design.
    assert np.max(np.abs(ref - b_zero)) < 1e-12, (
        "explicit aniso_log_scales=[0,0] must reduce to the isotropic Matérn; "
        f"max|diff| = {np.max(np.abs(ref - b_zero)):.3e}"
    )
    assert np.max(np.abs(b_zero - b_none)) < 1e-12


def test_zero_aniso_is_isotropic_across_nu() -> None:
    rng = np.random.default_rng(7)
    pts = np.column_stack(
        [rng.uniform(-15.0, 15.0, 5), rng.uniform(-0.4, 0.4, 5)]
    )
    ctr = np.column_stack(
        [rng.uniform(-15.0, 15.0, 4), rng.uniform(-0.4, 0.4, 4)]
    )
    for nu in ("3/2", "5/2", "7/2", "9/2"):
        b_none = np.asarray(_api.matern_basis(pts, ctr, length_scale=2.0, nu=nu))
        b_zero = np.asarray(
            _api.matern_basis(
                pts, ctr, length_scale=2.0, nu=nu, aniso_log_scales=np.array([0.0, 0.0])
            )
        )
        gap = float(np.max(np.abs(b_zero - b_none)))
        assert gap < 1e-12, f"explicit aniso=[0,0] must be isotropic for nu={nu}; gap={gap:.3e}"
