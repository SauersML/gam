"""Regression for issue #795: the single-planted-circle ``sae_manifold_fit``
quickstart must converge with the **default** regularizer settings.

#795 was a `RemlConvergenceError` on the simplest possible manifold-SAE fit —
one planted circle, K=1, d=1 — caused by the (then-default) MeanProfiled
isometry penalty: its energy is not scale-invariant (it scales as ``decoder⁴``),
so during the joint solve it exploded and saturated the arrow-Schur proximal
ridge at 1e15, rejecting every trial step. The fix flipped the default
``isometry_weight`` from ``1.0`` to ``0.0``.

The existing public-API tests fit this same geometry but pass
``isometry_weight=0.0`` *explicitly*, so they would survive a future re-flip of
the default back to a non-zero value while the documented quickstart silently
broke again. This test deliberately constructs the fit **without** naming
``isometry_weight`` — exactly the documented quickstart pattern — so it pins the
default itself. Re-enabling the isometry penalty by default without first
scale-normalizing its residual will fail here.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")


def _planted_circle(n: int = 200, d_ambient: int = 12, noise: float = 0.02, seed: int = 0):
    """The exact minimal repro from issue #795: a 1-D circle linearly embedded
    in ``d_ambient`` dimensions with light isotropic noise."""
    rng = np.random.default_rng(seed)
    basis = rng.standard_normal((2, d_ambient))
    basis /= np.linalg.norm(basis, axis=1, keepdims=True)
    t = rng.uniform(0.0, 2.0 * np.pi, n)
    clean = np.column_stack([np.cos(t), np.sin(t)]) @ basis
    return clean + noise * rng.standard_normal((n, d_ambient))


def test_single_circle_quickstart_converges_with_default_isometry() -> None:
    z = _planted_circle()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            # No `isometry_weight=` here on purpose — exercise the shipped default.
            fit = gamfit.sae_manifold_fit(
                X=z,
                K=1,
                d_atom=1,
                atom_topology="circle",
                n_iter=20,
                random_state=0,
            )
    except Exception as exc:  # pragma: no cover - the regression is an exception
        pytest.fail(
            "single planted circle must converge with default settings (#795); "
            f"got {type(exc).__name__}: {exc}"
        )
    # A converged fit must produce a finite reconstruction of the data.
    assert np.all(np.isfinite(fit.fitted)), "fitted reconstruction has non-finite entries"
    assert np.isfinite(fit.reconstruction_r2), "reconstruction R² is non-finite"
