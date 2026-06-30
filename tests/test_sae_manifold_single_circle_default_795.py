"""Regression for issue #795: the single-planted-circle ``sae_manifold_fit``
quickstart must converge with the shipped default regularizer settings.

#795 was a `RemlConvergenceError` on the simplest possible manifold-SAE fit —
one planted circle, K=1, d=1 — caused by the old absolute-speed isometry
penalty: its energy scaled as ``decoder⁴``, so during the joint solve it
exploded and saturated the arrow-Schur proximal ridge at 1e15, rejecting every
trial step. The fix has two parts: (a) the pin normalizes ``g = JᵀJ`` by mean
trace before comparing to identity AND folds the frozen normalizer ``1 / gbar²``
into the assembled Gauss-Newton curvature, so the value, gradient, AND curvature
are all decoder-scale-invariant (the ``‖B‖⁴`` Gram block cancels the ``‖B‖⁻⁴``
normalizer); and (b) the outer-REML row-gauge evidence-deflation guard no longer
charges its reversal budget on a BOUNDED low-amplitude flicker of the per-row
near-null count (the circle fit flickers 150↔147 on N=200), only on a
wide-amplitude runaway. With both fixes the gauge is enabled by default
(``isometry_weight=1.0``).

The existing public-API tests fit this same geometry but pass
``isometry_weight=0.0`` *explicitly*, so they would survive a future change to
the default while the documented quickstart silently broke again. This test
deliberately constructs the fit **without** naming
``isometry_weight`` — exactly the documented quickstart pattern — so it pins the
default itself. The second test pins the default-on acceptance the issue asked
for: convergence, an honest near-2π chart span, and a rotation/isometry residual
gauge rather than Diff escalation.
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


def _chart_span_radians(coords: np.ndarray) -> float:
    span = float(np.ptp(np.asarray(coords, dtype=float).reshape(-1)))
    # The public periodic/circle basis reports normalized phase coordinates.
    return span * (2.0 * np.pi) if span <= 1.5 else span


def test_single_circle_quickstart_converges_with_default_regularizers() -> None:
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


# #795 — the default-on acceptance: a positive isometry pin must recover an
# honest full-circle chart span and report the rotation/isometry residual gauge.
# This previously stood FAILING (the curvature walk bifurcated and the proximal
# ridge saturated); it now passes once the assembled GN curvature carries the
# `1/gbar²` scale-invariance fold and the deflation guard tolerates bounded
# flicker.
def test_single_circle_positive_isometry_recovers_honest_chart_span() -> None:
    z = _planted_circle()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit = gamfit.sae_manifold_fit(
            X=z,
            K=1,
            d_atom=1,
            atom_topology="circle",
            isometry_weight=0.05,
            n_iter=20,
            random_state=0,
        )
    span = _chart_span_radians(fit.coords[0])
    assert abs(span - 2.0 * np.pi) <= 0.10 * (2.0 * np.pi), (
        "positive isometry pin must recover an honest full-circle chart span; "
        f"got {span:.4f} radians"
    )
    gauge = fit.residual_gauge
    assert gauge is not None, "fit must report a residual-gauge certificate"
    signature = str(gauge.get("group_signature", ""))
    assert "Diff(" not in signature and "diffeomorphism-unpinned" not in signature, (
        "isometry-on fit must report the rotation/isometry residual group, not "
        f"Diff escalation; signature={signature!r}"
    )
    assert "Isom(M_k)" in signature, (
        "single-circle residual gauge should be the circle isometry group; "
        f"signature={signature!r}"
    )
