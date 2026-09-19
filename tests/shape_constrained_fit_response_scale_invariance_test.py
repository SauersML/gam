"""Shape-constrained fits converge at every response scale (pyGAM audit A4).

A monotone smooth on 300 rows fit fine with ``y`` as given and with ``y``
scaled by ``1e-6``, but refused to fit with ``y`` scaled by ``1e6``:

    KKT residuals exceed tolerance: primal=5.821e-11, comp=9.650e-5, ...
    ‖grad‖∞=1.519e6

The outer startup gate compared the gradient-unit KKT channels (dual
feasibility, complementarity) against an absolute ``1e-7``. Those residuals
carry the gradient's units, so rescaling the response rescales them while the
constrained minimizer's geometry stays put: a vertex the inner active-set
solver had certified at ``6e-11`` of its gradient was refused as a violation.
Both the solver and the gate now judge every gradient-unit channel at the
gradient's own scale, through one shared predicate.

The fitted curve is equivariant under ``y -> c·y``; this test asserts that the
three scales all converge and agree with each other once divided back out.
"""

from __future__ import annotations

import numpy as np

import gamfit


def _monotone_data(n: int = 300) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    x = np.sort(rng.uniform(0.0, 10.0, n))
    y = np.log1p(x) + rng.normal(0.0, 0.4, n)
    return x, y


def test_monotone_fit_is_equivariant_under_response_rescaling() -> None:
    x, y = _monotone_data()
    grid = np.linspace(0.0, 10.0, 60)
    curves = {}
    for scale in (1.0, 1.0e-6, 1.0e6):
        model = gamfit.fit(
            {"x": x, "y": scale * y}, "y ~ s(x, shape=monotone_increasing)"
        )
        fitted = np.asarray(model.predict({"x": grid}), dtype=float) / scale
        assert np.all(np.isfinite(fitted)), f"scale {scale:g}: non-finite fit"
        assert np.all(np.diff(fitted) >= -1e-9 * np.ptp(fitted)), (
            f"scale {scale:g}: the fit violates the requested monotonicity"
        )
        curves[scale] = fitted
    reference = curves[1.0]
    spread = np.ptp(reference)
    for scale in (1.0e-6, 1.0e6):
        gap = np.max(np.abs(curves[scale] - reference))
        assert gap <= 1e-3 * spread, (
            f"scale {scale:g}: rescaled fit differs from the unit-scale fit by "
            f"{gap:.3e} (curve range {spread:.3e})"
        )
