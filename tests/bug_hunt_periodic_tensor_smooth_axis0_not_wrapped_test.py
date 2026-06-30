"""Bug hunt: a periodic tensor-product smooth must wrap on *every* axis it
declares periodic — but the first tensor axis (axis 0) is not wrapped.

``te(x, z, periodic=c(1,1), period=c(1,1))`` requests a tensor-product smooth
that is periodic with period 1 on *both* margins. A function that is periodic
on axis ``a`` with period ``T`` satisfies, by definition,

    f(..., u_a + T, ...) == f(..., u_a, ...)      for every point.

So the fitted surface ``ŝ(x, z)`` must be invariant to shifting *either*
coordinate by a full period: ``ŝ(x+1, z) == ŝ(x, z)`` and
``ŝ(x, z+1) == ŝ(x, z)``, to machine precision (the closed-curve seam is
exact for a correctly assembled cyclic basis — see the 1-D
``gamfit.periodic_spline_curve_basis`` whose rows at ``t`` and ``t+1`` agree to
~1e-16, and the 1-D ``s(x, bs="cc")`` smooth whose endpoints coincide exactly).

Observed: the **second** margin (axis 1, here ``z``) wraps perfectly
(``|ŝ(x, z+1) − ŝ(x, z)| ~ 3e-16``), but the **first** margin (axis 0, here
``x``) does not wrap at all: ``|ŝ(x+1, z) − ŝ(x, z)| ~ 4.3`` — larger than the
amplitude of the fitted signal. The defect follows the *axis position*, not the
variable: swapping to ``te(z, x, ...)`` makes ``z`` (now axis 0) fail and ``x``
(now axis 1) wrap exactly. This points at the tensor margin evaluation /
periodic-metadata propagation skipping the periodic fold on the leading
Kronecker factor (crates/gam-terms/src/smooth/design_freezing.rs and the margin
periodic-wrap path in crates/gam-terms/src/basis/bspline_eval.rs around the
`wrap_to_period` call, lines ~1196-1205; per-axis period assignment in
crates/gam-terms/src/term_builder.rs `parse_tensor_periodic_axes`).

Expected: both axes wrap to near machine precision, as axis 1 already does.

The test is deterministic and seeded; it first asserts the fitted surface is
non-trivial (so there is a real periodic surface to wrap) and that axis 1 wraps
(the working reference), then asserts axis 0 wraps to the same tolerance.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")
pd: Any = pytest.importorskip("pandas")

import gamfit


def _fit_periodic_tensor(seed: int):
    rng = np.random.default_rng(seed)
    n = 2000
    x = rng.uniform(0.0, 1.0, n)
    z = rng.uniform(0.0, 1.0, n)
    # Doubly-periodic truth on [0,1)^2.
    y = np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * z) + rng.normal(0.0, 0.05, n)
    df = pd.DataFrame({"x": x, "z": z, "y": y})
    return gamfit.fit(
        df, "y ~ te(x, z, periodic=c(1,1), period=c(1,1))", family="gaussian"
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_periodic_tensor_wraps_on_both_axes(seed: int) -> None:
    model = _fit_periodic_tensor(seed)

    base = pd.DataFrame({"x": [0.2, 0.5, 0.7, 0.9], "z": [0.3, 0.6, 0.1, 0.8]})
    wrap_axis0 = pd.DataFrame(
        {"x": [1.2, 1.5, 1.7, 1.9], "z": [0.3, 0.6, 0.1, 0.8]}  # x + period
    )
    wrap_axis1 = pd.DataFrame(
        {"x": [0.2, 0.5, 0.7, 0.9], "z": [1.3, 1.6, 1.1, 1.8]}  # z + period
    )

    p0 = np.asarray(model.predict(base)).ravel()
    d_axis0 = float(np.max(np.abs(np.asarray(model.predict(wrap_axis0)).ravel() - p0)))
    d_axis1 = float(np.max(np.abs(np.asarray(model.predict(wrap_axis1)).ravel() - p0)))

    # The fitted surface must be non-trivial, otherwise wrapping is vacuous.
    assert float(np.ptp(p0)) > 0.1, (
        f"seed {seed}: fitted periodic surface is flat (ptp={np.ptp(p0):.3e}); "
        "the wrap test would be vacuous"
    )

    # Reference: axis 1 (the second margin) already wraps to machine precision.
    assert d_axis1 < 1e-8, (
        f"seed {seed}: axis-1 wrap unexpectedly broken (|s(x,z+1)-s(x,z)|="
        f"{d_axis1:.3e}); the periodic-tensor baseline assumption is wrong"
    )

    # The bug: axis 0 (the first margin) does not wrap (observed ~4.3).
    assert d_axis0 < 1e-8, (
        f"seed {seed}: periodic tensor does NOT wrap on axis 0 "
        f"(|s(x+1,z)-s(x,z)|={d_axis0:.3e}, should be ~0 like axis 1's "
        f"{d_axis1:.3e}). The first tensor margin's declared period=1 is ignored."
    )
