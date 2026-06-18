"""Translation invariance of a univariate thin-plate smooth (issue #1269).

A univariate thin-plate regression spline ``s(x, bs="tp")`` is a functional of
the radial kernel ``phi(x_i - x_j)`` (coordinate *differences*) plus a polynomial
nullspace ``{1, x}`` penalised by ``integral (f'')**2``.  Both pieces depend on
the covariate only through differences and a smoothness functional that is itself
shift-invariant, so the fitted curve MUST be exactly translation-equivariant:
fitting on ``x`` and on ``x + c`` and predicting at correspondingly shifted query
points must give identical predictions.

The defect (now fixed, #1269 — the additive sibling of the #1215 rescaling fix):
the thin-plate polynomial nullspace block ``P = {1, x}`` and the side-constraint
nullspace ``P(knots)^T alpha = 0`` were assembled in RAW covariate coordinates.
When ``x`` carries a large offset (centred-vs-raw "year", an axis with a big
mean, ...) the ``{1, x}`` columns become near-collinear, the design ill-conditions,
and REML lambda-selection lands in a slightly different basin — moving the fit by
~1.4% of signal range even though ``{1, x + c}`` spans the same model space.
``bs="ps"`` and ``bs="cr"`` (local bases built on a knot grid that shifts with the
data) were already invariant to ~1e-13.

The fix builds the kernel / polynomial / constraint blocks in a location-
standardised (knot-mean-centred) frame, so the basis — and hence the fit — is
invariant to a pure translation.  The control below pins ``bs="ps"`` / ``bs="cr"``
as already-invariant references that isolate the defect to the tp path.
"""

from __future__ import annotations

import contextlib
import importlib
import io
from typing import Any

import numpy as np
import pandas as pd

pytest: Any = importlib.import_module("pytest")

pytest.importorskip("gamfit._rust")

import gamfit


OFFSETS = (0.0, 1.0, 10.0, 50.0, 100.0, 1000.0)


def _signal(x: np.ndarray) -> np.ndarray:
    # A smooth, clearly non-polynomial target so the smooth genuinely bends.
    return np.sin(1.7 * x) + 0.5 * np.cos(0.9 * x)


def _fit_and_predict_on_grid(bs: str, offset: float) -> tuple[np.ndarray, np.ndarray]:
    """Fit ``y ~ s(x_shifted, bs=...)`` where the covariate is translated by
    ``offset`` and the response is unchanged, then predict on a grid that is
    shifted by the SAME ``offset``.  Returns ``(grid_unshifted, prediction)`` so
    predictions across offsets are directly comparable point-by-point."""
    rng = np.random.default_rng(7)
    n = 600
    x0 = np.sort(rng.uniform(-3.0, 3.0, n))
    y = _signal(x0) + 0.15 * rng.standard_normal(n)

    x_shifted = x0 + offset
    fr = pd.DataFrame({"x": x_shifted, "y": y})
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        m = gamfit.fit(fr, f'y ~ s(x, bs="{bs}")', family="gaussian")

    grid0 = np.linspace(-2.8, 2.8, 80)
    grid_shifted = grid0 + offset
    pred = np.asarray(m.predict(pd.DataFrame({"x": grid_shifted}))).ravel()
    return grid0, pred


def _max_drift(bs: str) -> tuple[float, float]:
    """Max absolute prediction drift across all offsets vs. the unshifted fit,
    relative to the signal range over the evaluation grid."""
    grid0, ref = _fit_and_predict_on_grid(bs, OFFSETS[0])
    signal_range = float(_signal(grid0).max() - _signal(grid0).min())
    assert signal_range > 0.5, f"degenerate signal range {signal_range}"
    worst = 0.0
    for off in OFFSETS[1:]:
        _, pred = _fit_and_predict_on_grid(bs, off)
        drift = float(np.max(np.abs(pred - ref)))
        worst = max(worst, drift)
    return worst, signal_range


def test_thinplate_fit_is_invariant_to_covariate_translation() -> None:
    """The tp fit must not move when the covariate is purely translated."""
    drift, signal_range = _max_drift("tp")
    rel = drift / signal_range
    assert rel < 1e-3, (
        f"thin-plate s(x, bs='tp') is NOT translation invariant: "
        f"max drift {drift:.3e} = {rel:.3%} of signal range {signal_range:.3f} "
        f"(ceiling 1e-3)"
    )


def test_local_bases_are_translation_invariant_control() -> None:
    """Control: bs='ps' and bs='cr' are already translation invariant, which
    isolates the defect to the tp path."""
    for bs in ("ps", "cr"):
        drift, signal_range = _max_drift(bs)
        rel = drift / signal_range
        assert rel < 1e-3, (
            f"control basis bs='{bs}' unexpectedly drifted under translation: "
            f"max drift {drift:.3e} = {rel:.3%} of signal range {signal_range:.3f}"
        )


if __name__ == "__main__":  # pragma: no cover - manual smoke run
    for bs in ("tp", "ps", "cr"):
        d, sr = _max_drift(bs)
        print(f"bs={bs:>3}: max drift {d:.3e}  ({d / sr:.3%} of range {sr:.3f})")
    raise SystemExit(pytest.main([__file__, "-v"]))
