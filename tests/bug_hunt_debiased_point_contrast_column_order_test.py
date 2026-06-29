"""Regression test: ``debiased_functional(target="point" | "contrast")`` must
work for a smooth model when the response column precedes the predictor column
in the training frame (the natural ``DataFrame({"y": ..., "x": ...})`` order).

``Model.debiased_functional`` documents ``"point"`` (``m(x0)``) and
``"contrast"`` (``m(x0) - m(x1)``) as its first two first-class estimands. For a
plain ``y ~ s(x)`` Gaussian/identity fit they currently raise

    GamError: debiased_functional: x0 design failed: Dimension mismatch:
    smooth term 's(x)' feature column 1 out of bounds for 1 columns

whenever the training frame is built response-column-first
(``DataFrame({"y": y, "x": x})``) — which is the natural order matching the
formula ``"y ~ s(x)"``. The exact same model fit from an ``{"x": x, "y": y}``
frame succeeds, so the failure is a pure column-ordering artifact.

Root cause (read, not patched): the ``"point"`` / ``"contrast"`` arms of
``debiased_functional`` in
``crates/gam-pyffi/src/latent/reml_latent_fit_ffi.rs`` (~line 3666) build the
``x0`` query design via ``dataset_with_model_schema(&model, &query_headers, ..)``
where ``query_headers`` are *only* the keys of ``x0`` (e.g. just ``["x"]``).
``dataset_with_model_schema``
(``crates/gam-pyffi/src/manifold/manifold_and_posterior_ffi.rs:3733``) projects
to the model's predictor columns, producing a one-column frame with ``x`` at
position 0. ``build_term_collection_design`` then indexes the smooth term's
feature column using the *full* training-schema offset — which is column 1 when
the training frame was ``[y, x]`` — and runs off the end of the projected frame.
With an ``[x, y]`` training frame the offset is 0 and the bug is masked.

The plug-in value of the ``"point"`` functional for a Gaussian/identity model is
exactly ``m(x0) = predict(x0)``; for ``"contrast"`` it is
``m(x0) - m(x1) = predict(x0) - predict(x1)``. This test asserts both arms
return that value (and do not raise) for the response-first column order. When
the design builder is fixed it passes without edits.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd

pytest: Any = importlib.import_module("pytest")

pytest.importorskip("gamfit._rust")

import gamfit


def _fit_response_first() -> tuple[Any, pd.DataFrame]:
    rng = np.random.default_rng(20240629)
    n = 800
    x = rng.uniform(0.0, 1.0, n)
    y = np.sin(2.0 * np.pi * x) + rng.normal(0.0, 0.1, n)
    # Response column FIRST — the natural order matching "y ~ s(x)".
    df = pd.DataFrame({"y": y, "x": x})
    assert list(df.columns) == ["y", "x"]
    model = gamfit.fit(df, "y ~ s(x)")
    return model, df


def _predict_at(model: Any, x0: float) -> float:
    out = model.predict(pd.DataFrame({"x": [x0]}))
    return float(np.asarray(out).ravel()[0])


def test_point_functional_works_with_response_first_columns() -> None:
    model, df = _fit_response_first()
    x0 = 0.3

    # Must not raise on the natural response-first column order (#bug).
    res = model.debiased_functional(df, target="point", x0={"x": x0})

    expected = _predict_at(model, x0)
    assert np.isfinite(res["theta_plugin"]), res
    # Plug-in point functional m(x0) == predict(x0) for Gaussian/identity.
    assert abs(res["theta_plugin"] - expected) < 1e-5, (
        f"point plug-in {res['theta_plugin']!r} != predict(x0)={expected!r}"
    )
    assert np.isfinite(res["theta_debiased"]), res


def test_contrast_functional_works_with_response_first_columns() -> None:
    model, df = _fit_response_first()
    x0, x1 = 0.25, 0.75

    res = model.debiased_functional(
        df, target="contrast", x0={"x": x0}, x1={"x": x1}
    )

    expected = _predict_at(model, x0) - _predict_at(model, x1)
    assert np.isfinite(res["theta_plugin"]), res
    # The signal sin(2*pi*x) makes m(0.25) - m(0.75) clearly non-zero, so this
    # is not a trivially-satisfiable assertion.
    assert abs(expected) > 0.5, expected
    assert abs(res["theta_plugin"] - expected) < 1e-5, (
        f"contrast plug-in {res['theta_plugin']!r} != "
        f"predict(x0)-predict(x1)={expected!r}"
    )
