"""Regression test (#1621, companion angle): ``debiased_functional`` ``point`` /
``contrast`` query designs must be built against the model's FULL training
schema, so the estimate is INVARIANT to the training frame's column order and
works for multi-predictor and mixed parametric+smooth schemas — not only the
single-smooth, predictor-first layout the sibling test covers.

The root cause of #1621 was that the ``point``/``contrast`` arms encoded the
``x0`` query using only the *keys of* ``x0``, producing a frame whose predictor
sat at position 0, while ``build_term_collection_design`` indexed the smooth's
feature column at its FULL training-schema offset. That made the result depend
on where the response (and every other column) sat in the training frame: an
``[x, y]`` frame happened to work, an ``[y, x]`` frame ran off the end of the
1-column projection. This test attacks the defect from the invariance side:

* **Column-order invariance.** The same data fit from several column
  permutations must yield the same ``point`` plug-in (it is ``m(x0)``, a
  property of the fitted function, not of the dataframe layout).
* **Multi-predictor full schema.** ``y ~ s(x) + s(z)`` from a ``[y, x, z]``
  frame — where ``s(z)``'s feature offset is 2 — was the exact case the issue
  reported as ``feature column 2 out of bounds for 2 columns``.
* **Mixed parametric + smooth.** ``y ~ s(x) + w`` exercises a parametric column
  living inside the full schema.
* **Clear contract on a partial ``x0``.** A query that omits a model predictor
  must fail with an explicit "must specify a value for every model predictor"
  message — not a cryptic out-of-bounds abort, and never a silent placeholder
  evaluation.

Every assertion is keyed to ``predict`` (the independent oracle: the plug-in
``point`` functional of a Gaussian/identity model is exactly ``predict(x0)``).
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd

pytest: Any = importlib.import_module("pytest")

pytest.importorskip("gamfit._rust")

import gamfit


def _predict_at(model: Any, row: dict[str, float]) -> float:
    out = model.predict(pd.DataFrame({k: [v] for k, v in row.items()}))
    return float(np.asarray(out).ravel()[0])


def test_point_is_invariant_to_training_column_order() -> None:
    rng = np.random.default_rng(11)
    n = 700
    x = rng.uniform(0.0, 1.0, n)
    y = np.sin(2.0 * np.pi * x) + rng.normal(0.0, 0.1, n)
    base = {"x": x, "y": y}
    x0 = {"x": 0.37}

    estimates = []
    predicts = []
    # Several column permutations of the *same* data + formula. The fitted
    # function — hence m(x0) — is identical; only the dataframe layout changes.
    for order in (["y", "x"], ["x", "y"]):
        df = pd.DataFrame({c: base[c] for c in order})
        assert list(df.columns) == order
        model = gamfit.fit(df, "y ~ s(x)")
        res = model.debiased_functional(df, target="point", x0=x0)
        assert np.isfinite(res["theta_plugin"]), (order, res)
        estimates.append(res["theta_plugin"])
        predicts.append(_predict_at(model, x0))

    # point plug-in == predict(x0) for every layout ...
    for est, pred in zip(estimates, predicts):
        assert abs(est - pred) < 1e-5, (est, pred)
    # ... and the layouts agree with each other to fitting tolerance (the fits
    # are the same model up to optimizer noise, so well under 1e-3).
    assert abs(estimates[0] - estimates[1]) < 1e-3, estimates


def test_point_contrast_multi_predictor_response_first() -> None:
    rng = np.random.default_rng(101)
    n = 900
    x = rng.uniform(0.0, 1.0, n)
    z = rng.uniform(0.0, 1.0, n)
    y = np.sin(2.0 * np.pi * x) + 0.8 * np.cos(np.pi * z) + rng.normal(0.0, 0.1, n)
    # [y, x, z] order: s(z)'s feature offset is 2 — the issue's reported
    # "feature column 2 out of bounds for 2 columns" case.
    df = pd.DataFrame({"y": y, "x": x, "z": z})
    assert list(df.columns) == ["y", "x", "z"]
    model = gamfit.fit(df, "y ~ s(x) + s(z)")

    x0 = {"x": 0.2, "z": 0.6}
    x1 = {"x": 0.8, "z": 0.1}
    pt = model.debiased_functional(df, target="point", x0=x0)
    ct = model.debiased_functional(df, target="contrast", x0=x0, x1=x1)

    assert abs(pt["theta_plugin"] - _predict_at(model, x0)) < 1e-5, pt
    expected_contrast = _predict_at(model, x0) - _predict_at(model, x1)
    assert abs(expected_contrast) > 0.3, expected_contrast
    assert abs(ct["theta_plugin"] - expected_contrast) < 1e-5, (ct, expected_contrast)


def test_point_mixed_parametric_plus_smooth_response_first() -> None:
    rng = np.random.default_rng(202)
    n = 800
    x = rng.uniform(0.0, 1.0, n)
    w = rng.uniform(-1.0, 1.0, n)
    y = np.sin(2.0 * np.pi * x) + 1.3 * w + rng.normal(0.0, 0.1, n)
    df = pd.DataFrame({"y": y, "x": x, "w": w})
    model = gamfit.fit(df, "y ~ s(x) + w")

    x0 = {"x": 0.4, "w": 0.5}
    res = model.debiased_functional(df, target="point", x0=x0)
    assert abs(res["theta_plugin"] - _predict_at(model, x0)) < 1e-5, res


def test_partial_x0_reports_clear_missing_predictor_error() -> None:
    rng = np.random.default_rng(303)
    n = 600
    x = rng.uniform(0.0, 1.0, n)
    z = rng.uniform(0.0, 1.0, n)
    y = np.sin(2.0 * np.pi * x) + 0.5 * z + rng.normal(0.0, 0.1, n)
    df = pd.DataFrame({"y": y, "x": x, "z": z})
    model = gamfit.fit(df, "y ~ s(x) + s(z)")

    # x0 omits 'z' — m(x0) is undefined. Must be an explicit contract error,
    # not the old "feature column out of bounds" abort and not a silent answer.
    with pytest.raises(Exception) as excinfo:
        model.debiased_functional(df, target="point", x0={"x": 0.5})
    msg = str(excinfo.value)
    assert "every model predictor" in msg and "z" in msg, msg
    assert "out of bounds" not in msg, msg
