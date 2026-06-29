"""Regression test (#1621, follow-on): ``debiased_functional`` ``point`` /
``contrast`` must not abort when the training frame carries an UNREFERENCED
categorical bookkeeping column.

The #1621 fix introduced ``debiased_query_design_full_schema``, which lays out
the ``x0`` / ``x1`` query row over *every* training column (filling any column
``x0`` does not name with a neutral ``"0"`` placeholder) and encodes the whole
row against the saved training schema — deliberately WITHOUT the predict-time
projection to predictor-only columns, so the smooth terms' full-schema feature
offsets resolve.

That helper, as first written, granted lenient (unseen-OK) encoding only to
random-effect grouping columns. So a perfectly ordinary frame that carried an
extra **categorical** bookkeeping column the formula never references —
``DataFrame({"y": ..., "x": ..., "group": ["a", "b", ...]})`` fit ``y ~ s(x)``
— filled ``group`` with the ``"0"`` placeholder and re-validated it against the
saved levels, aborting with

    GamError: ... unseen level '0' in categorical column 'group' ...

even though ``group`` never enters the mean design. That is exactly the #840
"leave-one-group-out" foot-gun ``predict`` avoids by projecting the frame to
the model's columns (``project_frame_to_model_columns``): a column the model
does not reference must be ignored, not strict-encoded.

The fix extends the lenient policy to every column NOT in the model's
required-prediction set (the placeholder-filled columns), so the estimate is
unaffected by any bookkeeping column. Each assertion is keyed to ``predict``
(the independent oracle: the ``point`` plug-in of a Gaussian/identity model is
exactly ``predict(x0)``).
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


def test_point_works_with_unreferenced_categorical_column() -> None:
    rng = np.random.default_rng(424242)
    n = 800
    x = rng.uniform(0.0, 1.0, n)
    y = np.sin(2.0 * np.pi * x) + rng.normal(0.0, 0.1, n)
    # An extra categorical bookkeeping column the formula never references.
    group = rng.choice(["alpha", "beta", "gamma"], size=n)
    # Response first, then predictor, then the bookkeeping factor.
    df = pd.DataFrame({"y": y, "x": x, "group": pd.Categorical(group)})
    model = gamfit.fit(df, "y ~ s(x)")

    x0 = 0.3
    # Must NOT abort with `unseen level '0' in categorical column 'group'`.
    res = model.debiased_functional(df, target="point", x0={"x": x0})

    expected = _predict_at(model, {"x": x0})
    assert np.isfinite(res["theta_plugin"]), res
    assert abs(res["theta_plugin"] - expected) < 1e-5, (
        f"point plug-in {res['theta_plugin']!r} != predict(x0)={expected!r}"
    )
    assert np.isfinite(res["theta_debiased"]), res


def test_contrast_works_with_unreferenced_categorical_column() -> None:
    rng = np.random.default_rng(525252)
    n = 800
    x = rng.uniform(0.0, 1.0, n)
    y = np.sin(2.0 * np.pi * x) + rng.normal(0.0, 0.1, n)
    label = rng.choice(["lo", "hi"], size=n)
    df = pd.DataFrame({"y": y, "x": x, "label": pd.Categorical(label)})
    model = gamfit.fit(df, "y ~ s(x)")

    x0, x1 = 0.25, 0.75
    res = model.debiased_functional(df, target="contrast", x0={"x": x0}, x1={"x": x1})

    expected = _predict_at(model, {"x": x0}) - _predict_at(model, {"x": x1})
    assert abs(expected) > 0.5, expected
    assert abs(res["theta_plugin"] - expected) < 1e-5, (
        f"contrast plug-in {res['theta_plugin']!r} != "
        f"predict(x0)-predict(x1)={expected!r}"
    )


def test_estimate_is_invariant_to_unreferenced_column_presence() -> None:
    """The point estimate must be identical whether or not the bookkeeping
    column is present in the training frame: it is a property of the fitted
    function m(x), not of the dataframe layout."""
    rng = np.random.default_rng(636363)
    n = 700
    x = rng.uniform(0.0, 1.0, n)
    y = np.sin(2.0 * np.pi * x) + rng.normal(0.0, 0.1, n)
    group = rng.choice(["a", "b", "c"], size=n)
    x0 = {"x": 0.42}

    df_plain = pd.DataFrame({"y": y, "x": x})
    df_extra = pd.DataFrame({"y": y, "x": x, "group": pd.Categorical(group)})

    m_plain = gamfit.fit(df_plain, "y ~ s(x)")
    m_extra = gamfit.fit(df_extra, "y ~ s(x)")

    est_plain = m_plain.debiased_functional(df_plain, target="point", x0=x0)["theta_plugin"]
    est_extra = m_extra.debiased_functional(df_extra, target="point", x0=x0)["theta_plugin"]

    # Same data + formula; the bookkeeping column is inert, so the fits — hence
    # m(x0) — agree to well under fitting tolerance.
    assert abs(est_plain - est_extra) < 1e-6, (est_plain, est_extra)


def test_multiple_bookkeeping_columns_far_from_placeholder() -> None:
    """Two unreferenced bookkeeping columns (one categorical, one continuous),
    with the predictor's domain far from the ``"0"`` placeholder, must not
    perturb the estimate — proving the placeholder genuinely never reaches the
    mean design rather than merely encoding to a harmless small value."""
    rng = np.random.default_rng(747474)
    n = 600
    x = rng.uniform(5.0, 10.0, n)  # far from the "0" placeholder
    y = 2.0 * x + rng.normal(0.0, 0.1, n)
    ident = rng.choice(["p", "q", "r"], size=n)
    note = rng.uniform(100.0, 200.0, n)
    # A categorical bookkeeping column BEFORE the response, and a continuous one
    # after the predictor — both unreferenced by `y ~ s(x)`.
    df = pd.DataFrame(
        {"id": pd.Categorical(ident), "y": y, "x": x, "note": note}
    )
    model = gamfit.fit(df, "y ~ s(x)")

    res = model.debiased_functional(df, target="point", x0={"x": 7.0})
    expected = _predict_at(model, {"x": 7.0})
    assert abs(res["theta_plugin"] - expected) < 1e-5, (res, expected)


def test_contrast_over_categorical_predictor_with_bookkeeping_column() -> None:
    """A contrast over a genuine categorical PREDICTOR (group b vs a at fixed x)
    must match ``predict(x0) - predict(x1)`` while an unreferenced bookkeeping
    column rides along — the predictor is strict-encoded (real level), the
    bookkeeping column lenient-encoded (placeholder)."""
    rng = np.random.default_rng(858585)
    n = 600
    x = rng.uniform(0.0, 1.0, n)
    grp = rng.choice(["a", "b"], size=n)
    y = np.sin(2.0 * np.pi * x) + (grp == "b") * 0.7 + rng.normal(0.0, 0.1, n)
    tag = rng.choice(["t1", "t2", "t3"], size=n)  # inert bookkeeping factor
    df = pd.DataFrame(
        {"y": y, "x": x, "group": pd.Categorical(grp), "tag": pd.Categorical(tag)}
    )
    model = gamfit.fit(df, "y ~ s(x) + group")

    res = model.debiased_functional(
        df, target="contrast", x0={"x": 0.5, "group": "b"}, x1={"x": 0.5, "group": "a"}
    )

    def _pred(g: str) -> float:
        frame = pd.DataFrame(
            {"x": [0.5], "group": pd.Categorical([g], categories=["a", "b"])}
        )
        return float(np.asarray(model.predict(frame)).ravel()[0])

    expected = _pred("b") - _pred("a")
    assert abs(expected) > 0.3, expected
    assert abs(res["theta_plugin"] - expected) < 1e-5, (res, expected)
