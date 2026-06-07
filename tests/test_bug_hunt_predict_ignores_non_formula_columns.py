"""Bug hunt: ``fit``/``predict`` must ignore DataFrame columns the formula
never references, instead of silently re-encoding them against the training
schema.

A fitted GAM is a function of *exactly* the variables named in its formula
(plus offset / weights / response). Yet ``Model.predict`` re-encoded **every**
column present in both the prediction frame and the training schema — including
a column the user never put in the model — strictly against the training
levels. So an unrelated string label/ID column (``g`` below), kept in the frame
only to *group* rows, became a hidden strict categorical: a held-out fold whose
``g`` value never appeared in training aborted predict with::

    unseen level 'a' in categorical column 'g' at row 1; allowed levels: b,d,c,e

That is the classic leave-one-group-out / bootstrap foot-gun (#840): the error
names a column the user never typed into a formula, and the only workaround was
to manually subset ``df[[<formula columns>]]`` before every call.

Root cause (``crates/gam-pyffi/src/lib.rs``): ``dataset_with_model_schema``
(the single ingestion point for every predict variant) and ``schema_check``
passed the *full* prediction frame to ``encode_recordswith_schema``, which
strict-validates any column that exists in the saved schema. The fix projects
the frame onto ``prediction_consumable_columns(model)`` first, so columns the
model does not reference are dropped before encoding — mirroring
mgcv / glm / scikit-learn ``Pipeline`` semantics, where extra columns are
ignored.

The tests below assert the contract from several angles:

* the exact issue repro (a new ``g`` level in the held-out fold) now predicts;
* predictions are *invariant* to the presence/values of an unrelated column —
  an extra column cannot change the fitted surface;
* ``Model.check`` no longer flags the unrelated column;
* a column the model *does* use (``group(g)`` random effect) still handles an
  unseen level gracefully (regression guard that the projection did not loosen
  validation for genuinely-referenced columns).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gamfit


def _mean(out: object) -> np.ndarray:
    """Response-scale point predictions from any ``Model.predict`` return shape.

    A plain ``predict`` returns a 1-D ndarray of means; an interval predict
    returns a table with a ``mean`` column.
    """
    if isinstance(out, np.ndarray):
        return np.asarray(out, dtype=float).ravel()
    return np.asarray(out["mean"], dtype=float)


def _make_frame(seed: int = 0, n: int = 60) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(
        {
            "x": rng.uniform(0.0, 1.0, n),
            # Unrelated string label NOT in the formula — only used to group rows.
            "g": rng.choice(list("abcde"), n),
            "y": np.zeros(n),
        }
    )
    df["y"] = np.sin(3.0 * df["x"].values) + 0.05 * rng.randn(n)
    return df


def test_predict_with_unseen_level_in_non_formula_column() -> None:
    """The exact #840 repro: held-out fold carries a brand-new ``g`` level."""
    df = _make_frame()
    train = df[df["g"] != "a"].copy()  # levels b,c,d,e seen in training
    held_out = df[df["g"] == "a"].copy()  # 'a' is brand-new to the model
    assert not held_out.empty

    model = gamfit.fit(train, "y ~ s(x)")  # formula references only x (and y)

    # Must NOT raise "unseen level 'a' in categorical column 'g'".
    out = model.predict(held_out)
    mean = _mean(out)
    assert mean.shape[0] == held_out.shape[0]
    assert np.all(np.isfinite(mean))


def test_extra_column_does_not_change_predictions() -> None:
    """A column the formula never names cannot influence the prediction."""
    df = _make_frame(seed=1)
    train = df.copy()
    model = gamfit.fit(train, "y ~ s(x)")

    grid = pd.DataFrame({"x": np.linspace(0.05, 0.95, 12)})
    base = _mean(model.predict(grid))

    # Same grid, but with an unrelated categorical column whose levels were
    # never seen during training. It must be ignored entirely.
    grid_with_extra = grid.copy()
    grid_with_extra["g"] = ["zzz"] * len(grid)  # all-unseen level
    with_extra = _mean(model.predict(grid_with_extra))

    np.testing.assert_allclose(
        with_extra,
        base,
        rtol=0.0,
        atol=1e-12,
        err_msg="an unrelated, non-formula column changed the predictions",
    )

    # And a numeric junk column is equally inert.
    grid_with_numeric = grid.copy()
    grid_with_numeric["junk_id"] = np.arange(len(grid)) * 1.0e6
    with_numeric = _mean(model.predict(grid_with_numeric))
    np.testing.assert_allclose(with_numeric, base, rtol=0.0, atol=1e-12)


def test_check_does_not_flag_non_formula_column() -> None:
    """``Model.check`` validates only the model's own input columns."""
    df = _make_frame(seed=2)
    train = df[df["g"] != "a"].copy()
    held_out = df[df["g"] == "a"].copy()
    model = gamfit.fit(train, "y ~ s(x)")

    report = model.check(held_out)
    assert report.ok, (
        "Model.check flagged a column the formula never references: "
        f"{getattr(report, 'issues', report)}"
    )


def test_missing_required_column_still_errors() -> None:
    """Dropping unrelated columns must not mask a genuinely missing feature."""
    df = _make_frame(seed=3)
    model = gamfit.fit(df, "y ~ s(x)")

    # 'x' is the only required predictor; a frame without it must still fail.
    bad = pd.DataFrame({"g": ["b", "c"], "y": [0.0, 0.0]})
    with pytest.raises(Exception) as excinfo:
        model.predict(bad)
    assert "x" in str(excinfo.value)


def test_used_categorical_column_unseen_level_still_handled() -> None:
    """Regression guard: a column the model *does* use is still validated.

    When ``g`` is genuinely in the model as a random effect, an unseen level
    must be handled by the random-effect mechanism (prior-mean zero effect),
    not silently dropped — i.e. the projection only removes columns the model
    does not reference.
    """
    df = _make_frame(seed=4)
    train = df[df["g"] != "a"].copy()
    held_out = df[df["g"] == "a"].copy()
    assert not held_out.empty

    model = gamfit.fit(train, "y ~ s(x) + group(g)")
    out = model.predict(held_out)
    mean = _mean(out)
    assert mean.shape[0] == held_out.shape[0]
    assert np.all(np.isfinite(mean)), (
        "unseen random-effect level should map to the prior-mean effect, "
        "not produce NaN or an error"
    )
