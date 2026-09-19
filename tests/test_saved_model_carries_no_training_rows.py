"""A saved standard GAM carries no per-row training data (speed F6).

The fitted model used to persist the exact full-conformal substrate (the
training design ``X`` and response ``y``) and the final PIRLS working weights
and response, so ``Model.dumps()`` grew by about 617 bytes per training row, and
every accessor re-parsed that JSON. The saved model now keeps only O(p^2) state
plus term metadata: its size does not depend on the training rows, full
conformal takes the labeled rows again at predict time, and accessors read the
payload compiled once when the model is built.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit

FORMULA = "y ~ s(x1, k=8) + s(x2, k=8)"


def _frame(n: int, seed: int) -> "pd.DataFrame":
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0.0, 1.0, n)
    x2 = rng.uniform(0.0, 1.0, n)
    y = np.sin(2.0 * np.pi * x1) + 0.5 * np.cos(np.pi * x2) + rng.normal(0.0, 0.3, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def test_saved_model_size_does_not_grow_with_training_rows() -> None:
    small = gamfit.fit(_frame(500, 1), FORMULA).dumps()
    large = gamfit.fit(_frame(8000, 2), FORMULA).dumps()
    # 16x the rows. At ~617 bytes/row the old payload grew by several MB; the
    # O(p^2) payload differs only by the digits of its fitted numbers.
    assert len(large) < 1.1 * len(small), (
        f"saved model grew from {len(small)} to {len(large)} bytes for 16x the "
        "training rows; it must hold no per-row training data"
    )


def test_full_conformal_takes_the_labeled_rows_at_predict_time() -> None:
    train = _frame(300, 3)
    model = gamfit.loads(gamfit.fit(train, FORMULA).dumps())
    test = _frame(200, 4)

    with pytest.raises(ValueError, match="exactly one of training_data"):
        model.predict(test[["x1", "x2"]], interval="conformal")
    with pytest.raises(ValueError, match="exactly one of training_data"):
        model.predict(
            test[["x1", "x2"]],
            interval="conformal",
            training_data=train,
            calibration=train,
        )
    with pytest.raises(ValueError, match="training_data= applies only"):
        model.predict(test[["x1", "x2"]], training_data=train)

    out = model.predict(
        test[["x1", "x2"]],
        interval="conformal",
        training_data=train,
        conformal_level=0.9,
        return_type="dict",
    )
    lower = np.asarray(out["posterior_mean_lower"], dtype=float)
    upper = np.asarray(out["posterior_mean_upper"], dtype=float)
    mean = np.asarray(out["posterior_mean"], dtype=float)
    assert np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
    assert np.all(lower <= mean) and np.all(mean <= upper)
    covered = float(np.mean((test["y"] >= lower) & (test["y"] <= upper)))
    assert covered >= 0.8, f"full-conformal 0.9 band covered only {covered:.3f}"


def test_accessors_read_the_compiled_model() -> None:
    model = gamfit.fit(_frame(300, 5), FORMULA)
    reloaded = gamfit.loads(model.dumps())
    compiled = reloaded._prediction_model
    assert reloaded.formula == compiled.formula == FORMULA
    assert reloaded.notes == list(compiled.inference_notes)
    assert reloaded.used_device is compiled.used_device
    assert reloaded.model_class == compiled.predict_class_name
    assert reloaded._training_table_kind == compiled.training_table_kind == "pandas"
    # The summary is built once per compiled model and served from it after.
    assert reloaded.summary().formula == model.summary().formula
    assert reloaded.smoothing_parameters() == model.smoothing_parameters()
    # The byte-level JSON re-parsing accessors are gone from the extension.
    rust = importlib.import_module("gamfit._rust")
    for removed in (
        "saved_model_payload_string",
        "required_saved_model_payload_string",
        "inference_notes_from_model",
        "saved_model_predict_class_name",
    ):
        assert not hasattr(rust, removed), removed
