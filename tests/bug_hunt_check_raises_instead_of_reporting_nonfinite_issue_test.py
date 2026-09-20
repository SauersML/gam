"""Bug hunt: ``Model.check(data)`` raises on a non-finite covariate instead of
reporting it as a ``SchemaIssue``, which breaks the guard pattern the docs
prescribe for exactly this situation.

``check()`` is documented as the non-raising probe you run *before* ``predict``:

    docs/exceptions.md:129-131
        ``Model.check(data)`` reports missing columns directly and returns
        schema encoder failures as issues **without raising**

    docs/exceptions.md:178-186 and docs/cookbook.md:321-330 (identical recipe)
        def safe_predict(model, data):
            check = model.check(data)
            if not check.ok:
                check.raise_for_error()
            return model.predict(data)

    docs/diagnostics.md:104
        check.raise_for_error()       # raises ValueError

The whole point of that shape is that the caller decides when to raise, and gets
a structured ``(kind, column, message)`` list to log or surface. It holds for the
failure modes the guard already covers — a missing column returns
``ok=False, issues=[('missing_column', 'z')]``, and a string cell in a numeric
column returns ``ok=False, issues=[('schema_error', None)]`` — but not for a
``NaN`` / ``inf`` cell, which is the single most common defect in a real serving
frame. There, ``check()`` itself throws:

    model.check({"x": [0.1, float("nan"), 0.3]})
    -> gamfit._rust.GamfitError: non-finite value at row 2, column 'x'

so ``safe_predict`` blows up inside its own guard, with a bare ``GamfitError``
rather than the mapped ``SchemaMismatchError`` whose docstring points the reader
back at ``check()`` ("Compare the serving data with the training schema using
model.check(...)", docs/exceptions.md:168).

Mechanism: ``check()`` reaches the engine through the same
``gamfit/_tables.py:160`` ``normalize_table`` -> ``encoded_table_from_columns``
(``crates/gam-pyffi/src/model/model_ffi.rs:786``) ingestion path as ``predict``,
and that path rejects non-finite cells at lines 833-840 before any schema
comparison happens. The issue-returning code in ``Model.check`` only ever sees
tables that already normalized successfully, so the one class of defect that
dies during normalization can never be reported as an issue.

Observed: ``check()`` raises ``GamfitError`` for a non-finite value in a modelled
column.

Expected: ``check()`` returns a ``SchemaCheck`` with ``ok is False`` and an
issue naming the offending column, exactly as it does for ``missing_column``
and ``schema_error``. ``check.raise_for_error()`` remains the caller's choice.

Related: the ingest scan is also unscoped, so the same raise happens for columns
no term consumes — that is a separate defect, tested in
``tests/bug_hunt_unreferenced_column_missing_value_blocks_fit_test.py``. This
file uses a column the model DOES reference, so it stays red independently.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit


def _fitted() -> Any:
    rng = np.random.default_rng(20260823)
    x = np.linspace(0.0, 1.0, 120)
    y = np.sin(2.0 * np.pi * x) + 0.15 * rng.standard_normal(x.size)
    return gamfit.fit({"x": x, "y": y}, "y ~ s(x)", family="gaussian")


def _holed(bad: float) -> dict[str, Any]:
    column = np.linspace(0.1, 0.9, 5)
    column[2] = bad
    return {"x": column}


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_check_reports_nonfinite_covariate_as_an_issue(bad: float) -> None:
    """The documented guard must answer the question, not throw it back."""
    model = _fitted()

    result = model.check(_holed(bad))

    assert not result.ok, (
        "check() accepted a frame carrying a non-finite covariate that "
        "predict() refuses"
    )
    issues = list(result.issues or [])
    assert issues, "check() reported ok=False with no issue explaining why"
    assert any(issue.column == "x" for issue in issues), (
        "no reported issue names the offending column 'x': "
        f"{[(issue.kind, issue.column, issue.message) for issue in issues]}"
    )


def test_documented_safe_predict_guard_raises_valueerror_not_gamerror() -> None:
    """docs/exceptions.md:178-186 / docs/cookbook.md:321-330, verbatim."""
    model = _fitted()

    def safe_predict(model: Any, data: Any) -> Any:
        check = model.check(data)
        if not check.ok:
            check.raise_for_error()
        return model.predict(data)

    # The guard's contract: a bad frame surfaces as the ValueError that
    # ``raise_for_error()`` documents, never as a raw engine error escaping
    # from inside ``check()`` itself.
    with pytest.raises(ValueError) as caught:
        safe_predict(model, _holed(np.nan))

    assert not isinstance(caught.value, gamfit.errors.GamfitError), (
        "the engine error escaped from inside check(), so the guard never ran: "
        f"{type(caught.value).__name__}: {caught.value}"
    )


def test_predict_still_rejects_a_nonfinite_modeled_covariate() -> None:
    """Only ``check`` is non-raising; prediction must never consume the hole."""
    model = _fitted()

    with pytest.raises(gamfit.errors.GamfitError, match="non-finite.*column 'x'"):
        model.predict(_holed(np.nan))


def test_arrow_null_in_a_modeled_covariate_is_a_structured_nonfinite_issue() -> None:
    """The Arrow and ordinary-table paths share one model-aware contract."""
    pa = pytest.importorskip("pyarrow")
    model = _fitted()
    frame = pa.table({"x": pa.array([0.1, 0.3, None, 0.7, 0.9])})

    result = model.check(frame)

    assert not result.ok
    issues = list(result.issues or [])
    assert any(
        issue.kind == "non_finite" and issue.column == "x" for issue in issues
    ), [(issue.kind, issue.column, issue.message) for issue in issues]
    with pytest.raises(gamfit.errors.GamfitError, match="non-finite.*column 'x'"):
        model.predict(frame)
