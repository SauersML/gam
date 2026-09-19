"""A default fit is quiet; solver traces are opt-in through ``logging``.

The pyGAM audit (bench/pygam_audit: api.md F12/F13, packaging PKG-03/PKG-04,
DOC-15) found two kinds of noise on every default ``gamfit.fit``:

* the engine's stderr logger was installed at ``warn`` on import, and the
  solver logs its routine progress (``[OUTER]``, ``[HGB]``, ``[INDEF-HESS]``
  lines) at that level, so every fit printed solver internals to stderr with
  only a private level setter to stop it;
* every inference note became a ``GamInferenceWarning``, including the
  informational "Automatically set N internal knots" note a default ``s(x)``
  always records, so every default fit warned about nothing the user did.

Now the engine's records go to the ``gamfit`` Python logger at debug/trace
level (silent unless the caller lowers that logger's level), and only
advisories — the model differs from the literal request — warn, attributed to
the caller's line.
"""

from __future__ import annotations

import importlib
import logging
import warnings
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit


def _default_data(seed: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    n = 400
    x = rng.uniform(0.0, 1.0, n)
    y = np.sin(2.0 * np.pi * x) + rng.normal(0.0, 0.3, n)
    return {"x": x.tolist(), "y": y.tolist()}


def _ternary_data(seed: int = 1542) -> dict:
    rng = np.random.default_rng(seed)
    n = 600
    x = rng.integers(0, 3, n).astype(float)
    y = 0.5 * x + rng.normal(0.0, 0.3, n)
    return {"x": x.tolist(), "y": y.tolist()}


def test_default_fit_writes_nothing_to_stdout_or_stderr(capfd: Any) -> None:
    data = _default_data()
    capfd.readouterr()
    gamfit.fit(data, "y ~ s(x)")
    out, err = capfd.readouterr()
    assert out == "", f"a default fit wrote to stdout:\n{out}"
    assert err == "", f"a default fit wrote to stderr:\n{err}"


def test_default_fit_raises_no_warning_when_warnings_are_errors() -> None:
    data = _default_data()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = gamfit.fit(data, "y ~ s(x)")
    # The default knot choice is still recorded — as a note, not a warning.
    assert any("internal knots" in note for note in model.notes), model.notes
    assert any("internal knots" in note for note in model.summary().notes)
    assert "Notes:" in str(model.summary())


def test_solver_records_reach_the_gamfit_logger_at_debug(caplog: Any) -> None:
    data = _default_data()
    with caplog.at_level(logging.DEBUG, logger="gamfit"):
        gamfit.fit(data, "y ~ s(x)")
        # Records produced on engine worker threads are delivered with the GIL
        # held no later than the next engine call.
        gamfit.fit(data, "y ~ s(x)")
    records = [record for record in caplog.records if record.name == "gamfit"]
    assert records, "no engine records reached the gamfit logger at DEBUG"
    assert all(record.levelno <= logging.DEBUG for record in records), [
        (record.levelname, record.getMessage()) for record in records
    ]
    assert all(hasattr(record, "rust_target") for record in records)


def test_no_engine_records_at_the_default_level(caplog: Any) -> None:
    data = _default_data()
    with caplog.at_level(logging.WARNING, logger="gamfit"):
        gamfit.fit(data, "y ~ s(x)")
        gamfit.fit(data, "y ~ s(x)")
    assert [record for record in caplog.records if record.name == "gamfit"] == []


def test_advisory_warning_points_at_the_callers_line() -> None:
    data = _ternary_data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gamfit.fit(data, "y ~ s(x, bs='cr', k=10)")
    advisories = [w for w in caught if issubclass(w.category, gamfit.GamInferenceWarning)]
    assert advisories, "a capped cr basis must still warn"
    for warning in advisories:
        assert warning.filename == __file__, (
            f"GamInferenceWarning attributed to {warning.filename}:{warning.lineno}, "
            f"not to the caller in {__file__}"
        )
