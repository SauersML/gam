"""``summary().smooth_terms`` is empty without a reason only when the model has
no smooth or random-effect terms; every other absence is labeled by
``smooth_terms_unavailable`` (the ``reml_score_unavailable`` contract), and an
empty ``smooth_terms_frame()`` keeps its documented columns (#2787).
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit
from gamfit._summary import Summary

_COLUMNS = ["name", "edf", "ref_df", "chi_sq", "statistic", "p_value", "lambdas"]


def _data() -> dict[str, Any]:
    rng = np.random.default_rng(2787)
    x = rng.uniform(0.0, 1.0, 300)
    return {
        "x": x,
        "g": np.array([f"g{i % 4}" for i in range(x.size)]),
        "y": np.sin(2.0 * np.pi * x) + 0.3 * rng.standard_normal(x.size),
    }


def test_a_fit_with_smooth_terms_reports_the_table_and_no_reason() -> None:
    summary = gamfit.fit(_data(), "y ~ factor(g) + s(x)", family="gaussian").summary()
    assert summary.smooth_terms_unavailable is None
    names = [row["name"] for row in summary.smooth_terms]
    assert "s(x)" in names, names
    frame = summary.smooth_terms_frame()
    for column in _COLUMNS:
        assert column in frame.columns, list(frame.columns)


def test_a_fit_without_smooth_terms_is_empty_without_a_reason() -> None:
    """A parametric fit has nothing to tabulate: empty, and no reason."""
    summary = gamfit.fit(_data(), "y ~ x", family="gaussian").summary()
    assert summary.smooth_terms == []
    assert summary.smooth_terms_unavailable is None
    frame = summary.smooth_terms_frame()
    assert list(frame.columns) == _COLUMNS
    assert len(frame) == 0


def test_a_labeled_absence_survives_the_typed_summary() -> None:
    """The reason is a first-class field, not an ``extras`` leftover."""
    summary = Summary.from_dict(
        {
            "formula": "y ~ s(x)",
            "family_name": "Gaussian Identity",
            "smooth_terms_unavailable": "frozen-basis design replay failed: probe",
        }
    )
    assert summary.smooth_terms == []
    assert summary.smooth_terms_unavailable == "frozen-basis design replay failed: probe"
    assert "smooth_terms_unavailable" not in summary.extras
    assert list(summary.smooth_terms_frame().columns) == _COLUMNS
