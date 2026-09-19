"""pyGAM audit DOC-18 / F7: two-level label responses and the one-row fit.

A response column holding exactly two string labels is a binary outcome, so
family auto-detection picks binomial for it, the same as for a numeric 0/1 or
bool column. The labels are coded in sorted order (the first level is 0, the
second is 1, so ``{"no", "yes"}`` models ``P(y = "yes")``) whatever order the
data arrived in, and the fit's notes state that coding.

A one-row fit has no residual contrast for REML once the intercept is fit, so
it is refused as too few rows before any family is inferred, rather than as
the degenerate binomial or constant covariate the single row would otherwise
trip first.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamfit


def _outcome(n: int = 240) -> tuple[np.ndarray, np.ndarray]:
    # Deterministic: threshold the true P(event) against a golden-ratio sequence.
    x = np.linspace(0.0, 1.0, n)
    probability = 1.0 / (1.0 + np.exp(-(3.0 * x - 1.5)))
    u = np.mod(np.arange(n) * (np.sqrt(5.0) - 1.0) / 2.0, 1.0)
    return x, u < probability


def test_two_level_string_response_is_auto_detected_as_binomial():
    x, event = _outcome()
    # "yes" comes first in the data, so encounter order and sorted order differ.
    labels = ["yes" if e else "no" for e in event]
    assert labels[0] == "yes"
    model = gamfit.fit({"x": x, "y": labels}, "y ~ s(x)")
    assert model.family_name.lower().startswith("binomial")

    reference = gamfit.fit({"x": x, "y": event.astype(float)}, "y ~ s(x)", family="binomial")
    grid = {"x": np.linspace(0.0, 1.0, 11)}
    p_labels = np.asarray(model.predict(grid), dtype=float)
    p_numeric = np.asarray(reference.predict(grid), dtype=float)
    # The label fit is the 0/1 fit with 'yes' = 1: same probabilities, rising in x.
    np.testing.assert_allclose(p_labels, p_numeric, rtol=1e-8, atol=1e-10)
    assert p_labels[0] < 0.3 < 0.7 < p_labels[-1]

    assert any("'no' = 0" in note and "'yes' = 1" in note for note in model.notes), model.notes

    explicit = gamfit.fit({"x": x, "y": labels}, "y ~ s(x)", family="binomial")
    np.testing.assert_allclose(np.asarray(explicit.predict(grid), dtype=float), p_labels)


def test_three_label_response_still_asks_for_a_family():
    x, event = _outcome()
    labels = ["maybe" if i % 7 == 0 else ("yes" if e else "no") for i, e in enumerate(event)]
    with pytest.raises(gamfit.errors.FormulaError, match="multinomial"):
        gamfit.fit({"x": x, "y": labels}, "y ~ s(x)")


@pytest.mark.parametrize("family", [None, "binomial", "gaussian", "poisson"])
def test_single_row_reports_too_few_rows(family):
    kwargs = {} if family is None else {"family": family}
    with pytest.raises(gamfit.errors.DataError, match="too few rows"):
        gamfit.fit({"x": [0.5], "y": [1.0]}, "y ~ s(x)", **kwargs)
