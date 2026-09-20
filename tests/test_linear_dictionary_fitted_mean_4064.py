"""Issue #4064: the linear dictionary's affine origin comes from the Rust fit.

``center_rank_one=True`` with ``K=1`` fits the AFFINE model ``mean + code·atom``.
The Python facade used to re-derive both the "is this fit centered" rule and the
training mean in numpy. The Rust fit now returns the exact mean it baked into
``fitted`` (``None`` for a linear model); ``transform`` hands it back to Rust,
which also owns the input contract instead of Python.
"""

from __future__ import annotations

import importlib
import typing

import numpy as np

pytest = typing.cast(typing.Any, importlib.import_module("pytest"))
pytest.importorskip("gamfit._rust")

from gamfit.sae import linear_dictionary_fit


def _offset_line(n: int = 40) -> np.ndarray:
    rows = np.arange(n, dtype=np.float64)
    t = np.sin(0.37 * rows) * 2.5 + 0.1 * rows
    w = 0.05 * np.cos(1.3 * rows)
    offset = np.array([5.0, -3.0, 2.0])
    direction = np.array([1.0, 2.0, 2.0]) / 3.0
    wobble = np.array([2.0, 1.0, -2.0]) / 3.0
    return offset + t[:, None] * direction + w[:, None] * wobble


def test_centered_fit_reads_its_mean_from_rust() -> None:
    x = _offset_line()
    fit = linear_dictionary_fit(x, 1, center_rank_one=True)
    assert fit.centered
    assert fit.mean is not None
    np.testing.assert_allclose(fit.mean, x.mean(axis=0), rtol=0.0, atol=1e-13)
    # The fitted model is exactly assignments·atoms + mean, and the facade's
    # reconstruct and transform reproduce the fit's own state.
    np.testing.assert_allclose(fit.reconstruct(), fit.fitted, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(fit.transform(x), fit.assignments, rtol=0.0, atol=1e-12)


def test_linear_fits_carry_no_mean() -> None:
    x = _offset_line()
    for fit in (
        linear_dictionary_fit(x, 1),
        linear_dictionary_fit(x, 1, center_rank_one=False),
    ):
        assert fit.mean is None
        assert not fit.centered
        np.testing.assert_allclose(fit.reconstruct(), fit.fitted, rtol=0.0, atol=1e-12)


def test_transform_contract_is_enforced_in_rust() -> None:
    x = _offset_line()
    fit = linear_dictionary_fit(x, 1, center_rank_one=True)
    with pytest.raises(ValueError, match="top_k must be in"):
        fit.transform(x, top_k=2)
    poisoned = x.copy()
    poisoned[3, 1] = np.nan
    with pytest.raises(ValueError, match="X must be finite"):
        fit.transform(poisoned)
