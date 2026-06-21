"""Bug hunt: a categorical ``by=`` factor smooth is silently treated as a
*numeric* by-variable when the data is supplied as a **dict of columns** (or
any other "untyped" container) and the level labels happen to be numeric
strings (``"0"``, ``"1"``, …) — even though the identical data passed as a
pandas DataFrame fits a correct per-level factor smooth.

Background
----------
Issue #1317 fixed exactly this misclassification for *typed* table libraries:
``gamfit/_tables.py`` now stamps a categorical sentinel on every column whose
*source dtype* is non-numeric, so a pandas/polars/pyarrow string column whose
labels parse as numbers ("0"/"1") is still lowered to a categorical by-factor.
That fix lives in ``categorical_dtype_columns`` (``_tables.py:110``), which only
inspects ``kind in {"pandas", "polars", "pyarrow"}`` and returns an EMPTY set
for every other container (``_tables.py:152-157`` and the docstring at
``:110-115``: *"untyped inputs (mappings, record/row sequences, numpy) return an
empty set and keep the value-based numeric inference"*).

So for a plain ``dict`` of columns — a documented, first-class input
(``README``: "dict of columns") — no column is ever marked categorical, and the
Rust column-kind inference (``src/inference/data.rs`` ``all_numeric`` →
``Binary``/``Continuous``) lowers the all-numeric-looking string column ``g`` to
a NUMERIC by-variable. ``src/terms/term_builder.rs`` then builds
``value(g) * f(x)`` — one shared shape scaled by the integer value of the label
— instead of one independent centred smooth per level.

Observable consequence (this test)
----------------------------------
Two groups labeled ``"0"`` and ``"1"`` share the *same* true shape ``sin(2x)``.
Fitting ``y ~ g + s(x, by=g)``:

  * As a **pandas DataFrame** (the #1317-fixed path): both per-group curves
    recover ``sin(2x)`` (range > 1, well correlated).
  * As a **dict of columns** (this bug): the level labeled ``"0"`` is multiplied
    by ``0`` and collapses to a flat line, and the level labeled ``"1"`` carries
    the whole effect — a fit that depends only on the *container type* of the
    otherwise-identical input.

This is a silent, container-dependent corruption that violates the same
invariance #1317 was filed to protect: a model must not depend on the spelling
of categorical levels — nor on whether the caller used a DataFrame or a dict.

The test asserts that the **dict** path recovers BOTH levels' shape. It fails
today because level ``"0"`` is annihilated by the ``x0`` numeric scaling. When
``categorical_dtype_columns`` also marks all-string columns from dict / record /
numpy inputs (the fix), every level gets its own smooth and the assertions hold
without edits.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit


def _range_and_corr(pred: "np.ndarray", mask: "np.ndarray", truth: "np.ndarray") -> tuple[float, float]:
    p = pred[mask]
    rng = float(np.max(p) - np.min(p))
    if rng <= 1e-9:
        return rng, 0.0
    corr = float(np.corrcoef(p, truth[mask])[0, 1])
    return rng, corr


def test_dict_input_by_factor_smooth_recovers_every_numeric_labeled_level() -> None:
    rng = np.random.default_rng(20260621)
    n = 1200
    gi = rng.integers(0, 2, n)
    x = rng.uniform(-3.0, 3.0, n)
    # IDENTICAL true shape for both groups: a correct per-level by-factor smooth
    # must recover the SAME curve for each. A numeric-by misread forces level
    # "k" to amplitude proportional to k, annihilating level "0".
    truth = np.sin(2.0 * x)
    y = truth + rng.normal(0.0, 0.10, n)
    labels = gi.astype(str)  # string labels "0" / "1"

    # A plain dict of columns — a documented, first-class input. ``g`` is a
    # column of *strings*, i.e. a categorical column whose labels happen to look
    # numeric. (Python ``list`` values, the most container-agnostic form.)
    data = {"x": list(map(float, x)), "g": list(map(str, labels)), "y": list(map(float, y))}

    model = gamfit.fit(data, "y ~ g + s(x, by=g)")
    pred = np.asarray(model.predict(data), dtype=float).reshape(-1)

    rng0, corr0 = _range_and_corr(pred, gi == 0, truth)
    rng1, corr1 = _range_and_corr(pred, gi == 1, truth)

    # Both levels share the same true shape sin(2x): each per-level smooth must
    # be a genuine (non-flat) curve that recovers it. Level "0" collapses to a
    # flat line today because the dict's string column is lowered to a numeric
    # by-variable and multiplied by 0.
    assert rng0 > 1.0, (
        f"level '0' by-factor smooth collapsed to a flat line (range={rng0:.3f}); "
        f"the dict's string column was lowered to a numeric by-variable and "
        f"multiplied by 0 (passes as a pandas DataFrame, fails as a dict)"
    )
    assert corr0 > 0.9, f"level '0' smooth does not recover sin(2x) (corr={corr0:.3f})"
    assert rng1 > 1.0 and corr1 > 0.9, (
        f"level '1' smooth not recovered (range={rng1:.3f}, corr={corr1:.3f})"
    )
