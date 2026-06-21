"""Bug hunt: a categorical *main effect* ``y ~ g`` is silently collapsed to a
single linear NUMERIC term when the data is supplied as a **dict of columns**
(or list of records / numpy) and the level labels are numeric strings
(``"0"``, ``"1"``, ``"2"``) — even though the identical data passed as a pandas
DataFrame fits a correct factor with one mean per level.

Background
----------
Issue #1318 fixed this exact misclassification for the *typed* table path: a
pandas/polars/pyarrow string column whose labels parse as numbers stays a
factor. The fix is the categorical-dtype sentinel in
``gamfit/_tables.py`` (``categorical_dtype_columns``, ``:110``), which only
inspects ``kind in {"pandas", "polars", "pyarrow"}`` and returns an EMPTY set
for every other container (``:152-157``: *"untyped inputs (mappings,
record/row sequences, numpy) return an empty set and keep the value-based
numeric inference"*).

So a plain ``dict`` of columns — a documented, first-class input
(``README``: "dict of columns") — never marks ``g`` categorical. Rust's
column-kind inference (``src/inference/data.rs`` ``all_numeric`` →
``Binary``/``Continuous``) then lowers the all-numeric-looking string column to
a single numeric covariate, and ``y ~ g`` becomes a straight line in the
*integer value of the label* instead of a factor with one coefficient per
level.

This is the main-effect sibling of #1467 (the ``by=`` factor-smooth case);
both stem from the same dict/records/numpy gap in
``categorical_dtype_columns``, but the observable failure and the lowered term
are different (a numeric *linear main effect* vs a numeric *by-variable*).

Observable consequence (this test)
----------------------------------
Three groups with the **non-monotone** mean pattern ``"0" -> 5``,
``"1" -> 0``, ``"2" -> 5`` (a "V" in the label, which no straight line in the
integer label can fit). As a factor, the per-group means are recovered exactly;
lowered to a numeric line the symmetric V collapses to a near-flat fit
(slope ~0, every group predicted near the global mean ~3.3), so the model
explains essentially none of the between-group variance.

The test fits ``y ~ g`` from a **dict** and asserts each level's prediction
recovers its true mean. It fails today because the dict's string column is
lowered to a numeric covariate. When ``categorical_dtype_columns`` also marks
all-string columns from dict / record / numpy inputs (the fix), ``g`` is a
factor and the assertions hold without edits.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit


def test_dict_input_categorical_main_effect_recovers_per_level_means() -> None:
    rng = np.random.default_rng(20260621)
    n = 900
    # Non-monotone "V" means: a straight line in the integer label cannot fit
    # this, but a factor with one coefficient per level fits it exactly.
    means = {"0": 5.0, "1": 0.0, "2": 5.0}
    gi = rng.integers(0, 3, n)
    labels = np.array(["0", "1", "2"])[gi]
    y = np.array([means[s] for s in labels]) + rng.normal(0.0, 0.30, n)

    data = {"g": list(map(str, labels)), "y": list(map(float, y))}
    model = gamfit.fit(data, "y ~ g")

    # Predict one row per level and compare to the true per-level mean.
    preds = {}
    for lvl in ("0", "1", "2"):
        p = np.asarray(model.predict({"g": [lvl]}), dtype=float).reshape(-1)
        preds[lvl] = float(p[0])

    # A correct factor recovers each level mean within ~0.3 (noise sd / sqrt n).
    # The numeric-line misread predicts ~3.3 (the global mean) for every level.
    for lvl in ("0", "1", "2"):
        assert abs(preds[lvl] - means[lvl]) < 0.5, (
            f"level '{lvl}' predicted {preds[lvl]:.3f}, expected ~{means[lvl]}; "
            f"the dict's string column was lowered to a numeric covariate and "
            f"`y ~ g` collapsed to a (flat) line in the integer label "
            f"(this fits correctly as a pandas DataFrame)"
        )

    # And the between-level spread must be recovered, not flattened to ~0.
    spread = max(preds.values()) - min(preds.values())
    assert spread > 3.0, (
        f"per-level predictions collapsed to a near-flat fit (spread={spread:.3f}); "
        f"expected ~5.0 between the V's arms and trough"
    )
