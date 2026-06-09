"""Bug hunt: a purely-parametric survival model ``Surv(...) ~ x`` cannot be
predicted — ``predict`` raises a covariate-design dimension mismatch.

Fitting ``Surv(entry, exit, event) ~ x`` succeeds, but ``model.predict(...)``
(even on the training data) raises::

    GamError: failed to build survival prediction design:
    Dimension mismatch: linear term 'x' feature column 3 out of bounds for 3 columns

For a table ``[entry, exit, event, x]`` the linear term ``x`` is replayed at its
absolute training-table index (3), but the data handed to
``build_term_collection_design_inner`` has only 3 columns (the response/event
column is dropped without re-indexing the linear-term feature columns), so the
bounds check added in commit f7bc5eb7b bails. A second covariate shifts the
failure to index 4 ("out of bounds for 4 columns"), confirming the off-by-the-
dropped-column indexing rather than a one-off.

Root cause (files read, no patch):

* ``src/families/survival_predict.rs:281`` builds the covariate design via
  ``build_term_collection_design(cov_input, &termspec)`` where ``termspec`` is
  ``resolve_termspec_for_prediction(...)`` (lines ~2057-2093). The remapped
  linear-term ``feature_col`` indexes a wider table than the array actually
  passed to ``build_term_collection_design_inner``
  (``src/terms/smooth.rs``), whose bounds check then fails:
  ``"linear term '{}' feature column {} out of bounds for {} columns"``.

This reproduces for ``weibull``, ``transformation`` and ``location-scale`` modes
whenever the formula has only parametric (non-smooth) covariates. Adding a
smooth (e.g. ``~ x + s(age)``) takes a different design path and avoids this
particular crash.

This test is intentionally narrow: it asserts only that prediction *succeeds*
and returns a finite survival surface — it does not assert the values are
well-calibrated (the Weibull surface degeneracy is tracked separately).

Related: #896, #897

When the parametric survival predict design is built consistently with the
runtime column layout, this test passes without edits.
"""

import importlib
from typing import Any, cast

pytest = cast(Any, importlib.import_module("pytest"))
pytest.importorskip("gamfit._rust")

import numpy as np
import pandas as pd

import gamfit


def _make_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(9)
    n = 300
    x = rng.normal(0.0, 1.0, n)
    eta = -1.5 + 0.7 * x
    u = rng.uniform(1e-9, 1.0, n)
    t_lat = np.exp(-eta / 1.5) * (-np.log(u)) ** (1.0 / 1.5)
    cens = np.minimum(rng.exponential(12.0, n), 20.0)
    exit_t = np.minimum(t_lat, cens)
    event = (t_lat <= cens).astype(int)
    return pd.DataFrame(
        {"entry": np.zeros(n), "exit": exit_t, "event": event, "x": x}
    )


def test_parametric_only_survival_model_is_predictable() -> None:
    df = _make_dataset()
    model = gamfit.fit(
        df,
        "Surv(entry, exit, event) ~ x",
        survival_likelihood="weibull",
    )

    # Predicting on the training frame must not raise: every required column is
    # present, so a fitted survival model has to be evaluable on its own data.
    pred = model.predict(df)

    survival = np.asarray(pred.survival, dtype=float)
    assert survival.ndim == 2
    assert survival.shape[0] == len(df)
    assert np.all(np.isfinite(survival)), "survival surface contains non-finite values"
    assert np.all((survival >= -1e-9) & (survival <= 1.0 + 1e-9)), (
        "survival surface escaped [0, 1]"
    )
