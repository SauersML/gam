"""Bug hunt: an intercept-only (covariate-free) GAM cannot be predicted from a
held-out frame that — correctly — omits the response column.

``gamfit.fit(df, "y ~ 1")`` is the canonical *null model*: a valid, documented
baseline (it is exactly what ``compare_models`` and any AIC/likelihood-ratio
comparison fit as the reference). For a Gaussian intercept-only model the fit is
just the sample mean, and predicting it on any ``n``-row frame must return ``n``
copies of that intercept — the row count is the *only* thing the prediction
needs, since the model is a constant function of the covariates.

It does not. Predicting on a held-out frame that carries only covariate columns
(the realistic case: you are predicting, so you do **not** have the response)
aborts before returning anything with::

    GamError: table must have at least one column

The abort is a column-projection defect in the Python (PyFFI) predict path, not
a data problem. ``dataset_with_model_schema``
(``crates/gam-pyffi/src/manifold_and_posterior_ffi.rs:3551``) calls
``project_frame_to_model_columns`` (same file, line 3948), which narrows the
frame to ``prediction_consumable_columns`` = (required prediction columns) ∪
{response}. An intercept-only model references **no** covariate column, so the
consumable set is just ``{"y"}``; a held-out frame that omits ``y`` is projected
to **zero** columns, and ``string_records_from_rows`` (line 3977) then rejects
the now-empty table with the message above. The ``n`` rows that carry the only
information actually needed — the prediction count — are discarded.

The same model, the same frame, fitted and predicted through the ``gam`` CLI
(``gam fit train.csv 'y ~ 1' --out m.gam`` then
``gam predict m.gam new.csv --out p.csv --mode posterior-mean``) succeeds and
writes ``n`` constant predictions equal to the intercept. So the two front-ends
disagree — directly contradicting the projection helper's own docstring, which
states the CLI and PyFFI predict paths "agree exactly". The defect is specific
to the Python predict path on a covariate-free model.

This test fits a deterministic Gaussian intercept-only model, predicts on a
held-out frame containing only a covariate column (no response), and asserts the
prediction is the well-posed constant a null model must produce: ``n`` finite,
identical values equal to the training-response mean (the exact OLS intercept of
``y ~ 1``). It currently fails because ``predict`` raises before any assertion
can run. When the projection stops collapsing a covariate-free model's frame to
zero columns, ``predict`` returns and every assertion below holds without edits.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit


def test_intercept_only_model_predicts_on_response_free_frame() -> None:
    rng = np.random.default_rng(20260618)
    n = 400
    x = rng.uniform(-2.0, 2.0, n)
    y = rng.normal(4.0, 1.0, n)
    train = pd.DataFrame({"x": x, "y": y})

    model = gamfit.fit(train, "y ~ 1")

    # The exact OLS intercept of an (unpenalised) Gaussian `y ~ 1` is mean(y).
    intercept = float(np.mean(y))

    # A held-out frame as a user predicting genuinely-new rows would build it:
    # covariate column(s) only, NO response column (the response is what we are
    # predicting and is unknown).
    new = pd.DataFrame({"x": rng.uniform(-2.0, 2.0, 5)})

    preds = np.asarray(model.predict(new), dtype=float).reshape(-1)

    assert preds.shape[0] == 5, f"expected one prediction per row, got {preds.shape[0]}"
    assert np.all(np.isfinite(preds)), f"predictions must be finite, got {preds}"
    # A null model is constant in the covariates.
    assert float(np.ptp(preds)) <= 1e-9, f"intercept-only predictions must be constant, got {preds}"
    # And that constant is the fitted intercept = training-response mean.
    assert abs(float(preds[0]) - intercept) <= 1e-4, (
        f"intercept-only prediction {float(preds[0])} != fitted intercept {intercept}"
    )
