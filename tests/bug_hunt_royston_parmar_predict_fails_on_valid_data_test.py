"""Bug hunt: a fitted Royston-Parmar survival model cannot be predicted.

``gamfit.fit(df, "Surv(time, event) ~ x", family="royston-parmar")`` is a
documented, advertised family (it appears in the CLI ``--family`` enum, in
``docs/families-and-links.md``, and in ``docs/posterior-sampling.md``). The fit
succeeds and reports ``family_name == "Royston Parmar"``. The *only* thing a
user does next with a survival fit is evaluate the survival surface, exactly as
``docs/survival.md`` shows::

    pred = model.predict(test_df)
    S = pred.survival_at([1, 5, 10, 20])

For a genuine Royston-Parmar model this is impossible: ``model.predict(...)``
aborts before returning anything with

    GamError: survival ages must be finite and positive for baseline hazard
              evaluation

even though every query row carries a finite, strictly-positive ``time``. The
abort comes from the ``age <= 0.0`` guard in ``validated_baseline_params``
(``src/families/survival_construction.rs:2382``): the genuine RP predict path
evaluates the parametric baseline-hazard term at a non-positive age (the
``t = 0`` survival-curve origin / the ``age_entry == 0`` left-truncation
boundary) and the guard rejects it.

The defect is specific to the genuine ``ResponseFamily::RoystonParmar`` model
that is reachable *only* through the Python ``family="royston-parmar"`` path.
The same data + formula fit with ``survival_likelihood="transformation"`` (or a
``--baseline-target weibull`` CLI fit) predicts fine and produces a valid
survival surface — so this is not a data problem and not an intrinsic property
of the time grid; it is a defect in the RP predict path itself.

This test fits a deterministic Weibull-generated survival dataset as a
Royston-Parmar model, predicts on held-out covariates at valid positive query
times, and asserts the survival surface is a well-posed survival function:
finite, in ``[0, 1]``, monotone non-increasing in time, and ordered correctly
across the covariate (a longer Weibull scale must give a higher survival
probability). It currently fails because ``predict`` raises before any of these
checks can run. When the RP predict path stops evaluating the baseline at a
non-positive age, ``predict`` returns and every assertion below holds without
edits.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit


def _weibull_survival_frame(seed: int = 20260611, n: int = 600) -> pd.DataFrame:
    """A right-censored Weibull survival dataset with a covariate that shifts
    the (log) scale, so survival is monotone increasing in ``x``."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    scale = np.exp(0.5 + 1.0 * x)  # longer scale -> longer survival
    shape = 1.3
    event_time = scale * rng.weibull(shape, n)
    censor_time = rng.exponential(6.0, n)
    observed = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(float)
    return pd.DataFrame({"time": observed, "event": event, "x": x})


def test_royston_parmar_model_can_be_predicted() -> None:
    df = _weibull_survival_frame()

    model = gamfit.fit(df, "Surv(time, event) ~ x", family="royston-parmar")
    # Sanity: the fit really is the genuine Royston-Parmar family.
    assert model.family_name == "Royston Parmar"

    # Two held-out covariate rows: a low-scale and a high-scale subject. The
    # `time`/`event` columns are required by the survival predict contract; the
    # actual survival surface is queried via `survival_at` below.
    new_data = pd.DataFrame(
        {"time": [3.0, 3.0], "event": [1.0, 1.0], "x": [0.2, 0.8]}
    )

    # This is the line that currently raises
    #   GamError: survival ages must be finite and positive for baseline hazard
    #             evaluation
    # despite every query time being finite and strictly positive.
    prediction = model.predict(new_data)

    query_times = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    survival = np.asarray(prediction.survival_at(query_times), dtype=float)
    assert survival.shape == (2, query_times.size)

    # A survival function is finite and lives in [0, 1].
    assert np.all(np.isfinite(survival))
    assert np.all(survival >= -1e-9)
    assert np.all(survival <= 1.0 + 1e-9)

    # S(t) is monotone non-increasing in t for every subject.
    assert np.all(np.diff(survival, axis=1) <= 1e-9)

    # The high-scale subject (x = 0.8) must have a higher survival probability
    # than the low-scale subject (x = 0.2) at an interior time — the covariate
    # effect the model was given.
    assert survival[1, 3] > survival[0, 3]
