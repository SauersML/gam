"""Bug hunt: the documented competing-risks survival fit is unusable.

`docs/survival.md` advertises a joint competing-risks fit: "If the fitted event
column contains contiguous positive event codes ``1..K``, ``Model.predict(...)``
returns a ``CompetingRisksPrediction`` directly", and "joint competing-risks fit
for the transformation and Weibull modes." The entry point is

    gamfit.fit(df, "Surv(time, event) ~ x", survival_likelihood="transformation")

with ``event`` in ``{0 (censored), 1, 2, ...}``.

On clean, well-balanced two-cause data this **always** aborts at fit time:

    IntegrationError: cause-specific survival custom-family fit failed:
      identifiability audit refused the fit: identifiability audit: 2 block(s),
      28 joint columns, joint rank 28, 38 alias pair(s) above leverage-based
      report threshold, 0 dropped column(s) — FATAL: alias pair:
      'time_cause_1'[13] ~ 'time_cause_2'[13] overlap=1.0000 >= leverage-based
      halt half-width 0.3171 ...

The two cause-specific baselines share the *same* I-spline time basis evaluated
at the *same* observed event times, so their time-basis columns coincide in the
joint design (``time_cause_1[j] == time_cause_2[j]``). The cause-specific
likelihood separates the two baselines via the per-cause risk sets, so the model
is identifiable — and the audit itself reports **joint rank 28 of 28 columns,
0 dropped** (i.e. the joint design is full column rank, no exact rank
deficiency). Yet the leverage-based alias gate halts the fit on the
near-collinear cause-baseline pair anyway. The result: the documented
competing-risks fit cannot be run, in *any* mode (transformation, weibull, or a
parametric baseline), and there is no user knob to orthogonalise the shared time
direction.

This test fits a deterministic two-cause dataset and asserts the fitted model
yields a well-posed competing-risks prediction: a valid overall survival
function and per-cause cumulative-incidence functions that obey the total
probability identity

    S_overall(t) + sum_k CIF_k(t) = 1

(everyone is, at any time t, either still at risk or has already failed from
exactly one of the K causes). It fails today at the ``fit`` call. When the
identifiability gate stops halting this full-rank, penalised, cause-specific
model, the fit succeeds and the survival/CIF identity holds without edits.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit


def _two_cause_frame(seed: int = 20260611, n: int = 700) -> pd.DataFrame:
    """Two competing causes whose cause-specific log-hazards move in opposite
    directions with the covariate, plus independent right-censoring."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    scale1 = np.exp(0.2 + 0.7 * x)
    scale2 = np.exp(-0.2 - 0.7 * x)
    t1 = scale1 * rng.weibull(1.2, n)
    t2 = scale2 * rng.weibull(1.4, n)
    censor = rng.exponential(8.0, n)
    observed = np.minimum.reduce([t1, t2, censor])
    event = np.where(observed == censor, 0, np.where(t1 < t2, 1, 2)).astype(float)
    return pd.DataFrame({"time": observed, "event": event, "x": x})


def test_competing_risks_transformation_fit_succeeds_and_is_consistent() -> None:
    df = _two_cause_frame()
    # Sanity: a genuine 0/1/2 competing-risks coding (both causes well populated).
    counts = {int(k): int(v) for k, v in zip(*np.unique(df["event"], return_counts=True))}
    assert set(counts) == {0, 1, 2}
    assert counts[1] > 50 and counts[2] > 50

    # This is the line that currently raises IntegrationError (identifiability
    # audit refused the fit) — the documented competing-risks entry point.
    model = gamfit.fit(
        df, "Surv(time, event) ~ x", survival_likelihood="transformation"
    )

    new_data = pd.DataFrame(
        {"time": [3.0, 3.0], "event": [1.0, 1.0], "x": [0.2, 0.8]}
    )
    pred = model.predict(new_data)

    n_rows = len(new_data)
    endpoints = list(pred.endpoint_names)
    k = len(endpoints)
    assert k == 2

    times = np.asarray(pred.times, dtype=float)
    overall = np.asarray(pred.overall_survival, dtype=float)  # (n_rows, n_times)
    cif = np.asarray(pred.cif, dtype=float)                   # (K * n_rows, n_times)
    assert overall.shape == (n_rows, times.size)
    assert cif.shape == (k * n_rows, times.size)
    cif = cif.reshape(k, n_rows, times.size)

    # Overall survival: finite, in [0, 1], monotone non-increasing in time.
    assert np.all(np.isfinite(overall))
    assert np.all(overall >= -1e-9) and np.all(overall <= 1.0 + 1e-9)
    assert np.all(np.diff(overall, axis=1) <= 1e-9)

    # Each cause CIF: finite, in [0, 1], monotone non-decreasing in time.
    assert np.all(np.isfinite(cif))
    assert np.all(cif >= -1e-9) and np.all(cif <= 1.0 + 1e-9)
    assert np.all(np.diff(cif, axis=2) >= -1e-9)

    # The competing-risks total-probability identity:
    #   S_overall(t) + sum_k CIF_k(t) = 1   for every subject and time.
    total = overall + cif.sum(axis=0)
    assert np.max(np.abs(total - 1.0)) < 1e-3
