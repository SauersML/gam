"""Bug hunt: a ``Surv(...)`` fit through ``gamfit.fit`` with no explicit
``survival_likelihood`` does NOT default to ``"transformation"`` (the CLI's
default) — it silently routes to the broken ``"location-scale"`` ``FitConfig``
default and aborts the pre-fit identifiability audit on textbook right-censored
data the CLI fits without complaint.

The CLI declares the default formulation explicitly::

    // src/main/cli_args.rs:97
    #[arg(long = "survival-likelihood", default_value = "transformation", ...)]

so `gam fit data.csv 'Surv(time,event) ~ s(x)'` fits via
``likelihood=transformation`` and succeeds. But the Python/FFI path builds the
core ``FitConfig`` through ``resolve_fit_config_from_json``
(``src/config_resolve.rs:220``), which only overrides ``survival_likelihood``
when the JSON value is ``Some``::

    if let Some(mode) = json_config.survival_likelihood {
        fit_config.survival_likelihood = resolve_nonempty_string(mode, ...)?;
    }

When the caller passes no ``survival_likelihood`` (the natural call), the JSON
value is ``None`` and the field keeps the ``FitConfig::default()`` value, which
is::

    // src/solver/workflow/request.rs:1070
    survival_likelihood: "location-scale".into(),

So the Python default is ``location-scale``, NOT ``transformation``. On a
standard exponential right-censored dataset the ``location-scale`` formulation
fails its pre-fit channel-aware identifiability audit::

    IntegrationError: pre-fit channel-aware identifiability audit failed: ...
    FullyAliased { block_idx: 0, reason: "structural residual Gram has no
    positive eigenspace (block of width 0 has zero structural span before any
    anchor exists)" }

while the identical model fit with ``survival_likelihood="transformation"``
(the documented/CLI default) succeeds and produces a valid, monotone survival
surface. The broken ``FitConfig`` default is masked everywhere the CLI is the
entry point and surfaces only on the Python path.

This test asserts the contract the CLI already honors: a plain ``Surv()`` fit
must use the same default formulation as the CLI (``transformation``), succeed
on textbook censored data, and yield the same survival surface as the explicit
``transformation`` fit. It currently fails at the default ``gamfit.fit`` call.

Related: #1123 (the transformation-survival Python path aborting via the inner
solve — a different failure mode of the survival Python path).
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit


def _censored_dataset(seed: int = 4, n: int = 600) -> pd.DataFrame:
    """Textbook proportional-hazards exponential data with right censoring.

    Hazard ``lambda(x) = exp(-0.5 + 0.7 x)`` rises with the covariate, ~50%
    administrative censoring from an independent exponential. This is the
    simplest non-degenerate survival GAM dataset.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    rate = np.exp(-0.5 + 0.7 * x)
    event_time = rng.exponential(1.0 / rate)
    censor_time = rng.exponential(2.0, n)
    observed = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(int)
    return pd.DataFrame({"x": x, "time": observed, "event": event})


def test_default_survival_likelihood_matches_cli_transformation_default() -> None:
    df = _censored_dataset()

    # The documented/CLI default formulation fits this data fine.
    explicit = gamfit.fit(
        df, "Surv(time, event) ~ s(x)", survival_likelihood="transformation"
    )
    assert explicit.is_survival

    # The natural call — no survival_likelihood — MUST behave identically to the
    # CLI, which defaults to "transformation". Today it routes to the broken
    # "location-scale" FitConfig default and raises IntegrationError here.
    default = gamfit.fit(df, "Surv(time, event) ~ s(x)")
    assert default.is_survival, "default Surv() fit must be a survival model"

    # And it must be the SAME fit as the explicit transformation default: equal
    # survival surface on held-out covariate values.
    test = pd.DataFrame({"x": [0.2, 0.8], "time": [0.5, 0.5], "event": [1, 1]})
    surv_default = np.asarray(default.predict(test).survival)
    surv_explicit = np.asarray(explicit.predict(test).survival)

    # A valid survival surface: in [0, 1] and non-increasing in time.
    assert np.all(surv_default >= -1e-9) and np.all(surv_default <= 1.0 + 1e-9)
    assert np.all(np.diff(surv_default, axis=1) <= 1e-9), "S(t) must be non-increasing"

    assert np.allclose(surv_default, surv_explicit, atol=1e-8, rtol=1e-6), (
        "the default Surv() fit must equal the explicit "
        "survival_likelihood='transformation' fit (the CLI default), but the "
        "Python path defaults to the broken 'location-scale' formulation"
    )
