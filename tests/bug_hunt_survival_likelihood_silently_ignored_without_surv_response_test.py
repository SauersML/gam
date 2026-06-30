"""``survival_likelihood=`` is silently ignored when the formula response is not
``Surv(...)`` — the fit degrades to an ordinary Gaussian GAM with no error and no
warning.

What happens
------------
A user who forgets the ``Surv(time, event)`` wrapper but explicitly asks for a
survival likelihood:

```python
gamfit.fit(data, "time ~ s(x)", survival_likelihood="weibull")
```

does **not** get a Weibull survival model, an error, or a warning. They get a
plain ``standard`` / ``Gaussian Identity`` GAM that regresses the raw event-time
column on ``x``, ignoring censoring entirely — yet the call reports success:

```
survival_likelihood='weibull'         -> model_class='standard'  family='Gaussian Identity'
survival_likelihood='location-scale'  -> model_class='standard'  family='Gaussian Identity'
survival_likelihood='latent'          -> model_class='standard'  family='Gaussian Identity'
Surv(time,event) ~ s(x), weibull      -> model_class='survival'   (correct)
```

Every downstream number (coefficients, predictions, SEs, AIC) is then wrong for
the model the user believes they fitted, with nothing to signal the mistake.

The asymmetry that proves it is a bug
-------------------------------------
The sibling survival entry point *does* validate the same misconfiguration:

```python
gamfit.fit(data, "time ~ s(x)", family="royston-parmar")   # raises IntegrationError
```

so the engine is perfectly capable of rejecting a survival request whose response
is not a survival response — the ``survival_likelihood=`` knob just isn't wired
into that check.

Root cause (files/lines read)
-----------------------------
``materialize`` in
``crates/gam-models/src/fit_orchestration/entry.rs`` routes to
``materialize_survival`` *only* when the response parses as ``Surv(...)`` /
``SurvInterval(...)`` (≈ lines 1048, 1075). Otherwise it takes the non-survival
``else`` branch (≈ line 1095), whose single chokepoint guard is
``reject_survival_only_terms_for_nonsurvival(&parsed)`` (defined in
``crates/gam-models/src/fit_orchestration/materialize/validation.rs:36``). That
guard rejects survival-only *formula terms* (``timewiggle(...)`` / ``survmodel(...)``)
— added for exactly this failure mode in #371, whose comment states that without
it "every non-survival materializer below would silently drop them, fitting an
ordinary GAM while the user believes they requested a time-varying / survival
model". But ``config.survival_likelihood`` is a *config* knob, not a formula
term, so it slips past this guard and is consulted *only* inside
``materialize_survival`` (``materialize/survival.rs``), which the non-``Surv()``
branch never reaches. ``survival_likelihood`` is therefore silently discarded.
The symmetric fix is to reject (or honor) a non-default
``config.survival_likelihood`` at this same chokepoint, exactly as the
survival-only formula terms are rejected.

What this test asserts
----------------------
For each non-default survival likelihood, fitting a non-``Surv()`` response with
``survival_likelihood=<mode>`` must NOT silently return a non-survival standard
Gaussian model: the call must either raise a configuration error or produce an
actual survival model. A positive control confirms that the *same*
``survival_likelihood`` does produce a survival model once the response is
wrapped in ``Surv(time, event)`` (so a fix cannot simply break the happy path).
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

pytest: Any = importlib.import_module("pytest")

pytest.importorskip("gamfit._rust")

import gamfit


def _survival_frame(seed: int = 20260630, n: int = 500) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    scale = np.exp(0.5 + 1.0 * x)
    event_time = scale * rng.weibull(1.4, n)
    censor_time = rng.exponential(6.0, n)
    time = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(float)
    return {"time": list(time), "event": list(event), "x": list(x)}


# Non-default survival formulations (the default is "transformation", which a
# non-Surv() response cannot be distinguished from "unset", so it is excluded).
NON_DEFAULT_LIKELIHOODS = ["weibull", "location-scale", "latent"]


def test_survival_likelihood_not_silently_ignored_without_surv_response() -> None:
    data = _survival_frame()

    offenders: list[str] = []
    for mode in NON_DEFAULT_LIKELIHOODS:
        try:
            model = gamfit.fit(data, "time ~ s(x)", survival_likelihood=mode)
        except Exception:
            # A clean configuration error is an acceptable outcome: the survival
            # request was *not* silently dropped.
            continue
        # The fit "succeeded" — it must then be an actual survival model, not a
        # plain Gaussian GAM that ignored the survival likelihood.
        if not model.is_survival:
            offenders.append(
                f"survival_likelihood={mode!r} -> model_class={model.model_class!r}, "
                f"family={model.family_name!r}"
            )

    assert not offenders, (
        "survival_likelihood= was silently ignored on a non-Surv() response — the "
        "fit degraded to an ordinary Gaussian GAM with no error or warning, so "
        "every downstream estimate is wrong for the requested model. "
        "Offending fits: " + "; ".join(offenders) + ". The non-survival branch of "
        "materialize() (crates/gam-models/src/fit_orchestration/entry.rs ~line 1095) "
        "rejects survival-only *formula terms* via "
        "reject_survival_only_terms_for_nonsurvival (#371) but not the "
        "config.survival_likelihood knob; family='royston-parmar' on the same "
        "formula DOES raise, confirming the missing symmetric guard."
    )


def test_surv_response_positive_control_still_builds_a_survival_model() -> None:
    # Happy path: with the Surv(...) wrapper, survival_likelihood is honored and
    # the model is a genuine survival model. Guards against a fix that merely
    # breaks every survival request.
    data = _survival_frame()
    model = gamfit.fit(data, "Surv(time, event) ~ s(x)", survival_likelihood="weibull")
    assert model.is_survival, (
        "positive control failed: Surv(time, event) ~ s(x) with "
        f"survival_likelihood='weibull' produced a non-survival "
        f"{model.model_class!r} / {model.family_name!r} model"
    )


if __name__ == "__main__":  # pragma: no cover - manual smoke run
    test_survival_likelihood_not_silently_ignored_without_surv_response()
    test_surv_response_positive_control_still_builds_a_survival_model()
    print("ok")
