"""gam#3298: a link spelling the fit never reads is refused by name.

The multinomial family fixes its softmax link, and the Weibull survival
likelihood has no link at all. Each accepted ``link=``, ``flexible_link=True``
or a formula ``link(...)`` and fit bit-identically without it, so a user who
asked for a probit multinomial got a logit one with no word. Each spelling is
now refused, naming the spelling.
"""

import numpy as np
import pytest

import gamfit


def _multinomial_data() -> dict[str, list]:
    rng = np.random.default_rng(3298)
    x = rng.uniform(-1.0, 1.0, size=240)
    logits = np.column_stack([np.zeros_like(x), 1.5 * x, -1.5 * x])
    probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    labels = np.array(["a", "b", "c"])
    y = [labels[rng.choice(3, p=p)] for p in probabilities]
    return {"x": list(map(float, x)), "y": y}


def _survival_data() -> dict[str, list]:
    rng = np.random.default_rng(32980)
    x = rng.uniform(-1.0, 1.0, size=240)
    time = rng.weibull(1.5, size=240) * np.exp(-0.5 * x)
    event = (rng.uniform(size=240) < 0.8).astype(float)
    return {"x": list(map(float, x)), "time": list(map(float, time)), "event": list(event)}


@pytest.mark.parametrize(
    ("formula", "options", "spelling"),
    [
        ("y ~ s(x)", {"link": "probit"}, 'link="probit"'),
        ("y ~ s(x)", {"flexible_link": True}, "flexible_link=True"),
        ("y ~ s(x) + link(type=probit)", {}, "link(type=probit)"),
    ],
)
def test_the_multinomial_family_refuses_every_link_spelling(formula, options, spelling) -> None:
    data = _multinomial_data()
    with pytest.raises(gamfit.GamfitError, match="multinomial") as refused:
        gamfit.fit(data, formula, family="multinomial", **options)
    assert spelling in str(refused.value)
    # The same fit without the spelling succeeds.
    gamfit.fit(data, "y ~ s(x)", family="multinomial")


@pytest.mark.parametrize(
    ("options", "spelling"),
    [
        ({"link": "logit"}, 'link="logit"'),
        ({"flexible_link": True}, "flexible_link=True"),
    ],
)
def test_the_weibull_survival_likelihood_refuses_every_link_spelling(options, spelling) -> None:
    data = _survival_data()
    with pytest.raises(gamfit.GamfitError, match="weibull") as refused:
        gamfit.fit(data, "Surv(time, event) ~ s(x)", survival_likelihood="weibull", **options)
    assert spelling in str(refused.value)
    gamfit.fit(data, "Surv(time, event) ~ s(x)", survival_likelihood="weibull")


@pytest.mark.parametrize(
    ("formula", "options"),
    [
        ("Surv(time, event) ~ x + link(type=loglog)", {"link": "flexible(loglog)"}),
        ("Surv(time, event) ~ x + link(type=cauchit)", {"flexible_link": True}),
        ("Surv(time, event) ~ x", {"link": "loglog", "flexible_link": True}),
    ],
)
def test_survival_location_scale_reads_a_flexible_request_from_every_spelling(
    formula, options
) -> None:
    # A flexed survival loglog/cauchit link is refused, so the refusal shows the
    # flexible request was read. Before, the location-scale fit took its link
    # choice from the formula's name alone, dropped a `flexible(...)` in `link=`
    # beside it, and fitted the plain link.
    data = _survival_data()
    with pytest.raises(gamfit.GamfitError, match="single-component mixture"):
        gamfit.fit(data, formula, survival_likelihood="location-scale", **options)
