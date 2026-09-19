"""#2781 regression, approached through spelling-equivalence rather than seams.

``tests/bug_hunt_period_declaration_silently_inert_without_periodic_flag_test.py``
pins the symptom: a period declared without the separate ``periodic=`` switch
used to produce a fit bit-identical to the plain aperiodic one, seam gap and
all.  Closing the seam is necessary but not sufficient -- an implementation
could close it by, say, wrapping at the observed data range and still be wrong
about the declared period.

So this file asserts the rule instead: **a declared period is the periodicity
declaration**, therefore

* the inferred spelling must be *bit-identical* to the explicit
  ``periodic=``/``bc='periodic'`` one on the same period -- they are two
  spellings of one model, not two models that happen to both wrap;
* the fitted function must be genuinely 24-periodic far outside the training
  range, not merely continuous at one seam;
* on a tensor, only the margin that was named wraps -- the other must keep its
  boundary;

and that the four declarations which name no axis are refused with a message
that says which key and why, rather than being discarded.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit

_PERIOD = 24.0


def _hourly() -> dict[str, Any]:
    rng = np.random.default_rng(2781)
    t = rng.uniform(0.0, _PERIOD, 900)
    y = (
        2.0 * np.sin(2.0 * np.pi * t / _PERIOD)
        + np.cos(4.0 * np.pi * t / _PERIOD)
        + 0.3 * rng.standard_normal(t.size)
    )
    return {"t": t, "y": y}


def _cylinder() -> dict[str, Any]:
    rng = np.random.default_rng(1781)
    theta = rng.uniform(0.0, 2.0 * np.pi, 700)
    h = rng.uniform(0.0, 1.0, theta.size)
    return {
        "theta": theta,
        "h": h,
        "y": np.sin(theta) + 0.5 * h + 0.2 * rng.standard_normal(theta.size),
    }


def _mean(model: Any, frame: dict[str, Any]) -> Any:
    return np.asarray(
        model.predict(frame, return_type="dict")["posterior_mean"], dtype=float
    )


@pytest.mark.parametrize(
    "inferred",
    ["y ~ s(t, period=24)", "y ~ s(t, periods=24)"],
)
def test_inferred_and_explicit_spellings_are_the_same_model(inferred: str) -> None:
    data = _hourly()
    grid = {"t": np.linspace(0.0, _PERIOD, 97)}
    explicit = gamfit.fit(data, "y ~ s(t, periodic=true, period=24)", family="gaussian")
    implied = gamfit.fit(data, inferred, family="gaussian")

    assert float(implied.summary().deviance) == float(explicit.summary().deviance), (
        f"{inferred} must be the SAME model as the periodic= spelling, not merely "
        "another one that also wraps"
    )
    assert np.array_equal(_mean(implied, grid), _mean(explicit, grid))


def test_declared_period_makes_the_function_periodic_beyond_the_data() -> None:
    """Closing one seam is not periodicity; f(t) == f(t + 24) everywhere is."""
    model = gamfit.fit(_hourly(), "y ~ s(t, period=24)", family="gaussian")
    t = np.linspace(-3.0 * _PERIOD, 4.0 * _PERIOD, 211)
    base = _mean(model, {"t": t})
    for shift in (-2.0, -1.0, 1.0, 3.0):
        shifted = _mean(model, {"t": t + shift * _PERIOD})
        assert float(np.max(np.abs(shifted - base))) < 1e-9, (
            f"a period of 24 must hold under a shift of {shift} periods"
        )


def test_endpoint_spelling_declares_the_domain_it_names() -> None:
    """`period_start`/`period_end` pin the domain, not just the length."""
    data = _hourly()
    endpoints = gamfit.fit(
        data, "y ~ s(t, period_start=0, period_end=24)", family="gaussian"
    )
    origin_and_length = gamfit.fit(
        data, "y ~ s(t, origin=0, period=24)", family="gaussian"
    )
    grid = {"t": np.linspace(0.0, _PERIOD, 97)}
    assert np.array_equal(_mean(endpoints, grid), _mean(origin_and_length, grid)), (
        "[0, 24) declared by its endpoints and by origin+length is one domain"
    )


def test_tensor_period_wraps_only_the_margin_it_names() -> None:
    data = _cylinder()
    implied = gamfit.fit(data, "y ~ te(theta, h, periods=[2*pi, None])", family="gaussian")
    explicit = gamfit.fit(
        data,
        "y ~ te(theta, h, periodic=[true,false], period=[2*pi,None])",
        family="gaussian",
    )
    assert float(implied.summary().deviance) == float(explicit.summary().deviance)

    # theta wraps...
    theta_seam = {"theta": np.array([0.0, 2.0 * np.pi]), "h": np.array([0.4, 0.4])}
    seam = _mean(implied, theta_seam)
    assert abs(float(seam[0] - seam[1])) < 1e-9

    # ...and h, which was declared `None`, does NOT: an h-margin that had been
    # made periodic too would tie its two ends together.
    h_ends = {"theta": np.array([1.0, 1.0]), "h": np.array([0.0, 1.0])}
    ends = _mean(implied, h_ends)
    assert abs(float(ends[0] - ends[1])) > 1e-3, (
        "the h margin was declared aperiodic and must not wrap"
    )


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("y ~ s(t, origin=0)", "declares no period"),
        ("y ~ s(t, periodic=false, period=24)", "denies the periodicity"),
    ],
)
def test_one_dimensional_unconsumable_declarations_are_refused(
    formula: str, expected: str
) -> None:
    with pytest.raises(gamfit.GamfitError) as excinfo:
        gamfit.fit(_hourly(), formula, family="gaussian")
    assert expected in str(excinfo.value), f"got: {excinfo.value}"


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("y ~ te(theta, h, period=2*pi)", "does not say which"),
        ("y ~ te(theta, h, period_start=0, period_end=6.283185307179586)", "periods="),
        ("y ~ te(theta, h, origins=[0, None])", "declares no period"),
    ],
)
def test_tensor_unconsumable_declarations_are_refused(
    formula: str, expected: str
) -> None:
    with pytest.raises(gamfit.GamfitError) as excinfo:
        gamfit.fit(_cylinder(), formula, family="gaussian")
    assert expected in str(excinfo.value), f"got: {excinfo.value}"
