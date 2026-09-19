"""Bug hunt: a CERTIFIED fit whose smoothing parameter railed to the
infinite-smoothing bound cannot produce prediction intervals, observation
intervals, split-conformal intervals, or diagnostics -- and ``diagnose()`` has
no escape hatch at all.

The canonical trigger is a variance component at the boundary: a random-effect
model ``y ~ group(site)`` where the between-group variance is (near) zero. REML
sends the group lambda to the box bound, and the solver ships the fit with the
plug-in covariance and no smoothing correction. That path is deliberate --
``crates/gam-solve/src/estimate/optimizer.rs:3138-3155``:

    // A fit certified at an infinite-smoothing rail ... is TYPED-unavailable
    // rather than a defect. Ship the certified fit with the plug-in covariance
    // and no correction ... instead of the whole fit dying over an enhancement.

The fit indeed does not die. Everything downstream does. ``summary()`` publishes
finite coefficient standard errors and labels their provenance
(``covariance_kind='conditional'``, ``coefficient_se_source='conditional'``), and
``predict(..., covariance_mode="conditional")`` returns a perfectly ordered band
that even reports ``covariance_source='conditional'`` in its own output. But
every DEFAULT uncertainty surface refuses with the same string:

    predict(interval=0.95)                    GamfitError: ... does not contain smoothing-corrected covariance
    predict(interval=0.95, observation_interval=True)   same
    predict(interval="conformal", calibration=...)      same
    diagnose(data)                                      same

``Model.diagnose(data, *, y=None, interval=0.95)`` takes no ``covariance_mode``,
so the only workaround is ``interval=None``, i.e. giving up the intervals
entirely. The split-conformal band is worse in kind: its documented guarantee is
distribution-free -- "finite-sample marginal coverage >= conformal_level
regardless of model misspecification ... applies to standard GAM models"
(``docs/predictions.md:117-125``) -- yet it is unobtainable by default because a
Bayesian covariance refinement is missing.

This is not a corner case. On a plain 20-group x 10-observation random-effects
design, sweeping 40 seeds per setting, the share of fits whose intervals are
refused is 7/40 at tau/sigma = 0 and 0.05, 4/40 at 0.10, and 2/40 at 0.20 --
i.e. roughly one dataset in six whenever the variance component sits near or
below its own sampling noise.

Observed: a certified, summarisable fit refuses every default uncertainty
surface, and ``diagnose()`` cannot be asked for a weaker one.

Expected: a certified fit reports uncertainty through its default surfaces. The
conditional covariance it already publishes through ``summary()`` (and returns
on request through ``predict``) is available; the correction that is missing is
an enhancement, exactly as the solver comment says.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit

_N = 300
_GROUPS = 6


def _null_variance_component() -> dict[str, Any]:
    """Balanced groups whose between-group variation is exactly zero.

    Removing each group's own mean makes the boundary case exact and therefore
    seed-independent; the sweep quoted in the module docstring shows the same
    refusal at realistic non-zero tau.
    """
    labels = np.array([f"g{i % _GROUPS}" for i in range(_N)])
    y = np.random.default_rng(0).normal(0.0, 1.0, _N)
    for level in np.unique(labels):
        mask = labels == level
        y[mask] -= y[mask].mean()
    return {"g": labels, "y": y}


@pytest.fixture(scope="module")
def railed() -> tuple[Any, dict[str, Any]]:
    data = _null_variance_component()
    model = gamfit.fit(data, "y ~ group(g)", family="gaussian")
    summary = model.summary()
    # Guard the premise: this must actually be the rail-certified state, or the
    # assertions below would be testing nothing.
    assert summary.convergence["certified"] is True
    assert summary.convergence["outer"]["lambdas_railed"] == [0]
    assert summary.covariance_kind == "conditional"
    assert all(np.isfinite(c["std_error"]) for c in summary.coefficients)
    return model, data


def _levels() -> dict[str, Any]:
    return {"g": np.array([f"g{k}" for k in range(_GROUPS)])}


def _assert_usable_band(table: Any, label: str) -> None:
    lower = np.asarray(table["posterior_mean_lower"], dtype=float)
    point = np.asarray(table["posterior_mean"], dtype=float)
    upper = np.asarray(table["posterior_mean_upper"], dtype=float)
    assert np.all(np.isfinite(lower)) and np.all(np.isfinite(upper)), f"{label}: non-finite band"
    assert np.all(lower <= point) and np.all(point <= upper), f"{label}: band excludes the point estimate"
    assert np.all(upper > lower), f"{label}: degenerate zero-width band"


def test_default_prediction_interval_is_available(railed: tuple[Any, dict[str, Any]]) -> None:
    model, _ = railed
    _assert_usable_band(
        model.predict(_levels(), interval=0.95, return_type="dict"),
        "predict(interval=0.95)",
    )


def test_observation_interval_is_available(railed: tuple[Any, dict[str, Any]]) -> None:
    model, _ = railed
    _assert_usable_band(
        model.predict(
            _levels(), interval=0.95, observation_interval=True, return_type="dict"
        ),
        "predict(observation_interval=True)",
    )


def test_split_conformal_guarantee_is_available(railed: tuple[Any, dict[str, Any]]) -> None:
    """The split-conformal band is distribution-free; it must not need Vp."""
    model, data = railed
    _assert_usable_band(
        model.predict(
            _levels(),
            interval="conformal",
            calibration=data,
            conformal_level=0.9,
            return_type="dict",
        ),
        'predict(interval="conformal", calibration=...)',
    )


def test_diagnose_works_at_its_own_default(railed: tuple[Any, dict[str, Any]]) -> None:
    """``diagnose`` has no covariance_mode parameter, so there is no workaround."""
    model, data = railed
    result = model.diagnose(data)
    assert result is not None
