"""Bug hunt: every shape-constrained smooth refuses ``predict(interval=...)``,
``observation_interval``, the split-conformal band and ``diagnose(data)`` at their
DEFAULTS -- and ``diagnose`` has no switch to ask for anything else.

    m = gamfit.fit(d, "y ~ s(x, shape='monotone-increasing')")
    m.summary().covariance_kind          # 'smoothing-corrected'
    m.predict(grid, interval=0.95)
    # GamfitError: prediction failed: Invalid input: inequality-truncated credible
    #   intervals require the persisted conditional posterior; smoothing-corrected
    #   covariance does not define a truncated law

The same message comes back for ``covariance_mode='smoothing'``,
``observation_interval=True``, ``predict(interval="conformal", calibration=...)``
and ``diagnose(data)``.
Only ``covariance_mode='conditional'`` works. It reproduces for
``monotone-increasing``, ``monotone-decreasing``, ``convex`` and ``concave``.

The refusal contradicts the fit that produced the object. ``summary()`` reports
``covariance_kind == 'smoothing-corrected'`` and builds its coefficient standard
errors from that matrix, and the two covariances are materially different -- the
band SEs implied by the published covariance are 8-32% wider than the
conditional ones on this data:

    published-cov band SE  [0.047295 0.043895 0.044395 0.038736 0.043404 0.041079 0.041842]
    conditional   band SE  [0.043719 0.034050 0.034028 0.031859 0.032962 0.032363 0.038488]

So the user gets three different stories from one fitted model: ``summary()``
answers from Vp, ``predict(covariance_mode='conditional')`` answers from Vb, and
the default surfaces raise.

Mechanism: ``crates/gam-predict/src/lib.rs:2736-2742`` in
``predict_gamwith_uncertainty`` --

    let constrained_fit = source.constrained_fit_result();
    if constrained_fit.is_some() {
        if requested_mode != InferenceCovarianceMode::Conditional {
            return Err(EstimationError::InvalidInput(
                "inequality-truncated credible intervals require the persisted conditional \
                 posterior; smoothing-corrected covariance does not define a truncated law"));
        }

and ``covariance_mode`` defaults to ``None`` -> ``SmoothingCorrected``
(``crates/gam-predict/src/lib.rs:303-348``; ``docs/predictions.md:27``: "``None``
requires smoothing-corrected covariance"), so the default is always the refused
one.

That gate also contradicts the fit-side assembly it is refusing.
``crates/gam-solve/src/estimate/optimizer.rs:3473-3510`` deliberately truncates
the smoothing-corrected covariance to the feasible set and argues at length that
the result IS a truncated law:

    // The feasible set constrains β and says nothing about ρ, so the indicator
    // 1_C(β) factors straight out of the ρ-integral ... i.e. the β-marginal of
    // the TRUNCATED joint posterior is exactly the truncation of the β-marginal
    // of the untruncated one. So the truncation belongs on `Vp`, applied last ...
    // the published matrix is a genuine truncated-Gaussian covariance

``Model.diagnose(data, *, y=None, interval=0.95)`` exposes no ``covariance_mode``
at all, so its only escape is ``interval=None`` -- giving up the intervals.
The split-conformal band's guarantee is distribution-free by construction
(``docs/predictions.md:117-125``), so losing it to a Bayesian-covariance gate is
the sharpest instance.

Observed: every default uncertainty surface on a shape-constrained fit raises.

Expected: they return a band, from whichever covariance the project decides a
truncated fit publishes -- the same one ``summary()`` already reports.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit

_GRID = {"x": np.linspace(0.05, 0.95, 7)}
_SHAPES = ["monotone-increasing", "monotone-decreasing"]


def _data() -> dict[str, Any]:
    rng = np.random.default_rng(4)
    x = np.sort(rng.uniform(0.0, 1.0, 400))
    return {"x": x, "y": 3.0 * x * x + 0.3 * rng.standard_normal(x.size)}


def _fit(shape: str | None) -> Any:
    formula = "y ~ s(x)" if shape is None else f"y ~ s(x, shape='{shape}')"
    return gamfit.fit(_data(), formula, family="gaussian")


def _assert_usable_band(table: Any, label: str) -> None:
    lower = np.asarray(table["posterior_mean_lower"], dtype=float)
    point = np.asarray(table["posterior_mean"], dtype=float)
    upper = np.asarray(table["posterior_mean_upper"], dtype=float)
    assert np.all(np.isfinite(lower)) and np.all(np.isfinite(upper)), f"{label}: non-finite band"
    assert np.all(lower <= point) and np.all(point <= upper), f"{label}: band excludes the point"
    assert np.all(upper > lower), f"{label}: degenerate band"


def test_unconstrained_control_supports_every_default_surface() -> None:
    """Green today: nothing about the data or grid is the problem."""
    model = _fit(None)
    data = _data()
    _assert_usable_band(model.predict(_GRID, interval=0.95, return_type="dict"), "predict")
    _assert_usable_band(
        model.predict(_GRID, interval=0.95, observation_interval=True, return_type="dict"),
        "observation_interval",
    )
    _assert_usable_band(
        model.predict(
            _GRID, interval="conformal", calibration=data, conformal_level=0.9, return_type="dict"
        ),
        "split conformal",
    )
    assert model.diagnose(data) is not None


@pytest.mark.parametrize("shape", _SHAPES)
def test_shape_constrained_default_prediction_interval(shape: str) -> None:
    model = _fit(shape)
    _assert_usable_band(
        model.predict(_GRID, interval=0.95, return_type="dict"),
        f"{shape}: predict(interval=0.95)",
    )


@pytest.mark.parametrize("shape", _SHAPES)
def test_shape_constrained_observation_interval(shape: str) -> None:
    model = _fit(shape)
    _assert_usable_band(
        model.predict(_GRID, interval=0.95, observation_interval=True, return_type="dict"),
        f"{shape}: observation_interval",
    )


@pytest.mark.parametrize("shape", _SHAPES)
def test_shape_constrained_split_conformal(shape: str) -> None:
    """Split conformal is distribution-free; it must not need a Bayesian Vp."""
    model = _fit(shape)
    _assert_usable_band(
        model.predict(
            _GRID, interval="conformal", calibration=_data(), conformal_level=0.9, return_type="dict"
        ),
        f"{shape}: split conformal",
    )


@pytest.mark.parametrize("shape", _SHAPES)
def test_shape_constrained_diagnose_at_its_default(shape: str) -> None:
    """``diagnose`` has no covariance_mode parameter, so there is no workaround."""
    model = _fit(shape)
    assert model.diagnose(_data()) is not None
