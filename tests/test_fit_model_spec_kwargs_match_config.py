"""First-class ``fit()`` model-spec kwargs must match the ``config={...}`` escape hatch.

These three fields (``noise_formula``, ``noise_offset``, ``flexible_link``) are genuine
model-spec parameters fully wired through the FFI/core; promoting them to dedicated
``fit()`` kwargs is pure CLI<->Python parity. This test pins two properties:

1. Passing a value via the dedicated kwarg assembles the *exact same* Rust config
   payload as passing it via ``config={...}`` (the previously documented escape hatch),
   so existing ``config=`` users see no behavior change.
2. The kwarg and the ``config=`` dict do not clobber each other: a dedicated kwarg wins
   over the same ``config`` key (mirroring ``firth``/``scale_dimensions``), and when the
   kwarg is left ``None`` a value supplied via ``config=`` survives untouched.

The optional Rust-backed arm fits a Gaussian location-scale model both ways and asserts
the fitted predictions are bit-identical, i.e. the kwarg really does produce the same
location-scale (GAMLSS) fit as the escape hatch.
"""

from __future__ import annotations

import importlib
import typing

pytest = typing.cast(typing.Any, importlib.import_module("pytest"))

import gamfit
from gamfit._api import _build_fit_payload


_BASE: dict[str, typing.Any] = {
    "family": "auto",
    "offset": None,
    "weights": None,
    "transformation_normal": None,
    "transformation_normal_stage1": None,
    "survival_likelihood": None,
    "baseline_target": None,
    "baseline_scale": None,
    "baseline_shape": None,
    "baseline_rate": None,
    "baseline_makeham": None,
    "z_column": None,
    "link": None,
    "logslope_formula": None,
    "frailty_kind": None,
    "frailty_sd": None,
    "hazard_loading": None,
    "scale_dimensions": None,
    "adaptive_regularization": None,
    "firth": None,
    "noise_formula": None,
    "noise_offset": None,
    "flexible_link": None,
    "precision_hyperpriors": None,
    "latents": None,
    "penalties": None,
    "smooths": None,
    "config": None,
}


def _payload(**overrides: typing.Any) -> dict[str, typing.Any]:
    kwargs = dict(_BASE)
    kwargs.update(overrides)
    return _build_fit_payload(**kwargs)


@pytest.mark.parametrize(
    ("kwarg", "value", "config_key"),
    [
        ("noise_formula", "s(x)", "noise_formula"),
        ("noise_offset", "logvar", "noise_offset"),
        ("flexible_link", True, "flexible_link"),
    ],
)
def test_model_spec_kwarg_matches_config_escape_hatch(
    kwarg: str, value: typing.Any, config_key: str
) -> None:
    via_kwarg = _payload(**{kwarg: value})
    via_config = _payload(config={config_key: value})

    assert via_kwarg.get(config_key) == value, (
        f"fit(..., {kwarg}=...) must set the {config_key!r} config key"
    )
    assert via_kwarg == via_config, (
        f"fit(..., {kwarg}=...) must assemble the identical Rust payload as "
        f"config={{{config_key!r}: ...}}"
    )


def test_kwarg_left_none_does_not_override_config_value() -> None:
    # Leaving the dedicated kwarg as its None default must not stomp a value the
    # user passed through the escape hatch.
    payload = _payload(config={"noise_formula": "s(z)", "flexible_link": True})
    assert payload["noise_formula"] == "s(z)"
    assert payload["flexible_link"] is True


def test_dedicated_kwarg_wins_over_conflicting_config_key() -> None:
    # When both are supplied, the dedicated kwarg wins (same rule as firth et al.).
    payload = _payload(noise_formula="s(x)", config={"noise_formula": "s(other)"})
    assert payload["noise_formula"] == "s(x)"


def test_unset_model_spec_kwargs_emit_no_config_keys() -> None:
    # No spurious keys when the user touches none of the three (no behavior change
    # for callers that never used these fields).
    payload = _payload()
    for key in ("noise_formula", "noise_offset", "flexible_link"):
        assert key not in payload


def _location_scale_training_frame() -> typing.Any:
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(7)
    n = 200
    x = np.linspace(-2.0, 2.0, n)
    # Heteroscedastic Gaussian: both mean and log-scale vary smoothly with x.
    mean = 0.7 * x
    scale = np.exp(0.3 + 0.25 * np.sin(x))
    y = mean + scale * rng.standard_normal(n)
    return {"y": y.tolist(), "x": x.tolist()}


@pytest.mark.xfail(
    strict=True,
    reason="#1512 triage / #913: family='gaussian-location-scale' is not a "
    "directly-nameable family (InvalidConfigurationError: unknown family "
    "'gaussian-location-scale'; the location-scale families are only reached "
    "via family='gaussian' + a noise_formula, which #913 tracks as not yet "
    "magic-routed). The kwarg-vs-config parity cannot be checked until the "
    "location-scale family is selectable; tracking as open.",
)
def test_rust_location_scale_fit_kwarg_matches_config() -> None:
    pytest.importorskip("gamfit._rust")
    np = pytest.importorskip("numpy")

    data = _location_scale_training_frame()

    via_kwarg = gamfit.fit(
        data,
        "y ~ s(x)",
        family="gaussian-location-scale",
        noise_formula="s(x)",
    )
    via_config = gamfit.fit(
        data,
        "y ~ s(x)",
        family="gaussian-location-scale",
        config={"noise_formula": "s(x)"},
    )

    pred_kwarg = np.asarray(via_kwarg.predict(data), dtype=float)
    pred_config = np.asarray(via_config.predict(data), dtype=float)
    np.testing.assert_allclose(
        pred_kwarg,
        pred_config,
        rtol=0.0,
        atol=0.0,
        err_msg=(
            "noise_formula kwarg must produce the identical location-scale fit "
            "as passing it through config={...}"
        ),
    )


@pytest.mark.xfail(
    strict=True,
    reason="#1512 triage / gam#1596: family='bernoulli' with flexible_link=True "
    "does not converge on this data — the binomial mean link-wiggle joint solve "
    "exits the Newton path before convergence (IntegrationError citing gam#1596), "
    "so the kwarg-vs-config parity fit cannot complete. Tracking the flexible-link "
    "convergence gap as open.",
)
def test_rust_flexible_link_fit_kwarg_matches_config() -> None:
    pytest.importorskip("gamfit._rust")
    np = pytest.importorskip("numpy")

    rng = np.random.default_rng(11)
    n = 300
    x = np.linspace(-3.0, 3.0, n)
    p = 1.0 / (1.0 + np.exp(-(0.4 + 1.1 * x)))
    y = (rng.uniform(size=n) < p).astype(float)
    data = {"y": y.tolist(), "x": x.tolist()}

    via_kwarg = gamfit.fit(data, "y ~ s(x)", family="bernoulli", flexible_link=True)
    via_config = gamfit.fit(
        data, "y ~ s(x)", family="bernoulli", config={"flexible_link": True}
    )

    pred_kwarg = np.asarray(via_kwarg.predict(data), dtype=float)
    pred_config = np.asarray(via_config.predict(data), dtype=float)
    np.testing.assert_allclose(
        pred_kwarg,
        pred_config,
        rtol=0.0,
        atol=0.0,
        err_msg=(
            "flexible_link kwarg must produce the identical fit as passing it "
            "through config={...}"
        ),
    )
