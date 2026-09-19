"""A ``fit()`` model-spec field has one spelling: its dedicated keyword.

``config={...}`` carries only request fields without a dedicated keyword (for
example ``group_metadata`` or ``precompute_conformal``). A ``config`` key that
duplicates a keyword used to be silently overridden by the keyword, or silently
taken when the keyword was left ``None``; it is now refused, naming the keyword
to use. This test pins three properties:

1. Each dedicated keyword sets its request key, and unset keywords emit no key.
2. Every ``config`` spelling of a keyword is refused, including the Rust request
   names of ``latents`` / ``penalties`` / ``smooths`` / ``transformation_normal_stage1``
   and the response-geometry keywords.
3. A ``config`` key with no dedicated keyword passes through unchanged.
"""

from __future__ import annotations

import importlib
import typing

pytest = typing.cast(typing.Any, importlib.import_module("pytest"))

import gamfit
from gamfit._api import _build_fit_payload


_BASE: dict[str, typing.Any] = {
    "family": "auto",
    "negative_binomial_theta": None,
    "expectile_tau": None,
    "offset": None,
    "weights": None,
    "transformation_normal": None,
    "transformation_normal_stage1": None,
    "survival_likelihood": None,
    "survival_time_anchor": None,
    "baseline_target": None,
    "baseline_scale": None,
    "baseline_shape": None,
    "baseline_rate": None,
    "baseline_makeham": None,
    "z_column": None,
    "link": None,
    "slope_formula": None,
    "frailty_kind": None,
    "frailty_sd": None,
    "hazard_loading": None,
    "scale_dimensions": None,
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
    ("kwarg", "value"),
    [
        ("noise_formula", "s(x)"),
        ("negative_binomial_theta", 2.5),
        ("expectile_tau", 0.9),
        ("noise_offset", "logvar"),
        ("flexible_link", True),
        ("survival_time_anchor", 25.0),
    ],
)
def test_model_spec_kwarg_sets_its_request_key(kwarg: str, value: typing.Any) -> None:
    assert _payload(**{kwarg: value}).get(kwarg) == value


@pytest.mark.parametrize(
    ("config_key", "keyword"),
    [
        ("noise_formula", "noise_formula"),
        ("flexible_link", "flexible_link"),
        ("family", "family"),
        ("offset", "offset"),
        ("weights", "weights"),
        ("latent_coordinates", "latents"),
        ("analytic_penalties", "penalties"),
        ("smooth_descriptors", "smooths"),
        ("ctn_stage1", "transformation_normal_stage1"),
        ("response_geometry", "response_geometry"),
    ],
)
def test_config_spelling_of_a_keyword_is_refused(config_key: str, keyword: str) -> None:
    with pytest.raises(ValueError, match=rf"duplicates the {keyword}= keyword"):
        _payload(config={config_key: "value"})


def test_config_key_without_a_keyword_passes_through() -> None:
    payload = _payload(
        config={"precompute_conformal": False, "group_metadata": {"g": {}}}
    )
    assert payload["precompute_conformal"] is False
    assert payload["group_metadata"] == {"g": {}}


def test_fit_takes_no_on_disk_warm_start_root() -> None:
    """A cache directory does not change the fitted model, so it has no keyword.

    The request document is ``deny_unknown_fields`` and no longer carries the
    key either, so the ``config`` spelling is refused by the Rust resolver.
    """

    data = {"y": [0.1, 0.4, 0.2, 0.9, 0.5, 0.7], "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]}
    with pytest.raises(TypeError, match=r"persistent_warm_start_root"):
        gamfit.fit(data, "y ~ x", family="gaussian", persistent_warm_start_root="warm")
    with pytest.raises(Exception, match=r"persistent_warm_start_root"):
        gamfit.fit(data, "y ~ x", family="gaussian", config={"persistent_warm_start_root": "warm"})


def test_unset_model_spec_kwargs_emit_no_config_keys() -> None:
    payload = _payload()
    for key in (
        "noise_formula",
        "noise_offset",
        "flexible_link",
        "survival_time_anchor",
    ):
        assert key not in payload


def test_fit_refuses_config_spelling_before_fitting() -> None:
    data = {"y": [0.1, 0.4, 0.2, 0.9, 0.5, 0.7], "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]}
    with pytest.raises(ValueError, match=r"duplicates the noise_formula= keyword"):
        gamfit.fit(data, "y ~ x", family="gaussian", config={"noise_formula": "x"})


def test_solver_tolerances_are_not_request_options() -> None:
    """Solver tolerances are derived (gam SPEC 18-23), never requested.

    The request document refuses the keys by name, so ``config`` cannot carry
    one either.
    """

    pytest.importorskip("gamfit._rust")
    data = {"y": [0.1, 0.4, 0.2, 0.9, 0.5, 0.7], "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]}
    for key in ("outer_tol", "inner_tol"):
        with pytest.raises(Exception, match=rf"unknown field `{key}`"):
            gamfit.fit(data, "y ~ x", family="gaussian", config={key: 1e-8})


def test_warm_start_from_takes_a_fitted_model() -> None:
    data = {"y": [0.1, 0.4, 0.2, 0.9, 0.5, 0.7], "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]}
    with pytest.raises(TypeError, match=r"warm_start_from takes a fitted gamfit.Model"):
        gamfit.fit(data, "y ~ x", family="gaussian", warm_start_from=object())


def test_warm_start_from_a_model_with_no_certified_point_is_refused() -> None:
    """A standard GAM records no custom-family outer point to resume from.

    The refusal comes from the Rust side, so it proves the model crossed the
    wire and was read.
    """

    pytest.importorskip("gamfit._rust")
    data = {"y": [0.1, 0.4, 0.2, 0.9, 0.5, 0.7], "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]}
    model = gamfit.fit(data, "y ~ x", family="gaussian")
    with pytest.raises(Exception, match=r"warm_start_from"):
        gamfit.fit(data, "y ~ x", family="gaussian", warm_start_from=model)
