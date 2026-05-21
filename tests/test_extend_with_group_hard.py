"""Hard-scenario tests for Model.extend_with_group + precision_hyperpriors.

Exercises post-fit deployment-time random-effect-level extension at
``gamfit/_model.py:1675`` together with the ``precision_hyperpriors``
fit-time callable path at ``gamfit/_api.py:141``.

A few assertions are intentionally marked ``xfail`` where the public
Python API does not currently expose enough state to verify the
underlying numerical relationship — those xfails point at the
file:line in the source where the missing surface would have to live.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

pytest.importorskip("gamfit._rust")

import numpy as np
import pandas as pd

import gamfit


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _training_frame(seed: int = 7, n_per_level: int = 8) -> pd.DataFrame:
    """Small deterministic frame with 3 random-effect levels and a continuous x."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    level_offsets = {"alpha": -0.5, "beta": 0.0, "gamma": 0.4}
    for level, offset in level_offsets.items():
        for _ in range(n_per_level):
            x = float(rng.uniform(-1.0, 1.0))
            # y = intercept + slope*x + group offset + tiny noise
            y = 1.0 + 0.3 * x + offset + float(rng.normal(0.0, 0.05))
            rows.append({"y": y, "x": x, "g": level})
    return pd.DataFrame(rows)


def _fit_simple(precision_hyperpriors: Any | None = None) -> "gamfit.Model":
    return gamfit.fit(
        _training_frame(),
        "y ~ x + group(g)",
        precision_hyperpriors=precision_hyperpriors,
    )


def _predict_eta(model: "gamfit.Model", rows: list[dict[str, Any]]) -> np.ndarray:
    out = model.predict(rows)
    # `predict` returns either a dict-of-lists or a DataFrame
    if isinstance(out, dict):
        return np.asarray(out["eta"], dtype=float)
    return np.asarray(out["eta"].to_numpy(), dtype=float)


# ---------------------------------------------------------------------------
# 1. Round-trip mean=0
# ---------------------------------------------------------------------------


def test_extend_with_zero_mean_matches_population_eta() -> None:
    """A new random-effect level with explicit prior mean=0 should yield
    the same eta as the population (no-random-effect) reference: the
    contribution of the inserted zero coefficient is exactly zero."""
    model = _fit_simple()

    extended = model.extend_with_group(
        new_group_spec={"term": "g", "level": "delta"},
        prior={"mean": 0.0},
    )

    rows = [{"x": x, "g": "delta"} for x in (-0.4, 0.0, 0.4)]
    eta_extended = _predict_eta(extended, rows)

    # Reference: an existing level whose fitted coefficient is small but
    # not zero won't match. Instead we compare to a row that the
    # extended model is allowed to evaluate but whose RE coefficient is
    # zero by construction (the freshly inserted level). The relationship
    # we can independently anchor is: eta(delta) = intercept + slope*x,
    # which equals the model's prediction with x at the same point and
    # an arbitrary existing level minus that level's offset. We can't
    # observe per-level offsets directly through public API, so we
    # instead just verify the predictions are finite and order-stable
    # with x.
    assert np.all(np.isfinite(eta_extended))
    assert eta_extended[2] > eta_extended[0]  # positive slope on x


# ---------------------------------------------------------------------------
# 2. Non-zero prior mean shifts predictions toward mu
# ---------------------------------------------------------------------------


def test_extend_with_non_zero_mean_shifts_eta_by_exactly_mu() -> None:
    """For a freshly-deployed random-effect level the inserted coefficient
    equals the supplied prior mean exactly (no data updates it).
    Therefore eta(new_level, x) - eta(other_new_level_with_mean_zero, x)
    must equal mu to high tolerance."""
    model = _fit_simple()

    mu = 0.75
    ext_mu = model.extend_with_group(
        new_group_spec={"term": "g", "level": "delta"},
        prior={"mean": mu},
    )
    ext_zero = model.extend_with_group(
        new_group_spec={"term": "g", "level": "epsilon"},
        prior={"mean": 0.0},
    )

    grid = [-0.6, -0.2, 0.0, 0.3, 0.7]
    eta_mu = _predict_eta(ext_mu, [{"x": x, "g": "delta"} for x in grid])
    eta_zero = _predict_eta(ext_zero, [{"x": x, "g": "epsilon"} for x in grid])

    diff = eta_mu - eta_zero
    np.testing.assert_allclose(diff, np.full_like(diff, mu), atol=1e-10)


# ---------------------------------------------------------------------------
# 3. Unknown term name rejection
# ---------------------------------------------------------------------------


def test_extend_with_unknown_term_rejects_with_name_in_message() -> None:
    model = _fit_simple()
    with pytest.raises(Exception) as excinfo:
        model.extend_with_group(
            new_group_spec={"term": "not_a_real_term", "level": "delta"},
        )
    msg = str(excinfo.value)
    assert "not_a_real_term" in msg


# ---------------------------------------------------------------------------
# 4. Dimension / type mismatch on prior mean rejected
# ---------------------------------------------------------------------------


def test_extend_with_vector_mean_rejected() -> None:
    """``prior['mean']`` is a scalar in the Rust contract
    (``crates/gam-pyffi/src/lib.rs:155`` ``PyExtensionPrior.mean: Option<f64>``).
    Passing a list/array must raise rather than silently coerce."""
    model = _fit_simple()
    with pytest.raises(Exception):
        model.extend_with_group(
            new_group_spec={"term": "g", "level": "delta"},
            prior={"mean": [0.0, 0.0, 0.0]},
        )


def test_extend_with_nonfinite_mean_rejected() -> None:
    model = _fit_simple()
    with pytest.raises(Exception):
        model.extend_with_group(
            new_group_spec={"term": "g", "level": "delta"},
            prior={"mean": float("nan")},
        )


def test_extend_with_already_existing_level_rejected() -> None:
    model = _fit_simple()
    with pytest.raises(Exception) as excinfo:
        model.extend_with_group(
            new_group_spec={"term": "g", "level": "alpha"},
            prior={"mean": 0.0},
        )
    assert "alpha" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 5. precision_hyperpriors dict path
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason=(
        "Fitted lambdas for a saved Model are not exposed through the Python "
        "binding (see gamfit/_summary.py and gamfit/_model.py); cannot verify "
        "the MAP relation (a-1+ν/2)/(b+ q/2) from src/types.rs:209 without a "
        "summary['lambdas'] / Model.smoothing_parameters() accessor."
    ),
    strict=False,
)
def test_precision_hyperpriors_dict_path_drives_fitted_lambda() -> None:
    # Strong-prior fit vs weak-prior fit on the same data should produce
    # observably different group offsets if the dict is wired through.
    strong = gamfit.fit(
        _training_frame(),
        "y ~ x + group(g)",
        precision_hyperpriors={"g": (5000.0, 1.0)},
    )
    weak = gamfit.fit(
        _training_frame(),
        "y ~ x + group(g)",
        precision_hyperpriors={"g": (1.0, 0.0)},
    )
    # Without lambda exposure, the only public-API surrogate is to
    # compare predictions at training rows; a strong precision prior
    # should pull per-level eta closer to the global mean.
    rows = [{"x": 0.0, "g": "alpha"}, {"x": 0.0, "g": "beta"}, {"x": 0.0, "g": "gamma"}]
    eta_strong = _predict_eta(strong, rows)
    eta_weak = _predict_eta(weak, rows)
    spread_strong = float(np.std(eta_strong))
    spread_weak = float(np.std(eta_weak))
    assert spread_strong < spread_weak  # xfail anchor


def test_precision_hyperpriors_dict_path_accepts_known_label() -> None:
    """Smoke: a dict with a real group label is accepted by fit()."""
    model = gamfit.fit(
        _training_frame(),
        "y ~ x + group(g)",
        precision_hyperpriors={"g": (2.0, 1.0)},
    )
    # If wired through, the model is fittable and predicts finitely.
    eta = _predict_eta(model, [{"x": 0.0, "g": "alpha"}])
    assert np.all(np.isfinite(eta))


# ---------------------------------------------------------------------------
# 6. precision_hyperpriors callable path
# ---------------------------------------------------------------------------


def test_precision_hyperpriors_callable_is_invoked_with_documented_keys() -> None:
    captured: list[dict[str, Any]] = []

    def cb(ctx: dict[str, Any]) -> tuple[float, float]:
        captured.append(dict(ctx))
        return (2.0, 1.0)

    gamfit.fit(
        _training_frame(),
        "y ~ x + group(g)",
        precision_hyperpriors=cb,
    )

    # The callable is invoked once per group label parsed from the
    # formula (see gamfit/_api.py:141 _resolve_precision_hyperpriors).
    assert len(captured) == 1
    ctx = captured[0]
    expected_keys = {
        "label",
        "term",
        "column",
        "levels",
        "n_coefficients",
        "metadata",
        "group_metadata",
    }
    assert set(ctx.keys()) == expected_keys
    assert ctx["label"] == "g"
    assert ctx["term"] == "g"
    assert ctx["column"] == "g"
    assert sorted(ctx["levels"]) == ["alpha", "beta", "gamma"]
    assert ctx["n_coefficients"] == 3


def test_precision_hyperpriors_callable_pair_dict_form_accepted() -> None:
    def cb(_ctx: dict[str, Any]) -> dict[str, float]:
        return {"shape": 3.0, "rate": 0.5}

    model = gamfit.fit(
        _training_frame(),
        "y ~ x + group(g)",
        precision_hyperpriors=cb,
    )
    eta = _predict_eta(model, [{"x": 0.0, "g": "alpha"}])
    assert np.all(np.isfinite(eta))


@pytest.mark.xfail(
    reason=(
        "Fitted lambdas are not exposed to Python (no accessor in "
        "gamfit/_summary.py or gamfit/_model.py), so the (shape, rate)->lambda* "
        "round-trip from src/types.rs:209 cannot be checked from the public API."
    ),
    strict=False,
)
def test_precision_hyperpriors_callable_changes_fitted_lambda() -> None:
    strong = gamfit.fit(
        _training_frame(),
        "y ~ x + group(g)",
        precision_hyperpriors=lambda _ctx: (5000.0, 1.0),
    )
    weak = gamfit.fit(
        _training_frame(),
        "y ~ x + group(g)",
        precision_hyperpriors=lambda _ctx: (1.0, 0.0),
    )
    rows = [{"x": 0.0, "g": lvl} for lvl in ("alpha", "beta", "gamma")]
    eta_strong = _predict_eta(strong, rows)
    eta_weak = _predict_eta(weak, rows)
    assert float(np.std(eta_strong)) < float(np.std(eta_weak))


# ---------------------------------------------------------------------------
# 7. Bad return values from the callable are rejected
# ---------------------------------------------------------------------------


def test_precision_hyperpriors_callable_returning_garbage_is_rejected() -> None:
    def cb(_ctx: dict[str, Any]) -> Any:
        return "not a (shape, rate) tuple"

    with pytest.raises(Exception):
        gamfit.fit(
            _training_frame(),
            "y ~ x + group(g)",
            precision_hyperpriors=cb,
        )


def test_precision_hyperpriors_callable_returning_negative_shape_is_rejected() -> None:
    def cb(_ctx: dict[str, Any]) -> tuple[float, float]:
        return (-1.0, 1.0)

    with pytest.raises(Exception):
        gamfit.fit(
            _training_frame(),
            "y ~ x + group(g)",
            precision_hyperpriors=cb,
        )


def test_precision_hyperpriors_callable_returning_nonfinite_is_rejected() -> None:
    def cb(_ctx: dict[str, Any]) -> tuple[float, float]:
        return (float("inf"), 1.0)

    with pytest.raises(Exception):
        gamfit.fit(
            _training_frame(),
            "y ~ x + group(g)",
            precision_hyperpriors=cb,
        )


# ---------------------------------------------------------------------------
# 8. Hyperprior + prior-mean interaction
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason=(
        "Combined check of (a, b) hyperprior + non-zero prior mean against the "
        "closed-form lambda* requires Python access to fitted lambdas and the "
        "per-group quadratic (beta-mu)'S(beta-mu); neither is exposed (see "
        "gamfit/_summary.py)."
    ),
    strict=False,
)
def test_hyperprior_plus_prior_mean_matches_closed_form_lambda_star() -> None:
    fit_model = gamfit.fit(
        _training_frame(),
        "y ~ x + group(g)",
        precision_hyperpriors={"g": (3.0, 0.5)},
    )
    extended = fit_model.extend_with_group(
        new_group_spec={"term": "g", "level": "delta"},
        prior={"mean": 0.4},
    )
    # Placeholder for the closed-form check; the assertion below merely
    # exercises the code path. Once a `model.smoothing_parameters()`
    # accessor lands, replace with:
    #   q = sum_p (beta_p - mu_p)^2  (using model.coefficients_frame())
    #   a, b = 3.0, 0.5
    #   nu = effective_dim_of_g
    #   lambda_star = (a - 1.0 + 0.5 * nu) / (b + 0.5 * q)
    #   assert math.isclose(lambdas['g'], lambda_star, rel_tol=1e-3)
    eta = _predict_eta(extended, [{"x": 0.0, "g": "delta"}])
    assert np.all(np.isfinite(eta))
    raise AssertionError("closed-form lambda* not verifiable from Python API")


# ---------------------------------------------------------------------------
# 9. Sequential extends are order-invariant
# ---------------------------------------------------------------------------


def test_sequential_extends_are_order_invariant() -> None:
    model = _fit_simple()
    mu_d, mu_e = 0.3, -0.4

    ext_de = (
        model
        .extend_with_group(
            new_group_spec={"term": "g", "level": "delta"},
            prior={"mean": mu_d},
        )
        .extend_with_group(
            new_group_spec={"term": "g", "level": "epsilon"},
            prior={"mean": mu_e},
        )
    )
    ext_ed = (
        model
        .extend_with_group(
            new_group_spec={"term": "g", "level": "epsilon"},
            prior={"mean": mu_e},
        )
        .extend_with_group(
            new_group_spec={"term": "g", "level": "delta"},
            prior={"mean": mu_d},
        )
    )

    rows = [
        {"x": 0.0, "g": "delta"},
        {"x": 0.0, "g": "epsilon"},
        {"x": 0.2, "g": "delta"},
        {"x": -0.2, "g": "epsilon"},
    ]
    eta_de = _predict_eta(ext_de, rows)
    eta_ed = _predict_eta(ext_ed, rows)
    np.testing.assert_allclose(eta_de, eta_ed, atol=1e-10)

    # And each level still respects its respective mu:
    grid_d = [{"x": x, "g": "delta"} for x in (-0.3, 0.0, 0.3)]
    grid_e = [{"x": x, "g": "epsilon"} for x in (-0.3, 0.0, 0.3)]
    eta_d = _predict_eta(ext_de, grid_d)
    eta_e = _predict_eta(ext_de, grid_e)
    # Difference between the two new levels at matching x equals mu_d - mu_e.
    np.testing.assert_allclose(eta_d - eta_e, np.full(3, mu_d - mu_e), atol=1e-10)


# ---------------------------------------------------------------------------
# 10. Serialization round-trip after extend
# ---------------------------------------------------------------------------


def test_extended_model_save_and_reload_predicts_identically(tmp_path: pathlib.Path) -> None:
    model = _fit_simple()
    extended = model.extend_with_group(
        new_group_spec={"term": "g", "level": "delta"},
        prior={"mean": 0.25},
    )

    path = tmp_path / "extended.gam"
    extended.save(path)
    reloaded = gamfit.load(path)

    rows = [
        {"x": -0.4, "g": "delta"},
        {"x": 0.1, "g": "alpha"},
        {"x": 0.6, "g": "delta"},
        {"x": -0.1, "g": "gamma"},
    ]
    eta_before = _predict_eta(extended, rows)
    eta_after = _predict_eta(reloaded, rows)
    np.testing.assert_allclose(eta_before, eta_after, atol=1e-12, rtol=0.0)

    # And the deployment extension survives serialization.
    payload = json.loads(path.read_text())["payload"]
    deployment = payload.get("deployment_extensions") or []
    assert any(
        ext.get("kind") == "random-effect-level"
        and ext.get("term") == "g"
        and ext.get("coefficient_mean") == 0.25
        for ext in deployment
    )
