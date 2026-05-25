import importlib
from typing import Any, NoReturn, Protocol, cast


class _PytestModule(Protocol):
    def importorskip(
        self,
        modname: str,
        minversion: str | None = None,
        reason: str | None = None,
        *,
        exc_type: type[ImportError] | tuple[type[ImportError], ...] | None = None,
    ) -> Any: ...

    def fail(self, reason: str = "", pytrace: bool = True) -> NoReturn: ...


pytest = cast(_PytestModule, importlib.import_module("pytest"))
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit


def _weibull_frame(n: int = 240, seed: int = 7):
    rng = np.random.default_rng(seed)
    age = rng.uniform(40.0, 75.0, n)
    bmi = rng.uniform(18.0, 40.0, n)
    eta = -2.1 + 0.05 * (age - 50.0) + 0.03 * (bmi - 26.0)
    shape = 1.4
    u = rng.uniform(1e-9, 1.0, n)
    latent = np.exp(-eta / shape) * (-np.log(u)) ** (1.0 / shape)
    latent *= 9.0
    censor = rng.exponential(14.0, n)
    censor = np.minimum(censor, 18.0)
    exit_t = np.minimum(latent, censor)
    event = (latent <= censor).astype(int)
    return pd.DataFrame({"entry": np.zeros(n), "exit": exit_t, "event": event, "age": age, "bmi": bmi})


def test_bug_custom_family_coefficient_group_labels_are_stably_routed() -> None:
    train = _weibull_frame(180)
    model = gamfit.fit(
        train,
        "Surv(entry, exit, event) ~ age + bmi",
        survival_likelihood="marginal-slope",
        z_column="age",
        logslope_formula="bmi",
        custom_family={
            "coefficient_groups": [
                {"label": "mean.age", "indices": [1]},
                {"label": "slope.bmi", "indices": [2]},
            ]
        },
    )
    state = model.coefficient_state_json()
    assert '"mean.age"' in state
    assert '"slope.bmi"' in state


def test_bug_transformation_normal_time_basis_dimension_matches_response_basis() -> None:
    train = _weibull_frame(260)
    model = gamfit.fit(
        train,
        "Surv(entry, exit, event) ~ age",
        survival_likelihood="transformation",
        transformation_normal=True,
    )
    saved = model.save_bytes()
    loaded = gamfit.load_bytes(saved)
    pred = loaded.predict(train.iloc[:8].copy())
    assert pred.linear_predictor.shape[0] == 8


def test_bug_latent_survival_frailty_hazard_loading_requires_hazard_multiplier() -> None:
    train = _weibull_frame(120)
    with pytest.raises(Exception):
        gamfit.validate_formula(
            train,
            "Surv(entry, exit, event) ~ age",
            survival_likelihood="latent",
            baseline_target="weibull",
            baseline_shape=1.5,
            baseline_scale=10.0,
            frailty_kind="gaussian-shift",
            hazard_loading="full",
        )


def test_bug_latent_glm_family_synonyms_route_to_distinct_likelihood_specs() -> None:
    train = pd.DataFrame(
        {
            "x": np.linspace(-1.0, 1.0, 60),
            "y": (np.linspace(-1.0, 1.0, 60) > 0).astype(int),
        }
    )
    m1 = gamfit.fit(train, "y ~ x", family="binomial_logit")
    m2 = gamfit.fit(train, "y ~ x", family="binomial_probit")
    p1 = np.asarray(m1.predict(train).mu, dtype=float)
    p2 = np.asarray(m2.predict(train).mu, dtype=float)
    assert np.max(np.abs(p1 - p2)) > 1e-4
