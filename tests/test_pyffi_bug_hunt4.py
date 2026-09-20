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
    # A latent standard-normal score for the marginal-slope lane. Marginal
    # slope reserves its `z_column` as the auxiliary score and refuses to see
    # it in the main formula, so the score has to be its own column rather
    # than a covariate the baseline also wants (gam#2432).
    prs_z = rng.normal(0.0, 1.0, n)
    eta = -2.1 + 0.05 * (age - 50.0) + 0.03 * (bmi - 26.0) + 0.30 * prs_z
    shape = 1.4
    u = rng.uniform(1e-9, 1.0, n)
    latent = np.exp(-eta / shape) * (-np.log(u)) ** (1.0 / shape)
    latent *= 9.0
    censor = rng.exponential(14.0, n)
    censor = np.minimum(censor, 18.0)
    exit_t = np.minimum(latent, censor)
    event = (latent <= censor).astype(int)
    return pd.DataFrame(
        {
            "entry": np.zeros(n),
            "exit": exit_t,
            "event": event,
            "age": age,
            "bmi": bmi,
            "prs_z": prs_z,
        }
    )


def test_bug_custom_family_coefficient_group_labels_are_stably_routed() -> None:
    train = _weibull_frame(180)
    # `z_column` used to name `age`, which the baseline formula also uses. That
    # is the configuration marginal slope exists to reject — the score would be
    # in both the marginal design and the score block — and the engine refused
    # it at parse time, so this test never reached the block routing it exists
    # to check (gam#2432). The score is now its own column.
    model = gamfit.fit(
        train,
        "Surv(entry, exit, event) ~ age + bmi",
        survival_likelihood="marginal-slope",
        z_column="prs_z",
        slope_formula="bmi",
    )
    blocks = {b.name: (b.kind, b.start, b.end) for b in model.term_blocks}
    assert blocks["intercept"] == ("intercept", 0, 1)
    assert blocks["age"] == ("linear", 1, 2)
    assert blocks["bmi"] == ("linear", 2, 3)


def test_bug_transformation_survival_time_basis_dimension_matches_response_basis() -> None:
    train = _weibull_frame(260)
    model = gamfit.fit(
        train,
        "Surv(entry, exit, event) ~ age",
        survival_likelihood="transformation",
    )
    loaded = gamfit.loads(model.dumps())
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
    rng = np.random.default_rng(71)
    x = np.linspace(-2.0, 2.0, 120)
    probability = 1.0 / (1.0 + np.exp(-(0.25 + 0.8 * x)))
    train = pd.DataFrame(
        {
            "x": x,
            "y": rng.binomial(1, probability),
        }
    )
    m1 = gamfit.fit(train, "y ~ x", family="binomial-logit")
    m2 = gamfit.fit(train, "y ~ x", family="binomial-probit")
    # `Model.predict` returns the response-scale mean directly for a standard
    # GAM; there is no `.mu` attribute on the result and there never has been
    # (`grep -rn "def mu" gamfit/` is empty). Reading `.mu` raised
    # `AttributeError: 'numpy.ndarray' object has no attribute 'mu'` before the
    # two fits could ever be compared, so this test was failing on a dead
    # accessor rather than on the routing it exists to check. The named column
    # is `mean` under `return_type="dict"`; the bare call already IS that array.
    p1 = np.asarray(m1.predict(train), dtype=float)
    p2 = np.asarray(m2.predict(train), dtype=float)
    assert np.max(np.abs(p1 - p2)) > 1e-4
