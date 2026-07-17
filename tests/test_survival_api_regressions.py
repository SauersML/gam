import importlib
import multiprocessing as mp
import queue
import time
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

pytest.importorskip("gamfit._rust")

import numpy as np
import pandas as pd

import gamfit


def make_weibull(n: int = 600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.uniform(40.0, 75.0, n)
    bmi = rng.uniform(18.0, 40.0, n)
    hba1c = rng.uniform(4.5, 9.0, n)
    eta = -2.0 + 0.04 * (age - 50.0) + 0.05 * (bmi - 25.0) + 0.4 * (hba1c - 6.0)
    shape = 1.5
    u = rng.uniform(1e-9, 1.0, n)
    t_lat = np.exp(-eta / shape) * (-np.log(u)) ** (1.0 / shape)
    t_lat *= 10.0
    c = rng.exponential(1.0 / (-np.log(0.5) / 25.0), n)
    c = np.minimum(c, 25.0)
    t_obs = np.minimum(t_lat, c)
    event = (t_lat <= c).astype(int)
    return pd.DataFrame(
        {
            "entry": np.zeros(n),
            "exit": t_obs,
            "event": event,
            "age": age,
            "bmi": bmi,
            "hba1c": hba1c,
        }
    )


def prediction_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry": [0.0, 0.0, 0.0],
            "exit": [5.0, 10.0, 20.0],
            "event": [1, 1, 1],
            "age": [45.0, 55.0, 65.0],
            "bmi": [22.0, 28.0, 35.0],
            "hba1c": [5.2, 5.7, 6.5],
        }
    )


def make_competing_risks(n: int = 320, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.uniform(40.0, 75.0, n)
    x = (age - 55.0) / 10.0
    t1 = rng.exponential(scale=1.0 / np.exp(-3.0 + 0.25 * x), size=n)
    t2 = rng.exponential(scale=1.0 / np.exp(-3.2 - 0.20 * x), size=n)
    censor = rng.exponential(scale=22.0, size=n)
    exit_time = np.minimum.reduce([t1, t2, censor]) + 0.1
    event = np.where((t1 < t2) & (t1 < censor), 1.0, 0.0)
    event = np.where((t2 < t1) & (t2 < censor), 2.0, event)
    return pd.DataFrame(
        {
            "entry": np.zeros(n),
            "exit": exit_time,
            "event": event,
            "age": age,
        }
    )


def test_survival_transformation_is_reachable_from_fit() -> None:
    model = gamfit.fit(
        make_weibull(260),
        "Surv(entry, exit, event) ~ age",
        survival_likelihood="transformation",
    )
    pred = model.predict(prediction_rows())
    assert model.is_survival
    assert np.all(np.isfinite(np.asarray(pred.linear_predictor, dtype=float)))
    assert np.all(np.asarray(pred.survival, dtype=float) > 0.0)


def test_transformation_survival_parameter_count_is_independent_of_n() -> None:
    # Regression for issue #385: the transformation likelihood (the CLI/auto
    # default survival mode) must carry a *fixed-size* baseline+covariate basis,
    # not one free coefficient per observation row. The released 0.1.135 build
    # allocated `n + 13` coefficients — an O(n) parameterization that forced an
    # n-wide dense design, O(n^2) memory, and a hard densify-refusal past ~5k
    # rows. The fixed model has p = p_time_basis + p_covariate, which is a small
    # constant independent of n.
    #
    # We fit at two materially different sample sizes and demand:
    #   (1) the coefficient count is *identical* across n (scale invariance), and
    #   (2) it is far below n (no per-observation block).
    counts = {}
    for n in (300, 1500):
        model = gamfit.fit(
            make_weibull(n),
            "Surv(entry, exit, event) ~ age",
            survival_likelihood="transformation",
        )
        coefs = model.summary().coefficients
        counts[n] = len(coefs)

    assert counts[300] == counts[1500], (
        "transformation survival coefficient count must not depend on n "
        f"(issue #385: per-observation block regression): {counts}"
    )
    # The fixed model is the default 8-internal-knot I-spline baseline (~10
    # columns) plus the single `age` covariate. Allow generous headroom for
    # basis/intercept bookkeeping, but it must be a small constant — and in
    # particular must be far below n, not the `n + 13` of the buggy build.
    fixed_count = counts[300]
    assert fixed_count < 64, (
        "transformation survival baseline must be a fixed-size basis, not "
        f"O(n) per-observation parameters (issue #385): got {fixed_count}"
    )
    assert fixed_count < 300, (
        f"coefficient count {fixed_count} grew with n=300 (issue #385)"
    )


def test_joint_competing_risks_survival_is_reachable_from_fit(tmp_path) -> None:
    train = make_competing_risks()
    rows = prediction_rows()[["entry", "exit", "event", "age"]]
    for likelihood_mode in ("weibull", "transformation"):
        validation = gamfit.validate_formula(
            train,
            "Surv(entry, exit, event) ~ age",
            survival_likelihood=likelihood_mode,
        )
        assert validation["model_class"] == "competing risks survival"

        model = gamfit.fit(
            train,
            "Surv(entry, exit, event) ~ age",
            survival_likelihood=likelihood_mode,
            # Cause-specific penalty blocks are distinct model components. Use
            # the same prior on both causes so this prediction-path fixture does
            # not manufacture a one-sided hyperparameter rail.
            precision_hyperpriors={
                "cause_specific_survival_cause_1_penalty_0": [2.0, 1.0],
                "cause_specific_survival_cause_2_penalty_0": [2.0, 1.0],
            },
        )
        pred = model.predict(rows)
        assert isinstance(pred, gamfit.CompetingRisksPrediction)
        assert pred.covariance_source is None
        assert pred.endpoint_names == ("cause_1", "cause_2")
        assert pred.cif.shape == (2 * 3, pred.times.size)
        assert np.all(np.isfinite(pred.cif))
        assert np.all((pred.cif >= 0.0) & (pred.cif <= 1.0))
        assert np.all(
            (pred.overall_survival >= 0.0) & (pred.overall_survival <= 1.0)
        )

        replicates = np.asarray(model.sample_replicates(rows, 41, seed=2300))
        streamed = np.concatenate(
            list(model.iter_replicates(rows, 41, chunk_size=7, seed=2300)),
            axis=0,
        )
        np.testing.assert_array_equal(replicates, streamed)
        assert replicates.shape == (41, len(rows))
        assert set(np.unique(replicates)).issubset({0.0, 1.0, 2.0})

        model_path = tmp_path / f"competing-{likelihood_mode}.gam"
        model.save(model_path)
        restored = gamfit.load(model_path)
        np.testing.assert_array_equal(
            replicates,
            restored.sample_replicates(rows, 41, seed=2300),
        )

        # DEFAULT covariance mode: per the #2296 provenance contract this is
        # the smoothing-corrected covariance with no silent fallback. #2346
        # landed the fit-side corrected matrix for custom-family (competing
        # risks) fits, so the default mode must work end-to-end here.
        interval_pred = model.predict(rows, interval=0.9)
        assert isinstance(interval_pred, gamfit.CompetingRisksPrediction)
        assert interval_pred.interval_level == 0.9
        assert interval_pred.covariance_source == "smoothing-corrected"
        np.testing.assert_allclose(interval_pred.cif, pred.cif)
        np.testing.assert_allclose(
            interval_pred.overall_survival, pred.overall_survival
        )

        cause_surface_fields = (
            "hazard",
            "survival",
            "cumulative_hazard",
            "cif",
        )
        for field in cause_surface_fields:
            point = np.asarray(getattr(interval_pred, field), dtype=float)
            se = np.asarray(getattr(interval_pred, f"{field}_se"), dtype=float)
            lower = np.asarray(getattr(interval_pred, f"{field}_lower"), dtype=float)
            upper = np.asarray(getattr(interval_pred, f"{field}_upper"), dtype=float)
            assert point.shape == se.shape == lower.shape == upper.shape
            assert point.shape == (2 * 3, interval_pred.times.size)
            assert np.all(np.isfinite(se))
            assert np.all(se >= 0.0)
            assert np.all(lower <= point)
            assert np.all(point <= upper)

        overall = np.asarray(interval_pred.overall_survival, dtype=float)
        overall_se = np.asarray(interval_pred.overall_survival_se, dtype=float)
        overall_lower = np.asarray(interval_pred.overall_survival_lower, dtype=float)
        overall_upper = np.asarray(interval_pred.overall_survival_upper, dtype=float)
        assert (
            overall.shape
            == overall_se.shape
            == overall_lower.shape
            == overall_upper.shape
        )
        assert np.all(overall_se >= 0.0)
        assert np.all(overall_lower <= overall)
        assert np.all(overall <= overall_upper)

        eta = np.asarray(interval_pred.linear_predictor, dtype=float)
        eta_se = np.asarray(interval_pred.eta_se, dtype=float)
        eta_lower = np.asarray(interval_pred.eta_lower, dtype=float)
        eta_upper = np.asarray(interval_pred.eta_upper, dtype=float)
        assert (
            eta.shape
            == eta_se.shape
            == eta_lower.shape
            == eta_upper.shape
            == (2 * 3,)
        )
        assert np.all(eta_se >= 0.0)
        assert np.all(eta_lower <= eta)
        assert np.all(eta <= eta_upper)
        assert np.any(np.asarray(interval_pred.cif_se, dtype=float) > 0.0)

        # #2346: cause-specific custom-family fits now carry the smoothing-
        # corrected joint covariance (first-order rho-uncertainty inflation
        # with FirstOrderIdentifiedSubspace provenance), so the omitted mode
        # and the explicit spelling both succeed and report the corrected
        # provenance — never a silent relabeling of Vb as Vp.
        for requested_mode in (None, "smoothing"):
            kwargs = (
                {}
                if requested_mode is None
                else {"covariance_mode": requested_mode}
            )
            corrected_pred = model.predict(rows, interval=0.9, **kwargs)
            assert corrected_pred.covariance_source == "smoothing-corrected", (
                f"{likelihood_mode}/{requested_mode}: expected smoothing-corrected "
                f"provenance, got {corrected_pred.covariance_source!r}"
            )
            assert np.any(
                np.asarray(corrected_pred.cif_se, dtype=float) > 0.0
            ), f"{likelihood_mode}/{requested_mode}: corrected CIF SEs are all zero"


def test_survival_location_scale_regressor_prediction_does_not_saturate() -> None:
    model = gamfit.fit(
        make_weibull(500),
        "Surv(entry, exit, event) ~ age",
        survival_likelihood="location-scale",
    )
    pred = model.predict(prediction_rows())
    eta = np.asarray(pred.linear_predictor, dtype=float)
    survival = np.asarray(pred.survival, dtype=float)
    assert np.all(np.isfinite(eta))
    assert float(np.max(np.abs(eta))) < 50.0
    assert float(np.min(survival)) > 1.0e-12
    replicates = np.asarray(model.sample_replicates(prediction_rows(), 37, seed=2300))
    assert replicates.shape == (37, len(prediction_rows()))
    assert set(np.unique(replicates)).issubset({0.0, 1.0})


def test_latent_survival_accepts_frailty_kwargs() -> None:
    validation = gamfit.validate_formula(
        make_weibull(120),
        "Surv(entry, exit, event) ~ age",
        survival_likelihood="latent",
        baseline_target="weibull",
        baseline_shape=1.5,
        baseline_scale=10.0,
        frailty_kind="hazard-multiplier",
        hazard_loading="full",
    )
    assert validation["model_class"] == "latent survival"
    assert validation.supported_by_python is True


def test_survival_at_accepts_array_like_times() -> None:
    # Regression for #380: survival_at / hazard_at / cumulative_hazard_at
    # accept any array-like ``times`` (list / tuple / scalar), not only an
    # ndarray, and produce results identical to the ndarray path.
    model = gamfit.fit(
        make_weibull(260),
        "Surv(entry, exit, event) ~ age",
        survival_likelihood="transformation",
    )
    pred = model.predict(prediction_rows())

    times_list = [0.5, 1.0, 2.0]
    times_arr = np.array(times_list, dtype=float)

    for method in ("survival_at", "hazard_at", "cumulative_hazard_at"):
        reference = np.asarray(getattr(pred, method)(times_arr), dtype=float)
        from_list = np.asarray(getattr(pred, method)(times_list), dtype=float)
        from_tuple = np.asarray(getattr(pred, method)(tuple(times_list)), dtype=float)
        assert reference.shape == (3, len(times_list))
        np.testing.assert_allclose(from_list, reference, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(from_tuple, reference, rtol=0.0, atol=0.0)

    # A bare Python scalar must coerce to a single-column 1-time grid.
    scalar_result = np.asarray(pred.survival_at(1.0), dtype=float)
    scalar_reference = np.asarray(
        pred.survival_at(np.array([1.0], dtype=float)), dtype=float
    )
    assert scalar_result.shape == (3, 1)
    np.testing.assert_allclose(scalar_result, scalar_reference, rtol=0.0, atol=0.0)


def test_competing_risks_cif_accepts_array_like_times() -> None:
    # Regression for #380: the standalone competing_risks_cif assembler also
    # routes ``times`` through coercion and must accept array-likes.
    train = make_weibull(280)
    cause = gamfit.fit(
        train,
        "Surv(entry, exit, event) ~ age",
        survival_likelihood="transformation",
    )
    pred = cause.predict(prediction_rows())

    times_list = [1.0, 3.0, 6.0]
    from_arr = gamfit.competing_risks_cif(
        {"cause_1": pred}, times=np.array(times_list, dtype=float)
    )
    from_list = gamfit.competing_risks_cif({"cause_1": pred}, times=times_list)
    np.testing.assert_allclose(
        np.asarray(from_list.cif, dtype=float),
        np.asarray(from_arr.cif, dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    assert np.asarray(from_list.times, dtype=float).tolist() == times_list


def _fit_marginal_slope_worker(result_queue) -> None:
    try:
        gamfit.fit(
            make_weibull(220),
            "Surv(entry, exit, event) ~ bmi + hba1c",
            survival_likelihood="marginal-slope",
            z_column="age",
            logslope_formula="bmi + hba1c",
        )
    except BaseException as exc:  # pragma: no cover - child process reporting
        result_queue.put(("error", type(exc).__name__, str(exc)))
    else:
        result_queue.put(("ok",))


def test_survival_marginal_slope_fit_returns() -> None:
    result_queue = mp.Queue()
    proc = mp.Process(target=_fit_marginal_slope_worker, args=(result_queue,))
    start = time.monotonic()
    proc.start()
    proc.join(45.0)
    if proc.is_alive():
        proc.terminate()
        proc.join(5.0)
        pytest.fail("survival marginal-slope fit did not return within 45 seconds")
    assert proc.exitcode == 0
    try:
        result = result_queue.get_nowait()
    except queue.Empty as exc:
        raise AssertionError("survival marginal-slope worker returned no result") from exc
    assert result == ("ok",), result
    assert time.monotonic() - start < 45.0
