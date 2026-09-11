"""Bound this native acceptance test externally; no xfail or convergence bypass."""
import json
import numpy as np
import pandas as pd
import gamfit
import pytest


@pytest.mark.parametrize("mode", ["keyword", "config"])
def test_standalone_ctn_schema_uses_fit_request(tmp_path, mode):
    rng = np.random.default_rng(714)
    x = rng.uniform(-1, 1, 160)
    data = pd.DataFrame({"pgs": 2 + .4 * x + rng.normal(size=160), "x": x,
                         "irrelevant_date": pd.Timestamp("2020-01-01")})
    config = {"transformation_normal_config": {"response_num_internal_knots": 2}}
    kwargs = {"transformation_normal": True} if mode == "keyword" else {}
    if mode == "config":
        config["transformation_normal"] = True
    model = gamfit.fit(data, "pgs ~ x", config=config, **kwargs,
                       persistent_warm_start_root=tmp_path / "warm")
    assert np.isfinite(model.transformation_score(data)).all()
    posterior = json.loads(model.dumps())["payload"]["unified"]["geometry"]["constrained_posterior"]
    assert posterior["moment_status"] == "Available"
    declined = json.loads(model.dumps())
    for key in ("unified", "fit_result"):
        fit = declined["payload"][key]
        geometry = fit["geometry"]["constrained_posterior"]
        geometry["moment_status"] = {"Declined": {
            "ambient_precision_failure": "regression fixture",
            "properness": {"CertificationFailed": {"reason": "regression fixture"}}}}
        geometry["unconstrained_center"] = None
        geometry["correction"] = None
        fit["covariance_conditional"] = None
        fit["covariance_corrected"] = None
    with pytest.raises(gamfit.GamError, match="posterior-mean"):
        gamfit.loads(json.dumps(declined).encode()).transformation_score(data)


def test_native_ctn_chain_save_load_and_batches(tmp_path):
    rng = np.random.default_rng(714)
    n = 160
    x = rng.uniform(-1, 1, n)
    z = rng.normal(size=n)
    data = pd.DataFrame({"pgs": 2 + .4 * x + z, "x": x,
                         "y": (rng.normal(size=n) < -.2 + .3 * x + .5 * z).astype(int),
                         "group": np.arange(n)})
    data["irrelevant_date"] = pd.Timestamp("2020-01-01")
    gamfit.validate_formula(
        data, "y ~ x", family="bernoulli-marginal-slope", slope_formula="1",
        transformation_normal_stage1=gamfit.CtnStage1(
            "pgs", "x", group_column="group", folds=2,
            response_num_internal_knots=2))
    model = gamfit.fit(
        data, "y ~ x", family="bernoulli-marginal-slope", slope_formula="1",
        transformation_normal_stage1=gamfit.CtnStage1(
            "pgs", "x", group_column="group", folds=2,
            response_num_internal_knots=2),
        persistent_warm_start_root=tmp_path / "warm")
    test = data[["pgs", "x"]].iloc[:8].copy()
    before = np.asarray(model.predict(test))
    assert np.isfinite(before).all() and ((before >= 0) & (before <= 1)).all()
    model.save(tmp_path / "chain.gamfit")
    restored = gamfit.load(tmp_path / "chain.gamfit")
    np.testing.assert_allclose(restored.predict(test), before, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(restored.predict(test.iloc[::-1]), before[::-1], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(restored.predict(test.iloc[[3]]), before[[3]], rtol=1e-8, atol=1e-10)
    payload = json.loads(restored.dumps())
    transform_payload = payload["payload"]["score_transform"]
    payload["payload"]["score_transform"] = None
    payload["payload"]["score_crossfit_folds"] = None
    outcome = gamfit.loads(json.dumps(payload).encode())
    # Explicit application uses the same saved native CTN evaluator.
    manual = outcome.predict(test.assign(__gamfit_ctn_score=restored.transformation_score(test)))
    np.testing.assert_allclose(manual, before, rtol=1e-8, atol=1e-10)
    assert transform_payload["score_transform"] is None
    transform = gamfit.loads(json.dumps({"model_type": "transformation-normal", "payload": transform_payload}).encode())
    gamfit.validate_formula(
        data, "y ~ x", family="bernoulli-marginal-slope", slope_formula="1",
        transformation_normal_stage1=transform)
    attached = gamfit.fit(
        data, "y ~ x", family="bernoulli-marginal-slope", slope_formula="1",
        transformation_normal_stage1=transform,
        persistent_warm_start_root=tmp_path / "warm-attached")
    np.testing.assert_allclose(attached.transformation_score(test), transform.transformation_score(test),
                               rtol=1e-8, atol=1e-10)
    attached.save(tmp_path / "attached.gamfit")
    np.testing.assert_allclose(gamfit.load(tmp_path / "attached.gamfit").predict(test),
                               attached.predict(test), rtol=1e-8, atol=1e-10)
    # The same saved transform also crosses the native survival boundary.
    event_time = rng.exponential(np.exp(-.4 * z))
    censor_time = rng.uniform(.5, 2., n)
    survival_data = data.assign(entry=0., exit=np.minimum(event_time, censor_time),
                                event=(event_time <= censor_time).astype(int))
    survival = gamfit.fit(
        survival_data, "Surv(entry, exit, event) ~ x",
        survival_likelihood="marginal-slope", slope_formula="1",
        transformation_normal_stage1=transform,
        config={"time_num_internal_knots": 2},
        persistent_warm_start_root=tmp_path / "warm-survival")
    prospective = test.assign(entry=0., exit=2., event=0)
    times = [.1, .5, 1., 2.]
    probabilities = np.asarray(survival.predict(prospective).survival_at(times))
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert (np.diff(probabilities, axis=1) <= 1e-10).all()
    survival.save(tmp_path / "survival.gamfit")
    loaded_survival = gamfit.load(tmp_path / "survival.gamfit")
    np.testing.assert_allclose(loaded_survival.predict(prospective).survival_at(times),
                               probabilities, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(loaded_survival.predict(prospective.iloc[[3]]).survival_at(times),
                               probabilities[[3]], rtol=1e-8, atol=1e-10)
    for state in (model.dumps(), restored.dumps()):
        saved = json.loads(state)["payload"]
        assert saved.get("latent_z_rank_int_calibration") is None
        assert saved.get("latent_z_conditional_calibration") is None
