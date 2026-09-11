"""Bound this native acceptance test externally; no xfail or convergence bypass."""
import numpy as np
import pandas as pd
import gamfit


def test_native_ctn_chain_save_load_and_batches(tmp_path):
    rng = np.random.default_rng(714)
    n = 160
    x = rng.uniform(-1, 1, n)
    z = rng.normal(size=n)
    data = pd.DataFrame({"pgs": 2 + .4 * x + z, "x": x,
                         "y": (rng.normal(size=n) < -.2 + .3 * x + .5 * z).astype(int),
                         "group": np.arange(n)})
    data["irrelevant_date"] = pd.Timestamp("2020-01-01")
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
    import json
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
    attached = gamfit.fit(
        data, "y ~ x", family="bernoulli-marginal-slope", slope_formula="1",
        transformation_normal_stage1=transform,
        persistent_warm_start_root=tmp_path / "warm-attached")
    np.testing.assert_allclose(attached.transformation_score(test), transform.transformation_score(test),
                               rtol=1e-8, atol=1e-10)
    attached.save(tmp_path / "attached.gamfit")
    np.testing.assert_allclose(gamfit.load(tmp_path / "attached.gamfit").predict(test),
                               attached.predict(test), rtol=1e-8, atol=1e-10)
    for state in (model.dumps(), restored.dumps()):
        saved = json.loads(state)["payload"]
        assert saved.get("latent_z_rank_int_calibration") is None
        assert saved.get("latent_z_conditional_calibration") is None
