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
    from gamfit._ctn_model import _with_score
    manual = restored.outcome.predict(_with_score(test, restored.transform.transformation_score(test)))
    np.testing.assert_allclose(manual, before, rtol=1e-8, atol=1e-10)
    for payload in (model.outcome.dumps(), restored.outcome.dumps()):
        import json
        saved = json.loads(payload)["payload"]
        assert saved.get("latent_z_rank_int_calibration") is None
        assert saved.get("latent_z_conditional_calibration") is None
