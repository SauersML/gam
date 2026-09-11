"""Native CTN regression; run under an external wall limit."""
import numpy as np
import pandas as pd
import gamfit


def test_ctn_small_score_units_keep_affine_directions(tmp_path):
    rng = np.random.default_rng(8016)
    n = 240
    data = pd.DataFrame({f"PC{i}": rng.normal(size=n) for i in range(1, 7)})
    data["PGS"] = 1e-5 * (.2 * data.PC1 + rng.normal(size=n))
    data["age0"] = rng.uniform(40, 70, n)
    data["sex"] = rng.integers(0, 2, n)
    model = gamfit.fit(
        data,
        "PGS ~ s(age0, k=5) + sex + "
        "duchon(PC1, PC2, PC3, PC4, PC5, PC6, centers=8, scale_dims=true)",
        transformation_normal=True,
        config={"transformation_normal_config": {"response_num_internal_knots": 2}},
        persistent_warm_start_root=tmp_path / "warm",
    )
    # Vary only the observed score to check the frozen conditional transform.
    held = data.iloc[[0] * 9].copy()
    held["PGS"] = np.linspace(data.PGS.quantile(.05), data.PGS.quantile(.95), 9)
    score = np.asarray(model.transformation_score(held))
    assert np.isfinite(score).all()
    assert (np.diff(score) > 0).all()
    model.save(tmp_path / "transform.gamfit")
    restored = gamfit.load(tmp_path / "transform.gamfit")
    np.testing.assert_allclose(restored.transformation_score(held), score,
                               rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(restored.transformation_score(held.iloc[[4]]), score[[4]],
                               rtol=1e-8, atol=1e-10)
