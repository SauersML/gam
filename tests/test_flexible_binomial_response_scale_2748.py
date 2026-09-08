"""#2748: fitted and saved flexible binomial models predict probabilities."""

import numpy as np
import pytest

import gamfit


@pytest.mark.parametrize("link", ["logit", "probit"])
def test_flexible_binomial_response_scale_survives_save_load(link, tmp_path):
    # Match the six-column Rust payload fixture, including its nonzero warp.
    t = np.linspace(-2.75, 2.75, 256)
    data = {
        "x0": t,
        "x1": np.sin(1.3 * t),
        "x2": np.cos(0.7 * t),
        "x3": t**2 - 2.5,
        "x4": np.sin(2.1 * t + 0.2),
    }
    eta = (
        0.15 + 0.65 * data["x0"] - 0.45 * data["x1"]
        + 0.30 * data["x2"] - 0.08 * data["x3"] + 0.22 * data["x4"]
    )
    probability = 1 / (1 + np.exp(-(eta + 0.35 * np.tanh(eta))))
    uniform = (np.arange(1, t.size + 1) * 0.6180339887498949) % 1
    data["y"] = (uniform < probability).astype(float)
    formula = f"y ~ x0 + x1 + x2 + x3 + x4 + link(type=flexible({link}))"
    model = gamfit.fit(data, formula, family="binomial")
    assert "binomial" in model.family_name.lower()

    # Predict between training locations; a metadata-only test cannot establish
    # that consumers actually apply the fitted response map.
    held_out = {
        name: (values[:-1] + values[1:]) / 2
        for name, values in data.items() if name != "y"
    }
    before = model.predict(held_out, return_type="dict")
    path = tmp_path / f"flexible-{link}.gam"
    gamfit.save(model, path)
    loaded = gamfit.load(path)
    assert loaded.family_name == model.family_name
    after = loaded.predict(held_out, return_type="dict")
    linear = np.asarray(before["linear_predictor_plugin"])
    for column in ("mean_plugin", "posterior_mean"):
        means = np.asarray(before[column])
        assert np.isfinite(means).all()
        assert np.all((means >= 0) & (means <= 1))
        assert np.max(np.abs(means - linear)) > 0.1
        np.testing.assert_array_equal(after[column], means)
    np.testing.assert_array_equal(after["linear_predictor_plugin"], linear)
    np.testing.assert_array_equal(loaded.predict(held_out), after["posterior_mean"])
