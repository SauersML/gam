from __future__ import annotations

import importlib

pytest = importlib.import_module("pytest")
pytest.importorskip("gamfit._rust")


def test_model_save_load_predict_round_trip_is_bitwise_stable(tmp_path) -> None:
    import numpy as np
    import pandas as pd
    import gamfit

    rng = np.random.default_rng(42)
    x = np.linspace(-1.0, 1.0, 64)
    y = 0.5 + 1.7 * x + rng.normal(0.0, 0.01, size=x.shape[0])
    df = pd.DataFrame({"y": y, "x": x})

    model = gamfit.fit(df, "y ~ x")
    original = model.predict(df)

    path = tmp_path / "roundtrip_model.gam"
    model.save(path)
    loaded = gamfit.load(path)
    reloaded = loaded.predict(df)

    orig_arr = np.asarray(original["mean"], dtype=np.float64)
    reloaded_arr = np.asarray(reloaded["mean"], dtype=np.float64)

    assert np.array_equal(
        orig_arr.view(np.uint64), reloaded_arr.view(np.uint64)
    ), "save → load should preserve predict() output bit-for-bit for a freshly fitted model"
