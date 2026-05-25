from __future__ import annotations

import importlib

pytest = importlib.import_module("pytest")
pytest.importorskip("gamfit._rust")


def test_summary_payload_includes_edf_sigma2_loglik_aic_bic() -> None:
    import numpy as np
    import pandas as pd
    import gamfit

    rng = np.random.default_rng(7)
    x = np.linspace(-2.0, 2.0, 80)
    y = -0.2 + 0.8 * x + rng.normal(0.0, 0.1, size=x.shape[0])
    df = pd.DataFrame({"y": y, "x": x})

    model = gamfit.fit(df, "y ~ x")
    payload = model.summary().to_dict()

    for key in ("edf", "sigma2", "log_likelihood", "aic", "bic"):
        assert key in payload, f"summary payload must include '{key}' for table consistency"
        assert payload[key] is not None, f"summary payload field '{key}' must be populated"
