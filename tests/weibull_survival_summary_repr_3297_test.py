"""gam#3297: ``Model.summary()`` and ``repr(model)`` of a Weibull
(Royston-Parmar) survival fit report no scalar dispersion instead of raising,
because that family's scale contract has none."""

import numpy as np
import pandas as pd

import gamfit


def test_weibull_survival_summary_and_repr_3297():
    rng = np.random.default_rng(3)
    n = 500
    age = rng.uniform(40.0, 80.0, n)
    scale = np.exp(-(age - 60.0) / 20.0) * 10.0
    latent = scale * (-np.log(rng.uniform(0.0, 1.0, n))) ** (1.0 / 1.5)
    censor = rng.uniform(0.0, 20.0, n)
    frame = pd.DataFrame(
        {
            "entry": np.zeros(n),
            "exit": np.minimum(latent, censor),
            "event": (latent <= censor).astype(int),
            "age": age,
        }
    )
    model = gamfit.fit(
        frame, "Surv(entry, exit, event) ~ s(age)", survival_likelihood="weibull"
    )
    summary = model.summary()
    assert summary.scale is None
    assert model.scale is None
    assert np.isfinite(summary.aic_conditional)
    assert summary.aic_corrected is None
    assert repr(model).startswith("Model(")
    assert str(model)
