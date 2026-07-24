"""Public regression for the final binomial location-scale refusal in #2358.

The internal orchestration fixture uses sixty evenly spaced rows, alternating
binary outcomes, and an intercept-only noise model.  The inner solve reaches a
valid constrained mode on this geometry; historically the outer LAML
certificate nevertheless refused the no-wiggle pilot before the public fit
could mint.

Keep this as a public-wheel contract in addition to the Rust engine/reference
parity test: users care that the exact supported call returns a usable model,
not merely that two internal orchestration routes fail in the same way.
"""

from __future__ import annotations

import numpy as np

import gamfit


def test_binomial_location_scale_no_wiggle_fixture_mints() -> None:
    rows = 60
    data = {
        "x": np.linspace(-2.0, 2.0, rows),
        "y": (np.arange(rows) % 2 == 0).astype(float),
    }

    model = gamfit.fit(
        data,
        "y ~ x",
        family="binomial",
        noise_formula="1",
    )
    prediction = np.asarray(model.predict(data), dtype=float)

    assert prediction.shape == (rows,)
    assert np.all(np.isfinite(prediction))
    assert np.all((prediction >= 0.0) & (prediction <= 1.0))
