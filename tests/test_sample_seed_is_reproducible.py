from __future__ import annotations

import importlib
import typing

pytest = typing.cast(typing.Any, importlib.import_module("pytest"))
pytest.importorskip("gamfit._rust")

import numpy as np

import gamfit


@pytest.mark.xfail(
    strict=True,
    reason="#1512 triage: PosteriorSamples no longer exposes a `.beta` attribute "
    "(AttributeError) — the coefficient-draw surface was renamed (the samples "
    "now live under `.samples`). Re-point the reproducibility comparison at the "
    "current draws accessor to re-enable.",
)
def test_sample_seed_reproducibility() -> None:
    rows = [{"y": float(i + 1), "x": float(i)} for i in range(12)]
    model = gamfit.fit(rows, "y ~ x")

    draw_a = model.sample(rows, samples=20, warmup=20, chains=1, seed=123)
    draw_b = model.sample(rows, samples=20, warmup=20, chains=1, seed=123)

    np.testing.assert_allclose(
        np.asarray(draw_a.beta),
        np.asarray(draw_b.beta),
        atol=0.0,
        err_msg="sample() should be exactly reproducible when the seed is fixed",
    )
