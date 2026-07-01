"""Bug hunt: the documented posterior-predictive replicate API is dead — the
Python wrapper calls an FFI symbol that was never added to the Rust module.

`gamfit.Model.sample_replicates` was introduced by commit 5a33b0703
("feat(#1057): wire generative replicate sampling + posterior-predictive
checks").  The commit message claims a "New
pyffi `generative_replicates(model_bytes, headers, rows, n_draws, seed)`" was
added, but the commit's diff only touched `gamfit/_model.py` and a test — the
`#[pyfunction] generative_replicates` it describes was never committed to
`crates/gam-pyffi/src/lib.rs`.  The Rust extension module therefore exposes
`build_sample_payload_json`, `posterior_samples_summary_json`, and
`sample_table`, but NOT `generative_replicates`:

    >>> import gamfit._rust as r
    >>> [s for s in dir(r) if "replic" in s or "generat" in s]
    []

So `gamfit/_model.py:483`

    return rust_module().generative_replicates(...)

raises `AttributeError: module 'gamfit._rust' has no attribute
'generative_replicates'` on every call.  The whole #1057 feature — replicate
sampling, posterior-predictive checks, simulation-based calibration — is
unreachable from Python, even though the Rust core
(`src/inference/generative.rs::sampleobservation_replicates`) is implemented.

This test asserts the user-facing contract with OBJECTIVE checks (the family
generative law is its own ground truth, no reference tool needed):

  * `sample_replicates` returns an `(n_draws, n_rows)` array;
  * for a Poisson fit every replicate is a non-negative integer (Poisson
    support), and the per-row replicate mean converges to the model's plug-in
    predicted rate `g^{-1}(Xβ̂)` as `n_draws` grows;
  * draws are seed-deterministic (same seed identical, different seed differs).

It fails today at the very first call (missing FFI symbol) and will pass
unchanged once `generative_replicates` is actually wired into the FFI.
"""

from __future__ import annotations

import importlib
import typing

pytest = typing.cast(typing.Any, importlib.import_module("pytest"))
pytest.importorskip("gamfit._rust")

import numpy as np

import gamfit


def _poisson_frame(n: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    rate = np.exp(0.4 + np.sin(2.0 * np.pi * x))
    y = rng.poisson(rate).astype(float)
    import pandas as pd

    return pd.DataFrame({"x": x, "y": y})


def test_sample_replicates_is_reachable_and_family_correct() -> None:
    import pandas as pd

    train = _poisson_frame(600, seed=0)
    model = gamfit.fit(train, "y ~ s(x)", family="poisson")

    newdata = pd.DataFrame({"x": np.linspace(0.05, 0.95, 12)})
    mu = np.asarray(model.predict(newdata), dtype=float)
    assert mu.shape == (12,)

    n_draws = 8000
    reps = np.asarray(model.sample_replicates(newdata, n_draws, seed=0), dtype=float)

    # Shape contract: (n_draws, n_rows).
    assert reps.shape == (n_draws, 12), f"expected (8000, 12), got {reps.shape}"

    # Poisson support: every replicate is a non-negative integer.
    assert np.all(reps >= 0.0), "Poisson replicates must be non-negative"
    assert np.allclose(reps, np.round(reps)), "Poisson replicates must be integers"

    # The generative law is the ground truth: the per-row replicate mean must
    # converge to the model's plug-in predicted rate. Poisson Monte-Carlo
    # standard error of the column mean is sqrt(mu / n_draws); allow 5 sigma
    # plus a small absolute floor.
    col_mean = reps.mean(axis=0)
    mc_se = np.sqrt(np.maximum(mu, 1e-6) / n_draws)
    assert np.all(np.abs(col_mean - mu) <= 5.0 * mc_se + 1e-3), (
        "replicate column means must track the predicted Poisson rate: "
        f"|mean - mu| = {np.abs(col_mean - mu)}, 5*mc_se = {5.0 * mc_se}"
    )

    # Seed determinism.
    reps_same = np.asarray(model.sample_replicates(newdata, 256, seed=0), dtype=float)
    reps_again = np.asarray(model.sample_replicates(newdata, 256, seed=0), dtype=float)
    assert np.array_equal(reps_same, reps_again), "same seed must reproduce draws"
    reps_diff = np.asarray(model.sample_replicates(newdata, 256, seed=1), dtype=float)
    assert not np.array_equal(reps_same, reps_diff), "different seed must differ"
