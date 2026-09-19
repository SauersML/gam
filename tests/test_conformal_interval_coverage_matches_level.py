"""``predict(interval="conformal", conformal_level=L)`` must cover ~``L``.

With ``training_data`` (the training table) the conformal route is the exact full-conformal set
at the fitted smoothing parameters (``gam_predict::conformal_routes``, shared
with ``gam predict --conformal``), built at ``alpha = 1 - conformal_level``. The
jackknife+ route this path replaced over-covered at ``(1+L)/2`` because it
halved ``alpha`` to make the worst-case ``1 - 2*alpha`` bound read ``L``
(~0.75 at ``L=0.50``).

This test measures empirical marginal coverage at ``conformal_level=0.5`` over a
handful of seeds with a large test set per fit, and asserts the coverage is not
grossly inflated past the requested level.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit


def _truef(x: "np.ndarray") -> "np.ndarray":
    return np.sin(2.0 * np.pi * x)


def test_conformal_interval_coverage_matches_requested_level() -> None:
    level = 0.5
    n_train, n_test, seeds = 250, 3000, 6
    rng = np.random.default_rng(20260624)
    coverages = []
    for _ in range(seeds):
        xtr = rng.uniform(0.0, 1.0, n_train)
        ytr = _truef(xtr) + rng.normal(0.0, 0.3, n_train)
        train = pd.DataFrame({"x": xtr, "y": ytr})
        model = gamfit.fit(train, "y ~ s(x)")

        xte = rng.uniform(0.0, 1.0, n_test)
        yte = _truef(xte) + rng.normal(0.0, 0.3, n_test)
        out = model.predict(
            pd.DataFrame({"x": xte}),
            interval="conformal",
            training_data=train,
            conformal_level=level,
        )
        lo = np.asarray(out["posterior_mean_lower"], dtype=float)
        hi = np.asarray(out["posterior_mean_upper"], dtype=float)
        assert np.all(hi >= lo), "conformal interval must satisfy lower <= upper"
        coverages.append(float(np.mean((yte >= lo) & (yte <= hi))))

    cov = float(np.mean(coverages))
    # Requested coverage is 0.50. The halved-alpha (1+L)/2 law gives 0.75. Allow a
    # generous band for finite-sample conservativeness and Monte-Carlo noise, but
    # reject the gross over-coverage.
    assert cov <= 0.62, (
        f"interval='conformal' at conformal_level={level} covers {cov:.3f} of "
        f"held-out points, far above the requested {level}; the halved-alpha law "
        f"gives (1+level)/2 = {(1 + level) / 2:.3f}. "
        f"Per-seed coverages: {[round(c, 3) for c in coverages]}"
    )
