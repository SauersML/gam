"""Bug hunt: ``predict(interval="conformal", conformal_level=L)`` (the
jackknife+ path) systematically OVER-covers — it delivers ``(1+L)/2`` marginal
coverage instead of the advertised ``L``.

The documented contract (``gamfit/_model.py`` lines ~88-94) is that
``interval="conformal"`` gives ~``conformal_level`` marginal coverage. Empirically,
on i.i.d. Gaussian data with a smooth mean, the jackknife+ intervals cover at
``(1+L)/2``: ~0.75 at L=0.50, ~0.90 at L=0.80, ~0.945 at L=0.90 — and are
correspondingly ~2x too wide at low levels. (Split conformal via
``predict_conformal`` and ``interval="full_conformal"`` are correctly calibrated,
isolating the defect to this one path.)

Root cause (``crates/gam-pyffi/src/geometry_ffi.rs``, lines ~6332-6338):

    // ... set α = (1 − conformal_level) / 2.
    // Using α = 1 − conformal_level would yield only 1 − 2(1 − level) =
    // 2·level − 1 coverage (80% at level=0.9), mismatching the advertised guarantee.
    let alpha = (1.0 - conformal_level) / 2.0;

This conflates the jackknife+ *worst-case lower bound* (Barber et al. 2021:
coverage ≥ 1 − 2α) with the coverage actually delivered. A jackknife+ set built
at parameter α delivers ≈ 1 − α coverage in practice; the 1 − 2α bound is
pessimistic and almost never binds. Halving α to "guarantee" 1 − 2α = level
therefore yields ≈ 1 − α = (1+L)/2 typical coverage. The correct setting is
``alpha = 1 - conformal_level`` — exactly what the (correctly-calibrated)
``full_conformal`` path uses.

This test measures empirical marginal coverage at ``conformal_level=0.5`` over a
handful of seeds with a large test set per fit, and asserts the coverage is not
grossly inflated. It currently fails (coverage ~0.74, matching the buggy
``(1+0.5)/2 = 0.75``); once ``alpha = 1 - conformal_level`` the coverage drops to
~0.50 and the assertion holds without edits.
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


def test_jackknife_plus_conformal_coverage_matches_requested_level() -> None:
    level = 0.5
    n_train, n_test, seeds = 250, 3000, 6
    rng = np.random.default_rng(20260624)
    coverages = []
    for _ in range(seeds):
        xtr = rng.uniform(0.0, 1.0, n_train)
        ytr = _truef(xtr) + rng.normal(0.0, 0.3, n_train)
        model = gamfit.fit(pd.DataFrame({"x": xtr, "y": ytr}), "y ~ s(x)")

        xte = rng.uniform(0.0, 1.0, n_test)
        yte = _truef(xte) + rng.normal(0.0, 0.3, n_test)
        out = model.predict(
            pd.DataFrame({"x": xte}), interval="conformal", conformal_level=level
        )
        lo = np.asarray(out["mean_lower"], dtype=float)
        hi = np.asarray(out["mean_upper"], dtype=float)
        assert np.all(hi >= lo), "conformal interval must satisfy lower <= upper"
        coverages.append(float(np.mean((yte >= lo) & (yte <= hi))))

    cov = float(np.mean(coverages))
    # Requested coverage is 0.50. The buggy (1+L)/2 law gives 0.75. Allow a
    # generous band for jackknife+'s mild finite-sample conservativeness and
    # Monte-Carlo noise, but reject the gross over-coverage.
    assert cov <= 0.62, (
        f"jackknife+ interval='conformal' at conformal_level={level} covers "
        f"{cov:.3f} of held-out points, far above the requested {level}; this "
        f"matches the buggy (1+level)/2 = {(1 + level) / 2:.3f} law (alpha halved). "
        f"Per-seed coverages: {[round(c, 3) for c in coverages]}"
    )
