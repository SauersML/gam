"""Bug hunt: ``compare_models`` silently compares fits made on *different*
data (different numbers of observations), returning a confident but meaningless
winner driven purely by the sample size.

``gamfit.compare_models([m_a, m_b])`` ranks fits by the conditional AIC
``score = −2·loglik + 2·edf`` and reports a "winner" plus a Bayes-factor
column. AIC / REML evidence are only comparable across models fit to the **same
response on the same observations** — ``−2·loglik`` grows with the number of
observations ``n``, so two fits with different ``n`` (hence different data) live
on incomparable scales and their score difference is not a Bayes factor.

The library already knows this for one axis of incomparability: comparing fits
of *different families* raises a clear error

    "compare_models: cannot compare fits of different response families ...
     Compare models fit to the same response under the same family."

(see ``crates/gam-solve/src/evidence.rs`` ``compare_reml_fits`` and the FFI in
``crates/gam-pyffi/src/model/model_ffi.rs``). But the same routine carries no
observation-count guard: ``RemlCandidate`` stores ``score``, ``edf``,
``log_lik`` and ``family`` but never ``n``, so a mismatch in ``n`` cannot be
detected and the comparison proceeds.

Observed: comparing a Gaussian ``y ~ s(x)`` fit on n=500 against an
*identically distributed* one on n=100 declares the n=100 model the winner on
every seed, by a Bayes factor of ~1e14–1e18 — purely because fewer
observations give a less-negative total log-likelihood. The fits are of equal
statistical quality (same data-generating process, same noise level); the
"winner" is an artifact of sample size.

Expected: ``compare_models`` must refuse to compare fits made on a different
number of observations (the same way it refuses different families), rather than
returning a confident, sample-size-driven verdict.

The test asserts that comparing two same-family fits with different ``n``
raises, and — as a sanity reference — that the existing family-mismatch guard
does raise.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit


def _gaussian_fit(n: int, seed: int):
    """Identical Gaussian DGP and noise level; only the sample size differs."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    y = np.sin(2.0 * np.pi * x) + rng.normal(0.0, 0.3, n)
    return gamfit.fit({"y": y, "x": x}, "y ~ s(x)", family="gaussian")


def test_compare_models_rejects_different_observation_counts() -> None:
    m_big = _gaussian_fit(n=500, seed=0)
    m_small = _gaussian_fit(n=100, seed=100)

    # Two same-family fits on different-sized (hence different) data are not
    # comparable by AIC/evidence; comparing them must raise, not silently pick a
    # winner. (Today it returns the n=100 model with a Bayes factor ~1e14-1e18.)
    with pytest.raises(Exception) as excinfo:
        gamfit.compare_models([m_big, m_small], names=["n500", "n100"])

    msg = str(excinfo.value).lower()
    assert "compare_models" in msg or "observ" in msg or "same" in msg, (
        "compare_models raised, but not with a message explaining the "
        f"incomparable-data reason: {excinfo.value!r}"
    )


def test_compare_models_family_guard_reference_still_fires() -> None:
    """Reference: the analogous different-family guard already raises. This
    pins the established contract that compare_models refuses incomparable
    fits, so the n-guard above is the consistent extension, not a new policy."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 1.0, 300)
    m_gauss = gamfit.fit(
        {"y": np.sin(x) + rng.normal(0.0, 0.2, 300), "x": x},
        "y ~ s(x)",
        family="gaussian",
    )
    m_pois = gamfit.fit(
        {"y": rng.poisson(np.exp(x)).astype(float), "x": x},
        "y ~ s(x)",
        family="poisson",
    )
    with pytest.raises(Exception):
        gamfit.compare_models([m_gauss, m_pois])
