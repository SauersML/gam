"""Bug hunt: the ``layer_transport_ladder`` composition-law test rejects a TRUE
null on near-noiseless chart chains (issue #2143), revisited by #3364.

The two-hop composition test studentizes the composition defect
``d(t) = h_ac(t) ⊖ (h_bc ∘ h_ab)(t)`` against the composed delta-method band
variance ``var(h_ac) + var(h_bc) + h_bc′²·var(h_ab)``. That variance is a
SAMPLING variance, and a noiseless chart chain has no sampling variability:
there is no null distribution to calibrate a p-value against.

So the pair law is declared by the caller. ``pairs="deterministic"`` says every
coordinate is an exact function of the row, and each transport is the exact
minimum-curvature interpolant of its pairs: no smoothing parameter, no
dispersion, zero band. The composition defect is then a pure interpolation
error, which collapses with the site spacing, and the report carries
``composition_p_value = None`` rather than a statistic divided by a variance
that does not exist. ``pairs="stochastic"`` (the default) keeps the REML band
and the calibrated test, which must accept a composable noisy chain.

For a deterministic 1-D chart chain the composition law is a mathematical
identity, so a composable chain must never be flagged as a law violation.
"""

import os

os.environ.setdefault("GAM_LOG", "off")

import numpy as np

import gamfit


def _circle_chain(noise, seed=3, n=400):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 2 * np.pi, n))
    b1 = t + 0.3 * np.sin(t)
    b2 = b1 + 0.2 * np.sin(2 * b1)
    jit = (lambda: rng.normal(0, noise, n)) if noise else (lambda: 0.0)
    return [t + jit(), b1 + jit(), b2 + jit()]


def _two_hop(chain, topology="circle", pairs="stochastic"):
    return gamfit.sae.layer_transport_ladder(chain, topology=topology, pairs=pairs)[
        "two_hop"
    ][0]


def test_noiseless_composable_chain_is_accepted():
    """The core #2143 defect: a noiseless composable chain was reported as a
    strong violation (p = 0). Declared deterministic, it carries only the
    interpolation error and no p-value at all."""
    r = _two_hop(_circle_chain(0.0, seed=3), pairs="deterministic")
    assert r["pairs"] == "deterministic"
    # The defect is interpolation error, tiny relative to the 2π coordinate span.
    assert r["composition_defect"] < 1e-3, (
        f"defect should be interpolation-level, got {r['composition_defect']:.3e}"
    )
    assert r["composition_p_value"] is None, (
        f"a deterministic chain has no sampling null; got p={r['composition_p_value']}"
    )


def test_noiseless_accept_is_robust_across_seeds():
    for seed in (0, 3, 7, 11):
        r = _two_hop(_circle_chain(0.0, seed=seed), pairs="deterministic")
        assert r["composition_defect"] < 1e-3, (
            f"seed {seed}: defect {r['composition_defect']:.3e}"
        )
        assert r["composition_p_value"] is None


def test_noisy_composable_chain_is_accepted_control():
    """Control: with realistic noise the stochastic law's sampling band is the
    right yardstick, and the composable chain is accepted."""
    r = _two_hop(_circle_chain(0.05, seed=3))
    assert r["pairs"] == "stochastic"
    assert r["composition_p_value"] > 0.05, (
        f"noisy composable chain rejected: p={r['composition_p_value']:.3e}"
    )


def test_smaller_defect_is_not_more_significant_than_larger():
    """The inversion the bug produced: the noiseless chain (smaller, more
    composable defect) was judged MORE significant than the noisy chain. Under
    the declared laws the clean chain has the smaller defect and no p-value,
    and the noisy chain is accepted."""
    clean = _two_hop(_circle_chain(0.0, seed=3), pairs="deterministic")
    noisy = _two_hop(_circle_chain(0.05, seed=3))
    assert clean["composition_defect"] < noisy["composition_defect"], (
        "setup: the noiseless defect should be the smaller one"
    )
    assert clean["composition_p_value"] is None
    assert noisy["composition_p_value"] > 0.05


def test_interval_topology_noiseless_is_accepted():
    """A composable interval chain declared deterministic has an
    interpolation-level defect and no p-value."""
    rng = np.random.default_rng(5)
    n = 400
    a = np.sort(rng.uniform(0.03, 0.97, n))
    b = a + 0.02 * np.sin(2 * np.pi * a)
    c = b + 0.015 * np.sin(4 * np.pi * b)
    r = _two_hop([a, b, c], topology="interval", pairs="deterministic")
    assert r["composition_defect"] < 1e-3, (
        f"noiseless interval defect {r['composition_defect']:.3e}"
    )
    assert r["composition_p_value"] is None


if __name__ == "__main__":
    test_noiseless_composable_chain_is_accepted()
    test_noiseless_accept_is_robust_across_seeds()
    test_noisy_composable_chain_is_accepted_control()
    test_smaller_defect_is_not_more_significant_than_larger()
    test_interval_topology_noiseless_is_accepted()
    print("all passed")
