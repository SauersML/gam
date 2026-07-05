"""Bug hunt: the ``layer_transport_ladder`` composition-law test rejects a TRUE
null on near-noiseless chart chains (issue #2143).

The two-hop composition test studentizes the composition defect
``d(t) = h_ac(t) ⊖ (h_bc ∘ h_ab)(t)`` against the composed delta-method band
variance ``var(h_ac) + var(h_bc) + h_bc′²·var(h_ab)``. That variance is a pure
SAMPLING variance: it collapses toward zero as the adjacent REML transports
approach noiselessness. But composing two penalized-spline chart maps is not
closed in a single finite basis, so even a perfectly composable chain carries an
irreducible defect at the spline-representation scale (~1e-5 of the coordinate
span) that the sampling variance does not model. With only a RELATIVE variance
floor (``max_var·1e-10``), that machine-level defect is studentized against a
collapsed variance and inflated into a spurious ``composition_p_value = 0`` — and
the test is inverted: adding realistic noise makes the defect LARGER yet the test
correctly ACCEPTS, so a smaller, more-composable defect is judged far more
significant.

The fix adds an ABSOLUTE variance floor tied to the target chart's coordinate
span, so a representation-level defect reads as non-significant while genuine
violations and real sampling scales (both far above the floor) are unaffected.

For a deterministic 1-D chart chain the composition law is a mathematical
identity, so these are type-I (true-null) calibration tests: a composable chain
must NOT be flagged as a law violation.
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


def _two_hop(chain, topology="circle"):
    return gamfit.layer_transport_ladder(chain, topology=topology)["two_hop"][0]


def test_noiseless_composable_chain_is_accepted():
    """The core #2143 defect: a machine-level composition defect on a noiseless
    composable chain was reported as a strong violation (p = 0)."""
    r = _two_hop(_circle_chain(0.0, seed=3))
    # The defect really is at the spline-representation level (tiny relative to
    # the 2π coordinate span) — so it must NOT read as a law violation.
    assert r["composition_defect"] < 1e-3, (
        f"defect should be machine-level, got {r['composition_defect']:.3e}"
    )
    assert r["composition_p_value"] > 0.05, (
        f"noiseless composable chain flagged as a law violation: "
        f"p={r['composition_p_value']:.3e}, defect={r['composition_defect']:.3e}"
    )


def test_noiseless_accept_is_robust_across_seeds():
    for seed in (0, 3, 7, 11):
        r = _two_hop(_circle_chain(0.0, seed=seed))
        assert r["composition_p_value"] > 0.05, (
            f"seed {seed}: noiseless composable chain rejected "
            f"(p={r['composition_p_value']:.3e}, defect={r['composition_defect']:.3e})"
        )


def test_noisy_composable_chain_is_accepted_control():
    """Control: with realistic noise the sampling variance is well above the
    floor, so the test proceeds unchanged and (correctly) accepts."""
    r = _two_hop(_circle_chain(0.05, seed=3))
    assert r["composition_p_value"] > 0.05, (
        f"noisy composable chain rejected: p={r['composition_p_value']:.3e}"
    )


def test_smaller_defect_is_not_more_significant_than_larger():
    """The inversion the bug produced: the near-noiseless chain (smaller, more
    composable defect) must not be judged MORE significant than the noisy chain
    (larger defect). Both are composable, so both must accept."""
    clean = _two_hop(_circle_chain(0.0, seed=3))
    noisy = _two_hop(_circle_chain(0.05, seed=3))
    assert clean["composition_defect"] < noisy["composition_defect"], (
        "setup: the noiseless defect should be the smaller one"
    )
    assert clean["composition_p_value"] > 0.05 and noisy["composition_p_value"] > 0.05
    # A smaller defect must not yield a (much) smaller p-value than a larger one.
    assert clean["composition_p_value"] >= 0.5 * noisy["composition_p_value"], (
        f"test inverted: smaller defect p={clean['composition_p_value']:.3e} "
        f"< larger defect p={noisy['composition_p_value']:.3e}"
    )


def test_interval_topology_noiseless_is_accepted():
    """Exercises the Interval branch of the scale-aware floor: a composable chain
    on an interval chart must also accept when near-noiseless."""
    # Interval charts are the unit interval [0, 1]; keep every coordinate
    # strictly interior so the fold-free homeomorphism fits cleanly.
    rng = np.random.default_rng(5)
    n = 400
    a = np.sort(rng.uniform(0.03, 0.97, n))
    b = a + 0.02 * np.sin(2 * np.pi * a)
    c = b + 0.015 * np.sin(4 * np.pi * b)
    r = gamfit.layer_transport_ladder([a, b, c], topology="interval")["two_hop"][0]
    assert r["composition_p_value"] > 0.05, (
        f"noiseless composable interval chain rejected: "
        f"p={r['composition_p_value']:.3e}, defect={r['composition_defect']:.3e}"
    )


if __name__ == "__main__":
    test_noiseless_composable_chain_is_accepted()
    test_noiseless_accept_is_robust_across_seeds()
    test_noisy_composable_chain_is_accepted_control()
    test_smaller_defect_is_not_more_significant_than_larger()
    test_interval_topology_noiseless_is_accepted()
    print("all passed")
