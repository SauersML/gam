"""End-to-end Python reachability of the landed inference instruments.

Covers the three previously-unwired instruments now exposed through the
``gamfit`` Python API:

* #1013 functorial layer transport (``layer_transport_fit`` /
  ``layer_transport_ladder``),
* #984 anytime-valid structure discovery (``atom_birth_gate`` /
  ``e_bh_dictionary_certificate`` / ``split_likelihood_log_e`` /
  ``log_e_from_p_value``),
* #939 Lawley likelihood-ratio Bartlett correction (``lawley_bartlett_factor``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")
pytest.importorskip("gamfit._rust")


def test_layer_transport_fit_reaches_python():
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0.0, 2.0 * math.pi, size=200))
    # Identity-ish circle->circle map plus small wiggle: degree 1, low defect.
    s = (t + 0.05 * np.sin(t)) % (2.0 * math.pi)
    report = gamfit.layer_transport_fit(t, s, "circle", "circle")
    assert report["degree"] == 1
    assert report["topology_preserved"] is True
    assert report["isometry_defect"] >= 0.0
    assert report["isometry_defect_se"] >= 0.0


def test_structure_discovery_gate_and_certificate():
    # split-LR log e-value is just the likelihood gap.
    assert gamfit.split_likelihood_log_e(-8.0, -10.0) == pytest.approx(2.0)

    # Gate certifies once the running e-process supremum crosses 1/alpha.
    gate = gamfit.atom_birth_gate(0.05)
    assert gate.certified() is False
    gate.absorb_shard(-8.0, -10.0)  # log e = 2.0
    gate.absorb_shard(-8.0, -10.0)  # cumulative 4.0 > log(1/0.05)
    assert gate.certified() is True
    verdict = gate.verdict()
    assert verdict["verdict"] == "certified"
    assert verdict["log_e"] == pytest.approx(4.0)

    # e-BH dictionary certificate confirms the overwhelming claim only.
    confirmed = gamfit.e_bh_dictionary_certificate([25.0, 0.01, -0.2, 0.0], 0.05)
    assert confirmed == [0]

    # p->e calibration is the conservative 1/p lower bound family.
    assert gamfit.log_e_from_p_value(0.04) == pytest.approx(math.log(2.5))


def test_lawley_bartlett_factor_exponential_fixture():
    # Exponential (Gamma-log, phi=1), intercept-only: ε_0 = 0 so the factor is
    # the certified textbook c = 1 + 1/(6n).
    n = 32
    design = np.ones((n, 1))
    eta = np.full(n, 0.4)
    out = gamfit.lawley_bartlett_factor(
        design, "gamma", eta, 0, 1, 1.0, dispersion=1.0, lr_statistic=5.0
    )
    assert out["bartlett_factor"] == pytest.approx(1.0 + 1.0 / (6.0 * n), rel=1e-8)
    # Corrected statistic divides by the factor; corrected p-value is larger.
    assert out["corrected_statistic"] == pytest.approx(5.0 / out["bartlett_factor"])
    assert out["p_value_corrected"] >= out["p_value_uncorrected"]
