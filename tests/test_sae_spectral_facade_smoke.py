"""Smoke tests for the SAE spectral / routing-geometry facade wrappers.

Exercises each newly exposed Rust binding through its thin ``gamfit`` facade on
tiny synthetic data: the dimension spectrometer, block-firing circle
coordinates, the routability floor + audit, the sparse-dict dual certificate,
harmonic super-resolution, and the contract / holonomy calculus. These assert
the FFI marshals shapes and dicts correctly and that the numbers are finite and
in-range, not the full statistical contract (that lives in the Rust unit tests).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gamfit import (
    atlas_nerve_diagram,
    block_firing_coordinates,
    block_sparse_dictionary_fit,
    compose_contracts,
    conditional_coactivation_influence,
    coupling_robustness_certificate,
    dimension_spectrometer,
    effect_weighted_retention,
    loop_holonomy,
    recover_spikes,
    routability_audit,
    routability_floor,
    separation_limit,
    sparse_dict_dual_certificate,
    sparse_dictionary_fit,
)


def _planted_atoms(rng, k, p, n):
    q, _ = np.linalg.qr(rng.standard_normal((p, p)))
    atoms = q[:k].astype(np.float32)
    x = np.zeros((n, p), dtype=np.float32)
    for row in range(n):
        primary = row % k
        x[row] = np.float32(0.7 + 0.01 * (row // k)) * atoms[primary]
    return x, atoms


def test_dimension_spectrometer_smoke():
    rng = np.random.default_rng(0)
    # Rows spread along a low-dimensional planted structure in R^12.
    x, _ = _planted_atoms(rng, k=6, p=12, n=400)
    x = x + np.float32(0.01) * rng.standard_normal(x.shape).astype(np.float32)
    report = dimension_spectrometer(x, k_min=2, n_doublings=3, max_epochs=8)
    assert len(report.rungs) == 4  # n_doublings + 1 rungs
    assert [k for k, _ in report.rungs] == [2, 4, 8, 16]
    assert all(math.isfinite(loss) for _, loss in report.rungs)
    assert math.isfinite(report.slope)
    assert math.isfinite(report.d_hat) or report.floor_saturated
    assert isinstance(report.floor_saturated, bool)


def test_routability_floor_smoke():
    floor = routability_floor(p=64, n_blocks=32, b_max=2, delta=0.05)
    assert floor.p == 64 and floor.n_blocks == 32 and floor.b_max == 2
    assert floor.floor > 0.0 and math.isfinite(floor.floor)
    assert 0.0 < floor.minimum_routable_energy < 1.0
    # Closed form: sqrt(b/p) + sqrt(2 ln(K/delta)/p).
    expected = math.sqrt(2 / 64) + math.sqrt(2 * math.log(32 / 0.05) / 64)
    assert floor.floor == pytest.approx(expected, rel=1e-9)


def test_routability_floor_rejects_degenerate():
    with pytest.raises(ValueError):
        routability_floor(p=0, n_blocks=4, b_max=1, delta=0.05)
    with pytest.raises(ValueError):
        routability_floor(p=4, n_blocks=4, b_max=8, delta=0.05)


def test_routability_audit_smoke():
    rng = np.random.default_rng(1)
    p, k = 16, 8
    q, _ = np.linalg.qr(rng.standard_normal((p, p)))
    decoder = q[:k].astype(np.float32)  # unit-norm atom rows
    residuals = rng.standard_normal((200, p)).astype(np.float32)
    audit = routability_audit(
        decoder, residuals, block_size=1, delta=0.05, quantile_levels=(0.5, 0.9, 0.99)
    )
    assert audit.n_rows == 200
    assert len(audit.quantiles) == 3
    assert audit.floor.p == p and audit.floor.n_blocks == k
    assert math.isfinite(audit.coherence_excess)
    assert 0.0 <= audit.fraction_below_floor <= 1.0


def test_sparse_dict_dual_certificate_smoke():
    rng = np.random.default_rng(2)
    x, _ = _planted_atoms(rng, k=6, p=12, n=300)
    fit = sparse_dictionary_fit(x, k=6, active=1, max_epochs=12)
    report = sparse_dict_dual_certificate(
        x, fit.decoder, fit.indices, fit.codes, max_candidates=8
    )
    assert report.n_rows == x.shape[0]
    assert 0.0 <= report.frac_certified <= 1.0
    assert len(report.optimality_ratio_quantiles) == 4
    assert len(report.birth_candidates) <= 8


def test_block_firing_coordinates_smoke():
    rng = np.random.default_rng(3)
    # Plant b=2 circle blocks: each row is a phasor in one of G 2-planes.
    g, p, n = 4, 16, 240
    q, _ = np.linalg.qr(rng.standard_normal((p, p)))
    frames = q[: 2 * g].astype(np.float32)  # (2G) x P, block g = rows [2g, 2g+2)
    x = np.zeros((n, p), dtype=np.float32)
    for row in range(n):
        blk = row % g
        theta = 2 * math.pi * rng.random()
        z = np.float32([math.cos(theta), math.sin(theta)])
        x[row] = z[0] * frames[2 * blk] + z[1] * frames[2 * blk + 1]
    fit = block_sparse_dictionary_fit(x, n_blocks=g, block_size=2, block_topk=1, max_epochs=12)
    report = block_firing_coordinates(fit.decoder, fit.blocks, fit.gates, fit.codes, block=0)
    assert report.n_firings == report.t.shape[0]
    assert report.t.shape == report.amplitude.shape
    if report.n_firings > 0:
        assert np.all((report.t >= 0.0) & (report.t < 1.0))
        assert np.all(report.amplitude >= 0.0)


def test_super_resolution_smoke():
    assert separation_limit(8) == pytest.approx(2.0 / 8)
    assert math.isinf(separation_limit(0))
    # Two well-separated spikes on the circle, evaluated at harmonics 1..H.
    spikes = [(0.3, 0.12), (0.75, 0.4)]  # (amplitude, t)
    harmonics = 12
    coeffs = []
    for h in range(1, harmonics + 1):
        c = sum(a * math.cos(2 * math.pi * h * t) for a, t in spikes)
        s = sum(a * math.sin(2 * math.pi * h * t) for a, t in spikes)
        coeffs.append((c, s))
    recovery = recover_spikes(coeffs, sigma=0.0)
    assert recovery.model_order >= 1
    assert recovery.t.shape == recovery.amplitude.shape
    assert recovery.hankel_singular_values.ndim == 1
    assert math.isfinite(recovery.residual)


def test_compose_contracts_smoke():
    chain = [
        ("encode", 1.0, 0.01, 1.2),
        ("transport", 1.0, 0.02, 0.9),
        ("decode", 1.0, 0.005, 1.0),
    ]
    composed = compose_contracts(chain)
    assert len(composed.per_stage_contribution) == 3
    assert composed.total_defect == pytest.approx(sum(composed.per_stage_contribution))
    assert isinstance(composed.domain_ok, bool)


def test_loop_holonomy_smoke():
    # A trivial rotation loop that returns to identity.
    edges = [(1, 0.5), (1, -0.5)]
    defects = [0.01, 0.01]
    report = loop_holonomy(edges, defects)
    assert report.loop_len == 2
    assert report.net_sign == 1
    assert report.net_angle == pytest.approx(0.0, abs=1e-9)
    assert report.is_trivial
    # A single reflection is never trivial.
    reflect = loop_holonomy([(-1, 0.0)], [0.0])
    assert reflect.net_sign == -1
    assert not reflect.is_trivial


def test_atlas_nerve_diagram_smoke():
    # Two b = 2 blocks (K = 4) firing on disjoint rows: a 2-chart nerve.
    rng = np.random.default_rng(3)
    n = 64
    codes = np.zeros((n, 4), dtype=np.float32)
    codes[: n // 2, 0:2] = rng.standard_normal((n // 2, 2)).astype(np.float32) + 1.0
    codes[n // 2 :, 2:4] = rng.standard_normal((n // 2, 2)).astype(np.float32) + 1.0
    report = atlas_nerve_diagram(codes, block_size=2)
    assert report.computed
    assert report.chart_blocks == [0, 1]
    assert set(report.betti) == {"b0", "b1", "b2"}
    assert report.betti["b0"] >= 1
    assert report.n_vertices >= 2
    assert report.n_edges >= 1
    assert math.isfinite(report.max_filtration)
    assert isinstance(report.note, str)
    # A scalar (block_size == 1) shape is not applicable: computed is False.
    skipped = atlas_nerve_diagram(codes, block_size=1)
    assert not skipped.computed
    assert skipped.reason is not None


def test_conditional_coactivation_influence_smoke():
    # gate_j fires on exactly the rows where gate_i fires -> P(j|i) == 1.
    active_i = [True, True, False, True, False]
    active_j = [True, True, False, True, False]
    rows = [0, 1, 2, 3, 4]
    weights = [1.0, 1.0, 1.0, 1.0, 1.0]
    report = conditional_coactivation_influence(active_i, active_j, rows, weights)
    assert report.conditional_probability == pytest.approx(1.0)
    assert report.active_mass_i > 0.0
    assert len(report.psi) == len(rows)
    assert len(report.normalized_weights) == len(rows)
    assert all(math.isfinite(v) for v in report.psi)


def test_coupling_robustness_certificate_smoke():
    rng = np.random.default_rng(4)
    gate_i = rng.standard_normal(48)
    gate_j = gate_i + 0.1 * rng.standard_normal(48)  # strongly coupled
    rows = list(range(48))
    weights = [1.0] * 48
    cert = coupling_robustness_certificate(gate_i, gate_j, rows, weights, epsilon=0.05)
    assert -1.0 <= cert.rho <= 1.0
    assert cert.rho > 0.0
    assert cert.robustness_radius_epsilon >= 0.0
    assert cert.epsilon == pytest.approx(0.05)
    assert math.isfinite(cert.worst_case_coupling)
    assert len(cert.psi) == len(rows)


def test_effect_weighted_retention_smoke():
    # atom 0: variance charge clears its budget -> retained_by_variance.
    # atom 1: only Fisher-effect firings -> retained_by_effect (or not), never crashes.
    variance = [(5.0, 1.0), None]
    firings = [(0, 0.0), (1, 8.0), (1, 8.0)]
    atoms = effect_weighted_retention(variance, firings)
    assert len(atoms) == 2
    assert atoms[0].atom == 0
    assert atoms[0].retained_by_variance
    assert atoms[0].retained
    assert isinstance(atoms[1].retained, bool)
