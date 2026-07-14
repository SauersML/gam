"""Wheel-surface pytest for the #2263/#2266 installed-state audit entry.

#2266 / #2263 item 4 asks for a way to certify an externally-trained
(torch-lane) SAE-manifold state — no closed-form solve — and get the same
certificate/diagnostics payload a native fit returns. There is no real
torch-trained artifact available at test-authoring time, so this test uses an
honest replay: fit a tiny model NATIVELY first (so its decoder, coordinates,
routing logits, and regularization state are genuinely converged, not
fabricated), then feed those
exact arrays back into ``gamfit.sae_manifold_certify_external`` as if they had
arrived from an external (e.g. torch) trainer, and check the certify path
reproduces the SAME pinned contract the Rust test checks:

* ``termination["verdict"] == "audited_stationary"`` and
  ``termination["evals"] == 0`` (no optimization ran on this path);
* ``structure_certificate`` is absent when structure search is disabled;
* ``reconstruction_r2`` is finite and ``fitted`` / ``assignments_z`` carry the
  right shapes;
* the certified atom count matches the seeded dictionary (structure search is
  disabled so nothing should grow or shrink it).

These tests are authored under the campaign rule "write the test, do NOT run
it"; they are HQ-verified centrally.
"""
from __future__ import annotations

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")


def _fit_circle(n: int = 200, noise: float = 0.15, seed: int = 0, n_iter: int = 30):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, 1.0, n)
    clean = np.column_stack([np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)])
    x = clean + noise * rng.standard_normal((n, 2))
    fit = gamfit.sae_manifold_fit(
        X=x, K=1, d_atom=1, atom_topology="circle", assignment="softmax",
        isometry_weight=0.0, ard_per_atom=True, sparsity_weight=0.01,
        smoothness_weight=0.01, n_iter=n_iter, learning_rate=1.0, random_state=seed,
    )
    return fit, x


def test_certify_external_round_trips_a_genuinely_converged_native_fit():
    fit, x = _fit_circle(seed=5)

    # Pull the genuinely-converged native state out of the fitted model, as if
    # it had come back from an external (e.g. torch) trainer: per-atom
    # decoder blocks, trained on-manifold coordinates, trained routing
    # logits, and the terminal regularization state the decoder was trained
    # under. None of these are fabricated -- they are read directly off the
    # native fit.
    decoder_blocks = [np.asarray(block, dtype=float) for block in fit.decoder_blocks]
    t_init = [np.asarray(block, dtype=float) for block in fit.coords]
    a_init = np.asarray(fit.low_level_logits, dtype=float)
    geometry_plans = list(fit.geometry_plans)

    assert fit.selected_log_lambda_smooth is not None, (
        "a converged fit must expose its selected log_lambda_smooth"
    )
    assert fit.selected_log_ard is not None, "a converged fit must expose its selected log_ard"
    assert fit.selected_log_lambda_sparse is not None, (
        "a converged fit must expose its selected log_lambda_sparse"
    )
    log_lambda_smooth = [float(v) for v in fit.selected_log_lambda_smooth]
    log_ard = [[float(v) for v in atom_ard] for atom_ard in fit.selected_log_ard]
    log_lambda_sparse = float(fit.selected_log_lambda_sparse)
    assert fit.tier0_scale is not None, "the native fit must expose its Tier-0 scale"

    report = gamfit.sae_manifold_certify_external(
        X=x,
        geometry_plans=geometry_plans,
        decoder_blocks=decoder_blocks,
        t_init=t_init,
        a_init=a_init,
        log_lambda_smooth=log_lambda_smooth,
        log_ard=log_ard,
        assignment=fit.assignment,
        alpha=float(fit.alpha),
        tau=float(fit.tau),
        log_lambda_sparse=log_lambda_sparse,
        tier0_mean=np.asarray(fit.training_mean, dtype=float),
        tier0_scale=np.asarray(fit.tier0_scale, dtype=float),
        learnable_alpha=bool(fit.learnable_alpha),
        top_k=None if fit.top_k is None else int(fit.top_k),
        threshold_gate_threshold=float(fit.threshold_gate_threshold),
        # Mirrors the Rust unit test's own choice: keep this deterministic and
        # cheap, and pin the certify entry's postlude on the state AS
        # PROVIDED rather than exercising structure search here (that is the
        # native fit entry's own coverage).
        run_structure_search=False,
    )

    # --- the pinned #2266 contract (same as the Rust unit test) -----------
    assert report["status"] == "certified"
    assert report["is_fit"] is True
    assert report["termination"]["verdict"] == "audited_stationary", (
        "the fit must identify the exact-point audit rather than claim a search ran"
    )
    assert report["termination"]["evals"] == 0, (
        "the exact-point audit must not claim an optimization iteration"
    )
    assert np.isfinite(report["reconstruction_r2"]), "reconstruction R^2 must be finite"

    assert report["structure_search"] is None
    assert report["structure_certificate"] is None

    assert "certificates" in report
    assert isinstance(report["certificates"], dict)
    assert "overall" in report["certificates"]

    fitted = np.asarray(report["fitted"])
    assert fitted.shape == x.shape, "the certified reconstruction must match the target shape"
    np.testing.assert_allclose(fitted, np.asarray(fit.fitted), rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(report["tier0_mean"], fit.training_mean)
    np.testing.assert_allclose(report["tier0_scale"], fit.tier0_scale)
    for certified_atom, decoder in zip(report["atoms"], decoder_blocks):
        np.testing.assert_allclose(
            certified_atom["decoder_B"], decoder, rtol=1e-12, atol=1e-12
        )

    assignments = np.asarray(report["assignments_z"])
    assert assignments.shape[0] == x.shape[0], (
        "assignments must carry one row per observation"
    )

    assert report["chosen_k"] == len(decoder_blocks), (
        "with structure search disabled, the certified atom count is exactly the seed's"
    )
    assert len(report["atoms"]) == len(decoder_blocks)

    assert np.isfinite(report["penalized_quasi_laplace_criterion"]), (
        "the penalized objective evaluated at the installed state must be finite"
    )


def test_certify_external_returns_typed_nonfit_for_perturbed_state():
    fit, x = _fit_circle(seed=11)
    assert fit.tier0_scale is not None, "the native fit must expose its Tier-0 scale"
    decoder_blocks = [np.asarray(block, dtype=float).copy() for block in fit.decoder_blocks]
    decoder_blocks[0].flat[0] += 0.25
    report = gamfit.sae_manifold_certify_external(
        X=x,
        geometry_plans=list(fit.geometry_plans),
        decoder_blocks=decoder_blocks,
        t_init=[np.asarray(block, dtype=float) for block in fit.coords],
        a_init=np.asarray(fit.low_level_logits, dtype=float),
        log_lambda_smooth=[float(v) for v in fit.selected_log_lambda_smooth],
        log_ard=[[float(v) for v in atom] for atom in fit.selected_log_ard],
        assignment=fit.assignment,
        alpha=float(fit.alpha),
        tau=float(fit.tau),
        log_lambda_sparse=float(fit.selected_log_lambda_sparse),
        tier0_mean=np.asarray(fit.training_mean, dtype=float),
        tier0_scale=np.asarray(fit.tier0_scale, dtype=float),
        learnable_alpha=bool(fit.learnable_alpha),
        top_k=None if fit.top_k is None else int(fit.top_k),
        threshold_gate_threshold=float(fit.threshold_gate_threshold),
        run_structure_search=False,
    )

    assert report["status"] == "nonstationary"
    assert report["is_fit"] is False
    assert report["optimization_iterations"] == 0
    assert report["structure_search"] is None
    assert report["structure_certificate"] is None
    assert "fitted" not in report
    assert "termination" not in report
    assert report["inner_kkt"]["certifies"] is False or (
        report["outer_stationarity"]["projected_gradient_norm"]
        > report["outer_stationarity"]["stationarity_bound"]
    )


def test_certify_external_requires_matching_per_atom_metadata_lengths():
    fit, x = _fit_circle(seed=9)
    decoder_blocks = [np.asarray(block, dtype=float) for block in fit.decoder_blocks]
    t_init = [np.asarray(block, dtype=float) for block in fit.coords]
    a_init = np.asarray(fit.low_level_logits, dtype=float)

    with pytest.raises(ValueError):
        gamfit.sae_manifold_certify_external(
            X=x,
            # Two geometry plans, one decoder block: a deliberate
            # per-atom-metadata length mismatch that must be a clean error,
            # not a panic or a silently-wrong certificate.
            geometry_plans=list(fit.geometry_plans) * 2,
            decoder_blocks=decoder_blocks,
            t_init=t_init,
            a_init=a_init,
            log_lambda_smooth=[0.0, 0.0],
            log_ard=[[], []],
            run_structure_search=False,
        )
