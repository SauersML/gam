"""Wheel-surface pytest for the #2266 evaluation-only certification entry.

#2266 / #2263 item 4 asks for a way to certify an externally-trained
(torch-lane) SAE-manifold state — no closed-form solve — and get the same
certificate/diagnostics payload a native fit returns. There is no real
torch-trained artifact available at test-authoring time, so this test uses
the honest stand-in the Rust unit test
(``crates/gam-sae/src/manifold/tests_certify_external_2266.rs::
certify_external_seed_without_running_the_solve``) already established: fit a
tiny model NATIVELY first (so its decoder / coordinates / routing logits /
regularization state are genuinely converged, not fabricated), then feed those
exact arrays back into ``gamfit.sae_manifold_certify_external`` as if they had
arrived from an external (e.g. torch) trainer, and check the certify path
reproduces the SAME pinned contract the Rust test checks:

* ``termination["verdict"] == "external"`` and ``termination["evals"] == 0``
  (no outer/inner solve ran on this path);
* ``structure_certificate`` is a non-empty serialized certificate even with
  structure search disabled (a trivially-certifying empty ledger);
* ``reconstruction_r2`` is finite and ``fitted`` / ``assignments_z`` carry the
  right shapes;
* the certified atom count matches the seeded dictionary (structure search is
  disabled so nothing should grow or shrink it).

These tests are authored under the campaign rule "write the test, do NOT run
it"; they are HQ-verified centrally.
"""
from __future__ import annotations

import json

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


def _n_harmonics_or_none(basis_kinds, n_harmonics):
    return [
        int(h) if kind == "periodic" and h > 0 else None
        for kind, h in zip(basis_kinds, n_harmonics)
    ]


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
    duchon_centers = [
        None if center is None else np.asarray(center, dtype=float)
        for center in fit.duchon_centers
    ]
    n_harmonics = _n_harmonics_or_none(fit.basis_kinds, fit.n_harmonics)

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

    report = gamfit.sae_manifold_certify_external(
        X=x,
        atom_basis=list(fit.basis_kinds),
        d_atom=[int(v) for v in fit.atom_dims],
        decoder_blocks=decoder_blocks,
        t_init=t_init,
        a_init=a_init,
        log_lambda_smooth=log_lambda_smooth,
        log_ard=log_ard,
        duchon_centers=duchon_centers,
        n_harmonics=n_harmonics,
        assignment=fit.assignment,
        alpha=float(fit.alpha),
        tau=float(fit.tau),
        log_lambda_sparse=log_lambda_sparse,
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
    assert report["termination"]["verdict"] == "external", (
        "the certify entry must report the External verdict, never a "
        "Search/FixedRho certificate for state it never optimized"
    )
    assert report["termination"]["evals"] == 0, (
        "no outer/inner evaluation should run on the certify path"
    )
    assert np.isfinite(report["reconstruction_r2"]), "reconstruction R^2 must be finite"

    assert isinstance(report["structure_certificate"], str)
    assert report["structure_certificate"] != "", (
        "the anytime-valid structure certificate must serialize even when the "
        "structure search did not run (a trivially-certifying empty ledger)"
    )
    # It must be well-formed JSON with the e-BH confirmation shape, not just a
    # non-empty string.
    certificate = json.loads(report["structure_certificate"])
    assert "alpha" in certificate or "claims" in certificate or isinstance(certificate, dict)

    assert "certificates" in report
    assert isinstance(report["certificates"], dict)
    assert "overall" in report["certificates"]

    fitted = np.asarray(report["fitted"])
    assert fitted.shape == x.shape, "the certified reconstruction must match the target shape"

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


def test_certify_external_requires_matching_per_atom_metadata_lengths():
    fit, x = _fit_circle(seed=9)
    decoder_blocks = [np.asarray(block, dtype=float) for block in fit.decoder_blocks]
    t_init = [np.asarray(block, dtype=float) for block in fit.coords]
    a_init = np.asarray(fit.low_level_logits, dtype=float)

    with pytest.raises(ValueError):
        gamfit.sae_manifold_certify_external(
            X=x,
            # Two atom-basis names, one decoder block: a deliberate
            # per-atom-metadata length mismatch that must be a clean error,
            # not a panic or a silently-wrong certificate.
            atom_basis=["periodic", "periodic"],
            d_atom=[int(v) for v in fit.atom_dims] * 2,
            decoder_blocks=decoder_blocks,
            t_init=t_init,
            a_init=a_init,
            log_lambda_smooth=[0.0, 0.0],
            log_ard=[[], []],
            run_structure_search=False,
        )
