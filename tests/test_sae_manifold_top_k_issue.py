"""Tests pinning the top_k / target_k contract for SAE manifold fits.

The Rust ``sae_manifold_fit_minimal`` ``#[pyfunction]`` takes ``top_k`` and
owns the hard per-row top-k projection end to end; ``gamfit/_sae_manifold.py``
forwards it unconditionally and ``gamfit.torch.manifold_sae.ManifoldSAE.fit``
honours ``cfg.sparsity.target_k``.

Each path must produce assignments where at most ``top_k`` atoms are active
per row. Silently dropping the parameter is the regression these tests guard.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamfit


def _random_inputs() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((24, 4))


def test_sae_manifold_fit_top_k_one_yields_at_most_one_active_atom() -> None:
    """Public ``gamfit.sae_manifold_fit`` must honour ``top_k=1``."""
    X = _random_inputs()
    fit = gamfit.sae_manifold_fit(
        X=X,
        K=3,
        atom_basis="periodic",
        d_atom=1,
        assignment="softmax",
        top_k=1,
        n_iter=5,
        random_state=0,
    )
    A = np.asarray(fit.assignments)
    active_per_row = (A > 1e-12).sum(axis=1)
    assert int(active_per_row.max()) <= 1, (
        f"top_k=1 should leave at most 1 atom active per row, got max={int(active_per_row.max())}"
    )


@pytest.mark.parametrize("k", [1, 2])
def test_sae_manifold_fit_top_k_general(k: int) -> None:
    X = _random_inputs()
    fit = gamfit.sae_manifold_fit(
        X=X,
        K=4,
        atom_basis="periodic",
        d_atom=1,
        assignment="softmax",
        top_k=k,
        n_iter=5,
        random_state=0,
    )
    A = np.asarray(fit.assignments)
    active_per_row = (A > 1e-12).sum(axis=1)
    assert int(active_per_row.max()) <= k


def _euclidean_data_fit(z: np.ndarray, fitted: np.ndarray) -> float:
    """Reconstruction data-fit ``0.5 * sum (z - fitted)**2`` (Euclidean, no row
    weights) — the exact term the Rust ``penalized_loss`` carries under the
    default (no Fisher shard / no honesty-weight) path."""
    resid = z - fitted
    return 0.5 * float(np.sum(resid * resid))


def _raw_topk_payload(z: np.ndarray, *, K: int, top_k: int) -> dict:
    """Fit through the low-level FFI so the test can read the raw payload dict
    (``penalized_loss_breakdown``, ``pre_topk``, ``top_k_projection``) the public
    ``sae_manifold_fit`` facade folds into a ``ManifoldSAE``."""
    rust = gamfit._sae_manifold.rust_module()
    payload = rust.sae_manifold_fit_minimal(
        np.ascontiguousarray(z),
        ["periodic"] * K,
        [1] * K,
        1.0,  # alpha
        0.5,  # tau
        False,  # learnable alpha
        "softmax",
        sparsity_strength=1.0,
        smoothness=1.0,
        max_iter=8,
        learning_rate=0.04,
        gumbel_schedule=None,
        analytic_penalties=None,
        random_state=0,
        top_k=top_k,
        initial_logits=None,
        initial_coords=None,
        threshold_gate_threshold=0.0,
        fisher_factors=None,
        fisher_mass_residual=None,
        fisher_provenance=None,
        row_loss_weights=None,
    )
    return dict(payload)


def test_top_k_payload_score_describes_the_projected_model() -> None:
    """#1232 — the payload must describe ONE coherent model: the top-level
    ``penalized_loss_score`` (its ``data_fit``) must be measured on the SAME
    ``fitted`` / ``assignments`` the payload returns at top level, not on the
    pre-projection smooth model. Pre-fix the score's ``data_fit`` was the
    unprojected reconstruction's, so it disagreed with the projected ``fitted``.
    """
    rng = np.random.default_rng(0)
    z = rng.standard_normal((24, 4))
    payload = _raw_topk_payload(z, K=4, top_k=1)

    # Projection genuinely fired (top_k=1 < K=4), so the split descriptor is set.
    assert payload.get("top_k_projection_applied") is True
    desc = payload["top_k_projection"]
    assert desc["applied"] is True
    assert desc["top_k"] == 1
    assert desc["top_level_model"] == "post_topk"
    assert "pre_topk" in payload

    fitted = np.asarray(payload["fitted"], dtype=float)
    reported_data_fit = float(payload["penalized_loss_breakdown"]["data_fit"])
    recomputed = _euclidean_data_fit(z, fitted)
    assert abs(reported_data_fit - recomputed) < 1e-8, (
        "top-level penalized_loss data_fit must match the residual of the "
        f"top-level (projected) fitted; reported={reported_data_fit}, "
        f"from fitted={recomputed}"
    )


def test_top_k_pre_topk_block_describes_the_unprojected_model() -> None:
    """#1232 — the ``pre_topk`` sub-dict is the honest unprojected layer: its
    ``data_fit`` must match its OWN ``fitted`` (the smooth reconstruction), and
    when top-k genuinely drops mass the two layers differ (a hard projection can
    only worsen the reconstruction), so the split is not a no-op relabel."""
    rng = np.random.default_rng(1)
    z = rng.standard_normal((24, 4))
    payload = _raw_topk_payload(z, K=4, top_k=1)

    pre = payload["pre_topk"]
    pre_fitted = np.asarray(pre["fitted"], dtype=float)
    pre_data_fit = float(pre["penalized_loss_breakdown"]["data_fit"])
    assert abs(pre_data_fit - _euclidean_data_fit(z, pre_fitted)) < 1e-8, (
        "pre_topk data_fit must match the pre_topk (unprojected) fitted"
    )

    post_data_fit = float(payload["penalized_loss_breakdown"]["data_fit"])
    # Hard top-k restricts the reconstruction, so its data-fit cannot be smaller
    # than the smooth model's; and with K=4 -> top_k=1 it strictly worsens here.
    assert post_data_fit >= pre_data_fit - 1e-9
    assert post_data_fit > pre_data_fit, (
        "top_k=1 from K=4 should change the reconstruction (and hence the "
        "data-fit); pre and post being identical would mean the projection was "
        "silently a no-op"
    )


def test_no_top_k_projection_keeps_single_unsplit_payload() -> None:
    """Without a projecting top-k (``top_k is None`` or ``top_k == K``) there is
    no second model, so the split keys are absent and the single top-level score
    still matches the single ``fitted`` — the payload is trivially consistent."""
    rng = np.random.default_rng(2)
    z = rng.standard_normal((24, 4))
    payload = _raw_topk_payload(z, K=3, top_k=3)  # top_k == K -> no projection

    assert "top_k_projection_applied" not in payload
    assert "top_k_projection" not in payload
    assert "pre_topk" not in payload

    fitted = np.asarray(payload["fitted"], dtype=float)
    reported_data_fit = float(payload["penalized_loss_breakdown"]["data_fit"])
    assert abs(reported_data_fit - _euclidean_data_fit(z, fitted)) < 1e-8
