"""Regression for issue #2088: ``sae_manifold_fit`` rejected the documented
heterogeneous ``d_atom`` / ``atom_basis`` path with a confusing
``RemlConvergenceError`` raised deep in the REML cascade.

The SAE row-block analytic penalties (isometry, native ARD, SCAD/MCP coord
sparsity, block-orthogonality) target the unified "t" latent block whose width
is a single ``d_max`` shared by every atom, so heterogeneous per-atom coord dims
cannot be dispatched. Because ``isometry_weight`` and ``ard_per_atom`` default
ON, EVERY heterogeneous call hit that refusal.

The fix validates the incompatibility up front and raises a direct ``ValueError``
naming the conflict and both resolutions (issue #2088 option 2), and — with those
row-block penalties disabled — lets the heterogeneous path run (option 1). Under
#2098 (SPEC-8) this validation lives in the Rust engine
(``SaeManifoldTerm::validate_heterogeneous_atom_compatibility``); the Python
facade is a thin wrapper that surfaces the engine error.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamfit


def test_heterogeneous_d_atom_with_default_row_block_penalties_raises_clear_error() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 4))

    with pytest.raises(ValueError) as excinfo:
        gamfit.sae_manifold_fit(
            X=X,
            K=2,
            d_atom=[2, 1],
            atom_basis=["euclidean", "periodic"],
            assignment="ordered_beta_bernoulli",
            n_iter=1,
            sparsity_weight=0.01,
            coord_sparsity="l1",
            smoothness_weight=0.01,
            isometry_weight=0.1,
            ard_per_atom=False,
            decoder_incoherence_weight=0.1,
            nuclear_norm_weight=0.0,
            random_state=0,
            alpha="auto",
        )
    msg = str(excinfo.value)
    # A direct, actionable engine error (moved into gam-sae under #2098/SPEC-8),
    # surfaced as a ValueError — not a deep RemlConvergenceError.
    assert "heterogeneous atom coordinate dims" in msg
    assert "isometry" in msg
    assert "uniform atom_dim" in msg
    assert "RemlConvergence" not in type(excinfo.value).__name__


def test_heterogeneous_d_atom_passes_validation_when_row_block_penalties_disabled() -> None:
    # With every row-block "t"-block penalty disabled, the facade validation must
    # NOT raise the heterogeneous-refusal error — the documented mixed-d_atom
    # path is admitted (issue #2088 option 1). We do not assert the deeper fit
    # succeeds bit-for-bit here (that exercises unrelated solver machinery); we
    # assert only that our up-front validation lets this configuration through.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(24, 4))

    try:
        model = gamfit.sae_manifold_fit(
            X=X,
            K=2,
            d_atom=[2, 1],
            atom_basis=["euclidean", "periodic"],
            assignment="ordered_beta_bernoulli",
            n_iter=1,
            sparsity_weight=0.01,
            coord_sparsity="l1",
            smoothness_weight=0.01,
            isometry_weight=0.0,
            ard_per_atom=False,
            decoder_incoherence_weight=0.0,
            nuclear_norm_weight=0.0,
            block_orthogonality_weight=0.0,
            random_state=0,
            alpha="auto",
        )
    except ValueError as exc:
        assert "heterogeneous atom coordinate dims" not in str(exc), (
            "penalties-off heterogeneous path must not hit the row-block "
            f"refusal validation: {exc}"
        )
    else:
        assert model is not None
