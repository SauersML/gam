"""Regression for issue #2088: ``sae_manifold_fit`` rejected the documented
heterogeneous ``d_atom`` / ``atom_basis`` path with a confusing
``RemlConvergenceError`` raised deep in the REML cascade.

The fix validates the incompatibility up front and raises a direct ``ValueError``
naming the conflict and both resolutions (issue #2088 option 2), and — with the
offending penalty disabled — lets the heterogeneous path run (option 1). Under
#2098 (SPEC-8) this validation lives in the Rust engine
(``SaeManifoldTerm::validate_heterogeneous_atom_compatibility``); the Python
facade is a thin wrapper that surfaces the engine error.

Post-F6 the refusal is NARROWER than the original issue described, and that
narrowing is itself part of the contract. Only a row-block penalty that fails
``sae_row_block_penalty_composes_over_heterogeneous_coord_dims`` is refused:
block-orthogonality (reshapes to ``(n_eff × d)`` and groups a fixed axis set),
TopK / ThresholdGate (per-axis thresholds), and the row-precision priors (an
``(n_eff × d × d)`` stack). The dim-adaptive ones — isometry, SCAD/MCP coord
sparsity, plain sparsity, and native ARD — are per-atom-additive, read each
atom own ``d_k``, and are ADMITTED on a mixed dictionary
(``crates/gam-sae/src/manifold/construction.rs:3044-3050``).
"""

from __future__ import annotations

import numpy as np
import pytest

import gamfit


def test_heterogeneous_d_atom_with_fixed_d_row_block_penalty_raises_clear_error() -> None:
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
            isometry_weight=0.0,
            block_orthogonality_weight=0.1,
            decoder_incoherence_weight=0.1,
            nuclear_norm_weight=0.0,
            random_state=0,
        )
    msg = str(excinfo.value)
    # A direct, actionable engine error (moved into gam-sae under #2098/SPEC-8),
    # surfaced as a ValueError — not a deep RemlConvergenceError. The engine
    # names the offending penalty by its registry name, the conflict, and the
    # resolution; all three are asserted so a message that keeps the words but
    # loses the identity of the offender still fails.
    assert "heterogeneous atom coordinate dims" in msg
    assert "block_orthogonality" in msg
    assert "uniform atom_dim" in msg
    assert "RemlConvergence" not in type(excinfo.value).__name__


def test_isometry_gauge_is_admitted_on_heterogeneous_atom_dims() -> None:
    # The post-F6 narrowing, pinned from the Python surface: the isometry gauge
    # is per-atom-additive (`corrected_isometry_penalty` is rebuilt per atom), so
    # it composes over mixed `d_k` and must NOT trigger the refusal. This is the
    # exact configuration the original #2088 repro used to drive the refusal, so
    # if the blanket pre-F6 behaviour ever comes back this test catches it.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 4))

    try:
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
            block_orthogonality_weight=0.0,
            decoder_incoherence_weight=0.1,
            nuclear_norm_weight=0.0,
            random_state=0,
        )
    except Exception as exc:  # noqa: BLE001 - the claim is about ONE message
        assert "heterogeneous atom coordinate dims" not in str(exc), (
            "the isometry gauge composes per atom over mixed coord dims and must "
            f"not be refused up front: {exc}"
        )


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
            decoder_incoherence_weight=0.0,
            nuclear_norm_weight=0.0,
            block_orthogonality_weight=0.0,
            random_state=0,
        )
    except ValueError as exc:
        assert "heterogeneous atom coordinate dims" not in str(exc), (
            "penalties-off heterogeneous path must not hit the row-block "
            f"refusal validation: {exc}"
        )
    else:
        assert model is not None
