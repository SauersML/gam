"""Regression for issue #2090: ``sae_manifold_fit`` must accept
``smoothness_weight=0.0``.

Every other penalty weight in the public facade (``sparsity_weight``,
``isometry_weight``, ``decoder_incoherence_weight``, ``nuclear_norm_weight``)
uses ``0.0`` to disable its term. ``smoothness_weight`` was the odd one out: the
Rust FFI rejected zero with ``GamError: smoothness must be finite and positive;
got 0``, breaking configuration sweeps where zero means "turn this penalty off".

The fix floors a zero ``smoothness`` to the same tiny positive log-domain
sentinel already used for ``sparsity_strength == 0.0`` (both only seed the outer
REML cascade's log-rho), and relaxes the validator to "finite and non-negative".
This test asserts the documented zero-disables-the-term contract: the fit
returns a model and does not raise.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamfit


def test_smoothness_weight_zero_is_accepted() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 4))

    model = gamfit.sae_manifold_fit(
        X=X,
        K=1,
        d_atom=1,
        atom_topology="circle",
        assignment="softmax",
        n_iter=1,
        sparsity_weight=0.0,
        smoothness_weight=0.0,
        isometry_weight=0.0,
        ard_per_atom=False,
        decoder_incoherence_weight=0.0,
        nuclear_norm_weight=0.0,
        random_state=0,
    )
    assert model is not None
    # A negative value is still rejected — only zero-disables is relaxed.
    with pytest.raises((ValueError, Exception)):
        gamfit.sae_manifold_fit(
            X=X,
            K=1,
            d_atom=1,
            atom_topology="circle",
            assignment="softmax",
            n_iter=1,
            sparsity_weight=0.0,
            smoothness_weight=-1.0,
            isometry_weight=0.0,
            ard_per_atom=False,
            decoder_incoherence_weight=0.0,
            nuclear_norm_weight=0.0,
            random_state=0,
        )
