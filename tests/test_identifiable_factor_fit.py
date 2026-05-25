"""Smoke test for :func:`gamfit.identifiable_factor_fit`.

Generates a tiny synthetic dataset with a 3-dim auxiliary-conditioned
latent and a 3-dim free latent, mixes them linearly into a 12-dim
observation space, then asserts the recipe returns the right shapes and a
finite evidence score.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")
pytest.importorskip("torch")


def _toy_dataset(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, p = 80, 12
    n_sup, n_free = 3, 3
    aux = rng.normal(size=(n, n_sup))
    t_sup = aux + 0.1 * rng.normal(size=aux.shape)
    t_free = rng.normal(size=(n, n_free))
    latents = np.concatenate([t_sup, t_free], axis=1)
    mixing = rng.normal(size=(latents.shape[1], p))
    x = latents @ mixing + 0.05 * rng.normal(size=(n, p))
    return x, aux


def test_identifiable_factor_fit_smoke() -> None:
    x, aux = _toy_dataset()
    result = gamfit.identifiable_factor_fit(
        x,
        aux=aux,
        n_supervised=3,
        n_free=3,
        mech_sparsity_weight=1.0,
        aux_prior_weight=1.0,
        encoder="mlp[32, 32]",
        max_iter=20,
        learning_rate=5e-3,
        random_state=1,
    )
    assert result.T_supervised.shape == (80, 3)
    assert result.T_free.shape == (80, 3)
    assert math.isfinite(result.evidence)
    assert result.decoder.shape == (12, 6)
    assert result.aux_prior_weight > 0.0
    assert result.mech_sparsity_weight > 0.0
    # All preconditions of the iVAE + mech-sparsity theorem should hold
    # for this configuration (aux varies, decoder is full-rank generically,
    # sparsity weight is positive, encoder has 3 Linear layers).
    assert result.warnings == []


def test_identifiable_factor_fit_warns_on_constant_aux() -> None:
    x, _ = _toy_dataset(seed=2)
    aux = np.ones((x.shape[0], 1))  # constant aux -> iVAE precondition fails
    with pytest.warns(UserWarning, match="auxiliary covariate variation"):
        result = gamfit.identifiable_factor_fit(
            x,
            aux=aux,
            n_supervised=1,
            n_free=2,
            mech_sparsity_weight=1.0,
            aux_prior_weight=1.0,
            encoder="mlp[16, 16]",
            max_iter=5,
            learning_rate=1e-2,
            random_state=3,
        )
    assert any("auxiliary" in w for w in result.warnings)


def test_identifiability_check_flags_constant_aux() -> None:
    """``gamfit.identifiability.check`` flags a constant aux as iVAE fail.

    Builds the smallest possible fit that violates one and only one
    theorem precondition (constant aux -> iVAE fails; decoder + encoder
    are otherwise healthy) and verifies the structured report.
    """

    x, _ = _toy_dataset(seed=4)
    aux = np.ones((x.shape[0], 1))
    with pytest.warns(UserWarning):
        result = gamfit.identifiable_factor_fit(
            x,
            aux=aux,
            n_supervised=1,
            n_free=2,
            mech_sparsity_weight=1.0,
            aux_prior_weight=1.0,
            encoder="mlp[16, 16]",
            max_iter=5,
            learning_rate=1e-2,
            random_state=5,
        )
    assert result.report is not None
    by_name = {t.theorem_name: t for t in result.report.theorems}
    assert by_name["iVAE"].status == "fail"
    assert "constant" in by_name["iVAE"].reason.lower()
    assert by_name["iVAE"].metric["aux_min_std"] == 0.0
    # Re-running check() against the saved fit reproduces the same verdict.
    rerun = gamfit.identifiability_check(result)
    assert rerun.status == "fail"
    assert {t.theorem_name for t in rerun.theorems} == {
        "iVAE", "MechanismSparsity", "RandomProjection",
    }


def test_identifiable_factor_fit_rejects_unknown_encoder() -> None:
    x, aux = _toy_dataset()
    with pytest.raises(ValueError, match="not a recognized encoder"):
        gamfit.identifiable_factor_fit(
            x, aux=aux, n_supervised=3, n_free=3,
            mech_sparsity_weight=1.0, aux_prior_weight=1.0,
            encoder="transformer[8]",
        )
