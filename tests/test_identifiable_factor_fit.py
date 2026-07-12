"""Smoke test for :func:`gamfit.identifiable_factor_fit`.

Generates a tiny synthetic dataset with a 3-dim auxiliary-conditioned
latent and a 3-dim free latent, mixes them linearly into a 12-dim
observation space, then asserts the recipe returns the right shapes and a
finite evidence score.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")
torch = pytest.importorskip("torch")


def _best_permutation_min_abscorr(a: np.ndarray, b: np.ndarray) -> float:
    """Largest achievable *min* |corr| over column permutations of ``a`` vs ``b``.

    Khemakhem 2107.10098 Thm. 1 identifies the supervised latent only up to a
    component-wise invertible transform (here: permutation + signed scaling),
    so genuine recovery means *some* permutation pairs every true axis with a
    learned axis at high absolute correlation. Returns the best (over
    permutations) of the worst-paired |corr| — a permutation/sign/scale
    -invariant recovery score in [0, 1].
    """

    k = a.shape[1]
    corr = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            c = abs(np.corrcoef(a[:, i], b[:, j])[0, 1])
            # A degenerate (constant-variance) column makes corrcoef NaN; that
            # is a recovery of zero, not an undefined comparison.
            corr[i, j] = c if math.isfinite(c) else 0.0
    best = 0.0
    for perm in itertools.permutations(range(k)):
        best = max(best, min(corr[i, perm[i]] for i in range(k)))
    return float(best)


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


def _issue_790_dataset(seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, d, k = 240, 32, 4
    t = rng.normal(size=(n, k))
    w = rng.normal(size=(d, k))
    x = t @ w.T + 0.01 * rng.normal(size=(n, d))
    aux = t[:, :2] + 0.02 * rng.normal(size=(n, 2))
    return x, aux


def _mean_best_abscorr(t: np.ndarray, aux: np.ndarray) -> float:
    tc = t - t.mean(axis=0, keepdims=True)
    ac = aux - aux.mean(axis=0, keepdims=True)
    t_std = tc.std(axis=0, keepdims=True) + 1.0e-12
    a_std = ac.std(axis=0, keepdims=True) + 1.0e-12
    corr = np.abs((tc / t_std).T @ (ac / a_std) / t.shape[0])
    return float(corr.max(axis=1).mean())


def test_identifiable_factor_fit_does_not_mutate_global_torch_rng() -> None:
    x, aux = _toy_dataset(seed=17)
    torch.manual_seed(54_321)
    state_before = torch.random.get_rng_state().clone()
    gamfit.identifiable_factor_fit(
        x,
        aux=aux,
        n_supervised=3,
        n_free=1,
        encoder="linear",
        max_iter=1,
        random_state=8,
        check_identifiability=False,
    )
    state_after = torch.random.get_rng_state()
    assert torch.equal(state_after, state_before), (
        "identifiable_factor_fit must isolate torch module initialization from "
        "the caller's global RNG stream"
    )


def test_identifiable_factor_fit_default_auto_weights_issue_790() -> None:
    x, aux = _issue_790_dataset()
    result = gamfit.identifiable_factor_fit(
        x,
        aux=aux,
        n_supervised=2,
        n_free=2,
        max_iter=400,
        random_state=0,
        check_identifiability=False,
    )
    corr_sup = _mean_best_abscorr(result.T_supervised, aux)
    assert result.mech_sparsity_weight == pytest.approx(1.0e-4)
    assert result.aux_prior_weight == pytest.approx(2.0)
    assert corr_sup > 0.9


def test_identifiable_factor_fit_smoke() -> None:
    x, aux = _toy_dataset()
    with pytest.warns(UserWarning, match="MechanismSparsity"):
        result = gamfit.identifiable_factor_fit(
            x,
            aux=aux,
            n_supervised=3,
            n_free=3,
            encoder="mlp[32, 32]",
            max_iter=800,
            learning_rate=5e-3,
            random_state=1,
        )
    assert result.T_supervised.shape == (80, 3)
    assert result.T_free.shape == (80, 3)
    assert math.isfinite(result.evidence)
    assert result.decoder.shape == (12, 6)
    assert result.aux_prior_weight > 0.0
    assert result.mech_sparsity_weight > 0.0
    # The supervised iVAE precondition should hold for this configuration.
    # The mechanism-sparsity theorem is reported separately below because
    # the smoothed-L1 decoder penalty may not produce exact zeros on this
    # tiny smoke fixture.
    assert any("MechanismSparsity" in w for w in result.report.as_warnings())
    # The fit completing at all is the regression guard for #576: the
    # supervised iVAE prior previously raised because its conditional scale
    # σ(u) was hardcoded to ones, collapsing the Khemakhem natural-parameter
    # signature [μ(u) ‖ log σ(u)] to rank k < 2k. The derived varying scale
    # genuinely satisfies the 2k rank condition, so the iVAE theorem is the
    # *passing* reason this fit succeeds.
    by_name = {t.theorem_name: t for t in result.report.theorems}
    assert by_name["iVAE"].status == "pass"

    # Real recovery: the supervised latent must recover the auxiliary up to
    # the permutation + signed scaling that Khemakhem Thm. 1 allows, AND must
    # do so far better than the free block (which is unsupervised). t_sup was
    # generated as aux + small noise, so a converged supervised block aligns
    # with aux per-axis.
    sup_recovery = _best_permutation_min_abscorr(aux, result.T_supervised)
    free_recovery = _best_permutation_min_abscorr(aux, result.T_free)
    assert sup_recovery > 0.85, (
        f"supervised block failed to recover the auxiliary: "
        f"min paired |corr| = {sup_recovery:.3f}"
    )
    assert sup_recovery > free_recovery + 0.2, (
        f"auxiliary information leaked into the free block: "
        f"sup={sup_recovery:.3f} free={free_recovery:.3f}"
    )


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
    assert any("auxiliary" in w for w in result.report.as_warnings())


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
