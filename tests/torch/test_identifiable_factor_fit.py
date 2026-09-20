"""Tests for :func:`gamfit.identifiability.identifiable_factor_fit`.

A result is only ever returned from a certified stationary point of the
penalized objective (#3997): these tests check the certificate, the unit
second-moment gauge of ``T_free`` that gives the objective a minimizer, the
typed failure of an uncertified run, and that a degenerate auxiliary is
refused before any optimization instead of silently dropping the iVAE prior.
"""
from __future__ import annotations

import itertools
import math
import warnings

import numpy as np
import pytest
import torch

import gamfit
from gamfit.errors import FitConvergenceError


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
    # One evaluation cannot certify stationarity, so the fit raises; the
    # caller's RNG stream must be untouched either way.
    with pytest.raises(FitConvergenceError):
        gamfit.identifiability.identifiable_factor_fit(
            x,
            aux=aux,
            n_supervised=3,
            n_free=1,
            encoder="linear",
            max_evals=1,
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
    result = gamfit.identifiability.identifiable_factor_fit(
        x,
        aux=aux,
        n_supervised=2,
        n_free=2,
        random_state=0,
        check_identifiability=False,
    )
    corr_sup = _mean_best_abscorr(result.T_supervised, aux)
    assert result.mech_sparsity_weight == pytest.approx(1.0e-4)
    assert result.aux_prior_weight == pytest.approx(2.0)
    assert result.stationarity <= 1.0e-8
    assert corr_sup > 0.9


def test_identifiable_factor_fit_smoke() -> None:
    x, aux = _toy_dataset()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = gamfit.identifiability.identifiable_factor_fit(
            x,
            aux=aux,
            n_supervised=3,
            n_free=3,
            encoder="mlp[32, 32]",
            random_state=1,
        )
    assert result.T_supervised.shape == (80, 3)
    assert result.T_free.shape == (80, 3)
    assert math.isfinite(result.profile_log_likelihood)
    assert result.decoder.shape == (12, 6)
    assert result.aux_prior_weight > 0.0
    assert result.mech_sparsity_weight > 0.0
    # Every emitted warning is a report line, and every report line is
    # emitted: the report is the single source of the precondition verdicts.
    emitted = [str(w.message) for w in record if issubclass(w.category, UserWarning)]
    assert emitted == result.report.as_warnings()
    # #576: the derived varying σ(u) satisfies the Khemakhem 2k rank
    # condition, so the iVAE theorem passes on this fixture.
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


def _linear_fit(max_evals: int = 5000, **kw):
    x, aux = _toy_dataset(seed=6)
    return gamfit.identifiability.identifiable_factor_fit(
        x,
        aux=aux,
        n_supervised=3,
        n_free=2,
        encoder="linear",
        max_evals=max_evals,
        random_state=2,
        check_identifiability=False,
        **kw,
    )


def test_identifiable_factor_fit_certifies_stationarity() -> None:
    result = _linear_fit()
    assert 0.0 <= result.stationarity <= 1.0e-8
    assert result.n_iter >= 1


def test_identifiable_factor_fit_free_block_has_unit_second_moment() -> None:
    # The objective is invariant to the free block's scale only through this
    # normalization; without it the sparsity penalty has no minimizer (#3997).
    result = _linear_fit()
    np.testing.assert_allclose(
        np.mean(result.T_free**2, axis=0), np.ones(2), rtol=1.0e-12, atol=0.0
    )
    assert result.free_scale.shape == (2,)
    assert np.all(result.free_scale > 0.0)


def test_identifiable_factor_fit_budget_does_not_move_certified_point() -> None:
    # The budget is not a stopping rule: once the certificate holds the fit
    # stops, so doubling the budget returns the identical point.
    a = _linear_fit()
    b = _linear_fit(max_evals=10000)
    assert a.n_iter == b.n_iter
    np.testing.assert_array_equal(a.T_supervised, b.T_supervised)
    np.testing.assert_array_equal(a.T_free, b.T_free)
    np.testing.assert_array_equal(a.decoder, b.decoder)


def test_identifiable_factor_fit_uncertified_run_raises_with_checkpoint() -> None:
    with pytest.raises(FitConvergenceError, match="stationary point") as info:
        _linear_fit(max_evals=3)
    exc = info.value
    assert exc.grad_inf > exc.grad_tol * exc.grad_inf_init
    assert exc.max_evals == 3
    assert math.isfinite(exc.objective_value)
    assert exc.checkpoint_decoder.shape == (12, 5)
    assert set(exc.checkpoint_encoder_state) == {"0.weight", "0.bias"}


def test_identifiable_factor_fit_refuses_constant_aux() -> None:
    # A constant auxiliary cannot identify the iVAE conditional prior
    # (Khemakhem Thm. 1). The fit must refuse it before optimizing rather
    # than drop the prior and return a model the recipe does not describe.
    x, _ = _toy_dataset(seed=2)
    aux = np.ones((x.shape[0], 1))
    with pytest.raises(ValueError, match="Khemakhem"):
        gamfit.identifiability.identifiable_factor_fit(
            x,
            aux=aux,
            n_supervised=1,
            n_free=2,
            mech_sparsity_weight=1.0,
            aux_prior_weight=1.0,
            encoder="mlp[16, 16]",
            random_state=3,
        )


def test_identifiability_check_flags_constant_aux() -> None:
    """``gamfit.identifiability.check`` flags a constant aux as iVAE fail.

    Checks a certified fit against a constant auxiliary, which violates one
    and only one theorem precondition (iVAE), and verifies the structured
    report.
    """

    x, aux = _toy_dataset(seed=4)
    result = gamfit.identifiability.identifiable_factor_fit(
        x,
        aux=aux[:, :1],
        n_supervised=1,
        n_free=2,
        encoder="linear",
        random_state=5,
        check_identifiability=False,
    )
    report = gamfit.identifiability.check(result, aux=np.ones((x.shape[0], 1)))
    by_name = {t.theorem_name: t for t in report.theorems}
    assert by_name["iVAE"].status == "fail"
    assert "constant" in by_name["iVAE"].reason.lower()
    assert by_name["iVAE"].metric["aux_min_std"] == 0.0
    assert report.status == "fail"
    assert {t.theorem_name for t in report.theorems} == {
        "iVAE", "MechanismSparsity", "RandomProjection",
    }


def test_identifiable_factor_fit_rejects_unknown_encoder() -> None:
    x, aux = _toy_dataset()
    with pytest.raises(ValueError, match="not a recognized encoder"):
        gamfit.identifiability.identifiable_factor_fit(
            x, aux=aux, n_supervised=3, n_free=3,
            mech_sparsity_weight=1.0, aux_prior_weight=1.0,
            encoder="transformer[8]",
        )
