"""K>1 forward-consistency for :class:`ManifoldSAE` after ``.fit()``.

Companion to ``test_manifold_sae_fit_consistency_issue_242.py``, which only
exercises the ``K=1`` path. After ``sae.fit(x)`` the module routes every
subsequent ``sae(x)`` through ``_forward_from_closed_form``, which rebuilds
``x_hat`` / ``assignments`` / ``positions`` straight from the closed-form fit
object (``fit.fitted``, ``fit.coords``, ``fit.assignments``), bypassing the
random encoder / anchors / ``log_lambda`` that ``_copy_fit_into_params`` never
touches. The contract is therefore that for ``K>1`` the module forward must
reproduce the closed-form fit bit-for-bit and must not splice a closed-form
``reml_score`` onto a random-encoder reconstruction.

These tests pin that contract so any future refactor of ``fit`` or
``_forward_from_closed_form`` that breaks the multi-atom reshape/stack of
``fit.coords`` / ``fit.assignments`` is caught.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np

pytest: Any = import_module("pytest")
torch = pytest.importorskip("torch")
gt = pytest.importorskip("gamfit.torch")


def _make_synth(n: int = 48, d: int = 6, k: int = 3, seed: int = 0) -> np.ndarray:
    """Mixture-of-cosines synthetic with ``k`` latent atoms in ``R^d``.

    Identical generator to ``test_manifold_sae_parity._make_synth`` so the data
    distribution (and therefore the determinism) is shared across the suite.
    """
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=(n, k))
    bases = rng.standard_normal(size=(k, d))
    bases /= np.linalg.norm(bases, axis=1, keepdims=True) + 1e-9
    amps = rng.uniform(0.5, 1.5, size=(n, k))
    x = (amps * np.cos(angles))[:, :, None] * bases[None, :, :]
    return x.sum(axis=1) + 0.05 * rng.standard_normal((n, d))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _fit_k3_module(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = _make_synth(seed=seed)
    cfg = gt.ManifoldSAEConfig(
        input_dim=X.shape[1],
        n_atoms=3,
        intrinsic_rank=1,
        atom_manifold="circle",
        atom_basis="fourier",
        basis_order=2,
        n_basis_per_atom=4,
        sparsity={"kind": "ibp_gumbel", "init_alpha": 1.0, "tau_schedule": "linear:4.0->1.0"},
        decoder={"ortho_weight": 0.0, "monotonicity_weight": 0.0},
        reml={"enabled": True, "select": ["lambda"]},
    )
    sae = gt.ManifoldSAE(cfg).double()
    x = torch.as_tensor(X, dtype=torch.float64)
    fit = sae.fit(x, max_iter=10, random_state=42)
    return sae, x, fit


def test_module_forward_x_hat_matches_closed_form_fit_k3() -> None:
    sae, x, fit = _fit_k3_module(seed=0)
    out = sae(x)
    fitted = torch.as_tensor(np.asarray(fit.fitted, dtype=np.float64), dtype=torch.float64)
    assert out.x_hat.shape == fitted.shape
    assert out.assignments.shape == (x.shape[0], 3)
    assert torch.allclose(out.x_hat, fitted, atol=1e-6, rtol=1e-6), (
        f"After K=3 sae.fit(x), sae(x).x_hat must equal fit.fitted, but "
        f"||diff||={float((out.x_hat - fitted).norm()):.4e}, "
        f"||fit.fitted||={float(fitted.norm()):.4e}"
    )


def test_module_forward_r2_matches_closed_form_r2_k3() -> None:
    sae, x, fit = _fit_k3_module(seed=1)
    out = sae(x)
    r2_module = _r2(x.detach().numpy(), out.x_hat.detach().numpy())
    r2_fit = float(getattr(fit, "reconstruction_r2", float("nan")))
    assert abs(r2_module - r2_fit) <= 5e-3, (
        f"K=3 module forward R²={r2_module:.4f} must match closed-form "
        f"fit.reconstruction_r2={r2_fit:.4f} after .fit()."
    )


def test_module_does_not_attach_alien_reml_score_when_x_hat_is_random_k3() -> None:
    """With ``K=3`` the same x_hat/reml_score coherence contract must hold: if
    x_hat is the random-encoder reconstruction, reml_score must not be silently
    borrowed from the closed-form fit. Because ``_forward_from_closed_form``
    reproduces fit.fitted exactly, the left disjunct holds.
    """
    sae, x, fit = _fit_k3_module(seed=2)
    out = sae(x)
    fitted = torch.as_tensor(np.asarray(fit.fitted, dtype=np.float64), dtype=torch.float64)
    x_hat_matches_fit = torch.allclose(out.x_hat, fitted, atol=1e-6, rtol=1e-6)
    reml_borrowed = bool(torch.allclose(
        out.reml_score,
        torch.tensor(float(fit.reml_score), dtype=out.reml_score.dtype),
    ))
    assert x_hat_matches_fit or not reml_borrowed, (
        "K=3 module forward is inconsistent: x_hat is NOT the closed-form fit "
        "but reml_score IS borrowed from the closed-form fit."
    )
