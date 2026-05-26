"""RED tests for issue #242: ManifoldSAE.fit() leaves the torch module inconsistent.

After `sae.fit(x)`:
  - `sae(x).x_hat` must match `fit.fitted` (close-form reconstruction).
  - Module forward R² must match `fit.reconstruction_r2`.

Today both fail: `_copy_fit_into_params` only writes `decoder_blocks`, leaving
encoder/anchors/log_lambda/sparsity at random init, so `forward()` mixes a
random-encoder `x_hat` with the closed-form `reml_score`.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gt = pytest.importorskip("gamfit.torch")


def _make_fit_and_module(seed: int = 0):
    rng = np.random.default_rng(seed)
    N, D = 64, 5
    theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    factor = np.stack(
        [np.cos(theta), np.sin(theta), np.cos(2 * theta), np.sin(2 * theta), theta / (2 * np.pi)],
        axis=1,
    )
    X = factor + 0.01 * rng.standard_normal((N, D))

    cfg = gt.ManifoldSAEConfig(
        input_dim=D,
        n_atoms=1,
        intrinsic_rank=1,
        atom_manifold="circle",
        atom_basis="fourier",
        n_basis_per_atom=5,
    )
    sae = gt.ManifoldSAE(cfg).double()
    x = torch.as_tensor(X, dtype=torch.float64)
    fit = sae.fit(x, max_iter=20, random_state=seed)
    return sae, x, fit


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def test_module_forward_x_hat_matches_closed_form_fit() -> None:
    sae, x, fit = _make_fit_and_module(seed=0)
    out = sae(x)
    fitted = torch.as_tensor(np.asarray(fit.fitted, dtype=np.float64), dtype=torch.float64)
    assert torch.allclose(out.x_hat, fitted, atol=1e-6, rtol=1e-6), (
        f"After sae.fit(x), sae(x).x_hat must equal fit.fitted, but "
        f"||diff||={float((out.x_hat - fitted).norm()):.4e}, "
        f"||fit.fitted||={float(fitted.norm()):.4e}"
    )


def test_module_forward_r2_matches_closed_form_r2() -> None:
    sae, x, fit = _make_fit_and_module(seed=1)
    out = sae(x)
    r2_module = _r2(x.detach().numpy(), out.x_hat.detach().numpy())
    r2_fit = float(getattr(fit, "reconstruction_r2", float("nan")))
    assert abs(r2_module - r2_fit) <= 5e-3, (
        f"Module forward R²={r2_module:.4f} must match closed-form "
        f"fit.reconstruction_r2={r2_fit:.4f} after .fit(); difference exposes the "
        f"random-encoder/closed-form-decoder inconsistency."
    )


def test_module_does_not_attach_alien_reml_score_when_x_hat_is_random() -> None:
    """If x_hat is the random-encoder reconstruction, reml_score must not be
    silently borrowed from the closed-form fit. Either x_hat is consistent with
    fit (preferred) OR reml_score reflects the actual module state.

    This test catches the symptom where the two are mixed: x_hat is random but
    reml_score is the closed-form value, giving a misleading scalar.
    """
    sae, x, fit = _make_fit_and_module(seed=2)
    out = sae(x)
    fitted = torch.as_tensor(np.asarray(fit.fitted, dtype=np.float64), dtype=torch.float64)
    x_hat_matches_fit = torch.allclose(out.x_hat, fitted, atol=1e-6, rtol=1e-6)
    reml_borrowed = bool(torch.allclose(
        out.reml_score,
        torch.tensor(float(fit.reml_score), dtype=out.reml_score.dtype),
    ))
    assert x_hat_matches_fit or not reml_borrowed, (
        "Module forward is inconsistent: x_hat is NOT the closed-form fit but "
        "reml_score IS borrowed from the closed-form fit. Either copy the full "
        "fit state into module params or stop borrowing the score."
    )
