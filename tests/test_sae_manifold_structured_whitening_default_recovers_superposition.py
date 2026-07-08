"""#2021 — structured-residual whitening is engaged BY DEFAULT and improves
held-out recovery under a genuinely correlated (superposition) residual.

Repo doctrine is "magic by default": the best-known behavior must be the
default, not an opt-in knob. #2021's complaint was exactly that the
superposition-aware likelihood is "built but never engaged — production is
always iid" because the single-shot ``sae_manifold_fit`` defaulted
``structured_residual_passes=0``.

Two gates, split so the OBJECTIVE-quality claim does not depend on the exact
default value:

* ``test_structured_whitening_improves_held_out_recovery`` — the objective
  justification for defaulting the metric ON. On a planted circle whose
  residual carries a strong shared rank-1 factor (an off-diagonal residual
  covariance the iid metric mismodels), a structured-whitened fit recovers the
  NOISELESS signal on HELD-OUT rows strictly better than the iid fit. Averaged
  over independent planted draws so the strict inequality is a real effect, not
  one lucky seed. If whitening were a no-op the two would tie and this fails.

* ``test_structured_whitening_is_default_on`` — the flip itself. The production
  default must engage the structured metric: a default fit must differ from an
  explicit ``structured_residual_passes=0`` (iid) fit. Before the default flip
  (default was 0 == iid) these were bit-identical and this FAILED; after the
  flip the default engages whitening and they differ.
"""
from __future__ import annotations

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")

from gamfit._sae_manifold import ManifoldSAE, sae_manifold_fit  # noqa: E402


def _planted_superposition(
    n: int,
    seed: int,
    *,
    p: int = 6,
    factor_scale: float = 0.6,
    iid_scale: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X, S)``: observed rows ``X`` and the NOISELESS planted signal
    ``S`` (both ``(n, p)``).

    The signal is a circle lifted into ``p`` channels. The residual
    ``X - S`` carries a STRONG shared rank-1 factor ``factor_scale * outer(f, v)``
    (per-row score ``f`` along a fixed channel loading ``v``) plus a small iid
    term, so ``cov(X - S) = factor_scale**2 * v vᵀ + iid_scale**2 I`` is
    off-diagonal — the regime an iid metric mismodels and a whitened metric
    down-weights along ``v``.
    """
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    base = np.column_stack([np.cos(theta), np.sin(theta), 0.5 * np.cos(2.0 * theta)])
    lift = rng.standard_normal((3, p))
    signal = base @ lift  # (n, p) noiseless
    v = rng.standard_normal(p)
    v /= np.linalg.norm(v)
    f = rng.standard_normal(n)
    structured = factor_scale * np.outer(f, v)
    idiosyncratic = iid_scale * rng.standard_normal((n, p))
    observed = signal + structured + idiosyncratic
    return (
        np.ascontiguousarray(observed, dtype=np.float64),
        np.ascontiguousarray(signal, dtype=np.float64),
    )


def _held_out_mspe(passes: int, x_train: np.ndarray, x_test: np.ndarray, s_test: np.ndarray) -> float:
    """Fit on ``x_train`` with ``passes`` whitening passes, then score the
    out-of-sample reconstruction of ``x_test`` against the noiseless truth."""
    fit = sae_manifold_fit(
        x_train,
        K=4,
        atom_topology="circle",
        n_iter=40,
        random_state=0,
        structured_residual_passes=passes,
    )
    assert isinstance(fit, ManifoldSAE)
    recon = np.asarray(fit.reconstruct(x_test), dtype=np.float64)
    assert recon.shape == s_test.shape
    return float(np.mean((recon - s_test) ** 2))


def test_structured_whitening_improves_held_out_recovery() -> None:
    """Whitened (structured) fit recovers the noiseless signal on held-out rows
    strictly better than the iid fit, averaged over independent planted draws."""
    n_train, n_test = 180, 60
    mspe_iid: list[float] = []
    mspe_whitened: list[float] = []
    for seed in (1, 2, 3):
        x_tr, _ = _planted_superposition(n_train, seed=seed)
        x_te, s_te = _planted_superposition(n_test, seed=seed + 100)
        mspe_iid.append(_held_out_mspe(0, x_tr, x_te, s_te))
        mspe_whitened.append(_held_out_mspe(2, x_tr, x_te, s_te))

    mean_iid = float(np.mean(mspe_iid))
    mean_whitened = float(np.mean(mspe_whitened))
    assert mean_whitened < mean_iid, (
        "structured-residual whitening did not improve held-out recovery under a "
        f"correlated residual: mean held-out MSPE(signal) whitened={mean_whitened:.6e} "
        f"is not < iid={mean_iid:.6e} (per-seed iid={mspe_iid}, whitened={mspe_whitened})"
    )


def test_structured_whitening_is_default_on() -> None:
    """The production default engages the structured metric: a default fit must
    differ from an explicit iid (``structured_residual_passes=0``) fit. Before
    the default flip these were bit-identical."""
    x_tr, _ = _planted_superposition(200, seed=7)

    default_fit = sae_manifold_fit(
        x_tr, K=4, atom_topology="circle", n_iter=25, random_state=0
    )
    iid_fit = sae_manifold_fit(
        x_tr, K=4, atom_topology="circle", n_iter=25, random_state=0,
        structured_residual_passes=0,
    )
    assert isinstance(default_fit, ManifoldSAE)
    assert isinstance(iid_fit, ManifoldSAE)

    recon_default = np.asarray(default_fit.fitted, dtype=np.float64)
    recon_iid = np.asarray(iid_fit.fitted, dtype=np.float64)
    assert recon_default.shape == recon_iid.shape
    assert not np.allclose(recon_default, recon_iid), (
        "the production sae_manifold_fit default did not engage structured-residual "
        "whitening — a default fit is bit-identical to an explicit "
        "structured_residual_passes=0 (iid) fit, so the metric is still opt-in."
    )
