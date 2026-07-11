"""Regression for issue #2132 — the frozen-decoder OOS encode must decode the
SAME model the training fit reports.

The BSF micro arena isolated a cold out-of-sample encode collapse: a healthy
fitted state (native train R² 0.75) re-encoded its OWN training rows at R²
0.20 cold, and — dispositively — at R² 0.02 when warm-started at the trained
coords/logits. Warm-worse-than-cold proved the OOS solve was optimizing a
DIFFERENT objective than the one the training state converged under: the FFI
rebuilt ρ from the INITIAL ``sparsity_strength`` / ``smoothness`` scalars and
zero ARD (α = 1 per axis) instead of the terminal REML-selected ρ*. Under that
foreign objective the trained optimum is not stationary, so the Newton
descent walked the warm start away from it (ARD α = 1 drags every on-manifold
coordinate toward the chart origin, where all rows decode to one point).

The fix threads the terminal ρ* (``log_lambda_sparse``, per-atom
``log_lambda_smooth``, per-atom/axis ``log_ard``) from the fit payload through
``ManifoldSAE`` into ``sae_manifold_predict_oos``. This test pins the
re-encode contract on the public surface:

* warm re-encode of the training rows (seeded at the trained coords/logits)
  reconstructs at least as well as the fit's own ``fitted`` state, minus a
  small refinement tolerance — the trained state is (near-)stationary for the
  matched objective, so the solve must not walk away from it;
* cold re-encode of the training rows lands in the same basin (no collapse);
* the terminal ρ* is retained on the model and survives save/load, so a
  reloaded model re-encodes under the same objective.

Sized for the shared local box: N = 220, K = 2, n_iter = 6 (single-digit
seconds). The full-scale six-factor arena reproduction (N = 700, K = 6,
L0 = 2) runs on MSI via bench/bsf_manifold_zoo.py.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np

pytest: Any = import_module("pytest")
gamfit = pytest.importorskip("gamfit")


def _r2(x: np.ndarray, recon: np.ndarray) -> float:
    ss_res = float(np.sum((x - recon) ** 2))
    ss_tot = float(np.sum((x - x.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def _planted_two_circles(n: int = 220, p: int = 6, seed: int = 0) -> np.ndarray:
    """Two disjoint planted circle factors, one active per row (L0 = 1)."""
    rng = np.random.default_rng(seed)
    frames = rng.standard_normal((2, 2, p))
    for f in range(2):
        # Orthonormalize each factor's 2-frame so the circles are round.
        q, _ = np.linalg.qr(frames[f].T)
        frames[f] = q.T[:2]
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    active = rng.integers(0, 2, size=n)
    x = np.zeros((n, p))
    for i in range(n):
        f = frames[active[i]]
        x[i] = 3.0 * (np.cos(theta[i]) * f[0] + np.sin(theta[i]) * f[1])
    return x + 0.05 * rng.standard_normal((n, p))


def _fit(x: np.ndarray) -> Any:
    return gamfit.sae_manifold_fit(X=x, K=2, top_k=1, n_iter=6, random_state=0)


def _warm_start(fit: Any, n: int) -> tuple[np.ndarray, np.ndarray]:
    coords = [np.asarray(c, dtype=float) for c in fit.coords]
    d_max = max(int(c.shape[1]) for c in coords)
    t_init = np.zeros((len(coords), n, d_max))
    for k, c in enumerate(coords):
        t_init[k, :, : c.shape[1]] = c
    return t_init, np.asarray(fit.low_level_logits, dtype=float)


def test_oos_reencode_of_training_rows_matches_native_reconstruction() -> None:
    x = _planted_two_circles()
    fit = _fit(x)
    native = _r2(x, np.asarray(fit.fitted, dtype=float))
    assert native > 0.5, f"planted two-circle fit unexpectedly weak: native R²={native:.4f}"

    # The terminal REML-selected ρ* must be retained for the OOS solve (#2132).
    assert fit.selected_log_lambda_sparse is not None
    assert fit.selected_log_lambda_smooth is not None
    assert len(fit.selected_log_lambda_smooth) == len(fit.atoms)
    assert fit.selected_log_ard is not None
    assert len(fit.selected_log_ard) == len(fit.atoms)

    # Warm re-encode: seeded AT the trained coords/logits, the frozen-decoder
    # solve descends the matched objective, so it must not walk away from the
    # trained reconstruction. (Under the pre-fix foreign objective this decayed
    # to R² ≈ 0.02 from a 0.75 native state.)
    t_init, a_init = _warm_start(fit, x.shape[0])
    warm = fit.converged_latents(x, t_init=t_init, a_init=a_init)
    warm_r2 = _r2(x, np.asarray(warm["fitted"], dtype=float))
    assert warm_r2 >= native - 0.05, (
        f"warm re-encode walked away from the trained state: warm R²={warm_r2:.4f} "
        f"vs native R²={native:.4f} — the OOS solve is optimizing a different "
        "objective than the training fit reports"
    )

    # Cold re-encode of the SAME rows: the projection seed + matched objective
    # must land in the trained basin — no encode collapse.
    cold = fit.converged_latents(x)
    cold_r2 = _r2(x, np.asarray(cold["fitted"], dtype=float))
    assert cold_r2 >= native - 0.15, (
        f"cold re-encode collapsed: cold R²={cold_r2:.4f} vs native R²={native:.4f}"
    )


def test_selected_rho_survives_save_load_roundtrip(tmp_path: Any) -> None:
    x = _planted_two_circles(n=200)
    fit = _fit(x)
    path = tmp_path / "sae_2132.json"
    fit.save(path)
    loaded = gamfit.ManifoldSAE.load(path)
    assert loaded.selected_log_lambda_sparse == fit.selected_log_lambda_sparse
    np.testing.assert_allclose(
        np.asarray(loaded.selected_log_lambda_smooth, dtype=float),
        np.asarray(fit.selected_log_lambda_smooth, dtype=float),
    )
    assert len(loaded.selected_log_ard) == len(fit.selected_log_ard)
    for got, want in zip(loaded.selected_log_ard, fit.selected_log_ard):
        np.testing.assert_allclose(np.asarray(got, float), np.asarray(want, float))
    # The reloaded model re-encodes under the same objective: warm re-encode
    # still reproduces the trained reconstruction.
    native = _r2(x, np.asarray(fit.fitted, dtype=float))
    t_init, a_init = _warm_start(fit, x.shape[0])
    warm = loaded.converged_latents(x, t_init=t_init, a_init=a_init)
    warm_r2 = _r2(x, np.asarray(warm["fitted"], dtype=float))
    assert warm_r2 >= native - 0.05
