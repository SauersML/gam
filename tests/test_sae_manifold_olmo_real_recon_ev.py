"""#1026 — RED e2e: production manifold-SAE on REAL OLMo-3-32B activations.

This is the real end-to-end task the manifold-SAE machinery exists for:
fit the production SAE (`gamfit.sae_manifold_fit` / `ManifoldSAE`) on a
train split of genuine LLM residual-stream activations, reconstruct a
*held-out* split via the OOS path (`m.reconstruct(z_test)`), and require a
principled, defensible held-out reconstruction-quality bar.

The fixture `tests/data/olmo_l25_pca64_768.npy` is a real-activation slice:
OLMo-3-32B layer-25 residual stream, 768 token positions, mean-centered and
PCA-reduced to 64 dims (the SAE PCA-reduces its input anyway). Spectrum:
PC1=0.253, cum@K8=0.459, cum@K32=0.631, cum@K64=0.737 — a structured
minority sitting on an unstructured bulk, which is exactly the regime the
manifold dictionary is supposed to win in.

WHY THIS IS RED (a real e2e gap, not a flaky/bogus assertion):
  The defensible bar here is *linear parity*: a TopK / linear dictionary of
  K shards trivially attains held-out EV equal to the cumulative PCA
  spectrum at its rank. The whole wager of curved atoms (#1026 recon-parity
  roadmap) is that the manifold SAE matches-or-beats that linear ceiling at
  equal or lower K on real activations. We assert the production SAE clears
  a *modest fraction* of the K8 linear ceiling (cum@K8 = 0.459) on held-out
  real data. On real OLMo activations the curved-atom reconstruct path does
  not currently reach this bar — it under-recovers on the long-tailed real
  spectrum (degenerate-basin under-recovery, the #976/#1117 collapse class
  manifesting on real-vs-synthetic data). This pins the genuine e2e gap:
  the SAE reconstructs clean planted synthetic circles/spheres but does not
  yet earn its EV on real LLM activations.

Closing this requires the manifold dictionary to actually earn held-out EV
on real activations at parity with (or beating) the linear PCA ceiling at
equal K — i.e. the #1026 recon-parity frontier landing in the green on real
data, not just on planted synthetic harmonics.
"""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
import traceback

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")

_FIXTURE = Path(__file__).resolve().parent / "data" / "olmo_l25_pca64_768.npy"

_K = 8
_N_TRAIN = 384
_N_TEST = 128
_N_ITER = 32
_FIT_TIMEOUT_SECONDS = 90.0
# We do not even demand full parity with the linear ceiling — only that the
# curved SAE earns a meaningful fraction of the linear EV at matched rank on
# held-out real activations. The linear ceiling itself is recomputed from the
# committed fixture below (self-grounding; no trust in a hardcoded constant).
_TARGET_FRACTION = 0.50


def _ev(x: np.ndarray, fitted: np.ndarray) -> float:
    """Explained-variance fraction = 1 - SS_res / SS_tot, centered on x's
    own mean (the held-out baseline a constant predictor would attain)."""
    ss_res = float(np.sum((x - fitted) ** 2))
    ss_tot = float(np.sum((x - x.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def _heldout_linear_ceiling(z_train: np.ndarray, z_test: np.ndarray, k: int) -> float:
    """Held-out EV of a rank-k *linear* (PCA) reconstruction: fit the top-k
    principal subspace on the train split (centered on the train mean), project
    the held-out split onto it, score EV. This is the linear-dictionary ceiling
    a TopK SAE attains for free; the manifold SAE must match-or-beat a fraction
    of it. Computed from the committed fixture so the bar is self-grounding."""
    mu = z_train.mean(axis=0, keepdims=True)
    ztr = z_train - mu
    zte = z_test - mu
    _, _, vt = np.linalg.svd(ztr, full_matrices=False)
    proj = vt[:k].T @ vt[:k]
    recon = zte @ proj
    return _ev(z_test, recon)


def _load_fixture() -> np.ndarray:
    assert _FIXTURE.exists(), f"real OLMo fixture missing: {_FIXTURE}"
    z = np.load(_FIXTURE).astype(np.float64)
    assert z.shape == (768, 64), f"unexpected fixture shape {z.shape}"
    assert np.all(np.isfinite(z)), "fixture contains non-finite values"
    return z


def _fit_and_score_olmo_real(queue: mp.Queue) -> None:
    try:
        z = _load_fixture()
        z_train = z[:_N_TRAIN]
        z_test = z[_N_TRAIN : _N_TRAIN + _N_TEST]
        linear_ceiling = _heldout_linear_ceiling(z_train, z_test, _K)
        fit = gamfit.sae_manifold_fit(
            X=z_train,
            K=_K,
            atom_basis="periodic",
            d_atom=2,
            assignment="ibp_map",
            n_iter=_N_ITER,
            learning_rate=0.04,
            random_state=0,
        )
        in_sample_ev = _ev(z_train, fit.fitted)
        if not hasattr(fit, "reconstruct"):
            queue.put(
                {
                    "ok": False,
                    "error": (
                        "ManifoldSAE fit must expose `reconstruct` for OOS scoring; "
                        f"in-sample EV was {in_sample_ev:.4f}."
                    ),
                }
            )
            return
        oos = fit.reconstruct(z_test)
        queue.put(
            {
                "ok": True,
                "linear_ceiling": float(linear_ceiling),
                "in_sample_ev": float(in_sample_ev),
                "oos_ev": float(_ev(z_test, oos)),
                "oos_shape": tuple(oos.shape),
                "oos_finite": bool(np.all(np.isfinite(oos))),
            }
        )
    except BaseException:
        queue.put({"ok": False, "error": traceback.format_exc()})


def test_olmo_real_heldout_reconstruction_ev_meets_linear_parity():
    """Production manifold-SAE held-out reconstruction EV on real OLMo-3-32B
    activations must clear a fixed fraction of the rank-8 *linear* PCA ceiling,
    where that ceiling is recomputed from the committed fixture itself.

    RED: on real activations the curved-atom reconstruct path under-recovers
    relative to this bar (it clears the synthetic planted-circle bars but not
    the real long-tailed spectrum). This pins the #1026 recon-parity gap on
    REAL data.
    """
    z = _load_fixture()
    # Bounded deterministic split (no shuffle: the
    # fixture rows were already randomly sampled at extraction time).
    z_train = z[:_N_TRAIN]
    z_test = z[_N_TRAIN : _N_TRAIN + _N_TEST]
    assert z_train.shape == (_N_TRAIN, 64)
    assert z_test.shape == (_N_TEST, 64)

    # Self-grounding linear ceiling: the rank-K PCA held-out EV a TopK / linear
    # dictionary attains for free at the SAME rank K. The manifold SAE must earn
    # a fixed fraction of it on real data.
    linear_ceiling = _heldout_linear_ceiling(z_train, z_test, _K)
    ev_target = _TARGET_FRACTION * linear_ceiling

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_fit_and_score_olmo_real, args=(queue,))
    proc.start()
    proc.join(_FIT_TIMEOUT_SECONDS)
    if proc.is_alive():
        proc.terminate()
        proc.join(5.0)
        pytest.fail(
            "Real OLMo SAE fixture exceeded the bounded fit budget "
            f"({_FIT_TIMEOUT_SECONDS:.0f}s for N={_N_TRAIN}, K={_K}, n_iter={_N_ITER}); "
            "the outer loop is pinned or the evidence path is unbounded."
        )
    assert proc.exitcode == 0, f"OLMo fit subprocess exited with {proc.exitcode}"
    assert not queue.empty(), "OLMo fit subprocess produced no result"
    result = queue.get()
    assert result["ok"], result.get("error", "OLMo fit failed")
    assert result["oos_shape"] == z_test.shape
    assert result["oos_finite"], "OOS reconstruction produced NaN/Inf"
    assert result["linear_ceiling"] == pytest.approx(linear_ceiling)
    in_sample_ev = result["in_sample_ev"]
    oos_ev = result["oos_ev"]

    assert oos_ev >= ev_target, (
        f"Held-out reconstruction EV on real OLMo activations = {oos_ev:.4f}, "
        f"below the bar {ev_target:.4f} (= {_TARGET_FRACTION:.0%} of the "
        f"measured rank-{_K} linear PCA ceiling {linear_ceiling:.4f}). "
        f"In-sample EV was {in_sample_ev:.4f}. "
        f"The manifold SAE does not yet earn its EV on real LLM activations "
        f"at parity with the linear dictionary — the #1026 recon-parity gap "
        f"on REAL (not synthetic) data."
    )
