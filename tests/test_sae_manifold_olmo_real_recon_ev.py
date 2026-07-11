"""#1026 — RED e2e: production manifold-SAE on REAL OLMo-3-32B activations.

This is the real end-to-end task the manifold-SAE machinery exists for:
fit the production SAE (`gamfit.sae_manifold_fit` / `ManifoldSAE`) on a
train split of genuine LLM residual-stream activations, reconstruct a
*held-out* split via the OOS path (`m.reconstruct(z_test)`), and require a
principled, defensible held-out reconstruction-quality bar.

The fixture `tests/data/olmo_mixedlayer_pca64_768.npy` is a real-activation
*cross-layer mixture* (see #1199): 768 rows sampled across the flattened
`(prompt, layer)` axis of the banked OLMo-3-32B `635 × 64 × 5120` corpus,
mean-centered and PCA-reduced to 64 dims. It is NOT a single-layer L25 slice
(a true L25 slice has 635 rows, one per prompt); its 64 leading PCs are
dominated by cross-layer nuisance variation. Spectrum: PC1=0.253, cum@K8=0.459,
cum@K32=0.631, cum@K64=0.737 — a long-tailed cloud, exactly the
ill-conditioned regime the manifold dictionary is stress-tested on.

WHY THIS IS RED (a real e2e gap, not a flaky/bogus assertion):
  The reference bar here is the rank-k held-out EV of the affine PCA subspace.
  This is a *reference upper bound* on what any rank-k LINEAR reconstruction
  can attain on held-out data (the optimal rank-k subspace by Eckart–Young),
  NOT a "ceiling the matched linear SAE attains for free": the production SAE
  decoder path is gated/assignment-structured and (for the euclidean atom) a
  degree-2 patch, not a dense rank-k orthogonal projector, so it does not
  automatically realize this subspace. The whole wager of curved atoms (#1026
  recon-parity roadmap) is that the manifold SAE matches-or-beats this linear
  reference at equal or lower K on real activations. We assert the production
  SAE clears a *modest fraction* of the K8 linear reference (cum@K8 = 0.459)
  on held-out real data. On real OLMo activations the curved-atom reconstruct path does
  not currently reach this bar — it under-recovers on the long-tailed real
  spectrum (degenerate-basin under-recovery, the #976/#1117 collapse class
  manifesting on real-vs-synthetic data). This pins the genuine e2e gap:
  the SAE reconstructs clean planted synthetic circles/spheres but does not
  yet earn its EV on real LLM activations.

Closing this requires the manifold dictionary to actually earn held-out EV
on real activations at parity with (or beating) the rank-k linear PCA
reference at equal K — i.e. the #1026 recon-parity frontier landing in the
green on real data, not just on planted synthetic harmonics.
"""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
import traceback

import numpy as np
import pytest

# #1200 — this is a real-data REGRESSION GATE, not a convenience demo. The fit runs
# on CPU (no GPU dependency), so there is no environment in which this is legitimately
# skippable: a missing `gamfit` wheel must HARD-FAIL the suite, never silently skip —
# a skipped gate is not a gate. The import is therefore unconditional and unguarded
# (no `pytest.importorskip`, no env-var toggle): if the wheel is not built, collecting
# this module raises ImportError and the suite errors, which is the whole point.
import gamfit  # noqa: F401

_FIXTURE = Path(__file__).resolve().parent / "data" / "olmo_mixedlayer_pca64_768.npy"

_K = 8
_N_TRAIN = 384
_N_TEST = 128
_N_ITER = 32
_FIT_TIMEOUT_SECONDS = 90.0
# We do not even demand full parity with the linear PCA reference — only that
# the curved SAE earns a meaningful fraction of the linear EV at matched rank on
# held-out real activations. The linear reference itself is recomputed from the
# committed fixture below (self-grounding; no trust in a hardcoded constant).
_TARGET_FRACTION = 0.50


def _ev(x: np.ndarray, fitted: np.ndarray) -> float:
    """Explained-variance fraction = 1 - SS_res / SS_tot, centered on x's
    own mean (the held-out baseline a constant predictor would attain)."""
    ss_res = float(np.sum((x - fitted) ** 2))
    ss_tot = float(np.sum((x - x.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def _heldout_linear_ceiling(z_train: np.ndarray, z_test: np.ndarray, k: int) -> float:
    """Held-out EV of a rank-k *linear* (affine PCA) reconstruction: fit the
    top-k principal subspace on the train split (centered on the train mean),
    project the held-out split onto it, restore the mean, score EV. This is the
    optimal rank-k LINEAR subspace by Eckart–Young — a *reference upper bound*
    on any rank-k linear reconstruction of held-out data, NOT a ceiling the
    gated/quadratic-patch SAE decoder attains for free. The manifold SAE must
    match-or-beat a fraction of it. Computed from the committed fixture so the
    bar is self-grounding."""
    mu = z_train.mean(axis=0, keepdims=True)
    ztr = z_train - mu
    zte = z_test - mu
    _, _, vt = np.linalg.svd(ztr, full_matrices=False)
    proj = vt[:k].T @ vt[:k]
    # Restore the train mean before scoring so the reconstruction lives in the
    # SAME (uncentered) coordinates as the target z_test. Scoring a centered
    # reconstruction `zte @ proj` against an uncentered z_test mixes coordinate
    # frames and yields a wrong EV (the projection cannot reproduce the constant
    # mean offset it never saw). With the mean added back, EV is consistently the
    # held-out reconstruction quality of the rank-k AFFINE PCA model.
    recon = mu + zte @ proj
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
            assignment="ordered_beta_bernoulli",
            n_iter=_N_ITER,
            learning_rate=0.04,
            random_state=0,
        )
        uses_affine_pca_lane = (
            getattr(fit, "_oos_affine_pca_mean", None) is not None
            or getattr(fit, "_oos_affine_pca_components", None) is not None
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
                "uses_affine_pca_lane": bool(uses_affine_pca_lane),
            }
        )
    except BaseException:
        queue.put({"ok": False, "error": traceback.format_exc()})


def test_olmo_real_heldout_reconstruction_ev_meets_linear_parity():
    """Production manifold-SAE held-out reconstruction EV on real OLMo-3-32B
    activations must clear a fixed fraction of the rank-8 *linear* PCA reference
    upper bound, where that reference is recomputed from the committed fixture
    itself.

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

    # Self-grounding linear reference: the rank-K affine PCA held-out EV — the
    # optimal rank-K LINEAR subspace (Eckart–Young upper bound), NOT a ceiling
    # the gated/quadratic-patch SAE decoder attains for free. The manifold SAE
    # must earn a fixed fraction of it on real data.
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
    assert not result["uses_affine_pca_lane"], (
        "The real-OLMo #1026 gate must exercise the production manifold-SAE "
        "decoder and frozen-decoder OOS solve. Returning an affine PCA projector "
        "from sae_manifold_fit(..., atom_basis='periodic', assignment='ordered_beta_bernoulli') "
        "only reproduces the linear reference model this test is using as the "
        "bar; it does not show the curved manifold dictionary earned held-out EV "
        "on real activations."
    )
    assert result["linear_ceiling"] == pytest.approx(linear_ceiling)
    in_sample_ev = result["in_sample_ev"]
    oos_ev = result["oos_ev"]

    assert oos_ev >= ev_target, (
        f"Held-out reconstruction EV on real OLMo activations = {oos_ev:.4f}, "
        f"below the bar {ev_target:.4f} (= {_TARGET_FRACTION:.0%} of the "
        f"measured rank-{_K} linear PCA reference upper bound "
        f"{linear_ceiling:.4f}). "
        f"In-sample EV was {in_sample_ev:.4f}. "
        f"The manifold SAE does not yet earn its EV on real LLM activations "
        f"at parity with the linear dictionary — the #1026 recon-parity gap "
        f"on REAL (not synthetic) data."
    )
