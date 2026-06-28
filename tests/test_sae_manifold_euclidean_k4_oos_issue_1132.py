"""#1132 regression: K>=4 euclidean SAE fit + out-of-sample reconstruct.

The real-data EV-vs-K curve (#1026) drives ``sae_manifold_fit`` at K>=4 with
euclidean (monomial-patch) atoms and then reconstructs a held-out block. Two
wrapper/FFI bugs killed that path:

* **bug 2** — the REML Laplace normalizer needs the joint-Hessian log-det. A
  beta-profiled euclidean atom assembles an arrow system with an empty shared
  beta block (``k == 0``), so the dense Direct solve forms no Schur complement
  and ``arrow_log_det_from_cache`` bailed with ``None``, erroring
  "arrow_log_det_from_cache returned None at ridge=0 Direct mode". A ``k == 0``
  cache is now treated as the per-row-sum (empty-Schur) case.
* **bug 3** — the euclidean OOS rebuild recovered the monomial degree from the
  user-facing ``(atom_dim, basis_size)`` while the trained build read the
  dimension off ``centers.ncols()``. When those disagreed the rebuilt basis
  width ``M`` no longer matched the trained decoder, erroring
  "decoder_blocks[0] has M=1 but rebuilt basis has M=3". The rebuild now anchors
  ``M`` to the trained decoder block row count.

This test asserts the end-to-end K=4 euclidean fit + OOS reconstruct completes
and returns a finite reconstruction (no log-det ``None`` and no M-mismatch).
"""
from __future__ import annotations

import numpy as np
import pytest

# #1512: this fit exceeds the standard Python-API CI runner budget (>60s in
# triage), so it is tagged slow and excluded from the directory-level
# `-m "not slow"` CI step while still being collected (run by a bare pytest).
pytestmark = pytest.mark.slow

gamfit = pytest.importorskip("gamfit")


def _patch_data(n: int, p: int, k: int, seed: int) -> np.ndarray:
    """k locally-linear 2-D patches mixed into a p-dim ambient space."""
    rng = np.random.default_rng(seed)
    blocks = []
    per = n // k
    for _ in range(k):
        coords = rng.normal(size=(per, 2))
        mixing = rng.normal(size=(2, p))
        mixing /= np.maximum(np.linalg.norm(mixing, axis=0, keepdims=True), 1e-8)
        blocks.append(coords @ mixing + 0.02 * rng.normal(size=(per, p)))
    x = np.vstack(blocks)
    x -= x.mean(axis=0, keepdims=True)
    return x


def test_euclidean_k4_fit_and_oos_reconstruct_issue_1132():
    x = _patch_data(n=480, p=48, k=4, seed=7)
    x_train, x_test = x[:360], x[360:]

    # K=4 is the regime that tripped the empty-Schur log-det and the OOS
    # M-mismatch; if either bug is live the fit or reconstruct raises here.
    fit = gamfit.sae_manifold_fit(
        X=x_train,
        K=4,
        atom_basis="euclidean",
        d_atom=2,
        assignment="ibp_map",
        n_iter=40,
        learning_rate=0.04,
        random_state=0,
    )

    assert hasattr(fit, "reconstruct") or hasattr(fit, "predict"), (
        "K>=4 euclidean fit must expose `reconstruct` or `predict` for "
        "held-out reconstruction (#1132)."
    )
    recon = fit.reconstruct(x_test) if hasattr(fit, "reconstruct") else fit.predict(x_test)
    recon = np.asarray(recon, dtype=float)

    assert recon.shape == x_test.shape, (
        f"OOS reconstruction shape {recon.shape} must match held-out block "
        f"{x_test.shape} (#1132 bug 3 M-consistency)."
    )
    assert np.all(np.isfinite(recon)), (
        "OOS reconstruction must be finite — a None joint-Hessian log-det "
        "(#1132 bug 2) corrupts the REML fit and poisons the reconstruction."
    )
