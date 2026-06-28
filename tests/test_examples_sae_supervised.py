"""End-to-end test for gamfit.examples.sae_supervised.

The example orchestrates two Rust kernels: ``sae_manifold_fit`` (full
``X``, unsupervised) and ``gamfit.fit`` (GLM head on the supervised
slice of the SAE latents). We check that:

1. The runner completes on a small synthetic dataset and exposes the
   uniform ``SaeSupervisedFit`` surface (``.sae``, ``.model``,
   ``.report()``, ``.predict``).
2. ``result.predict(X_train)`` returns predictions of the right shape
   and the in-sample fit is non-degenerate (positive R² on the
   supervised rows).
3. An empty supervised mask raises a clean ``ValueError`` *before* any
   expensive Rust call.
4. A wrong-length supervised mask raises a clean ``ValueError``.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamfit

# #1512 triage: the supervised-SAE example fits run well past the standard
# Python-API CI runner budget (the file timed out at >240s in triage), so they
# are tagged slow and excluded from the directory-level `-m "not slow"` CI step
# (still collected, and run by a bare `pytest tests/` locally).
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def synthetic() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Small synthetic dataset with structure detectable by the SAE.

    Two periodic factors drive 5 observed features plus mild noise; the
    response is a linear function of one of the factors.
    """
    rng = np.random.default_rng(0)
    n = 80
    theta = rng.uniform(-np.pi, np.pi, size=n)
    phi = rng.uniform(-np.pi, np.pi, size=n)
    # 5 features = sin/cos of theta and phi plus a noise channel.
    X = np.column_stack(
        [
            np.sin(theta),
            np.cos(theta),
            np.sin(phi),
            np.cos(phi),
            0.1 * rng.standard_normal(n),
        ]
    )
    X = X + 0.05 * rng.standard_normal(X.shape)
    # Response: linear-ish in theta with small noise.
    y = 1.5 * np.sin(theta) + 0.2 * rng.standard_normal(n)
    mask = np.zeros(n, dtype=bool)
    mask[: n // 2] = True  # First half is supervised.
    rng.shuffle(mask)
    return X.astype(np.float64), y.astype(np.float64), mask


def test_sae_supervised_end_to_end_returns_uniform_result(synthetic):
    X, y, mask = synthetic
    result = gamfit.examples.sae_supervised(
        X, y, mask, K=4, d_atom=2, atom_topology="circle",
    )
    assert isinstance(result, gamfit.SaeSupervisedFit)
    assert isinstance(result.sae, gamfit.ManifoldSAE)
    assert result.n_train == X.shape[0]
    assert result.n_supervised == int(mask.sum())
    assert len(result.latent_names) == 4
    # Report is a dict with expected top-level keys.
    report = result.report()
    assert report["example"] == "sae_supervised"
    assert report["latent_dim"] == 4
    assert "sae" in report and "head" in report


def test_sae_supervised_predicts_on_training_X(synthetic):
    X, y, mask = synthetic
    result = gamfit.examples.sae_supervised(
        X, y, mask, K=4, d_atom=2, atom_topology="circle",
    )
    preds = result.predict(X)
    preds = np.asarray(preds, dtype=np.float64).reshape(-1)
    assert preds.shape == (X.shape[0],)
    # Non-degenerate in-sample fit on the supervised slice.
    y_sup = y[mask]
    p_sup = preds[mask]
    ss_res = float(np.sum((y_sup - p_sup) ** 2))
    ss_tot = float(np.sum((y_sup - y_sup.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    assert r2 > 0.0, f"expected positive in-sample R^2, got {r2}"


def test_sae_supervised_oos_predict_is_explicit_not_silent(synthetic):
    X, y, mask = synthetic
    result = gamfit.examples.sae_supervised(
        X, y, mask, K=4, d_atom=2, atom_topology="circle",
    )
    X_new = X + 1e-6  # Not bit-equal to training.
    with pytest.raises(NotImplementedError, match="OOS SAE assignments"):
        result.predict(X_new)


def test_sae_supervised_empty_mask_is_clean_error():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((30, 4))
    y = rng.standard_normal(30)
    mask = np.zeros(30, dtype=bool)
    with pytest.raises(ValueError, match="zero rows"):
        gamfit.examples.sae_supervised(X, y, mask, K=3, d_atom=2)


def test_sae_supervised_wrong_length_mask_is_clean_error():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((30, 4))
    y = rng.standard_normal(30)
    mask = np.ones(20, dtype=bool)
    with pytest.raises(ValueError, match="length 20 but X has 30 rows"):
        gamfit.examples.sae_supervised(X, y, mask, K=3, d_atom=2)
