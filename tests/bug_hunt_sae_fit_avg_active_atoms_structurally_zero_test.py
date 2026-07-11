"""The native fit summary must count the same applied codes it reconstructs with."""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit


def test_sae_fit_avg_active_atoms_is_consistent_with_reconstruction() -> None:
    rng = np.random.default_rng(0)
    N, p, K = 200, 16, 4
    Q, _ = np.linalg.qr(rng.standard_normal((p, p)))
    dirs = Q[:K]
    labels = rng.integers(0, K, size=N)
    X = rng.uniform(1.0, 3.0, size=N)[:, None] * dirs[labels]

    res = gamfit.sae_manifold_fit(X, K=K)
    s = res.summary()
    r2 = float(s["reconstruction_r2"])
    avg_active = float(s["avg_active_atoms"])
    assignments = np.asarray(res.assignments, dtype=float)

    # The fit genuinely reconstructs the data, so atoms ARE active.
    assert r2 > 0.5, f"sanity: SAE should reconstruct this structured data, got R²={r2:.3f}"
    # Every row carries mass on several atoms (responsibilities sum to ~1).
    true_l0 = float(np.mean(np.sum(assignments > 1e-8, axis=1)))
    assert true_l0 >= 1.0

    # A sparsity/L0 diagnostic cannot report zero active atoms when the model
    # reconstructs the data well and the assignment matrix shows atoms carrying
    # mass. avg_active_atoms must be at least 1 active atom per row on average.
    assert avg_active >= 1.0, (
        f"avg_active_atoms={avg_active} is internally inconsistent: "
        f"reconstruction_r2={r2:.3f} and mean nonzero atoms/row={true_l0:.2f}, "
        f"yet the diagnostic reports ~0 active atoms (it thresholds normalized "
        f"responsibilities at 0.5, which K≥2 rows cannot reach)."
    )
