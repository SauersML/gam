"""Bug hunt: ``sae_fit(...)["summary"]["avg_active_atoms"]`` is structurally ~0
— the SAE sparsity (L0) diagnostic thresholds the wrong quantity, so it reports
that *no* atoms are active even when reconstruction R² ≈ 0.99 and every atom
carries mass in every row.

Root cause (``gamfit/_sae_manifold.py``, ``ManifoldSAE.summary``, lines
~2457-2470). For the default ``ordered_beta_bernoulli`` assignment mode the active-atom count
is computed as

    threshold = 0.5            # "active if its posterior gate exceeds 0.5"
    avg_active, _ = rust_module().sae_manifold_assignment_summary(self.assignments, threshold)

but ``self.assignments`` does **not** hold posterior gates — it holds the
normalized reconstruction *responsibilities* (``assignments_z``), which sum to
~1 across the K atoms per row. With K ≥ 2 the largest per-row responsibility can
essentially never reach 0.5 (here the max entry is 0.499), so the count of
entries ≥ 0.5 is ~0 and ``avg_active_atoms`` collapses to 0.0 regardless of how
many atoms are genuinely active. (The Rust helper is correct; the same module's
``_closed_form_trust_diagnostics`` even defines "active" as ``> 1e-8`` on the
same array, contradicting the 0.5 rule.)

This test fits an SAE on data built from K well-separated rank-1 atoms (so the
fit reconstructs it well and all atoms are used), then asserts that the reported
``avg_active_atoms`` is internally consistent with that reconstruction: a model
that reconstructs the data with R² ≈ 0.99 *must* have at least one active atom
per row on average. It currently fails (``avg_active_atoms == 0.0`` while
``reconstruction_r2 ≈ 0.99``); once the diagnostic thresholds the right quantity
the count becomes ≥ 1 and the assertion holds without edits.
"""

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

    res = gamfit.sae_fit(X)
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
