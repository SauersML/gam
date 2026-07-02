"""Bug hunt (#2030): ``gam diagnose <model> <data>`` must succeed on a standard
Gaussian smooth fit and emit non-empty ALO diagnostics.

Root cause: ``compact_fit_result_for_batch`` (``run_fit.rs``) was called
unconditionally on the main fit path and zeroed the row-sized working vectors
(``working_weights`` / ``working_response``) on the persisted *geometry*
carrier as well as the inference carrier. ``gam diagnose`` takes the geometry
ALO path in ``run_diagnose.rs`` whenever ``unified.geometry`` is ``Some`` and
hands ``geom.working_weights`` to ``AloInput::from_geometry``. With the vectors
emptied, ALO failed its length-N validation

    error: compute_alo_from_input (geometry path) failed: Invalid input:
           ALO diagnostics require hessian_weights length N; got 0

on EVERY standard fit. Because the field was present-but-empty, diagnose never
fell through to its refit fallback.

The fix retains the geometry carrier's working vectors through compaction (and
defensively treats an emptied carrier as "geometry unavailable" so diagnose
falls back to a refit). This test drives ``gam fit`` then ``gam diagnose`` on a
small standard Gaussian ``y ~ s(x)`` and asserts a zero exit plus non-empty ALO
output. It fails before the fix (non-zero exit, empty ALO) and passes after.
"""

from __future__ import annotations

import csv
import os
import subprocess

import numpy as np
import pytest

GAM_BIN = os.path.join(os.path.dirname(__file__), "..", "target", "release", "gam")

pytestmark = pytest.mark.skipif(
    not os.path.exists(GAM_BIN),
    reason="release `gam` CLI binary not built (target/release/gam absent)",
)


def _write_csv(path: str, cols: dict[str, np.ndarray]) -> None:
    keys = list(cols)
    n = len(cols[keys[0]])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(n):
            w.writerow([cols[k][i] for k in keys])


def test_diagnose_standard_gaussian_smooth_emits_alo(tmp_path):
    rng = np.random.default_rng(2030)
    n = 90
    x = rng.uniform(0, 1, n)
    y = 2.0 + np.sin(2 * np.pi * x) + rng.normal(0, 0.15, n)

    data = str(tmp_path / "d.csv")
    _write_csv(data, {"x": x, "y": y})
    model = str(tmp_path / "m.json")

    fit = subprocess.run(
        [GAM_BIN, "fit", data, "y ~ s(x)", "--out", model],
        capture_output=True, text=True, timeout=600,
    )
    assert fit.returncode == 0, f"gam fit failed:\n{fit.stdout}\n{fit.stderr}"

    diag = subprocess.run(
        [GAM_BIN, "diagnose", model, data],
        capture_output=True, text=True, timeout=600,
    )
    # Pre-fix: non-zero exit with the "hessian_weights length N; got 0" error.
    assert diag.returncode == 0, (
        f"gam diagnose failed (#2030):\n{diag.stdout}\n{diag.stderr}"
    )
    assert "hessian_weights length" not in diag.stderr, diag.stderr
    # ALO must have actually produced diagnostics (the leverage table).
    assert "ALO diagnostics" in diag.stdout, diag.stdout
    # The top-leverage table must have at least one data row (non-empty ALO).
    assert "leverage" in diag.stdout, diag.stdout
