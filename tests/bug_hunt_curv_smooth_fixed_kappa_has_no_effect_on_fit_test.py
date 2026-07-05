"""Bug hunt: the ``curv()`` smooth's documented FIXED ``kappa=`` has zero effect.

The constant-curvature (``M_κ``) geodesic-kernel smooth ``curv(x, z, kappa=…)``
is documented as taking a *fixed sectional curvature* that selects the smooth's
geometry — ``docs/formulas.md`` says it "interpolates ``Sᵈ → ℝᵈ → Hᵈ`` via
``kappa=`` (default 0, flat)", and the term-builder's own comment
(``crates/gam-terms/src/term_builder.rs:2652``) states "``kappa=`` is the
**fixed** sectional curvature (default 0 = flat)". Positive κ is a spherical
geometry; negative κ is hyperbolic; the two produce genuinely different
geodesic-distance kernels and therefore different design matrices — the basis
builder honours ``spec.kappa`` (``constant_curvature_kernel_matrix(...,
spec.kappa, ...)`` at ``constant_curvature_smooth.rs:668/695``).

Observed (this test, reproduced identically through the ``gam`` CLI):
``curv(x, z, kappa=K)`` fits are **bit-for-bit identical** for every fixed
``K`` — a spherical ``kappa=+3`` fit, a hyperbolic ``kappa=-3`` fit, and the
flat ``kappa=0`` fit return the same predictions to the last ULP
(``max|Δ| == 0.0``). The user's fixed curvature is silently discarded.

Expected: a fixed ``kappa=+3`` and a fixed ``kappa=-3`` build different
geodesic kernels, so their fits (hence predictions) must differ.

Root-cause read: for a pure-``curv()`` spatial problem the fit does not hold the
user's ``spec.kappa`` fixed — it re-derives κ. The pure-CC branch in
``crates/gam-models/src/fit_orchestration/drivers/spatial_optimization.rs``
(``all_spatial_are_cc``, line ~2583) either overrides ``cc.kappa = kappa_hat``
from the κ-fair argmin (line 2606) or, for the non-hyperbolic case, "falls
through to the joint path below" that jointly optimises ``[ρ, κ]`` — a search
the surrounding comments describe as monotone toward the +chart bound and hence
seed-independent. Either way the converged κ̂ does not depend on the user's fixed
``kappa=``, so every ``kappa=`` seed lands on the same optimum and the "fixed
sectional curvature" the docs and the term-builder promise is unreachable. (An
independent tell: ``curv(x, z, kappa=-8)`` was accepted on data with
``‖x‖² ≈ 8``, which lies OUTSIDE the ``1 + κ‖x‖² > 0`` chart of ``κ=-8`` and
would have to be rejected by ``validate_chart_points`` if that κ were actually
used to build the design.)

When the fixed ``kappa=`` is honoured, the spherical and hyperbolic fits below
diverge and the test passes with no edits. The data lives strictly inside the
``κ = ±3`` chart (``‖x‖² < 1/3``) so both fixed-curvature fits are well posed.

Related: #2151 (the workspace does not build at HEAD; this test — like every
test target — can only run once that build break is cleared).
"""

from __future__ import annotations

import contextlib
import importlib
import os
from typing import Any

import numpy as np
import pandas as pd

pytest: Any = importlib.import_module("pytest")

pytest.importorskip("gamfit._rust")

import gamfit


@contextlib.contextmanager
def _silence() -> Any:
    so, se = os.dup(1), os.dup(2)
    dn = os.open(os.devnull, os.O_WRONLY)
    os.dup2(dn, 1)
    os.dup2(dn, 2)
    try:
        yield
    finally:
        os.dup2(so, 1)
        os.dup2(se, 2)
        os.close(dn)
        os.close(so)
        os.close(se)


def _make_data(seed: int = 41, n: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A smooth 2-D surface on a small ball ``‖(x, z)‖² < 1/3``.

    The radius bound keeps every row strictly inside BOTH the ``κ = +3`` and the
    ``κ = -3`` stereographic charts (``1 + κ‖x‖² > 0``), so a fixed-curvature fit
    at either sign is well posed rather than chart-rejected.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-0.3, 0.3, n)
    z = rng.uniform(-0.3, 0.3, n)
    y = np.sin(4.0 * x) * np.cos(4.0 * z) + 0.05 * rng.standard_normal(n)
    train = pd.DataFrame({"x": x, "z": z, "y": y})
    gx = rng.uniform(-0.28, 0.28, 60)
    gz = rng.uniform(-0.28, 0.28, 60)
    grid = pd.DataFrame({"x": gx, "z": gz})
    assert float((grid["x"] ** 2 + grid["z"] ** 2).max()) < 1.0 / 3.0
    return train, grid


def _fit_predict(train: pd.DataFrame, grid: pd.DataFrame, kappa: float) -> np.ndarray:
    with _silence():
        model = gamfit.fit(train, f"y ~ curv(x, z, kappa={kappa})", family="gaussian")
        pred = np.asarray(model.predict(grid), dtype=float).ravel()
    return pred


def test_curv_fixed_kappa_changes_the_fit() -> None:
    """A fixed spherical (+κ) and hyperbolic (−κ) curv() fit must differ.

    They currently do not: the fixed ``kappa=`` is discarded and every κ lands
    on the same fit (``max|Δ| == 0.0``).
    """
    train, grid = _make_data()

    pred_flat = _fit_predict(train, grid, 0.0)
    pred_sph = _fit_predict(train, grid, 3.0)
    pred_hyp = _fit_predict(train, grid, -3.0)

    for name, pred in (("kappa=0", pred_flat), ("kappa=+3", pred_sph), ("kappa=-3", pred_hyp)):
        assert np.all(np.isfinite(pred)), f"{name}: predictions must be finite"

    # A spherical and a hyperbolic geodesic kernel are genuinely different
    # geometries; their fixed-curvature fits cannot coincide to the last ULP.
    sph_vs_hyp = float(np.max(np.abs(pred_sph - pred_hyp)))
    assert sph_vs_hyp > 1e-6, (
        "fixed curv() kappa= has no effect: spherical kappa=+3 and hyperbolic "
        f"kappa=-3 fits are identical (max|Δ| = {sph_vs_hyp:.3e}); the documented "
        "'fixed sectional curvature' is silently discarded by the fit's κ re-derivation"
    )

    # The flat (κ=0) fit must also differ from a curved one.
    flat_vs_sph = float(np.max(np.abs(pred_flat - pred_sph)))
    assert flat_vs_sph > 1e-6, (
        "fixed curv() kappa= has no effect: flat kappa=0 and spherical kappa=+3 "
        f"fits are identical (max|Δ| = {flat_vs_sph:.3e})"
    )
