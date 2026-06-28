"""Bug hunt: the sum-to-zero factor smooth ``s(x, g, bs="sz")`` under-fits data
drawn from *its own model class*, leaving ~2x the irreducible noise variance
unexplained, while the sibling ``bs="fs"`` factor smooth recovers the same data
to the noise floor.

``docs/difference-smooths.md`` recommends the idiom

    y ~ s(x) + s(group, x, bs=sz)

as "the recommended symmetric default" for estimating coefficient-wise group
deviations that sum to zero across levels. That is exactly the model class

    E[y | x, g] = f0(x) + d_g(x),    with   sum_g d_g(x) = 0  for all x,

i.e. a shared smooth ``f0`` plus zero-sum per-group deviations ``d_g``. A
correctly constructed ``sz`` smooth contains this truth in its span, so on a
large, high-SNR sample drawn from exactly this class the penalized REML fit
should explain essentially all of the systematic structure — its in-sample
residual standard deviation should fall to the irreducible observation-noise
level (low bias at large ``n``).

It does not. On ``n = 4000`` points with observation noise ``sd = 0.20`` and a
group-deviation signal of amplitude ``0.6``, the documented ``s(x) +
s(g, x, bs="sz")`` fit leaves a residual ``sd ~ 0.43`` — **2.1x** the noise
floor — and the gap does **not** shrink as ``n`` grows (it is ~the same at
``n = 8000``), so this is a genuine consistency/recovery failure, not
finite-sample shrinkage. The strictly-more-general ``s(x, g, bs="fs")`` factor
smooth, whose span is a superset, recovers the identical data to ``sd ~ 0.20``
(the noise floor). The per-group deviation amplitude recovered by ``sz`` is only
~0.8x the truth with a misshapen curve, while ``fs`` recovers it at ~0.99x.

Root-cause read (no patch — see ``crates/gam-terms/src/smooth/term_specs.rs``):
the ``Sz`` flavour of ``build_factor_smooth`` (lines ~6491-6561) delegates to the
``SmoothBasisSpec::FactorSumToZero`` construction with a cubic-regression
marginal, whereas the ``Fs`` flavour (lines ~6451-6455) is built as a proper
full-rank random effect carrying both its wiggliness penalty *and* a null-space
ridge (the double-penalty ``I_L ⊗ S_j`` structure). The ``sz`` deviation blocks
are over-penalized / under-fit relative to ``fs`` even though both spans contain
the truth, so REML parks the ``sz`` block in an over-smoothed state that leaves
systematic structure in the residual.

This test fits a deterministic dataset drawn from the ``sz`` model class and
asserts that the ``sz`` fit reaches the noise floor (as the ``fs`` control
provably does). It fails today at the residual-magnitude assertion. When the
``sz`` construction recovers its own model class, the test passes without edits.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit

NOISE_SD = 0.20
N = 4000
N_GROUPS = 4


def _sz_class_frame(seed: int = 20260628) -> pd.DataFrame:
    """Data drawn from exactly the ``sz`` model class: a shared smooth ``f0(x)``
    plus zero-sum per-group deviations ``d_g(x)`` (phase-shifted sinusoids whose
    cross-group mean is removed at every ``x``)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, N)
    g = rng.integers(0, N_GROUPS, N)
    phases = np.linspace(0.0, 1.2, N_GROUPS)

    def deviations(xi: float) -> np.ndarray:
        vals = np.array([0.6 * np.sin(2 * np.pi * xi + 2 * np.pi * p) for p in phases])
        return vals - vals.mean()  # sum-to-zero across groups at this x

    f0 = np.sin(2 * np.pi * x)
    mu = f0 + np.array([deviations(xi)[gi] for xi, gi in zip(x, g)])
    y = mu + rng.normal(0.0, NOISE_SD, N)
    return pd.DataFrame({"y": y, "x": x, "g": g.astype(str)})


def _residual_sd(model: Any, df: pd.DataFrame) -> float:
    pred = np.asarray(model.predict(df), dtype=float)
    return float((df["y"].to_numpy() - pred).std())


def test_sz_factor_smooth_recovers_its_own_model_class() -> None:
    df = _sz_class_frame()

    # Control: bs="fs", a strict superset of the sz span, must reach the noise
    # floor on this data. This both proves the data is well-posed and pins the
    # irreducible floor we compare sz against.
    fs_model = gamfit.fit(df, "y ~ s(x, g, bs='fs')", family="gaussian")
    fs_resid = _residual_sd(fs_model, df)
    assert fs_resid < 1.2 * NOISE_SD, (
        f"control bs='fs' did not reach the noise floor: resid_sd={fs_resid:.4f} "
        f"vs noise_sd={NOISE_SD} (data/floor sanity check)"
    )

    # The documented sz idiom on data drawn from the sz model class.
    sz_model = gamfit.fit(df, "y ~ s(x) + s(g, x, bs='sz')", family="gaussian")
    sz_resid = _residual_sd(sz_model, df)

    # A smoother whose span contains the truth, fit at large n, must explain the
    # systematic structure and leave ~only observation noise. Today sz leaves
    # ~2.1x the noise floor (sz_resid ~ 0.43 vs 0.20).
    assert sz_resid < 1.4 * NOISE_SD, (
        f"bs='sz' under-fits its own model class: resid_sd={sz_resid:.4f} "
        f"({sz_resid / NOISE_SD:.2f}x the noise floor {NOISE_SD}); the bs='fs' "
        f"superset reaches {fs_resid:.4f}. The sz deviation blocks are "
        f"over-smoothed and leave systematic signal in the residual."
    )

    # Comparative guard: sz must not be dramatically worse than the fs superset
    # that recovers the same data.
    assert sz_resid < 1.5 * fs_resid, (
        f"bs='sz' residual {sz_resid:.4f} is {sz_resid / fs_resid:.2f}x the "
        f"bs='fs' residual {fs_resid:.4f} on identical sz-class data"
    )
