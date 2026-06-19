"""Single-penalty thin-plate smooth over-fits exactly-linear data (issue #1271).

On purely linear data ``y = 2 + 3x + N(0, 0.15)`` a thin-plate regression spline
``s(x, bs="tp")`` should reduce to the affine fit: an intercept + a linear trend,
i.e. an effective degrees of freedom (EDF) of about ``2``.  mgcv's ``bs="tp"`` with
``select=TRUE`` lands at ``EDF ~ 2.10`` here; gam's ``bs="ps"`` single-penalty path
lands near ``EDF ~ 2.56`` (mgcv-consistent).

The defect (#1271): gam's ``bs="tp"`` single-penalty path over-fit the wiggle on
exactly-linear data, landing at ``EDF ~ 4.87`` (mean over 5 seeds) — REML
under-penalised the thin-plate bending modes relative to mgcv.  This is
INDEPENDENT of the double penalty (the null-space ridge is inert for tp on linear
data: ``double_penalty=True`` gave ``~4.88``), so it is distinct from #1266; it is
a pure single-penalty over-fit specific to the ``tp`` basis.

The control pins ``bs="ps"`` as the already-correct reference (``EDF ~ 2.5``),
isolating the defect to the ``tp`` path.
"""

from __future__ import annotations

import contextlib
import importlib
import io
from typing import Any

import numpy as np
import pandas as pd

pytest: Any = importlib.import_module("pytest")

pytest.importorskip("gamfit._rust")

import gamfit


N = 800
SEEDS = range(5)


def _linear_dgp(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, N)
    y = 2.0 + 3.0 * x + rng.normal(0.0, 0.15, N)
    return pd.DataFrame({"x": x, "y": y})


def _edf(formula: str, seed: int) -> float:
    df = _linear_dgp(seed)
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        m = gamfit.fit(df, formula, family="gaussian")
    return float(m.summary().edf_total)


def test_thinplate_single_penalty_does_not_overfit_linear_data() -> None:
    """``s(x, bs="tp")`` on exactly-linear data must collapse to the affine fit
    (EDF ~ 2), not over-fit to EDF ~ 4.87."""
    edfs = [_edf("y ~ s(x, k=20, bs=tp, double_penalty=False)", seed) for seed in SEEDS]
    mean_edf = float(np.mean(edfs))
    # The true expected value is ~2 (intercept + linear trend); mgcv lands at
    # ~2.10.  A correct fit lands comfortably below 3; the defect produced ~4.87.
    assert mean_edf <= 3.0, (
        f"thin-plate single-penalty EDF over-fits linear data: "
        f"mean EDF {mean_edf:.3f} across seeds {list(SEEDS)} (per-seed {edfs}); "
        f"expected ~2 (mgcv ~2.10), ceiling 3.0"
    )


def test_pspline_single_penalty_linear_data_control() -> None:
    """Control: ``s(x, bs="ps")`` is already correct on linear data (EDF ~ 2.5),
    isolating the defect to the tp path."""
    edfs = [_edf("y ~ s(x, k=20, bs=ps, double_penalty=False)", seed) for seed in SEEDS]
    mean_edf = float(np.mean(edfs))
    assert mean_edf <= 3.0, (
        f"control basis bs='ps' unexpectedly over-fit linear data: "
        f"mean EDF {mean_edf:.3f} (per-seed {edfs})"
    )


if __name__ == "__main__":  # pragma: no cover - manual smoke run
    for f in (
        "y ~ s(x, k=20, bs=tp, double_penalty=False)",
        "y ~ s(x, k=20, bs=ps, double_penalty=False)",
    ):
        vals = [_edf(f, s) for s in SEEDS]
        print(f"{f}: mean EDF {np.mean(vals):.4f}  per-seed {vals}")
    raise SystemExit(pytest.main([__file__, "-v"]))
