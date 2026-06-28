"""Regression guard for #808 (survival-marginal-slope clustered-PC stall).

#808: on a clustered-PC large-scale-style survival marginal-slope design (a matern
PC surface shared by the marginal-mean and log-slope channels, with enough
well-separated centers to create an operating-point alias), the inner
joint-Newton solve freezes with a huge non-stationary residual
(``residual ~= 3.7e8`` vs ``tol ~= 1.6e-4``) that never drops, the outer
REML/ARC smoothing never converges (frozen ``|g| = 1.863``), and the fit either
hard-errors (``survival marginal-slope outer optimization did not converge``)
or runs to its iteration/wall budget.

This test reconstructs that repro deterministically (it was lost once when the
ad-hoc ``repro_surv.py`` was deleted) so the bug can never silently regress and
the reproducer is preserved in-tree.

Status: #808 is OPEN. The v2 "W-aware operating-point identifiability
reduction" landed but does NOT fix it: on this design the priority-ordered
Gram-Schmidt selector drops the *entire* logslope block (``logslope N -> 0``,
because time+marginal already span its W-metric directions) yet the frozen
residual lives in the *time* block (``block_grad_inf ~= [159, 19, 2]``), so the
stall persists. The test therefore asserts the DESIRED post-fix contract
(the fit converges to a usable model) and is marked ``xfail(strict=True)``.
When #808 is genuinely fixed this test will XPASS; flip it to a hard assertion
at that point and delete the xfail marker.

The fit is run in a child process under a hard wall timeout so a stalled
solve cannot hang CI; a timeout is treated as "did not converge" (the bug).
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("gamfit")

# Wall budget for the child fit. A healthy (fixed) fit on this small clustered
# design converges in well under this; the #808 stall blows straight through it.
_FIT_TIMEOUT_SEC = 90

_CHILD = textwrap.dedent(
    """
    import sys
    import numpy as np
    import pandas as pd
    import gamfit

    rng = np.random.default_rng(0)
    n_centers, per = 3, 70
    centers = rng.normal(0, 5, size=(n_centers, 3))
    rows = []
    for c in range(n_centers):
        P = centers[c] + rng.normal(0, 0.5, size=(per, 3))
        z = rng.normal(size=per)
        a = -0.4 + 0.3 * P[:, 0] - 0.2 * P[:, 1]
        log_b = 0.0 + 0.2 * P[:, 0] - 0.1 * P[:, 2]
        eta = a + np.exp(log_b) * z
        age_entry = 40.0 + rng.uniform(0, 10, size=per)
        h = 0.02 * np.exp(0.5 * eta)
        follow = 3.0 + rng.uniform(0, 7, size=per)
        u = rng.uniform(size=per)
        t_event = -np.log(1.0 - u) / np.maximum(h, 1e-12)
        event = (t_event < follow).astype(int)
        dt = np.where(
            event == 1,
            np.maximum(np.minimum(t_event, follow - 0.01), 0.01),
            follow,
        )
        rows.append(pd.DataFrame({
            "age_entry": age_entry,
            "age_exit": age_entry + dt,
            "event": event,
            "PGS_z": z,
            "PC1": P[:, 0], "PC2": P[:, 1], "PC3": P[:, 2],
        }))
    d = pd.concat(rows, ignore_index=True)

    s = "matern(PC1, PC2, PC3, centers=8)"
    model = gamfit.fit(
        d,
        "Surv(age_entry, age_exit, event) ~ " + s,
        survival_likelihood="marginal-slope",
        z_column="PGS_z",
        logslope_formula=s,
    )
    # If we get here the outer solve converged (a non-converged outer raises
    # "outer optimization did not converge"). Sanity-check the fit is usable:
    # predict must produce finite, non-constant survival probabilities.
    pred = model.predict(d)
    sys.stdout.write("CONVERGED\\n")
    """
)


# #1512 / SPEC.md (xfail is never allowed): this stands FAILING as the signal of
# the open #808 bug — clustered-PC survival marginal-slope inner solve stalls
# (residual ~3.7e8 >> tol, frozen |g|=1.863); v2 W-aware reduction drops the
# whole logslope block but the residual lives in the time block, so the outer
# REML never converges. Fix #808 to green this.
def test_survival_marginal_slope_clustered_pc_converges_808() -> None:
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _CHILD],
            capture_output=True,
            text=True,
            timeout=_FIT_TIMEOUT_SEC,
            env=env,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            "survival marginal-slope clustered-PC fit did not converge within "
            f"{_FIT_TIMEOUT_SEC}s (the #808 inner-solve stall)."
        )
    assert proc.returncode == 0, (
        "survival marginal-slope clustered-PC fit failed to converge "
        f"(rc={proc.returncode}).\nstderr tail:\n"
        + "\n".join(proc.stderr.strip().splitlines()[-8:])
    )
    assert "CONVERGED" in proc.stdout, (
        "fit returned without the expected CONVERGED marker; stdout:\n"
        + proc.stdout
    )
