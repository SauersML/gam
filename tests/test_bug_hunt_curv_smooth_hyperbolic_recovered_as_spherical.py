"""#1464 contract: curv() constant-curvature smooth must recover the *sign* of the
true curvature through the Python full-fit path.

The bug (#1464) was that hyperbolic data (kappa* < 0) was recovered as spherical —
`kappa_hat` railed to the positive chart bound because the accept-gate scored a
sign-blind raw V_p criterion. The fix pins the constant-curvature baseline kappa to
the sign-correct kappa-fair fast-path scan (spatial_optimization.rs). The Rust e2e
contract is covered by tests/owed_1464.rs / bug_hunt_1464_*; this is the matching
guard for the *Python* `gamfit.fit(...).curvature(...)` full-fit path the issue names.

Previously this file was a print-only diagnostic (no test_ function, no asserts) and
was not collected by pytest, so it guarded nothing — it is now an asserting gate.
"""

import math

import numpy as np
import pandas as pd
import pytest

import gamfit


def curved(kappa_star, seed=1, n=600, radius=0.68, noise=0.02):
    """Sample a radial response whose decay distance follows the constant-curvature
    geodesic for the given signed curvature (hyperbolic kappa*<0, spherical kappa*>0,
    flat kappa*=0)."""
    rng = np.random.default_rng(seed)
    root = math.sqrt(abs(kappa_star))
    x1, x2, y = [], [], []
    while len(y) < n:
        a, b = 2 * rng.random() - 1, 2 * rng.random() - 1
        if a * a + b * b > 1.0:
            continue
        u, v = a * radius, b * radius
        r = math.hypot(u, v)
        if kappa_star < 0:  # hyperbolic
            d = 2 * math.atanh(min(root * r, 1 - 1e-9)) / root
        elif kappa_star > 0:  # spherical
            d = 2 * math.atan(root * r) / root
        else:  # flat
            d = 2 * r
        x1.append(u)
        x2.append(v)
        y.append(2 * math.exp(-d) - 1 + noise * rng.standard_normal())
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


@pytest.mark.parametrize("kappa_star", [+2.0, -2.0])
def test_curv_recovers_constant_curvature_sign(kappa_star):
    df = curved(kappa_star)
    rep = gamfit.fit(df, "y ~ curv(x1, x2, centers=10)").curvature(df)[0]
    kappa_hat = rep["kappa_hat"]
    print(
        f"truth kappa*={kappa_star:+}: kappa_hat={kappa_hat:+.4f} "
        f"ci=({rep['ci_lo']:+.3f},{rep['ci_hi']:+.3f}) "
        f"verdict={rep['verdict']} flat_p={rep['flatness_p_value']:.3g}"
    )
    assert math.isfinite(kappa_hat), f"kappa_hat is not finite: {kappa_hat}"
    # #1464 core contract: the recovered curvature must carry the TRUE sign.
    # The original bug railed hyperbolic (kappa*=-2) data to a positive
    # (spherical) kappa_hat; the regression is precisely that this no longer
    # happens for either chart.
    assert math.copysign(1.0, kappa_hat) == math.copysign(1.0, kappa_star), (
        f"curv() recovered the WRONG curvature sign: truth kappa*={kappa_star:+}, "
        f"kappa_hat={kappa_hat:+.4f} (hyperbolic-as-spherical #1464 regression)"
    )
