"""#1464 contract: curv() constant-curvature smooth must recover the *sign* of the
true curvature through the Python full-fit path, on data drawn from its own kernel
family.

The bug (#1464) is that hyperbolic data (kappa* < 0) can be recovered as spherical:
`kappa_hat` rails to the positive chart bound. The Rust `bug_hunt_1464_*` tests
exercise the same curvature-identifiability contract. This is the matching guard
for the *Python* `gamfit.fit(...).curvature(...)` full-fit path the issue names.

Previously this file was a print-only diagnostic (no test_ function, no asserts) and
was not collected by pytest, so it guarded nothing — it is now an asserting gate.
"""

import math

import numpy as np
import pandas as pd
import pytest

import gamfit


def _mobius_add(kappa, x, y):
    """κ-stereographic Möbius addition, the chart realization `gam_geometry` uses."""
    xy, xx, yy = float(x @ y), float(x @ x), float(y @ y)
    denom = 1.0 - 2.0 * kappa * xy + kappa * kappa * xx * yy
    return ((1.0 - 2.0 * kappa * xy - kappa * yy) * x + (1.0 + kappa * xx) * y) / denom


def _distance(kappa, x, y):
    """Geodesic distance `d_κ(x, y) = 2·T(√|κ|·‖w‖)/√|κ|`, `w = (−x) ⊕_κ y`."""
    w = np.linalg.norm(_mobius_add(kappa, -x, y))
    if kappa > 0:
        return 2.0 * math.atan(math.sqrt(kappa) * w) / math.sqrt(kappa)
    if kappa < 0:
        return 2.0 * math.atanh(math.sqrt(-kappa) * w) / math.sqrt(-kappa)
    return 2.0 * w


def _realized_centres(points, count=10):
    """The centres `curv(x1, x2, centers=count)` realizes on `points`.

    Farthest-point sampling seeded at the row nearest the centroid (column sums
    in value order), each step taking the row farthest from those already chosen
    (`gam_terms::basis::select_thin_plate_knots`), then the centre nearest the
    chart origin moved onto it (`select_constant_curvature_centers`). The exact
    tie-breaks that rule applies to symmetric clouds never engage on a uniform
    random sample."""
    centroid = np.sort(points, axis=0).sum(axis=0) / len(points)
    seed = int(np.argmin(((points - centroid) ** 2).sum(axis=1)))
    chosen = [seed]
    nearest = ((points - points[seed]) ** 2).sum(axis=1)
    while len(chosen) < count:
        nearest[chosen] = -np.inf
        pick = int(np.argmax(nearest))
        chosen.append(pick)
        nearest = np.minimum(nearest, ((points - points[pick]) ** 2).sum(axis=1))
    centres = points[chosen].copy()
    centres[int(np.argmin((centres**2).sum(axis=1)))] = 0.0
    # Lexicographic order, the order the draw assigns its coefficients in, so the
    # draw does not depend on the order the selector lists its centres in.
    return centres[np.lexsort((centres[:, 1], centres[:, 0]))]


# Coefficients of the draw, in lexicographic centre order; they sum to zero, the
# term's `CenterSumToZero` constraint.
WEIGHTS = np.array([2.0, -1.5, 1.2, -1.0, 0.8, -0.6, 0.5, -0.4, 0.3, -1.3])


def kernel_draw(kappa_star, seed=1, n=600, radius=0.68, noise=0.02):
    """A response drawn from the family curv() fits: its own kernel
    exp(-d_κ*(x, c_j)/ℓ*), ℓ* = 1, at the term's realized centres `c_j`, with
    coefficients summing to zero as the term's constraint requires.

    The contract can only ask curv() to recover the κ of data its family generates.
    The previous generator, `2·exp(-d_hyp) - 1`, is not such a draw: within the
    family (whose range is closed by the geodesic-distance kernel -d_κ, #2747) a
    spherical -d_κ explains it better than any hyperbolic fit, by ~90 log-likelihood
    units at this n and noise, so the likelihood contradicted the sign it asked for
    (gam#1464, see the Rust `kappa_recovery_1464_tests`, which pin that case). A
    kernel combination at centres the basis does not hold, or with coefficients off
    its constraint, is outside the span at every κ too, and its best approximation
    need not sit at κ*: hand-placed centres published κ* = −2 as κ̂ ≈ +2."""
    rng = np.random.default_rng(seed)
    points, eps = [], []
    while len(points) < n:
        a, b = 2 * rng.random() - 1, 2 * rng.random() - 1
        if a * a + b * b > 1.0:
            continue
        points.append([a * radius, b * radius])
        eps.append(noise * rng.standard_normal())
    points = np.array(points)
    centres = _realized_centres(points)
    y = [
        sum(w * math.exp(-_distance(kappa_star, p, c)) for c, w in zip(centres, WEIGHTS)) + e
        for p, e in zip(points, eps)
    ]
    return pd.DataFrame({"y": y, "x1": points[:, 0], "x2": points[:, 1]})


@pytest.mark.parametrize("kappa_star", [+2.0, -2.0])
def test_curv_recovers_constant_curvature_sign(kappa_star):
    df = kernel_draw(kappa_star)
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
