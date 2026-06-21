"""Bug hunt: the default ``thinplate(x, z)`` smooth is not rotation-invariant.

A thin-plate spline is, mathematically, *exactly* rotation-invariant. Its kernel
``r^2 log r`` depends only on the pairwise Euclidean distance ``r`` between
points, and its polynomial null space ``span{1, x, z}`` is mapped onto itself by
any orthogonal rotation. So if the 2-D covariate ``(x, z)`` is rigidly rotated
about the data centroid by an angle ``theta`` and the same rotation is applied to
the prediction points, the fitted surface (the function on the plane) must be
unchanged: ``f_rot(R p) == f_orig(p)`` to machine precision.

It is not. The default low-rank ``thinplate(x, z)`` fit drifts ~1.5-4 % of the
signal range under rotation (a full 90-degree rotation included, which is exactly
representable in floating point and so introduces no rounding of its own).

Root cause (read, not patched): the low-rank knot set is chosen by
``select_equal_mass_centers`` in ``src/terms/basis/center_selection.rs``. Its
``choose_split_dim`` closure (same file, ~line 66) recursively splits each leaf
**along its widest *coordinate* dimension** — an axis-aligned k-d partition. That
selection rule is not rotation-equivariant: rotating the data changes which
coordinate is "widest" in each leaf, so a *different* set of knots is chosen, and
the rotated fit is a genuinely different (lower-rank) approximation. The rest of
the pipeline is invariant: with ``centers = n`` (every data point a knot, no
axis-aligned selection) the drift collapses to exactly 0, and a pure row
permutation leaves the fit bit-stable. So the defect is isolated to the
axis-aligned knot selection feeding the otherwise-isotropic thin-plate basis.

This is the *rotation* sibling of the thin-plate / Duchon *translation* defects
(#1269, #1375); the mechanism here is the axis-aligned center selector, distinct
from the uncentered-polynomial-null-space mechanism of those issues.

The test fits ``y ~ thinplate(x, z)`` on a (near-)isotropic radial signal, fits
again on the data rotated 90 degrees about its centroid, predicts each model at
its own (correspondingly rotated) training points, and asserts the two fitted
surfaces agree. A row-permutation control asserts the pipeline is otherwise
exact, so the rotation failure is specifically a rotation-invariance defect.

The test drives the ``gam`` CLI (on $PATH); the Python ``gamfit`` wheel is not
built in the hunt environment. It currently FAILS (rotation drift >> tolerance)
and will PASS once thin-plate knot selection is made rotation-equivariant.
"""

import csv
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

GAM = shutil.which("gam")


def _write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(list(r))


def _fit_predict(workdir, tag, train_xy, y, grid_xy):
    """Fit ``y ~ thinplate(x, z)`` and return the predicted mean at ``grid_xy``."""
    train = workdir / f"train_{tag}.csv"
    grid = workdir / f"grid_{tag}.csv"
    model = workdir / f"model_{tag}.gam"
    pred = workdir / f"pred_{tag}.csv"
    _write(train, ["x", "z", "y"], [(a, b, c) for (a, b), c in zip(train_xy, y)])
    _write(grid, ["x", "z"], [(a, b) for (a, b) in grid_xy])
    fit = subprocess.run(
        [GAM, "fit", str(train), "y ~ thinplate(x, z)", "--out", str(model)],
        capture_output=True,
        text=True,
    )
    assert fit.returncode == 0, f"fit ({tag}) failed: {fit.stderr}"
    pr = subprocess.run(
        [GAM, "predict", str(model), str(grid), "--out", str(pred)],
        capture_output=True,
        text=True,
    )
    assert pr.returncode == 0, f"predict ({tag}) failed: {pr.stderr}"
    return np.array([float(r["mean"]) for r in csv.DictReader(open(pred))])


@pytest.mark.skipif(GAM is None, reason="gam CLI not on PATH")
def test_thinplate_default_fit_is_rotation_invariant():
    rng = np.random.default_rng(21)
    n = 300
    X = rng.uniform(-1.0, 1.0, size=(n, 2))
    r2 = (X**2).sum(axis=1)
    # Near-isotropic radial signal (a true thin-plate spline reproduces it
    # identically in any rotated frame).
    y = 1.5 * np.exp(-r2 / 0.5) + rng.normal(0.0, 0.08, size=n)
    centroid = X.mean(axis=0)

    # Exact 90-degree rotation about the centroid: R = [[0,-1],[1,0]], so the
    # rotation itself is computed with only subtractions/additions (no rounding).
    R = np.array([[0.0, -1.0], [1.0, 0.0]])
    X_rot = (X - centroid) @ R.T + centroid

    with tempfile.TemporaryDirectory() as d:
        wd = Path(d)
        base = _fit_predict(wd, "orig", X, y, X)
        rot = _fit_predict(wd, "rot", X_rot, y, X_rot)

        # Row-permutation control: a pure relabeling must leave the fit
        # bit-stable, proving the rest of the pipeline is invariant.
        perm = rng.permutation(n)
        permuted = _fit_predict(wd, "perm", X[perm], y[perm], X)

    signal_range = float(base.max() - base.min())
    assert signal_range > 0.5, f"degenerate signal range {signal_range}"

    perm_drift = float(np.abs(base - permuted).max()) / signal_range
    rot_drift = float(np.abs(base - rot).max()) / signal_range

    # Control: permutation invariance holds to floating-point summation noise.
    assert perm_drift < 1e-6, (
        f"thin-plate fit is not even permutation-invariant (drift {perm_drift:.2e} "
        f"of signal range); the test harness assumption is violated."
    )

    # The bug: the rotated fit drifts far above the numerical floor that the
    # permutation control just established. A rotation-equivariant knot selection
    # drives this to ~1e-10; the current axis-aligned selector leaves ~1.5-4%.
    assert rot_drift < 1e-3, (
        f"thinplate(x, z) is NOT rotation-invariant: a 90-degree rotation about "
        f"the centroid moves the fitted surface by {rot_drift:.3%} of the signal "
        f"range (permutation control drifts only {perm_drift:.2e}). A thin-plate "
        f"spline depends only on pairwise distances and a rotation-invariant "
        f"{{1, x, z}} null space, so it must be exactly invariant. Root cause: "
        f"the axis-aligned equal-mass knot selector "
        f"(select_equal_mass_centers / choose_split_dim in "
        f"src/terms/basis/center_selection.rs) picks different knots in a rotated "
        f"frame; with centers=n the drift is exactly 0."
    )
