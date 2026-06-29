"""Bug hunt: the documented ``response_geometry="stiefel(k=...)"`` is
categorically unfittable for k >= 2.

``docs/response-geometry.md`` lists ``"stiefel(k=...)"`` (orthonormal k-frames)
as a supported manifold-valued response geometry, alongside ``"spherical"``,
``"simplex"``, ``"spd"``, ``"grassmann(k=...)"`` and ``"poincare"`` — and the
``gamfit.fit`` docstring advertises the same. A Stiefel response fit maps each
frame to the tangent space at the intrinsic Fréchet/Karcher mean of the training
frames, fits Gaussian REML in tangent coordinates, and exponentiates back.

For k == 1 the Stiefel manifold St(n, 1) is the unit sphere and the fit works
(it dispatches to the sphere maps). For **k >= 2 the fit cannot even start**:
the Fréchet-mean initializer
(``crates/gam-geometry/src/response_geometry.rs``, the seed loop at ~734-766)
seeds the Karcher iteration by evaluating ``log_point(base, x)`` for every
sample, but ``StiefelManifold::log_map``
(``crates/gam-geometry/src/manifolds/stiefel.rs:136-139``) deliberately returns
``GeometryError::Unsupported("Stiefel log_map: no closed-form Riemannian
logarithm for k > 1")`` for EVERY pair. So no sample is ever an admissible seed,
and the init aborts with

    response geometry Fréchet mean init: no admissible seed among samples
    (every sample lies at another's cut locus; last error: unsupported geometry
     operation: Stiefel log_map: no closed-form Riemannian logarithm for k > 1)

The "every sample lies at another's cut locus" wording misattributes the cause
to the data — but the failure is data-independent: there is *no* St(n, k>=2)
dataset, however tightly clustered, that can be fit, because the Stiefel
logarithm the initializer relies on is simply not implemented for k >= 2.

This test builds a deterministic, **tightly clustered** set of St(3, 2) frames
(0.05-scale perturbations of the canonical [e0, e1] frame — so the intrinsic
mean is manifestly well defined and unambiguous, with no genuine cut-locus
pair) and asserts the documented Stiefel response fit succeeds and that its
predictions are valid orthonormal frames (YᵀY = I_2). It currently fails at
``gamfit.fit`` with the Fréchet-mean init error above.

When the mean initializer is made to work for Stiefel k >= 2 — e.g. by seeding
from a retraction-based / induced mean (QR of the Euclidean average, which the
crate already has via ``StiefelManifold::retract``) instead of requiring the
unimplemented logarithm, or by implementing the canonical-metric Stiefel log —
the fit succeeds and this test passes with no further edits. (A k == 1 control
fit is included to show the breakage is specific to k >= 2.)
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit


def _clustered_stiefel_frames(seed: int = 11, n: int = 400):
    """``n`` St(3, 2) frames tightly clustered around the canonical [e0, e1]
    frame: small Gaussian perturbations re-orthonormalized by QR, with the
    column signs pinned so the frames do not straddle a sign cut. Returned as an
    ``n x 6`` row-major flatten of each 3x2 frame, plus two smooth covariates."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    z = rng.uniform(-1.0, 1.0, n)
    base = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    frames = np.empty((n, 6))
    for i in range(n):
        pert = base + 0.05 * rng.standard_normal((3, 2))
        q, _ = np.linalg.qr(pert)
        for j in range(2):
            if q[j, j] < 0.0:
                q[:, j] *= -1.0
        frames[i] = q.reshape(-1)
    return x, z, frames


def _predict_matrix(prediction: Any) -> np.ndarray:
    if hasattr(prediction, "values"):
        return np.asarray(prediction.values, dtype=float)
    return np.asarray(prediction, dtype=float)


def test_stiefel_k2_response_geometry_is_fittable():
    x, z, frames = _clustered_stiefel_frames()
    cols = {f"f{i}": frames[:, i] for i in range(6)}
    df = pd.DataFrame({"x": x, "z": z, **cols})

    # The documented Stiefel response geometry must accept tightly clustered
    # frames. Today this raises a Fréchet-mean init error because the k>1
    # Stiefel logarithm is unimplemented.
    model = gamfit.fit(
        df,
        "f0 ~ s(x) + s(z)",
        response_geometry="stiefel(k=2)",
        response_columns=list(cols),
    )

    preds = _predict_matrix(model.predict(df.head(20)))
    assert preds.shape[1] == 6, f"expected 6 frame columns, got {preds.shape}"

    # Every predicted frame must lie on St(3, 2): YᵀY = I_2.
    worst = 0.0
    for row in preds:
        y = row.reshape(3, 2)
        worst = max(worst, float(np.max(np.abs(y.T @ y - np.eye(2)))))
    assert worst < 1e-8, f"predicted frames are not orthonormal: max|YᵀY - I| = {worst:.3e}"


def test_stiefel_k1_control_fits():
    """Control: St(n, 1) == sphere reduction fits fine, isolating the breakage
    to k >= 2 (so the bug is the missing Stiefel logarithm, not the response-
    geometry machinery as a whole)."""
    rng = np.random.default_rng(5)
    n = 400
    x = rng.uniform(-1.0, 1.0, n)
    z = rng.uniform(-1.0, 1.0, n)
    v = np.stack(
        [
            np.cos(1.2 * x) * np.cos(0.8 * z),
            np.sin(1.2 * x) * np.cos(0.8 * z),
            np.sin(0.8 * z),
        ],
        axis=1,
    )
    df = pd.DataFrame({"x": x, "z": z, "u0": v[:, 0], "u1": v[:, 1], "u2": v[:, 2]})
    model = gamfit.fit(
        df,
        "u0 ~ s(x) + s(z)",
        response_geometry="stiefel(k=1)",
        response_columns=["u0", "u1", "u2"],
    )
    preds = _predict_matrix(model.predict(df.head(10)))
    norms = np.linalg.norm(preds, axis=1)
    assert np.max(np.abs(norms - 1.0)) < 1e-8
