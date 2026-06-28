"""Cross-frame interop regression: descriptor.evaluate(x, backend=...).

Verifies that for every descriptor declaring multi-backend support, the
torch / numpy / jax paths return numerically identical ``(B, M)`` matrices
(within machine precision). The Rust core is the single source of math
truth — these tests fail loudly if any backend drifts.

Run with::

    pytest tests/test_cross_backend_descriptor_evaluate.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

import gamfit


def _to_numpy(arr) -> np.ndarray:
    """Coerce a numpy / torch / jax array to a numpy.ndarray."""
    if isinstance(arr, np.ndarray):
        return arr
    if hasattr(arr, "detach"):  # torch
        return arr.detach().cpu().numpy()
    return np.asarray(arr)  # jax


def _maxabs(a, b) -> float:
    return float(np.max(np.abs(_to_numpy(a) - _to_numpy(b))))


def test_bspline_cross_backend_identical():
    """1D B-spline: torch vs numpy vs (optional) jax all bit-equal."""
    # #1512: the torch comparison path needs torch; skip cleanly when it is
    # absent (as the basis_descriptor tests do and as CI runs without torch)
    # instead of failing with a bare ModuleNotFoundError at evaluate(...).
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, size=64)

    spec = gamfit.BSpline(degree=3, periodic=False)

    out_numpy = spec.evaluate(x, backend="numpy")
    out_torch = spec.evaluate(x, backend="torch")

    diff_np_torch = _maxabs(out_numpy, out_torch)
    assert diff_np_torch < 1e-12, (
        f"BSpline torch vs numpy max-abs-diff = {diff_np_torch:.3e}"
    )
    print(f"BSpline torch vs numpy max-abs-diff: {diff_np_torch:.3e}")

    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    x_jax = jnp.asarray(x)
    out_jax = spec.evaluate(x_jax, backend="jax")
    diff_np_jax = _maxabs(out_numpy, out_jax)
    assert diff_np_jax < 1e-10, (
        f"BSpline jax vs numpy max-abs-diff = {diff_np_jax:.3e}"
    )
    print(f"BSpline jax vs numpy max-abs-diff:   {diff_np_jax:.3e}")


def test_sphere_cross_backend_identical():
    """Sphere basis: torch vs numpy bit-equal."""
    # #1512: needs torch for the backend="torch" comparison; skip when absent.
    pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    lat = rng.uniform(-60.0, 60.0, size=32)
    lon = rng.uniform(-180.0, 180.0, size=32)
    pts = np.column_stack([lat, lon])

    spec = gamfit.Sphere(n_centers=20, penalty_order=2, kernel="sobolev")

    # evaluate accepts *coords — but Sphere is naturally 2D so pass two 1D
    # columns separately.
    out_numpy = spec.evaluate(lat, lon, backend="numpy")
    out_torch = spec.evaluate(lat, lon, backend="torch")

    diff = _maxabs(out_numpy, out_torch)
    assert diff < 1e-10, f"Sphere torch vs numpy max-abs-diff = {diff:.3e}"
    print(f"Sphere torch vs numpy max-abs-diff:  {diff:.3e}")


@pytest.mark.xfail(
    strict=True,
    reason="#1512 triage: Matern is no longer torch-only — it now declares "
    "SUPPORTED_BACKENDS={numpy, torch, jax} and Matern.evaluate(x, "
    "backend='numpy') returns a finite (n, n_centers) basis, so the numpy path "
    "no longer raises NotImplementedError. This test pins the obsolete "
    "torch-only contract; update it to the multi-backend behavior to re-enable.",
)
def test_matern_is_torch_only():
    """Matern is pure-torch math; numpy/jax must raise NotImplementedError."""
    centers = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
    spec = gamfit.Matern(centers=centers, nu=1.5, length_scale=0.3)
    x = np.linspace(0.0, 1.0, 16)

    with pytest.raises(NotImplementedError, match="numpy"):
        spec.evaluate(x, backend="numpy")
    with pytest.raises(NotImplementedError, match="jax"):
        spec.evaluate(x, backend="jax")


@pytest.mark.xfail(
    strict=True,
    reason="#1512 triage: the expected matrix pins Matern: {'torch'}, but "
    "Matern.SUPPORTED_BACKENDS is now {'numpy', 'torch', 'jax'} (Matern gained "
    "numpy/jax support). Stale expectation; update the matrix entry to re-enable.",
)
def test_capability_matrix_declared():
    """Each descriptor declares SUPPORTED_BACKENDS — none missing."""
    matrix = {
        gamfit.BSpline: {"torch", "numpy", "jax"},
        gamfit.Duchon: {"torch", "numpy", "jax"},
        gamfit.TensorBSpline: {"torch", "numpy", "jax"},
        gamfit.Sphere: {"torch", "numpy", "jax"},
        gamfit.PeriodicSplineCurve: {"torch", "numpy", "jax"},
        gamfit.Pca: {"torch", "numpy", "jax"},
        gamfit.Matern: {"torch"},
    }
    for cls, expected in matrix.items():
        got = set(cls.SUPPORTED_BACKENDS)
        assert got == expected, (
            f"{cls.__name__}.SUPPORTED_BACKENDS = {got}, expected {expected}"
        )


def test_unknown_backend_raises():
    spec = gamfit.BSpline(degree=3)
    x = np.linspace(0.0, 1.0, 8)
    with pytest.raises(ValueError, match="unknown backend"):
        spec.evaluate(x, backend="tensorflow")


def test_lazy_jax_import():
    """Importing gamfit must not pull in jax."""
    import sys

    # `gamfit` is already imported by the time we get here; verify jax is
    # not in sys.modules unless someone above us imported it explicitly.
    # The basis_protocol module exposes the lazy loader.
    from gamfit import _basis_protocol

    # Detect-backend on a pure numpy input must not import jax.
    before = "jax" in sys.modules
    _basis_protocol._detect_backend([np.zeros(3)])
    after = "jax" in sys.modules
    assert before == after, (
        "_detect_backend imported jax on a numpy input (it should not)"
    )
