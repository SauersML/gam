"""Bug hunt: the compositional primitives ``gamfit.clr`` / ``alr`` / ``closure``
fail with an opaque ``TypeError`` when handed a single composition as a 1-D
array — the most natural way to call them.

These are public, NumPy-facing helpers (``"Centered log-ratio coordinates for
positive compositions"`` etc.). The Python wrappers do
``_ffi("response_geometry_clr", np.asarray(values, dtype=float))`` with no
reshape, and the Rust ``#[pyfunction]`` signatures take ``PyReadonlyArray2``
(2-D only — see ``crates/gam-pyffi/src/latent/reml_latent_fit_ffi.rs``
``response_geometry_clr`` / ``response_geometry_alr`` /
``response_geometry_closure``, ~lines 6344-6378). When a user passes a single
composition as a 1-D vector — ``gamfit.clr([0.2, 0.3, 0.5])`` — the pyo3/numpy
downcast to a 2-D array fails and surfaces as

    TypeError: 'ndarray' object is not an instance of 'ndarray'

which is uninformative and gives no hint that a 2-D ``(rows, components)`` array
is required.

The mathematics of clr/alr/closure is defined on a single composition, and the
rest of the NumPy-facing surface accepts a 1-D vector of points
(``gamfit.bspline_basis``, ``gamfit.sphere_basis``, ...). The natural,
useful behaviour is to accept a 1-D composition and return its 1-D coordinates,
consistent with the 2-D batch row.

Observed: ``clr([0.2,0.3,0.5])`` raises ``TypeError: 'ndarray' object is not an
instance of 'ndarray'`` (same for ``alr`` and ``closure``), while the identical
composition wrapped as a ``(1, 3)`` 2-D array works and returns the correct
coordinates.

Expected: a single 1-D composition is accepted and yields the same coordinates
as the corresponding row of the 2-D batch call.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit


_COMPOSITION = [0.2, 0.3, 0.5]


@pytest.mark.parametrize("fn_name", ["clr", "alr", "closure"])
def test_compositional_primitive_accepts_1d_composition(fn_name: str) -> None:
    fn = getattr(gamfit, fn_name)

    # Batch (2-D) call is the known-good reference.
    batch = np.asarray(fn(np.array([_COMPOSITION], dtype=float)))
    assert batch.shape[0] == 1
    expected_row = batch[0]

    # Single composition as a 1-D vector — the natural call. This currently
    # raises an opaque ``TypeError: 'ndarray' object is not an instance of
    # 'ndarray'`` because the FFI only accepts 2-D input.
    single = np.asarray(fn(np.array(_COMPOSITION, dtype=float)))

    flat = np.asarray(single, dtype=float).ravel()
    assert flat.shape == expected_row.shape, (
        f"gamfit.{fn_name} on a 1-D composition returned shape "
        f"{single.shape}, incompatible with the batch row {expected_row.shape}"
    )
    assert np.allclose(flat, expected_row, atol=1e-12), (
        f"gamfit.{fn_name} 1-D result {flat} disagrees with the 2-D batch row "
        f"{expected_row}"
    )
