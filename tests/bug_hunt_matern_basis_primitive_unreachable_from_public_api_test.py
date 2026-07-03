"""Bug hunt: the Matérn kernel basis primitive is unreachable from the public
``gamfit`` namespace, even though it is a documented, working peer of the
exported ``gamfit.duchon_basis``.

`gamfit/_api.py` defines ``matern_basis(points, centers, *, length_scale, nu,
aniso_log_scales)`` with a full public-style docstring that explicitly states it
is "exposed through the same PyFFI surface as :func:`duchon_basis`". The
underlying FFI (`rust_module().matern_basis`) works and returns the correct
Matérn kernel. But `gamfit/__init__.py` never imports the symbol: its
`from ._api import (...)` block lists the whole basis / penalty primitive family
— ``bspline_basis``, ``bspline_basis_derivative``, ``duchon_basis``,
``duchon_function_norm_penalty``, ``periodic_spline_curve_basis``,
``sphere_basis``, ``smoothness_penalty`` — but omits ``matern_basis`` (and its
jet-peer ``sphere_basis_jet``). Because ``__all__`` is auto-derived from what is
importable, the omission also drops it from star-imports and from the
``docs/api-reference.md`` mkdocstrings surface (which documents
``bspline_basis`` / ``duchon_basis`` / ``sphere_basis`` but has no
``::: gamfit.matern_basis`` entry).

Net effect: a user following the docstring's "peer of ``duchon_basis``" framing
finds ``gamfit.duchon_basis`` but ``gamfit.matern_basis`` raises
``AttributeError`` — the Matérn radial primitive advertised in the README
("radial smooths ... ``matern``") has no direct Python entry point, unlike every
other exposed basis primitive.

This is the same defect class as the accepted
``bug_hunt_expectile_family_unreachable_from_public_interfaces`` /
``bug_hunt_matern_periodic_option_rejected_despite_builder_support`` hunts: a
real capability that exists in the engine but cannot be reached from the public
interface.

Root cause: a one-line omission in the ``from ._api import (...)`` block of
`gamfit/__init__.py`. The fix is to export ``matern_basis`` there next to
``duchon_basis`` (and, for full parity, ``sphere_basis_jet`` next to
``sphere_basis``).

This test fails today with ``AttributeError`` (the symbol is not on the public
namespace). Once the export is added it passes: the assertions below check both
that ``gamfit.matern_basis`` is reachable AND that the public entry point
computes the correct closed-form Matérn kernel — so the test pins a real
primitive, not merely the presence of a name.

NOTE: this test can only run once the workspace builds — the `gamfit` wheel does
not compile at the current `main` HEAD because of the gam-sae Arrow-Schur build
break (Related: #2119). Fixing that build break is a prerequisite for this test
to execute and reach its own (export-gap) assertion.
"""

import numpy as np
import pytest

import gamfit


def test_matern_basis_is_reachable_from_public_namespace():
    # Its exported siblings are all reachable; the Matérn primitive must be too.
    for sibling in ("duchon_basis", "bspline_basis", "sphere_basis"):
        assert hasattr(gamfit, sibling), f"expected sibling {sibling} to be public"
    assert hasattr(gamfit, "matern_basis"), (
        "gamfit.matern_basis is not reachable from the public namespace, even "
        "though its docstring frames it as a peer of the exported "
        "gamfit.duchon_basis and the underlying FFI works"
    )


def test_public_matern_basis_computes_correct_kernel():
    matern_basis = getattr(gamfit, "matern_basis", None)
    if matern_basis is None:
        pytest.fail("gamfit.matern_basis is unreachable (public export missing)")

    # 1-D Matérn kernel value vs distance, against the standard closed forms
    # (with the sqrt(2*nu) scaling used by the engine, verified numerically).
    d = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
    pts = d.reshape(-1, 1)
    ctr = np.array([[0.0]])

    k32 = np.asarray(matern_basis(pts, ctr, length_scale=1.0, nu="3/2"))[:, 0]
    expect32 = (1.0 + np.sqrt(3.0) * d) * np.exp(-np.sqrt(3.0) * d)
    assert np.allclose(k32, expect32, atol=1e-9), (
        f"Matérn 3/2 kernel wrong: got {k32}, expected {expect32}"
    )

    k52 = np.asarray(matern_basis(pts, ctr, length_scale=1.0, nu="5/2"))[:, 0]
    r5 = np.sqrt(5.0) * d
    expect52 = (1.0 + r5 + r5 * r5 / 3.0) * np.exp(-r5)
    assert np.allclose(k52, expect52, atol=1e-9), (
        f"Matérn 5/2 kernel wrong: got {k52}, expected {expect52}"
    )

    # Basic kernel invariants through the public surface.
    x = np.linspace(-2.0, 2.0, 7).reshape(-1, 1)
    gram = np.asarray(matern_basis(x, x, length_scale=0.8, nu="3/2"))
    assert gram.shape == (7, 7)
    assert np.allclose(gram, gram.T, atol=1e-12), "Matérn Gram must be symmetric"
    assert np.allclose(np.diag(gram), 1.0, atol=1e-12), "K(x, x) must be 1"
