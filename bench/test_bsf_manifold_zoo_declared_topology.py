from __future__ import annotations

import pytest

from bench.manifold_zoo_geometry import ZOO_ORDER, declared_atom_spec


def test_declared_atom_spec_covers_every_exact_zoo_object() -> None:
    bases, dims = declared_atom_spec(list(ZOO_ORDER), len(ZOO_ORDER))
    assert bases == [
        "euclidean",
        "periodic",
        "duchon",
        "sphere",
        "torus",
        "mobius",
        "duchon",
        "duchon",
    ]
    assert dims == [1, 1, 2, 2, 2, 2, 2, 1]


def test_declared_atom_spec_refuses_non_joint_one_to_one_shape() -> None:
    with pytest.raises(ValueError, match="one fitted atom per planted factor"):
        declared_atom_spec(list(ZOO_ORDER), len(ZOO_ORDER) + 1)
