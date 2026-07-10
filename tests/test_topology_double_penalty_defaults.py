import pytest

import gamfit.topology as topology
from gamfit._select_topology import _Candidate, _candidate_to_rust_payload


FACTORIES = [
    topology.Circle,
    topology.Cylinder,
    topology.Torus,
    topology.Sphere,
    lambda **kwargs: topology.EuclideanPatch(d=1, **kwargs),
]


@pytest.mark.parametrize("factory", FACTORIES)
def test_topology_descriptors_defer_to_the_rust_null_recovery_default(factory):
    spec = factory()

    assert spec.double_penalty is None
    assert "double_penalty" not in spec.to_rust_descriptor()
    assert _candidate_to_rust_payload(_Candidate("candidate", spec))["double_penalty"] is None


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("enabled", [False, True])
def test_topology_descriptors_preserve_explicit_null_recovery_choice(factory, enabled):
    spec = factory(double_penalty=enabled)

    assert spec.double_penalty is enabled
    assert spec.to_rust_descriptor()["double_penalty"] is enabled
    assert (
        _candidate_to_rust_payload(_Candidate("candidate", spec))["double_penalty"]
        is enabled
    )
