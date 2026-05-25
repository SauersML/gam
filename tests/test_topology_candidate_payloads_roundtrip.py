import json

import gamfit.topology as topology
from gamfit._select_topology import _Candidate, _candidate_to_rust_payload


def test_topology_candidate_payloads_are_json_serializable_and_have_required_kind_fields():
    candidates = [
        _Candidate("circle", topology.Circle(name="theta")),
        _Candidate("cylinder", topology.Cylinder(name="cyl")),
        _Candidate("torus", topology.Torus(name="tor")),
        _Candidate("sphere", topology.Sphere(name="omega")),
        _Candidate("euclidean", topology.EuclideanPatch(d=2, name="x", centers=[[0.0, 0.0], [1.0, 1.0]])),
    ]

    payloads = [_candidate_to_rust_payload(candidate) for candidate in candidates]
    kinds = {payload["kind"] for payload in payloads}

    for payload in payloads:
        json.dumps(payload)

    assert kinds == {"periodic_spline_curve", "tensor", "sphere", "duchon"}, "Each topology spec must produce a valid Rust TopologyCandidate JSON payload with an expected kind tag."
