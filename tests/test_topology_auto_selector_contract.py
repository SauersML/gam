"""The public selector is a real, canonically named multi-fit orchestrator."""

import pytest
from gamfit import TopologyAutoSelector
from gamfit._select_topology import _normalize_topology_name


def test_topology_auto_selector_exposes_only_the_real_fit_path() -> None:
    selector = TopologyAutoSelector()

    assert callable(selector.fit)
    assert not hasattr(selector, "to_rust_descriptor")
    assert not hasattr(selector, "_to_rust_payload")


@pytest.mark.parametrize(
    "name", ["euclidean", "circle", "sphere", "torus", "cylinder"]
)
def test_topology_auto_selector_accepts_exact_canonical_names(name: str) -> None:
    assert _normalize_topology_name(name) == name


@pytest.mark.parametrize(
    "alias",
    [
        " circle",
        "circle ",
        "Circle",
        "flat",
        "euclideanpatch",
        "euclidean_patch",
        "periodic",
        "s1",
        "s2",
    ],
)
def test_topology_auto_selector_rejects_noncanonical_aliases(alias: str) -> None:
    with pytest.raises(ValueError, match="exact canonical name"):
        _normalize_topology_name(alias)
