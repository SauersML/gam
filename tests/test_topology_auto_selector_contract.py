"""The public selector is a real multi-fit orchestrator, not a fit descriptor."""

from gamfit import TopologyAutoSelector


def test_topology_auto_selector_exposes_only_the_real_fit_path() -> None:
    selector = TopologyAutoSelector()

    assert callable(selector.fit)
    assert not hasattr(selector, "to_rust_descriptor")
    assert not hasattr(selector, "_to_rust_payload")
