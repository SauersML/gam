"""The public selector is a real, canonically named multi-fit orchestrator."""

import pytest
import gamfit._select_topology as select_topology_module
from gamfit import topology
from gamfit._binding import rust_module
from gamfit.topology import TopologyAutoSelector
from gamfit._select_topology import _Candidate, _candidate_formula, _normalize_topology_name


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


@pytest.mark.parametrize(
    "formula",
    ["y ~ te(t, u, type=AUTO)", "y ~ s(t, type=\nAUTO)", "y ~ s(t) + f(type=AUTO)"],
)
def test_auto_branch_reads_the_assembler_scan_2899(formula: str) -> None:
    # Each formula matched the deleted Python regex, so the selector took the
    # substitution branch and the Rust assembler then refused every candidate.
    # One scan decides now: no `s(..., type=AUTO)` term, no substitution.
    assert not hasattr(select_topology_module, "_AUTO_RE")
    candidate = _Candidate("circle", topology.Circle(name="theta"))
    auto = rust_module().has_auto_smooth_term(formula)
    assert auto is False
    assert _candidate_formula(formula, auto, candidate) == formula


def test_auto_branch_substitutes_a_smooth_auto_term_2899() -> None:
    formula = "y ~ s(t, type=AUTO)"
    candidate = _Candidate("circle", topology.Circle(name="theta"))
    auto = rust_module().has_auto_smooth_term(formula)
    assert auto is True
    assembled = _candidate_formula(formula, auto, candidate)
    assert assembled.startswith("y ~ s(t, ")
    assert "AUTO" not in assembled
