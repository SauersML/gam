"""Guard difference-smooth scientific orchestration against PyFFI drift."""

from pathlib import Path


def test_difference_smooth_binding_is_marshalling_only() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "crates/gam-pyffi/src/manifold/manifold_and_posterior_ffi.rs"
    ).read_text(encoding="utf-8")
    start = source.index("fn difference_smooth_json_impl(")
    end = source.index("\nfn json_f64_vec", start)
    binding = source[start:end]

    assert "difference_smooth_report(" in binding
    assert "effect_report(" not in binding
    assert "let mut contrast" not in binding
    assert "subtract_design_ranges" not in binding
    assert "zero_design_ranges" not in binding
    assert "for (level_1, level_2)" not in binding


def test_typed_core_owns_schema_design_and_simulation_policy() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "crates/gam-inference/src/difference_smooth.rs"
    ).read_text(encoding="utf-8")
    for required in (
        "pub struct DifferenceSmoothRequest",
        "pub fn difference_smooth_report",
        "let band_options",
        "let mut contrast = &left - &right",
        "random_effect_ranges",
        "complete_template",
    ):
        assert required in source
