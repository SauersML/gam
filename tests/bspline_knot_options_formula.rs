//! Formula-DSL coverage for the two B-spline knot capabilities exposed through
//! `s(x, ...)` and tensor margins:
//!
//!   1. `knots=[k0, k1, ...]` — explicit *internal* knot positions (mgcv
//!      `knots=` vector semantics). These must land verbatim in the clamped
//!      knot vector of `BSplineKnotSpec::Provided`.
//!   2. `knot_placement="quantile"` — route the automatically generated knots
//!      through `BSplineKnotSpec::Automatic { placement: Quantile }`.
//!
//! Back-compat: a *scalar* `knots=<int>` keeps its historical meaning as an
//! internal-knot count and still produces `BSplineKnotSpec::Generate`.

use gam::ResourcePolicy;
use gam::basis::{BSplineKnotPlacement, BSplineKnotSpec};
use gam::inference::data::EncodedDataset;
use gam::inference::formula_dsl::parse_formula;
use gam::inference::model::{ColumnKindTag, DataSchema, SchemaColumn};
use gam::smooth::SmoothBasisSpec;
use gam::terms::term_builder::build_termspec;
use ndarray::Array2;

/// Build a dataset with two continuous covariates `x` and `z` spanning [0, 1]
/// with enough distinct values to support quantile knot placement.
fn dataset() -> EncodedDataset {
    let n = 40usize;
    let mut values = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        values[(i, 0)] = (t * std::f64::consts::TAU).sin(); // y
        values[(i, 1)] = t; // x in [0, 1]
        values[(i, 2)] = t * t; // z in [0, 1], skewed
    }
    EncodedDataset {
        headers: vec!["y".into(), "x".into(), "z".into()],
        values,
        schema: DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "y".into(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "x".into(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "z".into(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
            ],
        },
        column_kinds: vec![
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
        ],
    }
}

fn build(formula: &str) -> gam::smooth::TermCollectionSpec {
    let parsed = parse_formula(formula).expect("parse formula");
    let ds = dataset();
    let mut notes = Vec::new();
    build_termspec(
        &parsed.terms,
        &ds,
        &ds.column_map(),
        &mut notes,
        &ResourcePolicy::default_library(),
    )
    .expect("termspec")
}

fn bspline_spec(collection: &gam::smooth::TermCollectionSpec) -> &gam::basis::BSplineBasisSpec {
    match &collection.smooth_terms[0].basis {
        SmoothBasisSpec::BSpline1D { spec, .. } => spec,
        other => panic!("expected BSpline1D smooth, got {other:?}"),
    }
}

#[test]
fn explicit_knot_list_populates_provided_knotspec_with_those_positions() {
    let spec = build("y ~ s(x, bs=ps, knots=[0.2, 0.5, 0.8])");
    let bspline = bspline_spec(&spec);
    let degree = bspline.degree;
    match &bspline.knotspec {
        BSplineKnotSpec::Provided(knots) => {
            // Clamped full vector = degree+1 left boundary repeats, then the
            // user's interior positions, then degree+1 right boundary repeats.
            let n = knots.len();
            assert_eq!(
                n,
                3 + 2 * (degree + 1),
                "Provided vector should wrap 3 interior knots in clamped boundaries"
            );
            // Interior block carries the supplied positions verbatim and sorted.
            let interior: Vec<f64> = knots.iter().copied().skip(degree + 1).take(3).collect();
            let expected = [0.2_f64, 0.5, 0.8];
            for (got, want) in interior.iter().zip(expected.iter()) {
                assert!(
                    (got - want).abs() < 1e-12,
                    "interior knot {got} should equal supplied position {want}"
                );
            }
            // Boundary repeats clamp to the observed data range [0, 1].
            assert!(
                knots
                    .iter()
                    .take(degree + 1)
                    .all(|&k| (k - 0.0).abs() < 1e-12)
            );
            assert!(
                knots
                    .iter()
                    .rev()
                    .take(degree + 1)
                    .all(|&k| (k - 1.0).abs() < 1e-12)
            );
        }
        other => panic!("expected Provided knotspec from knots=[...], got {other:?}"),
    }
}

#[test]
fn explicit_knot_list_accepts_r_vector_syntax() {
    let spec = build("y ~ s(x, bs=ps, knots=c(0.25, 0.75))");
    match &bspline_spec(&spec).knotspec {
        BSplineKnotSpec::Provided(knots) => {
            let degree = bspline_spec(&spec).degree;
            let interior: Vec<f64> = knots.iter().copied().skip(degree + 1).take(2).collect();
            assert!((interior[0] - 0.25).abs() < 1e-12);
            assert!((interior[1] - 0.75).abs() < 1e-12);
        }
        other => panic!("expected Provided from c(...) syntax, got {other:?}"),
    }
}

#[test]
fn quantile_placement_routes_to_automatic_quantile_knotspec() {
    let spec = build("y ~ s(z, bs=ps, k=8, knot_placement=\"quantile\")");
    match &bspline_spec(&spec).knotspec {
        BSplineKnotSpec::Automatic { placement, .. } => {
            assert_eq!(*placement, BSplineKnotPlacement::Quantile);
        }
        other => panic!("expected Automatic{{Quantile}}, got {other:?}"),
    }
}

#[test]
fn uniform_placement_default_and_explicit_uses_generate() {
    // Default (no knot_placement) is uniform Generate.
    match &bspline_spec(&build("y ~ s(x, bs=ps, k=8)")).knotspec {
        BSplineKnotSpec::Generate { .. } => {}
        other => panic!("default placement should be uniform Generate, got {other:?}"),
    }
    // Explicit "uniform" is identical.
    match &bspline_spec(&build("y ~ s(x, bs=ps, k=8, knot_placement=uniform)")).knotspec {
        BSplineKnotSpec::Generate { .. } => {}
        other => panic!("knot_placement=uniform should be Generate, got {other:?}"),
    }
}

#[test]
fn scalar_knots_count_still_means_internal_knot_count() {
    // Back-compat: knots=<int> is a COUNT and yields Generate with that many
    // internal knots — unchanged from before this feature.
    match &bspline_spec(&build("y ~ s(x, bs=ps, knots=8)")).knotspec {
        BSplineKnotSpec::Generate {
            num_internal_knots, ..
        } => assert_eq!(*num_internal_knots, 8),
        other => panic!("scalar knots= should remain a Generate count, got {other:?}"),
    }
}

#[test]
fn tensor_margins_honor_quantile_placement() {
    let spec = build("y ~ te(x, z, k=6, knot_placement=quantile)");
    match &spec.smooth_terms[0].basis {
        SmoothBasisSpec::TensorBSpline { spec, .. } => {
            assert!(!spec.marginalspecs.is_empty());
            for margin in &spec.marginalspecs {
                match &margin.knotspec {
                    BSplineKnotSpec::Automatic { placement, .. } => {
                        assert_eq!(*placement, BSplineKnotPlacement::Quantile);
                    }
                    other => {
                        panic!("tensor margin should use Automatic{{Quantile}}, got {other:?}")
                    }
                }
            }
        }
        other => panic!("expected TensorBSpline, got {other:?}"),
    }
}

#[test]
fn explicit_knot_list_conflicts_with_k() {
    let parsed = parse_formula("y ~ s(x, bs=ps, knots=[0.5], k=7)").expect("parse");
    let ds = dataset();
    let mut notes = Vec::new();
    let err = build_termspec(
        &parsed.terms,
        &ds,
        &ds.column_map(),
        &mut notes,
        &ResourcePolicy::default_library(),
    )
    .expect_err("knots=[...] together with k= must be rejected");
    let err = err.to_string();
    assert!(
        err.contains("knots") && err.contains('k'),
        "error should explain the knots/k conflict, got: {err}"
    );
}

#[test]
fn out_of_range_explicit_knot_is_rejected() {
    let parsed = parse_formula("y ~ s(x, bs=ps, knots=[2.0])").expect("parse");
    let ds = dataset();
    let mut notes = Vec::new();
    let err = build_termspec(
        &parsed.terms,
        &ds,
        &ds.column_map(),
        &mut notes,
        &ResourcePolicy::default_library(),
    )
    .expect_err("interior knot outside the data range must be rejected");
    let err = err.to_string();
    assert!(
        err.contains("strictly inside") || err.contains("data range"),
        "error should explain the out-of-range knot, got: {err}"
    );
}
