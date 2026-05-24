use gam::inference::formula_dsl::parse_formula;
use gam::terms::basis::{
    BSplineBasisSpec, BSplineIdentifiability, BSplineKnotSpec, CenterStrategy, MaternBasisSpec,
    MaternIdentifiability, MaternNu,
};
use gam::terms::smooth::{
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, TensorBSplineIdentifiability,
    TensorBSplineSpec, TermCollectionSpec, build_term_collection_design,
};
use ndarray::{Array2, array};

#[test]
fn formula_parser_accepts_cylinder_periodic_options() {
    let s = parse_formula("y ~ s(theta, h, periodic=[0], period=[2*pi, None])")
        .expect("parse periodic radial smooth");
    assert_eq!(s.terms.len(), 1);

    let te = parse_formula("y ~ te(theta, h, bc=['periodic', 'natural'], period=[2*pi, None])")
        .expect("parse periodic tensor smooth");
    assert_eq!(te.terms.len(), 1);
}

#[test]
fn tensor_periodic_margin_is_exactly_cyclic_at_period_boundary() {
    let two_pi = 2.0 * std::f64::consts::PI;
    let data = array![[0.0, 0.25], [two_pi, 0.25], [std::f64::consts::PI, 0.75]];
    let marginal = |range| BSplineBasisSpec {
        degree: 3,
        penalty_order: 2,
        knotspec: BSplineKnotSpec::Generate {
            data_range: range,
            num_internal_knots: 4,
        },
        double_penalty: false,
        identifiability: BSplineIdentifiability::None,
    };
    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            name: "te(theta,h)".to_string(),
            basis: SmoothBasisSpec::TensorBSpline {
                feature_cols: vec![0, 1],
                spec: TensorBSplineSpec {
                    marginalspecs: vec![marginal((0.0, two_pi)), marginal((0.0, 1.0))],
                    periods: vec![Some(two_pi), None],
                    double_penalty: false,
                    identifiability: TensorBSplineIdentifiability::None,
                },
            },
            shape: ShapeConstraint::None,
        }],
    };

    let design = build_term_collection_design(data.view(), &spec).expect("tensor periodic design");
    let dense = design.smooth.term_designs[0].to_dense();
    assert_eq!(dense.nrows(), 3);
    for col in 0..dense.ncols() {
        assert!(
            (dense[[0, col]] - dense[[1, col]]).abs() < 1e-12,
            "tensor cyclic seam mismatch at col {col}: {} vs {}",
            dense[[0, col]],
            dense[[1, col]]
        );
    }
}

#[test]
fn radial_periodic_smooth_uses_ghost_centers_but_freezes_original_centers() {
    let two_pi = 2.0 * std::f64::consts::PI;
    let data = array![[0.0, 0.0], [two_pi / 2.0, 0.5], [two_pi, 1.0]];
    let centers: Array2<f64> = data.clone();
    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            name: "s(theta,h)".to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: vec![0, 1],
                spec: MaternBasisSpec {
                    center_strategy: CenterStrategy::UserProvided(centers.clone()),
                    periodic: Some(vec![Some(two_pi), None]),
                    length_scale: 1.0,
                    nu: MaternNu::FiveHalves,
                    include_intercept: false,
                    double_penalty: true,
                    identifiability: MaternIdentifiability::None,
                    aniso_log_scales: None,
                },
                input_scales: None,
            },
            shape: ShapeConstraint::None,
        }],
    };

    let design = build_term_collection_design(data.view(), &spec).expect("radial periodic design");
    let dense = design.smooth.term_designs[0].to_dense();
    assert_eq!(dense.nrows(), 3);
    assert_eq!(
        dense.ncols(),
        centers.nrows() * 3,
        "one periodic axis should add ±period ghosts"
    );

    match &design.smooth.terms[0].metadata {
        gam::terms::basis::BasisMetadata::Matern {
            centers: frozen,
            periodic,
            ..
        } => {
            assert_eq!(
                frozen.nrows(),
                centers.nrows(),
                "metadata stores unexpanded centers"
            );
            assert_eq!(periodic.as_ref().unwrap()[0], Some(two_pi));
        }
        other => panic!("expected Matern metadata, got {other:?}"),
    }
}
