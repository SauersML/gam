use gam::ResourcePolicy;
use gam::inference::data::EncodedDataset;
use gam::inference::model::{ColumnKindTag, DataSchema, SchemaColumn};
use gam::terms::basis::{BSplineBasisSpec, BSplineIdentifiability, BSplineKnotSpec, OneDimensionalBoundary, PeriodicBSplineBasisSpec, build_bspline_basis_1d, build_periodic_bspline_basis_1d, cyclic_bspline_derivative_penalty_matrix, periodic_bspline_first_derivative_nd};
use gam::terms::smooth::{
    SmoothBasisSpec, SmoothTermSpec, TermCollectionSpec, build_term_collection_design,
};
use gam::terms::term_builder::build_termspec;
use ndarray::{Array2, Axis, array};

fn periodic_derivative_dense(
    u: ndarray::ArrayView1<'_, f64>,
    spec: &PeriodicBSplineBasisSpec,
) -> Array2<f64> {
    let t = u.to_owned().insert_axis(Axis(1));
    periodic_bspline_first_derivative_nd(
        t.view(),
        (spec.origin, spec.origin + spec.period),
        spec.degree,
        spec.num_basis,
    )
    .unwrap()
    .index_axis(Axis(2), 0)
    .to_owned()
}

#[test]
fn periodic_basis_wraps_partitions_unity_and_has_periodic_derivative() {
    let u = array![0.0, 0.07, 0.25, 0.61, 0.999_999, 1.0, 1.07, -0.93];
    let spec = PeriodicBSplineBasisSpec::new(3, 12, 1.0, 0.0, 2);
    let basis = build_periodic_bspline_basis_1d(u.view(), &spec).unwrap();
    for row in basis.rows() {
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "row sum {sum}");
        assert!(row.iter().all(|v| *v >= -1e-14));
    }

    let endpoints = array![0.0, 1.0, 2.0, -1.0];
    let endpoint_basis = build_periodic_bspline_basis_1d(endpoints.view(), &spec).unwrap();
    for i in 1..endpoint_basis.nrows() {
        for j in 0..endpoint_basis.ncols() {
            assert!((endpoint_basis[[0, j]] - endpoint_basis[[i, j]]).abs() < 1e-12);
        }
    }

    let endpoint_deriv = periodic_derivative_dense(endpoints.view(), &spec);
    for i in 1..endpoint_deriv.nrows() {
        for j in 0..endpoint_deriv.ncols() {
            assert!((endpoint_deriv[[0, j]] - endpoint_deriv[[i, j]]).abs() < 1e-12);
        }
    }
}

#[test]
fn cyclic_derivative_penalty_wraps_and_has_constant_nullspace() {
    let s = cyclic_bspline_derivative_penalty_matrix(3, 10, 1.0, 2).unwrap();
    assert_eq!(s.nrows(), 10);
    assert_eq!(s.ncols(), 10);

    let ones = Array2::from_elem((10, 1), 1.0);
    let penalized = s.dot(&ones);
    let scale = s.iter().fold(0.0_f64, |m, value| m.max(value.abs()));
    assert!(penalized.iter().all(|v| v.abs() < 1e-12 * scale));

    for i in 0..10 {
        for j in 0..10 {
            assert!((s[[i, j]] - s[[(i + 1) % 10, (j + 1) % 10]]).abs() < 1e-12 * scale);
            assert_eq!(s[[i, j]], s[[j, i]]);
        }
    }
}

#[test]
fn periodic_bspline_terms_build_with_cyclic_penalty_and_formula_alias() {
    let x = array![0.0, 0.125, 0.25, 0.5, 0.75, 1.0];
    let data = Array2::from_shape_vec((x.len(), 1), x.to_vec()).unwrap();
    let term = SmoothTermSpec {
        frozen_parametric_residualization: None,
        name: "periodic_u".to_string(),
        basis: SmoothBasisSpec::BSpline1D {
            feature_col: 0,
            spec: BSplineBasisSpec {
                degree: 3,
                penalty_order: 2,
                knotspec: BSplineKnotSpec::PeriodicUniform {
                    data_range: (0.0, 1.0),
                    num_basis: 10,
                },
                double_penalty: true,
                identifiability: BSplineIdentifiability::None,
                boundary: OneDimensionalBoundary::Open,
                boundary_conditions: Default::default(),
            },
        },
        shape: gam::terms::smooth::ShapeConstraint::None,
        joint_null_rotation: None,
    };
    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![term],
    };
    let design = build_term_collection_design(data.view(), &spec).unwrap();
    assert_eq!(design.smooth.terms.len(), 1);
    // A cyclic P-spline is a SINGLE-penalty smooth even under `double_penalty`
    // (#874, matching mgcv's `bs="cc"`): the cyclic difference penalty's only
    // null direction is the constant, which the periodic sum-to-zero
    // identifiability constraint removes wholesale. The null-space ("double")
    // penalty is the projector onto exactly that constant, so after the
    // constraint transform it is identically zero — an unidentified λ that
    // makes the outer REML objective flat and prevents convergence. The builder
    // therefore emits only the wiggliness penalty (see bspline_build.rs).
    assert_eq!(design.smooth.terms[0].active_penalties.len(), 1);

    let built = build_bspline_basis_1d(
        x.view(),
        &BSplineBasisSpec {
            degree: 3,
            penalty_order: 2,
            knotspec: BSplineKnotSpec::PeriodicUniform {
                data_range: (0.0, 1.0),
                num_basis: 10,
            },
            double_penalty: false,
            identifiability: BSplineIdentifiability::None,
            boundary: OneDimensionalBoundary::Open,
            boundary_conditions: Default::default(),
        },
    )
    .unwrap();
    match &built.metadata {
        gam::terms::basis::BasisMetadata::BSpline1D { periodic, .. } => {
            assert_eq!(*periodic, Some((0.0, 1.0, 10)));
        }
        other => panic!("unexpected metadata {other:?}"),
    }

    let ds = EncodedDataset {
        headers: vec!["y".to_string(), "u".to_string()],
        values: Array2::from_shape_vec(
            (x.len(), 2),
            x.iter().flat_map(|&v| [v.sin(), v]).collect(),
        )
        .unwrap(),
        schema: DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "y".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "u".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
            ],
        },
        column_kinds: vec![ColumnKindTag::Continuous, ColumnKindTag::Continuous],
    };
    // periodic smooths require explicit `period=` (cycle 27): silent
    // inference from data range is sample-dependent and rarely matches
    // user intent (e.g. uniform draws on [0, 1] give period < 1).
    let parsed =
        gam::inference::formula_dsl::parse_formula("y ~ s(u, type=periodic, k=9, period=1)")
            .unwrap();
    let cmap = ds.column_map();
    let mut notes = Vec::new();
    let terms = build_termspec(
        &parsed.terms,
        &ds,
        &cmap,
        &mut notes,
        &ResourcePolicy::default_library(),
    )
    .unwrap();
    match &terms.smooth_terms[0].basis {
        SmoothBasisSpec::BSpline1D { spec, .. } => match spec.knotspec {
            BSplineKnotSpec::PeriodicUniform { num_basis, .. } => assert_eq!(num_basis, 9),
            _ => panic!("formula alias did not create periodic knotspec"),
        },
        _ => panic!("formula alias did not create 1D periodic smooth"),
    }

    let cyclic = gam::inference::formula_dsl::parse_formula(
        "y ~ cyclic(u, k=9, period_start=0, period_end=1)",
    )
    .unwrap();
    let cyclic_terms = build_termspec(
        &cyclic.terms,
        &ds,
        &cmap,
        &mut notes,
        &ResourcePolicy::default_library(),
    )
    .unwrap();
    match &cyclic_terms.smooth_terms[0].basis {
        SmoothBasisSpec::BSpline1D { spec, .. } => match spec.knotspec {
            BSplineKnotSpec::PeriodicUniform {
                data_range,
                num_basis,
            } => {
                assert_eq!(num_basis, 9);
                assert_eq!(data_range, (0.0, 1.0));
            }
            _ => panic!("cyclic() did not create periodic knotspec"),
        },
        _ => panic!("cyclic() did not create 1D periodic smooth"),
    }
}

#[test]
fn cyclic_alias_default_basis_size_matches_periodic_s_smooth() {
    let n = 40usize;
    let values = Array2::from_shape_vec(
        (n, 2),
        (0..n)
            .flat_map(|i| {
                let u = i as f64 / n as f64;
                [(std::f64::consts::TAU * u).sin(), u]
            })
            .collect(),
    )
    .unwrap();
    let ds = EncodedDataset {
        headers: vec!["y".to_string(), "u".to_string()],
        values,
        schema: DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "y".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "u".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
            ],
        },
        column_kinds: vec![ColumnKindTag::Continuous, ColumnKindTag::Continuous],
    };
    let cmap = ds.column_map();
    let mut notes = Vec::new();
    let formulas = [
        "y ~ s(u, periodic=true, period_start=0, period_end=1)",
        "y ~ cyclic(u, period_start=0, period_end=1)",
    ];
    let mut basis_sizes = Vec::new();
    for formula in formulas {
        let parsed = gam::inference::formula_dsl::parse_formula(formula).unwrap();
        let terms = build_termspec(
            &parsed.terms,
            &ds,
            &cmap,
            &mut notes,
            &ResourcePolicy::default_library(),
        )
        .unwrap();
        match &terms.smooth_terms[0].basis {
            SmoothBasisSpec::BSpline1D { spec, .. } => match spec.knotspec {
                BSplineKnotSpec::PeriodicUniform { num_basis, .. } => {
                    basis_sizes.push(num_basis);
                }
                _ => panic!("{formula} did not create periodic knotspec"),
            },
            _ => panic!("{formula} did not create 1D periodic smooth"),
        }
    }
    assert_eq!(basis_sizes[0], basis_sizes[1]);
}

#[test]
fn cylinder_formula_builds_tensor_with_periodic_margin() {
    let two_pi = std::f64::consts::TAU;
    let data = EncodedDataset {
        headers: vec!["y".to_string(), "theta".to_string(), "h".to_string()],
        values: array![
            [0.0, 0.0, 0.25],
            [0.0, two_pi, 0.25],
            [0.0, two_pi / 2.0, 0.75],
        ],
        schema: DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "y".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "theta".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "h".to_string(),
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
    };
    let parsed = gam::inference::formula_dsl::parse_formula(
        "y ~ s(theta, h, periodic=[0], period=[2*pi, None], k=8)",
    )
    .unwrap();
    let cmap = data.column_map();
    let mut notes = Vec::new();
    let terms = build_termspec(
        &parsed.terms,
        &data,
        &cmap,
        &mut notes,
        &ResourcePolicy::default_library(),
    )
    .unwrap();
    match &terms.smooth_terms[0].basis {
        SmoothBasisSpec::TensorBSpline { spec, .. } => {
            assert!(matches!(
                spec.marginalspecs[0].knotspec,
                BSplineKnotSpec::PeriodicUniform { .. }
            ));
            assert!(matches!(
                spec.marginalspecs[1].knotspec,
                BSplineKnotSpec::Generate { .. }
            ));
        }
        _ => panic!("mixed periodic s(theta,h) should build a tensor smooth"),
    }
    let design = build_term_collection_design(data.values.view(), &terms).unwrap();
    let dense = design.smooth.term_designs[0].to_dense();
    for col in 0..dense.ncols() {
        assert!(
            (dense[[0, col]] - dense[[1, col]]).abs() < 1e-12,
            "periodic tensor margin differs across seam at column {col}"
        );
    }
}
