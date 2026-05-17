use gam::inference::data::EncodedDataset;
use gam::inference::model::{ColumnKindTag, DataSchema, SchemaColumn};
use gam::resource::ResourcePolicy;
use gam::terms::basis::{
    BSplineBasisSpec, BSplineIdentifiability, BSplineKnotSpec, PeriodicSplineCurveSpec,
    build_bspline_basis_1d, create_cyclic_difference_penalty_matrix,
    create_periodic_bspline_basis_dense, create_periodic_bspline_derivative_dense,
    evaluate_periodic_spline_curve, fit_periodic_spline_curve,
};
use gam::terms::smooth::{
    SmoothBasisSpec, SmoothTermSpec, TermCollectionSpec, build_term_collection_design,
};
use gam::terms::term_builder::build_termspec;
use ndarray::{Array1, Array2, array};

fn max_abs(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn periodic_basis_wraps_partitions_unity_and_has_periodic_derivative() {
    let u = array![0.0, 0.07, 0.25, 0.61, 0.999_999, 1.0, 1.07, -0.93];
    let basis = create_periodic_bspline_basis_dense(u.view(), (0.0, 1.0), 3, 12).unwrap();
    for row in basis.rows() {
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "row sum {sum}");
        assert!(row.iter().all(|v| *v >= -1e-14));
    }

    let endpoints = array![0.0, 1.0, 2.0, -1.0];
    let endpoint_basis =
        create_periodic_bspline_basis_dense(endpoints.view(), (0.0, 1.0), 3, 12).unwrap();
    for i in 1..endpoint_basis.nrows() {
        for j in 0..endpoint_basis.ncols() {
            assert!((endpoint_basis[[0, j]] - endpoint_basis[[i, j]]).abs() < 1e-12);
        }
    }

    let endpoint_deriv =
        create_periodic_bspline_derivative_dense(endpoints.view(), (0.0, 1.0), 3, 12).unwrap();
    for i in 1..endpoint_deriv.nrows() {
        for j in 0..endpoint_deriv.ncols() {
            assert!((endpoint_deriv[[0, j]] - endpoint_deriv[[i, j]]).abs() < 1e-12);
        }
    }
}

#[test]
fn cyclic_difference_penalty_wraps_and_has_constant_nullspace() {
    let s = create_cyclic_difference_penalty_matrix(10, 2).unwrap();
    assert_eq!(s.nrows(), 10);
    assert_eq!(s.ncols(), 10);

    let ones = Array2::from_elem((10, 1), 1.0);
    let penalized = s.dot(&ones);
    assert!(penalized.iter().all(|v| v.abs() < 1e-12));

    for i in 0..10 {
        for j in 0..10 {
            assert!((s[[i, j]] - s[[(i + 1) % 10, (j + 1) % 10]]).abs() < 1e-12);
            assert!((s[[i, j]] - s[[j, i]]).abs() < 1e-12);
        }
    }
}

#[test]
fn multi_output_periodic_curve_fits_anisotropic_ellipse_in_ambient_3d() {
    let n = 160;
    let period = std::f64::consts::TAU;
    let u = Array1::from_iter((0..n).map(|i| period * (i as f64) / (n as f64)));
    let mut y = Array2::<f64>::zeros((n, 3));
    for (i, &t) in u.iter().enumerate() {
        let c = t.cos();
        let s = t.sin();
        y[[i, 0]] = 1.2 + 3.5 * c - 0.4 * s;
        y[[i, 1]] = -0.7 + 0.9 * c + 1.8 * s;
        y[[i, 2]] = 0.3 - 1.1 * c + 0.6 * s;
    }

    let spec = PeriodicSplineCurveSpec::new(3, 32, period, 0.0, 2).unwrap();
    let control = fit_periodic_spline_curve(u.view(), y.view(), &spec, 1e-10).unwrap();
    assert_eq!(control.dim(), (32, 3));

    let fitted = evaluate_periodic_spline_curve(u.view(), control.view(), &spec).unwrap();
    let err = max_abs(&fitted, &y);
    assert!(err < 2.5e-3, "max ellipse fit error {err}");

    let seam = array![0.0, period];
    let seam_fit = evaluate_periodic_spline_curve(seam.view(), control.view(), &spec).unwrap();
    for col in 0..3 {
        assert!((seam_fit[[0, col]] - seam_fit[[1, col]]).abs() < 1e-11);
    }

    let range_x = y
        .column(0)
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    let range_y = y
        .column(1)
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    assert!(range_x.1 - range_x.0 > 1.5 * (range_y.1 - range_y.0));
}

#[test]
fn multi_output_periodic_curve_handles_distorted_closed_loop_in_4d() {
    let n = 192;
    let period = 1.0;
    let u = Array1::from_iter((0..n).map(|i| (i as f64) / (n as f64)));
    let mut y = Array2::<f64>::zeros((n, 4));
    for (i, &t01) in u.iter().enumerate() {
        let t = std::f64::consts::TAU * t01;
        y[[i, 0]] = 2.0 * t.cos() + 0.25 * (3.0 * t).cos();
        y[[i, 1]] = 0.55 * t.sin() - 0.18 * (2.0 * t).sin();
        y[[i, 2]] = 0.4 * (t + 0.3).cos() + 0.7 * (2.0 * t).sin();
        y[[i, 3]] = 0.1 * t.cos() - 0.35 * (4.0 * t).sin();
    }

    let spec = PeriodicSplineCurveSpec::new(3, 48, period, 0.0, 2).unwrap();
    let control = fit_periodic_spline_curve(u.view(), y.view(), &spec, 1e-9).unwrap();
    let fitted = evaluate_periodic_spline_curve(u.view(), control.view(), &spec).unwrap();
    let err = max_abs(&fitted, &y);
    assert!(err < 4e-3, "max distorted-loop fit error {err}");

    let shifted = array![0.125, 1.125, -0.875];
    let shifted_fit =
        evaluate_periodic_spline_curve(shifted.view(), control.view(), &spec).unwrap();
    for row in 1..shifted_fit.nrows() {
        for col in 0..shifted_fit.ncols() {
            assert!((shifted_fit[[0, col]] - shifted_fit[[row, col]]).abs() < 1e-11);
        }
    }
}

#[test]
fn periodic_bspline_terms_build_with_cyclic_penalty_and_formula_alias() {
    let x = array![0.0, 0.125, 0.25, 0.5, 0.75, 1.0];
    let data = Array2::from_shape_vec((x.len(), 1), x.to_vec()).unwrap();
    let term = SmoothTermSpec {
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
                boundary_conditions: Default::default(),
            },
        },
        shape: gam::terms::smooth::ShapeConstraint::None,
    };
    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![term],
    };
    let design = build_term_collection_design(data.view(), &spec).unwrap();
    assert_eq!(design.smooth.terms.len(), 1);
    assert_eq!(design.smooth.terms[0].penalties_local.len(), 2);

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
    let parsed = gam::inference::formula_dsl::parse_formula(
        "y ~ s(u, type=periodic, k=9, period=1)",
    )
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
