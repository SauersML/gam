//! Cross-check the EXACT analytic average-derivative design
//! (`build_term_collection_derivative_design`) against a high-accuracy central
//! finite-difference reference. Finite differences are permitted in tests; the
//! production path is fully analytic. This lives as a standalone integration
//! target (it touches only the public gam-terms surface) so it builds against
//! the crate library rather than the in-crate unit-test harness.

use gam_terms::basis::{
    BSplineBasisSpec, BSplineBoundaryConditions, BSplineEndpointBoundaryCondition,
    BSplineIdentifiability, BSplineKnotSpec, BasisMetadata, OneDimensionalBoundary,
};
use gam_terms::smooth::{
    LinearCoefficientGeometry, LinearTermSpec, ShapeConstraint, SmoothBasisSpec, SmoothTermSpec,
    TermCollectionSpec, build_term_collection_derivative_design, build_term_collection_design,
    freeze_term_collection_from_design,
};
use ndarray::{Array2, ArrayView2};

/// Build a FROZEN single-`s(x)` term-collection spec on `n` deterministic,
/// well-conditioned interior points in `[0.1, 0.9]`. Freezing pins the
/// sum-to-zero identifiability chart as a `FrozenTransform` constant so the
/// finite-difference reference replays the IDENTICAL transform on the shifted
/// data; an unfrozen sum-to-zero would recompute the chart per shift and inject
/// a spurious `dZ/dx` term that no analytic derivative carries.
fn frozen_bspline_spec_and_data() -> (TermCollectionSpec, Array2<f64>) {
    let n = 200usize;
    let mut data = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        // Mildly non-uniform interior grid, strictly inside the (0,1) knot
        // domain so every evaluation (and ±h shift) stays in support.
        let t = i as f64 / (n as f64 - 1.0);
        data[[i, 0]] = 0.1 + 0.8 * (0.5 * t + 0.5 * t * t);
    }

    let unfrozen = TermCollectionSpec {
        linear_terms: Vec::new(),
        random_effect_terms: Vec::new(),
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "s(x)".to_string(),
            basis: SmoothBasisSpec::BSpline1D {
                feature_col: 0,
                spec: BSplineBasisSpec {
                    degree: 3,
                    penalty_order: 2,
                    knotspec: BSplineKnotSpec::Generate {
                        data_range: (0.0, 1.0),
                        num_internal_knots: 8,
                    },
                    double_penalty: false,
                    identifiability: BSplineIdentifiability::WeightedSumToZero { weights: None },
                    boundary: OneDimensionalBoundary::Open,
                    boundary_conditions: Default::default(),
                },
            },
            shape: ShapeConstraint::None.into(),
            joint_null_rotation: None,
        }],
        level: Default::default(),
    };

    // Build once to recover the realized knots + composed identifiability chart,
    // then pin them into a frozen spec.
    let value = build_term_collection_design(data.view(), &unfrozen).expect("value design build");
    let (knots, degree, transform) = match &value.smooth.terms[0].metadata {
        BasisMetadata::BSpline1D {
            knots,
            degree,
            identifiability_transform,
            ..
        } => (
            knots.clone(),
            degree.expect("degree recorded"),
            identifiability_transform.clone(),
        ),
        other => panic!("expected B-spline metadata, got {other:?}"),
    };

    let frozen = TermCollectionSpec {
        linear_terms: Vec::new(),
        random_effect_terms: Vec::new(),
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "s(x)".to_string(),
            basis: SmoothBasisSpec::BSpline1D {
                feature_col: 0,
                spec: BSplineBasisSpec {
                    degree,
                    penalty_order: 2,
                    knotspec: BSplineKnotSpec::Provided(knots),
                    double_penalty: false,
                    identifiability: match transform {
                        Some(z) => BSplineIdentifiability::FrozenTransform { transform: z },
                        None => BSplineIdentifiability::None,
                    },
                    boundary: OneDimensionalBoundary::Open,
                    boundary_conditions: Default::default(),
                },
            },
            shape: ShapeConstraint::None.into(),
            joint_null_rotation: None,
        }],
        level: Default::default(),
    };

    (frozen, data)
}

fn dense_value_design(spec: &TermCollectionSpec, data: ArrayView2<'_, f64>) -> Array2<f64> {
    let built = build_term_collection_design(data, spec).expect("value design build");
    built
        .design
        .try_to_dense_arc("test value design")
        .expect("densify value design")
        .as_ref()
        .to_owned()
}

/// High-accuracy central finite-difference reference for `∂X/∂x_{deriv_col}`
/// on a frozen design. Richardson extrapolation of two central differences
/// gives an O(h^4) reference, comfortably below the 1e-6 acceptance bar.
fn richardson_central_difference(
    spec: &TermCollectionSpec,
    data: ArrayView2<'_, f64>,
    deriv_col: usize,
    spread: f64,
) -> Array2<f64> {
    let h = 1.0e-4 * spread;
    let central = |step: f64| -> Array2<f64> {
        let mut plus = data.to_owned();
        plus.column_mut(deriv_col).mapv_inplace(|v| v + step);
        let mut minus = data.to_owned();
        minus.column_mut(deriv_col).mapv_inplace(|v| v - step);
        let dp = dense_value_design(spec, plus.view());
        let dm = dense_value_design(spec, minus.view());
        (dp - dm) / (2.0 * step)
    };
    let cd_h = central(h);
    let cd_half = central(h / 2.0);
    (&cd_half * 4.0 - &cd_h) / 3.0
}

#[test]
fn analytic_average_derivative_matches_central_difference() {
    let (spec, data) = frozen_bspline_spec_and_data();
    let deriv_col = 0usize;

    let analytic = build_term_collection_derivative_design(data.view(), &spec, deriv_col)
        .expect("analytic derivative design");

    let reference = richardson_central_difference(&spec, data.view(), deriv_col, 0.8);

    assert_eq!(
        analytic.design.dim(),
        reference.dim(),
        "derivative design shape"
    );

    let max_abs_err = analytic
        .design
        .iter()
        .zip(reference.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    assert!(
        max_abs_err < 1e-6,
        "analytic average-derivative design disagrees with reference: \
         max abs error {max_abs_err:.3e} (tol 1e-6)"
    );
}

/// `x + s(x)`: the smooth's constraint block is `C = [1 | x]`. Its
/// sum-to-zero columns are orthogonal to the constant, so the constant is not
/// contained in their span and the collection takes the span-preserving
/// residualizing arm (#2747): the realized block is `X·T − C·R` with a nonzero
/// `x` row in `R`. `C` moves with `x`, so the exact derivative block is
/// `B'·M − C'·R`, and `C' = [0 | 1]` makes the correction a constant shift of
/// every derivative row. Dropping it biases every average derivative the fit
/// reports for the smooth.
#[test]
fn residualized_linear_plus_smooth_derivative_matches_central_difference() {
    let n = 200usize;
    let mut data = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        data[[i, 0]] = 0.1 + 0.8 * (0.5 * t + 0.5 * t * t);
    }
    let spec = TermCollectionSpec {
        linear_terms: vec![LinearTermSpec {
            name: "x".to_string(),
            feature_col: 0,
            feature_cols: vec![0],
            categorical_levels: Vec::new(),
            double_penalty: false,
            coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
            coefficient_min: None,
            coefficient_max: None,
            frozen_function_mass: None,
        }],
        random_effect_terms: Vec::new(),
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "s(x)".to_string(),
            basis: SmoothBasisSpec::BSpline1D {
                feature_col: 0,
                spec: BSplineBasisSpec {
                    degree: 3,
                    penalty_order: 2,
                    knotspec: BSplineKnotSpec::Generate {
                        data_range: (0.0, 1.0),
                        num_internal_knots: 8,
                    },
                    double_penalty: false,
                    identifiability: BSplineIdentifiability::WeightedSumToZero { weights: None },
                    boundary: OneDimensionalBoundary::Open,
                    boundary_conditions: Default::default(),
                },
            },
            shape: ShapeConstraint::None.into(),
            joint_null_rotation: None,
        }],
        level: Default::default(),
    };
    let design = build_term_collection_design(data.view(), &spec).expect("value design build");
    let frozen = freeze_term_collection_from_design(&spec, &design).expect("freeze");

    // The premise: this configuration took the residualizing arm, and its
    // correction really moves the derivative (a zero `x` row would make the
    // comparison below vacuous).
    let chart = frozen.smooth_terms[0]
        .frozen_parametric_residualization
        .as_ref()
        .expect("x + s(x) must carry a parametric residualization chart");
    assert!(chart.has_parametric_block, "[1 | x] must lead the block");
    assert_eq!(chart.correction.nrows(), 2, "C = [1 | x]");
    let slope_row_max = chart
        .correction
        .row(1)
        .iter()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));
    assert!(
        slope_row_max > 1e-3,
        "the x row of the correction must be material, got {slope_row_max:e}"
    );

    let deriv_col = 0usize;
    let analytic = build_term_collection_derivative_design(data.view(), &frozen, deriv_col)
        .expect("analytic derivative design");
    let reference = richardson_central_difference(&frozen, data.view(), deriv_col, 0.8);
    assert_eq!(
        analytic.design.dim(),
        reference.dim(),
        "derivative design shape"
    );
    let max_abs_err = analytic
        .design
        .iter()
        .zip(reference.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_abs_err < 1e-6,
        "residualized x + s(x) derivative design disagrees with the frozen value design's \
         central difference: max abs error {max_abs_err:.3e} (tol 1e-6)"
    );
}

#[test]
fn nonzero_anchor_derivative_carries_exact_affine_slope() {
    let data = ndarray::Array1::linspace(0.0, 1.0, 41).insert_axis(ndarray::Axis(1));
    let anchor = 1.75;
    let spec = TermCollectionSpec {
        linear_terms: Vec::new(),
        random_effect_terms: Vec::new(),
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "anchored s(x)".to_string(),
            basis: SmoothBasisSpec::BSpline1D {
                feature_col: 0,
                spec: BSplineBasisSpec {
                    degree: 3,
                    penalty_order: 2,
                    knotspec: BSplineKnotSpec::Generate {
                        data_range: (0.0, 1.0),
                        num_internal_knots: 7,
                    },
                    double_penalty: false,
                    identifiability: BSplineIdentifiability::None,
                    boundary: OneDimensionalBoundary::Open,
                    boundary_conditions: BSplineBoundaryConditions {
                        left: BSplineEndpointBoundaryCondition::Anchored { value: anchor },
                        right: BSplineEndpointBoundaryCondition::Free,
                    },
                },
            },
            shape: ShapeConstraint::None.into(),
            joint_null_rotation: None,
        }],
        level: Default::default(),
    };

    let value = build_term_collection_design(data.view(), &spec).expect("anchored value design");
    let derivative = build_term_collection_derivative_design(data.view(), &spec, 0)
        .expect("anchored derivative design");
    let beta = ndarray::Array1::<f64>::zeros(value.design.ncols());
    let fitted_value = value
        .apply(beta.view())
        .expect("apply anchored value design");
    let fitted_slope = derivative
        .apply(beta.view())
        .expect("apply anchored derivative design");

    assert!((fitted_value[0] - anchor).abs() < 1e-10);
    assert!(
        fitted_slope[0].abs() < 1e-10,
        "Hermite anchor requires zero endpoint slope, got {}",
        fitted_slope[0]
    );
    assert!(derivative.affine_offset[0].abs() < 1e-10);
}
