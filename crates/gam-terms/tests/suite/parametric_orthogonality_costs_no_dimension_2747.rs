//! gam#2747: the span-preserving parametric orthogonalization, end to end
//! through `build_term_collection_design`.
//!
//! `apply_global_smooth_identifiability` exists to enforce one invariant — the
//! realized smooth block is orthogonal to `[intercept | owned linear axes |
//! owner smooths]` — and it enforced it by DELETING one coefficient direction
//! per constraint direction. `76a520c45` established that the deletion is
//! licensed only under containment and withheld it otherwise, which left the
//! invariant unenforced for every basis whose span does not contain its
//! constraint block (measured: `1.6e-1 … 4.9e-1` against the step's own `1e-8`
//! bar, and `analyze_smooth_ownership`'s hierarchy inert for every dependent
//! smooth).
//!
//! The two claims that make projection the right answer are asserted here as
//! properties of the SHIPPED pipeline rather than of the numerical helper:
//!
//! 1. **orthogonality at no cost** — the realized block is orthogonal to its
//!    constraint block, and carries the same number of coefficient directions
//!    the raw basis emitted;
//! 2. **the chart is a REPLAY, not a rederivation** — a rebuild from the frozen
//!    spec on a SUBSET of the training rows reproduces the fit-time design's
//!    corresponding rows bit for bit. This is the property `76a520c45`'s carrier
//!    could not have had and the one #978 names: re-deriving `C(CᵀC)⁻¹CᵀX` from
//!    prediction rows silently evaluates a different model, and every row of it
//!    still looks like a design.

use gam_terms::basis::{
    BSplineBasisSpec, BSplineIdentifiability, BSplineKnotSpec, CenterStrategy,
    ConstantCurvatureBasisSpec, ConstantCurvatureIdentifiability, MaternBasisSpec,
    MaternIdentifiability, MaternLengthScale, MaternNu, OneDimensionalBoundary,
};
use gam_terms::smooth::{
    LinearCoefficientGeometry, LinearTermSpec, ShapeConstraint, SmoothBasisSpec, SmoothTermSpec,
    TermCollectionDesign, TermCollectionSpec, build_term_collection_design,
    freeze_term_collection_from_design,
};
use ndarray::{Array2, s};

const N: usize = 300;
const CENTERS: usize = 18;

/// A deterministic 2-D cloud in the unit disk, plus a response column the
/// linear-term machinery can name.
fn cloud() -> Array2<f64> {
    let mut state = 0x2747_0000_0000_0003_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut out = Array2::<f64>::zeros((N, 2));
    let mut filled = 0usize;
    while filled < N {
        let a = 2.0 * next() - 1.0;
        let b = 2.0 * next() - 1.0;
        if a * a + b * b <= 1.0 {
            out[(filled, 0)] = a;
            out[(filled, 1)] = b;
            filled += 1;
        }
    }
    out
}

fn curvature_term(name: &str) -> SmoothTermSpec {
    SmoothTermSpec {
        name: name.to_string(),
        basis: SmoothBasisSpec::ConstantCurvature {
            feature_cols: vec![0, 1],
            spec: ConstantCurvatureBasisSpec {
                center_strategy: CenterStrategy::FarthestPoint {
                    num_centers: CENTERS,
                },
                kappa: 0.5,
                kappa_fixed: true,
                length_scale: 1.0,
                length_scale_fixed: true,
                double_penalty: false,
                identifiability: ConstantCurvatureIdentifiability::CenterSumToZero,
            },
        },
        shape: ShapeConstraint::None.into(),
        joint_null_rotation: None,
        frozen_parametric_residualization: None,
    }
}

fn owner_spline_term(name: &str) -> SmoothTermSpec {
    SmoothTermSpec {
        name: name.to_string(),
        basis: SmoothBasisSpec::BSpline1D {
            feature_col: 0,
            spec: BSplineBasisSpec {
                degree: 3,
                penalty_order: 2,
                knotspec: BSplineKnotSpec::Generate {
                    data_range: (-1.0, 1.0),
                    num_internal_knots: 5,
                },
                double_penalty: false,
                identifiability: BSplineIdentifiability::WeightedSumToZero { weights: None },
                boundary: OneDimensionalBoundary::Open,
                boundary_conditions: Default::default(),
            },
        },
        shape: ShapeConstraint::None.into(),
        joint_null_rotation: None,
        frozen_parametric_residualization: None,
    }
}

fn linear_term(name: &str, col: usize) -> LinearTermSpec {
    LinearTermSpec {
        name: name.to_string(),
        feature_col: col,
        feature_cols: vec![col],
        categorical_levels: Vec::new(),
        double_penalty: false,
        coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
        coefficient_min: None,
        coefficient_max: None,
        frozen_function_mass: None,
    }
}

/// Where the smooth block starts in the full design: `SmoothTerm::coeff_range`
/// is smooth-block-local.
fn smooth_offset(design: &TermCollectionDesign) -> usize {
    design
        .intercept_range
        .end
        .max(
            design
                .linear_ranges
                .iter()
                .map(|(_, range)| range.end)
                .max()
                .unwrap_or(0),
        )
        .max(
            design
                .random_effect_ranges
                .iter()
                .map(|(_, range)| range.end)
                .max()
                .unwrap_or(0),
        )
}

fn block_of(design: &TermCollectionDesign, name: &str) -> Array2<f64> {
    let dense = design.design.to_dense();
    let offset = smooth_offset(design);
    let built = design
        .smooth
        .terms
        .iter()
        .find(|term| term.name == name)
        .unwrap_or_else(|| panic!("term '{name}' must be in the built design"));
    dense
        .slice(s![
            ..,
            offset + built.coeff_range.start..offset + built.coeff_range.end
        ])
        .to_owned()
}

fn relative_cross(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let cross = a.t().dot(b);
    let num = cross.iter().map(|v| v * v).sum::<f64>().sqrt();
    let a_norm = a.iter().map(|v| v * v).sum::<f64>().sqrt();
    let b_norm = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    num / (a_norm * b_norm).max(1e-300)
}

/// The bar the step itself asserts on a transformed design
/// (`ORTHOGONALITY_REL_RESIDUAL_TOL`). Stated here so a change to it in
/// production shows up as a disagreement rather than as a silent loosening.
const SHIPPED_ORTHOGONALITY_BAR: f64 = 1e-8;

/// A kernel smooth whose span does NOT contain the constant comes out
/// orthogonal to the intercept AND keeps every coefficient direction.
///
/// The dimension half is the one the deletion cannot satisfy: before the
/// projection arm this configuration either lost one direction (pre-`76a520c45`)
/// or kept the direction and abandoned the orthogonality (after it).
#[test]
fn a_kernel_smooth_is_orthogonal_to_the_intercept_and_keeps_its_width_2747() {
    let data = cloud();
    let spec = TermCollectionSpec {
        linear_terms: Vec::new(),
        random_effect_terms: Vec::new(),
        smooth_terms: vec![curvature_term("curv")],
        level: Default::default(),
    };
    let design = build_term_collection_design(data.view(), &spec).expect("design builds");
    let block = block_of(&design, "curv");
    let intercept = Array2::<f64>::ones((data.nrows(), 1));

    // The premise: this basis really does NOT contain the constant, so the test
    // is about the configuration it was written for and not a vacuous one.
    let contained = gam_terms::basis::contained_constraint_directions(
        &gam_linalg::matrix::DesignMatrix::from(block.clone()),
        intercept.view(),
        None,
    )
    .expect("containment test runs");
    assert_eq!(
        contained.ncols(),
        0,
        "premise: the realized kernel span must NOT contain the constant, or the \
         deletion would be free here and this test would be measuring nothing"
    );

    let cross = relative_cross(&block, &intercept);
    assert!(
        cross <= SHIPPED_ORTHOGONALITY_BAR,
        "the realized kernel block must be orthogonal to the intercept: {cross:e} > {SHIPPED_ORTHOGONALITY_BAR:e}"
    );

    // And it costs nothing: the raw basis at these centers emits `CENTERS - 1`
    // columns after its own center-space sum-to-zero, and the global step must
    // hand every one of them through.
    let raw = gam_terms::basis::build_constant_curvature_basis(
        data.view(),
        match &spec.smooth_terms[0].basis {
            SmoothBasisSpec::ConstantCurvature { spec, .. } => spec,
            other => panic!("expected a constant-curvature term, got {other:?}"),
        },
    )
    .expect("raw basis builds");
    assert_eq!(
        block.ncols(),
        raw.design.ncols(),
        "the orthogonalization must not delete a coefficient direction on a span \
         that does not contain the constant"
    );
}

/// The ownership hierarchy: a broader smooth is orthogonal to its OWNER's
/// realized columns, and still keeps its width.
///
/// This is the configuration `76a520c45` silently disabled for every dependent
/// smooth — an owner's realized basis columns are contained in no other basis's
/// span, so a containment gate withholds the whole block rather than one
/// direction of it.
#[test]
fn a_dependent_smooth_is_orthogonal_to_its_owner_and_keeps_its_width_2747() {
    let data = cloud();
    let spec = TermCollectionSpec {
        linear_terms: Vec::new(),
        random_effect_terms: Vec::new(),
        smooth_terms: vec![owner_spline_term("s(x1)"), curvature_term("curv")],
        level: Default::default(),
    };
    let design = build_term_collection_design(data.view(), &spec).expect("design builds");
    let owner = block_of(&design, "s(x1)");
    let dependent = block_of(&design, "curv");
    assert!(owner.ncols() > 1, "the owner must be a real block");

    let cross = relative_cross(&dependent, &owner);
    assert!(
        cross <= SHIPPED_ORTHOGONALITY_BAR,
        "the dependent smooth must be orthogonal to its owner's realized columns: \
         {cross:e} > {SHIPPED_ORTHOGONALITY_BAR:e}"
    );

    let alone = TermCollectionSpec {
        linear_terms: Vec::new(),
        random_effect_terms: Vec::new(),
        smooth_terms: vec![curvature_term("curv")],
        level: Default::default(),
    };
    let alone_design = build_term_collection_design(data.view(), &alone).expect("design builds");
    assert_eq!(
        dependent.ncols(),
        block_of(&alone_design, "curv").ncols(),
        "acquiring an owner must not cost the dependent smooth a coefficient direction"
    );
}

/// An overlapping LINEAR term is part of the same constraint block, and the
/// same two properties hold against `[1 | x1]`.
#[test]
fn an_overlapping_linear_term_is_orthogonalized_at_no_cost_2747() {
    let data = cloud();
    let spec = TermCollectionSpec {
        linear_terms: vec![linear_term("x1", 0)],
        random_effect_terms: Vec::new(),
        smooth_terms: vec![curvature_term("curv")],
        level: Default::default(),
    };
    let design = build_term_collection_design(data.view(), &spec).expect("design builds");
    let block = block_of(&design, "curv");
    let mut constraint = Array2::<f64>::ones((data.nrows(), 2));
    constraint.slice_mut(s![.., 1]).assign(&data.column(0));

    let cross = relative_cross(&block, &constraint);
    assert!(
        cross <= SHIPPED_ORTHOGONALITY_BAR,
        "the smooth must be orthogonal to [1 | x1]: {cross:e} > {SHIPPED_ORTHOGONALITY_BAR:e}"
    );
    let alone = TermCollectionSpec {
        linear_terms: Vec::new(),
        random_effect_terms: Vec::new(),
        smooth_terms: vec![curvature_term("curv")],
        level: Default::default(),
    };
    let alone_design = build_term_collection_design(data.view(), &alone).expect("design builds");
    assert_eq!(
        block.ncols(),
        block_of(&alone_design, "curv").ncols(),
        "an overlapping linear term must not cost the smooth a coefficient direction"
    );
}

/// The residualization is REPLAYED, never re-derived.
///
/// A rebuild from the frozen spec on a SUBSET of the training rows must
/// reproduce the fit-time design's corresponding rows bit for bit. Re-deriving
/// `C(CᵀC)⁻¹CᵀX` from those rows would produce a different — and perfectly
/// plausible-looking — design, which is the failure #978 names for the
/// coefficient half of the same chart.
#[test]
fn the_residualization_chart_replays_bit_for_bit_on_a_row_subset_2747() {
    let data = cloud();
    let spec = TermCollectionSpec {
        linear_terms: vec![linear_term("x1", 0)],
        random_effect_terms: Vec::new(),
        smooth_terms: vec![owner_spline_term("s(x1)"), curvature_term("curv")],
        level: Default::default(),
    };
    let design = build_term_collection_design(data.view(), &spec).expect("design builds");
    let frozen = freeze_term_collection_from_design(&spec, &design).expect("freeze");

    // The premise: this configuration really did take the residualizing arm.
    let chart = frozen.smooth_terms[1]
        .frozen_parametric_residualization
        .as_ref()
        .expect("the dependent kernel smooth must carry a residualization chart");
    assert!(
        chart.has_parametric_block,
        "the parametric block led this term's constraint block"
    );
    assert_eq!(
        chart.owner_terms,
        vec![0],
        "the chart must name the owner it was built against"
    );

    let fit_time = block_of(&design, "curv");
    let rows: Vec<usize> = (0..N).step_by(7).collect();
    let mut subset = Array2::<f64>::zeros((rows.len(), data.ncols()));
    for (slot, &row) in rows.iter().enumerate() {
        subset.row_mut(slot).assign(&data.row(row));
    }
    let replayed =
        build_term_collection_design(subset.view(), &frozen).expect("frozen design rebuilds");
    let replayed_block = block_of(&replayed, "curv");
    assert_eq!(
        replayed_block.ncols(),
        fit_time.ncols(),
        "the replayed design must have the fit-time width"
    );
    let mut worst = 0.0_f64;
    for (slot, &row) in rows.iter().enumerate() {
        for col in 0..fit_time.ncols() {
            worst = worst.max((replayed_block[[slot, col]] - fit_time[[row, col]]).abs());
        }
    }
    assert!(
        worst <= 1e-12,
        "a rebuild on a subset of the training rows must reproduce those rows of the \
         fit-time design; worst entry gap {worst:e}"
    );
    // And it is not vacuous: a rebuild that DISCARDED the chart would move, and
    // by far more than the bar above.
    let mut chartless = frozen.clone();
    chartless.smooth_terms[1].frozen_parametric_residualization = None;
    let without =
        build_term_collection_design(subset.view(), &chartless).expect("chartless design rebuilds");
    let without_block = block_of(&without, "curv");
    let mut divergence = 0.0_f64;
    for (slot, &row) in rows.iter().enumerate() {
        for col in 0..fit_time.ncols().min(without_block.ncols()) {
            divergence = divergence.max((without_block[[slot, col]] - fit_time[[row, col]]).abs());
        }
    }
    assert!(
        divergence > 1e-6,
        "dropping the chart must visibly change the design, or this test cannot tell a \
         replay from a rederivation; divergence {divergence:e}"
    );
}

/// The same two properties on a MATERN smooth, from a spec whose identifiability
/// is still `CenterSumToZero`.
///
/// This arm exists to localize a SECOND gap rather than to re-test the first.
/// Through `fit_from_formula`, a Matern smooth measures `4.15e-1` against the
/// intercept even with the projection arm landed
/// (measured under #2747), and this test says why that
/// cannot be the arm's fault: from an unfrozen spec the same basis comes out
/// orthogonal at the shipped bar. What differs is the route —
/// `freeze_geometry_from_metadata` (`spatial_optimization.rs:4849`) freezes the
/// kappa optimizer's cold-build chart as `MaternIdentifiability::FrozenTransform`,
/// that chart comes from `realize_single_smooth_term`, whose own comment says it
/// "never runs the global ownership pass", and
/// `smooth_requires_parametric_orthogonality`'s doc then excludes
/// `FrozenTransform` bases on the premise that such a transform "already has the
/// parametric orthogonalization composed in". For that producer the premise is
/// false, and it has been since long before #2747.
#[test]
fn an_unfrozen_matern_smooth_is_orthogonalized_at_no_cost_2747() {
    let data = cloud();
    let matern = SmoothTermSpec {
        name: "matern".to_string(),
        basis: SmoothBasisSpec::Matern {
            feature_cols: vec![0, 1],
            spec: MaternBasisSpec {
                center_strategy: CenterStrategy::FarthestPoint {
                    num_centers: CENTERS,
                },
                periodic: None,
                length_scale: MaternLengthScale::fixed(1.0),
                nu: MaternNu::ThreeHalves,
                include_intercept: false,
                double_penalty: false,
                identifiability: MaternIdentifiability::CenterSumToZero,
                aniso_log_scales: None,
            },
            input_scale: None,
        },
        shape: ShapeConstraint::None.into(),
        joint_null_rotation: None,
        frozen_parametric_residualization: None,
    };
    let spec = TermCollectionSpec {
        linear_terms: Vec::new(),
        random_effect_terms: Vec::new(),
        smooth_terms: vec![matern],
        level: Default::default(),
    };
    let design = build_term_collection_design(data.view(), &spec).expect("design builds");
    let block = block_of(&design, "matern");
    let intercept = Array2::<f64>::ones((data.nrows(), 1));

    let contained = gam_terms::basis::contained_constraint_directions(
        &gam_linalg::matrix::DesignMatrix::from(block.clone()),
        intercept.view(),
        None,
    )
    .expect("containment test runs");
    assert_eq!(
        contained.ncols(),
        0,
        "premise: the realized Matern span must NOT contain the constant"
    );
    let cross = relative_cross(&block, &intercept);
    assert!(
        cross <= SHIPPED_ORTHOGONALITY_BAR,
        "an unfrozen Matern block must be orthogonal to the intercept: {cross:e}"
    );
    assert_eq!(
        block.ncols(),
        CENTERS - 1,
        "the center-space sum-to-zero emits `centers - 1` columns and the global \
         step must hand every one of them through"
    );
}
