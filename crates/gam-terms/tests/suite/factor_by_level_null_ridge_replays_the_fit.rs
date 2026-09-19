//! A factor-level `by=` smooth's double-penalty ridge must be the same at fit
//! and in the frozen rebuild of the saved model.
//!
//! The level wrapper defers the inner smooth's centering to the collection,
//! which centers the gated block against the level indicator and rebuilds the
//! ridge on the null space of the centered wiggliness penalty. The freeze stores
//! that composed chart as `FrozenTransform`, and the frozen B-spline build
//! charges the null function along the mean slope. The fit charged it as the
//! Euclidean complement of the chart instead, so every rebuild (prediction,
//! summary) carried a ridge whose direction made cosine 0.11 to 0.40 with the
//! fitted one, and the summary's smooth test was formed from penalties the fit
//! never used.

use gam_terms::basis::{
    BSplineBasisSpec, BSplineIdentifiability, BSplineKnotSpec, OneDimensionalBoundary,
    PenaltySource,
};
use gam_terms::smooth::{
    BySmoothKind, ByVariableSpec, ShapeConstraint, SmoothBasisSpec, SmoothTermSpec,
    TermCollectionSpec, build_term_collection_design, freeze_term_collection_from_design,
};
use ndarray::Array2;

const N: usize = 300;
const LEVELS: [f64; 3] = [0.0, 1.0, 2.0];

fn data() -> Array2<f64> {
    let mut state = 0x1561_0000_0000_0007_u64;
    let mut out = Array2::<f64>::zeros((N, 2));
    for row in 0..N {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        out[(row, 0)] = (state >> 11) as f64 / (1u64 << 53) as f64;
        out[(row, 1)] = LEVELS[row % LEVELS.len()];
    }
    out
}

fn level_term(level: f64) -> SmoothTermSpec {
    SmoothTermSpec {
        name: format!("s(x):g{level}"),
        basis: SmoothBasisSpec::ByVariable {
            inner: Box::new(SmoothBasisSpec::BSpline1D {
                feature_col: 0,
                spec: BSplineBasisSpec {
                    degree: 3,
                    penalty_order: 2,
                    knotspec: BSplineKnotSpec::Generate {
                        data_range: (0.0, 1.0),
                        num_internal_knots: 8,
                    },
                    double_penalty: true,
                    identifiability: BSplineIdentifiability::WeightedSumToZero { weights: None },
                    boundary: OneDimensionalBoundary::Open,
                    boundary_conditions: Default::default(),
                },
            }),
            by_col: 1,
            kind: BySmoothKind::Level {
                level_bits: level.to_bits(),
            },
            by: ByVariableSpec::Level {
                value_bits: level.to_bits(),
                label: format!("g{level}"),
            },
        },
        shape: ShapeConstraint::None.into(),
        joint_null_rotation: None,
        frozen_parametric_residualization: None,
    }
}

/// Each term's null ridge, Frobenius-normalized (the builds normalize every
/// block, so this only removes rounding in the recorded scale).
fn null_ridges(design: &gam_terms::smooth::TermCollectionDesign) -> Vec<Array2<f64>> {
    design
        .smooth
        .terms
        .iter()
        .map(|term| {
            let ridges: Vec<_> = term
                .active_penalties
                .iter()
                .filter(|penalty| {
                    matches!(penalty.info.source, PenaltySource::DoublePenaltyNullspace)
                })
                .collect();
            assert_eq!(ridges.len(), 1, "term '{}' carries one null ridge", term.name);
            let matrix = &ridges[0].matrix;
            let norm = matrix.iter().map(|value| value * value).sum::<f64>().sqrt();
            matrix / norm
        })
        .collect()
}

#[test]
fn factor_by_level_null_ridge_is_the_same_at_fit_and_in_the_frozen_rebuild() {
    let data = data();
    let spec = TermCollectionSpec {
        linear_terms: Vec::new(),
        random_effect_terms: Vec::new(),
        smooth_terms: LEVELS.iter().map(|&level| level_term(level)).collect(),
        level: Default::default(),
    };
    let fitted = build_term_collection_design(data.view(), &spec).expect("fit design builds");
    let frozen = freeze_term_collection_from_design(&spec, &fitted).expect("freeze");
    let replayed =
        build_term_collection_design(data.view(), &frozen).expect("frozen design rebuilds");

    let fitted_ridges = null_ridges(&fitted);
    let replayed_ridges = null_ridges(&replayed);
    assert_eq!(fitted_ridges.len(), LEVELS.len());
    for (level, (fit, replay)) in LEVELS.iter().zip(fitted_ridges.iter().zip(&replayed_ridges)) {
        assert_eq!(fit.dim(), replay.dim(), "level {level}: ridge shapes differ");
        let gap = (fit - replay).iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
        assert!(
            gap <= 1e-9,
            "level {level}: the fitted null ridge and its frozen rebuild differ by {gap:e}"
        );
    }
}
