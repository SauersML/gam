//! gam#2953 (#2817): a term-local build hands back its penalty builder's
//! numbering when its renormalization chokepoint re-filters the active blocks.
//!
//! The incremental realizer aligns a trial's rebuilt blocks to the cached
//! topology by `(original_index, source)`. MSI job 1219335 printed README.md:201's
//! fatal trial: at ψ = 21.09 (ℓ = 6.9e-10) its ν = 5/2 Matérn term had cached
//! [Mass 0, Tension 1, Stiffness 2, ThirdOrder 3], and the rebuild returned
//! active [Mass 0, Stiffness 1] beside dropped [Tension 1, ThirdOrder 3]. At a
//! length scale far below the center spacing the kernel underflows between
//! centers and every odd radial derivative is zero at its own center, so the
//! two odd-order Grams are exactly zero and dropping them is the builder's
//! answer. Renumbering Stiffness to 1 was the defect: the realizer read the
//! cached Tension slot as Stiffness and aborted the fit instead of refusing the
//! trial.

use super::*;
use crate::basis::{BasisWorkspace, MaternIdentifiability, MaternNu};
use ndarray::array;

type Numbering = Vec<(usize, PenaltySource)>;

/// A build's active and dropped blocks as `(original_index, source)`.
fn numbering(build: &LocalSmoothTermBuild) -> (Numbering, Numbering) {
    let active = build
        .active_penalties
        .iter()
        .map(|penalty| (penalty.info.original_index, penalty.info.source.clone()))
        .collect();
    let dropped = build
        .dropped_penalties
        .iter()
        .map(|dropped| (dropped.original_index, dropped.source.clone()))
        .collect();
    (active, dropped)
}

#[test]
fn a_trial_dropping_odd_order_blocks_keeps_the_builders_numbering_2953() {
    let data = array![
        [-1.7, -0.4],
        [-1.1, 0.8],
        [-0.2, -1.3],
        [0.5, 1.6],
        [1.4, -0.7],
        [2.1, 0.5],
    ];
    let input_scale = estimate_isotropic_scale(data.view()).expect("isotropic input scale");
    let mut centers = data.clone();
    input_scale.standardize(&mut centers);
    let build_at = |length_scale: f64| {
        let term = SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "matern".to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: vec![0, 1],
                spec: MaternBasisSpec {
                    periodic: None,
                    center_strategy: CenterStrategy::UserProvided(centers.clone()),
                    length_scale: crate::basis::MaternLengthScale::fixed(length_scale),
                    nu: MaternNu::FiveHalves,
                    include_intercept: false,
                    double_penalty: true,
                    identifiability: MaternIdentifiability::CenterSumToZero,
                    aniso_log_scales: None,
                },
                input_scale: Some(input_scale),
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        };
        let mut workspace = BasisWorkspace::default();
        build_single_local_smooth_term(data.view(), &term, &mut workspace)
            .expect("term-local Matérn build")
    };

    let (cold_active, cold_dropped) = numbering(&build_at(1.0));
    assert_eq!(
        cold_active,
        vec![
            (0, PenaltySource::OperatorMass),
            (1, PenaltySource::OperatorTension),
            (2, PenaltySource::OperatorStiffness),
            (3, PenaltySource::OperatorThirdOrder),
        ],
        "the cold build carries all four operator blocks in candidate order"
    );
    assert!(cold_dropped.is_empty(), "the cold build drops nothing, got {cold_dropped:?}");

    let (trial_active, trial_dropped) = numbering(&build_at(1e-9));
    // The regime, read from sources so the numbering under test cannot decide
    // it: a block dropped at the trial precedes an active one in the cold
    // candidate order. With drops only at the tail, compacting and keeping the
    // numbering agree, and this pin would pass on the defect.
    let cold_position =
        |source: &PenaltySource| cold_active.iter().position(|(_, cold)| cold == source);
    assert!(
        trial_dropped.iter().any(|(_, dropped)| {
            trial_active
                .iter()
                .any(|(_, active)| cold_position(active) > cold_position(dropped))
        }),
        "the trial must drop a block ahead of an active one, got active {trial_active:?} \
         dropped {trial_dropped:?}"
    );
    for (index, source) in trial_active.iter().chain(&trial_dropped) {
        assert_eq!(
            cold_active.get(*index).map(|(_, cold_source)| cold_source),
            Some(source),
            "trial block {index} is {source:?} but the cold build's block {index} is not: \
             active {trial_active:?}, dropped {trial_dropped:?}"
        );
    }
    let mut indices: Vec<usize> = trial_active
        .iter()
        .chain(&trial_dropped)
        .map(|(index, _)| *index)
        .collect();
    indices.sort_unstable();
    assert_eq!(
        indices,
        vec![0, 1, 2, 3],
        "every cold block is either active or dropped at the trial, once"
    );
}
