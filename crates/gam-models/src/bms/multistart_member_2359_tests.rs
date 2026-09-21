//! gnomon#2359: a member of a parallel multistart reuses only its own same-β
//! exact caches and rigid tensors, never another search's and never the
//! process-wide store's, so its result does not depend on which searches ran
//! beside it: a member's values are bit-identical at one lane and at six.
//!
//! Ban-scanner-safe: a bare `#[cfg(test)] mod multistart_member_2359_tests;` in
//! `bms/mod.rs` with the allowed `*_tests` name.

use super::family::*;
use super::hessian_paths::*;
use super::row_kernel::{
    BernoulliMarginalSlopeExactNewtonJointHessianWorkspace, BernoulliRigidRowKernel,
};
use super::*;
use crate::custom_family::{BlockwiseFitOptions, ExactNewtonJointHessianWorkspace};
use gam_linalg::matrix::{DenseDesignMatrix, DesignMatrix};
use gam_model_api::families::custom_family::IndependentOuterSearch;
use gam_problem::{InverseLink, ParameterBlockState, StandardLink};
use ndarray::{Array1, Array2};
use std::sync::{Arc, Mutex};

/// A rigid probit family on 24 rows and a β-state for it.
pub(super) fn rigid_fixture() -> (BernoulliMarginalSlopeFamily, Vec<ParameterBlockState>) {
    let n = 24usize;
    let marginal_x = Array2::from_shape_fn((n, 3), |(i, j)| {
        if j == 0 {
            1.0
        } else {
            ((i * (j + 2)) as f64 * 0.31).sin()
        }
    });
    let slope_x = Array2::from_shape_fn((n, 2), |(i, j)| {
        if j == 0 {
            1.0
        } else {
            ((i + 3 * j) as f64 * 0.23).cos()
        }
    });
    let policy = gam_runtime::resource::ResourcePolicy::default_library();
    let family = BernoulliMarginalSlopeFamily {
        jeffreys_armed: true,
        residual: None,
        search: None,
        y: Arc::new(Array1::from_shape_fn(n, |i| {
            if (i * 7) % 5 < 2 { 1.0 } else { 0.0 }
        })),
        weights: Arc::new(Array1::from_shape_fn(n, |i| 0.6 + 0.02 * i as f64)),
        z: Arc::new(Array1::from_shape_fn(n, |i| 1.3 * (i as f64 * 0.57).sin())),
        latent_measure: LatentMeasureKind::StandardNormal,
        gaussian_frailty_sd: None,
        base_link: InverseLink::Standard(StandardLink::Probit),
        marginal_design: DesignMatrix::Dense(DenseDesignMatrix::from(marginal_x.clone())),
        slope_design: DesignMatrix::Dense(DenseDesignMatrix::from(slope_x.clone())),
        score_warp: None,
        link_dev: None,
        policy: policy.clone(),
        cell_moment_lru: new_cell_moment_lru_cache(&policy),
        cell_moment_cache_stats: new_cell_moment_cache_stats(),
        jet_scratch: crate::bms::hessian_paths::new_jet_scratch(),
        intercept_warm_starts: Some(
            new_intercept_warm_start_cache_on_law(&LatentMeasureKind::StandardNormal, n)
                .expect("an intercept cache on the standard-normal law"),
        ),
        auto_subsample_phase_counter: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let marginal_beta = Array1::from_vec(vec![-0.2, 0.3, -0.15]);
    let slope_beta = Array1::from_vec(vec![0.4, -0.25]);
    let states = vec![
        ParameterBlockState {
            eta: marginal_x.dot(&marginal_beta),
            beta: marginal_beta,
        },
        ParameterBlockState {
            eta: slope_x.dot(&slope_beta),
            beta: slope_beta,
        },
    ];
    (family, states)
}

pub(super) fn member(family: &BernoulliMarginalSlopeFamily) -> BernoulliMarginalSlopeFamily {
    family.outer_search_member(Arc::new(gam_runtime::resource::SearchLaneBudget::new(
        u64::MAX, None,
    )))
}

/// Members share the parent's data buffers, so their exact-cache fingerprints
/// at one β agree. Each member builds its own cache there and hits it again,
/// and neither reads the process-wide store a fit on its own uses.
#[test]
fn a_multistart_member_reuses_only_its_own_exact_caches_2359() {
    let (family, states) = rigid_fixture();
    let options = BlockwiseFitOptions::default();
    let first = member(&family);
    let second = member(&family);
    let build = |family: &BernoulliMarginalSlopeFamily| {
        family
            .build_or_reuse_shared_exact_cache(&states, &options, false)
            .expect("rigid exact eval cache")
    };
    let parent = build(&family);
    let first_cache = build(&first);
    let second_cache = build(&second);
    assert!(
        !Arc::ptr_eq(&first_cache, &parent),
        "a member reused the process-wide store's exact cache"
    );
    assert!(
        !Arc::ptr_eq(&second_cache, &first_cache),
        "a member reused another member's exact cache"
    );
    assert!(
        Arc::ptr_eq(&build(&first), &first_cache),
        "a member did not keep its own same-β exact cache"
    );
}

/// The rigid per-row third and fourth tensors follow the same rule: a second
/// kernel of one member at one β reads its tensors, another member's does not.
#[test]
fn a_multistart_member_reuses_only_its_own_rigid_tensors_2359() {
    let (family, states) = rigid_fixture();
    let first = member(&family);
    let second = member(&family);
    let kernel = |family: &BernoulliMarginalSlopeFamily| {
        BernoulliRigidRowKernel::new(family.clone(), states.clone())
    };
    let (first_a, first_b, other, parent) =
        (kernel(&first), kernel(&first), kernel(&second), kernel(&family));
    assert!(
        std::ptr::eq(first_a.third_rows(), first_b.third_rows()),
        "a member did not keep its own third tensors"
    );
    assert!(
        std::ptr::eq(first_a.fourth_rows(), first_b.fourth_rows()),
        "a member did not keep its own fourth tensors"
    );
    assert!(
        !std::ptr::eq(other.third_rows(), first_a.third_rows()),
        "a member reused another member's third tensors"
    );
    assert!(
        !std::ptr::eq(parent.fourth_rows(), first_a.fourth_rows()),
        "a member reused the process-wide store's fourth tensors"
    );
}

/// A link-deviation probit family on 24 rows, with its designs. Its row
/// intercepts are root solves started from the family's own warm starts, so an
/// exact cache carries the roots its builder's history converged to.
fn flex_fixture() -> (BernoulliMarginalSlopeFamily, Array2<f64>, Array2<f64>, usize) {
    let n = 24usize;
    let marginal_x = Array2::from_shape_fn((n, 3), |(i, j)| {
        if j == 0 {
            1.0
        } else {
            ((i * (j + 2)) as f64 * 0.31).sin()
        }
    });
    let slope_x = Array2::from_shape_fn((n, 2), |(i, j)| {
        if j == 0 {
            1.0
        } else {
            ((i + 3 * j) as f64 * 0.23).cos()
        }
    });
    let config = DeviationBlockConfig {
        num_internal_knots: 3,
        ..DeviationBlockConfig::default()
    };
    let link_seed = Array1::linspace(-1.8, 1.8, 8);
    let link =
        build_link_deviation_block_from_knots_design_seed_and_weights(&link_seed, &link_seed, &config)
            .expect("link deviation block");
    let link_width = link.runtime.basis_dim();
    let policy = gam_runtime::resource::ResourcePolicy::default_library();
    let family = BernoulliMarginalSlopeFamily {
        jeffreys_armed: false,
        residual: None,
        search: None,
        y: Arc::new(Array1::from_shape_fn(n, |i| {
            if (i * 7) % 5 < 2 { 1.0 } else { 0.0 }
        })),
        weights: Arc::new(Array1::from_shape_fn(n, |i| 0.6 + 0.02 * i as f64)),
        z: Arc::new(Array1::from_shape_fn(n, |i| 1.3 * (i as f64 * 0.57).sin())),
        latent_measure: LatentMeasureKind::StandardNormal,
        gaussian_frailty_sd: None,
        base_link: InverseLink::Standard(StandardLink::Probit),
        marginal_design: DesignMatrix::Dense(DenseDesignMatrix::from(marginal_x.clone())),
        slope_design: DesignMatrix::Dense(DenseDesignMatrix::from(slope_x.clone())),
        score_warp: None,
        link_dev: Some(link.runtime.clone()),
        policy: policy.clone(),
        cell_moment_lru: new_cell_moment_lru_cache(&policy),
        cell_moment_cache_stats: new_cell_moment_cache_stats(),
        jet_scratch: crate::bms::hessian_paths::new_jet_scratch(),
        intercept_warm_starts: Some(
            new_intercept_warm_start_cache_on_law(&LatentMeasureKind::StandardNormal, n)
                .expect("an intercept cache on the standard-normal law"),
        ),
        auto_subsample_phase_counter: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    (family, marginal_x, slope_x, link_width)
}

/// The flex fixture's β-state moved `t` along one fixed direction.
fn flex_states(
    marginal_x: &Array2<f64>,
    slope_x: &Array2<f64>,
    link_width: usize,
    t: f64,
) -> Vec<ParameterBlockState> {
    let marginal_beta = Array1::from_vec(vec![-0.2 + 0.03 * t, 0.3 - 0.02 * t, -0.15 + 0.01 * t]);
    let slope_beta = Array1::from_vec(vec![0.4 + 0.02 * t, -0.25 + 0.015 * t]);
    let link_beta =
        Array1::from_shape_fn(link_width, |index| -0.001 * (1.0 + 0.1 * t) * (index as f64 + 1.0));
    vec![
        ParameterBlockState {
            eta: marginal_x.dot(&marginal_beta),
            beta: marginal_beta,
        },
        ParameterBlockState {
            eta: slope_x.dot(&slope_beta),
            beta: slope_beta,
        },
        ParameterBlockState {
            eta: Array1::zeros(marginal_x.nrows()),
            beta: link_beta,
        },
    ]
}

/// What a search's value at β is built from, as bits: the log-likelihood,
/// gradient and joint Hessian of the exact cache its store serves at β.
fn evaluation_bits(family: &BernoulliMarginalSlopeFamily, states: &[ParameterBlockState]) -> Vec<u64> {
    let workspace = BernoulliMarginalSlopeExactNewtonJointHessianWorkspace::new(
        family.clone(),
        states.to_vec(),
        BlockwiseFitOptions::default(),
    )
    .expect("flex joint Hessian workspace");
    let fused = workspace.fused_gradient_dense().expect("flex fused evaluation");
    let hessian = workspace
        .hessian_dense_forced()
        .expect("flex dense joint Hessian")
        .expect("a flex workspace assembles its dense joint Hessian");
    std::iter::once(fused.gradient.log_likelihood)
        .chain(fused.gradient.gradient.iter().copied())
        .chain(hessian.iter().copied())
        .map(f64::to_bits)
        .collect()
}

/// gam-97's pin: six members walk their own β paths, which start at the β the
/// parent evaluated last and return to it, one member after another (one lane)
/// and all at once in lockstep (six lanes). Every evaluation is bit-identical
/// between the two. Were members to share the process-wide store, the lockstep
/// members would all read the parent's β0 cache, built from the parent's warm
/// starts, where one after another every member but the first rebuilds it from
/// its own.
#[test]
fn a_multistart_members_values_are_bit_identical_at_one_and_six_lanes_2359() {
    let (family, marginal_x, slope_x, link_width) = flex_fixture();
    let states = |t: f64| flex_states(&marginal_x, &slope_x, link_width, t);
    let paths: Vec<Vec<Vec<ParameterBlockState>>> = (0..6)
        .map(|k| {
            let step = 0.1 * (k as f64 + 1.0);
            vec![states(0.0), states(step), states(-step), states(0.0)]
        })
        .collect();
    // Before each schedule the parent leaves its β0 cache, built from its own
    // warm starts after another β, in the process-wide store.
    let parent_evaluates = || {
        evaluation_bits(&family, &states(0.7));
        evaluation_bits(&family, &states(0.0));
    };
    parent_evaluates();
    let one_lane: Vec<Vec<Vec<u64>>> = paths
        .iter()
        .map(|path| {
            let searching = member(&family);
            path.iter()
                .map(|point| evaluation_bits(&searching, point))
                .collect()
        })
        .collect();
    parent_evaluates();
    let lockstep = std::sync::Barrier::new(paths.len());
    let six_lanes: Vec<Vec<Vec<u64>>> = std::thread::scope(|scope| {
        let lanes: Vec<_> = paths
            .iter()
            .map(|path| {
                let searching = member(&family);
                let lockstep = &lockstep;
                scope.spawn(move || {
                    path.iter()
                        .map(|point| {
                            lockstep.wait();
                            evaluation_bits(&searching, point)
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        lanes
            .into_iter()
            .map(|lane| lane.join().expect("a lane panicked"))
            .collect()
    });
    for (index, (alone, beside)) in one_lane.iter().zip(&six_lanes).enumerate() {
        assert_eq!(
            alone, beside,
            "member {index}'s evaluations differ between one lane and six"
        );
    }
}

/// A fit on its own reads the process-wide store, which outlives it. Its key is
/// the content its rows read: a family with the same values in fresh buffers
/// hits, and a family whose empirical law alone differs — same data, same β,
/// same η — misses, where a key on buffer addresses and the law's variant hit.
#[test]
fn the_process_wide_exact_cache_is_keyed_on_the_data_and_the_law_it_reads() {
    let on_law = |weights: [f64; 3]| {
        let (mut family, states) = rigid_fixture();
        let law = LatentMeasureKind::GlobalEmpirical {
            grid: EmpiricalZGrid::new(vec![-1.5, 0.0, 1.5], weights.to_vec(), "test law")
                .expect("valid grid"),
        };
        family.intercept_warm_starts = Some(
            new_intercept_warm_start_cache_on_law(&law, family.y.len())
                .expect("an intercept cache on the empirical law"),
        );
        family.latent_measure = law;
        (family, states)
    };
    let options = BlockwiseFitOptions::default();
    let build = |(family, states): &(BernoulliMarginalSlopeFamily, Vec<ParameterBlockState>)| {
        family
            .build_or_reuse_shared_exact_cache(states, &options, false)
            .expect("empirical exact eval cache")
    };
    let first = build(&on_law([0.25, 0.5, 0.25]));
    assert!(
        Arc::ptr_eq(&build(&on_law([0.25, 0.5, 0.25])), &first),
        "the same data, law and β in fresh buffers missed the exact cache"
    );
    assert!(
        !Arc::ptr_eq(&build(&on_law([0.3, 0.4, 0.3])), &first),
        "a different empirical law reused another law's exact cache"
    );
}
