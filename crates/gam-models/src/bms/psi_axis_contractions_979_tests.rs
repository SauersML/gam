//! gam#979: the rigid marginal-slope ψ axis contractions equal the materialized
//! all-beta-axes tensors contracted against the same symmetric kernels and weights.
//!
//! Ban-scanner-safe: a bare `#[cfg(test)] mod psi_axis_contractions_979_tests;` in
//! `bms/mod.rs` with the allowed `*_tests` name.

use super::family::*;
use super::hessian_paths::*;
use super::*;
use crate::custom_family::{BlockwiseFitOptions, CustomFamilyBlockPsiDerivative};
use crate::marginal_slope_shared::psi_derivative_location;
use gam_linalg::matrix::{DenseDesignMatrix, DesignMatrix};
use gam_problem::{InverseLink, ParameterBlockState, StandardLink};
use ndarray::{Array1, Array2};
use std::sync::{Arc, Mutex};

fn symmetric(p: usize, seed: f64) -> Array2<f64> {
    let raw = Array2::from_shape_fn((p, p), |(i, j)| {
        (seed + 0.37 * i as f64 - 0.19 * j as f64).sin() + 0.5 * ((i + j) as f64 * seed).cos()
    });
    (&raw + &raw.t()).mapv(|value| 0.5 * value)
}

fn psi_derivative(x_psi: Array2<f64>) -> CustomFamilyBlockPsiDerivative {
    let width = x_psi.ncols();
    CustomFamilyBlockPsiDerivative {
        penalty_index: None,
        x_psi,
        s_psi: Array2::zeros((width, width)),
        s_psi_components: None,
        s_psi_penalty_components: None,
        x_psi_psi: None,
        s_psi_psi: None,
        s_psi_psi_components: None,
        s_psi_psi_penalty_components: None,
        implicit_operator: None,
        implicit_axis: 0,
        implicit_group_id: None,
    }
}

/// One marginal and two slope ψ axes on a rigid probit fixture. Each row-pass
/// contraction must equal the materialized tensor's Frobenius products.
#[test]
fn rigid_psi_axis_contractions_match_the_materialized_tensors_979() {
    let n = 36usize;
    let (pm, pg) = (4usize, 3usize);
    let p = pm + pg;
    let marginal_x = Array2::from_shape_fn((n, pm), |(i, j)| {
        if j == 0 {
            1.0
        } else {
            ((i * (j + 2)) as f64 * 0.31).sin()
        }
    });
    let slope_x = Array2::from_shape_fn((n, pg), |(i, j)| {
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
        search_lane: None,
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
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let marginal_beta = Array1::from_vec(vec![-0.2, 0.3, -0.15, 0.1]);
    let slope_beta = Array1::from_vec(vec![0.4, -0.25, 0.2]);
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
    let derivative_blocks = vec![
        vec![psi_derivative(Array2::from_shape_fn((n, pm), |(i, j)| {
            0.3 * ((i + j) as f64 * 0.41).sin()
        }))],
        vec![
            psi_derivative(Array2::from_shape_fn((n, pg), |(i, j)| {
                0.2 * ((2 * i + j) as f64 * 0.17).cos()
            })),
            psi_derivative(Array2::from_shape_fn((n, pg), |(i, j)| {
                0.25 * ((i + 5 * j) as f64 * 0.29).sin()
            })),
        ],
    ];
    let cache = family
        .build_exact_eval_cache(&states)
        .expect("rigid exact eval cache");
    let options = BlockwiseFitOptions::default();
    let psi_count = 3usize;
    let axes: Vec<PsiAxisSpec> = (0..psi_count)
        .map(|psi_index| {
            let (block_idx, local_idx) = psi_derivative_location(&derivative_blocks, psi_index)
                .expect("every fixture axis is a design derivative");
            family
                .resolve_psi_axis_spec(&derivative_blocks, block_idx, local_idx)
                .expect("psi axis spec")
        })
        .collect();
    let kernels: Vec<Array2<f64>> = (0..p).map(|b| symmetric(p, 1.0 + b as f64)).collect();
    let mixed_weights: Vec<Array2<f64>> = (0..psi_count)
        .map(|i| symmetric(p, 11.0 + 3.0 * i as f64))
        .collect();
    let contracted = family
        .rigid_psi_hessian_all_beta_axes_contractions(
            &states,
            &axes,
            &cache,
            &options,
            &kernels,
            &mixed_weights,
        )
        .expect("rigid psi axis contractions");
    assert_eq!(contracted.len(), psi_count);
    let frobenius = |left: &Array2<f64>, right: &Array2<f64>| -> f64 {
        left.iter().zip(right.iter()).map(|(&l, &r)| l * r).sum()
    };
    let max_abs = |values: &[f64]| values.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    for (psi_index, (kernel_contractions, mixed_contractions)) in contracted.iter().enumerate() {
        let tensor = family
            .rigid_psi_hessian_all_beta_axes(&states, &derivative_blocks, psi_index, &cache, &options)
            .expect("materialized psi tensor")
            .expect("the rigid sweep serves every design axis");
        let reference_kernel =
            Array2::from_shape_fn((p, p), |(a, b)| frobenius(&tensor[a], &kernels[b]));
        let reference_mixed =
            Array1::from_shape_fn(p, |a| frobenius(&tensor[a], &mixed_weights[psi_index]));
        let kernel_gap = kernel_contractions - &reference_kernel;
        let mixed_gap = mixed_contractions - &reference_mixed;
        let scale = max_abs(reference_kernel.as_slice().expect("owned reference"))
            .max(max_abs(reference_mixed.as_slice().expect("owned reference")));
        let gap = max_abs(kernel_gap.as_slice().expect("owned gap"))
            .max(max_abs(mixed_gap.as_slice().expect("owned gap")));
        assert!(
            scale > 1e-8,
            "psi axis {psi_index}: the tensor is not exercised (max |contraction| {scale:e})"
        );
        assert!(
            gap <= 1e-10 * scale,
            "psi axis {psi_index}: row-pass contractions differ from the materialized tensor by \
             {gap:e} (max |contraction| {scale:e})"
        );
    }
}

/// The full-data correction traces fold each memory chunk's rows over the parallel
/// pairwise tree. On a flex fixture wide enough that the tree splits, they must equal
/// the per-row route, which projects every row through the factor itself, for the
/// batched gradient contraction and for the single-direction contraction alike.
#[test]
fn flex_full_row_correction_traces_match_the_per_row_route_979() {
    let n = 300usize;
    assert!(
        n > gam_linalg::pairwise_reduce::BASE_CHUNK,
        "the fixture must be wide enough for the row fold to split"
    );
    let (pm, pg) = (3usize, 2usize);
    let marginal_x = Array2::from_shape_fn((n, pm), |(i, j)| {
        if j == 0 {
            1.0
        } else {
            ((i * (j + 2)) as f64 * 0.31).sin()
        }
    });
    let slope_x = Array2::from_shape_fn((n, pg), |(i, j)| {
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
    let score_seed = Array1::linspace(-2.0, 2.0, 8);
    let link_seed = Array1::linspace(-1.8, 1.8, 8);
    let score = build_score_warp_deviation_block_from_seed(&score_seed, &config)
        .expect("score-warp deviation block");
    let link =
        build_link_deviation_block_from_knots_design_seed_and_weights(&link_seed, &link_seed, &config)
            .expect("link deviation block");
    let policy = gam_runtime::resource::ResourcePolicy::default_library();
    let family = BernoulliMarginalSlopeFamily {
        jeffreys_armed: false,
        search_lane: None,
        y: Arc::new(Array1::from_shape_fn(n, |i| {
            if (i * 7) % 5 < 2 { 1.0 } else { 0.0 }
        })),
        weights: Arc::new(Array1::from_shape_fn(n, |i| 0.6 + 0.002 * i as f64)),
        z: Arc::new(Array1::from_shape_fn(n, |i| 1.3 * (i as f64 * 0.57).sin())),
        latent_measure: LatentMeasureKind::StandardNormal,
        gaussian_frailty_sd: Some(0.15),
        base_link: InverseLink::Standard(StandardLink::Probit),
        marginal_design: DesignMatrix::Dense(DenseDesignMatrix::from(marginal_x.clone())),
        slope_design: DesignMatrix::Dense(DenseDesignMatrix::from(slope_x.clone())),
        score_warp: Some(score.runtime.clone()),
        link_dev: Some(link.runtime.clone()),
        residual: None,
        policy: policy.clone(),
        cell_moment_lru: new_cell_moment_lru_cache(&policy),
        cell_moment_cache_stats: new_cell_moment_cache_stats(),
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let marginal_beta = Array1::from_vec(vec![-0.2, 0.3, -0.15]);
    let slope_beta = Array1::from_vec(vec![0.4, -0.25]);
    let score_beta =
        Array1::from_shape_fn(score.runtime.basis_dim(), |index| 0.0015 * (index as f64 + 1.0));
    let link_beta =
        Array1::from_shape_fn(link.runtime.basis_dim(), |index| -0.001 * (index as f64 + 1.0));
    let p = pm + pg + score_beta.len() + link_beta.len();
    let states = vec![
        ParameterBlockState {
            eta: marginal_x.dot(&marginal_beta),
            beta: marginal_beta,
        },
        ParameterBlockState {
            eta: slope_x.dot(&slope_beta),
            beta: slope_beta,
        },
        ParameterBlockState {
            eta: Array1::zeros(n),
            beta: score_beta,
        },
        ParameterBlockState {
            eta: Array1::zeros(n),
            beta: link_beta,
        },
    ];
    let cache = family
        .build_exact_eval_cache(&states)
        .expect("flex exact eval cache");
    let factor = symmetric(p, 2.3);
    let directions = Array2::from_shape_fn((p, 3), |(i, j)| ((i + 2 * j) as f64 * 0.43).sin());
    let every_row: Vec<WeightedOuterRow> = (0..n)
        .map(|index| WeightedOuterRow {
            index,
            weight: 1.0,
            stratum: 0,
        })
        .collect();
    let max_abs = |values: &Array1<f64>| values.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    for n_dirs in [3usize, 1] {
        let directions = directions.slice(ndarray::s![.., ..n_dirs]).to_owned();
        let folded = family
            .batched_rho_correction_logdet_traces_full_rows(&states, &cache, &factor, &directions)
            .expect("full-row correction traces");
        let per_row = family
            .batched_rho_correction_logdet_traces_for_rows(
                &states,
                &cache,
                &factor,
                &directions,
                &every_row,
            )
            .expect("per-row correction traces");
        let scale = max_abs(&per_row);
        let gap = max_abs(&(&folded - &per_row));
        assert!(
            scale > 1e-8,
            "{n_dirs} direction(s): the traces are not exercised (max |trace| {scale:e})"
        );
        assert!(
            gap <= 1e-10 * scale,
            "{n_dirs} direction(s): the full-row fold differs from the per-row route by {gap:e} \
             (max |trace| {scale:e})"
        );
    }
}
