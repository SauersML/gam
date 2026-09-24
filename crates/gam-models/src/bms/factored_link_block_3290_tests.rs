//! gam#3290: an empirical-law FLEX row's third and fourth contractions sum
//! the row's calibration cells, the lowering the standard-normal law uses,
//! instead of carrying the row program's dense `r`-wide jets. The dense jets
//! stay here as the oracle: at two link widths, on a row with both deviation
//! blocks active, the production contractions match the dense row program
//! contracted along the same directions from the same intercept root.
//!
//! Ban-scanner-safe: a bare `#[cfg(test)] mod factored_link_block_3290_tests;`
//! in `bms/mod.rs` with the allowed `*_tests` name.

use super::family::*;
use super::flex_row_program::BmsFlexRowProgram;
use super::hessian_paths::*;
use super::*;
use gam_linalg::matrix::{DenseDesignMatrix, DesignMatrix};
use gam_math::jet_scalar::{DynamicJetBatchWorkspace, DynamicOneSeedBatch, DynamicTwoSeedBatch};
use gam_math::roundoff::accumulation_growth;
use gam_problem::{InverseLink, ParameterBlockState, StandardLink};
use ndarray::{Array1, Array2};
use std::sync::{Arc, Mutex};

/// Rounded operations in one node term of the widest contraction, the fourth
/// (`EmpiricalCalibrationCell::fourth`): fifteen cubic coefficient
/// polynomials and the cell's cubic `η` evaluated by Horner's rule (six each),
/// the two Hermite weights `η² − 1` and `3η − η³` (five), the fourteen products
/// and sums of the Faà di Bruno combination (seven `3+1`/`2+2` products and
/// their sum, six `2+1+1` triple products and their sum, the quadruple
/// product), and the weighted accumulation (two). A contraction over `G` nodes
/// is then a sum of `G` such terms.
pub(super) const EMPIRICAL_NODE_TERM_OPERATIONS: usize = 16 * 6 + 5 + (7 * 2 + 6 * 3 + 4) + 2;

/// The dense oracle for a third contraction: the canonical row program in the
/// runtime-width one-seed jets, `lanes` directions per pass.
pub(super) fn dense_third_contracted(
    plan: &BmsFlexRowProgram,
    point: &[f64],
    directions: &[Array1<f64>],
    r: usize,
    lanes: usize,
) -> Result<Vec<Array2<f64>>, String> {
    let mut workspace = DynamicJetBatchWorkspace::new(lanes);
    let mut contracted = Vec::with_capacity(directions.len());
    for chunk in directions.chunks(lanes) {
        workspace.reset(chunk.len());
        let vars = workspace.alloc_slice_fill_with(r, |axis| {
            DynamicOneSeedBatch::seed_directions(point[axis], axis, r, &workspace, |lane| {
                chunk[lane][axis]
            })
        });
        let jet = plan.evaluate(vars, 3, &workspace)?;
        for lane in 0..chunk.len() {
            contracted.push(
                Array2::from_shape_vec((r, r), jet.contracted_third(lane).to_vec())
                    .map_err(|error| format!("dense third-contraction shape: {error}"))?,
            );
        }
    }
    Ok(contracted)
}

/// The dense oracle for ordered fourth contractions `T4[u, v, ·, ·]`: the
/// canonical row program in the runtime-width two-seed jets, `lanes` pairs per
/// pass.
pub(super) fn dense_fourth_contracted(
    plan: &BmsFlexRowProgram,
    point: &[f64],
    direction_pairs: &[(&Array1<f64>, &Array1<f64>)],
    r: usize,
    lanes: usize,
) -> Result<Vec<Array2<f64>>, String> {
    let mut workspace = DynamicJetBatchWorkspace::new(lanes);
    let mut contracted = Vec::with_capacity(direction_pairs.len());
    for pairs in direction_pairs.chunks(lanes) {
        workspace.reset(pairs.len());
        let vars = workspace.alloc_slice_fill_with(r, |axis| {
            DynamicTwoSeedBatch::seed_direction_pairs(point[axis], axis, r, &workspace, |lane| {
                (pairs[lane].0[axis], pairs[lane].1[axis])
            })
        });
        let jet = plan.evaluate(vars, 4, &workspace)?;
        for lane in 0..pairs.len() {
            contracted.push(
                Array2::from_shape_vec((r, r), jet.contracted_fourth(lane).to_vec())
                    .map_err(|error| format!("dense fourth-contraction shape: {error}"))?,
            );
        }
    }
    Ok(contracted)
}

/// A 65-node global empirical grid over `[−2.6, 2.6]` with bell-shaped
/// weights, the production grid size.
pub(super) fn grid() -> EmpiricalZGrid {
    const NODES: usize = 65;
    let nodes: Vec<f64> = (0..NODES)
        .map(|i| -2.6 + 5.2 * (i as f64) / ((NODES - 1) as f64))
        .collect();
    let raw: Vec<f64> = nodes.iter().map(|z| (-0.5 * z * z).exp()).collect();
    let total: f64 = raw.iter().sum();
    let weights: Vec<f64> = raw.iter().map(|w| w / total).collect();
    EmpiricalZGrid::new(nodes, weights, "gam#3290 grid").expect("valid 65-node grid")
}

/// A one-row global-empirical FLEX family with a score warp and a link
/// deviation of `link_internal_knots` internal knots, at a fitted state whose
/// deviation coefficients are all nonzero.
pub(super) fn empirical_flex_fixture(
    link_internal_knots: usize,
) -> (BernoulliMarginalSlopeFamily, Vec<ParameterBlockState>) {
    let score_seed = Array1::linspace(-2.0, 2.0, 8);
    let link_seed = Array1::linspace(-1.8, 1.8, 8);
    let score = build_score_warp_deviation_block_from_seed(
        &score_seed,
        &DeviationBlockConfig {
            num_internal_knots: 3,
            ..DeviationBlockConfig::default()
        },
    )
    .expect("score-warp block");
    let link = build_link_deviation_block_from_knots_design_seed_and_weights(
        &link_seed,
        &link_seed,
        &DeviationBlockConfig {
            num_internal_knots: link_internal_knots,
            ..DeviationBlockConfig::default()
        },
    )
    .expect("link-deviation block");
    let marginal_x = Array2::ones((1, 1));
    let slope_x = Array2::ones((1, 1));
    let policy = gam_runtime::resource::ResourcePolicy::default_library();
    let family = BernoulliMarginalSlopeFamily {
        jeffreys_armed: true,
        residual: None,
        search: None,
        y: Arc::new(Array1::from_vec(vec![1.0])),
        weights: Arc::new(Array1::from_vec(vec![0.9])),
        z: Arc::new(Array1::from_vec(vec![0.35])),
        latent_measure: LatentMeasureKind::GlobalEmpirical { grid: grid() },
        gaussian_frailty_sd: None,
        base_link: InverseLink::Standard(StandardLink::Probit),
        marginal_design: DesignMatrix::Dense(DenseDesignMatrix::from(marginal_x.clone())),
        slope_design: DesignMatrix::Dense(DenseDesignMatrix::from(slope_x.clone())),
        score_warp: Some(score.runtime.clone()),
        link_dev: Some(link.runtime.clone()),
        policy: policy.clone(),
        cell_moment_lru: new_cell_moment_lru_cache(&policy),
        cell_moment_cache_stats: new_cell_moment_cache_stats(),
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        outer_start_levels: None,
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let marginal_beta = Array1::from_vec(vec![0.18]);
    let slope_beta = Array1::from_vec(vec![0.32]);
    let score_beta =
        Array1::from_shape_fn(score.runtime.basis_dim(), |index| 0.015 * (index as f64 + 1.0));
    let link_beta =
        Array1::from_shape_fn(link.runtime.basis_dim(), |index| -0.01 * (index as f64 + 1.0));
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
            eta: Array1::zeros(1),
            beta: score_beta,
        },
        ParameterBlockState {
            eta: Array1::zeros(1),
            beta: link_beta,
        },
    ];
    (family, states)
}

/// The production and dense contractions of one fixture row: the third along
/// `u`, and the ordered fourth along `(u, v)`, with the time each route took.
struct RowContractions {
    r: usize,
    link_width: usize,
    third: [Array2<f64>; 2],
    fourth: [Array2<f64>; 2],
    elapsed: [[std::time::Duration; 2]; 2],
}

fn row_contractions(link_internal_knots: usize, repetitions: usize) -> RowContractions {
    let row = 0usize;
    let (family, states) = empirical_flex_fixture(link_internal_knots);
    let cache = family
        .build_exact_eval_cache(&states)
        .expect("empirical FLEX exact cache");
    let primary = cache.primary.clone();
    let r = primary.total;
    let h_range = primary.h.clone().expect("active score-warp range");
    let w_range = primary.w.clone().expect("active link-deviation range");
    let row_ctx = BernoulliMarginalSlopeFamily::row_ctx(&cache, row);
    let grid = family
        .training_row_grid(row)
        .expect("row grid")
        .expect("global-empirical fixture reads its grid");

    // Two distinct directions through every primary, so the q, slope, score
    // and link blocks all cross in both contractions.
    let dir_u = Array1::from_shape_fn(r, |i| 0.5 + 0.3 * ((i % 3) as f64) - 0.2 * ((i % 2) as f64));
    let dir_v = Array1::from_shape_fn(r, |i| -0.4 + 0.3 * (((i + 1) % 4) as f64) - 0.1 * ((i % 2) as f64));
    assert!(dir_u[h_range.start] != 0.0 && dir_v[w_range.start] != 0.0);

    let third_call = || {
        family
            .row_primary_third_contracted_with_moments(row, &states, &cache, row_ctx, &dir_u)
            .expect("production third contraction")
    };
    let fourth_call = || {
        family
            .row_primary_fourth_contracted_ordered(row, &states, &cache, row_ctx, &dir_u, &dir_v)
            .expect("production fourth contraction")
    };
    let point = family
        .primary_point_from_block_states(row, &states, &primary)
        .expect("primary point");
    let (q, b, beta_h, beta_w) = family.primary_point_components(&point, &primary);
    let primary_point = BernoulliMarginalSlopeFamily::intercept_primary_point(
        q,
        b,
        beta_h.as_ref(),
        beta_w.as_ref(),
    );
    // The dense route compiles the row program on every contraction, as its
    // production callers did, so its time includes the compile.
    let plan = || {
        family
            .compile_empirical_bms_row_program(
                row,
                &primary,
                q,
                b,
                beta_h.as_ref(),
                beta_w.as_ref(),
                row_ctx.intercept,
                &grid,
            )
            .expect("canonical empirical-flex row plan")
    };
    let dense_third_call = || {
        dense_third_contracted(&plan(), &primary_point, std::slice::from_ref(&dir_u), r, 1)
            .expect("dense third contraction")
            .remove(0)
    };
    let dense_fourth_call = || {
        dense_fourth_contracted(&plan(), &primary_point, &[(&dir_u, &dir_v)], r, 1)
            .expect("dense fourth contraction")
            .remove(0)
    };
    let third = third_call();
    let fourth = fourth_call();
    let dense_third = dense_third_call();
    let dense_fourth = dense_fourth_call();
    // After those warm calls, the median of `repetitions` calls per route.
    let median = |call: &dyn Fn() -> Array2<f64>| -> std::time::Duration {
        let mut times: Vec<std::time::Duration> = (0..repetitions)
            .map(|repetition| {
                let started = std::time::Instant::now();
                let value = call();
                let elapsed = started.elapsed();
                assert!(value.iter().all(|entry| entry.is_finite()), "repetition {repetition}");
                elapsed
            })
            .collect();
        times.sort_unstable();
        times[times.len() / 2]
    };
    let elapsed = [
        [median(&third_call), median(&dense_third_call)],
        [median(&fourth_call), median(&dense_fourth_call)],
    ];
    RowContractions {
        r,
        link_width: w_range.len(),
        third: [third, dense_third],
        fourth: [fourth, dense_fourth],
        elapsed,
    }
}

/// Both routes sum the same grid's contributions node by node, the cell route
/// with at most [`EMPIRICAL_NODE_TERM_OPERATIONS`] rounded operations per node
/// term, then one `r`-wide implicit and observed finalizer each. The band is
/// Wilkinson's factor over that accumulation, `G·K + r²` rounded operations,
/// on the largest entry.
fn assert_matches_dense(label: &str, production: &Array2<f64>, dense: &Array2<f64>, grid_nodes: usize) {
    let r = dense.nrows();
    let scale = dense.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
    assert!(scale > 0.0, "{label}: the dense contraction is identically zero");
    let band = accumulation_growth(grid_nodes * EMPIRICAL_NODE_TERM_OPERATIONS + r * r) * scale;
    let mut worst = 0.0_f64;
    for ((index, &got), &want) in production.indexed_iter().zip(dense.iter()) {
        assert!(
            got.is_finite() && (got - want).abs() <= band,
            "{label} {index:?}: cells {got:+.17e} vs dense {want:+.17e} (band {band:e})"
        );
        worst = worst.max((got - want).abs());
    }
    eprintln!("gam#3290 {label}: max |cells − dense| = {worst:e}, relative {:e}", worst / scale);
}

#[test]
fn empirical_cell_contractions_match_the_dense_row_program_3290() {
    let grid_nodes = grid().nodes.len();
    for link_internal_knots in [2usize, 4] {
        let result = row_contractions(link_internal_knots, 1);
        let label = format!("r={} link width={}", result.r, result.link_width);
        assert_matches_dense(
            &format!("{label} third"),
            &result.third[0],
            &result.third[1],
            grid_nodes,
        );
        assert_matches_dense(
            &format!("{label} fourth"),
            &result.fourth[0],
            &result.fourth[1],
            grid_nodes,
        );
    }
}

/// The per-contraction cost record for gam#3290: the cell route and the dense
/// row program from the issue's link width (`internal_knots = 2`) past the
/// eight-knot default, one row, the median of nine warm calls per route, the
/// dense one including the row-program compile its callers paid per
/// contraction. Diagnostic only: wall time on
/// a shared runner is not asserted here; the acceptance measurement is the
/// #3011 fit's per-evaluation wall on its own hardware.
#[test]
fn zz_measure_3290_cell_and_dense_contraction_cost() {
    for link_internal_knots in [2usize, 4, 8, 16] {
        let result = row_contractions(link_internal_knots, 9);
        let [[third, dense_third], [fourth, dense_fourth]] = result.elapsed;
        eprintln!(
            "gam#3290 cost r={} link width={}: third cells {third:?} dense {dense_third:?}; \
             fourth cells {fourth:?} dense {dense_fourth:?}",
            result.r, result.link_width
        );
    }
}
