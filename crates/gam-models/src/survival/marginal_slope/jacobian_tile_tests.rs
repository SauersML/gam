#![cfg(test)]
//! Gates for the row-tiled whole-projection `J · F` of the rigid survival
//! kernel (gnomon#2337).
//!
//! The tiled build forms every channel of a row tile through the design's
//! `row_chunk_matmul_into`, where the former build formed each channel over all
//! rows (a GEMM on a dense design or its memo, one full matvec per factor column
//! on an operator-backed one) and packed the blocks afterwards. On a dense or
//! memoized design each entry is the same k-ordered GEMM sum, so the two agree
//! bit for bit. On a sparse design the tile's GEMM replaces the sparse matvec,
//! and a gauged operator forms `X·(T·F)` with compensated products, so the gate
//! there is accuracy against a double-double reference of `J · F`. Tiles write
//! disjoint rows, so the result is pinned bitwise at 1, 4 and 12 workers.

use super::*;
use crate::row_kernel::RowKernel;
use faer::sparse::{SparseColMat, Triplet};
use gam_linalg::matrix::{
    BlockDesignOperator, CoefficientTransformOperator, DenseDesignMatrix, DesignBlock,
};
use ndarray::{Array1, Array2};
use std::sync::Arc;

const N_ROWS: usize = 1_000;
const TIME_COLS: usize = 5;
const MARGINAL_COLS: usize = 7;
const SLOPE_COLS: usize = 4;
const GAUGE_RANK: usize = 3;
const FACTOR_RANK: usize = 6;

fn noise(i: usize, j: usize, salt: f64) -> f64 {
    ((i as f64 * 12.9898 + j as f64 * 78.233 + salt).sin() * 43758.5453).fract() - 0.5
}

/// A design's exact entries: dense columns `x`, less `q · r` when gauged.
struct DesignSource {
    x: Array2<f64>,
    gauge: Option<(Array2<f64>, Array2<f64>)>,
}

#[derive(Clone, Copy, Debug)]
enum Storage {
    Dense,
    Sparse,
    Gauged,
}

fn source(cols: usize, salt: f64, storage: Storage) -> DesignSource {
    // A quarter of the entries are zero so the sparse storage is sparse.
    let x = Array2::from_shape_fn((N_ROWS, cols), |(i, j)| {
        if (i * 7 + j * 3) % 4 == 0 { 0.0 } else { noise(i, j, salt) }
    });
    let gauge = matches!(storage, Storage::Gauged).then(|| {
        (
            Array2::from_shape_fn((N_ROWS, GAUGE_RANK), |(i, j)| noise(i, j, salt + 0.5)),
            Array2::from_shape_fn((GAUGE_RANK, cols), |(i, j)| noise(i, j, salt + 0.25)),
        )
    });
    DesignSource { x, gauge }
}

fn design(source: &DesignSource, storage: Storage) -> DesignMatrix {
    match storage {
        Storage::Dense => DesignMatrix::from(source.x.clone()),
        Storage::Sparse => {
            let mut triplets = Vec::<Triplet<usize, usize, f64>>::new();
            for ((i, j), &value) in source.x.indexed_iter() {
                if value != 0.0 {
                    triplets.push(Triplet::new(i, j, value));
                }
            }
            let sparse = SparseColMat::try_new_from_triplets(N_ROWS, source.x.ncols(), &triplets)
                .expect("assemble sparse design");
            DesignMatrix::Sparse(gam_linalg::matrix::SparseDesignMatrix::new(sparse))
        }
        Storage::Gauged => {
            let (q, r) = source.gauge.as_ref().expect("a gauged source carries its gauge");
            // The gauged Duchon shape: a coefficient transform [I; −R] over the
            // stacked block operator [X | Q].
            let stacked = BlockDesignOperator::new(vec![
                DesignBlock::Dense(DenseDesignMatrix::from(source.x.clone())),
                DesignBlock::Dense(DenseDesignMatrix::from(q.clone())),
            ])
            .expect("stacked gauge block");
            let cols = source.x.ncols();
            let mut transform = Array2::<f64>::zeros((cols + GAUGE_RANK, cols));
            for column in 0..cols {
                transform[[column, column]] = 1.0;
                for range_column in 0..GAUGE_RANK {
                    transform[[cols + range_column, column]] = -r[[range_column, column]];
                }
            }
            let operator = CoefficientTransformOperator::new(
                DenseDesignMatrix::from(Arc::new(stacked)),
                transform,
            )
            .expect("gauge coefficient transform");
            DesignMatrix::Dense(DenseDesignMatrix::from(Arc::new(operator)))
        }
    }
}

struct Fixture {
    kernel: SurvivalMarginalSlopeRowKernel<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry>,
    time: [DesignSource; 3],
    marginal: DesignSource,
    slope: DesignSource,
    factor: Array2<f64>,
}

fn fixture(time_storage: Storage, covariate_storage: Storage) -> Fixture {
    let time = [
        source(TIME_COLS, 1.0, time_storage),
        source(TIME_COLS, 2.0, time_storage),
        source(TIME_COLS, 3.0, time_storage),
    ];
    let marginal = source(MARGINAL_COLS, 4.0, covariate_storage);
    let slope = source(SLOPE_COLS, 5.0, covariate_storage);
    let n = N_ROWS;
    let family = SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n,
        entry_at_origin: Arc::new(Array1::from_elem(n, false)),
        event: Arc::new(Array1::from_shape_fn(n, |r| ((r % 3 == 0) as u8) as f64)),
        weights: Arc::new(Array1::from_shape_fn(n, |r| 0.7 + 0.5 * ((r % 5) as f64) / 5.0)),
        z: Arc::new(Array2::from_shape_fn((n, 1), |(r, _)| ((r as f64) * 0.37).sin() * 1.1)),
        score_covariance: ScoreCovarianceField::pooled(
            MarginalSlopeCovariance::diagonal(ndarray::array![1.0])
                .expect("a 1x1 unit latent-score covariance"),
        ),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-8,
        design_entry: design(&time[0], time_storage),
        design_exit: design(&time[1], time_storage),
        design_derivative_exit: design(&time[2], time_storage),
        offset_entry: Arc::new(Array1::from_shape_fn(n, |r| 0.05 * (r as f64).sin() - 0.2)),
        offset_exit: Arc::new(Array1::from_shape_fn(n, |r| 0.15 - 0.03 * (r as f64).cos())),
        derivative_offset_exit: Arc::new(Array1::from_elem(n, 1.0)),
        marginal_design: design(&marginal, covariate_storage),
        slope_layout: design(&slope, covariate_storage).into(),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
    };
    let beta_marginal = Array1::from_shape_fn(MARGINAL_COLS, |j| 0.03 * (j as f64) - 0.08);
    let beta_slope = Array1::from_shape_fn(SLOPE_COLS, |j| 0.05 - 0.04 * (j as f64));
    let block_states = vec![
        ParameterBlockState {
            beta: Array1::from_shape_fn(TIME_COLS, |j| 0.1 + 0.02 * (j as f64)),
            eta: Array1::zeros(n),
        },
        ParameterBlockState {
            eta: family.marginal_design.dot(&beta_marginal),
            beta: beta_marginal,
        },
        ParameterBlockState {
            eta: family
                .slope_layout
                .static_coefficient_design()
                .expect("static slope design")
                .dot(&beta_slope),
            beta: beta_slope,
        },
    ];
    let kernel = SurvivalMarginalSlopeRowKernel::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry>::new(
        family,
        block_states,
    );
    let p = RowKernel::n_coefficients(&kernel);
    assert_eq!(p, TIME_COLS + MARGINAL_COLS + SLOPE_COLS);
    let factor = Array2::from_shape_fn((p, FACTOR_RANK), |(k, c)| noise(k, c, 9.0) * 2.0);
    Fixture {
        kernel,
        time,
        marginal,
        slope,
        factor,
    }
}

/// The former whole-projection build: each channel over all rows, then packed.
fn former_jf(fixture: &Fixture) -> Array2<f64> {
    fixture.kernel.assemble_jf(fixture.factor.view(), N_ROWS, |design, block| {
        crate::row_kernel::row_kernel_design_jf(design, block, N_ROWS)
    })
}

fn tiled_jf(fixture: &Fixture, workers: usize) -> Array2<f64> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .expect("test worker pool")
        .install(|| RowKernel::jacobian_action_matrix(&fixture.kernel, fixture.factor.view()))
        .expect("tiled J·F")
}

fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let sum = a + b;
    let b_part = sum - a;
    (sum, (a - (sum - b_part)) + (b - b_part))
}

fn two_product(a: f64, b: f64) -> (f64, f64) {
    let product = a * b;
    (product, a.mul_add(b, -product))
}

/// Double-double `J · F` and its entrywise term mass, from the designs' exact
/// entries (a gauged entry is `x − q·r`, expanded rather than rounded).
fn reference_jf(fixture: &Fixture) -> (Array2<f64>, Array2<f64>) {
    let p_time = TIME_COLS;
    let f = &fixture.factor;
    let mut exact = Array2::<f64>::zeros((N_ROWS, STATIC_SLOPE_PRIMARIES * FACTOR_RANK));
    let mut mass = Array2::<f64>::zeros(exact.dim());
    for row in 0..N_ROWS {
        for c in 0..FACTOR_RANK {
            for primary in 0..STATIC_SLOPE_PRIMARIES {
                // Each term as an unevaluated sum high + low that equals the
                // exact product: FMA recovers a product's rounding error, and a
                // three-factor product carries the first product's error along.
                let mut terms: Vec<(f64, f64)> = Vec::new();
                let push_design = |source: &DesignSource, offset: usize, terms: &mut Vec<(f64, f64)>| {
                    for k in 0..source.x.ncols() {
                        let weight = f[[offset + k, c]];
                        terms.push(two_product(source.x[[row, k]], weight));
                        if let Some((q, r)) = &source.gauge {
                            for l in 0..GAUGE_RANK {
                                let (inner, inner_error) = two_product(q[[row, l]], r[[l, k]]);
                                let (outer, outer_error) = two_product(inner, weight);
                                terms.push((-outer, -(outer_error + inner_error * weight)));
                            }
                        }
                    }
                };
                match primary {
                    PRIMARY_Q0 | PRIMARY_Q1 | PRIMARY_QD1 => {
                        let time = match primary {
                            PRIMARY_Q0 => &fixture.time[0],
                            PRIMARY_Q1 => &fixture.time[1],
                            _ => &fixture.time[2],
                        };
                        push_design(time, 0, &mut terms);
                        if primary != PRIMARY_QD1 {
                            push_design(&fixture.marginal, p_time, &mut terms);
                        }
                    }
                    _ => push_design(&fixture.slope, p_time + MARGINAL_COLS, &mut terms),
                }
                let (mut high, mut low) = (0.0_f64, 0.0_f64);
                let mut absolute = 0.0_f64;
                for &(term, term_error) in &terms {
                    let (sum, error) = two_sum(high, term);
                    let (renormalized, rest) = two_sum(sum, low + error + term_error);
                    high = renormalized;
                    low = rest;
                    absolute += term.abs();
                }
                exact[[row, primary * FACTOR_RANK + c]] = high + low;
                mass[[row, primary * FACTOR_RANK + c]] = absolute;
            }
        }
    }
    (exact, mass)
}

fn worst_scaled_error(jf: &Array2<f64>, exact: &Array2<f64>, mass: &Array2<f64>) -> f64 {
    let mut worst = 0.0_f64;
    for ((value, reference), scale) in jf.iter().zip(exact.iter()).zip(mass.iter()) {
        if *scale > 0.0 {
            worst = worst.max((value - reference).abs() / scale);
        }
    }
    worst
}

fn assert_width_invariant(fixture: &Fixture, label: &str) -> Array2<f64> {
    let one = tiled_jf(fixture, 1);
    for workers in [4, 12] {
        let wide = tiled_jf(fixture, workers);
        for (index, (a, b)) in one.iter().zip(wide.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{label} J·F entry {index}: 1 worker {a:e} vs {workers} workers {b:e}"
            );
        }
    }
    one
}

#[test]
fn tiled_jf_matches_the_former_build_bitwise_on_dense_designs_2337() {
    let fixture = fixture(Storage::Dense, Storage::Dense);
    let tiled = assert_width_invariant(&fixture, "dense");
    let former = former_jf(&fixture);
    assert_eq!(tiled.dim(), former.dim());
    for (index, (a, b)) in tiled.iter().zip(former.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "dense J·F entry {index}: tiled {a:e} vs former {b:e}");
    }
}

/// A gauged design memoized to dense, as fit entry memoizes it whenever the
/// governor admits the copy, is read from its memo: the tiled J·F equals the
/// former build bit for bit, as on a dense design.
#[test]
fn tiled_jf_reads_a_memoized_operator_design_bitwise_2337() {
    let fixture = fixture(Storage::Dense, Storage::Gauged);
    fixture.kernel.family.memoize_operator_backed_designs();
    assert!(
        fixture.kernel.family.marginal_design.as_dense_ref().is_some(),
        "the gauged marginal design is memoized"
    );
    let tiled = assert_width_invariant(&fixture, "memoized gauged");
    let former = former_jf(&fixture);
    for (index, (a, b)) in tiled.iter().zip(former.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "memoized gauged J·F entry {index}: tiled {a:e} vs former {b:e}"
        );
    }
}

fn rms_scaled_error(jf: &Array2<f64>, exact: &Array2<f64>, mass: &Array2<f64>) -> (f64, usize) {
    let (mut sum, mut count) = (0.0_f64, 0_usize);
    for ((value, reference), scale) in jf.iter().zip(exact.iter()).zip(mass.iter()) {
        if *scale > 0.0 {
            let error = (value - reference) / scale;
            sum += error * error;
            count += 1;
        }
    }
    ((sum / count.max(1) as f64).sqrt(), count)
}

/// Accuracy on sparse and operator-backed designs, where the tiled GEMM replaces
/// the design's own product. Each entry's error against the double-double
/// reference is scaled by its term mass, and per storage mix:
///
/// 1. The largest scaled error is at most the former build's plus `u = 2⁻⁵³`.
///    On its worst entry each build carries one final rounding of the entry, at
///    most `u·|entry|/mass ≤ u` once scaled, so two builds with equal method
///    error can differ there by up to `2u`; allowing `u` is the stricter bound.
/// 2. The RMS scaled error over the mix's N entries is at most the former
///    build's plus `u/√N`. It measures the method error rather than one entry's
///    rounding: with equal method error, which way each entry's final rounding
///    (at most `u` scaled) falls is a chance reshuffle that moves the RMS by
///    order `u/√N`. A systematic error of `2u` on every entry of one channel
///    raises the RMS by order `u`, far above that margin.
/// 3. The largest scaled error is inside the `γ_m` bound of the longest sum.
#[test]
fn tiled_jf_is_no_less_accurate_on_sparse_and_operator_designs_2337() {
    let unit_roundoff = f64::EPSILON / 2.0;
    for (time_storage, covariate_storage) in [
        (Storage::Sparse, Storage::Gauged),
        (Storage::Dense, Storage::Gauged),
        (Storage::Sparse, Storage::Sparse),
    ] {
        let fixture = fixture(time_storage, covariate_storage);
        let label = format!("time {time_storage:?}, covariates {covariate_storage:?}");
        let tiled = assert_width_invariant(&fixture, &label);
        let former = former_jf(&fixture);
        let (exact, mass) = reference_jf(&fixture);
        let (tiled_worst, former_worst) = (
            worst_scaled_error(&tiled, &exact, &mass),
            worst_scaled_error(&former, &exact, &mass),
        );
        let ((tiled_rms, entries), (former_rms, _)) = (
            rms_scaled_error(&tiled, &exact, &mass),
            rms_scaled_error(&former, &exact, &mass),
        );
        let rms_margin = unit_roundoff / (entries as f64).sqrt();
        // Longest sum: a gauged entry's (1 + GAUGE_RANK)·cols products per design,
        // two designs, each product of three doubles.
        let terms = (3 + 2 * (1 + GAUGE_RANK) * MARGINAL_COLS) as f64;
        let gamma_bound = terms * unit_roundoff / (1.0 - terms * unit_roundoff);
        eprintln!(
            "[2337 J·F] {label}: scaled error worst tiled {tiled_worst:e} former {former_worst:e}, \
             rms tiled {tiled_rms:e} former {former_rms:e}, gamma_m bound {gamma_bound:e}"
        );
        assert!(
            tiled_worst <= former_worst + unit_roundoff,
            "{label}: tiled J·F worst scaled error {tiled_worst:e} exceeds the former build's \
             {former_worst:e} by more than one final rounding"
        );
        assert!(
            tiled_rms <= former_rms + rms_margin,
            "{label}: tiled J·F RMS scaled error {tiled_rms:e} exceeds the former build's \
             {former_rms:e} by more than u/√N = {rms_margin:e}"
        );
        assert!(
            tiled_worst <= gamma_bound,
            "{label}: tiled J·F worst scaled error {tiled_worst:e} exceeds the gamma_m bound \
             {gamma_bound:e}"
        );
    }
}

/// The build under a byte limit: every reservation above `limit` is refused.
/// Also returns the byte requests that were admitted.
fn limited_jf(fixture: &Fixture, workers: usize, limit: usize) -> (Option<Array2<f64>>, Vec<usize>) {
    let governor = gam_runtime::resource::MemoryGovernor::global();
    let admitted = std::sync::Mutex::new(Vec::new());
    let jf = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .expect("test worker pool")
        .install(|| {
            fixture.kernel.tiled_jacobian_action_matrix_within(
                fixture.factor.view(),
                &|bytes| {
                    let reservation = (bytes <= limit)
                        .then(|| governor.try_reserve(bytes, "J·F refusal test").ok())
                        .flatten();
                    if reservation.is_some() {
                        admitted.lock().expect("admitted requests").push(bytes);
                    }
                    reservation
                },
                &|| limit,
            )
        });
    (jf, admitted.into_inner().expect("admitted requests"))
}

/// The per-row route the caller takes when the tiled build declines: one
/// `jacobian_action` per row and factor column.
fn per_row_jf(fixture: &Fixture) -> Array2<f64> {
    let mut jf = Array2::<f64>::zeros((N_ROWS, STATIC_SLOPE_PRIMARIES * FACTOR_RANK));
    for row in 0..N_ROWS {
        for c in 0..FACTOR_RANK {
            let column = fixture.factor.column(c).to_vec();
            let action = RowKernel::jacobian_action(&fixture.kernel, row, &column);
            for (primary, value) in action.iter().enumerate() {
                jf[[row, primary * FACTOR_RANK + c]] = *value;
            }
        }
    }
    jf
}

/// A refused reservation walks the ladder: fewer tiles in flight, then tile
/// heights halved from 256 to 32 rows. At this shape a tile row's buffers take
/// (2 · 7 design columns + 3 · rank 6) · 8 = 256 bytes, and the 1,000 rows split
/// into near-equal slices, so each limit below admits exactly one tile of one
/// rung: 334 rows (slices 333, 333, 334), 143 (142 and 143), 67 (66 and 67) or
/// 33 (32 and 33). On a dense design every rung, at 1 and at 12 workers, gives
/// the unconstrained build's bits, which pins that no slice height moves a
/// GEMM entry's k-order; on an operator-backed design every rung keeps the
/// accuracy bound. With no budget at all the build declines, and the per-row
/// route it hands over to meets the same bound.
#[test]
fn refused_jf_tiles_degrade_to_the_same_result_and_decline_to_the_per_row_route_2337() {
    // (limit, the one admitted request: tile rows · 256 bytes)
    let rungs = [(100_000, 334 * 256), (50_000, 143 * 256), (25_000, 67 * 256), (12_000, 33 * 256)];
    for (time_storage, covariate_storage) in [
        (Storage::Dense, Storage::Dense),
        (Storage::Sparse, Storage::Gauged),
    ] {
        let fixture = fixture(time_storage, covariate_storage);
        let label = format!("time {time_storage:?}, covariates {covariate_storage:?}");
        let dense = matches!((time_storage, covariate_storage), (Storage::Dense, Storage::Dense));
        let unconstrained = tiled_jf(&fixture, 12);
        let (exact, mass) = reference_jf(&fixture);
        let terms = (3 + 2 * (1 + GAUGE_RANK) * MARGINAL_COLS) as f64;
        let gamma_bound = terms * f64::EPSILON / 2.0 / (1.0 - terms * f64::EPSILON / 2.0);
        for (limit, admitted_request) in rungs {
            for workers in [1, 12] {
                let (degraded, admitted) = limited_jf(&fixture, workers, limit);
                let degraded = degraded.expect("one tile of this rung fits the limit");
                assert_eq!(
                    admitted,
                    vec![admitted_request],
                    "{label}: limit {limit} at {workers} workers admitted {admitted:?}"
                );
                if dense {
                    for (index, (a, b)) in degraded.iter().zip(unconstrained.iter()).enumerate() {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "{label} J·F entry {index}: limit {limit} at {workers} workers {a:e} \
                             vs unconstrained {b:e}"
                        );
                    }
                }
                let degraded_error = worst_scaled_error(&degraded, &exact, &mass);
                assert!(
                    degraded_error <= gamma_bound,
                    "{label}: J·F at limit {limit} has worst scaled error {degraded_error:e}, \
                     above {gamma_bound:e}"
                );
            }
        }
        for workers in [1, 12] {
            assert!(
                limited_jf(&fixture, workers, 0).0.is_none(),
                "{label}: the build must decline when no tile is admitted"
            );
        }
        let per_row_error = worst_scaled_error(&per_row_jf(&fixture), &exact, &mass);
        eprintln!("[2337 J·F] {label}: per-row route worst scaled error {per_row_error:e}");
        assert!(
            per_row_error <= gamma_bound,
            "{label}: per-row J·F worst scaled error {per_row_error:e} exceeds {gamma_bound:e}"
        );
    }
}
