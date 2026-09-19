//! #2946 pins for the chaos pair route: every row, full and from its own column on, against the closed-form row within
//! the kernel's own band plus the reference's, for zero-bias and biased units, with a corrupted coefficient the pin must
//! reject; the vectorized order passes bit for bit against the portable body; and the speed contract of a row at
//! LLM-like reader geometry. The columns come from their owner, `response::hermite`.

use super::{PairChaosTable, PairRowScratch, chaos_operations, chaos_orders, chaos_orders_body};
use crate::response::hermite::HermiteColumns;
use crate::response::subspace::KnownBlock;
use gam_linalg::roundoff::accumulation_growth;
use gam_math::gaussian_activation::{GaussianActivation, PairKernel, PreactivationPair, pair_kernel};
use gam_math::paired_timing::{SpeedGate, paired_interleaved};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::f64::consts::PI;

const FIXTURE_ORDER: usize = 48;

/// The spread of the biased fixtures' `b_j`, of the size Qwen3-8B layer 18's biases take under its declared law
/// (`|b_j|` p50 0.39, p99 1.44; #2946 comment 5718398668).
const BIAS_SCALE: f64 = 0.75;

fn standard_normal(rng: &mut StdRng) -> f64 {
    // `1 − U[0, 1)` lies in `(0, 1]`, so the logarithm is finite.
    let u1: f64 = 1.0 - rng.random_range(0.0..1.0);
    let u2: f64 = rng.random_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

/// A block's readers, writers and biases, its covariance rows at `P = I` and at a random rank-two frame, and its metric
/// rows.
struct Fixture {
    width: usize,
    /// `width × dim`.
    readers: Array2<f64>,
    /// `outputs × width`.
    writers: Array2<f64>,
    biases: Vec<f64>,
    /// The rank-two frame, `dim × 2` with orthonormal columns.
    frame: Array2<f64>,
    /// `W Wᵀ`, row-major.
    identity_covariances: Vec<f64>,
    /// `(W Q)(W Q)ᵀ` for the rank-two frame `Q`, row-major.
    frame_covariances: Vec<f64>,
    /// The first-order growth of either Gram: a `dim`-term inner product, and at the frame two projections of `dim`
    /// terms followed by a two-term inner product, so at most `2 dim + 2` operations on any path.
    covariance_growth: f64,
    /// `D = Uᵀ U`, row-major.
    metric_rows: Vec<f64>,
}

/// How the fixture's readers are drawn.
#[derive(Clone, Copy)]
enum ReaderLaw {
    /// Readers share one direction, so the correlations spread from near zero to near one, and the second reader
    /// duplicates the first, so one pair sits at correlation one at `P = I`.
    SharedWithDuplicate,
    /// Independent readers, whose correlations concentrate near `1/√dim`, as the Qwen3-8B readers do near
    /// `1/√4096` (#2946 comment 5716095908).
    Independent,
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn fixture(seed: u64, width: usize, dim: usize, outputs: usize, law: ReaderLaw, bias_scale: f64) -> Fixture {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut base = vec![0.0; dim];
    for entry in base.iter_mut() {
        *entry = standard_normal(&mut rng);
    }
    let mut readers = vec![0.0; width * dim];
    for unit in 0..width {
        let loading = match law {
            ReaderLaw::SharedWithDuplicate => standard_normal(&mut rng),
            ReaderLaw::Independent => 0.0,
        };
        for coordinate in 0..dim {
            readers[unit * dim + coordinate] = loading * base[coordinate] + 0.5 * standard_normal(&mut rng);
        }
    }
    if let ReaderLaw::SharedWithDuplicate = law {
        for coordinate in 0..dim {
            readers[dim + coordinate] = readers[coordinate];
        }
    }
    let reader = |unit: usize| &readers[unit * dim..(unit + 1) * dim];
    let mut first = vec![0.0; dim];
    let mut second = vec![0.0; dim];
    for coordinate in 0..dim {
        first[coordinate] = standard_normal(&mut rng);
        second[coordinate] = standard_normal(&mut rng);
    }
    let first_norm = dot(&first, &first).sqrt();
    first.iter_mut().for_each(|entry| *entry /= first_norm);
    let overlap = dot(&first, &second);
    for coordinate in 0..dim {
        second[coordinate] -= overlap * first[coordinate];
    }
    let second_norm = dot(&second, &second).sqrt();
    second.iter_mut().for_each(|entry| *entry /= second_norm);
    let mut writers = vec![0.0; outputs * width];
    for entry in writers.iter_mut() {
        *entry = standard_normal(&mut rng);
    }
    let mut biases = vec![0.0; width];
    for bias in biases.iter_mut() {
        *bias = bias_scale * standard_normal(&mut rng);
    }
    let mut identity_covariances = vec![0.0; width * width];
    let mut frame_covariances = vec![0.0; width * width];
    let mut metric_rows = vec![0.0; width * width];
    for unit in 0..width {
        let unit_frame = [dot(reader(unit), &first), dot(reader(unit), &second)];
        for other in 0..width {
            let other_frame = [dot(reader(other), &first), dot(reader(other), &second)];
            identity_covariances[unit * width + other] = dot(reader(unit), reader(other));
            frame_covariances[unit * width + other] = dot(&unit_frame, &other_frame);
            let mut metric_product = 0.0;
            for output in 0..outputs {
                metric_product += writers[output * width + unit] * writers[output * width + other];
            }
            metric_rows[unit * width + other] = metric_product;
        }
    }
    let frame = Array2::from_shape_fn((dim, 2), |(coordinate, axis)| {
        if axis == 0 { first[coordinate] } else { second[coordinate] }
    });
    Fixture {
        width,
        readers: Array2::from_shape_vec((width, dim), readers).expect("width × dim readers"),
        writers: Array2::from_shape_vec((outputs, width), writers).expect("outputs × width writers"),
        biases,
        frame,
        identity_covariances,
        frame_covariances,
        covariance_growth: accumulation_growth(2 * dim + 2),
        metric_rows,
    }
}

/// The block's closed-form Hermite columns, from their owner, through the fixture's rank-two frame (the columns do not
/// depend on the frame).
fn columns(activation: GaussianActivation, fixture: &Fixture, order: usize) -> HermiteColumns {
    let outputs = fixture.writers.nrows();
    let block = KnownBlock::new(
        fixture.readers.clone(),
        Array1::from_vec(fixture.biases.clone()),
        fixture.writers.clone(),
        Array1::zeros(outputs),
        Array2::<f64>::eye(outputs).view(),
        activation,
    )
    .expect("a finite fixture block is admitted");
    HermiteColumns::for_block(&block, fixture.frame.view(), order).expect("finite fixture units admit every order")
}

fn table(columns: &HermiteColumns) -> PairChaosTable {
    PairChaosTable::for_columns(columns).expect("closed-form fixture columns build a table")
}

/// What one sweep of a table's rows found against the closed-form rows.
struct Tally {
    violations: usize,
    chaos_pairs: usize,
    direct_off_diagonal: usize,
}

/// The rounding the fixture's Gram states for the pair `(unit, other)`: `γ` of its operations times the Cauchy–Schwarz
/// bound `s_j s_k` on the product sum.
fn fixture_rounding(table: &PairChaosTable, covariance_growth: f64, unit: usize, other: usize) -> f64 {
    covariance_growth * (table.variances[unit] * table.variances[other]).sqrt()
}

/// The closed form of one pair, with the rounding its covariance projection may absorb, as `pair_row` passes it.
fn closed_form_pair(
    table: &PairChaosTable,
    unit: usize,
    other: usize,
    covariance: f64,
    covariance_growth: f64,
) -> PairKernel {
    pair_kernel(
        table.activation,
        PreactivationPair {
            mean_x: table.biases[unit],
            mean_y: table.biases[other],
            variance_x: table.variances[unit],
            variance_y: table.variances[other],
            covariance,
            covariance_rounding: fixture_rounding(table, covariance_growth, unit, other),
        },
    )
    .expect("a finite pair is admitted")
}

/// The closed-form row the retained-response operator runs: `Σ_k D_jk [K_σ − m_j m_k]`, leaving `B_jk` in `weights`.
#[inline(never)]
fn closed_form_row(
    table: &PairChaosTable,
    unit: usize,
    covariance_row: &[f64],
    covariance_growth: f64,
    weights: &mut [f64],
) -> f64 {
    let mut sum = 0.0;
    for other in 0..table.width {
        let kernel = closed_form_pair(table, unit, other, covariance_row[other], covariance_growth);
        let metric_product = weights[other];
        sum += metric_product * (kernel.value - table.means[unit] * table.means[other]);
        weights[other] = metric_product * kernel.covariance_derivative;
    }
    sum
}

#[inline(never)]
fn chaos_row(
    table: &PairChaosTable,
    unit: usize,
    covariance_row: &[f64],
    covariance_growth: f64,
    weights: &mut [f64],
    scratch: &mut PairRowScratch,
) -> f64 {
    table
        .pair_row(
            unit,
            0,
            covariance_row,
            |other| fixture_rounding(table, covariance_growth, unit, other),
            weights,
            true,
            scratch,
        )
        .expect("finite fixture pairs are admitted")
        .value
}

/// Compare every row of `table` against the closed-form row, over all columns and over the columns `k ≥ j` a
/// symmetric caller reads. A row value may differ by at most the kernel's reported band plus the closed-form
/// reference's own: per pair the kernel's stated `value_rounding` and `γ_2` of the mean-product difference, and `γ` of
/// the row's terms times the fold's absolute sum. A slope entry may differ by at most its truncation bound and Horner's
/// rounding, each at most `γ` of the chaos operations times `d_j(0) d_k(0)/(s_j s_k)`, the columns' errors weighted by
/// `√n`, and the reference's stated `covariance_derivative_rounding`.
fn sweep(table: &PairChaosTable, columns: &HermiteColumns, fixture: &Fixture, covariances: &[f64]) -> Tally {
    let width = table.width;
    let growth = fixture.covariance_growth;
    let chaos_band = accumulation_growth(chaos_operations(table.order));
    let difference_band = accumulation_growth(2);
    let band = columns.coefficient_band();
    let slope_tails = columns.slope_tails().expect("closed-form columns carry slope tails");
    let mut slope_errors = vec![0.0; width];
    for unit in 0..width {
        let mut weighted = 0.0;
        for degree in 1..=table.order {
            let error = band[[unit, degree]];
            weighted += degree as f64 * error * error;
        }
        slope_errors[unit] = weighted.sqrt();
    }
    let mut scratch = table.row_scratch();
    let mut tally = Tally {
        violations: 0,
        chaos_pairs: 0,
        direct_off_diagonal: 0,
    };
    for unit in 0..width {
        for first in [0, unit] {
            let covariance_row = &covariances[unit * width + first..(unit + 1) * width];
            let metric_row = &fixture.metric_rows[unit * width + first..(unit + 1) * width];
            let mut weights = metric_row.to_vec();
            let chaos = table
                .pair_row(
                    unit,
                    first,
                    covariance_row,
                    |other| fixture_rounding(table, growth, unit, other),
                    &mut weights,
                    true,
                    &mut scratch,
                )
                .expect("finite fixture pairs are admitted");
            if first == 0 {
                tally.direct_off_diagonal += scratch.direct.len() - 1;
                tally.chaos_pairs += width - scratch.direct.len();
            }
            let mut closed_sum = 0.0;
            let mut reference_band = 0.0;
            let mut absolute = 0.0;
            let unit_slope_scale = slope_tails[[unit, 0]];
            for (local, other) in (first..width).enumerate() {
                let kernel = closed_form_pair(table, unit, other, covariance_row[local], growth);
                let mean_product = table.means[unit] * table.means[other];
                let metric_product = metric_row[local];
                let term = metric_product * (kernel.value - mean_product);
                closed_sum += term;
                absolute += term.abs();
                reference_band += metric_product.abs()
                    * (kernel.value_rounding + difference_band * (kernel.value.abs() + mean_product.abs()));
                let other_slope_scale = slope_tails[[other, 0]];
                let series = 2.0 * chaos_band * unit_slope_scale * other_slope_scale
                    + slope_errors[unit] * other_slope_scale
                    + unit_slope_scale * slope_errors[other]
                    + slope_errors[unit] * slope_errors[other];
                let slope_allowed = metric_product.abs()
                    * (series * table.inverse_scales[unit] * table.inverse_scales[other]
                        + kernel.covariance_derivative_rounding);
                let closed_weight = metric_product * kernel.covariance_derivative;
                if (weights[local] - closed_weight).abs() > slope_allowed {
                    tally.violations += 1;
                }
            }
            let allowed = chaos.band + reference_band + accumulation_growth(width - first) * absolute;
            if (chaos.value - closed_sum).abs() > allowed {
                tally.violations += 1;
            }
        }
    }
    tally
}

#[test]
fn chaos_rows_match_the_closed_form_rows_within_the_derived_band() {
    for bias_scale in [0.0, BIAS_SCALE] {
        let fixture = fixture(0x2946_7117, 24, 6, 3, ReaderLaw::SharedWithDuplicate, bias_scale);
        for activation in [GaussianActivation::Relu, GaussianActivation::ExactGelu] {
            let columns = columns(activation, &fixture, FIXTURE_ORDER);
            let table = table(&columns);
            for (label, covariances) in [
                ("P = I", &fixture.identity_covariances),
                ("rank-two frame", &fixture.frame_covariances),
            ] {
                let tally = sweep(&table, &columns, &fixture, covariances);
                assert_eq!(
                    tally.violations, 0,
                    "{activation:?} at {label}, bias scale {bias_scale}: {} row or slope entries left the derived band",
                    tally.violations
                );
                assert!(
                    tally.chaos_pairs > 0,
                    "{activation:?} at {label}, bias scale {bias_scale}: no pair took the chaos route, so the pin \
                     checked nothing of it"
                );
            }
            let identity = sweep(&table, &columns, &fixture, &fixture.identity_covariances);
            assert!(
                identity.direct_off_diagonal > 0,
                "{activation:?}, bias scale {bias_scale}: the duplicated reader pair sits at correlation one at P = I \
                 and must take the closed form"
            );
            // Positive control: a chaos coefficient of order two scaled by 5/4 moves every chaos pair with a nonzero
            // correlation by `(25/16 − 1) ρ² a₂ a₂`, far outside the band, and the pin must say so.
            let mut corrupted = table.clone();
            for unit in 0..corrupted.width {
                corrupted.coefficients[corrupted.width + unit] *= 1.25;
            }
            let corrupted_tally = sweep(&corrupted, &columns, &fixture, &fixture.identity_covariances);
            assert!(
                corrupted_tally.violations > 0,
                "{activation:?}, bias scale {bias_scale}: a corrupted order-two coefficient passed the pin, so the \
                 pin cannot fail"
            );
        }
    }
}

/// `for_columns` keeps only the orders up to the saturation order. Its rows must write the value and `B_jk` words of
/// the table built at the columns' full order, for every unit, for smooth and kinked activations with and without bias:
/// no row stops above the saturation order, so the orders past it move no route and no value.
///
/// The band is not compared word for word. It is an upper bound on a row's error, and the full table charges the
/// coefficient errors of every order it carries, so the two tables' bands may differ in their last bits (gate 1288792:
/// ExactGelu, bias 0, unit 0, equal value bits, band bits 4404238933005659811 kept against 4404238933005659817 full)
/// and both be valid. What each band must do is bound its own table's measured error: [`sweep`] compares every row of
/// each table, full and from its own column on, against the closed-form row within that table's band plus the
/// reference's, and must find no violation. The corrupted-coefficient control in
/// `chaos_rows_match_the_closed_form_rows_within_the_derived_band` shows that comparison can fail. A row whose band
/// swallows its whole value would bound nothing, so some row's value must exceed its band.
#[test]
fn a_table_kept_to_its_saturation_order_writes_the_full_tables_values_2946() {
    let mut informative = 0usize;
    let mut saturated = 0usize;
    for bias_scale in [0.0, BIAS_SCALE] {
        let fixture = fixture(0x2946_5A7D, 48, 2, 4, ReaderLaw::Independent, bias_scale);
        let width = fixture.width;
        let growth = fixture.covariance_growth;
        for activation in [GaussianActivation::Relu, GaussianActivation::ExactGelu] {
            // Twice the covering order puts the saturation order strictly below the columns' order where one exists; a
            // block without one (ReLU) keeps the fixture order, where the kept and full tables coincide.
            let probe = columns(activation, &fixture, 1);
            let order = PairChaosTable::covering_column_order(&probe)
                .expect("closed-form columns")
                .map_or(FIXTURE_ORDER, |covering| 2 * covering);
            let columns = columns(activation, &fixture, order);
            let kept = table(&columns);
            let biases = columns.closed_form_units().expect("closed-form columns").1.to_vec();
            let full = PairChaosTable::from_columns(
                activation,
                &biases,
                &columns.scales().to_vec(),
                columns.coefficients(),
                columns.coefficient_band(),
                columns.value_tails().expect("closed-form tails"),
                columns.slope_tails().expect("closed-form tails"),
            )
            .expect("the full-order table builds");
            assert!(
                kept.order() <= full.order() && kept.order() == full.saturation_order(),
                "{activation:?}, bias scale {bias_scale}: kept order {} is not min(saturation {}, columns {})",
                kept.order(),
                full.saturation_order(),
                full.order()
            );
            if kept.order() < full.order() {
                saturated += 1;
            }
            let mut kept_scratch = kept.row_scratch();
            let mut full_scratch = full.row_scratch();
            for unit in 0..width {
                let covariance_row = &fixture.identity_covariances[unit * width..(unit + 1) * width];
                let metric_row = &fixture.metric_rows[unit * width..(unit + 1) * width];
                let mut kept_weights = metric_row.to_vec();
                let mut full_weights = metric_row.to_vec();
                let kept_row = kept
                    .pair_row(
                        unit,
                        0,
                        covariance_row,
                        |other| fixture_rounding(&kept, growth, unit, other),
                        &mut kept_weights,
                        true,
                        &mut kept_scratch,
                    )
                    .expect("finite fixture pairs are admitted");
                let full_row = full
                    .pair_row(
                        unit,
                        0,
                        covariance_row,
                        |other| fixture_rounding(&full, growth, unit, other),
                        &mut full_weights,
                        true,
                        &mut full_scratch,
                    )
                    .expect("finite fixture pairs are admitted");
                assert_eq!(
                    kept_row.value.to_bits(),
                    full_row.value.to_bits(),
                    "{activation:?}, bias scale {bias_scale}, unit {unit}: the kept table's row value differs from the \
                     full table's"
                );
                assert_eq!(
                    kept_weights.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                    full_weights.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                    "{activation:?}, bias scale {bias_scale}, unit {unit}: the kept table's B_jk differ from the full table's"
                );
                if kept_row.value.abs() > kept_row.band && full_row.value.abs() > full_row.band {
                    informative += 1;
                }
            }
            for (label, candidate) in [("kept", &kept), ("full", &full)] {
                let tally = sweep(candidate, &columns, &fixture, &fixture.identity_covariances);
                assert_eq!(
                    tally.violations, 0,
                    "{activation:?}, bias scale {bias_scale}: the {label} table's band does not bound its own rows' \
                     error against the closed-form rows"
                );
            }
        }
    }
    assert!(
        saturated > 0,
        "no fixture saturated below the columns' order, so the pin compared only identical tables"
    );
    assert!(
        informative > 0,
        "every row's band swallowed its whole value, so the bands bound nothing"
    );
}

/// Built to its covering column order, an exact GELU block's table saturates at or below it: every unit meets both
/// chaos bounds at correlation one at the kept order. ReLU's slope envelope is `+∞`, so it has no covering order.
#[test]
fn columns_built_to_the_covering_order_give_a_saturated_table_2946() {
    for bias_scale in [0.0, BIAS_SCALE] {
        let fixture = fixture(0x2946_C0DE, 48, 2, 4, ReaderLaw::Independent, bias_scale);
        let probe = columns(GaussianActivation::ExactGelu, &fixture, 1);
        let order = PairChaosTable::covering_column_order(&probe)
            .expect("closed-form columns")
            .expect("exact GELU's envelopes decay geometrically, so a covering order exists");
        let built = columns(GaussianActivation::ExactGelu, &fixture, order);
        let table = table(&built);
        assert!(
            table.order() <= order
                && (0..table.width).all(|unit| table.meets_bounds(unit, table.order(), 1.0, true)),
            "bias scale {bias_scale}: columns built to the covering order {order} gave a table at order {} that does \
             not meet both bounds at correlation one",
            table.order()
        );
        let relu = columns(GaussianActivation::Relu, &fixture, 1);
        assert_eq!(
            PairChaosTable::covering_column_order(&relu).expect("closed-form columns"),
            None,
            "bias scale {bias_scale}: ReLU's infinite slope envelope admits no covering order"
        );
    }
}

/// The order `pair_row` stopped at for a row over the columns `k ≥ first`: the first order meeting the bounds at the
/// row's largest chaos correlation.
fn stopping_order(table: &PairChaosTable, unit: usize, first: usize, scratch: &PairRowScratch) -> usize {
    let largest = (first..table.width)
        .zip(&scratch.correlations[..table.width - first])
        .filter(|(other, _)| !scratch.direct.contains(other))
        .fold(0.0f64, |largest, (_, correlation)| largest.max(correlation.abs()));
    table.row_order(unit, largest, true)
}

/// A row read from its own column on writes, for every pair it holds, the words of that pair in the full row whenever
/// both rows stop at the same order: the pair is the same function of its own correlation.
#[test]
fn a_row_from_its_own_column_writes_the_full_rows_pairs_2946() {
    let fixture = fixture(0x2946_4A1F, 64, 16, 4, ReaderLaw::Independent, BIAS_SCALE);
    let width = fixture.width;
    let columns = columns(GaussianActivation::ExactGelu, &fixture, FIXTURE_ORDER);
    let table = table(&columns);
    let growth = fixture.covariance_growth;
    let mut full_scratch = table.row_scratch();
    let mut halved_scratch = table.row_scratch();
    let mut compared = 0usize;
    for unit in 0..width {
        let covariance_row = &fixture.identity_covariances[unit * width..(unit + 1) * width];
        let metric_row = &fixture.metric_rows[unit * width..(unit + 1) * width];
        let mut full_weights = metric_row.to_vec();
        table
            .pair_row(
                unit,
                0,
                covariance_row,
                |other| fixture_rounding(&table, growth, unit, other),
                &mut full_weights,
                true,
                &mut full_scratch,
            )
            .expect("finite fixture pairs are admitted");
        let mut halved_weights = metric_row[unit..].to_vec();
        table
            .pair_row(
                unit,
                unit,
                &covariance_row[unit..],
                |other| fixture_rounding(&table, growth, unit, other),
                &mut halved_weights,
                true,
                &mut halved_scratch,
            )
            .expect("finite fixture pairs are admitted");
        if stopping_order(&table, unit, 0, &full_scratch) != stopping_order(&table, unit, unit, &halved_scratch) {
            continue;
        }
        for (local, other) in (unit..width).enumerate() {
            assert_eq!(
                (halved_scratch.values[local].to_bits(), halved_weights[local].to_bits()),
                (full_scratch.values[other].to_bits(), full_weights[other].to_bits()),
                "unit {unit}, pair {other}: the row from column {unit} differs from the full row at the same order"
            );
            compared += 1;
        }
    }
    assert!(
        compared > width,
        "only {compared} pairs shared an order between the two rows, so the pin checked too little"
    );
}
#[test]
fn vectorized_order_passes_write_the_portable_words() {
    let mut rng = StdRng::seed_from_u64(0x2946_0A5E);
    // An odd width leaves a remainder after every SIMD lane width.
    let width = 37;
    let order = 9;
    let mut unit_coefficients = vec![0.0; order];
    for entry in unit_coefficients.iter_mut() {
        *entry = standard_normal(&mut rng);
    }
    let mut columns = vec![0.0; order * width];
    for entry in columns.iter_mut() {
        *entry = standard_normal(&mut rng);
    }
    let mut correlations = vec![0.0; width];
    for entry in correlations.iter_mut() {
        *entry = rng.random_range(-1.0..1.0);
    }
    // A row from column 5 on shifts every lane against the full row, and its own length leaves another remainder.
    for (gradient, first) in [(false, 0), (true, 0), (false, 5), (true, 5)] {
        let count = width - first;
        let correlations = &correlations[first..];
        let mut dispatched = [vec![0.0; count], vec![0.0; count]];
        let mut portable = [vec![0.0; count], vec![0.0; count]];
        let [dispatched_values, dispatched_slopes] = &mut dispatched;
        chaos_orders(
            &unit_coefficients,
            &columns,
            width,
            first,
            correlations,
            dispatched_values,
            dispatched_slopes,
            gradient,
        );
        let [portable_values, portable_slopes] = &mut portable;
        chaos_orders_body(
            &unit_coefficients,
            &columns,
            width,
            first,
            correlations,
            portable_values,
            portable_slopes,
            gradient,
        );
        for k in 0..count {
            assert_eq!(
                dispatched[0][k].to_bits(),
                portable[0][k].to_bits(),
                "value {k} differs between the dispatched and portable passes (gradient {gradient}, first {first})"
            );
            assert_eq!(
                dispatched[1][k].to_bits(),
                portable[1][k].to_bits(),
                "slope {k} differs between the dispatched and portable passes (gradient {gradient}, first {first})"
            );
        }
        assert!(
            dispatched[0].iter().any(|value| *value != 0.0),
            "the fixture wrote no values, so the comparison checked nothing"
        );
    }
}

/// The chaos row must beat the closed-form row it replaces, on a row at LLM-like reader geometry, for zero-bias units
/// and for biased units whose closed form needs the bivariate normal CDF. Parity is pinned in every build first; the
/// timing runs only where the codegen is the shipped one.
#[test]
fn chaos_row_is_faster_than_the_closed_form_row() {
    let mut blocks = Vec::new();
    for bias_scale in [0.0, BIAS_SCALE] {
        let fixture = fixture(0x2946_5EED, 1024, 256, 16, ReaderLaw::Independent, bias_scale);
        let mut tables = Vec::new();
        for activation in [GaussianActivation::Relu, GaussianActivation::ExactGelu] {
            let columns = columns(activation, &fixture, FIXTURE_ORDER);
            let table = table(&columns);
            let tally = sweep(&table, &columns, &fixture, &fixture.identity_covariances);
            assert_eq!(
                tally.violations, 0,
                "{activation:?}, bias scale {bias_scale}: the timed rows left the derived band, so their timing \
                 compares different results"
            );
            assert!(
                tally.chaos_pairs > tally.direct_off_diagonal,
                "{activation:?}, bias scale {bias_scale}: at independent-reader geometry most pairs must take the \
                 chaos route"
            );
            tables.push((activation, table));
        }
        blocks.push((bias_scale, fixture, tables));
    }
    if cfg!(debug_assertions) {
        return; // dev lane: the codegen is not the shipped one
    }
    let mut gate = SpeedGate::open("FR-PERF-CHAOS-ROW-2946");
    for (bias_scale, fixture, tables) in &blocks {
        let width = fixture.width;
        let unit = width / 2;
        let growth = fixture.covariance_growth;
        let covariance_row = &fixture.identity_covariances[unit * width..(unit + 1) * width];
        let metric_row = &fixture.metric_rows[unit * width..(unit + 1) * width];
        for (activation, table) in tables {
            let mut chaos_weights = metric_row.to_vec();
            let mut scratch = table.row_scratch();
            let mut closed_weights = metric_row.to_vec();
            let timing = paired_interleaved(
                15,
                200,
                0x2946_71AE,
                |nudge| {
                    chaos_weights.copy_from_slice(metric_row);
                    chaos_weights[0] += nudge;
                    chaos_row(table, unit, covariance_row, growth, &mut chaos_weights, &mut scratch)
                },
                |nudge| {
                    closed_weights.copy_from_slice(metric_row);
                    closed_weights[0] += nudge;
                    closed_form_row(table, unit, covariance_row, growth, &mut closed_weights)
                },
            );
            let cell = format!("{activation:?} bias scale {bias_scale} row 1024 order {}", table.order());
            gate.faster(&cell, &timing, "chaos", "closed_form");
        }
    }
    gate.finish();
}
