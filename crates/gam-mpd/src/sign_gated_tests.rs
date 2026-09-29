#![cfg(test)]
//! Known answers and controls for the sign-gated split: saturated gates drop nothing,
//! gates at the peak attain the operator bound, gates just above zero make the correction
//! cancel half the law, and random blocks keep the executed identity inside its bands.

use super::*;
use crate::supports::EvidenceStatus;
use crate::test_support::test_governor;
use ndarray::{Array1, Array2, array};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn silu(t: f64) -> f64 {
    t * logistic(t)
}

/// A block whose unit `n` reads coordinate `n mod d` of the input through `gate_scale`, and
/// whose up and down weights are the given ones.
fn coordinate_block(gate_scale: f64, up: Array2<f64>, down: Array2<f64>, width: usize) -> ResidualSwiglu {
    let units = up.nrows();
    let gate = Array2::from_shape_fn((units, width), |(n, j)| if j == n % width { gate_scale } else { 0.0 });
    ResidualSwiglu {
        units: SwigluUnits::new(gate, up, down).expect("block"),
        input_norm: None,
        output_norm: None,
    }
}

fn sign_rows(rows: usize, width: usize) -> Array2<f64> {
    Array2::from_shape_fn((rows, width), |(t, j)| if (t + 2 * j) % 3 == 0 { -1.0 } else { 1.0 })
}

fn random_matrix(rng: &mut StdRng, shape: (usize, usize), scale: f64) -> Array2<f64> {
    Array2::from_shape_simple_fn(shape, || scale * (rng.random::<f64>() * 2.0 - 1.0))
}

/// The correction `R` of the rows by plain loops, for comparison with the banded execution.
fn naive_correction(block: &ResidualSwiglu, x: &Array2<f64>) -> Array2<f64> {
    let g = x.dot(&block.units.gate().t());
    let u = x.dot(&block.units.up().t());
    let c = Array2::from_shape_fn(g.raw_dim(), |index| (silu(g[index]) - g[index].max(0.0)) * u[index]);
    c.dot(&block.units.down().t())
}

#[test]
fn the_correction_is_even_nonpositive_half_lipschitz_and_peaks_at_lambert_w() {
    let peak = correction_peak();
    let step = 1e-3;
    let mut previous = relu_correction(-40.0);
    for k in -39_999..=40_000 {
        let t = f64::from(k) * step;
        let e = relu_correction(t);
        assert_eq!(e.to_bits(), relu_correction(-t).to_bits(), "even at {t}");
        assert!(e <= 0.0 && -e <= peak, "|e({t})| = {} above W(1/e)", -e);
        let direct = silu(t) - t.max(0.0);
        assert!((direct - e).abs() <= 8.0 * f64::EPSILON * (silu(t).abs() + t.abs()) + f64::MIN_POSITIVE);
        let slope = (e - previous).abs() / step;
        assert!(slope <= 0.5 * (1.0 + 1e-9), "slope {slope} at {t}");
        previous = e;
    }
    let at_peak = -relu_correction(CORRECTION_PEAK_GATE);
    assert!((at_peak - CORRECTION_PEAK_NEAREST).abs() <= 4.0 * f64::EPSILON * CORRECTION_PEAK_NEAREST);
    for offset in [1e-4, 1e-2, 0.5] {
        assert!(-relu_correction(CORRECTION_PEAK_GATE + offset) < at_peak);
        assert!(-relu_correction(CORRECTION_PEAK_GATE - offset) < at_peak);
    }
    // W(1/e) solves ω e^ω = 1/e.
    let omega = CORRECTION_PEAK_NEAREST;
    assert!((omega * omega.exp() - (-1.0_f64).exp()).abs() <= 4.0 * f64::EPSILON);
}

#[test]
fn saturated_gates_drop_nothing_within_the_bound() {
    let (width, units) = (3, 6);
    let mut rng = StdRng::seed_from_u64(11);
    let block = coordinate_block(60.0, random_matrix(&mut rng, (units, width), 1.0), random_matrix(&mut rng, (4, units), 1.0), width);
    let rows = sign_rows(5, width);
    let split = block.split(test_governor(), rows.view()).expect("split");
    for (row, (correction, native)) in split.correction_norms.iter().zip(&split.native_norms).enumerate() {
        // |e(60)| = 60 σ(−60) < 1e-24: the computed correction is that small, and its enclosure is the
        // gate read's rounding band carried through the Lipschitz constant ½.
        let computed = split.correction.values.row(row).iter().fold(0.0_f64, |largest, value| largest.max(value.abs()));
        assert!(computed <= 1e-20 * native.lower);
        assert!(correction.upper <= 1e-12 * native.lower, "row {row}: {correction:?} vs {native:?}");
        assert!(split.operator_bounds[row] <= 1e-12 * native.lower);
    }
    for signs in &split.gate_signs {
        assert_eq!(signs.active + signs.inactive, units);
        assert_eq!(signs.undecided, 0);
    }
    let explained = split.shares.law_explained_variance.expect("variance");
    // One directed rounding step below 1 is all that separates the interval from 1.
    assert!(explained.lower >= 1.0 - 4.0 * f64::EPSILON && explained.upper >= 1.0, "{explained:?}");
    assert!(split.identity_excess <= 0.0);
}

#[test]
fn gates_at_the_peak_attain_the_operator_bound() {
    // W_d = I, so R = c = e(±(1 + ω)) u = −ω u and σ₁ = 1: the norm-only bound is attained.
    let (width, units) = (4, 8);
    let mut rng = StdRng::seed_from_u64(12);
    let up = random_matrix(&mut rng, (units, width), 1.0);
    let block = coordinate_block(CORRECTION_PEAK_GATE, up.clone(), Array2::eye(units), width);
    let rows = sign_rows(6, width);
    let split = block.split(test_governor(), rows.view()).expect("split");
    let u = rows.dot(&up.t());
    for (row, correction) in split.correction_norms.iter().enumerate() {
        let expected = CORRECTION_PEAK_NEAREST * u.row(row).dot(&u.row(row)).sqrt();
        assert!(correction.lower <= expected * (1.0 + 1e-14) && expected <= correction.upper * (1.0 + 1e-14));
        let tightness = split.operator_bounds[row] / correction.lower;
        assert!((1.0..=1.0 + 1e-12).contains(&tightness), "row {row}: operator bound over ‖R‖ = {tightness}");
    }
    assert!(split.down_gains.operator.lower <= 1.0 && 1.0 <= split.down_gains.operator.upper);
}

#[test]
fn gates_just_above_zero_make_the_correction_cancel_half_the_law() {
    // g = 1e-8 > 0: s(g) ≈ g/2, relu(g) = g, e(g) ≈ −g/2, so P ≈ 2F and R ≈ −F.
    let (width, units) = (3, 5);
    let mut rng = StdRng::seed_from_u64(13);
    let block = coordinate_block(1e-8, random_matrix(&mut rng, (units, width), 1.0), random_matrix(&mut rng, (3, units), 1.0), width);
    let rows = Array2::from_elem((4, width), 1.0) + &random_matrix(&mut rng, (4, width), 0.5).mapv(f64::abs);
    let split = block.split(test_governor(), rows.view()).expect("split");
    let ratio = split.shares.correction_over_native.expect("ratio");
    assert!(ratio.lower > 1.0 - 1e-7 && ratio.upper < 1.0 + 1e-7, "{ratio:?}");
    let cosine = split.shares.law_native_cosine.expect("cosine");
    assert!(cosine.lower > 1.0 - 1e-12);
    for (law, native) in split.law_norms.iter().zip(&split.native_norms) {
        assert!((law.lower / native.upper - 2.0).abs() < 1e-7);
    }
    for signs in &split.gate_signs {
        assert_eq!(signs.active, units);
    }
}

fn random_prenorm_block(rng: &mut StdRng, width: usize, units: usize, out: usize) -> ResidualSwiglu {
    let gain = Array1::from_shape_simple_fn(width, || 0.5 + rng.random::<f64>());
    ResidualSwiglu {
        units: SwigluUnits::new(
            random_matrix(rng, (units, width), 1.0),
            random_matrix(rng, (units, width), 1.0),
            random_matrix(rng, (out, units), 0.3),
        )
        .expect("block"),
        input_norm: Some(LayerRmsNorm::native(1e-6, gain)),
        output_norm: None,
    }
}

#[test]
fn random_blocks_keep_the_identity_bands_bounds_and_shares_sound() {
    let governor = test_governor();
    let mut rng = StdRng::seed_from_u64(14);
    for (seed, scale) in [(0_u64, 1e-3), (1, 1.0), (2, 1e3)] {
        let (width, units, out) = (6, 17, 6);
        let block = random_prenorm_block(&mut StdRng::seed_from_u64(100 + seed), width, units, out);
        let residual = random_matrix(&mut rng, (9, width), scale);
        let split = block.split(governor, residual.view()).expect("split");
        assert!(split.identity_excess <= 0.0);
        let norm = block.input_norm.as_ref().expect("norm");
        let x = Array2::from_shape_fn(residual.raw_dim(), |(t, j)| {
            let row = residual.row(t);
            norm.gain[j] * row[j] / (row.dot(&row) / width as f64 + norm.epsilon).sqrt()
        });
        let naive = naive_correction(&block, &x);
        let bound = split.region_bound.expect("region bound");
        for row in 0..residual.nrows() {
            let correction = split.correction_norms[row];
            let width_relative = (correction.upper - correction.lower) / correction.upper;
            assert!(width_relative < 1e-10, "enclosure too wide: {correction:?}");
            let naive_norm = naive.row(row).dot(&naive.row(row)).sqrt();
            // The naive loops err by far less than the bands at these sizes.
            assert!(correction.lower * (1.0 - 1e-12) <= naive_norm && naive_norm <= correction.upper * (1.0 + 1e-12));
            assert!(split.operator_bounds[row] >= correction.upper);
            assert!(bound >= correction.upper);
            let signs = split.gate_signs[row];
            assert_eq!(signs.active + signs.inactive + signs.undecided, units);
            assert!(split.law_units90[row] >= 1 && split.law_units90[row] <= units);
        }
        // The write is F without an output norm, so both shares agree.
        let (law, write) = (
            split.shares.law_explained_variance.expect("law"),
            split.shares.write_explained_variance.expect("write"),
        );
        assert!(law.lower <= write.upper && write.lower <= law.upper);
        assert!(law.upper - law.lower < 1e-8);
    }
}

#[test]
fn a_correction_moved_beyond_its_band_is_refused() {
    let mut rng = StdRng::seed_from_u64(15);
    let block = random_prenorm_block(&mut rng, 5, 11, 5);
    let residual = random_matrix(&mut rng, (4, 5), 1.0);
    let split = block.split(test_governor(), residual.view()).expect("split");
    assert!(identity_excess(&split.native, &split.law, &split.correction).is_ok());
    let mut moved = split.correction.clone();
    moved.values[[2, 3]] += 10.0 * (moved.radius[[2, 3]] + split.native.radius[[2, 3]] + split.law.radius[[2, 3]]) + 1e-12;
    assert!(matches!(
        identity_excess(&split.native, &split.law, &moved),
        Err(SignGatedError::Identity { row: 2, column: 3, .. })
    ));
}

#[test]
fn a_post_norm_block_changes_its_write_by_the_normalized_difference() {
    let mut rng = StdRng::seed_from_u64(16);
    let (width, units) = (6, 13);
    let mut block = random_prenorm_block(&mut rng, width, units, width);
    block.input_norm = None;
    let gain = Array1::from_shape_simple_fn(width, || 0.5 + rng.random::<f64>());
    block.output_norm = Some(LayerRmsNorm::native(1e-6, gain.clone()));
    let residual = random_matrix(&mut rng, (7, width), 2.0);
    let split = block.split(test_governor(), residual.view()).expect("split");
    let normalize = |rows: &Array2<f64>| {
        Array2::from_shape_fn(rows.raw_dim(), |(t, j)| {
            let row = rows.row(t);
            gain[j] * row[j] / (row.dot(&row) / width as f64 + 1e-6).sqrt()
        })
    };
    let naive = normalize(&split.law.values) - normalize(&split.native.values);
    for (index, &value) in naive.indexed_iter() {
        let gap = (value - split.change.values[index]).abs();
        assert!(gap <= split.change.radius[index] + 1e-15, "{index:?}: {gap} vs {}", split.change.radius[index]);
    }
    assert!(split.region_bound.is_none());
}

#[test]
fn the_change_supremum_is_exact_or_a_counterexample_to_its_tolerance() {
    let mut rng = StdRng::seed_from_u64(17);
    let block = random_prenorm_block(&mut rng, 5, 9, 5);
    let residual = random_matrix(&mut rng, (6, 5), 1.0);
    let split = block.split(test_governor(), residual.view()).expect("split");
    let largest = split.change_norms.iter().map(|interval| interval.upper).fold(0.0, f64::max);
    match split.change_supremum(None).expect("supremum") {
        EvidenceStatus::Exact { value, numerical_error, .. } => {
            assert!(value <= largest && largest <= value + 2.0 * numerical_error);
        }
        other => panic!("expected an exact supremum, got {other:?}"),
    }
    assert!(matches!(
        split.change_supremum(Some(0.5 * largest)).expect("supremum"),
        EvidenceStatus::Counterexample { .. }
    ));
    assert!(matches!(split.change_supremum(Some(2.0 * largest)).expect("supremum"), EvidenceStatus::Exact { .. }));
}

fn kl(reference: ArrayView1<'_, f64>, candidate: ArrayView1<'_, f64>) -> f64 {
    let lse = |row: ArrayView1<'_, f64>| {
        let top = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        top + row.iter().map(|value| (value - top).exp()).sum::<f64>().ln()
    };
    let (a, b) = (lse(reference), lse(candidate));
    reference.iter().zip(candidate.iter()).map(|(&r, &c)| (r - a).exp() * ((r - a) - (c - b))).sum()
}

#[test]
fn a_last_layer_replacement_is_verified_at_the_logits() {
    let governor = test_governor();
    let mut rng = StdRng::seed_from_u64(18);
    let (width, units, vocabulary) = (6, 14, 9);
    let block = random_prenorm_block(&mut rng, width, units, width);
    let residual = random_matrix(&mut rng, (7, width), 1.0);
    let split = block.split(governor, residual.view()).expect("split");
    let gain = Array1::from_shape_simple_fn(width, || 0.5 + rng.random::<f64>());
    let readout = Readout { norm: LayerRmsNorm::native(1e-6, gain.clone()), unembedding: random_matrix(&mut rng, (vocabulary, width), 2.0) };
    let tolerance = Tolerance { kl: 10.0, centred_logit_gap: 100.0 };
    let verification = replacement_readout(governor, &block, residual.view(), &split, &readout, &tolerance, 3).expect("readout");
    assert!(verification.certified_within(&tolerance));
    assert_eq!(verification.rows.len(), residual.nrows());
    let logits = |written: &Array2<f64>| {
        let out = &residual + written;
        let normed = Array2::from_shape_fn(out.raw_dim(), |(t, j)| {
            let row = out.row(t);
            gain[j] * row[j] / (row.dot(&row) / width as f64 + 1e-6).sqrt()
        });
        normed.dot(&readout.unembedding.t())
    };
    let (native, replaced) = (logits(&split.native_write.values), logits(&split.replaced_write.values));
    for (witness, comparison) in &verification.rows {
        let row = witness.input * 3 + witness.row;
        let naive = kl(native.row(row), replaced.row(row));
        let RowValue::Resolved { value, numerical_error } = RowValue::of(&comparison.forward_kl) else {
            panic!("row {row} unresolved");
        };
        assert!((naive - value).abs() <= numerical_error + 1e-14, "row {row}: {naive} vs {value} ± {numerical_error}");
    }
    let tight = Tolerance { kl: 0.0, centred_logit_gap: 100.0 };
    let refuted = replacement_readout(governor, &block, residual.view(), &split, &readout, &tight, 4).expect("readout");
    assert!(!refuted.certified_within(&tight));
    let short = Readout { norm: readout.norm.clone(), unembedding: array![[1.0, 2.0]] };
    assert!(matches!(
        replacement_readout(governor, &block, residual.view(), &split, &short, &tolerance, 3),
        Err(SignGatedError::Shape { .. })
    ));
}

#[test]
fn malformed_rows_are_refused() {
    let mut rng = StdRng::seed_from_u64(19);
    let block = random_prenorm_block(&mut rng, 4, 6, 4);
    let governor = test_governor();
    assert!(matches!(block.split(governor, Array2::<f64>::zeros((0, 4)).view()), Err(SignGatedError::EmptyRows)));
    assert!(matches!(block.split(governor, Array2::<f64>::zeros((2, 3)).view()), Err(SignGatedError::Shape { .. })));
    let mut bad = Array2::<f64>::ones((2, 4));
    bad[[1, 2]] = f64::NAN;
    assert!(matches!(block.split(governor, bad.view()), Err(SignGatedError::NonFinite { .. })));
}
