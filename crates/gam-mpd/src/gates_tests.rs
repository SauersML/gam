#![cfg(test)]
//! Switching functions on planted gates: the code finds the function that made the labels, and no more.

use super::gates::{Feature, Switch, Unit, base, best, fit, masks, screen};
use ndarray::Array2;

/// A deterministic uniform draw in `[0, 1)`.
fn uniform(seed: usize) -> f64 {
    let mut x = (seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    (x >> 11) as f64 / (1u64 << 53) as f64
}

/// A standard normal draw (Box–Muller).
fn normal(seed: usize) -> f64 {
    let (u, v) = (uniform(2 * seed).max(1e-300), uniform(2 * seed + 1));
    (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}

fn feature(piece: usize) -> Feature {
    Feature { site: 0, piece, lag: 0 }
}

/// A subcomponent on when its amplitude's magnitude is past 1.5 (a two-sided gate no linear switch
/// can say), with 2% of labels flipped.
fn planted(n: usize) -> (Array2<f64>, Vec<bool>) {
    let x = Array2::from_shape_fn((n, 1), |(t, _)| normal(t));
    let y = (0..n).map(|t| (x[[t, 0]].abs() > 1.5) != (uniform(1_000_003 + t) < 0.02)).collect();
    (x, y)
}

#[test]
fn a_two_sided_gate_takes_units_and_beats_the_base_rate() {
    let (x, y) = planted(4000);
    let rate = base(&y);
    let switch = fit(x.view(), &y, &[feature(0)], 0.0, None);
    assert!(!switch.units.is_empty(), "a two-sided gate needs a unit: {switch:?}");
    assert!(switch.total_bits() < 0.5 * rate.total_bits(), "switch {} bits, base rate {}", switch.total_bits(), rate.total_bits());
    // On fresh inputs the switch's decisions match the planted gate up to its label noise.
    let (fresh, truth) = planted(8000);
    let fresh = fresh.slice(ndarray::s![4000.., ..]).to_owned();
    let wrong = (0..4000).filter(|t| switch.on(&[fresh[[*t, 0]]]) != truth[4000 + t]).count();
    assert!(wrong < 4000 * 5 / 100, "{wrong} wrong decisions of 4000");
    // Every coefficient is on the switch's dyadic lattice.
    let step = (-switch.precision as f64).exp2();
    for c in std::iter::once(switch.beta).chain(switch.linear.iter().copied()).chain(switch.units.iter().flat_map(|u| u.w.iter().copied().chain([u.d, u.c]))) {
        assert_eq!((c / step).round() * step, c);
    }
}

#[test]
fn labels_independent_of_the_feature_keep_the_base_rate() {
    let n = 4000;
    let x = Array2::from_shape_fn((n, 1), |(t, _)| normal(t));
    let y: Vec<bool> = (0..n).map(|t| uniform(77 + t) < 0.1).collect();
    let rate = base(&y);
    let switch = fit(x.view(), &y, &[feature(0)], 0.0, None);
    assert!(rate.total_bits() <= switch.total_bits(), "noise bought a switch: base {} switch {}", rate.total_bits(), switch.total_bits());
}

#[test]
fn a_rare_subcomponent_keeps_its_base_rate() {
    // One on-label in 4000 inputs: no function of a feature pays for itself.
    let n = 4000;
    let x = Array2::from_shape_fn((n, 1), |(t, _)| normal(t));
    let mut y = vec![false; n];
    y[17] = true;
    let chosen = best(x.view(), &y, &[feature(0)], (40_000f64).log2());
    assert!(chosen.features.is_empty(), "{chosen:?}");
}

#[test]
fn the_screen_ranks_the_feature_that_drives_the_residual_first() {
    let n = 3000;
    // Candidates: four pure-noise amplitudes and, at index 2, the one the labels follow.
    let candidates = Array2::from_shape_fn((n, 5), |(t, k)| normal(10_000 * k + t));
    let y: Vec<bool> = (0..n).map(|t| candidates[[t, 2]] > 0.8).collect();
    let rate = base(&y);
    let p = 1.0 / (1.0 + (-rate.beta).exp());
    let residuals = Array2::from_shape_fn((n, 1), |(t, _)| if y[t] { 1.0 } else { 0.0 } - p);
    let weights = Array2::from_elem((n, 1), p * (1.0 - p));
    let gains = screen(&candidates, &residuals, &weights).expect("screen");
    let best = (0..5).max_by(|a, b| gains[[*a, 0]].total_cmp(&gains[[*b, 0]])).unwrap_or(0);
    assert_eq!(best, 2, "gains {gains:?}");
    // A feature added to the switch it was screened for lowers its total.
    let switch = fit(candidates.slice(ndarray::s![.., 2..3]).view(), &y, &[feature(2)], (5f64).log2(), None);
    assert!(switch.total_bits() < rate.total_bits() - gains[[2, 0]] / 4.0);
}

#[test]
fn masks_read_lagged_features_from_the_previous_row_and_zero_at_a_sequence_start() {
    // Site 1's one subcomponent is on when site 0's amplitude at the previous row is past 1.
    let switch = Switch {
        features: vec![Feature { site: 0, piece: 0, lag: 1 }],
        beta: -1.0,
        linear: vec![1.0],
        units: vec![Unit { w: vec![0.0], d: 0.0, c: 0.0 }],
        precision: 0,
        function_bits: 0.0,
        listing_bits: 0.0,
    };
    let off = Switch { features: Vec::new(), beta: -1.0, linear: Vec::new(), units: Vec::new(), precision: 0, function_bits: 0.0, listing_bits: 0.0 };
    let amplitudes = vec![Array2::from_shape_vec((4, 1), vec![3.0, 0.0, 3.0, 3.0]).expect("shape"), Array2::zeros((4, 1))];
    // Rows 0..2 one sequence, rows 2..4 another.
    let previous = vec![None, Some(0), None, Some(2)];
    let chosen = masks(&[vec![off], vec![switch]], &amplitudes, &previous);
    assert_eq!(chosen[0].column(0).to_vec(), vec![0.0; 4]);
    assert_eq!(chosen[1].column(0).to_vec(), vec![0.0, 1.0, 0.0, 1.0]);
}
