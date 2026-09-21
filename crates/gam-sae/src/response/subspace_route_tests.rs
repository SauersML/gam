#![cfg(test)]
//! #2946 route pins for the retained-response operator's pair pass: the halved pass over `j ≤ k` against the full
//! double sum, and the streamed reader Gram against the cached one, on a block wide enough to span three reader-Gram
//! tiles.
//!
//! The fixture's readers and writers are small integers and its metric is dyadic, so every covariance `w_jᵀ w_k` and
//! every `D_jk` is formed exactly by any product shape. A route difference is then confined to the pair kernel's
//! evaluation order and to summation, and both have published bounds. Every pin prints its numbers whether it passes
//! or fails.

use super::{CoordinateFormation, CovarianceFormation, KnownBlock, covariance_rounding_band};
use crate::response::reader_gram::upper_tile_rows;
use gam_linalg::roundoff::accumulation_growth;
use gam_math::gaussian_activation::{GaussianActivation, PreactivationPair, pair_kernel};
use ndarray::{Array1, Array2, array};

/// The fixture width: three reader-Gram tiles, the last one short.
const MULTI_TILE_WIDTH: usize = 1449;

/// A block of `width` units with integer readers in `{−3, …, 3}³` (never the zero row), integer writers in two
/// outputs, a dyadic metric, and zero biases, so every pair runs the closed-form zero-mean ReLU kernel.
fn integer_block(width: usize) -> KnownBlock {
    let readers = Array2::from_shape_fn((width, 3), |(unit, input)| {
        ((unit * (input + 2) + input) % 7) as f64 - 3.0
    });
    let writers = Array2::from_shape_fn((2, width), |(output, unit)| ((unit * 5 + output * 3) % 7) as f64 - 3.0);
    KnownBlock::new(
        readers,
        Array1::zeros(width),
        writers,
        Array1::zeros(2),
        array![[2.0, 0.5], [0.5, 1.0]].view(),
        GaussianActivation::Relu,
    )
    .expect("a finite integer block with a symmetric positive definite metric")
}

#[test]
fn a_streamed_reader_gram_reproduces_the_cached_pass_bit_for_bit_across_tiles() {
    let tile = upper_tile_rows(MULTI_TILE_WIDTH);
    let cached = integer_block(MULTI_TILE_WIDTH);
    let mut streamed = cached.clone();
    streamed.units.reader_gram = None;
    // A frame with a non-dyadic turn, so the gradient route forms rounded covariances.
    let frame = array![[0.6, 0.0], [0.8, 0.0], [0.0, 1.0]];
    let cached_gradient = cached
        .explained_variance_gradient(frame.view())
        .expect("an orthonormal frame");
    let streamed_gradient = streamed
        .explained_variance_gradient(frame.view())
        .expect("an orthonormal frame");
    let streamed_total = streamed
        .units
        .pair_pass(streamed.units.readers.view(), CoordinateFormation::Copied, None)
        .expect("the streamed V(I) pass")
        .energy;
    let resident = cached
        .units
        .reader_gram
        .as_ref()
        .map_or(0, |gram| gram.resident_bytes());
    eprintln!(
        "#2946 reader Gram routes: width {MULTI_TILE_WIDTH}, tile {tile}; V(I) cached {:?} streamed {streamed_total:?}; V(P) cached {:?} streamed {:?}; cache resident {resident} bytes",
        cached.total_variance(),
        cached_gradient.explained_variance,
        streamed_gradient.explained_variance,
    );
    // The fixture's premises: three tiles, and the cached route really taken.
    assert!(
        2 * tile < MULTI_TILE_WIDTH,
        "the fixture must span three tiles, got tile {tile} of width {MULTI_TILE_WIDTH}",
    );
    assert!(cached.units.reader_gram.is_some(), "the fixture's reader Gram must be admitted");
    assert_eq!(streamed_total, cached.total_variance());
    assert_eq!(streamed_gradient.explained_variance, cached_gradient.explained_variance);
    assert_eq!(streamed_gradient.horizontal_gradient, cached_gradient.horizontal_gradient);
}

/// The full double sum over every ordered pair at `P = I`, and the bands the halved pass must meet.
struct FullSquare {
    variance: f64,
    weighted: Array2<f64>,
    /// The upper triangle `k ≥ j` alone: the positive control, which omits exactly what the halving mirrors.
    upper_only_variance: f64,
    upper_only_weighted: Array2<f64>,
    variance_band: f64,
    weighted_band: Array2<f64>,
}

/// The full double sum `Σ_j Σ_k` over every ordered pair of [`integer_block`], each pair kernel evaluated at its own
/// ordered law with the operator's own stated rounding for copied coordinates.
///
/// A term `(j, k)` with `k ≥ j` gets the arguments the halved pass gives it, so it is the same word. A term with
/// `k < j` is evaluated at the swapped law. The exact kernel is symmetric, so that term differs from the halved pass's
/// `(k, j)` word by at most the two published roundings `ρ_jk + ρ_kj` (`ρ'` for the Price derivative). Each route
/// forms a term in at most two roundings. The value route sums at most `h²` terms (the halved pass uses `2h`
/// additions per term), and `B R` sums `h` products plus one merge per tile. So the bands add `γ_{h²+2}` and
/// `γ_{2h+1}` of the absolute terms, once per route.
fn full_square_reference(block: &KnownBlock) -> FullSquare {
    let units = &block.units;
    let readers = units.readers.view();
    let (width, terms) = readers.dim();
    let norms: Array1<f64> = readers
        .rows()
        .into_iter()
        .map(|row| row.dot(&row).sqrt())
        .collect();
    let kernel = |unit: usize, other: usize| {
        let covariance_rounding = covariance_rounding_band(
            &CovarianceFormation {
                terms,
                left_norm: norms[unit],
                right_norm: norms[other],
                left_row_error: 0.0,
                right_row_error: 0.0,
                law_gap: 0.0,
                variance_error_x: units.reader_variance_errors[unit],
                variance_error_y: units.reader_variance_errors[other],
            },
            units.reader_variances[unit],
            units.reader_variances[other],
        );
        pair_kernel(
            units.activation,
            PreactivationPair {
                mean_x: units.biases[unit],
                mean_y: units.biases[other],
                variance_x: units.reader_variances[unit],
                variance_y: units.reader_variances[other],
                covariance: readers.row(unit).dot(&readers.row(other)),
                covariance_rounding,
            },
        )
        .expect("an exact pair law")
    };
    let mut variance = 0.0;
    let mut upper_only_variance = 0.0;
    let mut absolute_variance = 0.0;
    let mut swap_variance = 0.0;
    let mut weighted = Array2::<f64>::zeros((width, terms));
    let mut upper_only_weighted = Array2::<f64>::zeros((width, terms));
    let mut absolute_weighted = Array2::<f64>::zeros((width, terms));
    let mut swap_weighted = Array2::<f64>::zeros((width, terms));
    for unit in 0..width {
        for other in 0..width {
            let metric_product = units
                .writers
                .column(unit)
                .dot(&units.metric_writers.column(other));
            let own = kernel(unit, other);
            let product_of_means = units.unit_means[unit] * units.unit_means[other];
            let term = metric_product * (own.value - product_of_means);
            let weight = metric_product * own.covariance_derivative;
            variance += term;
            absolute_variance += metric_product.abs() * (own.value.abs() + product_of_means.abs());
            if other >= unit {
                upper_only_variance += term;
            } else {
                let swapped = kernel(other, unit);
                swap_variance += metric_product.abs() * (own.value_rounding + swapped.value_rounding);
                for input in 0..terms {
                    swap_weighted[[unit, input]] += metric_product.abs()
                        * (own.covariance_derivative_rounding + swapped.covariance_derivative_rounding)
                        * readers[[other, input]].abs();
                }
            }
            for input in 0..terms {
                let contribution = weight * readers[[other, input]];
                weighted[[unit, input]] += contribution;
                absolute_weighted[[unit, input]] += contribution.abs();
                if other >= unit {
                    upper_only_weighted[[unit, input]] += contribution;
                }
            }
        }
    }
    let variance_growth = accumulation_growth(width * width + 2);
    let weighted_growth = accumulation_growth(2 * width + 1);
    FullSquare {
        variance,
        weighted,
        upper_only_variance,
        upper_only_weighted,
        variance_band: swap_variance + 2.0 * variance_growth * absolute_variance,
        weighted_band: swap_weighted + absolute_weighted * (2.0 * weighted_growth),
    }
}

#[test]
fn the_halved_pair_pass_is_the_full_double_sum_across_tiles() {
    // The halved pass evaluates `k ≥ j` only and mirrors `B` across tiles. Against the full double sum it must agree
    // within the band of `full_square_reference`, and the same band must reject the upper triangle alone, the part the
    // mirror adds.
    let block = integer_block(MULTI_TILE_WIDTH);
    let units = &block.units;
    let mut weighted = Array2::<f64>::zeros(units.readers.dim());
    let pass = units
        .pair_pass(units.readers.view(), CoordinateFormation::Copied, Some(&mut weighted))
        .expect("the halved V(I) pass");
    let halved = pass.energy.value;
    let reference = full_square_reference(&block);
    let worst_ratio = (&weighted - &reference.weighted)
        .mapv(f64::abs)
        .iter()
        .zip(reference.weighted_band.iter())
        .map(|(gap, band)| gap / band)
        .fold(0.0_f64, f64::max);
    let upper_only_rejected = (&weighted - &reference.upper_only_weighted)
        .mapv(f64::abs)
        .iter()
        .zip(reference.weighted_band.iter())
        .filter(|(gap, band)| gap > band)
        .count();
    eprintln!(
        "#2946 halved pass: width {MULTI_TILE_WIDTH}, tile {}; V(I) halved {halved} full {} gap {:e} band {:e}; upper triangle alone {} gap {:e}; B R worst gap/band {worst_ratio:e}; upper-triangle B R entries beyond the band {upper_only_rejected} of {}",
        upper_tile_rows(MULTI_TILE_WIDTH),
        reference.variance,
        (halved - reference.variance).abs(),
        reference.variance_band,
        reference.upper_only_variance,
        (halved - reference.upper_only_variance).abs(),
        weighted.len(),
    );
    assert_eq!(pass.energy, block.total_variance(), "the V(I) pass must reproduce the cached V(I)");
    assert!(
        (halved - reference.variance).abs() <= reference.variance_band,
        "halved V(I) {halved} vs the full double sum {} beyond the band {}",
        reference.variance,
        reference.variance_band,
    );
    assert!(
        worst_ratio <= 1.0,
        "the halved B R departs from the full double sum by {worst_ratio} bands",
    );
    // Positive controls: the band resolves the mirrored half.
    assert!(
        (halved - reference.upper_only_variance).abs() > reference.variance_band,
        "the upper triangle alone {} must be rejected against {halved} ± {}",
        reference.upper_only_variance,
        reference.variance_band,
    );
    assert!(upper_only_rejected > 0, "the upper triangle's B R must be rejected somewhere");
}
