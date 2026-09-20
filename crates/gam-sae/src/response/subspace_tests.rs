#![cfg(test)]
//! #2946 A2, A3 and A4 pins for the retained-response operator, and R9 pins for the gated block.
//!
//! The exact identities (A2 and the planted subspace) run on dyadic data: small-integer readers, frames and rotations
//! built from `H = I − ½·11ᵀ`, whose entries are all `±½`. Every covariance `w_jᵀ P w_k` the operator forms is then
//! computed exactly, both routes of an identity feed the kernel bit-identical arguments, and the identity holds
//! bit for bit with no tolerance. The Monte Carlo pins run the executed block and accept within a standard-error
//! multiple derived from a declared false-alarm rate, each with a positive control that the test shows it rejects.
//! The gradient pin (A4) accepts against a Richardson ladder of central differences within the ladder's own measured
//! truncation, with the same kind of positive control. Every pin prints its numbers whether it passes or fails, so a
//! green run is a receipt.

use super::{BandedEnergy, KnownBlock, KnownGatedBlock, ResponseError, pair_covariance, signed_sum};
use gam_linalg::roundoff::accumulation_growth;
use gam_math::gaussian_activation::{
    GaussianActivation, PreactivationPair, gaussian_smoothing_derivatives, pair_kernel,
};
use gam_math::gaussian_gated::silu_derivatives;
use gam_math::probability::{normal_cdf, normal_pdf, standard_normal_quantile};
use gam_math::roundoff::{UNIT_ROUNDOFF, inflated};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, array, s};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::f64::consts::PI;

/// The declared probability that a correct operator fails one Monte Carlo test for a fresh seed. It is split evenly
/// over the test's two-sided agreement arms (Bonferroni), which fixes each arm's standard-error multiple.
const FAMILY_FALSE_ALARM: f64 = 1.0e-6;

/// The acceptance multiple of one two-sided arm among `arms`.
fn standard_error_multiple(arms: usize) -> f64 {
    standard_normal_quantile(1.0 - FAMILY_FALSE_ALARM / (2.0 * arms as f64))
        .expect("the false-alarm split lies inside (0, 1)")
}

/// `H = I − ½·11ᵀ` in four dimensions: symmetric, orthogonal, every entry `±½`.
fn half_reflector() -> Array2<f64> {
    Array2::from_shape_fn((4, 4), |(row, column)| if row == column { 0.5 } else { -0.5 })
}

fn identity(dim: usize) -> Array2<f64> {
    Array2::from_shape_fn((dim, dim), |(row, column)| if row == column { 1.0 } else { 0.0 })
}

/// The first two columns of `H` span `{(1, −1, 0, 0), (0, 0, 1, 1)}`, its last two `{(1, 1, 0, 0), (0, 0, 1, −1)}`.
fn retained_frame() -> Array2<f64> {
    half_reflector().slice(s![.., ..2]).to_owned()
}

/// Readers whose discarded parts all overlap: `(2,0,0,0) = (1,−1,0,0) + (1,1,0,0)`,
/// `(1,1,1,1) = (0,0,1,1) + (1,1,0,0)`, `(2,0,1,1) = (1,−1,0,0) + (0,0,1,1) + (1,1,0,0)` and
/// `(1,1,1,−1) = (0,0,1,−1) + (1,1,0,0)`. Every pair has positive covariance both inside and outside the frame, so the
/// cross terms carry a large share of `V` and `E`.
fn overlapping_readers() -> Array2<f64> {
    array![
        [2.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [2.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, -1.0],
    ]
}

/// Readers inside the span of [`retained_frame`].
fn planted_readers() -> Array2<f64> {
    array![
        [1.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, -1.0, 1.0, 1.0],
        [2.0, -2.0, -1.0, -1.0],
    ]
}

/// Writer columns `(1,0,1)`, `(1,1,0)`, `(0,1,1)`, `(1,1,1)`: nonnegative, so every `D_jk = u_jᵀ M u_k` is positive.
fn writers() -> Array2<f64> {
    array![
        [1.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 1.0],
    ]
}

/// A dyadic, symmetric, positive definite output metric with a coupled pair of outputs.
fn metric() -> Array2<f64> {
    array![[2.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn relu_block_with_output_bias(readers: Array2<f64>, output_bias: Array1<f64>) -> KnownBlock {
    let width = readers.nrows();
    KnownBlock::new(
        readers,
        Array1::zeros(width),
        writers(),
        output_bias,
        metric().view(),
        GaussianActivation::Relu,
    )
    .expect("a finite block with a symmetric positive definite metric")
}

fn relu_block(readers: Array2<f64>) -> KnownBlock {
    relu_block_with_output_bias(readers, Array1::zeros(3))
}

fn standard_normal(rng: &mut StdRng) -> f64 {
    // `1 − U[0, 1)` lies in `(0, 1]`, so the logarithm is finite.
    let u1: f64 = 1.0 - rng.random_range(0.0..1.0);
    let u2: f64 = rng.random_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

fn standard_normal_vector(rng: &mut StdRng, dim: usize) -> Array1<f64> {
    let mut draw = Array1::<f64>::zeros(dim);
    for entry in draw.iter_mut() {
        *entry = standard_normal(rng);
    }
    draw
}

/// The executed zero-bias ReLU block `U relu(W z)`.
fn execute_relu_block(readers: &Array2<f64>, z: ArrayView1<'_, f64>) -> Array1<f64> {
    writers().dot(&readers.dot(&z).mapv(|activation| activation.max(0.0)))
}

/// `Pz = Q Qᵀ z`.
fn project(frame: ArrayView2<'_, f64>, z: ArrayView1<'_, f64>) -> Array1<f64> {
    frame.dot(&frame.t().dot(&z))
}

/// The mean of i.i.d. terms and its standard error.
fn mean_and_standard_error(terms: &[f64]) -> (f64, f64) {
    let count = terms.len() as f64;
    let mean = terms.iter().sum::<f64>() / count;
    let variance = terms.iter().map(|term| (term - mean).powi(2)).sum::<f64>() / (count - 1.0);
    (mean, (variance / count).sqrt())
}

/// `Σ_j D_jj [K_σ(0, 0; v_j, v_j, ‖Qᵀ w_j‖²) − m_j²]`: `V(P)` with every cross term `j ≠ k` dropped, the positive
/// control the Monte Carlo pins must reject.
fn diagonal_only_relu_variance(readers: &Array2<f64>, frame: ArrayView2<'_, f64>) -> f64 {
    let metric_writers = metric().dot(&writers());
    let coordinates = readers.dot(&frame);
    let mut total = 0.0;
    for unit in 0..readers.nrows() {
        let variance = readers.row(unit).dot(&readers.row(unit));
        let covariance = coordinates.row(unit).dot(&coordinates.row(unit));
        // Integer readers and the dyadic frames of these tests form every variance and covariance exactly, so the
        // law is exact and states no rounding.
        let second_moment = pair_kernel(
            GaussianActivation::Relu,
            PreactivationPair {
                mean_x: 0.0,
                mean_y: 0.0,
                variance_x: variance,
                variance_y: variance,
                covariance,
                covariance_rounding: 0.0,
            },
        )
        .expect("ReLU pair kernel at zero mean")
        .value;
        let mut mean = [0.0];
        gaussian_smoothing_derivatives(GaussianActivation::Relu, 0.0, variance, &mut mean)
            .expect("ReLU smoothing at a finite variance");
        let metric_diagonal = writers().column(unit).dot(&metric_writers.column(unit));
        total += metric_diagonal * (second_moment - mean[0] * mean[0]);
    }
    total
}

#[test]
fn discarded_error_vanishes_on_the_full_frame_and_shrinks_strictly_as_the_frame_grows() {
    let block = relu_block(overlapping_readers());
    let reflector = half_reflector();
    let errors: Vec<BandedEnergy> = (1..=4)
        .map(|rank| {
            block
                .discarded_error(reflector.slice(s![.., ..rank]))
                .expect("an orthonormal frame inside the input")
        })
        .collect();
    let total = block.total_variance();
    eprintln!(
        "#2946 A2 V(I) = {} ± {:e}; E at frame rank 1..4 = {} ± {:e}, {} ± {:e}, {} ± {:e}, {} ± {:e}",
        total.value,
        total.band,
        errors[0].value,
        errors[0].band,
        errors[1].value,
        errors[1].band,
        errors[2].value,
        errors[2].band,
        errors[3].value,
        errors[3].band,
    );
    // `W H` has half-integer entries and `(W H)(W H)ᵀ = W Wᵀ` holds exactly, so the full frame feeds the kernel the
    // same covariances as `V(I)` and the discarded error is exactly zero.
    assert_eq!(errors[3].value, 0.0, "E(I) must be exactly 0, got {:?}", errors[3]);
    assert_eq!(
        block
            .explained_variance(reflector.view())
            .expect("the full frame")
            .value,
        total.value,
    );
    assert!(total.resolved_positive(), "V(I) must clear its band: {total:?}");
    // Conditioning on a larger subspace can only reduce the discarded error, and these readers overlap every column of
    // `H`, so each added column removes a positive share. The band resolves every step, so it is not vacuous.
    for rank in 0..3 {
        let step = signed_sum(&[errors[rank]], &[errors[rank + 1]]);
        assert!(
            step.resolved_positive(),
            "E must shrink resolvably as the frame grows: E(rank {}) = {:?} vs E(rank {}) = {:?}",
            rank + 1,
            errors[rank],
            rank + 2,
            errors[rank + 1],
        );
    }
}

#[test]
fn explained_variance_and_retained_response_are_invariant_under_an_orthogonal_input_rotation() {
    // W → W H and Q → Hᵀ Q = H Q, with H orthogonal and not a signed permutation. Every reader coordinate
    // `(W H)(H Q) = W Q`, every reader variance and every discarded residual is formed exactly, so the rotated block
    // must reproduce the unrotated one bit for bit.
    let reflector = half_reflector();
    let readers = overlapping_readers();
    let rotated_readers = readers.dot(&reflector);
    assert_ne!(rotated_readers, readers, "the rotation must move the readers");
    let block = relu_block(readers);
    let rotated_block = relu_block(rotated_readers);
    let frame = identity(4).slice(s![.., ..2]).to_owned();
    let rotated_frame = reflector.dot(&frame);

    assert_eq!(rotated_block.total_variance(), block.total_variance());
    assert_eq!(
        rotated_block
            .explained_variance(rotated_frame.view())
            .expect("rotated frame"),
        block.explained_variance(frame.view()).expect("frame"),
    );
    assert_eq!(
        rotated_block
            .discarded_error(rotated_frame.view())
            .expect("rotated frame"),
        block.discarded_error(frame.view()).expect("frame"),
    );

    let points = array![[0.0, 0.0, 0.0, 0.0], [1.0, -2.0, 0.5, 3.0], [-2.0, 0.5, 1.0, -1.5]];
    let rotated_points = points.dot(&reflector);
    let response = block
        .retained_response(frame.view(), points.view())
        .expect("finite points");
    let rotated_response = rotated_block
        .retained_response(rotated_frame.view(), rotated_points.view())
        .expect("finite rotated points");
    assert_eq!(rotated_response, response);
}

#[test]
fn a_coordinate_frame_reads_the_selected_reader_columns() {
    // Integer readers and a 0/1 frame form every covariance exactly, so the coordinate route and the frame route
    // feed the kernel the same bits.
    let block = relu_block(overlapping_readers());
    let unit_frame = identity(4);
    for retained in [vec![], vec![0], vec![1, 3], vec![0, 2, 3], vec![0, 1, 2, 3]] {
        let frame = unit_frame.select(Axis(1), &retained);
        // The values agree bit for bit; the frame route states a frame's formation errors, so its band differs.
        assert_eq!(
            block
                .explained_variance_of_coordinates(&retained)
                .expect("strictly increasing coordinates")
                .value,
            block
                .explained_variance(frame.view())
                .expect("a coordinate frame")
                .value,
            "coordinates {retained:?}",
        );
    }
    assert_eq!(
        block
            .explained_variance_of_coordinates(&[0, 1, 2, 3])
            .expect("every coordinate"),
        block.total_variance(),
    );
    let refusal = block.explained_variance_of_coordinates(&[2, 1]);
    assert!(
        matches!(refusal, Err(ResponseError::InvalidRetainedCoordinates { position: 1 })),
        "decreasing coordinates must be refused, got {refusal:?}",
    );
    let refusal = block.explained_variance_of_coordinates(&[0, 4]);
    assert!(
        matches!(refusal, Err(ResponseError::InvalidRetainedCoordinates { position: 1 })),
        "an out-of-range coordinate must be refused, got {refusal:?}",
    );
}

#[test]
fn a_planted_exact_subspace_discards_nothing_and_its_complement_discards_everything() {
    let block = relu_block(planted_readers());
    let reflector = half_reflector();
    let retained = block
        .discarded_error(reflector.slice(s![.., ..2]))
        .expect("the planted frame");
    // The planted readers are exactly orthogonal to the complement, so the complement frame feeds the kernel the
    // zero covariances of the empty frame: both discard the same, positive, error.
    let complement = block
        .discarded_error(reflector.slice(s![.., 2..]))
        .expect("the complement frame");
    let empty = block
        .discarded_error(reflector.slice(s![.., ..0]))
        .expect("the empty frame");
    eprintln!("#2946 A3 planted: E(planted) = {retained:?}, E(complement) = {complement:?}, E(empty) = {empty:?}");
    assert_eq!(retained.value, 0.0, "E on the planted subspace must be exactly 0, got {retained:?}");
    assert_eq!(complement.value, empty.value);
    assert!(
        complement.resolved_positive(),
        "the complement must discard a resolved positive error, got {complement:?}",
    );
}

#[test]
fn monte_carlo_of_the_executed_block_reproduces_explained_variance_and_discarded_error() {
    // Couple Z' = P Z + (I − P) Z̃. For the executed block, E⟨F(Z) − μ, F(Z') − μ⟩_M = V(P), ½ E‖F(Z) − F(Z')‖²_M =
    // E(P), E‖F(Z) − μ‖²_M = V(I), and by R2 E‖F(Z) − F̄_P(PZ)‖²_M = E(P). None of these estimators touches the
    // kernels, so they check V, E and F̄_P against the block itself.
    let readers = overlapping_readers();
    let block = relu_block(readers.clone());
    let frame = retained_frame();
    let output_metric = metric();
    let draws = 1 << 18;
    let mut rng = StdRng::seed_from_u64(0x2946_a3);
    let mut outputs = Array2::<f64>::zeros((draws, 3));
    let mut coupled_outputs = Array2::<f64>::zeros((draws, 3));
    let mut inputs = Array2::<f64>::zeros((draws, 4));
    for draw in 0..draws {
        let z = standard_normal_vector(&mut rng, 4);
        let independent = standard_normal_vector(&mut rng, 4);
        let coupled = &independent + &project(frame.view(), (&z - &independent).view());
        outputs.row_mut(draw).assign(&execute_relu_block(&readers, z.view()));
        coupled_outputs
            .row_mut(draw)
            .assign(&execute_relu_block(&readers, coupled.view()));
        inputs.row_mut(draw).assign(&z);
    }
    let mean = outputs.mean_axis(Axis(0)).expect("draws");
    let coupled_mean = coupled_outputs.mean_axis(Axis(0)).expect("draws");
    let retained = block
        .retained_response(frame.view(), inputs.view())
        .expect("finite draws");
    let mut total_terms = Vec::with_capacity(draws);
    let mut explained_terms = Vec::with_capacity(draws);
    let mut coupled_error_terms = Vec::with_capacity(draws);
    let mut residual_error_terms = Vec::with_capacity(draws);
    for draw in 0..draws {
        let centered = &outputs.row(draw) - &mean;
        let coupled_centered = &coupled_outputs.row(draw) - &coupled_mean;
        let difference = &outputs.row(draw) - &coupled_outputs.row(draw);
        let residual = &outputs.row(draw) - &retained.row(draw);
        total_terms.push(centered.dot(&output_metric.dot(&centered)));
        explained_terms.push(centered.dot(&output_metric.dot(&coupled_centered)));
        coupled_error_terms.push(0.5 * difference.dot(&output_metric.dot(&difference)));
        residual_error_terms.push(residual.dot(&output_metric.dot(&residual)));
    }
    let multiple = standard_error_multiple(4);
    // Centring on the sample mean scales each covariance estimate by (n − 1)/n; undo it so the estimate is unbiased.
    let unbias = draws as f64 / (draws as f64 - 1.0);
    let (total_estimate, total_se) = mean_and_standard_error(&total_terms);
    let (explained_estimate, explained_se) = mean_and_standard_error(&explained_terms);
    let (coupled_error_estimate, coupled_error_se) = mean_and_standard_error(&coupled_error_terms);
    let (residual_error_estimate, residual_error_se) = mean_and_standard_error(&residual_error_terms);

    let explained = block.explained_variance(frame.view()).expect("frame").value;
    let discarded = block.discarded_error(frame.view()).expect("frame").value;
    let arms = [
        ("V(I)", block.total_variance().value, total_estimate * unbias, total_se),
        ("V(P)", explained, explained_estimate * unbias, explained_se),
        ("E(P) coupled", discarded, coupled_error_estimate, coupled_error_se),
        ("E(P) residual of F-bar", discarded, residual_error_estimate, residual_error_se),
    ];
    for (label, analytic, estimate, standard_error) in arms {
        eprintln!(
            "#2946 A3 {label}: analytic {analytic} executed MC {estimate} se {standard_error} z {:.3} multiple {multiple:.3} draws {draws}",
            (analytic - estimate) / standard_error,
        );
        assert!(
            (analytic - estimate).abs() <= multiple * standard_error,
            "{label}: analytic {analytic} vs executed Monte Carlo {estimate} ± {standard_error} (multiple {multiple})",
        );
    }

    // Positive control: the same operator without its cross terms is rejected by the same arms.
    let diagonal_total = diagonal_only_relu_variance(&readers, identity(4).view());
    let diagonal_explained = diagonal_only_relu_variance(&readers, frame.view());
    let controls = [
        ("V(P) without cross terms", diagonal_explained, explained_estimate * unbias, explained_se),
        (
            "E(P) without cross terms",
            diagonal_total - diagonal_explained,
            coupled_error_estimate,
            coupled_error_se,
        ),
    ];
    for (label, control, estimate, standard_error) in controls {
        eprintln!(
            "#2946 A3 control {label}: {control} executed MC {estimate} se {standard_error} z {:.3}",
            (control - estimate) / standard_error,
        );
        assert!(
            (control - estimate).abs() > multiple * standard_error,
            "{label} must be rejected: control {control} vs executed Monte Carlo {estimate} ± {standard_error}",
        );
    }
}

#[test]
fn retained_response_is_the_executed_block_averaged_over_the_discarded_input() {
    // R1 at fixed points: F̄_P(Pz) = E F(Pz + (I − P) Z̃), estimated by executing the block.
    let readers = overlapping_readers();
    let block = relu_block(readers.clone());
    let frame = retained_frame();
    let points = array![[0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 0.5, 2.0], [-2.0, 0.5, 1.0, -1.0]];
    let response = block
        .retained_response(frame.view(), points.view())
        .expect("finite points");
    let draws = 1 << 16;
    let mut rng = StdRng::seed_from_u64(0x2946_a1);
    let multiple = standard_error_multiple(points.nrows() * 3);
    for point in 0..points.nrows() {
        let retained = project(frame.view(), points.row(point));
        let mut samples = Array2::<f64>::zeros((draws, 3));
        for draw in 0..draws {
            let independent = standard_normal_vector(&mut rng, 4);
            let input = &retained + &independent - &project(frame.view(), independent.view());
            samples.row_mut(draw).assign(&execute_relu_block(&readers, input.view()));
        }
        let plug_in = execute_relu_block(&readers, retained.view());
        for output in 0..3 {
            let column: Vec<f64> = samples.column(output).to_vec();
            let (estimate, standard_error) = mean_and_standard_error(&column);
            let analytic = response[[point, output]];
            eprintln!(
                "#2946 R1 point {point} output {output}: analytic {analytic} executed MC {estimate} se {standard_error} plug-in {}",
                plug_in[output],
            );
            assert!(
                (analytic - estimate).abs() <= multiple * standard_error,
                "point {point} output {output}: analytic {analytic} vs executed Monte Carlo {estimate} ± {standard_error}",
            );
            // Positive control at the origin, where the smoothing gap is largest: plugging the projected point into
            // F, which ignores the discarded variance, is rejected.
            if point == 0 {
                assert!(
                    (plug_in[output] - estimate).abs() > multiple * standard_error,
                    "the plug-in F(Pz) must be rejected at the origin: {} vs {estimate} ± {standard_error}",
                    plug_in[output],
                );
            }
        }
    }
}

#[test]
fn the_output_bias_moves_every_response_and_no_variance() {
    let readers = overlapping_readers();
    let output_bias = array![0.5, -1.25, 2.0];
    let unbiased = relu_block(readers.clone());
    let biased = relu_block_with_output_bias(readers, output_bias.clone());
    let frame = retained_frame();
    let points = array![[0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 0.5, 2.0], [-2.0, 0.5, 1.0, -1.0]];
    let response = unbiased
        .retained_response(frame.view(), points.view())
        .expect("finite points");
    assert_eq!(
        biased
            .retained_response(frame.view(), points.view())
            .expect("finite points"),
        &response + &output_bias,
    );
    assert_eq!(biased.total_variance(), unbiased.total_variance());
    assert_eq!(
        biased.explained_variance(frame.view()).expect("frame"),
        unbiased.explained_variance(frame.view()).expect("frame"),
    );
    assert_eq!(
        biased
            .explained_variance_gradient(frame.view())
            .expect("frame")
            .horizontal_gradient,
        unbiased
            .explained_variance_gradient(frame.view())
            .expect("frame")
            .horizontal_gradient,
    );
    let refusal = KnownBlock::new(
        overlapping_readers(),
        Array1::zeros(4),
        writers(),
        Array1::zeros(2),
        metric().view(),
        GaussianActivation::Relu,
    );
    assert!(
        matches!(
            refusal,
            Err(ResponseError::DimensionMismatch {
                context: "block output bias",
                expected: 3,
                got: 2,
            })
        ),
        "an output bias of the wrong length must be refused, got {refusal:?}",
    );
}

#[test]
fn the_operator_refuses_an_asymmetric_or_indefinite_metric_and_an_over_wide_frame() {
    let mut asymmetric = metric();
    asymmetric[[0, 1]] = 0.25;
    let refusal = KnownBlock::new(
        overlapping_readers(),
        Array1::zeros(4),
        writers(),
        Array1::zeros(3),
        asymmetric.view(),
        GaussianActivation::Relu,
    );
    assert!(
        matches!(refusal, Err(ResponseError::MetricNotSymmetric { row: 0, column: 1 })),
        "an asymmetric metric must be refused, got {refusal:?}",
    );
    let indefinite = array![[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let refusal = KnownBlock::new(
        overlapping_readers(),
        Array1::zeros(4),
        writers(),
        Array1::zeros(3),
        indefinite.view(),
        GaussianActivation::Relu,
    );
    assert!(
        matches!(refusal, Err(ResponseError::MetricNotPositiveDefinite { .. })),
        "an indefinite metric must be refused, got {refusal:?}",
    );
    let block = relu_block(overlapping_readers());
    let wide = Array2::<f64>::zeros((4, 5));
    let refusal = block.explained_variance(wide.view());
    assert!(
        matches!(refusal, Err(ResponseError::FrameWiderThanInput { rank: 5, input_dim: 4 })),
        "a frame wider than the input must be refused, got {refusal:?}",
    );
}

#[test]
fn a_rounding_scale_frame_defect_is_projected_and_a_stretched_frame_is_refused() {
    // (1 + 1e-12)·H[:, :2]: its Gram overshoots I by about 2e-12, so each planted reader's in-frame covariance exceeds
    // its variance by that share. The measured defect enters the stated covariance rounding, so the kernel projects
    // the law onto the Cauchy–Schwarz boundary instead of refusing it.
    let readers = planted_readers();
    let block = relu_block(readers.clone());
    let frame = retained_frame() * (1.0 + 1.0e-12);
    let explained = block.explained_variance(frame.view());
    assert!(
        explained.is_ok(),
        "a rounding-scale defect must be projected, got {explained:?}",
    );
    // Positive control: the same diagonal law stating no rounding is refused, so the band is what projects it.
    let coordinates = readers.dot(&frame);
    let variance = readers.row(0).dot(&readers.row(0));
    let covariance = coordinates.row(0).dot(&coordinates.row(0));
    assert!(covariance > variance, "the stretched frame must overshoot: {covariance} vs {variance}");
    let refusal = pair_kernel(
        GaussianActivation::Relu,
        PreactivationPair {
            mean_x: 0.0,
            mean_y: 0.0,
            variance_x: variance,
            variance_y: variance,
            covariance,
            covariance_rounding: 0.0,
        },
    );
    assert!(
        matches!(
            refusal,
            Err(gam_math::gaussian_activation::GaussianActivationError::CovarianceOutsideCauchySchwarz { .. })
        ),
        "without a stated rounding the overshooting law must be refused, got {refusal:?}",
    );
    // A frame stretched far past rounding has no nearest orthonormal frame the operator could be exact for.
    let stretched = array![[2.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];
    let refusal = block.explained_variance(stretched.view());
    assert!(
        matches!(refusal, Err(ResponseError::FrameNotOrthonormal { .. })),
        "a stretched frame must be refused, got {refusal:?}",
    );
}

#[test]
fn a_signed_sum_carries_its_operand_bands_and_the_rounding_of_its_additions() {
    // Dyadic operands, so the value is exact and the expected band is rebuilt in the implementation's own order.
    let added = [
        BandedEnergy {
            value: 3.0,
            band: 1.0e-12,
        },
        BandedEnergy {
            value: -0.5,
            band: 2.0e-12,
        },
    ];
    let subtracted = [BandedEnergy {
        value: 1.25,
        band: 0.0,
    }];
    let sum = signed_sum(&added, &subtracted);
    assert_eq!(sum.value, 1.25);
    let operand_bands = 0.0 + 1.0e-12 + 2.0e-12 + 0.0;
    assert_eq!(sum.band, operand_bands + accumulation_growth(2) * 4.75);
    assert!(sum.resolved_positive());
    assert!(!BandedEnergy::ZERO.resolved_positive());
    // Equal values cancel exactly and do not clear a positive band: the difference is unresolved.
    let unresolved = signed_sum(
        &[BandedEnergy {
            value: 2.0,
            band: 1.0e-15,
        }],
        &[BandedEnergy {
            value: 2.0,
            band: 0.0,
        }],
    );
    assert_eq!(unresolved.value, 0.0);
    assert!(!unresolved.resolved_positive());
}

/// A block on [`overlapping_readers`] with dyadic biases of both signs, so every pair runs a biased kernel.
fn biased_block(activation: GaussianActivation) -> KnownBlock {
    KnownBlock::new(
        overlapping_readers(),
        array![0.5, -0.25, 1.0, -1.0],
        writers(),
        Array1::zeros(3),
        metric().view(),
        activation,
    )
    .expect("a finite biased block with a symmetric positive definite metric")
}

/// `Q(t) = Q + (Q y (cos t − 1) + x sin t) yᵀ`, the Grassmann geodesic from `frame` whose velocity at `t = 0` is the
/// horizontal `x yᵀ`, for a unit `x` orthogonal to the frame and a unit `y`. It turns the retained direction `Q y`
/// toward `x`, and `Q(t)ᵀ Q(t) = (I − y yᵀ) + y yᵀ = I`.
fn geodesic_frame(frame: ArrayView2<'_, f64>, x: ArrayView1<'_, f64>, y: ArrayView1<'_, f64>, t: f64) -> Array2<f64> {
    let turned = frame.dot(&y) * (t.cos() - 1.0) + &x * t.sin();
    let mut moved = frame.to_owned();
    for ((row, column), entry) in moved.indexed_iter_mut() {
        *entry += turned[row] * y[column];
    }
    moved
}

/// `(V(Q(t)) − V(Q(−t))) / 2t` along [`geodesic_frame`].
fn geodesic_central_difference(
    block: &KnownBlock,
    frame: ArrayView2<'_, f64>,
    x: ArrayView1<'_, f64>,
    y: ArrayView1<'_, f64>,
    step: f64,
) -> f64 {
    let ahead = block
        .explained_variance(geodesic_frame(frame, x, y, step).view())
        .expect("a geodesic frame is orthonormal")
        .value;
    let behind = block
        .explained_variance(geodesic_frame(frame, x, y, -step).view())
        .expect("a geodesic frame is orthonormal")
        .value;
    (ahead - behind) / (2.0 * step)
}

/// `2 Σ_j (w_jᵀ x) B_jj (w_jᵀ Q y)`: the directional derivative `⟨2 Wᵀ B W Q, x yᵀ⟩` with every cross term `j ≠ k` of
/// `B` dropped, the positive control the gradient pin must reject.
fn diagonal_only_directional_derivative(
    block: &KnownBlock,
    frame: ArrayView2<'_, f64>,
    x: ArrayView1<'_, f64>,
    y: ArrayView1<'_, f64>,
) -> f64 {
    let readers = block.readers();
    let coordinates = readers.dot(&frame);
    let retained_direction = frame.dot(&y);
    let mut total = 0.0;
    for unit in 0..block.width() {
        let reader = readers.row(unit);
        // Integer readers and the dyadic frame form the variance and the covariance exactly, so the law states no
        // rounding.
        let variance = reader.dot(&reader);
        let covariance = coordinates.row(unit).dot(&coordinates.row(unit));
        let slope = pair_kernel(
            block.activation(),
            PreactivationPair {
                mean_x: block.biases()[unit],
                mean_y: block.biases()[unit],
                variance_x: variance,
                variance_y: variance,
                covariance,
                covariance_rounding: 0.0,
            },
        )
        .expect("a biased pair kernel inside its Cauchy–Schwarz interval")
        .covariance_derivative;
        let metric_diagonal = block
            .writers()
            .column(unit)
            .dot(&block.metric_writers().column(unit));
        total += 2.0 * reader.dot(&x) * metric_diagonal * slope * reader.dot(&retained_direction);
    }
    total
}

#[test]
fn the_frame_gradient_is_the_derivative_of_explained_variance_along_a_grassmann_geodesic() {
    // A4. Along the geodesic `Q(t)` with velocity `x yᵀ`, `φ(t) = V(Q(t))` is smooth, so the central difference is
    // `D(t) = φ'(0) + φ'''(0) t²/6 + O(t⁴)` and the Richardson value `R = (4 D(t/2) − D(t))/3 = φ'(0) + O(t⁴)`, both
    // with roundoff `O(ε V / t)`. The ladder's step `D(t) − D(t/2) = φ'''(0) t²/8 + O(t⁴)` is the measured truncation
    // of the coarser difference, and `R` is finer by two orders of `t`, so the gradient is accepted when
    // `|⟨∇_Q V, x yᵀ⟩ − R| ≤ |D(t) − D(t/2)|`. Every pair is biased, so both biased kernels and their Price derivatives
    // are exercised. The positive control is the same derivative with every cross term of `B` dropped.
    let frame = retained_frame();
    // `h₃` is a unit direction orthogonal to the frame, and `y` turns a mix of both retained directions toward it.
    let x = half_reflector().column(2).to_owned();
    let y = array![0.6, 0.8];
    let step = 2.0_f64.powi(-7);
    for activation in [GaussianActivation::Relu, GaussianActivation::ExactGelu] {
        let block = biased_block(activation);
        let gradient = block
            .explained_variance_gradient(frame.view())
            .expect("an orthonormal frame");
        let analytic = x.dot(&gradient.horizontal_gradient.dot(&y));
        let coarse = geodesic_central_difference(&block, frame.view(), x.view(), y.view(), step);
        let fine = geodesic_central_difference(&block, frame.view(), x.view(), y.view(), step / 2.0);
        let extrapolated = (4.0 * fine - coarse) / 3.0;
        let truncation = (coarse - fine).abs();
        let diagonal_only = diagonal_only_directional_derivative(&block, frame.view(), x.view(), y.view());
        eprintln!(
            "#2946 A4 {activation:?}: V(P) = {:?}; <grad V, x y^T> = {analytic}; central differences D({step}) = {coarse}, D({}) = {fine}; Richardson {extrapolated}; |analytic - Richardson| = {:e} against the ladder truncation {truncation:e}; control without cross terms {diagonal_only}, off by {:e}",
            gradient.explained_variance,
            step / 2.0,
            (analytic - extrapolated).abs(),
            (diagonal_only - extrapolated).abs(),
        );
        // One pass returns the value pass's V and E, bands included, bit for bit.
        assert_eq!(
            gradient.explained_variance,
            block
                .explained_variance(frame.view())
                .expect("an orthonormal frame"),
        );
        assert_eq!(
            gradient.discarded_error,
            signed_sum(&[block.total_variance()], &[gradient.explained_variance]),
        );
        assert!(
            (analytic - extrapolated).abs() <= truncation,
            "{activation:?}: the frame gradient's directional derivative {analytic} must match the Richardson value {extrapolated} within the ladder's truncation {truncation}",
        );
        assert!(
            (diagonal_only - extrapolated).abs() > truncation,
            "{activation:?}: the derivative without cross terms {diagonal_only} must be rejected against {extrapolated} ± {truncation}",
        );
    }
}

/// #2946's saturation counterexample `F(z) = (GELU(10 z₁ − 10), 0.1 z₂)` as an exact-GELU block. The linear
/// coordinate is `0.1 z₂ = 0.1 GELU(z₂) − 0.1 GELU(−z₂)`, because `GELU(t) − GELU(−t) = t (Φ(t) + Φ(−t)) = t`.
fn saturation_block() -> KnownBlock {
    KnownBlock::new(
        array![[10.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
        array![-10.0, 0.0, 0.0],
        array![[1.0, 0.0, 0.0], [0.0, 0.1, -0.1]],
        Array1::zeros(2),
        identity(2).view(),
        GaussianActivation::ExactGelu,
    )
    .expect("the saturation block")
}

/// The executed exact GELU `t Φ(t)`.
fn exact_gelu(t: f64) -> f64 {
    t * normal_cdf(t)
}

/// `GELU'(t) = Φ(t) + t φ(t)`.
fn exact_gelu_slope(t: f64) -> f64 {
    normal_cdf(t) + t * normal_pdf(t)
}

/// The executed saturation block.
fn execute_saturation_block(z: ArrayView1<'_, f64>) -> Array1<f64> {
    array![
        exact_gelu(10.0 * z[0] - 10.0),
        0.1 * exact_gelu(z[1]) - 0.1 * exact_gelu(-z[1]),
    ]
}

/// The active subspace `E[∇Fᵀ M ∇F] = Wᵀ B(I) W`. With `∇F(z) = U diag(σ'(b + W z)) W`, the average is
/// `Wᵀ (D ∘ E[σ'(X_j) σ'(X_k)]) W`, and at the full covariance `w_jᵀ w_k` each `E[σ'(X_j) σ'(X_k)]` is the pair
/// kernel's Price derivative `∂_r K_σ`. Integer readers make every law exact.
fn active_subspace_matrix(block: &KnownBlock) -> Array2<f64> {
    let readers = block.readers();
    let input_dim = block.input_dim();
    let mut average = Array2::<f64>::zeros((input_dim, input_dim));
    for unit in 0..block.width() {
        for other in 0..block.width() {
            let slope_product = pair_kernel(
                block.activation(),
                PreactivationPair {
                    mean_x: block.biases()[unit],
                    mean_y: block.biases()[other],
                    variance_x: readers.row(unit).dot(&readers.row(unit)),
                    variance_y: readers.row(other).dot(&readers.row(other)),
                    covariance: readers.row(unit).dot(&readers.row(other)),
                    covariance_rounding: 0.0,
                },
            )
            .expect("an exact pair law")
            .covariance_derivative;
            let weight = block
                .writers()
                .column(unit)
                .dot(&block.metric_writers().column(other))
                * slope_product;
            for row in 0..input_dim {
                for column in 0..input_dim {
                    average[[row, column]] += readers[[unit, row]] * weight * readers[[other, column]];
                }
            }
        }
    }
    average
}

#[test]
fn the_saturated_coordinate_carries_the_discarded_error_and_the_averaged_gradient_ranks_it_first() {
    // Retaining z₂ alone discards Var GELU(10 Z − 10), and retaining z₁ alone discards Var(0.1 Z₂) = 0.01. So the
    // discarded error ranks z₁ first, although the block is nearly flat in z₁ at the baseline z = 0. Both errors are
    // checked against the executed block through the coupling Z' = P Z + (I − P) Z̃, whose estimator
    // ½‖F(Z) − F(Z')‖² touches no kernel code. The control is the active subspace E[∇Fᵀ M ∇F]: it ranks z₁ first too,
    // so the ratio of discarded errors beats only a baseline-point Jacobian, which ranks z₂ first.
    let block = saturation_block();
    let coordinates = identity(2);
    let discard_second = block
        .discarded_error(coordinates.slice(s![.., ..1]))
        .expect("retain z1")
        .value;
    let discard_first = block
        .discarded_error(coordinates.slice(s![.., 1..]))
        .expect("retain z2")
        .value;
    let active = active_subspace_matrix(&block);

    let draws = 1 << 18;
    let mut rng = StdRng::seed_from_u64(0x2946_5a7);
    let mut first_discard_terms = Vec::with_capacity(draws);
    let mut second_discard_terms = Vec::with_capacity(draws);
    let mut active_first_terms = Vec::with_capacity(draws);
    while first_discard_terms.len() < draws {
        let z = standard_normal_vector(&mut rng, 2);
        let independent = standard_normal_vector(&mut rng, 2);
        let output = execute_saturation_block(z.view());
        // Retaining z₂ redraws z₁, and retaining z₁ redraws z₂.
        let without_first = execute_saturation_block(array![independent[0], z[1]].view());
        let without_second = execute_saturation_block(array![z[0], independent[1]].view());
        let difference_first = &output - &without_first;
        let difference_second = &output - &without_second;
        first_discard_terms.push(0.5 * difference_first.dot(&difference_first));
        second_discard_terms.push(0.5 * difference_second.dot(&difference_second));
        // The executed Jacobian's first column is (10 GELU'(10 z₁ − 10), 0), and M = I.
        active_first_terms.push((10.0 * exact_gelu_slope(10.0 * z[0] - 10.0)).powi(2));
    }
    let multiple = standard_error_multiple(3);
    let (first_estimate, first_se) = mean_and_standard_error(&first_discard_terms);
    let (second_estimate, second_se) = mean_and_standard_error(&second_discard_terms);
    let (active_first_estimate, active_first_se) = mean_and_standard_error(&active_first_terms);
    // The baseline-point Jacobian at z = 0, as squared column norms.
    let point_first = (10.0 * exact_gelu_slope(-10.0)).powi(2);
    let point_second = (0.1 * (exact_gelu_slope(0.0) + exact_gelu_slope(0.0))).powi(2);
    eprintln!(
        "#2946 A3 saturation: E(retain z2) = {discard_first} executed MC {first_estimate} se {first_se}; E(retain z1) = {discard_second} executed MC {second_estimate} se {second_se}; ratio {}; active subspace diagonal ({}, {}), off-diagonal {}, executed MC of the first {active_first_estimate} se {active_first_se}; baseline-point Jacobian diagonal ({point_first:e}, {point_second}); multiple {multiple:.3} draws {draws}",
        discard_first / discard_second,
        active[[0, 0]],
        active[[1, 1]],
        active[[0, 1]],
    );
    let arms = [
        ("E(retain z2)", discard_first, first_estimate, first_se),
        ("E(retain z1)", discard_second, second_estimate, second_se),
        ("E[grad F^T M grad F]_11", active[[0, 0]], active_first_estimate, active_first_se),
    ];
    for (label, analytic, estimate, standard_error) in arms {
        assert!(
            (analytic - estimate).abs() <= multiple * standard_error,
            "{label}: analytic {analytic} vs executed Monte Carlo {estimate} ± {standard_error} (multiple {multiple})",
        );
    }
    assert!(
        discard_first > discard_second,
        "the saturated coordinate must carry the discarded error: {discard_first} vs {discard_second}",
    );
    assert!(
        active[[0, 0]] > active[[1, 1]],
        "the active subspace must rank z1 first: {} vs {}",
        active[[0, 0]],
        active[[1, 1]],
    );
    assert!(
        point_first < point_second,
        "the baseline-point Jacobian must rank z2 first: {point_first:e} vs {point_second}",
    );
}

/// Gate readers `W`, gate biases `b`, up readers `A`, up biases `c` and writers `U` of a small SwiGLU block
/// (`d = 4`, `h = 3`, `p = 2`): integers and halves, so every product with a dyadic point is exact.
fn gated_fixture() -> (Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>, Array2<f64>) {
    (
        array![[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, -1.0]],
        array![0.5, -1.0, 0.0],
        array![[0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0], [1.0, -1.0, 1.0, 0.0]],
        array![1.0, 0.0, -0.5],
        array![[1.0, 0.5, -1.0], [0.0, 1.0, 1.0]],
    )
}

fn gated_block(output_bias: Array1<f64>) -> KnownGatedBlock {
    let (gate_readers, gate_biases, up_readers, up_biases, writers) = gated_fixture();
    KnownGatedBlock::new(
        gate_readers,
        gate_biases,
        up_readers,
        up_biases,
        writers,
        output_bias,
        identity(2).view(),
    )
    .expect("a finite gated block with a positive definite metric")
}

/// The executed block `Σ_j u_j (a_jᵀ z + c_j) s(w_jᵀ z + b_j) + c_out`, and its unit terms `(a_jᵀ z + c_j) s(w_jᵀ z + b_j)`.
fn execute_gated_block(output_bias: &Array1<f64>, z: ArrayView1<'_, f64>) -> (Array1<f64>, Array1<f64>) {
    let (gate_readers, gate_biases, up_readers, up_biases, writers) = gated_fixture();
    let units: Array1<f64> = (0..writers.ncols())
        .map(|unit| {
            (up_readers.row(unit).dot(&z) + up_biases[unit])
                * silu_derivatives(gate_readers.row(unit).dot(&z) + gate_biases[unit])[0]
        })
        .collect();
    (writers.dot(&units) + output_bias, units)
}

#[test]
fn the_gated_retained_response_on_the_full_frame_is_the_executed_block() {
    // Q = I discards nothing: v⊥ = 0 and κ⊥ = 0 exactly, the SiLU smoothing at v = 0 is the gate itself with a zero
    // quadrature bound, and each unit term α s(t) + 0·s'(t) is the executed term's word. The two outputs then add the
    // same h terms and the bias in possibly different orders, so they agree within 2γ_{h+1} (Σ_j |u_oj t_j| + |c_o|).
    let output_bias = array![0.25, -0.5];
    let block = gated_block(output_bias.clone());
    let writers = gated_fixture().4;
    let points = array![[0.0, 0.0, 0.0, 0.0], [1.0, -2.0, 0.5, 3.0], [-2.0, 1.0, 1.0, -1.0]];
    let response = block
        .retained_response(identity(4).view(), points.view())
        .expect("finite points");
    let growth = gam_linalg::roundoff::accumulation_growth(writers.ncols() + 1);
    for point in 0..points.nrows() {
        let (executed, units) = execute_gated_block(&output_bias, points.row(point));
        for output in 0..block.output_dim() {
            let magnitude: f64 = (0..units.len())
                .map(|unit| (writers[[output, unit]] * units[unit]).abs())
                .sum::<f64>()
                + output_bias[output].abs();
            let band = 2.0 * growth * magnitude;
            let operator = response.values[[point, output]];
            let quadrature = response.quadrature_band[[point, output]];
            eprintln!(
                "#2946 R9 full frame point {point} output {output}: operator {operator} executed {} rounding band {band:e} quadrature band {quadrature:e}",
                executed[output],
            );
            assert_eq!(quadrature, 0.0, "the full frame smooths nothing, so it has no quadrature error");
            assert!(
                (operator - executed[output]).abs() <= band,
                "point {point} output {output}: operator {operator} vs executed {} beyond {band:e}",
                executed[output],
            );
        }
    }
}

#[test]
fn the_gated_retained_response_is_the_executed_block_averaged_over_the_discarded_input() {
    // R9 at fixed points: F̄_P(Pz) = E F(Pz + (I − P) Z̃), estimated by executing the block. Each arm accepts within
    // its standard-error multiple plus the kernel's derived quadrature band. The positive control is the plug-in
    // F(Pz), which ignores the discarded input and must be rejected at the origin.
    let output_bias = Array1::zeros(2);
    let block = gated_block(output_bias.clone());
    let frame = retained_frame();
    let points = array![[0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 0.5, 2.0], [-2.0, 0.5, 1.0, -1.0]];
    let response = block
        .retained_response(frame.view(), points.view())
        .expect("finite points");
    let draws = 1 << 16;
    let mut rng = StdRng::seed_from_u64(0x2946_a6);
    let multiple = standard_error_multiple(points.nrows() * block.output_dim());
    for point in 0..points.nrows() {
        let retained = project(frame.view(), points.row(point));
        let mut samples = Array2::<f64>::zeros((draws, block.output_dim()));
        for draw in 0..draws {
            let independent = standard_normal_vector(&mut rng, 4);
            let input = &retained + &independent - &project(frame.view(), independent.view());
            samples
                .row_mut(draw)
                .assign(&execute_gated_block(&output_bias, input.view()).0);
        }
        let plug_in = execute_gated_block(&output_bias, retained.view()).0;
        for output in 0..block.output_dim() {
            let column: Vec<f64> = samples.column(output).to_vec();
            let (estimate, standard_error) = mean_and_standard_error(&column);
            let analytic = response.values[[point, output]];
            let band = response.quadrature_band[[point, output]];
            eprintln!(
                "#2946 R9 point {point} output {output}: analytic {analytic} quadrature band {band:e} executed MC {estimate} se {standard_error} z {:.3} plug-in {} multiple {multiple:.3} draws {draws}",
                (analytic - estimate) / standard_error,
                plug_in[output],
            );
            assert!(
                (analytic - estimate).abs() <= multiple * standard_error + band,
                "point {point} output {output}: analytic {analytic} vs executed Monte Carlo {estimate} ± {standard_error}",
            );
            if point == 0 {
                assert!(
                    (plug_in[output] - estimate).abs() > multiple * standard_error + band,
                    "the plug-in F(Pz) = {} must be rejected at the origin against {estimate} ± {standard_error}",
                    plug_in[output],
                );
            }
        }
    }
}

/// `Q S` for the Givens rotation `S` of the frame's two columns by `angle`: the same span through a non-dyadic basis.
fn rotated_basis(frame: ArrayView2<'_, f64>, angle: f64) -> Array2<f64> {
    let (sine, cosine) = angle.sin_cos();
    frame.dot(&array![[cosine, -sine], [sine, cosine]])
}

#[test]
fn every_discarded_error_is_nonnegative_the_empty_frame_discards_the_total_and_two_bases_agree_within_their_bands() {
    // A2 with the operator's band. At P = 0 every pair law is independent, so V(0) = 0 and E(0) = V(I) exactly; the
    // computed V(0) sums K(r = 0) − m_j m_k, zero only up to rounding, so its band must cover it. Every E(P) ≥ 0 within
    // its band. A basis Q S of the same span has the same projector, since its nearest orthonormal frame spans it, so
    // V(Q S) and V(Q) estimate one exact V and must agree within their two bands. Positive control: a frame turned
    // off the span by 2⁻²⁰ changes V by far more than those bands, so the bar resolves a real move.
    let reflector = half_reflector();
    let frame = retained_frame();
    let x = reflector.column(2).to_owned();
    let y = array![0.6, 0.8];
    for activation in [GaussianActivation::Relu, GaussianActivation::ExactGelu] {
        let block = biased_block(activation);
        let total = block.total_variance();
        let empty = reflector.slice(s![.., ..0]);
        let nothing = block.explained_variance(empty).expect("the empty frame");
        let everything_discarded = block.discarded_error(empty).expect("the empty frame");
        let explained = block.explained_variance(frame.view()).expect("an orthonormal frame");
        let rebased = block
            .explained_variance(rotated_basis(frame.view(), 0.7).view())
            .expect("a rotated basis of the frame");
        let turned = block
            .explained_variance(geodesic_frame(frame.view(), x.view(), y.view(), 2.0_f64.powi(-20)).view())
            .expect("a turned frame");
        let errors: Vec<BandedEnergy> = (0..=4)
            .map(|rank| {
                block
                    .discarded_error(reflector.slice(s![.., ..rank]))
                    .expect("a frame of H")
            })
            .collect();
        eprintln!(
            "#2946 A2 band {activation:?}: V(I) = {total:?}; V(0) = {nothing:?}; E(0) = {everything_discarded:?}; V(Q) = {explained:?}; V(Q S) = {rebased:?} (gap {:e}); V(turned 2^-20) = {turned:?} (gap {:e}); E at rank 0..4 = {errors:?}",
            (rebased.value - explained.value).abs(),
            (turned.value - explained.value).abs(),
        );
        assert!(
            nothing.value.abs() <= nothing.band,
            "{activation:?}: V(0) = {nothing:?} must be zero within its band",
        );
        assert!(
            (everything_discarded.value - total.value).abs() <= everything_discarded.band + total.band,
            "{activation:?}: E(0) = {everything_discarded:?} must equal V(I) = {total:?} within their bands",
        );
        for (rank, error) in errors.iter().enumerate() {
            assert!(
                error.value >= -error.band,
                "{activation:?}: E at rank {rank} = {error:?} must be nonnegative within its band",
            );
        }
        assert!(
            (rebased.value - explained.value).abs() <= rebased.band + explained.band,
            "{activation:?}: two bases of one span give V = {explained:?} and {rebased:?}, beyond their bands",
        );
        assert!(
            (turned.value - explained.value).abs() > turned.band + explained.band,
            "{activation:?}: a frame turned by 2^-20 must move V beyond the bands: {explained:?} vs {turned:?}",
        );
    }
}

/// An exact law: variances `variance_x`, `variance_y` and covariance `covariance`, stated with no rounding.
fn exact_pair(mean_x: f64, mean_y: f64, variance_x: f64, variance_y: f64, covariance: f64) -> PreactivationPair {
    PreactivationPair {
        mean_x,
        mean_y,
        variance_x,
        variance_y,
        covariance,
        covariance_rounding: 0.0,
    }
}

/// `K − m_x m_y` and `K`: the subtraction the compose step formed before #4351.
fn raw_pair_covariance(activation: GaussianActivation, pair: PreactivationPair) -> (f64, f64) {
    let kernel = pair_kernel(activation, pair).expect("the pair kernel");
    let mean = |location: f64, variance: f64| {
        let mut value = [0.0];
        gaussian_smoothing_derivatives(activation, location, variance, &mut value).expect("the unit mean");
        value[0]
    };
    (
        kernel.value - mean(pair.mean_x, pair.variance_x) * mean(pair.mean_y, pair.variance_y),
        kernel.value,
    )
}

/// Checks `pair_covariance` against a reference. The value must lie within its band plus the rounding of the
/// reference literal. The band must fall below one rounding of `K`, which the subtraction `K − m_x m_y` can never
/// beat.
fn assert_centred_covariance(
    label: &str,
    activation: GaussianActivation,
    pair: PreactivationPair,
    reference: f64,
) -> (BandedEnergy, f64) {
    let covariance = pair_covariance(activation, pair).expect("the centred pair covariance");
    let (raw, kernel) = raw_pair_covariance(activation, pair);
    let error = (covariance.value - reference).abs();
    let allowed = covariance.band + accumulation_growth(1) * reference.abs();
    eprintln!(
        "#4351 {label} {activation:?}: centred {covariance:?}, reference {reference:e}, error {error:e} (allowed \
         {allowed:e}); raw K − m m = {raw:e}, error {:e}, K = {kernel:e}",
        (raw - reference).abs(),
    );
    assert!(
        error <= allowed,
        "{label}: {covariance:?} misses the reference {reference:e} by {error:e}"
    );
    assert!(
        covariance.value.abs() > covariance.band,
        "{label}: {covariance:?} must resolve the covariance from zero"
    );
    assert!(
        covariance.band < UNIT_ROUNDOFF * kernel.abs(),
        "{label}: the centred band {:e} must fall below one rounding of K = {kernel:e}",
        covariance.band,
    );
    (covariance, raw)
}

#[test]
fn pair_covariance_matches_the_delta_method_at_a_discarded_variance_of_1e_16() {
    // ReLU at `b = 1`, `c = 2` with `s = 1e-8`. Both units sit `10⁸` standard deviations inside their linear piece,
    // so `σ(X) = X` on all but `Φ(−10⁸)` of the mass. The covariance is therefore `r` itself, which is the delta
    // method's `σ'(b) σ'(c) r` with both slopes `1`.
    let variance = 1.0e-16;
    let covariance = 0.3e-16;
    let (_, raw) = assert_centred_covariance(
        "ReLU delta",
        GaussianActivation::Relu,
        exact_pair(1.0, 2.0, variance, variance, covariance),
        covariance,
    );
    // `K ≈ 2` and `m_x m_y ≈ 2` lie in `[1, 4)`, so their computed difference is a multiple of `2⁻⁵² ≈ 2.2e-16`. No
    // such multiple comes within half of `3e-17`, so the subtraction keeps no digit.
    assert!(
        (raw - covariance).abs() > 0.5 * covariance,
        "the raw ReLU subtraction {raw:e} unexpectedly resolved {covariance:e}",
    );

    // Exact GELU at `b = 0.7`, `c = −0.4`, with the same law. The reference is the exact covariance at these f64
    // inputs, to 30 digits. It is `E σ(X) σ(Y)` less the closed-form means, where `E σ(X) σ(Y)` is adaptive quadrature
    // over `E₁` of the closed-form smoothing of `σ` in `E₂`. It was computed at 60 and at 90 digits and agrees to 30
    // digits with the Mehler series `Σ ρⁿ a_n b_n`.
    let exact = 5.779_705_838_062_695_190_220_885e-18;
    let (gelu, raw) = assert_centred_covariance(
        "GELU delta",
        GaussianActivation::ExactGelu,
        exact_pair(0.7, -0.4, variance, variance, covariance),
        exact,
    );
    // `σ'(0.7) σ'(−0.4) r` with `σ' = Φ + t φ`, at 40 digits.
    let delta = 5.779_705_838_062_694_444_815_729e-18;
    // The delta method's remainder has two parts.
    // - The first chaos term. `a_1 = s E σ'(b + sE)`, and `|E σ'(b + sE) − σ'(b)| ≤ s² M₃/2`. Here
    //   `M₃ = sup|σ⁽³⁾| = sup|(t³ − 4t) φ(t)| = 0.778 77…`, attained at `t² = (7 − √33)/2`. So the first term moves by at
    //   most `|ρ| s_x s_y (L (v + w) M₃/2 + v w M₃²/4)`, with `L = sup|σ'|`.
    // - The rest. It is at most `ρ² √(Σ_{n≥2} a_n²) √(Σ_{n≥2} b_n²)`, and
    //   `Σ_{n≥2} a_n² ≤ ½ Σ n(n−1) a_n² = ½ s⁴ E σ''(b + sE)² ≤ ½ s⁴ M₂²`, with `M₂ = sup|(2 − t²) φ(t)| = 2 φ(0)`.
    let slope = GaussianActivation::ExactGelu
        .slope_bound_squared()
        .expect("the GELU slope bound")
        .sqrt();
    let third = 0.7788;
    let second = 0.798;
    let correlation = covariance / variance;
    let remainder = correlation.abs() * variance * (slope * variance * third + variance * variance * third * third / 4.0)
        + correlation * correlation * variance * variance * second * second / 2.0;
    // Forming the remainder takes seventeen roundings.
    let delta_error = (gelu.value - delta).abs();
    let delta_allowed = gelu.band + inflated(remainder, 17) + accumulation_growth(1) * delta;
    eprintln!("#4351 GELU delta method {delta:e}: error {delta_error:e}, allowed {delta_allowed:e}");
    assert!(
        delta_error <= delta_allowed,
        "GELU: {gelu:?} is {delta_error:e} from the delta method {delta:e}"
    );
    // `K ≈ σ(0.7) σ(−0.4) ≈ −0.073` and `m_x m_y` both have magnitude in `[2⁻⁴, 2⁻³)`, so their difference is a
    // multiple of `2⁻⁵⁶ ≈ 1.4e-17`. None comes within half of `5.8e-18`.
    assert!(
        (raw - exact).abs() > 0.5 * exact,
        "the raw GELU subtraction {raw:e} unexpectedly resolved {exact:e}",
    );
}

#[test]
fn pair_covariance_keeps_its_digits_at_small_correlation() {
    // ReLU at zero means, `v = w = 1`, `ρ = 10⁻⁶`. Here `K = (√(1 − ρ²) + ρ(π − acos ρ))/(2π)` is compared with
    // `m² = 1/(2π)`, so the subtraction keeps about ten digits. The reference is that closed form at 50 digits, at
    // the f64 `ρ`.
    assert_centred_covariance(
        "ReLU zero means",
        GaussianActivation::Relu,
        exact_pair(0.0, 0.0, 1.0, 1.0, 1.0e-6),
        2.500_000_795_774_715_346_413_201e-7,
    );
    // ReLU at `b = 0.5`, `c = −0.3`, `ρ = 10⁻⁸`. The reference is quadrature of `E σ(X) σ(Y)` less the means, at 60
    // and at 90 digits. It agrees with the Mehler series of the exact coefficients `a_n = Φ^{(n−1)}(b)/√(n!)`.
    assert_centred_covariance(
        "ReLU biased",
        GaussianActivation::Relu,
        exact_pair(0.5, -0.3, 1.0, 1.0, 1.0e-8),
        2.641_999_091_092_812_169_561_513e-9,
    );
}

#[test]
fn pair_covariance_keeps_its_digits_at_large_means() {
    // `b = c = 10⁶` with unit variances. Both units are linear on all but `Φ(−10⁶)` of the mass, so the covariance is
    // exactly `r`. Meanwhile `K ≈ 10¹²` leaves the subtraction an error near `10⁻⁴`.
    for activation in [GaussianActivation::Relu, GaussianActivation::ExactGelu] {
        for covariance in [0.3, -0.3] {
            let pair = exact_pair(1.0e6, 1.0e6, 1.0, 1.0, covariance);
            let centred = pair_covariance(activation, pair).expect("the centred pair covariance");
            let (raw, kernel) = raw_pair_covariance(activation, pair);
            let error = (centred.value - covariance).abs();
            eprintln!(
                "#4351 large means {activation:?} r = {covariance}: centred {centred:?}, error {error:e}; raw {raw:e}, \
                 error {:e}, K = {kernel:e}",
                (raw - covariance).abs(),
            );
            assert!(
                error <= centred.band,
                "{activation:?}: {centred:?} misses r = {covariance}"
            );
            assert!(centred.value.abs() > centred.band);
            assert!(centred.band < UNIT_ROUNDOFF * kernel.abs());
        }
    }
}

#[test]
fn pair_covariance_keeps_the_closed_form_on_the_diagonal() {
    // `ρ = 1` at zero means. The ReLU tail decays only like `1/N`, so the series cannot beat the closed form. The value
    // is `1/2 − 1/(2π)` within the closed form's band.
    let exact = 0.340_845_056_908_104_664_231_116_2;
    let covariance = pair_covariance(GaussianActivation::Relu, exact_pair(0.0, 0.0, 1.0, 1.0, 1.0))
        .expect("the diagonal pair covariance");
    eprintln!("#4351 diagonal: {covariance:?} against {exact:e}");
    assert!((covariance.value - exact).abs() <= covariance.band + accumulation_growth(1) * exact);
    assert!(covariance.resolved_positive());
}

#[test]
fn pair_covariance_is_exactly_zero_for_independent_units() {
    for activation in [GaussianActivation::Relu, GaussianActivation::ExactGelu] {
        for pair in [
            exact_pair(0.4, -0.2, 0.0, 1.0, 0.0),
            exact_pair(0.4, -0.2, 1.0, 0.5, 0.0),
        ] {
            assert_eq!(
                pair_covariance(activation, pair).expect("an independent pair"),
                BandedEnergy::ZERO
            );
        }
    }
}
