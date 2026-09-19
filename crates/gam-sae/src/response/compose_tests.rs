#![cfg(test)]
//! Composition across known blocks (#2946). The tests cover:
//! - the ReLU² negative control;
//! - affine exactness;
//! - the Gaussian closure's error, measured against the executed composition;
//! - the residual stack on its enlarged state: agreement with the one-block route when nothing is read, and with the
//!   executed stack when it is.
//!
//! The closed-form witnesses read the zero-mean kernels. The retained-point tests read the biased kernels, because a
//! retained point shifts `μ = b + W P z` off zero even when `b = 0`.

use super::{
    ResidualStackError, compose_affine_stage, enlarged_frame, gaussian_closure_response, quadratic_readout,
    relu_two_moment_bound, residual_block_energies, residual_stack_response,
};
use crate::response::subspace::{KnownBlock, smoothing};
use gam_math::gaussian_activation::GaussianActivation;
use gam_linalg::roundoff::accumulation_growth;
use gam_linalg::utils::splitmix64;
use gam_math::probability::{normal_cdf, normal_pdf, standard_normal_from_uniform_bits, standard_normal_quantile};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, array, s};
use std::f64::consts::{PI, TAU};

/// The family-wise probability that any Monte Carlo comparison in one test rejects a true agreement.
const FAMILY_WISE_FALSE_ALARM: f64 = 1e-6;

/// The rounding floor of the zero-mean closed forms compared here. Each composes a few dozen operations of condition
/// `O(1)` (`acos`, `sqrt`, `Φ`, `φ`, products and sums), and every value compared is `O(1)`.
const CLOSED_FORM_FLOOR: f64 = 64.0 * f64::EPSILON;

/// The standard-error multiple for `comparisons` two-sided comparisons at family-wise false alarm
/// `FAMILY_WISE_FALSE_ALARM` (Bonferroni).
fn comparison_multiple(comparisons: usize) -> f64 {
    standard_normal_quantile(1.0 - FAMILY_WISE_FALSE_ALARM / (2.0 * comparisons as f64))
        .expect("the standard normal quantile of a probability inside (0, 1)")
}

/// One standard normal draw from the owner's stream: one SplitMix64 word per draw, continuing across calls.
fn standard_normal(rng: &mut u64) -> f64 {
    standard_normal_from_uniform_bits(splitmix64(rng)).expect("the quantile of an interior dyadic uniform")
}

fn gaussian_matrix(rng: &mut u64, rows: usize, cols: usize, scale: f64) -> Array2<f64> {
    Array2::from_shape_simple_fn((rows, cols), || scale * standard_normal(rng))
}

/// `rank` orthonormal columns in `ℝ^input_dim`, by modified Gram–Schmidt on Gaussian columns.
fn orthonormal_frame(rng: &mut u64, input_dim: usize, rank: usize) -> Array2<f64> {
    let mut frame = gaussian_matrix(rng, input_dim, rank, 1.0);
    for column in 0..rank {
        for previous in 0..column {
            let previous_column = frame.column(previous).to_owned();
            let projection = previous_column.dot(&frame.column(column));
            frame.column_mut(column).scaled_add(-projection, &previous_column);
        }
        let norm = frame.column(column).dot(&frame.column(column)).sqrt();
        frame.column_mut(column).mapv_inplace(|entry| entry / norm);
    }
    frame
}

/// The executed ReLU block `U ReLU(b + W z)`.
fn executed_relu_block(
    readers: &Array2<f64>,
    biases: &Array1<f64>,
    writers: &Array2<f64>,
    input: ArrayView1<'_, f64>,
) -> Array1<f64> {
    writers.dot(&(readers.dot(&input) + biases).mapv(|preactivation| preactivation.max(0.0)))
}

/// The sample mean and standard error of `executed(Z)` over `draws` samples of `Z = P z + (I − P) ξ`,
/// `ξ ~ N(0, I_d)`.
fn executed_conditional_mean(
    rng: &mut u64,
    frame: ArrayView2<'_, f64>,
    point: ArrayView1<'_, f64>,
    draws: usize,
    output_dim: usize,
    executed: impl Fn(ArrayView1<'_, f64>) -> Array1<f64>,
) -> (Array1<f64>, Array1<f64>) {
    let retained = frame.dot(&frame.t().dot(&point));
    let noise = gaussian_matrix(rng, draws, point.len(), 1.0);
    let mut sum = Array1::<f64>::zeros(output_dim);
    let mut square_sum = Array1::<f64>::zeros(output_dim);
    for row in noise.rows() {
        let input = &retained + &row - &frame.dot(&frame.t().dot(&row));
        let output = executed(input.view());
        square_sum += &output.mapv(|value| value * value);
        sum += &output;
    }
    let count = draws as f64;
    let mean = sum / count;
    let variance = (square_sum / count - mean.mapv(|value| value * value)) * (count / (count - 1.0));
    let standard_error = variance.mapv(|value| (value / count).sqrt());
    (mean, standard_error)
}

#[test]
fn relu_square_mean_composition_fails_by_exactly_the_conditional_variance() {
    let identity = array![[1.0]];
    let block = KnownBlock::new(
        array![[1.0]],
        array![0.0],
        array![[1.0]],
        array![0.0],
        identity.view(),
        GaussianActivation::Relu,
    )
    .expect("one ReLU unit is a valid block");
    let frame = Array2::<f64>::zeros((1, 0));
    let points = Array2::<f64>::zeros((1, 1));
    let readout = quadratic_readout(&block, identity.view(), frame.view(), points.view())
        .expect("the quadratic readout of one ReLU unit");
    let second_moment = 0.5;
    let squared_mean = 1.0 / TAU;
    assert!(
        (readout.exact[0] - second_moment).abs() <= CLOSED_FORM_FLOOR,
        "E ReLU(Z)² = {} is not 1/2",
        readout.exact[0]
    );
    assert!(
        (readout.mean_composition[0] - squared_mean).abs() <= CLOSED_FORM_FLOOR,
        "(E ReLU Z)² = {} is not 1/(2π)",
        readout.mean_composition[0]
    );
    assert!(
        (readout.covariance_term[0] - (second_moment - squared_mean)).abs() <= CLOSED_FORM_FLOOR,
        "the mean composition fails by {}, not by Var ReLU(Z) = 1/2 − 1/(2π)",
        readout.covariance_term[0]
    );

    let mut rng: u64 = 0x2946_c001;
    let (executed_mean, standard_error) =
        executed_conditional_mean(&mut rng, frame.view(), points.row(0), 1 << 20, 1, |input| {
            array![input[0].max(0.0).powi(2)]
        });
    let gate = comparison_multiple(2) * standard_error[0];
    eprintln!(
        "#2946 compose ReLU²: exact {} mean composition {} gap {} executed {} ± {} gate {gate}",
        readout.exact[0],
        readout.mean_composition[0],
        readout.covariance_term[0],
        executed_mean[0],
        standard_error[0]
    );
    assert!(
        (executed_mean[0] - readout.exact[0]).abs() <= gate,
        "executed E ReLU(Z)² = {} disagrees with the exact readout {} beyond {gate}",
        executed_mean[0],
        readout.exact[0]
    );
    // Positive control: the same gate rejects the mean composition.
    assert!(
        (executed_mean[0] - readout.mean_composition[0]).abs() > gate,
        "the gate {gate} cannot tell the executed {} from the mean composition {}",
        executed_mean[0],
        readout.mean_composition[0]
    );
}

#[test]
fn affine_last_stage_composes_into_one_block_exactly() {
    let (input_dim, width, output_dim, stage_dim, rank, point_count) = (5, 6, 4, 5, 2, 3);
    let mut rng: u64 = 0x2946_c002;
    let readers = gaussian_matrix(&mut rng, width, input_dim, 1.0);
    let biases = Array1::<f64>::zeros(width);
    let writers = gaussian_matrix(&mut rng, output_dim, width, 1.0);
    let matrix = gaussian_matrix(&mut rng, stage_dim, output_dim, 1.0);
    let offset = gaussian_matrix(&mut rng, stage_dim, 1, 1.0).column(0).to_owned();
    let frame = orthonormal_frame(&mut rng, input_dim, rank);
    let points = gaussian_matrix(&mut rng, point_count, input_dim, 1.0);
    let first = KnownBlock::new(
        readers.clone(),
        biases.clone(),
        writers.clone(),
        Array1::<f64>::zeros(output_dim),
        Array2::<f64>::eye(output_dim).view(),
        GaussianActivation::Relu,
    )
    .expect("a Gaussian ReLU block is valid");
    let stage_metric = Array2::<f64>::eye(stage_dim);
    let composition = compose_affine_stage(&first, matrix.view(), offset.view(), stage_metric.view())
        .expect("a full-column-rank affine stage composes");

    // The retained response of the composition is `A F̄_{1,P} + c`, to the rounding of two nested sums.
    let composed = composition
        .retained_response(frame.view(), points.view())
        .expect("the composed retained response");
    let staged = first
        .retained_response(frame.view(), points.view())
        .expect("the first block's retained response")
        .dot(&matrix.t())
        + &offset;
    let projected_points = points.dot(&frame).dot(&frame.t());
    let retained_preactivations = projected_points.dot(&readers.t()) + &biases;
    let residual_readers = &readers - &readers.dot(&frame).dot(&frame.t());
    let discarded_variances: Array1<f64> = residual_readers.rows().into_iter().map(|row| row.dot(&row)).collect();
    // `T_v ReLU(t) ≤ E|t + √v E| ≤ √(t² + v)` bounds every smoothed unit.
    let unit_bounds = Array2::from_shape_fn(retained_preactivations.dim(), |(point, unit)| {
        retained_preactivations[[point, unit]].hypot(discarded_variances[unit].sqrt())
    });
    let magnitudes = unit_bounds.dot(&writers.mapv(f64::abs).t()).dot(&matrix.mapv(f64::abs).t())
        + &offset.mapv(f64::abs);
    let nested_terms = (output_dim + width + 1) as f64;
    for ((composed_value, staged_value), magnitude) in composed.iter().zip(staged.iter()).zip(magnitudes.iter()) {
        assert!(
            (composed_value - staged_value).abs() <= 2.0 * nested_terms * f64::EPSILON * magnitude,
            "composed response {composed_value} differs from A F̄ + c = {staged_value} beyond rounding"
        );
    }

    // `V(P)` of the composition is `V(P)` of the first block under the pulled-back metric `Aᵀ M A`.
    let pulled_back = matrix.t().dot(&stage_metric).dot(&matrix);
    let pulled_back = (&pulled_back + &pulled_back.t()) * 0.5;
    let pulled = KnownBlock::new(
        readers.clone(),
        biases.clone(),
        writers.clone(),
        Array1::<f64>::zeros(output_dim),
        pulled_back.view(),
        GaussianActivation::Relu,
    )
    .expect("the pulled-back metric of a full-column-rank stage is positive definite");
    let composed_variance = composition
        .explained_variance(frame.view())
        .expect("V(P) of the composition")
        .value;
    let pulled_variance = pulled
        .explained_variance(frame.view())
        .expect("V(P) under the pulled-back metric")
        .value;
    // Both evaluate the same kernels at the same arguments, so only `D = Uᵀ Aᵀ M A U` is rounded differently. For
    // ReLU every bracket obeys `|K − m m| ≤ 2 √((b_j² + v_j)(b_k² + v_k))`.
    let absolute_metric = writers
        .mapv(f64::abs)
        .t()
        .dot(&matrix.mapv(f64::abs).t())
        .dot(&matrix.mapv(f64::abs))
        .dot(&writers.mapv(f64::abs));
    let unit_scales: Array1<f64> = readers
        .rows()
        .into_iter()
        .zip(biases.iter())
        .map(|(row, bias)| (bias * bias + row.dot(&row)).sqrt())
        .collect();
    let pair_magnitude = 2.0 * unit_scales.dot(&absolute_metric.dot(&unit_scales));
    let summed_terms = (width * width + 2 * output_dim + stage_dim + 1) as f64;
    assert!(
        (composed_variance - pulled_variance).abs() <= 2.0 * summed_terms * f64::EPSILON * pair_magnitude,
        "V(P) of the composition {composed_variance} differs from the pulled-back V(P) {pulled_variance}"
    );

    // The executed composition agrees with the exact response, and the same gate rejects the plug-in route
    // `A F₁(Pz) + c`, which ignores the discarded law.
    let multiple = comparison_multiple(2 * point_count * stage_dim);
    let mut plug_in_rejections = 0;
    for (index, point) in points.rows().into_iter().enumerate() {
        let (executed_mean, standard_error) =
            executed_conditional_mean(&mut rng, frame.view(), point, 1 << 16, stage_dim, |input| {
                matrix.dot(&executed_relu_block(&readers, &biases, &writers, input)) + &offset
            });
        let plug_in =
            matrix.dot(&executed_relu_block(&readers, &biases, &writers, projected_points.row(index))) + &offset;
        for coordinate in 0..stage_dim {
            let gate = multiple * standard_error[coordinate];
            assert!(
                (executed_mean[coordinate] - composed[[index, coordinate]]).abs() <= gate,
                "point {index}, coordinate {coordinate}: executed {} disagrees with the exact composition {} beyond {gate}",
                executed_mean[coordinate],
                composed[[index, coordinate]]
            );
            if (executed_mean[coordinate] - plug_in[coordinate]).abs() > gate {
                plug_in_rejections += 1;
            }
        }
    }
    eprintln!(
        "#2946 compose affine: V(P) composed {composed_variance} pulled back {pulled_variance}; plug-in route rejected at \
         {plug_in_rejections} of {} coordinates",
        point_count * stage_dim
    );
    assert!(
        plug_in_rejections > 0,
        "positive control: the gate never rejects the plug-in route A F₁(Pz) + c"
    );
}

#[test]
fn gaussian_closure_error_is_measured_against_the_executed_composition() {
    // Witness at `P = 0`, with zero biases. `F₁(z) = (ReLU(z), ReLU(−z))` and `F₂(y) = ReLU(y₁ + y₂)`, so the executed
    // composition is `|Z|`, with exact mean `m = √(2/π)`. The pre-activation `c = |Z|` has mean `m` and variance
    // `s² = 1 − 2/π`. The closure gives `m Φ(m/s) + s φ(m/s)`.
    let identity = array![[1.0]];
    let first_readers = array![[1.0], [-1.0]];
    let first_biases = Array1::<f64>::zeros(2);
    let first_writers = Array2::<f64>::eye(2);
    let second_readers = array![[1.0, 1.0]];
    let second_biases = Array1::<f64>::zeros(1);
    let second_writers = array![[1.0]];
    let first = KnownBlock::new(
        first_readers.clone(),
        first_biases.clone(),
        first_writers.clone(),
        Array1::<f64>::zeros(2),
        Array2::<f64>::eye(2).view(),
        GaussianActivation::Relu,
    )
    .expect("two opposed ReLU units are a valid block");
    let second = KnownBlock::new(
        second_readers.clone(),
        second_biases.clone(),
        second_writers.clone(),
        array![0.0],
        identity.view(),
        GaussianActivation::Relu,
    )
    .expect("one ReLU readout unit is a valid block");
    let frame = Array2::<f64>::zeros((1, 0));
    let points = Array2::<f64>::zeros((1, 1));
    let closure = gaussian_closure_response(&first, &second, frame.view(), points.view())
        .expect("the Gaussian closure of the absolute-value stack");
    let absolute_mean = (2.0 / PI).sqrt();
    let absolute_variance = 1.0 - 2.0 / PI;
    let deviation = absolute_variance.sqrt();
    let standardized = absolute_mean / deviation;
    let closed_form_closure = absolute_mean * normal_cdf(standardized) + deviation * normal_pdf(standardized);
    assert!(
        (closure.preactivation_means[[0, 0]] - absolute_mean).abs() <= CLOSED_FORM_FLOOR,
        "the pre-activation mean {} is not √(2/π) = {absolute_mean}",
        closure.preactivation_means[[0, 0]]
    );
    assert!(
        (closure.preactivation_variances[[0, 0]] - absolute_variance).abs() <= CLOSED_FORM_FLOOR,
        "the pre-activation variance {} is not 1 − 2/π = {absolute_variance}",
        closure.preactivation_variances[[0, 0]]
    );
    assert!(
        (closure.approximate_response[[0, 0]] - closed_form_closure).abs() <= CLOSED_FORM_FLOOR,
        "the closure {} is not m Φ(m/s) + s φ(m/s) = {closed_form_closure}",
        closure.approximate_response[[0, 0]]
    );
    // The cancellation-free bound agrees with the direct form `½ max(√(m² + s²) − g, g − |m|)`, with `m² + s² = 1`.
    let gaussian_absolute_mean =
        absolute_mean * (2.0 * normal_cdf(standardized) - 1.0) + 2.0 * deviation * normal_pdf(standardized);
    let direct_bound = 0.5 * (1.0 - gaussian_absolute_mean).max(gaussian_absolute_mean - absolute_mean);
    assert!(
        (relu_two_moment_bound(absolute_mean, absolute_variance) - direct_bound).abs() <= CLOSED_FORM_FLOOR,
        "the two-moment bound {} differs from its direct form {direct_bound}",
        relu_two_moment_bound(absolute_mean, absolute_variance)
    );
    let bound = closure
        .relu_moment_class_bound
        .as_ref()
        .expect("a ReLU second stage carries the two-moment bound")[0];
    let defect = closed_form_closure - absolute_mean;
    assert!(
        defect > 0.0 && defect <= bound,
        "the closure's closed-form defect {defect} must be positive and within its two-moment bound {bound}"
    );

    let mut rng: u64 = 0x2946_c003;
    let (executed_mean, standard_error) =
        executed_conditional_mean(&mut rng, frame.view(), points.row(0), 1 << 20, 1, |input| {
            let hidden = executed_relu_block(&first_readers, &first_biases, &first_writers, input);
            executed_relu_block(&second_readers, &second_biases, &second_writers, hidden.view())
        });
    let gate = comparison_multiple(4) * standard_error[0];
    let measured_error = closure.approximate_response[[0, 0]] - executed_mean[0];
    eprintln!(
        "#2946 compose |Z| witness: exact {absolute_mean} closure {} closed-form defect {defect} bound {bound} executed {} \
         ± {} measured error {measured_error} gate {gate}",
        closure.approximate_response[[0, 0]],
        executed_mean[0],
        standard_error[0]
    );
    assert!(
        (executed_mean[0] - absolute_mean).abs() <= gate,
        "the executed composition {} disagrees with the exact √(2/π) = {absolute_mean} beyond {gate}",
        executed_mean[0]
    );
    assert!(
        measured_error.abs() > gate,
        "the executed composition cannot detect the closure's defect {defect}: measured {measured_error}, gate {gate}"
    );
    assert!(
        measured_error.abs() <= bound + gate,
        "measured closure error {measured_error} exceeds the two-moment bound {bound} beyond {gate}"
    );
    // The mean composition `F₂(F̄₁) = ReLU(m) = m` is exact here, because `F₂` is affine on the support of `Y`, so the
    // closure does not dominate it.
    let mean_composition = executed_relu_block(
        &second_readers,
        &second_biases,
        &second_writers,
        first
            .retained_response(frame.view(), points.view())
            .expect("the first block's retained response")
            .row(0),
    )[0];
    assert!(
        (executed_mean[0] - mean_composition).abs() <= gate,
        "the mean composition {mean_composition} disagrees with the executed {} beyond {gate}",
        executed_mean[0]
    );

    // A random two-block ReLU stack at `P = 0`. The measured closure error respects its bound in the output norm, and
    // the same coordinate gates reject the mean composition somewhere.
    let (input_dim, first_width, hidden_dim, second_width, output_dim) = (6, 8, 5, 7, 4);
    let first_readers = gaussian_matrix(&mut rng, first_width, input_dim, 1.0);
    let first_biases = Array1::<f64>::zeros(first_width);
    let first_writers = gaussian_matrix(&mut rng, hidden_dim, first_width, 1.0);
    let second_readers = gaussian_matrix(&mut rng, second_width, hidden_dim, 1.0 / (hidden_dim as f64).sqrt());
    let second_biases = Array1::<f64>::zeros(second_width);
    let second_writers = gaussian_matrix(&mut rng, output_dim, second_width, 1.0);
    let first = KnownBlock::new(
        first_readers.clone(),
        first_biases.clone(),
        first_writers.clone(),
        Array1::<f64>::zeros(hidden_dim),
        Array2::<f64>::eye(hidden_dim).view(),
        GaussianActivation::Relu,
    )
    .expect("a Gaussian ReLU block is valid");
    let second = KnownBlock::new(
        second_readers.clone(),
        second_biases.clone(),
        second_writers.clone(),
        Array1::<f64>::zeros(output_dim),
        Array2::<f64>::eye(output_dim).view(),
        GaussianActivation::Relu,
    )
    .expect("a Gaussian ReLU readout block is valid");
    let frame = Array2::<f64>::zeros((input_dim, 0));
    let points = Array2::<f64>::zeros((1, input_dim));
    let closure = gaussian_closure_response(&first, &second, frame.view(), points.view())
        .expect("the Gaussian closure of a random ReLU stack at P = 0");
    let bound = closure
        .relu_moment_class_bound
        .as_ref()
        .expect("a ReLU second stage carries the two-moment bound")[0];
    let (executed_mean, standard_error) =
        executed_conditional_mean(&mut rng, frame.view(), points.row(0), 1 << 18, output_dim, |input| {
            let hidden = executed_relu_block(&first_readers, &first_biases, &first_writers, input);
            executed_relu_block(&second_readers, &second_biases, &second_writers, hidden.view())
        });
    // Bonferroni over the output coordinates for the bound, and again for the positive control.
    let multiple = comparison_multiple(2 * output_dim);
    let measured_error = &closure.approximate_response.row(0) - &executed_mean;
    let measured_norm = measured_error.dot(&measured_error).sqrt();
    let gate_norm = multiple * standard_error.dot(&standard_error).sqrt();
    eprintln!(
        "#2946 compose random ReLU stack at P = 0: closure {} executed {} ± {}; error norm {measured_norm} bound {bound} \
         gate {gate_norm}",
        closure.approximate_response.row(0),
        executed_mean,
        standard_error
    );
    assert!(
        measured_norm <= bound + gate_norm,
        "measured closure error norm {measured_norm} exceeds the two-moment bound {bound} beyond {gate_norm}"
    );
    let first_means = first
        .retained_response(frame.view(), points.view())
        .expect("the first block's retained response");
    let mean_composition =
        executed_relu_block(&second_readers, &second_biases, &second_writers, first_means.row(0));
    let mean_composition_rejections = (0..output_dim)
        .filter(|&coordinate| {
            (mean_composition[coordinate] - executed_mean[coordinate]).abs() > multiple * standard_error[coordinate]
        })
        .count();
    assert!(
        mean_composition_rejections > 0,
        "positive control: the coordinate gates never reject the mean composition F₂(F̄₁)"
    );
}

#[test]
fn posted_closure_witness_matches_its_posted_digits() {
    // The #2946 witness (comment 5715944769). `F₁ = ReLU` at `P = 0` and `F₂(y) = ReLU(y − μ + s)`, with
    // `μ = E ReLU(Z)` and `s² = Var ReLU(Z)`. Since `c = ReLU(Z) + s − μ > 0`, the executed mean is `s`, and the closure
    // errs by `s (Φ(1) + φ(1) − 1)`.
    // The biased second unit waits on the biased pair kernels, so the closure is read from its owners:
    // - `μ` from the retained response;
    // - `s²` from the exact quadratic readout;
    // - `T_{s²} ReLU(s)` from the smoothing;
    // - the bound from `relu_two_moment_bound`.
    // Both posted values carry seven decimals, so each exact value lies within half a unit of the seventh decimal. The
    // computation adds only its closed-form rounding.
    let posted_defect = 0.0486412;
    let posted_bound = 0.0722718;
    let posted_rounding_band = 0.5e-7 + CLOSED_FORM_FLOOR;
    let identity = array![[1.0]];
    let block = KnownBlock::new(
        array![[1.0]],
        array![0.0],
        array![[1.0]],
        array![0.0],
        identity.view(),
        GaussianActivation::Relu,
    )
    .expect("one ReLU unit is a valid block");
    let frame = Array2::<f64>::zeros((1, 0));
    let points = Array2::<f64>::zeros((1, 1));
    let unit_mean = block
        .retained_response(frame.view(), points.view())
        .expect("E ReLU(Z) from the retained response")[[0, 0]];
    let unit_variance = quadratic_readout(&block, identity.view(), frame.view(), points.view())
        .expect("Var ReLU(Z) from the quadratic readout")
        .covariance_term[0];
    let deviation = unit_variance.sqrt();
    let closure = smoothing(GaussianActivation::Relu, unit_variance, deviation).expect("T_{s²} ReLU(s)");
    let defect = closure - deviation;
    let bound = relu_two_moment_bound(deviation, unit_variance);
    assert!(
        (defect - posted_defect).abs() <= posted_rounding_band,
        "the witness defect {defect} is not the posted {posted_defect}"
    );
    assert!(
        (bound - posted_bound).abs() <= posted_rounding_band,
        "the witness bound {bound} is not the posted {posted_bound}"
    );
    assert!(defect <= bound, "the witness defect {defect} exceeds its bound {bound}");

    let bias = deviation - unit_mean;
    let mut rng: u64 = 0x2946_c004;
    let (executed_mean, standard_error) =
        executed_conditional_mean(&mut rng, frame.view(), points.row(0), 1 << 20, 1, |input| {
            array![(input[0].max(0.0) + bias).max(0.0)]
        });
    let gate = comparison_multiple(3) * standard_error[0];
    let measured_error = closure - executed_mean[0];
    eprintln!(
        "#2946 compose posted witness: s {deviation} closure {closure} defect {defect} bound {bound} executed {} ± {} \
         measured error {measured_error} gate {gate}",
        executed_mean[0],
        standard_error[0]
    );
    assert!(
        (executed_mean[0] - deviation).abs() <= gate,
        "the executed witness {} disagrees with the exact s = {deviation} beyond {gate}",
        executed_mean[0]
    );
    assert!(
        measured_error.abs() > gate,
        "the executed witness cannot detect the closure's defect {defect}: measured {measured_error}, gate {gate}"
    );
    assert!(
        measured_error.abs() <= bound + gate,
        "measured witness error {measured_error} exceeds the bound {bound} beyond {gate}"
    );
}

#[test]
fn residual_discarded_error_carries_a_negative_cross_term_and_stays_nonnegative() {
    // `G(z) = u₁ ReLU(a z₂) + u₂ ReLU(−a z₂)` with `u₁ = −u₂ = −e₂/(2a)` is `−z₂ e₂/2`, so `F(z) = z + G(z) = (z₁, z₂/2)`.
    // Retaining `z₁` discards `E(P) = 1/4`. The skip alone discards `tr(M(I − P)) = 1`, and `G` alone discards
    // `E_G(P) = 1/4`. The cross term is `2 Σ_j u_jᵀ (I − P) w_j Φ(0) = −1`, so dropping it would report `5/4`.
    let scale = 1.5;
    let readers = array![[0.0, scale], [0.0, -scale]];
    let biases = Array1::<f64>::zeros(2);
    let metric = Array2::<f64>::eye(2);
    let frame = array![[1.0], [0.0]];
    let halving = KnownBlock::new(
        readers.clone(),
        biases.clone(),
        array![[0.0, 0.0], [-0.5 / scale, 0.5 / scale]],
        Array1::<f64>::zeros(2),
        metric.view(),
        GaussianActivation::Relu,
    )
    .expect("an opposed ReLU pair is a valid block");
    let energies = residual_block_energies(&halving, frame.view()).expect("residual energies of the halving block");
    eprintln!(
        "#2946 compose residual halving block: E(P) {} V(P) {} cross term {}",
        energies.discarded_error, energies.explained_variance, energies.discarded_cross_term
    );
    assert!(
        (energies.discarded_cross_term + 1.0).abs() <= CLOSED_FORM_FLOOR,
        "the cross term {} is not −1",
        energies.discarded_cross_term
    );
    assert!(
        energies.discarded_error >= 0.0,
        "the residual discarded error {} is negative",
        energies.discarded_error
    );
    assert!(
        (energies.discarded_error - 0.25).abs() <= CLOSED_FORM_FLOOR,
        "the residual discarded error {} is not 1/4",
        energies.discarded_error
    );
    assert!(
        (energies.explained_variance - 1.0).abs() <= CLOSED_FORM_FLOOR,
        "the residual explained variance {} is not 1",
        energies.explained_variance
    );
    assert!(
        (energies.discarded_error - energies.discarded_cross_term - 1.25).abs() <= CLOSED_FORM_FLOOR,
        "without its cross term the error would read {}, not 5/4",
        energies.discarded_error - energies.discarded_cross_term
    );

    // `u₁ = −u₂ = −e₂/a` cancels the skip on `z₂`. Then `F(z) = (z₁, 0)`, so `E(P) = 0` with cross term `−2`.
    let cancelling = KnownBlock::new(
        readers,
        biases,
        array![[0.0, 0.0], [-1.0 / scale, 1.0 / scale]],
        Array1::<f64>::zeros(2),
        metric.view(),
        GaussianActivation::Relu,
    )
    .expect("an opposed ReLU pair is a valid block");
    let energies =
        residual_block_energies(&cancelling, frame.view()).expect("residual energies of the cancelling block");
    assert!(
        (energies.discarded_cross_term + 2.0).abs() <= CLOSED_FORM_FLOOR,
        "the cross term {} is not −2",
        energies.discarded_cross_term
    );
    assert!(
        energies.discarded_error.abs() <= CLOSED_FORM_FLOOR,
        "the cancelling residual block discards {}, not 0",
        energies.discarded_error
    );
}

#[test]
fn residual_discarded_error_matches_the_executed_residual_block() {
    // A random residual block plus an opposed ReLU pair that writes `−z₄` into `e₄`, a discarded direction, so the cross
    // term carries `−2`. The executed `E‖F(Z) − F̄_P(PZ)‖²` must agree with `E(P)`, and the same gate must reject the
    // energy without its cross term.
    let (input_dim, random_width, rank, draws) = (4, 6, 2, 1 << 16);
    let scale = 1.5;
    let mut rng: u64 = 0x2946_c005;
    let mut readers = Array2::<f64>::zeros((random_width + 2, input_dim));
    readers
        .slice_mut(s![..random_width, ..])
        .assign(&gaussian_matrix(&mut rng, random_width, input_dim, 1.0));
    readers[[random_width, 3]] = scale;
    readers[[random_width + 1, 3]] = -scale;
    let mut writers = Array2::<f64>::zeros((input_dim, random_width + 2));
    writers
        .slice_mut(s![.., ..random_width])
        .assign(&gaussian_matrix(&mut rng, input_dim, random_width, 0.2));
    writers[[3, random_width]] = -1.0 / scale;
    writers[[3, random_width + 1]] = 1.0 / scale;
    let biases = Array1::<f64>::zeros(random_width + 2);
    let block = KnownBlock::new(
        readers.clone(),
        biases.clone(),
        writers.clone(),
        Array1::<f64>::zeros(input_dim),
        Array2::<f64>::eye(input_dim).view(),
        GaussianActivation::Relu,
    )
    .expect("a residual ReLU block is valid");
    let frame = Array2::<f64>::eye(input_dim).slice(s![.., ..rank]).to_owned();
    let energies = residual_block_energies(&block, frame.view()).expect("residual energies of the random block");
    assert!(
        energies.discarded_error >= 0.0,
        "the residual discarded error {} is negative",
        energies.discarded_error
    );

    let noise = gaussian_matrix(&mut rng, draws, input_dim, 1.0);
    let retained_points = noise.dot(&frame).dot(&frame.t());
    let retained_response = block
        .retained_response(frame.view(), retained_points.view())
        .expect("the retained response at the sampled points");
    let mut sum = 0.0;
    let mut square_sum = 0.0;
    for (index, row) in noise.rows().into_iter().enumerate() {
        let discarded = &row + &executed_relu_block(&readers, &biases, &writers, row)
            - &retained_points.row(index)
            - &retained_response.row(index);
        let squared_norm = discarded.dot(&discarded);
        sum += squared_norm;
        square_sum += squared_norm * squared_norm;
    }
    let count = draws as f64;
    let executed_error = sum / count;
    let standard_error =
        ((square_sum / count - executed_error * executed_error) * (count / (count - 1.0)) / count).sqrt();
    let gate = comparison_multiple(2) * standard_error;
    eprintln!(
        "#2946 compose residual random block: E(P) {} cross term {} executed {executed_error} ± {standard_error} gate {gate}",
        energies.discarded_error, energies.discarded_cross_term
    );
    assert!(
        (executed_error - energies.discarded_error).abs() <= gate,
        "the executed residual error {executed_error} disagrees with E(P) = {} beyond {gate}",
        energies.discarded_error
    );
    assert!(
        (executed_error - (energies.discarded_error - energies.discarded_cross_term)).abs() > gate,
        "positive control: the gate {gate} cannot reject the energy without its cross term {}",
        energies.discarded_cross_term
    );
}

#[test]
fn quadratic_readout_is_exact_at_retained_points_of_a_biased_block() {
    // A random biased ReLU block with a rank-2 frame. Every unit covariance is a biased pair kernel at the discarded
    // covariance, and the retained point moves each mean. The executed `E[‖F₁(Z)‖² | PZ]` agrees with the exact
    // readout, and the same gate rejects the mean composition `‖F̄₁‖²`.
    let (input_dim, width, output_dim, rank, point_count) = (5, 6, 3, 2, 3);
    let mut rng: u64 = 0x2946_c006;
    let readers = gaussian_matrix(&mut rng, width, input_dim, 1.0);
    let biases = gaussian_matrix(&mut rng, width, 1, 0.5).column(0).to_owned();
    let writers = gaussian_matrix(&mut rng, output_dim, width, 1.0);
    let output_bias = gaussian_matrix(&mut rng, output_dim, 1, 0.5).column(0).to_owned();
    let frame = orthonormal_frame(&mut rng, input_dim, rank);
    let points = gaussian_matrix(&mut rng, point_count, input_dim, 1.0);
    let block = KnownBlock::new(
        readers.clone(),
        biases.clone(),
        writers.clone(),
        output_bias.clone(),
        Array2::<f64>::eye(output_dim).view(),
        GaussianActivation::Relu,
    )
    .expect("a biased Gaussian ReLU block is valid");
    let identity = Array2::<f64>::eye(output_dim);
    let readout = quadratic_readout(&block, identity.view(), frame.view(), points.view())
        .expect("the quadratic readout at retained points");
    let multiple = comparison_multiple(2 * point_count);
    let mut mean_composition_rejections = 0;
    for (index, point) in points.rows().into_iter().enumerate() {
        let (executed_mean, standard_error) =
            executed_conditional_mean(&mut rng, frame.view(), point, 1 << 16, 1, |input| {
                let output = executed_relu_block(&readers, &biases, &writers, input) + &output_bias;
                array![output.dot(&output)]
            });
        let gate = multiple * standard_error[0];
        eprintln!(
            "#2946 compose quadratic readout at point {index}: exact {} mean composition {} executed {} ± {} gate {gate}",
            readout.exact[index],
            readout.mean_composition[index],
            executed_mean[0],
            standard_error[0]
        );
        assert!(
            (executed_mean[0] - readout.exact[index]).abs() <= gate,
            "point {index}: executed {} disagrees with the exact readout {} beyond {gate}",
            executed_mean[0],
            readout.exact[index]
        );
        if (executed_mean[0] - readout.mean_composition[index]).abs() > gate {
            mean_composition_rejections += 1;
        }
    }
    assert!(
        mean_composition_rejections > 0,
        "positive control: the gate never rejects the mean composition ‖F̄₁‖²"
    );
}

#[test]
fn gaussian_closure_error_is_measured_at_retained_points_of_a_biased_stack() {
    // A random biased two-block ReLU stack with a rank-2 frame. At each retained point the measured closure error stays
    // within its two-moment bound, and the same gate rejects the mean composition somewhere.
    let (input_dim, first_width, hidden_dim, second_width, rank, point_count) = (6, 8, 5, 7, 2, 4);
    let mut rng: u64 = 0x2946_c007;
    let first_readers = gaussian_matrix(&mut rng, first_width, input_dim, 1.0);
    let first_biases = gaussian_matrix(&mut rng, first_width, 1, 0.5).column(0).to_owned();
    let first_writers = gaussian_matrix(&mut rng, hidden_dim, first_width, 1.0);
    let first_output_bias = gaussian_matrix(&mut rng, hidden_dim, 1, 0.5).column(0).to_owned();
    let second_readers = gaussian_matrix(&mut rng, second_width, hidden_dim, 1.0 / (hidden_dim as f64).sqrt());
    let second_biases = gaussian_matrix(&mut rng, second_width, 1, 0.5).column(0).to_owned();
    let second_writers = gaussian_matrix(&mut rng, 1, second_width, 1.0);
    let second_output_bias = array![0.25];
    let frame = orthonormal_frame(&mut rng, input_dim, rank);
    let points = gaussian_matrix(&mut rng, point_count, input_dim, 1.0);
    let first = KnownBlock::new(
        first_readers.clone(),
        first_biases.clone(),
        first_writers.clone(),
        first_output_bias.clone(),
        Array2::<f64>::eye(hidden_dim).view(),
        GaussianActivation::Relu,
    )
    .expect("a biased Gaussian ReLU block is valid");
    let second = KnownBlock::new(
        second_readers.clone(),
        second_biases.clone(),
        second_writers.clone(),
        second_output_bias.clone(),
        array![[1.0]].view(),
        GaussianActivation::Relu,
    )
    .expect("a biased scalar ReLU readout block is valid");
    let closure = gaussian_closure_response(&first, &second, frame.view(), points.view())
        .expect("the Gaussian closure at retained points");
    let bounds = closure
        .relu_moment_class_bound
        .as_ref()
        .expect("a ReLU second stage carries the two-moment bound");
    let first_means = first
        .retained_response(frame.view(), points.view())
        .expect("the first block's retained response");
    let multiple = comparison_multiple(2 * point_count);
    let mut mean_composition_rejections = 0;
    for (index, point) in points.rows().into_iter().enumerate() {
        let (executed_mean, standard_error) =
            executed_conditional_mean(&mut rng, frame.view(), point, 1 << 16, 1, |input| {
                let hidden =
                    executed_relu_block(&first_readers, &first_biases, &first_writers, input) + &first_output_bias;
                executed_relu_block(&second_readers, &second_biases, &second_writers, hidden.view())
                    + &second_output_bias
            });
        let gate = multiple * standard_error[0];
        let measured_error = closure.approximate_response[[index, 0]] - executed_mean[0];
        eprintln!(
            "#2946 compose closure at point {index}: closure {} executed {} ± {} measured error {measured_error} bound {} \
             gate {gate}",
            closure.approximate_response[[index, 0]],
            executed_mean[0],
            standard_error[0],
            bounds[index]
        );
        assert!(
            measured_error.abs() <= bounds[index] + gate,
            "point {index}: measured closure error {measured_error} exceeds its two-moment bound {} beyond {gate}",
            bounds[index]
        );
        let mean_composition =
            executed_relu_block(&second_readers, &second_biases, &second_writers, first_means.row(index))
                + &second_output_bias;
        if (mean_composition[0] - executed_mean[0]).abs() > gate {
            mean_composition_rejections += 1;
        }
    }
    assert!(
        mean_composition_rejections > 0,
        "positive control: the gate never rejects the mean composition F₂(F̄₁)"
    );
}

fn gaussian_vector(rng: &mut u64, len: usize, scale: f64) -> Array1<f64> {
    gaussian_matrix(rng, len, 1, scale).column(0).to_owned()
}

/// A symmetric positive definite metric `(A Aᵀ + I)` with its two triangles bitwise equal.
fn spd_metric(rng: &mut u64, dim: usize) -> Array2<f64> {
    let factor = gaussian_matrix(rng, dim, dim, 0.4);
    let product = factor.dot(&factor.t()) + Array2::<f64>::eye(dim);
    (&product + &product.t()) * 0.5
}

/// The executed ReLU layer `U ReLU(b + W y) + c` of a known block.
fn executed_relu_layer(block: &KnownBlock, input: ArrayView1<'_, f64>) -> Array1<f64> {
    block
        .writers()
        .dot(&(block.readers().dot(&input) + &block.biases()).mapv(|preactivation| preactivation.max(0.0)))
        + &block.output_bias()
}

/// The executed ReLU residual stack `z + G₁(z) + G₂(z + G₁(z))`.
fn executed_relu_residual_stack(first: &KnownBlock, second: &KnownBlock, input: ArrayView1<'_, f64>) -> Array1<f64> {
    let stream = &input + &executed_relu_layer(first, input);
    &stream + &executed_relu_layer(second, stream.view())
}

/// The sample mean of a sample and its standard error.
fn sample_mean_and_error(sample: ArrayView1<'_, f64>) -> (f64, f64) {
    let count = sample.len() as f64;
    let mean = sample.sum() / count;
    let variance = sample.iter().map(|value| (value - mean) * (value - mean)).sum::<f64>() / (count - 1.0);
    (mean, (variance / count).sqrt())
}

/// Executed moments of `executed(Z)` over `draws` samples of `Z = P z + (I − P) ξ`, `ξ ~ N(0, I_d)`.
struct ExecutedMoments {
    /// The sample mean and its standard errors, per coordinate.
    mean: Array1<f64>,
    mean_errors: Array1<f64>,
    /// The sample mean of `‖executed(Z) − center‖²_M` and its standard error.
    squared_deviation: f64,
    squared_deviation_error: f64,
}

fn executed_conditional_moments(
    rng: &mut u64,
    frame: ArrayView2<'_, f64>,
    point: ArrayView1<'_, f64>,
    center: ArrayView1<'_, f64>,
    metric: ArrayView2<'_, f64>,
    draws: usize,
    executed: impl Fn(ArrayView1<'_, f64>) -> Array1<f64>,
) -> ExecutedMoments {
    let retained = frame.dot(&frame.t().dot(&point));
    let noise = gaussian_matrix(rng, draws, point.len(), 1.0);
    let mut outputs = Array2::<f64>::zeros((draws, center.len()));
    let mut squared_deviations = Array1::<f64>::zeros(draws);
    for (draw, row) in noise.rows().into_iter().enumerate() {
        let input = &retained + &row - &frame.dot(&frame.t().dot(&row));
        let output = executed(input.view());
        let deviation = &output - &center;
        squared_deviations[draw] = deviation.dot(&metric.dot(&deviation));
        outputs.row_mut(draw).assign(&output);
    }
    let (mean, mean_errors): (Vec<f64>, Vec<f64>) =
        outputs.columns().into_iter().map(|column| sample_mean_and_error(column)).unzip();
    let (squared_deviation, squared_deviation_error) = sample_mean_and_error(squared_deviations.view());
    ExecutedMoments {
        mean: Array1::from(mean),
        mean_errors: Array1::from(mean_errors),
        squared_deviation,
        squared_deviation_error,
    }
}

#[test]
fn a_residual_stack_whose_second_block_reads_no_first_block_write_is_one_residual_block() {
    // `G₁` writes only coordinates `0..3` and `G₂` reads only `3..6`, so `W₂ U₁ = 0` and `N = ∅`. The stack is then the
    // one residual block `B = G₁ ⊕ G₂` with biases `[b₁; b₂ + W₂ c₁]` and output bias `c₁ + c₂`. The stack route must
    // agree with `B`'s retained response at points, and its conditional variance averaged over `PZ` must agree with
    // `B`'s `E(P)`. The same gate must reject `E(P)` without its Stein cross term. So that the cross term is resolvable,
    // `G₂` also holds an opposed ReLU pair on `e₅` that writes `−(ReLU(s y₅) − ReLU(−s y₅))/s = −y₅` into `e₅`,
    // cancelling the skip there, with a cross term of about `−2 M₅₅ (1 − ‖P e₅‖²)`.
    let (input_dim, split, first_width, random_width, rank, point_count, outer_draws) = (6, 3, 4, 3, 2, 3, 1 << 14);
    let (second_width, pair_scale) = (random_width + 2, 1.5);
    let mut rng: u64 = 0x2946_c101;
    let metric = spd_metric(&mut rng, input_dim);
    let first_readers = gaussian_matrix(&mut rng, first_width, input_dim, 1.0);
    let first_biases = gaussian_vector(&mut rng, first_width, 0.5);
    let mut first_writers = Array2::<f64>::zeros((input_dim, first_width));
    first_writers
        .slice_mut(s![..split, ..])
        .assign(&gaussian_matrix(&mut rng, split, first_width, 0.8));
    let first_output_bias = gaussian_vector(&mut rng, input_dim, 0.5);
    let mut second_readers = Array2::<f64>::zeros((second_width, input_dim));
    second_readers
        .slice_mut(s![..random_width, split..])
        .assign(&gaussian_matrix(&mut rng, random_width, input_dim - split, 1.0));
    second_readers[[random_width, input_dim - 1]] = pair_scale;
    second_readers[[random_width + 1, input_dim - 1]] = -pair_scale;
    let mut second_biases = Array1::<f64>::zeros(second_width);
    second_biases
        .slice_mut(s![..random_width])
        .assign(&gaussian_vector(&mut rng, random_width, 0.5));
    let mut second_writers = Array2::<f64>::zeros((input_dim, second_width));
    second_writers
        .slice_mut(s![.., ..random_width])
        .assign(&gaussian_matrix(&mut rng, input_dim, random_width, 0.8));
    second_writers[[input_dim - 1, random_width]] = -1.0 / pair_scale;
    second_writers[[input_dim - 1, random_width + 1]] = 1.0 / pair_scale;
    let second_output_bias = gaussian_vector(&mut rng, input_dim, 0.5);
    let first = KnownBlock::new(
        first_readers.clone(),
        first_biases.clone(),
        first_writers.clone(),
        first_output_bias.clone(),
        metric.view(),
        GaussianActivation::Relu,
    )
    .expect("the first residual block is valid");
    let second = KnownBlock::new(
        second_readers.clone(),
        second_biases.clone(),
        second_writers.clone(),
        second_output_bias.clone(),
        metric.view(),
        GaussianActivation::Relu,
    )
    .expect("the second residual block is valid");
    let width = first_width + second_width;
    let mut joint_readers = Array2::<f64>::zeros((width, input_dim));
    joint_readers.slice_mut(s![..first_width, ..]).assign(&first_readers);
    joint_readers.slice_mut(s![first_width.., ..]).assign(&second_readers);
    let mut joint_biases = Array1::<f64>::zeros(width);
    joint_biases.slice_mut(s![..first_width]).assign(&first_biases);
    joint_biases
        .slice_mut(s![first_width..])
        .assign(&(&second_biases + &second_readers.dot(&first_output_bias)));
    let mut joint_writers = Array2::<f64>::zeros((input_dim, width));
    joint_writers.slice_mut(s![.., ..first_width]).assign(&first_writers);
    joint_writers.slice_mut(s![.., first_width..]).assign(&second_writers);
    let joint_output_bias = &first_output_bias + &second_output_bias;
    let joint = KnownBlock::new(
        joint_readers.clone(),
        joint_biases.clone(),
        joint_writers.clone(),
        joint_output_bias.clone(),
        metric.view(),
        GaussianActivation::Relu,
    )
    .expect("the one residual block G₁ ⊕ G₂ is valid");

    let caller_frame = orthonormal_frame(&mut rng, input_dim, rank);
    let enlarged = enlarged_frame(&first, &second, caller_frame.view()).expect("the enlarged frame of the stack");
    assert!(
        enlarged.read_units.is_empty(),
        "the second block reads no first-block write, yet N = {:?}",
        enlarged.read_units
    );
    assert_eq!(
        enlarged.frame.ncols(),
        rank,
        "an empty N must keep the frame at rank {rank}"
    );
    let frame = enlarged.frame.view();

    let points = gaussian_matrix(&mut rng, point_count, input_dim, 1.0);
    let stack = residual_stack_response(&first, &second, frame, points.view()).expect("the stack route at the points");
    assert!(
        stack.state_leak == 0.0,
        "with N = ∅ nothing leaks, yet Λ = {}",
        stack.state_leak
    );
    let one_block = joint
        .retained_response(frame, points.view())
        .expect("the one-block retained response")
        + &points.dot(&frame).dot(&frame.t());
    // Both routes form each term of a coordinate with at most `d + h + k + 4` rounded operations on a chain, and the
    // smoothing closed forms add their own `CLOSED_FORM_FLOOR`. ReLU's smoothing is 1-Lipschitz in its location,
    // `|τ_u| ≤ |μ_u| + √v_u ≤ β_u + ‖w_u‖ (‖z‖ + 1)` with `β_u` the absolute sum of the bias terms the routes add, and
    // two evaluations of one exact value differ by at most twice one's bound.
    let agreement_growth = 2.0 * (accumulation_growth(input_dim + width + rank + 4) + CLOSED_FORM_FLOOR);
    let mut bias_scales = joint_biases.mapv(f64::abs);
    bias_scales
        .slice_mut(s![first_width..])
        .assign(&(second_biases.mapv(f64::abs) + second_readers.mapv(f64::abs).dot(&first_output_bias.mapv(f64::abs))));
    for point in 0..point_count {
        let z = points.row(point);
        let z_norm = z.dot(&z).sqrt();
        let coordinates = frame.t().dot(&z).mapv(f64::abs);
        let mut resolved = false;
        for coordinate in 0..input_dim {
            let unit_scale: f64 = (0..width)
                .map(|unit| {
                    let reader = joint_readers.row(unit);
                    2.0 * joint_writers[[coordinate, unit]].abs()
                        * (bias_scales[unit] + reader.dot(&reader).sqrt() * (z_norm + 1.0))
                })
                .sum();
            let scale = joint_output_bias[coordinate].abs()
                + frame.row(coordinate).mapv(f64::abs).dot(&coordinates)
                + unit_scale;
            let floor = agreement_growth * scale;
            let (routed, expected) = (stack.conditional_mean[[point, coordinate]], one_block[[point, coordinate]]);
            assert!(
                (routed - expected).abs() <= floor,
                "point {point} coordinate {coordinate}: the stack route's mean {routed} disagrees with the one-block response {expected} beyond {floor}"
            );
            resolved |= expected.abs() > floor;
        }
        assert!(
            resolved,
            "magnitude floor: every coordinate of the one-block response at point {point} is within its rounding floor of zero"
        );
    }

    let energies = residual_block_energies(&joint, frame).expect("the one-block residual energies");
    let outer_points = gaussian_matrix(&mut rng, outer_draws, input_dim, 1.0);
    let outer = residual_stack_response(&first, &second, frame, outer_points.view()).expect("the stack route over PZ");
    let (average, standard_error) = sample_mean_and_error(outer.conditional_variance.view());
    let gate = comparison_multiple(2) * standard_error;
    eprintln!(
        "#2946 compose residual stack, N = ∅: average conditional variance {average} ± {standard_error} against E(P) {} (cross term {}) gate {gate}",
        energies.discarded_error, energies.discarded_cross_term
    );
    assert!(
        energies.discarded_error > gate,
        "magnitude floor: E(P) = {} is not resolved from zero at the gate {gate}",
        energies.discarded_error
    );
    assert!(
        (average - energies.discarded_error).abs() <= gate,
        "the stack route's average conditional variance {average} disagrees with the one-block E(P) = {} beyond {gate}",
        energies.discarded_error
    );
    assert!(
        (average - (energies.discarded_error - energies.discarded_cross_term)).abs() > gate,
        "positive control: the gate {gate} cannot reject E(P) without its Stein cross term {}",
        energies.discarded_cross_term
    );
}

#[test]
fn the_enlarged_frame_makes_a_reading_residual_stack_agree_with_its_executed_output() {
    // `G₂` reads coordinates `0, 1`, where only `G₁`'s units `0, 1` write, so `N = {0, 1}`. On `Q̃ = orth[Q, w₁₀, w₁₁]`
    // those units are functions of `P̃Z`, and the executed `E[out | P̃Z]` and `E[‖out − E[out | P̃Z]‖²_M | P̃Z]` must agree
    // with the stack route at retained points. On the caller's rank-1 frame the route is only the model `out'`: the same
    // gate must reject it, and its measured mean error must stay within its state leak `Λ`.
    let (input_dim, first_width, second_width, rank, point_count, draws) = (5, 4, 3, 1, 3, 1 << 16);
    let mut rng: u64 = 0x2946_c102;
    let metric = Array2::from_diag(&array![1.0, 2.0, 0.5, 1.5, 1.0]);
    let mut first_writers = gaussian_matrix(&mut rng, input_dim, first_width, 1.0);
    first_writers.slice_mut(s![..2, 2..]).fill(0.0);
    let mut second_readers = Array2::<f64>::zeros((second_width, input_dim));
    second_readers
        .slice_mut(s![.., ..2])
        .assign(&gaussian_matrix(&mut rng, second_width, 2, 1.5));
    let first = KnownBlock::new(
        gaussian_matrix(&mut rng, first_width, input_dim, 1.0),
        gaussian_vector(&mut rng, first_width, 0.5),
        first_writers,
        gaussian_vector(&mut rng, input_dim, 0.5),
        metric.view(),
        GaussianActivation::Relu,
    )
    .expect("the first residual block is valid");
    let second = KnownBlock::new(
        second_readers,
        gaussian_vector(&mut rng, second_width, 0.5),
        gaussian_matrix(&mut rng, input_dim, second_width, 1.0),
        gaussian_vector(&mut rng, input_dim, 0.5),
        metric.view(),
        GaussianActivation::Relu,
    )
    .expect("the second residual block is valid");
    let caller_frame = orthonormal_frame(&mut rng, input_dim, rank);
    let enlarged = enlarged_frame(&first, &second, caller_frame.view()).expect("the enlarged frame of the stack");
    assert_eq!(enlarged.read_units, vec![0, 1], "the second block reads exactly the first block's units 0 and 1");
    assert_eq!(
        enlarged.frame.ncols(),
        rank + 2,
        "the enlarged frame retains the caller's direction and both read readers"
    );
    let points = gaussian_matrix(&mut rng, point_count, input_dim, 1.0);
    let exact = residual_stack_response(&first, &second, enlarged.frame.view(), points.view())
        .expect("the stack route on the enlarged frame");
    let model = residual_stack_response(&first, &second, caller_frame.view(), points.view())
        .expect("the stack route on the caller's frame");
    let executed = |input: ArrayView1<'_, f64>| executed_relu_residual_stack(&first, &second, input);
    let multiple = comparison_multiple(point_count * (input_dim + 1));
    // Laurent–Massart: a Gaussian quadratic form with total weight `V/n` exceeds `(1 + 2√x + 2x) V/n` with probability at
    // most `e^{−x}`, so this bounds `‖m̂ − m‖_M` of the sample mean at every point at the family-wise false alarm.
    let tail = (point_count as f64 / FAMILY_WISE_FALSE_ALARM).ln();
    let tail_factor = 1.0 + 2.0 * tail.sqrt() + 2.0 * tail;
    let mut model_rejections = 0;
    for point in 0..point_count {
        let z = points.row(point);
        let on_enlarged = executed_conditional_moments(
            &mut rng,
            enlarged.frame.view(),
            z,
            exact.conditional_mean.row(point),
            metric.view(),
            draws,
            executed,
        );
        let smallest_error = on_enlarged.mean_errors.iter().fold(on_enlarged.squared_deviation_error, |least, &error| least.min(error));
        assert!(
            exact.state_leak < smallest_error,
            "the state leak Λ = {} on the enlarged frame is resolvable at the Monte Carlo's resolution {smallest_error}",
            exact.state_leak
        );
        for coordinate in 0..input_dim {
            let gate = multiple * on_enlarged.mean_errors[coordinate];
            let (routed, measured) = (exact.conditional_mean[[point, coordinate]], on_enlarged.mean[coordinate]);
            assert!(
                (routed - measured).abs() <= gate,
                "point {point} coordinate {coordinate}: the stack route's mean {routed} disagrees with the executed {measured} beyond {gate}"
            );
        }
        let variance_gate = multiple * on_enlarged.squared_deviation_error;
        let routed_variance = exact.conditional_variance[point];
        eprintln!(
            "#2946 compose residual stack on the enlarged frame, point {point}: conditional variance {routed_variance} executed {} ± {} state leak {}",
            on_enlarged.squared_deviation, on_enlarged.squared_deviation_error, exact.state_leak
        );
        assert!(
            routed_variance > variance_gate,
            "magnitude floor: the conditional variance {routed_variance} at point {point} is not resolved from zero at {variance_gate}"
        );
        assert!(
            (routed_variance - on_enlarged.squared_deviation).abs() <= variance_gate,
            "point {point}: the stack route's conditional variance {routed_variance} disagrees with the executed {} beyond {variance_gate}",
            on_enlarged.squared_deviation
        );

        let on_caller = executed_conditional_moments(
            &mut rng,
            caller_frame.view(),
            z,
            model.conditional_mean.row(point),
            metric.view(),
            draws,
            executed,
        );
        let rejected_mean = (0..input_dim).any(|coordinate| {
            (model.conditional_mean[[point, coordinate]] - on_caller.mean[coordinate]).abs()
                > multiple * on_caller.mean_errors[coordinate]
        });
        let rejected_variance = (model.conditional_variance[point] - on_caller.squared_deviation).abs()
            > multiple * on_caller.squared_deviation_error;
        if rejected_mean || rejected_variance {
            model_rejections += 1;
        }
        let mean_error = &on_caller.mean - &model.conditional_mean.row(point);
        let mean_error_norm = mean_error.dot(&metric.dot(&mean_error)).sqrt();
        let spread = (on_caller.squared_deviation - mean_error_norm * mean_error_norm).max(0.0);
        let allowance = (tail_factor * spread / draws as f64).sqrt();
        eprintln!(
            "#2946 compose residual stack on the caller's frame, point {point}: mean error {mean_error_norm} state leak {} allowance {allowance} rejected {}",
            model.state_leak,
            rejected_mean || rejected_variance
        );
        assert!(
            mean_error_norm <= model.state_leak + allowance,
            "point {point}: the model's mean error {mean_error_norm} exceeds its state leak {} plus the sampling allowance {allowance}",
            model.state_leak
        );
    }
    assert!(
        model_rejections > 0,
        "positive control: the gate never rejects the route on the caller's frame, which discards the read readers"
    );
}

#[test]
fn a_dense_second_block_enlarges_the_state_to_the_whole_stream() {
    // Every `G₂` reader reads every coordinate, so `N` is every unit of `G₁`, and its `h₁ = 4 ≥ d − k = 2` readers fill
    // `ℝ³`: the enlarged state is the whole stream, and nothing is compressed.
    let (input_dim, first_width, second_width, rank) = (3, 4, 2, 1);
    let mut rng: u64 = 0x2946_c103;
    let metric = Array2::<f64>::eye(input_dim);
    let first = KnownBlock::new(
        gaussian_matrix(&mut rng, first_width, input_dim, 1.0),
        gaussian_vector(&mut rng, first_width, 0.5),
        gaussian_matrix(&mut rng, input_dim, first_width, 1.0),
        Array1::<f64>::zeros(input_dim),
        metric.view(),
        GaussianActivation::ExactGelu,
    )
    .expect("the first residual block is valid");
    let second = KnownBlock::new(
        gaussian_matrix(&mut rng, second_width, input_dim, 1.0),
        gaussian_vector(&mut rng, second_width, 0.5),
        gaussian_matrix(&mut rng, input_dim, second_width, 1.0),
        Array1::<f64>::zeros(input_dim),
        metric.view(),
        GaussianActivation::ExactGelu,
    )
    .expect("the second residual block is valid");
    let caller_frame = orthonormal_frame(&mut rng, input_dim, rank);
    let enlarged = enlarged_frame(&first, &second, caller_frame.view()).expect("the enlarged frame of the stack");
    assert_eq!(
        enlarged.read_units,
        (0..first_width).collect::<Vec<_>>(),
        "a dense second block reads every unit"
    );
    assert_eq!(
        enlarged.frame.ncols(),
        input_dim,
        "k + |N| = {} readers in ℝ^{input_dim} must enlarge the state to the whole stream",
        rank + first_width
    );
}

#[test]
fn a_residual_stack_of_two_activations_is_refused() {
    let identity = Array2::<f64>::eye(2);
    let block = |activation| {
        KnownBlock::new(
            identity.clone(),
            Array1::<f64>::zeros(2),
            identity.clone(),
            Array1::<f64>::zeros(2),
            identity.view(),
            activation,
        )
        .expect("an identity block is valid")
    };
    let (first, second) = (block(GaussianActivation::Relu), block(GaussianActivation::ExactGelu));
    let refusal = residual_stack_response(
        &first,
        &second,
        Array2::<f64>::zeros((2, 0)).view(),
        Array2::<f64>::zeros((1, 2)).view(),
    )
    .expect_err("a ReLU block followed by a GELU block has no mixed pair kernel");
    assert_eq!(
        refusal,
        ResidualStackError::MixedActivations {
            first: GaussianActivation::Relu,
            second: GaussianActivation::ExactGelu,
        },
        "the refusal must name both activations"
    );
}
