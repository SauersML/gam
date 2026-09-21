#![cfg(test)]
//! Tests for the Hermite coefficients of a known response (#2946 R8).
//!
//! Every tolerance is a derived bound, never a hand constant:
//! - The production bands (running error bounds and absolute shadows) are the tolerance of the
//!   production side. The reference side carries its own running error bound, through the
//!   roundoff owner's `γ_n` (Higham, *ASNA* §3.1-§3.3); libm's `exp`, `pow` and `erfc` are taken
//!   within one ulp.
//! - The kernel owner's coefficient and pair-kernel values carry its published rounding bounds.
//! - A Gauss-Hermite rule's own error enters through its a-posteriori node and weight residuals,
//!   and a non-polynomial integrand is checked against a doubled rule.
//! - Truncation tails are bounded by Cramér's inequality.
//!
//! Each test also shows a mutant failing the same bound, so the bound can fail.

use super::*;
use crate::response::subspace::{KnownBlock, pair_moments};
use gam_linalg::roundoff::accumulation_growth as gamma;
use gam_math::gaussian_activation::{
    GaussianActivation, GaussianActivationError, PreactivationPair, gaussian_hermite_coefficients,
    gaussian_smoothing_derivatives,
};
use gam_math::probability::{normal_cdf, normal_pdf};
use ndarray::{Array1, Array2, Axis, array};
use std::collections::BTreeMap;

/// `start·h_k(x)` for `k < count` from the production recurrence, with running error bounds.
///
/// A step `(x h_k − √k h_{k−1})/√(k+1)` rounds six times (two products, a difference, two roots
/// and a quotient), so it adds `γ_6 (|x h_k| + √k |h_{k−1}|)/√(k+1)` to the propagated error.
fn recurrence_with_error(
    x: f64,
    start: f64,
    start_error: f64,
    count: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut values = vec![0.0; count];
    scaled_hermite_recurrence(x, start, &mut values);
    let mut errors = vec![0.0; count];
    if count > 0 {
        errors[0] = start_error;
    }
    if count > 1 {
        errors[1] = x.abs() * start_error + gamma(1) * values[1].abs();
    }
    for rank in 1..count.saturating_sub(1) {
        let index = rank as f64;
        let divisor = (index + 1.0).sqrt();
        errors[rank + 1] = (x.abs() * errors[rank] + index.sqrt() * errors[rank - 1]) / divisor
            + gamma(6) * (x.abs() * values[rank].abs() + index.sqrt() * values[rank - 1].abs())
                / divisor;
    }
    (values, errors)
}

/// `Φ(t) = erfc(−t/√2)/2` with a one-ulp `erfc` and a rounded argument.
fn normal_cdf_error(t: f64) -> f64 {
    gamma(4) * normal_cdf(t) + gamma(2) * t.abs() * normal_pdf(t)
}

/// `normal_pdf` carries its square exactly (see its doc), leaving `exp`, the constant and the fused
/// correction.
fn normal_pdf_error(t: f64) -> f64 {
    gamma(4) * normal_pdf(t)
}

/// Unnormalized `He_k(x)` for `k < count`, by `He_{k+1} = x He_k − k He_{k−1}`, with running error
/// bounds.
fn unnormalized_hermite_with_error(x: f64, count: usize) -> (Vec<f64>, Vec<f64>) {
    let mut values = vec![0.0; count];
    let mut errors = vec![0.0; count];
    if count > 0 {
        values[0] = 1.0;
    }
    if count > 1 {
        values[1] = x;
    }
    for rank in 1..count.saturating_sub(1) {
        let index = rank as f64;
        values[rank + 1] = x * values[rank] - index * values[rank - 1];
        errors[rank + 1] = x.abs() * errors[rank]
            + index * errors[rank - 1]
            + gamma(3) * (x.abs() * values[rank].abs() + index * values[rank - 1].abs());
    }
    (values, errors)
}

/// The production scaled column `a_0..=a_order` with its bands.
fn coefficients_of(
    activation: GaussianActivation,
    bias: f64,
    scale: f64,
    order: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut values = vec![0.0; order + 1];
    let mut bands = vec![0.0; order + 1];
    let result = gaussian_hermite_coefficients(activation, bias, scale, &mut values, &mut bands);
    assert!(
        result.is_ok(),
        "finite unit (b={bias}, s={scale}) must be accepted"
    );
    (values, bands)
}

fn factorial_root(alpha: &[usize]) -> f64 {
    alpha
        .iter()
        .map(|exponent| (1..=*exponent).map(|rank| (rank as f64).sqrt()).product::<f64>())
        .product()
}

/// The activations whose Hermite coefficients have closed forms.
#[derive(Clone, Copy, Debug)]
enum ClosedForm {
    Relu,
    ExactGelu,
}

impl ClosedForm {
    const ALL: [ClosedForm; 2] = [ClosedForm::Relu, ClosedForm::ExactGelu];

    fn activation(self) -> GaussianActivation {
        match self {
            ClosedForm::Relu => GaussianActivation::Relu,
            ClosedForm::ExactGelu => GaussianActivation::ExactGelu,
        }
    }
}

/// `T_{s²}σ(b)` by R3's closed forms, with a bound: ReLU `bΦ(β) + sφ(β)` and exact GELU
/// `bΦ(x) + (s²/a)φ(x)`, including the rounding of the standardized argument through `T′`.
fn smoothing_reference(form: ClosedForm, bias: f64, scale: f64) -> (f64, f64) {
    match form {
        ClosedForm::Relu => {
            let beta = bias / scale;
            let cdf = normal_cdf(beta);
            let density = normal_pdf(beta);
            let value = bias * cdf + scale * density;
            let error = bias.abs() * normal_cdf_error(beta)
                + scale * normal_pdf_error(beta)
                + gamma(4) * (bias.abs() * cdf + scale * density)
                + scale * cdf * gamma(2) * beta.abs();
            (value, error)
        }
        ClosedForm::ExactGelu => {
            let total = 1.0 + scale * scale;
            let root_total = total.sqrt();
            let standardized = bias / root_total;
            let cdf = normal_cdf(standardized);
            let density = normal_pdf(standardized);
            let value = bias * cdf + scale * scale / root_total * density;
            let slope = cdf + standardized.abs() * density / total;
            let error = bias.abs() * normal_cdf_error(standardized)
                + scale * scale / root_total * normal_pdf_error(standardized)
                + gamma(6) * (bias.abs() * cdf + scale * scale / root_total * density)
                + root_total * slope * gamma(4) * standardized.abs();
            (value, error)
        }
    }
}

/// The ReLU coefficients by the direct integral, never through the Stein derivative route or the
/// normalized recurrence:
/// `a_n √(n!) = ∫_{−β}^∞ (b + se) He_n(e) φ(e) de = φ(β)[b He_{n−1}(−β) + s(He_n(−β) + n He_{n−2}(−β))]`
/// for `n ≥ 2`, and `a_1 = bφ(β) + s(Φ(β) − βφ(β))`. Entry `n − 1` holds `a_n`. The bound adds the
/// mismatch `|b − sβ|` of the rounded `β`, which the closed form assumes away.
fn relu_reference(bias: f64, scale: f64, order: usize) -> (Vec<f64>, Vec<f64>) {
    let beta = bias / scale;
    let mismatch = gamma(2) * bias.abs();
    let (he, he_errors) = unnormalized_hermite_with_error(-beta, order + 1);
    let density = normal_pdf(beta);
    let density_error = normal_pdf_error(beta);
    let cdf = normal_cdf(beta);
    let mut values = vec![0.0; order];
    let mut errors = vec![0.0; order];
    values[0] = bias * density + scale * (cdf - beta * density);
    errors[0] = (bias.abs() + scale * beta.abs()) * density_error
        + scale * normal_cdf_error(beta)
        + gamma(6) * (bias.abs() * density + scale * (cdf + beta.abs() * density))
        + mismatch * density;
    let mut factorial = 1.0;
    for degree in 2..=order {
        let rank = degree as f64;
        factorial *= rank.sqrt();
        let bracket = bias * he[degree - 1] + scale * (he[degree] + rank * he[degree - 2]);
        let magnitude = bias.abs() * he[degree - 1].abs()
            + scale * (he[degree].abs() + rank * he[degree - 2].abs());
        let bracket_error = bias.abs() * he_errors[degree - 1]
            + scale * (he_errors[degree] + rank * he_errors[degree - 2])
            + gamma(5) * magnitude
            + mismatch * he[degree - 1].abs();
        let value = density * bracket / factorial;
        values[degree - 1] = value;
        errors[degree - 1] = (density_error * bracket.abs() + density * bracket_error) / factorial
            + gamma(2 * degree + 2) * value.abs();
    }
    (values, errors)
}

/// Exact GELU through `GELU(t) = tΦ(t)` and `E[Φ(b + sE)] = Φ(b/a)`, never through `T_v GELU`.
///
/// `J_k = E[Φ(b + sE) He_k(E)] = s^k ∂_b^k Φ(b/a) = (−1)^{k−1} ρ^k He_{k−1}(x) φ(x)` for `k ≥ 1`,
/// `J_0 = Φ(x)`, and `E·He_n = He_{n+1} + n He_{n−1}` gives `a_n √(n!) = b J_n + s(J_{n+1} + n J_{n−1})`.
/// Entry `n − 1` holds `a_n`.
fn gelu_reference(bias: f64, scale: f64, order: usize) -> (Vec<f64>, Vec<f64>) {
    let smoothed_scale = (1.0 + scale * scale).sqrt();
    let standardized = bias / smoothed_scale;
    let contraction = scale / smoothed_scale;
    let (he, he_errors) = unnormalized_hermite_with_error(standardized, order + 1);
    let density = normal_pdf(standardized);
    let density_error = normal_pdf_error(standardized);
    let mut smoothed = vec![0.0; order + 2];
    let mut smoothed_errors = vec![0.0; order + 2];
    smoothed[0] = normal_cdf(standardized);
    smoothed_errors[0] = normal_cdf_error(standardized);
    let mut power = 1.0;
    for rank in 1..=order + 1 {
        power *= contraction;
        let sign = if rank % 2 == 1 { 1.0 } else { -1.0 };
        smoothed[rank] = sign * power * he[rank - 1] * density;
        smoothed_errors[rank] = power
            * (he[rank - 1].abs() * density_error + density * he_errors[rank - 1])
            + gamma(2 * rank + 4) * smoothed[rank].abs();
    }
    let mut values = vec![0.0; order];
    let mut errors = vec![0.0; order];
    let mut factorial = 1.0;
    for degree in 1..=order {
        let rank = degree as f64;
        factorial *= rank.sqrt();
        let sum = bias * smoothed[degree]
            + scale * (smoothed[degree + 1] + rank * smoothed[degree - 1]);
        let magnitude = bias.abs() * smoothed[degree].abs()
            + scale * (smoothed[degree + 1].abs() + rank * smoothed[degree - 1].abs());
        let sum_error = bias.abs() * smoothed_errors[degree]
            + scale * (smoothed_errors[degree + 1] + rank * smoothed_errors[degree - 1])
            + gamma(8) * magnitude;
        values[degree - 1] = sum / factorial;
        errors[degree - 1] = sum_error / factorial + gamma(degree + 1) * values[degree - 1].abs();
    }
    (values, errors)
}

/// Bounds on the smoothing owner's raw derivatives `T⁽ᵏ⁾`, `k ≥ 2`, through the unnormalized
/// recurrence its closed forms run on, including the rounding of the standardized argument
/// (`∂_u [He_j(u) φ(u)] = −He_{j+1}(u) φ(u)`).
fn raw_jet_errors(form: ClosedForm, bias: f64, scale: f64, jets: &[f64]) -> Vec<f64> {
    let order = jets.len() - 1;
    let mut bounds = vec![0.0; order + 1];
    match form {
        ClosedForm::Relu => {
            let standardized = bias / scale;
            let (he, he_errors) = unnormalized_hermite_with_error(standardized, order + 1);
            let density = normal_pdf(standardized);
            for rank in 2..=order {
                let divisor = integer_power(scale, rank - 1);
                bounds[rank] = density
                    * (he_errors[rank - 2] + he[rank - 1].abs() * gamma(2) * standardized.abs())
                    / divisor
                    + gamma(rank + 10) * jets[rank].abs();
            }
        }
        ClosedForm::ExactGelu => {
            let total = 1.0 + scale * scale;
            let root_total = total.sqrt();
            let standardized = bias / root_total;
            let (he, he_errors) = unnormalized_hermite_with_error(standardized, order + 2);
            let density = normal_pdf(standardized);
            for rank in 2..=order {
                let divisor = total * integer_power(root_total, rank - 1);
                bounds[rank] = density
                    * (total * he_errors[rank - 2]
                        + he_errors[rank]
                        + (total * he[rank - 1].abs() + he[rank + 1].abs())
                            * gamma(4)
                            * standardized.abs())
                    / divisor
                    + gamma(rank + 12) * jets[rank].abs();
            }
        }
    }
    bounds
}

/// Pins both closed forms, within the production bands, against independent derivations:
/// - the mean against R3's closed forms;
/// - ReLU by the direct truncated integral;
/// - exact GELU by Stein's identity applied to `Φ`, on the unnormalized recurrence;
/// - both against the smoothing owner's raw derivatives `sⁿ T⁽ⁿ⁾/√(n!)`.
///
/// The normalization slip `√(n(n−1)) → n` must fail the same bound.
#[test]
fn unit_coefficients_match_independent_derivations_and_the_smoothing_jets_2946() {
    let order = 24;
    let jet_order = 12;
    let units = [(-1.5, 0.6), (0.0, 1.0), (0.9, 1.8), (2.4, 0.8), (-0.3, 2.5)];
    let mut silu_values = vec![0.0; 3];
    let mut silu_bands = vec![0.0; 3];
    assert!(
        matches!(
            gaussian_hermite_coefficients(
                GaussianActivation::Silu,
                0.2,
                1.0,
                &mut silu_values,
                &mut silu_bands
            ),
            Err(GaussianActivationError::NoClosedForm { .. })
        ),
        "SiLU has no closed form and the coefficient owner must refuse it"
    );
    assert!(
        matches!(
            unit_tail_envelope(GaussianActivation::Silu, 0.2, 1.0, 3),
            Err(HermiteError::NoClosedFormActivation { .. })
        ),
        "SiLU has no tail envelope"
    );
    for form in ClosedForm::ALL {
        let activation = form.activation();
        for (bias, scale) in units {
            let (values, bands) = coefficients_of(activation, bias, scale, order);
            let (mean, mean_error) = smoothing_reference(form, bias, scale);
            assert!(
                (values[0] - mean).abs() <= bands[0] + mean_error,
                "{activation:?} a_0(b={bias}, s={scale}) = {} against R3 {mean}: bound {:e}",
                values[0],
                bands[0] + mean_error
            );
            let (reference, reference_errors) = match form {
                ClosedForm::Relu => relu_reference(bias, scale, order),
                ClosedForm::ExactGelu => gelu_reference(bias, scale, order),
            };
            let mut mutant_rejections = 0;
            for degree in 1..=order {
                let bound = bands[degree] + reference_errors[degree - 1];
                let gap = (values[degree] - reference[degree - 1]).abs();
                assert!(
                    gap <= bound,
                    "{activation:?} a_{degree}(b={bias}, s={scale}) = {} against the independent \
                     derivation {}: gap {gap:e} above the derived bound {bound:e}",
                    values[degree],
                    reference[degree - 1]
                );
                let rank = degree as f64;
                let mutant = if degree >= 2 {
                    values[degree] * (rank * (rank - 1.0)).sqrt() / rank
                } else {
                    values[degree]
                };
                if (mutant - reference[degree - 1]).abs() > bound {
                    mutant_rejections += 1;
                }
            }
            assert!(
                mutant_rejections > 0,
                "the normalization mutant must fail the {activation:?} bound at (b={bias}, s={scale})"
            );
            let mut jets = vec![0.0; jet_order + 1];
            let result = gaussian_smoothing_derivatives(activation, bias, scale * scale, &mut jets);
            assert!(result.is_ok(), "smoothing jets at (b={bias}, s={scale})");
            let jet_bounds = raw_jet_errors(form, bias, scale, &jets);
            for degree in 2..=jet_order {
                let scaling = integer_power(scale, degree) / factorial_root(&[degree]);
                let from_jets = jets[degree] * scaling;
                let bound = bands[degree]
                    + jet_bounds[degree] * scaling
                    + gamma(2 * degree + 2) * from_jets.abs();
                let gap = (values[degree] - from_jets).abs();
                assert!(
                    gap <= bound,
                    "{activation:?} a_{degree}(b={bias}, s={scale}) = {} against the smoothing \
                     owner's s^n T^(n)/sqrt(n!) = {from_jets}: gap {gap:e}, bound {bound:e}",
                    values[degree]
                );
            }
        }
    }
}

type RecurrenceTable = (Vec<f64>, Vec<f64>);

/// `Π_axis h_{α_axis}(x_axis)` at grid nodes from per-node recurrence tables, with its bound.
fn tabled_basis(alpha: &[usize], grid: &[usize], tables: &[RecurrenceTable]) -> (f64, f64) {
    let factors = alpha
        .iter()
        .zip(grid)
        .map(|(exponent, node)| tables[*node].0[*exponent])
        .collect::<Vec<_>>();
    let value = factors.iter().product::<f64>();
    let mut error = gamma(alpha.len()) * value.abs();
    for axis in 0..alpha.len() {
        let others = factors
            .iter()
            .enumerate()
            .filter(|entry| entry.0 != axis)
            .map(|entry| entry.1.abs())
            .product::<f64>();
        error += tables[grid[axis]].1[alpha[axis]] * others;
    }
    (value, error)
}

/// The proposal basis `h_α` is orthonormal under the declared law. Its Gram matrix under a tensor
/// rule exact for per-axis degree `2N` is the identity, within the rule's residuals and the
/// running roundoff. The unnormalized basis `√(α!) h_α` must fail the same bound.
#[test]
fn normalized_hermite_basis_is_orthonormal_under_the_declared_law_2946() {
    let dimension = 2;
    let order = 6;
    let rule = GaussianRule::exact_for_degree(2 * order).expect("rule exact for degree 12");
    let nodes = rule.nodes();
    let weights = rule.weights();
    let node_count = nodes.len();
    let tables = nodes
        .iter()
        .map(|node| recurrence_with_error(*node, 1.0, 0.0, order + 1))
        .collect::<Vec<_>>();
    let indices = multi_indices(dimension, order);
    let mut unnormalized_rejections = 0;
    for left in &indices {
        for right in &indices {
            let mut gram = 0.0;
            let mut bound = 0.0;
            let mut weighted_absolute = 0.0;
            for first in 0..node_count {
                for second in 0..node_count {
                    let grid = [first, second];
                    let point = [nodes[first], nodes[second]];
                    let weight = weights[first] * weights[second];
                    let left_value = hermite_basis_value(left, &point).expect("matching shape");
                    let right_value = hermite_basis_value(right, &point).expect("matching shape");
                    let (left_tabled, left_error) = tabled_basis(left, &grid, &tables);
                    let (right_tabled, right_error) = tabled_basis(right, &grid, &tables);
                    assert!(
                        left_value == left_tabled && right_value == right_tabled,
                        "the basis evaluator must run the tabled recurrence"
                    );
                    let product = left_value * right_value;
                    gram += weight * product;
                    let mut slope = 0.0;
                    for axis in 0..dimension {
                        let table = &tables[grid[axis]].0;
                        let lowered = |exponent: usize| {
                            if exponent > 0 {
                                (exponent as f64).sqrt() * table[exponent - 1]
                            } else {
                                0.0
                            }
                        };
                        let axis_slope = (lowered(left[axis]) * table[right[axis]]
                            + table[left[axis]] * lowered(right[axis]))
                        .abs();
                        let others = (0..dimension)
                            .filter(|other| *other != axis)
                            .map(|other| {
                                let other_table = &tables[grid[other]].0;
                                (other_table[left[other]] * other_table[right[other]]).abs()
                            })
                            .product::<f64>();
                        slope += axis_slope * others;
                    }
                    let weight_error = (weights[first] + rule.weight_residual())
                        * (weights[second] + rule.weight_residual())
                        - weight;
                    bound += weight
                        * (left_value.abs() * right_error
                            + right_value.abs() * left_error
                            + gamma(2) * product.abs())
                        + weight * slope * rule.node_residual()
                        + weight_error * product.abs();
                    weighted_absolute += weight * product.abs();
                }
            }
            bound += gamma(node_count * node_count) * weighted_absolute;
            let expected = if left == right { 1.0 } else { 0.0 };
            assert!(
                (gram - expected).abs() <= bound,
                "<h_{left:?}, h_{right:?}> = {gram:e}, expected {expected}, derived bound {bound:e}"
            );
            let scale = factorial_root(left) * factorial_root(right);
            if (scale * gram - expected).abs() > scale * bound {
                unnormalized_rejections += 1;
            }
        }
    }
    assert!(
        unnormalized_rejections > 0,
        "the unnormalized basis must fail the orthonormality bound"
    );
}

type Polynomial = BTreeMap<Vec<u32>, Vec<f64>>;

fn gaussian_moment(order: u32) -> f64 {
    if order % 2 == 1 {
        return 0.0;
    }
    let mut moment = 1.0;
    let mut factor = 1u32;
    while factor < order {
        moment *= f64::from(factor);
        factor += 2;
    }
    moment
}

fn factorial(n: u32) -> f64 {
    (1..=n).map(f64::from).product()
}

/// `(b + wᵀz)³ = Σ_{a+|e|=3} 3!/(a!·e!) · bᵃ · wᵉ · zᵉ`, times `output`, added to `terms`.
fn add_cubic_unit(terms: &mut Polynomial, bias: f64, reader: &[f64], output: &[f64]) {
    let dims = reader.len();
    for code in 0..4usize.pow(dims as u32) {
        let mut exponents = vec![0u32; dims];
        let mut rest = code;
        for exponent in exponents.iter_mut() {
            *exponent = (rest % 4) as u32;
            rest /= 4;
        }
        let degree = exponents.iter().sum::<u32>();
        if degree > 3 {
            continue;
        }
        let mut scale = 6.0 / factorial(3 - degree) * bias.powi((3 - degree) as i32);
        for (coordinate, exponent) in exponents.iter().enumerate() {
            scale *= reader[coordinate].powi(*exponent as i32) / factorial(*exponent);
        }
        if scale == 0.0 {
            continue;
        }
        let slot = terms
            .entry(exponents)
            .or_insert_with(|| vec![0.0; output.len()]);
        for (entry, weight) in slot.iter_mut().zip(output) {
            *entry += weight * scale;
        }
    }
}

/// `F(z) = Σ_j u_j (b_j + w_jᵀz)³` with integer data, so the oracle's arithmetic is exact.
struct CubicBlock {
    biases: Vec<f64>,
    readers: Array2<f64>,
    outputs: Array2<f64>,
}

impl CubicBlock {
    fn polynomial(&self) -> Polynomial {
        let mut terms = Polynomial::new();
        for (unit, bias) in self.biases.iter().enumerate() {
            let reader = self.readers.row(unit).to_vec();
            let output = self.outputs.column(unit).to_vec();
            add_cubic_unit(&mut terms, *bias, &reader, &output);
        }
        terms
    }

    /// `μ_n = E[σ⁽ⁿ⁾(b + √v·E)]` for `σ(t) = t³`, with `v = ‖w‖²` of the given reader rows.
    fn moments_with_norms_of(&self, readers: ArrayView2<'_, f64>) -> Array2<f64> {
        let mut moments = Array2::<f64>::zeros((self.biases.len(), 4));
        for (unit, bias) in self.biases.iter().enumerate() {
            let variance = readers.row(unit).dot(&readers.row(unit));
            moments[[unit, 0]] = bias.powi(3) + 3.0 * bias * variance;
            moments[[unit, 1]] = 3.0 * (bias * bias + variance);
            moments[[unit, 2]] = 6.0 * bias;
            moments[[unit, 3]] = 6.0;
        }
        moments
    }
}

fn row_norms(readers: ArrayView2<'_, f64>) -> Array1<f64> {
    readers
        .rows()
        .into_iter()
        .map(|row| row.dot(&row).sqrt())
        .collect()
}

/// Blocks `{0,1}`, `{2,3,4}` (a chain: no unit reads both 2 and 4) and `{5}`.
fn planted_block() -> CubicBlock {
    CubicBlock {
        biases: vec![1.0, 0.0, -1.0, 1.0],
        readers: array![
            [1.0, -2.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
        ],
        outputs: array![[1.0, -1.0, 1.0, 1.0], [2.0, 1.0, 0.0, 1.0]],
    }
}

fn planted_metric() -> Array2<f64> {
    array![[2.0, 1.0], [1.0, 1.0]]
}

/// `V(S) = E‖E[F | Z_S] − E F‖²_M` of a polynomial response under `N(0, I)`, by moment algebra,
/// with a band from its absolute shadow.
fn polynomial_retained_variance(
    terms: &Polynomial,
    metric: &Array2<f64>,
    ports: usize,
    retained: &[usize],
) -> BandedEnergy {
    let outputs = metric.nrows();
    let expectation = |exponents: &[u32]| {
        retained
            .iter()
            .map(|port| gaussian_moment(exponents[*port]))
            .product::<f64>()
    };
    let mut conditional: BTreeMap<Vec<u32>, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
    let mut highest = 0u32;
    for (exponents, coefficients) in terms {
        let mut key = exponents.clone();
        let mut factor = 1.0;
        for port in 0..ports {
            highest = highest.max(exponents[port]);
            if !retained.contains(&port) {
                factor *= gaussian_moment(exponents[port]);
                key[port] = 0;
            }
        }
        let slot = conditional
            .entry(key)
            .or_insert_with(|| (vec![0.0; outputs], vec![0.0; outputs]));
        for output in 0..outputs {
            slot.0[output] += coefficients[output] * factor;
            slot.1[output] += coefficients[output].abs() * factor.abs();
        }
    }
    let mut mean = vec![0.0; outputs];
    let mut absolute_mean = vec![0.0; outputs];
    for (exponents, (coefficients, absolute_coefficients)) in &conditional {
        let moment = expectation(exponents);
        for output in 0..outputs {
            mean[output] += coefficients[output] * moment;
            absolute_mean[output] += absolute_coefficients[output] * moment.abs();
        }
    }
    let mut value = 0.0;
    let mut absolute = 0.0;
    for (left, (left_coefficients, left_absolute)) in &conditional {
        for (right, (right_coefficients, right_absolute)) in &conditional {
            let summed = left
                .iter()
                .zip(right)
                .map(|(first, second)| first + second)
                .collect::<Vec<_>>();
            let moment = expectation(&summed);
            for a in 0..outputs {
                for b in 0..outputs {
                    value += left_coefficients[a] * metric[[a, b]] * right_coefficients[b] * moment;
                    absolute +=
                        left_absolute[a] * metric[[a, b]].abs() * right_absolute[b] * moment.abs();
                }
            }
        }
    }
    for a in 0..outputs {
        for b in 0..outputs {
            value -= mean[a] * metric[[a, b]] * mean[b];
            absolute += absolute_mean[a] * metric[[a, b]].abs() * absolute_mean[b];
        }
    }
    let operations = 2 * ports
        + 4 * highest as usize
        + 8
        + terms.len()
        + (conditional.len() * conditional.len() + 1) * outputs * outputs;
    BandedEnergy {
        value,
        band: gamma(operations) * absolute,
    }
}

/// `E[F · He_α(Z)]` per output by moment algebra over the integer Hermite coefficients of `He_n`.
fn polynomial_projection(terms: &Polynomial, alpha: &[usize], outputs: usize) -> Vec<f64> {
    let highest = alpha.iter().copied().max().unwrap_or(0);
    let mut hermite: Vec<Vec<f64>> = vec![vec![1.0], vec![0.0, 1.0]];
    for order in 1..highest.max(1) {
        let mut next = vec![0.0; order + 2];
        for (degree, weight) in hermite[order].iter().enumerate() {
            next[degree + 1] += weight;
        }
        for (degree, weight) in hermite[order - 1].iter().enumerate() {
            next[degree] -= order as f64 * weight;
        }
        hermite.push(next);
    }
    let mut projection = vec![0.0; outputs];
    for (exponents, coefficients) in terms {
        let mut marginal_product = 1.0;
        for (coordinate, power) in alpha.iter().enumerate() {
            let mut marginal = 0.0;
            for (degree, weight) in hermite[*power].iter().enumerate() {
                marginal += weight * gaussian_moment(exponents[coordinate] + degree as u32);
            }
            marginal_product *= marginal;
        }
        for output in 0..outputs {
            projection[output] += coefficients[output] * marginal_product;
        }
    }
    projection
}

/// Every `C_α` of a planted cubic block, built from its moments, equals the exact Gaussian
/// projection `E[F·He_α]/√(α!)`, computed by moment algebra over integer Hermite coefficients. So
/// does the mean. The unnormalized projection `E[F·He_α]` must fail the same bound.
#[test]
fn moment_columns_give_the_gaussian_projections_of_a_cubic_block_2946() {
    let block = planted_block();
    let metric = planted_metric();
    let dims = block.readers.ncols();
    let outputs = block.outputs.nrows();
    let moments = block.moments_with_norms_of(block.readers.view());
    let zero_band = Array2::<f64>::zeros(moments.dim());
    let norms = row_norms(block.readers.view());
    let columns = HermiteColumns::from_moments(
        block.readers.view(),
        norms.view(),
        moments.view(),
        zero_band.view(),
    )
    .expect("the planted block has moment columns");
    assert!(matches!(
        columns.clone().set_order(4),
        Err(HermiteError::NoClosedForm)
    ));
    let no_output_bias = Array1::<f64>::zeros(outputs);
    let response = HermiteResponse::new(
        columns,
        block.outputs.view(),
        no_output_bias.view(),
        metric.view(),
    )
    .expect("matching writers, output bias and metric");
    let polynomial = block.polynomial();
    let (mean, mean_band) = response.mean();
    let expected_mean = polynomial_projection(&polynomial, &vec![0; dims], outputs);
    for output in 0..outputs {
        assert!(
            (mean[output] - expected_mean[output]).abs()
                <= mean_band[output] + gamma(3) * expected_mean[output].abs(),
            "E F[{output}] = {} against the projection {}",
            mean[output],
            expected_mean[output]
        );
    }
    let mut visited = 0usize;
    let mut failures = Vec::new();
    let mut mutant_rejections = 0usize;
    response.for_each_coefficient(|visit| {
        visited += 1;
        let projection = polynomial_projection(&polynomial, visit.alpha, outputs);
        let normalization = factorial_root(visit.alpha);
        for output in 0..outputs {
            let expected = projection[output] / normalization;
            let bound = visit.band[output] + gamma(3) * expected.abs();
            if (visit.value[output] - expected).abs() > bound {
                failures.push((visit.alpha.to_vec(), output, visit.value[output], expected, bound));
            }
            if (visit.value[output] - projection[output]).abs() > bound {
                mutant_rejections += 1;
            }
        }
    });
    assert_eq!(visited, multi_index_count(dims, 3).expect("small count") - 1);
    assert_eq!(visited, 83, "C(6 + 3, 3) − 1");
    assert!(
        failures.is_empty(),
        "coefficients off their projections: {failures:?}"
    );
    assert!(
        mutant_rejections > 0,
        "the unnormalized projection must fail the coefficient bound"
    );
}

/// The Mehler shells equal the enumerated shells, and shells `1..=3` sum to `V([6])`, since the
/// cubic proposal is exact at order 3. Over the retained coordinates `{2,3}` they sum to `V({2,3})`
/// with the full reader norm, and an order-2 proposal leaves exactly shell 3. Retained-norm
/// moments must leave a resolved gap.
#[test]
fn mehler_shells_match_the_enumerated_shells_and_sum_to_the_retained_variance_2946() {
    let block = planted_block();
    let metric = planted_metric();
    let dims = block.readers.ncols();
    let polynomial = block.polynomial();
    let moments = block.moments_with_norms_of(block.readers.view());
    let zero_band = Array2::<f64>::zeros(moments.dim());
    let norms = row_norms(block.readers.view());
    let no_output_bias = Array1::<f64>::zeros(block.outputs.nrows());
    let response = HermiteResponse::new(
        HermiteColumns::from_moments(
            block.readers.view(),
            norms.view(),
            moments.view(),
            zero_band.view(),
        )
        .expect("moment columns"),
        block.outputs.view(),
        no_output_bias.view(),
        metric.view(),
    )
    .expect("matching writers, output bias and metric");
    let shells = response
        .degree_energies(1, 3)
        .expect("degrees within the order");
    let mut enumerated = [(0.0, 0.0, 0.0, 0usize); 3];
    response.for_each_coefficient(|visit| {
        let slot = &mut enumerated[visit.alpha.iter().sum::<usize>() - 1];
        slot.0 += visit.energy.value;
        slot.1 += visit.energy.band;
        slot.2 += visit.energy.value.abs();
        slot.3 += 1;
    });
    for (index, (value, band, absolute, count)) in enumerated.iter().enumerate() {
        let enumerated_band = band + gamma(*count) * absolute;
        assert!(
            (shells[index].value - value).abs() <= shells[index].band + enumerated_band,
            "shell {}: Mehler {:?} against enumerated {value} (band {enumerated_band:e})",
            index + 1,
            shells[index]
        );
    }
    let full = polynomial_retained_variance(&polynomial, &metric, dims, &(0..dims).collect::<Vec<_>>());
    let tail = hermite_truncation_energy(full, &shells);
    assert!(
        tail.value.abs() <= tail.band,
        "the cubic proposal is exact at order 3: tail {tail:?}"
    );

    let retained = [2usize, 3];
    let retained_readers = block.readers.select(Axis(1), &retained);
    let retained_variance = polynomial_retained_variance(&polynomial, &metric, dims, &retained);
    let retained_response = HermiteResponse::new(
        HermiteColumns::from_moments(
            retained_readers.view(),
            norms.view(),
            moments.view(),
            zero_band.view(),
        )
        .expect("full-norm moment columns"),
        block.outputs.view(),
        no_output_bias.view(),
        metric.view(),
    )
    .expect("matching writers, output bias and metric");
    let retained_shells = retained_response
        .degree_energies(1, 3)
        .expect("degrees within the order");
    let retained_tail = hermite_truncation_energy(retained_variance, &retained_shells);
    assert!(
        retained_tail.value.abs() <= retained_tail.band,
        "full-norm moments: tail {retained_tail:?}"
    );
    let order_two = hermite_truncation_energy(retained_variance, &retained_shells[..2]);
    assert!(
        (order_two.value - retained_shells[2].value).abs()
            <= order_two.band + retained_shells[2].band,
        "an order-2 proposal leaves exactly shell 3: {order_two:?} against {:?}",
        retained_shells[2]
    );
    assert!(order_two.resolved_positive());

    let retained_norms = row_norms(retained_readers.view());
    let wrong_moments = block.moments_with_norms_of(retained_readers.view());
    let wrong_response = HermiteResponse::new(
        HermiteColumns::from_moments(
            retained_readers.view(),
            retained_norms.view(),
            wrong_moments.view(),
            zero_band.view(),
        )
        .expect("retained-norm moment columns"),
        block.outputs.view(),
        no_output_bias.view(),
        metric.view(),
    )
    .expect("matching writers, output bias and metric");
    let wrong_shells = wrong_response
        .degree_energies(1, 3)
        .expect("degrees within the order");
    let wrong_tail = hermite_truncation_energy(retained_variance, &wrong_shells);
    assert!(
        wrong_tail.value.abs() > wrong_tail.band,
        "retained-norm moments must miss V({{2,3}}): tail {wrong_tail:?}"
    );
}

/// `Var ReLU(b + sE) = (b² + s²)Φ(β) + bsφ(β) − (bΦ(β) + sφ(β))²`, with its bound.
fn relu_variance(bias: f64, scale: f64) -> (f64, f64) {
    let beta = bias / scale;
    let cdf = normal_cdf(beta);
    let density = normal_pdf(beta);
    let square_scale = bias * bias + scale * scale;
    let second = square_scale * cdf + bias * scale * density;
    let mean = bias * cdf + scale * density;
    let second_error = square_scale * normal_cdf_error(beta)
        + (bias * scale).abs() * normal_pdf_error(beta)
        + gamma(8) * (square_scale * cdf + (bias * scale).abs() * density);
    let mean_error = bias.abs() * normal_cdf_error(beta)
        + scale * normal_pdf_error(beta)
        + gamma(4) * (bias.abs() * cdf + scale * density);
    let variance = second - mean * mean;
    let variance_error = second_error
        + 2.0 * mean.abs() * mean_error
        + gamma(2) * (second + mean * mean)
        + (density * scale + cdf * bias.abs()) * gamma(2) * bias.abs();
    (variance, variance_error)
}

/// `Var GELU(b + sE)` by a probabilists' rule, with the running roundoff of both sums and the rule's
/// node and weight residuals.
fn gelu_variance_by_rule(rule: &GaussianRule, bias: f64, scale: f64) -> (f64, f64) {
    let mut first = 0.0;
    let mut second = 0.0;
    let mut first_error = 0.0;
    let mut second_error = 0.0;
    let mut first_absolute = 0.0;
    for (node, weight) in rule.nodes().iter().zip(rule.weights()) {
        let argument = bias + scale * node;
        let cdf = normal_cdf(argument);
        let activation = argument * cdf;
        let slope = cdf + argument * normal_pdf(argument);
        let activation_error = argument.abs() * normal_cdf_error(argument)
            + gamma(1) * activation.abs()
            + slope.abs() * gamma(2) * (bias.abs() + (scale * node).abs());
        first += weight * activation;
        second += weight * activation * activation;
        first_error += weight * (activation_error + gamma(1) * activation.abs())
            + rule.node_residual() * weight * (scale * slope).abs()
            + rule.weight_residual() * activation.abs();
        second_error += weight
            * (2.0 * activation.abs() * activation_error + gamma(2) * activation * activation)
            + rule.node_residual() * weight * (2.0 * activation * scale * slope).abs()
            + rule.weight_residual() * activation * activation;
        first_absolute += weight * activation.abs();
    }
    let node_count = rule.nodes().len();
    first_error += gamma(node_count) * first_absolute;
    second_error += gamma(node_count) * second;
    let variance = second - first * first;
    let variance_error =
        second_error + 2.0 * first.abs() * first_error + gamma(2) * (second + first * first);
    (variance, variance_error)
}

/// Checks each tail `V − Σ_{n≤N} a_n²` against Cramér's envelope, then checks that the degree-shift
/// mutant breaks it. `coefficients` and `bounds` hold `a_1..=a_N`.
fn assert_tails_within_envelope(
    activation: GaussianActivation,
    bias: f64,
    scale: f64,
    variance: f64,
    variance_error: f64,
    coefficients: &[f64],
    bounds: &[f64],
) {
    let mut captured = 0.0;
    let mut captured_error = 0.0;
    for (index, (value, error)) in coefficients.iter().zip(bounds).enumerate() {
        let degree = index + 1;
        captured += value * value;
        captured_error += 2.0 * value.abs() * error + error * error + gamma(2) * value * value;
        let tail = variance - captured;
        let tolerance = variance_error + captured_error + gamma(degree + 1) * captured;
        let envelope = unit_tail_envelope(activation, bias, scale, degree).expect("finite unit");
        assert!(
            tail >= -tolerance,
            "{activation:?} (b={bias}, s={scale}) tail at N={degree} is {tail:e}, below -{tolerance:e}"
        );
        assert!(
            tail <= envelope + tolerance,
            "{activation:?} (b={bias}, s={scale}) tail at N={degree} is {tail:e}, above the Cramér \
             envelope {envelope:e} + {tolerance:e}"
        );
    }
    let shifted = coefficients[1..].iter().map(|value| value * value).sum::<f64>();
    let last = coefficients.len();
    let envelope = unit_tail_envelope(activation, bias, scale, last - 1).expect("finite unit");
    let tolerance = variance_error + captured_error + gamma(last + 1) * captured;
    assert!(
        variance - shifted > envelope + tolerance,
        "the degree-shift mutant, which misses a_1, must break the {activation:?} envelope at (b={bias}, s={scale})"
    );
}

/// `s² E GELU′(b + sE)²` by a probabilists' rule, with the running roundoff and the rule's node and
/// weight residuals. `GELU′(t) = Φ(t) + tφ(t)` and `GELU″(t) = (2 − t²)φ(t)`.
fn gelu_slope_energy_by_rule(rule: &GaussianRule, bias: f64, scale: f64) -> (f64, f64) {
    let mut energy = 0.0;
    let mut error = 0.0;
    for (node, weight) in rule.nodes().iter().zip(rule.weights()) {
        let argument = bias + scale * node;
        let density = normal_pdf(argument);
        let slope = normal_cdf(argument) + argument * density;
        let curvature = (2.0 - argument * argument) * density;
        let slope_error = normal_cdf_error(argument)
            + argument.abs() * normal_pdf_error(argument)
            + gamma(2) * slope.abs()
            + curvature.abs() * gamma(2) * (bias.abs() + (scale * node).abs());
        energy += weight * slope * slope;
        error += weight * (2.0 * slope.abs() * slope_error + gamma(2) * slope * slope)
            + rule.node_residual() * weight * (2.0 * slope * curvature * scale).abs()
            + rule.weight_residual() * slope * slope;
    }
    error += gamma(rule.nodes().len()) * energy;
    let square_scale = scale * scale;
    (square_scale * energy, square_scale * error + gamma(2) * square_scale * energy)
}

/// A unit's value and slope tail bounds bound its true tails at every order:
/// - `t(n)² ≥ Var − Σ_{m≤n} a_m²` and `d(n)² ≥ s² E σ′² − Σ_{m≤n} m a_m²`;
/// - the references are the closed-form ReLU variance and `s²Φ(β)`, and exact GELU's variance and
///   slope energy by a doubled rule.
///
/// Value tails standing in for slope tails must fall below the slope reference.
#[test]
fn unit_tails_bound_the_value_and_slope_tails_2946() {
    let order = 48;
    let rule = GaussianRule::with_node_count(160).expect("160-node rule");
    let doubled = GaussianRule::with_node_count(320).expect("320-node rule");
    for form in ClosedForm::ALL {
        let activation = form.activation();
        let units = match form {
            ClosedForm::Relu => [(-1.0, 0.8), (0.0, 1.0), (0.75, 1.5)],
            ClosedForm::ExactGelu => [(-0.8, 0.6), (0.0, 1.0), (0.5, 1.2)],
        };
        let mut mutant_rejections = 0;
        for (bias, scale) in units {
            let (values, bands) = coefficients_of(activation, bias, scale, order);
            let (value_tails, slope_tails) =
                unit_tails(activation, bias, scale, &values, &bands).expect("consistent columns");
            let ((variance, variance_error), (slope_energy, slope_error)) = match form {
                ClosedForm::Relu => {
                    let beta = bias / scale;
                    let square_scale = scale * scale;
                    (
                        relu_variance(bias, scale),
                        (
                            square_scale * normal_cdf(beta),
                            square_scale * normal_cdf_error(beta) + gamma(2) * square_scale,
                        ),
                    )
                }
                ClosedForm::ExactGelu => {
                    let (coarse, coarse_error) = gelu_variance_by_rule(&rule, bias, scale);
                    let (fine, fine_error) = gelu_variance_by_rule(&doubled, bias, scale);
                    let (coarse_slope, coarse_slope_error) =
                        gelu_slope_energy_by_rule(&rule, bias, scale);
                    let (fine_slope, fine_slope_error) =
                        gelu_slope_energy_by_rule(&doubled, bias, scale);
                    (
                        (fine, fine_error + coarse_error + (coarse - fine).abs()),
                        (
                            fine_slope,
                            fine_slope_error + coarse_slope_error + (coarse_slope - fine_slope).abs(),
                        ),
                    )
                }
            };
            let mut captured = 0.0;
            let mut captured_error = 0.0;
            let mut slope_captured = 0.0;
            let mut slope_captured_error = 0.0;
            for degree in 0..=order {
                if degree >= 1 {
                    let rank = degree as f64;
                    let value = values[degree];
                    let band = bands[degree];
                    captured += value * value;
                    captured_error += 2.0 * value.abs() * band + band * band;
                    slope_captured += rank * value * value;
                    slope_captured_error += rank * (2.0 * value.abs() * band + band * band);
                }
                let value_reference = variance - captured;
                let value_tolerance =
                    variance_error + captured_error + gamma(degree + 2) * (variance + captured);
                let slope_reference = slope_energy - slope_captured;
                let slope_tolerance = slope_error
                    + slope_captured_error
                    + gamma(degree + 2) * (slope_energy + slope_captured);
                assert!(
                    value_tails[degree] * value_tails[degree] >= value_reference - value_tolerance,
                    "{activation:?} (b={bias}, s={scale}) t({degree})² = {:e} below the value tail \
                     {value_reference:e} - {value_tolerance:e}",
                    value_tails[degree] * value_tails[degree]
                );
                assert!(
                    slope_tails[degree] * slope_tails[degree] >= slope_reference - slope_tolerance,
                    "{activation:?} (b={bias}, s={scale}) d({degree})² = {:e} below the slope tail \
                     {slope_reference:e} - {slope_tolerance:e}",
                    slope_tails[degree] * slope_tails[degree]
                );
                if value_tails[degree] * value_tails[degree] < slope_reference - slope_tolerance {
                    mutant_rejections += 1;
                }
            }
        }
        assert!(
            mutant_rejections > 0,
            "value tails standing in for {activation:?} slope tails must fall below the slope reference"
        );
    }
}

/// A unit's captured energy converges to its variance, within Cramér's envelope at every order.
/// ReLU uses its closed-form variance and exact GELU a doubled rule; the degree-shift mutant must
/// break the envelope.
#[test]
fn unit_hermite_energy_converges_to_the_unit_variance_within_the_cramer_envelope_2946() {
    let relu_order = 400;
    for (bias, scale) in [(-1.0, 0.8), (0.0, 1.0), (0.75, 1.5)] {
        let (variance, variance_error) = relu_variance(bias, scale);
        let (values, bands) = coefficients_of(GaussianActivation::Relu, bias, scale, relu_order);
        assert_tails_within_envelope(
            GaussianActivation::Relu,
            bias,
            scale,
            variance,
            variance_error,
            &values[1..],
            &bands[1..],
        );
    }
    let gelu_order = 64;
    let rule = GaussianRule::with_node_count(160).expect("160-node rule");
    let doubled = GaussianRule::with_node_count(320).expect("320-node rule");
    for (bias, scale) in [(-0.8, 0.6), (0.0, 1.0), (0.5, 1.2)] {
        let (coarse, coarse_error) = gelu_variance_by_rule(&rule, bias, scale);
        let (fine, fine_error) = gelu_variance_by_rule(&doubled, bias, scale);
        let resolution = (coarse - fine).abs();
        assert!(
            resolution <= coarse_error + fine_error,
            "GELU variance at (b={bias}, s={scale}) is unresolved between 160 and 320 nodes: \
             {resolution:e} above roundoff {:e}",
            coarse_error + fine_error
        );
        let (values, bands) =
            coefficients_of(GaussianActivation::ExactGelu, bias, scale, gelu_order);
        assert_tails_within_envelope(
            GaussianActivation::ExactGelu,
            bias,
            scale,
            fine,
            fine_error + resolution,
            &values[1..],
            &bands[1..],
        );
    }
}

/// A block with zero unit biases, so its `V` carries the zero-mean band of
/// [`zero_mean_variance_band`]. Its output bias is nonzero, and moves no energy on either side.
fn zero_bias_block(activation: GaussianActivation) -> KnownBlock {
    let readers = array![
        [0.9, -0.4, 0.3],
        [0.2, 1.1, -0.5],
        [-0.7, 0.3, 0.8],
        [0.5, 0.5, 0.5]
    ];
    let writers = array![[1.0, -0.5, 0.25, 0.8], [0.3, 0.9, -1.1, 0.4]];
    let metric = array![[2.0, 0.3], [0.3, 1.0]];
    KnownBlock::new(
        readers,
        Array1::zeros(4),
        writers,
        array![0.4, -1.2],
        metric.view(),
        activation,
    )
    .expect("zero-bias block")
}

fn retained_frame() -> Array2<f64> {
    array![[0.6, 0.0], [0.8, 0.0], [0.0, 1.0]]
}

/// A bound on a zero-mean block's `V` as the retained-response operator forms it,
/// `Σ_jk D_jk (K_jk − m_j m_k)`. It covers the kernel owner's published rounding of `K` and of each
/// `m_j`, the rounding of each covariance through `∂_r K`, and `γ_{h² + p + 8}` for forming `D` and
/// summing the pairs.
fn zero_mean_variance_band(block: &KnownBlock, coordinates: &Array2<f64>) -> f64 {
    let units = block.width();
    let activation = block.activation();
    let readers = block.readers();
    let writers = block.writers();
    let metric_writers = block.metric_writers();
    let retained = coordinates.ncols();
    let mut band = 0.0;
    let mut absolute = 0.0;
    for left in 0..units {
        let left_variance = readers.row(left).dot(&readers.row(left));
        let (left_column, left_bands) = coefficients_of(activation, 0.0, left_variance.sqrt(), 0);
        for right in 0..units {
            let right_variance = readers.row(right).dot(&readers.row(right));
            let (right_column, right_bands) =
                coefficients_of(activation, 0.0, right_variance.sqrt(), 0);
            let (left_mean, right_mean) = (left_column[0], right_column[0]);
            let covariance = coordinates.row(left).dot(&coordinates.row(right));
            let covariance_scale = coordinates
                .row(left)
                .iter()
                .zip(coordinates.row(right).iter())
                .map(|(first, second)| (first * second).abs())
                .sum::<f64>();
            let moments = pair_moments(
                activation,
                PreactivationPair {
                    mean_x: 0.0,
                    mean_y: 0.0,
                    variance_x: left_variance,
                    variance_y: right_variance,
                    covariance,
                    covariance_rounding: gamma(retained + 2) * covariance_scale,
                },
            )
            .expect("zero-mean pair kernel");
            let coupling = writers.column(left).dot(&metric_writers.column(right)).abs();
            let product = (left_mean * right_mean).abs();
            band += coupling
                * (moments.value_rounding
                    + left_mean.abs() * right_bands[0]
                    + left_bands[0] * right_mean.abs()
                    + moments.covariance_derivative.abs() * gamma(retained + 2) * covariance_scale);
            absolute += coupling * (moments.value.abs() + product);
        }
    }
    band + gamma(units * units + block.output_dim() + 8) * absolute
}

/// The block's shells converge to the retained-response operator's `V(I)` and `V(P)`, within the
/// block envelope at every order. This is Mehler's formula `K_σ − m_j m_k = Σ_{n≥1} a_{j,n} a_{k,n} ρⁿ`
/// tested against the kernel owner's zero-mean pair kernels. The degree-shift mutant must break
/// the envelope.
#[test]
fn degree_energies_converge_to_the_blocks_explained_variance_within_the_envelope_2946() {
    let frames = [Array2::<f64>::eye(3), retained_frame()];
    for form in ClosedForm::ALL {
        let activation = form.activation();
        let block = zero_bias_block(activation);
        let order = match form {
            ClosedForm::Relu => 200,
            ClosedForm::ExactGelu => 80,
        };
        for frame in &frames {
            let variance = block
                .explained_variance(frame.view())
                .expect("admissible frame")
                .value;
            let coordinates = block.readers().dot(frame);
            let total = BandedEnergy {
                value: variance,
                band: zero_mean_variance_band(&block, &coordinates),
            };
            if frame.ncols() == block.input_dim() {
                let gap = (variance - block.total_variance().value).abs();
                assert!(
                    gap <= 2.0 * total.band,
                    "{activation:?} V(I) through the identity frame {variance} against the block's \
                     total {}: gap {gap:e}, band {:e}",
                    block.total_variance().value,
                    2.0 * total.band
                );
            }
            let response =
                HermiteResponse::for_block(&block, frame.view(), order).expect("admissible frame");
            let shells = response
                .degree_energies(1, order)
                .expect("degrees within the order");
            for degree in 1..=order {
                let tail = hermite_truncation_energy(total, &shells[..degree]);
                let envelope = response.tail_envelope(degree).expect("closed-form columns");
                assert!(
                    tail.value >= -tail.band,
                    "{activation:?} k={} tail at N={degree} is {tail:?}, below zero",
                    frame.ncols()
                );
                assert!(
                    tail.value <= envelope + tail.band,
                    "{activation:?} k={} tail at N={degree} is {tail:?}, above the envelope {envelope:e}",
                    frame.ncols()
                );
            }
            let shifted = hermite_truncation_energy(total, &shells[1..]);
            let envelope = response.tail_envelope(order).expect("closed-form columns");
            assert!(
                shifted.value > envelope + shifted.band,
                "the degree-shift mutant, which misses shell 1, must break the {activation:?} envelope"
            );
        }
    }
}

/// The derived bound on a tensor-rule projection of `f = Σ_monomials Π_i y_i^{e_i}` onto `h_α`.
/// It covers each term's roundoff, the node residual times `Σ W |∂(f h_α)|`, the weight residual's
/// effect on the product weight, and `γ` for the sum.
fn projection_bound(rule: &GaussianRule, monomials: &[[usize; 3]], alpha: &[usize]) -> f64 {
    let nodes = rule.nodes();
    let weights = rule.weights();
    let node_count = nodes.len();
    let highest = alpha.iter().copied().max().unwrap_or(0);
    let tables = nodes
        .iter()
        .map(|node| recurrence_with_error(*node, 1.0, 0.0, highest + 1))
        .collect::<Vec<_>>();
    let mut bound = 0.0;
    let mut weighted_absolute = 0.0;
    for point in 0..node_count.pow(3) {
        let grid = [
            point % node_count,
            (point / node_count) % node_count,
            point / (node_count * node_count),
        ];
        let weight = grid.iter().map(|node| weights[*node]).product::<f64>();
        let weight_error = grid
            .iter()
            .map(|node| weights[*node] + rule.weight_residual())
            .product::<f64>()
            - weight;
        for monomial in monomials {
            let factors = (0..3)
                .map(|axis| {
                    let node = nodes[grid[axis]];
                    let power = if monomial[axis] == 1 { node } else { 1.0 };
                    power * tables[grid[axis]].0[alpha[axis]]
                })
                .collect::<Vec<_>>();
            let product = factors.iter().product::<f64>();
            let mut product_error = gamma(6) * product.abs();
            let mut slope = 0.0;
            for axis in 0..3 {
                let node = nodes[grid[axis]];
                let table = &tables[grid[axis]];
                let power = if monomial[axis] == 1 { node.abs() } else { 1.0 };
                let others = (0..3)
                    .filter(|other| *other != axis)
                    .map(|other| factors[other].abs())
                    .product::<f64>();
                product_error += power * table.1[alpha[axis]] * others;
                let lowered = if alpha[axis] > 0 {
                    (alpha[axis] as f64).sqrt() * table.0[alpha[axis] - 1]
                } else {
                    0.0
                };
                let axis_slope = if monomial[axis] == 1 {
                    table.0[alpha[axis]] + node * lowered
                } else {
                    lowered
                };
                slope += axis_slope.abs() * others;
            }
            bound += weight * product_error
                + weight * slope * rule.node_residual()
                + weight_error * product.abs();
            weighted_absolute += weight * product.abs();
        }
    }
    bound + gamma(node_count.pow(3) * monomials.len()) * weighted_absolute
}

/// `U₁U₂U₃` projects onto exactly one coefficient, `α = (1,1,1)`, so its only ANOVA term is the
/// three-way one and every pair-only term is zero. The additive control `U₁U₂ + U₃` must fail the
/// same three-way detector.
#[test]
fn pure_three_way_product_has_its_only_coefficient_at_one_one_one_2946() {
    let order = 4;
    let rule = GaussianRule::exact_for_degree(1 + order).expect("rule exact for degree 5");
    let product = [[1, 1, 1]];
    let coefficients = project_response(3, order, 1, &rule, |y, out| out[0] = y[0] * y[1] * y[2])
        .expect("representable grid");
    assert_eq!(coefficients.len(), multi_indices(3, order).len());
    let mut supports: BTreeMap<Vec<usize>, (f64, f64)> = BTreeMap::new();
    for coefficient in &coefficients {
        let bound = projection_bound(&rule, &product, &coefficient.alpha);
        let expected = if coefficient.alpha == [1, 1, 1] { 1.0 } else { 0.0 };
        let value = coefficient.value[0];
        assert!(
            (value - expected).abs() <= bound,
            "U1U2U3 c_{:?} = {value:e}, expected {expected}, bound {bound:e}",
            coefficient.alpha
        );
        let slot = supports
            .entry(support_of(&coefficient.alpha))
            .or_insert((0.0, 0.0));
        slot.0 += value * value;
        slot.1 += (expected + bound) * (expected + bound);
    }
    for (support, (energy, allowance)) in &supports {
        if support.len() < 3 {
            assert!(
                energy <= allowance,
                "U1U2U3 support {support:?} carries {energy:e}, above its roundoff allowance {allowance:e}"
            );
        }
    }
    let control = [[1, 1, 0], [0, 0, 1]];
    let control_coefficients =
        project_response(3, order, 1, &rule, |y, out| out[0] = y[0] * y[1] + y[2])
            .expect("representable grid");
    let mut three_way_rejected = false;
    for coefficient in &control_coefficients {
        let bound = projection_bound(&rule, &control, &coefficient.alpha);
        let expected = if coefficient.alpha == [1, 1, 0] || coefficient.alpha == [0, 0, 1] {
            1.0
        } else {
            0.0
        };
        let value = coefficient.value[0];
        assert!(
            (value - expected).abs() <= bound,
            "U1U2+U3 c_{:?} = {value:e}, expected {expected}, bound {bound:e}",
            coefficient.alpha
        );
        if coefficient.alpha == [1, 1, 1] && (value - 1.0).abs() > bound {
            three_way_rejected = true;
        }
    }
    assert!(
        three_way_rejected,
        "the additive control must fail the three-way detector"
    );
}

/// Pins the budgeted order on a scalar unit:
/// - the selected order is the smallest whose banded truncation error against the block's `V(I)`
///   meets the declared budget;
/// - a scalar unit's shells are its squared coefficients;
/// - the materialized proposal carries the captured energy and the mean, output bias included, and
///   a mean without the output bias fails the same bound;
/// - a total the shells can never reach is refused at the envelope order;
/// - an unaddressable proposal count is refused.
#[test]
fn order_is_the_smallest_meeting_the_declared_budget_and_an_inconsistent_total_is_refused_2946() {
    let scale = 1.3;
    let output_bias = 0.7;
    let metric = array![[1.0]];
    let block = KnownBlock::new(
        array![[scale]],
        array![0.0],
        array![[1.0]],
        array![output_bias],
        metric.view(),
        GaussianActivation::Relu,
    )
    .expect("scalar block");
    let frame = array![[1.0]];
    let budget = FidelityBudget {
        unexplained_fraction: 1.0e-3,
    };
    let (response, selection) =
        select_order(&block, frame.view(), budget).expect("consistent total");
    let variance = selection.total_energy.value;
    let (closed_form, closed_form_error) = relu_variance(0.0, scale);
    assert!(
        (variance - closed_form).abs() <= closed_form_error + gamma(24) * variance,
        "the block's V(I) {variance} must match the closed-form unit variance {closed_form}"
    );
    let allowed = budget.unexplained_fraction * variance;
    assert!(selection.order >= 1 && selection.order < selection.envelope_order);
    assert_eq!(response.order(), selection.order);
    assert_eq!(selection.degree_energies.len(), selection.order);
    assert!(selection.truncation_error.value + selection.truncation_error.band <= allowed);
    let before = hermite_truncation_energy(
        selection.total_energy,
        &selection.degree_energies[..selection.order - 1],
    );
    assert!(
        before.value + before.band > allowed,
        "order {} is not minimal: the tail {before:?} at the previous order already meets {allowed:e}",
        selection.order
    );
    let (values, bands) = coefficients_of(GaussianActivation::Relu, 0.0, scale, selection.order);
    for (index, shell) in selection.degree_energies.iter().enumerate() {
        let degree = index + 1;
        let square = values[degree] * values[degree];
        let bound = shell.band + 2.0 * values[degree].abs() * bands[degree] + gamma(4) * square;
        assert!(
            (shell.value - square).abs() <= bound,
            "scalar unit shell {degree} = {shell:?} must equal a_n^2 = {square:e}"
        );
    }
    let proposal = propose(&block, frame.view(), budget).expect("addressable proposal");
    let term_energies = proposal
        .terms
        .iter()
        .map(|term| term.energy)
        .collect::<Vec<_>>();
    let term_total = hermite_truncation_energy(BandedEnergy::ZERO, &term_energies);
    let captured = proposal.selection.captured_energy;
    assert!(
        (-term_total.value - captured.value).abs() <= term_total.band + captured.band,
        "proposal terms {term_total:?} must carry the captured energy {captured:?}"
    );
    assert_eq!(proposal.terms.len(), 1, "a scalar response has one support");
    let single = response
        .support_coefficients(&[0])
        .expect("an addressable support");
    assert_eq!(single.coefficients.len(), selection.order);
    assert!(
        (single.energy.value - proposal.terms[0].energy.value).abs()
            <= single.energy.band + proposal.terms[0].energy.band,
        "support {{0}} energy {:?} against the proposal term {:?}",
        single.energy,
        proposal.terms[0].energy
    );
    assert!(matches!(
        response.support_coefficients(&[1]),
        Err(HermiteError::InvalidSupport { .. })
    ));
    assert!(single.reserved_bytes() > 0);
    let (unit_mean, unit_mean_error) = smoothing_reference(ClosedForm::Relu, 0.0, scale);
    let mean = output_bias + unit_mean;
    let mean_bound = proposal.mean_band[0] + unit_mean_error + gamma(1) * mean.abs();
    assert!(
        (proposal.mean[0] - mean).abs() <= mean_bound,
        "the proposal mean {} must be c + T_s²σ(0) = {mean}",
        proposal.mean[0]
    );
    assert!(
        (proposal.mean[0] - unit_mean).abs() > mean_bound,
        "a mean without the output bias must fail the same bound"
    );
    let inconsistent = BandedEnergy {
        value: 1.5 * variance,
        band: 0.0,
    };
    let refused = select_order_with_total_energy(&block, frame.view(), inconsistent, budget);
    assert!(
        matches!(refused, Err(HermiteError::TailAboveEnvelope { .. })),
        "a total the shells cannot reach must be refused at the envelope order"
    );
    assert_eq!(multi_index_count(6, 3), Some(84));
    assert_eq!(multi_index_count(usize::MAX, 2), None);
}

/// The coordinate frame spanned by `kept` columns of `I_d`.
fn coordinate_frame(input_dim: usize, kept: &[usize]) -> Array2<f64> {
    let mut frame = Array2::zeros((input_dim, kept.len()));
    for (column, coordinate) in kept.iter().enumerate() {
        frame[[*coordinate, column]] = 1.0;
    }
    frame
}

/// `V` at a coordinate frame, with the derived band of [`zero_mean_variance_band`].
fn coordinate_variance(block: &KnownBlock, kept: &[usize]) -> BandedEnergy {
    let frame = coordinate_frame(block.input_dim(), kept);
    let value = block
        .explained_variance(frame.view())
        .expect("admissible coordinate frame")
        .value;
    let coordinates = block.readers().dot(&frame);
    BandedEnergy {
        value,
        band: zero_mean_variance_band(block, &coordinates),
    }
}

/// The superset energies bracket the total interaction exactly:
/// `0 ≤ I_ij − Σ_{|α|≤N, supp α ⊇ {i,j}} ‖c_α‖²_M ≤ V(I) − Σ_{1≤|α|≤N} ‖c_α‖²_M`.
/// `I_ij = V([d]) − V([d]∖i) − V([d]∖j) + V([d]∖{i,j}) = Σ_{T⊇{i,j}} ‖f_T‖²_M` (R7) is formed from
/// the retained-response operator. The superset energies are part of the orthogonal Hermite
/// energy, so their missing part is bounded by the whole tail. Summing the total effect of `i` in
/// place of the pair must break the lower side.
#[test]
fn superset_hermite_energy_brackets_the_total_interaction_2946() {
    let order = 40;
    let dimension = 3;
    for activation in [GaussianActivation::Relu, GaussianActivation::ExactGelu] {
        let block = zero_bias_block(activation);
        let frame = coordinate_frame(dimension, &[0, 1, 2]);
        let response =
            HermiteResponse::for_block(&block, frame.view(), order).expect("admissible frame");
        let pairs = [(0usize, 1usize), (0, 2), (1, 2)];
        let mut superset = vec![Vec::new(); pairs.len()];
        let mut total_effect = vec![Vec::new(); dimension];
        let mut captured = Vec::new();
        response.for_each_coefficient(|visit| {
            captured.push(visit.energy);
            for (index, (first, second)) in pairs.iter().enumerate() {
                if visit.alpha[*first] > 0 && visit.alpha[*second] > 0 {
                    superset[index].push(visit.energy);
                }
            }
            for (coordinate, effect) in total_effect.iter_mut().enumerate() {
                if visit.alpha[coordinate] > 0 {
                    effect.push(visit.energy);
                }
            }
        });
        let full = coordinate_variance(&block, &[0, 1, 2]);
        let tail = hermite_truncation_energy(full, &captured);
        let mut mutant_rejections = 0;
        for (index, (first, second)) in pairs.iter().enumerate() {
            let without_first = (0..dimension)
                .filter(|coordinate| coordinate != first)
                .collect::<Vec<_>>();
            let without_second = (0..dimension)
                .filter(|coordinate| coordinate != second)
                .collect::<Vec<_>>();
            let without_both = (0..dimension)
                .filter(|coordinate| coordinate != first && coordinate != second)
                .collect::<Vec<_>>();
            let interaction = signed_sum(
                &[full, coordinate_variance(&block, &without_both)],
                &[
                    coordinate_variance(&block, &without_first),
                    coordinate_variance(&block, &without_second),
                ],
            );
            assert!(
                interaction.resolved_positive(),
                "{activation:?} I_{first}{second} = {interaction:?} must be resolved"
            );
            let gap = hermite_truncation_energy(interaction, &superset[index]);
            assert!(
                gap.value >= -gap.band,
                "{activation:?} I_{first}{second} lies below its superset Hermite energy: gap {gap:?}"
            );
            assert!(
                gap.value <= tail.value + gap.band + tail.band,
                "{activation:?} I_{first}{second} minus its superset energy {gap:?} exceeds the tail {tail:?}"
            );
            let mutant = hermite_truncation_energy(interaction, &total_effect[*first]);
            if mutant.value < -mutant.band {
                mutant_rejections += 1;
            }
        }
        assert!(
            mutant_rejections > 0,
            "the total-effect mutant must break the {activation:?} lower bracket"
        );
    }
}
