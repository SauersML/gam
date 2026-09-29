//! Exact two-endpoint finite-change ("secant") operators for the softmax and
//! bilinear products the native edit compiler executes (#2951).
//!
//! A local Jacobian describes an infinitesimal change. A component ablation is a
//! finite change, and in saturated regimes the two disagree by orders of
//! magnitude. For each primitive below there is an operator `A(x, y)`, built from
//! the two endpoints only, with `f(y) − f(x) = A(x, y)(y − x)` EXACTLY (not to
//! first order). Each operator returns its values beside a per-entry
//! forward-error band derived from the arithmetic that produced them (Higham's
//! `γ_k`, through [`accumulation_growth`] and [`accumulation_band`]); the band bounds the distance to the exact operator of the STORED endpoints,
//! first order in the unit roundoff `u` and then inflated by [`inflated`]. Input
//! rounding belongs to the caller. `exp`, `exp_m1` and `ln` are taken to return
//! within one unit in the last place (`2u` relatively), the contract
//! `gam_math::categorical` states for the same functions.
//!
//! # Softmax
//!
//! `p = softmax u`, `q = softmax v`, `δ = v − u`. With the logarithmic means
//! `ℓ_i = (q_i − p_i)/(log q_i − log p_i)` (`ℓ_i = p_i` when `p_i = q_i`) and
//! `s = Σ ℓ_i`,
//!
//! ```text
//! q − p = ℓ ⊙ δ − ℓ (ℓᵀδ)/s = ℓ ⊙ (δ − c·1),      c = ℓᵀδ / s.
//! ```
//!
//! Derivation: `log q_i − log p_i = δ_i − κ` with `κ = lse(v) − lse(u)`, so by the
//! definition of `ℓ`, `q_i − p_i = ℓ_i (δ_i − κ)`. Summing over `i`, `0 = Σ(q_i −
//! p_i) = ℓᵀδ − κ s`, so `κ = c`. The operator `diag ℓ − ℓℓᵀ/s` is symmetric and
//! positive semidefinite (`xᵀAx = s·Var_{ℓ/s}(x) ≥ 0`), annihilates constant
//! shifts, and tends to the softmax Jacobian `diag p − ppᵀ` as `v → u` (`ℓ → p`,
//! `s → 1`). `ℓ` is formed from the log-probabilities of
//! [`log_softmax_with_error`], never from differences of probabilities: with
//! `h = max(log p_i, log q_i)` and `d = |log q_i − log p_i|`,
//!
//! ```text
//! ℓ_i = e^h · E(d),       E(d) = (1 − e^{−d})/d = −expm1(−d)/d,  E(0) = 1.
//! ```
//!
//! `E` loses no accuracy near `d = 0` (`expm1` is relatively accurate there and
//! the quotient divides by the same `d` it was evaluated at), and
//! `|d log E/dd| = 1/d − 1/(e^d − 1) ≤ ½`, so an absolute error `e` in `d` is at
//! most the relative error `e/2` in `E`.
//!
//! # Bilinear products
//!
//! With `L̄ = (L + L′)/2`, `ΔL = L′ − L` and the same for `R`,
//!
//! ```text
//! L′R′ − LR = L̄ ΔR + ΔL R̄
//! ```
//!
//! exactly: expanding the right side gives `½(L + L′)(R′ − R) + ½(L′ − L)(R + R′)
//! = ½(LR′ − LR + L′R′ − L′R) + ½(L′R + L′R′ − LR − LR′) = L′R′ − LR`. A weight
//! matrix times an input and attention weights times values are the same rule.

use gam_linalg::roundoff::{UNIT_ROUNDOFF, accumulation_band, accumulation_growth};
use gam_math::categorical::{CategoricalError, log_softmax_with_error};
use gam_math::roundoff::inflated;
use ndarray::{Array2, ArrayView2};
use std::fmt;

/// A refused secant evaluation.
#[derive(Clone, Debug, PartialEq)]
pub enum SecantError {
    /// An operand holds `NaN` or `±∞`.
    NonFinite {
        operand: &'static str,
        index: usize,
        value: f64,
    },
    /// An operand has no entries.
    Empty { operand: &'static str },
    /// An operand's length or shape disagrees with its partner.
    Shape {
        operand: &'static str,
        expected: (usize, usize),
        found: (usize, usize),
    },
    /// A difference or a sum of squares of finite operands overflowed.
    Overflow { quantity: &'static str },
    /// The categorical owner refused a logit vector.
    Categorical(CategoricalError),
}

impl fmt::Display for SecantError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite {
                operand,
                index,
                value,
            } => write!(
                formatter,
                "secant refused: {operand}[{index}] = {value} is not finite"
            ),
            Self::Empty { operand } => write!(formatter, "secant refused: {operand} is empty"),
            Self::Shape {
                operand,
                expected,
                found,
            } => write!(
                formatter,
                "secant refused: {operand} has shape {found:?}, its partner needs {expected:?}"
            ),
            Self::Overflow { quantity } => {
                write!(formatter, "secant refused: {quantity} overflowed")
            }
            Self::Categorical(err) => write!(formatter, "secant refused: {err}"),
        }
    }
}

impl std::error::Error for SecantError {}

impl From<CategoricalError> for SecantError {
    fn from(err: CategoricalError) -> Self {
        Self::Categorical(err)
    }
}

/// Values with a per-entry bound on `|computed − exact|`.
#[derive(Clone, Debug, PartialEq)]
pub struct BandedVector {
    pub values: Vec<f64>,
    pub bands: Vec<f64>,
}

/// A matrix with a per-entry bound on `|computed − exact|`.
#[derive(Clone, Debug, PartialEq)]
pub struct BandedMatrix {
    pub values: Array2<f64>,
    pub bands: Array2<f64>,
}

fn require_finite(
    operand: &'static str,
    values: impl IntoIterator<Item = f64>,
) -> Result<(), SecantError> {
    match values
        .into_iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        Some((index, value)) => Err(SecantError::NonFinite {
            operand,
            index,
            value,
        }),
        None => Ok(()),
    }
}

fn require_length(operand: &'static str, expected: usize, found: usize) -> Result<(), SecantError> {
    if expected == found {
        Ok(())
    } else {
        Err(SecantError::Shape {
            operand,
            expected: (expected, 1),
            found: (found, 1),
        })
    }
}

/// `fl(end − start)` per entry with its radius `u·|Δ_i|`, refusing an overflow.
fn rounded_differences(
    start: &[f64],
    end: &[f64],
    quantity: &'static str,
) -> Result<(Vec<f64>, Vec<f64>), SecantError> {
    let differences: Vec<f64> = start.iter().zip(end).map(|(a, b)| b - a).collect();
    if differences.iter().any(|value| !value.is_finite()) {
        return Err(SecantError::Overflow { quantity });
    }
    let radii = differences
        .iter()
        .map(|value| UNIT_ROUNDOFF * value.abs())
        .collect();
    Ok((differences, radii))
}

/// `E(d) = (1 − e^{−d})/d` for `d ≥ 0`, `E(0) = 1`: the secant slope of `1 − e^{−x}` over `[0, d]`. Relative error
/// at most `3u` (`expm1` two units, the quotient one).
fn exponential_secant(d: f64) -> f64 {
    if d == 0.0 { 1.0 } else { -(-d).exp_m1() / d }
}

/// The exact softmax secant operator between two logit vectors, held as its logarithmic means.
#[derive(Clone, Debug, PartialEq)]
struct SoftmaxSecant {
    means: Vec<f64>,
    mean_bands: Vec<f64>,
    total: f64,
    total_band: f64,
}

impl SoftmaxSecant {
    /// The operator `A = diag ℓ − ℓℓᵀ/s` between `softmax(start)` and `softmax(end)`.
    ///
    /// Band of `ℓ_i`: the log-probabilities carry radii `r_p`, `r_q` from [`log_softmax_with_error`]. `e^h` takes the
    /// relative error `expm1(r_h) + 2u`, `d = |log q − log p|` the absolute error `r_p + r_q + u·d`, which moves `E` by
    /// at most half of it relatively, `E` itself `3u`, and the product `u`. `s` is a sum of positive terms: the
    /// bands of its terms plus `γ_{n−1}·s`.
    fn between(start: &[f64], end: &[f64]) -> Result<Self, SecantError> {
        if start.is_empty() {
            return Err(SecantError::Empty {
                operand: "start logits",
            });
        }
        require_length("end logits", start.len(), end.len())?;
        require_finite("start logits", start.iter().copied())?;
        require_finite("end logits", end.iter().copied())?;
        let (log_p, radius_p) = log_softmax_with_error(start)?;
        let (log_q, radius_q) = log_softmax_with_error(end)?;
        let u = UNIT_ROUNDOFF;
        let mut means = Vec::with_capacity(start.len());
        let mut mean_bands = Vec::with_capacity(start.len());
        for i in 0..start.len() {
            let (head, head_radius) = if log_p[i] >= log_q[i] {
                (log_p[i], radius_p[i])
            } else {
                (log_q[i], radius_q[i])
            };
            let gap = (log_q[i] - log_p[i]).abs();
            let mean = head.exp() * exponential_secant(gap);
            let gap_error = radius_p[i] + radius_q[i] + u * gap;
            let relative = head_radius.exp_m1() + 2.0 * u + 0.5 * gap_error + 3.0 * u + u;
            means.push(mean);
            mean_bands.push(inflated(mean * relative, 4));
        }
        let total: f64 = means.iter().sum();
        let total_band = inflated(
            mean_bands.iter().sum::<f64>() + accumulation_growth(means.len() - 1) * total,
            means.len(),
        );
        Ok(Self {
            means,
            mean_bands,
            total,
            total_band,
        })
    }

    /// `A δ` for a `δ` known within `radii`. `ℓᵀδ` errs by `accumulation_band(n, Σ|ℓ_iδ_i|)` plus the propagated
    /// `Σ(band_i|δ_i| + ℓ_i r_i)`; `c` adds `|c|·band_s/s` and one quotient rounding; `ℓ_i(δ_i − c)` adds its factors'
    /// bands and two roundings.
    fn apply_with_radii(&self, direction: &[f64], radii: &[f64]) -> BandedVector {
        let u = UNIT_ROUNDOFF;
        let n = self.means.len();
        let mut dot = 0.0;
        let mut absolute = 0.0;
        let mut propagated = 0.0;
        for i in 0..n {
            dot += self.means[i] * direction[i];
            absolute += (self.means[i] * direction[i]).abs();
            propagated += self.mean_bands[i] * direction[i].abs() + self.means[i] * radii[i];
        }
        let dot_band = accumulation_band(n, absolute) + propagated;
        let centre = dot / self.total;
        let centre_band =
            (dot_band + centre.abs() * self.total_band) / self.total + u * centre.abs();
        let mut values = Vec::with_capacity(n);
        let mut bands = Vec::with_capacity(n);
        for i in 0..n {
            let offset = direction[i] - centre;
            let value = self.means[i] * offset;
            let offset_band = radii[i] + centre_band + u * offset.abs();
            values.push(value);
            bands.push(inflated(
                self.mean_bands[i] * offset.abs() + self.means[i] * offset_band + u * value.abs(),
                4,
            ));
        }
        BandedVector { values, bands }
    }
}

/// `softmax(end) − softmax(start)` through the exact secant operator, with `δ = fl(end − start)` and its rounding
/// carried as a radius.
pub fn softmax_change(start: &[f64], end: &[f64]) -> Result<BandedVector, SecantError> {
    let operator = SoftmaxSecant::between(start, end)?;
    let (direction, radii) = rounded_differences(start, end, "logit difference")?;
    Ok(operator.apply_with_radii(&direction, &radii))
}

/// `L′R′ − LR = L̄ ΔR + ΔL R̄` for `L` (`p × q`) and `R` (`q × s`): a weight matrix times inputs, or attention
/// weights times values.
///
/// Band: each term of an entry's two length-`q` inner products has computed operands one rounding off (the
/// averages and differences), the inner products add `γ_q` and the final sum one more, so the entry errs by at most
/// `γ_{q+3}·S` with `S = |L̄||ΔR| + |ΔL||R̄|` (Higham, ASNA §3.1, for any summation order); `S` is itself a
/// computed sum of positive terms and is inflated by its own `γ_{q+1}`.
pub fn bilinear_change(
    left_start: ArrayView2<'_, f64>,
    left_end: ArrayView2<'_, f64>,
    right_start: ArrayView2<'_, f64>,
    right_end: ArrayView2<'_, f64>,
) -> Result<BandedMatrix, SecantError> {
    let expect = |operand, expected: (usize, usize), found: (usize, usize)| {
        if expected == found {
            Ok(())
        } else {
            Err(SecantError::Shape {
                operand,
                expected,
                found,
            })
        }
    };
    expect("left end", left_start.dim(), left_end.dim())?;
    expect("right end", right_start.dim(), right_end.dim())?;
    expect(
        "right start",
        (left_start.ncols(), right_start.ncols()),
        (right_start.nrows(), right_start.ncols()),
    )?;
    require_finite("left start", left_start.iter().copied())?;
    require_finite("left end", left_end.iter().copied())?;
    require_finite("right start", right_start.iter().copied())?;
    require_finite("right end", right_end.iter().copied())?;
    let left_mean = (&left_start * 0.5) + (&left_end * 0.5);
    let left_step = &left_end - &left_start;
    let right_mean = (&right_start * 0.5) + (&right_end * 0.5);
    let right_step = &right_end - &right_start;
    if left_step
        .iter()
        .chain(right_step.iter())
        .any(|value| !value.is_finite())
    {
        return Err(SecantError::Overflow {
            quantity: "bilinear endpoint difference",
        });
    }
    let values = left_mean.dot(&right_step) + left_step.dot(&right_mean);
    let absolute = left_mean.mapv(f64::abs).dot(&right_step.mapv(f64::abs))
        + left_step.mapv(f64::abs).dot(&right_mean.mapv(f64::abs));
    let inner = left_start.ncols();
    let growth = accumulation_growth(inner + 3);
    let bands = absolute.mapv(|sum| inflated(growth * sum, inner + 1));
    Ok(BandedMatrix { values, bands })
}

#[cfg(test)]
mod tests {
    use super::*;

    const U: f64 = UNIT_ROUNDOFF;

    fn logits(n: usize, phase: f64, scale: f64) -> Vec<f64> {
        (0..n)
            .map(|i| {
                scale * ((i as f64 + 1.0) * (1.7 + phase)).sin()
                    + 0.4 * ((i * i) as f64 * 0.31 + phase).cos()
            })
            .collect()
    }

    /// `softmax(z)` from the owner's log-softmax, each entry within `p·(expm1 r + 2u)` of the exact value.
    fn probabilities(logits: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let (log_p, radii) = log_softmax_with_error(logits).expect("finite logits");
        let values: Vec<f64> = log_p.iter().map(|value| value.exp()).collect();
        let bands = values
            .iter()
            .zip(&radii)
            .map(|(p, r)| inflated(p * (r.exp_m1() + 2.0 * U), 2))
            .collect();
        (values, bands)
    }

    /// `q − p` evaluated directly, with its band.
    fn direct_softmax_change(start: &[f64], end: &[f64]) -> BandedVector {
        let (p, p_bands) = probabilities(start);
        let (q, q_bands) = probabilities(end);
        let values: Vec<f64> = q.iter().zip(&p).map(|(q, p)| q - p).collect();
        let bands = (0..p.len())
            .map(|i| inflated(p_bands[i] + q_bands[i] + U * values[i].abs(), 1))
            .collect();
        BandedVector { values, bands }
    }

    /// Whether `candidate` lies within `claimed + reference.bands` of `reference` at every entry.
    fn agrees(candidate: &[f64], claimed: &[f64], reference: &BandedVector) -> bool {
        (0..candidate.len())
            .all(|i| (candidate[i] - reference.values[i]).abs() <= claimed[i] + reference.bands[i])
    }

    #[test]
    fn softmax_secant_reproduces_the_finite_change_and_the_local_forms_do_not() {
        for (phase, step) in [(0.1, 0.9), (0.7, 2.5), (1.3, 6.0)] {
            let start = logits(7, phase, 2.0);
            let end: Vec<f64> = start
                .iter()
                .enumerate()
                .map(|(i, z)| z + step * ((i as f64) * 0.83 + phase).cos())
                .collect();
            let secant = softmax_change(&start, &end).expect("finite logits");
            let direct = direct_softmax_change(&start, &end);
            assert!(
                agrees(&secant.values, &secant.bands, &direct),
                "secant {:?} ± {:?} misses direct {:?} ± {:?}",
                secant.values,
                secant.bands,
                direct.values,
                direct.bands
            );
            // Positive controls, held to the secant's own band: the local Jacobian at the start, and the operator
            // without its rank-one centring.
            let (p, _) = probabilities(&start);
            let delta: Vec<f64> = start.iter().zip(&end).map(|(a, b)| b - a).collect();
            let mean: f64 = p.iter().zip(&delta).map(|(p, d)| p * d).sum();
            let jacobian: Vec<f64> = p.iter().zip(&delta).map(|(p, d)| p * (d - mean)).collect();
            assert!(
                !agrees(&jacobian, &secant.bands, &direct),
                "the local Jacobian passed the secant band"
            );
            let operator = SoftmaxSecant::between(&start, &end).expect("finite logits");
            let diagonal_only: Vec<f64> = operator
                .means
                .iter()
                .zip(&delta)
                .map(|(l, d)| l * d)
                .collect();
            assert!(
                !agrees(&diagonal_only, &secant.bands, &direct),
                "dropping the rank-one term passed"
            );
            // Each logarithmic mean lies between its two probabilities.
            let (q, _) = probabilities(&end);
            for i in 0..p.len() {
                let l = operator.means[i];
                let band = operator.mean_bands[i];
                assert!(l + band >= p[i].min(q[i]) * (1.0 - 4.0 * U));
                assert!(l - band <= p[i].max(q[i]) * (1.0 + 4.0 * U));
            }
        }
    }

    #[test]
    fn softmax_secant_annihilates_constant_shifts() {
        let start = logits(6, 0.4, 1.5);
        let end = logits(6, 1.1, 3.0);
        let operator = SoftmaxSecant::between(&start, &end).expect("finite logits");
        let constant = operator.apply_with_radii(&[2.75; 6], &[0.0; 6]);
        for (value, band) in constant.values.iter().zip(&constant.bands) {
            assert!(
                value.abs() <= *band,
                "a constant shift moved the output by {value:e} > {band:e}"
            );
        }
        // Softmax is shift invariant, and so is the secant change.
        let shifted_end: Vec<f64> = end.iter().map(|z| z + 13.0).collect();
        let direct = direct_softmax_change(&start, &end);
        let shifted = softmax_change(&start, &shifted_end).expect("finite logits");
        assert!(agrees(&shifted.values, &shifted.bands, &direct));
    }

    #[test]
    fn softmax_secant_is_symmetric_and_positive_semidefinite() {
        let start = logits(8, 0.2, 2.5);
        let end = logits(8, 0.9, 1.0);
        let operator = SoftmaxSecant::between(&start, &end).expect("finite logits");
        let x = logits(8, 2.1, 1.0);
        let y = logits(8, 3.4, 1.7);
        let ax = operator.apply_with_radii(&x, &[0.0; 8]);
        let ay = operator.apply_with_radii(&y, &[0.0; 8]);
        let pairing = |left: &[f64], right: &BandedVector| {
            let value: f64 = left.iter().zip(&right.values).map(|(a, b)| a * b).sum();
            let absolute: f64 = left
                .iter()
                .zip(&right.values)
                .map(|(a, b)| (a * b).abs())
                .sum();
            let propagated: f64 = left
                .iter()
                .zip(&right.bands)
                .map(|(a, b)| a.abs() * b)
                .sum();
            (
                value,
                inflated(accumulation_band(left.len(), absolute) + propagated, 1),
            )
        };
        let (xay, xay_band) = pairing(&x, &ay);
        let (yax, yax_band) = pairing(&y, &ax);
        assert!(
            (xay - yax).abs() <= xay_band + yax_band,
            "xᵀAy = {xay:e}, yᵀAx = {yax:e}"
        );
        for vector in [&x, &y] {
            let (quadratic, band) = pairing(vector, &operator.apply_with_radii(vector, &[0.0; 8]));
            assert!(quadratic >= -band, "xᵀAx = {quadratic:e} below −{band:e}");
            assert!(
                quadratic > band,
                "a non-constant direction must have positive curvature"
            );
        }
    }

    #[test]
    fn softmax_secant_tends_to_the_softmax_jacobian() {
        let start = logits(5, 0.6, 1.8);
        let direction = logits(5, 1.9, 1.0);
        let (p, p_bands) = probabilities(&start);
        let sup = direction.iter().fold(0.0_f64, |m, w| m.max(w.abs()));
        let mean: f64 = p.iter().zip(&direction).map(|(p, w)| p * w).sum();
        let jacobian: Vec<f64> = p
            .iter()
            .zip(&direction)
            .map(|(p, w)| p * (w - mean))
            .collect();
        let band_sum: f64 = p_bands.iter().sum();
        let mut previous_gap = f64::INFINITY;
        for step in [1e-1, 1e-3, 1e-5] {
            let end: Vec<f64> = start
                .iter()
                .zip(&direction)
                .map(|(z, w)| z + step * w)
                .collect();
            let action = SoftmaxSecant::between(&start, &end)
                .expect("finite")
                .apply_with_radii(&direction, &[0.0; 5]);
            let change = direct_softmax_change(&start, &end);
            // With ℓ = p + e, |e_i| ≤ |q_i − p_i|, and E = ‖q − p‖₁, the two operators differ on w by at most
            // ‖w‖∞(2E + 2E/(1 − E)) at every entry.
            let total: f64 = change
                .values
                .iter()
                .zip(&change.bands)
                .map(|(v, b)| v.abs() + b)
                .sum();
            let allowance = sup * (2.0 * total + 2.0 * total / (1.0 - total));
            let mut gap = 0.0_f64;
            for i in 0..p.len() {
                let jacobian_band = inflated(
                    2.0 * sup * (p_bands[i] + p[i] * band_sum + accumulation_growth(8) * p[i]),
                    2,
                );
                let difference = (action.values[i] - jacobian[i]).abs();
                assert!(difference <= allowance + action.bands[i] + jacobian_band);
                gap = gap.max(difference);
            }
            assert!(
                gap < previous_gap,
                "the secant did not approach the Jacobian"
            );
            previous_gap = gap;
        }
    }

    #[test]
    fn saturated_softmax_change_is_captured_by_the_secant_and_missed_by_the_jacobian() {
        let start = [12.0, -12.0];
        let end = [-12.0, 12.0];
        let secant = softmax_change(&start, &end).expect("finite");
        // q − p = (−tanh 12, tanh 12) exactly.
        let exact = 12.0_f64.tanh();
        for (value, (band, sign)) in secant
            .values
            .iter()
            .zip(secant.bands.iter().zip([-1.0, 1.0]))
        {
            assert!(
                (value - sign * exact).abs() <= band + U * exact,
                "{value} vs {}",
                sign * exact
            );
        }
        let norm = secant.values.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((norm - std::f64::consts::SQRT_2 * exact).abs() <= 4.0 * U * norm);
        // The local Jacobian at the start: 48·√2·p₁p₂ = 12√2/cosh²12 ≈ 2.6e-9.
        let (p, _) = probabilities(&start);
        let delta = [-24.0, 24.0];
        // (diag p − ppᵀ)δ = p₁p₂(δ₁ − δ₂)(1, −1) for two categories, formed without the cancelling δ − pᵀδ.
        let coupling = p[0] * p[1] * (delta[0] - delta[1]);
        let jacobian = [coupling, -coupling];
        let jacobian_norm = (jacobian[0] * jacobian[0] + jacobian[1] * jacobian[1]).sqrt();
        let predicted = 12.0 * std::f64::consts::SQRT_2 / 12.0_f64.cosh().powi(2);
        assert!(
            (jacobian_norm - predicted).abs() <= 1e-6 * predicted,
            "{jacobian_norm:e} vs {predicted:e}"
        );
        assert!(jacobian_norm < 3e-9 && norm > 1.414);
        let direct = direct_softmax_change(&start, &end);
        assert!(!agrees(&jacobian, &secant.bands, &direct));
    }

    #[test]
    fn bilinear_secant_is_exact_and_the_first_order_form_is_not() {
        let fixture = |rows: usize, cols: usize, phase: f64| {
            Array2::from_shape_fn((rows, cols), |(i, j)| {
                ((i as f64 + 1.0) * (0.37 + phase) + (j as f64 + 1.0) * (0.61 - 0.5 * phase)).sin()
            })
        };
        let direct_band = |left: &Array2<f64>,
                           right: &Array2<f64>,
                           left_new: &Array2<f64>,
                           right_new: &Array2<f64>| {
            let value = left_new.dot(right_new) - left.dot(right);
            let absolute = left_new.mapv(f64::abs).dot(&right_new.mapv(f64::abs))
                + left.mapv(f64::abs).dot(&right.mapv(f64::abs));
            let inner = left.ncols();
            let band = Array2::from_shape_fn(value.dim(), |(i, k)| {
                inflated(
                    accumulation_growth(inner + 1) * absolute[[i, k]] + U * value[[i, k]].abs(),
                    inner,
                )
            });
            (value, band)
        };
        let (l, l_new) = (fixture(4, 6, 0.1), fixture(4, 6, 0.9));
        let (r, r_new) = (fixture(6, 3, 0.4), fixture(6, 3, 1.3));
        let change =
            bilinear_change(l.view(), l_new.view(), r.view(), r_new.view()).expect("finite");
        let (direct, band) = direct_band(&l, &r, &l_new, &r_new);
        let first_order = l.dot(&(&r_new - &r)) + (&l_new - &l).dot(&r);
        let mut first_order_fails = false;
        for ((i, k), value) in change.values.indexed_iter() {
            let allowance = change.bands[[i, k]] + band[[i, k]];
            assert!((value - direct[[i, k]]).abs() <= allowance);
            first_order_fails |= (first_order[[i, k]] - direct[[i, k]]).abs() > allowance;
        }
        assert!(
            first_order_fails,
            "the first-order product rule passed the secant band"
        );
        // Attention weights (rows of a softmax) times values.
        let weights = Array2::from_shape_fn((3, 4), |(i, j)| {
            probabilities(&logits(4, i as f64, 1.0)).0[j]
        });
        let weights_new = Array2::from_shape_fn((3, 4), |(i, j)| {
            probabilities(&logits(4, i as f64 + 0.5, 2.0)).0[j]
        });
        let (values, values_new) = (fixture(4, 5, 0.2), fixture(4, 5, 0.7));
        let attention = bilinear_change(
            weights.view(),
            weights_new.view(),
            values.view(),
            values_new.view(),
        )
        .expect("finite");
        let (attention_direct, attention_band) =
            direct_band(&weights, &values, &weights_new, &values_new);
        for ((i, k), value) in attention.values.indexed_iter() {
            assert!(
                (value - attention_direct[[i, k]]).abs()
                    <= attention.bands[[i, k]] + attention_band[[i, k]]
            );
        }
    }

    #[test]
    fn non_finite_and_malformed_inputs_are_refused() {
        let nan = f64::NAN;
        assert!(matches!(
            SoftmaxSecant::between(&[0.0, nan], &[0.0, 1.0]),
            Err(SecantError::NonFinite { index: 1, .. })
        ));
        assert!(matches!(
            softmax_change(&[0.0], &[0.0, 1.0]),
            Err(SecantError::Shape { .. })
        ));
        assert!(matches!(
            softmax_change(&[], &[]),
            Err(SecantError::Empty { .. })
        ));
        let zeros = Array2::<f64>::zeros((2, 2));
        let mut bad = zeros.clone();
        bad[[1, 0]] = nan;
        assert!(matches!(
            bilinear_change(zeros.view(), bad.view(), zeros.view(), zeros.view()),
            Err(SecantError::NonFinite { .. })
        ));
        let column = Array2::<f64>::zeros((3, 1));
        assert!(matches!(
            bilinear_change(zeros.view(), zeros.view(), column.view(), column.view()),
            Err(SecantError::Shape { .. })
        ));
    }
}
