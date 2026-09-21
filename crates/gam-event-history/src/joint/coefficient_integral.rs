//! Importance correction of the joint Laplace coefficient law (#2961, SPEC 3).
//!
//! The Laplace law N(θ̂, S⁻¹) of `strength_fit.rs` is an approximation. Fixed
//! independent draws from `q = ½ N(θ̂, S⁻¹) + ½ t₃(θ̂, S⁻¹)` carry their complete
//! normalized proposal densities. The Student component has the same covariance
//! and polynomial tails, so a target whose tails are lighter than polynomial
//! keeps finite weight variance. The self-normalized weights
//! `w_j ∝ exp(ℓ(θ_j) + log π(θ_j | ρ̂) − log q(θ_j))` estimate the coefficient
//! posterior mean, and forecasts average final probabilities with the same
//! weights.
//!
//! Which mean is published is decided by estimated errors in the Laplace metric
//! `‖v‖²_S = vᵀ S v`, with no tolerance constant:
//! - the shift `Δ̂ = μ̂ − θ̂` has `E‖Δ̂‖²_S = ‖Δ‖²_S + tr(S C)`, with C the
//!   delta-method covariance of the self-normalized mean μ̂;
//! - the corrected mean's error is `tr(S C) + b²`, where b bounds the effect of
//!   the inner log-likelihood error d on the weights (weight ratios lie within
//!   `e^(±2d)`);
//! - the Laplace mean's estimated squared bias is `‖Δ̂‖²_S − tr(S C)`.
//!
//! The corrected law is published when that bias exceeds the corrected error.
//! Otherwise the Laplace law stands with its estimated error bound. The bank is
//! refused as unresolved when the corrected error reaches the weighted posterior
//! dispersion `Σ_j w_j ‖θ_j − μ̂‖²_S`: the estimate then does not locate the mean
//! within the posterior it summarizes. These are estimated errors, not
//! deterministic or simultaneous bounds. Over n draws the self-normalized mean
//! also carries a bias, `−E_q[w̃² (θ − μ)] / n` to first order with
//! `w̃ = w / E_q w` (delta method on the ratio of means). Its square is O(1/n²)
//! against `tr(S C) = O(1/n)`, so the error estimates omit it at that order. It
//! is a first-order asymptotic statement, not a bound, and bias/SE = O(n^(-1/2)).
use super::law::{invalid, numerical};
use crate::EventHistoryError;
use crate::chain::log_sum_exp;
use gam_linalg::faer_ndarray::{FaerLlt, cholesky_factor_logdet};
use rand::{Rng, RngExt};
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};

/// One independent draw with its complete proposal log density.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CoefficientDraw {
    pub coefficients: Vec<f64>,
    pub log_proposal_density: f64,
    /// `log N(θ_j; θ̂, S⁻¹)`. With the proposal density it gives the Laplace-law
    /// weights of the same draws, so a forecast under the Laplace law averages
    /// over that law without a second bank.
    pub log_laplace_density: f64,
}

/// A fixed correction bank. It owns its draws and unnormalized log weights,
/// which are kept even when a weight underflows, because a new history can make
/// an underflowing training weight relevant again.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CoefficientBank {
    pub draws: Vec<CoefficientDraw>,
    /// `ℓ(θ_j) + log π(θ_j | ρ̂) − log q(θ_j)`.
    pub log_weights: Vec<f64>,
    /// The largest per-draw estimated error of ℓ. Inner integrals are reused
    /// across draws, so this is never divided by the square root of the count.
    pub inner_log_error_estimate: f64,
}

/// The coefficient law a fit serves.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum CoefficientLaw {
    /// The bank did not resolve a mean shift above its own error. The Laplace
    /// mean stands, with this estimated bound on its error in `‖·‖_S` units.
    Laplace { mean_error_bound: f64 },
    /// The importance-corrected law. Forecasts average with the bank's weights.
    Corrected {
        mean: Vec<f64>,
        /// `√(tr(S C) + b²)`.
        mean_error: f64,
    },
}

/// The corrected-versus-Laplace decision and the quantities it compared.
pub(super) struct CorrectionAssessment {
    pub law: CoefficientLaw,
    /// `‖Δ̂‖²_S`.
    pub shift_squared: f64,
    /// `tr(S C) + b²`.
    pub correction_error_squared: f64,
    /// `Σ_j w_j ‖θ_j − μ̂‖²_S`.
    pub dispersion: f64,
    pub effective_samples: f64,
}

/// ℓ(θ) with every latent path integrated, and an estimate of its absolute
/// numerical error.
pub(super) trait CoefficientLikelihood {
    fn log_likelihood(&mut self, theta: &[f64]) -> Result<(f64, f64), EventHistoryError>;
}

/// `lnΓ((p+3)/2) − lnΓ(3/2)` by its exact recurrence.
fn student_log_gamma_ratio(dimension: usize) -> f64 {
    if dimension % 2 == 0 {
        (0..dimension / 2).map(|j| (1.5 + j as f64).ln()).sum()
    } else {
        let m = (dimension - 1) / 2;
        (1..=m + 1).map(|j| (j as f64).ln()).sum::<f64>() - 0.5 * std::f64::consts::PI.ln()
            + std::f64::consts::LN_2
    }
}

/// `Lᵀ v` for `S = L Lᵀ`, whose norm is `‖v‖_S`.
fn whiten(factor: &FaerLlt<f64>, v: &[f64]) -> Vec<f64> {
    let lower = factor.lower();
    (0..v.len())
        .map(|j| (j..v.len()).map(|i| lower[(i, j)] * v[i]).sum())
        .collect()
}

fn metric_norm(factor: &FaerLlt<f64>, v: &[f64]) -> f64 {
    whiten(factor, v).iter().fold(0.0_f64, |a, &b| a.hypot(b))
}

/// `log q` of the equal Gaussian/t₃ mixture with covariance Σ, at a point whose
/// distance from the centre in the metric Σ⁻¹ is `distance`, and its Gaussian
/// component. `half_log_determinant` is `½ log det Σ⁻¹`. The Student component
/// has scale Σ/3, so it shares the covariance Σ and its quadratic is the same
/// squared distance.
pub(super) fn mixture_log_density(
    dimension: usize,
    half_log_determinant: f64,
    distance: f64,
) -> Result<(f64, f64), EventHistoryError> {
    let gaussian = half_log_determinant
        - 0.5 * dimension as f64 * (2.0 * std::f64::consts::PI).ln()
        - 0.5 * distance * distance;
    let log_quadratic_term = if distance == 0.0 {
        0.0
    } else {
        let log_squared = 2.0 * distance.ln();
        if log_squared > 0.0 {
            log_squared + (-log_squared).exp().ln_1p()
        } else {
            log_squared.exp().ln_1p()
        }
    };
    let student = student_log_gamma_ratio(dimension)
        - 0.5 * dimension as f64 * std::f64::consts::PI.ln()
        + half_log_determinant
        - 0.5 * (dimension + 3) as f64 * log_quadratic_term;
    let density = log_sum_exp(&[gaussian, student]) - std::f64::consts::LN_2;
    if !density.is_finite() || !gaussian.is_finite() {
        return Err(numerical("mixture proposal density is not representable"));
    }
    Ok((density, gaussian))
}

/// The radial scale of one mixture draw applied to an N(0, Σ) displacement: 1
/// for the Gaussian component, or `1/√χ²₃` for the t₃ component with scale Σ/3,
/// each with probability ½.
pub(super) fn mixture_scale<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    if rng.random::<bool>() {
        1.0
    } else {
        let chi_squared: f64 = std::iter::repeat_with(|| {
            let z: f64 = StandardNormal.sample(&mut *rng);
            z * z
        })
        .take(3)
        .sum();
        chi_squared.sqrt().recip()
    }
}

/// `log q(θ)` for the mixture with mean θ̂ and covariance S⁻¹, and the Gaussian
/// component `log N(θ; θ̂, S⁻¹)`.
fn proposal_log_density(
    mean: &[f64],
    factor: &FaerLlt<f64>,
    theta: &[f64],
) -> Result<(f64, f64), EventHistoryError> {
    let delta: Vec<f64> = theta.iter().zip(mean).map(|(a, b)| a - b).collect();
    mixture_log_density(
        mean.len(),
        0.5 * cholesky_factor_logdet(factor.lower()),
        metric_norm(factor, &delta),
    )
}

/// Independent draws from the mixture. A draw whose density is unrepresentable
/// is an error, never discarded and replaced, which would truncate the law.
pub(super) fn draw_coefficients<R: Rng + ?Sized>(
    mean: &[f64],
    factor: &FaerLlt<f64>,
    count: usize,
    rng: &mut R,
) -> Result<Vec<CoefficientDraw>, EventHistoryError> {
    let p = mean.len();
    if p == 0 || factor.nrows() != p || count < 2 || mean.iter().any(|v| !v.is_finite()) {
        return Err(invalid(
            "coefficient draws need a finite mean, a matching precision factor and at least two draws",
        ));
    }
    let lower = factor.lower();
    let mut draws = Vec::with_capacity(count);
    while draws.len() < count {
        let z: Vec<f64> = std::iter::repeat_with(|| StandardNormal.sample(&mut *rng))
            .take(p)
            .collect();
        // x = L⁻ᵀ z has covariance S⁻¹.
        let mut x = z;
        for i in (0..p).rev() {
            let mut value = x[i];
            for j in i + 1..p {
                value -= lower[(j, i)] * x[j];
            }
            x[i] = value / lower[(i, i)];
        }
        let scale = mixture_scale(rng);
        let coefficients: Vec<f64> = x.iter().zip(mean).map(|(d, m)| m + scale * d).collect();
        if coefficients.iter().any(|v| !v.is_finite()) {
            return Err(numerical("coefficient draw is not representable"));
        }
        let (log_proposal_density, log_laplace_density) =
            proposal_log_density(mean, factor, &coefficients)?;
        draws.push(CoefficientDraw {
            coefficients,
            log_proposal_density,
            log_laplace_density,
        });
    }
    Ok(draws)
}

/// Evaluate the target at every draw once and freeze the unnormalized weights.
pub(super) fn coefficient_bank<L: CoefficientLikelihood>(
    likelihood: &mut L,
    prior_log_density: &mut dyn FnMut(&[f64]) -> Result<f64, EventHistoryError>,
    draws: Vec<CoefficientDraw>,
) -> Result<CoefficientBank, EventHistoryError> {
    if draws.len() < 2 {
        return Err(invalid("a coefficient bank needs at least two draws"));
    }
    let mut log_weights = Vec::with_capacity(draws.len());
    let mut inner_log_error_estimate = 0.0_f64;
    for draw in &draws {
        let (value, error) = likelihood.log_likelihood(&draw.coefficients)?;
        if !error.is_finite() || error < 0.0 {
            return Err(invalid(
                "a coefficient likelihood error estimate must be finite and nonnegative",
            ));
        }
        let log_weight = value + prior_log_density(&draw.coefficients)? - draw.log_proposal_density;
        if !log_weight.is_finite() {
            return Err(numerical("non-finite coefficient importance weight"));
        }
        log_weights.push(log_weight);
        inner_log_error_estimate = inner_log_error_estimate.max(error);
    }
    Ok(CoefficientBank {
        draws,
        log_weights,
        inner_log_error_estimate,
    })
}

/// Decide between the corrected and the Laplace coefficient law.
pub(super) fn assess_correction(
    bank: &CoefficientBank,
    mean: &[f64],
    factor: &FaerLlt<f64>,
) -> Result<CorrectionAssessment, EventHistoryError> {
    let n = bank.draws.len();
    let p = mean.len();
    if n < 2
        || bank.log_weights.len() != n
        || factor.nrows() != p
        || bank.draws.iter().any(|d| d.coefficients.len() != p)
    {
        return Err(invalid(
            "coefficient bank dimensions do not match the Laplace law",
        ));
    }
    let log_total = log_sum_exp(&bank.log_weights);
    let weights: Vec<f64> = bank
        .log_weights
        .iter()
        .map(|w| (w - log_total).exp())
        .collect();
    let mut shift = vec![0.0; p];
    let mut correction = vec![0.0; p];
    for (draw, &w) in bank.draws.iter().zip(&weights) {
        for k in 0..p {
            let contribution = w * (draw.coefficients[k] - mean[k]) - correction[k];
            let next = shift[k] + contribution;
            correction[k] = (next - shift[k]) - contribution;
            shift[k] = next;
        }
    }
    let corrected: Vec<f64> = mean.iter().zip(&shift).map(|(m, s)| m + s).collect();
    let mut weighted_error = 0.0_f64;
    let mut inner_spread = 0.0;
    let mut dispersion = 0.0;
    let mut squared_weights = 0.0;
    for (draw, &w) in bank.draws.iter().zip(&weights) {
        let centered: Vec<f64> = draw
            .coefficients
            .iter()
            .zip(&corrected)
            .map(|(a, b)| a - b)
            .collect();
        let distance = metric_norm(factor, &centered);
        weighted_error = weighted_error.hypot(w * distance);
        inner_spread += w * distance;
        dispersion += w * distance * distance;
        squared_weights += w * w;
    }
    let trace = weighted_error * weighted_error * n as f64 / (n - 1) as f64;
    let inner = (2.0 * bank.inner_log_error_estimate).exp_m1() * inner_spread;
    let correction_error_squared = trace + inner * inner;
    let shift_norm = metric_norm(factor, &shift);
    let shift_squared = shift_norm * shift_norm;
    let effective_samples = squared_weights.recip();
    if [
        trace,
        inner,
        dispersion,
        shift_squared,
        effective_samples,
    ]
    .iter()
    .any(|v| !v.is_finite())
    {
        return Err(numerical(
            "coefficient correction moments are not representable",
        ));
    }
    if correction_error_squared >= dispersion {
        return Err(EventHistoryError::IntegrationResolution {
            reason: format!(
                "coefficient importance correction unresolved: estimated mean error {correction_error_squared:.3e} reaches the weighted posterior dispersion {dispersion:.3e} in the Laplace metric ({effective_samples:.1} effective of {n} draws); refine the bank or its inner integrals"
            ),
        });
    }
    let law = if shift_squared - trace > correction_error_squared {
        CoefficientLaw::Corrected {
            mean: corrected,
            mean_error: correction_error_squared.sqrt(),
        }
    } else {
        CoefficientLaw::Laplace {
            mean_error_bound: shift_norm + correction_error_squared.sqrt(),
        }
    };
    Ok(CorrectionAssessment {
        law,
        shift_squared,
        correction_error_squared,
        dispersion,
        effective_samples,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::SmallRng};

    /// `ε · magnitude`: the running bound of routes whose intermediate results
    /// add to `magnitude` in absolute value.
    fn rounding(magnitude: f64) -> f64 {
        f64::EPSILON * magnitude
    }

    /// `z = Φ⁻¹(1 − α/(2m))` for m compared cells at the declared rate α.
    fn declared_z(alpha: f64, cells: usize) -> f64 {
        gam_math::probability::standard_normal_quantile(1.0 - alpha / (2.0 * cells as f64)).unwrap()
    }

    fn scalar_factor(precision: f64) -> FaerLlt<f64> {
        let matrix = faer::Mat::from_fn(1, 1, |_, _| precision);
        FaerLlt::new(matrix.as_ref(), faer::Side::Lower).unwrap()
    }

    struct Poisson {
        count: f64,
        exposure: f64,
        error: f64,
    }

    impl CoefficientLikelihood for Poisson {
        fn log_likelihood(&mut self, theta: &[f64]) -> Result<(f64, f64), EventHistoryError> {
            Ok((self.count * theta[0] - self.exposure * theta[0].exp(), self.error))
        }
    }

    struct Gaussian {
        center: f64,
        precision: f64,
    }

    impl CoefficientLikelihood for Gaussian {
        fn log_likelihood(&mut self, theta: &[f64]) -> Result<(f64, f64), EventHistoryError> {
            Ok((-0.5 * self.precision * (theta[0] - self.center).powi(2), 0.0))
        }
    }

    #[test]
    fn mixture_density_is_the_normalized_gaussian_and_student_components() {
        for (precision, center) in [(1.0_f64, 0.0_f64), (8.0, 0.47)] {
            let factor = scalar_factor(precision);
            for x in [center, center + 0.3, center - 2.0, center + 40.0] {
                let r = x - center;
                let gaussian =
                    0.5 * (precision / (2.0 * std::f64::consts::PI)).ln() - 0.5 * precision * r * r;
                // One-dimensional t₃ with covariance 1/S: 2√S/π (1 + S r²)⁻².
                let student = (2.0 * precision.sqrt() / std::f64::consts::PI).ln()
                    - 2.0 * (precision * r * r).ln_1p();
                let expected = log_sum_exp(&[gaussian, student]) - std::f64::consts::LN_2;
                let (actual, laplace) = proposal_log_density(&[center], &factor, &[x]).unwrap();
                // Two routes, each charging its own intermediate results: the
                // closed forms here, and the log-determinant, quadratic, Student
                // normalizer and log-sum-exp of the production route.
                let quadratic = precision * r * r;
                let closed = 0.5 * (precision / (2.0 * std::f64::consts::PI)).ln().abs()
                    + 0.5 * quadratic
                    + (2.0 * precision.sqrt() / std::f64::consts::PI).ln().abs()
                    + 2.0 * quadratic.ln_1p()
                    + gaussian.abs()
                    + student.abs()
                    + expected.abs();
                let production = 0.5 * precision.ln().abs()
                    + 0.5 * (2.0 * std::f64::consts::PI).ln()
                    + 0.5 * quadratic
                    + student_log_gamma_ratio(1).abs()
                    + 0.5 * std::f64::consts::PI.ln()
                    + 2.0 * quadratic.ln_1p()
                    + laplace.abs()
                    + actual.abs();
                let bar = rounding(closed + production + 2.0 * std::f64::consts::LN_2);
                assert!((actual - expected).abs() <= bar, "x {x}: {actual} vs {expected}, bar {bar}");
                assert!((laplace - gaussian).abs() <= bar, "x {x}: {laplace} vs {gaussian}, bar {bar}");
            }
        }
        // lnΓ(5/2) − lnΓ(3/2) = ln(3/2); lnΓ(3) − lnΓ(3/2) = 2 ln 2 − ½ ln π.
        let even = 1.5_f64.ln();
        assert!((student_log_gamma_ratio(2) - even).abs() <= rounding(2.0 * even));
        let odd = 2.0 * std::f64::consts::LN_2 - 0.5 * std::f64::consts::PI.ln();
        let odd_terms = 2.0 * std::f64::consts::LN_2 + 0.5 * std::f64::consts::PI.ln();
        assert!((student_log_gamma_ratio(3) - odd).abs() <= rounding(2.0 * odd_terms + odd.abs()));
    }

    #[test]
    fn corrected_mean_is_published_when_the_bank_resolves_the_laplace_shift() {
        // Seven events over exposure four; the rate has an Exponential(1) prior in
        // its log chart. The posterior is Gamma(8, 5): mode ln(8/5), Laplace
        // precision 8, exact E[θ] = ψ(8) − ln 5.
        let mode = (8.0_f64 / 5.0).ln();
        let factor = scalar_factor(8.0);
        let mut rng = SmallRng::seed_from_u64(2961);
        let draws = draw_coefficients(&[mode], &factor, 16384, &mut rng).unwrap();
        let mut prior = |theta: &[f64]| Ok(theta[0] - theta[0].exp());
        let exact = gam_math::special::digamma(8.0) - 5.0_f64.ln();
        let bank = coefficient_bank(
            &mut Poisson {
                count: 7.0,
                exposure: 4.0,
                error: 0.0,
            },
            &mut prior,
            draws.clone(),
        )
        .unwrap();
        let assessment = assess_correction(&bank, &[mode], &factor).unwrap();
        let CoefficientLaw::Corrected { mean, mean_error } = &assessment.law else {
            panic!("the resolved shift must publish the corrected mean: {:?}", assessment.law);
        };
        // Monte Carlo bar at the declared α = 0.01 over m = 2 cells: the
        // corrected mean against the exact mean, and the mode's separation.
        // mean_error is in ‖·‖_S units; the coordinate error is mean_error / √8.
        let bar = declared_z(0.01, 2) * mean_error / 8.0_f64.sqrt();
        assert!(
            (mean[0] - exact).abs() <= bar,
            "{} vs {exact}, bar {bar}",
            mean[0]
        );
        // Magnitude floor: the Laplace mode misses the exact mean by more than
        // the bar, so the correction is not vacuous.
        assert!(
            (mode - exact).abs() > bar,
            "measured margin {} over bar {bar}",
            (mode - exact).abs()
        );
        // The decision the law came from: estimated Laplace bias above the
        // corrected error, and a bank error inside the posterior spread.
        assert!(
            assessment.shift_squared - assessment.correction_error_squared
                > assessment.correction_error_squared
        );
        assert!(assessment.correction_error_squared < assessment.dispersion);
        // The same draws with an inner error large enough to swamp the weights
        // are refused as unresolved, never reported as a law.
        let noisy = coefficient_bank(
            &mut Poisson {
                count: 7.0,
                exposure: 4.0,
                error: 5.0,
            },
            &mut prior,
            draws,
        )
        .unwrap();
        assert!(matches!(
            assess_correction(&noisy, &[mode], &factor),
            Err(EventHistoryError::IntegrationResolution { .. })
        ));
    }

    #[test]
    fn laplace_mean_stands_when_the_laplace_law_is_exact() {
        let factor = scalar_factor(8.0);
        let mut rng = SmallRng::seed_from_u64(2962);
        // Each draw with its reflection about the mode: q is symmetric there,
        // so an exact Gaussian target gives a weighted shift of zero and the
        // decision does not depend on the seed.
        let mut draws = draw_coefficients(&[0.3], &factor, 2048, &mut rng).unwrap();
        let reflected: Vec<CoefficientDraw> = draws
            .iter()
            .map(|d| {
                let x = 0.6 - d.coefficients[0];
                let (log_proposal_density, log_laplace_density) =
                    proposal_log_density(&[0.3], &factor, &[x]).unwrap();
                CoefficientDraw {
                    coefficients: vec![x],
                    log_proposal_density,
                    log_laplace_density,
                }
            })
            .collect();
        draws.extend(reflected);
        let bank = coefficient_bank(
            &mut Gaussian {
                center: 0.3,
                precision: 8.0,
            },
            &mut |_: &[f64]| Ok(0.0),
            draws,
        )
        .unwrap();
        let assessment = assess_correction(&bank, &[0.3], &factor).unwrap();
        let CoefficientLaw::Laplace { mean_error_bound } = assessment.law else {
            panic!("an exact Laplace law has no resolvable shift: {:?}", assessment.law);
        };
        // Symmetric draws: the shift is below the bank's own error, and the
        // reported bound covers that error.
        assert!(assessment.shift_squared < assessment.correction_error_squared);
        assert!(mean_error_bound >= assessment.correction_error_squared.sqrt());
        // An exact Laplace target: every training weight is the saved
        // Laplace-law weight N/q up to one normalizing constant, within the
        // rounding of the three summands on each side.
        let offsets: Vec<(f64, f64)> = bank
            .draws
            .iter()
            .zip(&bank.log_weights)
            .map(|(d, w)| {
                (
                    w - (d.log_laplace_density - d.log_proposal_density),
                    w.abs() + d.log_laplace_density.abs() + d.log_proposal_density.abs(),
                )
            })
            .collect();
        for &(offset, magnitude) in &offsets {
            let bar = rounding(magnitude + offsets[0].1);
            assert!(
                (offset - offsets[0].0).abs() <= bar,
                "offset {offset} vs {}, bar {bar}",
                offsets[0].0
            );
        }
        // The effective count follows from those weights, r_j = N/q ≤ 2 for the
        // equal mixture: (Σr)²/Σr² ≥ Σr/2.
        let ratios: Vec<f64> = bank
            .draws
            .iter()
            .map(|d| (d.log_laplace_density - d.log_proposal_density).exp())
            .collect();
        let sum: f64 = ratios.iter().sum();
        let squares: f64 = ratios.iter().map(|r| r * r).sum();
        let direct = sum * sum / squares;
        assert!(ratios.iter().all(|&r| r <= 2.0 + rounding(2.0)));
        assert!(direct >= 0.5 * sum);
        assert!(
            (assessment.effective_samples - direct).abs() <= rounding(direct * ratios.len() as f64),
            "{} vs {direct}",
            assessment.effective_samples
        );
        assert!(draw_coefficients(&[0.3], &factor, 1, &mut rng).is_err());
    }
}
