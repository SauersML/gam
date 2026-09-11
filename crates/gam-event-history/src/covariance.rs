//! The latent covariance an event-history fit reports, and the covariance
//! score that grows its rank.
//!
//! The atoms have unit variance, so with loadings `A` (marks × atoms) and
//! rates `r_k` the latent part of the log-intensities has covariance
//!
//! ```text
//! C(Δ) = A diag(e^{−r_k |Δ|}) Aᵀ,        C(0) = A Aᵀ.
//! ```
//!
//! `C` is the scientific object: two atoms with equal rates can be rotated
//! into each other without changing the process, but `C` is invariant. The
//! loadings are its factor coordinates.
//!
//! Filtered residual products supply a cheap rate proposal. They are not
//! the curvature of the integrated likelihood once latent factors exist.
//! The fitting layer differentiates that likelihood for the actual loading
//! curvature, uses sampled directional profiles to propose a prior, and
//! compares jointly fitted ranks using the reported Laplace criterion.
//!
//! The quartic integrals below belong only to the proposal model. Products
//! of one-dimensional profiles omit cross-direction interactions; their
//! gain must not be reported as the joint model's evidence.

use super::cohort::EventHistoryError;
use faer::Side;
use gam_linalg::faer_ndarray::strict_symmetric_eigh;
use ndarray::{Array1, Array2};

/// `A Aᵀ`.
pub fn factor_covariance(loadings: &Array2<f64>) -> Array2<f64> {
    loadings.dot(&loadings.t())
}

/// `C(Δ) = Σ_k C_k e^{−r_k |Δ|}` for one covariance share `C_k = E[a_k a_kᵀ]`
/// per atom and rates in the data's time unit.
pub fn temporal_covariance(
    marks: usize,
    atom_covariances: &[Array2<f64>],
    rates: &[f64],
    lag: f64,
) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((marks, marks));
    for (share, rate) in atom_covariances.iter().zip(rates.iter()) {
        out += &(share * (-rate * lag.abs()).exp());
    }
    out
}

/// Eigenvalues (descending) and matching unit eigenvectors (columns) of a
/// symmetric matrix.
pub fn eigenmodes(matrix: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>), EventHistoryError> {
    let n = matrix.nrows();
    if n == 0 {
        return Ok((Array1::zeros(0), Array2::zeros((0, 0))));
    }
    let (values, vectors) = strict_symmetric_eigh(matrix, Side::Lower).map_err(|error| {
        EventHistoryError::NumericalFailure {
            reason: format!("eigendecomposition of a {n} × {n} covariance failed: {error}"),
        }
    })?;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| values[b].total_cmp(&values[a]));
    let sorted_values = Array1::from_iter(order.iter().map(|&i| values[i]));
    let mut sorted_vectors = Array2::<f64>::zeros((n, n));
    for (j, &i) in order.iter().enumerate() {
        sorted_vectors.column_mut(j).assign(&vectors.column(i));
    }
    Ok((sorted_values, sorted_vectors))
}

/// The participation ratio `(tr C)² / tr(C²)`: the number of equal
/// eigenvalues that would spread the same total variance as `C` does, a
/// continuous count of the directions the covariance actually uses. One for
/// a rank-one covariance, the dimension for an isotropic one, zero for the
/// zero matrix.
pub fn effective_rank(covariance: &Array2<f64>) -> f64 {
    let trace: f64 = covariance.diag().sum();
    let frobenius: f64 = covariance.iter().map(|c| c * c).sum();
    if frobenius > 0.0 {
        trace * trace / frobenius
    } else {
        0.0
    }
}

/// One subject's residual scores at the current fit.
pub(crate) struct SubjectResiduals {
    pub times: Vec<f64>,
    /// `s̄_{nd}`, index `n * marks + d`.
    pub scores: Vec<f64>,
    /// `c̄_{nd}`, index `n * marks + d`.
    pub curvatures: Vec<f64>,
}

/// `Σ_m |t_n − t_m|^j e^{−r |t_n − t_m|} x_m` for `j = 0, 1, 2` at every
/// `n`, for a vector-valued sequence `x` of width `width`, by the
/// forward–backward recursion of the exponential kernel.
fn kernel_sums(times: &[f64], x: &[f64], width: usize, rate: f64) -> [Vec<f64>; 3] {
    let n = times.len();
    let mut forward = [vec![0.0; n * width], vec![0.0; n * width], vec![0.0; n * width]];
    let mut backward = [vec![0.0; n * width], vec![0.0; n * width], vec![0.0; n * width]];
    for i in 0..n {
        let (decay, delta) = if i == 0 {
            (0.0, 0.0)
        } else {
            let delta = times[i] - times[i - 1];
            ((-rate * delta).exp(), delta)
        };
        for c in 0..width {
            let idx = i * width + c;
            let prev = if i == 0 { 0 } else { (i - 1) * width + c };
            let (f0, f1, f2) = if i == 0 {
                (0.0, 0.0, 0.0)
            } else {
                (forward[0][prev], forward[1][prev], forward[2][prev])
            };
            forward[0][idx] = decay * f0 + x[idx];
            forward[1][idx] = decay * (f1 + delta * f0);
            forward[2][idx] = decay * (f2 + 2.0 * delta * f1 + delta * delta * f0);
        }
    }
    for i in (0..n).rev() {
        let (decay, delta) = if i + 1 == n {
            (0.0, 0.0)
        } else {
            let delta = times[i + 1] - times[i];
            ((-rate * delta).exp(), delta)
        };
        for c in 0..width {
            let idx = i * width + c;
            let next = if i + 1 == n { 0 } else { (i + 1) * width + c };
            let (b0, b1, b2) = if i + 1 == n {
                (0.0, 0.0, 0.0)
            } else {
                (backward[0][next], backward[1][next], backward[2][next])
            };
            backward[0][idx] = decay * b0 + x[idx];
            backward[1][idx] = decay * (b1 + delta * b0);
            backward[2][idx] = decay * (b2 + 2.0 * delta * b1 + delta * delta * b0);
        }
    }
    let mut out = forward;
    for c in 0..n * width {
        out[0][c] += backward[0][c] - x[c];
        out[1][c] += backward[1][c];
        out[2][c] += backward[2][c];
    }
    out
}

/// `M`, `dM/dρ`, `d²M/dρ²` at log-rate `ρ` (`rate = e^ρ / time_scale`).
pub(crate) fn covariance_score(
    subjects: &[SubjectResiduals],
    marks: usize,
    log_rate: f64,
    time_scale: f64,
) -> [Array2<f64>; 3] {
    let rate = log_rate.exp() / time_scale;
    let mut m0 = Array2::<f64>::zeros((marks, marks));
    let mut m1 = Array2::<f64>::zeros((marks, marks));
    let mut m2 = Array2::<f64>::zeros((marks, marks));
    for subject in subjects {
        let n = subject.times.len();
        let sums = kernel_sums(&subject.times, &subject.scores, marks, rate);
        for node in 0..n {
            for d in 0..marks {
                let s = subject.scores[node * marks + d];
                if s == 0.0 {
                    continue;
                }
                for e in 0..marks {
                    let idx = node * marks + e;
                    m0[[d, e]] += s * sums[0][idx];
                    m1[[d, e]] -= s * sums[1][idx];
                    m2[[d, e]] += s * sums[2][idx];
                }
            }
            for d in 0..marks {
                m0[[d, d]] -= subject.curvatures[node * marks + d];
            }
        }
    }
    // Symmetrise the double sums, then chain to the log-rate.
    let symmetric = |m: &Array2<f64>| 0.5 * (m + &m.t());
    let m0 = symmetric(&m0);
    let d1 = symmetric(&m1) * rate;
    let d2 = &d1 + symmetric(&m2) * (rate * rate);
    [m0, d1, d2]
}

/// The Fisher information `J` of the variance component along the unit
/// direction `v` at `rate`, and its derivative in the log-rate when the
/// direction's own derivative `v_slope` is supplied.
///
/// The score of the variance component `τ` along `v` at `τ = 0` is
/// `U = ½ vᵀ M v`, and `J` is twice its variance under the fitted null so
/// that the second-order evidence is `½ μ τ − ¼ J τ²`. For a marked counting
/// process the compensated score `s = y − wμ` has, at each node and mark,
/// the Poisson cumulants `Var s = c`, `Var s² = c + 2c²`, so
///
/// ```text
/// J = Σ_{n,m} e^{−2r|t_n−t_m|} κ_n κ_m + ½ Σ_n Σ_d v_d⁴ c_{nd},
/// κ_n = Σ_d v_d² c_{nd}.
/// ```
///
/// The second term is the process's own diagonal — the variance a count
/// carries at a single instant — and it is what stops the standardised gain
/// from running away as the kernel narrows: the smooth part collapses to
/// `Σ_n κ_n²` there while the diagonal stays, so a kernel narrower than the
/// data's own structure buys nothing. Dropping it (the Gaussian-response
/// form of the same statistic) sends the proposal to white noise on any
/// cohort whose residuals are event spikes, which is every point process.
fn direction_information(
    subjects: &[SubjectResiduals],
    marks: usize,
    rate: f64,
    v: &[f64],
    v_slope: Option<&[f64]>,
) -> (f64, f64) {
    let mut information = 0.0;
    let mut information_slope = 0.0;
    for subject in subjects {
        let n = subject.times.len();
        let kappa: Vec<f64> = (0..n)
            .map(|node| (0..marks).map(|d| v[d] * v[d] * subject.curvatures[node * marks + d]).sum())
            .collect();
        let sums = kernel_sums(&subject.times, &kappa, 1, 2.0 * rate);
        for node in 0..n {
            information += kappa[node] * sums[0][node];
            for d in 0..marks {
                let c = subject.curvatures[node * marks + d];
                let v2 = v[d] * v[d];
                information += 0.5 * v2 * v2 * c;
            }
        }
        if let Some(v_slope) = v_slope {
            let kappa_slope: Vec<f64> = (0..n)
                .map(|node| {
                    (0..marks)
                        .map(|d| 2.0 * v[d] * v_slope[d] * subject.curvatures[node * marks + d])
                        .sum()
                })
                .collect();
            for node in 0..n {
                // d/dρ of e^{−2r|Δ|} is −2 r |Δ| e^{−2r|Δ|}; the quadratic
                // form is symmetric, so the κ derivative enters twice. The
                // diagonal is independent of the rate, so it enters the slope
                // only through `v`.
                information_slope += -2.0 * rate * kappa[node] * sums[1][node]
                    + 2.0 * kappa_slope[node] * sums[0][node];
                for d in 0..marks {
                    let c = subject.curvatures[node * marks + d];
                    information_slope += 2.0 * v[d] * v[d] * v[d] * v_slope[d] * c;
                }
            }
        }
    }
    (information, information_slope)
}

/// The exact one-dimensional marginal of the quartic evidence model along
/// one direction: `ln ∫ exp(½ a t² − ¼ J t⁴) dt` with `a = μ − λ`, and the
/// moments `E[t²]`, `E[t⁴]` of `t` under that density.
///
/// The integrand is entire and even, so the trapezoidal rule converges
/// faster than any power of its spacing. The grid resolves the narrowest
/// feature — the peak's width `1/√max(|a|, √J)` — at eight points, where
/// the rule's aliasing error on a feature of that width is `exp(−2π²·64)`,
/// far below roundoff, and it extends to where the integrand has fallen
/// sixty nats below its peak, which is `e⁻⁶⁰` of it. The sums are formed in
/// log space.
pub fn quartic_moments(mu: f64, information: f64, lambda: f64) -> (f64, f64, f64) {
    let a = mu - lambda;
    let j = information;
    let g = |t: f64| 0.5 * a * t * t - 0.25 * j * t * t * t * t;
    let (peak, g_peak) = if a > 0.0 {
        ((a / j).sqrt(), a * a / (4.0 * j))
    } else {
        (0.0, 0.0)
    };
    let width = 1.0 / a.abs().max(j.sqrt()).sqrt();
    let mut half_range = peak + 8.0 * width;
    while g(half_range) > g_peak - 60.0 {
        half_range *= 2.0;
    }
    let spacing = width / 8.0;
    let steps = (half_range / spacing).ceil() as usize;
    let mut shift = f64::NEG_INFINITY;
    let mut values: Vec<(f64, f64)> = Vec::with_capacity(2 * steps + 1);
    for i in -(steps as i64)..=(steps as i64) {
        let t = i as f64 * spacing;
        let value = g(t);
        shift = shift.max(value);
        values.push((t, value));
    }
    let mut mass = 0.0;
    let mut second = 0.0;
    let mut fourth = 0.0;
    for &(t, value) in &values {
        let weight = (value - shift).exp();
        mass += weight;
        second += weight * t * t;
        fourth += weight * t * t * t * t;
    }
    let log_integral = shift + (mass * spacing).ln();
    (log_integral, second / mass, fourth / mass)
}

/// A sampled profile of the marginal log-likelihood along one direction of
/// the loading space, `g(t) = ℓ(t·v) − ℓ(0)` for `t ≥ 0`, sampled with its
/// slope and carried by the cubic Hermite interpolant through the samples.
/// The likelihood is even in `t` (an atom's sign is a gauge), so the
/// profile is stored on `t ≥ 0` and reflected.
#[derive(Clone, Debug)]
pub struct DirectionProfile {
    /// Sample points, `0 = t_0 < t_1 < …`.
    pub points: Vec<f64>,
    /// `g(t_i)`.
    pub values: Vec<f64>,
    /// `g'(t_i)`.
    pub slopes: Vec<f64>,
}

/// Subpoints per sample interval for numerical integration of the proposal.
/// This does not bound the sampled profile's interpolation or truncation error.
const PROFILE_SUBPOINTS: usize = 64;

impl DirectionProfile {
    /// Hermite cubic within the sampled interval.
    fn evaluate(&self, t: f64) -> f64 {
        let n = self.points.len();
        assert!(n >= 2 && t >= 0.0 && t <= self.points[n - 1]);
        let i = match self.points.iter().position(|&p| t < p) {
            Some(0) => 0,
            Some(i) => i - 1,
            None => n - 2,
        };
        let (t0, t1) = (self.points[i], self.points[i + 1]);
        let h = t1 - t0;
        let s = (t - t0) / h;
        let (s2, s3) = (s * s, s * s * s);
        let (h00, h10, h01, h11) = (
            2.0 * s3 - 3.0 * s2 + 1.0,
            s3 - 2.0 * s2 + s,
            -2.0 * s3 + 3.0 * s2,
            s3 - s2,
        );
        h00 * self.values[i]
            + h10 * h * self.slopes[i]
            + h01 * self.values[i + 1]
            + h11 * h * self.slopes[i + 1]
    }

    /// `ln ∫ exp(g(t) − ½λt²) dt` over the whole line and the moments
    /// `E[t²]`, `E[t⁴]` of `t` under that density, by the trapezoidal rule on
    /// the interpolant, plus the maximiser of `g(t) − ½λt²`.
    fn moments(&self, lambda: f64) -> (f64, f64, f64, f64) {
        let last = self.points[self.points.len() - 1];
        let steps = (self.points.len() - 1) * PROFILE_SUBPOINTS;
        let spacing = last / steps as f64;
        let mut shift = f64::NEG_INFINITY;
        let mut mode = 0.0;
        let mut samples: Vec<(f64, f64)> = Vec::with_capacity(steps + 1);
        for k in 0..=steps {
            // `steps · (last / steps)` can round one ulp above `last`, outside
            // the sampled interval; the upper endpoint is the last sample itself.
            let t = if k == steps {
                last
            } else {
                k as f64 * spacing
            };
            let value = self.evaluate(t) - 0.5 * lambda * t * t;
            if value > shift {
                shift = value;
                mode = t;
            }
            samples.push((t, value));
        }
        let mut mass = 0.0;
        let mut second = 0.0;
        let mut fourth = 0.0;
        for (k, &(t, value)) in samples.iter().enumerate() {
            // Reflect the trapezoidal rule: both endpoints have half
            // weight before reflection, including the finite upper endpoint.
            let weight = (value - shift).exp() * if k == 0 || k == steps { 1.0 } else { 2.0 };
            mass += weight;
            second += weight * t * t;
            fourth += weight * t * t * t * t;
        }
        (shift + (mass * spacing).ln(), second / mass, fourth / mass, mode)
    }
}

/// The evidence one direction of the loading space carries, in the form the
/// marginal likelihood integral needs.
#[derive(Clone, Debug)]
pub enum DirectionEvidence {
    /// The score's quartic model `g(t) = ½μt² − ¼Jt⁴`.
    Quartic { eigenvalue: f64, information: f64 },
    /// A finite sampled profile; used for proposing a loading prior.
    Sampled(DirectionProfile),
}

impl DirectionEvidence {
    /// `ln ∫ exp(g(t) − ½λt²) dt`, `E[t²]`, `E[t⁴]`, and the mode of
    /// `g(t) − ½λt²` on `t ≥ 0`.
    fn moments(&self, lambda: f64) -> (f64, f64, f64, f64) {
        match self {
            Self::Quartic {
                eigenvalue,
                information,
            } => {
                let (log_integral, second, fourth) =
                    quartic_moments(*eigenvalue, *information, lambda);
                let mode = if *eigenvalue > lambda {
                    ((eigenvalue - lambda) / information).sqrt()
                } else {
                    0.0
                };
                (log_integral, second, fourth, mode)
            }
            Self::Sampled(profile) => profile.moments(lambda),
        }
    }

    /// The Laplace-scale prior `λ = 1/t*²` at the direction's own maximiser
    /// `t*`, the search's start; `None` when the maximiser is at zero.
    fn scale_start(&self) -> Option<f64> {
        match self {
            Self::Quartic {
                eigenvalue,
                information,
            } => (*eigenvalue > 0.0 && *information > 0.0).then(|| information / eigenvalue),
            Self::Sampled(profile) => {
                let (_, _, _, mode) = profile.moments(0.0);
                (mode > 0.0).then(|| 1.0 / (mode * mode))
            }
        }
    }
}

/// The empirical-Bayes prior of a new atom's loadings and the decision it
/// carries.
#[derive(Clone, Debug)]
pub struct RidgeProfile {
    /// `ln λ̂`: the precision of the isotropic Gaussian prior that maximises
    /// the marginal likelihood under the quartic model; `+∞` when no finite
    /// prior raises it.
    pub log_lambda: f64,
    /// `ln p(y | λ̂) − ln p(y | λ = ∞)` in nats: the evidence the prior buys
    /// against the current rank (zero when nothing is bought).
    pub gain: f64,
    /// The loading magnitude at the posterior mode along the top direction,
    /// `√((μ_max − λ̂) / J_max)`; zero when the mode is at zero.
    pub mode_scale: f64,
    /// `λ̂ < μ_max`: the prior places the posterior mode of the loading away
    /// from zero. This is the acceptance of the atom.
    pub accepted: bool,
}

/// The empirical-Bayes prior for a loading vector from the evidence along
/// the eigen-directions of the covariance score: the first direction's
/// integrand is the quartic or sampled profile carried by
/// [`DirectionEvidence`]. This factorisation is a proposal approximation.
///
/// The negative log marginal likelihood
/// `c(λ) = −Σ_i ln Z_i(λ)` is `+∞` at `λ → 0` (a prior too loose to
/// normalise) and tends to zero as `λ → ∞` (the atom pinned to zero, the
/// current rank). It is minimised over `ρ = ln λ` by a safeguarded Newton
/// on its exact derivatives, `dc/dλ = Σ_i [½ E_i[t²] − 1/(2λ)]` and
/// `d²c/dλ² = Σ_i [1/(2λ²) − ¼ Var_i(t²)]`, from the Laplace-scale start of
/// the first direction. A profile whose first direction has its maximiser
/// at zero has no finite minimiser and is refused without a search.
pub fn empirical_bayes_ridge(directions: &[DirectionEvidence]) -> RidgeProfile {
    let refused = RidgeProfile {
        log_lambda: f64::INFINITY,
        gain: 0.0,
        mode_scale: 0.0,
        accepted: false,
    };
    let Some(start) = directions.first().and_then(DirectionEvidence::scale_start) else {
        return refused;
    };
    let profile = |rho: f64| -> (f64, f64, f64, f64) {
        let lambda = rho.exp();
        let mut value = 0.0;
        let mut d_lambda = 0.0;
        let mut d2_lambda = 0.0;
        let mut top_mode = 0.0;
        for (i, direction) in directions.iter().enumerate() {
            let (log_integral, second, fourth, mode) = direction.moments(lambda);
            if i == 0 {
                top_mode = mode;
            }
            value += -0.5 * (lambda / (2.0 * std::f64::consts::PI)).ln() - log_integral;
            d_lambda += 0.5 * second - 0.5 / lambda;
            d2_lambda += 0.5 / (lambda * lambda) - 0.25 * (fourth - second * second);
        }
        let d_rho = lambda * d_lambda;
        let d2_rho = lambda * d_lambda + lambda * lambda * d2_lambda;
        (value, d_rho, d2_rho, top_mode)
    };
    let mut rho = start.ln();
    let (mut value, mut slope, mut curvature, mut mode_scale) = profile(rho);
    for _ in 0..200 {
        let tolerance = f64::EPSILON.sqrt() * (1.0 + value.abs());
        if !(slope.abs() > tolerance) {
            break;
        }
        let direction = if curvature > 0.0 {
            (-slope / curvature).clamp(-2.0, 2.0)
        } else {
            -slope.signum()
        };
        let mut t = 1.0;
        let mut moved = false;
        for _ in 0..60 {
            let trial = rho + t * direction;
            let (trial_value, trial_slope, trial_curvature, trial_mode) = profile(trial);
            if trial_value < value {
                rho = trial;
                value = trial_value;
                slope = trial_slope;
                curvature = trial_curvature;
                mode_scale = trial_mode;
                moved = true;
                break;
            }
            t *= 0.5;
        }
        if !moved {
            break;
        }
    }
    let accepted = value < 0.0 && mode_scale > 0.0;
    RidgeProfile {
        log_lambda: rho,
        gain: -value,
        mode_scale: if accepted { mode_scale } else { 0.0 },
        accepted,
    }
}

/// The atom the covariance score proposes and the evidence's verdict on it.
#[derive(Clone, Debug)]
pub(crate) struct NewAtom {
    pub log_rate: f64,
    /// The unit direction the score proposes: the top eigenvector of `M`.
    pub direction: Vec<f64>,
    /// The loading vector at the posterior mode under the empirical-Bayes
    /// prior: `mode_scale · direction`.
    pub loading: Vec<f64>,
    /// The score's top eigenvalue at the proposed rate.
    pub eigenvalue: f64,
    /// `μ_max² / (4 J_max)`: the second-order evidence gain of the top
    /// direction, the matched-filter statistic the rate maximises.
    pub standardised_gain: f64,
    /// The empirical-Bayes prior and its decision.
    pub ridge: RidgeProfile,
    /// The residuals cannot tell the proposed rate from one twice as slow
    /// (the gain is flat to double precision across that doubling), or the
    /// proposal sits at [`resolvable_rate_band`]'s lower limit. This proposes
    /// an actual static factor, whose likelihood is evaluated at rate zero.
    /// Flatness of the proposal statistic does not prove likelihood equality.
    pub at_lower_limit: bool,
    /// The residuals cannot tell the proposed rate from one twice as fast,
    /// or the proposal wanted a rate the node mesh cannot resolve and was
    /// held at [`resolvable_rate_band`]'s upper limit: the residuals carry
    /// structure faster than the quadrature mesh, and a finer mesh would
    /// see more of it.
    pub at_upper_limit: bool,
}

impl NewAtom {
    /// On either plateau the likelihood is flat in the log-rate to double
    /// precision, so the rate is held there as data rather than fitted as a
    /// coordinate whose mode no certificate could resolve.
    pub fn rate_held(&self) -> bool {
        self.at_lower_limit || self.at_upper_limit
    }
}


/// The band `(ν_min, ν_max)` of dimensionless rates `ν = rate · T̄` a set of
/// breakpoints resolves — the fit passes the cohort's own level-0 cell
/// boundaries, never quadrature nodes, so the band is a property of the
/// data and the same at every mesh refinement. `None` when no subject has
/// two distinct times.
///
/// Below `lower` the Ornstein–Uhlenbeck kernel `exp(−ν · gap / T̄)` is one
/// to double precision across the longest follow-up: the atom is a static
/// frailty, and every slower rate is the same model. That is a
/// floating-point statement, and the fit is free to sit on it — a static
/// frailty is a perfectly good latent state.
///
/// Above `upper` the atom decorrelates between consecutive breakpoints of
/// the data (`κ = rate · width ≤ 1` at the median level-0 cell): between
/// two breakpoints the data observe only the integrated exposure, so a
/// faster atom is a static offset of the intensity, not a process the data
/// can time, and a proposal that wants to sit on this limit is telling the
/// caller the residuals carry structure the design cannot resolve.
pub fn resolvable_rate_band<'a>(
    times: impl IntoIterator<Item = &'a [f64]>,
    time_scale: f64,
) -> Option<(f64, f64)> {
    let mut longest = 0.0_f64;
    let mut gaps: Vec<f64> = Vec::new();
    for times in times {
        if let (Some(first), Some(last)) = (times.first(), times.last()) {
            longest = longest.max(last - first);
        }
        gaps.extend(times.windows(2).map(|w| w[1] - w[0]).filter(|g| *g > 0.0));
    }
    if !(longest > 0.0) || gaps.is_empty() || !(time_scale.is_finite() && time_scale > 0.0) {
        return None;
    }
    gaps.sort_by(f64::total_cmp);
    let median = gaps[gaps.len() / 2];
    let lower = f64::EPSILON.sqrt() * time_scale / longest;
    let upper = time_scale / median;
    if !(lower.is_finite() && upper.is_finite() && lower > 0.0) || upper <= lower {
        return None;
    }
    Some((lower, upper))
}

/// The standardised evidence gain of the top direction at log-rate `ρ`,
/// with its exact derivative in `ρ`, and the score's spectrum there.
struct GainPoint {
    rho: f64,
    gain: f64,
    slope: f64,
    top: f64,
    information: f64,
    values: Array1<f64>,
    vectors: Array2<f64>,
}

/// Propose the next atom: at every log-rate the top eigenpair `(μ, v)` of
/// `M(ρ)` is the direction with the largest second-order evidence slope in
/// its variance, and the standardised gain `μ² / (4 J)` is maximised over
/// `ρ` by a secant Newton on its exact derivative (Hellmann–Feynman for
/// `μ`, first-order eigenvector perturbation for `v`, the kernel recursion
/// for `J`). At the rate found, the empirical-Bayes prior of the loadings is
/// computed from the score's whole spectrum and decides the atom. `None`
/// only when the cohort's mesh admits no rate at all.
pub(crate) fn best_new_atom(
    subjects: &[SubjectResiduals],
    marks: usize,
    time_scale: f64,
    band: (f64, f64),
) -> Result<Option<NewAtom>, EventHistoryError> {
    let evaluate = |rho: f64| -> Result<GainPoint, EventHistoryError> {
        let [m0, m1, _] = covariance_score(subjects, marks, rho, time_scale);
        let (values, vectors) = eigenmodes(&m0)?;
        let top = values[0];
        let v: Vec<f64> = vectors.column(0).to_vec();
        let vv = Array1::from(v.clone());
        let m1v = m1.dot(&vv);
        let top_slope = vv.dot(&m1v);
        let mut v_slope = vec![0.0; marks];
        for j in 1..marks {
            let gap = top - values[j];
            if gap > 0.0 {
                let coupling = vectors.column(j).dot(&m1v) / gap;
                for d in 0..marks {
                    v_slope[d] += coupling * vectors[[d, j]];
                }
            }
        }
        let rate = rho.exp() / time_scale;
        let (information, information_slope) =
            direction_information(subjects, marks, rate, &v, Some(&v_slope));
        if !(information > 0.0) {
            return Ok(GainPoint {
                rho,
                gain: 0.0,
                slope: 0.0,
                top,
                information,
                values,
                vectors,
            });
        }
        // Signed gain μ|μ| / (4J): odd in μ, so the ascent also moves a
        // negative top eigenvalue toward where it turns positive.
        let gain = top * top.abs() / (4.0 * information);
        let slope = (2.0 * top.abs() * top_slope * information - top * top.abs() * information_slope)
            / (4.0 * information * information);
        Ok(GainPoint {
            rho,
            gain,
            slope,
            top,
            information,
            values,
            vectors,
        })
    };
    let (lower, upper) = (band.0.ln(), band.1.ln());
    if !(lower.is_finite() && upper.is_finite() && upper > lower) {
        return Ok(None);
    }
    // Start at the cohort's own time scale (rate = 1 / T̄).
    let mut point = evaluate(0.0_f64.clamp(lower, upper))?;
    let mut curvature: Option<f64> = None;
    for _ in 0..100 {
        let tolerance = f64::EPSILON.sqrt() * (1.0 + point.gain.abs());
        if !(point.slope.abs() > tolerance) {
            break;
        }
        // Secant Newton on the exact slope; a unit step in log-rate when no
        // negative curvature is known yet, never more than two per step.
        let direction = match curvature {
            Some(h) if h < 0.0 => (-point.slope / h).clamp(-2.0, 2.0),
            _ => point.slope.signum(),
        };
        let mut t = 1.0;
        let mut moved = false;
        for _ in 0..60 {
            let target = (point.rho + t * direction).clamp(lower, upper);
            if target == point.rho {
                break;
            }
            let trial = evaluate(target)?;
            if trial.gain > point.gain {
                curvature = Some((trial.slope - point.slope) / (trial.rho - point.rho));
                point = trial;
                moved = true;
                break;
            }
            t *= 0.5;
        }
        if !moved {
            break;
        }
    }
    let rate = point.rho.exp() / time_scale;
    // The information along every eigen-direction of the score at the rate
    // found: the quartic model of the evidence is read on the whole spectrum,
    // because the prior is isotropic and its Occam factor charges every
    // direction the loading could take.
    let other_directions: Vec<(f64, f64)> = (1..marks)
        .map(|i| {
            let v: Vec<f64> = point.vectors.column(i).to_vec();
            let (information, _) = direction_information(subjects, marks, rate, &v, None);
            (point.values[i], information)
        })
        .collect();
    let directions: Vec<DirectionEvidence> = std::iter::once(DirectionEvidence::Quartic {
        eigenvalue: point.top,
        information: point.information,
    })
    .chain(
        other_directions
            .iter()
            .map(|&(eigenvalue, information)| DirectionEvidence::Quartic {
                eigenvalue,
                information,
            }),
    )
    .collect();
    let ridge = empirical_bayes_ridge(&directions);
    let standardised_gain = if point.information > 0.0 {
        point.top * point.top.abs() / (4.0 * point.information)
    } else {
        0.0
    };
    let direction: Vec<f64> = point.vectors.column(0).to_vec();
    let loading: Vec<f64> = direction.iter().map(|x| ridge.mode_scale * x).collect();
    // A plateau: the gain does not change, to the resolution the search
    // itself converged at, when the rate is halved or doubled. The kernel is
    // monotone in the rate, so a gain flat across one doubling is flat all
    // the way to the limit on that side.
    let margin = |limit: f64| f64::EPSILON.sqrt() * (1.0 + limit.abs());
    let resolution = f64::EPSILON.sqrt() * (1.0 + point.gain.abs());
    let flat_toward = |delta: f64| -> Result<bool, EventHistoryError> {
        let target = (point.rho + delta).clamp(lower, upper);
        if target == point.rho {
            return Ok(true);
        }
        let trial = evaluate(target)?;
        Ok((trial.gain - point.gain).abs() <= resolution)
    };
    let flat_slower = flat_toward(-std::f64::consts::LN_2)?;
    let flat_faster = flat_toward(std::f64::consts::LN_2)?;
    // Flat on both sides is the static plateau: the rate is already too slow
    // for the follow-up to see, and doubling it changes nothing either. The
    // fast plateau is the one that is flat only toward faster rates, or the
    // wall the search was clamped at.
    let at_lower_limit = point.rho <= lower + margin(lower) || flat_slower;
    let at_upper_limit = point.rho >= upper - margin(upper) || (flat_faster && !flat_slower);
    Ok(Some(NewAtom {
        log_rate: if at_lower_limit { f64::NEG_INFINITY } else { point.rho },
        direction,
        loading,
        eigenvalue: point.top,
        standardised_gain,
        ridge,
        at_lower_limit,
        at_upper_limit,
    }))
}
