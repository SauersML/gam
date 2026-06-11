//! Exact marginal smoothing inference over the smoothing parameters `ρ`
//! (issue #938), Tier 0: the **PSIS certificate**.
//!
//! Every GAM ecosystem conditions inference on the estimated smoothing
//! parameters `ρ̂`; intervals from `V(β̂|ρ̂)` undercover because they ignore
//! `ρ`-uncertainty. The honest marginal posterior factorizes as
//! `π(β, ρ | y) = π(β | ρ, y) · π(ρ | y)`, where `π(ρ|y) ∝ exp(−criterion(ρ))`
//! is exactly the LAML/REML objective the outer optimizer already minimizes
//! with exact gradients.
//!
//! Tier 0 turns "should I worry about `ρ`-uncertainty?" — currently folklore —
//! into a *computed* diagnostic on every fit, with no MCMC:
//!
//! 1. Treat the Laplace approximation at `ρ̂`, `N(ρ̂, H_ρ⁻¹)` (with `H_ρ` the
//!    exact outer Hessian), as the importance proposal.
//! 2. Draw `M` whitened samples `z_m ~ N(0, I)`, map them to
//!    `ρ_m = ρ̂ + L z_m` where `L Lᵀ = H_ρ⁻¹`.
//! 3. The importance weight is `w_m = π(ρ_m|y) / proposal(ρ_m)`; in log-space the
//!    Gaussian proposal's quadratic cancels to the whitened norm, giving
//!    `log w_m = −criterion(ρ_m) + criterion(ρ̂) + ½‖z_m‖²` (the `criterion(ρ̂)`
//!    shift makes the weights self-normalized and finite).
//! 4. Pareto-smooth the weights ([`crate::inference::psis`]) and read the
//!    Zhang–Stephens tail shape `k̂`. `k̂ < 0.5` ⇒ the plug-in + first-order
//!    correction answer is **certified** adequate; `0.5 ≤ k̂ ≤ 0.7` ⇒ usable as
//!    a self-normalized importance correction; `k̂ > 0.7` ⇒ the Laplace proposal
//!    is a poor fit and the honest path is a full quadrature/NUTS escalation.
//!
//! The certificate is deterministic: the whitened draws come from a fixed-seed
//! splitmix64 + Box–Muller stream, so the same fit yields the same `k̂` every
//! run.

use crate::inference::psis::pareto_smooth_weights;
use ndarray::{Array1, Array2};

/// Reliability tier read off the Pareto tail-shape `k̂` of the `ρ`-importance
/// weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RhoCertificate {
    /// `k̂ < 0.5`: the Laplace proposal is excellent — the plug-in (REML
    /// conditional) intervals plus the first-order `V_ρ` correction are
    /// certified adequate; `ρ`-uncertainty does not need a heavier treatment.
    PlugInCertified,
    /// `0.5 ≤ k̂ ≤ 0.7`: the proposal is usable but the self-normalized
    /// importance weights should be used to correct moments.
    ImportanceCorrect,
    /// `k̂ > 0.7`: the Laplace proposal poorly captures `π(ρ|y)`; escalate to
    /// quadrature (small `K`) or NUTS over `ρ`.
    Escalate,
}

impl RhoCertificate {
    fn from_k_hat(k_hat: f64) -> Self {
        if !k_hat.is_finite() || k_hat > 0.7 {
            RhoCertificate::Escalate
        } else if k_hat < 0.5 {
            RhoCertificate::PlugInCertified
        } else {
            RhoCertificate::ImportanceCorrect
        }
    }
}

/// The Tier-0 `ρ`-uncertainty certificate for a fit.
#[derive(Debug, Clone)]
pub struct RhoPosteriorCertificate {
    /// Pareto tail-shape of the importance weights — the reliability diagnostic.
    pub k_hat: f64,
    /// The reliability tier derived from `k_hat`.
    pub certificate: RhoCertificate,
    /// Number of proposal draws `M`.
    pub n_samples: usize,
    /// Self-normalized importance weights (length `M`), Pareto-smoothed. These
    /// turn the `M` conditional Gaussians into a free self-normalized mixture
    /// when the tier is `ImportanceCorrect`.
    pub weights: Array1<f64>,
    /// Kish effective sample size `(Σw)² / Σw²` — how many of the `M` draws are
    /// "really" contributing after importance weighting.
    pub effective_sample_size: f64,
}

const DEFAULT_M: usize = 64;
const CERTIFICATE_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// Deterministic standard-normal stream (splitmix64 + Box–Muller). No RNG / env
/// dependency: the same seed yields the same draws every run.
struct DetNormal {
    state: u64,
}
impl DetNormal {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn uniform(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (((z >> 11) as f64) + 0.5) / ((1u64 << 53) as f64)
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Lower-triangular Cholesky `L` (`L Lᵀ = a`) with diagonal jitter for safety.
/// Returns `None` when `a` is not positive definite even after jitter.
fn cholesky_lower(a: &Array2<f64>) -> Option<Array2<f64>> {
    let n = a.nrows();
    if n == 0 || a.ncols() != n {
        return None;
    }
    let scale = (0..n)
        .map(|i| a[[i, i]].abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let jitter = 1e-10 * scale;
    let mut l = Array2::<f64>::zeros((n, n));
    for j in 0..n {
        let mut d = a[[j, j]] + jitter;
        for k in 0..j {
            d -= l[[j, k]] * l[[j, k]];
        }
        if !(d.is_finite() && d > 0.0) {
            return None;
        }
        let ljj = d.sqrt();
        l[[j, j]] = ljj;
        for i in (j + 1)..n {
            let mut s = a[[i, j]];
            for k in 0..j {
                s -= l[[i, k]] * l[[j, k]];
            }
            l[[i, j]] = s / ljj;
        }
    }
    Some(l)
}

/// Solve `H_ρ⁻¹`'s Cholesky `L_inv` (with `L_inv L_invᵀ = H_ρ⁻¹`) from the outer
/// Hessian `H_ρ`. We factor `H_ρ = R Rᵀ` (lower `R`) and use `L_inv = R⁻ᵀ`: then
/// `L_inv L_invᵀ = R⁻ᵀ R⁻¹ = (R Rᵀ)⁻¹ = H_ρ⁻¹`. Mapping `ρ_m = ρ̂ + L_inv z_m`
/// gives draws with covariance `H_ρ⁻¹`, and `‖z_m‖² = (ρ_m−ρ̂)ᵀ H_ρ (ρ_m−ρ̂)`.
fn whitening_factor_from_outer_hessian(outer_hessian: &Array2<f64>) -> Option<Array2<f64>> {
    let r = cholesky_lower(outer_hessian)?;
    let n = r.nrows();
    // Invert-transpose: solve R z = e_i columns to build R⁻¹, then transpose.
    // L_inv = R⁻ᵀ, so column j of L_inv is row j of R⁻¹. Build R⁻¹ by forward
    // substitution against identity columns.
    let mut r_inv = Array2::<f64>::zeros((n, n));
    for col in 0..n {
        // Solve R x = e_col (lower triangular forward substitution).
        let mut x = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut acc = if i == col { 1.0 } else { 0.0 };
            for k in 0..i {
                acc -= r[[i, k]] * x[k];
            }
            let rii = r[[i, i]];
            if !(rii.is_finite() && rii.abs() > 0.0) {
                return None;
            }
            x[i] = acc / rii;
        }
        for i in 0..n {
            r_inv[[i, col]] = x[i];
        }
    }
    // L_inv = R⁻ᵀ.
    let mut l_inv = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            l_inv[[i, j]] = r_inv[[j, i]];
        }
    }
    Some(l_inv)
}

/// Compute the Tier-0 PSIS `ρ`-certificate.
///
/// * `rho_hat` — the converged smoothing parameters `ρ̂` (length `K`).
/// * `outer_hessian` — the exact outer Hessian `H_ρ` of the criterion at `ρ̂`
///   (`K × K`, SPD). The proposal covariance is `H_ρ⁻¹`.
/// * `criterion` — evaluates the outer criterion `−log π(ρ|y)` (the LAML/REML
///   objective) at a trial `ρ`; returns `None` for infeasible `ρ`. This is the
///   `OuterObjective::eval_cost` contract, supplied by the caller that retains
///   (or rebuilds) the objective.
/// * `n_samples` — proposal draw count `M` (defaults to 64 when `None`).
///
/// Returns `None` when `K = 0`, the outer Hessian is not usable, the criterion
/// at `ρ̂` is infeasible, or too few finite weights survive for a tail fit.
pub fn rho_posterior_certificate<F>(
    rho_hat: &Array1<f64>,
    outer_hessian: &Array2<f64>,
    criterion: F,
    n_samples: Option<usize>,
) -> Option<RhoPosteriorCertificate>
where
    F: Fn(&Array1<f64>) -> Option<f64>,
{
    let k = rho_hat.len();
    if k == 0 || outer_hessian.nrows() != k || outer_hessian.ncols() != k {
        return None;
    }
    let cost_hat = criterion(rho_hat)?;
    if !cost_hat.is_finite() {
        return None;
    }
    let l_inv = whitening_factor_from_outer_hessian(outer_hessian)?;
    let m = n_samples
        .unwrap_or(DEFAULT_M)
        .max(2 * crate::inference::psis::MIN_TAIL_COUNT);

    let mut rng = DetNormal::new(CERTIFICATE_SEED);
    let mut raw_weights: Vec<f64> = Vec::with_capacity(m);
    for _ in 0..m {
        let z: Array1<f64> = Array1::from_iter((0..k).map(|_| rng.normal()));
        // ρ_m = ρ̂ + L_inv z.
        let mut rho_m = rho_hat.clone();
        for i in 0..k {
            let mut acc = 0.0;
            for j in 0..k {
                acc += l_inv[[i, j]] * z[j];
            }
            rho_m[i] += acc;
        }
        let half_norm_sq = 0.5 * z.iter().map(|&v| v * v).sum::<f64>();
        // log w_m = −criterion(ρ_m) + criterion(ρ̂) + ½‖z_m‖².
        let log_w = match criterion(&rho_m) {
            Some(c) if c.is_finite() => -c + cost_hat + half_norm_sq,
            // Infeasible / non-finite criterion ⇒ zero importance weight.
            _ => f64::NEG_INFINITY,
        };
        raw_weights.push(log_w);
    }

    // Stabilize and exponentiate: subtract the max log-weight (cancels in the
    // self-normalized weights and the Pareto fit).
    let max_lw = raw_weights
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    if !max_lw.is_finite() {
        return None;
    }
    let weights: Vec<f64> = raw_weights
        .iter()
        .map(|&lw| {
            if lw.is_finite() {
                (lw - max_lw).exp()
            } else {
                0.0
            }
        })
        .collect();

    let psis = pareto_smooth_weights(&weights)?;
    let k_hat = psis.k_hat;

    // Self-normalize the smoothed weights.
    let total: f64 = psis.smoothed.iter().sum();
    if !(total.is_finite() && total > 0.0) {
        return None;
    }
    let normalized: Array1<f64> = Array1::from_iter(psis.smoothed.iter().map(|&w| w / total));
    let sum_sq: f64 = normalized.iter().map(|&w| w * w).sum();
    let ess = if sum_sq > 0.0 { 1.0 / sum_sq } else { 0.0 };

    Some(RhoPosteriorCertificate {
        k_hat,
        certificate: RhoCertificate::from_k_hat(k_hat),
        n_samples: m,
        weights: normalized,
        effective_sample_size: ess,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// CLOSED-FORM FIXTURE: when the criterion IS exactly the Gaussian
    /// `−log π(ρ|y) = ½(ρ−ρ̂)ᵀ H_ρ (ρ−ρ̂)` that the Laplace proposal assumes,
    /// the importance weights are all identically 1 — the proposal is the
    /// target. PSIS must then report a tiny `k̂` and certify the plug-in.
    #[test]
    fn exact_gaussian_target_certifies_plug_in() {
        let rho_hat = array![0.3, -0.7];
        let h = array![[2.0, 0.5], [0.5, 1.5]];
        // criterion(ρ) = ½ (ρ−ρ̂)ᵀ H (ρ−ρ̂): exactly the proposal's negative log
        // density up to the constant that cancels in the self-normalized weight.
        let crit = |rho: &Array1<f64>| {
            let d = rho - &rho_hat;
            let mut q = 0.0;
            for i in 0..2 {
                for j in 0..2 {
                    q += d[i] * h[[i, j]] * d[j];
                }
            }
            Some(0.5 * q)
        };
        let cert = rho_posterior_certificate(&rho_hat, &h, crit, Some(256)).expect("certificate");
        // All weights equal ⇒ ESS == M and k̂ small ⇒ plug-in certified.
        assert!(
            (cert.effective_sample_size - cert.n_samples as f64).abs() < 1e-6,
            "uniform weights must give ESS == M: ess={} M={}",
            cert.effective_sample_size,
            cert.n_samples
        );
        assert!(
            cert.k_hat < 0.5,
            "exact-Gaussian target must yield small k̂, got {}",
            cert.k_hat
        );
        assert_eq!(cert.certificate, RhoCertificate::PlugInCertified);
    }

    /// When the true `π(ρ|y)` is much HEAVIER-tailed than the Gaussian Laplace
    /// proposal (a criterion far flatter than the proposal quadratic in the
    /// tails), the importance weights blow up and PSIS must refuse to certify
    /// the plug-in — `k̂` rises and the tier escalates.
    #[test]
    fn heavy_tailed_target_refuses_to_certify() {
        let rho_hat = array![0.0];
        let h = array![[4.0]]; // tight proposal (variance 0.25).
        // Target ∝ a heavy Student-like tail: criterion grows only
        // logarithmically, so far in the tail π(ρ)/proposal(ρ) → ∞.
        let crit = |rho: &Array1<f64>| {
            let r = rho[0];
            Some((1.0 + r * r).ln())
        };
        let cert = rho_posterior_certificate(&rho_hat, &h, crit, Some(512)).expect("certificate");
        assert!(
            cert.k_hat > 0.5,
            "heavy-tailed target must raise k̂ above 0.5, got {}",
            cert.k_hat
        );
        assert_ne!(cert.certificate, RhoCertificate::PlugInCertified);
    }

    #[test]
    fn weights_are_normalized_and_deterministic() {
        let rho_hat = array![1.0];
        let h = array![[1.0]];
        let crit = |rho: &Array1<f64>| {
            let d = rho[0] - 1.0;
            Some(0.5 * d * d)
        };
        let a = rho_posterior_certificate(&rho_hat, &h, crit, Some(64)).expect("a");
        let b = rho_posterior_certificate(&rho_hat, &h, crit, Some(64)).expect("b");
        // Self-normalized weights sum to 1.
        let s: f64 = a.weights.iter().sum();
        assert!((s - 1.0).abs() < 1e-10, "weights must sum to 1, got {s}");
        // Deterministic: identical k̂ across runs (fixed-seed stream).
        assert_eq!(a.k_hat.to_bits(), b.k_hat.to_bits());
    }

    #[test]
    fn empty_rho_returns_none() {
        let rho_hat: Array1<f64> = array![];
        let h = Array2::<f64>::zeros((0, 0));
        assert!(rho_posterior_certificate(&rho_hat, &h, |_| Some(0.0), None).is_none());
    }
}
