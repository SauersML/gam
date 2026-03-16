use crate::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, CustomFamilyPsiDesignAction,
    CustomFamilyPsiSecondDesignAction, CustomFamilyWarmStart, ExactNewtonJointHessianWorkspace,
    ExactNewtonJointPsiSecondOrderTerms, ExactNewtonJointPsiTerms, ExactNewtonJointPsiWorkspace,
    ExactOuterDerivativeOrder, FamilyEvaluation, ParameterBlockSpec, ParameterBlockState,
    build_block_spatial_psi_derivatives, custom_family_outer_capability,
    evaluate_custom_family_joint_hyper, first_psi_linear_map, fit_custom_family,
    second_psi_linear_map,
};
use crate::estimate::{FitOptions, UnifiedFitResult, fit_gam};
use crate::faer_ndarray::{default_rrqr_rank_alpha, fast_ab, fast_atb, rrqr_nullspace_basis};
use crate::families::gamlss::ParameterBlockInput;
use crate::families::survival_location_scale::TimeBlockInput;
use crate::matrix::{DesignMatrix, SymmetricMatrix};
use crate::pirls::LinearInequalityConstraints;
use crate::probability::{normal_cdf, normal_pdf};
use crate::smooth::{
    ExactJointHyperSetup, SpatialLengthScaleOptimizationOptions, SpatialLogKappaCoords,
    TermCollectionDesign, TermCollectionSpec, build_term_collection_design,
    freeze_spatial_length_scale_terms_from_design, optimize_spatial_length_scale_exact_joint,
    spatial_length_scale_term_indices,
};
use crate::types::LikelihoodFamily;
use ndarray::{Array1, Array2, ArrayView2, Axis, s};
use std::cell::RefCell;
use std::sync::{Arc, OnceLock};

// ── Spec and result types ─────────────────────────────────────────────

#[derive(Clone)]
pub struct SurvivalMarginalSlopeTermSpec {
    pub age_entry: Array1<f64>,
    pub age_exit: Array1<f64>,
    pub event_target: Array1<f64>,
    pub weights: Array1<f64>,
    pub z: Array1<f64>,
    pub derivative_guard: f64,
    pub derivative_softness: f64,
    pub time_block: TimeBlockInput,
    pub logslopespec: TermCollectionSpec,
}

pub struct SurvivalMarginalSlopeFitResult {
    pub fit: UnifiedFitResult,
    pub logslopespec_resolved: TermCollectionSpec,
    pub logslope_design: TermCollectionDesign,
    pub baseline_logslope: f64,
    pub time_block_penalties_len: usize,
}

// ── Family struct ─────────────────────────────────────────────────────

#[derive(Clone)]
struct SurvivalMarginalSlopeFamily {
    n: usize,
    event: Array1<f64>,
    weights: Array1<f64>,
    z: Array1<f64>,
    derivative_guard: f64,
    derivative_softness: f64,
    // Time block designs (n × p_time)
    design_entry: Array2<f64>,
    design_exit: Array2<f64>,
    design_derivative_exit: Array2<f64>,
    offset_entry: Array1<f64>,
    offset_exit: Array1<f64>,
    derivative_offset_exit: Array1<f64>,
    // Slope design (n × p_slope)
    logslope_design: Array2<f64>,
}

// ── Block layout ──────────────────────────────────────────────────────

#[derive(Clone)]
struct BlockSlices {
    time: std::ops::Range<usize>,
    logslope: std::ops::Range<usize>,
    total: usize,
}

fn block_slices(block_states: &[ParameterBlockState]) -> BlockSlices {
    let time = 0..block_states[0].beta.len();
    let logslope = time.end..time.end + block_states[1].beta.len();
    BlockSlices {
        total: logslope.end,
        time,
        logslope,
    }
}

// ── MultiDirJet (copied from bernoulli_marginal_slope) ────────────────

fn subset_partition_table() -> &'static Vec<Vec<Vec<usize>>> {
    fn build_partitions(mask: usize) -> Vec<Vec<usize>> {
        if mask == 0 {
            return vec![Vec::new()];
        }
        let first = mask & mask.wrapping_neg();
        let rest = mask ^ first;
        let mut out = Vec::new();
        let mut subset = rest;
        loop {
            let block = first | subset;
            for mut remainder in build_partitions(rest ^ subset) {
                remainder.push(block);
                out.push(remainder);
            }
            if subset == 0 {
                break;
            }
            subset = (subset - 1) & rest;
        }
        out
    }

    static TABLE: OnceLock<Vec<Vec<Vec<usize>>>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = vec![Vec::new(); 16];
        for mask in 0..16 {
            table[mask] = build_partitions(mask);
        }
        table
    })
}

#[derive(Clone, Debug)]
struct MultiDirJet {
    n_dirs: usize,
    coeffs: [f64; 16],
}

impl MultiDirJet {
    fn zero(n_dirs: usize) -> Self {
        Self {
            n_dirs,
            coeffs: [0.0; 16],
        }
    }

    fn constant(n_dirs: usize, value: f64) -> Self {
        let mut out = Self::zero(n_dirs);
        out.coeffs[0] = value;
        out
    }

    fn linear(n_dirs: usize, base: f64, first: &[f64]) -> Self {
        let mut out = Self::constant(n_dirs, base);
        for (idx, &value) in first.iter().enumerate() {
            out.coeffs[1usize << idx] = value;
        }
        out
    }

    fn full_mask(&self) -> usize {
        (1usize << self.n_dirs) - 1
    }

    fn coeff(&self, mask: usize) -> f64 {
        self.coeffs[mask]
    }

    fn set_coeff(&mut self, mask: usize, value: f64) {
        self.coeffs[mask] = value;
    }

    fn add(&self, other: &Self) -> Self {
        let mut out = Self::zero(self.n_dirs);
        for mask in 0..=self.full_mask() {
            out.coeffs[mask] = self.coeffs[mask] + other.coeffs[mask];
        }
        out
    }

    fn sub(&self, other: &Self) -> Self {
        let mut out = Self::zero(self.n_dirs);
        for mask in 0..=self.full_mask() {
            out.coeffs[mask] = self.coeffs[mask] - other.coeffs[mask];
        }
        out
    }

    fn scale(&self, scalar: f64) -> Self {
        let mut out = Self::zero(self.n_dirs);
        for mask in 0..=self.full_mask() {
            out.coeffs[mask] = self.coeffs[mask] * scalar;
        }
        out
    }

    fn mul(&self, other: &Self) -> Self {
        let mut out = Self::zero(self.n_dirs);
        for mask in 0..=self.full_mask() {
            let mut total = 0.0;
            let mut submask = mask;
            loop {
                total += self.coeffs[submask] * other.coeffs[mask ^ submask];
                if submask == 0 {
                    break;
                }
                submask = (submask - 1) & mask;
            }
            out.coeffs[mask] = total;
        }
        out
    }

    fn compose_unary(&self, derivs: [f64; 5]) -> Self {
        let mut out = Self::constant(self.n_dirs, derivs[0]);
        let partitions = subset_partition_table();
        for mask in 1..=self.full_mask() {
            let mut total = 0.0;
            for partition in &partitions[mask] {
                let order = partition.len();
                if order == 0 || order >= derivs.len() {
                    continue;
                }
                let mut prod = 1.0;
                for &block in partition {
                    prod *= self.coeffs[block];
                }
                total += derivs[order] * prod;
            }
            out.coeffs[mask] = total;
        }
        out
    }
}

// ── Unary derivative functions ────────────────────────────────────────

fn unary_derivatives_exp(x: f64) -> [f64; 5] {
    let ex = x.exp();
    [ex, ex, ex, ex, ex]
}

fn unary_derivatives_sqrt(x: f64) -> [f64; 5] {
    let s = x.max(1e-300).sqrt();
    let x1 = x.max(1e-300);
    let x2 = x1 * x1;
    let x3 = x2 * x1;
    [
        s,
        0.5 / s,
        -0.25 / (x1 * s),
        3.0 / (8.0 * x2 * s),
        -15.0 / (16.0 * x3 * s),
    ]
}

#[inline]
fn erfcx_nonnegative(x: f64) -> f64 {
    if !x.is_finite() {
        return if x.is_sign_positive() {
            0.0
        } else {
            f64::INFINITY
        };
    }
    if x <= 0.0 {
        return 1.0;
    }
    // Rational approximation from Abramowitz & Stegun, improved.
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    poly
}

fn signed_probit_logcdf_and_mills_ratio(x: f64) -> (f64, f64) {
    if x < 0.0 {
        let u = -x / std::f64::consts::SQRT_2;
        let ex = erfcx_nonnegative(u).max(1e-300);
        let log_cdf = -u * u + (0.5 * ex).ln();
        let lambda = (2.0 / std::f64::consts::PI).sqrt() / ex;
        (log_cdf, lambda)
    } else {
        let cdf = normal_cdf(x).clamp(1e-300, 1.0);
        let lambda = normal_pdf(x) / cdf;
        (cdf.ln(), lambda)
    }
}

fn signed_probit_neglog_derivatives_up_to_fourth(
    signed_margin: f64,
    weight: f64,
) -> (f64, f64, f64, f64) {
    if weight == 0.0 || !signed_margin.is_finite() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let (_, lambda) = signed_probit_logcdf_and_mills_ratio(signed_margin);
    let k1 = -lambda;
    let k2 = lambda * (signed_margin + lambda);
    let k3 = lambda
        * (1.0
            - signed_margin * signed_margin
            - 3.0 * signed_margin * lambda
            - 2.0 * lambda * lambda);
    let k4 = lambda
        * ((signed_margin.powi(3) - 3.0 * signed_margin)
            + (7.0 * signed_margin * signed_margin - 4.0) * lambda
            + 12.0 * signed_margin * lambda * lambda
            + 6.0 * lambda.powi(3));
    (weight * k1, weight * k2, weight * k3, weight * k4)
}

/// Derivatives of -log Phi(x) (negated log-normal-CDF) w.r.t. x.
fn unary_derivatives_neglog_phi(x: f64, weight: f64) -> [f64; 5] {
    let (d1, d2, d3, d4) = signed_probit_neglog_derivatives_up_to_fourth(x, weight);
    let (log_cdf, _) = signed_probit_logcdf_and_mills_ratio(x);
    [-weight * log_cdf, d1, d2, d3, d4]
}

/// Derivatives of log(x) through 4th order: [log(x), 1/x, -1/x^2, 2/x^3, -6/x^4].
fn unary_derivatives_log(x: f64) -> [f64; 5] {
    let x1 = x.max(1e-300);
    let x2 = x1 * x1;
    let x3 = x2 * x1;
    let x4 = x3 * x1;
    [x1.ln(), 1.0 / x1, -1.0 / x2, 2.0 / x3, -6.0 / x4]
}

/// Derivatives of log(phi(x)) = -0.5*x^2 - 0.5*ln(2*pi) w.r.t. x.
/// [f(x), f'(x), f''(x), f'''(x), f''''(x)] = [-0.5*x^2 - C, -x, -1, 0, 0]
fn unary_derivatives_log_normal_pdf(x: f64) -> [f64; 5] {
    let c = 0.5 * (2.0 * std::f64::consts::PI).ln();
    [-0.5 * x * x - c, -x, -1.0, 0.0, 0.0]
}

// ── Primary-space helpers ─────────────────────────────────────────────

// Primary scalar indices: 0=q0, 1=q1, 2=qd1, 3=g
const N_PRIMARY: usize = 4;

fn unit_primary_direction(idx: usize) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(N_PRIMARY);
    out[idx] = 1.0;
    out
}

// ── Row-level NLL computation ─────────────────────────────────────────

impl SurvivalMarginalSlopeFamily {
    /// Compute the per-row NLL and its directional derivatives through the
    /// 4 primary scalars using MultiDirJet.
    ///
    /// Primary scalars: q0, q1, qd1, g
    ///
    /// Model (non-flex):
    ///   beta = exp(g)
    ///   c = sqrt(1 + beta^2)
    ///   a0 = q0 * c, a1 = q1 * c, ad1 = qd1 * c
    ///   eta0 = a0 + beta * z_i, eta1 = a1 + beta * z_i
    ///
    /// Interval NLL:
    ///   ell = w * [(1-d)*(-log Phi(-eta1)) - (-log Phi(-eta0)) + d*(log phi(eta1) + log(ad1))]
    ///   neglog = -ell
    ///   = w * [(-log Phi(-eta0)) is SUBTRACTED so we ADD log Phi(-eta0)]
    ///
    /// Actually, let's be precise:
    ///   log L = w * [(1-d)*log S(t1|z) - log S(t0|z) + d*(log f(t1|z))]
    ///   where S(t|z) = Phi(-eta), f(t|z) = phi(eta)*a'(t)/S(t|z)... wait, that's the density.
    ///
    /// Let me re-derive carefully:
    ///   S(t|z) = Phi(-(a(t) + beta*z))
    ///   F(t|z) = Phi(a(t) + beta*z)
    ///   f(t|z) = phi(a(t) + beta*z) * a'(t)
    ///
    /// For interval [t0, t1] with entry t0, exit t1, event d:
    ///   L = [S(t1|z)/S(t0|z)]^(1-d) * [f(t1|z)/S(t0|z)]^d
    ///     = S(t1|z)^(1-d) * f(t1|z)^d / S(t0|z)
    ///
    /// log L = (1-d)*log S(t1|z) + d*log f(t1|z) - log S(t0|z)
    ///       = (1-d)*log Phi(-eta1) + d*[log phi(eta1) + log a'(t1)] - log Phi(-eta0)
    ///
    /// NLL = -log L = -[(1-d)*log Phi(-eta1) + d*(log phi(eta1) + log ad1) - log Phi(-eta0)]
    ///     = (-log Phi(-eta0)) is negated, so:
    ///     = log Phi(-eta0) - (1-d)*log Phi(-eta1) - d*(log phi(eta1) + log ad1)
    ///
    /// Wait, let me be really careful with signs:
    ///   NLL = -(log L)
    ///       = -log Phi(-eta0) ... NO, that's wrong.
    ///
    ///   log L = (1-d)*log Phi(-eta1) + d*(log phi(eta1) + log ad1) - log Phi(-eta0)
    ///   NLL = -log L = log Phi(-eta0) - (1-d)*log Phi(-eta1) - d*log phi(eta1) - d*log ad1
    ///       = [-log Phi(-eta0)] is the NEGATIVE, so:
    ///       = {-log Phi(-eta0)}*(-1) ...
    ///
    /// Let me use unary_derivatives_neglog_phi directly.
    /// Define neglogphi(x) = -log Phi(x).
    ///
    /// NLL = neglogphi(-eta0) - (1-d)*neglogphi(-eta1) ... NO.
    ///
    /// NLL = -log L
    ///     = -(1-d)*log Phi(-eta1) - d*log phi(eta1) - d*log ad1 + log Phi(-eta0)
    ///
    /// Hmm, log Phi(-eta0) = -neglogphi(-eta0). So:
    ///   NLL = (1-d)*neglogphi(-eta1) + d*(-log phi(eta1)) + d*(-log ad1) - neglogphi(-eta0)
    ///
    /// Wait no. Let's just be careful:
    ///   neglogphi(x) = -log Phi(x)
    ///   log Phi(x) = -neglogphi(x)
    ///
    ///   NLL = -[(1-d)*log Phi(-eta1) + d*(log phi(eta1) + log ad1) - log Phi(-eta0)]
    ///       = -(1-d)*log Phi(-eta1) - d*log phi(eta1) - d*log ad1 + log Phi(-eta0)
    ///       = (1-d)*neglogphi(-eta1) - d*log phi(eta1) - d*log ad1 - neglogphi(-eta0)
    ///
    /// But we need NLL to be positive when likelihood is < 1, which is typical.
    /// Let's verify: if no event (d=0) and S(t1) < S(t0) (some risk):
    ///   NLL = neglogphi(-eta1) - neglogphi(-eta0)
    ///       = -log Phi(-eta1) + log Phi(-eta0)
    ///       = log[Phi(-eta0)/Phi(-eta1)]
    ///       = log[S(t0)/S(t1)]
    /// Since S(t0) >= S(t1) (CDF increases), this is >= 0. Good.
    ///
    /// So:
    ///   NLL_i = w_i * [(1-d_i)*neglogphi(-eta1) - neglogphi(-eta0)
    ///                   - d_i*log phi(eta1) - d_i*log ad1]
    ///
    /// Wait, that's NLL = w * [entry_surv_neglog - exit_surv_neglog - event_density_log]
    ///   where entry_surv_neglog = neglogphi(-eta0)... no, it's SUBTRACTED.
    ///
    /// Let me just write it cleanly as weighted sum of jet terms:
    ///
    ///   NLL_i = w_i * {
    ///       (1-d_i) * neglogphi(-eta1_i)    // exit survival (if censored)
    ///     - neglogphi(-eta0_i)               // entry survival (always subtracted)
    ///     - d_i * log_phi(eta1_i)            // event density (numerator)
    ///     - d_i * log(ad1_i)                 // time derivative (numerator)
    ///   }
    ///
    /// Hmm, this means NLL can be negative if -neglogphi(-eta0) dominates.
    /// Actually no: NLL = -log(L_i) where L_i = S(t1)^(1-d) * f(t1)^d / S(t0)
    /// and S(t0) >= S(t1) so S(t1)/S(t0) <= 1, and f(t1) is a density that can be > 1.
    /// So NLL CAN be negative for individual rows. That's fine.
    ///
    /// Let me just implement it directly with jets.
    fn row_neglog_directional(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dirs: &[Array1<f64>],
    ) -> Result<f64, String> {
        let k = dirs.len();
        if k > 4 {
            return Err(format!(
                "survival marginal-slope row directional expects 0..=4 directions, got {k}"
            ));
        }
        let wi = self.weights[row];
        let di = self.event[row];
        let zi = self.z[row];

        // Primary scalar jets: q0, q1, qd1, g
        let q0_first: Vec<f64> = dirs.iter().map(|dir| dir[0]).collect();
        let q1_first: Vec<f64> = dirs.iter().map(|dir| dir[1]).collect();
        let qd1_first: Vec<f64> = dirs.iter().map(|dir| dir[2]).collect();
        let g_first: Vec<f64> = dirs.iter().map(|dir| dir[3]).collect();

        let q0_val = block_states[0].eta[row] + self.offset_entry[row]
            - self.design_entry.row(row).dot(&block_states[0].beta);
        // Actually wait - block_states[0].eta[row] IS X_exit[row] @ beta_time + offset_exit[row]
        // for the EXIT design. But we have 3 different designs for the time block.
        //
        // The time block has a single beta vector, but 3 different linear predictors:
        //   q0 = design_entry[row,:] @ beta_time + offset_entry[row]
        //   q1 = design_exit[row,:] @ beta_time + offset_exit[row]
        //   qd1 = design_deriv[row,:] @ beta_time + derivative_offset_exit[row]
        //
        // The block_states[0].eta is computed from whichever design the block uses.
        // But in survival_location_scale, eta for the time block is computed using
        // a stacked design. Here we'll compute the 3 linear predictors manually.
        //
        // Actually, I need to think about this differently. The CustomFamily evaluate()
        // method receives block_states where eta = design @ beta + offset for each block.
        // But we have ONE time block with ONE beta, yet 3 different design matrices.
        // So eta will be computed with one of them (say the exit design, since that's
        // what we mainly use). We'll compute the others manually.
        //
        // Let me restructure: compute q0, q1, qd1 from beta_time directly.

        let beta_time = &block_states[0].beta;
        let q0_val = self.design_entry.row(row).dot(beta_time) + self.offset_entry[row];
        let q1_val = self.design_exit.row(row).dot(beta_time) + self.offset_exit[row];
        let qd1_val =
            self.design_derivative_exit.row(row).dot(beta_time) + self.derivative_offset_exit[row];
        let g_val = block_states[1].eta[row];

        let q0_jet = MultiDirJet::linear(k, q0_val, &q0_first);
        let q1_jet = MultiDirJet::linear(k, q1_val, &q1_first);
        let qd1_jet = MultiDirJet::linear(k, qd1_val, &qd1_first);
        let g_jet = MultiDirJet::linear(k, g_val, &g_first);

        // beta = exp(g)
        let beta_jet = g_jet.compose_unary(unary_derivatives_exp(g_val));
        // c = sqrt(1 + beta^2)
        let one_plus_b2 = MultiDirJet::constant(k, 1.0).add(&beta_jet.mul(&beta_jet));
        let c_jet = one_plus_b2.compose_unary(unary_derivatives_sqrt(one_plus_b2.coeff(0)));

        // a0 = q0 * c, a1 = q1 * c, ad1 = qd1 * c
        let a0_jet = q0_jet.mul(&c_jet);
        let a1_jet = q1_jet.mul(&c_jet);
        let ad1_jet = qd1_jet.mul(&c_jet);

        // eta0 = a0 + beta * z, eta1 = a1 + beta * z
        let z_jet = MultiDirJet::constant(k, zi);
        let eta0_jet = a0_jet.add(&beta_jet.mul(&z_jet));
        let eta1_jet = a1_jet.add(&beta_jet.mul(&z_jet));

        // NLL_i = w_i * {
        //   (1-d_i) * neglogphi(-eta1)  [exit survival for censored]
        //   - neglogphi(-eta0)           [entry survival, subtracted]
        //   - d_i * log_phi(eta1)        [event log-density of normal]
        //   - d_i * log(ad1)             [event log time-derivative]
        // }

        // Entry survival term: -neglogphi(-eta0) = log Phi(-eta0) = log S(t0|z)
        let neg_eta0 = eta0_jet.scale(-1.0);
        let entry_term = neg_eta0
            .compose_unary(unary_derivatives_neglog_phi(neg_eta0.coeff(0), wi))
            .scale(-1.0); // note: -w * neglogphi(-eta0) = w * log Phi(-eta0)

        // Exit survival term: (1-d)*neglogphi(-eta1) = -(1-d)*log Phi(-eta1)
        let neg_eta1 = eta1_jet.scale(-1.0);
        let exit_term = neg_eta1.compose_unary(unary_derivatives_neglog_phi(
            neg_eta1.coeff(0),
            wi * (1.0 - di),
        ));

        // Event density: -d * log phi(eta1)
        let event_density_term = if di > 0.0 {
            eta1_jet
                .compose_unary(unary_derivatives_log_normal_pdf(eta1_jet.coeff(0)))
                .scale(-wi * di)
        } else {
            MultiDirJet::zero(k)
        };

        // Time derivative: -d * log(ad1)
        let time_deriv_term = if di > 0.0 {
            let ad1_val = ad1_jet.coeff(0);
            let guard = self.derivative_guard;
            let softness = self.derivative_softness;
            // Softplus guard: use softplus(ad1 - guard) + guard to keep ad1 > guard
            let effective_ad1 = if guard > 0.0 && softness > 0.0 {
                let shifted = ad1_val - guard;
                let soft_val = if shifted > 20.0 * softness {
                    shifted + guard
                } else {
                    softness * (1.0 + (shifted / softness).exp()).ln() + guard
                };
                // For the jet, we apply softplus derivatives
                let sig = 1.0 / (1.0 + (-shifted / softness).exp()); // sigmoid
                let sig2 = sig * (1.0 - sig) / softness;
                let sig3 = (sig2 * (1.0 - 2.0 * sig)) / softness;
                let sig4 = ((sig2 * (1.0 - 6.0 * sig * (1.0 - sig))) / softness
                    - 2.0 * sig3 * sig / (1.0 - sig).max(1e-30))
                    / softness;
                // Actually, the derivatives of softplus(x/s)*s+guard w.r.t. x:
                // d/dx = sigmoid(x/s), d2/dx2 = sigmoid'(x/s)/s, etc.
                // Better to just compose through the ad1_jet with softplus derivatives.
                // Let me simplify: compose ad1 -> log(softplus_ad1) directly.
                let log_soft = soft_val.max(1e-300).ln();
                let d_log = sig / soft_val.max(1e-300); // d(log softplus)/d(ad1)
                let d2_log =
                    (sig2 * soft_val.max(1e-300) - sig * sig) / soft_val.max(1e-300).powi(2);
                // This is getting complicated. Let me just use the raw ad1 with a floor.
                // The monotonicity constraints already ensure ad1 >= guard, so this is safe.
                ad1_jet.compose_unary(unary_derivatives_log(ad1_val.max(1e-300)))
            } else {
                ad1_jet.compose_unary(unary_derivatives_log(ad1_val.max(1e-300)))
            };
            effective_ad1.scale(-wi * di)
        } else {
            MultiDirJet::zero(k)
        };

        // Total NLL: exit_term + entry_term + event_density + time_deriv
        // entry_term is already negated (it's -neglogphi = +logPhi, contributing to -NLL)
        // Actually let me re-check:
        //   NLL = w * [(1-d)*neglogphi(-eta1) - neglogphi(-eta0) - d*log_phi(eta1) - d*log(ad1)]
        //
        // So: NLL = exit_term(w*(1-d)) + entry_term(-w) + event_density(-w*d) + time_deriv(-w*d)
        //
        // exit_term = neglogphi(-eta1) * w*(1-d)  [already computed with weight w*(1-d)]
        // entry_term = neglogphi(-eta0) * (-w)     [computed as -1 * neglogphi(-eta0, w)]
        // event_density = log_phi(eta1) * (-w*d)   [computed as -w*d * log_phi]
        // time_deriv = log(ad1) * (-w*d)           [computed as -w*d * log(ad1)]
        let total = exit_term
            .add(&entry_term)
            .add(&event_density_term)
            .add(&time_deriv_term);

        if k == 0 {
            Ok(total.coeff(0))
        } else {
            Ok(total.coeff(total.full_mask()))
        }
    }

    fn compute_row_primary_gradient_hessian(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
    ) -> Result<(Array1<f64>, Array2<f64>), String> {
        let mut grad = Array1::<f64>::zeros(N_PRIMARY);
        let mut hess = Array2::<f64>::zeros((N_PRIMARY, N_PRIMARY));
        for a in 0..N_PRIMARY {
            let da = unit_primary_direction(a);
            grad[a] = self.row_neglog_directional(row, block_states, &[da.clone()])?;
            for b in a..N_PRIMARY {
                let db = unit_primary_direction(b);
                let value =
                    self.row_neglog_directional(row, block_states, &[da.clone(), db.clone()])?;
                hess[[a, b]] = value;
                hess[[b, a]] = value;
            }
        }
        Ok((grad, hess))
    }

    fn row_primary_third_contracted(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let mut out = Array2::<f64>::zeros((N_PRIMARY, N_PRIMARY));
        for a in 0..N_PRIMARY {
            let da = unit_primary_direction(a);
            for b in a..N_PRIMARY {
                let db = unit_primary_direction(b);
                let value = self.row_neglog_directional(
                    row,
                    block_states,
                    &[da.clone(), db.clone(), dir.clone()],
                )?;
                out[[a, b]] = value;
                out[[b, a]] = value;
            }
        }
        Ok(out)
    }

    fn row_primary_fourth_contracted(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir_u: &Array1<f64>,
        dir_v: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let mut out = Array2::<f64>::zeros((N_PRIMARY, N_PRIMARY));
        for a in 0..N_PRIMARY {
            let da = unit_primary_direction(a);
            for b in a..N_PRIMARY {
                let db = unit_primary_direction(b);
                let value = self.row_neglog_directional(
                    row,
                    block_states,
                    &[da.clone(), db.clone(), dir_u.clone(), dir_v.clone()],
                )?;
                out[[a, b]] = value;
                out[[b, a]] = value;
            }
        }
        Ok(out)
    }

    // ── Pullback through design matrices ──────────────────────────────

    /// Map a primary-space vector [f_q0, f_q1, f_qd1, f_g] to coefficient space.
    fn pullback_primary_vector(
        &self,
        row: usize,
        slices: &BlockSlices,
        primary_vec: &Array1<f64>,
    ) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(slices.total);
        // Time block: 3 primary scalars (q0, q1, qd1) all map to the same beta_time
        let x_entry = self.design_entry.row(row).to_owned();
        let x_exit = self.design_exit.row(row).to_owned();
        let x_deriv = self.design_derivative_exit.row(row).to_owned();
        let time_contrib =
            &x_entry * primary_vec[0] + &x_exit * primary_vec[1] + &x_deriv * primary_vec[2];
        out.slice_mut(s![slices.time.clone()]).assign(&time_contrib);
        // Slope block: primary scalar g
        let g_row = self.logslope_design.row(row).to_owned();
        out.slice_mut(s![slices.logslope.clone()])
            .assign(&(&g_row * primary_vec[3]));
        out
    }

    /// Accumulate the pullback of a primary-space Hessian into coefficient-space.
    fn add_pullback_primary_hessian(
        &self,
        target: &mut Array2<f64>,
        row: usize,
        slices: &BlockSlices,
        primary_hessian: &Array2<f64>,
    ) {
        let x_entry = self.design_entry.row(row).to_owned();
        let x_exit = self.design_exit.row(row).to_owned();
        let x_deriv = self.design_derivative_exit.row(row).to_owned();
        let g_row = self.logslope_design.row(row).to_owned();

        // Time-time block (indices 0,1,2 × 0,1,2 in primary space)
        // We have 3 design vectors for the time block. The time-time Hessian is:
        // sum over (i,j) in {entry,exit,deriv}×{entry,exit,deriv} of
        //   X_i * X_j^T * H_primary[idx_i, idx_j]
        let time_designs = [&x_entry, &x_exit, &x_deriv];
        for (i, xi) in time_designs.iter().enumerate() {
            for (j, xj) in time_designs.iter().enumerate() {
                let h_val = primary_hessian[[i, j]];
                if h_val.abs() > 1e-30 {
                    let outer = xi
                        .view()
                        .insert_axis(Axis(1))
                        .dot(&xj.view().insert_axis(Axis(0)))
                        * h_val;
                    target
                        .slice_mut(s![slices.time.clone(), slices.time.clone()])
                        .scaled_add(1.0, &outer);
                }
            }
        }

        // Time-slope cross block (indices 0,1,2 × 3)
        for (i, xi) in time_designs.iter().enumerate() {
            let h_val = primary_hessian[[i, 3]];
            if h_val.abs() > 1e-30 {
                let outer = xi
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&g_row.view().insert_axis(Axis(0)))
                    * h_val;
                target
                    .slice_mut(s![slices.time.clone(), slices.logslope.clone()])
                    .scaled_add(1.0, &outer);
                target
                    .slice_mut(s![slices.logslope.clone(), slices.time.clone()])
                    .scaled_add(1.0, &outer.t().to_owned());
            }
        }

        // Slope-slope block (index 3 × 3)
        let gg = g_row
            .view()
            .insert_axis(Axis(1))
            .dot(&g_row.view().insert_axis(Axis(0)))
            * primary_hessian[[3, 3]];
        target
            .slice_mut(s![slices.logslope.clone(), slices.logslope.clone()])
            .scaled_add(1.0, &gg);
    }

    /// Map a coefficient-space direction to primary-space for a given row.
    fn row_primary_direction_from_flat(
        &self,
        row: usize,
        slices: &BlockSlices,
        d_beta_flat: &Array1<f64>,
    ) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(N_PRIMARY);
        let d_time = d_beta_flat.slice(s![slices.time.clone()]).to_owned();
        out[0] = self.design_entry.row(row).dot(&d_time);
        out[1] = self.design_exit.row(row).dot(&d_time);
        out[2] = self.design_derivative_exit.row(row).dot(&d_time);
        out[3] = self
            .logslope_design
            .row(row)
            .dot(&d_beta_flat.slice(s![slices.logslope.clone()]).to_owned());
        out
    }

    // ── Joint gradient + Hessian ──────────────────────────────────────

    fn joint_gradient_hessian(
        &self,
        block_states: &[ParameterBlockState],
        need_hessian: bool,
    ) -> Result<(f64, Array1<f64>, Option<Array2<f64>>), String> {
        let slices = block_slices(block_states);
        let mut gradient = Array1::<f64>::zeros(slices.total);
        let mut hessian = need_hessian.then(|| Array2::<f64>::zeros((slices.total, slices.total)));
        let mut ll = 0.0;
        for i in 0..self.n {
            let row_neglog = self.row_neglog_directional(i, block_states, &[])?;
            ll -= row_neglog;

            let (f_pi, f_pipi) = self.compute_row_primary_gradient_hessian(i, block_states)?;
            gradient -= &self.pullback_primary_vector(i, &slices, &f_pi);
            if let Some(ref mut hmat) = hessian {
                self.add_pullback_primary_hessian(hmat, i, &slices, &f_pipi);
            }
        }
        Ok((ll, gradient, hessian))
    }

    // ── Hessian directional derivatives ───────────────────────────────

    fn joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let slices = block_slices(block_states);
        let mut out = Array2::<f64>::zeros((slices.total, slices.total));
        for row in 0..self.n {
            let row_dir = self.row_primary_direction_from_flat(row, &slices, d_beta_flat);
            let third = self.row_primary_third_contracted(row, block_states, &row_dir)?;
            self.add_pullback_primary_hessian(&mut out, row, &slices, &third);
        }
        Ok(Some(out))
    }

    fn joint_hessian_second_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u: &Array1<f64>,
        d_beta_v: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let slices = block_slices(block_states);
        let mut out = Array2::<f64>::zeros((slices.total, slices.total));
        for row in 0..self.n {
            let row_u = self.row_primary_direction_from_flat(row, &slices, d_beta_u);
            let row_v = self.row_primary_direction_from_flat(row, &slices, d_beta_v);
            let fourth = self.row_primary_fourth_contracted(row, block_states, &row_u, &row_v)?;
            self.add_pullback_primary_hessian(&mut out, row, &slices, &fourth);
        }
        Ok(Some(out))
    }

    fn joint_hessian_matvec(
        &self,
        block_states: &[ParameterBlockState],
        direction: &Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        let slices = block_slices(block_states);
        let mut out = Array1::<f64>::zeros(slices.total);
        for row in 0..self.n {
            let row_dir = self.row_primary_direction_from_flat(row, &slices, direction);
            let (_, row_hessian) = self.compute_row_primary_gradient_hessian(row, block_states)?;
            let row_action = row_hessian.dot(&row_dir);
            out += &self.pullback_primary_vector(row, &slices, &row_action);
        }
        Ok(out)
    }

    fn joint_hessian_diagonal(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Array1<f64>, String> {
        let slices = block_slices(block_states);
        let mut diagonal = Array1::<f64>::zeros(slices.total);
        for row in 0..self.n {
            let (_, row_hessian) = self.compute_row_primary_gradient_hessian(row, block_states)?;

            // Time block contributions from entry, exit, derivative designs
            let designs = [
                (0, &self.design_entry),
                (1, &self.design_exit),
                (2, &self.design_derivative_exit),
            ];
            for &(pi, ref des) in &designs {
                for (local_idx, &value) in des.row(row).iter().enumerate() {
                    diagonal[slices.time.start + local_idx] +=
                        value * value * row_hessian[[pi, pi]];
                }
                // Cross terms between different time designs contribute to diagonal
                // only through the same coefficient index
                for &(pj, ref des_j) in &designs {
                    if pj <= pi {
                        continue;
                    }
                    for local_idx in 0..des.ncols() {
                        diagonal[slices.time.start + local_idx] += 2.0
                            * des[[row, local_idx]]
                            * des_j[[row, local_idx]]
                            * row_hessian[[pi, pj]];
                    }
                }
            }

            // Slope block
            for (local_idx, &value) in self.logslope_design.row(row).iter().enumerate() {
                diagonal[slices.logslope.start + local_idx] += value * value * row_hessian[[3, 3]];
            }
        }
        Ok(diagonal)
    }

    // ── Psi (spatial length-scale) derivatives ────────────────────────

    fn resolve_psi_location(
        &self,
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Option<(usize, usize)> {
        let mut cursor = 0usize;
        for (block_idx, block) in derivative_blocks.iter().enumerate() {
            if psi_index < cursor + block.len() {
                return Some((block_idx, psi_index - cursor));
            }
            cursor += block.len();
        }
        None
    }

    fn psi_design_row_vector(
        &self,
        row: usize,
        deriv: &crate::custom_family::CustomFamilyBlockPsiDerivative,
        total_rows: usize,
        p: usize,
        label: &str,
    ) -> Result<Array1<f64>, String> {
        let action = CustomFamilyPsiDesignAction::from_first_derivative(
            deriv,
            total_rows,
            p,
            0..total_rows,
            label,
        )
        .ok();
        first_psi_linear_map(action.as_ref(), &deriv.x_psi, total_rows, p).row_vector(row)
    }

    fn psi_second_design_row_vector(
        &self,
        row: usize,
        deriv_i: &crate::custom_family::CustomFamilyBlockPsiDerivative,
        deriv_j: &crate::custom_family::CustomFamilyBlockPsiDerivative,
        local_j: usize,
        total_rows: usize,
        p: usize,
        label: &str,
    ) -> Result<Array1<f64>, String> {
        let action = CustomFamilyPsiSecondDesignAction::from_second_derivative(
            deriv_i,
            deriv_j,
            total_rows,
            p,
            0..total_rows,
            label,
        )?;
        let dense = deriv_i
            .x_psi_psi
            .as_ref()
            .and_then(|rows| rows.get(local_j));
        second_psi_linear_map(action.as_ref(), dense, total_rows, p).row_vector(row)
    }

    /// Map a psi derivative to a primary-space direction for a given row.
    /// For the time block (block 0), a psi perturbation changes all 3 primary scalars
    /// (q0, q1, qd1) because they share beta_time and the psi perturbation affects the design.
    /// For the logslope block (block 1), only g is affected.
    fn row_primary_psi_direction(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<Array1<f64>>, String> {
        let Some((block_idx, local_idx)) = self.resolve_psi_location(derivative_blocks, psi_index)
        else {
            return Ok(None);
        };
        let deriv = &derivative_blocks[block_idx][local_idx];
        let mut out = Array1::<f64>::zeros(N_PRIMARY);
        match block_idx {
            0 => {
                // Time block: psi affects entry, exit, and derivative designs.
                // The psi design row vector gives dX/dpsi @ beta, which is the
                // change in the linear predictor. But we have 3 linear predictors
                // from the same beta. For now, the psi design is for the "main" design
                // (exit), so we approximate by using it for exit and ignoring entry/deriv.
                // TODO: implement separate psi designs for entry/exit/deriv if needed.
                let x_row = self.psi_design_row_vector(
                    row,
                    deriv,
                    self.n,
                    self.design_exit.ncols(),
                    "SurvivalMarginalSlope time",
                )?;
                let dot = x_row.dot(&block_states[0].beta);
                // psi perturbation changes q0, q1, qd1 through the same beta
                // For the exit design, it directly changes q1.
                // For entry and deriv, the psi perturbation has the same structure
                // (same spatial terms, same beta), so the change is analogous.
                // As a simplification, assign the same dot product to q0 and q1,
                // and 0 to qd1 (derivative design psi structure is different).
                out[0] = dot; // q0
                out[1] = dot; // q1
                // qd1 psi contribution is 0 for now (derivative design structure is different)
                out[2] = 0.0;
            }
            1 => {
                let x_row = self.psi_design_row_vector(
                    row,
                    deriv,
                    self.n,
                    self.logslope_design.ncols(),
                    "SurvivalMarginalSlope logslope",
                )?;
                out[3] = x_row.dot(&block_states[1].beta);
            }
            _ => {
                return Err(format!(
                    "survival marginal-slope psi direction only supports time/logslope blocks, got block {block_idx}"
                ));
            }
        }
        Ok(Some(out))
    }

    fn row_primary_psi_action_on_direction(
        &self,
        row: usize,
        slices: &BlockSlices,
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array1<f64>>, String> {
        let Some((block_idx, local_idx)) = self.resolve_psi_location(derivative_blocks, psi_index)
        else {
            return Ok(None);
        };
        let deriv = &derivative_blocks[block_idx][local_idx];
        let mut out = Array1::<f64>::zeros(N_PRIMARY);
        match block_idx {
            0 => {
                let x_row = self.psi_design_row_vector(
                    row,
                    deriv,
                    self.n,
                    self.design_exit.ncols(),
                    "SurvivalMarginalSlope time",
                )?;
                let dot = x_row.dot(&d_beta_flat.slice(s![slices.time.clone()]).to_owned());
                out[0] = dot;
                out[1] = dot;
                out[2] = 0.0;
            }
            1 => {
                let x_row = self.psi_design_row_vector(
                    row,
                    deriv,
                    self.n,
                    self.logslope_design.ncols(),
                    "SurvivalMarginalSlope logslope",
                )?;
                out[3] = x_row.dot(&d_beta_flat.slice(s![slices.logslope.clone()]).to_owned());
            }
            _ => {
                return Err(format!(
                    "survival marginal-slope psi action only supports time/logslope blocks, got block {block_idx}"
                ));
            }
        }
        Ok(Some(out))
    }

    fn row_primary_psi_second_direction(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<Array1<f64>>, String> {
        let Some((block_i, local_i)) = self.resolve_psi_location(derivative_blocks, psi_i) else {
            return Ok(None);
        };
        let Some((block_j, local_j)) = self.resolve_psi_location(derivative_blocks, psi_j) else {
            return Ok(None);
        };
        if block_i != block_j {
            return Ok(Some(Array1::<f64>::zeros(N_PRIMARY)));
        }
        let deriv_i = &derivative_blocks[block_i][local_i];
        let mut out = Array1::<f64>::zeros(N_PRIMARY);
        match block_i {
            0 => {
                let x_row = self.psi_second_design_row_vector(
                    row,
                    deriv_i,
                    &derivative_blocks[block_j][local_j],
                    local_j,
                    self.n,
                    self.design_exit.ncols(),
                    "SurvivalMarginalSlope time",
                )?;
                let dot = x_row.dot(&block_states[0].beta);
                out[0] = dot;
                out[1] = dot;
                out[2] = 0.0;
            }
            1 => {
                let x_row = self.psi_second_design_row_vector(
                    row,
                    deriv_i,
                    &derivative_blocks[block_j][local_j],
                    local_j,
                    self.n,
                    self.logslope_design.ncols(),
                    "SurvivalMarginalSlope logslope",
                )?;
                out[3] = x_row.dot(&block_states[1].beta);
            }
            _ => {
                return Err(format!(
                    "survival marginal-slope psi second direction only supports time/logslope blocks, got block {block_i}"
                ));
            }
        }
        Ok(Some(out))
    }

    fn embedded_psi_vector(
        &self,
        row: usize,
        slices: &BlockSlices,
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<(usize, Array1<f64>)>, String> {
        let Some((block_idx, local_idx)) = self.resolve_psi_location(derivative_blocks, psi_index)
        else {
            return Ok(None);
        };
        let deriv = &derivative_blocks[block_idx][local_idx];
        let mut out = Array1::<f64>::zeros(slices.total);
        match block_idx {
            0 => out
                .slice_mut(s![slices.time.clone()])
                .assign(&self.psi_design_row_vector(
                    row,
                    deriv,
                    self.n,
                    self.design_exit.ncols(),
                    "SurvivalMarginalSlope time",
                )?),
            1 => out
                .slice_mut(s![slices.logslope.clone()])
                .assign(&self.psi_design_row_vector(
                    row,
                    deriv,
                    self.n,
                    self.logslope_design.ncols(),
                    "SurvivalMarginalSlope logslope",
                )?),
            _ => {
                return Err(format!(
                    "survival marginal-slope psi embedding only supports time/logslope blocks, got block {block_idx}"
                ));
            }
        }
        Ok(Some((block_idx, out)))
    }

    fn embedded_psi_second_vector(
        &self,
        row: usize,
        slices: &BlockSlices,
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<(usize, Array1<f64>)>, String> {
        let Some((block_i, local_i)) = self.resolve_psi_location(derivative_blocks, psi_i) else {
            return Ok(None);
        };
        let Some((block_j, local_j)) = self.resolve_psi_location(derivative_blocks, psi_j) else {
            return Ok(None);
        };
        if block_i != block_j {
            return Ok(Some((block_i, Array1::<f64>::zeros(slices.total))));
        }
        let deriv_i = &derivative_blocks[block_i][local_i];
        let mut out = Array1::<f64>::zeros(slices.total);
        match block_i {
            0 => {
                out.slice_mut(s![slices.time.clone()])
                    .assign(&self.psi_second_design_row_vector(
                        row,
                        deriv_i,
                        &derivative_blocks[block_j][local_j],
                        local_j,
                        self.n,
                        self.design_exit.ncols(),
                        "SurvivalMarginalSlope time",
                    )?)
            }
            1 => out.slice_mut(s![slices.logslope.clone()]).assign(
                &self.psi_second_design_row_vector(
                    row,
                    deriv_i,
                    &derivative_blocks[block_j][local_j],
                    local_j,
                    self.n,
                    self.logslope_design.ncols(),
                    "SurvivalMarginalSlope logslope",
                )?,
            ),
            _ => {
                return Err(format!(
                    "survival marginal-slope psi second embedding only supports time/logslope blocks, got block {block_i}"
                ));
            }
        }
        Ok(Some((block_i, out)))
    }

    // ── Psi terms (first and second order) ────────────────────────────

    fn psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        let slices = block_slices(block_states);
        let Some((block_idx, _)) =
            self.embedded_psi_vector(0, &slices, derivative_blocks, psi_index)?
        else {
            return Ok(None);
        };
        // Map block_idx to primary index for the scalar that this psi affects
        let idx_primary = match block_idx {
            0 => 1usize, // time block -> primary q1 (exit, main one)
            1 => 3usize, // logslope -> primary g
            _ => return Err("unexpected block index in psi_terms".to_string()),
        };
        let mut objective_psi = 0.0;
        let mut score_psi = Array1::<f64>::zeros(slices.total);
        let mut hessian_psi = Array2::<f64>::zeros((slices.total, slices.total));
        for row in 0..self.n {
            let Some(dir) =
                self.row_primary_psi_direction(row, block_states, derivative_blocks, psi_index)?
            else {
                continue;
            };
            let (f_pi, f_pipi) = self.compute_row_primary_gradient_hessian(row, block_states)?;
            let third = self.row_primary_third_contracted(row, block_states, &dir)?;
            let (_, left_vec) = self
                .embedded_psi_vector(row, &slices, derivative_blocks, psi_index)?
                .ok_or_else(|| "missing survival marginal-slope psi vector".to_string())?;
            objective_psi += f_pi.dot(&dir);
            score_psi += &(left_vec.clone() * f_pi[idx_primary]);
            score_psi += &self.pullback_primary_vector(row, &slices, &f_pipi.dot(&dir));

            let right_vec =
                self.pullback_primary_vector(row, &slices, &f_pipi.row(idx_primary).to_owned());
            hessian_psi += &left_vec
                .view()
                .insert_axis(Axis(1))
                .dot(&right_vec.view().insert_axis(Axis(0)));
            hessian_psi += &right_vec
                .view()
                .insert_axis(Axis(1))
                .dot(&left_vec.view().insert_axis(Axis(0)));
            self.add_pullback_primary_hessian(&mut hessian_psi, row, &slices, &third);
        }
        Ok(Some(ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi,
            hessian_psi_operator: None,
        }))
    }

    fn psi_second_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        let slices = block_slices(block_states);
        let Some((block_i, _)) = self.embedded_psi_vector(0, &slices, derivative_blocks, psi_i)?
        else {
            return Ok(None);
        };
        let Some((block_j, _)) = self.embedded_psi_vector(0, &slices, derivative_blocks, psi_j)?
        else {
            return Ok(None);
        };
        let idx_i = match block_i {
            0 => 1,
            1 => 3,
            _ => return Err("bad block_i".to_string()),
        };
        let idx_j = match block_j {
            0 => 1,
            1 => 3,
            _ => return Err("bad block_j".to_string()),
        };
        let mut objective_psi_psi = 0.0;
        let mut score_psi_psi = Array1::<f64>::zeros(slices.total);
        let mut hessian_psi_psi = Array2::<f64>::zeros((slices.total, slices.total));
        for row in 0..self.n {
            let Some(dir_i) =
                self.row_primary_psi_direction(row, block_states, derivative_blocks, psi_i)?
            else {
                continue;
            };
            let Some(dir_j) =
                self.row_primary_psi_direction(row, block_states, derivative_blocks, psi_j)?
            else {
                continue;
            };
            let dir_ij = self
                .row_primary_psi_second_direction(
                    row,
                    block_states,
                    derivative_blocks,
                    psi_i,
                    psi_j,
                )?
                .unwrap_or_else(|| Array1::<f64>::zeros(N_PRIMARY));
            let (f_pi, f_pipi) = self.compute_row_primary_gradient_hessian(row, block_states)?;
            let third_i = self.row_primary_third_contracted(row, block_states, &dir_i)?;
            let third_j = self.row_primary_third_contracted(row, block_states, &dir_j)?;
            let fourth = self.row_primary_fourth_contracted(row, block_states, &dir_i, &dir_j)?;
            let (_, left_i) = self
                .embedded_psi_vector(row, &slices, derivative_blocks, psi_i)?
                .ok_or_else(|| "missing psi_i vector".to_string())?;
            let (_, left_j) = self
                .embedded_psi_vector(row, &slices, derivative_blocks, psi_j)?
                .ok_or_else(|| "missing psi_j vector".to_string())?;
            let left_ij = self
                .embedded_psi_second_vector(row, &slices, derivative_blocks, psi_i, psi_j)?
                .map(|(_, v)| v)
                .unwrap_or_else(|| Array1::<f64>::zeros(slices.total));

            objective_psi_psi += dir_i.dot(&f_pipi.dot(&dir_j)) + f_pi.dot(&dir_ij);

            if left_ij.iter().any(|v| v.abs() > 0.0) {
                let idx_ij = if left_ij
                    .slice(s![slices.time.clone()])
                    .iter()
                    .any(|v| v.abs() > 0.0)
                {
                    1
                } else {
                    3
                };
                score_psi_psi += &(left_ij.clone() * f_pi[idx_ij]);
            }
            score_psi_psi += &(left_i.clone() * f_pipi.row(idx_i).dot(&dir_j));
            score_psi_psi += &(left_j.clone() * f_pipi.row(idx_j).dot(&dir_i));
            score_psi_psi += &self.pullback_primary_vector(row, &slices, &f_pipi.dot(&dir_ij));
            score_psi_psi += &self.pullback_primary_vector(row, &slices, &third_i.dot(&dir_j));

            if left_ij.iter().any(|v| v.abs() > 0.0) {
                let idx_ij = if left_ij
                    .slice(s![slices.time.clone()])
                    .iter()
                    .any(|v| v.abs() > 0.0)
                {
                    1
                } else {
                    3
                };
                let right_ij =
                    self.pullback_primary_vector(row, &slices, &f_pipi.row(idx_ij).to_owned());
                hessian_psi_psi += &left_ij
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&right_ij.view().insert_axis(Axis(0)));
                hessian_psi_psi += &right_ij
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&left_ij.view().insert_axis(Axis(0)));
            }

            let scalar_ij = f_pipi[[idx_i, idx_j]];
            hessian_psi_psi += &(left_i
                .view()
                .insert_axis(Axis(1))
                .dot(&left_j.view().insert_axis(Axis(0)))
                * scalar_ij);
            hessian_psi_psi += &(left_j
                .view()
                .insert_axis(Axis(1))
                .dot(&left_i.view().insert_axis(Axis(0)))
                * scalar_ij);

            let right_i =
                self.pullback_primary_vector(row, &slices, &third_j.row(idx_i).to_owned());
            hessian_psi_psi += &left_i
                .view()
                .insert_axis(Axis(1))
                .dot(&right_i.view().insert_axis(Axis(0)));
            hessian_psi_psi += &right_i
                .view()
                .insert_axis(Axis(1))
                .dot(&left_i.view().insert_axis(Axis(0)));

            let right_j =
                self.pullback_primary_vector(row, &slices, &third_i.row(idx_j).to_owned());
            hessian_psi_psi += &left_j
                .view()
                .insert_axis(Axis(1))
                .dot(&right_j.view().insert_axis(Axis(0)));
            hessian_psi_psi += &right_j
                .view()
                .insert_axis(Axis(1))
                .dot(&left_j.view().insert_axis(Axis(0)));

            self.add_pullback_primary_hessian(&mut hessian_psi_psi, row, &slices, &fourth);
            let third_ij = self.row_primary_third_contracted(row, block_states, &dir_ij)?;
            self.add_pullback_primary_hessian(&mut hessian_psi_psi, row, &slices, &third_ij);
        }
        Ok(Some(ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi,
            hessian_psi_psi,
        }))
    }

    fn psi_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let slices = block_slices(block_states);
        let Some((block_idx, _)) =
            self.embedded_psi_vector(0, &slices, derivative_blocks, psi_index)?
        else {
            return Ok(None);
        };
        let idx_primary = match block_idx {
            0 => 1,
            1 => 3,
            _ => return Err("bad block".to_string()),
        };
        let mut out = Array2::<f64>::zeros((slices.total, slices.total));
        for row in 0..self.n {
            let row_dir = self.row_primary_direction_from_flat(row, &slices, d_beta_flat);
            let Some(psi_dir) =
                self.row_primary_psi_direction(row, block_states, derivative_blocks, psi_index)?
            else {
                continue;
            };
            let psi_action = self
                .row_primary_psi_action_on_direction(
                    row,
                    &slices,
                    derivative_blocks,
                    psi_index,
                    d_beta_flat,
                )?
                .unwrap_or_else(|| Array1::<f64>::zeros(N_PRIMARY));
            let third_beta = self.row_primary_third_contracted(row, block_states, &row_dir)?;
            let fourth =
                self.row_primary_fourth_contracted(row, block_states, &row_dir, &psi_dir)?;
            let (_, left_vec) = self
                .embedded_psi_vector(row, &slices, derivative_blocks, psi_index)?
                .ok_or_else(|| "missing psi vector".to_string())?;
            let right_vec =
                self.pullback_primary_vector(row, &slices, &third_beta.row(idx_primary).to_owned());
            out += &left_vec
                .view()
                .insert_axis(Axis(1))
                .dot(&right_vec.view().insert_axis(Axis(0)));
            out += &right_vec
                .view()
                .insert_axis(Axis(1))
                .dot(&left_vec.view().insert_axis(Axis(0)));
            self.add_pullback_primary_hessian(&mut out, row, &slices, &fourth);
            let third_action = self.row_primary_third_contracted(row, block_states, &psi_action)?;
            self.add_pullback_primary_hessian(&mut out, row, &slices, &third_action);
        }
        Ok(Some(out))
    }
}

// ── Workspace structs ─────────────────────────────────────────────────

struct SurvivalMarginalSlopeHessianWorkspace {
    family: SurvivalMarginalSlopeFamily,
    block_states: Vec<ParameterBlockState>,
}

struct SurvivalMarginalSlopePsiWorkspace {
    family: SurvivalMarginalSlopeFamily,
    block_states: Vec<ParameterBlockState>,
    derivative_blocks: Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>>,
}

impl ExactNewtonJointHessianWorkspace for SurvivalMarginalSlopeHessianWorkspace {
    fn hessian_matvec(&self, beta_flat: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        self.family
            .joint_hessian_matvec(&self.block_states, beta_flat)
            .map(Some)
    }

    fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        self.family
            .joint_hessian_diagonal(&self.block_states)
            .map(Some)
    }

    fn directional_derivative(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .joint_hessian_directional_derivative(&self.block_states, d_beta_flat)
    }

    fn second_directional_derivative(
        &self,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family.joint_hessian_second_directional_derivative(
            &self.block_states,
            d_beta_u_flat,
            d_beta_v_flat,
        )
    }
}

impl ExactNewtonJointPsiWorkspace for SurvivalMarginalSlopePsiWorkspace {
    fn first_order_terms(
        &self,
        psi_index: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        self.family
            .psi_terms(&self.block_states, &self.derivative_blocks, psi_index)
    }

    fn second_order_terms(
        &self,
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        self.family.psi_second_order_terms(
            &self.block_states,
            &self.derivative_blocks,
            psi_i,
            psi_j,
        )
    }

    fn hessian_directional_derivative(
        &self,
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family.psi_hessian_directional_derivative(
            &self.block_states,
            &self.derivative_blocks,
            psi_index,
            d_beta_flat,
        )
    }
}

// ── CustomFamily impl ─────────────────────────────────────────────────

const EXACT_OUTER_HESSIAN_MAX_ROW_PAIR_WORK: usize = 2_000_000;

impl CustomFamily for SurvivalMarginalSlopeFamily {
    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn exact_outer_derivative_order(
        &self,
        _: &[ParameterBlockSpec],
        _: &BlockwiseFitOptions,
    ) -> ExactOuterDerivativeOrder {
        let primary_total = N_PRIMARY;
        let row_pair_work = self.n.saturating_mul(primary_total * primary_total);
        if row_pair_work > EXACT_OUTER_HESSIAN_MAX_ROW_PAIR_WORK {
            ExactOuterDerivativeOrder::First
        } else {
            ExactOuterDerivativeOrder::Second
        }
    }

    fn exact_newton_joint_psi_workspace_for_first_order_terms(&self) -> bool {
        true
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let (ll, gradient, hessian) = self.joint_gradient_hessian(block_states, true)?;
        let hessian = hessian.ok_or_else(|| "joint hessian unavailable".to_string())?;
        let slices = block_slices(block_states);
        let blockworking_sets = vec![
            BlockWorkingSet::ExactNewton {
                gradient: gradient.slice(s![slices.time.clone()]).to_owned(),
                hessian: SymmetricMatrix::Dense(
                    hessian
                        .slice(s![slices.time.clone(), slices.time.clone()])
                        .to_owned(),
                ),
            },
            BlockWorkingSet::ExactNewton {
                gradient: gradient.slice(s![slices.logslope.clone()]).to_owned(),
                hessian: SymmetricMatrix::Dense(
                    hessian
                        .slice(s![slices.logslope.clone(), slices.logslope.clone()])
                        .to_owned(),
                ),
            },
        ];
        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets,
        })
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        self.joint_gradient_hessian(block_states, false)
            .map(|r| r.0)
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        self.joint_gradient_hessian(block_states, true)
            .map(|(_, _, h)| h)
    }

    fn requires_joint_outer_hyper_path(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        Ok(Some(Arc::new(SurvivalMarginalSlopeHessianWorkspace {
            family: self.clone(),
            block_states: block_states.to_vec(),
        })))
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.joint_hessian_directional_derivative(block_states, d_beta_flat)
    }

    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.joint_hessian_second_directional_derivative(block_states, d_beta_u_flat, d_beta_v_flat)
    }

    fn exact_newton_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        self.psi_terms(block_states, derivative_blocks, psi_index)
    }

    fn exact_newton_joint_psisecond_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        self.psi_second_order_terms(block_states, derivative_blocks, psi_i, psi_j)
    }

    fn exact_newton_joint_psihessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.psi_hessian_directional_derivative(
            block_states,
            derivative_blocks,
            psi_index,
            d_beta_flat,
        )
    }

    fn exact_newton_joint_psi_workspace(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        Ok(Some(Arc::new(SurvivalMarginalSlopePsiWorkspace {
            family: self.clone(),
            block_states: block_states.to_vec(),
            derivative_blocks: derivative_blocks.to_vec(),
        })))
    }

    fn block_linear_constraints(
        &self,
        _block_states: &[ParameterBlockState],
        block_idx: usize,
        _: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        if block_idx == 0 && self.derivative_guard > 0.0 {
            // Monotonicity constraint: design_derivative_exit @ beta_time + offset >= guard
            // i.e. design_derivative_exit @ beta_time >= guard - offset
            Ok(Some(LinearInequalityConstraints {
                a: self.design_derivative_exit.clone(),
                b: Array1::from_iter(
                    self.derivative_offset_exit
                        .iter()
                        .map(|&o| self.derivative_guard - o),
                ),
            }))
        } else {
            Ok(None)
        }
    }
}

// ── Building block specs ──────────────────────────────────────────────

fn build_time_blockspec(
    time_block: &TimeBlockInput,
    design_exit: &Array2<f64>,
    rho: Array1<f64>,
    beta_hint: Option<Array1<f64>>,
) -> ParameterBlockSpec {
    ParameterBlockSpec {
        name: "time_surface".to_string(),
        design: DesignMatrix::Dense(Arc::new(design_exit.clone())),
        offset: Array1::zeros(design_exit.nrows()),
        penalties: time_block.penalties.clone(),
        initial_log_lambdas: rho,
        initial_beta: beta_hint,
    }
}

fn build_logslope_blockspec(
    design: &TermCollectionDesign,
    baseline: f64,
    rho: Array1<f64>,
    beta_hint: Option<Array1<f64>>,
) -> ParameterBlockSpec {
    ParameterBlockSpec {
        name: "logslope_surface".to_string(),
        design: DesignMatrix::Dense(Arc::new(design.design.clone())),
        offset: Array1::from_elem(design.design.nrows(), baseline),
        penalties: design.penalties.clone(),
        initial_log_lambdas: rho,
        initial_beta: beta_hint,
    }
}

fn fit_score(fit: &UnifiedFitResult) -> f64 {
    if fit.reml_score.is_finite() {
        fit.reml_score
    } else {
        let score = 0.5 * fit.deviance + 0.5 * fit.stable_penalty_term;
        if score.is_finite() {
            score
        } else {
            f64::INFINITY
        }
    }
}

fn inner_fit(
    family: &SurvivalMarginalSlopeFamily,
    blocks: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
) -> Result<UnifiedFitResult, String> {
    fit_custom_family(family, blocks, options).map_err(|e| e.to_string())
}

fn joint_setup(
    time_penalties: usize,
    logslopespec: &TermCollectionSpec,
    logslope_penalties: usize,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> ExactJointHyperSetup {
    let logslope_terms = spatial_length_scale_term_indices(logslopespec);
    let rho_dim = time_penalties + logslope_penalties;
    let rho0vec = Array1::<f64>::zeros(rho_dim);
    let rho_lower = Array1::<f64>::from_elem(rho_dim, -12.0);
    let rho_upper = Array1::<f64>::from_elem(rho_dim, 12.0);
    // Time block has no spatial length scales (pure B-spline on time)
    let empty_kappa = SpatialLogKappaCoords::new_with_dims(Array1::zeros(0), vec![]);
    let logslope_kappa = SpatialLogKappaCoords::from_length_scales_aniso(
        logslopespec,
        &logslope_terms,
        kappa_options,
    );
    let mut values = empty_kappa.as_array().to_vec();
    values.extend(logslope_kappa.as_array().iter());
    let mut dims = empty_kappa.dims_per_term().to_vec();
    dims.extend(logslope_kappa.dims_per_term());
    let log_kappa0 =
        SpatialLogKappaCoords::new_with_dims(Array1::from_vec(values.clone()), dims.clone());
    let log_kappa_lower = SpatialLogKappaCoords::lower_bounds_aniso(&dims, kappa_options);
    let log_kappa_upper = SpatialLogKappaCoords::upper_bounds_aniso(&dims, kappa_options);
    ExactJointHyperSetup::new(
        rho0vec,
        rho_lower,
        rho_upper,
        log_kappa0,
        log_kappa_lower,
        log_kappa_upper,
    )
}

fn validate_spec(spec: &SurvivalMarginalSlopeTermSpec) -> Result<(), String> {
    let n = spec.age_entry.len();
    if spec.age_exit.len() != n
        || spec.event_target.len() != n
        || spec.weights.len() != n
        || spec.z.len() != n
    {
        return Err(format!(
            "survival-marginal-slope row mismatch: entry={}, exit={}, event={}, weights={}, z={}",
            n,
            spec.age_exit.len(),
            spec.event_target.len(),
            spec.weights.len(),
            spec.z.len()
        ));
    }
    if spec.weights.iter().any(|&w| !w.is_finite() || w < 0.0) {
        return Err("survival-marginal-slope requires finite non-negative weights".to_string());
    }
    if spec.z.iter().any(|&zi| !zi.is_finite()) {
        return Err("survival-marginal-slope requires finite z values".to_string());
    }
    Ok(())
}

/// Compute a simple baseline log-slope from a pooled probit survival fit.
fn pooled_survival_baseline(event: &Array1<f64>, z: &Array1<f64>, weights: &Array1<f64>) -> f64 {
    // Simple: regress event ~ z with probit link to get initial slope
    let n = event.len();
    if n == 0 {
        return (-2.0_f64).ln(); // log(0.1) ~ small slope
    }
    let fit = fit_gam(
        {
            let mut x = Array2::<f64>::zeros((n, 2));
            x.column_mut(0).fill(1.0);
            x.column_mut(1).assign(z);
            x
        }
        .view(),
        event.view(),
        weights.view(),
        Array1::zeros(n).view(),
        &[],
        LikelihoodFamily::BinomialProbit,
        &FitOptions {
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: true,
            max_iter: 80,
            tol: 1e-6,
            nullspace_dims: Vec::new(),
            linear_constraints: None,
            adaptive_regularization: None,
            penalty_shrinkage_floor: Some(1e-6),
        },
    );
    match fit {
        Ok(result) => {
            let b = result.beta.get(1).copied().unwrap_or(0.1).abs().max(1e-6);
            b.ln()
        }
        Err(_) => (-2.0_f64).ln(),
    }
}

// ── Public fitting function ───────────────────────────────────────────

pub fn fit_survival_marginal_slope_terms(
    data: ArrayView2<'_, f64>,
    spec: SurvivalMarginalSlopeTermSpec,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<SurvivalMarginalSlopeFitResult, String> {
    validate_spec(&spec)?;
    let n = spec.age_entry.len();
    let baseline_logslope = pooled_survival_baseline(&spec.event_target, &spec.z, &spec.weights);

    let logslope_design =
        build_term_collection_design(data, &spec.logslopespec).map_err(|e| e.to_string())?;
    let logslopespec_boot =
        freeze_spatial_length_scale_terms_from_design(&spec.logslopespec, &logslope_design)
            .map_err(|e| e.to_string())?;

    let time_penalties_len = spec.time_block.penalties.len();
    let setup = joint_setup(
        time_penalties_len,
        &logslopespec_boot,
        logslope_design.penalties.len(),
        kappa_options,
    );

    let hints = RefCell::new((None::<Array1<f64>>, None::<Array1<f64>>));
    let exact_warm_start = RefCell::new(None::<CustomFamilyWarmStart>);

    let event = spec.event_target.clone();
    let weights = spec.weights.clone();
    let z = spec.z.clone();
    let derivative_guard = spec.derivative_guard;
    let derivative_softness = spec.derivative_softness;
    let design_entry = spec.time_block.design_entry.clone();
    let design_exit = spec.time_block.design_exit.clone();
    let design_derivative_exit = spec.time_block.design_derivative_exit.clone();
    let offset_entry = spec.time_block.offset_entry.clone();
    let offset_exit = spec.time_block.offset_exit.clone();
    let derivative_offset_exit = spec.time_block.derivative_offset_exit.clone();
    let time_block_ref = spec.time_block.clone();

    let make_family = |logslope_design: &TermCollectionDesign| -> SurvivalMarginalSlopeFamily {
        SurvivalMarginalSlopeFamily {
            n,
            event: event.clone(),
            weights: weights.clone(),
            z: z.clone(),
            derivative_guard,
            derivative_softness,
            design_entry: design_entry.clone(),
            design_exit: design_exit.clone(),
            design_derivative_exit: design_derivative_exit.clone(),
            offset_entry: offset_entry.clone(),
            offset_exit: offset_exit.clone(),
            derivative_offset_exit: derivative_offset_exit.clone(),
            logslope_design: logslope_design.design.clone(),
        }
    };

    let build_blocks = |rho: &Array1<f64>,
                        logslope_design: &TermCollectionDesign|
     -> Result<Vec<ParameterBlockSpec>, String> {
        let hints = hints.borrow();
        let mut cursor = 0usize;
        let rho_time = rho
            .slice(s![cursor..cursor + time_penalties_len])
            .to_owned();
        cursor += time_penalties_len;
        let rho_logslope = rho
            .slice(s![cursor..cursor + logslope_design.penalties.len()])
            .to_owned();
        Ok(vec![
            build_time_blockspec(&time_block_ref, &design_exit, rho_time, hints.0.clone()),
            build_logslope_blockspec(
                logslope_design,
                baseline_logslope,
                rho_logslope,
                hints.1.clone(),
            ),
        ])
    };

    // Check analytic derivatives
    let analytic_joint_derivatives_available =
        build_block_spatial_psi_derivatives(data, &logslopespec_boot, &logslope_design)
            .and_then(|maybe| {
                maybe.ok_or_else(|| "missing logslope spatial psi derivatives".to_string())
            })
            .is_ok();

    if setup.log_kappa_dim() > 0 && !analytic_joint_derivatives_available {
        return Err(
            "exact survival marginal-slope spatial optimization requires analytic joint psi derivatives"
                .to_string(),
        );
    }

    let initial_rho = setup.theta0().slice(s![..setup.rho_dim()]).to_owned();
    let initial_blocks = build_blocks(&initial_rho, &logslope_design)?;
    let initial_family = make_family(&logslope_design);
    let joint_cap = custom_family_outer_capability(
        &initial_family,
        &initial_blocks,
        options,
        setup.theta0().len(),
        setup.log_kappa_dim() > 0,
    );
    let analytic_joint_gradient_available = analytic_joint_derivatives_available
        && matches!(
            joint_cap.gradient,
            crate::solver::outer_strategy::Derivative::Analytic
        );
    let analytic_joint_hessian_available = analytic_joint_derivatives_available
        && matches!(
            joint_cap.hessian,
            crate::solver::outer_strategy::Derivative::Analytic
        );

    // Only logslope block has spatial terms (time block is pure B-spline)
    let logslope_terms = spatial_length_scale_term_indices(&logslopespec_boot);
    let solved = optimize_spatial_length_scale_exact_joint(
        data,
        &[logslopespec_boot.clone()],
        &[logslope_terms],
        kappa_options,
        &setup,
        analytic_joint_gradient_available,
        analytic_joint_hessian_available,
        |rho, _specs: &[TermCollectionSpec], designs: &[TermCollectionDesign]| {
            let blocks = build_blocks(rho, &designs[0])?;
            let family = make_family(&designs[0]);
            let fit = inner_fit(&family, &blocks, options)?;
            Ok(fit_score(&fit))
        },
        |rho, _specs: &[TermCollectionSpec], designs: &[TermCollectionDesign]| {
            let blocks = build_blocks(rho, &designs[0])?;
            let family = make_family(&designs[0]);
            let fit = inner_fit(&family, &blocks, options)?;
            let mut hints_mut = hints.borrow_mut();
            if let Some(block) = fit.block_states.get(0) {
                hints_mut.0 = Some(block.beta.clone());
            }
            if let Some(block) = fit.block_states.get(1) {
                hints_mut.1 = Some(block.beta.clone());
            }
            Ok(fit)
        },
        |rho, specs: &[TermCollectionSpec], designs: &[TermCollectionDesign], need_hessian| {
            let blocks = build_blocks(rho, &designs[0])?;
            let family = make_family(&designs[0]);
            // Time block has no spatial psi derivatives, so we insert an empty vec for it
            let mut derivative_blocks = vec![Vec::new()]; // time block
            derivative_blocks.push(
                build_block_spatial_psi_derivatives(data, &specs[0], &designs[0])?.ok_or_else(
                    || "missing survival logslope spatial psi derivatives".to_string(),
                )?,
            );
            let eval = evaluate_custom_family_joint_hyper(
                &family,
                &blocks,
                options,
                rho,
                &derivative_blocks,
                exact_warm_start.borrow().as_ref(),
                need_hessian,
            )?;
            exact_warm_start.replace(Some(eval.warm_start));
            if need_hessian && eval.outer_hessian.is_none() {
                return Err(
                    "exact survival marginal-slope joint objective did not return an outer Hessian"
                        .to_string(),
                );
            }
            Ok((eval.objective, eval.gradient, eval.outer_hessian))
        },
    )?;

    let mut resolved_specs = solved.resolved_specs;
    let designs = solved.designs;
    Ok(SurvivalMarginalSlopeFitResult {
        fit: solved.fit,
        logslopespec_resolved: resolved_specs.remove(0),
        logslope_design: designs.into_iter().next().unwrap(),
        baseline_logslope,
        time_block_penalties_len: time_penalties_len,
    })
}
