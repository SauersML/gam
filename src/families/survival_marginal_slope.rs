use crate::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, CustomFamilyPsiDesignAction,
    CustomFamilyPsiSecondDesignAction, CustomFamilyWarmStart, ExactNewtonJointHessianWorkspace,
    ExactNewtonJointPsiSecondOrderTerms, ExactNewtonJointPsiTerms, ExactNewtonJointPsiWorkspace,
    ExactOuterDerivativeOrder, FamilyEvaluation, ParameterBlockSpec, ParameterBlockState,
    PenaltyMatrix, build_block_spatial_psi_derivatives, custom_family_outer_capability,
    evaluate_custom_family_joint_hyper, first_psi_linear_map, fit_custom_family,
    second_psi_linear_map,
};
use crate::estimate::UnifiedFitResult;
use crate::families::bernoulli_marginal_slope::{
    MultiDirJet, signed_probit_logcdf_and_mills_ratio,
    signed_probit_neglog_derivatives_up_to_fourth, unary_derivatives_log,
    unary_derivatives_log_normal_pdf, unary_derivatives_neglog_phi, unary_derivatives_sqrt,
};
use crate::families::survival_location_scale::{
    TimeBlockInput, structural_nonnegative_constraints,
};
use crate::matrix::{DesignMatrix, SymmetricMatrix};
use crate::pirls::LinearInequalityConstraints;
use crate::smooth::{
    ExactJointHyperSetup, SpatialLengthScaleOptimizationOptions, SpatialLogKappaCoords,
    TermCollectionDesign, TermCollectionSpec, build_term_collection_design,
    freeze_spatial_length_scale_terms_from_design, optimize_spatial_length_scale_exact_joint,
    spatial_length_scale_term_indices,
};
use ndarray::{Array1, Array2, ArrayView2, ArrayViewMut1, Axis, s};
use rayon::prelude::*;
use std::cell::RefCell;
use std::sync::Arc;

// ── Spec and result types ─────────────────────────────────────────────

#[derive(Clone)]
pub struct SurvivalMarginalSlopeTermSpec {
    pub age_entry: Array1<f64>,
    pub age_exit: Array1<f64>,
    pub event_target: Array1<f64>,
    pub weights: Array1<f64>,
    pub z: Array1<f64>,
    pub marginalspec: TermCollectionSpec,
    /// Strict lower bound on q'(t) used by both the likelihood domain and
    /// the monotonicity constraints.
    pub derivative_guard: f64,
    pub time_block: TimeBlockInput,
    pub logslopespec: TermCollectionSpec,
}

pub const DEFAULT_SURVIVAL_MARGINAL_SLOPE_DERIVATIVE_GUARD: f64 = 1e-6;

pub struct SurvivalMarginalSlopeFitResult {
    pub fit: UnifiedFitResult,
    pub marginalspec_resolved: TermCollectionSpec,
    pub logslopespec_resolved: TermCollectionSpec,
    pub marginal_design: TermCollectionDesign,
    pub logslope_design: TermCollectionDesign,
    pub baseline_slope: f64,
    pub time_block_penalties_len: usize,
}

// ── Family struct ─────────────────────────────────────────────────────

/// The time block has one beta vector but THREE design matrices (entry, exit,
/// derivative-at-exit). The ParameterBlockSpec uses the exit design as its
/// "official" design, so block_states[0].eta = design_exit @ beta + offset_exit.
/// This eta is NOT used in the likelihood computation — row_neglog_directional
/// recomputes all 3 linear predictors from beta_time directly. The exit-design
/// eta exists only to satisfy the CustomFamily/PIRLS interface; ExactNewton
/// blocks do not use eta for working response/weights.
#[derive(Clone)]
struct SurvivalMarginalSlopeFamily {
    n: usize,
    event: Arc<Array1<f64>>,
    weights: Arc<Array1<f64>>,
    z: Arc<Array1<f64>>,
    derivative_guard: f64,
    /// Time block: 3 designs sharing one beta vector.
    design_entry: Arc<Array2<f64>>,
    design_exit: Arc<Array2<f64>>,
    design_derivative_exit: Arc<Array2<f64>>,
    offset_entry: Arc<Array1<f64>>,
    offset_exit: Arc<Array1<f64>>,
    derivative_offset_exit: Arc<Array1<f64>>,
    /// Baseline covariate block: contributes additively to q0 and q1, but not qd1.
    marginal_design: DesignMatrix,
    /// Log-slope block: standard single design.
    logslope_design: DesignMatrix,
    time_linear_constraints: Option<LinearInequalityConstraints>,
}

// ── Block layout ──────────────────────────────────────────────────────

#[derive(Clone)]
struct BlockSlices {
    time: std::ops::Range<usize>,
    marginal: std::ops::Range<usize>,
    logslope: std::ops::Range<usize>,
    total: usize,
}

fn block_slices(block_states: &[ParameterBlockState]) -> BlockSlices {
    let time = 0..block_states[0].beta.len();
    let marginal = time.end..time.end + block_states[1].beta.len();
    let logslope = marginal.end..marginal.end + block_states[2].beta.len();
    let total = logslope.end;
    BlockSlices {
        time,
        marginal,
        logslope,
        total,
    }
}

fn dense_row_axpy_into(
    design: &Array2<f64>,
    row: usize,
    alpha: f64,
    out: &mut ArrayViewMut1<'_, f64>,
) {
    if alpha == 0.0 {
        return;
    }
    for (dst, &value) in out.iter_mut().zip(design.row(row).iter()) {
        *dst += alpha * value;
    }
}

fn dense_row_outer_into(
    lhs: &Array2<f64>,
    rhs: &Array2<f64>,
    row: usize,
    alpha: f64,
    target: &mut Array2<f64>,
) {
    if alpha == 0.0 {
        return;
    }
    let x = lhs.row(row);
    let y = rhs.row(row);
    for i in 0..x.len() {
        let xi = x[i];
        if xi == 0.0 {
            continue;
        }
        for j in 0..y.len() {
            target[[i, j]] += alpha * xi * y[j];
        }
    }
}

// ── Primary-space helpers ─────────────────────────────────────────────

// Primary scalar indices: 0=q0, 1=q1, 2=qd1, 3=g
const N_PRIMARY: usize = 4;

fn unit_primary_direction(idx: usize) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(N_PRIMARY);
    out[idx] = 1.0;
    out
}

fn spatial_block_primary_loading(block_idx: usize) -> Result<Array1<f64>, String> {
    match block_idx {
        1 => Ok(Array1::from_vec(vec![1.0, 1.0, 0.0, 0.0])),
        2 => Ok(Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0])),
        _ => Err(format!(
            "survival marginal-slope spatial psi loading requested for unsupported block {block_idx}"
        )),
    }
}

// ── Closed-form row kernel ─────────────────────────────────────────────
//
// The survival marginal-slope NLL for row i is:
//
//   ℓ_i = w_i [ (1-d)·neglogΦ(-η₁) − neglogΦ(-η₀) − d·logφ(η₁) − d·log(a'₁) ]
//
// with η₀ = q₀c + gz, η₁ = q₁c + gz, a'₁ = qd₁·c, c = √(1+g²).
//
// All derivatives w.r.t. the 4 primary scalars (q₀, q₁, qd₁, g) are
// closed-form scalar formulas. No jets, no per-row heap allocation.

/// Derivatives of c(g) = √(1+g²) up to 4th order.
#[inline]
fn c_derivatives(g: f64) -> (f64, f64, f64, f64, f64) {
    let g2 = g * g;
    let c = (1.0 + g2).sqrt();
    let c2 = c * c; // = 1 + g²
    let c3 = c2 * c;
    let c5 = c3 * c2;
    let c7 = c5 * c2;
    let c1 = g2 / c; // g * g / c = g² / c
    let c2d = g2 * (g2 + 2.0) / c3;
    let c3d = g2 * (g2 * g2 + 2.0 * g2 + 4.0) / c5;
    let c4d = g2 * (g2 * g2 * g2 + 4.0 * g2 * g2 - 4.0 * g2 + 8.0) / c7;
    (c, c1, c2d, c3d, c4d)
}

/// Derivatives of neglog(x) = -log(x): [-1/x, 1/x², -2/x³, 6/x⁴].
#[inline]
fn neglog_derivatives(x: f64) -> (f64, f64, f64, f64) {
    let x1 = x.max(1e-300);
    let inv = 1.0 / x1;
    let inv2 = inv * inv;
    (-inv, inv2, -2.0 * inv2 * inv, 6.0 * inv2 * inv2)
}

/// Row-level primary gradient (4-vector) and Hessian (4×4 symmetric)
/// computed entirely from closed-form scalar formulas.
///
/// Returns (nll, gradient[4], hessian[4][4]) on the stack.
#[inline]
fn row_primary_closed_form(
    q0: f64,
    q1: f64,
    qd1: f64,
    g: f64,
    z: f64,
    w: f64,
    d: f64,
    derivative_guard: f64,
) -> Result<(f64, [f64; N_PRIMARY], [[f64; N_PRIMARY]; N_PRIMARY]), String> {
    let (c, c1, c2, ..) = c_derivatives(g);

    // Linear predictors
    let eta0 = q0 * c + g * z;
    let eta1 = q1 * c + g * z;
    let ad1 = qd1 * c;

    if qd1 < derivative_guard {
        return Err(format!(
            "survival marginal-slope monotonicity violated: qd1={qd1:.3e} < guard={derivative_guard:.3e}"
        ));
    }

    // ── NLL terms ──
    // Entry survival: -neglogΦ(-η₀) = logΦ(-η₀)
    let (logcdf_neg_eta0, _) = signed_probit_logcdf_and_mills_ratio(-eta0);
    // Exit survival: (1-d)·neglogΦ(-η₁)
    let (logcdf_neg_eta1, _) = signed_probit_logcdf_and_mills_ratio(-eta1);
    // Event density: d·logφ(η₁)
    let log_phi_eta1 = -0.5 * (eta1 * eta1 + std::f64::consts::TAU.ln());
    // Time derivative: d·log(ad1)
    let log_ad1 = ad1.max(1e-300).ln();

    let nll = w * ((1.0 - d) * (-logcdf_neg_eta1) - logcdf_neg_eta0 - d * log_phi_eta1 - d * log_ad1);

    // ── First and second derivatives of each NLL component ──
    // For neglogΦ(-η): f(η) = -logΦ(-η), f'(η) = φ(-η)/Φ(-η) = λ(-η).
    // signed_probit_neglog_derivatives gives (k1, k2, k3, k4) for -logΦ(m)
    // where k1 = -λ(m), k2 = λ(m+λ), etc.
    // For entry: m = -η₀, weight = -w (because we subtract this term)
    let (e0_k1, e0_k2, _, _) =
        signed_probit_neglog_derivatives_up_to_fourth(-eta0, -w);
    // For exit: m = -η₁, weight = w(1-d)
    let (e1_k1, e1_k2, _, _) =
        signed_probit_neglog_derivatives_up_to_fourth(-eta1, w * (1.0 - d));
    // Event density: -d·logφ(η₁) = d·(η₁²/2 + const).
    // d/dη₁ = d·w·η₁, d²/dη₁² = d·w.
    let phi_u1 = w * d * eta1;
    let phi_u2 = w * d;
    // Time derivative: -d·log(ad1).
    let (nl_u1, nl_u2, _, _) = neglog_derivatives(ad1);
    let td_u1 = w * d * nl_u1;
    let td_u2 = w * d * nl_u2;

    // ── Chain rule to primary space ──
    // η₀ depends on (q₀, g): ∂η₀/∂q₀ = c, ∂η₀/∂g = q₀c₁ + z
    // η₁ depends on (q₁, g): ∂η₁/∂q₁ = c, ∂η₁/∂g = q₁c₁ + z
    // ad1 depends on (qd1, g): ∂ad1/∂qd1 = c, ∂ad1/∂g = qd1·c₁
    let deta0_dq0 = c;
    let deta0_dg = q0 * c1 + z;
    let deta1_dq1 = c;
    let deta1_dg = q1 * c1 + z;
    let dad1_dqd1 = c;
    let dad1_dg = qd1 * c1;

    // Combined first derivatives of total NLL:
    // u1 for η₀ terms = e0_k1 (entry)
    // u1 for η₁ terms = e1_k1 + phi_u1 (exit + density)
    // u1 for ad1 term = td_u1 (time derivative)
    let u1_eta0 = e0_k1;
    let u1_eta1 = e1_k1 + phi_u1;
    let u1_ad1 = td_u1;

    let mut grad = [0.0_f64; N_PRIMARY];
    grad[0] = u1_eta0 * deta0_dq0; // ∂ℓ/∂q₀
    grad[1] = u1_eta1 * deta1_dq1; // ∂ℓ/∂q₁
    grad[2] = u1_ad1 * dad1_dqd1; // ∂ℓ/∂qd₁
    grad[3] = u1_eta0 * deta0_dg + u1_eta1 * deta1_dg + u1_ad1 * dad1_dg; // ∂ℓ/∂g

    // Combined second derivatives:
    let u2_eta0 = e0_k2;
    let u2_eta1 = e1_k2 + phi_u2;
    let u2_ad1 = td_u2;

    // Second mixed derivatives of η w.r.t. primary scalars:
    let d2eta0_dq0dg = c1;
    let d2eta1_dq1dg = c1;
    let d2ad1_dqd1dg = c1;
    // d²η₀/dg² = q₀·c₂ (z is linear in g, so its second derivative is 0)
    let d2eta0_dg2 = q0 * c2;
    let d2eta1_dg2 = q1 * c2;
    let d2ad1_dg2 = qd1 * c2;

    let mut hess = [[0.0_f64; N_PRIMARY]; N_PRIMARY];

    // (q0, q0)
    hess[0][0] = u2_eta0 * deta0_dq0 * deta0_dq0;
    // (q1, q1)
    hess[1][1] = u2_eta1 * deta1_dq1 * deta1_dq1;
    // (qd1, qd1)
    hess[2][2] = u2_ad1 * dad1_dqd1 * dad1_dqd1;
    // (q0, q1) = 0 (η₀ and η₁ share no primary scalars except g)
    hess[0][1] = 0.0;
    hess[1][0] = 0.0;
    // (q0, qd1) = 0
    hess[0][2] = 0.0;
    hess[2][0] = 0.0;
    // (q1, qd1) = 0
    hess[1][2] = 0.0;
    hess[2][1] = 0.0;
    // (q0, g) = u2_η₀ · (∂η₀/∂q₀)(∂η₀/∂g) + u1_η₀ · (∂²η₀/∂q₀∂g)
    hess[0][3] = u2_eta0 * deta0_dq0 * deta0_dg + u1_eta0 * d2eta0_dq0dg;
    hess[3][0] = hess[0][3];
    // (q1, g)
    hess[1][3] = u2_eta1 * deta1_dq1 * deta1_dg + u1_eta1 * d2eta1_dq1dg;
    hess[3][1] = hess[1][3];
    // (qd1, g)
    hess[2][3] = u2_ad1 * dad1_dqd1 * dad1_dg + u1_ad1 * d2ad1_dqd1dg;
    hess[3][2] = hess[2][3];
    // (g, g) = Σ_terms [u2·(dterm/dg)² + u1·(d²term/dg²)]
    hess[3][3] = u2_eta0 * deta0_dg * deta0_dg + u1_eta0 * d2eta0_dg2
        + u2_eta1 * deta1_dg * deta1_dg + u1_eta1 * d2eta1_dg2
        + u2_ad1 * dad1_dg * dad1_dg + u1_ad1 * d2ad1_dg2;

    Ok((nll, grad, hess))
}

// ── Eval cache ────────────────────────────────────────────────────────
//
// Third and fourth order contracted derivatives for the outer REML path
// continue to use the MultiDirJet engine via row_neglog_directional.
// That path is called O(n_rho²) times, not O(n × inner_iters) times,
// so the jet overhead is acceptable there.

#[derive(Clone)]
struct RowPrimaryBase {
    gradient: Array1<f64>,
    hessian: Array2<f64>,
}

struct EvalCache {
    slices: BlockSlices,
    row_bases: Vec<RowPrimaryBase>,
}

// ── Row-level NLL computation ─────────────────────────────────────────

impl SurvivalMarginalSlopeFamily {
    fn time_derivative_lower_bound(&self) -> f64 {
        assert!(self.derivative_guard.is_finite() && self.derivative_guard > 0.0);
        self.derivative_guard
    }

    /// Per-row NLL and its directional derivatives through 4 primary scalars.
    ///
    /// NLL_i = w_i * [ (1-d)·neglogΦ(-η₁) − neglogΦ(-η₀) − d·logφ(η₁) − d·log(a'₁) ]
    ///
    /// where η = a(t) + β·z, a(t) = q(t)·√(1+β²), β = g.
    ///
    /// block_states[0].eta is from the exit design and is NOT used here;
    /// all 3 time-block linear predictors are recomputed from beta_time
    /// because the time block has 3 design matrices sharing one coefficient vector.
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

        // Compute q0, q1, qd1 from the shared time block plus the baseline
        // covariate block. The time block has a single beta vector but 3
        // different design matrices (entry, exit, derivative).
        let beta_time = &block_states[0].beta;
        let beta_marginal = &block_states[1].beta;
        let q0_val = self.design_entry.row(row).dot(beta_time)
            + self.offset_entry[row]
            + self.marginal_design.dot_row(row, beta_marginal);
        let q1_val = self.design_exit.row(row).dot(beta_time)
            + self.offset_exit[row]
            + self.marginal_design.dot_row(row, beta_marginal);
        let qd1_val =
            self.design_derivative_exit.row(row).dot(beta_time) + self.derivative_offset_exit[row];
        let g_val = block_states[2].eta[row];

        let q0_jet = MultiDirJet::linear(k, q0_val, &q0_first);
        let q1_jet = MultiDirJet::linear(k, q1_val, &q1_first);
        let qd1_jet = MultiDirJet::linear(k, qd1_val, &qd1_first);
        let g_jet = MultiDirJet::linear(k, g_val, &g_first);

        // beta = g (signed slope allowed)
        let beta_jet = g_jet.clone();
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
        // The model domain is enforced on qd1 itself, which is the same quantity
        // constrained by the time monotonicity inequalities. Since c = sqrt(1+beta^2)
        // is strictly positive, qd1 >= guard implies ad1 > 0 as required by log(ad1).
        let qd1_lower = self.time_derivative_lower_bound();
        let qd1_val = qd1_jet.coeff(0);
        let ad1_val = ad1_jet.coeff(0);
        if qd1_val < qd1_lower {
            return Err(format!(
                "survival marginal-slope monotonicity violated at row {row}: raw time derivative={qd1_val:.3e} must be at least derivative_guard={qd1_lower:.3e}; transformed time derivative={ad1_val:.3e}"
            ));
        }
        let time_deriv_term = if di > 0.0 {
            ad1_jet
                .compose_unary(unary_derivatives_log(ad1_val))
                .scale(-wi * di)
        } else {
            MultiDirJet::zero(k)
        };

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

    /// Compute per-row primary gradient and Hessian using the closed-form
    /// scalar kernel.  The hot inner computation uses stack arrays only;
    /// conversion to Array1/Array2 happens once at the boundary for API
    /// compatibility with outer-derivative paths.
    fn compute_row_primary_gradient_hessian_uncached(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        let beta_time = &block_states[0].beta;
        let beta_marginal = &block_states[1].beta;
        let q0 = self.design_entry.row(row).dot(beta_time)
            + self.offset_entry[row]
            + self.marginal_design.dot_row(row, beta_marginal);
        let q1 = self.design_exit.row(row).dot(beta_time)
            + self.offset_exit[row]
            + self.marginal_design.dot_row(row, beta_marginal);
        let qd1 = self.design_derivative_exit.row(row).dot(beta_time)
            + self.derivative_offset_exit[row];
        let g = block_states[2].eta[row];
        let (nll, grad_arr, hess_arr) = row_primary_closed_form(
            q0,
            q1,
            qd1,
            g,
            self.z[row],
            self.weights[row],
            self.event[row],
            self.derivative_guard,
        )?;
        // Convert stack arrays to ndarray types at the boundary.
        let grad = Array1::from_vec(grad_arr.to_vec());
        let mut hess = Array2::zeros((N_PRIMARY, N_PRIMARY));
        for i in 0..N_PRIMARY {
            for j in 0..N_PRIMARY {
                hess[[i, j]] = hess_arr[i][j];
            }
        }
        Ok((nll, grad, hess))
    }

    fn build_eval_cache(&self, block_states: &[ParameterBlockState]) -> Result<EvalCache, String> {
        let slices = block_slices(block_states);
        let row_bases = (0..self.n)
            .map(|row| {
                let (_, gradient, hessian) =
                    self.compute_row_primary_gradient_hessian_uncached(row, block_states)?;
                Ok(RowPrimaryBase { gradient, hessian })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(EvalCache { slices, row_bases })
    }

    fn row_primary_gradient_hessian<'a>(
        &self,
        row: usize,
        cache: &'a EvalCache,
    ) -> (&'a Array1<f64>, &'a Array2<f64>) {
        let base = &cache.row_bases[row];
        (&base.gradient, &base.hessian)
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
    ) -> Result<Array1<f64>, String> {
        let mut out = Array1::<f64>::zeros(slices.total);
        // Time block: 3 primary scalars (q0, q1, qd1) all map to the same beta_time
        {
            let mut time = out.slice_mut(s![slices.time.clone()]);
            dense_row_axpy_into(&self.design_entry, row, primary_vec[0], &mut time);
            dense_row_axpy_into(&self.design_exit, row, primary_vec[1], &mut time);
            dense_row_axpy_into(&self.design_derivative_exit, row, primary_vec[2], &mut time);
        }
        // Baseline block: contributes equally to q0 and q1.
        {
            let mut marginal = out.slice_mut(s![slices.marginal.clone()]);
            self.marginal_design
                .axpy_row_into(row, primary_vec[0] + primary_vec[1], &mut marginal)?;
        }
        // Slope block: primary scalar g
        {
            let mut logslope = out.slice_mut(s![slices.logslope.clone()]);
            self.logslope_design
                .axpy_row_into(row, primary_vec[3], &mut logslope)?;
        }
        Ok(out)
    }

    /// Accumulate the pullback of a primary-space Hessian into coefficient-space.
    fn add_pullback_primary_hessian(
        &self,
        target: &mut Array2<f64>,
        row: usize,
        slices: &BlockSlices,
        primary_hessian: &Array2<f64>,
    ) {
        let mut tt = Array2::<f64>::zeros((slices.time.len(), slices.time.len()));
        dense_row_outer_into(&self.design_entry, &self.design_entry, row, primary_hessian[[0, 0]], &mut tt);
        dense_row_outer_into(&self.design_entry, &self.design_exit, row, primary_hessian[[0, 1]], &mut tt);
        dense_row_outer_into(
            &self.design_entry,
            &self.design_derivative_exit,
            row,
            primary_hessian[[0, 2]],
            &mut tt,
        );
        dense_row_outer_into(&self.design_exit, &self.design_entry, row, primary_hessian[[1, 0]], &mut tt);
        dense_row_outer_into(&self.design_exit, &self.design_exit, row, primary_hessian[[1, 1]], &mut tt);
        dense_row_outer_into(
            &self.design_exit,
            &self.design_derivative_exit,
            row,
            primary_hessian[[1, 2]],
            &mut tt,
        );
        dense_row_outer_into(
            &self.design_derivative_exit,
            &self.design_entry,
            row,
            primary_hessian[[2, 0]],
            &mut tt,
        );
        dense_row_outer_into(
            &self.design_derivative_exit,
            &self.design_exit,
            row,
            primary_hessian[[2, 1]],
            &mut tt,
        );
        dense_row_outer_into(
            &self.design_derivative_exit,
            &self.design_derivative_exit,
            row,
            primary_hessian[[2, 2]],
            &mut tt,
        );
        target
            .slice_mut(s![slices.time.clone(), slices.time.clone()])
            .scaled_add(1.0, &tt);

        let marginal_row = self.marginal_design.row_chunk(row..row + 1).row(0).to_owned();
        let logslope_row = self.logslope_design.row_chunk(row..row + 1).row(0).to_owned();
        let mm = marginal_row
            .view()
            .insert_axis(Axis(1))
            .dot(&marginal_row.view().insert_axis(Axis(0)))
            * (primary_hessian[[0, 0]]
                + primary_hessian[[0, 1]]
                + primary_hessian[[1, 0]]
                + primary_hessian[[1, 1]]);
        target
            .slice_mut(s![slices.marginal.clone(), slices.marginal.clone()])
            .scaled_add(1.0, &mm);

        let mg = marginal_row
            .view()
            .insert_axis(Axis(1))
            .dot(&logslope_row.view().insert_axis(Axis(0)))
            * (primary_hessian[[0, 3]] + primary_hessian[[1, 3]]);
        target
            .slice_mut(s![slices.marginal.clone(), slices.logslope.clone()])
            .scaled_add(1.0, &mg);
        target
            .slice_mut(s![slices.logslope.clone(), slices.marginal.clone()])
            .scaled_add(1.0, &mg.t().to_owned());

        let tm_g = [
            (&self.design_entry, primary_hessian[[0, 3]]),
            (&self.design_exit, primary_hessian[[1, 3]]),
            (&self.design_derivative_exit, primary_hessian[[2, 3]]),
        ];
        let mut tg = Array2::<f64>::zeros((slices.time.len(), slices.logslope.len()));
        for (design, alpha) in tm_g {
            if alpha == 0.0 {
                continue;
            }
            let lhs = design.row(row);
            for i in 0..lhs.len() {
                let xi = lhs[i];
                if xi == 0.0 {
                    continue;
                }
                for j in 0..logslope_row.len() {
                    tg[[i, j]] += alpha * xi * logslope_row[j];
                }
            }
        }
        target
            .slice_mut(s![slices.time.clone(), slices.logslope.clone()])
            .scaled_add(1.0, &tg);
        target
            .slice_mut(s![slices.logslope.clone(), slices.time.clone()])
            .scaled_add(1.0, &tg.t().to_owned());

        let tm_m = [
            (&self.design_entry, primary_hessian[[0, 0]] + primary_hessian[[0, 1]]),
            (&self.design_exit, primary_hessian[[1, 0]] + primary_hessian[[1, 1]]),
            (&self.design_derivative_exit, primary_hessian[[2, 0]] + primary_hessian[[2, 1]]),
        ];
        let mut tm = Array2::<f64>::zeros((slices.time.len(), slices.marginal.len()));
        for (design, alpha) in tm_m {
            if alpha == 0.0 {
                continue;
            }
            let lhs = design.row(row);
            for i in 0..lhs.len() {
                let xi = lhs[i];
                if xi == 0.0 {
                    continue;
                }
                for j in 0..marginal_row.len() {
                    tm[[i, j]] += alpha * xi * marginal_row[j];
                }
            }
        }
        target
            .slice_mut(s![slices.time.clone(), slices.marginal.clone()])
            .scaled_add(1.0, &tm);
        target
            .slice_mut(s![slices.marginal.clone(), slices.time.clone()])
            .scaled_add(1.0, &tm.t().to_owned());

        let mut gg = Array2::<f64>::zeros((slices.logslope.len(), slices.logslope.len()));
        self.logslope_design
            .syr_row_into(row, primary_hessian[[3, 3]], &mut gg)
            .expect("survival logslope syr_row_into should match block dimensions");
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
        let d_time = d_beta_flat.slice(s![slices.time.clone()]);
        out[0] = self.design_entry.row(row).dot(&d_time);
        out[0] += self
            .marginal_design
            .dot_row_view(row, d_beta_flat.slice(s![slices.marginal.clone()]));
        out[1] = self.design_exit.row(row).dot(&d_time);
        out[1] += self
            .marginal_design
            .dot_row_view(row, d_beta_flat.slice(s![slices.marginal.clone()]));
        out[2] = self.design_derivative_exit.row(row).dot(&d_time);
        out[3] = self
            .logslope_design
            .dot_row_view(row, d_beta_flat.slice(s![slices.logslope.clone()]));
        out
    }

    // ── Dense exact Hessian (streaming, uncached) ─────────────────────

    fn joint_hessian_dense_streaming(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Array2<f64>, String> {
        let slices = block_slices(block_states);
        let mut hessian = Array2::<f64>::zeros((slices.total, slices.total));
        for row in 0..self.n {
            let (_, _, f_pipi) = self.compute_row_primary_gradient_hessian_uncached(row, block_states)?;
            self.add_pullback_primary_hessian(&mut hessian, row, &slices, &f_pipi);
        }
        Ok(hessian)
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

    fn joint_hessian_matvec_from_cache(
        &self,
        direction: &Array1<f64>,
        cache: &EvalCache,
    ) -> Result<Array1<f64>, String> {
        let slices = &cache.slices;
        let mut out = Array1::<f64>::zeros(slices.total);
        for row in 0..self.n {
            let row_dir = self.row_primary_direction_from_flat(row, slices, direction);
            let (_, row_hessian) = self.row_primary_gradient_hessian(row, cache);
            let row_action = row_hessian.dot(&row_dir);
            out += &self.pullback_primary_vector(row, slices, &row_action)?;
        }
        Ok(out)
    }

    fn joint_hessian_diagonal_from_cache(&self, cache: &EvalCache) -> Result<Array1<f64>, String> {
        let slices = &cache.slices;
        let mut diagonal = Array1::<f64>::zeros(slices.total);
        for row in 0..self.n {
            let (_, row_hessian) = self.row_primary_gradient_hessian(row, cache);

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

            let marginal_row = self.marginal_design.row_chunk(row..row + 1);
            let marginal_row = marginal_row.row(0);
            for (local_idx, &value) in marginal_row.iter().enumerate() {
                diagonal[slices.marginal.start + local_idx] +=
                    value * value * (row_hessian[[0, 0]] + 2.0 * row_hessian[[0, 1]] + row_hessian[[1, 1]]);
            }

            let logslope_row = self.logslope_design.row_chunk(row..row + 1);
            let logslope_row = logslope_row.row(0);
            for (local_idx, &value) in logslope_row.iter().enumerate() {
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
    ///
    /// Only the baseline and slope surface blocks can carry spatial parameters.
    /// The time block (block 0) is a pure monotone time basis with no spatial terms.
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
            1 => {
                let x_row = self.psi_design_row_vector(
                    row,
                    deriv,
                    self.n,
                    self.marginal_design.ncols(),
                    "SurvivalMarginalSlope marginal",
                )?;
                let value = x_row.dot(&block_states[1].beta);
                out[0] = value;
                out[1] = value;
            }
            2 => {
                let x_row = self.psi_design_row_vector(
                    row,
                    deriv,
                    self.n,
                    self.logslope_design.ncols(),
                    "SurvivalMarginalSlope logslope",
                )?;
                out[3] = x_row.dot(&block_states[2].beta);
            }
            _ => {
                return Err(format!(
                    "survival marginal-slope psi: only baseline/slope spatial blocks are supported, got block {block_idx}"
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
            1 => {
                let x_row = self.psi_design_row_vector(
                    row,
                    deriv,
                    self.n,
                    self.marginal_design.ncols(),
                    "SurvivalMarginalSlope marginal",
                )?;
                let value = x_row.dot(&d_beta_flat.slice(s![slices.marginal.clone()]).to_owned());
                out[0] = value;
                out[1] = value;
            }
            2 => {
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
                    "survival marginal-slope psi action: only baseline/slope spatial blocks are supported, got block {block_idx}"
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
            1 => {
                let x_row = self.psi_second_design_row_vector(
                    row,
                    deriv_i,
                    &derivative_blocks[block_j][local_j],
                    local_j,
                    self.n,
                    self.marginal_design.ncols(),
                    "SurvivalMarginalSlope marginal",
                )?;
                let value = x_row.dot(&block_states[1].beta);
                out[0] = value;
                out[1] = value;
            }
            2 => {
                let x_row = self.psi_second_design_row_vector(
                    row,
                    deriv_i,
                    &derivative_blocks[block_j][local_j],
                    local_j,
                    self.n,
                    self.logslope_design.ncols(),
                    "SurvivalMarginalSlope logslope",
                )?;
                out[3] = x_row.dot(&block_states[2].beta);
            }
            _ => {
                return Err(format!(
                    "survival marginal-slope psi second: only baseline/slope spatial blocks are supported, got block {block_i}"
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
            1 => out
                .slice_mut(s![slices.marginal.clone()])
                .assign(&self.psi_design_row_vector(
                    row,
                    deriv,
                    self.n,
                    self.marginal_design.ncols(),
                    "SurvivalMarginalSlope marginal",
                )?),
            2 => out
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
                    "survival marginal-slope psi embedding: only baseline/slope spatial blocks are supported, got block {block_idx}"
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
            1 => out.slice_mut(s![slices.marginal.clone()]).assign(
                &self.psi_second_design_row_vector(
                    row,
                    deriv_i,
                    &derivative_blocks[block_j][local_j],
                    local_j,
                    self.n,
                    self.marginal_design.ncols(),
                    "SurvivalMarginalSlope marginal",
                )?,
            ),
            2 => out.slice_mut(s![slices.logslope.clone()]).assign(
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
                    "survival marginal-slope psi second embedding: only baseline/slope spatial blocks are supported, got block {block_i}"
                ));
            }
        }
        Ok(Some((block_i, out)))
    }

    // ── Psi terms (first and second order) ────────────────────────────

    fn psi_terms_inner(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        cache: Option<&EvalCache>,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        let slices = block_slices(block_states);
        let Some((block_idx, _)) =
            self.embedded_psi_vector(0, &slices, derivative_blocks, psi_index)?
        else {
            return Ok(None);
        };
        let loading = spatial_block_primary_loading(block_idx)?;
        let mut objective_psi = 0.0;
        let mut score_psi = Array1::<f64>::zeros(slices.total);
        let mut hessian_psi = Array2::<f64>::zeros((slices.total, slices.total));
        for row in 0..self.n {
            let Some(dir) =
                self.row_primary_psi_direction(row, block_states, derivative_blocks, psi_index)?
            else {
                continue;
            };
            let (f_pi, f_pipi) = if let Some(c) = cache {
                let (g, h) = self.row_primary_gradient_hessian(row, c);
                (g.clone(), h.clone())
            } else {
                let (_, g, h) =
                    self.compute_row_primary_gradient_hessian_uncached(row, block_states)?;
                (g, h)
            };
            let third = self.row_primary_third_contracted(row, block_states, &dir)?;
            let (_, left_vec) = self
                .embedded_psi_vector(row, &slices, derivative_blocks, psi_index)?
                .ok_or_else(|| "missing survival marginal-slope psi vector".to_string())?;
            objective_psi += f_pi.dot(&dir);
            score_psi += &(left_vec.clone() * f_pi.dot(&loading));
            score_psi += &self.pullback_primary_vector(row, &slices, &f_pipi.dot(&dir))?;

            let right_vec = self.pullback_primary_vector(row, &slices, &f_pipi.dot(&loading))?;
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

    fn psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        self.psi_terms_inner(block_states, derivative_blocks, psi_index, None)
    }

    fn psi_second_order_terms_inner(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
        cache: Option<&EvalCache>,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        let slices = block_slices(block_states);
        let Some((_, _)) = self.embedded_psi_vector(0, &slices, derivative_blocks, psi_i)? else {
            return Ok(None);
        };
        let Some((_, _)) = self.embedded_psi_vector(0, &slices, derivative_blocks, psi_j)? else {
            return Ok(None);
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
            let (f_pi, f_pipi) = if let Some(c) = cache {
                let (g, h) = self.row_primary_gradient_hessian(row, c);
                (g.clone(), h.clone())
            } else {
                let (_, g, h) =
                    self.compute_row_primary_gradient_hessian_uncached(row, block_states)?;
                (g, h)
            };
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
            let loading_i = spatial_block_primary_loading(
                self.resolve_psi_location(derivative_blocks, psi_i)
                    .map(|(idx, _)| idx)
                    .ok_or_else(|| "missing psi_i block".to_string())?,
            )?;
            let loading_j = spatial_block_primary_loading(
                self.resolve_psi_location(derivative_blocks, psi_j)
                    .map(|(idx, _)| idx)
                    .ok_or_else(|| "missing psi_j block".to_string())?,
            )?;

            objective_psi_psi += dir_i.dot(&f_pipi.dot(&dir_j)) + f_pi.dot(&dir_ij);

            if left_ij.iter().any(|v| v.abs() > 0.0) {
                score_psi_psi += &(left_ij.clone() * f_pi.dot(&loading_i));
            }
            score_psi_psi += &(left_i.clone() * loading_i.dot(&f_pipi.dot(&dir_j)));
            score_psi_psi += &(left_j.clone() * loading_j.dot(&f_pipi.dot(&dir_i)));
            score_psi_psi +=
                &self.pullback_primary_vector(row, &slices, &f_pipi.dot(&dir_ij))?;
            score_psi_psi +=
                &self.pullback_primary_vector(row, &slices, &third_i.dot(&dir_j))?;

            if left_ij.iter().any(|v| v.abs() > 0.0) {
                let right_ij =
                    self.pullback_primary_vector(row, &slices, &f_pipi.dot(&loading_i))?;
                hessian_psi_psi += &left_ij
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&right_ij.view().insert_axis(Axis(0)));
                hessian_psi_psi += &right_ij
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&left_ij.view().insert_axis(Axis(0)));
            }

            let scalar_ij = loading_i.dot(&f_pipi.dot(&loading_j));
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
                self.pullback_primary_vector(row, &slices, &third_j.t().dot(&loading_i))?;
            hessian_psi_psi += &left_i
                .view()
                .insert_axis(Axis(1))
                .dot(&right_i.view().insert_axis(Axis(0)));
            hessian_psi_psi += &right_i
                .view()
                .insert_axis(Axis(1))
                .dot(&left_i.view().insert_axis(Axis(0)));

            let right_j =
                self.pullback_primary_vector(row, &slices, &third_i.t().dot(&loading_j))?;
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

    fn psi_second_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        self.psi_second_order_terms_inner(block_states, derivative_blocks, psi_i, psi_j, None)
    }

    fn psi_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let slices = block_slices(block_states);
        let Some((_, _)) = self.embedded_psi_vector(0, &slices, derivative_blocks, psi_index)?
        else {
            return Ok(None);
        };
        let block_idx = self
            .embedded_psi_vector(0, &slices, derivative_blocks, psi_index)?
            .map(|(idx, _)| idx)
            .ok_or_else(|| "missing psi block".to_string())?;
        let loading = spatial_block_primary_loading(block_idx)?;
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
                self.pullback_primary_vector(row, &slices, &third_beta.t().dot(&loading))?;
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
    cache: EvalCache,
}

struct SurvivalMarginalSlopePsiWorkspace {
    family: SurvivalMarginalSlopeFamily,
    block_states: Vec<ParameterBlockState>,
    derivative_blocks: Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>>,
    cache: EvalCache,
}

impl SurvivalMarginalSlopeHessianWorkspace {
    fn new(
        family: SurvivalMarginalSlopeFamily,
        block_states: Vec<ParameterBlockState>,
    ) -> Result<Self, String> {
        let cache = family.build_eval_cache(&block_states)?;
        Ok(Self {
            family,
            block_states,
            cache,
        })
    }
}

impl SurvivalMarginalSlopePsiWorkspace {
    fn new(
        family: SurvivalMarginalSlopeFamily,
        block_states: Vec<ParameterBlockState>,
        derivative_blocks: Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>>,
    ) -> Result<Self, String> {
        let cache = family.build_eval_cache(&block_states)?;
        Ok(Self {
            family,
            block_states,
            derivative_blocks,
            cache,
        })
    }
}

impl ExactNewtonJointHessianWorkspace for SurvivalMarginalSlopeHessianWorkspace {
    fn hessian_matvec(&self, beta_flat: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        self.family
            .joint_hessian_matvec_from_cache(beta_flat, &self.cache)
            .map(Some)
    }

    fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        self.family
            .joint_hessian_diagonal_from_cache(&self.cache)
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
        self.family.psi_terms_inner(
            &self.block_states,
            &self.derivative_blocks,
            psi_index,
            Some(&self.cache),
        )
    }

    fn second_order_terms(
        &self,
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        self.family.psi_second_order_terms_inner(
            &self.block_states,
            &self.derivative_blocks,
            psi_i,
            psi_j,
            Some(&self.cache),
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

impl SurvivalMarginalSlopeFamily {
    fn evaluate_blockwise_exact_newton(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<FamilyEvaluation, String> {
        let slices = block_slices(block_states);
        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();

        // Parallel accumulation: each worker accumulates into its own
        // block-local gradient + Hessian arrays, then we reduce by addition.
        // This avoids both per-chunk Vec allocation AND serial bottlenecks.
        type Acc = (
            f64,           // ll
            Array1<f64>,   // grad_time
            Array1<f64>,   // grad_marginal
            Array1<f64>,   // grad_logslope
            Array2<f64>,   // hess_time
            Array2<f64>,   // hess_marginal
            Array2<f64>,   // hess_logslope
        );

        let make_acc = || -> Acc {
            (
                0.0,
                Array1::zeros(p_t),
                Array1::zeros(p_m),
                Array1::zeros(p_g),
                Array2::zeros((p_t, p_t)),
                Array2::zeros((p_m, p_m)),
                Array2::zeros((p_g, p_g)),
            )
        };
        let (ll, grad_time, grad_marginal, grad_logslope, hess_time, hess_marginal, hess_logslope) =
            (0..self.n)
                .into_par_iter()
                .fold(make_acc, |mut acc, row| {
                    let (row_nll, f_pi, f_pipi) = match self
                        .compute_row_primary_gradient_hessian_uncached(row, block_states)
                    {
                        Ok(v) => v,
                        Err(_) => return acc,
                    };
                    acc.0 -= row_nll;

                    {
                        let mut time = acc.1.view_mut();
                        dense_row_axpy_into(&self.design_entry, row, -f_pi[0], &mut time);
                        dense_row_axpy_into(&self.design_exit, row, -f_pi[1], &mut time);
                        dense_row_axpy_into(
                            &self.design_derivative_exit,
                            row,
                            -f_pi[2],
                            &mut time,
                        );
                    }
                    self.marginal_design
                        .axpy_row_into(row, -(f_pi[0] + f_pi[1]), &mut acc.2.view_mut())
                        .expect("survival marginal block axpy should match block dimensions");
                    self.logslope_design
                        .axpy_row_into(row, -f_pi[3], &mut acc.3.view_mut())
                        .expect("survival logslope block axpy should match block dimensions");

                    let designs = [
                        &self.design_entry,
                        &self.design_exit,
                        &self.design_derivative_exit,
                    ];
                    for a in 0..3 {
                        for b in 0..3 {
                            dense_row_outer_into(
                                designs[a], designs[b], row, f_pipi[[a, b]], &mut acc.4,
                            );
                        }
                    }

                    self.marginal_design
                        .syr_row_into(
                            row,
                            f_pipi[[0, 0]] + f_pipi[[0, 1]] + f_pipi[[1, 0]] + f_pipi[[1, 1]],
                            &mut acc.5,
                        )
                        .expect("survival marginal block syr should match block dimensions");
                    self.logslope_design
                        .syr_row_into(row, f_pipi[[3, 3]], &mut acc.6)
                        .expect("survival logslope block syr should match block dimensions");
                    acc
                })
                .reduce(make_acc, |mut a, b| {
                    a.0 += b.0;
                    a.1 += &b.1;
                    a.2 += &b.2;
                    a.3 += &b.3;
                    a.4 += &b.4;
                    a.5 += &b.5;
                    a.6 += &b.6;
                    a
                });

        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: vec![
                BlockWorkingSet::ExactNewton {
                    gradient: grad_time,
                    hessian: SymmetricMatrix::Dense(hess_time),
                },
                BlockWorkingSet::ExactNewton {
                    gradient: grad_marginal,
                    hessian: SymmetricMatrix::Dense(hess_marginal),
                },
                BlockWorkingSet::ExactNewton {
                    gradient: grad_logslope,
                    hessian: SymmetricMatrix::Dense(hess_logslope),
                },
            ],
        })
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
        self.evaluate_blockwise_exact_newton(block_states)
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        // Fast path: just compute NLL without gradient/Hessian.
        // Avoids build_eval_cache which computes per-row gradient+Hessian
        // (14 jet evaluations per row) that log_likelihood_only never uses.
        let mut ll = 0.0;
        for i in 0..self.n {
            let row_neglog = self.row_neglog_directional(i, block_states, &[])?;
            ll -= row_neglog;
        }
        Ok(ll)
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        self.joint_hessian_dense_streaming(block_states).map(Some)
    }

    fn requires_joint_outer_hyper_path(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        Ok(Some(Arc::new(SurvivalMarginalSlopeHessianWorkspace::new(
            self.clone(),
            block_states.to_vec(),
        )?)))
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
        Ok(Some(Arc::new(SurvivalMarginalSlopePsiWorkspace::new(
            self.clone(),
            block_states.to_vec(),
            derivative_blocks.to_vec(),
        )?)))
    }

    fn block_linear_constraints(
        &self,
        _: &[ParameterBlockState],
        block_idx: usize,
        _: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        if block_idx == 0 {
            Ok(self.time_linear_constraints.clone())
        } else {
            Ok(None)
        }
    }
}

// ── Building block specs ──────────────────────────────────────────────

fn build_time_blockspec(
    time_block: &TimeBlockInput,
    design_exit: &Arc<Array2<f64>>,
    rho: Array1<f64>,
    beta_hint: Option<Array1<f64>>,
) -> ParameterBlockSpec {
    ParameterBlockSpec {
        name: "time_surface".to_string(),
        design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::Materialized(Arc::clone(
            design_exit,
        ))),
        offset: Array1::zeros(design_exit.nrows()),
        penalties: time_block
            .penalties
            .iter()
            .cloned()
            .map(PenaltyMatrix::Dense)
            .collect(),
        nullspace_dims: time_block.nullspace_dims.clone(),
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
        design: design.design.clone(),
        offset: Array1::from_elem(design.design.nrows(), baseline),
        penalties: design.penalties_as_penalty_matrix(),
        nullspace_dims: design.nullspace_dims.clone(),
        initial_log_lambdas: rho,
        initial_beta: beta_hint,
    }
}

fn build_marginal_blockspec(
    design: &TermCollectionDesign,
    rho: Array1<f64>,
    beta_hint: Option<Array1<f64>>,
) -> ParameterBlockSpec {
    ParameterBlockSpec {
        name: "marginal_surface".to_string(),
        design: design.design.clone(),
        offset: Array1::zeros(design.design.nrows()),
        penalties: design.penalties_as_penalty_matrix(),
        nullspace_dims: design.nullspace_dims.clone(),
        initial_log_lambdas: rho,
        initial_beta: beta_hint,
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
    marginalspec: &TermCollectionSpec,
    marginal_penalties: usize,
    logslopespec: &TermCollectionSpec,
    logslope_penalties: usize,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> ExactJointHyperSetup {
    let marginal_terms = spatial_length_scale_term_indices(marginalspec);
    let logslope_terms = spatial_length_scale_term_indices(logslopespec);
    let rho_dim = time_penalties + marginal_penalties + logslope_penalties;
    let rho0vec = Array1::<f64>::zeros(rho_dim);
    let rho_lower = Array1::<f64>::from_elem(rho_dim, -12.0);
    let rho_upper = Array1::<f64>::from_elem(rho_dim, 12.0);
    // Time block has no spatial length scales (pure B-spline on time)
    let empty_kappa = SpatialLogKappaCoords::new_with_dims(Array1::zeros(0), vec![]);
    let marginal_kappa = SpatialLogKappaCoords::from_length_scales_aniso(
        marginalspec,
        &marginal_terms,
        kappa_options,
    );
    let logslope_kappa = SpatialLogKappaCoords::from_length_scales_aniso(
        logslopespec,
        &logslope_terms,
        kappa_options,
    );
    let mut values = empty_kappa.as_array().to_vec();
    values.extend(marginal_kappa.as_array().iter());
    values.extend(logslope_kappa.as_array().iter());
    let mut dims = empty_kappa.dims_per_term().to_vec();
    dims.extend(marginal_kappa.dims_per_term());
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

fn validate_standardized_z(
    z: &Array1<f64>,
    weights: &Array1<f64>,
    context: &str,
) -> Result<(), String> {
    let weight_sum = weights.iter().copied().sum::<f64>();
    if !(weight_sum.is_finite() && weight_sum > 0.0) {
        return Err(format!("{context} requires positive finite total weight"));
    }
    let mean = z
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| wi * zi)
        .sum::<f64>()
        / weight_sum;
    let var = z
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| wi * (zi - mean) * (zi - mean))
        .sum::<f64>()
        / weight_sum;
    let sd = var.sqrt();
    if mean.abs() > 1e-6 || (sd - 1.0).abs() > 1e-6 {
        return Err(format!(
            "{context} requires z to be pre-standardized to weighted mean 0 and weighted sd 1; got mean={mean:.6e}, sd={sd:.6e}"
        ));
    }
    Ok(())
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
    validate_standardized_z(
        &spec.z,
        &spec.weights,
        "survival-marginal-slope",
    )?;
    if spec.event_target.iter().any(|&d| d != 0.0 && d != 1.0) {
        return Err(
            "survival-marginal-slope requires binary event indicators (0.0 or 1.0)".to_string(),
        );
    }
    if !spec.derivative_guard.is_finite() || spec.derivative_guard <= 0.0 {
        return Err(format!(
            "survival-marginal-slope requires derivative_guard > 0, got {}",
            spec.derivative_guard
        ));
    }
    for i in 0..n {
        if spec.age_exit[i] < spec.age_entry[i] {
            return Err(format!(
                "survival-marginal-slope row {i}: exit time ({}) < entry time ({})",
                spec.age_exit[i], spec.age_entry[i]
            ));
        }
    }
    let n_entry = spec.time_block.design_entry.nrows();
    let n_exit = spec.time_block.design_exit.nrows();
    let n_deriv = spec.time_block.design_derivative_exit.nrows();
    if n_entry != n || n_exit != n || n_deriv != n {
        return Err(format!(
            "survival-marginal-slope time block design row mismatch: \
             data={n}, design_entry={n_entry}, design_exit={n_exit}, design_derivative_exit={n_deriv}"
        ));
    }
    let p_entry = spec.time_block.design_entry.ncols();
    let p_exit = spec.time_block.design_exit.ncols();
    let p_deriv = spec.time_block.design_derivative_exit.ncols();
    if p_exit != p_entry || p_deriv != p_entry {
        return Err(format!(
            "survival-marginal-slope time block design column mismatch: entry={p_entry}, exit={p_exit}, deriv={p_deriv}"
        ));
    }
    if !spec.time_block.structural_monotonicity {
        return Err(
            "survival-marginal-slope requires structural time monotonicity by construction; non-structural time transforms are no longer supported"
                .to_string(),
        );
    }
    Ok(())
}

/// Compute a simple baseline slope from the actual survival marginal-slope likelihood,
/// using the baseline offsets alone as a time-only pilot q(t).
fn pooled_survival_baseline(
    event: &Array1<f64>,
    weights: &Array1<f64>,
    z: &Array1<f64>,
    q0: &Array1<f64>,
    q1: &Array1<f64>,
    qd1: &Array1<f64>,
) -> f64 {
    let n = event.len();
    if n == 0 {
        return 0.0;
    }
    let objective = |slope: f64| -> f64 {
        let mut total = 0.0;
        for i in 0..n {
            let wi = weights[i];
            let di = event[i];
            let zi = z[i];
            let c = (1.0 + slope * slope).sqrt();
            let eta0 = q0[i] * c + slope * zi;
            let eta1 = q1[i] * c + slope * zi;
            let ad1 = qd1[i] * c;
            if !eta0.is_finite() || !eta1.is_finite() || !ad1.is_finite() || ad1 <= 0.0 {
                return f64::INFINITY;
            }
            total += wi * (1.0 - di) * unary_derivatives_neglog_phi(-eta1, 1.0)[0];
            total -= wi * unary_derivatives_neglog_phi(-eta0, 1.0)[0];
            if di > 0.0 {
                total -= wi * unary_derivatives_log_normal_pdf(eta1)[0];
                total -= wi * ad1.ln();
            }
        }
        total
    };
    let mut best_slope = 0.0;
    let mut best_obj = f64::INFINITY;
    for step in -160..=160 {
        let slope = step as f64 * 0.05;
        let obj = objective(slope);
        if obj.is_finite() && obj < best_obj {
            best_obj = obj;
            best_slope = slope;
        }
    }
    best_slope
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
    let baseline_slope = pooled_survival_baseline(
        &spec.event_target,
        &spec.weights,
        &spec.z,
        &spec.time_block.offset_entry,
        &spec.time_block.offset_exit,
        &spec.time_block.derivative_offset_exit,
    );

    let marginal_design =
        build_term_collection_design(data, &spec.marginalspec).map_err(|e| e.to_string())?;
    let marginalspec_boot =
        freeze_spatial_length_scale_terms_from_design(&spec.marginalspec, &marginal_design)
            .map_err(|e| e.to_string())?;
    let logslope_design =
        build_term_collection_design(data, &spec.logslopespec).map_err(|e| e.to_string())?;
    let logslopespec_boot =
        freeze_spatial_length_scale_terms_from_design(&spec.logslopespec, &logslope_design)
            .map_err(|e| e.to_string())?;

    let time_penalties_len = spec.time_block.penalties.len();
    let setup = joint_setup(
        time_penalties_len,
        &marginalspec_boot,
        marginal_design.penalties.len(),
        &logslopespec_boot,
        logslope_design.penalties.len(),
        kappa_options,
    );

    let hints = RefCell::new((
        None::<Array1<f64>>,
        None::<Array1<f64>>,
        None::<Array1<f64>>,
    ));
    let exact_warm_start = RefCell::new(None::<CustomFamilyWarmStart>);

    let event = Arc::new(spec.event_target.clone());
    let weights = Arc::new(spec.weights.clone());
    let z = Arc::new(spec.z.clone());
    let derivative_guard = spec.derivative_guard;
    let design_entry = Arc::new(spec.time_block.design_entry.clone());
    let design_exit = Arc::new(spec.time_block.design_exit.clone());
    let design_derivative_exit = Arc::new(spec.time_block.design_derivative_exit.clone());
    let offset_entry = Arc::new(spec.time_block.offset_entry.clone());
    let offset_exit = Arc::new(spec.time_block.offset_exit.clone());
    let derivative_offset_exit = Arc::new(spec.time_block.derivative_offset_exit.clone());
    let time_block_ref = spec.time_block.clone();
    let time_linear_constraints =
        structural_nonnegative_constraints(&Array2::eye(design_exit.ncols()));

    let make_family = |marginal_design: &TermCollectionDesign,
                       logslope_design: &TermCollectionDesign|
     -> SurvivalMarginalSlopeFamily {
        SurvivalMarginalSlopeFamily {
            n,
            event: Arc::clone(&event),
            weights: Arc::clone(&weights),
            z: Arc::clone(&z),
            derivative_guard,
            design_entry: Arc::clone(&design_entry),
            design_exit: Arc::clone(&design_exit),
            design_derivative_exit: Arc::clone(&design_derivative_exit),
            offset_entry: Arc::clone(&offset_entry),
            offset_exit: Arc::clone(&offset_exit),
            derivative_offset_exit: Arc::clone(&derivative_offset_exit),
            marginal_design: marginal_design.design.clone(),
            logslope_design: logslope_design.design.clone(),
            time_linear_constraints: time_linear_constraints.clone(),
        }
    };

    let build_blocks = |rho: &Array1<f64>,
                        marginal_design: &TermCollectionDesign,
                        logslope_design: &TermCollectionDesign|
     -> Result<Vec<ParameterBlockSpec>, String> {
        let hints = hints.borrow();
        let mut cursor = 0usize;
        let rho_time = rho
            .slice(s![cursor..cursor + time_penalties_len])
            .to_owned();
        cursor += time_penalties_len;
        let rho_marginal = rho
            .slice(s![cursor..cursor + marginal_design.penalties.len()])
            .to_owned();
        cursor += marginal_design.penalties.len();
        let rho_logslope = rho
            .slice(s![cursor..cursor + logslope_design.penalties.len()])
            .to_owned();
        Ok(vec![
            build_time_blockspec(&time_block_ref, &design_exit, rho_time, hints.0.clone()),
            build_marginal_blockspec(marginal_design, rho_marginal, hints.1.clone()),
            build_logslope_blockspec(
                logslope_design,
                baseline_slope,
                rho_logslope,
                hints.2.clone(),
            ),
        ])
    };

    // ── Pilot fit: rigid (zero-penalty) to seed coefficients ────────────
    {
        let rigid_rho = Array1::<f64>::zeros(
            time_penalties_len + marginal_design.penalties.len() + logslope_design.penalties.len(),
        );
        let rigid_blocks = build_blocks(&rigid_rho, &marginal_design, &logslope_design)?;
        let rigid_family = make_family(&marginal_design, &logslope_design);
        if let Ok(rigid_fit) = inner_fit(&rigid_family, &rigid_blocks, options) {
            let mut hints_mut = hints.borrow_mut();
            if let Some(block) = rigid_fit.block_states.get(0) {
                hints_mut.0 = Some(block.beta.clone());
            }
            if let Some(block) = rigid_fit.block_states.get(1) {
                hints_mut.1 = Some(block.beta.clone());
            }
            if let Some(block) = rigid_fit.block_states.get(2) {
                hints_mut.2 = Some(block.beta.clone());
            }
        }
    }

    // Check analytic derivatives
    let marginal_derivatives = build_block_spatial_psi_derivatives(
        data,
        &marginalspec_boot,
        &marginal_design,
    )?;
    let logslope_derivatives = build_block_spatial_psi_derivatives(
        data,
        &logslopespec_boot,
        &logslope_design,
    )?;
    let analytic_joint_derivatives_available = marginal_derivatives.is_some()
        || logslope_derivatives.is_some()
        || setup.log_kappa_dim() == 0;

    if setup.log_kappa_dim() > 0
        && !(marginal_derivatives.is_some() || logslope_derivatives.is_some())
    {
        return Err(
            "exact survival marginal-slope spatial optimization requires analytic joint psi derivatives"
                .to_string(),
        );
    }

    let initial_rho = setup.theta0().slice(s![..setup.rho_dim()]).to_owned();
    let initial_blocks = build_blocks(&initial_rho, &marginal_design, &logslope_design)?;
    let initial_family = make_family(&marginal_design, &logslope_design);
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

    // Only the baseline and slope surface blocks can have spatial terms
    let marginal_terms = spatial_length_scale_term_indices(&marginalspec_boot);
    let logslope_terms = spatial_length_scale_term_indices(&logslopespec_boot);
    let solved = optimize_spatial_length_scale_exact_joint(
        data,
        &[marginalspec_boot.clone(), logslopespec_boot.clone()],
        &[marginal_terms, logslope_terms],
        kappa_options,
        &setup,
        analytic_joint_gradient_available,
        analytic_joint_hessian_available,
        |rho, _: &[TermCollectionSpec], designs: &[TermCollectionDesign]| {
            let blocks = build_blocks(rho, &designs[0], &designs[1])?;
            let family = make_family(&designs[0], &designs[1]);
            let fit = inner_fit(&family, &blocks, options)?;
            let mut hints_mut = hints.borrow_mut();
            if let Some(block) = fit.block_states.get(0) {
                hints_mut.0 = Some(block.beta.clone());
            }
            if let Some(block) = fit.block_states.get(1) {
                hints_mut.1 = Some(block.beta.clone());
            }
            if let Some(block) = fit.block_states.get(2) {
                hints_mut.2 = Some(block.beta.clone());
            }
            Ok(fit)
        },
        |rho, specs: &[TermCollectionSpec], designs: &[TermCollectionDesign], need_hessian| {
            let blocks = build_blocks(rho, &designs[0], &designs[1])?;
            let family = make_family(&designs[0], &designs[1]);
            let derivative_blocks = vec![
                Vec::new(),
                build_block_spatial_psi_derivatives(data, &specs[0], &designs[0])?
                    .unwrap_or_default(),
                build_block_spatial_psi_derivatives(data, &specs[1], &designs[1])?
                    .unwrap_or_default(),
            ];
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
        marginalspec_resolved: resolved_specs.remove(0),
        logslopespec_resolved: resolved_specs.remove(0),
        marginal_design: designs[0].clone(),
        logslope_design: designs[1].clone(),
        baseline_slope,
        time_block_penalties_len: time_penalties_len,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn empty_termspec() -> TermCollectionSpec {
        TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![],
        }
    }

    fn base_time_block() -> TimeBlockInput {
        TimeBlockInput {
            design_entry: Array2::zeros((1, 1)),
            design_exit: Array2::zeros((1, 1)),
            design_derivative_exit: Array2::ones((1, 1)),
            offset_entry: Array1::zeros(1),
            offset_exit: Array1::zeros(1),
            derivative_offset_exit: Array1::zeros(1),
            structural_monotonicity: true,
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: None,
            initial_beta: Some(Array1::zeros(1)),
        }
    }

    #[test]
    fn validate_spec_rejects_nonstructural_time_block() {
        let spec = SurvivalMarginalSlopeTermSpec {
            age_entry: array![0.0],
            age_exit: array![1.0],
            event_target: array![0.0],
            weights: array![1.0],
            z: array![0.0],
            marginalspec: empty_termspec(),
            derivative_guard: 1e-4,
            time_block: TimeBlockInput {
                structural_monotonicity: false,
                ..base_time_block()
            },
            logslopespec: empty_termspec(),
        };

        let err = validate_spec(&spec).expect_err("non-structural time block should fail");
        assert!(
            err.contains("requires structural time monotonicity"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn censored_rows_still_reject_invalid_time_derivative() {
        let family = SurvivalMarginalSlopeFamily {
            n: 1,
            event: Arc::new(array![0.0]),
            weights: Arc::new(array![1.0]),
            z: Arc::new(array![0.0]),
            derivative_guard: 1e-4,
            design_entry: Arc::new(Array2::zeros((1, 1))),
            design_exit: Arc::new(Array2::zeros((1, 1))),
            design_derivative_exit: Arc::new(Array2::ones((1, 1))),
            offset_entry: Arc::new(Array1::zeros(1)),
            offset_exit: Arc::new(Array1::zeros(1)),
            derivative_offset_exit: Arc::new(Array1::zeros(1)),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((1, 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((1, 0)),
            )),
            time_linear_constraints: None,
        };
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.0],
                eta: array![0.0],
            },
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: array![0.0],
            },
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: array![0.0],
            },
        ];

        let err = family
            .row_neglog_directional(0, &block_states, &[])
            .expect_err("censored rows must still enforce the time-derivative domain");
        assert!(
            err.contains("monotonicity violated at row 0"),
            "unexpected error: {err}"
        );
    }
}
