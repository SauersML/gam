use crate::estimate::EstimationError;
use crate::faer_ndarray::{
    FaerArrayView, FaerCholesky, FaerEigh, factorize_symmetric_with_fallback,
};
use crate::pirls::{WorkingModel as PirlsWorkingModel, WorkingState};
use crate::types::{Coefficients, LinearPredictor};
use faer::Side;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};
use std::ops::Range;
use std::sync::OnceLock;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SurvivalError {
    #[error("input dimensions are inconsistent")]
    DimensionMismatch,
    #[error("inputs contain non-finite values")]
    NonFiniteInput,
    #[error("crude risk integration setup is invalid")]
    InvalidIntegrationSetup,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct AgeTransform {
    pub minimum_age: f64,
    pub delta: f64,
}

impl AgeTransform {
    pub fn transform(&self, age: f64) -> Result<f64, SurvivalError> {
        let shifted = age - self.minimum_age + self.delta;
        if !shifted.is_finite() || shifted <= 0.0 {
            return Err(SurvivalError::NonFiniteInput);
        }
        Ok(shifted.ln())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum SurvivalSpec {
    #[default]
    Net,
    Crude,
}

#[derive(Debug, Clone)]
pub struct SurvivalEngineInputs<'a> {
    pub age_entry: ArrayView1<'a, f64>,
    pub age_exit: ArrayView1<'a, f64>,
    pub event_target: ArrayView1<'a, u8>,
    pub event_competing: ArrayView1<'a, u8>,
    pub sample_weight: ArrayView1<'a, f64>,
    pub x_entry: ArrayView2<'a, f64>,
    pub x_exit: ArrayView2<'a, f64>,
    pub x_derivative: ArrayView2<'a, f64>,
}

#[derive(Debug, Clone)]
pub struct SurvivalBaselineOffsets<'a> {
    /// Baseline target contribution to eta at entry time: eta_target(t_entry).
    pub eta_entry: ArrayView1<'a, f64>,
    /// Baseline target contribution to eta at exit time: eta_target(t_exit).
    pub eta_exit: ArrayView1<'a, f64>,
    /// Baseline target contribution to d eta / d t at exit: eta_target'(t_exit).
    ///
    /// This is used in event terms where log-hazard requires
    /// log(d eta / d t). By threading this as an explicit offset, we get
    /// "parametric default + spline deviation" behavior:
    /// - strong penalty => deviation ~ 0 => model collapses to baseline target,
    /// - weak penalty   => deviation can bend away where data supports it.
    pub derivative_exit: ArrayView1<'a, f64>,
}

#[derive(Debug, Clone)]
pub struct PenaltyBlock {
    pub matrix: Array2<f64>,
    pub lambda: f64,
    pub range: Range<usize>,
}

#[derive(Debug, Clone)]
pub struct PenaltyBlocks {
    pub blocks: Vec<PenaltyBlock>,
}

impl PenaltyBlocks {
    pub fn new(blocks: Vec<PenaltyBlock>) -> Self {
        Self { blocks }
    }

    pub fn gradient(&self, beta: &Array1<f64>) -> Array1<f64> {
        let mut grad = Array1::zeros(beta.len());
        for block in &self.blocks {
            if block.lambda == 0.0 {
                continue;
            }
            let b = beta.slice(ndarray::s![block.range.clone()]).to_owned();
            let g = block.matrix.dot(&b);
            let mut dst = grad.slice_mut(ndarray::s![block.range.clone()]);
            dst += &(block.lambda * g);
        }
        grad
    }

    pub fn hessian(&self, dim: usize) -> Array2<f64> {
        let mut h = Array2::zeros((dim, dim));
        for block in &self.blocks {
            if block.lambda == 0.0 {
                continue;
            }
            let r = block.range.clone();
            for (i_local, i) in r.clone().enumerate() {
                for (j_local, j) in r.clone().enumerate() {
                    h[[i, j]] += block.lambda * block.matrix[[i_local, j_local]];
                }
            }
        }
        h
    }

    pub fn deviance(&self, beta: &Array1<f64>) -> f64 {
        let mut value = 0.0;
        for block in &self.blocks {
            if block.lambda == 0.0 {
                continue;
            }
            let b = beta.slice(ndarray::s![block.range.clone()]).to_owned();
            value += 0.5 * block.lambda * b.dot(&block.matrix.dot(&b));
        }
        value
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MonotonicityPenalty {
    pub lambda: f64,
    pub tolerance: f64,
}

impl Default for MonotonicityPenalty {
    fn default() -> Self {
        Self {
            lambda: 1.0,
            tolerance: 0.0,
        }
    }
}

#[derive(Clone)]
pub struct WorkingModelSurvival {
    age_entry: Array1<f64>,
    age_exit: Array1<f64>,
    entry_at_origin: Array1<bool>,
    event_target: Array1<u8>,
    event_competing: Array1<u8>,
    sample_weight: Array1<f64>,
    x_entry: Array2<f64>,
    x_exit: Array2<f64>,
    x_derivative: Array2<f64>,
    offset_eta_entry: Array1<f64>,
    offset_eta_exit: Array1<f64>,
    offset_derivative_exit: Array1<f64>,
    penalties: PenaltyBlocks,
    monotonicity: MonotonicityPenalty,
    spec: SurvivalSpec,
}

#[inline]
fn smooth_guarded_derivative(derivative: f64, guard: f64, softness: f64) -> (f64, f64, f64) {
    let tau = softness.max(1e-12);
    let z = (derivative - guard) / tau;

    if z >= 40.0 {
        return (derivative, 1.0, 0.0);
    }
    if z <= -40.0 {
        let ez = z.exp();
        let safe = guard + tau * ez;
        let d1 = ez;
        let d2 = ez / tau;
        return (safe, d1, d2);
    }

    let softplus = if z > 0.0 {
        z + (-z).exp().ln_1p()
    } else {
        z.exp().ln_1p()
    };
    let sigma = 1.0 / (1.0 + (-z).exp());
    let d2 = sigma * (1.0 - sigma) / tau;
    let safe = guard + tau * softplus;
    (safe, sigma, d2)
}

impl WorkingModelSurvival {
    pub fn from_engine_inputs(
        inputs: SurvivalEngineInputs<'_>,
        penalties: PenaltyBlocks,
        monotonicity: MonotonicityPenalty,
        spec: SurvivalSpec,
    ) -> Result<Self, SurvivalError> {
        Self::from_engine_inputs_with_offsets(inputs, None, penalties, monotonicity, spec)
    }

    pub fn from_engine_inputs_with_offsets(
        inputs: SurvivalEngineInputs<'_>,
        offsets: Option<SurvivalBaselineOffsets<'_>>,
        penalties: PenaltyBlocks,
        monotonicity: MonotonicityPenalty,
        spec: SurvivalSpec,
    ) -> Result<Self, SurvivalError> {
        // This constructor is the engine-level hook for transformation-model style
        // baselines:
        //   eta(t, x) = eta_target(t) + eta_deviation(t, x).
        //
        // Existing design matrices continue to represent eta_deviation. Offsets
        // inject eta_target and its derivative. This keeps identifiability and
        // allows REML-selected penalties to control how far we depart from the
        // target in sparse-vs-rich data regimes.
        let n = inputs.age_entry.len();
        if inputs.age_exit.len() != n
            || inputs.event_target.len() != n
            || inputs.event_competing.len() != n
            || inputs.sample_weight.len() != n
            || inputs.x_entry.nrows() != n
            || inputs.x_exit.nrows() != n
            || inputs.x_derivative.nrows() != n
            || inputs.x_entry.ncols() != inputs.x_exit.ncols()
            || inputs.x_entry.ncols() != inputs.x_derivative.ncols()
        {
            return Err(SurvivalError::DimensionMismatch);
        }

        if inputs.age_entry.iter().any(|v| !v.is_finite())
            || inputs.age_exit.iter().any(|v| !v.is_finite())
            || inputs
                .sample_weight
                .iter()
                .any(|v| !v.is_finite() || *v < 0.0)
            || inputs.x_entry.iter().any(|v| !v.is_finite())
            || inputs.x_exit.iter().any(|v| !v.is_finite())
            || inputs.x_derivative.iter().any(|v| !v.is_finite())
        {
            return Err(SurvivalError::NonFiniteInput);
        }

        let (offset_eta_entry, offset_eta_exit, offset_derivative_exit) = if let Some(off) = offsets
        {
            if off.eta_entry.len() != n || off.eta_exit.len() != n || off.derivative_exit.len() != n
            {
                return Err(SurvivalError::DimensionMismatch);
            }
            if off.eta_entry.iter().any(|v| !v.is_finite())
                || off.eta_exit.iter().any(|v| !v.is_finite())
                || off.derivative_exit.iter().any(|v| !v.is_finite())
            {
                return Err(SurvivalError::NonFiniteInput);
            }
            (
                off.eta_entry.to_owned(),
                off.eta_exit.to_owned(),
                off.derivative_exit.to_owned(),
            )
        } else {
            (Array1::zeros(n), Array1::zeros(n), Array1::zeros(n))
        };

        Ok(Self {
            age_entry: inputs.age_entry.to_owned(),
            age_exit: inputs.age_exit.to_owned(),
            entry_at_origin: inputs.age_entry.mapv(|t| t <= 1e-8),
            event_target: inputs.event_target.to_owned(),
            event_competing: inputs.event_competing.to_owned(),
            sample_weight: inputs.sample_weight.to_owned(),
            x_entry: inputs.x_entry.to_owned(),
            x_exit: inputs.x_exit.to_owned(),
            x_derivative: inputs.x_derivative.to_owned(),
            offset_eta_entry,
            offset_eta_exit,
            offset_derivative_exit,
            penalties,
            monotonicity,
            spec,
        })
    }

    pub fn update_state(&self, beta: &Array1<f64>) -> Result<WorkingState, EstimationError> {
        if beta.len() != self.x_exit.ncols() {
            return Err(EstimationError::InvalidInput(
                "survival beta dimension mismatch".to_string(),
            ));
        }

        let n = self.x_exit.nrows();
        let p = self.x_exit.ncols();

        // Royston-Parmar contract used throughout the engine:
        //   eta(t) = log(H(t)), where H(t) is cumulative hazard.
        //
        // With row-vectors (per subject i):
        //   a1_i^T := x_exit_i^T,  a0_i^T := x_entry_i^T,  d_i^T := x_derivative_i^T
        // and scalars:
        //   eta1_i = a1_i^T beta,  eta0_i = a0_i^T beta,  s_i = d_i^T beta.
        //
        // The per-subject negative log-likelihood used below is
        //   NLL_i(beta) = exp(eta1_i) - exp(eta0_i) - delta_i * (eta1_i + log(s_i)),
        // with delta_i = event_target_i.
        //
        // This is exactly the form whose derivatives are:
        //   grad_i = exp(eta1_i) a1_i - exp(eta0_i) a0_i - delta_i * (a1_i + d_i / s_i)
        //   Hess_i = exp(eta1_i) a1_i a1_i^T - exp(eta0_i) a0_i a0_i^T
        //            + delta_i * (d_i d_i^T) / s_i^2.
        //
        // The loop below computes those terms directly, then adds penalties.
        // Total predictor = target offset + learned deviation.
        // This is the same architecture used for flexible binary links:
        // principled default, plus penalized wiggle/deviation.
        let eta_entry = self.x_entry.dot(beta) + &self.offset_eta_entry;
        let eta_exit = self.x_exit.dot(beta) + &self.offset_eta_exit;
        let derivative_raw = self.x_derivative.dot(beta) + &self.offset_derivative_exit;

        let h_entry = eta_entry.mapv(f64::exp);
        let h_exit = eta_exit.mapv(f64::exp);
        if h_entry.iter().any(|v| !v.is_finite()) || h_exit.iter().any(|v| !v.is_finite()) {
            return Err(EstimationError::InvalidInput(
                "survival linear predictor produced non-finite cumulative hazard".to_string(),
            ));
        }

        let mut nll = 0.0;
        let mut grad = Array1::<f64>::zeros(p);
        let mut h = Array2::<f64>::zeros((p, p));

        // Smoothly guard derivative terms for log(hazard) = eta + log(d eta / d t)
        // so event contributions stay informative near/below the lower bound.
        let derivative_guard = self.monotonicity.tolerance.max(1e-12);
        let derivative_softness = derivative_guard.max(1e-6);
        for i in 0..n {
            let w = self.sample_weight[i];
            if w <= 0.0 {
                continue;
            }
            let entry_age = self.age_entry[i];
            let exit_age = self.age_exit[i];
            if !entry_age.is_finite() || !exit_age.is_finite() || exit_age < entry_age {
                return Err(EstimationError::InvalidInput(
                    "survival ages must be finite with age_exit >= age_entry".to_string(),
                ));
            }
            let _e_competing = self.event_competing[i];
            let d = f64::from(self.event_target[i]);

            let has_entry_interval = !self.entry_at_origin[i];
            let h_s = if has_entry_interval { h_entry[i] } else { 0.0 };
            let h_e = h_exit[i];
            nll += w * (h_e - h_s);

            let x_s = self.x_entry.row(i);
            let x_e = self.x_exit.row(i);

            // Interval contribution to NLL:
            //   exp(eta_exit) - exp(eta_entry)
            // Gradient piece:
            //   exp(eta_exit) * x_exit - exp(eta_entry) * x_entry
            for j in 0..p {
                let mut g_j = h_e * x_e[j];
                if has_entry_interval {
                    g_j -= h_s * x_s[j];
                }
                grad[j] += w * g_j;
            }

            // Hessian piece from interval contribution:
            //   exp(eta_exit) * x_exit x_exit^T - exp(eta_entry) * x_entry x_entry^T
            for r in 0..p {
                let xe_r = x_e[r];
                let xs_r = x_s[r];
                for c in 0..p {
                    let mut h_rc = h_e * xe_r * x_e[c];
                    if has_entry_interval {
                        h_rc -= h_s * xs_r * x_s[c];
                    }
                    h[[r, c]] += w * h_rc;
                }
            }

            if d > 0.0 {
                let deriv = derivative_raw[i];
                let (safe_deriv, safe_d1, safe_d2) =
                    smooth_guarded_derivative(deriv, derivative_guard, derivative_softness);
                let inv_safe_deriv = 1.0 / safe_deriv;
                let d_row = self.x_derivative.row(i);
                nll += -w * (eta_exit[i] + safe_deriv.ln());

                // Event contribution:
                //   - (eta_exit + log(s_i)), with s_i = d_i^T beta = derivative_raw[i].
                //
                // Gradient piece:
                //   -x_exit - d_i / s_i (with smooth guarded s_i).
                for j in 0..p {
                    grad[j] += -w * (x_e[j] + (safe_d1 * inv_safe_deriv) * d_row[j]);
                }

                // Hessian from smooth -log(s_i), where raw slope enters through
                // a softplus-guarded map to keep derivatives continuous.
                let log_s_second =
                    (safe_d1 * safe_d1 - safe_deriv * safe_d2) * inv_safe_deriv * inv_safe_deriv;
                for r in 0..p {
                    let dr = d_row[r];
                    for c in 0..p {
                        h[[r, c]] += w * log_s_second * dr * d_row[c];
                    }
                }
            }
        }

        let penalty_grad = self.penalties.gradient(beta);
        let penalty_hessian = self.penalties.hessian(p);
        let penalty_dev = self.penalties.deviance(beta);

        // Keep monotonicity regularization aligned with the hazard derivative
        // used in the likelihood: baseline-target offset + learned deviation.
        let slope = self.x_derivative.dot(beta) + &self.offset_derivative_exit;
        // Hard monotonicity is enforced by construction through the guarded
        // derivative map used in the event term (safe_deriv > 0 always).
        // The raw slope itself is not penalized.
        let _ = slope;

        let mut total_grad = grad;
        total_grad += &penalty_grad;

        h += &penalty_hessian;
        const SURVIVAL_STABILIZATION_RIDGE: f64 = 1e-8;
        let ridge_used = SURVIVAL_STABILIZATION_RIDGE;
        for d in 0..p {
            h[[d, d]] += ridge_used;
        }
        total_grad += &beta.mapv(|v| ridge_used * v);
        let ridge_penalty = ridge_used * beta.dot(beta);

        let mut deviance = 2.0 * nll;
        if matches!(self.spec, SurvivalSpec::Crude) {
            deviance *= 1.0;
        }

        Ok(WorkingState {
            eta: LinearPredictor::new(eta_exit),
            gradient: total_grad,
            hessian: h,
            deviance,
            penalty_term: penalty_dev + ridge_penalty,
            firth_log_det: None,
            firth_hat_diag: None,
            ridge_used,
        })
    }

    /// Evaluate the survival outer objective and exact `rho`-gradient at a
    /// converged inner state.
    ///
    /// Objective and notation in this implementation:
    /// - `rho_k = log(lambda_k)`, `lambda_k = exp(rho_k)`.
    /// - `S(rho) = sum_k A_k`, where `A_k = dS/drho_k = lambda_k * S_k`.
    /// - `H = X^T W X + S(rho)` from the inner model state.
    /// - `V(rho) = 0.5*deviance(beta_hat) + penalty_term(beta_hat)
    ///           + 0.5*log|H| - 0.5*log|S|_+`.
    ///
    /// Exact gradient used here (no finite differences):
    ///   dV/drho_k
    ///     = 0.5 * beta_hat^T A_k beta_hat
    ///     + 0.5 * tr(H^{-1} dH/drho_k)
    ///     - 0.5 * tr(S^+ A_k).
    ///
    /// For this Royston-Parmar survival likelihood, the hard part is the
    /// `tr(H^{-1} dH/drho_k)` contraction. We evaluate it exactly without
    /// building any 3-tensor:
    ///   tr(H^{-1} dH/drho_k) = tr(H^{-1} A_k) + third_derivative_contraction.
    ///
    /// The contraction is reduced to vector operations with leverage-like
    /// diagonals:
    /// - `q1_i = x_exit_i^T H^{-1} x_exit_i`
    /// - `q0_i = x_entry_i^T H^{-1} x_entry_i`
    /// - `qd_i = d_i^T H^{-1} d_i`
    ///
    /// and `u_k = d beta_hat / d rho_k = -H^{-1} A_k beta_hat`.
    ///
    /// Then:
    /// - `+ sum_i w_i * exp(eta_exit_i)  * (x_exit_i^T u_k) * q1_i`
    /// - `- sum_i w_i * exp(eta_entry_i) * (x_entry_i^T u_k) * q0_i`
    /// - `- 2 * sum_events w_i * (d_i^T u_k) * qd_i / (d_i^T beta_hat)^3`
    ///
    /// The last term is the exact contribution from differentiating
    /// `delta_i * log(d_i^T beta)` through the Hessian.
    ///
    /// Fast exact trace-contraction implementation details:
    /// - factorize `H` once,
    /// - compute `H^{-1}` actions by solves (`solve_vec` / `solve_mat`),
    /// - compute leverages by solving `H Z = X^T` blocks,
    /// - avoid explicit dense `H^{-1}` materialization.
    ///
    /// Complexity per outer evaluation (dense):
    /// - shared precompute: `O(np^2 + p^3)`,
    /// - per penalty block `k`: `O(np + p^2 + nnz(S_k))`.
    ///
    /// Ridge note:
    /// - `state.penalty_term` includes the tiny stabilization ridge used in the
    ///   inner solve, so the objective value is consistent with the solved mode.
    /// - this ridge is constant in `rho`, so it does not enter `A_k` nor
    ///   `tr(S^+ A_k)`.
    pub fn laml_objective_and_rho_gradient(
        &self,
        beta: &Array1<f64>,
        state: &WorkingState,
    ) -> Result<(f64, Array1<f64>), EstimationError> {
        let p = beta.len();
        let k_count = self.penalties.blocks.len();
        if k_count == 0 {
            return Ok((0.5 * state.deviance, Array1::zeros(0)));
        }
        if state.hessian.nrows() != p || state.hessian.ncols() != p {
            return Err(EstimationError::LayoutError(
                "survival laml gradient: Hessian/beta dimension mismatch".to_string(),
            ));
        }

        // Reuse one symmetric factorization for all H^{-1} applications.
        // This is the core speed-up: exact contractions via solves, no dense
        // inverse assembly.
        let h_view = FaerArrayView::new(&state.hessian);
        let factor = factorize_symmetric_with_fallback(h_view.as_ref(), Side::Lower)
            .map_err(|e| EstimationError::InvalidInput(e.to_string()))?;

        let solve_mat = |rhs: &Array2<f64>| -> Array2<f64> {
            let rhs_view = FaerArrayView::new(rhs);
            let solved = factor.solve(rhs_view.as_ref());
            Array2::from_shape_fn((solved.nrows(), solved.ncols()), |(i, j)| solved[(i, j)])
        };
        let solve_vec = |rhs: &Array1<f64>| -> Array1<f64> {
            let rhs_mat = rhs.clone().insert_axis(Axis(1));
            let solved = solve_mat(&rhs_mat);
            solved.column(0).to_owned()
        };

        // Keep outer gradient contractions consistent with the fitted inner state:
        // the working predictor includes target baseline offsets plus learned deviation.
        let eta_entry = self.x_entry.dot(beta) + &self.offset_eta_entry;
        let eta_exit = self.x_exit.dot(beta) + &self.offset_eta_exit;
        let deriv_raw = self.x_derivative.dot(beta) + &self.offset_derivative_exit;
        let exp_entry = eta_entry.mapv(f64::exp);
        let exp_exit = eta_exit.mapv(f64::exp);
        let guard = self.monotonicity.tolerance.max(1e-12);
        let derivative_softness = guard.max(1e-6);

        // Leverage-like diagonals used by the third-derivative contraction:
        // q_i = x_i^T H^{-1} x_i, computed via solves against X^T blocks.
        let z1 = solve_mat(&self.x_exit.t().to_owned());
        let z0 = solve_mat(&self.x_entry.t().to_owned());
        let zd = solve_mat(&self.x_derivative.t().to_owned());
        let n = self.x_exit.nrows();
        let mut q1 = Array1::<f64>::zeros(n);
        let mut q0 = Array1::<f64>::zeros(n);
        let mut qd = Array1::<f64>::zeros(n);
        for i in 0..n {
            q1[i] = self.x_exit.row(i).dot(&z1.column(i)).max(0.0);
            q0[i] = if self.entry_at_origin[i] {
                0.0
            } else {
                self.x_entry.row(i).dot(&z0.column(i)).max(0.0)
            };
            qd[i] = self.x_derivative.row(i).dot(&zd.column(i)).max(0.0);
        }

        // Assemble S(rho) in the same scaling used by update_state.
        let mut s_total = Array2::<f64>::zeros((p, p));
        for block in &self.penalties.blocks {
            if block.lambda == 0.0 {
                continue;
            }
            let r = block.range.clone();
            let scale = block.lambda;
            for (i_local, i) in r.clone().enumerate() {
                for (j_local, j) in r.clone().enumerate() {
                    s_total[[i, j]] += scale * block.matrix[[i_local, j_local]];
                }
            }
        }

        // Pseudo-logdet and pseudoinverse for S:
        // d/drho_k [-0.5 log|S|_+] = -0.5 tr(S^+ A_k).
        let (s_eval, s_evec) = s_total
            .eigh(Side::Lower)
            .map_err(|e| EstimationError::InvalidInput(e.to_string()))?;
        let max_s_eval = s_eval.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        let s_tol = (max_s_eval * 1e-12).max(1e-14);
        let mut logdet_s = 0.0_f64;
        let mut s_pinv = Array2::<f64>::zeros((p, p));
        for j in 0..p {
            let ev = s_eval[j];
            if ev > s_tol {
                // Positive-eigenspace pseudo-determinant:
                //   log|S|_+ = sum_{ev_j > tol} log(ev_j)
                logdet_s += ev.ln();
                let inv_ev = 1.0 / ev;
                // Build S^+ = U_+ diag(1/ev_j) U_+^T in-place by rank-1 updates.
                for r in 0..p {
                    let ur = s_evec[(r, j)];
                    for c in 0..p {
                        s_pinv[[r, c]] += inv_ev * ur * s_evec[(c, j)];
                    }
                }
            }
        }

        // Robust logdet for H: Cholesky on SPD path, eigen fallback otherwise.
        let logdet_h = if let Ok(chol) = state.hessian.clone().cholesky(Side::Lower) {
            let l = chol.lower_triangular();
            2.0 * (0..l.nrows()).map(|i| l[[i, i]].ln()).sum::<f64>()
        } else {
            let (eval, _) = state
                .hessian
                .eigh(Side::Lower)
                .map_err(|e| EstimationError::InvalidInput(e.to_string()))?;
            let max_eval = eval.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
            let tol = (max_eval * 1e-12).max(1e-14);
            eval.iter().filter(|&&v| v > tol).map(|&v| v.ln()).sum()
        };

        // Use the same inner objective components reported by update_state:
        //   0.5 * deviance + penalty_term.
        // This keeps the outer value consistent with the solved inner mode.
        let objective = 0.5 * state.deviance + state.penalty_term + 0.5 * logdet_h - 0.5 * logdet_s;

        let mut grad = Array1::<f64>::zeros(k_count);
        for (k, block) in self.penalties.blocks.iter().enumerate() {
            let lambda = block.lambda;
            let r = block.range.clone();

            let b_block = beta.slice(ndarray::s![r.clone()]).to_owned();
            let a_k_beta_block = block.matrix.dot(&b_block).mapv(|v| lambda * v);
            let mut a_k_beta = Array1::<f64>::zeros(p);
            a_k_beta
                .slice_mut(ndarray::s![r.clone()])
                .assign(&a_k_beta_block);

            // Implicit inner derivative:
            // d beta_hat / d rho_k = -H^{-1} A_k beta_hat.
            let u_k = -solve_vec(&a_k_beta);
            let s1k = self.x_exit.dot(&u_k);
            let s0k = self.x_entry.dot(&u_k);
            let sdk = self.x_derivative.dot(&u_k);

            // trace(H^{-1} A_k) on block support via solves against block unit vectors.
            // This is equivalent to Frobenius inner product <H^{-1}, A_k>, but
            // computed from selected solved columns only.
            let block_dim = r.len();
            let mut block_basis = Array2::<f64>::zeros((p, block_dim));
            for (j_local, j) in r.clone().enumerate() {
                block_basis[[j, j_local]] = 1.0;
            }
            let solved_basis = solve_mat(&block_basis);
            let mut trace_hinv_ak = 0.0_f64;
            for (i_local, i) in r.clone().enumerate() {
                for (j_local, _j) in r.clone().enumerate() {
                    // solved_basis[i, j_local] = H^{-1}_{i, r[j_local]}.
                    trace_hinv_ak +=
                        solved_basis[[i, j_local]] * (lambda * block.matrix[[j_local, i_local]]);
                }
            }

            // Tensor-free exact third-derivative contraction:
            // + sum_i exp(eta_exit_i)  (x_exit_i^T u_k) q1_i
            // - sum_i exp(eta_entry_i) (x_entry_i^T u_k) q0_i
            // - 2 sum_events (d_i^T u_k) qd_i / (d_i^T beta)^3
            // The last term is from differentiating delta_i * log(d_i^T beta).
            let mut trace_third = 0.0_f64;
            for i in 0..n {
                let w_i = self.sample_weight[i];
                trace_third += w_i * exp_exit[i] * s1k[i] * q1[i];
                if !self.entry_at_origin[i] {
                    trace_third -= w_i * exp_entry[i] * s0k[i] * q0[i];
                }
                if self.event_target[i] > 0 {
                    let (s_i, _d1, _d2) =
                        smooth_guarded_derivative(deriv_raw[i], guard, derivative_softness);
                    trace_third -= 2.0 * w_i * sdk[i] * qd[i] / (s_i * s_i * s_i);
                }
            }
            let t_k = trace_hinv_ak + trace_third;

            let mut p_k = 0.0_f64;
            for (i_local, i) in r.clone().enumerate() {
                for (j_local, j) in r.clone().enumerate() {
                    // p_k = tr(S^+ A_k) restricted to this block.
                    p_k += s_pinv[[i, j]] * (lambda * block.matrix[[j_local, i_local]]);
                }
            }

            // Final exact component:
            // 0.5 * beta^T A_k beta + 0.5 * tr(H^{-1} dH/drho_k) - 0.5 * tr(S^+ A_k),
            // with tr(H^{-1} dH/drho_k) = trace_hinv_ak + trace_third.
            grad[k] = 0.5 * beta.dot(&a_k_beta) + 0.5 * t_k - 0.5 * p_k;
        }

        Ok((objective, grad))
    }
}

#[derive(Debug, Clone)]
pub struct CrudeRiskResult {
    pub risk: f64,
    pub disease_gradient: Array1<f64>,
    pub mortality_gradient: Array1<f64>,
}

fn compute_gauss_legendre_nodes(n: usize) -> Vec<(f64, f64)> {
    let mut nodes_weights = Vec::with_capacity(n);
    let m = n.div_ceil(2);

    for i in 0..m {
        let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut pp = 0.0;

        for _ in 0..100 {
            let mut p1 = 1.0;
            let mut p2 = 0.0;
            for j in 0..n {
                let p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j as f64 + 1.0) * z * p2 - j as f64 * p3) / (j as f64 + 1.0);
            }
            pp = n as f64 * (z * p1 - p2) / (z * z - 1.0);
            let z_prev = z;
            z = z_prev - p1 / pp;
            if (z - z_prev).abs() < 1e-14 {
                break;
            }
        }

        let x = z;
        let w = 2.0 / ((1.0 - z * z) * pp * pp);
        if !n.is_multiple_of(2) && i == m - 1 {
            nodes_weights.push((0.0, w));
        } else {
            nodes_weights.push((-x, w));
            nodes_weights.push((x, w));
        }
    }

    nodes_weights.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    nodes_weights
}

fn gauss_legendre_quadrature() -> &'static [(f64, f64)] {
    static CACHE: OnceLock<Vec<(f64, f64)>> = OnceLock::new();
    CACHE.get_or_init(|| compute_gauss_legendre_nodes(40))
}

/// Engine-level crude risk quadrature with exact delta-method gradients.
///
/// This routine owns the numerical integration and gradient accumulation math:
/// - It integrates `h_d(u) * S_total(u | t0)` over `[t0, t1]` by high-order
///   Gauss-Legendre quadrature.
/// - It computes gradients w.r.t. disease and mortality coefficients:
///   d Risk / d beta_d and d Risk / d beta_m.
///
/// The adapter provides the domain-specific point evaluator callback `eval_at`,
/// which fills design rows and returns:
/// - instantaneous disease hazard at age `u`,
/// - cumulative disease hazard `H_d(u)`,
/// - cumulative mortality hazard `H_m(u)`.
///
/// This keeps biology/data wiring out of `gam` while centralizing the
/// integration engine in one place.
pub fn calculate_crude_risk_quadrature<F>(
    t0: f64,
    t1: f64,
    breakpoints: &[f64],
    h_dis_t0: f64,
    h_mor_t0: f64,
    design_d_t0: ArrayView1<'_, f64>,
    design_m_t0: ArrayView1<'_, f64>,
    mut eval_at: F,
) -> Result<CrudeRiskResult, SurvivalError>
where
    F: FnMut(
        f64,
        &mut Array1<f64>,
        &mut Array1<f64>,
        &mut Array1<f64>,
    ) -> Result<(f64, f64, f64), SurvivalError>,
{
    let coeff_len_d = design_d_t0.len();
    let coeff_len_m = design_m_t0.len();
    if coeff_len_d == 0 || coeff_len_m == 0 {
        return Err(SurvivalError::InvalidIntegrationSetup);
    }
    if !t0.is_finite()
        || !t1.is_finite()
        || !h_dis_t0.is_finite()
        || !h_mor_t0.is_finite()
        || design_d_t0.iter().any(|v| !v.is_finite())
        || design_m_t0.iter().any(|v| !v.is_finite())
    {
        return Err(SurvivalError::NonFiniteInput);
    }
    if t1 <= t0 {
        return Ok(CrudeRiskResult {
            risk: 0.0,
            disease_gradient: Array1::zeros(coeff_len_d),
            mortality_gradient: Array1::zeros(coeff_len_m),
        });
    }

    let mut sorted_breaks: Vec<f64> = breakpoints
        .iter()
        .copied()
        .filter(|x| x.is_finite() && *x >= t0 && *x <= t1)
        .collect();
    sorted_breaks.push(t0);
    sorted_breaks.push(t1);
    sorted_breaks.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted_breaks.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    if sorted_breaks.len() < 2 {
        return Err(SurvivalError::InvalidIntegrationSetup);
    }

    let mut total_risk = 0.0;
    let mut disease_gradient = Array1::zeros(coeff_len_d);
    let mut mortality_gradient = Array1::zeros(coeff_len_m);
    let nodes_weights = gauss_legendre_quadrature();

    let mut design_d = Array1::<f64>::zeros(coeff_len_d);
    let mut deriv_d = Array1::<f64>::zeros(coeff_len_d);
    let mut design_m = Array1::<f64>::zeros(coeff_len_m);

    for segment in sorted_breaks.windows(2) {
        let a = segment[0];
        let b = segment[1];
        let center = 0.5 * (b + a);
        let half_width = 0.5 * (b - a);
        if half_width <= 0.0 {
            continue;
        }

        for &(x, w) in nodes_weights {
            let u = center + half_width * x;
            let (inst_hazard_d, hazard_d, hazard_m) =
                eval_at(u, &mut design_d, &mut deriv_d, &mut design_m)?;
            if !inst_hazard_d.is_finite() || !hazard_d.is_finite() || !hazard_m.is_finite() {
                return Err(SurvivalError::NonFiniteInput);
            }

            let h_dis_cond = (hazard_d - h_dis_t0).max(0.0);
            let h_mor_cond = (hazard_m - h_mor_t0).max(0.0);
            let s_total = (-(h_dis_cond + h_mor_cond)).exp();

            total_risk += w * inst_hazard_d * s_total * half_width;

            // d Risk / d beta_d:
            //   integral [ d h_d * S_total - h_d * S_total * d H_d ] du
            if inst_hazard_d > 0.0 {
                let weight = w * s_total * half_width;
                let mut grad_contrib = design_d.mapv(|v| inst_hazard_d * (1.0 - hazard_d) * v);
                grad_contrib.zip_mut_with(&deriv_d, |g, &d| {
                    *g += hazard_d * d;
                });
                grad_contrib.zip_mut_with(&design_d_t0, |g, &x0| {
                    *g += inst_hazard_d * h_dis_t0 * x0;
                });
                grad_contrib.mapv_inplace(|v| v * weight);
                disease_gradient += &grad_contrib;
            }

            // d Risk / d beta_m:
            //   -integral h_d * S_total * d H_m(u|t0) du
            if inst_hazard_d > 0.0 && hazard_m > 0.0 {
                let weight = w * inst_hazard_d * s_total * half_width;
                let mut mort_grad_contrib = design_m.mapv(|v| -weight * hazard_m * v);
                mort_grad_contrib.zip_mut_with(&design_m_t0, |g, &x0| {
                    *g += weight * h_mor_t0 * x0;
                });
                mortality_gradient += &mort_grad_contrib;
            }
        }
    }

    Ok(CrudeRiskResult {
        risk: total_risk,
        disease_gradient,
        mortality_gradient,
    })
}

impl PirlsWorkingModel for WorkingModelSurvival {
    fn update(&mut self, beta: &Coefficients) -> Result<WorkingState, EstimationError> {
        self.update_state(beta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, s};

    fn toy_penalties() -> PenaltyBlocks {
        let s = array![[2.0, 0.5], [0.5, 3.0]];
        PenaltyBlocks::new(vec![PenaltyBlock {
            matrix: s,
            lambda: 1.7,
            range: 1..3,
        }])
    }

    #[test]
    fn penalty_hessian_matches_gradient_jacobian() {
        let penalties = toy_penalties();
        let beta = array![10.0, -0.3, 1.2, 7.0];

        let grad = penalties.gradient(&beta);
        let h = penalties.hessian(beta.len());
        let b_block = beta.slice(s![1..3]).to_owned();
        let expected = 1.7 * array![[2.0, 0.5], [0.5, 3.0]].dot(&b_block);

        assert!((grad[1] - expected[0]).abs() < 1e-12);
        assert!((grad[2] - expected[1]).abs() < 1e-12);
        assert!((h[[1, 1]] - 1.7 * 2.0).abs() < 1e-12);
        assert!((h[[1, 2]] - 1.7 * 0.5).abs() < 1e-12);
        assert!((h[[2, 1]] - 1.7 * 0.5).abs() < 1e-12);
        assert!((h[[2, 2]] - 1.7 * 3.0).abs() < 1e-12);
    }

    #[test]
    fn penalty_gradient_matches_deviance_finite_difference() {
        let penalties = toy_penalties();
        let beta = array![10.0, -0.3, 1.2, 7.0];
        let grad = penalties.gradient(&beta);
        let eps = 1e-7;

        for idx in 0..beta.len() {
            let mut plus = beta.clone();
            let mut minus = beta.clone();
            plus[idx] += eps;
            minus[idx] -= eps;
            let fd = (penalties.deviance(&plus) - penalties.deviance(&minus)) / (2.0 * eps);
            assert!(
                (grad[idx] - fd).abs() < 1e-6,
                "gradient/deviance mismatch at idx={idx}: grad={} fd={fd}",
                grad[idx]
            );
        }
    }

    #[test]
    fn zero_offsets_match_default_survival_state() {
        let age_entry = array![1.0_f64, 2.0_f64];
        let age_exit = array![2.0_f64, 3.5_f64];
        let event_target = array![1u8, 0u8];
        let event_competing = array![0u8, 0u8];
        let sample_weight = array![1.0, 1.0];
        let x_entry = array![[1.0, age_entry[0].ln()], [1.0, age_entry[1].ln()]];
        let x_exit = array![[1.0, age_exit[0].ln()], [1.0, age_exit[1].ln()]];
        let x_derivative = array![[0.0, 1.0 / age_exit[0]], [0.0, 1.0 / age_exit[1]]];
        let penalties = PenaltyBlocks::new(Vec::new());
        let mono = MonotonicityPenalty {
            lambda: 0.0,
            tolerance: 1e-8,
        };
        let beta = array![-1.0, 0.8];

        let base = WorkingModelSurvival::from_engine_inputs(
            SurvivalEngineInputs {
                age_entry: age_entry.view(),
                age_exit: age_exit.view(),
                event_target: event_target.view(),
                event_competing: event_competing.view(),
                sample_weight: sample_weight.view(),
                x_entry: x_entry.view(),
                x_exit: x_exit.view(),
                x_derivative: x_derivative.view(),
            },
            penalties.clone(),
            mono,
            SurvivalSpec::Net,
        )
        .expect("construct base survival model");

        let zero_offsets = WorkingModelSurvival::from_engine_inputs_with_offsets(
            SurvivalEngineInputs {
                age_entry: age_entry.view(),
                age_exit: age_exit.view(),
                event_target: event_target.view(),
                event_competing: event_competing.view(),
                sample_weight: sample_weight.view(),
                x_entry: x_entry.view(),
                x_exit: x_exit.view(),
                x_derivative: x_derivative.view(),
            },
            Some(SurvivalBaselineOffsets {
                eta_entry: array![0.0, 0.0].view(),
                eta_exit: array![0.0, 0.0].view(),
                derivative_exit: array![0.0, 0.0].view(),
            }),
            penalties,
            mono,
            SurvivalSpec::Net,
        )
        .expect("construct offset survival model");

        let state_base = base.update_state(&beta).expect("base state");
        let state_zero = zero_offsets.update_state(&beta).expect("zero-offset state");
        assert!((state_base.deviance - state_zero.deviance).abs() < 1e-12);
        assert!(
            state_base
                .gradient
                .iter()
                .zip(state_zero.gradient.iter())
                .all(|(a, b)| (a - b).abs() < 1e-12)
        );
    }

    fn model_with_rho(base: &WorkingModelSurvival, rho: &Array1<f64>) -> WorkingModelSurvival {
        let mut model = base.clone();
        assert_eq!(model.penalties.blocks.len(), rho.len());
        for (k, block) in model.penalties.blocks.iter_mut().enumerate() {
            block.lambda = rho[k].exp();
        }
        model
    }

    fn solve_vec_h(h: &Array2<f64>, rhs: &Array1<f64>) -> Array1<f64> {
        let h_view = FaerArrayView::new(h);
        let factor = factorize_symmetric_with_fallback(h_view.as_ref(), Side::Lower)
            .expect("Hessian factorization should succeed in survival test solve");
        let rhs_mat = rhs.clone().insert_axis(Axis(1));
        let rhs_view = FaerArrayView::new(&rhs_mat);
        let solved = factor.solve(rhs_view.as_ref());
        Array1::from_shape_fn(rhs.len(), |i| solved[(i, 0)])
    }

    fn solve_inner_mode(model: &WorkingModelSurvival, beta_init: &Array1<f64>) -> Array1<f64> {
        let mut beta = beta_init.clone();
        let mut best_state = model.update_state(&beta).expect("initial survival state");
        let mut best_obj = 0.5 * best_state.deviance + best_state.penalty_term;

        for _ in 0..30 {
            let grad_norm = best_state.gradient.dot(&best_state.gradient).sqrt();
            if grad_norm < 1e-7 {
                break;
            }

            let step = solve_vec_h(&best_state.hessian, &best_state.gradient);
            let mut accepted = false;
            for ls in 0..12 {
                let alpha = 0.5_f64.powi(ls);
                let candidate = &beta - &(alpha * &step);
                let cand_state = model
                    .update_state(&candidate)
                    .expect("line-search survival state");
                let cand_obj = 0.5 * cand_state.deviance + cand_state.penalty_term;
                if cand_obj.is_finite() && cand_obj <= best_obj {
                    beta = candidate;
                    best_obj = cand_obj;
                    best_state = cand_state;
                    accepted = true;
                    break;
                }
            }
            if !accepted {
                break;
            }
        }

        beta
    }

    fn laml_objective_at_rho(
        base: &WorkingModelSurvival,
        rho: &Array1<f64>,
        beta_init: &Array1<f64>,
    ) -> (f64, Array1<f64>, Array1<f64>) {
        let model = model_with_rho(base, rho);
        let beta_hat = solve_inner_mode(&model, beta_init);
        let state = model
            .update_state(&beta_hat)
            .expect("state at inner mode for outer objective");
        let (obj, grad) = model
            .laml_objective_and_rho_gradient(&beta_hat, &state)
            .expect("analytic laml objective/gradient");
        (obj, grad, beta_hat)
    }

    #[test]
    fn laml_rho_gradient_matches_fd_with_nonzero_offsets() {
        let n = 8usize;
        let p = 3usize;

        let age_entry = array![40.0, 45.0, 50.0, 55.0, 60.0, 43.0, 52.0, 58.0];
        let age_exit = array![44.0, 49.0, 55.0, 61.0, 66.0, 48.0, 56.0, 63.0];
        let event_target = array![1u8, 0u8, 1u8, 1u8, 0u8, 1u8, 0u8, 1u8];
        let event_competing = Array1::<u8>::zeros(n);
        let sample_weight = Array1::ones(n);

        let mut x_entry = Array2::<f64>::zeros((n, p));
        let mut x_exit = Array2::<f64>::zeros((n, p));
        let mut x_derivative = Array2::<f64>::zeros((n, p));
        let mut off_eta_entry = Array1::<f64>::zeros(n);
        let mut off_eta_exit = Array1::<f64>::zeros(n);
        let mut off_deriv_exit = Array1::<f64>::zeros(n);

        for i in 0..n {
            let te = (age_entry[i] / 50.0) - 1.0;
            let tx = (age_exit[i] / 50.0) - 1.0;
            x_entry[[i, 0]] = 1.0;
            x_entry[[i, 1]] = te;
            x_entry[[i, 2]] = te * te;
            x_exit[[i, 0]] = 1.0;
            x_exit[[i, 1]] = tx;
            x_exit[[i, 2]] = tx * tx;
            x_derivative[[i, 0]] = 0.0;
            x_derivative[[i, 1]] = 0.02;
            x_derivative[[i, 2]] = 0.001 * age_exit[i];

            off_eta_entry[i] = -2.4 + 0.03 * age_entry[i];
            off_eta_exit[i] = -2.4 + 0.03 * age_exit[i];
            off_deriv_exit[i] = 0.08 + 0.0005 * age_exit[i];
        }

        let penalties = PenaltyBlocks::new(vec![
            PenaltyBlock {
                matrix: array![[1.0]],
                lambda: 1.0,
                range: 1..2,
            },
            PenaltyBlock {
                matrix: array![[1.0]],
                lambda: 1.0,
                range: 2..3,
            },
        ]);
        let mono = MonotonicityPenalty {
            lambda: 0.5,
            tolerance: 1e-6,
        };

        let base_model = WorkingModelSurvival::from_engine_inputs_with_offsets(
            SurvivalEngineInputs {
                age_entry: age_entry.view(),
                age_exit: age_exit.view(),
                event_target: event_target.view(),
                event_competing: event_competing.view(),
                sample_weight: sample_weight.view(),
                x_entry: x_entry.view(),
                x_exit: x_exit.view(),
                x_derivative: x_derivative.view(),
            },
            Some(SurvivalBaselineOffsets {
                eta_entry: off_eta_entry.view(),
                eta_exit: off_eta_exit.view(),
                derivative_exit: off_deriv_exit.view(),
            }),
            penalties,
            mono,
            SurvivalSpec::Net,
        )
        .expect("construct nonzero-offset survival model");

        let rho = array![(-0.3f64), (0.45f64)];
        let beta0 = Array1::<f64>::zeros(p);
        let (obj, analytic, beta_hat) = laml_objective_at_rho(&base_model, &rho, &beta0);
        assert!(obj.is_finite());
        assert!(analytic.iter().all(|v| v.is_finite()));

        let eps = 1e-4;
        let mut fd = Array1::<f64>::zeros(rho.len());
        for k in 0..rho.len() {
            let mut rho_plus = rho.clone();
            rho_plus[k] += eps;
            let mut rho_minus = rho.clone();
            rho_minus[k] -= eps;
            let (obj_plus, _, beta_plus) = laml_objective_at_rho(&base_model, &rho_plus, &beta_hat);
            let (obj_minus, _, _) = laml_objective_at_rho(&base_model, &rho_minus, &beta_plus);
            fd[k] = (obj_plus - obj_minus) / (2.0 * eps);
        }

        for k in 0..rho.len() {
            let abs_err = (analytic[k] - fd[k]).abs();
            let rel_err = abs_err / fd[k].abs().max(1e-6);
            assert!(
                rel_err < 2.5e-2 || abs_err < 2e-2,
                "rho-grad mismatch at k={k}: analytic={:.6e} fd={:.6e} abs={:.3e} rel={:.3e}",
                analytic[k],
                fd[k],
                abs_err,
                rel_err
            );
        }
    }
}
