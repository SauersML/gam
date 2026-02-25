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
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SurvivalError {
    #[error("input dimensions are inconsistent")]
    DimensionMismatch,
    #[error("inputs contain non-finite values")]
    NonFiniteInput,
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
            dst += &(2.0 * block.lambda * g);
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
                    h[[i, j]] += 2.0 * block.lambda * block.matrix[[i_local, j_local]];
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
            value += block.lambda * b.dot(&block.matrix.dot(&b));
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
    event_target: Array1<u8>,
    event_competing: Array1<u8>,
    sample_weight: Array1<f64>,
    x_entry: Array2<f64>,
    x_exit: Array2<f64>,
    x_derivative: Array2<f64>,
    penalties: PenaltyBlocks,
    monotonicity: MonotonicityPenalty,
    spec: SurvivalSpec,
}

impl WorkingModelSurvival {
    pub fn from_engine_inputs(
        inputs: SurvivalEngineInputs<'_>,
        penalties: PenaltyBlocks,
        monotonicity: MonotonicityPenalty,
        spec: SurvivalSpec,
    ) -> Result<Self, SurvivalError> {
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

        Ok(Self {
            age_entry: inputs.age_entry.to_owned(),
            age_exit: inputs.age_exit.to_owned(),
            event_target: inputs.event_target.to_owned(),
            event_competing: inputs.event_competing.to_owned(),
            sample_weight: inputs.sample_weight.to_owned(),
            x_entry: inputs.x_entry.to_owned(),
            x_exit: inputs.x_exit.to_owned(),
            x_derivative: inputs.x_derivative.to_owned(),
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
        let eta_entry = self.x_entry.dot(beta);
        let eta_exit = self.x_exit.dot(beta);
        let derivative_raw = self.x_derivative.dot(beta);

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

        // Guard derivative terms for log(hazard) = eta + log(d eta / d t)
        let derivative_guard = self.monotonicity.tolerance.max(1e-12);
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

            let h_s = h_entry[i];
            let h_e = h_exit[i];
            nll += w * (h_e - h_s);

            let x_s = self.x_entry.row(i);
            let x_e = self.x_exit.row(i);

            // Interval contribution to NLL:
            //   exp(eta_exit) - exp(eta_entry)
            // Gradient piece:
            //   exp(eta_exit) * x_exit - exp(eta_entry) * x_entry
            for j in 0..p {
                grad[j] += w * (h_e * x_e[j] - h_s * x_s[j]);
            }

            // Hessian piece from interval contribution:
            //   exp(eta_exit) * x_exit x_exit^T - exp(eta_entry) * x_entry x_entry^T
            for r in 0..p {
                let xe_r = x_e[r];
                let xs_r = x_s[r];
                for c in 0..p {
                    h[[r, c]] += w * (h_e * xe_r * x_e[c] - h_s * xs_r * x_s[c]);
                }
            }

            if d > 0.0 {
                let deriv = derivative_raw[i];
                let safe_deriv = if deriv > derivative_guard {
                    deriv
                } else {
                    derivative_guard
                };
                let inv_deriv = if deriv > derivative_guard {
                    1.0 / deriv
                } else {
                    0.0
                };
                let d_row = self.x_derivative.row(i);
                nll += -w * (eta_exit[i] + safe_deriv.ln());

                // Event contribution:
                //   - (eta_exit + log(s_i)), with s_i = d_i^T beta = derivative_raw[i].
                //
                // Gradient piece:
                //   -x_exit - d_i / s_i
                // implemented as -x_exit - inv_deriv * d_row.
                for j in 0..p {
                    grad[j] += -w * (x_e[j] + inv_deriv * d_row[j]);
                }

                // Hessian piece from -log(s_i):
                //   + (d_i d_i^T) / s_i^2
                // i.e. + inv_deriv_sq * d_row d_row^T.
                let inv_deriv_sq = inv_deriv * inv_deriv;
                if inv_deriv_sq > 0.0 {
                    for r in 0..p {
                        let dr = d_row[r];
                        for c in 0..p {
                            h[[r, c]] += w * inv_deriv_sq * dr * d_row[c];
                        }
                    }
                }
            }
        }

        let penalty_grad = self.penalties.gradient(beta);
        let penalty_hessian = self.penalties.hessian(p);
        let penalty_dev = self.penalties.deviance(beta);

        let slope = self.x_derivative.dot(beta);
        let mut mono_dev = 0.0;
        let mut mono_grad = Array1::<f64>::zeros(p);
        let mut mono_h = Array2::<f64>::zeros((p, p));
        if self.monotonicity.lambda > 0.0 {
            for i in 0..n {
                let s = slope[i] - self.monotonicity.tolerance;
                if s < 0.0 {
                    let v = -s;
                    mono_dev += self.monotonicity.lambda * v * v;
                    let xi = self.x_derivative.row(i).to_owned();
                    mono_grad += &(2.0 * self.monotonicity.lambda * s * &xi);
                    let outer = xi
                        .view()
                        .insert_axis(Axis(1))
                        .dot(&xi.view().insert_axis(Axis(0)));
                    mono_h += &(2.0 * self.monotonicity.lambda * outer);
                }
            }
        }

        let mut total_grad = grad;
        total_grad += &penalty_grad;
        total_grad += &mono_grad;

        h += &penalty_hessian;
        h += &mono_h;
        const SURVIVAL_STABILIZATION_RIDGE: f64 = 1e-8;
        let ridge_used = SURVIVAL_STABILIZATION_RIDGE;
        for d in 0..p {
            h[[d, d]] += ridge_used;
        }
        total_grad += &beta.mapv(|v| ridge_used * v);
        let ridge_penalty = ridge_used * beta.dot(beta);

        let mut deviance = 2.0 * (nll + mono_dev);
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

    /// Evaluate a Laplace-style survival outer objective and exact rho-gradient
    /// from a converged inner state without finite-difference outer probes.
    ///
    /// Derivation map (symbols -> code):
    /// - `rho_k = log(lambda_k)` and `A_k = dS/drho_k = 2 * lambda_k * S_k`.
    ///   The `2` factor comes from this module's penalty convention, where
    ///   `penalty_grad = 2 * lambda * S_k * beta`.
    /// - `V(rho) = f(beta_hat, rho) + 0.5 log|H| - 0.5 log|S|_+`.
    ///   Here `f` is represented by `0.5 * deviance + penalty_term`, where
    ///   `penalty_term` includes both configured penalties and the tiny solver
    ///   stabilization ridge used in `update_state`.
    /// - `dV/drho_k = 0.5 * beta^T A_k beta
    ///               + 0.5 * tr(H^{-1} dH/drho_k)
    ///               - 0.5 * tr(S^+ A_k)`.
    /// - `tr(H^{-1} dH/drho_k)` is evaluated with a tensor-free contraction:
    ///   `tr(H^{-1} A_k)` plus third-derivative terms induced by
    ///   `exp(eta_exit)`, `exp(eta_entry)`, and `log(d^T beta)`.
    ///   We avoid any explicit `p x p x p` tensor by precomputing:
    ///   `q1_i = x_exit_i^T H^{-1} x_exit_i`,
    ///   `q0_i = x_entry_i^T H^{-1} x_entry_i`,
    ///   `qd_i = d_i^T H^{-1} d_i`.
    /// Event-log term note:
    /// - the contribution from `delta_i * log(d_i^T beta)` appears in the
    ///   third-derivative contraction as:
    ///   `-2 * (d_i^T u_k) * qd_i / (d_i^T beta)^3` for event rows.
    ///
    /// Ridge note:
    /// - `state.penalty_term` includes a tiny stabilization ridge used by the
    ///   inner solve, so the objective value stays consistent with the solved
    ///   mode.
    /// - that ridge is constant w.r.t. `rho`, so it does not contribute to
    ///   `A_k` or the `-0.5 * tr(S^+ A_k)` pseudo-determinant derivative.
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

        let eta_entry = self.x_entry.dot(beta);
        let eta_exit = self.x_exit.dot(beta);
        let deriv_raw = self.x_derivative.dot(beta);
        let exp_entry = eta_entry.mapv(f64::exp);
        let exp_exit = eta_exit.mapv(f64::exp);
        let guard = self.monotonicity.tolerance.max(1e-12);

        // Leverage-like diagonals used by the third-derivative contraction:
        // q_i = x_i^T H^{-1} x_i.
        // Compute via factor-solve against transposed design blocks (no dense H^{-1} build).
        let z1 = solve_mat(&self.x_exit.t().to_owned());
        let z0 = solve_mat(&self.x_entry.t().to_owned());
        let zd = solve_mat(&self.x_derivative.t().to_owned());
        let n = self.x_exit.nrows();
        let mut q1 = Array1::<f64>::zeros(n);
        let mut q0 = Array1::<f64>::zeros(n);
        let mut qd = Array1::<f64>::zeros(n);
        for i in 0..n {
            q1[i] = self.x_exit.row(i).dot(&z1.column(i)).max(0.0);
            q0[i] = self.x_entry.row(i).dot(&z0.column(i)).max(0.0);
            qd[i] = self.x_derivative.row(i).dot(&zd.column(i)).max(0.0);
        }

        // Assemble S(rho) in the same scaling used by update_state.
        let mut s_total = Array2::<f64>::zeros((p, p));
        for block in &self.penalties.blocks {
            if block.lambda == 0.0 {
                continue;
            }
            let r = block.range.clone();
            let scale = 2.0 * block.lambda;
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
                logdet_s += ev.ln();
                let inv_ev = 1.0 / ev;
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
            let a_k_beta_block = block.matrix.dot(&b_block).mapv(|v| 2.0 * lambda * v);
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
                    trace_hinv_ak += solved_basis[[i, j_local]]
                        * (2.0 * lambda * block.matrix[[j_local, i_local]]);
                }
            }

            // Tensor-free contraction of third-derivative terms:
            // + sum_i exp(eta_exit_i)  (x_exit_i^T u_k) q1_i
            // - sum_i exp(eta_entry_i) (x_entry_i^T u_k) q0_i
            // - 2 sum_events (d_i^T u_k) qd_i / (d_i^T beta)^3
            // The last term is from differentiating delta_i * log(d_i^T beta).
            let mut trace_third = 0.0_f64;
            for i in 0..n {
                let w_i = self.sample_weight[i];
                trace_third += w_i * exp_exit[i] * s1k[i] * q1[i];
                trace_third -= w_i * exp_entry[i] * s0k[i] * q0[i];
                if self.event_target[i] > 0 {
                    let s_i = deriv_raw[i];
                    if s_i > guard {
                        trace_third -= 2.0 * w_i * sdk[i] * qd[i] / (s_i * s_i * s_i);
                    }
                }
            }
            let t_k = trace_hinv_ak + trace_third;

            let mut p_k = 0.0_f64;
            for (i_local, i) in r.clone().enumerate() {
                for (j_local, j) in r.clone().enumerate() {
                    // p_k = tr(S^+ A_k) restricted to this block.
                    p_k += s_pinv[[i, j]] * (2.0 * lambda * block.matrix[[j_local, i_local]]);
                }
            }

            grad[k] = 0.5 * beta.dot(&a_k_beta) + 0.5 * t_k - 0.5 * p_k;
        }

        Ok((objective, grad))
    }
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
        let expected = 2.0 * 1.7 * array![[2.0, 0.5], [0.5, 3.0]].dot(&b_block);

        assert!((grad[1] - expected[0]).abs() < 1e-12);
        assert!((grad[2] - expected[1]).abs() < 1e-12);
        assert!((h[[1, 1]] - 2.0 * 1.7 * 2.0).abs() < 1e-12);
        assert!((h[[1, 2]] - 2.0 * 1.7 * 0.5).abs() < 1e-12);
        assert!((h[[2, 1]] - 2.0 * 1.7 * 0.5).abs() < 1e-12);
        assert!((h[[2, 2]] - 2.0 * 1.7 * 3.0).abs() < 1e-12);
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
}
