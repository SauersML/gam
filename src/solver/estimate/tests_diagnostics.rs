//! Debug-only finite-difference probes on `ExternalJointHyperEvaluator`.
//!
//! These inherent methods expose the dense effective Hessian, its projected
//! log-determinant, and the converged `(η, weights, c)` state at a fixed `theta`
//! so the analytic ψ-trace formulas can be checked against centered finite
//! differences from the test crate. They are compiled only under `cfg(test)`.

use super::*;

impl<'a> ExternalJointHyperEvaluator<'a> {
    /// DEBUG ONLY: run PIRLS at `theta` (cost-only path) and return the dense
    /// effective Hessian `H_total = X' W_F X + S_λ + ridge I` in the
    /// transformed basis. This is the same matrix the analytic operator
    /// differentiates, so centered finite-difference probes of this H w.r.t.
    /// ψ should match the analytic `B_i + correction`.
    pub fn debug_full_h(
        &mut self,
        x: &DesignMatrix,
        s_list: &[BlockwisePenalty],
        nullspace_dims: &[usize],
        linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,
        theta: &Array1<f64>,
        rho_dim: usize,
        context: &str,
    ) -> Result<Array2<f64>, EstimationError> {
        if rho_dim > theta.len() {
            crate::bail_invalid_estim!(
                "rho_dim {} exceeds theta dimension {}",
                rho_dim,
                theta.len()
            );
        }
        self.prepare_eval_state_cost_only(
            x,
            s_list,
            nullspace_dims,
            linear_constraints,
            theta,
            rho_dim,
            None,
            context,
            None,
        )?;
        let rho = theta.slice(s![..rho_dim]).to_owned();
        // Drive PIRLS at this theta (populates eval bundle cache).
        self.reml_state.compute_cost(&rho)?;
        self.reml_state.objective_innerhessian(&rho)
    }

    /// Debug-only: return the *projected* Hessian log-determinant
    /// `log|U_Sᵀ H U_S|_+` at the PIRLS state driven to convergence at this
    /// `theta`.  This is the same scalar that the REML/LAML cost identity
    /// uses (via `hop.logdet() + hessian_logdet_correction`), so a centered
    /// finite difference of it along ψ gives the analytic `d/dψ log|H_proj|`
    /// that the production trace formula computes — i.e. the correct
    /// finite-difference reference for the penalty-subspace projection invariant.
    pub fn debug_logdet_h_proj(
        &mut self,
        x: &DesignMatrix,
        s_list: &[BlockwisePenalty],
        nullspace_dims: &[usize],
        linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,
        theta: &Array1<f64>,
        rho_dim: usize,
        context: &str,
    ) -> Result<f64, EstimationError> {
        if rho_dim > theta.len() {
            crate::bail_invalid_estim!(
                "rho_dim {} exceeds theta dimension {}",
                rho_dim,
                theta.len()
            );
        }
        self.prepare_eval_state_cost_only(
            x,
            s_list,
            nullspace_dims,
            linear_constraints,
            theta,
            rho_dim,
            None,
            context,
            None,
        )?;
        let rho = theta.slice(s![..rho_dim]).to_owned();
        self.reml_state.compute_cost(&rho)?;
        self.reml_state.objective_logdet_h_proj(&rho)
    }

    /// Debug-only: return `(η, finalweights, solve_c_array)` at this theta.
    pub fn debug_full_eta_w_c(
        &mut self,
        x: &DesignMatrix,
        s_list: &[BlockwisePenalty],
        nullspace_dims: &[usize],
        linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,
        theta: &Array1<f64>,
        rho_dim: usize,
        context: &str,
    ) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), EstimationError> {
        self.prepare_eval_state_cost_only(
            x,
            s_list,
            nullspace_dims,
            linear_constraints,
            theta,
            rho_dim,
            None,
            context,
            None,
        )?;
        let rho = theta.slice(s![..rho_dim]).to_owned();
        self.reml_state.compute_cost(&rho)?;
        self.reml_state.debug_eta_w_c(&rho)
    }
}
