//! Debug finite-difference probes on `ExternalJointHyperEvaluator`.
//!
//! These inherent methods expose the dense effective Hessian, its projected
//! log-determinant, and the converged `(η, weights, c)` state at a fixed `theta`
//! so the analytic ψ-trace formulas can be checked against centered finite
//! differences from regression tests — including the #1601-orphaned
//! design-assembly guards re-homed into the separate gam-models crate, which is
//! why these are `pub` rather than `pub(crate)`. They are compiled
//! unconditionally (the workspace ban-scanner forbids feature gating); a `pub`
//! debug helper the production path never calls is inert by construction.

use super::*;
use super::joint_hyper::ExternalJointHyperEvaluator;

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
}
