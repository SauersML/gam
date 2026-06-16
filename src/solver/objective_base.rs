use ndarray::Array2;
use std::sync::Arc;

use crate::solver::rho_optimizer::{OuterStrategyError, RhoBlockAdditiveOuterHessian};

// `OuterEval` and `HessianResult` moved DOWN to the `crate::solver_contract`
// lower layer (#1135) so the `families` layer can name them without importing
// up into `solver`. Re-exported here so existing `crate::solver::objective_base`
// paths keep resolving. The solver-internal `add_rho_block_dense` extension
// (which needs the solver-private `RhoBlockAdditiveOuterHessian`) stays here.
pub use crate::solver_contract::{HessianResult, OuterEval};

impl HessianResult {
    /// Solver-internal extension: additive rho-block update. Lives here (not in
    /// the lower contract layer) because it constructs the solver-private
    /// `RhoBlockAdditiveOuterHessian` wrapper. Inherent impls may live in any
    /// module of the defining crate.
    pub fn add_rho_block_dense(&mut self, rho_block: &Array2<f64>) -> Result<(), String> {
        add_rho_block_dense_to_hessian(self, rho_block)
    }
}

fn add_rho_block_dense_to_hessian(
    hessian: &mut HessianResult,
    rho_block: &Array2<f64>,
) -> Result<(), String> {
    if rho_block.nrows() != rho_block.ncols() {
        return Err(OuterStrategyError::RhoBlockShape {
            reason: format!(
                "rho-block Hessian update must be square, got {}x{}",
                rho_block.nrows(),
                rho_block.ncols()
            ),
        }
        .into());
    }
    match hessian {
        HessianResult::Analytic(h) => {
            if rho_block.nrows() > h.nrows() || rho_block.ncols() > h.ncols() {
                return Err(OuterStrategyError::RhoBlockShape {
                    reason: format!(
                        "rho-block Hessian update shape mismatch: got {}x{}, outer Hessian is {}x{}",
                        rho_block.nrows(),
                        rho_block.ncols(),
                        h.nrows(),
                        h.ncols()
                    ),
                }
                .into());
            }
            let k = rho_block.nrows();
            let mut sl = h.slice_mut(ndarray::s![..k, ..k]);
            sl += rho_block;
            Ok(())
        }
        HessianResult::Operator(op) => {
            let base = Arc::clone(op);
            let dim = base.dim();
            if rho_block.nrows() > dim {
                return Err(OuterStrategyError::RhoBlockShape {
                    reason: format!(
                        "rho-block Hessian update dimension mismatch: got {}x{}, operator dim is {}",
                        rho_block.nrows(),
                        rho_block.ncols(),
                        dim
                    ),
                }
                .into());
            }
            *hessian = HessianResult::Operator(Arc::new(RhoBlockAdditiveOuterHessian {
                base,
                rho_block: rho_block.clone(),
                dim,
            }));
            Ok(())
        }
        HessianResult::Unavailable => Ok(()),
    }
}

#[inline]
pub(crate) fn failed_inner_residual_barrier_cost(
    cost: f64,
    inner_failed_max_iterations: bool,
    relative_gradient_norm: f64,
) -> f64 {
    if !cost.is_finite() || !inner_failed_max_iterations {
        return cost;
    }
    if relative_gradient_norm.is_finite() {
        cost + 0.5 * relative_gradient_norm * relative_gradient_norm
    } else {
        f64::INFINITY
    }
}
