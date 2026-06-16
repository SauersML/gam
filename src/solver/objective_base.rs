use ndarray::{Array1, Array2};
use std::sync::Arc;

use crate::solver::rho_optimizer::{
    OuterHessianOperator, OuterStrategyError, RhoBlockAdditiveOuterHessian,
};

pub(crate) fn outer_strategy_contract_panic(message: impl Into<String>) -> ! {
    std::panic::panic_any(message.into())
}

/// Shared outer-objective result used by optimizer-facing objective
/// implementations.
pub struct OuterEval {
    pub cost: f64,
    pub gradient: Array1<f64>,
    pub hessian: HessianResult,
    /// Optional inner-solver iterate at this rho. Families whose inner solve
    /// produces a PIRLS beta populate this so the persistent-cache layer can
    /// store `(rho, beta)` together.
    pub inner_beta_hint: Option<Array1<f64>>,
}

impl OuterEval {
    /// Conventional representation of an infeasible trial point.
    pub fn infeasible(n_params: usize) -> Self {
        Self {
            cost: f64::INFINITY,
            gradient: Array1::zeros(n_params),
            hessian: HessianResult::Unavailable,
            inner_beta_hint: None,
        }
    }

    pub(crate) fn value_only(
        cost: f64,
        n_params: usize,
        inner_beta_hint: Option<Array1<f64>>,
    ) -> Self {
        Self {
            cost,
            gradient: Array1::zeros(n_params),
            hessian: HessianResult::Unavailable,
            inner_beta_hint,
        }
    }
}

/// Explicit Hessian result replacing `Option<Array2<f64>>`.
pub enum HessianResult {
    /// Analytic Hessian was computed and returned.
    Analytic(Array2<f64>),
    /// Analytic Hessian is available as an exact Hessian-vector product.
    Operator(Arc<dyn OuterHessianOperator>),
    /// No analytic Hessian available for this model path.
    Unavailable,
}

impl Clone for OuterEval {
    fn clone(&self) -> Self {
        Self {
            cost: self.cost,
            gradient: self.gradient.clone(),
            hessian: self.hessian.clone(),
            inner_beta_hint: self.inner_beta_hint.clone(),
        }
    }
}

impl std::fmt::Debug for OuterEval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OuterEval")
            .field("cost", &self.cost)
            .field("gradient", &self.gradient)
            .field("hessian", &self.hessian)
            .finish()
    }
}

impl Clone for HessianResult {
    fn clone(&self) -> Self {
        match self {
            Self::Analytic(h) => Self::Analytic(h.clone()),
            Self::Operator(op) => Self::Operator(Arc::clone(op)),
            Self::Unavailable => Self::Unavailable,
        }
    }
}

impl std::fmt::Debug for HessianResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Analytic(h) => f
                .debug_tuple("Analytic")
                .field(&format!("{}x{}", h.nrows(), h.ncols()))
                .finish(),
            Self::Operator(op) => f
                .debug_tuple("Operator")
                .field(&format!("dim={}", op.dim()))
                .finish(),
            Self::Unavailable => f.write_str("Unavailable"),
        }
    }
}

impl HessianResult {
    /// Extract the Hessian matrix, panicking if unavailable.
    pub fn unwrap_analytic(self) -> Array2<f64> {
        match self {
            HessianResult::Analytic(h) => h,
            HessianResult::Operator(_) => outer_strategy_contract_panic(
                "expected dense analytic Hessian but got HessianResult::Operator",
            ),
            HessianResult::Unavailable => outer_strategy_contract_panic(
                "expected analytic Hessian but got HessianResult::Unavailable",
            ),
        }
    }

    /// Returns `true` if an analytic Hessian is present in any exact form.
    pub fn is_analytic(&self) -> bool {
        matches!(
            self,
            HessianResult::Analytic(_) | HessianResult::Operator(_)
        )
    }

    /// Convert to the optional Hessian shape used by the opt bridge.
    pub fn into_option(self) -> Option<Array2<f64>> {
        match self {
            HessianResult::Analytic(h) => Some(h),
            HessianResult::Operator(_) | HessianResult::Unavailable => None,
        }
    }

    pub fn dim(&self) -> Option<usize> {
        match self {
            HessianResult::Analytic(h) => Some(h.nrows()),
            HessianResult::Operator(op) => Some(op.dim()),
            HessianResult::Unavailable => None,
        }
    }

    pub fn materialize_dense(&self) -> Result<Option<Array2<f64>>, String> {
        match self {
            HessianResult::Analytic(h) => Ok(Some(h.clone())),
            HessianResult::Operator(op) => op.materialize_dense().map(Some),
            HessianResult::Unavailable => Ok(None),
        }
    }

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
