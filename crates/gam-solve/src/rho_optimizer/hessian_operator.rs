use super::*;

// The outer Hessian operator contract lives in `gam-problem`. Re-exported here
// while solver-private wrappers stay in `solver`.
pub use gam_problem::{
    DeclaredHessianForm, Derivative, HessianMaterialization, HessianOperator,
    OuterStrategyError,
};

pub(crate) struct RhoBlockAdditiveOuterHessian {
    pub(crate) base: Arc<dyn HessianOperator>,
    pub(crate) rho_block: Array2<f64>,
    pub(crate) dim: usize,
}

impl HessianOperator for RhoBlockAdditiveOuterHessian {
    fn dim(&self) -> usize {
        self.dim
    }

    fn matvec(&self, v: &Array1<f64>) -> Result<Array1<f64>, String> {
        if v.len() != self.dim {
            return Err(OuterStrategyError::OperatorShape {
                reason: format!(
                    "outer Hessian operator input length mismatch: got {}, expected {}",
                    v.len(),
                    self.dim
                ),
            }
            .into());
        }
        let mut out = self.base.matvec(v)?;
        let k = self.rho_block.nrows();
        if k > 0 {
            let rho_v = v.slice(ndarray::s![..k]).to_owned();
            let rho_out = self.rho_block.dot(&rho_v);
            out.slice_mut(ndarray::s![..k]).scaled_add(1.0, &rho_out);
        }
        Ok(out)
    }

    /// Zero-alloc override for the inner-CG hot path.
    ///
    /// Delegates to `base.apply_into` (which may itself be zero-alloc when the
    /// base overrides) then adds the rho-block correction using a row-dot loop
    /// rather than materialising an intermediate `rho_v.to_owned()` +
    /// `rho_block.dot(&rho_v)` pair.
    fn apply_into(&self, v: &Array1<f64>, out: &mut Array1<f64>) -> Result<(), String> {
        if v.len() != self.dim {
            return Err(OuterStrategyError::OperatorShape {
                reason: format!(
                    "outer Hessian operator input length mismatch: got {}, expected {}",
                    v.len(),
                    self.dim
                ),
            }
            .into());
        }
        if out.len() != self.dim {
            return Err(OuterStrategyError::OperatorShape {
                reason: format!(
                    "outer Hessian apply_into output length mismatch: got {}, expected {}",
                    out.len(),
                    self.dim
                ),
            }
            .into());
        }
        self.base.apply_into(v, out)?;
        let k = self.rho_block.nrows();
        if k > 0 {
            let v_top = v.slice(ndarray::s![..k]);
            for i in 0..k {
                out[i] += self.rho_block.row(i).dot(&v_top);
            }
        }
        Ok(())
    }

    /// Batched apply: delegate to the inner operator's `mul_mat` (which may
    /// itself parallelize), then add `rho_block` to the leading `k × k`
    /// block. This propagates the batched-amortization benefit to wrappers
    /// — `materialize_dense` (which goes through `mul_mat(eye)`) and any
    /// future K-column inner-CG batching path.
    fn mul_mat(&self, factor: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
        let mut out = self.base.mul_mat(factor)?;
        let k = self.rho_block.nrows();
        if k > 0 {
            if k > out.nrows() {
                return Err(OuterStrategyError::RhoBlockShape {
                    reason: format!(
                        "rho-block Hessian update shape mismatch: rho_block is {}x{}, mul_mat output has {} rows",
                        self.rho_block.nrows(),
                        self.rho_block.ncols(),
                        out.nrows()
                    ),
                }
                .into());
            }
            // Update the leading-k rows of `out` by the rho_block contribution
            // to the first k rows of v: out[..k, :] += rho_block · factor[..k, :].
            let factor_top = factor.slice(ndarray::s![..k, ..]);
            let rho_contrib = self.rho_block.dot(&factor_top);
            out.slice_mut(ndarray::s![..k, ..])
                .scaled_add(1.0, &rho_contrib);
        }
        Ok(out)
    }

    fn is_cheap_to_materialize(&self) -> bool {
        self.base.is_cheap_to_materialize()
    }

    fn materialization_capability(&self) -> HessianMaterialization {
        self.base.materialization_capability()
    }
}

/// Upper safety bound for operator materialization after the operator has
/// explicitly declared that dense probing is cheap. Dimension alone is never
/// sufficient: a 50-column operator can still mean 50 full row-streaming CTN,
/// Duchon, or survival passes.
pub(crate) const OUTER_HVP_MATERIALIZE_MAX_DIM: usize = 64;
