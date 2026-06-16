//! The outer (ρ) objective: the `inner_blockwise_fit` driver, the joint
//! derivative providers (borrowed / owned / Jeffreys-aware), the ext-coord bundle
//! and scaled hyper-operators, inner-assembly construction, the unified joint
//! cost/gradient/EFS evaluators, and the outer-objective entry points
//! (`outerobjectivegradienthessian_internal`, `outerobjectiveefs`). Also the
//! blockwise-fit assembly-from-parts, warm-start carriers, outer-Hessian operator
//! wrappers, and labeled-lambda layout helpers shared with the outer engine.

use super::*;

impl crate::solver::rho_optimizer::OuterHessianOperator for OwnedDenseOuterHessianOperator {
    fn dim(&self) -> usize {
        self.matrix.nrows()
    }

    fn matvec(&self, v: &Array1<f64>) -> Result<Array1<f64>, String> {
        if v.len() != self.matrix.ncols() {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "batched dense outer Hessian matvec length mismatch: got {}, expected {}",
                    v.len(),
                    self.matrix.ncols()
                ),
            }
            .into());
        }
        Ok(self.matrix.dot(v))
    }

    /// Zero-alloc override: write `matrix · v` directly into `out` using a
    /// row-dot loop, avoiding the `matrix.dot(v)` allocation.
    fn apply_into(&self, v: &Array1<f64>, out: &mut Array1<f64>) -> Result<(), String> {
        if v.len() != self.matrix.ncols() {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "batched dense outer Hessian apply_into input length mismatch: got {}, expected {}",
                    v.len(),
                    self.matrix.ncols()
                ),
            }
            .into());
        }
        if out.len() != self.matrix.nrows() {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "batched dense outer Hessian apply_into output length mismatch: got {}, expected {}",
                    out.len(),
                    self.matrix.nrows()
                ),
            }
            .into());
        }
        for (row, cell) in self.matrix.rows().into_iter().zip(out.iter_mut()) {
            *cell = row.dot(v);
        }
        Ok(())
    }

    fn is_cheap_to_materialize(&self) -> bool {
        true
    }
}

pub(crate) struct LabeledOuterHessianOperator {
    pub(crate) base: Arc<dyn crate::solver::rho_optimizer::OuterHessianOperator>,
    pub(crate) physical_to_outer: Vec<Option<usize>>,
    pub(crate) outer_dim: usize,
    /// Scratch buffers reused across `apply_into` calls to avoid
    /// per-call allocation of the permuted input and output vectors.
    /// `(physical_in, physical_out)`, each of length `physical_to_outer.len()`.
    pub(crate) scratch: std::sync::Mutex<(ndarray::Array1<f64>, ndarray::Array1<f64>)>,
}

impl LabeledOuterHessianOperator {
    pub(crate) fn new(
        base: Arc<dyn crate::solver::rho_optimizer::OuterHessianOperator>,
        layout: &PenaltyLabelLayout,
    ) -> Self {
        let n_physical = layout.physical_to_outer.len();
        Self {
            base,
            physical_to_outer: layout.physical_to_outer.clone(),
            outer_dim: layout.initial_rho.len(),
            scratch: std::sync::Mutex::new((
                ndarray::Array1::zeros(n_physical),
                ndarray::Array1::zeros(n_physical),
            )),
        }
    }
}

impl crate::solver::rho_optimizer::OuterHessianOperator for LabeledOuterHessianOperator {
    fn dim(&self) -> usize {
        self.outer_dim
    }

    fn matvec(&self, v: &Array1<f64>) -> Result<Array1<f64>, String> {
        if v.len() != self.outer_dim {
            return Err(format!(
                "labeled outer Hessian input length mismatch: got {}, expected {}",
                v.len(),
                self.outer_dim
            ));
        }
        let mut physical = Array1::<f64>::zeros(self.physical_to_outer.len());
        for (physical_idx, outer_idx) in self.physical_to_outer.iter().enumerate() {
            physical[physical_idx] = outer_idx.map(|idx| v[idx]).unwrap_or(0.0);
        }
        let physical_out = self.base.matvec(&physical)?;
        if physical_out.len() != self.physical_to_outer.len() {
            return Err(format!(
                "labeled outer Hessian physical matvec length mismatch: got {}, expected {}",
                physical_out.len(),
                self.physical_to_outer.len()
            ));
        }
        let mut out = Array1::<f64>::zeros(self.outer_dim);
        for (physical_idx, outer_idx) in self.physical_to_outer.iter().enumerate() {
            if let Some(outer_idx) = *outer_idx {
                out[outer_idx] += physical_out[physical_idx];
            }
        }
        Ok(out)
    }

    /// Zero-alloc override: reuses hoisted scratch buffers to avoid the
    /// per-call `physical` and `out` allocations in `matvec`.
    fn apply_into(
        &self,
        v: &ndarray::Array1<f64>,
        out: &mut ndarray::Array1<f64>,
    ) -> Result<(), String> {
        if v.len() != self.outer_dim {
            return Err(format!(
                "labeled outer Hessian apply_into input length mismatch: got {}, expected {}",
                v.len(),
                self.outer_dim
            ));
        }
        if out.len() != self.outer_dim {
            return Err(format!(
                "labeled outer Hessian apply_into output length mismatch: got {}, expected {}",
                out.len(),
                self.outer_dim
            ));
        }
        let mut guard = self
            .scratch
            .lock()
            .map_err(|_| "labeled outer Hessian scratch lock poisoned".to_string())?;
        let (physical_in, physical_out) = &mut *guard;
        for (physical_idx, outer_idx) in self.physical_to_outer.iter().enumerate() {
            physical_in[physical_idx] = outer_idx.map(|idx| v[idx]).unwrap_or(0.0);
        }
        self.base.apply_into(physical_in, physical_out)?;
        if physical_out.len() != self.physical_to_outer.len() {
            return Err(format!(
                "labeled outer Hessian physical apply_into length mismatch: got {}, expected {}",
                physical_out.len(),
                self.physical_to_outer.len()
            ));
        }
        out.fill(0.0);
        for (physical_idx, outer_idx) in self.physical_to_outer.iter().enumerate() {
            if let Some(outer_idx) = *outer_idx {
                out[outer_idx] += physical_out[physical_idx];
            }
        }
        Ok(())
    }

    fn mul_mat(&self, factor: ndarray::ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
        if factor.nrows() != self.outer_dim {
            return Err(format!(
                "labeled outer Hessian factor row mismatch: got {}, expected {}",
                factor.nrows(),
                self.outer_dim
            ));
        }
        let mut physical_factor =
            Array2::<f64>::zeros((self.physical_to_outer.len(), factor.ncols()));
        for (physical_idx, outer_idx) in self.physical_to_outer.iter().enumerate() {
            if let Some(outer_idx) = *outer_idx {
                physical_factor
                    .row_mut(physical_idx)
                    .assign(&factor.row(outer_idx));
            }
        }
        let physical_out = self.base.mul_mat(physical_factor.view())?;
        if physical_out.nrows() != self.physical_to_outer.len() {
            return Err(format!(
                "labeled outer Hessian physical output row mismatch: got {}, expected {}",
                physical_out.nrows(),
                self.physical_to_outer.len()
            ));
        }
        let mut out = Array2::<f64>::zeros((self.outer_dim, factor.ncols()));
        for (physical_idx, outer_idx) in self.physical_to_outer.iter().enumerate() {
            if let Some(outer_idx) = *outer_idx {
                let physical_row = physical_out.row(physical_idx);
                out.row_mut(outer_idx).scaled_add(1.0, &physical_row);
            }
        }
        Ok(out)
    }

    fn is_cheap_to_materialize(&self) -> bool {
        self.base.is_cheap_to_materialize()
    }

    fn materialization_capability(
        &self,
    ) -> crate::solver::rho_optimizer::OuterHessianMaterialization {
        self.base.materialization_capability()
    }
}

pub(crate) fn custom_family_batched_outer_hessian_operator<F: CustomFamily>(
    family: &F,
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
    rho: &Array1<f64>,
    workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
    eval_mode: EvalMode,
) -> Result<Option<Arc<dyn crate::solver::rho_optimizer::OuterHessianOperator>>, String> {
    if eval_mode != EvalMode::ValueGradientHessian {
        return Ok(None);
    }
    let Some(terms) =
        family.batched_outer_hessian_terms(states, specs, derivative_blocks, rho, workspace)?
    else {
        return Ok(None);
    };
    match terms.outer_hessian {
        crate::solver::rho_optimizer::HessianResult::Operator(operator) => Ok(Some(operator)),
        crate::solver::rho_optimizer::HessianResult::Analytic(matrix) => {
            Ok(Some(Arc::new(OwnedDenseOuterHessianOperator { matrix })))
        }
        crate::solver::rho_optimizer::HessianResult::Unavailable => Ok(None),
    }
}

pub(crate) fn outer_efs_result_to_joint_hyper_efs_result(
    efs_eval: crate::solver::rho_optimizer::EfsEval,
    warm_start: ConstrainedWarmStart,
    inner_converged: bool,
) -> CustomFamilyJointHyperEfsResult {
    CustomFamilyJointHyperEfsResult {
        efs_eval,
        warm_start: CustomFamilyWarmStart { inner: warm_start },
        inner_converged,
    }
}

// Unified exact joint hyper-calculus over theta = [rho, psi].
//
// The correct outer problem is not “a rho objective plus a separate psi
// objective”. It is one profiled/Laplace surface over one flattened hypervector
//
//   theta = [rho, psi],
//
// one flattened joint coefficient vector
//
//   beta = [beta_1; ...; beta_B],
//
// and one joint exact mode system
//
//   F(beta, theta) := V_beta(beta, theta) = 0,
//   H(beta, theta) := V_beta_beta(beta, theta).
//
// For every hypercoordinate theta_i we need the fixed-beta objects
//
//   V_i = partial_{theta_i} V,
//   g_i = partial_{theta_i} F,
//   H_i = partial_{theta_i} H,
//
// and for every pair (i, j)
//
//   V_ij, g_ij, H_ij,
//
// together with the beta-curvature contractions
//
//   D_beta H[u],
//   D_beta^2 H[u, v],
//   T_i[u] := D_beta H_i[u].
//
// The exact profiled mode response and total Hessian drifts are then
//
//   beta_i  = -H^{-1} g_i,
//   beta_ij = -H^{-1}(g_ij + H_i beta_j + H_j beta_i + D_beta H[beta_i] beta_j),
//
//   dot H_i
//   = H_i + D_beta H[beta_i],
//
//   ddot H_ij
//   = H_ij
//     + T_i[beta_j]
//     + T_j[beta_i]
//     + D_beta H[beta_ij]
//     + D_beta^2 H[beta_i, beta_j].
//
// Hence the exact joint profiled/Laplace derivatives are
//
//   J_i
//   = V_i + 0.5 tr(H^{-1} dot H_i) - 0.5 partial_i log|S(theta)|_+,
//
//   J_ij
//   = (V_ij - g_i^T H^{-1} g_j)
//     + 0.5 [ tr(H^{-1} ddot H_ij)
//             - tr(H^{-1} dot H_j H^{-1} dot H_i) ]
//     - 0.5 partial^2_{ij} log|S(theta)|_+.
//
// In this unified view rho and psi are the same outer calculus. They differ
// only in where their fixed-beta derivative objects come from:
//
// - rho coordinates often contribute only through the penalty surface,
//     but the generic assembler intentionally treats the penalty as S(theta),
//     not S(rho), so mixed rho/psi penalty terms are allowed whenever realized
//     component penalties move with psi:
//       V_i  = D_i  + 0.5 beta^T S_i beta
//       g_i  = D_beta_i  + S_i beta
//       H_i  = D_beta_beta_i + S_i
//       V_ij = D_ij + 0.5 beta^T S_ij beta
//       g_ij = D_beta_ij + S_ij beta
//       H_ij = D_beta_beta_ij + S_ij.
//
// - psi coordinates come from the family-specific joint exact psi hooks, while
//   the generic assembler still owns any realized-penalty motion through
//   S_i / S_ij:
//     objective_psi            <-> V_i
//     score_psi                <-> g_i
//     hessian_psi              <-> H_i
//     objective_psi_psi        <-> V_ij
//     score_psi_psi            <-> g_ij
//     hessian_psi_psi          <-> H_ij
//     D_beta H_psi[u]          <-> T_i[u].
//
// For coupled families this means any block-local psi path is wrong. Even when
// g_i is sparse or penalty-local, beta_i is defined by the full joint solve
//
//   beta_i = -H^{-1} g_i,
//
// so every exact outer derivative must be assembled in this joint flattened
// space.

pub(crate) fn with_block_geometry<F: CustomFamily + ?Sized, T>(
    family: &F,
    block_states: &[ParameterBlockState],
    spec: &ParameterBlockSpec,
    block_idx: usize,
    f: impl FnOnce(&DesignMatrix, &Array1<f64>) -> Result<T, String>,
) -> Result<T, String> {
    if family.block_geometry_is_dynamic() {
        let (x_dyn, off_dyn) = family.block_geometry(block_states, spec)?;
        let expected_rows = spec.solver_design().nrows();
        if x_dyn.nrows() != expected_rows {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "block {block_idx} dynamic design row mismatch: got {}, expected {}",
                    x_dyn.nrows(),
                    expected_rows
                ),
            }
            .into());
        }
        if x_dyn.ncols() != spec.design.ncols() {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "block {block_idx} dynamic design col mismatch: got {}, expected {}",
                    x_dyn.ncols(),
                    spec.design.ncols()
                ),
            }
            .into());
        }
        if off_dyn.len() != expected_rows {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "block {block_idx} dynamic offset length mismatch: got {}, expected {}",
                    off_dyn.len(),
                    expected_rows
                ),
            }
            .into());
        }
        f(&x_dyn, &off_dyn)
    } else {
        f(spec.solver_design(), spec.solver_offset())
    }
}

// Penalty-label layout + labeled log-λ (de)aggregation helpers moved to the
// sibling `penalty_labels` module (re-exported through the parent), keeping this
// file under the tracked-file line limit.
