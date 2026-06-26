//! The neutral ψ (hyperparameter) design-derivative contract carriers and
//! operator traits shared by the `CustomFamily` trait layer (`gam-model-api`)
//! and the solver: the per-block ψ-derivative carrier, the matrix-free
//! `CustomFamilyPsiDerivativeOperator` trait (+ its dense-materialization
//! extension), and the joint-Hessian source-preference / materialization-intent
//! enums.
//!
//! These carry no dependency on the `CustomFamily` trait itself, so they live
//! in the neutral `gam-problem` crate and are re-exported upward, keeping a
//! single definition shared across crates.

use crate::{BasisError, PenaltyMatrix};
use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1};
use std::any::Any;
use std::ops::Range;
use std::sync::Arc;

#[derive(Clone)]
pub struct CustomFamilyBlockPsiDerivative {
    pub penalty_index: Option<usize>,
    pub x_psi: Array2<f64>,
    pub s_psi: Array2<f64>,
    pub s_psi_components: Option<Vec<(usize, Array2<f64>)>>,
    pub s_psi_penalty_components: Option<Vec<(usize, PenaltyMatrix)>>,
    pub x_psi_psi: Option<Vec<Array2<f64>>>,
    pub s_psi_psi: Option<Vec<Array2<f64>>>,
    pub s_psi_psi_components: Option<Vec<Vec<(usize, Array2<f64>)>>>,
    pub s_psi_psi_penalty_components: Option<Vec<Vec<(usize, PenaltyMatrix)>>>,
    pub implicit_operator: Option<Arc<dyn CustomFamilyPsiDerivativeOperator>>,
    pub implicit_axis: usize,
    pub implicit_group_id: Option<usize>,
}

pub type SharedDerivativeBlocks = Arc<Vec<Vec<CustomFamilyBlockPsiDerivative>>>;

impl CustomFamilyBlockPsiDerivative {
    /// Public constructor for use in tests and external consumers.
    /// Sets `implicit_operator` to `None`.
    pub fn new(
        penalty_index: Option<usize>,
        x_psi: Array2<f64>,
        s_psi: Array2<f64>,
        s_psi_components: Option<Vec<(usize, Array2<f64>)>>,
        x_psi_psi: Option<Vec<Array2<f64>>>,
        s_psi_psi: Option<Vec<Array2<f64>>>,
        s_psi_psi_components: Option<Vec<Vec<(usize, Array2<f64>)>>>,
    ) -> Self {
        Self {
            penalty_index,
            x_psi,
            s_psi,
            s_psi_components,
            s_psi_penalty_components: None,
            x_psi_psi,
            s_psi_psi,
            s_psi_psi_components,
            s_psi_psi_penalty_components: None,
            implicit_operator: None,
            implicit_axis: 0,
            implicit_group_id: None,
        }
    }
}

pub trait CustomFamilyPsiDerivativeOperator: Send + Sync + Any {
    fn as_any(&self) -> &dyn Any;
    fn n_data(&self) -> usize;
    fn p_out(&self) -> usize;
    fn transpose_mul(
        &self,
        axis: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, BasisError>;
    fn forward_mul(&self, axis: usize, u: &ArrayView1<'_, f64>) -> Result<Array1<f64>, BasisError>;
    fn transpose_mul_second_diag(
        &self,
        axis: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, BasisError>;
    fn transpose_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, BasisError>;
    fn forward_mul_second_diag(
        &self,
        axis: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, BasisError>;
    fn forward_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, BasisError>;
    fn row_chunk_first(&self, axis: usize, rows: Range<usize>) -> Result<Array2<f64>, BasisError>;
    /// Single-row specialization of `row_chunk_first`. Default implementation
    /// delegates to `row_chunk_first(axis, row..row+1)` and copies the
    /// resulting row into the output buffer; implementations that can avoid
    /// the temporary matrix allocation should override this method.
    fn row_vector_first_into(
        &self,
        axis: usize,
        row: usize,
        mut out: ArrayViewMut1<'_, f64>,
    ) -> Result<(), BasisError> {
        let chunk = self.row_chunk_first(axis, row..row + 1)?;
        out.assign(&chunk.row(0));
        Ok(())
    }
    fn row_chunk_second_diag(
        &self,
        axis: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, BasisError>;
    fn row_chunk_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, BasisError>;

    /// Optional upcast to the dense materialization surface. Production exact
    /// paths should prefer the analytic matvec / row-chunk methods above and
    /// avoid forming the full derivative matrix; implementations that *do*
    /// support dense materialization (used by diagnostics, tests, and
    /// small-data fallbacks) should override this to return `Some(self)`.
    fn as_materializable(&self) -> Option<&dyn MaterializablePsiDerivativeOperator> {
        None
    }
}

/// Diagnostic / small-data extension that exposes dense materialization of
/// `\partial X / \partial \psi`. Production exact-Hessian code MUST NOT depend
/// on dense second-derivative materialization; second-order paths use the
/// row-chunk and matvec methods on [`CustomFamilyPsiDerivativeOperator`].
pub trait MaterializablePsiDerivativeOperator: CustomFamilyPsiDerivativeOperator {
    fn materialize_first(&self, axis: usize) -> Result<Array2<f64>, BasisError>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JointHessianSourcePreference {
    Dense,
    Operator,
}

/// What the consumer is going to *do* with the joint Hessian. This is the
/// intent half of #738's capability-vs-representation split: the call site
/// states what it needs, and the workspace picks the cheapest representation
/// that serves that need (rather than a single per-workspace preference being
/// applied uniformly regardless of how the result is consumed).
///
/// The distinction matters because the same workspace serves several
/// consumers with opposite ideal representations:
/// - the inner Newton/PCG solve only ever applies `H · v`, so a matrix-free
///   HVP (`Operator`) is ideal and a dense build is pure waste;
/// - the REML logdet term factorizes `H + S_λ` (Cholesky / eigendecomposition),
///   so it must hold a dense matrix anyway — handing it an `Operator` only
///   forces an immediate column-basis (or `dense_forced`) re-materialization,
///   so a workspace with a structural direct-dense build should answer `Dense`
///   here and skip the operator wrapper entirely.
///
/// Workspaces refine their representation choice per intent via
/// [`ExactNewtonJointHessianWorkspace::hessian_source_preference_for_intent`];
/// the default keeps the legacy single-preference behaviour so existing
/// workspaces are unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaterializationIntent {
    /// Inner Newton / PCG solve — only applies `H · v`. Matrix-free is ideal.
    InnerSolve,
    /// REML/LAML logdet term — factorizes `H + S_λ`, needs a dense matrix.
    LogdetFactorization,
    /// Outer-Hessian / EFS evaluation — builds the joint hyper terms; today
    /// these route through the same source as the gradient path.
    OuterEvaluation,
    /// Outer-gradient / IFT term assembly.
    OuterGradient,
}
