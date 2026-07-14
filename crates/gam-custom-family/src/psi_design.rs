//! The ψ (hyperparameter) design-derivative operators: every concrete operator
//! (embedded-implicit / zero / embedded-dense / rowwise-Kronecker), the ψ
//! design / second-design actions and linear-map refs, the joint ψ operator,
//! the ψ-derivative-map resolvers, and the exact-Newton joint-ψ direct cache.
//!
//! Relocated from the pre-carve monolith `families/custom_family/psi_design.rs`
//! (#1521). The neutral ψ-derivative carriers + operator traits
//! (`CustomFamilyBlockPsiDerivative`, `CustomFamilyPsiDerivativeOperator`,
//! `MaterializablePsiDerivativeOperator`, the joint-Hessian source-preference /
//! materialization-intent enums) live in `gam-problem`; the
//! `ExactNewtonJointHessianWorkspace` workspace trait lives in `gam-model-api`.
//! Both arrive here through the `custom_family` prelude glob (`use super::*`).

use super::*;
use gam_linalg::matrix::{EmbeddedColumnBlock, dense_rowwise_kronecker};
use gam_runtime::resource::{DerivativeStorageMode, ResourcePolicy};
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Weak;

pub struct EmbeddedImplicitPsiDerivativeOperator {
    pub(crate) base: Arc<gam_terms::basis::ImplicitDesignPsiDerivative>,
    pub(crate) total_p: usize,
    pub(crate) global_range: Range<usize>,
}

impl EmbeddedImplicitPsiDerivativeOperator {
    pub fn new(
        base: Arc<gam_terms::basis::ImplicitDesignPsiDerivative>,
        global_range: Range<usize>,
        total_p: usize,
    ) -> Result<Self, String> {
        if base.p_out() != global_range.len() {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "embedded implicit psi operator width mismatch: got {}, expected {}",
                    base.p_out(),
                    global_range.len()
                ),
            }
            .into());
        }
        if global_range.end > total_p {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "embedded implicit psi operator range {}..{} exceeds total width {total_p}",
                    global_range.start, global_range.end
                ),
            }
            .into());
        }
        Ok(Self {
            base,
            total_p,
            global_range,
        })
    }

    pub fn embed_vector(&self, local: Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.total_p);
        out.slice_mut(ndarray::s![self.global_range.clone()])
            .assign(&local);
        out
    }

    pub fn local_coeffs(
        &self,
        u: &ArrayView1<'_, f64>,
        context: &str,
    ) -> Result<Array1<f64>, gam_terms::basis::BasisError> {
        if u.len() != self.total_p {
            return Err(gam_terms::basis::BasisError::Other(format!(
                "{context} expected coefficient length {}, got {}",
                self.total_p,
                u.len()
            )));
        }
        Ok(u.slice(ndarray::s![self.global_range.clone()]).to_owned())
    }
}

impl CustomFamilyPsiDerivativeOperator for EmbeddedImplicitPsiDerivativeOperator {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn n_data(&self) -> usize {
        self.base.n_data()
    }

    fn p_out(&self) -> usize {
        self.total_p
    }

    fn transpose_mul(
        &self,
        axis: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, gam_terms::basis::BasisError> {
        Ok(self.embed_vector(self.base.transpose_mul(axis, v)?))
    }

    fn forward_mul(
        &self,
        axis: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, gam_terms::basis::BasisError> {
        let local = self.local_coeffs(u, "embedded implicit psi forward_mul")?;
        self.base.forward_mul(axis, &local.view())
    }

    fn transpose_mul_second_diag(
        &self,
        axis: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, gam_terms::basis::BasisError> {
        Ok(self.embed_vector(self.base.transpose_mul_second_diag(axis, v)?))
    }

    fn transpose_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, gam_terms::basis::BasisError> {
        Ok(self.embed_vector(self.base.transpose_mul_second_cross(axis_d, axis_e, v)?))
    }

    fn forward_mul_second_diag(
        &self,
        axis: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, gam_terms::basis::BasisError> {
        let local = self.local_coeffs(u, "embedded implicit psi forward_mul_second_diag")?;
        self.base.forward_mul_second_diag(axis, &local.view())
    }

    fn forward_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, gam_terms::basis::BasisError> {
        let local = self.local_coeffs(u, "embedded implicit psi forward_mul_second_cross")?;
        self.base
            .forward_mul_second_cross(axis_d, axis_e, &local.view())
    }

    fn row_chunk_first(
        &self,
        axis: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, gam_terms::basis::BasisError> {
        let local = self.base.row_chunk_first(axis, rows)?;
        Ok(EmbeddedColumnBlock::new(&local, self.global_range.clone(), self.total_p).materialize())
    }

    fn row_vector_first_into(
        &self,
        axis: usize,
        row: usize,
        mut out: ArrayViewMut1<'_, f64>,
    ) -> Result<(), gam_terms::basis::BasisError> {
        out.fill(0.0);
        let local_slice = out.slice_mut(ndarray::s![self.global_range.clone()]);
        self.base.row_vector_first_into(axis, row, local_slice)
    }

    fn row_chunk_second_diag(
        &self,
        axis: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, gam_terms::basis::BasisError> {
        let local = self.base.row_chunk_second_diag(axis, rows)?;
        Ok(EmbeddedColumnBlock::new(&local, self.global_range.clone(), self.total_p).materialize())
    }

    fn row_chunk_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, gam_terms::basis::BasisError> {
        let local = self.base.row_chunk_second_cross(axis_d, axis_e, rows)?;
        Ok(EmbeddedColumnBlock::new(&local, self.global_range.clone(), self.total_p).materialize())
    }

    fn as_materializable(&self) -> Option<&dyn MaterializablePsiDerivativeOperator> {
        Some(self)
    }
}

impl MaterializablePsiDerivativeOperator for EmbeddedImplicitPsiDerivativeOperator {
    fn materialize_first(&self, axis: usize) -> Result<Array2<f64>, gam_terms::basis::BasisError> {
        Ok(EmbeddedColumnBlock::new(
            &self.base.materialize_first(axis)?,
            self.global_range.clone(),
            self.total_p,
        )
        .materialize())
    }
}

/// Non-allocating zero operator for `\partial X / \partial \psi` derivative
/// blocks whose ψ coordinate does not move the design matrix at all (e.g.
/// the spatial-adaptive overlay's mass / tension / stiffness / ε
/// hyperparameters, which act through the penalty stack alone).
///
/// All matvec/transpose_mul methods return zero vectors of the correct
/// length, all row-chunk methods return chunk-sized zero matrices. The
/// operator never allocates an `(n, p)` dense buffer, which saves ~1.45 GiB
/// at the large-scale spatial-adaptive overlay (n ≈ 320 000, p ≈ 101,
/// six hyperparameters).
pub struct ZeroPsiDerivativeOperator {
    pub(crate) n: usize,
    pub(crate) p: usize,
}

impl ZeroPsiDerivativeOperator {
    pub fn new(n: usize, p: usize) -> Self {
        Self { n, p }
    }
}

impl CustomFamilyPsiDerivativeOperator for ZeroPsiDerivativeOperator {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn n_data(&self) -> usize {
        self.n
    }

    fn p_out(&self) -> usize {
        self.p
    }

    fn transpose_mul(
        &self,
        _: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, gam_terms::basis::BasisError> {
        // Default implementation ignores this parameter.
        assert_eq!(v.len(), self.n, "zero psi transpose_mul length mismatch");
        Ok(Array1::<f64>::zeros(self.p))
    }

    fn forward_mul(
        &self,
        _: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, gam_terms::basis::BasisError> {
        // Default implementation ignores this parameter.
        assert_eq!(u.len(), self.p, "zero psi forward_mul length mismatch");
        Ok(Array1::<f64>::zeros(self.n))
    }

    fn transpose_mul_second_diag(
        &self,
        _: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, gam_terms::basis::BasisError> {
        // Default implementation ignores this parameter.
        assert_eq!(
            v.len(),
            self.n,
            "zero psi transpose_mul_second_diag length mismatch"
        );
        Ok(Array1::<f64>::zeros(self.p))
    }

    fn transpose_mul_second_cross(
        &self,
        _: usize,
        _: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, gam_terms::basis::BasisError> {
        // Default implementation ignores this parameter.
        // Default implementation ignores this parameter.
        assert_eq!(
            v.len(),
            self.n,
            "zero psi transpose_mul_second_cross length mismatch"
        );
        Ok(Array1::<f64>::zeros(self.p))
    }

    fn forward_mul_second_diag(
        &self,
        _: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, gam_terms::basis::BasisError> {
        // Default implementation ignores this parameter.
        assert_eq!(
            u.len(),
            self.p,
            "zero psi forward_mul_second_diag length mismatch"
        );
        Ok(Array1::<f64>::zeros(self.n))
    }

    fn forward_mul_second_cross(
        &self,
        _: usize,
        _: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, gam_terms::basis::BasisError> {
        // Default implementation ignores this parameter.
        // Default implementation ignores this parameter.
        assert_eq!(
            u.len(),
            self.p,
            "zero psi forward_mul_second_cross length mismatch"
        );
        Ok(Array1::<f64>::zeros(self.n))
    }

    fn row_chunk_first(
        &self,
        _: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, gam_terms::basis::BasisError> {
        // Default implementation ignores this parameter.
        assert!(
            rows.start <= rows.end && rows.end <= self.n,
            "zero psi row_chunk_first row range out of bounds"
        );
        Ok(Array2::<f64>::zeros((rows.end - rows.start, self.p)))
    }

    fn row_vector_first_into(
        &self,
        _: usize,
        row: usize,
        mut out: ArrayViewMut1<'_, f64>,
    ) -> Result<(), gam_terms::basis::BasisError> {
        // Default implementation ignores this parameter.
        assert!(
            row < self.n,
            "zero psi row_vector_first_into row out of bounds"
        );
        assert_eq!(
            out.len(),
            self.p,
            "zero psi row_vector_first_into output length mismatch"
        );
        out.fill(0.0);
        Ok(())
    }

    fn row_chunk_second_diag(
        &self,
        _: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, gam_terms::basis::BasisError> {
        // Default implementation ignores this parameter.
        assert!(
            rows.start <= rows.end && rows.end <= self.n,
            "zero psi row_chunk_second_diag row range out of bounds"
        );
        Ok(Array2::<f64>::zeros((rows.end - rows.start, self.p)))
    }

    fn row_chunk_second_cross(
        &self,
        _: usize,
        _: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, gam_terms::basis::BasisError> {
        // Default implementation ignores this parameter.
        // Default implementation ignores this parameter.
        assert!(
            rows.start <= rows.end && rows.end <= self.n,
            "zero psi row_chunk_second_cross row range out of bounds"
        );
        Ok(Array2::<f64>::zeros((rows.end - rows.start, self.p)))
    }
}

pub(crate) fn stack_dense_row_blocks(blocks: &[Array2<f64>]) -> Array2<f64> {
    let total_rows = blocks.iter().map(Array2::nrows).sum();
    let p = blocks.first().map(Array2::ncols).unwrap_or(0);
    let mut stacked = Array2::<f64>::zeros((total_rows, p));
    let mut row_start = 0usize;
    for block in blocks {
        assert_eq!(block.ncols(), p);
        let row_end = row_start + block.nrows();
        stacked
            .slice_mut(ndarray::s![row_start..row_end, ..])
            .assign(block);
        row_start = row_end;
    }
    stacked
}

pub(crate) struct EmbeddedDensePsiDerivativeOperator {
    pub(crate) axis: usize,
    pub(crate) total_p: usize,
    pub(crate) global_range: Range<usize>,
    pub(crate) first_local: Array2<f64>,
    pub(crate) second_diag_local: Array2<f64>,
    pub(crate) second_cross_local: HashMap<usize, Array2<f64>>,
}

impl EmbeddedDensePsiDerivativeOperator {
    pub fn new(
        axis: usize,
        total_p: usize,
        global_range: Range<usize>,
        first_local: Array2<f64>,
        second_diag_local: Array2<f64>,
        second_cross_local: HashMap<usize, Array2<f64>>,
    ) -> Result<Self, String> {
        let local_p = global_range.len();
        if first_local.ncols() != local_p {
            return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                "embedded dense psi operator first-derivative width mismatch: got {}, expected {local_p}",
                first_local.ncols()
            ) }.into());
        }
        if second_diag_local.ncols() != local_p {
            return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                "embedded dense psi operator second-diag width mismatch: got {}, expected {local_p}",
                second_diag_local.ncols()
            ) }.into());
        }
        for (cross_axis, local) in &second_cross_local {
            if local.ncols() != local_p {
                return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                    "embedded dense psi operator cross axis {cross_axis} width mismatch: got {}, expected {local_p}",
                    local.ncols()
                ) }.into());
            }
        }
        Ok(Self {
            axis,
            total_p,
            global_range,
            first_local,
            second_diag_local,
            second_cross_local,
        })
    }

    pub fn validate_axis(
        &self,
        axis: usize,
        context: &str,
    ) -> Result<(), gam_terms::basis::BasisError> {
        if axis == self.axis {
            Ok(())
        } else {
            Err(gam_terms::basis::BasisError::Other(format!(
                "{context} expected axis {}, got {axis}",
                self.axis
            )))
        }
    }

    pub fn embed_vector(&self, local: Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.total_p);
        out.slice_mut(ndarray::s![self.global_range.clone()])
            .assign(&local);
        out
    }

    pub fn local_coeffs(
        &self,
        u: &ArrayView1<'_, f64>,
        context: &str,
    ) -> Result<Array1<f64>, gam_terms::basis::BasisError> {
        if u.len() != self.total_p {
            return Err(gam_terms::basis::BasisError::Other(format!(
                "{context} expected coefficient length {}, got {}",
                self.total_p,
                u.len()
            )));
        }
        Ok(u.slice(ndarray::s![self.global_range.clone()]).to_owned())
    }

    pub fn cross_local(
        &self,
        axis_e: usize,
        context: &str,
    ) -> Result<&Array2<f64>, gam_terms::basis::BasisError> {
        self.second_cross_local.get(&axis_e).ok_or_else(|| {
            gam_terms::basis::BasisError::Other(format!(
                "{context} is missing cross-derivative data for axis {}",
                axis_e
            ))
        })
    }
}

impl CustomFamilyPsiDerivativeOperator for EmbeddedDensePsiDerivativeOperator {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn n_data(&self) -> usize {
        self.first_local.nrows()
    }

    fn p_out(&self) -> usize {
        self.total_p
    }

    fn transpose_mul(
        &self,
        axis: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, gam_terms::basis::BasisError> {
        self.validate_axis(axis, "embedded dense psi transpose_mul")?;
        if v.len() != self.n_data() {
            return Err(gam_terms::basis::BasisError::Other(format!(
                "embedded dense psi transpose_mul expected {} rows, got {}",
                self.n_data(),
                v.len()
            )));
        }
        Ok(self.embed_vector(gam_linalg::faer_ndarray::fast_atv(&self.first_local, v)))
    }

    fn forward_mul(
        &self,
        axis: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, gam_terms::basis::BasisError> {
        self.validate_axis(axis, "embedded dense psi forward_mul")?;
        Ok(self
            .first_local
            .dot(&self.local_coeffs(u, "embedded dense psi forward_mul")?))
    }

    fn transpose_mul_second_diag(
        &self,
        axis: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, gam_terms::basis::BasisError> {
        self.validate_axis(axis, "embedded dense psi transpose_mul_second_diag")?;
        if v.len() != self.second_diag_local.nrows() {
            return Err(gam_terms::basis::BasisError::Other(format!(
                "embedded dense psi transpose_mul_second_diag expected {} rows, got {}",
                self.second_diag_local.nrows(),
                v.len()
            )));
        }
        Ok(self.embed_vector(gam_linalg::faer_ndarray::fast_atv(
            &self.second_diag_local,
            v,
        )))
    }

    fn transpose_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, gam_terms::basis::BasisError> {
        self.validate_axis(axis_d, "embedded dense psi transpose_mul_second_cross")?;
        let local = self.cross_local(axis_e, "embedded dense psi transpose_mul_second_cross")?;
        if v.len() != local.nrows() {
            return Err(gam_terms::basis::BasisError::Other(format!(
                "embedded dense psi transpose_mul_second_cross expected {} rows, got {}",
                local.nrows(),
                v.len()
            )));
        }
        Ok(self.embed_vector(gam_linalg::faer_ndarray::fast_atv(local, v)))
    }

    fn forward_mul_second_diag(
        &self,
        axis: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, gam_terms::basis::BasisError> {
        self.validate_axis(axis, "embedded dense psi forward_mul_second_diag")?;
        Ok(self
            .second_diag_local
            .dot(&self.local_coeffs(u, "embedded dense psi forward_mul_second_diag")?))
    }

    fn forward_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, gam_terms::basis::BasisError> {
        self.validate_axis(axis_d, "embedded dense psi forward_mul_second_cross")?;
        Ok(self
            .cross_local(axis_e, "embedded dense psi forward_mul_second_cross")?
            .dot(&self.local_coeffs(u, "embedded dense psi forward_mul_second_cross")?))
    }

    fn row_chunk_first(
        &self,
        axis: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, gam_terms::basis::BasisError> {
        self.validate_axis(axis, "embedded dense psi row_chunk_first")?;
        let local = self.first_local.slice(ndarray::s![rows, ..]).to_owned();
        Ok(EmbeddedColumnBlock::new(&local, self.global_range.clone(), self.total_p).materialize())
    }

    fn row_vector_first_into(
        &self,
        axis: usize,
        row: usize,
        mut out: ArrayViewMut1<'_, f64>,
    ) -> Result<(), gam_terms::basis::BasisError> {
        self.validate_axis(axis, "embedded dense psi row_vector_first_into")?;
        if row >= self.first_local.nrows() {
            return Err(gam_terms::basis::BasisError::Other(format!(
                "embedded dense psi row_vector_first_into row {row} out of bounds for {}",
                self.first_local.nrows()
            )));
        }
        if out.len() != self.total_p {
            return Err(gam_terms::basis::BasisError::Other(format!(
                "embedded dense psi row_vector_first_into expected length {}, got {}",
                self.total_p,
                out.len()
            )));
        }
        out.fill(0.0);
        out.slice_mut(ndarray::s![self.global_range.clone()])
            .assign(&self.first_local.row(row));
        Ok(())
    }

    fn row_chunk_second_diag(
        &self,
        axis: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, gam_terms::basis::BasisError> {
        self.validate_axis(axis, "embedded dense psi row_chunk_second_diag")?;
        let local = self
            .second_diag_local
            .slice(ndarray::s![rows, ..])
            .to_owned();
        Ok(EmbeddedColumnBlock::new(&local, self.global_range.clone(), self.total_p).materialize())
    }

    fn row_chunk_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, gam_terms::basis::BasisError> {
        self.validate_axis(axis_d, "embedded dense psi row_chunk_second_cross")?;
        let local = self
            .cross_local(axis_e, "embedded dense psi row_chunk_second_cross")?
            .slice(ndarray::s![rows, ..])
            .to_owned();
        Ok(EmbeddedColumnBlock::new(&local, self.global_range.clone(), self.total_p).materialize())
    }

    fn as_materializable(&self) -> Option<&dyn MaterializablePsiDerivativeOperator> {
        Some(self)
    }
}

impl MaterializablePsiDerivativeOperator for EmbeddedDensePsiDerivativeOperator {
    fn materialize_first(&self, axis: usize) -> Result<Array2<f64>, gam_terms::basis::BasisError> {
        self.validate_axis(axis, "embedded dense psi materialize_first")?;
        Ok(
            EmbeddedColumnBlock::new(&self.first_local, self.global_range.clone(), self.total_p)
                .materialize(),
        )
    }
}

pub fn build_embedded_dense_psi_operator(
    first_local: &Array2<f64>,
    second_diag_local: &Array2<f64>,
    second_cross_local: Option<&Vec<(usize, Array2<f64>)>>,
    global_range: Range<usize>,
    total_p: usize,
    axis: usize,
) -> Result<Arc<dyn CustomFamilyPsiDerivativeOperator>, String> {
    let second_cross_local = second_cross_local
        .map(|rows| {
            rows.iter()
                .map(|(axis, local)| (*axis, local.clone()))
                .collect()
        })
        .unwrap_or_default();
    Ok(Arc::new(EmbeddedDensePsiDerivativeOperator::new(
        axis,
        total_p,
        global_range,
        first_local.clone(),
        second_diag_local.clone(),
        second_cross_local,
    )?))
}

pub(crate) struct RowwiseKroneckerPsiDerivativeOperator {
    pub(crate) base: Arc<dyn CustomFamilyPsiDerivativeOperator>,
    pub(crate) time_bases: Vec<Arc<Array2<f64>>>,
    pub(crate) n_per_block: usize,
    pub(crate) p_time: usize,
    pub(crate) p_out: usize,
}

impl RowwiseKroneckerPsiDerivativeOperator {
    pub fn new(
        base: Arc<dyn CustomFamilyPsiDerivativeOperator>,
        time_bases: Vec<Arc<Array2<f64>>>,
    ) -> Result<Self, String> {
        let first = time_bases.first().ok_or_else(|| {
            "rowwise kronecker psi operator needs at least one time basis".to_string()
        })?;
        let n_per_block = first.nrows();
        let p_time = first.ncols();
        for (idx, basis) in time_bases.iter().enumerate() {
            if basis.nrows() != n_per_block || basis.ncols() != p_time {
                return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                    "rowwise kronecker psi operator time basis {idx} shape mismatch: got {}x{}, expected {}x{}",
                    basis.nrows(),
                    basis.ncols(),
                    n_per_block,
                    p_time
                ) }.into());
            }
        }
        if base.n_data() != n_per_block {
            return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                "rowwise kronecker psi operator base row mismatch: got {}, expected {n_per_block}",
                base.n_data()
            ) }.into());
        }
        Ok(Self {
            p_out: base.p_out() * p_time,
            base,
            time_bases,
            n_per_block,
            p_time,
        })
    }

    pub fn split_time_columns(&self, u: &ArrayView1<'_, f64>) -> Vec<Array1<f64>> {
        let p_base = self.base.p_out();
        assert_eq!(u.len(), self.p_out);
        let mut cols = vec![Array1::<f64>::zeros(p_base); self.p_time];
        for j in 0..p_base {
            for t in 0..self.p_time {
                cols[t][j] = u[j * self.p_time + t];
            }
        }
        cols
    }

    pub fn lifted_row_chunk_with_base<F>(
        &self,
        rows: Range<usize>,
        mut base_chunk: F,
    ) -> Result<Array2<f64>, gam_terms::basis::BasisError>
    where
        F: FnMut(Range<usize>) -> Result<Array2<f64>, gam_terms::basis::BasisError>,
    {
        if rows.start > rows.end || rows.end > self.n_data() {
            return Err(gam_terms::basis::BasisError::Other(format!(
                "rowwise kronecker psi row chunk {}..{} out of bounds for {} rows",
                rows.start,
                rows.end,
                self.n_data()
            )));
        }
        if rows.is_empty() {
            return Ok(Array2::<f64>::zeros((0, self.p_out)));
        }

        let first_block = rows.start / self.n_per_block;
        let last_block = (rows.end - 1) / self.n_per_block;
        let mut blocks = Vec::with_capacity(last_block + 1 - first_block);
        for block_idx in first_block..=last_block {
            let block_global_start = block_idx * self.n_per_block;
            let local_start = rows.start.saturating_sub(block_global_start);
            let local_end = (rows.end - block_global_start).min(self.n_per_block);
            let local_rows = local_start..local_end;
            let base = base_chunk(local_rows.clone())?;
            let time = self.time_bases[block_idx]
                .slice(ndarray::s![local_rows, ..])
                .to_owned();
            blocks.push(dense_rowwise_kronecker(base.view(), time.view()));
        }
        Ok(stack_dense_row_blocks(&blocks))
    }

    /// Canonical transpose-direction lifted matvec: for each time column `t`,
    /// weight `v` by the time basis column, delegate to the base operator via
    /// `base_op`, and scatter the per-base accumulator into the lifted layout.
    pub fn lifted_transpose_mul_with_base<F>(
        &self,
        v: &ArrayView1<'_, f64>,
        mut base_op: F,
    ) -> Result<Array1<f64>, gam_terms::basis::BasisError>
    where
        F: FnMut(&ArrayView1<'_, f64>) -> Result<Array1<f64>, gam_terms::basis::BasisError>,
    {
        assert_eq!(v.len(), self.n_data());
        let p_base = self.base.p_out();
        let mut out = Array1::<f64>::zeros(self.p_out);
        for t in 0..self.p_time {
            let mut accum = Array1::<f64>::zeros(p_base);
            for (block_idx, time_basis) in self.time_bases.iter().enumerate() {
                let row_start = block_idx * self.n_per_block;
                let row_end = row_start + self.n_per_block;
                let weighted = &v.slice(ndarray::s![row_start..row_end]).to_owned()
                    * &time_basis.column(t).to_owned();
                accum += &base_op(&weighted.view())?;
            }
            for j in 0..p_base {
                out[j * self.p_time + t] = accum[j];
            }
        }
        Ok(out)
    }

    /// Canonical forward-direction lifted matvec: split `u` into per-time-column
    /// coefficient vectors, delegate each to the base operator via `base_op`, and
    /// accumulate the time-basis-weighted contributions into the block rows.
    pub fn lifted_forward_mul_with_base<F>(
        &self,
        u: &ArrayView1<'_, f64>,
        mut base_op: F,
    ) -> Result<Array1<f64>, gam_terms::basis::BasisError>
    where
        F: FnMut(&ArrayView1<'_, f64>) -> Result<Array1<f64>, gam_terms::basis::BasisError>,
    {
        let time_cols = self.split_time_columns(u);
        let mut out = Array1::<f64>::zeros(self.n_data());
        for (t, coeffs) in time_cols.iter().enumerate() {
            let base_eval = base_op(&coeffs.view())?;
            for (block_idx, time_basis) in self.time_bases.iter().enumerate() {
                let row_start = block_idx * self.n_per_block;
                let row_end = row_start + self.n_per_block;
                let contrib = &base_eval * &time_basis.column(t).to_owned();
                let mut out_block = out.slice_mut(ndarray::s![row_start..row_end]);
                out_block += &contrib;
            }
        }
        Ok(out)
    }
}

impl CustomFamilyPsiDerivativeOperator for RowwiseKroneckerPsiDerivativeOperator {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn n_data(&self) -> usize {
        self.n_per_block * self.time_bases.len()
    }

    fn p_out(&self) -> usize {
        self.p_out
    }

    fn transpose_mul(
        &self,
        axis: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, gam_terms::basis::BasisError> {
        self.lifted_transpose_mul_with_base(v, |weighted| self.base.transpose_mul(axis, weighted))
    }

    fn forward_mul(
        &self,
        axis: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, gam_terms::basis::BasisError> {
        self.lifted_forward_mul_with_base(u, |coeffs| self.base.forward_mul(axis, coeffs))
    }

    fn transpose_mul_second_diag(
        &self,
        axis: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, gam_terms::basis::BasisError> {
        self.lifted_transpose_mul_with_base(v, |weighted| {
            self.base.transpose_mul_second_diag(axis, weighted)
        })
    }

    fn transpose_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, gam_terms::basis::BasisError> {
        self.lifted_transpose_mul_with_base(v, |weighted| {
            self.base
                .transpose_mul_second_cross(axis_d, axis_e, weighted)
        })
    }

    fn forward_mul_second_diag(
        &self,
        axis: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, gam_terms::basis::BasisError> {
        self.lifted_forward_mul_with_base(u, |coeffs| {
            self.base.forward_mul_second_diag(axis, coeffs)
        })
    }

    fn forward_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, gam_terms::basis::BasisError> {
        self.lifted_forward_mul_with_base(u, |coeffs| {
            self.base.forward_mul_second_cross(axis_d, axis_e, coeffs)
        })
    }

    fn row_chunk_first(
        &self,
        axis: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, gam_terms::basis::BasisError> {
        self.lifted_row_chunk_with_base(rows, |local_rows| {
            self.base.row_chunk_first(axis, local_rows)
        })
    }

    fn row_chunk_second_diag(
        &self,
        axis: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, gam_terms::basis::BasisError> {
        self.lifted_row_chunk_with_base(rows, |local_rows| {
            self.base.row_chunk_second_diag(axis, local_rows)
        })
    }

    fn row_chunk_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, gam_terms::basis::BasisError> {
        self.lifted_row_chunk_with_base(rows, |local_rows| {
            self.base.row_chunk_second_cross(axis_d, axis_e, local_rows)
        })
    }

    fn as_materializable(&self) -> Option<&dyn MaterializablePsiDerivativeOperator> {
        Some(self)
    }
}

impl MaterializablePsiDerivativeOperator for RowwiseKroneckerPsiDerivativeOperator {
    fn materialize_first(&self, axis: usize) -> Result<Array2<f64>, gam_terms::basis::BasisError> {
        let base_mat = self.base.as_materializable().ok_or_else(|| {
            gam_terms::basis::BasisError::Other(
                "rowwise kronecker psi operator: base operator does not support materialization"
                    .to_string(),
            )
        })?;
        let base = base_mat.materialize_first(axis)?;
        let blocks: Vec<Array2<f64>> = self
            .time_bases
            .iter()
            .map(|basis| dense_rowwise_kronecker(base.view(), basis.view()))
            .collect();
        Ok(stack_dense_row_blocks(&blocks))
    }
}

pub fn build_rowwise_kronecker_psi_operator(
    base: Arc<dyn CustomFamilyPsiDerivativeOperator>,
    time_bases: Vec<Arc<Array2<f64>>>,
) -> Result<Arc<dyn CustomFamilyPsiDerivativeOperator>, String> {
    Ok(Arc::new(RowwiseKroneckerPsiDerivativeOperator::new(
        base, time_bases,
    )?))
}

#[derive(Clone)]
pub struct CustomFamilyPsiDesignAction {
    pub(crate) operator: Arc<dyn CustomFamilyPsiDerivativeOperator>,
    pub(crate) axis: usize,
    pub(crate) row_range: Range<usize>,
    pub(crate) p: usize,
}

impl CustomFamilyPsiDesignAction {
    pub fn from_first_derivative(
        deriv: &CustomFamilyBlockPsiDerivative,
        total_rows: usize,
        p: usize,
        row_range: Range<usize>,
        label: &str,
    ) -> Result<Self, String> {
        if row_range.end > total_rows {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "{label} row range {}..{} exceeds total rows {total_rows}",
                    row_range.start, row_range.end
                ),
            }
            .into());
        }
        if let Some(op) = deriv.implicit_operator.as_ref()
            && op.n_data() == total_rows
            && op.p_out() == p
        {
            return Ok(Self {
                operator: Arc::clone(op),
                axis: deriv.implicit_axis,
                row_range,
                p,
            });
        }
        Err(CustomFamilyError::UnsupportedConfiguration { reason: format!(
            "{label} is missing an implicit x_psi operator with shape {}x{}; got dense payload {}x{} instead",
            total_rows,
            p,
            deriv.x_psi.nrows(),
            deriv.x_psi.ncols(),
        ) }.into())
    }

    pub fn is_implicit(&self) -> bool {
        true
    }

    pub fn nrows(&self) -> usize {
        self.row_range.end - self.row_range.start
    }

    pub fn slice_rows(&self, row_range: Range<usize>) -> Result<Self, String> {
        if row_range.end > self.nrows() {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "psi design row range {}..{} exceeds available rows {}",
                    row_range.start,
                    row_range.end,
                    self.nrows()
                ),
            }
            .into());
        }
        Ok(Self {
            operator: Arc::clone(&self.operator),
            axis: self.axis,
            row_range: (self.row_range.start + row_range.start)
                ..(self.row_range.start + row_range.end),
            p: self.p,
        })
    }

    pub fn forward_mul(&self, u: ArrayView1<'_, f64>) -> Array1<f64> {
        assert_eq!(u.len(), self.p);
        self.operator
            .forward_mul(self.axis, &u)
            .expect("radial scalar evaluation failed during implicit psi forward_mul")
            .slice(ndarray::s![self.row_range.clone()])
            .to_owned()
    }

    pub fn transpose_mul(&self, v: ArrayView1<'_, f64>) -> Array1<f64> {
        assert_eq!(v.len(), self.row_range.end - self.row_range.start);
        if self.row_range.start == 0 && self.row_range.end == self.operator.n_data() {
            self.operator
                .transpose_mul(self.axis, &v)
                .expect("radial scalar evaluation failed during implicit psi transpose_mul")
        } else {
            let mut expanded = Array1::<f64>::zeros(self.operator.n_data());
            expanded
                .slice_mut(ndarray::s![self.row_range.clone()])
                .assign(&v);
            self.operator
                .transpose_mul(self.axis, &expanded.view())
                .expect("radial scalar evaluation failed during implicit psi transpose_mul")
        }
    }

    pub fn absolute_rows(&self, rows: Range<usize>) -> Range<usize> {
        (self.row_range.start + rows.start)..(self.row_range.start + rows.end)
    }

    pub fn row_chunk(&self, rows: Range<usize>) -> Result<Array2<f64>, String> {
        if rows.end > self.nrows() {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "psi design row range {}..{} exceeds available rows {}",
                    rows.start,
                    rows.end,
                    self.nrows()
                ),
            }
            .into());
        }
        self.operator
            .row_chunk_first(self.axis, self.absolute_rows(rows))
            .map_err(|e| e.to_string())
    }

    pub fn row_vector(&self, row: usize) -> Result<Array1<f64>, String> {
        if row >= self.nrows() {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "psi design row {row} exceeds available rows {}",
                    self.nrows()
                ),
            }
            .into());
        }
        let absolute_row = self.row_range.start + row;
        let mut out = Array1::<f64>::zeros(self.p);
        self.operator
            .row_vector_first_into(self.axis, absolute_row, out.view_mut())
            .map_err(|e| e.to_string())?;
        Ok(out)
    }
}

#[derive(Clone, Copy)]
pub(crate) enum CustomFamilyPsiSecondDesignLevel {
    Diag(usize),
    Cross(usize, usize),
}

#[derive(Clone)]
pub struct CustomFamilyPsiSecondDesignAction {
    pub(crate) operator: Arc<dyn CustomFamilyPsiDerivativeOperator>,
    pub(crate) level: CustomFamilyPsiSecondDesignLevel,
    pub(crate) row_range: Range<usize>,
    pub(crate) p: usize,
}

impl CustomFamilyPsiSecondDesignAction {
    pub fn from_second_derivative(
        deriv_i: &CustomFamilyBlockPsiDerivative,
        deriv_j: &CustomFamilyBlockPsiDerivative,
        total_rows: usize,
        p: usize,
        row_range: Range<usize>,
        label: &str,
    ) -> Result<Option<Self>, String> {
        if row_range.end > total_rows {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "{label} row range {}..{} exceeds total rows {total_rows}",
                    row_range.start, row_range.end
                ),
            }
            .into());
        }
        let Some(op) = deriv_i.implicit_operator.as_ref() else {
            return Ok(None);
        };
        if op.n_data() != total_rows || op.p_out() != p {
            return Err(CustomFamilyError::UnsupportedConfiguration {
                reason: format!(
                    "{label} is missing an implicit x_psi_psi operator with shape {}x{}",
                    total_rows, p
                ),
            }
            .into());
        }
        // The implicit second-design derivative `∂²X/∂ψ_i∂ψ_j` is nonzero when
        // the two ψ axes act on the SAME implicit operator. That is either (a) an
        // anisotropic group sharing one operator (`implicit_group_id` Some and
        // equal — the Diag/Cross axis pair below), or (b) a plain ISOTROPIC
        // single-length-scale block whose lone axis carries no group id
        // (`implicit_group_id == None`): its self-second-derivative (the ψψ
        // DIAGONAL, `implicit_axis == implicit_axis`) is still genuinely nonzero.
        // The old `is_some()` guard silently returned `None` (⇒ a ZERO second
        // design derivative) for that isotropic diagonal, dropping the entire
        // `∂²X/∂ψ² = J̈` contribution from every ψψ joint-Hessian second
        // derivative (gam#1607: the matern-length-scale outer-Hessian ψψ term).
        let same_operator = deriv_i.implicit_group_id == deriv_j.implicit_group_id
            && (deriv_i.implicit_group_id.is_some()
                || deriv_i.implicit_axis == deriv_j.implicit_axis);
        if !same_operator {
            return Ok(None);
        }
        let level = if deriv_i.implicit_axis == deriv_j.implicit_axis {
            CustomFamilyPsiSecondDesignLevel::Diag(deriv_i.implicit_axis)
        } else {
            CustomFamilyPsiSecondDesignLevel::Cross(deriv_i.implicit_axis, deriv_j.implicit_axis)
        };
        Ok(Some(Self {
            operator: Arc::clone(op),
            level,
            row_range,
            p,
        }))
    }

    pub fn nrows(&self) -> usize {
        self.row_range.end - self.row_range.start
    }

    pub fn forward_mul(&self, u: ArrayView1<'_, f64>) -> Array1<f64> {
        assert_eq!(u.len(), self.p);
        let out = match self.level {
            CustomFamilyPsiSecondDesignLevel::Diag(axis) => self
                .operator
                .forward_mul_second_diag(axis, &u)
                .expect("radial scalar evaluation failed during implicit psi second forward_mul"),
            CustomFamilyPsiSecondDesignLevel::Cross(axis_d, axis_e) => self
                .operator
                .forward_mul_second_cross(axis_d, axis_e, &u)
                .expect("radial scalar evaluation failed during implicit psi second forward_mul"),
        };
        out.slice(ndarray::s![self.row_range.clone()]).to_owned()
    }

    pub fn transpose_mul(&self, v: ArrayView1<'_, f64>) -> Array1<f64> {
        assert_eq!(v.len(), self.nrows());
        let expanded = if self.row_range.start == 0 && self.row_range.end == self.operator.n_data()
        {
            None
        } else {
            let mut expanded = Array1::<f64>::zeros(self.operator.n_data());
            expanded
                .slice_mut(ndarray::s![self.row_range.clone()])
                .assign(&v);
            Some(expanded)
        };
        let full_v = expanded.as_ref().map_or(v, |arr| arr.view());
        match self.level {
            CustomFamilyPsiSecondDesignLevel::Diag(axis) => self
                .operator
                .transpose_mul_second_diag(axis, &full_v)
                .expect("radial scalar evaluation failed during implicit psi second transpose_mul"),
            CustomFamilyPsiSecondDesignLevel::Cross(axis_d, axis_e) => self
                .operator
                .transpose_mul_second_cross(axis_d, axis_e, &full_v)
                .expect("radial scalar evaluation failed during implicit psi second transpose_mul"),
        }
    }

    pub fn absolute_rows(&self, rows: Range<usize>) -> Range<usize> {
        (self.row_range.start + rows.start)..(self.row_range.start + rows.end)
    }

    pub fn row_chunk(&self, rows: Range<usize>) -> Result<Array2<f64>, String> {
        if rows.end > self.nrows() {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "psi second-design row range {}..{} exceeds available rows {}",
                    rows.start,
                    rows.end,
                    self.nrows()
                ),
            }
            .into());
        }
        match self.level {
            CustomFamilyPsiSecondDesignLevel::Diag(axis) => self
                .operator
                .row_chunk_second_diag(axis, self.absolute_rows(rows))
                .map_err(|e| e.to_string()),
            CustomFamilyPsiSecondDesignLevel::Cross(axis_d, axis_e) => self
                .operator
                .row_chunk_second_cross(axis_d, axis_e, self.absolute_rows(rows))
                .map_err(|e| e.to_string()),
        }
    }

    pub fn row_vector(&self, row: usize) -> Result<Array1<f64>, String> {
        self.row_chunk(row..row + 1).map(|m| m.row(0).to_owned())
    }
}

#[derive(Clone, Copy)]
pub enum CustomFamilyPsiLinearMapRef<'a> {
    Dense(&'a Array2<f64>),
    First(&'a CustomFamilyPsiDesignAction),
    Second(&'a CustomFamilyPsiSecondDesignAction),
    Zero { nrows: usize, ncols: usize },
}

impl CustomFamilyPsiLinearMapRef<'_> {
    pub fn nrows(&self) -> usize {
        match self {
            Self::Dense(mat) => mat.nrows(),
            Self::First(action) => action.nrows(),
            Self::Second(action) => action.nrows(),
            Self::Zero { nrows, .. } => *nrows,
        }
    }

    pub fn ncols(&self) -> usize {
        match self {
            Self::Dense(mat) => mat.ncols(),
            Self::First(action) => action.p,
            Self::Second(action) => action.p,
            Self::Zero { ncols, .. } => *ncols,
        }
    }

    pub fn forward_mul(&self, u: ArrayView1<'_, f64>) -> Array1<f64> {
        match self {
            Self::Dense(mat) => mat.dot(&u),
            Self::First(action) => action.forward_mul(u),
            Self::Second(action) => action.forward_mul(u),
            Self::Zero { nrows, .. } => Array1::<f64>::zeros(*nrows),
        }
    }

    pub fn transpose_mul(&self, v: ArrayView1<'_, f64>) -> Array1<f64> {
        match self {
            Self::Dense(mat) => gam_linalg::faer_ndarray::fast_atv(mat, &v),
            Self::First(action) => action.transpose_mul(v),
            Self::Second(action) => action.transpose_mul(v),
            Self::Zero { ncols, .. } => Array1::<f64>::zeros(*ncols),
        }
    }

    pub fn row_vector(&self, row: usize) -> Result<Array1<f64>, String> {
        if row >= self.nrows() {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "psi linear-map row {row} out of bounds for {} rows",
                    self.nrows()
                ),
            }
            .into());
        }
        Ok(match self {
            Self::Dense(mat) => mat.row(row).to_owned(),
            Self::First(action) => action.row_vector(row)?,
            Self::Second(action) => action.row_vector(row)?,
            Self::Zero { ncols, .. } => Array1::<f64>::zeros(*ncols),
        })
    }

    pub fn row_chunk(&self, rows: Range<usize>) -> Result<Array2<f64>, String> {
        if rows.end > self.nrows() {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "psi linear-map row range {}..{} out of bounds for {} rows",
                    rows.start,
                    rows.end,
                    self.nrows()
                ),
            }
            .into());
        }
        Ok(match self {
            Self::Dense(mat) => mat.slice(ndarray::s![rows, ..]).to_owned(),
            Self::First(action) => action.row_chunk(rows)?,
            Self::Second(action) => action.row_chunk(rows)?,
            Self::Zero { ncols, .. } => Array2::<f64>::zeros((rows.end - rows.start, *ncols)),
        })
    }
}

#[derive(Clone)]
pub enum PsiDesignMap {
    Zero {
        nrows: usize,
        ncols: usize,
    },
    Dense {
        matrix: Arc<Array2<f64>>,
    },
    First {
        action: CustomFamilyPsiDesignAction,
    },
    Second {
        action: CustomFamilyPsiSecondDesignAction,
    },
}

impl PsiDesignMap {
    pub fn ncols(&self) -> usize {
        match self {
            Self::Zero { ncols, .. } => *ncols,
            Self::Dense { matrix } => matrix.ncols(),
            Self::First { action } => action.p,
            Self::Second { action } => action.p,
        }
    }

    pub fn forward_mul(&self, u: ArrayView1<'_, f64>) -> Result<Array1<f64>, String> {
        match self {
            Self::Zero { nrows, .. } => Ok(Array1::<f64>::zeros(*nrows)),
            Self::Dense { matrix } => Ok(matrix.dot(&u)),
            Self::First { action } => Ok(action.forward_mul(u)),
            Self::Second { action } => Ok(action.forward_mul(u)),
        }
    }

    pub fn row_chunk(&self, rows: Range<usize>) -> Result<Array2<f64>, String> {
        let ncols = self.ncols();
        match self {
            Self::Zero { .. } => Ok(Array2::<f64>::zeros((rows.end - rows.start, ncols))),
            Self::Dense { matrix } => Ok(matrix.slice(ndarray::s![rows, ..]).to_owned()),
            Self::First { action } => action.row_chunk(rows),
            Self::Second { action } => action.row_chunk(rows),
        }
    }

    pub fn row_vector(&self, row: usize) -> Result<Array1<f64>, String> {
        match self {
            Self::Zero { ncols, .. } => Ok(Array1::<f64>::zeros(*ncols)),
            Self::Dense { matrix } => Ok(matrix.row(row).to_owned()),
            Self::First { action } => action.row_vector(row),
            Self::Second { action } => action.row_vector(row),
        }
    }

    /// Borrow this map as a `CustomFamilyPsiLinearMapRef`, handling every
    /// variant. This is the zero-allocation replacement for the pattern
    /// `first_psi_linear_map(action.as_ref(), dense.as_ref(), n, p)`.
    pub fn as_linear_map_ref(&self) -> CustomFamilyPsiLinearMapRef<'_> {
        match self {
            Self::Zero { nrows, ncols } => CustomFamilyPsiLinearMapRef::Zero {
                nrows: *nrows,
                ncols: *ncols,
            },
            Self::Dense { matrix } => CustomFamilyPsiLinearMapRef::Dense(matrix.as_ref()),
            Self::First { action } => CustomFamilyPsiLinearMapRef::First(action),
            Self::Second { action } => CustomFamilyPsiLinearMapRef::Second(action),
        }
    }

    /// Return a reference to the first-derivative operator action if this map
    /// holds one. Useful for callers that need to pass ownership of the action
    /// into downstream operator builders.
    pub fn as_first_action(&self) -> Option<&CustomFamilyPsiDesignAction> {
        match self {
            Self::First { action } => Some(action),
            _ => None,
        }
    }

    /// Clone the first-derivative operator action if this map holds one.
    pub fn cloned_first_action(&self) -> Option<CustomFamilyPsiDesignAction> {
        self.as_first_action().cloned()
    }
}

pub(crate) fn is_zero_array(a: &Array2<f64>) -> bool {
    a.iter().all(|x| *x == 0.0)
}

pub fn weighted_crossprod_psi_maps(
    left: CustomFamilyPsiLinearMapRef<'_>,
    weights: ArrayView1<'_, f64>,
    right: CustomFamilyPsiLinearMapRef<'_>,
) -> Result<Array2<f64>, String> {
    if left.nrows() != weights.len() || right.nrows() != weights.len() {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "psi weighted crossprod row mismatch: left={}, weights={}, right={}",
                left.nrows(),
                weights.len(),
                right.nrows()
            ),
        }
        .into());
    }
    let p_left = left.ncols();
    let p_right = right.ncols();
    if p_left == 0 || p_right == 0 {
        return Ok(Array2::<f64>::zeros((p_left, p_right)));
    }
    // Zero fast path: either operand being the Zero variant makes the full product zero.
    if matches!(left, CustomFamilyPsiLinearMapRef::Zero { .. })
        || matches!(right, CustomFamilyPsiLinearMapRef::Zero { .. })
    {
        return Ok(Array2::<f64>::zeros((p_left, p_right)));
    }

    let mut out = Array2::<f64>::zeros((p_left, p_right));
    // Stream row chunks of both operands so the weighted intermediate is never
    // materialized at full n x p_right size. Chunk size is governed by the
    // resource policy's row_chunk_target_bytes.
    let policy = ResourcePolicy::default_library();
    let rows_per_chunk = gam_runtime::resource::rows_for_target_bytes(
        policy.row_chunk_target_bytes,
        p_left.saturating_add(p_right).max(1),
    );

    let n = weights.len();
    for start in (0..n).step_by(rows_per_chunk) {
        let end = (start + rows_per_chunk).min(n);
        let rows = start..end;
        let xl = left.row_chunk(rows.clone())?;
        let mut xr = right.row_chunk(rows.clone())?;
        for local in 0..xr.nrows() {
            let w = weights[start + local];
            if w != 1.0 {
                for j in 0..p_right {
                    xr[[local, j]] *= w;
                }
            }
        }
        out += &fast_atb(&xl, &xr);
    }
    Ok(out)
}

pub fn first_psi_linear_map<'a>(
    action: Option<&'a CustomFamilyPsiDesignAction>,
    dense: Option<&'a Array2<f64>>,
    nrows: usize,
    ncols: usize,
) -> CustomFamilyPsiLinearMapRef<'a> {
    if let Some(action) = action {
        CustomFamilyPsiLinearMapRef::First(action)
    } else if let Some(dense) = dense
        && dense.nrows() == nrows
        && dense.ncols() == ncols
    {
        CustomFamilyPsiLinearMapRef::Dense(dense)
    } else {
        CustomFamilyPsiLinearMapRef::Zero { nrows, ncols }
    }
}

pub fn second_psi_linear_map<'a>(
    action: Option<&'a CustomFamilyPsiSecondDesignAction>,
    dense: Option<&'a Array2<f64>>,
    nrows: usize,
    ncols: usize,
) -> CustomFamilyPsiLinearMapRef<'a> {
    if let Some(action) = action {
        CustomFamilyPsiLinearMapRef::Second(action)
    } else if let Some(dense) = dense
        && dense.nrows() == nrows
        && dense.ncols() == ncols
    {
        CustomFamilyPsiLinearMapRef::Dense(dense)
    } else {
        CustomFamilyPsiLinearMapRef::Zero { nrows, ncols }
    }
}

pub struct CustomFamilyJointDesignChannel {
    pub(crate) range: Range<usize>,
    pub(crate) design: DesignMatrix,
    pub(crate) psi_derivative: Option<CustomFamilyPsiDesignAction>,
}

impl CustomFamilyJointDesignChannel {
    pub fn new<D>(
        range: Range<usize>,
        design: D,
        psi_derivative: Option<CustomFamilyPsiDesignAction>,
    ) -> Self
    where
        D: Into<DesignMatrix>,
    {
        Self {
            range,
            design: design.into(),
            psi_derivative,
        }
    }

    pub fn coefficients(&self, full: &Array1<f64>) -> Array1<f64> {
        full.slice(ndarray::s![self.range.clone()]).to_owned()
    }

    pub fn apply(&self, full: &Array1<f64>) -> Array1<f64> {
        let coeffs = self.coefficients(full);
        self.design.matrixvectormultiply(&coeffs)
    }

    pub fn apply_transpose(&self, values: &Array1<f64>) -> Array1<f64> {
        self.design.transpose_vector_multiply(values)
    }
}

pub struct CustomFamilyJointDesignPairContribution {
    pub(crate) left_channel: usize,
    pub(crate) right_channel: usize,
    pub(crate) weights: Array1<f64>,
    pub(crate) drift_weights: Array1<f64>,
}

impl CustomFamilyJointDesignPairContribution {
    pub fn new(
        left_channel: usize,
        right_channel: usize,
        weights: Array1<f64>,
        drift_weights: Array1<f64>,
    ) -> Self {
        Self {
            left_channel,
            right_channel,
            weights,
            drift_weights,
        }
    }
}

pub struct CustomFamilyJointPsiOperator {
    pub(crate) total_dim: usize,
    pub(crate) channels: Vec<CustomFamilyJointDesignChannel>,
    pub(crate) pair_contributions: Vec<CustomFamilyJointDesignPairContribution>,
    /// Optional dense correction for small cross-blocks (e.g. h/w parameters)
    /// that don't warrant their own weighted-Gram channel.
    pub(crate) dense_correction: Option<Array2<f64>>,
}

impl CustomFamilyJointPsiOperator {
    pub fn new(
        total_dim: usize,
        channels: Vec<CustomFamilyJointDesignChannel>,
        pair_contributions: Vec<CustomFamilyJointDesignPairContribution>,
    ) -> Self {
        Self {
            total_dim,
            channels,
            pair_contributions,
            dense_correction: None,
        }
    }
}

impl HyperOperator for CustomFamilyJointPsiOperator {
    fn dim(&self) -> usize {
        self.total_dim
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        assert_eq!(v.len(), self.total_dim);
        let base_vals: Vec<Array1<f64>> = self
            .channels
            .iter()
            .map(|channel| channel.apply(v))
            .collect();
        let deriv_vals: Vec<Option<Array1<f64>>> = self
            .channels
            .iter()
            .map(|channel| {
                channel
                    .psi_derivative
                    .as_ref()
                    .map(|deriv| deriv.forward_mul(v.slice(ndarray::s![channel.range.clone()])))
            })
            .collect();

        let mut out = if let Some(ref corr) = self.dense_correction {
            corr.dot(v)
        } else {
            Array1::<f64>::zeros(self.total_dim)
        };
        for pair in &self.pair_contributions {
            let left = &self.channels[pair.left_channel];
            let right_base = &base_vals[pair.right_channel];
            let weighted_drift = &pair.drift_weights * right_base;
            let mut contrib = left.apply_transpose(&weighted_drift);

            if let Some(left_deriv) = left.psi_derivative.as_ref() {
                let weighted_right = &pair.weights * right_base;
                contrib += &left_deriv.transpose_mul(weighted_right.view());
            }

            if let Some(right_deriv) = deriv_vals[pair.right_channel].as_ref() {
                let weighted_right = &pair.weights * right_deriv;
                contrib += &left.apply_transpose(&weighted_right);
            }

            let mut out_slice = out.slice_mut(ndarray::s![left.range.clone()]);
            out_slice += &contrib;
        }

        out
    }

    fn bilinear(&self, v: &Array1<f64>, u: &Array1<f64>) -> f64 {
        assert_eq!(v.len(), self.total_dim);
        assert_eq!(u.len(), self.total_dim);
        let base_v: Vec<Array1<f64>> = self
            .channels
            .iter()
            .map(|channel| channel.apply(v))
            .collect();
        let base_u: Vec<Array1<f64>> = self
            .channels
            .iter()
            .map(|channel| channel.apply(u))
            .collect();
        let deriv_v: Vec<Option<Array1<f64>>> = self
            .channels
            .iter()
            .map(|channel| {
                channel
                    .psi_derivative
                    .as_ref()
                    .map(|deriv| deriv.forward_mul(v.slice(ndarray::s![channel.range.clone()])))
            })
            .collect();
        let deriv_u: Vec<Option<Array1<f64>>> = self
            .channels
            .iter()
            .map(|channel| {
                channel
                    .psi_derivative
                    .as_ref()
                    .map(|deriv| deriv.forward_mul(u.slice(ndarray::s![channel.range.clone()])))
            })
            .collect();

        let mut total = if let Some(ref corr) = self.dense_correction {
            v.dot(&corr.dot(u))
        } else {
            0.0
        };
        for pair in &self.pair_contributions {
            let left_base_u = &base_u[pair.left_channel];
            let right_base_v = &base_v[pair.right_channel];
            total += left_base_u.dot(&(&pair.drift_weights * right_base_v));

            if let Some(left_deriv_u) = deriv_u[pair.left_channel].as_ref() {
                total += left_deriv_u.dot(&(&pair.weights * right_base_v));
            }
            if let Some(right_deriv_v) = deriv_v[pair.right_channel].as_ref() {
                total += left_base_u.dot(&(&pair.weights * right_deriv_v));
            }
        }

        total
    }

    fn to_dense(&self) -> Array2<f64> {
        let mut out = self
            .dense_correction
            .clone()
            .unwrap_or_else(|| Array2::<f64>::zeros((self.total_dim, self.total_dim)));
        let mut basis = Array1::<f64>::zeros(self.total_dim);
        for j in 0..self.total_dim {
            basis[j] = 1.0;
            // Use mul_vec without the dense_correction part (already in `out`).
            let base_vals: Vec<Array1<f64>> = self
                .channels
                .iter()
                .map(|channel| channel.apply(&basis))
                .collect();
            let deriv_vals: Vec<Option<Array1<f64>>> = self
                .channels
                .iter()
                .map(|channel| {
                    channel.psi_derivative.as_ref().map(|deriv| {
                        deriv.forward_mul(basis.slice(ndarray::s![channel.range.clone()]))
                    })
                })
                .collect();
            let mut col = Array1::<f64>::zeros(self.total_dim);
            for pair in &self.pair_contributions {
                let left = &self.channels[pair.left_channel];
                let right_base = &base_vals[pair.right_channel];
                let weighted_drift = &pair.drift_weights * right_base;
                let mut contrib = left.apply_transpose(&weighted_drift);
                if let Some(left_deriv) = left.psi_derivative.as_ref() {
                    let weighted_right = &pair.weights * right_base;
                    contrib += &left_deriv.transpose_mul(weighted_right.view());
                }
                if let Some(right_deriv) = deriv_vals[pair.right_channel].as_ref() {
                    let weighted_right = &pair.weights * right_deriv;
                    contrib += &left.apply_transpose(&weighted_right);
                }
                col.slice_mut(ndarray::s![left.range.clone()])
                    .scaled_add(1.0, &contrib);
            }
            out.column_mut(j).scaled_add(1.0, &col);
            basis[j] = 0.0;
        }
        out
    }

    fn is_implicit(&self) -> bool {
        self.dense_correction.is_none()
            && self.channels.iter().any(|channel| {
                channel
                    .psi_derivative
                    .as_ref()
                    .is_some_and(|d| d.is_implicit())
            })
    }
}

pub(crate) fn shared_dense_design_cache()
-> &'static Mutex<HashMap<(usize, usize, usize), Weak<Array2<f64>>>> {
    static CACHE: OnceLock<Mutex<HashMap<(usize, usize, usize), Weak<Array2<f64>>>>> =
        OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn shared_dense_arc(x: &Array2<f64>) -> Arc<Array2<f64>> {
    let key = (x.as_ptr() as usize, x.nrows(), x.ncols());
    let cache = shared_dense_design_cache();
    if let Ok(mut guard) = cache.lock() {
        if let Some(shared) = guard.get(&key).and_then(Weak::upgrade) {
            return shared;
        }
        guard.retain(|_, shared| shared.strong_count() > 0);
        let shared = Arc::new(x.clone());
        guard.insert(key, Arc::downgrade(&shared));
        shared
    } else {
        Arc::new(x.clone())
    }
}

pub fn resolve_custom_family_x_psi_map(
    deriv: &CustomFamilyBlockPsiDerivative,
    n: usize,
    p: usize,
    row_range: Range<usize>,
    label: &str,
    policy: &ResourcePolicy,
) -> Result<PsiDesignMap, String> {
    if row_range.end > n {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "{label}: row range {}..{} exceeds total rows {n}",
                row_range.start, row_range.end
            ),
        }
        .into());
    }

    // An explicitly supplied operator is authoritative. A width mismatch is a
    // coordinate-frame error, not permission to reinterpret the derivative as
    // the empty/zero sentinel below.
    if let Some(op) = deriv.implicit_operator.as_ref() {
        if op.n_data() != n || op.p_out() != p {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "{label}: implicit x_psi operator shape ({}, {}) does not match ({n}, {p})",
                    op.n_data(),
                    op.p_out(),
                ),
            }
            .into());
        }
        return Ok(PsiDesignMap::First {
            action: CustomFamilyPsiDesignAction::from_first_derivative(
                deriv, n, p, row_range, label,
            )?,
        });
    }

    // Dense fallback guarded by policy.
    if deriv.x_psi.nrows() == n && deriv.x_psi.ncols() == p {
        match policy.derivative_storage_mode {
            DerivativeStorageMode::AnalyticOperatorRequired => {
                if is_zero_array(&deriv.x_psi) {
                    return Ok(PsiDesignMap::Zero {
                        nrows: row_range.end - row_range.start,
                        ncols: p,
                    });
                }
                return Err(CustomFamilyError::UnsupportedConfiguration {
                    reason: format!(
                        "{label}: dense x_psi fallback disabled by AnalyticOperatorRequired"
                    ),
                }
                .into());
            }
            DerivativeStorageMode::MaterializeIfSmall | DerivativeStorageMode::DiagnosticsOnly => {
                let matrix = if row_range.start == 0 && row_range.end == n {
                    Arc::new(deriv.x_psi.clone())
                } else {
                    Arc::new(
                        deriv
                            .x_psi
                            .slice(ndarray::s![row_range.clone(), ..])
                            .to_owned(),
                    )
                };
                return Ok(PsiDesignMap::Dense { matrix });
            }
        }
    }

    // Empty / zero sentinel.
    if deriv.x_psi.nrows() == 0 || deriv.x_psi.ncols() == 0 {
        return Ok(PsiDesignMap::Zero {
            nrows: row_range.end - row_range.start,
            ncols: p,
        });
    }

    Err(CustomFamilyError::DimensionMismatch {
        reason: format!(
            "{label}: x_psi shape {:?} does not match ({n}, {p})",
            deriv.x_psi.dim()
        ),
    }
    .into())
}

pub fn resolve_custom_family_x_psi_psi_map(
    deriv_i: &CustomFamilyBlockPsiDerivative,
    deriv_j: &CustomFamilyBlockPsiDerivative,
    local_j: usize,
    n: usize,
    p: usize,
    row_range: Range<usize>,
    label: &str,
    policy: &ResourcePolicy,
) -> Result<PsiDesignMap, String> {
    if row_range.end > n {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "{label}: row range {}..{} exceeds total rows {n}",
                row_range.start, row_range.end
            ),
        }
        .into());
    }

    // Prefer operator action when dimensions match.
    if let Some(op) = deriv_i.implicit_operator.as_ref()
        && op.n_data() == n
        && op.p_out() == p
    {
        // Same-operator predicate — see `from_second_derivative` for the full
        // rationale. Crucially, an ISOTROPIC single-axis block (`group_id ==
        // None`) still has a nonzero ψψ DIAGONAL self-second-derivative, so the
        // old `is_some()` guard must not short-circuit it to zero (gam#1607).
        let same_operator = deriv_i.implicit_group_id == deriv_j.implicit_group_id
            && (deriv_i.implicit_group_id.is_some()
                || deriv_i.implicit_axis == deriv_j.implicit_axis);
        if !same_operator {
            return Ok(PsiDesignMap::Zero {
                nrows: row_range.end - row_range.start,
                ncols: p,
            });
        }
        match CustomFamilyPsiSecondDesignAction::from_second_derivative(
            deriv_i,
            deriv_j,
            n,
            p,
            row_range.clone(),
            label,
        )? {
            Some(action) => {
                return Ok(PsiDesignMap::Second { action });
            }
            None => {
                return Ok(PsiDesignMap::Zero {
                    nrows: row_range.end - row_range.start,
                    ncols: p,
                });
            }
        }
    }

    // Dense fallback guarded by policy, reading from the per-second-derivative
    // slot `x_psi_psi[local_j]` if provided.
    if let Some(x_psi_psi) = deriv_i.x_psi_psi.as_ref()
        && let Some(x_ab) = x_psi_psi.get(local_j)
    {
        if x_ab.nrows() == n && x_ab.ncols() == p {
            match policy.derivative_storage_mode {
                DerivativeStorageMode::AnalyticOperatorRequired => {
                    if is_zero_array(x_ab) {
                        return Ok(PsiDesignMap::Zero {
                            nrows: row_range.end - row_range.start,
                            ncols: p,
                        });
                    }
                    return Err(CustomFamilyError::UnsupportedConfiguration {
                        reason: format!(
                            "{label}: dense x_psi_psi fallback disabled by AnalyticOperatorRequired"
                        ),
                    }
                    .into());
                }
                DerivativeStorageMode::MaterializeIfSmall
                | DerivativeStorageMode::DiagnosticsOnly => {
                    let matrix = if row_range.start == 0 && row_range.end == n {
                        Arc::new(x_ab.clone())
                    } else {
                        Arc::new(x_ab.slice(ndarray::s![row_range.clone(), ..]).to_owned())
                    };
                    return Ok(PsiDesignMap::Dense { matrix });
                }
            }
        }
        if x_ab.is_empty() {
            return Ok(PsiDesignMap::Zero {
                nrows: row_range.end - row_range.start,
                ncols: p,
            });
        }
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "{label}: x_psi_psi shape {:?} does not match ({n}, {p})",
                x_ab.dim()
            ),
        }
        .into());
    }

    // No operator, no dense slot: treat as zero.
    Ok(PsiDesignMap::Zero {
        nrows: row_range.end - row_range.start,
        ncols: p,
    })
}

pub struct ExactNewtonJointPsiDirectCache<T> {
    pub(crate) entries: Vec<Mutex<Option<Option<Arc<T>>>>>,
    pub(crate) lru: Mutex<std::collections::VecDeque<usize>>,
    pub(crate) limit: usize,
}

impl<T> ExactNewtonJointPsiDirectCache<T> {
    pub fn new(len: usize) -> Self {
        Self {
            entries: (0..len).map(|_| Mutex::new(None)).collect(),
            lru: Mutex::new(std::collections::VecDeque::new()),
            limit: len,
        }
    }

    pub fn touch_lru(&self, index: usize) -> Result<(), String> {
        let mut lru = self
            .lru
            .lock()
            .map_err(|_| "joint psi direct cache lru poisoned".to_string())?;
        lru.retain(|&existing| existing != index);
        lru.push_back(index);
        while lru.len() > self.limit {
            let Some(evict_index) = lru.pop_front() else {
                break;
            };
            if evict_index == index {
                continue;
            }
            if let Some(entry) = self.entries.get(evict_index) {
                let mut guard = entry
                    .lock()
                    .map_err(|_| "joint psi direct cache poisoned".to_string())?;
                *guard = None;
            }
        }
        Ok(())
    }

    pub fn get_or_try_init<F>(&self, index: usize, init: F) -> Result<Option<Arc<T>>, String>
    where
        F: FnOnce() -> Result<Option<T>, String>,
    {
        let Some(entry) = self.entries.get(index) else {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "psi cache index {index} out of bounds for size {}",
                    self.entries.len()
                ),
            }
            .into());
        };
        {
            let guard = entry
                .lock()
                .map_err(|_| "joint psi direct cache poisoned".to_string())?;
            if let Some(cached) = guard.as_ref() {
                let cached = cached.clone();
                // release-early-on-purpose: update LRU after releasing the entry mutex.
                drop(guard);
                self.touch_lru(index)?;
                return Ok(cached);
            }
        }

        let computed = init()?.map(Arc::new);
        let mut guard = entry
            .lock()
            .map_err(|_| "joint psi direct cache poisoned".to_string())?;
        let cached = guard.get_or_insert_with(|| computed.clone());
        let out = cached.clone();
        // release-early-on-purpose: update LRU after releasing the entry mutex.
        drop(guard);
        self.touch_lru(index)?;
        Ok(out)
    }
}
