use super::policy::{Operation, OperationDecision};
use super::runtime::GpuRuntime;

#[derive(Clone, Debug)]
pub struct DeviceCsrMatrix<T> {
    pub row_offsets: Vec<i32>,
    pub col_indices: Vec<i32>,
    pub values: Vec<T>,
    pub rows: usize,
    pub cols: usize,
}

impl<T> DeviceCsrMatrix<T> {
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
}

#[must_use]
pub fn should_dispatch_sparse_xt_diag_x(nnz: usize, pattern_reuse: bool) -> bool {
    GpuRuntime::global().selected_context().is_some_and(|ctx| {
        ctx.target_for(Operation::SparseXtDiagX { nnz, pattern_reuse }) == OperationDecision::Gpu
    })
}
