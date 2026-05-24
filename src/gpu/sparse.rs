use super::memory::DeviceCsrMatrix;
use super::profile::{OperationKind, record_cpu_fallback};

#[derive(Clone, Debug)]
pub enum SparseGpuMode {
    DenseOuterProduct,
    CusparseSpGemm,
    MatrixFreePcg,
}

pub fn try_sparse_xtwx(
    csr: &DeviceCsrMatrix,
    weights_len: usize,
    mode: SparseGpuMode,
) -> Option<super::memory::DeviceMatrix> {
    let _ = (weights_len, mode);
    record_cpu_fallback(
        "gpu.sparse.xtwx",
        OperationKind::SparseXtWx,
        csr.rows,
        csr.cols,
        0,
        csr.values.len(),
    );
    None
}
