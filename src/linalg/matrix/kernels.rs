//! Spatial-kernel design operators, split out of `matrix/mod.rs` by
//! concern (#1145). Re-exported from `matrix` so the public paths
//! `crate::matrix::{SpatialKernelEvaluator, ChunkedKernelDesignOperator}`
//! stay stable.

use super::*;
use crate::solver::gauge::Gauge;

pub trait SpatialKernelEvaluator: Send + Sync + 'static {
    fn eval(&self, x: &[f64], c: &[f64]) -> f64;
}

impl<F> SpatialKernelEvaluator for F
where
    F: Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static,
{
    fn eval(&self, x: &[f64], c: &[f64]) -> f64 {
        self(x, c)
    }
}

impl<F> SpatialKernelEvaluator for Arc<F>
where
    F: Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static + ?Sized,
{
    fn eval(&self, x: &[f64], c: &[f64]) -> f64 {
        self.as_ref()(x, c)
    }
}

/// Chunked kernel design operator for spatial smooths (TPS, Matérn, Duchon).
///
/// Instead of storing a dense n × k matrix, evaluates K(data[i], center[j])
/// on-the-fly in row chunks. Memory usage is O(chunk_size × k) instead of O(n × k).
///
/// The optional `poly_basis` appends polynomial columns after the kernel columns
/// (e.g., linear polynomial for TPS identifiability).
///
/// The optional `kernel_gauge` restricts the kernel coefficient block through
/// a Gauge section, so the effective design is [K_reduced | poly] instead of
/// [K | poly].
pub struct ChunkedKernelDesignOperator<K: SpatialKernelEvaluator> {
    /// Observation data points (n × d).
    data: Arc<Array2<f64>>,
    /// Radial basis centers (k × d).
    centers: Arc<Array2<f64>>,
    /// Kernel evaluator: (data_row, center_row) -> f64.
    kernel: K,
    /// Optional coefficient-space gauge applied to kernel columns.
    kernel_gauge: Option<Arc<Gauge>>,
    /// Optional polynomial basis columns (n × m) appended after kernel columns.
    poly_basis: Option<Arc<Array2<f64>>>,
    n: usize,
    total_cols: usize,
    /// One-time-materialized [K_eff | poly] block, populated on first hot use.
    /// Only allocated when the dense block fits within the materialization budget;
    /// reused across all PIRLS iterations and outer-seed evaluations.
    materialized: OnceLock<Option<Arc<Array2<f64>>>>,
}

impl<K: SpatialKernelEvaluator> ChunkedKernelDesignOperator<K> {
    pub fn new(
        data: Arc<Array2<f64>>,
        centers: Arc<Array2<f64>>,
        kernel: K,
        kernel_gauge: Option<Arc<Gauge>>,
        poly_basis: Option<Arc<Array2<f64>>>,
    ) -> Result<Self, String> {
        let n = data.nrows();
        let k = centers.nrows();
        if data.ncols() != centers.ncols() {
            return Err(format!(
                "ChunkedKernelDesignOperator: data dim {} != centers dim {}",
                data.ncols(),
                centers.ncols(),
            ));
        }
        if let Some(gauge) = kernel_gauge.as_ref()
            && gauge.raw_total() != k
        {
            return Err(format!(
                "ChunkedKernelDesignOperator: kernel gauge raw width {} != centers rows {}",
                gauge.raw_total(),
                k,
            ));
        }
        if let Some(poly) = poly_basis.as_ref()
            && poly.nrows() != n
        {
            return Err(format!(
                "ChunkedKernelDesignOperator: poly_basis rows {} != data rows {}",
                poly.nrows(),
                n,
            ));
        }
        let k_eff = kernel_gauge.as_ref().map_or(k, |g| g.reduced_total());
        let poly_cols = poly_basis.as_ref().map_or(0, |p| p.ncols());
        Ok(Self {
            data: Arc::new(data.as_standard_layout().to_owned()),
            centers: Arc::new(centers.as_standard_layout().to_owned()),
            kernel,
            kernel_gauge,
            poly_basis,
            n,
            total_cols: k_eff + poly_cols,
            materialized: OnceLock::new(),
        })
    }

    /// Maximum bytes we are willing to spend on the one-shot materialized
    /// [K_eff | poly] block. The lazy operator was originally selected because
    /// the *initial* fit-time allocation budget was tight, but once PIRLS is
    /// running we will pay the kernel-evaluation cost on every iteration unless
    /// we cache the result. 1 GB is generous enough to cover large-scale
    /// dense Duchon / TPS (n = 320k, p = 117 → ~300 MiB) while still rejecting
    /// pathological dense kernels.
    const MATERIALIZE_MAX_BYTES: usize = 1024 * 1024 * 1024;

    /// Get-or-build the materialized [K_eff | poly] dense block.  Returns
    /// `None` when the block would exceed `MATERIALIZE_MAX_BYTES`; in that
    /// case callers must fall back to row-chunked evaluation.
    ///
    /// Implementation note: the build path runs `par_chunks_mut` inside
    /// `kernel_chunk`, so we deliberately compute *outside* the
    /// `OnceLock`. Using `get_or_init` would hold the lock across that
    /// nested rayon work, and any sibling rayon workers that reach
    /// `get_or_init` while the build is in flight would park on the
    /// `OnceLock`'s OS mutex — starving the build's nested `par_iter`
    /// and deadlocking the whole pool (every worker in
    /// `pthread_mutex_wait`, init thread in `pthread_cond_wait`,
    /// 0% CPU). See `feedback_oncelock_rayon_deadlock`. Computing
    /// without the lock costs at most one redundant build per racing
    /// caller — `OnceLock::set` discards losers; `get` after `set`
    /// always observes the winning value regardless of who won.
    fn materialized_combined(&self) -> Option<&Array2<f64>> {
        if let Some(slot) = self.materialized.get() {
            return slot.as_ref().map(|a| a.as_ref());
        }
        let bytes = self
            .n
            .checked_mul(self.total_cols)
            .and_then(|cells| cells.checked_mul(std::mem::size_of::<f64>()));
        let computed = match bytes {
            Some(b) if b <= Self::MATERIALIZE_MAX_BYTES => {
                Some(Arc::new(self.build_row_chunk_combined(0..self.n)))
            }
            _ => None,
        };
        if self.materialized.set(computed).is_err() {
            return self
                .materialized
                .get()
                .and_then(|opt| opt.as_ref().map(|a| a.as_ref()));
        }
        self.materialized
            .get()
            .and_then(|opt| opt.as_ref().map(|a| a.as_ref()))
    }

    /// Evaluate kernel block for a range of rows, then restrict it through the
    /// coefficient Gauge when present.
    ///
    /// This is not a matrix Kronecker product. The center rows are coordinate
    /// arguments to `kernel.eval(data_row, center_row)`; each output entry is a
    /// scalar kernel value before the optional column projection.
    fn kernel_chunk(&self, rows: Range<usize>) -> Array2<f64> {
        let chunk_n = rows.end - rows.start;
        let k_raw = self.centers.nrows();
        let dim = self.data.ncols();
        let data = self
            .data
            .as_slice()
            .expect("ChunkedKernelDesignOperator stores standard-layout data");
        let centers = self
            .centers
            .as_slice()
            .expect("ChunkedKernelDesignOperator stores standard-layout centers");
        let kernel = &self.kernel;
        let mut values = vec![0.0_f64; chunk_n * k_raw];
        values
            .par_chunks_mut(k_raw)
            .enumerate()
            .for_each(|(local, out_row)| {
                let global = rows.start + local;
                let x_start = global * dim;
                let x = &data[x_start..x_start + dim];
                for j in 0..k_raw {
                    let c_start = j * dim;
                    out_row[j] = kernel.eval(x, &centers[c_start..c_start + dim]);
                }
            });
        let kernel_block = Array2::from_shape_vec((chunk_n, k_raw), values)
            .expect("kernel chunk shape should match generated values");
        if let Some(gauge) = self.kernel_gauge.as_ref() {
            gauge.restrict_design(&kernel_block)
        } else {
            kernel_block
        }
    }
}

impl<K: SpatialKernelEvaluator> LinearOperator for ChunkedKernelDesignOperator<K> {
    fn nrows(&self) -> usize {
        self.n
    }
    fn ncols(&self) -> usize {
        self.total_cols
    }
    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        if let Some(combined) = self.materialized_combined() {
            return fast_av(combined, vector);
        }
        let k_eff = self
            .kernel_gauge
            .as_ref()
            .map_or(self.centers.nrows(), |g| g.reduced_total());
        let v_kernel = vector.slice(s![..k_eff]);
        let mut result = Array1::<f64>::zeros(self.n);
        // Process in chunks to limit memory.
        for start in (0..self.n).step_by(KERNEL_OPERATOR_ROW_CHUNK_SIZE) {
            let end = (start + KERNEL_OPERATOR_ROW_CHUNK_SIZE).min(self.n);
            let chunk = self.kernel_chunk(start..end);
            let partial = fast_av(&chunk, &v_kernel);
            result.slice_mut(s![start..end]).assign(&partial);
        }
        if let Some(poly) = self.poly_basis.as_ref() {
            let v_poly = vector.slice(s![k_eff..]);
            let poly_part = fast_av(poly, &v_poly);
            result += &poly_part;
        }
        result
    }
    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        if let Some(combined) = self.materialized_combined() {
            return fast_atv(combined, vector);
        }
        let k_eff = self
            .kernel_gauge
            .as_ref()
            .map_or(self.centers.nrows(), |g| g.reduced_total());
        let mut result = Array1::<f64>::zeros(self.total_cols);
        // Kernel part: chunked accumulation of K^T v.
        for start in (0..self.n).step_by(KERNEL_OPERATOR_ROW_CHUNK_SIZE) {
            let end = (start + KERNEL_OPERATOR_ROW_CHUNK_SIZE).min(self.n);
            let chunk = self.kernel_chunk(start..end);
            let v_slice = vector.slice(s![start..end]);
            let partial = fast_atv(&chunk, &v_slice);
            result.slice_mut(s![..k_eff]).scaled_add(1.0, &partial);
        }
        // Poly part.
        if let Some(poly) = self.poly_basis.as_ref() {
            let poly_part = fast_atv(poly, vector);
            result.slice_mut(s![k_eff..]).assign(&poly_part);
        }
        result
    }
    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        let p = self.total_cols;
        // [STAGE] kernel-xtwx: prefer the one-shot materialized [K_eff | poly]
        // block + faer streaming GEMM.  This is the BLAS-3 path that beats
        // the per-iteration kernel rebuild by an order of magnitude on dense
        // Duchon / TPS designs.
        if let Some(combined) = self.materialized_combined() {
            let mut xtwx = Array2::<f64>::zeros((p, p));
            stream_weighted_crossprod_into(
                combined,
                weights,
                &mut xtwx,
                CrossprodStructure::Full,
                CrossprodAccum::Replace,
                effective_global_parallelism(),
            );
            return Ok(xtwx);
        }
        // Fallback: design too large to materialize.  Run row chunks in
        // parallel, each thread folding into its own p×p accumulator and
        // performing one BLAS-3 GEMM (Xc^T·(W·Xc)) per chunk.
        let n = self.n;
        if n == 0 || p == 0 {
            return Ok(Array2::<f64>::zeros((p, p)));
        }
        let chunk_starts: Vec<usize> = (0..n).step_by(KERNEL_OPERATOR_ROW_CHUNK_SIZE).collect();
        let xtwx = chunk_starts
            .into_par_iter()
            .fold(
                || Array2::<f64>::zeros((p, p)),
                |mut acc, start| {
                    let end = (start + KERNEL_OPERATOR_ROW_CHUNK_SIZE).min(n);
                    let chunk = self.row_chunk_combined(start..end);
                    let mut wchunk = chunk.clone();
                    for local in 0..(end - start) {
                        let wi = weights[start + local];
                        wchunk.row_mut(local).mapv_inplace(|v| v * wi);
                    }
                    let chunk_view = FaerArrayView::new(&chunk);
                    let wchunk_view = FaerArrayView::new(&wchunk);
                    let mut acc_view = array2_to_matmut(&mut acc);
                    matmul(
                        acc_view.as_mut(),
                        Accum::Add,
                        chunk_view.as_ref().transpose(),
                        wchunk_view.as_ref(),
                        1.0,
                        Par::Seq,
                    );
                    acc
                },
            )
            .reduce(
                || Array2::<f64>::zeros((p, p)),
                |mut a, b| {
                    a += &b;
                    a
                },
            );
        Ok(xtwx)
    }
}

impl<K: SpatialKernelEvaluator> ChunkedKernelDesignOperator<K> {
    /// Combined row chunk: [kernel_chunk | poly_chunk]. Reuses the cached
    /// materialization when available to avoid recomputing kernel evaluations.
    pub(crate) fn row_chunk_combined(&self, rows: Range<usize>) -> Array2<f64> {
        if let Some(combined) = self.materialized_combined() {
            return combined.slice(s![rows, ..]).to_owned();
        }
        self.build_row_chunk_combined(rows)
    }

    /// Build a combined row chunk from scratch, bypassing the cache. Used by
    /// `row_chunk_combined` on a cache miss and by `materialized_combined`'s
    /// initializer (which must avoid re-entering the OnceLock).
    fn build_row_chunk_combined(&self, rows: Range<usize>) -> Array2<f64> {
        let chunk_n = rows.end - rows.start;
        let k_eff = self
            .kernel_gauge
            .as_ref()
            .map_or(self.centers.nrows(), |g| g.reduced_total());
        let kernel = self.kernel_chunk(rows.clone());
        let poly_cols = self.poly_basis.as_ref().map_or(0, |p| p.ncols());
        let mut combined = Array2::<f64>::zeros((chunk_n, k_eff + poly_cols));
        combined.slice_mut(s![.., ..k_eff]).assign(&kernel);
        if let Some(poly) = self.poly_basis.as_ref() {
            combined
                .slice_mut(s![.., k_eff..])
                .assign(&poly.slice(s![rows, ..]));
        }
        combined
    }
}

impl<K: SpatialKernelEvaluator> DenseDesignOperator for ChunkedKernelDesignOperator<K> {
    /// Expose the cached [K_eff | poly] materialization so cross-block paths
    /// can use the Dense × Dense BLAS-3 fast path instead of falling back to
    /// chunked scalar accumulation.
    fn as_dense_ref(&self) -> Option<&Array2<f64>> {
        self.materialized_combined()
    }

    fn row_chunk_into(
        &self,
        rows: Range<usize>,
        mut out: ArrayViewMut2<'_, f64>,
    ) -> Result<(), MatrixMaterializationError> {
        if out.nrows() != rows.end - rows.start || out.ncols() != self.total_cols {
            return Err(MatrixMaterializationError::MissingRowChunk {
                context: "ChunkedKernelDesignOperator::row_chunk_into shape mismatch",
            });
        }
        if let Some(combined) = self.materialized_combined() {
            out.assign(&combined.slice(s![rows, ..]));
        } else {
            out.assign(&self.row_chunk_combined(rows));
        }
        Ok(())
    }

    fn to_dense(&self) -> Array2<f64> {
        if let Some(combined) = self.materialized_combined() {
            return combined.clone();
        }
        self.row_chunk_combined(0..self.n)
    }
}
