//! Spatial-kernel design operators for basis construction.

use faer::Accum;
use faer::Par;
use faer::linalg::matmul::matmul;
use gam_linalg::faer_ndarray::{FaerArrayView, array2_to_matmut, fast_atv, fast_av};
use gam_linalg::matrix::{DenseDesignOperator, FiniteSignedWeightsView, LinearOperator};
use gam_problem::Gauge;
use gam_runtime::resource::{MaterializationPolicy, MatrixMaterializationError};
use ndarray::{Array1, Array2, ArrayViewMut2, s};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;
use std::ops::Range;
use std::sync::Arc;

const KERNEL_OPERATOR_ROW_CHUNK_SIZE: usize = 2048;

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

impl SpatialKernelEvaluator for Arc<dyn SpatialKernelEvaluator> {
    fn eval(&self, x: &[f64], c: &[f64]) -> f64 {
        self.as_ref().eval(x, c)
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
    /// The routing contract that selected this streamed representation. It is
    /// propagated through coefficient/block wrappers so a later permissive
    /// caller cannot silently materialize the full design.
    materialization_policy: MaterializationPolicy,
}

impl<K: SpatialKernelEvaluator> ChunkedKernelDesignOperator<K> {
    pub fn new(
        data: Arc<Array2<f64>>,
        centers: Arc<Array2<f64>>,
        kernel: K,
        kernel_gauge: Option<Arc<Gauge>>,
        poly_basis: Option<Arc<Array2<f64>>>,
        materialization_policy: MaterializationPolicy,
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
            materialization_policy,
        })
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
        if weights.len() != self.n {
            return Err(format!(
                "ChunkedSpatialKernelDesign::diag_xtw_x weight length mismatch: weights={}, nrows={}",
                weights.len(),
                self.n
            ));
        }
        FiniteSignedWeightsView::try_from_array(weights)
            .map_err(|reason| format!("ChunkedSpatialKernelDesign::diag_xtw_x: {reason}"))?;
        let p = self.total_cols;
        // The basis router chose this operator because the dense design was not
        // admitted. Preserve that decision and accumulate XᵀWX from bounded
        // row chunks instead of hiding a second, operator-local dense cache.
        let n = self.n;
        if n == 0 || p == 0 {
            return Ok(Array2::<f64>::zeros((p, p)));
        }
        let chunk_starts: Vec<usize> = (0..n).step_by(KERNEL_OPERATOR_ROW_CHUNK_SIZE).collect();
        // Deterministic parallel reduction over the row chunks: length-only
        // pairwise tree, so the accumulated XᵀWX never depends on thread count
        // or rayon's demand-driven fold/reduce grouping (#2228).
        let xtwx = gam_linalg::pairwise_reduce::par_deterministic_block_fold(
            chunk_starts.len(),
            |idx_range: core::ops::Range<usize>| {
                let mut acc = Array2::<f64>::zeros((p, p));
                for &start in &chunk_starts[idx_range] {
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
                }
                acc
            },
            |mut a, b| {
                a += &b;
                a
            },
        )
        .unwrap_or_else(|| Array2::<f64>::zeros((p, p)));
        Ok(xtwx)
    }
}

impl<K: SpatialKernelEvaluator> ChunkedKernelDesignOperator<K> {
    /// Build a combined row chunk `[kernel_chunk | poly_chunk]` without ever
    /// materializing the full design.
    pub(crate) fn row_chunk_combined(&self, rows: Range<usize>) -> Array2<f64> {
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
    /// A chunked kernel design never exposes an implicit full-design cache.
    /// Callers that explicitly accept densification must go through the
    /// governed `DenseDesignMatrix` materialization APIs.
    fn as_dense_ref(&self) -> Option<&Array2<f64>> {
        None
    }

    fn materialization_policy(&self) -> Option<MaterializationPolicy> {
        Some(self.materialization_policy.clone())
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
        out.assign(&self.row_chunk_combined(rows));
        Ok(())
    }

    fn to_dense(&self) -> Array2<f64> {
        self.row_chunk_combined(0..self.n)
    }
}

#[cfg(test)]
mod chunked_kernel_operator_tests {
    use super::*;
    use gam_linalg::matrix::DenseDesignMatrix;
    use ndarray::{Array1, Array2, array};
    use std::sync::Arc;

    fn strict_materialization_policy() -> MaterializationPolicy {
        gam_runtime::resource::ResourcePolicy::analytic_operator_required().material_policy()
    }

    #[test]
    fn chunked_kernel_operator_uses_center_rows_for_column_count() {
        let data = Arc::new(array![[0.0, 1.0], [1.0, 0.5]]);
        let centers = Arc::new(array![[0.0, 0.0], [1.0, 1.0], [2.0, -1.0]]);
        let kernel =
            |x: &[f64], c: &[f64]| x.iter().zip(c.iter()).map(|(xi, ci)| xi * ci).sum::<f64>();
        let operator = ChunkedKernelDesignOperator::new(
            data,
            centers,
            kernel,
            None,
            None,
            strict_materialization_policy(),
        )
        .expect("chunked kernel operator");

        assert_eq!(operator.ncols(), 3);
        let chunk = operator.row_chunk_combined(0..2);
        assert_eq!(chunk.dim(), (2, 3));
    }

    #[test]
    fn chunked_kernel_operator_rejects_incompatible_optional_shapes() {
        let data = Arc::new(array![[0.0, 1.0], [1.0, 0.5]]);
        let centers = Arc::new(array![[0.0, 0.0], [1.0, 1.0], [2.0, -1.0]]);
        let kernel = |_: &[f64], _: &[f64]| 0.0;
        let bad_gauge = Arc::new(gam_problem::Gauge::from_block_transforms(&[
            Array2::<f64>::zeros((2, 1)),
        ]));
        let bad_poly = Arc::new(Array2::<f64>::zeros((3, 1)));

        let gauge_err = match ChunkedKernelDesignOperator::new(
            data.clone(),
            centers.clone(),
            kernel,
            Some(bad_gauge),
            None,
            strict_materialization_policy(),
        ) {
            // SAFETY: test asserting validation rejects mismatched gauge raw width; Ok means the validator regressed.
            Ok(_) => panic!("gauge raw width should match centers rows"),
            Err(err) => err,
        };
        assert!(gauge_err.contains("kernel gauge raw width 2 != centers rows 3"));

        let poly_err = match ChunkedKernelDesignOperator::new(
            data,
            centers,
            kernel,
            None,
            Some(bad_poly),
            strict_materialization_policy(),
        ) {
            // SAFETY: test asserting validation rejects mismatched poly rows; Ok means the validator regressed.
            Ok(_) => panic!("poly rows should match data rows"),
            Err(err) => err,
        };
        assert!(poly_err.contains("poly_basis rows 3 != data rows 2"));
    }

    #[test]
    fn chunked_kernel_operator_canonicalizes_non_contiguous_inputs() {
        let data = Arc::new(array![[0.0, 1.0], [1.0, 0.5]].reversed_axes());
        let centers = Arc::new(array![[0.0, 1.0, 2.0], [0.0, 1.0, -1.0]].reversed_axes());
        assert!(!data.is_standard_layout());
        assert!(!centers.is_standard_layout());

        let kernel =
            |x: &[f64], c: &[f64]| x.iter().zip(c.iter()).map(|(xi, ci)| xi * ci).sum::<f64>();
        let operator = ChunkedKernelDesignOperator::new(
            data,
            centers,
            kernel,
            None,
            None,
            strict_materialization_policy(),
        )
        .expect("chunked kernel operator");
        let chunk = operator.row_chunk_combined(0..2);

        assert_eq!(chunk.dim(), (2, 3));
        assert_eq!(chunk[[0, 0]], 0.0);
        assert_eq!(chunk[[1, 1]], 1.5);
    }
    #[test]
    fn chunked_kernel_operator_never_exposes_an_implicit_dense_cache() {
        let data = Arc::new(array![[0.0, 1.0], [1.0, 0.5], [2.0, -1.0]]);
        let centers = Arc::new(array![[0.0, 0.0], [1.0, 1.0]]);
        let kernel =
            |x: &[f64], c: &[f64]| x.iter().zip(c.iter()).map(|(xi, ci)| xi * ci).sum::<f64>();
        let op = ChunkedKernelDesignOperator::new(
            data,
            centers,
            kernel,
            None,
            None,
            strict_materialization_policy(),
        )
        .expect("chunked kernel operator");
        let expected = op.to_dense();

        let dense_design = DenseDesignMatrix::from(Arc::new(op));

        let probe = Array1::from_elem(3, 1.0);
        let applied = dense_design.apply_transpose(&probe);
        let expected_applied = expected.t().dot(&probe);
        for (got, want) in applied.iter().zip(expected_applied.iter()) {
            assert!((got - want).abs() < 1e-12);
        }
        assert!(
            dense_design.as_dense_ref().is_none(),
            "chunked kernel operations must not warm a hidden full-design cache"
        );
    }

    #[test]
    fn chunked_kernel_gram_is_signed_and_rejects_nonfinite_rows() {
        let data = Arc::new(array![[1.0], [2.0], [3.0]]);
        let centers = Arc::new(array![[0.5], [-1.0]]);
        let kernel = |x: &[f64], c: &[f64]| x[0] + c[0];
        let op = ChunkedKernelDesignOperator::new(
            data,
            centers,
            kernel,
            None,
            None,
            strict_materialization_policy(),
        )
        .unwrap();
        let dense = op.to_dense();
        let weights = array![2.0, -3.0, 0.25];
        let weighted_dense = dense.clone() * weights.view().insert_axis(ndarray::Axis(1));
        let expected = dense.t().dot(&weighted_dense);
        let got = op.diag_xtw_x(&weights).unwrap();
        assert!((&got - &expected).iter().all(|value| value.abs() < 1e-12));

        let err = op
            .diag_xtw_x(&array![1.0, f64::NAN, f64::INFINITY])
            .unwrap_err();
        assert!(err.contains("row 1"), "unexpected diagnostic: {err}");
    }
}
