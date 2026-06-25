use super::*;

pub(crate) fn ctn_penalty_scale_log_lambdas(
    penalties: &[PenaltyMatrix],
    likelihood_gram: &Array2<f64>,
) -> Array1<f64> {
    if penalties.is_empty() {
        return Array1::zeros(0);
    }

    let likelihood_scale = matrix_diag_mean_abs(likelihood_gram).max(CTN_SEED_SCALE_FLOOR);
    Array1::from_iter(penalties.iter().map(|penalty| {
        let penalty_scale = penalty_diag_scale(penalty).max(CTN_SEED_SCALE_FLOOR);
        // Lower-bound the SEED log-lambda at 0 (i.e., λ ≥ 1) so we never
        // start the outer optimizer in the under-regularized regime where
        // the CTN inner is structurally rank-deficient (small-n / p > n).
        // The outer BFGS is free to step below 0 when there's enough data
        // to support it; only the cold-start is constrained. Without this
        // floor, ratios like n=64, p_resp×p_cov ≈ 200 produce a seed of
        // log_lambda ≈ -12 (λ ≈ 6e-6), which leaves the inner solve to
        // pick wild β coefficients and cascade into predict-time monotonicity
        // violations (h' < 0 on the response grid, observed as -1e15 spikes
        // in CI synthetic-large-scale tests).
        (likelihood_scale / penalty_scale)
            .ln()
            .clamp(CTN_SEED_LOG_LAMBDA_MIN, CTN_SEED_LOG_LAMBDA_MAX)
    }))
}

pub(crate) fn penalty_diag_scale(penalty: &PenaltyMatrix) -> f64 {
    match penalty {
        PenaltyMatrix::Dense(matrix) => {
            matrix_diag_mean_abs(matrix).max(matrix_frobenius_rms(matrix))
        }
        PenaltyMatrix::KroneckerFactored { left, right } => {
            let diag_scale = matrix_diag_mean_abs(left) * matrix_diag_mean_abs(right);
            let frob_scale = matrix_frobenius_rms(left) * matrix_frobenius_rms(right);
            diag_scale.max(frob_scale)
        }
        PenaltyMatrix::Blockwise { local, .. } => {
            matrix_diag_mean_abs(local).max(matrix_frobenius_rms(local))
        }
        PenaltyMatrix::Labeled { inner, .. } => penalty_diag_scale(inner),
        PenaltyMatrix::Fixed { inner, .. } => penalty_diag_scale(inner),
    }
}

pub(crate) fn matrix_diag_mean_abs(matrix: &Array2<f64>) -> f64 {
    let d = matrix.nrows().min(matrix.ncols());
    if d == 0 {
        return 0.0;
    }
    matrix.diag().iter().map(|v| v.abs()).sum::<f64>() / d as f64
}

pub(crate) fn matrix_frobenius_rms(matrix: &Array2<f64>) -> f64 {
    let d = matrix.nrows().max(1).min(matrix.ncols().max(1));
    (matrix.iter().map(|v| v * v).sum::<f64>() / d as f64).sqrt()
}

/// Weighted cross-product of two rowwise-Kronecker designs, kept strictly
/// factored: output block (a, c) equals `B^T diag(w_i A_{ia} C_{ic}) D`.
pub(crate) fn factored_weighted_cross(
    a: &Array2<f64>,
    b: &Array2<f64>,
    weights: ndarray::ArrayView1<'_, f64>,
    c: &Array2<f64>,
    d: &Array2<f64>,
    policy: &ResourcePolicy,
) -> Result<Array2<f64>, String> {
    let n = weights.len();
    if a.nrows() != n || b.nrows() != n || c.nrows() != n || d.nrows() != n {
        return Err(TransformationNormalError::InvalidInput {
            reason: format!(
                "factored_weighted_cross row mismatch: weights={n}, a={}, b={}, c={}, d={}",
                a.nrows(),
                b.nrows(),
                c.nrows(),
                d.nrows()
            ),
        }
        .into());
    }
    let pa = a.ncols();
    let pc = c.ncols();
    let pb = b.ncols();
    let pd = d.ncols();

    let mut out = Array2::<f64>::zeros((pa * pb, pc * pd));

    // The weighted Gram of a rowwise-Kronecker (te(x,z)) design is the `pa × pc`
    // grid of independent `pb × pd` blocks `B^T diag(w · A_{·,ia} · C_{·,ic}) D`.
    // Each block streams all `n` rows, so on a tensor smooth with modest marginal
    // bases (te(x,z,k=7) ⇒ pb,pd ≈ 6) the per-block GEMM is tiny while the grid
    // has `pa·pc` ≈ 36–49 fully independent entries — the prior serial double
    // loop left every core but one idle and paid faer's parallel-dispatch
    // overhead on each tiny block. Fan the OUTER `ia` rows across the Rayon pool:
    // block `(ia, ic)` lands in output rows `ia·pb..` and never overlaps another
    // `ia`, so `axis_chunks_iter_mut` hands each task a disjoint `pb`-row band —
    // no unsafe, no shared writes. `with_nested_parallel` pins the inner
    // `chunked_weighted_bt_d` GEMM to `Par::Seq` so the row fan-out does not
    // multiply against the faer pool (gam#1082).
    use crate::faer_ndarray::with_nested_parallel;
    use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

    out.axis_chunks_iter_mut(ndarray::Axis(0), pb.max(1))
        .into_par_iter()
        .enumerate()
        .for_each(|(ia, mut row_band)| {
            with_nested_parallel(|| {
                let a_col = a.column(ia);
                let mut pair_weights = Array1::<f64>::zeros(n);
                for ic in 0..pc {
                    let c_col = c.column(ic);
                    for r in 0..n {
                        pair_weights[r] = weights[r] * a_col[r] * c_col[r];
                    }
                    let block = chunked_weighted_bt_d(b, pair_weights.view(), d, policy);
                    row_band
                        .slice_mut(s![.., ic * pd..(ic + 1) * pd])
                        .assign(&block);
                }
            });
        });

    Ok(out)
}

/// Chunked weighted B^T diag(w) D product without materializing any
/// full rowwise-Kronecker intermediate.
///
/// Each chunk weights `D` rows in-place, then accumulates `B[chunk]^T · DW`
/// directly into `out` via a faer SIMD matmul with `Accum::Add`. This
/// eliminates the per-chunk `Array2` from the previous `out += &bl.t().dot(&dw)`
/// pattern (one allocation + one element-wise `+=` pass per chunk) and uses
/// the multi-threaded faer kernel instead of ndarray's serial dot.
pub(crate) fn chunked_weighted_bt_d(
    b: &Array2<f64>,
    weights: ndarray::ArrayView1<'_, f64>,
    d: &Array2<f64>,
    policy: &ResourcePolicy,
) -> Array2<f64> {
    use crate::faer_ndarray::{FaerArrayView, array2_to_matmut, matmul_parallelism};
    use faer::Accum;
    use faer::linalg::matmul::matmul;

    let n = weights.len();
    let pb = b.ncols();
    let pd = d.ncols();
    let rows_per_chunk =
        gam_runtime::resource::rows_for_target_bytes(policy.row_chunk_target_bytes, pb + pd);
    let mut out = Array2::<f64>::zeros((pb, pd));
    if n == 0 || pb == 0 || pd == 0 {
        return out;
    }
    let mut out_view = array2_to_matmut(&mut out);
    let mut dw_buf = Array2::<f64>::zeros((rows_per_chunk.min(n), pd));
    for start in (0..n).step_by(rows_per_chunk) {
        let end = (start + rows_per_chunk).min(n);
        let rows = end - start;
        let bl = b.slice(s![start..end, ..]);
        let dl = d.slice(s![start..end, ..]);
        {
            let mut dw_slice = dw_buf.slice_mut(s![..rows, ..]);
            for local in 0..rows {
                let w = weights[start + local];
                let drow = dl.row(local);
                let mut wrow = dw_slice.row_mut(local);
                ndarray::Zip::from(&mut wrow)
                    .and(&drow)
                    .for_each(|dst, &src| *dst = w * src);
            }
        }
        let bl_view = FaerArrayView::new(&bl);
        let dw_slice = dw_buf.slice(s![..rows, ..]);
        let dw_view = FaerArrayView::new(&dw_slice);
        let par = matmul_parallelism(pb, pd, rows);
        matmul(
            out_view.as_mut(),
            Accum::Add,
            bl_view.as_ref().transpose(),
            dw_view.as_ref(),
            1.0,
            par,
        );
    }
    out
}

/// Chunked weighted `B^T diag(w) D` product where `B` and `D` are
/// operator-backed `DesignMatrix` instances. Materializes only one row chunk
/// at a time using the operator's `row_chunk` primitive, so neither factor's
/// full dense form ever lives in memory.
///
/// Each chunk's contribution accumulates into `out` via a faer SIMD matmul
/// with `Accum::Add` rather than `out += &bl.t().dot(&dw)`. This drops one
/// `Array2` allocation per chunk and routes the inner GEMM through faer's
/// multi-threaded kernel with a work-aware parallelism choice.
pub(crate) fn chunked_weighted_bt_d_designmatrix(
    b: &DesignMatrix,
    weights: ndarray::ArrayView1<'_, f64>,
    d: &DesignMatrix,
    policy: &ResourcePolicy,
) -> Result<Array2<f64>, String> {
    use crate::faer_ndarray::{FaerArrayView, array2_to_matmut, matmul_parallelism};
    use faer::Accum;
    use faer::linalg::matmul::matmul;

    let n = weights.len();
    let pb = b.ncols();
    let pd = d.ncols();
    let rows_per_chunk =
        gam_runtime::resource::rows_for_target_bytes(policy.row_chunk_target_bytes, pb + pd);
    let mut out = Array2::<f64>::zeros((pb, pd));
    if n == 0 || pb == 0 || pd == 0 {
        return Ok(out);
    }
    let mut out_view = array2_to_matmut(&mut out);
    for start in (0..n).step_by(rows_per_chunk) {
        let end = (start + rows_per_chunk).min(n);
        let rows = end - start;
        let bl = b.try_row_chunk(start..end).map_err(|e| e.to_string())?;
        let mut dw = d.try_row_chunk(start..end).map_err(|e| e.to_string())?;
        for local in 0..rows {
            let w = weights[start + local];
            if w != 1.0 {
                let mut wrow = dw.row_mut(local);
                wrow.mapv_inplace(|v| w * v);
            }
        }
        let bl_view = FaerArrayView::new(&bl);
        let dw_view = FaerArrayView::new(&dw);
        let par = matmul_parallelism(pb, pd, rows);
        matmul(
            out_view.as_mut(),
            Accum::Add,
            bl_view.as_ref().transpose(),
            dw_view.as_ref(),
            1.0,
            par,
        );
    }
    Ok(out)
}
