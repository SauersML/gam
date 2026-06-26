//! Low-rank (Woodbury) weighted-Gram path: XᵀWX / XᵀWy with a diagonal-plus-
//! low-rank weight, and the Woodbury capacitance assembly.

use super::*;

// - The diagonal part flows through `xt_diag_x_signed` / `xt_diag_x_psd`
//   exactly as before. When `LowRankWeight::is_rank_zero()` the path is
//   bit-identical to the legacy diagonal flow.
// - The low-rank correction is `(XᵀU)(VᵀX)`, a `p × p` outer product of
//   tall-skinny projections — dimension `p × p`, never `n × n`.
// - Cholesky-friendly factorisation uses the parameter-space Woodbury
//   identity: factor `A = XᵀDX + S` once (the existing dense / sparse
//   path), then solve the small `r × r` capacitance system.
// ---------------------------------------------------------------------------

use gam_linalg::low_rank_weight::LowRankWeight;

/// `Xᵀ W X` for a low-rank-corrected weight, where the diagonal part is
/// assembled by the **existing** signed-Gram kernels and the rank-r
/// correction is added in place via [`LowRankWeight::add_low_rank_xtwx_correction`].
///
/// This is the new sibling of `GamWorkingModel::compute_xtwx_blas`; it is
/// a free function (not a method on `GamWorkingModel`) so it can be reused
/// for backward passes through downstream models without holding a borrow
/// on a working-model instance.
///
/// Rank-0 fast path: returns the legacy diagonal-W Gram unchanged.
pub fn compute_xtwx_low_rank(
    workspace: &mut PirlsWorkspace,
    design: &DesignMatrix,
    weight: &LowRankWeight<'_>,
) -> Result<Array2<f64>, EstimationError> {
    // Diagonal part: reuse the diagonal-W BLAS / sparse path verbatim.
    let diag_owned = weight.diag.to_owned();
    let mut xtwx = GamWorkingModel::compute_xtwx_blas(workspace, design, &diag_owned)?;
    if weight.is_rank_zero() {
        return Ok(xtwx);
    }
    weight
        .add_low_rank_xtwx_correction(design, &mut xtwx)
        .map_err(EstimationError::InvalidInput)?;
    Ok(xtwx)
}

/// `Xᵀ W y` for a low-rank-corrected weight. Used in the right-hand side
/// of the weighted-LS normal equation `(XᵀWX + S) β = XᵀWz`. Rank-0 fast
/// path coincides with `design.compute_xtwy(&d, &y)`.
pub fn compute_xtwy_low_rank(
    design: &DesignMatrix,
    weight: &LowRankWeight<'_>,
    y: &Array1<f64>,
) -> Result<Array1<f64>, EstimationError> {
    weight
        .xtw_y(design, y.view())
        .map_err(EstimationError::InvalidInput)
}

/// Dense multi-output block Fisher assembly for latent / coupled GLM fits.
///
/// Given `X` with shape `(N, K)` and per-row output Fisher blocks `W_i`
/// with shape `(N, P, P)`, this returns the coupled coefficient Hessian
/// ordered as output-major coefficients: `a*K + i`.
///
/// `H[a*K+i, b*K+j] = Σ_n row_weight[n] * X[n,i] * W[n,a,b] * X[n,j]`.
/// When `row_weights` is `None`, all row weights are one.
pub fn dense_block_xtwx(
    design: ArrayView2<'_, f64>,
    fisher_blocks: ArrayView3<'_, f64>,
    row_weights: Option<ArrayView1<'_, f64>>,
) -> Result<Array2<f64>, EstimationError> {
    let n = design.nrows();
    let k = design.ncols();
    let shape = fisher_blocks.shape();
    if shape.len() != 3 || shape[0] != n || shape[1] != shape[2] {
        crate::bail_invalid_estim!(
            "dense block Fisher shape mismatch: expected ({n}, p, p), got {shape:?}"
        );
    }
    if let Some(w) = row_weights.as_ref() {
        if w.len() != n {
            crate::bail_invalid_estim!(
                "dense block row weight length mismatch: expected {n}, got {}",
                w.len()
            );
        }
        if w.iter().any(|v| !v.is_finite() || *v < 0.0) {
            crate::bail_invalid_estim!("dense block row weights must be finite and non-negative");
        }
    }
    let p_out = shape[1];
    let dim = k * p_out;
    // Coupled multi-output Gram `Σ_row (W_row ⊗ x_row x_rowᵀ)` of dimension
    // `(M·k) × (M·k)`. For the multinomial softmax family this `X^T W X` is
    // rebuilt at every inner Newton cycle of every outer smoothing-parameter
    // trial, so its `O(n · M² · k²)` accumulation is the dominant inner cost
    // (#722). The per-row contributions are an independent sum, so fan the row
    // loop across the rayon pool with per-thread dense accumulators reduced by
    // addition — the arithmetic is identical to the serial accumulation,
    // bit-for-bit up to the associativity of the row partition.
    //
    // Finiteness is validated up front in a cheap `O(n · M²)` parallel scan so
    // the hot accumulation stays branch-light and the error is reported with
    // the offending `(row, a, b)` index, preserving the serial contract.
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let nonfinite = (0..n)
        .into_par_iter()
        .filter_map(|row| {
            let rw = row_weights.as_ref().map(|w| w[row]).unwrap_or(1.0);
            for a in 0..p_out {
                for b in 0..p_out {
                    if !(rw * fisher_blocks[[row, a, b]]).is_finite() {
                        return Some((row, a, b));
                    }
                }
            }
            None
        })
        .min();
    if let Some((row, a, b)) = nonfinite {
        crate::bail_invalid_estim!("dense block Fisher entry ({row},{a},{b}) is not finite");
    }
    let mut out = (0..n)
        .into_par_iter()
        .fold(
            || Array2::<f64>::zeros((dim, dim)),
            |mut acc, row| {
                let rw = row_weights.as_ref().map(|w| w[row]).unwrap_or(1.0);
                for a in 0..p_out {
                    for b in 0..p_out {
                        let wab = rw * fisher_blocks[[row, a, b]];
                        if wab == 0.0 {
                            continue;
                        }
                        let row_a = a * k;
                        let row_b = b * k;
                        for i in 0..k {
                            let xi = design[[row, i]];
                            if xi == 0.0 {
                                continue;
                            }
                            let scaled = wab * xi;
                            for j in 0..k {
                                acc[[row_a + i, row_b + j]] += scaled * design[[row, j]];
                            }
                        }
                    }
                }
                acc
            },
        )
        .reduce(
            || Array2::<f64>::zeros((dim, dim)),
            |mut a, b| {
                a += &b;
                a
            },
        );
    for i in 0..dim {
        for j in (i + 1)..dim {
            let avg = 0.5 * (out[[i, j]] + out[[j, i]]);
            out[[i, j]] = avg;
            out[[j, i]] = avg;
        }
    }
    Ok(out)
}

/// Dense multi-output block right-hand side `X^T W Y`, using the same
/// output-major coefficient ordering as [`dense_block_xtwx`].
pub fn dense_block_xtwy(
    design: ArrayView2<'_, f64>,
    fisher_blocks: ArrayView3<'_, f64>,
    response: ArrayView2<'_, f64>,
    row_weights: Option<ArrayView1<'_, f64>>,
) -> Result<Array1<f64>, EstimationError> {
    let n = design.nrows();
    let k = design.ncols();
    let shape = fisher_blocks.shape();
    if shape.len() != 3 || shape[0] != n || shape[1] != shape[2] {
        crate::bail_invalid_estim!(
            "dense block Fisher shape mismatch: expected ({n}, p, p), got {shape:?}"
        );
    }
    let p_out = shape[1];
    if response.dim() != (n, p_out) {
        crate::bail_invalid_estim!(
            "dense block response shape mismatch: expected ({n}, {p_out}), got {}x{}",
            response.nrows(),
            response.ncols()
        );
    }
    if let Some(w) = row_weights.as_ref()
        && w.len() != n
    {
        crate::bail_invalid_estim!(
            "dense block row weight length mismatch: expected {n}, got {}",
            w.len()
        );
    }
    let mut out = Array1::<f64>::zeros(k * p_out);
    for row in 0..n {
        let rw = row_weights.as_ref().map(|w| w[row]).unwrap_or(1.0);
        for a in 0..p_out {
            let mut wy = 0.0_f64;
            for b in 0..p_out {
                let wab = rw * fisher_blocks[[row, a, b]];
                if !wab.is_finite() {
                    crate::bail_invalid_estim!(
                        "dense block Fisher entry ({row},{a},{b}) is not finite"
                    );
                }
                wy += wab * response[[row, b]];
            }
            for i in 0..k {
                out[a * k + i] += design[[row, i]] * wy;
            }
        }
    }
    Ok(out)
}

/// Build the small `r × r` capacitance for the parameter-space Woodbury
/// solve `(A + Û V̂ᵀ)⁻¹ b`, where `A = XᵀDX + S` has already been factored
/// by the caller and `a_inv_uhat = A⁻¹ Û` came out of `r` back-solves
/// against that factor. The returned matrix is `I_r + V̂ᵀ A⁻¹ Û`, the
/// system the caller inverts (Cholesky for symmetric metrics, dense LU
/// otherwise) to apply the low-rank correction to the Newton direction.
pub fn woodbury_gram_capacitance(
    a_inv_uhat: &Array2<f64>,
    vhat: &Array2<f64>,
) -> Result<Array2<f64>, EstimationError> {
    LowRankWeight::gram_capacitance(a_inv_uhat, vhat).map_err(EstimationError::InvalidInput)
}

#[cfg(test)]
mod low_rank_weight_pirls_tests {
    use super::{
        DesignMatrix, LowRankWeight, PirlsWorkspace, compute_xtwx_low_rank, compute_xtwy_low_rank,
        woodbury_gram_capacitance,
    };
    use gam_linalg::matrix::{LinearOperator, SignedWeightsView};
    use ndarray::{Array2, array};

    fn tiny_design() -> DesignMatrix {
        let x = array![
            [1.0, 0.5, -0.2],
            [0.3, 1.2, 0.4],
            [-0.1, 0.7, 1.0],
            [0.6, -0.3, 0.8],
            [0.2, 0.9, -0.5],
        ];
        DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(x))
    }

    #[test]
    pub(crate) fn xtwx_low_rank_matches_diagonal_when_rank_zero() {
        let design = tiny_design();
        let d = array![1.0, 2.0, 0.5, 1.5, 0.8];
        let u = Array2::<f64>::zeros((5, 0));
        let v = Array2::<f64>::zeros((5, 0));
        let weight = LowRankWeight::new(d.view(), u.view(), v.view()).unwrap();
        let mut ws = PirlsWorkspace::new(5, 3, 0, 0);
        let got = compute_xtwx_low_rank(&mut ws, &design, &weight).unwrap();
        let want = design
            .xt_diag_x_signed_op(SignedWeightsView::from_array(&d))
            .unwrap();
        let diff = (&got - &want).mapv(f64::abs).sum();
        assert!(diff < 1e-12, "rank-0 path diverged from diagonal: {}", diff);
    }

    #[test]
    pub(crate) fn xtwy_low_rank_matches_dense_reference() {
        let design = tiny_design();
        let d = array![1.0, 2.0, 0.5, 1.5, 0.8];
        let u = array![
            [0.1, -0.2],
            [0.4, 0.3],
            [-0.1, 0.5],
            [0.2, 0.1],
            [0.0, -0.3]
        ];
        let v = array![[0.2, 0.1], [0.0, 0.4], [0.3, -0.2], [-0.1, 0.6], [0.5, 0.0]];
        let weight = LowRankWeight::new(d.view(), u.view(), v.view()).unwrap();
        let y = array![0.7, -1.2, 0.3, 0.9, -0.4];
        let got = compute_xtwy_low_rank(&design, &weight, &y).unwrap();

        let xdense = design.as_dense().unwrap().to_owned();
        let mut w = Array2::<f64>::zeros((5, 5));
        for i in 0..5 {
            w[[i, i]] = d[i];
        }
        w += &u.dot(&v.t());
        let want = xdense.t().dot(&w.dot(&y));
        let diff: f64 = got
            .iter()
            .zip(want.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff < 1e-10, "xtwy_low_rank diverged: {}", diff);
    }

    #[test]
    pub(crate) fn woodbury_capacitance_is_well_formed() {
        let uhat = array![[0.5, 0.1], [-0.2, 0.7], [0.3, -0.4]];
        let vhat = array![[0.1, 0.2], [0.6, -0.1], [-0.3, 0.4]];
        let cap = woodbury_gram_capacitance(&uhat, &vhat).unwrap();
        let want = {
            let mut m = vhat.t().dot(&uhat);
            for k in 0..2 {
                m[[k, k]] += 1.0;
            }
            m
        };
        let diff: f64 = cap
            .iter()
            .zip(want.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff < 1e-12);
    }
}
