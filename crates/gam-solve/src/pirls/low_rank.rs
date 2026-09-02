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
    // Normalize arbitrary ndarray views once at the API boundary. The Gram
    // kernel below is inherently dense and touches every element many times;
    // retaining strided views here would pay dynamic stride and bounds checks
    // in the O(n · P² · K²) loop. `as_standard_layout` borrows already-standard
    // callers and materializes only genuinely strided inputs.
    let design_standard = design.as_standard_layout();
    let fisher_standard = fisher_blocks.as_standard_layout();
    let fisher_values = fisher_standard
        .as_slice()
        .expect("standard-layout Fisher blocks must be contiguous");
    // Coupled multi-output Gram `Σ_row (W_row ⊗ x_row x_rowᵀ)` of dimension
    // `(M·k) × (M·k)`. Its `(a,b)` coefficient block is exactly the ordinary
    // weighted Gram `Xᵀ diag(W[:,a,b]) X`. Decomposing by output pair routes
    // every block through the shared streaming weighted-crossproduct kernel
    // (tuned matmul, bounded workspace, deterministic chunk association)
    // instead of interpreting the O(n·M²·k²) scalar loop in debug builds.
    //
    // Only the symmetric part of each Fisher block contributes to the Hessian:
    // the old kernel accumulated both `(a,b)` and `(b,a)` and averaged the
    // completed matrix. Computing the average row weight up front is the same
    // linear operation and halves the off-diagonal Gram work.
    //
    // Finiteness is validated up front in a cheap `O(n · M²)` parallel scan so
    // the weighted-Gram kernels receive their documented finite-weight input
    // and errors retain the offending `(row, a, b)` index.
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let nonfinite = (0..n)
        .into_par_iter()
        .filter_map(|row| {
            let rw = row_weights.as_ref().map(|w| w[row]).unwrap_or(1.0);
            for a in 0..p_out {
                for b in 0..p_out {
                    let fisher_index = (row * p_out + a) * p_out + b;
                    if !(rw * fisher_values[fisher_index]).is_finite() {
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

    let mut out = Array2::<f64>::zeros((dim, dim));
    let mut weights = Array1::<f64>::zeros(n);
    for a in 0..p_out {
        for b in a..p_out {
            for row in 0..n {
                let rw = row_weights.as_ref().map(|w| w[row]).unwrap_or(1.0);
                let fisher_row_start = row * p_out * p_out;
                let weight = if a == b {
                    fisher_values[fisher_row_start + a * p_out + a]
                } else {
                    0.5 * (fisher_values[fisher_row_start + a * p_out + b]
                        + fisher_values[fisher_row_start + b * p_out + a])
                };
                weights[row] = rw * weight;
            }
            let gram = gam_linalg::faer_ndarray::fast_xt_diag_x(&design_standard, &weights);
            for i in 0..k {
                for j in 0..k {
                    let value = gram[[i, j]];
                    out[[a * k + i, b * k + j]] = value;
                    out[[b * k + j, a * k + i]] = value;
                }
            }
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

#[cfg(test)]
mod low_rank_weight_pirls_tests {
    use super::dense_block_xtwx;
    use ndarray::{Array2, Array3, array, s};

    #[test]
    pub(crate) fn dense_block_xtwx_matches_scalar_definition_for_strided_views() {
        let design_storage = array![
            [1.0, 99.0, -0.5, 99.0],
            [0.3, 99.0, 1.2, 99.0],
            [-0.7, 99.0, 0.4, 99.0],
        ];
        let design = design_storage.slice(s![.., ..;2]);
        let mut fisher_storage = Array3::<f64>::zeros((3, 2, 4));
        let blocks = [
            [[1.5, -0.2], [-0.2, 0.8]],
            [[0.9, 0.1], [0.1, 1.1]],
            [[1.3, -0.4], [-0.4, 0.7]],
        ];
        for row in 0..3 {
            for a in 0..2 {
                for b in 0..2 {
                    fisher_storage[[row, a, 2 * b]] = blocks[row][a][b];
                }
            }
        }
        let fisher = fisher_storage.slice(s![.., .., ..;2]);
        let got = dense_block_xtwx(design, fisher, None).unwrap();
        let mut want = Array2::<f64>::zeros((4, 4));
        for row in 0..3 {
            for a in 0..2 {
                for b in 0..2 {
                    for i in 0..2 {
                        for j in 0..2 {
                            want[[2 * a + i, 2 * b + j]] +=
                                blocks[row][a][b] * design[[row, i]] * design[[row, j]];
                        }
                    }
                }
            }
        }
        let error = got
            .iter()
            .zip(want.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f64, f64::max);
        assert!(error <= 4.0 * f64::EPSILON, "maximum error {error:e}");
    }
}
