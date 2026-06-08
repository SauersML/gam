//! Fisher–Rao precision-weight normalization for response-geometry REML.
//!
//! A caller may supply the per-observation Fisher–Rao precision metric coupling
//! the tangent-coordinate residuals as a 1-D vector (a shared isotropic scale
//! per row), a single 2-D `(dim, dim)` matrix (shared across all rows), or a
//! full 3-D `(n_rows, dim, dim)` stack. This broadcasts any of those to the
//! canonical per-row block form and validates finiteness, per-row symmetry, and
//! non-negative diagonals. Single source of truth shared by the
//! `response_geometry_normalize_fisher_rao` FFI shim and any core consumer.

use ndarray::{Array3, ArrayViewD, IxDyn};

/// Broadcast and validate a Fisher–Rao weight array into `(n_rows, dim, dim)`
/// precision blocks. Accepts a 1-D `(n_rows,)` isotropic scale, a 2-D
/// `(dim, dim)` shared matrix, or a 3-D `(n_rows, dim, dim)` stack.
pub fn normalize_fisher_rao_blocks(
    arr: ArrayViewD<'_, f64>,
    n_rows: usize,
    dim: usize,
) -> Result<Array3<f64>, String> {
    if !arr.iter().all(|v| v.is_finite()) {
        return Err("fisher_rao_w must contain only finite values".to_string());
    }
    let shape = arr.shape().to_vec();
    let out: Array3<f64> = match arr.ndim() {
        1 => {
            if shape[0] != n_rows {
                return Err(format!(
                    "fisher_rao_w vector must have length {n_rows}; got {}",
                    shape[0]
                ));
            }
            let mut block = Array3::<f64>::zeros((n_rows, dim, dim));
            for row in 0..n_rows {
                let value = arr[IxDyn(&[row])];
                for d in 0..dim {
                    block[[row, d, d]] = value;
                }
            }
            block
        }
        2 => {
            if shape[0] != dim || shape[1] != dim {
                return Err(format!(
                    "fisher_rao_w matrix must have shape ({dim}, {dim}); got ({}, {})",
                    shape[0], shape[1]
                ));
            }
            let mut block = Array3::<f64>::zeros((n_rows, dim, dim));
            for row in 0..n_rows {
                for r in 0..dim {
                    for c in 0..dim {
                        block[[row, r, c]] = arr[IxDyn(&[r, c])];
                    }
                }
            }
            block
        }
        3 => {
            if shape[0] != n_rows || shape[1] != dim || shape[2] != dim {
                return Err(format!(
                    "fisher_rao_w must have shape ({n_rows}, {dim}, {dim}); got ({}, {}, {})",
                    shape[0], shape[1], shape[2]
                ));
            }
            let mut block = Array3::<f64>::zeros((n_rows, dim, dim));
            for row in 0..n_rows {
                for r in 0..dim {
                    for c in 0..dim {
                        block[[row, r, c]] = arr[IxDyn(&[row, r, c])];
                    }
                }
            }
            block
        }
        _ => return Err("fisher_rao_w must be a 1-D, 2-D, or 3-D numeric array".to_string()),
    };
    for row in 0..n_rows {
        for r in 0..dim {
            for c in 0..dim {
                let a = out[[row, r, c]];
                let b = out[[row, c, r]];
                if (a - b).abs() > 1.0e-10 * (1.0 + a.abs() + b.abs()) {
                    return Err("fisher_rao_w must be symmetric in every row block".to_string());
                }
            }
            if out[[row, r, r]] < 0.0 {
                return Err("fisher_rao_w diagonal entries must be non-negative".to_string());
            }
        }
    }
    Ok(out)
}
