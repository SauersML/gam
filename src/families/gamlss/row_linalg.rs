//! Small pure leaf linear-algebra helpers shared across the GAMLSS family
//! implementations: closed-form symmetric 2×2 PSD-clamped eigendecomposition
//! and per-row matrix scaling. Both are dependency-free scalar/dense kernels
//! relocated here verbatim from `gamlss.rs` to shrink the parent module; the
//! parent re-imports them so every call site is unchanged.

use super::GamlssError;
use ndarray::{Array1, Array2};

pub(crate) fn psd_clamp_2x2(a: f64, b: f64, d: f64) -> (f64, f64, f64, f64, f64, f64) {
    // Symmetric 2×2 eigenvalues via the closed-form formula.
    let trace = a + d;
    let det = a * d - b * b;
    let disc = (trace * trace * 0.25 - det).max(0.0).sqrt();
    let lam1 = (trace * 0.5 + disc).max(0.0); // larger eigenvalue (clamped)
    let lam2 = (trace * 0.5 - disc).max(0.0); // smaller eigenvalue (clamped)
    // Eigenvector for lam1: (lam1-d, b) normalized. Check: b*(lam1-d)+(d-lam1)*b=0 ✓
    let (u1_0, u1_1, u2_0, u2_1) = if b.abs() > 1e-15 * (a.abs() + d.abs()).max(1.0) {
        let ex = lam1 - d;
        let ey = b;
        let norm = (ex * ex + ey * ey).sqrt().max(1e-300);
        let (e0x, e0y) = (ex / norm, ey / norm);
        // Second eigenvector orthogonal to first: (-e0y, e0x).
        let (e1x, e1y) = (-e0y, e0x);
        (e0x, e0y, e1x, e1y)
    } else if a >= d {
        // Already diagonal (or nearly so), larger eigenvalue is a.
        (1.0, 0.0, 0.0, 1.0)
    } else {
        (0.0, 1.0, 1.0, 0.0)
    };
    (lam1, lam2, u1_0, u1_1, u2_0, u2_1)
}

pub(crate) fn scale_matrix_rows(
    mat: &Array2<f64>,
    coeffs: &Array1<f64>,
) -> Result<Array2<f64>, String> {
    if mat.nrows() != coeffs.len() {
        return Err(GamlssError::DimensionMismatch {
            reason: format!(
                "row scaling dimension mismatch: matrix has {} rows but coeffs have {} entries",
                mat.nrows(),
                coeffs.len()
            ),
        }
        .into());
    }
    Ok(Array2::from_shape_fn(mat.dim(), |(i, j)| {
        mat[[i, j]] * coeffs[i]
    }))
}
