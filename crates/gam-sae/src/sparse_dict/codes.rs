//! Per-row sparse codes via a small active-set least-squares solve.
//!
//! Given a row `x` and the `s` atoms the router selected for it, the optimal
//! codes minimise `‖x − Σ_j c_j d_{a_j}‖² + ρ‖c‖²`. That is the tiny
//! `s×s` normal-equation system `(Gᵃ + ρI) c = Dᵃ x` where `Gᵃ` is the Gram of
//! the active atoms and `Dᵃ x` are their projections. `s` is the shared active
//! budget (a handful), so this is a cheap dense solve regardless of `K`.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// One row's fixed-width sparse code.
#[derive(Clone, Debug)]
pub struct SparseCode {
    /// Active atom indices, length `s` (padded with the last live index when the
    /// row had fewer than `s` candidates; padded entries carry a zero code).
    pub indices: Vec<u32>,
    /// Codes aligned with [`Self::indices`], length `s`.
    pub codes: Vec<f32>,
}

/// Solve the active-set least-squares codes for one row.
///
/// `active` is the router's `(atom, score)` shortlist; only the atom indices are
/// used (the score chose the set, the LS solve sets the magnitudes). `s` is the
/// fixed output width: shorter shortlists are padded so every row stores exactly
/// `s` slots.
pub fn solve_row_codes(
    row: ArrayView1<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    active: &[(u32, f32)],
    s: usize,
    ridge: f32,
) -> SparseCode {
    assert!(s > 0, "sparse-code support width must be positive");
    assert!(
        ridge.is_finite() && ridge >= 0.0,
        "active-set ridge must be finite and nonnegative, got {ridge}"
    );
    assert_eq!(
        row.len(),
        decoder.ncols(),
        "row width must equal decoder width"
    );
    let m = active.len().min(s);
    if m == 0 {
        // No live atom — emit zero code on atom 0 (padding contract).
        return SparseCode {
            indices: vec![0u32; s],
            codes: vec![0.0f32; s],
        };
    }
    let p = row.len();
    // Active Gram (m×m) and rhs (m) in f64 for a well-conditioned solve.
    let mut gram = Array2::<f64>::zeros((m, m));
    let mut rhs = Array1::<f64>::zeros(m);
    for i in 0..m {
        let ai = active[i].0 as usize;
        assert!(
            ai < decoder.nrows(),
            "active atom index {ai} is out of range"
        );
        let di = decoder.row(ai);
        let mut proj = 0.0f64;
        for c in 0..p {
            proj += di[c] as f64 * row[c] as f64;
        }
        rhs[i] = proj;
        for j in i..m {
            let aj = active[j].0 as usize;
            let dj = decoder.row(aj);
            let mut g = 0.0f64;
            for c in 0..p {
                g += di[c] as f64 * dj[c] as f64;
            }
            gram[[i, j]] = g;
            gram[[j, i]] = g;
        }
        gram[[i, i]] += ridge as f64;
    }
    let solution = solve_spd(&gram, &rhs);

    let mut indices = Vec::with_capacity(s);
    let mut codes = Vec::with_capacity(s);
    for i in 0..m {
        indices.push(active[i].0);
        codes.push(solution[i] as f32);
    }
    // Pad to fixed width with the first active index, zero code.
    while indices.len() < s {
        indices.push(active[0].0);
        codes.push(0.0f32);
    }
    SparseCode { indices, codes }
}

/// Solve the positive-semidefinite active Gram system. A positive ridge makes
/// the system strictly positive definite and takes the Cholesky path. With
/// zero ridge, collinear selected atoms are legitimate; the Moore--Penrose
/// solution is then the unique minimum-norm joint least-squares code. At no
/// point are off-diagonal Gram terms discarded.
fn solve_spd(gram: &Array2<f64>, rhs: &Array1<f64>) -> Array1<f64> {
    use faer::Side;
    use gam_linalg::faer_ndarray::{FaerCholesky, FaerEigh};

    let m = rhs.len();
    if let Ok(factor) = gram.cholesky(Side::Lower) {
        return factor.solvevec(rhs);
    }

    let (eigenvalues, eigenvectors) = gram
        .eigh(Side::Lower)
        .expect("an active Gram matrix must admit a symmetric eigendecomposition");
    let spectral_radius = eigenvalues
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if spectral_radius == 0.0 {
        return Array1::<f64>::zeros(m);
    }
    let cutoff = f64::EPSILON * (m as f64) * spectral_radius;
    let mut out = Array1::<f64>::zeros(m);
    for eigen_index in 0..m {
        let eigenvalue = eigenvalues[eigen_index];
        assert!(
            eigenvalue >= -cutoff,
            "active Gram matrix is not positive semidefinite: eigenvalue {eigenvalue:e}, cutoff {cutoff:e}"
        );
        if eigenvalue <= cutoff {
            continue;
        }
        let eigenvector = eigenvectors.column(eigen_index);
        let projection = eigenvector.dot(rhs) / eigenvalue;
        for coordinate in 0..m {
            out[coordinate] += projection * eigenvector[coordinate];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn duplicate_selected_atoms_use_joint_minimum_norm_least_squares() {
        let row = array![1.0_f32, 0.0];
        let decoder = array![[1.0_f32, 0.0], [1.0_f32, 0.0]];
        let active = vec![(0_u32, 0.0_f32), (1_u32, 0.0_f32)];

        let code = solve_row_codes(row.view(), decoder.view(), &active, 2, 0.0);

        assert!((code.codes[0] - 0.5).abs() < 1.0e-6);
        assert!((code.codes[1] - 0.5).abs() < 1.0e-6);
        let reconstructed = code.codes[0] * decoder[[0, 0]] + code.codes[1] * decoder[[1, 0]];
        assert!((reconstructed - 1.0).abs() < 1.0e-6);
    }
}
