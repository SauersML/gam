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
    let m = active.len();
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
    for i in 0..m.min(s) {
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

/// SPD solve via Cholesky with a Tikhonov-bumped fallback. The system is `s×s`
/// with `s` tiny, so an in-place dense factorisation is appropriate.
fn solve_spd(gram: &Array2<f64>, rhs: &Array1<f64>) -> Array1<f64> {
    use gam_linalg::faer_ndarray::FaerCholesky;
    use faer::Side;

    let m = rhs.len();
    let mut a = gram.clone();
    let mut bump = 0.0f64;
    for _attempt in 0..6 {
        if let Ok(factor) = a.cholesky(Side::Lower) {
            return factor.solvevec(rhs);
        }
        // Indefinite (e.g. exactly collinear atoms): bump the diagonal and retry.
        bump = if bump == 0.0 { 1.0e-8 } else { bump * 16.0 };
        a = gram.clone();
        for i in 0..m {
            a[[i, i]] += bump;
        }
    }
    // Degenerate beyond recovery: fall back to the diagonal (independent atoms).
    let mut out = Array1::<f64>::zeros(m);
    for i in 0..m {
        let d = gram[[i, i]].max(1.0e-12);
        out[i] = rhs[i] / d;
    }
    out
}
