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
        !ridge.is_nan() && ridge >= 0.0,
        "active-set ridge must be nonnegative, got {ridge}"
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
    assert!(
        active
            .iter()
            .take(m)
            .all(|&(atom, _)| (atom as usize) < decoder.nrows()),
        "active atom index is out of range"
    );
    // At the zero prior-variance boundary every posterior-mean code is exactly
    // zero. Handle this analytic model before constructing an infinite Gram.
    if ridge == f32::INFINITY {
        return SparseCode {
            indices: active
                .iter()
                .take(m)
                .map(|&(atom, _)| atom)
                .chain(std::iter::repeat_n(active[0].0, s - m))
                .collect(),
            codes: vec![0.0; s],
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
    }
    let solution = ResolvedActiveGram::new(&gram, ridge as f64, p).solve(rhs.view());

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

/// The ridged active Gram `G + ρI` restricted to the eigenspace the stored atoms
/// resolve, applied as an inverse. [`solve_row_codes`] applies it to `Dᵀx` for the
/// posterior-mean codes, and the decoder Newton step (#3193) applies it to the code
/// sensitivities, so both read one resolution rule.
///
/// `gram` is `G = DᵀD` over the `m` active f32 decoder rows, formed in f64 over `p`
/// entries per pair, and `ridge` is `ρ`. Two sources bound what `G` separates from zero.
/// The stored rows round by at most `μ = ε_f32/2` relative per entry, so `‖E‖_F ≤ μ‖D‖_F`,
/// and by Weyl a singular value of `D` at or below `μ‖D‖_F` is not resolved: an eigenvalue
/// `λ ≤ μ²·tr(G)`. Each Gram entry is a sum of `p` products and rounds by
/// `γ_p = p·ε/(1 − p·ε)` of `Σ_c |d_ic d_jc| ≤ ‖d_i‖‖d_j‖`, which moves an eigenvalue by at
/// most `γ_p·(Σ_i ‖d_i‖)²`.
///
/// An eigendirection `v` inside that band is a combination of atoms that the stored
/// dictionary separates only at rounding, such as a near-duplicate pair. Its right-hand
/// side `vᵀDᵀx = (Dv)ᵀx` is itself a rounding-level quantity, so neither the data nor the
/// prior identifies that coordinate. A solve that keeps it swings the split of the code
/// between those atoms on rounding-level decoder changes while the reconstruction stays put
/// (#2283, job 612377: routing residual 2.5e-4..7.8e-4 for 30 epochs at a fixed decoder,
/// ρ = 1.07e-14). Resolved directions contribute their exact coordinate `vᵀb/(λ + ρ)`, and
/// unresolved ones contribute nothing, which is the minimum-norm code on the resolved
/// subspace. With zero ridge and exactly collinear atoms this is the Moore–Penrose joint
/// least-squares code. At no point are off-diagonal Gram terms discarded.
pub(super) struct ResolvedActiveGram {
    /// `(λ + ρ, v)` for every resolved eigenpair of `G`.
    directions: Vec<(f64, Array1<f64>)>,
}

impl ResolvedActiveGram {
    /// Factor `gram`, the f64 Gram of `m` active f32 decoder rows over `p` entries.
    pub(super) fn new(gram: &Array2<f64>, ridge: f64, p: usize) -> Self {
        use faer::Side;
        use gam_linalg::faer_ndarray::FaerEigh;

        let m = gram.nrows();
        let (eigenvalues, eigenvectors) = gram
            .eigh(Side::Lower)
            .expect("an active Gram matrix must admit a symmetric eigendecomposition");
        let trace = gram.diag().sum();
        let norm_sum: f64 = gram.diag().iter().map(|value| value.max(0.0).sqrt()).sum();
        let unit_roundoff = f64::from(f32::EPSILON) / 2.0;
        let accumulation = p as f64 * f64::EPSILON;
        let resolution = if accumulation < 1.0 {
            unit_roundoff * unit_roundoff * trace
                + accumulation / (1.0 - accumulation) * norm_sum * norm_sum
        } else {
            f64::INFINITY
        };
        let mut directions = Vec::with_capacity(m);
        for eigen_index in 0..m {
            let eigenvalue = eigenvalues[eigen_index];
            assert!(
                eigenvalue >= -resolution,
                "active Gram matrix is not positive semidefinite: eigenvalue {eigenvalue:e}, resolution {resolution:e}"
            );
            if eigenvalue <= resolution {
                continue;
            }
            directions.push((
                eigenvalue + ridge,
                eigenvectors.column(eigen_index).to_owned(),
            ));
        }
        Self { directions }
    }

    /// `Σ_resolved v vᵀ rhs / (λ + ρ)`.
    pub(super) fn solve(&self, rhs: ArrayView1<'_, f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(rhs.len());
        for (scale, eigenvector) in &self.directions {
            let projection = eigenvector.dot(&rhs) / scale;
            out.scaled_add(projection, eigenvector);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn zero_prior_variance_has_zero_codes_with_the_requested_sparse_support() {
        let row = array![3.0_f32, -2.0];
        let decoder = array![[1.0_f32, 0.0], [0.0, 1.0]];
        let codes = solve_row_codes(row.view(), decoder.view(), &[(1, 2.0)], 2, f32::INFINITY);
        assert_eq!(codes.indices, vec![1, 1]);
        assert_eq!(codes.codes, vec![0.0, 0.0]);
    }

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

    #[test]
    fn atoms_equal_to_rounding_share_their_code_instead_of_amplifying_the_difference_2283() {
        // Two f32 atoms one ulp apart in their first entry: a near-duplicate pair whose
        // difference is pure rounding (Gram λ_min ≈ 1e-15, inside the resolution band). The
        // row is orthogonal to their common direction, so the difference gets a right-hand
        // side of about 5e-8. A solve on the whole Gram at ρ = 1e-14 amplifies that by 1/λ_min
        // into codes of order 1e6, and they change sign when the ulp moves. The resolved
        // posterior mean gives the pair one shared coordinate that stays put.
        let row = array![0.8_f32, -0.6, 0.0];
        let ridge = 1.0e-14_f32;
        let solve = |first_entry: f32| {
            let decoder = array![[0.6_f32, 0.8, 0.0], [first_entry, 0.8, 0.0]];
            solve_row_codes(row.view(), decoder.view(), &[(0, 0.0), (1, 0.0)], 2, ridge)
        };
        let up = solve(f32::from_bits(0.6_f32.to_bits() + 1));
        let down = solve(f32::from_bits(0.6_f32.to_bits() - 1));
        for code in [&up, &down] {
            assert!(
                code.codes.iter().all(|value| value.abs() < 1.0e-6),
                "a rounding-level atom difference must not carry a code: {} {}",
                code.codes[0],
                code.codes[1]
            );
            assert!(
                (code.codes[0] - code.codes[1]).abs() < 1.0e-12,
                "the pair must share one coordinate: {} {}",
                code.codes[0],
                code.codes[1]
            );
        }
        assert!(
            (up.codes[0] - down.codes[0]).abs() < 1.0e-6,
            "moving the ulp must not move the code: {} vs {}",
            up.codes[0],
            down.codes[0]
        );
    }
}
