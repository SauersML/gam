//! Per-row sparse codes via a small active-set least-squares solve.
//!
//! Given a row `x` and the `s` atoms the router selected for it, the optimal
//! Gaussian posterior-mean codes minimise `‖x − Σ_j c_j d_{a_j}‖² + ρ‖c‖²`.
//! Solve from the active decoder's thin SVD, retaining resolved directions that
//! forming its Gram would erase. Storage is O(P·s), independent of total K.

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
    let solution = posterior_mean(row, decoder, &active[..m], ridge as f64);

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

fn posterior_mean(
    row: ArrayView1<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    active: &[(u32, f32)],
    ridge: f64,
) -> Array1<f64> {
    use gam_linalg::faer_ndarray::FaerSvd;

    let m = active.len();
    if m == 1 {
        let direction = decoder.row(active[0].0 as usize);
        let norm: f64 = direction.iter().map(|&v| (v as f64).powi(2)).sum();
        let projection: f64 = direction
            .iter()
            .zip(row.iter())
            .map(|(&d, &x)| d as f64 * x as f64)
            .sum();
        return Array1::from_vec(vec![if norm + ridge == 0.0 {
            0.0
        } else {
            projection / (norm + ridge)
        }]);
    }
    let basis = Array2::from_shape_fn((row.len(), m), |(feature, slot)| {
        decoder[[active[slot].0 as usize, feature]] as f64
    });
    let (left, singular, right_t) = basis
        .svd(true, true)
        .expect("finite active decoder must admit a thin SVD");
    let left = left.expect("requested left singular vectors");
    let right_t = right_t.expect("requested right singular vectors");
    let spectral_radius = singular.iter().copied().fold(0.0_f64, f64::max);
    let rank_floor = f64::EPSILON * row.len().max(m) as f64 * spectral_radius;
    let mut solution = Array1::<f64>::zeros(m);
    for (mode, &value) in singular.iter().enumerate() {
        // Only the unregularized minimum-norm inverse needs a numerical rank
        // decision. Positive ridge defines every mode, however small it is.
        if value == 0.0 || (ridge == 0.0 && value <= rank_floor) {
            continue;
        }
        let projection: f64 = left
            .column(mode)
            .iter()
            .zip(row.iter())
            .map(|(&u, &x)| u * x as f64)
            .sum();
        let coefficient = value / (value * value + ridge) * projection;
        for slot in 0..m {
            solution[slot] += right_t[[mode, slot]] * coefficient;
        }
    }
    solution
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn posterior_mean_preserves_a_small_direction_lost_by_the_normal_equations() {
        let delta = 1e-10_f32;
        let row = array![0.0_f32, 1.0];
        let decoder = array![[1.0_f32, 0.0], [1.0, delta]];
        let rho = delta * delta;
        let code = solve_row_codes(row.view(), decoder.view(), &[(0, 0.0), (1, 0.0)], 2, rho);
        // Exact solve gives c1=delta / (delta² + rho + rho/(1+rho)).
        let expected =
            delta as f64 / ((delta as f64).powi(2) + rho as f64 + rho as f64 / (1.0 + rho as f64));
        assert!((code.codes[1] as f64 / expected - 1.0).abs() < 1e-6);
        assert!((code.codes[0] as f64 / -expected - 1.0).abs() < 1e-6);
        assert!((code.codes[1] * delta - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn posterior_mean_matches_response_space_smoother_for_wide_support() {
        use gam_linalg::faer_ndarray::FaerCholesky;
        let row = array![1.2_f32, -0.7];
        let decoder = array![[1.0_f32, 0.0], [0.6, 0.8], [0.0, 1.0]];
        let active = [(0, 0.0), (1, 0.0), (2, 0.0)];
        for rho in [0.01_f32, 0.5, 10.0] {
            let code = solve_row_codes(row.view(), decoder.view(), &active, 3, rho);
            let basis = decoder.mapv(f64::from).reversed_axes();
            let covariance = basis.dot(&basis.t());
            let mut total = covariance.clone();
            total.diag_mut().mapv_inplace(|value| value + rho as f64);
            let response = total
                .cholesky(faer::Side::Lower)
                .unwrap()
                .solvevec(&row.mapv(f64::from));
            let expected = basis.t().dot(&response);
            for (&actual, &expected) in code.codes.iter().zip(expected.iter()) {
                assert!((actual as f64 - expected).abs() < 1e-6);
            }
        }
    }

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
}
