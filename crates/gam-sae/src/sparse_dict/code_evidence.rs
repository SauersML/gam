//! Exact effective degrees of freedom of the posterior-mean row codes.
//!
//! Conditional on a row's selected directions B (features × support), its
//! Gaussian random coefficients have smoother H=B(BᵀB+rho I)⁻¹Bᵀ. Therefore
//! tr(H)=sum sigma_j²/(sigma_j²+rho), using the singular values of B. The full
//! design is block diagonal over rows, not the global decoder normal matrix.
//! This also defines a prior on the reconstructed function: the minimum-norm
//! coefficients have squared norm fᵀ(BBᵀ)⁺f on the range of B.

use gam_linalg::faer_ndarray::FaerSvd;
use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;
use std::collections::BTreeMap;

pub(super) fn effective_dof(
    decoder: ArrayView2<'_, f32>,
    indices: ArrayView2<'_, u32>,
    rho: f64,
) -> Result<f64, String> {
    if !(rho.is_finite() && rho > 0.0) {
        return Err("row-code evidence requires a finite positive variance ratio".into());
    }
    if indices.iter().any(|&atom| atom as usize >= decoder.nrows()) {
        return Err("row-code evidence support index is out of range".into());
    }
    if indices.ncols() == 1 {
        let norms: Vec<f64> = decoder
            .outer_iter()
            .map(|row| row.iter().map(|&v| (v as f64).powi(2)).sum())
            .collect();
        return Ok(indices
            .column(0)
            .iter()
            .map(|&atom| {
                let norm = norms[atom as usize];
                norm / (norm + rho)
            })
            .sum());
    }
    // Padding repeats an index with zero code; it does not add another random
    // coefficient. Distinct atoms with identical directions remain distinct
    // prior contributions and are retained. Cache only occupied supports: at
    // most O(N*active) indices, the size of the fit's existing sparse routing.
    let mut supports = BTreeMap::<Vec<u32>, usize>::new();
    for row in indices.outer_iter() {
        let mut support = row.to_vec();
        support.sort_unstable();
        support.dedup();
        *supports.entry(support).or_default() += 1;
    }
    let supports: Vec<_> = supports.into_iter().collect();
    let contributions: Result<Vec<f64>, String> = supports
        .into_par_iter()
        .map(|(support, count)| {
            let basis =
                Array2::from_shape_fn((decoder.ncols(), support.len()), |(feature, slot)| {
                    decoder[[support[slot] as usize, feature]] as f64
                });
            // Work on B itself: an eigensolve of BᵀB destroys a small but resolved
            // singular direction when squaring its condition number.
            let (_, singular, _) = basis
                .svd(false, false)
                .map_err(|error| format!("row-code evidence SVD failed: {error}"))?;
            if singular
                .iter()
                .any(|&value| !value.is_finite() || value < 0.0)
            {
                return Err("row-code evidence has an invalid singular spectrum".into());
            }
            let trace: f64 = singular
                .iter()
                .map(|&value| {
                    let energy = value * value;
                    energy / (energy + rho)
                })
                .sum();
            Ok(trace * count as f64)
        })
        .collect();
    // Sorted support keys and a serial final reduction make this independent
    // of worker count. No Monte Carlo probes or artificial residual DOF floor.
    Ok(contributions?.into_iter().sum())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn support_padding_and_distinct_collinear_atoms_have_different_prior_variance() {
        let decoder = array![[1.0_f32, 0.0], [1.0, 0.0]];
        let padding = array![[0_u32, 0], [1, 1]];
        let separate = array![[0_u32, 1], [1, 0]];
        assert!((effective_dof(decoder.view(), padding.view(), 1.0).unwrap() - 1.0).abs() < 1e-14);
        assert!(
            (effective_dof(decoder.view(), separate.view(), 1.0).unwrap() - 4.0 / 3.0).abs()
                < 1e-14
        );
    }

    #[test]
    fn near_dependent_support_retains_its_resolved_small_direction() {
        let delta = 1e-10_f32;
        let decoder = array![[1.0_f32, 0.0], [1.0, delta]];
        let indices = array![[0_u32, 1]];
        let rho = (delta as f64).powi(2);
        // The small Gram eigenvalue is delta²/2 to O(delta⁴), so it contributes
        // 1/3 at this rho. Forming the Gram rounds its diagonal to (1,1) and
        // loses that entire contribution, despite the resolved input direction.
        let trace = effective_dof(decoder.view(), indices.view(), rho).unwrap();
        assert!((trace - 4.0 / 3.0).abs() < 1e-12, "{trace}");
    }

    #[test]
    fn exact_support_trace_matches_the_response_space_smoother() {
        use gam_linalg::faer_ndarray::FaerCholesky;
        let decoder = array![[1.0_f32, 0.0, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]];
        let indices = array![[0_u32, 1], [1, 0], [1, 2], [2, 2]];
        for rho in [0.01_f64, 0.5, 10.0] {
            let mut expected = 0.0;
            for row in indices.outer_iter() {
                let mut support = row.to_vec();
                support.sort_unstable();
                support.dedup();
                let b = Array2::from_shape_fn((3, support.len()), |(i, j)| {
                    decoder[[support[j] as usize, i]] as f64
                });
                let covariance = b.dot(&b.t());
                let mut total = covariance.clone();
                for i in 0..3 {
                    total[[i, i]] += rho;
                }
                let smoother = total
                    .cholesky(faer::Side::Lower)
                    .unwrap()
                    .solve_mat(&covariance);
                expected += smoother.diag().sum();
            }
            let actual = effective_dof(decoder.view(), indices.view(), rho).unwrap();
            assert!(
                (actual - expected).abs() < 1e-12,
                "rho={rho}: {actual} vs {expected}"
            );
        }
    }
}
