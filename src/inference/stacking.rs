//! Stacking of predictive distributions — robust model averaging instead of
//! winner-take-all selection.
//!
//! Topology / model selection that returns a single winner is brittle when the
//! evidence gap between candidates is small (sphere-vs-torus-vs-flat manifold
//! ambiguity, partial periodic coverage, near-tied smooth structures). Stacking
//! keeps every candidate and assigns predictive weights `w` on the simplex that
//! maximize the leave-one-out log predictive score of the mixture
//!
//! ```text
//! maximize_w   (1/n) Σ_i log( Σ_T w_T p_{T,i} ),   w_T ≥ 0, Σ_T w_T = 1,
//! ```
//!
//! where `p_{T,i}` is candidate `T`'s (held-out / LOO / ALO) predictive density
//! at point `i`. This is the stacking-of-predictive-distributions objective of
//! Yao, Vehtari, Simpson & Gelman (*Bayesian Analysis* 2018); it is concave in
//! `w` (log of a linear function) and provably does at least as well in
//! expected log score as the best single candidate, because the single models
//! are the vertices of the simplex over which we optimize.
//!
//! The objective is maximized by the standard EM / multiplicative update
//!
//! ```text
//! w_T ← w_T · (1/n) Σ_i p_{T,i} / ( Σ_S w_S p_{S,i} ),
//! ```
//!
//! which is a monotone ascent that stays on the simplex automatically.

use ndarray::{Array1, ArrayView1, ArrayView2};

use crate::estimate::EstimationError;

fn validate_density(pred_density: ArrayView2<'_, f64>) -> Result<(usize, usize), EstimationError> {
    let n = pred_density.nrows();
    let t = pred_density.ncols();
    if n == 0 || t == 0 {
        return Err(EstimationError::InvalidInput(
            "stacking requires a non-empty (points × candidates) predictive-density matrix"
                .to_string(),
        ));
    }
    for i in 0..n {
        let mut row_sum = 0.0;
        for c in 0..t {
            let v = pred_density[[i, c]];
            if !(v.is_finite() && v >= 0.0) {
                return Err(EstimationError::InvalidInput(format!(
                    "stacking: predictive density[{i},{c}] must be finite and non-negative; got {v}"
                )));
            }
            row_sum += v;
        }
        if row_sum <= 0.0 {
            return Err(EstimationError::InvalidInput(format!(
                "stacking: point {i} has zero predictive density under every candidate"
            )));
        }
    }
    Ok((n, t))
}

/// Mean log predictive score `(1/n) Σ_i log(Σ_T w_T p_{T,i})` of the mixture
/// with weights `w` over the predictive densities `pred_density` (`n×T`).
pub fn mean_log_score(pred_density: ArrayView2<'_, f64>, weights: ArrayView1<'_, f64>) -> f64 {
    let n = pred_density.nrows();
    let t = pred_density.ncols();
    let mut acc = 0.0;
    for i in 0..n {
        let mut mix = 0.0;
        for c in 0..t {
            mix += weights[c] * pred_density[[i, c]];
        }
        acc += mix.max(f64::MIN_POSITIVE).ln();
    }
    acc / n as f64
}

/// Stacking weights: the simplex point maximizing [`mean_log_score`] over the
/// candidate predictive densities `pred_density` (`n_points × n_candidates`).
///
/// Returns weights that are non-negative and sum to one. The EM update is run
/// for up to `max_iter` iterations or until the maximum weight change drops
/// below `tol`.
pub fn stacking_weights(
    pred_density: ArrayView2<'_, f64>,
    max_iter: usize,
    tol: f64,
) -> Result<Array1<f64>, EstimationError> {
    let (n, t) = validate_density(pred_density)?;
    let mut w = Array1::<f64>::from_elem(t, 1.0 / t as f64);
    for _ in 0..max_iter {
        // mixture density per point under the current weights
        let mut numer = Array1::<f64>::zeros(t);
        for i in 0..n {
            let mut mix = 0.0;
            for c in 0..t {
                mix += w[c] * pred_density[[i, c]];
            }
            let inv = 1.0 / mix.max(f64::MIN_POSITIVE);
            for c in 0..t {
                numer[c] += pred_density[[i, c]] * inv;
            }
        }
        let mut max_delta = 0.0_f64;
        for c in 0..t {
            let updated = w[c] * numer[c] / n as f64;
            max_delta = max_delta.max((updated - w[c]).abs());
            w[c] = updated;
        }
        // renormalize defensively against floating-point drift
        let total: f64 = w.sum();
        if total > 0.0 {
            w.mapv_inplace(|v| v / total);
        }
        if max_delta < tol {
            break;
        }
    }
    Ok(w)
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};

    use super::*;

    #[test]
    fn concentrates_on_a_dominant_candidate() {
        // Candidate 0 assigns higher density everywhere ⇒ weight → ~1 on it.
        let pred = Array2::from_shape_vec(
            (5, 2),
            vec![0.9, 0.1, 0.8, 0.2, 0.95, 0.05, 0.7, 0.3, 0.85, 0.15],
        )
        .unwrap();
        let w = stacking_weights(pred.view(), 1000, 1e-12).unwrap();
        assert!(w[0] > 0.95, "expected dominance on candidate 0, got {w:?}");
        assert!((w.sum() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn beats_or_matches_the_best_single_candidate() {
        // Each candidate is good on a different half of the points, so the
        // stacked mixture must strictly beat either single model.
        let pred = Array2::from_shape_vec(
            (6, 2),
            vec![
                0.9, 0.1, //
                0.8, 0.2, //
                0.85, 0.15, //
                0.1, 0.9, //
                0.2, 0.8, //
                0.15, 0.85,
            ],
        )
        .unwrap();
        let w = stacking_weights(pred.view(), 2000, 1e-12).unwrap();
        let stacked = mean_log_score(pred.view(), w.view());
        let only0 = mean_log_score(pred.view(), Array1::from(vec![1.0, 0.0]).view());
        let only1 = mean_log_score(pred.view(), Array1::from(vec![0.0, 1.0]).view());
        assert!(
            stacked >= only0.max(only1) - 1e-9,
            "stacked {stacked} < best single {}",
            only0.max(only1)
        );
        // genuine improvement (both candidates only half-good)
        assert!(stacked > only0.max(only1) + 1e-3);
        // both candidates get real weight
        assert!(w[0] > 0.2 && w[1] > 0.2, "weights {w:?}");
    }

    #[test]
    fn weights_live_on_the_simplex() {
        let pred = Array2::from_shape_vec(
            (4, 3),
            vec![0.5, 0.3, 0.2, 0.4, 0.4, 0.2, 0.6, 0.1, 0.3, 0.2, 0.5, 0.3],
        )
        .unwrap();
        let w = stacking_weights(pred.view(), 500, 1e-12).unwrap();
        assert!((w.sum() - 1.0).abs() < 1e-10);
        assert!(w.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn rejects_degenerate_inputs() {
        let empty = Array2::<f64>::zeros((0, 2));
        assert!(stacking_weights(empty.view(), 10, 1e-9).is_err());
        // a point with zero density under every candidate
        let zero_row = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 0.5, 0.5]).unwrap();
        assert!(stacking_weights(zero_row.view(), 10, 1e-9).is_err());
        let negative = Array2::from_shape_vec((2, 2), vec![0.5, -0.1, 0.5, 0.5]).unwrap();
        assert!(stacking_weights(negative.view(), 10, 1e-9).is_err());
    }
}
