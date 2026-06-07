//! Stacking of predictive distributions over retained candidate fits.
//!
//! Topology AUTO selection (and any candidate-enumeration consumer) retains one
//! fitted predictor per candidate. Winner-take-all discards every non-winning
//! predictor; the principled alternative is *stacking of predictive
//! distributions* (Yao, Vehtari, Simpson & Gelman, 2018, "Using Stacking to
//! Average Bayesian Predictive Distributions", Bayesian Analysis 13(3)): choose
//! mixture weights `w` on the simplex that maximise the held-out logarithmic
//! score of the stacked predictive density
//!
//! ```text
//!     w* = argmax_{w in S}  (1/n) Σ_i log( Σ_k w_k p_{ik} ),
//! ```
//!
//! where `p_{ik}` is candidate `k`'s **held-out** (leave-one-out / ALO or
//! genuine out-of-fold) predictive density of observation `i` and
//! `S = { w : w_k ≥ 0, Σ_k w_k = 1 }`. This is *not* pseudo-BMA softmax over
//! scalar evidence scores — it consumes a full per-point held-out density table,
//! exactly the consumer the topology selector's module doc-comment names as the
//! prerequisite for moving off winner-take-all.
//!
//! The objective is concave in `w` (log of an affine map), so its constrained
//! maximiser is unique up to ties between identical columns. We solve it with
//! the classic finite-mixture EM / multiplicative-weight fixed point
//!
//! ```text
//!     w_k ← w_k · (1/n) Σ_i p_{ik} / ( Σ_j w_j p_{ij} ),
//! ```
//!
//! which stays on the simplex by construction and is monotone-ascent on the
//! log-score (it is EM for a mixture with fixed components and unknown mixing
//! weights). Inputs arrive as log-densities `L_{ik} = log p_{ik}`; all per-row
//! work is done in log space with a per-row shift so that overflow/underflow of
//! the raw densities never enters the update.

use ndarray::{Array1, Array2, ArrayView2};

/// Convergence + iteration controls for the stacking-weight EM fixed point.
#[derive(Debug, Clone, Copy)]
pub struct StackingConfig {
    /// Maximum EM iterations before returning the current iterate.
    pub max_iter: usize,
    /// Stop once the max absolute weight change between iterations is below
    /// this threshold.
    pub weight_tol: f64,
}

impl Default for StackingConfig {
    fn default() -> Self {
        // The EM ascent is linearly convergent; a few hundred iterations with a
        // tight weight tolerance reaches a stable simplex point for the modest
        // candidate counts (≤ a handful of topologies) this serves, while the
        // tolerance guards against spinning on a flat ridge of tied columns.
        Self {
            max_iter: 1000,
            weight_tol: 1e-10,
        }
    }
}

/// Optimal stacking weights together with the achieved held-out mean log-score.
#[derive(Debug, Clone)]
pub struct StackingWeights {
    /// Simplex weights, one per candidate column, in input column order.
    pub weights: Array1<f64>,
    /// Mean held-out logarithmic score `(1/n) Σ_i log Σ_k w_k p_{ik}` at the
    /// returned weights — the stacking objective value (higher is better).
    pub mean_log_score: f64,
    /// EM iterations actually consumed.
    pub iterations: usize,
}

/// Solve the stacking-weight problem from a per-point held-out log-density table
/// `log_density[i, k] = log p_{ik}` (`n_obs` rows × `n_candidates` columns).
///
/// A column whose every entry is non-finite (a candidate with no usable
/// held-out density anywhere) is rejected up front: it can never contribute to
/// the mixture and would otherwise pin a weight at its initialisation. Rows
/// whose surviving entries are all non-finite carry no discriminating
/// information and are dropped from the objective. At least one finite entry
/// must remain.
pub fn solve_stacking_weights(
    log_density: ArrayView2<'_, f64>,
    config: StackingConfig,
) -> Result<StackingWeights, String> {
    let n_obs = log_density.nrows();
    let n_cand = log_density.ncols();
    if n_cand == 0 {
        return Err("stacking requires at least one candidate column".to_string());
    }
    if n_obs == 0 {
        return Err("stacking requires at least one held-out observation row".to_string());
    }

    // Keep only candidates that have a finite held-out density somewhere; map
    // surviving columns back to original indices so the returned weight vector
    // is aligned with the caller's candidate order (rejected columns get 0).
    let mut kept_cols = Vec::with_capacity(n_cand);
    for k in 0..n_cand {
        if (0..n_obs).any(|i| log_density[[i, k]].is_finite()) {
            kept_cols.push(k);
        }
    }
    if kept_cols.is_empty() {
        return Err("stacking found no candidate with any finite held-out density".to_string());
    }

    // Build the dense log-density table over surviving columns and the rows
    // that carry at least one finite surviving entry.
    let mut rows = Vec::with_capacity(n_obs);
    for i in 0..n_obs {
        if kept_cols.iter().any(|&k| log_density[[i, k]].is_finite()) {
            rows.push(i);
        }
    }
    if rows.is_empty() {
        return Err("stacking found no held-out row with a finite density".to_string());
    }

    let m = rows.len();
    let kept = kept_cols.len();
    let mut log_p = Array2::<f64>::from_elem((m, kept), f64::NEG_INFINITY);
    for (ri, &i) in rows.iter().enumerate() {
        for (ci, &k) in kept_cols.iter().enumerate() {
            let v = log_density[[i, k]];
            // A non-finite entry for a present candidate means "no density
            // here": treat it as zero probability (−∞ log-density) so it
            // contributes nothing to that row's mixture rather than poisoning it.
            log_p[[ri, ci]] = if v.is_finite() { v } else { f64::NEG_INFINITY };
        }
    }

    let mut weights = Array1::<f64>::from_elem(kept, 1.0 / kept as f64);
    let mut iterations = 0usize;
    for _ in 0..config.max_iter {
        iterations += 1;
        let (next, _) = em_step(log_p.view(), weights.view());
        let delta = next
            .iter()
            .zip(weights.iter())
            .fold(0.0_f64, |acc, (a, b)| acc.max((a - b).abs()));
        weights = next;
        if delta <= config.weight_tol {
            break;
        }
    }

    let mean_log_score = mean_log_score(log_p.view(), weights.view());

    // Scatter the surviving weights back into the full candidate ordering.
    let mut full = Array1::<f64>::zeros(n_cand);
    for (ci, &k) in kept_cols.iter().enumerate() {
        full[k] = weights[ci];
    }
    Ok(StackingWeights {
        weights: full,
        mean_log_score,
        iterations,
    })
}

/// One EM / multiplicative-weight update. Returns the next weight iterate and
/// the mean log-score at the *incoming* weights (so callers can monitor ascent
/// without a second pass).
fn em_step(log_p: ArrayView2<'_, f64>, w: ndarray::ArrayView1<'_, f64>) -> (Array1<f64>, f64) {
    let m = log_p.nrows();
    let kept = log_p.ncols();
    let log_w: Array1<f64> = w.mapv(|wk| if wk > 0.0 { wk.ln() } else { f64::NEG_INFINITY });
    let mut numer = Array1::<f64>::zeros(kept);
    let mut log_score_sum = 0.0_f64;
    for i in 0..m {
        // Responsibilities r_k ∝ w_k p_{ik}, computed in log space with a
        // per-row max shift for stability. log_mix is this row's log Σ w_k p_ik.
        let mut log_terms = Array1::<f64>::from_elem(kept, f64::NEG_INFINITY);
        let mut row_max = f64::NEG_INFINITY;
        for k in 0..kept {
            let t = log_w[k] + log_p[[i, k]];
            log_terms[k] = t;
            if t > row_max {
                row_max = t;
            }
        }
        if !row_max.is_finite() {
            // No component places any mass on this row under the current
            // weights; it cannot inform the update. Skip it.
            continue;
        }
        let mut row_sum = 0.0_f64;
        for k in 0..kept {
            row_sum += (log_terms[k] - row_max).exp();
        }
        let log_mix = row_max + row_sum.ln();
        log_score_sum += log_mix;
        for k in 0..kept {
            // r_{ik} = exp(log_terms - log_mix); accumulate Σ_i r_{ik}.
            numer[k] += (log_terms[k] - log_mix).exp();
        }
    }
    let total: f64 = numer.sum();
    let next = if total > 0.0 {
        numer.mapv(|v| v / total)
    } else {
        // Degenerate row set: keep the incoming weights rather than divide by
        // zero (the objective is flat here).
        w.to_owned()
    };
    let mean = if m > 0 {
        log_score_sum / m as f64
    } else {
        f64::NEG_INFINITY
    };
    (next, mean)
}

/// Mean held-out log-score `(1/m) Σ_i log Σ_k w_k p_{ik}` at fixed weights,
/// computed in log space.
fn mean_log_score(log_p: ArrayView2<'_, f64>, w: ndarray::ArrayView1<'_, f64>) -> f64 {
    let m = log_p.nrows();
    let kept = log_p.ncols();
    let log_w: Array1<f64> = w.mapv(|wk| if wk > 0.0 { wk.ln() } else { f64::NEG_INFINITY });
    let mut acc = 0.0_f64;
    let mut counted = 0usize;
    for i in 0..m {
        let mut row_max = f64::NEG_INFINITY;
        let mut terms = Array1::<f64>::from_elem(kept, f64::NEG_INFINITY);
        for k in 0..kept {
            let t = log_w[k] + log_p[[i, k]];
            terms[k] = t;
            if t > row_max {
                row_max = t;
            }
        }
        if !row_max.is_finite() {
            continue;
        }
        let mut row_sum = 0.0_f64;
        for k in 0..kept {
            row_sum += (terms[k] - row_max).exp();
        }
        acc += row_max + row_sum.ln();
        counted += 1;
    }
    if counted > 0 {
        acc / counted as f64
    } else {
        f64::NEG_INFINITY
    }
}

/// Combine per-candidate predictive means into one stacked predictive mean at a
/// set of new rows: `μ̄(x) = Σ_k w_k μ_k(x)`. `cand_means[k]` is candidate `k`'s
/// response-scale predictive mean over the same new rows in the same order; all
/// columns must share the row count. Weights align with `cand_means` order.
pub fn stacked_predictive_mean(
    weights: &Array1<f64>,
    cand_means: &[Array1<f64>],
) -> Result<Array1<f64>, String> {
    if cand_means.len() != weights.len() {
        return Err(format!(
            "stacked_predictive_mean: {} weights but {} candidate mean vectors",
            weights.len(),
            cand_means.len()
        ));
    }
    if cand_means.is_empty() {
        return Err("stacked_predictive_mean requires at least one candidate".to_string());
    }
    let n_rows = cand_means[0].len();
    if cand_means.iter().any(|m| m.len() != n_rows) {
        return Err("stacked_predictive_mean: candidate mean vectors disagree on row count".to_string());
    }
    let mut out = Array1::<f64>::zeros(n_rows);
    for (k, means) in cand_means.iter().enumerate() {
        let wk = weights[k];
        if wk == 0.0 {
            continue;
        }
        out.scaled_add(wk, means);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    fn gaussian_logpdf(y: f64, mean: f64, sd: f64) -> f64 {
        let z = (y - mean) / sd;
        -0.5 * (2.0 * std::f64::consts::PI).ln() - sd.ln() - 0.5 * z * z
    }

    #[test]
    fn single_candidate_gets_full_weight() {
        let log_density = Array2::from_shape_vec((3, 1), vec![-1.0, -2.0, -0.5]).unwrap();
        let out = solve_stacking_weights(log_density.view(), StackingConfig::default()).unwrap();
        assert!((out.weights[0] - 1.0).abs() < 1e-12);
        assert_eq!(out.weights.len(), 1);
    }

    #[test]
    fn dominant_candidate_attracts_nearly_all_weight() {
        // Candidate 0 assigns much higher held-out density everywhere.
        let mut log_density = Array2::<f64>::zeros((50, 2));
        for i in 0..50 {
            log_density[[i, 0]] = -0.1; // tight, high density
            log_density[[i, 1]] = -5.0; // diffuse, low density
        }
        let out = solve_stacking_weights(log_density.view(), StackingConfig::default()).unwrap();
        assert!(out.weights[0] > 0.99, "w0 = {}", out.weights[0]);
        assert!(out.weights[1] < 0.01, "w1 = {}", out.weights[1]);
    }

    #[test]
    fn complementary_candidates_share_weight() {
        // Each candidate is the better predictor on its own half of the data;
        // stacking should keep both with substantial weight, unlike pure
        // winner-take-all selection.
        let n = 40;
        let mut log_density = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            if i < n / 2 {
                log_density[[i, 0]] = gaussian_logpdf(0.0, 0.0, 0.5);
                log_density[[i, 1]] = gaussian_logpdf(0.0, 1.5, 0.5);
            } else {
                log_density[[i, 0]] = gaussian_logpdf(0.0, 1.5, 0.5);
                log_density[[i, 1]] = gaussian_logpdf(0.0, 0.0, 0.5);
            }
        }
        let out = solve_stacking_weights(log_density.view(), StackingConfig::default()).unwrap();
        assert!(out.weights[0] > 0.2 && out.weights[0] < 0.8, "w0 = {}", out.weights[0]);
        assert!((out.weights.sum() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn weights_stay_on_the_simplex() {
        let log_density = array![[-1.0, -2.0, -3.0], [-2.5, -1.0, -2.0], [-3.0, -2.0, -1.0]];
        let out = solve_stacking_weights(log_density.view(), StackingConfig::default()).unwrap();
        assert!((out.weights.sum() - 1.0).abs() < 1e-9);
        assert!(out.weights.iter().all(|&w| w >= -1e-12));
    }

    #[test]
    fn em_is_monotone_in_mean_log_score() {
        let log_density = array![[-0.2, -3.0], [-3.0, -0.2], [-0.5, -1.5], [-1.5, -0.5]];
        let mut w = Array1::from_elem(2, 0.5);
        let mut prev = mean_log_score(log_density.view(), w.view());
        for _ in 0..20 {
            let (next, _) = em_step(log_density.view(), w.view());
            w = next;
            let now = mean_log_score(log_density.view(), w.view());
            assert!(now >= prev - 1e-12, "log-score decreased: {prev} -> {now}");
            prev = now;
        }
    }

    #[test]
    fn dead_candidate_column_is_rejected_and_zero_weighted() {
        // Candidate 1 has no finite density anywhere.
        let log_density = array![
            [-1.0, f64::NEG_INFINITY],
            [-2.0, f64::NAN],
            [-0.5, f64::NEG_INFINITY]
        ];
        let out = solve_stacking_weights(log_density.view(), StackingConfig::default()).unwrap();
        assert_eq!(out.weights[1], 0.0);
        assert!((out.weights[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rows_with_no_finite_density_are_dropped() {
        let log_density = array![
            [-1.0, -2.0],
            [f64::NAN, f64::NEG_INFINITY], // dead row
            [-2.0, -1.0]
        ];
        let out = solve_stacking_weights(log_density.view(), StackingConfig::default()).unwrap();
        assert!((out.weights.sum() - 1.0).abs() < 1e-9);
        assert!(out.mean_log_score.is_finite());
    }

    #[test]
    fn all_dead_table_errors() {
        let log_density = Array2::from_elem((2, 2), f64::NEG_INFINITY);
        assert!(solve_stacking_weights(log_density.view(), StackingConfig::default()).is_err());
    }

    #[test]
    fn stacked_mean_is_weighted_combination() {
        let weights = array![0.25, 0.75];
        let means = vec![array![1.0, 2.0, 3.0], array![5.0, 6.0, 7.0]];
        let out = stacked_predictive_mean(&weights, &means).unwrap();
        assert!((out[0] - (0.25 * 1.0 + 0.75 * 5.0)).abs() < 1e-12);
        assert!((out[2] - (0.25 * 3.0 + 0.75 * 7.0)).abs() < 1e-12);
    }

    #[test]
    fn stacked_mean_rejects_shape_mismatch() {
        let weights = array![0.5, 0.5];
        let means = vec![array![1.0, 2.0], array![3.0]];
        assert!(stacked_predictive_mean(&weights, &means).is_err());
    }
}
