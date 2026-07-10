//! Model-currency fidelity metrics (scorecard axis 1).
//!
//! A featurizer's held-out EV prices reconstruction error in Euclidean distance,
//! but the model reads directions unequally (rogue dims: huge variance, ~0
//! leverage). These reductions price reconstruction in the MODEL's own currency,
//! by comparing losses (and output distributions) when the harvest-layer
//! activation is run clean, replaced by the reconstruction, or replaced by the
//! information-free mean-ablation baseline. They are the reusable numeric core
//! behind the Python fidelity harness (patching itself lives at the torch/HF
//! boundary); the harness's pure-Python fallback is pinned to these functions to
//! double precision by `tests/metrics/test_fidelity_metrics_parity.py`.
//!
//! Definitions (all in the model's loss units / nats):
//!   loss_recovered = (L_ablate − L_recon) / (L_ablate − L_clean)
//!     1 = reconstruction recovers the clean loss; 0 = no better than ablation.
//!   r2_score       = 1 − Σ(clean − approx)² / Σ(clean − colmean(clean))²
//!     Euclidean explained variance about the per-column mean of `clean`.
//!   kl_categorical_rows = mean_row Σ_v p·(logp_clean − logp_other), p = exp(logp_clean)
//!     mean over rows of KL(clean ‖ other) from natural-log probabilities (nats).
//!   distortion_floor_r2 = the highest quantisation-R² at which loss_recovered
//!     has fallen more than `tol_frac` below the finest-precision plateau — the
//!     precision floor below which model-usable fidelity starts to degrade, and
//!     above which extra precision is wasted (read all metrics AT the floor).

/// (L_ablate − L_recon) / (L_ablate − L_clean).
///
/// Returns NaN when the clean↔ablate loss gap is degenerate (|denom| < `EPS`),
/// so a flat baseline surfaces as not-a-number rather than a divide blow-up.
pub fn loss_recovered(l_clean: f64, l_recon: f64, l_ablate: f64) -> f64 {
    const EPS: f64 = 1e-12;
    let denom = l_ablate - l_clean;
    if denom.abs() < EPS {
        return f64::NAN;
    }
    (l_ablate - l_recon) / denom
}

/// Euclidean R² of `approx` against `clean`, both row-major `(n_rows, n_cols)`.
///
/// TSS is taken about the per-column mean of `clean` (the same convention the
/// harness's `r2_against` uses). Returns 0.0 when TSS is non-positive (a constant
/// `clean`), matching the reference's guard.
pub fn r2_score(clean: &[f64], approx: &[f64], n_rows: usize, n_cols: usize) -> f64 {
    assert_eq!(clean.len(), n_rows * n_cols, "clean shape mismatch");
    assert_eq!(approx.len(), n_rows * n_cols, "approx shape mismatch");
    if n_rows == 0 || n_cols == 0 {
        return 0.0;
    }
    let mut col_mean = vec![0.0_f64; n_cols];
    for row in 0..n_rows {
        let base = row * n_cols;
        for col in 0..n_cols {
            col_mean[col] += clean[base + col];
        }
    }
    let inv_rows = 1.0 / n_rows as f64;
    for value in col_mean.iter_mut() {
        *value *= inv_rows;
    }
    let mut rss = 0.0_f64;
    let mut tss = 0.0_f64;
    for row in 0..n_rows {
        let base = row * n_cols;
        for col in 0..n_cols {
            let c = clean[base + col];
            let residual = c - approx[base + col];
            rss += residual * residual;
            let centered = c - col_mean[col];
            tss += centered * centered;
        }
    }
    if tss > 0.0 { 1.0 - rss / tss } else { 0.0 }
}

/// Mean over rows of KL(clean ‖ other) from natural-log probability rows.
///
/// Each of `clean_logprobs`/`other_logprobs` is row-major `(n_rows, n_cols)` of
/// log-probabilities (natural log). Per row: Σ_v exp(logp_clean)·(logp_clean −
/// logp_other); averaged over rows. Units are nats. `n_cols` is the vocabulary
/// width.
pub fn kl_categorical_rows(
    clean_logprobs: &[f64],
    other_logprobs: &[f64],
    n_rows: usize,
    n_cols: usize,
) -> f64 {
    assert_eq!(
        clean_logprobs.len(),
        n_rows * n_cols,
        "clean logprobs shape mismatch"
    );
    assert_eq!(
        other_logprobs.len(),
        n_rows * n_cols,
        "other logprobs shape mismatch"
    );
    if n_rows == 0 {
        return 0.0;
    }
    let mut total = 0.0_f64;
    for row in 0..n_rows {
        let base = row * n_cols;
        let mut row_kl = 0.0_f64;
        for col in 0..n_cols {
            let logp_clean = clean_logprobs[base + col];
            let p = logp_clean.exp();
            row_kl += p * (logp_clean - other_logprobs[base + col]);
        }
        total += row_kl;
    }
    total / n_rows as f64
}

/// The distortion-floor R²: the highest-R² point at which `loss_recovered` has
/// dropped more than `tol_frac` (relative) below the finest-precision plateau.
///
/// `r2s` and `loss_recovered` are parallel arrays over quantisation levels (one
/// entry per level; order irrelevant — the routine sorts by R² descending). The
/// plateau is the loss_recovered at the highest R² (finest precision); the floor
/// is the first (in descending-R² order) point that falls below `plateau −
/// tol_frac·|plateau|`. If nothing drops, the coarsest R² is returned. `None`
/// only for an empty sweep.
pub fn distortion_floor_r2(r2s: &[f64], loss_recovered: &[f64], tol_frac: f64) -> Option<f64> {
    assert_eq!(
        r2s.len(),
        loss_recovered.len(),
        "parallel arrays length mismatch"
    );
    if r2s.is_empty() {
        return None;
    }
    let mut order: Vec<usize> = (0..r2s.len()).collect();
    // Descending by R² (finest precision first). Total order via partial_cmp with
    // a stable tie-break on index so the plateau pick is deterministic.
    order.sort_by(|&a, &b| {
        r2s[b]
            .partial_cmp(&r2s[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let plateau = loss_recovered[order[0]];
    let threshold = plateau - tol_frac * plateau.abs();
    for &idx in &order {
        if loss_recovered[idx] < threshold {
            return Some(r2s[idx]);
        }
    }
    Some(r2s[order[order.len() - 1]])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loss_recovered_endpoints_and_degenerate() {
        // recon == clean -> full recovery = 1; recon == ablate -> 0.
        assert!((loss_recovered(1.0, 1.0, 3.0) - 1.0).abs() < 1e-15);
        assert!((loss_recovered(1.0, 3.0, 3.0) - 0.0).abs() < 1e-15);
        // halfway
        assert!((loss_recovered(1.0, 2.0, 3.0) - 0.5).abs() < 1e-15);
        // degenerate gap -> NaN
        assert!(loss_recovered(2.0, 2.0, 2.0).is_nan());
    }

    #[test]
    fn r2_perfect_and_mean_baselines() {
        let clean = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // perfect reconstruction -> 1
        assert!((r2_score(&clean, &clean, 3, 2) - 1.0).abs() < 1e-15);
        // predict the per-column mean everywhere -> R² = 0
        // col means: col0 mean of {1,3,5}=3, col1 mean of {2,4,6}=4
        let mean_pred = [3.0, 4.0, 3.0, 4.0, 3.0, 4.0];
        assert!(r2_score(&clean, &mean_pred, 3, 2).abs() < 1e-15);
    }

    #[test]
    fn r2_constant_clean_is_zero() {
        let clean = [7.0, 7.0, 7.0, 7.0];
        let approx = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(r2_score(&clean, &approx, 2, 2), 0.0);
    }

    #[test]
    fn kl_identical_is_zero_and_positive_otherwise() {
        // two rows, 3-way categorical, natural-log probs
        let p = [0.5_f64, 0.25, 0.25, 0.1, 0.6, 0.3];
        let logp: Vec<f64> = p.iter().map(|v| v.ln()).collect();
        assert!(kl_categorical_rows(&logp, &logp, 2, 3).abs() < 1e-15);
        // uniform "other" -> KL = mean row (Σ p logp + log 3) > 0
        let uniform = vec![(1.0_f64 / 3.0).ln(); 6];
        let kl = kl_categorical_rows(&logp, &uniform, 2, 3);
        // closed form: mean over rows of (-H(p) + log 3)
        let mut expect = 0.0;
        for row in 0..2 {
            let mut h = 0.0;
            for col in 0..3 {
                let pv = p[row * 3 + col];
                h += pv * pv.ln();
            }
            expect += h + (3.0_f64).ln();
        }
        expect /= 2.0;
        assert!((kl - expect).abs() < 1e-15);
    }

    #[test]
    fn floor_flat_plateau_returns_coarsest() {
        // loss_recovered flat across all precisions -> nothing drops -> coarsest R².
        let r2s = [0.99, 0.90, 0.50];
        let lr = [1.0, 1.0, 1.0];
        assert_eq!(distortion_floor_r2(&r2s, &lr, 0.05), Some(0.50));
    }

    #[test]
    fn floor_detects_drop_point() {
        // finest plateau lr=1.0; coarse point drops to 0.8 (>5% below) at R²=0.60.
        let r2s = [0.99, 0.95, 0.60, 0.30];
        let lr = [1.0, 0.99, 0.80, 0.40];
        // descending R²: 0.99(1.0 plateau), 0.95(0.99 ok), 0.60(0.80 < 0.95) -> floor 0.60
        assert_eq!(distortion_floor_r2(&r2s, &lr, 0.05), Some(0.60));
    }

    #[test]
    fn floor_empty_is_none() {
        assert_eq!(distortion_floor_r2(&[], &[], 0.05), None);
    }
}
