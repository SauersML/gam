//! Pareto-smoothed importance-sampling utilities.
//!
//! The implementation is intentionally self-contained: it estimates the
//! generalized-Pareto tail shape `k` from the largest positive weights and
//! replaces only that empirical tail by monotone GPD expected quantiles.  The
//! returned `k_hat` has the usual GPD tail interpretation: values near zero
//! indicate light tails, `k > 0.5` indicates that the fitted tail has infinite
//! variance, and larger values mark increasingly unstable upper tails. Consumers
//! decide whether that tail is a draw-wise PSIS reliability diagnostic or another
//! influence diagnostic based on what the supplied weights represent.
//!
//! The shape is recovered with the Zhang–Stephens (2009) empirical-Bayes
//! profile estimator — the same GPD tail fit used by `loo`/ArviZ for draw-wise
//! PSIS diagnostics.
//! Crucially it is consistent across the entire `k ∈ (−∞, ∞)` range, including
//! the dangerous `k ≥ 0.5` regime where the GPD variance is infinite and the
//! older method-of-moments form `k = ½(1 − μ²/Var)` is structurally capped
//! below `0.5` and so cannot fire a heavy-tail gate.

#[derive(Debug, Clone)]
pub struct PsisResult {
    pub smoothed: Vec<f64>,
    pub k_hat: f64,
    pub tail_start: usize,
    pub tail_count: usize,
}

pub(crate) const MIN_TAIL_COUNT: usize = 5;
const MAX_TAIL_FRACTION: f64 = 0.2;

/// Pareto-smooth a non-negative weight vector and report the fitted GPD tail
/// shape.  Non-tail observations are left bit-identical; only the largest tail
/// observations are replaced by sorted GPD expected quantiles and then clipped
/// to be non-decreasing in the original sorted order.
pub fn pareto_smooth_weights(weights: &[f64]) -> Option<PsisResult> {
    if weights.len() < MIN_TAIL_COUNT * 2 || weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
        return None;
    }
    let mut indexed: Vec<(usize, f64)> = weights.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1));
    let n = indexed.len();
    let tail_count = ((n as f64).sqrt().ceil() as usize)
        .max(MIN_TAIL_COUNT)
        .min(((MAX_TAIL_FRACTION * n as f64).ceil() as usize).max(MIN_TAIL_COUNT))
        .min(n - 1);
    let tail_start = n - tail_count;
    let threshold = indexed[tail_start - 1].1;
    let excesses: Vec<f64> = indexed[tail_start..]
        .iter()
        .map(|(_, w)| (w - threshold).max(0.0))
        .collect();
    let (k_hat, sigma_hat) = fit_gpd_moments(&excesses)?;
    let mut smoothed = weights.to_vec();
    let mut previous = threshold;
    for (rank, (original_idx, _)) in indexed[tail_start..].iter().enumerate() {
        let p = (rank as f64 + 0.5) / tail_count as f64;
        let q = threshold + gpd_quantile(p, k_hat, sigma_hat).max(0.0);
        let monotone = q.max(previous);
        smoothed[*original_idx] = monotone;
        previous = monotone;
    }
    Some(PsisResult {
        smoothed,
        k_hat,
        tail_start,
        tail_count,
    })
}

/// Fit a generalized-Pareto distribution `1 − (1 + k·x/σ)^(−1/k)` to positive
/// excesses with the Zhang–Stephens (2009) empirical-Bayes profile estimator.
///
/// The GPD is reparameterised by the single scalar `b = −k/σ`.  For a fixed
/// `b` the shape has the closed-form profile MLE
/// `k(b) = mean_i log(1 − b·xᵢ)`, and the profile log-likelihood (up to an
/// additive constant) is `ℓ(b) = n·(log(−b/k(b)) − k(b) − 1)`.  Zhang &
/// Stephens place a diffuse grid of candidate `b` values around `1/x_max`
/// (the boundary of the admissible region) plus a quartile-scaled spread, then
/// average `b` under the softmax of `ℓ` — an empirical-Bayes posterior mean
/// rather than a single maximiser, which is markedly more stable in small
/// samples.  The shape is read back off the posterior-mean `b` and shrunk
/// toward `0.5` by a weak `N(0.5, …)`-style prior worth `PRIOR_K`
/// pseudo-observations (negligible once `n ≫ PRIOR_K`).
///
/// Unlike the method-of-moments identity `k = ½(1 − μ²/Var)` — which is bounded
/// above by `0.5` for every real sample and so can never report the heavy tails
/// (`k > 0.7`) the diagnostic exists to flag — this estimator is consistent for
/// `k` up to and beyond `1`, recovering the true shape to a couple percent from
/// a large exact sample.
///
/// Returns `(k_hat, sigma_hat)`, or `None` when there are too few positive
/// excesses or the data are degenerate (all equal / non-finite).
pub fn fit_gpd_moments(excesses: &[f64]) -> Option<(f64, f64)> {
    let mut x: Vec<f64> = excesses
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .collect();
    if x.len() < MIN_TAIL_COUNT {
        return None;
    }
    x.sort_by(f64::total_cmp);
    let n = x.len();
    let nf = n as f64;

    // Admissible region is `b < 1/x_max`; the smallest order statistic sets the
    // quartile scale (Zhang & Stephens use the lower-quartile observation).
    let x_max = x[n - 1];
    let q_idx = ((nf / 4.0 + 0.5).floor() as usize)
        .saturating_sub(1)
        .min(n - 1);
    let x_star = x[q_idx];
    if !(x_max.is_finite() && x_max > 0.0 && x_star.is_finite() && x_star > 0.0) {
        return None;
    }

    // Weakly-informative prior: `PRIOR_BS` controls grid spread, `PRIOR_K`
    // pseudo-observations shrink the final shape toward 0.5 (Zhang–Stephens / loo).
    const PRIOR_BS: f64 = 3.0;
    const PRIOR_K: f64 = 10.0;
    let m_est = 30 + (nf.sqrt() as usize);

    // Profile log-likelihood `ℓ(b)` on the candidate grid (the `n·` factor and
    // additive constants are retained so the softmax weights match loo/ArviZ).
    let mut b_grid = Vec::with_capacity(m_est);
    let mut len_scale = Vec::with_capacity(m_est);
    let mut max_ls = f64::NEG_INFINITY;
    for j in 1..=m_est {
        let b =
            (1.0 - (m_est as f64 / (j as f64 - 0.5)).sqrt()) / (PRIOR_BS * x_star) + 1.0 / x_max;
        let k = profile_shape(b, &x);
        let arg = k.map(|k| -(b / k));
        let ls = if let (Some(k), Some(arg)) = (k, arg) {
            if arg.is_finite() && arg > 0.0 {
                nf * (arg.ln() - k - 1.0)
            } else {
                f64::NEG_INFINITY
            }
        } else {
            f64::NEG_INFINITY
        };
        if ls > max_ls {
            max_ls = ls;
        }
        b_grid.push(b);
        len_scale.push(ls);
    }
    if !max_ls.is_finite() {
        return None;
    }

    // Posterior mean of `b` under the (numerically stable) softmax of `ℓ`.
    let mut weight_sum = 0.0;
    let mut b_post = 0.0;
    for (&b, &ls) in b_grid.iter().zip(len_scale.iter()) {
        let w = if ls.is_finite() {
            (ls - max_ls).exp()
        } else {
            0.0
        };
        weight_sum += w;
        b_post += w * b;
    }
    if !(weight_sum.is_finite() && weight_sum > 0.0) {
        return None;
    }
    b_post /= weight_sum;
    if !b_post.is_finite() || b_post == 0.0 {
        return None;
    }

    let k_raw = profile_shape(b_post, &x)?;
    // Shrink toward 0.5 by `PRIOR_K` pseudo-observations.
    let k = (nf * k_raw + PRIOR_K * 0.5) / (nf + PRIOR_K);
    let sigma = -k / b_post;
    if !(k.is_finite() && sigma.is_finite() && sigma > 0.0) {
        return None;
    }
    Some((k, sigma))
}

/// Closed-form profile shape `k(b) = mean_i log(1 − b·xᵢ)` for fixed `b`.
///
/// A candidate is admissible only when every `1 - b*x_i` is finite and strictly
/// positive. Boundary or out-of-domain candidates are rejected before any log is
/// evaluated, so they cannot leak NaN/Inf through the profile likelihood.
#[inline]
fn profile_shape(b: f64, x: &[f64]) -> Option<f64> {
    if !b.is_finite() || x.is_empty() {
        return None;
    }
    let mut acc = 0.0_f64;
    for &xi in x {
        let arg = 1.0 - b * xi;
        if !(arg.is_finite() && arg > 0.0) {
            return None;
        }
        acc += arg.ln();
    }
    Some(acc / x.len() as f64)
}

#[inline]
fn gpd_quantile(p: f64, k: f64, sigma: f64) -> f64 {
    let survival = (1.0 - p).clamp(1e-12, 1.0);
    if k.abs() < 1e-8 {
        -sigma * survival.ln()
    } else {
        sigma * (survival.powf(-k) - 1.0) / k
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpd_sample(u: f64, k: f64, sigma: f64) -> f64 {
        if k.abs() < 1e-12 {
            -sigma * (1.0 - u).ln()
        } else {
            sigma * ((1.0 - u).powf(-k) - 1.0) / k
        }
    }

    #[test]
    fn psis_k_hat_recovers_known_generalized_pareto_tail() {
        let k_true = 0.35_f64;
        let sigma = 1.7_f64;
        let mut xs = Vec::new();
        for i in 1..=10_000 {
            let u = (i as f64 - 0.5) / 10_000.0;
            xs.push(gpd_sample(u, k_true, sigma));
        }
        let (k_hat, sigma_hat) = fit_gpd_moments(&xs).expect("GPD fit should succeed");
        assert!(
            (k_hat - k_true).abs() < 0.03,
            "k_hat={k_hat}, k_true={k_true}"
        );
        assert!(
            (sigma_hat - sigma).abs() < 0.08,
            "sigma_hat={sigma_hat}, sigma={sigma}"
        );
    }

    #[test]
    fn pareto_smoothing_preserves_nontail_and_reports_heavy_tail() {
        // A flat block of baseline weights plus a genuine GPD(k=0.7) tail. The
        // tail is heavy enough to sit in the infinite-variance regime, so a
        // consistent estimator must report `k_hat > 0.5` for it.
        let mut w = vec![1.0; 200];
        for i in 1..=120 {
            let u = (i as f64 - 0.5) / 120.0;
            w.push(1.0 + gpd_sample(u, 0.7, 0.5));
        }
        let out = pareto_smooth_weights(&w).expect("PSIS should fit a positive tail");
        assert_eq!(out.smoothed[0], 1.0);
        assert!(
            out.k_hat > 0.5,
            "genuine GPD(k=0.7) tail should be flagged as heavy (infinite variance); got k_hat={}",
            out.k_hat
        );
    }

    /// Regression for #585 from the *shape-recovery* angle: the moment estimator
    /// `k = ½(1 − μ²/Var)` is structurally `≤ 0.5`, so heavy tails collapsed
    /// onto an indistinguishable ~0.5. The Zhang–Stephens estimator must instead
    /// recover the true shape across the dangerous range — and the fitted shapes
    /// must stay *strictly ordered*, since a diagnostic that cannot separate
    /// `k=0.7` from `k=1.0` is useless.
    #[test]
    fn psis_k_hat_tracks_and_orders_heavy_tails() {
        let recover = |k_true: f64| -> f64 {
            let xs: Vec<f64> = (1..=100_000)
                .map(|i| gpd_sample((i as f64 - 0.5) / 100_000.0, k_true, 1.0))
                .collect();
            fit_gpd_moments(&xs).expect("GPD fit should succeed").0
        };
        let mut last = f64::NEG_INFINITY;
        for &k_true in &[0.5_f64, 0.7, 0.85, 1.0, 1.2] {
            let k_hat = recover(k_true);
            assert!(
                (k_hat - k_true).abs() < 0.05,
                "k_true={k_true}: fitted k_hat={k_hat} not within 0.05"
            );
            assert!(
                k_hat > last,
                "k_hat must increase with the true shape: {k_hat} !> {last}"
            );
            last = k_hat;
        }
    }

    /// The estimator must degrade gracefully: too few positive excesses and
    /// fully degenerate (all-equal) inputs return `None` rather than NaN/`0.5`.
    #[test]
    fn psis_gpd_fit_handles_degenerate_inputs() {
        assert!(
            fit_gpd_moments(&[1.0, 2.0, 3.0]).is_none(),
            "fewer than MIN_TAIL_COUNT"
        );
        assert!(
            fit_gpd_moments(&[0.0, -1.0, f64::NAN, 0.0]).is_none(),
            "no positive finite excesses"
        );
        // All-equal positive excesses are degenerate (x_max == x_star). The fit
        // must not panic or emit NaN, and — most importantly — must never report
        // a spuriously *heavy* tail; a near-constant block is the lightest tail
        // there is, so any returned shape must sit well below the 0.5 gate.
        if let Some((k_hat, sigma_hat)) = fit_gpd_moments(&[2.0; 50]) {
            assert!(k_hat.is_finite() && sigma_hat.is_finite() && sigma_hat > 0.0);
            assert!(
                k_hat < 0.5,
                "degenerate equal excesses must not be flagged heavy; got k_hat={k_hat}"
            );
        }
    }

    #[test]
    fn psis_profile_shape_rejects_inadmissible_candidates() {
        let x = [0.25, 1.0, 2.0];
        assert!(
            profile_shape(0.49, &x).is_some(),
            "b below 1/x_max is admissible for all excesses"
        );
        assert!(
            profile_shape(0.5, &x).is_none(),
            "b at 1/x_max puts the largest excess on the log boundary"
        );
        assert!(
            profile_shape(0.75, &x).is_none(),
            "b above 1/x_max makes at least one log argument negative"
        );
    }
}
