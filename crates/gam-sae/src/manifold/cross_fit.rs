//! K-fold cross-fitting for the SAE-manifold headline artifacts (explained
//! variance, coordinates/metrics, dose forecasts).
//!
//! # The post-selection optimism this removes
//!
//! The production fit **discovers structure** (which atoms are born, their
//! charts, the routed coordinates) and then **evaluates** the artifacts it
//! advertises — explained variance, the learned metric Θ, dose forecasts — on
//! the SAME rows the structure was discovered from. That double use of the data
//! makes every such artifact optimistic: a dictionary can always be bent to
//! explain the particular finite sample it was selected on, so in-sample EV
//! overstates the population EV, and the overstatement grows with the number of
//! degrees of freedom the search was allowed to spend.
//!
//! The size of that optimism is not hand-wavy. Fitting `q` freely-chosen linear
//! reconstruction directions to `n` rows of a **signal-free** target captures,
//! in expectation, a fraction `q / n` of the variance — the textbook null
//! coefficient of determination (this is exactly the `q/n` collapse floor the
//! outer wall already uses, see `SAE_FIT_DATA_COLLAPSE_EV_FLOOR`). So on pure
//! noise the naive in-sample EV sits near `q/n > 0`, not at the honest `0`.
//!
//! # Cross-fitting
//!
//! K-fold cross-fitting breaks the double use. Partition the rows into `K`
//! folds. For each fold `f`:
//!   1. **discover** the structure on the fold-COMPLEMENT (all rows outside
//!      `f`) — births, charts, decoder, learned subspace/metric;
//!   2. **evaluate** the artifact on the held-out rows of `f` ONLY, applying the
//!      structure discovered in step 1 without re-selecting anything on `f`.
//! Aggregate the held-out evaluations across folds. Because no row is ever
//! evaluated under a structure that saw it, the aggregate is (asymptotically)
//! unbiased for the population artifact: on noise it collapses to `≈ 0`, and on
//! genuine signal it recovers the true value. The gap between the naive
//! (all-rows discover + all-rows evaluate) artifact and the cross-fit aggregate
//! is a direct, reportable **optimism** estimate.
//!
//! This module provides the fold machinery, a generic scalar cross-fit driver
//! ([`cross_fit_scalar`]) that is agnostic to what "structure" and "artifact"
//! mean (the caller supplies the discover/evaluate closures, so it wires onto
//! EV, Θ, or dose forecasts identically), and the linear-subspace reconstruction
//! ([`fit_subspace`] / [`project_onto_subspace`]) that is the honest minimal
//! analog of a linear dictionary — used by the optimism test to exhibit the
//! `q/n` naive inflation and the cross-fit's honesty.
//!
//! No tuned constants (SPEC.md law): the only knob is `K`, a caller-owned
//! resolution choice, and the fold assignment is a deterministic function of `K`
//! and a caller-owned seed.

use super::*;

/// Caller-owned cross-fitting resolution.
#[derive(Debug, Clone, Copy)]
pub struct CrossFitConfig {
    /// Number of folds `K ≥ 2`. Structure is discovered on `K−1` folds and
    /// evaluated on the held-out one, `K` times.
    pub k_folds: usize,
    /// Deterministic seed for the row→fold assignment.
    pub seed: u64,
}

impl CrossFitConfig {
    /// A standard 5-fold configuration.
    pub fn five_fold(seed: u64) -> Self {
        CrossFitConfig { k_folds: 5, seed }
    }
}

/// Deterministic partition of `0..n` into `k` folds by a seeded permutation.
///
/// Uses a splitmix64 hash of `(seed, row)` to assign each row a fold, giving a
/// balanced, reproducible, order-independent split with no external RNG state.
#[derive(Debug, Clone)]
pub struct KFoldAssignment {
    fold_of_row: Vec<usize>,
    k_folds: usize,
}

impl KFoldAssignment {
    /// Build the assignment for `n` rows and `k` folds.
    pub fn new(n: usize, k_folds: usize, seed: u64) -> Result<Self, String> {
        if k_folds < 2 {
            return Err(format!("KFoldAssignment: need k_folds ≥ 2, got {k_folds}"));
        }
        if n < k_folds {
            return Err(format!(
                "KFoldAssignment: need n ≥ k_folds, got n={n} k={k_folds}"
            ));
        }
        // Deterministic near-balanced split: sort row indices by a splitmix64
        // hash keyed on the seed, then deal them round-robin into folds. Sorting
        // (not hash-mod) guarantees fold sizes differ by at most one regardless
        // of hash collisions, so no fold is ever starved.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&row| splitmix64(seed ^ (row as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
        let mut fold_of_row = vec![0usize; n];
        for (rank, &row) in order.iter().enumerate() {
            fold_of_row[row] = rank % k_folds;
        }
        Ok(KFoldAssignment {
            fold_of_row,
            k_folds,
        })
    }

    /// Number of folds.
    pub fn k_folds(&self) -> usize {
        self.k_folds
    }

    /// Rows held OUT in fold `f` (the evaluation rows).
    pub fn held_out(&self, fold: usize) -> Vec<usize> {
        (0..self.fold_of_row.len())
            .filter(|&row| self.fold_of_row[row] == fold)
            .collect()
    }

    /// Rows in the COMPLEMENT of fold `f` (the discovery/train rows).
    pub fn complement(&self, fold: usize) -> Vec<usize> {
        (0..self.fold_of_row.len())
            .filter(|&row| self.fold_of_row[row] != fold)
            .collect()
    }
}

/// splitmix64 — a tiny deterministic finalizer, used only for reproducible fold
/// assignment (never for statistical sampling).
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Result of a scalar cross-fit: the naive (double-use) artifact, the cross-fit
/// (held-out) aggregate, the per-fold held-out values, and their difference (the
/// optimism the naive path carries).
#[derive(Debug, Clone)]
pub struct CrossFitReport {
    /// Structure discovered on ALL rows, artifact evaluated on ALL rows.
    pub naive: f64,
    /// Mean of the held-out per-fold evaluations (the honest estimate).
    pub cross_fit: f64,
    /// One held-out artifact value per fold, fold order.
    pub per_fold: Vec<f64>,
    /// `naive − cross_fit` — the post-selection optimism.
    pub optimism: f64,
}

/// Generic K-fold cross-fit of a scalar artifact.
///
/// `discover(train_rows) -> S` fits the structure on the given rows and returns
/// a discovered-structure handle `S`. `evaluate(&S, eval_rows) -> Option<f64>`
/// applies that fixed structure to the given rows and returns the artifact (or
/// `None` if it is undefined on those rows, e.g. a degenerate fold). The naive
/// value uses the full row set for BOTH discover and evaluate; the cross-fit
/// value discovers on each fold-complement and evaluates on the held-out fold,
/// averaging the defined per-fold values.
///
/// Returns an error only if the fold split is ill-posed or EVERY fold is
/// undefined; individual undefined folds are skipped and reported via a shorter
/// `per_fold`.
pub fn cross_fit_scalar<S, D, E>(
    n: usize,
    config: CrossFitConfig,
    discover: D,
    evaluate: E,
) -> Result<CrossFitReport, String>
where
    D: Fn(&[usize]) -> Result<S, String>,
    E: Fn(&S, &[usize]) -> Option<f64>,
{
    let all_rows: Vec<usize> = (0..n).collect();
    let full_structure = discover(&all_rows)?;
    let naive = evaluate(&full_structure, &all_rows)
        .ok_or_else(|| "cross_fit_scalar: naive artifact undefined on full data".to_string())?;

    let folds = KFoldAssignment::new(n, config.k_folds, config.seed)?;
    let mut per_fold = Vec::with_capacity(config.k_folds);
    for f in 0..config.k_folds {
        let train = folds.complement(f);
        let test = folds.held_out(f);
        if train.is_empty() || test.is_empty() {
            continue;
        }
        let structure = discover(&train)?;
        if let Some(v) = evaluate(&structure, &test) {
            if v.is_finite() {
                per_fold.push(v);
            }
        }
    }
    if per_fold.is_empty() {
        return Err("cross_fit_scalar: every fold's held-out artifact was undefined".to_string());
    }
    let cross_fit = per_fold.iter().sum::<f64>() / per_fold.len() as f64;
    Ok(CrossFitReport {
        naive,
        cross_fit,
        optimism: naive - cross_fit,
        per_fold,
    })
}

/// The top-`q` right-singular subspace of the selected rows of `data`, centered
/// by the per-column mean of those rows — the honest minimal linear "dictionary"
/// whose optimism the cross-fit removes.
///
/// Returns `(mean, basis)` where `mean` is the length-`p` column mean over the
/// selected rows and `basis` is `(q × p)` orthonormal rows spanning the top-`q`
/// principal subspace. `q` is capped at `min(#rows, p)`.
pub(crate) fn fit_subspace(
    data: ArrayView2<'_, f64>,
    rows: &[usize],
    q: usize,
) -> Result<(Array1<f64>, Array2<f64>), String> {
    let p = data.ncols();
    let n = rows.len();
    if n == 0 || p == 0 {
        return Err("fit_subspace: empty selection".to_string());
    }
    let q = q.min(n).min(p);
    if q == 0 {
        return Err("fit_subspace: q resolved to 0".to_string());
    }
    let mut mean = Array1::<f64>::zeros(p);
    for &r in rows {
        for c in 0..p {
            mean[c] += data[[r, c]];
        }
    }
    mean.mapv_inplace(|v| v / n as f64);
    let mut centered = Array2::<f64>::zeros((n, p));
    for (i, &r) in rows.iter().enumerate() {
        for c in 0..p {
            centered[[i, c]] = data[[r, c]] - mean[c];
        }
    }
    let (_u, _s, vt) = centered
        .svd(false, true)
        .map_err(|e| format!("fit_subspace: SVD failed: {e:?}"))?;
    let vt = vt.ok_or_else(|| "fit_subspace: SVD returned no Vt".to_string())?;
    let rank = vt.nrows();
    let take = q.min(rank);
    let basis = vt.slice(s![0..take, ..]).to_owned();
    Ok((mean, basis))
}

/// Reconstruct the selected rows of `data` by projecting their centered form
/// onto `basis` (rows = orthonormal directions) and adding `mean` back — then
/// return the explained variance of that reconstruction ON those rows.
///
/// This is the "apply fixed structure to held-out rows" step: `mean`/`basis`
/// come from [`fit_subspace`] on the TRAIN rows, `rows` are the TEST rows.
pub(crate) fn subspace_reconstruction_ev(
    data: ArrayView2<'_, f64>,
    rows: &[usize],
    mean: ArrayView1<'_, f64>,
    basis: ArrayView2<'_, f64>,
) -> Option<f64> {
    let p = data.ncols();
    if rows.is_empty() || mean.len() != p || basis.ncols() != p {
        return None;
    }
    let n = rows.len();
    let mut target = Array2::<f64>::zeros((n, p));
    let mut fitted = Array2::<f64>::zeros((n, p));
    for (i, &r) in rows.iter().enumerate() {
        // centered row
        let mut coeff = Array1::<f64>::zeros(basis.nrows());
        for (b, dir) in basis.rows().into_iter().enumerate() {
            let mut acc = 0.0;
            for c in 0..p {
                acc += (data[[r, c]] - mean[c]) * dir[c];
            }
            coeff[b] = acc;
        }
        for c in 0..p {
            target[[i, c]] = data[[r, c]];
            let mut recon = mean[c];
            for (b, dir) in basis.rows().into_iter().enumerate() {
                recon += coeff[b] * dir[c];
            }
            fitted[[i, c]] = recon;
        }
    }
    reconstruction_explained_variance(target.view(), fitted.view())
}

/// A selected-feature OLS predictor: the honest minimal analog of a **selected**
/// structure whose optimism is the sharpest — the columns themselves are chosen
/// from the training data (a discrete selection), which is exactly the
/// post-selection inference trap the naive in-sample R² falls into and the
/// dose-forecast headline is exposed to.
pub(crate) struct SelectedRegression {
    /// Feature columns selected (top-`q` by |correlation| with the response on
    /// the training rows).
    pub cols: Vec<usize>,
    /// OLS intercept.
    pub intercept: f64,
    /// OLS slope per selected column, aligned with [`Self::cols`].
    pub coef: Vec<f64>,
}

/// Fit [`SelectedRegression`] on `rows`: select the `q` feature columns of `x`
/// most correlated (in absolute value) with `y`, then fit an intercept + slopes
/// by ordinary least squares on those columns.
pub(crate) fn fit_selected_regression(
    x: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    rows: &[usize],
    q: usize,
) -> Result<SelectedRegression, String> {
    let p = x.ncols();
    let n = rows.len();
    if n == 0 || p == 0 {
        return Err("fit_selected_regression: empty selection".to_string());
    }
    let q = q.min(p).min(n.saturating_sub(1)).max(1);
    // Column and response means over the training rows.
    let mut xbar = vec![0.0_f64; p];
    let mut ybar = 0.0_f64;
    for &r in rows {
        ybar += y[r];
        for c in 0..p {
            xbar[c] += x[[r, c]];
        }
    }
    ybar /= n as f64;
    for v in xbar.iter_mut() {
        *v /= n as f64;
    }
    // |corr(x_c, y)| via centered cross-products.
    let mut score = vec![0.0_f64; p];
    let mut yvar = 0.0_f64;
    for &r in rows {
        let dy = y[r] - ybar;
        yvar += dy * dy;
        for c in 0..p {
            score[c] += (x[[r, c]] - xbar[c]) * dy;
        }
    }
    let mut xvar = vec![0.0_f64; p];
    for &r in rows {
        for c in 0..p {
            let dx = x[[r, c]] - xbar[c];
            xvar[c] += dx * dx;
        }
    }
    let mut order: Vec<usize> = (0..p).collect();
    order.sort_by(|&a, &b| {
        let ca = score[a].abs() / (xvar[a].sqrt() * yvar.sqrt()).max(f64::MIN_POSITIVE);
        let cb = score[b].abs() / (xvar[b].sqrt() * yvar.sqrt()).max(f64::MIN_POSITIVE);
        cb.partial_cmp(&ca).unwrap_or(std::cmp::Ordering::Equal)
    });
    let cols: Vec<usize> = order.into_iter().take(q).collect();
    // OLS with intercept on the selected columns: design D = [1, x_cols].
    let d = cols.len() + 1;
    let mut xtx = Array2::<f64>::zeros((d, d));
    let mut xty = Array1::<f64>::zeros(d);
    for &r in rows {
        let mut drow = vec![0.0_f64; d];
        drow[0] = 1.0;
        for (j, &c) in cols.iter().enumerate() {
            drow[j + 1] = x[[r, c]];
        }
        for a in 0..d {
            xty[a] += drow[a] * y[r];
            for b in 0..d {
                xtx[[a, b]] += drow[a] * drow[b];
            }
        }
    }
    // Ridge-stabilize the normal equations by a whisker so a degenerate fold is
    // still solvable (does not bias the demonstration; ε is tiny relative to the
    // Gram diagonal). Not a tuned constant — it is the SPD floor that keeps the
    // Cholesky well-posed.
    for a in 0..d {
        xtx[[a, a]] += 1e-10 * (1.0 + xtx[[a, a]].abs());
    }
    let chol = xtx
        .cholesky(Side::Lower)
        .map_err(|e| format!("fit_selected_regression: normal-equation Cholesky: {e:?}"))?;
    let beta = chol.solvevec(&xty);
    Ok(SelectedRegression {
        cols,
        intercept: beta[0],
        coef: beta.iter().skip(1).copied().collect(),
    })
}

/// Out-of-sample `R²` of a fitted [`SelectedRegression`] on `rows`, against the
/// response's own mean over those rows (so a predictor no better than the mean
/// scores 0, and one worse than the mean scores negative — the honest sign that
/// a selected-on-noise structure does not generalize).
pub(crate) fn selected_regression_r2(
    x: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    rows: &[usize],
    fit: &SelectedRegression,
) -> Option<f64> {
    if rows.is_empty() {
        return None;
    }
    let n = rows.len();
    let ybar = rows.iter().map(|&r| y[r]).sum::<f64>() / n as f64;
    let mut ssr = 0.0_f64;
    let mut sst = 0.0_f64;
    for &r in rows {
        let mut pred = fit.intercept;
        for (j, &c) in fit.cols.iter().enumerate() {
            pred += fit.coef[j] * x[[r, c]];
        }
        let e = y[r] - pred;
        ssr += e * e;
        let dm = y[r] - ybar;
        sst += dm * dm;
    }
    if sst > f64::MIN_POSITIVE && ssr.is_finite() && sst.is_finite() {
        Some(1.0 - ssr / sst)
    } else {
        None
    }
}

/// Cross-fitted reconstruction explained variance — the honest, optimism-free
/// companion to the in-sample reconstruction EV the SAE headline reports.
///
/// Discovers the top-`q` linear reconstruction subspace on each fold-complement
/// and scores its EV on the held-out fold, aggregating across folds (see
/// [`cross_fit_scalar`]). The returned [`CrossFitReport`] carries the naive
/// (all-rows discover + score) EV, the cross-fit aggregate, and their difference
/// — the post-selection optimism. This is the reconstruction analog of a linear
/// dictionary; a curved/gated SAE plugs into [`cross_fit_scalar`] the same way by
/// supplying its own discover/score closures.
pub fn cross_fit_reconstruction_ev(
    data: ArrayView2<'_, f64>,
    config: CrossFitConfig,
    q: usize,
) -> Result<CrossFitReport, String> {
    let n = data.nrows();
    cross_fit_scalar(
        n,
        config,
        |train| fit_subspace(data, train, q),
        |(mean, basis), test| subspace_reconstruction_ev(data, test, mean.view(), basis.view()),
    )
}

/// Cross-fitted `R²` of a selected-feature linear forecast — the honest
/// companion to the in-sample forecast fit (the dose-forecast headline's
/// analog), which is exposed to post-SELECTION optimism because the predictor
/// columns are themselves chosen from the data.
///
/// Selects and fits on each fold-complement, scores held-out `R²`, aggregates.
/// The naive/cross-fit/optimism split in the returned [`CrossFitReport`] is the
/// direct, reportable measure of how much the naive in-sample forecast overstates
/// generalization.
pub fn cross_fit_selected_forecast_r2(
    x: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    config: CrossFitConfig,
    q: usize,
) -> Result<CrossFitReport, String> {
    let n = x.nrows();
    if y.len() != n {
        return Err(format!(
            "cross_fit_selected_forecast_r2: x has {n} rows but y has {}",
            y.len()
        ));
    }
    cross_fit_scalar(
        n,
        config,
        |train| fit_selected_regression(x, y, train, q),
        |fit, test| selected_regression_r2(x, y, test, fit),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    #[test]
    fn folds_are_balanced_and_partition() {
        let n = 103;
        let k = 5;
        let folds = KFoldAssignment::new(n, k, 42).unwrap();
        let mut seen = vec![false; n];
        let mut sizes = vec![0usize; k];
        for f in 0..k {
            let held = folds.held_out(f);
            let comp = folds.complement(f);
            assert_eq!(held.len() + comp.len(), n, "held+comp must cover all rows");
            for &r in &held {
                assert!(!seen[r], "row {r} in two folds");
                seen[r] = true;
                sizes[f] += 1;
            }
            // complement is exactly the non-held rows
            assert!(comp.iter().all(|&r| !folds.held_out(f).contains(&r)));
        }
        assert!(seen.iter().all(|&s| s), "every row assigned");
        let lo = *sizes.iter().min().unwrap();
        let hi = *sizes.iter().max().unwrap();
        assert!(hi - lo <= 1, "fold sizes must differ by ≤ 1, got {sizes:?}");
    }

    #[test]
    fn selection_cross_fit_is_zero_on_noise_and_naive_is_inflated() {
        // FLAGSHIP optimism test. Many candidate features, a response that is PURE
        // NOISE independent of every feature. "Structure discovery" selects the q
        // features most correlated with the response and fits OLS — a discrete
        // selection, the sharpest post-selection trap. On the population the R² is
        // exactly 0. The naive path (select AND score on the same rows) reports a
        // clearly-positive R² (the selected columns were chosen to fit THIS
        // sample); cross-fitting scores on rows the selection never saw, so it
        // must collapse to ≈ 0 — the honesty the charge demands.
        let n = 300;
        let p = 60;
        let q = 5;
        let mut rng = StdRng::seed_from_u64(2024);
        let mut x = Array2::<f64>::zeros((n, p));
        for v in x.iter_mut() {
            *v = rng.random_range(-1.0..1.0);
        }
        let mut y = Array1::<f64>::zeros(n);
        for v in y.iter_mut() {
            *v = rng.random_range(-1.0..1.0); // independent of x — no signal
        }
        let report = cross_fit_scalar(
            n,
            CrossFitConfig::five_fold(7),
            |train| fit_selected_regression(x.view(), y.view(), train, q),
            |fit, test| selected_regression_r2(x.view(), y.view(), test, fit),
        )
        .unwrap();

        println!(
            "[optimism/noise] naive R²={:.4} cross_fit R²={:.4} optimism={:.4} per_fold={:?}",
            report.naive, report.cross_fit, report.optimism, report.per_fold
        );
        // Naive in-sample R² is inflated well above 0 by the selection.
        assert!(
            report.naive > 0.05,
            "naive selected R² should be inflated on noise, got {}",
            report.naive
        );
        // Cross-fit collapses to ≈ 0 (typically slightly negative — the selected
        // features do worse than the held-out mean, the honest verdict).
        assert!(
            report.cross_fit < 0.02,
            "cross-fit R² on noise must be ≈ 0 (not positive), got {}",
            report.cross_fit
        );
        // And the optimism (naive − cross_fit) is real and clearly positive.
        assert!(
            report.optimism > 0.05,
            "optimism should be clearly positive on noise, got {}",
            report.optimism
        );
    }

    #[test]
    fn selection_cross_fit_recovers_signal() {
        // Response genuinely depends on a few features. Cross-fit R² must stay
        // high (the selected structure generalizes) and the optimism must be
        // small — cross-fitting removes selection bias without destroying signal.
        let n = 400;
        let p = 60;
        let q = 5;
        let mut rng = StdRng::seed_from_u64(51);
        let mut x = Array2::<f64>::zeros((n, p));
        for v in x.iter_mut() {
            *v = rng.random_range(-1.0..1.0);
        }
        // y depends on columns 3, 17, 42 plus small noise.
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            y[i] = 1.5 * x[[i, 3]] - 2.0 * x[[i, 17]] + 1.0 * x[[i, 42]]
                + 0.05 * rng.random_range(-1.0..1.0);
        }
        let report = cross_fit_scalar(
            n,
            CrossFitConfig::five_fold(5),
            |train| fit_selected_regression(x.view(), y.view(), train, q),
            |fit, test| selected_regression_r2(x.view(), y.view(), test, fit),
        )
        .unwrap();
        assert!(
            report.cross_fit > 0.9,
            "cross-fit must keep high R² on real signal, got {}",
            report.cross_fit
        );
        assert!(
            report.optimism.abs() < 0.05,
            "optimism must be small when the structure is real, got {}",
            report.optimism
        );
    }

    #[test]
    fn subspace_reconstruction_optimism_is_positive_on_noise() {
        // The reconstruction-EV analog of the SAE headline. A q-dim linear
        // subspace inevitably captures the ≈ q/p chance fraction of held-out
        // isotropic noise (dimension counting), so the cross-fit EV sits near
        // that chance floor — NOT at 0 — while the naive in-sample EV is strictly
        // higher because it picks the sample's own top-q eigendirections. The
        // load-bearing, honest claim is therefore that the OPTIMISM
        // (naive − cross_fit) is clearly positive, and cross_fit ≈ q/p.
        let n = 400;
        let p = 20;
        let q = 6;
        let mut rng = StdRng::seed_from_u64(2024);
        let mut data = Array2::<f64>::zeros((n, p));
        for v in data.iter_mut() {
            *v = rng.random_range(-1.0..1.0);
        }
        let report = cross_fit_scalar(
            n,
            CrossFitConfig::five_fold(7),
            |train| fit_subspace(data.view(), train, q),
            |(mean, basis), test| {
                subspace_reconstruction_ev(data.view(), test, mean.view(), basis.view())
            },
        )
        .unwrap();
        let chance = q as f64 / p as f64;
        println!(
            "[optimism/recon] naive EV={:.4} cross_fit EV={:.4} optimism={:.4} chance q/p={:.4}",
            report.naive, report.cross_fit, report.optimism, chance
        );
        assert!(
            report.optimism > 0.05,
            "reconstruction optimism (naive − cross_fit) should be positive on noise, got {}",
            report.optimism
        );
        assert!(
            (report.cross_fit - chance).abs() < 0.08,
            "cross-fit reconstruction EV should sit near the q/p={chance} chance floor, got {}",
            report.cross_fit
        );
        assert!(
            report.naive > report.cross_fit,
            "naive EV must exceed cross-fit EV (optimism), naive={} cross_fit={}",
            report.naive,
            report.cross_fit
        );
    }

    #[test]
    fn subspace_cross_fit_recovers_true_ev_on_signal() {
        // Rank-3 signal + small noise. Cross-fit reconstruction EV recovers the
        // true high value and the optimism is small.
        let n = 500;
        let p = 24;
        let r_true = 3;
        let q = 3;
        let mut rng = StdRng::seed_from_u64(99);
        let mut loadings = Array2::<f64>::zeros((r_true, p));
        for v in loadings.iter_mut() {
            *v = rng.random_range(-1.0..1.0);
        }
        let mut data = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let mut scores = [0.0_f64; 3];
            for s in scores.iter_mut() {
                *s = rng.random_range(-2.0..2.0);
            }
            for c in 0..p {
                let mut v = 0.05 * rng.random_range(-1.0..1.0);
                for (k, &sc) in scores.iter().enumerate() {
                    v += sc * loadings[[k, c]];
                }
                data[[i, c]] = v;
            }
        }
        let report = cross_fit_scalar(
            n,
            CrossFitConfig::five_fold(5),
            |train| fit_subspace(data.view(), train, q),
            |(mean, basis), test| {
                subspace_reconstruction_ev(data.view(), test, mean.view(), basis.view())
            },
        )
        .unwrap();
        assert!(
            report.cross_fit > 0.95,
            "cross-fit must recover the true high EV on real signal, got {}",
            report.cross_fit
        );
        assert!(
            report.optimism.abs() < 0.02,
            "optimism must be small when the structure is real, got {}",
            report.optimism
        );
    }
}
