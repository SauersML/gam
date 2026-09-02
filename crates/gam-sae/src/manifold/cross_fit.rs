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
//! coefficient of determination. So on pure noise the naive in-sample EV sits
//! near `q/n > 0`, not at the honest `0`. This quantifies selection optimism; it
//! is not a production decoder-disappearance verdict.
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

/// Deterministic partition of `0..n` into `k` folds by a seeded permutation.
///
/// Uses a splitmix64 hash of `(seed, row)` to assign each row a fold, giving a
/// balanced, reproducible, order-independent split with no external RNG state.
#[derive(Debug, Clone)]
pub struct KFoldAssignment {
    fold_of_row: Vec<usize>,
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
        order.sort_by_key(|&row| {
            splitmix64(seed ^ (row as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        });
        let mut fold_of_row = vec![0usize; n];
        for (rank, &row) in order.iter().enumerate() {
            fold_of_row[row] = rank % k_folds;
        }
        Ok(KFoldAssignment {
            fold_of_row,
        })
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
fn splitmix64(x: u64) -> u64 {
    gam_linalg::utils::splitmix64_hash(x)
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

/// Row-subset implementation shared by the public full-data PCA seed and the
/// honest cross-fit scorer below.
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

#[cfg(test)]
mod tests {
    use super::*;

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

}
