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
//! The aggregate is POOLED: every fold's held-out residual sum of squares is
//! summed and divided by the same total sum of squares (about the full-data
//! column means) the naive value uses, so `naive − cross_fit` is exactly the
//! held-out minus held-in residual over one shared denominator. Averaging
//! per-fold ratios instead — each over its own fold-mean TSS, which is short by
//! about one row's worth of variance — biases every held-out value by `≈ K/n`,
//! the same order as the `q/n` optimism being measured.
//!
//! This module provides the fold machinery and the linear-subspace
//! reconstruction ([`fit_subspace`] / [`subspace_rss`]) that is the honest
//! minimal analog of a linear dictionary, cross-fitted by
//! [`cross_fit_reconstruction_ev`].
//!
//! No tuned constants (SPEC.md law): the only knob is `K`, a caller-owned
//! resolution choice, and the fold assignment is a deterministic function of `K`
//! and a caller-owned seed.

use super::*;
use gam_linalg::utils::splitmix64_hash;

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
pub(crate) struct KFoldAssignment {
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
            splitmix64_hash(seed ^ (row as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        });
        let mut fold_of_row = vec![0usize; n];
        for (rank, &row) in order.iter().enumerate() {
            fold_of_row[row] = rank % k_folds;
        }
        Ok(KFoldAssignment { fold_of_row })
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

/// Result of a reconstruction cross-fit: the naive (double-use) explained
/// variance, the cross-fit (held-out) explained variance, the per-fold held-out
/// values, and their difference (the optimism the naive path carries).
#[derive(Debug, Clone)]
pub struct CrossFitReport {
    /// Structure discovered on ALL rows, `1 − RSS_in / TSS` evaluated on ALL rows.
    pub naive: f64,
    /// Pooled held-out explained variance `1 − Σ_f RSS_f / TSS`, over the same
    /// `TSS` as [`Self::naive`]. Equal to the `TSS_f / TSS`-weighted mean of
    /// [`Self::per_fold`].
    pub cross_fit: f64,
    /// One held-out explained variance per fold, fold order: `1 − RSS_f / TSS_f`
    /// with `TSS_f` about the same full-data column means (`NaN` for a fold whose
    /// held-out rows all sit exactly at those means). Every fold is present.
    pub per_fold: Vec<f64>,
    /// `naive − cross_fit = (Σ_f RSS_f − RSS_in) / TSS` — the post-selection
    /// optimism.
    pub optimism: f64,
}

/// Top-`q` linear reconstruction subspace of the selected rows: their column
/// means and the leading right singular vectors of the centered rows (one
/// orthonormal direction per row of the returned basis).
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

/// Residual sum of squares of reconstructing the selected rows of `data` by
/// projecting their centered form onto `basis` (rows = orthonormal directions)
/// and adding `mean` back.
///
/// This is the "apply fixed structure to held-out rows" step: `mean`/`basis`
/// come from [`fit_subspace`] on the TRAIN rows, `rows` are the TEST rows.
pub(crate) fn subspace_rss(
    data: ArrayView2<'_, f64>,
    rows: &[usize],
    mean: ArrayView1<'_, f64>,
    basis: ArrayView2<'_, f64>,
) -> f64 {
    let p = data.ncols();
    assert_eq!(mean.len(), p);
    assert_eq!(basis.ncols(), p);
    let mut coeff = vec![0.0_f64; basis.nrows()];
    let mut rss = 0.0_f64;
    for &r in rows {
        for (b, dir) in basis.rows().into_iter().enumerate() {
            let mut acc = 0.0;
            for c in 0..p {
                acc += (data[[r, c]] - mean[c]) * dir[c];
            }
            coeff[b] = acc;
        }
        for c in 0..p {
            let mut recon = mean[c];
            for (b, dir) in basis.rows().into_iter().enumerate() {
                recon += coeff[b] * dir[c];
            }
            let residual = data[[r, c]] - recon;
            rss += residual * residual;
        }
    }
    rss
}

/// Total sum of squares of the selected rows about a fixed `center`.
fn total_sum_of_squares(
    data: ArrayView2<'_, f64>,
    rows: &[usize],
    center: ArrayView1<'_, f64>,
) -> f64 {
    let mut tss = 0.0_f64;
    for &r in rows {
        for c in 0..data.ncols() {
            let centered = data[[r, c]] - center[c];
            tss += centered * centered;
        }
    }
    tss
}

/// Cross-fitted reconstruction explained variance — the honest, optimism-free
/// companion to the in-sample reconstruction EV the SAE headline reports.
///
/// Discovers the top-`q` linear reconstruction subspace on each fold-complement,
/// scores its residual on the held-out fold, and pools the held-out residuals
/// over the naive value's own total sum of squares (see [`CrossFitReport`]).
/// This is the reconstruction analog of a linear dictionary.
///
/// Every fold is scored. A target with no variance about its column means (no
/// explained variance to define) or a non-finite residual is an error, never a
/// silently shortened aggregate.
pub(crate) fn cross_fit_reconstruction_ev(
    data: ArrayView2<'_, f64>,
    config: CrossFitConfig,
    q: usize,
) -> Result<CrossFitReport, String> {
    let (n, p) = data.dim();
    let folds = KFoldAssignment::new(n, config.k_folds, config.seed)?;
    let all_rows: Vec<usize> = (0..n).collect();
    let mut center = Array1::<f64>::zeros(p);
    for c in 0..p {
        let mut acc = 0.0;
        for r in 0..n {
            acc += data[[r, c]];
        }
        center[c] = acc / n as f64;
    }
    let tss = total_sum_of_squares(data, &all_rows, center.view());
    if !(tss.is_finite() && tss > 0.0) {
        return Err(format!(
            "cross_fit_reconstruction_ev: total sum of squares {tss:e} about the column \
             means is not positive and finite, so explained variance is undefined"
        ));
    }

    let (mean, basis) = fit_subspace(data, &all_rows, q)?;
    let naive_rss = subspace_rss(data, &all_rows, mean.view(), basis.view());

    let mut per_fold = Vec::with_capacity(config.k_folds);
    let mut held_out_rss = 0.0_f64;
    for fold in 0..config.k_folds {
        let (mean, basis) = fit_subspace(data, &folds.complement(fold), q)?;
        let test = folds.held_out(fold);
        let rss = subspace_rss(data, &test, mean.view(), basis.view());
        held_out_rss += rss;
        per_fold.push(crate::tiered::explained_variance_from_sums(
            rss,
            total_sum_of_squares(data, &test, center.view()),
        ));
    }
    if !(naive_rss.is_finite() && held_out_rss.is_finite()) {
        return Err(format!(
            "cross_fit_reconstruction_ev: non-finite residual (held-in {naive_rss:e}, \
             held-out {held_out_rss:e})"
        ));
    }
    let naive = crate::tiered::explained_variance_from_sums(naive_rss, tss);
    let cross_fit = crate::tiered::explained_variance_from_sums(held_out_rss, tss);
    Ok(CrossFitReport {
        naive,
        cross_fit,
        optimism: naive - cross_fit,
        per_fold,
    })
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

    /// A deterministic `(n, p)` target with a dominant direction plus spread in
    /// every other column, so a `q = 1` subspace leaves a held-out residual.
    fn structured_target(n: usize, p: usize) -> Array2<f64> {
        Array2::from_shape_fn((n, p), |(r, c)| {
            let t = r as f64 / n as f64;
            let signal = (c as f64 + 1.0) * (2.0 * t - 1.0);
            let spread = ((r * (c + 3) * 7919) % 97) as f64 / 97.0 - 0.5;
            signal + 0.3 * spread
        })
    }

    /// The cross-fit value shares the naive value's denominator, so it is the
    /// `TSS_f / TSS`-weighted mean of the per-fold values (the per-fold `TSS_f`
    /// are taken about the same full-data means and partition `TSS`), and every
    /// fold is scored. The old mean of per-fold ratios, each over its own
    /// fold-mean TSS, satisfied neither identity.
    #[test]
    fn cross_fit_pools_every_fold_over_the_naive_denominator() {
        let (n, p, k) = (40usize, 4usize, 5usize);
        let data = structured_target(n, p);
        let report = cross_fit_reconstruction_ev(
            data.view(),
            CrossFitConfig {
                k_folds: k,
                seed: 7,
            },
            1,
        )
        .expect("cross-fit on a varying target");
        assert_eq!(report.per_fold.len(), k, "every fold must be scored");

        let center = data.mean_axis(ndarray::Axis(0)).expect("n > 0");
        let all_rows: Vec<usize> = (0..n).collect();
        let tss = total_sum_of_squares(data.view(), &all_rows, center.view());
        let folds = KFoldAssignment::new(n, k, 7).expect("folds");
        let weighted: f64 = (0..k)
            .map(|f| {
                let w = total_sum_of_squares(data.view(), &folds.held_out(f), center.view()) / tss;
                w * report.per_fold[f]
            })
            .sum();
        // Both sides are the same k-term sum of O(1) quantities, so they agree to
        // a few ulps of each term.
        let tol = 8.0 * k as f64 * f64::EPSILON;
        assert!(
            (weighted - report.cross_fit).abs() <= tol,
            "cross_fit {} must equal the TSS-weighted per-fold mean {weighted}",
            report.cross_fit
        );
        assert_eq!(report.optimism, report.naive - report.cross_fit);
    }

    #[test]
    fn constant_target_has_no_explained_variance_to_cross_fit() {
        let data = Array2::<f64>::from_elem((10, 3), 2.5);
        let err = cross_fit_reconstruction_ev(
            data.view(),
            CrossFitConfig {
                k_folds: 2,
                seed: 0,
            },
            1,
        )
        .expect_err("a constant target has no explained variance");
        assert!(err.contains("undefined"), "{err}");
    }
}
