//! #974 — the structured-residual covariance estimator and the single producer
//! of [`MetricProvenance::WhitenedStructured`](gam_problem::MetricProvenance::WhitenedStructured).
//!
//! # What this estimates
//!
//! Given a residual matrix `R ∈ ℝ^{n×p}` (one `p`-dimensional reconstruction
//! residual per row) and a smooth *activity coordinate* `z ∈ ℝ^n`, this fits the
//! **structured residual-covariance model**
//!
//! ```text
//!     Cov(r_n) = Σ_n = Λ · c(z_n) · Λᵀ + D ,
//! ```
//!
//! where
//!
//! * `Λ ∈ ℝ^{p×r}` is a **low-rank interference factor** (the shared
//!   off-isotropic subspace the residuals correlate along — e.g. a planted
//!   interference subspace or a topology-race confound),
//! * `D = diag(d) ≻ 0` is the **idiosyncratic diagonal** (per-channel
//!   independent noise), and
//! * `c(z) > 0` is the **activity-scale law**: a strictly-positive scalar that
//!   modulates the factor energy with the activity coordinate, piecewise
//!   constant over equal-*count* bins of `z` and normalized to row-mean one.
//!   Equal counts are equal Fisher information per bin at the null law, so the
//!   law has one resolution number, and that number is read from the evidence
//!   on a nested doubling ladder rather than fixed (#3337); see
//!   [`StructuredResidualModel::fit`].
//!
//! Each rank `r` is fitted to its certified posterior mode by a Newton
//! trust-region solve with exact analytic derivatives, under priors centred on
//! "no factor" and on the null activity law `c ≡ 1`, and scored by its Laplace
//! log marginal likelihood (see [`evidence`]). The **factor count `r`** is the
//! evidence maximizer over every rank the model identifies,
//! `0 ≤ r ≤ L(p)` with `L(p)` the Ledermann bound; a higher rank is taken only
//! on strictly larger evidence. The candidate set is fixed by identifiability
//! alone, so no caller-supplied cap can truncate the model comparison. The
//! **resolution of `c(z)`** is read from the same evidence on the nested
//! doubling ladder `B = 1, 2, 4, …`, and is likewise not a knob.
//!
//! # What it produces
//!
//! `StructuredResidualModel::row_metric` materializes the **per-row precision
//! factor** `U_n ∈ ℝ^{p×p}` with `U_n U_nᵀ = Σ_n^{-1}`, packaged as a
//! [`RowMetric`](gam_problem::RowMetric) with
//! [`MetricProvenance::WhitenedStructured`](gam_problem::MetricProvenance::WhitenedStructured).
//! Whitening a residual `r_n` through it (`U_nᵀ r_n`) yields a vector whose
//! squared Euclidean norm is `r_nᵀ Σ_n^{-1} r_n` — the Mahalanobis residual under
//! the estimated noise model, which is exactly the likelihood-correct data-fit.
//! The factor is built from the SPD covariance itself: `Σ_n = L_n L_nᵀ`
//! (Cholesky) and `U_n = L_n^{-ᵀ}`, so `U_n U_nᵀ = Σ_n^{-1}` exactly and
//! whitening is the triangular solve `L_n^{-1} r_n`. The precision is never
//! assembled, so there is no subtraction to cancel and no inverse to repair.
//!
//! This is the first real producer of `WhitenedStructured`, and therefore the
//! first metric whose `whitens_likelihood()` is `true`: see
//! [`RowMetric::whitens_likelihood`](gam_problem::RowMetric::whitens_likelihood).

mod evidence;

use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use faer::Side;
use gam_linalg::faer_ndarray::FaerCholesky;
use gam_problem::RowMetric;

/// The bin of every row at resolution `bins`: equal-*count* (quantile) bins of
/// the activity coordinate `z`.
///
/// Equal count, not equal width, because the quantity a bin has to carry is
/// information about its own scale parameter. At the null law `c ≡ 1` — the
/// Dirichlet prior's centre, and the state the resolution is chosen at — every
/// bin sees the same `Σ_b = ΛΛᵀ + D`, so the Fisher information for `ln c_b`
/// is `(n_b / 2)·tr[(Σ⁻¹ΛΛᵀ)²]`: the same constant for every bin times its own
/// row count. Equal counts therefore *are* equal information per bin, which is
/// what makes one resolution number meaningful for the whole law. Equal-width
/// bins put that information wherever `z` happens to be dense.
///
/// The edges are the `j / bins` empirical quantiles. Doubling `bins` keeps
/// every existing edge (`⌊2j·n / 2B⌋ = ⌊j·n / B⌋`), so the ladder in
/// [`StructuredResidualModel::fit`] is nested: the finer partition refines the
/// coarser one, and the finer model's law contains the coarser one's.
///
/// A row's bin is a function of its `z` alone (the count of edges at or below
/// it), so tied `z` values always share a bin and the assignment is monotone.
/// `bins <= 1`, and a `z` with no spread, put every row in one bin — the
/// homoscedastic law `c ≡ 1`.
fn activity_bin_assignment(z: ArrayView1<'_, f64>, bins: usize) -> Vec<usize> {
    let n = z.len();
    if bins <= 1 || n == 0 {
        return vec![0; n];
    }
    let mut sorted: Vec<f64> = z.iter().copied().collect();
    sorted.sort_by(f64::total_cmp);
    let edges: Vec<f64> = (1..bins)
        .map(|j| sorted[((j as u128 * n as u128) / bins as u128) as usize])
        .collect();
    z.iter()
        .map(|value| edges.partition_point(|edge| edge <= value))
        .collect()
}

/// How many of the `bins` bins hold at least one row.
fn occupied_bin_count(row_bin: &[usize], bins: usize) -> usize {
    let mut seen = vec![false; bins];
    let mut count = 0usize;
    for &b in row_bin {
        if !seen[b] {
            seen[b] = true;
            count += 1;
        }
    }
    count
}

/// How many distinct values `z` takes. The saturated resolution: once that
/// many bins are occupied every bin holds exactly one value of `z`, and no
/// finer partition can regroup a single row.
fn distinct_activity_values(z: ArrayView1<'_, f64>) -> usize {
    let mut sorted: Vec<f64> = z.iter().copied().collect();
    sorted.sort_by(f64::total_cmp);
    sorted.dedup();
    sorted.len()
}

/// The fitted structured residual-covariance model: low-rank factor `Λ`,
/// idiosyncratic diagonal `D`, and the smooth activity-scale `c(z)` evaluated at
/// every row. Produces per-row precision factors and the
/// [`MetricProvenance::WhitenedStructured`](gam_problem::MetricProvenance::WhitenedStructured)
/// `RowMetric`.
#[derive(Clone, Debug)]
pub struct StructuredResidualModel {
    /// Output dimensionality `p` (residual width).
    p: usize,
    /// Selected factor rank `r` (`0 ≤ r ≤ p`). `0` ⇒ pure-diagonal noise model.
    factor_rank: usize,
    /// Interference factor `Λ ∈ ℝ^{p×r}` (the shared off-diagonal subspace).
    lambda: Array2<f64>,
    /// Idiosyncratic diagonal `d ∈ ℝ^p` (`D = diag(d)`, every `d_j > 0`).
    diagonal: Array1<f64>,
    /// Per-row activity scale `c(z_n) > 0`, length `n`.
    row_scale: Array1<f64>,
    /// Selected resolution of the activity law: the number of equal-count bins
    /// of `z` the evidence kept. `1` is the homoscedastic law `c ≡ 1`.
    activity_bins: usize,
    /// Laplace log marginal likelihood of the selected rank and resolution, in
    /// raw residual units: the value the search maximized.
    log_evidence: f64,
}

/// Estimator inputs: the residual matrix and the smooth activity coordinate.
///
/// `residuals` is `R ∈ ℝ^{n×p}`. `activity` is `z ∈ ℝ^n` — the coordinate the
/// scale law `c(z)` is smooth in (e.g. an assignment-mass or activation-strength
/// summary per row). When no genuine activity coordinate is available, passing a
/// constant `z` recovers a homoscedastic factor model (`c(z) ≡ const`).
pub struct ResidualFactorInput<'a> {
    /// Residual matrix `R ∈ ℝ^{n×p}`.
    pub residuals: ArrayView2<'a, f64>,
    /// Activity coordinate `z ∈ ℝ^n` the scale law is smooth in.
    pub activity: ArrayView1<'a, f64>,
}

/// A persistent, evidence-earning residual factor direction — a promotion
/// candidate for the #2021 Λ nursery→promotion birth channel. Emitted by
/// [`StructuredResidualModel::promotion_candidates`] when a column of this
/// pass's `Λ` both (a) aligns with a column of the previous pass's `Λ` (it
/// *persisted* across the outer alternation) and (b) explains residual energy
/// above the idiosyncratic-noise floor (it *earns its complexity*). The driver
/// accumulates persistence across passes (the nursery) and, once a direction
/// survives long enough, promotes it to a new curved/linear atom seeded by
/// [`Self::direction`].
#[derive(Clone, Debug)]
pub struct FactorPromotion {
    /// Unit-norm factor direction in output space (`p`-vector): the L2-normalized
    /// column of `Λ`. This is the decoder direction a promoted atom is born with.
    pub direction: Array1<f64>,
    /// Explained residual energy `‖Λ_:,j‖²` (pre-normalization squared column
    /// norm) — the factor's contribution to `Σ = c·ΛΛᵀ + D`. Candidates are
    /// returned in descending energy so the driver promotes the strongest first.
    pub energy: f64,
    /// `|cos|` alignment (∈ `[0, 1]`) between this direction and the best-matching
    /// column of the previous pass's `Λ` — the persistence score gating promotion.
    pub persistence_alignment: f64,
    /// Index of the best-matching previous-pass `Λ` column (the nursery lineage
    /// this candidate continues), so the driver can track a stable identity for a
    /// direction across passes.
    pub prev_column: usize,
}

impl StructuredResidualModel {
    /// Fit the structured residual-covariance model: every identifiable rank
    /// `r = 0, 1, …, L(p)` (`L(p)` the Ledermann bound, the largest rank a
    /// `p`-channel factor model identifies) is fitted to its certified posterior mode and
    /// scored by its Laplace evidence, and the maximizer is kept (the lower rank
    /// on a tie). Errors on shape / non-finite input, on a residual channel that
    /// is identically zero, and on any rank whose mode the solver cannot certify:
    /// a rank with no certified evidence cannot be compared.
    ///
    /// # The activity law's resolution
    ///
    /// The number of bins the activity law `c(z)` is piecewise constant over is
    /// read from the same evidence, on the same ladder discipline as the rank
    /// (#3337). It is not a constant of this module.
    ///
    /// The rungs are `B = 1, 2, 4, …` equal-count bins of `z`
    /// (`activity_bin_assignment`). Doubling is the least nested refinement
    /// that reaches every bin — the smallest integer ratio above one — and it
    /// keeps every existing edge, so rung `2B` refines rung `B` and its law
    /// contains rung `B`'s. Rung 1 is the null law `c ≡ 1` exactly: one slot
    /// leaves no free scale, so it is always recoverable and it is where the
    /// search starts.
    ///
    /// Evidences are comparable across rungs for the same reason they are
    /// comparable across ranks: the same rows under the same Gaussian
    /// likelihood, the factor and activity priors both proper with their
    /// normalizers included, and the flat `η` prior and the standardization
    /// offset `n·Σ_j ln s_j` identical at every rung (the channel scales sum
    /// over all slots, so they do not move with `B`).
    ///
    /// The ladder stops at the first refinement whose best evidence does not
    /// beat the best so far — the adequacy discipline a nested refinement is
    /// entitled to, and the one the spatial resolution loop uses (#1689). A
    /// rung whose best rank is 0 does not count as a refinement whatever its
    /// evidence reads: rank 0 has no activity law (`has_kappa` is false), so
    /// its evidence is the same number at every resolution and what separates
    /// two readings of it is the rounding of a different grouping of the same
    /// sums. Two structural stops bound the ladder further: a refinement that
    /// splits no bin is the
    /// identical model (nested edges, so an unchanged occupied-bin count means
    /// no bin split; `binned_moments` drops the empty bins and returns the same
    /// moments), so it is skipped without a refit; and once the occupied bins
    /// number as many as `z` has distinct values, every bin holds one value and
    /// no finer partition regroups anything. `B` never exceeds `n`, the
    /// saturated one-row-per-bin resolution.
    ///
    /// Cost: one rank ladder per *fitted* rung instead of one in total. Under a
    /// null activity law the search stops at rung 2, and rung 1 carries one
    /// slot where the fixed eight-bin law carried eight, so it is cheaper than
    /// what it replaces; under a real law it pays `log2(B) + 2` rank ladders
    /// for the resolution it keeps.
    pub fn fit(input: ResidualFactorInput<'_>) -> Result<Self, String> {
        let r = input.residuals;
        let z = input.activity;
        let n = r.nrows();
        let p = r.ncols();
        if n == 0 || p == 0 {
            return Err(format!(
                "StructuredResidualModel::fit: residuals must be non-empty; got ({n}, {p})"
            ));
        }
        if z.len() != n {
            return Err(format!(
                "StructuredResidualModel::fit: activity length {} != residual rows {n}",
                z.len()
            ));
        }
        if !r.iter().all(|v| v.is_finite()) {
            return Err("StructuredResidualModel::fit: residuals must be finite".to_string());
        }
        if !z.iter().all(|v| v.is_finite()) {
            return Err("StructuredResidualModel::fit: activity must be finite".to_string());
        }

        let max_rank = evidence::ledermann_bound(p);
        let saturated = distinct_activity_values(z);
        let mut selected: Option<(usize, usize, evidence::RankFit, Vec<usize>)> = None;
        let mut fitted_occupancy = 0usize;
        let mut bins = 1usize;
        while bins <= n {
            let row_bin = activity_bin_assignment(z, bins);
            let occupied = occupied_bin_count(&row_bin, bins);
            if occupied > fitted_occupancy {
                fitted_occupancy = occupied;
                let moments = evidence::binned_moments(r, &row_bin, bins)?;
                let mut rung: Option<(usize, evidence::RankFit)> = None;
                for rank in 0..=max_rank {
                    let candidate = evidence::fit_rank(&moments, rank)?;
                    let take = rung
                        .as_ref()
                        .map_or(true, |(_, b)| candidate.log_evidence > b.log_evidence);
                    if take {
                        rung = Some((rank, candidate));
                    }
                }
                let (rank, fit) = rung.ok_or_else(|| {
                    "StructuredResidualModel::fit: empty rank search".to_string()
                })?;
                // A rung whose best rank is 0 carries no activity law at all —
                // `c ≡ 1` exactly, at every resolution — so its evidence is
                // rung 1's rank-0 evidence recomputed on a different grouping
                // of the same sums, and the difference between them is
                // rounding. It is never a refinement of anything, and it can
                // never beat an accepted rank-1 rung either, because that rung
                // already beat the same number at rung 1.
                let refines = match selected.as_ref() {
                    None => true,
                    Some((_, _, best, _)) => rank > 0 && fit.log_evidence > best.log_evidence,
                };
                if !refines {
                    break;
                }
                selected = Some((bins, rank, fit, moments.row_slot));
            }
            if fitted_occupancy >= saturated {
                break;
            }
            bins = bins.saturating_mul(2);
        }
        let (activity_bins, factor_rank, fit, row_slot) = selected
            .ok_or_else(|| "StructuredResidualModel::fit: empty resolution ladder".to_string())?;
        let row_scale = Array1::from_iter(row_slot.iter().map(|&s| fit.slot_scale[s]));
        Ok(Self {
            p,
            factor_rank,
            lambda: fit.lambda,
            diagonal: fit.diagonal,
            row_scale,
            activity_bins,
            log_evidence: fit.log_evidence,
        })
    }

    /// The resolution of the fitted activity law: the number of equal-count
    /// bins of `z` the evidence kept. `1` is the homoscedastic law `c ≡ 1`.
    pub fn activity_bins(&self) -> usize {
        self.activity_bins
    }

    /// Selected factor rank `r`.
    pub fn factor_rank(&self) -> usize {
        self.factor_rank
    }

    /// The fitted interference factor `Λ ∈ ℝ^{p×r}` (the shared off-isotropic
    /// residual subspace). Consumed by the planted-subspace recovery test to
    /// compare `range(Λ)` against the planted interference subspace.
    pub fn factor(&self) -> ArrayView2<'_, f64> {
        self.lambda.view()
    }

    /// The idiosyncratic diagonal `d ∈ ℝ^p` (`D = diag(d)`).
    pub fn diagonal(&self) -> ArrayView1<'_, f64> {
        self.diagonal.view()
    }

    /// The Laplace log marginal likelihood of the selected rank (raw residual
    /// units): the likelihood integrated against the factor, diagonal and
    /// activity-law priors, evaluated at the certified posterior mode with the
    /// exact Hessian.
    pub fn log_evidence(&self) -> f64 {
        self.log_evidence
    }

    /// #2021 Λ nursery→promotion: detect *persistent, evidence-earning* factor
    /// directions relative to the previous outer-alternation pass's model.
    ///
    /// A column `j` of this model's `Λ` is a [`FactorPromotion`] candidate iff
    /// both gates hold:
    /// 1. **Earns its complexity** (evidence gate): its explained energy
    ///    `‖Λ_:,j‖² ≥ energy_floor_mult · mean(diag(D))`. Every column is already
    ///    inside the evidence-selected rank (so its rank earned its evidence
    ///    globally); this per-direction floor additionally requires the
    ///    factor to explain more than an average channel's idiosyncratic noise,
    ///    so we never promote a direction that only barely survived rank
    ///    selection.
    /// 2. **Persists** (nursery gate): its `|cos|` alignment with the best-
    ///    matching column of `prev`'s `Λ` is `≥ align_min` — the direction is the
    ///    same subspace the previous pass already found, not a new
    ///    pass-to-pass artifact.
    ///
    /// Returns candidates sorted by energy (descending). `prev = None` (the first
    /// structured pass, damping toward `I`) yields no candidates — a direction
    /// must survive at least one pass to enter the nursery. The driver holds the
    /// cross-pass persistence count (promote after it clears the direction's
    /// nursery dwell) and does the actual atom birth; this method is the pure,
    /// per-pass detector.
    ///
    /// Errors on non-finite / out-of-range gates (`align_min ∈ [0,1]`,
    /// `energy_floor_mult ≥ 0`) or a `prev` with a different output dim `p`.
    pub fn promotion_candidates(
        &self,
        prev: Option<&StructuredResidualModel>,
        align_min: f64,
        energy_floor_mult: f64,
    ) -> Result<Vec<FactorPromotion>, String> {
        if !align_min.is_finite() || !(0.0..=1.0).contains(&align_min) {
            return Err(format!(
                "StructuredResidualModel::promotion_candidates: align_min must be finite in [0,1]; got {align_min}"
            ));
        }
        if !energy_floor_mult.is_finite() || energy_floor_mult < 0.0 {
            return Err(format!(
                "StructuredResidualModel::promotion_candidates: energy_floor_mult must be finite and ≥ 0; got {energy_floor_mult}"
            ));
        }
        let prev = match prev {
            Some(pv) => pv,
            None => return Ok(Vec::new()),
        };
        if prev.p != self.p {
            return Err(format!(
                "StructuredResidualModel::promotion_candidates: prev output dim {} != {}",
                prev.p, self.p
            ));
        }
        let r = self.factor_rank;
        let prev_r = prev.factor_rank;
        if r == 0 || prev_r == 0 {
            return Ok(Vec::new());
        }
        // Idiosyncratic-noise floor: a promoted direction must explain more than
        // an average channel's independent variance.
        let mean_d = self.diagonal.iter().copied().sum::<f64>() / self.p as f64;
        let energy_floor = energy_floor_mult * mean_d;

        let mut out: Vec<FactorPromotion> = Vec::new();
        for j in 0..r {
            let col = self.lambda.column(j);
            let energy: f64 = col.iter().map(|v| v * v).sum();
            if energy <= 0.0 || energy < energy_floor {
                continue;
            }
            let norm = energy.sqrt();
            // Best |cos| against the previous pass's columns.
            let mut best_align = 0.0_f64;
            let mut best_k = 0usize;
            for k in 0..prev_r {
                let pcol = prev.lambda.column(k);
                let pnorm: f64 = pcol.iter().map(|v| v * v).sum::<f64>().sqrt();
                if pnorm <= 0.0 {
                    continue;
                }
                let dot: f64 = col.iter().zip(pcol.iter()).map(|(a, b)| a * b).sum();
                let cos_abs = (dot / (norm * pnorm)).abs();
                if cos_abs > best_align {
                    best_align = cos_abs;
                    best_k = k;
                }
            }
            if best_align >= align_min {
                out.push(FactorPromotion {
                    direction: col.mapv(|v| v / norm),
                    energy,
                    persistence_alignment: best_align,
                    prev_column: best_k,
                });
            }
        }
        out.sort_by(|a, b| b.energy.total_cmp(&a.energy));
        Ok(out)
    }

    /// Build the per-row precision factor stack `U_n ∈ ℝ^{p×p}` with
    /// `U_n U_nᵀ = Σ_n^{-1}` and package it as a
    /// [`MetricProvenance::WhitenedStructured`](gam_problem::MetricProvenance::WhitenedStructured)
    /// `RowMetric`. This is the single
    /// production site of `WhitenedStructured`.
    ///
    /// The factor comes from the SPD covariance `Σ_n = c_n·ΛΛᵀ + D` directly:
    /// `Σ_n = L_n L_nᵀ` (Cholesky) and `U_n = L_n^{-ᵀ}` (`rank = p`), so
    /// `U_n U_nᵀ = Σ_n^{-1}` exactly and `whiten_residual_row` returns
    /// `L_n^{-1} r_n`, whose squared norm is the exact Mahalanobis residual
    /// `r_nᵀ Σ_n^{-1} r_n`. The precision is never assembled (no Woodbury
    /// subtraction to cancel), and a Cholesky failure of `Σ_n` is reported as an
    /// error rather than repaired.
    pub fn row_metric(&self, n_rows: usize) -> Result<RowMetric, String> {
        if n_rows != self.row_scale.len() {
            return Err(format!(
                "StructuredResidualModel::row_metric: requested {n_rows} rows but model has {}",
                self.row_scale.len()
            ));
        }
        let p = self.p;
        // Hoist the row-INDEPENDENT Gram ΛΛᵀ out of the per-row loop: only the
        // per-row activity scale c_n multiplies it, so each row's covariance is
        // Σ_n = c_n·ΛΛᵀ + D, an O(p²) reweight of the SAME Gram.
        let gram = outer_product(&self.lambda);
        // Row-major flat factor matrix: u[n, i*p + k] = U_n[i, k]. Each row's
        // covariance assemble + p×p Cholesky is independent of every other row's
        // (no cross-row reduction), so the stack parallelizes over rows with
        // BIT-IDENTICAL output — every row runs the exact serial arithmetic and
        // writes only its own p² chunk. This was the dominant serial wall of the
        // #974 metric install (n_rows × O(p³) on one core while the inner fit
        // parallelizes cleanly). Same engagement discipline as
        // `scaled_second_moment`: only above a row threshold (serial avoids
        // rayon overhead on small stacks) and only at top level
        // (nested calls keep the outer region's cores). Error
        // selection stays deterministic: the indexed collect preserves row
        // order, and the first `Some` scanned in that order is the same
        // lowest-row error the serial loop returned.
        let mut u = Array2::<f64>::zeros((n_rows, p * p));
        let first_error = {
            use rayon::prelude::*;
            const PARALLEL_ROW_MIN: usize = 64;
            let build_row = |row: usize, urow: &mut [f64]| -> Option<String> {
                let c = self.row_scale[row].max(f64::MIN_POSITIVE);
                let mut sigma = Array2::<f64>::zeros((p, p));
                for a in 0..p {
                    for b in 0..p {
                        sigma[[a, b]] = c * gram[[a, b]];
                    }
                    sigma[[a, a]] += self.diagonal[a];
                }
                covariance_whitening_factor(&sigma, urow)
                    .map_err(|e| format!("StructuredResidualModel::row_metric row {row}: {e}"))
                    .err()
            };
            let u_flat = u.as_slice_mut().ok_or_else(|| {
                "StructuredResidualModel::row_metric: factor stack must be standard-layout"
                    .to_string()
            })?;
            if p > 0 && n_rows >= PARALLEL_ROW_MIN && gam_runtime::parallel::at_top_level() {
                gam_runtime::parallel::fan_out(|| {
                    u_flat
                        .par_chunks_mut(p * p)
                        .enumerate()
                        .map(|(row, urow)| build_row(row, urow))
                        .collect::<Vec<Option<String>>>()
                })
                .into_iter()
                .flatten()
                .next()
            } else if p > 0 {
                u_flat
                    .chunks_mut(p * p)
                    .enumerate()
                    .find_map(|(row, urow)| build_row(row, urow))
            } else {
                None
            }
        };
        if let Some(err) = first_error {
            return Err(err);
        }
        RowMetric::whitened_structured(Arc::new(u), p, p)
    }

    /// The model's isotropic (iid-MLE) dispersion: the per-coordinate average of
    /// its own fitted total residual variance,
    /// ```text
    ///   φ̂ = (1/p) · mean_row tr(Σ̂(row)) = mean(c)·‖Λ‖²_F / p + mean(d).
    /// ```
    /// This is the honest single-scalar summary of the SAME second moment the
    /// structured model fits — the scale an iid Gaussian residual model would
    /// estimate from these residuals. Used as the first-pass damping anchor in
    /// [`Self::row_metric_damped`] (#2243 cap #2): anchoring at `φ̂·I` instead of
    /// the unit `I_p` means near-noiseless data is whitened by its MEASURED
    /// noise, so the downstream unit-dispersion REML criterion prices the
    /// smoothing penalty against the real dispersion rather than an assumed
    /// unit one. Floored at `f64::MIN_POSITIVE` so the blend stays SPD.
    pub(crate) fn isotropic_dispersion(&self) -> f64 {
        let p = self.p.max(1) as f64;
        let n = self.row_scale.len().max(1) as f64;
        let mean_c = self.row_scale.iter().copied().sum::<f64>() / n;
        let lambda_energy: f64 = self.lambda.iter().map(|v| v * v).sum();
        let mean_d = self.diagonal.iter().copied().sum::<f64>() / p;
        (mean_c * lambda_energy / p + mean_d).max(f64::MIN_POSITIVE)
    }

    /// Damped per-row metric for the #2021 driver: blend covariances in the
    /// **covariance domain** (before the Cholesky whitening) between this model's
    /// estimate and a previous one,
    /// ```text
    ///   Σ_t(row) = (1 − γ) · Σ_prev(row) + γ · Σ̂_t(row),
    /// ```
    /// where `Σ̂_t(row) = c_t(z)·ΛΛᵀ + D` is this model's per-row covariance
    /// (built from the hoisted-Gram / occupancy-weighted `c(z)` path), and
    /// `Σ_prev(row)` is `prev`'s per-row covariance when `Some`, else the
    /// MEASURED iid anchor `φ̂·I_p` (`Self::isotropic_dispersion`, #2243 cap
    /// #2: a unit `I_p` anchor silently assumed unit noise, which on
    /// near-noiseless data pinned the whitened likelihood — and therefore the
    /// REML smoothing balance — at a noise scale ~1/φ̂ too coarse, i.e. the
    /// clean-data over-penalization).
    ///
    /// Endpoints (exact):
    /// * `γ = 1.0` ⇒ this model's [`Self::row_metric`] exactly (byte-identical);
    /// * `γ = 0.0` ⇒ `prev`'s [`Self::row_metric`] when `Some` (byte-identical),
    ///   else the measured-scale identity `(φ̂·I)^{-1}` factors.
    ///
    /// `γ` must be finite and in `[0, 1]`; when `prev` is `Some` it must share
    /// this model's `p` and row count.
    pub fn row_metric_damped(
        &self,
        n_rows: usize,
        gamma: f64,
        prev: Option<&StructuredResidualModel>,
    ) -> Result<RowMetric, String> {
        if n_rows != self.row_scale.len() {
            return Err(format!(
                "StructuredResidualModel::row_metric_damped: requested {n_rows} rows but model has {}",
                self.row_scale.len()
            ));
        }
        if !gamma.is_finite() || !(0.0..=1.0).contains(&gamma) {
            return Err(format!(
                "StructuredResidualModel::row_metric_damped: gamma must be finite in [0,1]; got {gamma}"
            ));
        }
        if let Some(pv) = prev {
            if pv.p != self.p {
                return Err(format!(
                    "StructuredResidualModel::row_metric_damped: prev output dim {} != {}",
                    pv.p, self.p
                ));
            }
            if pv.row_scale.len() != n_rows {
                return Err(format!(
                    "StructuredResidualModel::row_metric_damped: prev has {} rows but requested {n_rows}",
                    pv.row_scale.len()
                ));
            }
        }
        // Exact endpoints — reuse the undamped producers so the result is
        // byte-identical (γ=1 ⇒ this model; γ=0 ⇒ prev, or Euclidean identity).
        if gamma == 1.0 {
            return self.row_metric(n_rows);
        }
        if gamma == 0.0 {
            if let Some(pv) = prev {
                return pv.row_metric(n_rows);
            }
            // prev = None falls through to the general path: the blend is then
            // exactly the measured-scale identity `φ̂·I` (#2243 cap #2), not the
            // unit Euclidean identity.
        }

        let p = self.p;
        // #2243 cap #2 — the first-pass (prev = None) damping anchor is the
        // model's own measured isotropic dispersion, hoisted out of the row loop.
        let iid_anchor = self.isotropic_dispersion();
        // Row-INDEPENDENT outer products ΛΛᵀ (this model and, if present, prev):
        // only the per-row activity scale c(z) multiplies them, so hoist the Gram
        // out of the per-row loop (mirroring the row_metric hoist).
        let self_gram = outer_product(&self.lambda);
        let prev_gram = prev.map(|pv| outer_product(&pv.lambda));

        // Per-row blend + Cholesky whitening, parallelized over rows exactly as
        // in [`Self::row_metric`]: rows are independent (each writes only its
        // own p² chunk of `u`, no cross-row reduction), so the parallel stack is
        // bit-identical to the serial one, with the same engagement discipline
        // (row threshold, only at top level, never nested) and the same
        // deterministic lowest-row error selection.
        let mut u = Array2::<f64>::zeros((n_rows, p * p));
        let first_error = {
            use rayon::prelude::*;
            const PARALLEL_ROW_MIN: usize = 64;
            let build_row = |row: usize, urow: &mut [f64]| -> Option<String> {
                self.damped_row_factor(
                    row,
                    gamma,
                    prev,
                    iid_anchor,
                    &self_gram,
                    prev_gram.as_ref(),
                    urow,
                )
                .err()
            };
            let u_flat = u
                .as_slice_mut()
                .ok_or_else(|| "StructuredResidualModel::row_metric_damped: factor stack must be standard-layout".to_string())?;
            if p > 0 && n_rows >= PARALLEL_ROW_MIN && gam_runtime::parallel::at_top_level() {
                gam_runtime::parallel::fan_out(|| {
                    u_flat
                        .par_chunks_mut(p * p)
                        .enumerate()
                        .map(|(row, urow)| build_row(row, urow))
                        .collect::<Vec<Option<String>>>()
                })
                .into_iter()
                .flatten()
                .next()
            } else if p > 0 {
                u_flat
                    .chunks_mut(p * p)
                    .enumerate()
                    .find_map(|(row, urow)| build_row(row, urow))
            } else {
                None
            }
        };
        if let Some(err) = first_error {
            return Err(err);
        }
        RowMetric::whitened_structured(Arc::new(u), p, p)
    }

    /// One row of the damped-metric stack: assemble the convex-blend covariance
    /// `Σ_t(row) = γ·(c·ΛΛᵀ + D) + (1−γ)·Σ_prev(row)`, symmetrize, and write
    /// its whitening factor `U = L^{-ᵀ}` (`Σ_t = L Lᵀ`) into `urow` (row-major
    /// `p×p`).
    /// Factored out of [`Self::row_metric_damped`] so the serial and parallel
    /// row drivers share one arithmetic body. `iid_anchor` is the measured
    /// isotropic dispersion `φ̂` used as `Σ_prev = φ̂·I` when `prev` is `None`
    /// (#2243 cap #2).
    fn damped_row_factor(
        &self,
        row: usize,
        gamma: f64,
        prev: Option<&StructuredResidualModel>,
        iid_anchor: f64,
        self_gram: &Array2<f64>,
        prev_gram: Option<&Array2<f64>>,
        urow: &mut [f64],
    ) -> Result<(), String> {
        let p = self.p;
        let c = self.row_scale[row].max(f64::MIN_POSITIVE);
        // γ · Σ̂_t = γ·(c·ΛΛᵀ + D).
        let mut sigma = Array2::<f64>::zeros((p, p));
        for a in 0..p {
            for b in 0..p {
                sigma[[a, b]] = gamma * c * self_gram[[a, b]];
            }
            sigma[[a, a]] += gamma * self.diagonal[a];
        }
        // (1−γ) · Σ_prev  (prev's per-row Σ, or I_p when prev is None).
        match (prev, prev_gram) {
            (Some(pv), Some(pg)) => {
                let cp = pv.row_scale[row].max(f64::MIN_POSITIVE);
                for a in 0..p {
                    for b in 0..p {
                        sigma[[a, b]] += (1.0 - gamma) * cp * pg[[a, b]];
                    }
                    sigma[[a, a]] += (1.0 - gamma) * pv.diagonal[a];
                }
            }
            (None, _) => {
                // First-pass anchor: the MEASURED iid dispersion φ̂·I, not the
                // unit identity (#2243 cap #2 — clean-data over-penalization).
                for a in 0..p {
                    sigma[[a, a]] += (1.0 - gamma) * iid_anchor;
                }
            }
            (Some(_), None) => {
                return Err(
                    "previous residual model supplied without its precomputed gram".to_string()
                );
            }
        }
        // Symmetrize against round-off before the Cholesky.
        for a in 0..p {
            for b in (a + 1)..p {
                let avg = 0.5 * (sigma[[a, b]] + sigma[[b, a]]);
                sigma[[a, b]] = avg;
                sigma[[b, a]] = avg;
            }
        }
        // Σ_t is a convex combination of SPD matrices (D ≻ 0 / φ̂·I ≻ 0) ⇒ SPD,
        // so it is whitened through its own Cholesky exactly as in
        // [`Self::row_metric`].
        covariance_whitening_factor(&sigma, urow).map_err(|e| {
            format!("StructuredResidualModel::row_metric_damped row {row}: {e}")
        })
    }
}

/// Outer product `Λ Λᵀ ∈ ℝ^{p×p}` of a factor matrix `Λ ∈ ℝ^{p×r}` — the
/// row-independent factor covariance the per-row activity scale multiplies.
/// Used by [`StructuredResidualModel::row_metric_damped`] to hoist the Gram out
/// of its per-row covariance-blend loop.
fn outer_product(lambda: &Array2<f64>) -> Array2<f64> {
    let p = lambda.nrows();
    let r = lambda.ncols();
    let mut g = Array2::<f64>::zeros((p, p));
    for a in 0..p {
        for b in 0..p {
            let mut acc = 0.0_f64;
            for k in 0..r {
                acc += lambda[[a, k]] * lambda[[b, k]];
            }
            g[[a, b]] = acc;
        }
    }
    g
}

/// Whitening factor of an SPD covariance: Cholesky `Σ = L Lᵀ`, then write
/// `U = L^{-ᵀ}` into `urow` (row-major `p×p`, `urow[i*p + k] = U[i, k]`), so
/// `U Uᵀ = L^{-ᵀ} L^{-1} = Σ^{-1}` and `Uᵀ r = L^{-1} r`. `L^{-1}` is formed
/// column by column by forward substitution on `L x = e_j`. The precision is
/// never assembled, and a covariance that does not factor is an error (the
/// fitted model guarantees `Σ ≻ 0`, so a failure is a genuine defect upstream).
fn covariance_whitening_factor(sigma: &Array2<f64>, urow: &mut [f64]) -> Result<(), String> {
    let p = sigma.nrows();
    if urow.len() != p * p {
        return Err(format!(
            "whitening factor buffer holds {} entries for a {p}x{p} covariance",
            urow.len()
        ));
    }
    let l = sigma
        .cholesky(Side::Lower)
        .map_err(|e| format!("residual covariance is not SPD: {e:?}"))?
        .lower_triangular();
    urow.fill(0.0);
    let mut x = vec![0.0_f64; p];
    for j in 0..p {
        // Column j of L^{-1}: x_i = 0 for i < j (L^{-1} is lower triangular).
        for i in j..p {
            let mut acc = if i == j { 1.0 } else { 0.0 };
            for k in j..i {
                acc -= l[[i, k]] * x[k];
            }
            x[i] = acc / l[[i, i]];
        }
        // U = (L^{-1})ᵀ ⇒ U[j, i] = L^{-1}[i, j].
        for i in j..p {
            urow[j * p + i] = x[i];
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    fn lcg_uniform(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn lcg_normal(state: &mut u64) -> f64 {
        // `lcg_uniform` is in [0, 1); its complement is in (0, 1], so `ln u1` is finite.
        let u1 = 1.0 - lcg_uniform(state);
        let u2 = lcg_uniform(state);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// The planted single-factor activity-law DGP: `r_n = √c(z_n)·λ₀·f_n + ε_n`
    /// with `c(z) = e^{1.3 z}` and `z` uniform on `[0, 1]`.
    fn planted_rank_one_activity() -> (Array2<f64>, Array1<f64>) {
        let n = 5000usize;
        let p = 4usize;
        let lambda0 = ndarray::array![1.5, 1.2, -0.4, 0.3];
        let sigma_eps = 0.2_f64;
        let slope = 1.3_f64;
        let mut seed = 0xD1B54A32D192ED03_u64;
        let mut residuals = Array2::<f64>::zeros((n, p));
        let mut activity = Array1::<f64>::zeros(n);
        for row in 0..n {
            let z = (row as f64) / (n as f64 - 1.0);
            activity[row] = z;
            let amp = (slope * z).exp().sqrt();
            let f = lcg_normal(&mut seed);
            for i in 0..p {
                residuals[[row, i]] = amp * lambda0[i] * f + sigma_eps * lcg_normal(&mut seed);
            }
        }
        (residuals, activity)
    }

    /// The best evidence over every rank at one resolution, with the rank that
    /// carries it: the rung `fit`'s ladder scores.
    fn rung_evidence(
        residuals: ArrayView2<'_, f64>,
        activity: &Array1<f64>,
        bins: usize,
    ) -> (usize, f64) {
        let row_bin = activity_bin_assignment(activity.view(), bins);
        let moments = evidence::binned_moments(residuals, &row_bin, bins).expect("moments");
        let max_rank = evidence::ledermann_bound(residuals.ncols());
        (0..=max_rank)
            .map(|rank| {
                (
                    rank,
                    evidence::fit_rank(&moments, rank).expect("rank fit").log_evidence,
                )
            })
            .fold((0usize, f64::NEG_INFINITY), |best, cand| {
                if cand.1 > best.1 { cand } else { best }
            })
    }

    /// Per-rank evidence on the planted single-factor activity-law DGP: the
    /// planted rank 1 must carry strictly larger evidence than the null rank 0,
    /// and `fit` must select it. For `p = 4` the Ledermann bound is 1, so these
    /// are the only identifiable ranks.
    #[test]
    fn evidence_prefers_planted_rank_one() {
        let (residuals, activity) = planted_rank_one_activity();
        assert_eq!(evidence::ledermann_bound(4), 1);
        let model = StructuredResidualModel::fit(ResidualFactorInput {
            residuals: residuals.view(),
            activity: activity.view(),
        })
        .expect("fit");
        assert_eq!(model.factor_rank(), 1);

        // At the resolution the fit selected, the planted rank 1 must beat the
        // null rank 0, and the fit's own evidence must be that rank's.
        let bins = model.activity_bins();
        let row_bin = activity_bin_assignment(activity.view(), bins);
        let moments =
            evidence::binned_moments(residuals.view(), &row_bin, bins).expect("moments");
        let ev: Vec<f64> = (0..=1usize)
            .map(|rank| evidence::fit_rank(&moments, rank).expect("rank fit").log_evidence)
            .collect();
        assert!(
            ev[1] > ev[0],
            "evidence must prefer the planted rank 1 at the selected {bins}-bin \
             resolution: log Z_0 = {:.3}, log Z_1 = {:.3}",
            ev[0],
            ev[1]
        );
        assert_eq!(model.log_evidence(), ev[1]);
    }

    /// #3337: the activity law's resolution is read from the evidence, not
    /// fixed. On the planted `c(z) = e^{1.3 z}` DGP the search must leave the
    /// null resolution, and the resolution it keeps must be the one the ladder
    /// actually maximizes: strictly better than the rung below it, and not
    /// beaten by the rung above it (the rung the ladder stopped at).
    #[test]
    fn the_activity_resolution_is_the_evidence_maximizer_3337() {
        let (residuals, activity) = planted_rank_one_activity();
        let model = StructuredResidualModel::fit(ResidualFactorInput {
            residuals: residuals.view(),
            activity: activity.view(),
        })
        .expect("fit");
        let bins = model.activity_bins();
        assert!(
            bins > 1,
            "a planted activity law must buy more than the null resolution; got {bins} bin(s)"
        );
        assert!(bins.is_power_of_two(), "the ladder doubles; got {bins}");

        let here = rung_evidence(residuals.view(), &activity, bins);
        assert_eq!(model.log_evidence(), here.1);
        assert_eq!(model.factor_rank(), here.0);
        let below = rung_evidence(residuals.view(), &activity, bins / 2);
        assert!(
            here.1 > below.1,
            "the kept resolution must beat the rung below it: {bins} bins gives \
             {:.4} against {:.4} at {} bins",
            here.1,
            below.1,
            bins / 2
        );
        let above = rung_evidence(residuals.view(), &activity, bins * 2);
        assert!(
            above.0 == 0 || above.1 <= here.1,
            "the ladder stopped at {bins} bins, so the next rung must carry no \
             activity law or not beat it: {} bins gives rank {} at {:.4} against \
             {:.4}",
            bins * 2,
            above.0,
            above.1,
            here.1
        );
    }

    /// #3337: a constant activity coordinate carries no resolution to buy, and
    /// the law is the null `c ≡ 1` on one bin. This is the recoverability the
    /// ladder has to keep: the search starts at the null and never leaves it
    /// when `z` cannot separate a single row.
    #[test]
    fn a_constant_activity_coordinate_keeps_the_null_law_3337() {
        let n = 400usize;
        let p = 3usize;
        let mut seed = 0x1234_5678_9ABC_DEF0_u64;
        let mut residuals = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            for j in 0..p {
                residuals[[row, j]] = (0.4 + 0.2 * j as f64) * lcg_normal(&mut seed);
            }
        }
        let activity = Array1::<f64>::zeros(n);
        let model = StructuredResidualModel::fit(ResidualFactorInput {
            residuals: residuals.view(),
            activity: activity.view(),
        })
        .expect("fit");
        assert_eq!(model.activity_bins(), 1);
        assert!(model.row_scale.iter().all(|&c| c == 1.0));
    }

    /// #3337: the bin assignment is equal count, tie-safe, monotone in `z`, and
    /// nested under doubling — the four properties the resolution ladder rests
    /// on. Nesting is what makes rung `2B`'s law contain rung `B`'s, so the
    /// evidence comparison between them is a refinement comparison.
    #[test]
    fn the_activity_bins_are_equal_count_and_nested_3337() {
        let n = 96usize;
        // Heavily tied, unevenly spread: equal-width bins would put most rows
        // in one bin.
        let z = Array1::from_iter((0..n).map(|i| ((i % 12) as f64 / 11.0).powi(3)));
        for bins in [2usize, 4, 8, 16] {
            let coarse = activity_bin_assignment(z.view(), bins);
            let fine = activity_bin_assignment(z.view(), 2 * bins);
            for i in 0..n {
                for k in 0..n {
                    // Tie-safe and monotone: the bin is a function of z alone,
                    // and it does not decrease as z increases.
                    if z[i] == z[k] {
                        assert_eq!(coarse[i], coarse[k], "tied z must share a bin");
                    }
                    if z[i] < z[k] {
                        assert!(coarse[i] <= coarse[k], "bins must be monotone in z");
                    }
                    // Nested: rows together at 2B were together at B.
                    if fine[i] == fine[k] {
                        assert_eq!(
                            coarse[i], coarse[k],
                            "the {}-bin partition must refine the {bins}-bin one",
                            2 * bins
                        );
                    }
                }
            }
            let occupied = occupied_bin_count(&coarse, bins);
            assert!(occupied >= 1 && occupied <= bins);
        }
    }

    /// #3337: with no tie to hold rows together, the assignment is exactly
    /// equal count — every bin holds `⌊n/B⌋` or `⌈n/B⌉` rows, for every `B` up
    /// to one row per bin. That is the property that makes a bin's Fisher
    /// information for its own scale the same as every other bin's at the null
    /// law, which is what the one resolution number means.
    #[test]
    fn tie_free_activity_bins_hold_equal_counts_3337() {
        let n = 97usize; // prime, so no B divides it evenly
        let z = Array1::from_iter((0..n).map(|i| (i as f64) * 0.37 - 5.0));
        for bins in 1..=n {
            let row_bin = activity_bin_assignment(z.view(), bins);
            let mut counts = vec![0usize; bins];
            for &b in &row_bin {
                counts[b] += 1;
            }
            let floor = n / bins;
            let ceil = n.div_ceil(bins);
            assert_eq!(counts.iter().sum::<usize>(), n, "every row is binned");
            for (b, &count) in counts.iter().enumerate() {
                assert!(
                    count == floor || count == ceil,
                    "bin {b} of {bins} holds {count} rows, not {floor} or {ceil}"
                );
            }
        }
    }

    /// The fitted model is the posterior mode, so its likelihood score in each
    /// log-diagonal `η_j = ln d_j` is exactly balanced by the prior's: with
    /// `W_n = Σ_n⁻¹`, the likelihood score is
    /// `½ d_j Σ_n [(W_n r_n)_j² − (W_n)_jj]`, and the matrix-Cauchy prior's
    /// η-score lies in `[−r/2, p/2)`, so at the mode the likelihood score is
    /// bounded by `(p + 2r)/2` whatever `n` is. The retired alternation stopped
    /// at a state whose η-scores reached −641 on this DGP (it was not a
    /// stationary point of anything); this computes the score from scratch, per
    /// row, independently of the fitting code.
    #[test]
    fn fitted_diagonal_is_stationary_for_the_likelihood() {
        let (residuals, activity) = planted_rank_one_activity();
        let model = StructuredResidualModel::fit(ResidualFactorInput {
            residuals: residuals.view(),
            activity: activity.view(),
        })
        .expect("fit");
        let (n, p) = residuals.dim();
        let rank = model.factor_rank();
        assert_eq!(rank, 1);
        let lambda = model.factor().to_owned();
        let d = model.diagonal().to_owned();
        let mut score = Array1::<f64>::zeros(p);
        for row in 0..n {
            let mut sigma = lambda.dot(&lambda.t()) * model.row_scale[row];
            for j in 0..p {
                sigma[[j, j]] += d[j];
            }
            let w = sigma
                .cholesky(Side::Lower)
                .expect("Σ_n SPD")
                .solve_mat(&Array2::<f64>::eye(p));
            let wr = w.dot(&residuals.row(row));
            for j in 0..p {
                score[j] += 0.5 * d[j] * (wr[j] * wr[j] - w[[j, j]]);
            }
        }
        let bound = (p + 2 * rank) as f64 / 2.0;
        for j in 0..p {
            assert!(
                score[j].abs() <= bound,
                "likelihood η-score {j} = {} exceeds the prior's reach {bound} at the fitted \
                 mode; scores = {score:?}",
                score[j]
            );
        }
        let row_mean_scale = model.row_scale.sum() / n as f64;
        assert!(
            (row_mean_scale - 1.0).abs() <= 1e-12,
            "the activity law must have row mean one; got {row_mean_scale}"
        );
    }

    /// Null recoverability: independent heteroscedastic channels with a varying
    /// activity coordinate carry no factor, and the evidence must return the
    /// pure-diagonal model on every seed, with the whole identifiable ladder
    /// (`p = 6`: ranks 0‥3) open to it.
    #[test]
    fn evidence_selects_rank_zero_under_the_null() {
        let n = 2000usize;
        let p = 6usize;
        for seed0 in [0x3C6EF372FE94F82B_u64, 0xA54FF53A5F1D36F1, 0x510E527FADE682D1] {
            let mut seed = seed0;
            let mut residuals = Array2::<f64>::zeros((n, p));
            let mut activity = Array1::<f64>::zeros(n);
            for row in 0..n {
                activity[row] = lcg_uniform(&mut seed);
                for j in 0..p {
                    residuals[[row, j]] = (0.5 + 0.3 * j as f64) * lcg_normal(&mut seed);
                }
            }
            let model = StructuredResidualModel::fit(ResidualFactorInput {
                residuals: residuals.view(),
                activity: activity.view(),
            })
            .expect("fit");
            assert_eq!(
                model.factor_rank(),
                0,
                "seed {seed0:#x}: the null carries no factor, but rank {} was selected \
                 (log Z = {:.3})",
                model.factor_rank(),
                model.log_evidence()
            );
            assert!(model.row_scale.iter().all(|&c| c == 1.0));
        }
    }

    /// Orthonormalize the columns of `m` (modified Gram–Schmidt), dropping
    /// numerically-null columns. Test-side helper for subspace comparisons.
    fn orthonormal_columns(m: ArrayView2<'_, f64>) -> Vec<Array1<f64>> {
        let mut basis: Vec<Array1<f64>> = Vec::new();
        for k in 0..m.ncols() {
            let mut v = m.column(k).to_owned();
            for q in &basis {
                let c = v.dot(q);
                v = &v - &(q * c);
            }
            let norm = v.dot(&v).sqrt();
            if norm > 1e-10 {
                basis.push(v / norm);
            }
        }
        basis
    }

    /// Squared norm of the projection of unit vector `v` onto span(basis) —
    /// `cos²` of the principal angle between `v` and the subspace.
    fn projection_energy(v: &Array1<f64>, basis: &[Array1<f64>]) -> f64 {
        basis.iter().map(|q| v.dot(q).powi(2)).sum()
    }

    /// #974 verification arm (a): the fitted factor must recover the PLANTED
    /// interference subspace. Two orthogonal planted directions with distinct
    /// strengths; the principal angles between each planted direction and
    /// range(Λ̂) must be small, and the evidence must select rank 2.
    #[test]
    fn factor_recovers_planted_interference_subspace() {
        let n = 6000usize;
        let p = 6usize;
        // Two orthogonal planted unit directions.
        let raw1: Array1<f64> = ndarray::array![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let raw2: Array1<f64> = ndarray::array![1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let v1 = &raw1 / raw1.dot(&raw1).sqrt();
        let v2 = &raw2 / raw2.dot(&raw2).sqrt();
        let (amp1, amp2) = (1.4_f64, 0.9_f64);
        let sigma_eps = 0.15_f64;

        let mut seed = 0x9E3779B97F4A7C15_u64;
        let mut residuals = Array2::<f64>::zeros((n, p));
        let activity = Array1::<f64>::zeros(n); // constant ⇒ homoscedastic law
        for row in 0..n {
            let f1 = amp1 * lcg_normal(&mut seed);
            let f2 = amp2 * lcg_normal(&mut seed);
            for i in 0..p {
                residuals[[row, i]] = f1 * v1[i] + f2 * v2[i] + sigma_eps * lcg_normal(&mut seed);
            }
        }

        let model = StructuredResidualModel::fit(ResidualFactorInput {
            residuals: residuals.view(),
            activity: activity.view(),
        })
        .expect("fit");

        assert_eq!(
            model.factor_rank(),
            2,
            "evidence must select the planted rank 2 (got {}, log Z {:.3})",
            model.factor_rank(),
            model.log_evidence()
        );
        let basis = orthonormal_columns(model.factor());
        assert_eq!(basis.len(), 2, "fitted factor must span 2 directions");
        let e1 = projection_energy(&v1, &basis);
        let e2 = projection_energy(&v2, &basis);
        // cos² of each principal angle ≥ 0.95 ⇒ angle ≤ ~13°.
        assert!(
            e1 > 0.95 && e2 > 0.95,
            "planted directions must lie in range(Λ̂): cos² = ({e1:.4}, {e2:.4})"
        );
    }


    /// Fit a small structured model on a planted single-factor DGP — shared
    /// fixture builder for the producer / damped-metric integration tests.
    fn fit_small_model(seed0: u64, lambda0: &Array1<f64>) -> (usize, StructuredResidualModel) {
        let n = 300usize;
        let p = lambda0.len();
        let sigma_eps = 0.25_f64;
        let slope = 1.4_f64;
        let mut seed = seed0;
        let mut residuals = Array2::<f64>::zeros((n, p));
        let mut activity = Array1::<f64>::zeros(n);
        for row in 0..n {
            let u = (row as f64) / (n as f64 - 1.0);
            let z = u * u;
            activity[row] = z;
            let amp = (slope * z).exp().sqrt();
            let f = lcg_normal(&mut seed);
            for i in 0..p {
                residuals[[row, i]] = amp * lambda0[i] * f + sigma_eps * lcg_normal(&mut seed);
            }
        }
        let model = StructuredResidualModel::fit(ResidualFactorInput {
            residuals: residuals.view(),
            activity: activity.view(),
        })
        .expect("fit");
        (n, model)
    }

    /// WAVE-2 #2021 damped metric endpoint contracts: γ=1 ≡ row_metric (ignores
    /// prev), γ=0 ≡ prev.row_metric (or Euclidean identity), 0<γ<1 is SPD, and
    /// out-of-range / non-finite γ is rejected.
    #[test]
    fn row_metric_damped_endpoints() {
        let lambda_a = ndarray::array![1.5, 1.2, -0.4, 0.3];
        let lambda_b = ndarray::array![-0.6, 1.1, 0.9, -1.3];
        let (n, model) = fit_small_model(0x51ED270B_u64 ^ 0xF3A5C7D1_u64, &lambda_a);
        let (n_prev, prev) = fit_small_model(0x2545F491_u64 ^ 0x4F6CDD1D_u64, &lambda_b);
        assert_eq!(n, n_prev);
        let p = 4usize;
        let v: Array1<f64> = ndarray::array![0.7, -1.3, 0.4, 0.9];

        // γ = 1 ⇒ byte-identical to this model's row_metric, regardless of prev.
        let base = model.row_metric(n).expect("row_metric");
        for prev_opt in [None, Some(&prev)] {
            let damped = model
                .row_metric_damped(n, 1.0, prev_opt)
                .expect("damped γ=1");
            for row in [0usize, n / 2, n - 1] {
                for i in 0..p {
                    for k in 0..p {
                        assert_eq!(
                            damped.factor_entry(row, i, k),
                            base.factor_entry(row, i, k),
                            "γ=1 must be byte-identical to row_metric at ({row},{i},{k})"
                        );
                    }
                }
            }
        }

        // γ = 0, prev = None ⇒ the MEASURED-scale identity (#2243 cap #2):
        // Σ = φ̂·I with φ̂ the model's own isotropic dispersion, so
        // quad_form = ‖v‖² / φ̂ (a unit-I anchor would silently assume unit
        // noise and over-penalize clean data).
        let ident = model
            .row_metric_damped(n, 0.0, None)
            .expect("damped γ=0 None");
        let phi = model.isotropic_dispersion();
        assert!(phi.is_finite() && phi > 0.0, "φ̂ must be positive");
        let sumsq: f64 = v.iter().map(|x| x * x).sum();
        let expected = sumsq / phi;
        for row in [0usize, n / 2, n - 1] {
            let q = ident.quad_form(row, v.view());
            assert!(
                (q - expected).abs() <= 1e-9 * (1.0 + expected),
                "γ=0/None must be the measured-scale identity: quad_form {q} vs ‖v‖²/φ̂ {expected}"
            );
        }

        // γ = 0, prev = Some ⇒ byte-identical to prev.row_metric.
        let prev_metric = prev.row_metric(n).expect("prev row_metric");
        let damped0 = model
            .row_metric_damped(n, 0.0, Some(&prev))
            .expect("damped γ=0 Some");
        for row in [0usize, n / 2, n - 1] {
            for i in 0..p {
                for k in 0..p {
                    assert_eq!(
                        damped0.factor_entry(row, i, k),
                        prev_metric.factor_entry(row, i, k),
                        "γ=0/Some must be byte-identical to prev.row_metric at ({row},{i},{k})"
                    );
                }
            }
        }

        // 0 < γ < 1 ⇒ valid SPD metric.
        let mid = model
            .row_metric_damped(n, 0.5, Some(&prev))
            .expect("damped γ=0.5");
        for row in [0usize, n / 2, n - 1] {
            let q = mid.quad_form(row, v.view());
            assert!(
                q.is_finite() && q > 0.0,
                "γ=0.5 metric must be SPD; got {q}"
            );
        }

        // Invalid γ rejected.
        assert!(model.row_metric_damped(n, 1.5, None).is_err());
        assert!(model.row_metric_damped(n, -0.1, None).is_err());
        assert!(model.row_metric_damped(n, f64::NAN, None).is_err());
    }

    /// WAVE-2 #2021 Λ nursery→promotion: `promotion_candidates` must fire only
    /// for a factor that BOTH persists across passes (aligns with the previous
    /// model's Λ) AND clears the idiosyncratic-noise energy floor; a fresh
    /// orthogonal direction, an over-high energy floor, and `prev = None` all
    /// yield no candidates, and out-of-range gates are rejected.
    #[test]
    fn promotion_candidates_gates_on_persistence_and_energy() {
        let lambda_a = ndarray::array![1.5, 1.2, -0.4, 0.3];
        let lambda_b = ndarray::array![-0.6, 1.1, 0.9, -1.3];
        // Same planted direction across two passes ⇒ persistent.
        let (_, prev) = fit_small_model(0xA1B2C3D4_u64 ^ 0x0F0F0F0F_u64, &lambda_a);
        let (_, cur) = fit_small_model(0x5566778899AABBCC_u64, &lambda_a);
        // A different (well-separated) planted direction ⇒ NOT aligned with cur.
        let (_, other) = fit_small_model(0x1122334455667788_u64, &lambda_b);

        assert!(prev.factor_rank() >= 1 && cur.factor_rank() >= 1 && other.factor_rank() >= 1);

        // Persistent + energetic ⇒ at least one candidate, aligned with the
        // planted direction and above the noise floor.
        let cands = cur
            .promotion_candidates(Some(&prev), 0.9, 1.0)
            .expect("promotion_candidates");
        assert!(
            !cands.is_empty(),
            "a persistent, energetic factor must yield a promotion candidate"
        );
        let top = &cands[0];
        assert!(
            top.persistence_alignment >= 0.9,
            "top candidate must clear the alignment gate; got {}",
            top.persistence_alignment
        );
        // The promoted unit direction must align with the planted (unit) lambda_a.
        let la_norm = lambda_a.dot(&lambda_a).sqrt();
        let la_unit = lambda_a.mapv(|v| v / la_norm);
        let dir_cos = top.direction.dot(&la_unit).abs();
        assert!(
            dir_cos > 0.9,
            "promoted direction must recover the planted factor; |cos| = {dir_cos:.4}"
        );
        assert!(
            (top.direction.dot(&top.direction) - 1.0).abs() < 1e-10,
            "promoted direction must be unit-norm"
        );
        assert!(top.energy > 0.0);

        // A fresh, well-separated direction does NOT persist ⇒ no candidate at 0.9.
        let cross = cur
            .promotion_candidates(Some(&other), 0.9, 1.0)
            .expect("promotion_candidates cross");
        assert!(
            cross.is_empty(),
            "a non-persistent (unaligned) factor must not be promoted; got {} candidate(s)",
            cross.len()
        );

        // An over-high energy floor rejects even the persistent factor.
        let floored = cur
            .promotion_candidates(Some(&prev), 0.9, 1.0e6)
            .expect("promotion_candidates floored");
        assert!(
            floored.is_empty(),
            "energy floor must gate out factors below the noise-scaled threshold"
        );

        // prev = None (first structured pass, damping toward I) ⇒ no candidates.
        assert!(cur.promotion_candidates(None, 0.9, 1.0).unwrap().is_empty());

        // Invalid gates rejected.
        assert!(cur.promotion_candidates(Some(&prev), 1.5, 1.0).is_err());
        assert!(cur.promotion_candidates(Some(&prev), -0.1, 1.0).is_err());
        assert!(cur.promotion_candidates(Some(&prev), 0.9, -1.0).is_err());
        assert!(
            cur.promotion_candidates(Some(&prev), f64::NAN, 1.0)
                .is_err()
        );
    }
}