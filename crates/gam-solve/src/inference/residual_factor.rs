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
//! * `c(z) > 0` is the **smooth activity-scale law**: a strictly-positive scalar
//!   that modulates the factor energy with the activity coordinate, recovered as
//!   a binned-then-smoothed function of `z`.
//!
//! The fit is a deterministic, fixed-iteration **alternation** (no clock, no
//! RNG; any tie is broken by index): it alternates
//!
//! 1. *(scale | Λ, D)* — re-estimate the per-row factor activity `c(z_n)` and
//!    smooth it across `z`, holding the factor model fixed; and
//! 2. *(Λ, D | scale)* — re-estimate the factor and diagonal from the
//!    scale-deflated second-moment, holding the activity law fixed,
//!
//! a fixed small number of times. The **factor count `r`** is chosen by an
//! evidence ladder: each candidate `r` is scored by its penalized Gaussian
//! log-evidence and the best is kept.
//!
//! # What it produces
//!
//! [`StructuredResidualModel::row_metric`] materializes the **per-row precision
//! factor** `U_n ∈ ℝ^{p×p}` with `U_n U_nᵀ = Σ_n^{-1}`, packaged as a
//! [`RowMetric`](gam_problem::RowMetric) with
//! [`MetricProvenance::WhitenedStructured`](gam_problem::MetricProvenance::WhitenedStructured).
//! Whitening a residual `r_n` through it (`U_nᵀ r_n`) yields a vector whose
//! squared Euclidean norm is `r_nᵀ Σ_n^{-1} r_n` — the Mahalanobis residual under
//! the estimated noise model, which is exactly the likelihood-correct data-fit.
//! The factor is built from `Σ_n^{-1}` computed in **Woodbury form** (an
//! `r × r` solve, never a `p × p` inverse), so the estimator scales with the
//! factor rank, not the dense output dimension.
//!
//! This is the first real producer of `WhitenedStructured`, and therefore the
//! first metric whose `whitens_likelihood()` is `true`: see
//! [`RowMetric::whitens_likelihood`](gam_problem::RowMetric::whitens_likelihood).

use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use gam_problem::RowMetric;
use gam_linalg::faer_ndarray::{FaerCholesky, FaerEigh};
use faer::Side;

/// Number of (scale | factor) ↔ (factor | scale) alternation sweeps. Fixed and
/// deterministic: the alternation is a smooth descent on the structured-Gaussian
/// objective and converges geometrically, so a small fixed budget is both
/// sufficient and reproducible (no clock/RNG-driven stopping).
const ALTERNATION_SWEEPS: usize = 8;

/// Number of bins the activity coordinate `z` is partitioned into for the smooth
/// activity-scale `c(z)`. The per-bin factor activity is estimated then linearly
/// interpolated across bin centers, giving a continuous piecewise-linear scale
/// law. Chosen as a fixed structural constant (magic-by-default): enough bins to
/// resolve a smooth monotone or unimodal scale trend without over-fitting the
/// per-row noise.
const ACTIVITY_SCALE_BINS: usize = 8;

/// Relative floor on the idiosyncratic diagonal `D`, as a fraction of the mean
/// residual variance. Keeps `Σ_n ≻ 0` and the Woodbury `r × r` capacitance
/// invertible even when a channel is (near-)perfectly explained by the factor.
const DIAGONAL_REL_FLOOR: f64 = 1e-6;

/// Relative floor on the activity scale `c(z)`, as a fraction of its mean. Keeps
/// `c(z) > 0` (a covariance scale) across the whole `z` range.
const SCALE_REL_FLOOR: f64 = 1e-4;

/// The fitted structured residual-covariance model: low-rank factor `Λ`,
/// idiosyncratic diagonal `D`, and the smooth activity-scale `c(z)` evaluated at
/// every row. Produces per-row precision factors and the
/// [`MetricProvenance::WhitenedStructured`](gam_problem::MetricProvenance::WhitenedStructured)
/// [`RowMetric`](gam_problem::RowMetric).
#[derive(Clone, Debug)]
pub struct StructuredResidualModel {
    /// Output dimensionality `p` (residual width).
    p: usize,
    /// Selected factor rank `r` (`0 ≤ r ≤ p`). `0` ⇒ pure-diagonal noise model.
    factor_rank: usize,
    /// Interference factor `Λ ∈ ℝ^{p×r}` (the shared off-diagonal subspace).
    lambda: Array2<f64>,
    /// Idiosyncratic diagonal `d ∈ ℝ^p` (`D = diag(d)`), floored `≻ 0`.
    diagonal: Array1<f64>,
    /// Per-row activity scale `c(z_n) > 0`, length `n`.
    row_scale: Array1<f64>,
    /// Penalized Gaussian log-evidence of the selected model (higher is better).
    /// The value the evidence ladder maximized over the candidate ranks.
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
    /// Maximum factor rank the evidence ladder is allowed to consider. The
    /// ladder scores `r = 0, 1, …, min(max_factor_rank, p−1)` and keeps the
    /// penalized-evidence maximizer. `0` forces the pure-diagonal model.
    pub max_factor_rank: usize,
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
    /// Fit the structured residual-covariance model by the deterministic
    /// fixed-iteration alternation, selecting the factor rank by the evidence
    /// ladder. Returns an error only on shape / non-finite-input violations; the
    /// numerical path is total (every floor and solve is guarded).
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

        // Bin assignment for the activity-scale law: deterministic equal-width
        // bins over the observed z-range. A degenerate (zero-width) range maps
        // every row to bin 0, recovering a single homoscedastic scale.
        let bins = ACTIVITY_SCALE_BINS.max(1);
        let z_min = z.iter().copied().fold(f64::INFINITY, f64::min);
        let z_max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let z_span = z_max - z_min;
        let row_bin: Vec<usize> = (0..n)
            .map(|i| {
                if z_span <= 0.0 {
                    0
                } else {
                    let frac = (z[i] - z_min) / z_span;
                    let idx = (frac * bins as f64).floor() as isize;
                    idx.clamp(0, bins as isize - 1) as usize
                }
            })
            .collect();

        let max_rank = input.max_factor_rank.min(p.saturating_sub(1));

        // Evidence ladder over candidate factor ranks. Each candidate is fit by
        // the full alternation and scored by its penalized Gaussian log-evidence;
        // the maximizer is kept. Index order breaks any tie (lowest rank wins on
        // an exact tie — Occam).
        let mut best: Option<StructuredResidualModel> = None;
        for rank in 0..=max_rank {
            let model = Self::fit_fixed_rank(r, &row_bin, bins, rank)?;
            let take = match &best {
                None => true,
                Some(b) => model.log_evidence > b.log_evidence,
            };
            if take {
                best = Some(model);
            }
        }
        best.ok_or_else(|| "StructuredResidualModel::fit: evidence ladder empty".to_string())
    }

    /// Fit the model at a fixed factor rank by the deterministic alternation.
    fn fit_fixed_rank(
        r: ArrayView2<'_, f64>,
        row_bin: &[usize],
        bins: usize,
        rank: usize,
    ) -> Result<Self, String> {
        let n = r.nrows();
        let p = r.ncols();

        // Mean residual variance — the scale reference for the diagonal floor.
        let mut total_var = 0.0_f64;
        for i in 0..n {
            for j in 0..p {
                total_var += r[[i, j]] * r[[i, j]];
            }
        }
        let mean_var = (total_var / (n as f64 * p as f64)).max(f64::MIN_POSITIVE);
        let diag_floor = DIAGONAL_REL_FLOOR * mean_var;

        // Initialize the per-row scale to 1 (homoscedastic start), the diagonal
        // to the per-channel sample variance, and Λ to the leading eigenvectors
        // of the (scale-1) second moment. The alternation refines all three.
        let mut row_scale = Array1::<f64>::ones(n);
        let mut bin_scale = Array1::<f64>::ones(bins);
        // Raw (undeflated) per-channel second moment — the D estimator's data
        // term. Constant across sweeps.
        let raw_diag = column_variances(r);
        let mut diagonal = raw_diag.mapv(|v| v.max(diag_floor));
        let mut lambda = Array2::<f64>::zeros((p, rank));

        for _sweep in 0..ALTERNATION_SWEEPS {
            // (Λ, D | scale): scale-deflated second moment
            //   S = (1/n) Σ_n (r_n r_nᵀ) / c(z_n).
            // Under the model E[r_n r_nᵀ] = c_n ΛΛᵀ + D, so S ≈ ΛΛᵀ + D̄ with
            // D̄ the scale-averaged diagonal; the leading eigenpairs of S − D
            // give Λ, the residual diagonal gives D.
            let s = scaled_second_moment(r, &row_scale);
            let (evals, evecs) = symmetric_eig_ascending(&s)?;
            // Leading `rank` eigenpairs (eigenvalues ascending ⇒ take the tail).
            if rank > 0 {
                for k in 0..rank {
                    let col = p - 1 - k;
                    // Factor energy above the idiosyncratic floor: the part of
                    // the eigenvalue not explained by the mean diagonal.
                    let mean_diag = diagonal.iter().copied().sum::<f64>() / p as f64;
                    let energy = (evals[col] - mean_diag).max(0.0);
                    let amp = energy.sqrt();
                    for row in 0..p {
                        lambda[[row, k]] = amp * evecs[[row, col]];
                    }
                }
            }
            // D update from the RAW (undeflated) moment, floored ≻ 0. The model
            // is Σ_n = c_n·ΛΛᵀ + D with D NOT scale-multiplied, and c is mean-1
            // normalized, so E[(1/n)Σ r_n r_nᵀ] = ΛΛᵀ + D exactly. The deflated
            // moment `s` is the right object for the FACTOR block (its factor
            // part is scale-free) but its diagonal carries D·mean(1/c) — a
            // Jensen-inflated D (mean(1/c) > 1 for any non-constant law), which
            // biased D upward by exactly mean(1/c̃) and let a spurious
            // higher-rank candidate win the evidence ladder on a better D
            // alone (the probe's rank-2 winner had a zero second column).
            for j in 0..p {
                let mut factor_var = 0.0_f64;
                for k in 0..rank {
                    factor_var += lambda[[j, k]] * lambda[[j, k]];
                }
                diagonal[j] = (raw_diag[j] - factor_var).max(diag_floor);
            }

            // (scale | Λ, D): per-row factor activity. With residual r_n, the
            // factor-subspace energy is r_nᵀ P r_n where P projects onto
            // range(Λ) in the D-whitened metric; the maximum-likelihood scalar
            // multiplier on ΛΛᵀ that matches the row's factor-subspace energy is
            //   c_n = (r̃_nᵀ B (BᵀB)^{-1} Bᵀ r̃_n) / tr(...)-normalizer.
            // We use a stable closed-form proxy: the row's factor-coordinate
            // energy ‖Λ⁺ r_n‖² normalized by the unit-scale expectation, then
            // bin-smoothed across z. With rank 0 there is no factor ⇒ c ≡ 1.
            if rank > 0 {
                let mut bin_num = Array1::<f64>::zeros(bins);
                let mut bin_den = Array1::<f64>::zeros(bins);
                let coords = factor_coordinates(&lambda, &diagonal, r)?;
                for i in 0..n {
                    let mut energy = 0.0_f64;
                    for k in 0..rank {
                        energy += coords[[i, k]] * coords[[i, k]];
                    }
                    let b = row_bin[i];
                    bin_num[b] += energy;
                    bin_den[b] += rank as f64;
                }
                // Per-bin mean factor energy = activity scale. Empty bins inherit
                // the global mean so the scale law stays defined everywhere.
                let global = {
                    let num: f64 = bin_num.iter().sum();
                    let den: f64 = bin_den.iter().sum();
                    if den > 0.0 { num / den } else { 1.0 }
                };
                for b in 0..bins {
                    bin_scale[b] = if bin_den[b] > 0.0 {
                        bin_num[b] / bin_den[b]
                    } else {
                        global
                    };
                }
                // Smooth (3-point moving average over bins) for a continuous law,
                // then floor ≻ 0.
                let scale_floor = SCALE_REL_FLOOR * global.max(f64::MIN_POSITIVE);
                let smoothed = moving_average_3(&bin_scale);
                for b in 0..bins {
                    bin_scale[b] = smoothed[b].max(scale_floor);
                }
                // Re-normalize so the mean scale is 1 (the factor amplitude lives
                // in Λ; c(z) carries only the relative activity law). This keeps
                // the (Λ, D) ↔ (scale) split identified.
                //
                // The mean MUST be taken over ROWS, not over bins. The identity
                // that makes `raw_diag` an unbiased ΛΛᵀ + D estimator is
                //   E[(1/n) Σ_n r_n r_nᵀ] = (1/n) Σ_i c(z_i) · ΛΛᵀ + D,
                // which reduces to ΛΛᵀ + D iff the ROW mean of c is 1:
                //   (1/n) Σ_i c(z_i) = Σ_b (n_b / n) · bin_scale[b] = 1,
                // where n_b is the occupancy (row count) of bin b. Under uneven
                // occupancy (the common case — z is data-driven) the bin-UNIFORM
                // mean (1/bins) Σ_b bin_scale[b] ≠ this occupancy-weighted mean, so
                // normalizing by it would leave raw_diag = ΛΛᵀ + D biased by
                // exactly (occupancy mean / bin mean). Divide by the occupancy-
                // weighted mean instead, so (1/n) Σ_i row_scale[i] is exactly 1.
                // ORDERING: the positivity floor was applied above FIRST, so the
                // floored per-bin values are the ones this normalization sees; the
                // per-row assignment below therefore needs no second clamp (a
                // re-clamp would use pre-normalization floor units and break the
                // exact row-mean-1 invariant just established).
                let mut bin_count = vec![0.0_f64; bins];
                for &b in row_bin.iter() {
                    bin_count[b] += 1.0;
                }
                let mean_scale = (0..bins)
                    .map(|b| bin_count[b] * bin_scale[b])
                    .sum::<f64>()
                    / n as f64;
                if mean_scale > 0.0 {
                    bin_scale.mapv_inplace(|v| v / mean_scale);
                }
                // Each bin_scale[b] is already ≥ scale_floor / mean_scale > 0.
                for i in 0..n {
                    row_scale[i] = bin_scale[row_bin[i]];
                }
            }
        }

        let log_evidence = penalized_log_evidence(r, &lambda, &diagonal, &row_scale, rank);
        let mut model = Self {
            p,
            factor_rank: rank,
            lambda,
            diagonal,
            row_scale,
            log_evidence,
        };
        // Guard against any non-finite leak from a degenerate fit: fall back to a
        // pure-diagonal model with the same evidence accounting.
        if !model.is_finite() {
            model.lambda = Array2::<f64>::zeros((p, rank));
            model.row_scale = Array1::<f64>::ones(n);
        }
        Ok(model)
    }

    fn is_finite(&self) -> bool {
        self.lambda.iter().all(|v| v.is_finite())
            && self.diagonal.iter().all(|v| v.is_finite() && *v > 0.0)
            && self.row_scale.iter().all(|v| v.is_finite() && *v > 0.0)
            && self.log_evidence.is_finite()
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

    /// The per-row activity scale `c(z_n) > 0`, length `n`. Recovers the smooth
    /// activity-scale law evaluated at every observed `z_n`.
    pub fn row_scale(&self) -> ArrayView1<'_, f64> {
        self.row_scale.view()
    }

    /// The penalized Gaussian log-evidence the rank-selection ladder maximized.
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
    ///    inside the evidence-ladder-selected rank (so it cleared the BIC
    ///    penalty globally); this per-direction floor additionally requires the
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
    /// [`RowMetric`](gam_problem::RowMetric). This is the single
    /// production site of `WhitenedStructured`.
    ///
    /// The precision is formed in **Woodbury form**:
    /// ```text
    ///   Σ_n^{-1} = D^{-1} − D^{-1} Λ ( c^{-1} I_r + Λᵀ D^{-1} Λ )^{-1} Λᵀ D^{-1},
    /// ```
    /// an `r × r` capacitance solve (never a `p × p` inverse). The factor `U_n`
    /// is the lower-Cholesky of the assembled `Σ_n^{-1}` (`rank = p`), so
    /// `whiten_residual_row` returns coordinates whose squared norm is the exact
    /// Mahalanobis residual `r_nᵀ Σ_n^{-1} r_n`.
    pub fn row_metric(&self, n_rows: usize) -> Result<RowMetric, String> {
        if n_rows != self.row_scale.len() {
            return Err(format!(
                "StructuredResidualModel::row_metric: requested {n_rows} rows but model has {}",
                self.row_scale.len()
            ));
        }
        let p = self.p;
        let r = self.factor_rank;
        // Hoist every row-INDEPENDENT Woodbury part out of the per-row loop: the
        // inverse diagonal D^{-1}, B = D^{-1}Λ, its transpose Bᵀ, and the Gram
        // M0 = ΛᵀD^{-1}Λ. Only the c_n^{-1} I_r shift on the capacitance is
        // per-row, so the per-row capacitance is M_n = M0 + c_n^{-1} I_r — a
        // scalar-diagonal reweight of the SAME M0 (mirroring the Fix-B hoist in
        // `penalized_log_evidence`). Building the n-row U_n stack now costs
        // O(p·r² + n·(p·r + r³ + p³)) instead of rebuilding B and the Gram every
        // row. The summation order per row is unchanged, so the assembled U_n is
        // bit-for-bit identical to the per-row-rebuild it replaces.
        let d_inv: Vec<f64> = (0..p).map(|i| 1.0 / self.diagonal[i]).collect();
        let mut b = Array2::<f64>::zeros((p, r));
        let mut bt = Array2::<f64>::zeros((r, p));
        let mut m0 = Array2::<f64>::zeros((r, r));
        if r > 0 {
            for i in 0..p {
                for k in 0..r {
                    b[[i, k]] = d_inv[i] * self.lambda[[i, k]];
                }
            }
            for a in 0..r {
                for bk in 0..r {
                    let mut acc = 0.0_f64;
                    for i in 0..p {
                        acc += self.lambda[[i, a]] * b[[i, bk]];
                    }
                    m0[[a, bk]] = acc;
                }
            }
            for k in 0..r {
                for i in 0..p {
                    bt[[k, i]] = b[[i, k]];
                }
            }
        }
        // Row-major flat factor matrix: u[n, i*p + k] = U_n[i, k].
        let mut u = Array2::<f64>::zeros((n_rows, p * p));
        for row in 0..n_rows {
            let precision = self.row_precision(&d_inv, &b, &bt, &m0, row)?;
            let factor = lower_cholesky_psd(&precision)?;
            for i in 0..p {
                for k in 0..p {
                    u[[row, i * p + k]] = factor[[i, k]];
                }
            }
        }
        RowMetric::whitened_structured(Arc::new(u), p, p)
    }

    /// Convenience for the #2021 fit-path install seam: fit the structured
    /// residual model on `input` and immediately materialize its per-row
    /// `WhitenedStructured` [`RowMetric`] over all `input.residuals.nrows()`
    /// rows. Equivalent to `Self::fit(input)?.row_metric(n)` — the single call
    /// the outer alternation loop consumes when it installs the whitening metric
    /// but does not also need the fitted factor (`factor()` / birth mining).
    pub fn fit_row_metric(input: ResidualFactorInput<'_>) -> Result<RowMetric, String> {
        let n = input.residuals.nrows();
        Self::fit(input)?.row_metric(n)
    }

    /// Damped per-row metric for the #2021 driver: blend covariances in the
    /// **covariance domain** (before the Woodbury→Cholesky) between this model's
    /// estimate and a previous one,
    /// ```text
    ///   Σ_t(row) = (1 − γ) · Σ_prev(row) + γ · Σ̂_t(row),
    /// ```
    /// where `Σ̂_t(row) = c_t(z)·ΛΛᵀ + D` is this model's per-row covariance
    /// (built from the hoisted-M0 / occupancy-weighted `c(z)` path), and
    /// `Σ_prev(row)` is `prev`'s per-row covariance when `Some`, else `I_p`. The
    /// returned factor `U_n` satisfies `U_n U_nᵀ = Σ_t(row)^{-1}`, packaged as a
    /// [`RowMetric`](gam_problem::RowMetric).
    ///
    /// Endpoints (exact, byte-identical to the undamped producers):
    /// * `γ = 1.0` ⇒ this model's [`Self::row_metric`] exactly (Woodbury path);
    /// * `γ = 0.0` ⇒ `prev`'s [`Self::row_metric`] when `Some`, else the
    ///   Euclidean identity metric.
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
            return match prev {
                Some(pv) => pv.row_metric(n_rows),
                None => RowMetric::euclidean(n_rows, self.p),
            };
        }

        let p = self.p;
        // Row-INDEPENDENT outer products ΛΛᵀ (this model and, if present, prev):
        // only the per-row activity scale c(z) multiplies them, so hoist the Gram
        // out of the per-row loop (mirroring the row_metric / penalized_log_evidence
        // hoist).
        let self_gram = outer_product(&self.lambda);
        let prev_gram = prev.map(|pv| outer_product(&pv.lambda));

        let mut u = Array2::<f64>::zeros((n_rows, p * p));
        for row in 0..n_rows {
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
            match prev {
                Some(pv) => {
                    let cp = pv.row_scale[row].max(f64::MIN_POSITIVE);
                    let pg = prev_gram.as_ref().unwrap();
                    for a in 0..p {
                        for b in 0..p {
                            sigma[[a, b]] += (1.0 - gamma) * cp * pg[[a, b]];
                        }
                        sigma[[a, a]] += (1.0 - gamma) * pv.diagonal[a];
                    }
                }
                None => {
                    for a in 0..p {
                        sigma[[a, a]] += 1.0 - gamma;
                    }
                }
            }
            // Symmetrize against round-off before inversion.
            for a in 0..p {
                for b in (a + 1)..p {
                    let avg = 0.5 * (sigma[[a, b]] + sigma[[b, a]]);
                    sigma[[a, b]] = avg;
                    sigma[[b, a]] = avg;
                }
            }
            // Σ_t is a convex combination of SPD matrices (D ≻ 0 / I ≻ 0) ⇒ SPD.
            // Precision = Σ_t^{-1} via a Cholesky solve against I_p, then the U_n
            // factor is the lower-Cholesky of the precision (row_metric's U
            // convention).
            let precision = invert_spd(&sigma)?;
            let factor = lower_cholesky_psd(&precision)?;
            for i in 0..p {
                for k in 0..p {
                    u[[row, i * p + k]] = factor[[i, k]];
                }
            }
        }
        RowMetric::whitened_structured(Arc::new(u), p, p)
    }

    /// Per-row precision `Σ_n^{-1}` via the Woodbury identity (an `r × r` solve),
    /// given the row-independent parts precomputed by [`Self::row_metric`]:
    /// `d_inv = D^{-1}`, `b = D^{-1}Λ`, `bt = Bᵀ`, and the Gram `m0 = ΛᵀD^{-1}Λ`.
    /// Only the per-row capacitance `M_n = m0 + c_n^{-1} I_r` and the back-solve
    /// depend on the row.
    fn row_precision(
        &self,
        d_inv: &[f64],
        b: &Array2<f64>,
        bt: &Array2<f64>,
        m0: &Array2<f64>,
        row: usize,
    ) -> Result<Array2<f64>, String> {
        let p = self.p;
        let r = self.factor_rank;
        // Start from D^{-1}.
        let mut precision = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            precision[[i, i]] = d_inv[i];
        }
        if r == 0 {
            return Ok(precision);
        }
        let c = self.row_scale[row].max(f64::MIN_POSITIVE);
        // Per-row capacitance M_n = M0 + c^{-1} I_r (copy the hoisted Gram, then
        // add c^{-1} to the diagonal). M_n ≻ 0 since c^{-1} > 0 and M0 ⪰ 0.
        let mut cap = m0.clone();
        for a in 0..r {
            cap[[a, a]] += 1.0 / c;
        }
        // Σ_n^{-1} = D^{-1} − B M_n^{-1} Bᵀ. Solve M_n X = Bᵀ for X = M_n^{-1} Bᵀ
        // (r × p) via Cholesky.
        let chol = cap
            .cholesky(Side::Lower)
            .map_err(|e| format!("StructuredResidualModel::row_precision capacitance: {e:?}"))?;
        let x = chol.solve_mat(bt); // r × p
        for i in 0..p {
            for j in 0..p {
                let mut acc = 0.0_f64;
                for k in 0..r {
                    acc += b[[i, k]] * x[[k, j]];
                }
                precision[[i, j]] -= acc;
            }
        }
        // Symmetrize against round-off so the Cholesky downstream sees an exactly
        // symmetric PSD matrix.
        for i in 0..p {
            for j in (i + 1)..p {
                let avg = 0.5 * (precision[[i, j]] + precision[[j, i]]);
                precision[[i, j]] = avg;
                precision[[j, i]] = avg;
            }
        }
        Ok(precision)
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

/// Inverse of a symmetric positive-definite matrix via a Cholesky solve against
/// the identity, symmetrized against round-off. Used to form `Σ_t^{-1}` from a
/// densely-blended covariance in [`StructuredResidualModel::row_metric_damped`]
/// (the blended covariance is no longer low-rank-plus-diagonal, so Woodbury does
/// not apply).
fn invert_spd(a: &Array2<f64>) -> Result<Array2<f64>, String> {
    let p = a.nrows();
    let chol = a
        .cholesky(Side::Lower)
        .map_err(|e| format!("invert_spd: blended covariance not SPD: {e:?}"))?;
    let mut inv = chol.solve_mat(&Array2::<f64>::eye(p));
    for i in 0..p {
        for j in (i + 1)..p {
            let avg = 0.5 * (inv[[i, j]] + inv[[j, i]]);
            inv[[i, j]] = avg;
            inv[[j, i]] = avg;
        }
    }
    Ok(inv)
}

/// Per-channel (column) sample second moment of the residual matrix.
fn column_variances(r: ArrayView2<'_, f64>) -> Array1<f64> {
    let n = r.nrows();
    let p = r.ncols();
    let mut v = Array1::<f64>::zeros(p);
    for j in 0..p {
        let mut acc = 0.0_f64;
        for i in 0..n {
            acc += r[[i, j]] * r[[i, j]];
        }
        v[j] = acc / n as f64;
    }
    v
}

/// Scale-deflated second moment `S = (1/n) Σ_n (r_n r_nᵀ) / c_n`.
/// Per-row-chunk contribution to the scaled second moment — the inner
/// `p×p` accumulation of one contiguous row block, summed in row order.
fn scaled_second_moment_chunk(
    r: ArrayView2<'_, f64>,
    row_scale: &Array1<f64>,
    lo: usize,
    hi: usize,
) -> Array2<f64> {
    let p = r.ncols();
    let mut s = Array2::<f64>::zeros((p, p));
    for i in lo..hi {
        let w = 1.0 / row_scale[i].max(f64::MIN_POSITIVE);
        for a in 0..p {
            let ra = r[[i, a]];
            for b in 0..p {
                s[[a, b]] += w * ra * r[[i, b]];
            }
        }
    }
    s
}

/// `S = (1/n) Σ_n (r_n r_nᵀ) / c(z_n)` — the O(N·p²) scale-deflated second moment
/// that dominates each alternation sweep of the residual-factor fit.
///
/// Parallelized over FIXED contiguous row chunks whose `p×p` partials are summed
/// in CHUNK ORDER, so the result is bit-reproducible and independent of the thread
/// count (the estimator's determinism contract holds) — it differs from a single
/// running row-sum only in the harmless grouping of accumulation round-off, which
/// the trailing symmetrization already absorbs. Engaged only above a row threshold
/// (the serial path stays exact on small inputs and avoids rayon overhead) and only
/// when NOT already inside a rayon worker (the same nesting discipline the
/// Arrow-Schur solve uses — a nested call runs the serial reduction so an outer
/// parallel region keeps its cores).
fn scaled_second_moment(r: ArrayView2<'_, f64>, row_scale: &Array1<f64>) -> Array2<f64> {
    use rayon::prelude::*;
    const PARALLEL_ROW_MIN: usize = 8192;
    const CHUNK_ROWS: usize = 2048;
    let n = r.nrows();
    let p = r.ncols();

    let mut s = if n >= PARALLEL_ROW_MIN && rayon::current_thread_index().is_none() {
        let n_chunks = n.div_ceil(CHUNK_ROWS);
        let partials: Vec<Array2<f64>> = (0..n_chunks)
            .into_par_iter()
            .map(|c| {
                let lo = c * CHUNK_ROWS;
                let hi = ((c + 1) * CHUNK_ROWS).min(n);
                scaled_second_moment_chunk(r, row_scale, lo, hi)
            })
            .collect();
        let mut acc = Array2::<f64>::zeros((p, p));
        for part in &partials {
            acc += part;
        }
        acc
    } else {
        scaled_second_moment_chunk(r, row_scale, 0, n)
    };

    s.mapv_inplace(|v| v / n as f64);
    // Symmetrize against accumulation round-off.
    for a in 0..p {
        for b in (a + 1)..p {
            let avg = 0.5 * (s[[a, b]] + s[[b, a]]);
            s[[a, b]] = avg;
            s[[b, a]] = avg;
        }
    }
    s
}

/// Factor coordinates `Λ⁺_D r_n` per row: the generalized-least-squares
/// projection of each residual onto `range(Λ)` in the `D^{-1}` metric, returned
/// as an `n × r` matrix. Solves the `r × r` normal equations
/// `(Λᵀ D^{-1} Λ) γ = Λᵀ D^{-1} r_n` per row (shared factorization).
fn factor_coordinates(
    lambda: &Array2<f64>,
    diagonal: &Array1<f64>,
    r: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, String> {
    let p = lambda.nrows();
    let rank = lambda.ncols();
    let n = r.nrows();
    // GLS weights 1/D_ii, with zero-variance channels DROPPED (weight 0): a
    // channel whose residual is identically zero carries no factor information,
    // and its 1/0 = ∞ weight poisons the whole normal matrix into NaN — the
    // fully-explained-target abort (Cholesky NonPositivePivot) that killed
    // stagewise runs on targets the dictionary explains exactly. Dropping the
    // channel is the pseudo-inverse limit; with every channel degenerate the
    // ridged normal matrix stays PD and the coordinates are the least-norm 0.
    let d_inv: Vec<f64> = (0..p)
        .map(|i| {
            let d = diagonal[i];
            if !(d > 0.0 && d.is_finite()) {
                return 0.0;
            }
            // A subnormal-floored variance (the zero-residual case floors the
            // scale reference at f64::MIN_POSITIVE) passes `d > 0` but its
            // reciprocal OVERFLOWS to ∞ — the same NaN poisoning through the
            // second door. A non-finite weight is the same degenerate-channel
            // verdict: drop it.
            let w = d.recip();
            if w.is_finite() { w } else { 0.0 }
        })
        .collect();
    // Normal matrix ΛᵀD^{-1}Λ (+ tiny ridge for invertibility).
    let mut normal = Array2::<f64>::zeros((rank, rank));
    for a in 0..rank {
        for b in 0..rank {
            let mut acc = 0.0_f64;
            for i in 0..p {
                acc += lambda[[i, a]] * d_inv[i] * lambda[[i, b]];
            }
            normal[[a, b]] = acc;
        }
    }
    let trace = (0..rank).map(|k| normal[[k, k]]).sum::<f64>().max(1.0);
    let ridge = 1e-10 * trace / rank.max(1) as f64;
    for k in 0..rank {
        normal[[k, k]] += ridge;
    }
    let chol = normal
        .cholesky(Side::Lower)
        .map_err(|e| format!("factor_coordinates normal solve: {e:?}"))?;
    let mut coords = Array2::<f64>::zeros((n, rank));
    let mut rhs = Array1::<f64>::zeros(rank);
    for i in 0..n {
        for a in 0..rank {
            let mut acc = 0.0_f64;
            for j in 0..p {
                acc += lambda[[j, a]] * d_inv[j] * r[[i, j]];
            }
            rhs[a] = acc;
        }
        let gamma = chol.solvevec(&rhs);
        for a in 0..rank {
            coords[[i, a]] = gamma[a];
        }
    }
    Ok(coords)
}

/// 3-point moving average over a bin vector (edge-clamped), giving the smooth
/// activity-scale law a continuous, low-curvature shape.
fn moving_average_3(v: &Array1<f64>) -> Array1<f64> {
    let m = v.len();
    let mut out = Array1::<f64>::zeros(m);
    for i in 0..m {
        let lo = i.saturating_sub(1);
        let hi = (i + 1).min(m - 1);
        let mut acc = 0.0_f64;
        let mut cnt = 0.0_f64;
        for j in lo..=hi {
            acc += v[j];
            cnt += 1.0;
        }
        out[i] = acc / cnt;
    }
    out
}

/// Ascending-eigenvalue symmetric eigendecomposition (faer convention).
fn symmetric_eig_ascending(m: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>), String> {
    m.eigh(Side::Lower)
        .map_err(|e| format!("symmetric_eig: {e:?}"))
}

/// Lower-triangular Cholesky factor `L` of a (numerically) PSD matrix `A` with
/// `L Lᵀ = A`, with a relative spectral floor so a marginally-indefinite
/// precision (round-off) still factors. Used to turn `Σ_n^{-1}` into the
/// `RowMetric` factor `U_n` (here `U_n = L`).
fn lower_cholesky_psd(a: &Array2<f64>) -> Result<Array2<f64>, String> {
    if let Ok(chol) = a.cholesky(Side::Lower) {
        return Ok(chol.lower_triangular());
    }
    // Eigen-repair: clamp eigenvalues to a small positive floor, rebuild the
    // REPAIRED matrix Q·diag(λ_clamped)·Qᵀ itself, and Cholesky that (always
    // succeeds, PD). The returned factor must satisfy L·Lᵀ = A_repaired —
    // rebuilding the symmetric square root here and factoring THAT would hand
    // callers a factor with L·Lᵀ = A^{1/2}, silently taking every whitened
    // quadratic form against the square root of the intended precision.
    let (evals, evecs) = symmetric_eig_ascending(a)?;
    let max_ev = evals.iter().copied().fold(0.0_f64, f64::max).max(1.0);
    let floor = 1e-10 * max_ev;
    let p = a.nrows();
    let mut repaired = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        for j in 0..p {
            let mut acc = 0.0_f64;
            for k in 0..p {
                let ev = evals[k].max(floor);
                acc += evecs[[i, k]] * ev * evecs[[j, k]];
            }
            repaired[[i, j]] = acc;
        }
    }
    repaired
        .cholesky(Side::Lower)
        .map(|c| c.lower_triangular())
        .map_err(|e| format!("lower_cholesky_psd eigen-repair: {e:?}"))
}

/// Penalized Gaussian log-evidence of the structured model at the fitted
/// parameters — the evidence ladder's rank-selection score.
///
/// The per-row log-density of `r_n ~ N(0, Σ_n)` is
/// `−½ ( log|Σ_n| + r_nᵀ Σ_n^{-1} r_n + p log 2π )`. We sum it across rows and
/// subtract a parameter-count penalty `½ k_params · log n` (a BIC-style Occam
/// term over the `p·r` factor entries + `p` diagonal entries + the bin scales),
/// so adding a spurious factor that does not improve the fit is rejected. Both
/// `log|Σ_n|` and the quadratic use the Woodbury / matrix-determinant lemma so no
/// dense `p × p` inverse or determinant is formed.
fn penalized_log_evidence(
    r: ArrayView2<'_, f64>,
    lambda: &Array2<f64>,
    diagonal: &Array1<f64>,
    row_scale: &Array1<f64>,
    rank: usize,
) -> f64 {
    let n = r.nrows();
    let p = r.ncols();
    let d_inv: Vec<f64> = (0..p).map(|i| 1.0 / diagonal[i]).collect();
    let log_det_d: f64 = diagonal.iter().map(|&d| d.ln()).sum();
    let two_pi_ln = (2.0 * std::f64::consts::PI).ln();

    // Row-INDEPENDENT Gram M0 = ΛᵀD^{-1}Λ (r × r). This does not depend on the
    // row, so build it ONCE here rather than rebuilding it inside the per-row loop
    // (which was O(n·p·r²)). The per-row capacitance is only a scalar-diagonal
    // reweight of this SAME M0 — M_n = M0 + (1/c_n) I_r — so each row copies M0 and
    // adds 1/c_n to its diagonal (cheap, O(r)) before its own Cholesky. The
    // summation order over j (0..p) is preserved exactly and the diagonal add is
    // the identical `+= 1.0 / c` op, so the hoist is bit-for-bit identical to the
    // pre-hoist per-row rebuild (same log|Σ_n|, same quadratic, same evidence).
    let mut m0 = Array2::<f64>::zeros((rank, rank));
    if rank > 0 {
        for a in 0..rank {
            for b in 0..rank {
                let mut acc = 0.0_f64;
                for j in 0..p {
                    acc += lambda[[j, a]] * d_inv[j] * lambda[[j, b]];
                }
                m0[[a, b]] = acc;
            }
        }
    }

    let mut log_lik = 0.0_f64;
    for i in 0..n {
        let c = row_scale[i].max(f64::MIN_POSITIVE);
        // Quadratic r_nᵀ Σ_n^{-1} r_n via Woodbury:
        //   r_nᵀ D^{-1} r_n − (Bᵀ r_n)ᵀ M^{-1} (Bᵀ r_n),
        // with B = D^{-1}Λ and M = c^{-1}I + ΛᵀD^{-1}Λ.
        let mut quad = 0.0_f64;
        for j in 0..p {
            quad += r[[i, j]] * d_inv[j] * r[[i, j]];
        }
        let mut log_det = log_det_d;
        if rank > 0 {
            // Per-row capacitance M_n = M0 + (1/c) I_r (copy the hoisted M0, then
            // add 1/c to the diagonal), and w = Bᵀ r_n = ΛᵀD^{-1} r_n.
            let mut m = m0.clone();
            for a in 0..rank {
                m[[a, a]] += 1.0 / c;
            }
            let mut w = Array1::<f64>::zeros(rank);
            for a in 0..rank {
                let mut wa = 0.0_f64;
                for j in 0..p {
                    wa += lambda[[j, a]] * d_inv[j] * r[[i, j]];
                }
                w[a] = wa;
            }
            // Cholesky M = R Rᵀ → log|M|, and solve M y = w.
            match m.cholesky(Side::Lower) {
                Ok(chol) => {
                    let y = chol.solvevec(&w);
                    let mut wy = 0.0_f64;
                    for a in 0..rank {
                        wy += w[a] * y[a];
                    }
                    quad -= wy;
                    // log|Σ_n| = log|D| + log|M| + r·log c   (matrix-determinant
                    // lemma; the c^{-1}I shift carries the +r·log c).
                    let diag = chol.diag();
                    let log_det_m: f64 = diag.iter().map(|&l| (l * l).ln()).sum();
                    log_det = log_det_d + log_det_m + rank as f64 * c.ln();
                }
                Err(_) => {
                    // Degenerate capacitance — fall back to the diagonal model's
                    // accounting for this row (no factor correction).
                    log_det = log_det_d;
                }
            }
        }
        log_lik += -0.5 * (log_det + quad + p as f64 * two_pi_ln);
    }

    let k_params = (p * rank + p + ACTIVITY_SCALE_BINS) as f64;
    log_lik - 0.5 * k_params * (n.max(2) as f64).ln()
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
        let u1 = lcg_uniform(state).max(1e-12);
        let u2 = lcg_uniform(state);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// Per-rank evidence breakdown on the planted single-factor activity-law
    /// DGP (the `fitted_scale_recovers_planted_activity_law` plant). Pins the
    /// rank-selection decision itself: the ladder must prefer rank 1, and this
    /// test names the margin so an over-selection regression is diagnosable
    /// from the failure message alone.
    #[test]
    fn evidence_ladder_prefers_planted_rank_one() {
        let n = 5000usize;
        let p = 4usize;
        let lambda0 = ndarray::array![[1.5], [1.2], [-0.4], [0.3]];
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
                residuals[[row, i]] = amp * lambda0[[i, 0]] * f + sigma_eps * lcg_normal(&mut seed);
            }
        }
        // Reproduce fit()'s bin assignment, then score each rank directly.
        let bins = ACTIVITY_SCALE_BINS.max(1);
        let row_bin: Vec<usize> = (0..n)
            .map(|i| {
                let frac = activity[i];
                (frac * bins as f64).floor().clamp(0.0, bins as f64 - 1.0) as usize
            })
            .collect();
        let mut report = String::new();
        let mut ev = Vec::new();
        for rank in 0..=2usize {
            let m = StructuredResidualModel::fit_fixed_rank(residuals.view(), &row_bin, bins, rank)
                .expect("fixed-rank fit");
            let k_params = (p * rank + p + ACTIVITY_SCALE_BINS) as f64;
            let log_lik = m.log_evidence() + 0.5 * k_params * (n as f64).ln();
            let col_norms: Vec<f64> = (0..rank)
                .map(|k| {
                    m.factor()
                        .column(k)
                        .iter()
                        .map(|v| v * v)
                        .sum::<f64>()
                        .sqrt()
                })
                .collect();
            report.push_str(&format!(
                "rank {rank}: evidence={:.3} loglik={:.3} penalty={:.3} col_norms={:?} diag={:?}\n",
                m.log_evidence(),
                log_lik,
                0.5 * k_params * (n as f64).ln(),
                col_norms,
                m.diagonal()
                    .iter()
                    .map(|v| (v * 1e4).round() / 1e4)
                    .collect::<Vec<_>>()
            ));
            ev.push(m.log_evidence());
        }
        assert!(
            ev[1] > ev[0] && ev[1] > ev[2],
            "evidence ladder must prefer the planted rank 1; breakdown:\n{report}"
        );
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
    /// range(Λ̂) must be small, and the evidence ladder must select rank 2.
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
            max_factor_rank: 4,
        })
        .expect("fit");

        assert_eq!(
            model.factor_rank(),
            2,
            "ladder must select the planted rank 2 (got {}, evidence {:.3})",
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

    /// #974 verification arm (d): recovery of the planted activity-variance
    /// law. Single planted factor with per-row energy `exp(slope·z)`; the
    /// fitted `c(z_n)` must reproduce the law's shape — strongly correlated
    /// with the planted log-scale and with the right dynamic range.
    #[test]
    fn fitted_scale_recovers_planted_activity_law() {
        let n = 6000usize;
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

        let model = StructuredResidualModel::fit(ResidualFactorInput {
            residuals: residuals.view(),
            activity: activity.view(),
            max_factor_rank: 2,
        })
        .expect("fit");
        assert_eq!(model.factor_rank(), 1, "planted rank is 1");

        // Pearson correlation between fitted log c(z_n) and the planted
        // log-law slope·z (mean-1 normalization cancels in the correlation).
        let fitted_log: Vec<f64> = model.row_scale().iter().map(|c| c.ln()).collect();
        let planted_log: Vec<f64> = activity.iter().map(|z| slope * z).collect();
        let mean_f = fitted_log.iter().sum::<f64>() / n as f64;
        let mean_p = planted_log.iter().sum::<f64>() / n as f64;
        let mut cov = 0.0_f64;
        let mut var_f = 0.0_f64;
        let mut var_p = 0.0_f64;
        for i in 0..n {
            let df = fitted_log[i] - mean_f;
            let dp = planted_log[i] - mean_p;
            cov += df * dp;
            var_f += df * df;
            var_p += dp * dp;
        }
        let corr = cov / (var_f.sqrt() * var_p.sqrt());
        assert!(
            corr > 0.9,
            "fitted activity law must track the planted exp({slope}·z): corr = {corr:.4}"
        );

        // Dynamic range: planted c(top)/c(bottom) over the inner bin centers
        // is exp(slope·7/8) ≈ 3.1; the binned/smoothed estimate must land in
        // a generous bracket around it (smoothing shrinks the edges).
        let lo = model.row_scale()[n / 16]; // first-bin interior
        let hi = model.row_scale()[n - 1 - n / 16]; // last-bin interior
        let ratio = hi / lo;
        assert!(
            ratio > 1.8 && ratio < 5.5,
            "fitted dynamic range {ratio:.3} must bracket the planted ≈3.1"
        );
    }

    /// Reproduce `fit`'s equal-width bin assignment for a test activity vector.
    fn assign_bins(activity: &Array1<f64>, bins: usize) -> Vec<usize> {
        let n = activity.len();
        let z_min = activity.iter().copied().fold(f64::INFINITY, f64::min);
        let z_max = activity.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let span = z_max - z_min;
        (0..n)
            .map(|i| {
                if span <= 0.0 {
                    0
                } else {
                    let frac = (activity[i] - z_min) / span;
                    (frac * bins as f64).floor().clamp(0.0, bins as f64 - 1.0) as usize
                }
            })
            .collect()
    }

    /// FIX A invariant: with deliberately UNEVEN bin occupancy the fitted
    /// per-row scale must have `(1/n) Σ_i row_scale[i] = 1` (occupancy-weighted
    /// mean-1), NOT the bin-uniform mean-1 the old code enforced. We assert the
    /// row-mean is 1 to tight tolerance, and that the bin-UNIFORM mean of the
    /// distinct per-bin scales is materially ≠ 1 — which is exactly the quantity
    /// the old normalization forced to 1, so this proves the two means differ
    /// under uneven occupancy and the test bites.
    #[test]
    fn occupancy_weighted_scale_has_row_mean_one() {
        let n = 4000usize;
        let p = 4usize;
        let lambda0 = ndarray::array![1.5, 1.2, -0.4, 0.3];
        let sigma_eps = 0.2_f64;
        let slope = 2.0_f64;
        let mut seed = 0xB5297A4D_u64 ^ 0x68E31DA4_u64;
        let mut residuals = Array2::<f64>::zeros((n, p));
        let mut activity = Array1::<f64>::zeros(n);
        for row in 0..n {
            // Cubic warp concentrates rows in the low-z bins ⇒ uneven occupancy.
            let u = (row as f64) / (n as f64 - 1.0);
            let z = u * u * u;
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
            max_factor_rank: 2,
        })
        .expect("fit");
        assert!(
            model.factor_rank() >= 1,
            "need a non-trivial scale law (rank ≥ 1) for this invariant to bite; got rank 0"
        );

        // Occupancy-weighted (row) mean must be exactly 1.
        let row_mean = model.row_scale().iter().sum::<f64>() / n as f64;
        assert!(
            (row_mean - 1.0).abs() < 1e-9,
            "occupancy-weighted row mean of c(z) must be 1; got {row_mean:.12}"
        );

        // Bins are genuinely unevenly occupied, and every occupied bin's rows
        // share one scale value (collect the distinct value per bin).
        let bins = ACTIVITY_SCALE_BINS.max(1);
        let row_bin = assign_bins(&activity, bins);
        let mut counts = vec![0usize; bins];
        let mut bin_val = vec![f64::NAN; bins];
        for i in 0..n {
            let b = row_bin[i];
            counts[b] += 1;
            bin_val[b] = model.row_scale()[i];
        }
        let occupied: Vec<usize> = (0..bins).filter(|&b| counts[b] > 0).collect();
        let max_c = *counts.iter().max().unwrap();
        let min_c = *counts.iter().filter(|&&c| c > 0).min().unwrap();
        assert!(
            max_c as f64 > 2.0 * min_c as f64,
            "fixture must have uneven occupancy; counts = {counts:?}"
        );

        // The bin-UNIFORM mean of the per-bin scales is what the OLD code forced
        // to 1. Under uneven occupancy + a non-constant law it is materially ≠ 1,
        // so the old normalization would NOT satisfy the row-mean-1 identity.
        let uniform_mean =
            occupied.iter().map(|&b| bin_val[b]).sum::<f64>() / occupied.len() as f64;
        assert!(
            (uniform_mean - 1.0).abs() > 0.1,
            "bin-uniform mean must differ from 1 (proving occupancy weighting matters); \
             got uniform_mean = {uniform_mean:.6}, row_mean = {row_mean:.6}"
        );
    }

    /// Naive, pre-hoist reference for `penalized_log_evidence`: rebuilds the
    /// row-independent Gram M0 = ΛᵀD⁻¹Λ INSIDE the per-row loop (the original
    /// formula). The production function hoists M0 out; the two must agree.
    fn naive_penalized_log_evidence(
        r: ArrayView2<'_, f64>,
        lambda: &Array2<f64>,
        diagonal: &Array1<f64>,
        row_scale: &Array1<f64>,
        rank: usize,
    ) -> f64 {
        let n = r.nrows();
        let p = r.ncols();
        let d_inv: Vec<f64> = (0..p).map(|i| 1.0 / diagonal[i]).collect();
        let log_det_d: f64 = diagonal.iter().map(|&d| d.ln()).sum();
        let two_pi_ln = (2.0 * std::f64::consts::PI).ln();
        let mut log_lik = 0.0_f64;
        for i in 0..n {
            let c = row_scale[i].max(f64::MIN_POSITIVE);
            let mut quad = 0.0_f64;
            for j in 0..p {
                quad += r[[i, j]] * d_inv[j] * r[[i, j]];
            }
            let mut log_det = log_det_d;
            if rank > 0 {
                let mut m = Array2::<f64>::zeros((rank, rank));
                let mut w = Array1::<f64>::zeros(rank);
                for a in 0..rank {
                    let mut wa = 0.0_f64;
                    for j in 0..p {
                        wa += lambda[[j, a]] * d_inv[j] * r[[i, j]];
                    }
                    w[a] = wa;
                    for b in 0..rank {
                        let mut acc = 0.0_f64;
                        for j in 0..p {
                            acc += lambda[[j, a]] * d_inv[j] * lambda[[j, b]];
                        }
                        m[[a, b]] = acc;
                    }
                    m[[a, a]] += 1.0 / c;
                }
                match m.cholesky(Side::Lower) {
                    Ok(chol) => {
                        let y = chol.solvevec(&w);
                        let mut wy = 0.0_f64;
                        for a in 0..rank {
                            wy += w[a] * y[a];
                        }
                        quad -= wy;
                        let diag = chol.diag();
                        let log_det_m: f64 = diag.iter().map(|&l| (l * l).ln()).sum();
                        log_det = log_det_d + log_det_m + rank as f64 * c.ln();
                    }
                    Err(_) => {
                        log_det = log_det_d;
                    }
                }
            }
            log_lik += -0.5 * (log_det + quad + p as f64 * two_pi_ln);
        }
        let k_params = (p * rank + p + ACTIVITY_SCALE_BINS) as f64;
        log_lik - 0.5 * k_params * (n.max(2) as f64).ln()
    }

    /// Naive, per-row-rebuild reference for `factor_coordinates`: rebuilds and
    /// re-factors the (row-independent) normal matrix ΛᵀD⁻¹Λ for EVERY row.
    /// Mathematically identical to the shared-factorization production path.
    fn naive_factor_coordinates(
        lambda: &Array2<f64>,
        diagonal: &Array1<f64>,
        r: ArrayView2<'_, f64>,
    ) -> Array2<f64> {
        let p = lambda.nrows();
        let rank = lambda.ncols();
        let n = r.nrows();
        let d_inv: Vec<f64> = (0..p).map(|i| 1.0 / diagonal[i]).collect();
        let mut coords = Array2::<f64>::zeros((n, rank));
        for i in 0..n {
            let mut normal = Array2::<f64>::zeros((rank, rank));
            for a in 0..rank {
                for b in 0..rank {
                    let mut acc = 0.0_f64;
                    for j in 0..p {
                        acc += lambda[[j, a]] * d_inv[j] * lambda[[j, b]];
                    }
                    normal[[a, b]] = acc;
                }
            }
            let trace = (0..rank).map(|k| normal[[k, k]]).sum::<f64>().max(1.0);
            let ridge = 1e-10 * trace / rank.max(1) as f64;
            for k in 0..rank {
                normal[[k, k]] += ridge;
            }
            let chol = normal.cholesky(Side::Lower).expect("naive normal solve");
            let mut rhs = Array1::<f64>::zeros(rank);
            for a in 0..rank {
                let mut acc = 0.0_f64;
                for j in 0..p {
                    acc += lambda[[j, a]] * d_inv[j] * r[[i, j]];
                }
                rhs[a] = acc;
            }
            let gamma = chol.solvevec(&rhs);
            for a in 0..rank {
                coords[[i, a]] = gamma[a];
            }
        }
        coords
    }

    /// FIX B equivalence: the hoisted `penalized_log_evidence` and the shared-
    /// factorization `factor_coordinates` must equal their naive per-row-rebuild
    /// references to ~1e-10 (in fact bit-for-bit — the hoist preserves op order).
    #[test]
    fn hoisted_gram_matches_naive_per_row_rebuild() {
        let n = 200usize;
        let p = 5usize;
        let rank = 2usize;
        let mut seed = 0x243F6A8885A308D3_u64;
        let mut lambda = Array2::<f64>::zeros((p, rank));
        for i in 0..p {
            for k in 0..rank {
                lambda[[i, k]] = lcg_normal(&mut seed);
            }
        }
        let mut diagonal = Array1::<f64>::zeros(p);
        for j in 0..p {
            diagonal[j] = 0.3 + lcg_uniform(&mut seed); // strictly positive
        }
        let mut row_scale = Array1::<f64>::zeros(n);
        for i in 0..n {
            row_scale[i] = 0.5 + 1.5 * lcg_uniform(&mut seed); // strictly positive
        }
        let mut residuals = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            for j in 0..p {
                residuals[[i, j]] = lcg_normal(&mut seed);
            }
        }

        let ev_hoisted =
            penalized_log_evidence(residuals.view(), &lambda, &diagonal, &row_scale, rank);
        let ev_naive =
            naive_penalized_log_evidence(residuals.view(), &lambda, &diagonal, &row_scale, rank);
        assert!(
            (ev_hoisted - ev_naive).abs() <= 1e-10 * (1.0 + ev_naive.abs()),
            "hoisted log-evidence must equal naive rebuild: {ev_hoisted} vs {ev_naive}"
        );

        let coords_hoisted =
            factor_coordinates(&lambda, &diagonal, residuals.view()).expect("coords");
        let coords_naive = naive_factor_coordinates(&lambda, &diagonal, residuals.view());
        let mut max_abs = 0.0_f64;
        for i in 0..n {
            for a in 0..rank {
                max_abs = max_abs.max((coords_hoisted[[i, a]] - coords_naive[[i, a]]).abs());
            }
        }
        assert!(
            max_abs <= 1e-10,
            "hoisted factor coordinates must equal naive rebuild; max |Δ| = {max_abs:e}"
        );
    }

    /// FIX A regression: on an uneven-bin synthetic with a KNOWN planted single
    /// factor, the low-rank reconstruction ΛΛᵀ + D built from the OCCUPANCY-
    /// weighted scale law reconstructs the empirical second moment
    /// (1/n) Σ_n r_n r_nᵀ strictly better (Frobenius) than the one built from
    /// the bin-UNIFORM scale law. Uses the module's own `scaled_second_moment` /
    /// eigen path so it exercises the real (Λ, D | scale) step.
    #[test]
    fn occupancy_scale_improves_second_moment_reconstruction() {
        let n = 4000usize;
        let p = 4usize;
        let lambda0 = ndarray::array![1.5, 1.2, -0.4, 0.3];
        let sigma_eps = 0.2_f64;
        let slope = 2.0_f64;
        let bins = ACTIVITY_SCALE_BINS.max(1);
        let mut seed = 0xCA62C1D6_u64 ^ 0x9B05688C_u64;
        let mut residuals = Array2::<f64>::zeros((n, p));
        let mut activity = Array1::<f64>::zeros(n);
        let mut c_true = Array1::<f64>::zeros(n);
        for row in 0..n {
            let u = (row as f64) / (n as f64 - 1.0);
            let z = u * u * u; // cubic warp ⇒ uneven bin occupancy
            activity[row] = z;
            let c = (slope * z).exp();
            c_true[row] = c;
            let amp = c.sqrt();
            let f = lcg_normal(&mut seed);
            for i in 0..p {
                residuals[[row, i]] = amp * lambda0[i] * f + sigma_eps * lcg_normal(&mut seed);
            }
        }

        // Empirical (undeflated) second moment T = (1/n) Σ_n r_n r_nᵀ — the
        // object the model's ΛΛᵀ + D must reconstruct.
        let mut t = Array2::<f64>::zeros((p, p));
        for i in 0..n {
            for a in 0..p {
                for b in 0..p {
                    t[[a, b]] += residuals[[i, a]] * residuals[[i, b]];
                }
            }
        }
        t.mapv_inplace(|v| v / n as f64);

        let raw_diag = column_variances(residuals.view());
        let mean_var = raw_diag.iter().sum::<f64>() / p as f64;
        let diag_floor = DIAGONAL_REL_FLOOR * mean_var.max(f64::MIN_POSITIVE);

        // Per-bin raw scale law: mean of the true c(z) within each bin.
        let row_bin = assign_bins(&activity, bins);
        let mut bin_sum = vec![0.0_f64; bins];
        let mut bin_cnt = vec![0.0_f64; bins];
        for i in 0..n {
            bin_sum[row_bin[i]] += c_true[i];
            bin_cnt[row_bin[i]] += 1.0;
        }
        let bin_raw: Vec<f64> = (0..bins)
            .map(|b| if bin_cnt[b] > 0.0 { bin_sum[b] / bin_cnt[b] } else { 1.0 })
            .collect();

        // Occupancy-weighted mean-1 (Fix A) vs bin-uniform mean-1 (old).
        let mean_occ = (0..bins).map(|b| bin_cnt[b] * bin_raw[b]).sum::<f64>() / n as f64;
        let occupied: Vec<usize> = (0..bins).filter(|&b| bin_cnt[b] > 0.0).collect();
        let mean_uni =
            occupied.iter().map(|&b| bin_raw[b]).sum::<f64>() / occupied.len() as f64;
        let row_scale_occ: Array1<f64> =
            (0..n).map(|i| bin_raw[row_bin[i]] / mean_occ).collect();
        let row_scale_uni: Array1<f64> =
            (0..n).map(|i| bin_raw[row_bin[i]] / mean_uni).collect();

        // One (Λ, D | scale) extraction from the deflated moment, mirroring the
        // production first sweep, returning the reconstruction ΛΛᵀ + D.
        let extract_recon = |row_scale: &Array1<f64>| -> Array2<f64> {
            let s = scaled_second_moment(residuals.view(), row_scale);
            let (evals, evecs) = symmetric_eig_ascending(&s).expect("eig");
            let mean_diag =
                raw_diag.iter().map(|&v| v.max(diag_floor)).sum::<f64>() / p as f64;
            let col = p - 1;
            let amp = (evals[col] - mean_diag).max(0.0).sqrt();
            let mut lam = Array1::<f64>::zeros(p);
            for j in 0..p {
                lam[j] = amp * evecs[[j, col]];
            }
            let mut recon = Array2::<f64>::zeros((p, p));
            for a in 0..p {
                for b in 0..p {
                    recon[[a, b]] = lam[a] * lam[b];
                }
            }
            for j in 0..p {
                let d = (raw_diag[j] - lam[j] * lam[j]).max(diag_floor);
                recon[[j, j]] += d;
            }
            recon
        };

        let frob = |m: &Array2<f64>| -> f64 {
            let mut acc = 0.0_f64;
            for a in 0..p {
                for b in 0..p {
                    let d = m[[a, b]] - t[[a, b]];
                    acc += d * d;
                }
            }
            acc.sqrt()
        };

        let dist_occ = frob(&extract_recon(&row_scale_occ));
        let dist_uni = frob(&extract_recon(&row_scale_uni));
        assert!(
            dist_occ < dist_uni,
            "occupancy-weighted reconstruction must beat bin-uniform: \
             ‖·‖_F occ = {dist_occ:.6} vs uni = {dist_uni:.6}"
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
            max_factor_rank: 2,
        })
        .expect("fit");
        (n, model)
    }

    /// WAVE-2 producer integration (#2021): the WhitenedStructured RowMetric from
    /// `row_metric` must deliver the exact Mahalanobis `vᵀ Σ_n^{-1} v` for
    /// `Σ_n = c_n·ΛΛᵀ + D` over the fitted occupancy-normalized scale. A
    /// deterministic refit reproduces the same metric (the seam `fit_row_metric`
    /// relies on).
    #[test]
    fn row_metric_precision_matches_woodbury_over_fitted_scale() {
        let lambda0 = ndarray::array![1.5, 1.2, -0.4, 0.3];
        let (n, model) = fit_small_model(0x14057B7EF767814F_u64, &lambda0);
        let p = 4usize;
        assert!(model.factor_rank() >= 1, "need a factor for a non-trivial Σ_n");

        let metric = model.row_metric(n).expect("row_metric");
        assert!(
            metric.whitens_likelihood(),
            "WhitenedStructured metric must whiten the likelihood"
        );

        let rank = model.factor_rank();
        let lam = model.factor();
        let diag = model.diagonal();
        let v: Array1<f64> = ndarray::array![0.7, -1.3, 0.4, 0.9];

        for &row in &[0usize, n / 3, n / 2, n - 1] {
            let c = model.row_scale()[row];
            let mut sigma = Array2::<f64>::zeros((p, p));
            for a in 0..p {
                for b in 0..p {
                    let mut fac = 0.0_f64;
                    for k in 0..rank {
                        fac += lam[[a, k]] * lam[[b, k]];
                    }
                    sigma[[a, b]] = c * fac;
                }
                sigma[[a, a]] += diag[a];
            }
            let chol = sigma.cholesky(Side::Lower).expect("Σ_n PD");
            let x = chol.solvevec(&v);
            let mahal_dense: f64 = v.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            let mahal_metric = metric.quad_form(row, v.view());
            assert!(
                (mahal_dense - mahal_metric).abs() <= 1e-8 * (1.0 + mahal_dense.abs()),
                "row {row}: metric quad_form {mahal_metric} must equal dense vᵀΣ⁻¹v {mahal_dense}"
            );
        }

        // Deterministic refit reproduces the identical metric (fit_row_metric seam).
        let (n2, model2) = fit_small_model(0x14057B7EF767814F_u64, &lambda0);
        assert_eq!(n2, n);
        let metric_again = model2.row_metric(n2).expect("row_metric again");
        for &row in &[0usize, n / 2, n - 1] {
            let q1 = metric.quad_form(row, v.view());
            let q2 = metric_again.quad_form(row, v.view());
            assert!(
                (q1 - q2).abs() <= 1e-12 * (1.0 + q1.abs()),
                "deterministic refit must match at row {row}: {q2} vs {q1}"
            );
        }
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
            let damped = model.row_metric_damped(n, 1.0, prev_opt).expect("damped γ=1");
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

        // γ = 0, prev = None ⇒ Euclidean identity: quad_form = ‖v‖².
        let ident = model.row_metric_damped(n, 0.0, None).expect("damped γ=0 None");
        let sumsq: f64 = v.iter().map(|x| x * x).sum();
        for row in [0usize, n / 2, n - 1] {
            let q = ident.quad_form(row, v.view());
            assert!(
                (q - sumsq).abs() <= 1e-12 * (1.0 + sumsq),
                "γ=0/None must be the identity metric: quad_form {q} vs ‖v‖² {sumsq}"
            );
        }

        // γ = 0, prev = Some ⇒ byte-identical to prev.row_metric.
        let prev_metric = prev.row_metric(n).expect("prev row_metric");
        let damped0 = model.row_metric_damped(n, 0.0, Some(&prev)).expect("damped γ=0 Some");
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
        let mid = model.row_metric_damped(n, 0.5, Some(&prev)).expect("damped γ=0.5");
        for row in [0usize, n / 2, n - 1] {
            let q = mid.quad_form(row, v.view());
            assert!(q.is_finite() && q > 0.0, "γ=0.5 metric must be SPD; got {q}");
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
        assert!(cur.promotion_candidates(Some(&prev), f64::NAN, 1.0).is_err());
    }
}
