//! `RowSamplingMeasure` — the Fisher-mass **enrichment** producer (role (c) of #980).
//!
//! # What this is, and what it must never be
//!
//! A [`RowSamplingMeasure`] turns a [`RowMetric`] into a per-row **sampling measure**:
//! a normalized non-negative weight per row, proportional to that row's
//! behavioral *liveness* (its output-Fisher mass). It exists for **discovery /
//! seeding only** — to OVERSAMPLE the behaviorally-live rows so that a rare but
//! behaviorally-important feature (few rows, high Fisher mass, drowned among
//! many common low-coupling rows) is actually *seen* by a discovery batch.
//!
//! ## The load-bearing invariant
//!
//! **The measure NEVER enters the reconstruction loss, the gradient, the
//! evidence criterion, or any optimizer-facing quantity.** Sampling ADDS
//! attention; it never reweights representation. Concretely:
//!
//! * it does not multiply any residual, any `quad_form`, any whitened Jacobian,
//!   or any penalty;
//! * it does not feed REML / LAML, the ρ trust-region ratio, or `φ̂`;
//! * it only chooses *which rows a discovery/seeding pass looks at first*, and
//!   how many times, leaving every per-row loss bit-for-bit unchanged.
//!
//! This is the dual of the #980 failure mode (where an output-Fisher inner
//! product silently replaced the reconstruction loss): here the Fisher mass is
//! used *strictly* as an attention prior over rows, with the loss untouched.
//! The enrichment ordering returns row indices with multiplicity — the consumer
//! visits those rows for *seeding/proposal* purposes; the fit it then runs on
//! any selected row uses the unmodified per-row objective.
//!
//! # Graceful degradation (absent harvest ⇒ today's behavior)
//!
//! The measure is **magic-by-default**, mirroring [`RowMetric`]:
//!
//! * [`MetricProvenance::Euclidean`] (no per-row Fisher factors were harvested)
//!   ⇒ every row's liveness is identical (`tr(I_p) = p`), so the measure is
//!   **exactly uniform** and the enrichment ordering is the plain index order
//!   with uniform multiplicity. Absent harvest is therefore bit-for-bit today's
//!   "look at every row equally" behavior, never an error.
//! * A factored provenance ([`MetricProvenance::OutputFisher`] /
//!   [`MetricProvenance::WhitenedStructured`]) ⇒ rows are weighted by their
//!   `tr(M_n)` Fisher mass, oversampling the live rows.
//!
//! Any pathological metric (all-zero mass, a non-finite block) also degrades to
//! the uniform measure rather than producing a degenerate or `NaN` sampling
//! distribution.
//!
//! # Why `tr(M_n)` is the right liveness scalar
//!
//! The per-row metric `M_n = U_n U_nᵀ` is the output-Fisher inner product on
//! latent motion at row `n`. Its trace `tr(M_n) = Σ_i e_iᵀ M_n e_i =
//! Σ_i fisher_mass(n, e_i)` is the total behavioral mass of that row summed over
//! output coordinates — basis-independent and exactly the quantity
//! [`RowMetric::fisher_mass`] reports for a unit of motion along each axis. It
//! is the canonical row liveness derivable from the metric *alone*, with no
//! external tangent supplied, and it collapses to the constant `p` under
//! Euclidean — which is precisely the uniform-measure degeneracy we want.

use faer::Side;
use gam_linalg::faer_ndarray::{FaerEigh, FaerSvd};
use gam_linalg::utils::splitmix64_hash;
use gam_problem::{MetricProvenance, RowMetric};
use ndarray::{Array2, ArrayView2};

/// Where a [`RowSamplingMeasure`] came from — the honest record of whether the
/// enrichment is real (Fisher-mass driven) or the graceful uniform fallback.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MeasureProvenance {
    /// No behavioral signal was available (Euclidean metric, or a degenerate
    /// metric that produced no usable mass). The measure is exactly uniform:
    /// every row carries weight `1 / n`. This is bit-for-bit "look at every row
    /// equally" — today's behavior with no harvest.
    Uniform,
    /// The measure is `∝ tr(M_n)` from a factored [`RowMetric`]. Behaviorally
    /// live rows carry proportionally more sampling weight. The carried
    /// [`MetricProvenance`] is the metric provenance that produced the mass, so
    /// a consumer can certify the inner product behind the enrichment.
    FisherMass(MetricProvenance),
}

/// A per-row **sampling measure** over `n` rows, normalized to sum to 1.
///
/// Built from a [`RowMetric`] via [`RowSamplingMeasure::from_metric`]. The weights are a
/// proper probability measure (non-negative, finite, summing to 1) used for
/// **discovery/seeding oversampling only** — see the module docs for the
/// invariant that it touches no loss / gradient / criterion.
#[derive(Clone, Debug)]
pub struct RowSamplingMeasure {
    provenance: MeasureProvenance,
    /// Normalized per-row sampling weights; `weights.len() == n_rows` and
    /// `Σ weights == 1` (exactly uniform `1/n` in the fallback).
    weights: Vec<f64>,
}

/// Certified coreset error budget carried to race consumers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoresetCertificate {
    /// Spectral approximation radius for the log-determinant term:
    /// `(1 - eps_spectral)H <= H_C <= (1 + eps_spectral)H` on the effective
    /// eigenspace.
    pub eps_spectral: f64,
    /// Additive likelihood error radius supplied by the sensitivity coreset on
    /// its documented chart ball.
    pub eps_likelihood: f64,
    /// Rank of the factored border plus active-coordinate subspace actually
    /// certified. Null directions of the summed row sketch are excluded.
    pub dim_effective: usize,
    /// Number of distinct rows retained by the coreset.
    pub n_selected: usize,
}

impl CoresetCertificate {
    pub fn new(
        eps_spectral: f64,
        eps_likelihood: f64,
        dim_effective: usize,
        n_selected: usize,
    ) -> Result<Self, String> {
        if !(eps_spectral.is_finite() && eps_spectral >= 0.0 && eps_spectral < 1.0) {
            return Err(format!(
                "coreset certificate requires 0 <= eps_spectral < 1, got {eps_spectral}"
            ));
        }
        if !(eps_likelihood.is_finite() && eps_likelihood >= 0.0) {
            return Err(format!(
                "coreset certificate requires finite non-negative eps_likelihood, got {eps_likelihood}"
            ));
        }
        Ok(Self {
            eps_spectral,
            eps_likelihood,
            dim_effective,
            n_selected,
        })
    }

    /// Worst-case log-determinant transfer error implied by the spectral
    /// certificate.
    pub fn logdet_error_bound(&self) -> f64 {
        self.dim_effective as f64 * ((1.0 + self.eps_spectral) / (1.0 - self.eps_spectral)).ln()
    }

    /// Race-transfer margin: consumers must require a coreset decision margin
    /// strictly above this value before inheriting the full-corpus verdict.
    pub fn race_transfer_margin(&self) -> f64 {
        2.0 * (self.logdet_error_bound() + self.eps_likelihood)
    }

    /// Explicit verdict for a proposed coreset race margin. Consumers should
    /// propagate [`CoresetMarginVerdict::InsufficientMargin`] instead of making
    /// a silent decision below the certificate margin.
    pub fn certify_margin(&self, decision_margin: f64) -> CoresetMarginVerdict {
        let required_margin = self.race_transfer_margin();
        if decision_margin.is_finite() && decision_margin > required_margin {
            CoresetMarginVerdict::Certified {
                decision_margin,
                required_margin,
            }
        } else {
            CoresetMarginVerdict::InsufficientMargin {
                decision_margin,
                required_margin,
            }
        }
    }
}

/// Certificate gate for coreset-backed race decisions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CoresetMarginVerdict {
    Certified {
        decision_margin: f64,
        required_margin: f64,
    },
    InsufficientMargin {
        decision_margin: f64,
        required_margin: f64,
    },
}

/// Output of deterministic BSS spectral row selection.
#[derive(Clone, Debug, PartialEq)]
pub struct SpectralCoreset {
    /// Distinct selected row indices, ascending.
    pub indices: Vec<usize>,
    /// Non-negative row weights aligned with `indices`.
    pub weights: Vec<f64>,
    /// Spectral certificate for this row coreset. `eps_likelihood` is zero here;
    /// combine with a sensitivity certificate before certifying full evidence.
    pub certificate: CoresetCertificate,
}

/// Deterministic Batson-Spielman-Srivastava spectral row coreset.
///
/// Each input item is a small row factor `R_i` with contribution
/// `H_i = R_i.t() R_i`. Selection is run on the effective eigenspace of
/// `sum_i H_i`; rank-null directions are excluded from the certificate. The
/// algorithm whitens the factors into that effective space, then applies the
/// standard two-barrier BSS potential update with deterministic row-index
/// tie-breaking. Per-row dense `H_i` blocks are never materialized.
///
/// Deterministic BSS spectral row coreset with the attached certificate.
pub fn bss_spectral_coreset_certified<'a, I>(
    rows: I,
    target_eps: f64,
) -> Result<SpectralCoreset, String>
where
    I: IntoIterator<Item = ArrayView2<'a, f64>>,
{
    if !(target_eps.is_finite() && target_eps > 0.0 && target_eps < 1.0) {
        return Err(format!(
            "BSS spectral coreset requires 0 < target_eps < 1, got {target_eps}"
        ));
    }

    let factors = collect_row_factors(rows)?;
    let n = factors.len();
    if n == 0 {
        let certificate = CoresetCertificate::new(target_eps, 0.0, 0, 0)?;
        return Ok(SpectralCoreset {
            indices: Vec::new(),
            weights: Vec::new(),
            certificate,
        });
    }

    let ambient_dim = factors[0].ncols();
    let effective = stacked_factor_whitener(&factors, ambient_dim)?;
    let dim = effective.ncols();
    if dim == 0 {
        let certificate = CoresetCertificate::new(target_eps, 0.0, 0, 0)?;
        return Ok(SpectralCoreset {
            indices: Vec::new(),
            weights: Vec::new(),
            certificate,
        });
    }

    let whitened = whiten_row_factors(&factors, &effective);
    let eta = 0.5 * target_eps;
    let steps = ((dim as f64) / (eta * eta)).ceil().max(dim as f64) as usize;
    let delta_lower = 1.0_f64;
    let delta_upper = (1.0 + eta) / (1.0 - eta);
    let root = (steps as f64 * dim as f64).sqrt();
    let mut barrier_matrix = Array2::<f64>::zeros((dim, dim));
    let mut row_weights = vec![0.0_f64; n];

    for step in 0..steps {
        let lower = step as f64 - root;
        let upper = delta_upper * (step as f64 + root);
        let lower_next = lower + delta_lower;
        let upper_next = upper + delta_upper;

        let lower_inv = inverse_shifted_lower(&barrier_matrix, lower_next)?;
        let upper_inv = inverse_shifted_upper(&barrier_matrix, upper_next)?;
        let lower_denom = lower_potential(&barrier_matrix, lower_next)?
            - lower_potential(&barrier_matrix, lower)?;
        let upper_denom = upper_potential(&barrier_matrix, upper)?
            - upper_potential(&barrier_matrix, upper_next)?;
        if !(lower_denom.is_finite() && lower_denom > 0.0) {
            return Err(format!(
                "BSS lower potential denominator became invalid at step {step}: {lower_denom}"
            ));
        }
        if !(upper_denom.is_finite() && upper_denom > 0.0) {
            return Err(format!(
                "BSS upper potential denominator became invalid at step {step}: {upper_denom}"
            ));
        }

        let mut chosen: Option<(usize, f64, f64)> = None;
        for (row, factor) in whitened.iter().enumerate() {
            let lower_trace = trace_factor_quadratic(factor, &lower_inv);
            let lower_trace_sq = trace_factor_quadratic_square(factor, &lower_inv);
            let upper_trace = trace_factor_quadratic(factor, &upper_inv);
            let upper_trace_sq = trace_factor_quadratic_square(factor, &upper_inv);
            let lower_score = lower_trace_sq / lower_denom - lower_trace;
            let upper_score = upper_trace_sq / upper_denom + upper_trace;
            if lower_score.is_finite()
                && upper_score.is_finite()
                && lower_score > 0.0
                && upper_score > 0.0
                && lower_score + BSS_SCORE_TOL >= upper_score
            {
                match chosen {
                    None => chosen = Some((row, lower_score, upper_score)),
                    Some((best_row, best_lower, best_upper)) => {
                        let gap = lower_score - upper_score;
                        let best_gap = best_lower - best_upper;
                        if gap > best_gap + BSS_SCORE_TOL
                            || ((gap - best_gap).abs() <= BSS_SCORE_TOL && row < best_row)
                        {
                            chosen = Some((row, lower_score, upper_score));
                        }
                    }
                }
            }
        }

        let (row, lower_score, upper_score) = chosen
            .ok_or_else(|| format!("BSS failed to find a barrier-admissible row at step {step}"))?;
        let inv_step_weight = 0.5 * (lower_score + upper_score);
        if !(inv_step_weight.is_finite() && inv_step_weight > 0.0) {
            return Err(format!(
                "BSS invalid inverse step weight at step {step}: {inv_step_weight}"
            ));
        }
        let step_weight = 1.0 / inv_step_weight;
        add_factor_gram_scaled(&mut barrier_matrix, &whitened[row], step_weight);
        row_weights[row] += step_weight;
    }

    let lower_final = steps as f64 - root;
    let upper_final = delta_upper * (steps as f64 + root);
    let scale = 2.0 / (lower_final + upper_final);
    let mut indexed: Vec<(usize, f64)> = row_weights
        .into_iter()
        .enumerate()
        .filter_map(|(row, weight)| {
            let scaled = weight * scale;
            (scaled > 0.0).then_some((row, scaled))
        })
        .collect();
    indexed.sort_by_key(|&(row, _)| row);
    let indices: Vec<usize> = indexed.iter().map(|&(row, _)| row).collect();
    let weights: Vec<f64> = indexed.iter().map(|&(_, weight)| weight).collect();
    let certificate = CoresetCertificate::new(target_eps, 0.0, dim, indices.len())?;
    Ok(SpectralCoreset {
        indices,
        weights,
        certificate,
    })
}

/// Sensitivity upper bounds on the chart ball
/// `||chart(theta) - chart(theta_anchor)|| <= chart_radius`.
///
/// The bound uses the linear-anchor leverage and inflates it by the curvature
/// slack `kappa_hat * chart_radius`, i.e.
/// `sigma_i <= leverage_i * (1 + kappa_hat * chart_radius)`. The same ball and
/// curvature estimate must be used by the likelihood consumer that interprets
/// the returned additive `eps_likelihood` certificate.
pub fn sensitivity_upper_bounds(
    linear_anchor_leverage: &[f64],
    kappa_hat: f64,
    chart_radius: f64,
) -> Result<Vec<f64>, String> {
    if !(kappa_hat.is_finite() && kappa_hat >= 0.0) {
        return Err(format!(
            "sensitivity bounds require finite non-negative kappa_hat, got {kappa_hat}"
        ));
    }
    if !(chart_radius.is_finite() && chart_radius >= 0.0) {
        return Err(format!(
            "sensitivity bounds require finite non-negative chart_radius, got {chart_radius}"
        ));
    }
    let inflation = 1.0 + kappa_hat * chart_radius;
    linear_anchor_leverage
        .iter()
        .enumerate()
        .map(|(row, &lev)| {
            if lev.is_finite() && lev >= 0.0 {
                Ok(lev * inflation)
            } else {
                Err(format!(
                    "sensitivity leverage at row {row} must be finite and non-negative, got {lev}"
                ))
            }
        })
        .collect()
}

/// Greedy deterministic sensitivity coreset under a row budget.
#[derive(Clone, Debug, PartialEq)]
pub struct SensitivityCoreset {
    /// Selected rows sorted by decreasing sensitivity, then row index.
    pub indices: Vec<usize>,
    /// Sensitivity mass retained by the selected rows.
    pub selected_sensitivity_mass: f64,
    /// Sensitivity mass not retained by the budget. A likelihood consumer can
    /// map this to its additive `eps_likelihood` on the documented chart ball.
    pub residual_sensitivity_mass: f64,
}

pub fn greedy_sensitivity_coreset(
    sigma_upper_bounds: &[f64],
    budget: usize,
) -> Result<SensitivityCoreset, String> {
    let mut indexed = Vec::with_capacity(sigma_upper_bounds.len());
    for (row, &sigma) in sigma_upper_bounds.iter().enumerate() {
        if !(sigma.is_finite() && sigma >= 0.0) {
            return Err(format!(
                "sensitivity upper bound at row {row} must be finite and non-negative, got {sigma}"
            ));
        }
        indexed.push((row, sigma));
    }
    indexed.sort_by(|&(row_a, sigma_a), &(row_b, sigma_b)| {
        sigma_b
            .partial_cmp(&sigma_a)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(row_a.cmp(&row_b))
    });
    let selected_len = budget.min(indexed.len());
    let indices: Vec<usize> = indexed
        .iter()
        .take(selected_len)
        .map(|&(row, _)| row)
        .collect();
    let selected_sensitivity_mass: f64 = indexed
        .iter()
        .take(selected_len)
        .map(|&(_, sigma)| sigma)
        .sum();
    let residual_sensitivity_mass: f64 = indexed
        .iter()
        .skip(selected_len)
        .map(|&(_, sigma)| sigma)
        .sum();
    Ok(SensitivityCoreset {
        indices,
        selected_sensitivity_mass,
        residual_sensitivity_mass,
    })
}

impl RowSamplingMeasure {
    /// Build the enrichment measure from a [`RowMetric`].
    ///
    /// The per-row liveness is the Fisher mass `tr(M_n)` read from the metric's
    /// validated PSD blocks. The result is normalized to a proper sampling
    /// measure. Degrades to the **uniform** measure (every row `1/n`) when the
    /// metric is Euclidean, carries no usable mass (all rows ≤ 0), or yields any
    /// non-finite mass — never an error, mirroring [`RowMetric`]'s
    /// magic-by-default discipline.
    ///
    /// This function reads only the metric's geometry; it writes nothing into
    /// the metric, the loss, the gradient, or any criterion.
    pub fn from_metric(metric: &RowMetric) -> Self {
        let n = metric.n_rows();
        if n == 0 {
            return Self {
                provenance: MeasureProvenance::Uniform,
                weights: Vec::new(),
            };
        }

        // Euclidean ⇒ exactly uniform by construction. Short-circuit so the
        // fallback is bit-for-bit `1/n`, not "tr(I_p)=p then renormalize" (which
        // is the same value, but the explicit path documents intent and avoids
        // any floating-point renormalization noise).
        if matches!(metric.provenance(), MetricProvenance::Euclidean) {
            return Self::uniform(n);
        }

        let mass = per_row_fisher_mass(metric);
        Self::from_masses(metric.provenance(), mass)
    }

    /// The uniform measure over `n` rows: every row weight `1 / n`. The graceful
    /// fallback and the explicit "no behavioral harvest" measure.
    pub fn uniform(n: usize) -> Self {
        let w = if n == 0 { 0.0 } else { 1.0 / n as f64 };
        Self {
            provenance: MeasureProvenance::Uniform,
            weights: vec![w; n],
        }
    }

    /// Construct from raw per-row masses, normalizing to a proper measure.
    /// Falls back to uniform if the masses carry no usable signal.
    ///
    /// Crate-visible so the two-tier harvest (`gam_inference::harvest`)
    /// can lift designed-subsample Fisher masses to a full-corpus measure
    /// through the same validation/normalization path.
    pub fn from_masses(metric_provenance: MetricProvenance, masses: Vec<f64>) -> Self {
        let n = masses.len();
        if n == 0 {
            return Self::uniform(0);
        }
        // Clamp negatives to zero (a validated PSD block has `tr ≥ 0`, but a
        // tiny normalizer round-off could dip below) and reject non-finite.
        let mut total = 0.0_f64;
        let mut clean = vec![0.0_f64; n];
        let mut all_finite = true;
        for (i, &m) in masses.iter().enumerate() {
            if !m.is_finite() {
                all_finite = false;
                break;
            }
            let v = if m > 0.0 { m } else { 0.0 };
            clean[i] = v;
            total += v;
        }

        if !all_finite || !(total > 0.0) {
            // No usable behavioral signal ⇒ degrade to uniform, never NaN.
            return Self::uniform(n);
        }

        let inv = 1.0 / total;
        for w in clean.iter_mut() {
            *w *= inv;
        }
        Self {
            provenance: MeasureProvenance::FisherMass(metric_provenance),
            weights: clean,
        }
    }

    /// The normalized per-row sampling weights (`Σ == 1`). Read-only; this is a
    /// sampling measure, never a loss weight.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// The measure's provenance — `Uniform` (graceful fallback / no harvest) or
    /// `FisherMass` (real behavioral enrichment).
    pub fn provenance(&self) -> MeasureProvenance {
        self.provenance
    }

    /// Number of rows the measure is defined over.
    pub fn n_rows(&self) -> usize {
        self.weights.len()
    }

    /// Whether this measure actually enriches (is non-uniform Fisher-mass).
    /// `false` for the uniform fallback.
    pub fn is_enriched(&self) -> bool {
        matches!(self.provenance, MeasureProvenance::FisherMass(_))
    }

    /// Deterministic **systematic-resampling** enrichment ordering.
    ///
    /// Returns a length-`count` vector of row indices drawn `∝ weights`, using
    /// low-variance systematic resampling with a fixed, *index-derived* jitter —
    /// there is **no clock randomness**; the same `(measure, count, seed)`
    /// always yields the same ordering. Behaviorally-live rows therefore appear
    /// with multiplicity proportional to their Fisher mass, so a rare-but-live
    /// feature's rows are oversampled relative to uniform.
    ///
    /// Systematic resampling places `count` equally spaced pointers
    /// `(j + u) / count`, `j = 0..count`, against the cumulative weight CDF and
    /// emits the row each pointer lands in. The single offset `u ∈ [0, 1)` is a
    /// `splitmix64`-hash of `seed` (deterministic), giving an unbiased draw
    /// whose per-row expected count is `count · weights[row]` while guaranteeing
    /// every weight-`≥ 1/count` row appears at least once (the recall property
    /// the rare-feature control asserts).
    ///
    /// The uniform fallback reproduces an even, deterministic round-robin over
    /// all rows — i.e. plain attention to every row, today's behavior.
    ///
    /// This ordering is consumed **only** by a discovery/seeding pass. The rows
    /// it names carry their ordinary, unmodified per-row objective.
    pub fn enrichment_order(&self, count: usize, seed: u64) -> Vec<usize> {
        let n = self.weights.len();
        if n == 0 || count == 0 {
            return Vec::new();
        }

        // Deterministic offset u ∈ [0, 1) from the seed (index-/seed-derived,
        // never the clock). 53-bit mantissa for an exact double in [0, 1).
        let u = {
            let bits = splitmix64_hash(seed ^ ENRICHMENT_SALT);
            let mantissa = (bits >> 11) as f64; // top 53 bits
            mantissa / ((1_u64 << 53) as f64)
        };

        // Cumulative distribution over rows. `weights` already sums to 1; guard
        // the last bucket to exactly 1.0 against round-off so every pointer
        // lands in a valid row.
        let mut cdf = vec![0.0_f64; n];
        let mut acc = 0.0_f64;
        for i in 0..n {
            acc += self.weights[i];
            cdf[i] = acc;
        }
        cdf[n - 1] = 1.0;

        let mut out = Vec::with_capacity(count);
        let step = 1.0 / count as f64;
        let mut cursor = 0usize;
        for j in 0..count {
            let pointer = (j as f64 + u) * step;
            // Advance the CDF cursor to the first bucket whose cumulative mass
            // covers the pointer. Monotone in `j`, so this is one linear sweep.
            while cursor < n - 1 && pointer > cdf[cursor] {
                cursor += 1;
            }
            out.push(cursor);
        }
        out
    }

    /// Expected number of times each row is drawn in a `count`-sized enrichment
    /// batch: `count · weights[row]`. A diagnostic for the discovery-recall
    /// control — it lets a test assert that a rare-but-live feature's rows have
    /// markedly higher expected representation under enrichment than under
    /// uniform, with no sampling noise.
    pub fn expected_representation(&self, count: usize) -> Vec<f64> {
        let c = count as f64;
        self.weights.iter().map(|&w| c * w).collect()
    }

    /// Draw a **designed subsample** with honest inclusion weights — the
    /// frontier estimator of #987 (mechanizing the #973 subsample-honesty
    /// contract for measure-driven designs).
    ///
    /// This is a different animal from [`Self::enrichment_order`], and the
    /// distinction is load-bearing:
    ///
    /// * **Enrichment** orders rows for *discovery/seeding attention*; each
    ///   visited row keeps its ordinary, unweighted per-row objective. The
    ///   measure never touches the loss.
    /// * A **designed subsample** *replaces the full corpus* as what the fit
    ///   sums over. That is only sound if every selected row's loss term is
    ///   multiplied by `1 / π_i` (its inclusion probability), so that the
    ///   subsampled criterion is **unbiased** for the full-corpus criterion:
    ///   `E[Σ_{i ∈ S} ℓ_i / π_i] = Σ_i ℓ_i`. The returned
    ///   [`DesignedRowSample`] carries exactly those weights; the caller folds
    ///   them into the likelihood as row weights. These are sampling-design
    ///   corrections — they are *not* a Fisher reweighting of residuals (the
    ///   #980 failure mode), and under the uniform measure they degrade to the
    ///   constant `n / budget`, the plain Horvitz–Thompson scale-up.
    ///
    /// Design: inclusion probabilities are water-filled as
    /// `π_i = min(1, τ · w'_i)` with `τ` solved so `Σ π_i = budget`, where
    /// `w'` is the measure defensively mixed with
    /// [`DESIGNED_SAMPLE_UNIFORM_MIX`] of uniform — the standard
    /// defensive-mixture guard that keeps every row's `π_i > 0` (no row's loss
    /// is unreachable, so the estimator stays unbiased) and bounds the largest
    /// weight. Selection is Madow systematic sampling against the cumulative
    /// `π` with a single deterministic `splitmix64`-derived offset — no clock
    /// randomness; the same `(measure, budget, seed)` always yields the same
    /// sample. Rows are returned in ascending order (stream-friendly).
    ///
    /// `budget ≥ n` returns every row with weight `1.0` — the exact full pass,
    /// bit-for-bit today's behavior, so a driver can call this unconditionally
    /// and let the budget decide.
    pub fn designed_subsample(&self, budget: usize, seed: u64) -> DesignedRowSample {
        let n = self.weights.len();
        if n == 0 || budget == 0 {
            return DesignedRowSample {
                provenance: self.provenance,
                rows: Vec::new(),
                likelihood_weights: Vec::new(),
                expected_size: 0.0,
            };
        }
        if budget >= n {
            return DesignedRowSample {
                provenance: self.provenance,
                rows: (0..n).collect(),
                likelihood_weights: vec![1.0; n],
                expected_size: n as f64,
            };
        }

        // Defensive mixture: w' = (1 − ε)·w + ε/n. Keeps every π_i > 0.
        let eps = DESIGNED_SAMPLE_UNIFORM_MIX;
        let unif = 1.0 / n as f64;
        let mixed: Vec<f64> = self
            .weights
            .iter()
            .map(|&w| (1.0 - eps) * w + eps * unif)
            .collect();

        // Water-fill τ so that Σ min(1, τ·w'_i) = budget. Sort descending and
        // peel off the capped prefix; deterministic (index tie-break).
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            mixed[b]
                .partial_cmp(&mixed[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        let total: f64 = mixed.iter().sum();
        let mut capped = 0usize;
        let mut tail_mass = total;
        let mut tau = budget as f64 / tail_mass;
        while capped < n {
            let next = mixed[order[capped]];
            if tau * next <= 1.0 {
                break;
            }
            // Cap this row at π = 1 and re-solve τ over the remainder.
            capped += 1;
            tail_mass -= next;
            let remaining_budget = budget as f64 - capped as f64;
            if remaining_budget <= 0.0 || tail_mass <= 0.0 {
                break;
            }
            tau = remaining_budget / tail_mass;
        }
        let mut pi = vec![0.0_f64; n];
        for (rank, &i) in order.iter().enumerate() {
            pi[i] = if rank < capped {
                1.0
            } else {
                (tau * mixed[i]).min(1.0)
            };
        }

        // Madow systematic selection in row order: row i is selected iff an
        // integer pointer k + u falls inside its cumulative-π interval.
        // Deterministic offset u ∈ [0, 1) from the seed.
        let u = {
            let bits = splitmix64_hash(seed ^ DESIGNED_SAMPLE_SALT);
            let mantissa = (bits >> 11) as f64;
            mantissa / ((1_u64 << 53) as f64)
        };
        let mut rows = Vec::with_capacity(budget + 1);
        let mut likelihood_weights = Vec::with_capacity(budget + 1);
        let mut acc = 0.0_f64;
        for (i, &p) in pi.iter().enumerate() {
            let before = acc;
            acc += p;
            // Selected iff ⌊acc − u⌋ > ⌊before − u⌋ (a pointer crossed).
            if (acc - u).floor() > (before - u).floor() {
                rows.push(i);
                likelihood_weights.push(1.0 / p);
            }
        }
        DesignedRowSample {
            provenance: self.provenance,
            rows,
            likelihood_weights,
            expected_size: pi.iter().sum(),
        }
    }

    /// Draw a **certified** designed subsample within a target `eps` of the full
    /// corpus on BOTH evidence halves (#1012).
    ///
    /// Unlike [`Self::designed_subsample`] — whose Horvitz–Thompson design is
    /// unbiased only in expectation — this is the deterministic CERTIFIED mode:
    ///
    /// * **spectral half (`½log|H|`):** deterministic Batson–Spielman–Srivastava
    ///   selection of `O(dim/eps²)` weighted rows from the per-row factors
    ///   `R_i` (`H_i = R_iᵀR_i`), giving `(1−eps)H ⪯ H_C ⪯ (1+eps)H` and hence
    ///   `|log|H_C| − log|H|| ≤ dim·log((1+eps)/(1−eps))`;
    /// * **likelihood half (`L`):** the sensitivity bounds
    ///   `σ_i ≤ leverage_i·(1 + κ̂·chart_radius)` on the documented chart ball,
    ///   greedily selected against the row budget; the residual sensitivity mass
    ///   is the additive `eps_likelihood·L` the certificate carries.
    ///
    /// The two selections are unioned (a row certified for either half is kept),
    /// the rows carry their deterministic BSS / sensitivity weights, and the
    /// [`CoresetCertificate`] rides the result so a race consumer can gate the
    /// transfer with [`CoresetCertificate::race_transfer_margin`] — the SAME
    /// margin seam the enclosure path (#1011) declares. Below that margin the
    /// consumer must grow the coreset, never silently decide.
    ///
    /// `row_factors` is the per-row factor list aligned with this measure's rows;
    /// `leverage`, `kappa_hat`, `chart_radius` are the sensitivity inputs (the
    /// #1007 SVD-anchor leverage and the #1008 curvature slack). `budget` caps
    /// the likelihood-half greedy selection.
    pub fn designed_subsample_certified<'a, I>(
        &self,
        row_factors: I,
        target_eps: f64,
        leverage: &[f64],
        kappa_hat: f64,
        chart_radius: f64,
        budget: usize,
    ) -> Result<CertifiedRowSample, String>
    where
        I: IntoIterator<Item = ArrayView2<'a, f64>>,
    {
        // Spectral half: deterministic BSS coreset + its spectral certificate.
        let spectral = bss_spectral_coreset_certified(row_factors, target_eps)?;

        // Likelihood half: sensitivity-bounded greedy coreset; the residual mass
        // not covered by the budget becomes the additive eps_likelihood.
        let sigma = sensitivity_upper_bounds(leverage, kappa_hat, chart_radius)?;
        let sensitivity = greedy_sensitivity_coreset(&sigma, budget)?;
        let total_sensitivity =
            sensitivity.selected_sensitivity_mass + sensitivity.residual_sensitivity_mass;
        let eps_likelihood = if total_sensitivity > 0.0 {
            sensitivity.residual_sensitivity_mass / total_sensitivity
        } else {
            0.0
        };

        // Union the two selections; a row certified for either half is retained.
        // Carry the BSS weight where present, else the HT scale-up `1/π` proxy
        // (uniform `n/|S|`) so the likelihood-only rows still enter the criterion
        // unbiasedly.
        let n = self.weights.len();
        let bss_weight: std::collections::BTreeMap<usize, f64> = spectral
            .indices
            .iter()
            .zip(spectral.weights.iter())
            .map(|(&i, &w)| (i, w))
            .collect();
        let mut selected: std::collections::BTreeSet<usize> =
            spectral.indices.iter().copied().collect();
        for &i in &sensitivity.indices {
            selected.insert(i);
        }
        let selected_len = selected.len().max(1);
        let ht_scale = if n > 0 {
            n as f64 / selected_len as f64
        } else {
            1.0
        };

        let rows: Vec<usize> = selected.iter().copied().collect();
        let weights: Vec<f64> = rows
            .iter()
            .map(|i| *bss_weight.get(i).unwrap_or(&ht_scale))
            .collect();

        let certificate = CoresetCertificate::new(
            spectral.certificate.eps_spectral,
            eps_likelihood,
            spectral.certificate.dim_effective,
            rows.len(),
        )?;

        Ok(CertifiedRowSample {
            provenance: self.provenance,
            rows,
            weights,
            certificate,
        })
    }
}

/// A designed importance subsample with honest Horvitz–Thompson likelihood
/// weights — what a frontier fit sums over instead of the full corpus
/// (#987 / #973). Produced by [`RowSamplingMeasure::designed_subsample`].
#[derive(Clone, Debug)]
pub struct DesignedRowSample {
    /// Provenance of the measure that shaped the design (uniform fallback or
    /// Fisher mass), echoed for consumer certification.
    pub provenance: MeasureProvenance,
    /// Selected row indices, ascending.
    pub rows: Vec<usize>,
    /// Per-selected-row likelihood weight `1 / π_i`, aligned with `rows`.
    /// Multiplying row `i`'s loss term by this makes the subsampled criterion
    /// unbiased for the full-corpus criterion.
    pub likelihood_weights: Vec<f64>,
    /// `Σ π_i` — the design's expected sample size (≈ the requested budget;
    /// Madow selection realizes `⌊·⌋` or `⌈·⌉` of it).
    pub expected_size: f64,
}

impl DesignedRowSample {
    /// Number of rows actually selected.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// `Σ 1/π_i` over the selected rows — the Horvitz–Thompson estimate of the
    /// corpus row count. A consumer can sanity-gate the design by checking
    /// this lands near `n` (it is exactly `n` in expectation).
    pub fn estimated_corpus_rows(&self) -> f64 {
        self.likelihood_weights.iter().sum()
    }
}

/// A **certified** designed subsample (#1012): the rows that certify BOTH
/// evidence halves within the target `eps`, their deterministic BSS /
/// sensitivity weights, and the [`CoresetCertificate`] a race consumer gates
/// the verdict transfer against. Produced by
/// [`RowSamplingMeasure::designed_subsample_certified`].
#[derive(Clone, Debug)]
pub struct CertifiedRowSample {
    /// Provenance of the measure that shaped the design.
    pub provenance: MeasureProvenance,
    /// Selected row indices, ascending (union of the spectral and sensitivity
    /// coresets).
    pub rows: Vec<usize>,
    /// Per-selected-row weight aligned with `rows`: the BSS spectral weight
    /// where the row was chosen for the log-determinant half, else the
    /// Horvitz–Thompson scale-up for a likelihood-only row.
    pub weights: Vec<f64>,
    /// The certificate bounding the worst-case evidence transfer error. Feed
    /// [`CoresetCertificate::race_transfer_margin`] to the race consumer's
    /// margin gate.
    pub certificate: CoresetCertificate,
}

impl CertifiedRowSample {
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// The race-transfer margin a consumer must clear before inheriting the
    /// full-corpus verdict from this coreset — the shared #1011/#1012 seam.
    pub fn race_transfer_margin(&self) -> f64 {
        self.certificate.race_transfer_margin()
    }
}

/// Defensive uniform mixture fraction for [`RowSamplingMeasure::designed_subsample`]:
/// the design samples from `(1 − ε)·measure + ε·uniform`. Guarantees every
/// row's inclusion probability is positive (unbiasedness needs `π_i > 0`
/// wherever `ℓ_i ≠ 0`) and caps the worst-case `1/π` weight at
/// `n / (ε · budget)`. The standard defensive-importance-sampling guard.
const DESIGNED_SAMPLE_UNIFORM_MIX: f64 = 0.1;

/// Salt for the designed-sample systematic offset, distinct from
/// [`ENRICHMENT_SALT`] so the two draws never share a stream for one seed.
const DESIGNED_SAMPLE_SALT: u64 = 0x73AD_0987_5EED_D51F;

/// Salt mixed into the enrichment seed so the offset hash is distinct from any
/// other `splitmix64_hash` use of the same numeric seed elsewhere in the crate.
const ENRICHMENT_SALT: u64 = 0x980E_1C45_F00D_AC70;

const BSS_SCORE_TOL: f64 = 1e-10;

/// Per-row Fisher mass `tr(M_n)` from the metric's criterion-facing traces.
///
/// The traces are recorded at metric construction (un-floored), so the solver
/// `δ` never enters the measure — consistent with the `RowMetric` #747
/// discipline, and irrelevant anyway because the measure feeds no criterion.
/// Pure read; touches nothing.
pub fn per_row_fisher_mass(metric: &RowMetric) -> Vec<f64> {
    metric.row_traces().to_vec()
}

fn collect_row_factors<'a, I>(rows: I) -> Result<Vec<Array2<f64>>, String>
where
    I: IntoIterator<Item = ArrayView2<'a, f64>>,
{
    let mut out = Vec::new();
    let mut ambient_dim: Option<usize> = None;
    for (row, factor) in rows.into_iter().enumerate() {
        if factor.iter().any(|value| !value.is_finite()) {
            return Err(format!("BSS row factor {row} contains a non-finite value"));
        }
        match ambient_dim {
            None => ambient_dim = Some(factor.ncols()),
            Some(expected) if expected != factor.ncols() => {
                return Err(format!(
                    "BSS row factor {row} has {} columns, expected {expected}",
                    factor.ncols()
                ));
            }
            Some(_) => {}
        }
        out.push(factor.to_owned());
    }
    Ok(out)
}

fn stacked_factor_whitener(
    factors: &[Array2<f64>],
    ambient_dim: usize,
) -> Result<Array2<f64>, String> {
    let total_factor_rows: usize = factors.iter().map(|factor| factor.nrows()).sum();
    if total_factor_rows == 0 || ambient_dim == 0 {
        return Ok(Array2::<f64>::zeros((ambient_dim, 0)));
    }

    let mut stacked = Array2::<f64>::zeros((total_factor_rows, ambient_dim));
    let mut cursor = 0usize;
    for factor in factors {
        for row in 0..factor.nrows() {
            for col in 0..ambient_dim {
                stacked[[cursor + row, col]] = factor[[row, col]];
            }
        }
        cursor += factor.nrows();
    }

    let (_, singular, vt) = stacked
        .svd(false, true)
        .map_err(|err| format!("BSS stacked row-factor SVD failed: {err}"))?;
    let vt = vt.ok_or_else(|| "BSS stacked row-factor SVD did not return Vt".to_string())?;
    let max_sigma = singular.iter().copied().fold(0.0_f64, f64::max);
    if !(max_sigma.is_finite() && max_sigma >= 0.0) {
        return Err("BSS stacked row sketch has invalid singular values".to_string());
    }
    let tol = (ambient_dim.max(1) as f64) * f64::EPSILON * max_sigma.max(1.0) * 100.0;
    let kept: Vec<usize> = singular
        .iter()
        .enumerate()
        .filter_map(|(idx, &sigma)| (sigma > tol).then_some(idx))
        .collect();
    let mut whitener = Array2::<f64>::zeros((ambient_dim, kept.len()));
    for (out_col, &sv_col) in kept.iter().enumerate() {
        let scale = 1.0 / singular[sv_col];
        for ambient_col in 0..ambient_dim {
            whitener[[ambient_col, out_col]] = vt[[sv_col, ambient_col]] * scale;
        }
    }
    Ok(whitener)
}

fn whiten_row_factors(factors: &[Array2<f64>], whitener: &Array2<f64>) -> Vec<Array2<f64>> {
    factors.iter().map(|factor| factor.dot(whitener)).collect()
}

fn inverse_shifted_lower(matrix: &Array2<f64>, lower: f64) -> Result<Array2<f64>, String> {
    let n = matrix.nrows();
    let mut shifted = matrix.clone();
    for i in 0..n {
        shifted[[i, i]] -= lower;
    }
    inverse_symmetric_positive(&shifted, "BSS lower barrier inverse")
}

fn inverse_shifted_upper(matrix: &Array2<f64>, upper: f64) -> Result<Array2<f64>, String> {
    let n = matrix.nrows();
    let mut shifted = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        shifted[[i, i]] = upper;
    }
    for i in 0..n {
        for j in 0..n {
            shifted[[i, j]] -= matrix[[i, j]];
        }
    }
    inverse_symmetric_positive(&shifted, "BSS upper barrier inverse")
}

fn inverse_symmetric_positive(matrix: &Array2<f64>, context: &str) -> Result<Array2<f64>, String> {
    let (evals, evecs) = matrix
        .eigh(Side::Lower)
        .map_err(|err| format!("{context} eigendecomposition failed: {err}"))?;
    let n = matrix.nrows();
    let max_eval = evals.iter().copied().fold(0.0_f64, f64::max).max(1.0);
    let tol = (n.max(1) as f64) * f64::EPSILON * max_eval * 100.0;
    let mut inv = Array2::<f64>::zeros((n, n));
    for k in 0..n {
        let lambda = evals[k];
        if !(lambda.is_finite() && lambda > tol) {
            return Err(format!(
                "{context} expected a positive barrier matrix, eigenvalue {k} was {lambda}"
            ));
        }
        let inv_lambda = 1.0 / lambda;
        for i in 0..n {
            for j in 0..n {
                inv[[i, j]] += evecs[[i, k]] * inv_lambda * evecs[[j, k]];
            }
        }
    }
    Ok(inv)
}

fn lower_potential(matrix: &Array2<f64>, lower: f64) -> Result<f64, String> {
    let inv = inverse_shifted_lower(matrix, lower)?;
    Ok((0..inv.nrows()).map(|i| inv[[i, i]]).sum())
}

fn upper_potential(matrix: &Array2<f64>, upper: f64) -> Result<f64, String> {
    let inv = inverse_shifted_upper(matrix, upper)?;
    Ok((0..inv.nrows()).map(|i| inv[[i, i]]).sum())
}

fn trace_factor_quadratic(factor: &Array2<f64>, matrix: &Array2<f64>) -> f64 {
    let mut trace = 0.0_f64;
    for row in 0..factor.nrows() {
        for i in 0..factor.ncols() {
            let xi = factor[[row, i]];
            if xi == 0.0 {
                continue;
            }
            for j in 0..factor.ncols() {
                trace += xi * matrix[[i, j]] * factor[[row, j]];
            }
        }
    }
    trace
}

fn trace_factor_quadratic_square(factor: &Array2<f64>, matrix: &Array2<f64>) -> f64 {
    let mut trace = 0.0_f64;
    for row in 0..factor.nrows() {
        for i in 0..factor.ncols() {
            let mut v = 0.0_f64;
            for j in 0..factor.ncols() {
                v += matrix[[i, j]] * factor[[row, j]];
            }
            trace += v * v;
        }
    }
    trace
}

fn add_factor_gram_scaled(target: &mut Array2<f64>, factor: &Array2<f64>, scale: f64) {
    let dim = factor.ncols();
    for row in 0..factor.nrows() {
        for i in 0..dim {
            let xi = factor[[row, i]];
            if xi == 0.0 {
                continue;
            }
            for j in 0..dim {
                target[[i, j]] += scale * xi * factor[[row, j]];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use ndarray::array;
    use std::sync::Arc;

    fn summed_factor_gram(factors: &[Array2<f64>], ambient_dim: usize) -> Array2<f64> {
        let mut total = Array2::<f64>::zeros((ambient_dim, ambient_dim));
        for factor in factors {
            add_factor_gram_scaled(&mut total, factor, 1.0);
        }
        total
    }

    fn factors_from_rows(rows: &[Vec<f64>], p: usize, rank: usize) -> Arc<Array2<f64>> {
        let n = rows.len();
        let mut u = Array2::<f64>::zeros((n, p * rank));
        for (r, row) in rows.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                u[[r, c]] = v;
            }
        }
        Arc::new(u)
    }

    #[test]
    fn euclidean_degrades_to_uniform() {
        let metric = RowMetric::euclidean(5, 3).expect("euclidean");
        let measure = RowSamplingMeasure::from_metric(&metric);
        assert_eq!(measure.provenance(), MeasureProvenance::Uniform);
        assert!(!measure.is_enriched());
        for &w in measure.weights() {
            assert!((w - 0.2).abs() < 1e-12);
        }
    }

    #[test]
    fn weights_normalize_to_one_and_track_mass() {
        // p = 1, rank = 1 ⇒ tr(M_n) = u_n². Row 2 is far louder.
        let rows = vec![vec![1.0], vec![1.0], vec![3.0], vec![1.0]];
        let u = factors_from_rows(&rows, 1, 1);
        let metric = RowMetric::output_fisher(u, 1, 1).expect("of");
        let measure = RowSamplingMeasure::from_metric(&metric);
        assert!(measure.is_enriched());
        let w = measure.weights();
        let sum: f64 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
        // tr masses: 1, 1, 9, 1 ⇒ total 12.
        assert!((w[0] - 1.0 / 12.0).abs() < 1e-12);
        assert!((w[2] - 9.0 / 12.0).abs() < 1e-12);
        assert!(w[2] > w[0] * 8.0);
    }

    #[test]
    fn all_zero_mass_degrades_to_uniform() {
        let rows = vec![vec![0.0], vec![0.0], vec![0.0]];
        let u = factors_from_rows(&rows, 1, 1);
        let metric = RowMetric::output_fisher(u, 1, 1).expect("of");
        let measure = RowSamplingMeasure::from_metric(&metric);
        assert_eq!(measure.provenance(), MeasureProvenance::Uniform);
        for &w in measure.weights() {
            assert!((w - 1.0 / 3.0).abs() < 1e-12);
        }
    }

    #[test]
    fn enrichment_order_is_deterministic() {
        let rows = vec![vec![1.0], vec![3.0], vec![1.0]];
        let u = factors_from_rows(&rows, 1, 1);
        let metric = RowMetric::output_fisher(u, 1, 1).expect("of");
        let measure = RowSamplingMeasure::from_metric(&metric);
        let a = measure.enrichment_order(20, 7);
        let b = measure.enrichment_order(20, 7);
        assert_eq!(a, b, "same seed must give identical ordering");
        let c = measure.enrichment_order(20, 8);
        // Different seed ⇒ (generally) different ordering, but same length.
        assert_eq!(c.len(), 20);
    }

    #[test]
    fn enrichment_oversamples_loud_row() {
        // Row 1 has 9x the mass of rows 0 and 2.
        let rows = vec![vec![1.0], vec![3.0], vec![1.0]];
        let u = factors_from_rows(&rows, 1, 1);
        let metric = RowMetric::output_fisher(u, 1, 1).expect("of");
        let measure = RowSamplingMeasure::from_metric(&metric);
        let count = 110;
        let order = measure.enrichment_order(count, 1);
        let loud = order.iter().filter(|&&r| r == 1).count();
        let quiet0 = order.iter().filter(|&&r| r == 0).count();
        // Expected: 9/11 of 110 = 90 for the loud row, 10 each for the quiet.
        assert!(
            loud > quiet0 * 5,
            "loud row must be oversampled: loud={loud} quiet0={quiet0}"
        );
    }

    #[test]
    fn expected_representation_matches_count_times_weight() {
        let rows = vec![vec![1.0], vec![3.0]];
        let u = factors_from_rows(&rows, 1, 1);
        let metric = RowMetric::output_fisher(u, 1, 1).expect("of");
        let measure = RowSamplingMeasure::from_metric(&metric);
        let rep = measure.expected_representation(10);
        // masses 1, 9 ⇒ weights 0.1, 0.9 ⇒ reps 1.0, 9.0.
        assert!((rep[0] - 1.0).abs() < 1e-12);
        assert!((rep[1] - 9.0).abs() < 1e-12);
    }

    #[test]
    fn designed_subsample_is_deterministic_and_honest() {
        // 200 rows, one loud block. The design must (a) be reproducible for a
        // fixed seed, (b) carry weights 1/π whose HT total estimates n, and
        // (c) hit roughly the requested budget.
        let n = 200usize;
        let rows: Vec<Vec<f64>> = (0..n)
            .map(|i| vec![if i % 10 == 0 { 3.0 } else { 1.0 }])
            .collect();
        let u = factors_from_rows(&rows, 1, 1);
        let metric = RowMetric::output_fisher(u, 1, 1).expect("of");
        let measure = RowSamplingMeasure::from_metric(&metric);

        let budget = 40usize;
        let a = measure.designed_subsample(budget, 17);
        let b = measure.designed_subsample(budget, 17);
        assert_eq!(a.rows, b.rows, "same seed must give the identical design");
        assert_eq!(a.likelihood_weights, b.likelihood_weights);

        // Madow realizes ⌊Σπ⌋ or ⌈Σπ⌉ rows; Σπ is the budget by construction.
        assert!((a.expected_size - budget as f64).abs() < 1e-9);
        assert!(a.len() == budget || a.len() == budget + 1 || a.len() + 1 == budget);

        // Horvitz–Thompson corpus-size identity: Σ 1/π over a systematic
        // sample concentrates near n (exact in expectation; systematic
        // sampling keeps it within a small relative band here).
        let est = a.estimated_corpus_rows();
        assert!(
            (est - n as f64).abs() < 0.25 * n as f64,
            "HT corpus estimate {est} too far from n = {n}"
        );

        // Rows ascend and weights are finite and ≥ 1 (π ≤ 1).
        assert!(a.rows.windows(2).all(|w| w[0] < w[1]));
        assert!(
            a.likelihood_weights
                .iter()
                .all(|&w| w.is_finite() && w >= 1.0 - 1e-12)
        );
    }

    #[test]
    fn designed_subsample_full_budget_is_the_exact_pass() {
        let measure = RowSamplingMeasure::uniform(7);
        let s = measure.designed_subsample(7, 3);
        assert_eq!(s.rows, (0..7).collect::<Vec<_>>());
        assert!(s.likelihood_weights.iter().all(|&w| w == 1.0));
        let s = measure.designed_subsample(100, 3);
        assert_eq!(s.rows.len(), 7);
    }

    #[test]
    fn designed_subsample_uniform_measure_gives_flat_weights() {
        // Under the uniform fallback every π is budget/n, so every selected
        // row carries the same n/budget weight — plain HT scale-up.
        let n = 120usize;
        let budget = 30usize;
        let measure = RowSamplingMeasure::uniform(n);
        let s = measure.designed_subsample(budget, 5);
        assert_eq!(s.provenance, MeasureProvenance::Uniform);
        let expect = n as f64 / budget as f64;
        for &w in &s.likelihood_weights {
            assert!(
                (w - expect).abs() < 1e-9,
                "uniform design weight {w} != {expect}"
            );
        }
        assert_eq!(s.len(), budget);
    }

    #[test]
    fn designed_subsample_oversamples_loud_rows_with_downweighted_loss() {
        // A loud row should be (nearly) always included — but with a SMALLER
        // likelihood weight (its π is larger), so inclusion does not bias the
        // criterion toward loud rows.
        let rows: Vec<Vec<f64>> = (0..50)
            .map(|i| vec![if i == 7 { 30.0 } else { 1.0 }])
            .collect();
        let u = factors_from_rows(&rows, 1, 1);
        let metric = RowMetric::output_fisher(u, 1, 1).expect("of");
        let measure = RowSamplingMeasure::from_metric(&metric);
        let s = measure.designed_subsample(10, 99);
        let pos = s.rows.iter().position(|&r| r == 7);
        assert!(pos.is_some(), "the dominant-mass row must be in the design");
        let w7 = s.likelihood_weights[pos.unwrap()];
        let w_other = s
            .likelihood_weights
            .iter()
            .enumerate()
            .filter(|&(k, _)| s.rows[k] != 7)
            .map(|(_, &w)| w)
            .next()
            .expect("some quiet row selected");
        assert!(
            w7 < w_other,
            "loud row weight {w7} must be below quiet row weight {w_other}"
        );
    }

    fn coreset_dense_oracle(rows: &[Array2<f64>], coreset: &SpectralCoreset) -> Array2<f64> {
        let dim = rows[0].ncols();
        let mut approx = Array2::<f64>::zeros((dim, dim));
        for (&row, &weight) in coreset.indices.iter().zip(coreset.weights.iter()) {
            add_factor_gram_scaled(&mut approx, &rows[row], weight);
        }
        approx
    }

    fn generalized_effective_spectrum(full: &Array2<f64>, approx: &Array2<f64>) -> Vec<f64> {
        let (evals, evecs) = full.eigh(Side::Lower).expect("oracle eigh");
        let max_eval = evals.iter().copied().fold(0.0_f64, f64::max);
        let tol = (full.ncols().max(1) as f64) * f64::EPSILON * max_eval.max(1.0) * 100.0;
        let kept: Vec<usize> = evals
            .iter()
            .enumerate()
            .filter_map(|(idx, &lambda)| (lambda > tol).then_some(idx))
            .collect();
        let mut whitener = Array2::<f64>::zeros((full.ncols(), kept.len()));
        for (col, &eig_idx) in kept.iter().enumerate() {
            let scale = 1.0 / evals[eig_idx].sqrt();
            for row in 0..full.ncols() {
                whitener[[row, col]] = evecs[[row, eig_idx]] * scale;
            }
        }
        let reduced = whitener.t().dot(approx).dot(&whitener);
        let (spectrum, _) = reduced.eigh(Side::Lower).expect("reduced oracle eigh");
        spectrum.to_vec()
    }

    #[test]
    fn bss_planted_low_rank_rows_match_dense_oracle_spectrum() {
        let rows = vec![
            array![[1.0, 0.0, 0.0, 0.0]],
            array![[0.0, 2.0, 0.0, 0.0]],
            array![[1.0, 1.0, 0.0, 0.0]],
            array![[2.0, -1.0, 0.0, 0.0]],
            array![[0.5, 1.5, 0.0, 0.0]],
            array![[1.25, -0.25, 0.0, 0.0]],
        ];
        let eps = 0.35;
        let coreset = bss_spectral_coreset_certified(rows.iter().map(|row| row.view()), eps)
            .expect("BSS coreset");
        let full = summed_factor_gram(&rows, rows[0].ncols());
        let approx = coreset_dense_oracle(&rows, &coreset);
        let spectrum = generalized_effective_spectrum(&full, &approx);

        assert_eq!(coreset.certificate.dim_effective, 2);
        assert_eq!(spectrum.len(), 2);
        for lambda in spectrum {
            assert!(
                lambda >= 1.0 - eps - 1e-8 && lambda <= 1.0 + eps + 1e-8,
                "coreset generalized eigenvalue {lambda} outside [{}, {}]",
                1.0 - eps,
                1.0 + eps
            );
        }
    }

    #[test]
    fn bss_selects_single_row_carrying_unique_direction() {
        let rows = vec![
            array![[3.0, 0.0]],
            array![[2.0, 0.0]],
            array![[1.0, 0.0]],
            array![[0.0, 4.0]],
        ];
        let coreset = bss_spectral_coreset_certified(rows.iter().map(|row| row.view()), 0.4)
            .expect("BSS coreset");
        assert!(
            coreset.indices.contains(&3),
            "the only row carrying direction e2 must be selected: {:?}",
            coreset.indices
        );
    }

    #[test]
    fn bss_selection_is_deterministic() {
        let rows = vec![
            array![[1.0, 0.0, 0.0]],
            array![[0.0, 1.0, 0.0]],
            array![[0.0, 0.0, 1.0]],
            array![[1.0, 1.0, 0.0]],
            array![[0.0, 1.0, 1.0]],
        ];
        let a = bss_spectral_coreset_certified(rows.iter().map(|row| row.view()), 0.45)
            .expect("first BSS coreset");
        let b = bss_spectral_coreset_certified(rows.iter().map(|row| row.view()), 0.45)
            .expect("second BSS coreset");
        assert_eq!(a.indices, b.indices);
        assert_eq!(a.weights, b.weights);
        assert_eq!(a.certificate, b.certificate);
    }

    #[test]
    fn certificate_reports_insufficient_margin_explicitly() {
        let certificate = CoresetCertificate::new(0.1, 0.25, 3, 5).expect("certificate");
        let required = certificate.race_transfer_margin();
        assert!(matches!(
            certificate.certify_margin(required),
            CoresetMarginVerdict::InsufficientMargin { .. }
        ));
        assert!(matches!(
            certificate.certify_margin(required + 1.0),
            CoresetMarginVerdict::Certified { .. }
        ));
    }

    #[test]
    fn sensitivity_bounds_and_greedy_budget_are_deterministic() {
        let leverage = vec![0.2, 0.5, 0.5, 0.1];
        let sigma = sensitivity_upper_bounds(&leverage, 2.0, 0.25).expect("sigma");
        let expected = [0.3, 0.75, 0.75, 0.15];
        for (got, want) in sigma.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-12);
        }
        let selected = greedy_sensitivity_coreset(&sigma, 2).expect("greedy");
        assert_eq!(selected.indices, vec![1, 2]);
        assert!((selected.selected_sensitivity_mass - 1.5).abs() < 1e-12);
        assert!((selected.residual_sensitivity_mass - 0.45).abs() < 1e-12);
    }

    /// #1012 certified designed subsample: the result carries a certificate
    /// whose race-transfer margin equals the certificate's, and the adversarial
    /// heavy-tail row (one row carrying the curvature signal) is FORCED into the
    /// coreset by the sensitivity bound — the classic uniform-subsampling miss.
    #[test]
    fn certified_subsample_forces_the_heavy_tail_row_and_carries_a_certificate() {
        // Five rows: four ordinary low-leverage rows and one heavy-tail row
        // (index 4) with an order-of-magnitude larger leverage and a unique
        // spectral direction e2.
        let row_factors = vec![
            array![[1.0, 0.0]],
            array![[1.0, 0.0]],
            array![[1.0, 0.0]],
            array![[1.0, 0.0]],
            array![[0.0, 5.0]],
        ];
        let leverage = vec![0.05, 0.05, 0.05, 0.05, 0.9];
        let measure = RowSamplingMeasure::uniform(5);
        let certified = measure
            .designed_subsample_certified(
                row_factors.iter().map(|r| r.view()),
                0.4,
                &leverage,
                1.0,
                0.1,
                1, // budget admits a single sensitivity row
            )
            .expect("certified subsample");

        assert!(
            certified.rows.contains(&4),
            "the heavy-tail row carrying the curvature signal must be forced in: {:?}",
            certified.rows
        );
        assert_eq!(certified.rows.len(), certified.weights.len());
        // The race-transfer margin is the certificate's — the shared #1011/#1012
        // seam a race consumer gates on.
        assert!(
            (certified.race_transfer_margin() - certified.certificate.race_transfer_margin()).abs()
                < 1e-12
        );
        assert!(certified.certificate.race_transfer_margin() > 0.0);
        // The certificate's selected count matches the realized coreset.
        assert_eq!(certified.certificate.n_selected, certified.rows.len());
    }
}
