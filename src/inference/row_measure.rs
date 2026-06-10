//! `RowMeasure` — the Fisher-mass **enrichment** producer (role (c) of #980).
//!
//! # What this is, and what it must never be
//!
//! A [`RowMeasure`] turns a [`RowMetric`] into a per-row **sampling measure**:
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

use crate::inference::row_metric::{MetricProvenance, RowMetric};
use crate::linalg::utils::splitmix64_hash;

/// Where a [`RowMeasure`] came from — the honest record of whether the
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
/// Built from a [`RowMetric`] via [`RowMeasure::from_metric`]. The weights are a
/// proper probability measure (non-negative, finite, summing to 1) used for
/// **discovery/seeding oversampling only** — see the module docs for the
/// invariant that it touches no loss / gradient / criterion.
#[derive(Clone, Debug)]
pub struct RowMeasure {
    provenance: MeasureProvenance,
    /// Normalized per-row sampling weights; `weights.len() == n_rows` and
    /// `Σ weights == 1` (exactly uniform `1/n` in the fallback).
    weights: Vec<f64>,
}

impl RowMeasure {
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
    fn from_masses(metric_provenance: MetricProvenance, masses: Vec<f64>) -> Self {
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
}

/// Salt mixed into the enrichment seed so the offset hash is distinct from any
/// other `splitmix64_hash` use of the same numeric seed elsewhere in the crate.
const ENRICHMENT_SALT: u64 = 0x980E_1C45_F00D_AC70;

/// Per-row Fisher mass `tr(M_n)` from the metric's validated PSD blocks.
///
/// `tr(M_n) = Σ_i M_n[i, i]`. This reads the criterion-facing (un-floored)
/// blocks, so the solver `δ` never enters the measure — consistent with the
/// `RowMetric` #747 discipline, and irrelevant anyway because the measure feeds
/// no criterion. Pure read; touches nothing.
fn per_row_fisher_mass(metric: &RowMetric) -> Vec<f64> {
    let blocks = metric.blocks();
    let n = metric.n_rows();
    let p = metric.p_out();
    let mut mass = vec![0.0_f64; n];
    for row in 0..n {
        let mut tr = 0.0_f64;
        for i in 0..p {
            tr += blocks[[row, i, i]];
        }
        mass[row] = tr;
    }
    mass
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use std::sync::Arc;

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
        let measure = RowMeasure::from_metric(&metric);
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
        let measure = RowMeasure::from_metric(&metric);
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
        let measure = RowMeasure::from_metric(&metric);
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
        let measure = RowMeasure::from_metric(&metric);
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
        let measure = RowMeasure::from_metric(&metric);
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
        let measure = RowMeasure::from_metric(&metric);
        let rep = measure.expected_representation(10);
        // masses 1, 9 ⇒ weights 0.1, 0.9 ⇒ reps 1.0, 9.0.
        assert!((rep[0] - 1.0).abs() < 1e-12);
        assert!((rep[1] - 9.0).abs() < 1e-12);
    }
}
