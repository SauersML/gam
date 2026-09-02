//! Residual-energy-**stratified** birth screen — making the dictionary tail
//! reachable at frontier scale (#973 follow-on).
//!
//! # The reachability problem this closes
//!
//! Discovery of a new atom is driven by what the birth producer (the ISA seed /
//! harvest emitter feeding the stagewise births) is *shown*. At frontier scale
//! the producer never sees the whole corpus; it sees a **designed subsample**
//! ([`super::designed_target`]). That subsample is drawn either uniformly (cold
//! start) or importance-weighted by a harvested Fisher measure
//! ([`gam_solve::row_sampling_measure::RowSamplingMeasure`]). Both are *proportional* designs:
//! a structure that is active on a fraction `f_rare` of tokens is presented to
//! the producer on ≈ `f_rare · budget` rows. A `10⁻⁵`-frequency structure among
//! `10⁸` tokens has ≈ `10³` active rows; at a `2·10⁶` budget it contributes
//! ≈ `0.02` expected rows to the sample — i.e. it is *essentially never seen*,
//! so it can never be proposed, so the dictionary tail is unreachable no matter
//! how long discovery runs.
//!
//! The fix is not to reweight the loss (that would bias the fit — the #980
//! failure mode). It is to **stratify the design by residual energy** so the
//! rare-but-high-residual rows are guaranteed representation in what the producer
//! sees, while every selected row still carries its exact Horvitz–Thompson
//! inclusion weight `1/π` so the criterion the accept decision runs on stays
//! *unbiased*. Stratification changes **which structures get proposed**, never
//! **whether a proposed atom is accepted** (that stays a REML/evidence decision
//! on the HT-weighted criterion).
//!
//! # The estimator (why this is unbiased for any allocation)
//!
//! Partition the `N` corpus rows into strata `H_1..H_K` by residual energy
//! `e_i = ‖x_i − P x_i‖²` (the energy the *current* dictionary cannot explain;
//! `P` projects onto its column span). Stratum `h` has population `N_h`,
//! `Σ_h N_h = N`. Sample stratum `h` at rate `π_h ∈ (0,1]`: row `i ∈ H_h` is
//! included independently with probability `π_h` (realized deterministically by
//! hashing its stable `row_id`, so no clock randomness), and if included carries
//! weight `w_i = 1/π_h`.
//!
//! For **any** per-row statistic `ℓ_i` (a residual, a log-likelihood term, a
//! Fisher block — anything the fit sums over) the stratified expansion estimator
//! is unbiased for the full-corpus sum:
//!
//! ```text
//!   E[ Σ_{i∈S} w_i ℓ_i ]
//!     = Σ_h Σ_{i∈H_h} E[𝟙{i∈S}] · (1/π_h) · ℓ_i
//!     = Σ_h Σ_{i∈H_h} π_h · (1/π_h) · ℓ_i
//!     = Σ_i ℓ_i.                                    (Horvitz–Thompson)
//! ```
//!
//! The allocation `{π_h}` affects only the **variance**
//! `Var = Σ_h Σ_{i∈H_h} ℓ_i² (1−π_h)/π_h`, never the expectation. So REML/LAML,
//! the evidence criterion, `φ̂`, and the ρ gradient are all unbiased on this
//! stream regardless of how we allocate — the accept decision is untouched. The
//! ρ cascade ([`super::rho_cascade`]) can therefore keep its own
//! importance-weighted uniform stream unchanged; this stratified stream is the
//! *discovery* target, and its HT weights keep it a valid unbiased design in its
//! own right.
//!
//! # The allocation (every boundary derived, no magic constants — SPEC.md §19)
//!
//! * **Strata boundaries** are the IEEE-754 binary exponents of `e_i`: stratum
//!   membership is `⌊log₂ e_i⌋`, a factor-of-two energy band. This is derived
//!   from the data's own representation (no chosen cut points) and isolates the
//!   tail *by magnitude*: the rare high-residual rows land in high-exponent bands
//!   with tiny `N_h`, distinct from the dominant low-residual bulk. (Equal-
//!   population quantile strata would bury a `10⁻⁵` tail inside the top quantile;
//!   energy-magnitude bands do not.)
//! * **Stratum count** is capped at Sturges' rule `K_max = ⌊log₂ N⌋ + 1` — the
//!   standard derived bin count for `N` observations. If more energy bands are
//!   occupied than `K_max`, adjacent **low-energy** bands are merged (tail
//!   resolution, where discovery lives, is preserved).
//! * **Census of the tail** (the discovery guarantee): with an equal share
//!   `s = budget / K`, every stratum whose whole population fits its share
//!   (`N_h ≤ s`) is taken *in full* (`π_h = 1`, weight `1`, zero design
//!   variance). Iterated to a fixed point (water-filling), this censuses the
//!   rare high-energy strata, so the producer is shown *all* `N_h` of their rows
//!   instead of `f · N_h`. Equal-share census is a named standard scheme.
//! * **Neyman allocation** of the leftover budget over the big strata:
//!   `π_h = B' · S_h / Σ_g N_g S_g` with `S_h` the within-stratum energy
//!   standard deviation — the variance-optimal allocation for resolving residual
//!   energy. When there is no residual variation (`S_h ≡ 0`, e.g. a cold uniform
//!   corpus) this degenerates to proportional `π_h = B'/N'`, recovering the plain
//!   uniform design bit-for-bit.
//! * **Uniform-rate floor**: every non-empty stratum gets `π_h ≥ f = budget/N`,
//!   so stratification only ever *adds* attention to the tail, never samples any
//!   band below the uniform baseline.
//!
//! The census + Neyman + floor are solved by the same peel-and-refill
//! water-filling the importance design uses, so nothing here is a tuned knob.

use ndarray::{Array2, ArrayView1};

/// A cheap, no-inner-solve per-row residual energy under the current dictionary.
///
/// The value must be a finite, non-negative scalar: the energy of row `x` that
/// the current dictionary leaves unexplained. The canonical implementation is
/// the projection residual `‖x‖² − ‖Qᵀx‖²` onto an orthonormal basis `Q` of the
/// dictionary's column span ([`SpanResidualEnergy`]); a cold start with no
/// dictionary uses the raw row energy `‖x‖²`. Any non-finite or negative value
/// is treated as zero by the screen (degrades to the low-energy bulk), never an
/// error.
pub trait RowResidualEnergy {
    /// Non-negative residual energy of one activation row.
    fn energy(&self, row: ArrayView1<f64>) -> f64;
}

/// Projection-residual energy `‖x‖² − ‖Qᵀx‖²` onto an orthonormal span `Q`
/// (`p × r`, columns orthonormal). This is the energy the dictionary whose
/// column span is `Q` cannot represent — cheap (one `gemv`, no inner solve).
#[derive(Debug, Clone)]
pub struct SpanResidualEnergy {
    /// `p × r` orthonormal basis of the current dictionary's column span.
    basis: Array2<f64>,
}

impl SpanResidualEnergy {

    /// The activation width `p` this screen expects.
    pub fn width(&self) -> usize {
        self.basis.nrows()
    }
}

impl RowResidualEnergy for SpanResidualEnergy {
    #[inline]
    fn energy(&self, row: ArrayView1<f64>) -> f64 {
        let full: f64 = row.iter().map(|&v| v * v).sum();
        if self.basis.ncols() == 0 {
            return full;
        }
        // ‖Qᵀx‖² = Σ_j (q_jᵀx)²; the explained energy in the span.
        let mut explained = 0.0_f64;
        for col in self.basis.columns() {
            let proj: f64 = col.iter().zip(row.iter()).map(|(&q, &x)| q * x).sum();
            explained += proj * proj;
        }
        (full - explained).max(0.0)
    }
}

/// One stratum of the residual-energy design: a contiguous range of exponent
/// bins, its population and within-stratum energy statistics, and the sampling
/// rate `π_h` the allocation assigned it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stratum {
    /// Inclusive lowest exponent bin in this stratum.
    pub exp_lo: usize,
    /// Inclusive highest exponent bin in this stratum.
    pub exp_hi: usize,
    /// Population `N_h`.
    pub n_rows: u64,
    /// Within-stratum mean residual energy.
    pub mean_energy: f64,
    /// Within-stratum residual-energy standard deviation `S_h`.
    pub std_energy: f64,
    /// Sampling rate `π_h ∈ (0, 1]` assigned by the allocation.
    pub pi: f64,
    /// Whether this stratum was censused (`π_h = 1`, taken in full).
    pub censused: bool,
}

/// The full stratified design: the strata (ascending energy) with their assigned
/// sampling rates, plus the exponent→stratum lookup used at collection time.
#[derive(Debug, Clone)]
pub struct StratumDesign {
    strata: Vec<Stratum>,
    /// `bin_to_stratum[exp]` is the stratum index owning exponent bin `exp`
    /// (`usize::MAX` for an empty bin — no row maps there).
    total_rows: u64,
    budget: usize,
}

impl StratumDesign {
    /// The strata, ascending in residual energy.
    pub fn strata(&self) -> &[Stratum] {
        &self.strata
    }

    pub fn total_rows(&self) -> u64 {
        self.total_rows
    }

    pub fn budget(&self) -> usize {
        self.budget
    }

}

/// Sturges' rule stratum cap `K_max = ⌊log₂ N⌋ + 1` — the standard derived bin
/// count for `N` observations. `N ≤ 1` degenerates to a single stratum.
fn sturges_stratum_cap(total_rows: u64) -> usize {
    if total_rows <= 1 {
        return 1;
    }
    // `⌊log₂ N⌋ = 63 − leading_zeros(N)`; +1 for Sturges.
    (u64::BITS - total_rows.leading_zeros()) as usize
}

/// One in-memory residual-energy stratum: the row indices that fall in a
/// factor-of-two energy band, with the band's population moments. The energy
/// bands and the Sturges cap are exactly [`design_stratified_subsample`]'s
/// (`⌊log₂ e_i⌋` bins, `K_max = ⌊log₂ N⌋ + 1`, adjacent low-energy bands merged),
/// so an in-core caller (the stagewise birth loop) stratifies its residual by the
/// SAME derived boundaries as the streaming corpus screen — no new cut points.
#[derive(Clone, Debug, PartialEq)]
pub struct RowStratum {
    /// Inclusive lowest exponent bin in this stratum.
    pub exp_lo: usize,
    /// Inclusive highest exponent bin in this stratum.
    pub exp_hi: usize,
    /// Row indices assigned to this stratum (ascending).
    pub rows: Vec<usize>,
    /// Within-stratum mean residual energy.
    pub mean_energy: f64,
    /// Within-stratum residual-energy standard deviation `S_h`.
    pub std_energy: f64,
}

/// Biased base-2 exponent bin for a non-negative energy. Zero / subnormal /
/// non-finite / negative energies fall in bin `0` (the low-energy floor).
#[inline]
fn energy_exponent_bin(energy: f64) -> usize {
    if !energy.is_finite() || energy <= 0.0 {
        return 0;
    }
    ((energy.to_bits() >> 52) & 0x7ff) as usize
}

/// Stratify a set of per-row residual energies into factor-of-two energy bands,
/// returning the row-index groups ASCENDING in energy. Reuses the same IEEE-754
/// binary-exponent bins (`energy_exponent_bin`) and Sturges cap
/// (`sturges_stratum_cap`) as the streaming design; adjacent lowest-energy bands
/// are merged to the cap so the high-energy tail (where rare discoverable structure
/// concentrates) keeps its resolution. Empty / non-finite / negative energies fall
/// in the low-energy floor bin. An empty input yields no strata.
///
/// This is the in-core companion to [`design_stratified_subsample`]: where the
/// streaming screen samples the tail for the corpus producer, this exposes the same
/// tail-preserving partition to an already-materialized residual so a consumer can
/// process each stratum's rows locally (the stagewise stratum-local birth screen).
pub fn stratify_row_energies(energies: &[f64]) -> Vec<RowStratum> {
    let n = energies.len();
    if n == 0 {
        return Vec::new();
    }
    // Group rows by exponent bin (ascending energy).
    let mut bin_rows: std::collections::BTreeMap<usize, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (i, &e) in energies.iter().enumerate() {
        bin_rows
            .entry(energy_exponent_bin(e))
            .or_default()
            .push(i);
    }
    // One stratum per occupied bin, with population moments over the (clamped)
    // energies (negatives / non-finite → 0, matching the histogram floor bin).
    let clamp = |e: f64| if e.is_finite() && e > 0.0 { e } else { 0.0 };
    let mut strata: Vec<RowStratum> = bin_rows
        .into_iter()
        .map(|(exp, rows)| {
            let m = rows.len() as f64;
            let sum: f64 = rows.iter().map(|&i| clamp(energies[i])).sum();
            let mean = sum / m;
            let var = (rows
                .iter()
                .map(|&i| clamp(energies[i]).powi(2))
                .sum::<f64>()
                / m
                - mean * mean)
                .max(0.0);
            RowStratum {
                exp_lo: exp,
                exp_hi: exp,
                rows,
                mean_energy: mean,
                std_energy: var.sqrt(),
            }
        })
        .collect();

    let k_max = sturges_stratum_cap(n as u64);
    // Merge adjacent lowest-energy strata until within the cap (tail preserved).
    while strata.len() > k_max && strata.len() >= 2 {
        let mut merged = std::mem::take(&mut strata[0]);
        let hi = std::mem::take(&mut strata[1]);
        let na = merged.rows.len() as f64;
        let nb = hi.rows.len() as f64;
        let nt = na + nb;
        let mean = (merged.mean_energy * na + hi.mean_energy * nb) / nt;
        let sumsq_a = (merged.std_energy.powi(2) + merged.mean_energy.powi(2)) * na;
        let sumsq_b = (hi.std_energy.powi(2) + hi.mean_energy.powi(2)) * nb;
        let var = ((sumsq_a + sumsq_b) / nt - mean * mean).max(0.0);
        merged.rows.extend(hi.rows);
        merged.rows.sort_unstable();
        merged.exp_hi = hi.exp_hi.max(merged.exp_hi);
        merged.mean_energy = mean;
        merged.std_energy = var.sqrt();
        strata[1] = merged;
        strata.remove(0);
    }
    strata
}

impl Default for RowStratum {
    fn default() -> Self {
        RowStratum {
            exp_lo: 0,
            exp_hi: 0,
            rows: Vec::new(),
            mean_energy: 0.0,
            std_energy: 0.0,
        }
    }
}

/// The collected stratified row set: the dense fit target the birth producer
/// runs on, plus the honesty weights that keep every criterion unbiased.
#[derive(Debug, Clone)]
pub struct StratifiedCorpusTarget {
    /// `(n_selected × p)` upcast activations of the selected rows, ascending
    /// global row order.
    pub target: Array2<f64>,
    /// Global corpus `row_id` of each target row (ascending).
    pub row_ids: Vec<u64>,
    /// Per-row Horvitz–Thompson weight `1/π_h`, aligned with `target`. Hand to
    /// `SaeManifoldTerm::set_row_loss_weights`; an all-censused design yields
    /// all-`1.0` weights and the exact unweighted path.
    pub likelihood_weights: Vec<f64>,
    /// The design the collection realized (strata + rates), for diagnostics /
    /// the ISA post-fit audit.
    pub design: StratumDesign,
    /// Total corpus rows the design was drawn from.
    pub corpus_rows: u64,
}

impl StratifiedCorpusTarget {
    pub fn len(&self) -> usize {
        self.row_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.row_ids.is_empty()
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sturges_cap_matches_floor_log2_plus_one() {
        assert_eq!(sturges_stratum_cap(1), 1);
        assert_eq!(sturges_stratum_cap(2), 2);
        assert_eq!(sturges_stratum_cap(255), 8);
        assert_eq!(sturges_stratum_cap(256), 9);
        assert_eq!(sturges_stratum_cap(100_000_000), 27);
    }
}
