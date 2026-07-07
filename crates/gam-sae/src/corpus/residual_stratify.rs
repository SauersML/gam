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

use gam_linalg::utils::splitmix64_hash;

use super::shard_reader::CorpusRowSource;

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
    /// Build from a `p × r` matrix whose columns are (assumed) orthonormal.
    /// An empty basis (`r == 0`) yields the raw-energy `‖x‖²` cold-start screen.
    pub fn new(basis: Array2<f64>) -> Self {
        Self { basis }
    }

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

/// Number of IEEE-754 binary-exponent bins (biased exponent is 11 bits).
const N_EXPONENT_BINS: usize = 1 << 11;

/// Bounded streaming summary of the residual-energy distribution: one accumulator
/// per IEEE-754 biased exponent. `O(2048)` memory regardless of `N`, so the
/// screen never risks OOM (SPEC.md §9). Populated in a single streaming pass.
#[derive(Debug, Clone)]
struct EnergyExponentHistogram {
    count: Vec<u64>,
    sum: Vec<f64>,
    sumsq: Vec<f64>,
    total_rows: u64,
}

impl EnergyExponentHistogram {
    fn new() -> Self {
        Self {
            count: vec![0; N_EXPONENT_BINS],
            sum: vec![0.0; N_EXPONENT_BINS],
            sumsq: vec![0.0; N_EXPONENT_BINS],
            total_rows: 0,
        }
    }

    /// Biased base-2 exponent bin for a non-negative energy. Zero / subnormal /
    /// non-finite / negative energies fall in bin `0` (the low-energy floor).
    #[inline]
    fn bin_of(energy: f64) -> usize {
        if !energy.is_finite() || energy <= 0.0 {
            return 0;
        }
        ((energy.to_bits() >> 52) & 0x7ff) as usize
    }

    #[inline]
    fn observe(&mut self, energy: f64) {
        let e = if energy.is_finite() && energy > 0.0 {
            energy
        } else {
            0.0
        };
        let bin = Self::bin_of(e);
        self.count[bin] += 1;
        self.sum[bin] += e;
        self.sumsq[bin] += e * e;
        self.total_rows += 1;
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
    bin_to_stratum: Vec<usize>,
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

    /// The uniform baseline sampling rate `f = budget / N` — the floor every
    /// stratum's rate clears.
    pub fn uniform_rate(&self) -> f64 {
        if self.total_rows == 0 {
            0.0
        } else {
            (self.budget as f64 / self.total_rows as f64).min(1.0)
        }
    }

    /// Sampling rate `π_h` for a row of the given residual energy, and whether
    /// its stratum was censused. An energy outside every occupied band (should
    /// not happen for a row seen in pass A) falls back to the uniform rate.
    fn rate_for_energy(&self, energy: f64) -> f64 {
        let bin = EnergyExponentHistogram::bin_of(energy);
        let s = self.bin_to_stratum[bin];
        if s == usize::MAX {
            self.uniform_rate()
        } else {
            self.strata[s].pi
        }
    }

    /// Expected number of sampled rows across all strata, `Σ_h N_h π_h`.
    pub fn expected_sample_size(&self) -> f64 {
        self.strata
            .iter()
            .map(|s| s.n_rows as f64 * s.pi)
            .sum()
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

/// Build the strata from the exponent histogram: one stratum per occupied
/// exponent band, then merge adjacent **low-energy** bands until at most
/// `K_max` strata remain (preserving high-energy tail resolution).
fn build_strata(hist: &EnergyExponentHistogram) -> (Vec<Stratum>, Vec<usize>) {
    // Occupied exponent bins, ascending.
    let mut strata: Vec<Stratum> = Vec::new();
    for (exp, &c) in hist.count.iter().enumerate() {
        if c == 0 {
            continue;
        }
        let n = c as f64;
        let mean = hist.sum[exp] / n;
        // Population variance; clamp round-off negatives (energies span decades).
        let var = (hist.sumsq[exp] / n - mean * mean).max(0.0);
        strata.push(Stratum {
            exp_lo: exp,
            exp_hi: exp,
            n_rows: c,
            mean_energy: mean,
            std_energy: var.sqrt(),
            pi: 0.0,
            censused: false,
        });
    }

    let k_max = sturges_stratum_cap(hist.total_rows);
    // Merge adjacent lowest-energy strata until within the cap. Tail (high
    // energy, discovery-relevant) resolution is preserved by only ever merging
    // the two lowest.
    while strata.len() > k_max && strata.len() >= 2 {
        let merged = merge_stratum(&strata[0], &strata[1]);
        strata[1] = merged;
        strata.remove(0);
    }

    // Exponent → stratum lookup.
    let mut bin_to_stratum = vec![usize::MAX; N_EXPONENT_BINS];
    for (idx, s) in strata.iter().enumerate() {
        for exp in s.exp_lo..=s.exp_hi {
            bin_to_stratum[exp] = idx;
        }
    }
    (strata, bin_to_stratum)
}

/// Combine two adjacent strata, recovering the pooled mean/variance from their
/// counts and per-stratum moments (no raw data retained).
fn merge_stratum(a: &Stratum, b: &Stratum) -> Stratum {
    let na = a.n_rows as f64;
    let nb = b.n_rows as f64;
    let n = na + nb;
    // Pooled first moment.
    let sum = a.mean_energy * na + b.mean_energy * nb;
    let mean = if n > 0.0 { sum / n } else { 0.0 };
    // Pooled second moment: Σx² = Σ(var + mean²)·n per part.
    let sumsq_a = (a.std_energy * a.std_energy + a.mean_energy * a.mean_energy) * na;
    let sumsq_b = (b.std_energy * b.std_energy + b.mean_energy * b.mean_energy) * nb;
    let var = if n > 0.0 {
        ((sumsq_a + sumsq_b) / n - mean * mean).max(0.0)
    } else {
        0.0
    };
    Stratum {
        exp_lo: a.exp_lo.min(b.exp_lo),
        exp_hi: a.exp_hi.max(b.exp_hi),
        n_rows: a.n_rows + b.n_rows,
        mean_energy: mean,
        std_energy: var.sqrt(),
        pi: 0.0,
        censused: false,
    }
}

/// Solve the census + Neyman + floor allocation in place, writing `pi` /
/// `censused` on each stratum. See the module docs for the derivation.
fn allocate_rates(strata: &mut [Stratum], total_rows: u64, budget: usize) {
    let k = strata.len();
    if k == 0 || total_rows == 0 {
        return;
    }
    let n_total = total_rows as f64;
    let uniform_rate = (budget as f64 / n_total).min(1.0);

    if budget as u64 >= total_rows {
        // Full pass: everything censused at π = 1 (exact, bit-for-bit).
        for s in strata.iter_mut() {
            s.pi = 1.0;
            s.censused = true;
        }
        return;
    }

    // --- Census water-filling: peel off strata that fit their equal share. ---
    for s in strata.iter_mut() {
        s.censused = false;
    }
    loop {
        let remaining: Vec<usize> = (0..k).filter(|&i| !strata[i].censused).collect();
        if remaining.is_empty() {
            break;
        }
        let censused_pop: u64 = strata
            .iter()
            .filter(|s| s.censused)
            .map(|s| s.n_rows)
            .sum();
        let budget_left = budget as f64 - censused_pop as f64;
        if budget_left <= 0.0 {
            break;
        }
        let share = budget_left / remaining.len() as f64;
        // Census any not-yet-censused stratum whose whole population fits its
        // equal share. Take the smallest first for a deterministic fixed point.
        let mut newly = false;
        for &i in &remaining {
            if (strata[i].n_rows as f64) <= share {
                strata[i].censused = true;
                newly = true;
            }
        }
        if !newly {
            break;
        }
    }

    // --- Neyman allocation of the leftover budget over the big strata. ---
    let censused_pop: u64 = strata
        .iter()
        .filter(|s| s.censused)
        .map(|s| s.n_rows)
        .sum();
    let budget_left = (budget as f64 - censused_pop as f64).max(0.0);
    let neyman_mass: f64 = strata
        .iter()
        .filter(|s| !s.censused)
        .map(|s| s.n_rows as f64 * s.std_energy)
        .sum();
    let big_pop: f64 = strata
        .iter()
        .filter(|s| !s.censused)
        .map(|s| s.n_rows as f64)
        .sum();

    for s in strata.iter_mut() {
        if s.censused {
            s.pi = 1.0;
            continue;
        }
        let n_h = s.n_rows as f64;
        let target = if neyman_mass > 0.0 {
            // Neyman: n_h = B' · (N_h S_h) / Σ N_g S_g ⇒ π_h = n_h / N_h.
            budget_left * (n_h * s.std_energy / neyman_mass) / n_h
        } else if big_pop > 0.0 {
            // No residual variation ⇒ proportional (uniform rate on the big
            // strata), recovering the plain uniform design.
            budget_left / big_pop
        } else {
            uniform_rate
        };
        // Floor at the uniform rate; cap at 1.
        s.pi = target.max(uniform_rate).min(1.0);
    }
}

/// Design a residual-energy-stratified subsample from the streamed energies.
///
/// One streaming pass over `source` under `energy` builds the bounded exponent
/// histogram; the strata and their sampling rates are then solved. This is the
/// *design* step — no rows are materialized here. `budget ≥ N` yields an
/// all-censused (full, exact) design.
pub fn design_stratified_subsample(
    source: &mut dyn CorpusRowSource,
    energy: &dyn RowResidualEnergy,
    budget: usize,
) -> Result<StratumDesign, String> {
    let total_rows = source.total_rows();
    let mut hist = EnergyExponentHistogram::new();

    source.reset();
    while let Some(batch) = source
        .next_batch()
        .map_err(|e| format!("design_stratified_subsample: shard read failed: {e}"))?
    {
        for k in 0..batch.rows.nrows() {
            hist.observe(energy.energy(batch.rows.row(k)));
        }
    }
    if hist.total_rows != total_rows {
        return Err(format!(
            "design_stratified_subsample: streamed {} rows but source declared {total_rows}",
            hist.total_rows
        ));
    }

    let (mut strata, bin_to_stratum) = build_strata(&hist);
    allocate_rates(&mut strata, total_rows, budget);
    Ok(StratumDesign {
        strata,
        bin_to_stratum,
        total_rows,
        budget,
    })
}

/// Salt mixing the per-row inclusion hash so it never collides with any other
/// `splitmix64_hash` use of the same `row_id` in the crate.
const STRATIFY_SALT: u64 = 0x5372_4154_1F9E_0B25;

/// Deterministic per-row inclusion at rate `pi`: hash `(row_id, seed)` and
/// include iff the hash falls in the leading `pi` fraction of the `u64` space.
/// No clock randomness; the same `(row_id, seed, pi)` always decides the same.
#[inline]
fn row_included(row_id: u64, seed: u64, pi: f64) -> bool {
    if pi >= 1.0 {
        return true;
    }
    if pi <= 0.0 {
        return false;
    }
    let h = splitmix64_hash(row_id ^ seed.wrapping_mul(STRATIFY_SALT) ^ STRATIFY_SALT);
    let threshold = (pi * (u64::MAX as f64 + 1.0)) as u64;
    h < threshold
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

    /// `Σ 1/π_i` over the selected rows — the Horvitz–Thompson estimate of the
    /// corpus row count (exactly `N` in expectation). A consumer can sanity-gate
    /// the design by checking this lands near `N`.
    pub fn estimated_corpus_rows(&self) -> f64 {
        self.likelihood_weights.iter().sum()
    }
}

/// Collect a residual-energy-stratified target from a streaming source.
///
/// Two cheap streaming passes: pass A ([`design_stratified_subsample`]) builds
/// the design; pass B recomputes each row's energy (cheap, no inner solve),
/// maps it to its stratum rate `π_h`, includes the row iff [`row_included`], and
/// materializes exactly the selected rows with their `1/π_h` weights. Memory is
/// `O(budget)` for the collected rows plus `O(2048)` for the histogram — never
/// `O(N)` (SPEC.md §9).
///
/// The result is a valid unbiased design (module-doc Horvitz–Thompson estimator):
/// hand `likelihood_weights` to the term's `√w` honesty seam and every criterion
/// — REML/LAML, evidence, `φ̂` — is unbiased for the full-corpus criterion, so the
/// accept decision is unchanged; only *which* rare structures reach the birth
/// producer changes.
pub fn collect_stratified_target(
    source: &mut dyn CorpusRowSource,
    energy: &dyn RowResidualEnergy,
    budget: usize,
    seed: u64,
) -> Result<StratifiedCorpusTarget, String> {
    let design = design_stratified_subsample(source, energy, budget)?;
    let corpus_rows = source.total_rows();
    let p = source.width();

    let mut rows_out: Vec<f64> = Vec::new();
    let mut row_ids: Vec<u64> = Vec::new();
    let mut likelihood_weights: Vec<f64> = Vec::new();

    source.reset();
    while let Some(batch) = source
        .next_batch()
        .map_err(|e| format!("collect_stratified_target: shard read failed: {e}"))?
    {
        for k in 0..batch.rows.nrows() {
            let rid = batch.row_ids[k];
            let e = energy.energy(batch.rows.row(k));
            let pi = design.rate_for_energy(e);
            if row_included(rid, seed, pi) {
                rows_out.extend(batch.rows.row(k).iter().copied());
                row_ids.push(rid);
                likelihood_weights.push(1.0 / pi);
            }
        }
    }

    let n_sel = row_ids.len();
    let target = Array2::from_shape_vec((n_sel, p), rows_out)
        .map_err(|e| format!("collect_stratified_target: target assembly failed: {e}"))?;
    Ok(StratifiedCorpusTarget {
        target,
        row_ids,
        likelihood_weights,
        design,
        corpus_rows,
    })
}

#[cfg(test)]
mod tests {
    use super::super::shard_reader::{MmapShardSource, encode_shard_bytes};
    use super::*;
    use gam_solve::row_sampling_measure::RowSamplingMeasure;
    use ndarray::{Array2, s};
    use std::io::Write;
    use std::path::PathBuf;

    /// Deterministic small pseudo-normal from a counter (no RNG dependency).
    fn gauss(counter: &mut u64) -> f64 {
        // Two hashed uniforms → Box–Muller. Deterministic in the counter.
        *counter = counter.wrapping_add(1);
        let a = splitmix64_hash(*counter ^ 0x1234_5678);
        *counter = counter.wrapping_add(1);
        let b = splitmix64_hash(*counter ^ 0x9ABC_DEF0);
        let u1 = ((a >> 11) as f64 / (1u64 << 53) as f64).max(1e-12);
        let u2 = (b >> 11) as f64 / (1u64 << 53) as f64;
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// A planted corpus: `n` rows in `p` dims. The dominant structure lives in
    /// the span of the first `k_dom` basis vectors (large amplitude). A rare
    /// `rare_rows`-row structure adds a moderate component along basis vector
    /// `k_dom` (out of the dominant span) — high residual energy under a
    /// dictionary that only spans the dominant directions. Returns the rows and
    /// the sorted indices of the planted rare rows.
    fn planted_corpus(
        n: usize,
        p: usize,
        k_dom: usize,
        rare_rows: usize,
    ) -> (Array2<f64>, Vec<usize>) {
        let mut rows = Array2::<f64>::zeros((n, p));
        let mut ctr = 1u64;
        // Scatter the rare rows at pseudo-random DISTINCT positions. A fixed
        // stride (`n / rare_rows`) aliases with the Horvitz–Thompson design's
        // Madow SYSTEMATIC selection: a uniform π = budget/n subsample picks
        // every ⌊n/budget⌋-th row, so periodic rare rows land on that grid and
        // are surfaced by the uniform baseline TOO — which collapses the
        // stratification-vs-uniform contrast this test asserts. Deterministic
        // pseudo-random placement makes the uniform baseline draw rare rows ∝
        // frequency (≈ f·rare rows), as the statistical claim requires, while
        // the residual-energy census still surfaces all of them by their energy.
        let mut rare_idx: Vec<usize> = Vec::with_capacity(rare_rows.min(n));
        let mut seen: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        let mut probe = 0u64;
        while rare_idx.len() < rare_rows.min(n) {
            let idx = (splitmix64_hash(probe ^ 0x5236_9E11_A73C_D0F5) as usize) % n;
            probe = probe.wrapping_add(1);
            if seen.insert(idx) {
                rare_idx.push(idx);
            }
        }
        rare_idx.sort_unstable();
        let rare_set: std::collections::BTreeSet<usize> = rare_idx.iter().copied().collect();

        for i in 0..n {
            // Dominant structure: large energy in the first k_dom dims.
            for d in 0..k_dom.min(p) {
                rows[[i, d]] = 4.0 * gauss(&mut ctr);
            }
            // Small isotropic noise everywhere (keeps bulk residual tiny but
            // nonzero, so the exponent histogram has a well-defined floor).
            for d in 0..p {
                rows[[i, d]] += 0.02 * gauss(&mut ctr);
            }
            // Rare structure: a clear out-of-span component on dim k_dom.
            if rare_set.contains(&i) && k_dom < p {
                rows[[i, k_dom]] += 3.0;
            }
        }
        (rows, rare_idx)
    }

    fn temp_shard_dir(name: &str, rows: &Array2<f64>, split_at: usize) -> PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "gam-residual-stratify-test-{}-{}",
            std::process::id(),
            name
        ));
        std::fs::create_dir_all(&dir).expect("create dir");
        let n = rows.nrows();
        let split = split_at.min(n);
        let parts = [
            ("a.shard", rows.slice(s![..split, ..])),
            ("b.shard", rows.slice(s![split.., ..])),
        ];
        for (key, part) in parts {
            let bytes = encode_shard_bytes(part);
            let mut f = std::fs::File::create(dir.join(key)).expect("create shard");
            f.write_all(&bytes).expect("write shard");
            f.sync_all().expect("sync");
        }
        dir
    }

    /// Orthonormal basis for the dominant span (first `k_dom` canonical axes).
    fn dominant_basis(p: usize, k_dom: usize) -> Array2<f64> {
        let mut q = Array2::<f64>::zeros((p, k_dom));
        for d in 0..k_dom {
            q[[d, d]] = 1.0;
        }
        q
    }

    #[test]
    fn stratification_surfaces_rare_structure_uniform_does_not() {
        // ~1e-4-frequency rare structure among a dominant bulk (the reviewer's
        // regime: a rare curved structure drowned in dominant ones). At this
        // budget the uniform design expects ≈ f·rare = 0.02·30 ≈ 0.6 rare rows —
        // essentially never presented to the birth producer — while the
        // residual-energy census takes the whole high-residual tail (π = 1).
        let n = 300_000usize;
        let p = 8usize;
        let k_dom = 4usize;
        let rare_rows = 30usize; // frequency = 1e-4
        let budget = 6_000usize; // uniform rate f = 0.02

        let (rows, rare_idx) = planted_corpus(n, p, k_dom, rare_rows);
        let dir = temp_shard_dir("surface", &rows, n / 2);
        let mut src = MmapShardSource::open_dir(&dir).expect("open");

        let screen = SpanResidualEnergy::new(dominant_basis(p, k_dom));
        let collected =
            collect_stratified_target(&mut src, &screen, budget, 7).expect("collect");

        // The design must have censused the high-energy tail (π = 1 there).
        let top = collected
            .design
            .strata()
            .last()
            .expect("nonempty strata");
        assert!(
            top.censused && (top.pi - 1.0).abs() < 1e-12,
            "top residual-energy stratum must be censused: {top:?}"
        );

        // Recall of the planted rare rows under stratification.
        let selected: std::collections::BTreeSet<u64> =
            collected.row_ids.iter().copied().collect();
        let rare_seen_strat = rare_idx
            .iter()
            .filter(|&&i| selected.contains(&(i as u64)))
            .count();
        assert!(
            rare_seen_strat as f64 >= 0.8 * rare_idx.len() as f64,
            "stratification must surface ≥80% of the rare structure: {rare_seen_strat}/{}",
            rare_idx.len()
        );

        // Rare rows are censused ⇒ carry weight exactly 1.0.
        for (k, &rid) in collected.row_ids.iter().enumerate() {
            if rare_idx.binary_search(&(rid as usize)).is_ok() {
                assert!(
                    (collected.likelihood_weights[k] - 1.0).abs() < 1e-12,
                    "censused rare row {rid} must have HT weight 1.0"
                );
            }
        }

        // Contrast: a plain uniform designed subsample at the SAME budget.
        // With no residual stratification the rare rows are drawn ∝ frequency,
        // so the birth producer essentially never sees the structure.
        let uniform_measure = RowSamplingMeasure::uniform(n);
        let uniform_sample = uniform_measure.designed_subsample(budget, 7);
        let uniform_selected: std::collections::BTreeSet<usize> =
            uniform_sample.rows.iter().copied().collect();
        let rare_seen_uniform = rare_idx
            .iter()
            .filter(|&&i| uniform_selected.contains(&i))
            .count();

        assert!(
            rare_seen_strat > rare_seen_uniform * 4,
            "stratification must vastly out-recall uniform: strat={rare_seen_strat} \
             uniform={rare_seen_uniform}"
        );
        // The uniform baseline surfaces only a handful (≈ f · rare = 0.05·80 = 4).
        assert!(
            (rare_seen_uniform as f64) <= 0.25 * rare_idx.len() as f64,
            "uniform baseline should surface few rare rows, got {rare_seen_uniform}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn stratified_design_is_horvitz_thompson_unbiased() {
        // Σ 1/π over the selected rows estimates N (the HT corpus-size identity),
        // certifying the design is unbiased — the property that keeps REML /
        // evidence / φ̂ untouched.
        let n = 40_000usize;
        let p = 6usize;
        let k_dom = 3usize;
        let (rows, _rare) = planted_corpus(n, p, k_dom, 40);
        let dir = temp_shard_dir("ht", &rows, n / 3);
        let mut src = MmapShardSource::open_dir(&dir).expect("open");

        let screen = SpanResidualEnergy::new(dominant_basis(p, k_dom));
        let collected =
            collect_stratified_target(&mut src, &screen, 4_000, 3).expect("collect");

        let est = collected.estimated_corpus_rows();
        assert!(
            (est - n as f64).abs() < 0.15 * n as f64,
            "HT corpus estimate {est} too far from N = {n}"
        );
        // Every weight is ≥ 1 (π ≤ 1) and finite.
        assert!(
            collected
                .likelihood_weights
                .iter()
                .all(|&w| w.is_finite() && w >= 1.0 - 1e-12)
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn collection_is_deterministic() {
        let n = 20_000usize;
        let p = 5usize;
        let (rows, _rare) = planted_corpus(n, p, 3, 30);
        let dir = temp_shard_dir("determinism", &rows, n / 2);
        let mut src = MmapShardSource::open_dir(&dir).expect("open");
        let screen = SpanResidualEnergy::new(dominant_basis(p, 3));

        let a = collect_stratified_target(&mut src, &screen, 2_000, 11).expect("a");
        let b = collect_stratified_target(&mut src, &screen, 2_000, 11).expect("b");
        assert_eq!(a.row_ids, b.row_ids, "same seed ⇒ identical selection");
        assert_eq!(a.likelihood_weights, b.likelihood_weights);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn full_budget_is_the_exact_pass_with_unit_weights() {
        let n = 5_000usize;
        let p = 4usize;
        let (rows, _rare) = planted_corpus(n, p, 2, 20);
        let dir = temp_shard_dir("full", &rows, n / 2);
        let mut src = MmapShardSource::open_dir(&dir).expect("open");
        let screen = SpanResidualEnergy::new(dominant_basis(p, 2));

        // Budget ≥ N ⇒ every row censused, weight 1.0, all rows collected.
        let collected =
            collect_stratified_target(&mut src, &screen, n, 1).expect("collect");
        assert_eq!(collected.len(), n);
        assert_eq!(collected.row_ids, (0..n as u64).collect::<Vec<_>>());
        assert!(collected.likelihood_weights.iter().all(|&w| w == 1.0));
        assert!(collected.design.strata().iter().all(|s| s.censused));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn no_residual_variation_degrades_to_uniform_rate() {
        // Constant residual energy everywhere ⇒ one stratum, Neyman degenerate,
        // proportional π = f. The design is then the plain uniform rate.
        let n = 10_000usize;
        let p = 3usize;
        let rows = Array2::<f64>::from_shape_fn((n, p), |(_, d)| if d == 0 { 1.0 } else { 0.0 });
        let dir = temp_shard_dir("flat", &rows, n / 2);
        let mut src = MmapShardSource::open_dir(&dir).expect("open");
        // Empty basis ⇒ raw energy ‖x‖² = 1 for every row (constant).
        let screen = SpanResidualEnergy::new(Array2::<f64>::zeros((p, 0)));
        let budget = 1_000usize;
        let design = design_stratified_subsample(&mut src, &screen, budget).expect("design");
        // All rows share one exponent band ⇒ a single stratum at the uniform rate.
        assert_eq!(design.strata().len(), 1);
        let f = budget as f64 / n as f64;
        assert!(
            (design.strata()[0].pi - f).abs() < 1e-9,
            "flat energy must give the uniform rate f = {f}, got {}",
            design.strata()[0].pi
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn sturges_cap_matches_floor_log2_plus_one() {
        assert_eq!(sturges_stratum_cap(1), 1);
        assert_eq!(sturges_stratum_cap(2), 2);
        assert_eq!(sturges_stratum_cap(255), 8);
        assert_eq!(sturges_stratum_cap(256), 9);
        assert_eq!(sturges_stratum_cap(100_000_000), 27);
    }
}
