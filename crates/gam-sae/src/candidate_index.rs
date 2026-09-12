//! Sublinear candidate-atom index for active-set proposal (#985 part 1).
//!
//! A frontier SAE dictionary holds `K ≈ 10^4–10^5` atoms. The per-row *local*
//! block — the small linear/Newton system over the atoms that are actually
//! active in a row — is cheap, because the active set collapses it to a handful
//! of atoms. The expensive step is *proposing* that active set: a naive scan
//! scores every one of the `K` atom frames against every row, which is `O(K)`
//! per row and dominates the whole solve once `K` is large.
//!
//! This module builds a **sublinear** candidate index over per-atom *sketches*
//! of each atom's decoder column-space (its Grassmann frame `U_k`). Given a row
//! residual direction it returns the top candidate atom ids likely to be
//! active, touching only `O(log K)`-ish buckets instead of all `K` atoms.
//!
//! ## Layering against Track 1
//!
//! Track 1 owns the *real* atom frames `U_k` and has not landed yet, so this
//! module is written against a [`AtomFrameSketch`] trait. Any frame source —
//! the eventual Grassmann frames, or the decoder column blocks `B_k` already
//! present on [`crate::manifold::SaeManifoldAtom`] — can implement
//! it. The index ([`SaeCandidateIndex`]) is a deterministic multi-table
//! random-hyperplane LSH over those sketches.
//!
//! Determinism: every random choice is seeded by an explicit index seed; no
//! clock, no global RNG.

use ndarray::{Array1, Array2, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::{HashMap, HashSet};

/// Salt mixed into the per-table hyperplane seed.
const INDEX_HYPERPLANE_SALT: u64 = 0x9E37_79B9_7F4A_7C15;

/// Lower bound of the auto-derived per-row candidate budget `C` (#985). Below
/// this the proposal set is too small for the solver's accepted active set to
/// have headroom over the planted/active atom count.
pub const CANDIDATE_BUDGET_MIN: usize = 32;

/// Upper bound of the auto-derived per-row candidate budget `C` (#985). The
/// per-row local block stays a small dense solve no matter how large the
/// dictionary grows; beyond this the proposal step stops being the bottleneck
/// reduction it exists to be.
pub const CANDIDATE_BUDGET_MAX: usize = 128;

/// Auto-derive the per-row candidate budget `C` from the dictionary size `K`
/// (#985): `C = 8·⌈log₂ K⌉`, clamped to
/// [[`CANDIDATE_BUDGET_MIN`], [`CANDIDATE_BUDGET_MAX`]]. Logarithmic growth
/// keeps the per-row local block effectively constant-size while giving larger
/// dictionaries a little more recall headroom; the clamp realizes the issue's
/// `C ≈ 32–128` band. Magic-by-default: derived from `K` alone, no flag.
///
/// Concretely: `K = 64 → 48`, `K = 1024 → 80`, `K = 10⁵ → 128`.
pub fn auto_candidate_budget(num_atoms: usize) -> usize {
    let log2 = if num_atoms <= 1 {
        1
    } else {
        (usize::BITS - (num_atoms - 1).leading_zeros()) as usize
    };
    (8 * log2).clamp(CANDIDATE_BUDGET_MIN, CANDIDATE_BUDGET_MAX)
}

// ---------------------------------------------------------------------------
// Sketch interface
// ---------------------------------------------------------------------------

/// A low-dimensional sketch of one atom's decoder column-space (its Grassmann
/// frame `U_k`).
///
/// The index never needs the full frame: it only needs (a) the sketch
/// dimension, shared by every atom in a dictionary, and (b), for any query
/// direction in output space, the atom's *sketch coordinates* of that direction
/// — i.e. the projection of the direction onto the atom's column-space,
/// expressed in the sketch's coordinates. A frame `U_k` (orthonormal columns
/// spanning the decoder range) yields these as `sketch = R · (U_kᵀ d)` for a
/// shared random projection `R`; a raw decoder block `B_k` yields them by first
/// orthonormalizing its columns. Both are valid implementors.
pub trait AtomFrameSketch {
    /// Dimension of the sketch vectors this implementor produces. Must be the
    /// same positive value for every atom in one dictionary so the index can
    /// build a single hyperplane bank.
    fn sketch_dim(&self) -> usize;

    /// Dimension of the ambient output space the query directions live in.
    fn output_dim(&self) -> usize;

    /// Number of atoms this source can sketch.
    fn num_atoms(&self) -> usize;

    /// Sketch of atom `atom_id`'s *frame itself* (a representative point of the
    /// atom's column-space on the sphere of sketch space), used to place the
    /// atom into the LSH tables at build time. Returns a vector of length
    /// [`AtomFrameSketch::sketch_dim`].
    fn atom_sketch(&self, atom_id: usize) -> Array1<f64>;

    /// ALL bucket representatives for atom `atom_id` (each of length
    /// [`AtomFrameSketch::sketch_dim`]).
    ///
    /// A single representative point cannot cover an atom whose range is a
    /// genuine SUBSPACE: an on-manifold query direction sweeps the whole
    /// r-plane, and cosine-LSH collision decays with the query→representative
    /// angle, which reaches `arccos(1/√r)` (45° at r = 2, 54.7° at r = 3) even
    /// for a PERFECT on-atom query. Bucketing only the dominant column made
    /// the miss probability deterministic in the query's phase — measured as
    /// an 11.4% routing miss on planted K=1024 circle atoms — because every
    /// table shares the same lone representative, so more tables cannot
    /// recover it. Bucketing one representative PER FRAME COLUMN bounds the
    /// worst-case angle to the nearest bucketed point by `arccos(1/√r)` with
    /// equality only on the diagonal, restoring the per-table collision the
    /// table count was sized for. Build cost grows by the factor r (bucket
    /// entries only); queries are unchanged and the exact alignment rescore
    /// already owns ranking.
    ///
    /// The default covers implementors whose range is genuinely
    /// one-directional: the single [`AtomFrameSketch::atom_sketch`].
    fn atom_bucket_sketches(&self, atom_id: usize) -> Vec<Array1<f64>> {
        vec![self.atom_sketch(atom_id)]
    }

    /// Sketch of a query *direction* `d` (length [`AtomFrameSketch::output_dim`])
    /// as seen through atom `atom_id`'s frame: the direction's component inside
    /// the atom's column-space, mapped into sketch coordinates. Used at query
    /// time to score how strongly a row residual aligns with the atom.
    fn project_direction(&self, atom_id: usize, direction: ArrayView1<f64>) -> Array1<f64>;

    /// Alignment score in `[0, 1]`: the fraction of the query direction's energy
    /// that lies inside atom `atom_id`'s column-space. `1.0` means the direction
    /// lies fully in the atom's range, `0.0` means it is orthogonal. Used to
    /// rank the (small) candidate set the index returns.
    fn alignment(&self, atom_id: usize, direction: ArrayView1<f64>) -> f64;

    /// Sketch-space **probe** for a raw query direction (length
    /// [`AtomFrameSketch::sketch_dim`]), comparable to the
    /// [`AtomFrameSketch::atom_sketch`] representatives the LSH tables were
    /// built from (#994).
    ///
    /// Implementors must return the exact cosine-LSH probe for their sketching
    /// policy. For the shared-projection sketch this is `normalize(R · d)`,
    /// `O(p · s)` per query, touching no atom.
    fn query_sketch(&self, direction: ArrayView1<f64>) -> Array1<f64>;
}

// ---------------------------------------------------------------------------
// Sublinear index: multi-table random-hyperplane LSH over sketches
// ---------------------------------------------------------------------------

/// A deterministic, sublinear candidate index over atom-frame sketches.
///
/// The structure is a **random-hyperplane LSH** with `num_tables` independent
/// tables, each defined by `bits_per_table` seeded random hyperplanes in sketch
/// space. An atom's sketch is reduced to a `bits_per_table`-bit sign signature
/// per table (the sign of its dot with each hyperplane), and the atom id is
/// stored in the bucket keyed by that signature. At query time the query
/// direction is sketched *through each atom's frame*; we instead hash the *query
/// sketch* per table and gather the union of atoms in the matching (and, to
/// improve recall, the Hamming-1 neighbouring) buckets. Because each table
/// touches only the atoms colliding in one bucket, total work is sublinear in
/// `K` for well-spread sketches.
///
/// The gathered candidates are then ranked by exact
/// [`AtomFrameSketch::alignment`] and the top `candidate_budget` are returned.
/// All hyperplanes are seeded; building twice with the same seed yields byte-
/// identical tables.
pub struct SaeCandidateIndex {
    /// Hyperplane banks, one per table: each `(bits_per_table, sketch_dim)`.
    hyperplanes: Vec<Array2<f64>>,
    /// Buckets per table: signature -> atom ids.
    tables: Vec<HashMap<u64, Vec<usize>>>,
    /// Sketch dimension shared by every atom.
    sketch_dim: usize,
    /// Number of atoms indexed.
    num_atoms: usize,
}

/// Tuning for [`SaeCandidateIndex::build`]. All fields are explicit so the index
/// never reads global state; no CLI flags.
#[derive(Clone, Copy, Debug)]
pub struct IndexConfig {
    /// Number of independent LSH tables. More tables → higher recall, more work.
    pub num_tables: usize,
    /// Random hyperplanes per table (signature bit-width). More bits → finer
    /// buckets (fewer collisions, lower recall per table).
    pub bits_per_table: usize,
    /// Whether to also probe Hamming-distance-1 neighbouring buckets per table
    /// (multi-probe LSH). Cheap and a large recall win; kept on by default.
    pub multiprobe: bool,
    /// Master seed for all hyperplane banks.
    pub seed: u64,
}

impl IndexConfig {
    /// A default configuration sized for a sketch of dimension `sketch_dim` and
    /// roughly `num_atoms` atoms. Chooses `bits_per_table ≈ log2(num_atoms)`
    /// (capped by the sketch dimension) so the expected bucket occupancy is a
    /// small constant, and a handful of tables for recall — both grow only
    /// logarithmically in `num_atoms`, keeping queries sublinear.
    pub fn auto(sketch_dim: usize, num_atoms: usize, seed: u64) -> Self {
        let log2 = |n: usize| -> usize {
            if n <= 1 {
                1
            } else {
                (usize::BITS - (n - 1).leading_zeros()) as usize
            }
        };
        // Cap at 63: sign_signature packs bits into a u64, so bits_per_table must be ≤ 63.
        let bits = log2(num_atoms.max(2)).clamp(1, sketch_dim.max(1).min(63));
        // Aim for ~constant per-bucket occupancy; a few tables recover recall
        // lost to any single table's quantization.
        let num_tables = log2(num_atoms.max(2)).clamp(4, 16);
        Self {
            num_tables,
            bits_per_table: bits,
            multiprobe: true,
            seed,
        }
    }
}

impl SaeCandidateIndex {
    /// Build the index over every atom of `sketch`.
    pub fn build<S: AtomFrameSketch>(sketch: &S, config: IndexConfig) -> Result<Self, String> {
        let sketch_dim = sketch.sketch_dim();
        if sketch_dim == 0 {
            return Err("SaeCandidateIndex: sketch_dim must be positive".into());
        }
        if config.num_tables == 0 || config.bits_per_table == 0 {
            return Err("SaeCandidateIndex: num_tables and bits_per_table must be positive".into());
        }
        // sign_signature packs bits into a u64 with `1u64 << r` for r in 0..bits_per_table.
        // Shifting by 64+ is a panic in debug and undefined behaviour in release; cap at 63.
        if config.bits_per_table > 63 {
            return Err(format!(
                "SaeCandidateIndex: bits_per_table {} exceeds 63 (u64 signature limit)",
                config.bits_per_table
            ));
        }
        let num_atoms = sketch.num_atoms();

        // One seeded hyperplane bank per table; seed is mixed per-table so the
        // tables are independent yet fully reproducible.
        let hyperplanes: Vec<Array2<f64>> = (0..config.num_tables)
            .map(|t| {
                let table_seed = mix_seed(config.seed ^ INDEX_HYPERPLANE_SALT, t as u64);
                gaussian_projection(config.bits_per_table, sketch_dim, table_seed)
            })
            .collect();

        let mut tables: Vec<HashMap<u64, Vec<usize>>> =
            (0..config.num_tables).map(|_| HashMap::new()).collect();

        for atom_id in 0..num_atoms {
            let bucket_sketches = sketch.atom_bucket_sketches(atom_id);
            if bucket_sketches.is_empty() {
                return Err(format!(
                    "SaeCandidateIndex: atom {atom_id} produced no bucket representatives"
                ));
            }
            for s in &bucket_sketches {
                if s.len() != sketch_dim {
                    return Err(format!(
                        "SaeCandidateIndex: atom {atom_id} sketch length {} != sketch_dim {sketch_dim}",
                        s.len()
                    ));
                }
                for (table, bank) in tables.iter_mut().zip(hyperplanes.iter()) {
                    let sig = sign_signature(bank, s.view());
                    let bucket = table.entry(sig).or_default();
                    // Distinct representatives of one atom can share a bucket;
                    // store the id once so occupancy statistics stay honest.
                    if bucket.last() != Some(&atom_id) {
                        bucket.push(atom_id);
                    }
                }
            }
        }

        Ok(Self {
            hyperplanes,
            tables,
            sketch_dim,
            num_atoms,
        })
    }

    /// Number of atoms in the index.
    pub fn num_atoms(&self) -> usize {
        self.num_atoms
    }

    /// Gather the raw candidate atom-id set for a query `direction`, *without*
    /// ranking or budget truncation. This is the sublinear part: it sketches the
    /// query once per table (using a frame-agnostic global query sketch — the
    /// query direction projected by the index's own representative projection)
    /// and unions the colliding buckets (plus Hamming-1 neighbours when
    /// multi-probe is enabled).
    ///
    /// `query_sketch` is the sketch-space query vector (length `sketch_dim`),
    /// produced by the caller from the row residual via the
    /// [`AtomFrameSketch`]. We probe each table with this single vector.
    pub fn gather_candidates(&self, query_sketch: ArrayView1<f64>, multiprobe: bool) -> Vec<usize> {
        let mut seen: HashSet<usize> = HashSet::new();
        for (table, bank) in self.tables.iter().zip(self.hyperplanes.iter()) {
            let (sig, margins) = sign_signature_with_margins(bank, query_sketch);
            if let Some(ids) = table.get(&sig) {
                seen.extend(ids.iter().copied());
            }
            if multiprobe {
                // Flip the lowest-margin bit (the one most likely to be on the
                // wrong side of its hyperplane) to reach the nearest neighbour
                // bucket — standard multi-probe LSH, biggest recall win.
                let flip_bit = lowest_margin_bit(&margins);
                let neighbour = canonical_signature(sig ^ (1u64 << flip_bit), bank.nrows());
                if let Some(ids) = table.get(&neighbour) {
                    seen.extend(ids.iter().copied());
                }
            }
        }
        let mut out: Vec<usize> = seen.into_iter().collect();
        out.sort_unstable();
        out
    }

    /// Propose the top `candidate_budget` atoms for a row whose residual is
    /// `direction` (length `sketch.output_dim()`), ranked by exact frame
    /// alignment.
    ///
    /// Pipeline: probe with [`AtomFrameSketch::query_sketch`] (`O(p·s)` for
    /// shared-projection sketches, #994 — no atom is touched before the
    /// gather), gather the sublinear candidate union, score each by
    /// [`AtomFrameSketch::alignment`], and keep the highest-scoring
    /// `candidate_budget`.
    ///
    /// Returns `(proposed_ids, dropped_for_budget)` where the second element
    /// lists every gathered candidate that was truncated by the budget (never
    /// silently discarded).
    pub fn propose<S: AtomFrameSketch>(
        &self,
        sketch: &S,
        direction: ArrayView1<f64>,
        candidate_budget: usize,
        config_multiprobe: bool,
    ) -> Proposal {
        let query_sketch = sketch.query_sketch(direction);
        let gathered = if query_sketch.len() == self.sketch_dim {
            self.gather_candidates(query_sketch.view(), config_multiprobe)
        } else {
            // A probe of the wrong dimension cannot be hashed against the
            // tables; gather nothing rather than hash garbage. The recall
            // report will then attribute every planted atom to `NotGathered`,
            // which is the loud, attributable failure mode.
            Vec::new()
        };

        // Exact-score every gathered candidate by frame alignment.
        let mut scored: Vec<(usize, f64)> = gathered
            .iter()
            .map(|&id| (id, sketch.alignment(id, direction)))
            .collect();
        // Descending by alignment; ties broken by id for determinism.
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });

        let keep = candidate_budget.min(scored.len());
        let proposed: Vec<usize> = scored[..keep].iter().map(|&(id, _)| id).collect();
        let dropped_for_budget: Vec<usize> = scored[keep..].iter().map(|&(id, _)| id).collect();

        Proposal {
            proposed,
            dropped_for_budget,
            gathered_count: gathered.len(),
        }
    }

    /// EXACT routing (#1777 / roadmap "real exact-routing guarantee"): return the
    /// **global argmax** of the routing score over the WHOLE dictionary — the atom
    /// whose frame best aligns with `direction` — with a guarantee that no
    /// ungathered atom is silently better.
    ///
    /// The sublinear [`Self::propose`] gather is only a HEURISTIC: a gathered atom
    /// at alignment `0.6` does not rule out an *ungathered* atom at `1.0`, because
    /// the gather's alignment is a lower bound on the selected atom, never an upper
    /// bound on the atoms it skipped. So `propose` alone can silently miss the true
    /// best atom. This method closes that hole and is the path the encode router
    /// uses.
    ///
    /// Correctness mechanism (sound, not heuristic):
    /// * **LSH fast path with a TRUE upper bound.** The routing score is the frame
    ///   alignment `‖U_kᵀ d‖ / ‖d‖ ∈ [0, 1]`, so [`ROUTING_ALIGNMENT_UPPER_BOUND`]
    ///   (`1.0`) is a hard ceiling for *every* atom — gathered or not. If the best
    ///   gathered candidate already sits within [`ROUTING_CERT_EPS`] of that
    ///   ceiling, no ungathered atom can beat it: the gathered best is a certified
    ///   global score-maximizer and we return it WITHOUT a full scan
    ///   ([`ExactRoute::lsh_certified`] = `true`).
    /// * **Exact fallback otherwise.** When the gathered best is not certified by
    ///   that bound (no tighter sound bound on ungathered atoms is available), run
    ///   the full [`brute_force_best_atom`] scan and return its argmax. This is the
    ///   ground truth — correctness over speed, exactly the roadmap contract.
    ///
    /// In both branches the returned atom has no atom of strictly greater routing
    /// score anywhere in the dictionary (no silent miss). Returns `None` only for
    /// an empty dictionary or an all-non-finite scan (a degenerate sketch).
    pub fn route_exact<S: AtomFrameSketch>(
        &self,
        sketch: &S,
        direction: ArrayView1<f64>,
        candidate_budget: usize,
        multiprobe: bool,
    ) -> Option<ExactRoute> {
        // Heuristic LSH gather first (sublinear) — the speed fast path.
        let proposal = self.propose(sketch, direction, candidate_budget, multiprobe);
        let lsh_best = proposal
            .proposed
            .first()
            .copied()
            .map(|id| (id, sketch.alignment(id, direction)));

        if let Some((b, a_b)) = lsh_best {
            if a_b.is_finite() && a_b >= ROUTING_ALIGNMENT_UPPER_BOUND - ROUTING_CERT_EPS {
                // Universal-bound certificate: the routing score is capped at 1.0
                // for EVERY atom, so a gathered atom already at the ceiling cannot
                // be beaten by any ungathered one. Sound global optimality with no
                // full scan.
                return Some(ExactRoute {
                    atom: b,
                    alignment: a_b,
                    lsh_certified: true,
                    lsh_agreed: true,
                    did_full_scan: false,
                });
            }
        }

        // Not certified by the bound ⇒ the gather might have missed a better
        // ungathered atom. The only sound recourse without a tighter per-atom upper
        // bound is the exact full scan: it IS the global argmax.
        let (atom, alignment) = brute_force_best_atom(sketch, direction)?;
        let lsh_agreed = lsh_best.is_some_and(|(b, _)| b == atom);
        Some(ExactRoute {
            atom,
            alignment,
            lsh_certified: false,
            lsh_agreed,
            did_full_scan: true,
        })
    }
}

/// Hard upper bound on the routing score (frame alignment) of ANY atom: the
/// alignment `‖U_kᵀ d‖ / ‖d‖` is the fraction of a direction's energy inside the
/// atom's column-space, so it lies in `[0, 1]` for every atom, gathered or not.
/// This is the *true* upper bound that makes `SaeCandidateIndex::route_exact`'s
/// LSH fast path sound: a gathered atom at the ceiling cannot be beaten.
pub const ROUTING_ALIGNMENT_UPPER_BOUND: f64 = 1.0;

/// Tolerance for certifying the LSH fast path against [`ROUTING_ALIGNMENT_UPPER_BOUND`].
/// A gathered best within this of the ceiling is treated as a certified global
/// maximizer (floating-point slack on the `‖·‖`/`‖·‖` ratio).
pub const ROUTING_CERT_EPS: f64 = 1e-12;

/// Brute-force EXACT global argmax of the routing score (frame alignment) over the
/// WHOLE dictionary: scan every atom, return `(atom_id, alignment)` of the highest
/// scorer. Ties break to the LOWEST id (a strict `>` replacement keeps the first
/// maximizer), matching [`SaeCandidateIndex::propose`]'s id-ascending tie-break so
/// the two agree atom-for-atom. Non-finite alignments are skipped. Returns `None`
/// for an empty dictionary (or one whose every atom scored non-finite).
///
/// This is `O(K)` per call and is the ground truth [`SaeCandidateIndex::route_exact`]
/// falls back to whenever the LSH gather is not certified optimal.
pub fn brute_force_best_atom<S: AtomFrameSketch>(
    sketch: &S,
    direction: ArrayView1<f64>,
) -> Option<(usize, f64)> {
    let mut best: Option<(usize, f64)> = None;
    for id in 0..sketch.num_atoms() {
        let a = sketch.alignment(id, direction);
        if !a.is_finite() {
            continue;
        }
        match best {
            Some((_, ba)) if a <= ba => {}
            _ => best = Some((id, a)),
        }
    }
    best
}

/// Result of [`SaeCandidateIndex::route_exact`]: the certified-or-exact global
/// argmax of the routing score for one row, plus how it was obtained.
#[derive(Clone, Copy, Debug)]
pub struct ExactRoute {
    /// The chosen atom id — a GLOBAL routing-score argmax (no atom in the
    /// dictionary has a strictly greater score). No silent miss.
    pub atom: usize,
    /// The chosen atom's exact frame alignment with the row direction.
    pub alignment: f64,
    /// `true` ⇒ the LSH fast path certified optimality via the universal upper
    /// bound (gathered best at the `1.0` ceiling); no full scan was needed.
    pub lsh_certified: bool,
    /// Whether the LSH gather's best candidate equalled the returned argmax.
    /// `true` whenever `lsh_certified`; a diagnostic of the gather's recall.
    pub lsh_agreed: bool,
    /// `true` ⇒ the exact `O(K)` fallback scan ran (the LSH bound did not certify).
    pub did_full_scan: bool,
}

/// One row's proposal: the budgeted candidate set plus what the budget dropped.
#[derive(Clone, Debug)]
pub struct Proposal {
    /// The top `candidate_budget` atom ids by frame alignment.
    pub proposed: Vec<usize>,
    /// Gathered candidates truncated by the budget — logged, never silent.
    pub dropped_for_budget: Vec<usize>,
    /// How many candidates the sublinear gather returned before budgeting.
    pub gathered_count: usize,
}

// ---------------------------------------------------------------------------
// Helpers (deterministic, dependency-light)
// ---------------------------------------------------------------------------

/// Mix a base seed with an index into a well-spread `u64` (SplitMix64 finalizer
/// on the sum). Deterministic, no clock.
#[inline]
fn mix_seed(base: u64, idx: u64) -> u64 {
    // Finalize `base + idx·G` with the canonical SplitMix64 step. The stateful
    // form adds G internally, so pre-subtract one G to land on the same input
    // and keep the output bit-identical to the previous inlined finalizer.
    let mut state = base
        .wrapping_add(idx.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_sub(0x9E37_79B9_7F4A_7C15);
    gam_linalg::utils::splitmix64(&mut state)
}

/// A seeded Gaussian random matrix of shape `(rows, cols)` (rows of hyperplanes
/// / projection rows). Uses Box–Muller off a seeded `StdRng`.
fn gaussian_projection(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
    use rand::RngExt as _;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut m = Array2::<f64>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            // `random` draws from [0, 1); its complement lies in (0, 1], where `ln` is finite.
            let u1 = 1.0 - rng.random::<f64>();
            let u2 = rng.random::<f64>();
            m[(r, c)] = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        }
    }
    m
}

#[inline]
fn vec_norm(v: ArrayView1<f64>) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Pack the sign bits of `bank · s` into a `u64` signature. `bank` is
/// `(bits, sketch_dim)`; `bits ≤ 64` (enforced by config-derived bit widths).
/// Canonicalize a sign signature under global sign flip.
///
/// The routing metric is the SIGN-FREE subspace alignment `‖U_kᵀd‖/‖d‖`, but a
/// raw sign signature is not sign-free: `sig(−s)` is the bitwise complement of
/// `sig(s)`, so a query anti-aligned with a bucketed representative lands in
/// the complementary bucket and misses DETERMINISTICALLY (caught by the
/// phase-sweep regression at the negative axis phases: `d = −u` never found
/// `u`'s bucket). Folding each signature onto the lexicographic minimum of
/// {sig, ¬sig} makes the hash antipodally invariant — exactly the invariance
/// the metric has — at the cost of one bit of table discrimination (bucket
/// occupancy doubles), which the exact alignment rescore absorbs.
fn canonical_signature(sig: u64, bits: usize) -> u64 {
    let mask = if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    let complement = (!sig) & mask;
    sig.min(complement)
}

fn sign_signature(bank: &Array2<f64>, s: ArrayView1<f64>) -> u64 {
    let mut sig = 0u64;
    for r in 0..bank.nrows() {
        let row = bank.row(r);
        let dot: f64 = row.iter().zip(s.iter()).map(|(&a, &b)| a * b).sum();
        if dot >= 0.0 {
            sig |= 1u64 << r;
        }
    }
    canonical_signature(sig, bank.nrows())
}

/// Signature plus per-bit signed margins (the dot products), used by multi-probe
/// to find the least-confident bit to flip.
fn sign_signature_with_margins(bank: &Array2<f64>, s: ArrayView1<f64>) -> (u64, Vec<f64>) {
    let mut sig = 0u64;
    let mut margins = Vec::with_capacity(bank.nrows());
    for r in 0..bank.nrows() {
        let row = bank.row(r);
        let dot: f64 = row.iter().zip(s.iter()).map(|(&a, &b)| a * b).sum();
        if dot >= 0.0 {
            sig |= 1u64 << r;
        }
        margins.push(dot);
    }
    (canonical_signature(sig, bank.nrows()), margins)
}

/// Index of the bit whose hyperplane the query sits closest to (smallest `|dot|`)
/// — the most likely to have landed in the wrong bucket.
fn lowest_margin_bit(margins: &[f64]) -> usize {
    let mut best = 0usize;
    let mut best_abs = f64::INFINITY;
    for (i, &m) in margins.iter().enumerate() {
        let a = m.abs();
        if a < best_abs {
            best_abs = a;
            best = i;
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

