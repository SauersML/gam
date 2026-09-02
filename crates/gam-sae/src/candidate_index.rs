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
//! it. A concrete, dependency-free default
//! ([`RandomProjectionFrameSketch`]) is provided: a seeded random-projection /
//! random-hyperplane signature of the atom's orthonormalized column span. The
//! index ([`SaeCandidateIndex`]) is a deterministic multi-table
//! random-hyperplane LSH over those sketches.
//!
//! ## Recall contract
//!
//! Sublinear proposal is only safe if it *almost never* drops a truly-active
//! atom. [`SaeCandidateIndex::recall_report`] takes a set of planted
//! truly-active atoms per row, runs the proposal at a stated candidate budget,
//! and records the rate at which planted atoms appear in the proposed set —
//! **logging every miss** rather than silently truncating. The returned
//! [`RecallReport`] carries `recall@budget` and the full miss list so a caller
//! can widen the budget or fall back to a dense scan for the affected rows.
//!
//! Determinism: every random choice is seeded by an explicit index seed; no
//! clock, no global RNG.

use ndarray::{Array1, Array2, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::HashMap;

/// Salt mixed into the per-table hyperplane seed so the index tables and the
/// default sketch never share a random stream even when handed the same base
/// seed.
const INDEX_HYPERPLANE_SALT: u64 = 0x9E37_79B9_7F4A_7C15;

/// Numerical floor below which a direction / column is treated as zero.
const DIRECTION_NORM_FLOOR: f64 = 1e-12;

/// Lower bound of the auto-derived per-row candidate budget `C` (#985). Below
/// this the proposal set is too small for the solver's accepted active set to
/// have headroom over the planted/active atom count.
pub const CANDIDATE_BUDGET_MIN: usize = 32;

/// Upper bound of the auto-derived per-row candidate budget `C` (#985). The
/// per-row local block stays a small dense solve no matter how large the
/// dictionary grows; beyond this the proposal step stops being the bottleneck
/// reduction it exists to be.
pub const CANDIDATE_BUDGET_MAX: usize = 128;

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
// Default concrete sketch: seeded random projection of the column span
// ---------------------------------------------------------------------------

/// A concrete [`AtomFrameSketch`] built from raw decoder column blocks `B_k`.
///
/// For each atom it orthonormalizes the decoder columns (modified Gram–Schmidt)
/// to obtain a frame `U_k` with orthonormal columns spanning the decoder range,
/// then sketches via a single shared seeded Gaussian random projection
/// `R ∈ ℝ^{s×p}` applied to the in-range component of a direction:
///
/// * `atom_sketch(k)   = normalize( R · u_k0 )`, the sketch of the atom's first
///   (dominant) frame column — a stable representative point used to bucket the
///   atom.
/// * `project_direction(k, d) = R · (U_k U_kᵀ d)`, the sketch of the part of `d`
///   that lies in the atom's range.
/// * `alignment(k, d) = ‖U_kᵀ d‖ / ‖d‖`, the exact in-range energy fraction.
///
/// The shared `R` is a Johnson–Lindenstrauss style random projection, so sketch
/// inner products approximately preserve angles between in-range directions —
/// exactly what the LSH index needs. Everything is seeded; the same atoms +
/// seed always produce the same sketches.
pub struct RandomProjectionFrameSketch {
    /// Orthonormal frame `U_k` per atom, shape `(p, r_k)` with `r_k` ≤ columns.
    frames: Vec<Array2<f64>>,
    /// Shared random projection `R`, shape `(sketch_dim, p)`.
    projection: Array2<f64>,
    /// Ambient output dimension `p`.
    output_dim: usize,
    /// Sketch dimension `s`.
    sketch_dim: usize,
}

impl RandomProjectionFrameSketch {

    /// In-range component `U_k U_kᵀ d` of a direction (length `output_dim`).
    fn in_range_component(&self, atom_id: usize, direction: ArrayView1<f64>) -> Array1<f64> {
        let frame = &self.frames[atom_id];
        // coords = U_kᵀ d  (length r_k)
        let mut comp = Array1::<f64>::zeros(self.output_dim);
        for col in 0..frame.ncols() {
            let u = frame.column(col);
            let coord: f64 = u.iter().zip(direction.iter()).map(|(&a, &b)| a * b).sum();
            for (c, &uval) in comp.iter_mut().zip(u.iter()) {
                *c += coord * uval;
            }
        }
        comp
    }
}

impl AtomFrameSketch for RandomProjectionFrameSketch {
    fn sketch_dim(&self) -> usize {
        self.sketch_dim
    }

    fn output_dim(&self) -> usize {
        self.output_dim
    }

    fn num_atoms(&self) -> usize {
        self.frames.len()
    }

    fn atom_sketch(&self, atom_id: usize) -> Array1<f64> {
        let frame = &self.frames[atom_id];
        // Sketch the dominant (first) frame column as the atom's representative.
        // If the frame is empty (rank-0 atom), fall back to a deterministic
        // nonzero point so the atom is still bucketed somewhere.
        if frame.ncols() == 0 {
            let mut s = self.projection.column(0).to_owned();
            normalize_in_place(&mut s);
            return s;
        }
        let u0 = frame.column(0);
        let mut s = mat_vec(&self.projection, u0);
        normalize_in_place(&mut s);
        s
    }

    fn atom_bucket_sketches(&self, atom_id: usize) -> Vec<Array1<f64>> {
        let frame = &self.frames[atom_id];
        if frame.ncols() == 0 {
            return vec![self.atom_sketch(atom_id)];
        }
        // Bucket representatives covering the atom's whole range (see the
        // trait doc — the deterministic phase-miss fix): one per orthonormal
        // frame column PLUS the pairwise bisectors (u_i ± u_j)/√2. Columns
        // alone leave a 45° worst-case query angle at r = 2 (the diagonal
        // phases), where the per-table cosine-LSH collision is weak enough
        // that a phase sweep still loses ~2% of exact on-plane queries; with
        // the bisectors the worst case drops to 22.5° (r = 2) and the
        // per-query miss probability is negligible at every table
        // configuration the auto-config produces. Signature canonicalization
        // folds ±, so each bisector line needs bucketing once, and the entry
        // count stays r² — bounded by the tiny intrinsic dimension.
        let r = frame.ncols();
        let mut sketches = Vec::with_capacity(r * r);
        for col in 0..r {
            let mut sk = mat_vec(&self.projection, frame.column(col));
            normalize_in_place(&mut sk);
            sketches.push(sk);
        }
        for i in 0..r {
            for j in (i + 1)..r {
                for &sign in &[1.0_f64, -1.0] {
                    let mut dir = frame.column(i).to_owned();
                    dir.scaled_add(sign, &frame.column(j));
                    let mut sk = mat_vec(&self.projection, dir.view());
                    normalize_in_place(&mut sk);
                    sketches.push(sk);
                }
            }
        }
        sketches
    }

    fn project_direction(&self, atom_id: usize, direction: ArrayView1<f64>) -> Array1<f64> {
        let comp = self.in_range_component(atom_id, direction);
        mat_vec(&self.projection, comp.view())
    }

    /// Exact `O(p·s)` probe (#994): every atom shares the one projection `R`,
    /// and the table representatives are `normalize(R · u_k0)`, so the correct
    /// cosine-LSH probe for a direction is simply `normalize(R · d)` — no atom
    /// is touched, and no masked-average approximation is involved.
    fn query_sketch(&self, direction: ArrayView1<f64>) -> Array1<f64> {
        let mut s = mat_vec(&self.projection, direction);
        normalize_in_place(&mut s);
        s
    }

    fn alignment(&self, atom_id: usize, direction: ArrayView1<f64>) -> f64 {
        let dnorm = vec_norm(direction);
        if dnorm < DIRECTION_NORM_FLOOR {
            return 0.0;
        }
        let comp = self.in_range_component(atom_id, direction);
        (vec_norm(comp.view()) / dnorm).clamp(0.0, 1.0)
    }
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
    /// Buckets per table: signature -> atom ids.
    /// Sketch dimension shared by every atom.
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
            num_atoms,
        })
    }

    /// Number of atoms in the index.
    pub fn num_atoms(&self) -> usize {
        self.num_atoms
    }

}

/// Hard upper bound on the routing score (frame alignment) of ANY atom: the
/// alignment `‖U_kᵀ d‖ / ‖d‖` is the fraction of a direction's energy inside the
/// atom's column-space, so it lies in `[0, 1]` for every atom, gathered or not.
/// This is the *true* upper bound that makes [`SaeCandidateIndex::route_exact`]'s
/// LSH fast path sound: a gathered atom at the ceiling cannot be beaten.
pub const ROUTING_ALIGNMENT_UPPER_BOUND: f64 = 1.0;

/// Tolerance for certifying the LSH fast path against [`ROUTING_ALIGNMENT_UPPER_BOUND`].
/// A gathered best within this of the ceiling is treated as a certified global
/// maximizer (floating-point slack on the `‖·‖`/`‖·‖` ratio).
pub const ROUTING_CERT_EPS: f64 = 1e-12;

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

/// Why a planted atom failed to appear in a row's proposed candidate set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MissReason {
    /// The index never gathered this atom into the candidate union (an LSH
    /// recall miss — widen tables / probes).
    NotGathered,
    /// The atom *was* gathered but the budget truncated it (widen the budget).
    TruncatedByBudget,
}

/// One recorded recall miss.
#[derive(Clone, Copy, Debug)]
pub struct RecallMiss {
    /// Row index in the report's input.
    pub row: usize,
    /// The planted atom id that was missed.
    pub atom: usize,
    /// The atom's exact frame alignment with the row direction (diagnostic).
    pub alignment: f64,
    /// Whether the miss was an index miss or a budget truncation.
    pub reason: MissReason,
}

/// Result of [`SaeCandidateIndex::recall_report`].
#[derive(Clone, Debug)]
pub struct RecallReport {
    /// Candidate budget the recall was measured at.
    pub candidate_budget: usize,
    /// Number of rows evaluated.
    pub num_rows: usize,
    /// Total planted truly-active atoms across all rows.
    pub total_planted: usize,
    /// How many of them appeared in the proposed sets.
    pub total_recovered: usize,
    /// `recall@candidate_budget` = recovered / planted (1.0 if nothing planted).
    pub recall: f64,
    /// Mean number of candidates the sublinear gather returned per row — the
    /// sublinearity witness; compare against `num_atoms`.
    pub avg_candidates_gathered: f64,
    /// Total atoms in the index (for the sublinearity ratio).
    pub num_atoms: usize,
    /// Every miss, with its row, atom, alignment, and reason. No silent drops.
    pub misses: Vec<RecallMiss>,
}

/// Result of [`SaeCandidateIndex::proposal_recall_report`] — the two-stage
/// routing license: how much of the EXACT top-`s` rescore the sublinear proposal
/// recovered, plus every miss.
#[derive(Clone, Debug)]
pub struct ProposalRecallReport {
    /// Candidate budget `C` the proposal ran at.
    pub candidate_budget: usize,
    /// Sparse routing width `s` (the top-s the recall is measured over).
    pub top_s: usize,
    /// Number of row directions evaluated.
    pub num_rows: usize,
    /// Total exact-top-s slots across all rows (`Σ_row min(s, finite-scoring atoms)`).
    pub total_true: usize,
    /// How many of those exact-top-s atoms the proposal recovered.
    pub total_recovered: usize,
    /// `recall@s` = recovered / true (`1.0` when there was nothing to recover — a
    /// null row set or `s = 0`). At `1.0` the proposal is licensed to stand in for
    /// the exact rescore over this regime.
    pub recall: f64,
    /// Mean gathered-candidate count per row — the sublinearity witness; compare
    /// against `num_atoms` (see [`ProposalRecallReport::sublinearity_ratio`]).
    pub avg_candidates_gathered: f64,
    /// Total atoms in the index (for the sublinearity ratio).
    pub num_atoms: usize,
    /// Every miss (true-top-s atom the proposal dropped), with row, atom,
    /// alignment, and reason. No silent drops — the license's honesty contract.
    pub misses: Vec<RecallMiss>,
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
            let u1 = rng.random::<f64>().max(1e-16);
            let u2 = rng.random::<f64>();
            m[(r, c)] = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        }
    }
    m
}

/// `M · v` for `M` shape `(rows, cols)`, `v` length `cols`.
fn mat_vec(m: &Array2<f64>, v: ArrayView1<f64>) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(m.nrows());
    for r in 0..m.nrows() {
        let row = m.row(r);
        out[r] = row.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum();
    }
    out
}

#[inline]
fn vec_norm(v: ArrayView1<f64>) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

#[inline]
fn normalize_in_place(v: &mut Array1<f64>) {
    let n = vec_norm(v.view());
    if n > DIRECTION_NORM_FLOOR {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

