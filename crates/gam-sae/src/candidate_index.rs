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
//! atom. `SaeCandidateIndex::recall_report` takes a set of planted
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
use std::collections::{HashMap, HashSet};

/// Salt mixed into the per-table hyperplane seed so the index tables and the
/// default sketch never share a random stream even when handed the same base
/// seed.
const INDEX_HYPERPLANE_SALT: u64 = 0x9E37_79B9_7F4A_7C15;

/// Salt for the default random-projection sketch's projection matrix.
const SKETCH_PROJECTION_SALT: u64 = 0xC2B2_AE3D_27D4_EB4F;

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

/// Two-stage routing shortlist size (stage-1 candidate budget `C`) DERIVED from
/// the **routability floor** (#985 / E1) — no magic constant.
///
/// The default large-`K` route is two-stage: an LSH shortlist of size `C` (stage
/// 1, sublinear) is exactly rescored on the frame gate (stage 2). This function
/// sizes that shortlist from the interference floor rather than a hand-tuned band.
///
/// The routability floor's **union-bound term** ([`crate::routability`]) is
/// `u = √(2·ln(K/δ)/p)`: the deviation, in target-to-clutter units, at which the
/// EXPECTED number of off-target atoms whose random-projection gate exceeds a
/// routable target's gate equals `δ`, because `K·exp(−p·u²/2) = δ` under the same
/// Gaussian–Lipschitz union bound that sets the floor. Isolating the `s` genuine
/// top-`s` winners among `K` atoms down to that floor, at confidence `1 − δ`,
/// costs a shortlist that additionally covers the confusable band, whose
/// log-multiplicity is exactly `p·u² = 2·ln(K/δ)`:
///
/// ```text
///     C  =  s  +  ⌈ p·u² ⌉  =  s + ⌈ 2·ln(K/δ) ⌉.
/// ```
///
/// The term `p·u²` is read straight off [`crate::routability::routability_floor`]
/// (union = `floor − √(1/p)`), so the shortlist and the interference floor move
/// together. `C` is LOGARITHMIC (sublinear) in `K`, monotone non-decreasing in
/// `K`, and widens as the confidence tightens (`δ → 0`); it is clamped to
/// `[s+1, K]`. The recall license
/// ([`SaeCandidateIndex::proposal_recall_report`]) then certifies empirically that
/// at this `C` the two-stage route recovers the exact top-`s` for the routable
/// rows — the "matches exhaustive top-s to a derived bound" acceptance.
pub fn routability_shortlist_size(p: usize, num_atoms: usize, top_s: usize, delta: f64) -> usize {
    let k = num_atoms.max(1);
    // The closed-form floor for the linear atom lane (b_max = 1). Reading it back
    // out ties the shortlist to the exact quantity the router's floor is built on.
    let floor = crate::routability::routability_floor(p.max(1), k, 1, delta);
    // Peel the subspace term √(b_max/p) = √(1/p) to recover the union term u.
    let subspace = (1.0 / p.max(1) as f64).sqrt();
    let union = (floor.floor - subspace).max(0.0);
    // Confusable-band log-multiplicity p·u² = 2·ln(K/δ).
    let band = (p.max(1) as f64) * union * union;
    let c = top_s.saturating_add(band.ceil() as usize);
    c.clamp(top_s.saturating_add(1), k)
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
    /// Build the sketch from decoder column blocks.
    ///
    /// `decoder_blocks[k]` is `B_k` with shape `(p, m_k)`: `p` rows in output
    /// space, `m_k` decoder columns for atom `k`. (`SaeManifoldAtom` stores the
    /// transpose `(m_k, p)`; orient it `p`-rows before passing in.) All blocks
    /// must share the same `p`. `sketch_dim` is the target sketch length `s`;
    /// `seed` makes the projection deterministic.
    pub fn from_decoder_blocks(
        decoder_blocks: &[Array2<f64>],
        sketch_dim: usize,
        seed: u64,
    ) -> Result<Self, String> {
        if decoder_blocks.is_empty() {
            return Err("RandomProjectionFrameSketch: need at least one decoder block".into());
        }
        if sketch_dim == 0 {
            return Err("RandomProjectionFrameSketch: sketch_dim must be positive".into());
        }
        let output_dim = decoder_blocks[0].nrows();
        if output_dim == 0 {
            return Err("RandomProjectionFrameSketch: output dimension must be positive".into());
        }
        for (k, block) in decoder_blocks.iter().enumerate() {
            if block.nrows() != output_dim {
                return Err(format!(
                    "RandomProjectionFrameSketch: atom {k} has {} output rows, expected {output_dim}",
                    block.nrows()
                ));
            }
        }

        let frames: Vec<Array2<f64>> = decoder_blocks.iter().map(orthonormal_frame).collect();

        let projection = gaussian_projection(sketch_dim, output_dim, seed ^ SKETCH_PROJECTION_SALT);

        Ok(Self {
            frames,
            projection,
            output_dim,
            sketch_dim,
        })
    }

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
        if dnorm == 0.0 {
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

    /// Recall contract. For a set of rows, each with planted truly-active atom
    /// ids and a residual direction, run [`SaeCandidateIndex::propose`] at the
    /// given `candidate_budget` and record what fraction of planted atoms
    /// appear in the proposed set. Every miss is logged — no silent truncation.
    ///
    /// `rows` is `(direction, planted_active_ids)` per row.
    pub fn recall_report<S: AtomFrameSketch>(
        &self,
        sketch: &S,
        rows: &[(Array1<f64>, Vec<usize>)],
        candidate_budget: usize,
        multiprobe: bool,
    ) -> RecallReport {
        let mut total_planted: usize = 0;
        let mut total_recovered: usize = 0;
        let mut misses: Vec<RecallMiss> = Vec::new();
        let mut total_gathered: usize = 0;

        for (row_idx, (direction, planted)) in rows.iter().enumerate() {
            let proposal = self.propose(sketch, direction.view(), candidate_budget, multiprobe);
            total_gathered += proposal.gathered_count;
            let proposed_set: HashSet<usize> = proposal.proposed.iter().copied().collect();
            // A candidate that was gathered but truncated by the budget counts
            // as a miss *attributable to the budget*; one never gathered at all
            // is a miss *attributable to the index*. We record both, flagged.
            let dropped_set: HashSet<usize> = proposal.dropped_for_budget.iter().copied().collect();

            for &atom in planted {
                total_planted += 1;
                if proposed_set.contains(&atom) {
                    total_recovered += 1;
                } else {
                    let reason = if dropped_set.contains(&atom) {
                        MissReason::TruncatedByBudget
                    } else {
                        MissReason::NotGathered
                    };
                    misses.push(RecallMiss {
                        row: row_idx,
                        atom,
                        alignment: sketch.alignment(atom, direction.view()),
                        reason,
                    });
                }
            }
        }

        let recall = if total_planted == 0 {
            1.0
        } else {
            total_recovered as f64 / total_planted as f64
        };
        let avg_gathered = if rows.is_empty() {
            0.0
        } else {
            total_gathered as f64 / rows.len() as f64
        };

        RecallReport {
            candidate_budget,
            num_rows: rows.len(),
            total_planted,
            total_recovered,
            recall,
            avg_candidates_gathered: avg_gathered,
            num_atoms: self.num_atoms,
            misses,
        }
    }

    /// The two-stage routing **LICENSE** (#985 / E1): the fraction of the EXACT
    /// top-`s` atoms (the true rescore winners, [`brute_force_top_s`]) that the
    /// sublinear two-stage proposal ([`SaeCandidateIndex::propose`] at
    /// `candidate_budget = C`) recovers, per row and in aggregate, with EVERY miss
    /// logged (never silent).
    ///
    /// This is the *license* the default two-stage router runs under. Unlike
    /// [`SaeCandidateIndex::recall_report`] it needs NO planted ground truth: the
    /// reference is the exact rescore itself, so it can be run over any sample of
    /// real row directions to certify that dropping from the `O(K)` exact scan to
    /// the `O(C)` proposal did not lose the atoms the exact router would have kept.
    /// A run with `recall = 1.0` licenses the proposal for that regime; misses (and
    /// their reason — [`MissReason::NotGathered`] = an LSH recall miss, widen tables
    /// / probes; [`MissReason::TruncatedByBudget`] = widen the budget) name exactly
    /// where and why to widen.
    ///
    /// `directions` is one query direction per row (length `sketch.output_dim()`).
    /// `top_s` is the sparse routing width `s`. The returned
    /// [`ProposalRecallReport`] also carries the mean gathered-candidate count as
    /// the sublinearity witness (compare against `num_atoms`).
    pub fn proposal_recall_report<S: AtomFrameSketch>(
        &self,
        sketch: &S,
        directions: &[Array1<f64>],
        top_s: usize,
        candidate_budget: usize,
        multiprobe: bool,
    ) -> ProposalRecallReport {
        let mut total_true: usize = 0;
        let mut total_recovered: usize = 0;
        let mut total_gathered: usize = 0;
        let mut misses: Vec<RecallMiss> = Vec::new();

        for (row_idx, direction) in directions.iter().enumerate() {
            // The exact rescore top-s over the WHOLE dictionary — the reference.
            let exact = brute_force_top_s(sketch, direction.view(), top_s);
            // The sublinear two-stage proposal for the same row.
            let proposal = self.propose(sketch, direction.view(), candidate_budget, multiprobe);
            total_gathered += proposal.gathered_count;
            let proposed_set: HashSet<usize> = proposal.proposed.iter().copied().collect();
            let dropped_set: HashSet<usize> = proposal.dropped_for_budget.iter().copied().collect();

            for &atom in &exact {
                total_true += 1;
                if proposed_set.contains(&atom) {
                    total_recovered += 1;
                } else {
                    // A true-top-s atom the proposal dropped: gathered-but-budgeted
                    // (widen C) vs never-gathered (an LSH miss). Both logged.
                    let reason = if dropped_set.contains(&atom) {
                        MissReason::TruncatedByBudget
                    } else {
                        MissReason::NotGathered
                    };
                    misses.push(RecallMiss {
                        row: row_idx,
                        atom,
                        alignment: sketch.alignment(atom, direction.view()),
                        reason,
                    });
                }
            }
        }

        let recall = if total_true == 0 {
            1.0
        } else {
            total_recovered as f64 / total_true as f64
        };
        let avg_gathered = if directions.is_empty() {
            0.0
        } else {
            total_gathered as f64 / directions.len() as f64
        };

        ProposalRecallReport {
            candidate_budget,
            top_s,
            num_rows: directions.len(),
            total_true,
            total_recovered,
            recall,
            avg_candidates_gathered: avg_gathered,
            num_atoms: self.num_atoms,
            misses,
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

/// EXACT top-`s` reference: the `s` atoms of GREATEST frame alignment with
/// `direction` over the WHOLE dictionary (brute force, `O(K·p)`). This is the
/// ground truth the sublinear two-stage proposal's recall is licensed against —
/// the "true top-s" of the E1 acceptance (#985): the atoms the exact rescore
/// router would select for a row. Ties break to the LOWEST id, matching
/// [`SaeCandidateIndex::propose`]'s id-ascending tie-break so the two rank
/// identically wherever alignments coincide; non-finite alignments are skipped.
/// Returns up to `s` atom ids in descending-alignment order (fewer only when the
/// dictionary has fewer than `s` finite-scoring atoms).
pub fn brute_force_top_s<S: AtomFrameSketch>(
    sketch: &S,
    direction: ArrayView1<f64>,
    s: usize,
) -> Vec<usize> {
    let mut scored: Vec<(usize, f64)> = (0..sketch.num_atoms())
        .filter_map(|id| {
            let a = sketch.alignment(id, direction);
            a.is_finite().then_some((id, a))
        })
        .collect();
    // Descending by alignment; ties broken by id — identical policy to `propose`.
    scored.sort_by(|x, y| {
        y.1.partial_cmp(&x.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(x.0.cmp(&y.0))
    });
    scored.into_iter().take(s).map(|(id, _)| id).collect()
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

/// Result of `SaeCandidateIndex::recall_report`.
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

impl RecallReport {
    /// Convenience: ratio of mean gathered candidates to dictionary size. A
    /// value far below `1.0` is the evidence that proposal touched a sublinear
    /// slice of the dictionary.
    pub fn sublinearity_ratio(&self) -> f64 {
        if self.num_atoms == 0 {
            0.0
        } else {
            self.avg_candidates_gathered / self.num_atoms as f64
        }
    }
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
    /// against `num_atoms` (see `ProposalRecallReport::sublinearity_ratio`).
    pub avg_candidates_gathered: f64,
    /// Total atoms in the index (for the sublinearity ratio).
    pub num_atoms: usize,
    /// Every miss (true-top-s atom the proposal dropped), with row, atom,
    /// alignment, and reason. No silent drops — the license's honesty contract.
    pub misses: Vec<RecallMiss>,
}

impl ProposalRecallReport {
    /// Ratio of mean gathered candidates to dictionary size. Far below `1.0` is the
    /// evidence the proposal touched a sublinear slice of the dictionary — the
    /// `O(C)` vs `O(K)` witness that pairs with `recall`.
    pub fn sublinearity_ratio(&self) -> f64 {
        if self.num_atoms == 0 {
            0.0
        } else {
            self.avg_candidates_gathered / self.num_atoms as f64
        }
    }
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

/// Modified Gram–Schmidt orthonormalization of a decoder block's columns.
/// Input `block` is `(p, m)`; output `U` is `(p, r)` with orthonormal columns
/// spanning `range(block)`, `r ≤ m` (rank-deficient columns are dropped).
fn orthonormal_frame(block: &Array2<f64>) -> Array2<f64> {
    let p = block.nrows();
    let m = block.ncols();
    let mut cols: Vec<Array1<f64>> = Vec::with_capacity(m);
    for j in 0..m {
        let mut v = block.column(j).to_owned();
        let entering = vec_norm(v.view());
        for q in &cols {
            let proj: f64 = q.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum();
            for (vi, &qi) in v.iter_mut().zip(q.iter()) {
                *vi -= proj * qi;
            }
        }
        let nrm = vec_norm(v.view());
        // A residual inside the rounding band of the `p`-term projections it went
        // through is an artefact of the columns already kept, not a new direction.
        let band = gam_linalg::roundoff::accumulation_growth(p * (cols.len() + 1)) * entering;
        if nrm > band {
            for vi in v.iter_mut() {
                *vi /= nrm;
            }
            cols.push(v);
        }
    }
    let r = cols.len();
    let mut u = Array2::<f64>::zeros((p, r));
    for (j, col) in cols.into_iter().enumerate() {
        u.column_mut(j).assign(&col);
    }
    u
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
    if n > 0.0 {
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

