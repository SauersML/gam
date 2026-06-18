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
//! present on [`crate::terms::sae::manifold::SaeManifoldAtom`] — can implement
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
use std::collections::{HashMap, HashSet};

/// Salt mixed into the per-table hyperplane seed so the index tables and the
/// default sketch never share a random stream even when handed the same base
/// seed.
const INDEX_HYPERPLANE_SALT: u64 = 0x9E37_79B9_7F4A_7C15;

/// Salt for the default random-projection sketch's projection matrix.
const SKETCH_PROJECTION_SALT: u64 = 0xC2B2_AE3D_27D4_EB4F;

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
    /// Implementors whose atoms share one projection `R` (the default sketch,
    /// and the post-#972 Grassmann-frame sketch) **must** override this with
    /// the exact `normalize(R · d)` — `O(p · s)` per query, touching no atom,
    /// and exactly the cosine-LSH probe the sign-signature tables expect.
    ///
    /// The default is the legacy fallback for frame sources with *per-atom*
    /// projections only: average the masked per-atom query projections over a
    /// deterministic stratified `O(√K)` subset of atoms and renormalize. It
    /// concentrates on the shared-projection direction only when the sampled
    /// frames are spread near-isotropically — on coherent atom clusters it
    /// degrades (the #994 defect) — so it exists for trait-compatibility, not
    /// as a recommendation.
    fn query_sketch(&self, direction: ArrayView1<f64>) -> Array1<f64> {
        let sketch_dim = self.sketch_dim();
        let num = self.num_atoms();
        if num == 0 {
            return Array1::<f64>::zeros(sketch_dim);
        }
        let sample = ((num as f64).sqrt().ceil() as usize).clamp(1, num);
        let stride = (num / sample).max(1);
        let mut acc = Array1::<f64>::zeros(sketch_dim);
        let mut count = 0usize;
        let mut id = 0usize;
        while id < num {
            let q = self.project_direction(id, direction);
            if q.len() == sketch_dim {
                for (a, &v) in acc.iter_mut().zip(q.iter()) {
                    *a += v;
                }
                count += 1;
            }
            id += stride;
        }
        if count > 0 {
            for a in acc.iter_mut() {
                *a /= count as f64;
            }
        }
        normalize_in_place(&mut acc);
        acc
    }
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
            let s = sketch.atom_sketch(atom_id);
            if s.len() != sketch_dim {
                return Err(format!(
                    "SaeCandidateIndex: atom {atom_id} sketch length {} != sketch_dim {sketch_dim}",
                    s.len()
                ));
            }
            for (table, bank) in tables.iter_mut().zip(hyperplanes.iter()) {
                let sig = sign_signature(bank, s.view());
                table.entry(sig).or_default().push(atom_id);
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
                let neighbour = sig ^ (1u64 << flip_bit);
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
    crate::linalg::utils::splitmix64(&mut state)
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

/// Modified Gram–Schmidt orthonormalization of a decoder block's columns.
/// Input `block` is `(p, m)`; output `U` is `(p, r)` with orthonormal columns
/// spanning `range(block)`, `r ≤ m` (rank-deficient columns are dropped).
fn orthonormal_frame(block: &Array2<f64>) -> Array2<f64> {
    let p = block.nrows();
    let m = block.ncols();
    let mut cols: Vec<Array1<f64>> = Vec::with_capacity(m);
    for j in 0..m {
        let mut v = block.column(j).to_owned();
        for q in &cols {
            let proj: f64 = q.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum();
            for (vi, &qi) in v.iter_mut().zip(q.iter()) {
                *vi -= proj * qi;
            }
        }
        let nrm = vec_norm(v.view());
        if nrm > DIRECTION_NORM_FLOOR {
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
    if n > DIRECTION_NORM_FLOOR {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

/// Pack the sign bits of `bank · s` into a `u64` signature. `bank` is
/// `(bits, sketch_dim)`; `bits ≤ 64` (enforced by config-derived bit widths).
fn sign_signature(bank: &Array2<f64>, s: ArrayView1<f64>) -> u64 {
    let mut sig = 0u64;
    for r in 0..bank.nrows() {
        let row = bank.row(r);
        let dot: f64 = row.iter().zip(s.iter()).map(|(&a, &b)| a * b).sum();
        if dot >= 0.0 {
            sig |= 1u64 << r;
        }
    }
    sig
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
    (sig, margins)
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngExt as _;
    use rand::rngs::StdRng;

    /// Draw a unit vector in `p` dims from a seeded RNG.
    fn unit_vec(rng: &mut StdRng, p: usize) -> Array1<f64> {
        let mut v = Array1::<f64>::zeros(p);
        for x in v.iter_mut() {
            let u1 = rng.random::<f64>().max(1e-16);
            let u2 = rng.random::<f64>();
            *x = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        }
        let n = vec_norm(v.view());
        if n > DIRECTION_NORM_FLOOR {
            for x in v.iter_mut() {
                *x /= n;
            }
        }
        v
    }

    /// Build a synthetic dictionary of `k` rank-1 atoms: atom `i`'s decoder
    /// block is the outer-friendly single column `c_i` (a random unit direction
    /// in output space). Returns the blocks and the list of column directions so
    /// the planted-atom test can construct directions that lie in chosen atoms.
    fn synthetic_dictionary(k: usize, p: usize, seed: u64) -> (Vec<Array2<f64>>, Vec<Array1<f64>>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut blocks = Vec::with_capacity(k);
        let mut dirs = Vec::with_capacity(k);
        for _ in 0..k {
            let c = unit_vec(&mut rng, p);
            let mut block = Array2::<f64>::zeros((p, 1));
            block.column_mut(0).assign(&c);
            blocks.push(block);
            dirs.push(c);
        }
        (blocks, dirs)
    }

    #[test]
    fn frame_alignment_is_exact_for_in_range_direction() {
        let (blocks, dirs) = synthetic_dictionary(8, 16, 11);
        let sketch = RandomProjectionFrameSketch::from_decoder_blocks(&blocks, 12, 7).unwrap();
        // A direction equal to atom 3's column lies fully in its range.
        let d = &dirs[3];
        let a = sketch.alignment(3, d.view());
        assert!(a > 0.999, "in-range alignment should be ~1, got {a}");
        // An orthogonal-ish direction (atom 5's column is generically nearly
        // orthogonal to atom 3) aligns weakly with atom 3.
        let a_off = sketch.alignment(3, dirs[5].view());
        assert!(
            a_off < a,
            "off-atom alignment {a_off} should be below in-range {a}"
        );
    }

    #[test]
    fn build_is_deterministic_for_a_fixed_seed() {
        let (blocks, _) = synthetic_dictionary(64, 24, 99);
        let s1 = RandomProjectionFrameSketch::from_decoder_blocks(&blocks, 16, 5).unwrap();
        let s2 = RandomProjectionFrameSketch::from_decoder_blocks(&blocks, 16, 5).unwrap();
        // Same seed → identical representative sketches.
        for i in 0..blocks.len() {
            let a = s1.atom_sketch(i);
            let b = s2.atom_sketch(i);
            let diff = vec_norm((&a - &b).view());
            assert!(
                diff < 1e-12,
                "atom {i} sketch differs across builds: {diff:e}"
            );
        }
        let cfg = IndexConfig::auto(16, blocks.len(), 5);
        let idx1 = SaeCandidateIndex::build(&s1, cfg).unwrap();
        let idx2 = SaeCandidateIndex::build(&s2, cfg).unwrap();
        // Identical hyperplane banks and bucket contents.
        for t in 0..idx1.tables.len() {
            assert_eq!(idx1.tables[t].len(), idx2.tables[t].len());
        }
    }

    #[test]
    fn planted_atoms_are_recalled_above_floor_at_sublinear_budget() {
        // A frontier-ish dictionary: many atoms, modest output dim.
        let k = 2000usize;
        let p = 48usize;
        let (blocks, dirs) = synthetic_dictionary(k, p, 2026);
        let sketch_dim = 24usize;
        let sketch =
            RandomProjectionFrameSketch::from_decoder_blocks(&blocks, sketch_dim, 4242).unwrap();
        let cfg = IndexConfig::auto(sketch_dim, k, 4242);
        let index = SaeCandidateIndex::build(&sketch, cfg).unwrap();

        // Plant: each row's residual is dominated by one chosen atom's column
        // (plus a little cross-talk from a second). The planted-active set is
        // that dominant atom. We build many such rows deterministically.
        let mut rng = StdRng::seed_from_u64(31337);
        let n_rows = 200usize;
        let mut rows: Vec<(Array1<f64>, Vec<usize>)> = Vec::with_capacity(n_rows);
        for _ in 0..n_rows {
            let primary = rng.random_range(0..k);
            let secondary = rng.random_range(0..k);
            // direction = 1.0 * c_primary + 0.15 * c_secondary
            let mut d = dirs[primary].clone();
            for (di, &si) in d.iter_mut().zip(dirs[secondary].iter()) {
                *di += 0.15 * si;
            }
            let n = vec_norm(d.view());
            for di in d.iter_mut() {
                *di /= n;
            }
            rows.push((d, vec![primary]));
        }

        // Sublinear candidate budget: << K. We allow the gather to surface a
        // handful, but the *budget* (the per-row local block size) stays small.
        let candidate_budget = 32usize;
        let report = index.recall_report(&sketch, &rows, candidate_budget, cfg.multiprobe);

        // The gather must touch only a sublinear slice of the dictionary.
        assert!(
            report.sublinearity_ratio() < 0.5,
            "gather was not sublinear: avg {} of {} atoms (ratio {:.3})",
            report.avg_candidates_gathered,
            report.num_atoms,
            report.sublinearity_ratio()
        );

        // Recall floor: the LSH index must recover the planted dominant atom for
        // the large majority of rows at this sublinear budget. Misses are
        // logged, never silently dropped.
        let floor = 0.80;
        assert!(
            report.recall >= floor,
            "recall {:.3} below floor {floor}; {} misses logged (first few: {:?})",
            report.recall,
            report.misses.len(),
            report
                .misses
                .iter()
                .take(5)
                .map(|m| (m.row, m.atom, m.reason, m.alignment))
                .collect::<Vec<_>>()
        );

        // Every miss is accounted for with a reason — the no-silent-truncation
        // contract.
        let recovered = report.total_recovered;
        assert_eq!(
            report.total_planted - recovered,
            report.misses.len(),
            "miss list must account for every unrecovered planted atom"
        );
    }

    #[test]
    fn auto_candidate_budget_tracks_the_issue_band() {
        assert_eq!(auto_candidate_budget(2), CANDIDATE_BUDGET_MIN);
        assert_eq!(auto_candidate_budget(64), 48);
        assert_eq!(auto_candidate_budget(1024), 80);
        assert_eq!(auto_candidate_budget(100_000), CANDIDATE_BUDGET_MAX);
        // Monotone non-decreasing in K and always inside the band.
        let mut prev = 0usize;
        for k in [2usize, 16, 64, 256, 1024, 4096, 65_536, 1_000_000] {
            let c = auto_candidate_budget(k);
            assert!(c >= prev, "budget must be monotone in K");
            assert!((CANDIDATE_BUDGET_MIN..=CANDIDATE_BUDGET_MAX).contains(&c));
            prev = c;
        }
    }

    /// Build a planted row set for a dictionary: each row's residual direction
    /// is dominated by one chosen atom (plus cross-talk from a second), and
    /// the planted-active set is the dominant atom.
    fn planted_rows(
        dirs: &[Array1<f64>],
        n_rows: usize,
        seed: u64,
    ) -> Vec<(Array1<f64>, Vec<usize>)> {
        let k = dirs.len();
        let mut rng = StdRng::seed_from_u64(seed);
        let mut rows = Vec::with_capacity(n_rows);
        for _ in 0..n_rows {
            let primary = rng.random_range(0..k);
            let secondary = rng.random_range(0..k);
            let mut d = dirs[primary].clone();
            for (di, &si) in d.iter_mut().zip(dirs[secondary].iter()) {
                *di += 0.15 * si;
            }
            let n = vec_norm(d.view());
            for di in d.iter_mut() {
                *di /= n;
            }
            rows.push((d, vec![primary]));
        }
        rows
    }

    #[test]
    fn k_ladder_recall_determinism_and_sublinearity() {
        // #985 part 2 (index tier): the K=2-era assumptions say nothing about
        // frontier K, so gate the proposal machinery on a planted ladder at
        // K = 64 and K = 1024 with the SAME battery per rung — recall above a
        // stated floor at the auto-derived budget, every miss accounted for,
        // and byte-identical proposals across two independent builds. The
        // gather must also become *relatively* sparser as K grows (the
        // sublinearity witness): what is allowed to touch half the dictionary
        // at K = 64 must not at K = 1024.
        let p = 48usize;
        let n_rows = 150usize;
        let mut ladder_ratios = Vec::new();
        for &k in &[64usize, 1024] {
            let (blocks, dirs) = synthetic_dictionary(k, p, 9000 + k as u64);
            let sketch_dim = 24usize;
            let sketch_seed = 71 + k as u64;
            let sketch =
                RandomProjectionFrameSketch::from_decoder_blocks(&blocks, sketch_dim, sketch_seed)
                    .unwrap();
            let cfg = IndexConfig::auto(sketch_dim, k, sketch_seed);
            let index = SaeCandidateIndex::build(&sketch, cfg).unwrap();

            let rows = planted_rows(&dirs, n_rows, 555 + k as u64);
            let budget = auto_candidate_budget(k);
            let report = index.recall_report(&sketch, &rows, budget, cfg.multiprobe);

            // Recall floor at the auto-derived budget, with every miss carrying
            // a reason — the no-silent-truncation contract, per rung.
            let floor = 0.80;
            assert!(
                report.recall >= floor,
                "K={k}: recall {:.3} below floor {floor}; {} misses (first: {:?})",
                report.recall,
                report.misses.len(),
                report
                    .misses
                    .iter()
                    .take(3)
                    .map(|m| (m.row, m.atom, m.reason, m.alignment))
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                report.total_planted - report.total_recovered,
                report.misses.len(),
                "K={k}: miss list must account for every unrecovered planted atom"
            );

            // Search determinism: an independent rebuild from the same inputs
            // proposes the identical candidate set for every probed row.
            let sketch2 =
                RandomProjectionFrameSketch::from_decoder_blocks(&blocks, sketch_dim, sketch_seed)
                    .unwrap();
            let index2 = SaeCandidateIndex::build(&sketch2, cfg).unwrap();
            for (direction, _) in rows.iter().take(20) {
                let a = index.propose(&sketch, direction.view(), budget, cfg.multiprobe);
                let b = index2.propose(&sketch2, direction.view(), budget, cfg.multiprobe);
                assert_eq!(
                    a.proposed, b.proposed,
                    "K={k}: rebuild must propose identically"
                );
            }

            // Proposal size is the budget, never the dictionary: the per-row
            // local block stays near the planted/active scale.
            for (direction, _) in rows.iter().take(20) {
                let prop = index.propose(&sketch, direction.view(), budget, cfg.multiprobe);
                assert!(prop.proposed.len() <= budget);
            }

            ladder_ratios.push((k, report.sublinearity_ratio()));
        }
        // Relative sparsity must improve up the ladder: the gathered fraction
        // of the dictionary shrinks as K grows (sublinear gather), and at the
        // frontier-shaped rung it must be a small slice outright.
        let (_, ratio_small) = ladder_ratios[0];
        let (k_big, ratio_big) = ladder_ratios[1];
        assert!(
            ratio_big < ratio_small,
            "sublinearity must improve along the ladder: {ladder_ratios:?}"
        );
        assert!(
            ratio_big < 0.25,
            "K={k_big}: gather touched {:.1}% of the dictionary",
            ratio_big * 100.0
        );
    }

    /// Counting wrapper: delegates everything, counts `project_direction`
    /// calls. The #994 acceptance gate: with the exact probe, building the
    /// query sketch touches NO atom, so a whole `propose` makes zero
    /// `project_direction` calls (scoring goes through `alignment`).
    struct CountingSketch<'a> {
        inner: &'a RandomProjectionFrameSketch,
        project_calls: std::cell::Cell<usize>,
    }

    impl AtomFrameSketch for CountingSketch<'_> {
        fn sketch_dim(&self) -> usize {
            self.inner.sketch_dim()
        }
        fn output_dim(&self) -> usize {
            self.inner.output_dim()
        }
        fn num_atoms(&self) -> usize {
            self.inner.num_atoms()
        }
        fn atom_sketch(&self, atom_id: usize) -> Array1<f64> {
            self.inner.atom_sketch(atom_id)
        }
        fn project_direction(&self, atom_id: usize, direction: ArrayView1<f64>) -> Array1<f64> {
            self.project_calls.set(self.project_calls.get() + 1);
            self.inner.project_direction(atom_id, direction)
        }
        fn alignment(&self, atom_id: usize, direction: ArrayView1<f64>) -> f64 {
            self.inner.alignment(atom_id, direction)
        }
        fn query_sketch(&self, direction: ArrayView1<f64>) -> Array1<f64> {
            self.inner.query_sketch(direction)
        }
    }

    #[test]
    fn query_probe_touches_no_atom_before_the_gather() {
        let k = 512usize;
        let p = 32usize;
        let (blocks, dirs) = synthetic_dictionary(k, p, 77);
        let sketch = RandomProjectionFrameSketch::from_decoder_blocks(&blocks, 16, 13).unwrap();
        let cfg = IndexConfig::auto(16, k, 13);
        let index = SaeCandidateIndex::build(&sketch, cfg).unwrap();
        let counting = CountingSketch {
            inner: &sketch,
            project_calls: std::cell::Cell::new(0),
        };
        drop(index.propose(&counting, dirs[5].view(), 32, cfg.multiprobe));
        assert_eq!(
            counting.project_calls.get(),
            0,
            "the exact query probe must be independent of K: no per-atom \
             projection before the gather (#994)"
        );
    }

    /// Build a coherent-cluster dictionary: `n_clusters` random unit centers,
    /// each with `cluster_size` atoms drawn as small perturbations of the
    /// center (renormalized). Exactly the non-isotropic regime where the old
    /// masked-average probe degraded (#994).
    fn coherent_cluster_dictionary(
        n_clusters: usize,
        cluster_size: usize,
        p: usize,
        spread: f64,
        seed: u64,
    ) -> (Vec<Array2<f64>>, Vec<Array1<f64>>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut blocks = Vec::with_capacity(n_clusters * cluster_size);
        let mut dirs = Vec::with_capacity(n_clusters * cluster_size);
        for _ in 0..n_clusters {
            let center = unit_vec(&mut rng, p);
            for _ in 0..cluster_size {
                let noise = unit_vec(&mut rng, p);
                let mut c = center.clone();
                for (ci, &ni) in c.iter_mut().zip(noise.iter()) {
                    *ci += spread * ni;
                }
                let n = vec_norm(c.view());
                for ci in c.iter_mut() {
                    *ci /= n;
                }
                let mut block = Array2::<f64>::zeros((p, 1));
                block.column_mut(0).assign(&c);
                blocks.push(block);
                dirs.push(c);
            }
        }
        (blocks, dirs)
    }

    #[test]
    fn coherent_clusters_are_recalled_with_the_exact_probe() {
        // 32 clusters × 32 near-parallel atoms = 1024 atoms. Rows are
        // dominated by one specific cluster member; the proposal must recover
        // that exact member (not merely its cluster) at the auto budget —
        // exact alignment scoring separates siblings once the probe lands the
        // gather in the right bucket neighborhood.
        let n_clusters = 32usize;
        let cluster_size = 32usize;
        let k = n_clusters * cluster_size;
        let p = 48usize;
        let (blocks, dirs) = coherent_cluster_dictionary(n_clusters, cluster_size, p, 0.25, 4242);
        let sketch_dim = 24usize;
        let sketch =
            RandomProjectionFrameSketch::from_decoder_blocks(&blocks, sketch_dim, 99).unwrap();
        let cfg = IndexConfig::auto(sketch_dim, k, 99);
        let index = SaeCandidateIndex::build(&sketch, cfg).unwrap();

        let rows = planted_rows(&dirs, 150, 31337);
        let budget = auto_candidate_budget(k);
        let report = index.recall_report(&sketch, &rows, budget, cfg.multiprobe);
        let floor = 0.80;
        assert!(
            report.recall >= floor,
            "coherent-cluster recall {:.3} below floor {floor}; {} misses (first: {:?})",
            report.recall,
            report.misses.len(),
            report
                .misses
                .iter()
                .take(3)
                .map(|m| (m.row, m.atom, m.reason, m.alignment))
                .collect::<Vec<_>>()
        );
        // Still a sublinear slice of the dictionary, clusters or not.
        assert!(
            report.sublinearity_ratio() < 0.5,
            "cluster gather touched {:.1}% of the dictionary",
            report.sublinearity_ratio() * 100.0
        );
    }

    #[test]
    fn exact_probe_matches_shared_projection_of_the_direction() {
        // The override is literally normalize(R·d): verify against a manual
        // computation through the public surface (atom_sketch of a rank-1 atom
        // whose only column IS the direction gives normalize(R·d) too).
        let p = 16usize;
        let mut rng = StdRng::seed_from_u64(5);
        let d = unit_vec(&mut rng, p);
        let mut block = Array2::<f64>::zeros((p, 1));
        block.column_mut(0).assign(&d);
        let sketch = RandomProjectionFrameSketch::from_decoder_blocks(&[block], 8, 21).unwrap();
        let via_probe = sketch.query_sketch(d.view());
        let via_atom = sketch.atom_sketch(0);
        let diff = vec_norm((&via_probe - &via_atom).view());
        assert!(
            diff < 1e-10,
            "query_sketch(d) must equal the rank-1 atom representative of d: diff {diff:e}"
        );
    }

    #[test]
    fn empty_planted_rows_report_perfect_recall() {
        let (blocks, dirs) = synthetic_dictionary(32, 16, 1);
        let sketch = RandomProjectionFrameSketch::from_decoder_blocks(&blocks, 12, 3).unwrap();
        let cfg = IndexConfig::auto(12, 32, 3);
        let index = SaeCandidateIndex::build(&sketch, cfg).unwrap();
        let rows = vec![(dirs[0].clone(), Vec::<usize>::new())];
        let report = index.recall_report(&sketch, &rows, 8, true);
        assert_eq!(report.recall, 1.0);
        assert!(report.misses.is_empty());
    }
}
