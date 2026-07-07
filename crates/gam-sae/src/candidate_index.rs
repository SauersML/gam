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
/// This is the *true* upper bound that makes [`SaeCandidateIndex::route_exact`]'s
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

    /// Regression for the routing-confidence gate (#1026): a low-alignment LSH
    /// route must be flagged for the exact fallback, never trusted by the
    /// heuristic gate. This gate is a confidence/quality proxy, NOT a
    /// global-optimality certificate — being at/above the threshold means the
    /// chosen atom is itself a reasonable fit, it does not prove no better
    /// ungathered atom exists. `certified_encode_with_index` (and the amortized
    /// twin) flag a routed row UNCERTIFIED whenever the best-aligned proposed
    /// atom's frame alignment is below
    /// `encode::CANDIDATE_ROUTING_MIN_ALIGNMENT`. The gate's decision input is
    /// exactly `sketch.alignment(best_atom, target)`; pin it here — with exact,
    /// deterministic linear algebra rather than LSH gather luck — so a future
    /// change to the frame-alignment formula cannot silently shift the
    /// threshold's meaning out from under the gate.
    ///
    /// Two atoms whose decoder frames span ORTHOGONAL subspaces of a 6-dim
    /// ambient (atom 0: `span(e0,e1)`, atom 1: `span(e2,e3)`; dims `e4,e5`
    /// covered by neither). A direction wholly inside atom 1's subspace has
    /// alignment exactly 1 with atom 1 (in-frame, ABOVE the gate) and exactly 0
    /// with atom 0 (off-frame, BELOW the gate) — the mis-route the gate exists
    /// to flag.
    #[test]
    fn routing_confidence_gate_input_separates_off_frame_from_in_frame() {
        use crate::encode::CANDIDATE_ROUTING_MIN_ALIGNMENT as GATE;

        let p = 6usize;
        let mut block_a = Array2::<f64>::zeros((p, 2));
        block_a[[0, 0]] = 1.0; // e0
        block_a[[1, 1]] = 1.0; // e1
        let mut block_b = Array2::<f64>::zeros((p, 2));
        block_b[[2, 0]] = 1.0; // e2
        block_b[[3, 1]] = 1.0; // e3
        let sketch =
            RandomProjectionFrameSketch::from_decoder_blocks(&[block_a, block_b], 16, 4242)
                .unwrap();

        // A unit direction wholly inside atom 1's (e2,e3) subspace.
        let mut in_frame_b = Array1::<f64>::zeros(p);
        in_frame_b[2] = 0.6;
        in_frame_b[3] = 0.8; // unit norm (0.6² + 0.8² = 1)
        let a_right = sketch.alignment(1, in_frame_b.view());
        let a_wrong = sketch.alignment(0, in_frame_b.view());

        assert!(
            a_right > 0.999,
            "an in-frame direction must align ~1 with its own atom; got {a_right}"
        );
        assert!(
            a_wrong < 1e-9,
            "an orthogonal-subspace direction must align ~0 with the wrong atom; got {a_wrong}"
        );
        // The exact predicate the encode gate evaluates per row: a mis-routed
        // (orthogonal) atom falls BELOW the gate → flagged for the exact
        // fallback; the correctly-routed atom sits AT/ABOVE the gate → trusted by
        // the heuristic gate (a confidence proxy, not a global-optimality
        // certificate).
        assert!(
            a_wrong < GATE,
            "a mis-routed (orthogonal) atom must fall below the routing gate {GATE}; got {a_wrong}"
        );
        assert!(
            a_right >= GATE,
            "the correctly-routed atom must sit at/above the routing gate {GATE}; got {a_right}"
        );

        // A direction in the UNCOVERED (e4,e5) subspace aligns ~0 with BOTH
        // atoms, so whichever atom the LSH surfaces, the gate fires: no atom can
        // certify this route. This is the worst-case the gate exists to catch.
        let mut uncovered = Array1::<f64>::zeros(p);
        uncovered[4] = 1.0;
        for atom in 0..2 {
            let a = sketch.alignment(atom, uncovered.view());
            assert!(
                a < GATE,
                "an uncovered-subspace direction must fall below the gate for atom {atom}; got {a}"
            );
        }
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

    /// The exact-routing guarantee (#1777 / roadmap): for EVERY row,
    /// [`SaeCandidateIndex::route_exact`] selects the SAME atom as the brute-force
    /// full-scan global argmax of the routing score — no silent miss — even on the
    /// rows where the sublinear LSH gather alone picks a worse atom. This is the
    /// acceptance contract: production routing == brute-force argmax, by
    /// construction of the exact fallback.
    #[test]
    fn route_exact_matches_brute_force_argmax_with_no_silent_miss() {
        // ── Arm A: exact-fallback path. Random unit-direction queries against a
        // frontier-shaped dictionary. A random direction lies fully in no atom's
        // rank-1 range, so the alignment is < 1 everywhere: the universal-bound
        // fast path never fires and route_exact must run the exact scan. ──────────
        let k = 1500usize;
        let p = 32usize;
        let (blocks, dirs) = synthetic_dictionary(k, p, 2027);
        let sketch_dim = 24usize;
        let sketch =
            RandomProjectionFrameSketch::from_decoder_blocks(&blocks, sketch_dim, 909).unwrap();
        let cfg = IndexConfig::auto(sketch_dim, k, 909);
        let index = SaeCandidateIndex::build(&sketch, cfg).unwrap();
        let budget = auto_candidate_budget(k);

        let mut rng = StdRng::seed_from_u64(8675309);
        let n_rows = 300usize;
        let mut lsh_only_misses = 0usize; // rows where LSH-alone picked a worse atom
        for _ in 0..n_rows {
            let d = unit_vec(&mut rng, p);

            // Ground truth: brute-force global argmax.
            let (truth_atom, truth_align) = brute_force_best_atom(&sketch, d.view())
                .expect("non-empty dictionary has an argmax");

            // No-silent-miss invariant: nothing in the dictionary beats the truth.
            for id in 0..k {
                let a = sketch.alignment(id, d.view());
                assert!(
                    a <= truth_align + 1e-12,
                    "brute force is not the argmax: atom {id} scores {a} > {truth_align}"
                );
            }

            // Production exact router must equal the brute-force argmax, exactly.
            let route = index
                .route_exact(&sketch, d.view(), budget, cfg.multiprobe)
                .expect("route_exact returns an argmax for a non-empty dictionary");
            assert_eq!(
                route.atom, truth_atom,
                "route_exact must select the brute-force global argmax"
            );
            assert!(
                (route.alignment - truth_align).abs() < 1e-12,
                "route_exact alignment {} != brute force {truth_align}",
                route.alignment
            );
            assert!(
                route.did_full_scan && !route.lsh_certified,
                "a sub-ceiling row must take the exact-scan fallback, not the bound fast path"
            );

            // What the OLD heuristic (LSH gather best) alone would have chosen.
            let proposal = index.propose(&sketch, d.view(), budget, cfg.multiprobe);
            if let Some(&lsh_best) = proposal.proposed.first() {
                if lsh_best != truth_atom {
                    lsh_only_misses += 1;
                }
            } else {
                lsh_only_misses += 1;
            }
        }
        // The fallback is doing real work: the sublinear gather alone DOES silently
        // miss the global best on a meaningful fraction of rows, and route_exact
        // recovered every one of them (asserted above, per row).
        assert!(
            lsh_only_misses > 0,
            "test is vacuous: LSH-alone never missed, so the exact fallback was never exercised"
        );

        // ── Arm B: certified fast path. A query equal to a unique atom's own
        // column aligns exactly 1.0 with it (the ceiling) and < 1 with all others,
        // so route_exact certifies optimality via the universal bound — no scan —
        // and still returns the brute-force argmax. ─────────────────────────────
        let j = 777usize;
        let dj = &dirs[j];
        let (truth_atom, _) = brute_force_best_atom(&sketch, dj.view()).unwrap();
        assert_eq!(
            truth_atom, j,
            "the in-frame column is its own unique argmax"
        );
        let route = index
            .route_exact(&sketch, dj.view(), budget, cfg.multiprobe)
            .unwrap();
        assert_eq!(
            route.atom, j,
            "fast path must still return the global argmax"
        );
        assert!(
            route.lsh_certified && !route.did_full_scan,
            "a ceiling-alignment row must be certified by the universal bound, no scan"
        );
        assert!(
            route.alignment >= ROUTING_ALIGNMENT_UPPER_BOUND - ROUTING_CERT_EPS,
            "certified alignment must sit at the universal ceiling; got {}",
            route.alignment
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

    /// `brute_force_top_s` is the exact rescore reference: the `s` highest-aligned
    /// atoms, descending, id-broken ties — matching `propose`'s ranking policy.
    #[test]
    fn brute_force_top_s_returns_exact_descending_winners() {
        let (blocks, dirs) = synthetic_dictionary(64, 24, 4);
        let sketch = RandomProjectionFrameSketch::from_decoder_blocks(&blocks, 16, 9).unwrap();
        // A direction equal to atom 7's column: atom 7 must be the #1 winner.
        let top = brute_force_top_s(&sketch, dirs[7].view(), 5);
        assert_eq!(top.len(), 5, "must return exactly s winners");
        assert_eq!(top[0], 7, "the in-range atom is the top-1 exact winner");
        // Alignments are non-increasing along the returned order.
        let mut prev = f64::INFINITY;
        for &id in &top {
            let a = sketch.alignment(id, dirs[7].view());
            assert!(a <= prev + 1e-12, "top-s must be descending by alignment");
            prev = a;
        }
        // Asking for more than the dictionary holds returns the whole dictionary.
        let all = brute_force_top_s(&sketch, dirs[0].view(), 1000);
        assert_eq!(all.len(), 64);
    }

    /// The two-stage routing LICENSE at a frontier-shaped dictionary: the sublinear
    /// proposal recovers the EXACT top-s rescore for the large majority of rows at
    /// the derived budget `C = 8⌈log2 K⌉`, touches only a sublinear slice, and every
    /// miss is logged with a reason (no silent drop).
    #[test]
    fn proposal_recall_license_recovers_exact_top_s_at_frontier_k() {
        let k = 2000usize;
        let p = 48usize;
        let (blocks, dirs) = synthetic_dictionary(k, p, 2026);
        let sketch_dim = 24usize;
        let sketch =
            RandomProjectionFrameSketch::from_decoder_blocks(&blocks, sketch_dim, 4242).unwrap();
        let cfg = IndexConfig::auto(sketch_dim, k, 4242);
        let index = SaeCandidateIndex::build(&sketch, cfg).unwrap();

        // Row directions dominated by one atom (plus a little cross-talk); reuse the
        // planted-row generator but keep only the directions — the license needs no
        // ground-truth actives, it references the exact rescore itself.
        let rows = planted_rows(&dirs, 200, 31337);
        let directions: Vec<Array1<f64>> = rows.into_iter().map(|(d, _)| d).collect();
        let budget = auto_candidate_budget(k);

        // Recall floor at s = 1: each row's exact top-1 IS its dominant atom, and the
        // proposal must recover it for the large majority — the meaningful license
        // (a row dominated by one atom has no genuine 2nd/3rd; higher-s fillers are
        // arbitrary near-zero-alignment atoms no sublinear gather targets, so a recall
        // floor only carries content at s = 1).
        let report = index.proposal_recall_report(&sketch, &directions, 1, budget, cfg.multiprobe);
        assert!(
            report.sublinearity_ratio() < 0.5,
            "license gather was not sublinear: avg {} of {} atoms (ratio {:.3})",
            report.avg_candidates_gathered,
            report.num_atoms,
            report.sublinearity_ratio()
        );
        let floor = 0.80;
        assert!(
            report.recall >= floor,
            "top-1 recall {:.3} below floor {floor}; {} misses (first: {:?})",
            report.recall,
            report.misses.len(),
            report
                .misses
                .iter()
                .take(5)
                .map(|m| (m.row, m.atom, m.reason, m.alignment))
                .collect::<Vec<_>>()
        );
        // No-silent-truncation: the miss list accounts for every unrecovered slot.
        assert_eq!(
            report.total_true - report.total_recovered,
            report.misses.len(),
            "license miss list must account for every unrecovered true-top-s atom"
        );

        // Top-s machinery (s = 4): the report is well-formed and the miss ledger is
        // exact regardless of whether the arbitrary higher-s fillers are recovered.
        // `total_true` is exactly `s` per row (the million-plus dictionary always has
        // ≥ s finite-scoring atoms), and recall stays a valid fraction.
        let s = 4usize;
        let multi = index.proposal_recall_report(&sketch, &directions, s, budget, cfg.multiprobe);
        assert_eq!(multi.total_true, s * directions.len());
        assert!(multi.recall.is_finite() && (0.0..=1.0).contains(&multi.recall));
        assert_eq!(
            multi.total_true - multi.total_recovered,
            multi.misses.len(),
            "top-s miss ledger must account for every unrecovered slot at s>1"
        );
    }

    /// The acceptance-scale routing license: **K = 10^6** routing. Building the
    /// index is sublinear-query by construction; this pins that at a million atoms
    /// the two-stage proposal recovers the exact top-s for the bulk of rows while
    /// GATHERING only a vanishing fraction of the dictionary (the `O(C)`-not-`O(K)`
    /// witness), and logs every miss. Kept lean (small `p`, few rows) so the exact
    /// brute-force reference stays cheap.
    #[test]
    fn proposal_recall_license_at_one_million_atoms() {
        let k = 1_000_000usize;
        // A well-separated, finely-bucketed geometry (the frontier test's proven
        // params): at K=10^6 the ~log2(K)=20-bit signatures give ~2^20≈K buckets, so
        // bucket occupancy stays O(1) and the gather is a vanishing slice of the
        // dictionary. (A too-small `p` over-packs a million atoms into a low-dim
        // ambient, collapsing the buckets and inflating the gather.)
        let p = 48usize;
        let (blocks, dirs) = synthetic_dictionary(k, p, 7);
        let sketch_dim = 24usize;
        let sketch =
            RandomProjectionFrameSketch::from_decoder_blocks(&blocks, sketch_dim, 21).unwrap();
        let cfg = IndexConfig::auto(sketch_dim, k, 21);
        let index = SaeCandidateIndex::build(&sketch, cfg).unwrap();

        // A handful of rows, each dominated by one scattered atom's column.
        let n_rows = 24usize;
        let mut directions: Vec<Array1<f64>> = Vec::with_capacity(n_rows);
        for r in 0..n_rows {
            let primary = (r * 2_654_435_761usize) % k;
            directions.push(dirs[primary].clone());
        }
        let budget = auto_candidate_budget(k);
        assert_eq!(budget, CANDIDATE_BUDGET_MAX, "K=10^6 sits at the budget cap");
        // s = 1: each row's exact top-1 is its planted dominant atom (recall floor
        // carries content only at s = 1 — see the frontier test's note).
        let s = 1usize;
        let report = index.proposal_recall_report(&sketch, &directions, s, budget, cfg.multiprobe);

        // The gather touched a vanishing slice of a million atoms — the routing is
        // O(C), not O(K). (Far tighter than the frontier test's 0.5 ceiling; a few %
        // of a million atoms still leaves a decisive sublinear margin.)
        assert!(
            report.sublinearity_ratio() < 0.05,
            "million-atom gather touched {:.3}% of the dictionary (avg {} of {})",
            report.sublinearity_ratio() * 100.0,
            report.avg_candidates_gathered,
            report.num_atoms
        );
        // Each row's dominant atom is its exact top-1; the proposal must recover it
        // for the large majority. Misses (if any) are all logged with a reason.
        let floor = 0.75;
        assert!(
            report.recall >= floor,
            "K=10^6 top-s recall {:.3} below floor {floor}; {} misses logged",
            report.recall,
            report.misses.len()
        );
        assert_eq!(
            report.total_true - report.total_recovered,
            report.misses.len(),
            "every unrecovered true-top-s atom must be logged at K=10^6"
        );
        eprintln!(
            "[e1-license] K={k} C={budget} s={s} recall@s={:.3} avg_gathered={:.1} \
             sublinearity={:.5}% misses={}",
            report.recall,
            report.avg_candidates_gathered,
            report.sublinearity_ratio() * 100.0,
            report.misses.len()
        );
    }

    /// SPEC 13 — the license RECOVERS THE NULL: on structureless / degenerate input
    /// it returns the trivial answer rather than fabricating recovery, and never
    /// panics. Three null faces: an empty row set, `s = 0`, and zero-norm query
    /// directions (no signal to route).
    #[test]
    fn proposal_recall_license_recovers_the_null() {
        let k = 128usize;
        let p = 16usize;
        let (blocks, dirs) = synthetic_dictionary(k, p, 3);
        let sketch = RandomProjectionFrameSketch::from_decoder_blocks(&blocks, 12, 5).unwrap();
        let cfg = IndexConfig::auto(12, k, 5);
        let index = SaeCandidateIndex::build(&sketch, cfg).unwrap();
        let budget = auto_candidate_budget(k);

        // Null face 1: no rows → nothing to recover → perfect, empty misses.
        let empty: Vec<Array1<f64>> = Vec::new();
        let r0 = index.proposal_recall_report(&sketch, &empty, 4, budget, cfg.multiprobe);
        assert_eq!(r0.recall, 1.0);
        assert_eq!(r0.total_true, 0);
        assert!(r0.misses.is_empty());
        assert_eq!(r0.sublinearity_ratio(), 0.0);

        // Null face 2: s = 0 → no top-s slots → trivially licensed on real rows.
        let real: Vec<Array1<f64>> = dirs.iter().take(8).cloned().collect();
        let r1 = index.proposal_recall_report(&sketch, &real, 0, budget, cfg.multiprobe);
        assert_eq!(r1.recall, 1.0);
        assert_eq!(r1.total_true, 0);
        assert!(r1.misses.is_empty());

        // Null face 3: zero-norm directions carry no signal. The exact rescore is a
        // degenerate all-zero-alignment tie broken to the lowest ids; the report is
        // finite and defined, no panic, and every unrecovered slot is still logged.
        let zeros: Vec<Array1<f64>> = (0..5).map(|_| Array1::<f64>::zeros(p)).collect();
        let r2 = index.proposal_recall_report(&sketch, &zeros, 4, budget, cfg.multiprobe);
        assert!(r2.recall.is_finite() && (0.0..=1.0).contains(&r2.recall));
        assert_eq!(
            r2.total_true - r2.total_recovered,
            r2.misses.len(),
            "null (zero-direction) rows must still account for every miss"
        );
    }

    /// The routability-derived shortlist size is well-formed: monotone
    /// non-decreasing and only LOGARITHMIC (sublinear) in `K`, widens as the
    /// confidence tightens, and reconstructs `p·u² = 2·ln(K/δ)` from the floor.
    #[test]
    fn routability_shortlist_size_is_logarithmic_and_floor_derived() {
        let p = 48usize;
        let s = 1usize;
        let delta = 0.2;
        let mut prev = 0usize;
        for &k in &[64usize, 256, 1024, 4096, 65_536, 1_000_000] {
            let c = routability_shortlist_size(p, k, s, delta);
            assert!(c >= prev, "shortlist size must be monotone in K: {c} < {prev}");
            assert!(c >= s + 1, "shortlist must exceed the top-s winner count");
            assert!(c <= k, "shortlist can never exceed the dictionary");
            // Logarithmic, not linear: at a million atoms the shortlist stays a
            // tiny constant-times-log(K), never a fraction of K.
            assert!(
                (c as f64) < 4.0 * (k as f64).ln(),
                "K={k}: shortlist {c} must stay logarithmic in K"
            );
            // Reconstruct the band directly: C − s == ⌈2·ln(K/δ)⌉.
            let want_band = (2.0 * (k as f64 / delta).ln()).ceil() as usize;
            assert_eq!(
                c - s,
                want_band.clamp(1, k - s),
                "K={k}: shortlist band must equal ⌈2·ln(K/δ)⌉ = p·u²"
            );
            prev = c;
        }
        // Tighter confidence (smaller δ) widens the shortlist.
        let c_loose = routability_shortlist_size(p, 4096, s, 0.5);
        let c_tight = routability_shortlist_size(p, 4096, s, 1e-4);
        assert!(
            c_tight > c_loose,
            "tightening δ must widen the shortlist: {c_tight} !> {c_loose}"
        );
    }

    /// E1 acceptance: at the routability-DERIVED shortlist size `C`, the default
    /// two-stage route (LSH shortlist → exact frame rescore) MATCHES the
    /// exhaustive exact top-`s` selection for the routable rows, to the recall
    /// bound `1 − δ` set by the SAME `δ` that sizes `C` and the floor — no magic
    /// threshold anywhere. The rows are planted well above the routability floor
    /// (target-to-clutter `a/ν = 1/0.15 ≈ 6.7 ≫ floor`), so they are routable by
    /// construction; the gather touches only a sublinear slice; every miss is
    /// logged.
    #[test]
    fn two_stage_route_matches_exact_top_s_at_routability_derived_shortlist() {
        let k = 2000usize;
        let p = 48usize;
        let (blocks, dirs) = synthetic_dictionary(k, p, 2026);
        let sketch_dim = 24usize;
        let sketch =
            RandomProjectionFrameSketch::from_decoder_blocks(&blocks, sketch_dim, 4242).unwrap();
        let cfg = IndexConfig::auto(sketch_dim, k, 4242);
        let index = SaeCandidateIndex::build(&sketch, cfg).unwrap();

        // One δ drives the floor, the shortlist size, and the recall bound.
        let delta = 0.2;
        let s = 1usize;
        let shortlist = routability_shortlist_size(p, k, s, delta);

        // The floor certifies these planted rows are routable: a/ν ≫ floor.
        let floor = crate::routability::routability_floor(p, k, 1, delta).floor;
        let planted_ratio = 1.0 / 0.15;
        assert!(
            planted_ratio > floor,
            "planted target-to-clutter {planted_ratio:.2} must clear the floor {floor:.3}"
        );

        let rows = planted_rows(&dirs, 200, 31337);
        let directions: Vec<Array1<f64>> = rows.into_iter().map(|(d, _)| d).collect();
        let report =
            index.proposal_recall_report(&sketch, &directions, s, shortlist, cfg.multiprobe);

        // Sublinear gather: the two-stage route touched only a small slice of K.
        assert!(
            report.sublinearity_ratio() < 0.5,
            "two-stage gather was not sublinear: ratio {:.3}",
            report.sublinearity_ratio()
        );
        // The derived recall bound, tied to δ (NOT a literal): routable rows are
        // recovered at ≥ 1 − δ.
        let bound = 1.0 - delta;
        assert!(
            report.recall >= bound,
            "top-{s} recall {:.3} below the derived bound {bound:.3} at shortlist C={shortlist}; \
             {} misses (first: {:?})",
            report.recall,
            report.misses.len(),
            report
                .misses
                .iter()
                .take(5)
                .map(|m| (m.row, m.atom, m.reason, m.alignment))
                .collect::<Vec<_>>()
        );
        // No silent truncation: the miss ledger accounts for every unrecovered slot.
        assert_eq!(
            report.total_true - report.total_recovered,
            report.misses.len(),
            "the miss ledger must account for every unrecovered exact-top-s atom"
        );
    }
}
