//! Unified support-sparse assignment state for the SAE fit-path unification
//! (design: `docs/design/sae-unification.md`, Increment 1).
//!
//! [`SaeAssignmentState`] is the ONE per-row routing state the unified engine
//! carries: per row it stores the active atom set `S_i ⊆ [K]` as
//!
//!   * `indices[i]`      — the active atom indices (`u32`), `|S_i|` of them;
//!   * `gate_params[i]`  — one routing scalar per active atom (the dense
//!     `logits[i, k]` value; the realized gate is derived from it through the
//!     [`AssignmentMode`], so this is the *fundamental* stored parameter, and it
//!     is exactly the `f64` the [`SaeTopKCurvedBudget`] "gate values" slot
//!     budgets, one per active atom — see the layout contract below);
//!   * `coords[i]`       — the on-manifold coordinates of ONLY the active atoms,
//!     flattened in support order (`Σ_{k∈S_i} d_k` scalars per row).
//!
//! The dense [`SaeAssignment`] (`assignment.rs`) is the FULL-SUPPORT
//! materialization of this state: `S_i = [0, K)` for every row, so the
//! per-`(row, atom)` coordinate storage is the transpose of the dense
//! `Vec<LatentCoordValues>` per-atom blocks, and `gate_params[i]` is the dense
//! `logits` row. [`SaeAssignmentState::materialize_dense`] reconstructs that dense
//! layout bit-for-bit, and [`SaeAssignment::as_state`] is its inverse.
//!
//! # Layout contract vs. the `SaeTopKCurvedBudget` ledger
//!
//! `manifold/streaming_plan.rs` charges the honest support-sparse assignment
//! state as
//!
//! ```text
//!   active_state_bytes = N · k_active · (2 + d_max) · 8
//! ```
//!
//! (`sae_topk_curved_budget_from_budget`, streaming_plan.rs:482-485). The three
//! `(2 + d_max)` terms are exactly this state's three per-row arrays: `1` for
//! `indices`, `1` for `gate_params`, and `d_max` for `coords` — every cell an
//! 8-byte word (`SAE_BYTES_PER_F64`; the `u32` index cell is budgeted as a full
//! 8-byte slot, matching the ledger's uniform-word accounting). For a uniform
//! `k_active = s`, `d_k = d_max` shape this state therefore occupies
//! [`SaeAssignmentState::active_state_bytes`]
//! `= N · s · (2 + d_max) · 8 = active_state_bytes`, verified by
//! `sparse_topk_state_memory_shape_matches_budget_formula`.

use gam_problem::LatentRetractionRegistry;
use gam_terms::latent::{LatentCoordValues, LatentIdMode, LatentManifold};
use ndarray::{Array1, Array2};

use crate::assignment::{AssignmentMode, SaeAssignment};

/// Byte width of one budgeted state cell, matching
/// `streaming_plan::SAE_BYTES_PER_F64`. Kept as a local constant so this module
/// does not reach into the private streaming-plan module for a single integer;
/// the layout-contract test pins it against the real budget arithmetic.
const STATE_CELL_BYTES: usize = 8;

/// Per-atom coordinate metadata needed to reconstruct the dense
/// [`LatentCoordValues`] block bit-for-bit on materialization.
#[derive(Debug, Clone)]
struct AtomCoordMeta {
    latent_dim: usize,
    id_mode: LatentIdMode,
    manifold: LatentManifold,
    retraction: LatentRetractionRegistry,
    /// Process-local stable identity of the source block. Preserved so a
    /// dense → state → dense round-trip is identity-stable, not merely
    /// value-stable.
    latent_id: u64,
}

/// Support-sparse per-row assignment state (see module docs). Internal type: the
/// unified engine's ONE routing state, of which the dense [`SaeAssignment`] is
/// the full-support specialization.
#[derive(Debug, Clone)]
pub struct SaeAssignmentState {
    n_obs: usize,
    k_atoms: usize,
    /// Active atom indices per row (`indices[i]` sorted-ascending for a
    /// full-support state; TopK constructors pass the routed support).
    indices: Vec<Vec<u32>>,
    /// Routing scalar per active atom per row — the dense `logits[i, k]` value.
    gate_params: Vec<Vec<f64>>,
    /// Active-atom coordinates per row, flattened in support order: for row `i`
    /// the concatenation over `j` of the `d_{indices[i][j]}` coordinate scalars.
    coords: Vec<Vec<f64>>,
    /// Per-atom coordinate metadata (length `K`) for dense reconstruction.
    atom_coord_meta: Vec<AtomCoordMeta>,
    mode: AssignmentMode,
    /// #1026 per-atom ungated flag (length `K`).
    ungated: Vec<bool>,
    /// #1033 frozen/amortized routing, dense `(N, K)` when engaged (a
    /// full-support-only field; the sparse TopK lane never freezes routing).
    frozen_logits: Option<Array2<f64>>,
    /// #1777 per-fit IBP-α override.
    ibp_alpha_override: Option<f64>,
}

impl SaeAssignmentState {
    /// Skeleton full-support state: `N` rows, `K` atoms, `S_i = [0, K)`, zero
    /// routing scalars, zero-dimensional Euclidean coordinates, softmax mode, no
    /// ungated atoms, free (non-frozen) routing. The minimal coherent
    /// full-support state; callers fill in routing scalars / coordinates through
    /// the mutable accessors or (more commonly) obtain a populated state via
    /// [`SaeAssignment::as_state`].
    #[must_use]
    pub fn full_support(n_obs: usize, k_atoms: usize) -> Self {
        let indices: Vec<Vec<u32>> = (0..n_obs)
            .map(|_| (0..k_atoms as u32).collect())
            .collect();
        let gate_params = vec![vec![0.0_f64; k_atoms]; n_obs];
        // d_k = 0 for every atom ⇒ empty per-row coord blocks.
        let coords = vec![Vec::new(); n_obs];
        let atom_coord_meta = (0..k_atoms)
            .map(|_| AtomCoordMeta {
                latent_dim: 0,
                id_mode: LatentIdMode::None,
                manifold: LatentManifold::Euclidean,
                retraction: LatentRetractionRegistry::all_euclidean(),
                latent_id: 0,
            })
            .collect();
        Self {
            n_obs,
            k_atoms,
            indices,
            gate_params,
            coords,
            atom_coord_meta,
            mode: AssignmentMode::softmax(1.0),
            ungated: vec![false; k_atoms],
            frozen_logits: None,
            ibp_alpha_override: None,
        }
    }

    /// Support-sparse hard-TopK state (the honest `O(N · k_active)` lane, design
    /// Increment 1). Every row carries exactly `support_k` active atoms whose
    /// coordinates are `d_max`-dimensional Euclidean, so the state occupies
    /// exactly the [`SaeTopKCurvedBudget`] `active_state_bytes`.
    ///
    /// * `indices[i]`     — length `support_k`, each `< k_atoms`;
    /// * `gate_params[i]` — length `support_k` (routing scalars);
    /// * `coords[i]`      — length `support_k · d_max`, active-atom coords in the
    ///   same order as `indices[i]`.
    ///
    /// [`SaeTopKCurvedBudget`]: crate::manifold::SaeTopKCurvedBudget
    #[must_use = "state build error must be handled"]
    pub fn from_topk_support(
        n_obs: usize,
        k_atoms: usize,
        support_k: usize,
        d_max: usize,
        indices: Vec<Vec<u32>>,
        gate_params: Vec<Vec<f64>>,
        coords: Vec<Vec<f64>>,
    ) -> Result<Self, String> {
        if support_k == 0 || support_k > k_atoms {
            return Err(format!(
                "SaeAssignmentState::from_topk_support: support_k must satisfy 1 <= s <= K={k_atoms}; got {support_k}"
            ));
        }
        if indices.len() != n_obs || gate_params.len() != n_obs || coords.len() != n_obs {
            return Err(format!(
                "SaeAssignmentState::from_topk_support: per-row arrays must all have length N={n_obs}; \
                 got indices={}, gate_params={}, coords={}",
                indices.len(),
                gate_params.len(),
                coords.len()
            ));
        }
        for i in 0..n_obs {
            if indices[i].len() != support_k
                || gate_params[i].len() != support_k
                || coords[i].len() != support_k * d_max
            {
                return Err(format!(
                    "SaeAssignmentState::from_topk_support: row {i} widths must be indices={support_k}, \
                     gate_params={support_k}, coords={}; got {}, {}, {}",
                    support_k * d_max,
                    indices[i].len(),
                    gate_params[i].len(),
                    coords[i].len()
                ));
            }
            for &atom in &indices[i] {
                if atom as usize >= k_atoms {
                    return Err(format!(
                        "SaeAssignmentState::from_topk_support: row {i} atom index {atom} out of range K={k_atoms}"
                    ));
                }
            }
        }
        let atom_coord_meta = (0..k_atoms)
            .map(|_| AtomCoordMeta {
                latent_dim: d_max,
                id_mode: LatentIdMode::None,
                manifold: LatentManifold::Euclidean,
                retraction: LatentRetractionRegistry::all_euclidean(),
                latent_id: 0,
            })
            .collect();
        Ok(Self {
            n_obs,
            k_atoms,
            indices,
            gate_params,
            coords,
            atom_coord_meta,
            mode: AssignmentMode::top_k_support(support_k),
            ungated: vec![false; k_atoms],
            frozen_logits: None,
            ibp_alpha_override: None,
        })
    }

    pub fn n_obs(&self) -> usize {
        self.n_obs
    }

    pub fn k_atoms(&self) -> usize {
        self.k_atoms
    }

    pub fn mode(&self) -> AssignmentMode {
        self.mode
    }

    /// Active atom indices `S_i` for `row`.
    pub fn support_indices(&self, row: usize) -> &[u32] {
        &self.indices[row]
    }

    /// Routing scalars (dense `logits` values) on `row`'s active support.
    pub fn gate_params(&self, row: usize) -> &[f64] {
        &self.gate_params[row]
    }

    /// The whole flattened active-atom coordinate block for `row`
    /// (`Σ_{k∈S_i} d_k` scalars, support order).
    pub fn coords_row(&self, row: usize) -> &[f64] {
        &self.coords[row]
    }

    /// Whether every row's support is the full `[0, K)` in ascending order (the
    /// dense-materialization precondition).
    pub fn is_full_support(&self) -> bool {
        if self.n_obs == 0 {
            return true;
        }
        self.indices.iter().all(|row| {
            row.len() == self.k_atoms && row.iter().enumerate().all(|(k, &a)| a as usize == k)
        })
    }

    // -- Layout-contract cell accounting (see module docs) -------------------

    /// Total `indices` cells `Σ_i |S_i|`.
    pub fn index_cells(&self) -> usize {
        self.indices.iter().map(Vec::len).sum()
    }

    /// Total `gate_params` cells `Σ_i |S_i|`.
    pub fn gate_cells(&self) -> usize {
        self.gate_params.iter().map(Vec::len).sum()
    }

    /// Total coordinate cells `Σ_i Σ_{k∈S_i} d_k`.
    pub fn coord_cells(&self) -> usize {
        self.coords.iter().map(Vec::len).sum()
    }

    /// Total support-sparse state cells `indices + gate_params + coords`.
    pub fn active_state_cells(&self) -> usize {
        self.index_cells() + self.gate_cells() + self.coord_cells()
    }

    /// Support-sparse state footprint in bytes, one 8-byte word per cell —
    /// equal to the [`SaeTopKCurvedBudget`] `active_state_bytes` for a uniform
    /// TopK shape (see the module layout contract).
    ///
    /// [`SaeTopKCurvedBudget`]: crate::manifold::SaeTopKCurvedBudget
    pub fn active_state_bytes(&self) -> usize {
        self.active_state_cells().saturating_mul(STATE_CELL_BYTES)
    }

    /// Materialize the exact dense [`SaeAssignment`] layout this state
    /// represents. Requires [`Self::is_full_support`]: the dense engine only
    /// exists for the `S_i = [0, K)` specialization, and a proper-sparse state
    /// has no dense `N×K` image.
    #[must_use = "materialization error must be handled"]
    pub fn materialize_dense(&self) -> Result<SaeAssignment, String> {
        if !self.is_full_support() {
            return Err(
                "SaeAssignmentState::materialize_dense: requires a full-support state (S_i = [0, K) \
                 for every row); a proper support-sparse state has no dense N×K materialization"
                    .to_string(),
            );
        }
        let n = self.n_obs;
        let k = self.k_atoms;

        // logits[i, k] = gate_params[i][k] (full support ⇒ index j == atom k).
        let mut logits = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            for (col, &g) in self.gate_params[i].iter().enumerate() {
                logits[[i, col]] = g;
            }
        }

        // Rebuild each atom's LatentCoordValues from the per-row support blocks.
        // In full-support order the coord offset of atom k in a row is the prefix
        // sum of the atoms' latent dims.
        let mut coord_offsets = Vec::with_capacity(k);
        let mut cursor = 0usize;
        for meta in &self.atom_coord_meta {
            coord_offsets.push(cursor);
            cursor += meta.latent_dim;
        }
        let mut coords = Vec::with_capacity(k);
        for (atom, meta) in self.atom_coord_meta.iter().enumerate() {
            let d = meta.latent_dim;
            let mut flat = Array1::<f64>::zeros(n * d);
            if d > 0 {
                let off = coord_offsets[atom];
                for i in 0..n {
                    let row = &self.coords[i];
                    for axis in 0..d {
                        flat[i * d + axis] = row[off + axis];
                    }
                }
            }
            coords.push(LatentCoordValues::from_flat_with_manifold_and_retraction_and_id(
                flat,
                n,
                d,
                meta.id_mode.clone(),
                meta.manifold.clone(),
                meta.retraction.clone(),
                meta.latent_id,
            ));
        }

        // Direct field construction (all fields are `pub`, in-crate): the logits
        // were captured already canonicalized by `as_state`, so re-routing them
        // through the validating/canonicalizing `with_mode` is unnecessary and
        // would only risk a non-identity round-trip. This reverses `as_state`
        // exactly.
        Ok(SaeAssignment {
            logits,
            coords,
            mode: self.mode,
            ungated: self.ungated.clone(),
            frozen_logits: self.frozen_logits.clone(),
            ibp_alpha_override: self.ibp_alpha_override,
        })
    }
}

impl SaeAssignment {
    /// View this dense assignment as its full-support [`SaeAssignmentState`]
    /// (design Increment 1). The inverse of
    /// [`SaeAssignmentState::materialize_dense`]: for every row the support is
    /// `[0, K)`, `gate_params` is the dense `logits` row, and the coordinate
    /// block is the per-atom `LatentCoordValues` rows gathered in atom order.
    #[must_use]
    pub fn as_state(&self) -> SaeAssignmentState {
        let n = self.n_obs();
        let k = self.k_atoms();

        let per_atom_dim: Vec<usize> = self.coords.iter().map(LatentCoordValues::latent_dim).collect();

        let indices: Vec<Vec<u32>> = (0..n).map(|_| (0..k as u32).collect()).collect();
        let mut gate_params = Vec::with_capacity(n);
        let mut coords = Vec::with_capacity(n);
        for i in 0..n {
            gate_params.push(self.logits.row(i).to_vec());
            let row_len: usize = per_atom_dim.iter().sum();
            let mut row_coords = Vec::with_capacity(row_len);
            for atom in 0..k {
                row_coords.extend_from_slice(self.coords[atom].row(i));
            }
            coords.push(row_coords);
        }

        let atom_coord_meta = self
            .coords
            .iter()
            .map(|c| AtomCoordMeta {
                latent_dim: c.latent_dim(),
                id_mode: c.id_mode().clone(),
                manifold: c.manifold().clone(),
                retraction: c.retraction_registry().clone(),
                latent_id: c.latent_id(),
            })
            .collect();

        SaeAssignmentState {
            n_obs: n,
            k_atoms: k,
            indices,
            gate_params,
            coords,
            atom_coord_meta,
            mode: self.mode,
            ungated: self.ungated.clone(),
            frozen_logits: self.frozen_logits.clone(),
            ibp_alpha_override: self.ibp_alpha_override,
        }
    }

    /// Construct the dense assignment that a full-support [`SaeAssignmentState`]
    /// materializes (design Increment 1). Errors if `state` is not full-support.
    #[must_use = "build error must be handled"]
    pub fn from_full_support_state(state: &SaeAssignmentState) -> Result<Self, String> {
        state.materialize_dense()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn coord_block(n: usize, d: usize, seed: f64) -> Array2<f64> {
        let mut m = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                m[[i, j]] = seed + (i as f64) * 0.5 - (j as f64) * 0.25;
            }
        }
        m
    }

    fn dense_fixture(mode: AssignmentMode, n: usize, k: usize, d: usize) -> SaeAssignment {
        let mut logits = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            for j in 0..k {
                // Distinct, non-symmetric logits so softmax canonicalization and
                // TopK tie-breaks are actually exercised.
                logits[[i, j]] = 0.3 * (i as f64) - 0.7 * (j as f64) + 0.11 * ((i * k + j) as f64);
            }
        }
        let coord_blocks: Vec<Array2<f64>> =
            (0..k).map(|atom| coord_block(n, d, 0.2 * atom as f64)).collect();
        SaeAssignment::from_blocks_with_mode(logits, coord_blocks, mode)
            .expect("dense fixture builds")
    }

    fn assert_bit_exact_roundtrip(mode: AssignmentMode) {
        let (n, k, d) = (7usize, 4usize, 2usize);
        let dense = dense_fixture(mode, n, k, d);
        let state = dense.as_state();
        assert!(state.is_full_support(), "as_state is full-support");
        let back = SaeAssignment::from_full_support_state(&state).expect("materialize dense");

        // Fundamental stored parameter: logits bit-for-bit.
        assert_eq!(dense.logits, back.logits, "logits round-trip");
        // Derived gates bit-for-bit (pins mode equivalence through the gate map).
        assert_eq!(
            dense.assignments(),
            back.assignments(),
            "assignments round-trip"
        );
        // Mode, shape, and per-atom flags.
        assert_eq!(format!("{:?}", dense.mode), format!("{:?}", back.mode), "mode");
        assert_eq!(dense.n_obs(), back.n_obs());
        assert_eq!(dense.k_atoms(), back.k_atoms());
        assert_eq!(dense.ungated, back.ungated, "ungated flags");
        assert_eq!(dense.frozen_logits, back.frozen_logits, "frozen routing");
        assert_eq!(
            dense.ibp_alpha_override, back.ibp_alpha_override,
            "ibp alpha override"
        );
        // Coordinates bit-for-bit, including stable identity and geometry.
        assert_eq!(dense.coords.len(), back.coords.len(), "coord block count");
        for atom in 0..k {
            let a = &dense.coords[atom];
            let b = &back.coords[atom];
            assert_eq!(a.latent_dim(), b.latent_dim(), "atom {atom} latent dim");
            assert_eq!(a.latent_id(), b.latent_id(), "atom {atom} latent id");
            assert_eq!(a.as_flat(), b.as_flat(), "atom {atom} coord values");
        }
    }

    #[test]
    fn dense_state_dense_roundtrip_is_bit_exact_softmax() {
        assert_bit_exact_roundtrip(AssignmentMode::softmax(0.8));
    }

    #[test]
    fn dense_state_dense_roundtrip_is_bit_exact_topk() {
        assert_bit_exact_roundtrip(AssignmentMode::top_k_support(2));
    }

    #[test]
    fn dense_state_dense_roundtrip_is_bit_exact_ibp_map() {
        assert_bit_exact_roundtrip(AssignmentMode::ibp_map(0.9, 1.0, false));
    }

    #[test]
    fn dense_state_dense_roundtrip_is_bit_exact_threshold_gate() {
        assert_bit_exact_roundtrip(AssignmentMode::threshold_gate(0.7, 0.1));
    }

    #[test]
    fn full_support_skeleton_materializes_and_reports_full_support() {
        let state = SaeAssignmentState::full_support(5, 3);
        assert!(state.is_full_support());
        assert_eq!(state.n_obs(), 5);
        assert_eq!(state.k_atoms(), 3);
        let dense = state.materialize_dense().expect("skeleton materializes");
        assert_eq!(dense.n_obs(), 5);
        assert_eq!(dense.k_atoms(), 3);
        assert_eq!(dense.total_coord_dim(), 0, "d_k = 0 skeleton");
    }

    #[test]
    fn sparse_topk_state_memory_shape_matches_budget_formula() {
        // The (N, K, s, d) shape the layout contract pins.
        let (n, k, s, d) = (1000usize, 5000usize, 8usize, 2usize);
        let indices: Vec<Vec<u32>> = (0..n)
            .map(|i| (0..s as u32).map(|j| ((i + j as usize) % k) as u32).collect())
            .collect();
        let gate_params: Vec<Vec<f64>> = (0..n).map(|_| vec![1.0_f64; s]).collect();
        let coords: Vec<Vec<f64>> = (0..n).map(|_| vec![0.0_f64; s * d]).collect();
        let state = SaeAssignmentState::from_topk_support(
            n, k, s, d, indices, gate_params, coords,
        )
        .expect("sparse topk state builds");

        // Cell counts equal the (2 + d_max) budget decomposition.
        assert_eq!(state.index_cells(), n * s, "index cells");
        assert_eq!(state.gate_cells(), n * s, "gate cells");
        assert_eq!(state.coord_cells(), n * s * d, "coord cells");
        assert_eq!(
            state.active_state_cells(),
            n * s * (2 + d),
            "total active-state cells = N·s·(2+d_max)"
        );

        // Bytes equal the SaeTopKCurvedBudget.active_state_bytes for this shape.
        let budget = crate::manifold::sae_topk_curved_budget_from_budget(
            n,
            /* output_dim */ 128,
            k,
            d,
            s,
            /* in_core_budget_bytes */ usize::MAX / 2,
        );
        assert_eq!(
            state.active_state_bytes(),
            budget.active_state_bytes,
            "state footprint equals SaeTopKCurvedBudget.active_state_bytes"
        );
        // And equals the closed-form formula directly.
        assert_eq!(state.active_state_bytes(), n * s * (2 + d) * 8);
    }

    #[test]
    fn proper_sparse_state_refuses_dense_materialization() {
        let state = SaeAssignmentState::from_topk_support(
            4,
            10,
            2,
            1,
            (0..4).map(|_| vec![0u32, 1]).collect(),
            (0..4).map(|_| vec![0.0, 0.0]).collect(),
            (0..4).map(|_| vec![0.0, 0.0]).collect(),
        )
        .expect("state builds");
        assert!(!state.is_full_support());
        assert!(state.materialize_dense().is_err());
    }
}
