use super::*;

/// Per-row layout for the explicit hard-TopK assignment model.
///
/// TopK gates are exactly zero or one and carry no optimizable logit
/// coordinates. Only the selected atoms' manifold coordinates enter a row's
/// Arrow-Schur block, whose dimension is exactly
/// `Σ_{k ∈ support_i} d_k`. Dense Softmax, ordered independent
/// Beta--Bernoulli, and threshold gates never use this type: each has nonzero
/// derivatives on its full support and is assembled exactly or refused before
/// allocation if the exact system does not fit the declared memory budget.
#[derive(Debug, Clone)]
pub struct SaeRowLayout {
    /// `active_atoms[row]` — sorted indices of active atoms for that row. Every
    /// active atom carries a coordinate block.
    pub active_atoms: Vec<Vec<usize>>,
    /// For row `i`, active atom `active_atoms[i][j]` has its coord block
    /// starting at compressed position `coord_starts[i][j]`.
    pub coord_starts: Vec<Vec<usize>>,
    /// Full-q coordinate offset for atom `k` (length `k_atoms`).
    pub coord_offsets_full: Vec<usize>,
    /// Per-atom coordinate dimensions, indexed by atom index.
    pub coord_dims: Vec<usize>,
}

impl SaeRowLayout {
    /// Build directly from the canonical support-sparse state. This is the
    /// production TopK path: it never constructs K-wide gates merely to recover
    /// the support indices that are already the fundamental state.
    pub(crate) fn from_assignment_state(
        state: &crate::assignment_state::SaeAssignmentState,
    ) -> Result<Self, String> {
        let mut coord_offsets_full = Vec::with_capacity(state.k_atoms());
        let mut cursor = 0usize;
        let mut coord_dims = Vec::with_capacity(state.k_atoms());
        for atom in 0..state.k_atoms() {
            coord_offsets_full.push(cursor);
            let d = state.atom_coord_dim(atom);
            coord_dims.push(d);
            cursor += d;
        }
        let active_atoms = (0..state.n_obs())
            .map(|row| {
                state
                    .support_indices(row)
                    .iter()
                    .map(|&atom| atom as usize)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut coord_starts = Vec::with_capacity(state.n_obs());
        for active in &active_atoms {
            let mut row_cursor = 0usize;
            let mut starts = Vec::with_capacity(active.len());
            for &atom in active {
                starts.push(row_cursor);
                row_cursor += coord_dims[atom];
            }
            coord_starts.push(starts);
        }
        Ok(Self {
            active_atoms,
            coord_starts,
            coord_offsets_full,
            coord_dims,
        })
    }

    /// Build the exact compact layout from hard TopK gates. Every row must have
    /// exactly `support_size` entries equal to one and every other entry equal
    /// to zero; accepting approximate weights here would silently change the
    /// model by dropping nonzero derivatives.
    pub(crate) fn from_topk_gates(
        assignments: &[Array1<f64>],
        support_size: usize,
        coord_dims: Vec<usize>,
        coord_offsets_full: Vec<usize>,
    ) -> Result<Self, String> {
        if support_size == 0 {
            return Err("SaeRowLayout::from_topk_gates requires positive support_size".to_string());
        }
        let mut per_row = Vec::with_capacity(assignments.len());
        for (row, gates) in assignments.iter().enumerate() {
            let mut active = Vec::with_capacity(support_size);
            for (atom, &gate) in gates.iter().enumerate() {
                if gate == 1.0 {
                    active.push(atom);
                } else if gate != 0.0 {
                    return Err(format!(
                        "SaeRowLayout::from_topk_gates: row {row}, atom {atom} has non-binary gate {gate}"
                    ));
                }
            }
            if active.len() != support_size.min(gates.len()) {
                return Err(format!(
                    "SaeRowLayout::from_topk_gates: row {row} has {} active atoms; expected {}",
                    active.len(),
                    support_size.min(gates.len())
                ));
            }
            per_row.push(active);
        }
        let mut coord_starts = Vec::with_capacity(per_row.len());
        for active in &per_row {
            let mut cursor = 0usize;
            let mut starts = Vec::with_capacity(active.len());
            for &atom in active {
                starts.push(cursor);
                cursor += coord_dims[atom];
            }
            coord_starts.push(starts);
        }
        Ok(Self {
            active_atoms: per_row,
            coord_starts,
            coord_offsets_full,
            coord_dims,
        })
    }

    /// Per-row compressed dimension: coordinate blocks for active atoms.
    pub fn row_q_active(&self, row: usize) -> usize {
        let active = &self.active_atoms[row];
        let coord_sum: usize = active.iter().map(|&k| self.coord_dims[k]).sum();
        coord_sum
    }

    /// Expand a compact TopK coordinate step back into the full coordinate row,
    /// writing zeros for inactive atoms.
    pub fn expand_row(&self, row: usize, delta_t_row: &[f64], out: &mut [f64]) {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        let active = &self.active_atoms[row];
        let starts = &self.coord_starts[row];
        for (pos, &k) in active.iter().enumerate() {
            let d = self.coord_dims[k];
            let full_off = self.coord_offsets_full[k];
            for axis in 0..d {
                out[full_off + axis] = delta_t_row[starts[pos] + axis];
            }
        }
    }
}
