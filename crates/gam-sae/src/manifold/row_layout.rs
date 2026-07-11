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
    /// active atom carries a coord block; not every one carries a free logit slot
    /// (see `logit_atoms`).
    pub active_atoms: Vec<Vec<usize>>,
    /// Free assignment-logit slots. Empty for every production TopK row; kept as
    /// an explicit field because the common row-assembly operator indexes this
    /// prefix before the coordinate blocks.
    pub logit_atoms: Vec<Vec<usize>>,
    /// For row `i`, active atom `active_atoms[i][j]` has its coord block
    /// starting at compressed position `coord_starts[i][j]`.
    pub coord_starts: Vec<Vec<usize>>,
    /// Full-q coordinate offset for atom `k` (length `k_atoms`).
    pub coord_offsets_full: Vec<usize>,
    /// Per-atom coordinate dimensions, indexed by atom index.
    pub coord_dims: Vec<usize>,
}

impl SaeRowLayout {
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
            logit_atoms: vec![Vec::new(); per_row.len()],
            active_atoms: per_row,
            coord_starts,
            coord_offsets_full,
            coord_dims,
        })
    }

    #[cfg(test)]
    /// Build honoring an optional `reference_atom` that carries a COORD block but
    /// NO free logit slot (softmax's pinned reference `K−1`). When a row's active
    /// set contains it, it is excluded from `logit_atoms` but kept in
    /// `active_atoms`. Since the reference is the largest atom and `active_atoms`
    /// is sorted, it is always the last active element, so the leading
    /// `logit_atoms.len()` compact slots are the logit slots. (#Bug1)
    pub(crate) fn from_active_atoms_with_reference(
        active_atoms: Vec<Vec<usize>>,
        coord_dims: Vec<usize>,
        coord_offsets_full: Vec<usize>,
        reference_atom: Option<usize>,
    ) -> Self {
        let mut logit_atoms_all = Vec::with_capacity(active_atoms.len());
        let mut coord_starts_all = Vec::with_capacity(active_atoms.len());
        for active in &active_atoms {
            let logit_atoms: Vec<usize> = active
                .iter()
                .copied()
                .filter(|&k| Some(k) != reference_atom)
                .collect();
            let mut starts = Vec::with_capacity(active.len());
            // Coord blocks start AFTER the logit slots.
            let mut cursor = logit_atoms.len();
            for &k in active {
                starts.push(cursor);
                cursor += coord_dims[k];
            }
            logit_atoms_all.push(logit_atoms);
            coord_starts_all.push(starts);
        }
        Self {
            active_atoms,
            logit_atoms: logit_atoms_all,
            coord_starts: coord_starts_all,
            coord_offsets_full,
            coord_dims,
        }
    }

    /// Number of free logit slots in row `row`'s compact block.
    pub fn n_logit_active(&self, row: usize) -> usize {
        self.logit_atoms[row].len()
    }

    /// Per-row compressed dim: free logit slots + coord blocks for every active
    /// atom.
    pub fn row_q_active(&self, row: usize) -> usize {
        let active = &self.active_atoms[row];
        let coord_sum: usize = active.iter().map(|&k| self.coord_dims[k]).sum();
        self.logit_atoms[row].len() + coord_sum
    }

    /// Expand a compact `delta_t` row slice back into full-q, zeros for inactive.
    /// The softmax reference atom has no logit slot (its logit position does not
    /// exist in the reduced chart), so only its coord block is written. (#Bug1)
    pub fn expand_row(&self, row: usize, delta_t_row: &[f64], out: &mut [f64]) {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        for (j, &k) in self.logit_atoms[row].iter().enumerate() {
            out[k] = delta_t_row[j];
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

#[cfg(test)]
mod softmax_reference_chart_tests {
    //! #Bug1 — a SOFTMAX compact active set containing the reference atom `K−1`
    //! must give it a COORD block but NO free logit slot, and `expand_row` must
    //! never write a phantom reference logit into a coordinate position.
    use super::SaeRowLayout;

    #[test]
    fn softmax_reference_atom_has_coords_but_no_logit_slot() {
        // K=3, each atom coord dim 1. Full softmax chart (row_block_dim=5):
        // full 0,1 = free logits (atoms 0,1); full 2,3,4 = coords (atoms 0,1,2).
        // Atom 2 is the reference (K−1) with NO logit position.
        let coord_dims = vec![1usize, 1, 1];
        let coord_offsets_full = vec![2usize, 3, 4];
        let active = vec![vec![0usize, 2], vec![2usize]];
        let layout = SaeRowLayout::from_active_atoms_with_reference(
            active,
            coord_dims,
            coord_offsets_full,
            Some(2),
        );
        assert_eq!(layout.logit_atoms[0], vec![0]);
        assert_eq!(layout.n_logit_active(0), 1);
        assert_eq!(layout.row_q_active(0), 3); // 1 logit + coords(atom0)+coords(atom2)
        assert_eq!(layout.logit_atoms[1], Vec::<usize>::new());
        assert_eq!(layout.n_logit_active(1), 0);
        assert_eq!(layout.row_q_active(1), 1);
        // expand_row: compact [logit(atom0), coord(atom0), coord(atom2)] must land
        // as logit0→full0, coord atom0→full2, coord atom2→full4 — full index 2
        // (=coord atom 0) must receive the coordinate, never a phantom reference
        // logit.
        let mut out = vec![0.0_f64; 5];
        layout.expand_row(0, &[10.0, 20.0, 30.0], &mut out);
        assert_eq!(out, vec![10.0, 0.0, 20.0, 0.0, 30.0]);
        for (j, &k) in layout.logit_atoms[0].iter().enumerate() {
            assert_ne!(k, 2, "logit slot {j} must not be the reference atom");
        }
    }

}
