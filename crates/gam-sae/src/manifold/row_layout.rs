use super::*;

/// Per-row active-set layout for sparse SAE assignment (any mode).
///
/// When the assignment is effectively sparse (softmax / ordered
/// Beta--Bernoulli at large `K`, where the assignment mass
/// concentrates on a small support) — only a subset of `K` atoms are active
/// per observation.  The Arrow-Schur row block for observation `i` has dim
/// `q_active_i = |active_atoms_i| + Σ_{k ∈ active_i} d_k` rather than
/// `q = assignment_dim + Σ_k d_k`.  This struct records which atoms are active per row
/// and maps compressed block positions back to full-q positions so that
/// `apply_newton_step` can unpack the compact `delta_t` from the solve.
///
/// For ordered Beta--Bernoulli the active set is the union of a top-`k_active_cap`
/// truncation and a magnitude cutoff on `a_{n,k}`; this is only enabled when
/// `K` is large enough that the dense `(m_total · p)²` data Gram would not
/// fit the host / device working-set budget, and the dropped atoms carry
/// `O(a_{n,k}²)` curvature that is negligible by construction of the cutoff.
///
/// #1408: SOFTMAX engages this compact layout when an explicit `top_k`
/// (`softmax_active_cap`) and/or the in-core memory budget bounds the active
/// set — the `AssignmentMode::Softmax` arm of `assemble_arrow_schur` consults
/// [`crate::manifold::SaeManifoldTerm::softmax_active_plan`] and,
/// on `Some((cap, cutoff))`, builds the active set via
/// [`Self::from_dense_weights`]. The full-`K` dense softmax layout is retained
/// only when neither lever engages (no `top_k`, in-budget `K`). Folding softmax
/// `top_k` into the compact solve required writing the active×active Gershgorin
/// Loewner majorizer sub-block (#1419; the softmax entropy curvature is
/// indefinite, so its raw diagonal cannot be used) AND contracting that SAME
/// majorizer over the compact logit slots in the logdet ρ-trace
/// (`assignment_log_strength_hessian_trace`) and the θ-adjoint, so value,
/// `log|H|`, and Γ differentiate one operator on the compact support. That
/// coordinated change is landed and FD-certified; the FFI's after-the-fit
/// top-`k` projection is then a no-op at the optimum.
#[derive(Debug, Clone)]
pub struct SaeRowLayout {
    /// `active_atoms[row]` — sorted indices of active atoms for that row. Every
    /// active atom carries a coord block; not every one carries a free logit slot
    /// (see `logit_atoms`).
    pub active_atoms: Vec<Vec<usize>>,
    /// `logit_atoms[row]` — the subset of `active_atoms[row]` (same ascending
    /// order) carrying a FREE assignment-logit slot. Equals `active_atoms` for the
    /// column-separable modes. For SOFTMAX the reduced chart
    /// has only `K−1` free logits: the reference atom `K−1` (pinned to zero, no
    /// logit coordinate) is excluded. Since `K−1` is the largest atom and
    /// `active_atoms` is sorted, the reference (when active) is always last, so
    /// `logit_atoms[row] == active_atoms[row][..n_logit_active(row)]`. The compact
    /// block is `[one slot per logit atom]` then `[one coord block per active
    /// atom]`. (#Bug1)
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
    /// Mode-agnostic effective active set for dense-weight modes (softmax /
    /// ordered Beta--Bernoulli) at large `K`: keep, per row, the top-`k_active_cap` atoms by
    /// `|a_{n,k}|` whose magnitude also exceeds `relative_cutoff · rowpeak`.
    ///
    /// #1414: the cutoff is RELATIVE TO EACH ROW'S OWN PEAK `max_k |a_{n,k}|`,
    /// matching the documented `sparse_active_plan` contract
    /// (`construction.rs:1763-1766`). A global cutoff (one threshold from the
    /// whole-dataset peak) would wrongly drop both atoms of a uniformly-small row
    /// `[0.0009, 0.0008]` just because another row peaks at `1.0`, changing the
    /// high-`K` compact model.
    ///
    /// `assignments[row]` is the dense length-`K` assignment vector `a_{n,·}`.
    /// The active set is always non-empty (the single largest-magnitude atom is
    /// retained even if below cutoff) so every row keeps a valid block.
    /// `reference_atom` is `Some(K−1)` for the reduced SOFTMAX chart (that atom
    /// gets a coord block but no free logit slot) and `None` for ordered Beta--Bernoulli. (#Bug1)
    pub(crate) fn from_dense_weights(
        assignments: &[Array1<f64>],
        k_active_cap: usize,
        relative_cutoff: f64,
        coord_dims: Vec<usize>,
        coord_offsets_full: Vec<usize>,
        reference_atom: Option<usize>,
    ) -> Self {
        let cap = k_active_cap.max(1);
        let mut per_row = Vec::with_capacity(assignments.len());
        for a in assignments {
            let k = a.len();
            // #1411: select the top-`cap` atoms by |a_k| in O(K) with a PARTIAL
            // select (`select_nth_unstable_by`), not a full O(K log K) sort. Only
            // the cap-sized active prefix matters; its internal order is
            // irrelevant (sorted at the end). The row peak is a separate O(K) max
            // scan. End-to-end this keeps support proposal O(K) (single pass +
            // partial select), the contracted per-token cost the high-K plan
            // claims, instead of sorting all K per row.
            let row_peak = a.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
            let cutoff = relative_cutoff * row_peak;
            let mut idx: Vec<usize> = (0..k).collect();
            // Partition so the `cap` largest-|a| indices occupy `idx[..cap]`
            // (unordered within); cheaper than a full sort when `cap << k`.
            if cap < k {
                idx.select_nth_unstable_by(cap - 1, |&i, &j| {
                    a[j].abs()
                        .partial_cmp(&a[i].abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                idx.truncate(cap);
            }
            let mut active: Vec<usize> = idx
                .into_iter()
                .filter(|&k_idx| a[k_idx].abs() > cutoff)
                .collect();
            if active.is_empty() {
                // Retain the single largest-magnitude atom so the row block is
                // never empty (a degenerate empty block would zero the row).
                let top = (0..k).fold(None::<usize>, |best, i| match best {
                    Some(b) if a[b].abs() >= a[i].abs() => Some(b),
                    _ => Some(i),
                });
                if let Some(top) = top {
                    active.push(top);
                }
            }
            active.sort_unstable();
            per_row.push(active);
        }
        Self::from_active_atoms_with_reference(
            per_row,
            coord_dims,
            coord_offsets_full,
            reference_atom,
        )
    }

    /// Build from explicit per-row active-atom index lists. Every active atom is
    /// logit-bearing (the column-separable ordered Beta--Bernoulli chart). For the
    /// reduced SOFTMAX chart use [`Self::from_active_atoms_with_reference`].
    pub(crate) fn from_active_atoms(
        active_atoms: Vec<Vec<usize>>,
        coord_dims: Vec<usize>,
        coord_offsets_full: Vec<usize>,
    ) -> Self {
        Self::from_active_atoms_with_reference(active_atoms, coord_dims, coord_offsets_full, None)
    }

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

    /// Number of FREE logit slots in row `row`'s compact block (the leading
    /// slots). Equals `active_atoms[row].len()` except on a softmax row whose
    /// active set includes the reference atom, where it is one fewer. (#Bug1)
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

    /// #Bug1 — the production entry point: `from_dense_weights` with the softmax
    /// `reference_atom = Some(K−1)`, forcing the REFERENCE atom to be the LARGEST
    /// assignment weight (so it is always selected into the compact active set).
    /// The existing direct-`from_active_atoms_with_reference` test does not
    /// exercise this softmax-weight path, and the sibling `from_dense_weights`
    /// unit test uses the IBP/threshold gate K-slot layout (`reference_atom = None`).
    /// Asserts: the reference atom gets a coord block but NO free logit slot, and
    /// `expand_row` never writes a reference-logit delta into atom 0's coordinate
    /// offset.
    #[test]
    fn from_dense_weights_softmax_reference_largest_weight_has_no_logit_slot() {
        use ndarray::Array1;
        // K=3 softmax reduced chart. Full-q layout = [logit(atom0), logit(atom1),
        // coord(atom0), coord(atom1), coord(atom2)]; atom 2 = reference (K−1).
        let coord_dims = vec![1usize, 1, 1];
        let coord_offsets_full = vec![2usize, 3, 4];
        // Row 0: reference atom 2 is the LARGEST weight (would be selected first).
        // Row 1: reference and atom 0 both large; all three above the relative
        // cutoff so every atom is active (the reference-logit exclusion, not any
        // truncation, is what this test pins).
        let assignments = vec![
            Array1::from_vec(vec![0.1_f64, 0.05, 0.85]),
            Array1::from_vec(vec![0.45_f64, 0.001, 0.55]),
        ];
        let layout = SaeRowLayout::from_dense_weights(
            &assignments,
            /* k_active_cap */ 3,
            /* relative_cutoff */ 1.0e-3,
            coord_dims,
            coord_offsets_full,
            /* reference_atom */ Some(2),
        );
        // The reference atom, though the largest weight, must be active (coord
        // block) but carry NO free logit slot in any row.
        for row in 0..assignments.len() {
            assert!(
                layout.active_atoms[row].contains(&2),
                "row {row}: reference atom must be in the active (coord) set"
            );
            for (j, &k) in layout.logit_atoms[row].iter().enumerate() {
                assert_ne!(
                    k, 2,
                    "row {row} logit slot {j}: reference atom K−1 must have no logit slot"
                );
            }
            // n_logit_active is exactly one fewer than active_atoms when the
            // reference is active.
            assert_eq!(
                layout.n_logit_active(row),
                layout.active_atoms[row].len() - 1,
                "row {row}: reference atom must reduce the free-logit count by one"
            );
        }
        // Row 0 active = {0,1,2}: compact [logit0, logit1, coord0, coord1, coord2].
        // expand_row must place coord0 at full offset 2 (atom 0's coord slot) — the
        // slot the pre-fix code corrupted with a phantom reference logit.
        assert_eq!(layout.active_atoms[0], vec![0, 1, 2]);
        assert_eq!(layout.logit_atoms[0], vec![0, 1]);
        let mut out = vec![0.0_f64; 5];
        layout.expand_row(0, &[1.0, 2.0, 3.0, 4.0, 5.0], &mut out);
        // logit0→full0, logit1→full1, coord0→full2, coord1→full3, coord2→full4.
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        // Explicit: atom 0's coordinate offset (full index 2) holds the coord
        // delta (3.0), NOT a reference-logit delta.
        assert_eq!(
            out[2], 3.0,
            "atom 0 coord slot must not be a reference logit"
        );
    }
}
