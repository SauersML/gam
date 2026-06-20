use super::*;

// The JumpReLU optimization-inclusion band is the single canonical predicate
// `crate::terms::sae::assignment::jumprelu_in_optimization_band`, whose support
// is the machine-precision cutoff `(logit − threshold)/τ > −36` (`σ(−36) ≈
// 2e-16`). The compact Newton active set below MUST use exactly that band so
// every coordinate carrying nonzero sparsity-prior value/gradient/Hessian
// (assembled over the same −36 support in `assignment.rs`, and in the logdet
// third-derivative adjoint in `construction.rs`) has a Newton row to receive
// it. A former module-local copy used a tighter `−4·τ` band, which silently
// dropped coordinates in `(−36τ, −4τ]` from the solve while the prior still put
// gradient on them — an objective↔gradient desync that stalls the inner Newton
// fit. That copy and its `JUMPRELU_REACTIVATION_MARGIN` constant are deleted;
// there is now one band, one source of truth.

/// Per-row active-set layout for sparse SAE assignment (any mode).
///
/// When the assignment is sparse — structurally (JumpReLU gate) or
/// effectively (softmax / IBP-MAP at large `K`, where the assignment mass
/// concentrates on a small support) — only a subset of `K` atoms are active
/// per observation.  The Arrow-Schur row block for observation `i` has dim
/// `q_active_i = |active_atoms_i| + Σ_{k ∈ active_i} d_k` rather than
/// `q = assignment_dim + Σ_k d_k`.  This struct records which atoms are active per row
/// and maps compressed block positions back to full-q positions so that
/// `apply_newton_step` can unpack the compact `delta_t` from the solve.
///
/// For JumpReLU the active set is exactly the gated support
/// (`a_{n,k} ≠ 0`), so the compact solve is identity to the dense solve.
/// For softmax / IBP-MAP the active set is the union of a top-`k_active_cap`
/// truncation and a magnitude cutoff on `a_{n,k}`; this is only enabled when
/// `K` is large enough that the dense `(m_total · p)²` data Gram would not
/// fit the host / device working-set budget, and the dropped atoms carry
/// `O(a_{n,k}²)` curvature that is negligible by construction of the cutoff.
#[derive(Debug, Clone)]
pub struct SaeRowLayout {
    /// `active_atoms[row]` — sorted indices of active atoms for that row.
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
    /// JumpReLU optimization active set: atoms inside the smooth prior's
    /// machine-precision support `(logit - threshold)/tau > -36` (see
    /// [`crate::terms::sae::assignment::jumprelu_in_optimization_band`], the one
    /// canonical band). This is intentionally wider than the hard forward gate
    /// `logit > threshold` so gated-off atoms can remain in the Newton system for
    /// value-consistent prior terms. Their forward reconstruction contribution
    /// and data-fit logit JVP remain hard-zero while `a_k = 0`.
    pub(crate) fn from_jumprelu(
        n: usize,
        k_atoms: usize,
        threshold: f64,
        temperature: f64,
        logits: &Array2<f64>,
        coord_dims: Vec<usize>,
        coord_offsets_full: Vec<usize>,
    ) -> Self {
        let mut per_row = Vec::with_capacity(n);
        for row in 0..n {
            let row_logits = logits.row(row);
            let active: Vec<usize> = (0..k_atoms)
                .filter(|&k| {
                    crate::terms::sae::assignment::jumprelu_in_optimization_band(
                        row_logits[k],
                        threshold,
                        temperature,
                    )
                })
                .collect();
            per_row.push(active);
        }
        Self::from_active_atoms(per_row, coord_dims, coord_offsets_full)
    }

    /// Mode-agnostic effective active set for dense-weight modes (softmax /
    /// IBP-MAP) at large `K`: keep, per row, the top-`k_active_cap` atoms by
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
    pub(crate) fn from_dense_weights(
        assignments: &[Array1<f64>],
        k_active_cap: usize,
        relative_cutoff: f64,
        coord_dims: Vec<usize>,
        coord_offsets_full: Vec<usize>,
    ) -> Self {
        let cap = k_active_cap.max(1);
        let mut per_row = Vec::with_capacity(assignments.len());
        for a in assignments {
            let k = a.len();
            // Rank atoms by descending |a_k|; keep those above cutoff, capped.
            let mut idx: Vec<usize> = (0..k).collect();
            idx.sort_by(|&i, &j| {
                a[j].abs()
                    .partial_cmp(&a[i].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            // Row-relative absolute cutoff = relative_cutoff · this row's peak.
            let row_peak = idx.first().map_or(0.0, |&top| a[top].abs());
            let cutoff = relative_cutoff * row_peak;
            let mut active: Vec<usize> = idx
                .iter()
                .copied()
                .take(cap)
                .filter(|&k_idx| a[k_idx].abs() > cutoff)
                .collect();
            if active.is_empty() {
                // Retain the single largest-magnitude atom so the row block is
                // never empty (a degenerate empty block would zero the row).
                if let Some(&top) = idx.first() {
                    active.push(top);
                }
            }
            active.sort_unstable();
            per_row.push(active);
        }
        Self::from_active_atoms(per_row, coord_dims, coord_offsets_full)
    }

    /// Build from explicit per-row active-atom index lists.
    pub(crate) fn from_active_atoms(
        active_atoms: Vec<Vec<usize>>,
        coord_dims: Vec<usize>,
        coord_offsets_full: Vec<usize>,
    ) -> Self {
        let mut coord_starts_all = Vec::with_capacity(active_atoms.len());
        for active in &active_atoms {
            let mut starts = Vec::with_capacity(active.len());
            let mut cursor = active.len();
            for &k in active {
                starts.push(cursor);
                cursor += coord_dims[k];
            }
            coord_starts_all.push(starts);
        }
        Self {
            active_atoms,
            coord_starts: coord_starts_all,
            coord_offsets_full,
            coord_dims,
        }
    }

    /// Per-row compressed dim.
    pub fn row_q_active(&self, row: usize) -> usize {
        let active = &self.active_atoms[row];
        let coord_sum: usize = active.iter().map(|&k| self.coord_dims[k]).sum();
        active.len() + coord_sum
    }

    /// Expand a compact `delta_t` row slice back into full-q, zeros for inactive.
    pub fn expand_row(&self, row: usize, delta_t_row: &[f64], out: &mut [f64]) {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        let active = &self.active_atoms[row];
        let starts = &self.coord_starts[row];
        for (j, &k) in active.iter().enumerate() {
            out[k] = delta_t_row[j];
            let d = self.coord_dims[k];
            let full_off = self.coord_offsets_full[k];
            for axis in 0..d {
                out[full_off + axis] = delta_t_row[starts[j] + axis];
            }
        }
    }
}
