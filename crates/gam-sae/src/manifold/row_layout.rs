use super::*;

// The JumpReLU optimization-inclusion band is the single canonical predicate
// `crate::assignment::jumprelu_in_optimization_band`, whose support is the
// machine-precision cutoff `(logit − threshold)/τ > −36` (`σ(−36) ≈ 2e-16`).
// That band is the support over which the sparsity prior's value/gradient/
// Hessian are assembled in `assignment.rs` (and the logdet third-derivative
// adjoint in `construction.rs`); it defines where the prior is NONZERO.
//
// It is NOT, however, the Newton active set. Putting every in-band atom in the
// joint Arrow-Schur block made the per-token block `K·(1+d)` (at `threshold=0`,
// `τ=1` with sparse logits essentially every atom is in-band), breaking the
// `O(k_active)` contract. The priors are column-SEPARABLE, so a gated-off atom
// contributes a DIAGONAL-only block and never needs the joint (cross-coupling)
// solve: `from_jumprelu` keeps the hard-gated atoms plus the gated-off atoms
// whose separable contribution exceeds a relative cutoff, and drops the
// negligible deep-band tail (bounded gradient error; identical objective value,
// which the loss sums over the full band regardless of this layout). A tighter
// FIXED `−4·τ` band was tried once and STALLED the fit — it dropped coordinates
// in `(−36τ, −4τ]` from the solve while the prior still put O(1) gradient on
// them (objective↔gradient desync). The cutoff here is on the ACTUAL separable
// gradient magnitude, so an atom is only dropped when its contribution is
// already negligible; that desync cannot recur.

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
/// For JumpReLU the active set is the hard forward gate (`a_{n,k} ≠ 0`) plus any
/// gated-off atom still carrying a non-negligible column-separable prior
/// gradient; below-cutoff gated-off atoms are diagonal-only and dropped (see
/// [`Self::from_jumprelu`]). For IBP-MAP the active set is the union of a
/// top-`k_active_cap`
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
    /// column-separable modes (IBP-MAP / JumpReLU). For SOFTMAX the reduced chart
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

/// Inputs to [`SaeRowLayout::from_jumprelu`], grouped into one bundle so the
/// per-row active-set construction takes a single parameter rather than a long
/// positional argument list. Borrowed matrices (`logits`, `contribution`) stay
/// by reference; the owned `coord_dims` / `coord_offsets_full` are moved on into
/// the resulting layout.
pub(crate) struct JumpReluLayoutParams<'a> {
    /// Number of observation rows.
    pub n: usize,
    /// Number of dictionary atoms.
    pub k_atoms: usize,
    /// JumpReLU hard-gate threshold `θ`.
    pub threshold: f64,
    /// Gate temperature `τ`.
    pub temperature: f64,
    /// `(n × k_atoms)` per-row EFFECTIVE ROUTING logits (frozen-aware): the hard
    /// gate and in-band membership are decided on these, NOT the raw free
    /// `self.logits`. Under frozen routing these are the predicted logits; when not
    /// frozen they equal the free logits. (#Bug3)
    pub logits: &'a Array2<f64>,
    /// `(n × k_atoms)` per-row EFFECTIVE gate mass `a_{n,k}` (data-fit coupling),
    /// NOT the separable prior gradient — the #2071 / #Bug7 block-size contract.
    pub contribution: &'a Array2<f64>,
    /// Per-atom mask (length `k_atoms`) of UNGATED (#1026 background-tier) atoms
    /// whose effective gate is pinned at `1.0` regardless of their logit. These are
    /// force-included in the hard active set — they contribute a nonzero
    /// reconstruction gate, so they MUST be in the compact support. (#Bug3)
    pub ungated: &'a [bool],
    /// Cap on the per-row active-set size.
    pub k_active_cap: usize,
    /// Relative (per-row-peak) cutoff below which gated-off atoms are dropped.
    pub relative_cutoff: f64,
    /// Per-atom coordinate dimensions (length `k_atoms`).
    pub coord_dims: Vec<usize>,
    /// Full-q coordinate offset for each atom (length `k_atoms`).
    pub coord_offsets_full: Vec<usize>,
}

impl SaeRowLayout {
    /// JumpReLU compact active set, sized by the HARD FORWARD GATE plus the
    /// gated-off atoms that still carry a NON-NEGLIGIBLE column-separable prior
    /// contribution — NOT the whole machine-precision optimization band.
    ///
    /// The joint (cross-coupling) Newton row block only ever needs the atoms
    /// that actually couple: the atoms with nonzero assignment mass
    /// (`logit > threshold`, i.e. the hard forward gate `a_k ≠ 0`), whose
    /// data-fit reconstruction Jacobian cross-couples with the decoder and with
    /// every other on atom. Every OTHER in-band atom is gated OFF (`a_k = 0`),
    /// so its reconstruction contribution and data-fit logit JVP are hard-zero
    /// and its ONLY footprint in the Newton system is a DIAGONAL block from the
    /// column-SEPARABLE priors (the assignment sparsity prior on its own logit,
    /// and the ARD prior on its own coords). A diagonal-only atom never needed
    /// the joint block.
    ///
    /// The former layout put EVERY atom in the band
    /// `(logit − threshold)/τ > −36` (see
    /// [`crate::assignment::jumprelu_in_optimization_band`]) into the joint
    /// block. That band is the smooth prior's machine-precision support
    /// (`σ(−36) ≈ 2e-16`), so at `threshold = 0`, `τ = 1` with typical (sparse)
    /// logits essentially ALL `K` atoms qualified and the per-token block was
    /// `K·(1+d)`, violating the `O(k_active)` contract on the very lane meant
    /// for large `K`.
    ///
    /// #Bug7 / #2071: `contribution[row, k]` is the atom's DATA-FIT COUPLING
    /// magnitude — the EFFECTIVE gate mass `a_{n,k}` — NOT its separable prior
    /// gradient. This is the load-bearing choice for the `O(k_active)` block-size
    /// contract: gate mass is EXACTLY zero for every gated-off atom, so the
    /// relative cutoff drops precisely the gated-off tail and the per-token block
    /// stays `k_active·(1+d)` independent of `K`. (Scoring membership by the
    /// separable prior gradient — which is `O(1)` for the many gated-off logits
    /// sitting near the threshold — would re-bloat the block to `K·(1+d)`; see the
    /// `AssignmentMode::ThresholdGate` assembly-plan comment.) A dropped gated-off
    /// atom's separable diagonal prior term changes the assembled gradient by less
    /// than the cutoff and leaves the OBJECTIVE VALUE bit-identical (the loss sums
    /// the full −36 band independently of this layout). Atoms are kept when they
    /// are hard-gated (nonzero effective gate, including every ungated atom) OR
    /// their gate mass exceeds `relative_cutoff · row_peak`, capped at
    /// `k_active_cap` (hard-gated atoms are never dropped; the remaining budget
    /// is filled by the highest-mass gated-off atoms). Dropping an atom
    /// therefore changes the assembled gradient by less than the cutoff, and
    /// leaves the OBJECTIVE VALUE bit-identical (the loss's
    /// `assignment_prior_value` / `ard_value` sum the full −36 band
    /// independently of this layout). This mirrors the softmax / IBP-MAP
    /// [`Self::from_dense_weights`] truncation, which likewise keeps a row's
    /// top-`k` atoms above a relative cutoff and drops the negligible tail.
    pub(crate) fn from_jumprelu(params: JumpReluLayoutParams<'_>) -> Self {
        let JumpReluLayoutParams {
            n,
            k_atoms,
            threshold,
            temperature: _,
            logits,
            contribution,
            ungated,
            k_active_cap,
            relative_cutoff,
            coord_dims,
            coord_offsets_full,
        } = params;
        use std::cmp::Ordering::Equal;
        let cap = k_active_cap.max(1);
        let mut per_row = Vec::with_capacity(n);
        for row in 0..n {
            let row_logits = logits.row(row);
            let row_contrib = contribution.row(row);
            let in_band = |_k: usize| true;
            // Hard forward gate: nonzero EFFECTIVE assignment mass ⇒ data-fit
            // coupling in the joint block. Always retained. `logits` is the
            // effective routing (frozen-aware), and #Bug3: an UNGATED atom has its
            // gate pinned at 1.0 regardless of its logit, so it is always active in
            // the reconstruction and MUST be in the compact support even when its
            // raw logit sits below the threshold.
            let hard: Vec<usize> = (0..k_atoms)
                .filter(|&k| ungated[k] || row_contrib[k] != 0.0)
                .collect();
            // Relative-cutoff base: the largest separable contribution over all
            // in-band atoms in this row.
            let peak = (0..k_atoms)
                .filter(|&k| in_band(k))
                .fold(0.0_f64, |m, k| m.max(row_contrib[k].abs()));
            let cutoff = relative_cutoff * peak;
            // Gated-off, still-in-band atoms whose separable diagonal gradient is
            // non-negligible (> cutoff). The negligible deep-band tail is dropped.
            let mut extra: Vec<usize> = (0..k_atoms)
                .filter(|&k| {
                    row_logits[k] <= threshold && in_band(k) && row_contrib[k].abs() > cutoff
                })
                .collect();
            // Cap the total active set; hard-gated atoms are never dropped, so the
            // gated-off `extra` set absorbs the truncation.
            let budget = cap.saturating_sub(hard.len());
            if extra.len() > budget {
                if budget == 0 {
                    extra.clear();
                } else {
                    extra.select_nth_unstable_by(budget - 1, |&i, &j| {
                        row_contrib[j]
                            .abs()
                            .partial_cmp(&row_contrib[i].abs())
                            .unwrap_or(Equal)
                    });
                    extra.truncate(budget);
                }
            }
            let mut active: Vec<usize> = hard;
            active.extend(extra);
            if active.is_empty() {
                // Never emit an empty row block (a degenerate empty block zeroes
                // the row). Retain the single most-contributing in-band atom, or
                // — if the row has no band atom at all — the largest-logit atom.
                let best = (0..k_atoms)
                    .filter(|&k| in_band(k))
                    .max_by(|&i, &j| {
                        row_contrib[i]
                            .abs()
                            .partial_cmp(&row_contrib[j].abs())
                            .unwrap_or(Equal)
                    })
                    .or_else(|| {
                        (0..k_atoms).max_by(|&i, &j| {
                            row_logits[i].partial_cmp(&row_logits[j]).unwrap_or(Equal)
                        })
                    });
                if let Some(b) = best {
                    active.push(b);
                }
            }
            active.sort_unstable();
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
    /// `reference_atom` is `Some(K−1)` for the reduced SOFTMAX chart (that atom
    /// gets a coord block but no free logit slot) and `None` for IBP-MAP. (#Bug1)
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
    /// logit-bearing (the column-separable IBP-MAP / JumpReLU chart). For the
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
mod jumprelu_hard_gate_tests {
    // #5 — JumpReLU joint block sized by the hard forward gate plus the gated-off
    // atoms with a non-negligible column-separable prior gradient; the negligible
    // deep-band tail is dropped (see [`SaeRowLayout::from_jumprelu`]).
    use super::{JumpReluLayoutParams, SaeRowLayout};
    use crate::assignment::{AssignmentMode, SaeAssignment, assignment_prior_grad_hdiag};
    use crate::manifold::{
        SAE_DENSE_BETA_PENALTY_PROBE_MAX_DIM, SaeAtomBasisKind, SaeManifoldAtom, SaeManifoldRho,
        SaeManifoldTerm,
    };
    use gam_terms::latent::LatentManifold;
    use ndarray::{Array1, Array2, Array3};

    // Exact sparsity-prior separable diagonal gradient magnitude used as the
    // per-atom selection score: |P'(logit)| = λ·σ(1−σ)/τ at (threshold=0, τ=1, λ=1).
    fn logit_slope(logit: f64) -> f64 {
        let a = 1.0 / (1.0 + (-logit).exp());
        a * (1.0 - a)
    }

    /// GATE (i): with logits where most atoms are IN-BAND (all `> −36`) but only
    /// a few are hard-gated or near the threshold, the built block tracks the
    /// hard-gate + near-threshold count `n_active·(1+d)`, NOT `K·(1+d)`. Pins the
    /// exact retained set, the deep-band drop, and the cap truncation.
    #[test]
    fn from_jumprelu_block_size_tracks_hard_gate_not_band() {
        let n = 2usize;
        let k = 6usize;
        let (threshold, temperature) = (0.0_f64, 1.0_f64);
        // Two hard-gated (>0), one near-threshold gated-off (retained), three deep
        // gated-off (in the −36 band but negligible separable gradient → dropped).
        let logits = Array2::from_shape_vec(
            (n, k),
            vec![
                1.0, 0.5, -0.3, -20.0, -25.0, -30.0, // row 0
                0.8, 0.2, -0.4, -22.0, -26.0, -31.0, // row 1
            ],
        )
        .unwrap();
        // Every logit is inside the −36 optimization band, so the OLD full-band
        // layout would put all K=6 atoms in the joint block.
        for &l in logits.iter() {
            assert!(crate::assignment::jumprelu_in_optimization_band(
                l,
                threshold,
                temperature
            ));
        }
        let contribution = logits.mapv(logit_slope);
        let coord_dims = vec![1usize; k];
        // Full-q coord offsets: assignment_dim (=K logits) then one axis per atom.
        let coord_offsets_full: Vec<usize> = (0..k).map(|i| k + i).collect();

        // No cap (cap = K): the relative cutoff is the sole lever.
        let layout = SaeRowLayout::from_jumprelu(JumpReluLayoutParams {
            n,
            k_atoms: k,
            threshold,
            temperature,
            logits: &logits,
            contribution: &contribution,
            ungated: &vec![false; k],
            k_active_cap: k,
            relative_cutoff: 1.0e-3,
            coord_dims: coord_dims.clone(),
            coord_offsets_full: coord_offsets_full.clone(),
        });
        for row in 0..n {
            // Hard-gated {0,1} plus the near-threshold gated-off atom 2; deep atoms
            // {3,4,5} dropped.
            assert_eq!(layout.active_atoms[row], vec![0, 1, 2], "row {row}");
            // Block dim = 3·(1+d) = 6, versus the full-band 6·(1+d) = 12.
            assert_eq!(layout.row_q_active(row), 3 * (1 + 1));
            assert!(layout.row_q_active(row) < k * (1 + 1));
        }

        // Cap = 2: hard-gated atoms are never dropped, so the budget for gated-off
        // atoms is 0 and only the two hard-gated atoms remain.
        let capped = SaeRowLayout::from_jumprelu(JumpReluLayoutParams {
            n,
            k_atoms: k,
            threshold,
            temperature,
            logits: &logits,
            contribution: &contribution,
            ungated: &vec![false; k],
            k_active_cap: 2,
            relative_cutoff: 1.0e-3,
            coord_dims,
            coord_offsets_full,
        });
        for row in 0..n {
            assert_eq!(capped.active_atoms[row], vec![0, 1], "capped row {row}");
        }
    }

    /// GATE (ii) — LOAD-BEARING value/grad parity of the `from_jumprelu`
    /// separable-gradient layout. Assembles the SAME JumpReLU term two ways: dense
    /// full-band (`Some(None)`) and the compact `from_jumprelu` layout
    /// (`Some(Some(layout))`) built from the separable prior gradient. The compact
    /// per-row gradient `gt`, expanded back to full-q, must equal the dense
    /// gradient to tight tolerance — the retained atoms (hard-gated AND the
    /// near-threshold gated-off atom, whose separable sparsity-prior diagonal
    /// `assignment_grad` is the load-bearing term) reproduce it EXACTLY, and the
    /// dropped deep-band atoms differ only by their own (negligible, sub-cutoff)
    /// separable contribution. The objective VALUE is layout-independent by
    /// construction (the loss sums the full band), so value parity is structural.
    ///
    /// #1801 — the production DEFAULT path (`None`) is DISTINCT from this
    /// separable-gradient layout: it sizes the joint block by the HARD FORWARD
    /// GATE only (data-fit coupling `a_k`, hard-zero for gated-off atoms), so it
    /// keeps only the hard-gated atoms — strictly SMALLER than the near-threshold-
    /// retaining `from_jumprelu` block. The per-token cost therefore tracks
    /// `k_active`, independent of K
    /// (`sae_streaming_arrow_schur_contract::per_token_block_dim_is_independent_of_k_at_fixed_active`).
    #[test]
    fn jumprelu_compact_gradient_matches_dense_full_band() {
        let n = 3usize;
        let k = 5usize;
        let p = 2usize;
        let (threshold, temperature) = (0.0_f64, 1.0_f64);
        // Per row: 2 hard-gated (>0), 1 near-threshold gated-off (retained), 2 deep
        // gated-off (in-band but negligible → dropped).
        let logits = Array2::from_shape_vec(
            (n, k),
            vec![
                1.0, 0.3, -0.4, -30.0, -34.0, // row 0
                0.8, 0.1, -0.6, -28.0, -33.0, // row 1
                1.2, 0.5, -0.2, -31.0, -35.0, // row 2
            ],
        )
        .unwrap();
        // Euclidean atoms (identity geometry, so no Riemannian projection mixes
        // the gradient across the scatter); coords at the origin so the ARD
        // separable gradient is zero and `contribution` is purely the sparsity
        // prior — isolating the column-separable term the fix must preserve.
        let atoms: Vec<SaeManifoldAtom> = (0..k)
            .map(|i| {
                let f = (i as f64) + 1.0;
                SaeManifoldAtom::new(
                    format!("atom{i}"),
                    SaeAtomBasisKind::EuclideanPatch,
                    1,
                    Array2::<f64>::from_elem((n, 2), 1.0),
                    Array3::<f64>::zeros((n, 2, 1)),
                    Array2::<f64>::from_shape_vec(
                        (2, p),
                        vec![0.1 * f, -0.2 * f, 0.15 * f, 0.3 * f],
                    )
                    .unwrap(),
                    Array2::<f64>::eye(2),
                )
                .unwrap()
            })
            .collect();
        let coords: Vec<Array2<f64>> = (0..k).map(|_| Array2::<f64>::zeros((n, 1))).collect();
        let manifolds = vec![LatentManifold::Euclidean; k];
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits.clone(),
            coords,
            manifolds,
            AssignmentMode::threshold_gate(temperature, threshold),
        )
        .unwrap();
        let mut term = SaeManifoldTerm::new(atoms, assignment).unwrap();
        let target =
            Array2::<f64>::from_shape_fn((n, p), |(r, c)| 0.1 * (r as f64) - 0.05 * (c as f64));
        // λ_sparse = 1 (log 0), small smoothness, α = 1 ARD on the single axis.
        let rho = SaeManifoldRho::new(0.0, -6.0, vec![Array1::<f64>::from_elem(1, 0.0); k]);

        // Reproduce the production `contribution` (construction.rs): the sparsity
        // prior separable gradient magnitude (ARD is zero at the origin).
        let (assignment_grad, _) = assignment_prior_grad_hdiag(&term.assignment, &rho).unwrap();
        let contribution = Array2::from_shape_fn((n, k), |(r, c)| assignment_grad[r * k + c].abs());
        let coord_dims = vec![1usize; k];
        let coord_offsets = term.assignment.coord_offsets();
        let layout = SaeRowLayout::from_jumprelu(JumpReluLayoutParams {
            n,
            k_atoms: k,
            threshold,
            temperature,
            logits: &logits,
            contribution: &contribution,
            ungated: &vec![false; k],
            k_active_cap: k,
            relative_cutoff: 1.0e-3,
            coord_dims,
            coord_offsets_full: coord_offsets,
        });
        // The near-threshold gated-off atom 2 is retained (non-negligible
        // separable diagonal); the deep atoms 3,4 are dropped.
        for row in 0..n {
            assert_eq!(layout.active_atoms[row], vec![0, 1, 2], "layout row {row}");
        }

        let probe = SAE_DENSE_BETA_PENALTY_PROBE_MAX_DIM;
        let dense = term
            .assemble_arrow_schur_inner(target.view(), &rho, None, 1.0, probe, Some(None))
            .unwrap();
        let compact = term
            .assemble_arrow_schur_inner(
                target.view(),
                &rho,
                None,
                1.0,
                probe,
                Some(Some(layout.clone())),
            )
            .unwrap();
        // DEFAULT path: no override → construction.rs runs the real production
        // `ThresholdGate` wiring. #1801 — that wiring sizes the joint block by the
        // HARD FORWARD GATE only (it scores membership by the data-fit coupling gate
        // mass `a_k`, hard-zero for every gated-off atom), so the production block
        // keeps ONLY the hard-gated atoms {0,1} — dropping the near-threshold gated-
        // off atom 2 that the separable-gradient `from_jumprelu` layout above
        // retains, and every deep atom. The production per-token cost therefore
        // tracks `k_active` (the hard-gate count), independent of K.
        let default = term
            .assemble_arrow_schur_inner(target.view(), &rho, None, 1.0, probe, None)
            .unwrap();
        // Hard-gate reference layout {0,1} (the production block dim).
        let hard_gate = SaeRowLayout::from_active_atoms(
            (0..n).map(|_| vec![0usize, 1usize]).collect(),
            vec![1usize; k],
            term.assignment.coord_offsets(),
        );

        let q = term.assignment.row_block_dim();
        assert_eq!(dense.rows.len(), n);
        assert_eq!(compact.rows.len(), n);
        assert_eq!(default.rows.len(), n);
        let mut max_diff = 0.0_f64;
        let mut saw_drop = false;
        for row in 0..n {
            let dgt = &dense.rows[row].gt;
            assert_eq!(dgt.len(), q, "dense row {row} must be full-q");
            // The compact block is strictly smaller (deep-band atoms dropped).
            assert!(
                compact.rows[row].gt.len() < dgt.len(),
                "row {row}: compact block ({}) must be smaller than dense ({})",
                compact.rows[row].gt.len(),
                dgt.len()
            );
            saw_drop = true;
            // #1801 — the production default path is sized by the HARD FORWARD GATE
            // {0,1}, strictly SMALLER than the separable-gradient `from_jumprelu`
            // compact layout {0,1,2}: the near-threshold gated-off atom 2 (a
            // diagonal-only atom that never data-fit-couples) is dropped from the
            // joint block.
            assert_eq!(
                default.rows[row].gt.len(),
                hard_gate.row_q_active(row),
                "row {row}: default (production) path must be sized by the hard forward gate"
            );
            assert!(
                default.rows[row].gt.len() < compact.rows[row].gt.len(),
                "row {row}: production hard-gate block ({}) must be smaller than the \
                 near-threshold-retaining from_jumprelu block ({})",
                default.rows[row].gt.len(),
                compact.rows[row].gt.len()
            );
            // Expand the compact gradient to full-q and compare to dense.
            let compact_gt: Vec<f64> = compact.rows[row].gt.iter().copied().collect();
            let mut expanded = vec![0.0_f64; q];
            layout.expand_row(row, &compact_gt, &mut expanded);
            for i in 0..q {
                let diff = (expanded[i] - dgt[i]).abs();
                max_diff = max_diff.max(diff);
                assert!(
                    diff < 1.0e-8,
                    "row {row} coord {i}: compact gt {} vs dense {} (diff {diff:e})",
                    expanded[i],
                    dgt[i]
                );
            }
            // Load-bearing: the near-threshold gated-off atom's separable
            // sparsity-prior gradient is present and non-trivial in BOTH assemblies
            // (dropping it would blow past the tolerance).
            assert!(
                dgt[2].abs() > 1.0e-2,
                "row {row}: near-threshold band-only prior gradient must be O(0.1), got {}",
                dgt[2]
            );
        }
        assert!(
            saw_drop,
            "the compact layout must actually drop deep-band atoms"
        );
        // Tolerance is met with margin: dropped atoms' separable contribution
        // (~e^{-28..-35}) is far below the 1e-8 gate.
        assert!(max_diff < 1.0e-8, "max full-q gradient diff {max_diff:e}");
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
    /// unit test uses the IBP/JumpReLU K-slot layout (`reference_atom = None`).
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
