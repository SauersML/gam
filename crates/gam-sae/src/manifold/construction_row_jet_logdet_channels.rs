// [#780 line-count gate] Cohesive per-row jet / reconstruction-channel
// assembly for the streaming-exact arrow log-det, split out of
// `construction.rs` (which sits against the 10k-line gate). These are the
// `SaeManifoldTerm` methods that turn the converged cache into the per-row
// `SaeRowJets` the streaming log-det consumes: the row reconstruction program
// builder, the const-generic reconstruction / β-border channel fills (and
// their dynamic dispatchers), the unified structure-compiled row-jet builder,
// and the bounded tile refill. Included via `include!`
// from `construction.rs` so they keep the SAME module scope (`use super::*`),
// the same `impl SaeManifoldTerm` surface, and full private-field access.

thread_local! {
    /// One reusable packed-jet arena per worker thread. Non-softmax row programs
    /// copy their derivative channels into owned `SaeRowJets` before returning,
    /// so the arena can be reset for the next row with no borrowed state escape.
    static SAE_ROW_JET_ARENA: std::cell::RefCell<gam_math::jet_scalar::DynamicJetArena> =
        std::cell::RefCell::new(gam_math::jet_scalar::DynamicJetArena::new());
}

/// Zero-copy production adapter for the structure-compiled softmax row program.
/// It borrows the term's live basis/decoder tensors and the cache-derived primary
/// layout; no per-row `AtomRowBasisJet` clone is constructed.
struct ProductionSoftmaxRowProgram<'a> {
    term: &'a SaeManifoldTerm,
    row: usize,
    vars: &'a [SaeLocalRowVar],
    assignments: ArrayView1<'a, f64>,
    second_jets: &'a [Array4<f64>],
    border: &'a [SaeBorderChannel],
}

impl ProductionSoftmaxRowProgram<'_> {
    #[inline]
    fn atom_is_active_inner(&self, atom: usize) -> bool {
        self.term
            .last_row_layout
            .as_ref()
            .is_none_or(|layout| layout.active_atoms[self.row].binary_search(&atom).is_ok())
    }
}

impl crate::row_jet_program::SaeSoftmaxRowProgramSource for ProductionSoftmaxRowProgram<'_> {
    fn n_atoms(&self) -> usize {
        self.term.k_atoms()
    }

    fn out_dim(&self) -> usize {
        self.term.output_dim()
    }

    fn n_primaries(&self) -> usize {
        self.vars.len()
    }

    fn primary(&self, slot: usize) -> crate::row_jet_program::SaeRowPrimary {
        match self.vars[slot] {
            SaeLocalRowVar::Logit { atom } => crate::row_jet_program::SaeRowPrimary::Logit { atom },
            SaeLocalRowVar::Coord { atom, axis } => {
                crate::row_jet_program::SaeRowPrimary::Coord { atom, axis }
            }
        }
    }

    fn gate_value(&self, atom: usize) -> f64 {
        self.assignments[atom]
    }

    fn atom_is_active(&self, atom: usize) -> bool {
        self.atom_is_active_inner(atom)
    }

    fn fill_decoded(&self, atom: usize, out: &mut [f64]) {
        if self.atom_is_active_inner(atom) {
            self.term.atoms[atom].fill_decoded_row(self.row, out);
        } else {
            out.fill(0.0);
        }
    }

    fn fill_decoded_first(&self, atom: usize, axis: usize, out: &mut [f64]) {
        if self.atom_is_active_inner(atom) {
            self.term.atoms[atom].fill_decoded_derivative_row(self.row, axis, out);
        } else {
            out.fill(0.0);
        }
    }

    fn fill_decoded_second(&self, atom: usize, axis_a: usize, axis_b: usize, out: &mut [f64]) {
        out.fill(0.0);
        if !self.atom_is_active_inner(atom) {
            return;
        }
        let atom_ref = &self.term.atoms[atom];
        for basis_col in 0..atom_ref.basis_size() {
            let d2phi = self.second_jets[atom][[self.row, basis_col, axis_a, axis_b]];
            if d2phi == 0.0 {
                continue;
            }
            for out_col in 0..atom_ref.output_dim() {
                out[out_col] += d2phi * atom_ref.decoder_coefficients[[basis_col, out_col]];
            }
        }
    }

    fn n_beta_borders(&self) -> usize {
        self.border.len()
    }

    fn beta_border_atom(&self, border: usize) -> usize {
        self.border[border].atom
    }

    fn beta_border_basis_value(&self, border: usize) -> f64 {
        let channel = &self.border[border];
        self.term.atoms[channel.atom].basis_values[[self.row, channel.basis_col]]
    }

    fn beta_border_basis_first(&self, border: usize, axis: usize) -> f64 {
        let channel = &self.border[border];
        self.term.atoms[channel.atom].basis_jacobian[[self.row, channel.basis_col, axis]]
    }

    fn beta_border_output(&self, border: usize) -> &[f64] {
        &self.border[border].output
    }
}

impl SaeManifoldTerm {
    pub(crate) fn reconstruction_row_program_for_logdet(
        &self,
        row: usize,
        vars: &[SaeLocalRowVar],
        assignments: ArrayView1<'_, f64>,
        second_jets: &[Array4<f64>],
    ) -> Result<crate::row_jet_program::SaeReconstructionRowProgram, String> {
        use crate::row_jet_program::{
            AtomRowBasisJet, RowGate, SAE_FIXED_COORD_SLOT, SaeReconstructionRowProgram,
        };

        let p = self.output_dim();
        let k_atoms = self.k_atoms();
        if assignments.len() != k_atoms {
            return Err(format!(
                "reconstruction_row_program_for_logdet: assignments length {} != K={k_atoms}",
                assignments.len()
            ));
        }
        if second_jets.len() != k_atoms {
            return Err(format!(
                "reconstruction_row_program_for_logdet: second_jets length {} != K={k_atoms}",
                second_jets.len()
            ));
        }

        let mut logit_slot = vec![None; k_atoms];
        let mut coord_slot: Vec<Vec<usize>> = self
            .atoms
            .iter()
            .map(|atom| vec![SAE_FIXED_COORD_SLOT; atom.latent_dim()])
            .collect();
        for (slot, var) in vars.iter().enumerate() {
            match *var {
                SaeLocalRowVar::Logit { atom } => {
                    if atom >= k_atoms {
                        return Err(format!(
                            "reconstruction_row_program_for_logdet: logit atom {atom} outside K={k_atoms}"
                        ));
                    }
                    logit_slot[atom] = Some(slot);
                }
                SaeLocalRowVar::Coord { atom, axis } => {
                    if atom >= k_atoms || axis >= coord_slot[atom].len() {
                        return Err(format!(
                            "reconstruction_row_program_for_logdet: coord ({atom},{axis}) outside atom layout"
                        ));
                    }
                    coord_slot[atom][axis] = slot;
                }
            }
        }

        let active_atoms = self
            .last_row_layout
            .as_ref()
            .map(|layout| layout.active_atoms[row].as_slice());
        let atom_is_active = |atom_idx: usize| {
            active_atoms.is_none_or(|active| active.binary_search(&atom_idx).is_ok())
        };
        let atoms: Vec<AtomRowBasisJet> = self
            .atoms
            .iter()
            .enumerate()
            .map(|(atom_idx, atom)| {
                let m = atom.basis_size();
                let d = atom.latent_dim();
                let second = &second_jets[atom_idx];
                AtomRowBasisJet {
                    phi: (0..m)
                        .map(|basis_col| atom.basis_values[[row, basis_col]])
                        .collect(),
                    d_phi: (0..m)
                        .map(|basis_col| {
                            (0..d)
                                .map(|axis| atom.basis_jacobian[[row, basis_col, axis]])
                                .collect()
                        })
                        .collect(),
                    d2_phi: (0..m)
                        .map(|basis_col| {
                            (0..d)
                                .map(|axis_a| {
                                    (0..d)
                                        .map(|axis_b| second[[row, basis_col, axis_a, axis_b]])
                                        .collect()
                                })
                                .collect()
                        })
                        .collect(),
                    decoder: (0..m)
                        .map(|basis_col| {
                            (0..p)
                                .map(|out_col| {
                                    if atom_is_active(atom_idx) {
                                        atom.decoder_coefficients[[basis_col, out_col]]
                                    } else {
                                        0.0
                                    }
                                })
                                .collect()
                        })
                        .collect(),
                    latent_dim: d,
                }
            })
            .collect();

        // Read the ACTIVE routing logits (frozen/amortized when routing is
        // frozen #1033, else the free `self.logits`) — the single source the gate
        // value is derived from. Reading raw `self.assignment.logits` here would
        // re-derive free-logit gates that disagree with the value the assembly
        // used under frozen routing.
        let logits = self.assignment.routing_logits_row(row).to_vec();
        // #1026/#1033 — atoms whose logit is NOT a free Newton parameter (ungated
        // or frozen routing) must gate through a CONSTANT equal to the active
        // routing value (`assignments[k]`), with zero logit derivative, rather
        // than re-derive a gate from a stale/pinned logit. `logit_is_fixed`
        // covers both cases (the same mask the arrow-Schur assembly uses).
        let fixed_gate_value: Vec<Option<f64>> = (0..k_atoms)
            .map(|k| {
                if !atom_is_active(k) {
                    // A compact reconstruction is the fixed-support map
                    // sum_{k in A_i} a_ik g_k.  Dropped atoms are identically
                    // zero functions (including all beta derivatives), even
                    // though their full-softmax probabilities still enter the
                    // normalization and therefore the active gates' logit jets.
                    Some(0.0)
                } else if self.assignment.logit_is_fixed(k) {
                    Some(assignments[k])
                } else {
                    None
                }
            })
            .collect();
        let (gate, gate_shift) = match self.assignment.mode {
            AssignmentMode::Softmax { temperature, .. } => (
                RowGate::Softmax {
                    inv_tau: 1.0 / temperature,
                },
                vec![0.0; k_atoms],
            ),
            AssignmentMode::OrderedBetaBernoulli { temperature, .. } => (
                RowGate::PerAtomLogistic {
                    inv_tau: 1.0 / temperature,
                },
                vec![0.0; k_atoms],
            ),
            AssignmentMode::ThresholdGate {
                temperature,
                threshold,
            } => (
                RowGate::PerAtomLogistic {
                    inv_tau: 1.0 / temperature,
                },
                vec![threshold; k_atoms],
            ),
            // TopK: every atom is `logit_is_fixed`, so `fixed_gate_value`
            // (= the exact {0, 1} support gates) overrides the gate machinery
            // for ALL atoms — these are never-evaluated placeholders.
            AssignmentMode::TopK { .. } => (
                RowGate::PerAtomLogistic { inv_tau: 1.0 },
                vec![0.0; k_atoms],
            ),
        };

        Ok(SaeReconstructionRowProgram {
            atoms,
            gate_value: assignments.to_vec(),
            logits,
            gate_shift,
            gate,
            logit_slot,
            coord_slot,
            fixed_gate_value,
            n_primaries: vars.len(),
        })
    }

    fn fill_reconstruction_channels_from_program_dynamic(
        program: &crate::row_jet_program::SaeReconstructionRowProgram,
        arena: &gam_math::jet_scalar::DynamicJetArena,
        sqrt_row_w: f64,
        channels: &mut crate::row_jet_program::SaeScheduledRowJets,
    ) {
        // Build every output column at once with the per-atom gate / basis jets
        // hoisted out of the column loop (#932 perf): the softmax gate jet and
        // the basis jets are column-independent, so this removes the `out_dim×`
        // redundant recomputation the per-column path incurred (~9× faster at
        // K=8, out_dim=16). Bit-identical to per-column `_packed` assembly.
        let q = program.n_primaries;
        let columns = program.reconstruction_all_columns_dynamic(arena);
        for (out_col, tower) in columns.iter().enumerate() {
            let g = tower.g();
            let h = tower.h();
            for a in 0..q {
                channels.first_mut(a)[out_col] = sqrt_row_w * g[a];
                for b in 0..q {
                    channels.second_mut(a, b)[out_col] = sqrt_row_w * h[a * q + b];
                }
            }
        }
    }

    fn fill_beta_border_channels_from_program_dynamic(
        program: &crate::row_jet_program::SaeReconstructionRowProgram,
        arena: &gam_math::jet_scalar::DynamicJetArena,
        sqrt_row_w: f64,
        border: &[SaeBorderChannel],
        channels: &mut crate::row_jet_program::SaeScheduledRowJets,
    ) {
        let p = program.out_dim();
        // s = ζ_k(ℓ)·Φ_b(t_k) over the local (logit/coord) primaries, built from
        // the SAME gate_tower / basis_tower primitives as the reconstruction
        // column. Build all channels at once with the per-atom gate jet hoisted
        // (#932 perf): border channels sharing an atom reuse one gate jet instead
        // of recomputing it. The reconstruction is LINEAR in β, so this consumer
        // reads only the value (`beta`) and gradient (`beta_deriv` /
        // `beta_l_deriv`) channels — never a Hessian — so the jets are built as
        // first-order `Order1<K>` (value + grad), skipping the K×K Hessian the
        // `Order2` path would compute and discard. `Order1`'s value/grad are
        // bit-identical to `Order2`'s (#1591 order1 oracle).
        let chans = arena.alloc_slice_fill_with(border.len(), |index| {
            let channel = &border[index];
            (channel.atom, channel.basis_col)
        });
        let q = program.n_primaries;
        let sjets = program.beta_border_order1_dynamic(chans, arena);
        for (beta_pos, channel) in border.iter().enumerate() {
            let s = &sjets[beta_pos];
            let s_v = s.v;
            let s_g = s.g();
            for out_col in 0..p {
                let out_c = channel.output[out_col];
                channels.beta_mut(beta_pos)[out_col] = sqrt_row_w * s_v * out_c;
                for a in 0..q {
                    // Reconstruction is linear in β, so beta_deriv and
                    // beta_l_deriv are the identical mixed ∂²ẑ_c/∂β∂p_a channel.
                    let mixed = sqrt_row_w * s_g[a] * out_c;
                    channels.beta_deriv_mut(a, beta_pos)[out_col] = mixed;
                    channels.beta_l_deriv_mut(a, beta_pos)[out_col] = mixed;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests_softmax_hand_reference {
    use super::*;

    impl SaeManifoldTerm {
        /// `∂²g_k/∂t_{ik,axis_a}∂t_{ik,axis_b}` for one row/atom: the decoded second
        /// derivative, packed as `Σ_b ∂²Φ_b·B_{b,c}` over output columns. Recovered
        /// verbatim from 8404ff658^ (the commit before the #932 jet cutover) for the
        /// reinstated hand `row_jets_for_logdet` path.
        fn decoded_second_row(
            atom: &SaeManifoldAtom,
            second_jet: &Array4<f64>,
            row: usize,
            axis_a: usize,
            axis_b: usize,
            out: &mut [f64],
        ) {
            out.fill(0.0);
            for basis_col in 0..atom.basis_size() {
                let d2phi = second_jet[[row, basis_col, axis_a, axis_b]];
                if d2phi == 0.0 {
                    continue;
                }
                for out_col in 0..atom.output_dim() {
                    out[out_col] += d2phi * atom.decoder_coefficients[[basis_col, out_col]];
                }
            }
        }

        /// Historical hand reconstruction + β-border channels for the SOFTMAX
        /// gate, recovered verbatim from 8404ff658^ (before the #932 Taylor-jet
        /// cutover). It was the production path after the generic tower measured
        /// 25–57× slower; it is now test-only as the strongest pre-schedule
        /// performance and correctness baseline.
        ///
        /// The generic jet is retained as an independent oracle: the program
        /// tower (`SaeReconstructionRowProgram::reconstruction_column` /
        /// `reconstruction_all_columns_packed` / `beta_border_tower`) is
        /// cross-checked against this hand
        /// arithmetic to ≤1e-9 (value/grad) / ≤1e-8 (Hessian) by
        /// `sae_row_jet_program_matches_production_row_jets_on_converged_cache` (on a
        /// real converged cache, weighted + unweighted √w arms) and by the
        /// `row_jet_program` unit oracles (incl. the planted-cross-block-sign-flip
        /// #736 guard).
        ///
        /// Softmax-only: the per-atom-logistic (ordered Beta--Bernoulli / ThresholdGate) modes keep the jet
        /// path (their hand gate prior diverged from the live ordered-geometric
        /// prior, so routing them through the jet is the value-preserving choice).
        pub(crate) fn fill_row_jets_hand_softmax_reference(
            &self,
            row: usize,
            vars: &[SaeLocalRowVar],
            assignments: ArrayView1<'_, f64>,
            second_jets: &[Array4<f64>],
            border: &[SaeBorderChannel],
            inv_tau: f64,
            sqrt_row_w: f64,
            first: &mut [Vec<f64>],
            second: &mut [Vec<Vec<f64>>],
            beta: &mut [Vec<f64>],
            beta_deriv: &mut [Vec<Vec<f64>>],
            beta_l_deriv: &mut [Vec<Vec<f64>>],
        ) {
            let p = self.output_dim();
            let q = vars.len();
            let k_atoms = self.k_atoms();
            let active_atoms = self
                .last_row_layout
                .as_ref()
                .map(|layout| layout.active_atoms[row].as_slice());
            let atom_is_active = |atom_idx: usize| {
                active_atoms.is_none_or(|active| active.binary_search(&atom_idx).is_ok())
            };

            // Softmax gate derivatives (closed form; NO exps — the K softmax values
            // `assignments` are precomputed upstream).
            let mut dz = vec![vec![0.0_f64; k_atoms]; q];
            let mut d2z = vec![vec![vec![0.0_f64; k_atoms]; q]; q];
            for (a_idx, var_a) in vars.iter().enumerate() {
                let SaeLocalRowVar::Logit { atom: j } = *var_a else {
                    continue;
                };
                for k in 0..k_atoms {
                    let indicator = if k == j { 1.0 } else { 0.0 };
                    dz[a_idx][k] = assignments[k] * (indicator - assignments[j]) * inv_tau;
                }
            }
            for (a_idx, var_a) in vars.iter().enumerate() {
                let SaeLocalRowVar::Logit { atom: j } = *var_a else {
                    continue;
                };
                for (b_idx, var_b) in vars.iter().enumerate() {
                    let SaeLocalRowVar::Logit { atom: l } = *var_b else {
                        continue;
                    };
                    for k in 0..k_atoms {
                        let ikl = if k == l { 1.0 } else { 0.0 };
                        let ikj = if k == j { 1.0 } else { 0.0 };
                        let ijl = if j == l { 1.0 } else { 0.0 };
                        d2z[a_idx][b_idx][k] = assignments[k]
                            * ((ikl - assignments[l]) * (ikj - assignments[j])
                                - assignments[j] * (ijl - assignments[l]))
                            * inv_tau
                            * inv_tau;
                    }
                }
            }

            // decoded value / first / second derivatives per atom (from the SAME
            // production tensors `basis_values` / `basis_jacobian` / `second_jets` /
            // `decoder_coefficients` the jet reads).
            let mut decoded = vec![vec![0.0_f64; p]; k_atoms];
            let mut d1: Vec<Vec<Vec<f64>>> = self
                .atoms
                .iter()
                .map(|atom| vec![vec![0.0_f64; p]; atom.latent_dim()])
                .collect();
            let mut d2: Vec<Vec<Vec<Vec<f64>>>> = self
                .atoms
                .iter()
                .map(|atom| vec![vec![vec![0.0_f64; p]; atom.latent_dim()]; atom.latent_dim()])
                .collect();
            let mut scratch = vec![0.0_f64; p];
            for k in 0..k_atoms {
                if !atom_is_active(k) {
                    continue;
                }
                self.atoms[k].fill_decoded_row(row, &mut decoded[k]);
                for axis in 0..self.atoms[k].latent_dim() {
                    self.atoms[k].fill_decoded_derivative_row(row, axis, &mut d1[k][axis]);
                }
                for axis_a in 0..self.atoms[k].latent_dim() {
                    for axis_b in 0..self.atoms[k].latent_dim() {
                        Self::decoded_second_row(
                            &self.atoms[k],
                            &second_jets[k],
                            row,
                            axis_a,
                            axis_b,
                            &mut scratch,
                        );
                        d2[k][axis_a][axis_b].clone_from_slice(&scratch);
                    }
                }
            }

            // first channel: ∂ẑ_c/∂ℓ_j = Σ_k dz[j][k]·decoded[k][c] (logit primary);
            // ∂ẑ_c/∂t_{k,axis} = ζ_k·d1[k][axis][c] (coord primary). √w-scaled.
            for (idx, var) in vars.iter().enumerate() {
                match *var {
                    SaeLocalRowVar::Logit { .. } => {
                        for k in 0..k_atoms {
                            if !atom_is_active(k) {
                                continue;
                            }
                            let coeff = dz[idx][k] * sqrt_row_w;
                            if coeff == 0.0 {
                                continue;
                            }
                            for out_col in 0..p {
                                first[idx][out_col] += coeff * decoded[k][out_col];
                            }
                        }
                    }
                    SaeLocalRowVar::Coord { atom, axis } => {
                        let coeff = assignments[atom] * sqrt_row_w;
                        for out_col in 0..p {
                            first[idx][out_col] = coeff * d1[atom][axis][out_col];
                        }
                    }
                }
            }

            // second channel — block-sparse: the cross-atom coord×coord blocks are
            // structural zeros and are NOT computed (the hand form's advantage over
            // the jet's dense K×K Hessian).
            for a in 0..q {
                for b in 0..q {
                    match (vars[a], vars[b]) {
                        (SaeLocalRowVar::Logit { .. }, SaeLocalRowVar::Logit { .. }) => {
                            for k in 0..k_atoms {
                                if !atom_is_active(k) {
                                    continue;
                                }
                                let coeff = d2z[a][b][k] * sqrt_row_w;
                                if coeff == 0.0 {
                                    continue;
                                }
                                for out_col in 0..p {
                                    second[a][b][out_col] += coeff * decoded[k][out_col];
                                }
                            }
                        }
                        (SaeLocalRowVar::Logit { .. }, SaeLocalRowVar::Coord { atom, axis }) => {
                            let coeff = dz[a][atom] * sqrt_row_w;
                            for out_col in 0..p {
                                second[a][b][out_col] = coeff * d1[atom][axis][out_col];
                            }
                        }
                        (SaeLocalRowVar::Coord { atom, axis }, SaeLocalRowVar::Logit { .. }) => {
                            let coeff = dz[b][atom] * sqrt_row_w;
                            for out_col in 0..p {
                                second[a][b][out_col] = coeff * d1[atom][axis][out_col];
                            }
                        }
                        (
                            SaeLocalRowVar::Coord {
                                atom: atom_a,
                                axis: axis_a,
                            },
                            SaeLocalRowVar::Coord {
                                atom: atom_b,
                                axis: axis_b,
                            },
                        ) if atom_a == atom_b => {
                            let coeff = assignments[atom_a] * sqrt_row_w;
                            for out_col in 0..p {
                                second[a][b][out_col] = coeff * d2[atom_a][axis_a][axis_b][out_col];
                            }
                        }
                        _ => {}
                    }
                }
            }

            // β BORDER CHANNELS: one free decoder coefficient whose per-row
            // contribution to output column `c` is ζ_k(ℓ)·Φ_b(t_k)·output_c — linear
            // in β. `beta` is the value channel; `beta_deriv` / `beta_l_deriv` are the
            // identical mixed ∂²ẑ_c/∂β∂p_a channel (both filled the same because the
            // map is linear in β).
            for (beta_pos, channel) in border.iter().enumerate() {
                let atom = channel.atom;
                if !atom_is_active(atom) {
                    continue;
                }
                let phi = self.atoms[atom].basis_values[[row, channel.basis_col]];
                let base = assignments[atom] * phi * sqrt_row_w;
                for out_col in 0..p {
                    beta[beta_pos][out_col] = base * channel.output[out_col];
                }
                for (var_idx, var) in vars.iter().enumerate() {
                    let scalar = match *var {
                        SaeLocalRowVar::Logit { .. } => dz[var_idx][atom] * phi * sqrt_row_w,
                        SaeLocalRowVar::Coord {
                            atom: coord_atom,
                            axis,
                        } if coord_atom == atom => {
                            assignments[atom]
                                * self.atoms[atom].basis_jacobian[[row, channel.basis_col, axis]]
                                * sqrt_row_w
                        }
                        _ => 0.0,
                    };
                    if scalar != 0.0 {
                        for out_col in 0..p {
                            beta_deriv[var_idx][beta_pos][out_col] =
                                scalar * channel.output[out_col];
                        }
                    }
                    let scalar_l = match *var {
                        SaeLocalRowVar::Logit { .. } => {
                            dz[var_idx][atom]
                                * self.atoms[atom].basis_values[[row, channel.basis_col]]
                                * sqrt_row_w
                        }
                        SaeLocalRowVar::Coord {
                            atom: coord_atom,
                            axis,
                        } if coord_atom == atom => {
                            assignments[atom]
                                * self.atoms[atom].basis_jacobian[[row, channel.basis_col, axis]]
                                * sqrt_row_w
                        }
                        _ => 0.0,
                    };
                    if scalar_l != 0.0 {
                        for out_col in 0..p {
                            beta_l_deriv[var_idx][beta_pos][out_col] =
                                scalar_l * channel.output[out_col];
                        }
                    }
                }
            }
        }
    }
}

impl SaeManifoldTerm {
    pub(crate) fn row_jets_for_logdet(
        &self,
        row: usize,
        vars: Vec<SaeLocalRowVar>,
        assignments: ArrayView1<'_, f64>,
        second_jets: &[Array4<f64>],
        border: &[SaeBorderChannel],
    ) -> Result<SaeRowJets, String> {
        let p = self.output_dim();
        let q = vars.len();
        let sqrt_row_w = self
            .row_loss_weights
            .as_deref()
            .map_or(1.0, |w| w[row].sqrt());
        let channels = match self.assignment.mode {
            AssignmentMode::Softmax { temperature, .. } => {
                // Structure-compiled unified row program: the borrowed adapter
                // reads the same live tensors as the former hand kernel, while
                // `execute_softmax_row_program` derives all channels from one
                // sparse softmax-moment schedule.  The generic Tower remains an
                // independent exact oracle; no copied basis/decoder program and
                // no dense structural-zero jet are built on this hot path.
                let inv_tau = 1.0 / temperature;
                let source = ProductionSoftmaxRowProgram {
                    term: self,
                    row,
                    vars: &vars,
                    assignments,
                    second_jets,
                    border,
                };
                let scheduled = crate::row_jet_program::execute_softmax_row_program(
                    &source, inv_tau, sqrt_row_w,
                );
                scheduled
            }
            AssignmentMode::OrderedBetaBernoulli { .. }
            | AssignmentMode::ThresholdGate { .. }
            | AssignmentMode::TopK { .. } => {
                // PER-ATOM modes keep the jet path: value-preserving (their hand
                // gate prior diverged from the live ordered-geometric prior, and
                // the structure-compiled softmax schedule does not apply to
                // these gate graphs). TopK is the degenerate member: its gates
                // are constants {0, 1} with NO logit variables in the row block
                // (`assignment_coord_dim() == 0`), so the program simply carries
                // no gate channels.
                let program = self.reconstruction_row_program_for_logdet(
                    row,
                    &vars,
                    assignments,
                    second_jets,
                )?;
                let mut channels =
                    crate::row_jet_program::SaeScheduledRowJets::zeros(q, p, border.len());
                SAE_ROW_JET_ARENA.with(|cell| {
                    let mut arena = cell.borrow_mut();
                    arena.reset();
                    Self::fill_reconstruction_channels_from_program_dynamic(
                        &program,
                        &arena,
                        sqrt_row_w,
                        &mut channels,
                    );
                    Self::fill_beta_border_channels_from_program_dynamic(
                        &program,
                        &arena,
                        sqrt_row_w,
                        border,
                        &mut channels,
                    );
                });
                channels
            }
        };

        Ok(SaeRowJets { vars, channels })
    }
}

impl SaeManifoldTerm {
    /// Refill the bounded look-ahead window through the authoritative complete
    /// row-jet batch seam. Softmax rows with a common packed width are evaluated
    /// in a memory-ledgered CUDA tile when the calibrated policy admits it; all
    /// logdet/HVP consumers share this refill, so no consumer can accidentally
    /// retain the former host-only coordinate-channel path. Non-softmax gates
    /// continue through their distinct dynamic row program one row at a time.
    fn refill_jet_window(
        &self,
        start: usize,
        cache: &ArrowFactorCache,
        second_jets: &[Array4<f64>],
        border: &[SaeBorderChannel],
        window: &mut std::collections::VecDeque<SaeRowJets>,
    ) -> Result<usize, String> {
        if let AssignmentMode::Softmax { temperature, .. } = self.assignment.mode {
            let q = cache.row_dims[start];
            let same_shape_rows = cache.row_dims[start..]
                .iter()
                .take_while(|&&candidate| candidate == q)
                .count();
            let plan = crate::gpu_kernels::sae_rowjet::plan_softmax_row_jets(
                same_shape_rows,
                self.k_atoms(),
                q,
                self.output_dim(),
                border.len(),
                self.gpu_policy,
            )?;
            let tile_rows = plan.tile_rows;
            if tile_rows == 0 {
                return Err(format!(
                    "complete SAE row-jet planner returned an empty tile at nonempty row {start}"
                ));
            }
            let mut inputs = Vec::with_capacity(tile_rows);
            let mut layouts = Vec::with_capacity(tile_rows);
            let mut assignments = Array1::<f64>::zeros(self.k_atoms());
            let mut shared_beta_layout = None;
            for row in start..start + tile_rows {
                let vars = self.row_vars_for_cache_row(row, cache)?;
                self.assignment.try_assignments_row_into(
                    row,
                    assignments.as_slice_mut().ok_or_else(|| {
                        "complete SAE row-jet assignment scratch is not contiguous".to_string()
                    })?,
                )?;
                let source = ProductionSoftmaxRowProgram {
                    term: self,
                    row,
                    vars: &vars,
                    assignments: assignments.view(),
                    second_jets,
                    border,
                };
                let sqrt_row_weight = self
                    .row_loss_weights
                    .as_deref()
                    .map_or(1.0, |weights| weights[row].sqrt());
                let input = crate::gpu_kernels::sae_rowjet::SaeSoftmaxRowJetInput::from_source(
                    &source,
                    sqrt_row_weight,
                    shared_beta_layout.clone(),
                )?;
                shared_beta_layout = Some((input.beta_atoms.clone(), input.beta_outputs.clone()));
                inputs.push(input);
                layouts.push(vars);
            }
            let channels = crate::gpu_kernels::sae_rowjet::execute_softmax_row_jet_tile(
                &inputs,
                1.0 / temperature,
                plan.path,
            )?;
            let scheduled = channels.into_scheduled_rows();
            for (vars, channels) in layouts.into_iter().zip(scheduled) {
                window.push_back(SaeRowJets { vars, channels });
            }
            return Ok(start + tile_rows);
        }

        let vars = self.row_vars_for_cache_row(start, cache)?;
        let mut a = Array1::<f64>::zeros(self.k_atoms());
        self.assignment.try_assignments_row_into(
            start,
            a.as_slice_mut().ok_or_else(|| {
                "SAE scalar row-jet assignment scratch is not contiguous".to_string()
            })?,
        )?;
        let jets = self.row_jets_for_logdet(start, vars, a.view(), second_jets, border)?;
        window.push_back(jets);
        Ok(start + 1)
    }

    /// #2304 resident IFT RHS for softmax gates: evaluate
    /// `t[row][a] = ⟨first(row,a,·), probe_row⟩` and
    /// `beta_out[row][c] = ⟨beta(row,c,·), probe_row⟩` through the contracted
    /// row-jet seam, never materializing the packed channel tensors. The
    /// per-row probe is supplied by the caller (the masked, √w-scaled target
    /// column block, with any whitening metric already folded in as
    /// `M_n v = U_n(U_nᵀ v)` — exactly the consumer's former
    /// `⟨U_nᵀ jet, U_nᵀ v⟩` dot). Rows are processed in the same
    /// memory-ledgered same-shape tiles as [`Self::refill_jet_window`]; the
    /// planner still owns the CPU/device choice, and the CPU path reduces the
    /// identical authoritative row program in the identical dot order.
    ///
    /// `emit` receives `(row, q, t_row, beta_row)` for each processed row,
    /// where `t_row` has length `q` and `beta_row` has length `border.len()`.
    fn contracted_softmax_linear_rhs(
        &self,
        cache: &ArrowFactorCache,
        second_jets: &[Array4<f64>],
        border: &[SaeBorderChannel],
        mut probe_for_row: impl FnMut(usize) -> Result<Vec<f64>, String>,
        mut emit: impl FnMut(usize, usize, &[f64], &[f64]) -> Result<(), String>,
    ) -> Result<(), String> {
        let AssignmentMode::Softmax { temperature, .. } = self.assignment.mode else {
            return Err("contracted softmax row-jet RHS called on a non-softmax gate".to_string());
        };
        let n = self.n_obs();
        let p = self.output_dim();
        let n_beta = border.len();
        let mut assignments = Array1::<f64>::zeros(self.k_atoms());
        let mut start = 0usize;
        while start < n {
            let q = cache.row_dims[start];
            let same_shape_rows = cache.row_dims[start..]
                .iter()
                .take_while(|&&candidate| candidate == q)
                .count();
            let plan = crate::gpu_kernels::sae_rowjet::plan_softmax_row_jets_contracted(
                same_shape_rows,
                self.k_atoms(),
                q,
                p,
                n_beta,
                self.gpu_policy,
            )?;
            let tile_rows = plan.tile_rows;
            if tile_rows == 0 {
                return Err(format!(
                    "contracted SAE row-jet planner returned an empty tile at nonempty row {start}"
                ));
            }
            let mut inputs = Vec::with_capacity(tile_rows);
            let mut probe = Vec::with_capacity(tile_rows * p);
            let mut shared_beta_layout = None;
            for row in start..start + tile_rows {
                let vars = self.row_vars_for_cache_row(row, cache)?;
                self.assignment.try_assignments_row_into(
                    row,
                    assignments.as_slice_mut().ok_or_else(|| {
                        "contracted SAE row-jet assignment scratch is not contiguous".to_string()
                    })?,
                )?;
                let source = ProductionSoftmaxRowProgram {
                    term: self,
                    row,
                    vars: &vars,
                    assignments: assignments.view(),
                    second_jets,
                    border,
                };
                let sqrt_row_weight = self
                    .row_loss_weights
                    .as_deref()
                    .map_or(1.0, |weights| weights[row].sqrt());
                let input = crate::gpu_kernels::sae_rowjet::SaeSoftmaxRowJetInput::from_source(
                    &source,
                    sqrt_row_weight,
                    shared_beta_layout.clone(),
                )?;
                shared_beta_layout = Some((input.beta_atoms.clone(), input.beta_outputs.clone()));
                inputs.push(input);
                let probe_row = probe_for_row(row)?;
                if probe_row.len() != p {
                    return Err(format!(
                        "contracted SAE row-jet probe for row {row} has length {}; expected {p}",
                        probe_row.len()
                    ));
                }
                probe.extend_from_slice(&probe_row);
            }
            let tile = crate::gpu_kernels::sae_rowjet::execute_softmax_row_jet_tile_contracted(
                &inputs,
                1.0 / temperature,
                plan.path,
                crate::gpu_kernels::sae_rowjet::SaeRowJetContraction::Linear { probe: &probe },
            )?;
            if tile.n_rows != tile_rows || tile.q != q || tile.n_beta != n_beta {
                return Err(format!(
                    "contracted SAE row-jet tile returned shape ({}, {}, {}); expected ({tile_rows}, {q}, {n_beta})",
                    tile.n_rows, tile.q, tile.n_beta
                ));
            }
            for (local, row) in (start..start + tile_rows).enumerate() {
                emit(
                    row,
                    q,
                    &tile.t[local * q..(local + 1) * q],
                    &tile.beta[local * n_beta..(local + 1) * n_beta],
                )?;
            }
            start += tile_rows;
        }
        Ok(())
    }

    /// #2304 resident residual-curvature HVP for softmax gates: the bilinear
    /// contraction
    ///
    /// `t[row][a]    = Σ_b ⟨probe_row, second(a,b,·)⟩ v_t[row][b]
    ///              + Σ_c ⟨probe_row, mixed(a,c,·)⟩ v_beta[c]`
    /// `beta[row][c] = Σ_a ⟨probe_row, mixed(a,c,·)⟩ v_t[row][a]`
    ///
    /// evaluated through the contracted row-jet seam with the (metric-applied,
    /// √w-scaled) residual as the probe. `v_beta_row` is the border-ordered
    /// gather of the direction's β block, identical for every row. The same
    /// tile plan, CPU/device dispatch, and shape checks as
    /// [`Self::contracted_softmax_linear_rhs`] apply.
    fn contracted_softmax_bilinear_hvp(
        &self,
        cache: &ArrowFactorCache,
        second_jets: &[Array4<f64>],
        border: &[SaeBorderChannel],
        mut probe_for_row: impl FnMut(usize) -> Result<Vec<f64>, String>,
        mut v_t_for_row: impl FnMut(usize, usize) -> Result<Vec<f64>, String>,
        v_beta_row: &[f64],
        mut emit: impl FnMut(usize, usize, &[f64], &[f64]) -> Result<(), String>,
    ) -> Result<(), String> {
        let AssignmentMode::Softmax { temperature, .. } = self.assignment.mode else {
            return Err("contracted softmax row-jet HVP called on a non-softmax gate".to_string());
        };
        let n = self.n_obs();
        let p = self.output_dim();
        let n_beta = border.len();
        if v_beta_row.len() != n_beta {
            return Err(format!(
                "contracted SAE row-jet v_beta has length {}; expected {n_beta}",
                v_beta_row.len()
            ));
        }
        let mut assignments = Array1::<f64>::zeros(self.k_atoms());
        let mut start = 0usize;
        while start < n {
            let q = cache.row_dims[start];
            let same_shape_rows = cache.row_dims[start..]
                .iter()
                .take_while(|&&candidate| candidate == q)
                .count();
            let plan = crate::gpu_kernels::sae_rowjet::plan_softmax_row_jets_contracted(
                same_shape_rows,
                self.k_atoms(),
                q,
                p,
                n_beta,
                self.gpu_policy,
            )?;
            let tile_rows = plan.tile_rows;
            if tile_rows == 0 {
                return Err(format!(
                    "contracted SAE row-jet planner returned an empty tile at nonempty row {start}"
                ));
            }
            let mut inputs = Vec::with_capacity(tile_rows);
            let mut probe = Vec::with_capacity(tile_rows * p);
            let mut v_t = Vec::with_capacity(tile_rows * q);
            let mut v_beta = Vec::with_capacity(tile_rows * n_beta);
            let mut shared_beta_layout = None;
            for row in start..start + tile_rows {
                let vars = self.row_vars_for_cache_row(row, cache)?;
                self.assignment.try_assignments_row_into(
                    row,
                    assignments.as_slice_mut().ok_or_else(|| {
                        "contracted SAE row-jet assignment scratch is not contiguous".to_string()
                    })?,
                )?;
                let source = ProductionSoftmaxRowProgram {
                    term: self,
                    row,
                    vars: &vars,
                    assignments: assignments.view(),
                    second_jets,
                    border,
                };
                let sqrt_row_weight = self
                    .row_loss_weights
                    .as_deref()
                    .map_or(1.0, |weights| weights[row].sqrt());
                let input = crate::gpu_kernels::sae_rowjet::SaeSoftmaxRowJetInput::from_source(
                    &source,
                    sqrt_row_weight,
                    shared_beta_layout.clone(),
                )?;
                shared_beta_layout = Some((input.beta_atoms.clone(), input.beta_outputs.clone()));
                inputs.push(input);
                let probe_row = probe_for_row(row)?;
                if probe_row.len() != p {
                    return Err(format!(
                        "contracted SAE row-jet probe for row {row} has length {}; expected {p}",
                        probe_row.len()
                    ));
                }
                probe.extend_from_slice(&probe_row);
                let v_t_row = v_t_for_row(row, q)?;
                if v_t_row.len() != q {
                    return Err(format!(
                        "contracted SAE row-jet v_t for row {row} has length {}; expected {q}",
                        v_t_row.len()
                    ));
                }
                v_t.extend_from_slice(&v_t_row);
                v_beta.extend_from_slice(v_beta_row);
            }
            let tile = crate::gpu_kernels::sae_rowjet::execute_softmax_row_jet_tile_contracted(
                &inputs,
                1.0 / temperature,
                plan.path,
                crate::gpu_kernels::sae_rowjet::SaeRowJetContraction::Bilinear {
                    probe: &probe,
                    v_t: &v_t,
                    v_beta: &v_beta,
                },
            )?;
            if tile.n_rows != tile_rows || tile.q != q || tile.n_beta != n_beta {
                return Err(format!(
                    "contracted SAE row-jet tile returned shape ({}, {}, {}); expected ({tile_rows}, {q}, {n_beta})",
                    tile.n_rows, tile.q, tile.n_beta
                ));
            }
            for (local, row) in (start..start + tile_rows).enumerate() {
                emit(
                    row,
                    q,
                    &tile.t[local * q..(local + 1) * q],
                    &tile.beta[local * n_beta..(local + 1) * n_beta],
                )?;
            }
            start += tile_rows;
        }
        Ok(())
    }
}
