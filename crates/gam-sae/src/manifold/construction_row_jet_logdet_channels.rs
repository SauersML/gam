// [#780 line-count gate] Cohesive per-row jet / reconstruction-channel
// assembly for the streaming-exact arrow log-det, split out of
// `construction.rs` (which sits against the 10k-line gate). These are the
// `SaeManifoldTerm` methods that turn the converged cache into the per-row
// `SaeRowJets` the streaming log-det consumes: the row reconstruction program
// builder, the const-generic reconstruction / β-border channel fills (and
// their dynamic dispatchers), the scalar and 4-row-SIMD-batch row-jet
// builders, and the bounded look-ahead window refill. Included via `include!`
// from `construction.rs` so they keep the SAME module scope (`use super::*`),
// the same `impl SaeManifoldTerm` surface, and full private-field access.

thread_local! {
    /// One reusable packed-jet arena per worker thread. Non-softmax row programs
    /// copy their derivative channels into owned `SaeRowJets` before returning,
    /// so the arena can be reset for the next row with no borrowed state escape.
    static SAE_ROW_JET_ARENA: std::cell::RefCell<gam_math::jet_scalar::DynamicJetArena> =
        std::cell::RefCell::new(gam_math::jet_scalar::DynamicJetArena::new());
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
            .map(|atom| vec![SAE_FIXED_COORD_SLOT; atom.latent_dim])
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
                let d = atom.latent_dim;
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
        first: &mut [Vec<f64>],
        second: &mut [Vec<Vec<f64>>],
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
                first[a][out_col] = sqrt_row_w * g[a];
                for b in 0..q {
                    second[a][b][out_col] = sqrt_row_w * h[a * q + b];
                }
            }
        }
    }

    fn fill_beta_border_channels_from_program_dynamic(
        program: &crate::row_jet_program::SaeReconstructionRowProgram,
        arena: &gam_math::jet_scalar::DynamicJetArena,
        sqrt_row_w: f64,
        border: &[SaeBorderChannel],
        beta: &mut [Vec<f64>],
        beta_deriv: &mut [Vec<Vec<f64>>],
        beta_l_deriv: &mut [Vec<Vec<f64>>],
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
                beta[beta_pos][out_col] = sqrt_row_w * s_v * out_c;
                for a in 0..q {
                    // Reconstruction is linear in β, so beta_deriv and
                    // beta_l_deriv are the identical mixed ∂²ẑ_c/∂β∂p_a channel.
                    let mixed = sqrt_row_w * s_g[a] * out_c;
                    beta_deriv[a][beta_pos][out_col] = mixed;
                    beta_l_deriv[a][beta_pos][out_col] = mixed;
                }
            }
        }
    }

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

    /// HAND reconstruction + β-border channels for the SOFTMAX gate — the
    /// closed-form arithmetic recovered VERBATIM from 8404ff658^ (the commit
    /// before the #932 Taylor-jet cutover). The jet that briefly replaced this is
    /// a measured 25–57× throughput regression on the REML/log-det trace loop
    /// (two independent standalone audits, bit-identical to ≤1.4e-15; see
    /// `scratchpad/sae_recon_bench.rs`), so the hand form is reinstated as the
    /// production path for the dominant softmax mode.
    ///
    /// The jet is RETAINED as the bit-identity oracle, NOT deleted: the program
    /// tower (`SaeReconstructionRowProgram::reconstruction_column` /
    /// `reconstruction_all_columns_packed` / `beta_border_tower`, plus the SIMD
    /// `reconstruction_all_columns_batch4`) is cross-checked against this hand
    /// arithmetic to ≤1e-9 (value/grad) / ≤1e-8 (Hessian) by
    /// `sae_row_jet_program_matches_production_row_jets_on_converged_cache` (on a
    /// real converged cache, weighted + unweighted √w arms) and by the
    /// `row_jet_program` unit oracles (incl. the planted-cross-block-sign-flip
    /// #736 guard) — keeping this single-source hand path guarded against the
    /// forgotten-channel bug class.
    ///
    /// Softmax-only: the per-atom-logistic (ordered Beta--Bernoulli / JumpReLU) modes keep the jet
    /// path (their hand gate prior diverged from the live ordered-geometric
    /// prior, so routing them through the jet is the value-preserving choice).
    fn fill_row_jets_hand_softmax(
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
            .map(|atom| vec![vec![0.0_f64; p]; atom.latent_dim])
            .collect();
        let mut d2: Vec<Vec<Vec<Vec<f64>>>> = self
            .atoms
            .iter()
            .map(|atom| vec![vec![vec![0.0_f64; p]; atom.latent_dim]; atom.latent_dim])
            .collect();
        let mut scratch = vec![0.0_f64; p];
        for k in 0..k_atoms {
            if !atom_is_active(k) {
                continue;
            }
            self.atoms[k].fill_decoded_row(row, &mut decoded[k]);
            for axis in 0..self.atoms[k].latent_dim {
                self.atoms[k].fill_decoded_derivative_row(row, axis, &mut d1[k][axis]);
            }
            for axis_a in 0..self.atoms[k].latent_dim {
                for axis_b in 0..self.atoms[k].latent_dim {
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
                        beta_deriv[var_idx][beta_pos][out_col] = scalar * channel.output[out_col];
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
        let mut first = vec![vec![0.0_f64; p]; q];
        let mut second = vec![vec![vec![0.0_f64; p]; q]; q];
        let mut beta = vec![vec![0.0_f64; p]; border.len()];
        let mut beta_deriv = vec![vec![vec![0.0_f64; p]; border.len()]; q];
        let mut beta_l_deriv = vec![vec![vec![0.0_f64; p]; border.len()]; q];

        match self.assignment.mode {
            AssignmentMode::Softmax { temperature, .. } => {
                // HAND PATH (#932 revert): closed-form reconstruction + β-border
                // channels, ~25–57× faster than the Tower jet it replaced and
                // bit-identical (≤1.4e-15). The jet is retained as the oracle that
                // pins this arithmetic (see `fill_row_jets_hand_softmax`).
                let inv_tau = 1.0 / temperature;
                self.fill_row_jets_hand_softmax(
                    row,
                    &vars,
                    assignments,
                    second_jets,
                    border,
                    inv_tau,
                    sqrt_row_w,
                    &mut first,
                    &mut second,
                    &mut beta,
                    &mut beta_deriv,
                    &mut beta_l_deriv,
                );
            }
            AssignmentMode::OrderedBetaBernoulli { .. }
            | AssignmentMode::ThresholdGate { .. }
            | AssignmentMode::TopK { .. } => {
                // PER-ATOM modes keep the jet path: value-preserving (their hand
                // gate prior diverged from the live ordered-geometric prior, and
                // the batched SIMD speedup that motivated the revert is
                // softmax-only anyway). TopK is the degenerate member: its gates
                // are constants {0, 1} with NO logit variables in the row block
                // (`assignment_coord_dim() == 0`), so the program simply carries
                // no gate channels.
                let program = self.reconstruction_row_program_for_logdet(
                    row,
                    &vars,
                    assignments,
                    second_jets,
                )?;
                SAE_ROW_JET_ARENA.with(|cell| {
                    let mut arena = cell.borrow_mut();
                    arena.reset();
                    Self::fill_reconstruction_channels_from_program_dynamic(
                        &program,
                        &arena,
                        sqrt_row_w,
                        &mut first,
                        &mut second,
                    );
                    Self::fill_beta_border_channels_from_program_dynamic(
                        &program,
                        &arena,
                        sqrt_row_w,
                        border,
                        &mut beta,
                        &mut beta_deriv,
                        &mut beta_l_deriv,
                    );
                });
            }
        }

        Ok(SaeRowJets {
            vars,
            first,
            second,
            beta,
            beta_deriv,
            beta_l_deriv,
        })
    }
}

impl SaeManifoldTerm {
    /// Refill the bounded look-ahead jet window with the next row's
    /// [`SaeRowJets`], built by the hand `row_jets_for_logdet`. Returns the next
    /// unbuilt row index.
    ///
    /// #932 revert: the previous 4-row SIMD-batch fast path
    /// (`row_jets_for_logdet_batch4`) is a 25–57× throughput regression versus
    /// the hand closed form, so production builds one hand row per refill. The
    /// window machinery is retained (the call sites still drain one row at a
    /// time); `cache` stays in the signature for `row_vars_for_cache_row`.
    fn refill_jet_window(
        &self,
        start: usize,
        cache: &ArrowFactorCache,
        second_jets: &[Array4<f64>],
        border: &[SaeBorderChannel],
        window: &mut std::collections::VecDeque<SaeRowJets>,
    ) -> Result<usize, String> {
        let vars = self.row_vars_for_cache_row(start, cache)?;
        let mut a = Array1::<f64>::zeros(self.k_atoms());
        self.assignment.try_assignments_row_into(
            start,
            a.as_slice_mut().expect("contiguous assignment scratch"),
        )?;
        let jets = self.row_jets_for_logdet(start, vars, a.view(), second_jets, border)?;
        window.push_back(jets);
        Ok(start + 1)
    }
}
