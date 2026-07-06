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

impl SaeManifoldTerm {
    pub(crate) fn reconstruction_row_program_for_logdet(
        &self,
        rho: &SaeManifoldRho,
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
                                .map(|out_col| atom.decoder_coefficients[[basis_col, out_col]])
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
                if self.assignment.logit_is_fixed(k) {
                    Some(assignments[k])
                } else {
                    None
                }
            })
            .collect();
        let (gate, gate_shift, gate_scale) = match self.assignment.mode {
            AssignmentMode::Softmax { temperature, .. } => (
                RowGate::Softmax {
                    inv_tau: 1.0 / temperature,
                },
                vec![0.0; k_atoms],
                vec![1.0; k_atoms],
            ),
            AssignmentMode::IBPMap {
                temperature, alpha, ..
            } => {
                let effective_alpha = self
                    .assignment
                    .resolved_ibp_alpha(rho)
                    .unwrap_or(alpha);
                (
                    RowGate::PerAtomLogistic {
                        inv_tau: 1.0 / temperature,
                    },
                    vec![0.0; k_atoms],
                    ordered_geometric_shrinkage_prior(k_atoms, effective_alpha).to_vec(),
                )
            }
            AssignmentMode::ThresholdGate {
                temperature,
                threshold,
            } => (
                RowGate::PerAtomLogistic {
                    inv_tau: 1.0 / temperature,
                },
                vec![threshold; k_atoms],
                logits
                    .iter()
                    .map(|&logit| if logit > threshold { 1.0 } else { 0.0 })
                    .collect(),
            ),
        };

        Ok(SaeReconstructionRowProgram {
            atoms,
            gate_value: assignments.to_vec(),
            logits,
            gate_scale,
            gate_shift,
            gate,
            logit_slot,
            coord_slot,
            fixed_gate_value,
            n_primaries: vars.len(),
        })
    }

    fn fill_reconstruction_channels_from_program<const K: usize>(
        program: &crate::row_jet_program::SaeReconstructionRowProgram,
        sqrt_row_w: f64,
        first: &mut [Vec<f64>],
        second: &mut [Vec<Vec<f64>>],
    ) {
        // Build every output column at once with the per-atom gate / basis jets
        // hoisted out of the column loop (#932 perf): the softmax gate jet and
        // the basis jets are column-independent, so this removes the `out_dim×`
        // redundant recomputation the per-column path incurred (~9× faster at
        // K=8, out_dim=16). Bit-identical to per-column `_packed` assembly.
        let columns = program.reconstruction_all_columns_packed::<K>();
        for (out_col, tower) in columns.iter().enumerate() {
            let g = tower.g();
            let h = tower.h();
            for a in 0..K {
                first[a][out_col] = sqrt_row_w * g[a];
                for b in 0..K {
                    second[a][b][out_col] = sqrt_row_w * h[a][b];
                }
            }
        }
    }

    fn fill_reconstruction_channels_from_program_dynamic(
        program: &crate::row_jet_program::SaeReconstructionRowProgram,
        sqrt_row_w: f64,
        first: &mut [Vec<f64>],
        second: &mut [Vec<Vec<f64>>],
    ) -> Result<(), String> {
        macro_rules! dispatch {
            ($($k:literal),* $(,)?) => {
                match program.n_primaries {
                    $(
                        $k => {
                            Self::fill_reconstruction_channels_from_program::<$k>(
                                program,
                                sqrt_row_w,
                                first,
                                second,
                            );
                            Ok(())
                        }
                    )*
                    q => Err(format!(
                        "SAE row reconstruction Tower4 production path supports at most 16 row primaries, got {q}"
                    )),
                }
            };
        }
        dispatch!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
    }

    fn fill_beta_border_channels_from_program<const K: usize>(
        program: &crate::row_jet_program::SaeReconstructionRowProgram,
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
        let chans: Vec<(usize, usize)> = border.iter().map(|c| (c.atom, c.basis_col)).collect();
        let sjets = program.beta_border_order1_packed::<K>(&chans);
        for (beta_pos, channel) in border.iter().enumerate() {
            let s = &sjets[beta_pos];
            let s_v = s.value();
            let s_g = s.g();
            for out_col in 0..p {
                let out_c = channel.output[out_col];
                beta[beta_pos][out_col] = sqrt_row_w * s_v * out_c;
                for a in 0..K {
                    // Reconstruction is linear in β, so beta_deriv and
                    // beta_l_deriv are the identical mixed ∂²ẑ_c/∂β∂p_a channel.
                    let mixed = sqrt_row_w * s_g[a] * out_c;
                    beta_deriv[a][beta_pos][out_col] = mixed;
                    beta_l_deriv[a][beta_pos][out_col] = mixed;
                }
            }
        }
    }

    fn fill_beta_border_channels_from_program_dynamic(
        program: &crate::row_jet_program::SaeReconstructionRowProgram,
        sqrt_row_w: f64,
        border: &[SaeBorderChannel],
        beta: &mut [Vec<f64>],
        beta_deriv: &mut [Vec<Vec<f64>>],
        beta_l_deriv: &mut [Vec<Vec<f64>>],
    ) -> Result<(), String> {
        macro_rules! dispatch {
            ($($k:literal),* $(,)?) => {
                match program.n_primaries {
                    $(
                        $k => {
                            Self::fill_beta_border_channels_from_program::<$k>(
                                program,
                                sqrt_row_w,
                                border,
                                beta,
                                beta_deriv,
                                beta_l_deriv,
                            );
                            Ok(())
                        }
                    )*
                    q => Err(format!(
                        "SAE β border Tower4 production path supports at most 16 row primaries, got {q}"
                    )),
                }
            };
        }
        dispatch!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
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
    /// Softmax-only: the per-atom-logistic (IBP / JumpReLU) modes keep the jet
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
        rho: &SaeManifoldRho,
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
            AssignmentMode::IBPMap { .. } | AssignmentMode::ThresholdGate { .. } => {
                // PER-ATOM-LOGISTIC modes keep the jet path: value-preserving
                // (their hand gate prior diverged from the live ordered-geometric
                // prior, and the batched SIMD speedup that motivated the revert is
                // softmax-only anyway).
                let program = self.reconstruction_row_program_for_logdet(
                    rho,
                    row,
                    &vars,
                    assignments,
                    second_jets,
                )?;
                Self::fill_reconstruction_channels_from_program_dynamic(
                    &program,
                    sqrt_row_w,
                    &mut first,
                    &mut second,
                )?;
                Self::fill_beta_border_channels_from_program_dynamic(
                    &program,
                    sqrt_row_w,
                    border,
                    &mut beta,
                    &mut beta_deriv,
                    &mut beta_l_deriv,
                )?;
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

/// Test-only oracle (#932 revert): the demoted 4-row SIMD jet batch, retained as
/// a live cross-check of the production hand `row_jets_for_logdet`. It lives in a
/// `#[cfg(test)]` module (rather than carrying a bare `#[cfg(test)]` on the items
/// inside the production `impl`) so the first-party dead-code / hygiene gates see
/// it as test support rather than an unreferenced production item.
#[cfg(test)]
mod batch4_oracle_tests {
    use super::*;

    impl SaeManifoldTerm {
        /// Build [`SaeRowJets`] for FOUR rows at once via the 4-row SIMD batch
        /// (#932), returning `None` when the four rows are not softmax-aligned (same
        /// primary layout / temperature). Each lane's `SaeRowJets` is BIT-IDENTICAL
        /// to `row_jets_for_logdet` on that row: the batch primitives
        /// (`reconstruction_all_columns_batch4` / `beta_border_order1_batch4`) are
        /// proven lane-`i` `to_bits`-identical to the scalar `*_packed` paths, and the
        /// `√w` / `output_c` scaling here mirrors the scalar fills term-for-term.
        ///
        /// DEMOTED TO ORACLE (#932 revert): no longer on the production hot path —
        /// `refill_jet_window` now builds the hand `row_jets_for_logdet` per row (the
        /// jet was a 25–57× regression). Retained as the live cross-check of the hand
        /// path (`batch4_jet_lanes_match_scalar_hand_row_jets`).
        pub(crate) fn row_jets_for_logdet_batch4(
            &self,
            rho: &SaeManifoldRho,
            rows: [usize; 4],
            cache: &ArrowFactorCache,
            second_jets: &[Array4<f64>],
            border: &[SaeBorderChannel],
        ) -> Result<Option<[SaeRowJets; 4]>, String> {
            let p = self.output_dim();
            let k_atoms = self.k_atoms();
            let mut progs: Vec<crate::row_jet_program::SaeReconstructionRowProgram> =
                Vec::with_capacity(4);
            let mut vars_each: Vec<Vec<SaeLocalRowVar>> = Vec::with_capacity(4);
            let mut sqrt_w = [1.0_f64; 4];
            for (i, &row) in rows.iter().enumerate() {
                let vars = self.row_vars_for_cache_row(row, cache)?;
                let mut a = Array1::<f64>::zeros(k_atoms);
                self.assignment.try_assignments_row_for_rho_into(
                    row,
                    rho,
                    a.as_slice_mut().expect("contiguous assignment scratch"),
                )?;
                let prog = self.reconstruction_row_program_for_logdet(
                    rho,
                    row,
                    &vars,
                    a.view(),
                    second_jets,
                )?;
                sqrt_w[i] = self
                    .row_loss_weights
                    .as_deref()
                    .map_or(1.0, |w| w[row].sqrt());
                vars_each.push(vars);
                progs.push(prog);
            }
            let refs = [&progs[0], &progs[1], &progs[2], &progs[3]];
            macro_rules! dispatch {
            ($($k:literal),* $(,)?) => {
                match progs[0].n_primaries {
                    $( $k => Self::batch4_assemble::<$k>(refs, &vars_each, &sqrt_w, border, p), )*
                    _ => Ok(None),
                }
            };
        }
            dispatch!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        }

        /// Assemble the four lanes of a SIMD batch into per-row [`SaeRowJets`],
        /// applying the identical `√w` / `output_c` scaling the scalar fills use.
        /// Returns `None` if the rows are not batchable (the batch primitives
        /// decline). Test-only oracle helper for `row_jets_for_logdet_batch4`.
        fn batch4_assemble<const K: usize>(
            rows: [&crate::row_jet_program::SaeReconstructionRowProgram; 4],
            vars_each: &[Vec<SaeLocalRowVar>],
            sqrt_w: &[f64; 4],
            border: &[SaeBorderChannel],
            p: usize,
        ) -> Result<Option<[SaeRowJets; 4]>, String> {
            use crate::row_jet_program::SaeReconstructionRowProgram;
            let recon =
                match SaeReconstructionRowProgram::reconstruction_all_columns_batch4::<K>(rows) {
                    Some(r) => r,
                    None => return Ok(None),
                };
            let chans: Vec<(usize, usize)> = border.iter().map(|c| (c.atom, c.basis_col)).collect();
            let bjets =
                match SaeReconstructionRowProgram::beta_border_order1_batch4::<K>(rows, &chans) {
                    Some(b) => b,
                    None => return Ok(None),
                };
            let mut outs: Vec<SaeRowJets> = Vec::with_capacity(4);
            for lane in 0..4 {
                let sqrt = sqrt_w[lane];
                let mut first = vec![vec![0.0_f64; p]; K];
                let mut second = vec![vec![vec![0.0_f64; p]; K]; K];
                for (out_col, tower) in recon[lane].iter().enumerate() {
                    let g = tower.g();
                    let h = tower.h();
                    for a in 0..K {
                        first[a][out_col] = sqrt * g[a];
                        for b in 0..K {
                            second[a][b][out_col] = sqrt * h[a][b];
                        }
                    }
                }
                let mut beta = vec![vec![0.0_f64; p]; border.len()];
                let mut beta_deriv = vec![vec![vec![0.0_f64; p]; border.len()]; K];
                let mut beta_l_deriv = vec![vec![vec![0.0_f64; p]; border.len()]; K];
                for (beta_pos, channel) in border.iter().enumerate() {
                    let s = &bjets[lane][beta_pos];
                    let s_v = s.value();
                    let s_g = s.g();
                    for out_col in 0..p {
                        let out_c = channel.output[out_col];
                        beta[beta_pos][out_col] = sqrt * s_v * out_c;
                        for a in 0..K {
                            let mixed = sqrt * s_g[a] * out_c;
                            beta_deriv[a][beta_pos][out_col] = mixed;
                            beta_l_deriv[a][beta_pos][out_col] = mixed;
                        }
                    }
                }
                outs.push(SaeRowJets {
                    vars: vars_each[lane].clone(),
                    first,
                    second,
                    beta,
                    beta_deriv,
                    beta_l_deriv,
                });
            }
            let arr: [SaeRowJets; 4] = outs
                .try_into()
                .map_err(|_| "batch4_assemble produced wrong lane count".to_string())?;
            Ok(Some(arr))
        }
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
        rho: &SaeManifoldRho,
        start: usize,
        cache: &ArrowFactorCache,
        second_jets: &[Array4<f64>],
        border: &[SaeBorderChannel],
        window: &mut std::collections::VecDeque<SaeRowJets>,
    ) -> Result<usize, String> {
        let vars = self.row_vars_for_cache_row(start, cache)?;
        let mut a = Array1::<f64>::zeros(self.k_atoms());
        self.assignment.try_assignments_row_for_rho_into(
            start,
            rho,
            a.as_slice_mut().expect("contiguous assignment scratch"),
        )?;
        let jets = self.row_jets_for_logdet(rho, start, vars, a.view(), second_jets, border)?;
        window.push_back(jets);
        Ok(start + 1)
    }
}
