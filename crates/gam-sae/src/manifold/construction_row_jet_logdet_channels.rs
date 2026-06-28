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
    fn reconstruction_row_program_for_logdet(
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

        let logits = self.assignment.logits.row(row).to_vec();
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
                    .mode
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
            AssignmentMode::JumpReLU {
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
        let program =
            self.reconstruction_row_program_for_logdet(rho, row, &vars, assignments, second_jets)?;
        Self::fill_reconstruction_channels_from_program_dynamic(
            &program,
            sqrt_row_w,
            &mut first,
            &mut second,
        )?;

        // β BORDER CHANNELS (#932): single-sourced through the SAME
        // reconstruction row program used above. A β border channel is one free
        // decoder coefficient whose per-row contribution to output column `c` is
        // ζ_k(ℓ)·Φ_b(t_k)·output_c — linear in β — so the value channel is
        // `beta_border_tower.v · output_c` and the mixed ∂²ẑ_c/∂β∂p_a channel
        // (both `beta_deriv` and `beta_l_deriv`, identical because the map is
        // linear in β) is `beta_border_tower.g[a] · output_c`. The former hand
        // packing — with its own gate first-derivative recursion
        // (`gate_first_derivatives_for_row`) and term-by-term basis/jacobian
        // reads — is replaced by the tower, which carries every gate/basis
        // derivative automatically and is pinned to the prior hand path by
        // `sae_row_jet_program_matches_production_row_jets_on_converged_cache`.
        let mut beta = vec![vec![0.0_f64; p]; border.len()];
        let mut beta_deriv = vec![vec![vec![0.0_f64; p]; border.len()]; q];
        let mut beta_l_deriv = vec![vec![vec![0.0_f64; p]; border.len()]; q];
        Self::fill_beta_border_channels_from_program_dynamic(
            &program,
            sqrt_row_w,
            border,
            &mut beta,
            &mut beta_deriv,
            &mut beta_l_deriv,
        )?;

        Ok(SaeRowJets {
            vars,
            first,
            second,
            beta,
            beta_deriv,
            beta_l_deriv,
        })
    }

    /// Build [`SaeRowJets`] for FOUR rows at once via the 4-row SIMD batch
    /// (#932), returning `None` (so the caller falls back to the scalar per-row
    /// `row_jets_for_logdet`) when the four rows are not softmax-aligned (same
    /// primary layout / temperature). Each lane's `SaeRowJets` is BIT-IDENTICAL
    /// to `row_jets_for_logdet` on that row: the batch primitives
    /// (`reconstruction_all_columns_batch4` / `beta_border_order1_batch4`) are
    /// proven lane-`i` `to_bits`-identical to the scalar `*_packed` paths, and the
    /// `√w` / `output_c` scaling here mirrors the scalar fills term-for-term.
    fn row_jets_for_logdet_batch4(
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
            let prog =
                self.reconstruction_row_program_for_logdet(rho, row, &vars, a.view(), second_jets)?;
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
    /// decline), so the caller can fall back to scalar.
    fn batch4_assemble<const K: usize>(
        rows: [&crate::row_jet_program::SaeReconstructionRowProgram; 4],
        vars_each: &[Vec<SaeLocalRowVar>],
        sqrt_w: &[f64; 4],
        border: &[SaeBorderChannel],
        p: usize,
    ) -> Result<Option<[SaeRowJets; 4]>, String> {
        use crate::row_jet_program::SaeReconstructionRowProgram;
        let recon = match SaeReconstructionRowProgram::reconstruction_all_columns_batch4::<K>(rows) {
            Some(r) => r,
            None => return Ok(None),
        };
        let chans: Vec<(usize, usize)> = border.iter().map(|c| (c.atom, c.basis_col)).collect();
        let bjets = match SaeReconstructionRowProgram::beta_border_order1_batch4::<K>(rows, &chans) {
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

    /// Refill the bounded (≤4-row) look-ahead jet window starting at `start`,
    /// using the 4-row SIMD batch when the next four rows are softmax-aligned and
    /// falling back to one scalar row otherwise (also the `<4` remainder).
    /// Returns the next unbuilt row index. Bit-identical to per-row
    /// `row_jets_for_logdet` either way.
    fn refill_jet_window(
        &self,
        rho: &SaeManifoldRho,
        start: usize,
        n: usize,
        cache: &ArrowFactorCache,
        second_jets: &[Array4<f64>],
        border: &[SaeBorderChannel],
        window: &mut std::collections::VecDeque<SaeRowJets>,
    ) -> Result<usize, String> {
        if start + 4 <= n {
            if let Some(batch) = self.row_jets_for_logdet_batch4(
                rho,
                [start, start + 1, start + 2, start + 3],
                cache,
                second_jets,
                border,
            )? {
                window.extend(batch);
                return Ok(start + 4);
            }
        }
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
