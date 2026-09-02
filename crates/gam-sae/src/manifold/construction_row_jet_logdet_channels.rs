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

/// Zero-copy production adapter for the structure-compiled SAE row programs.
/// It borrows the term's live basis/decoder tensors and the cache-derived primary
/// layout; no per-row `AtomRowBasisJet` clone is constructed.
struct ProductionRowProgram<'a> {
    term: &'a SaeManifoldTerm,
    row: usize,
    vars: &'a [SaeLocalRowVar],
    assignments: ArrayView1<'a, f64>,
    second_jets: &'a [Array4<f64>],
    border: &'a [SaeBorderChannel],
}

impl ProductionRowProgram<'_> {
    #[inline]
    fn atom_is_active_inner(&self, atom: usize) -> bool {
        self.term
            .last_row_layout
            .as_ref()
            .is_none_or(|layout| layout.active_atoms[self.row].binary_search(&atom).is_ok())
    }
}

impl crate::row_jet_program::SaeOrder2RowProgramSource for ProductionRowProgram<'_> {
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
                out[out_col] += d2phi * atom_ref.decoder_coefficients()[[basis_col, out_col]];
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

#[cfg(test)]
mod tests_reconstruction_program_builder {
    use super::*;

    impl SaeManifoldTerm {
    }
}

#[cfg(test)]
mod tests_hand_reference {
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
                    out[out_col] += d2phi * atom.decoder_coefficients()[[basis_col, out_col]];
                }
            }
        }

        /// Historical hand reconstruction + β-border channels, recovered from
        /// 8404ff658^ (before the #932 Taylor-jet cutover) and updated only for
        /// the current independent-logistic assignment names. It was the
        /// production path after the generic tower measured 25–57× slower; it
        /// is now test-only as the strongest non-abstracted performance and
        /// correctness baseline.
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
        pub(crate) fn fill_row_jets_hand_reference(
            &self,
            row: usize,
            vars: &[SaeLocalRowVar],
            assignments: ArrayView1<'_, f64>,
            second_jets: &[Array4<f64>],
            border: &[SaeBorderChannel],
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

            let mut dz = vec![vec![0.0_f64; k_atoms]; q];
            let mut d2z = vec![vec![vec![0.0_f64; k_atoms]; q]; q];
            match self.assignment.mode {
                AssignmentMode::Softmax { temperature, .. } => {
                    let inv_tau = 1.0 / temperature;
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
                }
                AssignmentMode::OrderedBetaBernoulli { temperature, .. }
                | AssignmentMode::ThresholdGate { temperature, .. } => {
                    let inv_tau = 1.0 / temperature;
                    for (slot, var) in vars.iter().enumerate() {
                        let SaeLocalRowVar::Logit { atom } = *var else {
                            continue;
                        };
                        let z = assignments[atom];
                        dz[slot][atom] = inv_tau * z * (1.0 - z);
                        d2z[slot][slot][atom] = inv_tau * inv_tau * z * (1.0 - z) * (1.0 - 2.0 * z);
                    }
                }
                AssignmentMode::TopK { .. } => {}
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
                        // Distinct atoms (the guard above took `atom_a == atom_b`):
                        // an atom's decoder sees only its own coordinates, so the
                        // cross-atom coord×coord block is a structural zero and the
                        // caller's zeroed `second` entry already holds it. Naming
                        // the variants keeps a new `SaeLocalRowVar` from silently
                        // inheriting this "block is zero" claim.
                        (SaeLocalRowVar::Coord { .. }, SaeLocalRowVar::Coord { .. }) => {}
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
                let source = ProductionRowProgram {
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
            AssignmentMode::OrderedBetaBernoulli { temperature, .. }
            | AssignmentMode::ThresholdGate { temperature, .. } => {
                let source = ProductionRowProgram {
                    term: self,
                    row,
                    vars: &vars,
                    assignments,
                    second_jets,
                    border,
                };
                crate::row_jet_program::execute_independent_logistic_row_program(
                    &source,
                    1.0 / temperature,
                    sqrt_row_w,
                )
            }
            AssignmentMode::TopK { .. } => {
                // TopK is the constant-gate degeneration of the independent
                // schedule: the row has no logit primaries, so inv_tau is
                // unobservable and every gate derivative is structurally zero.
                let source = ProductionRowProgram {
                    term: self,
                    row,
                    vars: &vars,
                    assignments,
                    second_jets,
                    border,
                };
                crate::row_jet_program::execute_independent_logistic_row_program(
                    &source, 1.0, sqrt_row_w,
                )
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
        self.refill_jet_window_with_row_dims(start, &cache.row_dims, second_jets, border, window)
    }

    /// [`Self::refill_jet_window`] against the per-row dimensions read directly
    /// off an ArrowSchurSystem (`sys.row_dims`) instead of a factor cache.
    ///
    /// #2509 Phase-2b: the only thing the jet window ever took from the cache
    /// was this layout vector, so the exact-`A` row assembly — which must run
    /// before any factorization — can share the identical jet seam.
    pub(crate) fn refill_jet_window_with_row_dims(
        &self,
        start: usize,
        row_dims: &[usize],
        second_jets: &[Array4<f64>],
        border: &[SaeBorderChannel],
        window: &mut std::collections::VecDeque<SaeRowJets>,
    ) -> Result<usize, String> {
        if let AssignmentMode::Softmax { temperature, .. } = self.assignment.mode {
            // #2560 — one cgroup-aware budget reading per window, passed down,
            // instead of one per planner call.
            let host_budget = crate::manifold::sae_host_in_core_budget_bytes().0;
            let q = row_dims[start];
            let same_shape_rows = row_dims[start..]
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
                host_budget,
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
                let vars = self.row_vars_for_row_dim(row, row_dims[row])?;
                self.assignment.try_assignments_row_into(
                    row,
                    assignments.as_slice_mut().ok_or_else(|| {
                        "complete SAE row-jet assignment scratch is not contiguous".to_string()
                    })?,
                )?;
                let source = ProductionRowProgram {
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

        let vars = self.row_vars_for_row_dim(start, row_dims[start])?;
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
        // #2560 — the cgroup-aware budget is a property of the host, not of the
        // row chunk, so read it once here rather than once per loop turn.
        let host_budget = crate::manifold::sae_host_in_core_budget_bytes().0;
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
                host_budget,
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
                let source = ProductionRowProgram {
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
        // #2560 — the cgroup-aware budget is a property of the host, not of the
        // row chunk, so read it once here rather than once per loop turn.
        let host_budget = crate::manifold::sae_host_in_core_budget_bytes().0;
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
                host_budget,
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
                let source = ProductionRowProgram {
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

    /// Resident softmax `Γ = tr(H⁻¹ ∂H/∂θ)` reduction (#2333).
    ///
    /// This is the sole softmax Trace consumer. It constructs the same selected
    /// inverse blocks as the former hand loop, folds the row deflation map into
    /// `E_tt`, projects every semantic output base into the row metric chart,
    /// and sends the complete data-curvature tower through the typed Trace seam.
    /// Scalar majorizer/ARD channels and the residual third-jet term are host
    /// post-folds because they are not row-jet channels; all use the same `E_tt`
    /// so the conditioned operator is differentiated exactly once.
    fn contracted_softmax_trace_adjoint(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        solver: &DeflatedArrowSolver<'_>,
        joint_block: bool,
        operator: EvidenceOperator,
        residual_target: Option<ArrayView2<'_, f64>>,
    ) -> Result<SaeArrowVector, String> {
        let AssignmentMode::Softmax {
            temperature,
            sparsity,
        } = self.assignment.mode
        else {
            return Err("contracted softmax Trace called on a non-softmax gate".to_string());
        };
        let exact_a = operator.is_exact_a();
        let n = self.n_obs();
        let p = self.output_dim();
        let total_t = cache.delta_t_len();
        let mut gamma_t = Array1::<f64>::zeros(total_t);
        let mut gamma_beta = Array1::<f64>::zeros(cache.k);
        let second_jets = self.atom_second_jets()?;
        let border = self.border_channels_for_cache(cache)?;
        let n_beta = border.len();
        let ard_precisions = self.validated_ard_precisions(rho)?;
        let inv_tau = temperature.recip();
        let entropy_scale = if self.k_atoms() > 1 {
            rho.lambda_sparse()? * sparsity * inv_tau * inv_tau
        } else {
            0.0
        };
        let fast_selected = joint_block && solver.plain_selected_inverse_available();
        let beta_inv = if joint_block {
            Self::selected_inverse_beta_block(
                solver,
                cache,
                fast_selected,
                "contracted_softmax_trace_adjoint",
            )?
        } else {
            Array2::<f64>::zeros((cache.k, cache.k))
        };
        let mut beta_inv_border = vec![0.0_f64; n_beta * n_beta];
        for (i, channel_i) in border.iter().enumerate() {
            for (j, channel_j) in border.iter().enumerate() {
                beta_inv_border[i * n_beta + j] =
                    beta_inv[[channel_i.index, channel_j.index]];
            }
        }
        let rhs_beta_zero = Array1::<f64>::zeros(cache.k);
        let selected_ctx = SelectedInverseRowSolve {
            solver,
            cache,
            beta_inv: &beta_inv,
            fast_selected,
            rhs_beta_zero: rhs_beta_zero.view(),
            context: "contracted_softmax_trace_adjoint",
        };
        let mut rhs_t_scratch = Array1::<f64>::zeros(total_t);
        let whiten = self.whiten_logdet_row_jets();
        let metric = if whiten {
            Some(
                self.row_metric
                    .as_ref()
                    .ok_or_else(|| "contracted softmax Trace whitening metric absent".to_string())?,
            )
        } else {
            None
        };
        let projected_p = metric.map_or(p, |metric| metric.metric_rank());
        let patchd_residual = exact_a.then_some(residual_target).flatten();
        let patchd_third_jets = if patchd_residual.is_some() {
            Some(self.atom_third_jets()?)
        } else {
            None
        };
        let host_budget = crate::manifold::sae_host_in_core_budget_bytes().0;
        let mut assignments_scratch = Array1::<f64>::zeros(self.k_atoms());
        let mut start = 0usize;
        while start < n {
            let q = cache.row_dims[start];
            let same_shape_rows = cache.row_dims[start..]
                .iter()
                .take_while(|&&candidate| candidate == q)
                .count();
            let plan = crate::gpu_kernels::sae_rowjet::plan_softmax_row_jets_trace(
                same_shape_rows,
                self.k_atoms(),
                q,
                projected_p,
                n_beta,
                self.gpu_policy,
                host_budget,
            )?;
            if plan.tile_rows == 0 {
                return Err(format!(
                    "contracted softmax Trace planner returned an empty tile at row {start}"
                ));
            }
            let tile_rows = plan.tile_rows;
            let mut inputs = Vec::with_capacity(tile_rows);
            let mut layouts = Vec::with_capacity(tile_rows);
            let mut e_tt = Vec::with_capacity(tile_rows * q * q);
            let mut inv_vbeta = Vec::with_capacity(tile_rows * q * n_beta);
            let mut shared_beta_layout = None;
            for row in start..start + tile_rows {
                let base = cache.row_offsets[row];
                let vars = self.row_vars_for_cache_row(row, cache)?;
                self.assignment.try_assignments_row_into(
                    row,
                    assignments_scratch
                        .as_slice_mut()
                        .expect("softmax assignment scratch is contiguous"),
                )?;
                let source = ProductionRowProgram {
                    term: self,
                    row,
                    vars: &vars,
                    assignments: assignments_scratch.view(),
                    second_jets: &second_jets,
                    border: &border,
                };
                let sqrt_row_weight = self
                    .row_loss_weights
                    .as_deref()
                    .map_or(1.0, |weights| weights[row].sqrt());
                let mut input =
                    crate::gpu_kernels::sae_rowjet::SaeSoftmaxRowJetInput::from_source(
                        &source,
                        sqrt_row_weight,
                        if metric.is_some() {
                            None
                        } else {
                            shared_beta_layout.clone()
                        },
                    )?;
                if let Some(metric) = metric {
                    input.project_output_bases(projected_p, |source, projected| {
                        for rank_col in 0..projected_p {
                            let mut acc = 0.0_f64;
                            for out_col in 0..p {
                                acc += metric.factor_entry(row, out_col, rank_col)
                                    * source[out_col];
                            }
                            projected[rank_col] = acc;
                        }
                    })?;
                } else {
                    shared_beta_layout =
                        Some((input.beta_atoms.clone(), input.beta_outputs.clone()));
                }
                let (inv_vv_row, inv_vbeta_row) = if joint_block {
                    Self::selected_inverse_row_blocks_or_solve(
                        &selected_ctx,
                        row,
                        base,
                        q,
                        &mut rhs_t_scratch,
                    )?
                } else {
                    let factor = cache.undamped_factor(row);
                    let mut inverse = Array2::<f64>::zeros((q, q));
                    let mut unit = Array1::<f64>::zeros(q);
                    for col in 0..q {
                        unit[col] = 1.0;
                        let solved = cholesky_solve_vector(factor, unit.view());
                        unit[col] = 0.0;
                        for inverse_row in 0..q {
                            inverse[[inverse_row, col]] = solved[inverse_row];
                        }
                    }
                    (inverse, Array2::<f64>::zeros((q, cache.k)))
                };
                let defl_dirs = cache
                    .deflated_row_directions
                    .get(row)
                    .map(Vec::as_slice)
                    .unwrap_or(&[]);
                let defl_spectrum = cache
                    .deflation_row_spectra
                    .get(row)
                    .and_then(Option::as_ref);
                let e_row = Self::deflation_folded_trace_weight(
                    &inv_vv_row,
                    defl_dirs,
                    defl_spectrum,
                );
                e_tt.extend(e_row.iter().copied());
                for a in 0..q {
                    for channel in &border {
                        inv_vbeta.push(inv_vbeta_row[[a, channel.index]]);
                    }
                }
                inputs.push(input);
                layouts.push(vars);
            }
            let trace = crate::gpu_kernels::sae_rowjet::execute_softmax_row_jet_tile_contracted(
                &inputs,
                inv_tau,
                plan.path,
                crate::gpu_kernels::sae_rowjet::SaeRowJetContraction::Trace {
                    e_tt: &e_tt,
                    inv_vbeta: &inv_vbeta,
                    beta_inv: &beta_inv_border,
                    exact_a,
                },
            )?;
            if (trace.n_rows, trace.q, trace.n_beta) != (tile_rows, q, n_beta) {
                return Err(format!(
                    "contracted softmax Trace returned shape ({}, {}, {}); expected ({tile_rows}, {q}, {n_beta})",
                    trace.n_rows, trace.q, trace.n_beta
                ));
            }
            for local in 0..tile_rows {
                let row = start + local;
                let base = cache.row_offsets[row];
                let vars = &layouts[local];
                let assignments = Array1::from_vec(inputs[local].gate_values.clone());
                let e_row = &e_tt[local * q * q..(local + 1) * q * q];
                let vbeta_row =
                    &inv_vbeta[local * q * n_beta..(local + 1) * q * n_beta];
                let m = softmax_majorizer_log_mean(
                    assignments
                        .as_slice()
                        .expect("softmax assignments are contiguous"),
                );
                let w_row = self.row_loss_weights.as_deref().map_or(1.0, |w| w[row]);
                let patchd_error_metric = patchd_residual.map(|target| {
                    self.patchd_row_error_metric(row, w_row, target, &assignments, whiten)
                });
                let patchd_ctx = patchd_error_metric.as_deref().map(|error_metric| {
                    PatchDResidualCtx {
                        row,
                        error_metric,
                        sqrt_w: w_row.sqrt(),
                        assignments: &assignments,
                        second_jets: &second_jets,
                        third_jets: patchd_third_jets.as_deref(),
                        is_obb: false,
                        inv_tau: 0.0,
                    }
                });
                for w in 0..q {
                    let mut gamma = trace.t[local * q + w];
                    if let SaeLocalRowVar::Logit { atom: atom_w } = vars[w] {
                        let a_soft = assignments
                            .as_slice()
                            .expect("softmax assignments are contiguous");
                        for a in 0..q {
                            if let SaeLocalRowVar::Logit { atom: atom_a } = vars[a] {
                                gamma += e_row[a * q + a]
                                    * w_row
                                    * active_softmax_majorizer_logit_derivative_entry(
                                        a_soft,
                                        atom_a,
                                        atom_w,
                                        m,
                                        entropy_scale,
                                        inv_tau,
                                    );
                            }
                        }
                    }
                    if let SaeLocalRowVar::Coord { atom, axis } = vars[w] {
                        if !ard_precisions[atom].is_empty() {
                            let derivative = if exact_a {
                                self.ard_exact_hessian_derivative(
                                    ard_precisions[atom][axis],
                                    row,
                                    atom,
                                    axis,
                                )
                            } else {
                                self.ard_majorized_hessian_derivative(
                                    ard_precisions[atom][axis],
                                    row,
                                    atom,
                                    axis,
                                )
                            };
                            gamma += e_row[w * q + w] * derivative;
                        }
                    }
                    if let Some(ctx) = patchd_ctx.as_ref() {
                        for a in 0..q {
                            for b in 0..q {
                                gamma += e_row[a * q + b]
                                    * self.patchd_residual_third_leg(
                                        ctx, vars[a], vars[b], vars[w],
                                    );
                            }
                            for (border_pos, channel) in border.iter().enumerate() {
                                gamma += 2.0
                                    * vbeta_row[a * n_beta + border_pos]
                                    * self.patchd_residual_third_leg_beta(
                                        ctx,
                                        vars[a],
                                        vars[w],
                                        channel,
                                    );
                            }
                        }
                    }
                    gamma_t[base + w] = gamma;
                }
                for (border_pos, channel) in border.iter().enumerate() {
                    gamma_beta[channel.index] += trace.beta[local * n_beta + border_pos];
                }
            }
            start += tile_rows;
        }
        Ok(SaeArrowVector {
            t: gamma_t,
            beta: gamma_beta,
        })
    }
}
