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
        /// The #932 compiled-schedule tests in `tests_row_jet_and_outer_objective_780.rs`
        /// compare the compiled row-jet schedules against this hand arithmetic.
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
                crate::gpu_kernels::sae_rowjet::SaeRowJetContractionKind::Linear,
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
    /// CPU/device dispatch and shape checks as
    /// [`Self::contracted_softmax_linear_rhs`] apply.
    ///
    /// #2822 — every tile's row-program inputs and residual probe rows come from
    /// `prepared` ([`Self::prepare_softmax_row_jets`]), built once per state, so an
    /// apply gathers only the direction and contracts it. Re-deriving them per apply
    /// cost the high-p gauge-deflated circle's dense exact-A build (p = 512, 1537
    /// probes) 125–228 ms per probe: the busy eu-stack samples of pool job 614715 sat
    /// in `SaeSoftmaxRowJetInput::from_source`, the zeroed row-jet schedule and the
    /// border outputs' copies.
    fn contracted_softmax_bilinear_hvp(
        &self,
        prepared: &PreparedSoftmaxRowJets,
        mut v_t_for_row: impl FnMut(usize, usize) -> Result<Vec<f64>, String>,
        v_beta_row: &[f64],
        mut emit: impl FnMut(usize, usize, &[f64], &[f64]) -> Result<(), String>,
    ) -> Result<(), String> {
        let AssignmentMode::Softmax { temperature, .. } = self.assignment.mode else {
            return Err("contracted softmax row-jet HVP called on a non-softmax gate".to_string());
        };
        let n_beta = prepared.border.len();
        if v_beta_row.len() != n_beta {
            return Err(format!(
                "contracted SAE row-jet v_beta has length {}; expected {n_beta}",
                v_beta_row.len()
            ));
        }
        for tile_plan in &prepared.tiles {
            let start = tile_plan.start;
            let q = tile_plan.q;
            let tile_rows = tile_plan.inputs.len();
            let mut v_t = Vec::with_capacity(tile_rows * q);
            let mut v_beta = Vec::with_capacity(tile_rows * n_beta);
            for row in start..start + tile_rows {
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
            let contracted = |path| {
                crate::gpu_kernels::sae_rowjet::execute_softmax_row_jet_tile_contracted(
                    &tile_plan.inputs,
                    1.0 / temperature,
                    path,
                    crate::gpu_kernels::sae_rowjet::SaeRowJetContraction::Bilinear {
                        probe: &tile_plan.probe,
                        v_t: &v_t,
                        v_beta: &v_beta,
                    },
                )
            };
            let cpu = |kept: Option<
                &crate::gpu_kernels::sae_rowjet::bilinear::SaeRowJetBilinearContractions,
            >| match kept {
                Some(kept) => kept.apply(&v_t, &v_beta),
                None => contracted(crate::gpu_kernels::sae_rowjet::SaeRowJetPath::Cpu),
            };
            let tile = match &tile_plan.executor {
                PreparedSoftmaxRowJetExecutor::Cpu { kept } => cpu(kept.as_ref())?,
                PreparedSoftmaxRowJetExecutor::Device => {
                    contracted(crate::gpu_kernels::sae_rowjet::SaeRowJetPath::Device)?
                }
                PreparedSoftmaxRowJetExecutor::Racing { kept, race } => race.apply(
                    || cpu(kept.as_ref()),
                    || {
                        contracted(crate::gpu_kernels::sae_rowjet::SaeRowJetPath::Device)
                            .map(|_| ())
                    },
                )?,
            };
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
        }
        Ok(())
    }

    /// #2822 — the state-only half of [`Self::contracted_softmax_bilinear_hvp`]: the
    /// border channels and, per tile, the planner's path, the row-program inputs and the
    /// √w-scaled metric-applied residual probe rows. Tiles, rows and every value are built
    /// as the per-apply form built them, in the same order, so an apply against the plan is
    /// bit-identical to one that re-derived them. A plan is valid for exactly the state it
    /// was built from.
    pub(crate) fn prepare_softmax_row_jets(
        &self,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
    ) -> Result<PreparedSoftmaxRowJets, String> {
        if !matches!(self.assignment.mode, AssignmentMode::Softmax { .. }) {
            return Err("softmax row-jet plan requested for a non-softmax gate".to_string());
        }
        let n = self.n_obs();
        let p = self.output_dim();
        let k_atoms = self.k_atoms();
        let second_jets = self.atom_second_jets()?;
        let border = self.border_channels_for_cache(cache)?;
        let n_beta = border.len();
        let row_loss_w = self.row_loss_weights.as_deref();
        let whitens = self
            .row_metric
            .as_ref()
            .is_some_and(|metric| metric.whitens_likelihood());
        let mut assignments = Array1::<f64>::zeros(k_atoms);
        let mut probe_assignments = Array1::<f64>::zeros(k_atoms);
        let mut decoded = vec![0.0_f64; p];
        let mut fitted = Array1::<f64>::zeros(p);
        let mut error = Array1::<f64>::zeros(p);
        // #2560 — the cgroup-aware budget is a property of the host, not of the
        // row chunk, so read it once here rather than once per loop turn.
        let host_budget = crate::manifold::sae_host_in_core_budget_bytes().0;
        let mut tiles = Vec::new();
        let mut start = 0usize;
        while start < n {
            let q = cache.row_dims[start];
            let same_shape_rows = cache.row_dims[start..]
                .iter()
                .take_while(|&&candidate| candidate == q)
                .count();
            let plan = crate::gpu_kernels::sae_rowjet::plan_softmax_row_jets_contracted(
                same_shape_rows,
                k_atoms,
                q,
                p,
                n_beta,
                crate::gpu_kernels::sae_rowjet::SaeRowJetContractionKind::Bilinear,
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
                    second_jets: &second_jets,
                    border: &border,
                };
                let sqrt_row_weight = row_loss_w.map_or(1.0, |weights| weights[row].sqrt());
                let input = crate::gpu_kernels::sae_rowjet::SaeSoftmaxRowJetInput::from_source(
                    &source,
                    sqrt_row_weight,
                    shared_beta_layout.clone(),
                )?;
                shared_beta_layout = Some((input.beta_atoms.clone(), input.beta_outputs.clone()));
                inputs.push(input);
                self.assignment.try_assignments_row_into(
                    row,
                    probe_assignments.as_slice_mut().ok_or_else(|| {
                        "apply_exact_hessian_minus_b: assignment scratch is not contiguous"
                            .to_string()
                    })?,
                )?;
                fitted.fill(0.0);
                let active_atoms = self
                    .last_row_layout
                    .as_ref()
                    .map(|layout| layout.active_atoms[row].as_slice());
                for k in 0..k_atoms {
                    if active_atoms.is_some_and(|active| active.binary_search(&k).is_err()) {
                        continue;
                    }
                    self.atoms[k].fill_decoded_row(row, &mut decoded);
                    let a_k = probe_assignments[k];
                    for out_col in 0..p {
                        fitted[out_col] += a_k * decoded[out_col];
                    }
                }
                let sqrt_row_w = row_loss_w.map_or(1.0, |w| w[row].sqrt());
                for out_col in 0..p {
                    error[out_col] = sqrt_row_w * (fitted[out_col] - target[[row, out_col]]);
                }
                let probe_row = match self.row_metric.as_ref() {
                    Some(metric) if whitens => metric.apply_metric_row(row, error.view()),
                    _ => error.to_vec(),
                };
                if probe_row.len() != p {
                    return Err(format!(
                        "contracted SAE row-jet probe for row {row} has length {}; expected {p}",
                        probe_row.len()
                    ));
                }
                probe.extend_from_slice(&probe_row);
            }
            let keep = || match self.assignment.mode {
                AssignmentMode::Softmax { temperature, .. } => {
                    crate::gpu_kernels::sae_rowjet::bilinear::prepare_bilinear_contractions(
                        &inputs,
                        1.0 / temperature,
                        &probe,
                    )
                }
                _ => Ok(None),
            };
            let executor = match plan.path {
                crate::gpu_kernels::sae_rowjet::SaeRowJetPath::Cpu => {
                    PreparedSoftmaxRowJetExecutor::Cpu { kept: keep()? }
                }
                crate::gpu_kernels::sae_rowjet::SaeRowJetPath::Device => {
                    PreparedSoftmaxRowJetExecutor::Device
                }
                crate::gpu_kernels::sae_rowjet::SaeRowJetPath::Race(shape) => {
                    let (race, kept) = gam_gpu::ReusedStateRace::build(
                        gam_gpu::RowKernelShape {
                            rows: inputs.len(),
                            ..shape
                        },
                        keep,
                    )?;
                    PreparedSoftmaxRowJetExecutor::Racing { kept, race }
                }
            };
            tiles.push(PreparedSoftmaxRowJetTile {
                start,
                q,
                executor,
                inputs,
                probe,
            });
            start += tile_rows;
        }
        Ok(PreparedSoftmaxRowJets { border, tiles })
    }
}

/// #2333 — the algebraic identity the deflation-folded trace weight rests on.
///
/// `SaeManifoldTerm::deflation_folded_trace_weight` returns ONE weight `E_tt`
/// whose contraction `Σ_{a,b} E[a,b]·D[a,b]` equals the contract-then-subtract
/// shape `tr(inv_vv·D) − deflation_block_correction(inv_vv, D, …)` for every
/// symmetric `D`, so a consumer that contracts a row block against `E`
/// (`evidence_metric_raw_weight`) prices the Daleckii–Krein correction without
/// a separate subtraction. These tests pin exactly that, at the RELATIVE
/// tolerance the fold's reassociation of the same `f64` sum earns, for every
/// branch of the correction the fold has to mirror.
#[cfg(test)]
mod tests_deflation_trace_fold_2333 {
    use crate::manifold::{RowDeflationSpectrum, RowSpectralConditioning, SaeManifoldTerm};
    use ndarray::{Array1, Array2, array};
    use std::sync::Arc;

    /// A `q×q` orthonormal basis built from an explicit rotation product, so the
    /// fixture depends on no eigensolver's sign or ordering convention.
    fn orthonormal_basis() -> Array2<f64> {
        let (c1, s1) = (0.6_f64, 0.8_f64);
        let (c2, s2) = ((0.28_f64).cos(), (0.28_f64).sin());
        let r1 = array![[c1, -s1, 0.0], [s1, c1, 0.0], [0.0, 0.0, 1.0]];
        let r2 = array![[1.0, 0.0, 0.0], [0.0, c2, -s2], [0.0, s2, c2]];
        r1.dot(&r2)
    }

    /// Symmetric and positive definite, as a selected inverse of a symmetric PD
    /// system is.
    fn symmetric_inverse() -> Array2<f64> {
        array![
            [1.70, 0.35, -0.20],
            [0.35, 1.15, 0.40],
            [-0.20, 0.40, 2.05],
        ]
    }

    /// Symmetric derivative blocks `D`. The identity must hold for EVERY
    /// symmetric `D` the tower can produce, so the probe set spans diagonal,
    /// off-diagonal, indefinite and rank-one shapes rather than one convenient
    /// matrix.
    fn symmetric_probes() -> Vec<Array2<f64>> {
        let rank_one = {
            let v = array![0.7_f64, -0.4, 0.55];
            let mut m = Array2::<f64>::zeros((3, 3));
            for a in 0..3 {
                for b in 0..3 {
                    m[[a, b]] = v[a] * v[b];
                }
            }
            m
        };
        vec![
            array![[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            array![[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            array![[0.9, -0.3, 0.15], [-0.3, -1.4, 0.22], [0.15, 0.22, 0.65]],
            array![[-2.2, 0.05, -0.9], [0.05, 0.31, 0.04], [-0.9, 0.04, 1.7]],
            rank_one,
        ]
    }

    /// Exactly what the Trace kernel reduces: `Σ_{a,b} E[a,b]·dh[a,b]` over the
    /// full `q×q` index range (`cpu_contracted_tile`'s Trace arm walks both
    /// indices, so the weight is never required to be symmetric).
    fn seam_contraction(e: &Array2<f64>, d: &Array2<f64>) -> f64 {
        let q = e.nrows();
        let mut acc = 0.0_f64;
        for a in 0..q {
            for b in 0..q {
                acc += e[[a, b]] * d[[a, b]];
            }
        }
        acc
    }

    /// `tr(inv·D)` written out, with no symmetry assumed of either operand —
    /// the same quantity the retired hand loop accumulated as
    /// `Σ_{a,b} inv_vv[[b,a]]·dh[[a,b]]`.
    fn trace_of_product(inv: &Array2<f64>, d: &Array2<f64>) -> f64 {
        let q = inv.nrows();
        let mut acc = 0.0_f64;
        for a in 0..q {
            for b in 0..q {
                acc += inv[[a, b]] * d[[b, a]];
            }
        }
        acc
    }

    /// The retired shape: contract against the raw selected inverse, then
    /// subtract the Daleckii–Krein deflation correction.
    fn contract_then_subtract(
        inv: &Array2<f64>,
        d: &Array2<f64>,
        dirs: &[Array1<f64>],
        spectrum: Option<&RowDeflationSpectrum>,
    ) -> f64 {
        trace_of_product(inv, d)
            - SaeManifoldTerm::deflation_block_correction(inv, d, dirs, spectrum)
    }

    /// The spectral fixture: one raw-kept, one floor-clamped and one
    /// unit-deflated direction, so `F` exercises the divided difference AND both
    /// degenerate fallbacks in one weight.
    fn spectral_fixture(u: Array2<f64>) -> RowDeflationSpectrum {
        RowDeflationSpectrum {
            evecs: u,
            raw_evals: array![2.40_f64, 0.35, 0.02],
            cond_evals: array![2.40_f64, 0.50, 1.00],
            conditioning: Arc::from(
                [
                    RowSpectralConditioning::Raw,
                    RowSpectralConditioning::FloorClamped,
                    RowSpectralConditioning::UnitDeflated,
                ]
                .as_slice(),
            ),
        }
    }

    /// Relative to the trace being reproduced, because the fold reassociates the
    /// same `f64` sum: an absolute bar would silently tighten or loosen with the
    /// fixture's scale.
    fn fold_tolerance(expected: f64) -> f64 {
        1.0e-10 * (1.0 + expected.abs())
    }

    /// Spectral deflation: the folded seam weight must reproduce
    /// `tr(inv_vv·D) − deflation_block_correction(…)` for every symmetric `D`.
    #[test]
    fn deflation_folded_trace_weight_reproduces_contract_then_subtract_2333() {
        let inv = symmetric_inverse();
        let spectrum = spectral_fixture(orthonormal_basis());
        let dirs: Vec<Array1<f64>> = Vec::new();

        let e = SaeManifoldTerm::deflation_folded_trace_weight(&inv, &dirs, Some(&spectrum));
        assert_eq!(e.dim(), (3, 3), "the seam weight must stay q x q");

        let mut worst_correction = 0.0_f64;
        for d in symmetric_probes() {
            let expected = contract_then_subtract(&inv, &d, &dirs, Some(&spectrum));
            let folded = seam_contraction(&e, &d);
            let tolerance = fold_tolerance(expected);
            assert!(
                (folded - expected).abs() <= tolerance,
                "#2333 folded weight must reproduce contract-then-subtract: \
                 folded={folded:.17e} expected={expected:.17e} tol={tolerance:.3e}"
            );
            let raw = trace_of_product(&inv, &d);
            worst_correction = worst_correction.max((raw - expected).abs() / (1.0 + raw.abs()));
        }
        // Non-vacuity: a fold that simply returned `inv_vv` would satisfy the
        // assertion above on a fixture whose correction is numerically zero.
        assert!(
            worst_correction > 1.0e-10,
            "#2333 spectral fixture must carry a correction the fold has to \
             reproduce; largest relative correction was {worst_correction:.3e}"
        );
    }

    /// Gauge-only deflation (`spectrum = None`, non-empty `dirs`) folds to
    /// `inv_vv − Σᵢ vᵢvᵢᵀ`, and an UNDEFLATED row folds to `inv_vv` itself. Both
    /// are separate branches of the correction, so a fold that handled only the
    /// spectral branch would still pass the test above.
    #[test]
    fn deflation_folded_trace_weight_covers_gauge_only_and_undeflated_branches_2333() {
        let inv = symmetric_inverse();

        let dirs = vec![
            array![0.6_f64, -0.8, 0.0],
            // Deliberately SHORT: the correction zero-extends a direction whose
            // length is below `q`, and the fold must do the same.
            array![0.0_f64, 0.5],
        ];
        let gauge = SaeManifoldTerm::deflation_folded_trace_weight(&inv, &dirs, None);
        let mut worst_correction = 0.0_f64;
        for d in symmetric_probes() {
            let expected = contract_then_subtract(&inv, &d, &dirs, None);
            let folded = seam_contraction(&gauge, &d);
            let tolerance = fold_tolerance(expected);
            assert!(
                (folded - expected).abs() <= tolerance,
                "#2333 gauge-only fold must reproduce contract-then-subtract: \
                 folded={folded:.17e} expected={expected:.17e} tol={tolerance:.3e}"
            );
            let raw = trace_of_product(&inv, &d);
            worst_correction = worst_correction.max((raw - expected).abs() / (1.0 + raw.abs()));
        }
        assert!(
            worst_correction > 1.0e-10,
            "#2333 gauge-only fixture must carry a correction the fold has to \
             reproduce; largest relative correction was {worst_correction:.3e}"
        );

        // An undeflated row: the consumer hands the seam the raw selected
        // inverse, bit for bit, because the correction's own zero branch does.
        let undeflated = SaeManifoldTerm::deflation_folded_trace_weight(&inv, &[], None);
        for a in 0..3 {
            for b in 0..3 {
                assert_eq!(
                    undeflated[[a, b]].to_bits(),
                    inv[[a, b]].to_bits(),
                    "#2333 an undeflated row must fold to the raw selected inverse"
                );
            }
        }
    }

    /// A degenerate raw pair carrying DIFFERENT conditioning decisions.
    ///
    /// `|λ_a − λ_b| ≤ gap_threshold` takes the divided difference to its
    /// diagonal limit `f'(λ)`, which is `1` for a retained direction and `0` for
    /// a conditioned one — so `F`, and hence the seam weight, is genuinely
    /// ASYMMETRIC here. That is legal: the CPU Trace arm
    /// walks the full `(a,b)` range, and the identity still holds because
    /// `Σ_{a,b} (U G Uᵀ)[a,b]·D[a,b] = Σ_{a,b} G[a,b]·(UᵀDU)[a,b]` needs no
    /// symmetry of `G`.
    #[test]
    fn deflation_folded_trace_weight_handles_a_degenerate_pair_split_by_conditioning_2333() {
        let inv = symmetric_inverse();
        let spectrum = RowDeflationSpectrum {
            evecs: orthonormal_basis(),
            // The first two raw eigenvalues are EXACTLY equal, so their gap is
            // below any threshold and both off-diagonal entries take the
            // diagonal limit of their own row's conditioning.
            raw_evals: array![0.02_f64, 0.02, 2.40],
            cond_evals: array![0.02_f64, 1.00, 2.40],
            conditioning: Arc::from(
                [
                    RowSpectralConditioning::Raw,
                    RowSpectralConditioning::UnitDeflated,
                    RowSpectralConditioning::Raw,
                ]
                .as_slice(),
            ),
        };
        let dirs = vec![orthonormal_basis().column(1).to_owned()];

        let e = SaeManifoldTerm::deflation_folded_trace_weight(&inv, &dirs, Some(&spectrum));
        let mut asymmetry = 0.0_f64;
        for a in 0..3 {
            for b in 0..3 {
                asymmetry = asymmetry.max((e[[a, b]] - e[[b, a]]).abs());
            }
        }
        assert!(
            asymmetry > 1.0e-10,
            "#2333 the split degenerate pair must actually produce an asymmetric \
             weight, else this fixture proves nothing about the (a,b) walk; \
             asymmetry={asymmetry:.3e}"
        );

        for d in symmetric_probes() {
            let expected = contract_then_subtract(&inv, &d, &dirs, Some(&spectrum));
            let folded = seam_contraction(&e, &d);
            let tolerance = fold_tolerance(expected);
            assert!(
                (folded - expected).abs() <= tolerance,
                "#2333 degenerate-pair fold must reproduce contract-then-subtract: \
                 folded={folded:.17e} expected={expected:.17e} tol={tolerance:.3e}"
            );
        }
    }

    /// The clamp-basin shape: a spectrum whose Daleckii–Krein map is
    /// NON-IDENTITY while the row's deflated-direction list is EMPTY.
    ///
    /// `factor_spectral_deflated_criterion_row_with_geometry` reaches this state
    /// through `ExactADirectionClassification::ClampBasin`, which reprices a
    /// direction's conditioned eigenvalue without unit-deflating it — so it
    /// pushes no direction and leaves `conditioning` on `Raw`. The ρ-trace
    /// siblings gate on `spectrum.is_some() || !dirs.is_empty()` for exactly this
    /// reason (#2515/#2336). The fold inherits that convention structurally: it
    /// branches on the SPECTRUM, never on the direction list, so the seam
    /// differentiates the priced spectral map here. Pinned because the retired
    /// hand loop — and `logdet_theta_adjoint_dense`, this route's parity
    /// reference — instead gate on `!dirs.is_empty()` alone and skip it.
    #[test]
    fn deflation_folded_trace_weight_prices_a_spectrum_with_no_deflated_direction_2333() {
        let inv = symmetric_inverse();
        let spectrum = spectral_fixture(orthonormal_basis());
        let no_dirs: Vec<Array1<f64>> = Vec::new();

        let e = SaeManifoldTerm::deflation_folded_trace_weight(&inv, &no_dirs, Some(&spectrum));
        let mut worst_departure = 0.0_f64;
        for d in symmetric_probes() {
            let folded = seam_contraction(&e, &d);
            let raw = trace_of_product(&inv, &d);
            worst_departure = worst_departure.max((folded - raw).abs() / (1.0 + raw.abs()));
            // The identity itself still holds: `deflation_block_correction` also
            // reads only the spectrum on this branch.
            let expected = contract_then_subtract(&inv, &d, &no_dirs, Some(&spectrum));
            let tolerance = fold_tolerance(expected);
            assert!(
                (folded - expected).abs() <= tolerance,
                "#2333 empty-direction spectral fold must reproduce \
                 contract-then-subtract: folded={folded:.17e} \
                 expected={expected:.17e} tol={tolerance:.3e}"
            );
        }
        assert!(
            worst_departure > 1.0e-10,
            "#2333 an empty direction list must NOT disable the spectral fold; \
             largest relative departure from the raw selected inverse was \
             {worst_departure:.3e}"
        );
    }
}
