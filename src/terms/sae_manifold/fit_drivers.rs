use super::*;

/// Maximum number of LM ridge-escalation attempts before declaring the per-row
/// Hessian unfactorable.
const SAE_MANIFOLD_ROW_RIDGE_MAX_ATTEMPTS: usize = 12;

impl SaeManifoldTerm {
    pub fn solve_newton_step(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        analytic_penalties: Option<&AnalyticPenaltyRegistry>,
        ridge_ext_coord: f64,
        ridge_beta: f64,
    ) -> Result<(Array1<f64>, Array1<f64>), ArrowSchurError> {
        let sys = self
            .assemble_arrow_schur(target, rho, analytic_penalties)
            .map_err(|reason| ArrowSchurError::SchurFactorFailed { reason })?;
        // Self-heal against non-PD per-row blocks produced by PCA-seeded
        // latent coordinates on subset / out-of-sample data (#163, #175):
        // route every Newton-step solve through the Ceres-style LM ridge
        // escalation, reusing the caller-supplied Tikhonov ridges
        // (`ridge_ext_coord`, `ridge_beta`) as the base damping. No new
        // tuning knobs — just the existing proximal-correction schedule.
        let plan = self
            .streaming_plan()
            .admitted_or_error(self.n_obs(), self.output_dim(), self.k_atoms())
            .map_err(|reason| ArrowSchurError::SchurFactorFailed { reason })?;
        let options = plan.solve_options_for_border_dim(sys.k);
        solve_with_lm_escalation_inner(&sys, ridge_ext_coord, ridge_beta, &options)
            .map(|(delta_t, delta_beta, _diag)| (delta_t, delta_beta))
    }

    pub fn apply_newton_step(
        &mut self,
        delta_ext_coord: ArrayView1<'_, f64>,
        delta_beta: ArrayView1<'_, f64>,
        step_size: f64,
    ) -> Result<(), String> {
        self.apply_newton_step_impl(delta_ext_coord, delta_beta, step_size, true)
    }

    pub fn apply_newton_step_external_basis_refresh(
        &mut self,
        delta_ext_coord: ArrayView1<'_, f64>,
        delta_beta: ArrayView1<'_, f64>,
        step_size: f64,
    ) -> Result<(), String> {
        self.apply_newton_step_impl(delta_ext_coord, delta_beta, step_size, false)
    }

    /// Capture the mutable state perturbed by an `apply_newton_step` +
    /// `loss` line-search trial, plus the row-layout state read by
    /// `apply_newton_step` when unpacking compact Newton steps. See
    /// [`SaeManifoldMutableState`].
    pub(crate) fn snapshot_mutable_state(&self) -> SaeManifoldMutableState {
        let atoms = self
            .atoms
            .iter()
            .map(|atom| {
                (
                    atom.basis_values.clone(),
                    atom.basis_jacobian.clone(),
                    atom.decoder_coefficients.clone(),
                    atom.smooth_penalty.clone(),
                )
            })
            .collect();
        SaeManifoldMutableState {
            atoms,
            logits: self.assignment.logits.clone(),
            coords: self.assignment.coords.clone(),
            last_row_layout: self.last_row_layout.clone(),
        }
    }

    /// Restore the mutable state captured by [`Self::snapshot_mutable_state`].
    /// Assigns into the existing arrays in place so the restore reuses the
    /// already-allocated buffers rather than reallocating per trial.
    pub(crate) fn restore_mutable_state(&mut self, snapshot: &SaeManifoldMutableState) {
        for (atom, (basis_values, basis_jacobian, decoder, smooth_penalty)) in
            self.atoms.iter_mut().zip(snapshot.atoms.iter())
        {
            atom.basis_values.assign(basis_values);
            atom.basis_jacobian.assign(basis_jacobian);
            atom.decoder_coefficients.assign(decoder);
            atom.smooth_penalty.assign(smooth_penalty);
        }
        self.assignment.logits.assign(&snapshot.logits);
        self.assignment.coords.clone_from(&snapshot.coords);
        self.last_row_layout.clone_from(&snapshot.last_row_layout);
    }

    pub(crate) fn refresh_basis_from_current_coords(&mut self) -> Result<(), String> {
        for atom_idx in 0..self.k_atoms() {
            let coords = self.assignment.coords[atom_idx].as_matrix();
            self.atoms[atom_idx].refresh_basis(coords.view())?;
        }
        Ok(())
    }

    pub(crate) fn canonicalize_affine_gauge_after_accept(
        &mut self,
        rho: Option<&SaeManifoldRho>,
    ) -> Result<(), String> {
        for atom_idx in 0..self.k_atoms() {
            if !matches!(
                self.atoms[atom_idx].basis_kind,
                SaeAtomBasisKind::EuclideanPatch
                    | SaeAtomBasisKind::Duchon
                    | SaeAtomBasisKind::Poincare
            ) {
                continue;
            }
            self.canonicalize_atom_affine_gauge(atom_idx, rho)?;
        }
        Ok(())
    }

    pub(crate) fn canonicalize_atom_affine_gauge(
        &mut self,
        atom_idx: usize,
        rho: Option<&SaeManifoldRho>,
    ) -> Result<(), String> {
        let n = self.n_obs();
        let d = self.assignment.coords[atom_idx].latent_dim();
        if n == 0 || d == 0 {
            return Ok(());
        }
        let Some(evaluator) = self.atoms[atom_idx].basis_evaluator.as_ref() else {
            return Ok(());
        };
        let coords = self.assignment.coords[atom_idx].as_matrix();
        let weights = self.atom_affine_gauge_weights(atom_idx, rho)?;
        let weight_sum: f64 = weights.iter().sum();
        if !(weight_sum.is_finite() && weight_sum > 0.0) {
            return Ok(());
        }

        let mut shift = vec![0.0_f64; d];
        for row in 0..n {
            let w = weights[row];
            for axis in 0..d {
                shift[axis] += w * coords[[row, axis]];
            }
        }
        for value in &mut shift {
            *value /= weight_sum;
        }

        let mut scale = vec![1.0_f64; d];
        let mut changed = false;
        for axis in 0..d {
            let mut var = 0.0_f64;
            for row in 0..n {
                let centered = coords[[row, axis]] - shift[axis];
                var += weights[row] * centered * centered;
            }
            let rms = (var / weight_sum).sqrt();
            if rms.is_finite() && rms > 1.0e-12 {
                scale[axis] = rms;
            }
            if shift[axis].abs() > 1.0e-12 || (scale[axis] - 1.0).abs() > 1.0e-12 {
                changed = true;
            }
        }
        if !changed {
            return Ok(());
        }

        let Some(new_evaluator) = evaluator.affine_transformed_evaluator(
            &shift,
            &scale,
            self.atoms[atom_idx].basis_size(),
        )?
        else {
            return Ok(());
        };

        let mut new_coords = coords.clone();
        for row in 0..n {
            for axis in 0..d {
                new_coords[[row, axis]] = (coords[[row, axis]] - shift[axis]) / scale[axis];
            }
        }
        let (new_phi, new_jet) = if self.atoms[atom_idx].homotopy_eta == 1.0 {
            new_evaluator.evaluate(new_coords.view())?
        } else {
            let evaluated = new_evaluator
                .evaluate_phi_eta(new_coords.view(), self.atoms[atom_idx].homotopy_eta)?;
            (evaluated.phi, evaluated.jet)
        };
        let old_phi = self.atoms[atom_idx].basis_values.clone();
        if new_phi.dim() != old_phi.dim() {
            return Err(format!(
                "SaeManifoldTerm::canonicalize_atom_affine_gauge: transformed basis shape {:?} != {:?}",
                new_phi.dim(),
                old_phi.dim()
            ));
        }
        let transport = solve_basis_transport(new_phi.view(), old_phi.view())?;
        let old_decoder = self.atoms[atom_idx].decoder_coefficients.clone();
        let old_smooth_penalty = self.atoms[atom_idx].smooth_penalty.clone();
        let new_decoder = fast_ab(&transport, &old_decoder);
        let old_fit = fast_ab(&old_phi, &old_decoder);
        let new_fit = fast_ab(&new_phi, &new_decoder);
        let fit_scale = old_fit
            .iter()
            .chain(new_fit.iter())
            .fold(1.0_f64, |acc, &v| acc.max(v.abs()));
        let max_abs = old_fit
            .iter()
            .zip(new_fit.iter())
            .fold(0.0_f64, |acc, (&a, &b)| acc.max((a - b).abs()));
        if max_abs > 1.0e-8 * fit_scale {
            return Ok(());
        }

        let flat = Array1::from_iter(new_coords.iter().copied());
        self.assignment.coords[atom_idx].set_flat(flat.view());
        let atom = &mut self.atoms[atom_idx];
        atom.basis_values = new_phi;
        atom.basis_jacobian = new_jet;
        atom.decoder_coefficients = new_decoder;
        let base: Arc<dyn SaeBasisEvaluator> = new_evaluator.clone();
        atom.basis_evaluator = Some(base);
        atom.basis_second_jet = Some(new_evaluator);
        atom.smooth_penalty =
            transport_smooth_penalty_for_decoder(transport.view(), old_smooth_penalty.view())?;
        Ok(())
    }

    pub(crate) fn atom_affine_gauge_weights(
        &self,
        atom_idx: usize,
        rho: Option<&SaeManifoldRho>,
    ) -> Result<Array1<f64>, String> {
        let n = self.n_obs();
        let mut weights = Array1::<f64>::zeros(n);
        for row in 0..n {
            let assignments = match rho {
                Some(rho) => self.assignment.try_assignments_row_for_rho(row, rho)?,
                None => self.assignment.try_assignments_row(row)?,
            };
            let mut w = assignments[atom_idx].max(0.0);
            if let Some(row_weights) = self.row_loss_weights.as_ref() {
                w *= row_weights[row].max(0.0);
            }
            weights[row] = if w.is_finite() { w } else { 0.0 };
        }
        Ok(weights)
    }

    /// #1019 stage 1 — post-fit arc-length (unit-speed) chart canonicalization
    /// for every fitted `d = 1` atom with circle or interval topology.
    ///
    /// Image-frozen and gauge-legal: each eligible atom's latent chart is
    /// replaced by the canonical representative of its `Diff(M)` orbit — the
    /// unit-speed (arc-length) chart for `d = 1` circle/interval atoms, the
    /// minimum-isometry-defect flow chart for `d = 2` torus atoms (#1019
    /// stage 2) — and the decoder is recomposed by exact least squares so the
    /// decoded image is unchanged
    /// ([`crate::terms::sae_chart_canonicalization`]). Atoms whose basis
    /// cannot absorb the reparameterized image within the recomposition
    /// tolerance are left untouched (honest fallback, recorded by the flag
    /// staying `false`).
    ///
    /// The whole pass is additionally gated on the penalized objective: the
    /// canonical state is kept only when the same scalar the line search
    /// minimized does not increase beyond the image-invariance tolerance
    /// (the intrinsic smoothness penalty is reparameterization-invariant by
    /// design, so a genuine increase means the transport went numerically
    /// wrong and the fitted state is restored verbatim).
    ///
    /// Runs automatically from `into_fitted` after the joint fit converges,
    /// before the payload / residual-gauge certificate is assembled — never
    /// a flag (magic-by-default).
    pub fn canonicalize_charts_post_fit(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        analytic_penalties: Option<&AnalyticPenaltyRegistry>,
    ) -> Result<(), String> {
        use crate::terms::sae_chart_canonicalization::{
            CHART_RECOMPOSITION_REL_TOL, CanonicalChartTopology,
        };
        /// Which canonical-representative construction applies to an atom:
        /// arc length for `d = 1` (#1019 stage 1), the minimum-isometry-defect
        /// flow for `d = 2` torus atoms (#1019 stage 2), and the same flow
        /// against the flat reference for `d = 2` free/patch atoms (#1019
        /// free-chart arm — a contractible Euclidean patch admits a global
        /// polynomial flow basis, no hairy-ball obstruction).
        enum ChartPlan {
            UnitSpeed(CanonicalChartTopology),
            TorusFlow { period: f64 },
            PatchFlow,
            SphereFlow,
        }
        let mut eligible: Vec<(usize, ChartPlan)> = Vec::new();
        for atom_idx in 0..self.k_atoms() {
            let atom = &self.atoms[atom_idx];
            if atom.basis_evaluator.is_none()
                || atom.homotopy_eta != 1.0
                || self.assignment.coords[atom_idx].latent_dim() != atom.latent_dim
            {
                continue;
            }
            // Same fraction-of-period convention as the latent manifold
            // wiring (`SaeAtomBasisKind::latent_manifold`): the harmonic
            // evaluators read `t` as a fraction of one period.
            let plan = match (&atom.basis_kind, atom.latent_dim) {
                (SaeAtomBasisKind::Periodic | SaeAtomBasisKind::Torus, 1) => {
                    ChartPlan::UnitSpeed(CanonicalChartTopology::Circle { period: 1.0 })
                }
                (SaeAtomBasisKind::Duchon | SaeAtomBasisKind::EuclideanPatch, 1) => {
                    ChartPlan::UnitSpeed(CanonicalChartTopology::Interval)
                }
                // #1019 stage 2: d = 2 torus atoms pin to the
                // minimum-isometry-defect flow representative.
                (SaeAtomBasisKind::Torus, 2) => ChartPlan::TorusFlow { period: 1.0 },
                // #1019 free-chart arm: d = 2 free/patch (Euclidean-patch)
                // atoms admit a global polynomial flow basis (contractible —
                // no hairy ball), so they pin to the flat uniform-speed
                // minimum-anisotropy-defect representative.
                (SaeAtomBasisKind::Duchon | SaeAtomBasisKind::EuclideanPatch, 2) => {
                    ChartPlan::PatchFlow
                }
                // #1019 sphere arm: d = 2 sphere atoms pin to the
                // minimum-isometry-defect conformal-boost flow against the
                // round-sphere reference. The pin is scoped to data away from
                // the poles (the `1/cos lat` boost singularity — the residue of
                // the hairy-ball obstruction); an at-pole chart is honestly
                // refused by the reparameterization (Ok(false) → left as
                // fitted) rather than pinned with a near-singular flow.
                (SaeAtomBasisKind::Sphere, 2) => ChartPlan::SphereFlow,
                // d = 1 never matches Sphere; Precomputed bases carry no
                // evaluator semantics to re-express the image in.
                _ => continue,
            };
            eligible.push((atom_idx, plan));
        }

        // #1019 sphere arm: d = 2 sphere atoms are now first-class flow-pinned
        // above (ChartPlan::SphereFlow → the round-sphere conformal-boost flow),
        // so the isometry defect is the objective the pin descends rather than a
        // read-only side log. Data inside the pole-band guard (the `1/cos lat`
        // boost singularity — the scoped residue of the hairy-ball obstruction)
        // is honestly refused by the reparameterization and left as fitted.

        // #1026 EV-vs-Θ measurement: log each d = 1 atom's fitted TURNING
        // `Θ = ∫κ ds` (integrated curvature of its decoded curve). A linear SAE
        // shatters a curved feature of turning Θ into `N(ε) ≈ Θ/(2√(2ε))` rank-1
        // atoms at relative error ε, so the curved win is concentrated on high-Θ
        // features and vanishes as Θ → 0. Pairing this with the atom's EV
        // contribution (held-out reconstruction) is the discriminating
        // hybrid-vs-shatter signal: a Θ ≈ 0 atom that still earns EV is a linear
        // direction wearing a curved basis; a high-Θ atom earning EV is a genuine
        // curved family. Pure read-only diagnostic, never mutates the atom.
        //
        // The held-out EV half of the pair is the per-atom leave-one-atom-out
        // (LOAO) explained-variance drop `ΔEV_k` (see
        // [`Self::per_atom_loao_explained_variance`]): the EV lost when atom `k`
        // is withheld from the assembled reconstruction. Logging `(Θ, ΔEV)`
        // together per d = 1 atom is the discriminating EV-vs-Θ signal — a
        // Θ ≈ 0 atom earning a large ΔEV is a linear-tail direction, a high-Θ
        // atom earning a large ΔEV is a genuine curved family.
        let loao_ev = self
            .per_atom_loao_explained_variance(target, rho)
            .unwrap_or_else(|err| {
                log::warn!("[#1026] per-atom LOAO EV unavailable: {err}");
                vec![None; self.k_atoms()]
            });
        for atom_idx in 0..self.k_atoms() {
            let atom = &self.atoms[atom_idx];
            if atom.latent_dim != 1
                || atom.homotopy_eta != 1.0
                || self.assignment.coords[atom_idx].latent_dim() != atom.latent_dim
            {
                continue;
            }
            let Some(evaluator) = atom.basis_evaluator.as_ref().cloned() else {
                continue;
            };
            let coords = self.assignment.coords[atom_idx].as_matrix();
            let row_coords = coords.column(0);
            let dev = loao_ev
                .get(atom_idx)
                .copied()
                .flatten()
                .map_or_else(|| "unavailable".to_string(), |d| format!("{d:.6e}"));
            match crate::terms::sae_chart_canonicalization::d1_atom_fitted_turning(
                evaluator.as_ref(),
                atom.decoder_coefficients.view(),
                row_coords,
            ) {
                Ok(Some(theta)) => log::info!(
                    "[#1026] atom '{}' fitted turning Θ = {theta:.6e} rad, \
                     held-out ΔEV = {dev} \
                     (∫κ ds; 0 = linear-tail direction, 2π = full curved loop; \
                     Θ≈0 + large ΔEV = linear direction, high-Θ + large ΔEV = \
                     genuine curved family — the hybrid-vs-shatter signal)",
                    atom.name
                ),
                Ok(None) => log::info!(
                    "[#1026] atom '{}' fitted turning unavailable, held-out ΔEV = {dev} \
                     (no analytic second jet or degenerate curve)",
                    atom.name
                ),
                Err(err) => {
                    log::warn!("[#1026] atom '{}' fitted turning errored: {err}", atom.name)
                }
            }
        }

        // #1026 — make the curved-vs-linear split LOAD-BEARING. The Θ log above
        // is the read-only diagnostic; here we adjudicate, per eligible d = 1
        // atom, the fitted curved image against its straight (linear
        // special-case) sub-model on the common rank-aware Laplace evidence
        // scale and record the verdict. This is closed-form per atom (the
        // collapsed linear lane — exact penalized LS through the fitted decoded
        // points), so it does NOT re-enter the broken euclidean outer fit path
        // (#1051): no continuation spine, no joint Hessian. The dictionary now
        // honestly reports which atoms earn their curvature and which collapse to
        // the linear tail.
        match self.compute_hybrid_split_report(rho) {
            Ok(report) => {
                if let Some(report) = &report {
                    log::info!(
                        "[#1026] hybrid split: {} curved / {} linear atoms (Σ NLE = {:.6e})",
                        report.selection.curved_atom_count,
                        report.selection.linear_atom_count(),
                        report.selection.total_negative_log_evidence,
                    );
                }
                self.hybrid_split_report = report;
            }
            Err(err) => {
                log::warn!("[#1026] hybrid split report unavailable: {err}");
                self.hybrid_split_report = None;
            }
        }

        if eligible.is_empty() {
            return Ok(());
        }

        let snapshot = self.snapshot_mutable_state();
        let prior_flags: Vec<bool> = self
            .atoms
            .iter()
            .map(|atom| atom.chart_canonicalized)
            .collect();
        let pre_total = self.penalized_objective_total(target, rho, analytic_penalties, 1.0)?;

        let mut any_changed = false;
        for (atom_idx, plan) in &eligible {
            let outcome = match plan {
                ChartPlan::UnitSpeed(topology) => {
                    self.canonicalize_atom_unit_speed_chart(*atom_idx, topology)
                }
                ChartPlan::TorusFlow { period } => {
                    self.canonicalize_atom_torus_flow_chart(*atom_idx, *period)
                }
                ChartPlan::PatchFlow => self.canonicalize_atom_patch_flow_chart(*atom_idx),
                ChartPlan::SphereFlow => self.canonicalize_atom_sphere_flow_chart(*atom_idx),
            };
            match outcome {
                Ok(changed) => any_changed |= changed,
                Err(err) => {
                    self.restore_mutable_state(&snapshot);
                    for (atom, flag) in self.atoms.iter_mut().zip(prior_flags.iter()) {
                        atom.chart_canonicalized = *flag;
                    }
                    return Err(err);
                }
            }
        }
        if !any_changed {
            return Ok(());
        }

        // Keep the canonical state only when the optimized scalar is preserved
        // within the image-invariance tolerance (the data fit moved by at most
        // the certified recomposition residual; the intrinsic penalty is
        // reparameterization-invariant, transported exactly).
        let canonical_total = self.penalized_objective_total(target, rho, analytic_penalties, 1.0);
        let keep = match canonical_total {
            Ok(total) => {
                total.is_finite()
                    && total <= pre_total + CHART_RECOMPOSITION_REL_TOL * (1.0 + pre_total.abs())
            }
            Err(_) => false,
        };
        if !keep {
            self.restore_mutable_state(&snapshot);
            for (atom, flag) in self.atoms.iter_mut().zip(prior_flags.iter()) {
                atom.chart_canonicalized = *flag;
            }
        }
        Ok(())
    }

    /// Apply the arc-length reparameterization to one eligible `d = 1` atom.
    /// Returns `Ok(true)` when the atom was canonicalized, `Ok(false)` on an
    /// honest skip (degenerate chart, basis not closed under the
    /// reparameterization, or per-row image drift above tolerance).
    pub(crate) fn canonicalize_atom_unit_speed_chart(
        &mut self,
        atom_idx: usize,
        topology: &crate::terms::sae_chart_canonicalization::CanonicalChartTopology,
    ) -> Result<bool, String> {
        use crate::terms::sae_chart_canonicalization::{
            CHART_RECOMPOSITION_REL_TOL, unit_speed_reparameterization,
        };
        let n = self.n_obs();
        if n == 0 {
            return Ok(false);
        }
        let Some(evaluator) = self.atoms[atom_idx].basis_evaluator.as_ref().cloned() else {
            return Ok(false);
        };
        let coords = self.assignment.coords[atom_idx].as_matrix();
        let row_coords = coords.column(0).to_owned();
        let Some(repar) = unit_speed_reparameterization(
            evaluator.as_ref(),
            self.atoms[atom_idx].decoder_coefficients.view(),
            row_coords.view(),
            topology,
        )?
        else {
            return Ok(false);
        };

        // Per-row basis/jet at the canonical coordinates.
        let mut new_coords = Array2::<f64>::zeros((n, 1));
        for row in 0..n {
            new_coords[[row, 0]] = repar.new_row_coords[row];
        }
        let (new_phi, new_jet) = if self.atoms[atom_idx].homotopy_eta == 1.0 {
            evaluator.evaluate(new_coords.view())?
        } else {
            let evaluated =
                evaluator.evaluate_phi_eta(new_coords.view(), self.atoms[atom_idx].homotopy_eta)?;
            (evaluated.phi, evaluated.jet)
        };
        if new_phi.dim() != self.atoms[atom_idx].basis_values.dim()
            || new_jet.dim() != self.atoms[atom_idx].basis_jacobian.dim()
        {
            return Err(format!(
                "SaeManifoldTerm::canonicalize_atom_unit_speed_chart: canonical basis {:?} / jet {:?} must match the fitted shapes {:?} / {:?}",
                new_phi.dim(),
                new_jet.dim(),
                self.atoms[atom_idx].basis_values.dim(),
                self.atoms[atom_idx].basis_jacobian.dim()
            ));
        }

        // Per-row image-invariance gate: the grid gate certified the curve at
        // the audit nodes; this certifies it at the coordinates the fit
        // actually sits on. Same honest-fallback contract.
        let old_fit = fast_ab(
            &self.atoms[atom_idx].basis_values,
            &self.atoms[atom_idx].decoder_coefficients,
        );
        let new_fit = fast_ab(&new_phi, &repar.new_decoder);
        let mut fit_scale = 0.0_f64;
        let mut max_abs = 0.0_f64;
        for (a, b) in old_fit.iter().zip(new_fit.iter()) {
            fit_scale = fit_scale.max(a.abs()).max(b.abs());
            max_abs = max_abs.max((a - b).abs());
        }
        if !(fit_scale.is_finite() && max_abs.is_finite()) {
            return Ok(false);
        }
        if fit_scale > 0.0 && max_abs > CHART_RECOMPOSITION_REL_TOL * fit_scale {
            return Ok(false);
        }

        // Commit: canonical coordinates, basis, decoder, and the congruence-
        // transported smoothness Gram (`B̃ᵀ S̃ B̃ = Bᵀ S B`, same as the affine
        // gauge pass).
        let old_smooth_penalty = self.atoms[atom_idx].smooth_penalty.clone();
        let flat = Array1::from_iter(new_coords.iter().copied());
        self.assignment.coords[atom_idx].set_flat(flat.view());
        let atom = &mut self.atoms[atom_idx];
        atom.basis_values = new_phi;
        atom.basis_jacobian = new_jet;
        atom.decoder_coefficients = repar.new_decoder;
        atom.smooth_penalty = transport_smooth_penalty_for_decoder(
            repar.decoder_transport.view(),
            old_smooth_penalty.view(),
        )?;
        atom.chart_canonicalized = true;
        Ok(true)
    }

    /// Apply the minimum-isometry-defect flow reparameterization (#1019
    /// stage 2) to one eligible `d = 2` torus atom. Returns `Ok(true)` when
    /// the atom was canonicalized, `Ok(false)` on an honest skip (degenerate
    /// or already-canonical chart, only folded/non-improving flow candidates,
    /// basis not closed under the reparameterization, or per-row image drift
    /// above tolerance).
    pub(crate) fn canonicalize_atom_torus_flow_chart(
        &mut self,
        atom_idx: usize,
        period: f64,
    ) -> Result<bool, String> {
        use crate::terms::sae_chart_canonicalization::{
            CHART_RECOMPOSITION_REL_TOL, torus_isometry_flow_reparameterization,
        };
        let n = self.n_obs();
        if n == 0 {
            return Ok(false);
        }
        let Some(evaluator) = self.atoms[atom_idx].basis_evaluator.as_ref().cloned() else {
            return Ok(false);
        };
        let coords = self.assignment.coords[atom_idx].as_matrix();
        let Some(repar) = torus_isometry_flow_reparameterization(
            evaluator.as_ref(),
            self.atoms[atom_idx].decoder_coefficients.view(),
            coords.view(),
            period,
        )?
        else {
            return Ok(false);
        };

        // Per-row basis/jet at the canonical coordinates (eligibility pins
        // `homotopy_eta == 1.0`, so the plain evaluate IS the dialed path).
        let new_coords = repar.new_row_coords.clone();
        let (new_phi, new_jet) = evaluator.evaluate(new_coords.view())?;
        if new_phi.dim() != self.atoms[atom_idx].basis_values.dim()
            || new_jet.dim() != self.atoms[atom_idx].basis_jacobian.dim()
        {
            return Err(format!(
                "SaeManifoldTerm::canonicalize_atom_torus_flow_chart: canonical basis {:?} / jet {:?} must match the fitted shapes {:?} / {:?}",
                new_phi.dim(),
                new_jet.dim(),
                self.atoms[atom_idx].basis_values.dim(),
                self.atoms[atom_idx].basis_jacobian.dim()
            ));
        }

        // Per-row image-invariance gate: the audit grid certified the image
        // at the transport nodes; this certifies it at the coordinates the
        // fit actually sits on. Same honest-fallback contract as d = 1.
        let old_fit = fast_ab(
            &self.atoms[atom_idx].basis_values,
            &self.atoms[atom_idx].decoder_coefficients,
        );
        let new_fit = fast_ab(&new_phi, &repar.new_decoder);
        let mut fit_scale = 0.0_f64;
        let mut max_abs = 0.0_f64;
        for (a, b) in old_fit.iter().zip(new_fit.iter()) {
            fit_scale = fit_scale.max(a.abs()).max(b.abs());
            max_abs = max_abs.max((a - b).abs());
        }
        if !(fit_scale.is_finite() && max_abs.is_finite()) {
            return Ok(false);
        }
        if fit_scale > 0.0 && max_abs > CHART_RECOMPOSITION_REL_TOL * fit_scale {
            return Ok(false);
        }

        // Commit: canonical coordinates, basis, decoder, and the congruence-
        // transported smoothness Gram (`B̃ᵀ S̃ B̃ = Bᵀ S B`, same as the affine
        // gauge pass and the d = 1 path).
        let old_smooth_penalty = self.atoms[atom_idx].smooth_penalty.clone();
        let flat = Array1::from_iter(new_coords.iter().copied());
        self.assignment.coords[atom_idx].set_flat(flat.view());
        let atom = &mut self.atoms[atom_idx];
        atom.basis_values = new_phi;
        atom.basis_jacobian = new_jet;
        atom.decoder_coefficients = repar.new_decoder;
        atom.smooth_penalty = transport_smooth_penalty_for_decoder(
            repar.decoder_transport.view(),
            old_smooth_penalty.view(),
        )?;
        atom.chart_canonicalized = true;
        Ok(true)
    }

    /// Apply the minimum-isometry-defect flow reparameterization against the
    /// flat reference (#1019 free-chart arm) to one eligible `d = 2` free/patch
    /// (Euclidean-patch) atom. Returns `Ok(true)` when the atom was
    /// canonicalized, `Ok(false)` on an honest skip (degenerate or
    /// already-canonical chart, only folded/non-improving flow candidates,
    /// basis not closed under the reparameterization, or per-row image drift
    /// above tolerance).
    pub(crate) fn canonicalize_atom_patch_flow_chart(
        &mut self,
        atom_idx: usize,
    ) -> Result<bool, String> {
        use crate::terms::sae_chart_canonicalization::{
            CHART_RECOMPOSITION_REL_TOL, patch_isometry_flow_reparameterization,
        };
        let n = self.n_obs();
        if n == 0 {
            return Ok(false);
        }
        let Some(evaluator) = self.atoms[atom_idx].basis_evaluator.as_ref().cloned() else {
            return Ok(false);
        };
        let coords = self.assignment.coords[atom_idx].as_matrix();
        let Some(repar) = patch_isometry_flow_reparameterization(
            evaluator.as_ref(),
            self.atoms[atom_idx].decoder_coefficients.view(),
            coords.view(),
        )?
        else {
            return Ok(false);
        };

        // Per-row basis/jet at the canonical coordinates (eligibility pins
        // `homotopy_eta == 1.0`, so the plain evaluate IS the dialed path).
        let new_coords = repar.new_row_coords.clone();
        let (new_phi, new_jet) = evaluator.evaluate(new_coords.view())?;
        if new_phi.dim() != self.atoms[atom_idx].basis_values.dim()
            || new_jet.dim() != self.atoms[atom_idx].basis_jacobian.dim()
        {
            return Err(format!(
                "SaeManifoldTerm::canonicalize_atom_patch_flow_chart: canonical basis {:?} / jet {:?} must match the fitted shapes {:?} / {:?}",
                new_phi.dim(),
                new_jet.dim(),
                self.atoms[atom_idx].basis_values.dim(),
                self.atoms[atom_idx].basis_jacobian.dim()
            ));
        }

        // Per-row image-invariance gate: the audit grid certified the image at
        // the transport nodes; this certifies it at the coordinates the fit
        // actually sits on. Same honest-fallback contract as the torus path.
        let old_fit = fast_ab(
            &self.atoms[atom_idx].basis_values,
            &self.atoms[atom_idx].decoder_coefficients,
        );
        let new_fit = fast_ab(&new_phi, &repar.new_decoder);
        let mut fit_scale = 0.0_f64;
        let mut max_abs = 0.0_f64;
        for (a, b) in old_fit.iter().zip(new_fit.iter()) {
            fit_scale = fit_scale.max(a.abs()).max(b.abs());
            max_abs = max_abs.max((a - b).abs());
        }
        if !(fit_scale.is_finite() && max_abs.is_finite()) {
            return Ok(false);
        }
        if fit_scale > 0.0 && max_abs > CHART_RECOMPOSITION_REL_TOL * fit_scale {
            return Ok(false);
        }

        // Commit: canonical coordinates, basis, decoder, and the congruence-
        // transported smoothness Gram (`B̃ᵀ S̃ B̃ = Bᵀ S B`, same as every other
        // canonicalization path).
        let old_smooth_penalty = self.atoms[atom_idx].smooth_penalty.clone();
        let flat = Array1::from_iter(new_coords.iter().copied());
        self.assignment.coords[atom_idx].set_flat(flat.view());
        let atom = &mut self.atoms[atom_idx];
        atom.basis_values = new_phi;
        atom.basis_jacobian = new_jet;
        atom.decoder_coefficients = repar.new_decoder;
        atom.smooth_penalty = transport_smooth_penalty_for_decoder(
            repar.decoder_transport.view(),
            old_smooth_penalty.view(),
        )?;
        atom.chart_canonicalized = true;
        Ok(true)
    }

    /// Apply the minimum-isometry-defect conformal-boost flow reparameterization
    /// against the round-sphere reference (#1019 sphere arm) to one eligible
    /// `d = 2` sphere (`S²`) atom. Returns `Ok(true)` when the atom was
    /// canonicalized, `Ok(false)` on an honest skip (degenerate or
    /// already-canonical chart, data inside the pole-band guard, only
    /// folded/non-improving flow candidates, basis not closed under the
    /// reparameterization, or per-row image drift above tolerance).
    pub(crate) fn canonicalize_atom_sphere_flow_chart(
        &mut self,
        atom_idx: usize,
    ) -> Result<bool, String> {
        use crate::terms::sae_chart_canonicalization::{
            CHART_RECOMPOSITION_REL_TOL, sphere_isometry_flow_reparameterization,
        };
        let n = self.n_obs();
        if n == 0 {
            return Ok(false);
        }
        let Some(evaluator) = self.atoms[atom_idx].basis_evaluator.as_ref().cloned() else {
            return Ok(false);
        };
        let coords = self.assignment.coords[atom_idx].as_matrix();
        let Some(repar) = sphere_isometry_flow_reparameterization(
            evaluator.as_ref(),
            self.atoms[atom_idx].decoder_coefficients.view(),
            coords.view(),
        )?
        else {
            return Ok(false);
        };

        // Per-row basis/jet at the canonical coordinates (eligibility pins
        // `homotopy_eta == 1.0`, so the plain evaluate IS the dialed path).
        let new_coords = repar.new_row_coords.clone();
        let (new_phi, new_jet) = evaluator.evaluate(new_coords.view())?;
        if new_phi.dim() != self.atoms[atom_idx].basis_values.dim()
            || new_jet.dim() != self.atoms[atom_idx].basis_jacobian.dim()
        {
            return Err(format!(
                "SaeManifoldTerm::canonicalize_atom_sphere_flow_chart: canonical basis {:?} / jet {:?} must match the fitted shapes {:?} / {:?}",
                new_phi.dim(),
                new_jet.dim(),
                self.atoms[atom_idx].basis_values.dim(),
                self.atoms[atom_idx].basis_jacobian.dim()
            ));
        }

        // Per-row image-invariance gate: the audit grid certified the image at
        // the transport nodes; this certifies it at the coordinates the fit
        // actually sits on. Same honest-fallback contract as the torus path.
        let old_fit = fast_ab(
            &self.atoms[atom_idx].basis_values,
            &self.atoms[atom_idx].decoder_coefficients,
        );
        let new_fit = fast_ab(&new_phi, &repar.new_decoder);
        let mut fit_scale = 0.0_f64;
        let mut max_abs = 0.0_f64;
        for (a, b) in old_fit.iter().zip(new_fit.iter()) {
            fit_scale = fit_scale.max(a.abs()).max(b.abs());
            max_abs = max_abs.max((a - b).abs());
        }
        if !(fit_scale.is_finite() && max_abs.is_finite()) {
            return Ok(false);
        }
        if fit_scale > 0.0 && max_abs > CHART_RECOMPOSITION_REL_TOL * fit_scale {
            return Ok(false);
        }

        // Commit: canonical coordinates, basis, decoder, and the congruence-
        // transported smoothness Gram (`B̃ᵀ S̃ B̃ = Bᵀ S B`, same as every other
        // canonicalization path).
        let old_smooth_penalty = self.atoms[atom_idx].smooth_penalty.clone();
        let flat = Array1::from_iter(new_coords.iter().copied());
        self.assignment.coords[atom_idx].set_flat(flat.view());
        let atom = &mut self.atoms[atom_idx];
        atom.basis_values = new_phi;
        atom.basis_jacobian = new_jet;
        atom.decoder_coefficients = repar.new_decoder;
        atom.smooth_penalty = transport_smooth_penalty_for_decoder(
            repar.decoder_transport.view(),
            old_smooth_penalty.view(),
        )?;
        atom.chart_canonicalized = true;
        Ok(true)
    }

    /// The iterate scale `1 + ‖(logits, coords, decoder)‖` used to make the
    /// inner KKT gradient and Newton-step tolerances relative. This is the
    /// SINGLE source of truth for that scale: `reml_criterion`'s convergence
    /// gate and `run_joint_fit_arrow_schur`'s non-descent stationarity gate
    /// must agree on it, or a point one of them calls converged is mid-flight
    /// to the other (the objective↔gradient desync class).
    pub(crate) fn inner_iterate_scale(&self) -> f64 {
        let mut iterate_norm_sq = 0.0_f64;
        for &v in self.assignment.logits.iter() {
            iterate_norm_sq += v * v;
        }
        for coords in &self.assignment.coords {
            let matrix = coords.as_matrix();
            for &v in matrix.iter() {
                iterate_norm_sq += v * v;
            }
        }
        for atom in &self.atoms {
            for &v in atom.decoder_coefficients.iter() {
                iterate_norm_sq += v * v;
            }
        }
        1.0 + iterate_norm_sq.sqrt()
    }

    /// Full-length β-block flat directions left by a **rank-deficient decoder
    /// design** (#1051).
    ///
    /// The chart gauge orbit ([`Self::dense_step_gauge_vectors`]) only spans the
    /// per-latent-axis reparametrisation freedom — it never reaches a decoder
    /// column-space deficiency. A euclidean / Duchon patch fit to a shape that
    /// does not excite every monomial column (e.g. a straight line under the
    /// degree-2 patch `[1, t, t²]`: the `t²` column carries no signal) leaves a
    /// genuine flat direction in the β block: a vector `v` with `vᵀG_kv ≈ 0`
    /// **and** `vᵀS_kv ≈ 0`, where `G_k` is the weighted data Gram and `S_k` the
    /// smoothing penalty. Along such a `v` the penalised joint objective has no
    /// curvature, so the undamped Newton step there is unbounded — the inner
    /// solve's raw KKT residual and step never settle, and `reml_criterion`
    /// rejects an otherwise-stationary fit as non-converged (the 122 s line-fit
    /// stall + `1e12` sentinel).
    ///
    /// We identify exactly those directions as the joint null of `G_k + S_k`
    /// (penalty already carries the `λ_smooth` weight installed by
    /// `assemble_arrow_schur`, so a column the penalty regularises is NOT
    /// flagged — only the truly unidentified-and-unpenalised directions are).
    /// Each `M_k`-vector null direction is replicated across the `p` output
    /// channels via the decoder's `⊗ I_p` Kronecker structure and lifted into
    /// the full `(n·q + β)` coordinate so it can be quotiented out of the inner
    /// convergence measure and deflated in the outer gradient identically to a
    /// chart gauge.
    pub(crate) fn decoder_beta_null_directions(
        &self,
        penalized_gram_scale: f64,
    ) -> Result<Vec<Array1<f64>>, String> {
        let p = self.output_dim();
        let n = self.n_obs();
        let q = self.assignment.row_block_dim();
        let beta_dim = self.beta_dim();
        let total_len = n * q + beta_dim;
        if p == 0 || beta_dim == 0 {
            return Ok(Vec::new());
        }
        let mut grams = self.empty_decoder_gram_accumulator();
        self.accumulate_decoder_gram(&mut grams);
        let beta_offsets = self.beta_offsets();
        let mut out = Vec::new();
        for atom_idx in 0..self.k_atoms() {
            let m = self.atoms[atom_idx].basis_size();
            if m == 0 {
                continue;
            }
            // Penalised β-curvature of this atom's data-channel: `G_k + S_k`.
            // `accumulate_decoder_gram` returns the unweighted data Gram and
            // `smooth_penalty` already carries `λ_smooth`-equivalent weighting at
            // the assembled ρ; `penalized_gram_scale` lets the caller match the
            // exact relative weighting the Schur factor used so the null test is
            // computed against the SAME operator whose pivots went singular.
            let gram = &grams[atom_idx];
            let penalty = &self.atoms[atom_idx].smooth_penalty;
            if penalty.dim() != (m, m) {
                continue;
            }
            let mut joint = Array2::<f64>::zeros((m, m));
            for i in 0..m {
                for j in 0..m {
                    joint[[i, j]] = gram[[i, j]] + penalized_gram_scale * penalty[[i, j]];
                }
            }
            // Symmetrise defensively before the eigendecomposition.
            for i in 0..m {
                for j in 0..i {
                    let sym = 0.5 * (joint[[i, j]] + joint[[j, i]]);
                    joint[[i, j]] = sym;
                    joint[[j, i]] = sym;
                }
            }
            let (evals, evecs) = joint
                .eigh(Side::Lower)
                .map_err(|e| format!("decoder_beta_null_directions: eigh failed: {e}"))?;
            let max_eig = evals.iter().fold(0.0_f64, |acc, &v| acc.max(v));
            if !(max_eig > 0.0) {
                continue;
            }
            // A direction is genuinely flat (unidentified by data AND
            // unpenalised) when its penalised curvature is below the standard
            // relative spectral cutoff used across the codebase.
            let null_floor = SAE_DECODER_BETA_NULL_RELATIVE_FLOOR * max_eig;
            let beta_base = n * q + beta_offsets[atom_idx];
            for eig_idx in 0..evals.len() {
                if !(evals[eig_idx].is_finite() && evals[eig_idx] <= null_floor) {
                    continue;
                }
                // One full-length lift per output channel (the `⊗ I_p` replica).
                for out_col in 0..p {
                    let mut dir = Array1::<f64>::zeros(total_len);
                    for col in 0..m {
                        dir[beta_base + col * p + out_col] = evecs[[col, eig_idx]];
                    }
                    out.push(dir);
                }
            }
        }
        Ok(out)
    }

    pub(crate) fn quotient_newton_step_norm_sq(
        &self,
        delta_ext_coord: ArrayView1<'_, f64>,
        delta_beta: ArrayView1<'_, f64>,
        raw_step_norm_sq: f64,
        penalized_gram_scale: f64,
    ) -> Result<f64, String> {
        let n = self.n_obs();
        let q = self.assignment.row_block_dim();
        let beta_dim = self.beta_dim();
        if delta_ext_coord.len() != n * q || delta_beta.len() != beta_dim {
            return Ok(raw_step_norm_sq);
        }
        let mut residual = Array1::<f64>::zeros(delta_ext_coord.len() + delta_beta.len());
        for i in 0..delta_ext_coord.len() {
            residual[i] = delta_ext_coord[i];
        }
        let beta_base = delta_ext_coord.len();
        for i in 0..delta_beta.len() {
            residual[beta_base + i] = delta_beta[i];
        }
        let quotient = self.quotient_residual_norm_sq(residual, penalized_gram_scale)?;
        Ok(if quotient.is_finite() {
            quotient.max(0.0).min(raw_step_norm_sq)
        } else {
            raw_step_norm_sq
        })
    }

    /// Norm² of a full-length `(n·q + β)` residual vector after projecting out
    /// BOTH the chart reparametrisation orbit AND the rank-deficient decoder
    /// β-null (#1051/#1117): along either the penalised joint objective is flat,
    /// so a component there is gauge freedom, not un-converged motion. Shared by
    /// the convergence STEP gate ([`Self::quotient_newton_step_norm_sq`]) and the
    /// convergence GRADIENT gate ([`Self::quotient_gradient_norm_sq`]) so both
    /// measure progress on the SAME identified quotient — otherwise a
    /// rank-deficient circle whose only remaining motion is gauge/null crawl is
    /// recognised as converged by the step measure but rejected forever by the
    /// raw-gradient measure, burning the inner refine budget (the #1117 stall).
    pub(crate) fn quotient_residual_norm_sq(
        &self,
        mut residual: Array1<f64>,
        penalized_gram_scale: f64,
    ) -> Result<f64, String> {
        let mut orthonormal: Vec<Array1<f64>> = Vec::new();
        let gauges = self
            .dense_step_gauge_vectors()?
            .into_iter()
            .chain(self.decoder_beta_null_directions(penalized_gram_scale)?);
        for mut gauge in gauges {
            for basis in &orthonormal {
                let coeff = gauge.dot(basis);
                for i in 0..gauge.len() {
                    gauge[i] -= coeff * basis[i];
                }
            }
            let norm_sq = gauge.iter().map(|v| v * v).sum::<f64>();
            if norm_sq <= 1.0e-24 || !norm_sq.is_finite() {
                continue;
            }
            let inv_norm = norm_sq.sqrt().recip();
            for v in gauge.iter_mut() {
                *v *= inv_norm;
            }
            let coeff = residual.dot(&gauge);
            for i in 0..residual.len() {
                residual[i] -= coeff * gauge[i];
            }
            orthonormal.push(gauge);
        }
        Ok(residual.iter().map(|v| v * v).sum::<f64>())
    }

    /// Quotient KKT-gradient norm² for the inner convergence gate (#1117): the
    /// joint gradient `[g_t (per row); g_β]` with the chart-gauge orbit and the
    /// rank-deficient decoder β-null projected out, so a rank-deficient atom
    /// whose residual gradient lives ONLY in those flat directions is recognised
    /// as stationary on the identified quotient manifold.
    ///
    /// `grad_ext_coord` is the per-row `g_t` flattened into the dense `n·q`
    /// coordinate layout (row `i` axis `a` at `i·q + a`); `grad_beta` is `g_β`.
    /// Falls back to the raw norm when the layout does not match the dense gauge
    /// basis (e.g. a streaming/heterogeneous system), exactly as the step measure
    /// does, so non-dense paths are unaffected.
    pub(crate) fn quotient_gradient_norm_sq(
        &self,
        grad_ext_coord: ArrayView1<'_, f64>,
        grad_beta: ArrayView1<'_, f64>,
        raw_grad_norm_sq: f64,
        penalized_gram_scale: f64,
    ) -> Result<f64, String> {
        let n = self.n_obs();
        let q = self.assignment.row_block_dim();
        let beta_dim = self.beta_dim();
        if grad_ext_coord.len() != n * q || grad_beta.len() != beta_dim {
            return Ok(raw_grad_norm_sq);
        }
        let mut residual = Array1::<f64>::zeros(grad_ext_coord.len() + grad_beta.len());
        for i in 0..grad_ext_coord.len() {
            residual[i] = grad_ext_coord[i];
        }
        let beta_base = grad_ext_coord.len();
        for i in 0..grad_beta.len() {
            residual[beta_base + i] = grad_beta[i];
        }
        let quotient = self.quotient_residual_norm_sq(residual, penalized_gram_scale)?;
        Ok(if quotient.is_finite() {
            quotient.max(0.0).min(raw_grad_norm_sq)
        } else {
            raw_grad_norm_sq
        })
    }

    pub(crate) fn dense_step_gauge_vectors(&self) -> Result<Vec<Array1<f64>>, String> {
        let n = self.n_obs();
        let q = self.assignment.row_block_dim();
        let p = self.output_dim();
        let coord_offsets = self.assignment.coord_offsets();
        let beta_offsets = self.beta_offsets();
        let total_len = n * q + self.beta_dim();
        let mut out = Vec::new();
        for atom_idx in 0..self.k_atoms() {
            let d = self.assignment.coords[atom_idx].latent_dim();
            let coords = self.assignment.coords[atom_idx].as_matrix();
            match self.atoms[atom_idx].basis_kind {
                // The Poincaré tangent patch shares the Euclidean patch's
                // translation + scale gauge orbit on the tangent coordinate
                // (the hyperbolic structure lives in the penalty, not the
                // gauge), so it deflates the same step-gauge vectors.
                SaeAtomBasisKind::EuclideanPatch | SaeAtomBasisKind::Poincare => {
                    for axis in 0..d {
                        let mut field = Array2::<f64>::zeros((n, d));
                        field.column_mut(axis).fill(1.0);
                        if let Some(g) = self.dense_step_gauge_vector_from_field(
                            atom_idx,
                            field.view(),
                            &coord_offsets,
                            &beta_offsets,
                            total_len,
                        )? {
                            out.push(g);
                        }
                    }
                    for axis in 0..d {
                        let mut field = Array2::<f64>::zeros((n, d));
                        for row in 0..n {
                            field[[row, axis]] = coords[[row, axis]];
                        }
                        if let Some(g) = self.dense_step_gauge_vector_from_field(
                            atom_idx,
                            field.view(),
                            &coord_offsets,
                            &beta_offsets,
                            total_len,
                        )? {
                            out.push(g);
                        }
                    }
                }
                SaeAtomBasisKind::Duchon => {
                    for axis in 0..d {
                        let mut field = Array2::<f64>::zeros((n, d));
                        field.column_mut(axis).fill(1.0);
                        if let Some(g) = self.dense_step_gauge_vector_from_field(
                            atom_idx,
                            field.view(),
                            &coord_offsets,
                            &beta_offsets,
                            total_len,
                        )? {
                            out.push(g);
                        }
                    }
                    for axis in 0..d {
                        let mut field = Array2::<f64>::zeros((n, d));
                        for row in 0..n {
                            field[[row, axis]] = coords[[row, axis]];
                        }
                        if let Some(g) = self.dense_step_gauge_vector_from_field(
                            atom_idx,
                            field.view(),
                            &coord_offsets,
                            &beta_offsets,
                            total_len,
                        )? {
                            out.push(g);
                        }
                    }
                }
                SaeAtomBasisKind::Periodic | SaeAtomBasisKind::Torus => {
                    for axis in 0..d {
                        let mut field = Array2::<f64>::zeros((n, d));
                        field.column_mut(axis).fill(1.0);
                        if let Some(g) = self.dense_step_gauge_vector_from_field(
                            atom_idx,
                            field.view(),
                            &coord_offsets,
                            &beta_offsets,
                            total_len,
                        )? {
                            out.push(g);
                        }
                    }
                }
                // `Cylinder` (`S¹ × ℝ`) carries exactly one continuous gauge: the
                // shift (rotation) of the periodic axis 0. The line axis 1 has no
                // rotational gauge and its translation is pinned by the constant
                // column, so we deflate only the axis-0 constant-shift field —
                // matching the `AtomTopology::Circle` identifiability choice.
                SaeAtomBasisKind::Cylinder => {
                    let mut field = Array2::<f64>::zeros((n, d));
                    if d > 0 {
                        field.column_mut(0).fill(1.0);
                    }
                    if let Some(g) = self.dense_step_gauge_vector_from_field(
                        atom_idx,
                        field.view(),
                        &coord_offsets,
                        &beta_offsets,
                        total_len,
                    )? {
                        out.push(g);
                    }
                }
                _ => {}
            }
        }
        if p == 0 {
            return Ok(Vec::new());
        }
        Ok(out)
    }

    pub(crate) fn row_gauge_deflation_for_layout(
        &self,
        row_layout: Option<&SaeRowLayout>,
    ) -> Option<ArrowRowGaugeDeflation> {
        let n = self.n_obs();
        let mut rows: Vec<Vec<Array1<f64>>> = Vec::with_capacity(n);
        for row in 0..n {
            let q_row = match row_layout {
                Some(layout) => layout.row_q_active(row),
                None => self.assignment.row_block_dim(),
            };
            rows.push(Vec::with_capacity(self.k_atoms().min(4)));
            match row_layout {
                Some(layout) => {
                    for (active_pos, &atom_idx) in layout.active_atoms[row].iter().enumerate() {
                        let start = layout.coord_starts[row][active_pos];
                        self.push_atom_row_gauge_deflations(
                            &mut rows[row],
                            row,
                            atom_idx,
                            start,
                            q_row,
                        );
                    }
                }
                None => {
                    let coord_offsets = self.assignment.coord_offsets();
                    for atom_idx in 0..self.k_atoms() {
                        self.push_atom_row_gauge_deflations(
                            &mut rows[row],
                            row,
                            atom_idx,
                            coord_offsets[atom_idx],
                            q_row,
                        );
                    }
                }
            }
        }
        if rows.iter().all(Vec::is_empty) {
            None
        } else {
            Some(ArrowRowGaugeDeflation::new(rows))
        }
    }

    pub(crate) fn push_atom_row_gauge_deflations(
        &self,
        row_dirs: &mut Vec<Array1<f64>>,
        row: usize,
        atom_idx: usize,
        coord_start: usize,
        q_row: usize,
    ) {
        let d = self.assignment.coords[atom_idx].latent_dim();
        let mut tangent = vec![0.0_f64; self.output_dim()];
        match self.atoms[atom_idx].basis_kind {
            SaeAtomBasisKind::EuclideanPatch
            | SaeAtomBasisKind::Duchon
            | SaeAtomBasisKind::Poincare => {
                for axis in 0..d {
                    self.atoms[atom_idx].fill_decoded_derivative_row(row, axis, &mut tangent);
                    if tangent.iter().map(|&v| v * v).sum::<f64>() <= 1.0e-24 {
                        continue;
                    }
                    let mut translation = Array1::<f64>::zeros(q_row);
                    translation[coord_start + axis] = 1.0;
                    row_dirs.push(translation);

                    let coord_value = self.assignment.coords[atom_idx].as_matrix()[[row, axis]];
                    let mut scale = Array1::<f64>::zeros(q_row);
                    scale[coord_start + axis] = coord_value;
                    row_dirs.push(scale);
                }
            }
            SaeAtomBasisKind::Periodic | SaeAtomBasisKind::Torus => {
                for axis in 0..d {
                    self.atoms[atom_idx].fill_decoded_derivative_row(row, axis, &mut tangent);
                    if tangent.iter().map(|&v| v * v).sum::<f64>() <= 1.0e-24 {
                        continue;
                    }
                    let mut phase = Array1::<f64>::zeros(q_row);
                    phase[coord_start + axis] = 1.0;
                    row_dirs.push(phase);
                }
            }
            // `Cylinder` (`S¹ × ℝ`): only the periodic axis 0 carries a phase
            // (rotation) gauge; the line axis 1 has none (matching the
            // `AtomTopology::Circle` choice). Deflate the axis-0 phase only.
            SaeAtomBasisKind::Cylinder => {
                if d > 0 {
                    self.atoms[atom_idx].fill_decoded_derivative_row(row, 0, &mut tangent);
                    if tangent.iter().map(|&v| v * v).sum::<f64>() > 1.0e-24 {
                        let mut phase = Array1::<f64>::zeros(q_row);
                        phase[coord_start] = 1.0;
                        row_dirs.push(phase);
                    }
                }
            }
            _ => {}
        }
    }

    pub(crate) fn dense_step_gauge_vector_from_field(
        &self,
        atom_idx: usize,
        field: ArrayView2<'_, f64>,
        coord_offsets: &[usize],
        beta_offsets: &[usize],
        total_len: usize,
    ) -> Result<Option<Array1<f64>>, String> {
        let n = self.n_obs();
        let q = self.assignment.row_block_dim();
        let p = self.output_dim();
        let atom = &self.atoms[atom_idx];
        let m = atom.basis_size();
        let d = self.assignment.coords[atom_idx].latent_dim();
        if field.dim() != (n, d) {
            return Err(format!(
                "dense_step_gauge_vector_from_field: field shape {:?} != ({n}, {d})",
                field.dim()
            ));
        }
        let mut design = Array2::<f64>::zeros((n, m));
        let mut motion = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            let assignments = self.assignment.try_assignments_row(row)?;
            let a = assignments[atom_idx];
            if a == 0.0 {
                continue;
            }
            for col in 0..m {
                design[[row, col]] = a * atom.basis_values[[row, col]];
            }
            for axis in 0..d {
                let dt = field[[row, axis]];
                if dt == 0.0 {
                    continue;
                }
                for col in 0..m {
                    let w = a * dt * atom.basis_jacobian[[row, col, axis]];
                    if w == 0.0 {
                        continue;
                    }
                    for out_col in 0..p {
                        motion[[row, out_col]] += w * atom.decoder_coefficients[[col, out_col]];
                    }
                }
            }
        }
        let raw = motion.iter().map(|v| v * v).sum::<f64>();
        if raw <= f64::MIN_POSITIVE || !raw.is_finite() {
            return Ok(None);
        }
        motion.mapv_inplace(|v| -v);
        let delta_b = solve_design_least_squares(design.view(), motion.view())?;
        let mut gauge = Array1::<f64>::zeros(total_len);
        for row in 0..n {
            let row_base = row * q + coord_offsets[atom_idx];
            for axis in 0..d {
                gauge[row_base + axis] = field[[row, axis]];
            }
        }
        let beta_base = n * q + beta_offsets[atom_idx];
        for col in 0..m {
            for out_col in 0..p {
                gauge[beta_base + col * p + out_col] = delta_b[[col, out_col]];
            }
        }
        Ok(Some(gauge))
    }

    /// #976 Layer-1 guard ledger for the most recent joint fit (empty when no
    /// atom ever breached the active-mass floor). A terminal event here is the
    /// canonical death-proposal feed for the structure search.
    pub fn collapse_events(&self) -> &[CollapseEvent] {
        &self.collapse_events
    }

    /// Record an externally-observed collapse event on this term's guard ledger
    /// (#976/#997). The joint fit appends its own events during
    /// [`Self::run_joint_fit_arrow_schur`]; this lets a structure-search driver
    /// (or a streaming chunk loop reconciling per-chunk guard outcomes) feed a
    /// collapse observation back onto the term so the next
    /// [`crate::solver::structure_harvest::harvest_move_proposals`] pass sees it
    /// as a death trigger.
    pub fn record_collapse_event(&mut self, event: CollapseEvent) {
        self.collapse_events.push(event);
    }

    /// #1023 final fitted-data guard: a fit with material active atoms whose
    /// fitted matrix explains essentially none of the training variation is a
    /// structural collapse, not a quiet success. Record terminal CollapseEvents
    /// so the #976 structure-search layer and payload ledger see the outcome.
    pub fn record_fit_data_collapse_if_needed(
        &mut self,
        target: ArrayView2<'_, f64>,
        fitted: ArrayView2<'_, f64>,
        assignments: ArrayView2<'_, f64>,
        iteration: usize,
    ) -> Result<bool, String> {
        if target.dim() != fitted.dim() {
            return Err(format!(
                "SaeManifoldTerm::record_fit_data_collapse_if_needed: target {:?} != fitted {:?}",
                target.dim(),
                fitted.dim()
            ));
        }
        let (n, p) = target.dim();
        if assignments.dim() != (n, self.k_atoms()) {
            return Err(format!(
                "SaeManifoldTerm::record_fit_data_collapse_if_needed: assignments {:?} != ({}, {})",
                assignments.dim(),
                n,
                self.k_atoms()
            ));
        }
        if n == 0 || p == 0 || self.k_atoms() == 0 {
            return Ok(false);
        }

        let mut means = vec![0.0_f64; p];
        for col in 0..p {
            let mut acc = 0.0;
            for row in 0..n {
                acc += target[[row, col]];
            }
            means[col] = acc / n as f64;
        }
        let mut ssr = 0.0_f64;
        let mut sst = 0.0_f64;
        for row in 0..n {
            for col in 0..p {
                let r = target[[row, col]] - fitted[[row, col]];
                ssr += r * r;
                let centered = target[[row, col]] - means[col];
                sst += centered * centered;
            }
        }
        if !(ssr.is_finite() && sst.is_finite()) || sst <= f64::MIN_POSITIVE {
            return Ok(false);
        }
        let ev = 1.0 - ssr / sst;
        if !(ev.is_finite() && ev <= SAE_FIT_DATA_COLLAPSE_EV_FLOOR) {
            return Ok(false);
        }

        let mut collapsed_active_atom = false;
        for atom in 0..self.k_atoms() {
            let active_mass = assignments
                .column(atom)
                .iter()
                .copied()
                .fold(0.0_f64, f64::max);
            if active_mass <= 1.0e-8 {
                continue;
            }
            collapsed_active_atom = true;
            let already_terminal = self
                .collapse_events
                .iter()
                .any(|e| e.atom == atom && e.action == CollapseAction::Terminal);
            if already_terminal {
                continue;
            }
            self.collapse_events.push(CollapseEvent {
                iteration,
                atom,
                max_active_mass: ev,
                floor: SAE_FIT_DATA_COLLAPSE_EV_FLOOR,
                action: CollapseAction::Terminal,
            });
        }
        Ok(collapsed_active_atom)
    }

    /// Set the curvature-homotopy dial `η ∈ [0, 1]` on every atom (#1007). At
    /// the default `η = 1` the basis is the full curved basis; `η = 0` is the
    /// linear (Eckart-Young) relaxation. The next `refresh_basis` — which every
    /// joint-fit entry point runs — installs the dialed basis, so the dial takes
    /// effect on the following corrector solve. Errors on a non-finite or
    /// out-of-range `η`.
    pub fn set_homotopy_eta(&mut self, eta: f64) -> Result<(), String> {
        if !(eta.is_finite() && (0.0..=1.0).contains(&eta)) {
            return Err(format!(
                "SaeManifoldTerm::set_homotopy_eta: η must be finite in [0, 1]; got {eta}"
            ));
        }
        for atom in &mut self.atoms {
            atom.homotopy_eta = eta;
        }
        Ok(())
    }

    /// The most recent curvature-homotopy entry walk outcome (#1007), or `None`
    /// when no walk has run on this term. Read off the fitted term so the
    /// arrival / bifurcation / collapse outcome is observable.
    pub fn curvature_walk_report(&self) -> Option<&CurvatureWalkReport> {
        self.curvature_walk_report.as_ref()
    }

    /// Record the curvature-homotopy walk outcome on the fit payload (#1007).
    pub fn set_curvature_walk_report(&mut self, report: CurvatureWalkReport) {
        self.curvature_walk_report = Some(report);
    }

    /// Per-row reconstruction residual `r_i = fitted_i − z_i` of the current
    /// `(gates, coords, decoder)` state against `target`, in the term's native
    /// `(n, p)` layout. The curvature-homotopy predictor (#1007) contracts this
    /// against `∂Φ/∂η` to form the data-fit half of `∂g_β/∂η`.
    pub(crate) fn reconstruction_residual(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
    ) -> Result<Array2<f64>, String> {
        let fitted = self.try_fitted_for_rho(rho)?;
        if fitted.dim() != target.dim() {
            return Err(format!(
                "SaeManifoldTerm::reconstruction_residual: fitted {:?} != target {:?}",
                fitted.dim(),
                target.dim()
            ));
        }
        Ok(&fitted - &target)
    }

    /// True when the curvature-homotopy `η` dial cannot move the basis: no
    /// atom evaluator declares curved columns (caller-managed atoms have no
    /// evaluator, hence no split — equally immovable). A one-harmonic periodic
    /// bank (`M = 3`) is the canonical case: constant + fundamental are all
    /// linear columns. Combined with an all-zero isometry ramp this makes the
    /// entry walk's corrector problem η-invariant, which
    /// [`SaeManifoldOuterObjective::run_curvature_homotopy_entry_at_rho`] uses
    /// to collapse the η-grid to its first corrector.
    pub(crate) fn curvature_homotopy_eta_is_inert(&self) -> Result<bool, String> {
        for atom in &self.atoms {
            if let Some(evaluator) = atom.basis_evaluator.as_ref()
                && !evaluator
                    .phi_eta_split(atom.basis_size())?
                    .curved_cols
                    .is_empty()
            {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Per-atom curved-column basis derivative `∂Φ^η/∂η` (#1007): the raw
    /// (un-dialed) basis on each evaluator's *curved* columns and zero on the
    /// linear columns and on caller-managed atoms (no evaluator → no split).
    /// This is the η-independent derivative channel, so it is exact at any
    /// current `η`.
    pub(crate) fn curvature_basis_eta_derivatives(&self) -> Result<Vec<Array2<f64>>, String> {
        let n = self.n_obs();
        let mut out = Vec::with_capacity(self.k_atoms());
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let m = atom.basis_size();
            let mut d = Array2::<f64>::zeros((n, m));
            if let Some(evaluator) = atom.basis_evaluator.as_ref() {
                let split = evaluator.phi_eta_split(m)?;
                if !split.curved_cols.is_empty() {
                    let coords = self.assignment.coords[atom_idx].as_matrix();
                    let (phi_raw, _jet) = evaluator.evaluate(coords.view())?;
                    for &col in &split.curved_cols {
                        for row in 0..n {
                            d[[row, col]] = phi_raw[[row, col]];
                        }
                    }
                }
            }
            out.push(d);
        }
        Ok(out)
    }

    /// Build the β-block of the curvature-homotopy predictor RHS `∂g_β/∂η`
    /// (#1007) at the current corrected state, in the flat β layout
    /// [`Self::flatten_beta`] uses (`[atom][basis_col · p + out_col]`).
    ///
    /// The data-fit β-gradient is `g_β[k,μ,c] = Σ_i a_ik Φ^η_k[i,μ] r_i[c]`, so
    /// (W = I for the Gaussian reconstruction channel)
    /// `∂g_β/∂η[k,μ,c] = Σ_i a_ik (∂Φ^η_k[i,μ]/∂η) r_i[c]`
    /// `              + Σ_i a_ik Φ^η_k[i,μ] (∂r_i[c]/∂η)`,
    /// with `∂Φ^η/∂η` the raw curved-column basis (zero on linear columns) and
    /// `∂r_i/∂η = Σ_{k'} a_ik' (∂Φ^η_{k'}[i,:]/∂η) · B_{k'}`. The smoothness and
    /// ARD penalties do not depend on `η`, so they contribute nothing. The
    /// predictor solves `Δβ = −H⁻¹ · ∂g_β/∂η · Δη` on the cached evidence factor.
    pub(crate) fn curvature_beta_gradient_eta_derivative(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
    ) -> Result<Array1<f64>, String> {
        let n = self.n_obs();
        let p = self.output_dim();
        let offsets = self.beta_offsets();
        let residual = self.reconstruction_residual(target, rho)?;
        let dphi_deta = self.curvature_basis_eta_derivatives()?;
        // ∂fitted_i/∂η = Σ_{k'} a_ik' (dΦ_{k'}[i,:]) · B_{k'}.
        let mut dfitted = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            let a = self.assignment.try_assignments_row_for_rho(row, rho)?;
            for (atom_idx, atom) in self.atoms.iter().enumerate() {
                let a_k = a[atom_idx];
                if a_k == 0.0 {
                    continue;
                }
                let m = atom.basis_size();
                for mu in 0..m {
                    let dphi = dphi_deta[atom_idx][[row, mu]];
                    if dphi == 0.0 {
                        continue;
                    }
                    let w = a_k * dphi;
                    for c in 0..p {
                        dfitted[[row, c]] += w * atom.decoder_coefficients[[mu, c]];
                    }
                }
            }
        }
        // ∂g_β/∂η[k,μ,c] = Σ_i a_ik (dΦ_k[i,μ] r_i[c] + Φ^η_k[i,μ] dfitted_i[c]).
        let mut out = Array1::<f64>::zeros(self.beta_dim());
        for row in 0..n {
            let a = self.assignment.try_assignments_row_for_rho(row, rho)?;
            for (atom_idx, atom) in self.atoms.iter().enumerate() {
                let a_k = a[atom_idx];
                if a_k == 0.0 {
                    continue;
                }
                let m = atom.basis_size();
                let off = offsets[atom_idx];
                for mu in 0..m {
                    let dphi = dphi_deta[atom_idx][[row, mu]];
                    let phi = atom.basis_values[[row, mu]];
                    for c in 0..p {
                        out[off + mu * p + c] +=
                            a_k * (dphi * residual[[row, c]] + phi * dfitted[[row, c]]);
                    }
                }
            }
        }
        Ok(out)
    }

    /// #976 Layer-1 guard 3: the per-atom active-mass floor, checked once per
    /// accepted outer iteration of the joint fit.
    ///
    /// The collapse statistic is each atom's MAXIMUM assignment mass over rows
    /// (see [`SAE_ATOM_ACTIVE_MASS_FLOOR`] for why max, not mean). A breach is
    /// answered with a gate-logit re-seed — once per atom per fit
    /// ([`SAE_ATOM_COLLAPSE_RESEED_BUDGET`]) — and recorded as a
    /// [`CollapseEvent`]; a breach after the budget is recorded once as
    /// terminal and otherwise left alone: at that point the collapse is the
    /// objective's (local) verdict at the current hyperparameters, and the
    /// keep-or-kill decision belongs to the evidence-gated structure search,
    /// not to an inner-loop heuristic. Observable events, never silent deaths,
    /// never fit errors.
    pub(crate) fn enforce_active_mass_guard(
        &mut self,
        iteration: usize,
        rho: Option<&SaeManifoldRho>,
    ) -> Result<(), String> {
        let n = self.n_obs();
        let k = self.k_atoms();
        if n == 0 || k == 0 {
            return Ok(());
        }
        let mut max_mass = vec![0.0_f64; k];
        for row in 0..n {
            let a = match rho {
                Some(rho) => self.assignment.try_assignments_row_for_rho(row, rho),
                None => self.assignment.try_assignments_row(row),
            }
            .map_err(|e| format!("SaeManifoldTerm::enforce_active_mass_guard: {e}"))?;
            for atom in 0..k {
                if a[atom] > max_mass[atom] {
                    max_mass[atom] = a[atom];
                }
            }
        }
        for atom in 0..k {
            if max_mass[atom] >= SAE_ATOM_ACTIVE_MASS_FLOOR {
                continue;
            }
            let reseeds_used = self
                .collapse_events
                .iter()
                .filter(|e| e.atom == atom && e.action == CollapseAction::Reseeded)
                .count();
            if reseeds_used < SAE_ATOM_COLLAPSE_RESEED_BUDGET {
                self.reseed_collapsed_atom_logits(atom);
                self.collapse_events.push(CollapseEvent {
                    iteration,
                    atom,
                    max_active_mass: max_mass[atom],
                    floor: SAE_ATOM_ACTIVE_MASS_FLOOR,
                    action: CollapseAction::Reseeded,
                });
            } else {
                let already_terminal = self
                    .collapse_events
                    .iter()
                    .any(|e| e.atom == atom && e.action == CollapseAction::Terminal);
                if !already_terminal {
                    self.collapse_events.push(CollapseEvent {
                        iteration,
                        atom,
                        max_active_mass: max_mass[atom],
                        floor: SAE_ATOM_ACTIVE_MASS_FLOOR,
                        action: CollapseAction::Terminal,
                    });
                }
            }
        }
        Ok(())
    }

    /// Re-seed one collapsed atom's gate logits to the mode-appropriate
    /// neutral that restores material support — the data-fit term can then
    /// hold the atom active iff it carries signal. Latent coordinates are
    /// deliberately left untouched: gate-driven collapse kills the support,
    /// not the (still data-adjacent) coordinates, and a coordinate re-seed
    /// would discard exactly the warm state that makes the second chance
    /// cheap.
    pub(crate) fn reseed_collapsed_atom_logits(&mut self, atom: usize) {
        let n = self.n_obs();
        match self.assignment.mode {
            AssignmentMode::Softmax { .. } => {
                // Tie the re-seeded atom with each row's current winner so it
                // re-enters the simplex at parity instead of inheriting a
                // saturated deficit.
                for row in 0..n {
                    let row_max = self
                        .assignment
                        .logits
                        .row(row)
                        .iter()
                        .copied()
                        .fold(f64::NEG_INFINITY, f64::max);
                    self.assignment.logits[[row, atom]] =
                        if row_max.is_finite() { row_max } else { 0.0 };
                }
                canonicalize_softmax_logits(&mut self.assignment.logits);
            }
            AssignmentMode::IBPMap { .. } => {
                // σ(0/τ) = ½ — the gate's neutral point; the IBP prior π_k
                // still applies its geometric damping, as it should.
                for row in 0..n {
                    self.assignment.logits[[row, atom]] = 0.0;
                }
            }
            AssignmentMode::JumpReLU {
                temperature,
                threshold,
            } => {
                // One temperature unit above the hard gate threshold:
                // just-active, inside the smooth transition band.
                for row in 0..n {
                    self.assignment.logits[[row, atom]] = threshold + temperature;
                }
            }
        }
    }

    /// #976 Layer-1 guard (decoder arm): the per-atom **decoder-norm** floor,
    /// checked once per accepted outer iteration of the joint K>1 fit.
    ///
    /// The gate-mass guard ([`Self::enforce_active_mass_guard`]) catches an atom
    /// whose *assignment* support vanished, but it is blind to the real-data K>1
    /// failure (#853/#976 class) where the assignment gates stay spread across
    /// rows yet every atom's *decoder* `B_k` collapses to ≈0. A zero decoder
    /// decodes nothing, so the dictionary explains nothing (EV≈0) and every
    /// per-row coordinate Hessian `H_tt` — whose curvature is carried by `Φ·B`
    /// — goes rank-deficient at once, surfacing as the `0 → K·n` evidence
    /// gauge-deflation jump that aborts `reml_criterion`. The decoder-norm guard
    /// closes that blind spot.
    ///
    /// The collapse statistic is each atom's decoder Frobenius norm as a RATIO
    /// to the dictionary's MEDIAN decoder norm (scale-free: a uniformly small
    /// but well-conditioned decoder never trips it; only an atom that has fallen
    /// far behind its peers does). A breach is answered, within the SAME shared
    /// per-atom budget as the mass guard, by re-diversifying the atom's latent
    /// coordinates onto a distinct, currently-unexplained direction of the
    /// reconstruction residual and re-fitting all decoders by joint least
    /// squares so the reseeded atom claims real signal rather than re-collapsing
    /// — then recorded as a [`CollapseEvent`]. After the budget a breach is
    /// recorded once as terminal and left for the evidence-gated structure
    /// search, exactly like the mass arm.
    ///
    /// **K=1 is a strict no-op**: with a single atom there is no peer to fall
    /// behind, the median equals the atom's own norm, the ratio is exactly `1`,
    /// and the early return below fires before any state is touched. The K=1
    /// fit path is therefore byte-for-byte unchanged.
    pub(crate) fn enforce_decoder_norm_guard(
        &mut self,
        target: ArrayView2<'_, f64>,
        iteration: usize,
        rho: &SaeManifoldRho,
    ) -> Result<(), String> {
        let n = self.n_obs();
        let k = self.k_atoms();
        // A single atom has no dictionary peer to be distinct from, so the
        // decoder-incoherence failure mode this guard catches cannot exist;
        // returning here keeps the K=1 path untouched.
        if n == 0 || k < 2 {
            return Ok(());
        }
        let mut norms = vec![0.0_f64; k];
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let mut acc = 0.0_f64;
            for &value in atom.decoder_coefficients.iter() {
                acc += value * value;
            }
            norms[atom_idx] = acc.sqrt();
        }
        // Median decoder norm: the robust dictionary scale. (A mean would let a
        // few large atoms mask a cluster of collapsed ones.)
        let mut sorted = norms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if k % 2 == 1 {
            sorted[k / 2]
        } else {
            0.5 * (sorted[k / 2 - 1] + sorted[k / 2])
        };
        // No usable scale (every decoder is ≈0, e.g. the cold-start zero seed):
        // the joint solve has not yet placed any signal, so there is nothing to
        // be "behind"; defer to the mass guard / inner solve rather than reseed
        // against an all-zero reference.
        if !(median > 0.0) {
            return Ok(());
        }
        let floor = SAE_ATOM_DECODER_NORM_COLLAPSE_RATIO * median;
        let mut breached: Vec<usize> = Vec::new();
        for atom in 0..k {
            if norms[atom] < floor {
                breached.push(atom);
            }
        }
        if breached.is_empty() {
            // The median-relative test found no atom "behind" its peers, but the
            // whole dictionary can still have CO-collapsed: at K>=2 a degenerate
            // seed/basin can drive EVERY decoder small TOGETHER, so the median
            // collapses with them and no atom is *relatively* behind. The
            // median-relative test is structurally blind to this mode — it is the
            // real-data K=2 failure (both atoms fall into one basin and the fit
            // explains ~0 variance; empirically a different seed or stronger
            // decoder-incoherence repulsion avoids the basin, but the guard must
            // catch it for ANY seed). Detect it ABSOLUTELY from the
            // reconstruction: a dictionary that explains essentially none of the
            // centered target variance has collapsed regardless of relative
            // norms.
            let ev = self.dictionary_reconstruction_ev(target, rho)?;
            if ev >= SAE_DICTIONARY_COLLAPSE_EV_FLOOR {
                return Ok(());
            }
            // Co-collapsed. Reseed all atoms EXCEPT the strongest (kept as an
            // anchor so the reseed targets a non-degenerate residual and the set
            // does not re-symmetrise into the same basin) onto DISTINCT residual
            // PCs below.
            let anchor = (0..k)
                .max_by(|&a, &b| {
                    norms[a]
                        .partial_cmp(&norms[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            breached = (0..k).filter(|&a| a != anchor).collect();
            log::warn!(
                "SaeManifoldTerm: dictionary co-collapse (reconstruction EV={ev:.4} < \
                 {SAE_DICTIONARY_COLLAPSE_EV_FLOOR}) with no relative-norm breach; reseeding \
                 {} of {k} atoms onto distinct residual PCs (anchor atom {anchor})",
                breached.len()
            );
        }
        // Decide which breached atoms still have reseed budget (recording a
        // Reseeded or Terminal collapse event for each), then reseed the budgeted
        // set onto DISTINCT residual PCs in ONE pass. A per-atom top-PC reseed
        // would collide multiple simultaneously-collapsed atoms onto the same
        // residual direction and re-collapse them, so the batch seed (the #671
        // disjoint-PC rule across atom slots) is what actually breaks the basin.
        let mut to_reseed: Vec<usize> = Vec::new();
        for &atom in &breached {
            let reseeds_used = self
                .collapse_events
                .iter()
                .filter(|e| e.atom == atom && e.action == CollapseAction::Reseeded)
                .count();
            if reseeds_used < SAE_ATOM_COLLAPSE_RESEED_BUDGET {
                to_reseed.push(atom);
                self.collapse_events.push(CollapseEvent {
                    iteration,
                    atom,
                    max_active_mass: norms[atom] / median,
                    floor: SAE_ATOM_DECODER_NORM_COLLAPSE_RATIO,
                    action: CollapseAction::Reseeded,
                });
            } else {
                let already_terminal = self
                    .collapse_events
                    .iter()
                    .any(|e| e.atom == atom && e.action == CollapseAction::Terminal);
                if !already_terminal {
                    self.collapse_events.push(CollapseEvent {
                        iteration,
                        atom,
                        max_active_mass: norms[atom] / median,
                        floor: SAE_ATOM_DECODER_NORM_COLLAPSE_RATIO,
                        action: CollapseAction::Terminal,
                    });
                }
            }
        }
        if !to_reseed.is_empty() {
            self.reseed_atoms_onto_distinct_residual_pcs(&to_reseed, target, rho)?;
            for &atom in &to_reseed {
                self.reseed_collapsed_atom_logits(atom);
            }
            // One joint least-squares decoder refit at the re-diversified state
            // gives every atom — the reseeded ones especially — a fresh,
            // non-degenerate decoder that distributes the available signal across
            // the dictionary. Cheap (one n×ΣM_k design solve), run at most once
            // per outer iteration, only when an atom actually breached.
            self.refit_decoder_least_squares_at_current_state(target, Some(rho))?;
        }
        Ok(())
    }

    /// Fraction of the centered target variance the current dictionary explains
    /// (`EV = 1 − ‖fitted − target‖² / ‖target − mean‖²`). Used by
    /// [`Self::enforce_decoder_norm_guard`] as the ABSOLUTE co-collapse signal
    /// that the median-relative decoder-norm test is blind to: when every atom
    /// collapses together the relative test sees nothing, but a dictionary that
    /// explains ~zero variance has unambiguously failed. Column means use
    /// Welford's running update so a huge-but-finite target column cannot
    /// overflow the total-sum-of-squares.
    pub(crate) fn dictionary_reconstruction_ev(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
    ) -> Result<f64, String> {
        let residual = self.reconstruction_residual(target, rho)?;
        let mut ss_res = 0.0_f64;
        for &value in residual.iter() {
            ss_res += value * value;
        }
        let n = target.nrows();
        let mut ss_tot = 0.0_f64;
        for col in 0..target.ncols() {
            let mut mean = 0.0_f64;
            for (count, row) in (0..n).enumerate() {
                let x = target[[row, col]];
                mean += (x - mean) / (count as f64 + 1.0);
            }
            for row in 0..n {
                let dev = target[[row, col]] - mean;
                ss_tot += dev * dev;
            }
        }
        if !(ss_tot > 0.0) {
            // A constant target has zero variance to explain; treat a zero
            // residual as fully explained and anything else as collapsed.
            return Ok(if ss_res > 0.0 { 0.0 } else { 1.0 });
        }
        Ok(1.0 - ss_res / ss_tot)
    }

    /// Reseed a set of collapsed atoms onto DISTINCT principal directions of the
    /// current reconstruction residual in one pass, reusing the production #671
    /// disjoint-PC seeding ([`sae_pca_seed_initial_coords`], which assigns atom
    /// slot `j` its own PC pair). Seeding each collapsed atom independently would
    /// hand them all the SAME top residual PC and re-collapse the set, so the
    /// simultaneous-collapse arm seeds them together. A single-element `atoms`
    /// slice is the one-atom case (the atom seeded onto the top residual PC).
    /// Only the reseeded atoms' coordinates and basis caches move; decoders are
    /// left for the caller's joint LSQ refit.
    pub(crate) fn reseed_atoms_onto_distinct_residual_pcs(
        &mut self,
        atoms: &[usize],
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
    ) -> Result<(), String> {
        if atoms.is_empty() {
            return Ok(());
        }
        let residual = self.reconstruction_residual(target, rho)?;
        let basis_kinds: Vec<SaeAtomBasisKind> =
            atoms.iter().map(|&a| self.atoms[a].basis_kind.clone()).collect();
        let dims: Vec<usize> = atoms.iter().map(|&a| self.atoms[a].latent_dim).collect();
        let seeded = sae_pca_seed_initial_coords(residual.view(), &basis_kinds, &dims)?;
        let n = self.n_obs();
        for (slot, &atom) in atoms.iter().enumerate() {
            let d = dims[slot];
            let mut flat = Array1::<f64>::zeros(n * d);
            for row in 0..n {
                for axis in 0..d {
                    flat[row * d + axis] = seeded[[slot, row, axis]];
                }
            }
            self.assignment.coords[atom].set_flat(flat.view());
            let coords = self.assignment.coords[atom].as_matrix();
            self.atoms[atom].refresh_basis(coords.view())?;
        }
        Ok(())
    }

    pub(crate) fn apply_newton_step_impl(
        &mut self,
        delta_ext_coord: ArrayView1<'_, f64>,
        delta_beta: ArrayView1<'_, f64>,
        step_size: f64,
        refresh_basis: bool,
    ) -> Result<(), String> {
        if !(step_size.is_finite() && step_size > 0.0) {
            return Err(format!(
                "SaeManifoldTerm::apply_newton_step: step_size must be finite and positive; got {step_size}"
            ));
        }
        let n = self.n_obs();
        let q = self.assignment.row_block_dim();
        let k_atoms = self.k_atoms();
        let assignment_dim = self.assignment.assignment_coord_dim();
        // #972 / #977 T1: when the most recent assembly built the factored
        // β-tier, `delta_beta` is a factored ΔC (length `factored_border_dim`)
        // that must be LIFTED through each active frame (`ΔB_k = ΔC_k U_kᵀ`)
        // before being applied to the p-wide decoder. Otherwise it is a plain
        // ΔB of length `beta_dim`. The expected length and the application path
        // both branch on `last_frames_active`.
        let expected_delta_len = if self.last_frames_active {
            self.factored_border_dim()
        } else {
            self.beta_dim()
        };
        if delta_beta.len() != expected_delta_len {
            return Err(format!(
                "SaeManifoldTerm::apply_newton_step: delta_beta length {} != expected {}",
                delta_beta.len(),
                expected_delta_len
            ));
        }

        // When last_row_layout is set (compact active-set mode — JumpReLU
        // gate or large-K IBP truncation), delta_ext_coord uses a
        // variable-stride layout where row i occupies
        // [compact_offset_i .. compact_offset_i + q_active_i].
        // We expand each row back to full-q before applying.
        if let Some(ref layout) = self.last_row_layout.clone() {
            let total_len: usize = (0..n).map(|row| layout.row_q_active(row)).sum();
            if delta_ext_coord.len() != total_len {
                return Err(format!(
                    "SaeManifoldTerm::apply_newton_step: compact delta_ext_coord length {} != expected {}",
                    delta_ext_coord.len(),
                    total_len
                ));
            }
            // Expand compact layout to full-q flat buffer.
            let mut full_delta = vec![0.0_f64; n * q];
            let mut compact_off = 0usize;
            for row in 0..n {
                let q_active = layout.row_q_active(row);
                // Collect compact row (handles both contiguous and strided views).
                let compact_row: Vec<f64> = delta_ext_coord
                    .slice(ndarray::s![compact_off..compact_off + q_active])
                    .iter()
                    .copied()
                    .collect();
                layout.expand_row(row, &compact_row, &mut full_delta[row * q..(row + 1) * q]);
                compact_off += q_active;
            }
            // Apply logits from expanded buffer, clamped to the #976 gate-scale
            // step cap (see SAE_ASSIGNMENT_LOGIT_STEP_CAP_TAUS for the Armijo
            // consistency argument).
            let logit_step_cap =
                SAE_ASSIGNMENT_LOGIT_STEP_CAP_TAUS * self.assignment.mode.temperature();
            for row in 0..n {
                let row_base = row * q;
                for atom_idx in 0..assignment_dim {
                    self.assignment.logits[[row, atom_idx]] += (step_size
                        * full_delta[row_base + atom_idx])
                        .clamp(-logit_step_cap, logit_step_cap);
                }
            }
            // Apply coords from expanded buffer.
            let coord_offsets = self.assignment.coord_offsets();
            for atom_idx in 0..k_atoms {
                let d = self.assignment.coords[atom_idx].latent_dim();
                let mut delta_coord = Array1::<f64>::zeros(n * d);
                for row in 0..n {
                    let row_base = row * q + coord_offsets[atom_idx];
                    for axis in 0..d {
                        delta_coord[row * d + axis] = step_size * full_delta[row_base + axis];
                    }
                }
                self.assignment.coords[atom_idx].retract_flat_delta(delta_coord.view());
                if refresh_basis {
                    let coords = self.assignment.coords[atom_idx].as_matrix();
                    self.atoms[atom_idx].refresh_basis(coords.view())?;
                }
            }
        } else {
            // Dense layout: uniform q per row.
            if delta_ext_coord.len() != n * q {
                return Err(format!(
                    "SaeManifoldTerm::apply_newton_step: delta_ext_coord length {} != expected {}",
                    delta_ext_coord.len(),
                    n * q
                ));
            }
            let coord_offsets = self.assignment.coord_offsets();
            // #976 gate-scale step cap, as in the compact branch above.
            let logit_step_cap =
                SAE_ASSIGNMENT_LOGIT_STEP_CAP_TAUS * self.assignment.mode.temperature();
            for row in 0..n {
                let row_base = row * q;
                for atom_idx in 0..assignment_dim {
                    self.assignment.logits[[row, atom_idx]] += (step_size
                        * delta_ext_coord[row_base + atom_idx])
                        .clamp(-logit_step_cap, logit_step_cap);
                }
            }
            for atom_idx in 0..k_atoms {
                let d = self.assignment.coords[atom_idx].latent_dim();
                let mut delta_coord = Array1::<f64>::zeros(n * d);
                for row in 0..n {
                    let row_base = row * q + coord_offsets[atom_idx];
                    for axis in 0..d {
                        delta_coord[row * d + axis] = step_size * delta_ext_coord[row_base + axis];
                    }
                }
                self.assignment.coords[atom_idx].retract_flat_delta(delta_coord.view());
                if refresh_basis {
                    let coords = self.assignment.coords[atom_idx].as_matrix();
                    self.atoms[atom_idx].refresh_basis(coords.view())?;
                }
            }
        }
        if matches!(self.assignment.mode, AssignmentMode::Softmax { .. }) {
            canonicalize_softmax_logits(&mut self.assignment.logits);
        }

        let mut beta = self.flatten_beta();
        if self.last_frames_active {
            // Factored ΔC → lift to a p-wide ΔB and add `step·ΔB`. For atom `k`,
            // basis row `m`, output channel `i`:
            //   ΔB_k[m,i] = Σ_j ΔC[off_C[k] + m·r_k + j] · U_k[i,j].
            // Un-framed atoms (`U_k = I_p`, `r_k = p`) lift by identity, so a
            // mixed dictionary is handled uniformly. The decoder is then
            // refreshed below via `set_flat_beta` (the authoritative `B_k` is the
            // p-wide flatten; the active frames are re-synced from the decoder by
            // the polar refresh in the joint-fit driver).
            let delta_b = FrameProjection::new(self).lift_border_vec(delta_beta);
            for idx in 0..beta.len() {
                beta[idx] += step_size * delta_b[idx];
            }
        } else {
            for idx in 0..beta.len() {
                beta[idx] += step_size * delta_beta[idx];
            }
        }
        self.set_flat_beta(beta.view())
    }

    pub(crate) fn solve_fixed_decoder_row_step(
        h: ArrayView2<'_, f64>,
        g: ArrayView1<'_, f64>,
        base_ridge: f64,
    ) -> Result<Array1<f64>, String> {
        let d = h.nrows();
        if h.ncols() != d || g.len() != d {
            return Err(format!(
                "SaeManifoldTerm::solve_fixed_decoder_row_step: shape mismatch H={:?}, g={}",
                h.dim(),
                g.len()
            ));
        }
        if d == 0 {
            return Ok(Array1::<f64>::zeros(0));
        }
        let mut ridge = base_ridge.max(SAE_MANIFOLD_ROW_RIDGE_FLOOR);
        let mut last_err = String::new();
        for _ in 0..SAE_MANIFOLD_ROW_RIDGE_MAX_ATTEMPTS {
            let mut a = h.to_owned();
            for axis in 0..d {
                a[[axis, axis]] += ridge;
            }
            match sae_cholesky_solve_neg_gradient(a.view(), g) {
                Ok(delta) => return Ok(delta),
                Err(err) => {
                    last_err = err;
                    ridge *= SAE_MANIFOLD_ROW_RIDGE_GROWTH;
                }
            }
        }
        Err(format!(
            "SaeManifoldTerm::solve_fixed_decoder_row_step: row Hessian did not factor after LM escalation; last error: {last_err}"
        ))
    }

    pub(crate) fn fixed_decoder_step_from_rows(
        sys: &ArrowSchurSystem,
        ridge_ext_coord: f64,
    ) -> Result<Array1<f64>, String> {
        let total = sys.row_offsets[sys.rows.len()];
        let mut delta = Array1::<f64>::zeros(total);
        for (row_idx, row) in sys.rows.iter().enumerate() {
            let row_delta =
                Self::solve_fixed_decoder_row_step(row.htt.view(), row.gt.view(), ridge_ext_coord)?;
            let start = sys.row_offsets[row_idx];
            let end = sys.row_offsets[row_idx + 1];
            if row_delta.len() != end - start {
                return Err(format!(
                    "SaeManifoldTerm::fixed_decoder_step_from_rows: row {row_idx} delta len {} != row span {}",
                    row_delta.len(),
                    end - start
                ));
            }
            delta.slice_mut(s![start..end]).assign(&row_delta);
        }
        Ok(delta)
    }

    /// Row visitation order for the discovery/seeding pass, drawn from the
    /// per-row Fisher-mass enrichment measure (#980, role (c)).
    ///
    /// Builds [`RowMeasure::from_metric`](crate::inference::row_measure::RowMeasure::from_metric)
    /// from the term's installed [`RowMetric`] (Euclidean fallback when none is
    /// installed), draws a length-`n` systematic-resampling
    /// [`enrichment_order`](crate::inference::row_measure::RowMeasure::enrichment_order),
    /// and reduces it to a first-seen unique permutation. Behaviorally-live rows
    /// (high Fisher mass) appear earliest; any row the measure never named is
    /// appended in index order so **every** row is still visited exactly once.
    ///
    /// Under a Euclidean / no-harvest metric the measure is exactly uniform, the
    /// systematic-resampling draw is an even round-robin, and the first-seen
    /// reduction is the plain `0..n` index order — bit-for-bit today's behavior.
    ///
    /// Pure attention: the order is consumed only to decide *which row is looked
    /// at first*; each visited row runs the identical unmodified per-row
    /// objective, so this touches no loss / criterion / penalty.
    pub(crate) fn enrichment_visit_order(&self) -> Vec<usize> {
        let n = self.n_obs();
        // No installed metric ⇒ the measure is exactly uniform and the
        // systematic draw reduces to plain index order (documented below), so
        // skip building the Euclidean metric object entirely — this runs in
        // the seeding hot path, per seed-candidate evaluation.
        if self.row_metric.is_none() {
            return (0..n).collect();
        }
        let metric = match self.diagnostic_metric() {
            Ok(m) => m,
            // A metric build failure cannot occur for the term's own validated
            // shape, but degrade to the plain index sweep rather than propagate:
            // the order is attention-only and must never gate the seed.
            Err(_) => return (0..n).collect(),
        };
        let measure = crate::inference::row_measure::RowMeasure::from_metric(&metric);
        // Seed the deterministic systematic-resampling draw from the row count so
        // the ordering is reproducible across runs (no clock randomness).
        let drawn = measure.enrichment_order(n, n as u64);
        let mut order = Vec::with_capacity(n);
        let mut seen = vec![false; n];
        for row in drawn {
            if row < n && !seen[row] {
                seen[row] = true;
                order.push(row);
            }
        }
        // Append any row the enrichment draw never named so every row is seeded.
        for (row, &was_seen) in seen.iter().enumerate() {
            if !was_seen {
                order.push(row);
            }
        }
        order
    }

    /// Globally seed every atom's per-row latent coordinate by projecting each
    /// target row onto that atom's **frozen** decoder image manifold.
    ///
    /// For a fixed decoder the exact out-of-sample encoding of row `i` against
    /// atom `k` is the projection
    /// `t*_{ik} = argmin_t ‖x_i − Φ_k(t)·B_k‖²`. That objective is non-convex
    /// on a compact latent (a trigonometric polynomial for periodic / torus
    /// atoms, a chart function on the sphere), so the cold PCA-`atan2` seed plus
    /// a handful of Newton steps frequently converges into the wrong basin and
    /// mis-routes the row — the root cause of the negative-`R²`, near-uniform
    /// assignment OOS failures. We evaluate each atom's decoder once on a dense
    /// manifold-spanning grid (provided by the atom basis kind), take the per-row
    /// global argmin as the coordinate seed, refresh the atom basis there, and
    /// let the subsequent Newton refinement polish to full precision from inside
    /// the correct basin. Because the residual-based softmax logit seed reads the
    /// freshly decoded rows, routing then follows the true per-atom projection
    /// error rather than the cold-seed error.
    ///
    /// Atoms whose basis kind exposes no projection seed grid
    /// (unbounded / basis-linear latents) are left at their incoming seed. The
    /// decoder, assignment logits, smoothness penalties and ρ are all untouched;
    /// only the latent coordinates and the basis caches that depend on them move.
    pub fn seed_coords_by_decoder_projection(
        &mut self,
        target: ArrayView2<'_, f64>,
        resolution: usize,
    ) -> Result<(), String> {
        let n = self.n_obs();
        let p = self.output_dim();
        if target.dim() != (n, p) {
            return Err(format!(
                "SaeManifoldTerm::seed_coords_by_decoder_projection: target shape {:?} != ({n}, {p})",
                target.dim()
            ));
        }
        // ENRICHMENT (#980, role (c)): the order in which this discovery/seeding
        // pass *visits* rows is drawn from the per-row Fisher-mass sampling
        // measure when an output-Fisher harvest is present, so behaviorally-live
        // rows get attention FIRST. This is attention-only: every visited row
        // runs the identical, unmodified per-row argmin projection objective
        // below — the measure reweights *which row is looked at first*, never the
        // loss. Under a Euclidean / no-harvest metric the measure is exactly
        // uniform, so the order degrades to the plain `0..n` index sweep and the
        // result is bit-for-bit today's behavior. Because each row's seed is
        // computed independently and written exactly once, the visitation order
        // cannot change any seed value — confirming the attention-only invariant.
        let visit_order = self.enrichment_visit_order();
        for atom_idx in 0..self.k_atoms() {
            let d = self.atoms[atom_idx].latent_dim;
            let Some(grid) = self.atoms[atom_idx]
                .basis_kind
                .projection_seed_grid(d, resolution)
            else {
                continue;
            };
            let Some(evaluator) = self.atoms[atom_idx].basis_evaluator.clone() else {
                continue;
            };
            if grid.ncols() != d {
                return Err(format!(
                    "SaeManifoldTerm::seed_coords_by_decoder_projection: atom {atom_idx} grid has {} columns but latent_dim is {d}",
                    grid.ncols()
                ));
            }
            let g = grid.nrows();
            if g == 0 {
                continue;
            }
            // Decode the whole grid once: `decoded = Φ(grid) · B_k`  (g × p).
            let (phi_grid, _jet) = evaluator.evaluate(grid.view())?;
            if phi_grid.ncols() != self.atoms[atom_idx].basis_size() {
                return Err(format!(
                    "SaeManifoldTerm::seed_coords_by_decoder_projection: atom {atom_idx} grid Φ has {} columns but decoder expects {}",
                    phi_grid.ncols(),
                    self.atoms[atom_idx].basis_size()
                ));
            }
            let decoded = phi_grid.dot(&self.atoms[atom_idx].decoder_coefficients);
            // Per-row global argmin of ‖x_i − decoded_g‖² over the grid. Rows are
            // *visited* in the enrichment order (live rows first); the projection
            // objective for each row is unchanged, and each row is seeded exactly
            // once, so the order is pure attention and cannot move any seed.
            let mut seeded = Array2::<f64>::zeros((n, d));
            for &row in &visit_order {
                let mut best_idx = 0usize;
                let mut best_err = f64::INFINITY;
                for grid_idx in 0..g {
                    let mut err = 0.0_f64;
                    for col in 0..p {
                        let diff = target[[row, col]] - decoded[[grid_idx, col]];
                        err += diff * diff;
                    }
                    if err < best_err {
                        best_err = err;
                        best_idx = grid_idx;
                    }
                }
                for axis in 0..d {
                    seeded[[row, axis]] = grid[[best_idx, axis]];
                }
            }
            let flat = Array1::from_iter(seeded.iter().copied());
            self.assignment.coords[atom_idx].set_flat(flat.view());
            let coords = self.assignment.coords[atom_idx].as_matrix();
            self.atoms[atom_idx].refresh_basis(coords.view())?;
        }
        Ok(())
    }

    pub fn run_fixed_decoder_arrow_schur(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &mut SaeManifoldRho,
        analytic_penalties: Option<&AnalyticPenaltyRegistry>,
        max_iter: usize,
        step_size: f64,
        ridge_ext_coord: f64,
    ) -> Result<SaeManifoldLoss, String> {
        if !(step_size.is_finite() && step_size > 0.0) {
            return Err(format!(
                "SaeManifoldTerm::run_fixed_decoder_arrow_schur: step_size must be finite and positive; got {step_size}"
            ));
        }
        if max_iter < 1 {
            return Err(
                "SaeManifoldTerm::run_fixed_decoder_arrow_schur: max_iter must be positive".into(),
            );
        }
        let beta_zero = Array1::<f64>::zeros(self.beta_dim());
        let mut last_loss = self.loss(target, rho)?;
        for _ in 0..max_iter {
            self.advance_temperature_schedule()?;
            let pre_step_loss = self.loss(target, rho)?;
            let sys = self
                .assemble_arrow_schur(target, rho, analytic_penalties)
                .map_err(|err| format!("SaeManifoldTerm::run_fixed_decoder_arrow_schur: {err}"))?;
            let pre_step_total =
                self.penalized_objective_total(target, rho, analytic_penalties, 1.0)?;
            let delta_ext_coord = Self::fixed_decoder_step_from_rows(&sys, ridge_ext_coord)?;
            let directional_decrease = sae_manifold_newton_directional_decrease(
                &sys,
                delta_ext_coord.view(),
                beta_zero.view(),
            );
            let grad_norm_sq: f64 = sys
                .rows
                .iter()
                .flat_map(|row| row.gt.iter())
                .map(|&v| v * v)
                .sum();
            let step_norm_sq: f64 = delta_ext_coord.iter().map(|&v| v * v).sum();
            let directional_decrease_floor = SAE_MANIFOLD_DIRECTIONAL_DECREASE_REL_FLOOR
                * grad_norm_sq.sqrt()
                * step_norm_sq.sqrt();
            let snapshot = self.snapshot_mutable_state();
            if !(pre_step_total.is_finite()
                && directional_decrease.is_finite()
                && directional_decrease > 0.0
                && directional_decrease > directional_decrease_floor)
            {
                self.restore_mutable_state(&snapshot);
                last_loss = pre_step_loss;
                break;
            }

            let mut trial_step_size = step_size;
            let mut accepted_loss: Option<SaeManifoldLoss> = None;
            for halving in 0..=SAE_MANIFOLD_MAX_LINESEARCH_HALVINGS {
                if halving > 0 {
                    self.restore_mutable_state(&snapshot);
                }
                let trial_result = self
                    .apply_newton_step(delta_ext_coord.view(), beta_zero.view(), trial_step_size)
                    .and_then(|()| {
                        self.penalized_objective_total(target, rho, analytic_penalties, 1.0)
                    });
                if let Ok(post_step_total) = trial_result {
                    let armijo_bound = pre_step_total
                        - SAE_MANIFOLD_ARMIJO_C1 * trial_step_size * directional_decrease;
                    if post_step_total.is_finite() && post_step_total <= armijo_bound {
                        accepted_loss = Some(self.loss(target, rho)?);
                        break;
                    }
                }
                trial_step_size *= 0.5;
            }
            match accepted_loss {
                Some(loss) => last_loss = loss,
                None => {
                    self.restore_mutable_state(&snapshot);
                    last_loss = pre_step_loss;
                    break;
                }
            }
        }
        Ok(last_loss)
    }

    /// Rank-revealing adaptive basis depth for rank-deficient decoder designs
    /// (#1117 root-cause fix; supersedes the prior data-null projector deflation
    /// + post-fit range projection and the #1051 LM ridge for this case).
    ///
    /// On a near-degenerate input manifold (e.g. the OLMo L25 PCA-32 circle, or
    /// the `stage1-step0` checkpoint with post-PCA std ≈ 0.04) the fitted latent
    /// coordinate `t` plus the assignment weights fail to excite the full
    /// fixed-width periodic basis `[1, sin 2πt, cos 2πt, sin 4πt, cos 4πt]`: the
    /// bare data Gram `G_k = D_kᵀ D_k` is rank-deficient (`rank 3/5`, the
    /// `[SAE-AUDIT]` signal). WHICH `r_k`-d subspace the data DOES support is
    /// data-dependent — it is whatever combination of the fixed columns the data
    /// excites — so we discover it from the Gram rather than truncating a fixed
    /// set of columns.
    ///
    /// We symmetrise each `G_k`, eigendecompose it, and keep the eigenvectors
    /// ABOVE the relative spectral cutoff as the orthonormal data-supported
    /// column map `Q_k` (`M_k × r_k`, `r_k = rank(G_k)`). For any rank-deficient
    /// atom (`r_k < M_k`) we REPARAMETRIZE its basis onto `Q_k`
    /// ([`SaeManifoldAtom::reduce_basis_to_subspace`]): the design `Φ̃ = Φ Q_k`
    /// becomes full-rank by construction, the decoder is the rank-`r_k` oracle
    /// `B̃ = Q_kᵀ B`, the roughness Gram is `Q_kᵀ S Q_k`, and the evaluator is
    /// wrapped so the reduction survives every basis refresh. The inner solve no
    /// longer descends a flat valley and the outer REML log-det is well-posed —
    /// no step-time deflation, ridge floor, or post-fit projection needed for
    /// the deficiency. The depth decision is made ONCE here, before the outer
    /// loop, so it is held fixed across the inner Newton walk.
    ///
    /// A full-rank atom (`base`/`step_2300`, `r_k == M_k`) is SKIPPED entirely —
    /// its basis, decoder, penalty, and evaluator are left byte-for-byte the
    /// historical full-`B` path, so the well-conditioned fit is unchanged.
    pub(crate) fn reduce_atoms_to_data_supported_rank(&mut self) -> Result<(), String> {
        let p = self.output_dim();
        if p == 0 || self.beta_dim() == 0 {
            return Ok(());
        }
        let mut grams = self.empty_decoder_gram_accumulator();
        self.accumulate_decoder_gram(&mut grams);
        for atom_idx in 0..self.atoms.len() {
            let m = self.atoms[atom_idx].basis_size();
            if m == 0 || grams[atom_idx].dim() != (m, m) {
                continue;
            }
            // Symmetrise the bare data Gram `G_k` before the eigendecomposition.
            let mut data_gram = grams[atom_idx].clone();
            for i in 0..m {
                for j in 0..i {
                    let sym = 0.5 * (data_gram[[i, j]] + data_gram[[j, i]]);
                    data_gram[[i, j]] = sym;
                    data_gram[[j, i]] = sym;
                }
            }
            let (evals, evecs) = match data_gram.eigh(Side::Lower) {
                Ok(pair) => pair,
                Err(_) => continue,
            };
            let max_eig = evals.iter().fold(
                0.0_f64,
                |acc, &v| {
                    if v.is_finite() { acc.max(v) } else { acc }
                },
            );
            if !(max_eig > 0.0) {
                // An all-zero data Gram (no assignment mass) is handled by the
                // fatal pre-fit audit, not by a basis reduction here.
                continue;
            }
            let cutoff = SAE_MANIFOLD_SPECTRAL_RANK_CUTOFF * max_eig;
            // Eigenvectors whose eigenvalue clears the relative cutoff span the
            // data-supported subspace `Q_k` (the retained columns).
            let kept: Vec<usize> = (0..evals.len())
                .filter(|&idx| {
                    let lambda = evals[idx];
                    lambda.is_finite() && lambda > cutoff
                })
                .collect();
            let r = kept.len();
            // Full rank (`r == m`) → the well-conditioned path; leave it
            // byte-for-byte unchanged. `r == 0` is a degenerate all-null Gram
            // that the fatal pre-fit audit already rejects; do not reduce to a
            // zero-width basis here.
            if r == m || r == 0 {
                continue;
            }
            // Build the orthonormal column map `Q_k` (M × r) from the retained
            // eigenvectors. The reduction needs an analytic second-jet evaluator
            // to compose the reduced jets; atoms without one (caller-managed,
            // e.g. an out-of-band design) keep the historical projector-free
            // full-`B` path and rely on the LM ridge — the periodic/torus/
            // sphere/Duchon production atoms all carry a second jet.
            if self.atoms[atom_idx].basis_second_jet.is_none() {
                continue;
            }
            let mut q = Array2::<f64>::zeros((m, r));
            for (col, &eig_idx) in kept.iter().enumerate() {
                for row in 0..m {
                    q[[row, col]] = evecs[[row, eig_idx]];
                }
            }
            self.atoms[atom_idx]
                .reduce_basis_to_subspace(&q)
                .map_err(|err| {
                    format!(
                        "SaeManifoldTerm::reduce_atoms_to_data_supported_rank: atom {atom_idx}: {err}"
                    )
                })?;
        }
        Ok(())
    }

    pub fn run_joint_fit_arrow_schur(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &mut SaeManifoldRho,
        analytic_penalties: Option<&AnalyticPenaltyRegistry>,
        max_iter: usize,
        step_size: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
    ) -> Result<SaeManifoldLoss, String> {
        if !(step_size.is_finite() && step_size > 0.0) {
            return Err(format!(
                "SaeManifoldTerm::run_joint_fit_arrow_schur: step_size must be finite and positive; got {step_size}"
            ));
        }
        self.refresh_basis_from_current_coords()
            .map_err(|err| format!("SaeManifoldTerm::run_joint_fit_arrow_schur: {err}"))?;
        // #1117 root-cause fix — rank-revealing adaptive basis depth, applied
        // FIRST (before frame activation, the identifiability audit, and the
        // outer loop) so every downstream stage sees a full-rank design. A
        // fixed-width decoder basis (e.g. the periodic circle's
        // `[1, sin2πt, cos2πt, sin4πt, cos4πt]`) emits `M_k` columns whether or
        // not the fitted `t`/assignment excite them; on a near-degenerate
        // checkpoint (OLMo `stage1-step0` PCA-32: data Gram rank `3/5`) the
        // unexcited columns make the decoder design rank-deficient BY
        // CONSTRUCTION, flattening the outer REML surface so BFGS stalls. We
        // discover the data-supported subspace `Q_k = range(G_k)` ONCE here from
        // the bare data Gram and, for any rank-deficient atom, REPARAMETRIZE its
        // basis onto that subspace (`Φ̃ = Φ Q_k`, `B̃ = Q_kᵀ B`, `S̃ = Q_kᵀ S Q_k`,
        // and a `SubspaceReducedEvaluator` so the reduction survives every
        // refresh). The reduced design is full-rank, so the identifiability audit
        // passes, the frame profiles the reduced block, the inner solve needs no
        // step-time deflation, and the outer REML log-det is well-conditioned —
        // this SUPERSEDES the prior data-null projector deflation + post-fit
        // range projection for the rank-deficiency case. The depth decision is
        // made once here and held FIXED across the inner Newton walk. A full-rank
        // atom (`base`/`step_2300`, `r_k == M_k`) is left untouched, so its
        // design, decoder, and REML criterion are byte-for-byte the historical
        // full-`B` path.
        self.reduce_atoms_to_data_supported_rank()?;
        // #972 / #977 T1 — magic-by-default decoder-frame activation. Before the
        // outer loop, auto-derive and install the low-rank Grassmann frames
        // (each atom independently, only when the factorization materially
        // shrinks its border and leaves a positive Grassmann dimension). No
        // flag: small-`p` / full-rank atoms stay on the bit-for-bit full-`B`
        // path, so the small-model fits are unchanged; large-ambient-`p`,
        // low-decoder-rank atoms collapse their border `M_k·p → M_k·r_k` and the
        // joint solve runs in the factored coordinate space.
        self.ensure_decoder_frames_active_for_current_decoder()
            .map_err(|err| format!("SaeManifoldTerm::run_joint_fit_arrow_schur: {err}"))?;
        // #976 Layer-1 guard ledger is per joint fit: each inner solve gets a
        // fresh re-seed budget and reports only its own breaches.
        self.collapse_events.clear();
        // #1003 — run the active-mass guard at ENTRY (iteration 0), before the
        // pre-fit identifiability audit. A cold seed can hand the fit an atom
        // whose gates are vacuous on every row (the outer seed cascade sweeps
        // ρ states the seeding heuristics never saw); without this call the
        // audit below reports that atom as a fatal rank-0 weighted design and
        // the whole seed dies, even though the #976 guard exists precisely to
        // answer a support collapse with one observable re-seed. The guard's
        // shared per-fit budget applies unchanged: an atom re-seeded here that
        // collapses again mid-fit goes Terminal, which is the structure
        // search's signal, not the inner loop's. Genuinely degenerate bases
        // (zero rows regardless of gates) still fail the audit — correctly.
        self.enforce_active_mass_guard(0, Some(rho))?;
        // #976 decoder arm at entry: if a warm-started decoder arrives already
        // collapsed (e.g. a ρ-sweep state seeded from a prior degenerate fit),
        // reseed it before the audit. An all-zero cold seed has a zero median
        // decoder norm, so the guard returns early and the cold-start path is
        // untouched.
        self.enforce_decoder_norm_guard(target, 0, rho)?;
        // ── Pre-fit decoder identifiability audit ──────────────────────────
        //
        // Each decoder atom `k` contributes `η_i += a_ik · Φ_k(t_ik) · B_k`,
        // with `B_k ∈ ℝ^{M_k × p}`. The decoder Hessian for atom `k` is
        // `H_data = G_k ⊗ I_p` where `G_k = (diag(a_·k)·Φ_k)ᵀ (diag(a_·k)·Φ_k)`
        // (the diagonal `(atom_k, atom_k)` block of the sparse data Gram `G`
        // assembled in `assemble_arrow_schur`); the
        // `p` output channels share the identical `M_k × M_k` Gram, so decoder
        // identifiability is fully determined by the per-atom `(n, M_k)` design
        // `D_k = diag(a_·k)·Φ_k`. The `p`-fold output replication carries no
        // extra structural information and must NOT be materialised — doing so
        // (the former `(n·p, M_k·p)` channel-block route through the
        // cross-block flat audit) broadcast an `(n·p)`-row Jacobian into the
        // `n`-row placeholder design and panicked inside ndarray.
        //
        // We therefore run the rank check directly on each `D_k`. A
        // rank-deficient `D_k` means atom `k`'s decoder block Hessian is
        // singular (its Cholesky will fail or produce garbage steps), which is
        // surfaced as an immediate fatal error. Near-rank-deficient columns are
        // logged as INFO so callers can see which atoms are weakly identified.
        //
        // The check is performed chunk-aware through the per-atom Gram
        // accumulator: the full-batch path is the single-chunk special case.
        // `D_k`'s singular spectrum equals `√spec(G_k)` with
        // `G_k = D_kᵀ D_k`, so accumulating `G_k` over the whole design and
        // taking its eigenvalues reproduces the former pivoted-QR rank exactly
        // while never retaining an `(N × M_k)` design.
        {
            let mut grams = self.empty_decoder_gram_accumulator();
            self.accumulate_decoder_gram(&mut grams);
            self.finalize_decoder_identifiability_audit(&grams, self.n_obs())?;
        }
        for outer_iteration in 0..max_iter {
            self.advance_temperature_schedule()?;
            // ρ (including the ARD precisions) is owned by the outer engine
            // (`SaeManifoldOuterObjective`) and held FIXED across this inner
            // (t, β) Newton solve. The inner loop solves the joint manifold +
            // decoder system at the engine's current ρ; the engine alone
            // moves ρ by minimising the true REML criterion (see
            // `SaeManifoldTerm::reml_criterion`). The former in-loop
            // `update_ard_reml` rule (α = n / ‖t‖²) dropped the logdet /
            // effective-dof term and collapsed α on near-degenerate axes; it
            // has been removed in favour of the criterion-driven update.
            let sys = self
                .assemble_arrow_schur(target, rho, analytic_penalties)
                .map_err(|err| format!("SaeManifoldTerm::run_joint_fit_arrow_schur: {err}"))?;
            let plan = self
                .streaming_plan()
                .admitted_or_error(self.n_obs(), self.output_dim(), self.k_atoms())
                .map_err(|err| format!("SaeManifoldTerm::run_joint_fit_arrow_schur: {err}"))?;
            let solve_options = plan.solve_options_for_border_dim(sys.k);
            // Inner Newton step with principled LM-style ridge escalation. The
            // PCA-seed starting state on a small batch (e.g. `predict` on a
            // strict subset of the training set) can produce a per-row
            // `H_tt + ridge_t·I` whose Cholesky has a negative pivot, or a
            // near-singular Schur complement, at the caller's nominal ridges.
            // Rather than abort, mirror the proximal-correction outer wrapper
            // and grow both ridges geometrically until the linear system
            // factors. This is the same LM-trust-region damping the convergent
            // proximal_correction path applies; we route it through the same
            // factor-failure error variants so legitimate, non-recoverable
            // errors (PCG divergence with no factor failure, adaptive-step
            // exhaustion, …) still surface immediately.
            let (delta_ext_coord, delta_beta, _diag) =
                solve_with_lm_escalation_inner(&sys, ridge_ext_coord, ridge_beta, &solve_options)
                    .map_err(|err| format!("SaeManifoldTerm::run_joint_fit_arrow_schur: {err}"))?;
            let directional_decrease = sae_manifold_newton_directional_decrease(
                &sys,
                delta_ext_coord.view(),
                delta_beta.view(),
            );
            // Relative-scale floor on the directional decrease. When the
            // gradient is nearly orthogonal to the Newton step (ill-conditioned
            // near-convergence), `directional_decrease` collapses to O(machine
            // epsilon · ‖g‖ · ‖Δ‖). At that scale the Armijo bound
            // `pre_step_total − c1·step·directional_decrease` is numerically
            // indistinguishable from `pre_step_total`, so the line search would
            // "accept" on rounding noise. Treat that as converged and stop. The
            // norms are the natural scale of the inner product; the relative
            // constant keeps the reduction term distinguishable from rounding at
            // full step size given SAE_MANIFOLD_ARMIJO_C1 = 1e-4.
            let mut grad_norm_sq = 0.0;
            for (row_idx, row) in sys.rows.iter().enumerate() {
                let di = sys.row_dims[row_idx];
                for axis in 0..di {
                    grad_norm_sq += row.gt[axis] * row.gt[axis];
                }
            }
            for idx in 0..sys.k {
                grad_norm_sq += sys.gb[idx] * sys.gb[idx];
            }
            let mut step_norm_sq = 0.0;
            for &v in delta_ext_coord.iter() {
                step_norm_sq += v * v;
            }
            for &v in delta_beta.iter() {
                step_norm_sq += v * v;
            }
            // #1051 — gauge/null-aware stationarity. A rank-deficient (or merely
            // weakly-identified) decoder column makes the inner optimum a FLAT
            // VALLEY: the Newton step keeps crawling along the unidentified
            // direction with a tiny-but-nonzero gradient, so neither the raw
            // gradient nor the Armijo decrease ever clears tolerance and the
            // solve burns its whole budget making cosmetic progress (the 122 s
            // line fit). Project the step out of the chart-gauge orbit AND the
            // decoder β-null; when the IDENTIFIED-direction motion is below the
            // step tolerance the iterate is stationary on the quotient manifold —
            // ranking the Laplace criterion there is correct, and continuing
            // only chases gauge freedom. This mirrors the quotient convergence
            // `reml_criterion`'s undamped-evidence loop already applies, so the
            // inner solve and the criterion agree on "converged".
            if delta_ext_coord.len() == self.n_obs() * self.assignment.row_block_dim()
                && delta_beta.len() == self.beta_dim()
            {
                let quotient_step_norm_sq = self.quotient_newton_step_norm_sq(
                    delta_ext_coord.view(),
                    delta_beta.view(),
                    step_norm_sq,
                    rho.lambda_smooth(),
                )?;
                let step_tolerance = SAE_MANIFOLD_INNER_STEP_REL_TOL * self.inner_iterate_scale();
                if quotient_step_norm_sq.sqrt() <= step_tolerance {
                    break;
                }
            }
            let directional_decrease_floor = SAE_MANIFOLD_DIRECTIONAL_DECREASE_REL_FLOOR
                * grad_norm_sq.sqrt()
                * step_norm_sq.sqrt();
            // Capture the exact state whose assembled gradient/Hessian produced
            // `sys`, then evaluate the Armijo baseline from that same state.
            // Assembly installs compact active-set layout in `last_row_layout`;
            // computing the baseline after that mutation prevents comparing a
            // trial at one represented state against a bound from another.
            //
            // Each rejected trial restores from this snapshot in place; the
            // static atom metadata, smoothness penalties and basis-evaluator
            // `Arc`s are never re-cloned. This replaces the per-halving full
            // `self.clone()`, whose dominant cost was copying the
            // `O(N·M·d)` `basis_jacobian` and `O(N·M)` `basis_values` on every
            // backtrack.
            let snapshot = self.snapshot_mutable_state();
            let pre_step_total =
                self.penalized_objective_total(target, rho, analytic_penalties, 1.0)?;
            if !pre_step_total.is_finite() {
                // Pre-step state is unperturbed here; restore is a no-op but
                // keeps the invariant explicit.
                self.restore_mutable_state(&snapshot);
                break;
            }
            // A non-descent Newton direction (gᵀΔ ≤ 0 or below the rounding
            // floor) is only a STOPPING criterion when the iterate is actually
            // stationary: the floor exists for benign ill-conditioned
            // near-convergence, where `gᵀΔ` collapses to rounding noise while
            // ‖g‖ is already tiny. In degenerate multi-atom geometry (gate
            // tug-of-war, rank-deficient duchon columns) the LM solve factors
            // with a near-zero pivot, the step is dominated by that near-null
            // direction, and `gᵀΔ/(‖g‖·‖Δ‖)` collapses while ‖g‖ is HUGE —
            // breaking there silently froze the iterate and let the
            // `reml_criterion` refine loop re-measure the same point until its
            // budget died (the constant-‖g‖=1e12 signature). Gate the break on
            // genuine KKT stationarity — the SAME iterate-scaled tolerance
            // `reml_criterion` uses — and otherwise fall through to the
            // proximal-correction ridge escalation below: heavier LM damping
            // bends the step toward steepest descent, which is always a
            // descent direction for a consistent gradient.
            let descent_direction_ok = directional_decrease.is_finite()
                && directional_decrease > 0.0
                && directional_decrease > directional_decrease_floor;
            if !descent_direction_ok {
                let grad_tolerance = SAE_MANIFOLD_INNER_GRAD_REL_TOL * self.inner_iterate_scale();
                if grad_norm_sq.sqrt() <= grad_tolerance {
                    self.restore_mutable_state(&snapshot);
                    break;
                }
            }

            let mut trial_step_size = step_size;
            let mut accepted = false;
            for halving in 0..=SAE_MANIFOLD_MAX_LINESEARCH_HALVINGS {
                if !descent_direction_ok {
                    // No Armijo bound is meaningful along a non-descent
                    // direction; route straight to the proximal correction.
                    break;
                }
                if halving > 0 {
                    // Reset to the pre-step state before re-applying at the
                    // halved step. The first trial starts from the pre-step
                    // state already, so the restore is only needed after a
                    // rejected trial mutated `self`.
                    self.restore_mutable_state(&snapshot);
                }
                let trial_result = self
                    .apply_newton_step(delta_ext_coord.view(), delta_beta.view(), trial_step_size)
                    .and_then(|()| {
                        self.penalized_objective_total(target, rho, analytic_penalties, 1.0)
                    });
                if let Ok(post_step_total) = trial_result {
                    let armijo_bound = pre_step_total
                        - SAE_MANIFOLD_ARMIJO_C1 * trial_step_size * directional_decrease;
                    if post_step_total.is_finite() && post_step_total <= armijo_bound {
                        accepted = true;
                        break;
                    }
                }
                trial_step_size *= 0.5;
            }
            if !accepted {
                self.restore_mutable_state(&snapshot);
                let correction = ArrowProximalCorrectionOptions {
                    initial_ridge: ridge_ext_coord
                        .max(ridge_beta)
                        .max(SAE_MANIFOLD_ROW_RIDGE_FLOOR),
                    armijo_c1: SAE_MANIFOLD_ARMIJO_C1,
                    ..ArrowProximalCorrectionOptions::default()
                };
                let accepted_step = match solve_arrow_newton_step_with_proximal_correction(
                    &sys,
                    ridge_ext_coord,
                    ridge_beta,
                    pre_step_total,
                    // `sys.k` is the actual border width — factored
                    // (`factored_border_dim`) when frames are active, else
                    // `beta_dim` — so the direct/PCG mode threshold keys on the
                    // dimension the solve actually runs at.
                    &solve_options,
                    &correction,
                    |trial_delta_t, trial_delta_beta| {
                        self.restore_mutable_state(&snapshot);
                        self.apply_newton_step(trial_delta_t, trial_delta_beta, 1.0)
                            .and_then(|()| {
                                self.penalized_objective_total(target, rho, analytic_penalties, 1.0)
                            })
                            .unwrap_or(f64::INFINITY)
                    },
                ) {
                    Ok(step) => step,
                    Err(err) => {
                        log::debug!(
                            "run_joint_fit_arrow_schur: proximal correction errored at \
                             iteration {outer_iteration} (gᵀΔ={directional_decrease:.3e}, \
                             floor={directional_decrease_floor:.3e}, \
                             ‖g‖={:.3e}): {err}",
                            grad_norm_sq.sqrt()
                        );
                        self.restore_mutable_state(&snapshot);
                        break;
                    }
                };
                if !(accepted_step.trial_objective_value.is_finite()
                    && accepted_step.trial_objective_value < pre_step_total)
                {
                    log::debug!(
                        "run_joint_fit_arrow_schur: proximal correction made no decrease at \
                         iteration {outer_iteration} (trial={:.9e}, pre={pre_step_total:.9e}, \
                         ‖g‖={:.3e})",
                        accepted_step.trial_objective_value,
                        grad_norm_sq.sqrt()
                    );
                    self.restore_mutable_state(&snapshot);
                    break;
                }
            }
            // Affine gauge canonicalization is a representation change, but the
            // decoder smoothness term is part of the optimized objective. Keep the
            // canonicalized state only when the same scalar used by the line search
            // does not increase; otherwise REML would inspect an off-contract
            // post-accept state whose gradient was never accepted.
            let accepted_snapshot = self.snapshot_mutable_state();
            let accepted_total =
                self.penalized_objective_total(target, rho, analytic_penalties, 1.0)?;
            self.canonicalize_affine_gauge_after_accept(Some(rho))?;
            let canonical_total =
                self.penalized_objective_total(target, rho, analytic_penalties, 1.0)?;
            if !(canonical_total.is_finite() && canonical_total <= accepted_total) {
                self.restore_mutable_state(&accepted_snapshot);
            }
            // #976 Layer-1 guard 3: after an accepted step (Armijo or proximal
            // — the rejection paths `break` above), check every atom's support
            // and answer breaches with a bounded re-seed or a terminal
            // CollapseEvent. Runs post-acceptance so it never perturbs a
            // line-search trial, and any re-seed is simply the next
            // iteration's starting state.
            self.enforce_active_mass_guard(outer_iteration, Some(rho))?;
            // #976 Layer-1 guard 3b (decoder arm): the gate-mass guard above is
            // blind to a dictionary whose gates stay spread but whose decoders
            // have all collapsed to ≈0 (the real-data K>1 failure that drives
            // EV→0 and the `0 → K·n` evidence-deflation abort). Catch a decoder
            // that has fallen far behind its peers and reseed it onto the
            // residual; a strict no-op for K=1.
            self.enforce_decoder_norm_guard(target, outer_iteration, rho)?;
            // #972 / #977 T1 — U-block of the alternating block-coordinate ascent.
            // After the decoder `B` has been updated by the accepted (t, ΔC) step
            // (lifted through the OLD frames in `apply_newton_step`), re-polar each
            // ACTIVE atom's frame from the refreshed data evidence and re-project
            // the decoder onto it, so the next assembly's C-block solve runs in an
            // up-to-date frame. The refresh is a closed-form `O(p r²)` thin SVD per
            // atom run OUTSIDE the border; the C-coordinates are held fixed during
            // it (the block-coordinate split). Skipped entirely when no frame is
            // active (the full-`B` path never touches this). One refresh per
            // accepted outer iteration is a sensible cadence (the issue's
            // streaming-polar fixed point).
            if self.frames_active() {
                self.refresh_active_frames_from_data(target, rho)
                    .map_err(|err| format!("SaeManifoldTerm::run_joint_fit_arrow_schur: {err}"))?;
            }
        }
        // #1117 — the rank-`r_k` oracle is already pinned: each rank-deficient
        // atom was reparametrized onto its data-supported subspace at fit entry
        // (`reduce_atoms_to_data_supported_rank`), so its decoder lives in the
        // reduced `r_k`-wide coordinate by construction and carries no data-null
        // component to project away. No post-loop projection is needed.
        // ρ is owned by the outer engine and unchanged here; just return the
        // converged inner loss at the fixed ρ.
        self.loss(target, rho)
    }

    /// Allocate one zero `(M_k × M_k)` Gram accumulator per atom for the
    /// chunk-aware decoder identifiability audit.
    pub(crate) fn empty_decoder_gram_accumulator(&self) -> Vec<Array2<f64>> {
        self.atoms
            .iter()
            .map(|atom| {
                let m = atom.basis_size();
                Array2::<f64>::zeros((m, m))
            })
            .collect()
    }

    /// Accumulate this term's (chunk's) contribution to the per-atom weighted
    /// design Gram `G_k += D_kᵀ D_k`, with `D_k = diag(a_·k)·Φ_k`.
    ///
    /// `grams[k]` must be `(M_k × M_k)`. Streaming callers invoke this once per
    /// chunk against the freshly materialized chunk term; the full-batch path
    /// invokes it once against `self`. The Gram is symmetric and channel-free
    /// (the `p`-fold output replication is carried by the `⊗ I_p` Kronecker
    /// structure, so it adds no rank information), so accumulating `Φ` weighted
    /// by the per-row assignment exactly reproduces the data-fit decoder block
    /// curvature `G_k` that `assemble_arrow_schur` installs.
    pub(crate) fn accumulate_decoder_gram(&self, grams: &mut [Array2<f64>]) {
        let n = self.n_obs();
        let assignments = self.assignment.assignments();
        // Each atom's Gram `G_k = Φ_kᵀ diag(a_k²) Φ_k` is an independent
        // weighted cross-product over the N rows — the canonical `xt_diag_x`
        // shape, and independent across the per-atom axis. This feeds a
        // tolerance-based identifiability RANK decision (not a fitted quantity),
        // so the device path's accumulation order is admissible.
        //
        // Spread the atoms across EVERY device via `gpu::pool::scatter_batched`;
        // each device tile computes its atoms' Grams through the size-gated
        // `try_fast_xt_diag_x` shim. Atoms whose device path declines (no
        // runtime, sub-threshold size, or backend miss) drop to the exact CPU
        // rank-1 accumulation, so the result matches the all-CPU path.
        let weights: Vec<Array1<f64>> = (0..self.atoms.len())
            .map(|atom_idx| {
                let col = assignments.column(atom_idx);
                col.mapv(|a| a * a)
            })
            .collect();

        // CPU per-atom contribution, used for fallback and as the whole path
        // when no GPU runtime is present.
        let cpu_one = |atom_idx: usize, gram: &mut Array2<f64>| {
            let atom = &self.atoms[atom_idx];
            let m = atom.basis_size();
            let assign_col = assignments.column(atom_idx);
            let mut weighted = vec![0.0_f64; m];
            for row in 0..n {
                let a_k = assign_col[row];
                if a_k == 0.0 {
                    continue;
                }
                for col in 0..m {
                    weighted[col] = a_k * atom.basis_values[[row, col]];
                }
                for i in 0..m {
                    let wi = weighted[i];
                    if wi == 0.0 {
                        continue;
                    }
                    for j in 0..m {
                        gram[[i, j]] += wi * weighted[j];
                    }
                }
            }
        };

        let rt = crate::gpu::runtime::GpuRuntime::global();
        match rt {
            None => {
                for atom_idx in 0..self.atoms.len() {
                    if self.atoms[atom_idx].basis_size() == 0 {
                        continue;
                    }
                    cpu_one(atom_idx, &mut grams[atom_idx]);
                }
            }
            Some(rt) => {
                // Device tiles produce each owned atom's Gram into a side channel
                // keyed by atom index; splice them back into `grams` (with `+=`
                // accumulation) after the scatter. Atoms the device declines are
                // marked so the CPU fallback runs for exactly those.
                let mut items: Vec<usize> = (0..self.atoms.len())
                    .filter(|&i| self.atoms[i].basis_size() > 0)
                    .collect();
                let device_grams: std::sync::Mutex<Vec<(usize, Array2<f64>)>> =
                    std::sync::Mutex::new(Vec::with_capacity(items.len()));
                let declined: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());
                let atoms_ref = &self.atoms;
                let weights_ref = &weights;
                let ok = crate::gpu::pool::scatter_batched(rt, &mut items, |_ordinal, slice| {
                    for &atom_idx in slice.iter() {
                        let phi = atoms_ref[atom_idx].basis_values.view();
                        let w = weights_ref[atom_idx].view();
                        match crate::gpu::linalg::try_fast_xt_diag_x(phi, w) {
                            Some(g) => device_grams
                                .lock()
                                .expect("device_grams mutex poisoned")
                                .push((atom_idx, g)),
                            None => declined
                                .lock()
                                .expect("declined mutex poisoned")
                                .push(atom_idx),
                        }
                    }
                    Some(())
                });
                match ok {
                    Some(()) => {
                        for (atom_idx, g) in device_grams
                            .into_inner()
                            .expect("device_grams mutex poisoned")
                        {
                            grams[atom_idx] += &g;
                        }
                        for atom_idx in declined.into_inner().expect("declined mutex poisoned") {
                            cpu_one(atom_idx, &mut grams[atom_idx]);
                        }
                    }
                    None => {
                        for atom_idx in 0..self.atoms.len() {
                            if self.atoms[atom_idx].basis_size() == 0 {
                                continue;
                            }
                            cpu_one(atom_idx, &mut grams[atom_idx]);
                        }
                    }
                }
            }
        }
    }

    /// Decide rank-deficiency of each accumulated decoder Gram and surface the
    /// same fatal / INFO contract as the former pivoted-QR audit.
    pub(crate) fn finalize_decoder_identifiability_audit(
        &self,
        grams: &[Array2<f64>],
        n_total: usize,
    ) -> Result<(), String> {
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let m = atom.basis_size();
            if m == 0 {
                continue;
            }
            let rank =
                crate::solver::identifiability_audit::rank_of_gram(&grams[atom_idx], n_total)
                    .map_err(|e| {
                        format!(
                            "SaeManifoldTerm: pre-fit decoder audit (atom '{}'): \
                         Gram eigendecomposition failed: {e}",
                            atom.name,
                        )
                    })?;
            if rank < m {
                let dropped = m - rank;
                if rank == 0 {
                    return Err(format!(
                        "SaeManifoldTerm: pre-fit identifiability audit: decoder atom '{}' has \
                         rank-0 weighted design (n={n_total}, M_k={m}); all assignment weights \
                         vanish or the basis is degenerate, so the Arrow-Schur Newton system for \
                         this block is singular",
                        atom.name,
                    ));
                }
                log::info!(
                    "[SAE-AUDIT] decoder atom '{}' weighted design is rank-deficient \
                     (rank={rank}/{m}, {dropped} weakly-identified column(s), n={n_total}); the \
                     Arrow-Schur ridge will regularise the deficient directions",
                    atom.name,
                );
            }
        }
        Ok(())
    }

    /// Materialize a row-chunk `[start, end)` of this term as a standalone
    /// `SaeManifoldTerm`, recomputing `(basis_values, basis_jacobian)` on demand
    /// from the persisted decoder + atom geometry and the caller-supplied
    /// per-chunk `(logits, coords)`.
    ///
    /// The streaming joint fit NEVER persists the `(N × M)` basis or `(N × K)`
    /// logit buffers. Instead, for each chunk the caller re-seeds the chunk's
    /// latent state (the SAE PCA seed restricted to the chunk's `Z` rows for the
    /// coordinates, and the chunk's gating logits) and this constructor rebuilds
    /// a small `n_chunk`-row term whose atoms share the global decoder
    /// coefficients (`B_k`), smoothness penalties, and basis evaluators with
    /// `self`. Each atom's basis is re-evaluated at the chunk coordinates via
    /// its `basis_evaluator`, so the chunk term is exactly the restriction of
    /// the global model to those rows.
    ///
    /// Errors if any atom lacks a basis evaluator (a streaming fit must be able
    /// to re-evaluate `Φ(t)` at the per-chunk coordinates) or if the supplied
    /// shapes disagree with the term's atom layout.
    pub fn materialize_chunk(
        &self,
        chunk_logits: Array2<f64>,
        chunk_coords: Vec<Array2<f64>>,
    ) -> Result<SaeManifoldTerm, String> {
        let k_atoms = self.k_atoms();
        if chunk_logits.ncols() != k_atoms {
            return Err(format!(
                "SaeManifoldTerm::materialize_chunk: chunk_logits has {} cols but K={k_atoms}",
                chunk_logits.ncols()
            ));
        }
        if chunk_coords.len() != k_atoms {
            return Err(format!(
                "SaeManifoldTerm::materialize_chunk: chunk_coords has {} atoms but K={k_atoms}",
                chunk_coords.len()
            ));
        }
        let n_chunk = chunk_logits.nrows();
        let mut atoms = Vec::with_capacity(k_atoms);
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let coords = &chunk_coords[atom_idx];
            if coords.nrows() != n_chunk || coords.ncols() != atom.latent_dim {
                return Err(format!(
                    "SaeManifoldTerm::materialize_chunk: atom {atom_idx} coords shape {:?} != ({n_chunk}, {})",
                    coords.dim(),
                    atom.latent_dim
                ));
            }
            let evaluator = atom.basis_evaluator.as_ref().ok_or_else(|| {
                format!(
                    "SaeManifoldTerm::materialize_chunk: atom '{}' has no basis evaluator; a \
                     streaming fit must re-evaluate Φ(t) at each chunk's coordinates",
                    atom.name
                )
            })?;
            let (phi, jet) = evaluator.evaluate(coords.view())?;
            let m = atom.basis_size();
            if phi.dim() != (n_chunk, m) {
                return Err(format!(
                    "SaeManifoldTerm::materialize_chunk: atom '{}' evaluator returned Φ {:?}, expected ({n_chunk}, {m})",
                    atom.name,
                    phi.dim()
                ));
            }
            if jet.dim() != (n_chunk, m, atom.latent_dim) {
                return Err(format!(
                    "SaeManifoldTerm::materialize_chunk: atom '{}' evaluator returned jet {:?}, expected ({n_chunk}, {m}, {})",
                    atom.name,
                    jet.dim(),
                    atom.latent_dim
                ));
            }
            // Seed the chunk atom from the *raw* roughness Gram (not the
            // already arc-length-reweighted `smooth_penalty`), so its
            // constructor recovers the true operator order and its own
            // `refresh_intrinsic_smooth_penalty` reweights from the canonical
            // penalty on the chunk's coordinates rather than double-applying
            // the metric (issue #673).
            let mut chunk_atom = SaeManifoldAtom::new(
                atom.name.clone(),
                atom.basis_kind.clone(),
                atom.latent_dim,
                phi,
                jet,
                atom.decoder_coefficients.clone(),
                atom.smooth_penalty_raw.clone(),
            )?;
            chunk_atom.basis_evaluator = atom.basis_evaluator.clone();
            chunk_atom.basis_second_jet = atom.basis_second_jet.clone();
            // #972 / #977 T1: carry the active Grassmann frame onto the chunk
            // atom so the streaming per-chunk assembly uses the SAME factored
            // border layout as the dense path. Without this the chunk would
            // default to the full-`B` path and the streaming REML log-det would
            // be taken over a different (larger) β block than the dense one,
            // breaking the streaming↔dense log-det agreement (#847). The
            // decoder is unchanged, so the frame stays consistent with `B_k`.
            chunk_atom.decoder_frame = atom.decoder_frame.clone();
            atoms.push(chunk_atom);
        }
        // Rebuild the assignment from the chunk's logits + coords, preserving
        // each atom's latent manifold and the global assignment mode.
        let coord_values: Vec<LatentCoordValues> = chunk_coords
            .iter()
            .zip(self.assignment.coords.iter())
            .map(|(c, src)| {
                LatentCoordValues::from_matrix_with_manifold(
                    c.view(),
                    LatentIdMode::None,
                    src.manifold().clone(),
                )
            })
            .collect();
        let assignment =
            SaeAssignment::with_mode(chunk_logits, coord_values, self.assignment.mode)?;
        let mut term = SaeManifoldTerm::new(atoms, assignment)?;
        // The temperature schedule is global outer state; the chunk term is
        // assembled at the schedule's current temperature, which the caller
        // already baked into `self.assignment.mode` before materializing.
        term.temperature_schedule = self.temperature_schedule.clone();
        Ok(term)
    }

    /// Streaming / minibatch joint fit: refine the shared decoder coefficients
    /// `B_k` (and the ARD ρ axes) by sweeping the rows in chunks of
    /// `chunk_size`, accumulating the reduced Schur system over the shared β
    /// online, and NEVER materializing the `(N × M)` / `(N × K)` per-row
    /// buffers.
    ///
    /// For each outer iteration:
    ///
    /// 1. Each chunk `[start, end)` re-seeds its per-row latent state from the
    ///    chunk's `Z` slice (`chunk_init` supplies `(logits, coords)` — the SAE
    ///    PCA seed restricted to the chunk), materializes a small chunk term via
    ///    [`Self::materialize_chunk`], and assembles its Arrow-Schur system with
    ///    the β-tier penalties scaled by the chunk fraction `n_chunk / N` (so
    ///    they sum to exactly one global copy across the pass).
    /// 2. The chunk's reduced contribution `H_βt(H_tt)⁻¹H_tβ` and `H_βt(H_tt)⁻¹g_t`
    ///    are accumulated into a single global [`StreamingArrowSchur`] over β,
    ///    consuming each chunk's Kronecker `htbeta_matvec` procedurally.
    /// 3. After one full pass, the global reduced system is solved for `Δβ` with
    ///    the same LM ridge escalation as the full-batch driver, and a streaming
    ///    Armijo line search on `Δβ` accepts the step against the summed
    ///    per-chunk loss.
    /// 4. ARD ρ is refreshed online from the accumulated `Σ‖t‖²` and row count.
    ///
    /// Only the global decoder coefficients persist across chunks and outer
    /// iterations; the per-row `(logits, coords)` are re-seeded each pass and
    /// discarded. `self`'s own per-row buffers are left untouched — the fitted
    /// decoder is written back into `self`'s atoms.
    ///
    /// This is the out-of-core counterpart of [`Self::run_joint_fit_arrow_schur`]:
    /// the in-core driver holds the full `(N × M)` target and per-row state in
    /// memory, while this driver bounds peak memory to a single chunk by
    /// re-seeding `(logits, coords, Z)` through `chunk_init` on demand — the
    /// fit-side analogue of [`Self::streaming_exact_arrow_log_det`]'s chunked
    /// evidence assembly. Wired through [`Self::fit_streaming_in_memory`] for the
    /// in-memory case; a disk-backed `chunk_init` drives the LLM-scale fit.
    pub fn run_joint_fit_arrow_schur_streaming<F>(
        &mut self,
        n_total: usize,
        chunk_size: usize,
        rho: &mut SaeManifoldRho,
        analytic_penalties: Option<&AnalyticPenaltyRegistry>,
        max_iter: usize,
        step_size: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
        mut chunk_init: F,
    ) -> Result<SaeManifoldLoss, String>
    where
        F: FnMut(usize, usize) -> Result<(Array2<f64>, Vec<Array2<f64>>, Array2<f64>), String>,
    {
        if !(step_size.is_finite() && step_size > 0.0) {
            return Err(format!(
                "SaeManifoldTerm::run_joint_fit_arrow_schur_streaming: step_size must be finite and positive; got {step_size}"
            ));
        }
        if chunk_size == 0 {
            return Err(
                "SaeManifoldTerm::run_joint_fit_arrow_schur_streaming: chunk_size must be positive"
                    .to_string(),
            );
        }
        if n_total == 0 {
            return Err(
                "SaeManifoldTerm::run_joint_fit_arrow_schur_streaming: n_total must be positive"
                    .to_string(),
            );
        }
        // #972 / #977 T1: magic-by-default frame activation, mirroring the dense
        // driver, so the streaming fit runs in the same factored coordinate
        // space (the chunk terms inherit the frames via `materialize_chunk`).
        self.ensure_decoder_frames_active_for_current_decoder()
            .map_err(|err| {
                format!("SaeManifoldTerm::run_joint_fit_arrow_schur_streaming: {err}")
            })?;
        // The β-tier width the reduced-Schur accumulators are sized at: the
        // FACTORED border `Σ M_k·r_k` when frames are active (every chunk's
        // `sys.gb` / reduced Schur is in that space), else the full-`B`
        // `beta_dim`. The accepted `delta_beta` is a factored ΔC in the former
        // case and is lifted through the frames before being applied.
        let frames_engaged = self.frames_active();
        let border_dim = if frames_engaged {
            self.factored_border_dim()
        } else {
            self.beta_dim()
        };

        // ── Chunk-aware pre-fit decoder identifiability audit ───────────────
        {
            let mut grams = self.empty_decoder_gram_accumulator();
            let mut start = 0usize;
            while start < n_total {
                let end = (start + chunk_size).min(n_total);
                let (logits, coords, _z_chunk) = chunk_init(start, end)?;
                let chunk = self.materialize_chunk(logits, coords)?;
                chunk.accumulate_decoder_gram(&mut grams);
                start = end;
            }
            self.finalize_decoder_identifiability_audit(&grams, n_total)?;
        }

        let mut last_loss = SaeManifoldLoss {
            data_fit: 0.0,
            assignment_sparsity: 0.0,
            smoothness: 0.0,
            ard: 0.0,
            evidence_gauge_deflated_directions: 0,
        };
        for _ in 0..max_iter {
            self.advance_temperature_schedule()?;
            // ── Pass 1: accumulate the global reduced Schur over β online. ──
            let options = ArrowSolveOptions::automatic(border_dim);
            let mut s_acc = Array2::<f64>::zeros((border_dim, border_dim));
            let mut rhs_acc = Array1::<f64>::zeros(border_dim);
            let mut gb_acc = Array1::<f64>::zeros(border_dim);
            // ρ (including the ARD precisions) is owned by the outer engine and
            // held FIXED across this streaming inner solve; the former online
            // `Σ t²` ARD accumulator + `update_ard_reml_from_sumsq` rule has
            // been removed in favour of the criterion-driven update.
            let mut pre_step_total = 0.0_f64;
            // Retain only the per-chunk row ranges so the line search can
            // re-materialize each chunk by re-invoking `chunk_init` at trial β
            // values. The chunk's `(logits, coords, Z)` are re-provided by the
            // seeder each time — never retained — so the pass stays O(Σ M_k²)
            // in memory rather than O(N · M) / O(N · K).
            let mut chunk_ranges: Vec<(usize, usize)> = Vec::new();
            let mut start = 0usize;
            while start < n_total {
                let end = (start + chunk_size).min(n_total);
                let n_chunk = end - start;
                let penalty_scale = n_chunk as f64 / n_total as f64;
                let (logits, coords, z_chunk) = chunk_init(start, end)?;
                if z_chunk.dim() != (n_chunk, self.output_dim()) {
                    return Err(format!(
                        "SaeManifoldTerm::run_joint_fit_arrow_schur_streaming: chunk [{start}, {end}) \
                         Z slice shape {:?} != ({n_chunk}, {})",
                        z_chunk.dim(),
                        self.output_dim()
                    ));
                }
                let mut chunk = self.materialize_chunk(logits, coords)?;
                // #991: inherit the design honesty weight slice (see
                // streaming_exact_arrow_log_det for the no-renormalize rule).
                if let Some(w) = self.row_loss_weights.as_deref() {
                    chunk.row_loss_weights = Some(w[start..end].to_vec());
                }
                chunk_ranges.push((start, end));
                pre_step_total += chunk.penalized_objective_total(
                    z_chunk.view(),
                    rho,
                    analytic_penalties,
                    penalty_scale,
                )?;
                let sys = chunk
                    .assemble_arrow_schur_scaled(
                        z_chunk.view(),
                        rho,
                        analytic_penalties,
                        penalty_scale,
                    )
                    .map_err(|err| {
                        format!("SaeManifoldTerm::run_joint_fit_arrow_schur_streaming: {err}")
                    })?;
                // Accumulate the chunk's data-fit β gradient (its g_β already
                // carries the minibatch-scaled β-penalty gradient). `sys.gb` is
                // in the factored C-space when frames are engaged (the chunk
                // inherits them), matching `gb_acc`'s `border_dim` width.
                for j in 0..border_dim {
                    gb_acc[j] += sys.gb[j];
                }
                Self::accumulate_chunk_reduced_schur(
                    &sys,
                    ridge_ext_coord,
                    &options,
                    &mut s_acc,
                    &mut rhs_acc,
                )?;
                start = end;
            }
            // The summed chunk β-blocks already reconstruct the full
            // `H_ββ` (data-fit GN `G ⊗ I_p` + smoothness + analytic β); add the
            // global β ridge exactly once, and form the reduced RHS. After this
            // step `rhs_acc = Σ_i H_βt^(i)(H_tt^(i))⁻¹g_t^(i) − g_β` is the
            // negated Schur-reduced β gradient `−g_reduced`, so the reduced
            // system `S Δβ = rhs_acc` yields the marginal Newton step in β with
            // the per-row latent eliminated.
            for j in 0..border_dim {
                s_acc[[j, j]] += ridge_beta;
                rhs_acc[j] -= gb_acc[j];
            }
            // ── Solve the global reduced β system with LM ridge escalation. ──
            let delta_beta =
                solve_streaming_reduced_beta(&s_acc, &rhs_acc, &options).map_err(|err| {
                    format!("SaeManifoldTerm::run_joint_fit_arrow_schur_streaming: {err}")
                })?;
            // ── Streaming Armijo line search on Δβ. ──
            // The directional decrease uses the *reduced* β gradient
            // `g_reduced = −rhs_acc`, the true gradient of the β-marginal
            // objective along which the line search backtracks (the per-row
            // latent block is profiled out, not stepped, in streaming).
            let beta0 = self.flatten_beta();
            let mut directional_decrease = 0.0_f64;
            for j in 0..border_dim {
                // dd = −(g_reduced · Δβ) = −((−rhs_acc) · Δβ) = rhs_acc · Δβ.
                directional_decrease += rhs_acc[j] * delta_beta[j];
            }
            if !(pre_step_total.is_finite()
                && directional_decrease.is_finite()
                && directional_decrease > 0.0)
            {
                // No descent direction available; ρ is engine-owned and fixed,
                // so just record the loss and stop.
                last_loss = self.streaming_loss(&chunk_ranges, rho, n_total, &mut chunk_init)?;
                break;
            }
            // #972 / #977 T1: when frames are engaged, `delta_beta` is a factored
            // ΔC; pre-lift it ONCE to a full-`B` ΔB (`ΔB_k = ΔC_k U_kᵀ`) so the
            // per-trial β update is a plain `beta0 + step·ΔB` (the decoder lives
            // in the full p-space). Un-framed atoms lift by identity. On the
            // full-`B` path `delta_b` is just `delta_beta`.
            let delta_b: Array1<f64> = if frames_engaged {
                FrameProjection::new(self).lift_border_vec(delta_beta.view())
            } else {
                delta_beta.clone()
            };
            let mut trial_step = step_size;
            let mut accepted_loss: Option<SaeManifoldLoss> = None;
            for _ in 0..=SAE_MANIFOLD_MAX_LINESEARCH_HALVINGS {
                let mut trial_beta = beta0.clone();
                for j in 0..self.beta_dim() {
                    trial_beta[j] += trial_step * delta_b[j];
                }
                self.set_flat_beta(trial_beta.view())?;
                let (trial_loss, trial_total) = self.streaming_loss_and_penalized_objective_total(
                    &chunk_ranges,
                    rho,
                    analytic_penalties,
                    n_total,
                    &mut chunk_init,
                )?;
                let armijo_bound =
                    pre_step_total - SAE_MANIFOLD_ARMIJO_C1 * trial_step * directional_decrease;
                if trial_total.is_finite() && trial_total <= armijo_bound {
                    accepted_loss = Some(trial_loss);
                    break;
                }
                trial_step *= 0.5;
            }
            match accepted_loss {
                Some(loss) => {
                    last_loss = loss;
                }
                None => {
                    // Restore the pre-step β before stopping. ρ is engine-owned
                    // and held fixed across the streaming inner solve.
                    self.set_flat_beta(beta0.view())?;
                    last_loss =
                        self.streaming_loss(&chunk_ranges, rho, n_total, &mut chunk_init)?;
                    break;
                }
            }
        }
        Ok(last_loss)
    }

    /// In-memory driver for [`Self::run_joint_fit_arrow_schur_streaming`]: build
    /// the `chunk_init` seeder by slicing the resident `target`, `self.assignment`
    /// logits and `self.assignment` coords per row-range — the identical chunking
    /// [`Self::streaming_exact_arrow_log_det`] already uses for the evidence pass.
    ///
    /// This is the streaming fit's wiring for data that is already resident: it
    /// bounds the Newton solve's peak memory to one chunk (no `(N × M)` /
    /// `(N × K)` materialization) while reading from the in-core buffers. The
    /// out-of-core LLM-scale path swaps this seeder for a disk-backed loader and
    /// calls `run_joint_fit_arrow_schur_streaming` directly. `chunk_size` is the
    /// auto-derived [`Self::streaming_plan`] chunk (clamped to `n`).
    pub fn fit_streaming_in_memory(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &mut SaeManifoldRho,
        analytic_penalties: Option<&AnalyticPenaltyRegistry>,
        max_iter: usize,
        step_size: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
    ) -> Result<SaeManifoldLoss, String> {
        let n_total = self.n_obs();
        if target.dim() != (n_total, self.output_dim()) {
            return Err(format!(
                "SaeManifoldTerm::fit_streaming_in_memory: target must be ({}, {}); got {:?}",
                n_total,
                self.output_dim(),
                target.dim()
            ));
        }
        let chunk_size = self.streaming_plan().chunk_size.min(n_total.max(1));
        // Snapshot the resident seed state so the per-pass re-seed is a pure
        // read (the streaming driver re-invokes the closure every line-search
        // trial and must hand back identical seeds each time).
        let seed_logits = self.assignment.logits.clone();
        let seed_coords: Vec<Array2<f64>> = self
            .assignment
            .coords
            .iter()
            .map(|coord| coord.as_matrix().to_owned())
            .collect();
        // The `target` view is sliced per chunk (not cloned wholesale); the
        // driver re-invokes this seeder every line-search trial, and each call
        // returns owned per-chunk copies. Shape is validated inside
        // `run_joint_fit_arrow_schur_streaming` against `(n_chunk, output_dim)`.
        let chunk_init = move |start: usize, end: usize| {
            let logits = seed_logits.slice(s![start..end, ..]).to_owned();
            let coords: Vec<Array2<f64>> = seed_coords
                .iter()
                .map(|coord| coord.slice(s![start..end, ..]).to_owned())
                .collect();
            let z_chunk = target.slice(s![start..end, ..]).to_owned();
            Ok((logits, coords, z_chunk))
        };
        self.run_joint_fit_arrow_schur_streaming(
            n_total,
            chunk_size,
            rho,
            analytic_penalties,
            max_iter,
            step_size,
            ridge_ext_coord,
            ridge_beta,
            chunk_init,
        )
    }

    /// Accumulate one chunk system's reduced-Schur contribution into the shared
    /// `(β × β)` accumulator and reduced RHS, consuming the chunk's Kronecker
    /// `htbeta_matvec` procedurally via [`StreamingArrowSchur`].
    ///
    /// The chunk system's β-block already carries the chunk's data-fit
    /// Gauss-Newton curvature `G_chunk ⊗ I_p` (a genuine per-row sum) plus its
    /// minibatch-scaled smoothness / analytic-β penalty. So the contribution
    /// `s_acc_chunk = hbb_chunk − Σ_i H_βt^(i)(H_tt^(i))⁻¹H_tβ^(i)` and
    /// `rhs_acc_chunk = +Σ_i H_βt^(i)(H_tt^(i))⁻¹g_t^(i)` sum across a full pass
    /// to `H_ββ − Σ_all_i (…)` and `Σ_all_i (…)` respectively, with the global
    /// β ridge added exactly once by the caller. No per-chunk ridge is applied.
    pub(crate) fn accumulate_chunk_reduced_schur(
        sys: &ArrowSchurSystem,
        ridge_ext_coord: f64,
        options: &ArrowSolveOptions,
        s_acc: &mut Array2<f64>,
        rhs_acc: &mut Array1<f64>,
    ) -> Result<(), String> {
        let k = sys.k;
        let chunk_n = sys.rows.len();
        let mut streaming = StreamingArrowSchur::from_system(sys, chunk_n.max(1));
        // `reset_accumulator(0.0)` seeds `s_acc` with the chunk's dense β-block
        // (`hbb_chunk`, including the data-fit GN block and the minibatch-scaled
        // penalty) and no ridge; `accumulate_chunk` then subtracts the per-row
        // reduction. The global β ridge is applied once by the streaming driver.
        streaming
            .reset_accumulator(0.0)
            .map_err(|e| e.to_string())?;
        streaming
            .accumulate_chunk(0, chunk_n, ridge_ext_coord, options.mode)
            .map_err(|e| e.to_string())?;
        let (contrib_s, contrib_rhs) = streaming.take_accumulators();
        for i in 0..k {
            rhs_acc[i] += contrib_rhs[i];
            for j in 0..k {
                s_acc[[i, j]] += contrib_s[[i, j]];
            }
        }
        Ok(())
    }

    /// Streaming total loss: sum of the minibatch-scaled per-chunk losses at the
    /// current β, re-materializing each chunk from a fresh re-seed via
    /// `chunk_init`. The β-penalty terms are scaled by the chunk fraction so the
    /// global smoothness penalty is counted once across the pass.
    pub(crate) fn streaming_loss<F>(
        &self,
        chunk_ranges: &[(usize, usize)],
        rho: &SaeManifoldRho,
        n_total: usize,
        chunk_init: &mut F,
    ) -> Result<SaeManifoldLoss, String>
    where
        F: FnMut(usize, usize) -> Result<(Array2<f64>, Vec<Array2<f64>>, Array2<f64>), String>,
    {
        let mut data_fit = 0.0_f64;
        let mut assignment_sparsity = 0.0_f64;
        let mut smoothness = 0.0_f64;
        let mut ard = 0.0_f64;
        for &(start, end) in chunk_ranges {
            let n_chunk = end - start;
            let penalty_scale = n_chunk as f64 / n_total as f64;
            let (logits, coords, z_chunk) = chunk_init(start, end)?;
            let mut chunk = self.materialize_chunk(logits, coords)?;
            // #991: inherit the design honesty weight slice (global mean-1
            // normalization preserved; see streaming_exact_arrow_log_det).
            if let Some(w) = self.row_loss_weights.as_deref() {
                chunk.row_loss_weights = Some(w[start..end].to_vec());
            }
            let loss = chunk.loss_scaled(z_chunk.view(), rho, penalty_scale)?;
            data_fit += loss.data_fit;
            assignment_sparsity += loss.assignment_sparsity;
            smoothness += loss.smoothness;
            ard += loss.ard;
        }
        Ok(SaeManifoldLoss {
            data_fit,
            assignment_sparsity,
            smoothness,
            ard,
            evidence_gauge_deflated_directions: 0,
        })
    }

    pub(crate) fn streaming_loss_and_penalized_objective_total<F>(
        &self,
        chunk_ranges: &[(usize, usize)],
        rho: &SaeManifoldRho,
        analytic_penalties: Option<&AnalyticPenaltyRegistry>,
        n_total: usize,
        chunk_init: &mut F,
    ) -> Result<(SaeManifoldLoss, f64), String>
    where
        F: FnMut(usize, usize) -> Result<(Array2<f64>, Vec<Array2<f64>>, Array2<f64>), String>,
    {
        let mut data_fit = 0.0_f64;
        let mut assignment_sparsity = 0.0_f64;
        let mut smoothness = 0.0_f64;
        let mut ard = 0.0_f64;
        let mut total = 0.0_f64;
        for &(start, end) in chunk_ranges {
            let n_chunk = end - start;
            let penalty_scale = n_chunk as f64 / n_total as f64;
            let (logits, coords, z_chunk) = chunk_init(start, end)?;
            let mut chunk = self.materialize_chunk(logits, coords)?;
            // #991: inherit the design honesty weight slice (global mean-1
            // normalization preserved; see streaming_exact_arrow_log_det).
            if let Some(w) = self.row_loss_weights.as_deref() {
                chunk.row_loss_weights = Some(w[start..end].to_vec());
            }
            let loss = chunk.loss_scaled(z_chunk.view(), rho, penalty_scale)?;
            data_fit += loss.data_fit;
            assignment_sparsity += loss.assignment_sparsity;
            smoothness += loss.smoothness;
            ard += loss.ard;
            total += chunk.penalized_objective_total(
                z_chunk.view(),
                rho,
                analytic_penalties,
                penalty_scale,
            )?;
        }
        Ok((
            SaeManifoldLoss {
                data_fit,
                assignment_sparsity,
                smoothness,
                ard,
                evidence_gauge_deflated_directions: 0,
            },
            total,
        ))
    }
}
