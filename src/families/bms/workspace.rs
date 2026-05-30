use super::exact_eval_cache::*;
use super::family::*;
use super::gradient_paths::*;
use super::hessian_paths::*;
use super::row_kernel::*;
use super::*;

impl BernoulliMarginalSlopeFamily {
    #[inline]
    pub(super) fn probit_frailty_scale(&self) -> f64 {
        probit_frailty_scale(self.gaussian_frailty_sd)
    }

    #[inline]
    pub(super) fn unit_primary_direction(r: usize, idx: usize) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(r);
        out[idx] = 1.0;
        out
    }

    pub(super) fn empirical_rigid_intercept_for_row(
        &self,
        row: usize,
        marginal: BernoulliMarginalLinkMap,
        slope: f64,
        nodes: &[f64],
        measure_weights: &[f64],
    ) -> Result<f64, String> {
        // Cache slot is keyed by `(marginal.q, slope)`: a rejected TR trial
        // at one β and an accepted trial at another produce different
        // `(marginal_eta_row, slope_row)` for the same row, so without the
        // tag the slot can read back a value from a different trial and
        // poison the new root solve. The empirical-grid root depends only
        // on `(marginal.q, slope)` (the grid is immutable per latent measure),
        // so this two-scalar tag is sufficient.
        let beta_tag = hash_intercept_warm_start_key_rigid(marginal.q, slope);
        let cached = self
            .intercept_warm_starts
            .as_ref()
            .and_then(|cache| cache.load_tagged(row, beta_tag));
        let root = empirical_intercept_from_marginal(
            marginal.mu,
            marginal.q,
            slope,
            self.probit_frailty_scale(),
            nodes,
            measure_weights,
            cached,
        )?;
        if let Some(cache) = self.intercept_warm_starts.as_ref() {
            cache.store_tagged(row, root, beta_tag);
        }
        Ok(root)
    }

    pub(super) fn empirical_rigid_calibration_jets(
        &self,
        intercept: &MultiDirJet,
        mu: &MultiDirJet,
        slope: &MultiDirJet,
        nodes: &[f64],
        measure_weights: &[f64],
    ) -> (MultiDirJet, MultiDirJet) {
        let n_dirs = intercept.coeffs.len().trailing_zeros() as usize;
        let observed_slope = slope.scale(self.probit_frailty_scale());
        let mut f = mu.scale(-1.0);
        let mut f_a = MultiDirJet::zero(n_dirs);
        for (&node, &weight) in nodes.iter().zip(measure_weights.iter()) {
            let eta = intercept.add(&observed_slope.scale(node));
            let cdf = eta.compose_unary(unary_derivatives_normal_cdf(eta.coeff(0)));
            let pdf = eta.compose_unary(unary_derivatives_normal_pdf(eta.coeff(0)));
            f = f.add(&cdf.scale(weight));
            f_a = f_a.add(&pdf.scale(weight));
        }
        (f, f_a)
    }

    /// Objective-only fast path for the empirical-grid rigid kernel: returns
    /// just `-w · log Φ(s · (intercept + s_f·g·z))` evaluated at the
    /// converged scalar intercept.
    ///
    /// **Mathematically identical** to
    /// `empirical_rigid_neglog_jet(..).coeff(0)`: same scalar fixed point
    /// (the converged intercept root from `empirical_intercept_from_marginal`),
    /// same probit log-CDF evaluation. The skipped work is the per-row
    /// `MultiDirJet` construction + the 6 Newton-refinement passes that
    /// propagate the intercept through 16 directional coefficient slots;
    /// the line-search accept/reject decision never reads those coefficients.
    ///
    /// Reuses the same `intercept_warm_starts` cache as `empirical_rigid_neglog_jet`,
    /// so successive line-search trials at nearby intercepts converge in
    /// `O(1)` Newton iterations per row.
    pub(super) fn empirical_rigid_neglog_only(
        &self,
        row: usize,
        marginal: BernoulliMarginalLinkMap,
        slope: f64,
        nodes: &[f64],
        measure_weights: &[f64],
    ) -> Result<f64, String> {
        let intercept =
            self.empirical_rigid_intercept_for_row(row, marginal, slope, nodes, measure_weights)?;
        let observed_slope = slope * self.probit_frailty_scale();
        let observed_eta = intercept + observed_slope * self.z[row];
        let signed = (2.0 * self.y[row] - 1.0) * observed_eta;
        let (logcdf, _) = signed_probit_logcdf_and_mills_ratio(signed);
        if !logcdf.is_finite() {
            return Err(format!(
                "empirical rigid neglog_only: non-finite log Φ at row {row}"
            ));
        }
        Ok(-self.weights[row] * logcdf)
    }

    /// Unified scalar-objective dispatcher for the rigid Bernoulli kernel.
    /// Routes to [`RigidProbitKernel::neglog_only`] for the standard-normal
    /// latent measure and [`Self::empirical_rigid_neglog_only`] for any
    /// empirical-grid measure. Replaces `rigid_row_kernel_eval(...)`'s
    /// `(neglog, _, _)` return when only the scalar is needed.
    pub(super) fn rigid_row_neglog_only(
        &self,
        row: usize,
        marginal: BernoulliMarginalLinkMap,
        slope: f64,
    ) -> Result<f64, String> {
        match self.latent_measure.empirical_grid_for_training_row(row)? {
            None => RigidProbitKernel::neglog_only(
                marginal.q,
                slope,
                self.z[row],
                self.y[row],
                self.weights[row],
                self.probit_frailty_scale(),
            ),
            Some(grid) => {
                self.empirical_rigid_neglog_only(row, marginal, slope, &grid.nodes, &grid.weights)
            }
        }
    }

    pub(super) fn empirical_rigid_neglog_jet(
        &self,
        row: usize,
        marginal_eta: f64,
        marginal: BernoulliMarginalLinkMap,
        slope: f64,
        directions: &[[f64; 2]],
        nodes: &[f64],
        measure_weights: &[f64],
    ) -> Result<MultiDirJet, String> {
        let n_dirs = directions.len();
        let marginal_first = directions.iter().map(|dir| dir[0]).collect::<Vec<_>>();
        let slope_first = directions.iter().map(|dir| dir[1]).collect::<Vec<_>>();
        let marginal_eta_jet = MultiDirJet::linear(n_dirs, marginal_eta, &marginal_first);
        let mu_jet = marginal_eta_jet.compose_unary([
            marginal.mu,
            marginal.mu1,
            marginal.mu2,
            marginal.mu3,
            marginal.mu4,
        ]);
        let slope_jet = MultiDirJet::linear(n_dirs, slope, &slope_first);
        let intercept_root =
            self.empirical_rigid_intercept_for_row(row, marginal, slope, nodes, measure_weights)?;
        let mut intercept_jet = MultiDirJet::constant(n_dirs, intercept_root);
        for _ in 0..6 {
            let (f, f_a) = self.empirical_rigid_calibration_jets(
                &intercept_jet,
                &mu_jet,
                &slope_jet,
                nodes,
                measure_weights,
            );
            let inv_f_a = f_a.compose_unary(unary_derivatives_reciprocal(f_a.coeff(0)));
            intercept_jet = intercept_jet.add(&f.mul(&inv_f_a).scale(-1.0));
            intercept_jet.coeffs[0] = intercept_root;
        }
        let observed_slope = slope_jet.scale(self.probit_frailty_scale());
        let observed_eta = intercept_jet.add(&observed_slope.scale(self.z[row]));
        let signed = observed_eta.scale(2.0 * self.y[row] - 1.0);
        Ok(signed.compose_unary(unary_derivatives_neglog_phi(
            signed.coeff(0),
            self.weights[row],
        )))
    }

    pub(super) fn primary_component_jet(
        n_dirs: usize,
        base: f64,
        directions: &[ArrayView1<'_, f64>],
        idx: usize,
    ) -> Result<MultiDirJet, String> {
        let first = directions
            .iter()
            .map(|dir| {
                dir.get(idx).copied().ok_or_else(|| {
                    format!(
                        "bernoulli empirical flex direction length {} is too short for primary index {idx}",
                        dir.len()
                    )
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(MultiDirJet::linear(n_dirs, base, &first))
    }

    pub(super) fn local_cubic_value_jet(
        cubic: exact_kernel::LocalSpanCubic,
        x: &MultiDirJet,
    ) -> MultiDirJet {
        let n_dirs = x.coeffs.len().trailing_zeros() as usize;
        let t = x.add(&MultiDirJet::constant(n_dirs, -cubic.left));
        let t2 = t.mul(&t);
        let t3 = t2.mul(&t);
        MultiDirJet::constant(n_dirs, cubic.c0)
            .add(&t.scale(cubic.c1))
            .add(&t2.scale(cubic.c2))
            .add(&t3.scale(cubic.c3))
    }

    pub(super) fn local_cubic_first_derivative_jet(
        cubic: exact_kernel::LocalSpanCubic,
        x: &MultiDirJet,
    ) -> MultiDirJet {
        let n_dirs = x.coeffs.len().trailing_zeros() as usize;
        let t = x.add(&MultiDirJet::constant(n_dirs, -cubic.left));
        let t2 = t.mul(&t);
        MultiDirJet::constant(n_dirs, cubic.c1)
            .add(&t.scale(2.0 * cubic.c2))
            .add(&t2.scale(3.0 * cubic.c3))
    }

    pub(super) fn empirical_flex_eta_and_eta_a_jet_at_z(
        &self,
        primary: &PrimarySlices,
        a_jet: &MultiDirJet,
        b_jet: &MultiDirJet,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        directions: &[ArrayView1<'_, f64>],
        z: f64,
    ) -> Result<(MultiDirJet, MultiDirJet), String> {
        let n_dirs = directions.len();
        let mut inside = a_jet.add(&b_jet.scale(z));

        if let Some(h_range) = primary.h.as_ref() {
            let runtime = self.score_warp.as_ref().ok_or_else(|| {
                "empirical flex score-warp primary range without runtime".to_string()
            })?;
            let beta_h = beta_h.ok_or_else(|| {
                "empirical flex score-warp primary range without beta".to_string()
            })?;
            let mut h_jet = MultiDirJet::zero(n_dirs);
            Self::for_each_deviation_basis_cubic_at(
                runtime,
                h_range,
                z,
                "empirical flex score-warp",
                |local_idx, idx, basis_span| {
                    let basis_value = basis_span.evaluate(z);
                    let beta_jet =
                        Self::primary_component_jet(n_dirs, beta_h[local_idx], directions, idx)?;
                    h_jet = h_jet.add(&beta_jet.scale(basis_value));
                    Ok(())
                },
            )?;
            inside = inside.add(&b_jet.mul(&h_jet));
        }

        let u_jet = a_jet.add(&b_jet.scale(z));
        let mut w_jet = MultiDirJet::zero(n_dirs);
        let mut w_prime_jet = MultiDirJet::zero(n_dirs);
        if let Some(w_range) = primary.w.as_ref() {
            let runtime = self.link_dev.as_ref().ok_or_else(|| {
                "empirical flex link-deviation primary range without runtime".to_string()
            })?;
            let beta_w = beta_w.ok_or_else(|| {
                "empirical flex link-deviation primary range without beta".to_string()
            })?;
            let u0 = u_jet.coeff(0);
            Self::for_each_deviation_basis_cubic_at(
                runtime,
                w_range,
                u0,
                "empirical flex link-deviation",
                |local_idx, idx, basis_span| {
                    let beta_jet =
                        Self::primary_component_jet(n_dirs, beta_w[local_idx], directions, idx)?;
                    let basis_value = Self::local_cubic_value_jet(basis_span, &u_jet);
                    let basis_derivative =
                        Self::local_cubic_first_derivative_jet(basis_span, &u_jet);
                    w_jet = w_jet.add(&beta_jet.mul(&basis_value));
                    w_prime_jet = w_prime_jet.add(&beta_jet.mul(&basis_derivative));
                    Ok(())
                },
            )?;
        }

        let scale = self.probit_frailty_scale();
        let eta = inside.add(&w_jet).scale(scale);
        let eta_a = MultiDirJet::constant(n_dirs, 1.0)
            .add(&w_prime_jet)
            .scale(scale);
        Ok((eta, eta_a))
    }

    pub(super) fn empirical_flex_calibration_jets(
        &self,
        primary: &PrimarySlices,
        a_jet: &MultiDirJet,
        mu_jet: &MultiDirJet,
        b_jet: &MultiDirJet,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        directions: &[ArrayView1<'_, f64>],
        grid: &EmpiricalZGrid,
    ) -> Result<(MultiDirJet, MultiDirJet), String> {
        let n_dirs = directions.len();
        let mut f = mu_jet.scale(-1.0);
        let mut f_a = MultiDirJet::zero(n_dirs);
        for (node, weight) in grid.pairs() {
            let (eta, eta_a) = self.empirical_flex_eta_and_eta_a_jet_at_z(
                primary, a_jet, b_jet, beta_h, beta_w, directions, node,
            )?;
            let cdf = eta.compose_unary(unary_derivatives_normal_cdf(eta.coeff(0)));
            let pdf = eta.compose_unary(unary_derivatives_normal_pdf(eta.coeff(0)));
            f = f.add(&cdf.scale(weight));
            f_a = f_a.add(&pdf.mul(&eta_a).scale(weight));
        }
        Ok((f, f_a))
    }

    pub(super) fn empirical_flex_neglog_jet(
        &self,
        row: usize,
        primary: &PrimarySlices,
        q: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        directions: &[ArrayView1<'_, f64>],
        grid: &EmpiricalZGrid,
    ) -> Result<MultiDirJet, String> {
        let n_dirs = directions.len();
        if n_dirs > 6 {
            return Err(format!(
                "bernoulli empirical flex jet supports at most 6 directions, got {n_dirs}"
            ));
        }
        for dir in directions {
            if dir.len() != primary.total {
                return Err(format!(
                    "bernoulli empirical flex direction length {} != primary dimension {}",
                    dir.len(),
                    primary.total
                ));
            }
        }
        if !(row_ctx.intercept.is_finite() && row_ctx.m_a.is_finite() && row_ctx.m_a > 0.0) {
            return Err("non-finite empirical flexible row context in jet contraction".to_string());
        }

        let marginal = self.marginal_link_map(q)?;
        let q_jet = Self::primary_component_jet(n_dirs, q, directions, primary.q)?;
        let mu_jet = q_jet.compose_unary([
            marginal.mu,
            marginal.mu1,
            marginal.mu2,
            marginal.mu3,
            marginal.mu4,
        ]);
        let b_jet = Self::primary_component_jet(n_dirs, b, directions, primary.logslope)?;
        let intercept_root = row_ctx.intercept;
        let mut a_jet = MultiDirJet::constant(n_dirs, intercept_root);
        for _ in 0..6 {
            let (f, f_a) = self.empirical_flex_calibration_jets(
                primary, &a_jet, &mu_jet, &b_jet, beta_h, beta_w, directions, grid,
            )?;
            if !(f_a.coeff(0).is_finite() && f_a.coeff(0) > 0.0) {
                return Err(format!(
                    "empirical flex calibration jet has invalid F_a={}",
                    f_a.coeff(0)
                ));
            }
            let inv_f_a = f_a.compose_unary(unary_derivatives_reciprocal(f_a.coeff(0)));
            a_jet = a_jet.add(&f.mul(&inv_f_a).scale(-1.0));
            a_jet.coeffs[0] = intercept_root;
        }

        let (eta_observed, _) = self.empirical_flex_eta_and_eta_a_jet_at_z(
            primary,
            &a_jet,
            &b_jet,
            beta_h,
            beta_w,
            directions,
            self.z[row],
        )?;
        let signed = eta_observed.scale(2.0 * self.y[row] - 1.0);
        Ok(signed.compose_unary(unary_derivatives_neglog_phi(
            signed.coeff(0),
            self.weights[row],
        )))
    }

    pub(super) fn empirical_flex_row_third_contracted_recompute(
        &self,
        row: usize,
        primary: &PrimarySlices,
        q: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        dir: &Array1<f64>,
        grid: &EmpiricalZGrid,
    ) -> Result<Array2<f64>, String> {
        let r = primary.total;
        if dir.len() != r {
            return Err(format!(
                "bernoulli empirical flex third contraction direction length {} != primary dimension {r}",
                dir.len()
            ));
        }
        if dir.iter().all(|value| *value == 0.0) {
            return Ok(Array2::<f64>::zeros((r, r)));
        }
        let basis_dirs = (0..r)
            .map(|idx| Self::unit_primary_direction(r, idx))
            .collect::<Vec<_>>();
        let dir_owned = dir.to_owned();
        let mut out = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let directions = [basis_dirs[u].view(), basis_dirs[v].view(), dir_owned.view()];
                let jet = self.empirical_flex_neglog_jet(
                    row,
                    primary,
                    q,
                    b,
                    beta_h,
                    beta_w,
                    row_ctx,
                    &directions,
                    grid,
                )?;
                let val = jet.coeff(1 | 2 | 4);
                out[[u, v]] = val;
                out[[v, u]] = val;
            }
        }
        Ok(out)
    }

    pub(super) fn empirical_flex_row_fourth_contracted_recompute(
        &self,
        row: usize,
        primary: &PrimarySlices,
        q: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        dir_u: &Array1<f64>,
        dir_v: &Array1<f64>,
        grid: &EmpiricalZGrid,
    ) -> Result<Array2<f64>, String> {
        let r = primary.total;
        if dir_u.len() != r || dir_v.len() != r {
            return Err(format!(
                "bernoulli empirical flex fourth contraction direction lengths ({},{}) != primary dimension {r}",
                dir_u.len(),
                dir_v.len()
            ));
        }
        if dir_u.iter().all(|value| *value == 0.0) || dir_v.iter().all(|value| *value == 0.0) {
            return Ok(Array2::<f64>::zeros((r, r)));
        }
        let basis_dirs = (0..r)
            .map(|idx| Self::unit_primary_direction(r, idx))
            .collect::<Vec<_>>();
        let dir_u_owned = dir_u.to_owned();
        let dir_v_owned = dir_v.to_owned();
        let mut out = Array2::<f64>::zeros((r, r));
        for p in 0..r {
            for q_idx in p..r {
                let directions = [
                    basis_dirs[p].view(),
                    basis_dirs[q_idx].view(),
                    dir_u_owned.view(),
                    dir_v_owned.view(),
                ];
                let jet = self.empirical_flex_neglog_jet(
                    row,
                    primary,
                    q,
                    b,
                    beta_h,
                    beta_w,
                    row_ctx,
                    &directions,
                    grid,
                )?;
                let val = jet.coeff(1 | 2 | 4 | 8);
                out[[p, q_idx]] = val;
                out[[q_idx, p]] = val;
            }
        }
        Ok(out)
    }

    pub(super) fn rigid_row_kernel_eval(
        &self,
        row: usize,
        marginal_eta: f64,
        marginal: BernoulliMarginalLinkMap,
        slope: f64,
    ) -> Result<(f64, [f64; 2], [[f64; 2]; 2]), String> {
        match self.latent_measure.empirical_grid_for_training_row(row)? {
            None => {
                let kernel = RigidProbitKernel::new(
                    marginal.q,
                    slope,
                    self.z[row],
                    self.y[row],
                    self.weights[row],
                    self.probit_frailty_scale(),
                )?;
                Ok((
                    -self.weights[row] * kernel.logcdf,
                    rigid_transformed_gradient(marginal, &kernel),
                    rigid_transformed_hessian(marginal, &kernel),
                ))
            }
            Some(grid) => {
                let jet = self.empirical_rigid_neglog_jet(
                    row,
                    marginal_eta,
                    marginal,
                    slope,
                    &[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
                    &grid.nodes,
                    &grid.weights,
                )?;
                Ok((
                    jet.coeff(0),
                    [jet.coeff(1), jet.coeff(2)],
                    [
                        [jet.coeff(1 | 4), jet.coeff(1 | 2)],
                        [jet.coeff(1 | 2), jet.coeff(2 | 8)],
                    ],
                ))
            }
        }
    }

    pub(super) fn rigid_row_third_contracted(
        &self,
        row: usize,
        marginal_eta: f64,
        marginal: BernoulliMarginalLinkMap,
        slope: f64,
        dir_q: f64,
        dir_g: f64,
    ) -> Result<[[f64; 2]; 2], String> {
        let full = self.rigid_row_third_full(row, marginal_eta, marginal, slope)?;
        Ok(contract_third_full(&full, dir_q, dir_g))
    }

    /// Look up the per-row rigid uncontracted third-derivative tensor from
    /// the cache, populating it lazily on first access via one parallel
    /// row pass. Used by `row_primary_third_contracted_recompute` so the
    /// build-psi-hyper-coords sweep over 32 ψ-axes pays the heavy empirical
    /// jet at most once per row.
    ///
    /// Concurrent first callers may redundantly run the parallel build; the
    /// first published value wins and every subsequent caller observes the
    /// same stored result. A failed build is captured in the `Err` arm of the
    /// stored `Result` and propagates identically on every subsequent call.
    pub(super) fn rigid_third_full_cached<'a>(
        &self,
        block_states: &[ParameterBlockState],
        cache: &'a BernoulliMarginalSlopeExactEvalCache,
        row: usize,
    ) -> Result<&'a [[[f64; 2]; 2]; 2], String> {
        let stored = cache.rigid_third_full.get_or_compute(|| {
            (0..self.y.len())
                .into_par_iter()
                .map(|r| {
                    let marginal_eta = block_states[0].eta[r];
                    let marginal = self.marginal_link_map(marginal_eta)?;
                    let slope = block_states[1].eta[r];
                    self.rigid_row_third_full(r, marginal_eta, marginal, slope)
                })
                .collect::<Result<Vec<_>, String>>()
        });
        let table = stored.as_ref().map_err(|err| err.clone())?;
        Ok(&table[row])
    }

    /// Look up the per-row rigid uncontracted fourth-derivative tensor.
    /// Same lazy-build pattern as `rigid_third_full_cached`, but serves the
    /// outer-Hessian per-pair pullback path: at rank=32 ψ-axes the sweep
    /// touches `(rank² + rank)/2 = 528` (u, v) pairs, all reading the same
    /// per-row tensor. With this cache the empirical-grid 8-direction jet
    /// (or the closed-form 5-component build) runs at most once per row,
    /// then 528 cheap [`contract_fourth_full`] bilinears finish the work.
    pub(super) fn rigid_fourth_full_cached<'a>(
        &self,
        block_states: &[ParameterBlockState],
        cache: &'a BernoulliMarginalSlopeExactEvalCache,
        row: usize,
    ) -> Result<&'a [[[[f64; 2]; 2]; 2]; 2], String> {
        let stored = cache.rigid_fourth_full.get_or_compute(|| {
            (0..self.y.len())
                .into_par_iter()
                .map(|r| {
                    let marginal_eta = block_states[0].eta[r];
                    let marginal = self.marginal_link_map(marginal_eta)?;
                    let slope = block_states[1].eta[r];
                    self.rigid_row_fourth_full(r, marginal_eta, marginal, slope)
                })
                .collect::<Result<Vec<_>, String>>()
        });
        let table = stored.as_ref().map_err(|err| err.clone())?;
        Ok(&table[row])
    }

    /// Return the lazily-built row-cell-moments bundle at `required_degree`
    /// (15 or 21) for outer dH/d²H trace paths.
    ///
    /// On the first call the full-`n` bundle is built at `required_degree` and
    /// stored in the appropriate `RayonSafeOnce` slot.  Subsequent calls — even
    /// from concurrent Rayon workers — return the already-built bundle.  If the
    /// FLEX path is inactive or the memory budget is exceeded the slot stores
    /// `Ok(None)` and callers must use their per-row on-demand fallback.
    ///
    /// Returns `Ok(None)` for any `required_degree` outside {15, 21}; callers
    /// handle that the same way as a missing bundle.
    pub(super) fn bundle_for_degree<'a>(
        &self,
        block_states: &[ParameterBlockState],
        cache: &'a BernoulliMarginalSlopeExactEvalCache,
        required_degree: usize,
    ) -> Result<Option<&'a RowCellMomentsBundle>, String> {
        let slot = match required_degree {
            15 => &cache.row_cell_moments_d15,
            21 => &cache.row_cell_moments_d21,
            _ => return Ok(None),
        };
        // `get_or_compute` stores a `Result<Option<...>, String>` directly;
        // the closure returns that same type (it IS T).  The outer `?` then
        // unwraps the stored Result on every access.
        let stored = slot.get_or_compute(|| {
            // No subsample mask for the outer-derivative trace bundles: they
            // must cover all rows so that every row lookup succeeds.
            self.build_row_cell_moments_bundle(
                block_states,
                &cache.row_contexts,
                required_degree,
                None,
            )
        });
        Ok(stored.as_ref().map_err(|e| e.clone())?.as_ref())
    }

    /// Per-row uncontracted third-derivative tensor in the rigid path.
    ///
    /// Empirical-grid rows pay the heavy `empirical_rigid_neglog_jet` once
    /// (with six identity directions: three `e_q` plus three `e_g`, giving
    /// a 64-coefficient jet from which the four distinct symmetric components
    /// `T_qqq`, `T_qqg`, `T_qgg`, `T_ggg` are read directly). The `rank`-many
    /// ψ-axis directions are then folded in with a cheap 2×2 bilinear
    /// `[contract_third_full]` per call, replacing the previous
    /// `rank` separate 5-direction jets per row.
    pub(super) fn rigid_row_third_full(
        &self,
        row: usize,
        marginal_eta: f64,
        marginal: BernoulliMarginalLinkMap,
        slope: f64,
    ) -> Result<[[[f64; 2]; 2]; 2], String> {
        match self.latent_measure.empirical_grid_for_training_row(row)? {
            None => {
                let kernel = RigidProbitKernel::new(
                    marginal.q,
                    slope,
                    self.z[row],
                    self.y[row],
                    self.weights[row],
                    self.probit_frailty_scale(),
                )?;
                Ok(rigid_transformed_third_full(marginal, &kernel))
            }
            Some(grid) => {
                // Six identity directions: positions 0, 2, 4 are `e_q`,
                // positions 1, 3, 5 are `e_g`. The mask encoding for
                // `MultiDirJet::coeff` is `Σ 2^position`, so a 3-element
                // subset like {0, 2, 4} (three `e_q`s) becomes mask
                // 1 | 4 | 16 = 21 = 0b010101 — the partial `∂³f/∂q∂q∂q`.
                let jet = self.empirical_rigid_neglog_jet(
                    row,
                    marginal_eta,
                    marginal,
                    slope,
                    &[
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 0.0],
                        [0.0, 1.0],
                    ],
                    &grid.nodes,
                    &grid.weights,
                )?;
                let t_qqq = jet.coeff(1 | 4 | 16); // {0, 2, 4}: e_q × e_q × e_q
                let t_qqg = jet.coeff(1 | 4 | 2); // {0, 2, 1}: e_q × e_q × e_g
                let t_qgg = jet.coeff(1 | 2 | 8); // {0, 1, 3}: e_q × e_g × e_g
                let t_ggg = jet.coeff(2 | 8 | 32); // {1, 3, 5}: e_g × e_g × e_g
                Ok(third_full_from_symmetric_components(
                    t_qqq, t_qqg, t_qgg, t_ggg,
                ))
            }
        }
    }

    /// Per-row uncontracted fourth-derivative tensor in the rigid path.
    ///
    /// Closed-form path drops straight out of [`rigid_transformed_fourth_full`]
    /// with five primary-space components computed from the
    /// `rigid_internal_third_components` quantities and the link's higher
    /// derivatives — all axis-invariant, so the previous design that re-ran
    /// these for every (u, v) ψ-axis pair was paying `O(rank²)` redundancy.
    ///
    /// Empirical-grid path widens [`empirical_rigid_neglog_jet`] to eight
    /// identity directions (`[e_q, e_g] × 4`), giving a 256-coefficient jet
    /// from which the five distinct symmetric components `T_qqqq, T_qqqg,
    /// T_qqgg, T_qggg, T_gggg` are read directly. The (u, v) directions are
    /// folded in afterwards via the cheap [`contract_fourth_full`] bilinear
    /// — at most one jet per row total, instead of `(rank²+rank)/2` per row.
    pub(super) fn rigid_row_fourth_full(
        &self,
        row: usize,
        marginal_eta: f64,
        marginal: BernoulliMarginalLinkMap,
        slope: f64,
    ) -> Result<[[[[f64; 2]; 2]; 2]; 2], String> {
        match self.latent_measure.empirical_grid_for_training_row(row)? {
            None => {
                let kernel = RigidProbitKernel::new(
                    marginal.q,
                    slope,
                    self.z[row],
                    self.y[row],
                    self.weights[row],
                    self.probit_frailty_scale(),
                )?;
                Ok(rigid_transformed_fourth_full(marginal, &kernel))
            }
            Some(grid) => {
                // Eight identity directions: positions 0, 2, 4, 6 are `e_q`,
                // positions 1, 3, 5, 7 are `e_g`. The mask encoding for
                // `MultiDirJet::coeff` is `Σ 2^position`, so a 4-element
                // subset like {0, 2, 4, 6} (four `e_q`s) becomes mask
                // 1 | 4 | 16 | 64 = 85 — the partial `∂⁴f/∂q∂q∂q∂q`.
                let jet = self.empirical_rigid_neglog_jet(
                    row,
                    marginal_eta,
                    marginal,
                    slope,
                    &[
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 0.0],
                        [0.0, 1.0],
                    ],
                    &grid.nodes,
                    &grid.weights,
                )?;
                let t_qqqq = jet.coeff(1 | 4 | 16 | 64); // {0, 2, 4, 6}: 4× e_q
                let t_qqqg = jet.coeff(1 | 4 | 16 | 2); // {0, 2, 4, 1}: 3× e_q + 1× e_g
                let t_qqgg = jet.coeff(1 | 4 | 2 | 8); // {0, 2, 1, 3}: 2× e_q + 2× e_g
                let t_qggg = jet.coeff(1 | 2 | 8 | 32); // {0, 1, 3, 5}: 1× e_q + 3× e_g
                let t_gggg = jet.coeff(2 | 8 | 32 | 128); // {1, 3, 5, 7}: 4× e_g
                Ok(fourth_full_from_symmetric_components(
                    t_qqqq, t_qqqg, t_qqgg, t_qggg, t_gggg,
                ))
            }
        }
    }

    /// Above this `n`, line-search trial probes (calls with
    /// `options.early_exit_threshold = Some(_)`) install an
    /// auto-stratified Horvitz–Thompson subsample for the reject/accept
    /// short-circuit, and accepted trials are re-evaluated on the full
    /// data before the LL is handed to the solver. Below this `n` the
    /// full-data path is cheap enough that subsampling buys nothing
    /// (the per-row objective-only kernel collapses to a single
    /// `signed_probit_logcdf_and_mills_ratio` call on the standard-
    /// normal latent measure).
    ///
    /// Unbiasedness: every line-search *accept* decision is made on the
    /// full-data objective (the subsample only filters out
    /// *rejected* trials cheaply via the early-exit lower bound on
    /// `-Σ log Φ`), so the converged iterate is bit-identical to the
    /// full-data line search modulo the order in which the partial-sum
    /// chunks are summed.
    ///
    /// No flags, no env vars: the user-facing rule is auto-derived from
    /// problem characteristics, with this in-source constant as the
    /// canonical `n`-threshold.
    pub(crate) const AUTO_LINE_SEARCH_SUBSAMPLE_N: usize = 30_000;

    /// Outer-aware variant of `log_likelihood_only`. When
    /// `options.outer_score_subsample` is `None` this iterates over all rows
    /// and returns a value identical (bit-for-bit) to the legacy full-data
    /// implementation. When it is `Some`, only the sampled rows contribute,
    /// with their Horvitz-Thompson inverse-inclusion weights taken from
    /// `OuterScoreSubsample::rows`. This is the row-iter swap that lets outer-only
    /// score/gradient passes scale to biobank `n` without distorting the
    /// full-data inner-PIRLS or covariance code paths.
    pub(crate) fn log_likelihood_only_with_options(
        &self,
        block_states: &[ParameterBlockState],
        options: &BlockwiseFitOptions,
    ) -> Result<f64, String> {
        self.validate_exact_monotonicity(block_states)?;
        let flex_active = self.effective_flex_active(block_states)?;
        let n = self.y.len();
        // ── Auto line-search subsample ──────────────────────────────
        //
        // When the caller is a line-search trial probe
        // (`options.early_exit_threshold = Some(_)`) and has NOT
        // supplied their own outer-score subsample, and `n` is large
        // enough that the per-row trial cost dominates, auto-install a
        // stratified Horvitz-Thompson mask for the *trial* evaluation
        // only. On rejected trials this short-circuits the full row
        // sweep via `bernoulli_margslope_line_search_ll_with_early_exit`'s
        // existing early-exit. On accepted trials we re-evaluate the
        // objective on the full data before returning the LL so the
        // solver's Armijo/Wolfe check is bit-exact equivalent to the
        // full-data line search.
        //
        // The subsample uses {0, 1} y as the secondary stratum (mirrors
        // `batched_outer_gradient_terms`'s rho-gradient subsampling), so
        // rare-event imbalanced fits keep proper class representation
        // in every trial probe.
        let (effective_options, trial_subsample_installed) =
            if options.early_exit_threshold.is_some()
                && options.outer_score_subsample.is_none()
                && n >= Self::AUTO_LINE_SEARCH_SUBSAMPLE_N
            {
                let stratum_secondary: Vec<u8> = self
                    .y
                    .iter()
                    .map(|v| if *v > 0.5 { 1u8 } else { 0u8 })
                    .collect();
                let z_slice = self
                    .z
                    .as_slice()
                    .expect("BMS family z must be contiguous for line-search subsample");
                let auto_opts =
                    crate::families::marginal_slope_shared::AutoOuterSubsampleOptions::default();
                match crate::families::marginal_slope_shared::auto_outer_score_subsample(
                    z_slice,
                    Some(&stratum_secondary),
                    &auto_opts,
                ) {
                    Some(subsample) => {
                        let mut cloned = options.clone();
                        cloned.outer_score_subsample = Some(std::sync::Arc::new(subsample));
                        (std::borrow::Cow::Owned(cloned), true)
                    }
                    None => (std::borrow::Cow::Borrowed(options), false),
                }
            } else {
                (std::borrow::Cow::Borrowed(options), false)
            };
        let options: &BlockwiseFitOptions = &effective_options;
        let weighted_rows = outer_weighted_rows(options, n);
        if !flex_active {
            // Rigid probit under the active latent measure. Standard-normal
            // keeps the algebraic Gaussian identity; empirical measure solves
            // the calibrated intercept against its quadrature grid.
            //
            // **Objective-only fast path.** The line-search accept/reject
            // decision only needs the scalar negative log-likelihood; the
            // gradient and Hessian returned by `rigid_row_kernel_eval` would
            // be immediately discarded. `rigid_row_neglog_only` dispatches
            // to:
            //   * `RigidProbitKernel::neglog_only` (standard-normal): a single
            //     `signed_probit_logcdf_and_mills_ratio` call, skipping the
            //     `u_k`/`c_k`/`eta_*` chain-rule scaffolding.
            //   * `empirical_rigid_neglog_only` (empirical-grid): the
            //     converged scalar intercept (from
            //     `empirical_rigid_intercept_for_row`'s warm-start cache) plus
            //     a single probit log-CDF eval, skipping the four-direction
            //     `MultiDirJet` construction and its six Newton-refinement
            //     passes (the line search reads no derivative coefficients).
            // The returned value is bit-equivalent to
            // `rigid_row_kernel_eval(...).0` at the same row state.
            let b = &block_states[1].eta;
            let row_ll = |i: usize| -> Result<f64, String> {
                let marginal_eta = block_states[0].eta[i];
                let marginal = self.marginal_link_map(marginal_eta)?;
                let neglog = self.rigid_row_neglog_only(i, marginal, b[i])?;
                Ok(-neglog)
            };
            if let Some(threshold) = options.early_exit_threshold {
                let trial_result = bernoulli_margslope_line_search_ll_with_early_exit(
                    &weighted_rows,
                    threshold,
                    row_ll,
                );
                // Trial accepted on the auto-installed subsample: re-
                // evaluate on the full data so the solver's Armijo/Wolfe
                // check decides on the bit-exact full-data objective.
                // Rejected trials short-circuit through the `Err` path —
                // no full-data work paid on the reject.
                if trial_subsample_installed && let Ok(_subsample_ll) = trial_result {
                    let full_total: Result<f64, String> = (0..n)
                        .into_par_iter()
                        .try_fold(
                            || 0.0,
                            |mut ll, i| -> Result<_, String> {
                                ll += row_ll(i)?;
                                Ok(ll)
                            },
                        )
                        .try_reduce(
                            || 0.0,
                            |left, right| -> Result<_, String> { Ok(left + right) },
                        );
                    return full_total;
                }
                return trial_result;
            }
            let total: Result<f64, String> = weighted_rows
                .into_par_iter()
                .try_fold(
                    || 0.0,
                    |mut ll, wr| -> Result<_, String> {
                        ll += wr.weight * row_ll(wr.index)?;
                        Ok(ll)
                    },
                )
                .try_reduce(
                    || 0.0,
                    |left, right| -> Result<_, String> { Ok(left + right) },
                );
            return total;
        }
        let beta_h = self.score_beta(block_states)?;
        let beta_w = self.link_beta(block_states)?;
        let row_ll = |row: usize| -> Result<f64, String> {
            let intercept = self
                .solve_row_intercept_base(
                    row,
                    block_states[0].eta[row],
                    block_states[1].eta[row],
                    beta_h,
                    beta_w,
                    None,
                )?
                .0;
            let slope = block_states[1].eta[row];
            let obs =
                self.observed_denested_cell_partials(row, intercept, slope, beta_h, beta_w)?;
            let s_i = eval_coeff4_at(&obs.coeff, self.z[row]);
            let signed = (2.0 * self.y[row] - 1.0) * s_i;
            let (log_cdf, _) = signed_probit_logcdf_and_mills_ratio(signed);
            Ok(self.weights[row] * log_cdf)
        };
        if let Some(threshold) = options.early_exit_threshold {
            let trial_result = bernoulli_margslope_line_search_ll_with_early_exit(
                &weighted_rows,
                threshold,
                row_ll,
            );
            if trial_subsample_installed && let Ok(_subsample_ll) = trial_result {
                let full_total: Result<f64, String> = (0..n)
                    .into_par_iter()
                    .try_fold(
                        || 0.0,
                        |mut ll, i| -> Result<_, String> {
                            ll += row_ll(i)?;
                            Ok(ll)
                        },
                    )
                    .try_reduce(
                        || 0.0,
                        |left, right| -> Result<_, String> { Ok(left + right) },
                    );
                return full_total;
            }
            return trial_result;
        }
        let total: Result<f64, String> = weighted_rows
            .into_par_iter()
            .try_fold(
                || 0.0,
                |mut ll, wr| -> Result<_, String> {
                    ll += wr.weight * row_ll(wr.index)?;
                    Ok(ll)
                },
            )
            .try_reduce(
                || 0.0,
                |left, right| -> Result<_, String> { Ok(left + right) },
            );
        total
    }

    pub(super) fn is_sigma_aux_index(
        &self,
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> bool {
        shared_is_sigma_aux_index(self.gaussian_frailty_sd, derivative_blocks, psi_index)
    }

    pub(super) fn sigma_scale_jet(
        &self,
        n_dirs: usize,
        first_masks: &[usize],
        second_masks: &[usize],
    ) -> Result<MultiDirJet, String> {
        probit_frailty_scale_multi_dir_jet(
            self.gaussian_frailty_sd,
            "bernoulli marginal-slope log-sigma auxiliary requested without GaussianShift sigma",
            n_dirs,
            first_masks,
            second_masks,
        )
    }

    pub(super) fn row_neglog_directional_with_scale_jet(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dirs: &[Array1<f64>],
        scale_jet: &MultiDirJet,
    ) -> Result<f64, String> {
        let k = dirs.len();
        if k > 4 {
            return Err(format!(
                "bernoulli marginal-slope sigma row directional expects 0..=4 directions, got {k}"
            ));
        }
        if scale_jet.coeffs.len() != (1usize << k) {
            return Err(format!(
                "bernoulli marginal-slope sigma scale jet dimension mismatch: coeffs={}, dirs={k}",
                scale_jet.coeffs.len()
            ));
        }

        let first = |idx: usize| -> Vec<f64> { dirs.iter().map(|dir| dir[idx]).collect() };
        let marginal = self.marginal_link_map(block_states[0].eta[row])?;
        let eta_jet = MultiDirJet::linear(k, block_states[0].eta[row], &first(0));
        let q_jet = eta_jet.compose_unary([
            marginal.q,
            marginal.q1,
            marginal.q2,
            marginal.q3,
            marginal.q4,
        ]);
        let g_jet = MultiDirJet::linear(k, block_states[1].eta[row], &first(1));
        let observed_g_jet = g_jet.mul(scale_jet);
        let one_plus_b2 = MultiDirJet::constant(k, 1.0).add(&observed_g_jet.mul(&observed_g_jet));
        let c_jet = one_plus_b2.compose_unary(unary_derivatives_sqrt(one_plus_b2.coeff(0)));
        let z_jet = MultiDirJet::constant(k, self.z[row]);
        let eta_observed_jet = q_jet.mul(&c_jet).add(&observed_g_jet.mul(&z_jet));
        let signed_jet = eta_observed_jet.scale(2.0 * self.y[row] - 1.0);
        Ok(signed_jet
            .compose_unary(unary_derivatives_neglog_phi(
                signed_jet.coeff(0),
                self.weights[row],
            ))
            .coeff((1usize << k) - 1))
    }

    pub(super) fn row_sigma_primary_terms(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        second_sigma: bool,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        let primary_dim = 2usize;
        let zero = Array1::<f64>::zeros(primary_dim);
        // The leading prefix is the fixed number of zero primary directions the
        // log-sigma hyperderivative differentiates *through*: one for the first
        // log-sigma derivative, two for the second. The shared sweep appends the
        // unit primary directions for grad/hess on top of this prefix.
        let (leading, scales): (Vec<&Array1<f64>>, DirectionalScaleJets) = if second_sigma {
            (
                vec![&zero, &zero],
                DirectionalScaleJets {
                    obj: Some(self.sigma_scale_jet(2, &[1, 2], &[3])?),
                    grad: self.sigma_scale_jet(3, &[1, 2], &[3])?,
                    hess: self.sigma_scale_jet(4, &[1, 2], &[3])?,
                },
            )
        } else {
            (
                vec![&zero],
                DirectionalScaleJets {
                    obj: Some(self.sigma_scale_jet(1, &[1], &[])?),
                    grad: self.sigma_scale_jet(2, &[1], &[])?,
                    hess: self.sigma_scale_jet(3, &[1], &[])?,
                },
            )
        };
        let terms = directional_obj_grad_hess(
            primary_dim,
            &leading,
            &scales,
            |dirs, scale| {
                let owned: Vec<Array1<f64>> = dirs.iter().map(|d| (*d).clone()).collect();
                self.row_neglog_directional_with_scale_jet(row, block_states, &owned, scale)
            },
        )?;
        Ok((terms.objective, terms.grad, terms.hess))
    }

    pub(super) fn accumulate_rigid_sigma_pullback(
        &self,
        row: usize,
        slices: &BlockSlices,
        primary_grad: &Array1<f64>,
        primary_hessian: &Array2<f64>,
        score: &mut Array1<f64>,
        hessian: &mut BernoulliBlockHessianAccumulator,
    ) -> Result<(), String> {
        {
            let mut marginal = score.slice_mut(s![slices.marginal.clone()]);
            self.marginal_design
                .axpy_row_into(row, primary_grad[0], &mut marginal)?;
        }
        {
            let mut logslope = score.slice_mut(s![slices.logslope.clone()]);
            self.logslope_design
                .axpy_row_into(row, primary_grad[1], &mut logslope)?;
        }
        hessian.add_pullback(self, row, slices, &primary_slices(slices), primary_hessian);
        Ok(())
    }

    pub(super) fn sigma_exact_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        self.sigma_exact_joint_psi_terms_with_options(
            block_states,
            specs,
            &BlockwiseFitOptions::default(),
        )
    }

    /// Outer-aware variant of `sigma_exact_joint_psi_terms`. When
    /// `options.outer_score_subsample` is `None`, iterates all rows and is
    /// bit-for-bit equivalent to the legacy implementation. When `Some`, only
    /// the sampled rows contribute and every row-summed component (objective
    /// scalar, score vector, Hessian operator blocks) is accumulated with the
    /// row's Horvitz-Thompson inverse-inclusion weight.
    pub(crate) fn sigma_exact_joint_psi_terms_with_options(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        if specs.len() != block_states.len() {
            return Err(format!(
                "bernoulli marginal-slope sigma psi terms: specs/block_states length mismatch {} vs {}",
                specs.len(),
                block_states.len()
            ));
        }
        if self.effective_flex_active(block_states)? {
            return Err("bernoulli marginal-slope log-sigma hyperderivatives are implemented for the rigid probit marginal-slope kernel; flexible score/link kernels require the analytic denested cell-tensor sigma path"
                        .to_string());
        }
        if self.gaussian_frailty_sd.is_none() {
            return Ok(None);
        }
        let slices = block_slices(self);
        let n = self.y.len();
        let row_iter = outer_row_indices(options, n).to_vec();
        let row_weights =
            crate::families::marginal_slope_shared::outer_row_weights_by_index(options, n);
        // Per-row HT weighting: each row's (obj, grad, hess) is multiplied by
        // its inverse-inclusion weight `w_i` *before* accumulation, so the
        // final operator is the unbiased Horvitz-Thompson estimator. A single
        // post-sum scalar is biased under stratified subsampling because
        // per-stratum sampling fractions differ.
        let (objective_psi, score_psi, acc) = chunked_row_reduction(
            row_iter.as_slice(),
            || {
                (
                    0.0,
                    Array1::<f64>::zeros(slices.total),
                    BernoulliBlockHessianAccumulator::new(&slices),
                )
            },
            |row, acc| -> Result<(), String> {
                let (mut obj, mut grad, mut hess) =
                    self.row_sigma_primary_terms(row, block_states, false)?;
                let w = row_weights[row];
                if w != 1.0 {
                    obj *= w;
                    grad.mapv_inplace(|v| v * w);
                    hess.mapv_inplace(|v| v * w);
                }
                acc.0 += obj;
                self.accumulate_rigid_sigma_pullback(
                    row, &slices, &grad, &hess, &mut acc.1, &mut acc.2,
                )?;
                Ok(())
            },
            |total, chunk| {
                total.0 += chunk.0;
                total.1 += &chunk.1;
                total.2.add(&chunk.2);
            },
        )?;
        Ok(Some(ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi: Array2::zeros((0, 0)),
            hessian_psi_operator: Some(Arc::new(acc.into_operator(&slices))),
        }))
    }

    pub(super) fn sigma_exact_joint_psisecond_order_terms(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        self.sigma_exact_joint_psisecond_order_terms_with_options(
            block_states,
            &BlockwiseFitOptions::default(),
        )
    }

    /// Outer-aware variant of `sigma_exact_joint_psisecond_order_terms`. See
    /// `sigma_exact_joint_psi_terms_with_options` for the row-iter / weighting
    /// contract.
    pub(crate) fn sigma_exact_joint_psisecond_order_terms_with_options(
        &self,
        block_states: &[ParameterBlockState],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        if self.effective_flex_active(block_states)? {
            return Err("bernoulli marginal-slope second log-sigma hyperderivatives are implemented for the rigid probit marginal-slope kernel; flexible score/link kernels require the analytic denested cell-tensor sigma path"
                        .to_string());
        }
        if self.gaussian_frailty_sd.is_none() {
            return Ok(None);
        }
        let slices = block_slices(self);
        let n = self.y.len();
        let row_iter = outer_row_indices(options, n).to_vec();
        let row_weights =
            crate::families::marginal_slope_shared::outer_row_weights_by_index(options, n);
        let (objective_psi_psi, score_psi_psi, acc) = chunked_row_reduction(
            row_iter.as_slice(),
            || {
                (
                    0.0,
                    Array1::<f64>::zeros(slices.total),
                    BernoulliBlockHessianAccumulator::new(&slices),
                )
            },
            |row, acc| -> Result<(), String> {
                let (mut obj, mut grad, mut hess) =
                    self.row_sigma_primary_terms(row, block_states, true)?;
                let w = row_weights[row];
                if w != 1.0 {
                    obj *= w;
                    grad.mapv_inplace(|v| v * w);
                    hess.mapv_inplace(|v| v * w);
                }
                acc.0 += obj;
                self.accumulate_rigid_sigma_pullback(
                    row, &slices, &grad, &hess, &mut acc.1, &mut acc.2,
                )?;
                Ok(())
            },
            |total, chunk| {
                total.0 += chunk.0;
                total.1 += &chunk.1;
                total.2.add(&chunk.2);
            },
        )?;
        Ok(Some(ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi,
            hessian_psi_psi: Array2::zeros((0, 0)),
            hessian_psi_psi_operator: Some(Box::new(acc.into_operator(&slices))),
        }))
    }

    pub(super) fn sigma_exact_joint_psihessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.sigma_exact_joint_psihessian_directional_derivative_with_options(
            block_states,
            d_beta_flat,
            &BlockwiseFitOptions::default(),
        )
    }

    /// Outer-aware variant of `sigma_exact_joint_psihessian_directional_derivative`.
    /// See `sigma_exact_joint_psi_terms_with_options` for the row-iter /
    /// weighting contract — the returned dense Hessian-derivative matrix is
    /// accumulated with per-row inverse-inclusion weights when a subsample is active.
    pub(crate) fn sigma_exact_joint_psihessian_directional_derivative_with_options(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Array2<f64>>, String> {
        if self.effective_flex_active(block_states)? {
            return Err("bernoulli marginal-slope log-sigma Hessian directional derivatives are implemented for the rigid probit marginal-slope kernel; flexible score/link kernels require the analytic denested cell-tensor sigma path"
                        .to_string());
        }
        if self.gaussian_frailty_sd.is_none() {
            return Ok(None);
        }
        let slices = block_slices(self);
        if d_beta_flat.len() != slices.total {
            return Err(format!(
                "bernoulli marginal-slope d_beta length mismatch for sigma Hessian derivative: got {}, expected {}",
                d_beta_flat.len(),
                slices.total
            ));
        }
        let n = self.y.len();
        let primary = primary_slices(&slices);
        let row_iter = outer_row_indices(options, n).to_vec();
        let row_weights =
            crate::families::marginal_slope_shared::outer_row_weights_by_index(options, n);
        let acc = chunked_row_reduction(
            row_iter.as_slice(),
            || BernoulliBlockHessianAccumulator::new(&slices),
            |row, acc| -> Result<(), String> {
                let row_dir =
                    self.row_primary_direction_from_flat(row, &slices, &primary, d_beta_flat)?;
                let zero = Array1::<f64>::zeros(primary.total);
                let mut grad = Array1::<f64>::zeros(primary.total);
                for a in 0..primary.total {
                    let mut da = Array1::<f64>::zeros(primary.total);
                    da[a] = 1.0;
                    let scale = self.sigma_scale_jet(3, &[1], &[])?;
                    grad[a] = self.row_neglog_directional_with_scale_jet(
                        row,
                        block_states,
                        &[zero.clone(), row_dir.clone(), da],
                        &scale,
                    )?;
                }
                let mut hess = Array2::<f64>::zeros((primary.total, primary.total));
                for a in 0..primary.total {
                    let mut da = Array1::<f64>::zeros(primary.total);
                    da[a] = 1.0;
                    for b in a..primary.total {
                        let mut db = Array1::<f64>::zeros(primary.total);
                        db[b] = 1.0;
                        let scale = self.sigma_scale_jet(4, &[1], &[])?;
                        let value = self.row_neglog_directional_with_scale_jet(
                            row,
                            block_states,
                            &[zero.clone(), row_dir.clone(), da.clone(), db],
                            &scale,
                        )?;
                        hess[[a, b]] = value;
                        hess[[b, a]] = value;
                    }
                }
                let w = row_weights[row];
                if w != 1.0 {
                    hess.mapv_inplace(|v| v * w);
                }
                acc.add_pullback(self, row, &slices, &primary, &hess);
                Ok(())
            },
            |total, chunk| {
                total.add(&chunk);
            },
        )?;
        Ok(Some(acc.into_operator(&slices).to_dense()))
    }

    #[inline]
    pub(super) fn marginal_link_map(&self, eta: f64) -> Result<BernoulliMarginalLinkMap, String> {
        bernoulli_marginal_link_map(&self.base_link, eta)
    }

    #[inline]
    pub(super) fn exact_newton_score_component_from_objective_gradient(
        objective_gradient_component: f64,
    ) -> f64 {
        -objective_gradient_component
    }

    #[inline]
    pub(super) fn exact_newton_score_from_objective_gradient(
        objective_gradient: Array1<f64>,
    ) -> Array1<f64> {
        -objective_gradient
    }

    #[inline]
    pub(super) fn exact_newton_observed_information_from_objective_hessian(
        objective_hessian: Array2<f64>,
    ) -> Array2<f64> {
        objective_hessian
    }

    #[inline]
    pub(super) fn score_block_index(&self) -> Option<usize> {
        self.score_warp.as_ref().map(|_| 2)
    }

    #[inline]
    pub(super) fn link_block_index(&self) -> Option<usize> {
        self.link_dev
            .as_ref()
            .map(|_| 2 + usize::from(self.score_warp.is_some()))
    }

    pub(super) fn optional_exact_block_state<'a>(
        &self,
        block_states: &'a [ParameterBlockState],
        block_idx: Option<usize>,
        label: &str,
    ) -> Result<Option<&'a ParameterBlockState>, String> {
        match block_idx {
            Some(idx) => block_states
                .get(idx)
                .map(Some)
                .ok_or_else(|| format!("missing {label} block state")),
            None => Ok(None),
        }
    }

    pub(super) fn score_block_state<'a>(
        &self,
        block_states: &'a [ParameterBlockState],
    ) -> Result<Option<&'a ParameterBlockState>, String> {
        self.optional_exact_block_state(block_states, self.score_block_index(), "score-warp")
    }

    pub(super) fn link_block_state<'a>(
        &self,
        block_states: &'a [ParameterBlockState],
    ) -> Result<Option<&'a ParameterBlockState>, String> {
        self.optional_exact_block_state(block_states, self.link_block_index(), "link deviation")
    }

    pub(super) fn score_beta<'a>(
        &self,
        block_states: &'a [ParameterBlockState],
    ) -> Result<Option<&'a Array1<f64>>, String> {
        Ok(self
            .score_block_state(block_states)?
            .map(|state| &state.beta))
    }

    pub(super) fn link_beta<'a>(
        &self,
        block_states: &'a [ParameterBlockState],
    ) -> Result<Option<&'a Array1<f64>>, String> {
        Ok(self
            .link_block_state(block_states)?
            .map(|state| &state.beta))
    }

    pub(super) fn validate_exact_block_state_shapes(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(), String> {
        let expected_blocks =
            2usize + usize::from(self.score_warp.is_some()) + usize::from(self.link_dev.is_some());
        if block_states.len() != expected_blocks {
            return Err(format!(
                "bernoulli marginal-slope block count mismatch: got {}, expected {}",
                block_states.len(),
                expected_blocks
            ));
        }

        let n_rows = self.y.len();
        let marginal = &block_states[0];
        let marginal_ncols = self.marginal_design.ncols();
        if marginal_ncols > 0 && marginal.beta.len() != marginal_ncols {
            return Err(format!(
                "bernoulli marginal-slope marginal beta length mismatch: got {}, expected {}",
                marginal.beta.len(),
                marginal_ncols
            ));
        }
        if marginal.eta.len() != n_rows {
            return Err(format!(
                "bernoulli marginal-slope marginal eta length mismatch: got {}, expected {}",
                marginal.eta.len(),
                n_rows
            ));
        }

        let logslope = &block_states[1];
        let logslope_ncols = self.logslope_design.ncols();
        if logslope_ncols > 0 && logslope.beta.len() != logslope_ncols {
            return Err(format!(
                "bernoulli marginal-slope logslope beta length mismatch: got {}, expected {}",
                logslope.beta.len(),
                logslope_ncols
            ));
        }
        if logslope.eta.len() != n_rows {
            return Err(format!(
                "bernoulli marginal-slope logslope eta length mismatch: got {}, expected {}",
                logslope.eta.len(),
                n_rows
            ));
        }

        if let Some(runtime) = &self.score_warp {
            let score = self
                .score_block_state(block_states)?
                .expect("score-warp block should exist when runtime is present");
            if score.beta.len() != runtime.basis_dim() {
                return Err(format!(
                    "bernoulli marginal-slope score-warp beta length mismatch: got {}, expected {}",
                    score.beta.len(),
                    runtime.basis_dim()
                ));
            }
            if score.eta.len() != n_rows {
                return Err(format!(
                    "bernoulli marginal-slope score-warp eta length mismatch: got {}, expected {}",
                    score.eta.len(),
                    n_rows
                ));
            }
        }

        if let Some(runtime) = &self.link_dev {
            let link = self
                .link_block_state(block_states)?
                .expect("link-deviation block should exist when runtime is present");
            if link.beta.len() != runtime.basis_dim() {
                return Err(format!(
                    "bernoulli marginal-slope link-deviation beta length mismatch: got {}, expected {}",
                    link.beta.len(),
                    runtime.basis_dim()
                ));
            }
            if link.eta.len() != n_rows {
                return Err(format!(
                    "bernoulli marginal-slope link-deviation eta length mismatch: got {}, expected {}",
                    link.eta.len(),
                    n_rows
                ));
            }
        }

        Ok(())
    }

    pub(super) fn denested_partition_cells(
        &self,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<Vec<exact_kernel::DenestedPartitionCell>, String> {
        shared_denested_partition_cells(
            a,
            b,
            self.score_warp.as_ref(),
            beta_h,
            self.link_dev.as_ref(),
            beta_w,
            self.probit_frailty_scale(),
        )
    }

    pub(super) fn max_denested_partition_cells_per_row(&self) -> usize {
        let score_splits = self
            .score_warp
            .as_ref()
            .map_or(0usize, |runtime| runtime.breakpoints().len());
        let link_splits = self
            .link_dev
            .as_ref()
            .map_or(0usize, |runtime| runtime.breakpoints().len());
        score_splits.saturating_add(link_splits).saturating_add(1)
    }

    #[inline]
    pub(super) fn evaluate_cell_moments_lru(
        &self,
        cell: exact_kernel::DenestedCubicCell,
        max_degree: usize,
    ) -> Result<exact_kernel::CellMomentState, String> {
        self.cell_moment_cache_stats.record_miss();
        exact_kernel::evaluate_cell_moments_uncached(cell, max_degree)
    }

    #[inline]
    pub(super) fn evaluate_cell_derivative_moments_lru(
        &self,
        cell: exact_kernel::DenestedCubicCell,
        max_degree: usize,
    ) -> Result<exact_kernel::CellDerivativeMomentState, String> {
        exact_kernel::evaluate_cell_derivative_moments_cached(
            cell,
            max_degree,
            &self.cell_moment_lru,
            Some(&self.cell_moment_cache_stats),
        )
    }

    #[inline]
    pub(super) fn for_each_deviation_basis_cubic_at<F>(
        runtime: &DeviationRuntime,
        primary_range: &std::ops::Range<usize>,
        value: f64,
        label: &str,
        mut visit: F,
    ) -> Result<(), String>
    where
        F: FnMut(usize, usize, exact_kernel::LocalSpanCubic) -> Result<(), String>,
    {
        if primary_range.len() != runtime.basis_dim() {
            return Err(format!(
                "{label} primary range length {} does not match deviation basis dimension {}",
                primary_range.len(),
                runtime.basis_dim()
            ));
        }
        runtime.for_each_basis_cubic_at(value, |local_idx, basis_span| {
            visit(local_idx, primary_range.start + local_idx, basis_span)
        })
    }

    /// Newton-step evaluator for the inner-PIRLS row-intercept root solver.
    ///
    /// Returns `(f, f', 0.0)`: the third slot — `F''(a)` — is reported as
    /// zero, which makes [`monotone_root::solve_monotone_root`]'s safeguarded
    /// Halley step reduce to a Newton step. A measured degree-9 `F''(a)` path
    /// did not reduce calibration evaluations on the biobank FLEX repro, and
    /// it made each value-bearing cell evaluation slower; degree 4 is the
    /// correct cost/accuracy point for this solver.
    pub(super) fn evaluate_denested_calibration_newton(
        &self,
        a: f64,
        marginal_eta: f64,
        slope: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<(f64, f64, f64), String> {
        let marginal = self.marginal_link_map(marginal_eta)?;
        let cells = self.denested_partition_cells(a, slope, beta_h, beta_w)?;
        let scale = self.probit_frailty_scale();
        let mut f = -marginal.mu;
        let mut f_a = 0.0;
        for partition_cell in cells {
            let cell = partition_cell.cell;
            let state = self.evaluate_cell_moments_lru(cell, 4)?;
            f += state.value;
            let (dc_da_raw, _) = exact_kernel::denested_cell_coefficient_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                slope,
            );
            let dc_da = scale_coeff4(dc_da_raw, scale);
            f_a += exact_kernel::cell_first_derivative_from_moments(&dc_da, &state.moments)?;
        }
        Ok((f, f_a, 0.0))
    }

    pub(super) fn evaluate_empirical_grid_calibration_newton(
        &self,
        a: f64,
        marginal_eta: f64,
        slope: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        grid: &EmpiricalZGrid,
    ) -> Result<(f64, f64, f64), String> {
        let marginal = self.marginal_link_map(marginal_eta)?;
        let mut f = -marginal.mu;
        let mut f_a = 0.0;
        let mut f_aa = 0.0;
        for (node, weight) in grid.pairs() {
            let obs = self.observed_denested_cell_partials_at_z(node, a, slope, beta_h, beta_w)?;
            let eta = eval_coeff4_at(&obs.coeff, node);
            let eta_a = eval_coeff4_at(&obs.dc_da, node);
            let eta_aa = eval_coeff4_at(&obs.dc_daa, node);
            let pdf = normal_pdf(eta);
            f += weight * normal_cdf(eta);
            f_a += weight * pdf * eta_a;
            f_aa += weight * pdf * (eta_aa - eta * eta_a * eta_a);
        }
        if !(f.is_finite() && f_a.is_finite() && f_a > 0.0 && f_aa.is_finite()) {
            return Err(format!(
                "empirical latent denested calibration produced invalid root state: f={f}, f_a={f_a}, f_aa={f_aa}"
            ));
        }
        Ok((f, f_a, f_aa))
    }

    pub(super) fn evaluate_calibration_newton(
        &self,
        row: usize,
        a: f64,
        marginal_eta: f64,
        slope: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<(f64, f64, f64), String> {
        match self.latent_measure.empirical_grid_for_training_row(row)? {
            None => {
                self.evaluate_denested_calibration_newton(a, marginal_eta, slope, beta_h, beta_w)
            }
            Some(grid) => self.evaluate_empirical_grid_calibration_newton(
                a,
                marginal_eta,
                slope,
                beta_h,
                beta_w,
                &grid,
            ),
        }
    }

    pub(super) fn flex_active(&self) -> bool {
        self.score_warp.is_some() || self.link_dev.is_some()
    }

    /// The denested exact path is active whenever either deviation runtime is
    /// configured. Zero coefficient vectors still keep the flexible geometry
    /// live so derivatives with respect to those coefficients remain available.
    pub(super) fn effective_flex_active(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<bool, String> {
        if self.score_warp.is_some() && self.score_beta(block_states)?.is_none() {
            return Err("missing bernoulli score-warp block state".to_string());
        }
        if self.link_dev.is_some() && self.link_beta(block_states)?.is_none() {
            return Err("missing bernoulli link-deviation block state".to_string());
        }
        Ok(self.flex_active())
    }

    pub(super) fn validate_exact_monotonicity(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(), String> {
        self.validate_exact_block_state_shapes(block_states)?;
        if let (Some(runtime), Some(score)) =
            (&self.score_warp, self.score_block_state(block_states)?)
        {
            runtime.monotonicity_feasible(
                &score.beta,
                "bernoulli marginal-slope score-warp deviation",
            )?;
        }
        if let (Some(runtime), Some(beta_w)) = (&self.link_dev, self.link_beta(block_states)?) {
            runtime.monotonicity_feasible(beta_w, "bernoulli marginal-slope link deviation")?;
        }
        Ok(())
    }

    /// Single-row link-deviation value and first derivative at `eta0`,
    /// honouring any cross-block anchor residual on `link_dev`.
    ///
    /// The closed-form intercept seed `row_intercept_closed_form_seed` is
    /// called once per training row from `solve_row_intercept_base`; each
    /// call needs `ℓ(η_a) = η_a + Φ(η_a) · β` evaluated at the row's pre-
    /// scale rigid intercept `a_rigid_pre_scale`. When the link-deviation
    /// runtime has been reparameterised against the marginal+logslope
    /// parametric anchor, the per-row reparameterised basis is
    ///
    ///   Φ_new[row, :] = Φ_raw(η_a) − parametric_anchor[row, :] · M
    ///
    /// so the design value at `(row, η_a)` is the raw basis minus a row-
    /// specific subtraction. `runtime.design()` returns the raw basis
    /// only and `assert`s in this configuration so callers don't
    /// silently miscompute; instead route through `design_with_anchor_rows`
    /// with the runtime's cached training-row anchor sliced to a single
    /// row. The derivative path is unaffected — the subtraction is
    /// constant in `η`, so its derivative is identically zero.
    pub(super) fn link_terms_value_d1_at_row(
        &self,
        row: usize,
        eta0: f64,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<(f64, f64), String> {
        let (Some(runtime), Some(beta)) = (&self.link_dev, beta_w) else {
            return Ok((eta0, 1.0));
        };
        let values = Array1::from_vec(vec![eta0]);
        let basis = if let Some(anchor_rows) = runtime.anchor_rows_at_training() {
            if row >= anchor_rows.nrows() {
                return Err(format!(
                    "link_terms_value_d1_at_row: row {row} out of bounds for {} cached training anchor rows",
                    anchor_rows.nrows()
                ));
            }
            let anchor_view = anchor_rows.slice(ndarray::s![row..row + 1, ..]);
            runtime.design_with_anchor_rows(&values, anchor_view)?
        } else {
            runtime.design(&values)?
        };
        let d1 = runtime.first_derivative_design(&values)?;
        Ok((eta0 + basis.row(0).dot(beta), d1.row(0).dot(beta) + 1.0))
    }

    pub(super) fn row_intercept_closed_form_seed(
        &self,
        row: usize,
        marginal: BernoulliMarginalLinkMap,
        slope: f64,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<f64, String> {
        let probit_scale = self.probit_frailty_scale();
        let a_rigid_pre_scale =
            rigid_intercept_from_marginal(marginal.q, slope, probit_scale) / probit_scale;
        if beta_w.is_some() {
            let (l_val, l_d1) = self.link_terms_value_d1_at_row(row, a_rigid_pre_scale, beta_w)?;
            if l_d1 > BMS_DERIV_TOL {
                let ell0 = l_val - l_d1 * a_rigid_pre_scale;
                let observed_logslope = probit_scale * l_d1 * slope;
                return Ok(
                    (marginal.q * (1.0 + observed_logslope * observed_logslope).sqrt()
                        / probit_scale
                        - ell0)
                        / l_d1,
                );
            }
        }
        Ok(a_rigid_pre_scale)
    }

    /// Pre-seed cold (`NaN`) per-row intercept warm-start slots with the
    /// closed-form rigid/affine seed for the current `(marginal_eta, slope)`
    /// state, before the parallel root solves run. Slots already populated
    /// from a prior PIRLS/outer iteration are preserved verbatim — only NaN
    /// slots are CAS-installed. This avoids recomputing the seed inside every
    /// `solve_row_intercept_base` call on cold cycle 0.
    pub(super) fn preseed_intercept_warm_starts(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(), String> {
        if !self.effective_flex_active(block_states)? {
            return Ok(());
        }
        let Some(cache) = self.intercept_warm_starts.as_ref() else {
            return Ok(());
        };
        let beta_w = self.link_beta(block_states)?;
        let n = self.y.len();
        if cache.len() != n {
            return Ok(());
        }
        let marginal_eta = &block_states[0].eta;
        let slope_eta = &block_states[1].eta;
        let probit_scale = self.probit_frailty_scale();

        // Per-row marginal link map and rigid pre-scale intercept.
        let marginals: Vec<BernoulliMarginalLinkMap> = (0..n)
            .into_par_iter()
            .map(|row| self.marginal_link_map(marginal_eta[row]))
            .collect::<Result<Vec<_>, _>>()?;
        let a_pre_scale_vec: Array1<f64> = (0..n)
            .map(|row| {
                rigid_intercept_from_marginal(marginals[row].q, slope_eta[row], probit_scale)
                    / probit_scale
            })
            .collect();

        // Batched link-deviation evaluation at each row's pre-scale intercept.
        //
        // The closed-form intercept seed needs ℓ(a_pre_scale_i) and
        // ℓ'(a_pre_scale_i) where
        //
        //   ℓ(η) = η + Φ_link_dev(η) · β_link
        //
        // is the row-i link deviation. After
        // `install_compiled_flex_block_into_runtime`
        // reparameterised the link-deviation runtime against the
        // marginal+logslope parametric anchor union, the per-row
        // reparameterised basis is
        //
        //   Φ_new[i, :] = Φ_raw(η_i) − parametric_anchor[i, :] · M
        //
        // so ℓ depends on the row through both the raw basis evaluation
        // and the row-specific subtraction. The basis derivative is
        // unaffected: the subtraction is independent of η.
        //
        // Evaluating `link_dev.design()` on a single-row `eta0` vector
        // would discard the row-specific subtraction (`design()`
        // asserts that the runtime has no anchor residual exactly
        // to prevent this silent miscompute). Instead, feed the
        // full-length per-row `a_pre_scale_vec` through
        // `design_at_training_with_residual` so the runtime applies the
        // cached training-row parametric anchor matrix at the correct
        // row for every evaluation. For runtimes without an
        // anchor_residual the same call falls back to raw `design()`.
        let (l_val_vec, l_d1_vec) = match (&self.link_dev, beta_w) {
            (Some(runtime), Some(beta)) => {
                let basis = runtime.design_at_training_with_residual(&a_pre_scale_vec)?;
                let d1 = runtime.first_derivative_design(&a_pre_scale_vec)?;
                (&a_pre_scale_vec + &basis.dot(beta), d1.dot(beta) + 1.0)
            }
            _ => (a_pre_scale_vec.clone(), Array1::ones(n)),
        };

        let seeds: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|row| {
                let a = a_pre_scale_vec[row];
                let ell1 = l_d1_vec[row];
                if ell1 > BMS_DERIV_TOL {
                    let ell0 = l_val_vec[row] - ell1 * a;
                    let observed_logslope = probit_scale * ell1 * slope_eta[row];
                    (marginals[row].q * (1.0 + observed_logslope * observed_logslope).sqrt()
                        / probit_scale
                        - ell0)
                        / ell1
                } else {
                    a
                }
            })
            .collect();
        // Resolve β_h once for the preseed sweep so each row's tag includes
        // the joint β that the FLEX intercept root actually depends on.
        let beta_h = self.score_beta(block_states)?;
        let mut preseeded = 0usize;
        let mut kept_warm = 0usize;
        for (row, seed) in seeds.iter().enumerate() {
            if !seed.is_finite() {
                continue;
            }
            let beta_tag = hash_intercept_warm_start_key_flex(
                marginal_eta[row],
                slope_eta[row],
                beta_h,
                beta_w,
            );
            match cache.compare_exchange_unseeded(row, *seed, beta_tag) {
                Ok(()) => preseeded += 1,
                Err(prev_tag) => {
                    if prev_tag == beta_tag {
                        // A prior write at the same β already published a
                        // value for this row; the cached intercept is reused
                        // verbatim by the subsequent root solve.
                        kept_warm += 1;
                    }
                }
            }
        }
        log::info!(
            "[bernoulli intercept warm-start] preseeded={} (cold), kept_warm={} (carried over from previous PIRLS)",
            preseeded,
            kept_warm,
        );
        Ok(())
    }

    /// Row-subset variant of [`preseed_intercept_warm_starts`]: seeds only the
    /// entries in `rows`, building intermediate vectors over all `n` training
    /// rows only where the link-deviation runtime requires full-length input
    /// (so correctness is identical to the full-`n` path for those rows).
    ///
    /// Used when `build_exact_eval_cache_with_options_and_context_rows` is
    /// called with a non-`None` `context_rows` slice so that the warm-start
    /// preseed does not pay O(n) work for a subsampled cache build.
    pub(super) fn preseed_intercept_warm_starts_for_rows(
        &self,
        block_states: &[ParameterBlockState],
        rows: &[usize],
    ) -> Result<(), String> {
        if !self.effective_flex_active(block_states)? {
            return Ok(());
        }
        let Some(cache) = self.intercept_warm_starts.as_ref() else {
            return Ok(());
        };
        let beta_w = self.link_beta(block_states)?;
        let n = self.y.len();
        if cache.len() != n {
            return Ok(());
        }
        let marginal_eta = &block_states[0].eta;
        let slope_eta = &block_states[1].eta;
        let probit_scale = self.probit_frailty_scale();

        // Per-row marginal link map — computed only for the selected rows.
        let marginals_for_rows: Vec<(usize, BernoulliMarginalLinkMap)> = rows
            .iter()
            .copied()
            .filter(|&row| row < n)
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|row| {
                let m = self.marginal_link_map(marginal_eta[row])?;
                Ok((row, m))
            })
            .collect::<Result<Vec<_>, String>>()?;

        // Pre-scale intercept for selected rows.  We still need a full-length
        // array for the link-deviation design call (the runtime's anchor
        // residual is indexed by training-row position).  Fill non-selected
        // positions with NaN — they are never read by the seed computation.
        let mut a_pre_scale_vec: Array1<f64> = Array1::from_elem(n, f64::NAN);
        for &(row, ref m) in &marginals_for_rows {
            a_pre_scale_vec[row] =
                rigid_intercept_from_marginal(m.q, slope_eta[row], probit_scale) / probit_scale;
        }

        // Batched link-deviation evaluation — must pass the full-length vector
        // so the runtime's per-row anchor residual is applied at the correct
        // positions.  NaN entries at non-selected rows propagate safely: we
        // never read those positions below.
        let (l_val_vec, l_d1_vec) = match (&self.link_dev, beta_w) {
            (Some(runtime), Some(beta)) => {
                let basis = runtime.design_at_training_with_residual(&a_pre_scale_vec)?;
                let d1 = runtime.first_derivative_design(&a_pre_scale_vec)?;
                (&a_pre_scale_vec + &basis.dot(beta), d1.dot(beta) + 1.0)
            }
            _ => (a_pre_scale_vec.clone(), Array1::ones(n)),
        };

        // Compute seeds and seed the cache only for the selected rows.
        let seeds: Vec<(usize, f64)> = marginals_for_rows
            .par_iter()
            .map(|&(row, ref m)| {
                let a = a_pre_scale_vec[row];
                let ell1 = l_d1_vec[row];
                let seed = if ell1 > BMS_DERIV_TOL {
                    let ell0 = l_val_vec[row] - ell1 * a;
                    let observed_logslope = probit_scale * ell1 * slope_eta[row];
                    (m.q * (1.0 + observed_logslope * observed_logslope).sqrt() / probit_scale
                        - ell0)
                        / ell1
                } else {
                    a
                };
                (row, seed)
            })
            .collect();

        let beta_h = self.score_beta(block_states)?;
        let mut preseeded = 0usize;
        let mut kept_warm = 0usize;
        for (row, seed) in seeds {
            if !seed.is_finite() {
                continue;
            }
            let beta_tag = hash_intercept_warm_start_key_flex(
                marginal_eta[row],
                slope_eta[row],
                beta_h,
                beta_w,
            );
            match cache.compare_exchange_unseeded(row, seed, beta_tag) {
                Ok(()) => preseeded += 1,
                Err(prev_tag) => {
                    if prev_tag == beta_tag {
                        kept_warm += 1;
                    }
                }
            }
        }
        log::info!(
            "[bernoulli intercept warm-start rows={}] preseeded={} (cold), kept_warm={} (carried over from previous PIRLS)",
            rows.len(),
            preseeded,
            kept_warm,
        );
        Ok(())
    }

    #[inline]
    pub(super) fn row_intercept_newton_is_converged(
        a: f64,
        f: f64,
        f_a: f64,
        abs_tol: f64,
    ) -> bool {
        if !a.is_finite() || !f.is_finite() || !f_a.is_finite() || f_a == 0.0 {
            return false;
        }
        let correction = (f / f_a).abs();
        f.abs() <= abs_tol || correction <= 1e-10 * (1.0 + a.abs())
    }
}

#[derive(Default)]
pub(super) struct BernoulliInterceptSolveStats {
    pub(super) cached_short_circuit: AtomicUsize,
    pub(super) closed_form_short_circuit: AtomicUsize,
    pub(super) full_solver: AtomicUsize,
    pub(super) seed_residual_le_1e12: AtomicUsize,
    pub(super) seed_residual_le_1e10: AtomicUsize,
    pub(super) seed_residual_le_1e8: AtomicUsize,
    pub(super) seed_residual_le_abs_tol: AtomicUsize,
    pub(super) seed_residual_gt_abs_tol: AtomicUsize,
    pub(super) max_full_solver_iters: AtomicUsize,
}

impl BernoulliInterceptSolveStats {
    pub(super) fn record_seed_residual(&self, residual: f64, abs_tol: f64) {
        let abs = residual.abs();
        if abs <= 1e-12 {
            self.seed_residual_le_1e12.fetch_add(1, Ordering::Relaxed);
        } else if abs <= 1e-10 {
            self.seed_residual_le_1e10.fetch_add(1, Ordering::Relaxed);
        } else if abs <= 1e-8 {
            self.seed_residual_le_1e8.fetch_add(1, Ordering::Relaxed);
        } else if abs <= abs_tol {
            self.seed_residual_le_abs_tol
                .fetch_add(1, Ordering::Relaxed);
        } else {
            self.seed_residual_gt_abs_tol
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(super) fn record_full_solver(&self, refine_iters: usize) {
        self.full_solver.fetch_add(1, Ordering::Relaxed);
        let mut current = self.max_full_solver_iters.load(Ordering::Relaxed);
        while refine_iters > current {
            match self.max_full_solver_iters.compare_exchange_weak(
                current,
                refine_iters,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(next) => current = next,
            }
        }
    }
}

impl BernoulliMarginalSlopeFamily {
    pub(super) fn intercept_primary_point(
        q: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Vec<f64> {
        let mut point = Vec::with_capacity(
            2 + beta_h.map(|beta| beta.len()).unwrap_or(0)
                + beta_w.map(|beta| beta.len()).unwrap_or(0),
        );
        point.push(q);
        point.push(b);
        if let Some(beta) = beta_h {
            point.extend(beta.iter().copied());
        }
        if let Some(beta) = beta_w {
            point.extend(beta.iter().copied());
        }
        point
    }

    #[inline]
    pub(super) fn cache_row_intercept(
        &self,
        row: usize,
        a: f64,
        marginal_eta: f64,
        slope: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) {
        if let Some(cache) = self.intercept_warm_starts.as_ref() {
            let beta_tag = hash_intercept_warm_start_key_flex(marginal_eta, slope, beta_h, beta_w);
            cache.store_tagged(row, a, beta_tag);
        }
    }

    pub(super) fn cache_row_intercept_predictor(
        &self,
        row: usize,
        a: f64,
        q: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        a_u: &Array1<f64>,
    ) {
        let Some(cache) = self.intercept_warm_starts.as_ref() else {
            return;
        };
        let primary_point = Self::intercept_primary_point(q, b, beta_h, beta_w);
        if primary_point.len() != a_u.len() {
            return;
        }
        cache.store_predictor(row, a, primary_point, a_u.iter().copied().collect());
    }

    #[inline]
    pub(super) fn beta_linf(beta: Option<&Array1<f64>>) -> f64 {
        beta.map(|b| b.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs())))
            .unwrap_or(0.0)
    }

    pub(super) fn near_zero_deviation_residual_bound(
        &self,
        slope: f64,
        beta_h_linf: f64,
        beta_w_linf: f64,
    ) -> f64 {
        let score_basis_sup = self
            .score_warp
            .as_ref()
            .map(|runtime| runtime.value_basis_l1_sup_norm())
            .unwrap_or(0.0);
        let link_basis_sup = self
            .link_dev
            .as_ref()
            .map(|runtime| runtime.value_basis_l1_sup_norm())
            .unwrap_or(0.0);
        // At the rigid intercept, deviations perturb the probit argument by at
        // most `s * (|b|·||h||∞ + ||w||∞)`.  Since `Φ` is globally
        // `φ(0)`-Lipschitz, the calibration residual changes by no more than
        // `φ(0)` times this argument bound after integrating against the unit
        // normal density.  The L1 basis sup-norms give
        // `||h||∞ <= K_h ||β_h||∞` and `||w||∞ <= K_w ||β_w||∞`; if this is
        // below the solver's `abs_tol`, the rigid root is already acceptable.
        normal_pdf(0.0)
            * self.probit_frailty_scale()
            * (slope.abs() * score_basis_sup * beta_h_linf + link_basis_sup * beta_w_linf)
    }

    pub(super) fn solve_row_intercept_base(
        &self,
        row: usize,
        marginal_eta: f64,
        slope: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        stats: Option<&BernoulliInterceptSolveStats>,
    ) -> Result<(f64, f64, bool), String> {
        let marginal = self.marginal_link_map(marginal_eta)?;
        let probit_scale = self.probit_frailty_scale();
        let target = marginal.mu;
        let abs_tol = 1e-8_f64.max(1e-4 * target.abs());
        let rigid_a = rigid_prescale_intercept_from_marginal(marginal.q, slope, probit_scale);
        let rigid_abs_deriv =
            rigid_prescale_intercept_derivative_abs(marginal.q, slope, probit_scale);

        let beta_h_linf = Self::beta_linf(beta_h);
        let beta_w_linf = Self::beta_linf(beta_w);
        let exact_zero_deviation = beta_h_linf == 0.0 && beta_w_linf == 0.0;
        let standard_normal_law = matches!(self.latent_measure, LatentMeasureKind::StandardNormal);
        if exact_zero_deviation && standard_normal_law {
            self.cache_row_intercept(row, rigid_a, marginal_eta, slope, beta_h, beta_w);
            return Ok((rigid_a, rigid_abs_deriv, true));
        }

        let near_zero_bound =
            self.near_zero_deviation_residual_bound(slope, beta_h_linf, beta_w_linf);
        let beta_linf_max = beta_h_linf.max(beta_w_linf);
        if standard_normal_law && near_zero_bound <= abs_tol && beta_linf_max <= f64::EPSILON.sqrt()
        {
            // Numerical guardrail for the conservative perturbation bound: the
            // exact-zero path above avoids all cell machinery, while this
            // near-zero path spends one evaluator call to guarantee that every
            // accepted row satisfies the same residual contract as the solver.
            // The extra `sqrt(eps)` coefficient cap keeps numerical
            // derivative probes out of this value-only acceptance path;
            // mathematically nonzero deviations still fall through unless they
            // are too small to carry stable derivative information.
            let (f_rigid, _, _) = self.evaluate_calibration_newton(
                row,
                rigid_a,
                marginal_eta,
                slope,
                beta_h,
                beta_w,
            )?;
            if f_rigid.abs() <= abs_tol {
                self.cache_row_intercept(row, rigid_a, marginal_eta, slope, beta_h, beta_w);
                return Ok((rigid_a, rigid_abs_deriv, true));
            }
        }

        // Use the Newton-only calibration evaluator: `solve_monotone_root`
        // safely degrades its Halley step to Newton when `F''(a) = 0`, and
        // dropping the second derivative lets us skip order-9 value-bearing
        // cell moments in favour of degree-4 moments.
        let eval = |a: f64| -> Result<(f64, f64, f64), String> {
            self.evaluate_calibration_newton(row, a, marginal_eta, slope, beta_h, beta_w)
        };

        // Closed-form fallback initial guess: rigid probit in pre-scale
        // denested coordinates:
        //   a₀ = q·√(1 + (s_f b)²) / s_f,  s_f = 1/√(1+σ²).
        // When link deviation is active, upgrade to affine-link warm start:
        //   s_f·L(u) ≈ s_f·(ℓ₀ + ℓ₁·u)
        //   ⟹  a = (q·√(1 + (s_f ℓ₁ b)²) / s_f − ℓ₀) / ℓ₁
        let a_closed_form = self.row_intercept_closed_form_seed(row, marginal, slope, beta_w)?;

        // Prefer the previous PIRLS iter's converged intercept as the
        // initial guess; β changes only a little between consecutive PIRLS
        // iterations, so the previous answer is typically within a few
        // root-solver steps of the new one. If the cache slot is NaN
        // (uninitialised) or non-finite (stale), fall back to the closed-
        // form seed.
        let current_primary_point =
            Self::intercept_primary_point(marginal_eta, slope, beta_h, beta_w);
        let predictor_a = self
            .intercept_warm_starts
            .as_ref()
            .and_then(|cache| cache.predictor_seed(row, &current_primary_point));
        // FLEX cache slot must include β_h and β_w: under link-deviation and
        // score-warp the root depends on the joint coefficient vector, not
        // just `(marginal_eta, slope)`. Without the tag a TR trial at one β
        // can read back a converged value from a different trial at the same
        // row and poison the solve.
        let flex_beta_tag = hash_intercept_warm_start_key_flex(marginal_eta, slope, beta_h, beta_w);
        let cached_a = self
            .intercept_warm_starts
            .as_ref()
            .and_then(|cache| cache.load_tagged(row, flex_beta_tag));
        let a_init = predictor_a.or(cached_a).unwrap_or(a_closed_form);

        // Note: an explicit `eval(a_closed_form)` short-circuit at this point
        // would be redundant. On cold cycle-0 `cached_a` is None, so `a_init`
        // already equals `a_closed_form` and the two-step Newton probe below
        // evaluates there with the 1e-10 tolerance from
        // `row_intercept_newton_is_converged`, matching the exact-root path
        // in `monotone_root::solve_monotone_root` (see monotone_root.rs:50-66).
        // On warm cycles, evaluating at `a_closed_form` would add an extra
        // cell-moment call even when the cached seed is already the root.

        // Adaptive acceptance tolerance: for extreme slopes the intercept
        // equation becomes numerically flat and tight absolute precision is
        // not achievable. We accept any bracketed solution at this level, so
        // pass the same tolerance to the root solver — driving it tighter
        // than `abs_tol` is wasted cell-moment work, since at biobank scale
        // (n=320k, FLEX active with linkwiggle + score-warp) the solver is
        // called once per row per Hessian build and the per-row cell-moment
        // kernel dominates wall time. With this tolerance the closed-form /
        // affine warm start short-circuits at `monotone_root.rs:26` for the
        // common case, instead of forcing 30+ refinement iters down to 1e-10.

        // Local Newton probe before paying for the safeguarded bracket.
        // Cycle-0 is cold at biobank scale, so forcing every row through the
        // bracket spends most of the wall time rebuilding identical cell
        // value integrals. The rigid/affine seed is exact when deviations vanish
        // and first-order accurate when they are small; probe that local
        // Newton basin first. The convergence test uses the same residual
        // contract as the safeguarded solver plus a tight relative-correction
        // gate, so any accept here satisfies the existing final check. Hard
        // cases still fall through unchanged.
        let probe_result = (|| -> Result<(Option<(f64, f64, f64)>, f64), String> {
            let mut a = a_init;
            let mut seed_residual = None;
            for _ in 0..6 {
                let (f, f_a, _) = eval(a)?;
                if seed_residual.is_none() {
                    seed_residual = Some(f);
                }
                if Self::row_intercept_newton_is_converged(a, f, f_a, abs_tol) {
                    return Ok((Some((a, f_a.abs(), f)), seed_residual.unwrap_or(f)));
                }
                if !(f_a.is_finite() && f_a != 0.0) {
                    break;
                }
                let next_a = a - f / f_a;
                if !next_a.is_finite() {
                    break;
                }
                a = next_a;
            }
            Ok((None, seed_residual.unwrap_or(f64::INFINITY)))
        })();

        if let Ok((accepted, seed_residual)) = &probe_result {
            if let Some(stats) = stats {
                stats.record_seed_residual(*seed_residual, abs_tol);
            }
            if let Some((a, abs_deriv, _)) = accepted {
                if let Some(stats) = stats {
                    if predictor_a.is_some() || cached_a.is_some() {
                        stats.cached_short_circuit.fetch_add(1, Ordering::Relaxed);
                    } else {
                        stats
                            .closed_form_short_circuit
                            .fetch_add(1, Ordering::Relaxed);
                    }
                }
                self.cache_row_intercept(row, *a, marginal_eta, slope, beta_h, beta_w);
                return Ok((*a, *abs_deriv, false));
            }
        }

        let mut solve_result = crate::families::monotone_root::solve_monotone_root_detailed(
            eval,
            a_init,
            "bernoulli intercept",
            abs_tol,
            64,
            48,
        );

        // If the warm-started solve failed, retry once from the closed-form
        // seed. Cached `a` from a prior PIRLS iter can be far enough from
        // the current root (e.g., after a large β step) that the bracketing
        // search exhausts; the closed-form seed always sits in the correct
        // basin.
        if (predictor_a.is_some() || cached_a.is_some()) && solve_result.is_err() {
            solve_result = crate::families::monotone_root::solve_monotone_root_detailed(
                eval,
                a_closed_form,
                "bernoulli intercept",
                abs_tol,
                64,
                48,
            );
        }
        // Routine emits its own format!()-based String errors below
        // (residual rejection); enclosing return type stays Result<_, String>.
        let solve_solution = solve_result.map_err(|e| e.to_string())?;
        if let Some(stats) = stats {
            stats.record_full_solver(solve_solution.refine_iters);
        }
        let (a, abs_deriv, f_best) = (
            solve_solution.root,
            solve_solution.abs_deriv,
            solve_solution.residual,
        );

        if f_best.abs() > abs_tol {
            return Err(format!(
                "bernoulli marginal-slope intercept solve failed: \
                     residual={f_best:.3e} at a={a:.6}, target mu={target:.6}"
            ));
        }

        // Cache the converged intercept for the next PIRLS iter.
        self.cache_row_intercept(row, a, marginal_eta, slope, beta_h, beta_w);

        Ok((a, abs_deriv, false))
    }
    pub(super) fn build_row_exact_context_with_stats_and_cell_cache(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        stats: Option<&BernoulliInterceptSolveStats>,
        cache_degree9_cells: bool,
    ) -> Result<BernoulliMarginalSlopeRowExactContext, String> {
        let marginal_eta = block_states[0].eta[row];
        let marginal = self.marginal_link_map(marginal_eta)?;
        // The log-slope block now parameterizes the signed slope directly.
        let slope = block_states[1].eta[row];
        let beta_h = self.score_beta(block_states)?;
        let beta_w = self.link_beta(block_states)?;
        let (intercept, m_a, intercept_fast_path) = if self.effective_flex_active(block_states)? {
            self.solve_row_intercept_base(row, marginal_eta, slope, beta_h, beta_w, stats)?
        } else {
            let intercept = match self.latent_measure.empirical_grid_for_training_row(row)? {
                None => {
                    rigid_intercept_from_marginal(marginal.q, slope, self.probit_frailty_scale())
                }
                Some(grid) => self.empirical_rigid_intercept_for_row(
                    row,
                    marginal,
                    slope,
                    &grid.nodes,
                    &grid.weights,
                )?,
            };
            (intercept, f64::NAN, false)
        };
        // Cache degree-9 cell moments at the converged intercept so the
        // many gradient/diagonal/matvec passes that run *after* this point
        // for the same (row, β) don't re-evaluate `evaluate_cell_moments` /
        // `bivariate_normal_cdf` on identical inputs. This matters for the
        // FLEX path (linkwiggle + score-warp), where each per-row Hessian
        // build runs the cell-moment kernel once per cell per closure call.
        let degree9_cells = if cache_degree9_cells
            && self.effective_flex_active(block_states)?
            && matches!(self.latent_measure, LatentMeasureKind::StandardNormal)
        {
            let cells = self.denested_partition_cells(intercept, slope, beta_h, beta_w)?;
            // Per-row dedup: within ONE row's denested-partition output, the
            // score-warp and link-wiggle bases occasionally produce cells
            // whose `(left, right, c0, c1, c2, c3)` are bit-equal. Evaluating
            // moments once and cloning the result into the other slots is
            // numerically identical to evaluating each cell independently
            // (`evaluate_cell_moments_lru` is a pure function of the cell), and
            // skips redundant work. The dedup is purely intra-row, so it is
            // orthogonal to the per-family LRU (which is keyed across rows)
            // and the affine tail-cell memo (a separate mechanism).
            let mut dedup: HashMap<
                exact_kernel::CellFingerprint,
                exact_kernel::CellDerivativeMomentState,
            > = HashMap::new();
            let mut out: Vec<CachedDenestedCellMoments> = Vec::with_capacity(cells.len());
            for partition_cell in cells.into_iter() {
                let key = exact_kernel::CellFingerprint::new(partition_cell.cell);
                let state: exact_kernel::CellDerivativeMomentState =
                    if let Some(existing) = dedup.get(&key) {
                        existing.clone()
                    } else {
                        let computed =
                            self.evaluate_cell_derivative_moments_lru(partition_cell.cell, 9)?;
                        dedup.insert(key, computed.clone());
                        computed
                    };
                out.push(CachedDenestedCellMoments {
                    partition_cell,
                    state,
                });
            }
            Some(out)
        } else {
            None
        };
        Ok(BernoulliMarginalSlopeRowExactContext {
            intercept,
            m_a,
            intercept_fast_path,
            degree9_cells,
        })
    }

    /// Look up the pre-solved row context from the cache.
    #[inline]
    pub(super) fn row_ctx(
        cache: &BernoulliMarginalSlopeExactEvalCache,
        row: usize,
    ) -> &BernoulliMarginalSlopeRowExactContext {
        &cache.row_contexts[row]
    }

    pub(super) fn build_exact_eval_cache_with_order(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<BernoulliMarginalSlopeExactEvalCache, String> {
        self.build_exact_eval_cache_with_options(block_states, None)
    }

    pub(super) fn build_exact_eval_cache_with_options(
        &self,
        block_states: &[ParameterBlockState],
        options: Option<&BlockwiseFitOptions>,
    ) -> Result<BernoulliMarginalSlopeExactEvalCache, String> {
        self.build_exact_eval_cache_with_options_and_context_rows(block_states, options, None)
    }

    pub(super) fn build_exact_eval_cache_for_selected_context_rows(
        &self,
        block_states: &[ParameterBlockState],
        options: &BlockwiseFitOptions,
        context_rows: &[usize],
    ) -> Result<BernoulliMarginalSlopeExactEvalCache, String> {
        self.build_exact_eval_cache_with_options_and_context_rows(
            block_states,
            Some(options),
            Some(context_rows),
        )
    }

    pub(super) fn build_exact_eval_cache_with_options_and_context_rows(
        &self,
        block_states: &[ParameterBlockState],
        options: Option<&BlockwiseFitOptions>,
        context_rows: Option<&[usize]>,
    ) -> Result<BernoulliMarginalSlopeExactEvalCache, String> {
        self.validate_exact_block_state_shapes(block_states)?;
        let slices = block_slices(self);
        let primary = primary_slices(&slices);
        let n = self.y.len();
        let flex_active = self.effective_flex_active(block_states)?;
        let selected_context_rows = context_rows.map(|rows| {
            let mut selected = rows
                .iter()
                .copied()
                .filter(|&row| row < n)
                .collect::<Vec<_>>();
            selected.sort_unstable();
            selected.dedup();
            selected
        });
        let context_row_count = selected_context_rows.as_ref().map_or(n, |rows| rows.len());
        let started = std::time::Instant::now();
        let heartbeat_guard = crate::heartbeat::scope(format!(
            "BMS exact-cache build n={n} context_rows={context_row_count} p={} flex={flex_active}",
            slices.total
        ));
        if log_exact_work(n) {
            log::info!(
                "[BMS exact-cache] build start n={} context_rows={} p={} flex={}",
                n,
                context_row_count,
                slices.total,
                flex_active
            );
        }
        let preseed_started = std::time::Instant::now();
        if let Some(rows) = selected_context_rows.as_deref() {
            self.preseed_intercept_warm_starts_for_rows(block_states, rows)?;
        } else {
            self.preseed_intercept_warm_starts(block_states)?;
        }
        if log_exact_work(n) {
            log::info!(
                "[BMS exact-cache] preseed done n={} context_rows={} elapsed={:.3}s",
                n,
                context_row_count,
                preseed_started.elapsed().as_secs_f64()
            );
        }
        if flex_active {
            exact_kernel::reset_tail_cell_moment_cache();
        }
        let stats = BernoulliInterceptSolveStats::default();
        let cell_cache_before = self.cell_moment_cache_stats.snapshot();
        // Suppress per-row `degree9_cells` caching during the parallel context
        // build: when flex is active *and* the latent measure is StandardNormal
        // (i.e. exactly when `degree9_cells` would be populated), the top-of-
        // cycle `build_row_cell_moments_bundle` invocation below also calls
        // `denested_partition_cells` for every row. Suppressing the per-row
        // cache here avoids the duplicate partition computation and the
        // unused degree-9 moment evaluations whenever the bundle succeeds.
        // When the bundle returns `None` (budget exceeded), the per-row
        // `degree9_cells` cache is reconstructed below so the row-evaluation
        // fast path that consults `row_ctx.degree9_cells` still has its
        // cache. Numerical results are unchanged either way.
        let context_started = std::time::Instant::now();
        let progress_step = (context_row_count / 10).max(1);
        let completed_rows = AtomicUsize::new(0);
        let row_contexts = if let Some(selected_rows) = selected_context_rows.as_ref() {
            let computed = selected_rows
                .par_iter()
                .copied()
                .map(|row| {
                    let ctx = self.build_row_exact_context_with_stats_and_cell_cache(
                        row,
                        block_states,
                        Some(&stats),
                        false,
                    )?;
                    if log_exact_work(n) {
                        let done = completed_rows.fetch_add(1, Ordering::Relaxed) + 1;
                        if done == context_row_count || done % progress_step == 0 {
                            log::info!(
                                "[BMS exact-cache] row-context progress rows={}/{} elapsed={:.3}s",
                                done,
                                context_row_count,
                                context_started.elapsed().as_secs_f64()
                            );
                        }
                    }
                    Ok((row, ctx))
                })
                .collect::<Result<Vec<_>, String>>()?;
            let mut row_contexts = vec![
                BernoulliMarginalSlopeRowExactContext {
                    intercept: f64::NAN,
                    m_a: f64::NAN,
                    intercept_fast_path: false,
                    degree9_cells: None,
                };
                n
            ];
            for (row, ctx) in computed {
                row_contexts[row] = ctx;
            }
            row_contexts
        } else {
            (0..n)
                .into_par_iter()
                .map(|row| {
                    let ctx = self.build_row_exact_context_with_stats_and_cell_cache(
                        row,
                        block_states,
                        Some(&stats),
                        false,
                    )?;
                    if log_exact_work(n) {
                        let done = completed_rows.fetch_add(1, Ordering::Relaxed) + 1;
                        if done == context_row_count || done % progress_step == 0 {
                            log::info!(
                                "[BMS exact-cache] row-context progress rows={}/{} elapsed={:.3}s",
                                done,
                                context_row_count,
                                context_started.elapsed().as_secs_f64()
                            );
                        }
                    }
                    Ok(ctx)
                })
                .collect::<Result<Vec<_>, String>>()?
        };
        let fast_path_rows = row_contexts
            .iter()
            .filter(|ctx| ctx.intercept_fast_path)
            .count();
        if log_exact_work(n) {
            log::info!(
                "[BMS exact-cache] row-context done rows={} fast_path_rows={} elapsed={:.3}s",
                context_row_count,
                fast_path_rows,
                context_started.elapsed().as_secs_f64()
            );
        } else {
            log::debug!(
                "[BMS exact-cache] row-intercept zero-deviation fast path rows={}/{}",
                fast_path_rows,
                n
            );
        }
        if flex_active {
            log::info!(
                "bernoulli marginal-slope intercept seed short-circuit: cached={}, closed_form={}, full_solver={}, max_full_solver_iters={}, seed_residual_bins={{<=1e-12:{}, <=1e-10:{}, <=1e-8:{}, <=abs_tol:{}, >abs_tol:{}}}",
                stats.cached_short_circuit.load(Ordering::Relaxed),
                stats.closed_form_short_circuit.load(Ordering::Relaxed),
                stats.full_solver.load(Ordering::Relaxed),
                stats.max_full_solver_iters.load(Ordering::Relaxed),
                stats.seed_residual_le_1e12.load(Ordering::Relaxed),
                stats.seed_residual_le_1e10.load(Ordering::Relaxed),
                stats.seed_residual_le_1e8.load(Ordering::Relaxed),
                stats.seed_residual_le_abs_tol.load(Ordering::Relaxed),
                stats.seed_residual_gt_abs_tol.load(Ordering::Relaxed),
            );
        }
        if flex_active {
            let (cell_hits, cell_misses, cell_hit_rate) = self
                .cell_moment_cache_stats
                .hit_rate_delta(cell_cache_before);
            log::info!(
                "[BMS cell-moment LRU] cycle hits={} misses={} hit_rate={:.1}% entries={} resident_mib={:.1}/{:.1}",
                cell_hits,
                cell_misses,
                100.0 * cell_hit_rate,
                self.cell_moment_lru.len(),
                self.cell_moment_lru.resident_bytes() as f64 / (1024.0 * 1024.0),
                self.cell_moment_lru.max_bytes() as f64 / (1024.0 * 1024.0),
            );
            let tail_stats = exact_kernel::tail_cell_moment_cache_stats();
            log::info!(
                "[BMS exact-cache] affine tail-cell memo: hits={} misses={} entries={} hit_rate={:.3}%",
                tail_stats.hits,
                tail_stats.misses,
                tail_stats.entries,
                100.0 * tail_stats.hit_rate(),
            );
        }
        let row_cell_mask = options
            .and_then(|opts| opts.outer_score_subsample.as_ref())
            .map(|subsample| subsample.mask.as_slice());
        let row_cell_started = std::time::Instant::now();
        let row_cell_moments =
            self.build_row_cell_moments_bundle(block_states, &row_contexts, 9, row_cell_mask)?;
        if log_exact_work(n) {
            log::info!(
                "[BMS exact-cache] row-cell phase done n={} selected_rows={} built={} elapsed={:.3}s",
                n,
                row_cell_mask.map_or(n, <[usize]>::len),
                row_cell_moments.is_some(),
                row_cell_started.elapsed().as_secs_f64()
            );
            log::info!(
                "[BMS exact-cache] build done n={} context_rows={} p={} flex={} elapsed={:.3}s",
                n,
                context_row_count,
                slices.total,
                flex_active,
                started.elapsed().as_secs_f64()
            );
        }
        drop(heartbeat_guard);
        Ok(BernoulliMarginalSlopeExactEvalCache {
            slices,
            primary,
            row_contexts,
            row_cell_moments,
            row_cell_moments_d15: crate::resource::RayonSafeOnce::new(),
            row_cell_moments_d21: crate::resource::RayonSafeOnce::new(),
            row_primary_hessians: RowPrimaryEvalCache::Empty,
            rigid_third_full: crate::resource::RayonSafeOnce::new(),
            rigid_fourth_full: crate::resource::RayonSafeOnce::new(),
        })
    }

    /// Build a top-of-cycle [`RowCellMomentsBundle`] at the given
    /// `max_degree`. Returns `None` when the FLEX path is inactive, when an
    /// empirical latent grid is in effect (the row kernel takes a non-cell
    /// path), or when the estimated resident bytes exceed the active
    /// resource-policy budget. Numerical equivalence with the legacy per-row
    /// path is unconditional: callers always fall back to
    /// `degree9_cells`/on-demand cell evaluation when the bundle is absent.
    pub(super) fn build_row_cell_moments_bundle(
        &self,
        block_states: &[ParameterBlockState],
        row_contexts: &[BernoulliMarginalSlopeRowExactContext],
        max_degree: usize,
        row_mask: Option<&[usize]>,
    ) -> Result<Option<RowCellMomentsBundle>, String> {
        if !self.effective_flex_active(block_states)? {
            return Ok(None);
        }
        // Empirical-grid rows take a non-cell code path inside
        // `compute_row_analytic_flex_from_parts_into`, so the bundle would
        // never be consulted. Skip the build to avoid wasted work.
        if !matches!(self.latent_measure, LatentMeasureKind::StandardNormal) {
            return Ok(None);
        }
        let n = self.y.len();
        let beta_h = self.score_beta(block_states)?;
        let beta_w = self.link_beta(block_states)?;
        let selected_rows: Vec<usize> = match row_mask {
            Some(mask) => mask.iter().copied().filter(|&row| row < n).collect(),
            None => (0..n).collect(),
        };
        if selected_rows.is_empty() {
            return Ok(None);
        }
        let selected_row_count = selected_rows.len();
        let max_cells = self.max_denested_partition_cells_per_row();
        let max_n_cells = selected_row_count.saturating_mul(max_cells);
        let upper_bound_bytes =
            RowCellMomentsBundle::estimated_resident_bytes(n, max_n_cells, max_degree);
        let limit_bytes = self.policy.max_operator_cache_bytes;
        if upper_bound_bytes > limit_bytes {
            log::info!(
                "[BMS row-cell-moments] skip precompute n={} selected_rows={} max_cells_per_row={} degree={} upper_bound_bytes={} limit_bytes={}",
                n,
                selected_row_count,
                max_cells,
                max_degree,
                upper_bound_bytes,
                limit_bytes
            );
            return Ok(None);
        }
        let started = std::time::Instant::now();
        let heartbeat_guard = crate::heartbeat::scope(format!(
            "BMS row-cell-moments n={n} selected_rows={selected_row_count} degree={max_degree}"
        ));
        if log_exact_work(n) {
            log::info!(
                "[BMS row-cell-moments] partition start n={} selected_rows={} degree={}",
                n,
                selected_row_count,
                max_degree
            );
        }
        let partitions: Vec<(usize, Vec<exact_kernel::DenestedPartitionCell>)> = selected_rows
            .into_par_iter()
            .map(|row| {
                self.denested_partition_cells(
                    row_contexts[row].intercept,
                    block_states[1].eta[row],
                    beta_h,
                    beta_w,
                )
                .map(|cells| (row, cells))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let selected_n = partitions.len();
        let n_cells = partitions
            .iter()
            .map(|(_, cells)| cells.len())
            .sum::<usize>();
        if log_exact_work(n) {
            log::info!(
                "[BMS row-cell-moments] partition done n={} selected_rows={} cells={} elapsed={:.3}s",
                n,
                selected_n,
                n_cells,
                started.elapsed().as_secs_f64()
            );
        }
        let estimated_bytes =
            RowCellMomentsBundle::estimated_resident_bytes(n, n_cells, max_degree);
        if estimated_bytes > limit_bytes {
            log::warn!(
                "[BMS row-cell-moments] skip precompute n={} selected_rows={} cells={} degree={} estimated_bytes={} limit_bytes={}",
                n,
                selected_n,
                n_cells,
                max_degree,
                estimated_bytes,
                limit_bytes
            );
            return Ok(None);
        }
        let moment_started = std::time::Instant::now();
        let computed_rows = partitions
            .into_par_iter()
            .map(|(row, cells)| {
                let moments = cells
                    .into_iter()
                    .map(|partition_cell| {
                        // Use the per-family LRU here too so the bundle build
                        // benefits from cross-row cell-moment reuse and keeps
                        // the LRU's hit-rate accounting consistent.
                        self.evaluate_cell_derivative_moments_lru(partition_cell.cell, max_degree)
                            .map(|state| CachedDenestedCellMoments {
                                partition_cell,
                                state,
                            })
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                Ok((row, moments))
            })
            .collect::<Result<Vec<_>, String>>()?;
        // Block-12 Stage-1 GPU-substrate parity guard. The substrate
        // `try_build_cubic_cell_derivative_moments` is the future per-row
        // moment producer (host path today, NVRTC kernel on V100 later);
        // landing the call here makes it a real production consumer of the
        // substrate's pub(crate) entry point and surfaces any divergence
        // from the existing LRU evaluator the moment it appears. We sample
        // a small prefix of rows so the debug-build cost stays bounded for
        // biobank-scale fits; the production build pays nothing because the
        // block is `cfg(debug_assertions)`-gated.
        #[cfg(debug_assertions)]
        {
            use crate::gpu::cubic_cell::{
                CubicCellDerivativeMomentHostView, CubicCellMomentResidency, GpuCellBranchTag,
                GpuDenestedCubicCell, try_build_cubic_cell_derivative_moments,
            };
            const PARITY_ROW_BUDGET: usize = 4;
            let mut sample_cells: Vec<GpuDenestedCubicCell> = Vec::new();
            let mut sample_branches: Vec<GpuCellBranchTag> = Vec::new();
            let mut sample_cpu_moments: Vec<Vec<f64>> = Vec::new();
            for (_, moments) in computed_rows.iter().take(PARITY_ROW_BUDGET) {
                for cached in moments {
                    let cell = cached.partition_cell.cell;
                    let branch = if !cell.left.is_finite() || !cell.right.is_finite() {
                        GpuCellBranchTag::AffineTail
                    } else if cell.c2 == 0.0 && cell.c3 == 0.0 {
                        GpuCellBranchTag::Affine
                    } else {
                        GpuCellBranchTag::NonAffineFinite
                    };
                    sample_cells.push(GpuDenestedCubicCell {
                        left: cell.left,
                        right: cell.right,
                        c0: cell.c0,
                        c1: cell.c1,
                        c2: cell.c2,
                        c3: cell.c3,
                    });
                    sample_branches.push(branch);
                    sample_cpu_moments.push(cached.state.moments.to_vec());
                }
            }
            if !sample_cells.is_empty() {
                let view = CubicCellDerivativeMomentHostView {
                    cells: &sample_cells,
                    branches: &sample_branches,
                    max_degree,
                    residency: CubicCellMomentResidency::Host,
                };
                match try_build_cubic_cell_derivative_moments(view) {
                    Ok(Some(output)) => {
                        use crate::gpu::cubic_cell::{
                            CubicCellDerivativeMomentOutput, CubicCellMomentStatus,
                        };
                        let (sub_moments, sub_status, stride) = match output {
                            CubicCellDerivativeMomentOutput::Host {
                                moments,
                                status,
                                stride,
                            } => (moments, status, stride),
                            #[cfg(target_os = "linux")]
                            CubicCellDerivativeMomentOutput::Device { .. } => {
                                // The view above explicitly requested
                                // `CubicCellMomentResidency::Host`, and the substrate's
                                // contract (`try_build_cubic_cell_derivative_moments` in
                                // `src/gpu/cubic_cell/mod.rs:170`) guarantees that a Host
                                // request returns `Host(...)` even on Linux+CUDA. Reaching
                                // this arm means the substrate's contract was violated —
                                // a hard programming error in the GPU dispatcher, not a
                                // runtime condition we can recover from. Panicking
                                // surfaces it at the call site.
                                // SAFETY: unreachable by substrate contract — Host
                                // request must return Host residency; reaching this
                                // arm is a programming error, not a runtime condition.
                                panic!(
                                    "BMS row-cell-moments parity probe requested Host residency \
                                     but substrate returned device-resident output"
                                )
                            }
                        };
                        assert_eq!(stride, max_degree + 1);
                        assert_eq!(sub_status.len(), sample_cells.len());
                        for (i, cpu_row) in sample_cpu_moments.iter().enumerate() {
                            assert_eq!(
                                sub_status[i],
                                CubicCellMomentStatus::Ok as u8,
                                "BMS row-cell-moments parity: substrate refused cell {i} (status={})",
                                sub_status[i]
                            );
                            let sub_row = &sub_moments[i * stride..(i + 1) * stride];
                            let copy_len = cpu_row.len().min(stride);
                            for k in 0..copy_len {
                                let want = cpu_row[k];
                                let got = sub_row[k];
                                let denom = want.abs().max(1.0);
                                let rel = (got - want).abs() / denom;
                                let abs = (got - want).abs();
                                assert!(
                                    abs <= 1e-12 || rel <= 1e-11,
                                    "BMS row-cell-moments parity drift cell={i} k={k} \
                                     cpu={want:.17e} substrate={got:.17e} abs={abs:.3e} rel={rel:.3e}"
                                );
                            }
                        }
                    }
                    Ok(None) => {
                        // SAFETY: substrate's `Ok(None)` contract is
                        // reserved for empty input; the surrounding
                        // `if !sample_cells.is_empty()` guards against
                        // that. A `None` return for a populated sample
                        // is a contract violation that must be visible
                        // at the first fit in debug builds, not silently
                        // tolerated.
                        panic!(
                            "BMS row-cell-moments parity: substrate returned Ok(None) for a non-empty sample of {} cells",
                            sample_cells.len()
                        );
                    }
                    // SAFETY: substrate errors during the parity sample
                    // mean the host evaluator (which we are checking
                    // against the LRU path) disagreed on cells the LRU
                    // already accepted. Continuing past such a divergence
                    // hides correctness bugs the parity guard is here to
                    // catch; abort the debug-build fit.
                    Err(err) => panic!(
                        "BMS row-cell-moments parity: substrate failed on {} sample cells: {}",
                        sample_cells.len(),
                        err
                    ),
                }
            }
        }
        let mut rows = vec![None; n];
        for (row, moments) in computed_rows {
            rows[row] = Some(moments);
        }
        if log_exact_work(n) {
            log::info!(
                "[BMS row-cell-moments] precomputed n={} selected_rows={} cells={} degree={} estimated_bytes={} elapsed={:.3}s",
                n,
                selected_n,
                n_cells,
                max_degree,
                estimated_bytes,
                moment_started.elapsed().as_secs_f64()
            );
        }
        drop(heartbeat_guard);
        Ok(Some(RowCellMomentsBundle { max_degree, rows }))
    }

    /// BMS-FLEX GPU milestone 1: pack the row-primary Hessian inputs for the
    /// Stage-2 device kernel in `crate::gpu::bms_flex_row`. Returns `None`
    /// when any precondition fails (latent is not StandardNormal, the
    /// row-cell-moments bundle was not materialised, or score-warp /
    /// link-deviation runtimes are missing); the caller then falls back to
    /// the CPU rayon loop.
    ///
    /// The packed bundle mirrors `compute_row_analytic_flex_from_parts_into`
    /// (`StandardNormal` branch at lines 9047–9314) field-for-field. The
    /// per-cell coefficient families are built here on the host (cheap
    /// scalar work) so the device kernel reads only flat SoA buffers and
    /// keeps its inner loop free of cubic-cell partial-derivative math.
    pub(super) fn pack_bms_flex_row_kernel_inputs(
        &self,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<Option<crate::gpu::bms_flex_row::BmsFlexRowKernelInputsOwned>, String> {
        use super::exact_kernel as exact;
        use crate::families::marginal_slope_shared::SparsePrimaryCoeffJetView;

        // ── Preconditions: the Stage-2 kernel only handles the StandardNormal
        //    cell-loop branch with a pre-built row-cell-moments bundle. The
        //    empirical-grid branch and the per-row degree-9 fallback both
        //    require additional packing the kernel does not consume yet.
        if !matches!(self.latent_measure, LatentMeasureKind::StandardNormal) {
            return Ok(None);
        }
        let Some(bundle) = cache.row_cell_moments.as_ref() else {
            return Ok(None);
        };
        let primary = &cache.primary;
        let r = primary.total;
        if r < 2 || r > crate::gpu::bms_flex_row::MAX_R {
            return Ok(None);
        }
        let h_range = primary.h.clone();
        let w_range = primary.w.clone();
        let p_h = h_range.as_ref().map(|range| range.len()).unwrap_or(0);
        let p_w = w_range.as_ref().map(|range| range.len()).unwrap_or(0);
        if r != 2 + p_h + p_w {
            return Ok(None);
        }
        if p_h > 0 && self.score_warp.is_none() {
            return Ok(None);
        }
        if p_w > 0 && self.link_dev.is_none() {
            return Ok(None);
        }
        let beta_h = self.score_beta(block_states)?;
        let beta_w = self.link_beta(block_states)?;
        let scale = self.probit_frailty_scale();
        let n = self.y.len();
        let score_runtime = self.score_warp.as_ref();
        let link_runtime = self.link_dev.as_ref();

        // ── Phase-4 device-moment plan. On Linux+CUDA we skip the host
        //    `cell_moments` fill in the per-row loop and instead build the
        //    moments on the GPU via the cubic-cell substrate, attaching the
        //    resulting `CudaSlice<f64>` directly to the owned bundle so
        //    `launch_bms_flex_row_kernel` consumes it without a host
        //    upload. The host fill stays as the fallback on hosts without
        //    a runtime (and on every non-Linux build).
        #[cfg(target_os = "linux")]
        let build_device_moments = crate::gpu::runtime::GpuRuntime::global().is_some();
        #[cfg(not(target_os = "linux"))]
        let build_device_moments = false;

        // ── First pass: row offsets + total cell count. The Stage-2 kernel
        //    consumes a CSR `cell_offsets[n+1]` with `total_cells =
        //    cell_offsets[n]`; reject up front any row whose cells were
        //    not materialised at degree ≥ 9 (the kernel needs `m_0..m_9`).
        let mut cell_offsets: Vec<u32> = Vec::with_capacity(n + 1);
        cell_offsets.push(0);
        let mut total_cells: u32 = 0;
        for row in 0..n {
            let Some(row_cells) = bundle.row(row, 9) else {
                return Ok(None);
            };
            let len_u32 = u32::try_from(row_cells.len()).map_err(|_| {
                format!("bms_flex_row pack: row {row} cell count exceeds u32 range")
            })?;
            total_cells = total_cells
                .checked_add(len_u32)
                .ok_or_else(|| "bms_flex_row pack: total cell count overflows u32".to_string())?;
            cell_offsets.push(total_cells);
        }
        let total_cells_us = total_cells as usize;

        // ── Per-row scalars + observed-point pre-eval buffers.
        let mut row_q = Vec::<f64>::with_capacity(n);
        let mut row_b = Vec::<f64>::with_capacity(n);
        let mut row_mu1 = Vec::<f64>::with_capacity(n);
        let mut row_mu2 = Vec::<f64>::with_capacity(n);
        let mut row_zobs = Vec::<f64>::with_capacity(n);
        let mut row_y = Vec::<f64>::with_capacity(n);
        let mut row_w = Vec::<f64>::with_capacity(n);
        let mut row_chi = Vec::<f64>::with_capacity(n);
        let mut row_xi = Vec::<f64>::with_capacity(n);
        let mut row_rho = vec![0.0_f64; n * r];
        let mut row_tau = vec![0.0_f64; n * r];
        let mut row_ruv = vec![0.0_f64; n * r * r];

        // ── Per-cell SoA arrays sized once.
        let coeff4 = crate::gpu::bms_flex_row::COEFF4;
        let moment_stride = crate::gpu::bms_flex_row::MOMENT_STRIDE;
        let mut cell_c0 = vec![0.0_f64; total_cells_us];
        let mut cell_c1 = vec![0.0_f64; total_cells_us];
        let mut cell_c2 = vec![0.0_f64; total_cells_us];
        let mut cell_c3 = vec![0.0_f64; total_cells_us];
        let mut cell_a = vec![0.0_f64; total_cells_us * coeff4];
        let mut cell_aa = vec![0.0_f64; total_cells_us * coeff4];
        let r_minus_1 = r.saturating_sub(1);
        let mut cell_r_buf = vec![0.0_f64; total_cells_us * r_minus_1 * coeff4];
        let mut cell_ar = vec![0.0_f64; total_cells_us * r_minus_1 * coeff4];
        let mut cell_sbb = vec![0.0_f64; total_cells_us * coeff4];
        let mut cell_sbh = vec![0.0_f64; total_cells_us * p_h * coeff4];
        let mut cell_sbw = vec![0.0_f64; total_cells_us * p_w * coeff4];
        // When `build_device_moments` is set, the host `cell_moments` vec
        // is unused (the launcher consumes the device buffer); we leave
        // it empty so it doesn't waste RAM in biobank-scale jobs.
        let mut cell_moments: Vec<f64> = if build_device_moments {
            Vec::new()
        } else {
            vec![0.0_f64; total_cells_us * moment_stride]
        };
        // Per-cell SoA for the device cubic-cell substrate. Populated on
        // every code path so the compiler sees `gpu_cells`/`gpu_branches`
        // used unconditionally — the substrate dispatch below only fires
        // when `build_device_moments` is true, but the small Vec push cost
        // per cell is negligible compared to the moment compute itself.
        let mut gpu_cells: Vec<crate::gpu::cubic_cell::GpuDenestedCubicCell> =
            Vec::with_capacity(total_cells_us);
        let mut gpu_branches: Vec<crate::gpu::cubic_cell::GpuCellBranchTag> =
            Vec::with_capacity(total_cells_us);

        // Reusable per-row coefficient buffers. Same layout as
        // BernoulliMarginalSlopeFlexRowScratch's owned [f64;4] slices.
        let mut coeff_u: Vec<[f64; 4]> = vec![[0.0; 4]; r];
        let mut coeff_au: Vec<[f64; 4]> = vec![[0.0; 4]; r];
        let mut coeff_bu: Vec<[f64; 4]> = vec![[0.0; 4]; r];
        let zero_family: Vec<[f64; 4]> = vec![[0.0; 4]; r];

        for row in 0..n {
            let row_ctx = Self::row_ctx(cache, row);
            let a = row_ctx.intercept;
            let q = block_states[0].eta[row];
            let b = block_states[1].eta[row];
            let marginal = self.marginal_link_map(q)?;
            row_q.push(q);
            row_b.push(b);
            row_mu1.push(marginal.mu1);
            row_mu2.push(marginal.mu2);
            row_zobs.push(self.z[row]);
            row_y.push(self.y[row]);
            row_w.push(self.weights[row]);

            let start = cell_offsets[row] as usize;
            let row_cells = bundle
                .row(row, 9)
                .expect("row cell moments presence verified above");
            for (local_idx, entry) in row_cells.iter().enumerate() {
                let cell_idx = start + local_idx;
                let cell = entry.partition_cell.cell;
                let z_mid = exact::interval_probe_point(cell.left, cell.right)?;
                let u_mid = a + b * z_mid;

                cell_c0[cell_idx] = cell.c0;
                cell_c1[cell_idx] = cell.c1;
                cell_c2[cell_idx] = cell.c2;
                cell_c3[cell_idx] = cell.c3;

                // dc_da, dc_db (scaled)
                let (dc_da_raw, dc_db_raw) = exact::denested_cell_coefficient_partials(
                    entry.partition_cell.score_span,
                    entry.partition_cell.link_span,
                    a,
                    b,
                );
                let dc_da = scale_coeff4(dc_da_raw, scale);
                let dc_db = scale_coeff4(dc_db_raw, scale);
                let (dc_daa_raw, dc_dab_raw, dc_dbb_raw) = exact::denested_cell_second_partials(
                    entry.partition_cell.score_span,
                    entry.partition_cell.link_span,
                    a,
                    b,
                );
                let dc_daa = scale_coeff4(dc_daa_raw, scale);
                let dc_dab = scale_coeff4(dc_dab_raw, scale);
                let dc_dbb = scale_coeff4(dc_dbb_raw, scale);

                // cell_a, cell_aa.
                let a_base = cell_idx * coeff4;
                for k in 0..coeff4 {
                    cell_a[a_base + k] = dc_da[k];
                    cell_aa[a_base + k] = dc_daa[k];
                }

                // Reset per-row-cell coefficient families.
                for slot in coeff_u.iter_mut() {
                    *slot = [0.0; 4];
                }
                for slot in coeff_au.iter_mut() {
                    *slot = [0.0; 4];
                }
                for slot in coeff_bu.iter_mut() {
                    *slot = [0.0; 4];
                }
                coeff_u[1] = dc_db;
                coeff_au[1] = dc_dab;
                coeff_bu[1] = dc_dbb;

                if let (Some(h_range), Some(runtime)) = (h_range.as_ref(), score_runtime) {
                    Self::for_each_deviation_basis_cubic_at(
                        runtime,
                        h_range,
                        z_mid,
                        "score-warp",
                        |_, idx, basis_span| {
                            coeff_u[idx] = scale_coeff4(
                                exact::score_basis_cell_coefficients(basis_span, b),
                                scale,
                            );
                            coeff_bu[idx] = scale_coeff4(
                                exact::score_basis_cell_coefficients(basis_span, 1.0),
                                scale,
                            );
                            Ok(())
                        },
                    )?;
                }
                if let (Some(w_range), Some(runtime)) = (w_range.as_ref(), link_runtime) {
                    Self::for_each_deviation_basis_cubic_at(
                        runtime,
                        w_range,
                        u_mid,
                        "link-wiggle",
                        |_, idx, basis_span| {
                            coeff_u[idx] = scale_coeff4(
                                exact::link_basis_cell_coefficients(basis_span, a, b),
                                scale,
                            );
                            let (dc_aw_raw, dc_bw_raw) =
                                exact::link_basis_cell_coefficient_partials(basis_span, a, b);
                            coeff_au[idx] = scale_coeff4(dc_aw_raw, scale);
                            coeff_bu[idx] = scale_coeff4(dc_bw_raw, scale);
                            Ok(())
                        },
                    )?;
                }

                // cell_r / cell_ar: indexed u in 1..r → slot u-1.
                let r_base = cell_idx * r_minus_1 * coeff4;
                for u in 1..r {
                    let off = r_base + (u - 1) * coeff4;
                    for k in 0..coeff4 {
                        cell_r_buf[off + k] = coeff_u[u][k];
                        cell_ar[off + k] = coeff_au[u][k];
                    }
                }
                // cell_sbb = coeff_bu[1].
                for k in 0..coeff4 {
                    cell_sbb[a_base + k] = coeff_bu[1][k];
                }
                // cell_sbh[c, j, *] = coeff_bu[h_range.start + j].
                if let Some(h_range) = h_range.as_ref() {
                    let h_base = cell_idx * p_h * coeff4;
                    for j in 0..p_h {
                        let off = h_base + j * coeff4;
                        let src = &coeff_bu[h_range.start + j];
                        for k in 0..coeff4 {
                            cell_sbh[off + k] = src[k];
                        }
                    }
                }
                // cell_sbw[c, j, *] = coeff_bu[w_range.start + j].
                if let Some(w_range) = w_range.as_ref() {
                    let w_base = cell_idx * p_w * coeff4;
                    for j in 0..p_w {
                        let off = w_base + j * coeff4;
                        let src = &coeff_bu[w_range.start + j];
                        for k in 0..coeff4 {
                            cell_sbw[off + k] = src[k];
                        }
                    }
                }
                // Always push the cell into the device-substrate SoA so
                // it's available for the Phase-4 GPU moment build below.
                // Push order is `cell_idx` (= start + local_idx) so the
                // resulting `[total_cells, MOMENT_STRIDE]` device buffer
                // is indexed identically to the host `cell_moments` vec.
                assert_eq!(gpu_cells.len(), cell_idx);
                gpu_cells.push(crate::gpu::cubic_cell::GpuDenestedCubicCell {
                    left: cell.left,
                    right: cell.right,
                    c0: cell.c0,
                    c1: cell.c1,
                    c2: cell.c2,
                    c3: cell.c3,
                });
                let branch = if !cell.left.is_finite() || !cell.right.is_finite() {
                    crate::gpu::cubic_cell::GpuCellBranchTag::AffineTail
                } else if cell.c2 == 0.0 && cell.c3 == 0.0 {
                    crate::gpu::cubic_cell::GpuCellBranchTag::Affine
                } else {
                    crate::gpu::cubic_cell::GpuCellBranchTag::NonAffineFinite
                };
                gpu_branches.push(branch);

                // cell_moments: copy state.moments, zero-pad to 10 — only
                // when the host fallback path is in use. When the
                // device-moment build is selected, this storage is
                // skipped entirely and the substrate produces moments
                // directly on the GPU below.
                if !build_device_moments {
                    let mom_base = cell_idx * moment_stride;
                    let src_moments: &[f64] = &entry.state.moments;
                    let copy_len = src_moments.len().min(moment_stride);
                    for k in 0..copy_len {
                        cell_moments[mom_base + k] = src_moments[k];
                    }
                }
            }

            // ── Observed-point pre-evaluation (mirrors CPU lines 9265–9314).
            let z_obs = self.z[row];
            let u_obs = a + b * z_obs;
            let obs = self.observed_denested_cell_partials(row, a, b, beta_h, beta_w)?;
            let chi_obs = eval_coeff4_at(&obs.dc_da, z_obs);
            let xi_obs = eval_coeff4_at(&obs.dc_daa, z_obs);
            row_chi.push(chi_obs);
            row_xi.push(xi_obs);

            // g_u_fixed / g_au_fixed / g_bu_fixed at z_obs (score) / u_obs (link).
            let mut g_u_fixed: Vec<[f64; 4]> = vec![[0.0; 4]; r];
            let mut g_au_fixed: Vec<[f64; 4]> = vec![[0.0; 4]; r];
            let mut g_bu_fixed: Vec<[f64; 4]> = vec![[0.0; 4]; r];
            g_u_fixed[1] = obs.dc_db;
            g_au_fixed[1] = obs.dc_dab;
            g_bu_fixed[1] = obs.dc_dbb;
            if let (Some(h_range), Some(runtime)) = (h_range.as_ref(), score_runtime) {
                Self::for_each_deviation_basis_cubic_at(
                    runtime,
                    h_range,
                    z_obs,
                    "score-warp observed",
                    |_, idx, basis_span| {
                        g_u_fixed[idx] = scale_coeff4(
                            exact::score_basis_cell_coefficients(basis_span, b),
                            scale,
                        );
                        g_bu_fixed[idx] = scale_coeff4(
                            exact::score_basis_cell_coefficients(basis_span, 1.0),
                            scale,
                        );
                        Ok(())
                    },
                )?;
            }
            if let (Some(w_range), Some(runtime)) = (w_range.as_ref(), link_runtime) {
                Self::for_each_deviation_basis_cubic_at(
                    runtime,
                    w_range,
                    u_obs,
                    "link-wiggle observed",
                    |_, idx, basis_span| {
                        g_u_fixed[idx] = scale_coeff4(
                            exact::link_basis_cell_coefficients(basis_span, a, b),
                            scale,
                        );
                        let (dc_aw_raw, dc_bw_raw) =
                            exact::link_basis_cell_coefficient_partials(basis_span, a, b);
                        g_au_fixed[idx] = scale_coeff4(dc_aw_raw, scale);
                        g_bu_fixed[idx] = scale_coeff4(dc_bw_raw, scale);
                        Ok(())
                    },
                )?;
            }

            // Build rho / tau via eval_coeff4_at, mirroring CPU :9319–:9322.
            let row_rho_base = row * r;
            let row_tau_base = row * r;
            for u in 1..r {
                row_rho[row_rho_base + u] = eval_coeff4_at(&g_u_fixed[u], z_obs);
                row_tau[row_tau_base + u] = eval_coeff4_at(&g_au_fixed[u], z_obs);
            }
            // r_uv via pair_from_b_family(g_b_first, ·, ·, BHW), mirroring
            // CPU :9343–:9356. Symmetric — fill both off-diagonals.
            let g_jet = SparsePrimaryCoeffJetView::new(
                1,
                h_range.as_ref(),
                w_range.as_ref(),
                g_u_fixed.as_slice(),
                g_au_fixed.as_slice(),
                g_bu_fixed.as_slice(),
                zero_family.as_slice(),
                zero_family.as_slice(),
                zero_family.as_slice(),
                zero_family.as_slice(),
                zero_family.as_slice(),
                zero_family.as_slice(),
                zero_family.as_slice(),
            );
            let row_ruv_base = row * r * r;
            for u in 0..r {
                for v in u..r {
                    let pair = g_jet.pair_from_b_family(g_jet.b_first, u, v, COEFF_SUPPORT_BHW);
                    let val = eval_coeff4_at(&pair, z_obs);
                    row_ruv[row_ruv_base + u * r + v] = val;
                    if u != v {
                        row_ruv[row_ruv_base + v * r + u] = val;
                    }
                }
            }
        }

        // ── Phase-4: when device-moment build was selected, dispatch the
        //    cubic-cell substrate now (all rows' cells were collected in
        //    `gpu_cells` / `gpu_branches` during the per-row loop). The
        //    returned device buffer lives on the shared CUDA context the
        //    bms_flex_row backend also uses, so the launcher consumes it
        //    without any cross-context copying.
        #[cfg(target_os = "linux")]
        let cell_moments_device: Option<cudarc::driver::CudaSlice<f64>> = if build_device_moments {
            #[cfg(debug_assertions)]
            use crate::gpu::cubic_cell::CubicCellMomentStatus;
            use crate::gpu::cubic_cell::{
                CubicCellDerivativeMomentHostView, CubicCellDerivativeMomentOutput,
                CubicCellMomentResidency, try_build_cubic_cell_derivative_moments,
            };
            // Sanity: the per-row loop must have produced exactly one
            // entry per cell index.
            if gpu_cells.len() != total_cells_us || gpu_branches.len() != total_cells_us {
                return Err(format!(
                    "bms_flex_row pack: gpu_cells.len()={} branches.len()={} mismatch total_cells={}",
                    gpu_cells.len(),
                    gpu_branches.len(),
                    total_cells_us
                ));
            }
            let view = CubicCellDerivativeMomentHostView {
                cells: &gpu_cells,
                branches: &gpu_branches,
                max_degree: crate::gpu::bms_flex_row::MOMENT_STRIDE - 1,
                residency: CubicCellMomentResidency::Device,
            };
            match try_build_cubic_cell_derivative_moments(view)
                .map_err(|err| format!("bms_flex_row device-moment build: {err}"))?
            {
                Some(CubicCellDerivativeMomentOutput::Device {
                    d_moments,
                    status,
                    stride,
                    n_cells,
                }) => {
                    if stride != crate::gpu::bms_flex_row::MOMENT_STRIDE
                        || n_cells != total_cells_us
                    {
                        return Err(format!(
                            "bms_flex_row device-moment substrate returned bad shape: \
                             stride={stride} n_cells={n_cells} expected stride={} cells={}",
                            crate::gpu::bms_flex_row::MOMENT_STRIDE,
                            total_cells_us
                        ));
                    }
                    // Any non-OK status means a cell the kernel refused;
                    // the row buffer for that cell is zeroed, which is
                    // mathematically OK (zero moments → zero contribution)
                    // but indicates a classifier disagreement worth
                    // surfacing in debug builds.
                    #[cfg(debug_assertions)]
                    {
                        for (i, &s) in status.iter().enumerate() {
                            assert_eq!(
                                s,
                                CubicCellMomentStatus::Ok as u8,
                                "bms_flex_row device-moment cell {i} status={s} (kernel refused)"
                            );
                        }
                    }
                    // `status` is consumed only by the debug assert above;
                    // the runtime path keeps the device buffer alive on
                    // the owned bundle and lets the launcher feed it
                    // straight into the row kernel.
                    drop(status);
                    Some(d_moments)
                }
                Some(CubicCellDerivativeMomentOutput::Host { .. }) => {
                    // The substrate degraded to host residency (runtime
                    // probe failed mid-flight). Fall back to filling the
                    // host moments path here so the launcher still has a
                    // valid bundle. We need to do the work the per-row
                    // loop skipped: re-fill `cell_moments` from the
                    // existing CPU LRU cache entries.
                    cell_moments = vec![0.0_f64; total_cells_us * moment_stride];
                    for (row_idx, _) in (0..n).enumerate() {
                        let start = cell_offsets[row_idx] as usize;
                        let row_cells = bundle
                            .row(row_idx, 9)
                            .expect("row cell moments presence verified above");
                        for (local_idx, entry) in row_cells.iter().enumerate() {
                            let cell_idx = start + local_idx;
                            let mom_base = cell_idx * moment_stride;
                            let src_moments: &[f64] = &entry.state.moments;
                            let copy_len = src_moments.len().min(moment_stride);
                            for k in 0..copy_len {
                                cell_moments[mom_base + k] = src_moments[k];
                            }
                        }
                    }
                    None
                }
                None => {
                    return Err(
                        "bms_flex_row device-moment substrate returned Ok(None) on non-empty input"
                            .to_string(),
                    );
                }
            }
        } else {
            None
        };
        // Free the now-unneeded scratch.
        drop(gpu_cells);
        drop(gpu_branches);

        Ok(Some(
            crate::gpu::bms_flex_row::BmsFlexRowKernelInputsOwned {
                n_rows: n,
                r,
                p_h,
                p_w,
                s_f: scale,
                q: row_q,
                b: row_b,
                mu_1: row_mu1,
                mu_2: row_mu2,
                z_obs: row_zobs,
                y: row_y,
                w: row_w,
                cell_offsets,
                cell_c0,
                cell_c1,
                cell_c2,
                cell_c3,
                cell_a,
                cell_aa,
                cell_r: cell_r_buf,
                cell_ar,
                cell_sbb,
                cell_sbh,
                cell_sbw,
                cell_moments,
                #[cfg(target_os = "linux")]
                cell_moments_device,
                chi_obs: row_chi,
                xi_obs: row_xi,
                rho_u: row_rho,
                tau_u: row_tau,
                r_uv: row_ruv,
            },
        ))
    }

    pub(super) fn build_row_primary_hessian_cache(
        &self,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<RowPrimaryEvalCache, String> {
        if !self.effective_flex_active(block_states)? {
            return Ok(RowPrimaryEvalCache::Empty);
        }
        let n = self.y.len();
        let primary = &cache.primary;
        let r = primary.total;
        let runtime_available = runtime_available_memory_bytes();
        let workspace_pinned = bms_row_primary_hessian_pinned_bytes().load(Ordering::Acquire);
        let plan = decide_row_primary_hessian_cache(
            n,
            r,
            BMS_ROW_PRIMARY_HESSIAN_EXPECTED_REUSE_PASSES,
            runtime_available,
            workspace_pinned,
        );
        let gpu_decision = crate::gpu::bms_flex::require_row_primary_hessian_supported(n, r)?;
        // Milestone 2 (#210): when the policy says GPU, eagerly probe the
        // backend so any NVRTC compile / context init failure surfaces in
        // the cache-decision log instead of at first dispatch. Probe
        // returning `NotYetImplemented` is the expected pre-milestone-3
        // outcome and means dispatch falls through to CPU rows below —
        // the same path as today.
        if gpu_decision.use_gpu {
            match crate::gpu::bms_flex::BmsFlexGpuBackend::probe() {
                Ok(backend) => {
                    if log_exact_work(n) {
                        log::info!(
                            "[BMS row-primary-hessian-cache] gpu_backend_ready: {}",
                            backend.describe()
                        );
                    }
                }
                Err(crate::gpu::error::GpuError::NotYetImplemented { reason }) => {
                    log::info!(
                        "[BMS row-primary-hessian-cache] gpu_backend_pending: {reason}; \
                         falling back to CPU rows"
                    );
                }
                Err(err) => {
                    log::info!(
                        "[BMS row-primary-hessian-cache] gpu_backend_probe_failed: {err}; \
                         falling back to CPU rows"
                    );
                }
            }
        }
        if !plan.materialize {
            if log_exact_work(n) {
                log::info!(
                    "[BMS row-primary-hessian-cache] decision=stream need_bytes={} avail_bytes={} workspace_pinned={} single_cache_budget={} global_pin_budget={} n={} r={} expected_reuse_passes={} materialized_row_hessian_evals={} streamed_row_hessian_evals={} reason={} gpu_policy={} gpu_selected={} gpu_reason={}",
                    plan.bytes,
                    plan.runtime_available_bytes,
                    plan.workspace_pinned_bytes,
                    plan.single_cache_budget_bytes,
                    plan.global_pin_budget_bytes,
                    n,
                    r,
                    plan.expected_reuse_passes,
                    plan.materialized_row_hessian_evals,
                    plan.streamed_row_hessian_evals,
                    plan.reason.as_str(),
                    gpu_decision.policy.as_str(),
                    gpu_decision.use_gpu,
                    gpu_decision.reason,
                );
            }
            return Ok(RowPrimaryEvalCache::Empty);
        }
        let started = std::time::Instant::now();
        let heartbeat_guard = crate::heartbeat::scope(format!(
            "BMS row-primary-hessian-cache n={n} r={r} bytes={} single_budget={} global_budget={}",
            plan.bytes, plan.single_cache_budget_bytes, plan.global_pin_budget_bytes
        ));
        if log_exact_work(n) {
            log::info!(
                "[BMS row-primary-hessian-cache] decision=materialize need_bytes={} avail_bytes={} workspace_pinned={} single_cache_budget={} global_pin_budget={} n={} r={} expected_reuse_passes={} materialized_row_hessian_evals={} streamed_row_hessian_evals={} reason={} gpu_policy={} gpu_selected={} gpu_reason={}",
                plan.bytes,
                plan.runtime_available_bytes,
                plan.workspace_pinned_bytes,
                plan.single_cache_budget_bytes,
                plan.global_pin_budget_bytes,
                n,
                r,
                plan.expected_reuse_passes,
                plan.materialized_row_hessian_evals,
                plan.streamed_row_hessian_evals,
                plan.reason.as_str(),
                gpu_decision.policy.as_str(),
                gpu_decision.use_gpu,
                gpu_decision.reason,
            );
        }
        // ── BMS-FLEX GPU milestone 1: when the policy says use_gpu *and* the
        //    Stage-2 device kernel preconditions are met (StandardNormal
        //    latent, row-cell-moments bundle present, optional score-warp /
        //    link-deviation runtimes present), pack the host inputs once and
        //    dispatch the row kernel. A successful launch returns the
        //    `n × r²` row-major Hessian; the CPU rayon loop below is then
        //    skipped. Any failure (`NotYetImplemented`, driver errors, or
        //    pack-time precondition mismatch) logs a one-liner and falls
        //    through to the existing CPU path, preserving production
        //    behaviour under `gpu=auto`. Under `gpu=force`, the upstream
        //    `require_row_primary_hessian_supported` would already have
        //    failed; here we still fall back on launch failure rather than
        //    panic mid-fit.
        if gpu_decision.use_gpu {
            match self.pack_bms_flex_row_kernel_inputs(block_states, cache)? {
                Some(owned) => {
                    // Phase-3: when both marginal/logslope designs expose a
                    // contiguous dense view, take the device-resident path
                    // that keeps the n×r² row Hessian + designs resident on
                    // the GPU so subsequent HVP / diagonal launches do not
                    // round-trip 626 MB through host memory.
                    #[cfg(target_os = "linux")]
                    {
                        let marginal_dense = self.marginal_design.as_dense_ref();
                        let logslope_dense = self.logslope_design.as_dense_ref();
                        if let (Some(md), Some(gd)) = (marginal_dense, logslope_dense) {
                            // Both designs must be row-major contiguous for the
                            // device upload's `[n, p]` layout to be byte-correct.
                            let md_is_rowmajor = md.is_standard_layout();
                            let gd_is_rowmajor = gd.is_standard_layout();
                            if md_is_rowmajor && gd_is_rowmajor {
                                let block_layout = crate::gpu::bms_flex_row::BmsFlexBlockLayout {
                                    p_m: cache.slices.marginal.len(),
                                    p_g: cache.slices.logslope.len(),
                                    h: cache.slices.h.clone(),
                                    w: cache.slices.w.clone(),
                                    p_total: cache.slices.total,
                                };
                                let primary_layout =
                                    crate::gpu::bms_flex_row::BmsFlexPrimaryLayout {
                                        h: primary.h.clone(),
                                        w: primary.w.clone(),
                                        r: primary.total,
                                    };
                                let md_slice = md
                                    .as_slice()
                                    .expect("dense marginal_design is row-major contiguous");
                                let gd_slice = gd
                                    .as_slice()
                                    .expect("dense logslope_design is row-major contiguous");
                                match crate::gpu::bms_flex_row::launch_bms_flex_row_kernel_device_resident(
                                    owned.as_borrowed(),
                                    md_slice,
                                    gd_slice,
                                    block_layout,
                                    primary_layout,
                                ) {
                                    Ok(device_state) => {
                                        if log_exact_work(n) {
                                            log::info!(
                                                "[BMS row-primary-hessian-cache] gpu_device_resident_ok rows={} r={} elapsed={:.3}s",
                                                n,
                                                r,
                                                started.elapsed().as_secs_f64()
                                            );
                                        }
                                        drop(heartbeat_guard);
                                        return Ok(RowPrimaryEvalCache::Device(device_state));
                                    }
                                    Err(err) => {
                                        log::info!(
                                            "[BMS row-primary-hessian-cache] gpu_device_resident_failed: {err}; \
                                             falling back to host-pin GPU launch"
                                        );
                                    }
                                }
                            }
                        }
                    }
                    match crate::gpu::bms_flex_row::launch_bms_flex_row_kernel(owned.as_borrowed())
                    {
                        Ok(outputs) => {
                            if log_exact_work(n) {
                                log::info!(
                                    "[BMS row-primary-hessian-cache] gpu_launch_ok rows={} r={} elapsed={:.3}s",
                                    n,
                                    r,
                                    started.elapsed().as_secs_f64()
                                );
                            }
                            let packed_neglog = Array1::<f64>::from_vec(outputs.neglog);
                            let packed_grad =
                                Array2::<f64>::from_shape_vec((n, r), outputs.grad)
                                    .map_err(|err| format!("bms_flex_row grad shape: {err}"))?;
                            let packed_hess =
                                Array2::<f64>::from_shape_vec((n, r * r), outputs.hess)
                                    .map_err(|err| format!("bms_flex_row hess shape: {err}"))?;
                            drop(heartbeat_guard);
                            return Ok(RowPrimaryEvalCache::Host(RowPrimaryEvalPin::new(
                                packed_neglog,
                                packed_grad,
                                packed_hess,
                                plan.bytes,
                            )));
                        }
                        Err(err) => {
                            log::info!(
                                "[BMS row-primary-hessian-cache] gpu_launch_failed: {err}; \
                             falling back to CPU rows"
                            );
                        }
                    }
                }
                None => {
                    if log_exact_work(n) {
                        log::info!(
                            "[BMS row-primary-hessian-cache] gpu_unsupported_inputs; \
                             falling back to CPU rows"
                        );
                    }
                }
            }
        }
        let completed_rows = AtomicUsize::new(0);
        let progress_step = (n / 10).max(1);
        // Collect (neglog, grad_flat, hess_flat) per row in parallel, then
        // assemble into the three packed arrays. Using par_iter collect avoids
        // unsafe pointer aliasing while writing disjoint row slices.
        let row_evals: Vec<(f64, Vec<f64>, Vec<f64>)> = (0..n)
            .into_par_iter()
            .map(|row| -> Result<(f64, Vec<f64>, Vec<f64>), String> {
                let row_ctx = Self::row_ctx(cache, row);
                let mut scratch = BernoulliMarginalSlopeFlexRowScratch::new(r);
                let row_moments = cache
                    .row_cell_moments
                    .as_ref()
                    .and_then(|bundle| bundle.row(row, 9));
                let neglog = self.compute_row_analytic_flex_into_with_moments(
                    row,
                    block_states,
                    primary,
                    row_ctx,
                    row_moments,
                    true,
                    &mut scratch,
                )?;
                if log_exact_work(n) {
                    let done = completed_rows.fetch_add(1, Ordering::Relaxed) + 1;
                    if done == n || done % progress_step == 0 {
                        log::info!(
                            "[BMS row-primary-hessian-cache] progress rows={}/{} elapsed={:.3}s",
                            done,
                            n,
                            started.elapsed().as_secs_f64()
                        );
                    }
                }
                Ok((
                    neglog,
                    scratch.grad.to_vec(),
                    scratch
                        .hess
                        .as_slice()
                        .expect("hess is contiguous")
                        .to_vec(),
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mut packed_neglog = Array1::<f64>::zeros(n);
        let mut packed_grad = Array2::<f64>::zeros((n, r));
        let mut packed_hess = Array2::<f64>::zeros((n, r * r));
        for (row, (neglog, grad_flat, hess_flat)) in row_evals.into_iter().enumerate() {
            packed_neglog[row] = neglog;
            packed_grad
                .row_mut(row)
                .iter_mut()
                .zip(grad_flat.iter())
                .for_each(|(d, s)| *d = *s);
            packed_hess
                .row_mut(row)
                .iter_mut()
                .zip(hess_flat.iter())
                .for_each(|(d, s)| *d = *s);
        }
        if log_exact_work(n) {
            log::info!(
                "[BMS row-primary-hessian-cache] build done n={} r={} elapsed={:.3}s",
                n,
                r,
                started.elapsed().as_secs_f64()
            );
        }
        drop(heartbeat_guard);
        Ok(RowPrimaryEvalCache::Host(RowPrimaryEvalPin::new(
            packed_neglog,
            packed_grad,
            packed_hess,
            plan.bytes,
        )))
    }

    /// Look up the cached per-row primary Hessian (`r × r`) materialized at
    /// the workspace β snapshot when `row_primary_hessians` is populated.
    /// Returns `None` when the cache is absent or the row index is out of
    /// range, in which case the caller must fall back to the live row
    /// kernel.
    /// Returns the cached row-primary Hessian (`r × r`) for host-resident
    /// caches. Returns `None` when the cache is absent, device-resident, or
    /// the row is out of range.
    #[inline]
    pub(super) fn cached_row_primary_hessian<'a>(
        cache: &'a BernoulliMarginalSlopeExactEvalCache,
        row: usize,
    ) -> Option<ArrayView2<'a, f64>> {
        let hess = cache.row_primary_hessians.host_pin()?.hess();
        let r = cache.primary.total;
        if row >= hess.nrows() {
            return None;
        }
        let width = r.checked_mul(r)?;
        let start = row.checked_mul(width)?;
        let end = start.checked_add(width)?;
        ArrayView2::from_shape((r, r), hess.as_slice()?.get(start..end)?).ok()
    }

    /// Returns the cached row-primary (neglog, grad_row) for host-resident
    /// caches. Returns `None` when the cache is absent, device-resident, or
    /// the row is out of range. Device-resident caches recompute the row
    /// kernel on the rare CPU-fused-gradient fallback path; the GPU
    /// dense-block kernel handles the hot path for them directly.
    #[inline]
    pub(super) fn cached_row_primary_eval<'a>(
        cache: &'a BernoulliMarginalSlopeExactEvalCache,
        row: usize,
    ) -> Option<(f64, ArrayView1<'a, f64>)> {
        let pin = cache.row_primary_hessians.host_pin()?;
        let neglog = pin.neglog();
        let grad = pin.grad();
        let r = cache.primary.total;
        if row >= neglog.len() || row >= grad.nrows() {
            return None;
        }
        let neglog_val = neglog[row];
        let grad_row = grad.row(row);
        // Sanity: grad row must have exactly r elements.
        if grad_row.len() != r {
            return None;
        }
        Some((neglog_val, grad_row))
    }

    pub(super) fn build_exact_eval_cache(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<BernoulliMarginalSlopeExactEvalCache, String> {
        self.build_exact_eval_cache_with_order(block_states)
    }

    pub(super) fn row_primary_direction_from_flat(
        &self,
        row: usize,
        slices: &BlockSlices,
        primary: &PrimarySlices,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        let mut out = Array1::<f64>::zeros(primary.total);
        self.row_primary_direction_from_flat_into(row, slices, primary, d_beta_flat, &mut out)?;
        Ok(out)
    }

    /// Allocation-free variant of [`Self::row_primary_direction_from_flat`]:
    /// fills `out` (length `primary.total`) with the primary-space projection
    /// of `d_beta_flat`. `out` is fully overwritten on success.
    pub(super) fn row_primary_direction_from_flat_into(
        &self,
        row: usize,
        slices: &BlockSlices,
        primary: &PrimarySlices,
        d_beta_flat: &Array1<f64>,
        out: &mut Array1<f64>,
    ) -> Result<(), String> {
        if d_beta_flat.len() != slices.total {
            return Err(format!(
                "bernoulli marginal-slope d_beta length mismatch: got {}, expected {}",
                d_beta_flat.len(),
                slices.total
            ));
        }
        out[primary.q] = self
            .marginal_design
            .dot_row_view(row, d_beta_flat.slice(s![slices.marginal.clone()]));
        out[primary.logslope] = self
            .logslope_design
            .dot_row_view(row, d_beta_flat.slice(s![slices.logslope.clone()]));
        if let (Some(block_range), Some(primary_range)) = (slices.h.as_ref(), primary.h.as_ref()) {
            out.slice_mut(s![primary_range.start..primary_range.end])
                .assign(&d_beta_flat.slice(s![block_range.clone()]).to_owned());
        }
        if let (Some(block_range), Some(primary_range)) = (slices.w.as_ref(), primary.w.as_ref()) {
            out.slice_mut(s![primary_range.start..primary_range.end])
                .assign(&d_beta_flat.slice(s![block_range.clone()]).to_owned());
        }
        Ok(())
    }

    pub(super) fn stacked_direction_block(
        d_beta_flats: &[Array1<f64>],
        range: std::ops::Range<usize>,
    ) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((range.len(), d_beta_flats.len()));
        for (dir_idx, d_beta_flat) in d_beta_flats.iter().enumerate() {
            out.column_mut(dir_idx)
                .assign(&d_beta_flat.slice(s![range.clone()]));
        }
        out
    }

    pub(super) fn row_primary_directions_from_projected(
        local_row: usize,
        slices: &BlockSlices,
        primary: &PrimarySlices,
        d_beta_flats: &[Array1<f64>],
        marginal_projected: &Array2<f64>,
        logslope_projected: &Array2<f64>,
    ) -> Vec<Array1<f64>> {
        let mut out = Vec::with_capacity(d_beta_flats.len());
        for (dir_idx, d_beta_flat) in d_beta_flats.iter().enumerate() {
            let mut direction = Array1::<f64>::zeros(primary.total);
            direction[primary.q] = marginal_projected[[local_row, dir_idx]];
            direction[primary.logslope] = logslope_projected[[local_row, dir_idx]];
            if let (Some(block_range), Some(primary_range)) =
                (slices.h.as_ref(), primary.h.as_ref())
            {
                direction
                    .slice_mut(s![primary_range.start..primary_range.end])
                    .assign(&d_beta_flat.slice(s![block_range.clone()]));
            }
            if let (Some(block_range), Some(primary_range)) =
                (slices.w.as_ref(), primary.w.as_ref())
            {
                direction
                    .slice_mut(s![primary_range.start..primary_range.end])
                    .assign(&d_beta_flat.slice(s![block_range.clone()]));
            }
            out.push(direction);
        }
        out
    }

    pub(super) fn batched_directional_derivative_chunk_rows(
        n: usize,
        n_dirs: usize,
    ) -> (usize, bool) {
        // CPU-only path: chunk by a fixed float-count budget so each chunk is
        // small enough to keep the per-row workspaces in L2/L3 across the
        // directional sweep. The GPU dispatch path was removed when the
        // dense-PIRLS routing helpers were retired (no live device backend
        // in this build); revisit when a runtime device backend is
        // reintroduced.
        const CPU_TARGET_CHUNK_FLOATS: usize = 1 << 17;
        let cpu_rows = (CPU_TARGET_CHUNK_FLOATS / (3 * n_dirs).max(1)).clamp(1024, n.max(1));
        (cpu_rows.min(n.max(1)), false)
    }

    pub(super) fn row_primary_psi_direction_from_map(
        &self,
        row: usize,
        block_idx: usize,
        psi_map: &crate::families::custom_family::PsiDesignMap,
        block_states: &[ParameterBlockState],
        primary: &PrimarySlices,
    ) -> Result<Array1<f64>, String> {
        let mut out = Array1::<f64>::zeros(primary.total);
        match block_idx {
            0 => {
                let x_row = psi_map
                    .row_vector(row)
                    .map_err(|e| format!("survival rowwise psi map: {e}"))?;
                out[primary.q] = x_row.dot(&block_states[0].beta);
            }
            1 => {
                let x_row = psi_map
                    .row_vector(row)
                    .map_err(|e| format!("survival rowwise psi map: {e}"))?;
                out[primary.logslope] = x_row.dot(&block_states[1].beta);
            }
            _ => {
                return Err(format!(
                    "bernoulli marginal-slope psi direction only supports spatial marginal/logslope blocks, got block {block_idx}"
                ));
            }
        }
        Ok(out)
    }

    pub(super) fn row_primary_psi_action_on_direction_from_map(
        &self,
        row: usize,
        block_idx: usize,
        psi_map: &crate::families::custom_family::PsiDesignMap,
        slices: &BlockSlices,
        d_beta_flat: &Array1<f64>,
        primary: &PrimarySlices,
    ) -> Result<Array1<f64>, String> {
        let mut out = Array1::<f64>::zeros(primary.total);
        match block_idx {
            0 => {
                let x_row = psi_map
                    .row_vector(row)
                    .map_err(|e| format!("survival rowwise psi map: {e}"))?;
                out[primary.q] =
                    x_row.dot(&d_beta_flat.slice(s![slices.marginal.clone()]).to_owned())
            }
            1 => {
                let x_row = psi_map
                    .row_vector(row)
                    .map_err(|e| format!("survival rowwise psi map: {e}"))?;
                out[primary.logslope] =
                    x_row.dot(&d_beta_flat.slice(s![slices.logslope.clone()]).to_owned())
            }
            _ => {
                return Err(format!(
                    "bernoulli marginal-slope psi action only supports marginal/logslope blocks, got block {block_idx}"
                ));
            }
        }
        Ok(out)
    }

    pub(super) fn pullback_primary_vector(
        &self,
        row: usize,
        slices: &BlockSlices,
        primary: &PrimarySlices,
        primary_vec: &Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        let mut out = Array1::<f64>::zeros(slices.total);
        self.pullback_primary_vector_add_into(row, slices, primary, primary_vec, &mut out)?;
        Ok(out)
    }

    /// Allocation-free variant of [`Self::pullback_primary_vector`]: *adds*
    /// the pullback of `primary_vec` into the existing accumulator `out`
    /// (length `slices.total`).  `out` is **not** zeroed first; the caller
    /// must initialise it before the first call on a given accumulation.
    pub(super) fn pullback_primary_vector_add_into(
        &self,
        row: usize,
        slices: &BlockSlices,
        primary: &PrimarySlices,
        primary_vec: &Array1<f64>,
        out: &mut Array1<f64>,
    ) -> Result<(), String> {
        {
            let mut marginal = out.slice_mut(s![slices.marginal.clone()]);
            self.marginal_design
                .axpy_row_into(row, primary_vec[primary.q], &mut marginal)?;
        }
        {
            let mut logslope = out.slice_mut(s![slices.logslope.clone()]);
            self.logslope_design.axpy_row_into(
                row,
                primary_vec[primary.logslope],
                &mut logslope,
            )?;
        }
        if let Some(primary_h) = primary.h.as_ref()
            && let Some(block_h) = slices.h.as_ref()
        {
            out.slice_mut(s![block_h.clone()]).zip_mut_with(
                &primary_vec.slice(s![primary_h.start..primary_h.end]),
                |a, &b| {
                    *a += b;
                },
            );
        }
        if let Some(primary_w) = primary.w.as_ref()
            && let Some(block_w) = slices.w.as_ref()
        {
            out.slice_mut(s![block_w.clone()]).zip_mut_with(
                &primary_vec.slice(s![primary_w.start..primary_w.end]),
                |a, &b| {
                    *a += b;
                },
            );
        }
        Ok(())
    }

    pub(super) fn block_psi_row_from_map(
        &self,
        row: usize,
        block_idx: usize,
        psi_map: &crate::families::custom_family::PsiDesignMap,
        slices: &BlockSlices,
    ) -> Result<BlockPsiRow, String> {
        let (local_vec, range) = match block_idx {
            0 => (
                psi_map
                    .row_vector(row)
                    .map_err(|e| format!("survival rowwise psi map: {e}"))?,
                slices.marginal.clone(),
            ),
            1 => (
                psi_map
                    .row_vector(row)
                    .map_err(|e| format!("survival rowwise psi map: {e}"))?,
                slices.logslope.clone(),
            ),
            _ => {
                return Err(format!(
                    "bernoulli marginal-slope psi embedding only supports marginal/logslope blocks, got block {block_idx}"
                ));
            }
        };
        Ok(BlockPsiRow {
            block_idx,
            range,
            local_vec,
        })
    }

    /// Returns (neg_log_lik, gradient, Hessian) in primary coordinates.
    /// Fully analytic for both flex and non-flex paths — no AD jets.
    pub(super) fn compute_row_primary_gradient_hessian(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        primary: &PrimarySlices,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        // Flex path: full IFT analytic kernel.
        if self.effective_flex_active(block_states)? {
            let mut scratch = BernoulliMarginalSlopeFlexRowScratch::new(primary.total);
            let neglog = self.compute_row_analytic_flex_into(
                row,
                block_states,
                primary,
                row_ctx,
                true,
                &mut scratch,
            )?;
            return Ok((neglog, scratch.grad, scratch.hess));
        }
        // Rigid path: closed-form observed eta with probit frailty scaling.
        // primary.total == 2 (q at 0, g at 1), no h/w blocks.
        let marginal_eta = block_states[0].eta[row];
        let marginal = self.marginal_link_map(marginal_eta)?;
        let g = block_states[1].eta[row];
        let (neglog, grad_pair, h) = self.rigid_row_kernel_eval(row, marginal_eta, marginal, g)?;
        let mut grad = Array1::<f64>::zeros(2);
        grad[0] = grad_pair[0];
        grad[1] = grad_pair[1];

        let mut hess = Array2::<f64>::zeros((2, 2));
        hess[[0, 0]] = h[0][0];
        hess[[0, 1]] = h[0][1];
        hess[[1, 0]] = h[1][0];
        hess[[1, 1]] = h[1][1];

        Ok((neglog, grad, hess))
    }

    pub(super) fn compute_row_primary_gradient_hessian_reusing_cache(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        primary: &PrimarySlices,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<(Array1<f64>, Array2<f64>), String> {
        if self.effective_flex_active(block_states)?
            && let Some(cached_hessian) = Self::cached_row_primary_hessian(cache, row)
        {
            let mut scratch = BernoulliMarginalSlopeFlexRowScratch::new(primary.total);
            let row_moments = cache
                .row_cell_moments
                .as_ref()
                .and_then(|bundle| bundle.row(row, 3));
            self.compute_row_analytic_flex_into_with_moments(
                row,
                block_states,
                primary,
                row_ctx,
                row_moments,
                false,
                &mut scratch,
            )?;
            return Ok((scratch.grad, cached_hessian.to_owned()));
        }
        let (_, grad, hess) =
            self.compute_row_primary_gradient_hessian(row, block_states, primary, row_ctx)?;
        Ok((grad, hess))
    }

    pub(super) fn compute_row_analytic_flex_into(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        primary: &PrimarySlices,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        need_hessian: bool,
        scratch: &mut BernoulliMarginalSlopeFlexRowScratch,
    ) -> Result<f64, String> {
        self.compute_row_analytic_flex_into_with_moments(
            row,
            block_states,
            primary,
            row_ctx,
            None,
            need_hessian,
            scratch,
        )
    }

    pub(super) fn compute_row_analytic_flex_into_with_moments(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        primary: &PrimarySlices,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        row_cell_moments: Option<&[CachedDenestedCellMoments]>,
        need_hessian: bool,
        scratch: &mut BernoulliMarginalSlopeFlexRowScratch,
    ) -> Result<f64, String> {
        let q = block_states[0].eta[row];
        let b = block_states[1].eta[row];
        let beta_h = self.score_beta(block_states)?;
        let beta_w = self.link_beta(block_states)?;
        self.compute_row_analytic_flex_from_parts_into(
            row,
            primary,
            q,
            b,
            beta_h,
            beta_w,
            row_ctx,
            row_cell_moments,
            need_hessian,
            scratch,
        )
    }

    pub(super) fn compute_row_analytic_flex_from_parts_into(
        &self,
        row: usize,
        primary: &PrimarySlices,
        q: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        row_cell_moments: Option<&[CachedDenestedCellMoments]>,
        need_hessian: bool,
        scratch: &mut BernoulliMarginalSlopeFlexRowScratch,
    ) -> Result<f64, String> {
        use super::exact_kernel as exact;

        let r = primary.total;
        scratch.reset(need_hessian);
        // Reusable per-row coefficient buffers live on the scratch. Resize once
        // if the scratch was constructed for a different primary dimension; the
        // common case is `len == r` so this is a no-op.
        if scratch.coeff_u.len() != r {
            scratch.coeff_u.resize(r, [0.0; 4]);
            scratch.coeff_au.resize(r, [0.0; 4]);
            scratch.coeff_bu.resize(r, [0.0; 4]);
            scratch.g_u_fixed.resize(r, [0.0; 4]);
            scratch.g_au_fixed.resize(r, [0.0; 4]);
            scratch.g_bu_fixed.resize(r, [0.0; 4]);
            scratch.eta_u_cell.resize(r, 0.0);
            scratch.zero_family.resize(r, [0.0; 4]);
        }
        let a = row_ctx.intercept;
        let f_a = row_ctx.m_a;
        let y_i = self.y[row];
        let w_i = self.weights[row];
        let s_y = 2.0 * y_i - 1.0;
        let marginal = self.marginal_link_map(q)?;
        let inv_ma = 1.0 / f_a;
        let h_range = primary.h.as_ref();
        let w_range = primary.w.as_ref();
        let score_runtime = self.score_warp.as_ref();
        let link_runtime = self.link_dev.as_ref();
        let scale = self.probit_frailty_scale();

        // Split-borrow the scratch into disjoint mutable references; the
        // borrow checker permits this because every field access goes through
        // `scratch.<field>` directly rather than through `&mut scratch`.
        let f_u = &mut scratch.m_u;
        let f_au = &mut scratch.m_au;
        let f_uv = &mut scratch.m_uv;
        let coeff_u = &mut scratch.coeff_u;
        let coeff_au = &mut scratch.coeff_au;
        let coeff_bu = &mut scratch.coeff_bu;
        let g_u_fixed = &mut scratch.g_u_fixed;
        let g_au_fixed = &mut scratch.g_au_fixed;
        let g_bu_fixed = &mut scratch.g_bu_fixed;
        let eta_u_cell = &mut scratch.eta_u_cell;
        let zero_family: &[[f64; 4]] = scratch.zero_family.as_slice();
        let mut f_aa = 0.0f64;

        if let Some(empirical_grid) = self.latent_measure.empirical_grid_for_training_row(row)? {
            for (&node, &weight) in empirical_grid
                .nodes
                .iter()
                .zip(empirical_grid.weights.iter())
            {
                coeff_u.fill([0.0; 4]);
                coeff_au.fill([0.0; 4]);
                coeff_bu.fill([0.0; 4]);

                let obs = self.observed_denested_cell_partials_at_z(node, a, b, beta_h, beta_w)?;
                let eta = eval_coeff4_at(&obs.coeff, node);
                let eta_a = eval_coeff4_at(&obs.dc_da, node);
                let eta_aa = eval_coeff4_at(&obs.dc_daa, node);
                let phi = normal_pdf(eta);
                if need_hessian {
                    f_aa += weight * phi * (eta_aa - eta * eta_a * eta_a);
                }

                coeff_u[1] = obs.dc_db;
                if need_hessian {
                    coeff_au[1] = obs.dc_dab;
                    coeff_bu[1] = obs.dc_dbb;
                }

                if let (Some(h_range), Some(runtime)) = (h_range, score_runtime) {
                    Self::for_each_deviation_basis_cubic_at(
                        runtime,
                        h_range,
                        node,
                        "score-warp",
                        |_, idx, basis_span| {
                            coeff_u[idx] = scale_coeff4(
                                exact::score_basis_cell_coefficients(basis_span, b),
                                scale,
                            );
                            if need_hessian {
                                coeff_bu[idx] = scale_coeff4(
                                    exact::score_basis_cell_coefficients(basis_span, 1.0),
                                    scale,
                                );
                            }
                            Ok(())
                        },
                    )?;
                }

                if let (Some(w_range), Some(runtime)) = (w_range, link_runtime) {
                    let u_node = a + b * node;
                    Self::for_each_deviation_basis_cubic_at(
                        runtime,
                        w_range,
                        u_node,
                        "link-wiggle",
                        |_, idx, basis_span| {
                            coeff_u[idx] = scale_coeff4(
                                exact::link_basis_cell_coefficients(basis_span, a, b),
                                scale,
                            );
                            if need_hessian {
                                let (dc_aw_raw, dc_bw_raw) =
                                    exact::link_basis_cell_coefficient_partials(basis_span, a, b);
                                coeff_au[idx] = scale_coeff4(dc_aw_raw, scale);
                                coeff_bu[idx] = scale_coeff4(dc_bw_raw, scale);
                            }
                            Ok(())
                        },
                    )?;
                }

                for idx in 0..r {
                    eta_u_cell[idx] = eval_coeff4_at(&coeff_u[idx], node);
                }
                for u in 1..r {
                    f_u[u] += weight * phi * eta_u_cell[u];
                    if need_hessian {
                        let eta_au = eval_coeff4_at(&coeff_au[u], node);
                        f_au[u] += weight * phi * (eta_au - eta * eta_a * eta_u_cell[u]);
                    }
                }

                if need_hessian {
                    let coeff_jet = SparsePrimaryCoeffJetView::new(
                        1,
                        h_range,
                        w_range,
                        coeff_u.as_slice(),
                        coeff_au.as_slice(),
                        coeff_bu.as_slice(),
                        zero_family,
                        zero_family,
                        zero_family,
                        zero_family,
                        zero_family,
                        zero_family,
                        zero_family,
                    );
                    for u in 1..r {
                        for v in u..r {
                            let second_coeff = coeff_jet.pair_from_b_family(
                                coeff_jet.b_first,
                                u,
                                v,
                                COEFF_SUPPORT_BHW,
                            );
                            let eta_uv = eval_coeff4_at(&second_coeff, node);
                            let val = weight * phi * (eta_uv - eta * eta_u_cell[u] * eta_u_cell[v]);
                            f_uv[[u, v]] += val;
                            if u != v {
                                f_uv[[v, u]] += val;
                            }
                        }
                    }
                }
            }
        } else {
            // Reuse cached row moments whenever they cover the requested
            // derivative order. Degree-9 moments are exact for gradient-only
            // calls too, and avoiding a second degree-3 cell sweep preserves
            // the same calculus with less work.
            let owned_cells;
            let cached_cells: Vec<(
                exact::DenestedPartitionCell,
                std::borrow::Cow<'_, exact::CellDerivativeMomentState>,
            )> = if let Some(cached) = row_cell_moments {
                assert!(
                    !cached.is_empty(),
                    "row cell moments bundle was selected but row {row} has no cells"
                );
                cached
                    .iter()
                    .map(|entry| {
                        (
                            entry.partition_cell,
                            std::borrow::Cow::Borrowed(&entry.state),
                        )
                    })
                    .collect()
            } else if let Some(cached) = row_ctx.degree9_cells.as_ref() {
                cached
                    .iter()
                    .map(|entry| {
                        (
                            entry.partition_cell,
                            std::borrow::Cow::Borrowed(&entry.state),
                        )
                    })
                    .collect()
            } else {
                owned_cells = self.denested_partition_cells(a, b, beta_h, beta_w)?;
                owned_cells
                    .into_iter()
                    .map(|partition_cell| {
                        let degree = if need_hessian { 9 } else { 3 };
                        self.evaluate_cell_derivative_moments_lru(partition_cell.cell, degree)
                            .map(|state| (partition_cell, std::borrow::Cow::Owned(state)))
                    })
                    .collect::<Result<Vec<_>, String>>()?
            };
            for (partition_cell, state) in cached_cells {
                coeff_u.fill([0.0; 4]);
                coeff_au.fill([0.0; 4]);
                coeff_bu.fill([0.0; 4]);
                let cell = partition_cell.cell;
                let z_mid = exact::interval_probe_point(cell.left, cell.right)?;
                let u_mid = a + b * z_mid;
                let state: &exact::CellDerivativeMomentState = &state;
                let (dc_da_raw, dc_db_raw) = exact::denested_cell_coefficient_partials(
                    partition_cell.score_span,
                    partition_cell.link_span,
                    a,
                    b,
                );
                let dc_da = scale_coeff4(dc_da_raw, scale);
                let dc_db = scale_coeff4(dc_db_raw, scale);

                coeff_u[1] = dc_db;
                coeff_au[1] = [0.0; 4];
                coeff_bu[1] = [0.0; 4];
                if need_hessian {
                    let (dc_daa_raw, dc_dab_raw, dc_dbb_raw) = exact::denested_cell_second_partials(
                        partition_cell.score_span,
                        partition_cell.link_span,
                        a,
                        b,
                    );
                    let dc_daa = scale_coeff4(dc_daa_raw, scale);
                    let dc_dab = scale_coeff4(dc_dab_raw, scale);
                    let dc_dbb = scale_coeff4(dc_dbb_raw, scale);
                    f_aa += exact::cell_second_derivative_from_moments(
                        cell,
                        &dc_da,
                        &dc_da,
                        &dc_daa,
                        &state.moments,
                    )?;
                    coeff_au[1] = dc_dab;
                    coeff_bu[1] = dc_dbb;
                }

                if let (Some(h_range), Some(runtime)) = (h_range, score_runtime) {
                    Self::for_each_deviation_basis_cubic_at(
                        runtime,
                        h_range,
                        z_mid,
                        "score-warp",
                        |_, idx, basis_span| {
                            coeff_u[idx] = scale_coeff4(
                                exact::score_basis_cell_coefficients(basis_span, b),
                                scale,
                            );
                            if need_hessian {
                                coeff_bu[idx] = scale_coeff4(
                                    exact::score_basis_cell_coefficients(basis_span, 1.0),
                                    scale,
                                );
                            }
                            Ok(())
                        },
                    )?;
                }

                if let (Some(w_range), Some(runtime)) = (w_range, link_runtime) {
                    Self::for_each_deviation_basis_cubic_at(
                        runtime,
                        w_range,
                        u_mid,
                        "link-wiggle",
                        |_, idx, basis_span| {
                            coeff_u[idx] = scale_coeff4(
                                exact::link_basis_cell_coefficients(basis_span, a, b),
                                scale,
                            );
                            if need_hessian {
                                let (dc_aw_raw, dc_bw_raw) =
                                    exact::link_basis_cell_coefficient_partials(basis_span, a, b);
                                coeff_au[idx] = scale_coeff4(dc_aw_raw, scale);
                                coeff_bu[idx] = scale_coeff4(dc_bw_raw, scale);
                            }
                            Ok(())
                        },
                    )?;
                }

                for u in 1..r {
                    f_u[u] +=
                        exact::cell_first_derivative_from_moments(&coeff_u[u], &state.moments)?;
                    if need_hessian {
                        f_au[u] += exact::cell_second_derivative_from_moments(
                            cell,
                            &dc_da,
                            &coeff_u[u],
                            &coeff_au[u],
                            &state.moments,
                        )?;
                    }
                }

                if need_hessian {
                    let coeff_jet = SparsePrimaryCoeffJetView::new(
                        1,
                        h_range,
                        w_range,
                        coeff_u.as_slice(),
                        coeff_au.as_slice(),
                        coeff_bu.as_slice(),
                        zero_family,
                        zero_family,
                        zero_family,
                        zero_family,
                        zero_family,
                        zero_family,
                        zero_family,
                    );
                    for u in 1..r {
                        for v in u..r {
                            let second_coeff = coeff_jet.pair_from_b_family(
                                coeff_jet.b_first,
                                u,
                                v,
                                COEFF_SUPPORT_BHW,
                            );
                            let val = exact::cell_second_derivative_from_moments(
                                cell,
                                &coeff_jet.first[u],
                                &coeff_jet.first[v],
                                &second_coeff,
                                &state.moments,
                            )?;
                            f_uv[[u, v]] += val;
                            if u != v {
                                f_uv[[v, u]] += val;
                            }
                        }
                    }
                }
            }
        }

        f_u[0] = -marginal.mu1;
        if need_hessian {
            f_uv[[0, 0]] = -marginal.mu2;
        }

        let a_u = &mut scratch.a_u;
        for u in 0..r {
            a_u[u] = -f_u[u] * inv_ma;
        }
        self.cache_row_intercept_predictor(row, a, q, b, beta_h, beta_w, a_u);
        let a_uv = &mut scratch.a_uv;
        if need_hessian {
            for u in 0..r {
                for v in u..r {
                    let val = -(f_uv[[u, v]]
                        + f_au[u] * a_u[v]
                        + f_au[v] * a_u[u]
                        + f_aa * a_u[u] * a_u[v])
                        * inv_ma;
                    a_uv[[u, v]] = val;
                    a_uv[[v, u]] = val;
                }
            }
        }

        let z_obs = self.z[row];
        let u_obs = a + b * z_obs;
        let obs = self.observed_denested_cell_partials(row, a, b, beta_h, beta_w)?;
        let chi_obs = eval_coeff4_at(&obs.dc_da, z_obs);
        let eta_aa_obs = eval_coeff4_at(&obs.dc_daa, z_obs);
        let eta_val = eval_coeff4_at(&obs.coeff, z_obs);

        g_u_fixed.fill([0.0; 4]);
        g_au_fixed.fill([0.0; 4]);
        g_bu_fixed.fill([0.0; 4]);
        g_u_fixed[1] = obs.dc_db;
        g_au_fixed[1] = obs.dc_dab;
        g_bu_fixed[1] = obs.dc_dbb;
        if let (Some(h_range), Some(runtime)) = (h_range, score_runtime) {
            Self::for_each_deviation_basis_cubic_at(
                runtime,
                h_range,
                z_obs,
                "score-warp observed",
                |_, idx, basis_span| {
                    g_u_fixed[idx] =
                        scale_coeff4(exact::score_basis_cell_coefficients(basis_span, b), scale);
                    g_bu_fixed[idx] =
                        scale_coeff4(exact::score_basis_cell_coefficients(basis_span, 1.0), scale);
                    Ok(())
                },
            )?;
        }
        if let (Some(w_range), Some(runtime)) = (w_range, link_runtime) {
            Self::for_each_deviation_basis_cubic_at(
                runtime,
                w_range,
                u_obs,
                "link-wiggle observed",
                |_, idx, basis_span| {
                    g_u_fixed[idx] =
                        scale_coeff4(exact::link_basis_cell_coefficients(basis_span, a, b), scale);
                    let (dc_aw_raw, dc_bw_raw) =
                        exact::link_basis_cell_coefficient_partials(basis_span, a, b);
                    g_au_fixed[idx] = scale_coeff4(dc_aw_raw, scale);
                    g_bu_fixed[idx] = scale_coeff4(dc_bw_raw, scale);
                    Ok(())
                },
            )?;
        }
        let g_jet = SparsePrimaryCoeffJetView::new(
            1,
            h_range,
            w_range,
            g_u_fixed.as_slice(),
            g_au_fixed.as_slice(),
            g_bu_fixed.as_slice(),
            zero_family,
            zero_family,
            zero_family,
            zero_family,
            zero_family,
            zero_family,
            zero_family,
        );

        let rho = &mut scratch.rho;
        let tau = &mut scratch.tau;
        rho.fill(0.0);
        tau.fill(0.0);
        for u in 1..r {
            rho[u] = eval_coeff4_at(&g_jet.first[u], z_obs);
            tau[u] = eval_coeff4_at(&g_jet.a_first[u], z_obs);
        }

        let eta_u = &mut scratch.grad;
        for u in 0..r {
            eta_u[u] = chi_obs * a_u[u] + rho[u];
        }

        let signed_margin = s_y * eta_val;
        let (log_cdf, lambda) = signed_probit_logcdf_and_mills_ratio(signed_margin);
        let neglog_val = -w_i * log_cdf;
        let d1_m = -w_i * lambda;
        let d2_m = w_i * lambda * (signed_margin + lambda);

        if need_hessian {
            let hess = &mut scratch.hess;
            hess.fill(0.0);
            for u in 0..r {
                for v in u..r {
                    let r_uv = eval_coeff4_at(
                        &g_jet.pair_from_b_family(g_jet.b_first, u, v, COEFF_SUPPORT_BHW),
                        z_obs,
                    );
                    let eta_uv = chi_obs * a_uv[[u, v]]
                        + eta_aa_obs * a_u[u] * a_u[v]
                        + tau[u] * a_u[v]
                        + a_u[u] * tau[v]
                        + r_uv;
                    let val = d2_m * eta_u[u] * eta_u[v] + d1_m * s_y * eta_uv;
                    hess[[u, v]] = val;
                    hess[[v, u]] = val;
                }
            }
        }

        eta_u.mapv_inplace(|eu| d1_m * s_y * eu);
        Ok(neglog_val)
    }

    pub(super) fn primary_point_from_block_states(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        primary: &PrimarySlices,
    ) -> Result<Array1<f64>, String> {
        let mut point = Array1::<f64>::zeros(primary.total);
        point[primary.q] = block_states[0].eta[row];
        point[primary.logslope] = block_states[1].eta[row];
        if let Some(h_range) = primary.h.as_ref() {
            let score = self
                .score_block_state(block_states)?
                .ok_or_else(|| "missing score-warp beta".to_string())?;
            point
                .slice_mut(s![h_range.start..h_range.end])
                .assign(&score.beta);
        }
        if let Some(w_range) = primary.w.as_ref() {
            let beta_w = self
                .link_block_state(block_states)?
                .ok_or_else(|| "missing link deviation beta".to_string())?;
            point
                .slice_mut(s![w_range.start..w_range.end])
                .assign(&beta_w.beta);
        }
        Ok(point)
    }

    pub(super) fn primary_point_components(
        &self,
        point: &Array1<f64>,
        primary: &PrimarySlices,
    ) -> (f64, f64, Option<Array1<f64>>, Option<Array1<f64>>) {
        let beta_h = primary
            .h
            .as_ref()
            .map(|range| point.slice(s![range.start..range.end]).to_owned());
        let beta_w = primary
            .w
            .as_ref()
            .map(|range| point.slice(s![range.start..range.end]).to_owned());
        (point[primary.q], point[primary.logslope], beta_h, beta_w)
    }

    pub(super) fn observed_denested_cell_partials(
        &self,
        row: usize,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<ObservedDenestedCellPartials, String> {
        shared_observed_denested_cell_partials(
            self.z[row],
            a,
            b,
            self.score_warp.as_ref(),
            beta_h,
            self.link_dev.as_ref(),
            beta_w,
            self.probit_frailty_scale(),
        )
    }

    pub(super) fn observed_denested_cell_partials_at_z(
        &self,
        z_value: f64,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<ObservedDenestedCellPartials, String> {
        shared_observed_denested_cell_partials(
            z_value,
            a,
            b,
            self.score_warp.as_ref(),
            beta_h,
            self.link_dev.as_ref(),
            beta_w,
            self.probit_frailty_scale(),
        )
    }

    /// Third-derivative tensor contracted with direction `dir`:
    ///   out[k,l] = sum_m f_{klm} dir[m]
    /// Rigid path uses the closed-form kernel. The flexible de-nested
    /// transport path contracts the cell-moment kernel analytically.
    ///
    /// Keep this kernel row-local and single-threaded. Its production callers
    /// already parallelize the outer row reductions with Rayon (`row_iter` /
    /// chunk `into_par_iter()` folds), which avoids nested Rayon overhead for
    /// the small per-row matrices assembled here.
    pub(super) fn row_primary_third_contracted_recompute(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        dir: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        self.row_primary_third_contracted_recompute_with_moments(
            row,
            block_states,
            cache,
            row_ctx,
            dir,
        )
    }

    pub(super) fn row_primary_third_contracted_recompute_with_moments(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        dir: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        if !self.effective_flex_active(block_states)? {
            // Hit the per-cache uncontracted-tensor cache if it has already
            // been populated (typically by the first ψ-axis call in the
            // sweep, which forces the build). Direct lookup is `O(1)`; the
            // 32 ψ-axis sweep that consumes this method then pays the heavy
            // empirical-grid jet exactly once per row instead of once per
            // (row, axis) pair.
            let t = self.rigid_third_full_cached(block_states, cache, row)?;
            let m = contract_third_full(t, dir[0], dir[1]);
            let mut out = Array2::<f64>::zeros((2, 2));
            out[[0, 0]] = m[0][0];
            out[[0, 1]] = m[0][1];
            out[[1, 0]] = m[1][0];
            out[[1, 1]] = m[1][1];
            return Ok(out);
        }
        if dir.iter().all(|value| value.abs() <= 0.0) {
            return Ok(Array2::<f64>::zeros((
                cache.primary.total,
                cache.primary.total,
            )));
        }
        if !row_ctx.intercept.is_finite() || !row_ctx.m_a.is_finite() || row_ctx.m_a <= 0.0 {
            return Err(
                "non-finite flexible row context in third-order directional contraction"
                    .to_string(),
            );
        }
        use super::exact_kernel as exact;

        let primary = &cache.primary;
        let point = self.primary_point_from_block_states(row, block_states, primary)?;
        let (q, b, beta_h_owned, beta_w_owned) = self.primary_point_components(&point, primary);
        let beta_h = beta_h_owned.as_ref();
        let beta_w = beta_w_owned.as_ref();
        if let Some(grid) = self.latent_measure.empirical_grid_for_training_row(row)? {
            return self.empirical_flex_row_third_contracted_recompute(
                row, primary, q, b, beta_h, beta_w, row_ctx, dir, &grid,
            );
        }
        let a = row_ctx.intercept;
        let r = primary.total;
        let marginal = self.marginal_link_map(q)?;
        let h_range = primary.h.as_ref();
        let w_range = primary.w.as_ref();
        let score_runtime = self.score_warp.as_ref();
        let link_runtime = self.link_dev.as_ref();
        let scale = self.probit_frailty_scale();
        let zero_family = vec![[0.0; 4]; r];

        let mut f_a = 0.0;
        let mut f_aa = 0.0;
        let mut f_a_dir = 0.0;
        let mut f_aa_dir = 0.0;
        let mut f_u = Array1::<f64>::zeros(r);
        let mut f_au = Array1::<f64>::zeros(r);
        let mut f_au_dir = Array1::<f64>::zeros(r);
        let mut f_uv = Array2::<f64>::zeros((r, r));
        let mut f_uv_dir = Array2::<f64>::zeros((r, r));

        let owned_cells;
        let cells: &[CachedDenestedCellMoments] = if let Some(cached) = self
            .bundle_for_degree(block_states, cache, 15)?
            .and_then(|bundle| bundle.row(row, 15))
        {
            cached
        } else {
            let partitions = self.denested_partition_cells(a, b, beta_h, beta_w)?;
            owned_cells = partitions
                .into_iter()
                .map(|partition_cell| {
                    self.evaluate_cell_derivative_moments_lru(partition_cell.cell, 15)
                        .map(|state| CachedDenestedCellMoments {
                            partition_cell,
                            state,
                        })
                })
                .collect::<Result<Vec<_>, String>>()?;
            &owned_cells
        };
        for entry in cells {
            let partition_cell = entry.partition_cell;
            let cell = partition_cell.cell;
            let z_mid = exact::interval_probe_point(cell.left, cell.right)?;
            let u_mid = a + b * z_mid;
            let state = &entry.state;

            let (dc_da_raw, dc_db_raw) = exact::denested_cell_coefficient_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                b,
            );
            let (dc_daa_raw, dc_dab_raw, dc_dbb_raw) = exact::denested_cell_second_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                b,
            );
            let denested_third = exact::denested_cell_third_partials(partition_cell.link_span);
            let dc_da = scale_coeff4(dc_da_raw, scale);
            let dc_db = scale_coeff4(dc_db_raw, scale);
            let dc_daa = scale_coeff4(dc_daa_raw, scale);
            let dc_dab = scale_coeff4(dc_dab_raw, scale);
            let dc_dbb = scale_coeff4(dc_dbb_raw, scale);
            let dc_daab = scale_coeff4(denested_third.1, scale);
            let dc_dabb = scale_coeff4(denested_third.2, scale);
            let dc_dbbb = scale_coeff4(denested_third.3, scale);

            let mut coeff_u = vec![[0.0; 4]; r];
            let mut coeff_au = vec![[0.0; 4]; r];
            let mut coeff_bu = vec![[0.0; 4]; r];
            let mut coeff_aau = vec![[0.0; 4]; r];
            let mut coeff_abu = vec![[0.0; 4]; r];
            let mut coeff_bbu = vec![[0.0; 4]; r];

            coeff_u[1] = dc_db;
            coeff_au[1] = dc_dab;
            coeff_bu[1] = dc_dbb;
            coeff_aau[1] = dc_daab;
            coeff_abu[1] = dc_dabb;
            coeff_bbu[1] = dc_dbbb;

            if let (Some(h_range), Some(runtime)) = (h_range, score_runtime) {
                Self::for_each_deviation_basis_cubic_at(
                    runtime,
                    h_range,
                    z_mid,
                    "score-warp third-direction",
                    |_, idx, basis_span| {
                        coeff_u[idx] = scale_coeff4(
                            exact::score_basis_cell_coefficients(basis_span, b),
                            scale,
                        );
                        coeff_bu[idx] = scale_coeff4(
                            exact::score_basis_cell_coefficients(basis_span, 1.0),
                            scale,
                        );
                        Ok(())
                    },
                )?;
            }

            if let (Some(w_range), Some(runtime)) = (w_range, link_runtime) {
                Self::for_each_deviation_basis_cubic_at(
                    runtime,
                    w_range,
                    u_mid,
                    "link-wiggle third-direction",
                    |_, idx, basis_span| {
                        coeff_u[idx] = scale_coeff4(
                            exact::link_basis_cell_coefficients(basis_span, a, b),
                            scale,
                        );
                        let (dc_aw_raw, dc_bw_raw) =
                            exact::link_basis_cell_coefficient_partials(basis_span, a, b);
                        let (dc_aaw_raw, dc_abw_raw, dc_bbw_raw) =
                            exact::link_basis_cell_second_partials(basis_span, a, b);
                        coeff_au[idx] = scale_coeff4(dc_aw_raw, scale);
                        coeff_bu[idx] = scale_coeff4(dc_bw_raw, scale);
                        coeff_aau[idx] = scale_coeff4(dc_aaw_raw, scale);
                        coeff_abu[idx] = scale_coeff4(dc_abw_raw, scale);
                        coeff_bbu[idx] = scale_coeff4(dc_bbw_raw, scale);
                        Ok(())
                    },
                )?;
            }

            let coeff_jet = SparsePrimaryCoeffJetView::new(
                1,
                h_range,
                w_range,
                &coeff_u,
                &coeff_au,
                &coeff_bu,
                &coeff_aau,
                &coeff_abu,
                &coeff_bbu,
                &zero_family,
                &zero_family,
                &zero_family,
                &zero_family,
            );

            f_a += exact::cell_first_derivative_from_moments(&dc_da, &state.moments)?;
            f_aa += exact::cell_second_derivative_from_moments(
                cell,
                &dc_da,
                &dc_da,
                &dc_daa,
                &state.moments,
            )?;

            for u in 1..r {
                f_u[u] +=
                    exact::cell_first_derivative_from_moments(&coeff_jet.first[u], &state.moments)?;
                f_au[u] += exact::cell_second_derivative_from_moments(
                    cell,
                    &dc_da,
                    &coeff_jet.first[u],
                    &coeff_jet.a_first[u],
                    &state.moments,
                )?;
            }
            let coeff_dir = coeff_jet.directional_family(coeff_jet.first, dir, COEFF_SUPPORT_BHW);
            let coeff_a_dir =
                coeff_jet.directional_family(coeff_jet.a_first, dir, COEFF_SUPPORT_BW);
            let coeff_aa_dir =
                coeff_jet.directional_family(coeff_jet.aa_first, dir, COEFF_SUPPORT_BW);

            f_a_dir += exact::cell_second_derivative_from_moments(
                cell,
                &dc_da,
                &coeff_dir,
                &coeff_a_dir,
                &state.moments,
            )?;
            f_aa_dir += exact::cell_third_derivative_from_moments(
                cell,
                &dc_da,
                &dc_da,
                &coeff_dir,
                &dc_daa,
                &coeff_a_dir,
                &coeff_a_dir,
                &coeff_aa_dir,
                &state.moments,
            )?;

            let mut coeff_u_dir = vec![[0.0; 4]; r];
            let mut coeff_au_dir = vec![[0.0; 4]; r];
            for u in 1..r {
                coeff_u_dir[u] = coeff_jet.param_directional_from_b_family(
                    coeff_jet.b_first,
                    u,
                    dir,
                    COEFF_SUPPORT_BHW,
                );
                coeff_au_dir[u] = coeff_jet.param_directional_from_b_family(
                    coeff_jet.ab_first,
                    u,
                    dir,
                    COEFF_SUPPORT_BW,
                );
            }

            for u in 1..r {
                f_au_dir[u] += exact::cell_third_derivative_from_moments(
                    cell,
                    &dc_da,
                    &coeff_jet.first[u],
                    &coeff_dir,
                    &coeff_jet.a_first[u],
                    &coeff_a_dir,
                    &coeff_u_dir[u],
                    &coeff_au_dir[u],
                    &state.moments,
                )?;
            }

            for u in 1..r {
                for v in u..r {
                    let second_coeff =
                        coeff_jet.pair_from_b_family(coeff_jet.b_first, u, v, COEFF_SUPPORT_BHW);
                    let val = exact::cell_second_derivative_from_moments(
                        cell,
                        &coeff_jet.first[u],
                        &coeff_jet.first[v],
                        &second_coeff,
                        &state.moments,
                    )?;
                    f_uv[[u, v]] += val;
                    if u != v {
                        f_uv[[v, u]] += val;
                    }

                    let third_coeff = coeff_jet.pair_directional_from_bb_family(
                        coeff_jet.bb_first,
                        u,
                        v,
                        dir,
                        COEFF_SUPPORT_BW,
                    );
                    let dir_val = exact::cell_third_derivative_from_moments(
                        cell,
                        &coeff_jet.first[u],
                        &coeff_jet.first[v],
                        &coeff_dir,
                        &second_coeff,
                        &coeff_u_dir[u],
                        &coeff_u_dir[v],
                        &third_coeff,
                        &state.moments,
                    )?;
                    f_uv_dir[[u, v]] += dir_val;
                    if u != v {
                        f_uv_dir[[v, u]] += dir_val;
                    }
                }
            }
        }

        f_u[0] = -marginal.mu1;
        f_uv[[0, 0]] = -marginal.mu2;
        f_uv_dir[[0, 0]] = -dir[0] * marginal.mu3;

        let inv_f_a = 1.0 / f_a;
        let mut a_u = Array1::<f64>::zeros(r);
        for u in 0..r {
            a_u[u] = -f_u[u] * inv_f_a;
        }
        let mut a_uv = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let val =
                    -(f_uv[[u, v]] + f_au[u] * a_u[v] + f_au[v] * a_u[u] + f_aa * a_u[u] * a_u[v])
                        * inv_f_a;
                a_uv[[u, v]] = val;
                a_uv[[v, u]] = val;
            }
        }
        let a_dir = a_u.dot(dir);
        let a_u_dir = a_uv.dot(dir);
        let mut a_uv_dir = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let n_dir = f_uv_dir[[u, v]]
                    + f_au_dir[u] * a_u[v]
                    + f_au[u] * a_u_dir[v]
                    + f_au_dir[v] * a_u[u]
                    + f_au[v] * a_u_dir[u]
                    + f_aa_dir * a_u[u] * a_u[v]
                    + f_aa * (a_u_dir[u] * a_u[v] + a_u[u] * a_u_dir[v]);
                let val = -(n_dir + f_a_dir * a_uv[[u, v]]) * inv_f_a;
                a_uv_dir[[u, v]] = val;
                a_uv_dir[[v, u]] = val;
            }
        }

        let z_obs = self.z[row];
        let u_obs = a + b * z_obs;
        let obs = self.observed_denested_cell_partials(row, a, b, beta_h, beta_w)?;
        let eta_val = eval_coeff4_at(&obs.coeff, z_obs);

        let mut g_u_fixed = vec![[0.0; 4]; r];
        let mut g_au_fixed = vec![[0.0; 4]; r];
        let mut g_bu_fixed = vec![[0.0; 4]; r];
        let mut g_aau_fixed = vec![[0.0; 4]; r];
        let mut g_abu_fixed = vec![[0.0; 4]; r];
        let mut g_bbu_fixed = vec![[0.0; 4]; r];

        g_u_fixed[1] = obs.dc_db;
        g_au_fixed[1] = obs.dc_dab;
        g_bu_fixed[1] = obs.dc_dbb;
        g_aau_fixed[1] = obs.dc_daab;
        g_abu_fixed[1] = obs.dc_dabb;
        g_bbu_fixed[1] = obs.dc_dbbb;
        let scale = self.probit_frailty_scale();

        if let (Some(h_range), Some(runtime)) = (h_range, score_runtime) {
            Self::for_each_deviation_basis_cubic_at(
                runtime,
                h_range,
                z_obs,
                "score-warp third-direction observed",
                |_, idx, basis_span| {
                    g_u_fixed[idx] =
                        scale_coeff4(exact::score_basis_cell_coefficients(basis_span, b), scale);
                    g_bu_fixed[idx] =
                        scale_coeff4(exact::score_basis_cell_coefficients(basis_span, 1.0), scale);
                    Ok(())
                },
            )?;
        }

        if let (Some(w_range), Some(runtime)) = (w_range, link_runtime) {
            Self::for_each_deviation_basis_cubic_at(
                runtime,
                w_range,
                u_obs,
                "link-wiggle third-direction observed",
                |_, idx, basis_span| {
                    g_u_fixed[idx] =
                        scale_coeff4(exact::link_basis_cell_coefficients(basis_span, a, b), scale);
                    let (dc_aw_raw, dc_bw_raw) =
                        exact::link_basis_cell_coefficient_partials(basis_span, a, b);
                    let (dc_aaw_raw, dc_abw_raw, dc_bbw_raw) =
                        exact::link_basis_cell_second_partials(basis_span, a, b);
                    g_au_fixed[idx] = scale_coeff4(dc_aw_raw, scale);
                    g_bu_fixed[idx] = scale_coeff4(dc_bw_raw, scale);
                    g_aau_fixed[idx] = scale_coeff4(dc_aaw_raw, scale);
                    g_abu_fixed[idx] = scale_coeff4(dc_abw_raw, scale);
                    g_bbu_fixed[idx] = scale_coeff4(dc_bbw_raw, scale);
                    Ok(())
                },
            )?;
        }

        let g_jet = SparsePrimaryCoeffJetView::new(
            1,
            h_range,
            w_range,
            &g_u_fixed,
            &g_au_fixed,
            &g_bu_fixed,
            &g_aau_fixed,
            &g_abu_fixed,
            &g_bbu_fixed,
            &zero_family,
            &zero_family,
            &zero_family,
            &zero_family,
        );

        let g_a = eval_coeff4_at(&obs.dc_da, z_obs);
        let g_aa = eval_coeff4_at(&obs.dc_daa, z_obs);
        let g_aaa = eval_coeff4_at(&obs.dc_daaa, z_obs);
        let mut g_u = Array1::<f64>::zeros(r);
        let mut g_au = Array1::<f64>::zeros(r);
        let mut g_aau = Array1::<f64>::zeros(r);
        let mut g_uv = Array2::<f64>::zeros((r, r));
        let mut g_auv = Array2::<f64>::zeros((r, r));
        for u in 1..r {
            g_u[u] = eval_coeff4_at(&g_jet.first[u], z_obs);
            g_au[u] = eval_coeff4_at(&g_jet.a_first[u], z_obs);
            g_aau[u] = eval_coeff4_at(&g_jet.aa_first[u], z_obs);
        }
        for u in 1..r {
            for v in u..r {
                let second_coeff = g_jet.pair_from_b_family(g_jet.b_first, u, v, COEFF_SUPPORT_BHW);
                let val = eval_coeff4_at(&second_coeff, z_obs);
                g_uv[[u, v]] = val;
                g_uv[[v, u]] = val;

                let third_coeff = g_jet.pair_from_b_family(g_jet.ab_first, u, v, COEFF_SUPPORT_BW);
                let third_val = eval_coeff4_at(&third_coeff, z_obs);
                g_auv[[u, v]] = third_val;
                g_auv[[v, u]] = third_val;
            }
        }

        let mut g_u_dir_fixed = vec![[0.0; 4]; r];
        let mut g_au_dir_fixed = vec![[0.0; 4]; r];
        let g_dir_fixed = g_jet.directional_family(g_jet.first, dir, COEFF_SUPPORT_BHW);
        let g_a_dir_fixed = g_jet.directional_family(g_jet.a_first, dir, COEFF_SUPPORT_BW);
        let g_aa_dir_fixed = g_jet.directional_family(g_jet.aa_first, dir, COEFF_SUPPORT_BW);
        let g_dir = eval_coeff4_at(&g_dir_fixed, z_obs);
        let g_a_dir = eval_coeff4_at(&g_a_dir_fixed, z_obs);
        let g_aa_dir = eval_coeff4_at(&g_aa_dir_fixed, z_obs);

        for u in 1..r {
            g_u_dir_fixed[u] =
                g_jet.param_directional_from_b_family(g_jet.b_first, u, dir, COEFF_SUPPORT_BHW);
            g_au_dir_fixed[u] =
                g_jet.param_directional_from_b_family(g_jet.ab_first, u, dir, COEFF_SUPPORT_BW);
        }

        let mut g_u_dir = Array1::<f64>::zeros(r);
        let mut g_uv_dir = Array2::<f64>::zeros((r, r));
        for u in 1..r {
            g_u_dir[u] = eval_coeff4_at(&g_u_dir_fixed[u], z_obs);
        }
        for u in 1..r {
            for v in u..r {
                let third_coeff = g_jet.pair_directional_from_bb_family(
                    g_jet.bb_first,
                    u,
                    v,
                    dir,
                    COEFF_SUPPORT_BW,
                );
                let val = eval_coeff4_at(&third_coeff, z_obs);
                g_uv_dir[[u, v]] = val;
                g_uv_dir[[v, u]] = val;
            }
        }

        let eta_u = g_a * &a_u + &g_u;
        let mut eta_uv = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let val = g_a * a_uv[[u, v]]
                    + g_aa * a_u[u] * a_u[v]
                    + g_au[u] * a_u[v]
                    + g_au[v] * a_u[u]
                    + g_uv[[u, v]];
                eta_uv[[u, v]] = val;
                eta_uv[[v, u]] = val;
            }
        }
        let eta_dir = g_a * a_dir + g_dir;
        let eta_u_dir = eta_uv.dot(dir);
        let dg_a_dir = g_aa * a_dir + g_a_dir;
        let dg_aa_dir = g_aaa * a_dir + g_aa_dir;
        let mut dg_au_dir = Array1::<f64>::zeros(r);
        let mut dg_uv_dir = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            dg_au_dir[u] = g_aau[u] * a_dir + eval_coeff4_at(&g_au_dir_fixed[u], z_obs);
        }
        for u in 0..r {
            for v in u..r {
                let val = g_auv[[u, v]] * a_dir + g_uv_dir[[u, v]];
                dg_uv_dir[[u, v]] = val;
                dg_uv_dir[[v, u]] = val;
            }
        }

        let mut eta_uv_dir = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let val = dg_a_dir * a_uv[[u, v]]
                    + g_a * a_uv_dir[[u, v]]
                    + dg_aa_dir * a_u[u] * a_u[v]
                    + g_aa * (a_u_dir[u] * a_u[v] + a_u[u] * a_u_dir[v])
                    + dg_au_dir[u] * a_u[v]
                    + g_au[u] * a_u_dir[v]
                    + dg_au_dir[v] * a_u[u]
                    + g_au[v] * a_u_dir[u]
                    + dg_uv_dir[[u, v]];
                eta_uv_dir[[u, v]] = val;
                eta_uv_dir[[v, u]] = val;
            }
        }

        let y_i = self.y[row];
        let w_i = self.weights[row];
        let s_y = 2.0 * y_i - 1.0;
        let m = s_y * eta_val;
        let (k1, k2, k3, _) = signed_probit_neglog_derivatives_up_to_fourth(m, w_i)?;
        let u1 = s_y * k1;
        let u3 = s_y * k3;

        let mut out = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let val = u3 * eta_u[u] * eta_u[v] * eta_dir
                    + k2 * (eta_uv[[u, v]] * eta_dir
                        + eta_u[u] * eta_u_dir[v]
                        + eta_u[v] * eta_u_dir[u])
                    + u1 * eta_uv_dir[[u, v]];
                out[[u, v]] = val;
                out[[v, u]] = val;
            }
        }
        Ok(out)
    }

    #[inline]
    pub(super) fn coeff4_eval_adjoint(z: f64, scalar_adjoint: f64) -> [f64; 4] {
        let z2 = z * z;
        [
            scalar_adjoint,
            scalar_adjoint * z,
            scalar_adjoint * z2,
            scalar_adjoint * z2 * z,
        ]
    }

    #[inline]
    pub(super) fn add_coeff4_adjoint(target: &mut [f64; 4], source: &[f64; 4]) {
        for idx in 0..4 {
            target[idx] += source[idx];
        }
    }

    #[inline]
    pub(super) fn add_eval_directional_family_adjoint(
        jet: &SparsePrimaryCoeffJetView<'_>,
        family: &[[f64; 4]],
        support: CoeffSupport,
        z: f64,
        scalar_adjoint: f64,
        direction_adjoint: &mut [f64],
    ) {
        let coeff_adjoint = Self::coeff4_eval_adjoint(z, scalar_adjoint);
        jet.add_directional_family_adjoint(family, &coeff_adjoint, support, direction_adjoint);
    }

    #[inline]
    pub(super) fn add_eval_param_directional_adjoint(
        jet: &SparsePrimaryCoeffJetView<'_>,
        family: &[[f64; 4]],
        param: usize,
        support: CoeffSupport,
        z: f64,
        scalar_adjoint: f64,
        direction_adjoint: &mut [f64],
    ) {
        let coeff_adjoint = Self::coeff4_eval_adjoint(z, scalar_adjoint);
        jet.add_param_directional_from_b_family_adjoint(
            family,
            param,
            &coeff_adjoint,
            support,
            direction_adjoint,
        );
    }

    #[inline]
    pub(super) fn add_eval_pair_directional_adjoint(
        jet: &SparsePrimaryCoeffJetView<'_>,
        family: &[[f64; 4]],
        u: usize,
        v: usize,
        support: CoeffSupport,
        z: f64,
        scalar_adjoint: f64,
        direction_adjoint: &mut [f64],
    ) {
        let coeff_adjoint = Self::coeff4_eval_adjoint(z, scalar_adjoint);
        jet.add_pair_directional_from_bb_family_adjoint(
            family,
            u,
            v,
            &coeff_adjoint,
            support,
            direction_adjoint,
        );
    }

    pub(super) fn add_cell_second_direction_adjoint(
        cell: exact_kernel::DenestedCubicCell,
        first_r: &[f64; 4],
        moments: &[f64],
        scalar_adjoint: f64,
        first_s_adjoint: &mut [f64; 4],
        second_adjoint: &mut [f64; 4],
    ) -> Result<(), String> {
        if moments.len() < 10 {
            return Err(format!(
                "insufficient reduced moments for second-derivative adjoint: need 10, have {}",
                moments.len()
            ));
        }
        let scale = scalar_adjoint / std::f64::consts::TAU;
        let eta = [cell.c0, cell.c1, cell.c2, cell.c3];
        for k in 0..4 {
            second_adjoint[k] += scale * moments[k];
        }
        for s_idx in 0..4 {
            let mut eta_r_moment = 0.0;
            for (eta_idx, &eta_value) in eta.iter().enumerate() {
                for (r_idx, &r_value) in first_r.iter().enumerate() {
                    eta_r_moment += eta_value * r_value * moments[eta_idx + r_idx + s_idx];
                }
            }
            first_s_adjoint[s_idx] -= scale * eta_r_moment;
        }
        Ok(())
    }

    pub(super) fn add_cell_third_direction_adjoint(
        cell: exact_kernel::DenestedCubicCell,
        first_r: &[f64; 4],
        first_s: &[f64; 4],
        second_rs: &[f64; 4],
        moments: &[f64],
        scalar_adjoint: f64,
        first_t_adjoint: &mut [f64; 4],
        second_rt_adjoint: &mut [f64; 4],
        second_st_adjoint: &mut [f64; 4],
        third_rst_adjoint: &mut [f64; 4],
    ) -> Result<(), String> {
        if moments.len() < 16 {
            return Err(format!(
                "insufficient reduced moments for third-derivative adjoint: need 16, have {}",
                moments.len()
            ));
        }
        let scale = scalar_adjoint / std::f64::consts::TAU;
        let eta = [cell.c0, cell.c1, cell.c2, cell.c3];
        let mut eta_sq_minus_one = [0.0; 7];
        for (i, &eta_i) in eta.iter().enumerate() {
            for (j, &eta_j) in eta.iter().enumerate() {
                eta_sq_minus_one[i + j] += eta_i * eta_j;
            }
        }
        eta_sq_minus_one[0] -= 1.0;

        for k in 0..4 {
            third_rst_adjoint[k] += scale * moments[k];
        }
        for coeff_idx in 0..4 {
            let mut eta_s_moment = 0.0;
            let mut eta_r_moment = 0.0;
            for (eta_idx, &eta_value) in eta.iter().enumerate() {
                for basis_idx in 0..4 {
                    eta_s_moment +=
                        eta_value * first_s[basis_idx] * moments[eta_idx + coeff_idx + basis_idx];
                    eta_r_moment +=
                        eta_value * first_r[basis_idx] * moments[eta_idx + coeff_idx + basis_idx];
                }
            }
            second_rt_adjoint[coeff_idx] -= scale * eta_s_moment;
            second_st_adjoint[coeff_idx] -= scale * eta_r_moment;
        }
        for t_idx in 0..4 {
            let mut linear_second = 0.0;
            for (eta_idx, &eta_value) in eta.iter().enumerate() {
                for (second_idx, &second_value) in second_rs.iter().enumerate() {
                    linear_second +=
                        eta_value * second_value * moments[eta_idx + second_idx + t_idx];
                }
            }
            let mut cubic_product = 0.0;
            for (eta_idx, &eta_value) in eta_sq_minus_one.iter().enumerate() {
                for (r_idx, &r_value) in first_r.iter().enumerate() {
                    for (s_idx, &s_value) in first_s.iter().enumerate() {
                        cubic_product += eta_value
                            * r_value
                            * s_value
                            * moments[eta_idx + r_idx + s_idx + t_idx];
                    }
                }
            }
            first_t_adjoint[t_idx] += scale * (cubic_product - linear_second);
        }
        Ok(())
    }

    pub(super) fn row_primary_third_trace_gradient_with_moments(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        gram: &[f64],
    ) -> Result<Array1<f64>, String> {
        let primary = &cache.primary;
        let r = primary.total;
        if gram.len() != r * r {
            return Err(format!(
                "bernoulli marginal-slope row trace gram length {} != {}",
                gram.len(),
                r * r
            ));
        }

        if !self.effective_flex_active(block_states)? {
            let tensor = self.rigid_third_full_cached(block_states, cache, row)?;
            let mut grad = Array1::<f64>::zeros(r);
            for a_idx in 0..2 {
                for b_idx in 0..2 {
                    let weight = gram[a_idx * r + b_idx];
                    for dir_idx in 0..2 {
                        grad[dir_idx] += weight * tensor[a_idx][b_idx][dir_idx];
                    }
                }
            }
            return Ok(grad);
        }
        if !row_ctx.intercept.is_finite() || !row_ctx.m_a.is_finite() || row_ctx.m_a <= 0.0 {
            return Err(
                "non-finite flexible row context in third-order trace-gradient contraction"
                    .to_string(),
            );
        }

        let point = self.primary_point_from_block_states(row, block_states, primary)?;
        let (q, b, beta_h_owned, beta_w_owned) = self.primary_point_components(&point, primary);
        let beta_h = beta_h_owned.as_ref();
        let beta_w = beta_w_owned.as_ref();
        if let Some(grid) = self.latent_measure.empirical_grid_for_training_row(row)? {
            let mut grad = Array1::<f64>::zeros(r);
            for dir_idx in 0..r {
                let mut basis = Array1::<f64>::zeros(r);
                basis[dir_idx] = 1.0;
                let third = self.empirical_flex_row_third_contracted_recompute(
                    row, primary, q, b, beta_h, beta_w, row_ctx, &basis, &grid,
                )?;
                grad[dir_idx] = Self::row_primary_trace_contract(&third, gram);
            }
            return Ok(grad);
        }

        use super::exact_kernel as exact;

        let a = row_ctx.intercept;
        let marginal = self.marginal_link_map(q)?;
        let h_range = primary.h.as_ref();
        let w_range = primary.w.as_ref();
        let score_runtime = self.score_warp.as_ref();
        let link_runtime = self.link_dev.as_ref();
        let scale = self.probit_frailty_scale();
        let zero_family = vec![[0.0; 4]; r];

        let mut f_a = 0.0;
        let mut f_aa = 0.0;
        let mut f_u = Array1::<f64>::zeros(r);
        let mut f_au = Array1::<f64>::zeros(r);
        let mut f_uv = Array2::<f64>::zeros((r, r));

        let owned_cells;
        let cells: &[CachedDenestedCellMoments] = if let Some(cached) = self
            .bundle_for_degree(block_states, cache, 15)?
            .and_then(|bundle| bundle.row(row, 15))
        {
            cached
        } else {
            let partitions = self.denested_partition_cells(a, b, beta_h, beta_w)?;
            owned_cells = partitions
                .into_iter()
                .map(|partition_cell| {
                    self.evaluate_cell_derivative_moments_lru(partition_cell.cell, 15)
                        .map(|state| CachedDenestedCellMoments {
                            partition_cell,
                            state,
                        })
                })
                .collect::<Result<Vec<_>, String>>()?;
            &owned_cells
        };

        for entry in cells {
            let partition_cell = entry.partition_cell;
            let cell = partition_cell.cell;
            let z_mid = exact::interval_probe_point(cell.left, cell.right)?;
            let u_mid = a + b * z_mid;
            let state = &entry.state;

            let (dc_da_raw, dc_db_raw) = exact::denested_cell_coefficient_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                b,
            );
            let (dc_daa_raw, dc_dab_raw, dc_dbb_raw) = exact::denested_cell_second_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                b,
            );
            let denested_third = exact::denested_cell_third_partials(partition_cell.link_span);
            let dc_da = scale_coeff4(dc_da_raw, scale);
            let dc_db = scale_coeff4(dc_db_raw, scale);
            let dc_daa = scale_coeff4(dc_daa_raw, scale);
            let dc_dab = scale_coeff4(dc_dab_raw, scale);
            let dc_dbb = scale_coeff4(dc_dbb_raw, scale);
            let dc_daab = scale_coeff4(denested_third.1, scale);
            let dc_dabb = scale_coeff4(denested_third.2, scale);
            let dc_dbbb = scale_coeff4(denested_third.3, scale);

            let mut coeff_u = vec![[0.0; 4]; r];
            let mut coeff_au = vec![[0.0; 4]; r];
            let mut coeff_bu = vec![[0.0; 4]; r];
            let mut coeff_aau = vec![[0.0; 4]; r];
            let mut coeff_abu = vec![[0.0; 4]; r];
            let mut coeff_bbu = vec![[0.0; 4]; r];

            coeff_u[1] = dc_db;
            coeff_au[1] = dc_dab;
            coeff_bu[1] = dc_dbb;
            coeff_aau[1] = dc_daab;
            coeff_abu[1] = dc_dabb;
            coeff_bbu[1] = dc_dbbb;

            if let (Some(h_range), Some(runtime)) = (h_range, score_runtime) {
                Self::for_each_deviation_basis_cubic_at(
                    runtime,
                    h_range,
                    z_mid,
                    "score-warp trace-gradient base",
                    |_, idx, basis_span| {
                        coeff_u[idx] = scale_coeff4(
                            exact::score_basis_cell_coefficients(basis_span, b),
                            scale,
                        );
                        coeff_bu[idx] = scale_coeff4(
                            exact::score_basis_cell_coefficients(basis_span, 1.0),
                            scale,
                        );
                        Ok(())
                    },
                )?;
            }

            if let (Some(w_range), Some(runtime)) = (w_range, link_runtime) {
                Self::for_each_deviation_basis_cubic_at(
                    runtime,
                    w_range,
                    u_mid,
                    "link-wiggle trace-gradient base",
                    |_, idx, basis_span| {
                        coeff_u[idx] = scale_coeff4(
                            exact::link_basis_cell_coefficients(basis_span, a, b),
                            scale,
                        );
                        let (dc_aw_raw, dc_bw_raw) =
                            exact::link_basis_cell_coefficient_partials(basis_span, a, b);
                        let (dc_aaw_raw, dc_abw_raw, dc_bbw_raw) =
                            exact::link_basis_cell_second_partials(basis_span, a, b);
                        coeff_au[idx] = scale_coeff4(dc_aw_raw, scale);
                        coeff_bu[idx] = scale_coeff4(dc_bw_raw, scale);
                        coeff_aau[idx] = scale_coeff4(dc_aaw_raw, scale);
                        coeff_abu[idx] = scale_coeff4(dc_abw_raw, scale);
                        coeff_bbu[idx] = scale_coeff4(dc_bbw_raw, scale);
                        Ok(())
                    },
                )?;
            }

            let coeff_jet = SparsePrimaryCoeffJetView::new(
                1,
                h_range,
                w_range,
                &coeff_u,
                &coeff_au,
                &coeff_bu,
                &coeff_aau,
                &coeff_abu,
                &coeff_bbu,
                &zero_family,
                &zero_family,
                &zero_family,
                &zero_family,
            );

            f_a += exact::cell_first_derivative_from_moments(&dc_da, &state.moments)?;
            f_aa += exact::cell_second_derivative_from_moments(
                cell,
                &dc_da,
                &dc_da,
                &dc_daa,
                &state.moments,
            )?;
            for u in 1..r {
                f_u[u] +=
                    exact::cell_first_derivative_from_moments(&coeff_jet.first[u], &state.moments)?;
                f_au[u] += exact::cell_second_derivative_from_moments(
                    cell,
                    &dc_da,
                    &coeff_jet.first[u],
                    &coeff_jet.a_first[u],
                    &state.moments,
                )?;
            }
            for u in 1..r {
                for v in u..r {
                    let second_coeff =
                        coeff_jet.pair_from_b_family(coeff_jet.b_first, u, v, COEFF_SUPPORT_BHW);
                    let val = exact::cell_second_derivative_from_moments(
                        cell,
                        &coeff_jet.first[u],
                        &coeff_jet.first[v],
                        &second_coeff,
                        &state.moments,
                    )?;
                    f_uv[[u, v]] += val;
                    if u != v {
                        f_uv[[v, u]] += val;
                    }
                }
            }
        }

        f_u[0] = -marginal.mu1;
        f_uv[[0, 0]] = -marginal.mu2;

        let inv_f_a = 1.0 / f_a;
        let mut a_u = Array1::<f64>::zeros(r);
        for u in 0..r {
            a_u[u] = -f_u[u] * inv_f_a;
        }
        let mut a_uv = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let val =
                    -(f_uv[[u, v]] + f_au[u] * a_u[v] + f_au[v] * a_u[u] + f_aa * a_u[u] * a_u[v])
                        * inv_f_a;
                a_uv[[u, v]] = val;
                a_uv[[v, u]] = val;
            }
        }

        let z_obs = self.z[row];
        let u_obs = a + b * z_obs;
        let obs = self.observed_denested_cell_partials(row, a, b, beta_h, beta_w)?;
        let eta_val = eval_coeff4_at(&obs.coeff, z_obs);

        let mut g_u_fixed = vec![[0.0; 4]; r];
        let mut g_au_fixed = vec![[0.0; 4]; r];
        let mut g_bu_fixed = vec![[0.0; 4]; r];
        let mut g_aau_fixed = vec![[0.0; 4]; r];
        let mut g_abu_fixed = vec![[0.0; 4]; r];
        let mut g_bbu_fixed = vec![[0.0; 4]; r];

        g_u_fixed[1] = obs.dc_db;
        g_au_fixed[1] = obs.dc_dab;
        g_bu_fixed[1] = obs.dc_dbb;
        g_aau_fixed[1] = obs.dc_daab;
        g_abu_fixed[1] = obs.dc_dabb;
        g_bbu_fixed[1] = obs.dc_dbbb;

        if let (Some(h_range), Some(runtime)) = (h_range, score_runtime) {
            Self::for_each_deviation_basis_cubic_at(
                runtime,
                h_range,
                z_obs,
                "score-warp trace-gradient observed",
                |_, idx, basis_span| {
                    g_u_fixed[idx] =
                        scale_coeff4(exact::score_basis_cell_coefficients(basis_span, b), scale);
                    g_bu_fixed[idx] =
                        scale_coeff4(exact::score_basis_cell_coefficients(basis_span, 1.0), scale);
                    Ok(())
                },
            )?;
        }

        if let (Some(w_range), Some(runtime)) = (w_range, link_runtime) {
            Self::for_each_deviation_basis_cubic_at(
                runtime,
                w_range,
                u_obs,
                "link-wiggle trace-gradient observed",
                |_, idx, basis_span| {
                    g_u_fixed[idx] =
                        scale_coeff4(exact::link_basis_cell_coefficients(basis_span, a, b), scale);
                    let (dc_aw_raw, dc_bw_raw) =
                        exact::link_basis_cell_coefficient_partials(basis_span, a, b);
                    let (dc_aaw_raw, dc_abw_raw, dc_bbw_raw) =
                        exact::link_basis_cell_second_partials(basis_span, a, b);
                    g_au_fixed[idx] = scale_coeff4(dc_aw_raw, scale);
                    g_bu_fixed[idx] = scale_coeff4(dc_bw_raw, scale);
                    g_aau_fixed[idx] = scale_coeff4(dc_aaw_raw, scale);
                    g_abu_fixed[idx] = scale_coeff4(dc_abw_raw, scale);
                    g_bbu_fixed[idx] = scale_coeff4(dc_bbw_raw, scale);
                    Ok(())
                },
            )?;
        }

        let g_jet = SparsePrimaryCoeffJetView::new(
            1,
            h_range,
            w_range,
            &g_u_fixed,
            &g_au_fixed,
            &g_bu_fixed,
            &g_aau_fixed,
            &g_abu_fixed,
            &g_bbu_fixed,
            &zero_family,
            &zero_family,
            &zero_family,
            &zero_family,
        );

        let g_a = eval_coeff4_at(&obs.dc_da, z_obs);
        let g_aa = eval_coeff4_at(&obs.dc_daa, z_obs);
        let g_aaa = eval_coeff4_at(&obs.dc_daaa, z_obs);
        let mut g_u = Array1::<f64>::zeros(r);
        let mut g_au = Array1::<f64>::zeros(r);
        let mut g_aau = Array1::<f64>::zeros(r);
        let mut g_uv = Array2::<f64>::zeros((r, r));
        let mut g_auv = Array2::<f64>::zeros((r, r));
        for u in 1..r {
            g_u[u] = eval_coeff4_at(&g_jet.first[u], z_obs);
            g_au[u] = eval_coeff4_at(&g_jet.a_first[u], z_obs);
            g_aau[u] = eval_coeff4_at(&g_jet.aa_first[u], z_obs);
        }
        for u in 1..r {
            for v in u..r {
                let second_coeff = g_jet.pair_from_b_family(g_jet.b_first, u, v, COEFF_SUPPORT_BHW);
                let val = eval_coeff4_at(&second_coeff, z_obs);
                g_uv[[u, v]] = val;
                g_uv[[v, u]] = val;

                let third_coeff = g_jet.pair_from_b_family(g_jet.ab_first, u, v, COEFF_SUPPORT_BW);
                let third_val = eval_coeff4_at(&third_coeff, z_obs);
                g_auv[[u, v]] = third_val;
                g_auv[[v, u]] = third_val;
            }
        }

        let eta_u = g_a * &a_u + &g_u;
        let mut eta_uv = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let val = g_a * a_uv[[u, v]]
                    + g_aa * a_u[u] * a_u[v]
                    + g_au[u] * a_u[v]
                    + g_au[v] * a_u[u]
                    + g_uv[[u, v]];
                eta_uv[[u, v]] = val;
                eta_uv[[v, u]] = val;
            }
        }

        let y_i = self.y[row];
        let w_i = self.weights[row];
        let s_y = 2.0 * y_i - 1.0;
        let m = s_y * eta_val;
        let (k1, k2, k3, _) = signed_probit_neglog_derivatives_up_to_fourth(m, w_i)?;
        let u1 = s_y * k1;
        let u3 = s_y * k3;

        let mut direction_adjoint = vec![0.0; r];
        let mut adj_eta_dir = 0.0;
        let mut adj_eta_u_dir = vec![0.0; r];
        let mut adj_a_u_dir = vec![0.0; r];
        let mut adj_a_uv_dir = Array2::<f64>::zeros((r, r));
        let mut adj_dg_a_dir = 0.0;
        let mut adj_dg_aa_dir = 0.0;
        let mut adj_dg_au_dir = vec![0.0; r];
        let mut adj_a_dir = 0.0;

        for u in 0..r {
            for v in u..r {
                let weight = if u == v {
                    gram[u * r + v]
                } else {
                    gram[u * r + v] + gram[v * r + u]
                };
                if weight == 0.0 {
                    continue;
                }
                adj_eta_dir += weight * (u3 * eta_u[u] * eta_u[v] + k2 * eta_uv[[u, v]]);
                adj_eta_u_dir[v] += weight * k2 * eta_u[u];
                adj_eta_u_dir[u] += weight * k2 * eta_u[v];

                let adj_eta_uv_dir = weight * u1;
                adj_dg_a_dir += adj_eta_uv_dir * a_uv[[u, v]];
                adj_a_uv_dir[[u, v]] += adj_eta_uv_dir * g_a;
                adj_dg_aa_dir += adj_eta_uv_dir * a_u[u] * a_u[v];
                adj_a_u_dir[u] += adj_eta_uv_dir * g_aa * a_u[v];
                adj_a_u_dir[v] += adj_eta_uv_dir * g_aa * a_u[u];
                adj_dg_au_dir[u] += adj_eta_uv_dir * a_u[v];
                adj_a_u_dir[v] += adj_eta_uv_dir * g_au[u];
                adj_dg_au_dir[v] += adj_eta_uv_dir * a_u[u];
                adj_a_u_dir[u] += adj_eta_uv_dir * g_au[v];

                adj_a_dir += adj_eta_uv_dir * g_auv[[u, v]];
                Self::add_eval_pair_directional_adjoint(
                    &g_jet,
                    g_jet.bb_first,
                    u,
                    v,
                    COEFF_SUPPORT_BW,
                    z_obs,
                    adj_eta_uv_dir,
                    &mut direction_adjoint,
                );
            }
        }

        for u in 0..r {
            let adj = adj_dg_au_dir[u];
            if adj != 0.0 {
                adj_a_dir += adj * g_aau[u];
                Self::add_eval_param_directional_adjoint(
                    &g_jet,
                    g_jet.ab_first,
                    u,
                    COEFF_SUPPORT_BW,
                    z_obs,
                    adj,
                    &mut direction_adjoint,
                );
            }
        }
        adj_a_dir += adj_eta_dir * g_a + adj_dg_a_dir * g_aa + adj_dg_aa_dir * g_aaa;
        Self::add_eval_directional_family_adjoint(
            &g_jet,
            g_jet.first,
            COEFF_SUPPORT_BHW,
            z_obs,
            adj_eta_dir,
            &mut direction_adjoint,
        );
        Self::add_eval_directional_family_adjoint(
            &g_jet,
            g_jet.a_first,
            COEFF_SUPPORT_BW,
            z_obs,
            adj_dg_a_dir,
            &mut direction_adjoint,
        );
        Self::add_eval_directional_family_adjoint(
            &g_jet,
            g_jet.aa_first,
            COEFF_SUPPORT_BW,
            z_obs,
            adj_dg_aa_dir,
            &mut direction_adjoint,
        );

        for u in 0..r {
            let adj = adj_eta_u_dir[u];
            if adj != 0.0 {
                for dir_idx in 0..r {
                    direction_adjoint[dir_idx] += adj * eta_uv[[u, dir_idx]];
                }
            }
        }

        let mut adj_f_a_dir = 0.0;
        let mut adj_f_aa_dir = 0.0;
        let mut adj_f_au_dir = vec![0.0; r];
        let mut adj_f_uv_dir = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let adj = adj_a_uv_dir[[u, v]];
                if adj == 0.0 {
                    continue;
                }
                let adj_n = -adj * inv_f_a;
                adj_f_uv_dir[[u, v]] += adj_n;
                adj_f_au_dir[u] += adj_n * a_u[v];
                adj_a_u_dir[v] += adj_n * f_au[u];
                adj_f_au_dir[v] += adj_n * a_u[u];
                adj_a_u_dir[u] += adj_n * f_au[v];
                adj_f_aa_dir += adj_n * a_u[u] * a_u[v];
                adj_a_u_dir[u] += adj_n * f_aa * a_u[v];
                adj_a_u_dir[v] += adj_n * f_aa * a_u[u];
                adj_f_a_dir += adj_n * a_uv[[u, v]];
            }
        }
        direction_adjoint[0] -= adj_f_uv_dir[[0, 0]] * marginal.mu3;

        for u in 0..r {
            let adj = adj_a_u_dir[u];
            if adj != 0.0 {
                for dir_idx in 0..r {
                    direction_adjoint[dir_idx] += adj * a_uv[[u, dir_idx]];
                }
            }
        }
        if adj_a_dir != 0.0 {
            for dir_idx in 0..r {
                direction_adjoint[dir_idx] += adj_a_dir * a_u[dir_idx];
            }
        }

        for entry in cells {
            let partition_cell = entry.partition_cell;
            let cell = partition_cell.cell;
            let z_mid = exact::interval_probe_point(cell.left, cell.right)?;
            let u_mid = a + b * z_mid;
            let state = &entry.state;

            let (dc_da_raw, dc_db_raw) = exact::denested_cell_coefficient_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                b,
            );
            let (dc_daa_raw, dc_dab_raw, dc_dbb_raw) = exact::denested_cell_second_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                b,
            );
            let denested_third = exact::denested_cell_third_partials(partition_cell.link_span);
            let dc_da = scale_coeff4(dc_da_raw, scale);
            let dc_db = scale_coeff4(dc_db_raw, scale);
            let dc_daa = scale_coeff4(dc_daa_raw, scale);
            let dc_dab = scale_coeff4(dc_dab_raw, scale);
            let dc_dbb = scale_coeff4(dc_dbb_raw, scale);
            let dc_daab = scale_coeff4(denested_third.1, scale);
            let dc_dabb = scale_coeff4(denested_third.2, scale);
            let dc_dbbb = scale_coeff4(denested_third.3, scale);

            let mut coeff_u = vec![[0.0; 4]; r];
            let mut coeff_au = vec![[0.0; 4]; r];
            let mut coeff_bu = vec![[0.0; 4]; r];
            let mut coeff_aau = vec![[0.0; 4]; r];
            let mut coeff_abu = vec![[0.0; 4]; r];
            let mut coeff_bbu = vec![[0.0; 4]; r];

            coeff_u[1] = dc_db;
            coeff_au[1] = dc_dab;
            coeff_bu[1] = dc_dbb;
            coeff_aau[1] = dc_daab;
            coeff_abu[1] = dc_dabb;
            coeff_bbu[1] = dc_dbbb;

            if let (Some(h_range), Some(runtime)) = (h_range, score_runtime) {
                Self::for_each_deviation_basis_cubic_at(
                    runtime,
                    h_range,
                    z_mid,
                    "score-warp trace-gradient adjoint",
                    |_, idx, basis_span| {
                        coeff_u[idx] = scale_coeff4(
                            exact::score_basis_cell_coefficients(basis_span, b),
                            scale,
                        );
                        coeff_bu[idx] = scale_coeff4(
                            exact::score_basis_cell_coefficients(basis_span, 1.0),
                            scale,
                        );
                        Ok(())
                    },
                )?;
            }

            if let (Some(w_range), Some(runtime)) = (w_range, link_runtime) {
                Self::for_each_deviation_basis_cubic_at(
                    runtime,
                    w_range,
                    u_mid,
                    "link-wiggle trace-gradient adjoint",
                    |_, idx, basis_span| {
                        coeff_u[idx] = scale_coeff4(
                            exact::link_basis_cell_coefficients(basis_span, a, b),
                            scale,
                        );
                        let (dc_aw_raw, dc_bw_raw) =
                            exact::link_basis_cell_coefficient_partials(basis_span, a, b);
                        let (dc_aaw_raw, dc_abw_raw, dc_bbw_raw) =
                            exact::link_basis_cell_second_partials(basis_span, a, b);
                        coeff_au[idx] = scale_coeff4(dc_aw_raw, scale);
                        coeff_bu[idx] = scale_coeff4(dc_bw_raw, scale);
                        coeff_aau[idx] = scale_coeff4(dc_aaw_raw, scale);
                        coeff_abu[idx] = scale_coeff4(dc_abw_raw, scale);
                        coeff_bbu[idx] = scale_coeff4(dc_bbw_raw, scale);
                        Ok(())
                    },
                )?;
            }

            let coeff_jet = SparsePrimaryCoeffJetView::new(
                1,
                h_range,
                w_range,
                &coeff_u,
                &coeff_au,
                &coeff_bu,
                &coeff_aau,
                &coeff_abu,
                &coeff_bbu,
                &zero_family,
                &zero_family,
                &zero_family,
                &zero_family,
            );

            let mut coeff_dir_adj = [0.0; 4];
            let mut coeff_a_dir_adj = [0.0; 4];
            let mut coeff_aa_dir_adj = [0.0; 4];
            let mut coeff_u_dir_adj = vec![[0.0; 4]; r];
            let mut coeff_au_dir_adj = vec![[0.0; 4]; r];

            if adj_f_a_dir != 0.0 {
                Self::add_cell_second_direction_adjoint(
                    cell,
                    &dc_da,
                    &state.moments,
                    adj_f_a_dir,
                    &mut coeff_dir_adj,
                    &mut coeff_a_dir_adj,
                )?;
            }
            if adj_f_aa_dir != 0.0 {
                let mut a_rt_adj = [0.0; 4];
                let mut a_st_adj = [0.0; 4];
                Self::add_cell_third_direction_adjoint(
                    cell,
                    &dc_da,
                    &dc_da,
                    &dc_daa,
                    &state.moments,
                    adj_f_aa_dir,
                    &mut coeff_dir_adj,
                    &mut a_rt_adj,
                    &mut a_st_adj,
                    &mut coeff_aa_dir_adj,
                )?;
                Self::add_coeff4_adjoint(&mut coeff_a_dir_adj, &a_rt_adj);
                Self::add_coeff4_adjoint(&mut coeff_a_dir_adj, &a_st_adj);
            }
            for u in 1..r {
                let adj = adj_f_au_dir[u];
                if adj == 0.0 {
                    continue;
                }
                Self::add_cell_third_direction_adjoint(
                    cell,
                    &dc_da,
                    &coeff_jet.first[u],
                    &coeff_jet.a_first[u],
                    &state.moments,
                    adj,
                    &mut coeff_dir_adj,
                    &mut coeff_a_dir_adj,
                    &mut coeff_u_dir_adj[u],
                    &mut coeff_au_dir_adj[u],
                )?;
            }
            for u in 1..r {
                for v in u..r {
                    let adj = adj_f_uv_dir[[u, v]];
                    if adj == 0.0 {
                        continue;
                    }
                    let second_coeff =
                        coeff_jet.pair_from_b_family(coeff_jet.b_first, u, v, COEFF_SUPPORT_BHW);
                    let mut u_dir_adj = [0.0; 4];
                    let mut v_dir_adj = [0.0; 4];
                    let mut third_coeff_adj = [0.0; 4];
                    Self::add_cell_third_direction_adjoint(
                        cell,
                        &coeff_jet.first[u],
                        &coeff_jet.first[v],
                        &second_coeff,
                        &state.moments,
                        adj,
                        &mut coeff_dir_adj,
                        &mut u_dir_adj,
                        &mut v_dir_adj,
                        &mut third_coeff_adj,
                    )?;
                    Self::add_coeff4_adjoint(&mut coeff_u_dir_adj[u], &u_dir_adj);
                    Self::add_coeff4_adjoint(&mut coeff_u_dir_adj[v], &v_dir_adj);
                    coeff_jet.add_pair_directional_from_bb_family_adjoint(
                        coeff_jet.bb_first,
                        u,
                        v,
                        &third_coeff_adj,
                        COEFF_SUPPORT_BW,
                        &mut direction_adjoint,
                    );
                }
            }

            coeff_jet.add_directional_family_adjoint(
                coeff_jet.first,
                &coeff_dir_adj,
                COEFF_SUPPORT_BHW,
                &mut direction_adjoint,
            );
            coeff_jet.add_directional_family_adjoint(
                coeff_jet.a_first,
                &coeff_a_dir_adj,
                COEFF_SUPPORT_BW,
                &mut direction_adjoint,
            );
            coeff_jet.add_directional_family_adjoint(
                coeff_jet.aa_first,
                &coeff_aa_dir_adj,
                COEFF_SUPPORT_BW,
                &mut direction_adjoint,
            );
            for u in 1..r {
                coeff_jet.add_param_directional_from_b_family_adjoint(
                    coeff_jet.b_first,
                    u,
                    &coeff_u_dir_adj[u],
                    COEFF_SUPPORT_BHW,
                    &mut direction_adjoint,
                );
                coeff_jet.add_param_directional_from_b_family_adjoint(
                    coeff_jet.ab_first,
                    u,
                    &coeff_au_dir_adj[u],
                    COEFF_SUPPORT_BW,
                    &mut direction_adjoint,
                );
            }
        }

        Ok(Array1::from_vec(direction_adjoint))
    }

    pub(super) fn row_primary_third_trace_many_with_moments(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        row_dirs: &[Array1<f64>],
        gram: &[f64],
    ) -> Result<Vec<f64>, String> {
        let primary = &cache.primary;
        let r = primary.total;
        if row_dirs.is_empty() {
            return Ok(Vec::new());
        }
        if gram.len() != r * r {
            return Err(format!(
                "bernoulli marginal-slope row trace gram length {} != {}",
                gram.len(),
                r * r
            ));
        }
        if let Some((idx, dir)) = row_dirs.iter().enumerate().find(|(_, dir)| dir.len() != r) {
            return Err(format!(
                "bernoulli marginal-slope row trace direction {idx} length {} != {r}",
                dir.len()
            ));
        }

        if row_dirs.len() > 1 {
            let trace_gradient = self.row_primary_third_trace_gradient_with_moments(
                row,
                block_states,
                cache,
                row_ctx,
                gram,
            )?;
            let traces = row_dirs
                .iter()
                .map(|dir| trace_gradient.dot(dir))
                .collect::<Vec<_>>();
            return Ok(traces);
        }

        if !self.effective_flex_active(block_states)? {
            let t = self.rigid_third_full_cached(block_states, cache, row)?;
            let mut traces = vec![0.0; row_dirs.len()];
            for (dir_idx, dir) in row_dirs.iter().enumerate() {
                let m = contract_third_full(t, dir[0], dir[1]);
                traces[dir_idx] = m[0][0] * gram[0]
                    + m[0][1] * gram[1]
                    + m[1][0] * gram[r]
                    + m[1][1] * gram[r + 1];
            }
            return Ok(traces);
        }
        if !row_ctx.intercept.is_finite() || !row_ctx.m_a.is_finite() || row_ctx.m_a <= 0.0 {
            return Err(
                "non-finite flexible row context in batched third-order trace contraction"
                    .to_string(),
            );
        }
        let point = self.primary_point_from_block_states(row, block_states, primary)?;
        let (q, b, beta_h_owned, beta_w_owned) = self.primary_point_components(&point, primary);
        let beta_h = beta_h_owned.as_ref();
        let beta_w = beta_w_owned.as_ref();
        let a = row_ctx.intercept;

        if let Some(grid) = self.latent_measure.empirical_grid_for_training_row(row)? {
            let mut traces = vec![0.0; row_dirs.len()];
            for (dir_idx, dir) in row_dirs.iter().enumerate() {
                let third = self.empirical_flex_row_third_contracted_recompute(
                    row, primary, q, b, beta_h, beta_w, row_ctx, dir, &grid,
                )?;
                traces[dir_idx] = Self::row_primary_trace_contract(&third, gram);
            }
            return Ok(traces);
        }

        use super::exact_kernel as exact;

        let marginal = self.marginal_link_map(q)?;
        let h_range = primary.h.as_ref();
        let w_range = primary.w.as_ref();
        let score_runtime = self.score_warp.as_ref();
        let link_runtime = self.link_dev.as_ref();
        let scale = self.probit_frailty_scale();
        let zero_family = vec![[0.0; 4]; r];
        let n_dirs = row_dirs.len();

        let mut f_a = 0.0;
        let mut f_aa = 0.0;
        let mut f_u = Array1::<f64>::zeros(r);
        let mut f_au = Array1::<f64>::zeros(r);
        let mut f_uv = Array2::<f64>::zeros((r, r));
        let mut f_a_dir = vec![0.0; n_dirs];
        let mut f_aa_dir = vec![0.0; n_dirs];
        let mut f_au_dir = vec![0.0; n_dirs * r];
        let mut f_uv_dir = vec![0.0; n_dirs * r * r];

        let owned_cells;
        let cells: &[CachedDenestedCellMoments] = if let Some(cached) = self
            .bundle_for_degree(block_states, cache, 15)?
            .and_then(|bundle| bundle.row(row, 15))
        {
            cached
        } else {
            let partitions = self.denested_partition_cells(a, b, beta_h, beta_w)?;
            owned_cells = partitions
                .into_iter()
                .map(|partition_cell| {
                    self.evaluate_cell_derivative_moments_lru(partition_cell.cell, 15)
                        .map(|state| CachedDenestedCellMoments {
                            partition_cell,
                            state,
                        })
                })
                .collect::<Result<Vec<_>, String>>()?;
            &owned_cells
        };

        for entry in cells {
            let partition_cell = entry.partition_cell;
            let cell = partition_cell.cell;
            let z_mid = exact::interval_probe_point(cell.left, cell.right)?;
            let u_mid = a + b * z_mid;
            let state = &entry.state;

            let (dc_da_raw, dc_db_raw) = exact::denested_cell_coefficient_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                b,
            );
            let (dc_daa_raw, dc_dab_raw, dc_dbb_raw) = exact::denested_cell_second_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                b,
            );
            let denested_third = exact::denested_cell_third_partials(partition_cell.link_span);
            let dc_da = scale_coeff4(dc_da_raw, scale);
            let dc_db = scale_coeff4(dc_db_raw, scale);
            let dc_daa = scale_coeff4(dc_daa_raw, scale);
            let dc_dab = scale_coeff4(dc_dab_raw, scale);
            let dc_dbb = scale_coeff4(dc_dbb_raw, scale);
            let dc_daab = scale_coeff4(denested_third.1, scale);
            let dc_dabb = scale_coeff4(denested_third.2, scale);
            let dc_dbbb = scale_coeff4(denested_third.3, scale);

            let mut coeff_u = vec![[0.0; 4]; r];
            let mut coeff_au = vec![[0.0; 4]; r];
            let mut coeff_bu = vec![[0.0; 4]; r];
            let mut coeff_aau = vec![[0.0; 4]; r];
            let mut coeff_abu = vec![[0.0; 4]; r];
            let mut coeff_bbu = vec![[0.0; 4]; r];

            coeff_u[1] = dc_db;
            coeff_au[1] = dc_dab;
            coeff_bu[1] = dc_dbb;
            coeff_aau[1] = dc_daab;
            coeff_abu[1] = dc_dabb;
            coeff_bbu[1] = dc_dbbb;

            if let (Some(h_range), Some(runtime)) = (h_range, score_runtime) {
                Self::for_each_deviation_basis_cubic_at(
                    runtime,
                    h_range,
                    z_mid,
                    "score-warp batched third-trace direction",
                    |_, idx, basis_span| {
                        coeff_u[idx] = scale_coeff4(
                            exact::score_basis_cell_coefficients(basis_span, b),
                            scale,
                        );
                        coeff_bu[idx] = scale_coeff4(
                            exact::score_basis_cell_coefficients(basis_span, 1.0),
                            scale,
                        );
                        Ok(())
                    },
                )?;
            }

            if let (Some(w_range), Some(runtime)) = (w_range, link_runtime) {
                Self::for_each_deviation_basis_cubic_at(
                    runtime,
                    w_range,
                    u_mid,
                    "link-wiggle batched third-trace direction",
                    |_, idx, basis_span| {
                        coeff_u[idx] = scale_coeff4(
                            exact::link_basis_cell_coefficients(basis_span, a, b),
                            scale,
                        );
                        let (dc_aw_raw, dc_bw_raw) =
                            exact::link_basis_cell_coefficient_partials(basis_span, a, b);
                        let (dc_aaw_raw, dc_abw_raw, dc_bbw_raw) =
                            exact::link_basis_cell_second_partials(basis_span, a, b);
                        coeff_au[idx] = scale_coeff4(dc_aw_raw, scale);
                        coeff_bu[idx] = scale_coeff4(dc_bw_raw, scale);
                        coeff_aau[idx] = scale_coeff4(dc_aaw_raw, scale);
                        coeff_abu[idx] = scale_coeff4(dc_abw_raw, scale);
                        coeff_bbu[idx] = scale_coeff4(dc_bbw_raw, scale);
                        Ok(())
                    },
                )?;
            }

            let coeff_jet = SparsePrimaryCoeffJetView::new(
                1,
                h_range,
                w_range,
                &coeff_u,
                &coeff_au,
                &coeff_bu,
                &coeff_aau,
                &coeff_abu,
                &coeff_bbu,
                &zero_family,
                &zero_family,
                &zero_family,
                &zero_family,
            );

            f_a += exact::cell_first_derivative_from_moments(&dc_da, &state.moments)?;
            f_aa += exact::cell_second_derivative_from_moments(
                cell,
                &dc_da,
                &dc_da,
                &dc_daa,
                &state.moments,
            )?;
            for u in 1..r {
                f_u[u] +=
                    exact::cell_first_derivative_from_moments(&coeff_jet.first[u], &state.moments)?;
                f_au[u] += exact::cell_second_derivative_from_moments(
                    cell,
                    &dc_da,
                    &coeff_jet.first[u],
                    &coeff_jet.a_first[u],
                    &state.moments,
                )?;
            }
            for u in 1..r {
                for v in u..r {
                    let second_coeff =
                        coeff_jet.pair_from_b_family(coeff_jet.b_first, u, v, COEFF_SUPPORT_BHW);
                    let val = exact::cell_second_derivative_from_moments(
                        cell,
                        &coeff_jet.first[u],
                        &coeff_jet.first[v],
                        &second_coeff,
                        &state.moments,
                    )?;
                    f_uv[[u, v]] += val;
                    if u != v {
                        f_uv[[v, u]] += val;
                    }
                }
            }

            for (dir_idx, dir) in row_dirs.iter().enumerate() {
                let coeff_dir =
                    coeff_jet.directional_family(coeff_jet.first, dir, COEFF_SUPPORT_BHW);
                let coeff_a_dir =
                    coeff_jet.directional_family(coeff_jet.a_first, dir, COEFF_SUPPORT_BW);
                let coeff_aa_dir =
                    coeff_jet.directional_family(coeff_jet.aa_first, dir, COEFF_SUPPORT_BW);

                f_a_dir[dir_idx] += exact::cell_second_derivative_from_moments(
                    cell,
                    &dc_da,
                    &coeff_dir,
                    &coeff_a_dir,
                    &state.moments,
                )?;
                f_aa_dir[dir_idx] += exact::cell_third_derivative_from_moments(
                    cell,
                    &dc_da,
                    &dc_da,
                    &coeff_dir,
                    &dc_daa,
                    &coeff_a_dir,
                    &coeff_a_dir,
                    &coeff_aa_dir,
                    &state.moments,
                )?;

                let mut coeff_u_dir = vec![[0.0; 4]; r];
                let mut coeff_au_dir = vec![[0.0; 4]; r];
                for u in 1..r {
                    coeff_u_dir[u] = coeff_jet.param_directional_from_b_family(
                        coeff_jet.b_first,
                        u,
                        dir,
                        COEFF_SUPPORT_BHW,
                    );
                    coeff_au_dir[u] = coeff_jet.param_directional_from_b_family(
                        coeff_jet.ab_first,
                        u,
                        dir,
                        COEFF_SUPPORT_BW,
                    );
                }

                for u in 1..r {
                    f_au_dir[dir_idx * r + u] += exact::cell_third_derivative_from_moments(
                        cell,
                        &dc_da,
                        &coeff_jet.first[u],
                        &coeff_dir,
                        &coeff_jet.a_first[u],
                        &coeff_a_dir,
                        &coeff_u_dir[u],
                        &coeff_au_dir[u],
                        &state.moments,
                    )?;
                }

                let dir_base = dir_idx * r * r;
                for u in 1..r {
                    for v in u..r {
                        let second_coeff = coeff_jet.pair_from_b_family(
                            coeff_jet.b_first,
                            u,
                            v,
                            COEFF_SUPPORT_BHW,
                        );
                        let third_coeff = coeff_jet.pair_directional_from_bb_family(
                            coeff_jet.bb_first,
                            u,
                            v,
                            dir,
                            COEFF_SUPPORT_BW,
                        );
                        let dir_val = exact::cell_third_derivative_from_moments(
                            cell,
                            &coeff_jet.first[u],
                            &coeff_jet.first[v],
                            &coeff_dir,
                            &second_coeff,
                            &coeff_u_dir[u],
                            &coeff_u_dir[v],
                            &third_coeff,
                            &state.moments,
                        )?;
                        f_uv_dir[dir_base + u * r + v] += dir_val;
                        if u != v {
                            f_uv_dir[dir_base + v * r + u] += dir_val;
                        }
                    }
                }
            }
        }

        f_u[0] = -marginal.mu1;
        f_uv[[0, 0]] = -marginal.mu2;

        let inv_f_a = 1.0 / f_a;
        let mut a_u = Array1::<f64>::zeros(r);
        for u in 0..r {
            a_u[u] = -f_u[u] * inv_f_a;
        }
        let mut a_uv = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let val =
                    -(f_uv[[u, v]] + f_au[u] * a_u[v] + f_au[v] * a_u[u] + f_aa * a_u[u] * a_u[v])
                        * inv_f_a;
                a_uv[[u, v]] = val;
                a_uv[[v, u]] = val;
            }
        }

        let z_obs = self.z[row];
        let u_obs = a + b * z_obs;
        let obs = self.observed_denested_cell_partials(row, a, b, beta_h, beta_w)?;
        let eta_val = eval_coeff4_at(&obs.coeff, z_obs);

        let mut g_u_fixed = vec![[0.0; 4]; r];
        let mut g_au_fixed = vec![[0.0; 4]; r];
        let mut g_bu_fixed = vec![[0.0; 4]; r];
        let mut g_aau_fixed = vec![[0.0; 4]; r];
        let mut g_abu_fixed = vec![[0.0; 4]; r];
        let mut g_bbu_fixed = vec![[0.0; 4]; r];

        g_u_fixed[1] = obs.dc_db;
        g_au_fixed[1] = obs.dc_dab;
        g_bu_fixed[1] = obs.dc_dbb;
        g_aau_fixed[1] = obs.dc_daab;
        g_abu_fixed[1] = obs.dc_dabb;
        g_bbu_fixed[1] = obs.dc_dbbb;

        if let (Some(h_range), Some(runtime)) = (h_range, score_runtime) {
            Self::for_each_deviation_basis_cubic_at(
                runtime,
                h_range,
                z_obs,
                "score-warp batched third-trace observed",
                |_, idx, basis_span| {
                    g_u_fixed[idx] =
                        scale_coeff4(exact::score_basis_cell_coefficients(basis_span, b), scale);
                    g_bu_fixed[idx] =
                        scale_coeff4(exact::score_basis_cell_coefficients(basis_span, 1.0), scale);
                    Ok(())
                },
            )?;
        }

        if let (Some(w_range), Some(runtime)) = (w_range, link_runtime) {
            Self::for_each_deviation_basis_cubic_at(
                runtime,
                w_range,
                u_obs,
                "link-wiggle batched third-trace observed",
                |_, idx, basis_span| {
                    g_u_fixed[idx] =
                        scale_coeff4(exact::link_basis_cell_coefficients(basis_span, a, b), scale);
                    let (dc_aw_raw, dc_bw_raw) =
                        exact::link_basis_cell_coefficient_partials(basis_span, a, b);
                    let (dc_aaw_raw, dc_abw_raw, dc_bbw_raw) =
                        exact::link_basis_cell_second_partials(basis_span, a, b);
                    g_au_fixed[idx] = scale_coeff4(dc_aw_raw, scale);
                    g_bu_fixed[idx] = scale_coeff4(dc_bw_raw, scale);
                    g_aau_fixed[idx] = scale_coeff4(dc_aaw_raw, scale);
                    g_abu_fixed[idx] = scale_coeff4(dc_abw_raw, scale);
                    g_bbu_fixed[idx] = scale_coeff4(dc_bbw_raw, scale);
                    Ok(())
                },
            )?;
        }

        let g_jet = SparsePrimaryCoeffJetView::new(
            1,
            h_range,
            w_range,
            &g_u_fixed,
            &g_au_fixed,
            &g_bu_fixed,
            &g_aau_fixed,
            &g_abu_fixed,
            &g_bbu_fixed,
            &zero_family,
            &zero_family,
            &zero_family,
            &zero_family,
        );

        let g_a = eval_coeff4_at(&obs.dc_da, z_obs);
        let g_aa = eval_coeff4_at(&obs.dc_daa, z_obs);
        let g_aaa = eval_coeff4_at(&obs.dc_daaa, z_obs);
        let mut g_u = Array1::<f64>::zeros(r);
        let mut g_au = Array1::<f64>::zeros(r);
        let mut g_aau = Array1::<f64>::zeros(r);
        let mut g_uv = Array2::<f64>::zeros((r, r));
        let mut g_auv = Array2::<f64>::zeros((r, r));
        for u in 1..r {
            g_u[u] = eval_coeff4_at(&g_jet.first[u], z_obs);
            g_au[u] = eval_coeff4_at(&g_jet.a_first[u], z_obs);
            g_aau[u] = eval_coeff4_at(&g_jet.aa_first[u], z_obs);
        }
        for u in 1..r {
            for v in u..r {
                let second_coeff = g_jet.pair_from_b_family(g_jet.b_first, u, v, COEFF_SUPPORT_BHW);
                let val = eval_coeff4_at(&second_coeff, z_obs);
                g_uv[[u, v]] = val;
                g_uv[[v, u]] = val;

                let third_coeff = g_jet.pair_from_b_family(g_jet.ab_first, u, v, COEFF_SUPPORT_BW);
                let third_val = eval_coeff4_at(&third_coeff, z_obs);
                g_auv[[u, v]] = third_val;
                g_auv[[v, u]] = third_val;
            }
        }

        let eta_u = g_a * &a_u + &g_u;
        let mut eta_uv = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let val = g_a * a_uv[[u, v]]
                    + g_aa * a_u[u] * a_u[v]
                    + g_au[u] * a_u[v]
                    + g_au[v] * a_u[u]
                    + g_uv[[u, v]];
                eta_uv[[u, v]] = val;
                eta_uv[[v, u]] = val;
            }
        }

        let y_i = self.y[row];
        let w_i = self.weights[row];
        let s_y = 2.0 * y_i - 1.0;
        let m = s_y * eta_val;
        let (k1, k2, k3, _) = signed_probit_neglog_derivatives_up_to_fourth(m, w_i)?;
        let u1 = s_y * k1;
        let u3 = s_y * k3;
        let mut traces = vec![0.0; n_dirs];

        for (dir_idx, dir) in row_dirs.iter().enumerate() {
            let dir_base = dir_idx * r * r;
            f_uv_dir[dir_base] = -dir[0] * marginal.mu3;

            let a_dir = a_u.dot(dir);
            let a_u_dir = a_uv.dot(dir);
            let g_dir_fixed = g_jet.directional_family(g_jet.first, dir, COEFF_SUPPORT_BHW);
            let g_a_dir_fixed = g_jet.directional_family(g_jet.a_first, dir, COEFF_SUPPORT_BW);
            let g_aa_dir_fixed = g_jet.directional_family(g_jet.aa_first, dir, COEFF_SUPPORT_BW);
            let g_dir = eval_coeff4_at(&g_dir_fixed, z_obs);
            let g_a_dir = eval_coeff4_at(&g_a_dir_fixed, z_obs);
            let g_aa_dir = eval_coeff4_at(&g_aa_dir_fixed, z_obs);
            let eta_dir = g_a * a_dir + g_dir;
            let eta_u_dir = eta_uv.dot(dir);
            let dg_a_dir = g_aa * a_dir + g_a_dir;
            let dg_aa_dir = g_aaa * a_dir + g_aa_dir;
            let mut dg_au_dir = Array1::<f64>::zeros(r);
            for u in 0..r {
                let coeff =
                    g_jet.param_directional_from_b_family(g_jet.ab_first, u, dir, COEFF_SUPPORT_BW);
                dg_au_dir[u] = g_aau[u] * a_dir + eval_coeff4_at(&coeff, z_obs);
            }

            let mut trace = 0.0;
            for u in 0..r {
                for v in u..r {
                    let fuvd = f_uv_dir[dir_base + u * r + v];
                    let n_dir = fuvd
                        + f_au_dir[dir_idx * r + u] * a_u[v]
                        + f_au[u] * a_u_dir[v]
                        + f_au_dir[dir_idx * r + v] * a_u[u]
                        + f_au[v] * a_u_dir[u]
                        + f_aa_dir[dir_idx] * a_u[u] * a_u[v]
                        + f_aa * (a_u_dir[u] * a_u[v] + a_u[u] * a_u_dir[v]);
                    let a_uv_dir = -(n_dir + f_a_dir[dir_idx] * a_uv[[u, v]]) * inv_f_a;
                    let third_coeff = g_jet.pair_directional_from_bb_family(
                        g_jet.bb_first,
                        u,
                        v,
                        dir,
                        COEFF_SUPPORT_BW,
                    );
                    let dg_uv_dir = g_auv[[u, v]] * a_dir + eval_coeff4_at(&third_coeff, z_obs);
                    let eta_uv_dir = dg_a_dir * a_uv[[u, v]]
                        + g_a * a_uv_dir
                        + dg_aa_dir * a_u[u] * a_u[v]
                        + g_aa * (a_u_dir[u] * a_u[v] + a_u[u] * a_u_dir[v])
                        + dg_au_dir[u] * a_u[v]
                        + g_au[u] * a_u_dir[v]
                        + dg_au_dir[v] * a_u[u]
                        + g_au[v] * a_u_dir[u]
                        + dg_uv_dir;
                    let val = u3 * eta_u[u] * eta_u[v] * eta_dir
                        + k2 * (eta_uv[[u, v]] * eta_dir
                            + eta_u[u] * eta_u_dir[v]
                            + eta_u[v] * eta_u_dir[u])
                        + u1 * eta_uv_dir;
                    if u == v {
                        trace += val * gram[u * r + v];
                    } else {
                        trace += val * (gram[u * r + v] + gram[v * r + u]);
                    }
                }
            }
            traces[dir_idx] = trace;
        }

        Ok(traces)
    }

    /// Fourth-derivative tensor contracted with two directions dir_u, dir_v:
    ///   out[k,l] = sum_{m,n} f_{klmn} dir_u[m] dir_v[n]
    /// Rigid path uses the closed-form kernel. The flexible de-nested
    /// transport path contracts the cell-moment kernel analytically.
    pub(super) fn row_primary_fourth_contracted_recompute_ordered(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        dir_u: &Array1<f64>,
        dir_v: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let flex_active = self.effective_flex_active(block_states)?;
        let expected_dir_len = if flex_active { cache.primary.total } else { 2 };
        if dir_u.len() != expected_dir_len || dir_v.len() != expected_dir_len {
            return Err(format!(
                "bernoulli fourth contracted row {row}: direction lengths ({},{}) != {expected_dir_len}",
                dir_u.len(),
                dir_v.len()
            ));
        }

        // Keep this row-local helper serial. All expensive callers parallelize
        // across independent rows/chunks so Rayon workers do not nest inside
        // the high-allocation per-row cell-kernel transport below.
        if !flex_active {
            // Hit the per-cache uncontracted-tensor cache (lazy-built on
            // first access) so the heavy per-row jet runs at most once per
            // row across all `(rank²+rank)/2` ψ-axis pairs, instead of
            // once per (row, pair). The outer-Hessian sweep is the dominant
            // consumer of this method.
            let t = self.rigid_fourth_full_cached(block_states, cache, row)?;
            let f = contract_fourth_full(t, dir_u[0], dir_u[1], dir_v[0], dir_v[1]);
            let mut out = Array2::<f64>::zeros((2, 2));
            out[[0, 0]] = f[0][0];
            out[[0, 1]] = f[0][1];
            out[[1, 0]] = f[1][0];
            out[[1, 1]] = f[1][1];
            return Ok(out);
        }
        if dir_u.iter().all(|value| *value == 0.0) || dir_v.iter().all(|value| *value == 0.0) {
            return Ok(Array2::<f64>::zeros((
                cache.primary.total,
                cache.primary.total,
            )));
        }
        if !row_ctx.intercept.is_finite() || !row_ctx.m_a.is_finite() || row_ctx.m_a <= 0.0 {
            return Err(
                "non-finite flexible row context in fourth-order directional contraction"
                    .to_string(),
            );
        }
        use super::exact_kernel as exact;

        let primary = &cache.primary;
        let point = self.primary_point_from_block_states(row, block_states, primary)?;
        let (q, b, beta_h_owned, beta_w_owned) = self.primary_point_components(&point, primary);
        let beta_h = beta_h_owned.as_ref();
        let beta_w = beta_w_owned.as_ref();
        if let Some(grid) = self.latent_measure.empirical_grid_for_training_row(row)? {
            return self.empirical_flex_row_fourth_contracted_recompute(
                row, primary, q, b, beta_h, beta_w, row_ctx, dir_u, dir_v, &grid,
            );
        }
        let a = row_ctx.intercept;
        let r = primary.total;
        let marginal = self.marginal_link_map(q)?;
        let h_range = primary.h.as_ref();
        let w_range = primary.w.as_ref();
        let score_runtime = self.score_warp.as_ref();
        let link_runtime = self.link_dev.as_ref();
        let scale = self.probit_frailty_scale();

        let mut f_a = 0.0;
        let mut f_aa = 0.0;
        let mut f_u = Array1::<f64>::zeros(r);
        let mut f_au = Array1::<f64>::zeros(r);
        let mut f_uv = Array2::<f64>::zeros((r, r));

        let mut f_a_u = 0.0;
        let mut f_aa_u = 0.0;
        let mut f_au_u = Array1::<f64>::zeros(r);
        let mut f_uv_u = Array2::<f64>::zeros((r, r));

        let mut f_a_v = 0.0;
        let mut f_aa_v = 0.0;
        let mut f_au_v = Array1::<f64>::zeros(r);
        let mut f_uv_v = Array2::<f64>::zeros((r, r));

        let mut f_a_uv = 0.0;
        let mut f_aa_uv = 0.0;
        let mut f_au_uv = Array1::<f64>::zeros(r);
        let mut f_uv_uv = Array2::<f64>::zeros((r, r));

        let owned_cells;
        let cells: &[CachedDenestedCellMoments] = if let Some(cached) = self
            .bundle_for_degree(block_states, cache, 21)?
            .and_then(|bundle| bundle.row(row, 21))
        {
            cached
        } else {
            let partitions = self.denested_partition_cells(a, b, beta_h, beta_w)?;
            owned_cells = partitions
                .into_iter()
                .map(|partition_cell| {
                    self.evaluate_cell_derivative_moments_lru(partition_cell.cell, 21)
                        .map(|state| CachedDenestedCellMoments {
                            partition_cell,
                            state,
                        })
                })
                .collect::<Result<Vec<_>, String>>()?;
            &owned_cells
        };
        for entry in cells {
            let partition_cell = entry.partition_cell;
            let cell = partition_cell.cell;
            let z_mid = exact::interval_probe_point(cell.left, cell.right)?;
            let u_mid = a + b * z_mid;
            let state = &entry.state;

            let (dc_da_raw, dc_db_raw) = exact::denested_cell_coefficient_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                b,
            );
            let (dc_daa_raw, dc_dab_raw, dc_dbb_raw) = exact::denested_cell_second_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                b,
            );
            let denested_third = exact::denested_cell_third_partials(partition_cell.link_span);
            let dc_da = scale_coeff4(dc_da_raw, scale);
            let dc_db = scale_coeff4(dc_db_raw, scale);
            let dc_daa = scale_coeff4(dc_daa_raw, scale);
            let dc_dab = scale_coeff4(dc_dab_raw, scale);
            let dc_dbb = scale_coeff4(dc_dbb_raw, scale);
            let dc_daab = scale_coeff4(denested_third.1, scale);
            let dc_dabb = scale_coeff4(denested_third.2, scale);
            let dc_dbbb = scale_coeff4(denested_third.3, scale);

            let mut coeff_u = vec![[0.0; 4]; r];
            let mut coeff_au = vec![[0.0; 4]; r];
            let mut coeff_bu = vec![[0.0; 4]; r];
            let mut coeff_aau = vec![[0.0; 4]; r];
            let mut coeff_abu = vec![[0.0; 4]; r];
            let mut coeff_bbu = vec![[0.0; 4]; r];
            let mut coeff_aaau = vec![[0.0; 4]; r];
            let mut coeff_aabu = vec![[0.0; 4]; r];
            let mut coeff_abbu = vec![[0.0; 4]; r];
            let mut coeff_bbbu = vec![[0.0; 4]; r];

            coeff_u[1] = dc_db;
            coeff_au[1] = dc_dab;
            coeff_bu[1] = dc_dbb;
            coeff_aau[1] = dc_daab;
            coeff_abu[1] = dc_dabb;
            coeff_bbu[1] = dc_dbbb;

            if let (Some(h_range), Some(runtime)) = (h_range, score_runtime) {
                Self::for_each_deviation_basis_cubic_at(
                    runtime,
                    h_range,
                    z_mid,
                    "score-warp fourth-direction",
                    |_, idx, basis_span| {
                        coeff_u[idx] = scale_coeff4(
                            exact::score_basis_cell_coefficients(basis_span, b),
                            scale,
                        );
                        coeff_bu[idx] = scale_coeff4(
                            exact::score_basis_cell_coefficients(basis_span, 1.0),
                            scale,
                        );
                        Ok(())
                    },
                )?;
            }

            if let (Some(w_range), Some(runtime)) = (w_range, link_runtime) {
                Self::for_each_deviation_basis_cubic_at(
                    runtime,
                    w_range,
                    u_mid,
                    "link-wiggle fourth-direction",
                    |_, idx, basis_span| {
                        coeff_u[idx] = scale_coeff4(
                            exact::link_basis_cell_coefficients(basis_span, a, b),
                            scale,
                        );
                        let (dc_aw_raw, dc_bw_raw) =
                            exact::link_basis_cell_coefficient_partials(basis_span, a, b);
                        let (dc_aaw_raw, dc_abw_raw, dc_bbw_raw) =
                            exact::link_basis_cell_second_partials(basis_span, a, b);
                        let (dc_aaaw, dc_aabw, dc_abbw, dc_bbbw) =
                            exact::link_basis_cell_third_partials(basis_span);
                        coeff_au[idx] = scale_coeff4(dc_aw_raw, scale);
                        coeff_bu[idx] = scale_coeff4(dc_bw_raw, scale);
                        coeff_aau[idx] = scale_coeff4(dc_aaw_raw, scale);
                        coeff_abu[idx] = scale_coeff4(dc_abw_raw, scale);
                        coeff_bbu[idx] = scale_coeff4(dc_bbw_raw, scale);
                        coeff_aaau[idx] = scale_coeff4(dc_aaaw, scale);
                        coeff_aabu[idx] = scale_coeff4(dc_aabw, scale);
                        coeff_abbu[idx] = scale_coeff4(dc_abbw, scale);
                        coeff_bbbu[idx] = scale_coeff4(dc_bbbw, scale);
                        Ok(())
                    },
                )?;
            }

            let coeff_jet = SparsePrimaryCoeffJetView::new(
                1,
                h_range,
                w_range,
                &coeff_u,
                &coeff_au,
                &coeff_bu,
                &coeff_aau,
                &coeff_abu,
                &coeff_bbu,
                &coeff_aaau,
                &coeff_aabu,
                &coeff_abbu,
                &coeff_bbbu,
            );

            f_a += exact::cell_first_derivative_from_moments(&dc_da, &state.moments)?;
            f_aa += exact::cell_second_derivative_from_moments(
                cell,
                &dc_da,
                &dc_da,
                &dc_daa,
                &state.moments,
            )?;

            for u in 1..r {
                f_u[u] +=
                    exact::cell_first_derivative_from_moments(&coeff_jet.first[u], &state.moments)?;
                f_au[u] += exact::cell_second_derivative_from_moments(
                    cell,
                    &dc_da,
                    &coeff_jet.first[u],
                    &coeff_jet.a_first[u],
                    &state.moments,
                )?;
            }
            let coeff_dir_u =
                coeff_jet.directional_family(coeff_jet.first, dir_u, COEFF_SUPPORT_BHW);
            let coeff_dir_v =
                coeff_jet.directional_family(coeff_jet.first, dir_v, COEFF_SUPPORT_BHW);
            let coeff_a_dir_u =
                coeff_jet.directional_family(coeff_jet.a_first, dir_u, COEFF_SUPPORT_BW);
            let coeff_a_dir_v =
                coeff_jet.directional_family(coeff_jet.a_first, dir_v, COEFF_SUPPORT_BW);
            let coeff_aa_dir_u =
                coeff_jet.directional_family(coeff_jet.aa_first, dir_u, COEFF_SUPPORT_BW);
            let coeff_aa_dir_v =
                coeff_jet.directional_family(coeff_jet.aa_first, dir_v, COEFF_SUPPORT_BW);

            f_a_u += exact::cell_second_derivative_from_moments(
                cell,
                &dc_da,
                &coeff_dir_u,
                &coeff_a_dir_u,
                &state.moments,
            )?;
            f_a_v += exact::cell_second_derivative_from_moments(
                cell,
                &dc_da,
                &coeff_dir_v,
                &coeff_a_dir_v,
                &state.moments,
            )?;
            f_aa_u += exact::cell_third_derivative_from_moments(
                cell,
                &dc_da,
                &dc_da,
                &coeff_dir_u,
                &dc_daa,
                &coeff_a_dir_u,
                &coeff_a_dir_u,
                &coeff_aa_dir_u,
                &state.moments,
            )?;
            f_aa_v += exact::cell_third_derivative_from_moments(
                cell,
                &dc_da,
                &dc_da,
                &coeff_dir_v,
                &dc_daa,
                &coeff_a_dir_v,
                &coeff_a_dir_v,
                &coeff_aa_dir_v,
                &state.moments,
            )?;

            let coeff_dir_uv = coeff_jet.mixed_directional_from_b_family(
                coeff_jet.b_first,
                dir_u,
                dir_v,
                COEFF_SUPPORT_BHW,
            );
            let coeff_a_dir_uv = coeff_jet.mixed_directional_from_b_family(
                coeff_jet.ab_first,
                dir_u,
                dir_v,
                COEFF_SUPPORT_BW,
            );
            let coeff_aa_dir_uv = coeff_jet.mixed_directional_from_b_family(
                coeff_jet.aab_first,
                dir_u,
                dir_v,
                COEFF_SUPPORT_W,
            );

            f_a_uv += exact::cell_third_derivative_from_moments(
                cell,
                &dc_da,
                &coeff_dir_u,
                &coeff_dir_v,
                &coeff_a_dir_u,
                &coeff_a_dir_v,
                &coeff_dir_uv,
                &coeff_a_dir_uv,
                &state.moments,
            )?;
            f_aa_uv += exact::cell_fourth_derivative_from_moments(
                cell,
                &dc_da,
                &dc_da,
                &coeff_dir_u,
                &coeff_dir_v,
                &dc_daa,
                &coeff_a_dir_u,
                &coeff_a_dir_v,
                &coeff_a_dir_u,
                &coeff_a_dir_v,
                &coeff_dir_uv,
                &coeff_aa_dir_u,
                &coeff_aa_dir_v,
                &coeff_a_dir_uv,
                &coeff_a_dir_uv,
                &coeff_aa_dir_uv,
                &state.moments,
            )?;

            let mut coeff_u_dir_u = vec![[0.0; 4]; r];
            let mut coeff_u_dir_v = vec![[0.0; 4]; r];
            let mut coeff_u_dir_uv = vec![[0.0; 4]; r];
            let mut coeff_au_dir_u = vec![[0.0; 4]; r];
            let mut coeff_au_dir_v = vec![[0.0; 4]; r];
            let mut coeff_au_dir_uv = vec![[0.0; 4]; r];
            for u in 1..r {
                coeff_u_dir_u[u] = coeff_jet.param_directional_from_b_family(
                    coeff_jet.b_first,
                    u,
                    dir_u,
                    COEFF_SUPPORT_BHW,
                );
                coeff_u_dir_v[u] = coeff_jet.param_directional_from_b_family(
                    coeff_jet.b_first,
                    u,
                    dir_v,
                    COEFF_SUPPORT_BHW,
                );
                coeff_u_dir_uv[u] = coeff_jet.param_mixed_from_bb_family(
                    coeff_jet.bb_first,
                    u,
                    dir_u,
                    dir_v,
                    COEFF_SUPPORT_BW,
                );
                coeff_au_dir_u[u] = coeff_jet.param_directional_from_b_family(
                    coeff_jet.ab_first,
                    u,
                    dir_u,
                    COEFF_SUPPORT_BW,
                );
                coeff_au_dir_v[u] = coeff_jet.param_directional_from_b_family(
                    coeff_jet.ab_first,
                    u,
                    dir_v,
                    COEFF_SUPPORT_BW,
                );
                coeff_au_dir_uv[u] = coeff_jet.param_mixed_from_bb_family(
                    coeff_jet.abb_first,
                    u,
                    dir_u,
                    dir_v,
                    COEFF_SUPPORT_W,
                );
            }

            for u in 1..r {
                f_au_u[u] += exact::cell_third_derivative_from_moments(
                    cell,
                    &dc_da,
                    &coeff_u[u],
                    &coeff_dir_u,
                    &coeff_au[u],
                    &coeff_a_dir_u,
                    &coeff_u_dir_u[u],
                    &coeff_au_dir_u[u],
                    &state.moments,
                )?;
                f_au_v[u] += exact::cell_third_derivative_from_moments(
                    cell,
                    &dc_da,
                    &coeff_u[u],
                    &coeff_dir_v,
                    &coeff_au[u],
                    &coeff_a_dir_v,
                    &coeff_u_dir_v[u],
                    &coeff_au_dir_v[u],
                    &state.moments,
                )?;
                f_au_uv[u] += exact::cell_fourth_derivative_from_moments(
                    cell,
                    &dc_da,
                    &coeff_u[u],
                    &coeff_dir_u,
                    &coeff_dir_v,
                    &coeff_au[u],
                    &coeff_a_dir_u,
                    &coeff_a_dir_v,
                    &coeff_u_dir_u[u],
                    &coeff_u_dir_v[u],
                    &coeff_dir_uv,
                    &coeff_au_dir_u[u],
                    &coeff_au_dir_v[u],
                    &coeff_a_dir_uv,
                    &coeff_u_dir_uv[u],
                    &coeff_au_dir_uv[u],
                    &state.moments,
                )?;
            }

            for u in 1..r {
                for v in u..r {
                    let second_coeff =
                        coeff_jet.pair_from_b_family(coeff_jet.b_first, u, v, COEFF_SUPPORT_BHW);
                    let base_val = exact::cell_second_derivative_from_moments(
                        cell,
                        &coeff_jet.first[u],
                        &coeff_jet.first[v],
                        &second_coeff,
                        &state.moments,
                    )?;
                    f_uv[[u, v]] += base_val;
                    if u != v {
                        f_uv[[v, u]] += base_val;
                    }

                    let third_u = coeff_jet.pair_directional_from_bb_family(
                        coeff_jet.bb_first,
                        u,
                        v,
                        dir_u,
                        COEFF_SUPPORT_BW,
                    );
                    let third_v = coeff_jet.pair_directional_from_bb_family(
                        coeff_jet.bb_first,
                        u,
                        v,
                        dir_v,
                        COEFF_SUPPORT_BW,
                    );
                    let fourth_uv = coeff_jet.pair_mixed_from_bbb_family(
                        coeff_jet.bbb_first,
                        u,
                        v,
                        dir_u,
                        dir_v,
                        COEFF_SUPPORT_W,
                    );

                    let dir_u_val = exact::cell_third_derivative_from_moments(
                        cell,
                        &coeff_jet.first[u],
                        &coeff_jet.first[v],
                        &coeff_dir_u,
                        &second_coeff,
                        &coeff_u_dir_u[u],
                        &coeff_u_dir_u[v],
                        &third_u,
                        &state.moments,
                    )?;
                    let dir_v_val = exact::cell_third_derivative_from_moments(
                        cell,
                        &coeff_jet.first[u],
                        &coeff_jet.first[v],
                        &coeff_dir_v,
                        &second_coeff,
                        &coeff_u_dir_v[u],
                        &coeff_u_dir_v[v],
                        &third_v,
                        &state.moments,
                    )?;
                    let mix_val = exact::cell_fourth_derivative_from_moments(
                        cell,
                        &coeff_jet.first[u],
                        &coeff_jet.first[v],
                        &coeff_dir_u,
                        &coeff_dir_v,
                        &second_coeff,
                        &coeff_u_dir_u[u],
                        &coeff_u_dir_v[u],
                        &coeff_u_dir_u[v],
                        &coeff_u_dir_v[v],
                        &coeff_dir_uv,
                        &third_u,
                        &third_v,
                        &coeff_u_dir_uv[u],
                        &coeff_u_dir_uv[v],
                        &fourth_uv,
                        &state.moments,
                    )?;
                    f_uv_u[[u, v]] += dir_u_val;
                    f_uv_v[[u, v]] += dir_v_val;
                    f_uv_uv[[u, v]] += mix_val;
                    if u != v {
                        f_uv_u[[v, u]] += dir_u_val;
                        f_uv_v[[v, u]] += dir_v_val;
                        f_uv_uv[[v, u]] += mix_val;
                    }
                }
            }
        }

        f_u[0] = -marginal.mu1;
        f_uv[[0, 0]] = -marginal.mu2;
        f_uv_u[[0, 0]] = -dir_u[0] * marginal.mu3;
        f_uv_v[[0, 0]] = -dir_v[0] * marginal.mu3;
        f_uv_uv[[0, 0]] = -dir_u[0] * dir_v[0] * marginal.mu4;

        let inv_f_a = 1.0 / f_a;
        let mut a_u = Array1::<f64>::zeros(r);
        for u in 0..r {
            a_u[u] = -f_u[u] * inv_f_a;
        }
        let mut a_uv = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let val =
                    -(f_uv[[u, v]] + f_au[u] * a_u[v] + f_au[v] * a_u[u] + f_aa * a_u[u] * a_u[v])
                        * inv_f_a;
                a_uv[[u, v]] = val;
                a_uv[[v, u]] = val;
            }
        }
        let a_u_dir_u = a_uv.dot(dir_u);
        let a_u_dir_v = a_uv.dot(dir_v);
        let mut a_uv_u = Array2::<f64>::zeros((r, r));
        let mut a_uv_v = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let n_u = f_uv_u[[u, v]]
                    + f_au_u[u] * a_u[v]
                    + f_au[u] * a_u_dir_u[v]
                    + f_au_u[v] * a_u[u]
                    + f_au[v] * a_u_dir_u[u]
                    + f_aa_u * a_u[u] * a_u[v]
                    + f_aa * (a_u_dir_u[u] * a_u[v] + a_u[u] * a_u_dir_u[v]);
                let val_u = -(n_u + f_a_u * a_uv[[u, v]]) * inv_f_a;
                a_uv_u[[u, v]] = val_u;
                a_uv_u[[v, u]] = val_u;

                let n_v = f_uv_v[[u, v]]
                    + f_au_v[u] * a_u[v]
                    + f_au[u] * a_u_dir_v[v]
                    + f_au_v[v] * a_u[u]
                    + f_au[v] * a_u_dir_v[u]
                    + f_aa_v * a_u[u] * a_u[v]
                    + f_aa * (a_u_dir_v[u] * a_u[v] + a_u[u] * a_u_dir_v[v]);
                let val_v = -(n_v + f_a_v * a_uv[[u, v]]) * inv_f_a;
                a_uv_v[[u, v]] = val_v;
                a_uv_v[[v, u]] = val_v;
            }
        }
        let a_u_uv = a_uv_u.dot(dir_v);
        let mut a_uv_uv = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let n_uv = f_uv_uv[[u, v]]
                    + f_au_uv[u] * a_u[v]
                    + f_au_u[u] * a_u_dir_v[v]
                    + f_au_v[u] * a_u_dir_u[v]
                    + f_au[u] * a_u_uv[v]
                    + f_au_uv[v] * a_u[u]
                    + f_au_u[v] * a_u_dir_v[u]
                    + f_au_v[v] * a_u_dir_u[u]
                    + f_au[v] * a_u_uv[u]
                    + f_aa_uv * a_u[u] * a_u[v]
                    + f_aa_u * (a_u_dir_v[u] * a_u[v] + a_u[u] * a_u_dir_v[v])
                    + f_aa_v * (a_u_dir_u[u] * a_u[v] + a_u[u] * a_u_dir_u[v])
                    + f_aa
                        * (a_u_uv[u] * a_u[v]
                            + a_u_dir_u[u] * a_u_dir_v[v]
                            + a_u_dir_v[u] * a_u_dir_u[v]
                            + a_u[u] * a_u_uv[v]);
                let val = -(n_uv
                    + f_a_v * a_uv_u[[u, v]]
                    + f_a_u * a_uv_v[[u, v]]
                    + f_a_uv * a_uv[[u, v]])
                    * inv_f_a;
                a_uv_uv[[u, v]] = val;
                a_uv_uv[[v, u]] = val;
            }
        }

        let z_obs = self.z[row];
        let u_obs = a + b * z_obs;
        let obs = self.observed_denested_cell_partials(row, a, b, beta_h, beta_w)?;
        let eta_val = eval_coeff4_at(&obs.coeff, z_obs);

        let mut g_u_fixed = vec![[0.0; 4]; r];
        let mut g_au_fixed = vec![[0.0; 4]; r];
        let mut g_bu_fixed = vec![[0.0; 4]; r];
        let mut g_aau_fixed = vec![[0.0; 4]; r];
        let mut g_abu_fixed = vec![[0.0; 4]; r];
        let mut g_bbu_fixed = vec![[0.0; 4]; r];
        let mut g_aaau_fixed = vec![[0.0; 4]; r];
        let mut g_aabu_fixed = vec![[0.0; 4]; r];
        let mut g_abbu_fixed = vec![[0.0; 4]; r];
        let mut g_bbbu_fixed = vec![[0.0; 4]; r];

        g_u_fixed[1] = obs.dc_db;
        g_au_fixed[1] = obs.dc_dab;
        g_bu_fixed[1] = obs.dc_dbb;
        g_aau_fixed[1] = obs.dc_daab;
        g_abu_fixed[1] = obs.dc_dabb;
        g_bbu_fixed[1] = obs.dc_dbbb;

        if let (Some(h_range), Some(runtime)) = (h_range, score_runtime) {
            Self::for_each_deviation_basis_cubic_at(
                runtime,
                h_range,
                z_obs,
                "score-warp fourth-direction observed",
                |_, idx, basis_span| {
                    g_u_fixed[idx] =
                        scale_coeff4(exact::score_basis_cell_coefficients(basis_span, b), scale);
                    g_bu_fixed[idx] =
                        scale_coeff4(exact::score_basis_cell_coefficients(basis_span, 1.0), scale);
                    Ok(())
                },
            )?;
        }
        if let (Some(w_range), Some(runtime)) = (w_range, link_runtime) {
            Self::for_each_deviation_basis_cubic_at(
                runtime,
                w_range,
                u_obs,
                "link-wiggle fourth-direction observed",
                |_, idx, basis_span| {
                    g_u_fixed[idx] =
                        scale_coeff4(exact::link_basis_cell_coefficients(basis_span, a, b), scale);
                    let (dc_aw_raw, dc_bw_raw) =
                        exact::link_basis_cell_coefficient_partials(basis_span, a, b);
                    let (dc_aaw_raw, dc_abw_raw, dc_bbw_raw) =
                        exact::link_basis_cell_second_partials(basis_span, a, b);
                    let (dc_aaaw, dc_aabw, dc_abbw, dc_bbbw) =
                        exact::link_basis_cell_third_partials(basis_span);
                    g_au_fixed[idx] = scale_coeff4(dc_aw_raw, scale);
                    g_bu_fixed[idx] = scale_coeff4(dc_bw_raw, scale);
                    g_aau_fixed[idx] = scale_coeff4(dc_aaw_raw, scale);
                    g_abu_fixed[idx] = scale_coeff4(dc_abw_raw, scale);
                    g_bbu_fixed[idx] = scale_coeff4(dc_bbw_raw, scale);
                    g_aaau_fixed[idx] = scale_coeff4(dc_aaaw, scale);
                    g_aabu_fixed[idx] = scale_coeff4(dc_aabw, scale);
                    g_abbu_fixed[idx] = scale_coeff4(dc_abbw, scale);
                    g_bbbu_fixed[idx] = scale_coeff4(dc_bbbw, scale);
                    Ok(())
                },
            )?;
        }

        let g_jet = SparsePrimaryCoeffJetView::new(
            1,
            h_range,
            w_range,
            &g_u_fixed,
            &g_au_fixed,
            &g_bu_fixed,
            &g_aau_fixed,
            &g_abu_fixed,
            &g_bbu_fixed,
            &g_aaau_fixed,
            &g_aabu_fixed,
            &g_abbu_fixed,
            &g_bbbu_fixed,
        );

        let g_a = eval_coeff4_at(&obs.dc_da, z_obs);
        let g_aa = eval_coeff4_at(&obs.dc_daa, z_obs);
        let g_aaa = eval_coeff4_at(&obs.dc_daaa, z_obs);
        let mut g_u = Array1::<f64>::zeros(r);
        let mut g_au = Array1::<f64>::zeros(r);
        let mut g_aau = Array1::<f64>::zeros(r);
        let mut g_aaau = Array1::<f64>::zeros(r);
        let mut g_uv = Array2::<f64>::zeros((r, r));
        let mut g_auv = Array2::<f64>::zeros((r, r));
        let mut g_aauv = Array2::<f64>::zeros((r, r));
        for u in 1..r {
            g_u[u] = eval_coeff4_at(&g_jet.first[u], z_obs);
            g_au[u] = eval_coeff4_at(&g_jet.a_first[u], z_obs);
            g_aau[u] = eval_coeff4_at(&g_jet.aa_first[u], z_obs);
            g_aaau[u] = eval_coeff4_at(&g_jet.aaa_first[u], z_obs);
        }
        for u in 1..r {
            for v in u..r {
                let second_coeff = g_jet.pair_from_b_family(g_jet.b_first, u, v, COEFF_SUPPORT_BHW);
                let val = eval_coeff4_at(&second_coeff, z_obs);
                g_uv[[u, v]] = val;
                g_uv[[v, u]] = val;

                let third_coeff = g_jet.pair_from_b_family(g_jet.ab_first, u, v, COEFF_SUPPORT_BW);
                let fourth_coeff = g_jet.pair_from_b_family(g_jet.aab_first, u, v, COEFF_SUPPORT_W);
                let third_val = eval_coeff4_at(&third_coeff, z_obs);
                let fourth_val = eval_coeff4_at(&fourth_coeff, z_obs);
                g_auv[[u, v]] = third_val;
                g_auv[[v, u]] = third_val;
                g_aauv[[u, v]] = fourth_val;
                g_aauv[[v, u]] = fourth_val;
            }
        }

        let g_dir_u_fixed = g_jet.directional_family(g_jet.first, dir_u, COEFF_SUPPORT_BHW);
        let g_dir_v_fixed = g_jet.directional_family(g_jet.first, dir_v, COEFF_SUPPORT_BHW);
        let g_a_dir_u_fixed = g_jet.directional_family(g_jet.a_first, dir_u, COEFF_SUPPORT_BW);
        let g_a_dir_v_fixed = g_jet.directional_family(g_jet.a_first, dir_v, COEFF_SUPPORT_BW);
        let g_aa_dir_u_fixed = g_jet.directional_family(g_jet.aa_first, dir_u, COEFF_SUPPORT_BW);
        let g_aa_dir_v_fixed = g_jet.directional_family(g_jet.aa_first, dir_v, COEFF_SUPPORT_BW);
        let g_dir_uv_fixed =
            g_jet.mixed_directional_from_b_family(g_jet.b_first, dir_u, dir_v, COEFF_SUPPORT_BHW);
        let g_a_dir_uv_fixed =
            g_jet.mixed_directional_from_b_family(g_jet.ab_first, dir_u, dir_v, COEFF_SUPPORT_BW);
        let g_aa_dir_uv_fixed =
            g_jet.mixed_directional_from_b_family(g_jet.aab_first, dir_u, dir_v, COEFF_SUPPORT_W);

        let g_dir_u = eval_coeff4_at(&g_dir_u_fixed, z_obs);
        let g_dir_v = eval_coeff4_at(&g_dir_v_fixed, z_obs);
        let g_dir_uv = eval_coeff4_at(&g_dir_uv_fixed, z_obs);
        let g_a_u_fixed = eval_coeff4_at(&g_a_dir_u_fixed, z_obs);
        let g_a_v_fixed = eval_coeff4_at(&g_a_dir_v_fixed, z_obs);
        let g_aa_u_fixed = eval_coeff4_at(&g_aa_dir_u_fixed, z_obs);
        let g_aa_v_fixed = eval_coeff4_at(&g_aa_dir_v_fixed, z_obs);
        let g_a_uv_fixed = eval_coeff4_at(&g_a_dir_uv_fixed, z_obs);
        let g_aa_uv_fixed = eval_coeff4_at(&g_aa_dir_uv_fixed, z_obs);

        let mut g_u_u_fixed = Array1::<f64>::zeros(r);
        let mut g_u_v_fixed = Array1::<f64>::zeros(r);
        let mut g_u_uv_fixed = Array1::<f64>::zeros(r);
        let mut g_au_u_fixed = Array1::<f64>::zeros(r);
        let mut g_au_v_fixed = Array1::<f64>::zeros(r);
        let mut g_au_uv_fixed = Array1::<f64>::zeros(r);
        let mut g_uv_u_fixed = Array2::<f64>::zeros((r, r));
        let mut g_uv_v_fixed = Array2::<f64>::zeros((r, r));
        let mut g_uv_uv_fixed = Array2::<f64>::zeros((r, r));
        let mut g_auv_u_fixed = Array2::<f64>::zeros((r, r));
        let mut g_auv_v_fixed = Array2::<f64>::zeros((r, r));

        for u in 1..r {
            let tmp_u =
                g_jet.param_directional_from_b_family(g_jet.b_first, u, dir_u, COEFF_SUPPORT_BHW);
            let tmp_v =
                g_jet.param_directional_from_b_family(g_jet.b_first, u, dir_v, COEFF_SUPPORT_BHW);
            let tmp_uv =
                g_jet.param_mixed_from_bb_family(g_jet.bb_first, u, dir_u, dir_v, COEFF_SUPPORT_BW);
            let tmp_au_u =
                g_jet.param_directional_from_b_family(g_jet.ab_first, u, dir_u, COEFF_SUPPORT_BW);
            let tmp_au_v =
                g_jet.param_directional_from_b_family(g_jet.ab_first, u, dir_v, COEFF_SUPPORT_BW);
            let tmp_au_uv =
                g_jet.param_mixed_from_bb_family(g_jet.abb_first, u, dir_u, dir_v, COEFF_SUPPORT_W);
            g_u_u_fixed[u] = eval_coeff4_at(&tmp_u, z_obs);
            g_u_v_fixed[u] = eval_coeff4_at(&tmp_v, z_obs);
            g_u_uv_fixed[u] = eval_coeff4_at(&tmp_uv, z_obs);
            g_au_u_fixed[u] = eval_coeff4_at(&tmp_au_u, z_obs);
            g_au_v_fixed[u] = eval_coeff4_at(&tmp_au_v, z_obs);
            g_au_uv_fixed[u] = eval_coeff4_at(&tmp_au_uv, z_obs);
        }
        for u in 1..r {
            for v in u..r {
                let third_u = g_jet.pair_directional_from_bb_family(
                    g_jet.bb_first,
                    u,
                    v,
                    dir_u,
                    COEFF_SUPPORT_BW,
                );
                let third_v = g_jet.pair_directional_from_bb_family(
                    g_jet.bb_first,
                    u,
                    v,
                    dir_v,
                    COEFF_SUPPORT_BW,
                );
                let fourth_uv = g_jet.pair_mixed_from_bbb_family(
                    g_jet.bbb_first,
                    u,
                    v,
                    dir_u,
                    dir_v,
                    COEFF_SUPPORT_W,
                );
                let a_third_u = g_jet.pair_directional_from_bb_family(
                    g_jet.abb_first,
                    u,
                    v,
                    dir_u,
                    COEFF_SUPPORT_W,
                );
                let a_third_v = g_jet.pair_directional_from_bb_family(
                    g_jet.abb_first,
                    u,
                    v,
                    dir_v,
                    COEFF_SUPPORT_W,
                );
                let vu = eval_coeff4_at(&third_u, z_obs);
                let vv = eval_coeff4_at(&third_v, z_obs);
                let vuv = eval_coeff4_at(&fourth_uv, z_obs);
                g_uv_u_fixed[[u, v]] = vu;
                g_uv_v_fixed[[u, v]] = vv;
                g_uv_uv_fixed[[u, v]] = vuv;
                g_uv_u_fixed[[v, u]] = vu;
                g_uv_v_fixed[[v, u]] = vv;
                g_uv_uv_fixed[[v, u]] = vuv;
                let atu = eval_coeff4_at(&a_third_u, z_obs);
                let atv = eval_coeff4_at(&a_third_v, z_obs);
                g_auv_u_fixed[[u, v]] = atu;
                g_auv_v_fixed[[u, v]] = atv;
                g_auv_u_fixed[[v, u]] = atu;
                g_auv_v_fixed[[v, u]] = atv;
            }
        }

        let eta_u = g_a * &a_u + &g_u;
        let mut eta_uv = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let val = g_a * a_uv[[u, v]]
                    + g_aa * a_u[u] * a_u[v]
                    + g_au[u] * a_u[v]
                    + g_au[v] * a_u[u]
                    + g_uv[[u, v]];
                eta_uv[[u, v]] = val;
                eta_uv[[v, u]] = val;
            }
        }

        let a_dir_u = a_u.dot(dir_u);
        let a_dir_v = a_u.dot(dir_v);
        let g_a_u = g_aa * a_dir_u + g_a_u_fixed;
        let g_a_v = g_aa * a_dir_v + g_a_v_fixed;
        let g_aa_u = g_aaa * a_dir_u + g_aa_u_fixed;
        let g_aa_v = g_aaa * a_dir_v + g_aa_v_fixed;

        let mut g_u_u = Array1::<f64>::zeros(r);
        let mut g_u_v = Array1::<f64>::zeros(r);
        let mut g_au_u = Array1::<f64>::zeros(r);
        let mut g_au_v = Array1::<f64>::zeros(r);
        for u in 0..r {
            g_u_u[u] = g_au[u] * a_dir_u + g_u_u_fixed[u];
            g_u_v[u] = g_au[u] * a_dir_v + g_u_v_fixed[u];
            g_au_u[u] = g_aau[u] * a_dir_u + g_au_u_fixed[u];
            g_au_v[u] = g_aau[u] * a_dir_v + g_au_v_fixed[u];
        }

        let mut eta_uv_u = Array2::<f64>::zeros((r, r));
        let mut eta_uv_v = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let g_uv_u = g_auv[[u, v]] * a_dir_u + g_uv_u_fixed[[u, v]];
                let g_uv_v = g_auv[[u, v]] * a_dir_v + g_uv_v_fixed[[u, v]];
                let val_u = g_a_u * a_uv[[u, v]]
                    + g_a * a_uv_u[[u, v]]
                    + g_aa_u * a_u[u] * a_u[v]
                    + g_aa * (a_u_dir_u[u] * a_u[v] + a_u[u] * a_u_dir_u[v])
                    + g_au_u[u] * a_u[v]
                    + g_au[u] * a_u_dir_u[v]
                    + g_au_u[v] * a_u[u]
                    + g_au[v] * a_u_dir_u[u]
                    + g_uv_u;
                eta_uv_u[[u, v]] = val_u;
                eta_uv_u[[v, u]] = val_u;

                let val_v = g_a_v * a_uv[[u, v]]
                    + g_a * a_uv_v[[u, v]]
                    + g_aa_v * a_u[u] * a_u[v]
                    + g_aa * (a_u_dir_v[u] * a_u[v] + a_u[u] * a_u_dir_v[v])
                    + g_au_v[u] * a_u[v]
                    + g_au[u] * a_u_dir_v[v]
                    + g_au_v[v] * a_u[u]
                    + g_au[v] * a_u_dir_v[u]
                    + g_uv_v;
                eta_uv_v[[u, v]] = val_v;
                eta_uv_v[[v, u]] = val_v;
            }
        }

        let a_dir_uv = a_u_dir_u.dot(dir_v);
        let g_a_uv = g_aaa * a_dir_u * a_dir_v
            + g_aa * a_dir_uv
            + g_aa_u_fixed * a_dir_v
            + g_aa_v_fixed * a_dir_u
            + g_a_uv_fixed;
        let g_aa_uv = g_aaau.dot(dir_u) * a_dir_v
            + g_aaau.dot(dir_v) * a_dir_u
            + g_aaa * a_dir_uv
            + g_aa_uv_fixed;
        let mut g_u_uv = Array1::<f64>::zeros(r);
        let mut g_au_uv = Array1::<f64>::zeros(r);
        for u in 0..r {
            g_u_uv[u] = g_aau[u] * a_dir_u * a_dir_v
                + g_au[u] * a_dir_uv
                + g_au_u_fixed[u] * a_dir_v
                + g_au_v_fixed[u] * a_dir_u
                + g_u_uv_fixed[u];
            let row_u_u = g_aauv.row(u).dot(dir_u);
            let row_u_v = g_aauv.row(u).dot(dir_v);
            g_au_uv[u] = g_aaau[u] * a_dir_u * a_dir_v
                + g_aau[u] * a_dir_uv
                + row_u_u * a_dir_v
                + row_u_v * a_dir_u
                + g_au_uv_fixed[u];
        }

        let mut eta_uv_uv = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let g_uv_uv = g_aauv[[u, v]] * a_dir_u * a_dir_v
                    + g_auv[[u, v]] * a_dir_uv
                    + g_auv_u_fixed[[u, v]] * a_dir_v
                    + g_auv_v_fixed[[u, v]] * a_dir_u
                    + g_uv_uv_fixed[[u, v]];
                let val = g_a_uv * a_uv[[u, v]]
                    + g_a_u * a_uv_v[[u, v]]
                    + g_a_v * a_uv_u[[u, v]]
                    + g_a * a_uv_uv[[u, v]]
                    + g_aa_uv * a_u[u] * a_u[v]
                    + g_aa_u * (a_u_dir_v[u] * a_u[v] + a_u[u] * a_u_dir_v[v])
                    + g_aa_v * (a_u_dir_u[u] * a_u[v] + a_u[u] * a_u_dir_u[v])
                    + g_aa
                        * (a_u_uv[u] * a_u[v]
                            + a_u_dir_u[u] * a_u_dir_v[v]
                            + a_u_dir_v[u] * a_u_dir_u[v]
                            + a_u[u] * a_u_uv[v])
                    + g_au_uv[u] * a_u[v]
                    + g_au_u[u] * a_u_dir_v[v]
                    + g_au_v[u] * a_u_dir_u[v]
                    + g_au[u] * a_u_uv[v]
                    + g_au_uv[v] * a_u[u]
                    + g_au_u[v] * a_u_dir_v[u]
                    + g_au_v[v] * a_u_dir_u[u]
                    + g_au[v] * a_u_uv[u]
                    + g_uv_uv;
                eta_uv_uv[[u, v]] = val;
                eta_uv_uv[[v, u]] = val;
            }
        }

        let eta_dir_u = g_a * a_dir_u + g_dir_u;
        let eta_dir_v = g_a * a_dir_v + g_dir_v;
        let eta_u_dir_u = eta_uv.dot(dir_u);
        let eta_u_dir_v = eta_uv.dot(dir_v);
        let eta_dir_uv = g_a_v * a_dir_u + g_a_u_fixed * a_dir_v + g_a * a_dir_uv + g_dir_uv;
        let eta_u_uv = eta_uv_u.dot(dir_v);

        let y_i = self.y[row];
        let w_i = self.weights[row];
        let s_y = 2.0 * y_i - 1.0;
        let m = s_y * eta_val;
        let (k1, k2, k3, k4) = signed_probit_neglog_derivatives_up_to_fourth(m, w_i)?;
        let u1 = s_y * k1;
        let u3 = s_y * k3;

        let mut out = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let a_term = eta_u[u] * eta_u[v] * eta_dir_u;
                let a_term_v = eta_u_dir_v[u] * eta_u[v] * eta_dir_u
                    + eta_u[u] * eta_u_dir_v[v] * eta_dir_u
                    + eta_u[u] * eta_u[v] * eta_dir_uv;
                let b_term = eta_uv_u[[u, v]];
                let b_term_v = eta_uv_uv[[u, v]];
                let c_term = eta_uv[[u, v]] * eta_dir_u
                    + eta_u[u] * eta_u_dir_u[v]
                    + eta_u[v] * eta_u_dir_u[u];
                let c_term_v = eta_uv_v[[u, v]] * eta_dir_u
                    + eta_uv[[u, v]] * eta_dir_uv
                    + eta_u_dir_v[u] * eta_u_dir_u[v]
                    + eta_u[u] * eta_u_uv[v]
                    + eta_u_dir_v[v] * eta_u_dir_u[u]
                    + eta_u[v] * eta_u_uv[u];
                let val = k4 * eta_dir_v * a_term
                    + u3 * a_term_v
                    + u3 * eta_dir_v * c_term
                    + k2 * c_term_v
                    + k2 * eta_dir_v * b_term
                    + u1 * b_term_v;
                out[[u, v]] = val;
                out[[v, u]] = val;
            }
        }
        Ok(out)
    }

    pub(super) fn row_primary_fourth_contracted_recompute(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        dir_u: &Array1<f64>,
        dir_v: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let ordered = self.row_primary_fourth_contracted_recompute_ordered(
            row,
            block_states,
            cache,
            row_ctx,
            dir_u,
            dir_v,
        )?;
        if !self.effective_flex_active(block_states)? {
            return Ok(ordered);
        }

        let swapped = self.row_primary_fourth_contracted_recompute_ordered(
            row,
            block_states,
            cache,
            row_ctx,
            dir_v,
            dir_u,
        )?;
        let mut sym = ordered;
        for i in 0..sym.nrows() {
            for j in 0..sym.ncols() {
                sym[[i, j]] = 0.5 * (sym[[i, j]] + swapped[[i, j]]);
            }
        }
        Ok(sym)
    }

    /// Like `add_pullback_primary_hessian` but only accumulates the h/w
    /// cross-block contributions. The marginal-marginal, marginal-logslope,
    /// and logslope-logslope blocks are handled by the weighted-Gram operator.
    pub(super) fn add_pullback_primary_hessian_hw_only(
        &self,
        target: &mut Array2<f64>,
        row: usize,
        slices: &BlockSlices,
        primary: &PrimarySlices,
        primary_hessian: ArrayView2<'_, f64>,
    ) {
        let h = primary_hessian;
        if let (Some(primary_h), Some(block_h)) = (primary.h.as_ref(), slices.h.as_ref()) {
            for (local_idx, global_idx) in block_h.clone().enumerate() {
                let h_q = h[[0, primary_h.start + local_idx]];
                if h_q != 0.0 {
                    {
                        let mut col = target.slice_mut(s![slices.marginal.clone(), global_idx]);
                        self.marginal_design
                            .axpy_row_into(row, h_q, &mut col)
                            .expect("marginal axpy column mismatch");
                    }
                    {
                        let mut row_view =
                            target.slice_mut(s![global_idx, slices.marginal.clone()]);
                        self.marginal_design
                            .axpy_row_into(row, h_q, &mut row_view)
                            .expect("marginal axpy row mismatch");
                    }
                }

                let h_g = h[[1, primary_h.start + local_idx]];
                if h_g != 0.0 {
                    {
                        let mut col = target.slice_mut(s![slices.logslope.clone(), global_idx]);
                        self.logslope_design
                            .axpy_row_into(row, h_g, &mut col)
                            .expect("logslope axpy column mismatch");
                    }
                    {
                        let mut row_view =
                            target.slice_mut(s![global_idx, slices.logslope.clone()]);
                        self.logslope_design
                            .axpy_row_into(row, h_g, &mut row_view)
                            .expect("logslope axpy row mismatch");
                    }
                }
            }

            target
                .slice_mut(s![block_h.clone(), block_h.clone()])
                .scaled_add(
                    1.0,
                    &h.slice(s![
                        primary_h.start..primary_h.end,
                        primary_h.start..primary_h.end
                    ]),
                );
        }

        if let (Some(primary_w), Some(block_w)) = (primary.w.as_ref(), slices.w.as_ref()) {
            for (local_idx, global_idx) in block_w.clone().enumerate() {
                let w_q = h[[0, primary_w.start + local_idx]];
                if w_q != 0.0 {
                    {
                        let mut col = target.slice_mut(s![slices.marginal.clone(), global_idx]);
                        self.marginal_design
                            .axpy_row_into(row, w_q, &mut col)
                            .expect("marginal axpy column mismatch");
                    }
                    {
                        let mut row_view =
                            target.slice_mut(s![global_idx, slices.marginal.clone()]);
                        self.marginal_design
                            .axpy_row_into(row, w_q, &mut row_view)
                            .expect("marginal axpy row mismatch");
                    }
                }

                let w_g = h[[1, primary_w.start + local_idx]];
                if w_g != 0.0 {
                    {
                        let mut col = target.slice_mut(s![slices.logslope.clone(), global_idx]);
                        self.logslope_design
                            .axpy_row_into(row, w_g, &mut col)
                            .expect("logslope axpy column mismatch");
                    }
                    {
                        let mut row_view =
                            target.slice_mut(s![global_idx, slices.logslope.clone()]);
                        self.logslope_design
                            .axpy_row_into(row, w_g, &mut row_view)
                            .expect("logslope axpy row mismatch");
                    }
                }
            }

            if let (Some(primary_h), Some(block_h)) = (primary.h.as_ref(), slices.h.as_ref()) {
                target
                    .slice_mut(s![block_h.clone(), block_w.clone()])
                    .scaled_add(
                        1.0,
                        &h.slice(s![
                            primary_h.start..primary_h.end,
                            primary_w.start..primary_w.end
                        ]),
                    );
                target
                    .slice_mut(s![block_w.clone(), block_h.clone()])
                    .scaled_add(
                        1.0,
                        &h.slice(s![
                            primary_w.start..primary_w.end,
                            primary_h.start..primary_h.end
                        ]),
                    );
            }

            target
                .slice_mut(s![block_w.clone(), block_w.clone()])
                .scaled_add(
                    1.0,
                    &h.slice(s![
                        primary_w.start..primary_w.end,
                        primary_w.start..primary_w.end
                    ]),
                );
        }
    }

    pub(super) fn exact_newton_joint_hessian_dense_from_cache(
        &self,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<Array2<f64>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let n = self.y.len();
        let started = std::time::Instant::now();
        let heartbeat_guard =
            crate::heartbeat::scope(format!("BMS dense-H build n={n} p={}", slices.total));
        let hessian_source = if cache.row_primary_hessians.is_some() {
            "row-primary-cache"
        } else {
            "row-stream"
        };
        if log_exact_work(n) {
            log::info!(
                "[BMS dense-H] build start n={} p={} source={} route=workspace-dense",
                n,
                slices.total,
                hessian_source
            );
        }
        let n_chunks = n.div_ceil(ROW_CHUNK_SIZE);
        let completed_chunks = AtomicUsize::new(0);
        let progress_step = (n_chunks / 10).max(1);
        let acc = (0..n_chunks)
            .into_par_iter()
            .try_fold(
                || BernoulliBlockHessianAccumulator::new(slices),
                |mut acc, chunk_idx| -> Result<_, String> {
                    let start = chunk_idx * ROW_CHUNK_SIZE;
                    let end = (start + ROW_CHUNK_SIZE).min(n);
                    let chunk_len = end - start;
                    let mut w_mm = Array1::<f64>::zeros(chunk_len);
                    let mut w_mg = Array1::<f64>::zeros(chunk_len);
                    let mut w_gg = Array1::<f64>::zeros(chunk_len);
                    let mut h_q = primary
                        .h
                        .as_ref()
                        .map(|range| Array2::<f64>::zeros((chunk_len, range.len())));
                    let mut h_g = primary
                        .h
                        .as_ref()
                        .map(|range| Array2::<f64>::zeros((chunk_len, range.len())));
                    let mut h_h = primary
                        .h
                        .as_ref()
                        .map(|range| Array2::<f64>::zeros((range.len(), range.len())));
                    let mut w_q = primary
                        .w
                        .as_ref()
                        .map(|range| Array2::<f64>::zeros((chunk_len, range.len())));
                    let mut w_g = primary
                        .w
                        .as_ref()
                        .map(|range| Array2::<f64>::zeros((chunk_len, range.len())));
                    let mut h_w = match (primary.h.as_ref(), primary.w.as_ref()) {
                        (Some(h_range), Some(w_range)) => {
                            Some(Array2::<f64>::zeros((h_range.len(), w_range.len())))
                        }
                        _ => None,
                    };
                    let mut w_w = primary
                        .w
                        .as_ref()
                        .map(|range| Array2::<f64>::zeros((range.len(), range.len())));
                    let mut scratch = BernoulliMarginalSlopeFlexRowScratch::new(primary.total);
                    for (local, row) in (start..end).enumerate() {
                        let hess_view =
                            if let Some(cached) = Self::cached_row_primary_hessian(cache, row) {
                                cached
                            } else {
                                let row_ctx = Self::row_ctx(cache, row);
                                let row_moments = cache
                                    .row_cell_moments
                                    .as_ref()
                                    .and_then(|bundle| bundle.row(row, 9));
                                self.compute_row_analytic_flex_into_with_moments(
                                    row,
                                    block_states,
                                    primary,
                                    row_ctx,
                                    row_moments,
                                    true,
                                    &mut scratch,
                                )?;
                                scratch.hess.view()
                            };
                        w_mm[local] = hess_view[[0, 0]];
                        w_mg[local] = hess_view[[0, 1]];
                        w_gg[local] = hess_view[[1, 1]];
                        if let Some(primary_h) = primary.h.as_ref() {
                            if let Some(ref mut hq) = h_q {
                                hq.row_mut(local)
                                    .assign(&hess_view.slice(s![0, primary_h.clone()]));
                            }
                            if let Some(ref mut hg) = h_g {
                                hg.row_mut(local)
                                    .assign(&hess_view.slice(s![1, primary_h.clone()]));
                            }
                            if let Some(ref mut hh) = h_h {
                                hh.scaled_add(
                                    1.0,
                                    &hess_view.slice(s![primary_h.clone(), primary_h.clone()]),
                                );
                            }
                        }
                        if let Some(primary_w) = primary.w.as_ref() {
                            if let Some(ref mut wq) = w_q {
                                wq.row_mut(local)
                                    .assign(&hess_view.slice(s![0, primary_w.clone()]));
                            }
                            if let Some(ref mut wg) = w_g {
                                wg.row_mut(local)
                                    .assign(&hess_view.slice(s![1, primary_w.clone()]));
                            }
                            if let Some(ref mut ww) = w_w {
                                ww.scaled_add(
                                    1.0,
                                    &hess_view.slice(s![primary_w.clone(), primary_w.clone()]),
                                );
                            }
                            if let (Some(primary_h), Some(ref mut hw)) =
                                (primary.h.as_ref(), h_w.as_mut())
                            {
                                hw.scaled_add(
                                    1.0,
                                    &hess_view.slice(s![primary_h.clone(), primary_w.clone()]),
                                );
                            }
                        }
                    }
                    acc.add_weighted_design_grams(self, start..end, &w_mm, &w_mg, &w_gg)?;
                    acc.add_weighted_hw_cross_terms(
                        self,
                        start..end,
                        slices,
                        h_q.as_ref(),
                        h_g.as_ref(),
                        h_h.as_ref(),
                        w_q.as_ref(),
                        w_g.as_ref(),
                        h_w.as_ref(),
                        w_w.as_ref(),
                    )?;
                    if log_exact_work(n) {
                        let done = completed_chunks.fetch_add(1, Ordering::Relaxed) + 1;
                        if done == n_chunks || done % progress_step == 0 {
                            log::info!(
                                "[BMS dense-H] progress chunks={}/{} rows={}/{} elapsed={:.3}s",
                                done,
                                n_chunks,
                                (done * ROW_CHUNK_SIZE).min(n),
                                n,
                                started.elapsed().as_secs_f64()
                            );
                        }
                    }
                    Ok(acc)
                },
            )
            .try_reduce(
                || BernoulliBlockHessianAccumulator::new(slices),
                |mut left, right| -> Result<_, String> {
                    left.add(&right);
                    Ok(left)
                },
            )?;
        let dense = acc.to_dense(slices);
        if log_exact_work(n) {
            log::info!(
                "[BMS dense-H] build done n={} p={} source={} route=workspace-dense elapsed={:.3}s",
                n,
                slices.total,
                hessian_source,
                started.elapsed().as_secs_f64()
            );
        }
        drop(heartbeat_guard);
        Ok(dense)
    }

    pub(super) fn exact_newton_joint_fused_gradient_dense_from_cache(
        &self,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<ExactNewtonJointFusedDenseEvaluation, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let n = self.y.len();
        let started = std::time::Instant::now();
        let heartbeat_guard = crate::heartbeat::scope(format!(
            "BMS fused exact-gradient+dense-H n={n} p={}",
            slices.total
        ));
        if log_exact_work(n) {
            log::info!(
                "[BMS fused exact-gradient+dense-H] eval start n={} p={} source=cache row_primary_hessian_cache={}",
                n,
                slices.total,
                cache.row_primary_hessians.is_some()
            );
        }
        let make_acc = || {
            (
                0.0_f64,
                Array1::<f64>::zeros(slices.marginal.len()),
                Array1::<f64>::zeros(slices.logslope.len()),
                slices
                    .h
                    .as_ref()
                    .map(|range| Array1::<f64>::zeros(range.len())),
                slices
                    .w
                    .as_ref()
                    .map(|range| Array1::<f64>::zeros(range.len())),
                BernoulliBlockHessianAccumulator::new(slices),
            )
        };
        let n_chunks = n.div_ceil(ROW_CHUNK_SIZE);
        let completed_chunks = AtomicUsize::new(0);
        let progress_step = (n_chunks / 10).max(1);
        let (log_likelihood, grad_marginal, grad_logslope, grad_h, grad_w, hessian_acc) =
            (0..n_chunks)
                .into_par_iter()
                .try_fold(make_acc, |mut acc, chunk_idx| -> Result<_, String> {
                    let start = chunk_idx * ROW_CHUNK_SIZE;
                    let end = (start + ROW_CHUNK_SIZE).min(n);
                    let chunk_len = end - start;
                    let mut w_mm = Array1::<f64>::zeros(chunk_len);
                    let mut w_mg = Array1::<f64>::zeros(chunk_len);
                    let mut w_gg = Array1::<f64>::zeros(chunk_len);
                    let mut h_q = primary
                        .h
                        .as_ref()
                        .map(|range| Array2::<f64>::zeros((chunk_len, range.len())));
                    let mut h_g = primary
                        .h
                        .as_ref()
                        .map(|range| Array2::<f64>::zeros((chunk_len, range.len())));
                    let mut h_h = primary
                        .h
                        .as_ref()
                        .map(|range| Array2::<f64>::zeros((range.len(), range.len())));
                    let mut w_q = primary
                        .w
                        .as_ref()
                        .map(|range| Array2::<f64>::zeros((chunk_len, range.len())));
                    let mut w_g = primary
                        .w
                        .as_ref()
                        .map(|range| Array2::<f64>::zeros((chunk_len, range.len())));
                    let mut h_w = match (primary.h.as_ref(), primary.w.as_ref()) {
                        (Some(h_range), Some(w_range)) => {
                            Some(Array2::<f64>::zeros((h_range.len(), w_range.len())))
                        }
                        _ => None,
                    };
                    let mut w_w = primary
                        .w
                        .as_ref()
                        .map(|range| Array2::<f64>::zeros((range.len(), range.len())));
                    let mut scratch = BernoulliMarginalSlopeFlexRowScratch::new(primary.total);
                    for (local, row) in (start..end).enumerate() {
                        // When both neglog+grad+hess are cached (Host variant),
                        // consume them directly — no second row kernel pass.
                        let cached_hessian;
                        let neglog;
                        // We need a stable place for the cached grad row.
                        let cached_grad_row_storage;
                        if let Some((cached_neglog, cached_grad_row)) =
                            Self::cached_row_primary_eval(cache, row)
                        {
                            neglog = cached_neglog;
                            // The cached grad row is an ArrayView1; we hold it.
                            cached_grad_row_storage = Some(cached_grad_row);
                            let cached_hess =
                                Self::cached_row_primary_hessian(cache, row);
                            cached_hessian = cached_hess;
                        } else {
                            // Cache miss (device-resident or no cache): run the
                            // row kernel once for neglog + grad + (maybe) hess.
                            cached_grad_row_storage = None;
                            let row_ctx = Self::row_ctx(cache, row);
                            let cached_hess =
                                Self::cached_row_primary_hessian(cache, row);
                            let row_moments = cache
                                .row_cell_moments
                                .as_ref()
                                .and_then(|bundle| bundle.row(row, 9));
                            let computed_neglog =
                                self.compute_row_analytic_flex_into_with_moments(
                                    row,
                                    block_states,
                                    primary,
                                    row_ctx,
                                    row_moments,
                                    cached_hess.is_none(),
                                    &mut scratch,
                                )?;
                            neglog = computed_neglog;
                            cached_hessian = cached_hess;
                        }
                        // Resolve grad source: cached row or scratch.grad.
                        let grad_ref: &dyn std::ops::Index<usize, Output = f64> =
                            if let Some(ref cgr) = cached_grad_row_storage {
                                cgr
                            } else {
                                &scratch.grad
                            };
                        acc.0 -= neglog;
                        {
                            let mut marginal = acc.1.view_mut();
                            self.marginal_design.axpy_row_into(
                                row,
                                Self::exact_newton_score_component_from_objective_gradient(
                                    grad_ref[0],
                                ),
                                &mut marginal,
                            )?;
                        }
                        {
                            let mut logslope = acc.2.view_mut();
                            self.logslope_design.axpy_row_into(
                                row,
                                Self::exact_newton_score_component_from_objective_gradient(
                                    grad_ref[1],
                                ),
                                &mut logslope,
                            )?;
                        }
                        if let (Some(primary_h), Some(grad_h)) =
                            (primary.h.as_ref(), acc.3.as_mut())
                        {
                            for idx in 0..primary_h.len() {
                                grad_h[idx] +=
                                    Self::exact_newton_score_component_from_objective_gradient(
                                        grad_ref[primary_h.start + idx],
                                    );
                            }
                        }
                        if let (Some(primary_w), Some(grad_w)) =
                            (primary.w.as_ref(), acc.4.as_mut())
                        {
                            for idx in 0..primary_w.len() {
                                grad_w[idx] +=
                                    Self::exact_newton_score_component_from_objective_gradient(
                                        grad_ref[primary_w.start + idx],
                                    );
                            }
                        }

                        let hess_view = cached_hessian.unwrap_or_else(|| scratch.hess.view());
                        w_mm[local] = hess_view[[0, 0]];
                        w_mg[local] = hess_view[[0, 1]];
                        w_gg[local] = hess_view[[1, 1]];
                        if let Some(primary_h) = primary.h.as_ref() {
                            if let Some(ref mut hq) = h_q {
                                hq.row_mut(local)
                                    .assign(&hess_view.slice(s![0, primary_h.clone()]));
                            }
                            if let Some(ref mut hg) = h_g {
                                hg.row_mut(local)
                                    .assign(&hess_view.slice(s![1, primary_h.clone()]));
                            }
                            if let Some(ref mut hh) = h_h {
                                hh.scaled_add(
                                    1.0,
                                    &hess_view.slice(s![primary_h.clone(), primary_h.clone()]),
                                );
                            }
                        }
                        if let Some(primary_w) = primary.w.as_ref() {
                            if let Some(ref mut wq) = w_q {
                                wq.row_mut(local)
                                    .assign(&hess_view.slice(s![0, primary_w.clone()]));
                            }
                            if let Some(ref mut wg) = w_g {
                                wg.row_mut(local)
                                    .assign(&hess_view.slice(s![1, primary_w.clone()]));
                            }
                            if let Some(ref mut ww) = w_w {
                                ww.scaled_add(
                                    1.0,
                                    &hess_view.slice(s![primary_w.clone(), primary_w.clone()]),
                                );
                            }
                            if let (Some(primary_h), Some(ref mut hw)) =
                                (primary.h.as_ref(), h_w.as_mut())
                            {
                                hw.scaled_add(
                                    1.0,
                                    &hess_view.slice(s![primary_h.clone(), primary_w.clone()]),
                                );
                            }
                        }
                    }
                    acc.5
                        .add_weighted_design_grams(self, start..end, &w_mm, &w_mg, &w_gg)?;
                    acc.5.add_weighted_hw_cross_terms(
                        self,
                        start..end,
                        slices,
                        h_q.as_ref(),
                        h_g.as_ref(),
                        h_h.as_ref(),
                        w_q.as_ref(),
                        w_g.as_ref(),
                        h_w.as_ref(),
                        w_w.as_ref(),
                    )?;
                    if log_exact_work(n) {
                        let done = completed_chunks.fetch_add(1, Ordering::Relaxed) + 1;
                        if done == n_chunks || done % progress_step == 0 {
                            log::info!(
                                "[BMS fused exact-gradient+dense-H] progress chunks={}/{} rows={}/{} elapsed={:.3}s",
                                done,
                                n_chunks,
                                (done * ROW_CHUNK_SIZE).min(n),
                                n,
                                started.elapsed().as_secs_f64()
                            );
                        }
                    }
                    Ok(acc)
                })
                .try_reduce(make_acc, |mut left, right| -> Result<_, String> {
                    left.0 += right.0;
                    left.1 += &right.1;
                    left.2 += &right.2;
                    if let (Some(lhs), Some(rhs)) = (left.3.as_mut(), right.3.as_ref()) {
                        *lhs += rhs;
                    }
                    if let (Some(lhs), Some(rhs)) = (left.4.as_mut(), right.4.as_ref()) {
                        *lhs += rhs;
                    }
                    left.5.add(&right.5);
                    Ok(left)
                })?;

        let mut gradient = Array1::<f64>::zeros(slices.total);
        gradient
            .slice_mut(s![slices.marginal.clone()])
            .assign(&grad_marginal);
        gradient
            .slice_mut(s![slices.logslope.clone()])
            .assign(&grad_logslope);
        if let (Some(range), Some(grad_h)) = (slices.h.as_ref(), grad_h.as_ref()) {
            gradient.slice_mut(s![range.clone()]).assign(grad_h);
        }
        if let (Some(range), Some(grad_w)) = (slices.w.as_ref(), grad_w.as_ref()) {
            gradient.slice_mut(s![range.clone()]).assign(grad_w);
        }
        let hessian = hessian_acc.to_dense(slices);
        if log_exact_work(n) {
            log::info!(
                "[BMS fused exact-gradient+dense-H] eval done n={} p={} source=cache elapsed={:.3}s",
                n,
                slices.total,
                started.elapsed().as_secs_f64()
            );
        }
        drop(heartbeat_guard);
        Ok(ExactNewtonJointFusedDenseEvaluation {
            gradient: ExactNewtonJointGradientEvaluation {
                log_likelihood,
                gradient,
            },
            hessian,
        })
    }

    pub(super) fn log_likelihood_from_exact_cache(
        &self,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<f64, String> {
        if !self.effective_flex_active(block_states)? {
            return self
                .log_likelihood_only_with_options(block_states, &BlockwiseFitOptions::default());
        }
        let n = self.y.len();
        let started = std::time::Instant::now();
        let heartbeat_guard = crate::heartbeat::scope(format!(
            "BMS exact-loglik eval n={n} p={}",
            cache.slices.total
        ));
        if log_exact_work(n) {
            log::info!(
                "[BMS exact-loglik] eval start n={} p={} source=cache",
                n,
                cache.slices.total
            );
        }
        let beta_h = self.score_beta(block_states)?;
        let beta_w = self.link_beta(block_states)?;
        let total: Result<f64, String> = (0..n)
            .into_par_iter()
            .try_fold(
                || 0.0,
                |mut log_likelihood, row| -> Result<_, String> {
                    let row_ctx = Self::row_ctx(cache, row);
                    let slope = block_states[1].eta[row];
                    let obs = self.observed_denested_cell_partials(
                        row,
                        row_ctx.intercept,
                        slope,
                        beta_h,
                        beta_w,
                    )?;
                    let s_i = eval_coeff4_at(&obs.coeff, self.z[row]);
                    let signed = (2.0 * self.y[row] - 1.0) * s_i;
                    let (log_cdf, _) = signed_probit_logcdf_and_mills_ratio(signed);
                    log_likelihood += self.weights[row] * log_cdf;
                    Ok(log_likelihood)
                },
            )
            .try_reduce(
                || 0.0,
                |left, right| -> Result<_, String> { Ok(left + right) },
            );
        let log_likelihood = total?;
        if log_exact_work(n) {
            log::info!(
                "[BMS exact-loglik] eval done n={} p={} source=cache elapsed={:.3}s",
                n,
                cache.slices.total,
                started.elapsed().as_secs_f64()
            );
        }
        drop(heartbeat_guard);
        Ok(log_likelihood)
    }

    pub(super) fn exact_newton_joint_gradient_evaluation_from_cache(
        &self,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<ExactNewtonJointGradientEvaluation, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let n = self.y.len();
        let started = std::time::Instant::now();
        let heartbeat_guard =
            crate::heartbeat::scope(format!("BMS exact-gradient eval n={n} p={}", slices.total));
        if log_exact_work(n) {
            log::info!(
                "[BMS exact-gradient] eval start n={} p={} source=cache",
                n,
                slices.total
            );
        }
        let make_acc = || {
            (
                0.0_f64,
                Array1::<f64>::zeros(slices.marginal.len()),
                Array1::<f64>::zeros(slices.logslope.len()),
                slices
                    .h
                    .as_ref()
                    .map(|range| Array1::<f64>::zeros(range.len())),
                slices
                    .w
                    .as_ref()
                    .map(|range| Array1::<f64>::zeros(range.len())),
            )
        };
        let (log_likelihood, grad_marginal, grad_logslope, grad_h, grad_w) = (0..n
            .div_ceil(ROW_CHUNK_SIZE))
            .into_par_iter()
            .try_fold(make_acc, |mut acc, chunk_idx| -> Result<_, String> {
                let start = chunk_idx * ROW_CHUNK_SIZE;
                let end = (start + ROW_CHUNK_SIZE).min(n);
                let mut scratch = BernoulliMarginalSlopeFlexRowScratch::new(primary.total);
                for row in start..end {
                    let row_ctx = Self::row_ctx(cache, row);
                    let row_moments = cache
                        .row_cell_moments
                        .as_ref()
                        .and_then(|bundle| bundle.row(row, 3));
                    let neglog = self.compute_row_analytic_flex_into_with_moments(
                        row,
                        block_states,
                        primary,
                        row_ctx,
                        row_moments,
                        false,
                        &mut scratch,
                    )?;
                    acc.0 -= neglog;
                    {
                        let mut marginal = acc.1.view_mut();
                        self.marginal_design.axpy_row_into(
                            row,
                            Self::exact_newton_score_component_from_objective_gradient(
                                scratch.grad[0],
                            ),
                            &mut marginal,
                        )?;
                    }
                    {
                        let mut logslope = acc.2.view_mut();
                        self.logslope_design.axpy_row_into(
                            row,
                            Self::exact_newton_score_component_from_objective_gradient(
                                scratch.grad[1],
                            ),
                            &mut logslope,
                        )?;
                    }
                    if let (Some(primary_h), Some(grad_h)) = (primary.h.as_ref(), acc.3.as_mut()) {
                        for idx in 0..primary_h.len() {
                            grad_h[idx] +=
                                Self::exact_newton_score_component_from_objective_gradient(
                                    scratch.grad[primary_h.start + idx],
                                );
                        }
                    }
                    if let (Some(primary_w), Some(grad_w)) = (primary.w.as_ref(), acc.4.as_mut()) {
                        for idx in 0..primary_w.len() {
                            grad_w[idx] +=
                                Self::exact_newton_score_component_from_objective_gradient(
                                    scratch.grad[primary_w.start + idx],
                                );
                        }
                    }
                }
                Ok(acc)
            })
            .try_reduce(make_acc, |mut left, right| -> Result<_, String> {
                left.0 += right.0;
                left.1 += &right.1;
                left.2 += &right.2;
                if let (Some(lhs), Some(rhs)) = (left.3.as_mut(), right.3.as_ref()) {
                    *lhs += rhs;
                }
                if let (Some(lhs), Some(rhs)) = (left.4.as_mut(), right.4.as_ref()) {
                    *lhs += rhs;
                }
                Ok(left)
            })?;

        let mut gradient = Array1::<f64>::zeros(slices.total);
        gradient
            .slice_mut(s![slices.marginal.clone()])
            .assign(&grad_marginal);
        gradient
            .slice_mut(s![slices.logslope.clone()])
            .assign(&grad_logslope);
        if let (Some(range), Some(grad_h)) = (slices.h.as_ref(), grad_h.as_ref()) {
            gradient.slice_mut(s![range.clone()]).assign(grad_h);
        }
        if let (Some(range), Some(grad_w)) = (slices.w.as_ref(), grad_w.as_ref()) {
            gradient.slice_mut(s![range.clone()]).assign(grad_w);
        }
        if log_exact_work(n) {
            log::info!(
                "[BMS exact-gradient] eval done n={} p={} source=cache elapsed={:.3}s",
                n,
                slices.total,
                started.elapsed().as_secs_f64()
            );
        }
        drop(heartbeat_guard);
        Ok(ExactNewtonJointGradientEvaluation {
            log_likelihood,
            gradient,
        })
    }

    pub(super) fn exact_newton_joint_hessian_matvec_from_cache(
        &self,
        direction: &Array1<f64>,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<Array1<f64>, String> {
        let mut out = Array1::<f64>::zeros(cache.slices.total);
        self.exact_newton_joint_hessian_matvec_from_cache_into(
            direction,
            block_states,
            cache,
            &mut out,
        )?;
        Ok(out)
    }

    /// Allocation-free HVP entry point.  Fills `out` (length
    /// `cache.slices.total`) with `H·direction`.  `out` is zeroed on entry
    /// and fully overwritten on success.
    pub(crate) fn exact_newton_joint_hessian_matvec_from_cache_into(
        &self,
        direction: &Array1<f64>,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        out: &mut Array1<f64>,
    ) -> Result<(), String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let n = self.y.len();

        out.fill(0.0);

        // ── Rigid closed-form: scalar kernel + design row ops ────────
        if !self.effective_flex_active(block_states)? {
            let partial = (0..n.div_ceil(ROW_CHUNK_SIZE))
                .into_par_iter()
                .try_fold(
                    || Array1::<f64>::zeros(slices.total),
                    |mut chunk_out, chunk_idx| -> Result<_, String> {
                        let start = chunk_idx * ROW_CHUNK_SIZE;
                        let end = (start + ROW_CHUNK_SIZE).min(n);
                        for row in start..end {
                            let marginal_eta = block_states[0].eta[row];
                            let marginal = self.marginal_link_map(marginal_eta)?;
                            let g = block_states[1].eta[row];
                            let (_, _, h) =
                                self.rigid_row_kernel_eval(row, marginal_eta, marginal, g)?;
                            let v_q = self
                                .marginal_design
                                .dot_row_view(row, direction.slice(s![slices.marginal.clone()]));
                            let v_g = self
                                .logslope_design
                                .dot_row_view(row, direction.slice(s![slices.logslope.clone()]));
                            let a_q = h[0][0] * v_q + h[0][1] * v_g;
                            let a_g = h[1][0] * v_q + h[1][1] * v_g;
                            {
                                let mut m = chunk_out.slice_mut(s![slices.marginal.clone()]);
                                self.marginal_design.axpy_row_into(row, a_q, &mut m)?;
                            }
                            {
                                let mut l = chunk_out.slice_mut(s![slices.logslope.clone()]);
                                self.logslope_design.axpy_row_into(row, a_g, &mut l)?;
                            }
                        }
                        Ok(chunk_out)
                    },
                )
                .try_reduce(
                    || Array1::<f64>::zeros(slices.total),
                    |mut left, right| -> Result<_, String> {
                        left += &right;
                        Ok(left)
                    },
                )?;
            *out += &partial;
            return Ok(());
        }

        // Phase-3 device-resident shortcut: when the row Hessian + designs
        // are pinned on the GPU, dispatch the HVP partial+reduce kernels and
        // return the joint-β image without ever touching the host
        // `row_primary_hessians` array.
        #[cfg(target_os = "linux")]
        {
            if let Some(device_state) = cache.row_primary_hessians.device() {
                match crate::gpu::bms_flex_row::launch_bms_flex_row_hvp(
                    device_state,
                    direction.as_slice().expect("direction is contiguous"),
                ) {
                    Ok(host) => {
                        if host.len() != out.len() {
                            return Err(format!(
                                "BMS GPU HVP length mismatch: got {}, expected {}",
                                host.len(),
                                out.len()
                            ));
                        }
                        out.iter_mut().zip(host.iter()).for_each(|(o, &v)| *o = v);
                        return Ok(());
                    }
                    Err(err) => {
                        log::info!(
                            "[BMS exact-newton HVP] gpu_hvp_failed: {err}; falling \
                             back to CPU row-loop (this should be rare under \
                             gpu=auto and is treated as a runtime degradation)"
                        );
                    }
                }
            }
        }

        // Host-pin shortcut: when the per-row Hessian is materialised on host
        // (the legacy path before Phase 3), build the joint-β image by
        // batching the per-row primary directions and dispatching the
        // per-row matvec helper from `gpu::row_hessian_ops`. On Linux this
        // can be GPU-accelerated by `launch_row_hessian_matvec`; on every
        // host the CPU oracle `cpu_row_hessian_matvec` is the in-process
        // fallback so the call sites stay consistent. The design pullback
        // (`pullback_primary_vector_add_into`) stays on host because the
        // designs are not necessarily resident on the device in this branch.
        if let Some(host_pin) = cache.row_primary_hessians.host_pin() {
            let r_pr = primary.total;
            // Single scratch buffer reused across rows — no per-row Array1.
            let mut row_dir_scratch = Array1::<f64>::zeros(r_pr);
            let mut v_rows = vec![0.0_f64; n * r_pr];
            for row in 0..n {
                self.row_primary_direction_from_flat_into(
                    row,
                    slices,
                    primary,
                    direction,
                    &mut row_dir_scratch,
                )?;
                v_rows[row * r_pr..(row + 1) * r_pr]
                    .copy_from_slice(row_dir_scratch.as_slice().expect("contiguous"));
            }
            let h_rows_arr = host_pin.hess();
            let h_rows_slice = h_rows_arr
                .as_slice()
                .expect("row_primary_hessians.hess() is row-major contiguous");
            let inputs = crate::gpu::row_hessian_ops::RowHessianMatvecInputs {
                n_rows: n,
                r: r_pr,
                h_rows: h_rows_slice,
                v_rows: &v_rows,
            };
            let y_rows = {
                #[cfg(target_os = "linux")]
                {
                    match crate::gpu::row_hessian_ops::launch_row_hessian_matvec(
                        crate::gpu::row_hessian_ops::RowHessianMatvecInputs {
                            n_rows: n,
                            r: r_pr,
                            h_rows: h_rows_slice,
                            v_rows: &v_rows,
                        },
                    ) {
                        Ok(result) => result.y_rows,
                        Err(err) => {
                            log::info!(
                                "[BMS exact-newton HVP] host-pin GPU matvec failed: {err}; \
                                 falling back to CPU oracle"
                            );
                            crate::gpu::row_hessian_ops::cpu_row_hessian_matvec(&inputs)
                        }
                    }
                }
                #[cfg(not(target_os = "linux"))]
                {
                    crate::gpu::row_hessian_ops::cpu_row_hessian_matvec(&inputs)
                }
            };
            // Reuse `row_dir_scratch` as the per-row action buffer (same
            // length r_pr) to avoid per-row allocation in the pullback loop.
            for row in 0..n {
                let action_slice = &y_rows[row * r_pr..(row + 1) * r_pr];
                row_dir_scratch
                    .iter_mut()
                    .zip(action_slice.iter())
                    .for_each(|(dst, &src)| *dst = src);
                self.pullback_primary_vector_add_into(row, slices, primary, &row_dir_scratch, out)?;
            }
            return Ok(());
        }

        let partial = (0..n.div_ceil(ROW_CHUNK_SIZE))
            .into_par_iter()
            .try_fold(
                || Array1::<f64>::zeros(slices.total),
                |mut chunk_out, chunk_idx| -> Result<_, String> {
                    let start = chunk_idx * ROW_CHUNK_SIZE;
                    let end = (start + ROW_CHUNK_SIZE).min(n);
                    let mut scratch = BernoulliMarginalSlopeFlexRowScratch::new(primary.total);
                    // Per-thread scratch for row direction — allocated once per
                    // chunk thread rather than once per row.
                    let mut row_dir = Array1::<f64>::zeros(primary.total);
                    for row in start..end {
                        let row_ctx = Self::row_ctx(cache, row);
                        self.row_primary_direction_from_flat_into(
                            row,
                            slices,
                            primary,
                            direction,
                            &mut row_dir,
                        )?;
                        let row_action =
                            if let Some(row_hess) = Self::cached_row_primary_hessian(cache, row) {
                                row_hess.dot(&row_dir)
                            } else {
                                let row_moments = cache
                                    .row_cell_moments
                                    .as_ref()
                                    .and_then(|bundle| bundle.row(row, 9));
                                self.compute_row_analytic_flex_into_with_moments(
                                    row,
                                    block_states,
                                    primary,
                                    row_ctx,
                                    row_moments,
                                    true,
                                    &mut scratch,
                                )?;
                                scratch.hess.dot(&row_dir)
                            };
                        self.pullback_primary_vector_add_into(
                            row,
                            slices,
                            primary,
                            &row_action,
                            &mut chunk_out,
                        )?;
                    }
                    Ok(chunk_out)
                },
            )
            .try_reduce(
                || Array1::<f64>::zeros(slices.total),
                |mut left, right| -> Result<_, String> {
                    left += &right;
                    Ok(left)
                },
            )?;
        *out += &partial;
        Ok(())
    }
    pub(super) fn exact_newton_joint_hessian_diagonal_from_cache(
        &self,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<Array1<f64>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let n = self.y.len();

        // ── Rigid closed-form: no jets, no row contexts ──────────────
        if !self.effective_flex_active(block_states)? {
            let diagonal = (0..n.div_ceil(ROW_CHUNK_SIZE))
                .into_par_iter()
                .try_fold(
                    || Array1::<f64>::zeros(slices.total),
                    |mut chunk_diag, chunk_idx| -> Result<_, String> {
                        let start = chunk_idx * ROW_CHUNK_SIZE;
                        let end = (start + ROW_CHUNK_SIZE).min(n);
                        for row in start..end {
                            let marginal_eta = block_states[0].eta[row];
                            let marginal = self.marginal_link_map(marginal_eta)?;
                            let g = block_states[1].eta[row];
                            let (_, _, h) =
                                self.rigid_row_kernel_eval(row, marginal_eta, marginal, g)?;
                            {
                                let mut m = chunk_diag.slice_mut(s![slices.marginal.clone()]);
                                self.marginal_design
                                    .squared_axpy_row_into(row, h[0][0], &mut m)?;
                            }
                            {
                                let mut l = chunk_diag.slice_mut(s![slices.logslope.clone()]);
                                self.logslope_design
                                    .squared_axpy_row_into(row, h[1][1], &mut l)?;
                            }
                        }
                        Ok(chunk_diag)
                    },
                )
                .try_reduce(
                    || Array1::<f64>::zeros(slices.total),
                    |mut left, right| -> Result<_, String> {
                        left += &right;
                        Ok(left)
                    },
                )?;
            return Ok(diagonal);
        }

        // Phase-3 device-resident shortcut: same idea as the HVP path.
        #[cfg(target_os = "linux")]
        {
            if let Some(device_state) = cache.row_primary_hessians.device() {
                match crate::gpu::bms_flex_row::launch_bms_flex_row_diagonal(device_state) {
                    Ok(host) => {
                        return Ok(Array1::<f64>::from_vec(host));
                    }
                    Err(err) => {
                        log::info!(
                            "[BMS exact-newton diag] gpu_diag_failed: {err}; falling \
                             back to CPU row-loop"
                        );
                    }
                }
            }
        }

        // Host-pin shortcut: extract every row's primary diagonal via the
        // per-row diagonal helper from `gpu::row_hessian_ops`, then perform
        // the design² accumulation on host (matches the rayon-loop algebra
        // below without rebuilding `r²` blocks per row). On Linux this uses
        // the GPU `launch_row_hessian_diag` kernel; on every host the CPU
        // oracle `cpu_row_hessian_diag` is the in-process fallback so the
        // call sites stay consistent.
        if let Some(host_pin) = cache.row_primary_hessians.host_pin() {
            let r_pr = primary.total;
            let h_rows_arr = host_pin.hess();
            let h_rows_slice = h_rows_arr
                .as_slice()
                .expect("row_primary_hessians.hess() is row-major contiguous");
            let inputs = crate::gpu::row_hessian_ops::RowHessianDiagInputs {
                n_rows: n,
                r: r_pr,
                h_rows: h_rows_slice,
            };
            let d_rows = {
                #[cfg(target_os = "linux")]
                {
                    match crate::gpu::row_hessian_ops::launch_row_hessian_diag(
                        crate::gpu::row_hessian_ops::RowHessianDiagInputs {
                            n_rows: n,
                            r: r_pr,
                            h_rows: h_rows_slice,
                        },
                    ) {
                        Ok(out) => out.d_rows,
                        Err(err) => {
                            log::info!(
                                "[BMS exact-newton diag] host-pin GPU diag failed: {err}; \
                                 falling back to CPU oracle"
                            );
                            crate::gpu::row_hessian_ops::cpu_row_hessian_diag(&inputs)
                        }
                    }
                }
                #[cfg(not(target_os = "linux"))]
                {
                    crate::gpu::row_hessian_ops::cpu_row_hessian_diag(&inputs)
                }
            };
            let mut diagonal = Array1::<f64>::zeros(slices.total);
            for row in 0..n {
                let d_row_base = row * r_pr;
                let h00 = d_rows[d_row_base];
                let h11 = d_rows[d_row_base + 1];
                {
                    let mut marginal_diag = diagonal.slice_mut(s![slices.marginal.clone()]);
                    self.marginal_design
                        .squared_axpy_row_into(row, h00, &mut marginal_diag)?;
                }
                {
                    let mut logslope_diag = diagonal.slice_mut(s![slices.logslope.clone()]);
                    self.logslope_design
                        .squared_axpy_row_into(row, h11, &mut logslope_diag)?;
                }
                if let (Some(primary_h), Some(block_h)) = (primary.h.as_ref(), slices.h.as_ref()) {
                    for (local_idx, global_idx) in block_h.clone().enumerate() {
                        let ii = primary_h.start + local_idx;
                        diagonal[global_idx] += d_rows[d_row_base + ii];
                    }
                }
                if let (Some(primary_w), Some(block_w)) = (primary.w.as_ref(), slices.w.as_ref()) {
                    for (local_idx, global_idx) in block_w.clone().enumerate() {
                        let ii = primary_w.start + local_idx;
                        diagonal[global_idx] += d_rows[d_row_base + ii];
                    }
                }
            }
            return Ok(diagonal);
        }

        let diagonal = (0..n.div_ceil(ROW_CHUNK_SIZE))
            .into_par_iter()
            .try_fold(
                || Array1::<f64>::zeros(slices.total),
                |mut chunk_diag, chunk_idx| -> Result<_, String> {
                    let start = chunk_idx * ROW_CHUNK_SIZE;
                    let end = (start + ROW_CHUNK_SIZE).min(n);
                    let mut scratch = BernoulliMarginalSlopeFlexRowScratch::new(primary.total);
                    for row in start..end {
                        let row_ctx = Self::row_ctx(cache, row);
                        // When the per-row primary Hessian is materialized at
                        // workspace construction (`row_primary_hessians`), the
                        // entire `r×r` block lives in the cache and we can read
                        // every diagonal entry directly. Otherwise rebuild the
                        // scratch Hessian on the fly.
                        let cached_hess = Self::cached_row_primary_hessian(cache, row);
                        if cached_hess.is_none() {
                            let row_moments = cache
                                .row_cell_moments
                                .as_ref()
                                .and_then(|bundle| bundle.row(row, 9));
                            self.compute_row_analytic_flex_into_with_moments(
                                row,
                                block_states,
                                primary,
                                row_ctx,
                                row_moments,
                                true,
                                &mut scratch,
                            )?;
                        }
                        let h00 = if let Some(row_hess) = cached_hess {
                            row_hess[[0, 0]]
                        } else {
                            scratch.hess[[0, 0]]
                        };
                        let h11 = if let Some(row_hess) = cached_hess {
                            row_hess[[1, 1]]
                        } else {
                            scratch.hess[[1, 1]]
                        };
                        {
                            let mut marginal_diag =
                                chunk_diag.slice_mut(s![slices.marginal.clone()]);
                            self.marginal_design.squared_axpy_row_into(
                                row,
                                h00,
                                &mut marginal_diag,
                            )?;
                        }
                        {
                            let mut logslope_diag =
                                chunk_diag.slice_mut(s![slices.logslope.clone()]);
                            self.logslope_design.squared_axpy_row_into(
                                row,
                                h11,
                                &mut logslope_diag,
                            )?;
                        }

                        if let (Some(primary_h), Some(block_h)) =
                            (primary.h.as_ref(), slices.h.as_ref())
                        {
                            for (local_idx, global_idx) in block_h.clone().enumerate() {
                                let ii = primary_h.start + local_idx;
                                chunk_diag[global_idx] += if let Some(row_hess) = cached_hess {
                                    row_hess[[ii, ii]]
                                } else {
                                    scratch.hess[[ii, ii]]
                                };
                            }
                        }
                        if let (Some(primary_w), Some(block_w)) =
                            (primary.w.as_ref(), slices.w.as_ref())
                        {
                            for (local_idx, global_idx) in block_w.clone().enumerate() {
                                let ii = primary_w.start + local_idx;
                                chunk_diag[global_idx] += if let Some(row_hess) = cached_hess {
                                    row_hess[[ii, ii]]
                                } else {
                                    scratch.hess[[ii, ii]]
                                };
                            }
                        }
                    }
                    Ok(chunk_diag)
                },
            )
            .try_reduce(
                || Array1::<f64>::zeros(slices.total),
                |mut left, right| -> Result<_, String> {
                    left += &right;
                    Ok(left)
                },
            )?;
        Ok(diagonal)
    }

    pub(super) fn exact_newton_joint_psi_terms_from_cache(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        self.exact_newton_joint_psi_terms_from_cache_with_options(
            block_states,
            derivative_blocks,
            psi_index,
            cache,
            &BlockwiseFitOptions::default(),
        )
    }

    /// Outer-aware variant of `exact_newton_joint_psi_terms_from_cache`. When
    /// `options.outer_score_subsample` is `None`, iterates all rows and is
    /// bit-for-bit equivalent to the legacy implementation. When `Some`, only
    /// the sampled rows contribute and every row-summed component (objective
    /// scalar, score vector, Hessian operator blocks) is accumulated with the
    /// row's Horvitz-Thompson inverse-inclusion weight.
    pub(crate) fn exact_newton_joint_psi_terms_from_cache_with_options(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        cache: &BernoulliMarginalSlopeExactEvalCache,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        let Some((block_idx, local_idx)) = psi_derivative_location(derivative_blocks, psi_index)
        else {
            return Ok(None);
        };
        let axis = self.resolve_psi_axis_spec(derivative_blocks, block_idx, local_idx)?;
        let mut results = self.run_psi_row_pass_for_axes(block_states, cache, options, &[axis])?;
        Ok(Some(results.remove(0)))
    }

    pub(super) fn resolve_psi_axis_spec(
        &self,
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        block_idx: usize,
        local_idx: usize,
    ) -> Result<PsiAxisSpec, String> {
        let n = self.y.len();
        let deriv = &derivative_blocks[block_idx][local_idx];
        let (p_psi, psi_label) = match block_idx {
            0 => (
                self.marginal_design.ncols(),
                "BernoulliMarginalSlopeFamily marginal",
            ),
            1 => (
                self.logslope_design.ncols(),
                "BernoulliMarginalSlopeFamily log-slope",
            ),
            _ => {
                return Err(format!(
                    "bernoulli marginal-slope psi terms only support marginal/logslope blocks, got block {block_idx}"
                ));
            }
        };
        let psi_map = crate::families::custom_family::resolve_custom_family_x_psi_map(
            deriv,
            n,
            p_psi,
            0..n,
            psi_label,
            &self.policy,
        )?;
        Ok(PsiAxisSpec {
            block_idx,
            idx_primary: if block_idx == 0 { 0 } else { 1 },
            psi_map,
        })
    }

    pub(super) fn run_psi_row_pass_for_axes(
        &self,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        options: &BlockwiseFitOptions,
        axes: &[PsiAxisSpec],
    ) -> Result<Vec<ExactNewtonJointPsiTerms>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let n = self.y.len();
        let k = axes.len();

        // Eager-prime the per-row uncontracted third-derivative cache *before*
        // entering the per-axis row `par_iter` so the build's own `par_iter`
        // does not nest inside an active rayon job. Subsequent ψ-axis sweeps
        // hit the cache via O(1) lookups in `rigid_third_full_cached`. Skipped
        // on the FLEX path because that branch routes through the flex jet
        // machinery, which has its own row-cell-moments cache.
        if !self.effective_flex_active(block_states)? {
            let warmed = self.rigid_third_full_cached(block_states, cache, 0)?;
            ensure_finite_third_full_cache_row(
                warmed,
                "run_psi_row_pass_for_axes rigid third-cache warm-up",
            )?;
        }

        // Block-local accumulator path: avoids O(n p^2) dense Hessian
        // materialization by keeping one accumulator per ψ axis in the
        // rayon fold.
        let weighted_rows = outer_weighted_rows(options, n);
        let make_acc = || -> Vec<(f64, Array1<f64>, BernoulliBlockHessianAccumulator)> {
            (0..k)
                .map(|_| {
                    (
                        0.0f64,
                        Array1::<f64>::zeros(slices.total),
                        BernoulliBlockHessianAccumulator::new(slices),
                    )
                })
                .collect()
        };
        let folded = weighted_rows
            .into_par_iter()
            .try_fold(make_acc, |mut acc, wr| -> Result<_, String> {
                let row = wr.index;
                let w = wr.weight;
                let row_ctx = Self::row_ctx(cache, row);
                let (f_pi, f_pipi_base) = self.compute_row_primary_gradient_hessian_reusing_cache(
                    row,
                    block_states,
                    primary,
                    row_ctx,
                    cache,
                )?;
                for (axis_idx, axis) in axes.iter().enumerate() {
                    // Single psi-map row materialization shared by `dir` and
                    // `psi_row`; the prior code paths each issued an
                    // independent `psi_map.row_vector(row)` call for the
                    // same (row, axis) which doubled the per-row operator
                    // dispatch cost for joint-spatial Hessian builds.
                    let psi_local = axis
                        .psi_map
                        .row_vector(row)
                        .map_err(|e| format!("bernoulli psi map row {row}: {e}"))?;
                    let dir_idx = if axis.block_idx == 0 {
                        primary.q
                    } else {
                        primary.logslope
                    };
                    let mut dir = Array1::<f64>::zeros(primary.total);
                    dir[dir_idx] = psi_local.dot(&block_states[axis.block_idx].beta);
                    let mut f_pipi = f_pipi_base.clone();
                    let mut third = self.row_primary_third_contracted_recompute(
                        row,
                        block_states,
                        cache,
                        row_ctx,
                        &dir,
                    )?;
                    let psi_row = BlockPsiRow {
                        block_idx: axis.block_idx,
                        range: if axis.block_idx == 0 {
                            slices.marginal.clone()
                        } else {
                            slices.logslope.clone()
                        },
                        local_vec: psi_local,
                    };
                    let mut f_pipi_dir = f_pipi.dot(&dir);
                    if w != 1.0 {
                        f_pipi.mapv_inplace(|v| v * w);
                        third.mapv_inplace(|v| v * w);
                        f_pipi_dir.mapv_inplace(|v| v * w);
                    }
                    let slot = &mut acc[axis_idx];
                    slot.0 += w * f_pi.dot(&dir);
                    slot.1
                        .slice_mut(s![psi_row.range.clone()])
                        .scaled_add(w * f_pi[axis.idx_primary], &psi_row.local_vec);
                    slot.1 += &self.pullback_primary_vector(row, slices, primary, &f_pipi_dir)?;

                    let right_primary = f_pipi.row(axis.idx_primary).to_owned();
                    slot.2.add_rank1_psi_cross(
                        self,
                        row,
                        slices,
                        primary,
                        axis.block_idx,
                        &psi_row.local_vec,
                        &right_primary,
                    );
                    slot.2.add_pullback(self, row, slices, primary, &third);
                }
                Ok(acc)
            })
            .try_reduce(make_acc, |mut left, right| -> Result<_, String> {
                for (l, r) in left.iter_mut().zip(right.into_iter()) {
                    l.0 += r.0;
                    l.1 += &r.1;
                    l.2.add(&r.2);
                }
                Ok(left)
            })?;

        let mut out = Vec::with_capacity(k);
        for (objective_psi, score_psi, block_acc) in folded.into_iter() {
            out.push(ExactNewtonJointPsiTerms {
                objective_psi,
                score_psi,
                hessian_psi: Array2::zeros((0, 0)),
                hessian_psi_operator: Some(std::sync::Arc::new(block_acc.into_operator(slices))),
            });
        }
        Ok(out)
    }

    pub(super) fn exact_newton_joint_psisecond_order_terms_from_cache(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        self.exact_newton_joint_psisecond_order_terms_from_cache_with_options(
            block_states,
            derivative_blocks,
            psi_i,
            psi_j,
            cache,
            &BlockwiseFitOptions::default(),
        )
    }

    /// Outer-aware variant of `exact_newton_joint_psisecond_order_terms_from_cache`.
    /// See `exact_newton_joint_psi_terms_from_cache_with_options` for the
    /// row-iter / weighting contract.
    pub(crate) fn exact_newton_joint_psisecond_order_terms_from_cache_with_options(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
        cache: &BernoulliMarginalSlopeExactEvalCache,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let Some((block_i, local_i)) = psi_derivative_location(derivative_blocks, psi_i) else {
            return Ok(None);
        };
        let Some((block_j, local_j)) = psi_derivative_location(derivative_blocks, psi_j) else {
            return Ok(None);
        };
        let idx_i = if block_i == 0 { 0 } else { 1 };
        let idx_j = if block_j == 0 { 0 } else { 1 };
        let n = self.y.len();
        let deriv_i = &derivative_blocks[block_i][local_i];
        let deriv_j = &derivative_blocks[block_j][local_j];
        let (p_psi_i, label_i) = match block_i {
            0 => (
                self.marginal_design.ncols(),
                "BernoulliMarginalSlopeFamily marginal",
            ),
            1 => (
                self.logslope_design.ncols(),
                "BernoulliMarginalSlopeFamily log-slope",
            ),
            _ => {
                return Err(format!(
                    "bernoulli marginal-slope psi second-order only supports marginal/logslope blocks, got block {block_i}"
                ));
            }
        };
        let (p_psi_j, label_j) = match block_j {
            0 => (
                self.marginal_design.ncols(),
                "BernoulliMarginalSlopeFamily marginal",
            ),
            1 => (
                self.logslope_design.ncols(),
                "BernoulliMarginalSlopeFamily log-slope",
            ),
            _ => {
                return Err(format!(
                    "bernoulli marginal-slope psi second-order only supports marginal/logslope blocks, got block {block_j}"
                ));
            }
        };

        // Build psi design maps once outside the row loop.
        let psi_map_i = crate::families::custom_family::resolve_custom_family_x_psi_map(
            deriv_i,
            n,
            p_psi_i,
            0..n,
            label_i,
            &self.policy,
        )?;
        let psi_map_j = crate::families::custom_family::resolve_custom_family_x_psi_map(
            deriv_j,
            n,
            p_psi_j,
            0..n,
            label_j,
            &self.policy,
        )?;
        let psi_map_ij = if block_i == block_j {
            Some(
                crate::families::custom_family::resolve_custom_family_x_psi_psi_map(
                    deriv_i,
                    deriv_j,
                    local_j,
                    n,
                    p_psi_i,
                    0..n,
                    label_i,
                    &self.policy,
                )?,
            )
        } else {
            None
        };

        // Block-local accumulator path for second-order psi terms
        let weighted_rows = outer_weighted_rows(options, n);
        let (objective_psi_psi, score_psi_psi, block_acc) = weighted_rows
            .into_par_iter()
            .try_fold(
                || {
                    (
                        0.0f64,
                        Array1::<f64>::zeros(slices.total),
                        BernoulliBlockHessianAccumulator::new(slices),
                    )
                },
                |mut acc, wr| -> Result<_, String> {
                    let row = wr.index;
                    let w = wr.weight;
                    {
                        // Materialize each psi-design row once and reuse for
                        // both the primary-space direction and the
                        // BlockPsiRow embedding (previously two separate
                        // psi_map.row_vector(row) calls per map per row).
                        let psi_local_i = psi_map_i
                            .row_vector(row)
                            .map_err(|e| format!("bernoulli psi map_i row {row}: {e}"))?;
                        let psi_local_j = psi_map_j
                            .row_vector(row)
                            .map_err(|e| format!("bernoulli psi map_j row {row}: {e}"))?;
                        let dir_idx_i = if block_i == 0 {
                            primary.q
                        } else {
                            primary.logslope
                        };
                        let dir_idx_j = if block_j == 0 {
                            primary.q
                        } else {
                            primary.logslope
                        };
                        let mut dir_i = Array1::<f64>::zeros(primary.total);
                        dir_i[dir_idx_i] = psi_local_i.dot(&block_states[block_i].beta);
                        let mut dir_j = Array1::<f64>::zeros(primary.total);
                        dir_j[dir_idx_j] = psi_local_j.dot(&block_states[block_j].beta);

                        // dir_ij and br_ij share psi_map_ij; materialize once.
                        let (dir_ij, psi_local_ij) = if let Some(ref pm_ij) = psi_map_ij {
                            if block_i != block_j {
                                (Array1::<f64>::zeros(primary.total), None)
                            } else {
                                let v = pm_ij
                                    .row_vector(row)
                                    .map_err(|e| format!("bernoulli psi map_ij row {row}: {e}"))?;
                                let mut d = Array1::<f64>::zeros(primary.total);
                                d[dir_idx_i] = v.dot(&block_states[block_i].beta);
                                (d, Some(v))
                            }
                        } else {
                            (Array1::<f64>::zeros(primary.total), None)
                        };
                        let row_ctx = Self::row_ctx(cache, row);
                        let (mut f_pi, mut f_pipi) = self
                            .compute_row_primary_gradient_hessian_reusing_cache(
                                row,
                                block_states,
                                primary,
                                row_ctx,
                                cache,
                            )?;
                        let mut third_i = self.row_primary_third_contracted_recompute(
                            row,
                            block_states,
                            cache,
                            row_ctx,
                            &dir_i,
                        )?;
                        let mut third_j = self.row_primary_third_contracted_recompute(
                            row,
                            block_states,
                            cache,
                            row_ctx,
                            &dir_j,
                        )?;
                        let mut fourth = self.row_primary_fourth_contracted_recompute(
                            row,
                            block_states,
                            cache,
                            row_ctx,
                            &dir_i,
                            &dir_j,
                        )?;
                        // Per-row HT weighting: scale every row contribution
                        // (gradient, Hessian, third, fourth) by w before
                        // accumulation. The post-sum scalar that the legacy
                        // code path applied is biased under variable weights.
                        if w != 1.0 {
                            f_pi.mapv_inplace(|v| v * w);
                            f_pipi.mapv_inplace(|v| v * w);
                            third_i.mapv_inplace(|v| v * w);
                            third_j.mapv_inplace(|v| v * w);
                            fourth.mapv_inplace(|v| v * w);
                        }
                        let br_i = BlockPsiRow {
                            block_idx: block_i,
                            range: if block_i == 0 {
                                slices.marginal.clone()
                            } else {
                                slices.logslope.clone()
                            },
                            local_vec: psi_local_i,
                        };
                        let br_j = BlockPsiRow {
                            block_idx: block_j,
                            range: if block_j == 0 {
                                slices.marginal.clone()
                            } else {
                                slices.logslope.clone()
                            },
                            local_vec: psi_local_j,
                        };
                        let br_ij = psi_local_ij.map(|v| BlockPsiRow {
                            block_idx: block_i,
                            range: if block_i == 0 {
                                slices.marginal.clone()
                            } else {
                                slices.logslope.clone()
                            },
                            local_vec: v,
                        });

                        // --- scalar and score accumulation (unchanged) ---
                        acc.0 += dir_i.dot(&f_pipi.dot(&dir_j)) + f_pi.dot(&dir_ij);
                        if let Some(ref bij) = br_ij {
                            let idx_ij = if bij.block_idx == 0 { 0 } else { 1 };
                            acc.1
                                .slice_mut(s![bij.range.clone()])
                                .scaled_add(f_pi[idx_ij], &bij.local_vec);
                        }
                        acc.1
                            .slice_mut(s![br_i.range.clone()])
                            .scaled_add(f_pipi.row(idx_i).dot(&dir_j), &br_i.local_vec);
                        acc.1
                            .slice_mut(s![br_j.range.clone()])
                            .scaled_add(f_pipi.row(idx_j).dot(&dir_i), &br_j.local_vec);
                        acc.1 += &self.pullback_primary_vector(
                            row,
                            slices,
                            primary,
                            &f_pipi.dot(&dir_ij),
                        )?;
                        acc.1 += &self.pullback_primary_vector(
                            row,
                            slices,
                            primary,
                            &third_i.dot(&dir_j),
                        )?;

                        // --- Hessian: bij outer pullback(f_pipi[idx_ij,:]) + transpose ---
                        if let Some(ref bij) = br_ij {
                            let idx_ij = if bij.block_idx == 0 { 0 } else { 1 };
                            let right_primary_ij = f_pipi.row(idx_ij).to_owned();
                            acc.2.add_rank1_psi_cross(
                                self,
                                row,
                                slices,
                                primary,
                                bij.block_idx,
                                &bij.local_vec,
                                &right_primary_ij,
                            );
                        }

                        // --- br_i outer br_j * f_pipi[[idx_i, idx_j]] + transpose ---
                        let scalar_ij = f_pipi[[idx_i, idx_j]];
                        acc.2.add_psi_psi_outer(
                            block_i,
                            &br_i.local_vec,
                            block_j,
                            &br_j.local_vec,
                            scalar_ij,
                        );

                        // --- br_i outer pullback(third_j[idx_i,:]) + transpose ---
                        let right_primary_i = third_j.row(idx_i).to_owned();
                        acc.2.add_rank1_psi_cross(
                            self,
                            row,
                            slices,
                            primary,
                            block_i,
                            &br_i.local_vec,
                            &right_primary_i,
                        );

                        // --- br_j outer pullback(third_i[idx_j,:]) + transpose ---
                        let right_primary_j = third_i.row(idx_j).to_owned();
                        acc.2.add_rank1_psi_cross(
                            self,
                            row,
                            slices,
                            primary,
                            block_j,
                            &br_j.local_vec,
                            &right_primary_j,
                        );

                        // --- fourth tensor pullback ---
                        acc.2.add_pullback(self, row, slices, primary, &fourth);

                        // --- third_ij tensor pullback ---
                        let mut third_ij = self.row_primary_third_contracted_recompute(
                            row,
                            block_states,
                            cache,
                            row_ctx,
                            &dir_ij,
                        )?;
                        if w != 1.0 {
                            third_ij.mapv_inplace(|v| v * w);
                        }
                        acc.2.add_pullback(self, row, slices, primary, &third_ij);
                    }
                    Ok(acc)
                },
            )
            .try_reduce(
                || {
                    (
                        0.0f64,
                        Array1::<f64>::zeros(slices.total),
                        BernoulliBlockHessianAccumulator::new(slices),
                    )
                },
                |mut left, right| -> Result<_, String> {
                    left.0 += right.0;
                    left.1 += &right.1;
                    left.2.add(&right.2);
                    Ok(left)
                },
            )?;
        // Per-row HT weighting was applied inside the closure (every
        // gradient / Hessian / third / fourth tensor scaled by `w` before
        // accumulation), so the unbiased estimator is already in
        // `block_acc` and no post-sum rescale is required.
        Ok(Some(ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi,
            hessian_psi_psi: Array2::zeros((0, 0)),
            hessian_psi_psi_operator: Some(Box::new(block_acc.into_operator(slices))),
        }))
    }

    pub(super) fn exact_newton_joint_psihessian_directional_derivative_from_cache(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_psihessian_directional_derivative_from_cache_with_options(
            block_states,
            derivative_blocks,
            psi_index,
            d_beta_flat,
            cache,
            &BlockwiseFitOptions::default(),
        )
    }

    /// Outer-aware variant of `exact_newton_joint_psihessian_directional_derivative_from_cache`.
    /// When `options.outer_score_subsample` is `Some`, only the sampled rows
    /// are visited and the accumulated dense Hessian-action matrix uses
    /// per-row Horvitz-Thompson inverse-inclusion weights. See
    /// `exact_newton_joint_psi_terms_from_cache_with_options` for the
    /// row-iter / weighting contract.
    pub(crate) fn exact_newton_joint_psihessian_directional_derivative_from_cache_with_options(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
        cache: &BernoulliMarginalSlopeExactEvalCache,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Array2<f64>>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let Some((block_idx, local_idx)) = psi_derivative_location(derivative_blocks, psi_index)
        else {
            return Ok(None);
        };
        let idx_primary = if block_idx == 0 { 0 } else { 1 };
        let n = self.y.len();
        let deriv = &derivative_blocks[block_idx][local_idx];
        let (p_psi, psi_label) = match block_idx {
            0 => (
                self.marginal_design.ncols(),
                "BernoulliMarginalSlopeFamily marginal",
            ),
            1 => (
                self.logslope_design.ncols(),
                "BernoulliMarginalSlopeFamily log-slope",
            ),
            _ => {
                return Err(format!(
                    "bernoulli marginal-slope psi hessian only supports marginal/logslope blocks, got block {block_idx}"
                ));
            }
        };

        // Build the psi design map once; rowwise calls use direct row_vector(row).
        let psi_map = crate::families::custom_family::resolve_custom_family_x_psi_map(
            deriv,
            n,
            p_psi,
            0..n,
            psi_label,
            &self.policy,
        )?;

        let weighted_rows = outer_weighted_rows(options, n);
        let block_acc = weighted_rows
            .into_par_iter()
            .try_fold(
                || BernoulliBlockHessianAccumulator::new(slices),
                |mut acc, wr| -> Result<_, String> {
                    let row = wr.index;
                    let w = wr.weight;
                    let row_dir =
                        self.row_primary_direction_from_flat(row, slices, primary, d_beta_flat)?;
                    let psi_dir = self.row_primary_psi_direction_from_map(
                        row,
                        block_idx,
                        &psi_map,
                        block_states,
                        primary,
                    )?;
                    let psi_action = self.row_primary_psi_action_on_direction_from_map(
                        row,
                        block_idx,
                        &psi_map,
                        slices,
                        d_beta_flat,
                        primary,
                    )?;
                    let row_ctx = Self::row_ctx(cache, row);
                    let mut third_beta = self.row_primary_third_contracted_recompute(
                        row,
                        block_states,
                        cache,
                        row_ctx,
                        &row_dir,
                    )?;
                    let mut fourth = self.row_primary_fourth_contracted_recompute(
                        row,
                        block_states,
                        cache,
                        row_ctx,
                        &row_dir,
                        &psi_dir,
                    )?;
                    if w != 1.0 {
                        third_beta.mapv_inplace(|v| v * w);
                        fourth.mapv_inplace(|v| v * w);
                    }
                    let psi_row = self.block_psi_row_from_map(row, block_idx, &psi_map, slices)?;
                    let right_primary = third_beta.row(idx_primary).to_owned();
                    acc.add_rank1_psi_cross(
                        self,
                        row,
                        slices,
                        primary,
                        psi_row.block_idx,
                        &psi_row.local_vec,
                        &right_primary,
                    );
                    acc.add_pullback(self, row, slices, primary, &fourth);
                    let mut third_action = self.row_primary_third_contracted_recompute(
                        row,
                        block_states,
                        cache,
                        row_ctx,
                        &psi_action,
                    )?;
                    if w != 1.0 {
                        third_action.mapv_inplace(|v| v * w);
                    }
                    acc.add_pullback(self, row, slices, primary, &third_action);
                    Ok(acc)
                },
            )
            .try_reduce(
                || BernoulliBlockHessianAccumulator::new(slices),
                |mut left, right| -> Result<_, String> {
                    left.add(&right);
                    Ok(left)
                },
            )?;
        Ok(Some(block_acc.to_dense(slices)))
    }

    /// Outer-aware operator builder for the Hessian directional derivative
    /// from a cached eval. The default-options variant is unused (the
    /// workspace always threads its own `BlockwiseFitOptions`), so the legacy
    /// non-`_with_options` shim is omitted.
    /// When `options.outer_score_subsample` is `Some`, only the masked rows
    /// are visited and the accumulated block Hessian operator uses per-row
    /// Horvitz-Thompson inverse-inclusion weights before being wrapped in the
    /// `HyperOperator`. See
    /// `exact_newton_joint_psi_terms_from_cache_with_options` for the
    /// row-iter / weighting contract.
    pub(crate) fn exact_newton_joint_psihessian_directional_derivative_operator_from_cache_with_options(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
        cache: &BernoulliMarginalSlopeExactEvalCache,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let Some((block_idx, local_idx)) = psi_derivative_location(derivative_blocks, psi_index)
        else {
            return Ok(None);
        };
        let idx_primary = if block_idx == 0 { 0 } else { 1 };
        let n = self.y.len();
        let deriv = &derivative_blocks[block_idx][local_idx];
        let (p_psi, psi_label) = match block_idx {
            0 => (
                self.marginal_design.ncols(),
                "BernoulliMarginalSlopeFamily marginal",
            ),
            1 => (
                self.logslope_design.ncols(),
                "BernoulliMarginalSlopeFamily log-slope",
            ),
            _ => {
                return Err(format!(
                    "bernoulli marginal-slope psi hessian operator only supports marginal/logslope blocks, got block {block_idx}"
                ));
            }
        };

        // Build the psi design map once; rowwise calls use direct row_vector(row).
        let psi_map = crate::families::custom_family::resolve_custom_family_x_psi_map(
            deriv,
            n,
            p_psi,
            0..n,
            psi_label,
            &self.policy,
        )?;

        let weighted_rows = outer_weighted_rows(options, n);
        let block_acc = weighted_rows
            .into_par_iter()
            .try_fold(
                || BernoulliBlockHessianAccumulator::new(slices),
                |mut acc, wr| -> Result<_, String> {
                    let row = wr.index;
                    let w = wr.weight;
                    let row_dir =
                        self.row_primary_direction_from_flat(row, slices, primary, d_beta_flat)?;
                    let psi_dir = self.row_primary_psi_direction_from_map(
                        row,
                        block_idx,
                        &psi_map,
                        block_states,
                        primary,
                    )?;
                    let psi_action = self.row_primary_psi_action_on_direction_from_map(
                        row,
                        block_idx,
                        &psi_map,
                        slices,
                        d_beta_flat,
                        primary,
                    )?;
                    let row_ctx = Self::row_ctx(cache, row);
                    let mut third_beta = self.row_primary_third_contracted_recompute(
                        row,
                        block_states,
                        cache,
                        row_ctx,
                        &row_dir,
                    )?;
                    let mut fourth = self.row_primary_fourth_contracted_recompute(
                        row,
                        block_states,
                        cache,
                        row_ctx,
                        &row_dir,
                        &psi_dir,
                    )?;
                    if w != 1.0 {
                        third_beta.mapv_inplace(|v| v * w);
                        fourth.mapv_inplace(|v| v * w);
                    }
                    let psi_row = self.block_psi_row_from_map(row, block_idx, &psi_map, slices)?;
                    let right_primary = third_beta.row(idx_primary).to_owned();
                    acc.add_rank1_psi_cross(
                        self,
                        row,
                        slices,
                        primary,
                        psi_row.block_idx,
                        &psi_row.local_vec,
                        &right_primary,
                    );
                    acc.add_pullback(self, row, slices, primary, &fourth);
                    let mut third_action = self.row_primary_third_contracted_recompute(
                        row,
                        block_states,
                        cache,
                        row_ctx,
                        &psi_action,
                    )?;
                    if w != 1.0 {
                        third_action.mapv_inplace(|v| v * w);
                    }
                    acc.add_pullback(self, row, slices, primary, &third_action);
                    Ok(acc)
                },
            )
            .try_reduce(
                || BernoulliBlockHessianAccumulator::new(slices),
                |mut left, right| -> Result<_, String> {
                    left.add(&right);
                    Ok(left)
                },
            )?;
        Ok(Some(
            Arc::new(block_acc.into_operator(slices)) as Arc<dyn HyperOperator>
        ))
    }

    pub(super) fn exact_newton_joint_hessian_directional_derivative_from_cache(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_directional_derivative_from_cache_with_options(
            block_states,
            d_beta_flat,
            cache,
            &BlockwiseFitOptions::default(),
        )
    }

    /// Outer-aware variant of `exact_newton_joint_hessian_directional_derivative_from_cache`.
    /// When `options.outer_score_subsample` is `Some`, only the masked rows
    /// are visited and the accumulated dense Hessian directional-derivative
    /// matrix uses per-row Horvitz-Thompson inverse-inclusion weights before
    /// densification.
    pub(crate) fn exact_newton_joint_hessian_directional_derivative_from_cache_with_options(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
        cache: &BernoulliMarginalSlopeExactEvalCache,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Array2<f64>>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let n = self.y.len();
        let weighted_rows = outer_weighted_rows(options, n);

        // ── Rigid closed-form: 3rd-order scalar kernel ───────────────
        if !self.effective_flex_active(block_states)? {
            let block_acc = weighted_rows
                .into_par_iter()
                .try_fold(
                    || BernoulliBlockHessianAccumulator::new(slices),
                    |mut acc, wr| -> Result<_, String> {
                        let row = wr.index;
                        let w = wr.weight;
                        let marginal_eta = block_states[0].eta[row];
                        let marginal = self.marginal_link_map(marginal_eta)?;
                        let g = block_states[1].eta[row];
                        let dq = self
                            .marginal_design
                            .dot_row_view(row, d_beta_flat.slice(s![slices.marginal.clone()]));
                        let dg = self
                            .logslope_design
                            .dot_row_view(row, d_beta_flat.slice(s![slices.logslope.clone()]));
                        let t = self.rigid_row_third_contracted(
                            row,
                            marginal_eta,
                            marginal,
                            g,
                            dq,
                            dg,
                        )?;
                        acc.add_pullback_rigid_2x2(self, row, &t, w);
                        Ok(acc)
                    },
                )
                .try_reduce(
                    || BernoulliBlockHessianAccumulator::new(slices),
                    |mut left, right| {
                        left.add(&right);
                        Ok(left)
                    },
                )?;
            return Ok(Some(block_acc.to_dense(slices)));
        }

        let block_acc = weighted_rows
            .into_par_iter()
            .try_fold(
                || BernoulliBlockHessianAccumulator::new(slices),
                |mut acc, wr| -> Result<_, String> {
                    let row = wr.index;
                    let w = wr.weight;
                    let row_dir =
                        self.row_primary_direction_from_flat(row, slices, primary, d_beta_flat)?;
                    let row_ctx = Self::row_ctx(cache, row);
                    let mut third = self.row_primary_third_contracted_recompute(
                        row,
                        block_states,
                        cache,
                        row_ctx,
                        &row_dir,
                    )?;
                    if w != 1.0 {
                        third.mapv_inplace(|v| v * w);
                    }
                    acc.add_pullback(self, row, slices, primary, &third);
                    Ok(acc)
                },
            )
            .try_reduce(
                || BernoulliBlockHessianAccumulator::new(slices),
                |mut left, right| -> Result<_, String> {
                    left.add(&right);
                    Ok(left)
                },
            )?;
        Ok(Some(block_acc.to_dense(slices)))
    }

    /// Outer-aware operator builder for the joint-Hessian directional
    /// derivative. The default-options shim is omitted because the
    /// `BernoulliMarginalSlopeExactNewtonJointHessianWorkspace` always threads
    /// its own `BlockwiseFitOptions`. When `options.outer_score_subsample` is `Some`, only the
    /// sampled rows are visited and the accumulator uses per-row
    /// Horvitz-Thompson inverse-inclusion weights before being wrapped in the operator.
    pub(crate) fn exact_newton_joint_hessian_directional_derivative_operator_from_cache_with_options(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
        cache: &BernoulliMarginalSlopeExactEvalCache,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let n = self.y.len();
        let weighted_rows = outer_weighted_rows(options, n);

        if !self.effective_flex_active(block_states)? {
            let block_acc = weighted_rows
                .into_par_iter()
                .try_fold(
                    || BernoulliBlockHessianAccumulator::new(slices),
                    |mut acc, wr| -> Result<_, String> {
                        let row = wr.index;
                        let w = wr.weight;
                        let marginal_eta = block_states[0].eta[row];
                        let marginal = self.marginal_link_map(marginal_eta)?;
                        let g = block_states[1].eta[row];
                        let dq = self
                            .marginal_design
                            .dot_row_view(row, d_beta_flat.slice(s![slices.marginal.clone()]));
                        let dg = self
                            .logslope_design
                            .dot_row_view(row, d_beta_flat.slice(s![slices.logslope.clone()]));
                        let t = self.rigid_row_third_contracted(
                            row,
                            marginal_eta,
                            marginal,
                            g,
                            dq,
                            dg,
                        )?;
                        acc.add_pullback_rigid_2x2(self, row, &t, w);
                        Ok(acc)
                    },
                )
                .try_reduce(
                    || BernoulliBlockHessianAccumulator::new(slices),
                    |mut left, right| -> Result<_, String> {
                        left.add(&right);
                        Ok(left)
                    },
                )?;
            return Ok(Some(
                Arc::new(block_acc.into_operator(slices)) as Arc<dyn HyperOperator>
            ));
        }

        let block_acc = weighted_rows
            .into_par_iter()
            .try_fold(
                || BernoulliBlockHessianAccumulator::new(slices),
                |mut acc, wr| -> Result<_, String> {
                    let row = wr.index;
                    let w = wr.weight;
                    let row_dir =
                        self.row_primary_direction_from_flat(row, slices, primary, d_beta_flat)?;
                    let row_ctx = Self::row_ctx(cache, row);
                    let mut third = self.row_primary_third_contracted_recompute(
                        row,
                        block_states,
                        cache,
                        row_ctx,
                        &row_dir,
                    )?;
                    if w != 1.0 {
                        third.mapv_inplace(|v| v * w);
                    }
                    acc.add_pullback(self, row, slices, primary, &third);
                    Ok(acc)
                },
            )
            .try_reduce(
                || BernoulliBlockHessianAccumulator::new(slices),
                |mut left, right| -> Result<_, String> {
                    left.add(&right);
                    Ok(left)
                },
            )?;
        Ok(Some(
            Arc::new(block_acc.into_operator(slices)) as Arc<dyn HyperOperator>
        ))
    }

    pub(crate) fn exact_newton_joint_hessian_directional_derivative_operators_from_cache_with_options(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flats: &[Array1<f64>],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        options: &BlockwiseFitOptions,
    ) -> Result<Vec<Option<Arc<dyn HyperOperator>>>, String> {
        if d_beta_flats.is_empty() {
            return Ok(Vec::new());
        }
        let slices = &cache.slices;
        let primary = &cache.primary;
        let n = self.y.len();
        let weighted_rows = outer_weighted_rows(options, n);
        let make_accs = || {
            (0..d_beta_flats.len())
                .map(|_| BernoulliBlockHessianAccumulator::new(slices))
                .collect::<Vec<_>>()
        };
        let started = std::time::Instant::now();

        let n_rows = weighted_rows.len();
        let n_dirs = d_beta_flats.len();
        let flex_active = self.effective_flex_active(block_states)?;
        let bundle_present = cache.row_cell_moments.is_some();
        let heartbeat_guard = crate::heartbeat::scope(format!(
            "BMS batched dH n={n} rows={n_rows} p={} dirs={n_dirs} flex={flex_active} cell_moments_bundle={bundle_present}",
            slices.total
        ));
        log::info!(
            "[BMS batched dH start] n={} rows={} p={} dirs={} flex={} cell_moments_bundle={}",
            n,
            n_rows,
            slices.total,
            n_dirs,
            flex_active,
            bundle_present,
        );
        let progress = Arc::new(AtomicUsize::new(0));
        let progress_step = (n_rows / 8).max(1);
        let progress_started = started;
        let bump_progress = |progress: &AtomicUsize| {
            let now = progress.fetch_add(1, Ordering::Relaxed) + 1;
            if now == n_rows || now.is_multiple_of(progress_step) {
                log::info!(
                    "[BMS batched dH progress] rows={}/{} dirs={} elapsed={:.3}s",
                    now,
                    n_rows,
                    n_dirs,
                    progress_started.elapsed().as_secs_f64(),
                );
            }
        };
        let dense_contiguous_rows = weighted_rows.len() == n
            && weighted_rows
                .iter()
                .enumerate()
                .all(|(row, wr)| wr.index == row && wr.weight == 1.0);
        // Pre-warm the per-row caches the row loop transitively reaches, so the
        // first row to enter the par_iter does not lazily run a nested
        // `into_par_iter()` inside a lazy cache initializer / `RayonSafeOnce`
        // race and starve the outer pool. Two distinct hazards:
        //   1. The rigid third-derivative tensor cache (`rigid_third_full_cached`).
        //      Used by the chunked / else branches when `!flex_active`; harmless
        //      to populate even when the `!flex_active` rank-1 branch will not
        //      consult it.
        //   2. Lazy dense materialization of any kernel / coefficient-transform
        //      design operator (`ChunkedKernelDesignOperator::materialized_combined`
        //      → `build_row_chunk_combined` runs `par_chunks_mut`). Multiple
        //      racing outer workers each spawn that nested parallel build, and
        //      with the rigid rank-1 row body (`dot_row_view`, `syr_row_into`)
        //      every row touches both designs — guaranteed contention. Touching
        //      a single row on the main thread before the par_iter forces the
        //      lazy build once, leaving the par_iter body to read
        //      already-materialized `Array2` rows in O(p).
        // Mirrors the same discipline applied in
        // `exact_newton_joint_hessiansecond_directional_derivative_from_cache_with_options`.
        if !flex_active && n > 0 {
            let warmed = self.rigid_third_full_cached(block_states, cache, 0)?;
            ensure_finite_third_full_cache_row(
                warmed,
                "compute_gradient_and_hessian_via_psi_axes rigid third-cache warm-up",
            )?;
        }
        if n > 0 {
            let warm_marg = Array1::<f64>::zeros(slices.marginal.end - slices.marginal.start);
            let marginal_probe = self.marginal_design.dot_row_view(0, warm_marg.view());
            if !marginal_probe.is_finite() {
                return Err(
                    "compute_gradient_and_hessian_via_psi_axes marginal design warm-up produced a non-finite value"
                        .to_string(),
                );
            }
            let warm_log = Array1::<f64>::zeros(slices.logslope.end - slices.logslope.start);
            let logslope_probe = self.logslope_design.dot_row_view(0, warm_log.view());
            if !logslope_probe.is_finite() {
                return Err(
                    "compute_gradient_and_hessian_via_psi_axes logslope design warm-up produced a non-finite value"
                        .to_string(),
                );
            }
        }
        // Even with the warm-up above, fall back to a serial row loop when the
        // par_iter cannot pay for its own dispatch overhead, or when we are
        // already inside a rayon worker (so an outer par_iter is holding pool
        // slots and a nested `into_par_iter` here would risk pool starvation
        // on the LRU mutex inside `evaluate_cell_derivative_moments_lru`,
        // etc.). At biobank n_rows the per-row body's design materialization
        // and pullback work dominates dispatch, so the par_iter is preserved.
        const ROW_PAR_MIN_ROWS: usize = 4_096;
        let run_rows_serial = rayon::current_thread_index().is_some()
            || rayon::current_num_threads() <= 1
            || n_rows < ROW_PAR_MIN_ROWS;
        let mut accs = if !flex_active {
            if run_rows_serial {
                let mut accs = make_accs();
                for wr in weighted_rows.iter() {
                    let row = wr.index;
                    let w = wr.weight;
                    let marginal_eta = block_states[0].eta[row];
                    let marginal = self.marginal_link_map(marginal_eta)?;
                    let g = block_states[1].eta[row];
                    for (idx, d_beta_flat) in d_beta_flats.iter().enumerate() {
                        let dq = self
                            .marginal_design
                            .dot_row_view(row, d_beta_flat.slice(s![slices.marginal.clone()]));
                        let dg = self
                            .logslope_design
                            .dot_row_view(row, d_beta_flat.slice(s![slices.logslope.clone()]));
                        let t = self.rigid_row_third_contracted(
                            row,
                            marginal_eta,
                            marginal,
                            g,
                            dq,
                            dg,
                        )?;
                        accs[idx].add_pullback_rigid_2x2(self, row, &t, w);
                    }
                    bump_progress(&progress);
                }
                accs
            } else {
                weighted_rows
                    .clone()
                    .into_par_iter()
                    .try_fold(make_accs, |mut accs, wr| -> Result<_, String> {
                        let row = wr.index;
                        let w = wr.weight;
                        let marginal_eta = block_states[0].eta[row];
                        let marginal = self.marginal_link_map(marginal_eta)?;
                        let g = block_states[1].eta[row];
                        for (idx, d_beta_flat) in d_beta_flats.iter().enumerate() {
                            let dq = self
                                .marginal_design
                                .dot_row_view(row, d_beta_flat.slice(s![slices.marginal.clone()]));
                            let dg = self
                                .logslope_design
                                .dot_row_view(row, d_beta_flat.slice(s![slices.logslope.clone()]));
                            let t = self.rigid_row_third_contracted(
                                row,
                                marginal_eta,
                                marginal,
                                g,
                                dq,
                                dg,
                            )?;
                            accs[idx].add_pullback_rigid_2x2(self, row, &t, w);
                        }
                        bump_progress(&progress);
                        Ok(accs)
                    })
                    .try_reduce(make_accs, |mut left, right| -> Result<_, String> {
                        for (l, r) in left.iter_mut().zip(right.iter()) {
                            l.add(r);
                        }
                        Ok(left)
                    })?
            }
        } else if dense_contiguous_rows {
            let marginal_dirs =
                Self::stacked_direction_block(d_beta_flats, slices.marginal.clone());
            let logslope_dirs =
                Self::stacked_direction_block(d_beta_flats, slices.logslope.clone());
            let (chunk_rows, gpu_sized_chunks) =
                Self::batched_directional_derivative_chunk_rows(n, d_beta_flats.len());
            let chunks = (0..n)
                .step_by(chunk_rows)
                .map(|start| (start, (start + chunk_rows).min(n)))
                .collect::<Vec<_>>();
            log::info!(
                "[BMS batched dH chunks] rows_per_chunk={} chunks={} gpu_sized={}",
                chunk_rows,
                chunks.len(),
                gpu_sized_chunks,
            );
            let chunk_body =
                |(start, end): (usize, usize)| -> Result<Vec<BernoulliBlockHessianAccumulator>, String> {
                    let n_dirs = d_beta_flats.len();
                    let len = end - start;
                    let mut accs = make_accs();
                    let mut w_mm = (0..n_dirs)
                        .map(|_| Array1::<f64>::zeros(len))
                        .collect::<Vec<_>>();
                    let mut w_mg = (0..n_dirs)
                        .map(|_| Array1::<f64>::zeros(len))
                        .collect::<Vec<_>>();
                    let mut w_gg = (0..n_dirs)
                        .map(|_| Array1::<f64>::zeros(len))
                        .collect::<Vec<_>>();
                    let x_chunk = self
                        .marginal_design
                        .try_row_chunk(start..end)
                        .map_err(|e| format!("bernoulli marginal_design try_row_chunk: {e}"))?;
                    let g_chunk = self
                        .logslope_design
                        .try_row_chunk(start..end)
                        .map_err(|e| format!("bernoulli logslope_design try_row_chunk: {e}"))?;
                    let marginal_projected =
                        crate::faer_ndarray::fast_ab(&x_chunk, &marginal_dirs);
                    let logslope_projected =
                        crate::faer_ndarray::fast_ab(&g_chunk, &logslope_dirs);

                    for row in start..end {
                        let local = row - start;
                        let row_ctx = Self::row_ctx(cache, row);
                        let row_dirs = Self::row_primary_directions_from_projected(
                            local,
                            slices,
                            primary,
                            d_beta_flats,
                            &marginal_projected,
                            &logslope_projected,
                        );
                        let thirds = self.row_primary_third_contracted_many_with_moments(
                            row,
                            block_states,
                            cache,
                            row_ctx,
                            &row_dirs,
                        )?;
                        for (idx, third) in thirds.iter().enumerate() {
                            w_mm[idx][local] = third[[0, 0]];
                            w_mg[idx][local] = third[[0, 1]];
                            w_gg[idx][local] = third[[1, 1]];
                            accs[idx].add_hw_pullback_only(self, row, slices, primary, third);
                        }
                        bump_progress(&progress);
                    }

                    for idx in 0..n_dirs {
                        accs[idx].add_weighted_design_grams_from_chunks(
                            &x_chunk,
                            &g_chunk,
                            &w_mm[idx],
                            &w_mg[idx],
                            &w_gg[idx],
                        );
                    }
                    Ok(accs)
                };
            if run_rows_serial || gpu_sized_chunks {
                let mut accs = make_accs();
                for chunk in chunks {
                    let partial = chunk_body(chunk)?;
                    for (l, r) in accs.iter_mut().zip(partial.iter()) {
                        l.add(r);
                    }
                }
                accs
            } else {
                chunks.into_par_iter().map(chunk_body).try_reduce(
                    make_accs,
                    |mut left, right| -> Result<_, String> {
                        for (l, r) in left.iter_mut().zip(right.iter()) {
                            l.add(r);
                        }
                        Ok(left)
                    },
                )?
            }
        } else {
            let row_body = |wr: WeightedOuterRow,
                            accs: &mut Vec<BernoulliBlockHessianAccumulator>|
             -> Result<(), String> {
                let row = wr.index;
                let w = wr.weight;
                let row_ctx = Self::row_ctx(cache, row);
                let row_dirs = d_beta_flats
                    .iter()
                    .map(|d_beta_flat| {
                        self.row_primary_direction_from_flat(row, slices, primary, d_beta_flat)
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                let mut thirds = self.row_primary_third_contracted_many_with_moments(
                    row,
                    block_states,
                    cache,
                    row_ctx,
                    &row_dirs,
                )?;
                for (idx, third) in thirds.iter_mut().enumerate() {
                    if w != 1.0 {
                        third.mapv_inplace(|v| v * w);
                    }
                    accs[idx].add_pullback(self, row, slices, primary, third);
                }
                bump_progress(&progress);
                Ok(())
            };
            if run_rows_serial {
                let mut accs = make_accs();
                for wr in weighted_rows.iter() {
                    row_body(*wr, &mut accs)?;
                }
                accs
            } else {
                weighted_rows
                    .into_par_iter()
                    .try_fold(make_accs, |mut accs, wr| -> Result<_, String> {
                        row_body(wr, &mut accs)?;
                        Ok(accs)
                    })
                    .try_reduce(make_accs, |mut left, right| -> Result<_, String> {
                        for (l, r) in left.iter_mut().zip(right.iter()) {
                            l.add(r);
                        }
                        Ok(left)
                    })?
            }
        };

        let elapsed = started.elapsed().as_secs_f64();
        log::info!(
            "[BMS batched dH] n={} rows={} p={} dirs={} elapsed={:.3}s",
            n,
            n_rows,
            slices.total,
            n_dirs,
            elapsed
        );
        let operators = accs
            .drain(..)
            .map(|acc| Some(Arc::new(acc.into_operator(slices)) as Arc<dyn HyperOperator>))
            .collect();
        drop(heartbeat_guard);
        Ok(operators)
    }

    pub(super) fn exact_newton_joint_hessiansecond_directional_derivative_from_cache(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessiansecond_directional_derivative_from_cache_with_options(
            block_states,
            d_beta_u_flat,
            d_beta_v_flat,
            cache,
            &BlockwiseFitOptions::default(),
        )
    }

    /// Outer-aware variant of
    /// `exact_newton_joint_hessiansecond_directional_derivative_from_cache`.
    /// When `options.outer_score_subsample` is `Some`, only the masked rows
    /// are visited and the accumulated dense second-directional Hessian
    /// matrix uses per-row Horvitz-Thompson inverse-inclusion weights.
    pub(crate) fn exact_newton_joint_hessiansecond_directional_derivative_from_cache_with_options(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
        cache: &BernoulliMarginalSlopeExactEvalCache,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Array2<f64>>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let n = self.y.len();
        let make_acc = || BernoulliBlockHessianAccumulator::new(slices);
        let weighted_rows = outer_weighted_rows(options, n);

        // Eager-prime the per-row uncontracted fourth-derivative cache *before*
        // entering the per-row `par_iter` so the cache's nested-`par_iter`
        // build does not race with Rayon workers already inside the outer
        // loop — see `feedback_oncelock_rayon_deadlock` and the mirror
        // pre-warm for the third-derivative tensor at the top of
        // `compute_gradient_and_hessian_via_psi_axes`. Skipped on the FLEX
        // path because that branch routes through the flex jet machinery
        // instead of `rigid_fourth_full_cached`.
        if !self.effective_flex_active(block_states)? {
            let warmed = self.rigid_fourth_full_cached(block_states, cache, 0)?;
            ensure_finite_fourth_full_cache_row(
                warmed,
                "exact_newton_joint_hessiansecond_directional_derivative_from_cache rigid fourth-cache warm-up",
            )?;
        }

        // ── Rigid closed-form: 4th-order scalar kernel ───────────────
        if !self.effective_flex_active(block_states)? {
            let block_acc = weighted_rows
                .into_par_iter()
                .try_fold(make_acc, |mut acc, wr| -> Result<_, String> {
                    let row = wr.index;
                    let w = wr.weight;
                    let uq = self
                        .marginal_design
                        .dot_row_view(row, d_beta_u_flat.slice(s![slices.marginal.clone()]));
                    let ug = self
                        .logslope_design
                        .dot_row_view(row, d_beta_u_flat.slice(s![slices.logslope.clone()]));
                    let vq = self
                        .marginal_design
                        .dot_row_view(row, d_beta_v_flat.slice(s![slices.marginal.clone()]));
                    let vg = self
                        .logslope_design
                        .dot_row_view(row, d_beta_v_flat.slice(s![slices.logslope.clone()]));
                    let t = self.rigid_fourth_full_cached(block_states, cache, row)?;
                    let f = contract_fourth_full(t, uq, ug, vq, vg);
                    let mut f_arr = Array2::from_shape_fn((2, 2), |(a, b)| f[a][b]);
                    if w != 1.0 {
                        f_arr.mapv_inplace(|v| v * w);
                    }
                    acc.add_pullback(self, row, slices, primary, &f_arr);
                    Ok(acc)
                })
                .try_reduce(make_acc, |mut left, right| -> Result<_, String> {
                    left.add(&right);
                    Ok(left)
                })?;
            return Ok(Some(block_acc.to_dense(slices)));
        }

        let block_acc = weighted_rows
            .into_par_iter()
            .try_fold(make_acc, |mut acc, wr| -> Result<_, String> {
                let row = wr.index;
                let w = wr.weight;
                let row_u =
                    self.row_primary_direction_from_flat(row, slices, primary, d_beta_u_flat)?;
                let row_v =
                    self.row_primary_direction_from_flat(row, slices, primary, d_beta_v_flat)?;
                let row_ctx = Self::row_ctx(cache, row);
                let mut fourth = self.row_primary_fourth_contracted_recompute(
                    row,
                    block_states,
                    cache,
                    row_ctx,
                    &row_u,
                    &row_v,
                )?;
                if w != 1.0 {
                    fourth.mapv_inplace(|v| v * w);
                }
                acc.add_pullback(self, row, slices, primary, &fourth);
                Ok(acc)
            })
            .try_reduce(make_acc, |mut left, right| -> Result<_, String> {
                left.add(&right);
                Ok(left)
            })?;
        Ok(Some(block_acc.to_dense(slices)))
    }

    /// Outer-aware operator builder for the joint-Hessian second directional
    /// derivative. The default-options shim is omitted because the
    /// `BernoulliMarginalSlopeExactNewtonJointHessianWorkspace` always threads
    /// its own `BlockwiseFitOptions`. When `options.outer_score_subsample` is `Some`, only the
    /// sampled rows are visited and the accumulator uses per-row
    /// Horvitz-Thompson inverse-inclusion weights before being wrapped in the operator.
    pub(crate) fn exact_newton_joint_hessiansecond_directional_derivative_operator_from_cache_with_options(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
        cache: &BernoulliMarginalSlopeExactEvalCache,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let n = self.y.len();
        let make_acc = || BernoulliBlockHessianAccumulator::new(slices);
        let weighted_rows = outer_weighted_rows(options, n);

        // Eager-prime the per-row uncontracted fourth-derivative cache *before*
        // entering the per-row `par_iter` to avoid the lazy-cache-under-rayon
        // deadlock pattern — see `feedback_oncelock_rayon_deadlock`.
        if !self.effective_flex_active(block_states)? {
            let warmed = self.rigid_fourth_full_cached(block_states, cache, 0)?;
            ensure_finite_fourth_full_cache_row(
                warmed,
                "exact_newton_joint_hessiansecond_directional_derivative_operator_from_cache rigid fourth-cache warm-up",
            )?;
        }

        if !self.effective_flex_active(block_states)? {
            let block_acc = weighted_rows
                .into_par_iter()
                .try_fold(make_acc, |mut acc, wr| -> Result<_, String> {
                    let row = wr.index;
                    let w = wr.weight;
                    let uq = self
                        .marginal_design
                        .dot_row_view(row, d_beta_u_flat.slice(s![slices.marginal.clone()]));
                    let ug = self
                        .logslope_design
                        .dot_row_view(row, d_beta_u_flat.slice(s![slices.logslope.clone()]));
                    let vq = self
                        .marginal_design
                        .dot_row_view(row, d_beta_v_flat.slice(s![slices.marginal.clone()]));
                    let vg = self
                        .logslope_design
                        .dot_row_view(row, d_beta_v_flat.slice(s![slices.logslope.clone()]));
                    let t = self.rigid_fourth_full_cached(block_states, cache, row)?;
                    let f = contract_fourth_full(t, uq, ug, vq, vg);
                    let mut f_arr = Array2::from_shape_fn((2, 2), |(a, b)| f[a][b]);
                    if w != 1.0 {
                        f_arr.mapv_inplace(|v| v * w);
                    }
                    acc.add_pullback(self, row, slices, primary, &f_arr);
                    Ok(acc)
                })
                .try_reduce(make_acc, |mut left, right| -> Result<_, String> {
                    left.add(&right);
                    Ok(left)
                })?;
            return Ok(Some(
                Arc::new(block_acc.into_operator(slices)) as Arc<dyn HyperOperator>
            ));
        }

        let block_acc = weighted_rows
            .into_par_iter()
            .try_fold(make_acc, |mut acc, wr| -> Result<_, String> {
                let row = wr.index;
                let w = wr.weight;
                let row_u =
                    self.row_primary_direction_from_flat(row, slices, primary, d_beta_u_flat)?;
                let row_v =
                    self.row_primary_direction_from_flat(row, slices, primary, d_beta_v_flat)?;
                let row_ctx = Self::row_ctx(cache, row);
                let mut fourth = self.row_primary_fourth_contracted_recompute(
                    row,
                    block_states,
                    cache,
                    row_ctx,
                    &row_u,
                    &row_v,
                )?;
                if w != 1.0 {
                    fourth.mapv_inplace(|v| v * w);
                }
                acc.add_pullback(self, row, slices, primary, &fourth);
                Ok(acc)
            })
            .try_reduce(make_acc, |mut left, right| -> Result<_, String> {
                left.add(&right);
                Ok(left)
            })?;
        Ok(Some(
            Arc::new(block_acc.into_operator(slices)) as Arc<dyn HyperOperator>
        ))
    }

    pub(super) fn evaluate_flex_block_diagonals_from_cache(
        &self,
        block_states: &[ParameterBlockState],
        slices: &BlockSlices,
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<FamilyEvaluation, String> {
        let primary = cache.primary.clone();
        let n = self.y.len();
        let n_chunks = n.div_ceil(ROW_CHUNK_SIZE);
        // Pool of per-worker workspaces reused across chunks within this
        // evaluate. The previous implementation seeded a fresh accumulator
        // per try_fold chunk, paying p_marginal² + p_logslope² (+ optional
        // p_h², p_w²) dense Hessian allocations per chunk. The pool caps
        // total allocations at the number of distinct rayon workers that
        // ever grab a chunk; each chunk reuses the worker's existing
        // dense buffers via in-place += accumulation. Keep the row scratch in
        // the same pool: it owns primary_dim² arrays and is also chunk-local.
        let pool: Mutex<
            Vec<(
                BernoulliExactNewtonAccumulator,
                BernoulliMarginalSlopeFlexRowScratch,
            )>,
        > = Mutex::new(Vec::new());
        let result: Result<(), String> =
            (0..n_chunks)
                .into_par_iter()
                .try_for_each(|chunk_idx| -> Result<(), String> {
                    let (mut acc, mut scratch) = pool
                        .lock()
                        .expect("bernoulli exact newton accumulator pool poisoned")
                        .pop()
                        .unwrap_or_else(|| {
                            (
                                BernoulliExactNewtonAccumulator::new(slices),
                                BernoulliMarginalSlopeFlexRowScratch::new(primary.total),
                            )
                        });
                    let start = chunk_idx * ROW_CHUNK_SIZE;
                    let end = (start + ROW_CHUNK_SIZE).min(n);
                    let chunk_res: Result<(), String> = (|| {
                        for row in start..end {
                            let row_ctx = Self::row_ctx(cache, row);
                            let row_moments = cache
                                .row_cell_moments
                                .as_ref()
                                .and_then(|bundle| bundle.row(row, 9));
                            let row_neglog = self.compute_row_analytic_flex_into_with_moments(
                                row,
                                block_states,
                                &primary,
                                row_ctx,
                                row_moments,
                                true,
                                &mut scratch,
                            )?;
                            acc.add_pullback_block_diagonals(
                                self, row, &primary, row_neglog, &scratch,
                            )?;
                        }
                        Ok(())
                    })();
                    pool.lock()
                        .expect("bernoulli exact newton accumulator pool poisoned")
                        .push((acc, scratch));
                    chunk_res
                });
        result?;
        let mut pooled = pool
            .into_inner()
            .expect("bernoulli exact newton accumulator pool poisoned");
        let reduced = match pooled.pop() {
            Some((mut first, _)) => {
                for (other, _) in &pooled {
                    first.add(other);
                }
                first
            }
            None => BernoulliExactNewtonAccumulator::new(slices),
        };

        let BernoulliExactNewtonAccumulator {
            ll,
            grad_marginal,
            grad_logslope,
            hess_marginal,
            hess_logslope,
            grad_h,
            grad_w,
            hess_h,
            hess_w,
        } = reduced;

        let mut blockworking_sets = vec![
            BlockWorkingSet::ExactNewton {
                gradient: grad_marginal,
                hessian: SymmetricMatrix::Dense(hess_marginal),
            },
            BlockWorkingSet::ExactNewton {
                gradient: grad_logslope,
                hessian: SymmetricMatrix::Dense(hess_logslope),
            },
        ];
        if let (Some(gradient), Some(hessian)) = (grad_h, hess_h) {
            blockworking_sets.push(BlockWorkingSet::ExactNewton {
                gradient,
                hessian: SymmetricMatrix::Dense(hessian),
            });
        }
        if let (Some(gradient), Some(hessian)) = (grad_w, hess_w) {
            blockworking_sets.push(BlockWorkingSet::ExactNewton {
                gradient,
                hessian: SymmetricMatrix::Dense(hessian),
            });
        }
        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets,
        })
    }

    pub(super) fn evaluate_blockwise_exact_newton(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<FamilyEvaluation, String> {
        let slices = block_slices(self);
        let flex_active = self.effective_flex_active(block_states)?;

        // ── Block-diagonal direct path (rigid, p < 512) ─────────────────
        //
        // The RowKernel<2> is the single source of truth in objective space
        // (negative log-likelihood). The full joint Hessian's off-diagonal
        // marginal/logslope cross block is unused by the per-block working
        // sets the inner solver consumes, so we accumulate only the two
        // diagonal blocks via the family's sparse-aware syr / axpy.  This
        // avoids the Θ(n·(p_m+p_g)²) joint assembly + immediate slice that
        // the previous implementation paid.
        if !flex_active && slices.total < 512 {
            let kern = BernoulliRigidRowKernel::new(self.clone(), block_states.to_vec());
            let cache = build_row_kernel_cache(&kern, &crate::families::row_kernel::RowSet::All)?;
            let ll = row_kernel_log_likelihood(&cache, &crate::families::row_kernel::RowSet::All);
            let joint_gradient = Self::exact_newton_score_from_objective_gradient(
                row_kernel_gradient(&kern, &cache, &crate::families::row_kernel::RowSet::All),
            );

            let n = cache.n;
            let p_marginal = slices.marginal.len();
            let p_logslope = slices.logslope.len();
            let make_pair = || {
                (
                    Array2::<f64>::zeros((p_marginal, p_marginal)),
                    Array2::<f64>::zeros((p_logslope, p_logslope)),
                )
            };
            let (hess_marginal, hess_logslope) = (0..n.div_ceil(ROW_CHUNK_SIZE))
                .into_par_iter()
                .try_fold(
                    make_pair,
                    |(mut hm, mut hl), chunk_idx| -> Result<(Array2<f64>, Array2<f64>), String> {
                        let start = chunk_idx * ROW_CHUNK_SIZE;
                        let end = (start + ROW_CHUNK_SIZE).min(n);
                        let rows = end - start;
                        let marginal_chunk = self
                            .marginal_design
                            .try_row_chunk(start..end)
                            .map_err(|e| format!("bernoulli marginal_design try_row_chunk: {e}"))?;
                        let logslope_chunk = self
                            .logslope_design
                            .try_row_chunk(start..end)
                            .map_err(|e| format!("bernoulli logslope_design try_row_chunk: {e}"))?;
                        let mut hm_w_buf = [0.0f64; ROW_CHUNK_SIZE];
                        let mut hl_w_buf = [0.0f64; ROW_CHUNK_SIZE];
                        let hm_w = &mut hm_w_buf[..rows];
                        let hl_w = &mut hl_w_buf[..rows];
                        for local_row in 0..rows {
                            let h = &cache.hessians[start + local_row];
                            hm_w[local_row] = h[0][0];
                            hl_w[local_row] = h[1][1];
                        }
                        add_weighted_chunk_gram(&marginal_chunk, hm_w, &mut hm);
                        add_weighted_chunk_gram(&logslope_chunk, hl_w, &mut hl);
                        Ok((hm, hl))
                    },
                )
                .try_reduce(
                    make_pair,
                    |(mut lhm, mut lhl),
                     (rhm, rhl)|
                     -> Result<(Array2<f64>, Array2<f64>), String> {
                        lhm += &rhm;
                        lhl += &rhl;
                        Ok((lhm, lhl))
                    },
                )?;

            let hess_marginal =
                Self::exact_newton_observed_information_from_objective_hessian(hess_marginal);
            let hess_logslope =
                Self::exact_newton_observed_information_from_objective_hessian(hess_logslope);

            let grad_marginal = joint_gradient.slice(s![slices.marginal.clone()]).to_owned();
            let grad_logslope = joint_gradient.slice(s![slices.logslope.clone()]).to_owned();

            let mut sets = vec![
                BlockWorkingSet::ExactNewton {
                    gradient: grad_marginal,
                    hessian: SymmetricMatrix::Dense(hess_marginal),
                },
                BlockWorkingSet::ExactNewton {
                    gradient: grad_logslope,
                    hessian: SymmetricMatrix::Dense(hess_logslope),
                },
            ];
            if let Some(range) = slices.h.as_ref() {
                // Rigid mode does not exercise h/w; mirror the blockwise
                // fallback by exposing zero working sets.
                sets.push(BlockWorkingSet::ExactNewton {
                    gradient: Array1::zeros(range.len()),
                    hessian: SymmetricMatrix::Dense(Array2::zeros((range.len(), range.len()))),
                });
            }
            if let Some(range) = slices.w.as_ref() {
                sets.push(BlockWorkingSet::ExactNewton {
                    gradient: Array1::zeros(range.len()),
                    hessian: SymmetricMatrix::Dense(Array2::zeros((range.len(), range.len()))),
                });
            }
            return Ok(FamilyEvaluation {
                log_likelihood: ll,
                blockworking_sets: sets,
            });
        }

        // ── Flex block-diagonal path ─────────────────────────────────
        // Flexible rows are independent once the intercept cache is built, so
        // `evaluate_flex_block_diagonals_from_cache` accumulates each row into
        // Rayon thread-local block buffers and reduces the buffers once. The
        // off-diagonal joint blocks are not consumed by the inner block solver.
        if flex_active {
            let cache = self.build_exact_eval_cache_with_order(block_states)?;
            return self.evaluate_flex_block_diagonals_from_cache(block_states, &slices, &cache);
        }

        // ── Blockwise fallback (p >= 512) ───────────────────────────────
        //
        // The joint dense Hessian is too large to materialise.  Block
        // Hessians are assembled independently via the same per-row
        // kernel, so the algebra is correct but not structurally guaranteed
        // identical to the joint object.  This path should only be reached
        // for very large models where memory is the binding constraint.
        let n = self.y.len();
        let p_marginal = slices.marginal.len();
        let p_logslope = slices.logslope.len();
        let make_acc = || {
            (
                0.0_f64,
                Array1::<f64>::zeros(p_marginal),
                Array1::<f64>::zeros(p_logslope),
                Array2::<f64>::zeros((p_marginal, p_marginal)),
                Array2::<f64>::zeros((p_logslope, p_logslope)),
            )
        };
        let (ll, grad_marginal, grad_logslope, hess_marginal, hess_logslope) = (0..n
            .div_ceil(ROW_CHUNK_SIZE))
            .into_par_iter()
            .try_fold(
                make_acc,
                |(mut ll, mut gm, mut gl, mut hm, mut hl), chunk_idx| -> Result<_, String> {
                    let start = chunk_idx * ROW_CHUNK_SIZE;
                    let end = (start + ROW_CHUNK_SIZE).min(n);
                    let rows = end - start;
                    let marginal_chunk = self
                        .marginal_design
                        .try_row_chunk(start..end)
                        .map_err(|e| format!("bernoulli marginal_design try_row_chunk: {e}"))?;
                    let logslope_chunk = self
                        .logslope_design
                        .try_row_chunk(start..end)
                        .map_err(|e| format!("bernoulli logslope_design try_row_chunk: {e}"))?;
                    let mut gm_w_buf = [0.0f64; ROW_CHUNK_SIZE];
                    let mut gl_w_buf = [0.0f64; ROW_CHUNK_SIZE];
                    let mut hm_w_buf = [0.0f64; ROW_CHUNK_SIZE];
                    let mut hl_w_buf = [0.0f64; ROW_CHUNK_SIZE];
                    let gm_w = &mut gm_w_buf[..rows];
                    let gl_w = &mut gl_w_buf[..rows];
                    let hm_w = &mut hm_w_buf[..rows];
                    let hl_w = &mut hl_w_buf[..rows];
                    for local_row in 0..rows {
                        let row = start + local_row;
                        let marginal_eta = block_states[0].eta[row];
                        let marginal = self.marginal_link_map(marginal_eta)?;
                        let g = block_states[1].eta[row];
                        let (neglog, grad, h) =
                            self.rigid_row_kernel_eval(row, marginal_eta, marginal, g)?;
                        ll -= neglog;
                        gm_w[local_row] =
                            Self::exact_newton_score_component_from_objective_gradient(grad[0]);
                        gl_w[local_row] =
                            Self::exact_newton_score_component_from_objective_gradient(grad[1]);
                        hm_w[local_row] = h[0][0];
                        hl_w[local_row] = h[1][1];
                    }
                    add_weighted_chunk_gradient(&marginal_chunk, gm_w, &mut gm);
                    add_weighted_chunk_gradient(&logslope_chunk, gl_w, &mut gl);
                    add_weighted_chunk_gram(&marginal_chunk, hm_w, &mut hm);
                    add_weighted_chunk_gram(&logslope_chunk, hl_w, &mut hl);
                    Ok((ll, gm, gl, hm, hl))
                },
            )
            .try_reduce(
                make_acc,
                |(lll, mut lgm, mut lgl, mut lhm, mut lhl),
                 (rll, rgm, rgl, rhm, rhl)|
                 -> Result<_, String> {
                    lgm += &rgm;
                    lgl += &rgl;
                    lhm += &rhm;
                    lhl += &rhl;
                    Ok((lll + rll, lgm, lgl, lhm, lhl))
                },
            )?;

        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: {
                let mut sets = vec![
                    BlockWorkingSet::ExactNewton {
                        gradient: grad_marginal,
                        hessian: SymmetricMatrix::Dense(hess_marginal),
                    },
                    BlockWorkingSet::ExactNewton {
                        gradient: grad_logslope,
                        hessian: SymmetricMatrix::Dense(hess_logslope),
                    },
                ];
                if let Some(range) = slices.h.as_ref() {
                    sets.push(BlockWorkingSet::ExactNewton {
                        gradient: Array1::zeros(range.len()),
                        hessian: SymmetricMatrix::Dense(Array2::zeros((range.len(), range.len()))),
                    });
                }
                if let Some(range) = slices.w.as_ref() {
                    sets.push(BlockWorkingSet::ExactNewton {
                        gradient: Array1::zeros(range.len()),
                        hessian: SymmetricMatrix::Dense(Array2::zeros((range.len(), range.len()))),
                    });
                }
                sets
            },
        })
    }
}

impl CustomFamily for BernoulliMarginalSlopeFamily {
    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    // Historically the link-deviation basis used a β-dependent moment anchor
    // (empirical at the rigid-pilot η₀); as PIRLS drifted η away from η₀,
    // `Σᵢ wᵢ b(η_currentᵢ)` re-acquired a constant component aliased with
    // the location intercept, collapsing σ_min(joint H+S) to ridge_floor and
    // leaking spurious terms into d log|H|/dρ. The basis is now constructed
    // with a β-INDEPENDENT smoothness-penalty null-space drop (see
    // `DeviationRuntime::try_new` and
    // `smoothness_nullspace_orthogonal_complement` in `deviation_runtime.rs`):
    // the basis structurally cannot represent polynomials of degree
    // `< max_penalty_derivative_order`, so the location block's intercept and
    // any unpenalized location-linear absorb constants/linears in η at every
    // PIRLS iteration. With `S` full-rank on the transformed basis,
    // `σ_min(H+S) ≥ smallest positive eigenvalue of S` — orders of magnitude
    // above ridge_floor — so the original "near-null direction" no longer
    // appears. We retain `HardPseudo` for the exact-newton outer hessian
    // path because numerical rounding can still leave eigenvalues that are
    // very small (though no longer at the floor); excluding σ ≤ ε from both
    // log|H| and its gradient consistently is harmless when there is no
    // genuine null space and remains correct if a residual one ever
    // re-appears (defense in depth, mirroring
    // BinomialLocationScaleWiggleFamily).
    fn pseudo_logdet_mode(&self) -> crate::custom_family::PseudoLogdetMode {
        crate::custom_family::PseudoLogdetMode::HardPseudo
    }

    fn coefficient_hessian_cost(&self, specs: &[ParameterBlockSpec]) -> u64 {
        // Operator-aware: rigid Bernoulli marginal-slope wires the K=2
        // RowKernel through a matrix-free workspace that applies joint Hv at
        // O(n · (p_marginal + p_logslope + p_flex)) per call. Only fall back
        // to the dense `n · (Σ p_b)²` build when `use_joint_matrix_free_path`
        // declines the operator path.
        let n = self.y.len() as u64;
        let p_total: u64 = specs
            .iter()
            .map(|s| s.design.ncols() as u64)
            .fold(0u64, |a, p| a.saturating_add(p));
        if crate::custom_family::use_joint_matrix_free_path(p_total as usize, n as usize) {
            n.saturating_mul(p_total)
        } else {
            crate::custom_family::joint_coupled_coefficient_hessian_cost(n, specs)
        }
    }

    fn exact_outer_derivative_order(
        &self,
        specs: &[ParameterBlockSpec],
        options: &BlockwiseFitOptions,
    ) -> crate::custom_family::ExactOuterDerivativeOrder {
        assert!(std::mem::size_of_val(options) > 0);
        use crate::custom_family::ExactOuterDerivativeOrder;

        let flex_active = self.score_warp.is_some() || self.link_dev.is_some();
        let coefficient_work = self
            .coefficient_hessian_cost(specs)
            .max(self.coefficient_gradient_cost(specs));
        let dense_available = self.outer_hyper_hessian_dense_available(specs);
        let hvp_available = self.outer_hyper_hessian_hvp_available(specs);
        if !dense_available && !hvp_available {
            if log_exact_work(self.y.len()) {
                log::info!(
                    "[BMS outer-derivative-policy] n={} p={} flex={} order=First reason=no-outer-hessian dense_available={} outer_hvp_available={} coefficient_work={}",
                    self.y.len(),
                    specs.iter().map(|spec| spec.design.ncols()).sum::<usize>(),
                    flex_active,
                    dense_available,
                    hvp_available,
                    coefficient_work,
                );
            }
            return ExactOuterDerivativeOrder::First;
        }

        let order = crate::custom_family::exact_outer_order_with_outer_hvp(
            specs,
            coefficient_work,
            hvp_available,
        );
        if log_exact_work(self.y.len()) {
            let p_total = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
            let matrix_free_inner_requested =
                crate::custom_family::use_joint_matrix_free_path(p_total, self.y.len());
            let inner_route = if matrix_free_inner_requested
                && self.inner_coefficient_hessian_hvp_available(specs)
            {
                "workspace-hvp"
            } else if p_total < 512 {
                "workspace-dense"
            } else if self.inner_coefficient_hessian_hvp_available(specs) {
                "workspace-hvp"
            } else {
                "direct-dense"
            };
            log::info!(
                "[BMS outer-derivative-policy] n={} p={} flex={} order={:?} declared_hessian=analytic inner_route={} matrix_free_inner_requested={} dense_available={} outer_hvp_available={} coefficient_work={}",
                self.y.len(),
                p_total,
                flex_active,
                order,
                inner_route,
                matrix_free_inner_requested,
                dense_available,
                hvp_available,
                coefficient_work,
            );
        }
        order
    }

    fn outer_seed_config(&self, n_params: usize) -> crate::seeding::SeedConfig {
        let mut config = crate::seeding::SeedConfig::default();
        if n_params == 0 {
            return config;
        }
        config.max_seeds = 1;
        config.seed_budget = 1;
        config.screen_max_inner_iterations = 2;
        config
    }

    fn exact_newton_joint_psi_workspace_for_first_order_terms(&self) -> bool {
        true
    }

    fn batched_outer_gradient_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        rho: &Array1<f64>,
        options: &BlockwiseFitOptions,
        hessian_workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
    ) -> Result<Option<BatchedOuterGradientTerms>, String> {
        let psi_dim: usize = derivative_blocks.iter().map(Vec::len).sum();
        if psi_dim != 0 {
            return Ok(None);
        }
        if block_states.len() != specs.len() {
            return Ok(None);
        }
        // Two-phase auto-subsample schedule. Phase 1: stratified
        // Horvitz–Thompson mask (≈ 1 % gradient noise) for the first
        // BMS_AUTO_SUBSAMPLE_PHASE1_BUDGET outer evaluations, where
        // BFGS makes the bulk of its progress on the noisy gradient.
        // Phase 2: full data for every subsequent evaluation, so the
        // optimizer can drive ‖∇‖ below the user's tight `outer_tol`
        // without bumping into the stochastic noise floor. The phase
        // counter is per-family-instance, so each fresh fit (which
        // constructs a new family) starts at Phase 1.
        //
        // The mask is auto-derived only when `options.auto_outer_subsample`
        // is enabled and the caller has not already supplied a mask of their
        // own. All gating logic lives
        // in `maybe_install_auto_outer_subsample` so the survival
        // families can reuse the same schedule.
        let stratum_secondary: Vec<u8> = self
            .y
            .iter()
            .map(|v| if *v > 0.5 { 1u8 } else { 0u8 })
            .collect();
        let owned_options;
        let options: &BlockwiseFitOptions =
            match crate::families::marginal_slope_shared::maybe_install_auto_outer_subsample(
                options,
                self.z.as_slice().expect("z must be contiguous"),
                Some(&stratum_secondary),
                rho.as_slice().expect("outer rho must be contiguous"),
                &self.auto_subsample_phase_counter,
                &self.auto_subsample_last_rho,
                BMS_AUTO_SUBSAMPLE_PHASE1_BUDGET,
                "BMS",
                // Per-K work-unit cost for the BMS outer gradient kernel.
                // BMS uses a Polya–Gamma augmentation with a per-row
                // pseudo-Bernoulli factorisation that is materially cheaper
                // than the survival marginal-slope risk-set scan. Empirical
                // cost is ~50_000 units per K-unit at biobank scale, which
                // with `AUTO_OUTER_WORK_BUDGET = 5×10⁸` caps
                //   K_work ≈ 5e8 / 50_000 = 10_000,
                // matching the existing default `min_k = 10_000` and so
                // never binding tighter than the noise rule in current
                // production configurations — the cap exists to guard
                // against pathological per-row cost regressions, not to
                // change today's nominal K.
                50_000,
            ) {
                Some(cloned) => {
                    owned_options = cloned;
                    &owned_options
                }
                None => options,
            };
        let ranges = Self::block_ranges_from_specs(specs);
        let total = ranges.last().map(|(_, end)| *end).unwrap_or(0);
        if total == 0 {
            return Ok(Some(BatchedOuterGradientTerms {
                objective_theta: Array1::zeros(0),
                trace_h_inv_hdot: Array1::zeros(0),
                trace_s_pinv_sdot: Array1::zeros(0),
            }));
        }
        if rho.len() != specs.iter().map(|spec| spec.penalties.len()).sum::<usize>() {
            return Ok(None);
        }
        if total >= 512 {
            return Ok(None);
        }

        let batched_started = std::time::Instant::now();
        let beta = Self::flatten_block_state_betas_for_specs(block_states, specs)?;
        if log_exact_work(self.y.len()) {
            log::info!(
                "[BMS batched outer-gradient] start n={} p={} rho={} subsample_rows={} workspace={}",
                self.y.len(),
                total,
                rho.len(),
                options
                    .outer_score_subsample
                    .as_ref()
                    .map_or(self.y.len(), |subsample| subsample.len()),
                hessian_workspace.is_some()
            );
        }
        let hessian_started = std::time::Instant::now();
        let mut h = if let Some(workspace) = hessian_workspace.as_ref() {
            // Outer batched-gradient assembly genuinely needs the dense
            // joint Hessian (it pulls back `H_β` against an explicit factor),
            // so bypass the workspace's matrix-free inner-route gate via
            // `hessian_dense_forced` — that always materialises through the
            // fused row pass instead of falling through to column-basis HVP.
            workspace.hessian_dense_forced()?.ok_or_else(|| {
                "bernoulli marginal-slope batched gradient requires dense exact joint Hessian below p=512"
                    .to_string()
            })?
        } else {
            self.exact_newton_joint_hessian(block_states)?.ok_or_else(|| {
                "bernoulli marginal-slope batched gradient could not build dense exact joint Hessian"
                    .to_string()
            })?
        };
        if log_exact_work(self.y.len()) {
            log::info!(
                "[BMS batched outer-gradient] dense-hessian ready n={} p={} elapsed={:.3}s",
                self.y.len(),
                total,
                hessian_started.elapsed().as_secs_f64()
            );
        }
        if h.nrows() != total || h.ncols() != total {
            return Err(format!(
                "bernoulli marginal-slope batched gradient Hessian shape {}x{} != {total}x{total}",
                h.nrows(),
                h.ncols()
            ));
        }

        let penalty_started = std::time::Instant::now();
        let ridge = options.ridge_floor.max(1e-15);
        let trace_diagonal_ridge = if options.ridge_policy.include_quadratic_penalty
            || options.ridge_policy.include_penalty_logdet
        {
            ridge
        } else {
            0.0
        };
        let mut objective_theta = Array1::<f64>::zeros(rho.len());
        let mut trace_s_pinv_sdot = Array1::<f64>::zeros(rho.len());
        let mut penalty_cursor = 0usize;
        let mut per_block_rho: Vec<Array1<f64>> = Vec::with_capacity(specs.len());
        let mut penalties_dense: Vec<Vec<Array2<f64>>> = Vec::with_capacity(specs.len());
        for (block_idx, spec) in specs.iter().enumerate() {
            let count = spec.penalties.len();
            let block_rho = rho
                .slice(s![penalty_cursor..penalty_cursor + count])
                .to_owned();
            let lambdas = block_rho.mapv(f64::exp);
            per_block_rho.push(block_rho);
            let (start, end) = ranges[block_idx];
            let beta_block = beta.slice(s![start..end]);
            let mut s_lambda = Array2::<f64>::zeros((end - start, end - start));
            let mut block_penalties = Vec::with_capacity(count);
            for (local_idx, penalty) in spec.penalties.iter().enumerate() {
                let dense = penalty.to_dense();
                let lambda = lambdas[local_idx];
                let s_beta = dense.dot(&beta_block);
                objective_theta[penalty_cursor + local_idx] =
                    0.5 * lambda * beta_block.dot(&s_beta);
                s_lambda.scaled_add(lambda, &dense);
                block_penalties.push(dense);
            }
            h.slice_mut(s![start..end, start..end])
                .scaled_add(1.0, &s_lambda);
            penalties_dense.push(block_penalties);
            penalty_cursor += count;
        }
        if trace_diagonal_ridge != 0.0 {
            for diag in 0..total {
                h[[diag, diag]] += trace_diagonal_ridge;
            }
        }

        let penalty_logdet_ridge = if options.ridge_policy.include_penalty_logdet {
            ridge
        } else {
            0.0
        };
        let per_block_penalty_refs: Vec<&[Array2<f64>]> =
            penalties_dense.iter().map(Vec::as_slice).collect();
        let penalty_logdet = crate::estimate::reml::unified::compute_block_penalty_logdet_derivs(
            &per_block_rho,
            &per_block_penalty_refs,
            penalty_logdet_ridge,
        )?;
        trace_s_pinv_sdot.assign(&penalty_logdet.first);
        if log_exact_work(self.y.len()) {
            log::info!(
                "[BMS batched outer-gradient] penalty assembly/logdet done n={} p={} rho={} elapsed={:.3}s",
                self.y.len(),
                total,
                rho.len(),
                penalty_started.elapsed().as_secs_f64()
            );
        }

        let spectral_started = std::time::Instant::now();
        let spectral =
            DenseSpectralOperator::from_symmetric_with_mode(&h, self.pseudo_logdet_mode())?;
        let factor = spectral.logdet_gradient_factor();
        if log_exact_work(self.y.len()) {
            log::info!(
                "[BMS batched outer-gradient] spectral factor done n={} p={} rank={} elapsed={:.3}s",
                self.y.len(),
                total,
                factor.ncols(),
                spectral_started.elapsed().as_secs_f64()
            );
        }
        let mut trace_h_inv_hdot = Array1::<f64>::zeros(rho.len());
        let mut directions = Array2::<f64>::zeros((total, rho.len()));
        let direction_started = std::time::Instant::now();
        penalty_cursor = 0;
        for (block_idx, spec) in specs.iter().enumerate() {
            let (start, end) = ranges[block_idx];
            let beta_block = beta.slice(s![start..end]);
            for (local_idx, _penalty) in spec.penalties.iter().enumerate() {
                let idx = penalty_cursor + local_idx;
                let lambda = rho[idx].exp();
                let dense = &penalties_dense[block_idx][local_idx];
                trace_h_inv_hdot[idx] +=
                    spectral.trace_logdet_block_local(dense, lambda, start, end);
                let curvature_rhs = dense.dot(&beta_block).mapv(|value| lambda * value);
                let mut rhs = Array1::<f64>::zeros(total);
                rhs.slice_mut(s![start..end]).assign(&curvature_rhs);
                let v = spectral.solve(&rhs);
                directions.column_mut(idx).assign(&(-&v));
            }
            penalty_cursor += spec.penalties.len();
        }
        if log_exact_work(self.y.len()) {
            log::info!(
                "[BMS batched outer-gradient] direction solves done n={} p={} rho={} elapsed={:.3}s",
                self.y.len(),
                total,
                rho.len(),
                direction_started.elapsed().as_secs_f64()
            );
        }

        let started = std::time::Instant::now();
        // The workspace's projected trace path is full-data only and
        // does not consult `options.outer_score_subsample`. When a
        // subsample is active (auto-derived above or explicitly
        // supplied) we route through the family-side fallback which
        // honors the mask; otherwise the workspace fast path is fine.
        let workspace_traces = if options.outer_score_subsample.is_some() {
            None
        } else if let Some(workspace) = hessian_workspace.as_ref() {
            workspace.projected_directional_derivative_traces(factor, &directions)?
        } else {
            None
        };
        let correction_traces = if let Some(traces) = workspace_traces {
            traces
        } else {
            let owned_cache = if let Some(subsample) = options.outer_score_subsample.as_ref() {
                self.build_exact_eval_cache_for_selected_context_rows(
                    block_states,
                    options,
                    subsample.mask.as_slice(),
                )?
            } else {
                self.build_exact_eval_cache_with_options(block_states, Some(options))?
            };
            if options.outer_score_subsample.is_some() {
                let weighted_rows = outer_weighted_rows(options, self.y.len());
                self.batched_rho_correction_logdet_traces_for_rows(
                    block_states,
                    &owned_cache,
                    factor,
                    &directions,
                    &weighted_rows,
                )?
            } else {
                self.batched_rho_correction_logdet_traces_full_rows(
                    block_states,
                    &owned_cache,
                    factor,
                    &directions,
                )?
            }
        };
        trace_h_inv_hdot += &correction_traces;
        if log_exact_work(self.y.len()) {
            log::info!(
                "[BMS batched outer-gradient] done n={} p={} rho={} trace_elapsed={:.3}s total_elapsed={:.3}s",
                self.y.len(),
                total,
                rho.len(),
                started.elapsed().as_secs_f64(),
                batched_started.elapsed().as_secs_f64()
            );
        }

        Ok(Some(BatchedOuterGradientTerms {
            objective_theta,
            trace_h_inv_hdot,
            trace_s_pinv_sdot,
        }))
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        self.validate_exact_monotonicity(block_states)?;
        self.evaluate_blockwise_exact_newton(block_states)
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        Self::log_likelihood_only_with_options(self, block_states, &BlockwiseFitOptions::default())
    }

    /// Options-aware override: outer hyper-derivative callers may pass a row
    /// subsample, while coefficient inner line searches clear that option and
    /// evaluate the exact full-data objective.
    fn log_likelihood_only_with_options(
        &self,
        block_states: &[ParameterBlockState],
        options: &BlockwiseFitOptions,
    ) -> Result<f64, String> {
        Self::log_likelihood_only_with_options(self, block_states, options)
    }

    fn supports_log_likelihood_early_exit(&self) -> bool {
        true
    }

    fn joint_line_search_log_likelihood_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<(f64, Arc<dyn ExactNewtonJointHessianWorkspace>)>, String> {
        let Some(workspace) =
            self.exact_newton_joint_hessian_workspace_with_options(block_states, specs, options)?
        else {
            return Ok(None);
        };
        let log_likelihood = match workspace.joint_log_likelihood_evaluation()? {
            Some(value) => value,
            None => Self::log_likelihood_only_with_options(self, block_states, options)?,
        };
        Ok(Some((log_likelihood, workspace)))
    }

    fn has_explicit_joint_hessian(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        let slices = block_slices(self);
        if slices.total >= 512 {
            return Ok(None);
        }
        if !self.effective_flex_active(block_states)? {
            let kern = BernoulliRigidRowKernel::new(self.clone(), block_states.to_vec());
            let cache = build_row_kernel_cache(&kern, &crate::families::row_kernel::RowSet::All)?;
            return Ok(Some(row_kernel_hessian_dense(
                &kern,
                &cache,
                &crate::families::row_kernel::RowSet::All,
            )));
        }

        // Build the dense joint Hessian by accumulating per-row primary
        // Hessians once per row and pulling each one back through the design
        // matrices via `BernoulliBlockHessianAccumulator::add_pullback`. The
        // earlier implementation materialized the dense matrix column-by-column
        // by calling `exact_newton_joint_hessian_matvec_from_cache` once per
        // unit basis vector, and each matvec re-ran
        // `compute_row_analytic_flex_into` (cell partition + cell-moment
        // evaluation + basis evaluation) for every row. That made the dense
        // build cost `slices.total * n * per_row_cell_work`. The flex
        // per-row work is direction-independent, so the column loop was
        // recomputing the same primary-space Hessian `slices.total` times.
        // For the bern_scorewarp / bern_sw_frailty integration cases the
        // outer factor of `slices.total ≈ 35` blew the optimizer past the
        // 240s timeout even on a 300-row dataset.
        //
        // The single-pass form below evaluates the per-row primary Hessian
        // exactly once per row and lets the existing accumulator distribute
        // every (a, b) entry into the marginal/logslope/h/w block targets.
        let cache = self.build_exact_eval_cache_with_order(block_states)?;
        self.exact_newton_joint_hessian_dense_from_cache(block_states, &cache)
            .map(Some)
    }

    fn exact_newton_joint_gradient_evaluation(
        &self,
        block_states: &[ParameterBlockState],
        block_specs: &[ParameterBlockSpec],
    ) -> Result<Option<ExactNewtonJointGradientEvaluation>, String> {
        assert!(block_specs.len() <= isize::MAX as usize);
        self.validate_exact_monotonicity(block_states)?;
        if !self.effective_flex_active(block_states)? {
            let kern = BernoulliRigidRowKernel::new(self.clone(), block_states.to_vec());
            let cache = build_row_kernel_cache(&kern, &crate::families::row_kernel::RowSet::All)?;
            return Ok(Some(ExactNewtonJointGradientEvaluation {
                log_likelihood: row_kernel_log_likelihood(
                    &cache,
                    &crate::families::row_kernel::RowSet::All,
                ),
                gradient: Self::exact_newton_score_from_objective_gradient(row_kernel_gradient(
                    &kern,
                    &cache,
                    &crate::families::row_kernel::RowSet::All,
                )),
            }));
        }

        let cache = self.build_exact_eval_cache_with_order(block_states)?;
        self.exact_newton_joint_gradient_evaluation_from_cache(block_states, &cache)
            .map(Some)
    }

    fn requires_joint_outer_hyper_path(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        block_specs: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        assert!(block_specs.len() <= isize::MAX as usize);
        if !self.effective_flex_active(block_states)? {
            // Rigid path: use generic RowKernel<2> operator
            let kern = BernoulliRigidRowKernel::new(self.clone(), block_states.to_vec());
            Ok(Some(Arc::new(RowKernelHessianWorkspace::new(kern)?)))
        } else {
            // Flex path: keep existing workspace for variable-K primary space
            Ok(Some(Arc::new(
                BernoulliMarginalSlopeExactNewtonJointHessianWorkspace::new(
                    self.clone(),
                    block_states.to_vec(),
                    BlockwiseFitOptions::default(),
                )?,
            )))
        }
    }

    fn exact_newton_joint_hessian_workspace_with_options(
        &self,
        block_states: &[ParameterBlockState],
        block_specs: &[ParameterBlockSpec],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        assert!(block_specs.len() <= isize::MAX as usize);
        if !self.effective_flex_active(block_states)? {
            // Rigid path: RowKernel<2> operator wired through the supplied
            // `RowSet`. With no outer subsample this is `RowSet::All`
            // (full-data, bit-identical to the pre-threading behaviour);
            // with an outer subsample installed, the cache and every
            // assembly function honour it uniformly via the Horvitz–Thompson
            // weights carried on each `WeightedOuterRow`.
            let kern = BernoulliRigidRowKernel::new(self.clone(), block_states.to_vec());
            let rows = crate::families::row_kernel::RowSet::from_options(options, self.y.len());
            Ok(Some(Arc::new(RowKernelHessianWorkspace::with_rows(
                kern, rows,
            )?)))
        } else {
            // Flex path: store the options on the workspace so the
            // directional-derivative methods route through the
            // `_with_options` helpers and pick up the outer-row subsample.
            Ok(Some(Arc::new(
                BernoulliMarginalSlopeExactNewtonJointHessianWorkspace::new(
                    self.clone(),
                    block_states.to_vec(),
                    options.clone(),
                )?,
            )))
        }
    }

    fn inner_coefficient_hessian_hvp_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        // The workspace impl above unconditionally returns `Some(workspace)`
        // — the rigid path produces a `RowKernelHessianWorkspace` and the
        // flex path produces a
        // `BernoulliMarginalSlopeExactNewtonJointHessianWorkspace`. Both
        // route the joint Hessian through Hv operators rather than dense
        // assembly. This advertises β-space representation support only; it
        // does not imply a profiled outer θ-HVP.
        parameter_block_specs_match_rows(specs, self.y.len())
    }

    fn inner_joint_workspace_gradient_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        parameter_block_specs_match_rows(specs, self.y.len())
    }

    fn inner_joint_workspace_log_likelihood_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        parameter_block_specs_match_rows(specs, self.y.len())
    }

    /// Force the matrix-free inner-Newton/PCG path for BMS flex at biobank
    /// scale, on top of the generic `use_joint_matrix_free_path` heuristic.
    ///
    /// At n≈195k with `linkwiggle()` / `logslope_formula linkwiggle()`, dense
    /// joint-H assembly streams every row through the expensive flex row
    /// kernel and pays a BLAS-3 design-matrix gram per chunk on top — ~63s
    /// per inner cycle. Each HVP reuses the row stream at near-gradient cost
    /// (~3s), and PCG with the joint penalty preconditioner typically
    /// converges in a handful of iters. The generic gate only fires for
    /// `p >= 128`, but BMS-flex per-row work is heavy enough that the matrix-
    /// free path wins well below that — drop the `p` floor for this family.
    fn prefers_matrix_free_inner_joint(
        &self,
        specs: &[ParameterBlockSpec],
        states: &[ParameterBlockState],
    ) -> bool {
        assert!(specs.len() <= isize::MAX as usize);
        if self.y.len() < 16_384 {
            return false;
        }
        self.effective_flex_active(states).unwrap_or(false)
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if !self.effective_flex_active(block_states)? {
            let kern = BernoulliRigidRowKernel::new(self.clone(), block_states.to_vec());
            let sl = d_beta_flat.as_slice().ok_or("non-contiguous d_beta")?;
            crate::families::row_kernel::row_kernel_directional_derivative(
                &kern,
                &crate::families::row_kernel::RowSet::All,
                sl,
            )
            .map(Some)
        } else {
            let cache = self.build_exact_eval_cache(block_states)?;
            self.exact_newton_joint_hessian_directional_derivative_from_cache(
                block_states,
                d_beta_flat,
                &cache,
            )
        }
    }

    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if !self.effective_flex_active(block_states)? {
            let kern = BernoulliRigidRowKernel::new(self.clone(), block_states.to_vec());
            let su = d_beta_u_flat.as_slice().ok_or("non-contiguous d_beta_u")?;
            let sv = d_beta_v_flat.as_slice().ok_or("non-contiguous d_beta_v")?;
            crate::families::row_kernel::row_kernel_second_directional_derivative(
                &kern,
                &crate::families::row_kernel::RowSet::All,
                su,
                sv,
            )
            .map(Some)
        } else {
            let cache = self.build_exact_eval_cache(block_states)?;
            self.exact_newton_joint_hessiansecond_directional_derivative_from_cache(
                block_states,
                d_beta_u_flat,
                d_beta_v_flat,
                &cache,
            )
        }
    }

    fn exact_newton_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        if self.is_sigma_aux_index(derivative_blocks, psi_index) {
            return self.sigma_exact_joint_psi_terms(block_states, specs);
        }
        let cache = self.build_exact_eval_cache(block_states)?;
        self.exact_newton_joint_psi_terms_from_cache(
            block_states,
            derivative_blocks,
            psi_index,
            &cache,
        )
    }

    fn exact_newton_joint_psisecond_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        block_specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        assert!(block_specs.len() <= isize::MAX as usize);
        if self.is_sigma_aux_index(derivative_blocks, psi_i)
            || self.is_sigma_aux_index(derivative_blocks, psi_j)
        {
            if self.is_sigma_aux_index(derivative_blocks, psi_i)
                && self.is_sigma_aux_index(derivative_blocks, psi_j)
            {
                return self.sigma_exact_joint_psisecond_order_terms(block_states);
            }
            return Err("bernoulli marginal-slope mixed log-sigma/spatial psi second derivatives require cross auxiliary terms; only pure log-sigma second derivatives are supported"
                        .to_string());
        }
        let cache = self.build_exact_eval_cache(block_states)?;
        self.exact_newton_joint_psisecond_order_terms_from_cache(
            block_states,
            derivative_blocks,
            psi_i,
            psi_j,
            &cache,
        )
    }

    fn exact_newton_joint_psihessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert!(block_specs.len() <= isize::MAX as usize);
        if self.is_sigma_aux_index(derivative_blocks, psi_index) {
            return self
                .sigma_exact_joint_psihessian_directional_derivative(block_states, d_beta_flat);
        }
        let cache = self.build_exact_eval_cache(block_states)?;
        self.exact_newton_joint_psihessian_directional_derivative_from_cache(
            block_states,
            derivative_blocks,
            psi_index,
            d_beta_flat,
            &cache,
        )
    }

    fn exact_newton_joint_psi_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        Ok(Some(Arc::new(
            BernoulliMarginalSlopeExactNewtonJointPsiWorkspace::new(
                self.clone(),
                block_states.to_vec(),
                specs.to_vec(),
                derivative_blocks.to_vec(),
                BlockwiseFitOptions::default(),
            )?,
        )))
    }

    fn exact_newton_joint_psi_workspace_with_options(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        Ok(Some(Arc::new(
            BernoulliMarginalSlopeExactNewtonJointPsiWorkspace::new(
                self.clone(),
                block_states.to_vec(),
                specs.to_vec(),
                derivative_blocks.to_vec(),
                options.clone(),
            )?,
        )))
    }

    fn block_linear_constraints(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        spec: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        if block_states.len() == usize::MAX
            || block_idx == usize::MAX
            || spec.design.ncols() == usize::MAX
        {
            return Err("unreachable bernoulli marginal-slope constraint state".to_string());
        }
        if self.score_block_index().is_some_and(|idx| block_idx == idx) {
            return Ok(self
                .score_warp
                .as_ref()
                .map(DeviationRuntime::structural_monotonicity_constraints));
        }
        if self.link_block_index().is_some_and(|idx| block_idx == idx) {
            return Ok(self
                .link_dev
                .as_ref()
                .map(DeviationRuntime::structural_monotonicity_constraints));
        }
        Ok(None)
    }

    fn post_update_block_beta(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        block_spec: &ParameterBlockSpec,
        beta: Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        assert!(!block_spec.name.is_empty());
        self.validate_exact_block_state_shapes(block_states)?;
        if block_idx >= block_states.len() {
            return Err(format!(
                "post-update block index {} out of range for {} blocks",
                block_idx,
                block_states.len()
            ));
        }
        if self.score_block_index().is_some_and(|idx| block_idx == idx)
            && let (Some(runtime), Some(score)) =
                (&self.score_warp, self.score_block_state(block_states)?)
        {
            let current = &score.beta;
            if current.len() != beta.len() {
                return Err(format!(
                    "score-warp post-update beta length mismatch: current={}, proposed={}",
                    current.len(),
                    beta.len()
                ));
            }
            validate_monotone_structural_feasible(runtime, current, "score_warp_dev current")?;
            validate_monotone_structural_feasible(runtime, &beta, "score_warp_dev proposed")?;
            return Ok(beta);
        }
        if self.link_block_index().is_some_and(|idx| block_idx == idx)
            && let (Some(runtime), Some(link)) =
                (&self.link_dev, self.link_block_state(block_states)?)
        {
            let current = &link.beta;
            if current.len() != beta.len() {
                return Err(format!(
                    "link-deviation post-update beta length mismatch: current={}, proposed={}",
                    current.len(),
                    beta.len()
                ));
            }
            validate_monotone_structural_feasible(runtime, current, "link_dev current")?;
            validate_monotone_structural_feasible(runtime, &beta, "link_dev proposed")?;
            return Ok(beta);
        }
        Ok(beta)
    }
}

impl BernoulliMarginalSlopeExactNewtonJointHessianWorkspace {
    pub(super) fn new(
        family: BernoulliMarginalSlopeFamily,
        block_states: Vec<ParameterBlockState>,
        options: BlockwiseFitOptions,
    ) -> Result<Self, String> {
        let started = std::time::Instant::now();
        let heartbeat_guard = crate::heartbeat::scope(format!(
            "BMS Hessian-workspace build n={} p={} subsample_rows={}",
            family.y.len(),
            block_slices(&family).total,
            options
                .outer_score_subsample
                .as_ref()
                .map_or(family.y.len(), |subsample| subsample.len())
        ));
        if log_exact_work(family.y.len()) {
            log::info!(
                "[BMS Hessian-workspace] build start n={} p={} subsample_rows={}",
                family.y.len(),
                block_slices(&family).total,
                options
                    .outer_score_subsample
                    .as_ref()
                    .map_or(family.y.len(), |subsample| subsample.len())
            );
        }
        let mut cache =
            family.build_exact_eval_cache_with_options(&block_states, Some(&options))?;
        // Materialize per-row primary Hessians at construction time. The
        // matrix-free CG / inner-Newton loops contract these against many
        // trial directions at the same β, so caching the `r×r` blocks once
        // amortizes the cell-moment + flex-jet rebuild over every Hv product.
        cache.row_primary_hessians =
            family.build_row_primary_hessian_cache(&block_states, &cache)?;
        if log_exact_work(family.y.len()) {
            log::info!(
                "[BMS Hessian-workspace] build done n={} p={} primary_hessian_cache={} elapsed={:.3}s",
                family.y.len(),
                cache.slices.total,
                cache.row_primary_hessians.is_some(),
                started.elapsed().as_secs_f64()
            );
        }
        let workspace = Self::from_arc_cache(family, block_states, Arc::new(cache), options);
        drop(heartbeat_guard);
        workspace
    }

    pub(super) fn from_arc_cache(
        family: BernoulliMarginalSlopeFamily,
        block_states: Vec<ParameterBlockState>,
        cache: Arc<BernoulliMarginalSlopeExactEvalCache>,
        options: BlockwiseFitOptions,
    ) -> Result<Self, String> {
        Ok(Self {
            family,
            block_states,
            cache,
            matvec_calls: AtomicUsize::new(0),
            fused_gradient_dense: OnceLock::new(),
            options,
        })
    }

    pub(super) fn fused_gradient_dense(
        &self,
    ) -> Result<Arc<ExactNewtonJointFusedDenseEvaluation>, String> {
        self.fused_gradient_dense
            .get_or_init(|| {
                self.family
                    .exact_newton_joint_fused_gradient_dense_from_cache(
                        &self.block_states,
                        &self.cache,
                    )
                    .map(Arc::new)
            })
            .clone()
    }

    /// Matrix-free inner-Newton/CG route for BMS flex large-n.
    ///
    /// Auto-selected when the workspace's per-row primary Hessian cache could
    /// not be materialized (`n*r*r*8 > row_primary_cache_budget`). In that
    /// regime the dense joint-H build streams all `n` rows and pays the full
    /// flex row-kernel cost per chunk plus a BLAS-3 design-matrix gram on top;
    /// at biobank shape (n≈195k, p≈44) that pushes one dense build past 60s
    /// while each HVP reuses the same row stream at ~gradient-pass cost (~3s).
    /// PCG with the joint penalty preconditioner typically converges in a
    /// handful of HVPs, so routing the inner solve through the operator path
    /// beats per-cycle dense reassembly.
    pub(super) fn matrix_free_inner_route(&self) -> bool {
        if self.cache.row_primary_hessians.is_some() {
            return false;
        }
        match self.family.effective_flex_active(&self.block_states) {
            Ok(true) => {}
            _ => return false,
        }
        // Tiny problems should still take the dense path: a single dense build
        // is cheaper than CG bookkeeping when row counts are small.
        self.family.y.len() >= 16_384
    }
}

impl ExactNewtonJointHessianWorkspace for BernoulliMarginalSlopeExactNewtonJointHessianWorkspace {
    fn hessian_dense(&self) -> Result<Option<Array2<f64>>, String> {
        if self.cache.slices.total >= 512 {
            return Ok(None);
        }
        if self.matrix_free_inner_route() {
            // Route the inner-Newton solve through `hessian_matvec` /
            // `hessian_diagonal`. Callers that strictly need a dense matrix
            // (outer-Hessian assembly, logdet factor) reach the fused dense
            // build through `hessian_dense_forced`.
            if log_exact_work(self.family.y.len()) {
                log::info!(
                    "[BMS inner] route=matrix-free-CG n={} p={} primary_hessian_cache=false reason=flex+large-n",
                    self.family.y.len(),
                    self.cache.slices.total
                );
            }
            return Ok(None);
        }
        // Device dense-block shortcut: when the row-primary Hessian is pinned on
        // the GPU and `p_total` fits the shared-memory cap, dispatch the
        // device-resident dense build instead of the CPU fused gradient pass.
        #[cfg(target_os = "linux")]
        {
            if let Some(device_state) = self.cache.row_primary_hessians.device() {
                let p_total = self.cache.slices.total;
                if p_total <= crate::gpu::bms_flex_row::DENSE_BLOCK_MAX_P {
                    match crate::gpu::bms_flex_row::launch_bms_flex_row_dense_block(device_state) {
                        Ok(flat) => {
                            let h_arr =
                                Array2::from_shape_vec((p_total, p_total), flat).map_err(|e| {
                                    format!(
                                        "BMS hessian_dense: dense_block reshape \
                                             {p_total}x{p_total} failed: {e}"
                                    )
                                })?;
                            return Ok(Some(h_arr));
                        }
                        Err(err) => {
                            log::info!(
                                "[BMS hessian_dense] gpu_dense_block_failed: {err}; \
                                 falling back to CPU fused-gradient dense build"
                            );
                        }
                    }
                }
            }
        }
        if log_exact_work(self.family.y.len()) {
            log::info!(
                "[BMS inner] route=dense n={} p={} primary_hessian_cache={}",
                self.family.y.len(),
                self.cache.slices.total,
                self.cache.row_primary_hessians.is_some()
            );
        }
        self.fused_gradient_dense()
            .map(|fused| Some(fused.hessian.clone()))
    }

    fn hessian_dense_forced(&self) -> Result<Option<Array2<f64>>, String> {
        // Callers that genuinely require a dense joint Hessian (e.g. outer
        // batched-gradient assembly that pulls back the dense `H_β`) bypass
        // the matrix-free route gate above. The fused row pass is still the
        // structural direct-dense path here — column-basis HVP fallback would
        // be `total` extra row sweeps, which is exactly what the matrix-free
        // inner solver is designed to amortize across far fewer iterations.
        if self.cache.slices.total >= 512 {
            return Ok(None);
        }
        // Block 9 Phase 6 shortcut: when the row-primary Hessian is already
        // pinned on the GPU and `p_total` fits the dense-block kernel's
        // shared-memory cap, dispatch the device-resident dense build
        // instead of round-tripping through the CPU fused gradient pass.
        // Numerics match the CPU pullback to within reduction-order f.p.
        // noise (see `bms_flex_row_dense_block_kernel_matches_cpu_pullback`).
        #[cfg(target_os = "linux")]
        {
            if let Some(device_state) = self.cache.row_primary_hessians.device() {
                let p_total = self.cache.slices.total;
                if p_total <= crate::gpu::bms_flex_row::DENSE_BLOCK_MAX_P {
                    match crate::gpu::bms_flex_row::launch_bms_flex_row_dense_block(device_state) {
                        Ok(flat) => {
                            let h_arr =
                                Array2::from_shape_vec((p_total, p_total), flat).map_err(|e| {
                                    format!(
                                        "BMS hessian_dense_forced: dense_block reshape \
                                             {p_total}x{p_total} failed: {e}"
                                    )
                                })?;
                            return Ok(Some(h_arr));
                        }
                        Err(err) => {
                            log::info!(
                                "[BMS hessian_dense_forced] gpu_dense_block_failed: {err}; \
                                 falling back to CPU fused-gradient dense build"
                            );
                        }
                    }
                }
            }
        }
        self.fused_gradient_dense()
            .map(|fused| Some(fused.hessian.clone()))
    }

    fn joint_log_likelihood_evaluation(&self) -> Result<Option<f64>, String> {
        self.family
            .log_likelihood_from_exact_cache(&self.block_states, &self.cache)
            .map(Some)
    }

    fn joint_gradient_evaluation(
        &self,
    ) -> Result<Option<ExactNewtonJointGradientEvaluation>, String> {
        if self.cache.slices.total < 512 && !self.matrix_free_inner_route() {
            // The only current consumer of workspace-side joint gradients is
            // the exact joint-Newton path. For bounded dense systems it will
            // request the dense Hessian in the same cycle, so build the fused
            // row pass once and let `hessian_dense` reuse it. When the
            // matrix-free inner route is active, the inner solver will pull
            // the Hessian through HVPs instead, so the gradient pass must
            // stay gradient-only and not force a dense build.
            return self.fused_gradient_dense().map(|fused| {
                Some(ExactNewtonJointGradientEvaluation {
                    log_likelihood: fused.gradient.log_likelihood,
                    gradient: fused.gradient.gradient.clone(),
                })
            });
        }
        self.family
            .exact_newton_joint_gradient_evaluation_from_cache(&self.block_states, &self.cache)
            .map(Some)
    }

    fn hessian_matvec_available(&self) -> bool {
        true
    }

    fn hessian_matvec(&self, beta_flat: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        // Hv-action against the full coefficient Hessian is consumed by inner
        // PCG / inner-Newton paths (not outer-only score), so the row mask
        // does not apply here — keep full-data semantics.
        let call = self.matvec_calls.fetch_add(1, Ordering::Relaxed) + 1;
        let started = std::time::Instant::now();
        let heartbeat_guard = crate::heartbeat::scope(format!(
            "BMS Hessian-Hv call={call} n={} p={}",
            self.family.y.len(),
            self.cache.slices.total
        ));
        let result = self
            .family
            .exact_newton_joint_hessian_matvec_from_cache(
                beta_flat,
                &self.block_states,
                &self.cache,
            )
            .map(Some);
        if log_exact_work(self.family.y.len()) && (call <= 3 || call.is_power_of_two()) {
            log::info!(
                "[BMS Hessian-Hv] call={} n={} p={} primary_hessian_cache={} elapsed={:.3}s",
                call,
                self.family.y.len(),
                self.cache.slices.total,
                self.cache.row_primary_hessians.is_some(),
                started.elapsed().as_secs_f64()
            );
        }
        drop(heartbeat_guard);
        result
    }

    fn hessian_matvec_into(&self, v: &Array1<f64>, out: &mut Array1<f64>) -> Result<bool, String> {
        let call = self.matvec_calls.fetch_add(1, Ordering::Relaxed) + 1;
        let started = std::time::Instant::now();
        let heartbeat_guard = crate::heartbeat::scope(format!(
            "BMS Hessian-Hv (into) call={call} n={} p={}",
            self.family.y.len(),
            self.cache.slices.total
        ));
        self.family
            .exact_newton_joint_hessian_matvec_from_cache_into(
                v,
                &self.block_states,
                &self.cache,
                out,
            )?;
        if log_exact_work(self.family.y.len()) && (call <= 3 || call.is_power_of_two()) {
            log::info!(
                "[BMS Hessian-Hv] call={} n={} p={} primary_hessian_cache={} elapsed={:.3}s (into)",
                call,
                self.family.y.len(),
                self.cache.slices.total,
                self.cache.row_primary_hessians.is_some(),
                started.elapsed().as_secs_f64()
            );
        }
        drop(heartbeat_guard);
        Ok(true)
    }

    fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        // Diagonal is consumed by inner preconditioners; full-data semantics
        // preserved (see `hessian_matvec`).
        self.family
            .exact_newton_joint_hessian_diagonal_from_cache(&self.block_states, &self.cache)
            .map(Some)
    }

    fn projected_directional_derivative_traces(
        &self,
        factor: &Array2<f64>,
        directions: &Array2<f64>,
    ) -> Result<Option<Array1<f64>>, String> {
        let traces = if self.options.outer_score_subsample.is_some() {
            let weighted_rows = outer_weighted_rows(&self.options, self.family.y.len());
            self.family.batched_rho_correction_logdet_traces_for_rows(
                &self.block_states,
                &self.cache,
                factor,
                directions,
                &weighted_rows,
            )?
        } else {
            self.family.batched_rho_correction_logdet_traces_full_rows(
                &self.block_states,
                &self.cache,
                factor,
                directions,
            )?
        };
        Ok(Some(traces))
    }

    fn directional_derivative_operator(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        self.family
            .exact_newton_joint_hessian_directional_derivative_operator_from_cache_with_options(
                &self.block_states,
                d_beta_flat,
                &self.cache,
                &self.options,
            )
    }

    fn directional_derivative_operators(
        &self,
        d_beta_flats: &[Array1<f64>],
    ) -> Result<Vec<Option<Arc<dyn HyperOperator>>>, String> {
        self.family
            .exact_newton_joint_hessian_directional_derivative_operators_from_cache_with_options(
                &self.block_states,
                d_beta_flats,
                &self.cache,
                &self.options,
            )
    }

    fn directional_derivative(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .exact_newton_joint_hessian_directional_derivative_from_cache_with_options(
                &self.block_states,
                d_beta_flat,
                &self.cache,
                &self.options,
            )
    }

    fn second_directional_derivative_operator(
        &self,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        self.family
            .exact_newton_joint_hessiansecond_directional_derivative_operator_from_cache_with_options(
                &self.block_states,
                d_beta_u_flat,
                d_beta_v_flat,
                &self.cache,
                &self.options,
            )
    }

    fn second_directional_derivative(
        &self,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .exact_newton_joint_hessiansecond_directional_derivative_from_cache_with_options(
                &self.block_states,
                d_beta_u_flat,
                d_beta_v_flat,
                &self.cache,
                &self.options,
            )
    }
}

impl BernoulliMarginalSlopeFamily {
    pub(super) fn block_ranges_from_specs(specs: &[ParameterBlockSpec]) -> Vec<(usize, usize)> {
        let mut cursor = 0usize;
        specs
            .iter()
            .map(|spec| {
                let start = cursor;
                cursor += spec.design.ncols();
                (start, cursor)
            })
            .collect()
    }

    pub(super) fn flatten_block_state_betas_for_specs(
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Array1<f64>, String> {
        if block_states.len() != specs.len() {
            return Err(format!(
                "bernoulli marginal-slope batched gradient state/spec mismatch: states={}, specs={}",
                block_states.len(),
                specs.len()
            ));
        }
        let total = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
        let mut beta = Array1::<f64>::zeros(total);
        let mut cursor = 0usize;
        for (idx, (state, spec)) in block_states.iter().zip(specs.iter()).enumerate() {
            let width = spec.design.ncols();
            if state.beta.len() != width {
                return Err(format!(
                    "bernoulli marginal-slope batched gradient block {idx} beta length {} != spec width {}",
                    state.beta.len(),
                    width
                ));
            }
            beta.slice_mut(s![cursor..cursor + width])
                .assign(&state.beta);
            cursor += width;
        }
        Ok(beta)
    }

    pub(super) fn row_factor_primary_projection(
        &self,
        row: usize,
        slices: &BlockSlices,
        primary: &PrimarySlices,
        factor: &Array2<f64>,
        out: &mut [f64],
    ) -> Result<(), String> {
        let rank = factor.ncols();
        if out.len() != primary.total * rank {
            return Err(format!(
                "primary projection scratch length {} != {}",
                out.len(),
                primary.total * rank
            ));
        }
        out.fill(0.0);
        for col in 0..rank {
            out[primary.q * rank + col] = self
                .marginal_design
                .dot_row_view(row, factor.slice(s![slices.marginal.clone(), col]));
            out[primary.logslope * rank + col] = self
                .logslope_design
                .dot_row_view(row, factor.slice(s![slices.logslope.clone(), col]));
        }
        if let (Some(block_range), Some(primary_range)) = (slices.h.as_ref(), primary.h.as_ref()) {
            for (local, block_idx) in block_range.clone().enumerate() {
                let primary_idx = primary_range.start + local;
                for col in 0..rank {
                    out[primary_idx * rank + col] = factor[[block_idx, col]];
                }
            }
        }
        if let (Some(block_range), Some(primary_range)) = (slices.w.as_ref(), primary.w.as_ref()) {
            for (local, block_idx) in block_range.clone().enumerate() {
                let primary_idx = primary_range.start + local;
                for col in 0..rank {
                    out[primary_idx * rank + col] = factor[[block_idx, col]];
                }
            }
        }
        Ok(())
    }

    pub(super) fn row_primary_gram_from_projection(
        primary_total: usize,
        rank: usize,
        projection: &[f64],
    ) -> Vec<f64> {
        let mut gram = vec![0.0; primary_total * primary_total];
        for a in 0..primary_total {
            for b in 0..=a {
                let mut sum = 0.0;
                let a_base = a * rank;
                let b_base = b * rank;
                for col in 0..rank {
                    sum += projection[a_base + col] * projection[b_base + col];
                }
                gram[a * primary_total + b] = sum;
                gram[b * primary_total + a] = sum;
            }
        }
        gram
    }

    pub(super) fn primary_tail_block_pairs(
        slices: &BlockSlices,
        primary: &PrimarySlices,
    ) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        if let (Some(block_range), Some(primary_range)) = (slices.h.as_ref(), primary.h.as_ref()) {
            out.extend(
                block_range
                    .clone()
                    .enumerate()
                    .map(|(offset, block_idx)| (primary_range.start + offset, block_idx)),
            );
        }
        if let (Some(block_range), Some(primary_range)) = (slices.w.as_ref(), primary.w.as_ref()) {
            out.extend(
                block_range
                    .clone()
                    .enumerate()
                    .map(|(offset, block_idx)| (primary_range.start + offset, block_idx)),
            );
        }
        out
    }

    pub(super) fn primary_tail_tail_gram(
        primary_total: usize,
        rank: usize,
        factor: &Array2<f64>,
        tail_pairs: &[(usize, usize)],
    ) -> Vec<f64> {
        let mut gram = vec![0.0; primary_total * primary_total];
        for (a_pos, &(primary_a, block_a)) in tail_pairs.iter().enumerate() {
            for &(primary_b, block_b) in tail_pairs.iter().take(a_pos + 1) {
                let mut sum = 0.0;
                for col in 0..rank {
                    sum += factor[[block_a, col]] * factor[[block_b, col]];
                }
                gram[primary_a * primary_total + primary_b] = sum;
                gram[primary_b * primary_total + primary_a] = sum;
            }
        }
        gram
    }

    pub(super) fn row_primary_trace_contract(third: &Array2<f64>, gram: &[f64]) -> f64 {
        let r = third.nrows();
        assert_eq!(third.ncols(), r);
        assert_eq!(gram.len(), r * r);
        let mut total = 0.0;
        for a in 0..r {
            for b in 0..r {
                total += third[[a, b]] * gram[a * r + b];
            }
        }
        total
    }

    pub(super) fn row_primary_third_contracted_many_with_moments(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        row_dirs: &[Array1<f64>],
    ) -> Result<Vec<Array2<f64>>, String> {
        let primary = &cache.primary;
        let r = primary.total;
        if row_dirs.is_empty() {
            return Ok(Vec::new());
        }
        if let Some((idx, dir)) = row_dirs.iter().enumerate().find(|(_, dir)| dir.len() != r) {
            return Err(format!(
                "bernoulli marginal-slope row third direction {idx} length {} != {r}",
                dir.len()
            ));
        }
        if row_dirs.len() == 1 {
            return Ok(vec![
                self.row_primary_third_contracted_recompute_with_moments(
                    row,
                    block_states,
                    cache,
                    row_ctx,
                    &row_dirs[0],
                )?,
            ]);
        }
        if !self.effective_flex_active(block_states)? {
            let t = self.rigid_third_full_cached(block_states, cache, row)?;
            return row_dirs
                .iter()
                .map(|dir| {
                    let m = contract_third_full(t, dir[0], dir[1]);
                    let mut out = Array2::<f64>::zeros((2, 2));
                    out[[0, 0]] = m[0][0];
                    out[[0, 1]] = m[0][1];
                    out[[1, 0]] = m[1][0];
                    out[[1, 1]] = m[1][1];
                    Ok(out)
                })
                .collect();
        }
        if !row_ctx.intercept.is_finite() || !row_ctx.m_a.is_finite() || row_ctx.m_a <= 0.0 {
            return Err(
                "non-finite flexible row context in batched third-order contraction".to_string(),
            );
        }

        let point = self.primary_point_from_block_states(row, block_states, primary)?;
        let (q, b, beta_h_owned, beta_w_owned) = self.primary_point_components(&point, primary);
        let beta_h = beta_h_owned.as_ref();
        let beta_w = beta_w_owned.as_ref();
        if let Some(grid) = self.latent_measure.empirical_grid_for_training_row(row)? {
            return row_dirs
                .iter()
                .map(|dir| {
                    self.empirical_flex_row_third_contracted_recompute(
                        row, primary, q, b, beta_h, beta_w, row_ctx, dir, &grid,
                    )
                })
                .collect();
        }

        use super::exact_kernel as exact;

        let a = row_ctx.intercept;
        let marginal = self.marginal_link_map(q)?;
        let h_range = primary.h.as_ref();
        let w_range = primary.w.as_ref();
        let score_runtime = self.score_warp.as_ref();
        let link_runtime = self.link_dev.as_ref();
        let scale = self.probit_frailty_scale();
        let zero_family = vec![[0.0; 4]; r];
        let n_dirs = row_dirs.len();

        let mut f_a = 0.0;
        let mut f_aa = 0.0;
        let mut f_u = Array1::<f64>::zeros(r);
        let mut f_au = Array1::<f64>::zeros(r);
        let mut f_uv = Array2::<f64>::zeros((r, r));
        let mut f_a_dir = vec![0.0; n_dirs];
        let mut f_aa_dir = vec![0.0; n_dirs];
        let mut f_au_dir = vec![0.0; n_dirs * r];
        let mut f_uv_dir = vec![0.0; n_dirs * r * r];

        let owned_cells;
        let cells: &[CachedDenestedCellMoments] = if let Some(cached) = self
            .bundle_for_degree(block_states, cache, 15)?
            .and_then(|bundle| bundle.row(row, 15))
        {
            cached
        } else {
            let partitions = self.denested_partition_cells(a, b, beta_h, beta_w)?;
            owned_cells = partitions
                .into_iter()
                .map(|partition_cell| {
                    self.evaluate_cell_derivative_moments_lru(partition_cell.cell, 15)
                        .map(|state| CachedDenestedCellMoments {
                            partition_cell,
                            state,
                        })
                })
                .collect::<Result<Vec<_>, String>>()?;
            &owned_cells
        };

        for entry in cells {
            let partition_cell = entry.partition_cell;
            let cell = partition_cell.cell;
            let z_mid = exact::interval_probe_point(cell.left, cell.right)?;
            let u_mid = a + b * z_mid;
            let state = &entry.state;

            let (dc_da_raw, dc_db_raw) = exact::denested_cell_coefficient_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                b,
            );
            let (dc_daa_raw, dc_dab_raw, dc_dbb_raw) = exact::denested_cell_second_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                b,
            );
            let denested_third = exact::denested_cell_third_partials(partition_cell.link_span);
            let dc_da = scale_coeff4(dc_da_raw, scale);
            let dc_db = scale_coeff4(dc_db_raw, scale);
            let dc_daa = scale_coeff4(dc_daa_raw, scale);
            let dc_dab = scale_coeff4(dc_dab_raw, scale);
            let dc_dbb = scale_coeff4(dc_dbb_raw, scale);
            let dc_daab = scale_coeff4(denested_third.1, scale);
            let dc_dabb = scale_coeff4(denested_third.2, scale);
            let dc_dbbb = scale_coeff4(denested_third.3, scale);

            let mut coeff_u = vec![[0.0; 4]; r];
            let mut coeff_au = vec![[0.0; 4]; r];
            let mut coeff_bu = vec![[0.0; 4]; r];
            let mut coeff_aau = vec![[0.0; 4]; r];
            let mut coeff_abu = vec![[0.0; 4]; r];
            let mut coeff_bbu = vec![[0.0; 4]; r];

            coeff_u[1] = dc_db;
            coeff_au[1] = dc_dab;
            coeff_bu[1] = dc_dbb;
            coeff_aau[1] = dc_daab;
            coeff_abu[1] = dc_dabb;
            coeff_bbu[1] = dc_dbbb;

            if let (Some(h_range), Some(runtime)) = (h_range, score_runtime) {
                Self::for_each_deviation_basis_cubic_at(
                    runtime,
                    h_range,
                    z_mid,
                    "score-warp batched third direction",
                    |_, idx, basis_span| {
                        coeff_u[idx] = scale_coeff4(
                            exact::score_basis_cell_coefficients(basis_span, b),
                            scale,
                        );
                        coeff_bu[idx] = scale_coeff4(
                            exact::score_basis_cell_coefficients(basis_span, 1.0),
                            scale,
                        );
                        Ok(())
                    },
                )?;
            }

            if let (Some(w_range), Some(runtime)) = (w_range, link_runtime) {
                Self::for_each_deviation_basis_cubic_at(
                    runtime,
                    w_range,
                    u_mid,
                    "link-wiggle batched third direction",
                    |_, idx, basis_span| {
                        coeff_u[idx] = scale_coeff4(
                            exact::link_basis_cell_coefficients(basis_span, a, b),
                            scale,
                        );
                        let (dc_aw_raw, dc_bw_raw) =
                            exact::link_basis_cell_coefficient_partials(basis_span, a, b);
                        let (dc_aaw_raw, dc_abw_raw, dc_bbw_raw) =
                            exact::link_basis_cell_second_partials(basis_span, a, b);
                        coeff_au[idx] = scale_coeff4(dc_aw_raw, scale);
                        coeff_bu[idx] = scale_coeff4(dc_bw_raw, scale);
                        coeff_aau[idx] = scale_coeff4(dc_aaw_raw, scale);
                        coeff_abu[idx] = scale_coeff4(dc_abw_raw, scale);
                        coeff_bbu[idx] = scale_coeff4(dc_bbw_raw, scale);
                        Ok(())
                    },
                )?;
            }

            let coeff_jet = SparsePrimaryCoeffJetView::new(
                1,
                h_range,
                w_range,
                &coeff_u,
                &coeff_au,
                &coeff_bu,
                &coeff_aau,
                &coeff_abu,
                &coeff_bbu,
                &zero_family,
                &zero_family,
                &zero_family,
                &zero_family,
            );

            f_a += exact::cell_first_derivative_from_moments(&dc_da, &state.moments)?;
            f_aa += exact::cell_second_derivative_from_moments(
                cell,
                &dc_da,
                &dc_da,
                &dc_daa,
                &state.moments,
            )?;
            for u in 1..r {
                f_u[u] +=
                    exact::cell_first_derivative_from_moments(&coeff_jet.first[u], &state.moments)?;
                f_au[u] += exact::cell_second_derivative_from_moments(
                    cell,
                    &dc_da,
                    &coeff_jet.first[u],
                    &coeff_jet.a_first[u],
                    &state.moments,
                )?;
            }
            for u in 1..r {
                for v in u..r {
                    let second_coeff =
                        coeff_jet.pair_from_b_family(coeff_jet.b_first, u, v, COEFF_SUPPORT_BHW);
                    let val = exact::cell_second_derivative_from_moments(
                        cell,
                        &coeff_jet.first[u],
                        &coeff_jet.first[v],
                        &second_coeff,
                        &state.moments,
                    )?;
                    f_uv[[u, v]] += val;
                    if u != v {
                        f_uv[[v, u]] += val;
                    }
                }
            }

            for (dir_idx, dir) in row_dirs.iter().enumerate() {
                let coeff_dir =
                    coeff_jet.directional_family(coeff_jet.first, dir, COEFF_SUPPORT_BHW);
                let coeff_a_dir =
                    coeff_jet.directional_family(coeff_jet.a_first, dir, COEFF_SUPPORT_BW);
                let coeff_aa_dir =
                    coeff_jet.directional_family(coeff_jet.aa_first, dir, COEFF_SUPPORT_BW);

                f_a_dir[dir_idx] += exact::cell_second_derivative_from_moments(
                    cell,
                    &dc_da,
                    &coeff_dir,
                    &coeff_a_dir,
                    &state.moments,
                )?;
                f_aa_dir[dir_idx] += exact::cell_third_derivative_from_moments(
                    cell,
                    &dc_da,
                    &dc_da,
                    &coeff_dir,
                    &dc_daa,
                    &coeff_a_dir,
                    &coeff_a_dir,
                    &coeff_aa_dir,
                    &state.moments,
                )?;

                let mut coeff_u_dir = vec![[0.0; 4]; r];
                let mut coeff_au_dir = vec![[0.0; 4]; r];
                for u in 1..r {
                    coeff_u_dir[u] = coeff_jet.param_directional_from_b_family(
                        coeff_jet.b_first,
                        u,
                        dir,
                        COEFF_SUPPORT_BHW,
                    );
                    coeff_au_dir[u] = coeff_jet.param_directional_from_b_family(
                        coeff_jet.ab_first,
                        u,
                        dir,
                        COEFF_SUPPORT_BW,
                    );
                }

                for u in 1..r {
                    f_au_dir[dir_idx * r + u] += exact::cell_third_derivative_from_moments(
                        cell,
                        &dc_da,
                        &coeff_jet.first[u],
                        &coeff_dir,
                        &coeff_jet.a_first[u],
                        &coeff_a_dir,
                        &coeff_u_dir[u],
                        &coeff_au_dir[u],
                        &state.moments,
                    )?;
                }

                let dir_base = dir_idx * r * r;
                for u in 1..r {
                    for v in u..r {
                        let second_coeff = coeff_jet.pair_from_b_family(
                            coeff_jet.b_first,
                            u,
                            v,
                            COEFF_SUPPORT_BHW,
                        );
                        let third_coeff = coeff_jet.pair_directional_from_bb_family(
                            coeff_jet.bb_first,
                            u,
                            v,
                            dir,
                            COEFF_SUPPORT_BW,
                        );
                        let dir_val = exact::cell_third_derivative_from_moments(
                            cell,
                            &coeff_jet.first[u],
                            &coeff_jet.first[v],
                            &coeff_dir,
                            &second_coeff,
                            &coeff_u_dir[u],
                            &coeff_u_dir[v],
                            &third_coeff,
                            &state.moments,
                        )?;
                        f_uv_dir[dir_base + u * r + v] += dir_val;
                        if u != v {
                            f_uv_dir[dir_base + v * r + u] += dir_val;
                        }
                    }
                }
            }
        }

        f_u[0] = -marginal.mu1;
        f_uv[[0, 0]] = -marginal.mu2;

        let inv_f_a = 1.0 / f_a;
        let mut a_u = Array1::<f64>::zeros(r);
        for u in 0..r {
            a_u[u] = -f_u[u] * inv_f_a;
        }
        let mut a_uv = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let val =
                    -(f_uv[[u, v]] + f_au[u] * a_u[v] + f_au[v] * a_u[u] + f_aa * a_u[u] * a_u[v])
                        * inv_f_a;
                a_uv[[u, v]] = val;
                a_uv[[v, u]] = val;
            }
        }

        let z_obs = self.z[row];
        let u_obs = a + b * z_obs;
        let obs = self.observed_denested_cell_partials(row, a, b, beta_h, beta_w)?;
        let eta_val = eval_coeff4_at(&obs.coeff, z_obs);

        let mut g_u_fixed = vec![[0.0; 4]; r];
        let mut g_au_fixed = vec![[0.0; 4]; r];
        let mut g_bu_fixed = vec![[0.0; 4]; r];
        let mut g_aau_fixed = vec![[0.0; 4]; r];
        let mut g_abu_fixed = vec![[0.0; 4]; r];
        let mut g_bbu_fixed = vec![[0.0; 4]; r];

        g_u_fixed[1] = obs.dc_db;
        g_au_fixed[1] = obs.dc_dab;
        g_bu_fixed[1] = obs.dc_dbb;
        g_aau_fixed[1] = obs.dc_daab;
        g_abu_fixed[1] = obs.dc_dabb;
        g_bbu_fixed[1] = obs.dc_dbbb;

        if let (Some(h_range), Some(runtime)) = (h_range, score_runtime) {
            Self::for_each_deviation_basis_cubic_at(
                runtime,
                h_range,
                z_obs,
                "score-warp batched third observed",
                |_, idx, basis_span| {
                    g_u_fixed[idx] =
                        scale_coeff4(exact::score_basis_cell_coefficients(basis_span, b), scale);
                    g_bu_fixed[idx] =
                        scale_coeff4(exact::score_basis_cell_coefficients(basis_span, 1.0), scale);
                    Ok(())
                },
            )?;
        }

        if let (Some(w_range), Some(runtime)) = (w_range, link_runtime) {
            Self::for_each_deviation_basis_cubic_at(
                runtime,
                w_range,
                u_obs,
                "link-wiggle batched third observed",
                |_, idx, basis_span| {
                    g_u_fixed[idx] =
                        scale_coeff4(exact::link_basis_cell_coefficients(basis_span, a, b), scale);
                    let (dc_aw_raw, dc_bw_raw) =
                        exact::link_basis_cell_coefficient_partials(basis_span, a, b);
                    let (dc_aaw_raw, dc_abw_raw, dc_bbw_raw) =
                        exact::link_basis_cell_second_partials(basis_span, a, b);
                    g_au_fixed[idx] = scale_coeff4(dc_aw_raw, scale);
                    g_bu_fixed[idx] = scale_coeff4(dc_bw_raw, scale);
                    g_aau_fixed[idx] = scale_coeff4(dc_aaw_raw, scale);
                    g_abu_fixed[idx] = scale_coeff4(dc_abw_raw, scale);
                    g_bbu_fixed[idx] = scale_coeff4(dc_bbw_raw, scale);
                    Ok(())
                },
            )?;
        }

        let g_jet = SparsePrimaryCoeffJetView::new(
            1,
            h_range,
            w_range,
            &g_u_fixed,
            &g_au_fixed,
            &g_bu_fixed,
            &g_aau_fixed,
            &g_abu_fixed,
            &g_bbu_fixed,
            &zero_family,
            &zero_family,
            &zero_family,
            &zero_family,
        );

        let g_a = eval_coeff4_at(&obs.dc_da, z_obs);
        let g_aa = eval_coeff4_at(&obs.dc_daa, z_obs);
        let g_aaa = eval_coeff4_at(&obs.dc_daaa, z_obs);
        let mut g_u = Array1::<f64>::zeros(r);
        let mut g_au = Array1::<f64>::zeros(r);
        let mut g_aau = Array1::<f64>::zeros(r);
        let mut g_uv = Array2::<f64>::zeros((r, r));
        let mut g_auv = Array2::<f64>::zeros((r, r));
        for u in 1..r {
            g_u[u] = eval_coeff4_at(&g_jet.first[u], z_obs);
            g_au[u] = eval_coeff4_at(&g_jet.a_first[u], z_obs);
            g_aau[u] = eval_coeff4_at(&g_jet.aa_first[u], z_obs);
        }
        for u in 1..r {
            for v in u..r {
                let second_coeff = g_jet.pair_from_b_family(g_jet.b_first, u, v, COEFF_SUPPORT_BHW);
                let val = eval_coeff4_at(&second_coeff, z_obs);
                g_uv[[u, v]] = val;
                g_uv[[v, u]] = val;

                let third_coeff = g_jet.pair_from_b_family(g_jet.ab_first, u, v, COEFF_SUPPORT_BW);
                let third_val = eval_coeff4_at(&third_coeff, z_obs);
                g_auv[[u, v]] = third_val;
                g_auv[[v, u]] = third_val;
            }
        }

        let eta_u = g_a * &a_u + &g_u;
        let mut eta_uv = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let val = g_a * a_uv[[u, v]]
                    + g_aa * a_u[u] * a_u[v]
                    + g_au[u] * a_u[v]
                    + g_au[v] * a_u[u]
                    + g_uv[[u, v]];
                eta_uv[[u, v]] = val;
                eta_uv[[v, u]] = val;
            }
        }

        let y_i = self.y[row];
        let w_i = self.weights[row];
        let s_y = 2.0 * y_i - 1.0;
        let m = s_y * eta_val;
        let (k1, k2, k3, _) = signed_probit_neglog_derivatives_up_to_fourth(m, w_i)?;
        let u1 = s_y * k1;
        let u3 = s_y * k3;
        let mut out = (0..n_dirs)
            .map(|_| Array2::<f64>::zeros((r, r)))
            .collect::<Vec<_>>();

        for (dir_idx, dir) in row_dirs.iter().enumerate() {
            let dir_base = dir_idx * r * r;
            f_uv_dir[dir_base] = -dir[0] * marginal.mu3;

            let a_dir = a_u.dot(dir);
            let a_u_dir = a_uv.dot(dir);
            let g_dir_fixed = g_jet.directional_family(g_jet.first, dir, COEFF_SUPPORT_BHW);
            let g_a_dir_fixed = g_jet.directional_family(g_jet.a_first, dir, COEFF_SUPPORT_BW);
            let g_aa_dir_fixed = g_jet.directional_family(g_jet.aa_first, dir, COEFF_SUPPORT_BW);
            let g_dir = eval_coeff4_at(&g_dir_fixed, z_obs);
            let g_a_dir = eval_coeff4_at(&g_a_dir_fixed, z_obs);
            let g_aa_dir = eval_coeff4_at(&g_aa_dir_fixed, z_obs);
            let eta_dir = g_a * a_dir + g_dir;
            let eta_u_dir = eta_uv.dot(dir);
            let dg_a_dir = g_aa * a_dir + g_a_dir;
            let dg_aa_dir = g_aaa * a_dir + g_aa_dir;
            let mut dg_au_dir = Array1::<f64>::zeros(r);
            for u in 0..r {
                let coeff =
                    g_jet.param_directional_from_b_family(g_jet.ab_first, u, dir, COEFF_SUPPORT_BW);
                dg_au_dir[u] = g_aau[u] * a_dir + eval_coeff4_at(&coeff, z_obs);
            }

            for u in 0..r {
                for v in u..r {
                    let fuvd = f_uv_dir[dir_base + u * r + v];
                    let n_dir = fuvd
                        + f_au_dir[dir_idx * r + u] * a_u[v]
                        + f_au[u] * a_u_dir[v]
                        + f_au_dir[dir_idx * r + v] * a_u[u]
                        + f_au[v] * a_u_dir[u]
                        + f_aa_dir[dir_idx] * a_u[u] * a_u[v]
                        + f_aa * (a_u_dir[u] * a_u[v] + a_u[u] * a_u_dir[v]);
                    let a_uv_dir = -(n_dir + f_a_dir[dir_idx] * a_uv[[u, v]]) * inv_f_a;
                    let third_coeff = g_jet.pair_directional_from_bb_family(
                        g_jet.bb_first,
                        u,
                        v,
                        dir,
                        COEFF_SUPPORT_BW,
                    );
                    let dg_uv_dir = g_auv[[u, v]] * a_dir + eval_coeff4_at(&third_coeff, z_obs);
                    let eta_uv_dir = dg_a_dir * a_uv[[u, v]]
                        + g_a * a_uv_dir
                        + dg_aa_dir * a_u[u] * a_u[v]
                        + g_aa * (a_u_dir[u] * a_u[v] + a_u[u] * a_u_dir[v])
                        + dg_au_dir[u] * a_u[v]
                        + g_au[u] * a_u_dir[v]
                        + dg_au_dir[v] * a_u[u]
                        + g_au[v] * a_u_dir[u]
                        + dg_uv_dir;
                    let val = u3 * eta_u[u] * eta_u[v] * eta_dir
                        + k2 * (eta_uv[[u, v]] * eta_dir
                            + eta_u[u] * eta_u_dir[v]
                            + eta_u[v] * eta_u_dir[u])
                        + u1 * eta_uv_dir;
                    out[dir_idx][[u, v]] = val;
                    out[dir_idx][[v, u]] = val;
                }
            }
        }

        Ok(out)
    }

    pub(super) fn batched_rho_correction_logdet_traces_for_rows(
        &self,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        factor: &Array2<f64>,
        directions: &Array2<f64>,
        weighted_rows: &[WeightedOuterRow],
    ) -> Result<Array1<f64>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let rank = factor.ncols();
        let n_dirs = directions.ncols();
        if factor.nrows() != slices.total || directions.nrows() != slices.total {
            return Err(format!(
                "bernoulli marginal-slope batched trace shape mismatch: factor={}x{}, directions={}x{}, p={}",
                factor.nrows(),
                factor.ncols(),
                directions.nrows(),
                directions.ncols(),
                slices.total
            ));
        }
        let started = std::time::Instant::now();
        if log_exact_work(self.y.len()) {
            log::info!(
                "[BMS rho-correction-trace] sampled start n={} rows={} p={} rank={} dirs={}",
                self.y.len(),
                weighted_rows.len(),
                slices.total,
                rank,
                n_dirs
            );
        }
        let traces = weighted_rows
            .par_iter()
            .try_fold(
                || vec![0.0; n_dirs],
                |mut acc, wr| -> Result<_, String> {
                    let row = wr.index;
                    let row_ctx = Self::row_ctx(cache, row);
                    let mut projection = vec![0.0; primary.total * rank];
                    self.row_factor_primary_projection(
                        row,
                        slices,
                        primary,
                        factor,
                        &mut projection,
                    )?;
                    let gram =
                        Self::row_primary_gram_from_projection(primary.total, rank, &projection);
                    if n_dirs == 1 {
                        let d_beta = directions.column(0).to_owned();
                        let row_dir =
                            self.row_primary_direction_from_flat(row, slices, primary, &d_beta)?;
                        let row_traces = self.row_primary_third_trace_many_with_moments(
                            row,
                            block_states,
                            cache,
                            row_ctx,
                            &[row_dir],
                            &gram,
                        )?;
                        acc[0] += wr.weight * row_traces[0];
                        return Ok(acc);
                    }
                    let trace_gradient = self.row_primary_third_trace_gradient_with_moments(
                        row,
                        block_states,
                        cache,
                        row_ctx,
                        &gram,
                    )?;
                    for dir_idx in 0..n_dirs {
                        let mut trace = trace_gradient[primary.q]
                            * self.marginal_design.dot_row_view(
                                row,
                                directions.slice(s![slices.marginal.clone(), dir_idx]),
                            );
                        trace += trace_gradient[primary.logslope]
                            * self.logslope_design.dot_row_view(
                                row,
                                directions.slice(s![slices.logslope.clone(), dir_idx]),
                            );
                        if let (Some(block_range), Some(primary_range)) =
                            (slices.h.as_ref(), primary.h.as_ref())
                        {
                            for (offset, block_idx) in block_range.clone().enumerate() {
                                trace += trace_gradient[primary_range.start + offset]
                                    * directions[[block_idx, dir_idx]];
                            }
                        }
                        if let (Some(block_range), Some(primary_range)) =
                            (slices.w.as_ref(), primary.w.as_ref())
                        {
                            for (offset, block_idx) in block_range.clone().enumerate() {
                                trace += trace_gradient[primary_range.start + offset]
                                    * directions[[block_idx, dir_idx]];
                            }
                        }
                        acc[dir_idx] += wr.weight * trace;
                    }
                    Ok(acc)
                },
            )
            .try_reduce(
                || vec![0.0; n_dirs],
                |mut left, right| -> Result<_, String> {
                    for (l, r) in left.iter_mut().zip(right.iter()) {
                        *l += *r;
                    }
                    Ok(left)
                },
            )?;
        if log_exact_work(self.y.len()) {
            log::info!(
                "[BMS rho-correction-trace] sampled done n={} rows={} p={} rank={} dirs={} elapsed={:.3}s",
                self.y.len(),
                weighted_rows.len(),
                slices.total,
                rank,
                n_dirs,
                started.elapsed().as_secs_f64()
            );
        }
        Ok(Array1::from_vec(traces))
    }

    pub(super) fn batched_rho_correction_logdet_traces_full_rows(
        &self,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        factor: &Array2<f64>,
        directions: &Array2<f64>,
    ) -> Result<Array1<f64>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let n = self.y.len();
        let rank = factor.ncols();
        let n_dirs = directions.ncols();
        if n == 0 || rank == 0 || n_dirs == 0 {
            return Ok(Array1::zeros(n_dirs));
        }
        let rows_per_chunk = {
            const TARGET_BYTES: usize = 8 * 1024 * 1024;
            let panels = 4usize;
            let width = rank + n_dirs;
            (TARGET_BYTES / (panels * width.max(1) * 8)).max(512).min(n)
        };
        let factor_m = factor.slice(s![slices.marginal.clone(), ..]);
        let factor_g = factor.slice(s![slices.logslope.clone(), ..]);
        let dir_m = directions.slice(s![slices.marginal.clone(), ..]);
        let dir_g = directions.slice(s![slices.logslope.clone(), ..]);
        let tail_pairs = Self::primary_tail_block_pairs(slices, primary);
        let tail_tail_gram = Self::primary_tail_tail_gram(primary.total, rank, factor, &tail_pairs);
        let n_chunks = n.div_ceil(rows_per_chunk);
        let started = std::time::Instant::now();
        let completed_chunks = AtomicUsize::new(0);
        let progress_step = (n_chunks / 10).max(1);
        if log_exact_work(n) {
            log::info!(
                "[BMS rho-correction-trace] full start n={} chunks={} rows_per_chunk={} p={} rank={} dirs={}",
                n,
                n_chunks,
                rows_per_chunk,
                slices.total,
                rank,
                n_dirs
            );
        }
        let traces = (0..n_chunks)
            .into_par_iter()
            .map(|chunk_idx| -> Result<Vec<f64>, String> {
                let start = chunk_idx * rows_per_chunk;
                let end = (start + rows_per_chunk).min(n);
                let rows = start..end;
                let x_chunk = self
                    .marginal_design
                    .try_row_chunk(rows.clone())
                    .map_err(|err| format!("marginal trace row chunk failed: {err}"))?;
                let g_chunk = self
                    .logslope_design
                    .try_row_chunk(rows.clone())
                    .map_err(|err| format!("logslope trace row chunk failed: {err}"))?;
                let proj_m = crate::faer_ndarray::fast_ab(&x_chunk, &factor_m);
                let proj_g = crate::faer_ndarray::fast_ab(&g_chunk, &factor_g);
                let dir_proj_m = crate::faer_ndarray::fast_ab(&x_chunk, &dir_m);
                let dir_proj_g = crate::faer_ndarray::fast_ab(&g_chunk, &dir_g);
                let mut acc = vec![0.0; n_dirs];
                let mut gram = vec![0.0; primary.total * primary.total];
                let mut row_dir = Array1::<f64>::zeros(primary.total);
                for local in 0..(end - start) {
                    let row = start + local;
                    gram.copy_from_slice(&tail_tail_gram);
                    let mut qq = 0.0;
                    let mut qg = 0.0;
                    let mut gg = 0.0;
                    for col in 0..rank {
                        let qv = proj_m[[local, col]];
                        let gv = proj_g[[local, col]];
                        qq += qv * qv;
                        qg += qv * gv;
                        gg += gv * gv;
                    }
                    gram[primary.q * primary.total + primary.q] = qq;
                    gram[primary.q * primary.total + primary.logslope] = qg;
                    gram[primary.logslope * primary.total + primary.q] = qg;
                    gram[primary.logslope * primary.total + primary.logslope] = gg;
                    for &(primary_idx, block_idx) in &tail_pairs {
                        let mut q_tail = 0.0;
                        let mut g_tail = 0.0;
                        for col in 0..rank {
                            let tail = factor[[block_idx, col]];
                            q_tail += proj_m[[local, col]] * tail;
                            g_tail += proj_g[[local, col]] * tail;
                        }
                        gram[primary.q * primary.total + primary_idx] = q_tail;
                        gram[primary_idx * primary.total + primary.q] = q_tail;
                        gram[primary.logslope * primary.total + primary_idx] = g_tail;
                        gram[primary_idx * primary.total + primary.logslope] = g_tail;
                    }
                    let row_ctx = Self::row_ctx(cache, row);
                    if n_dirs == 1 {
                        row_dir.fill(0.0);
                        row_dir[primary.q] = dir_proj_m[[local, 0]];
                        row_dir[primary.logslope] = dir_proj_g[[local, 0]];
                        if let (Some(block_range), Some(primary_range)) =
                            (slices.h.as_ref(), primary.h.as_ref())
                        {
                            for (offset, block_idx) in block_range.clone().enumerate() {
                                row_dir[primary_range.start + offset] = directions[[block_idx, 0]];
                            }
                        }
                        if let (Some(block_range), Some(primary_range)) =
                            (slices.w.as_ref(), primary.w.as_ref())
                        {
                            for (offset, block_idx) in block_range.clone().enumerate() {
                                row_dir[primary_range.start + offset] = directions[[block_idx, 0]];
                            }
                        }
                        let row_traces = self.row_primary_third_trace_many_with_moments(
                            row,
                            block_states,
                            cache,
                            row_ctx,
                            std::slice::from_ref(&row_dir),
                            &gram,
                        )?;
                        acc[0] += row_traces[0];
                        continue;
                    }
                    let trace_gradient = self.row_primary_third_trace_gradient_with_moments(
                        row,
                        block_states,
                        cache,
                        row_ctx,
                        &gram,
                    )?;
                    for dir_idx in 0..n_dirs {
                        let mut trace = trace_gradient[primary.q] * dir_proj_m[[local, dir_idx]]
                            + trace_gradient[primary.logslope] * dir_proj_g[[local, dir_idx]];
                        if let (Some(block_range), Some(primary_range)) =
                            (slices.h.as_ref(), primary.h.as_ref())
                        {
                            for (offset, block_idx) in block_range.clone().enumerate() {
                                trace += trace_gradient[primary_range.start + offset]
                                    * directions[[block_idx, dir_idx]];
                            }
                        }
                        if let (Some(block_range), Some(primary_range)) =
                            (slices.w.as_ref(), primary.w.as_ref())
                        {
                            for (offset, block_idx) in block_range.clone().enumerate() {
                                trace += trace_gradient[primary_range.start + offset]
                                    * directions[[block_idx, dir_idx]];
                            }
                        }
                        acc[dir_idx] += trace;
                    }
                }
                if log_exact_work(n) {
                    let done = completed_chunks.fetch_add(1, Ordering::Relaxed) + 1;
                    if done == n_chunks || done % progress_step == 0 {
                        log::info!(
                            "[BMS rho-correction-trace] full progress chunks={}/{} rows={}/{} elapsed={:.3}s",
                            done,
                            n_chunks,
                            (done * rows_per_chunk).min(n),
                            n,
                            started.elapsed().as_secs_f64()
                        );
                    }
                }
                Ok(acc)
            })
            .try_reduce(
                || vec![0.0; n_dirs],
                |mut left, right| -> Result<_, String> {
                    for (l, r) in left.iter_mut().zip(right.iter()) {
                        *l += *r;
                    }
                    Ok(left)
                },
            )?;
        if log_exact_work(n) {
            log::info!(
                "[BMS rho-correction-trace] full done n={} chunks={} p={} rank={} dirs={} elapsed={:.3}s",
                n,
                n_chunks,
                slices.total,
                rank,
                n_dirs,
                started.elapsed().as_secs_f64()
            );
        }
        Ok(Array1::from_vec(traces))
    }
}

impl BernoulliMarginalSlopeExactNewtonJointPsiWorkspace {
    pub(super) fn new(
        family: BernoulliMarginalSlopeFamily,
        block_states: Vec<ParameterBlockState>,
        specs: Vec<ParameterBlockSpec>,
        derivative_blocks: Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>>,
        options: BlockwiseFitOptions,
    ) -> Result<Self, String> {
        let cache = family.build_exact_eval_cache_with_options(&block_states, Some(&options))?;
        // Prime the per-row uncontracted third-derivative tensor at workspace
        // construction (rigid path only). The build runs at top-level rayon
        // here, so the parallel row pass uses all cores. If we instead let
        // it run lazily inside `build_psi_hyper_coords` axis calls, those
        // calls are themselves at top level — so leaving lazy would also be
        // parallel — but priming here lifts the first-axis cost out of the
        // workspace's `first_order_terms` measurement.
        if !family.effective_flex_active(&block_states)? {
            let warmed_third = family.rigid_third_full_cached(&block_states, &cache, 0)?;
            ensure_finite_third_full_cache_row(
                warmed_third,
                "BernoulliMarginalSlopeExactNewtonJointPsiWorkspace third-cache warm-up",
            )?;
            // Outer-Hessian path consumes per-row fourth-tensor over every
            // (ψ-axis-i, ψ-axis-j) pair — prime here too so the 528-pair
            // sweep reads a populated cache instead of triggering the
            // 8-direction empirical jet on its first per-pair call.
            let warmed_fourth = family.rigid_fourth_full_cached(&block_states, &cache, 0)?;
            ensure_finite_fourth_full_cache_row(
                warmed_fourth,
                "BernoulliMarginalSlopeExactNewtonJointPsiWorkspace fourth-cache warm-up",
            )?;
        }
        Ok(Self {
            family,
            block_states,
            specs,
            derivative_blocks,
            cache: std::sync::Arc::new(cache),
            options,
        })
    }
}

impl ExactNewtonJointPsiWorkspace for BernoulliMarginalSlopeExactNewtonJointPsiWorkspace {
    fn first_order_terms(
        &self,
        psi_index: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        if self
            .family
            .is_sigma_aux_index(&self.derivative_blocks, psi_index)
        {
            return self.family.sigma_exact_joint_psi_terms_with_options(
                &self.block_states,
                &self.specs,
                &self.options,
            );
        }
        self.family
            .exact_newton_joint_psi_terms_from_cache_with_options(
                &self.block_states,
                &self.derivative_blocks,
                psi_index,
                &self.cache,
                &self.options,
            )
    }

    fn first_order_terms_all(&self) -> Result<Option<Vec<ExactNewtonJointPsiTerms>>, String> {
        let total: usize = self.derivative_blocks.iter().map(Vec::len).sum();
        if total == 0 {
            return Ok(Some(Vec::new()));
        }
        for psi_index in 0..total {
            if self
                .family
                .is_sigma_aux_index(&self.derivative_blocks, psi_index)
            {
                return Ok(None);
            }
        }
        let mut axes: Vec<PsiAxisSpec> = Vec::with_capacity(total);
        for psi_index in 0..total {
            let Some((block_idx, local_idx)) =
                psi_derivative_location(&self.derivative_blocks, psi_index)
            else {
                return Ok(None);
            };
            axes.push(self.family.resolve_psi_axis_spec(
                &self.derivative_blocks,
                block_idx,
                local_idx,
            )?);
        }
        let results = self.family.run_psi_row_pass_for_axes(
            &self.block_states,
            &self.cache,
            &self.options,
            &axes,
        )?;
        Ok(Some(results))
    }

    fn second_order_terms(
        &self,
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        if self
            .family
            .is_sigma_aux_index(&self.derivative_blocks, psi_i)
            || self
                .family
                .is_sigma_aux_index(&self.derivative_blocks, psi_j)
        {
            if self
                .family
                .is_sigma_aux_index(&self.derivative_blocks, psi_i)
                && self
                    .family
                    .is_sigma_aux_index(&self.derivative_blocks, psi_j)
            {
                return self
                    .family
                    .sigma_exact_joint_psisecond_order_terms_with_options(
                        &self.block_states,
                        &self.options,
                    );
            }
            return Err("bernoulli marginal-slope mixed log-sigma/spatial psi second derivatives require cross auxiliary terms; only pure log-sigma second derivatives are supported"
                        .to_string());
        }
        self.family
            .exact_newton_joint_psisecond_order_terms_from_cache_with_options(
                &self.block_states,
                &self.derivative_blocks,
                psi_i,
                psi_j,
                &self.cache,
                &self.options,
            )
    }

    fn hessian_directional_derivative(
        &self,
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<crate::solver::estimate::reml::unified::DriftDerivResult>, String> {
        if self
            .family
            .is_sigma_aux_index(&self.derivative_blocks, psi_index)
        {
            return self
                .family
                .sigma_exact_joint_psihessian_directional_derivative_with_options(
                    &self.block_states,
                    d_beta_flat,
                    &self.options,
                )
                .map(|result| {
                    result.map(crate::solver::estimate::reml::unified::DriftDerivResult::Dense)
                });
        }
        self.family
            .exact_newton_joint_psihessian_directional_derivative_operator_from_cache_with_options(
                &self.block_states,
                &self.derivative_blocks,
                psi_index,
                d_beta_flat,
                &self.cache,
                &self.options,
            )
            .map(|result| {
                result.map(crate::solver::estimate::reml::unified::DriftDerivResult::Operator)
            })
    }
}
