//! Second-order terms of a latent family's baseline-chart axis pairs and their
//! coefficient motion.
//!
//! The fixed-β part (#2677) is the explicit `(V_ij, g_ij, H_ij)` an exact outer
//! Hessian over the chart coordinates reads through
//! `exact_newton_joint_psisecond_order_terms`. An armed Jeffreys term differentiates
//! that information once more in β (#4510), so the same chart algebra also serves
//! `{D_β ∂²_{ψ_iψ_j}H[e_a]}` and the completion's `{∂_ψ D²_βH[v, e_a]}`, one seed
//! deeper in the same row lifts.

use super::*;

impl LatentSurvivalFamily {
    /// The row's second offset direction `∂²q_i/∂θ_a∂θ_b` of loaded chart axes `a`
    /// and `b` (#2677).
    fn baseline_theta_row_second_direction(
        rows: &crate::survival::construction::LatentSurvivalOffsetGeometry,
        row: usize,
        axis_a: usize,
        axis_b: usize,
    ) -> Array1<f64> {
        let mut direction = Array1::<f64>::zeros(LATENT_SURVIVAL_PRIMARY_DIM);
        direction[LATENT_SURVIVAL_PRIMARY_Q_ENTRY] =
            rows.offset_entry_theta_theta[[row, axis_a, axis_b]];
        direction[LATENT_SURVIVAL_PRIMARY_Q_EXIT] =
            rows.offset_exit_theta_theta[[row, axis_a, axis_b]];
        direction[LATENT_SURVIVAL_PRIMARY_QDOT_EXIT] =
            rows.derivative_offset_exit_theta_theta[[row, axis_a, axis_b]];
        direction[LATENT_SURVIVAL_PRIMARY_Q_RIGHT] =
            rows.offset_right_theta_theta[[row, axis_a, axis_b]];
        direction
    }

    /// Fixed-β second-order terms of baseline-chart axes `axis_i` and `axis_j`
    /// (#2677). The chart moves only the offsets, so with `d_a = ∂q_i/∂θ_a` and
    /// `d_ab = ∂²q_i/∂θ_a∂θ_b`:
    ///
    /// ```text
    ///   V_ab = −Σ_i w_i (∇ℓ_i·d_ab + ∇²ℓ_i[d_a, d_b]),
    ///   g_ab = Σ_i w_i X_iᵀ (−∇³ℓ_i[d_a, d_b] − ∇²ℓ_i d_ab),
    ///   H_ab = Σ_i w_i X_iᵀ (−∇⁴ℓ_i[d_a, d_b] − ∇³ℓ_i[d_ab]) X_i.
    /// ```
    ///
    /// The background-scale axis `ln m` moves no offset. Its pair with a loaded
    /// axis `a` reads `φ_i = ∂ℓ_i/∂ln m` along `d_a`: `V = −Σ w ∇φ·d_a`,
    /// `g = −Σ w Xᵀ ∇²φ d_a`, `H = −Σ w Xᵀ ∇³φ[d_a] X`. The pair `(ln m, ln m)`
    /// reads `∂²ℓ_i/∂(ln m)²` ([`latent_survival_row_background_scale_jet`]).
    pub(super) fn baseline_theta_psisecond_order_terms_dense(
        &self,
        block_states: &[ParameterBlockState],
        rows: &crate::survival::construction::LatentSurvivalOffsetGeometry,
        axis_i: usize,
        axis_j: usize,
    ) -> Result<gam_problem::ExactNewtonJointPsiSecondOrderTerms, String> {
        struct BaselinePsiPairAccum {
            objective: CompensatedRowSum,
            score: Array1<f64>,
            hessian: Array2<f64>,
        }
        let weights = ValidatedLikelihoodWeights::new(&self.weights, "latent-survival")
            .map_err(String::from)?;
        let (q_entry, q_exit, qdot_exit, mu) = self.split_time_eta(block_states)?;
        let q_right = self.time_q_right(block_states)?;
        let sigma = self.latent_sd(block_states)?;
        let slices = self.joint_slices();
        let include_log_sigma = slices.log_sigma.is_some();
        let total = slices.total;
        let n = self.event_target.len();
        let dim = rows.theta.len();
        if axis_i >= dim
            || axis_j >= dim
            || rows.offset_exit_theta.nrows() != n
            || rows.offset_exit_theta_theta.dim() != (n, dim, dim)
        {
            return Err(format!(
                "latent survival baseline psi pair ({axis_i}, {axis_j}) is outside a {dim}-coordinate chart realized over {} rows with second partials {:?} for {n} observations",
                rows.offset_exit_theta.nrows(),
                rows.offset_exit_theta_theta.dim(),
            ));
        }
        let unloaded_axis = rows.unloaded.as_ref().map(|unloaded| unloaded.axis);
        let acc = deterministic_latent_survival_row_reduction(
            n,
            || BaselinePsiPairAccum {
                objective: CompensatedRowSum::default(),
                score: Array1::<f64>::zeros(total),
                hessian: Array2::<f64>::zeros((total, total)),
            },
            |row_idx, acc| {
                let wi = weights.at(row_idx);
                if wi == 0.0 {
                    return Ok(());
                }
                let row = self.build_row_at(
                    row_idx,
                    q_entry[row_idx],
                    q_exit[row_idx],
                    qdot_exit[row_idx],
                    q_right[row_idx],
                )?;
                let point = || LatentSurvivalPrimaryPoint {
                    q_entry: q_entry[row_idx],
                    q_exit: q_exit[row_idx],
                    qdot_exit: qdot_exit[row_idx],
                    q_right: q_right[row_idx],
                    mu: mu[row_idx],
                    sigma,
                };
                let (value, gradient, information) =
                    match (Some(axis_i) == unloaded_axis, Some(axis_j) == unloaded_axis) {
                        (true, true) => {
                            let (value, gradient, hessian) =
                                latent_survival_row_unloaded_scale_second_channels(
                                    &self.quadctx,
                                    &row,
                                    point(),
                                    include_log_sigma,
                                )?;
                            (-value, -&gradient, -&hessian)
                        }
                        (true, false) | (false, true) => {
                            let loaded = if Some(axis_i) == unloaded_axis {
                                axis_j
                            } else {
                                axis_i
                            };
                            let direction = Self::baseline_theta_row_direction(rows, row_idx, loaded);
                            let (_, gradient, hessian) = latent_survival_row_unloaded_scale_channels(
                                &self.quadctx,
                                &row,
                                point(),
                                include_log_sigma,
                            )?;
                            let third = latent_survival_row_unloaded_scale_third(
                                &self.quadctx,
                                &row,
                                point(),
                                &direction,
                                include_log_sigma,
                            )?;
                            (-gradient.dot(&direction), -hessian.dot(&direction), -&third)
                        }
                        (false, false) => {
                            let direction_i = Self::baseline_theta_row_direction(rows, row_idx, axis_i);
                            let direction_j = Self::baseline_theta_row_direction(rows, row_idx, axis_j);
                            let direction_ij =
                                Self::baseline_theta_row_second_direction(rows, row_idx, axis_i, axis_j);
                            let (gradient, neg_hessian, neg_third_i) =
                                latent_survival_row_primary_one_seed_channels(
                                    &self.quadctx,
                                    &row,
                                    point(),
                                    &direction_i,
                                    include_log_sigma,
                                )?;
                            let (_, _, neg_third_ij) = latent_survival_row_primary_one_seed_channels(
                                &self.quadctx,
                                &row,
                                point(),
                                &direction_ij,
                                include_log_sigma,
                            )?;
                            let neg_fourth = latent_survival_row_primary_fourth_contracted(
                                &self.quadctx,
                                &row,
                                point(),
                                &direction_i,
                                &direction_j,
                                include_log_sigma,
                            )?;
                            (
                                -gradient.dot(&direction_ij)
                                    + direction_i.dot(&neg_hessian.dot(&direction_j)),
                                neg_third_i.dot(&direction_j) + neg_hessian.dot(&direction_ij),
                                neg_fourth + neg_third_ij,
                            )
                        }
                    };
                acc.objective.add(checked_weighted_row_value(
                    wi,
                    value,
                    row_idx,
                    "baseline psi pair objective",
                )?);
                self.add_pullback_primary_gradient(&mut acc.score, row_idx, &slices, &gradient, wi)?;
                let weighted_information = checked_weighted_row_matrix(
                    wi,
                    &information,
                    row_idx,
                    "baseline psi pair information",
                )?;
                self.add_pullback_primary_hessian(
                    &mut acc.hessian,
                    row_idx,
                    &slices,
                    &weighted_information,
                )?;
                Ok(())
            },
            |total_acc, chunk_acc| {
                total_acc.objective.add(chunk_acc.objective.value());
                total_acc.score += &chunk_acc.score;
                total_acc.hessian += &chunk_acc.hessian;
            },
        )?;
        let objective_psi_psi =
            require_finite_likelihood_scalar(acc.objective.value(), "baseline psi pair objective")?;
        require_finite_likelihood_vector(&acc.score, "baseline psi pair score")?;
        require_finite_likelihood_matrix(&acc.hessian, "baseline psi pair information derivative")?;
        Ok(gam_problem::ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi: acc.score,
            hessian_psi_psi: acc.hessian,
            hessian_psi_psi_operator: None,
        })
    }

    /// `{D_β ∂²_{ψ_iψ_j}H[e_a]}` over every coefficient axis (#4510): the
    /// coefficient motion of [`Self::baseline_theta_psisecond_order_terms_dense`]'s
    /// information derivative, which an armed Jeffreys curvature reads for every
    /// active chart pair. Differentiating that pair's row kernel once more in β
    /// adds one seed and nothing else, because the chart moves only the offsets:
    ///
    /// ```text
    ///   D_βH_ab[e_γ]_i = X_iᵀ (−∇⁵ℓ_i[d_a, d_b, e_γ] − ∇⁴ℓ_i[d_ab, e_γ]) X_i.
    /// ```
    ///
    /// The background-scale axis `ln m` moves no offset and is read through
    /// `φ_i = ∂ℓ_i/∂ln m`, so a pair `(ln m, a)` reads `−∇⁴φ_i[d_a, e_γ]` and the
    /// pair `(ln m, ln m)` reads `−∇³(∂φ_i/∂ln m)[e_γ]`. Each kernel is linear in
    /// its last seed, so the `p` axes close from one lift per touched primary, as
    /// the first derivative's do.
    pub(super) fn baseline_theta_psisecond_order_hessian_axes_dense(
        &self,
        block_states: &[ParameterBlockState],
        rows: &crate::survival::construction::LatentSurvivalOffsetGeometry,
        axis_i: usize,
        axis_j: usize,
    ) -> Result<Vec<Array2<f64>>, String> {
        let slices = self.joint_slices();
        let include_log_sigma = slices.log_sigma.is_some();
        let n = self.event_target.len();
        let dim = rows.theta.len();
        if axis_i >= dim
            || axis_j >= dim
            || rows.offset_exit_theta.nrows() != n
            || rows.offset_exit_theta_theta.dim() != (n, dim, dim)
        {
            return Err(format!(
                "latent survival baseline psi pair ({axis_i}, {axis_j}) coefficient motion is outside a {dim}-coordinate chart realized over {} rows with second partials {:?} for {n} observations",
                rows.offset_exit_theta.nrows(),
                rows.offset_exit_theta_theta.dim(),
            ));
        }
        let unloaded_axis = rows.unloaded.as_ref().map(|unloaded| unloaded.axis);
        self.joint_hessian_axes_from_primary_lifts(
            block_states,
            "baseline psi pair coefficient motion",
            |lift| match (Some(axis_i) == unloaded_axis, Some(axis_j) == unloaded_axis) {
                (true, true) => {
                    let third = latent_survival_row_unloaded_scale_second_third(
                        &self.quadctx,
                        lift.row,
                        lift.point,
                        lift.primary,
                        include_log_sigma,
                    )
                    .map_err(String::from)?;
                    Ok(-third)
                }
                (true, false) | (false, true) => {
                    let loaded = if Some(axis_i) == unloaded_axis {
                        axis_j
                    } else {
                        axis_i
                    };
                    let direction = Self::baseline_theta_row_direction(rows, lift.row_idx, loaded);
                    let fourth = latent_survival_row_unloaded_scale_fourth(
                        &self.quadctx,
                        lift.row,
                        lift.point,
                        &direction,
                        lift.primary,
                        include_log_sigma,
                    )
                    .map_err(String::from)?;
                    Ok(-fourth)
                }
                (false, false) => {
                    let direction_i =
                        Self::baseline_theta_row_direction(rows, lift.row_idx, axis_i);
                    let direction_j =
                        Self::baseline_theta_row_direction(rows, lift.row_idx, axis_j);
                    let direction_ij = Self::baseline_theta_row_second_direction(
                        rows,
                        lift.row_idx,
                        axis_i,
                        axis_j,
                    );
                    let fifth = latent_survival_row_primary_fifth_contracted(
                        &self.quadctx,
                        lift.row,
                        lift.point,
                        &direction_i,
                        &direction_j,
                        lift.primary,
                        include_log_sigma,
                    )
                    .map_err(String::from)?;
                    let fourth = latent_survival_row_primary_fourth_contracted(
                        &self.quadctx,
                        lift.row,
                        lift.point,
                        &direction_ij,
                        lift.primary,
                        include_log_sigma,
                    )
                    .map_err(String::from)?;
                    Ok(fifth + fourth)
                }
            },
        )
    }

    /// `{∂_ψ D²_βH[v, e_a]}` over every coefficient axis (#4510): the mixed third
    /// information derivative the ψ-moving Jeffreys completion reads. The chart
    /// moves only the offsets, so a loaded axis enters as one more seed of the same
    /// row lift and the background-scale axis enters through `φ_i = ∂ℓ_i/∂ln m`:
    ///
    /// ```text
    ///   loaded θ:  ∂_θ D²_βH[v, e_γ]_i = X_iᵀ (−∇⁵ℓ_i[d_θ, X_i v, e_γ]) X_i,
    ///   ln m:      ∂_{ln m} D²_βH[v, e_γ]_i = X_iᵀ (−∇⁴φ_i[X_i v, e_γ]) X_i.
    /// ```
    pub(super) fn baseline_theta_psihessian_second_directional_axes_dense(
        &self,
        block_states: &[ParameterBlockState],
        rows: &crate::survival::construction::LatentSurvivalOffsetGeometry,
        axis: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Vec<Array2<f64>>, String> {
        let slices = self.joint_slices();
        let include_log_sigma = slices.log_sigma.is_some();
        if d_beta_flat.len() != slices.total || axis >= rows.theta.len() {
            return Err(format!(
                "latent survival baseline psi second coefficient motion: direction length {} against {} coefficients, axis {axis} of {}",
                d_beta_flat.len(),
                slices.total,
                rows.theta.len()
            ));
        }
        let unloaded_axis = rows.unloaded.as_ref().map(|unloaded| unloaded.axis);
        self.joint_hessian_axes_from_primary_lifts(
            block_states,
            "baseline psi second coefficient motion",
            |lift| {
                let direction_beta =
                    self.row_primary_direction_from_flat(lift.row_idx, &slices, d_beta_flat);
                if Some(axis) == unloaded_axis {
                    let fourth = latent_survival_row_unloaded_scale_fourth(
                        &self.quadctx,
                        lift.row,
                        lift.point,
                        &direction_beta,
                        lift.primary,
                        include_log_sigma,
                    )
                    .map_err(String::from)?;
                    return Ok(-fourth);
                }
                let direction_theta = Self::baseline_theta_row_direction(rows, lift.row_idx, axis);
                latent_survival_row_primary_fifth_contracted(
                    &self.quadctx,
                    lift.row,
                    lift.point,
                    &direction_theta,
                    &direction_beta,
                    lift.primary,
                    include_log_sigma,
                )
                .map_err(String::from)
            },
        )
    }
}

impl LatentBinaryFamily {
    /// The latent-binary row's second offset direction `∂²q_i/∂θ_a∂θ_b` of loaded
    /// chart axes `a` and `b`; only the entry and exit offsets move (#2677).
    fn baseline_theta_row_second_direction(
        rows: &crate::survival::construction::LatentSurvivalOffsetGeometry,
        row: usize,
        axis_a: usize,
        axis_b: usize,
    ) -> Array1<f64> {
        let mut direction = Array1::<f64>::zeros(LATENT_SURVIVAL_PRIMARY_DIM);
        direction[LATENT_SURVIVAL_PRIMARY_Q_ENTRY] =
            rows.offset_entry_theta_theta[[row, axis_a, axis_b]];
        direction[LATENT_SURVIVAL_PRIMARY_Q_EXIT] =
            rows.offset_exit_theta_theta[[row, axis_a, axis_b]];
        direction
    }

    /// Fixed-β second-order terms of baseline-chart axes `axis_i` and `axis_j` for
    /// the binary deployment (#2677), the binary twin of
    /// [`LatentSurvivalFamily::baseline_theta_psisecond_order_terms_dense`]:
    /// `V_ab = −Σ w (∇ℓ_bin·d_ab + ∇²ℓ_bin[d_a, d_b])`,
    /// `g_ab = Σ w Xᵀ (D_{d_a}(−∇²ℓ_bin) d_b + (−∇²ℓ_bin) d_ab)`,
    /// `H_ab = Σ w Xᵀ (D_{d_a} D_{d_b}(−∇²ℓ_bin) + D_{d_ab}(−∇²ℓ_bin)) X`, with the
    /// background-scale axis `ln m` read through `φ = ∂ℓ_bin/∂ln m`.
    pub(super) fn baseline_theta_psisecond_order_terms_dense(
        &self,
        block_states: &[ParameterBlockState],
        rows: &crate::survival::construction::LatentSurvivalOffsetGeometry,
        axis_i: usize,
        axis_j: usize,
    ) -> Result<gam_problem::ExactNewtonJointPsiSecondOrderTerms, String> {
        let weights = ValidatedLikelihoodWeights::new(&self.weights, "latent-binary")
            .map_err(String::from)?;
        let (q_entry, q_exit, mu) = self.split_time_eta(block_states)?;
        let slices = self.joint_slices();
        let total = slices.total;
        let n = self.event_target.len();
        let dim = rows.theta.len();
        if axis_i >= dim
            || axis_j >= dim
            || rows.offset_exit_theta.nrows() != n
            || rows.offset_exit_theta_theta.dim() != (n, dim, dim)
        {
            return Err(format!(
                "latent binary baseline psi pair ({axis_i}, {axis_j}) is outside a {dim}-coordinate chart realized over {} rows with second partials {:?} for {n} observations",
                rows.offset_exit_theta.nrows(),
                rows.offset_exit_theta_theta.dim(),
            ));
        }
        let unloaded_axis = rows.unloaded.as_ref().map(|unloaded| unloaded.axis);
        let mut objective = CompensatedRowSum::default();
        let mut score = Array1::<f64>::zeros(total);
        let mut hessian = Array2::<f64>::zeros((total, total));
        for row_idx in 0..n {
            let wi = weights.at(row_idx);
            if wi == 0.0 {
                continue;
            }
            let row =
                self.build_right_censored_row_at(row_idx, q_entry[row_idx], q_exit[row_idx])?;
            let event = self.event_target[row_idx];
            let point = || LatentSurvivalPrimaryPoint {
                q_entry: q_entry[row_idx],
                q_exit: q_exit[row_idx],
                qdot_exit: 1.0,
                q_right: q_exit[row_idx],
                mu: mu[row_idx],
                sigma: self.latent_sd,
            };
            let (value, gradient, information) =
                match (Some(axis_i) == unloaded_axis, Some(axis_j) == unloaded_axis) {
                    (true, true) => {
                        let (value, gradient, curvature) =
                            latent_binary_row_unloaded_scale_second_channels(
                                &self.quadctx,
                                &row,
                                point(),
                                event,
                            )?;
                        (-value, -&gradient, -&curvature)
                    }
                    (true, false) | (false, true) => {
                        let loaded = if Some(axis_i) == unloaded_axis {
                            axis_j
                        } else {
                            axis_i
                        };
                        let direction = Self::baseline_theta_row_direction(rows, row_idx, loaded);
                        let (_, gradient, curvature) = latent_binary_row_unloaded_scale_channels(
                            &self.quadctx,
                            &row,
                            point(),
                            event,
                        )?;
                        let third = latent_binary_row_unloaded_scale_third(
                            &self.quadctx,
                            &row,
                            point(),
                            event,
                            &direction,
                        )?;
                        (-gradient.dot(&direction), -curvature.dot(&direction), -&third)
                    }
                    (false, false) => {
                        let direction_i = Self::baseline_theta_row_direction(rows, row_idx, axis_i);
                        let direction_j = Self::baseline_theta_row_direction(rows, row_idx, axis_j);
                        let direction_ij =
                            Self::baseline_theta_row_second_direction(rows, row_idx, axis_i, axis_j);
                        let (gradient, neg_hessian, neg_third_i) = latent_binary_row_one_seed_channels(
                            &self.quadctx,
                            &row,
                            point(),
                            event,
                            &direction_i,
                        )?;
                        let (_, _, neg_third_ij) = latent_binary_row_one_seed_channels(
                            &self.quadctx,
                            &row,
                            point(),
                            event,
                            &direction_ij,
                        )?;
                        let neg_fourth = latent_binary_row_contracted_fourth(
                            &self.quadctx,
                            &row,
                            point(),
                            event,
                            &direction_i,
                            &direction_j,
                        )?;
                        (
                            -gradient.dot(&direction_ij)
                                + direction_i.dot(&neg_hessian.dot(&direction_j)),
                            neg_third_i.dot(&direction_j) + neg_hessian.dot(&direction_ij),
                            neg_fourth + neg_third_ij,
                        )
                    }
                };
            objective.add(checked_weighted_row_value(
                wi,
                value,
                row_idx,
                "binary baseline psi pair objective",
            )?);
            self.add_pullback_primary_gradient(&mut score, row_idx, &slices, &gradient, wi)?;
            let weighted_information = checked_weighted_row_matrix(
                wi,
                &information,
                row_idx,
                "binary baseline psi pair information",
            )?;
            self.add_pullback_primary_hessian(&mut hessian, row_idx, &slices, &weighted_information);
        }
        let objective_psi_psi = require_finite_likelihood_scalar(
            objective.value(),
            "binary baseline psi pair objective",
        )?;
        require_finite_likelihood_vector(&score, "binary baseline psi pair score")?;
        require_finite_likelihood_matrix(&hessian, "binary baseline psi pair information derivative")?;
        Ok(gam_problem::ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi: score,
            hessian_psi_psi: hessian,
            hessian_psi_psi_operator: None,
        })
    }

    /// Every canonical axis `e_a` of a joint-Hessian derivative whose latent-binary
    /// row kernel is linear in one primary seed, from one lift per row along each
    /// binary primary (#4510):
    ///
    /// ```text
    ///   D[e_a]_i = X_iᵀ (Σ_γ (X_i e_a)_γ F_{i,γ}) X_i.
    /// ```
    ///
    /// `kernel` returns the row kernel `F_{i,γ}`; `channel` names it in refusals.
    /// The binary deployment holds `q̇_exit = 1` and reads no interval bound, so
    /// only the entry offset, the exit offset and the mean are live primaries.
    fn baseline_chart_axes_from_primary_lifts(
        &self,
        block_states: &[ParameterBlockState],
        channel: &str,
        kernel: impl Fn(LatentBinaryPrimaryLift<'_>) -> Result<Array2<f64>, String>,
    ) -> Result<Vec<Array2<f64>>, String> {
        const BINARY_PRIMARIES: [usize; 3] = [
            LATENT_SURVIVAL_PRIMARY_Q_ENTRY,
            LATENT_SURVIVAL_PRIMARY_Q_EXIT,
            LATENT_SURVIVAL_PRIMARY_MU,
        ];
        let weights = ValidatedLikelihoodWeights::new(&self.weights, "latent-binary")
            .map_err(String::from)?;
        let (q_entry, q_exit, mu) = self.split_time_eta(block_states)?;
        let slices = self.joint_slices();
        let total = slices.total;
        let unit = |index: usize, len: usize| {
            let mut axis = Array1::<f64>::zeros(len);
            axis[index] = 1.0;
            axis
        };
        let coefficient_axes: Vec<Array1<f64>> = (0..total).map(|a| unit(a, total)).collect();
        let mut out = vec![Array2::<f64>::zeros((total, total)); total];
        for row_idx in 0..self.event_target.len() {
            let wi = weights.at(row_idx);
            if wi == 0.0 {
                continue;
            }
            let row =
                self.build_right_censored_row_at(row_idx, q_entry[row_idx], q_exit[row_idx])?;
            let point = LatentSurvivalPrimaryPoint {
                q_entry: q_entry[row_idx],
                q_exit: q_exit[row_idx],
                qdot_exit: 1.0,
                q_right: q_exit[row_idx],
                mu: mu[row_idx],
                sigma: self.latent_sd,
            };
            let lifts = BINARY_PRIMARIES
                .iter()
                .map(|&gamma| {
                    kernel(LatentBinaryPrimaryLift {
                        row_idx,
                        row: &row,
                        point,
                        event: self.event_target[row_idx],
                        primary: &unit(gamma, LATENT_SURVIVAL_PRIMARY_DIM),
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            for (axis, target) in coefficient_axes.iter().zip(out.iter_mut()) {
                let direction = self.row_primary_direction_from_flat(row_idx, &slices, axis);
                let mut combined = Array2::<f64>::zeros((
                    LATENT_SURVIVAL_PRIMARY_DIM,
                    LATENT_SURVIVAL_PRIMARY_DIM,
                ));
                for (&gamma, lifted) in BINARY_PRIMARIES.iter().zip(lifts.iter()) {
                    if direction[gamma] != 0.0 {
                        combined.scaled_add(direction[gamma], lifted);
                    }
                }
                let weighted = checked_weighted_row_matrix(wi, &combined, row_idx, channel)?;
                self.add_pullback_primary_hessian(target, row_idx, &slices, &weighted);
            }
        }
        for axis in &out {
            require_finite_likelihood_matrix(axis, channel)?;
        }
        Ok(out)
    }

    /// `{D_β ∂²_{ψ_iψ_j}H[e_a]}` over every coefficient axis for the binary
    /// deployment (#4510), the binary twin of
    /// [`LatentSurvivalFamily::baseline_theta_psisecond_order_hessian_axes_dense`]:
    /// `D_βH_ab[e_γ]_i = X_iᵀ (−∇⁵ℓ_bin,i[d_a, d_b, e_γ] − ∇⁴ℓ_bin,i[d_ab, e_γ]) X_i`,
    /// with the background-scale axis `ln m` read through `φ_bin = ∂ℓ_bin/∂ln m`.
    pub(super) fn baseline_theta_psisecond_order_hessian_axes_dense(
        &self,
        block_states: &[ParameterBlockState],
        rows: &crate::survival::construction::LatentSurvivalOffsetGeometry,
        axis_i: usize,
        axis_j: usize,
    ) -> Result<Vec<Array2<f64>>, String> {
        let n = self.event_target.len();
        let dim = rows.theta.len();
        if axis_i >= dim
            || axis_j >= dim
            || rows.offset_exit_theta.nrows() != n
            || rows.offset_exit_theta_theta.dim() != (n, dim, dim)
        {
            return Err(format!(
                "latent binary baseline psi pair ({axis_i}, {axis_j}) coefficient motion is outside a {dim}-coordinate chart realized over {} rows with second partials {:?} for {n} observations",
                rows.offset_exit_theta.nrows(),
                rows.offset_exit_theta_theta.dim(),
            ));
        }
        let unloaded_axis = rows.unloaded.as_ref().map(|unloaded| unloaded.axis);
        self.baseline_chart_axes_from_primary_lifts(
            block_states,
            "binary baseline psi pair coefficient motion",
            |lift| match (Some(axis_i) == unloaded_axis, Some(axis_j) == unloaded_axis) {
                (true, true) => {
                    let third = latent_binary_row_unloaded_scale_second_third(
                        &self.quadctx,
                        lift.row,
                        lift.point,
                        lift.event,
                        lift.primary,
                    )
                    .map_err(String::from)?;
                    Ok(-third)
                }
                (true, false) | (false, true) => {
                    let loaded = if Some(axis_i) == unloaded_axis {
                        axis_j
                    } else {
                        axis_i
                    };
                    let direction = Self::baseline_theta_row_direction(rows, lift.row_idx, loaded);
                    let fourth = latent_binary_row_unloaded_scale_fourth(
                        &self.quadctx,
                        lift.row,
                        lift.point,
                        lift.event,
                        &direction,
                        lift.primary,
                    )
                    .map_err(String::from)?;
                    Ok(-fourth)
                }
                (false, false) => {
                    let direction_i =
                        Self::baseline_theta_row_direction(rows, lift.row_idx, axis_i);
                    let direction_j =
                        Self::baseline_theta_row_direction(rows, lift.row_idx, axis_j);
                    let direction_ij = Self::baseline_theta_row_second_direction(
                        rows,
                        lift.row_idx,
                        axis_i,
                        axis_j,
                    );
                    let fifth = latent_binary_row_contracted_fifth(
                        &self.quadctx,
                        lift.row,
                        lift.point,
                        lift.event,
                        &direction_i,
                        &direction_j,
                        lift.primary,
                    )
                    .map_err(String::from)?;
                    let fourth = latent_binary_row_contracted_fourth(
                        &self.quadctx,
                        lift.row,
                        lift.point,
                        lift.event,
                        &direction_ij,
                        lift.primary,
                    )
                    .map_err(String::from)?;
                    Ok(fifth + fourth)
                }
            },
        )
    }

    /// `{∂_ψ D²_βH[v, e_a]}` over every coefficient axis for the binary deployment
    /// (#4510), the binary twin of
    /// [`LatentSurvivalFamily::baseline_theta_psihessian_second_directional_axes_dense`]:
    /// a loaded axis reads `−∇⁵ℓ_bin,i[d_θ, X_i v, e_γ]` and the background-scale
    /// axis reads `−∇⁴φ_bin,i[X_i v, e_γ]`.
    pub(super) fn baseline_theta_psihessian_second_directional_axes_dense(
        &self,
        block_states: &[ParameterBlockState],
        rows: &crate::survival::construction::LatentSurvivalOffsetGeometry,
        axis: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Vec<Array2<f64>>, String> {
        let slices = self.joint_slices();
        if d_beta_flat.len() != slices.total || axis >= rows.theta.len() {
            return Err(format!(
                "latent binary baseline psi second coefficient motion: direction length {} against {} coefficients, axis {axis} of {}",
                d_beta_flat.len(),
                slices.total,
                rows.theta.len()
            ));
        }
        let unloaded_axis = rows.unloaded.as_ref().map(|unloaded| unloaded.axis);
        self.baseline_chart_axes_from_primary_lifts(
            block_states,
            "binary baseline psi second coefficient motion",
            |lift| {
                let direction_beta =
                    self.row_primary_direction_from_flat(lift.row_idx, &slices, d_beta_flat);
                if Some(axis) == unloaded_axis {
                    let fourth = latent_binary_row_unloaded_scale_fourth(
                        &self.quadctx,
                        lift.row,
                        lift.point,
                        lift.event,
                        &direction_beta,
                        lift.primary,
                    )
                    .map_err(String::from)?;
                    return Ok(-fourth);
                }
                let direction_theta = Self::baseline_theta_row_direction(rows, lift.row_idx, axis);
                latent_binary_row_contracted_fifth(
                    &self.quadctx,
                    lift.row,
                    lift.point,
                    lift.event,
                    &direction_theta,
                    &direction_beta,
                    lift.primary,
                )
                .map_err(String::from)
            },
        )
    }
}

/// `∇⁴φ[u, v]` of [`latent_survival_row_unloaded_scale_jet`] from one two-seed
/// lift, masked to the live primaries (#4510).
///
/// `∇³φ[u]` ([`latent_survival_row_unloaded_scale_third`]) is what a chart pair
/// `(ln m, loaded)` contributes to the information derivative; this is its motion
/// in a second coefficient direction, the one an armed Jeffreys curvature reads.
fn latent_survival_row_unloaded_scale_fourth(
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    point: LatentSurvivalPrimaryPoint,
    direction_u: &Array1<f64>,
    direction_v: &Array1<f64>,
    include_log_sigma: bool,
) -> Result<Array2<f64>, LatentSurvivalError> {
    let dim = LATENT_SURVIVAL_PRIMARY_DIM;
    if include_log_sigma {
        let backend = LatentTwoSeedBackend {
            direction_u: std::array::from_fn(|a| direction_u[a]),
            direction_v: std::array::from_fn(|a| direction_v[a]),
        };
        let fourth = latent_survival_row_unloaded_scale_jet::<LATENT_SURVIVAL_PRIMARY_DIM, _>(
            &backend, quadctx, row, point,
        )?
        .contracted_fourth();
        Ok(Array2::from_shape_fn((dim, dim), |(a, b)| fourth[a][b]))
    } else {
        let backend = LatentTwoSeedBackend {
            direction_u: std::array::from_fn(|a| direction_u[a]),
            direction_v: std::array::from_fn(|a| direction_v[a]),
        };
        let fourth =
            latent_survival_row_unloaded_scale_jet::<LATENT_SURVIVAL_PRIMARY_LOG_SIGMA, _>(
                &backend, quadctx, row, point,
            )?
            .contracted_fourth();
        let live = |a: usize| a < LATENT_SURVIVAL_PRIMARY_LOG_SIGMA;
        Ok(Array2::from_shape_fn((dim, dim), |(a, b)| {
            if live(a) && live(b) {
                fourth[a][b]
            } else {
                0.0
            }
        }))
    }
}

/// `∇³(∂φ/∂ln m)[u]` of [`latent_survival_row_background_scale_jet`] at order
/// `Second`, from one one-seed lift, masked to the live primaries (#4510): the
/// coefficient motion of the `(ln m, ln m)` pair's information derivative
/// `−∇²(∂φ/∂ln m)`.
fn latent_survival_row_unloaded_scale_second_third(
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    point: LatentSurvivalPrimaryPoint,
    direction: &Array1<f64>,
    include_log_sigma: bool,
) -> Result<Array2<f64>, LatentSurvivalError> {
    let dim = LATENT_SURVIVAL_PRIMARY_DIM;
    if include_log_sigma {
        let backend = LatentOneSeedBackend {
            direction: std::array::from_fn(|a| direction[a]),
        };
        let third = latent_survival_row_background_scale_jet::<LATENT_SURVIVAL_PRIMARY_DIM, _>(
            &backend,
            quadctx,
            row,
            point,
            LatentBackgroundScaleOrder::Second,
        )?
        .contracted_third();
        Ok(Array2::from_shape_fn((dim, dim), |(a, b)| third[a][b]))
    } else {
        let backend = LatentOneSeedBackend {
            direction: std::array::from_fn(|a| direction[a]),
        };
        let third =
            latent_survival_row_background_scale_jet::<LATENT_SURVIVAL_PRIMARY_LOG_SIGMA, _>(
                &backend,
                quadctx,
                row,
                point,
                LatentBackgroundScaleOrder::Second,
            )?
            .contracted_third();
        let live = |a: usize| a < LATENT_SURVIVAL_PRIMARY_LOG_SIGMA;
        Ok(Array2::from_shape_fn((dim, dim), |(a, b)| {
            if live(a) && live(b) { third[a][b] } else { 0.0 }
        }))
    }
}

/// `∇⁴φ_bin[u, v]` of [`latent_binary_row_unloaded_scale_jet`] from one two-seed
/// fixed-σ lift, masked to the live primaries (#4510): the binary twin of
/// [`latent_survival_row_unloaded_scale_fourth`].
fn latent_binary_row_unloaded_scale_fourth(
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    point: LatentSurvivalPrimaryPoint,
    event: u8,
    direction_u: &Array1<f64>,
    direction_v: &Array1<f64>,
) -> Result<Array2<f64>, LatentSurvivalError> {
    let dim = LATENT_SURVIVAL_PRIMARY_DIM;
    let live = |a: usize| a < LATENT_SURVIVAL_PRIMARY_LOG_SIGMA;
    let backend = LatentTwoSeedBackend {
        direction_u: std::array::from_fn(|a| direction_u[a]),
        direction_v: std::array::from_fn(|a| direction_v[a]),
    };
    let fourth = latent_binary_row_unloaded_scale_jet(&backend, quadctx, row, point, event)?
        .contracted_fourth();
    Ok(Array2::from_shape_fn((dim, dim), |(a, b)| {
        if live(a) && live(b) {
            fourth[a][b]
        } else {
            0.0
        }
    }))
}

/// `∇³(∂φ_bin/∂ln m)[u]` of [`latent_binary_row_background_scale_jet`] at order
/// `Second`, from one one-seed fixed-σ lift, masked to the live primaries
/// (#4510): the binary twin of [`latent_survival_row_unloaded_scale_second_third`].
fn latent_binary_row_unloaded_scale_second_third(
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    point: LatentSurvivalPrimaryPoint,
    event: u8,
    direction: &Array1<f64>,
) -> Result<Array2<f64>, LatentSurvivalError> {
    let dim = LATENT_SURVIVAL_PRIMARY_DIM;
    let live = |a: usize| a < LATENT_SURVIVAL_PRIMARY_LOG_SIGMA;
    let backend = LatentOneSeedBackend {
        direction: std::array::from_fn(|a| direction[a]),
    };
    let third = latent_binary_row_background_scale_jet(
        &backend,
        quadctx,
        row,
        point,
        event,
        LatentBackgroundScaleOrder::Second,
    )?
    .contracted_third();
    Ok(Array2::from_shape_fn((dim, dim), |(a, b)| {
        if live(a) && live(b) { third[a][b] } else { 0.0 }
    }))
}

/// One latent-binary row lift along a single primary seed (#4510), the binary
/// twin of `LatentSurvivalPrimaryLift`.
struct LatentBinaryPrimaryLift<'a> {
    row_idx: usize,
    row: &'a LatentSurvivalRow,
    point: LatentSurvivalPrimaryPoint,
    event: u8,
    primary: &'a Array1<f64>,
}
