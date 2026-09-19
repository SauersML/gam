//! Fixed-β second-order terms of a latent family's baseline-chart axis pairs
//! (#2677): the explicit `(V_ij, g_ij, H_ij)` an exact outer Hessian over the chart
//! coordinates reads through `exact_newton_joint_psisecond_order_terms`.

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
}
