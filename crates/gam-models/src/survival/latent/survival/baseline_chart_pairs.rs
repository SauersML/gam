//! Fixed-β second-order terms of a latent family's baseline-chart axis pairs
//! (#2677): the explicit `(V_ij, g_ij, H_ij)` an exact outer Hessian over the chart
//! coordinates reads through `exact_newton_joint_psisecond_order_terms`.
//!
//! Also the chart axes' third information derivatives `{D_βa D_βv D_θ H}` and
//! `{D_βa D_θi D_θj H}`, which an armed Jeffreys term's exact outer Hessian reads
//! along the chart, and the owned exact-ψ workspace through which the evaluator
//! reads every chart axis's ψ calculus.

use super::*;

/// `−∂_u ∂_v ∇²φ` of the background-scale derivative `φ = ∂ℓ/∂ln m` of one latent
/// survival row ([`latent_survival_row_unloaded_scale_jet`]) from one two-seed lift,
/// masked to the live primaries (#2677).
fn latent_survival_row_unloaded_scale_information_fourth(
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    point: LatentSurvivalPrimaryPoint,
    direction_u: &Array1<f64>,
    direction_v: &Array1<f64>,
    include_log_sigma: bool,
) -> Result<Array2<f64>, LatentSurvivalError> {
    let dim = LATENT_SURVIVAL_PRIMARY_DIM;
    if include_log_sigma {
        let backend = LatentTwoSeedBackend::<LATENT_SURVIVAL_PRIMARY_DIM> {
            direction_u: std::array::from_fn(|a| direction_u[a]),
            direction_v: std::array::from_fn(|a| direction_v[a]),
        };
        let fourth = latent_survival_row_unloaded_scale_jet::<LATENT_SURVIVAL_PRIMARY_DIM, _>(
            &backend, quadctx, row, point,
        )?
        .contracted_fourth();
        Ok(Array2::from_shape_fn((dim, dim), |(a, b)| -fourth[a][b]))
    } else {
        let backend = LatentTwoSeedBackend::<LATENT_SURVIVAL_PRIMARY_LOG_SIGMA> {
            direction_u: std::array::from_fn(|a| direction_u[a]),
            direction_v: std::array::from_fn(|a| direction_v[a]),
        };
        let fourth = latent_survival_row_unloaded_scale_jet::<LATENT_SURVIVAL_PRIMARY_LOG_SIGMA, _>(
            &backend, quadctx, row, point,
        )?
        .contracted_fourth();
        let live = |a: usize| a < LATENT_SURVIVAL_PRIMARY_LOG_SIGMA;
        Ok(Array2::from_shape_fn((dim, dim), |(a, b)| {
            if live(a) && live(b) {
                -fourth[a][b]
            } else {
                0.0
            }
        }))
    }
}

/// `−∂_u ∇²(∂φ/∂ln m)` of one latent survival row from one one-seed lift of the
/// second background-scale jet ([`latent_survival_row_background_scale_jet`]),
/// masked to the live primaries (#2677).
fn latent_survival_row_unloaded_scale_second_information_third(
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
        Ok(Array2::from_shape_fn((dim, dim), |(a, b)| -third[a][b]))
    } else {
        let backend = LatentOneSeedBackend {
            direction: std::array::from_fn(|a| direction[a]),
        };
        let third = latent_survival_row_background_scale_jet::<LATENT_SURVIVAL_PRIMARY_LOG_SIGMA, _>(
            &backend,
            quadctx,
            row,
            point,
            LatentBackgroundScaleOrder::Second,
        )?
        .contracted_third();
        let live = |a: usize| a < LATENT_SURVIVAL_PRIMARY_LOG_SIGMA;
        Ok(Array2::from_shape_fn((dim, dim), |(a, b)| {
            if live(a) && live(b) {
                -third[a][b]
            } else {
                0.0
            }
        }))
    }
}

/// `−∂_u ∂_v ∇²φ` of a latent-binary row's background-scale derivative
/// ([`latent_binary_row_unloaded_scale_jet`]) from one two-seed fixed-σ lift (#2677).
fn latent_binary_row_unloaded_scale_information_fourth(
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    point: LatentSurvivalPrimaryPoint,
    event: u8,
    direction_u: &Array1<f64>,
    direction_v: &Array1<f64>,
) -> Result<Array2<f64>, LatentSurvivalError> {
    let dim = LATENT_SURVIVAL_PRIMARY_DIM;
    let live = |a: usize| a < LATENT_SURVIVAL_PRIMARY_LOG_SIGMA;
    let backend = LatentTwoSeedBackend::<LATENT_SURVIVAL_PRIMARY_LOG_SIGMA> {
        direction_u: std::array::from_fn(|a| direction_u[a]),
        direction_v: std::array::from_fn(|a| direction_v[a]),
    };
    let fourth = latent_binary_row_unloaded_scale_jet(&backend, quadctx, row, point, event)?
        .contracted_fourth();
    Ok(Array2::from_shape_fn((dim, dim), |(a, b)| {
        if live(a) && live(b) {
            -fourth[a][b]
        } else {
            0.0
        }
    }))
}

/// `−∂_u ∇²(∂φ/∂ln m)` of a latent-binary row from one one-seed fixed-σ lift of the
/// second background-scale jet ([`latent_binary_row_background_scale_jet`]) (#2677).
fn latent_binary_row_unloaded_scale_second_information_third(
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    point: LatentSurvivalPrimaryPoint,
    event: u8,
    direction: &Array1<f64>,
) -> Result<Array2<f64>, LatentSurvivalError> {
    let dim = LATENT_SURVIVAL_PRIMARY_DIM;
    let live = |a: usize| a < LATENT_SURVIVAL_PRIMARY_LOG_SIGMA;
    let backend = LatentOneSeedBackend::<LATENT_SURVIVAL_PRIMARY_LOG_SIGMA> {
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
        if live(a) && live(b) {
            -third[a][b]
        } else {
            0.0
        }
    }))
}

/// The baseline-chart axes' exact-ψ workspace at one evaluation's states (#2677).
/// The evaluator reads a family-owned axis's coefficient drift only through an
/// owned workspace (`build_psi_drift_deriv_callback`); this one serves the family's
/// own chart hooks at the states, specs and layout it was built from.
pub(super) struct LatentBaselineChartPsiWorkspace<F> {
    family: F,
    block_states: Vec<ParameterBlockState>,
    specs: Vec<ParameterBlockSpec>,
    hyper_layout: crate::custom_family::CustomFamilyHyperLayout,
}

impl<F> LatentBaselineChartPsiWorkspace<F> {
    pub(super) fn new(
        family: F,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        hyper_layout: &crate::custom_family::CustomFamilyHyperLayout,
    ) -> Self {
        Self {
            family,
            block_states: block_states.to_vec(),
            specs: specs.to_vec(),
            hyper_layout: hyper_layout.clone(),
        }
    }
}

impl<F: CustomFamily + Send + Sync> gam_problem::ExactNewtonJointPsiWorkspace
    for LatentBaselineChartPsiWorkspace<F>
{
    fn first_order_terms(
        &self,
        psi_index: usize,
    ) -> Result<Option<gam_problem::ExactNewtonJointPsiTerms>, String> {
        self.family.exact_newton_joint_psi_terms(
            &self.block_states,
            &self.specs,
            &self.hyper_layout,
            psi_index,
        )
    }

    fn second_order_terms(
        &self,
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<gam_problem::ExactNewtonJointPsiSecondOrderTerms>, String> {
        self.family.exact_newton_joint_psisecond_order_terms(
            &self.block_states,
            &self.specs,
            &self.hyper_layout,
            psi_i,
            psi_j,
        )
    }

    fn hessian_directional_derivative(
        &self,
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<gam_problem::DriftDerivResult>, String> {
        Ok(self
            .family
            .exact_newton_joint_psihessian_directional_derivative(
                &self.block_states,
                &self.specs,
                &self.hyper_layout,
                psi_index,
                d_beta_flat,
            )?
            .map(gam_problem::DriftDerivResult::Dense))
    }

    fn hessian_second_directional_derivative_all_beta_axes(
        &self,
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Vec<Array2<f64>>>, String> {
        self.family
            .exact_newton_joint_psihessian_second_directional_derivative_all_beta_axes(
                &self.block_states,
                &self.specs,
                &self.hyper_layout,
                psi_index,
                d_beta_flat,
            )
    }

    fn second_order_hessian_directional_derivative_all_beta_axes(
        &self,
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<Vec<Array2<f64>>>, String> {
        self.family
            .exact_newton_joint_psisecond_order_hessian_directional_derivative_all_beta_axes(
                &self.block_states,
                &self.specs,
                &self.hyper_layout,
                psi_i,
                psi_j,
            )
    }
}

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

impl LatentSurvivalFamily {
    /// `{D_βa D_βv D_θ H}` of baseline-chart axis `axis` along every coefficient axis
    /// `a` (#2677): the β-drift of `D_βv D_θ H`
    /// ([`Self::baseline_theta_hessian_directional_derivative_dense`]). The chart and
    /// the design Jacobian do not move with β, so a loaded axis reads
    /// `−∇⁵ℓ_i[d_i, X_i v, X_i e_a]` and the background-scale axis `ln m` reads
    /// `−∇⁴φ_i[X_i v, X_i e_a]`.
    pub(super) fn baseline_theta_hessian_second_directional_derivative_all_axes_dense(
        &self,
        block_states: &[ParameterBlockState],
        rows: &crate::survival::construction::LatentSurvivalOffsetGeometry,
        axis: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Vec<Array2<f64>>, String> {
        let slices = self.joint_slices();
        if d_beta_flat.len() != slices.total || axis >= rows.theta.len() {
            return Err(format!(
                "latent survival baseline psi third derivative: direction length {} against {} coefficients, axis {axis} of {}",
                d_beta_flat.len(),
                slices.total,
                rows.theta.len()
            ));
        }
        let include_log_sigma = slices.log_sigma.is_some();
        let unloaded = rows
            .unloaded
            .as_ref()
            .is_some_and(|unloaded| unloaded.axis == axis);
        self.joint_hessian_axes_from_primary_lifts(
            block_states,
            "baseline psi mixed fifth",
            |lift| -> Result<Array2<f64>, String> {
                let direction_beta =
                    self.row_primary_direction_from_flat(lift.row_idx, &slices, d_beta_flat);
                if unloaded {
                    return Ok(latent_survival_row_unloaded_scale_information_fourth(
                        &self.quadctx,
                        lift.row,
                        lift.point,
                        &direction_beta,
                        lift.primary,
                        include_log_sigma,
                    )?);
                }
                let direction_theta = Self::baseline_theta_row_direction(rows, lift.row_idx, axis);
                Ok(latent_survival_row_primary_fifth_contracted(
                    &self.quadctx,
                    lift.row,
                    lift.point,
                    &direction_theta,
                    &direction_beta,
                    lift.primary,
                    include_log_sigma,
                )?)
            },
        )
    }

    /// `{D_βa D_θi D_θj H}` of baseline-chart axes `axis_i` and `axis_j` along every
    /// coefficient axis `a` (#2677): the β-drift of the pair's `H_ij`
    /// ([`Self::baseline_theta_psisecond_order_terms_dense`]). A loaded pair reads
    /// `−∇⁵ℓ_i[d_i, d_j, X_i e_a] − ∇⁴ℓ_i[d_ij, X_i e_a]`, a `(ln m, loaded)` pair
    /// `−∇⁴φ_i[d_loaded, X_i e_a]`, and `(ln m, ln m)` `−∇³(∂φ_i/∂ln m)[X_i e_a]`.
    pub(super) fn baseline_theta_psisecond_order_hessian_directional_derivative_all_axes_dense(
        &self,
        block_states: &[ParameterBlockState],
        rows: &crate::survival::construction::LatentSurvivalOffsetGeometry,
        axis_i: usize,
        axis_j: usize,
    ) -> Result<Vec<Array2<f64>>, String> {
        let dim = rows.theta.len();
        if axis_i >= dim || axis_j >= dim {
            return Err(format!(
                "latent survival baseline psi pair third derivative: pair ({axis_i}, {axis_j}) is outside a {dim}-coordinate chart"
            ));
        }
        let include_log_sigma = self.joint_slices().log_sigma.is_some();
        let unloaded_axis = rows.unloaded.as_ref().map(|unloaded| unloaded.axis);
        self.joint_hessian_axes_from_primary_lifts(
            block_states,
            "baseline psi pair mixed fifth",
            |lift| -> Result<Array2<f64>, String> {
                Ok(
                    match (Some(axis_i) == unloaded_axis, Some(axis_j) == unloaded_axis) {
                        (true, true) => latent_survival_row_unloaded_scale_second_information_third(
                            &self.quadctx,
                            lift.row,
                            lift.point,
                            lift.primary,
                            include_log_sigma,
                        )?,
                        (true, false) | (false, true) => {
                            let loaded = if Some(axis_i) == unloaded_axis {
                                axis_j
                            } else {
                                axis_i
                            };
                            let direction =
                                Self::baseline_theta_row_direction(rows, lift.row_idx, loaded);
                            latent_survival_row_unloaded_scale_information_fourth(
                                &self.quadctx,
                                lift.row,
                                lift.point,
                                &direction,
                                lift.primary,
                                include_log_sigma,
                            )?
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
                            latent_survival_row_primary_fifth_contracted(
                                &self.quadctx,
                                lift.row,
                                lift.point,
                                &direction_i,
                                &direction_j,
                                lift.primary,
                                include_log_sigma,
                            )? + latent_survival_row_primary_fourth_contracted(
                                &self.quadctx,
                                lift.row,
                                lift.point,
                                &direction_ij,
                                lift.primary,
                                include_log_sigma,
                            )?
                        }
                    },
                )
            },
        )
    }
}

impl LatentBinaryFamily {
    /// Every coefficient axis `e_a` of a binary joint-Hessian derivative whose row
    /// kernel is linear in one primary seed (#2677), the binary twin of
    /// [`LatentSurvivalFamily::joint_hessian_axes_from_primary_lifts`]:
    /// `D[e_a]_i = X_iᵀ (Σ_γ (X_i e_a)_γ F_{i,γ}) X_i` over the entry, exit and mean
    /// primaries, with `lift(row, point, event, e_γ)` the row kernel `F_{i,γ}`.
    fn joint_hessian_axes_from_binary_primary_lifts(
        &self,
        block_states: &[ParameterBlockState],
        channel: &str,
        lift: impl Fn(
            usize,
            &LatentSurvivalRow,
            LatentSurvivalPrimaryPoint,
            u8,
            &Array1<f64>,
        ) -> Result<Array2<f64>, String>,
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
            let event = self.event_target[row_idx];
            let kernels = BINARY_PRIMARIES
                .iter()
                .map(|&gamma| {
                    lift(
                        row_idx,
                        &row,
                        point,
                        event,
                        &unit(gamma, LATENT_SURVIVAL_PRIMARY_DIM),
                    )
                })
                .collect::<Result<Vec<_>, String>>()?;
            for (axis, target) in coefficient_axes.iter().zip(out.iter_mut()) {
                let direction = self.row_primary_direction_from_flat(row_idx, &slices, axis);
                let mut combined =
                    Array2::<f64>::zeros((LATENT_SURVIVAL_PRIMARY_DIM, LATENT_SURVIVAL_PRIMARY_DIM));
                for (&gamma, kernel) in BINARY_PRIMARIES.iter().zip(kernels.iter()) {
                    if direction[gamma] != 0.0 {
                        combined.scaled_add(direction[gamma], kernel);
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

    /// `{D_βa D_βv D_θ H}` of baseline-chart axis `axis` for the binary deployment
    /// (#2677), the twin of
    /// [`LatentSurvivalFamily::baseline_theta_hessian_second_directional_derivative_all_axes_dense`].
    pub(super) fn baseline_theta_hessian_second_directional_derivative_all_axes_dense(
        &self,
        block_states: &[ParameterBlockState],
        rows: &crate::survival::construction::LatentSurvivalOffsetGeometry,
        axis: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Vec<Array2<f64>>, String> {
        let slices = self.joint_slices();
        if d_beta_flat.len() != slices.total || axis >= rows.theta.len() {
            return Err(format!(
                "latent binary baseline psi third derivative: direction length {} against {} coefficients, axis {axis} of {}",
                d_beta_flat.len(),
                slices.total,
                rows.theta.len()
            ));
        }
        let unloaded = rows
            .unloaded
            .as_ref()
            .is_some_and(|unloaded| unloaded.axis == axis);
        self.joint_hessian_axes_from_binary_primary_lifts(
            block_states,
            "binary baseline psi mixed fifth",
            |row_idx, row, point, event, primary| -> Result<Array2<f64>, String> {
                let direction_beta = self.row_primary_direction_from_flat(row_idx, &slices, d_beta_flat);
                if unloaded {
                    return Ok(latent_binary_row_unloaded_scale_information_fourth(
                        &self.quadctx,
                        row,
                        point,
                        event,
                        &direction_beta,
                        primary,
                    )?);
                }
                let direction_theta = Self::baseline_theta_row_direction(rows, row_idx, axis);
                Ok(latent_binary_row_contracted_fifth(
                    &self.quadctx,
                    row,
                    point,
                    event,
                    &direction_theta,
                    &direction_beta,
                    primary,
                )?)
            },
        )
    }

    /// `{D_βa D_θi D_θj H}` of baseline-chart axes `axis_i` and `axis_j` for the
    /// binary deployment (#2677), the twin of
    /// [`LatentSurvivalFamily::baseline_theta_psisecond_order_hessian_directional_derivative_all_axes_dense`].
    pub(super) fn baseline_theta_psisecond_order_hessian_directional_derivative_all_axes_dense(
        &self,
        block_states: &[ParameterBlockState],
        rows: &crate::survival::construction::LatentSurvivalOffsetGeometry,
        axis_i: usize,
        axis_j: usize,
    ) -> Result<Vec<Array2<f64>>, String> {
        let dim = rows.theta.len();
        if axis_i >= dim || axis_j >= dim {
            return Err(format!(
                "latent binary baseline psi pair third derivative: pair ({axis_i}, {axis_j}) is outside a {dim}-coordinate chart"
            ));
        }
        let unloaded_axis = rows.unloaded.as_ref().map(|unloaded| unloaded.axis);
        self.joint_hessian_axes_from_binary_primary_lifts(
            block_states,
            "binary baseline psi pair mixed fifth",
            |row_idx, row, point, event, primary| -> Result<Array2<f64>, String> {
                Ok(
                    match (Some(axis_i) == unloaded_axis, Some(axis_j) == unloaded_axis) {
                        (true, true) => latent_binary_row_unloaded_scale_second_information_third(
                            &self.quadctx,
                            row,
                            point,
                            event,
                            primary,
                        )?,
                        (true, false) | (false, true) => {
                            let loaded = if Some(axis_i) == unloaded_axis {
                                axis_j
                            } else {
                                axis_i
                            };
                            let direction = Self::baseline_theta_row_direction(rows, row_idx, loaded);
                            latent_binary_row_unloaded_scale_information_fourth(
                                &self.quadctx,
                                row,
                                point,
                                event,
                                &direction,
                                primary,
                            )?
                        }
                        (false, false) => {
                            let direction_i = Self::baseline_theta_row_direction(rows, row_idx, axis_i);
                            let direction_j = Self::baseline_theta_row_direction(rows, row_idx, axis_j);
                            let direction_ij =
                                Self::baseline_theta_row_second_direction(rows, row_idx, axis_i, axis_j);
                            latent_binary_row_contracted_fifth(
                                &self.quadctx,
                                row,
                                point,
                                event,
                                &direction_i,
                                &direction_j,
                                primary,
                            )? + latent_binary_row_contracted_fourth(
                                &self.quadctx,
                                row,
                                point,
                                event,
                                &direction_ij,
                                primary,
                            )?
                        }
                    },
                )
            },
        )
    }
}
