//! The parametric baseline's shape θ as family-owned outer hyper axes (#3413).
//!
//! A non-linear baseline target (Weibull, Gompertz, Gompertz–Makeham) enters
//! the survival location-scale likelihood only through the time block's fixed
//! offsets: `offset(θ) = fixed + b(θ)` on the stacked `[exit; entry; deriv]`
//! channels. The outer REML/LAML criterion is therefore a smooth function of θ
//! whose partials at fixed β are the partials of the row likelihood along the
//! offset tangent `∂b/∂θ_k`.
//!
//! An offset shift along a tangent `t` is exactly a new time-block column `t`
//! whose coefficient is held at zero: the time block's predictor is linear in
//! its stacked design, every downstream quantity (the time warp, its wiggle
//! evaluated at the base predictor, the link wiggle) reads the time block only
//! through that predictor, and the time block's Jacobian is the design pushed
//! through the same chain rule. So prepending `t` as coefficient 0 of the
//! time block gives, at `β₀ = 0`,
//!
//! * `∂(−ℓ)/∂θ_k            = ∂(−ℓ)/∂β₀`,
//! * `∂θ_k ∇_β(−ℓ)          = H[1.., 0]`,
//! * `∂θ_k ∇²_β(−ℓ)         = (D_β H[e₀])[1.., 1..]`,
//! * `D_β(∂θ_k ∇²_β(−ℓ))[u] = (D²_β H[e₀, (0, u)])[1.., 1..]`,
//!
//! all from the family's own exact row program, with and without a link or
//! time wiggle. No second model of the likelihood is kept.

use super::*;

/// Per-row θ-tangents of the baseline time offsets, one column per baseline
/// shape axis: `entry[i, k] = ∂b_entry(i)/∂θ_k` (zero for a row entering at the
/// origin), `exit[i, k] = ∂b_exit(i)/∂θ_k`, `deriv[i, k] = ∂b'_exit(i)/∂θ_k`.
#[derive(Clone, Debug)]
pub struct SurvivalBaselineThetaTangents {
    pub entry: Array2<f64>,
    pub exit: Array2<f64>,
    pub deriv: Array2<f64>,
}

impl SurvivalBaselineThetaTangents {
    pub fn axis_count(&self) -> usize {
        self.exit.ncols()
    }
}

fn prepend_column(design: &Array2<f64>, column: ArrayView1<'_, f64>) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((design.nrows(), design.ncols() + 1));
    out.column_mut(0).assign(&column);
    out.slice_mut(s![.., 1..]).assign(design);
    out
}

impl SurvivalLocationScaleFamily {
    /// Number of free inverse-link shape parameters, the leading family-owned
    /// hyper axes (#2904). The baseline θ axes follow them.
    pub(crate) fn link_param_axis_count(&self) -> Result<usize, String> {
        use gam_solve::mixture_link::{InverseLinkKernel, LinkParamPartials};
        Ok(
            match self
                .inverse_link
                .param_partials(0.0)
                .map_err(|e| format!("inverse-link param partials probe failed: {e}"))?
            {
                None => 0,
                Some(LinkParamPartials::Sas(_)) => 2,
                Some(LinkParamPartials::Mixture(partials)) => partials.djet_drho.len(),
            },
        )
    }

    /// The family and block states with baseline tangent `axis` prepended as
    /// time-block coefficient 0 at value zero. The predictor, and so every
    /// likelihood quantity at the current β, is unchanged.
    fn baseline_theta_pseudo_column(
        &self,
        block_states: &[ParameterBlockState],
        axis: usize,
    ) -> Result<(Self, Vec<ParameterBlockState>), String> {
        let tangents = self.baseline_theta_tangents.as_ref().ok_or_else(|| {
            SurvivalLocationScaleError::InvalidConfiguration {
                reason: format!(
                    "baseline θ hyper axis {axis} requested but the family carries no baseline \
                     tangents"
                ),
            }
            .to_string()
        })?;
        let count = tangents.axis_count();
        if axis >= count {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "baseline θ hyper axis {axis} is out of range for a baseline with {count} \
                     shape parameters"
                ),
            }
            .into());
        }
        if block_states.len() != self.expected_blocks() {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "baseline θ hyper terms expect {} block states, got {}",
                    self.expected_blocks(),
                    block_states.len()
                ),
            }
            .into());
        }
        let mut augmented = self.clone();
        augmented.x_time_entry = Arc::new(prepend_column(
            &self.x_time_entry,
            tangents.entry.column(axis),
        ));
        augmented.x_time_exit = Arc::new(prepend_column(
            &self.x_time_exit,
            tangents.exit.column(axis),
        ));
        augmented.x_time_deriv = Arc::new(prepend_column(
            &self.x_time_deriv,
            tangents.deriv.column(axis),
        ));
        // The pseudo coefficient is never stepped, so it carries no feasibility
        // constraint; the tangent is not re-applied inside the augmented family.
        augmented.time_linear_constraints = None;
        augmented.baseline_theta_tangents = None;
        let mut states = block_states.to_vec();
        let time = &block_states[Self::BLOCK_TIME];
        let mut beta = Array1::<f64>::zeros(time.beta.len() + 1);
        beta.slice_mut(s![1..]).assign(&time.beta);
        states[Self::BLOCK_TIME] = ParameterBlockState {
            beta,
            eta: time.eta.clone(),
        };
        Ok((augmented, states))
    }

    /// First-order terms of baseline shape axis `axis` as a family-owned hyper
    /// axis: `objective_psi = ∂(−ℓ)/∂θ`, `score_psi = ∂θ ∇_β(−ℓ)` and
    /// `hessian_psi = ∂θ ∇²_β(−ℓ)` at fixed β, in the unscaled observed
    /// information the other ψ terms use.
    pub(crate) fn baseline_theta_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        axis: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        let (augmented, states) = self.baseline_theta_pseudo_column(block_states, axis)?;
        let (_, gradients) = augmented.evaluate_log_likelihood_and_block_gradients(&states)?;
        // Block gradients are ∇ℓ; the ψ objective is the NLL partial.
        let objective_psi = -gradients[Self::BLOCK_TIME][0];
        let hessian = CustomFamily::exact_newton_joint_hessian(&augmented, &states)?
            .ok_or_else(|| "baseline θ hyper terms require the exact joint Hessian".to_string())?;
        let score_psi = hessian.slice(s![1.., 0]).to_owned();
        let mut pseudo_axis = Array1::<f64>::zeros(hessian.nrows());
        pseudo_axis[0] = 1.0;
        let d_hessian = augmented
            .exact_newton_joint_hessian_directional_derivative_rescaled(
                &states,
                &pseudo_axis,
                0.0,
            )?
            .ok_or_else(|| {
                "baseline θ hyper terms require the exact joint Hessian directional derivative"
                    .to_string()
            })?;
        let hessian_psi = d_hessian.slice(s![1.., 1..]).to_owned();
        Ok(Some(ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi,
            hessian_psi_operator: None,
        }))
    }

    /// Mixed coefficient drift `D_β H_θ[u]` of the observed information along
    /// baseline shape axis `axis` over `rows`: the β-directional derivative of
    /// [`Self::baseline_theta_joint_psi_terms`]' `hessian_psi`, read by the
    /// explicit-ψ Jeffreys score correction and curvature drift.
    pub(crate) fn baseline_theta_joint_psihessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        axis: usize,
        d_beta_flat: &[f64],
        rows: &crate::row_kernel::RowSet,
    ) -> Result<Option<Array2<f64>>, String> {
        let (augmented, states) = self.baseline_theta_pseudo_column(block_states, axis)?;
        let p_augmented = *augmented
            .joint_block_offsets()
            .last()
            .ok_or_else(|| "missing joint block offsets".to_string())?;
        if d_beta_flat.len() + 1 != p_augmented {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "baseline θ drift direction has length {}, expected {}",
                    d_beta_flat.len(),
                    p_augmented - 1
                ),
            }
            .into());
        }
        let mut pseudo_axis = vec![0.0; p_augmented];
        pseudo_axis[0] = 1.0;
        let mut direction = vec![0.0; p_augmented];
        direction[1..].copy_from_slice(d_beta_flat);
        let dynamic = augmented.build_dynamic_geometry(&states)?;
        let second = if augmented.x_link_wiggle.is_some() {
            super::row_kernel::survival_ls_wiggle_second_directional_derivative_dense(
                &augmented,
                &dynamic,
                0.0,
                rows,
                &pseudo_axis,
                &direction,
            )?
        } else {
            let kernel = augmented.survival_ls_row_kernel_rescaled(&dynamic, 0.0);
            crate::row_kernel::row_kernel_second_directional_derivative(
                &kernel,
                rows,
                &pseudo_axis,
                &direction,
            )?
        };
        Ok(Some(second.slice(s![1.., 1..]).to_owned()))
    }
}
