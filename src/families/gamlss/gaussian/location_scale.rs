// Real concern-organized submodule of the gamlss family stack.
// Cross-module items are re-exported flat through the parent (`gamlss.rs`),
// so `use super::*;` makes the sibling-concern symbols this module references
// resolve through the parent namespace.
use super::*;

pub struct GaussianLocationScaleFamily {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub mu_design: Option<DesignMatrix>,
    pub log_sigma_design: Option<DesignMatrix>,
    /// Resource policy threaded into PsiDesignMap construction (and any other
    /// per-call materialization decision) made during exact-Newton joint psi
    /// derivative evaluation. Defaults to `ResourcePolicy::default_library()`
    /// when the family is built without an explicit policy.
    pub policy: crate::resource::ResourcePolicy,
    /// Cached per-observation row scalars keyed by 6-element fingerprint
    /// (first, mid, last elements of both eta vectors).
    /// Avoids recomputing O(n) scalars K+ times per REML gradient/Hessian evaluation.
    pub cached_row_scalars:
        std::sync::RwLock<Option<(f64, f64, f64, f64, f64, f64, Arc<GaussianJointRowScalars>)>>,
}

impl Clone for GaussianLocationScaleFamily {
    fn clone(&self) -> Self {
        Self {
            y: self.y.clone(),
            weights: self.weights.clone(),
            mu_design: self.mu_design.clone(),
            log_sigma_design: self.log_sigma_design.clone(),
            policy: self.policy.clone(),
            cached_row_scalars: std::sync::RwLock::new(
                self.cached_row_scalars
                    .read()
                    .expect("lock poisoned")
                    .clone(),
            ),
        }
    }
}

impl GaussianLocationScaleFamily {
    pub const BLOCK_MU: usize = 0;
    pub const BLOCK_LOG_SIGMA: usize = 1;

    pub(crate) fn get_or_compute_row_scalars(
        &self,
        etamu: &Array1<f64>,
        eta_ls: &Array1<f64>,
    ) -> Result<Arc<GaussianJointRowScalars>, String> {
        Ok(Arc::new(gaussian_jointrow_scalars(
            &self.y,
            etamu,
            eta_ls,
            &self.weights,
        )?))
    }

    pub fn parameternames() -> &'static [&'static str] {
        &["mu", "log_sigma"]
    }

    pub fn parameter_links() -> &'static [ParameterLink] {
        &[ParameterLink::Identity, ParameterLink::Log]
    }

    pub fn metadata() -> FamilyMetadata {
        FamilyMetadata {
            name: "gaussian_location_scale",
            parameternames: Self::parameternames(),
            parameter_links: Self::parameter_links(),
        }
    }

    pub(crate) fn exact_joint_supported(&self) -> bool {
        self.mu_design.is_some() && self.log_sigma_design.is_some()
    }

    pub(crate) fn exact_block_designs(
        &self,
    ) -> Result<(DenseOrOperator<'_>, DenseOrOperator<'_>), String> {
        let mu_design = self.mu_design.as_ref().ok_or_else(|| {
            "GaussianLocationScaleFamily exact path is missing mu design".to_string()
        })?;
        let log_sigma_design = self.log_sigma_design.as_ref().ok_or_else(|| {
            "GaussianLocationScaleFamily exact path is missing log-sigma design".to_string()
        })?;
        let planned = dense_blocks_planned_budget(&[mu_design, log_sigma_design]);
        let xmu = dense_block_or_operator(
            mu_design,
            mu_design.nrows(),
            mu_design.ncols(),
            planned[0],
            &self.policy,
        );
        let x_ls = dense_block_or_operator(
            log_sigma_design,
            log_sigma_design.nrows(),
            log_sigma_design.ncols(),
            planned[1],
            &self.policy,
        );
        Ok((xmu, x_ls))
    }

    pub(crate) fn exact_block_designs_fromspecs<'a>(
        &self,
        specs: &'a [ParameterBlockSpec],
    ) -> Result<(DenseOrOperator<'a>, DenseOrOperator<'a>), String> {
        if specs.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleFamily spec-aware exact path expects 2 specs, got {}",
                    specs.len()
                ),
            }
            .into());
        }
        let mu_design = &specs[Self::BLOCK_MU].design;
        let log_sigma_design = &specs[Self::BLOCK_LOG_SIGMA].design;
        let planned = dense_blocks_planned_budget(&[mu_design, log_sigma_design]);
        let xmu = dense_block_or_operator(
            mu_design,
            mu_design.nrows(),
            mu_design.ncols(),
            planned[0],
            &self.policy,
        );
        let x_ls = dense_block_or_operator(
            log_sigma_design,
            log_sigma_design.nrows(),
            log_sigma_design.ncols(),
            planned[1],
            &self.policy,
        );
        Ok((xmu, x_ls))
    }

    pub(crate) fn exact_joint_block_designs<'a>(
        &'a self,
        specs: Option<&'a [ParameterBlockSpec]>,
    ) -> Result<Option<(DenseOrOperator<'a>, DenseOrOperator<'a>)>, String> {
        if self.exact_joint_supported() {
            return self.exact_block_designs().map(Some);
        }
        if let Some(specs) = specs {
            return self.exact_block_designs_fromspecs(specs).map(Some);
        }
        Ok(None)
    }

    pub(crate) fn exact_joint_dense_block_designs<'a>(
        &'a self,
        specs: Option<&'a [ParameterBlockSpec]>,
    ) -> Result<Option<(Cow<'a, Array2<f64>>, Cow<'a, Array2<f64>>)>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_block_designs(specs)? else {
            return Ok(None);
        };
        let xmu = match xmu {
            DenseOrOperator::Borrowed(dense) => Cow::Borrowed(dense),
            DenseOrOperator::Owned(dense) => Cow::Owned(dense),
            DenseOrOperator::Operator(_) => {
                return Err(
                    "GaussianLocationScaleFamily exact psi path requires chunked operator support for oversized designs"
                        .to_string(),
                );
            }
        };
        let x_ls = match x_ls {
            DenseOrOperator::Borrowed(dense) => Cow::Borrowed(dense),
            DenseOrOperator::Owned(dense) => Cow::Owned(dense),
            DenseOrOperator::Operator(_) => {
                return Err(
                    "GaussianLocationScaleFamily exact psi path requires chunked operator support for oversized designs"
                        .to_string(),
                );
            }
        };
        Ok(Some((xmu, x_ls)))
    }

    pub(crate) fn exact_newton_joint_hessian_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: Option<&[ParameterBlockSpec]>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_block_designs(specs)? else {
            return Ok(None);
        };
        self.exact_newton_joint_hessian_from_designs(block_states, &xmu, &x_ls)
    }

    pub(crate) fn exact_newton_joint_hessian_directional_derivative_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: Option<&[ParameterBlockSpec]>,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_block_designs(specs)? else {
            return Ok(None);
        };
        self.exact_newton_joint_hessian_directional_derivative_from_designs(
            block_states,
            &xmu,
            &x_ls,
            d_beta_flat,
        )
    }

    pub(crate) fn exact_newton_joint_hessian_second_directional_derivative_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: Option<&[ParameterBlockSpec]>,
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_block_designs(specs)? else {
            return Ok(None);
        };
        self.exact_newton_joint_hessiansecond_directional_derivative_from_designs(
            block_states,
            &xmu,
            &x_ls,
            d_beta_u_flat,
            d_betav_flat,
        )
    }

    pub(crate) fn exact_newton_joint_psi_terms_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_psi_terms_from_designs(
            block_states,
            specs,
            derivative_blocks,
            psi_index,
            &xmu,
            &x_ls,
        )
    }

    pub(crate) fn exact_newton_joint_psisecond_order_terms_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_psisecond_order_terms_from_designs(
            block_states,
            derivative_blocks,
            psi_i,
            psi_j,
            &xmu,
            &x_ls,
        )
    }

    pub(crate) fn exact_newton_joint_psihessian_directional_derivative_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_psihessian_directional_derivative_from_designs(
            block_states,
            derivative_blocks,
            psi_index,
            d_beta_flat,
            &xmu,
            &x_ls,
        )
    }

    pub(crate) fn exact_newton_joint_hessian_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        xmu: &DenseOrOperator<'_>,
        x_ls: &DenseOrOperator<'_>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let etamu = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if etamu.len() != n || eta_ls.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleFamily input size mismatch".to_string(),
            }
            .into());
        }

        let rows = self.get_or_compute_row_scalars(etamu, eta_ls)?;
        // Block-diagonal Gaussian Fisher curvature (μ ⊥ σ ⇒ cross = 0, #684;
        // (ls,ls) = 2κ²a, #566), built from the shared single-source-of-truth
        // constructor so this dense path and the matrix-free workspace can never
        // disagree on the cross block. See `gaussian_locscale_fisher_joint_row_coeffs`.
        let (mm, cross, scale) = gaussian_locscale_fisher_joint_row_coeffs(&rows);
        Ok(Some(gaussian_joint_hessian_from_designs(
            xmu, x_ls, &mm, &cross, &scale,
        )?))
    }

    pub(crate) fn exact_newton_joint_hessian_directional_derivative_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        xmu: &DenseOrOperator<'_>,
        x_ls: &DenseOrOperator<'_>,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let etamu = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if etamu.len() != n || eta_ls.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleFamily input size mismatch".to_string(),
            }
            .into());
        }

        let pmu = xmu.ncols();
        let p_ls = x_ls.ncols();
        let total = pmu + p_ls;
        if d_beta_flat.len() != total {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleFamily joint d_beta length mismatch: got {}, expected {}",
                    d_beta_flat.len(),
                    total
                ),
            }
            .into());
        }
        let ximu = xmu.dot(d_beta_flat.slice(s![0..pmu]));
        let xi_ls = x_ls.dot(d_beta_flat.slice(s![pmu..pmu + p_ls]));
        let rows = self.get_or_compute_row_scalars(etamu, eta_ls)?;
        let directional = gaussian_joint_first_directionalweights(&rows, &ximu, &xi_ls);
        let dhmumu = directional.0;
        let dh_ls_ls = directional.2;
        // Fisher cross block E[H_{μ,ls}] ≡ 0 (μ ⊥ σ; see
        // exact_newton_joint_hessian_from_designs / #684), so its directional
        // derivative is identically 0 — keep the Hessian's curvature object the
        // block-diagonal Gaussian Fisher information at every order. The
        // observed-cross directional weight (`directional.1`) is therefore not
        // assembled.
        let dhmu_ls = Array1::<f64>::zeros(dhmumu.len());

        Ok(Some(gaussian_joint_hessian_from_designs(
            xmu, x_ls, &dhmumu, &dhmu_ls, &dh_ls_ls,
        )?))
    }

    pub(crate) fn exact_newton_joint_hessiansecond_directional_derivative_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        xmu: &DenseOrOperator<'_>,
        x_ls: &DenseOrOperator<'_>,
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let etamu = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if etamu.len() != n || eta_ls.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleFamily input size mismatch".to_string(),
            }
            .into());
        }

        let pmu = xmu.ncols();
        let p_ls = x_ls.ncols();
        let total = pmu + p_ls;
        if d_beta_u_flat.len() != total || d_betav_flat.len() != total {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "GaussianLocationScaleFamily joint second directional derivative length mismatch: got {} and {}, expected {}",
                d_beta_u_flat.len(),
                d_betav_flat.len(),
                total
            ) }.into());
        }
        let ximu_u = xmu.dot(d_beta_u_flat.slice(s![0..pmu]));
        let xi_ls_u = x_ls.dot(d_beta_u_flat.slice(s![pmu..pmu + p_ls]));
        let ximuv = xmu.dot(d_betav_flat.slice(s![0..pmu]));
        let xi_lsv = x_ls.dot(d_betav_flat.slice(s![pmu..pmu + p_ls]));
        let rows = self.get_or_compute_row_scalars(etamu, eta_ls)?;
        let second =
            gaussian_jointsecond_directionalweights(&rows, &ximu_u, &xi_ls_u, &ximuv, &xi_lsv);
        let d2hmumu = second.0;
        let d2h_ls_ls = second.2;
        // Fisher cross block E[H_{μ,ls}] ≡ 0 (μ ⊥ σ; #684), so its second
        // directional derivative is identically 0; `second.1` (observed) is not
        // assembled, keeping the curvature object block-diagonal Fisher.
        let d2hmu_ls = Array1::<f64>::zeros(d2hmumu.len());

        Ok(Some(gaussian_joint_hessian_from_designs(
            xmu, x_ls, &d2hmumu, &d2hmu_ls, &d2h_ls_ls,
        )?))
    }

    pub(crate) fn exact_newton_joint_psi_direction(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
        policy: &crate::resource::ResourcePolicy,
    ) -> Result<Option<LocationScaleJointPsiDirection>, String> {
        let Some(parts) = locscale_joint_psi_direction_parts(
            block_states,
            derivative_blocks,
            psi_index,
            self.y.len(),
            xmu.ncols(),
            x_ls.ncols(),
            Self::BLOCK_MU,
            Self::BLOCK_LOG_SIGMA,
            2,
            "GaussianLocationScaleFamily",
            "mu",
            policy,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(LocationScaleJointPsiDirection {
            block_idx: parts.block_idx,
            local_idx: parts.local_idx,
            z_primary_psi: parts.primary_z,
            z_ls_psi: parts.log_sigma_z,
            x_primary_psi: parts.primary_psi,
            x_ls_psi: parts.log_sigma_psi,
        }))
    }

    pub(crate) fn exact_newton_joint_psisecond_design_drifts(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_a: &LocationScaleJointPsiDirection,
        psi_b: &LocationScaleJointPsiDirection,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<LocationScaleJointPsiSecondDrifts, String> {
        locscale_joint_psisecond_design_drifts(
            block_states,
            derivative_blocks,
            psi_a,
            psi_b,
            LocScalePsiDriftConfig {
                n: self.y.len(),
                p_primary: xmu.ncols(),
                p_log_sigma: x_ls.ncols(),
                primary_block_idx: Self::BLOCK_MU,
                log_sigma_block_idx: Self::BLOCK_LOG_SIGMA,
                family_name: "GaussianLocationScaleFamily",
                primary_label: "mu",
                policy: &self.policy,
            },
        )
    }

    pub(crate) fn exact_newton_joint_psi_terms_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        if specs.len() != 2 || derivative_blocks.len() != 2 {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "GaussianLocationScaleFamily joint psi terms expect 2 specs and 2 derivative blocks, got {} and {}",
                specs.len(),
                derivative_blocks.len()
            ) }.into());
        }
        let Some(dir_a) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_index,
            xmu,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        // Gaussian 2-block location-scale family in the unified flattened
        // coefficient space beta = [betamu; beta_sigma]:
        //
        //   mu_i = z_i^T betamu,
        //   ell_i = x_i^T beta_sigma,
        //   s_i = exp(ell_i),
        //   r_i = y_i - mu_i,
        //   q_i = r_i / s_i,
        //   w_i = s_i^{-2},
        //   alpha_i = r_i s_i^{-2},
        //   b_i = q_i^2.
        //
        // The first fixed-beta psi object returned here is likelihood-only:
        //
        //   D_a         = -alpha^T m_a + (1 - b)^T ell_a
        //   D_{beta a}  = [ -Xmu^T alpha_a - X_{mu,a}^T alpha ;
        //                   -X_sigma^T b_a + X_{sigma,a}^T (1-b) ]
        //   D_{bb a}    = [ Xmu^T W_a Xmu + X_{mu,a}^T W Xmu + Xmu^T W X_{mu,a},
        //                   2( Xmu^T A_a X_sigma + X_{mu,a}^T A X_sigma + Xmu^T A X_{sigma,a} );
        //                   sym,
        //                   2( X_sigma^T B_a X_sigma + X_{sigma,a}^T B X_sigma + X_sigma^T B X_{sigma,a} ) ]
        //
        // with m_a = X_{mu,a} betamu, ell_a = X_{sigma,a} beta_sigma and
        // rowwise scalar drifts
        //
        //   w_a     = -2 w * ell_a
        //   alpha_a = -w * m_a - 2 alpha * ell_a
        //   b_a     = -2 alpha * m_a - 2 b * ell_a.
        //
        // Generic code in custom_family.rs promotes these likelihood-only
        // objects to the full fixed-beta V_a / g_a / H_a by adding S_a.
        let etamu = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let rows = self.get_or_compute_row_scalars(etamu, eta_ls)?;
        let weights_a =
            gaussian_joint_psi_firstweights(&rows, &dir_a.z_primary_psi, &dir_a.z_ls_psi);
        let objective_psi = weights_a.objective_psirow.sum();
        let xmu_map = dir_a.x_primary_psi.as_linear_map_ref();
        let x_ls_map = dir_a.x_ls_psi.as_linear_map_ref();
        let score_mu =
            xmu_map.transpose_mul(weights_a.scoremu.view()) + fast_atv(xmu, &weights_a.dscoremu);
        let score_ls = x_ls_map.transpose_mul(weights_a.score_ls.view())
            + fast_atv(x_ls, &weights_a.dscore_ls);
        let score_psi = gaussian_pack_joint_score(&score_mu, &score_ls);
        let hessian_psi_operator = build_two_block_custom_family_joint_psi_operator_from_actions(
            dir_a.x_primary_psi.cloned_first_action(),
            dir_a.x_ls_psi.cloned_first_action(),
            0..xmu.ncols(),
            xmu.ncols()..xmu.ncols() + x_ls.ncols(),
            xmu,
            x_ls,
            &weights_a.hmumu,
            &weights_a.hmu_ls,
            &weights_a.h_ls_ls,
            &weights_a.dhmumu,
            &weights_a.dhmu_ls,
            &weights_a.dh_ls_ls,
        )?;
        let hessian_psi = if hessian_psi_operator.is_some() {
            Array2::zeros((0, 0))
        } else {
            gaussian_joint_psihessian_fromweights(xmu, x_ls, xmu_map, x_ls_map, &weights_a)?
        };

        Ok(Some(crate::custom_family::ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi,
            hessian_psi_operator,
        }))
    }

    pub(crate) fn exact_newton_joint_psisecond_order_terms_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String> {
        let Some(dir_i) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_i,
            xmu,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        let Some(dir_j) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_j,
            xmu,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(
            self.exact_newton_joint_psisecond_order_terms_from_parts(
                block_states,
                derivative_blocks,
                &dir_i,
                &dir_j,
                xmu,
                x_ls,
                None,
            )?,
        ))
    }

    pub(crate) fn exact_newton_joint_psisecond_order_terms_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        dir_i: &LocationScaleJointPsiDirection,
        dir_j: &LocationScaleJointPsiDirection,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
        subsample: Option<&[crate::families::marginal_slope_shared::WeightedOuterRow]>,
    ) -> Result<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms, String> {
        let second_drifts = self.exact_newton_joint_psisecond_design_drifts(
            block_states,
            derivative_blocks,
            dir_i,
            dir_j,
            xmu,
            x_ls,
        )?;
        let n = self.y.len();
        let xmu_i_map = dir_i.x_primary_psi.as_linear_map_ref();
        let x_ls_i_map = dir_i.x_ls_psi.as_linear_map_ref();
        let xmu_j_map = dir_j.x_primary_psi.as_linear_map_ref();
        let x_ls_j_map = dir_j.x_ls_psi.as_linear_map_ref();
        let xmu_ab_map = second_psi_linear_map(
            second_drifts.x_primary_ab_action.as_ref(),
            second_drifts.x_primary_ab.as_ref(),
            n,
            xmu.ncols(),
        );
        let x_ls_ab_map = second_psi_linear_map(
            second_drifts.x_ls_ab_action.as_ref(),
            second_drifts.x_ls_ab.as_ref(),
            n,
            x_ls.ncols(),
        );
        // Second fixed-beta psi objects for the same Gaussian location-scale
        // kernel. Using the notation from the first-order comment, the rowwise
        // second psi drifts are
        //
        //   w_ab     = 4 w * ell_a * ell_b - 2 w * ell_ab
        //   alpha_ab = 2 w * (m_a * ell_b + m_b * ell_a)
        //              + 4 alpha * ell_a * ell_b
        //              - w * m_ab
        //              - 2 alpha * ell_ab
        //   b_ab     = 2 w * m_a * m_b
        //              + 4 alpha * (m_a * ell_b + m_b * ell_a)
        //              + 4 b * ell_a * ell_b
        //              - 2 alpha * m_ab
        //              - 2 b * ell_ab.
        //
        // The exact likelihood-only second-order objects are then:
        //
        //   D_ab,
        //   D_{beta ab},
        //   D_{beta beta ab},
        //
        // assembled from the usual product-rule expansion over realized
        // design motion X_{.,a}, X_{.,b}, X_{.,ab}. Generic code adds S_ab.
        let etamu = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let rows = self.get_or_compute_row_scalars(etamu, eta_ls)?;
        let mut weights_i =
            gaussian_joint_psi_firstweights(&rows, &dir_i.z_primary_psi, &dir_i.z_ls_psi);
        let mut weights_j =
            gaussian_joint_psi_firstweights(&rows, &dir_j.z_primary_psi, &dir_j.z_ls_psi);
        let mut secondweights = gaussian_joint_psisecondweights(
            &rows,
            &dir_i.z_primary_psi,
            &dir_i.z_ls_psi,
            &dir_j.z_primary_psi,
            &dir_j.z_ls_psi,
            &second_drifts.z_primary_ab,
            &second_drifts.z_ls_ab,
        );
        if let Some(sub_rows) = subsample {
            // HT mask: every downstream consumer (gaussian_joint_psisecondhessian_fromweights,
            // weighted_crossprod_psi_maps with weights_*.{hmumu,hmu_ls,h_ls_ls},
            // fast_atv on d2score_* and dscore_*) is row-linear in these arrays, so
            // scaling sampled rows by 1/π_i and zeroing the rest yields an unbiased
            // estimator of the full-data second-order ψ Hessian and ψ score.
            apply_ht_mask_first(&mut weights_i, sub_rows);
            apply_ht_mask_first(&mut weights_j, sub_rows);
            apply_ht_mask_second(&mut secondweights, sub_rows);
        }
        let objective_psi_psi = secondweights.objective_psi_psirow.sum();

        let score_psi_psi = gaussian_pack_joint_score(
            &(xmu_ab_map.transpose_mul(weights_i.scoremu.view())
                + xmu_i_map.transpose_mul(weights_j.dscoremu.view())
                + xmu_j_map.transpose_mul(weights_i.dscoremu.view())
                + fast_atv(xmu, &secondweights.d2scoremu)),
            &(x_ls_ab_map.transpose_mul(weights_i.score_ls.view())
                + x_ls_i_map.transpose_mul(weights_j.dscore_ls.view())
                + x_ls_j_map.transpose_mul(weights_i.dscore_ls.view())
                + fast_atv(x_ls, &secondweights.d2score_ls)),
        );
        let hessian_psi_psi = gaussian_joint_psisecondhessian_fromweights(
            xmu,
            x_ls,
            xmu_i_map,
            x_ls_i_map,
            xmu_j_map,
            x_ls_j_map,
            xmu_ab_map,
            x_ls_ab_map,
            &weights_i,
            &weights_j,
            &secondweights,
        )?;

        Ok(crate::custom_family::ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi,
            hessian_psi_psi,
            hessian_psi_psi_operator: None,
        })
    }

    pub(crate) fn exact_newton_joint_psihessian_directional_derivative_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some(dir_a) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_index,
            xmu,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(
            self.exact_newton_joint_psihessian_directional_derivative_from_parts(
                block_states,
                &dir_a,
                d_beta_flat,
                xmu,
                x_ls,
                None,
            )?,
        ))
    }

    pub(crate) fn exact_newton_joint_psihessian_directional_derivative_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        dir_a: &LocationScaleJointPsiDirection,
        d_beta_flat: &Array1<f64>,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
        subsample: Option<&[crate::families::marginal_slope_shared::WeightedOuterRow]>,
    ) -> Result<Array2<f64>, String> {
        let etamu = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let pmu = xmu.ncols();
        let p_ls = x_ls.ncols();
        let xmu_map = dir_a.x_primary_psi.as_linear_map_ref();
        let x_ls_map = dir_a.x_ls_psi.as_linear_map_ref();
        let total = pmu + p_ls;
        if d_beta_flat.len() != total {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "GaussianLocationScaleFamily joint psi hessian directional derivative length mismatch: got {}, expected {}",
                d_beta_flat.len(),
                total
            ) }.into());
        }
        // Only the log-σ–channel direction enters the surviving Fisher blocks
        // of the mixed drift (the μ-channel direction fed the observed cross
        // block, now Fisher 0; μ ⊥ σ, #684).
        let u_ls = d_beta_flat.slice(s![pmu..pmu + p_ls]);
        let xi_ls = fast_av(x_ls, &u_ls);
        let uza_ls = x_ls_map.forward_mul(u_ls);
        // Mixed drift T_a[u] = D_beta H_a^{(D)}[u] for the Gaussian family.
        //
        // Along u = [umu; u_sigma], define xi = Xmu umu and zeta = X_sigma u_sigma.
        // The first beta-directional drifts of the Gaussian row scalars are
        //
        //   d_u w     = -2 w * zeta
        //   d_u alpha = -w * xi - 2 alpha * zeta
        //   d_u b     = -2 alpha * xi - 2 b * zeta.
        //
        // Differentiating the psi-a scalar drifts once more gives
        //
        //   d_u w_a     = 4 w * ell_a * zeta - 2 w * zeta_a
        //   d_u alpha_a = 2 w * (m_a * zeta + ell_a * xi)
        //                 - w * xi_a
        //                 + 4 alpha * ell_a * zeta
        //                 - 2 alpha * zeta_a
        //   d_u b_a     = 2 w * m_a * xi
        //                 + 4 alpha * (m_a * zeta + ell_a * xi)
        //                 + 4 b * ell_a * zeta
        //                 - 2 alpha * xi_a
        //                 - 2 b * zeta_a.
        //
        // The matrix drift returned here is the exact likelihood-only
        //
        //   T_a[u] = D_beta H_{psi_a}^{(D)}[u],
        //
        // assembled blockwise as
        //
        //   Kmumu,a[u]   = Xmu^T W_a[u] Xmu
        //                   + X_{mu,a}^T W[u] Xmu
        //                   + Xmu^T W[u] X_{mu,a}
        //   Kmusigma,a[u]= 2( Xmu^T A_a[u] X_sigma
        //                   + X_{mu,a}^T A[u] X_sigma
        //                   + Xmu^T A[u] X_{sigma,a} )
        //   K_sigmasigma,a[u]
        //                   = 2( X_sigma^T B_a[u] X_sigma
        //                   + X_{sigma,a}^T B[u] X_sigma
        //                   + X_sigma^T B[u] X_{sigma,a} ).
        //
        // Generic code then combines this with S(theta)-motion and the profile
        // mode responses to form ddot H_{ij}.
        let rows = self.get_or_compute_row_scalars(etamu, eta_ls)?;
        let mut mixedweights =
            gaussian_joint_psi_mixed_driftweights(&rows, &xi_ls, &dir_a.z_ls_psi, &uza_ls);
        if let Some(sub_rows) = subsample {
            // HT mask: `gaussian_joint_psi_mixedhessian_drift_fromweights` is
            // row-linear in every `mixedweights.*` array via `xt_diag_*_dense`
            // and `weighted_crossprod_psi_maps`, so the masked Hessian-drift
            // remains an unbiased estimator of the full-data drift.
            apply_ht_mask_mixed(&mut mixedweights, sub_rows);
        }

        gaussian_joint_psi_mixedhessian_drift_fromweights(
            xmu,
            x_ls,
            xmu_map,
            x_ls_map,
            &mixedweights,
        )
    }

    /// Build the [`BlockEffectiveJacobian`] for block `block_idx` given the
    /// realised block specs.  Returns an [`AdditiveBlockJacobian`] encoding the
    /// linear map η_r[i] = X_r[i,:] · β_r:
    ///
    /// - block 0 (mu):       output 0 = design rows, output 1 = zeros
    /// - block 1 (log_sigma): output 0 = zeros, output 1 = design rows
    pub fn block_effective_jacobian(
        specs: &[ParameterBlockSpec],
        block_idx: usize,
    ) -> Result<Box<dyn BlockEffectiveJacobian>, String> {
        crate::util::block_jacobian::AdditiveWiggleBlockLayout {
            family: "GaussianLocationScaleFamily",
            n_outputs: 2,
            additive_blocks: &[Self::BLOCK_MU, Self::BLOCK_LOG_SIGMA],
            wiggle_block: None,
        }
        .block_effective_jacobian(specs, block_idx)
    }
}

/// Per-subject 2×2 channel Hessian `W_i` for Gaussian location-scale.
///
/// The row negative log-likelihood (with per-row weight `w_i`, response `y_i`,
/// mean predictor `μ_i`, log-scale predictor `s_i = log σ_i`) is
///
/// ```text
/// ρ_i(μ, s) = w_i [s + 0.5·(y_i − μ)²·exp(−2s)]
/// ```
///
/// The 2×2 Hessian in `(μ, s)` coordinates:
///
/// ```text
/// W_i[0,0] = w_i · exp(−2 s_i)                        ∂²ρ/∂μ²
/// W_i[1,1] = w_i · 2·(y_i − μ_i)²·exp(−2 s_i)        ∂²ρ/∂s²
/// W_i[0,1] = W_i[1,0] = w_i · 2·(y_i − μ_i)·exp(−2 s_i)  ∂²ρ/∂μ∂s
/// ```
///
/// The off-diagonal cross-channel term `∂²ρ/∂μ∂s` is nonzero whenever the
/// residual `(y_i − μ_i) ≠ 0`, i.e. away from the fitted mean.
pub struct GaussianLocationScaleChannelHessian {
    /// Row-major `(n × 2 × 2)` PSD-clamped per-subject Hessian.
    h: ndarray::Array3<f64>,
}

impl GaussianLocationScaleChannelHessian {
    /// Construct the raw (un-PSD-clamped) per-subject observed Hessian.
    ///
    /// For Gaussian location-scale the 2×2 observed Hessian
    /// `[[w·e^{-2s}, 2·w·r·e^{-2s}], [2·w·r·e^{-2s}, 2·w·r²·e^{-2s}]]`
    /// has determinant `-2·w²·r²·e^{-4s}` which is non-positive whenever
    /// the residual `r = y − μ ≠ 0`. Tests that finite-difference the row
    /// NLL must compare against this raw observed Hessian — PSD clamping
    /// alters the eigenvalues and the FD-versus-closed-form match fails.
    ///
    /// Production code that needs a PSD matrix (e.g. the canonicalize gate)
    /// must call [`Self::from_pilot`] which PSD-clamps via 2×2
    /// eigendecomposition.
    pub fn from_pilot_observed_unclamped(
        y: &ndarray::Array1<f64>,
        w: &ndarray::Array1<f64>,
        eta_mu: &ndarray::Array1<f64>,
        eta_log_sigma: &ndarray::Array1<f64>,
    ) -> Result<Self, String> {
        let n = y.len();
        if w.len() != n || eta_mu.len() != n || eta_log_sigma.len() != n {
            return Err(format!(
                "GaussianLocationScaleChannelHessian::from_pilot_observed_unclamped: \
                 length mismatch y={n} w={} eta_mu={} eta_log_sigma={}",
                w.len(),
                eta_mu.len(),
                eta_log_sigma.len(),
            ));
        }
        let mut h = ndarray::Array3::<f64>::zeros((n, 2, 2));
        for i in 0..n {
            let wi = w[i];
            let mu_i = eta_mu[i];
            let s_i = eta_log_sigma[i];
            let inv_sigma2 = (-2.0 * s_i).exp();
            let resid = y[i] - mu_i;
            h[[i, 0, 0]] = wi * inv_sigma2;
            h[[i, 1, 1]] = wi * 2.0 * resid * resid * inv_sigma2;
            h[[i, 0, 1]] = wi * 2.0 * resid * inv_sigma2;
            h[[i, 1, 0]] = h[[i, 0, 1]];
        }
        Ok(Self { h })
    }

    /// Construct from pilot predictors (μ and log σ at current β) and data,
    /// with PSD eigenvalue clamping applied per subject.
    ///
    /// `y` is the response, `w` the per-row sample weights, `eta_mu` and
    /// `eta_log_sigma` the current linear predictors. Negative eigenvalues
    /// are projected to zero (PSD clamp) before storage so the resulting
    /// matrix is a valid metric for the W-Gram identifiability compile.
    pub fn from_pilot(
        y: &ndarray::Array1<f64>,
        w: &ndarray::Array1<f64>,
        eta_mu: &ndarray::Array1<f64>,
        eta_log_sigma: &ndarray::Array1<f64>,
    ) -> Result<Self, String> {
        let n = y.len();
        if w.len() != n || eta_mu.len() != n || eta_log_sigma.len() != n {
            return Err(format!(
                "GaussianLocationScaleChannelHessian::from_pilot: \
                 length mismatch y={n} w={} eta_mu={} eta_log_sigma={}",
                w.len(),
                eta_mu.len(),
                eta_log_sigma.len(),
            ));
        }
        let mut h = ndarray::Array3::<f64>::zeros((n, 2, 2));
        for i in 0..n {
            let wi = w[i];
            let mu_i = eta_mu[i];
            let s_i = eta_log_sigma[i];
            let inv_sigma2 = (-2.0 * s_i).exp(); // exp(-2s) = 1/sigma^2
            let resid = y[i] - mu_i;
            // Hessian of w_i * ρ_i
            let h00 = wi * inv_sigma2;
            let h11 = wi * 2.0 * resid * resid * inv_sigma2;
            let h01 = wi * 2.0 * resid * inv_sigma2;
            // PSD clamp via eigendecomposition of 2×2 matrix.
            // psd_clamp_2x2 returns (λ1, λ2, u1[0], u1[1], u2[0], u2[1])
            // where u1 and u2 are unit eigenvectors for λ1 and λ2.
            // Reconstruction: H_psd = λ1·u1·u1ᵀ + λ2·u2·u2ᵀ
            let (e0, e1, u1_0, u1_1, u2_0, u2_1) = psd_clamp_2x2(h00, h01, h11);
            h[[i, 0, 0]] = e0 * u1_0 * u1_0 + e1 * u2_0 * u2_0;
            h[[i, 0, 1]] = e0 * u1_0 * u1_1 + e1 * u2_0 * u2_1;
            h[[i, 1, 0]] = h[[i, 0, 1]];
            h[[i, 1, 1]] = e0 * u1_1 * u1_1 + e1 * u2_1 * u2_1;
        }
        Ok(Self { h })
    }
}

impl FamilyChannelHessian for GaussianLocationScaleChannelHessian {
    fn n_outputs(&self) -> usize {
        2
    }

    fn n_subjects(&self) -> usize {
        self.h.shape()[0]
    }

    fn fill_subject(&self, i: usize, out: &mut [f64]) {
        assert_eq!(out.len(), 4);
        out[0] = self.h[[i, 0, 0]];
        out[1] = self.h[[i, 0, 1]];
        out[2] = self.h[[i, 1, 0]];
        out[3] = self.h[[i, 1, 1]];
    }

    fn evaluate_full(&self) -> ndarray::Array3<f64> {
        self.h.clone()
    }
}

impl CustomFamily for GaussianLocationScaleFamily {
    /// The Gaussian location-scale joint Hessian depends on β because the
    /// cross-block (μ,log σ) and (log σ,log σ) blocks contain the residual
    /// r = y − μ (via the row scalars m = r·w and n = r²·w), which changes
    /// when β_μ moves.  The (μ,μ) block weight w = 1/σ² also depends on
    /// β_{log σ}.  This override is essential for correct M_j[u] drift
    /// corrections when ψ hyperparameters move the design matrices.
    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    /// Two independent linear predictors: block 0 → μ channel, block 1 → log σ
    /// channel. Declaring the channel topology lets `fit_custom_family` route
    /// the identifiability audit channel-aware even when a caller builds the
    /// blocks by hand (without `build_location_scale_block`'s callbacks), so a
    /// shared μ/log-σ covariate basis is recognised as block-diagonal rather
    /// than mistaken for cross-block intercept aliases (#558).
    fn output_channel_assignment(
        &self,
        specs: &[ParameterBlockSpec],
    ) -> Option<Vec<usize>> {
        // Two-channel families: `[mu, log_sigma]`. The optional trailing
        // zero-channel wiggle block (when present) also drives channel 0.
        Some(
            (0..specs.len())
                .map(|i| usize::from(i == Self::BLOCK_LOG_SIGMA))
                .collect(),
        )
    }

    fn coefficient_hessian_cost(&self, specs: &[ParameterBlockSpec]) -> u64 {
        // Operator-aware: when the unified evaluator picks the matrix-free
        // joint Hessian path (see `use_joint_matrix_free_path`), the workspace
        // applies the joint Hessian via row-streaming Khatri-Rao matvecs at
        // O(n · (p_t + p_ℓ)) per Hv, never building the dense (p_t + p_ℓ)²
        // matrix. Report the operator work model so diagnostics and
        // first-order-only policies reflect the representation that actually
        // runs.
        crate::families::location_scale_engine::location_scale_coefficient_hessian_cost(
            self.y.len() as u64,
            specs,
        )
    }

    fn evaluate(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<FamilyEvaluation, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let etamu = &block_states[Self::BLOCK_MU].eta;
        let eta_log_sigma = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if etamu.len() != n || eta_log_sigma.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleFamily input size mismatch".to_string(),
            }
            .into());
        }

        // Diagonal IRLS weights for the inner solver.
        //
        // For the location block (identity link): wmu = pw / sigma^2. Since the
        // location link is identity, observed = Fisher --- no correction needed.
        //
        // For the log-sigma block (log link): w_ls = 2 * pw * (dsigma/deta)^2 / sigma^2.
        // This is the Fisher weight. For the outer REML, the joint
        // `exact_newton_joint_hessian` provides the full observed Hessian directly,
        // so these Diagonal weights are only used for the inner IRLS iteration
        // (where Fisher scoring is fine). See response.md Section 3.
        //
        let mut zmu = Array1::<f64>::zeros(n);
        let mut wmu = Array1::<f64>::zeros(n);
        let mut z_ls = Array1::<f64>::zeros(n);
        let mut w_ls = Array1::<f64>::zeros(n);
        let ln2pi = (2.0 * std::f64::consts::PI).ln();
        let mut ll = 0.0;

        const CHUNK: usize = 1024;
        if let (
            Some(y_s),
            Some(w_s),
            Some(mu_s),
            Some(ls_s),
            Some(zmu_s),
            Some(wmu_s),
            Some(zls_s),
            Some(wls_s),
        ) = (
            self.y.as_slice_memory_order(),
            self.weights.as_slice_memory_order(),
            etamu.as_slice_memory_order(),
            eta_log_sigma.as_slice_memory_order(),
            zmu.as_slice_memory_order_mut(),
            wmu.as_slice_memory_order_mut(),
            z_ls.as_slice_memory_order_mut(),
            w_ls.as_slice_memory_order_mut(),
        ) {
            // Per-row Gaussian LS kernel writes 4 working arrays directly into
            // the output slices; ll is reduced via Rayon's sum. Independent
            // across rows.
            ll += zmu_s
                .par_chunks_mut(CHUNK)
                .zip(wmu_s.par_chunks_mut(CHUNK))
                .zip(zls_s.par_chunks_mut(CHUNK))
                .zip(wls_s.par_chunks_mut(CHUNK))
                .enumerate()
                .map(|(chunk_idx, (((zmu_c, wmu_c), zls_c), wls_c))| {
                    let start = chunk_idx * CHUNK;
                    let mut local_ll = 0.0;
                    for local in 0..zmu_c.len() {
                        let i = start + local;
                        let row =
                            gaussian_diagonal_row_kernel(y_s[i], mu_s[i], ls_s[i], w_s[i], ln2pi);
                        zmu_c[local] = mu_s[i] + row.location_working_shift;
                        wmu_c[local] = row.location_working_weight;
                        zls_c[local] = row.log_sigma_working_response;
                        wls_c[local] = row.log_sigma_working_weight;
                        local_ll += row.log_likelihood;
                    }
                    local_ll
                })
                .sum::<f64>();
        } else {
            // Fallback path: inputs are not contiguous. Outputs (just-allocated
            // Array1::zeros) always are. Reborrow input views into the closure.
            let y_view = self.y.view();
            let w_view = self.weights.view();
            let mu_view = etamu.view();
            let ls_view = eta_log_sigma.view();
            let zmu_s = zmu
                .as_slice_memory_order_mut()
                .expect("zeros is contiguous");
            let wmu_s = wmu
                .as_slice_memory_order_mut()
                .expect("zeros is contiguous");
            let zls_s = z_ls
                .as_slice_memory_order_mut()
                .expect("zeros is contiguous");
            let wls_s = w_ls
                .as_slice_memory_order_mut()
                .expect("zeros is contiguous");
            ll += zmu_s
                .par_chunks_mut(CHUNK)
                .zip(wmu_s.par_chunks_mut(CHUNK))
                .zip(zls_s.par_chunks_mut(CHUNK))
                .zip(wls_s.par_chunks_mut(CHUNK))
                .enumerate()
                .map(|(chunk_idx, (((zmu_c, wmu_c), zls_c), wls_c))| {
                    let start = chunk_idx * CHUNK;
                    let mut local_ll = 0.0;
                    for local in 0..zmu_c.len() {
                        let i = start + local;
                        let row = gaussian_diagonal_row_kernel(
                            y_view[i], mu_view[i], ls_view[i], w_view[i], ln2pi,
                        );
                        zmu_c[local] = mu_view[i] + row.location_working_shift;
                        wmu_c[local] = row.location_working_weight;
                        zls_c[local] = row.log_sigma_working_response;
                        wls_c[local] = row.log_sigma_working_weight;
                        local_ll += row.log_likelihood;
                    }
                    local_ll
                })
                .sum::<f64>();
        }

        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: vec![
                BlockWorkingSet::diagonal_checked(zmu, wmu)?,
                BlockWorkingSet::diagonal_checked(z_ls, w_ls)?,
            ],
        })
    }

    fn log_likelihood_only(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<f64, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let etamu = &block_states[Self::BLOCK_MU].eta;
        let eta_log_sigma = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if etamu.len() != n || eta_log_sigma.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleFamily input size mismatch".to_string(),
            }
            .into());
        }
        // logb noise link: σ(η_ls) = LOGB_SIGMA_FLOOR + exp(η_ls). σ ≥ b > 0
        // bounds the loglik below (−Σlog σ ≥ −n log b) and bounds 1/σ² by 1/b²,
        // so the previous `inv_s2.min(1e24)` cap is structurally unnecessary.
        let ln2pi = (2.0 * std::f64::consts::PI).ln();
        let mut ll = 0.0;
        if let (Some(y_s), Some(w_s), Some(mu_s), Some(ls_s)) = (
            self.y.as_slice_memory_order(),
            self.weights.as_slice_memory_order(),
            etamu.as_slice_memory_order(),
            eta_log_sigma.as_slice_memory_order(),
        ) {
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            ll += (0..n)
                .into_par_iter()
                .map(|i| {
                    let wi = w_s[i];
                    if wi == 0.0 {
                        return 0.0;
                    }
                    let sigma_i = logb_sigma_from_eta_scalar(ls_s[i]);
                    let inv_s2 = (sigma_i * sigma_i).recip();
                    let r = y_s[i] - mu_s[i];
                    wi * (-0.5 * (r * r * inv_s2 + ln2pi + 2.0 * sigma_i.ln()))
                })
                .sum::<f64>();
        } else {
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            ll += (0..n)
                .into_par_iter()
                .map(|i| {
                    let wi = self.weights[i];
                    if wi == 0.0 {
                        return 0.0;
                    }
                    let sigma_i = logb_sigma_from_eta_scalar(eta_log_sigma[i]);
                    let inv_s2 = (sigma_i * sigma_i).recip();
                    let r = self.y[i] - etamu[i];
                    wi * (-0.5 * (r * r * inv_s2 + ln2pi + 2.0 * sigma_i.ln()))
                })
                .sum::<f64>();
        }
        Ok(ll)
    }

    /// Outer-only log-likelihood with optional row subsample.
    ///
    /// When `options.outer_score_subsample` is `Some`, only the sampled rows
    /// contribute; each row's per-row log-likelihood term is multiplied by
    /// `WeightedOuterRow.weight`, the Horvitz–Thompson inverse-inclusion
    /// factor 1/π_i (uniform or stratified sampling both supported), so the
    /// partial sum is an unbiased estimator of the full-data log-likelihood.
    /// When `None`, this returns the full-data `log_likelihood_only`. Inner
    /// PIRLS line searches never install the subsample option, so they
    /// continue to score the exact full-data log-likelihood.
    fn log_likelihood_only_with_options(
        &self,
        block_states: &[ParameterBlockState],
        options: &BlockwiseFitOptions,
    ) -> Result<f64, String> {
        let Some(subsample) = options.outer_score_subsample.as_ref() else {
            return self.log_likelihood_only(block_states);
        };
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let etamu = &block_states[Self::BLOCK_MU].eta;
        let eta_log_sigma = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if etamu.len() != n || eta_log_sigma.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let ln2pi = (2.0 * std::f64::consts::PI).ln();
        use rayon::iter::ParallelIterator;
        let ll: f64 = subsample
            .rows
            .par_iter()
            .map(|row| {
                let i = row.index;
                let wi = self.weights[i];
                if wi == 0.0 {
                    return 0.0;
                }
                let sigma_i = logb_sigma_from_eta_scalar(eta_log_sigma[i]);
                let inv_s2 = (sigma_i * sigma_i).recip();
                let r = self.y[i] - etamu[i];
                row.weight * wi * (-0.5 * (r * r * inv_s2 + ln2pi + 2.0 * sigma_i.ln()))
            })
            .sum();
        Ok(ll)
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_for_specs(block_states, None)
    }

    fn has_explicit_joint_hessian(&self) -> bool {
        true
    }

    /// The Gaussian location-scale likelihood has no separation /
    /// under-identification regime that the full-span Jeffreys curvature `H_Φ`
    /// is meant to regularize: with the soft floor `σ ≥ b > 0` the per-row
    /// Fisher information `diag(a/σ², 2κ²a)` is bounded and `O(n)` on every
    /// identified direction at every working point, so the well-conditioned-`H`
    /// Jeffreys gate smooth-steps `H_Φ` to ~0 — yet the matching score `∇Φ`
    /// kept leaking a *phantom* penalized-stationarity residual into the inner
    /// joint-Newton (a nonzero `|∇L − Sβ|` paired with a numerically null `H_Φ`
    /// and a full-rank `H_pen`), so the KKT certificate refused every iterate
    /// and the outer REML rejected all seeds — aborting heteroscedastic
    /// location-scale fits (#684–#688). This is the same opt-out
    /// `TransformationNormalFamily` takes for the same structural reason
    /// (continuous response, `O(n)` Fisher information everywhere); it removes
    /// the phantom residual and drops the per-cycle `O(n·p²)` Jeffreys
    /// directional-derivative overhead.
    fn joint_jeffreys_term_required(&self) -> bool {
        false
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_directional_derivative_for_specs(
            block_states,
            None,
            d_beta_flat,
        )
    }

    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_second_directional_derivative_for_specs(
            block_states,
            None,
            d_beta_u_flat,
            d_betav_flat,
        )
    }

    fn diagonalworking_weights_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        d_eta: &Array1<f64>,
    ) -> Result<Option<Array1<f64>>, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != n || eta_ls.len() != n || self.weights.len() != n || d_eta.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleFamily input size mismatch".to_string(),
            }
            .into());
        }

        let sigma = eta_ls.mapv(logb_sigma_from_eta_scalar);
        let mut dw = Array1::<f64>::zeros(n);
        match block_idx {
            Self::BLOCK_MU => {
                // Gaussian location block:
                //
                //   wmu = weight / sigma^2.
                //
                // This depends only on the scale predictor, so along a
                // location-only direction d etamu the directional derivative is
                // identically zero.
                Ok(Some(dw))
            }
            Self::BLOCK_LOG_SIGMA => {
                // Gaussian log-sigma block:
                //
                // The PIRLS information weight is
                //
                //   w_ls = max(2 * weight * clamp(g, -1, 1)^2, MIN_WEIGHT),
                //   g    = sigma'(eta_ls) / sigma(eta_ls),
                // with the semantic rule that zero observation weights stay zero.
                //
                // Along a direction d eta_ls,
                //
                //   dw_ls is the directional derivative of that piecewise
                //   definition. On the active clamp branch or active MIN_WEIGHT
                //   floor branch, the returned derivative is zero to match the
                //   selected local piece of the evaluated weight.
                //
                // This is the exact directional derivative needed by the REML
                // trace term
                //
                //   0.5 tr(J^{-1} D_beta J[u])
                //   = 0.5 sum_i (x_i^T J^{-1} x_i) dw_i
                //
                // for diagonal working-set blocks.
                use rayon::iter::{IntoParallelIterator, ParallelIterator};
                let dw_vec: Vec<f64> = (0..n)
                    .into_par_iter()
                    .map(|i| {
                        let d1 = crate::families::sigma_link::logb_sigma_jet1_scalar(eta_ls[i]).d1;
                        gaussian_log_sigma_irlsinfo_directional_derivative(
                            self.weights[i],
                            sigma[i],
                            d1,
                            d_eta[i],
                        )
                    })
                    .collect();
                for (i, v) in dw_vec.into_iter().enumerate() {
                    dw[i] = v;
                }
                Ok(Some(dw))
            }
            _ => Ok(None),
        }
    }

    fn exact_newton_joint_hessian_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_for_specs(block_states, Some(specs))
    }

    fn exact_newton_joint_hessian_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_directional_derivative_for_specs(
            block_states,
            Some(specs),
            d_beta_flat,
        )
    }

    fn exact_newton_joint_hessian_second_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_second_directional_derivative_for_specs(
            block_states,
            Some(specs),
            d_beta_u_flat,
            d_betav_flat,
        )
    }

    fn exact_newton_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        self.exact_newton_joint_psi_terms_for_specs(
            block_states,
            specs,
            derivative_blocks,
            psi_index,
        )
    }

    fn exact_newton_joint_psisecond_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String> {
        self.exact_newton_joint_psisecond_order_terms_for_specs(
            block_states,
            specs,
            derivative_blocks,
            psi_i,
            psi_j,
        )
    }

    fn exact_newton_joint_psihessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_psihessian_directional_derivative_for_specs(
            block_states,
            specs,
            derivative_blocks,
            psi_index,
            d_beta_flat,
        )
    }

    fn exact_newton_joint_psi_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        if block_states.len() != 2 || specs.len() != 2 || derivative_blocks.len() != 2 {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "GaussianLocationScaleFamily joint psi workspace expects 2 states, 2 specs, and 2 derivative block lists, got {} / {} / {}",
                block_states.len(),
                specs.len(),
                derivative_blocks.len()
            ) }.into());
        }
        Ok(Some(Arc::new(
            GaussianLocationScaleExactNewtonJointPsiWorkspace::new(
                self.clone(),
                block_states.to_vec(),
                specs,
                derivative_blocks.to_vec(),
            )?,
        )))
    }

    /// Outer-aware joint ψ workspace with optional row subsample.
    ///
    /// When `options.outer_score_subsample` is `None`, this is byte-identical
    /// to `exact_newton_joint_psi_workspace`. When `Some`, the subsample is
    /// stored in the workspace and forwarded into every per-row weight array
    /// produced by `gaussian_joint_psi_firstweights`,
    /// `gaussian_joint_psisecondweights`, and
    /// `gaussian_joint_psi_mixed_driftweights`: each sampled row's
    /// contribution is multiplied by `WeightedOuterRow.weight = 1/π_i` and
    /// non-sampled rows are zeroed. Every downstream assembly
    /// (`gaussian_joint_psi*_fromweights`, `weighted_crossprod_psi_maps`,
    /// `xt_diag_*_dense`,
    /// `build_two_block_custom_family_joint_psi_operator_from_actions`) is
    /// row-linear in these arrays via `Xᵀ diag(W) Y`, so the resulting
    /// second-order ψ Hessian and ψ-Hessian directional derivative are
    /// unbiased Horvitz–Thompson estimators of the full-data quantities.
    /// Inner-PIRLS and final-covariance paths never install the option.
    fn exact_newton_joint_psi_workspace_with_options(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        if block_states.len() != 2 || specs.len() != 2 || derivative_blocks.len() != 2 {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "GaussianLocationScaleFamily joint psi workspace expects 2 states, 2 specs, and 2 derivative block lists, got {} / {} / {}",
                block_states.len(),
                specs.len(),
                derivative_blocks.len()
            ) }.into());
        }
        Ok(Some(Arc::new(
            GaussianLocationScaleExactNewtonJointPsiWorkspace::new_with_subsample(
                self.clone(),
                block_states.to_vec(),
                specs,
                derivative_blocks.to_vec(),
                options.outer_score_subsample.clone(),
            )?,
        )))
    }

    fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        let workspace = GaussianLocationScaleHessianWorkspace::new(
            self.clone(),
            block_states.to_vec(),
            xmu.into_owned(),
            x_ls.into_owned(),
        )?;
        Ok(Some(Arc::new(workspace)))
    }

    /// Outer-aware joint-Hessian workspace with optional row subsample.
    ///
    /// When `options.outer_score_subsample` is `None`, this is byte-identical
    /// to `exact_newton_joint_hessian_workspace`. When `Some`, the precomputed
    /// per-row coefficient arrays (`coeff_mm`, `coeff_ml`, `coeff_ll`) — which
    /// every downstream assembly (`hessian_dense`, `hessian_matvec`,
    /// `hessian_diagonal`) consumes row-linearly via `Xᵀ diag(W) X` — are
    /// replaced by a Horvitz–Thompson mask: each sampled row's coefficient is
    /// multiplied by `WeightedOuterRow.weight` (the inverse-inclusion factor
    /// 1/π_i; uniform or stratified sampling both supported), and non-sampled
    /// rows are zeroed. The resulting joint Hessian is an unbiased estimator
    /// of the full-data joint Hessian. Inner PIRLS never installs the option,
    /// so the inner solve continues to consume the exact full-data Hessian.
    fn exact_newton_joint_hessian_workspace_with_options(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        let mut workspace = GaussianLocationScaleHessianWorkspace::new(
            self.clone(),
            block_states.to_vec(),
            xmu.into_owned(),
            x_ls.into_owned(),
        )?;
        if let Some(subsample) = options.outer_score_subsample.as_ref() {
            workspace.apply_outer_subsample(subsample.rows.as_ref());
        }
        Ok(Some(Arc::new(workspace)))
    }

    fn inner_coefficient_hessian_hvp_available(
        &self,
        specs: &[ParameterBlockSpec],
    ) -> bool {
        // The Gaussian location-scale workspace is returned by
        // `exact_newton_joint_hessian_workspace` whenever
        // `exact_joint_dense_block_designs` succeeds, which itself depends on
        // both block designs being present. This is only a β-space operator
        // capability; outer θθ Hessian availability is declared separately.
        self.exact_joint_supported()
            && matches!(
                self.exact_joint_dense_block_designs(Some(specs)),
                Ok(Some(_))
            )
    }

    /// Outer-derivative policy: declare HT-subsample capability.
    ///
    /// GaussianLocationScaleFamily overrides
    /// `log_likelihood_only_with_options`,
    /// `exact_newton_joint_hessian_workspace_with_options`, and
    /// `exact_newton_joint_psi_workspace_with_options` to consume
    /// `options.outer_score_subsample` with per-row Horvitz–Thompson weights
    /// (each sampled row's contribution is multiplied by
    /// `WeightedOuterRow.weight = 1/π_i`; non-sampled rows are zeroed),
    /// yielding unbiased estimators of the full-data log-likelihood, joint
    /// Hessian, and second-order ψ Hessian / ψ-Hessian directional
    /// derivative. The ψ-workspace masking happens inside
    /// `apply_ht_mask_first`, `apply_ht_mask_second`, and
    /// `apply_ht_mask_mixed` on the `GaussianJointPsi{First,Second,
    /// MixedDrift}Weights` per-row arrays, immediately after the row-scalar
    /// reductions and before the row-linear `weighted_crossprod_psi_maps` /
    /// `xt_diag_*_dense` assemblies, so the masked outputs remain unbiased.
    /// First-order ψ terms remain full-data exact (= trivially unbiased), so
    /// the total outer score is still unbiased. Inner-PIRLS and final-
    /// covariance paths never install the option, so they continue to
    /// consume the exact full-data quantities.
    fn outer_derivative_subsample_capable(&self) -> bool {
        true
    }
}

impl CustomFamilyGenerative for GaussianLocationScaleFamily {
    fn generativespec(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<GenerativeSpec, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let mu = block_states[Self::BLOCK_MU].eta.clone();
        let eta_log_sigma = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let sigma = gamlss_rowwise_map(eta_log_sigma.len(), |i| {
            logb_sigma_from_eta_scalar(eta_log_sigma[i])
        });
        Ok(GenerativeSpec {
            mean: mu,
            noise: NoiseModel::Gaussian { sigma },
        })
    }
}
