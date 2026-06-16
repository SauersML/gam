// Real concern-organized submodule of the gamlss family stack.
// Cross-module items are re-exported flat through the parent (`gamlss.rs`),
// so `use super::*;` makes the sibling-concern symbols this module references
// resolve through the parent namespace.
use super::*;

pub(crate) struct GaussianLocationScaleWiggleGeometry {
    pub(crate) basis: Array2<f64>,
    pub(crate) basis_d1: Array2<f64>,
    pub(crate) basis_d2: Array2<f64>,
    pub(crate) basis_d3: Array2<f64>,
    pub(crate) dq_dq0: Array1<f64>,
    pub(crate) d2q_dq02: Array1<f64>,
    pub(crate) d3q_dq03: Array1<f64>,
    pub(crate) d4q_dq04: Array1<f64>,
}

/// Per-row pieces of the 3-block Gaussian location-scale-wiggle joint
/// Hessian. Both the dense path and the matrix-free workspace share these
/// row coefficients; only the assembly differs.
pub(crate) struct GaussianLocationScaleWiggleHessianRowPieces {
    pub(crate) coeff_mm: Array1<f64>,
    pub(crate) coeff_ml: Array1<f64>,
    pub(crate) coeff_ll: Array1<f64>,
    pub(crate) coeff_mw_b: Array1<f64>,
    pub(crate) coeff_mw_d: Array1<f64>,
    pub(crate) coeff_lw_b: Array1<f64>,
    pub(crate) coeff_ww: Array1<f64>,
    pub(crate) basis: Array2<f64>,
    pub(crate) basis_d1: Array2<f64>,
}

impl GaussianLocationScaleWiggleHessianRowPieces {
    pub(crate) fn assemble_dense(
        &self,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Array2<f64>, String> {
        let h_mm = xt_diag_x_dense(xmu, &self.coeff_mm)?;
        let h_ml = xt_diag_y_dense(xmu, &self.coeff_ml, x_ls)?;
        let h_ll = xt_diag_x_dense(x_ls, &self.coeff_ll)?;
        let h_mw = xt_diag_y_dense(xmu, &self.coeff_mw_b, &self.basis)?
            + &xt_diag_y_dense(xmu, &self.coeff_mw_d, &self.basis_d1)?;
        let h_lw = xt_diag_y_dense(x_ls, &self.coeff_lw_b, &self.basis)?;
        let h_ww = xt_diag_x_dense(&self.basis, &self.coeff_ww)?;
        Ok(gaussian_pack_wiggle_joint_symmetrichessian(
            &h_mm, &h_ml, &h_mw, &h_ll, &h_lw, &h_ww,
        ))
    }
}

pub struct GaussianLocationScaleWiggleFamily {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub mu_design: Option<DesignMatrix>,
    pub log_sigma_design: Option<DesignMatrix>,
    pub wiggle_knots: Array1<f64>,
    pub wiggle_degree: usize,
    /// Resource policy threaded into PsiDesignMap construction (and any other
    /// per-call materialization decision) made during exact-Newton joint psi
    /// derivative evaluation. Defaults to `ResourcePolicy::default_library()`
    /// when the family is built without an explicit policy.
    pub policy: crate::solver::resource::ResourcePolicy,
    pub(crate) cached_row_scalars:
        std::sync::RwLock<Option<(f64, f64, f64, f64, f64, f64, Arc<GaussianJointRowScalars>)>>,
}

impl Clone for GaussianLocationScaleWiggleFamily {
    fn clone(&self) -> Self {
        Self {
            y: self.y.clone(),
            weights: self.weights.clone(),
            mu_design: self.mu_design.clone(),
            log_sigma_design: self.log_sigma_design.clone(),
            wiggle_knots: self.wiggle_knots.clone(),
            wiggle_degree: self.wiggle_degree,
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

impl GaussianLocationScaleWiggleFamily {
    pub const BLOCK_MU: usize = 0;
    pub const BLOCK_LOG_SIGMA: usize = 1;
    pub const BLOCK_WIGGLE: usize = 2;

    pub fn parameternames() -> &'static [&'static str] {
        &["mu", "log_sigma", "wiggle"]
    }

    pub fn parameter_links() -> &'static [ParameterLink] {
        &[
            ParameterLink::Identity,
            ParameterLink::Log,
            ParameterLink::Wiggle,
        ]
    }

    pub fn metadata() -> FamilyMetadata {
        FamilyMetadata {
            name: "gaussian_location_scalewiggle",
            parameternames: Self::parameternames(),
            parameter_links: Self::parameter_links(),
        }
    }

    pub(crate) fn exact_joint_supported(&self) -> bool {
        self.mu_design.is_some() && self.log_sigma_design.is_some()
    }

    pub(crate) fn wiggle_basiswith_options(
        &self,
        q0: ArrayView1<'_, f64>,
        options: BasisOptions,
    ) -> Result<Array2<f64>, String> {
        monotone_wiggle_basis_with_derivative_order(
            q0,
            &self.wiggle_knots,
            self.wiggle_degree,
            options.derivative_order,
        )
    }

    pub(crate) fn wiggle_design(&self, q0: ArrayView1<'_, f64>) -> Result<Array2<f64>, String> {
        self.wiggle_basiswith_options(q0, BasisOptions::value())
    }

    pub(crate) fn wiggle_dq_dq0(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d1 = self.wiggle_basiswith_options(q0, BasisOptions::first_derivative())?;
        if d1.ncols() != beta_link_wiggle.len() {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "wiggle derivative/beta mismatch: basis has {} columns but beta_link_wiggle has {} coefficients",
                d1.ncols(),
                beta_link_wiggle.len()
            ) }.into());
        }
        Ok(d1.dot(&beta_link_wiggle) + 1.0)
    }

    pub(crate) fn wiggle_d2q_dq02(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d2 = self.wiggle_basiswith_options(q0, BasisOptions::second_derivative())?;
        if d2.ncols() != beta_link_wiggle.len() {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "wiggle second-derivative/beta mismatch: basis has {} columns but beta_link_wiggle has {} coefficients",
                d2.ncols(),
                beta_link_wiggle.len()
            ) }.into());
        }
        Ok(d2.dot(&beta_link_wiggle))
    }

    pub(crate) fn wiggle_d3basis_constrained(
        &self,
        q0: ArrayView1<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        monotone_wiggle_basis_with_derivative_order(q0, &self.wiggle_knots, self.wiggle_degree, 3)
    }

    pub(crate) fn wiggle_d3q_dq03(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d3 = self.wiggle_d3basis_constrained(q0)?;
        if d3.ncols() != beta_link_wiggle.len() {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "wiggle third-derivative/beta mismatch: basis has {} columns but beta_link_wiggle has {} coefficients",
                d3.ncols(),
                beta_link_wiggle.len()
            ) }.into());
        }
        Ok(d3.dot(&beta_link_wiggle))
    }

    pub(crate) fn wiggle_d4q_dq04(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d4 = monotone_wiggle_basis_with_derivative_order(
            q0,
            &self.wiggle_knots,
            self.wiggle_degree,
            4,
        )?;
        if d4.ncols() != beta_link_wiggle.len() {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "wiggle fourth-derivative/beta mismatch: basis has {} columns but beta_link_wiggle has {} coefficients",
                d4.ncols(),
                beta_link_wiggle.len()
            ) }.into());
        }
        Ok(d4.dot(&beta_link_wiggle))
    }

    pub(crate) fn wiggle_geometry(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<GaussianLocationScaleWiggleGeometry, String> {
        let basis = self.wiggle_design(q0)?;
        let basis_d1 = self.wiggle_basiswith_options(q0, BasisOptions::first_derivative())?;
        let basis_d2 = self.wiggle_basiswith_options(q0, BasisOptions::second_derivative())?;
        let basis_d3 = self.wiggle_d3basis_constrained(q0)?;
        let dq_dq0 = self.wiggle_dq_dq0(q0, beta_link_wiggle)?;
        let d2q_dq02 = self.wiggle_d2q_dq02(q0, beta_link_wiggle)?;
        let d3q_dq03 = self.wiggle_d3q_dq03(q0, beta_link_wiggle)?;
        let d4q_dq04 = self.wiggle_d4q_dq04(q0, beta_link_wiggle)?;
        Ok(GaussianLocationScaleWiggleGeometry {
            basis,
            basis_d1,
            basis_d2,
            basis_d3,
            dq_dq0,
            d2q_dq02,
            d3q_dq03,
            d4q_dq04,
        })
    }

    pub(crate) fn get_or_compute_row_scalars(
        &self,
        q: &Array1<f64>,
        eta_ls: &Array1<f64>,
    ) -> Result<Arc<GaussianJointRowScalars>, String> {
        Ok(Arc::new(gaussian_jointrow_scalars(
            &self.y,
            q,
            eta_ls,
            &self.weights,
        )?))
    }

    pub(crate) fn dense_block_designs(
        &self,
    ) -> Result<(Cow<'_, Array2<f64>>, Cow<'_, Array2<f64>>), String> {
        dense_locscale_block_designs_cached(
            self.mu_design.as_ref(),
            self.log_sigma_design.as_ref(),
            "GaussianLocationScaleWiggleFamily",
            "GaussianLocationScaleWiggle",
            "mu",
            &self.policy.material_policy(),
        )
    }
    pub(crate) fn dense_block_designs_fromspecs<'a>(
        &self,
        specs: &'a [ParameterBlockSpec],
    ) -> Result<(Cow<'a, Array2<f64>>, Cow<'a, Array2<f64>>), String> {
        dense_locscale_block_designs_fromspecs(
            specs,
            3,
            "GaussianLocationScaleWiggleFamily",
            "GaussianLocationScaleWiggle",
            Self::BLOCK_MU,
            Self::BLOCK_LOG_SIGMA,
            "mu",
            &self.policy.material_policy(),
        )
    }

    pub(crate) fn exact_joint_dense_block_designs<'a>(
        &'a self,
        specs: Option<&'a [ParameterBlockSpec]>,
    ) -> Result<Option<(Cow<'a, Array2<f64>>, Cow<'a, Array2<f64>>)>, String> {
        if self.exact_joint_supported() {
            return self.dense_block_designs().map(Some);
        }
        if let Some(specs) = specs {
            return self.dense_block_designs_fromspecs(specs).map(Some);
        }
        Ok(None)
    }

    /// Build the [`BlockEffectiveJacobian`] for block `block_idx`.
    ///
    /// The wiggle block (block 2) modulates the inverse link nonlinearly and
    /// does not contribute a linear additive term to any output η; its
    /// Jacobian is an `(2 * n, p_wiggle)` zero matrix.
    ///
    /// - block 0 (mu):        output 0 = design rows, output 1 = zeros
    /// - block 1 (log_sigma): output 0 = zeros, output 1 = design rows
    /// - block 2 (wiggle):    all zeros (nonlinear link modulation)
    pub fn block_effective_jacobian(
        specs: &[ParameterBlockSpec],
        block_idx: usize,
    ) -> Result<Box<dyn BlockEffectiveJacobian>, String> {
        crate::families::block_layout::block_jacobian::AdditiveWiggleBlockLayout {
            family: "GaussianLocationScaleWiggleFamily",
            n_outputs: 2,
            additive_blocks: &[Self::BLOCK_MU, Self::BLOCK_LOG_SIGMA],
            wiggle_block: Some(Self::BLOCK_WIGGLE),
        }
        .block_effective_jacobian(specs, block_idx)
    }
}

/// Row-coefficient bundle for the GLS Wiggle joint second directional
/// derivative, shared by the matrix-free operator and the dense
/// `_from_designs` assemblies. Holds exactly the quantities both consumers
/// read downstream of the (identical) coefficient computation.
pub(crate) struct GlsWiggleSecondDirCoeffs {
    pub(crate) coeff_mm_uv: Array1<f64>,
    pub(crate) coeff_ml_uv: Array1<f64>,
    pub(crate) coeff_ll_uv: Array1<f64>,
    pub(crate) a_u: Array1<f64>,
    pub(crate) a_v: Array1<f64>,
    pub(crate) a_uv: Array1<f64>,
    pub(crate) c_u: Array1<f64>,
    pub(crate) c_v: Array1<f64>,
    pub(crate) c_uv: Array1<f64>,
    pub(crate) l_u: Array1<f64>,
    pub(crate) l_v: Array1<f64>,
    pub(crate) l_uv: Array1<f64>,
    pub(crate) dw_u: Array1<f64>,
    pub(crate) dw_v: Array1<f64>,
    pub(crate) dw_uv: Array1<f64>,
}

/// The two probe directions resolved to row space for the GLS Wiggle joint
/// second directional derivative: `xi`/`zeta` are the X_mu/X_ls contractions,
/// and `q`/`s1`/`g2` are the mixed first/second-derivative wiggle pieces.
pub(crate) struct GlsWiggleDirPieces<'a> {
    pub(crate) zeta_u: &'a Array1<f64>,
    pub(crate) zeta_v: &'a Array1<f64>,
    pub(crate) q_u: &'a Array1<f64>,
    pub(crate) q_v: &'a Array1<f64>,
    pub(crate) q_uv: &'a Array1<f64>,
    pub(crate) s1_u: &'a Array1<f64>,
    pub(crate) s1_v: &'a Array1<f64>,
    pub(crate) s1_uv: &'a Array1<f64>,
    pub(crate) g2_u: &'a Array1<f64>,
    pub(crate) g2_v: &'a Array1<f64>,
    pub(crate) g2_uv: &'a Array1<f64>,
}

/// Compute the shared GLS Wiggle second-directional row coefficients from the
/// per-row scalars, wiggle geometry, and the resolved probe directions.
pub(crate) fn gls_wiggle_second_directional_coeffs(
    rows: &GaussianJointRowScalars,
    geom: &GaussianLocationScaleWiggleGeometry,
    dir: &GlsWiggleDirPieces<'_>,
) -> GlsWiggleSecondDirCoeffs {
    let GlsWiggleDirPieces {
        zeta_u,
        zeta_v,
        q_u,
        q_v,
        q_uv,
        s1_u,
        s1_v,
        s1_uv,
        g2_u,
        g2_v,
        g2_uv,
    } = *dir;
    let szeta_u = &rows.kappa * zeta_u;
    let szeta_v = &rows.kappa * zeta_v;
    let zeta_u_zeta_v = zeta_u * zeta_v;
    let dw_u = -2.0 * &rows.w * &szeta_u;
    let dw_v = -2.0 * &rows.w * &szeta_v;
    let dw_uv =
        4.0 * &rows.w * &(&szeta_u * &szeta_v) - 2.0 * &rows.w * &rows.kappa_prime * &zeta_u_zeta_v;
    let dm_u = -(&rows.w * q_u) - &(2.0 * &rows.m * &szeta_u);
    let dm_v = -(&rows.w * q_v) - &(2.0 * &rows.m * &szeta_v);
    let dm_uv = &(2.0 * &rows.w * &(q_u * &szeta_v + q_v * &szeta_u)) - &(&rows.w * q_uv)
        + &(4.0 * &rows.m * &(&szeta_u * &szeta_v))
        - 2.0 * &rows.m * &rows.kappa_prime * &zeta_u_zeta_v;
    let coeff_mm_uv = &(&dw_uv * &geom.dq_dq0.mapv(|v| v * v))
        + &(2.0 * &dw_u * &geom.dq_dq0 * s1_v)
        + &(2.0 * &dw_v * &geom.dq_dq0 * s1_u)
        + &(2.0 * &rows.w * s1_u * s1_v)
        + &(2.0 * &rows.w * &geom.dq_dq0 * s1_uv)
        - &(&dm_uv * &geom.d2q_dq02)
        - &(&dm_u * g2_v)
        - &(&dm_v * g2_u)
        - &(&rows.m * g2_uv);
    let n = rows.m.len();
    // H_{μ,ls} ≡ Fisher 0 (mean⊥scale orthogonality; the wiggle and μ both
    // enter the mean, log σ is the only scale block), so every β-directional
    // derivative — including this second-order one — is identically 0.
    let coeff_ml_uv = Array1::<f64>::zeros(n);
    // Second directional derivative of the Fisher (log σ, log σ) block
    // coeff_ll = 2κ²a (#566). η_ls is linear in β (no zeta_uv), so the only
    // surviving term is ∂²(2κ²a)/∂η² · zeta_u·zeta_v = 4a(κ'²+κκ'')·zeta_u·zeta_v
    // — matching the dense helper `d_uv` (gaussian_jointsecond_directionalweights).
    let coeff_ll_uv = 4.0
        * &rows.obs_weight
        * &(&rows.kappa_prime * &rows.kappa_prime + &rows.kappa * &rows.kappa_dprime)
        * &zeta_u_zeta_v;

    let a_u = &dw_u * &geom.dq_dq0 + &rows.w * s1_u;
    let a_v = &dw_v * &geom.dq_dq0 + &rows.w * s1_v;
    let a_uv = &dw_uv * &geom.dq_dq0 + &dw_u * s1_v + &dw_v * s1_u + &rows.w * s1_uv;
    let c_u = -&dm_u;
    let c_v = -&dm_v;
    let c_uv = -&dm_uv;
    // H_{ls,w} ≡ Fisher 0 (wiggle is mean-side; mean⊥scale), so all of its
    // β-directional derivatives are 0.
    let l_u = Array1::<f64>::zeros(n);
    let l_v = Array1::<f64>::zeros(n);
    let l_uv = Array1::<f64>::zeros(n);

    GlsWiggleSecondDirCoeffs {
        coeff_mm_uv,
        coeff_ml_uv,
        coeff_ll_uv,
        a_u,
        a_v,
        a_uv,
        c_u,
        c_v,
        c_uv,
        l_u,
        l_v,
        l_uv,
        dw_u,
        dw_v,
        dw_uv,
    }
}

impl GaussianLocationScaleWiggleFamily {
    pub(crate) fn exact_newton_joint_hessian_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: Option<&[ParameterBlockSpec]>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(specs)? else {
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
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(specs)? else {
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
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(specs)? else {
            return Ok(None);
        };
        self.exact_newton_joint_hessiansecond_directional_derivative_from_designs(
            block_states,
            &xmu,
            &x_ls,
            d_beta_u_flat,
            d_beta_v_flat,
        )
    }

    pub(crate) fn exact_newton_joint_psi_direction(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
        policy: &crate::solver::resource::ResourcePolicy,
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
            3,
            "GaussianLocationScaleWiggleFamily",
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
                family_name: "GaussianLocationScaleWiggleFamily",
                primary_label: "mu",
                policy: &self.policy,
            },
        )
    }

    /// Compute the rowwise Hessian pieces shared by the dense path and the
    /// matrix-free workspace operator. The same coefficients reconstruct the
    /// dense p×p matrix or apply `Hv` directly without ever forming it.
    pub(crate) fn wiggle_hessian_row_pieces(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<GaussianLocationScaleWiggleHessianRowPieces, String> {
        validate_block_count::<GamlssError>(
            "GaussianLocationScaleWiggleFamily",
            3,
            block_states.len(),
        )?;
        let q0 = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        if q0.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let q = q0 + etaw;
        let geom = self.wiggle_geometry(q0.view(), betaw.view())?;
        if geom.basis.ncols() != betaw.len() {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "GaussianLocationScaleWiggleFamily wiggle basis/beta mismatch: basis has {} columns but beta has {} entries",
                geom.basis.ncols(),
                betaw.len()
            ) }.into());
        }
        let rows = self.get_or_compute_row_scalars(&q, eta_ls)?;
        let coeff_mm = &rows.w * &geom.dq_dq0.mapv(|v| v * v) - &rows.m * &geom.d2q_dq02;
        // Gaussian mean⊥scale Fisher orthogonality. μ (mu) AND the wiggle both
        // enter the MEAN q = q0 + B(q0)·βw (see `let q = q0 + etaw`); log σ is
        // the only scale-side block. The Fisher (expected) cross between any
        // mean-side parameter and log σ is exactly 0: H_{μ,ls} = 2κm·dq_dq0 and
        // H_{ls,w} = 2κm both carry m = r·w = (y−q)·weight/σ², and E[m] =
        // E[r]·w = 0. The dense and matrix-free workspace paths SHARE these row
        // pieces, so setting the cross coeffs to 0 fixes the curvature object
        // (the observed 2κm value) for both. Diagonal/same-side blocks
        // (coeff_mm within mean, coeff_ll within scale, coeff_mw_* within mean,
        // coeff_ww within mean) are untouched.
        let coeff_ml = Array1::<f64>::zeros(n);
        // Fisher/expected (log σ, log σ) information E[H_{ls,ls}] = 2κ²a (#566):
        // the observed 2κ²n + κ'(a−n) collapses at small residuals and
        // over-smooths the scale; E[n]=a gives the residual-free 2κ²a.
        let coeff_ll = 2.0 * &rows.kappa * &rows.kappa * &rows.obs_weight;
        let coeff_mw_b = &rows.w * &geom.dq_dq0;
        let coeff_mw_d = -&rows.m;
        // ls↔wiggle is a mean⊥scale cross (wiggle is mean-side): Fisher 0.
        let coeff_lw_b = Array1::<f64>::zeros(n);
        let coeff_ww = rows.w.clone();
        Ok(GaussianLocationScaleWiggleHessianRowPieces {
            coeff_mm,
            coeff_ml,
            coeff_ll,
            coeff_mw_b,
            coeff_mw_d,
            coeff_lw_b,
            coeff_ww,
            basis: geom.basis,
            basis_d1: geom.basis_d1,
        })
    }

    pub(crate) fn exact_newton_joint_hessian_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let pieces = self.wiggle_hessian_row_pieces(block_states)?;
        Ok(Some(pieces.assemble_dense(xmu, x_ls)?))
    }

    pub(crate) fn exact_newton_joint_hessian_directional_derivative_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        validate_block_count::<GamlssError>(
            "GaussianLocationScaleWiggleFamily",
            3,
            block_states.len(),
        )?;
        let pmu = xmu.ncols();
        let p_ls = x_ls.ncols();
        let q0 = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        let layout = GamlssBetaLayout::withwiggle(pmu, p_ls, betaw.len());
        let (umu, u_ls, uw) = layout.split_three(
            d_beta_flat,
            "GaussianLocationScaleWiggleFamily exact joint directional Hessian",
        )?;
        if q0.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let q = q0 + etaw;
        let geom = self.wiggle_geometry(q0.view(), betaw.view())?;
        let rows = self.get_or_compute_row_scalars(&q, eta_ls)?;
        let xi = fast_av(xmu, &umu);
        let zeta = fast_av(x_ls, &u_ls);
        // logb κ-scaled η_ls direction; κ' = dκ/dη_ls = κ(1−κ).
        let szeta = &rows.kappa * &zeta;
        let phi = fast_av(&geom.basis, &uw);
        let mut q_u = &geom.dq_dq0 * &xi;
        q_u += &phi;
        let mut s1_u = &geom.d2q_dq02 * &xi;
        s1_u += &fast_av(&geom.basis_d1, &uw);
        let mut g2_u = &geom.d3q_dq03 * &xi;
        g2_u += &fast_av(&geom.basis_d2, &uw);
        let basis_u = scale_matrix_rows(&geom.basis_d1, &xi)?;
        let basis1_u = scale_matrix_rows(&geom.basis_d2, &xi)?;
        let dw_u = -2.0 * &rows.w * &szeta;
        let dm_u = -(&rows.w * &q_u) - &(2.0 * &rows.m * &szeta);

        let coeff_mm_u = &(&dw_u * &geom.dq_dq0.mapv(|v| v * v))
            + &(2.0 * &rows.w * &geom.dq_dq0 * &s1_u)
            - &(&dm_u * &geom.d2q_dq02)
            - &(&rows.m * &g2_u);
        // Static blocks: H_{μ,ls} = Fisher 0 (mean⊥scale); H_{ls,ls} = Fisher
        // 2κ²a (#566). H_{μ,ls} ≡ 0 for all β, so its directional derivative is
        // also identically 0. The Fisher (ls,ls) block 2κ²a depends only on
        // η_ls (a is the constant prior weight), so its directional derivative
        // is 4κκ'a·zeta.
        let coeff_ml_u = Array1::<f64>::zeros(n);
        let coeff_ll_u = 4.0 * &rows.kappa * &rows.kappa_prime * &(&zeta * &rows.obs_weight);
        let a_u = &dw_u * &geom.dq_dq0 + &rows.w * &s1_u;
        let c_u = -&dm_u;
        // ls↔wiggle cross block: Fisher 0 (wiggle is mean-side), so its
        // directional derivative is 0 as well.
        let l_u = Array1::<f64>::zeros(n);
        let zeros_ls_b1 = Array1::<f64>::zeros(n);

        let h_mm = xt_diag_x_dense(xmu, &coeff_mm_u)?;
        let h_ml = xt_diag_y_dense(xmu, &coeff_ml_u, x_ls)?;
        let h_ll = xt_diag_x_dense(x_ls, &coeff_ll_u)?;
        let h_mw = xt_diag_y_dense(xmu, &a_u, &geom.basis)?
            + &xt_diag_y_dense(xmu, &(&rows.w * &geom.dq_dq0), &basis_u)?
            + &xt_diag_y_dense(xmu, &c_u, &geom.basis_d1)?
            + &xt_diag_y_dense(xmu, &(-&rows.m), &basis1_u)?;
        let h_lw = xt_diag_y_dense(x_ls, &l_u, &geom.basis)?
            + &xt_diag_y_dense(x_ls, &zeros_ls_b1, &basis_u)?;
        let a_ww = xt_diag_y_dense(&basis_u, &rows.w, &geom.basis)?;
        let h_ww = &a_ww + &a_ww.t() + &xt_diag_x_dense(&geom.basis, &dw_u)?;
        Ok(Some(gaussian_pack_wiggle_joint_symmetrichessian(
            &h_mm, &h_ml, &h_mw, &h_ll, &h_lw, &h_ww,
        )))
    }

    /// Build a matrix-free `RowCoeffOperator` for the GLS Wiggle joint
    /// directional derivative `D_β H_L[u]`. Output dimension is
    /// `pmu + p_ls + pw`. Channels (in order): X_mu, X_ls, B, B', B''.
    pub(crate) fn gls_wiggle_directional_operator(
        &self,
        block_states: &[ParameterBlockState],
        xmu_arc: Arc<Array2<f64>>,
        x_ls_arc: Arc<Array2<f64>>,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn crate::reml_contracts::HyperOperator>>, String> {
        validate_block_count::<GamlssError>(
            "GaussianLocationScaleWiggleFamily",
            3,
            block_states.len(),
        )?;
        let pmu = xmu_arc.ncols();
        let p_ls = x_ls_arc.ncols();
        let q0_eta = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        let layout = GamlssBetaLayout::withwiggle(pmu, p_ls, betaw.len());
        let (umu, u_ls, uw) =
            layout.split_three(d_beta_flat, "GLS Wiggle joint dH operator d_beta")?;
        if q0_eta.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let q = q0_eta + etaw;
        let geom = self.wiggle_geometry(q0_eta.view(), betaw.view())?;
        let rows = self.get_or_compute_row_scalars(&q, eta_ls)?;
        let xi = fast_av(xmu_arc.as_ref(), &umu);
        let zeta = fast_av(x_ls_arc.as_ref(), &u_ls);
        let szeta = &rows.kappa * &zeta;
        let phi = fast_av(&geom.basis, &uw);
        let mut q_u = &geom.dq_dq0 * &xi;
        q_u += &phi;
        let mut s1_u = &geom.d2q_dq02 * &xi;
        s1_u += &fast_av(&geom.basis_d1, &uw);
        let mut g2_u = &geom.d3q_dq03 * &xi;
        g2_u += &fast_av(&geom.basis_d2, &uw);
        let dw_u = -2.0 * &rows.w * &szeta;
        let dm_u = -(&rows.w * &q_u) - &(2.0 * &rows.m * &szeta);

        let coeff_mm_u = &(&dw_u * &geom.dq_dq0.mapv(|v| v * v))
            + &(2.0 * &rows.w * &geom.dq_dq0 * &s1_u)
            - &(&dm_u * &geom.d2q_dq02)
            - &(&rows.m * &g2_u);
        // H_{μ,ls} ≡ Fisher 0 (mean⊥scale); its directional derivative is 0.
        let coeff_ml_u = Array1::<f64>::zeros(n);
        // Fisher (ls,ls) 2κ²a directional derivative: 4κκ'a·zeta (#566).
        let coeff_ll_u = 4.0 * &rows.kappa * &rows.kappa_prime * &(&zeta * &rows.obs_weight);
        let a_u = &dw_u * &geom.dq_dq0 + &rows.w * &s1_u;
        let c_u = -&dm_u;
        // H_{ls,w} ≡ Fisher 0 (wiggle is mean-side); its derivative is 0 in
        // both the B channel (l_u) and the B' channel (coeff_ls_b1).
        let l_u = Array1::<f64>::zeros(n);

        // Pair-coefficient bundles. For (0=X_mu, 3=B'): combine
        // `xt_diag_y_dense(xmu, &(w·dq_dq0), &basis_u=diag(xi)·B')`
        // (giving coeff `w·dq_dq0·xi`) with `xt_diag_y_dense(xmu, &c_u, &B')`
        // (coeff `c_u`).
        let coeff_m_b1 = &(&rows.w * &geom.dq_dq0 * &xi) + &c_u;
        // (0=X_mu, 4=B''): from `xt_diag_y_dense(xmu, &(-m), &basis1_u=diag(xi)·B'')`.
        let coeff_m_b2 = -(&rows.m * &xi);
        // (1=X_ls, 3=B'): ls↔wiggle Fisher-0 cross → zero.
        let coeff_ls_b1 = Array1::<f64>::zeros(n);
        // (2=B, 3=B'): a_ww + a_ww^T where a_ww = (diag(xi)·B')^T diag(w) B
        // = B'^T diag(w·xi) B. The symmetric pair contribution in
        // `RowCoeffOperator` reproduces a_ww + a_ww^T with c = w·xi.
        let coeff_b_b1 = &rows.w * &xi;

        let basis: Arc<Array2<f64>> = Arc::new(geom.basis.clone());
        let basis_d1: Arc<Array2<f64>> = Arc::new(geom.basis_d1.clone());
        let basis_d2: Arc<Array2<f64>> = Arc::new(geom.basis_d2.clone());
        let pw = basis.ncols();

        Ok(Some(Arc::new(RowCoeffOperator::from_directions(
            vec![pmu, p_ls, pw],
            vec![
                (0, xmu_arc),
                (1, x_ls_arc),
                (2, basis),
                (2, basis_d1),
                (2, basis_d2),
            ],
            vec![
                // (X_mu, X_mu) ← `xt_diag_x_dense(xmu, &coeff_mm_u)`
                (0, 0, coeff_mm_u),
                // (X_mu, X_ls) ← `xt_diag_y_dense(xmu, &coeff_ml_u, x_ls)`
                (0, 1, coeff_ml_u),
                // (X_ls, X_ls) ← `xt_diag_x_dense(x_ls, &coeff_ll_u)`
                (1, 1, coeff_ll_u),
                // (X_mu, B) ← `xt_diag_y_dense(xmu, &a_u, &geom.basis)`
                (0, 2, a_u),
                // (X_mu, B') ← `xt_diag_y_dense(xmu, w·dq_dq0, basis_u=diag(ξ)·B') + xt_diag_y_dense(xmu, c_u, B')`
                (0, 3, coeff_m_b1),
                // (X_mu, B'') ← `xt_diag_y_dense(xmu, -m, basis1_u=diag(ξ)·B'')`
                (0, 4, coeff_m_b2),
                // (X_ls, B) ← `xt_diag_y_dense(x_ls, &l_u, &geom.basis)`
                (1, 2, l_u),
                // (X_ls, B') ← ls↔wiggle is mean⊥scale Fisher 0, so coeff_ls_b1 = 0
                (1, 3, coeff_ls_b1),
                // (B, B) ← `xt_diag_x_dense(&geom.basis, &dw_u)`
                (2, 2, dw_u),
                // (B, B') ← a_ww + a_ww^T = B^T diag(w·ξ) B' + B'^T diag(w·ξ) B
                (2, 3, coeff_b_b1),
            ],
            n,
        ))))
    }

    /// Build a matrix-free `RowCoeffOperator` for the GLS Wiggle joint
    /// second directional derivative `D²_β H_L[u, v]`. Channels: X_mu,
    /// X_ls, B, B', B'', B'''. Pair list mirrors the 8-term `xt_diag_*`
    /// assembly in `_from_designs`, with row-coefficient bundles that
    /// absorb the `ξ_u, ξ_v, ξ_u·ξ_v` row factors arising from
    /// `basis_u = diag(ξ_u)·B'`, `basis_uv = diag(ξ_u·ξ_v)·B''`, etc.
    pub(crate) fn gls_wiggle_second_directional_operator(
        &self,
        block_states: &[ParameterBlockState],
        xmu_arc: Arc<Array2<f64>>,
        x_ls_arc: Arc<Array2<f64>>,
        d_beta_u: &Array1<f64>,
        d_beta_v: &Array1<f64>,
    ) -> Result<Option<Arc<dyn crate::reml_contracts::HyperOperator>>, String> {
        validate_block_count::<GamlssError>(
            "GaussianLocationScaleWiggleFamily",
            3,
            block_states.len(),
        )?;
        let pmu = xmu_arc.ncols();
        let p_ls = x_ls_arc.ncols();
        let q0_eta = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        let layout = GamlssBetaLayout::withwiggle(pmu, p_ls, betaw.len());
        let (umu, u_ls, uw) = layout.split_three(d_beta_u, "GLS Wiggle d2H operator (u)")?;
        let (vmu, v_ls, vw) = layout.split_three(d_beta_v, "GLS Wiggle d2H operator (v)")?;
        if q0_eta.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let q = q0_eta + etaw;
        let geom = self.wiggle_geometry(q0_eta.view(), betaw.view())?;
        let rows = self.get_or_compute_row_scalars(&q, eta_ls)?;

        let xi_u = fast_av(xmu_arc.as_ref(), &umu);
        let xi_v = fast_av(xmu_arc.as_ref(), &vmu);
        let zeta_u = fast_av(x_ls_arc.as_ref(), &u_ls);
        let zeta_v = fast_av(x_ls_arc.as_ref(), &v_ls);
        let phi_u = fast_av(&geom.basis, &uw);
        let phi_v = fast_av(&geom.basis, &vw);
        let b1u = fast_av(&geom.basis_d1, &uw);
        let b1v = fast_av(&geom.basis_d1, &vw);
        let b2u = fast_av(&geom.basis_d2, &uw);
        let b2v = fast_av(&geom.basis_d2, &vw);
        let b3u = fast_av(&geom.basis_d3, &uw);
        let b3v = fast_av(&geom.basis_d3, &vw);

        let mut q_u = &geom.dq_dq0 * &xi_u;
        q_u += &phi_u;
        let mut q_v = &geom.dq_dq0 * &xi_v;
        q_v += &phi_v;
        let mut s1_u = &geom.d2q_dq02 * &xi_u;
        s1_u += &b1u;
        let mut s1_v = &geom.d2q_dq02 * &xi_v;
        s1_v += &b1v;
        let mut g2_u = &geom.d3q_dq03 * &xi_u;
        g2_u += &b2u;
        let mut g2_v = &geom.d3q_dq03 * &xi_v;
        g2_v += &b2v;
        let q_uv = &(&geom.d2q_dq02 * &(&xi_u * &xi_v)) + &(&b1u * &xi_v) + &(&b1v * &xi_u);
        let s1_uv = &(&geom.d3q_dq03 * &(&xi_u * &xi_v)) + &(&b2u * &xi_v) + &(&b2v * &xi_u);
        let g2_uv = &(&geom.d4q_dq04 * &(&xi_u * &xi_v)) + &(&b3u * &xi_v) + &(&b3v * &xi_u);

        let GlsWiggleSecondDirCoeffs {
            coeff_mm_uv,
            coeff_ml_uv,
            coeff_ll_uv,
            a_u,
            a_v,
            a_uv,
            c_u,
            c_v,
            c_uv,
            l_u,
            l_v,
            l_uv,
            dw_u,
            dw_v,
            dw_uv,
        } = gls_wiggle_second_directional_coeffs(
            &rows,
            &geom,
            &GlsWiggleDirPieces {
                zeta_u: &zeta_u,
                zeta_v: &zeta_v,
                q_u: &q_u,
                q_v: &q_v,
                q_uv: &q_uv,
                s1_u: &s1_u,
                s1_v: &s1_v,
                s1_uv: &s1_uv,
                g2_u: &g2_u,
                g2_v: &g2_v,
                g2_uv: &g2_uv,
            },
        );

        // Pair-coefficient bundles. Cross-block (mu, B'/B'') absorb basis_u/v/uv row scaling.
        let xi_u_xi_v = &xi_u * &xi_v;
        let coeff_m_b1 = &(&a_u * &xi_v) + &(&a_v * &xi_u) + &c_uv;
        let coeff_m_b2 = &(&rows.w * &geom.dq_dq0 * &xi_u_xi_v) + &(&c_u * &xi_v) + &(&c_v * &xi_u);
        let coeff_m_b3 = -(&rows.m * &xi_u_xi_v);
        // ls↔wiggle is Fisher-0 (mean⊥scale): the B' (coeff_ls_b1) and B''
        // (coeff_ls_b2) channels of its second directional derivative vanish.
        let coeff_ls_b1 = &(&l_u * &xi_v) + &(&l_v * &xi_u);
        let coeff_ls_b2 = Array1::<f64>::zeros(n);
        // Wiggle-wiggle from a_ab + a_ab^T + a_ij + a_ij^T + a_iwj + a_iwj^T + a_jwi + a_jwi^T:
        //   a_ab = B''^T diag(w·ξ_uξ_v) B    → pair (B, B'', w·ξ_uξ_v)
        //   a_ij = B'^T diag(w·ξ_uξ_v) B'   → pair (B', B', 2·w·ξ_uξ_v)  (a_ij + a_ij^T)
        //   a_iwj+a_jwi = B'^T diag(dw_v·ξ_u + dw_u·ξ_v) B → pair (B, B', sum)
        let coeff_b_b1 = &(&dw_u * &xi_v) + &(&dw_v * &xi_u);
        let coeff_b_b2 = &rows.w * &xi_u_xi_v;
        let coeff_b1_b1 = 2.0 * &(&rows.w * &xi_u_xi_v);

        let basis: Arc<Array2<f64>> = Arc::new(geom.basis.clone());
        let basis_d1: Arc<Array2<f64>> = Arc::new(geom.basis_d1.clone());
        let basis_d2: Arc<Array2<f64>> = Arc::new(geom.basis_d2.clone());
        let basis_d3: Arc<Array2<f64>> = Arc::new(geom.basis_d3.clone());
        let pw = basis.ncols();

        Ok(Some(Arc::new(RowCoeffOperator::from_directions(
            vec![pmu, p_ls, pw],
            vec![
                (0, xmu_arc),
                (1, x_ls_arc),
                (2, basis),
                (2, basis_d1),
                (2, basis_d2),
                (2, basis_d3),
            ],
            vec![
                // (X_mu, X_mu) ← `xt_diag_x_dense(xmu, &coeff_mm_uv)`
                (0, 0, coeff_mm_uv),
                // (X_mu, X_ls) ← `xt_diag_y_dense(xmu, &coeff_ml_uv, x_ls)`
                (0, 1, coeff_ml_uv),
                // (X_ls, X_ls) ← `xt_diag_x_dense(x_ls, &coeff_ll_uv)`
                (1, 1, coeff_ll_uv),
                // (X_mu, B) ← `xt_diag_y_dense(xmu, &a_uv, &geom.basis)`
                (0, 2, a_uv),
                // (X_mu, B') ← combined `a_u·ξ_v + a_v·ξ_u + c_uv` from
                // `xt_diag_y_dense(xmu, a_u, basis_v) + xt_diag_y_dense(xmu,
                // a_v, basis_u) + xt_diag_y_dense(xmu, c_uv, B')`
                (0, 3, coeff_m_b1),
                // (X_mu, B'') ← `xt_diag_y_dense(xmu, w·dq_dq0, basis_uv) +
                // xt_diag_y_dense(xmu, c_u, basis1_v) + xt_diag_y_dense(xmu,
                // c_v, basis1_u)` (basis_uv = diag(ξ_uξ_v)·B'';
                // basis1_{u,v} = diag(ξ_{u,v})·B'')
                (0, 4, coeff_m_b2),
                // (X_mu, B''') ← `xt_diag_y_dense(xmu, -m, basis1_uv)`
                // with basis1_uv = diag(ξ_uξ_v)·B'''
                (0, 5, coeff_m_b3),
                // (X_ls, B) ← `xt_diag_y_dense(x_ls, &l_uv, &geom.basis)`
                (1, 2, l_uv),
                // (X_ls, B') ← combined from `xt_diag_y_dense(x_ls, l_u,
                // basis_v) + xt_diag_y_dense(x_ls, l_v, basis_u)` =
                // `l_u·ξ_v + l_v·ξ_u`
                (1, 3, coeff_ls_b1),
                // (X_ls, B'') ← ls↔wiggle is mean⊥scale Fisher 0, so coeff_ls_b2 = 0
                (1, 4, coeff_ls_b2),
                // (B, B) ← `xt_diag_x_dense(&geom.basis, &dw_uv)`
                (2, 2, dw_uv),
                // (B, B') ← combined `a_iwj + a_iwj^T + a_jwi + a_jwi^T` =
                // B^T diag(dw_u·ξ_v + dw_v·ξ_u) B' + B'^T diag(...) B
                (2, 3, coeff_b_b1),
                // (B, B'') ← `a_ab + a_ab^T` with a_ab = B''^T diag(w·ξ_uξ_v) B
                (2, 4, coeff_b_b2),
                // (B', B') ← `a_ij + a_ij^T = 2·B'^T diag(w·ξ_uξ_v) B'`;
                // diagonal pair coeff doubles to absorb the factor of 2
                (3, 3, coeff_b1_b1),
            ],
            n,
        ))))
    }

    pub(crate) fn exact_newton_joint_hessiansecond_directional_derivative_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        validate_block_count::<GamlssError>(
            "GaussianLocationScaleWiggleFamily",
            3,
            block_states.len(),
        )?;
        let pmu = xmu.ncols();
        let p_ls = x_ls.ncols();
        let q0 = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        let layout = GamlssBetaLayout::withwiggle(pmu, p_ls, betaw.len());
        let (umu, u_ls, uw) = layout.split_three(
            d_beta_u_flat,
            "GaussianLocationScaleWiggleFamily exact joint second directional Hessian (u)",
        )?;
        let (vmu, v_ls, vw) = layout.split_three(
            d_beta_v_flat,
            "GaussianLocationScaleWiggleFamily exact joint second directional Hessian (v)",
        )?;
        if q0.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let q = q0 + etaw;
        let geom = self.wiggle_geometry(q0.view(), betaw.view())?;
        let rows = self.get_or_compute_row_scalars(&q, eta_ls)?;

        let xi_u = fast_av(xmu, &umu);
        let xi_v = fast_av(xmu, &vmu);
        let zeta_u = fast_av(x_ls, &u_ls);
        let zeta_v = fast_av(x_ls, &v_ls);
        let phi_u = fast_av(&geom.basis, &uw);
        let phi_v = fast_av(&geom.basis, &vw);
        let b1u = fast_av(&geom.basis_d1, &uw);
        let b1v = fast_av(&geom.basis_d1, &vw);
        let b2u = fast_av(&geom.basis_d2, &uw);
        let b2v = fast_av(&geom.basis_d2, &vw);
        let b3u = fast_av(&geom.basis_d3, &uw);
        let b3v = fast_av(&geom.basis_d3, &vw);

        let mut q_u = &geom.dq_dq0 * &xi_u;
        q_u += &phi_u;
        let mut q_v = &geom.dq_dq0 * &xi_v;
        q_v += &phi_v;
        let mut s1_u = &geom.d2q_dq02 * &xi_u;
        s1_u += &b1u;
        let mut s1_v = &geom.d2q_dq02 * &xi_v;
        s1_v += &b1v;
        let mut g2_u = &geom.d3q_dq03 * &xi_u;
        g2_u += &b2u;
        let mut g2_v = &geom.d3q_dq03 * &xi_v;
        g2_v += &b2v;
        let q_uv = &(&geom.d2q_dq02 * &(&xi_u * &xi_v)) + &(&b1u * &xi_v) + &(&b1v * &xi_u);
        let s1_uv = &(&geom.d3q_dq03 * &(&xi_u * &xi_v)) + &(&b2u * &xi_v) + &(&b2v * &xi_u);
        let g2_uv = &(&geom.d4q_dq04 * &(&xi_u * &xi_v)) + &(&b3u * &xi_v) + &(&b3v * &xi_u);

        let basis_u = scale_matrix_rows(&geom.basis_d1, &xi_u)?;
        let basis_v = scale_matrix_rows(&geom.basis_d1, &xi_v)?;
        let basis_uv = scale_matrix_rows(&geom.basis_d2, &(&xi_u * &xi_v))?;
        let basis1_u = scale_matrix_rows(&geom.basis_d2, &xi_u)?;
        let basis1_v = scale_matrix_rows(&geom.basis_d2, &xi_v)?;
        let basis1_uv = scale_matrix_rows(&geom.basis_d3, &(&xi_u * &xi_v))?;

        // Shared κ-aware second-directional row coefficients (κ' = κ(1−κ),
        // κ'' = κ(1−κ)(1−2κ), κ''' = κ''(1−2κ) − 2(κ')²): identical to the
        // matrix-free operator path, factored into one helper.
        let GlsWiggleSecondDirCoeffs {
            coeff_mm_uv,
            coeff_ml_uv,
            coeff_ll_uv,
            a_u,
            a_v,
            a_uv,
            c_u,
            c_v,
            c_uv,
            l_u,
            l_v,
            l_uv,
            dw_u,
            dw_v,
            dw_uv,
        } = gls_wiggle_second_directional_coeffs(
            &rows,
            &geom,
            &GlsWiggleDirPieces {
                zeta_u: &zeta_u,
                zeta_v: &zeta_v,
                q_u: &q_u,
                q_v: &q_v,
                q_uv: &q_uv,
                s1_u: &s1_u,
                s1_v: &s1_v,
                s1_uv: &s1_uv,
                g2_u: &g2_u,
                g2_v: &g2_v,
                g2_uv: &g2_uv,
            },
        );

        let h_mm = xt_diag_x_dense(xmu, &coeff_mm_uv)?;
        let h_ml = xt_diag_y_dense(xmu, &coeff_ml_uv, x_ls)?;
        let h_ll = xt_diag_x_dense(x_ls, &coeff_ll_uv)?;
        let h_mw = xt_diag_y_dense(xmu, &a_uv, &geom.basis)?
            + &xt_diag_y_dense(xmu, &a_u, &basis_v)?
            + &xt_diag_y_dense(xmu, &a_v, &basis_u)?
            + &xt_diag_y_dense(xmu, &(&rows.w * &geom.dq_dq0), &basis_uv)?
            + &xt_diag_y_dense(xmu, &c_uv, &geom.basis_d1)?
            + &xt_diag_y_dense(xmu, &c_u, &basis1_v)?
            + &xt_diag_y_dense(xmu, &c_v, &basis1_u)?
            + &xt_diag_y_dense(xmu, &(-&rows.m), &basis1_uv)?;
        // H_{ls,w} ≡ Fisher 0 (mean⊥scale): l_uv/l_u/l_v are 0 (shared helper)
        // and the 2κm·B'' channel vanishes too.
        let zeros_ls_b2 = Array1::<f64>::zeros(n);
        let h_lw = xt_diag_y_dense(x_ls, &l_uv, &geom.basis)?
            + &xt_diag_y_dense(x_ls, &l_u, &basis_v)?
            + &xt_diag_y_dense(x_ls, &l_v, &basis_u)?
            + &xt_diag_y_dense(x_ls, &zeros_ls_b2, &basis_uv)?;
        let a_ab = xt_diag_y_dense(&basis_uv, &rows.w, &geom.basis)?;
        let a_ij = xt_diag_y_dense(&basis_u, &rows.w, &basis_v)?;
        let a_iwj = xt_diag_y_dense(&basis_u, &dw_v, &geom.basis)?;
        let a_jwi = xt_diag_y_dense(&basis_v, &dw_u, &geom.basis)?;
        let h_ww = &a_ab
            + &a_ab.t()
            + &a_ij
            + a_ij.t()
            + &a_iwj
            + a_iwj.t()
            + &a_jwi
            + a_jwi.t()
            + &xt_diag_x_dense(&geom.basis, &dw_uv)?;
        Ok(Some(gaussian_pack_wiggle_joint_symmetrichessian(
            &h_mm, &h_ml, &h_mw, &h_ll, &h_lw, &h_ww,
        )))
    }

    pub(crate) fn exact_newton_joint_psi_terms_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
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
        let q0 = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let q = q0 + etaw;
        let geom = self.wiggle_geometry(q0.view(), betaw.view())?;
        let rows = self.get_or_compute_row_scalars(&q, eta_ls)?;
        let xmu_map = dir_a.x_primary_psi.as_linear_map_ref();
        let x_ls_map = dir_a.x_ls_psi.as_linear_map_ref();

        let q_a = &geom.dq_dq0 * &dir_a.z_primary_psi;
        let s1_a = &geom.d2q_dq02 * &dir_a.z_primary_psi;
        let g2_a = &geom.d3q_dq03 * &dir_a.z_primary_psi;
        let basis_a = scale_matrix_rows(&geom.basis_d1, &dir_a.z_primary_psi)?;
        let basis1_a = scale_matrix_rows(&geom.basis_d2, &dir_a.z_primary_psi)?;
        // logb κ-chain on η_ls; e_a = ∂η_ls/∂ψ_a row-direction.
        let e_a = &dir_a.z_ls_psi;
        let amn = &rows.obs_weight - &rows.n;
        let dw_a = -2.0 * &rows.w * &rows.kappa * e_a;
        let dm_a = -(&rows.w * &q_a) - &(2.0 * &rows.m * &rows.kappa * e_a);
        let dn_a = -(2.0 * &rows.m * &q_a) - &(2.0 * &rows.n * &rows.kappa * e_a);
        let s_mu = -&rows.m * &geom.dq_dq0;
        let s_mu_a = -(&dm_a * &geom.dq_dq0) - &(&rows.m * &s1_a);
        let s_ls = &rows.kappa * &amn;
        let s_ls_a = &rows.kappa_prime * &(e_a * &amn) - &rows.kappa * &dn_a;
        let s_w = -&rows.m;
        let s_w_a = -&dm_a;

        let objective_psi = (-&rows.m * &q_a + &s_ls * e_a).sum();
        let score_psi = gaussian_pack_wiggle_joint_score(
            &(xmu_map.transpose_mul(s_mu.view()) + fast_atv(xmu, &s_mu_a)),
            &(x_ls_map.transpose_mul(s_ls.view()) + fast_atv(x_ls, &s_ls_a)),
            &(fast_atv(&basis_a, &s_w) + fast_atv(&geom.basis, &s_w_a)),
        );

        // Static blocks under logb. Gaussian mean⊥scale Fisher orthogonality:
        // μ AND the wiggle both enter the MEAN q = q0 + B(q0)·βw, so log σ is
        // the only scale-side block. The Fisher (expected) cross between any
        // mean-side parameter and log σ is exactly 0 because it carries
        // m = r·weight/σ² and E[m] = E[r]·weight/σ² = 0:
        //   coeff_ml = E[H_{μ,ls}] = 0  (observed 2κmD)
        //   l        = E[H_{ls,w}] = 0  (observed 2κm)
        // A function identically 0 has 0 ψ-derivatives, so coeff_ml_a and l_a
        // vanish too. This mirrors the non-wiggle psi path
        // (gaussian_joint_psi_firstweights: hmu_ls = dhmu_ls = 0) and the
        // wiggle Newton/REML Hessian path (wiggle_hessian_row_pieces:
        // coeff_ml = coeff_lw_b = 0). The observed SCORE (s_mu/s_ls/s_w above)
        // stays exact so Fisher scoring still hits the joint MLE; only the
        // curvature feeding the REML determinant / IFT correction is the
        // (orthogonal) expectation. coeff_ll is the residual-free Fisher
        // 2κ²a (#566); its ψ-derivative coeff_ll_a = 4κκ'a·e_a depends only on
        // η_ls. Same-side blocks (coeff_mm within mean, a/c the μ↔wiggle
        // within-mean cross, coeff_ww within mean) are untouched.
        let n = rows.m.len();
        let coeff_mm = &rows.w * &geom.dq_dq0.mapv(|v| v * v) - &rows.m * &geom.d2q_dq02;
        let coeff_mm_a = &(&dw_a * &geom.dq_dq0.mapv(|v| v * v))
            + &(2.0 * &rows.w * &geom.dq_dq0 * &s1_a)
            - &(&dm_a * &geom.d2q_dq02)
            - &(&rows.m * &g2_a);
        let coeff_ml = Array1::<f64>::zeros(n);
        let coeff_ml_a = Array1::<f64>::zeros(n);
        let coeff_ll = 2.0 * &rows.kappa * &rows.kappa * &rows.obs_weight;
        let coeff_ll_a = 4.0 * &rows.kappa * &rows.kappa_prime * &rows.obs_weight * e_a;
        let a = &rows.w * &geom.dq_dq0;
        let a_a = &dw_a * &geom.dq_dq0 + &rows.w * &s1_a;
        let c = -&rows.m;
        let c_a = -&dm_a;
        let l = Array1::<f64>::zeros(n);
        let l_a = Array1::<f64>::zeros(n);
        let h_mm_a1 = weighted_crossprod_psi_maps(
            xmu_map,
            coeff_mm.view(),
            CustomFamilyPsiLinearMapRef::Dense(xmu),
        )?;
        let h_mm = &h_mm_a1 + &h_mm_a1.t() + &xt_diag_x_dense(xmu, &coeff_mm_a)?;
        let h_ml = weighted_crossprod_psi_maps(
            xmu_map,
            coeff_ml.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(xmu),
            coeff_ml.view(),
            x_ls_map,
        )? + &xt_diag_y_dense(xmu, &coeff_ml_a, x_ls)?;
        let h_ll_a1 = weighted_crossprod_psi_maps(
            x_ls_map,
            coeff_ll.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )?;
        let h_ll = &h_ll_a1 + &h_ll_a1.t() + &xt_diag_x_dense(x_ls, &coeff_ll_a)?;
        let h_mw = weighted_crossprod_psi_maps(
            xmu_map,
            a.view(),
            CustomFamilyPsiLinearMapRef::Dense(&geom.basis),
        )? + &xt_diag_y_dense(xmu, &a_a, &geom.basis)?
            + &xt_diag_y_dense(xmu, &a, &basis_a)?
            + &weighted_crossprod_psi_maps(
                xmu_map,
                c.view(),
                CustomFamilyPsiLinearMapRef::Dense(&geom.basis_d1),
            )?
            + &xt_diag_y_dense(xmu, &c_a, &geom.basis_d1)?
            + &xt_diag_y_dense(xmu, &c, &basis1_a)?;
        let h_lw = weighted_crossprod_psi_maps(
            x_ls_map,
            l.view(),
            CustomFamilyPsiLinearMapRef::Dense(&geom.basis),
        )? + &xt_diag_y_dense(x_ls, &l_a, &geom.basis)?
            + &xt_diag_y_dense(x_ls, &l, &basis_a)?;
        let h_ww_a1 = xt_diag_y_dense(&basis_a, &rows.w, &geom.basis)?;
        let h_ww = &h_ww_a1 + &h_ww_a1.t() + &xt_diag_x_dense(&geom.basis, &dw_a)?;

        Ok(Some(crate::custom_family::ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi: gaussian_pack_wiggle_joint_symmetrichessian(
                &h_mm, &h_ml, &h_mw, &h_ll, &h_lw, &h_ww,
            ),
            hessian_psi_operator: None,
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
        let Some(dir_a) = self.exact_newton_joint_psi_direction(
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
        let Some(dir_b) = self.exact_newton_joint_psi_direction(
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
                &dir_a,
                &dir_b,
                xmu,
                x_ls,
            )?,
        ))
    }

    pub(crate) fn exact_newton_joint_psisecond_order_terms_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        dir_a: &LocationScaleJointPsiDirection,
        dir_b: &LocationScaleJointPsiDirection,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms, String> {
        let second_drifts = self.exact_newton_joint_psisecond_design_drifts(
            block_states,
            derivative_blocks,
            dir_a,
            dir_b,
            xmu,
            x_ls,
        )?;
        let n = self.y.len();
        let xmu_a_map = dir_a.x_primary_psi.as_linear_map_ref();
        let x_ls_a_map = dir_a.x_ls_psi.as_linear_map_ref();
        let xmu_b_map = dir_b.x_primary_psi.as_linear_map_ref();
        let x_ls_b_map = dir_b.x_ls_psi.as_linear_map_ref();
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
        let q0 = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let q = q0 + etaw;
        let geom = self.wiggle_geometry(q0.view(), betaw.view())?;
        let rows = self.get_or_compute_row_scalars(&q, eta_ls)?;

        let q_a = &geom.dq_dq0 * &dir_a.z_primary_psi;
        let q_b = &geom.dq_dq0 * &dir_b.z_primary_psi;
        let q_ab = &(&geom.dq_dq0 * &second_drifts.z_primary_ab)
            + &(&geom.d2q_dq02 * &(&dir_a.z_primary_psi * &dir_b.z_primary_psi));
        let s1_a = &geom.d2q_dq02 * &dir_a.z_primary_psi;
        let s1_b = &geom.d2q_dq02 * &dir_b.z_primary_psi;
        let s1_ab = &(&geom.d3q_dq03 * &(&dir_a.z_primary_psi * &dir_b.z_primary_psi))
            + &(&geom.d2q_dq02 * &second_drifts.z_primary_ab);
        let g2_a = &geom.d3q_dq03 * &dir_a.z_primary_psi;
        let g2_b = &geom.d3q_dq03 * &dir_b.z_primary_psi;
        let g2_ab = &(&geom.d4q_dq04 * &(&dir_a.z_primary_psi * &dir_b.z_primary_psi))
            + &(&geom.d3q_dq03 * &second_drifts.z_primary_ab);
        let basis_a = scale_matrix_rows(&geom.basis_d1, &dir_a.z_primary_psi)?;
        let basis_b = scale_matrix_rows(&geom.basis_d1, &dir_b.z_primary_psi)?;
        let basis_ab = scale_matrix_rows(&geom.basis_d1, &second_drifts.z_primary_ab)?
            + &scale_matrix_rows(
                &geom.basis_d2,
                &(&dir_a.z_primary_psi * &dir_b.z_primary_psi),
            )?;
        let basis1_a = scale_matrix_rows(&geom.basis_d2, &dir_a.z_primary_psi)?;
        let basis1_b = scale_matrix_rows(&geom.basis_d2, &dir_b.z_primary_psi)?;
        let basis1_ab = scale_matrix_rows(&geom.basis_d2, &second_drifts.z_primary_ab)?
            + &scale_matrix_rows(
                &geom.basis_d3,
                &(&dir_a.z_primary_psi * &dir_b.z_primary_psi),
            )?;

        // logb κ-chain on η_ls; κ' = κ(1−κ), κ'' = κ(1−κ)(1−2κ),
        // κ''' = κ''(1−2κ) − 2(κ')².
        let e_a = &dir_a.z_ls_psi;
        let e_b = &dir_b.z_ls_psi;
        let e_ab = &second_drifts.z_ls_ab;
        let amn = &rows.obs_weight - &rows.n;
        // 4κ² − 2κ' (∂²w/∂η² style coefficient when both directions hit η_ls).
        let four_k2_minus_2kpi = 4.0 * &rows.kappa * &rows.kappa - 2.0 * &rows.kappa_prime;

        // Row drifts under logb. The η_ls direction picks up a κ on each step,
        // and η_ls·η_ls picks up (4κ²−2κ') from differentiating κ on the
        // second leg. The η_ab (z_ls_ab) leg uses just one κ from the chain.
        let dw_a = -2.0 * &rows.w * &rows.kappa * e_a;
        let dw_b = -2.0 * &rows.w * &rows.kappa * e_b;
        let dw_ab =
            &four_k2_minus_2kpi * &rows.w * &(e_a * e_b) - &(2.0 * &rows.w * &rows.kappa * e_ab);
        let dm_a = -(&rows.w * &q_a) - &(2.0 * &rows.m * &rows.kappa * e_a);
        let dm_b = -(&rows.w * &q_b) - &(2.0 * &rows.m * &rows.kappa * e_b);
        let dm_ab = &(2.0 * &rows.w * &rows.kappa * &(&q_a * e_b + &q_b * e_a))
            - &(&rows.w * &q_ab)
            + &(&four_k2_minus_2kpi * &rows.m * &(e_a * e_b))
            - &(2.0 * &rows.m * &rows.kappa * e_ab);
        let dn_a = -(2.0 * &rows.m * &q_a) - &(2.0 * &rows.n * &rows.kappa * e_a);
        let dn_b = -(2.0 * &rows.m * &q_b) - &(2.0 * &rows.n * &rows.kappa * e_b);
        let dn_ab = &(2.0 * &rows.w * &(&q_a * &q_b))
            + &(4.0 * &rows.m * &rows.kappa * &(&q_a * e_b + &q_b * e_a))
            - &(2.0 * &rows.m * &q_ab)
            + &(&four_k2_minus_2kpi * &rows.n * &(e_a * e_b))
            - &(2.0 * &rows.n * &rows.kappa * e_ab);

        let s_mu = -&rows.m * &geom.dq_dq0;
        let s_mu_a = -(&dm_a * &geom.dq_dq0) - &(&rows.m * &s1_a);
        let s_mu_b = -(&dm_b * &geom.dq_dq0) - &(&rows.m * &s1_b);
        let s_mu_ab =
            -(&dm_ab * &geom.dq_dq0) - &(&dm_a * &s1_b) - &(&dm_b * &s1_a) - &(&rows.m * &s1_ab);
        // score_ls = κ(a−n); ψ derivatives carry κ' / κ'' from chain on κ.
        let s_ls = &rows.kappa * &amn;
        let s_ls_a = &rows.kappa_prime * &(e_a * &amn) - &rows.kappa * &dn_a;
        let s_ls_b = &rows.kappa_prime * &(e_b * &amn) - &rows.kappa * &dn_b;
        // s_ls_ab = κ''·e_a·e_b·(a−n) + κ'·e_ab·(a−n)
        //         − κ'·(e_a·n_b + e_b·n_a) − κ·n_ab
        let s_ls_ab = &rows.kappa_dprime * &(e_a * e_b) * &amn + &rows.kappa_prime * e_ab * &amn
            - &rows.kappa_prime * &(e_a * &dn_b + e_b * &dn_a)
            - &rows.kappa * &dn_ab;
        let s_w = -&rows.m;
        let s_w_a = -&dm_a;
        let s_w_b = -&dm_b;
        let s_w_ab = -&dm_ab;

        let objective_psi_psi = (&rows.w * &(&q_a * &q_b)
            + &(2.0 * &rows.m * &rows.kappa * &(&q_a * e_b + &q_b * e_a))
            + &((2.0 * &rows.kappa * &rows.kappa * &rows.n + &rows.kappa_prime * &amn)
                * &(e_a * e_b))
            - &(&rows.m * &q_ab)
            + &(&rows.kappa * &amn * e_ab))
            .sum();

        let score_psi_psi = gaussian_pack_wiggle_joint_score(
            &(xmu_ab_map.transpose_mul(s_mu.view())
                + xmu_a_map.transpose_mul(s_mu_b.view())
                + xmu_b_map.transpose_mul(s_mu_a.view())
                + fast_atv(xmu, &s_mu_ab)),
            &(x_ls_ab_map.transpose_mul(s_ls.view())
                + x_ls_a_map.transpose_mul(s_ls_b.view())
                + x_ls_b_map.transpose_mul(s_ls_a.view())
                + fast_atv(x_ls, &s_ls_ab)),
            &(fast_atv(&basis_ab, &s_w)
                + fast_atv(&basis_a, &s_w_b)
                + fast_atv(&basis_b, &s_w_a)
                + fast_atv(&geom.basis, &s_w_ab)),
        );

        // Static blocks under logb. coeff_mm has no κ; coeff_ll = Fisher 2κ²a
        // (#566). Gaussian mean⊥scale Fisher orthogonality: the wiggle and μ
        // both enter the mean (q = q0 + B·βw), log σ is the only scale block,
        // so coeff_ml = E[H_{μ,ls}] = 0 and l = E[H_{ls,w}] = 0 (observed 2κm,
        // E[m]=0). All of their ψ-directional derivatives (a/b/ab) are 0 since
        // a function identically 0 has 0 derivatives. The Fisher (ls,ls) block
        // depends only on η_ls so its derivatives carry only κ.
        let n = rows.m.len();
        let coeff_mm = &rows.w * &geom.dq_dq0.mapv(|v| v * v) - &rows.m * &geom.d2q_dq02;
        let coeff_ml = Array1::<f64>::zeros(n);
        let coeff_ll = 2.0 * &rows.kappa * &rows.kappa * &rows.obs_weight;
        // coeff_mm_a/b/ab: structurally κ-free; correctness now follows from
        // dw_a/_b/_ab and dm_a/_b/_ab carrying the κ chain on η_ls (above).
        let coeff_mm_a = &(&dw_a * &geom.dq_dq0.mapv(|v| v * v))
            + &(2.0 * &rows.w * &geom.dq_dq0 * &s1_a)
            - &(&dm_a * &geom.d2q_dq02)
            - &(&rows.m * &g2_a);
        let coeff_mm_b = &(&dw_b * &geom.dq_dq0.mapv(|v| v * v))
            + &(2.0 * &rows.w * &geom.dq_dq0 * &s1_b)
            - &(&dm_b * &geom.d2q_dq02)
            - &(&rows.m * &g2_b);
        let coeff_mm_ab = &(&dw_ab * &geom.dq_dq0.mapv(|v| v * v))
            + &(2.0 * &dw_a * &geom.dq_dq0 * &s1_b)
            + &(2.0 * &dw_b * &geom.dq_dq0 * &s1_a)
            + &(2.0 * &rows.w * &s1_a * &s1_b)
            + &(2.0 * &rows.w * &geom.dq_dq0 * &s1_ab)
            - &(&dm_ab * &geom.d2q_dq02)
            - &(&dm_a * &g2_b)
            - &(&dm_b * &g2_a)
            - &(&rows.m * &g2_ab);
        // coeff_ml (μ↔logσ) is Fisher 0; its 1st/2nd ψ-directional derivatives
        // are 0 as well.
        let coeff_ml_a = Array1::<f64>::zeros(n);
        let coeff_ml_b = Array1::<f64>::zeros(n);
        let coeff_ml_ab = Array1::<f64>::zeros(n);
        // Fisher (ls,ls) coeff_ll = 2κ²a (a constant prior weight) depends only
        // on η_ls (#566): ∂(2κ²a)/∂η = 4κκ'a, so the ψ-first derivatives are
        // 4κκ'a·e_a / e_b. The η_ab leg carries one κ on top.
        let coeff_ll_a = 4.0 * &rows.kappa * &rows.kappa_prime * &rows.obs_weight * e_a;
        let coeff_ll_b = 4.0 * &rows.kappa * &rows.kappa_prime * &rows.obs_weight * e_b;
        // coeff_ll_ab = ∂²(2κ²a)/∂a∂b = 4a(κ'²+κκ'')·e_a·e_b + 4κκ'a·e_ab
        // (mirrors the dense helper `d2h_ls_ls`).
        let coeff_ll_ab = 4.0
            * &rows.obs_weight
            * &(&rows.kappa_prime * &rows.kappa_prime + &rows.kappa * &rows.kappa_dprime)
            * &(e_a * e_b)
            + 4.0 * &rows.kappa * &rows.kappa_prime * &rows.obs_weight * e_ab;
        let a = &rows.w * &geom.dq_dq0;
        let a_a = &dw_a * &geom.dq_dq0 + &rows.w * &s1_a;
        let a_b = &dw_b * &geom.dq_dq0 + &rows.w * &s1_b;
        let a_ab = &dw_ab * &geom.dq_dq0 + &dw_a * &s1_b + &dw_b * &s1_a + &rows.w * &s1_ab;
        let c = -&rows.m;
        let c_a = -&dm_a;
        let c_b = -&dm_b;
        let c_ab = -&dm_ab;
        // l (logσ↔wiggle) is Fisher 0 (wiggle is mean-side; mean⊥scale), so all
        // of its 1st/2nd ψ-directional derivatives vanish.
        let l = Array1::<f64>::zeros(n);
        let l_a = Array1::<f64>::zeros(n);
        let l_b = Array1::<f64>::zeros(n);
        let l_ab = Array1::<f64>::zeros(n);

        let hmm_ab = weighted_crossprod_psi_maps(
            xmu_ab_map,
            coeff_mm.view(),
            CustomFamilyPsiLinearMapRef::Dense(xmu),
        )?;
        let hmm_ij = weighted_crossprod_psi_maps(xmu_a_map, coeff_mm.view(), xmu_b_map)?;
        let hmm_iwj = weighted_crossprod_psi_maps(
            xmu_a_map,
            coeff_mm_b.view(),
            CustomFamilyPsiLinearMapRef::Dense(xmu),
        )?;
        let hmm_jwi = weighted_crossprod_psi_maps(
            xmu_b_map,
            coeff_mm_a.view(),
            CustomFamilyPsiLinearMapRef::Dense(xmu),
        )?;
        let h_mm = &hmm_ab
            + &hmm_ab.t()
            + &hmm_ij
            + hmm_ij.t()
            + &hmm_iwj
            + hmm_iwj.t()
            + &hmm_jwi
            + hmm_jwi.t()
            + &xt_diag_x_dense(xmu, &coeff_mm_ab)?;
        let h_ml = weighted_crossprod_psi_maps(
            xmu_ab_map,
            coeff_ml.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )? + &weighted_crossprod_psi_maps(xmu_a_map, coeff_ml.view(), x_ls_b_map)?
            + &weighted_crossprod_psi_maps(xmu_b_map, coeff_ml.view(), x_ls_a_map)?
            + &weighted_crossprod_psi_maps(
                xmu_a_map,
                coeff_ml_b.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_ls),
            )?
            + &weighted_crossprod_psi_maps(
                xmu_b_map,
                coeff_ml_a.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_ls),
            )?
            + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(xmu),
                coeff_ml_a.view(),
                x_ls_b_map,
            )?
            + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(xmu),
                coeff_ml_b.view(),
                x_ls_a_map,
            )?
            + &xt_diag_y_dense(xmu, &coeff_ml_ab, x_ls)?
            + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(xmu),
                coeff_ml.view(),
                x_ls_ab_map,
            )?;
        let hll_ab = weighted_crossprod_psi_maps(
            x_ls_ab_map,
            coeff_ll.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )?;
        let hll_ij = weighted_crossprod_psi_maps(x_ls_a_map, coeff_ll.view(), x_ls_b_map)?;
        let hll_iwj = weighted_crossprod_psi_maps(
            x_ls_a_map,
            coeff_ll_b.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )?;
        let hll_jwi = weighted_crossprod_psi_maps(
            x_ls_b_map,
            coeff_ll_a.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )?;
        let h_ll = &hll_ab
            + &hll_ab.t()
            + &hll_ij
            + hll_ij.t()
            + &hll_iwj
            + hll_iwj.t()
            + &hll_jwi
            + hll_jwi.t()
            + &xt_diag_x_dense(x_ls, &coeff_ll_ab)?;
        let h_mw = weighted_crossprod_psi_maps(
            xmu_ab_map,
            a.view(),
            CustomFamilyPsiLinearMapRef::Dense(&geom.basis),
        )? + &weighted_crossprod_psi_maps(
            xmu_a_map,
            a_b.view(),
            CustomFamilyPsiLinearMapRef::Dense(&geom.basis),
        )? + &weighted_crossprod_psi_maps(
            xmu_a_map,
            a.view(),
            CustomFamilyPsiLinearMapRef::Dense(&basis_b),
        )? + &weighted_crossprod_psi_maps(
            xmu_b_map,
            a_a.view(),
            CustomFamilyPsiLinearMapRef::Dense(&geom.basis),
        )? + &xt_diag_y_dense(xmu, &a_ab, &geom.basis)?
            + &xt_diag_y_dense(xmu, &a_a, &basis_b)?
            + &weighted_crossprod_psi_maps(
                xmu_b_map,
                a.view(),
                CustomFamilyPsiLinearMapRef::Dense(&basis_a),
            )?
            + &xt_diag_y_dense(xmu, &a_b, &basis_a)?
            + &xt_diag_y_dense(xmu, &a, &basis_ab)?
            + &weighted_crossprod_psi_maps(
                xmu_ab_map,
                c.view(),
                CustomFamilyPsiLinearMapRef::Dense(&geom.basis_d1),
            )?
            + &weighted_crossprod_psi_maps(
                xmu_a_map,
                c_b.view(),
                CustomFamilyPsiLinearMapRef::Dense(&geom.basis_d1),
            )?
            + &weighted_crossprod_psi_maps(
                xmu_a_map,
                c.view(),
                CustomFamilyPsiLinearMapRef::Dense(&basis1_b),
            )?
            + &weighted_crossprod_psi_maps(
                xmu_b_map,
                c_a.view(),
                CustomFamilyPsiLinearMapRef::Dense(&geom.basis_d1),
            )?
            + &xt_diag_y_dense(xmu, &c_ab, &geom.basis_d1)?
            + &xt_diag_y_dense(xmu, &c_a, &basis1_b)?
            + &weighted_crossprod_psi_maps(
                xmu_b_map,
                c.view(),
                CustomFamilyPsiLinearMapRef::Dense(&basis1_a),
            )?
            + &xt_diag_y_dense(xmu, &c_b, &basis1_a)?
            + &xt_diag_y_dense(xmu, &c, &basis1_ab)?;
        let h_lw = weighted_crossprod_psi_maps(
            x_ls_ab_map,
            l.view(),
            CustomFamilyPsiLinearMapRef::Dense(&geom.basis),
        )? + &weighted_crossprod_psi_maps(
            x_ls_a_map,
            l_b.view(),
            CustomFamilyPsiLinearMapRef::Dense(&geom.basis),
        )? + &weighted_crossprod_psi_maps(
            x_ls_a_map,
            l.view(),
            CustomFamilyPsiLinearMapRef::Dense(&basis_b),
        )? + &weighted_crossprod_psi_maps(
            x_ls_b_map,
            l_a.view(),
            CustomFamilyPsiLinearMapRef::Dense(&geom.basis),
        )? + &xt_diag_y_dense(x_ls, &l_ab, &geom.basis)?
            + &xt_diag_y_dense(x_ls, &l_a, &basis_b)?
            + &weighted_crossprod_psi_maps(
                x_ls_b_map,
                l.view(),
                CustomFamilyPsiLinearMapRef::Dense(&basis_a),
            )?
            + &xt_diag_y_dense(x_ls, &l_b, &basis_a)?
            + &xt_diag_y_dense(x_ls, &l, &basis_ab)?;
        let hww_ab = xt_diag_y_dense(&basis_ab, &rows.w, &geom.basis)?;
        let hww_ij = xt_diag_y_dense(&basis_a, &rows.w, &basis_b)?;
        let hww_iwj = xt_diag_y_dense(&basis_a, &dw_b, &geom.basis)?;
        let hww_jwi = xt_diag_y_dense(&basis_b, &dw_a, &geom.basis)?;
        let h_ww = &hww_ab
            + &hww_ab.t()
            + &hww_ij
            + hww_ij.t()
            + &hww_iwj
            + hww_iwj.t()
            + &hww_jwi
            + hww_jwi.t()
            + &xt_diag_x_dense(&geom.basis, &dw_ab)?;

        Ok(crate::custom_family::ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi,
            hessian_psi_psi: gaussian_pack_wiggle_joint_symmetrichessian(
                &h_mm, &h_ml, &h_mw, &h_ll, &h_lw, &h_ww,
            ),
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
    ) -> Result<Array2<f64>, String> {
        let pmu = xmu.ncols();
        let p_ls = x_ls.ncols();
        let xmu_map = dir_a.x_primary_psi.as_linear_map_ref();
        let x_ls_map = dir_a.x_ls_psi.as_linear_map_ref();
        let q0 = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let layout = GamlssBetaLayout::withwiggle(pmu, p_ls, betaw.len());
        let (umu, u_ls, uw) = layout.split_three(
            d_beta_flat,
            "GaussianLocationScaleWiggleFamily joint psi hessian directional derivative",
        )?;
        let q = q0 + etaw;
        let geom = self.wiggle_geometry(q0.view(), betaw.view())?;
        let rows = self.get_or_compute_row_scalars(&q, eta_ls)?;

        let xi = fast_av(xmu, &umu);
        let zeta = fast_av(x_ls, &u_ls);
        let zmu_a_u = xmu_map.forward_mul(umu.view());
        let zls_a_u = x_ls_map.forward_mul(u_ls.view());
        let b1u = fast_av(&geom.basis_d1, &uw);
        let b2u = fast_av(&geom.basis_d2, &uw);
        let b3u = fast_av(&geom.basis_d3, &uw);

        let q_u = &(&geom.dq_dq0 * &xi) + &fast_av(&geom.basis, &uw);
        let s1_u = &(&geom.d2q_dq02 * &xi) + &b1u;
        let g2_u = &(&geom.d3q_dq03 * &xi) + &b2u;
        let g3_u = &(&geom.d4q_dq04 * &xi) + &b3u;

        let q_a = &geom.dq_dq0 * &dir_a.z_primary_psi;
        let s1_a = &geom.d2q_dq02 * &dir_a.z_primary_psi;
        let g2_a = &geom.d3q_dq03 * &dir_a.z_primary_psi;
        let q_a_u = &(&s1_u * &dir_a.z_primary_psi) + &(&geom.dq_dq0 * &zmu_a_u);
        let s1_a_u = &(&g2_u * &dir_a.z_primary_psi) + &(&geom.d2q_dq02 * &zmu_a_u);
        let g2_a_u = &(&g3_u * &dir_a.z_primary_psi) + &(&geom.d3q_dq03 * &zmu_a_u);

        let basis_u = scale_matrix_rows(&geom.basis_d1, &xi)?;
        let basis1_u = scale_matrix_rows(&geom.basis_d2, &xi)?;
        let basis_a = scale_matrix_rows(&geom.basis_d1, &dir_a.z_primary_psi)?;
        let basis1_a = scale_matrix_rows(&geom.basis_d2, &dir_a.z_primary_psi)?;
        let basis_a_u = scale_matrix_rows(&geom.basis_d2, &(&xi * &dir_a.z_primary_psi))?
            + &scale_matrix_rows(&geom.basis_d1, &zmu_a_u)?;
        let basis1_a_u = scale_matrix_rows(&geom.basis_d3, &(&xi * &dir_a.z_primary_psi))?
            + &scale_matrix_rows(&geom.basis_d2, &zmu_a_u)?;

        // logb κ-chain on η_ls; e_a = ψ_a's η_ls direction, ζ = β-direction.
        // η_au = zls_a_u is the second mixed derivative (β·ψ).
        let e_a = &dir_a.z_ls_psi;
        let four_k2_minus_2kpi = 4.0 * &rows.kappa * &rows.kappa - 2.0 * &rows.kappa_prime;
        let dw_u = -2.0 * &rows.w * &rows.kappa * &zeta;
        let dm_u = -(&rows.w * &q_u) - &(2.0 * &rows.m * &rows.kappa * &zeta);
        let dw_a = -2.0 * &rows.w * &rows.kappa * e_a;
        let dm_a = -(&rows.w * &q_a) - &(2.0 * &rows.m * &rows.kappa * e_a);
        let dw_a_u = &four_k2_minus_2kpi * &rows.w * &(e_a * &zeta)
            - &(2.0 * &rows.w * &rows.kappa * &zls_a_u);
        let dm_a_u = &(2.0 * &rows.w * &rows.kappa * &(&q_a * &zeta + &q_u * e_a))
            - &(&rows.w * &q_a_u)
            + &(&four_k2_minus_2kpi * &rows.m * &(e_a * &zeta))
            - &(2.0 * &rows.m * &rows.kappa * &zls_a_u);

        let coeff_mm_u = &(&dw_u * &geom.dq_dq0.mapv(|v| v * v))
            + &(2.0 * &rows.w * &geom.dq_dq0 * &s1_u)
            - &(&dm_u * &geom.d2q_dq02)
            - &(&rows.m * &g2_u);
        // coeff_ml (μ↔logσ) is mean⊥scale Fisher 0 (E[m]=0), so both its
        // β-drift derivative coeff_ml_u and the mixed coeff_ml_a_u are 0.
        let n = rows.m.len();
        let coeff_ml_u = Array1::<f64>::zeros(n);
        // Fisher (ls,ls) coeff_ll = 2κ²a (#566); ∂(2κ²a)/∂η = 4κκ'a, so the
        // β-drift derivative along ζ is 4κκ'a·ζ.
        let coeff_ll_u = 4.0 * &rows.kappa * &rows.kappa_prime * &rows.obs_weight * &zeta;
        let coeff_mm_a_u = &(&dw_a_u * &geom.dq_dq0.mapv(|v| v * v))
            + &(2.0 * &dw_a * &geom.dq_dq0 * &s1_u)
            + &(2.0 * &dw_u * &geom.dq_dq0 * &s1_a)
            + &(2.0 * &rows.w * &s1_u * &s1_a)
            + &(2.0 * &rows.w * &geom.dq_dq0 * &s1_a_u)
            - &(&dm_a_u * &geom.d2q_dq02)
            - &(&dm_a * &g2_u)
            - &(&dm_u * &g2_a)
            - &(&rows.m * &g2_a_u);
        // coeff_ml_a_u = ∂²(coeff_ml)/∂a∂u = 0 (coeff_ml ≡ Fisher 0).
        let coeff_ml_a_u = Array1::<f64>::zeros(n);
        // coeff_ll_a_u = ∂²(2κ²a)/∂a∂u for the Fisher (ls,ls) block (#566):
        // 4a(κ'²+κκ'')·e_a·ζ + 4κκ'a·η_au (the η_au=zls_a_u mixed leg), mirroring
        // the dense mixed-drift helper.
        let coeff_ll_a_u = 4.0
            * &rows.obs_weight
            * &(&rows.kappa_prime * &rows.kappa_prime + &rows.kappa * &rows.kappa_dprime)
            * &(e_a * &zeta)
            + 4.0 * &rows.kappa * &rows.kappa_prime * &rows.obs_weight * &zls_a_u;

        let a = &rows.w * &geom.dq_dq0;
        let a_u = &dw_u * &geom.dq_dq0 + &rows.w * &s1_u;
        let a_a = &dw_a * &geom.dq_dq0 + &rows.w * &s1_a;
        let a_a_u = &dw_a_u * &geom.dq_dq0 + &dw_a * &s1_u + &dw_u * &s1_a + &rows.w * &s1_a_u;
        let c = -&rows.m;
        let c_u = -&dm_u;
        let c_a = -&dm_a;
        let c_a_u = -&dm_a_u;
        // l (logσ↔wiggle) is mean⊥scale Fisher 0 (wiggle is mean-side), so its
        // β-drift (l_u), ψ (l_a), and mixed (l_a_u) derivatives all vanish.
        let l = Array1::<f64>::zeros(n);
        let l_u = Array1::<f64>::zeros(n);
        let l_a = Array1::<f64>::zeros(n);
        let l_a_u = Array1::<f64>::zeros(n);

        let hmm_a1 = weighted_crossprod_psi_maps(
            xmu_map,
            coeff_mm_u.view(),
            CustomFamilyPsiLinearMapRef::Dense(xmu),
        )?;
        let h_mm = &hmm_a1 + &hmm_a1.t() + &xt_diag_x_dense(xmu, &coeff_mm_a_u)?;
        let h_ml = weighted_crossprod_psi_maps(
            xmu_map,
            coeff_ml_u.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(xmu),
            coeff_ml_u.view(),
            x_ls_map,
        )? + &xt_diag_y_dense(xmu, &coeff_ml_a_u, x_ls)?;
        let hll_a1 = weighted_crossprod_psi_maps(
            x_ls_map,
            coeff_ll_u.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )?;
        let h_ll = &hll_a1 + &hll_a1.t() + &xt_diag_x_dense(x_ls, &coeff_ll_a_u)?;
        let h_mw = weighted_crossprod_psi_maps(
            xmu_map,
            a_u.view(),
            CustomFamilyPsiLinearMapRef::Dense(&geom.basis),
        )? + &weighted_crossprod_psi_maps(
            xmu_map,
            a.view(),
            CustomFamilyPsiLinearMapRef::Dense(&basis_u),
        )? + &xt_diag_y_dense(xmu, &a_a_u, &geom.basis)?
            + &xt_diag_y_dense(xmu, &a_a, &basis_u)?
            + &xt_diag_y_dense(xmu, &a_u, &basis_a)?
            + &xt_diag_y_dense(xmu, &a, &basis_a_u)?
            + &weighted_crossprod_psi_maps(
                xmu_map,
                c_u.view(),
                CustomFamilyPsiLinearMapRef::Dense(&geom.basis_d1),
            )?
            + &weighted_crossprod_psi_maps(
                xmu_map,
                c.view(),
                CustomFamilyPsiLinearMapRef::Dense(&basis1_u),
            )?
            + &xt_diag_y_dense(xmu, &c_a_u, &geom.basis_d1)?
            + &xt_diag_y_dense(xmu, &c_a, &basis1_u)?
            + &xt_diag_y_dense(xmu, &c_u, &basis1_a)?
            + &xt_diag_y_dense(xmu, &c, &basis1_a_u)?;
        let h_lw = weighted_crossprod_psi_maps(
            x_ls_map,
            l_u.view(),
            CustomFamilyPsiLinearMapRef::Dense(&geom.basis),
        )? + &weighted_crossprod_psi_maps(
            x_ls_map,
            l.view(),
            CustomFamilyPsiLinearMapRef::Dense(&basis_u),
        )? + &xt_diag_y_dense(x_ls, &l_a_u, &geom.basis)?
            + &xt_diag_y_dense(x_ls, &l_a, &basis_u)?
            + &xt_diag_y_dense(x_ls, &l_u, &basis_a)?
            + &xt_diag_y_dense(x_ls, &l, &basis_a_u)?;
        let hww_a_u = xt_diag_y_dense(&basis_a_u, &rows.w, &geom.basis)?;
        let hww_aw = xt_diag_y_dense(&basis_a, &dw_u, &geom.basis)?;
        let hww_au = xt_diag_y_dense(&basis_a, &rows.w, &basis_u)?;
        let h_ww = &hww_a_u
            + &hww_a_u.t()
            + &hww_aw
            + hww_aw.t()
            + &hww_au
            + hww_au.t()
            + &xt_diag_x_dense(&geom.basis, &dw_a_u)?;

        Ok(gaussian_pack_wiggle_joint_symmetrichessian(
            &h_mm, &h_ml, &h_mw, &h_ll, &h_lw, &h_ww,
        ))
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
}

impl CustomFamily for GaussianLocationScaleWiggleFamily {
    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn coefficient_hessian_cost(&self, specs: &[ParameterBlockSpec]) -> u64 {
        // Operator-aware (see GaussianLocationScaleFamily for derivation): when
        // `use_joint_matrix_free_path` selects the workspace operator, joint
        // Hv apply is O(n · (p_t + p_ℓ + p_w)) — the row-streaming RowCoeffOperator
        // never materializes the dense (p_t + p_ℓ + p_w)² matrix.
        crate::families::location_scale_engine::location_scale_coefficient_hessian_cost(
            self.y.len() as u64,
            specs,
        )
    }

    fn block_linear_constraints(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        spec: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        if block_idx != Self::BLOCK_WIGGLE {
            return Ok(None);
        }
        Ok(monotone_wiggle_nonnegative_constraints(spec.design.ncols()))
    }

    fn post_update_block_beta(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        block_spec: &ParameterBlockSpec,
        beta: Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        assert!(!block_spec.name.is_empty());
        if block_idx != Self::BLOCK_WIGGLE {
            return Ok(beta);
        }
        validate_monotone_wiggle_beta_nonnegative(
            &beta,
            "GaussianLocationScaleWiggleFamily post-update",
        )?;
        Ok(beta)
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        validate_block_count::<GamlssError>(
            "GaussianLocationScaleWiggleFamily",
            3,
            block_states.len(),
        )?;
        let n = self.y.len();
        let eta_mu = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        if eta_mu.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let ln2pi = (2.0 * std::f64::consts::PI).ln();
        // Per-row kernel emits 6 working values into pre-allocated outputs;
        // ll is reduced via Rayon's sum. Independent across rows. Note
        // wmu == ww (both equal location_working_weight) and the mean+wiggle
        // working responses share row.location_working_shift, applied to
        // eta_mu[i] and etaw[i] respectively. The previous `q = eta_mu + etaw`
        // intermediate is inlined to avoid an extra n-vector allocation.
        let mut zmu = Array1::<f64>::zeros(n);
        let mut wmu = Array1::<f64>::zeros(n);
        let mut zls = Array1::<f64>::zeros(n);
        let mut wls = Array1::<f64>::zeros(n);
        let mut zw = Array1::<f64>::zeros(n);
        let mut ww = Array1::<f64>::zeros(n);
        const CHUNK: usize = 1024;
        let zmu_s = zmu
            .as_slice_memory_order_mut()
            .expect("zeros is contiguous");
        let wmu_s = wmu
            .as_slice_memory_order_mut()
            .expect("zeros is contiguous");
        let zls_s = zls
            .as_slice_memory_order_mut()
            .expect("zeros is contiguous");
        let wls_s = wls
            .as_slice_memory_order_mut()
            .expect("zeros is contiguous");
        let zw_s = zw.as_slice_memory_order_mut().expect("zeros is contiguous");
        let ww_s = ww.as_slice_memory_order_mut().expect("zeros is contiguous");
        let y_view = self.y.view();
        let w_view = self.weights.view();
        let eta_mu_view = eta_mu.view();
        let eta_ls_view = eta_ls.view();
        let etaw_view = etaw.view();
        let ll: f64 = zmu_s
            .par_chunks_mut(CHUNK)
            .zip(wmu_s.par_chunks_mut(CHUNK))
            .zip(zls_s.par_chunks_mut(CHUNK))
            .zip(wls_s.par_chunks_mut(CHUNK))
            .zip(zw_s.par_chunks_mut(CHUNK))
            .zip(ww_s.par_chunks_mut(CHUNK))
            .enumerate()
            .map(
                |(chunk_idx, (((((zmu_c, wmu_c), zls_c), wls_c), zw_c), ww_c))| {
                    let start = chunk_idx * CHUNK;
                    let mut local_ll = 0.0;
                    for local in 0..zmu_c.len() {
                        let i = start + local;
                        let q_i = eta_mu_view[i] + etaw_view[i];
                        let row = gaussian_diagonal_row_kernel(
                            y_view[i],
                            q_i,
                            eta_ls_view[i],
                            w_view[i],
                            ln2pi,
                        );
                        let w_i = row.location_working_weight;
                        let shift = row.location_working_shift;
                        zmu_c[local] = eta_mu_view[i] + shift;
                        wmu_c[local] = w_i;
                        zw_c[local] = etaw_view[i] + shift;
                        ww_c[local] = w_i;
                        zls_c[local] = row.log_sigma_working_response;
                        wls_c[local] = row.log_sigma_working_weight;
                        local_ll += row.log_likelihood;
                    }
                    local_ll
                },
            )
            .sum();

        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: vec![
                BlockWorkingSet::diagonal_checked(zmu, wmu)?,
                BlockWorkingSet::diagonal_checked(zls, wls)?,
                BlockWorkingSet::diagonal_checked(zw, ww)?,
            ],
        })
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        validate_block_count::<GamlssError>(
            "GaussianLocationScaleWiggleFamily",
            3,
            block_states.len(),
        )?;
        let eta_mu = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        if eta_mu.len() != self.y.len()
            || eta_ls.len() != self.y.len()
            || etaw.len() != self.y.len()
            || self.weights.len() != self.y.len()
        {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let q = eta_mu + etaw;
        let ln2pi = (2.0 * std::f64::consts::PI).ln();
        let mut ll = 0.0;
        for i in 0..self.y.len() {
            let sigma_i = logb_sigma_from_eta_scalar(eta_ls[i]);
            let inv_s2 = (sigma_i * sigma_i).recip();
            let r = self.y[i] - q[i];
            ll += self.weights[i] * (-0.5 * (r * r * inv_s2 + ln2pi + 2.0 * sigma_i.ln()));
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
        validate_block_count::<GamlssError>(
            "GaussianLocationScaleWiggleFamily",
            3,
            block_states.len(),
        )?;
        let n = self.y.len();
        let eta_mu = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        if eta_mu.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleWiggleFamily input size mismatch".to_string(),
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
                let sigma_i = logb_sigma_from_eta_scalar(eta_ls[i]);
                let inv_s2 = (sigma_i * sigma_i).recip();
                let r = self.y[i] - eta_mu[i] - etaw[i];
                row.weight * wi * (-0.5 * (r * r * inv_s2 + ln2pi + 2.0 * sigma_i.ln()))
            })
            .sum();
        Ok(ll)
    }

    fn requires_joint_outer_hyper_path(&self) -> bool {
        true
    }

    fn exact_newton_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        d_beta: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        validate_block_count::<GamlssError>(
            "GaussianLocationScaleWiggleFamily",
            3,
            block_states.len(),
        )?;
        let pmu = self
            .mu_design
            .as_ref()
            .ok_or_else(|| {
                "GaussianLocationScaleWiggleFamily exact path is missing mu design".to_string()
            })?
            .ncols();
        let p_ls = self
            .log_sigma_design
            .as_ref()
            .ok_or_else(|| {
                "GaussianLocationScaleWiggleFamily exact path is missing log-sigma design"
                    .to_string()
            })?
            .ncols();
        let pw = block_states[Self::BLOCK_WIGGLE].beta.len();
        let total = pmu + p_ls + pw;
        let (start, end) = match block_idx {
            Self::BLOCK_MU => (0usize, pmu),
            Self::BLOCK_LOG_SIGMA => (pmu, pmu + p_ls),
            Self::BLOCK_WIGGLE => (pmu + p_ls, total),
            _ => return Ok(None),
        };
        if d_beta.len() != end - start {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "GaussianLocationScaleWiggleFamily block {block_idx} d_beta length mismatch: got {}, expected {}",
                d_beta.len(),
                end - start
            ) }.into());
        }
        let mut d_beta_flat = Array1::<f64>::zeros(total);
        d_beta_flat.slice_mut(s![start..end]).assign(d_beta);
        let (xmu, x_ls) = self.dense_block_designs()?;
        let d_joint = self
            .exact_newton_joint_hessian_directional_derivative_from_designs(
                block_states,
                &xmu,
                &x_ls,
                &d_beta_flat,
            )?
            .ok_or_else(|| "missing Gaussian wiggle exact joint directional Hessian".to_string())?;
        Ok(Some(d_joint.slice(s![start..end, start..end]).to_owned()))
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
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_second_directional_derivative_for_specs(
            block_states,
            None,
            d_beta_u_flat,
            d_beta_v_flat,
        )
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
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_second_directional_derivative_for_specs(
            block_states,
            Some(specs),
            d_beta_u_flat,
            d_beta_v_flat,
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
        if !self.exact_joint_supported() {
            return Ok(None);
        }
        Ok(Some(Arc::new(
            GaussianLocationScaleWiggleExactNewtonJointPsiWorkspace::new(
                self.clone(),
                block_states.to_vec(),
                specs,
                derivative_blocks.to_vec(),
            )?,
        )))
    }

    /// Outer-aware joint ψ workspace with optional row subsample.
    ///
    /// The wiggle ψ workspace shares the generic `LocationScaleJointPsiWorkspace`
    /// with the non-wiggle GLS family, and the subsample is plumbed through
    /// the trait. The wiggle's `ws_psi_*_from_parts` impls currently drop the
    /// subsample and fall back to the full-data exact wiggle ψ path; see
    /// their inline rationale and the `apply_ht_mask_*` helpers used by the
    /// non-wiggle GLS family. Storing the subsample here keeps the workspace
    /// signature uniform across both families and leaves a hook for the
    /// follow-up that refactors the wiggle inline arrays into a weights
    /// struct so HT masking can be applied in one place. Even without that
    /// refactor, the total outer score under subsampling remains an unbiased
    /// estimator of the full-data outer score: HT-unbiased LL
    /// (`log_likelihood_only_with_options`) + HT-unbiased ρ-Hessian
    /// (`exact_newton_joint_hessian_workspace_with_options`) + exact-unbiased
    /// ψ (the wiggle workspace path) = unbiased.
    fn exact_newton_joint_psi_workspace_with_options(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        if !self.exact_joint_supported() {
            return Ok(None);
        }
        Ok(Some(Arc::new(
            GaussianLocationScaleWiggleExactNewtonJointPsiWorkspace::new_with_subsample(
                self.clone(),
                block_states.to_vec(),
                specs,
                derivative_blocks.to_vec(),
                options.outer_score_subsample.clone(),
            )?,
        )))
    }

    fn block_geometry(
        &self,
        block_states: &[ParameterBlockState],
        spec: &ParameterBlockSpec,
    ) -> Result<(DesignMatrix, Array1<f64>), String> {
        if spec.name != "wiggle" {
            return Ok((spec.design.clone(), spec.offset.clone()));
        }
        if block_states.is_empty() {
            return Err(GamlssError::UnsupportedConfiguration {
                reason: "Gaussian wiggle geometry requires mean block".to_string(),
            }
            .into());
        }
        let eta_mu = &block_states[Self::BLOCK_MU].eta;
        if eta_mu.len() != self.y.len() {
            return Err(GamlssError::DimensionMismatch {
                reason: "Gaussian wiggle geometry input size mismatch".to_string(),
            }
            .into());
        }
        let x = self.wiggle_design(eta_mu.view())?;
        if x.ncols() != spec.design.ncols() {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "Gaussian dynamic wiggle design col mismatch: got {}, expected {}",
                    x.ncols(),
                    spec.design.ncols()
                ),
            }
            .into());
        }
        let nrows = x.nrows();
        Ok((
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x)),
            Array1::zeros(nrows),
        ))
    }

    fn block_geometry_is_dynamic(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        let workspace = GaussianLocationScaleWiggleHessianWorkspace::new(
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
    /// per-row coefficient arrays in `pieces` (`coeff_mm`, `coeff_ml`,
    /// `coeff_ll`, `coeff_mw_b`, `coeff_mw_d`, `coeff_lw_b`, `coeff_ww`) —
    /// which every downstream assembly (`hessian_dense`, `hessian_matvec`,
    /// `hessian_diagonal`) consumes row-linearly via `Xᵀ diag(W) Y` — are
    /// replaced by a Horvitz–Thompson mask: each sampled row's coefficient
    /// is multiplied by `WeightedOuterRow.weight` (the inverse-inclusion
    /// factor 1/π_i; uniform or stratified sampling both supported), and
    /// non-sampled rows are zeroed. The `basis`/`basis_d1` matrices are
    /// row-weight-independent and remain unchanged. Note that the Gaussian
    /// wiggle has one fewer cross-coefficient than the binomial wiggle
    /// (no `coeff_lw_d`) because the wiggle enters the Gaussian likelihood
    /// only through `q = η_μ + η_w` (no σ-chain). The resulting joint Hessian
    /// is an unbiased estimator of the full-data joint Hessian. Inner PIRLS
    /// never installs the option, so the inner solve continues to consume
    /// the exact full-data Hessian.
    fn exact_newton_joint_hessian_workspace_with_options(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        let mut workspace = GaussianLocationScaleWiggleHessianWorkspace::new(
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

    /// Outer-derivative policy: declare HT-subsample capability.
    ///
    /// GaussianLocationScaleWiggleFamily overrides
    /// `log_likelihood_only_with_options` and
    /// `exact_newton_joint_hessian_workspace_with_options` to consume
    /// `options.outer_score_subsample` with per-row Horvitz–Thompson weights
    /// (each sampled row's contribution is multiplied by
    /// `WeightedOuterRow.weight = 1/π_i`; non-sampled rows are zeroed),
    /// yielding unbiased estimators of the full-data log-likelihood and
    /// joint Hessian. The ψ-workspace path is also subsample-aware via
    /// `exact_newton_joint_psi_workspace_with_options`, which threads the
    /// subsample down to per-row weight masking inside the joint-ψ second-
    /// order and directional-derivative reductions. Inner-PIRLS and final-
    /// covariance paths never install the option, so they continue to
    /// consume the exact full-data quantities.
    fn outer_derivative_subsample_capable(&self) -> bool {
        true
    }

    fn inner_coefficient_hessian_hvp_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        // Same gating as the workspace impl above: matrix-free fires when
        // `exact_joint_dense_block_designs` is satisfiable, which requires
        // both location and scale block designs to be present.  The wiggle
        // block is folded into the operator via the per-row pieces — its
        // presence is implied by reaching the wiggle family in the first
        // place — so the predicate matches the non-wiggle case.
        self.exact_joint_supported()
            && matches!(
                self.exact_joint_dense_block_designs(Some(specs)),
                Ok(Some(_))
            )
    }
}

/// Matrix-free joint-Hessian operator for the 3-block Gaussian
/// location-scale wiggle family. See `GaussianLocationScaleWiggleHessianRowPieces`
/// for the per-row weight structure. The matvec applies
///
///   r_μ  = D_mm u_μ + D_ml u_ls + D_mw_b (B v_w) + D_mw_d (B' v_w),
///   r_ls = D_ml u_μ + D_ll u_ls + D_lw_b (B v_w),
///   r_b  = D_mw_b u_μ + D_lw_b u_ls + D_ww (B v_w),
///   r_d  = D_mw_d u_μ,
///
/// then forms `out_w = B^T r_b + (B')^T r_d`. The ls-wiggle cross block has
/// no B' contribution because the wiggle enters the Gaussian likelihood only
/// through `q = η_μ + η_w` (no σ-chain), so the Gaussian wiggle has one
/// fewer cross-coefficient than the binomial wiggle.
pub(crate) struct GaussianLocationScaleWiggleHessianWorkspace {
    pub(crate) family: GaussianLocationScaleWiggleFamily,
    pub(crate) block_states: Vec<ParameterBlockState>,
    pub(crate) xmu: Arc<Array2<f64>>,
    pub(crate) x_ls: Arc<Array2<f64>>,
    pub(crate) pieces: GaussianLocationScaleWiggleHessianRowPieces,
}

impl GaussianLocationScaleWiggleHessianWorkspace {
    pub(crate) fn new(
        family: GaussianLocationScaleWiggleFamily,
        block_states: Vec<ParameterBlockState>,
        xmu: Array2<f64>,
        x_ls: Array2<f64>,
    ) -> Result<Self, String> {
        let pieces = family.wiggle_hessian_row_pieces(&block_states)?;
        Ok(Self {
            family,
            block_states,
            xmu: Arc::new(xmu),
            x_ls: Arc::new(x_ls),
            pieces,
        })
    }

    /// Apply a Horvitz–Thompson outer-row subsample mask to the precomputed
    /// per-row coefficient arrays in place.
    ///
    /// Each sampled row's `coeff_*[i]` is multiplied by its
    /// `WeightedOuterRow.weight` (the HT inverse-inclusion factor 1/π_i —
    /// uniform or stratified sampling both supported). All non-sampled rows
    /// are zeroed. Because every downstream assembly (`hessian_dense`,
    /// `hessian_matvec`, `hessian_diagonal`) is row-linear in these arrays
    /// via `Xᵀ diag(W) Y`, the resulting joint-Hessian is an unbiased
    /// estimator of the full-data joint Hessian. The `basis`/`basis_d1`
    /// matrices are independent of the per-row weights and remain unchanged.
    /// The Gaussian wiggle has 7 coefficient arrays (no `coeff_lw_d`, unlike
    /// the binomial wiggle's 8) because the wiggle enters the Gaussian
    /// likelihood only through `q = η_μ + η_w` (no σ-chain).
    pub(crate) fn apply_outer_subsample(
        &mut self,
        rows: &[crate::solver::outer_subsample::WeightedOuterRow],
    ) {
        let n = self.pieces.coeff_mm.len();
        let mut mask_mm = Array1::<f64>::zeros(n);
        let mut mask_ml = Array1::<f64>::zeros(n);
        let mut mask_ll = Array1::<f64>::zeros(n);
        let mut mask_mw_b = Array1::<f64>::zeros(n);
        let mut mask_mw_d = Array1::<f64>::zeros(n);
        let mut mask_lw_b = Array1::<f64>::zeros(n);
        let mut maskww = Array1::<f64>::zeros(n);
        for r in rows {
            let i = r.index;
            let w = r.weight;
            mask_mm[i] = self.pieces.coeff_mm[i] * w;
            mask_ml[i] = self.pieces.coeff_ml[i] * w;
            mask_ll[i] = self.pieces.coeff_ll[i] * w;
            mask_mw_b[i] = self.pieces.coeff_mw_b[i] * w;
            mask_mw_d[i] = self.pieces.coeff_mw_d[i] * w;
            mask_lw_b[i] = self.pieces.coeff_lw_b[i] * w;
            maskww[i] = self.pieces.coeff_ww[i] * w;
        }
        self.pieces.coeff_mm = mask_mm;
        self.pieces.coeff_ml = mask_ml;
        self.pieces.coeff_ll = mask_ll;
        self.pieces.coeff_mw_b = mask_mw_b;
        self.pieces.coeff_mw_d = mask_mw_d;
        self.pieces.coeff_lw_b = mask_lw_b;
        self.pieces.coeff_ww = maskww;
    }
}

impl ExactNewtonJointHessianWorkspace for GaussianLocationScaleWiggleHessianWorkspace {
    fn hessian_dense(&self) -> Result<Option<Array2<f64>>, String> {
        // Same Hv structure as `hessian_matvec`, but routed through the
        // already-existing `assemble_dense` row-pieces helper (six GEMMs:
        // h_mm, h_ml, h_mw_b, h_mw_d, h_lw, h_ww). Avoids `total` canonical-
        // basis HVPs in `MatrixFreeSpdOperator::materialize_dense_operator`,
        // which at large scale (n≈320k, p_total≈82) costs ~568s per κ-iter
        // versus ~1s for the dense build.
        let dense = self
            .pieces
            .assemble_dense(self.xmu.as_ref(), self.x_ls.as_ref())?;
        Ok(Some(dense))
    }

    fn hessian_matvec_available(&self) -> bool {
        true
    }

    fn hessian_matvec(&self, v: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        let pmu = self.xmu.ncols();
        let p_ls = self.x_ls.ncols();
        let pw = self.pieces.basis.ncols();
        let total = pmu + p_ls + pw;
        if v.len() != total {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleWiggle matvec dimension mismatch: got {}, expected {}",
                    v.len(),
                    total
                ),
            }
            .into());
        }
        let v_mu = v.slice(s![0..pmu]);
        let v_ls = v.slice(s![pmu..pmu + p_ls]);
        let v_w = v.slice(s![pmu + p_ls..total]);

        let u_mu = fast_av(self.xmu.as_ref(), &v_mu);
        let u_ls = fast_av(self.x_ls.as_ref(), &v_ls);
        let u_b = fast_av(&self.pieces.basis, &v_w);
        let u_d = fast_av(&self.pieces.basis_d1, &v_w);

        let r_mu = &self.pieces.coeff_mm * &u_mu
            + &self.pieces.coeff_ml * &u_ls
            + &self.pieces.coeff_mw_b * &u_b
            + &self.pieces.coeff_mw_d * &u_d;
        let r_ls = &self.pieces.coeff_ml * &u_mu
            + &self.pieces.coeff_ll * &u_ls
            + &self.pieces.coeff_lw_b * &u_b;
        let r_b = &self.pieces.coeff_mw_b * &u_mu
            + &self.pieces.coeff_lw_b * &u_ls
            + &self.pieces.coeff_ww * &u_b;
        let r_d = &self.pieces.coeff_mw_d * &u_mu;

        let out_mu = fast_atv(self.xmu.as_ref(), &r_mu);
        let out_ls = fast_atv(self.x_ls.as_ref(), &r_ls);
        let out_w = fast_atv(&self.pieces.basis, &r_b) + &fast_atv(&self.pieces.basis_d1, &r_d);

        let mut out = Array1::<f64>::zeros(total);
        out.slice_mut(s![0..pmu]).assign(&out_mu);
        out.slice_mut(s![pmu..pmu + p_ls]).assign(&out_ls);
        out.slice_mut(s![pmu + p_ls..total]).assign(&out_w);
        Ok(Some(out))
    }

    fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        let pmu = self.xmu.ncols();
        let p_ls = self.x_ls.ncols();
        let pw = self.pieces.basis.ncols();
        let total = pmu + p_ls + pw;
        // Diagonals are independent column-wise reductions: parallelize.
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let diag_mu: Vec<f64> = (0..pmu)
            .into_par_iter()
            .map(|j| {
                let col = self.xmu.column(j);
                col.iter()
                    .zip(self.pieces.coeff_mm.iter())
                    .map(|(&v, &c)| c * v * v)
                    .sum()
            })
            .collect();
        let diag_ls: Vec<f64> = (0..p_ls)
            .into_par_iter()
            .map(|j| {
                let col = self.x_ls.column(j);
                col.iter()
                    .zip(self.pieces.coeff_ll.iter())
                    .map(|(&v, &c)| c * v * v)
                    .sum()
            })
            .collect();
        let diag_w: Vec<f64> = (0..pw)
            .into_par_iter()
            .map(|j| {
                let col = self.pieces.basis.column(j);
                col.iter()
                    .zip(self.pieces.coeff_ww.iter())
                    .map(|(&v, &c)| c * v * v)
                    .sum()
            })
            .collect();
        let mut diag = Array1::<f64>::zeros(total);
        for (j, v) in diag_mu.into_iter().enumerate() {
            diag[j] = v;
        }
        for (j, v) in diag_ls.into_iter().enumerate() {
            diag[pmu + j] = v;
        }
        for (j, v) in diag_w.into_iter().enumerate() {
            diag[pmu + p_ls + j] = v;
        }
        Ok(Some(diag))
    }

    fn directional_derivative(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .exact_newton_joint_hessian_directional_derivative_from_designs(
                &self.block_states,
                self.xmu.as_ref(),
                self.x_ls.as_ref(),
                d_beta_flat,
            )
    }

    fn directional_derivative_operator(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn crate::reml_contracts::HyperOperator>>, String> {
        self.family.gls_wiggle_directional_operator(
            &self.block_states,
            self.xmu.clone(),
            self.x_ls.clone(),
            d_beta_flat,
        )
    }

    fn second_directional_derivative(
        &self,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .exact_newton_joint_hessiansecond_directional_derivative_from_designs(
                &self.block_states,
                self.xmu.as_ref(),
                self.x_ls.as_ref(),
                d_beta_u_flat,
                d_beta_v_flat,
            )
    }

    fn second_directional_derivative_operator(
        &self,
        d_beta_u: &Array1<f64>,
        d_beta_v: &Array1<f64>,
    ) -> Result<Option<Arc<dyn crate::reml_contracts::HyperOperator>>, String> {
        self.family.gls_wiggle_second_directional_operator(
            &self.block_states,
            self.xmu.clone(),
            self.x_ls.clone(),
            d_beta_u,
            d_beta_v,
        )
    }
}

impl CustomFamilyGenerative for GaussianLocationScaleWiggleFamily {
    fn generativespec(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<GenerativeSpec, String> {
        validate_block_count::<GamlssError>(
            "GaussianLocationScaleWiggleFamily",
            3,
            block_states.len(),
        )?;
        let eta_mu = &block_states[Self::BLOCK_MU].eta;
        let eta_wiggle = &block_states[Self::BLOCK_WIGGLE].eta;
        let eta_log_sigma = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let n = eta_mu.len();
        let mean = gamlss_rowwise_map(n, |i| eta_mu[i] + eta_wiggle[i]);
        let sigma = gamlss_rowwise_map(n, |i| logb_sigma_from_eta_scalar(eta_log_sigma[i]));
        Ok(GenerativeSpec {
            mean,
            noise: NoiseModel::Gaussian { sigma },
        })
    }
}

pub(crate) fn expect_single_block<'a>(
    block_states: &'a [ParameterBlockState],
    family_name: &str,
) -> Result<&'a ParameterBlockState, String> {
    validate_block_count::<GamlssError>(family_name, 1, block_states.len())?;
    Ok(&block_states[0])
}
