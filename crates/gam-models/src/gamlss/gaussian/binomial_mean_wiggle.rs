// Real concern-organized submodule of the gamlss family stack.
// Cross-module items are re-exported flat through the parent (`gamlss.rs`),
// so `use super::*;` makes the sibling-concern symbols this module references
// resolve through the parent namespace.
use super::*;

#[derive(Clone)]
pub struct BinomialMeanWiggleFamily {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub link_kind: InverseLink,
    pub wiggle_knots: Array1<f64>,
    pub wiggle_degree: usize,
    /// Resource policy threaded into PsiDesignMap construction during
    /// exact-Newton joint psi evaluation. Defaults to
    /// `ResourcePolicy::default_library()` when the family is built without
    /// an explicit policy.
    pub policy: gam_runtime::resource::ResourcePolicy,
    /// The **frozen, identifiable** warp design
    /// `B⊥ = (I - P_X) B(η̂)` used for the duration of one inner joint-Newton
    /// solve (#1596). `None` preserves the original dynamic-basis behaviour;
    /// `Some(B⊥)` switches the family into the
    /// frozen-basis Gauss-Newton mode.
    ///
    /// **Why frozen.** The fully-coupled `q = η + B(η)·β_w` model regenerates the
    /// monotone I-spline basis at the *moving* `η` every cycle. The trust-region
    /// quadratic model freezes `B` at the cycle-start `η`, but the line search
    /// re-evaluates the objective with `B` rebuilt at the trial `η`; for any step
    /// that moves `η` the actual reduction diverges from the model, the trust
    /// radius collapses, and the constrained KKT certificate refuses every
    /// iterate (`active_set_incomplete`) even when the optimal warp is flat.
    /// Freezing `B(η̂)` makes `q = η + B⊥·β_w` linear in `(β_η, β_w)` with
    /// `∂q/∂η = 1` and no `∂B/∂η` chain term — a well-conditioned two-block GLM
    /// that certifies. The caller re-freezes at the refit `η̂` and returns only
    /// after the caller-supplied outer convergence policy certifies the resulting
    /// Gauss-Newton fixed point (`fit_binomial_mean_wiggle`).
    ///
    /// **Why observation-space residualized (identifiable).** A monotone
    /// I-spline of the linear predictor `η` can represent mean-block directions,
    /// so the raw `B(η̂)` columns alias `X`. The caller fits
    /// `B⊥ = B - X A`, with `A = (XᵀX)^+XᵀB`, removing only that aliased
    /// observation-space component while preserving the standard I-spline
    /// coefficient coordinate. Consequently the exact structural constraint is
    /// still simply `β_w ≥ 0`; prediction reconstructs `B(η_new)·β_w` and
    /// compensates the saved mean coefficient by `-Aβ_w`.
    pub frozen_warp_design: Option<Arc<Array2<f64>>>,
}

pub(crate) struct BinomialMeanWiggleGeometry {
    pub(crate) basis: Array2<f64>,
    pub(crate) basis_d1: Array2<f64>,
    pub(crate) basis_d2: Array2<f64>,
    pub(crate) basis_d3: Array2<f64>,
    pub(crate) dq_dq0: Array1<f64>,
    pub(crate) d2q_dq02: Array1<f64>,
    pub(crate) d3q_dq03: Array1<f64>,
    pub(crate) d4q_dq04: Array1<f64>,
}

pub(crate) struct BinomialMeanWiggleJointPsiDirection {
    pub(crate) x_eta_psi: Option<Array2<f64>>,
    pub(crate) z_eta_psi: Array1<f64>,
}

impl BinomialMeanWiggleFamily {
    pub const BLOCK_ETA: usize = 0;
    pub const BLOCK_WIGGLE: usize = 1;

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
        let d_constrained = self.wiggle_basiswith_options(q0, BasisOptions::first_derivative())?;
        if d_constrained.ncols() != beta_link_wiggle.len() {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "wiggle derivative/beta mismatch: basis has {} columns but beta_link_wiggle has {} coefficients",
                d_constrained.ncols(),
                beta_link_wiggle.len()
            ) }.into());
        }
        Ok(d_constrained.dot(&beta_link_wiggle) + 1.0)
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
    ) -> Result<BinomialMeanWiggleGeometry, String> {
        // Frozen-basis (#1596): the warp offset `s = B⊥·β_w` uses the pinned,
        // identifiable design `B⊥ = (I-P_X)B(η̂)`, so it is a per-row constant w.r.t.
        // the *live* linear predictor `η`. Then `q = η + s` gives `∂q/∂η = 1`
        // exactly and every higher derivative of `q` in `η` vanishes, and the
        // `∂B/∂η` chain bases drop out (the warp basis does not move with `η`).
        // The value basis `B⊥` — the column block carrying `∂q/∂β_w` — is the
        // only surviving geometry term.
        if let Some(frozen) = self.frozen_warp_design.as_ref() {
            let n = frozen.nrows();
            let pw = frozen.ncols();
            return Ok(BinomialMeanWiggleGeometry {
                basis: frozen.as_ref().clone(),
                basis_d1: Array2::zeros((n, pw)),
                basis_d2: Array2::zeros((n, pw)),
                basis_d3: Array2::zeros((n, pw)),
                dq_dq0: Array1::ones(n),
                d2q_dq02: Array1::zeros(n),
                d3q_dq03: Array1::zeros(n),
                d4q_dq04: Array1::zeros(n),
            });
        }
        let basis = self.wiggle_design(q0)?;
        let basis_d1 = self.wiggle_basiswith_options(q0, BasisOptions::first_derivative())?;
        let basis_d2 = self.wiggle_basiswith_options(q0, BasisOptions::second_derivative())?;
        let basis_d3 = self.wiggle_d3basis_constrained(q0)?;
        let dq_dq0 = self.wiggle_dq_dq0(q0, beta_link_wiggle)?;
        let d2q_dq02 = self.wiggle_d2q_dq02(q0, beta_link_wiggle)?;
        let d3q_dq03 = self.wiggle_d3q_dq03(q0, beta_link_wiggle)?;
        let d4q_dq04 = self.wiggle_d4q_dq04(q0, beta_link_wiggle)?;
        Ok(BinomialMeanWiggleGeometry {
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

    pub(crate) fn neglog_q_derivatives(
        &self,
        y: f64,
        weight: f64,
        q: f64,
    ) -> Result<(f64, f64, f64), String> {
        let jet = inverse_link_jet_for_inverse_link(&self.link_kind, q)
            .map_err(|e| format!("fixed-link wiggle inverse-link evaluation failed: {e}"))?;
        // Pass μ RAW: the dispatch returns the exact q-derivatives of the
        // evaluated loss for every representable μ in (0,1) and handles the
        // saturated boundary itself. See binomial_location_scalerow (#948).
        Ok(binomial_neglog_q_derivatives_dispatch(
            y,
            weight,
            q,
            jet.mu,
            jet.d1,
            jet.d2,
            jet.d3,
            &self.link_kind,
        ))
    }

    pub(crate) fn neglog_q_fourth_derivative(
        &self,
        y: f64,
        weight: f64,
        q: f64,
    ) -> Result<f64, String> {
        let jet = inverse_link_jet_for_inverse_link(&self.link_kind, q)
            .map_err(|e| format!("fixed-link wiggle inverse-link evaluation failed: {e}"))?;
        // Pass μ RAW — see neglog_q_derivatives above (#948).
        binomial_neglog_q_fourth_derivative_dispatch(
            y,
            weight,
            q,
            jet.mu,
            jet.d1,
            jet.d2,
            jet.d3,
            &self.link_kind,
        )
    }

    pub(crate) fn dense_eta_design_fromspecs<'a>(
        &self,
        specs: &'a [ParameterBlockSpec],
    ) -> Result<Cow<'a, Array2<f64>>, String> {
        if specs.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialMeanWiggleFamily expects 2 specs, got {}",
                    specs.len()
                ),
            }
            .into());
        }
        Ok(match specs[Self::BLOCK_ETA].design.as_dense_ref() {
            Some(d) => Cow::Borrowed(d),
            None => Cow::Owned(
                specs[Self::BLOCK_ETA]
                    .design
                    .try_to_dense_with_policy(
                        &self.policy.material_policy(),
                        "BinomialMeanWiggle dense_eta_design_fromspecs eta",
                    )
                    .map_err(|e| e.to_string())?
                    .as_ref()
                    .clone(),
            ),
        })
    }

    pub(crate) fn exact_newton_joint_psi_direction(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        x_eta: &Array2<f64>,
    ) -> Result<Option<BinomialMeanWiggleJointPsiDirection>, String> {
        validate_block_count::<GamlssError>("BinomialMeanWiggleFamily", 2, block_states.len())?;
        if derivative_blocks.len() != 2 {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "BinomialMeanWiggleFamily joint psi direction expects 2 derivative block lists, got {}",
                derivative_blocks.len()
            ) }.into());
        }
        let n = self.y.len();
        let p_eta = x_eta.ncols();
        let beta_eta = &block_states[Self::BLOCK_ETA].beta;
        let mut global = 0usize;
        for (block_idx, block_derivs) in derivative_blocks.iter().enumerate() {
            for deriv in block_derivs {
                if global == psi_index {
                    if block_idx != Self::BLOCK_ETA {
                        return Ok(None);
                    }
                    let x_eta_psi_map = resolve_custom_family_x_psi_map(
                        deriv,
                        n,
                        p_eta,
                        0..n,
                        "BinomialMeanWiggleFamily eta",
                        &self.policy,
                    )?;
                    let x_eta_psi = x_eta_psi_map.row_chunk(0..n)?;
                    let z_eta_psi = x_eta_psi.dot(beta_eta);
                    return Ok(Some(BinomialMeanWiggleJointPsiDirection {
                        x_eta_psi: Some(x_eta_psi),
                        z_eta_psi,
                    }));
                }
                global += 1;
            }
        }
        Ok(None)
    }

    pub(crate) fn exact_newton_joint_psi_action(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        p_eta: usize,
    ) -> Result<Option<(CustomFamilyPsiDesignAction, Array1<f64>)>, String> {
        validate_block_count::<GamlssError>("BinomialMeanWiggleFamily", 2, block_states.len())?;
        if derivative_blocks.len() != 2 {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "BinomialMeanWiggleFamily joint psi action expects 2 derivative block lists, got {}",
                derivative_blocks.len()
            ) }.into());
        }
        let n = self.y.len();
        let beta_eta = &block_states[Self::BLOCK_ETA].beta;
        let mut global = 0usize;
        for (block_idx, block_derivs) in derivative_blocks.iter().enumerate() {
            for deriv in block_derivs {
                if global == psi_index {
                    if block_idx != Self::BLOCK_ETA {
                        return Ok(None);
                    }
                    let action = match CustomFamilyPsiDesignAction::from_first_derivative(
                        deriv,
                        n,
                        p_eta,
                        0..n,
                        "BinomialMeanWiggleFamily eta",
                    ) {
                        Ok(action) => action,
                        Err(_) => return Ok(None),
                    };
                    let z_eta_psi = action.forward_mul(beta_eta.view());
                    return Ok(Some((action, z_eta_psi)));
                }
                global += 1;
            }
        }
        Ok(None)
    }

    pub(crate) fn bmw_static_hessian_operator(
        &self,
        block_states: &[ParameterBlockState],
        x_eta_arc: Arc<Array2<f64>>,
    ) -> Result<Arc<RowCoeffOperator>, String> {
        validate_block_count::<GamlssError>("BinomialMeanWiggleFamily", 2, block_states.len())?;
        let eta = &block_states[Self::BLOCK_ETA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        if eta.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialMeanWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let geom = self.wiggle_geometry(eta.view(), betaw.view())?;
        let p_eta = x_eta_arc.ncols();
        let pw = geom.basis.ncols();
        let mut coeff_eta = Array1::<f64>::zeros(n);
        let mut coeff_etaw_b = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d1 = Array1::<f64>::zeros(n);
        let mut coeff_ww = Array1::<f64>::zeros(n);
        for row in 0..n {
            let q = eta[row] + etaw[row];
            let (m1, m2, _) = self.neglog_q_derivatives(self.y[row], self.weights[row], q)?;
            let a = geom.dq_dq0[row];
            let b = geom.d2q_dq02[row];
            coeff_eta[row] = hessian_coeff_fromobjective_q_terms(m1, m2, a, a, b);
            coeff_etaw_b[row] = m2 * a;
            coeff_etaw_d1[row] = m1;
            coeff_ww[row] = m2;
        }
        Ok(Arc::new(RowCoeffOperator::from_directions(
            vec![p_eta, pw],
            vec![
                (0, x_eta_arc),
                (1, Arc::new(geom.basis)),
                (1, Arc::new(geom.basis_d1)),
            ],
            vec![
                (0, 0, coeff_eta),
                (0, 1, coeff_etaw_b),
                (0, 2, coeff_etaw_d1),
                (1, 1, coeff_ww),
            ],
            n,
        )))
    }

    pub(crate) fn bmw_directional_operator(
        &self,
        block_states: &[ParameterBlockState],
        x_eta_arc: Arc<Array2<f64>>,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn gam_problem::HyperOperator>>, String> {
        validate_block_count::<GamlssError>("BinomialMeanWiggleFamily", 2, block_states.len())?;
        let eta = &block_states[Self::BLOCK_ETA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        if eta.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialMeanWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let geom = self.wiggle_geometry(eta.view(), betaw.view())?;
        let p_eta = x_eta_arc.ncols();
        let pw = geom.basis.ncols();
        let total = p_eta + pw;
        if d_beta_flat.len() != total {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialMeanWiggleFamily joint d_beta length mismatch: got {}, expected {}",
                    d_beta_flat.len(),
                    total
                ),
            }
            .into());
        }
        let u_eta = d_beta_flat.slice(s![0..p_eta]).to_owned();
        let uw = d_beta_flat.slice(s![p_eta..total]).to_owned();
        let xi = fast_av(x_eta_arc.as_ref(), &u_eta);
        let phi = fast_av(&geom.basis, &uw);
        let basis1_u = fast_av(&geom.basis_d1, &uw);
        let basis2_u = fast_av(&geom.basis_d2, &uw);

        let mut coeff_eta = Array1::<f64>::zeros(n);
        let mut coeff_etaw_b = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d1 = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d2 = Array1::<f64>::zeros(n);
        let mut coeff_ww_bb = Array1::<f64>::zeros(n);
        let mut coeff_ww_db = Array1::<f64>::zeros(n);
        for row in 0..n {
            let q = eta[row] + etaw[row];
            let (m1, m2, m3) = self.neglog_q_derivatives(self.y[row], self.weights[row], q)?;
            let a = geom.dq_dq0[row];
            let b = geom.d2q_dq02[row];
            let c = geom.d3q_dq03[row];
            let q_u = a * xi[row] + phi[row];
            let a_u = b * xi[row] + basis1_u[row];
            let b_u = c * xi[row] + basis2_u[row];
            coeff_eta[row] = directionalhessian_coeff_fromobjective_q_terms(
                m1, m2, m3, q_u, a, a, b, a_u, a_u, b_u,
            );
            coeff_etaw_b[row] = m3 * q_u * a + m2 * a_u;
            coeff_etaw_d1[row] = m2 * (a * xi[row] + q_u);
            coeff_etaw_d2[row] = m1 * xi[row];
            coeff_ww_bb[row] = m3 * q_u;
            coeff_ww_db[row] = m2 * xi[row];
        }
        Ok(Some(Arc::new(RowCoeffOperator::from_directions(
            vec![p_eta, pw],
            vec![
                (0, x_eta_arc),
                (1, Arc::new(geom.basis)),
                (1, Arc::new(geom.basis_d1)),
                (1, Arc::new(geom.basis_d2)),
            ],
            vec![
                (0, 0, coeff_eta),
                (0, 1, coeff_etaw_b),
                (0, 2, coeff_etaw_d1),
                (0, 3, coeff_etaw_d2),
                (1, 1, coeff_ww_bb),
                (1, 2, coeff_ww_db),
            ],
            n,
        ))))
    }

    pub(crate) fn bmw_second_directional_operator(
        &self,
        block_states: &[ParameterBlockState],
        x_eta_arc: Arc<Array2<f64>>,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn gam_problem::HyperOperator>>, String> {
        validate_block_count::<GamlssError>("BinomialMeanWiggleFamily", 2, block_states.len())?;
        let eta = &block_states[Self::BLOCK_ETA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        if eta.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialMeanWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let geom = self.wiggle_geometry(eta.view(), betaw.view())?;
        let p_eta = x_eta_arc.ncols();
        let pw = geom.basis.ncols();
        let total = p_eta + pw;
        if d_beta_u_flat.len() != total || d_beta_v_flat.len() != total {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "BinomialMeanWiggleFamily joint second d_beta length mismatch: got {} and {}, expected {}",
                d_beta_u_flat.len(),
                d_beta_v_flat.len(),
                total
            ) }.into());
        }
        let u_eta = d_beta_u_flat.slice(s![0..p_eta]).to_owned();
        let v_eta = d_beta_v_flat.slice(s![0..p_eta]).to_owned();
        let uw = d_beta_u_flat.slice(s![p_eta..total]).to_owned();
        let vw = d_beta_v_flat.slice(s![p_eta..total]).to_owned();

        let xi_u = fast_av(x_eta_arc.as_ref(), &u_eta);
        let xi_v = fast_av(x_eta_arc.as_ref(), &v_eta);
        let phi_u = fast_av(&geom.basis, &uw);
        let phi_v = fast_av(&geom.basis, &vw);
        let b1u = fast_av(&geom.basis_d1, &uw);
        let b1v = fast_av(&geom.basis_d1, &vw);
        let b2u = fast_av(&geom.basis_d2, &uw);
        let b2v = fast_av(&geom.basis_d2, &vw);
        let b3u = fast_av(&geom.basis_d3, &uw);
        let b3v = fast_av(&geom.basis_d3, &vw);

        let mut coeff_eta = Array1::<f64>::zeros(n);
        let mut coeff_etaw_b = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d1 = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d2 = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d3 = Array1::<f64>::zeros(n);
        let mut coeff_ww_bb = Array1::<f64>::zeros(n);
        let mut coeff_ww_db = Array1::<f64>::zeros(n);
        let mut coeff_ww_ddb = Array1::<f64>::zeros(n);
        let mut coeff_ww_dd = Array1::<f64>::zeros(n);

        for row in 0..n {
            let q = eta[row] + etaw[row];
            let (m1, m2, m3) = self.neglog_q_derivatives(self.y[row], self.weights[row], q)?;
            let m4 = self.neglog_q_fourth_derivative(self.y[row], self.weights[row], q)?;
            let a = geom.dq_dq0[row];
            let b = geom.d2q_dq02[row];
            let c = geom.d3q_dq03[row];
            let d = geom.d4q_dq04[row];

            let q_u = a * xi_u[row] + phi_u[row];
            let a_u = b * xi_u[row] + b1u[row];
            let b_u = c * xi_u[row] + b2u[row];
            let q_v = a * xi_v[row] + phi_v[row];
            let a_v = b * xi_v[row] + b1v[row];
            let b_v = c * xi_v[row] + b2v[row];
            let q_uv = b * xi_u[row] * xi_v[row] + b1u[row] * xi_v[row] + b1v[row] * xi_u[row];
            let a_uv = c * xi_u[row] * xi_v[row] + b2u[row] * xi_v[row] + b2v[row] * xi_u[row];
            let b_uv = d * xi_u[row] * xi_v[row] + b3u[row] * xi_v[row] + b3v[row] * xi_u[row];

            coeff_eta[row] = second_directionalhessian_coeff_fromobjective_q_terms(
                m1, m2, m3, m4, q_u, q_v, q_uv, a, a, b, a_u, a_v, a_u, a_v, a_uv, a_uv, b_u, b_v,
                b_uv,
            );
            let d2_c_b = m4 * q_u * q_v * a + m3 * (q_uv * a + q_u * a_v + q_v * a_u) + m2 * a_uv;
            let dc_b_u = m3 * q_u * a + m2 * a_u;
            let dc_b_v = m3 * q_v * a + m2 * a_v;
            let c_b_static = m2 * a;
            let d2_c_b1 = m3 * q_u * q_v + m2 * q_uv;
            let dc_b1_u = m2 * q_u;
            let dc_b1_v = m2 * q_v;

            coeff_etaw_b[row] = d2_c_b;
            coeff_etaw_d1[row] = dc_b_u * xi_v[row] + dc_b_v * xi_u[row] + d2_c_b1;
            coeff_etaw_d2[row] =
                c_b_static * xi_u[row] * xi_v[row] + dc_b1_u * xi_v[row] + dc_b1_v * xi_u[row];
            coeff_etaw_d3[row] = m1 * xi_u[row] * xi_v[row];

            let dw = m2;
            let dw_u = m3 * q_u;
            let dw_v = m3 * q_v;
            let dw_uv = m4 * q_u * q_v + m3 * q_uv;
            let xixj = xi_u[row] * xi_v[row];
            coeff_ww_bb[row] = dw_uv;
            coeff_ww_db[row] = dw_v * xi_u[row] + dw_u * xi_v[row];
            coeff_ww_ddb[row] = dw * xixj;
            coeff_ww_dd[row] = 2.0 * dw * xixj;
        }

        Ok(Some(Arc::new(RowCoeffOperator::from_directions(
            vec![p_eta, pw],
            vec![
                (0, x_eta_arc),
                (1, Arc::new(geom.basis)),
                (1, Arc::new(geom.basis_d1)),
                (1, Arc::new(geom.basis_d2)),
                (1, Arc::new(geom.basis_d3)),
            ],
            vec![
                (0, 0, coeff_eta),
                (0, 1, coeff_etaw_b),
                (0, 2, coeff_etaw_d1),
                (0, 3, coeff_etaw_d2),
                (0, 4, coeff_etaw_d3),
                (1, 1, coeff_ww_bb),
                (1, 2, coeff_ww_db),
                (1, 3, coeff_ww_ddb),
                (2, 2, coeff_ww_dd),
            ],
            n,
        ))))
    }

    /// Build the [`BlockEffectiveJacobian`] for block `block_idx`.
    ///
    /// `BinomialMeanWiggle` has a single location output (n_outputs = 1):
    /// - block 0 (eta):    output 0 = design rows
    /// - block 1 (wiggle): all zeros (nonlinear link modulation)
    pub fn block_effective_jacobian(
        specs: &[ParameterBlockSpec],
        block_idx: usize,
    ) -> Result<Box<dyn BlockEffectiveJacobian>, String> {
        crate::block_layout::block_jacobian::AdditiveWiggleBlockLayout {
            family: "BinomialMeanWiggleFamily",
            n_outputs: 1,
            additive_blocks: &[Self::BLOCK_ETA],
            wiggle_block: Some(Self::BLOCK_WIGGLE),
        }
        .block_effective_jacobian(specs, block_idx)
    }
}

impl CustomFamily for BinomialMeanWiggleFamily {
    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    /// The binomial mean link-wiggle refit must NOT carry the full-span
    /// Jeffreys/Firth augmentation, for the same structural reason
    /// `GaussianLocationScaleWiggleFamily` opts out (#684–#688) — and the
    /// binomial wiggle hits it harder. This is a *second-stage* refit: the
    /// pilot binomial mean fit has already converged through the ordinary
    /// PIRLS path (which is itself un-Firthed unless the user opts in — the
    /// standard binomial fit logs `firth=false` / `jeffreys_logdet=none`), so
    /// the wiggle refit only adds a *penalized*, *monotone-constrained*
    /// I-spline link-shape correction `q = η + B(η)·β_w` around an
    /// already-finite mode. Two failure modes follow from leaving the term on
    /// (default `true`):
    ///
    /// 1. **Phantom stationarity residual.** When `H_pen` is full-rank and
    ///    well-conditioned (the normal case — e.g. `cond ≈ 5.5e2` on the #872
    ///    pure-probit repro) the Jeffreys gate smooth-steps the curvature
    ///    `H_Φ → 0`, but the matching score `∇Φ` does not vanish in lock-step,
    ///    so it leaks a nonzero `|∇L − Sβ + ∇Φ|` into the inner joint-Newton
    ///    KKT residual. The certificate then refuses every iterate and the
    ///    outer REML rejects all seeds (exactly the #684–#688 abort signature).
    /// 2. **Saturation barrier / divergence.** `−Φ = −½log|I_J|` is folded into
    ///    the objective and `∇Φ ∝ I_J⁻¹` into the gradient. The I-spline warp
    ///    can drive the binomial linear predictor toward saturation, where the
    ///    reduced Fisher information `I_J` goes singular: `−Φ → +∞` and
    ///    `∇Φ → ∞`. The augmented objective grows a barrier that the joint
    ///    Newton diverges into — the #872 repro runs the full 1200-cycle budget
    ///    with the augmented objective pinned at ~4.6e9 and the augmented
    ///    residual at ~5.8e9 while the plain data gradient is only ~2.3e2,
    ///    aborting the documented `link(type=flexible(...)) + linkwiggle(...)`
    ///    fit.
    ///
    /// Separation robustness is not lost: the wiggle block carries both a
    /// difference penalty (λ selected by REML) and a hard non-negativity
    /// constraint, and the underlying mean is fit by the pilot; a penalized,
    /// constrained refit around a finite pilot mode does not run away to
    /// `β → ∞` the way an unpenalized MLE can. Turning the term off here makes
    /// the wiggle refit consistent with the un-Firthed pilot and removes the
    /// phantom residual that blocked convergence.
    fn joint_jeffreys_term_required(&self) -> bool {
        false
    }

    fn coefficient_hessian_cost(&self, specs: &[ParameterBlockSpec]) -> u64 {
        // The mean-wiggle Hessian is exposed as a row-coefficient operator,
        // so the hot representation cost is one Θ(n · (p_eta + p_w)) HVP
        // rather than dense Θ(n · (p_eta + p_w)^2) assembly.
        let p_total = specs
            .iter()
            .map(|s| s.design.ncols() as u64)
            .fold(0u64, |acc, p| acc.saturating_add(p));
        (self.y.len() as u64).saturating_mul(p_total.max(1))
    }

    fn block_linear_constraints(
        &self,
        _: &[ParameterBlockState],
        block_idx: usize,
        spec: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        if block_idx != Self::BLOCK_WIGGLE {
            return Ok(None);
        }
        // Frozen-basis residualization preserves the original I-spline
        // coefficient coordinate, so the same exact non-negative cone applies
        // in both dynamic and frozen modes. For β_w ≥ 0, the M-spline
        // derivative gives dq/dη = 1 + B'(η)·β_w ≥ 1 everywhere.
        Ok(monotone_wiggle_nonnegative_constraints(spec.design.ncols()))
    }

    fn post_update_block_beta(
        &self,
        _: &[ParameterBlockState],
        block_idx: usize,
        block_spec: &ParameterBlockSpec,
        beta: Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        assert!(!block_spec.name.is_empty());
        if block_idx != Self::BLOCK_WIGGLE {
            return Ok(beta);
        }
        let beta = project_monotone_wiggle_beta_nonnegative(beta);
        validate_monotone_wiggle_beta_nonnegative(&beta, "BinomialMeanWiggleFamily post-update")?;
        Ok(beta)
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        validate_block_count::<GamlssError>("BinomialMeanWiggleFamily", 2, block_states.len())?;
        let eta = &block_states[Self::BLOCK_ETA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        if eta.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialMeanWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        // Frozen-basis (#1596): with `B⊥` pinned, `q = η + B⊥·β_w` so
        // `∂q/∂η = 1` exactly (the warp offset is constant in the live `η`).
        // Otherwise `∂q/∂η = 1 + B'(η)·β_w`, the dynamic warp slope.
        let dq_dq0 = if self.frozen_warp_design.is_some() {
            Array1::<f64>::ones(n)
        } else {
            self.wiggle_dq_dq0(eta.view(), betaw.view())?
        };
        if dq_dq0.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialMeanWiggleFamily dq/dq0 length mismatch: got {}, expected {}",
                    dq_dq0.len(),
                    n
                ),
            }
            .into());
        }

        let mut ll = 0.0;
        let mut z_eta = Array1::<f64>::zeros(n);
        let mut w_eta = Array1::<f64>::zeros(n);
        let mut z_wiggle = Array1::<f64>::zeros(n);
        let mut w_wiggle = Array1::<f64>::zeros(n);
        for i in 0..n {
            let q = eta[i] + etaw[i];
            let (mu_q, d1_q) = inverse_link_mu_d1_for_inverse_link(&self.link_kind, q)
                .map_err(|e| format!("fixed-link wiggle inverse-link evaluation failed: {e}"))?;
            let yi = self.y[i];
            let wi = self.weights[i];
            ll += binomial_location_scale_log_likelihood(yi, wi, q, &self.link_kind, mu_q)?;

            let mu = mu_q.clamp(1e-12, 1.0 - 1e-12);
            let var = (mu * (1.0 - mu)).max(MIN_PROB);
            let dmu_deta = d1_q * dq_dq0[i];
            let dmu_dw = d1_q;
            if wi == 0.0 || !var.is_finite() {
                z_eta[i] = eta[i];
                z_wiggle[i] = etaw[i];
                continue;
            }

            if dmu_deta.is_finite() {
                w_eta[i] = floor_positiveweight(wi * (dmu_deta * dmu_deta / var), MIN_WEIGHT);
                z_eta[i] = eta[i] + (yi - mu) / signedwith_floor(dmu_deta, MIN_DERIV);
            } else {
                z_eta[i] = eta[i];
            }

            if dmu_dw.is_finite() {
                w_wiggle[i] = floor_positiveweight(wi * (dmu_dw * dmu_dw / var), MIN_WEIGHT);
                z_wiggle[i] = etaw[i] + (yi - mu) / signedwith_floor(dmu_dw, MIN_DERIV);
            } else {
                z_wiggle[i] = etaw[i];
            }
        }

        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: vec![
                BlockWorkingSet::diagonal_checked(z_eta, w_eta)?,
                BlockWorkingSet::diagonal_checked(z_wiggle, w_wiggle)?,
            ],
        })
    }

    fn exact_newton_joint_gradient_evaluation(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<ExactNewtonJointGradientEvaluation>, String> {
        // Assemble the exact joint score from the per-block IRLS working sets
        // (X_bᵀ(w⊙(z−η)) per block), the same source of truth the inner
        // joint-Newton RHS uses — consistent with the family's explicit joint
        // Hessian and matching FD of the log-likelihood.
        let eval = self.evaluate(block_states)?;
        gamlss_joint_gradient_from_working_sets(&eval, specs, block_states).map(Some)
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
                reason: "wiggle geometry requires eta block".to_string(),
            }
            .into());
        }
        let eta = &block_states[Self::BLOCK_ETA].eta;
        if eta.len() != self.y.len() {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialMeanWiggleFamily eta size mismatch".to_string(),
            }
            .into());
        }
        // Frozen-basis (#1596): return the pinned, identifiable warp design
        // `B⊥ = (I-P_X)B(η̂)` rather than the live `B(η)`. `B⊥` is constant across
        // inner cycles, so the engine rebuilds the *same* matrix every cycle —
        // the death-spiral source (a basis that moves under the line search) is
        // gone, while the dynamic-geometry plumbing is preserved unchanged.
        let x = match self.frozen_warp_design.as_ref() {
            Some(frozen) => frozen.as_ref().clone(),
            None => self.wiggle_design(eta.view())?,
        };
        if x.ncols() != spec.design.ncols() {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "dynamic wiggle design col mismatch: got {}, expected {}",
                    x.ncols(),
                    spec.design.ncols()
                ),
            }
            .into());
        }
        let nrows = x.nrows();
        Ok((
            DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(x)),
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
        let x_eta = self.dense_eta_design_fromspecs(specs)?.into_owned();
        let workspace =
            BinomialMeanWiggleHessianWorkspace::new(self.clone(), block_states.to_vec(), x_eta)?;
        Ok(Some(Arc::new(workspace)))
    }

    fn inner_coefficient_hessian_hvp_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        self.dense_eta_design_fromspecs(specs).is_ok()
    }

    fn exact_newton_joint_hessian_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Array2<f64>>, String> {
        validate_block_count::<GamlssError>("BinomialMeanWiggleFamily", 2, block_states.len())?;
        let x_eta = self.dense_eta_design_fromspecs(specs)?;
        let eta = &block_states[Self::BLOCK_ETA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        if eta.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialMeanWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let geom = self.wiggle_geometry(eta.view(), betaw.view())?;
        let p_eta = x_eta.ncols();
        let pw = geom.basis.ncols();
        let mut coeff_eta = Array1::<f64>::zeros(n);
        let mut coeff_etaw_b = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d1 = Array1::<f64>::zeros(n);
        let mut coeff_ww = Array1::<f64>::zeros(n);
        for row in 0..n {
            let q = eta[row] + etaw[row];
            let (m1, m2, _) = self.neglog_q_derivatives(self.y[row], self.weights[row], q)?;
            let a = geom.dq_dq0[row];
            let b = geom.d2q_dq02[row];
            coeff_eta[row] = hessian_coeff_fromobjective_q_terms(m1, m2, a, a, b);
            coeff_etaw_b[row] = m2 * a;
            coeff_etaw_d1[row] = m1;
            coeff_ww[row] = m2;
        }
        let h_eta_eta = xt_diag_x_dense(&x_eta, &coeff_eta)?;
        let h_eta_w = xt_diag_y_dense(&x_eta, &coeff_etaw_b, &geom.basis)?
            + &xt_diag_y_dense(&x_eta, &coeff_etaw_d1, &geom.basis_d1)?;
        let h_ww = xt_diag_x_dense(&geom.basis, &coeff_ww)?;
        assert_eq!(h_eta_eta.nrows(), p_eta);
        assert_eq!(h_ww.nrows(), pw);
        Ok(Some(binomial_pack_mean_wiggle_joint_symmetrichessian(
            &h_eta_eta, &h_eta_w, &h_ww,
        )))
    }

    fn exact_newton_joint_hessian_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        validate_block_count::<GamlssError>("BinomialMeanWiggleFamily", 2, block_states.len())?;
        let x_eta = self.dense_eta_design_fromspecs(specs)?;
        let eta = &block_states[Self::BLOCK_ETA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        if eta.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialMeanWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let geom = self.wiggle_geometry(eta.view(), betaw.view())?;
        let p_eta = x_eta.ncols();
        let pw = geom.basis.ncols();
        if d_beta_flat.len() != p_eta + pw {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialMeanWiggleFamily joint d_beta length mismatch: got {}, expected {}",
                    d_beta_flat.len(),
                    p_eta + pw
                ),
            }
            .into());
        }
        let u_eta = d_beta_flat.slice(s![0..p_eta]).to_owned();
        let uw = d_beta_flat.slice(s![p_eta..p_eta + pw]).to_owned();
        let xi = x_eta.dot(&u_eta);
        let phi = geom.basis.dot(&uw);
        let basis1_u = geom.basis_d1.dot(&uw);
        let basis2_u = geom.basis_d2.dot(&uw);

        let mut coeff_eta = Array1::<f64>::zeros(n);
        let mut coeff_etaw_b = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d1 = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d2 = Array1::<f64>::zeros(n);
        let mut coeff_ww_bb = Array1::<f64>::zeros(n);
        let mut coeff_ww_db = Array1::<f64>::zeros(n);
        for row in 0..n {
            let q = eta[row] + etaw[row];
            let (m1, m2, m3) = self.neglog_q_derivatives(self.y[row], self.weights[row], q)?;
            let a = geom.dq_dq0[row];
            let b = geom.d2q_dq02[row];
            let c = geom.d3q_dq03[row];
            let q_u = a * xi[row] + phi[row];
            let a_u = b * xi[row] + basis1_u[row];
            let b_u = c * xi[row] + basis2_u[row];
            coeff_eta[row] = directionalhessian_coeff_fromobjective_q_terms(
                m1, m2, m3, q_u, a, a, b, a_u, a_u, b_u,
            );
            coeff_etaw_b[row] = m3 * q_u * a + m2 * a_u;
            coeff_etaw_d1[row] = m2 * (a * xi[row] + q_u);
            coeff_etaw_d2[row] = m1 * xi[row];
            coeff_ww_bb[row] = m3 * q_u;
            coeff_ww_db[row] = m2 * xi[row];
        }

        let d_h_eta_eta = xt_diag_x_dense(&x_eta, &coeff_eta)?;
        let d_h_eta_w = xt_diag_y_dense(&x_eta, &coeff_etaw_b, &geom.basis)?
            + &xt_diag_y_dense(&x_eta, &coeff_etaw_d1, &geom.basis_d1)?
            + &xt_diag_y_dense(&x_eta, &coeff_etaw_d2, &geom.basis_d2)?;
        let a_ww = xt_diag_y_dense(&geom.basis_d1, &coeff_ww_db, &geom.basis)?;
        let d_h_ww = xt_diag_x_dense(&geom.basis, &coeff_ww_bb)? + &a_ww + a_ww.t();
        Ok(Some(binomial_pack_mean_wiggle_joint_symmetrichessian(
            &d_h_eta_eta,
            &d_h_eta_w,
            &d_h_ww,
        )))
    }

    /// Exact second-order directional derivative D²H[u,v] of the joint Hessian
    /// for the BinomialMeanWiggle two-block model (eta, wiggle).
    ///
    /// # Mathematical derivation
    ///
    /// The negative log-likelihood Hessian element for indices (a, b) in the
    /// joint coefficient vector is:
    ///
    ///   H_ab = m2 * q_a * q_b + m1 * q_ab
    ///
    /// where m_k = d^k F / dq^k (k-th derivative of the negative log-likelihood
    /// w.r.t. the effective predictor q), q_a = dq/d(beta_a), and q_ab =
    /// d²q/(d(beta_a) d(beta_b)).
    ///
    /// The effective predictor is q = q0 + w(q0) where q0 = X_eta * beta_eta
    /// and w(q0) = B(q0) * beta_w is the link wiggle.  Write:
    ///   a = dq/dq0 = 1 + B'·beta_w       (geometry first derivative)
    ///   b = d²q/dq0² = B''·beta_w         (geometry second derivative)
    ///   c = d³q/dq0³ = B'''·beta_w        (geometry third derivative)
    ///   d = d⁴q/dq0⁴ = B''''·beta_w       (geometry fourth derivative)
    ///
    /// For a perturbation direction u = (u_eta, u_w), the chain-rule
    /// perturbations are:
    ///   q_u   = a·xi_u + phi_u             (first-order predictor perturbation)
    ///   a_u   = b·xi_u + basis1_u          (perturbation of geometry factor a)
    ///   b_u   = c·xi_u + basis2_u          (perturbation of geometry factor b)
    ///   c_u   = d·xi_u + basis3_u          (perturbation of geometry factor c)
    ///
    /// where xi_u = X_eta·u_eta, phi_u = B·u_w, basis_k_u = B^(k)·u_w.
    ///
    /// Mixed second-order perturbations (u,v) are:
    ///   q_uv  = b·xi_u·xi_v + basis1_u·xi_v + basis1_v·xi_u
    ///   a_uv  = c·xi_u·xi_v + basis2_u·xi_v + basis2_v·xi_u
    ///   b_uv  = d·xi_u·xi_v + basis3_u·xi_v + basis3_v·xi_u
    ///
    /// ## Block decomposition
    ///
    /// **eta-eta block** (X_eta' diag(coeff) X_eta):
    ///   The Hessian element for eta indices (i,j) factors as
    ///     H(eta_i, eta_j) = [m2·a² + m1·b] · x_eta(i)·x_eta(j)
    ///   so D²H_eta_eta[u,v] = X_eta' diag(coeff_eta) X_eta
    ///   where coeff_eta uses `second_directionalhessian_coeff_fromobjective_q_terms`
    ///   with q_a=a, q_b=a, q_ab=b and their chain-rule perturbations.
    ///
    /// **eta-w block** (X_eta' diag(...) [B, B', B'', B''']):
    ///   The static Hessian is:
    ///     H(eta_i, w_j) = (m2·a)·x_eta(i)·B_j + m1·x_eta(i)·B'_j
    ///   Taking D²[u,v] requires differentiating both the scalar coefficients
    ///   (m2·a, m1) and the basis matrices (B, B' depend on q0 via the chain
    ///   rule dB_j/du = B'_j·xi_u).  The full product rule gives four basis-matrix
    ///   tiers: B, B', B'', B'''.
    ///
    /// **w-w block** (B' diag(...) B, etc.):
    ///   The static Hessian is H(w_i, w_j) = m2·B_i·B_j.
    ///   D²[u,v] expands via the product rule on m2, B_i, B_j, each of which
    ///   depends on beta through q and q0.  This gives terms involving
    ///   B·B, B'·B, B'·B', and B''·B (all symmetrised).
    fn exact_newton_joint_hessian_second_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        validate_block_count::<GamlssError>("BinomialMeanWiggleFamily", 2, block_states.len())?;
        let x_eta = self.dense_eta_design_fromspecs(specs)?;
        let eta = &block_states[Self::BLOCK_ETA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        if eta.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialMeanWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let geom = self.wiggle_geometry(eta.view(), betaw.view())?;
        let p_eta = x_eta.ncols();
        let pw = geom.basis.ncols();
        let total = p_eta + pw;
        if d_beta_u_flat.len() != total || d_beta_v_flat.len() != total {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "BinomialMeanWiggleFamily joint second d_beta length mismatch: got {} and {}, expected {}",
                d_beta_u_flat.len(),
                d_beta_v_flat.len(),
                total
            ) }.into());
        }

        // Split directions into eta and wiggle components.
        let u_eta = d_beta_u_flat.slice(s![0..p_eta]).to_owned();
        let v_eta = d_beta_v_flat.slice(s![0..p_eta]).to_owned();
        let uw = d_beta_u_flat.slice(s![p_eta..total]).to_owned();
        let vw = d_beta_v_flat.slice(s![p_eta..total]).to_owned();

        // Per-row linear-predictor perturbations from each direction.
        let xi_u = x_eta.dot(&u_eta); // eta perturbation in direction u
        let xi_v = x_eta.dot(&v_eta); // eta perturbation in direction v
        let phi_u = geom.basis.dot(&uw); // direct wiggle basis, direction u
        let phi_v = geom.basis.dot(&vw); // direct wiggle basis, direction v
        let b1u = geom.basis_d1.dot(&uw); // first-derivative basis, direction u
        let b1v = geom.basis_d1.dot(&vw);
        let b2u = geom.basis_d2.dot(&uw); // second-derivative basis, direction u
        let b2v = geom.basis_d2.dot(&vw);
        let b3u = geom.basis_d3.dot(&uw); // third-derivative basis, direction u
        let b3v = geom.basis_d3.dot(&vw);

        // Per-row chain-rule perturbations of q, a = dq/dq0, b = d²q/dq0²:
        //   q_u = a·xi_u + phi_u
        //   a_u = b·xi_u + basis1_u
        //   b_u = c·xi_u + basis2_u
        //   c_u = d·xi_u + basis3_u
        // Mixed second-order perturbations:
        //   q_uv = b·xi_u·xi_v + basis1_u·xi_v + basis1_v·xi_u
        //   a_uv = c·xi_u·xi_v + basis2_u·xi_v + basis2_v·xi_u
        //   b_uv = d·xi_u·xi_v + basis3_u·xi_v + basis3_v·xi_u

        // Scaled basis matrices for the cross-product terms in the w-w and eta-w
        // blocks (same pattern as GaussianLocationScaleWiggleFamily).
        let basis_u = scale_matrix_rows(&geom.basis_d1, &xi_u)?; // dB/du = B'·xi_u
        let basis_v = scale_matrix_rows(&geom.basis_d1, &xi_v)?; // dB/dv = B'·xi_v
        let basis_uv = scale_matrix_rows(&geom.basis_d2, &(&xi_u * &xi_v))?; // d²B/dudv = B''·xi_u·xi_v
        // Per-row coefficient arrays for assembling the block-matrix products.
        let mut coeff_eta = Array1::<f64>::zeros(n);

        // Coefficients for the eta-w block: X_eta' diag(c_*) M where M ∈ {B, B', B'', B'''}
        //
        // The static cross-Hessian is:
        //   H(eta_i, w_j) = (m2·a)·x_i·B_j + m1·x_i·B'_j
        // where B_j and B'_j are row evaluations of basis column j.
        //
        // Write C_B = m2·a (scalar coefficient multiplying B in the cross block)
        // and   C_B1 = m1  (scalar coefficient multiplying B' in the cross block).
        //
        // Product rule on C_B·B:
        //   d(C_B·B)/du = (dC_B/du)·B + C_B·B'·xi_u
        //   d²(C_B·B)/dudv = (d²C_B/dudv)·B + (dC_B/du)·B'·xi_v
        //                   + (dC_B/dv)·B'·xi_u + C_B·B''·xi_u·xi_v
        //
        // Product rule on C_B1·B':
        //   d²(C_B1·B')/dudv = (d²C_B1/dudv)·B' + (dC_B1/du)·B''·xi_v
        //                     + (dC_B1/dv)·B''·xi_u + C_B1·B'''·xi_u·xi_v
        //
        // Derivatives of the scalar coefficients:
        //   C_B  = m2·a
        //   dC_B/du  = m3·q_u·a + m2·a_u
        //   dC_B/dv  = m3·q_v·a + m2·a_v
        //   d²C_B/dudv = m4·q_u·q_v·a + m3·(q_uv·a + q_u·a_v + q_v·a_u) + m2·a_uv
        //
        //   C_B1 = m1
        //   dC_B1/du = m2·q_u
        //   dC_B1/dv = m2·q_v
        //   d²C_B1/dudv = m3·q_u·q_v + m2·q_uv
        //
        // Grouping by basis-matrix tier:
        //   B:   d²C_B/dudv
        //   B':  (dC_B/du)·xi_v + (dC_B/dv)·xi_u + d²C_B1/dudv
        //   B'': C_B·xi_u·xi_v + (dC_B1/du)·xi_v + (dC_B1/dv)·xi_u
        //   B''': C_B1·xi_u·xi_v
        let mut coeff_etaw_b = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d1 = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d2 = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d3 = Array1::<f64>::zeros(n);

        // Coefficients for the w-w block.
        //
        // The static w-w Hessian is:
        //   H(w_i, w_j) = m2·B_i·B_j
        //
        // Note: there is no m1·q_ij term because d²q/(d(beta_w_i) d(beta_w_j)) = 0
        // (the basis vectors B_i enter q linearly in beta_w).
        //
        // Product rule on m2·B_i·B_j, treating each factor as depending on beta:
        //   d²(m2·B_i·B_j)/dudv
        //     = (d²m2/dudv)·B_i·B_j                        → B'diag B  (symmetrised)
        //     + (dm2/du)·(B'_i·xi_v·B_j + B_i·B'_j·xi_v)  → dw_u terms
        //     + (dm2/dv)·(B'_i·xi_u·B_j + B_i·B'_j·xi_u)  → dw_v terms
        //     + m2·(B''_i·xi_u·xi_v·B_j + B'_i·xi_u·B'_j·xi_v
        //          + B'_i·xi_v·B'_j·xi_u + B_i·B''_j·xi_u·xi_v)
        //
        // where dm2/du = m3·q_u, dm2/dv = m3·q_v, d²m2/dudv = m4·q_u·q_v + m3·q_uv.
        //
        // Following the Gaussian LS wiggle pattern, we express this via:
        //   xt_diag_x_dense(B, dw_uv)                    — coeff: d²m2
        //   xt_diag_y_dense(basis_u, dw_v, B) + transpose — dB/du weighted by dm2/dv
        //   xt_diag_y_dense(basis_v, dw_u, B) + transpose — dB/dv weighted by dm2/du
        //   xt_diag_y_dense(basis_uv, w, B) + transpose   — d²B/dudv weighted by m2
        //   xt_diag_y_dense(basis_u, w, basis_v) + transpose — dB/du·dB/dv weighted by m2
        let mut dw = Array1::<f64>::zeros(n);
        let mut dw_u = Array1::<f64>::zeros(n);
        let mut dw_v = Array1::<f64>::zeros(n);
        let mut dw_uv = Array1::<f64>::zeros(n);

        for row in 0..n {
            let q = eta[row] + etaw[row];
            let (m1, m2, m3) = self.neglog_q_derivatives(self.y[row], self.weights[row], q)?;
            let m4 = self.neglog_q_fourth_derivative(self.y[row], self.weights[row], q)?;
            let a = geom.dq_dq0[row];
            let b = geom.d2q_dq02[row];
            let c = geom.d3q_dq03[row];
            let d = geom.d4q_dq04[row];

            // Chain-rule perturbations in direction u.
            let q_u = a * xi_u[row] + phi_u[row];
            let a_u = b * xi_u[row] + b1u[row];
            let b_u = c * xi_u[row] + b2u[row];

            // Chain-rule perturbations in direction v.
            let q_v = a * xi_v[row] + phi_v[row];
            let a_v = b * xi_v[row] + b1v[row];
            let b_v = c * xi_v[row] + b2v[row];

            // Mixed second-order perturbations.
            let q_uv = b * xi_u[row] * xi_v[row] + b1u[row] * xi_v[row] + b1v[row] * xi_u[row];
            let a_uv = c * xi_u[row] * xi_v[row] + b2u[row] * xi_v[row] + b2v[row] * xi_u[row];
            let b_uv = d * xi_u[row] * xi_v[row] + b3u[row] * xi_v[row] + b3v[row] * xi_u[row];

            // ── eta-eta block ──
            // H(eta_i, eta_j) uses q_a = a, q_b = a, q_ab = b (absorbing x_eta
            // into the matrix product).  The perturbations of these geometric
            // quantities are: dq_a/du = a_u, dq_b/du = a_u (since q_a = q_b = a),
            // dq_ab/du = b_u (since q_ab = b), and analogously for v.
            coeff_eta[row] = second_directionalhessian_coeff_fromobjective_q_terms(
                m1, m2, m3, m4, q_u, q_v, q_uv, a, a, b, // q_a, q_b, q_ab
                a_u, a_v, // dq_a_u, dq_a_v
                a_u, a_v, // dq_b_u, dq_b_v  (q_b = a so same perturbation)
                a_uv, a_uv, // d2q_a_uv, d2q_b_uv
                b_u, b_v,  // dq_ab_u, dq_ab_v  (q_ab = b)
                b_uv, // d2q_ab_uv
            );

            // ── eta-w block coefficients ──
            // See the derivation in the docstring above.  We group by which basis
            // matrix tier (B, B', B'', B''') the coefficient multiplies.

            // d²(m2·a)/dudv
            let d2_c_b = m4 * q_u * q_v * a + m3 * (q_uv * a + q_u * a_v + q_v * a_u) + m2 * a_uv;
            // d(m2·a)/du and d(m2·a)/dv
            let dc_b_u = m3 * q_u * a + m2 * a_u;
            let dc_b_v = m3 * q_v * a + m2 * a_v;
            // m2·a (static coefficient for B in the cross block)
            let c_b_static = m2 * a;
            // d²(m1)/dudv
            let d2_c_b1 = m3 * q_u * q_v + m2 * q_uv;
            // d(m1)/du and d(m1)/dv
            let dc_b1_u = m2 * q_u;
            let dc_b1_v = m2 * q_v;

            coeff_etaw_b[row] = d2_c_b;
            coeff_etaw_d1[row] = dc_b_u * xi_v[row] + dc_b_v * xi_u[row] + d2_c_b1;
            coeff_etaw_d2[row] =
                c_b_static * xi_u[row] * xi_v[row] + dc_b1_u * xi_v[row] + dc_b1_v * xi_u[row];
            coeff_etaw_d3[row] = m1 * xi_u[row] * xi_v[row];

            // ── w-w block coefficients ──
            // The w-w static Hessian coefficient is m2 (for B'diag B).
            dw[row] = m2;
            dw_u[row] = m3 * q_u;
            dw_v[row] = m3 * q_v;
            dw_uv[row] = m4 * q_u * q_v + m3 * q_uv;
        }

        // ── Assemble eta-eta block ──
        let d2_h_eta_eta = xt_diag_x_dense(&x_eta, &coeff_eta)?;

        // ── Assemble eta-w block ──
        // The second-order directional derivative of the cross block H_eta_w is:
        //   d²H_eta_w[u,v] = X_eta' diag(coeff_etaw_b)  B
        //                   + X_eta' diag(coeff_etaw_d1) B'
        //                   + X_eta' diag(coeff_etaw_d2) B''
        //                   + X_eta' diag(coeff_etaw_d3) B'''
        let d2_h_eta_w = xt_diag_y_dense(&x_eta, &coeff_etaw_b, &geom.basis)?
            + &xt_diag_y_dense(&x_eta, &coeff_etaw_d1, &geom.basis_d1)?
            + &xt_diag_y_dense(&x_eta, &coeff_etaw_d2, &geom.basis_d2)?
            + &xt_diag_y_dense(&x_eta, &coeff_etaw_d3, &geom.basis_d3)?;

        // ── Assemble w-w block ──
        // Following the Gaussian LS wiggle pattern (lines 6351-6363), the w-w
        // second directional derivative is assembled from scaled basis products:
        //
        //   d²(m2·B_i·B_j)/dudv decomposition:
        //     (d²m2)     · B_i·B_j        → xt_diag_x(B, dw_uv)
        //     (dm2/du)   · dB_j/dv · B_i  → xt_diag_y(basis_v, dw_u, B) + transpose
        //     (dm2/dv)   · dB_j/du · B_i  → xt_diag_y(basis_u, dw_v, B) + transpose
        //     m2 · d²B_j/dudv · B_i       → xt_diag_y(basis_uv, dw, B) + transpose
        //     m2 · dB_i/du · dB_j/dv      → xt_diag_y(basis_u, dw, basis_v) + transpose
        let a_ab = xt_diag_y_dense(&basis_uv, &dw, &geom.basis)?;
        let a_ij = xt_diag_y_dense(&basis_u, &dw, &basis_v)?;
        let a_iwj = xt_diag_y_dense(&basis_u, &dw_v, &geom.basis)?;
        let a_jwi = xt_diag_y_dense(&basis_v, &dw_u, &geom.basis)?;
        let d2_h_ww = &a_ab
            + &a_ab.t()
            + &a_ij
            + a_ij.t()
            + &a_iwj
            + a_iwj.t()
            + &a_jwi
            + a_jwi.t()
            + &xt_diag_x_dense(&geom.basis, &dw_uv)?;

        Ok(Some(binomial_pack_mean_wiggle_joint_symmetrichessian(
            &d2_h_eta_eta,
            &d2_h_eta_w,
            &d2_h_ww,
        )))
    }

    fn exact_newton_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        validate_block_count::<GamlssError>("BinomialMeanWiggleFamily", 2, block_states.len())?;
        if derivative_blocks.len() != 2 {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "BinomialMeanWiggleFamily joint psi terms expect 2 derivative block lists, got {}",
                derivative_blocks.len()
            ) }.into());
        }
        let x_eta = self.dense_eta_design_fromspecs(specs)?;
        let eta = &block_states[Self::BLOCK_ETA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        if eta.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialMeanWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let geom = self.wiggle_geometry(eta.view(), betaw.view())?;
        let p_eta = x_eta.ncols();
        let pw = geom.basis.ncols();
        let implicit_dir =
            self.exact_newton_joint_psi_action(block_states, derivative_blocks, psi_index, p_eta)?;
        let dense_dir = if implicit_dir.is_none() {
            self.exact_newton_joint_psi_direction(
                block_states,
                derivative_blocks,
                psi_index,
                &x_eta,
            )?
        } else {
            None
        };
        let z_eta_psi = if let Some((_, ref z_eta_psi)) = implicit_dir {
            z_eta_psi
        } else if let Some(ref dir_a) = dense_dir {
            &dir_a.z_eta_psi
        } else {
            return Ok(None);
        };

        let mut objective_psi = 0.0;
        let mut score_eta_xa = Array1::<f64>::zeros(n);
        let mut score_eta_x = Array1::<f64>::zeros(n);
        let mut score_w_b = Array1::<f64>::zeros(n);
        let mut score_w_d1 = Array1::<f64>::zeros(n);

        let mut coeff_eta_eta_xx = Array1::<f64>::zeros(n);
        let mut coeff_eta_eta_xa_x = Array1::<f64>::zeros(n);
        let mut coeff_eta_w_xa_b = Array1::<f64>::zeros(n);
        let mut coeff_eta_w_x_b = Array1::<f64>::zeros(n);
        let mut coeff_eta_w_x_d1 = Array1::<f64>::zeros(n);
        let mut coeff_eta_w_xa_d1 = Array1::<f64>::zeros(n);
        let mut coeff_eta_w_x_d2 = Array1::<f64>::zeros(n);
        let mut coeff_ww_bb = Array1::<f64>::zeros(n);
        let mut coeff_ww_db = Array1::<f64>::zeros(n);

        for row in 0..n {
            let q = eta[row] + etaw[row];
            let (m1, m2, m3) = self.neglog_q_derivatives(self.y[row], self.weights[row], q)?;
            let z_a = z_eta_psi[row];
            let a = geom.dq_dq0[row];
            let b = geom.d2q_dq02[row];
            let c = geom.d3q_dq03[row];
            let q_a = a * z_a;

            objective_psi += m1 * q_a;

            score_eta_xa[row] = m1 * a;
            score_eta_x[row] = m2 * q_a * a + m1 * b * z_a;
            score_w_b[row] = m2 * q_a;
            score_w_d1[row] = m1 * z_a;

            coeff_eta_eta_xx[row] =
                m3 * q_a * a * a + m2 * (2.0 * a * b * z_a + q_a * b) + m1 * c * z_a;
            coeff_eta_eta_xa_x[row] = m2 * a * a + m1 * b;
            coeff_eta_w_xa_b[row] = m2 * a;
            coeff_eta_w_x_b[row] = m3 * q_a * a + m2 * b * z_a;
            coeff_eta_w_x_d1[row] = m2 * (a * z_a + q_a);
            coeff_eta_w_xa_d1[row] = m1;
            coeff_eta_w_x_d2[row] = m1 * z_a;
            coeff_ww_bb[row] = m3 * q_a;
            coeff_ww_db[row] = m2 * z_a;
        }

        let score_w = gam_linalg::faer_ndarray::fast_atv(&geom.basis, &score_w_b)
            + gam_linalg::faer_ndarray::fast_atv(&geom.basis_d1, &score_w_d1);

        if let Some((action, _)) = implicit_dir {
            let score_eta = action.transpose_mul(score_eta_xa.view())
                + gam_linalg::faer_ndarray::fast_atv(x_eta.as_ref(), &score_eta_x);
            let score_psi = binomial_pack_mean_wiggle_joint_score(&score_eta, &score_w);
            let x_eta_arc = shared_dense_arc(x_eta.as_ref());
            let basis_arc = Arc::new(geom.basis.clone());
            let basis_d1_arc = Arc::new(geom.basis_d1.clone());
            let basis_d2_arc = Arc::new(geom.basis_d2.clone());
            let zeros = Array1::<f64>::zeros(n);
            let operator = CustomFamilyJointPsiOperator::new(
                p_eta + pw,
                vec![
                    CustomFamilyJointDesignChannel::new(
                        0..p_eta,
                        Arc::clone(&x_eta_arc),
                        Some(action),
                    ),
                    CustomFamilyJointDesignChannel::new(
                        p_eta..p_eta + pw,
                        Arc::clone(&basis_arc),
                        None,
                    ),
                    CustomFamilyJointDesignChannel::new(
                        p_eta..p_eta + pw,
                        Arc::clone(&basis_d1_arc),
                        None,
                    ),
                    CustomFamilyJointDesignChannel::new(
                        p_eta..p_eta + pw,
                        Arc::clone(&basis_d2_arc),
                        None,
                    ),
                ],
                vec![
                    CustomFamilyJointDesignPairContribution::new(
                        0,
                        0,
                        coeff_eta_eta_xa_x.clone(),
                        coeff_eta_eta_xx.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        0,
                        1,
                        coeff_eta_w_xa_b.clone(),
                        coeff_eta_w_x_b.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        1,
                        0,
                        coeff_eta_w_xa_b.clone(),
                        coeff_eta_w_x_b.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        0,
                        2,
                        coeff_eta_w_xa_d1.clone(),
                        coeff_eta_w_x_d1.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        2,
                        0,
                        coeff_eta_w_xa_d1.clone(),
                        coeff_eta_w_x_d1.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        0,
                        3,
                        zeros.clone(),
                        coeff_eta_w_x_d2.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        3,
                        0,
                        zeros.clone(),
                        coeff_eta_w_x_d2.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        1,
                        1,
                        zeros.clone(),
                        coeff_ww_bb.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        2,
                        1,
                        zeros.clone(),
                        coeff_ww_db.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(1, 2, zeros, coeff_ww_db.clone()),
                ],
            );
            return Ok(Some(crate::custom_family::ExactNewtonJointPsiTerms {
                objective_psi,
                score_psi,
                hessian_psi: Array2::zeros((0, 0)),
                hessian_psi_operator: Some(std::sync::Arc::new(operator)),
            }));
        }

        let dir_a =
            dense_dir.expect("dense psi direction should exist when implicit direction is absent");
        let x_eta_psi = dir_a
            .x_eta_psi
            .as_ref()
            .expect("dense eta psi design should exist when implicit direction is absent");
        let score_psi = binomial_pack_mean_wiggle_joint_score(
            &(gam_linalg::faer_ndarray::fast_atv(x_eta_psi, &score_eta_xa)
                + gam_linalg::faer_ndarray::fast_atv(x_eta.as_ref(), &score_eta_x)),
            &score_w,
        );
        let a_eta_eta = xt_diag_y_dense(x_eta_psi, &coeff_eta_eta_xa_x, &x_eta)?;
        let h_eta_eta = &a_eta_eta + &a_eta_eta.t() + &xt_diag_x_dense(&x_eta, &coeff_eta_eta_xx)?;
        let h_eta_w = xt_diag_y_dense(x_eta_psi, &coeff_eta_w_xa_b, &geom.basis)?
            + &xt_diag_y_dense(&x_eta, &coeff_eta_w_x_b, &geom.basis)?
            + &xt_diag_y_dense(&x_eta, &coeff_eta_w_x_d1, &geom.basis_d1)?
            + &xt_diag_y_dense(x_eta_psi, &coeff_eta_w_xa_d1, &geom.basis_d1)?
            + &xt_diag_y_dense(&x_eta, &coeff_eta_w_x_d2, &geom.basis_d2)?;
        let a_ww = xt_diag_y_dense(&geom.basis_d1, &coeff_ww_db, &geom.basis)?;
        let h_ww = xt_diag_x_dense(&geom.basis, &coeff_ww_bb)? + &a_ww + a_ww.t();

        Ok(Some(crate::custom_family::ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi: binomial_pack_mean_wiggle_joint_symmetrichessian(
                &h_eta_eta, &h_eta_w, &h_ww,
            ),
            hessian_psi_operator: None,
        }))
    }
}

pub(crate) struct BinomialMeanWiggleHessianWorkspace {
    pub(crate) family: BinomialMeanWiggleFamily,
    pub(crate) block_states: Vec<ParameterBlockState>,
    pub(crate) x_eta: Arc<Array2<f64>>,
    pub(crate) hessian_operator: Arc<RowCoeffOperator>,
}

impl BinomialMeanWiggleHessianWorkspace {
    pub(crate) fn new(
        family: BinomialMeanWiggleFamily,
        block_states: Vec<ParameterBlockState>,
        x_eta: Array2<f64>,
    ) -> Result<Self, String> {
        let x_eta = Arc::new(x_eta);
        let hessian_operator = family.bmw_static_hessian_operator(&block_states, x_eta.clone())?;
        Ok(Self {
            family,
            block_states,
            x_eta,
            hessian_operator,
        })
    }
}

impl ExactNewtonJointHessianWorkspace for BinomialMeanWiggleHessianWorkspace {
    fn hessian_matvec_available(&self) -> bool {
        true
    }

    fn hessian_matvec(&self, v: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        Ok(Some(gam_problem::HyperOperator::mul_vec(
            self.hessian_operator.as_ref(),
            v,
        )))
    }

    fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        Ok(None)
    }

    fn directional_derivative(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(self
            .directional_derivative_operator(d_beta_flat)?
            .map(|operator| operator.to_dense()))
    }

    fn directional_derivative_operator(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn gam_problem::HyperOperator>>, String> {
        self.family
            .bmw_directional_operator(&self.block_states, self.x_eta.clone(), d_beta_flat)
    }

    fn second_directional_derivative(
        &self,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(self
            .second_directional_derivative_operator(d_beta_u_flat, d_beta_v_flat)?
            .map(|operator| operator.to_dense()))
    }

    fn second_directional_derivative_operator(
        &self,
        d_beta_u: &Array1<f64>,
        d_beta_v: &Array1<f64>,
    ) -> Result<Option<Arc<dyn gam_problem::HyperOperator>>, String> {
        self.family.bmw_second_directional_operator(
            &self.block_states,
            self.x_eta.clone(),
            d_beta_u,
            d_beta_v,
        )
    }
}

impl CustomFamilyGenerative for BinomialMeanWiggleFamily {
    fn generativespec(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<GenerativeSpec, String> {
        validate_block_count::<GamlssError>("BinomialMeanWiggleFamily", 2, block_states.len())?;
        let eta = &block_states[Self::BLOCK_ETA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        if eta.len() != self.y.len() || etaw.len() != self.y.len() {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialMeanWiggleFamily generative size mismatch".to_string(),
            }
            .into());
        }
        let mean = gamlss_rowwise_map_result(self.y.len(), |i| {
            let jet = inverse_link_jet_for_inverse_link(&self.link_kind, eta[i] + etaw[i])
                .map_err(|e| format!("fixed-link wiggle inverse-link evaluation failed: {e}"))?;
            Ok(jet.mu)
        })?;
        Ok(GenerativeSpec {
            mean,
            noise: NoiseModel::Bernoulli,
        })
    }
}

#[cfg(test)]
mod exact_frozen_monotonicity_tests {
    use super::*;

    fn frozen_family_and_wiggle_spec() -> (BinomialMeanWiggleFamily, ParameterBlockSpec) {
        let n = 4;
        let p = 3;
        let frozen = Array2::<f64>::zeros((n, p));
        let family = BinomialMeanWiggleFamily {
            y: Array1::zeros(n),
            weights: Array1::ones(n),
            link_kind: InverseLink::Standard(StandardLink::Logit),
            wiggle_knots: Array1::linspace(-1.0, 1.0, 8),
            wiggle_degree: 3,
            policy: gam_runtime::resource::ResourcePolicy::default_library(),
            frozen_warp_design: Some(Arc::new(frozen.clone())),
        };
        let spec = ParameterBlockSpec {
            name: "wiggle".to_string(),
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(frozen)),
            offset: Array1::zeros(n),
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        };
        (family, spec)
    }

    #[test]
    fn frozen_warp_keeps_exact_nonnegative_i_spline_cone() {
        let (family, spec) = frozen_family_and_wiggle_spec();
        let constraints = family
            .block_linear_constraints(&[], BinomialMeanWiggleFamily::BLOCK_WIGGLE, &spec)
            .expect("frozen constraint construction")
            .expect("wiggle block must be constrained");
        assert_eq!(constraints.a, Array2::<f64>::eye(3));
        assert_eq!(constraints.b, Array1::<f64>::zeros(3));

        let solver_slop = Array1::from_vec(vec![
            -0.5 * crate::wiggle::MONOTONE_WIGGLE_ACTIVE_SET_TOL,
            0.2,
            0.0,
        ]);
        let projected = family
            .post_update_block_beta(
                &[],
                BinomialMeanWiggleFamily::BLOCK_WIGGLE,
                &spec,
                solver_slop,
            )
            .expect("active-set slop projects onto the exact cone");
        assert_eq!(projected[0], 0.0);

        let material_violation = Array1::from_vec(vec![
            -2.0 * crate::wiggle::MONOTONE_WIGGLE_ACTIVE_SET_TOL,
            0.2,
            0.0,
        ]);
        assert!(
            family
                .post_update_block_beta(
                    &[],
                    BinomialMeanWiggleFamily::BLOCK_WIGGLE,
                    &spec,
                    material_violation,
                )
                .is_err(),
            "a material negative I-spline coefficient must be rejected"
        );
    }
}
