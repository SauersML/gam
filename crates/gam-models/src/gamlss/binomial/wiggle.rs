// Real concern-organized submodule of the gamlss family stack.
// Cross-module items are re-exported flat through the parent (`gamlss.rs`),
// so `use super::*;` makes the sibling-concern symbols this module references
// resolve through the parent namespace.
use super::*;

/// Canonical runtime-width scalar program for one binomial location-scale
/// wiggle row.  The model expression is
///
/// `q0 = -eta_t * exp(-eta_ls); q = q0 + sum_j beta_w[j] * B_j(q0)`
///
/// followed by the scalar binomial negative log likelihood.  Basis functions
/// enter as certified derivative-stack atoms; all product and composition
/// combinatorics are owned by the jet algebra.  Production Hessian lowerings
/// use the same expression with a small set of typed probe axes, keeping their
/// cost linear in the runtime wiggle width instead of materializing a dense
/// runtime-width fourth-order tensor.
pub(crate) struct BinomialLocationScaleWiggleRowProgram<'a> {
    family: &'a BinomialLocationScaleWiggleFamily,
    eta_t: &'a Array1<f64>,
    eta_ls: &'a Array1<f64>,
    etaw: &'a Array1<f64>,
    beta_w: &'a Array1<f64>,
    core: BinomialLocationScaleCore,
    /// `basis_derivatives[d][[row, j]] = d^d B_j(q0[row]) / dq0^d`.
    basis_derivatives: Vec<Array2<f64>>,
}

/// Evaluate the one canonical predictor expression over an arbitrary scalar
/// algebra.  Operation closures let both const-width `JetScalar` and
/// runtime-width `RuntimeJetScalar` instantiate this exact body.
#[inline]
fn binomial_location_scale_wiggle_predictor_expression<S>(
    primaries: &[S],
    warp_stack: Option<[f64; 5]>,
    term_count: usize,
    term: impl Fn(usize) -> (usize, [f64; 5]),
    value: impl Fn(&S) -> f64,
    add: impl Fn(&S, &S) -> S,
    mul: impl Fn(&S, &S) -> S,
    neg: impl Fn(&S) -> S,
    compose: impl Fn(&S, [f64; 5]) -> S,
) -> S {
    let neg_eta_ls = neg(&primaries[1]);
    let exp_neg_eta_ls = value(&neg_eta_ls).exp();
    let inv_sigma = compose(
        &neg_eta_ls,
        [
            exp_neg_eta_ls,
            exp_neg_eta_ls,
            exp_neg_eta_ls,
            exp_neg_eta_ls,
            exp_neg_eta_ls,
        ],
    );
    let mut q = mul(&neg(&primaries[0]), &inv_sigma);
    if let Some(stack) = warp_stack {
        q = add(&q, &compose(&q, stack));
    }
    for slot in 0..term_count {
        let (axis, stack) = term(slot);
        q = add(&q, &mul(&primaries[axis], &compose(&q, stack)));
    }
    q
}

impl<'a> BinomialLocationScaleWiggleRowProgram<'a> {
    fn new(
        family: &'a BinomialLocationScaleWiggleFamily,
        block_states: &'a [ParameterBlockState],
        derivative_order: usize,
    ) -> Result<Self, String> {
        assert!(derivative_order <= 4);
        let (_, eta_t, eta_ls, etaw) = family.validated_block_etas(block_states)?;
        let beta_w = &block_states[BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE].beta;
        let core = binomial_location_scale_core(
            &family.y,
            &family.weights,
            eta_t,
            eta_ls,
            Some(etaw),
            &family.link_kind,
        )?;
        let mut basis_derivatives = Vec::with_capacity(derivative_order + 1);
        for order in 0..=derivative_order {
            let basis = monotone_wiggle_basis_with_derivative_order(
                core.q0.view(),
                &family.wiggle_knots,
                family.wiggle_degree,
                order,
            )?;
            if basis.ncols() != beta_w.len() {
                return Err(GamlssError::DimensionMismatch {
                    reason: format!(
                        "binomial wiggle row program derivative-{order} basis width {} != beta width {}",
                        basis.ncols(),
                        beta_w.len()
                    ),
                }
                .into());
            }
            basis_derivatives.push(basis);
        }
        Ok(Self {
            family,
            eta_t,
            eta_ls,
            etaw,
            beta_w,
            core,
            basis_derivatives,
        })
    }

    #[inline]
    fn primary_dimension(&self) -> usize {
        2 + self.beta_w.len()
    }

    #[inline]
    fn basis_stack(&self, row: usize, column: usize) -> [f64; 5] {
        let mut stack = [0.0; 5];
        for (order, basis) in self.basis_derivatives.iter().enumerate() {
            stack[order] = basis[[row, column]];
        }
        stack
    }

    #[inline]
    fn linear_basis_stack(
        &self,
        row: usize,
        coefficients: ArrayView1<'_, f64>,
        authoritative_value: Option<f64>,
    ) -> [f64; 5] {
        assert_eq!(coefficients.len(), self.beta_w.len());
        let mut stack = [0.0; 5];
        for (order, basis) in self.basis_derivatives.iter().enumerate() {
            stack[order] = basis.row(row).dot(&coefficients);
        }
        if let Some(value) = authoritative_value {
            stack[0] = value;
        }
        stack
    }

    #[inline]
    fn objective_stack(&self, row: usize, derivative_order: usize) -> Result<[f64; 5], String> {
        let q = self.core.q0[row] + self.etaw[row];
        let (m1, m2, m3) = binomial_neglog_q_derivatives_dispatch(
            self.family.y[row],
            self.family.weights[row],
            q,
            self.core.mu[row],
            self.core.dmu_dq[row],
            self.core.d2mu_dq2[row],
            self.core.d3mu_dq3[row],
            &self.family.link_kind,
        );
        let m4 = if derivative_order >= 4 {
            binomial_neglog_q_fourth_derivative_dispatch(
                self.family.y[row],
                self.family.weights[row],
                q,
                self.core.mu[row],
                self.core.dmu_dq[row],
                self.core.d2mu_dq2[row],
                self.core.d3mu_dq3[row],
                &self.family.link_kind,
            )?
        } else {
            0.0
        };
        Ok([0.0, m1, m2, m3, m4])
    }

    /// Runtime-width instantiation of the canonical row program.  This is the
    /// independent full-primary path used by generic row-program verification;
    /// production structured lowerings below call the same predictor body with
    /// fixed probe axes so they never allocate a `p_w^4` tensor.
    pub(crate) fn eval_runtime<'arena, S: gam_math::jet_scalar::RuntimeJetScalar<'arena>>(
        &self,
        row: usize,
        primaries: &[S],
    ) -> Result<S, String> {
        if primaries.len() != self.primary_dimension() {
            return Err(format!(
                "binomial wiggle row program primary width {} != {}",
                primaries.len(),
                self.primary_dimension()
            ));
        }
        let q = binomial_location_scale_wiggle_predictor_expression(
            primaries,
            None,
            self.beta_w.len(),
            |column| (2 + column, self.basis_stack(row, column)),
            |x| gam_math::jet_scalar::RuntimeJetScalar::value(x),
            |a, b| gam_math::jet_scalar::RuntimeJetScalar::add(a, b),
            |a, b| gam_math::jet_scalar::RuntimeJetScalar::mul(a, b),
            |x| gam_math::jet_scalar::RuntimeJetScalar::neg(x),
            |x, stack| gam_math::jet_scalar::RuntimeJetScalar::compose_unary(x, stack),
        );
        let q_value = gam_math::jet_scalar::RuntimeJetScalar::value(&q);
        let inverse = inverse_link_jet_for_inverse_link(&self.family.link_kind, q_value)
            .map_err(|e| format!("binomial wiggle row program inverse-link failed: {e}"))?;
        let (m1, m2, m3) = binomial_neglog_q_derivatives_dispatch(
            self.family.y[row],
            self.family.weights[row],
            q_value,
            inverse.mu,
            inverse.d1,
            inverse.d2,
            inverse.d3,
            &self.family.link_kind,
        );
        let m4 = binomial_neglog_q_fourth_derivative_dispatch(
            self.family.y[row],
            self.family.weights[row],
            q_value,
            inverse.mu,
            inverse.d1,
            inverse.d2,
            inverse.d3,
            &self.family.link_kind,
        )?;
        let neg_ll = -binomial_location_scale_log_likelihood(
            self.family.y[row],
            self.family.weights[row],
            q_value,
            &self.family.link_kind,
            inverse.mu,
        )?;
        Ok(gam_math::jet_scalar::RuntimeJetScalar::compose_unary(
            &q,
            [neg_ll, m1, m2, m3, m4],
        ))
    }

    #[inline]
    fn eval_fixed<const K: usize, S: gam_math::jet_scalar::JetScalar<K>>(
        &self,
        row: usize,
        primaries: &[S; K],
        warp_stack: [f64; 5],
        terms: &[(usize, [f64; 5])],
        derivative_order: usize,
    ) -> Result<S, String> {
        use gam_math::nested_dual::JetField;
        let q = binomial_location_scale_wiggle_predictor_expression(
            primaries,
            Some(warp_stack),
            terms.len(),
            |slot| terms[slot],
            JetField::value,
            JetField::add,
            JetField::mul,
            JetField::neg,
            JetField::compose_unary,
        );
        Ok(q.compose_unary(self.objective_stack(row, derivative_order)?))
    }

    fn order2_rows(&self) -> Result<BinomialWiggleOrder2Rows, String> {
        use gam_math::jet_scalar::{JetScalar, Order2};

        let n = self.family.y.len();
        let mut rows = BinomialWiggleOrder2Rows::zeros(
            n,
            self.basis_derivatives[0].clone(),
            self.basis_derivatives[1].clone(),
        );
        let probe_terms = [
            (2, [1.0, 0.0, 0.0, 0.0, 0.0]),
            (3, [0.0, 1.0, 0.0, 0.0, 0.0]),
        ];
        for row in 0..n {
            let values = [self.eta_t[row], self.eta_ls[row], 0.0, 0.0];
            let primaries: [Order2<4>; 4] =
                std::array::from_fn(|axis| Order2::variable(values[axis], axis));
            let warp = self.linear_basis_stack(row, self.beta_w.view(), Some(self.etaw[row]));
            let h = self
                .eval_fixed(row, &primaries, warp, &probe_terms, 2)?
                .into_channels()
                .2;
            rows.coeff_tt[row] = h[0][0];
            rows.coeff_tl[row] = h[0][1];
            rows.coeff_ll[row] = h[1][1];
            rows.coeff_tw_b[row] = h[0][2];
            rows.coeff_tw_d[row] = h[0][3];
            rows.coeff_lw_b[row] = h[1][2];
            rows.coeff_lw_d[row] = h[1][3];
            rows.coeff_ww[row] = h[2][2];
        }
        Ok(rows)
    }
}

#[derive(Clone)]
pub struct BinomialLocationScaleWiggleFamily {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub link_kind: InverseLink,
    pub threshold_design: Option<DesignMatrix>,
    pub log_sigma_design: Option<DesignMatrix>,
    pub wiggle_knots: Array1<f64>,
    pub wiggle_degree: usize,
    /// Resource policy threaded into PsiDesignMap construction (and any other
    /// per-call materialization decision) made during exact-Newton joint psi
    /// derivative evaluation. Defaults to `ResourcePolicy::default_library()`
    /// when the family is built without an explicit policy.
    pub policy: gam_runtime::resource::ResourcePolicy,
}

impl BinomialLocationScaleWiggleFamily {
    pub const BLOCK_T: usize = 0;
    pub const BLOCK_LOG_SIGMA: usize = 1;
    pub const BLOCK_WIGGLE: usize = 2;

    pub fn parameternames() -> &'static [&'static str] {
        &["threshold", "log_sigma", "wiggle"]
    }

    pub fn parameter_links() -> &'static [ParameterLink] {
        &[
            ParameterLink::InverseLink,
            ParameterLink::Log,
            ParameterLink::Wiggle,
        ]
    }

    pub fn metadata() -> FamilyMetadata {
        FamilyMetadata {
            name: "binomial_location_scalewiggle",
            parameternames: Self::parameternames(),
            parameter_links: Self::parameter_links(),
        }
    }

    pub(crate) fn exact_joint_supported(&self) -> bool {
        self.threshold_design.is_some() && self.log_sigma_design.is_some()
    }

    /// Validate that `block_states` carries exactly the three BLS-wiggle blocks
    /// (threshold, log-sigma, wiggle) with `η` slices all sized to `n = y.len()`
    /// and matching `weights`, then return `(n, η_t, η_ls, η_w)`.
    ///
    /// This is the shared entry guard for every per-row BLS-wiggle evaluator
    /// (log-likelihood, core, joint psi, directional operators). Hoisting it
    /// keeps the block-count contract and the dimension-mismatch error message
    /// identical across all call sites.
    pub(crate) fn validated_block_etas<'a>(
        &self,
        block_states: &'a [ParameterBlockState],
    ) -> Result<(usize, &'a Array1<f64>, &'a Array1<f64>, &'a Array1<f64>), String> {
        validate_block_count::<GamlssError>(
            "BinomialLocationScaleWiggleFamily",
            3,
            block_states.len(),
        )?;
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        if eta_t.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialLocationScaleWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        Ok((n, eta_t, eta_ls, etaw))
    }

    pub fn initializewiggle_knots_from_q(
        q_seed: ArrayView1<'_, f64>,
        degree: usize,
        num_internal_knots: usize,
    ) -> Result<Array1<f64>, String> {
        initializewiggle_knots_from_seed(q_seed, degree, num_internal_knots)
    }

    pub(crate) fn wiggle_basiswith_options(
        &self,
        q0: ArrayView1<'_, f64>,
        basis_options: BasisOptions,
    ) -> Result<Array2<f64>, String> {
        monotone_wiggle_basis_with_derivative_order(
            q0,
            &self.wiggle_knots,
            self.wiggle_degree,
            basis_options.derivative_order,
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
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "wiggle derivative col mismatch: got {}, expected {}",
                    d_constrained.ncols(),
                    beta_link_wiggle.len()
                ),
            }
            .into());
        }
        Ok(d_constrained.dot(&beta_link_wiggle) + 1.0)
    }

    pub(crate) fn wiggle_d2q_dq02(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d2_constrained =
            self.wiggle_basiswith_options(q0, BasisOptions::second_derivative())?;
        if d2_constrained.ncols() != beta_link_wiggle.len() {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "wiggle second-derivative col mismatch: got {}, expected {}",
                    d2_constrained.ncols(),
                    beta_link_wiggle.len()
                ),
            }
            .into());
        }
        Ok(d2_constrained.dot(&beta_link_wiggle))
    }

    pub(crate) fn wiggle_d3q_dq03(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d3_constrained = self.wiggle_d3basis_constrained(q0)?;
        if d3_constrained.ncols() != beta_link_wiggle.len() {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "wiggle third-derivative col mismatch: got {}, expected {}",
                    d3_constrained.ncols(),
                    beta_link_wiggle.len()
                ),
            }
            .into());
        }
        Ok(d3_constrained.dot(&beta_link_wiggle))
    }

    pub(crate) fn wiggle_d3basis_constrained(
        &self,
        q0: ArrayView1<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        monotone_wiggle_basis_with_derivative_order(q0, &self.wiggle_knots, self.wiggle_degree, 3)
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
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "wiggle fourth-derivative col mismatch: got {}, expected {}",
                    d4.ncols(),
                    beta_link_wiggle.len()
                ),
            }
            .into());
        }
        Ok(d4.dot(&beta_link_wiggle))
    }

    pub(crate) fn dense_block_designs(
        &self,
    ) -> Result<(Cow<'_, Array2<f64>>, Cow<'_, Array2<f64>>), String> {
        dense_locscale_block_designs_cached(
            self.threshold_design.as_ref(),
            self.log_sigma_design.as_ref(),
            "BinomialLocationScaleWiggleFamily",
            "BinomialLocationScaleWiggle",
            "threshold",
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
            "BinomialLocationScaleWiggleFamily",
            "BinomialLocationScaleWiggle",
            Self::BLOCK_T,
            Self::BLOCK_LOG_SIGMA,
            "threshold",
            &self.policy.material_policy(),
        )
    }

    pub(crate) fn exact_joint_dense_block_designs<'a>(
        &'a self,
        specs: Option<&'a [ParameterBlockSpec]>,
    ) -> Result<Option<(Cow<'a, Array2<f64>>, Cow<'a, Array2<f64>>)>, String> {
        if self.threshold_design.is_some() && self.log_sigma_design.is_some() {
            return self.dense_block_designs().map(Some);
        }
        if let Some(specs) = specs {
            return self.dense_block_designs_fromspecs(specs).map(Some);
        }
        Ok(None)
    }

    pub(crate) fn shadow_with_exact_joint_designs(
        &self,
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Self>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        Ok(Some(Self {
            y: self.y.clone(),
            weights: self.weights.clone(),
            link_kind: self.link_kind.clone(),
            threshold_design: Some(DesignMatrix::Dense(
                gam_linalg::matrix::DenseDesignMatrix::from(x_t.into_owned()),
            )),
            log_sigma_design: Some(DesignMatrix::Dense(
                gam_linalg::matrix::DenseDesignMatrix::from(x_ls.into_owned()),
            )),
            wiggle_knots: self.wiggle_knots.clone(),
            wiggle_degree: self.wiggle_degree,
            policy: self.policy.clone(),
        }))
    }

    pub(crate) fn exact_newton_joint_psi_terms_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<gam_problem::ExactNewtonJointPsiTerms>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_psi_terms_from_designs(
            block_states,
            derivative_blocks,
            psi_index,
            &x_t,
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
    ) -> Result<Option<gam_problem::ExactNewtonJointPsiSecondOrderTerms>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_psisecond_order_terms_from_designs(
            block_states,
            derivative_blocks,
            psi_i,
            psi_j,
            &x_t,
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
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_psihessian_directional_derivative_from_designs(
            block_states,
            derivative_blocks,
            psi_index,
            d_beta_flat,
            &x_t,
            &x_ls,
        )
    }

    pub(crate) fn exact_newton_joint_psi_direction(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
        policy: &gam_runtime::resource::ResourcePolicy,
    ) -> Result<Option<LocationScaleJointPsiDirection>, String> {
        let Some(parts) = locscale_joint_psi_direction_parts(
            block_states,
            derivative_blocks,
            psi_index,
            self.y.len(),
            x_t.ncols(),
            x_ls.ncols(),
            Self::BLOCK_T,
            Self::BLOCK_LOG_SIGMA,
            3,
            "BinomialLocationScaleWiggleFamily",
            "threshold",
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
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<LocationScaleJointPsiSecondDrifts, String> {
        locscale_joint_psisecond_design_drifts(
            block_states,
            derivative_blocks,
            psi_a,
            psi_b,
            LocScalePsiDriftConfig {
                n: self.y.len(),
                p_primary: x_t.ncols(),
                p_log_sigma: x_ls.ncols(),
                primary_block_idx: Self::BLOCK_T,
                log_sigma_block_idx: Self::BLOCK_LOG_SIGMA,
                family_name: "BinomialLocationScaleWiggleFamily",
                primary_label: "threshold",
                policy: &self.policy,
            },
        )
    }

    pub(crate) fn exact_newton_joint_psi_terms_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<gam_problem::ExactNewtonJointPsiTerms>, String> {
        if self
            .exact_newton_joint_psi_direction(
                block_states,
                derivative_blocks,
                psi_index,
                x_t,
                x_ls,
                &self.policy,
            )?
            .is_none()
        {
            return Ok(None);
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(etaw),
            &self.link_kind,
        )?;
        let base_core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let b0 = self.wiggle_design(base_core.q0.view())?;
        let d0 =
            self.wiggle_basiswith_options(base_core.q0.view(), BasisOptions::first_derivative())?;
        let dd0 =
            self.wiggle_basiswith_options(base_core.q0.view(), BasisOptions::second_derivative())?;
        let d3q = self.wiggle_d3q_dq03(base_core.q0.view(), betaw.view())?;
        let m = d0.dot(betaw) + 1.0;
        let g2 = self.wiggle_d2q_dq02(base_core.q0.view(), betaw.view())?;
        let g3 = d3q;
        let (sigma, ..) = exp_sigma_derivs_up_to_third(eta_ls.view());

        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let pw = b0.ncols();
        let total = pt + pls + pw;
        let Some(dir_a) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_index,
            x_t,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        let (z_t_psi, z_ls_psi) = (&dir_a.z_primary_psi, &dir_a.z_ls_psi);
        let mut objective_psi = 0.0;

        let mut score_t_xa = Array1::<f64>::zeros(n);
        let mut score_t_x = Array1::<f64>::zeros(n);
        let mut score_ls_xa = Array1::<f64>::zeros(n);
        let mut score_ls_x = Array1::<f64>::zeros(n);
        let mut score_w_b = Array1::<f64>::zeros(n);
        let mut score_w_d1 = Array1::<f64>::zeros(n);

        let mut coeff_tt_w = Array1::<f64>::zeros(n);
        let mut coeff_tt_d = Array1::<f64>::zeros(n);
        let mut coeff_tl_w = Array1::<f64>::zeros(n);
        let mut coeff_tl_d = Array1::<f64>::zeros(n);
        let mut coeff_ll_w = Array1::<f64>::zeros(n);
        let mut coeff_ll_d = Array1::<f64>::zeros(n);
        let mut coeff_tw_b_w = Array1::<f64>::zeros(n);
        let mut coeff_tw_b_d = Array1::<f64>::zeros(n);
        let mut coeff_tw_d1_w = Array1::<f64>::zeros(n);
        let mut coeff_tw_d1_d = Array1::<f64>::zeros(n);
        let mut coeff_tw_d2_d = Array1::<f64>::zeros(n);
        let mut coeff_lw_b_w = Array1::<f64>::zeros(n);
        let mut coeff_lw_b_d = Array1::<f64>::zeros(n);
        let mut coeff_lw_d1_w = Array1::<f64>::zeros(n);
        let mut coeff_lw_d1_d = Array1::<f64>::zeros(n);
        let mut coeff_lw_d2_d = Array1::<f64>::zeros(n);
        let mut coeff_ww_bb = Array1::<f64>::zeros(n);
        let mut coeff_ww_db = Array1::<f64>::zeros(n);

        // Exact likelihood-only joint psi terms for the probit wiggle family.
        //
        // This helper is intentionally the same generic rowwise kernel as the
        // non-wiggle family. The only difference is the location-side row:
        //
        //   gamma = [beta_t; betaw],
        //   delta = beta_ls,
        //   z_r   = [x_{t,r}; B_r(q0)],
        //   x_r   = x_{ls,r},
        //   a_r   = z_r^T gamma,
        //   ell_r = x_r^T delta,
        //   q_r   = -a_r * exp(-ell_r).
        //
        // In this wiggle family we realize the same kernel through the chain
        //
        //   q = q0 + betaw^T B(q0),
        //   q0 = -eta_t * exp(-eta_ls),
        //   m  = dq/dq0   = 1 + betaw^T B'(q0),
        //   g2 = d²q/dq0² = betaw^T B''(q0),
        //   g3 = d³q/dq0³ = betaw^T B'''(q0).
        //
        // For a realized hyperdirection psi_a:
        //
        //   h_a     = q_{psi_a},
        //   c_a     = q_{beta psi_a},
        //   R_a     = q_{beta beta psi_a},
        //
        // and the generic scalar-loss identities are
        //
        //   D_a            = sum_r r_r h_{r,a},
        //   D_{beta a}     = sum_r [ w_r h_{r,a} b_r + r_r c_{r,a} ],
        //   D_{beta beta a}
        //                  = sum_r [ nu_r h_{r,a} b_r b_r^T
        //                              + w_r(c_{r,a} b_r^T + b_r c_{r,a}^T + h_{r,a} Q_r)
        //                              + r_r R_{r,a} ].
        //
        // Generic exact-joint code adds all realized penalty motion S_a after
        // the fact, so this family hook must stay likelihood-only.
        //
        // The rowwise objects below are the wiggle specialization of the same
        // q_r = -a_r exp(-ell_r) kernel. All wiggle-specific complexity is
        // localized to the realized row B_r(q0) and its q0-derivatives.
        for row in 0..n {
            let q0 = base_core.q0[row];
            let q = q0 + etaw[row];
            let q0_geom = nonwiggle_q_derivs(eta_t[row], sigma[row]);
            let r_sigma = 1.0 / sigma[row];
            let q0_a = -r_sigma * z_t_psi[row] - q0 * z_ls_psi[row];
            let q0_t_a = q0_geom.q_tl * z_ls_psi[row];
            let q0_ls_a = q0_geom.q_tl * z_t_psi[row] + q0_geom.q_ll * z_ls_psi[row];
            let q0_tl_a = q0_geom.q_tl_ls * z_ls_psi[row];
            let q0_ll_a = q0_geom.q_tl_ls * z_t_psi[row] + q0_geom.q_ll_ls * z_ls_psi[row];

            let q_t = m[row] * q0_geom.q_t;
            let q_ls = m[row] * q0_geom.q_ls;
            let q_tt = g2[row] * q0_geom.q_t * q0_geom.q_t;
            let q_tl = g2[row] * q0_geom.q_t * q0_geom.q_ls + m[row] * q0_geom.q_tl;
            let q_ll = g2[row] * q0_geom.q_ls * q0_geom.q_ls + m[row] * q0_geom.q_ll;
            let q_t_a = g2[row] * q0_a * q0_geom.q_t + m[row] * q0_t_a;
            let q_ls_a = g2[row] * q0_a * q0_geom.q_ls + m[row] * q0_ls_a;
            let q_tt_a =
                g3[row] * q0_a * q0_geom.q_t * q0_geom.q_t + g2[row] * (2.0 * q0_geom.q_t * q0_t_a);
            let q_tl_a = g3[row] * q0_a * q0_geom.q_t * q0_geom.q_ls
                + g2[row] * (q0_t_a * q0_geom.q_ls + q0_geom.q_t * q0_ls_a + q0_a * q0_geom.q_tl)
                + m[row] * q0_tl_a;
            let q_ll_a = g3[row] * q0_a * q0_geom.q_ls * q0_geom.q_ls
                + g2[row] * (2.0 * q0_geom.q_ls * q0_ls_a + q0_a * q0_geom.q_ll)
                + m[row] * q0_ll_a;

            let (loss_1, loss_2, loss_3) = binomial_neglog_q_derivatives_dispatch(
                self.y[row],
                self.weights[row],
                q,
                core.mu[row],
                core.dmu_dq[row],
                core.d2mu_dq2[row],
                core.d3mu_dq3[row],
                &self.link_kind,
            );
            let alpha = m[row] * q0_a;
            objective_psi += loss_1 * alpha;

            score_t_xa[row] = loss_1 * q_t;
            score_t_x[row] = loss_2 * alpha * q_t + loss_1 * q_t_a;
            score_ls_xa[row] = loss_1 * q_ls;
            score_ls_x[row] = loss_2 * alpha * q_ls + loss_1 * q_ls_a;
            score_w_b[row] = loss_2 * alpha;
            score_w_d1[row] = loss_1 * q0_a;

            coeff_tt_w[row] = loss_2 * q_t * q_t + loss_1 * q_tt;
            coeff_tt_d[row] = loss_3 * alpha * q_t * q_t
                + 2.0 * loss_2 * q_t * q_t_a
                + loss_2 * alpha * q_tt
                + loss_1 * q_tt_a;
            coeff_tl_w[row] = loss_2 * q_t * q_ls + loss_1 * q_tl;
            coeff_tl_d[row] = loss_3 * alpha * q_t * q_ls
                + loss_2 * (q_t_a * q_ls + q_t * q_ls_a)
                + loss_2 * alpha * q_tl
                + loss_1 * q_tl_a;
            coeff_ll_w[row] = loss_2 * q_ls * q_ls + loss_1 * q_ll;
            coeff_ll_d[row] = loss_3 * alpha * q_ls * q_ls
                + 2.0 * loss_2 * q_ls * q_ls_a
                + loss_2 * alpha * q_ll
                + loss_1 * q_ll_a;

            coeff_tw_b_w[row] = loss_2 * q_t;
            coeff_tw_b_d[row] = loss_3 * alpha * q_t + loss_2 * q_t_a;
            coeff_tw_d1_w[row] = loss_1 * q0_geom.q_t;
            coeff_tw_d1_d[row] = loss_2 * (q_t * q0_a + alpha * q0_geom.q_t) + loss_1 * q0_t_a;
            coeff_tw_d2_d[row] = loss_1 * q0_a * q0_geom.q_t;

            coeff_lw_b_w[row] = loss_2 * q_ls;
            coeff_lw_b_d[row] = loss_3 * alpha * q_ls + loss_2 * q_ls_a;
            coeff_lw_d1_w[row] = loss_1 * q0_geom.q_ls;
            coeff_lw_d1_d[row] = loss_2 * (q_ls * q0_a + alpha * q0_geom.q_ls) + loss_1 * q0_ls_a;
            coeff_lw_d2_d[row] = loss_1 * q0_a * q0_geom.q_ls;

            coeff_ww_bb[row] = loss_3 * alpha;
            coeff_ww_db[row] = loss_2 * q0_a;
        }
        let x_t_map = dir_a.x_primary_psi.as_linear_map_ref();
        let x_ls_map = dir_a.x_ls_psi.as_linear_map_ref();
        let score_t = x_t_map.transpose_mul(score_t_xa.view()) + fast_atv(x_t, &score_t_x);
        let score_ls = x_ls_map.transpose_mul(score_ls_xa.view()) + fast_atv(x_ls, &score_ls_x);
        let score_w = fast_atv(&b0, &score_w_b) + fast_atv(&d0, &score_w_d1);
        let mut score_psi = Array1::<f64>::zeros(total);
        score_psi.slice_mut(s![0..pt]).assign(&score_t);
        score_psi.slice_mut(s![pt..pt + pls]).assign(&score_ls);
        score_psi.slice_mut(s![pt + pls..total]).assign(&score_w);

        let x_t_action_opt = dir_a.x_primary_psi.cloned_first_action();
        let x_ls_action_opt = dir_a.x_ls_psi.cloned_first_action();
        if x_t_action_opt.is_some() || x_ls_action_opt.is_some() {
            let basis_arc = Arc::new(b0.clone());
            let basis_d1_arc = Arc::new(d0.clone());
            let basis_d2_arc = Arc::new(dd0.clone());
            let zeros = Array1::<f64>::zeros(n);
            let operator = CustomFamilyJointPsiOperator::new(
                total,
                vec![
                    CustomFamilyJointDesignChannel::new(
                        0..pt,
                        shared_dense_arc(x_t),
                        x_t_action_opt,
                    ),
                    CustomFamilyJointDesignChannel::new(
                        pt..pt + pls,
                        shared_dense_arc(x_ls),
                        x_ls_action_opt,
                    ),
                    CustomFamilyJointDesignChannel::new(
                        pt + pls..total,
                        Arc::clone(&basis_arc),
                        None,
                    ),
                    CustomFamilyJointDesignChannel::new(
                        pt + pls..total,
                        Arc::clone(&basis_d1_arc),
                        None,
                    ),
                    CustomFamilyJointDesignChannel::new(
                        pt + pls..total,
                        Arc::clone(&basis_d2_arc),
                        None,
                    ),
                ],
                vec![
                    CustomFamilyJointDesignPairContribution::new(
                        0,
                        0,
                        coeff_tt_w.clone(),
                        coeff_tt_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        0,
                        1,
                        coeff_tl_w.clone(),
                        coeff_tl_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        1,
                        0,
                        coeff_tl_w.clone(),
                        coeff_tl_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        1,
                        1,
                        coeff_ll_w.clone(),
                        coeff_ll_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        0,
                        2,
                        coeff_tw_b_w.clone(),
                        coeff_tw_b_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        2,
                        0,
                        coeff_tw_b_w.clone(),
                        coeff_tw_b_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        0,
                        3,
                        coeff_tw_d1_w.clone(),
                        coeff_tw_d1_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        3,
                        0,
                        coeff_tw_d1_w.clone(),
                        coeff_tw_d1_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        0,
                        4,
                        zeros.clone(),
                        coeff_tw_d2_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        4,
                        0,
                        zeros.clone(),
                        coeff_tw_d2_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        1,
                        2,
                        coeff_lw_b_w.clone(),
                        coeff_lw_b_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        2,
                        1,
                        coeff_lw_b_w.clone(),
                        coeff_lw_b_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        1,
                        3,
                        coeff_lw_d1_w.clone(),
                        coeff_lw_d1_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        3,
                        1,
                        coeff_lw_d1_w.clone(),
                        coeff_lw_d1_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        1,
                        4,
                        zeros.clone(),
                        coeff_lw_d2_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        4,
                        1,
                        zeros.clone(),
                        coeff_lw_d2_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        2,
                        2,
                        zeros.clone(),
                        coeff_ww_bb.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        3,
                        2,
                        zeros.clone(),
                        coeff_ww_db.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(2, 3, zeros, coeff_ww_db.clone()),
                ],
            );
            return Ok(Some(gam_problem::ExactNewtonJointPsiTerms {
                objective_psi,
                score_psi,
                hessian_psi: Array2::zeros((0, 0)),
                hessian_psi_operator: Some(std::sync::Arc::new(operator)),
            }));
        }
        let h_tt_block = weighted_crossprod_psi_maps(
            x_t_map,
            coeff_tt_w.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_t),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_t),
            coeff_tt_w.view(),
            x_t_map,
        )? + &xt_diag_x_dense(x_t, &coeff_tt_d)?;
        let h_tl_block = weighted_crossprod_psi_maps(
            x_t_map,
            coeff_tl_w.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_t),
            coeff_tl_w.view(),
            x_ls_map,
        )? + &xt_diag_y_dense(x_t, &coeff_tl_d, x_ls)?;
        let h_ll_block = weighted_crossprod_psi_maps(
            x_ls_map,
            coeff_ll_w.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
            coeff_ll_w.view(),
            x_ls_map,
        )? + &xt_diag_x_dense(x_ls, &coeff_ll_d)?;
        let h_tw = weighted_crossprod_psi_maps(
            x_t_map,
            coeff_tw_b_w.view(),
            CustomFamilyPsiLinearMapRef::Dense(&b0),
        )? + &xt_diag_y_dense(x_t, &coeff_tw_b_d, &b0)?
            + &weighted_crossprod_psi_maps(
                x_t_map,
                coeff_tw_d1_w.view(),
                CustomFamilyPsiLinearMapRef::Dense(&d0),
            )?
            + &xt_diag_y_dense(x_t, &coeff_tw_d1_d, &d0)?
            + &xt_diag_y_dense(x_t, &coeff_tw_d2_d, &dd0)?;
        let h_lw = weighted_crossprod_psi_maps(
            x_ls_map,
            coeff_lw_b_w.view(),
            CustomFamilyPsiLinearMapRef::Dense(&b0),
        )? + &xt_diag_y_dense(x_ls, &coeff_lw_b_d, &b0)?
            + &weighted_crossprod_psi_maps(
                x_ls_map,
                coeff_lw_d1_w.view(),
                CustomFamilyPsiLinearMapRef::Dense(&d0),
            )?
            + &xt_diag_y_dense(x_ls, &coeff_lw_d1_d, &d0)?
            + &xt_diag_y_dense(x_ls, &coeff_lw_d2_d, &dd0)?;
        let a_ww = xt_diag_y_dense(&d0, &coeff_ww_db, &b0)?;
        let h_ww = xt_diag_x_dense(&b0, &coeff_ww_bb)? + &a_ww + a_ww.t();

        let mut hessian_psi = Array2::<f64>::zeros((total, total));
        hessian_psi.slice_mut(s![0..pt, 0..pt]).assign(&h_tt_block);
        hessian_psi
            .slice_mut(s![0..pt, pt..pt + pls])
            .assign(&h_tl_block);
        hessian_psi
            .slice_mut(s![pt..pt + pls, pt..pt + pls])
            .assign(&h_ll_block);
        hessian_psi
            .slice_mut(s![0..pt, pt + pls..total])
            .assign(&h_tw);
        hessian_psi
            .slice_mut(s![pt..pt + pls, pt + pls..total])
            .assign(&h_lw);
        hessian_psi
            .slice_mut(s![pt + pls..total, pt + pls..total])
            .assign(&h_ww);
        mirror_upper_to_lower(&mut hessian_psi);

        Ok(Some(gam_problem::ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi,
            hessian_psi_operator: None,
        }))
    }

    pub(crate) fn exact_newton_joint_psisecond_order_terms_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<gam_problem::ExactNewtonJointPsiSecondOrderTerms>, String> {
        validate_block_count::<GamlssError>(
            "BinomialLocationScaleWiggleFamily",
            3,
            block_states.len(),
        )?;
        if derivative_blocks.len() != 3 {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "BinomialLocationScaleWiggleFamily joint psi second-order terms expect 3 derivative block lists, got {}",
                derivative_blocks.len()
            ) }.into());
        }
        let Some(dir_a) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_i,
            x_t,
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
            x_t,
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
                x_t,
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
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<gam_problem::ExactNewtonJointPsiSecondOrderTerms, String> {
        let second_drifts = self.exact_newton_joint_psisecond_design_drifts(
            block_states,
            derivative_blocks,
            dir_a,
            dir_b,
            x_t,
            x_ls,
        )?;
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(etaw),
            &self.link_kind,
        )?;
        let base_core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let b0 = self.wiggle_design(base_core.q0.view())?;
        let d0 =
            self.wiggle_basiswith_options(base_core.q0.view(), BasisOptions::first_derivative())?;
        let dd0 =
            self.wiggle_basiswith_options(base_core.q0.view(), BasisOptions::second_derivative())?;
        let d3_basis = self.wiggle_d3basis_constrained(base_core.q0.view())?;
        let d3q = self.wiggle_d3q_dq03(base_core.q0.view(), betaw.view())?;
        let d4q = self.wiggle_d4q_dq04(base_core.q0.view(), betaw.view())?;
        if b0.ncols() != betaw.len()
            || d0.ncols() != betaw.len()
            || dd0.ncols() != betaw.len()
            || d3_basis.ncols() != betaw.len()
        {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "wiggle derivative/beta mismatch in joint psi psi terms: B={} B'={} B''={} B'''={} betaw={}",
                b0.ncols(),
                d0.ncols(),
                dd0.ncols(),
                d3_basis.ncols(),
                betaw.len()
            ) }.into());
        }
        let m = d0.dot(betaw) + 1.0;
        let g2 = dd0.dot(betaw);
        let g3 = d3q;
        let g4 = d4q;
        let (sigma, ds, d2s, d3s) = exp_sigma_derivs_up_to_third(eta_ls.view());

        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let pw = b0.ncols();
        let total = pt + pls + pw;
        let x_t_a_map = dir_a.x_primary_psi.as_linear_map_ref();
        let x_t_b_map = dir_b.x_primary_psi.as_linear_map_ref();
        let x_ls_a_map = dir_a.x_ls_psi.as_linear_map_ref();
        let x_ls_b_map = dir_b.x_ls_psi.as_linear_map_ref();
        let x_t_ab_map = second_psi_linear_map(
            second_drifts.x_primary_ab_action.as_ref(),
            second_drifts.x_primary_ab.as_ref(),
            n,
            pt,
        );
        let x_ls_ab_map = second_psi_linear_map(
            second_drifts.x_ls_ab_action.as_ref(),
            second_drifts.x_ls_ab.as_ref(),
            n,
            pls,
        );
        let mut objective_psi_psi = 0.0;
        let mut score_psi_psi = Array1::<f64>::zeros(total);
        let mut hessian_psi_psi = Array2::<f64>::zeros((total, total));

        // Likelihood-only exact psi/psi terms for the wiggle family.
        //
        // This is the same generic second-order kernel as the non-wiggle path,
        // still over the flattened coefficients beta = [beta_t; beta_ls; betaw].
        // The family provides only the likelihood-side fixed-beta objects
        //
        //   D_ab, D_{beta ab}, D_{beta beta ab},
        //
        // while generic exact-joint code in custom_family.rs adds all realized
        // penalty motion S_ab.
        //
        // Using the generic rowwise notation
        //
        //   h_a   = q_{psi_a},      h_b   = q_{psi_b},
        //   h_ab  = q_{psi_a psi_b},
        //   c_a   = q_{beta psi_a}, c_b   = q_{beta psi_b},
        //   c_ab  = q_{beta psi_a psi_b},
        //   R_a   = q_{beta beta psi_a},
        //   R_b   = q_{beta beta psi_b},
        //   R_ab  = q_{beta beta psi_a psi_b},
        //
        // the exact scalar-loss kernel is
        //
        //   D_ab
        //   = sum_r [ w_r h_{r,a} h_{r,b} + r_r h_{r,ab} ],
        //
        //   D_{beta ab}
        //   = sum_r [
        //       r_r c_{r,ab}
        //       + w_r h_{r,b} c_{r,a}
        //       + w_r h_{r,a} c_{r,b}
        //       + (w_r h_{r,ab} + nu_r h_{r,a} h_{r,b}) b_r
        //     ],
        //
        //   D_{beta beta ab}
        //   = sum_r [
        //       r_r R_{r,ab}
        //       + w_r h_{r,b} R_{r,a}
        //       + w_r h_{r,a} R_{r,b}
        //       + w_r(c_{r,ab} b_r^T + b_r c_{r,ab}^T
        //             + c_{r,a} c_{r,b}^T + c_{r,b} c_{r,a}^T
        //             + h_{r,ab} Q_r)
        //       + nu_r h_{r,b}(c_{r,a} b_r^T + b_r c_{r,a}^T)
        //       + nu_r h_{r,a}(c_{r,b} b_r^T + b_r c_{r,b}^T)
        //       + nu_r h_{r,a} h_{r,b} Q_r
        //       + (tau_r h_{r,a} h_{r,b} + nu_r h_{r,ab}) b_r b_r^T
        //     ].
        //
        // The wiggle specialization enters only through the rowwise q-objects
        // built below from the combined location-side row z_r = [x_{t,r}; B_r(q0)].
        let mut b = Array1::<f64>::zeros(total);
        let mut c_a = Array1::<f64>::zeros(total);
        let mut c_b = Array1::<f64>::zeros(total);
        let mut c_ab = Array1::<f64>::zeros(total);
        let mut q_mat = Array2::<f64>::zeros((total, total));
        let mut r_a = Array2::<f64>::zeros((total, total));
        let mut r_b = Array2::<f64>::zeros((total, total));
        let mut r_ab = Array2::<f64>::zeros((total, total));
        let mut qw_a = Array1::<f64>::zeros(pw);
        let mut qw_b = Array1::<f64>::zeros(pw);
        let mut qw_ab = Array1::<f64>::zeros(pw);
        let mut q_tw_a = Array1::<f64>::zeros(pw);
        let mut q_tw_b = Array1::<f64>::zeros(pw);
        let mut q_lw_a = Array1::<f64>::zeros(pw);
        let mut q_lw_b = Array1::<f64>::zeros(pw);
        let mut d0_ab = Array1::<f64>::zeros(pw);
        let mut q_tw_ab = Array1::<f64>::zeros(pw);
        let mut q_lw_ab = Array1::<f64>::zeros(pw);
        for row in 0..n {
            let q0 = base_core.q0[row];
            let q = q0 + etaw[row];
            let q0_geom = nonwiggle_q_derivs(eta_t[row], sigma[row]);
            let s_safe = sigma[row];
            let s2 = s_safe * s_safe;
            let s3 = s2 * s_safe;
            let s4 = s3 * s_safe;
            let q0_tl_ls_ls =
                d3s[row] / s2 - 6.0 * ds[row] * d2s[row] / s3 + 6.0 * ds[row].powi(3) / s4;
            let r_sigma = 1.0 / s_safe;

            let q0_a = -r_sigma * dir_a.z_primary_psi[row] - q0 * dir_a.z_ls_psi[row];
            let q0_b = -r_sigma * dir_b.z_primary_psi[row] - q0 * dir_b.z_ls_psi[row];
            let q0_ab = -r_sigma * second_drifts.z_primary_ab[row]
                + r_sigma
                    * (dir_a.z_primary_psi[row] * dir_b.z_ls_psi[row]
                        + dir_b.z_primary_psi[row] * dir_a.z_ls_psi[row])
                + q0 * (dir_a.z_ls_psi[row] * dir_b.z_ls_psi[row] - second_drifts.z_ls_ab[row]);

            let q0_t_a = q0_geom.q_tl * dir_a.z_ls_psi[row];
            let q0_t_b = q0_geom.q_tl * dir_b.z_ls_psi[row];
            let q0_t_ab = q0_geom.q_tl_ls * dir_a.z_ls_psi[row] * dir_b.z_ls_psi[row]
                + q0_geom.q_tl * second_drifts.z_ls_ab[row];
            let q0_ls_a =
                q0_geom.q_tl * dir_a.z_primary_psi[row] + q0_geom.q_ll * dir_a.z_ls_psi[row];
            let q0_ls_b =
                q0_geom.q_tl * dir_b.z_primary_psi[row] + q0_geom.q_ll * dir_b.z_ls_psi[row];
            let q0_ls_ab = -q0_ab;
            let q0_tl_a = q0_geom.q_tl_ls * dir_a.z_ls_psi[row];
            let q0_tl_b = q0_geom.q_tl_ls * dir_b.z_ls_psi[row];
            let q0_tl_ab = q0_tl_ls_ls * dir_a.z_ls_psi[row] * dir_b.z_ls_psi[row]
                + q0_geom.q_tl_ls * second_drifts.z_ls_ab[row];
            let q0_ll_a =
                q0_geom.q_tl_ls * dir_a.z_primary_psi[row] + q0_geom.q_ll_ls * dir_a.z_ls_psi[row];
            let q0_ll_b =
                q0_geom.q_tl_ls * dir_b.z_primary_psi[row] + q0_geom.q_ll_ls * dir_b.z_ls_psi[row];
            let q0_ll_ab = q0_ab;

            let m_a = g2[row] * q0_a;
            let m_b = g2[row] * q0_b;
            let m_ab = g3[row] * q0_a * q0_b + g2[row] * q0_ab;
            let g2_a = g3[row] * q0_a;
            let g2_b = g3[row] * q0_b;
            let g2_ab = g4[row] * q0_a * q0_b + g3[row] * q0_ab;

            let q_a = m[row] * q0_a;
            let q_b = m[row] * q0_b;
            let q_ab = m[row] * q0_ab + g2[row] * q0_a * q0_b;
            let q_t = m[row] * q0_geom.q_t;
            let q_ls = m[row] * q0_geom.q_ls;
            let q_tt = g2[row] * q0_geom.q_t * q0_geom.q_t;
            let q_tl = g2[row] * q0_geom.q_t * q0_geom.q_ls + m[row] * q0_geom.q_tl;
            let q_ll = g2[row] * q0_geom.q_ls * q0_geom.q_ls + m[row] * q0_geom.q_ll;
            let q_t_a = m_a * q0_geom.q_t + m[row] * q0_t_a;
            let q_t_b = m_b * q0_geom.q_t + m[row] * q0_t_b;
            let q_ls_a = m_a * q0_geom.q_ls + m[row] * q0_ls_a;
            let q_ls_b = m_b * q0_geom.q_ls + m[row] * q0_ls_b;
            let q_t_ab = m_ab * q0_geom.q_t + m_a * q0_t_b + m_b * q0_t_a + m[row] * q0_t_ab;
            let q_ls_ab = m_ab * q0_geom.q_ls + m_a * q0_ls_b + m_b * q0_ls_a + m[row] * q0_ls_ab;
            let q_tt_a = g2_a * q0_geom.q_t * q0_geom.q_t + g2[row] * 2.0 * q0_geom.q_t * q0_t_a;
            let q_tt_b = g2_b * q0_geom.q_t * q0_geom.q_t + g2[row] * 2.0 * q0_geom.q_t * q0_t_b;
            let q_tt_ab = g2_ab * q0_geom.q_t * q0_geom.q_t
                + g2_a * 2.0 * q0_geom.q_t * q0_t_b
                + g2_b * 2.0 * q0_geom.q_t * q0_t_a
                + g2[row] * (2.0 * q0_t_a * q0_t_b + 2.0 * q0_geom.q_t * q0_t_ab);
            let q_tl_a = g2_a * q0_geom.q_t * q0_geom.q_ls
                + g2[row] * (q0_t_a * q0_geom.q_ls + q0_geom.q_t * q0_ls_a)
                + m_a * q0_geom.q_tl
                + m[row] * q0_tl_a;
            let q_tl_b = g2_b * q0_geom.q_t * q0_geom.q_ls
                + g2[row] * (q0_t_b * q0_geom.q_ls + q0_geom.q_t * q0_ls_b)
                + m_b * q0_geom.q_tl
                + m[row] * q0_tl_b;
            let q_tl_ab = g2_ab * q0_geom.q_t * q0_geom.q_ls
                + g2_a * (q0_t_b * q0_geom.q_ls + q0_geom.q_t * q0_ls_b)
                + g2_b * (q0_t_a * q0_geom.q_ls + q0_geom.q_t * q0_ls_a)
                + g2[row]
                    * (q0_t_ab * q0_geom.q_ls
                        + q0_t_a * q0_ls_b
                        + q0_t_b * q0_ls_a
                        + q0_geom.q_t * q0_ls_ab)
                + m_ab * q0_geom.q_tl
                + m_a * q0_tl_b
                + m_b * q0_tl_a
                + m[row] * q0_tl_ab;
            let q_ll_a = g2_a * q0_geom.q_ls * q0_geom.q_ls
                + g2[row] * 2.0 * q0_geom.q_ls * q0_ls_a
                + m_a * q0_geom.q_ll
                + m[row] * q0_ll_a;
            let q_ll_b = g2_b * q0_geom.q_ls * q0_geom.q_ls
                + g2[row] * 2.0 * q0_geom.q_ls * q0_ls_b
                + m_b * q0_geom.q_ll
                + m[row] * q0_ll_b;
            let q_ll_ab = g2_ab * q0_geom.q_ls * q0_geom.q_ls
                + g2_a * 2.0 * q0_geom.q_ls * q0_ls_b
                + g2_b * 2.0 * q0_geom.q_ls * q0_ls_a
                + g2[row] * (2.0 * q0_ls_a * q0_ls_b + 2.0 * q0_geom.q_ls * q0_ls_ab)
                + m_ab * q0_geom.q_ll
                + m_a * q0_ll_b
                + m_b * q0_ll_a
                + m[row] * q0_ll_ab;

            let brow = b0.row(row);
            let drow = d0.row(row);
            let ddrow = dd0.row(row);
            let d3row = d3_basis.row(row);
            qw_a.fill(0.0);
            qw_a.scaled_add(q0_a, &drow);
            qw_b.fill(0.0);
            qw_b.scaled_add(q0_b, &drow);
            qw_ab.fill(0.0);
            qw_ab.scaled_add(q0_a * q0_b, &ddrow);
            qw_ab.scaled_add(q0_ab, &drow);
            q_tw_a.fill(0.0);
            q_tw_a.scaled_add(q0_a * q0_geom.q_t, &ddrow);
            q_tw_a.scaled_add(q0_t_a, &drow);
            q_tw_b.fill(0.0);
            q_tw_b.scaled_add(q0_b * q0_geom.q_t, &ddrow);
            q_tw_b.scaled_add(q0_t_b, &drow);
            q_lw_a.fill(0.0);
            q_lw_a.scaled_add(q0_a * q0_geom.q_ls, &ddrow);
            q_lw_a.scaled_add(q0_ls_a, &drow);
            q_lw_b.fill(0.0);
            q_lw_b.scaled_add(q0_b * q0_geom.q_ls, &ddrow);
            q_lw_b.scaled_add(q0_ls_b, &drow);
            d0_ab.fill(0.0);
            d0_ab.scaled_add(q0_a * q0_b, &d3row);
            d0_ab.scaled_add(q0_ab, &ddrow);
            q_tw_ab.fill(0.0);
            q_tw_ab.scaled_add(q0_geom.q_t, &d0_ab);
            q_tw_ab.scaled_add(q0_b * q0_t_a, &ddrow);
            q_tw_ab.scaled_add(q0_a * q0_t_b, &ddrow);
            q_tw_ab.scaled_add(q0_t_ab, &drow);
            q_lw_ab.fill(0.0);
            q_lw_ab.scaled_add(q0_geom.q_ls, &d0_ab);
            q_lw_ab.scaled_add(q0_b * q0_ls_a, &ddrow);
            q_lw_ab.scaled_add(q0_a * q0_ls_b, &ddrow);
            q_lw_ab.scaled_add(q0_ls_ab, &drow);

            let (loss_1, loss_2, loss_3) = binomial_neglog_q_derivatives_dispatch(
                self.y[row],
                self.weights[row],
                q,
                core.mu[row],
                core.dmu_dq[row],
                core.d2mu_dq2[row],
                core.d3mu_dq3[row],
                &self.link_kind,
            );
            let loss_4 = binomial_neglog_q_fourth_derivative_dispatch(
                self.y[row],
                self.weights[row],
                q,
                core.mu[row],
                core.dmu_dq[row],
                core.d2mu_dq2[row],
                core.d3mu_dq3[row],
                &self.link_kind,
            )?;
            objective_psi_psi += loss_2 * q_a * q_b + loss_1 * q_ab;

            let xtr = x_t.row(row);
            let xlsr = x_ls.row(row);
            let xta = x_t_a_map.row_vector(row)?;
            let xtb = x_t_b_map.row_vector(row)?;
            let xlsa = x_ls_a_map.row_vector(row)?;
            let xlsb = x_ls_b_map.row_vector(row)?;
            let xtab = x_t_ab_map.row_vector(row)?;
            let xlsab = x_ls_ab_map.row_vector(row)?;

            b.fill(0.0);
            b.slice_mut(s![0..pt]).scaled_add(q_t, &xtr);
            b.slice_mut(s![pt..pt + pls]).scaled_add(q_ls, &xlsr);
            b.slice_mut(s![pt + pls..]).assign(&brow);
            c_a.fill(0.0);
            c_a.slice_mut(s![0..pt]).scaled_add(q_t_a, &xtr);
            c_a.slice_mut(s![0..pt]).scaled_add(q_t, &xta.view());
            c_a.slice_mut(s![pt..pt + pls]).scaled_add(q_ls_a, &xlsr);
            c_a.slice_mut(s![pt..pt + pls])
                .scaled_add(q_ls, &xlsa.view());
            c_a.slice_mut(s![pt + pls..]).assign(&qw_a);
            c_b.fill(0.0);
            c_b.slice_mut(s![0..pt]).scaled_add(q_t_b, &xtr);
            c_b.slice_mut(s![0..pt]).scaled_add(q_t, &xtb.view());
            c_b.slice_mut(s![pt..pt + pls]).scaled_add(q_ls_b, &xlsr);
            c_b.slice_mut(s![pt..pt + pls])
                .scaled_add(q_ls, &xlsb.view());
            c_b.slice_mut(s![pt + pls..]).assign(&qw_b);
            c_ab.fill(0.0);
            c_ab.slice_mut(s![0..pt]).scaled_add(q_t_ab, &xtr);
            c_ab.slice_mut(s![0..pt]).scaled_add(q_t_b, &xta.view());
            c_ab.slice_mut(s![0..pt]).scaled_add(q_t_a, &xtb.view());
            c_ab.slice_mut(s![0..pt]).scaled_add(q_t, &xtab.view());
            c_ab.slice_mut(s![pt..pt + pls]).scaled_add(q_ls_ab, &xlsr);
            c_ab.slice_mut(s![pt..pt + pls])
                .scaled_add(q_ls_b, &xlsa.view());
            c_ab.slice_mut(s![pt..pt + pls])
                .scaled_add(q_ls_a, &xlsb.view());
            c_ab.slice_mut(s![pt..pt + pls])
                .scaled_add(q_ls, &xlsab.view());
            c_ab.slice_mut(s![pt + pls..]).assign(&qw_ab);

            score_psi_psi.scaled_add(loss_1, &c_ab);
            score_psi_psi.scaled_add(loss_2 * q_b, &c_a);
            score_psi_psi.scaled_add(loss_2 * q_a, &c_b);
            score_psi_psi.scaled_add(loss_2 * q_ab + loss_3 * q_a * q_b, &b);

            q_mat.fill(0.0);
            r_a.fill(0.0);
            r_b.fill(0.0);
            r_ab.fill(0.0);
            scaled_outer_add(q_mat.slice_mut(s![0..pt, 0..pt]), q_tt, xtr, xtr);
            scaled_outer_add(q_mat.slice_mut(s![0..pt, pt..pt + pls]), q_tl, xtr, xlsr);
            scaled_outer_add(
                q_mat.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                q_ll,
                xlsr,
                xlsr,
            );
            scaled_outer_add(
                q_mat.slice_mut(s![0..pt, pt + pls..]),
                q0_geom.q_t,
                xtr,
                drow,
            );
            scaled_outer_add(
                q_mat.slice_mut(s![pt..pt + pls, pt + pls..]),
                q0_geom.q_ls,
                xlsr,
                drow,
            );
            mirror_upper_to_lower(&mut q_mat);

            scaled_outer_add(r_a.slice_mut(s![0..pt, 0..pt]), q_tt_a, xtr, xtr);
            scaled_outer_add(r_a.slice_mut(s![0..pt, 0..pt]), q_tt, xta.view(), xtr);
            scaled_outer_add(r_a.slice_mut(s![0..pt, 0..pt]), q_tt, xtr, xta.view());
            scaled_outer_add(r_a.slice_mut(s![0..pt, pt..pt + pls]), q_tl_a, xtr, xlsr);
            scaled_outer_add(
                r_a.slice_mut(s![0..pt, pt..pt + pls]),
                q_tl,
                xta.view(),
                xlsr,
            );
            scaled_outer_add(
                r_a.slice_mut(s![0..pt, pt..pt + pls]),
                q_tl,
                xtr,
                xlsa.view(),
            );
            scaled_outer_add(
                r_a.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                q_ll_a,
                xlsr,
                xlsr,
            );
            scaled_outer_add(
                r_a.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                q_ll,
                xlsa.view(),
                xlsr,
            );
            scaled_outer_add(
                r_a.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                q_ll,
                xlsr,
                xlsa.view(),
            );
            scaled_outer_add(
                r_a.slice_mut(s![0..pt, pt + pls..]),
                q0_geom.q_t,
                xta.view(),
                drow,
            );
            scaled_outer_add(
                r_a.slice_mut(s![0..pt, pt + pls..]),
                1.0,
                xtr,
                q_tw_a.view(),
            );
            scaled_outer_add(
                r_a.slice_mut(s![pt..pt + pls, pt + pls..]),
                q0_geom.q_ls,
                xlsa.view(),
                drow,
            );
            scaled_outer_add(
                r_a.slice_mut(s![pt..pt + pls, pt + pls..]),
                1.0,
                xlsr,
                q_lw_a.view(),
            );
            mirror_upper_to_lower(&mut r_a);

            scaled_outer_add(r_b.slice_mut(s![0..pt, 0..pt]), q_tt_b, xtr, xtr);
            scaled_outer_add(r_b.slice_mut(s![0..pt, 0..pt]), q_tt, xtb.view(), xtr);
            scaled_outer_add(r_b.slice_mut(s![0..pt, 0..pt]), q_tt, xtr, xtb.view());
            scaled_outer_add(r_b.slice_mut(s![0..pt, pt..pt + pls]), q_tl_b, xtr, xlsr);
            scaled_outer_add(
                r_b.slice_mut(s![0..pt, pt..pt + pls]),
                q_tl,
                xtb.view(),
                xlsr,
            );
            scaled_outer_add(
                r_b.slice_mut(s![0..pt, pt..pt + pls]),
                q_tl,
                xtr,
                xlsb.view(),
            );
            scaled_outer_add(
                r_b.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                q_ll_b,
                xlsr,
                xlsr,
            );
            scaled_outer_add(
                r_b.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                q_ll,
                xlsb.view(),
                xlsr,
            );
            scaled_outer_add(
                r_b.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                q_ll,
                xlsr,
                xlsb.view(),
            );
            scaled_outer_add(
                r_b.slice_mut(s![0..pt, pt + pls..]),
                q0_geom.q_t,
                xtb.view(),
                drow,
            );
            scaled_outer_add(
                r_b.slice_mut(s![0..pt, pt + pls..]),
                1.0,
                xtr,
                q_tw_b.view(),
            );
            scaled_outer_add(
                r_b.slice_mut(s![pt..pt + pls, pt + pls..]),
                q0_geom.q_ls,
                xlsb.view(),
                drow,
            );
            scaled_outer_add(
                r_b.slice_mut(s![pt..pt + pls, pt + pls..]),
                1.0,
                xlsr,
                q_lw_b.view(),
            );
            mirror_upper_to_lower(&mut r_b);

            scaled_outer_add(r_ab.slice_mut(s![0..pt, 0..pt]), q_tt_ab, xtr, xtr);
            scaled_outer_add(r_ab.slice_mut(s![0..pt, 0..pt]), q_tt_b, xta.view(), xtr);
            scaled_outer_add(r_ab.slice_mut(s![0..pt, 0..pt]), q_tt_b, xtr, xta.view());
            scaled_outer_add(r_ab.slice_mut(s![0..pt, 0..pt]), q_tt_a, xtb.view(), xtr);
            scaled_outer_add(r_ab.slice_mut(s![0..pt, 0..pt]), q_tt_a, xtr, xtb.view());
            scaled_outer_add(r_ab.slice_mut(s![0..pt, 0..pt]), q_tt, xtab.view(), xtr);
            scaled_outer_add(r_ab.slice_mut(s![0..pt, 0..pt]), q_tt, xtr, xtab.view());
            scaled_outer_add(
                r_ab.slice_mut(s![0..pt, 0..pt]),
                q_tt,
                xta.view(),
                xtb.view(),
            );
            scaled_outer_add(
                r_ab.slice_mut(s![0..pt, 0..pt]),
                q_tt,
                xtb.view(),
                xta.view(),
            );

            scaled_outer_add(r_ab.slice_mut(s![0..pt, pt..pt + pls]), q_tl_ab, xtr, xlsr);
            scaled_outer_add(
                r_ab.slice_mut(s![0..pt, pt..pt + pls]),
                q_tl_b,
                xta.view(),
                xlsr,
            );
            scaled_outer_add(
                r_ab.slice_mut(s![0..pt, pt..pt + pls]),
                q_tl_b,
                xtr,
                xlsa.view(),
            );
            scaled_outer_add(
                r_ab.slice_mut(s![0..pt, pt..pt + pls]),
                q_tl_a,
                xtb.view(),
                xlsr,
            );
            scaled_outer_add(
                r_ab.slice_mut(s![0..pt, pt..pt + pls]),
                q_tl_a,
                xtr,
                xlsb.view(),
            );
            scaled_outer_add(
                r_ab.slice_mut(s![0..pt, pt..pt + pls]),
                q_tl,
                xtab.view(),
                xlsr,
            );
            scaled_outer_add(
                r_ab.slice_mut(s![0..pt, pt..pt + pls]),
                q_tl,
                xtr,
                xlsab.view(),
            );
            scaled_outer_add(
                r_ab.slice_mut(s![0..pt, pt..pt + pls]),
                q_tl,
                xta.view(),
                xlsb.view(),
            );
            scaled_outer_add(
                r_ab.slice_mut(s![0..pt, pt..pt + pls]),
                q_tl,
                xtb.view(),
                xlsa.view(),
            );

            scaled_outer_add(
                r_ab.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                q_ll_ab,
                xlsr,
                xlsr,
            );
            scaled_outer_add(
                r_ab.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                q_ll_b,
                xlsa.view(),
                xlsr,
            );
            scaled_outer_add(
                r_ab.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                q_ll_b,
                xlsr,
                xlsa.view(),
            );
            scaled_outer_add(
                r_ab.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                q_ll_a,
                xlsb.view(),
                xlsr,
            );
            scaled_outer_add(
                r_ab.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                q_ll_a,
                xlsr,
                xlsb.view(),
            );
            scaled_outer_add(
                r_ab.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                q_ll,
                xlsab.view(),
                xlsr,
            );
            scaled_outer_add(
                r_ab.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                q_ll,
                xlsr,
                xlsab.view(),
            );
            scaled_outer_add(
                r_ab.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                q_ll,
                xlsa.view(),
                xlsb.view(),
            );
            scaled_outer_add(
                r_ab.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                q_ll,
                xlsb.view(),
                xlsa.view(),
            );
            scaled_outer_add(
                r_ab.slice_mut(s![0..pt, pt + pls..]),
                q0_geom.q_t,
                xtab.view(),
                drow,
            );
            scaled_outer_add(
                r_ab.slice_mut(s![0..pt, pt + pls..]),
                1.0,
                xta.view(),
                q_tw_b.view(),
            );
            scaled_outer_add(
                r_ab.slice_mut(s![0..pt, pt + pls..]),
                1.0,
                xtb.view(),
                q_tw_a.view(),
            );
            scaled_outer_add(
                r_ab.slice_mut(s![0..pt, pt + pls..]),
                1.0,
                xtr,
                q_tw_ab.view(),
            );
            scaled_outer_add(
                r_ab.slice_mut(s![pt..pt + pls, pt + pls..]),
                q0_geom.q_ls,
                xlsab.view(),
                drow,
            );
            scaled_outer_add(
                r_ab.slice_mut(s![pt..pt + pls, pt + pls..]),
                1.0,
                xlsa.view(),
                q_lw_b.view(),
            );
            scaled_outer_add(
                r_ab.slice_mut(s![pt..pt + pls, pt + pls..]),
                1.0,
                xlsb.view(),
                q_lw_a.view(),
            );
            scaled_outer_add(
                r_ab.slice_mut(s![pt..pt + pls, pt + pls..]),
                1.0,
                xlsr,
                q_lw_ab.view(),
            );
            mirror_upper_to_lower(&mut r_ab);

            hessian_psi_psi.scaled_add(loss_1, &r_ab);
            hessian_psi_psi.scaled_add(loss_2 * q_b, &r_a);
            hessian_psi_psi.scaled_add(loss_2 * q_a, &r_b);
            scaled_outer_add(hessian_psi_psi.view_mut(), loss_2, c_ab.view(), b.view());
            scaled_outer_add(hessian_psi_psi.view_mut(), loss_2, b.view(), c_ab.view());
            scaled_outer_add(hessian_psi_psi.view_mut(), loss_2, c_a.view(), c_b.view());
            scaled_outer_add(hessian_psi_psi.view_mut(), loss_2, c_b.view(), c_a.view());
            hessian_psi_psi.scaled_add(loss_2 * q_ab, &q_mat);
            scaled_outer_add(
                hessian_psi_psi.view_mut(),
                loss_3 * q_b,
                c_a.view(),
                b.view(),
            );
            scaled_outer_add(
                hessian_psi_psi.view_mut(),
                loss_3 * q_b,
                b.view(),
                c_a.view(),
            );
            scaled_outer_add(
                hessian_psi_psi.view_mut(),
                loss_3 * q_a,
                c_b.view(),
                b.view(),
            );
            scaled_outer_add(
                hessian_psi_psi.view_mut(),
                loss_3 * q_a,
                b.view(),
                c_b.view(),
            );
            hessian_psi_psi.scaled_add(loss_3 * q_a * q_b, &q_mat);
            scaled_outer_add(
                hessian_psi_psi.view_mut(),
                loss_4 * q_a * q_b + loss_3 * q_ab,
                b.view(),
                b.view(),
            );
        }

        Ok(gam_problem::ExactNewtonJointPsiSecondOrderTerms {
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
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some(dir_a) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_index,
            x_t,
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
                x_t,
                x_ls,
            )?,
        ))
    }

    pub(crate) fn exact_newton_joint_psihessian_directional_derivative_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        dir_a: &LocationScaleJointPsiDirection,
        d_beta_flat: &Array1<f64>,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Array2<f64>, String> {
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(etaw),
            &self.link_kind,
        )?;
        let base_core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let b0 = self.wiggle_design(base_core.q0.view())?;
        let d0 =
            self.wiggle_basiswith_options(base_core.q0.view(), BasisOptions::first_derivative())?;
        let dd0 =
            self.wiggle_basiswith_options(base_core.q0.view(), BasisOptions::second_derivative())?;
        let d3_basis = self.wiggle_d3basis_constrained(base_core.q0.view())?;
        let d4q = self.wiggle_d4q_dq04(base_core.q0.view(), betaw.view())?;
        let pw = b0.ncols();
        let layout = GamlssBetaLayout::withwiggle(pt, pls, pw);
        let (u_t, u_ls, uw) = layout.split_three(
            d_beta_flat,
            "wiggle joint psi hessian directional derivative",
        )?;
        let total = pt + pls + pw;
        if d0.ncols() != betaw.len()
            || dd0.ncols() != betaw.len()
            || d3_basis.ncols() != betaw.len()
        {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "wiggle derivative/beta mismatch in joint psi mixed drift: B'={} B''={} B'''={} betaw={}",
                d0.ncols(),
                dd0.ncols(),
                d3_basis.ncols(),
                betaw.len()
            ) }.into());
        }
        let xi_t = x_t.dot(&u_t);
        let xi_ls = x_ls.dot(&u_ls);
        let x_t_map = dir_a.x_primary_psi.as_linear_map_ref();
        let x_ls_map = dir_a.x_ls_psi.as_linear_map_ref();
        let m = d0.dot(betaw) + 1.0;
        let g2 = dd0.dot(betaw);
        let g3 = self.wiggle_d3q_dq03(base_core.q0.view(), betaw.view())?;
        let g4 = d4q;
        let (sigma, ds, d2s, d3s, d4s) = exp_sigma_derivs_up_to_fourth_array(eta_ls.view());

        // Exact likelihood-side mixed drift T_a[u] = D_beta H_{psi_a}^{(D)}[u].
        //
        // The unified outer Hessian in custom_family.rs uses
        //   ddot H_ij = H_ij + T_i[beta_j] + T_j[beta_i]
        //             + D_beta H[beta_ij] + D_beta^2 H[beta_i, beta_j].
        //
        // For wiggle we still use the same scalar-loss row kernel as non-wiggle;
        // only the location-side row changes to z_r = [x_{t,r}; B_r(q0)] with
        // q = q0 + betaw^T B(q0), q0 = -eta_t * exp(-eta_ls).
        let mut out = Array2::<f64>::zeros((total, total));
        let mut b = Array1::<f64>::zeros(total);
        let mut c_a = Array1::<f64>::zeros(total);
        let mut gamma = Array1::<f64>::zeros(total);
        let mut gamma_a = Array1::<f64>::zeros(total);
        let mut q_mat = Array2::<f64>::zeros((total, total));
        let mut r_a = Array2::<f64>::zeros((total, total));
        let mut c_u = Array2::<f64>::zeros((total, total));
        let mut delta_a = Array2::<f64>::zeros((total, total));
        let mut q_tw = Array1::<f64>::zeros(pw);
        let mut q_lw = Array1::<f64>::zeros(pw);
        let mut qw_a = Array1::<f64>::zeros(pw);
        let mut q_tw_a = Array1::<f64>::zeros(pw);
        let mut q_lw_a = Array1::<f64>::zeros(pw);
        let mut dq_tw_u = Array1::<f64>::zeros(pw);
        let mut dq_lw_u = Array1::<f64>::zeros(pw);
        let mut dq_tw_a_u = Array1::<f64>::zeros(pw);
        let mut dq_lw_a_u = Array1::<f64>::zeros(pw);
        for row in 0..n {
            let q = core.q0[row] + etaw[row];
            let (loss_1, loss_2, loss_3) = binomial_neglog_q_derivatives_dispatch(
                self.y[row],
                self.weights[row],
                q,
                core.mu[row],
                core.dmu_dq[row],
                core.d2mu_dq2[row],
                core.d3mu_dq3[row],
                &self.link_kind,
            );
            let loss_4 = binomial_neglog_q_fourth_derivative_dispatch(
                self.y[row],
                self.weights[row],
                q,
                core.mu[row],
                core.dmu_dq[row],
                core.d2mu_dq2[row],
                core.d3mu_dq3[row],
                &self.link_kind,
            )?;
            let q0 = nonwiggle_q_derivs(eta_t[row], sigma[row]);
            let s_safe = sigma[row];
            let s2 = s_safe * s_safe;
            let s3 = s2 * s_safe;
            let s4 = s3 * s_safe;
            let s5 = s4 * s_safe;
            let q0_tl_ls_ls = d3s[row] / s2 - 6.0 * ds[row] * d2s[row] / s3
                + 6.0 * ds[row] * ds[row] * ds[row] / s4;
            let q0_tl_ls_ls_ls =
                d4s[row] / s2 - 8.0 * ds[row] * d3s[row] / s3 - 6.0 * d2s[row] * d2s[row] / s3
                    + 36.0 * ds[row] * ds[row] * d2s[row] / s4
                    - 24.0 * ds[row] * ds[row] * ds[row] * ds[row] / s5;
            let q0_ll_ls_ls = eta_t[row] * q0_tl_ls_ls_ls;

            let xtr = x_t.row(row);
            let xlsr = x_ls.row(row);
            let xta = x_t_map.row_vector(row)?;
            let xlsa = x_ls_map.row_vector(row)?;
            let br = b0.row(row);
            let dr = d0.row(row);
            let ddr = dd0.row(row);
            let d3r = d3_basis.row(row);

            let xi_t_i = xi_t[row];
            let xi_ls_i = xi_ls[row];
            let xi_ta_i = xta.dot(&u_t);
            let xi_lsa_i = xlsa.dot(&u_ls);
            let d_dot_u = dr.dot(&uw);
            let dd_dot_u = ddr.dot(&uw);
            let d3_dot_u = d3r.dot(&uw);

            let dq0_u = q0.q_t * xi_t_i + q0.q_ls * xi_ls_i;
            let dq0_t_u = q0.q_tl * xi_ls_i;
            let dq0_ls_u = q0.q_tl * xi_t_i + q0.q_ll * xi_ls_i;
            let dq0_tl_u = q0.q_tl_ls * xi_ls_i;
            let dq0_ll_u = q0.q_tl_ls * xi_t_i + q0.q_ll_ls * xi_ls_i;
            let dq0_tl_ls_u = q0_tl_ls_ls * xi_ls_i;
            let dq0_ll_ls_u = q0_tl_ls_ls * xi_t_i + q0_ll_ls_ls * xi_ls_i;

            let q0_a = -q0.q_t * dir_a.z_primary_psi[row] - q0.q_ls * dir_a.z_ls_psi[row];
            let q0_t_a = q0.q_tl_ls * dir_a.z_ls_psi[row];
            let q0_ls_a = q0.q_tl_ls * dir_a.z_primary_psi[row] + q0.q_ll_ls * dir_a.z_ls_psi[row];
            let q0_tl_a = q0.q_tl_ls * dir_a.z_ls_psi[row];
            let q0_ll_a = q0.q_tl_ls * dir_a.z_primary_psi[row] + q0.q_ll_ls * dir_a.z_ls_psi[row];
            let dq0_a_u = q0_t_a * xi_t_i + q0_ls_a * xi_ls_i;
            let dq0_t_a_u = dq0_tl_ls_u * dir_a.z_ls_psi[row];
            let dq0_ls_a_u =
                dq0_tl_ls_u * dir_a.z_primary_psi[row] + dq0_ll_ls_u * dir_a.z_ls_psi[row];
            let dq0_tl_a_u = dq0_tl_ls_u * dir_a.z_ls_psi[row];
            let dq0_ll_a_u =
                dq0_tl_ls_u * dir_a.z_primary_psi[row] + dq0_ll_ls_u * dir_a.z_ls_psi[row];

            let q_t = m[row] * q0.q_t;
            let q_ls = m[row] * q0.q_ls;
            let q_tt = g2[row] * q0.q_t * q0.q_t;
            let q_tl = g2[row] * q0.q_t * q0.q_ls + m[row] * q0.q_tl;
            let q_ll = g2[row] * q0.q_ls * q0.q_ls + m[row] * q0.q_ll;
            q_tw.fill(0.0);
            q_tw.scaled_add(q0.q_t, &dr);
            q_lw.fill(0.0);
            q_lw.scaled_add(q0.q_ls, &dr);

            let dm_u = g2[row] * dq0_u + d_dot_u;
            let dg2_u = g3[row] * dq0_u + dd_dot_u;
            let dg3_u = g4[row] * dq0_u + d3_dot_u;

            let q_a = m[row] * q0_a;
            let q_t_a = g2[row] * q0_a * q0.q_t + m[row] * q0_t_a;
            let q_ls_a = g2[row] * q0_a * q0.q_ls + m[row] * q0_ls_a;
            let q_tt_a = g3[row] * q0_a * q0.q_t * q0.q_t + g2[row] * (2.0 * q0.q_t * q0_t_a);
            let q_tl_a = g3[row] * q0_a * q0.q_t * q0.q_ls
                + g2[row] * (q0_t_a * q0.q_ls + q0.q_t * q0_ls_a + q0_a * q0.q_tl)
                + m[row] * q0_tl_a;
            let q_ll_a = g3[row] * q0_a * q0.q_ls * q0.q_ls
                + g2[row] * (2.0 * q0.q_ls * q0_ls_a + q0_a * q0.q_ll)
                + m[row] * q0_ll_a;
            qw_a.fill(0.0);
            qw_a.scaled_add(q0_a, &dr);
            q_tw_a.fill(0.0);
            q_tw_a.scaled_add(q0_a * q0.q_t, &ddr);
            q_tw_a.scaled_add(q0_t_a, &dr);
            q_lw_a.fill(0.0);
            q_lw_a.scaled_add(q0_a * q0.q_ls, &ddr);
            q_lw_a.scaled_add(q0_ls_a, &dr);

            let dq_tt_u = dg2_u * q0.q_t * q0.q_t + g2[row] * (2.0 * q0.q_t * dq0_t_u);
            let dq_tl_u = dg2_u * q0.q_t * q0.q_ls
                + g2[row] * (dq0_t_u * q0.q_ls + q0.q_t * dq0_ls_u)
                + dm_u * q0.q_tl
                + m[row] * dq0_tl_u;
            let dq_ll_u = dg2_u * q0.q_ls * q0.q_ls
                + g2[row] * (2.0 * q0.q_ls * dq0_ls_u)
                + dm_u * q0.q_ll
                + m[row] * dq0_ll_u;
            dq_tw_u.fill(0.0);
            dq_tw_u.scaled_add(dq0_u * q0.q_t, &ddr);
            dq_tw_u.scaled_add(dq0_t_u, &dr);
            dq_lw_u.fill(0.0);
            dq_lw_u.scaled_add(dq0_u * q0.q_ls, &ddr);
            dq_lw_u.scaled_add(dq0_ls_u, &dr);

            let dq_tt_a_u = dg3_u * q0_a * q0.q_t * q0.q_t
                + g3[row] * (dq0_a_u * q0.q_t * q0.q_t + 2.0 * q0_a * q0.q_t * dq0_t_u)
                + dg2_u * (2.0 * q0.q_t * q0_t_a)
                + g2[row] * (2.0 * dq0_t_u * q0_t_a + 2.0 * q0.q_t * dq0_t_a_u);
            let dq_tl_a_u = dg3_u * q0_a * q0.q_t * q0.q_ls
                + g3[row]
                    * (dq0_a_u * q0.q_t * q0.q_ls
                        + q0_a * dq0_t_u * q0.q_ls
                        + q0_a * q0.q_t * dq0_ls_u)
                + dg2_u * (q0_t_a * q0.q_ls + q0.q_t * q0_ls_a + q0_a * q0.q_tl)
                + g2[row]
                    * (dq0_t_a_u * q0.q_ls
                        + q0_t_a * dq0_ls_u
                        + dq0_t_u * q0_ls_a
                        + q0.q_t * dq0_ls_a_u
                        + dq0_a_u * q0.q_tl
                        + q0_a * dq0_tl_u)
                + dm_u * q0_tl_a
                + m[row] * dq0_tl_a_u;
            let dq_ll_a_u = dg3_u * q0_a * q0.q_ls * q0.q_ls
                + g3[row] * (dq0_a_u * q0.q_ls * q0.q_ls + 2.0 * q0_a * q0.q_ls * dq0_ls_u)
                + dg2_u * (2.0 * q0.q_ls * q0_ls_a + q0_a * q0.q_ll)
                + g2[row]
                    * (2.0 * dq0_ls_u * q0_ls_a
                        + 2.0 * q0.q_ls * dq0_ls_a_u
                        + dq0_a_u * q0.q_ll
                        + q0_a * dq0_ll_u)
                + dm_u * q0_ll_a
                + m[row] * dq0_ll_a_u;
            dq_tw_a_u.fill(0.0);
            dq_tw_a_u.scaled_add(dq0_u * q0_a * q0.q_t, &d3r);
            dq_tw_a_u.scaled_add(dq0_a_u * q0.q_t + q0_a * dq0_t_u + dq0_u * q0_t_a, &ddr);
            dq_tw_a_u.scaled_add(dq0_t_a_u, &dr);
            dq_lw_a_u.fill(0.0);
            dq_lw_a_u.scaled_add(dq0_u * q0_a * q0.q_ls, &d3r);
            dq_lw_a_u.scaled_add(dq0_a_u * q0.q_ls + q0_a * dq0_ls_u + dq0_u * q0_ls_a, &ddr);
            dq_lw_a_u.scaled_add(dq0_ls_a_u, &dr);

            b.fill(0.0);
            b.slice_mut(s![0..pt]).scaled_add(q_t, &xtr);
            b.slice_mut(s![pt..pt + pls]).scaled_add(q_ls, &xlsr);
            b.slice_mut(s![pt + pls..]).assign(&br);

            c_a.fill(0.0);
            c_a.slice_mut(s![0..pt]).scaled_add(q_t_a, &xtr);
            c_a.slice_mut(s![0..pt]).scaled_add(q_t, &xta.view());
            c_a.slice_mut(s![pt..pt + pls]).scaled_add(q_ls_a, &xlsr);
            c_a.slice_mut(s![pt..pt + pls])
                .scaled_add(q_ls, &xlsa.view());
            c_a.slice_mut(s![pt + pls..]).assign(&qw_a);

            gamma.fill(0.0);
            gamma
                .slice_mut(s![0..pt])
                .scaled_add(q_tt * xi_t_i + q_tl * xi_ls_i + q0.q_t * d_dot_u, &xtr);
            gamma
                .slice_mut(s![pt..pt + pls])
                .scaled_add(q_tl * xi_t_i + q_ll * xi_ls_i + q0.q_ls * d_dot_u, &xlsr);
            gamma.slice_mut(s![pt + pls..]).scaled_add(dq0_u, &dr);

            let q_tw_a_dot_u = q_tw_a.dot(&uw);
            let q_lw_a_dot_u = q_lw_a.dot(&uw);
            gamma_a.fill(0.0);
            gamma_a.slice_mut(s![0..pt]).scaled_add(
                q_tt_a * xi_t_i
                    + q_tt * xi_ta_i
                    + q_tl_a * xi_ls_i
                    + q_tl * xi_lsa_i
                    + q_tw_a_dot_u,
                &xtr,
            );
            gamma_a.slice_mut(s![0..pt]).scaled_add(
                q_tt * xi_t_i + q_tl * xi_ls_i + q0.q_t * d_dot_u,
                &xta.view(),
            );
            gamma_a.slice_mut(s![pt..pt + pls]).scaled_add(
                q_tl_a * xi_t_i
                    + q_tl * xi_ta_i
                    + q_ll_a * xi_ls_i
                    + q_ll * xi_lsa_i
                    + q_lw_a_dot_u,
                &xlsr,
            );
            gamma_a.slice_mut(s![pt..pt + pls]).scaled_add(
                q_tl * xi_t_i + q_ll * xi_ls_i + q0.q_ls * d_dot_u,
                &xlsa.view(),
            );
            gamma_a
                .slice_mut(s![pt + pls..])
                .scaled_add(xi_t_i, &q_tw_a);
            gamma_a.slice_mut(s![pt + pls..]).scaled_add(xi_ta_i, &q_tw);
            gamma_a
                .slice_mut(s![pt + pls..])
                .scaled_add(xi_ls_i, &q_lw_a);
            gamma_a
                .slice_mut(s![pt + pls..])
                .scaled_add(xi_lsa_i, &q_lw);

            let alpha = b.dot(d_beta_flat);
            let alpha_a = c_a.dot(d_beta_flat);

            q_mat.fill(0.0);
            scaled_outer_add(q_mat.slice_mut(s![0..pt, 0..pt]), q_tt, xtr, xtr);
            scaled_outer_add(q_mat.slice_mut(s![0..pt, pt..pt + pls]), q_tl, xtr, xlsr);
            scaled_outer_add(
                q_mat.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                q_ll,
                xlsr,
                xlsr,
            );
            scaled_outer_add(
                q_mat.slice_mut(s![0..pt, pt + pls..]),
                1.0,
                xtr,
                q_tw.view(),
            );
            scaled_outer_add(
                q_mat.slice_mut(s![pt..pt + pls, pt + pls..]),
                1.0,
                xlsr,
                q_lw.view(),
            );
            mirror_upper_to_lower(&mut q_mat);

            r_a.fill(0.0);
            scaled_outer_add(r_a.slice_mut(s![0..pt, 0..pt]), q_tt_a, xtr, xtr);
            scaled_outer_add(r_a.slice_mut(s![0..pt, 0..pt]), q_tt, xta.view(), xtr);
            scaled_outer_add(r_a.slice_mut(s![0..pt, 0..pt]), q_tt, xtr, xta.view());
            scaled_outer_add(r_a.slice_mut(s![0..pt, pt..pt + pls]), q_tl_a, xtr, xlsr);
            scaled_outer_add(
                r_a.slice_mut(s![0..pt, pt..pt + pls]),
                q_tl,
                xta.view(),
                xlsr,
            );
            scaled_outer_add(
                r_a.slice_mut(s![0..pt, pt..pt + pls]),
                q_tl,
                xtr,
                xlsa.view(),
            );
            scaled_outer_add(
                r_a.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                q_ll_a,
                xlsr,
                xlsr,
            );
            scaled_outer_add(
                r_a.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                q_ll,
                xlsa.view(),
                xlsr,
            );
            scaled_outer_add(
                r_a.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                q_ll,
                xlsr,
                xlsa.view(),
            );
            scaled_outer_add(
                r_a.slice_mut(s![0..pt, pt + pls..]),
                1.0,
                xta.view(),
                q_tw.view(),
            );
            scaled_outer_add(
                r_a.slice_mut(s![0..pt, pt + pls..]),
                1.0,
                xtr,
                q_tw_a.view(),
            );
            scaled_outer_add(
                r_a.slice_mut(s![pt..pt + pls, pt + pls..]),
                1.0,
                xlsa.view(),
                q_lw.view(),
            );
            scaled_outer_add(
                r_a.slice_mut(s![pt..pt + pls, pt + pls..]),
                1.0,
                xlsr,
                q_lw_a.view(),
            );
            mirror_upper_to_lower(&mut r_a);

            c_u.fill(0.0);
            scaled_outer_add(c_u.slice_mut(s![0..pt, 0..pt]), dq_tt_u, xtr, xtr);
            scaled_outer_add(c_u.slice_mut(s![0..pt, pt..pt + pls]), dq_tl_u, xtr, xlsr);
            scaled_outer_add(
                c_u.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                dq_ll_u,
                xlsr,
                xlsr,
            );
            scaled_outer_add(
                c_u.slice_mut(s![0..pt, pt + pls..]),
                1.0,
                xtr,
                dq_tw_u.view(),
            );
            scaled_outer_add(
                c_u.slice_mut(s![pt..pt + pls, pt + pls..]),
                1.0,
                xlsr,
                dq_lw_u.view(),
            );
            mirror_upper_to_lower(&mut c_u);

            delta_a.fill(0.0);
            scaled_outer_add(delta_a.slice_mut(s![0..pt, 0..pt]), dq_tt_a_u, xtr, xtr);
            scaled_outer_add(
                delta_a.slice_mut(s![0..pt, 0..pt]),
                dq_tt_u,
                xta.view(),
                xtr,
            );
            scaled_outer_add(
                delta_a.slice_mut(s![0..pt, 0..pt]),
                dq_tt_u,
                xtr,
                xta.view(),
            );
            scaled_outer_add(
                delta_a.slice_mut(s![0..pt, pt..pt + pls]),
                dq_tl_a_u,
                xtr,
                xlsr,
            );
            scaled_outer_add(
                delta_a.slice_mut(s![0..pt, pt..pt + pls]),
                dq_tl_u,
                xta.view(),
                xlsr,
            );
            scaled_outer_add(
                delta_a.slice_mut(s![0..pt, pt..pt + pls]),
                dq_tl_u,
                xtr,
                xlsa.view(),
            );
            scaled_outer_add(
                delta_a.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                dq_ll_a_u,
                xlsr,
                xlsr,
            );
            scaled_outer_add(
                delta_a.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                dq_ll_u,
                xlsa.view(),
                xlsr,
            );
            scaled_outer_add(
                delta_a.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                dq_ll_u,
                xlsr,
                xlsa.view(),
            );
            scaled_outer_add(
                delta_a.slice_mut(s![0..pt, pt + pls..]),
                1.0,
                xta.view(),
                dq_tw_u.view(),
            );
            scaled_outer_add(
                delta_a.slice_mut(s![0..pt, pt + pls..]),
                1.0,
                xtr,
                dq_tw_a_u.view(),
            );
            scaled_outer_add(
                delta_a.slice_mut(s![pt..pt + pls, pt + pls..]),
                1.0,
                xlsa.view(),
                dq_lw_u.view(),
            );
            scaled_outer_add(
                delta_a.slice_mut(s![pt..pt + pls, pt + pls..]),
                1.0,
                xlsr,
                dq_lw_a_u.view(),
            );
            mirror_upper_to_lower(&mut delta_a);

            out.scaled_add(loss_1, &delta_a);
            out.scaled_add(loss_2 * alpha, &r_a);
            out.scaled_add(loss_2 * q_a, &c_u);
            scaled_outer_add(out.view_mut(), loss_2, gamma_a.view(), b.view());
            scaled_outer_add(out.view_mut(), loss_2, b.view(), gamma_a.view());
            scaled_outer_add(out.view_mut(), loss_2, gamma.view(), c_a.view());
            scaled_outer_add(out.view_mut(), loss_2, c_a.view(), gamma.view());
            out.scaled_add(loss_2 * alpha_a, &q_mat);
            scaled_outer_add(out.view_mut(), loss_3 * alpha * q_a, b.view(), b.view());
            scaled_outer_add(out.view_mut(), loss_3 * q_a, gamma.view(), b.view());
            scaled_outer_add(out.view_mut(), loss_3 * q_a, b.view(), gamma.view());
            scaled_outer_add(out.view_mut(), loss_3 * alpha, c_a.view(), b.view());
            scaled_outer_add(out.view_mut(), loss_3 * alpha, b.view(), c_a.view());
            out.scaled_add(loss_3 * alpha * q_a, &q_mat);
            scaled_outer_add(
                out.view_mut(),
                loss_4 * alpha * q_a + loss_3 * alpha_a,
                b.view(),
                b.view(),
            );
        }
        mirror_upper_to_lower(&mut out);
        Ok(out)
    }

    /// Build a turnkey wiggle block from a q-seed vector and knot settings.
    /// Returns both the block input and the generated knot vector.
    pub fn buildwiggle_block_input(
        q_seed: ArrayView1<'_, f64>,
        degree: usize,
        num_internal_knots: usize,
        penalty_order: usize,
        double_penalty: bool,
    ) -> Result<(ParameterBlockInput, Array1<f64>), String> {
        let knots = Self::initializewiggle_knots_from_q(q_seed, degree, num_internal_knots)?;
        let block = buildwiggle_block_input_from_knots(
            q_seed,
            &knots,
            degree,
            penalty_order,
            double_penalty,
        )?;
        Ok((block, knots))
    }

    /// Lower the canonical runtime-width row program to the eight structured
    /// order-two coefficient channels consumed by dense and matrix-free paths.
    pub(crate) fn wiggle_order2_rows(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<BinomialWiggleOrder2Rows, String> {
        BinomialLocationScaleWiggleRowProgram::new(self, block_states, 2)?.order2_rows()
    }

    pub(crate) fn expected_wiggle_geometry_inputs<'a>(
        &'a self,
        block_states: &'a [ParameterBlockState],
        specs: Option<&'a [ParameterBlockSpec]>,
    ) -> Result<Option<ExpectedWiggleGeometryInputs<'a>>, String> {
        validate_block_count::<GamlssError>(
            "BinomialLocationScaleWiggleFamily",
            3,
            block_states.len(),
        )?;
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(specs)? else {
            return Ok(None);
        };
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        if eta_t.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason:
                    "BinomialLocationScaleWiggleFamily expected-information input size mismatch"
                        .to_string(),
            }
            .into());
        }
        Ok(Some(ExpectedWiggleGeometryInputs {
            x_t,
            x_ls,
            eta_t,
            eta_ls,
            etaw,
        }))
    }

    pub(crate) fn expected_wiggle_information_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Array2<f64>>, String> {
        let Some(inputs) = self.expected_wiggle_geometry_inputs(block_states, Some(specs))? else {
            return Ok(None);
        };
        let ExpectedWiggleGeometryInputs {
            x_t,
            x_ls,
            eta_t,
            eta_ls,
            etaw,
        } = inputs;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(etaw),
            &self.link_kind,
        )?;
        let b0 = self.wiggle_design(core.q0.view())?;
        let d0 = self.wiggle_basiswith_options(core.q0.view(), BasisOptions::first_derivative())?;
        let m = d0.dot(betaw) + 1.0;
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let pw = b0.ncols();
        let total = pt + pls + pw;
        let mut out = Array2::<f64>::zeros((total, total));
        let mut b = Array1::<f64>::zeros(total);
        for i in 0..self.y.len() {
            let q = core.q0[i] + etaw[i];
            let jet = inverse_link_jet_for_inverse_link(&self.link_kind, q).map_err(|e| {
                format!("BinomialLocationScaleWiggle expected information link jet failed: {e}")
            })?;
            let (f, _, _) = binomial_expected_q_information_derivatives(
                self.weights[i],
                jet.mu,
                jet.d1,
                jet.d2,
                jet.d3,
            );
            if f == 0.0 {
                continue;
            }
            let q0 = nonwiggle_q_derivs(eta_t[i], core.sigma[i]);
            b.fill(0.0);
            b.slice_mut(s![0..pt])
                .scaled_add(m[i] * q0.q_t, &x_t.row(i));
            b.slice_mut(s![pt..pt + pls])
                .scaled_add(m[i] * q0.q_ls, &x_ls.row(i));
            b.slice_mut(s![pt + pls..]).assign(&b0.row(i));
            scaled_outer_add(out.view_mut(), f, b.view(), b.view());
        }
        mirror_upper_to_lower(&mut out);
        Ok(Some(out))
    }

    pub(crate) fn expected_wiggle_information_directional_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some(inputs) = self.expected_wiggle_geometry_inputs(block_states, Some(specs))? else {
            return Ok(None);
        };
        let ExpectedWiggleGeometryInputs {
            x_t,
            x_ls,
            eta_t,
            eta_ls,
            etaw,
        } = inputs;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(etaw),
            &self.link_kind,
        )?;
        let b0 = self.wiggle_design(core.q0.view())?;
        let d0 = self.wiggle_basiswith_options(core.q0.view(), BasisOptions::first_derivative())?;
        let dd0 =
            self.wiggle_basiswith_options(core.q0.view(), BasisOptions::second_derivative())?;
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let pw = b0.ncols();
        let layout = GamlssBetaLayout::withwiggle(pt, pls, pw);
        let (u_t, u_ls, uw) = layout.split_three(d_beta_flat, "expected wiggle d_beta")?;
        let d_eta_t = fast_av(&x_t, &u_t);
        let d_eta_ls = fast_av(&x_ls, &u_ls);
        let m = d0.dot(betaw) + 1.0;
        let g2 = dd0.dot(betaw);
        let total = layout.total();
        let mut out = Array2::<f64>::zeros((total, total));
        let mut b = Array1::<f64>::zeros(total);
        let mut bu = Array1::<f64>::zeros(total);
        for i in 0..self.y.len() {
            let q = core.q0[i] + etaw[i];
            let jet = inverse_link_jet_for_inverse_link(&self.link_kind, q).map_err(|e| {
                format!("BinomialLocationScaleWiggle expected dI link jet failed: {e}")
            })?;
            let (f, f1, _) = binomial_expected_q_information_derivatives(
                self.weights[i],
                jet.mu,
                jet.d1,
                jet.d2,
                jet.d3,
            );
            if f == 0.0 && f1 == 0.0 {
                continue;
            }
            let q0 = nonwiggle_q_derivs(eta_t[i], core.sigma[i]);
            let dq0_u = q0.q_t * d_eta_t[i] + q0.q_ls * d_eta_ls[i];
            let dq0_t_u = q0.q_tl * d_eta_ls[i];
            let dq0_ls_u = q0.q_tl * d_eta_t[i] + q0.q_ll * d_eta_ls[i];
            let bu_w = b0.row(i).dot(&uw);
            let b1u = d0.row(i).dot(&uw);
            let dm_u = g2[i] * dq0_u + b1u;
            let alpha_u = m[i] * dq0_u + bu_w;
            let q_t = m[i] * q0.q_t;
            let q_ls = m[i] * q0.q_ls;
            let dq_t_u = dm_u * q0.q_t + m[i] * dq0_t_u;
            let dq_ls_u = dm_u * q0.q_ls + m[i] * dq0_ls_u;
            b.fill(0.0);
            bu.fill(0.0);
            b.slice_mut(s![0..pt]).scaled_add(q_t, &x_t.row(i));
            b.slice_mut(s![pt..pt + pls]).scaled_add(q_ls, &x_ls.row(i));
            b.slice_mut(s![pt + pls..]).assign(&b0.row(i));
            bu.slice_mut(s![0..pt]).scaled_add(dq_t_u, &x_t.row(i));
            bu.slice_mut(s![pt..pt + pls])
                .scaled_add(dq_ls_u, &x_ls.row(i));
            bu.slice_mut(s![pt + pls..]).scaled_add(dq0_u, &d0.row(i));
            scaled_outer_add(out.view_mut(), f1 * alpha_u, b.view(), b.view());
            scaled_outer_add(out.view_mut(), f, bu.view(), b.view());
            scaled_outer_add(out.view_mut(), f, b.view(), bu.view());
        }
        mirror_upper_to_lower(&mut out);
        Ok(Some(out))
    }

    pub(crate) fn expected_wiggle_information_second_directional_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some(inputs) = self.expected_wiggle_geometry_inputs(block_states, Some(specs))? else {
            return Ok(None);
        };
        let ExpectedWiggleGeometryInputs {
            x_t,
            x_ls,
            eta_t,
            eta_ls,
            etaw,
        } = inputs;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(etaw),
            &self.link_kind,
        )?;
        let b0 = self.wiggle_design(core.q0.view())?;
        let d0 = self.wiggle_basiswith_options(core.q0.view(), BasisOptions::first_derivative())?;
        let dd0 =
            self.wiggle_basiswith_options(core.q0.view(), BasisOptions::second_derivative())?;
        let d3q = self.wiggle_d3q_dq03(core.q0.view(), betaw.view())?;
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let pw = b0.ncols();
        let layout = GamlssBetaLayout::withwiggle(pt, pls, pw);
        let (u_t, u_ls, uw) = layout.split_three(d_beta_u_flat, "expected wiggle d_beta_u")?;
        let (v_t, v_ls, vw) = layout.split_three(d_betav_flat, "expected wiggle d_beta_v")?;
        let d_eta_t_u = fast_av(&x_t, &u_t);
        let d_eta_ls_u = fast_av(&x_ls, &u_ls);
        let d_eta_t_v = fast_av(&x_t, &v_t);
        let d_eta_ls_v = fast_av(&x_ls, &v_ls);
        let m = d0.dot(betaw) + 1.0;
        let g2 = dd0.dot(betaw);
        let total = layout.total();
        let mut out = Array2::<f64>::zeros((total, total));
        let mut b = Array1::<f64>::zeros(total);
        let mut bu = Array1::<f64>::zeros(total);
        let mut bv = Array1::<f64>::zeros(total);
        let mut buv = Array1::<f64>::zeros(total);
        for i in 0..self.y.len() {
            let q = core.q0[i] + etaw[i];
            let jet = inverse_link_jet_for_inverse_link(&self.link_kind, q).map_err(|e| {
                format!("BinomialLocationScaleWiggle expected d2I link jet failed: {e}")
            })?;
            let (f, f1, f2) = binomial_expected_q_information_derivatives(
                self.weights[i],
                jet.mu,
                jet.d1,
                jet.d2,
                jet.d3,
            );
            if f == 0.0 && f1 == 0.0 && f2 == 0.0 {
                continue;
            }
            let q0 = nonwiggle_q_derivs(eta_t[i], core.sigma[i]);
            let dq0_u = q0.q_t * d_eta_t_u[i] + q0.q_ls * d_eta_ls_u[i];
            let dq0_v = q0.q_t * d_eta_t_v[i] + q0.q_ls * d_eta_ls_v[i];
            let d2q0_uv = q0.q_tl * (d_eta_t_u[i] * d_eta_ls_v[i] + d_eta_t_v[i] * d_eta_ls_u[i])
                + q0.q_ll * d_eta_ls_u[i] * d_eta_ls_v[i];
            let dq0_t_u = q0.q_tl * d_eta_ls_u[i];
            let dq0_t_v = q0.q_tl * d_eta_ls_v[i];
            let dq0_ls_u = q0.q_tl * d_eta_t_u[i] + q0.q_ll * d_eta_ls_u[i];
            let dq0_ls_v = q0.q_tl * d_eta_t_v[i] + q0.q_ll * d_eta_ls_v[i];
            let d2q0_t_uv = q0.q_tl_ls * d_eta_ls_u[i] * d_eta_ls_v[i];
            let d2q0_ls_uv = q0.q_tl_ls
                * (d_eta_ls_u[i] * d_eta_t_v[i] + d_eta_ls_v[i] * d_eta_t_u[i])
                + q0.q_ll_ls * d_eta_ls_u[i] * d_eta_ls_v[i];

            let br = b0.row(i);
            let dr = d0.row(i);
            let ddr = dd0.row(i);
            let b_u = br.dot(&uw);
            let b_v = br.dot(&vw);
            let b1_u = dr.dot(&uw);
            let b1_v = dr.dot(&vw);
            let b2_u = ddr.dot(&uw);
            let b2_v = ddr.dot(&vw);
            let dm_u = g2[i] * dq0_u + b1_u;
            let dm_v = g2[i] * dq0_v + b1_v;
            let d2m_uv = d3q[i] * dq0_u * dq0_v + g2[i] * d2q0_uv + b2_v * dq0_u + b2_u * dq0_v;
            let alpha_u = m[i] * dq0_u + b_u;
            let alpha_v = m[i] * dq0_v + b_v;
            let alpha_uv = m[i] * d2q0_uv + g2[i] * dq0_u * dq0_v + b1_u * dq0_v + b1_v * dq0_u;

            let q_t = m[i] * q0.q_t;
            let q_ls = m[i] * q0.q_ls;
            let dq_t_u = dm_u * q0.q_t + m[i] * dq0_t_u;
            let dq_t_v = dm_v * q0.q_t + m[i] * dq0_t_v;
            let dq_ls_u = dm_u * q0.q_ls + m[i] * dq0_ls_u;
            let dq_ls_v = dm_v * q0.q_ls + m[i] * dq0_ls_v;
            let d2q_t_uv = d2m_uv * q0.q_t + dm_u * dq0_t_v + dm_v * dq0_t_u + m[i] * d2q0_t_uv;
            let d2q_ls_uv =
                d2m_uv * q0.q_ls + dm_u * dq0_ls_v + dm_v * dq0_ls_u + m[i] * d2q0_ls_uv;

            b.fill(0.0);
            bu.fill(0.0);
            bv.fill(0.0);
            buv.fill(0.0);
            b.slice_mut(s![0..pt]).scaled_add(q_t, &x_t.row(i));
            b.slice_mut(s![pt..pt + pls]).scaled_add(q_ls, &x_ls.row(i));
            b.slice_mut(s![pt + pls..]).assign(&br);
            bu.slice_mut(s![0..pt]).scaled_add(dq_t_u, &x_t.row(i));
            bu.slice_mut(s![pt..pt + pls])
                .scaled_add(dq_ls_u, &x_ls.row(i));
            bu.slice_mut(s![pt + pls..]).scaled_add(dq0_u, &dr);
            bv.slice_mut(s![0..pt]).scaled_add(dq_t_v, &x_t.row(i));
            bv.slice_mut(s![pt..pt + pls])
                .scaled_add(dq_ls_v, &x_ls.row(i));
            bv.slice_mut(s![pt + pls..]).scaled_add(dq0_v, &dr);
            buv.slice_mut(s![0..pt]).scaled_add(d2q_t_uv, &x_t.row(i));
            buv.slice_mut(s![pt..pt + pls])
                .scaled_add(d2q_ls_uv, &x_ls.row(i));
            buv.slice_mut(s![pt + pls..])
                .scaled_add(dq0_u * dq0_v, &ddr);
            buv.slice_mut(s![pt + pls..]).scaled_add(d2q0_uv, &dr);

            scaled_outer_add(
                out.view_mut(),
                f2 * alpha_u * alpha_v + f1 * alpha_uv,
                b.view(),
                b.view(),
            );
            scaled_outer_add(out.view_mut(), f1 * alpha_u, bv.view(), b.view());
            scaled_outer_add(out.view_mut(), f1 * alpha_u, b.view(), bv.view());
            scaled_outer_add(out.view_mut(), f1 * alpha_v, bu.view(), b.view());
            scaled_outer_add(out.view_mut(), f1 * alpha_v, b.view(), bu.view());
            scaled_outer_add(out.view_mut(), f, buv.view(), b.view());
            scaled_outer_add(out.view_mut(), f, b.view(), buv.view());
            scaled_outer_add(out.view_mut(), f, bu.view(), bv.view());
            scaled_outer_add(out.view_mut(), f, bv.view(), bu.view());
        }
        mirror_upper_to_lower(&mut out);
        Ok(Some(out))
    }
}

/// Per-row pieces of the 3-block wiggle joint Hessian.
///
/// `coeff_*` are diagonal weights (length n). `b0` and `d0` are the realized
/// wiggle basis values and first-derivative values at the current q0
/// (n × p_w). The dense Hessian path assembles these into a (p_t+p_ls+p_w)²
/// matrix; the matrix-free workspace applies the operator
///
///   r_t = D_tt u_t + D_tl u_ls + D_tw_b (B v_w) + D_tw_d (B' v_w),
///   r_ls = D_tl u_t + D_ll u_ls + D_lw_b (B v_w) + D_lw_d (B' v_w),
///   r_b = D_tw_b u_t + D_lw_b u_ls + D_ww (B v_w),
///   r_d = D_tw_d u_t + D_lw_d u_ls,
///
/// and combines `out_w = B^T r_b + (B')^T r_d` to form `H v` directly.
pub(crate) struct BinomialWiggleOrder2Rows {
    pub(crate) coeff_tt: Array1<f64>,
    pub(crate) coeff_tl: Array1<f64>,
    pub(crate) coeff_ll: Array1<f64>,
    pub(crate) coeff_tw_b: Array1<f64>,
    pub(crate) coeff_tw_d: Array1<f64>,
    pub(crate) coeff_lw_b: Array1<f64>,
    pub(crate) coeff_lw_d: Array1<f64>,
    pub(crate) coeff_ww: Array1<f64>,
    pub(crate) b0: Array2<f64>,
    pub(crate) d0: Array2<f64>,
}

pub(crate) struct ExpectedWiggleGeometryInputs<'a> {
    pub(crate) x_t: Cow<'a, Array2<f64>>,
    pub(crate) x_ls: Cow<'a, Array2<f64>>,
    pub(crate) eta_t: &'a Array1<f64>,
    pub(crate) eta_ls: &'a Array1<f64>,
    pub(crate) etaw: &'a Array1<f64>,
}

impl BinomialWiggleOrder2Rows {
    fn zeros(n: usize, b0: Array2<f64>, d0: Array2<f64>) -> Self {
        Self {
            coeff_tt: Array1::zeros(n),
            coeff_tl: Array1::zeros(n),
            coeff_ll: Array1::zeros(n),
            coeff_tw_b: Array1::zeros(n),
            coeff_tw_d: Array1::zeros(n),
            coeff_lw_b: Array1::zeros(n),
            coeff_lw_d: Array1::zeros(n),
            coeff_ww: Array1::zeros(n),
            b0,
            d0,
        }
    }

    pub(crate) fn assemble_dense(
        &self,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Array2<f64>, String> {
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let pw = self.b0.ncols();
        let total = pt + pls + pw;
        let h_tt = xt_diag_x_dense(x_t, &self.coeff_tt)?;
        let h_tl = xt_diag_y_dense(x_t, &self.coeff_tl, x_ls)?;
        let h_ll = xt_diag_x_dense(x_ls, &self.coeff_ll)?;
        let h_tw = xt_diag_y_dense(x_t, &self.coeff_tw_b, &self.b0)?
            + &xt_diag_y_dense(x_t, &self.coeff_tw_d, &self.d0)?;
        let h_lw = xt_diag_y_dense(x_ls, &self.coeff_lw_b, &self.b0)?
            + &xt_diag_y_dense(x_ls, &self.coeff_lw_d, &self.d0)?;
        let hww = xt_diag_x_dense(&self.b0, &self.coeff_ww)?;

        let mut h = Array2::<f64>::zeros((total, total));
        h.slice_mut(s![0..pt, 0..pt]).assign(&h_tt);
        h.slice_mut(s![0..pt, pt..pt + pls]).assign(&h_tl);
        h.slice_mut(s![pt..pt + pls, pt..pt + pls]).assign(&h_ll);
        h.slice_mut(s![0..pt, pt + pls..total]).assign(&h_tw);
        h.slice_mut(s![pt..pt + pls, pt + pls..total]).assign(&h_lw);
        h.slice_mut(s![pt + pls..total, pt + pls..total])
            .assign(&hww);
        mirror_upper_to_lower(&mut h);
        Ok(h)
    }

    /// Block-diagonal Hessians (h_tt, h_ll, h_ww) without ever materializing
    /// the cross blocks. Used by `evaluate()` to populate per-block working
    /// sets.
    pub(crate) fn assemble_block_diagonals(
        &self,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>), String> {
        let h_tt = xt_diag_x_dense(x_t, &self.coeff_tt)?;
        let h_ll = xt_diag_x_dense(x_ls, &self.coeff_ll)?;
        let h_ww = xt_diag_x_dense(&self.b0, &self.coeff_ww)?;
        Ok((h_tt, h_ll, h_ww))
    }
}

/// Per-row coefficient arrays for the BLS Wiggle joint first-directional
/// Hessian derivative `D_β H_L[u]`, shared by the dense `_directional_derivative`
/// assembly and the matrix-free `bls_wiggle_directional_operator`.
pub(crate) struct BinomialWiggleDhRowCoeffs {
    pub(crate) coeff_tt: Array1<f64>,
    pub(crate) coeff_tl: Array1<f64>,
    pub(crate) coeff_ll: Array1<f64>,
    pub(crate) coeff_tw_b: Array1<f64>,
    pub(crate) coeff_tw_d: Array1<f64>,
    pub(crate) coeff_tw_dd: Array1<f64>,
    pub(crate) coeff_lw_b: Array1<f64>,
    pub(crate) coeff_lw_d: Array1<f64>,
    pub(crate) coeff_lw_dd: Array1<f64>,
    pub(crate) coeffww_bb: Array1<f64>,
    pub(crate) coeffww_db: Array1<f64>,
}

/// All references needed to evaluate [`BinomialWiggleDhRowCoeffs`].
pub(crate) struct BinomialWiggleDhRowInputs<'a> {
    pub(crate) core0: &'a BinomialLocationScaleCore,
    pub(crate) eta_t: &'a Array1<f64>,
    pub(crate) etaw: &'a Array1<f64>,
    pub(crate) sigma: &'a Array1<f64>,
    pub(crate) m: &'a Array1<f64>,
    pub(crate) g2: &'a Array1<f64>,
    pub(crate) g3: &'a Array1<f64>,
    pub(crate) b0: &'a Array2<f64>,
    pub(crate) d0: &'a Array2<f64>,
    pub(crate) dd0: &'a Array2<f64>,
    pub(crate) uw: &'a Array1<f64>,
    pub(crate) d_eta_t: &'a Array1<f64>,
    pub(crate) d_eta_ls: &'a Array1<f64>,
}

impl BinomialLocationScaleWiggleFamily {
    /// Per-row coefficient loop for the joint first-directional Hessian
    /// derivative. The dense and operator paths build the identical 11
    /// coefficient arrays from the same canonical directional-q formulas.
    pub(crate) fn binomial_wiggle_dh_row_coeffs(
        &self,
        n: usize,
        inputs: &BinomialWiggleDhRowInputs<'_>,
    ) -> BinomialWiggleDhRowCoeffs {
        let BinomialWiggleDhRowInputs {
            core0,
            eta_t,
            etaw,
            sigma,
            m,
            g2,
            g3,
            b0,
            d0,
            dd0,
            uw,
            d_eta_t,
            d_eta_ls,
        } = *inputs;

        let mut coeff_tt = Array1::<f64>::zeros(n);
        let mut coeff_tl = Array1::<f64>::zeros(n);
        let mut coeff_ll = Array1::<f64>::zeros(n);
        let mut coeff_tw_b = Array1::<f64>::zeros(n);
        let mut coeff_tw_d = Array1::<f64>::zeros(n);
        let mut coeff_tw_dd = Array1::<f64>::zeros(n);
        let mut coeff_lw_b = Array1::<f64>::zeros(n);
        let mut coeff_lw_d = Array1::<f64>::zeros(n);
        let mut coeff_lw_dd = Array1::<f64>::zeros(n);
        let mut coeffww_bb = Array1::<f64>::zeros(n);
        let mut coeffww_db = Array1::<f64>::zeros(n);
        for i in 0..n {
            let q_i = core0.q0[i] + etaw[i];
            let (m1, m2, m3) = binomial_neglog_q_derivatives_dispatch(
                self.y[i],
                self.weights[i],
                q_i,
                core0.mu[i],
                core0.dmu_dq[i],
                core0.d2mu_dq2[i],
                core0.d3mu_dq3[i],
                &self.link_kind,
            );
            let q0 = nonwiggle_q_derivs(eta_t[i], sigma[i]);
            let dq0 = nonwiggle_q_directional(q0, d_eta_t[i], d_eta_ls[i]);

            let br = b0.row(i);
            let dr = d0.row(i);
            let ddr = dd0.row(i);
            let duw_i = dr.dot(uw);
            let dduw_i = ddr.dot(uw);

            let delta_m = g2[i] * dq0.delta_q + duw_i;
            let delta_g2 = g3[i] * dq0.delta_q + dduw_i;

            let q_t = m[i] * q0.q_t;
            let q_ls = m[i] * q0.q_ls;
            let q_tt = g2[i] * q0.q_t * q0.q_t;
            let q_tl = g2[i] * q0.q_t * q0.q_ls + m[i] * q0.q_tl;
            let q_ll = g2[i] * q0.q_ls * q0.q_ls + m[i] * q0.q_ll;

            let delta_q_t = delta_m * q0.q_t + m[i] * dq0.delta_q_t;
            let delta_q_ls = delta_m * q0.q_ls + m[i] * dq0.delta_q_ls;
            let delta_q_tt = delta_g2 * q0.q_t * q0.q_t + g2[i] * 2.0 * q0.q_t * dq0.delta_q_t;
            let delta_q_tl = delta_g2 * q0.q_t * q0.q_ls
                + g2[i] * (dq0.delta_q_t * q0.q_ls + q0.q_t * dq0.delta_q_ls)
                + delta_m * q0.q_tl
                + m[i] * dq0.delta_q_tl;
            let delta_q_ll = delta_g2 * q0.q_ls * q0.q_ls
                + g2[i] * 2.0 * q0.q_ls * dq0.delta_q_ls
                + delta_m * q0.q_ll
                + m[i] * dq0.delta_q_ll;

            let delta_q = m[i] * dq0.delta_q + br.dot(uw);

            coeff_tt[i] = directionalhessian_coeff_fromobjective_q_terms(
                m1, m2, m3, delta_q, q_t, q_t, q_tt, delta_q_t, delta_q_t, delta_q_tt,
            );
            coeff_tl[i] = directionalhessian_coeff_fromobjective_q_terms(
                m1, m2, m3, delta_q, q_t, q_ls, q_tl, delta_q_t, delta_q_ls, delta_q_tl,
            );
            coeff_ll[i] = directionalhessian_coeff_fromobjective_q_terms(
                m1, m2, m3, delta_q, q_ls, q_ls, q_ll, delta_q_ls, delta_q_ls, delta_q_ll,
            );
            coeff_tw_b[i] = m3 * delta_q * q_t + m2 * delta_q_t;
            coeff_tw_d[i] = m2 * (q_t * dq0.delta_q + delta_q * q0.q_t) + m1 * dq0.delta_q_t;
            coeff_tw_dd[i] = m1 * dq0.delta_q * q0.q_t;
            coeff_lw_b[i] = m3 * delta_q * q_ls + m2 * delta_q_ls;
            coeff_lw_d[i] = m2 * (q_ls * dq0.delta_q + delta_q * q0.q_ls) + m1 * dq0.delta_q_ls;
            coeff_lw_dd[i] = m1 * dq0.delta_q * q0.q_ls;
            coeffww_bb[i] = m3 * delta_q;
            coeffww_db[i] = m2 * dq0.delta_q;
        }

        BinomialWiggleDhRowCoeffs {
            coeff_tt,
            coeff_tl,
            coeff_ll,
            coeff_tw_b,
            coeff_tw_d,
            coeff_tw_dd,
            coeff_lw_b,
            coeff_lw_d,
            coeff_lw_dd,
            coeffww_bb,
            coeffww_db,
        }
    }

    /// Build the [`BlockEffectiveJacobian`] for block `block_idx`.
    ///
    /// The two-output map is (η_threshold, η_log_sigma).
    /// The wiggle block operates on the combined linear predictor through the
    /// nonlinear inverse link and has a zero effective linear Jacobian.
    ///
    /// - block 0 (threshold):  output 0 = design rows, output 1 = zeros
    /// - block 1 (log_sigma):  output 0 = zeros, output 1 = design rows
    /// - block 2 (wiggle):     all zeros (nonlinear link modulation)
    pub fn block_effective_jacobian(
        specs: &[ParameterBlockSpec],
        block_idx: usize,
    ) -> Result<Box<dyn BlockEffectiveJacobian>, String> {
        crate::block_layout::block_jacobian::AdditiveWiggleBlockLayout {
            family: "BinomialLocationScaleWiggleFamily",
            n_outputs: 2,
            additive_blocks: &[Self::BLOCK_T, Self::BLOCK_LOG_SIGMA],
            wiggle_block: Some(Self::BLOCK_WIGGLE),
        }
        .block_effective_jacobian(specs, block_idx)
    }
}

impl BinomialLocationScaleWiggleFamily {
    /// Build a matrix-free `RowCoeffOperator` for the BLS Wiggle joint
    /// directional derivative `D_β H_L[u]`. Channels (in order):
    /// X_t, X_ls, B (b0), B' (d0), B'' (dd0). The operator acts on the
    /// joint coefficient vector `(β_t, β_ls, β_w)`.
    pub(crate) fn bls_wiggle_directional_operator(
        &self,
        block_states: &[ParameterBlockState],
        x_t_arc: Arc<Array2<f64>>,
        x_ls_arc: Arc<Array2<f64>>,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn gam_problem::HyperOperator>>, String> {
        let (n, eta_t, eta_ls, etaw) = self.validated_block_etas(block_states)?;
        let pt = x_t_arc.ncols();
        let pls = x_ls_arc.ncols();
        let betaw0 = block_states[Self::BLOCK_WIGGLE].beta.clone();
        let core0 = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(etaw),
            &self.link_kind,
        )?;
        let b0 = self.wiggle_design(core0.q0.view())?;
        let pw = b0.ncols();
        let beta_layout = GamlssBetaLayout::withwiggle(pt, pls, pw);
        let total = beta_layout.total();
        if d_beta_flat.len() != total {
            return Err(GamlssError::InvalidInput {
                reason: format!(
                    "BLS wiggle dH operator: d_beta length {} != {}",
                    d_beta_flat.len(),
                    total
                ),
            }
            .into());
        }
        let (u_t, u_ls, uw) =
            beta_layout.split_three(d_beta_flat, "wiggle joint dH operator d_beta")?;
        let d_eta_t = fast_av(x_t_arc.as_ref(), &u_t);
        let d_eta_ls = fast_av(x_ls_arc.as_ref(), &u_ls);

        let d0 =
            self.wiggle_basiswith_options(core0.q0.view(), BasisOptions::first_derivative())?;
        let dd0 =
            self.wiggle_basiswith_options(core0.q0.view(), BasisOptions::second_derivative())?;
        let d3q = self.wiggle_d3q_dq03(core0.q0.view(), betaw0.view())?;
        if d0.ncols() != betaw0.len() || dd0.ncols() != betaw0.len() {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "wiggle derivative/beta mismatch in dH operator: B'={} B''={} betaw={}",
                    d0.ncols(),
                    dd0.ncols(),
                    betaw0.len()
                ),
            }
            .into());
        }
        let m = d0.dot(&betaw0) + 1.0;
        let g2 = dd0.dot(&betaw0);
        let g3 = d3q;
        let (sigma, ..) = exp_sigma_derivs_up_to_third(eta_ls.view());

        let BinomialWiggleDhRowCoeffs {
            coeff_tt,
            coeff_tl,
            coeff_ll,
            coeff_tw_b,
            coeff_tw_d,
            coeff_tw_dd,
            coeff_lw_b,
            coeff_lw_d,
            coeff_lw_dd,
            coeffww_bb,
            coeffww_db,
        } = self.binomial_wiggle_dh_row_coeffs(
            n,
            &BinomialWiggleDhRowInputs {
                core0: &core0,
                eta_t,
                etaw,
                sigma: &sigma,
                m: &m,
                g2: &g2,
                g3: &g3,
                b0: &b0,
                d0: &d0,
                dd0: &dd0,
                uw: &uw,
                d_eta_t: &d_eta_t,
                d_eta_ls: &d_eta_ls,
            },
        );

        let basis: Arc<Array2<f64>> = Arc::new(b0);
        let basis_d1: Arc<Array2<f64>> = Arc::new(d0);
        let basis_d2: Arc<Array2<f64>> = Arc::new(dd0);

        Ok(Some(Arc::new(RowCoeffOperator::from_directions(
            vec![pt, pls, pw],
            vec![
                (0, x_t_arc),
                (1, x_ls_arc),
                (2, basis),
                (2, basis_d1),
                (2, basis_d2),
            ],
            vec![
                // (X_t, X_t)  ← `xt_diag_x_dense(&x_t, &coeff_tt)`
                (0, 0, coeff_tt),
                // (X_t, X_ls) ← `xt_diag_y_dense(&x_t, &coeff_tl, &x_ls)`
                (0, 1, coeff_tl),
                // (X_ls, X_ls) ← `xt_diag_x_dense(&x_ls, &coeff_ll)`
                (1, 1, coeff_ll),
                // (X_t, B / B' / B'') ← three sub-blocks of d_h_tw =
                // `xt_diag_y_dense(x_t, coeff_tw_b, b0) + xt_diag_y_dense(
                //  x_t, coeff_tw_d, d0) + xt_diag_y_dense(x_t, coeff_tw_dd, dd0)`
                (0, 2, coeff_tw_b),
                (0, 3, coeff_tw_d),
                (0, 4, coeff_tw_dd),
                // (X_ls, B / B' / B'') ← analogous d_h_lw triple
                (1, 2, coeff_lw_b),
                (1, 3, coeff_lw_d),
                (1, 4, coeff_lw_dd),
                // (B, B) ← `xt_diag_x_dense(&b0, &coeffww_bb)`
                (2, 2, coeffww_bb),
                // (B, B') ← `xt_diag_y_dense(&d0, &coeffww_db, &b0) +
                // xt_diag_y_dense(&b0, &coeffww_db, &d0)` =
                // d0^T diag(c) b0 + b0^T diag(c) d0 (symmetric pair)
                (2, 3, coeffww_db),
            ],
            n,
        ))))
    }

    /// Build a matrix-free `RowCoeffOperator` for the BLS Wiggle joint
    /// second directional derivative `D²_β H_L[u, v]`. Channels: X_t,
    /// X_ls, B, B', B'', B'''.
    ///
    /// The dense path computes a per-row scalar `coeff_*(i, j[, k])` via
    /// `second_directionalhessian_coeff_fromobjective_q_terms` and outer-
    /// products it into the (t,t) / (t,ls) / (ls,ls) / (t,w) / (ls,w) /
    /// (w,w) blocks. Each `coeff_tw(i, j)` is *linear* in the basis
    /// derivatives at column j (`br[j], dr[j], ddr[j], d3r[j]` — they
    /// only ever appear once in the q-Hessian directional polynomial),
    /// so each per-(i,j) contribution decomposes into 4 channel-pair
    /// row coefficients (X_t, B/B'/B''/B'''). The wiggle-wiggle term
    /// `coeff_ww(i, j, k)` is *bilinear* in (br[j], dr[j], ddr[j]) ⊗
    /// (br[k], dr[k], ddr[k]), giving 4 symmetric pair coefficients on
    /// (B, B), (B, B'), (B, B''), (B', B'). No (B'', B'') term — the
    /// formula is at most degree 2 in any single basis derivative.
    pub(crate) fn bls_wiggle_second_directional_operator(
        &self,
        block_states: &[ParameterBlockState],
        x_t_arc: Arc<Array2<f64>>,
        x_ls_arc: Arc<Array2<f64>>,
        d_beta_u: &Array1<f64>,
        d_beta_v: &Array1<f64>,
    ) -> Result<Option<Arc<dyn gam_problem::HyperOperator>>, String> {
        let (n, eta_t, eta_ls, etaw) = self.validated_block_etas(block_states)?;
        let pt = x_t_arc.ncols();
        let pls = x_ls_arc.ncols();
        let betaw0 = block_states[Self::BLOCK_WIGGLE].beta.clone();
        let core0 = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(etaw),
            &self.link_kind,
        )?;
        let b0 = self.wiggle_design(core0.q0.view())?;
        let d0 =
            self.wiggle_basiswith_options(core0.q0.view(), BasisOptions::first_derivative())?;
        let dd0 =
            self.wiggle_basiswith_options(core0.q0.view(), BasisOptions::second_derivative())?;
        let d3_basis = self.wiggle_d3basis_constrained(core0.q0.view())?;
        let d3q = self.wiggle_d3q_dq03(core0.q0.view(), betaw0.view())?;
        let d4q = self.wiggle_d4q_dq04(core0.q0.view(), betaw0.view())?;
        let pw = b0.ncols();
        let beta_layout = GamlssBetaLayout::withwiggle(pt, pls, pw);
        let total = beta_layout.total();
        if d_beta_u.len() != total || d_beta_v.len() != total {
            return Err(GamlssError::InvalidInput {
                reason: format!(
                    "BLS wiggle d2H operator: d_beta_{{u,v}} length {}/{} != {}",
                    d_beta_u.len(),
                    d_beta_v.len(),
                    total
                ),
            }
            .into());
        }
        if d0.ncols() != betaw0.len()
            || dd0.ncols() != betaw0.len()
            || d3_basis.ncols() != betaw0.len()
        {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "wiggle derivative/beta mismatch in d2H operator: B'={} B''={} B'''={} betaw={}",
                d0.ncols(),
                dd0.ncols(),
                d3_basis.ncols(),
                betaw0.len()
            ) }.into());
        }

        let (u_t, u_ls, uw) = beta_layout.split_three(d_beta_u, "wiggle d2H op u")?;
        let (v_t, v_ls, vw) = beta_layout.split_three(d_beta_v, "wiggle d2H op v")?;
        let d_eta_t_u = fast_av(x_t_arc.as_ref(), &u_t);
        let d_eta_ls_u = fast_av(x_ls_arc.as_ref(), &u_ls);
        let d_eta_t_v = fast_av(x_t_arc.as_ref(), &v_t);
        let d_eta_ls_v = fast_av(x_ls_arc.as_ref(), &v_ls);

        let m = d0.dot(&betaw0) + 1.0;
        let g2 = dd0.dot(&betaw0);
        let g3 = d3q;
        let g4 = d4q;
        let (sigma, ds, d2s, d3s, d4s) = exp_sigma_derivs_up_to_fourth_array(eta_ls.view());

        // Per-row scalar pair coefficients.
        let mut coeff_tt = Array1::<f64>::zeros(n);
        let mut coeff_tl = Array1::<f64>::zeros(n);
        let mut coeff_ll = Array1::<f64>::zeros(n);
        // Per-row coefficients for the t↔wiggle decomposition into
        // (X_t, B), (X_t, B'), (X_t, B''), (X_t, B''') pair entries.
        let mut alpha_tw_b = Array1::<f64>::zeros(n);
        let mut alpha_tw_d = Array1::<f64>::zeros(n);
        let mut alpha_tw_dd = Array1::<f64>::zeros(n);
        let mut alpha_tw_d3 = Array1::<f64>::zeros(n);
        let mut alpha_lw_b = Array1::<f64>::zeros(n);
        let mut alpha_lw_d = Array1::<f64>::zeros(n);
        let mut alpha_lw_dd = Array1::<f64>::zeros(n);
        let mut alpha_lw_d3 = Array1::<f64>::zeros(n);
        // Wiggle-wiggle bilinear pair entries on (B,B), (B,B'), (B,B''), (B',B').
        let mut c_ww_bb = Array1::<f64>::zeros(n);
        let mut c_ww_bd = Array1::<f64>::zeros(n);
        let mut c_ww_bdd = Array1::<f64>::zeros(n);
        let mut c_ww_dd_pair = Array1::<f64>::zeros(n);

        for i in 0..n {
            let q_i = core0.q0[i] + etaw[i];
            let (m1, m2, m3) = binomial_neglog_q_derivatives_dispatch(
                self.y[i],
                self.weights[i],
                q_i,
                core0.mu[i],
                core0.dmu_dq[i],
                core0.d2mu_dq2[i],
                core0.d3mu_dq3[i],
                &self.link_kind,
            );
            let m4 = binomial_neglog_q_fourth_derivative_dispatch(
                self.y[i],
                self.weights[i],
                q_i,
                core0.mu[i],
                core0.dmu_dq[i],
                core0.d2mu_dq2[i],
                core0.d3mu_dq3[i],
                &self.link_kind,
            )?;

            let q0_d = nonwiggle_q_derivs(eta_t[i], sigma[i]);
            let s_safe = sigma[i];
            let s2 = s_safe * s_safe;
            let s3 = s2 * s_safe;
            let s4 = s3 * s_safe;
            let s5 = s4 * s_safe;
            let q0_tl_ls_ls =
                d3s[i] / s2 - 6.0 * ds[i] * d2s[i] / s3 + 6.0 * ds[i] * ds[i] * ds[i] / s4;
            let q0_tl_ls_ls_ls =
                d4s[i] / s2 - 8.0 * ds[i] * d3s[i] / s3 - 6.0 * d2s[i] * d2s[i] / s3
                    + 36.0 * ds[i] * ds[i] * d2s[i] / s4
                    - 24.0 * ds[i] * ds[i] * ds[i] * ds[i] / s5;
            let q0_ll_ls_ls = eta_t[i] * q0_tl_ls_ls_ls;

            let u_t_i = d_eta_t_u[i];
            let u_ls_i = d_eta_ls_u[i];
            let v_t_i = d_eta_t_v[i];
            let v_ls_i = d_eta_ls_v[i];

            let dq0_u = q0_d.q_t * u_t_i + q0_d.q_ls * u_ls_i;
            let dq0v = q0_d.q_t * v_t_i + q0_d.q_ls * v_ls_i;
            let d2q0_uv =
                q0_d.q_tl * (u_t_i * v_ls_i + v_t_i * u_ls_i) + q0_d.q_ll * u_ls_i * v_ls_i;

            let dq0_t_u = q0_d.q_tl * u_ls_i;
            let dq0_tv = q0_d.q_tl * v_ls_i;
            let dq0_ls_u = q0_d.q_tl * u_t_i + q0_d.q_ll * u_ls_i;
            let dq0_lsv = q0_d.q_tl * v_t_i + q0_d.q_ll * v_ls_i;
            let dq0_tl_u = q0_d.q_tl_ls * u_ls_i;
            let dq0_tlv = q0_d.q_tl_ls * v_ls_i;
            let dq0_ll_u = q0_d.q_tl_ls * u_t_i + q0_d.q_ll_ls * u_ls_i;
            let dq0_llv = q0_d.q_tl_ls * v_t_i + q0_d.q_ll_ls * v_ls_i;

            let d2q0_t_uv = q0_d.q_tl_ls * u_ls_i * v_ls_i;
            let d2q0_ls_uv =
                q0_d.q_tl_ls * (u_ls_i * v_t_i + v_ls_i * u_t_i) + q0_d.q_ll_ls * u_ls_i * v_ls_i;
            let d2q0_tl_uv = q0_tl_ls_ls * u_ls_i * v_ls_i;
            let d2q0_ll_uv =
                q0_tl_ls_ls * (u_t_i * v_ls_i + v_t_i * u_ls_i) + q0_ll_ls_ls * u_ls_i * v_ls_i;

            let br = b0.row(i);
            let dr = d0.row(i);
            let ddr = dd0.row(i);
            let d3r = d3_basis.row(i);
            let b_u = br.dot(&uw);
            let bv = br.dot(&vw);
            let b1_u = dr.dot(&uw);
            let b1v = dr.dot(&vw);
            let b2_u = ddr.dot(&uw);
            let b2v = ddr.dot(&vw);
            let b3_u = d3r.dot(&uw);
            let b3v = d3r.dot(&vw);

            let dm_u = b1_u + g2[i] * dq0_u;
            let dmv = b1v + g2[i] * dq0v;
            let d2m_uv = g3[i] * dq0_u * dq0v + g2[i] * d2q0_uv + b2v * dq0_u + b2_u * dq0v;
            let dg2_u = b2_u + g3[i] * dq0_u;
            let dg2v = b2v + g3[i] * dq0v;
            let d2g2_uv = g4[i] * dq0_u * dq0v + g3[i] * d2q0_uv + b3v * dq0_u + b3_u * dq0v;

            let dq_u = m[i] * dq0_u + b_u;
            let dqv = m[i] * dq0v + bv;
            let d2q_uv = m[i] * d2q0_uv + g2[i] * dq0_u * dq0v + b1_u * dq0v + b1v * dq0_u;

            let q_t = m[i] * q0_d.q_t;
            let q_ls = m[i] * q0_d.q_ls;
            let q_tt = g2[i] * q0_d.q_t * q0_d.q_t;
            let q_tl = g2[i] * q0_d.q_t * q0_d.q_ls + m[i] * q0_d.q_tl;
            let q_ll = g2[i] * q0_d.q_ls * q0_d.q_ls + m[i] * q0_d.q_ll;

            let dq_t_u = dm_u * q0_d.q_t + m[i] * dq0_t_u;
            let dq_tv = dmv * q0_d.q_t + m[i] * dq0_tv;
            let dq_ls_u = dm_u * q0_d.q_ls + m[i] * dq0_ls_u;
            let dq_lsv = dmv * q0_d.q_ls + m[i] * dq0_lsv;

            let d2q_t_uv = d2m_uv * q0_d.q_t + dm_u * dq0_tv + dmv * dq0_t_u + m[i] * d2q0_t_uv;
            let d2q_ls_uv =
                d2m_uv * q0_d.q_ls + dm_u * dq0_lsv + dmv * dq0_ls_u + m[i] * d2q0_ls_uv;

            let dq_tt_u = dg2_u * q0_d.q_t * q0_d.q_t + g2[i] * (2.0 * q0_d.q_t * dq0_t_u);
            let dq_ttv = dg2v * q0_d.q_t * q0_d.q_t + g2[i] * (2.0 * q0_d.q_t * dq0_tv);
            let d2q_tt_uv = d2g2_uv * q0_d.q_t * q0_d.q_t
                + dg2_u * (2.0 * q0_d.q_t * dq0_tv)
                + dg2v * (2.0 * q0_d.q_t * dq0_t_u)
                + g2[i] * (2.0 * dq0_t_u * dq0_tv + 2.0 * q0_d.q_t * d2q0_t_uv);

            let dq_tl_u = dg2_u * q0_d.q_t * q0_d.q_ls
                + g2[i] * (dq0_t_u * q0_d.q_ls + q0_d.q_t * dq0_ls_u)
                + dm_u * q0_d.q_tl
                + m[i] * dq0_tl_u;
            let dq_tlv = dg2v * q0_d.q_t * q0_d.q_ls
                + g2[i] * (dq0_tv * q0_d.q_ls + q0_d.q_t * dq0_lsv)
                + dmv * q0_d.q_tl
                + m[i] * dq0_tlv;
            let d2q_tl_uv = d2g2_uv * q0_d.q_t * q0_d.q_ls
                + dg2_u * (dq0_tv * q0_d.q_ls + q0_d.q_t * dq0_lsv)
                + dg2v * (dq0_t_u * q0_d.q_ls + q0_d.q_t * dq0_ls_u)
                + g2[i]
                    * (d2q0_t_uv * q0_d.q_ls
                        + dq0_t_u * dq0_lsv
                        + dq0_tv * dq0_ls_u
                        + q0_d.q_t * d2q0_ls_uv)
                + d2m_uv * q0_d.q_tl
                + dm_u * dq0_tlv
                + dmv * dq0_tl_u
                + m[i] * d2q0_tl_uv;

            let dq_ll_u = dg2_u * q0_d.q_ls * q0_d.q_ls
                + g2[i] * (2.0 * q0_d.q_ls * dq0_ls_u)
                + dm_u * q0_d.q_ll
                + m[i] * dq0_ll_u;
            let dq_llv = dg2v * q0_d.q_ls * q0_d.q_ls
                + g2[i] * (2.0 * q0_d.q_ls * dq0_lsv)
                + dmv * q0_d.q_ll
                + m[i] * dq0_llv;
            let d2q_ll_uv = d2g2_uv * q0_d.q_ls * q0_d.q_ls
                + dg2_u * (2.0 * q0_d.q_ls * dq0_lsv)
                + dg2v * (2.0 * q0_d.q_ls * dq0_ls_u)
                + g2[i] * (2.0 * dq0_ls_u * dq0_lsv + 2.0 * q0_d.q_ls * d2q0_ls_uv)
                + d2m_uv * q0_d.q_ll
                + dm_u * dq0_llv
                + dmv * dq0_ll_u
                + m[i] * d2q0_ll_uv;

            // Scalar pair coefficients on (X_t, X_t), (X_t, X_ls), (X_ls, X_ls).
            coeff_tt[i] = second_directionalhessian_coeff_fromobjective_q_terms(
                m1, m2, m3, m4, dq_u, dqv, d2q_uv, q_t, q_t, q_tt, dq_t_u, dq_tv, dq_t_u, dq_tv,
                d2q_t_uv, d2q_t_uv, dq_tt_u, dq_ttv, d2q_tt_uv,
            );
            coeff_tl[i] = second_directionalhessian_coeff_fromobjective_q_terms(
                m1, m2, m3, m4, dq_u, dqv, d2q_uv, q_t, q_ls, q_tl, dq_t_u, dq_tv, dq_ls_u, dq_lsv,
                d2q_t_uv, d2q_ls_uv, dq_tl_u, dq_tlv, d2q_tl_uv,
            );
            coeff_ll[i] = second_directionalhessian_coeff_fromobjective_q_terms(
                m1, m2, m3, m4, dq_u, dqv, d2q_uv, q_ls, q_ls, q_ll, dq_ls_u, dq_lsv, dq_ls_u,
                dq_lsv, d2q_ls_uv, d2q_ls_uv, dq_ll_u, dq_llv, d2q_ll_uv,
            );

            // Cross block (X_a, B/B'/B''/B''') with X_a ∈ {X_t, X_ls}. Each
            // `coeff_xw(i, j)` is linear in (br[j], dr[j], ddr[j], d3r[j])
            // because each q-Hessian variable carrying `j` (q_xw, dq_xw_u,
            // dq_xwv, d2q_xw_uv, qw, dqw_u, dqwv, d2qw_uv) is linear in those
            // four. We expand `second_directionalhessian_coeff_fromobjective_q_terms`
            // by collecting like basis-derivative powers; the coefficients are
            // the four α_xw_{b,d,dd,d3} arrays.
            //
            // qw=br, dqw_u=dr·dq0u, dqwv=dr·dq0v, d2qw_uv=ddr·dq0u·dq0v + dr·d2q0_uv
            // q_xw=dr·q0_x, dq_xw_u=ddr·dq0u·q0_x + dr·dq0_x_u, dq_xwv=ddr·dq0v·q0_x + dr·dq0_xv
            // d2q_xw_uv=d3r·dq0u·dq0v·q0_x + ddr·(d2q0_uv·q0_x + dq0u·dq0_xv + dq0v·dq0_x_u) + dr·d2q0_x_uv
            //
            // d_qaqb_u = dq_x_u·qw + q_x·dqw_u  →  dq_x_u·br + q_x·dr·dq0u
            // d_qaqbv  = dq_xv·qw + q_x·dqwv    →  dq_xv·br + q_x·dr·dq0v
            // d2_qaqb_uv = d2q_x_uv·br + dq_x_u·dr·dq0v + dq_xv·dr·dq0u + q_x·d2qw_uv
            //
            // The full formula (expanded for "tw"; "lw" identical with x→ls):
            //
            //   m4·dq_u·dqv·q_x·br
            // + m3·(d2q_uv·q_x·br + dq_u·(dq_xv·br + q_x·dr·dq0v) + dqv·(dq_x_u·br + q_x·dr·dq0u) + dq_u·dqv·dr·q0_x)
            // + m2·(d2q_x_uv·br + dq_x_u·dr·dq0v + dq_xv·dr·dq0u + q_x·(ddr·dq0u·dq0v + dr·d2q0_uv)
            //       + d2q_uv·dr·q0_x + dq_u·(ddr·dq0v·q0_x + dr·dq0_xv) + dqv·(ddr·dq0u·q0_x + dr·dq0_x_u))
            // + m1·(d3r·dq0u·dq0v·q0_x + ddr·(d2q0_uv·q0_x + dq0u·dq0_xv + dq0v·dq0_x_u) + dr·d2q0_x_uv)
            //
            // Collecting like basis-derivative terms produces the closed-form
            // expressions below.

            // X_t ↔ wiggle channels.
            alpha_tw_b[i] = m4 * dq_u * dqv * q_t
                + m3 * (d2q_uv * q_t + dq_u * dq_tv + dqv * dq_t_u)
                + m2 * d2q_t_uv;
            alpha_tw_d[i] = m3 * (dq_u * q_t * dq0v + dqv * q_t * dq0_u + dq_u * dqv * q0_d.q_t)
                + m2 * (dq_t_u * dq0v
                    + dq_tv * dq0_u
                    + q_t * d2q0_uv
                    + d2q_uv * q0_d.q_t
                    + dq_u * dq0_tv
                    + dqv * dq0_t_u)
                + m1 * d2q0_t_uv;
            alpha_tw_dd[i] = m2
                * (q_t * dq0_u * dq0v + dq_u * dq0v * q0_d.q_t + dqv * dq0_u * q0_d.q_t)
                + m1 * (d2q0_uv * q0_d.q_t + dq0_u * dq0_tv + dq0v * dq0_t_u);
            alpha_tw_d3[i] = m1 * dq0_u * dq0v * q0_d.q_t;

            // X_ls ↔ wiggle channels (same formulas, swap t→ls).
            alpha_lw_b[i] = m4 * dq_u * dqv * q_ls
                + m3 * (d2q_uv * q_ls + dq_u * dq_lsv + dqv * dq_ls_u)
                + m2 * d2q_ls_uv;
            alpha_lw_d[i] = m3 * (dq_u * q_ls * dq0v + dqv * q_ls * dq0_u + dq_u * dqv * q0_d.q_ls)
                + m2 * (dq_ls_u * dq0v
                    + dq_lsv * dq0_u
                    + q_ls * d2q0_uv
                    + d2q_uv * q0_d.q_ls
                    + dq_u * dq0_lsv
                    + dqv * dq0_ls_u)
                + m1 * d2q0_ls_uv;
            alpha_lw_dd[i] = m2
                * (q_ls * dq0_u * dq0v + dq_u * dq0v * q0_d.q_ls + dqv * dq0_u * q0_d.q_ls)
                + m1 * (d2q0_uv * q0_d.q_ls + dq0_u * dq0_lsv + dq0v * dq0_ls_u);
            alpha_lw_d3[i] = m1 * dq0_u * dq0v * q0_d.q_ls;

            // Wiggle ↔ wiggle (bilinear in (br, dr, ddr) ⊗ (br, dr, ddr); no d3r).
            //
            // qa=brj, qb=brk, qab=0, dqa_u=drj·dq0u, dqav=drj·dq0v,
            // dqb_u=drk·dq0u, dqbv=drk·dq0v, d2qa_uv=ddrj·dq0u·dq0v+drj·d2q0_uv,
            // d2qb_uv=ddrk·dq0u·dq0v+drk·d2q0_uv, dqab_u=0, dqabv=0, d2qab_uv=0.
            //
            //   m4·dq_u·dqv·brj·brk
            // + m3·(d2q_uv·brj·brk + dq_u·(drj·dq0v·brk + brj·drk·dq0v)
            //                       + dqv·(drj·dq0u·brk + brj·drk·dq0u))
            // + m2·d2_qaqb_uv
            // where d2_qaqb_uv = (ddrj·dq0u·dq0v+drj·d2q0_uv)·brk
            //                  + drj·dq0u·drk·dq0v + drj·dq0v·drk·dq0u
            //                  + brj·(ddrk·dq0u·dq0v+drk·d2q0_uv).
            //
            // Pair (B, B): m4·dq_u·dqv + m3·d2q_uv  → coefficient of br[j]·br[k].
            // Pair (B, B'): m3·(dq_u·dq0v + dqv·dq0u) + m2·d2q0_uv → br·dr + dr·br.
            // Pair (B, B''): m2·dq0u·dq0v → br·ddr + ddr·br.
            // Pair (B', B'): 2·m2·dq0u·dq0v → dr·dr (the diagonal pair only
            //   accumulates once in `RowCoeffOperator`, so we double-count
            //   here to match the symmetric `dr[j]·dq0u·dr[k]·dq0v +
            //   dr[j]·dq0v·dr[k]·dq0u` cross product).
            c_ww_bb[i] = m4 * dq_u * dqv + m3 * d2q_uv;
            c_ww_bd[i] = m3 * (dq_u * dq0v + dqv * dq0_u) + m2 * d2q0_uv;
            c_ww_bdd[i] = m2 * dq0_u * dq0v;
            c_ww_dd_pair[i] = 2.0 * m2 * dq0_u * dq0v;
        }

        let basis: Arc<Array2<f64>> = Arc::new(b0);
        let basis_d1: Arc<Array2<f64>> = Arc::new(d0);
        let basis_d2: Arc<Array2<f64>> = Arc::new(dd0);
        let basis_d3: Arc<Array2<f64>> = Arc::new(d3_basis);

        Ok(Some(Arc::new(RowCoeffOperator::from_directions(
            vec![pt, pls, pw],
            vec![
                (0, x_t_arc),
                (1, x_ls_arc),
                (2, basis),
                (2, basis_d1),
                (2, basis_d2),
                (2, basis_d3),
            ],
            vec![
                // (X_t, X_t)   ← `d2_h[a, b] += coeff_tt · xtr[a] · xtr[b]`
                (0, 0, coeff_tt),
                // (X_t, X_ls)  ← `d2_h[a, pt+b] += coeff_tl · xtr[a] · xlsr[b]`
                (0, 1, coeff_tl),
                // (X_ls, X_ls) ← `d2_h[pt+a, pt+b] += coeff_ll · xlsr[a] · xlsr[b]`
                (1, 1, coeff_ll),
                // (X_t, B/B'/B''/B''') ← per-row α_tw_{b,d,dd,d3} decomposition of
                // `d2_h[a, pt+pls+j] += coeff_tw(i,j) · xtr[a]` (coeff_tw is
                // linear in br[j], dr[j], ddr[j], d3r[j])
                (0, 2, alpha_tw_b),
                (0, 3, alpha_tw_d),
                (0, 4, alpha_tw_dd),
                (0, 5, alpha_tw_d3),
                // (X_ls, B/B'/B''/B''') ← analogous α_lw_{b,d,dd,d3} decomposition
                // of `d2_h[pt+a, pt+pls+j] += coeff_lw(i,j) · xlsr[a]`
                (1, 2, alpha_lw_b),
                (1, 3, alpha_lw_d),
                (1, 4, alpha_lw_dd),
                (1, 5, alpha_lw_d3),
                // (B, B/B'/B'') ← bilinear decomposition of
                // `d2_h[pt+pls+j, pt+pls+k] += coeff_ww(i,j,k)` in
                // (br, dr, ddr) ⊗ (br, dr, ddr); no d3r entry — coeff_ww is
                // at most degree 2 in any single basis derivative.
                (2, 2, c_ww_bb),
                (2, 3, c_ww_bd),
                (2, 4, c_ww_bdd),
                // (B', B') diagonal — coefficient absorbs a factor of 2 to
                // match the symmetric `dr[j]·dq0u·dr[k]·dq0v + dr[j]·dq0v·
                // dr[k]·dq0u` cross product (the diagonal pair only
                // accumulates once in `RowCoeffOperator::mul_vec`).
                (3, 3, c_ww_dd_pair),
            ],
            n,
        ))))
    }
}
