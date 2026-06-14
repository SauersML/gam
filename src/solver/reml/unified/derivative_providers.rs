use super::*;

/// Provider of family-specific Hessian derivative information.
///
/// The REML/LAML gradient requires ∂H/∂ρₖ. For Gaussian, this is just Aₖ = λₖSₖ.
/// For non-Gaussian GLMs, the working curvature W(η) depends on β̂, so
/// ∂H/∂ρₖ = Aₖ + Xᵀ diag(c ⊙ Xvₖ) X where vₖ = −dβ̂/dρₖ.
/// For block-coupled families (GAMLSS, survival), the correction is
/// D_β H_L[−vₖ] using the joint likelihood Hessian.
///
/// This trait abstracts over all three cases.
pub trait HessianDerivativeProvider: Send + Sync {
    /// Compute the third-derivative correction to Hₖ.
    ///
    /// Given the mode response vₖ = H⁻¹(Aₖβ̂), returns the correction matrix
    /// such that Hₖ = Aₖ + correction.
    ///
    /// Returns `None` for Gaussian (c=d=0, no correction needed).
    fn hessian_derivative_correction(
        &self,
        v_k: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String>;

    /// Operator-capable version of `hessian_derivative_correction`.
    ///
    /// Implementations may override this to return matrix-free or composite
    /// drifts without forcing dense materialization.
    fn hessian_derivative_correction_result(
        &self,
        v_k: &Array1<f64>,
    ) -> Result<Option<DriftDerivResult>, String> {
        Ok(self
            .hessian_derivative_correction(v_k)?
            .map(DriftDerivResult::Dense))
    }

    /// Batched first-order correction hook for families whose
    /// `D_beta H[u_k]` operators share row-local state across all smoothing
    /// coordinates. The default preserves the single-direction semantics.
    fn hessian_derivative_corrections_result(
        &self,
        v_ks: &[Array1<f64>],
    ) -> Result<Vec<Option<DriftDerivResult>>, String> {
        v_ks.iter()
            .map(|v_k| self.hessian_derivative_correction_result(v_k))
            .collect()
    }

    fn has_batched_hessian_derivative_corrections(&self) -> bool {
        false
    }

    /// Compute the second-order correction to H_{k,l} for the outer Hessian.
    ///
    /// Returns `None` if not needed or not implemented.
    fn hessian_second_derivative_correction(
        &self,
        arr: &Array1<f64>,
        arr2: &Array1<f64>,
        arr3: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert!(arr.iter().all(|v| !v.is_nan()));
        assert!(arr2.iter().all(|v| !v.is_nan()));
        assert!(arr3.iter().all(|v| !v.is_nan()));
        if self.has_corrections() {
            Err(
                "HessianDerivativeProvider reports first-order corrections but does not implement second-order correction"
                    .to_string(),
            )
        } else {
            Ok(None)
        }
    }

    /// Operator-capable version of `hessian_second_derivative_correction`.
    fn hessian_second_derivative_correction_result(
        &self,
        v_k: &Array1<f64>,
        v_l: &Array1<f64>,
        u_kl: &Array1<f64>,
    ) -> Result<Option<DriftDerivResult>, String> {
        Ok(self
            .hessian_second_derivative_correction(v_k, v_l, u_kl)?
            .map(DriftDerivResult::Dense))
    }

    /// Batched second-order correction hook. The K(K+1)/2 ρ-ρ pairs in
    /// `compute_outer_hessian` each call
    /// `hessian_second_derivative_correction_result(v_k, v_l, u_kl)`; for
    /// families whose `D²H[v_k, v_l]` operators share row-local state (one
    /// per-row scan across n observations that evaluates against all
    /// triples in parallel) the batched form amortises the row-walk across
    /// pairs instead of re-scanning n rows per pair. The default preserves
    /// the single-direction semantics by looping over the singular hook.
    /// Pair the override with
    /// `has_batched_hessian_second_derivative_corrections` so the unified
    /// evaluator only routes through this when a family actually fuses the
    /// per-row work.
    ///
    /// Wired into `compute_outer_hessian`'s parallel ρ-ρ pair loop: when a
    /// provider's `has_batched_hessian_second_derivative_corrections`
    /// returns `true`, the loop precomputes all K(K+1)/2 triples (one
    /// shared `hop.solve_multi` over the pair-stacked RHS), batch-calls
    /// this hook once per outer Hessian assembly, then traces the
    /// returned drifts through the projected subspace kernel before the
    /// parallel pair sweep starts. Otherwise the loop falls back to
    /// per-pair `hessian_second_derivative_correction_result`.
    fn hessian_second_derivative_corrections_result(
        &self,
        triples: &[(Array1<f64>, Array1<f64>, Array1<f64>)],
    ) -> Result<Vec<Option<DriftDerivResult>>, String> {
        triples
            .iter()
            .map(|(v_k, v_l, u_kl)| {
                self.hessian_second_derivative_correction_result(v_k, v_l, u_kl)
            })
            .collect()
    }

    fn has_batched_hessian_second_derivative_corrections(&self) -> bool {
        false
    }

    /// Whether this provider has non-trivial corrections.
    /// False for Gaussian, true for GLMs and coupled families.
    fn has_corrections(&self) -> bool;

    /// Raw ingredients for the adjoint trace optimization.
    ///
    /// When available, the evaluator can use these to compute
    /// tr(H⁻¹ C[u]) = uᵀ z_c  (O(p) dot product instead of O(p²) solve)
    /// and fourth-derivative traces directly, without the trait having to
    /// implement the optimization algorithm.
    ///
    /// Returns `None` for Gaussian (no corrections), multi-predictor,
    /// and coupled families where the optimization doesn't apply.
    fn scalar_glm_ingredients(&self) -> Option<ScalarGlmIngredients<'_>> {
        None
    }

    /// Owned data needed for matrix-free outer Hessian-vector products.
    ///
    /// Providers that can express their second-order corrections through an
    /// owned scalar-GLM kernel or owned callback closures should override
    /// this so the unified evaluator can return an exact outer Hv operator
    /// instead of forcing dense materialization.
    fn outer_hessian_derivative_kernel(&self) -> Option<OuterHessianDerivativeKernel> {
        self.scalar_glm_ingredients()
            .map(OuterHessianDerivativeKernel::from_scalar_glm)
    }

    /// Family-supplied exact outer Hessian operator over θ = (ρ, ψ).
    ///
    /// When a family can produce the full profiled outer Hessian as a
    /// matrix-free Hv operator without enumerating θ_iθ_j pairs, it returns
    /// `Some(op)` here.  The unified evaluator then short-circuits the
    /// kernel-based assembly path at
    /// [`reml_laml_evaluate`] and routes the result
    /// straight into [`HessianResult::Operator`].
    ///
    /// Default returns `None`, in which case the evaluator falls through to
    /// the existing `outer_hessian_derivative_kernel` / `compute_outer_hessian`
    /// path.  This is the contract surface for CTN, survival, GAMLSS and
    /// other families that ship a directional outer-HVP operator.
    fn family_outer_hessian_operator(
        &self,
    ) -> Option<Arc<dyn crate::solver::outer_strategy::OuterHessianOperator>> {
        None
    }
}


/// Raw ingredients for the adjoint trace optimization in scalar GLMs.
///
/// For single-predictor GLMs, the third-derivative correction is
///   C[u] = Xᵀ diag(c ⊙ Xu) X
/// and the fourth-derivative correction is
///   Q[vₖ, vₗ] = Xᵀ diag(d ⊙ (Xvₖ)(Xvₗ)) X
///
/// The evaluator uses these arrays to implement the adjoint trace trick
/// and compute fourth-derivative traces without materializing p×p matrices.
pub struct ScalarGlmIngredients<'a> {
    /// c = dW/dη, the third-derivative weight array.
    pub c_array: &'a Array1<f64>,
    /// d = d²W/dη², the fourth-derivative weight array (`None` if zero).
    pub d_array: Option<&'a Array1<f64>>,
    /// Design matrix X in the transformed basis.
    pub x: &'a DesignMatrix,
}


#[derive(Clone)]
pub enum OuterHessianDerivativeKernel {
    /// Gaussian/constant-curvature families have no likelihood drift corrections.
    /// This marker still enables the unified exact outer-HVP operator, whose
    /// penalty/logdet/profiled-dispersion terms are fully analytic and avoid
    /// dense pairwise assembly at large n.
    Gaussian,
    ScalarGlm {
        c_array: Array1<f64>,
        d_array: Option<Array1<f64>>,
        x: DesignMatrix,
    },
    Callback {
        first: Arc<dyn Fn(&Array1<f64>) -> Result<Option<DriftDerivResult>, String> + Send + Sync>,
        second: Arc<
            dyn Fn(&Array1<f64>, &Array1<f64>) -> Result<Option<DriftDerivResult>, String>
                + Send
                + Sync,
        >,
    },
}


impl OuterHessianDerivativeKernel {
    pub(crate) fn from_scalar_glm(ingredients: ScalarGlmIngredients<'_>) -> Self {
        Self::ScalarGlm {
            c_array: ingredients.c_array.clone(),
            d_array: ingredients.d_array.cloned(),
            x: ingredients.x.clone(),
        }
    }
}


/// Null implementation for Gaussian families (c=d=0).
pub struct GaussianDerivatives;


impl HessianDerivativeProvider for GaussianDerivatives {
    fn outer_hessian_derivative_kernel(&self) -> Option<OuterHessianDerivativeKernel> {
        Some(OuterHessianDerivativeKernel::Gaussian)
    }

    fn hessian_derivative_correction(
        &self,
        arr: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert!(arr.iter().all(|v| !v.is_nan()));
        Ok(None)
    }
    fn has_corrections(&self) -> bool {
        false
    }
}


/// Single-predictor GLM derivative provider.
///
/// For non-Gaussian single-predictor models, the third-derivative correction is:
///   Cₖ = Xᵀ diag(c ⊙ X vₖ) X
/// where c is the first eta-derivative of the working curvature W(η),
/// and vₖ = H⁻¹(Aₖβ̂) is the mode response.
///
/// When the link is not canonical — probit, cloglog, SAS, mixture, or
/// beta-logistic — `c_array` and `d_array` store the **observed-information**
/// weight derivatives (c_obs, d_obs) that include residual-dependent
/// corrections:
///
///   c_obs = c_F + h'·B − (y−μ)·B_η
///   d_obs = d_F + h''·B + 2h'·B_η − (y−μ)·B_ηη
///
/// where B = (h''V − h'²V') / (φV²).  For canonical links (logit for
/// binomial, log for Poisson), B = 0 so observed = Fisher and the arrays
/// are populated with the Fisher values unchanged. These arrays are carried
/// out of PIRLS as the accepted Hessian-side curvature surface and passed
/// through `RemlState::hessian_cd_arrays` at the construction sites in
/// `runtime.rs`.
///
/// The link-parameter ext_coord path (build_sas_link_ext_coords /
/// build_mixture_link_ext_coords) independently uses observed weight
/// derivatives computed inline.
pub struct SinglePredictorGlmDerivatives {
    /// c_array: dW_obs/dη, the first eta-derivative of the observed
    /// working curvature.  For canonical links this equals c_F.
    pub c_array: Array1<f64>,
    /// d_array: d²W_obs/dη², the second eta-derivative of the observed
    /// working curvature.  For canonical links this equals d_F.
    pub d_array: Option<Array1<f64>>,
    /// Hessian-side working weights whose active rows define the curvature
    /// surface being differentiated.
    pub hessian_weights: Array1<f64>,
    /// Design matrix X in the transformed basis.
    pub x_transformed: DesignMatrix,
}


impl HessianDerivativeProvider for SinglePredictorGlmDerivatives {
    fn hessian_derivative_correction(
        &self,
        v_k: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // The Hessian derivative is dH/dρₖ = Aₖ + D_β(X'W_HX)[−vₖ].
        // Since vₖ = H⁻¹(Aₖβ̂) = −dβ̂/dρₖ, the β-direction is −vₖ, giving:
        //   D_β(X'W_HX)[−vₖ] = X' diag(c · X(−vₖ)) X
        //                     = −X' diag(c ⊙ Xvₖ) X
        // where c = dW_H/dη (the Hessian-side third-derivative weight array).
        //
        // This method returns the correction (dH/dρₖ − Aₖ), which is NEGATIVE.
        // Stays matrix-free: `matrixvectormultiply` and `xt_diag_x_signed_op`
        // route through the operator-backed design's chunked kernels at large-scale
        // scale, so we never materialize the full (n×p) dense block.
        let x_v = self.x_transformed.matrixvectormultiply(v_k); // X vₖ: n-vector

        let crate::pirls::DirectionalWorkingCurvature::Diagonal(mut neg_c_xv) =
            crate::pirls::directionalworking_curvature_from_c_array(
                &self.c_array,
                &self.hessian_weights,
                &x_v,
            );
        neg_c_xv.mapv_inplace(|value| -value);

        // −Xᵀ diag(c ⊙ Xvₖ) X via the design's matrix-free weighted gram.
        let result = self
            .x_transformed
            .xt_diag_x_signed_op(SignedWeightsView::from_array(&neg_c_xv))
            .map_err(|e| format!("hessian_derivative_correction xtwx: {e}"))?;

        Ok(Some(result))
    }

    /// #901 layer-2 fix: the first-order correction stays in OPERATOR form.
    ///
    /// `coord_corrections` (the ρ AND ψ logdet-gradient drifts) are built
    /// through this method; returning `DriftDerivResult::Operator` routes
    /// every downstream spectral-kernel trace through
    /// `reduce_operator`/`trace_operator`, whose `C·u_a` probes evaluate the
    /// near-null quadratic forms stably (see
    /// [`GlmCurvatureCorrectionOperator`]). The dense
    /// `hessian_derivative_correction` above remains for consumers that
    /// genuinely need the materialized block (outer-Hessian pair assembly).
    fn hessian_derivative_correction_result(
        &self,
        v_k: &Array1<f64>,
    ) -> Result<Option<DriftDerivResult>, String> {
        let x_v = self.x_transformed.matrixvectormultiply(v_k);
        let crate::pirls::DirectionalWorkingCurvature::Diagonal(mut neg_c_xv) =
            crate::pirls::directionalworking_curvature_from_c_array(
                &self.c_array,
                &self.hessian_weights,
                &x_v,
            );
        neg_c_xv.mapv_inplace(|value| -value);
        Ok(Some(DriftDerivResult::Operator(Arc::new(
            GlmCurvatureCorrectionOperator {
                x_design: self.x_transformed.clone(),
                neg_c_xv,
                p: self.x_transformed.ncols(),
            },
        ))))
    }

    fn hessian_second_derivative_correction(
        &self,
        v_k: &Array1<f64>,
        v_l: &Array1<f64>,
        u_kl: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // Second-order correction for the outer Hessian.
        // H_{kl} includes contributions from both c (third) and d (fourth) derivatives:
        //   Xᵀ diag(c ⊙ X u_{kl} + d ⊙ (X vₖ) ⊙ (X vₗ)) X
        // Stays matrix-free via the design's `matrixvectormultiply` and
        // `xt_diag_x_signed_op` so large-scale designs never densify the (n×p)
        // block.
        let x_vk = self.x_transformed.matrixvectormultiply(v_k);
        let x_vl = self.x_transformed.matrixvectormultiply(v_l);
        let x_ukl = self.x_transformed.matrixvectormultiply(u_kl);

        let n = self.x_transformed.nrows();
        let mut weights = Array1::zeros(n);

        // c ⊙ X u_{kl}, masked the same way as the Hessian curvature surface.
        let crate::pirls::DirectionalWorkingCurvature::Diagonal(first_weights) =
            crate::pirls::directionalworking_curvature_from_c_array(
                &self.c_array,
                &self.hessian_weights,
                &x_ukl,
            );
        weights.assign(&first_weights);

        // + d ⊙ (X vₖ) ⊙ (X vₗ)
        if let Some(ref d_array) = self.d_array {
            Zip::from(&mut weights)
                .and(d_array)
                .and(&x_vk)
                .and(&x_vl)
                .and(&self.hessian_weights)
                .par_for_each(|w, &d, &xvk, &xvl, &h| {
                    if h > 0.0 {
                        let delta = d * xvk * xvl;
                        if delta.is_finite() {
                            *w += delta;
                        }
                    }
                });
        }

        // Xᵀ diag(weights) X via the design's matrix-free weighted gram.
        let result = self
            .x_transformed
            .xt_diag_x_signed_op(SignedWeightsView::from_array(&weights))
            .map_err(|e| format!("hessian_second_derivative_correction xtwx: {e}"))?;

        Ok(Some(result))
    }

    fn has_corrections(&self) -> bool {
        true
    }

    fn scalar_glm_ingredients(&self) -> Option<ScalarGlmIngredients<'_>> {
        Some(ScalarGlmIngredients {
            c_array: &self.c_array,
            d_array: self.d_array.as_ref(),
            x: &self.x_transformed,
        })
    }
}


/// Firth-aware GLM derivative provider.
///
/// Wraps the base GLM corrections with Firth/Jeffreys Hφ corrections:
///   H_k = A_k + base_correction(v_k) − D(Hφ)[B_k]
///   H_{kl} = base_second(v_k, v_l, u_kl) − D(Hφ)[B_{kl}] − D²(Hφ)[B_k, B_l]
///
/// where B_k = −v_k (mode response) and the Firth operators use δη = X·B_k.
pub struct FirthAwareGlmDerivatives {
    pub(super) base: SinglePredictorGlmDerivatives,
    pub(super) firth_op: std::sync::Arc<super::super::FirthDenseOperator>,
}


impl HessianDerivativeProvider for FirthAwareGlmDerivatives {
    fn hessian_derivative_correction(
        &self,
        v_k: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // Base GLM correction: −Xᵀ diag(c ⊙ X vₖ) X
        let base_corr = self.base.hessian_derivative_correction(v_k)?;

        // Firth correction: −D(Hφ)[B_k] where B_k = −v_k, δη_k = X·(−v_k).
        let deta_k: Array1<f64> =
            crate::faer_ndarray::fast_av(&self.firth_op.x_dense, v_k).mapv(|v| -v);
        let dir_k = self.firth_op.direction_from_deta(deta_k);
        let firth_corr = self.firth_op.hphi_direction(&dir_k);

        match base_corr {
            Some(mut bc) => {
                bc -= &firth_corr;
                Ok(Some(bc))
            }
            None => Ok(Some(-firth_corr)),
        }
    }

    fn hessian_second_derivative_correction(
        &self,
        v_k: &Array1<f64>,
        v_l: &Array1<f64>,
        u_kl: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // Base GLM second correction: Xᵀ diag(c ⊙ X u_{kl} + d ⊙ (X vₖ)(X vₗ)) X
        let base_corr = self
            .base
            .hessian_second_derivative_correction(v_k, v_l, u_kl)?;

        // Firth D(Hφ)[B_{kl}]: B_{kl} direction is u_kl in β-space.
        let deta_kl: Array1<f64> = crate::faer_ndarray::fast_av(&self.firth_op.x_dense, u_kl);
        let dir_kl = self.firth_op.direction_from_deta(deta_kl);
        let firth_first = self.firth_op.hphi_direction(&dir_kl);

        // Firth D²(Hφ)[B_k, B_l]: second directional derivative.
        let deta_k: Array1<f64> =
            crate::faer_ndarray::fast_av(&self.firth_op.x_dense, v_k).mapv(|v| -v);
        let dir_k = self.firth_op.direction_from_deta(deta_k);
        let deta_l: Array1<f64> =
            crate::faer_ndarray::fast_av(&self.firth_op.x_dense, v_l).mapv(|v| -v);
        let dir_l = self.firth_op.direction_from_deta(deta_l);
        let p = v_k.len();
        let eye = Array2::<f64>::eye(p);
        let firth_second = self
            .firth_op
            .hphisecond_direction_apply(&dir_k, &dir_l, &eye);

        let mut result = match base_corr {
            Some(bc) => bc,
            None => Array2::zeros((p, p)),
        };
        result -= &firth_first;
        result -= &firth_second;
        Ok(Some(result))
    }

    /// #901 layer-2: keep the base GLM cubic correction in operator form and
    /// graft the (dense, well-conditioned) Firth part on through
    /// [`CompositeHyperOperator`], mirroring `BarrierDerivativeProvider`.
    /// The roundoff-critical near-null quadratic forms live entirely in the
    /// base `Xᵀ diag(c⊙Xv) X` sandwich; the Firth `−D(Hφ)[B_k]` block stays
    /// dense as before.
    fn hessian_derivative_correction_result(
        &self,
        v_k: &Array1<f64>,
    ) -> Result<Option<DriftDerivResult>, String> {
        let base = self.base.hessian_derivative_correction_result(v_k)?;

        let deta_k: Array1<f64> =
            crate::faer_ndarray::fast_av(&self.firth_op.x_dense, v_k).mapv(|v| -v);
        let dir_k = self.firth_op.direction_from_deta(deta_k);
        let neg_firth_corr = -self.firth_op.hphi_direction(&dir_k);

        match base {
            Some(DriftDerivResult::Operator(operator)) => Ok(Some(DriftDerivResult::Operator(
                Arc::new(CompositeHyperOperator {
                    dense: Some(neg_firth_corr),
                    operators: vec![operator],
                    dim_hint: self.base.x_transformed.ncols(),
                }),
            ))),
            Some(DriftDerivResult::Dense(mut dense)) => {
                dense += &neg_firth_corr;
                Ok(Some(DriftDerivResult::Dense(dense)))
            }
            None => Ok(Some(DriftDerivResult::Dense(neg_firth_corr))),
        }
    }

    fn has_corrections(&self) -> bool {
        true
    }

    fn scalar_glm_ingredients(&self) -> Option<ScalarGlmIngredients<'_>> {
        None
    }
}


/// Exact Jeffreys/Firth term used by the unified outer evaluator.
///
/// The scalar contribution and all outer derivatives must be sourced from the
/// same operator in the same coefficient basis.
#[derive(Clone)]
pub struct ExactJeffreysTerm {
    /// Tier-A GLM dense operator carrying the value and all β-gradient
    /// machinery. `None` for the Tier-B value-only carrier (see
    /// [`ExactJeffreysTerm::value_only`]), where the coupled joint path
    /// supplies the curvature/drift terms through its own
    /// `H_Φ`-aware derivative provider and only the scalar `Φ(β̂)` needs to
    /// reach the LAML cost.
    pub(crate) operator: Option<std::sync::Arc<super::super::FirthDenseOperator>>,
    /// Tangent-projected value override. When `Some`, `value()` returns
    /// this scalar instead of the operator's full-space `½ log|J|`. This
    /// is used by `try_tangent_projected_evaluate` to substitute
    /// `½ log|ZᵀJZ|` while reusing the rest of the evaluator pipeline.
    /// The same `Arc<FirthDenseOperator>` is retained so any downstream
    /// consumer that accesses the operator (e.g. for β-gradient terms)
    /// sees the unmodified operator; only the scalar contribution to the
    /// outer LAML cost changes. For the Tier-B value-only carrier this is
    /// always `Some` (it IS the value).
    pub(crate) value_override: Option<f64>,
}


impl ExactJeffreysTerm {
    pub(crate) fn new(operator: std::sync::Arc<super::super::FirthDenseOperator>) -> Self {
        Self {
            operator: Some(operator),
            value_override: None,
        }
    }

    /// Tier-B value-only carrier: the coupled joint custom-family path folds
    /// the gated Jeffreys value `Φ(β̂) = ½ log|H_id|` into the LAML cost
    /// (`cost −= Φ`) so the outer criterion is the Laplace approximation of
    /// the SAME Firth-augmented objective `−ℓ + ½βᵀSβ − Φ` the inner Newton
    /// converged on. Without this fold the envelope identity breaks at every
    /// Firth-active mode: `∇_β(−ℓ + ½βᵀSβ)(β̂) = +∇Φ ≠ 0`, so the analytic
    /// outer gradient (which differentiates the Φ-folded criterion via the
    /// envelope) disagrees with the finite difference of the Φ-less value
    /// by `(∇Φ)ᵀ ∂β̂/∂ρ` (gam#979). The β-gradient/curvature machinery for
    /// Tier-B lives in the `H_Φ`-aware joint derivative provider, not here.
    pub(crate) fn value_only(phi: f64) -> Self {
        Self {
            operator: None,
            value_override: Some(phi),
        }
    }

    /// Construct a tangent-projected variant: wraps the same operator but
    /// returns `½ log|ZᵀJZ|` from `value()`.
    pub(crate) fn with_projected_value(
        operator: std::sync::Arc<super::super::FirthDenseOperator>,
        projected_value: f64,
    ) -> Self {
        Self {
            operator: Some(operator),
            value_override: Some(projected_value),
        }
    }

    #[inline]
    pub(crate) fn value(&self) -> f64 {
        self.value_override.unwrap_or_else(|| {
            self.operator
                .as_ref()
                .map_or(0.0, |operator| operator.jeffreys_logdet())
        })
    }

    #[inline]
    pub(crate) fn operator_arc(&self) -> Option<std::sync::Arc<super::super::FirthDenseOperator>> {
        self.operator.as_ref().map(std::sync::Arc::clone)
    }
}


// ═══════════════════════════════════════════════════════════════════════════
//  Guarded scalar correction (value + ρ-gradient under ONE include flag)
// ═══════════════════════════════════════════════════════════════════════════

/// A scalar objective correction whose VALUE and analytic ρ-GRADIENT are
/// carried together and applied through a SINGLE site under a SINGLE guard.
///
/// This is the structural cure for the recurring objective↔gradient desync
/// bug class (issues #752/#748/#808 and the latent Tierney–Kadane desync):
/// when a correction's value and its derivative are added to the cost and the
/// ρ-gradient in physically separate statements — each with its own
/// hand-written `if include_logdet_h { … }` guard — the two drift apart. Here
/// the `include` flag is read ONCE and gates BOTH contributions in
/// [`GuardedCorrection::apply`], so a future edit cannot re-introduce the
/// half-applied/half-omitted state by construction.
///
/// Mirrors the already-paired `PenaltyLogdetDerivs` / `joint_jeffreys_term`
/// objects, which return value+derivative together for exactly this reason.
pub(crate) struct GuardedCorrection {
    /// Scalar contribution to the outer REML/LAML cost.
    pub(crate) value: f64,
    /// Contribution to the ρ-gradient (one entry per active ρ coordinate),
    /// `None` when the correction is value-only (derivative-free regime).
    pub(crate) gradient: Option<Array1<f64>>,
    /// The SINGLE guard. When `false`, NEITHER the value nor the gradient is
    /// applied; when `true`, BOTH are.
    pub(crate) include: bool,
}


impl GuardedCorrection {
    /// Construct a guarded correction from a loose `(value, gradient)` pair and
    /// the include flag that must gate both.
    pub(crate) fn new(value: f64, gradient: Option<Array1<f64>>, include: bool) -> Self {
        Self {
            value,
            gradient,
            include,
        }
    }

    /// Apply the VALUE contribution to `cost` under the single `include` guard.
    pub(crate) fn apply_value(&self, cost: &mut f64) {
        if self.include {
            *cost += self.value;
        }
    }

    /// Apply the ρ-GRADIENT contribution to the leading entries of `rho_grad`
    /// under the SAME single `include` guard read from `self`.
    pub(crate) fn apply_gradient(&self, rho_grad: &mut Array1<f64>) {
        if !self.include {
            return;
        }
        if let Some(grad) = self.gradient.as_ref() {
            let k = grad.len();
            let mut sl = rho_grad.slice_mut(ndarray::s![..k]);
            sl += grad;
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════════
//  Log-barrier support for constrained coefficients
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for a log-barrier penalty on constrained coefficients.
///
/// The barrier-augmented objective adds `-τ Σ_{j ∈ C} log(s_j β_j − b_j)`,
/// where `s_j = 1` for lower bounds and `s_j = -1` for upper bounds.
/// τ is an algorithmic continuation parameter — NOT a hyperparameter.
#[derive(Clone, Debug)]
pub struct BarrierConfig {
    /// Barrier strength parameter (continuation schedule drives this → 0).
    pub tau: f64,
    /// Indices of constrained coefficients in the β vector.
    pub constrained_indices: Vec<usize>,
    /// Right-hand-side `b_j` for each directional coordinate constraint.
    pub lower_bounds: Vec<f64>,
    /// Direction `s_j` for each coordinate constraint `s_j β_j >= b_j`.
    pub bound_signs: Vec<f64>,
}


impl BarrierConfig {
    /// Construct a `BarrierConfig` from linear inequality constraints `A β ≥ b`
    /// by extracting rows that represent simple coordinate bounds
    /// (`β_j ≥ b_i` or `β_j ≤ -b_i`).
    ///
    /// A row is a simple bound iff it has exactly one nonzero entry equal to ±1.0.
    /// Returns `None` if the constraints are `None` or no simple-bound rows are found.
    pub fn from_constraints(
        constraints: Option<&crate::pirls::LinearInequalityConstraints>,
    ) -> Option<Self> {
        // Tolerance for recognizing a constraint-matrix entry as exactly 0 or
        // exactly ±1, so a row qualifies as a simple coordinate bound. The
        // constraint rows are assembled exactly, so any nonzero deviation this
        // large is a genuine multi-coefficient constraint, not round-off.
        const SIMPLE_BOUND_ENTRY_TOL: f64 = 1e-14;
        // Default log-barrier strength τ used when a simple-bound BarrierConfig
        // is synthesized from constraints (a weak barrier that keeps β strictly
        // feasible without materially perturbing an interior optimum).
        const DEFAULT_BARRIER_TAU: f64 = 1e-6;
        let constraints = constraints?;
        let mut indices = Vec::new();
        let mut lower_bounds = Vec::new();
        let mut bound_signs = Vec::new();
        for i in 0..constraints.a.nrows() {
            let row = constraints.a.row(i);
            let mut single_col = None;
            let mut single_sign = 0.0_f64;
            let mut is_simple = true;
            for (j, &val) in row.iter().enumerate() {
                if val.abs() < SIMPLE_BOUND_ENTRY_TOL {
                    continue;
                }
                if ((val - 1.0).abs() < SIMPLE_BOUND_ENTRY_TOL
                    || (val + 1.0).abs() < SIMPLE_BOUND_ENTRY_TOL)
                    && single_col.is_none()
                {
                    single_col = Some(j);
                    single_sign = if val > 0.0 { 1.0 } else { -1.0 };
                } else {
                    is_simple = false;
                    break;
                }
            }
            if is_simple && let Some(col) = single_col {
                indices.push(col);
                lower_bounds.push(constraints.b[i]);
                bound_signs.push(single_sign);
            }
        }
        if indices.is_empty() {
            return None;
        }
        Some(BarrierConfig {
            tau: DEFAULT_BARRIER_TAU,
            constrained_indices: indices,
            lower_bounds,
            bound_signs,
        })
    }

    /// Compute slack values Δ_j = s_j β_j − b_j. Returns `None` if infeasible.
    pub fn slacks(&self, beta: &Array1<f64>) -> Option<Vec<f64>> {
        let mut slacks = Vec::with_capacity(self.constrained_indices.len());
        for (ci, &idx) in self.constrained_indices.iter().enumerate() {
            let sign = self.bound_signs[ci];
            let delta = sign * beta[idx] - self.lower_bounds[ci];
            if delta <= 0.0 {
                return None;
            }
            slacks.push(delta);
        }
        Some(slacks)
    }

    /// Add the barrier Hessian diagonal τ·D^(2) to H in-place.
    pub fn add_barrier_hessian_diagonal(
        &self,
        h: &mut Array2<f64>,
        beta: &Array1<f64>,
    ) -> Result<(), String> {
        let slacks = self
            .slacks(beta)
            .ok_or_else(|| "Barrier: infeasible point (slack ≤ 0)".to_string())?;
        for (ci, &idx) in self.constrained_indices.iter().enumerate() {
            h[[idx, idx]] += self.tau / (slacks[ci] * slacks[ci]);
        }
        Ok(())
    }

    /// Compute the barrier cost `−τ Σ log(Δ_j)`.
    ///
    /// **Contract.** The log-barrier objective is, by construction, a
    /// real-valued function of β on the feasible interior that diverges to
    /// `+∞` as any slack `Δ_j = s_j β_j − b_j` approaches `0⁺`. We extend it
    /// continuously to the closed exterior `Δ_j ≤ 0` by the same limit:
    /// `barrier_cost(β) = +∞` whenever any constrained coordinate has reached
    /// or crossed its bound. This makes the barrier objective composable with
    /// generic line-search / trust-region code that compares scalar
    /// objectives — an infeasible trial step is automatically rejected by
    /// monotonicity, with no special-cased `Err` branch in every call site.
    ///
    /// We never return NaN: at `Δ_j = 0` exactly we shortcut to `+∞` rather
    /// than evaluating `ln(0) = −∞` (which would multiply with `−τ` to give
    /// `+∞` but only after a non-finite intermediate); at `Δ_j < 0` we
    /// shortcut to `+∞` rather than computing `ln(negative) = NaN`.
    pub fn barrier_cost(&self, beta: &Array1<f64>) -> f64 {
        let mut total = 0.0_f64;
        for (ci, &idx) in self.constrained_indices.iter().enumerate() {
            let sign = self.bound_signs[ci];
            let delta = sign * beta[idx] - self.lower_bounds[ci];
            if delta <= 0.0 {
                return f64::INFINITY;
            }
            // Δ > 0 here, so ln(Δ) is finite; contribution is finite real.
            total += delta.ln();
        }
        -self.tau * total
    }

    /// Detection of barrier-dominated geometry, where EFS — which assumes
    /// inner Hessian ≈ X'WX + S and ignores the log-barrier drift
    /// `τ / (β_j − l_j)²` on its diagonal — becomes unreliable. Returns
    /// `true` whenever at least one of the following holds (each captures a
    /// distinct failure mode of the EFS precondition):
    ///
    /// (a) **Asymmetric concentration.** With slacks Δ_j = β_j − l_j,
    /// `min_j Δ_j < ratio · median_j Δ_j`. This is a *scale-free* check
    /// using only slack ratios, so it is independent of the absolute scale
    /// of β. It catches the common pathology where one constrained
    /// coefficient runs to its bound while the rest stay healthy — that
    /// one coord's `τ/Δ²` then dominates the inner Hessian diagonal at
    /// that coord, and EFS's multiplicative update is no longer
    /// guaranteed-ascent there.
    ///
    /// (b) **Absolute saturation.** `τ / min_j Δ_j² ≥ saturation_threshold`.
    /// This is a *dimensional* check that catches the case (a) misses:
    /// when ALL slacks shrink together near the optimum, slack ratios stay
    /// near 1 but the per-coord barrier curvature still saturates. With
    /// the default `τ = 1e-6` and a `saturation_threshold` of 1.0 (the
    /// natural unit penalty scale), this fires at `Δ_min ≲ 1e-3`.
    ///
    /// Returns `true` on infeasible β (Δ_j ≤ 0).
    ///
    /// Replaces the older `barrier_curvature_is_significant(_, ref_diag, _)`,
    /// whose `ref_diag` was a representative diagonal of `X'W_HX + S` that
    /// no call site could compute correctly without surfacing the inner
    /// Hessian out to the EFS bridge.
    pub fn barrier_curvature_locally_concentrated(
        &self,
        beta: &Array1<f64>,
        ratio: f64,
        saturation_threshold: f64,
    ) -> bool {
        let Some(mut slacks) = self.slacks(beta) else {
            return true; // infeasible → conservatively unreliable
        };
        if slacks.is_empty() {
            return false;
        }
        let min_slack = slacks.iter().copied().fold(f64::INFINITY, f64::min);

        // (b) Absolute saturation: τ / Δ_min² ≥ threshold. Catches the
        // symmetric near-boundary regime that ratio-only checks miss.
        if min_slack > 0.0 && min_slack.is_finite() && saturation_threshold.is_finite() {
            let max_barrier_curv = self.tau / (min_slack * min_slack);
            if max_barrier_curv >= saturation_threshold {
                return true;
            }
        }

        // (a) Asymmetric concentration: min Δ ≪ median Δ.
        slacks.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if slacks.len() % 2 == 1 {
            slacks[slacks.len() / 2]
        } else {
            let mid = slacks.len() / 2;
            0.5 * (slacks[mid - 1] + slacks[mid])
        };
        if !median.is_finite() || median <= 0.0 {
            return true;
        }
        min_slack < ratio * median
    }

    /// Check whether the barrier curvature is non-negligible relative to a
    /// reference Hessian diagonal scale.
    ///
    /// Returns `true` when `max_j τ / (β_j − l_j)² > threshold * ref_diag`,
    /// indicating that EFS (which ignores the barrier Hessian drift) would be
    /// unreliable. If β is infeasible, conservatively returns `true`.
    ///
    /// `ref_diag` should be a representative diagonal of X'W_HX + S (e.g. the
    /// median or mean). A typical `threshold` is 0.01–0.1.
    pub fn barrier_curvature_is_significant(
        &self,
        beta: &Array1<f64>,
        ref_diag: f64,
        threshold: f64,
    ) -> bool {
        let Some(slacks) = self.slacks(beta) else {
            return true; // infeasible → conservatively active
        };
        let max_barrier_curv = slacks
            .iter()
            .map(|&d| self.tau / (d * d))
            .fold(0.0_f64, f64::max);
        max_barrier_curv > threshold * ref_diag
    }
}


/// Barrier-aware Hessian derivative provider wrapping an inner provider.
///
/// Adds C_bar[u] = −2τ·diag(u ⊙ d^(3)) and Q_bar[u,v] = 6τ·diag(u ⊙ v ⊙ d^(4)).
pub struct BarrierDerivativeProvider<'a> {
    pub(crate) inner: &'a dyn HessianDerivativeProvider,
    pub(crate) tau: f64,
    pub(crate) constrained_indices: &'a [usize],
    pub(crate) bound_signs: &'a [f64],
    pub(crate) slacks: Vec<f64>,
    pub(crate) p: usize,
}


impl<'a> BarrierDerivativeProvider<'a> {
    pub fn new(
        inner: &'a dyn HessianDerivativeProvider,
        config: &'a BarrierConfig,
        beta: &Array1<f64>,
    ) -> Result<Self, String> {
        let slacks = config
            .slacks(beta)
            .ok_or_else(|| "BarrierDerivativeProvider: infeasible point".to_string())?;
        Ok(Self {
            inner,
            tau: config.tau,
            constrained_indices: &config.constrained_indices,
            bound_signs: &config.bound_signs,
            slacks,
            p: beta.len(),
        })
    }

    pub(crate) fn barrier_correction(&self, u: &Array1<f64>) -> Array2<f64> {
        let mut result = Array2::zeros((self.p, self.p));
        for (ci, &idx) in self.constrained_indices.iter().enumerate() {
            let inv_cube = 1.0 / (self.slacks[ci].powi(3));
            result[[idx, idx]] = -2.0 * self.tau * self.bound_signs[ci] * u[idx] * inv_cube;
        }
        result
    }

    pub(crate) fn barrier_second_correction(&self, u: &Array1<f64>, v: &Array1<f64>) -> Array2<f64> {
        let mut result = Array2::zeros((self.p, self.p));
        for (ci, &idx) in self.constrained_indices.iter().enumerate() {
            let inv_4 = 1.0 / (self.slacks[ci].powi(4));
            result[[idx, idx]] = 6.0 * self.tau * u[idx] * v[idx] * inv_4;
        }
        result
    }
}


impl HessianDerivativeProvider for BarrierDerivativeProvider<'_> {
    fn hessian_derivative_correction(
        &self,
        v_k: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // The trait convention passes vₖ = H⁻¹(Aₖβ̂), but the barrier
        // third-derivative should be evaluated at the mode sensitivity
        // direction β̂_ρk = −vₖ.  barrier_correction(u) computes
        // D_β(B_ββ)[u] = −2τ u_j/gap³, so we negate vₖ to get:
        //   D_β(B_ββ)[−vₖ] = +2τ vₖ_j/gap³.
        let neg_v_k = v_k.mapv(|x| -x);
        let barrier_corr = self.barrier_correction(&neg_v_k);
        match self.inner.hessian_derivative_correction(v_k)? {
            Some(mut ic) => {
                ic += &barrier_corr;
                Ok(Some(ic))
            }
            None => Ok(Some(barrier_corr)),
        }
    }

    fn hessian_derivative_correction_result(
        &self,
        v_k: &Array1<f64>,
    ) -> Result<Option<DriftDerivResult>, String> {
        let neg_v_k = v_k.mapv(|x| -x);
        let barrier_corr = self.barrier_correction(&neg_v_k);
        match self.inner.hessian_derivative_correction_result(v_k)? {
            Some(DriftDerivResult::Dense(mut dense)) => {
                dense += &barrier_corr;
                Ok(Some(DriftDerivResult::Dense(dense)))
            }
            Some(DriftDerivResult::Operator(operator)) => Ok(Some(DriftDerivResult::Operator(
                Arc::new(CompositeHyperOperator {
                    dense: Some(barrier_corr),
                    operators: vec![operator],
                    dim_hint: self.p,
                }),
            ))),
            None => Ok(Some(DriftDerivResult::Dense(barrier_corr))),
        }
    }

    fn hessian_second_derivative_correction(
        &self,
        v_k: &Array1<f64>,
        v_l: &Array1<f64>,
        u_kl: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let barrier_total =
            &self.barrier_correction(u_kl) + &self.barrier_second_correction(v_k, v_l);
        match self
            .inner
            .hessian_second_derivative_correction(v_k, v_l, u_kl)?
        {
            Some(mut ic) => {
                ic += &barrier_total;
                Ok(Some(ic))
            }
            None => Ok(Some(barrier_total)),
        }
    }

    fn hessian_second_derivative_correction_result(
        &self,
        v_k: &Array1<f64>,
        v_l: &Array1<f64>,
        u_kl: &Array1<f64>,
    ) -> Result<Option<DriftDerivResult>, String> {
        let barrier_total =
            &self.barrier_correction(u_kl) + &self.barrier_second_correction(v_k, v_l);
        match self
            .inner
            .hessian_second_derivative_correction_result(v_k, v_l, u_kl)?
        {
            Some(DriftDerivResult::Dense(mut dense)) => {
                dense += &barrier_total;
                Ok(Some(DriftDerivResult::Dense(dense)))
            }
            Some(DriftDerivResult::Operator(operator)) => Ok(Some(DriftDerivResult::Operator(
                Arc::new(CompositeHyperOperator {
                    dense: Some(barrier_total),
                    operators: vec![operator],
                    dim_hint: self.p,
                }),
            ))),
            None => Ok(Some(DriftDerivResult::Dense(barrier_total))),
        }
    }

    fn has_corrections(&self) -> bool {
        true
    }

    fn scalar_glm_ingredients(&self) -> Option<ScalarGlmIngredients<'_>> {
        None
    }
}


// ═══════════════════════════════════════════════════════════════════════════
//  Link-wiggle derivative provider (exact second-order Hessian corrections)
// ═══════════════════════════════════════════════════════════════════════════

