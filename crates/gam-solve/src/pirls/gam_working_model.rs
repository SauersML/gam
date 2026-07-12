//! The concrete GLM `WorkingModel`: `GamWorkingModel` assembles the working
//! response/weights, the penalized Hessian, and the curvature arrays, and
//! implements the `WorkingModel` trait (update / candidate-screen). Carries the
//! fixed stabilization ridge and the `GamModelFinalState` snapshot.

use super::*;

// Fixed stabilization ridge for PIRLS/PLS. `penalty_term` carries this as
// ridge * ||beta||^2 (equivalently 0.5 * ridge * ||beta||^2 in the
// 0.5 * (deviance + penalty_term) objective), and it is constant w.r.t. rho.
//
// Math note:
//   Objective: V(ρ) includes log|H(ρ)| with H(ρ) = X' W X + S_λ(ρ) + δ I.
//   If δ = δ(ρ) is adaptive, V(ρ) is only piecewise-smooth and ∂V/∂ρ ignores
//   ∂δ/∂ρ, causing a mismatch between the optimized surface and the analytic
//   derivative surface. Using a fixed δ makes V(ρ) smooth and the standard
//   envelope-theorem gradient valid:
//     dV/dρ_k = 0.5 λ_k βᵀ S_k β + 0.5 λ_k tr(H^{-1} S_k) - 0.5 det1[k].
pub(crate) const FIXED_STABILIZATION_RIDGE: f64 = 1e-8;

pub(crate) struct GamWorkingModel<'a> {
    pub(crate) x_original: DesignMatrix,
    pub(crate) coordinate_design: WorkingCoordinateDesign,
    pub(crate) offset: Array1<f64>,
    pub(crate) y: ArrayView1<'a, f64>,
    pub(crate) priorweights: ArrayView1<'a, f64>,
    pub(crate) penalty: PirlsPenalty,
    pub(crate) workspace: PirlsWorkspace,
    pub(crate) likelihood: GlmLikelihoodSpec,
    pub(crate) link_kind: InverseLink,
    pub(crate) firth_bias_reduction: bool,
    pub(crate) lastmu: Array1<f64>,
    pub(crate) lastweights: Array1<f64>,
    pub(crate) lastz: Array1<f64>,
    pub(crate) last_c: Array1<f64>,
    pub(crate) last_d: Array1<f64>,
    pub(crate) lasthessian_weights: Array1<f64>,
    pub(crate) lasthessian_c: Array1<f64>,
    pub(crate) lasthessian_d: Array1<f64>,
    pub(crate) lasthessian_curvature: HessianCurvatureKind,
    pub(crate) last_dmu_deta: Array1<f64>,
    pub(crate) last_d2mu_deta2: Array1<f64>,
    pub(crate) last_d3mu_deta3: Array1<f64>,
    pub(crate) last_penalty_term: f64,
    pub(crate) x_original_csr: Option<SparseRowMat<usize, f64>>,
    /// Optional per-observation SE for integrated (GHQ) likelihood.
    /// When present, uses integrated family-dispatched working updates.
    pub(crate) covariate_se: Option<Array1<f64>>,
    /// Whether the Gamma dispersion shape has been estimated and frozen for the
    /// duration of this inner P-IRLS solve. The shape (= 1/φ) is a nuisance
    /// scale that multiplies both the working weight (`w = shape·prior`) and the
    /// reported deviance (`2·shape·Σ wᵢ dᵢ`). Re-estimating it per inner Newton/LM
    /// iterate moves the product φ·λ that the penalized argmin β̂ depends on, so
    /// the LM gain ratio compares two different objectives and the solve stalls.
    /// The shape is therefore estimated once from the warm-start η on the first
    /// curvature build and held fixed; it refreshes naturally across *outer*
    /// iterations because a fresh `GamWorkingModel` is built per inner solve.
    /// See issue #511 (regression of #359).
    pub(crate) gamma_shape_locked: bool,
    /// Whether the Beta-regression precision `phi` has been estimated and frozen
    /// for the duration of this inner P-IRLS solve. Like the Gamma shape, `phi`
    /// is a nuisance scale entering the working weight `w ∝ (1+phi)` and the
    /// variance `Var(y)=mu(1-mu)/(1+phi)`; re-estimating it per Newton/LM iterate
    /// moves the penalized argmin, so it is estimated once from the warm-start η
    /// and held fixed within the inner solve, refreshing across outer iterations
    /// (a fresh working model is built per inner solve). Issue #567.
    pub(crate) beta_phi_locked: bool,
    /// Whether the Tweedie dispersion `phi` has been estimated and frozen for the
    /// duration of this inner P-IRLS solve. Like the Gamma shape, `phi` is a
    /// nuisance scale entering only the working weight (`prior·μ^{2−p}/phi`) and
    /// not the working response, so re-estimating it per Newton/LM iterate would
    /// move the product `φ·λ` the penalized argmin β̂ depends on and stall the LM
    /// gain ratio. It is therefore estimated once from the warm-start η and held
    /// fixed within the inner solve, refreshing across outer iterations (a fresh
    /// working model is built per inner solve). Issue #771.
    pub(crate) tweedie_phi_locked: bool,
    /// Whether the Negative-Binomial overdispersion `theta` has been estimated
    /// and frozen for the duration of this inner P-IRLS solve. `theta` enters the
    /// working weight `W = μθ/(θ+μ)` (the NB2 Fisher information) and the working
    /// response, so — like the Beta precision, and unlike the scale-free Gamma
    /// shape — re-estimating it per Newton/LM iterate would move the penalized
    /// argmin β̂ and stall the LM gain ratio. It is therefore estimated once from
    /// the warm-start η and held fixed within the inner solve, refreshing across
    /// outer iterations (a fresh working model is built per inner solve). The
    /// converged-η joint refresh in `loop_driver` re-arms this lock so the
    /// reported `theta` is exactly the ML estimate at the reported η. Issue #802.
    pub(crate) negbin_theta_locked: bool,
    pub(crate) quadctx: crate::quadrature::QuadratureContext,
    /// Frozen-weight first-Fisher-step data-fit Gram `XᵀWX` (#1111 / #1033
    /// mechanism (c)), in the same *original* (conditioned `x_fit`) frame
    /// `penalized_hessian` forms `compute_xtwx_blas(self.x_original, ...)` in,
    /// i.e. BEFORE any Qs conjugation. When present it serves the FIRST
    /// Fisher-scoring iteration's `XᵀWX` n-free, eliding the dominant
    /// O(N·p²) weighted cross-product on a large-n GLM ψ-trial. Consumed at
    /// most once per inner solve (the first `penalized_hessian` build at the
    /// warm β); later iterations restream the true moving `W`.
    pub(crate) glm_first_step_gram: Option<Array2<f64>>,
    /// Set once the frozen-W first-step Gram has been consumed, so subsequent
    /// inner iterations restream `XᵀWX` from the (moving) working weights.
    pub(crate) glm_first_step_gram_consumed: bool,
    /// β-independent (design-only) factor of the Firth/Jeffreys operator,
    /// memoized for the lifetime of this inner P-IRLS solve (#1575). The design
    /// and prior weights are constant across the inner Newton iterations while
    /// `η` changes every iteration, so the O(n·p²) Gram, the O(p³) identifiable-
    /// subspace eigendecomposition, and the n×p design clones are computed once
    /// here and reused; only the cheap per-`η` reduced Fisher / hat-diagonal
    /// remainder is rebuilt per iteration. Lazily filled on the first Firth
    /// diagnostic build and reused thereafter; a fresh working model is built
    /// per inner solve so it refreshes naturally when the design changes.
    pub(crate) firth_design_factor: Option<Arc<FirthDesignFactor>>,
}

pub(crate) struct GamModelFinalState {
    pub(crate) likelihood: GlmLikelihoodSpec,
    pub(crate) coordinate_frame: PirlsCoordinateFrame,
    pub(crate) finalmu: Array1<f64>,
    pub(crate) finalweights: Array1<f64>,
    pub(crate) scoreweights: Array1<f64>,
    pub(crate) finalz: Array1<f64>,
    pub(crate) final_c: Array1<f64>,
    pub(crate) final_d: Array1<f64>,
    pub(crate) final_dmu_deta: Array1<f64>,
    pub(crate) final_d2mu_deta2: Array1<f64>,
    pub(crate) final_d3mu_deta3: Array1<f64>,
    pub(crate) penalty_term: f64,
}

impl<'a> GamWorkingModel<'a> {
    pub(crate) fn new(
        x_transformed: Option<DesignMatrix>,
        x_original: DesignMatrix,
        coordinate_frame: PirlsCoordinateFrame,
        offset: ArrayView1<f64>,
        y: ArrayView1<'a, f64>,
        priorweights: ArrayView1<'a, f64>,
        penalty: PirlsPenalty,
        workspace: PirlsWorkspace,
        likelihood: GlmLikelihoodSpec,
        link_kind: InverseLink,
        firth_bias_reduction: bool,
        transform: Option<WorkingReparamTransform>,
        quadctx: crate::quadrature::QuadratureContext,
        glm_first_step_gram: Option<Array2<f64>>,
    ) -> Self {
        let coordinate_design = match coordinate_frame {
            PirlsCoordinateFrame::OriginalSparseNative => {
                WorkingCoordinateDesign::OriginalSparseNative
            }
            PirlsCoordinateFrame::TransformedQs => {
                if let Some(x_transformed) = x_transformed {
                    WorkingCoordinateDesign::TransformedExplicit {
                        x_csr: x_transformed.to_csr_cache(),
                        x_transformed,
                    }
                } else {
                    WorkingCoordinateDesign::TransformedImplicit {
                        transform: transform.expect(
                            "TransformedQs PIRLS coordinate frame requires either x_transformed or qs",
                        ),
                    }
                }
            }
        };
        let x_original_csr = x_original.to_csr_cache();
        let n = match &coordinate_design {
            WorkingCoordinateDesign::OriginalSparseNative => x_original.nrows(),
            WorkingCoordinateDesign::TransformedExplicit { x_transformed, .. } => {
                x_transformed.nrows()
            }
            WorkingCoordinateDesign::TransformedImplicit { .. } => x_original.nrows(),
        };
        GamWorkingModel {
            x_original,
            coordinate_design,
            offset: offset.to_owned(),
            y,
            priorweights,
            penalty,
            workspace,
            likelihood,
            link_kind,
            firth_bias_reduction,
            lastmu: Array1::zeros(n),
            lastweights: Array1::zeros(n),
            lastz: Array1::zeros(n),
            last_c: Array1::zeros(n),
            last_d: Array1::zeros(n),
            lasthessian_weights: Array1::zeros(n),
            lasthessian_c: Array1::zeros(n),
            lasthessian_d: Array1::zeros(n),
            lasthessian_curvature: HessianCurvatureKind::Fisher,
            last_dmu_deta: Array1::zeros(n),
            last_d2mu_deta2: Array1::zeros(n),
            last_d3mu_deta3: Array1::zeros(n),
            last_penalty_term: 0.0,
            x_original_csr,
            covariate_se: None,
            gamma_shape_locked: false,
            beta_phi_locked: false,
            tweedie_phi_locked: false,
            negbin_theta_locked: false,
            quadctx,
            glm_first_step_gram,
            glm_first_step_gram_consumed: false,
            firth_design_factor: None,
        }
    }

    /// Set per-observation SE for integrated (GHQ) likelihood.
    /// When set, the working model uses uncertainty-aware IRLS updates.
    pub(crate) fn with_covariate_se(mut self, se: Array1<f64>) -> Self {
        self.covariate_se = Some(se);
        self
    }

    /// Build (once) and return the β-independent Firth/Jeffreys design factor for
    /// the current coordinate design (#1575). The factor is materialized in the
    /// SAME coefficient basis the inner objective is optimized in — transformed
    /// (`x_transformed`/`X·Qs`) when a reparameterization is in effect, original
    /// otherwise — exactly as the previous per-iteration diagnostics path. It is
    /// memoized on the working model and reused across the inner Newton
    /// iterations of this solve, since the design and prior weights are constant
    /// for the model's lifetime.
    fn ensure_firth_design_factor(&mut self) -> Result<Arc<FirthDesignFactor>, EstimationError> {
        if let Some(factor) = &self.firth_design_factor {
            return Ok(factor.clone());
        }
        let factor = match &self.coordinate_design {
            WorkingCoordinateDesign::TransformedExplicit {
                x_transformed,
                x_csr,
            } => {
                if x_transformed.as_sparse().is_some() {
                    let csr = x_csr.as_ref().ok_or_else(|| {
                        EstimationError::InvalidInput(
                            "missing CSR cache for sparse transformed design".to_string(),
                        )
                    })?;
                    build_firth_design_factor_sparse(csr, self.priorweights)?
                } else {
                    let x_dense_cow = x_transformed.to_dense_cow();
                    build_firth_design_factor_dense(x_dense_cow.view(), self.priorweights)?
                }
            }
            WorkingCoordinateDesign::TransformedImplicit { transform } => {
                // Materialize X·Qs on demand so the factor lives in the same
                // transformed basis as the inner objective.
                let x_t_dense =
                    fast_ab(&self.x_original.to_dense(), &transform.materialize_dense());
                build_firth_design_factor_dense(x_t_dense.view(), self.priorweights)?
            }
            WorkingCoordinateDesign::OriginalSparseNative => {
                if self.x_original.as_sparse().is_some() {
                    let csr = self.x_original_csr.as_ref().ok_or_else(|| {
                        EstimationError::InvalidInput(
                            "missing CSR cache for sparse original design".to_string(),
                        )
                    })?;
                    build_firth_design_factor_sparse(csr, self.priorweights)?
                } else {
                    let x_dense = self
                        .x_original
                        .try_to_dense_arc(
                            "Firth diagnostics require dense access to the original design",
                        )
                        .map_err(EstimationError::InvalidInput)?;
                    build_firth_design_factor_dense(x_dense.view(), self.priorweights)?
                }
            }
        };
        let factor = Arc::new(factor);
        self.firth_design_factor = Some(factor.clone());
        Ok(factor)
    }

    /// Convert the working model into its final state for outer REML consumption.
    ///
    /// The `finalweights` field is set to `lasthessian_weights`, which are the
    /// **observed-information** weights (for non-canonical links) or Fisher weights
    /// (for canonical links where observed = Fisher). These flow into the outer
    /// REML H = X'W_obs X + S, ensuring log|H| uses the correct Laplace curvature.
    /// See response.md Section 3 for the mathematical justification.
    pub(crate) fn into_final_state(self) -> GamModelFinalState {
        let GamWorkingModel {
            coordinate_design,
            lastmu,
            lastweights,
            lastz,
            last_c: _,
            last_d: _,
            lasthessian_weights,
            lasthessian_c,
            lasthessian_d,
            last_dmu_deta,
            last_d2mu_deta2,
            last_d3mu_deta3,
            last_penalty_term,
            ..
        } = self;
        let coordinate_frame = match coordinate_design {
            WorkingCoordinateDesign::OriginalSparseNative => {
                PirlsCoordinateFrame::OriginalSparseNative
            }
            WorkingCoordinateDesign::TransformedExplicit { .. } => {
                PirlsCoordinateFrame::TransformedQs
            }
            WorkingCoordinateDesign::TransformedImplicit { .. } => {
                PirlsCoordinateFrame::TransformedQs
            }
        };
        GamModelFinalState {
            likelihood: self.likelihood.clone(),
            coordinate_frame,
            finalmu: lastmu,
            finalweights: lasthessian_weights,
            scoreweights: lastweights,
            finalz: lastz,
            final_c: lasthessian_c,
            final_d: lasthessian_d,
            final_dmu_deta: last_dmu_deta,
            final_d2mu_deta2: last_d2mu_deta2,
            final_d3mu_deta3: last_d3mu_deta3,
            penalty_term: last_penalty_term,
        }
    }

    /// Compute X_transformed * β into a pre-allocated buffer, avoiding
    /// per-iteration allocation in the dense case.
    pub(crate) fn transformed_matvec_into(&self, beta: &Coefficients, out: &mut Array1<f64>) {
        self.transformed_matvec_array_into(beta.as_ref(), out);
    }

    /// View-based sibling of `transformed_matvec_into` that operates on a raw
    /// `&Array1<f64>` to avoid wrapping (and cloning into) `Coefficients` on
    /// hot LM-screen paths.
    pub(crate) fn transformed_matvec_array_into(&self, beta: &Array1<f64>, out: &mut Array1<f64>) {
        match &self.coordinate_design {
            WorkingCoordinateDesign::TransformedExplicit { x_transformed, .. } => {
                if let Some(dense) = x_transformed.as_dense() {
                    fast_av_into(dense, beta, out);
                    return;
                }
                out.assign(&x_transformed.matrixvectormultiply(beta));
            }
            WorkingCoordinateDesign::TransformedImplicit { transform } => {
                // Composed: X · (Qs · beta).  Qs·beta is p-dim (cheap),
                // then write X·(Qs·beta) directly into out when X is dense.
                let beta_orig = transform.apply(beta);
                if let Some(dense) = self.x_original.as_dense() {
                    fast_av_into(dense, &beta_orig, out);
                } else {
                    out.assign(&self.x_original.apply(&beta_orig));
                }
            }
            WorkingCoordinateDesign::OriginalSparseNative => {
                out.assign(&self.x_original.matrixvectormultiply(beta));
            }
        }
    }

    pub(crate) fn transformed_transpose_matvec(&self, vec: &Array1<f64>) -> Array1<f64> {
        match &self.coordinate_design {
            WorkingCoordinateDesign::OriginalSparseNative => {
                self.x_original.transpose_vector_multiply(vec)
            }
            WorkingCoordinateDesign::TransformedExplicit { x_transformed, .. } => {
                x_transformed.transpose_vector_multiply(vec)
            }
            WorkingCoordinateDesign::TransformedImplicit { transform } => {
                let xtv = self.x_original.transpose_vector_multiply(vec);
                transform.apply_transpose(&xtv)
            }
        }
    }

    /// Compute X^T W X via the shared dense assembly path.
    /// Falls back to the scalar loop for sparse matrices.
    pub(crate) fn compute_xtwx_blas(
        workspace: &mut PirlsWorkspace,
        design: &DesignMatrix,
        weights: &Array1<f64>,
    ) -> Result<Array2<f64>, EstimationError> {
        match design {
            // Only the materialized arm can use the shared dense assembly path.
            // Lazy operator-backed dense designs (TPS/Matern at large scale)
            // cannot be densified; fall through to the operator XᵀWX path.
            DesignMatrix::Dense(x) if x.is_materialized_dense() => {
                let p = x.ncols();
                let x_dense = x.to_dense_arc();
                // Reuse workspace hessian buffer to avoid per-iteration allocation.
                if workspace.hessian_buf.nrows() != p || workspace.hessian_buf.ncols() != p {
                    workspace.hessian_buf = Array2::zeros((p, p).f());
                } else {
                    workspace.hessian_buf.fill(0.0);
                }
                if gam_gpu::cuda_selected() {
                    // #1412: keep the n×p design `X` device-resident across the
                    // inner P-IRLS iterates. The Gram is rebuilt once per
                    // Newton/LM iterate with the SAME `X` (only `w` moves), so
                    // re-uploading the full `X` on every iterate starves the
                    // device on H2D staging. Cache the resident `X` keyed on its
                    // host data pointer + shape: the first iterate uploads `X`,
                    // every later iterate crosses only `w` (n doubles) H2D and
                    // the p×p Gram D2H. The resident `gram` is bit-identical to
                    // the per-call `weighted_crossprod_gpu` on the same device
                    // (same column-major `X`, same `cublasDdgmm` row-scale, same
                    // `gemm` reduction order). If residency declines (CUDA
                    // unavailable / below the GPU Gram threshold / upload
                    // failure) keep the per-call path.
                    let key = (x_dense.as_ptr() as usize, x_dense.nrows(), p);
                    let cache_hit = matches!(
                        &workspace.resident_design_gram,
                        Some((k0, k1, k2, _)) if (*k0, *k1, *k2) == key
                    );
                    if !cache_hit {
                        workspace.resident_design_gram =
                            gam_gpu::linalg_dispatch::ResidentDesignGram::try_new(x_dense.view())
                                .map(|g| (key.0, key.1, key.2, g));
                    }
                    if let Some((_, _, _, gram)) = workspace.resident_design_gram.as_ref() {
                        if let Some(h) = gram.gram(weights.view()) {
                            return Ok(h);
                        }
                    }
                    return crate::gpu::pirls_gpu::weighted_crossprod_gpu(
                        x_dense.view(),
                        weights.view(),
                    )
                    .map_err(EstimationError::InvalidInput);
                }
                gam_gpu::log_backend_inventory_once();
                // DenseXtWX has no compiled vendor backend on this path; the
                // workload-size predicate is computed only for diagnostic
                // logging via the `decide` reason channel.
                let gpu_decision = gam_gpu::decide(
                    gam_gpu::GpuKernel::DenseXtWX,
                    gam_gpu::GpuEligibility::BackendNotCompiled,
                );
                gpu_decision
                    .require_supported()
                    .map_err(EstimationError::InvalidInput)?;
                gpu_decision.log();
                if weights.iter().any(|&w| w < 0.0) {
                    // Observed-information assembly may have signed row
                    // weights.  Use Xᵀ(WX) exactly; never sqrt/clip.
                    PirlsWorkspace::add_dense_xtwx_signed(
                        weights,
                        &mut workspace.weighted_x_chunk,
                        x_dense.as_ref(),
                        &mut workspace.hessian_buf,
                    );
                } else {
                    // All weights are non-negative; the shared dense helper
                    // computes Xᵀ·diag(w)·X directly without sqrt/clip.
                    PirlsWorkspace::add_dense_xtwx_signed(
                        weights,
                        &mut workspace.weighted_x_chunk,
                        x_dense.as_ref(),
                        &mut workspace.hessian_buf,
                    );
                }
                // Move the buffer out instead of cloning — saves O(p²) memcpy.
                // Next call will reallocate (same cost as the existing zero-fill).
                Ok(std::mem::take(&mut workspace.hessian_buf))
            }
            // Observed-Hessian assembly: working weights may be signed
            // (binomial + cloglog, Gamma + identity, etc.). Route through the
            // signed-Gram API so the CSC / sparse-accumulator paths preserve
            // sign instead of silently clipping negative-curvature mass.
            _ => gam_linalg::matrix::xt_diag_x_signed(
                design,
                gam_linalg::matrix::FiniteSignedWeightsView::try_from_array(weights)
                    .map_err(EstimationError::InvalidInput)?,
            )
            .map(|h| h.to_dense())
            .map_err(EstimationError::InvalidInput),
        }
    }

    pub(crate) fn penalized_hessian(
        &mut self,
        weights: &Array1<f64>,
    ) -> Result<Array2<f64>, EstimationError> {
        // #1111 / #1033 mechanism (c): the frozen-weight first-Fisher-step Gram
        // `XᵀWX` (in the original / `x_fit` conditioned frame) serves the FIRST
        // Fisher-scoring iteration n-free, eliding the dominant O(N·p²) weighted
        // cross-product on a large-n GLM ψ-trial. It is only correct for the
        // first build at the warm β with FISHER curvature (the frozen tensor was
        // assembled from the canonical Fisher weights), and only in the two
        // original-frame coordinate designs (TransformedImplicit conjugates the
        // original-frame Gram afterward; OriginalSparseNative is already in that
        // frame). For TransformedExplicit the streamed Gram lives in the Qs frame
        // the tensor was not built in, so that variant always restreams. Every
        // later iteration restreams the true (moving) `W`, so the converged β̂ is
        // unchanged — only the first Gram build is skipped.
        let use_frozen_first_step = !self.glm_first_step_gram_consumed
            && self.glm_first_step_gram.is_some()
            && self.lasthessian_curvature == HessianCurvatureKind::Fisher
            && !matches!(
                self.coordinate_design,
                WorkingCoordinateDesign::TransformedExplicit { .. }
            );
        if use_frozen_first_step {
            // Take the cached original-frame Gram exactly once.
            let xtwx = self
                .glm_first_step_gram
                .take()
                .expect("frozen first-step Gram present by the guard above");
            self.glm_first_step_gram_consumed = true;
            log::debug!(
                "[frozen-glm-gram] serving first Fisher-step XᵀWX n-free (p={})",
                xtwx.nrows()
            );
            return match &self.coordinate_design {
                WorkingCoordinateDesign::TransformedImplicit { transform } => {
                    let mut h = transform.conjugate_matrix(&xtwx);
                    self.penalty.add_to_hessian(&mut h);
                    Ok(h)
                }
                WorkingCoordinateDesign::OriginalSparseNative => {
                    let mut h = xtwx;
                    self.penalty.add_to_hessian(&mut h);
                    Ok(h)
                }
                WorkingCoordinateDesign::TransformedExplicit { .. } => {
                    // Excluded from `use_frozen_first_step` by the guard above
                    // (the frozen Gram lives in the original frame the explicit
                    // transform was not built in). A clean error rather than a
                    // panic if a future refactor ever lets this state through.
                    Err(EstimationError::InvalidInput(
                        "frozen first-step Gram path reached with TransformedExplicit \
                         coordinate design, which the gate excludes"
                            .to_string(),
                    ))
                }
            };
        }
        match &self.coordinate_design {
            WorkingCoordinateDesign::TransformedExplicit { x_transformed, .. } => {
                let mut h = Self::compute_xtwx_blas(&mut self.workspace, x_transformed, weights)?;
                self.penalty.add_to_hessian(&mut h);
                Ok(h)
            }
            WorkingCoordinateDesign::TransformedImplicit { transform } => {
                let xtwx = Self::compute_xtwx_blas(&mut self.workspace, &self.x_original, weights)?;
                let mut h = transform.conjugate_matrix(&xtwx);
                self.penalty.add_to_hessian(&mut h);
                Ok(h)
            }
            WorkingCoordinateDesign::OriginalSparseNative => {
                let mut h =
                    Self::compute_xtwx_blas(&mut self.workspace, &self.x_original, weights)?;
                self.penalty.add_to_hessian(&mut h);
                Ok(h)
            }
        }
    }

    pub(crate) fn supports_observed_hessian_curvature(&self) -> bool {
        supports_observed_hessian_curvature_for_likelihood(&self.likelihood, &self.link_kind)
    }

    /// Compute the Hessian-side weight arrays (w, c, d) for the requested curvature kind.
    ///
    /// When `requested == Observed` and the link supports it, returns the
    /// **observed-information** weights including the residual-dependent correction:
    ///   W_obs = W_Fisher - (y - mu) * B,  B = (h'' V - h'^2 V') / (phi V^2)
    ///   c_obs = c_Fisher + h'*B - (y-mu)*B_eta
    ///   d_obs = d_Fisher + h''*B + 2*h'*B_eta - (y-mu)*B_etaeta
    ///
    /// For canonical links (for example logit-Binomial and log-Poisson), B = 0
    /// so observed = Fisher. Gamma-log is non-canonical and therefore needs its
    /// own observed-information correction.
    ///
    /// These arrays serve dual purpose:
    /// 1. **Inner iteration**: They define the Newton system H*delta = -g.
    ///    Fisher scoring (using W_Fisher) is also valid here since any convergent
    ///    algorithm finds the same mode.
    /// 2. **Outer REML**: They define the Laplace Hessian H_obs = X'W_obs X + S.
    ///    The outer log|H| and trace terms MUST use observed information for the
    ///    exact Laplace approximation. See response.md Section 3.
    pub(crate) fn update_hessian_curvature_arrays(
        &mut self,
        requested: HessianCurvatureKind,
    ) -> Result<HessianCurvatureKind, EstimationError> {
        if requested == HessianCurvatureKind::Fisher || !self.supports_observed_hessian_curvature()
        {
            self.lasthessian_weights.assign(&self.lastweights);
            self.lasthessian_c.assign(&self.last_c);
            self.lasthessian_d.assign(&self.last_d);
            return Ok(HessianCurvatureKind::Fisher);
        }

        compute_observed_hessian_curvature_arrays_into(
            &self.likelihood,
            &self.link_kind,
            &self.workspace.eta_buf,
            self.y,
            &self.lastweights,
            self.priorweights,
            &mut self.lasthessian_weights,
            &mut self.lasthessian_c,
            &mut self.lasthessian_d,
        )?;
        Ok(HessianCurvatureKind::Observed)
    }

    pub(crate) fn sparse_penalized_hessian(
        &mut self,
        weights: &Array1<f64>,
        ridge: f64,
    ) -> Result<SparseColMat<usize, f64>, EstimationError> {
        let x_sparse = self.x_original.as_sparse().ok_or_else(|| {
            EstimationError::InvalidInput(
                "sparse-native PIRLS requires a sparse original design".to_string(),
            )
        })?;
        let PirlsPenalty::Dense { s_transformed, .. } = &self.penalty else {
            crate::bail_invalid_estim!(
                "sparse-native PIRLS requires a dense transformed penalty matrix"
            );
        };
        self.workspace.assemble_sparse_penalized_hessian(
            x_sparse,
            weights,
            s_transformed,
            ridge,
            None,
        )
    }

    /// LM-screen helper: evaluates a candidate β by reusing the previous
    /// `current_eta` plus a single design-matrix matvec `X·δ`, then runs the
    /// inverse-link only far enough to recover μ, w, z and the deviance.
    /// No Hessian assembly, no derivative buffers, no Jeffreys logdet.
    ///
    /// The LM loop calls `update_with_curvature` to upgrade the screen to a
    /// full `WorkingState` only when the screen is accepted. Rejected LM
    /// candidates therefore skip the O(np²) curvature build entirely.
    pub(crate) fn screen_candidate_from_direction(
        &mut self,
        beta: &Coefficients,
        direction: &Array1<f64>,
        current_eta: &LinearPredictor,
    ) -> Result<CandidateScreen, EstimationError> {
        let n = self.offset.len();
        if self.workspace.eta_buf.len() != n {
            self.workspace.eta_buf = Array1::zeros(n);
        }
        if self.workspace.delta_eta.len() != n {
            self.workspace.delta_eta = Array1::zeros(n);
        }

        // Compute δη = X·direction once into the workspace, then assemble
        // η_cand = η_current + δη in parallel.
        let mut delta_eta = std::mem::take(&mut self.workspace.delta_eta);
        // Avoid wrapping/cloning `direction` into a `Coefficients` newtype just
        // to satisfy the &Coefficients overload — the view-based sibling
        // performs the identical matvec without the per-LM-attempt clone.
        self.transformed_matvec_array_into(direction, &mut delta_eta);
        Zip::from(&mut self.workspace.eta_buf)
            .and(current_eta.as_ref())
            .and(&delta_eta)
            .par_for_each(|eta, &base, &d| *eta = base + d);
        self.workspace.delta_eta = delta_eta;

        // NB: the Gamma dispersion shape is deliberately NOT re-estimated here.
        // This screen only evaluates a *trial* β to feed the LM gain-ratio
        // accept/reject test, whose predicted reduction comes from the gradient
        // and Hessian built (at the current shape) by the last accepted
        // `update_with_curvature`. Re-estimating the shape per trial — and per
        // halving attempt — silently changes the objective the screen reports
        // (deviance = 2·shape·Σ wᵢ dᵢ) relative to that predicted reduction, so
        // the gain ratio compares two different objectives, every step is
        // rejected, λ_LM runs to its ceiling, and the inner solve stalls with a
        // large residual gradient ("LM step search exhausted"). The shape is a
        // nuisance scale that must stay fixed within an inner Newton/LM step; it
        // is updated once per *accepted* iterate in `update_with_curvature`
        // (block-coordinate β | shape), exactly as mgcv holds the scale fixed
        // through the inner P-IRLS solve. See issue #511 (regression of #359).
        let integrated = self.covariate_se.as_ref().map(|se| IntegratedWorkingInput {
            quadctx: &self.quadctx,
            se: se.view(),
            mixture_link_state: self.link_kind.mixture_state(),
            sas_link_state: self.link_kind.sas_state(),
        });
        match &self.link_kind {
            InverseLink::Mixture(_)
            | InverseLink::LatentCLogLog(_)
            | InverseLink::Sas(_)
            | InverseLink::BetaLogistic(_) => {
                if let Some(integ) = integrated {
                    update_glmvectors_integrated_for_link(
                        integ.quadctx,
                        self.y,
                        &self.workspace.eta_buf,
                        integ.se,
                        &self.link_kind,
                        self.priorweights,
                        &mut self.lastmu,
                        &mut self.lastweights,
                        &mut self.lastz,
                        None,
                    )?;
                } else {
                    update_glmvectors(
                        self.y,
                        &self.workspace.eta_buf,
                        &self.link_kind,
                        self.priorweights,
                        &mut self.lastmu,
                        &mut self.lastweights,
                        &mut self.lastz,
                        None,
                    )?;
                }
            }
            InverseLink::Standard(_) => {
                self.likelihood.irls_update(
                    self.y,
                    &self.workspace.eta_buf,
                    self.priorweights,
                    &mut self.lastmu,
                    &mut self.lastweights,
                    &mut self.lastz,
                    integrated,
                    None,
                )?;
            }
        }

        let deviance = self.likelihood.loglik_deviance(
            self.y,
            &self.workspace.eta_buf,
            &self.lastmu,
            &self.link_kind,
            self.priorweights,
        )?;
        let penalty_term = self.penalty.shifted_quadratic(beta.as_ref());
        // Finiteness is a property of the (deviance, penalty) pair regardless of
        // the family dispersion scale `k` applied later in the gain ratio, so the
        // arithmetic screen uses the bare, unscaled `deviance + penalty_term`.
        let arithmetic_finite = (deviance + penalty_term).is_finite()
            && self.workspace.eta_buf.iter().all(|v| v.is_finite())
            && self.lastmu.iter().all(|v| v.is_finite())
            && self.lastweights.iter().all(|v| v.is_finite());
        Ok(CandidateScreen {
            deviance,
            penalty_term,
            arithmetic_finite,
        })
    }
}

impl<'a> WorkingModel for GamWorkingModel<'a> {
    fn update(&mut self, beta: &Coefficients) -> Result<WorkingState, EstimationError> {
        self.update_with_curvature(beta, HessianCurvatureKind::Fisher)
    }

    fn penalized_deviance_scale(&self) -> f64 {
        // Matches the constant dispersion factor `write_*_working_state` bakes
        // into `self.lastweights` (Gamma `·shape`, Tweedie/fixed-φ Gaussian
        // `/φ`), reading the SAME `self.likelihood` the weights are built from,
        // so the gain-ratio objective `k·D + penalty` is exactly consistent with
        // the k-scaled gradient/Hessian. For a Gamma smooth this is the locked
        // shape refreshed once per inner solve (see `gamma_shape_locked`).
        super::curvature::penalized_objective_deviance_scale(&self.likelihood)
    }

    fn update_with_curvature(
        &mut self,
        beta: &Coefficients,
        requested_curvature: HessianCurvatureKind,
    ) -> Result<WorkingState, EstimationError> {
        let n = self.offset.len();
        if self.workspace.eta_buf.len() != n {
            self.workspace.eta_buf = Array1::zeros(n);
        }
        if self.workspace.matvec_buf.len() != n {
            self.workspace.matvec_buf = Array1::zeros(n);
        }
        let mut matvec_tmp = std::mem::take(&mut self.workspace.matvec_buf);
        self.transformed_matvec_into(beta, &mut matvec_tmp);
        self.workspace.eta_buf.assign(&self.offset);
        self.workspace.eta_buf += &matvec_tmp;
        self.workspace.matvec_buf = matvec_tmp;

        // Estimate the Gamma dispersion shape once from the warm-start η and
        // freeze it for the remainder of this inner solve. Holding the shape
        // fixed keeps the product φ·λ constant, so the penalized argmin β̂ is a
        // stationary target and the LM gain ratio stays consistent across trial
        // and accepted iterates. The shape refreshes across outer iterations
        // because a fresh model is built per inner solve. See issue #511.
        if self.likelihood.scale.gamma_shape_is_estimated() && !self.gamma_shape_locked {
            let shape =
                estimate_gamma_shape_from_eta(self.y, &self.workspace.eta_buf, self.priorweights)?;
            self.likelihood = self.likelihood.clone().with_gamma_shape(shape);
            self.gamma_shape_locked = true;
        }

        // Estimate the Beta precision φ once from the warm-start η and freeze it
        // for this inner solve (issue #567). φ enters the IRLS weights and the
        // variance `Var(y)=mu(1-mu)/(1+φ)`; holding it fixed within the inner
        // solve keeps the penalized argmin β̂ stationary (mirroring the Gamma
        // shape lock above), and it refreshes across outer iterations as a fresh
        // working model is built per inner solve. With φ pinned at the seed of 1
        // the mean smooth was over-penalized / under-fit on precise data.
        if self.likelihood.scale.beta_phi_is_estimated() && !self.beta_phi_locked {
            let phi =
                estimate_beta_phi_from_eta(self.y, &self.workspace.eta_buf, self.priorweights)?;
            self.likelihood = self.likelihood.clone().with_beta_phi(phi);
            self.beta_phi_locked = true;
        }

        // Estimate the Tweedie dispersion φ once from the warm-start η and freeze
        // it for this inner solve (issue #771). φ enters the IRLS weight
        // `prior·μ^{2−p}/φ` (and so the covariance Vb = H⁻¹, giving SE ∝ √φ);
        // holding it fixed within the inner solve keeps the product φ·λ — hence
        // the penalized argmin β̂ — a stationary LM target (mirroring the Gamma
        // shape and Beta φ locks above), and it refreshes across outer iterations
        // as a fresh working model is built per inner solve.
        if self.likelihood.scale.tweedie_phi_is_estimated() && !self.tweedie_phi_locked {
            if let ResponseFamily::Tweedie { p } = self.likelihood.spec.response {
                let phi = estimate_tweedie_phi_from_eta(
                    self.y,
                    &self.workspace.eta_buf,
                    self.priorweights,
                    p,
                )?;
                self.likelihood = self.likelihood.clone().with_tweedie_phi(phi);
                self.tweedie_phi_locked = true;
            }
        }

        // Estimate the Negative-Binomial overdispersion `theta` once from the
        // warm-start η and freeze it for this inner solve (issue #802). `theta`
        // enters the working weight `W = μθ/(θ+μ)` (the NB2 Fisher information)
        // and the working response, so holding it fixed within the inner solve
        // keeps the penalized argmin β̂ a stationary LM target (mirroring the Beta
        // φ lock above); it refreshes across outer iterations as a fresh working
        // model is built per inner solve. With `theta` frozen at the seed every
        // coefficient/η SE ignored the data's overdispersion.
        if self.likelihood.scale.negbin_theta_is_estimated() && !self.negbin_theta_locked {
            let theta =
                estimate_negbin_theta_from_eta(self.y, &self.workspace.eta_buf, self.priorweights)?;
            self.likelihood = self.likelihood.clone().with_negbin_theta(theta);
            self.negbin_theta_locked = true;
        }

        // Use integrated (GHQ) likelihood if per-observation SE is available.
        // This coherently accounts for uncertainty in the base prediction.
        let integrated = self.covariate_se.as_ref().map(|se| IntegratedWorkingInput {
            quadctx: &self.quadctx,
            se: se.view(),
            mixture_link_state: self.link_kind.mixture_state(),
            sas_link_state: self.link_kind.sas_state(),
        });
        match &self.link_kind {
            InverseLink::Mixture(_) => {
                if let Some(integ) = integrated {
                    update_glmvectors_integrated_for_link(
                        integ.quadctx,
                        self.y,
                        &self.workspace.eta_buf,
                        integ.se,
                        &self.link_kind,
                        self.priorweights,
                        &mut self.lastmu,
                        &mut self.lastweights,
                        &mut self.lastz,
                        Some(WorkingDerivativeBuffersMut {
                            c: &mut self.last_c,
                            d: &mut self.last_d,
                            dmu_deta: &mut self.last_dmu_deta,
                            d2mu_deta2: &mut self.last_d2mu_deta2,
                            d3mu_deta3: &mut self.last_d3mu_deta3,
                        }),
                    )?;
                } else {
                    update_glmvectors(
                        self.y,
                        &self.workspace.eta_buf,
                        &self.link_kind,
                        self.priorweights,
                        &mut self.lastmu,
                        &mut self.lastweights,
                        &mut self.lastz,
                        Some(WorkingDerivativeBuffersMut {
                            c: &mut self.last_c,
                            d: &mut self.last_d,
                            dmu_deta: &mut self.last_dmu_deta,
                            d2mu_deta2: &mut self.last_d2mu_deta2,
                            d3mu_deta3: &mut self.last_d3mu_deta3,
                        }),
                    )?;
                }
            }
            InverseLink::LatentCLogLog(_) | InverseLink::Sas(_) | InverseLink::BetaLogistic(_) => {
                if let Some(integ) = integrated {
                    update_glmvectors_integrated_for_link(
                        integ.quadctx,
                        self.y,
                        &self.workspace.eta_buf,
                        integ.se,
                        &self.link_kind,
                        self.priorweights,
                        &mut self.lastmu,
                        &mut self.lastweights,
                        &mut self.lastz,
                        Some(WorkingDerivativeBuffersMut {
                            c: &mut self.last_c,
                            d: &mut self.last_d,
                            dmu_deta: &mut self.last_dmu_deta,
                            d2mu_deta2: &mut self.last_d2mu_deta2,
                            d3mu_deta3: &mut self.last_d3mu_deta3,
                        }),
                    )?;
                } else {
                    update_glmvectors(
                        self.y,
                        &self.workspace.eta_buf,
                        &self.link_kind,
                        self.priorweights,
                        &mut self.lastmu,
                        &mut self.lastweights,
                        &mut self.lastz,
                        Some(WorkingDerivativeBuffersMut {
                            c: &mut self.last_c,
                            d: &mut self.last_d,
                            dmu_deta: &mut self.last_dmu_deta,
                            d2mu_deta2: &mut self.last_d2mu_deta2,
                            d3mu_deta3: &mut self.last_d3mu_deta3,
                        }),
                    )?;
                }
            }
            InverseLink::Standard(_) => {
                self.likelihood.irls_update(
                    self.y,
                    &self.workspace.eta_buf,
                    self.priorweights,
                    &mut self.lastmu,
                    &mut self.lastweights,
                    &mut self.lastz,
                    integrated,
                    Some(WorkingDerivativeBuffersMut {
                        c: &mut self.last_c,
                        d: &mut self.last_d,
                        dmu_deta: &mut self.last_dmu_deta,
                        d2mu_deta2: &mut self.last_d2mu_deta2,
                        d3mu_deta3: &mut self.last_d3mu_deta3,
                    }),
                )?;
            }
        }
        let mut firth = FirthDiagnostics::Inactive;
        if self.firth_bias_reduction {
            if !self.link_kind.has_fisher_weight_jet() {
                crate::bail_invalid_estim!(
                    "Firth/Jeffreys PIRLS requested for unsupported inverse link {:?}",
                    self.link_kind
                );
            }
            // IMPORTANT: Jeffreys/Firth bias reduction must be computed in the
            // *same coefficient basis* as the inner objective being optimized by PIRLS.
            //
            // The working response (z) and the coefficients β are in the transformed
            // basis when a reparameterization is used. The Jeffreys term is the
            // identifiable-subspace Fisher logdet evaluated on a canonical
            // orthonormal basis of the transformed design column space,
            // not a raw-coordinate logdet. Its PIRLS hat-diagonal adjustment must
            // therefore be computed from that same transformed-design Fisher
            // matrix, otherwise the inner objective and the outer LAML
            // derivatives disagree.
            //
            // This mismatch is subtle but severe: it leaves the analytic gradient
            // differentiating a *different* objective than the one PIRLS actually
            // solved, and the gradient check fails catastrophically.
            //
            // Rule: use X_transformed if available; fall back to X_original only
            // when PIRLS is operating directly in the original basis.
            //
            // #1575: the design and prior weights are constant across the inner
            // Newton iterations of this solve, so the β-independent Firth design
            // factor (Gram, identifiable basis Q, reduced design X_r, retained
            // spectrum S_r) is built once and memoized; only the cheap per-η
            // reduced-Fisher/hat-diagonal remainder is rebuilt here. The factor
            // is built in the correct (transformed) coefficient basis exactly as
            // the per-iteration diagnostics path used to.
            let factor = self.ensure_firth_design_factor()?;
            let (hat_diag, jeffreys_logdet, firth_score_shift) =
                jeffreys_pirls_diagnostics_from_factor(
                    &factor,
                    &self.link_kind,
                    self.workspace.eta_buf.view(),
                )?;
            firth = FirthDiagnostics::Active {
                jeffreys_logdet,
                hat_diag: hat_diag.clone(),
            };
            // Apply the link-general Firth working-response shift `Δ_i` built by
            // the operator (`½ (w'_i/w_i) h_diag_i`). PIRLS then solves
            // `Xᵀ W (z* − η) = 0`, so the Firth term it adds to the score is
            // `Σ_i w_i Δ_i x_i = ½ Σ_i w'_i h_diag_i x_i = ∂Φ/∂β` — exactly the
            // Jeffreys score the outer REML differentiates. For the canonical
            // logit `Δ_i` equals the historical `h_i (½ − μ_i)/w_i`; for probit /
            // cloglog it carries the correct non-canonical `w'_i/w_i` instead of
            // the logit-pinned `(½ − μ_i)`, so the inner mode and the outer
            // objective no longer disagree.
            ndarray::Zip::from(&mut self.lastz)
                .and(&firth_score_shift)
                .and(&self.lastweights)
                .par_for_each(|zi, &delta_i, &wi| {
                    if wi > 0.0 {
                        *zi += delta_i;
                    }
                });
        }

        let z = &self.lastz;
        // Fused single-pass: compute weighted_residual = (eta - z) * w
        // and working_residual = eta - z simultaneously, avoiding two
        // separate O(n) passes and an intermediate copy.
        ndarray::Zip::from(&mut self.workspace.weighted_residual)
            .and(&mut self.workspace.working_residual)
            .and(&self.workspace.eta_buf)
            .and(z)
            .and(&self.lastweights)
            .par_for_each(|wr, r, &eta, &zi, &wi| {
                let residual = eta - zi;
                *r = residual;
                *wr = residual * wi;
            });
        let mut gradient = self.transformed_transpose_matvec(&self.workspace.weighted_residual);
        // Score norm ||X' (weighted residual)||_2 — captured before adding the
        // penalty contribution so the natural gradient scale can be assembled
        // for the scale-invariant convergence certificate.
        let score_norm = array1_l2_norm(&gradient);
        let s_beta = self.penalty.shifted_gradient(beta.as_ref());
        let s_beta_norm = array1_l2_norm(&s_beta);
        gradient += &s_beta;
        let hessian_curvature = self.update_hessian_curvature_arrays(requested_curvature)?;
        self.lasthessian_curvature = hessian_curvature;

        // Assemble the exact signed statistical Hessian.  Positive-definiteness
        // stabilization is applied only after X'WX + S has been assembled,
        // through the explicit matrix ridge below; changing individual row
        // weights would define a different likelihood surface.
        if self.workspace.matvec_buf.len() != n {
            self.workspace.matvec_buf = Array1::zeros(n);
        }
        self.workspace.matvec_buf.assign(&self.lasthessian_weights);
        let solver_weights = std::mem::take(&mut self.workspace.matvec_buf);

        let (penalized_hessian, sparsehessian, ridge_used) = if matches!(
            self.coordinate_design,
            WorkingCoordinateDesign::OriginalSparseNative
        ) {
            // The SPD-check factor is discarded here: the downstream consumer
            // is the LM Newton step, which always factorizes
            // (H + loop_lambda · I) with a non-zero loop_lambda (initial value
            // 1e-6), so it sees a different matrix.
            let (h_sparse, _factor, ridge_used) =
                ensure_sparse_positive_definitewithridge(|ridge| {
                    self.sparse_penalized_hessian(&solver_weights, ridge)
                })?;
            (Array2::zeros((0, 0)), Some(h_sparse), ridge_used)
        } else {
            let mut penalized_hessian = self.penalized_hessian(&solver_weights)?;
            assert_symmetric_tol(&penalized_hessian, "PIRLS penalized Hessian", 1e-8);
            let ridge_used = ensure_positive_definitewithridge(
                &mut penalized_hessian,
                "PIRLS penalized Hessian",
            )?;
            (penalized_hessian, None, ridge_used)
        };
        self.workspace.matvec_buf = solver_weights;

        // Match the stabilized Hessian used by the outer LAML objective.
        // If a ridge is needed, we treat it as an explicit penalty term:
        //
        //   l_p(β; ρ) = l(β) - 0.5 * βᵀ S_λ β - 0.5 * ridge * ||β||²
        //
        // This keeps the PIRLS fixed point aligned with the stabilized Hessian
        // that drives log|H| and the implicit-gradient correction.
        let deviance = self.likelihood.loglik_deviance(
            self.y,
            &self.workspace.eta_buf,
            &self.lastmu,
            &self.link_kind,
            self.priorweights,
        )?;
        let log_likelihood = calculate_loglikelihood_omitting_constants_from_eta(
            self.y,
            &self.workspace.eta_buf,
            &self.likelihood,
            &self.link_kind,
            self.priorweights,
        )?;

        let mut penalty_term = self.penalty.shifted_quadratic(beta.as_ref());
        let mut ridge_grad_norm = 0.0;
        if ridge_used > 0.0 {
            let ridge_penalty = ridge_used * beta.as_ref().dot(beta.as_ref());
            penalty_term += ridge_penalty;
            gradient.zip_mut_with(beta.as_ref(), |g, &b| *g += ridge_used * b);
            ridge_grad_norm = ridge_used * array1_l2_norm(beta.as_ref());
        }

        self.last_penalty_term = penalty_term;
        let gradient_natural_scale = score_norm + s_beta_norm + ridge_grad_norm;

        Ok(WorkingState {
            eta: LinearPredictor::new(std::mem::replace(
                &mut self.workspace.eta_buf,
                Array1::zeros(0),
            )),
            gradient,
            hessian: match sparsehessian {
                Some(h_sparse) => gam_linalg::matrix::SymmetricMatrix::Sparse(h_sparse),
                None => gam_linalg::matrix::SymmetricMatrix::Dense(penalized_hessian),
            },

            log_likelihood,
            deviance,
            penalty_term,
            firth,
            ridge_used,
            hessian_curvature,
            gradient_natural_scale,
        })
    }

    fn update_candidate(
        &mut self,
        beta: &Coefficients,
        curvature: HessianCurvatureKind,
    ) -> Result<WorkingState, EstimationError> {
        // The LM line-search candidate MUST be built with the SAME objective the
        // accepted state and `current_penalized` use — i.e. with Firth active
        // when `firth_bias_reduction` is set. Previously this method transiently
        // disabled Firth while building the candidate, so the candidate's
        // `WorkingState.firth` came back `Inactive` and
        // `CandidateEvaluation::penalized_objective` dropped the `−2·½log|XᵀWX|`
        // Jeffreys term for the candidate while `current_penalized` (built with
        // Firth) kept it. The line search then compared a Firth objective against
        // a non-Firth one, and — because the accepted state IS the candidate
        // state (`final_state = accepted_state`) and convergence is certified on
        // `accepted_state.gradient` — the inner solve converged on the ordinary
        // penalized-MLE stationarity `∇(−ℓ+½βᵀSβ)=0` instead of the
        // Firth-penalized stationarity `∇(−ℓ+½βᵀSβ)−∇Φ=0`. The returned β̂ then
        // sat at the WRONG mode, breaking the outer LAML envelope identity
        // (the dense path carries no KKT-residual correction), so the analytic
        // smoothing-selection gradient disagreed with the finite difference of
        // the cost for every Firth fit routed through the LM line search
        // (gam#1821). Keep Firth active for the candidate so the whole line
        // search optimizes one coherent Firth-penalized objective.
        self.update_with_curvature(beta, curvature)
    }

    fn screen_candidate(
        &mut self,
        beta: &Coefficients,
        direction: &Array1<f64>,
        current_eta: &LinearPredictor,
        curvature: HessianCurvatureKind,
    ) -> Result<CandidateEvaluation, EstimationError> {
        if self.firth_bias_reduction {
            return self
                .update_candidate(beta, curvature)
                .map(CandidateEvaluation::Full);
        }
        self.screen_candidate_from_direction(beta, direction, current_eta)
            .map(CandidateEvaluation::Screen)
    }

    fn supports_observed_information_curvature(&self) -> bool {
        self.supports_observed_hessian_curvature()
    }
}
