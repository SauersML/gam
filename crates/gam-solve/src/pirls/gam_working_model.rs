//! The concrete GLM `WorkingModel`: `GamWorkingModel` assembles the working
//! response/weights, the penalized Hessian, and the curvature arrays, and
//! implements the `WorkingModel` trait (update / candidate-screen). Carries the
//! `GamModelFinalState` snapshot.
//!
//! No stabilization ridge is added to the penalized Hessian (#2901 V22, SPEC
//! rules 5 and 23): H is exactly `XᵀWX + S_λ`, so the REML criterion carries no
//! coefficient-space ridge and no magic constant.

use super::*;
// `Unbind::unbound()` maps a faer bound sparse column index back to `usize`
// for dense-matrix indexing (see also newton_solve.rs). Imported directly at
// the call site rather than via the pirls prelude re-export (#2306/build).
use faer::Unbind;

fn augmented_root_represents_working_system(
    curvature: HessianCurvatureKind,
    firth_bias_reduction: bool,
    stiff_penalty: bool,
    hessian_weights: &Array1<f64>,
) -> bool {
    curvature == HessianCurvatureKind::Fisher
        && (firth_bias_reduction || stiff_penalty)
        && hessian_weights
            .iter()
            .all(|&weight| weight.is_finite() && weight >= 0.0)
}

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
    /// Whether the Gaussian (non-identity link) or inverse Gaussian dispersion
    /// `phi` has been estimated and frozen for this inner P-IRLS solve. `phi`
    /// scales the whole data term (`W ∝ prior/φ`, score `∝ 1/φ`), so, like the
    /// Tweedie `phi`, it is estimated once from the warm-start η and held fixed
    /// within the inner solve so the product `φ·λ` stays a stationary target.
    pub(crate) dispersion_phi_locked: bool,
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
    /// Exact `HΦ = ∇²β Φ` for the same state as the mutable Firth working
    /// arrays.  The inner objective curvature is `H₀ - HΦ`; keeping this beside
    /// the row-space score operands prevents the inner and outer Jeffreys
    /// geometries from diverging.
    pub(crate) last_firth_hessian: Option<Array2<f64>>,
    /// Exact coefficient bits for the state represented by the mutable working
    /// arrays (`lastz`, `lasthessian_weights`, and their derivative siblings).
    ///
    /// A Firth candidate screen is a full state evaluation.  Rejected LM
    /// candidates therefore leave these scratch arrays at the rejected point
    /// while the loop's authoritative [`WorkingState`] remains at the current
    /// coefficient vector.  Dense Newton solves consume only `WorkingState`,
    /// but the cancellation-safe square-root solve consumes these row-space
    /// arrays too.  The key makes that otherwise-hidden split state explicit so
    /// the root operands can be refreshed before use.  Exact bits are required:
    /// a hash collision cannot be allowed to select another state's Newton
    /// system.
    pub(crate) working_array_beta_bits: Vec<u64>,
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
    fn working_arrays_match_state(
        &self,
        beta: &Coefficients,
        state: &WorkingState,
    ) -> bool {
        self.lasthessian_curvature == state.hessian_curvature
            && self
                .working_array_beta_bits
                .iter()
                .copied()
                .eq(beta.as_ref().iter().map(|value| value.to_bits()))
    }

    pub(crate) fn refresh_working_arrays_for_state(
        &mut self,
        beta: &Coefficients,
        state: &WorkingState,
        operation: &'static str,
    ) -> Result<(), EstimationError> {
        if self.working_arrays_match_state(beta, state) {
            return Ok(());
        }
        let refreshed = self.update_with_curvature(beta, state.hessian_curvature)?;
        if refreshed.eta.as_ref() != state.eta.as_ref() {
            crate::bail_invalid_estim!(
                "PIRLS {operation} refresh changed the authoritative linear predictor"
            );
        }
        Ok(())
    }

    fn current_data_objective(&self) -> Result<(f64, f64), EstimationError> {
        if self.covariate_se.is_some() {
            if !matches!(self.likelihood.spec.response, ResponseFamily::Binomial) {
                crate::bail_invalid_estim!(
                    "integrated PIRLS objective requires a binomial response"
                );
            }
            binomial_deviance_and_log_kernel_from_mean(
                self.y,
                &self.lastmu,
                self.priorweights,
            )
        } else {
            if let Some(objective) = unit_measure_deviance_and_log_kernel_from_eta(
                self.y,
                &self.workspace.eta_buf,
                &self.likelihood,
                &self.link_kind,
                self.priorweights,
            )? {
                return Ok(objective);
            }
            let deviance = self.likelihood.loglik_deviance(
                self.y,
                &self.workspace.eta_buf,
                &self.link_kind,
                self.priorweights,
            )?;
            let log_kernel = pirls_data_log_kernel_from_eta(
                self.y,
                &self.workspace.eta_buf,
                &self.likelihood,
                &self.link_kind,
                self.priorweights,
                deviance,
            )?;
            Ok((deviance, log_kernel))
        }
    }

    fn current_deviance(&self) -> Result<f64, EstimationError> {
        if self.covariate_se.is_some() {
            if !matches!(self.likelihood.spec.response, ResponseFamily::Binomial) {
                crate::bail_invalid_estim!(
                    "integrated PIRLS objective requires a binomial response"
                );
            }
            binomial_deviance_and_log_kernel_from_mean(
                self.y,
                &self.lastmu,
                self.priorweights,
            )
            .map(|objective| objective.0)
        } else {
            self.likelihood.loglik_deviance(
                self.y,
                &self.workspace.eta_buf,
                &self.link_kind,
                self.priorweights,
            )
        }
    }

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
            dispersion_phi_locked: false,
            negbin_theta_locked: false,
            quadctx,
            glm_first_step_gram,
            glm_first_step_gram_consumed: false,
            firth_design_factor: None,
            last_firth_hessian: None,
            working_array_beta_bits: Vec::new(),
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

    /// Write rows `rows` of the working-coordinate design (the basis the inner
    /// objective is optimized in) into `out`, without materializing the full
    /// `n × p` design. When the Firth design factor has been built it already
    /// holds this basis densely, so its rows are copied directly; otherwise
    /// sparse designs scatter their CSR rows, explicit designs read their own
    /// row chunk, and the implicit reparameterization forms `X[rows, :] · Qs`
    /// for the chunk alone.
    fn write_working_design_rows(
        &self,
        rows: std::ops::Range<usize>,
        mut out: ndarray::ArrayViewMut2<'_, f64>,
    ) -> Result<(), EstimationError> {
        let materialization_error = |error: gam_runtime::resource::MatrixMaterializationError| {
            EstimationError::InvalidInput(format!(
                "PIRLS square-root solve could not read working design rows: {error}"
            ))
        };
        if let Some(factor) = self.firth_design_factor.as_ref() {
            if rows.end > factor.x_dense.nrows()
                || out.dim() != (rows.len(), factor.x_dense.ncols())
            {
                crate::bail_invalid_estim!(
                    "PIRLS square-root Firth design rows {:?} do not fit design={}x{} into chunk={}x{}",
                    rows,
                    factor.x_dense.nrows(),
                    factor.x_dense.ncols(),
                    out.nrows(),
                    out.ncols()
                );
            }
            out.assign(&factor.x_dense.slice(ndarray::s![rows, ..]));
            return Ok(());
        }
        match &self.coordinate_design {
            WorkingCoordinateDesign::TransformedExplicit {
                x_transformed,
                x_csr,
            } => match x_csr.as_ref() {
                Some(csr) => Self::write_csr_rows(csr, rows, out),
                None => x_transformed
                    .row_chunk_into(rows, out)
                    .map_err(materialization_error),
            },
            WorkingCoordinateDesign::TransformedImplicit {
                transform: WorkingReparamTransform::Dense(qs),
            } => self
                .x_original
                .row_chunk_matmul_into(rows, qs.view(), out)
                .map_err(materialization_error),
            WorkingCoordinateDesign::OriginalSparseNative => match self.x_original_csr.as_ref() {
                Some(csr) => Self::write_csr_rows(csr, rows, out),
                None => self
                    .x_original
                    .row_chunk_into(rows, out)
                    .map_err(materialization_error),
            },
        }
    }

    fn write_csr_rows(
        csr: &SparseRowMat<usize, f64>,
        rows: std::ops::Range<usize>,
        mut out: ndarray::ArrayViewMut2<'_, f64>,
    ) -> Result<(), EstimationError> {
        if rows.end > csr.nrows() || out.dim() != (rows.len(), csr.ncols()) {
            crate::bail_invalid_estim!(
                "PIRLS square-root sparse design rows {:?} do not fit design={}x{} into chunk={}x{}",
                rows,
                csr.nrows(),
                csr.ncols(),
                out.nrows(),
                out.ncols()
            );
        }
        out.fill(0.0);
        let view = csr.as_ref();
        for (local, row) in rows.enumerate() {
            for (&column, &value) in view
                .col_idx_of_row_raw(row)
                .iter()
                .zip(view.val_of_row(row).iter())
            {
                out[[local, column.unbound()]] = value;
            }
        }
        Ok(())
    }

    /// Solve the damped Fisher/penalty Newton system through its augmented
    /// square root `[W^{1/2} X; E; sqrt(λ D²)]`, streamed row-block by
    /// row-block through [`TallSkinnyQrLeastSquares`]. Every coordinate design
    /// takes the same route, so the live storage is set by the block height
    /// and `p`, never by `n`: a stiff penalty or Firth reduction no longer turns
    /// a large-`n` fit into an `(n + rank(E) + p) × p` dense allocation.
    fn solve_fisher_direction_from_root(
        &self,
        beta: &Coefficients,
        state: &WorkingState,
        loop_lambda: f64,
        lm_d2: &Array1<f64>,
        firth_hessian: Option<&Array2<f64>>,
        direction_out: &mut Array1<f64>,
    ) -> Result<f64, EstimationError> {
        let n = self.lasthessian_weights.len();
        let p = state.gradient.len();
        let penalty_rows = self.penalty.rank();
        let total_rows = n
            .checked_add(penalty_rows)
            .and_then(|value| value.checked_add(p))
            .ok_or_else(|| {
                EstimationError::InvalidInput(
                    "PIRLS square-root row count overflowed usize".to_string(),
                )
            })?;
        // Block height: the shared row-chunk byte budget used by every other
        // streamed design pass, raised to `p` because a Householder block
        // reduction needs at least as many rows as columns.
        let block_rows = gam_linalg::utils::row_chunk_for_byte_budget(total_rows, p).max(p);
        let block_storage_rows = block_rows.checked_add(p).ok_or_else(|| {
            EstimationError::InvalidInput(
                "PIRLS tall-skinny QR block row count overflowed usize".to_string(),
            )
        })?;
        // Peak live storage, all at most `(block_rows + p) × p`: the design
        // chunk, the pending QR block, and during a block reduction the stacked
        // `[R; block]`, faer's working factor, its reflector basis and the thin
        // Q. (Filling an implicit `X·Qs` chunk transiently needs two further
        // chunk-sized buffers, fewer than a reduction.) Charge six atomically.
        let qr_reservation = gam_runtime::resource::MemoryGovernor::global()
            .try_reserve_dense_f64_copies(
                block_storage_rows,
                p,
                6,
                "PIRLS tall-skinny QR square-root solve",
            )
            .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
        let mut qr = TallSkinnyQrLeastSquares::new(p, block_rows)?;

        let chunk_rows = block_rows.min(n);
        let mut design_chunk = Array2::<f64>::zeros((chunk_rows, p));
        let mut start = 0;
        while start < n {
            let end = (start + chunk_rows).min(n);
            let mut chunk = design_chunk.slice_mut(ndarray::s![..end - start, ..]);
            self.write_working_design_rows(start..end, chunk.view_mut())?;
            for i in start..end {
                let weight = self.lasthessian_weights[i];
                if !(weight.is_finite() && weight >= 0.0) {
                    crate::bail_invalid_estim!(
                        "Fisher square-root solve requires finite nonnegative weight, got {weight} at row {i}"
                    );
                }
                let scale = weight.sqrt();
                let mut row = chunk.row_mut(i - start);
                row.mapv_inplace(|value| scale * value);
                qr.push_row(row.view(), (state.eta[i] - self.lastz[i]) * scale)?;
            }
            start = end;
        }

        let mut penalty_root = Array2::<f64>::zeros((penalty_rows, p).f());
        let mut penalty_residual = Array1::<f64>::zeros(penalty_rows);
        self.penalty.write_root_rows(&mut penalty_root, 0);
        self.penalty
            .write_root_residual(beta.as_ref(), &mut penalty_residual, 0);
        for i in 0..penalty_rows {
            qr.push_row(penalty_root.row(i), penalty_residual[i])?;
        }
        let mut diagonal_row = Array1::<f64>::zeros(p);
        for j in 0..p {
            let energy = loop_lambda * lm_d2[j];
            if !(energy.is_finite() && energy >= 0.0) {
                crate::bail_invalid_estim!(
                    "PIRLS square-root LM diagonal must be finite and nonnegative, got {energy} at coefficient {j}"
                );
            }
            // The exact bare-Hessian stationarity certificate calls this path
            // with transient LM damping equal to zero. Its augmented diagonal
            // row is then mathematically absent and contributes a zero row,
            // rather than a manufactured ridge.
            diagonal_row[j] = energy.sqrt();
            qr.push_row(diagonal_row.view(), 0.0)?;
            diagonal_row[j] = 0.0;
        }
        let result = qr.solve(firth_hessian, direction_out);
        drop(qr_reservation);
        result
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
                if gam_gpu::cuda_selected()
                    .map_err(|error| EstimationError::InvalidInput(error.to_string()))?
                {
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
                    //
                    // The key is a VALUE identity of `X` (gam#2515): the host
                    // pointer alone would let a same-shaped design that reuses
                    // a freed allocation read the previous design's resident
                    // Gram. The fingerprint pass is `O(n·p)` against the
                    // `O(n·p²)` product it gates.
                    let key = (
                        gam_linalg::matrix::array2_bits_fingerprint(&x_dense) as usize,
                        x_dense.nrows(),
                        p,
                    );
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
                )
                .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
                gpu_decision
                    .require_supported()
                    .map_err(EstimationError::InvalidInput)?;
                gpu_decision.log();
                if weights.iter().any(|&w| w < 0.0) {
                    // Observed-information assembly may have signed row
                    // weights.  Use Xᵀ(WX) exactly; never sqrt/clip.
                    PirlsWorkspace::add_dense_xtwx_signed(
                        weights,
                        x_dense.as_ref(),
                        &mut workspace.hessian_buf,
                    );
                } else {
                    // All weights are non-negative; the shared dense helper
                    // computes Xᵀ·diag(w)·X directly without sqrt/clip.
                    PirlsWorkspace::add_dense_xtwx_signed(
                        weights,
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
            log::trace!(
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
    ) -> Result<SparseColMat<usize, f64>, EstimationError> {
        let x_sparse = self.x_original.as_sparse().ok_or_else(|| {
            EstimationError::InvalidInput(
                "sparse-native PIRLS requires a sparse original design".to_string(),
            )
        })?;
        let PirlsPenalty::Dense { s_transformed, .. } = &self.penalty;
        self.workspace
            .assemble_sparse_penalized_hessian(x_sparse, weights, s_transformed, None)
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
        // A screen is speculative and overwrites row-space scratch. Until a
        // full accepted-state update installs a new identity, those arrays are
        // not eligible for export.
        self.working_array_beta_bits.clear();
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

        let deviance = self.current_deviance()?;
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

    fn penalized_deviance_scale(&self) -> Result<f64, EstimationError> {
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
        // Invalidate before touching any scratch array.  A failed candidate
        // evaluation must never leave a key that certifies partially-updated
        // row-space operands as belonging to the prior successful state.
        self.working_array_beta_bits.clear();
        self.last_firth_hessian = None;
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
        let resolved_likelihood_scale = self
            .likelihood
            .resolved_scale()
            .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;

        // Estimate the Gamma dispersion shape once from the warm-start η and
        // freeze it for the remainder of this inner solve. Holding the shape
        // fixed keeps the product φ·λ constant, so the penalized argmin β̂ is a
        // stationary target and the LM gain ratio stays consistent across trial
        // and accepted iterates. The shape refreshes across outer iterations
        // because a fresh model is built per inner solve. See issue #511.
        if matches!(
            resolved_likelihood_scale,
            gam_problem::ResolvedLikelihoodScale::Gamma {
                estimated: true,
                ..
            }
        ) && !self.gamma_shape_locked
        {
            let shape =
                estimate_gamma_shape_from_eta(
                &self.likelihood.spec.link,
                self.y,
                &self.workspace.eta_buf,
                self.priorweights,
            )?;
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
        if matches!(
            resolved_likelihood_scale,
            gam_problem::ResolvedLikelihoodScale::BetaPrecision {
                estimated: true,
                ..
            }
        ) && !self.beta_phi_locked
        {
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
        if matches!(
            resolved_likelihood_scale,
            gam_problem::ResolvedLikelihoodScale::Tweedie {
                estimated: true,
                ..
            }
        ) && !self.tweedie_phi_locked
        {
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

        // Estimate the Gaussian (non-identity link) / inverse Gaussian dispersion
        // φ once from the warm-start η and freeze it for this inner solve, exactly
        // as the Tweedie φ above: φ scales the working weight and the score
        // uniformly, so holding it fixed keeps φ·λ — hence β̂ — stationary.
        if matches!(
            resolved_likelihood_scale,
            gam_problem::ResolvedLikelihoodScale::Dispersion {
                estimated: true,
                ..
            }
        ) && !self.dispersion_phi_locked
        {
            let phi = estimate_dispersion_phi_from_eta(
                &self.likelihood.spec.response,
                &self.likelihood.spec.link,
                self.y,
                &self.workspace.eta_buf,
                self.priorweights,
            )?;
            self.likelihood = self.likelihood.clone().with_dispersion_phi(phi);
            self.dispersion_phi_locked = true;
        }

        // Estimate the Negative-Binomial overdispersion `theta` once from the
        // warm-start η and freeze it for this inner solve (issue #802). `theta`
        // enters the working weight `W = μθ/(θ+μ)` (the NB2 Fisher information)
        // and the working response, so holding it fixed within the inner solve
        // keeps the penalized argmin β̂ a stationary LM target (mirroring the Beta
        // φ lock above); it refreshes across outer iterations as a fresh working
        // model is built per inner solve. With `theta` frozen at the seed every
        // coefficient/η SE ignored the data's overdispersion.
        if matches!(
            resolved_likelihood_scale,
            gam_problem::ResolvedLikelihoodScale::NegativeBinomial {
                estimated: true,
                ..
            }
        ) && !self.negbin_theta_locked
        {
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
            // spectrum S_r) is built once and memoized. The per-η operator
            // rebuild then shares its reduced Fisher inverse and Hadamard-Gram
            // contractions between the working-response diagnostics and the
            // exact Jeffreys coefficient Hessian. The factor is built in the
            // correct (transformed) coefficient basis.
            let factor = self.ensure_firth_design_factor()?;
            let (hat_diag, jeffreys_logdet, jeffreys_eta_score, firth_hessian) =
                jeffreys_pirls_diagnostics_and_hessian_from_factor(
                    &factor,
                    &self.link_kind,
                    self.workspace.eta_buf.view(),
                )?;
            self.last_firth_hessian = Some(firth_hessian);
            firth = FirthDiagnostics::Active {
                jeffreys_logdet,
                hat_diag: hat_diag.clone(),
            };
            // Turn the Jeffreys linear-predictor score `g_i = ∂Φ/∂η_i =
            // ½ w'_i h_diag_i` into a working-response shift. PIRLS forms its
            // score as `Xᵀ W (η − z)` with the SAME score weights
            // `W_i = lastweights_i`, so the shift `Δ_i = g_i / W_i` adds exactly
            // `Σ_i W_i Δ_i x_i = Xᵀ g = ∂Φ/∂β` — the Jeffreys score the
            // objective value `−Φ` and the curvature `HΦ` differentiate.
            //
            // `W_i` is the prior-weighted Fisher weight `a_i w_i`, while
            // `h_diag_i` already carries `a_i` once through the operator's
            // `A^{1/2} X` design. A shift divided by the family weight `w_i`
            // alone therefore scaled row i's Jeffreys score by `a_i` a second
            // time: harmless for unit weights, but under any other prior weight
            // the score no longer matched `Φ`, the Newton step stopped being a
            // descent direction for the Firth-penalized objective, and the LM
            // step search exhausted at every ρ. Dividing by the score weight
            // itself keeps one prior weight per row, so a weight-2 row is
            // exactly a duplicated row. For the canonical logit `Δ_i` is the
            // historical `h_i (½ − μ_i)/w_i`; for probit / cloglog it carries
            // the non-canonical `w'_i/w_i`.
            ndarray::Zip::from(&mut self.lastz)
                .and(&jeffreys_eta_score)
                .and(&self.lastweights)
                .par_for_each(|zi, &score_i, &wi| {
                    if wi > 0.0 {
                        *zi += score_i / wi;
                    }
                });
        }

        let z = &self.lastz;
        // The score's operands `XᵀWη` and `XᵀWz`, for the natural gradient
        // scale: the score is their difference and cancels at the optimum, so
        // the scale is built from them rather than from it (#3339). The
        // residual buffer holds each weighted operand in turn before the score
        // residual below overwrites it.
        ndarray::Zip::from(&mut self.workspace.weighted_residual)
            .and(&self.workspace.eta_buf)
            .and(&self.lastweights)
            .par_for_each(|wr, &eta, &wi| {
                *wr = eta * wi;
            });
        let xt_w_eta = self.transformed_transpose_matvec(&self.workspace.weighted_residual);
        ndarray::Zip::from(&mut self.workspace.weighted_residual)
            .and(z)
            .and(&self.lastweights)
            .par_for_each(|wr, &zi, &wi| {
                *wr = zi * wi;
            });
        let xt_w_z = self.transformed_transpose_matvec(&self.workspace.weighted_residual);
        // Single-pass score residual: W(eta - z).
        ndarray::Zip::from(&mut self.workspace.weighted_residual)
            .and(&self.workspace.eta_buf)
            .and(z)
            .and(&self.lastweights)
            .par_for_each(|wr, &eta, &zi, &wi| {
                *wr = (eta - zi) * wi;
            });
        let mut gradient = self.transformed_transpose_matvec(&self.workspace.weighted_residual);
        let s_beta = self.penalty.shifted_gradient(beta.as_ref());
        let gradient_natural_scale = penalized_gradient_natural_scale(&xt_w_eta, &xt_w_z, &s_beta);
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

        // #2273 — a Firth fit's omitted curvature term `HΦ` is a DENSE p×p
        // matrix (`jeffreys_pirls_diagnostics_and_hessian_from_factor` builds it
        // from the dense reduced design), so a sparse assembled Hessian cannot
        // carry the fold-in the Newton direction needs. Assemble densely
        // whenever Firth is active: the sparse-native route exists to avoid a
        // dense `p²`, and Firth has already paid it — its design factor is a
        // dense `n×p` and its Hessian a dense `p×p`, rebuilt every iteration —
        // so this costs nothing sparsity was still buying.
        let (penalized_hessian, sparsehessian) = if matches!(
            self.coordinate_design,
            WorkingCoordinateDesign::OriginalSparseNative
        ) && !self.firth_bias_reduction
        {
            // The SPD-check factor is discarded here: the downstream consumer
            // is the LM Newton step, which always factorizes
            // (H + loop_lambda · I) with a non-zero loop_lambda (initial value
            // 1e-6), so it sees a different matrix.
            let (h_sparse, _factor) =
                certify_sparse_penalized_hessian(self.sparse_penalized_hessian(&solver_weights)?)?;
            (Array2::zeros((0, 0)), Some(h_sparse))
        } else {
            let penalized_hessian = self.penalized_hessian(&solver_weights)?;
            // Asymmetry within the assembly's rounding is arithmetic: `n` row
            // products and `p²` penalty and conjugation products at the matrix's
            // own scale, the objective band's accounting. An absolute `1e-8`
            // panicked on a large-scale Hessian's rounding alone (#2469).
            let symmetry_band = gam_linalg::roundoff::accumulation_growth(
                self.x_original.nrows() + penalized_hessian.nrows() * penalized_hessian.nrows(),
            ) * penalized_hessian
                .iter()
                .fold(0.0_f64, |largest, value| largest.max(value.abs()));
            assert_symmetric_tol(&penalized_hessian, "PIRLS penalized Hessian", symmetry_band);
            certify_positive_semidefinite_hessian(&penalized_hessian, "PIRLS penalized Hessian")?;
            (penalized_hessian, None)
        };
        self.workspace.matvec_buf = solver_weights;

        // The penalized objective carries no stabilization ridge (#2901 V22):
        //
        //   l_p(β; ρ) = l(β) - 0.5 * βᵀ S_λ β
        //
        // so the PIRLS fixed point, `penalty_term` and the gradient expand the
        // same `H = XᵀWX + S_λ` that drives log|H| and the implicit-gradient
        // correction.
        let (deviance, log_likelihood) = self.current_data_objective()?;

        let penalty_term = self.penalty.shifted_quadratic(beta.as_ref());
        self.last_penalty_term = penalty_term;

        self.working_array_beta_bits
            .extend(beta.as_ref().iter().map(|value| value.to_bits()));

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
            deviance_magnitude: deviance.abs(),
            penalty_term,
            firth,
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

    fn solve_unconstrained_direction(
        &mut self,
        beta: &Coefficients,
        state: &WorkingState,
        loop_lambda: f64,
        lm_d2: &Array1<f64>,
        regularized_hessian: &Array2<f64>,
        direction_out: &mut Array1<f64>,
    ) -> Result<(), EstimationError> {
        // `screen_candidate` evaluates Firth candidates through the full
        // mutable working model.  If such a candidate is rejected, the loop
        // intentionally retains `beta`/`state`, but the model's row scratch
        // belongs to the rejected point.  Rehydrate the exact authoritative
        // state before constructing A and q for min ||A d + q||.  Without this
        // state-locality repair, an LM retry combined A(candidate), q(candidate),
        // and beta/state(current), so it was not a Newton or Fisher-scoring step
        // for any objective.
        self.refresh_working_arrays_for_state(beta, state, "square-root operand")?;
        let stabilizing_floor = lm_d2
            .iter()
            .map(|&scale| loop_lambda * scale)
            .fold(f64::INFINITY, f64::min);
        // Firth scoring is itself defined by the adjusted working residual.
        // Forming X'W(eta-z*) before solving discards digits whenever the
        // Jeffreys score cancels the ordinary score, even when the penalty is
        // not yet large enough to trip the stiffness gate.  If the realized
        // row curvature is PSD, the augmented least-squares root is the exact
        // same LM system and preserves that cancellation directly.  Observed
        // noncanonical curvature is not the curvature of the Firth working
        // residual (even when every realized row weight happens to be
        // positive), so it retains the assembled dense solve; Fisher fallback
        // supplies the exact PSD-root state.
        if augmented_root_represents_working_system(
            state.hessian_curvature,
            self.firth_bias_reduction,
            self.penalty.requires_root_solve(stabilizing_floor),
            &self.lasthessian_weights,
        ) {
            let firth_hessian = if self.firth_bias_reduction {
                Some(self.last_firth_hessian.as_ref().ok_or_else(|| {
                    EstimationError::InvalidInput(
                        "Firth root solve is missing the state-local Jeffreys Hessian".to_string(),
                    )
                })?)
            } else {
                None
            };
            self.solve_fisher_direction_from_root(
                beta,
                state,
                loop_lambda,
                lm_d2,
                firth_hessian,
                direction_out,
            )?;
            Ok(())
        } else {
            // #2273 — the assembled Hessian is `XᵀWX + S`, which omits `HΦ`, so
            // the direction has to be solved from the objective's own curvature.
            // The root branch above folds `HΦ` in by congruence; this branch
            // used to drop it, and dropping it is not a small error: with the
            // Jeffreys score in the gradient and no Jeffreys curvature in the
            // system, the iteration is not Newton for any objective and
            // contracts LINEARLY. Measured on the issue's n=6 exactly-separated
            // probit fixture — the branch every non-canonical binomial link
            // reaches, because `Observed != Fisher` there — 23 iterations at a
            // ratio of 0.4937 per step to `‖g‖ = 4.3e-7`, ending in
            // `StalledAtValidMinimum` and a refused fit, at a β̂ that an
            // independent reference confirms is the right one.
            let curvature = objective_curvature_for_direction(
                regularized_hessian,
                self.objective_hessian_matrix_correction(),
            )?;
            solve_newton_direction_dense(curvature.as_ref(), &state.gradient, direction_out)?;
            Ok(())
        }
    }

    /// The Jeffreys coefficient Hessian `HΦ`, which `WorkingState.hessian`
    /// deliberately omits (the outer Laplace layer consumes `H₀` and `HΦ`
    /// separately). This is the matrix behind
    /// `objective_hessian_quadratic_correction`'s `-dᵀHΦd`, so the two are one
    /// fact reported two ways rather than two independent choices.
    fn objective_hessian_matrix_correction(&self) -> Option<&Array2<f64>> {
        self.last_firth_hessian.as_ref()
    }

    fn objective_hessian_quadratic_correction(
        &self,
        direction: &Array1<f64>,
    ) -> Result<f64, EstimationError> {
        let Some(firth_hessian) = self.objective_hessian_matrix_correction() else {
            return Ok(0.0);
        };
        if firth_hessian.dim() != (direction.len(), direction.len()) {
            crate::bail_invalid_estim!(
                "Firth objective-curvature correction shape {}x{} does not match direction length {}",
                firth_hessian.nrows(),
                firth_hessian.ncols(),
                direction.len()
            );
        }
        let correction = -direction.dot(&firth_hessian.dot(direction));
        if !correction.is_finite() {
            crate::bail_invalid_estim!("Firth objective-curvature correction is non-finite");
        }
        Ok(correction)
    }

    fn exact_unconstrained_decrement_sq(
        &mut self,
        beta: &Coefficients,
        state: &WorkingState,
    ) -> Result<Option<f64>, EstimationError> {
        self.refresh_working_arrays_for_state(beta, state, "decrement")?;
        if !augmented_root_represents_working_system(
            state.hessian_curvature,
            self.firth_bias_reduction,
            self.penalty.requires_root_solve(0.0),
            &self.lasthessian_weights,
        ) {
            return Ok(None);
        }
        let mut direction = Array1::<f64>::zeros(state.gradient.len());
        let unit_diagonal = Array1::<f64>::ones(state.gradient.len());
        self.solve_fisher_direction_from_root(
            beta,
            state,
            0.0,
            &unit_diagonal,
            None,
            &mut direction,
        )
        .map(Some)
    }
}

#[cfg(test)]
mod augmented_root_route_tests {
    use super::*;

    #[test]
    fn firth_fisher_scoring_uses_root_before_penalty_becomes_stiff() {
        let weights = ndarray::array![0.25, 0.1, 0.0];
        assert!(augmented_root_represents_working_system(
            HessianCurvatureKind::Fisher,
            true,
            false,
            &weights,
        ));
        assert!(!augmented_root_represents_working_system(
            HessianCurvatureKind::Fisher,
            false,
            false,
            &weights,
        ));
    }

    #[test]
    fn root_route_requires_the_exact_psd_working_curvature() {
        let positive = ndarray::array![0.25, 0.1];
        assert!(!augmented_root_represents_working_system(
            HessianCurvatureKind::Observed,
            true,
            true,
            &positive,
        ));
        assert!(!augmented_root_represents_working_system(
            HessianCurvatureKind::Fisher,
            true,
            true,
            &ndarray::array![0.25, -0.1],
        ));
    }
}
