use super::*;

// ---------------------------------------------------------------------------
// Matrix-free joint Hessian workspace (Khatri-Rao operator-only)
// ---------------------------------------------------------------------------

/// Per-evaluation workspace for the SCOP-CTN joint Hessian.
///
/// The old linear-`h` CTN Hessian had the form
/// `X_val' W X_val + X_deriv' diag(w / h'^2) X_deriv`. SCOP is nonlinear in
/// the shape rows, so `H v` must be evaluated through the rowwise chain rule.
/// This workspace keeps the accepted `β` and row quantities and applies
/// `H`, `D H[u]`, and `D²H[u,v]` without materializing a dense p×p matrix.
pub(crate) struct TransformationNormalJointHessianWorkspace {
    /// Shared family handle. Cloning the workspace's family for each downstream
    /// matrix-free operator (dH, d²H per psi coord and per pair) would copy
    /// the full row-space Kronecker designs (~hundreds of MiB at large-scale
    /// scale) per call. Arc-sharing makes operator construction O(1).
    pub(crate) family: Arc<TransformationNormalFamily>,
    pub(crate) beta: Array1<f64>,
    pub(crate) row_quantities: TransformationNormalRowQuantityCache,
    pub(crate) dense_hessian_cache: OnceLock<Array2<f64>>,
}

impl TransformationNormalJointHessianWorkspace {
    pub(crate) fn new(
        family: Arc<TransformationNormalFamily>,
        beta: Array1<f64>,
        row_quantities: TransformationNormalRowQuantityCache,
    ) -> Result<Self, String> {
        Ok(Self {
            family,
            beta,
            row_quantities,
            dense_hessian_cache: OnceLock::new(),
        })
    }

    pub(crate) fn p_total(&self) -> usize {
        self.family.x_val_kron.ncols()
    }

    pub(crate) fn dense_hessian_cache_enabled(&self) -> bool {
        let p_total = self.p_total();
        if p_total > SCOP_HESSIAN_HVP_DENSE_CACHE_MAX_DIM {
            return false;
        }
        p_total
            .checked_mul(p_total)
            .and_then(|entries| entries.checked_mul(std::mem::size_of::<f64>()))
            .is_some_and(|bytes| bytes <= SCOP_HESSIAN_HVP_DENSE_CACHE_MAX_BYTES)
    }

    pub(crate) fn dense_hessian(&self) -> Result<&Array2<f64>, String> {
        if let Some(hessian) = self.dense_hessian_cache.get() {
            return Ok(hessian);
        }
        let dense_start = std::time::Instant::now();
        let (_, hessian) = self
            .family
            .scop_gradient_and_negative_hessian(&self.beta, &self.row_quantities)?;
        if hessian.nrows() != self.p_total() || hessian.ncols() != self.p_total() {
            return Err(format!(
                "CTN dense Hessian cache shape mismatch: got {}x{}, expected {}x{}",
                hessian.nrows(),
                hessian.ncols(),
                self.p_total(),
                self.p_total()
            ));
        }
        if hessian.iter().any(|value| !value.is_finite()) {
            return Err("CTN dense Hessian cache produced non-finite values".to_string());
        }
        self.dense_hessian_cache.set(hessian).ok();
        log::info!(
            "[STAGE] CTN dense Hessian cache build p={} elapsed={:.3}s",
            self.p_total(),
            dense_start.elapsed().as_secs_f64(),
        );
        self.dense_hessian_cache
            .get()
            .ok_or_else(|| "CTN dense Hessian cache was not initialized".to_string())
    }

    pub(crate) fn apply_hessian(&self, v: &Array1<f64>) -> Result<Array1<f64>, String> {
        if v.len() != self.p_total() {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "CTN joint Hessian matvec: input length {} != p_total {}",
                    v.len(),
                    self.p_total()
                ),
            }
            .into());
        }
        let mut out = Array1::<f64>::zeros(self.p_total());
        self.apply_hessian_into(v, &mut out)?;
        Ok(out)
    }

    pub(crate) fn apply_hessian_into(
        &self,
        v: &Array1<f64>,
        out: &mut Array1<f64>,
    ) -> Result<(), String> {
        if v.len() != self.p_total() || out.len() != self.p_total() {
            return Err(format!(
                "CTN joint Hessian matvec_into dimension mismatch: v={} out={} p_total={}",
                v.len(),
                out.len(),
                self.p_total()
            ));
        }
        if self.dense_hessian_cache_enabled() {
            let hessian = self.dense_hessian()?;
            crate::faer_ndarray::fast_av_view_into(hessian, v, out.view_mut());
            return Ok(());
        }
        self.family
            .scop_hessian_matvec_into(&self.beta, &self.row_quantities, v, out)
    }

    /// Exact diagonal of the unpenalized joint Hessian.
    pub(crate) fn compute_diagonal(&self) -> Result<Array1<f64>, String> {
        if self.dense_hessian_cache_enabled() {
            return Ok(self.dense_hessian()?.diag().to_owned());
        }
        self.family
            .scop_hessian_diagonal(&self.beta, &self.row_quantities)
    }
}

impl ExactNewtonJointHessianWorkspace for TransformationNormalJointHessianWorkspace {
    fn hessian_dense(&self) -> Result<Option<Array2<f64>>, String> {
        Ok(Some(self.dense_hessian()?.clone()))
    }

    fn hessian_source_preference(&self) -> JointHessianSourcePreference {
        JointHessianSourcePreference::Operator
    }

    /// Intent-aware refinement (#738). CTN exposes both a streamed Khatri-Rao
    /// HVP (`scop_hessian_matvec_into`, no `p_resp·p_cov` cross-product) and a
    /// structural direct-dense build (`scop_gradient_and_negative_hessian`).
    /// The right representation depends entirely on what the consumer does:
    ///
    /// - `InnerSolve` only applies `H · v`, so stream the HVP — building the
    ///   dense matrix would be `O(p²)` memory and a `Θ(n·p²)` build paid up
    ///   front for a solve that may take far fewer than `p` matvecs.
    /// - `LogdetFactorization` factorizes `H + S_λ` and therefore needs a dense
    ///   matrix regardless. Returning `Operator` here only makes the dispatch
    ///   wrap the HVP and forces the logdet consumer to immediately re-densify
    ///   it via `dense_forced` (the structural build) — a pointless round trip.
    ///   Answer `Dense` so the consumer gets the structural build directly.
    /// - `OuterEvaluation` / `OuterGradient` fall back to the blanket
    ///   `Operator` preference (matrix-free, matching the prior behaviour).
    fn hessian_source_preference_for_intent(
        &self,
        intent: MaterializationIntent,
    ) -> JointHessianSourcePreference {
        match intent {
            MaterializationIntent::LogdetFactorization => JointHessianSourcePreference::Dense,
            MaterializationIntent::InnerSolve
            | MaterializationIntent::OuterEvaluation
            | MaterializationIntent::OuterGradient => self.hessian_source_preference(),
        }
    }

    fn hessian_matvec_available(&self) -> bool {
        true
    }

    fn hessian_matvec(&self, v: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        Ok(Some(self.apply_hessian(v)?))
    }

    fn hessian_matvec_into(&self, v: &Array1<f64>, out: &mut Array1<f64>) -> Result<bool, String> {
        self.apply_hessian_into(v, out)?;
        Ok(true)
    }

    fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        Ok(Some(self.compute_diagonal()?))
    }

    fn directional_derivative(&self, arr: &Array1<f64>) -> Result<Option<Array2<f64>>, String> {
        assert!(arr.iter().all(|v| !v.is_nan()));
        Ok(None)
    }

    fn directional_derivative_operator(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        let p_total = self.p_total();
        if d_beta_flat.len() != p_total {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "CTN directional_derivative_operator length mismatch: got {}, expected {}",
                    d_beta_flat.len(),
                    p_total
                ),
            }
            .into());
        }
        let op = TransformationNormalDhMatrixFreeOperator::new(
            Arc::clone(&self.family),
            self.beta.clone(),
            self.row_quantities.clone(),
            d_beta_flat.clone(),
        );
        Ok(Some(Arc::new(op) as Arc<dyn HyperOperator>))
    }

    fn second_directional_derivative(
        &self,
        arr: &Array1<f64>,
        arr2: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert!(arr.iter().all(|v| !v.is_nan()));
        assert!(arr2.iter().all(|v| !v.is_nan()));
        Ok(None)
    }

    fn second_directional_derivative_operator(
        &self,
        d_beta_u: &Array1<f64>,
        d_beta_v: &Array1<f64>,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        let p_total = self.p_total();
        if d_beta_u.len() != p_total || d_beta_v.len() != p_total {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "CTN second_directional_derivative_operator length mismatch: u={}, v={}, expected {}",
                d_beta_u.len(),
                d_beta_v.len(),
                p_total
            ) }.into());
        }
        let op = TransformationNormalD2hMatrixFreeOperator::new(
            Arc::clone(&self.family),
            self.beta.clone(),
            self.row_quantities.clone(),
            d_beta_u.clone(),
            d_beta_v.clone(),
        );
        Ok(Some(Arc::new(op) as Arc<dyn HyperOperator>))
    }
}

/// Matrix-free directional derivative of the CTN joint Hessian.
///
/// SCOP makes the derivative row-dependent through `γ_k(x)`, so this operator
/// evaluates `D H[direction] · v` by streaming rows through the exact chain
/// rule instead of using the old scalar-weighted `X_deriv' diag(.) X_deriv`
/// identity.
pub(crate) struct TransformationNormalDhMatrixFreeOperator {
    pub(crate) family: Arc<TransformationNormalFamily>,
    pub(crate) beta: Array1<f64>,
    pub(crate) row_quantities: TransformationNormalRowQuantityCache,
    pub(crate) direction: Array1<f64>,
}

impl TransformationNormalDhMatrixFreeOperator {
    pub(crate) fn new(
        family: Arc<TransformationNormalFamily>,
        beta: Array1<f64>,
        row_quantities: TransformationNormalRowQuantityCache,
        direction: Array1<f64>,
    ) -> Self {
        Self {
            family,
            beta,
            row_quantities,
            direction,
        }
    }

    pub(crate) fn p_total(&self) -> usize {
        self.family.x_deriv_kron.ncols()
    }

    pub(crate) fn apply(&self, v: &Array1<f64>) -> Array1<f64> {
        self.family
            .scop_hessian_directional_matvec(&self.beta, &self.direction, &self.row_quantities, v)
            .expect("validated CTN dH operator inputs should not fail")
    }

    pub(crate) fn projected_gram_cache_id(&self) -> usize {
        let family_ptr = Arc::as_ptr(&self.family) as usize;
        let design_dims = self.family.covariate_design.nrows()
            ^ self.family.covariate_design.ncols().rotate_left(11);
        family_ptr ^ design_dims.rotate_left(23)
    }
}

impl HyperOperator for TransformationNormalDhMatrixFreeOperator {
    fn dim(&self) -> usize {
        self.p_total()
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        assert_eq!(v.len(), self.p_total());
        self.apply(v)
    }

    fn mul_mat(&self, factor: &Array2<f64>) -> Array2<f64> {
        assert_eq!(factor.nrows(), self.p_total());
        self.family
            .scop_hessian_directional_matmat(
                &self.beta,
                &self.direction,
                &self.row_quantities,
                factor,
            )
            .expect("validated CTN dH batched operator inputs should not fail")
    }

    fn trace_projected_factor(&self, factor: &Array2<f64>) -> f64 {
        assert_eq!(factor.nrows(), self.p_total());
        let row_grams = self
            .family
            .scop_projected_response_gram_table(factor.view())
            .expect("validated CTN dH projected Gram inputs should not fail");
        self.family
            .scop_hessian_directional_trace_from_response_grams(
                &self.beta,
                &self.direction,
                &self.row_quantities,
                row_grams.view(),
            )
            .expect("validated CTN dH projected trace inputs should not fail")
    }

    fn trace_projected_factor_cached(
        &self,
        factor: &Array2<f64>,
        cache: &ProjectedFactorCache,
    ) -> f64 {
        assert_eq!(factor.nrows(), self.p_total());
        let key =
            ProjectedFactorKey::from_factor_view(self.projected_gram_cache_id(), factor.view());
        let row_grams = cache.get_or_insert_with(key, || {
            self.family
                .scop_projected_response_gram_table(factor.view())
                .expect("validated CTN dH cached projected Gram inputs should not fail")
        });
        self.family
            .scop_hessian_directional_trace_from_response_grams(
                &self.beta,
                &self.direction,
                &self.row_quantities,
                row_grams.view(),
            )
            .expect("validated CTN dH cached projected trace inputs should not fail")
    }

    fn to_dense(&self) -> Array2<f64> {
        self.family
            .scop_hessian_directional_derivative(&self.beta, &self.direction, &self.row_quantities)
            .expect("validated CTN dH operator inputs should not fail")
    }

    fn is_implicit(&self) -> bool {
        true
    }
}

/// Matrix-free second directional derivative of the CTN joint Hessian.
///
/// This is the SCOP rowwise chain-rule operator for `D²H[u, v] · w`; it keeps
/// the memory profile of matrix-free REML while matching the dense exact
/// second derivative.
pub(crate) struct TransformationNormalD2hMatrixFreeOperator {
    pub(crate) family: Arc<TransformationNormalFamily>,
    pub(crate) beta: Array1<f64>,
    pub(crate) row_quantities: TransformationNormalRowQuantityCache,
    pub(crate) direction_u: Array1<f64>,
    pub(crate) direction_v: Array1<f64>,
}

impl TransformationNormalD2hMatrixFreeOperator {
    pub(crate) fn new(
        family: Arc<TransformationNormalFamily>,
        beta: Array1<f64>,
        row_quantities: TransformationNormalRowQuantityCache,
        direction_u: Array1<f64>,
        direction_v: Array1<f64>,
    ) -> Self {
        Self {
            family,
            beta,
            row_quantities,
            direction_u,
            direction_v,
        }
    }

    pub(crate) fn p_total(&self) -> usize {
        self.family.x_deriv_kron.ncols()
    }

    pub(crate) fn apply(&self, v: &Array1<f64>) -> Array1<f64> {
        self.family
            .scop_hessian_second_directional_matvec(
                &self.beta,
                &self.direction_u,
                &self.direction_v,
                &self.row_quantities,
                v,
            )
            .expect("validated CTN d2H operator inputs should not fail")
    }
}

impl HyperOperator for TransformationNormalD2hMatrixFreeOperator {
    fn dim(&self) -> usize {
        self.p_total()
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        assert_eq!(v.len(), self.p_total());
        self.apply(v)
    }

    fn mul_mat(&self, factor: &Array2<f64>) -> Array2<f64> {
        assert_eq!(factor.nrows(), self.p_total());
        self.family
            .scop_hessian_second_directional_matmat(
                &self.beta,
                &self.direction_u,
                &self.direction_v,
                &self.row_quantities,
                factor,
            )
            .expect("validated CTN d2H batched operator inputs should not fail")
    }

    fn to_dense(&self) -> Array2<f64> {
        self.family
            .scop_hessian_second_directional_derivative(
                &self.beta,
                &self.direction_u,
                &self.direction_v,
                &self.row_quantities,
            )
            .expect("validated CTN d2H operator inputs should not fail")
    }

    fn is_implicit(&self) -> bool {
        true
    }
}

pub(crate) struct TransformationNormalPsiHessianOperator {
    pub(crate) family: Arc<TransformationNormalFamily>,
    pub(crate) beta: Array1<f64>,
    pub(crate) op: Arc<dyn CustomFamilyPsiDerivativeOperator>,
    pub(crate) axis: usize,
    pub(crate) trace_axes: Arc<Vec<usize>>,
    pub(crate) trace_axis_pos: usize,
    pub(crate) trace_cache_id: usize,
    pub(crate) row_quantities: TransformationNormalRowQuantityCache,
}

impl TransformationNormalPsiHessianOperator {
    pub(crate) fn new(
        family: Arc<TransformationNormalFamily>,
        beta: Array1<f64>,
        op: Arc<dyn CustomFamilyPsiDerivativeOperator>,
        axis: usize,
        row_gamma: Arc<Array2<f64>>,
        row_h: Arc<Array1<f64>>,
        row_h_prime: Arc<Array1<f64>>,
        endpoint_q: Arc<Vec<LogNormalCdfDiffDerivatives>>,
    ) -> Self {
        Self::new_with_trace_axes(
            family,
            beta,
            op,
            axis,
            Arc::new(vec![axis]),
            0,
            row_gamma,
            row_h,
            row_h_prime,
            endpoint_q,
        )
    }

    pub(crate) fn new_with_trace_axes(
        family: Arc<TransformationNormalFamily>,
        beta: Array1<f64>,
        op: Arc<dyn CustomFamilyPsiDerivativeOperator>,
        axis: usize,
        trace_axes: Arc<Vec<usize>>,
        trace_axis_pos: usize,
        row_gamma: Arc<Array2<f64>>,
        row_h: Arc<Array1<f64>>,
        row_h_prime: Arc<Array1<f64>>,
        endpoint_q: Arc<Vec<LogNormalCdfDiffDerivatives>>,
    ) -> Self {
        let log_likelihood = 0.0;
        let op_ptr = Arc::as_ptr(&op) as *const () as usize;
        let row_ptr = Arc::as_ptr(&row_gamma) as usize;
        let axes_ptr = Arc::as_ptr(&trace_axes) as usize;
        let trace_cache_id = op_ptr ^ row_ptr.rotate_left(17) ^ axes_ptr.rotate_left(31);
        Self {
            family,
            beta: beta.clone(),
            op,
            axis,
            trace_axes,
            trace_axis_pos,
            trace_cache_id,
            row_quantities: TransformationNormalRowQuantityCache {
                beta: Arc::new(beta),
                gamma: row_gamma,
                h_lower: Arc::new(Array1::zeros(row_h.len())),
                h_upper: Arc::new(Array1::zeros(row_h.len())),
                h: row_h,
                h_prime: row_h_prime,
                endpoint_q,
                log_likelihood,
            },
        }
    }

    pub(crate) fn tensor_op(&self) -> &TensorKroneckerPsiOperator {
        self.op
            .as_any()
            .downcast_ref::<TensorKroneckerPsiOperator>()
            .expect("validated CTN psi operator must remain tensor-backed")
    }

    pub(crate) fn apply_columns_with_shared_cov(
        &self,
        factor: &Array2<f64>,
        cov: &Array2<f64>,
        cov_psi: &Array2<f64>,
    ) -> Array2<f64> {
        self.family
            .scop_psi_hessian_hvp_mat_from_cov(
                &self.beta,
                &self.row_quantities,
                self.axis,
                cov,
                cov_psi,
                factor.view(),
            )
            .expect("validated CTN psi Hessian operator batched input should not fail")
    }

    pub(crate) fn projected_trace_table(&self, factor: &Array2<f64>) -> Array2<f64> {
        assert_eq!(factor.nrows(), self.dim());
        let axes = self.trace_axes.as_slice();
        if axes.is_empty() {
            return Array2::<f64>::zeros((0, 1));
        }
        let n = self.family.response_val_basis.nrows();
        let p_cov = self.family.covariate_design.ncols();
        let policy = ResourcePolicy::default_library();
        let rows_per_chunk = gam_runtime::resource::rows_for_target_bytes(
            policy.row_chunk_target_bytes,
            p_cov.saturating_mul(axes.len() + 1).max(1),
        )
        .max(1)
        .min(n.max(1));

        let op = self.tensor_op();
        let mut traces = vec![0.0_f64; axes.len()];
        for start in (0..n).step_by(rows_per_chunk) {
            let end = (start + rows_per_chunk).min(n);
            let rows = start..end;
            let cov = self
                .family
                .covariate_design
                .try_row_chunk(rows.clone())
                .expect(
                    "validated CTN psi Hessian projected trace covariate chunk should not fail",
                );
            let mut cov_psi_chunks: Vec<Array2<f64>> = Vec::with_capacity(axes.len());
            for &axis in axes {
                cov_psi_chunks.push(
                    op.cov_first_axis_row_chunk_streaming(axis, rows.clone())
                        .expect("validated CTN psi Hessian projected trace first-axis chunk should not fail"),
                );
            }
            let cov_psi_views: Vec<ArrayView2<'_, f64>> =
                cov_psi_chunks.iter().map(|chunk| chunk.view()).collect();
            let chunk_traces = self
                .family
                .scop_psi_hessian_trace_factor_all_axes_chunk_from_cov(
                    &self.beta,
                    &self.row_quantities,
                    start,
                    cov.view(),
                    &cov_psi_views,
                    factor.view(),
                )
                .expect(
                    "validated CTN psi Hessian all-axis projected trace inputs should not fail",
                );
            assert_eq!(chunk_traces.len(), traces.len());
            for (total, value) in traces.iter_mut().zip(chunk_traces.into_iter()) {
                *total += value;
            }
        }
        Array2::from_shape_vec((traces.len(), 1), traces)
            .expect("validated CTN psi Hessian projected trace table shape")
    }
}

impl HyperOperator for TransformationNormalPsiHessianOperator {
    fn dim(&self) -> usize {
        self.beta.len()
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        self.family
            .scop_psi_hessian_apply_from_operator(
                &self.beta,
                &self.row_quantities,
                self.tensor_op(),
                self.axis,
                v,
            )
            .expect("validated CTN psi Hessian operator inputs should not fail")
    }

    fn mul_mat(&self, factor: &Array2<f64>) -> Array2<f64> {
        assert_eq!(factor.nrows(), self.dim());
        let p = factor.nrows();
        let k = factor.ncols();
        let cov = self
            .family
            .covariate_dense_arc()
            .expect("validated CTN psi Hessian operator covariate cache should not fail");
        let cov_psi = self
            .tensor_op()
            .materialize_cov_first_axis(self.axis)
            .expect("validated CTN psi Hessian operator covariate derivative should not fail");
        let out = self.apply_columns_with_shared_cov(factor, cov.as_ref(), &cov_psi);
        assert_eq!(out.nrows(), p);
        assert_eq!(out.ncols(), k);
        out
    }

    fn trace_projected_factor(&self, factor: &Array2<f64>) -> f64 {
        assert_eq!(factor.nrows(), self.dim());
        let cov = self
            .family
            .covariate_dense_arc()
            .expect("validated CTN psi Hessian projected trace covariate cache should not fail");
        let cov_psi = self
            .tensor_op()
            .materialize_cov_first_axis(self.axis)
            .expect(
                "validated CTN psi Hessian projected trace covariate derivative should not fail",
            );
        self.family
            .scop_psi_hessian_trace_factor_from_cov(
                &self.beta,
                &self.row_quantities,
                self.axis,
                cov.as_ref(),
                &cov_psi,
                factor.view(),
            )
            .expect("validated CTN psi Hessian projected trace inputs should not fail")
    }

    fn trace_projected_factor_cached(
        &self,
        factor: &Array2<f64>,
        cache: &ProjectedFactorCache,
    ) -> f64 {
        let key = ProjectedFactorKey::from_factor_view(self.trace_cache_id, factor.view());
        let table = cache.get_or_insert_with(key, || self.projected_trace_table(factor));
        table[[self.trace_axis_pos, 0]]
    }

    fn to_dense(&self) -> Array2<f64> {
        let p = self.dim();
        let dense = self.mul_mat(&Array2::<f64>::eye(p));
        0.5 * (&dense + &dense.t())
    }

    fn is_implicit(&self) -> bool {
        true
    }
}

pub(crate) struct TransformationNormalPsiDhMatrixFreeOperator {
    pub(crate) family: Arc<TransformationNormalFamily>,
    pub(crate) beta: Array1<f64>,
    pub(crate) direction: Array1<f64>,
    pub(crate) op: Arc<dyn CustomFamilyPsiDerivativeOperator>,
    pub(crate) axis: usize,
    pub(crate) row_quantities: TransformationNormalRowQuantityCache,
    // `RayonSafeOnce` (not `OnceLock`): `dense_matrix()` materializes via
    // `scop_psi_hessian_directional_derivative`, which dispatches an
    // `into_par_iter` over rows. This operator is invoked from outer
    // par_iter contexts (HyperOperator HVP / dense build paths); a plain
    // `OnceLock::get_or_init` here would park sibling rayon workers on the
    // OS mutex while the leader tries to schedule child rayon tasks no
    // worker is free to pick up — classic OnceLock-under-rayon deadlock
    // (see `feedback_oncelock_rayon_deadlock`). `RayonSafeOnce` keeps the
    // init lock-free; racers redundantly compute but `set()` discards the
    // losers.
    pub(crate) dense_cache: gam_runtime::resource::RayonSafeOnce<Array2<f64>>,
}

impl TransformationNormalPsiDhMatrixFreeOperator {
    pub(crate) fn new(
        family: Arc<TransformationNormalFamily>,
        beta: Array1<f64>,
        direction: Array1<f64>,
        op: Arc<dyn CustomFamilyPsiDerivativeOperator>,
        axis: usize,
        row_quantities: TransformationNormalRowQuantityCache,
    ) -> Self {
        Self {
            family,
            beta,
            direction,
            op,
            axis,
            row_quantities,
            dense_cache: gam_runtime::resource::RayonSafeOnce::new(),
        }
    }

    pub(crate) fn p_total(&self) -> usize {
        self.beta.len()
    }

    pub(crate) fn tensor_op(&self) -> &TensorKroneckerPsiOperator {
        self.op
            .as_any()
            .downcast_ref::<TensorKroneckerPsiOperator>()
            .expect("validated CTN psi dH operator must remain tensor-backed")
    }

    pub(crate) fn dense_matrix(&self) -> &Array2<f64> {
        self.dense_cache.get_or_compute(|| {
            self.family
                .scop_psi_hessian_directional_derivative(
                    &self.beta,
                    &self.direction,
                    &self.row_quantities,
                    self.tensor_op(),
                    self.axis,
                )
                .expect("validated CTN psi dH dense materialization inputs should not fail")
        })
    }

    pub(crate) fn trace_projected_factor_dense(&self, factor: &Array2<f64>) -> f64 {
        let dense_factor = crate::faer_ndarray::fast_ab(self.dense_matrix(), factor);
        factor
            .iter()
            .zip(dense_factor.iter())
            .map(|(&f, &bf)| f * bf)
            .sum()
    }

    pub(crate) fn projected_factor_cache_id(&self) -> usize {
        let family_ptr = Arc::as_ptr(&self.family) as usize;
        family_ptr
            ^ self.axis.wrapping_add(0x9e37_79b9).rotate_left(17)
            ^ self.family.response_val_basis.nrows().rotate_left(7)
            ^ self.family.covariate_design.ncols().rotate_left(29)
    }

    pub(crate) fn projected_factor_table_bytes(&self, factor: &Array2<f64>) -> usize {
        let n = self.family.response_val_basis.nrows();
        let p_resp = self.family.response_val_basis.ncols();
        let rank = factor.ncols();
        n.saturating_mul(p_resp)
            .saturating_mul(rank)
            .saturating_mul(2)
            .saturating_mul(std::mem::size_of::<f64>())
    }

    pub(crate) fn projected_factor_table_fits_budget(&self, factor: &Array2<f64>) -> bool {
        let bytes = self.projected_factor_table_bytes(factor);
        let policy = ResourcePolicy::default_library();
        bytes <= policy.max_single_materialization_bytes && bytes <= policy.max_operator_cache_bytes
    }

    pub(crate) fn projected_factor_table(&self, factor: &Array2<f64>) -> Array2<f64> {
        assert_eq!(factor.nrows(), self.p_total());
        let n = self.family.response_val_basis.nrows();
        let p_cov = self.family.covariate_design.ncols();
        let p_resp = self.family.response_val_basis.ncols();
        let rank = factor.ncols();
        let projected_len = p_resp * rank;
        let mut table = Array2::<f64>::zeros((n, 2 * projected_len));
        if n == 0 || rank == 0 {
            return table;
        }
        let op = self.tensor_op();
        let policy = ResourcePolicy::default_library();
        let live_cols = p_cov
            .saturating_mul(2)
            .saturating_add(p_resp.saturating_mul(rank).saturating_mul(2))
            .max(1);
        let rows_per_chunk =
            gam_runtime::resource::rows_for_target_bytes(policy.row_chunk_target_bytes, live_cols)
                .max(1)
                .min(n.max(1));
        for start in (0..n).step_by(rows_per_chunk) {
            let end = (start + rows_per_chunk).min(n);
            let rows = start..end;
            let cov = self
                .family
                .covariate_design
                .try_row_chunk(rows.clone())
                .expect("validated CTN psi dH projected-table covariate chunk should not fail");
            let cov_psi = op
                .cov_first_axis_row_chunk_streaming(self.axis, rows.clone())
                .expect("validated CTN psi dH projected-table covariate derivative chunk should not fail");
            for k in 0..p_resp {
                let factor_block = factor.slice(s![k * p_cov..(k + 1) * p_cov, ..]);
                let cov_projection = fast_ab(&cov, &factor_block);
                let psi_projection = fast_ab(&cov_psi, &factor_block);
                table
                    .slice_mut(s![start..end, k * rank..(k + 1) * rank])
                    .assign(&cov_projection);
                table
                    .slice_mut(s![
                        start..end,
                        projected_len + k * rank..projected_len + (k + 1) * rank
                    ])
                    .assign(&psi_projection);
            }
        }
        table
    }

    pub(crate) fn trace_projected_factor_with_projected_table(
        &self,
        factor: &Array2<f64>,
        table: ArrayView2<'_, f64>,
    ) -> f64 {
        assert_eq!(factor.nrows(), self.p_total());
        let n = self.family.response_val_basis.nrows();
        let p_cov = self.family.covariate_design.ncols();
        let p_resp = self.family.response_val_basis.ncols();
        let rank = factor.ncols();
        let projected_len = p_resp * rank;
        assert_eq!(table.dim(), (n, 2 * projected_len));
        let op = self.tensor_op();
        let policy = ResourcePolicy::default_library();
        let live_cols = p_cov.saturating_mul(2).max(1);
        let rows_per_chunk =
            gam_runtime::resource::rows_for_target_bytes(policy.row_chunk_target_bytes, live_cols)
                .max(1)
                .min(n.max(1));
        let mut total = 0.0;
        for start in (0..n).step_by(rows_per_chunk) {
            let end = (start + rows_per_chunk).min(n);
            let rows = start..end;
            let cov = self
                .family
                .covariate_design
                .try_row_chunk(rows.clone())
                .expect(
                    "validated CTN psi dH cached projected trace covariate chunk should not fail",
                );
            let cov_psi = op
                .cov_first_axis_row_chunk_streaming(self.axis, rows.clone())
                .expect("validated CTN psi dH cached projected trace covariate derivative chunk should not fail");
            let projected_cov = table.slice(s![start..end, ..projected_len]);
            let projected_psi = table.slice(s![start..end, projected_len..]);
            total += self
                .family
                .scop_psi_hessian_directional_trace_factor_chunk_from_cov(
                    &self.beta,
                    &self.direction,
                    &self.row_quantities,
                    start,
                    cov.view(),
                    cov_psi.view(),
                    factor.view(),
                    Some(projected_cov),
                    Some(projected_psi),
                )
                .expect("validated CTN psi dH cached projected trace inputs should not fail");
        }
        total
    }

    pub(crate) fn trace_projected_factor_streaming(&self, factor: &Array2<f64>) -> f64 {
        let n = self.family.response_val_basis.nrows();
        let p_cov = self.family.covariate_design.ncols();
        let rank = factor.ncols();
        let p_resp = self.family.response_val_basis.ncols();
        let policy = ResourcePolicy::default_library();
        let live_cols = p_cov
            .saturating_mul(2)
            .saturating_add(p_resp.saturating_mul(rank).saturating_mul(2))
            .max(1);
        let rows_per_chunk =
            gam_runtime::resource::rows_for_target_bytes(policy.row_chunk_target_bytes, live_cols)
                .max(1)
                .min(n.max(1));
        let op = self.tensor_op();
        let mut total = 0.0;
        for start in (0..n).step_by(rows_per_chunk) {
            let end = (start + rows_per_chunk).min(n);
            let rows = start..end;
            let cov = self
                .family
                .covariate_design
                .try_row_chunk(rows.clone())
                .expect("validated CTN psi dH projected trace covariate chunk should not fail");
            let cov_psi = op
                .cov_first_axis_row_chunk_streaming(self.axis, rows.clone())
                .expect("validated CTN psi dH projected trace covariate derivative chunk should not fail");
            total += self
                .family
                .scop_psi_hessian_directional_trace_factor_chunk_from_cov(
                    &self.beta,
                    &self.direction,
                    &self.row_quantities,
                    start,
                    cov.view(),
                    cov_psi.view(),
                    factor.view(),
                    None,
                    None,
                )
                .expect("validated CTN psi dH projected trace inputs should not fail");
        }
        total
    }
}

impl HyperOperator for TransformationNormalPsiDhMatrixFreeOperator {
    fn dim(&self) -> usize {
        self.p_total()
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        assert_eq!(v.len(), self.p_total());
        self.dense_matrix().dot(v)
    }

    fn mul_mat(&self, factor: &Array2<f64>) -> Array2<f64> {
        assert_eq!(factor.nrows(), self.p_total());
        self.dense_matrix().dot(factor)
    }

    fn trace_projected_factor(&self, factor: &Array2<f64>) -> f64 {
        assert_eq!(factor.nrows(), self.p_total());

        // At the CTN large-scale benchmark shape (`p≈264`, `n≈20k`), the
        // coefficient-space directional Hessian is tiny (< 1 MiB) while the
        // streaming projected trace repeats a full row-kernel pass for every
        // outer-gradient coordinate and every BFGS line-search evaluation.
        // Materializing the exact p×p directional derivative once and doing a
        // dense BLAS3 projection is mathematically identical to the streaming
        // contraction and is several times faster at these moderate p values.
        // Keep the old streaming path for larger coefficient systems where a
        // dense p×p cache can dominate memory or construction cost.
        if self.p_total() <= 512 {
            return self.trace_projected_factor_dense(factor);
        }

        self.trace_projected_factor_streaming(factor)
    }

    fn trace_projected_factor_cached(
        &self,
        factor: &Array2<f64>,
        cache: &crate::reml_contracts::ProjectedFactorCache,
    ) -> f64 {
        assert_eq!(factor.nrows(), self.p_total());
        if self.p_total() <= 512 || !self.projected_factor_table_fits_budget(factor) {
            return self.trace_projected_factor(factor);
        }
        let key =
            ProjectedFactorKey::from_factor_view(self.projected_factor_cache_id(), factor.view());
        let table = cache.get_or_insert_with(key, || self.projected_factor_table(factor));
        self.trace_projected_factor_with_projected_table(factor, table.view())
    }

    fn to_dense(&self) -> Array2<f64> {
        self.dense_matrix().clone()
    }

    fn is_implicit(&self) -> bool {
        true
    }
}

pub(crate) struct TransformationNormalPsiPsiHessianOperator {
    pub(crate) family: Arc<TransformationNormalFamily>,
    pub(crate) beta: Array1<f64>,
    pub(crate) op: Arc<dyn CustomFamilyPsiDerivativeOperator>,
    pub(crate) axis_i: usize,
    pub(crate) axis_j: usize,
    pub(crate) trace_axes: Arc<Vec<usize>>,
    pub(crate) trace_axis_i_pos: usize,
    pub(crate) trace_axis_j_pos: usize,
    pub(crate) trace_cache_id: usize,
    pub(crate) row_gamma: Arc<Array2<f64>>,
    pub(crate) row_h: Arc<Array1<f64>>,
    pub(crate) row_h_prime: Arc<Array1<f64>>,
    pub(crate) endpoint_q: Arc<Vec<LogNormalCdfDiffDerivatives>>,
}

impl TransformationNormalPsiPsiHessianOperator {
    pub(crate) fn new(
        family: Arc<TransformationNormalFamily>,
        beta: Array1<f64>,
        op: Arc<dyn CustomFamilyPsiDerivativeOperator>,
        axis_i: usize,
        axis_j: usize,
        row_gamma: Arc<Array2<f64>>,
        row_h: Arc<Array1<f64>>,
        row_h_prime: Arc<Array1<f64>>,
        endpoint_q: Arc<Vec<LogNormalCdfDiffDerivatives>>,
    ) -> Self {
        let trace_axes = if axis_i == axis_j {
            Arc::new(vec![axis_i])
        } else {
            Arc::new(vec![axis_i, axis_j])
        };
        let trace_axis_i_pos = 0;
        let trace_axis_j_pos = if axis_i == axis_j { 0 } else { 1 };
        Self::new_with_trace_axes(
            family,
            beta,
            op,
            axis_i,
            axis_j,
            trace_axes,
            trace_axis_i_pos,
            trace_axis_j_pos,
            row_gamma,
            row_h,
            row_h_prime,
            endpoint_q,
        )
    }

    pub(crate) fn new_with_trace_axes(
        family: Arc<TransformationNormalFamily>,
        beta: Array1<f64>,
        op: Arc<dyn CustomFamilyPsiDerivativeOperator>,
        axis_i: usize,
        axis_j: usize,
        trace_axes: Arc<Vec<usize>>,
        trace_axis_i_pos: usize,
        trace_axis_j_pos: usize,
        row_gamma: Arc<Array2<f64>>,
        row_h: Arc<Array1<f64>>,
        row_h_prime: Arc<Array1<f64>>,
        endpoint_q: Arc<Vec<LogNormalCdfDiffDerivatives>>,
    ) -> Self {
        let op_ptr = Arc::as_ptr(&op) as *const () as usize;
        let row_ptr = Arc::as_ptr(&row_gamma) as usize;
        let axes_ptr = Arc::as_ptr(&trace_axes) as usize;
        let trace_cache_id = op_ptr ^ row_ptr.rotate_left(17) ^ axes_ptr.rotate_left(31);
        Self {
            family,
            beta,
            op,
            axis_i,
            axis_j,
            trace_axes,
            trace_axis_i_pos,
            trace_axis_j_pos,
            trace_cache_id,
            row_gamma,
            row_h,
            row_h_prime,
            endpoint_q,
        }
    }

    pub(crate) fn p_total(&self) -> usize {
        self.beta.len()
    }

    pub(crate) fn tensor_op(&self) -> &TensorKroneckerPsiOperator {
        self.op
            .as_any()
            .downcast_ref::<TensorKroneckerPsiOperator>()
            .expect("validated CTN psi-psi operator must remain tensor-backed")
    }

    pub(crate) fn apply(&self, v: &Array1<f64>) -> Array1<f64> {
        self.family
            .scop_psi_psi_value_score_hvp_from_operator(
                &self.beta,
                self.tensor_op(),
                self.axis_i,
                self.axis_j,
                self.row_gamma.view(),
                self.row_h.view(),
                self.row_h_prime.view(),
                self.endpoint_q.as_slice(),
                Some(v),
            )
            .expect("validated CTN psi-psi operator inputs should not fail")
            .2
            .expect("CTN psi-psi operator called without HVP output")
    }

    pub(crate) fn apply_columns_with_shared_cov(
        &self,
        factor: &Array2<f64>,
        cov: &Array2<f64>,
        cov_i: &Array2<f64>,
        cov_j: &Array2<f64>,
        cov_ij: &Array2<f64>,
        row_start: usize,
        row_end: usize,
    ) -> Array2<f64> {
        assert_eq!(factor.nrows(), self.p_total());
        let p = factor.nrows();
        let k = factor.ncols();
        let mut out = Array2::<f64>::zeros((p, k));
        let tile_cols = SCOP_PSI_PSI_HVP_TILE_COLS.min(k.max(1));
        for start_col in (0..k).step_by(tile_cols) {
            let end_col = (start_col + tile_cols).min(k);
            let tile = factor.slice(s![.., start_col..end_col]);
            let applied = self
                .family
                .scop_psi_psi_hvp_mat_from_cov(
                    &self.beta,
                    self.row_gamma.slice(s![row_start..row_end, ..]),
                    self.row_h.slice(s![row_start..row_end]),
                    self.row_h_prime.slice(s![row_start..row_end]),
                    cov.view(),
                    cov_i.view(),
                    cov_j.view(),
                    cov_ij.view(),
                    row_start,
                    &self.endpoint_q[row_start..row_end],
                    tile,
                )
                .expect("validated CTN psi-psi batched HVP inputs should not fail");
            out.slice_mut(s![.., start_col..end_col]).assign(&applied);
        }
        out
    }

    pub(crate) fn trace_columns_with_shared_cov(
        &self,
        factor: &Array2<f64>,
        cov: &Array2<f64>,
        cov_i: &Array2<f64>,
        cov_j: &Array2<f64>,
        cov_ij: &Array2<f64>,
        row_start: usize,
        row_end: usize,
    ) -> f64 {
        self.family
            .scop_psi_psi_trace_factor_from_cov(
                &self.beta,
                self.row_gamma.slice(s![row_start..row_end, ..]),
                self.row_h.slice(s![row_start..row_end]),
                self.row_h_prime.slice(s![row_start..row_end]),
                cov.view(),
                cov_i.view(),
                cov_j.view(),
                cov_ij.view(),
                row_start,
                &self.endpoint_q[row_start..row_end],
                factor.view(),
            )
            .expect("validated CTN psi-psi projected trace inputs should not fail")
    }

    pub(crate) fn projected_trace_table(&self, factor: &Array2<f64>) -> Array2<f64> {
        assert_eq!(factor.nrows(), self.p_total());
        let n_axes = self.trace_axes.len();
        let n = self.family.response_val_basis.nrows();
        let p_cov = self.family.covariate_design.ncols();
        let policy = ResourcePolicy::default_library();
        let rows_per_chunk = gam_runtime::resource::rows_for_target_bytes(
            policy.row_chunk_target_bytes,
            p_cov.saturating_mul(n_axes + 2).max(1),
        )
        .max(1)
        .min(n.max(1));

        let op = self.tensor_op();
        let mut out = Array2::<f64>::zeros((n_axes, n_axes));
        for start in (0..n).step_by(rows_per_chunk) {
            let end = (start + rows_per_chunk).min(n);
            let rows = start..end;
            let cov = self
                .family
                .covariate_design
                .try_row_chunk(rows.clone())
                .expect("validated CTN psi-psi projected trace covariate chunk should not fail");
            let mut cov_psi_chunks: Vec<Array2<f64>> = Vec::with_capacity(n_axes);
            for &axis in self.trace_axes.iter() {
                cov_psi_chunks.push(
                    op.cov_first_axis_row_chunk_streaming(axis, rows.clone())
                        .expect("validated CTN psi-psi projected trace first-axis chunk should not fail"),
                );
            }

            for i in 0..n_axes {
                for j in i..n_axes {
                    let cov_ij = op
                        .cov_second_axis_row_chunk(self.trace_axes[i], self.trace_axes[j], rows.clone())
                        .expect("validated CTN psi-psi projected trace second-axis chunk should not fail");
                    let value = self.trace_columns_with_shared_cov(
                        factor,
                        &cov,
                        &cov_psi_chunks[i],
                        &cov_psi_chunks[j],
                        &cov_ij,
                        start,
                        end,
                    );
                    out[[i, j]] += value;
                    if i != j {
                        out[[j, i]] += value;
                    }
                }
            }
        }
        out
    }
}

impl HyperOperator for TransformationNormalPsiPsiHessianOperator {
    fn dim(&self) -> usize {
        self.p_total()
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        assert_eq!(v.len(), self.p_total());
        self.apply(v)
    }

    fn bilinear_view(&self, v: ArrayView1<'_, f64>, u: ArrayView1<'_, f64>) -> f64 {
        assert_eq!(v.len(), self.p_total());
        assert_eq!(u.len(), self.p_total());
        self.family
            .scop_psi_psi_bilinear_from_operator(
                &self.beta,
                self.tensor_op(),
                self.axis_i,
                self.axis_j,
                self.row_gamma.view(),
                self.row_h.view(),
                self.row_h_prime.view(),
                self.endpoint_q.as_slice(),
                v,
                u,
            )
            .expect("validated CTN psi-psi bilinear inputs should not fail")
    }

    fn has_fast_bilinear_view(&self) -> bool {
        true
    }

    fn trace_projected_factor(&self, factor: &Array2<f64>) -> f64 {
        assert_eq!(factor.nrows(), self.p_total());
        let n = self.family.response_val_basis.nrows();
        let p_cov = self.family.covariate_design.ncols();
        let rows_per_chunk = self
            .family
            .scop_psi_pair_rows_per_chunk(p_cov)
            .min(n.max(1));

        let mut total = 0.0;
        for start in (0..n).step_by(rows_per_chunk) {
            let end = (start + rows_per_chunk).min(n);
            let (cov, cov_i, cov_j, cov_ij) = self
                .family
                .scop_psi_pair_cov_chunks(self.tensor_op(), self.axis_i, self.axis_j, start..end)
                .expect("validated CTN psi-psi projected trace covariate chunks should not fail");
            total += self
                .trace_columns_with_shared_cov(factor, &cov, &cov_i, &cov_j, &cov_ij, start, end);
        }
        total
    }

    fn trace_projected_factor_cached(
        &self,
        factor: &Array2<f64>,
        cache: &ProjectedFactorCache,
    ) -> f64 {
        let key = ProjectedFactorKey::from_factor_view(self.trace_cache_id, factor.view());
        let table = cache.get_or_insert_with(key, || self.projected_trace_table(factor));
        table[[self.trace_axis_i_pos, self.trace_axis_j_pos]]
    }

    fn mul_mat(&self, factor: &Array2<f64>) -> Array2<f64> {
        assert_eq!(factor.nrows(), self.p_total());
        let p = factor.nrows();
        let k = factor.ncols();
        let mut out = Array2::<f64>::zeros((p, k));
        let n = self.family.response_val_basis.nrows();
        let p_cov = self.family.covariate_design.ncols();
        let rows_per_chunk = self
            .family
            .scop_psi_pair_rows_per_chunk(p_cov)
            .min(n.max(1));

        for start in (0..n).step_by(rows_per_chunk) {
            let end = (start + rows_per_chunk).min(n);
            let (cov, cov_i, cov_j, cov_ij) = self
                .family
                .scop_psi_pair_cov_chunks(self.tensor_op(), self.axis_i, self.axis_j, start..end)
                .expect("validated CTN psi-psi operator covariate chunks should not fail");
            let applied = self
                .apply_columns_with_shared_cov(factor, &cov, &cov_i, &cov_j, &cov_ij, start, end);
            out += &applied;
        }
        out
    }

    fn to_dense(&self) -> Array2<f64> {
        let p = self.p_total();
        let identity = Array2::<f64>::eye(p);
        self.mul_mat(&identity)
    }

    fn is_implicit(&self) -> bool {
        true
    }
}
