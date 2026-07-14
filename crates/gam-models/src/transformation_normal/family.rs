use super::*;

pub(crate) fn beta_bits_match(cached: &Array1<f64>, candidate: &Array1<f64>) -> bool {
    cached.len() == candidate.len()
        && cached
            .iter()
            .zip(candidate.iter())
            .all(|(&left, &right)| left.to_bits() == right.to_bits())
}

/// Optional warm-start for the transformation model: per-observation location and
/// scale values from a prior mean/SD normalizer.
#[derive(Clone, Debug)]
pub struct TransformationWarmStart {
    /// μ(x_i): conditional mean of the response at each observation's covariates.
    pub location: Array1<f64>,
    /// τ(x_i): conditional standard deviation at each observation's covariates.
    pub scale: Array1<f64>,
}

// ---------------------------------------------------------------------------
// The family
// ---------------------------------------------------------------------------

/// Conditional transformation model mapping Y|x to N(0,1).
///
/// Single-block `CustomFamily`. The block design is `x_val` (tensor product of
/// response value basis × covariate design). The family internally holds `x_deriv`
/// (tensor product of response derivative basis × covariate design) for the
/// Jacobian term in the likelihood.
#[derive(Clone)]
pub struct TransformationNormalFamily {
    // --- Tensor product design matrices ---
    /// Value design operator: keeps the tensor factors separate and materializes
    /// only row chunks or explicitly requested dense diagnostics.
    pub(crate) x_val_kron: KroneckerDesign,
    /// Derivative design operator: keeps the tensor factors separate.
    pub(crate) x_deriv_kron: KroneckerDesign,
    // --- Response-direction basis (fixed, does not depend on κ) ---
    /// Response value basis: n × p_resp. Columns: [1, I_1(y), ..., I_k(y)].
    pub(crate) response_val_basis: Array2<f64>,
    /// Response value basis at the finite lower support endpoint.
    pub(crate) response_lower_basis: Array1<f64>,
    /// Response value basis at the finite upper support endpoint.
    pub(crate) response_upper_basis: Array1<f64>,
    /// Response derivative basis: n × p_resp. Columns: [0, M_1(y), ..., M_k(y)].
    pub(crate) response_deriv_basis: Array2<f64>,

    // --- Covariate side (rebuilt on κ change) ---
    /// Original covariate design used on the right side of the tensor product.
    pub(crate) covariate_design: DesignMatrix,
    /// Dense covariate block shared by row-quantity and endpoint evaluations.
    ///
    /// CTN row quantities are rebuilt at every accepted/probed β, but the
    /// covariate design is fixed for the family. Caching this immutable
    /// `n × p_cov` block avoids repeated chunk materialization and keeps
    /// large-scale runs from churning large transient allocations.
    pub(crate) covariate_dense_cache: Arc<Mutex<Option<Arc<Array2<f64>>>>>,
    /// Optional non-negative row weights folded directly into the likelihood.
    pub(crate) weights: Arc<Array1<f64>>,
    /// Additive offset for the transformation linear predictor.
    pub(crate) offset: Arc<Array1<f64>>,
    // --- Tensor penalties ---
    pub(crate) tensor_penalties: Vec<PenaltyMatrix>,

    // --- Initial values ---
    pub(crate) initial_beta: Array1<f64>,

    // --- Config ---
    pub(crate) block_name: String,

    // --- Response basis metadata (for reconstruction at predict time) ---
    pub(crate) response_knots: Array1<f64>,
    pub(crate) response_transform: Array2<f64>,
    pub(crate) response_degree: usize,
    pub(crate) response_median: f64,
    pub(crate) response_floor_offset: Arc<Array1<f64>>,
    pub(crate) response_lower_floor_offset: f64,
    pub(crate) response_upper_floor_offset: f64,

    /// Last row-space transformation quantities for an exact beta vector.
    ///
    /// CTN line searches and exact-Newton workspace construction frequently ask
    /// for likelihood, gradient, and Hessian row factors at the same candidate
    /// coefficients. This cache keeps the expensive Khatri-Rao forward products
    /// and reciprocal powers behind a single exact-keyed entry instead of
    /// recomputing `h`, `h'`, `1/h'`, and derivative powers per call.
    pub(crate) row_quantity_cache: Arc<Mutex<Option<TransformationNormalRowQuantityCache>>>,
    /// Optional outer-score Horvitz-Thompson per-row weights.
    ///
    /// When present, this is an `n`-vector equal to the original `weights`
    /// pre-multiplied row-wise by the HT inverse-inclusion multiplier `m_i`
    /// (`m_i = 1/π_i` on sampled rows, `0.0` on unsampled rows). Assembly
    /// sites read row weights via [`Self::effective_weights`], which returns
    /// this array when present and `self.weights` otherwise. Because every
    /// per-row CTN contribution is linear in `w_i`, masking at this site
    /// gives `E[Σ_i (m_i · w_i) · f(row_i)] = Σ_i w_i · f(row_i) = full-sum`
    /// — i.e. an unbiased estimator across log-likelihood, gradient, joint
    /// Hessian (dense / matvec / diagonal), ψ, and ψ-ψ kernels.
    ///
    /// `None` preserves byte-identical legacy behavior (`effective_weights`
    /// returns the original `weights` array).
    pub(crate) outer_subsample_weights: Option<Arc<Array1<f64>>>,
}

#[derive(Clone)]
pub(crate) struct TransformationNormalRowQuantityCache {
    pub(crate) beta: Arc<Array1<f64>>,
    /// Per-row factored coordinates `α_k(x_i) = ψ_iᵀ A_{k,:}` (`n × p_resp`).
    pub(crate) alpha: Arc<Array2<f64>>,
    pub(crate) h: Arc<Array1<f64>>,
    pub(crate) h_prime: Arc<Array1<f64>>,
    pub(crate) h_lower: Arc<Array1<f64>>,
    pub(crate) h_upper: Arc<Array1<f64>>,
    pub(crate) endpoint_q: Arc<Vec<LogNormalCdfDiffDerivatives>>,
    pub(crate) log_likelihood: f64,
}

#[derive(Debug)]
pub(crate) struct TransformationNormalRowDerived {
    pub(crate) log_likelihood: f64,
    pub(crate) endpoint_q: Vec<LogNormalCdfDiffDerivatives>,
}

impl TransformationNormalRowQuantityCache {
    pub(crate) fn matches_beta(&self, beta: &Array1<f64>) -> bool {
        beta_bits_match(&self.beta, beta)
    }
}

pub(crate) fn build_transformation_row_derived(
    h: &Array1<f64>,
    h_prime: &Array1<f64>,
    h_lower: &Array1<f64>,
    h_upper: &Array1<f64>,
    weights: &Array1<f64>,
) -> Result<TransformationNormalRowDerived, String> {
    let n = h_prime.len();
    assert_eq!(h.len(), n);
    assert_eq!(h_lower.len(), n);
    assert_eq!(h_upper.len(), n);
    assert_eq!(weights.len(), n);

    if let Some((i, value)) = h
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(TransformationNormalError::NonFinite {
            reason: format!(
                "TransformationNormalFamily row_quantities: h[{i}] = {value} is not finite"
            ),
        }
        .into());
    }
    if let Some((i, value)) = weights
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(TransformationNormalError::NonFinite {
            reason: format!(
                "TransformationNormalFamily row_quantities: weight[{i}] = {value} is not finite"
            ),
        }
        .into());
    }

    // Parallelize the per-row endpoint-normalizer build: each row runs
    // `log_normal_cdf_diff_derivatives` (two `normal_logcdf` calls, three
    // 5x5 truncated polynomial multiplies, 32 `signed_normal_pdf_ratio`
    // calls) which dominates this function's runtime at large scale.
    // Rows are fully independent — no shared state, no OnceLock guards —
    // and `LogNormalCdfDiffDerivatives` is a POD struct that's `Send`.
    // The fast finiteness check rolls all eight derived quantities into
    // a single short-circuit `||` chain so the named-field error format
    // only runs on the non-finite slow path.
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let rows: Vec<(f64, LogNormalCdfDiffDerivatives)> = (0..n)
        .into_par_iter()
        .map(|i| -> Result<(f64, LogNormalCdfDiffDerivatives), String> {
            let hp = h_prime[i];
            let inv_h_prime = 1.0 / hp;
            let inv_h_prime_sq = inv_h_prime * inv_h_prime;
            let inv_h_prime_cu = inv_h_prime_sq * inv_h_prime;
            let inv_h_prime_qu = inv_h_prime_sq * inv_h_prime_sq;
            let w_i = weights[i];
            let h_i = h[i];
            let weighted_h = w_i * h_i;
            let weighted_inv_h_prime = w_i * inv_h_prime;
            let weighted_inv_h_prime_sq = w_i * inv_h_prime_sq;
            let q = log_normal_cdf_diff_derivatives(h_upper[i], h_lower[i]).map_err(|e| {
                format!("TransformationNormalFamily row_quantities: row {i} invalid endpoint normalizer: {e}")
            })?;
            let log_z = q.log_z;
            // Full truncated-normal density log φ(h) + log h' − log Z, including
            // the −½ln(2π) normalizer so the reported absolute log-likelihood
            // (and AIC) is comparable to reference tools (mlt/tram). The constant
            // is coefficient-independent: scores, Hessians, and PIT residuals
            // are unchanged.
            let row_ll = w_i
                * (-0.5 * h_i * h_i - 0.5 * (2.0 * std::f64::consts::PI).ln() + hp.ln() - log_z);
            // Fast path: a single short-circuited finiteness check. Only
            // when something is non-finite do we walk the named-field
            // table to produce a precise diagnostic.
            if !(inv_h_prime.is_finite()
                && inv_h_prime_sq.is_finite()
                && inv_h_prime_cu.is_finite()
                && inv_h_prime_qu.is_finite()
                && weighted_h.is_finite()
                && weighted_inv_h_prime.is_finite()
                && weighted_inv_h_prime_sq.is_finite()
                && log_z.is_finite())
            {
                let derived_values = [
                    ("1/h'", inv_h_prime),
                    ("1/h'^2", inv_h_prime_sq),
                    ("1/h'^3", inv_h_prime_cu),
                    ("1/h'^4", inv_h_prime_qu),
                    ("w*h", weighted_h),
                    ("w/h'", weighted_inv_h_prime),
                    ("w/h'^2", weighted_inv_h_prime_sq),
                    ("log normalizer", log_z),
                ];
                for (name, value) in derived_values {
                    if !value.is_finite() {
                        return Err(TransformationNormalError::NonFinite { reason: format!(
                            "TransformationNormalFamily row_quantities: {name} at row {i} is not finite ({value}); h'={hp} is outside the finite exact-derivative range",
                        ) }.into());
                    }
                }
                return Err(TransformationNormalError::NonFinite { reason: format!(
                    "TransformationNormalFamily row_quantities: row {i} entered non-finite branch but no named field was non-finite; h'={hp}",
                ) }.into());
            }
            Ok((row_ll, q))
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Sum row contributions in index order so the result is bit-identical
    // to the previous serial accumulation. The parallel section above only
    // parallelized the independent per-row computation; the final scalar
    // reduction stays serial to preserve numerical reproducibility against
    // existing tests.
    let mut log_likelihood = 0.0;
    let mut endpoint_q = Vec::with_capacity(n);
    for (row_ll, q) in rows {
        log_likelihood += row_ll;
        endpoint_q.push(q);
    }
    if !log_likelihood.is_finite() {
        return Err(TransformationNormalError::NonFinite { reason: format!(
            "TransformationNormalFamily row_quantities: log-likelihood is not finite ({log_likelihood})"
        ) }.into());
    }

    Ok(TransformationNormalRowDerived {
        log_likelihood,
        endpoint_q,
    })
}

impl TransformationNormalFamily {
    /// Build a transformation model from response values and a pre-built covariate
    /// design operator with associated penalties.
    ///
    /// # Arguments
    ///
    /// * `response` - The response variable y (n observations).
    /// * `covariate_design` - Pre-built covariate-side design operator (n × p_cov).
    /// * `covariate_penalties` - Penalty matrices for the covariate basis.
    /// * `config` - Response-direction basis configuration.
    /// * `warm_start` - Optional location/scale from a prior normalizer.
    pub fn new(
        response: &Array1<f64>,
        weights: &Array1<f64>,
        offset: &Array1<f64>,
        covariate_design: DesignMatrix,
        covariate_penalties: Vec<PenaltyMatrix>,
        config: &TransformationNormalConfig,
        warm_start: Option<&TransformationWarmStart>,
    ) -> Result<Self, String> {
        let n = response.len();
        if covariate_design.nrows() != n {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "response length {} != covariate design rows {}",
                    n,
                    covariate_design.nrows()
                ),
            }
            .into());
        }
        let p_cov = covariate_design.ncols();
        if p_cov == 0 {
            return Err(TransformationNormalError::DesignDegenerate {
                reason: "covariate design has zero columns".to_string(),
            }
            .into());
        }
        if weights.len() != n {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!("response length {} != weights length {}", n, weights.len()),
            }
            .into());
        }
        if offset.len() != n {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!("response length {} != offset length {}", n, offset.len()),
            }
            .into());
        }
        for (i, &weight) in weights.iter().enumerate() {
            if !weight.is_finite() {
                return Err(TransformationNormalError::NonFinite {
                    reason: format!("weights[{i}] is not finite: {weight}"),
                }
                .into());
            }
            if weight < 0.0 {
                return Err(TransformationNormalError::InvalidInput {
                    reason: format!("weights[{i}] must be non-negative: {weight}"),
                }
                .into());
            }
        }
        for (i, &value) in offset.iter().enumerate() {
            if !value.is_finite() {
                return Err(TransformationNormalError::NonFinite {
                    reason: format!("offset[{i}] is not finite: {value}"),
                }
                .into());
            }
        }
        for (i, sp) in covariate_penalties.iter().enumerate() {
            let (r, c) = sp.shape();
            if r != p_cov || c != p_cov {
                return Err(TransformationNormalError::InvalidInput {
                    reason: format!(
                        "covariate penalty {} has shape ({r}, {c}), expected ({p_cov}, {p_cov})",
                        i,
                    ),
                }
                .into());
            }
        }

        // ----- 1. Build response-direction basis -----
        let (resp_val, resp_deriv, resp_penalties, resp_knots, resp_transform) =
            build_response_basis(response, config)?;
        let p_resp = resp_val.ncols();
        let (response_lower_basis, response_upper_basis) =
            response_endpoint_value_bases(&resp_transform);

        // ----- 2. Row-wise Kronecker product (operator form) -----
        let x_val_kron = KroneckerDesign::new_khatri_rao(&resp_val, covariate_design.clone())?;
        let x_deriv_kron = KroneckerDesign::new_khatri_rao(&resp_deriv, covariate_design.clone())?;
        let p_total = p_resp * p_cov;
        assert_eq!(x_val_kron.ncols(), p_total);
        assert_eq!(x_deriv_kron.ncols(), p_total);

        // ----- 3. Warm start -----
        let initial_beta = compute_warm_start(
            response,
            weights,
            offset,
            &x_val_kron,
            &x_deriv_kron,
            &covariate_design,
            &covariate_penalties,
            p_resp,
            p_cov,
            warm_start,
        )?;

        // ----- 4. Tensor penalties (Kronecker-separable) -----
        let tensor_penalties = build_tensor_penalties_kronecker(
            &resp_penalties,
            covariate_penalties,
            p_resp,
            p_cov,
            config,
        )?;
        // Compute response median for anchoring
        let mut sorted_resp = response.to_vec();
        sorted_resp.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let resp_median = if sorted_resp.len() % 2 == 1 {
            sorted_resp[sorted_resp.len() / 2]
        } else {
            0.5 * (sorted_resp[sorted_resp.len() / 2 - 1] + sorted_resp[sorted_resp.len() / 2])
        };
        let (response_floor_offset, response_lower_floor_offset, response_upper_floor_offset) =
            response_floor_offsets(response, &resp_knots, resp_median);

        Ok(Self {
            x_val_kron,
            x_deriv_kron,
            response_val_basis: resp_val,
            response_lower_basis,
            response_upper_basis,
            response_deriv_basis: resp_deriv,
            covariate_design,
            weights: Arc::new(weights.clone()),
            offset: Arc::new(offset.clone()),
            tensor_penalties,
            initial_beta,
            block_name: "transformation".to_string(),
            response_knots: resp_knots,
            response_transform: resp_transform,
            response_degree: config.response_degree,
            response_median: resp_median,
            response_floor_offset: Arc::new(response_floor_offset),
            response_lower_floor_offset,
            response_upper_floor_offset,
            covariate_dense_cache: Arc::new(Mutex::new(None)),
            row_quantity_cache: Arc::new(Mutex::new(None)),
            outer_subsample_weights: None,
        })
    }

    /// Build from a prebuilt response basis, skipping response basis construction.
    ///
    /// For the outer loop where the response basis is precomputed once and reused
    /// across κ iterations.
    pub(crate) fn from_prebuilt_response_basis(
        response: &Array1<f64>,
        response_val_basis: Array2<f64>,
        response_deriv_basis: Array2<f64>,
        response_penalties: Vec<Array2<f64>>,
        response_knots: Array1<f64>,
        response_degree: usize,
        response_transform: Array2<f64>,
        weights: &Array1<f64>,
        offset: &Array1<f64>,
        covariate_design: DesignMatrix,
        covariate_penalties: Vec<PenaltyMatrix>,
        config: &TransformationNormalConfig,
        warm_start: Option<&TransformationWarmStart>,
    ) -> Result<Self, String> {
        let n = response_val_basis.nrows();
        if n == 0 {
            return Err(TransformationNormalError::InvalidInput {
                reason: "response basis has zero rows".to_string(),
            }
            .into());
        }
        if response.len() != n {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "response length {} != response basis rows {}",
                    response.len(),
                    n
                ),
            }
            .into());
        }
        if covariate_design.nrows() != n {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "response basis rows {} != covariate design rows {}",
                    n,
                    covariate_design.nrows()
                ),
            }
            .into());
        }
        let p_cov = covariate_design.ncols();
        if p_cov == 0 {
            return Err(TransformationNormalError::DesignDegenerate {
                reason: "covariate design has zero columns".to_string(),
            }
            .into());
        }
        if weights.len() != n {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "response basis rows {} != weights length {}",
                    n,
                    weights.len()
                ),
            }
            .into());
        }
        if offset.len() != n {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "response basis rows {} != offset length {}",
                    n,
                    offset.len()
                ),
            }
            .into());
        }
        for (i, &weight) in weights.iter().enumerate() {
            if !weight.is_finite() {
                return Err(TransformationNormalError::NonFinite {
                    reason: format!("weights[{i}] is not finite: {weight}"),
                }
                .into());
            }
            if weight < 0.0 {
                return Err(TransformationNormalError::InvalidInput {
                    reason: format!("weights[{i}] must be non-negative: {weight}"),
                }
                .into());
            }
        }
        for (i, &value) in offset.iter().enumerate() {
            if !value.is_finite() {
                return Err(TransformationNormalError::NonFinite {
                    reason: format!("offset[{i}] is not finite: {value}"),
                }
                .into());
            }
        }
        for (i, sp) in covariate_penalties.iter().enumerate() {
            let (r, c) = sp.shape();
            if r != p_cov || c != p_cov {
                return Err(TransformationNormalError::InvalidInput {
                    reason: format!(
                        "covariate penalty {} has shape ({r}, {c}), expected ({p_cov}, {p_cov})",
                        i,
                    ),
                }
                .into());
            }
        }

        let p_resp = response_val_basis.ncols();
        if response_transform.ncols() + 1 != p_resp {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "response transform columns {} imply p_resp {}, but response value basis has {} columns",
                response_transform.ncols(),
                response_transform.ncols() + 1,
                p_resp
            ) }.into());
        }
        let (response_lower_basis, response_upper_basis) =
            response_endpoint_value_bases(&response_transform);

        // Row-wise Kronecker product (operator form).
        let x_val_kron =
            KroneckerDesign::new_khatri_rao(&response_val_basis, covariate_design.clone())?;
        let x_deriv_kron =
            KroneckerDesign::new_khatri_rao(&response_deriv_basis, covariate_design.clone())?;
        let p_total = p_resp * p_cov;
        assert_eq!(x_val_kron.ncols(), p_total);
        assert_eq!(x_deriv_kron.ncols(), p_total);

        let initial_beta = compute_warm_start(
            response,
            weights,
            offset,
            &x_val_kron,
            &x_deriv_kron,
            &covariate_design,
            &covariate_penalties,
            p_resp,
            p_cov,
            warm_start,
        )?;

        // Tensor penalties (Kronecker-separable).
        let tensor_penalties = build_tensor_penalties_kronecker(
            &response_penalties,
            covariate_penalties,
            p_resp,
            p_cov,
            config,
        )?;
        // Compute response median.
        let mut sorted_resp = response.to_vec();
        sorted_resp.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let resp_median = if sorted_resp.len() % 2 == 1 {
            sorted_resp[sorted_resp.len() / 2]
        } else {
            0.5 * (sorted_resp[sorted_resp.len() / 2 - 1] + sorted_resp[sorted_resp.len() / 2])
        };
        let (response_floor_offset, response_lower_floor_offset, response_upper_floor_offset) =
            response_floor_offsets(response, &response_knots, resp_median);

        Ok(Self {
            x_val_kron,
            x_deriv_kron,
            response_val_basis,
            response_lower_basis,
            response_upper_basis,
            response_deriv_basis,
            covariate_design,
            weights: Arc::new(weights.clone()),
            offset: Arc::new(offset.clone()),
            tensor_penalties,
            initial_beta,
            block_name: "transformation".to_string(),
            response_knots: response_knots.clone(),
            response_transform: response_transform.clone(),
            response_degree,
            response_median: resp_median,
            response_floor_offset: Arc::new(response_floor_offset),
            response_lower_floor_offset,
            response_upper_floor_offset,
            covariate_dense_cache: Arc::new(Mutex::new(None)),
            row_quantity_cache: Arc::new(Mutex::new(None)),
            outer_subsample_weights: None,
        })
    }

    /// Response basis metadata for serialization/prediction.
    pub fn response_knots(&self) -> &Array1<f64> {
        &self.response_knots
    }
    pub fn response_transform(&self) -> &Array2<f64> {
        &self.response_transform
    }
    pub fn response_degree(&self) -> usize {
        self.response_degree
    }
    pub fn response_median(&self) -> f64 {
        self.response_median
    }

    /// Derive the one cold-start smoothing vector from the likelihood/penalty
    /// scale ratio without materializing the rowwise-Kronecker Gram.
    pub(crate) fn penalty_scale_log_lambdas(&self) -> Result<Array1<f64>, String> {
        let policy = ResourcePolicy::default_library();
        let likelihood_diagonal_mean = self
            .x_val_kron
            .weighted_gram_diagonal_mean(self.weights.as_ref(), &policy)?;
        Ok(ctn_penalty_scale_log_lambdas(
            &self.tensor_penalties,
            likelihood_diagonal_mean,
        ))
    }

    /// Return the single coefficient block under one explicit smoothing state.
    /// Family geometry owns penalties and coefficients; rho belongs to the
    /// optimizer/block state and has exactly one caller-supplied authority.
    pub(crate) fn block_spec(
        &self,
        initial_log_lambdas: &Array1<f64>,
    ) -> Result<ParameterBlockSpec, String> {
        if initial_log_lambdas.len() != self.tensor_penalties.len() {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "transformation smoothing vector has length {}, expected {}",
                    initial_log_lambdas.len(),
                    self.tensor_penalties.len(),
                ),
            }
            .into());
        }
        gam_problem::validate_log_strengths(initial_log_lambdas.iter().copied())
            .map_err(|error| format!("invalid transformation smoothing strength: {error}"))?;
        let offset = self.offset.as_ref() + self.response_floor_offset.as_ref();
        Ok(ParameterBlockSpec {
            name: self.block_name.clone(),
            design: DesignMatrix::Dense(DenseDesignMatrix::from(Arc::new(self.x_val_kron.clone()))),
            offset,
            penalties: self.tensor_penalties.clone(),
            nullspace_dims: vec![],
            initial_log_lambdas: initial_log_lambdas.clone(),
            initial_beta: Some(self.initial_beta.clone()),
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        })
    }

    /// Total number of coefficients.
    pub fn p_total(&self) -> usize {
        self.x_val_kron.ncols()
    }

    /// Number of observations.
    pub fn n_obs(&self) -> usize {
        self.x_val_kron.nrows()
    }

    /// Number of response-direction basis columns `p_resp` (`[1, I_1, …, I_K]`).
    pub(crate) fn p_resp(&self) -> usize {
        self.response_val_basis.ncols()
    }

    /// Number of covariate-side design columns `p_cov`.
    pub(crate) fn p_cov(&self) -> usize {
        self.covariate_design.ncols()
    }

    /// Response value basis evaluated at the finite lower support endpoint
    /// (row-independent; `[1, I_1(y_min), …, I_K(y_min)]`).
    pub(crate) fn response_lower_basis(&self) -> &Array1<f64> {
        &self.response_lower_basis
    }

    /// Response value basis evaluated at the finite upper support endpoint
    /// (row-independent; `[1, I_1(y_max), …, I_K(y_max)]`).
    pub(crate) fn response_upper_basis(&self) -> &Array1<f64> {
        &self.response_upper_basis
    }

    /// Monotonicity floor offset applied to the lower-endpoint score
    /// `ε·(y_min − median_y)`.
    pub(crate) fn response_lower_floor_offset(&self) -> f64 {
        self.response_lower_floor_offset
    }

    /// Monotonicity floor offset applied to the upper-endpoint score
    /// `ε·(y_max − median_y)`.
    pub(crate) fn response_upper_floor_offset(&self) -> f64 {
        self.response_upper_floor_offset
    }

    /// Per-row weight array used by every row-streaming SCOP assembly site.
    ///
    /// Returns the masked HT weights when an outer-score subsample is active
    /// (`outer_subsample_weights = Some(_)`), else the original `weights`.
    ///
    /// Math invariant: every CTN per-row contribution to the gradient,
    /// negative-Hessian, ψ-term, ψ-ψ-term, and log-likelihood is **linear**
    /// in this scalar — i.e. each `for i in 0..n` step is of the form
    /// `wᵢ · g(row_quantities_i, β)` with `wᵢ` appearing to the first power
    /// only. Replacing `wᵢ` with `wᵢ · m_i` (where `m_i = 1/πᵢ` on sampled
    /// rows and `0` on unsampled) yields an unbiased Horvitz-Thompson
    /// estimator: `E[Σᵢ mᵢ wᵢ g(row_i)] = Σᵢ wᵢ g(row_i) = full sum`.
    #[inline]
    pub(crate) fn effective_weights(&self) -> &Array1<f64> {
        match self.outer_subsample_weights.as_ref() {
            Some(w) => w.as_ref(),
            None => self.weights.as_ref(),
        }
    }

    /// Evaluate the response value basis `[1, I_1(y), …, I_K(y)]` (n × p_resp)
    /// at arbitrary response values using the *fitted* clamped knots and degree.
    ///
    /// This is the out-of-sample analogue of the in-sample `response_val_basis`:
    /// it reuses the exact I-spline kernel and stored knot vector so that the
    /// score-influence Jacobian (and any other predict-time geometry) evaluates
    /// `I_k(y)` consistently with how `h` was built during the fit. Knots are
    /// taken from the family (not re-derived from `response`), so the basis is
    /// identical to training whenever the response values coincide.
    pub(crate) fn evaluate_response_value_basis(
        &self,
        response: ArrayView1<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        let n = response.len();
        for (i, &v) in response.iter().enumerate() {
            if !v.is_finite() {
                return Err(TransformationNormalError::NonFinite {
                    reason: format!(
                        "evaluate_response_value_basis: response[{i}] is not finite: {v}"
                    ),
                }
                .into());
            }
        }
        let (i_val_basis, _) = create_basis::<Dense>(
            response,
            KnotSource::Provided(self.response_knots.view()),
            self.response_degree,
            BasisOptions::i_spline(),
        )
        .map_err(|e| format!("evaluate_response_value_basis: I-spline build failed: {e}"))?;
        let shape_val = i_val_basis.as_ref();
        let p_shape = shape_val.ncols();
        let p_resp = self.response_val_basis.ncols();
        if p_shape + 1 != p_resp {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "evaluate_response_value_basis: rebuilt shape columns {p_shape} imply p_resp {}, \
                     but fitted basis has {p_resp} columns",
                    p_shape + 1
                ),
            }
            .into());
        }
        let mut resp_val = Array2::<f64>::zeros((n, p_resp));
        resp_val.column_mut(0).fill(1.0);
        resp_val.slice_mut(s![.., 1..]).assign(shape_val);
        Ok(resp_val)
    }

    /// Clone the family with an outer-score Horvitz-Thompson mask installed.
    ///
    /// The mask `m` (length `n`) is `1/πᵢ` for sampled rows and `0.0` for
    /// unsampled. The returned family carries `outer_subsample_weights =
    /// Some(weights ⊙ m)`. The row-quantity cache and persistent dense
    /// Hessian cache are reset (they were keyed on β alone; the masked
    /// family's `log_likelihood` and Hessian differ from the full-data
    /// build at the same β so they must not alias). The subsample hash is
    /// computed over `m` so that two distinct masks at the same β never
    /// share a cache entry.
    pub(crate) fn with_outer_subsample(
        &self,
        mask: &Array1<f64>,
    ) -> Result<Self, TransformationNormalError> {
        let n = self.weights.len();
        if mask.len() != n {
            bail_invalid_tnorm!(
                "outer-score subsample mask length {} != n={}",
                mask.len(),
                n
            );
        }
        let mut effective = Array1::<f64>::zeros(n);
        for i in 0..n {
            let m = mask[i];
            if !m.is_finite() || m < 0.0 {
                bail_invalid_tnorm!(
                    "outer-score subsample mask[{i}] = {m} is invalid (must be finite and >= 0)"
                );
            }
            effective[i] = self.weights[i] * m;
        }
        Ok(Self {
            // Inherit immutable design / response state cheaply via Arc / clone.
            x_val_kron: self.x_val_kron.clone(),
            x_deriv_kron: self.x_deriv_kron.clone(),
            response_val_basis: self.response_val_basis.clone(),
            response_lower_basis: self.response_lower_basis.clone(),
            response_upper_basis: self.response_upper_basis.clone(),
            response_deriv_basis: self.response_deriv_basis.clone(),
            covariate_design: self.covariate_design.clone(),
            covariate_dense_cache: Arc::clone(&self.covariate_dense_cache),
            weights: Arc::clone(&self.weights),
            offset: Arc::clone(&self.offset),
            tensor_penalties: self.tensor_penalties.clone(),
            initial_beta: self.initial_beta.clone(),
            block_name: self.block_name.clone(),
            response_knots: self.response_knots.clone(),
            response_transform: self.response_transform.clone(),
            response_degree: self.response_degree,
            response_median: self.response_median,
            response_floor_offset: Arc::clone(&self.response_floor_offset),
            response_lower_floor_offset: self.response_lower_floor_offset,
            response_upper_floor_offset: self.response_upper_floor_offset,
            // Caches must NOT be shared between full-data and subsampled
            // families: the row-quantity cache stores the LL (mask-dependent),
            // and the persistent dense Hessian is keyed on β alone.
            row_quantity_cache: Arc::new(Mutex::new(None)),
            outer_subsample_weights: Some(Arc::new(effective)),
        })
    }

    /// Build an outer-subsample clone from a `BlockwiseFitOptions` row mask,
    /// returning `None` when no subsample is requested.
    pub(crate) fn maybe_with_outer_subsample_from_options(
        &self,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Self>, TransformationNormalError> {
        let Some(sub) = options.outer_score_subsample.as_ref() else {
            return Ok(None);
        };
        let n = self.weights.len();
        let mut mask = Array1::<f64>::zeros(n);
        for row in sub.rows.iter() {
            if row.index < n {
                mask[row.index] = row.weight;
            }
        }
        Ok(Some(self.with_outer_subsample(&mask)?))
    }

    // --- Internal helpers ---

    pub(crate) fn covariate_dense_arc(&self) -> Result<Arc<Array2<f64>>, String> {
        let mut cache = self
            .covariate_dense_cache
            .lock()
            .expect("CTN covariate dense cache mutex poisoned");
        if let Some(cached) = cache.as_ref() {
            return Ok(cached.clone());
        }
        let dense = Arc::new(
            self.covariate_design
                .try_row_chunk(0..self.response_val_basis.nrows())
                .map_err(|e| format!("SCOP covariate dense materialization failed: {e}"))?,
        );
        *cache = Some(dense.clone());
        Ok(dense)
    }

    pub(crate) fn row_quantities(
        &self,
        beta: &Array1<f64>,
    ) -> Result<TransformationNormalRowQuantityCache, String> {
        {
            let cache = self
                .row_quantity_cache
                .lock()
                .expect("CTN row quantity cache mutex poisoned");
            if let Some(cached) = cache.as_ref().filter(|cached| cached.matches_beta(beta)) {
                return Ok(cached.clone());
            }
        }

        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP endpoint beta reshape failed: {e}"))?;
        let cov = self.covariate_dense_arc()?;

        // Direct-α CTN (gam#2306): h(y, x) = α_0(x) + Σ_k α_k(x) · I_k(y),
        // with α_k(x) = ψ(x)ᵀ A_{k,:} and h'(y, x) = Σ_k α_k(x) · M_k(y).
        // Response column 0 is the unconstrained affine/location component;
        // the remaining response columns are the shape coordinates, kept
        // non-negative at every observation by the factored monotonicity cone
        // (`block_linear_constraints`), NOT by a squared latent chart. h is
        // exactly linear in the coefficients, so the function-space penalties
        // are quadratic in the FINAL function and the likelihood curvature
        // carries no chart second-derivative terms.
        //
        // The observed value, derivative value, and finite-support endpoints
        // all depend on the same covariate-side α_k(x_i).  Compute α once and
        // fan it out exactly.
        let alpha = fast_abt(cov.as_ref(), &beta_mat);
        let n = alpha.nrows();
        let mut h = Array1::<f64>::zeros(n);
        let mut h_prime = Array1::<f64>::zeros(n);
        let mut h_lower = Array1::<f64>::zeros(n);
        let mut h_upper = Array1::<f64>::zeros(n);
        // Write directly into the four preallocated arrays in parallel; the
        // previous path collected a `Vec<(f64,f64,f64,f64)>` then serially
        // scattered into these arrays, costing 32 bytes per row of transient
        // allocation and a single-threaded post-pass at large scale.
        ndarray::Zip::indexed(&mut h)
            .and(&mut h_prime)
            .and(&mut h_lower)
            .and(&mut h_upper)
            .par_for_each(|i, h_i, hp_i, lower_i, upper_i| {
                let alpha_row = alpha.row(i);
                let val_row = self.response_val_basis.row(i);
                let deriv_row = self.response_deriv_basis.row(i);
                let a0 = alpha_row[0];
                let offset_i = self.offset[i];
                let mut h_acc = val_row[0] * a0 + offset_i + self.response_floor_offset[i];
                let mut hp_acc = deriv_row[0] * a0 + TRANSFORMATION_MONOTONICITY_EPS;
                let mut lower_acc =
                    self.response_lower_basis[0] * a0 + offset_i + self.response_lower_floor_offset;
                let mut upper_acc =
                    self.response_upper_basis[0] * a0 + offset_i + self.response_upper_floor_offset;
                for k in 1..p_resp {
                    let a_k = alpha_row[k];
                    h_acc += val_row[k] * a_k;
                    hp_acc += deriv_row[k] * a_k;
                    lower_acc += self.response_lower_basis[k] * a_k;
                    upper_acc += self.response_upper_basis[k] * a_k;
                }
                *h_i = h_acc;
                *hp_i = hp_acc;
                *lower_i = lower_acc;
                *upper_i = upper_acc;
            });
        for (i, &value) in h.iter().enumerate() {
            if !value.is_finite() {
                return Err(TransformationNormalError::NonFinite {
                    reason: format!(
                        "TransformationNormalFamily row_quantities: h[{i}] = {value} is not finite"
                    ),
                }
                .into());
            }
            if value.abs() > TRANSFORMATION_NORMAL_H_ABS_MAX {
                return Err(TransformationNormalError::InvalidInput { reason: format!(
                    "TransformationNormalFamily row_quantities: h[{i}] = {value:.6e} exceeds the standard-normal domain bound ±{TRANSFORMATION_NORMAL_H_ABS_MAX}"
                ) }.into());
            }
        }
        // Hard monotonicity / finiteness gate: the reciprocal powers `1/h'^k`
        // for k ∈ {1,2,3,4} feed the gradient, Hessian, and psi-psi outer
        // Hessian formulas. A non-finite or non-positive h' produces +∞ /
        // signed-∞ reciprocals which then collide with zero-valued probe
        // vectors (`v_*_deriv * weights`) to yield NaN entries throughout the
        // dense psi-psi block (`hessian_psi_psi`). The likelihood gate in
        // `evaluate` already rejects such β; surface the same error here so
        // outer-Hessian probe callsites that call `row_quantities` directly
        // (psi/psi second-order terms, etc.) produce a clean Err for the
        // outer evaluator to retreat on, rather than a NaN dense block that
        // routes a flagrant non-finite Hessian back into the planner.
        let mut min_hp = f64::INFINITY;
        let mut nonfinite_idx: Option<usize> = None;
        for (i, &hp) in h_prime.iter().enumerate() {
            if !hp.is_finite() {
                nonfinite_idx = Some(i);
                break;
            }
            if hp < min_hp {
                min_hp = hp;
            }
        }
        if let Some(i) = nonfinite_idx {
            return Err(TransformationNormalError::NonFinite {
                reason: format!(
                    "TransformationNormalFamily row_quantities: h'[{i}] = {} is not finite",
                    h_prime[i]
                ),
            }
            .into());
        }
        if min_hp <= 0.0 {
            return Err(TransformationNormalError::MonotonicityViolated { reason: format!(
                "TransformationNormalFamily row_quantities: h' has non-positive values (min = {min_hp:.6e}). \
                 Monotonicity constraint may be violated."
            ) }.into());
        }
        // Compute exact f64 row derivatives. If any required reciprocal power
        // is outside the finite representable range, surface an evaluation
        // error so the outer solver can retreat; do not clamp or approximate
        // the analytic Hessian terms.
        let derived = build_transformation_row_derived(
            &h,
            &h_prime,
            &h_lower,
            &h_upper,
            self.effective_weights(),
        )?;
        let row_quantities = TransformationNormalRowQuantityCache {
            beta: Arc::new(beta.clone()),
            alpha: Arc::new(alpha),
            h: Arc::new(h),
            h_prime: Arc::new(h_prime),
            h_lower: Arc::new(h_lower),
            h_upper: Arc::new(h_upper),
            endpoint_q: Arc::new(derived.endpoint_q),
            log_likelihood: derived.log_likelihood,
        };

        let mut cache = self
            .row_quantity_cache
            .lock()
            .expect("CTN row quantity cache mutex poisoned");
        *cache = Some(row_quantities.clone());
        Ok(row_quantities)
    }
}
