use super::*;

// ---------------------------------------------------------------------------
// Kronecker-aware operator for large-scale tensor products
// ---------------------------------------------------------------------------

/// Row-wise Kronecker (face-splitting / Khatri-Rao) design operator for
/// transformation-normal value and derivative rows. It computes `forward_mul`
/// and `transpose_mul` from the natural factor pair without ever materializing
/// the full matrix.
#[derive(Clone)]
pub(crate) enum KroneckerDesign {
    /// Row-wise Khatri–Rao product `A ⊙ B`.
    ///
    /// Element-wise definition (with `n` shared rows, `p_a` and `p_b` columns):
    /// ```text
    ///     (A ⊙ B)[i, a*p_b + b]  =  A[i, a] · B[i, b]
    /// ```
    /// Forward identity (used by `forward_mul`):
    /// ```text
    ///     ((A ⊙ B) β)[i] = Σ_{a,b} A[i,a] · B[i,b] · β_mat[a,b]
    ///                    = Σ_a A[i,a] · (B · β_mat[a, :])[i]
    /// ```
    /// where `β_mat[a, b] = β[a*p_b + b]` (row-major reshape into `p_a × p_b`).
    ///
    /// Storage: `O(n·p_a + storage(B))`. The dense `n × (p_a · p_b)`
    /// materialization is never built.
    KhatriRao {
        left: Array2<f64>,   // n × p_a
        right: DesignMatrix, // n × p_b
    },
}

impl KroneckerDesign {
    pub(crate) fn new_khatri_rao(left: &Array2<f64>, right: DesignMatrix) -> Result<Self, String> {
        if left.nrows() != right.nrows() {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "KroneckerDesign row mismatch: left={}, right={}",
                    left.nrows(),
                    right.nrows()
                ),
            }
            .into());
        }
        assert_rowwise_kronecker_dimensions(left.nrows(), left.ncols(), right.ncols(), "CTN")?;
        Ok(KroneckerDesign::KhatriRao {
            left: left.clone(),
            right,
        })
    }

    pub(crate) fn nrows(&self) -> usize {
        match self {
            KroneckerDesign::KhatriRao { left, .. } => left.nrows(),
        }
    }

    pub(crate) fn ncols(&self) -> usize {
        match self {
            KroneckerDesign::KhatriRao { left, right } => left.ncols() * right.ncols(),
        }
    }

    /// Compute `self · beta` where beta has length p_a * p_b.
    /// Returns an n-vector.
    pub(crate) fn forward_mul(&self, beta: &Array1<f64>) -> Array1<f64> {
        match self {
            KroneckerDesign::KhatriRao { left, right } => {
                let pa = left.ncols();
                let pb = right.ncols();
                let n = left.nrows();
                assert_eq!(beta.len(), pa * pb);
                let beta_mat = beta.view().into_shape_with_order((pa, pb)).unwrap();
                let mut result = Array1::zeros(n);
                if let Some(right_dense) = right.as_dense_ref() {
                    let right_beta = fast_abt(right_dense, &beta_mat);
                    ndarray::Zip::from(&mut result)
                        .and(left.rows())
                        .and(right_beta.rows())
                        .par_for_each(|r, l_row, rb_row| {
                            let mut acc = 0.0;
                            for j in 0..pa {
                                acc += l_row[j] * rb_row[j];
                            }
                            *r = acc;
                        });
                    return result;
                }
                for j in 0..pa {
                    let cov_part = right.apply(&beta_mat.row(j).to_owned());
                    ndarray::Zip::from(&mut result)
                        .and(&cov_part)
                        .and(left.column(j))
                        .par_for_each(|r, &c, &l| *r += l * c);
                }
                result
            }
        }
    }

    /// Compute `self^T · v` where v is an n-vector.
    /// Returns a (p_a * p_b)-vector.
    pub(crate) fn transpose_mul(&self, v: &Array1<f64>) -> Array1<f64> {
        match self {
            KroneckerDesign::KhatriRao { left, right } => {
                let n = left.nrows();
                let pa = left.ncols();
                let pb = right.ncols();
                assert_eq!(v.len(), n);
                if let Some(right_dense) = right.as_dense_ref() {
                    let weighted_left = weight_rows(left, v);
                    let blocks = fast_atb(right_dense, &weighted_left).reversed_axes();
                    let mut out = Array1::<f64>::zeros(pa * pb);
                    for j in 0..pa {
                        out.slice_mut(s![j * pb..(j + 1) * pb])
                            .assign(&blocks.row(j));
                    }
                    return out;
                }
                let mut out = Array1::<f64>::zeros(pa * pb);
                for j in 0..pa {
                    let mut weighted_v = Array1::<f64>::zeros(n);
                    ndarray::Zip::from(&mut weighted_v)
                        .and(v)
                        .and(left.column(j))
                        .par_for_each(|w, &vi, &li| *w = vi * li);
                    let cov_block = right.apply_transpose(&weighted_v);
                    out.slice_mut(s![j * pb..(j + 1) * pb]).assign(&cov_block);
                }
                out
            }
        }
    }

    /// Compute `self^T · diag(w) · self` (weighted Gram).
    ///
    /// Thin wrapper over `weighted_cross_with(self, self, ...)`. Callers thread
    /// a real `ResourcePolicy` so chunk sizing matches the surrounding workload.
    pub(crate) fn weighted_gram(
        &self,
        w: &Array1<f64>,
        policy: &ResourcePolicy,
    ) -> Result<Array2<f64>, String> {
        self.weighted_cross_with(w.view(), self, policy)
    }

    /// Mean diagonal of `self^T diag(w) self` for non-negative weights without
    /// materializing the `(p_a p_b)^2` Gram.
    ///
    /// For one row of `X = A ⊙ B`, `||X_i||² = ||A_i||² ||B_i||²`; therefore
    /// `mean(diag(X^T W X)) = Σ_i w_i ||A_i||² ||B_i||² / (p_a p_b)`.
    /// This is the only likelihood scale the CTN cold smoothing seed consumes.
    pub(crate) fn weighted_gram_diagonal_mean(
        &self,
        weights: &Array1<f64>,
        policy: &ResourcePolicy,
    ) -> Result<f64, String> {
        PsdWeightsView::try_from_array(weights).map_err(|reason| {
            format!("KroneckerDesign::weighted_gram_diagonal_mean: {reason}")
        })?;
        match self {
            KroneckerDesign::KhatriRao { left, right } => {
                let n = left.nrows();
                if weights.len() != n || right.nrows() != n {
                    return Err(TransformationNormalError::InvalidInput {
                        reason: format!(
                            "KroneckerDesign diagonal-mean row mismatch: weights={}, left={}, right={}",
                            weights.len(),
                            n,
                            right.nrows(),
                        ),
                    }
                    .into());
                }
                let p_total = left.ncols().checked_mul(right.ncols()).ok_or_else(|| {
                    TransformationNormalError::InvalidInput {
                        reason: "KroneckerDesign diagonal-mean column product overflow"
                            .to_string(),
                    }
                    .to_string()
                })?;
                if p_total == 0 {
                    return Ok(0.0);
                }
                let rows_per_chunk = gam_runtime::resource::rows_for_target_bytes(
                    policy.row_chunk_target_bytes,
                    right.ncols().max(1),
                );
                let mut diagonal_sum = 0.0_f64;
                for start in (0..n).step_by(rows_per_chunk) {
                    let end = (start + rows_per_chunk).min(n);
                    let right_chunk = right
                        .try_row_chunk(start..end)
                        .map_err(|error| error.to_string())?;
                    for local in 0..right_chunk.nrows() {
                        let row = start + local;
                        let left_norm_squared =
                            left.row(row).iter().map(|value| value * value).sum::<f64>();
                        let right_norm_squared = right_chunk
                            .row(local)
                            .iter()
                            .map(|value| value * value)
                            .sum::<f64>();
                        diagonal_sum += weights[row] * left_norm_squared * right_norm_squared;
                    }
                }
                let mean = diagonal_sum / p_total as f64;
                if !mean.is_finite() {
                    return Err(TransformationNormalError::NonFinite {
                        reason: format!(
                            "KroneckerDesign weighted Gram diagonal mean is non-finite: {mean}"
                        ),
                    }
                    .into());
                }
                Ok(mean)
            }
        }
    }

    /// Compute `self^T · diag(w) · other` while keeping rowwise-Kronecker
    /// designs in factored form. Returns a dense (pa*pb) x (pc*pd) block matrix.
    pub(crate) fn weighted_cross_with(
        &self,
        weights: ndarray::ArrayView1<'_, f64>,
        other: &KroneckerDesign,
        policy: &ResourcePolicy,
    ) -> Result<Array2<f64>, String> {
        FiniteSignedWeightsView::try_new(weights)
            .map_err(|reason| format!("KroneckerDesign::weighted_cross_with: {reason}"))?;
        match (self, other) {
            (
                KroneckerDesign::KhatriRao { left: a, right: b },
                KroneckerDesign::KhatriRao { left: c, right: d },
            ) => {
                // If both covariate sides are dense, stay fully factored.
                if let (Some(b_dense), Some(d_dense)) = (b.as_dense_ref(), d.as_dense_ref()) {
                    return factored_weighted_cross(a, b_dense, weights, c, d_dense, policy);
                }
                // Fallback: operator-backed covariate side — iterate (a, c)
                // pairs and let the operator handle the B^T diag(w) D block.
                let n = weights.len();
                let pa = a.ncols();
                let pc = c.ncols();
                let pb = b.ncols();
                let pd = d.ncols();
                if a.nrows() != n || b.nrows() != n || c.nrows() != n || d.nrows() != n {
                    return Err(TransformationNormalError::InvalidInput {
                        reason: format!(
                            "KroneckerDesign::weighted_cross_with row mismatch: weights={n}, \
                         a={}, b={}, c={}, d={}",
                            a.nrows(),
                            b.nrows(),
                            c.nrows(),
                            d.nrows()
                        ),
                    }
                    .into());
                }
                let mut out = Array2::<f64>::zeros((pa * pb, pc * pd));
                let mut pair_weights = Array1::<f64>::zeros(n);
                for ia in 0..pa {
                    let a_col = a.column(ia);
                    for ic in 0..pc {
                        let c_col = c.column(ic);
                        for r in 0..n {
                            pair_weights[r] = weights[r] * a_col[r] * c_col[r];
                        }
                        // Route through the chunked DesignMatrix helper so the
                        // operator-backed covariate factors stay row-chunkable
                        // and never materialize n × p_cov in one shot.
                        let block =
                            chunked_weighted_bt_d_designmatrix(b, pair_weights.view(), d, policy)?;
                        out.slice_mut(s![ia * pb..(ia + 1) * pb, ic * pd..(ic + 1) * pd])
                            .assign(&block);
                    }
                }
                Ok(out)
            }
        }
    }
}

impl LinearOperator for KroneckerDesign {
    fn nrows(&self) -> usize {
        KroneckerDesign::nrows(self)
    }

    fn ncols(&self) -> usize {
        KroneckerDesign::ncols(self)
    }

    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.forward_mul(vector)
    }

    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.transpose_mul(vector)
    }

    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        if weights.len() != self.nrows() {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "KroneckerDesign::diag_xtw_x dimension mismatch: weights={}, nrows={}",
                    weights.len(),
                    self.nrows()
                ),
            }
            .into());
        }
        // The `LinearOperator` trait fixes the signature, so this entry point
        // defaults the resource policy. Internal callers in this file go
        // through `weighted_gram` directly with their own policy.
        let policy = ResourcePolicy::default_library();
        self.weighted_gram(weights, &policy)
    }
}

impl DenseDesignOperator for KroneckerDesign {
    fn row_chunk_into(
        &self,
        rows: std::ops::Range<usize>,
        mut out: ArrayViewMut2<'_, f64>,
    ) -> Result<(), MatrixMaterializationError> {
        if out.nrows() != rows.end - rows.start || out.ncols() != self.ncols() {
            return Err(MatrixMaterializationError::MissingRowChunk {
                context: "KroneckerDesign::row_chunk_into shape mismatch",
            });
        }
        match self {
            KroneckerDesign::KhatriRao { left, right } => {
                assert_rowwise_kronecker_dimensions(
                    rows.end.saturating_sub(rows.start),
                    left.ncols(),
                    right.ncols(),
                    "CTN row chunk",
                )
                .map_err(|_| MatrixMaterializationError::MissingRowChunk {
                    context: "KroneckerDesign::row_chunk_into invalid dimensions",
                })?;
                let left_chunk = left.slice(s![rows.clone(), ..]).to_owned();
                let right_chunk = right.try_row_chunk(rows)?;
                out.assign(&dense_rowwise_kronecker(
                    left_chunk.view(),
                    right_chunk.view(),
                ));
            }
        }
        Ok(())
    }

    fn to_dense(&self) -> Array2<f64> {
        match self {
            KroneckerDesign::KhatriRao { left, right } => {
                dense_rowwise_kronecker(left.view(), right.to_dense().view())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kronecker-form penalties
// ---------------------------------------------------------------------------

/// A penalty matrix in separable Kronecker form: `S_left ⊗ S_right`.
///
/// Build the direct-α (gam#2306) tensor product penalties in Kronecker-separable
/// form.
///
/// In the direct-α chart the transformation `h(y, x) = Σ_k α_k(x) v_k(y)` is
/// linear in the tensor coefficients `A ∈ R^{p_resp × p_cov}` (`β = vec(A)`,
/// row-major so `β[k·p_cov + a] = A[k, a]`), so the SPEC-5 function-space
/// roughness of `h` is exactly quadratic in `β`.
///
/// - Covariate roughness `∫∫ (L_x h)² dμ_y(y) dx = ½ βᵀ (G_y ⊗ S_{x,j}) β` with
///   `S_{x,j}` the covariate roughness Gram and `G_y = Vᵀ W V` the response
///   value-basis mass Gram under the empirical (weighted) data measure — never
///   an identity coefficient matrix, which equals the function integral only for
///   an orthonormal response basis. The location row `v_0 ≡ 1` participates:
///   this is the main-effect-of-`x` smoothing of the conditional centering
///   field, and a constant centering field lies in `null(S_{x,j})`, so the free
///   intercept is left unpenalized exactly. `G_y` is independent of the covariate
///   spatial hyper `κ` (the response basis is reused across `κ` iterations), so
///   the existing `dS_{x,j}/dκ` psi-derivative path — lifted through the same
///   `G_y` — keeps the outer criterion and its gradient in sync.
/// - Response roughness `½ βᵀ (S_{y,m} ⊗ G_x) β` with `G_x = Ψ(κ)ᵀ W Ψ(κ)` the
///   covariate value-basis mass Gram. `G_x` MOVES with the covariate spatial
///   hyper `κ`, so the response and double penalties carry a matching
///   `dG_x/dκ` / `d²G_x/dκ²` psi-derivative channel (emitted in
///   [`build_tensor_psi_derivatives`]) to keep the outer criterion and its
///   κ-gradient/Hessian in sync.
///
/// The returned [`CtnTensorPenaltyLayout`] pins the assembled order
/// `[covariate.., response.., double?]`, which the psi-derivative channel relies
/// on to address the `G_x`-bearing penalties by index.
pub(crate) fn build_tensor_penalties_kronecker(
    response_penalties: &[Array2<f64>],
    covariate_penalties: Vec<PenaltyMatrix>,
    response_val: ArrayView2<'_, f64>,
    covariate_dense: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
    p_resp: usize,
    p_cov: usize,
    config: &TransformationNormalConfig,
) -> Result<(Vec<PenaltyMatrix>, CtnTensorPenaltyLayout), String> {
    // Function-space mass Grams under the empirical (weighted) data measure.
    // `G_y` (response) is κ-independent; `G_x` (covariate) moves with κ.
    let g_resp = weighted_function_gram(response_val, weights, p_resp, "response")?;
    let g_cov = weighted_function_gram(covariate_dense, weights, p_cov, "covariate")?;

    // Shape-only response ridge (double-penalty null shrinkage). The location
    // row is the conditional centering field, identified by the likelihood; keep
    // it outside every SCOP shrinkage penalty so population shifts stay freely
    // calibrated in the selected covariate span.
    let mut shape_resp = Array2::<f64>::eye(p_resp);
    shape_resp[[0, 0]] = 0.0;

    let mut penalties = Vec::new();

    // Covariate roughness: G_y ⊗ S_{x,j}.
    let n_covariate = covariate_penalties.len();
    for s_cov in covariate_penalties {
        let fixed_log_lambda = s_cov.fixed_log_lambda();
        let right = match s_cov {
            PenaltyMatrix::Dense(right) => right,
            penalty @ PenaltyMatrix::Blockwise { .. } => penalty.to_dense(),
            PenaltyMatrix::Labeled { inner, .. } => inner.to_dense(),
            PenaltyMatrix::Fixed { inner, .. } => inner.to_dense(),
            PenaltyMatrix::KroneckerFactored { .. } => {
                return Err(
                    "transformation covariate penalties must be single-block, not already Kronecker-factored"
                        .to_string(),
                )
            }
        };
        let penalty = PenaltyMatrix::KroneckerFactored {
            left: g_resp.clone(),
            right,
        };
        penalties.push(match fixed_log_lambda {
            Some(value) => penalty.with_fixed_log_lambda(value),
            None => penalty,
        });
    }

    // Response roughness: S_{y,m} ⊗ G_x.
    let n_response = response_penalties.len();
    for s_resp in response_penalties {
        penalties.push(PenaltyMatrix::KroneckerFactored {
            left: s_resp.clone(),
            right: g_cov.clone(),
        });
    }

    // Double penalty: shape-row ridge in the covariate function measure.
    if config.double_penalty {
        penalties.push(PenaltyMatrix::KroneckerFactored {
            left: shape_resp,
            right: g_cov,
        });
    }

    let layout = CtnTensorPenaltyLayout {
        n_covariate,
        n_response,
        has_double: config.double_penalty,
    };
    // The psi-derivative channel addresses the G_x-bearing penalties by the
    // index arithmetic this layout defines; an inconsistent assembly order would
    // silently desync the κ-derivatives.
    if penalties.len() != layout.total() {
        return Err(format!(
            "CTN tensor penalty count {} disagrees with layout total {}",
            penalties.len(),
            layout.total()
        ));
    }
    Ok((penalties, layout))
}

/// Assembled order and counts of the CTN tensor penalty list. The list is laid
/// out as `[covariate.., response.., double?]`; the response and double
/// penalties carry the κ-moving `G_x` factor, so [`build_tensor_psi_derivatives`]
/// uses these ranges to emit their `dG_x/dκ` derivative components at the right
/// indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CtnTensorPenaltyLayout {
    pub n_covariate: usize,
    pub n_response: usize,
    pub has_double: bool,
}

impl CtnTensorPenaltyLayout {
    pub fn total(&self) -> usize {
        self.n_covariate + self.n_response + usize::from(self.has_double)
    }

    /// Indices of the response-roughness penalties (`S_{y,m} ⊗ G_x`).
    pub fn response_indices(&self) -> std::ops::Range<usize> {
        self.n_covariate..self.n_covariate + self.n_response
    }

    /// Index of the double (null-shrinkage) penalty (`shape_resp ⊗ G_x`), if any.
    pub fn double_index(&self) -> Option<usize> {
        self.has_double
            .then_some(self.n_covariate + self.n_response)
    }
}

/// Weighted function-space mass Gram `Bᵀ diag(w) B` of a value basis `B`
/// (`n × p`) under the empirical measure `w`.
pub(crate) fn weighted_function_gram(
    basis: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
    p: usize,
    label: &str,
) -> Result<Array2<f64>, String> {
    if basis.ncols() != p {
        return Err(format!(
            "{label} value basis has {} columns, expected {p}",
            basis.ncols()
        ));
    }
    if basis.nrows() != weights.len() {
        return Err(format!(
            "{label} value basis has {} rows but {} weights",
            basis.nrows(),
            weights.len()
        ));
    }
    let weighted = weight_rows(&basis.to_owned(), &weights.to_owned());
    let gram = basis.t().dot(&weighted);
    // Symmetrize against round-off so the factor is exactly symmetric PSD.
    let mut sym = &gram + &gram.t();
    sym.mapv_inplace(|v| 0.5 * v);
    Ok(sym)
}

/// Weighted cross Gram `Aᵀ diag(w) B` of two row-aligned bases `A` (`n × p`) and
/// `B` (`n × q`) under the empirical measure `w`. Used to differentiate the
/// covariate mass Gram `G_x = Ψᵀ W Ψ`: `dG_x/dκ_a = M + Mᵀ` with
/// `M = (∂Ψ/∂κ_a)ᵀ W Ψ`, and the second derivative adds the analogous
/// `(∂²Ψ/∂κ_a∂κ_b)ᵀ W Ψ` and `(∂Ψ/∂κ_a)ᵀ W (∂Ψ/∂κ_b)` cross terms.
pub(crate) fn weighted_cross_gram(
    a: ArrayView2<'_, f64>,
    b: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
) -> Array2<f64> {
    let weighted = weight_rows(&b.to_owned(), &weights.to_owned());
    a.t().dot(&weighted)
}

/// Symmetrize a cross-Gram contribution: `M + Mᵀ`. Used to assemble the exact
/// symmetric derivative matrices `dG_x/dκ` and `d²G_x/dκ²`.
pub(crate) fn symmetrize_sum(m: &Array2<f64>) -> Array2<f64> {
    m + &m.t()
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/// Multiply each row of a matrix by the corresponding weight.
pub(crate) fn weight_rows(x: &Array2<f64>, w: &Array1<f64>) -> Array2<f64> {
    let n = x.nrows();
    let p = x.ncols();
    assert_eq!(n, w.len());
    let mut out = Array2::zeros((n, p));
    ndarray::Zip::from(out.rows_mut())
        .and(x.rows())
        .and(w)
        .par_for_each(|mut o_row, x_row, &wi| {
            for j in 0..p {
                o_row[j] = x_row[j] * wi;
            }
        });
    out
}
