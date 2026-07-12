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

    /// SCOP-CTN forward: compute
    /// `left[i,0] · γ_0(x_i) + Σ_{k>=1} left[i,k] · γ_k(x_i)²`
    /// where `γ_k(x_i) = (right · β_mat[k, :])[i]` and
    /// `β_mat[k, j] = beta[k * p_cov + j]` (row-major reshape into
    /// `p_resp × p_cov`).
    ///
    /// Equivalent to forming `γ_mat = right · β_matᵀ` (shape `n × p_resp`),
    /// pointwise squaring columns `k>=1`, and contracting against the
    /// corresponding response basis row. The squaring is post-operator — the
    /// underlying `right` operator and the row-replicated Khatri-Rao image
    /// are never materialized. Storage cost matches `forward_mul`: an
    /// intermediate `n × p_resp` `right_beta` plus the `n` output.
    pub(crate) fn scop_affine_squared_forward(&self, beta: &Array1<f64>) -> Array1<f64> {
        match self {
            KroneckerDesign::KhatriRao { left, right } => {
                let pa = left.ncols();
                let pb = right.ncols();
                let n = left.nrows();
                assert_eq!(beta.len(), pa * pb);
                let beta_mat = beta.view().into_shape_with_order((pa, pb)).unwrap();
                let mut result = Array1::zeros(n);
                if let Some(right_dense) = right.as_dense_ref() {
                    // right_beta[i, k] = γ_k(x_i)
                    let right_beta = fast_abt(right_dense, &beta_mat);
                    ndarray::Zip::from(&mut result)
                        .and(left.rows())
                        .and(right_beta.rows())
                        .par_for_each(|r, l_row, gamma_row| {
                            let mut acc = l_row[0] * gamma_row[0];
                            for k in 1..pa {
                                let g = gamma_row[k];
                                acc += l_row[k] * g * g;
                            }
                            *r = acc;
                        });
                    return result;
                }
                // Sparse-right fallback: materialize γ_k column-by-column.
                let mut gamma_cols = Array2::<f64>::zeros((n, pa));
                for k in 0..pa {
                    let cov_part = right.apply(&beta_mat.row(k).to_owned());
                    gamma_cols.column_mut(k).assign(&cov_part);
                }
                ndarray::Zip::from(&mut result)
                    .and(left.rows())
                    .and(gamma_cols.rows())
                    .par_for_each(|r, l_row, gamma_row| {
                        let mut acc = l_row[0] * gamma_row[0];
                        for k in 1..pa {
                            let g = gamma_row[k];
                            acc += l_row[k] * g * g;
                        }
                        *r = acc;
                    });
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
/// Build tensor product penalties in Kronecker-separable form.
pub(crate) fn build_tensor_penalties_kronecker(
    response_penalties: &[Array2<f64>],
    covariate_penalties: Vec<PenaltyMatrix>,
    p_resp: usize,
    p_cov: usize,
    config: &TransformationNormalConfig,
) -> Result<Vec<PenaltyMatrix>, String> {
    let eye_cov = Array2::<f64>::eye(p_cov);
    let mut penalties = Vec::new();

    let mut shape_resp = Array2::<f64>::eye(p_resp);
    shape_resp[[0, 0]] = 0.0;

    // Covariate roughness is a latent γ prior on the squared monotone shape
    // rows. The derivative-free location row is the conditional centering
    // field itself; penalizing it by REML under-corrects broad population
    // shifts and leaves h(Y|x) calibrated only marginally instead of
    // conditionally.
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
            left: shape_resp.clone(),
            right,
        };
        penalties.push(match fixed_log_lambda {
            Some(value) => penalty.with_fixed_log_lambda(value),
            None => penalty,
        });
    }

    // Response penalties: S_resp_m ⊗ I_cov
    for s_resp in response_penalties {
        penalties.push(PenaltyMatrix::KroneckerFactored {
            left: s_resp.clone(),
            right: eye_cov.clone(),
        });
    }

    // Double penalty: shape-row ridge only. The location row is identified by
    // the likelihood as the conditional centering field; keep it outside every
    // SCOP roughness/shrinkage penalty so population shifts can be calibrated
    // in the selected covariate span.
    if config.double_penalty {
        penalties.push(PenaltyMatrix::KroneckerFactored {
            left: shape_resp,
            right: eye_cov,
        });
    }

    Ok(penalties)
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
