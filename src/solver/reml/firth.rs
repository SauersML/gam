use super::*;
use crate::mixture_link::logit_inverse_link_jet5;
use ndarray::ShapeBuilder;

impl<'a> RemlState<'a> {
    pub(crate) fn xt_diag_x_dense_into(
        x: &Array2<f64>,
        diag: &Array1<f64>,
        weighted: &mut Array2<f64>,
    ) -> Array2<f64> {
        let (n, p) = x.dim();
        if n == 0 || p == 0 {
            return Array2::<f64>::zeros((p, p));
        }
        const STREAMING_BYTES_THRESHOLD: usize = 8 * 1024 * 1024;
        let dense_work_bytes = n
            .checked_mul(p)
            .and_then(|cells| cells.checked_mul(std::mem::size_of::<f64>()))
            .unwrap_or(usize::MAX);
        if dense_work_bytes > STREAMING_BYTES_THRESHOLD {
            return Self::xt_diag_x_dense_chunked_into(x, diag, weighted);
        }
        if weighted.raw_dim() != x.raw_dim() {
            *weighted = Array2::<f64>::zeros(x.raw_dim());
        }
        weighted.assign(x);
        ndarray::Zip::from(weighted.rows_mut())
            .and(diag.view())
            .for_each(|mut row, w| row *= *w);
        fast_atb(x, weighted)
    }

    #[inline]
    fn dense_diag_gram_chunkrows(p: usize) -> usize {
        const MIN_ROWS: usize = 512;
        const MAX_ROWS: usize = 2048;
        const TARGET_BYTES: usize = 2 * 1024 * 1024;
        let bytes_per_row = p.max(1) * std::mem::size_of::<f64>();
        (TARGET_BYTES / bytes_per_row).clamp(MIN_ROWS, MAX_ROWS)
    }

    pub(crate) fn xt_diag_x_dense_chunked_into(
        x: &Array2<f64>,
        diag: &Array1<f64>,
        weighted_chunk: &mut Array2<f64>,
    ) -> Array2<f64> {
        let (n, p) = x.dim();
        debug_assert_eq!(diag.len(), n, "diag length must match row count");
        if n == 0 || p == 0 {
            return Array2::<f64>::zeros((p, p));
        }

        let chunkrows = Self::dense_diag_gram_chunkrows(p).min(n);
        if weighted_chunk.nrows() != chunkrows || weighted_chunk.ncols() != p {
            *weighted_chunk = Array2::zeros((chunkrows, p).f());
        }

        let mut out = Array2::<f64>::zeros((p, p));
        for row_start in (0..n).step_by(chunkrows) {
            let rows = (n - row_start).min(chunkrows);
            {
                let mut chunk = weighted_chunk.slice_mut(s![0..rows, ..]);
                for local_row in 0..rows {
                    let src_row = row_start + local_row;
                    let scale = diag[src_row];
                    for col in 0..p {
                        chunk[[local_row, col]] = x[[src_row, col]] * scale;
                    }
                }
            }
            let x_chunk = x.slice(s![row_start..row_start + rows, ..]);
            let weighted_view = weighted_chunk.slice(s![0..rows, ..]);
            out += &fast_atb(&x_chunk, &weighted_view);
        }
        out
    }

    pub(crate) fn row_scale(x: &Array2<f64>, scale: &Array1<f64>) -> Array2<f64> {
        let mut out = x.clone();
        ndarray::Zip::from(out.rows_mut())
            .and(scale.view())
            .for_each(|mut row, w| row *= *w);
        out
    }

    #[inline]
    fn logit_fisher_weight_derivatives(eta: f64) -> (f64, f64, f64, f64, f64) {
        let jet = logit_inverse_link_jet5(eta);
        (jet.d1, jet.d2, jet.d3, jet.d4, jet.d5)
    }

    pub(crate) fn weighted_cross(
        left: &Array2<f64>,
        right: &Array2<f64>,
        weights: &Array1<f64>,
    ) -> Array2<f64> {
        debug_assert_eq!(left.nrows(), right.nrows());
        debug_assert_eq!(left.nrows(), weights.len());
        crate::faer_ndarray::fast_xt_diag_y(left, weights, right)
    }

    pub(crate) fn trace_product(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        debug_assert_eq!(a.nrows(), b.ncols());
        debug_assert_eq!(a.ncols(), b.nrows());
        let elems = a.nrows().saturating_mul(a.ncols());
        if elems >= 32 * 32 {
            let aview = FaerArrayView::new(a);
            let bview = FaerArrayView::new(b);
            return faer_frob_inner(aview.as_ref(), bview.as_ref().transpose());
        }
        let m = a.nrows();
        let n = a.ncols();
        kahan_sum((0..m).map(|i| {
            let mut acc = 0.0_f64;
            for j in 0..n {
                acc += a[[i, j]] * b[[j, i]];
            }
            acc
        }))
    }

    pub(crate) fn reducedweighted_gram(z: &Array2<f64>, weights: &Array1<f64>) -> Array2<f64> {
        // Returns Zᵀ diag(weights) Z (exact).
        //
        // Used for:
        //   S = Zᵀ diag(v) Z               in Hadamard-Gram apply,
        //   G_u = X_rᵀ diag(s_u) X_r       with s_u = w' ⊙ (Xu),
        // both of which avoid constructing dense n×n intermediates.
        let weighted = Self::row_scale(z, weights);
        fast_atb(z, &weighted)
    }

    pub(crate) fn reduced_crossweighted_gram(
        z_left: &Array2<f64>,
        z_right: &Array2<f64>,
        weights: &Array1<f64>,
    ) -> Array2<f64> {
        // Returns Z_leftᵀ diag(weights) Z_right (exact).
        //
        // This is used for explicit design-moving terms where left/right
        // reduced designs differ (X_r vs X_{tau,r}) in Hadamard-Gram products.
        let weighted = Self::row_scale(z_right, weights);
        fast_atb(z_left, &weighted)
    }

    pub(crate) fn reduced_diag_gram(z: &Array2<f64>, a: &Array2<f64>) -> Array1<f64> {
        // Returns diag(Z A Zᵀ), exact without forming dense n×n matrix.
        //
        // Identity:
        //   [diag(Z A Zᵀ)]_i = z_iᵀ A z_i,
        // where z_i is row i of Z. We compute this as rowwise dot(Z, Z A).
        let za = fast_ab(z, a);
        (z * &za).sum_axis(ndarray::Axis(1))
    }

    pub(crate) fn apply_hadamard_gram(
        z: &Array2<f64>,
        a_left: &Array2<f64>,
        a_right: &Array2<f64>,
        vec: &Array1<f64>,
    ) -> Array1<f64> {
        // Exact apply of:
        //   y = ((Z A_left Z^T) ⊙ (Z A_right Z^T)) vec
        // using Gram/Hadamard identity with S = Z^T diag(vec) Z:
        //   y_i = z_i^T A_left S A_right z_i.
        //
        // This is the matrix-free kernel behind the Firth terms that would
        // otherwise require dense P = M⊙M and M⊙N_u products.
        // Complexity is O(n r^2) with r = rank(X), no n×n storage.
        let s = Self::reducedweighted_gram(z, vec);
        let left_s = a_left.dot(&s);
        let t = left_s.dot(a_right);
        Self::reduced_diag_gram(z, &t)
    }

    pub(crate) fn apply_hadamard_gram_to_matrix(
        z: &Array2<f64>,
        a_left: &Array2<f64>,
        a_right: &Array2<f64>,
        mat: &Array2<f64>,
    ) -> Array2<f64> {
        // Columnwise extension of apply_hadamard_gram:
        //   out[:,j] = ((Z A_left Zᵀ) ⊙ (Z A_right Zᵀ)) mat[:,j].
        //
        // In the Firth derivatives this is used with:
        //   A_left = K_r, A_right = K_r        for Hw⊙Hw actions,
        //   A_left = K_r, A_right = A_u        for Hw⊙Nbar_u actions,
        // and symmetric variants in mixed second-direction terms.
        let mut out = Array2::<f64>::zeros(mat.raw_dim());
        for col in 0..mat.ncols() {
            let v = mat.column(col).to_owned();
            let y = Self::apply_hadamard_gram(z, a_left, a_right, &v);
            out.column_mut(col).assign(&y);
        }
        out
    }

    pub(super) fn build_firth_dense_operator(
        x_dense: &Array2<f64>,
        eta: &Array1<f64>,
        observation_weights: ndarray::ArrayView1<'_, f64>,
    ) -> Result<FirthDenseOperator, EstimationError> {
        FirthDenseOperator::build_with_observation_weights(x_dense, eta, observation_weights)
    }

    pub(super) fn firth_exact_tau_kernel(
        op: &FirthDenseOperator,
        x_tau: &Array2<f64>,
        beta: &Array1<f64>,
        include_hphi_tau_kernel: bool,
    ) -> FirthTauExactKernel {
        op.exact_tau_kernel(x_tau, beta, include_hphi_tau_kernel)
    }

    pub(super) fn firth_hphi_tau_partial_apply(
        op: &FirthDenseOperator,
        x_tau: &Array2<f64>,
        kernel: &FirthTauPartialKernel,
        rhs: &Array2<f64>,
    ) -> Array2<f64> {
        op.hphi_tau_partial_apply(x_tau, kernel, rhs)
    }
}

impl FirthDenseOperator {
    fn canonicalize_basis_column_signs(q_basis: &mut Array2<f64>) {
        for col in 0..q_basis.ncols() {
            let mut pivot_row = 0usize;
            let mut pivot_abs = 0.0_f64;
            for row in 0..q_basis.nrows() {
                let value = q_basis[[row, col]];
                let abs_value = value.abs();
                if abs_value > pivot_abs {
                    pivot_abs = abs_value;
                    pivot_row = row;
                }
            }
            if pivot_abs > 0.0 && q_basis[[pivot_row, col]] < 0.0 {
                q_basis.column_mut(col).mapv_inplace(|v| -v);
            }
        }
    }

    fn identifiable_subspace_basis_from_gram(
        gram: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array1<f64>), EstimationError> {
        let p = gram.nrows();
        debug_assert_eq!(p, gram.ncols());
        if p == 0 {
            return Ok((Array2::<f64>::eye(0), Array1::<f64>::zeros(0)));
        }

        let (evals, evecs) = gram
            .eigh(Side::Lower)
            .map_err(EstimationError::EigendecompositionFailed)?;
        let max_eval = evals.iter().copied().fold(0.0_f64, f64::max).max(1.0);
        let tol = (p.max(1) as f64) * f64::EPSILON * max_eval;
        let mut keep: Vec<usize> = evals
            .iter()
            .enumerate()
            .filter_map(|(i, &value)| if value > tol { Some(i) } else { None })
            .collect();
        if keep.is_empty() {
            return Err(EstimationError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            });
        }

        // Use one orthonormal identifiable basis for both the full-rank and
        // rank-deficient cases. Sorting retained modes by descending design
        // energy makes the representation deterministic up to eigenspace
        // degeneracy; fixing the column signs removes the remaining trivial
        // sign ambiguity.
        keep.sort_by(|&lhs, &rhs| evals[rhs].total_cmp(&evals[lhs]));
        let r = keep.len();
        let mut q_basis = Array2::<f64>::zeros((p, r));
        let mut metric_spectrum = Array1::<f64>::zeros(r);
        for (col_idx, eig_idx) in keep.into_iter().enumerate() {
            q_basis.column_mut(col_idx).assign(&evecs.column(eig_idx));
            metric_spectrum[col_idx] = evals[eig_idx];
        }
        Self::canonicalize_basis_column_signs(&mut q_basis);
        Ok((q_basis, metric_spectrum))
    }

    #[inline]
    fn trace_diag_product(diag: &Array1<f64>, matrix: &Array2<f64>) -> f64 {
        debug_assert_eq!(diag.len(), matrix.nrows());
        debug_assert_eq!(matrix.nrows(), matrix.ncols());
        kahan_sum((0..diag.len()).map(|i| diag[i] * matrix[[i, i]]))
    }

    pub(crate) fn build(
        x_dense: &Array2<f64>,
        eta: &Array1<f64>,
    ) -> Result<FirthDenseOperator, EstimationError> {
        Self::build_with_observation_weights_impl(x_dense, eta, None)
    }

    pub(crate) fn build_with_observation_weights(
        x_dense: &Array2<f64>,
        eta: &Array1<f64>,
        observation_weights: ndarray::ArrayView1<'_, f64>,
    ) -> Result<FirthDenseOperator, EstimationError> {
        Self::build_with_observation_weights_impl(x_dense, eta, Some(observation_weights))
    }

    #[inline]
    pub(crate) fn pirls_hat_diag(&self) -> Array1<f64> {
        &self.w * &self.h_diag
    }

    fn build_with_observation_weights_impl(
        x_dense: &Array2<f64>,
        eta: &Array1<f64>,
        observation_weights: Option<ndarray::ArrayView1<'_, f64>>,
    ) -> Result<FirthDenseOperator, EstimationError> {
        // Precompute dense Firth objects at current β̂ for:
        //   Φ(β) = 0.5 log|Uᵀ W U|,
        // where U is a canonical orthonormal basis of the identifiable
        // subspace of A^{1/2} X.
        //
        // Identifiability note:
        // For rank-deficient design matrices X, I = Xᵀ W X is singular for all β
        // (assuming w_i > 0). The mathematically coherent Jeffreys/Firth term is
        // therefore the identifiable-subspace form:
        //   Φ(β) = 0.5 log|Uᵀ W U|
        //        = 0.5 log|I_r(β)| - 0.5 log|S_r|,
        // with
        //   I_r = X_rᵀ W X_r,
        //   S_r = X_rᵀ X_r,
        //   U   = X_r S_r^{-1/2}.
        // The beta differential is still
        //   dΦ = 0.5 tr(I_+^† dI),
        // because S_r is beta-independent for a fixed design.
        //
        // For binomial-logit with finite eta, 0 < w_i <= 1/4, so W is SPD and
        // Null(X'WX)=Null(X). Therefore singular directions are structural
        // (from X) and independent of beta, which is why fixed-Q reduced-space
        // derivatives are exact in this regime.
        //
        // This implementation uses the fixed identifiable-basis route directly:
        //   I_r = X_rᵀ W X_r,  S_r = X_rᵀ X_r,
        //   I_+^† = Q I_r^{-1} Qᵀ,
        //   Φ     = 0.5 (log|I_r| - log|S_r|).
        //
        // Why `eta` (not `mu`) enters here:
        //   all logistic weight derivatives are functions of eta through
        //   mu(eta)=sigmoid(eta), and exact derivative consistency is preserved by
        //   generating (w, w', w'', w''', w'''') from one coherent eta source.
        // Using eta avoids any mismatch from externally clamped/post-processed mu.
        //
        // We cache reduced operators so derivatives are exact but matrix-free in n:
        //   K_r = I_r^{-1},  h = diag(X_r K_r X_rᵀ),  B = diag(w')X,
        // along with logistic derivatives w', w'', w''', w'''' to evaluate:
        //   H_φ      = ∇²_β Φ,
        //   D H_φ[u],
        //   D² H_φ[u,v]
        // exactly via reduced-space products (no explicit high-order tensors).
        //
        // Fixed observation weights:
        // When callers provide nonnegative case weights a_i that are constant in
        // β, the Jeffreys information is
        //   I(β) = Xᵀ diag(a_i w_i(η)) X.
        // We fold those fixed a_i into the identifiable basis and reduced design
        // via X̃ = diag(sqrt(a_i)) X, so all derivative formulas continue to use
        // the same η-derivatives of the family Fisher weights w(η), w'(η), ....
        let n = x_dense.nrows();
        if eta.len() != n {
            return Err(EstimationError::InvalidInput(format!(
                "Firth operator shape mismatch: nrows={}, eta_len={}",
                n,
                eta.len()
            )));
        }
        let observation_weight_sqrt = if let Some(weights) = observation_weights {
            if weights.len() != n {
                return Err(EstimationError::InvalidInput(format!(
                    "Firth operator observation weight length {} != number of rows {}",
                    weights.len(),
                    n
                )));
            }
            let mut sqrt = Array1::<f64>::zeros(n);
            for i in 0..n {
                let weight = weights[i];
                if !weight.is_finite() || weight < 0.0 {
                    return Err(EstimationError::InvalidInput(format!(
                        "Firth operator requires finite nonnegative observation weights, got {} at row {}",
                        weight, i
                    )));
                }
                sqrt[i] = weight.sqrt();
            }
            Some(sqrt)
        } else {
            None
        };
        let mut w = Array1::<f64>::zeros(n);
        let mut w1 = Array1::<f64>::zeros(n);
        let mut w2 = Array1::<f64>::zeros(n);
        let mut w3 = Array1::<f64>::zeros(n);
        let mut w4 = Array1::<f64>::zeros(n);
        for i in 0..n {
            let ei = eta[i];
            let (wi, wi1, wi2, wi3, wi4) = RemlState::logit_fisher_weight_derivatives(ei);
            w[i] = wi;
            w1[i] = wi1;
            w2[i] = wi2;
            w3[i] = wi3;
            w4[i] = wi4;
        }
        let basis_design = if let Some(scale) = observation_weight_sqrt.as_ref() {
            RemlState::row_scale(x_dense, scale)
        } else {
            x_dense.clone()
        };

        // Build one orthonormal coefficient-space basis Q for the identifiable
        // subspace of A^{1/2} X (A = I without fixed case weights).
        //
        // This must happen even when the weighted design is full rank. The old
        // Q = I shortcut left full-rank designs in raw coordinates while the
        // singular branch switched to an orthonormal identifiable basis, so the
        // reduced Fisher determinant depended on which branch we took rather
        // than only on the identifiable subspace itself.
        //
        // Using the retained eigenspace of X̃ᵀ X̃ for every design keeps both
        // branches on one representation:
        //   X̃ = A^{1/2} X,
        //   Qᵀ Q = I,
        //   X_r = X̃ Q,
        //   S_r = X_rᵀ X_r = diag(positive spectrum of X̃ᵀ X̃).
        let gram = fast_atb(&basis_design, &basis_design);
        let (q_basis, metric_spectrum) = Self::identifiable_subspace_basis_from_gram(&gram)?;
        let x_reduced = fast_ab(&basis_design, &q_basis);
        let r = q_basis.ncols();

        // Reduced Fisher on the identifiable subspace:
        //   I_r = X_rᵀ W X_r.
        // Under finite-logit eta, W has strictly positive diagonal entries and
        // X_r has full column rank by construction, so I_r is SPD.
        let fisher_reduced = crate::faer_ndarray::fast_xt_diag_x(&x_reduced, &w);
        // Smooth-regime diagnostic:
        // for exact pseudodet derivatives we require I_r to stay SPD on the
        // fixed identifiable subspace. Emit a warning when I_r appears close
        // to singular, because this is where Jeffreys/Firth curvature becomes
        // numerically fragile and active-subspace assumptions may fail.
        if let Ok((eigvals_ir, _)) = fisher_reduced.eigh(Side::Lower) {
            let max_ev = eigvals_ir.iter().copied().fold(0.0_f64, f64::max).max(1.0);
            let min_ev = eigvals_ir
                .iter()
                .copied()
                .filter(|v| v.is_finite() && *v > 0.0)
                .fold(f64::INFINITY, f64::min);
            if min_ev.is_finite() {
                let rel = min_ev / max_ev;
                if rel < 1e-10 {
                    log::warn!(
                        "[REML/Firth] reduced Fisher I_r is near-singular (min/max={:.3e}/{:.3e}, rel={:.3e}); exact derivatives may be ill-conditioned near active-subspace boundaries.",
                        min_ev,
                        max_ev,
                        rel
                    );
                }
            }
        }

        let mut k_reduced = Array2::<f64>::zeros((r, r));
        let mut x_metric_reduced_inv_diag = Array1::<f64>::zeros(r);
        let mut half_log_det = 0.0_f64;
        if r > 0 {
            // The fixed-Q identifiable-space value is the generalized
            // determinant 0.5(log|I_r| - log|S_r|), which is equivalent to
            // evaluating W on the orthonormalized design U = X_r S_r^{-1/2}.
            //
            // Because Q diagonalizes X̃ᵀ X̃ by construction, S_r is exactly the
            // retained positive spectrum of that Gram matrix in these reduced
            // coordinates, so its inverse/logdet are diagonal operations here.
            // Prefer the fast SPD path for I_r, but when I_r is only
            // numerically semidefinite after projection, keep the exact
            // positive eigenspace instead of failing outright.
            if let Ok(chol) = fisher_reduced.cholesky(Side::Lower) {
                half_log_det = chol.diag().iter().map(|d| d.ln()).sum::<f64>();
                for col in 0..r {
                    let mut e_col = Array1::<f64>::zeros(r);
                    e_col[col] = 1.0;
                    let solved = chol.solvevec(&e_col);
                    k_reduced.column_mut(col).assign(&solved);
                }
            } else {
                let (evals_ir, evecs_ir) = fisher_reduced
                    .eigh(Side::Lower)
                    .map_err(EstimationError::EigendecompositionFailed)?;
                let max_eval = evals_ir.iter().copied().fold(0.0_f64, f64::max).max(1.0);
                let tol = (r.max(1) as f64) * f64::EPSILON * max_eval;
                let keep: Vec<usize> = evals_ir
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &v)| if v > tol { Some(i) } else { None })
                    .collect();
                if keep.is_empty() {
                    return Err(EstimationError::ModelIsIllConditioned {
                        condition_number: f64::INFINITY,
                    });
                }
                for &eig_idx in &keep {
                    let eig = evals_ir[eig_idx];
                    half_log_det += 0.5 * eig.ln();
                    let inv = eig.recip();
                    let vec = evecs_ir.column(eig_idx).to_owned();
                    for row in 0..r {
                        for col in 0..r {
                            k_reduced[[row, col]] += inv * vec[row] * vec[col];
                        }
                    }
                }
            }
            for col in 0..r {
                let metric_eig = metric_spectrum[col];
                half_log_det -= 0.5 * metric_eig.ln();
                x_metric_reduced_inv_diag[col] = metric_eig.recip();
            }
        }
        // Reduced design enters M = Z K_r Z' and P = M⊙M.
        // Without fixed observation weights, Z = X_r. With fixed observation
        // weights a_i, Z = diag(sqrt(a_i)) X_r.
        let z_reduced = x_reduced.clone();
        let h_diag = if r > 0 {
            RemlState::reduced_diag_gram(&z_reduced, &k_reduced)
        } else {
            Array1::<f64>::zeros(n)
        };
        let x_dense_t = x_dense.t().to_owned();
        let b_base = RemlState::row_scale(x_dense, &w1);
        let p_b_base =
            RemlState::apply_hadamard_gram_to_matrix(&z_reduced, &k_reduced, &k_reduced, &b_base);
        Ok(FirthDenseOperator {
            x_dense: x_dense.clone(),
            x_dense_t,
            q_basis,
            x_reduced,
            z_reduced,
            observation_weight_sqrt,
            k_reduced,
            x_metric_reduced_inv_diag,
            half_log_det,
            h_diag,
            w,
            w1,
            w2,
            w3,
            w4,
            b_base,
            p_b_base,
        })
    }

    #[inline]
    pub(crate) fn jeffreys_logdet(&self) -> f64 {
        self.half_log_det
    }

    #[inline]
    pub(crate) fn jeffreys_beta_gradient(&self) -> Array1<f64> {
        // For I(β) = Xᵀ A W(η) X with fixed observation weights A,
        //   ∂/∂β_j [0.5 log|I|]
        //   = 0.5 Σ_i h_i w_i'(η_i) x_{ij},
        // where h_i = [A^{1/2} X I^{-1} Xᵀ A^{1/2}]_{ii}.
        0.5 * self.x_dense_t.dot(&(&self.w1 * &self.h_diag))
    }

    #[inline]
    pub(crate) fn jeffreys_logdet_and_beta_gradient(&self) -> (f64, Array1<f64>) {
        (self.jeffreys_logdet(), self.jeffreys_beta_gradient())
    }

    #[inline]
    fn reduce_explicit_design(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut reduced = fast_ab(x, &self.q_basis);
        if let Some(scale) = self.observation_weight_sqrt.as_ref() {
            reduced = RemlState::row_scale(&reduced, scale);
        }
        reduced
    }

    pub(crate) fn direction_from_deta(&self, deta: Array1<f64>) -> FirthDirection {
        // Directional building blocks for u:
        //   δη_u = X u
        //   I_u  = Xᵀ diag(w' ⊙ δη_u) X
        //   T_u  = K I_u K
        //   N_u  = X T_u Xᵀ
        //   Dh[u] = -diag(N_u)
        //   P_u   = D(M⊙M)[u] = -2(M⊙N_u)
        // and B_u = diag(w'' ⊙ δη_u) X.
        //
        // In this implementation, active-subspace ambiguity is removed by fixed Q:
        // K is represented by K_r = I_r^{-1} in reduced coordinates, so
        //   A_u = K_r G_u K_r
        // is exact for logit with finite eta (w_i > 0) and fixed rank(X).
        // s_u is the diagonal weight for D I[u]:
        //   D I[u] = Xᵀ diag(s_u) X,  s_u = w' ⊙ (X u).
        let s_u = &self.w1 * &deta;
        // G_u = X_rᵀ diag(s_u) X_r,  A_u = K_r G_u K_r.
        // These are reduced-space forms of
        //   I_u and T_u = K I_u K
        // from the full-space derivation.
        let g_u_reduced = RemlState::reducedweighted_gram(&self.x_reduced, &s_u);
        let k_g_u = self.k_reduced.dot(&g_u_reduced);
        let a_u_reduced = k_g_u.dot(&self.k_reduced);
        // Dh[u] = -diag(N_u),  N_u = X T_u Xᵀ, represented here as
        //   N_u = Z A_u Zᵀ in weighted reduced coordinates.
        let dh = -RemlState::reduced_diag_gram(&self.z_reduced, &a_u_reduced);
        let b_uvec = &self.w2 * &deta;
        FirthDirection {
            deta,
            g_u_reduced,
            a_u_reduced,
            dh,
            b_uvec,
        }
    }

    #[inline]
    fn left_scaled_xt(&self, scale: &Array1<f64>, mat: &Array2<f64>) -> Array2<f64> {
        self.x_dense_t
            .dot(&(mat * &scale.view().insert_axis(Axis(1))))
    }

    #[inline]
    fn apply_p_u_to_matrix(&self, a_u_reduced: &Array2<f64>, mat: &Array2<f64>) -> Array2<f64> {
        let mut out = RemlState::apply_hadamard_gram_to_matrix(
            &self.z_reduced,
            &self.k_reduced,
            a_u_reduced,
            mat,
        );
        out.mapv_inplace(|v| -2.0 * v);
        out
    }

    pub(crate) fn hphi_direction_apply(
        &self,
        dir: &FirthDirection,
        rhs: &Array2<f64>,
    ) -> Array2<f64> {
        let p = self.x_dense.ncols();
        if rhs.nrows() != p {
            return Array2::<f64>::zeros((p, rhs.ncols()));
        }
        if rhs.ncols() == 0 || p == 0 {
            return Array2::<f64>::zeros((p, rhs.ncols()));
        }
        // Matrix-free apply of D(Hphi)[u] to a block V:
        //   D(Hphi)[u] V
        // = 0.5[ Xᵀ(c_u ⊙ (X V))
        //       - B_uᵀ P (B V) - Bᵀ P (B_u V) - Bᵀ P_u (B V) ].
        // This avoids dense p×p materialization and is used by sparse exact
        // trace contractions through tr(H^{-1} ·).
        let etav = self.x_dense.dot(rhs);
        let qv = &etav * &self.w1.view().insert_axis(Axis(1));
        let m_qv = RemlState::apply_hadamard_gram_to_matrix(
            &self.z_reduced,
            &self.k_reduced,
            &self.k_reduced,
            &qv,
        );
        let buvec = &dir.b_uvec;
        let m_buv = RemlState::apply_hadamard_gram_to_matrix(
            &self.z_reduced,
            &self.k_reduced,
            &self.k_reduced,
            &(&etav * &buvec.view().insert_axis(Axis(1))),
        );
        let p_u_qv = self.apply_p_u_to_matrix(&dir.a_u_reduced, &qv);
        let c_u = &(&self.w3 * &dir.deta) * &self.h_diag + &(&self.w2 * &dir.dh);
        let diag_term = self
            .x_dense_t
            .dot(&(&etav * &c_u.view().insert_axis(Axis(1))));
        let term1 = self.left_scaled_xt(&buvec, &m_qv);
        let term2 = self.left_scaled_xt(&self.w1, &m_buv);
        let term3 = self.left_scaled_xt(&self.w1, &p_u_qv);
        0.5 * (diag_term - (term1 + term2 + term3))
    }

    pub(crate) fn hphi_direction(&self, dir: &FirthDirection) -> Array2<f64> {
        let p = self.x_dense.ncols();
        let eye = Array2::<f64>::eye(p);
        let mut out = self.hphi_direction_apply(dir, &eye);
        // Exact first directional derivative of H_φ:
        //   D H_φ[u]
        //   = 0.5 [ Xᵀ diag(c_u) X
        //           - (B_uᵀ P B + Bᵀ P B_u + Bᵀ P_u B) ],
        // where
        //   c_u = w''' ⊙ δη_u ⊙ h + w'' ⊙ Dh[u],
        //   B   = diag(w') X.
        //
        // Matrix-free contraction map used below:
        //   Bᵀ P B      via apply_hadamard_gram_to_matrix(Z, K_r, K_r, B)
        //   Bᵀ P_u B    via apply_hadamard_gram_to_matrix(Z, K_r, A_u, B), then *(-2)
        // where P = M⊙M and P_u = -2(M⊙N_u), but M/N_u are never formed explicitly.
        for i in 0..out.nrows() {
            for j in 0..i {
                let avg = 0.5 * (out[[i, j]] + out[[j, i]]);
                out[[i, j]] = avg;
                out[[j, i]] = avg;
            }
        }
        out
    }

    pub(crate) fn hphisecond_direction_apply(
        &self,
        u: &FirthDirection,
        v: &FirthDirection,
        rhs: &Array2<f64>,
    ) -> Array2<f64> {
        let p = self.x_dense.ncols();
        if rhs.nrows() != p {
            return Array2::<f64>::zeros((p, rhs.ncols()));
        }
        if rhs.ncols() == 0 || p == 0 {
            return Array2::<f64>::zeros((p, rhs.ncols()));
        }
        // Exact mixed second directional derivative:
        //   D² H_φ[u,v] = 0.5 [ Xᵀ diag(c_uv) X - D²J₂[u,v] ], J₂ = Bᵀ P B.
        // Implemented with matrix identities for N_{u,v}, P_{u,v}, and the
        // nine-term expansion of D²J₂[u,v].
        //
        // Because we parameterize Phi through fixed-rank I_r = X_rᵀ W X_r (SPD for
        // finite-logit eta), this mixed derivative is evaluated on a smooth
        // manifold without dynamic active-set switching in the Firth block.
        //
        // The nine contraction terms below are the explicit D²J₂[u,v] expansion,
        // each computed through reduced Hadamard-Gram operators.
        let deta_uv = &u.deta * &v.deta;
        // Mixed reduced Gram:
        //   G_uv = X_rᵀ diag(w'' ⊙ (Xu) ⊙ (Xv)) X_r.
        let s_uv = &self.w2 * &deta_uv;
        let g_uv_reduced = RemlState::reducedweighted_gram(&self.x_reduced, &s_uv);
        let k_g_uv = self.k_reduced.dot(&g_uv_reduced);
        let k_gv = self.k_reduced.dot(&v.g_u_reduced);
        let k_g_u = self.k_reduced.dot(&u.g_u_reduced);
        // Reduced form of:
        //   T_{u,v} = K I_{u,v} K - K Iv K I_u K - K I_u K Iv K.
        let a_uv_reduced = k_g_uv.dot(&self.k_reduced)
            - k_gv.dot(&k_g_u).dot(&self.k_reduced)
            - k_g_u.dot(&k_gv).dot(&self.k_reduced);
        let d2h = -RemlState::reduced_diag_gram(&self.z_reduced, &a_uv_reduced);
        // Implements mixed diagonal coefficient:
        //   c_uv = w'''' ⊙ (Xu) ⊙ (Xv) ⊙ h
        //          + w''' ⊙ ((Xu) ⊙ Dh[v] + (Xv) ⊙ Dh[u])
        //          + w'' ⊙ D²h[u,v].
        let c_uv = &(&(&self.w4 * &deta_uv) * &self.h_diag)
            + &(&self.w3 * &(&u.deta * &v.dh))
            + &(&self.w3 * &(&v.deta * &u.dh))
            + &(&self.w2 * &d2h);

        let eta_rhs = self.x_dense.dot(rhs);
        let diag_term = self
            .x_dense_t
            .dot(&(&eta_rhs * &c_uv.view().insert_axis(Axis(1))));

        let b_uvvec = &self.w3 * &deta_uv;
        let b_uv_base = &self.x_dense * &b_uvvec.view().insert_axis(Axis(1));
        let qv = &eta_rhs * &self.w1.view().insert_axis(Axis(1));

        // Linearity in the rhs argument lets us precompute the expensive
        // Hadamard-Gram operator on the full base blocks B, B_u, Bv, B_uv once,
        // then post-multiply by rhs. This preserves the exact operator while
        // avoiding repeated O(n r^2 c) work for every rhs block.
        let p_b_rhs = fast_ab(&self.p_b_base, rhs);
        let p_bu_rhs = RemlState::apply_hadamard_gram_to_matrix(
            &self.z_reduced,
            &self.k_reduced,
            &self.k_reduced,
            &(&eta_rhs * &u.b_uvec.view().insert_axis(Axis(1))),
        );
        let p_bv_rhs = RemlState::apply_hadamard_gram_to_matrix(
            &self.z_reduced,
            &self.k_reduced,
            &self.k_reduced,
            &(&eta_rhs * &v.b_uvec.view().insert_axis(Axis(1))),
        );
        let p_buv_base = RemlState::apply_hadamard_gram_to_matrix(
            &self.z_reduced,
            &self.k_reduced,
            &self.k_reduced,
            &b_uv_base,
        );
        let p_buv_rhs = fast_ab(&p_buv_base, rhs);

        let pv_b_rhs = self.apply_p_u_to_matrix(&v.a_u_reduced, &qv);
        let pv_bu_rhs = self.apply_p_u_to_matrix(
            &v.a_u_reduced,
            &(&eta_rhs * &u.b_uvec.view().insert_axis(Axis(1))),
        );
        let p_u_b_rhs = self.apply_p_u_to_matrix(&u.a_u_reduced, &qv);
        let p_u_bv_rhs = self.apply_p_u_to_matrix(
            &u.a_u_reduced,
            &(&eta_rhs * &v.b_uvec.view().insert_axis(Axis(1))),
        );

        let p_nu_nv_base = RemlState::apply_hadamard_gram_to_matrix(
            &self.z_reduced,
            &u.a_u_reduced,
            &v.a_u_reduced,
            &self.b_base,
        );
        let p_hw_nuv_base = RemlState::apply_hadamard_gram_to_matrix(
            &self.z_reduced,
            &self.k_reduced,
            &a_uv_reduced,
            &self.b_base,
        );
        let p_uv_base = 2.0 * p_nu_nv_base - 2.0 * p_hw_nuv_base;
        let p_uv_rhs = fast_ab(&p_uv_base, rhs);

        // Nine-term expansion of D²J₂[u,v] with J₂ = Bᵀ P B.
        let d2_terms = [
            self.left_scaled_xt(&b_uvvec, &p_b_rhs),
            self.left_scaled_xt(&self.w1, &p_buv_rhs),
            self.left_scaled_xt(&u.b_uvec, &p_bv_rhs),
            self.left_scaled_xt(&v.b_uvec, &p_bu_rhs),
            self.left_scaled_xt(&u.b_uvec, &pv_b_rhs),
            self.left_scaled_xt(&self.w1, &pv_bu_rhs),
            self.left_scaled_xt(&v.b_uvec, &p_u_b_rhs),
            self.left_scaled_xt(&self.w1, &p_u_bv_rhs),
            self.left_scaled_xt(&self.w1, &p_uv_rhs),
        ];
        let mut d2_j2 = Array2::<f64>::zeros((p, rhs.ncols()));
        for term in d2_terms {
            d2_j2 += &term;
        }

        0.5 * (diag_term - d2_j2)
    }

    pub(super) fn rowwise_dot(a: &Array2<f64>, b: &Array2<f64>) -> Array1<f64> {
        debug_assert_eq!(a.nrows(), b.nrows());
        debug_assert_eq!(a.ncols(), b.ncols());
        let mut out = Array1::<f64>::zeros(a.nrows());
        for i in 0..a.nrows() {
            let mut acc = 0.0_f64;
            for j in 0..a.ncols() {
                acc += a[[i, j]] * b[[i, j]];
            }
            out[i] = acc;
        }
        out
    }

    pub(super) fn rowwise_bilinear(
        a: &Array2<f64>,
        m: &Array2<f64>,
        b: &Array2<f64>,
    ) -> Array1<f64> {
        // Returns vector with entries a_iᵀ M b_i for each row i.
        debug_assert_eq!(a.nrows(), b.nrows());
        debug_assert_eq!(a.ncols(), m.nrows());
        debug_assert_eq!(b.ncols(), m.ncols());
        let am = fast_ab(a, m);
        Self::rowwise_dot(&am, b)
    }

    fn dot_i_and_h_from_reduced(
        &self,
        x_tau_reduced: &Array2<f64>,
        deta: &Array1<f64>,
    ) -> (Array2<f64>, Array1<f64>) {
        // Reduced Fisher directional derivative under fixed identifiable basis:
        //   I_r = X_r' W X_r
        //   I_r,tau = X_{r,tau}' W X_r + X_r' W X_{r,tau} + X_r' W_tau X_r
        // with W_tau = diag(w' ⊙ eta_tau).
        //
        // Leverage derivative used by Firth score partial:
        //   h_i = x_{r,i}' K_r x_{r,i}, K_r = I_r^{-1}
        //   h_tau = 2*diag(X_{r,tau} K_r X_r') + diag(X_r K_{r,tau} X_r')
        //   K_{r,tau} = -K_r I_{r,tau} K_r.
        //
        // This is exactly the fixed-beta directional derivative required by
        //   (gphi)_tau and Phi_tau in the Jeffreys/Firth design-moving path:
        //   I_{r,tau}|beta = X_{r,tau}' W X_r + X_r' W X_{r,tau}
        //                    + X_r' diag(w' ⊙ eta_tau|beta) X_r,
        //   eta_tau|beta = X_tau beta.
        //
        // We return:
        //   dot_i  = I_{r,tau}|beta,
        //   dot_h  = h_tau|beta.
        let dw = &self.w1 * deta;
        let dot_i = RemlState::weighted_cross(x_tau_reduced, &self.x_reduced, &self.w)
            + RemlState::weighted_cross(&self.x_reduced, x_tau_reduced, &self.w)
            + crate::faer_ndarray::fast_xt_diag_x(&self.x_reduced, &dw);

        let dot_k = -self.k_reduced.dot(&dot_i).dot(&self.k_reduced);
        let x_tauk = x_tau_reduced.dot(&self.k_reduced);
        let dot_h_explicit = 2.0 * Self::rowwise_dot(&x_tauk, &self.x_reduced);
        let dot_h_implicit = Self::rowwise_dot(&self.x_reduced.dot(&dot_k), &self.x_reduced);
        let dot_h = dot_h_explicit + dot_h_implicit;
        (dot_i, dot_h)
    }

    pub(crate) fn exact_tau_kernel(
        &self,
        x_tau: &Array2<f64>,
        beta: &Array1<f64>,
        include_hphi_tau_kernel: bool,
    ) -> FirthTauExactKernel {
        // Shared exact tau-partial bundle used by both dense and sparse paths:
        //   (gphi)_tau | beta-fixed,
        //   Phi_tau | beta-fixed,
        // and optional H_{phi,tau}|beta kernel for later matrix-free applies.
        //
        // Closed forms (reduced Fisher, fixed active subspace):
        //   Phi = 0.5 log|I_r| - 0.5 log|S_r|,
        //   I_r = X_r' W X_r, K_r = I_r^{-1},
        //   S_r = X_r' X_r,   diag(G_r) = diag(S_r^{-1}),
        //   Phi_tau|beta = 0.5 tr(K_r I_{r,tau}) - 0.5 tr(G_r S_{r,tau}).
        // In the canonical reduced basis used here, G_r is diagonal.
        //
        //   (gphi)_tau = Phi_beta,tau
        //               = 0.5 X_tau' (w1 .* h)
        //                 + 0.5 X'((w2 .* eta_tau) .* h + w1 .* h_tau),
        //   where
        //     h_i = x_{r,i}' K_r x_{r,i},
        //     h_tau = 2*diag(X_{r,tau} K_r X_r') + diag(X_r K_{r,tau} X_r'),
        //     K_{r,tau} = -K_r I_{r,tau} K_r.
        //
        // Phi_beta,tau is unchanged by the -0.5 log|S_r| term because S_r does
        // not depend on beta. Only Phi_tau gets the explicit basis-drift
        // subtraction.
        let deta_partial = x_tau.dot(beta);
        let x_tau_reduced = self.reduce_explicit_design(x_tau);
        let (dot_i_partial, dot_h_partial) =
            self.dot_i_and_h_from_reduced(&x_tau_reduced, &deta_partial);
        let dot_s_partial =
            fast_atb(&x_tau_reduced, &self.x_reduced) + fast_atb(&self.x_reduced, &x_tau_reduced);

        let first = 0.5 * x_tau.t().dot(&(&self.w1 * &self.h_diag));
        let secondvec =
            &(&(&self.w2 * &deta_partial) * &self.h_diag) + &(&self.w1 * &dot_h_partial);
        let second = 0.5 * self.x_dense.t().dot(&secondvec);
        let gphi_tau = first + second;
        let phi_tau_partial = 0.5 * RemlState::trace_product(&self.k_reduced, &dot_i_partial)
            - 0.5 * Self::trace_diag_product(&self.x_metric_reduced_inv_diag, &dot_s_partial);

        let tau_kernel = if include_hphi_tau_kernel {
            Some(self.hphi_tau_partial_prepare_from_partials(
                x_tau_reduced,
                &deta_partial,
                dot_h_partial,
                dot_i_partial,
            ))
        } else {
            None
        };
        FirthTauExactKernel {
            gphi_tau,
            phi_tau_partial,
            tau_kernel,
        }
    }

    fn hphi_tau_partial_prepare_from_partials(
        &self,
        x_tau_reduced: Array2<f64>,
        deta_partial: &Array1<f64>,
        dot_h_partial: Array1<f64>,
        dot_i_partial: Array2<f64>,
    ) -> FirthTauPartialKernel {
        let dotw1 = &self.w2 * deta_partial;
        let dotw2 = &self.w3 * deta_partial;
        let dot_k = -self.k_reduced.dot(&dot_i_partial).dot(&self.k_reduced);
        FirthTauPartialKernel {
            dotw1,
            dotw2,
            dot_h_partial,
            x_tau_reduced,
            dot_k_reduced: dot_k,
        }
    }

    pub(crate) fn apply_pbar_to_matrix(&self, mat: &Array2<f64>) -> Array2<f64> {
        // Applies P̄ = (X_r K_r X_rᵀ)⊙(X_r K_r X_rᵀ) to each column of mat.
        RemlState::apply_hadamard_gram_to_matrix(
            &self.z_reduced,
            &self.k_reduced,
            &self.k_reduced,
            mat,
        )
    }

    pub(crate) fn apply_mtau_to_matrix(
        &self,
        kernel: &FirthTauPartialKernel,
        mat: &Array2<f64>,
    ) -> Array2<f64> {
        // Exact apply of
        //   M_tau = d/dtau[(P⊙P)]|_{beta fixed} = 2(P⊙P_tau)
        // without building dense n×n objects.
        //
        // Decomposition:
        //   P = Z K Zᵀ, Z = X_r
        //   P_tau = Z_tau K Zᵀ + Z K Z_tauᵀ + Z dotK Zᵀ
        // and for each vector v:
        //   (P⊙(Z_tau K Zᵀ))v   : rowwise bilinear with K (Zᵀdiag(v)Z) K
        //   (P⊙(Z K Z_tauᵀ))v   : diag_Z( K (Zᵀdiag(v)Z_tau) K )
        //   (P⊙(Z dotK Zᵀ))v    : Hadamard-Gram apply with (K, dotK).
        if mat.nrows() != self.x_dense.nrows() || mat.ncols() == 0 {
            return Array2::<f64>::zeros(mat.raw_dim());
        }
        let mut out = Array2::<f64>::zeros(mat.raw_dim());
        for col in 0..mat.ncols() {
            let v = mat.column(col).to_owned();
            let szz = RemlState::reducedweighted_gram(&self.z_reduced, &v);
            let mzz = self.k_reduced.dot(&szz).dot(&self.k_reduced);
            let t1 = Self::rowwise_bilinear(&self.z_reduced, &mzz, &kernel.x_tau_reduced);

            let szt =
                RemlState::reduced_crossweighted_gram(&self.z_reduced, &kernel.x_tau_reduced, &v);
            let mzt = self.k_reduced.dot(&szt).dot(&self.k_reduced);
            let t2 = RemlState::reduced_diag_gram(&self.z_reduced, &mzt);

            let t3 = RemlState::apply_hadamard_gram(
                &self.z_reduced,
                &self.k_reduced,
                &kernel.dot_k_reduced,
                &v,
            );

            let y = 2.0 * (t1 + t2 + t3);
            out.column_mut(col).assign(&y);
        }
        out
    }

    pub(crate) fn hphi_tau_partial_apply(
        &self,
        x_tau: &Array2<f64>,
        kernel: &FirthTauPartialKernel,
        rhs: &Array2<f64>,
    ) -> Array2<f64> {
        let p = self.x_dense.ncols();
        if rhs.nrows() != p {
            return Array2::<f64>::zeros((p, rhs.ncols()));
        }
        if rhs.ncols() == 0 || p == 0 {
            return Array2::<f64>::zeros((p, rhs.ncols()));
        }
        // Matrix-free block apply of Hphi,tau|beta:
        //   Hphi,tau|beta(V) = 0.5 [ X_tau' r(V) + X' r_tau(V) ].
        //
        // Tensor identity behind this apply:
        //   Hphi,tau|beta = Phi_beta,beta,tau
        // and for test vectors b1,b2 (matrix columns V are batched b2's):
        //   Phi_beta,beta,tau[b1,b2]
        //   = 0.5[
        //       tr(I^{-1} I_{b1,b2,tau})
        //       - tr(I^{-1} I_{b1,b2} I^{-1} I_tau)
        //       - tr(I^{-1} I_{b1,tau} I^{-1} I_{b2})
        //       - tr(I^{-1} I_{b2,tau} I^{-1} I_{b1})
        //       + 2 tr(I^{-1} I_{b1} I^{-1} I_{b2} I^{-1} I_tau)
        //     ].
        // This routine evaluates that form in reduced coordinates without forming
        // dense 3rd-order tensors explicitly.
        let etav = self.x_dense.dot(rhs);
        let etav_tau = x_tau.dot(rhs);
        let qv = &etav * &self.w1.view().insert_axis(Axis(1));
        let qv_tau = &etav * &kernel.dotw1.view().insert_axis(Axis(1))
            + &etav_tau * &self.w1.view().insert_axis(Axis(1));
        let m_qv = self.apply_pbar_to_matrix(&qv);
        let m_qv_tau = self.apply_mtau_to_matrix(kernel, &qv) + self.apply_pbar_to_matrix(&qv_tau);
        let rv = &(&etav * &self.w2.view().insert_axis(Axis(1)))
            * &self.h_diag.view().insert_axis(Axis(1))
            - &(&m_qv * &self.w1.view().insert_axis(Axis(1)));
        let rv_tau = (&(&etav * &kernel.dotw2.view().insert_axis(Axis(1)))
            + &(&etav_tau * &self.w2.view().insert_axis(Axis(1))))
            * &self.h_diag.view().insert_axis(Axis(1))
            + &(&etav * &self.w2.view().insert_axis(Axis(1)))
                * &kernel.dot_h_partial.view().insert_axis(Axis(1))
            - &(&m_qv * &kernel.dotw1.view().insert_axis(Axis(1))
                + &m_qv_tau * &self.w1.view().insert_axis(Axis(1)));
        0.5 * (x_tau.t().dot(&rv) + self.x_dense.t().dot(&rv_tau))
    }

    // ══════════════════════════════════════════════════════════════════════
    //  PRIMITIVE A — pair τ_i × τ_j Firth drift.
    //
    //  Purpose.  Evaluate `Hφ_{τ_iτ_j}|β · V` for a p×m test-vector block V,
    //  where Hφ = ∇²_β Φ and Φ is the Jeffreys log-determinant
    //  `0.5 log|I_r|`.  Matrix-free in n.
    //
    //  Derivation (chain-rule extension of the single-τ primitive
    //  `hphi_tau_partial_apply`).  Let, at fixed β and current X:
    //     η_V   = X · V          (n×m)          η_{V,α} = X_α · V
    //     η̇_α  = X_α · β                       η̈_{ij} = X_{ij} · β
    //     w1..w4 = logistic weight η-jets
    //     h     = diag(X_r K_r X_rᵀ)            ḣ_α = ∂_α h
    //     qv    = w1 ⊙ η_V                     ẇ1_α = w2 ⊙ η̇_α
    //     m_qv  = (M ⊙ M) · qv = P̄ · qv       Ṁ_α = 2(M ⊙ M_α)
    //  and the single-τ Firth-Hessian drift body reads
    //     Hφ_{τ_j}|β · V = 0.5 [X_{τ_j}ᵀ · r_V + Xᵀ · r_{V,τ_j}]
    //  with
    //     r_V       = (w2 ⊙ η_V) ⊙ h − m_qv ⊙ w1
    //     r_{V,τ_j} = (ẇ2_j ⊙ η_V + w2 ⊙ η_{V,τ_j}) ⊙ h
    //                + (w2 ⊙ η_V) ⊙ ḣ_j
    //                − (m_qv ⊙ ẇ1_j + m_qv_{τ_j} ⊙ w1)
    //
    //  Taking ∂_{τ_i} at fixed β (Schwarz-symmetric by construction in Φ):
    //     Hφ_{τ_iτ_j}|β · V
    //       = 0.5 [
    //             X_{τ_iτ_j}ᵀ · r_V                  (outer A — zero if X_{ij}=None)
    //           + X_{τ_j}ᵀ · ∂_{τ_i} r_V             (outer B)
    //           + X_{τ_i}ᵀ · r_{V,τ_j}               (outer C)
    //           + Xᵀ · ∂_{τ_i} r_{V,τ_j}             (outer D)
    //         ]
    //
    //  The two inner τ_i-derivatives expand mechanically:
    //
    //  ∂_{τ_i} r_V
    //    = ((w3 ⊙ η̇_i) ⊙ η_V + w2 ⊙ η_{V,τ_i}) ⊙ h
    //    + (w2 ⊙ η_V) ⊙ ḣ_i
    //    − [ (Ṁ_i · qv + P̄ · ∂_i qv) ⊙ w1 + m_qv ⊙ (w2 ⊙ η̇_i) ]
    //    with ∂_i qv = (w2 ⊙ η̇_i) ⊙ η_V + w1 ⊙ η_{V,τ_i}.
    //
    //  ∂_{τ_i} r_{V,τ_j} — a sum of four ∂_i pieces on the four r_{V,τ_j}
    //  additive terms:
    //  (1) ∂_i[(ẇ2_j ⊙ η_V) ⊙ h]
    //      = ((w4·η̇_i·η̇_j + w3·η̈_ij) ⊙ η_V + ẇ2_j ⊙ η_{V,τ_i}) ⊙ h
    //      + (ẇ2_j ⊙ η_V) ⊙ ḣ_i
    //  (2) ∂_i[(w2 ⊙ η_{V,τ_j}) ⊙ h]
    //      = ((w3 ⊙ η̇_i) ⊙ η_{V,τ_j} + w2 ⊙ η_{V,τ_iτ_j}) ⊙ h
    //      + (w2 ⊙ η_{V,τ_j}) ⊙ ḣ_i
    //  (3) ∂_i[(w2 ⊙ η_V) ⊙ ḣ_j]
    //      = ((w3 ⊙ η̇_i) ⊙ η_V + w2 ⊙ η_{V,τ_i}) ⊙ ḣ_j
    //      + (w2 ⊙ η_V) ⊙ ḧ_{ij}
    //  (4) ∂_i[m_qv ⊙ ẇ1_j + m_qv_{τ_j} ⊙ w1]
    //      = (Ṁ_i · qv + P̄ · ∂_i qv) ⊙ ẇ1_j
    //      + m_qv ⊙ (w3·η̇_i·η̇_j + w2·η̈_ij)
    //      + ( M̈_{ij}·qv + Ṁ_j·∂_i qv + Ṁ_i·qv_{τ_j} + P̄·∂_i qv_{τ_j} ) ⊙ w1
    //      + m_qv_{τ_j} ⊙ (w2 ⊙ η̇_i)
    //  with ∂_i qv_{τ_j}
    //      = (w3·η̇_i·η̇_j + w2·η̈_ij) ⊙ η_V
    //      + ẇ1_j ⊙ η_{V,τ_i} + ẇ1_i ⊙ η_{V,τ_j}
    //      + w1 ⊙ η_{V,τ_iτ_j}
    //
    //  The M̈_{ij} apply reuses the same Hadamard-Gram machinery as
    //  apply_mtau_to_matrix, specialized to the second-derivative pattern:
    //     M̈_{ij} · v = 2 (M_τ_i ⊙ M_τ_j) · v  +  2 (M ⊙ M_{τ_iτ_j}) · v
    //  with
    //     M_α          = X_{r,α} K X_rᵀ + X_r K X_{r,α}ᵀ + X_r K̇_α X_rᵀ,
    //     M_{τ_iτ_j}   = X_{r,ij} K X_rᵀ + X_r K X_{r,ij}ᵀ + X_{r,i} K X_{r,j}ᵀ
    //                   + X_{r,j} K X_{r,i}ᵀ + X_{r,i} K̇_j X_rᵀ
    //                   + X_r K̇_j X_{r,i}ᵀ + X_{r,j} K̇_i X_rᵀ
    //                   + X_r K̇_i X_{r,j}ᵀ + X_r K̈_{ij} X_rᵀ.
    //  ═══════════════════════════════════════════════════════════════════

    /// Build the reduced-basis pair-kernel used by `hphi_tau_tau_partial_apply`.
    ///
    /// Takes the two per-direction partial outputs from
    /// `dot_i_and_h_from_reduced` (for τ_i and τ_j), plus an optional
    /// second-design reduced drift `x_tau_tau_reduced` (`None` when the
    /// design is τ-linear, so X_{τ_iτ_j} ≡ 0) and its η̈_{ij} product.
    ///
    /// The returned kernel packages every reusable piece so that `apply`
    /// never recomputes shared Grams.
    #[allow(dead_code)]
    pub(crate) fn hphi_tau_tau_partial_prepare_from_partials(
        &self,
        x_tau_i_reduced: Array2<f64>,
        x_tau_j_reduced: Array2<f64>,
        deta_i: Array1<f64>,
        deta_j: Array1<f64>,
        dot_i_i_reduced: Array2<f64>,
        dot_i_j_reduced: Array2<f64>,
        dot_h_i: Array1<f64>,
        dot_h_j: Array1<f64>,
        x_tau_tau_reduced: Option<Array2<f64>>,
        deta_ij: Option<Array1<f64>>,
    ) -> FirthTauTauPartialKernel {
        //  Build the single-τ kernels first — these are the same as the ones
        //  consumed by the single-τ primitive, sharing its FD-verified
        //  structure.
        let k = &self.k_reduced;
        let dot_k_i = -k.dot(&dot_i_i_reduced).dot(k);
        let dot_k_j = -k.dot(&dot_i_j_reduced).dot(k);
        let dotw1_i = &self.w2 * &deta_i;
        let dotw1_j = &self.w2 * &deta_j;
        let dotw2_i = &self.w3 * &deta_i;
        let dotw2_j = &self.w3 * &deta_j;
        let tau_i = FirthTauPartialKernel {
            dotw1: dotw1_i.clone(),
            dotw2: dotw2_i.clone(),
            dot_h_partial: dot_h_i.clone(),
            x_tau_reduced: x_tau_i_reduced.clone(),
            dot_k_reduced: dot_k_i.clone(),
        };
        let tau_j = FirthTauPartialKernel {
            dotw1: dotw1_j.clone(),
            dotw2: dotw2_j.clone(),
            dot_h_partial: dot_h_j.clone(),
            x_tau_reduced: x_tau_j_reduced.clone(),
            dot_k_reduced: dot_k_j.clone(),
        };

        //  Ï_{r,ij} — 9-term reduced Fisher cross second derivative.
        //    Ï_ij = X_{r,ij}ᵀ W X_r + X_rᵀ W X_{r,ij}
        //          + X_{r,i}ᵀ W X_{r,j} + X_{r,j}ᵀ W X_{r,i}
        //          + X_{r,i}ᵀ Ẇ_j X_r + X_rᵀ Ẇ_j X_{r,i}
        //          + X_{r,j}ᵀ Ẇ_i X_r + X_rᵀ Ẇ_i X_{r,j}
        //          + X_rᵀ Ẅ_{ij} X_r
        //  with Ẇ_α = diag(w' ⊙ η̇_α) and
        //       Ẅ_{ij} = diag(w'' ⊙ η̇_i ⊙ η̇_j  +  w' ⊙ η̈_{ij}).
        let zeros_n = Array1::<f64>::zeros(self.x_dense.nrows());
        let deta_ij_ref: &Array1<f64> = deta_ij.as_ref().unwrap_or(&zeros_n);
        let dw_i = &self.w1 * &deta_i;
        let dw_j = &self.w1 * &deta_j;
        let ddw_ij = &(&self.w2 * &(&deta_i * &deta_j)) + &(&self.w1 * deta_ij_ref);
        let x_r = &self.x_reduced;
        let mut ddot_i_ij = Array2::<f64>::zeros(k.raw_dim());
        if let Some(x_rij) = x_tau_tau_reduced.as_ref() {
            ddot_i_ij = ddot_i_ij + RemlState::weighted_cross(x_rij, x_r, &self.w);
            ddot_i_ij = ddot_i_ij + RemlState::weighted_cross(x_r, x_rij, &self.w);
        }
        ddot_i_ij =
            ddot_i_ij + RemlState::weighted_cross(&x_tau_i_reduced, &x_tau_j_reduced, &self.w);
        ddot_i_ij =
            ddot_i_ij + RemlState::weighted_cross(&x_tau_j_reduced, &x_tau_i_reduced, &self.w);
        ddot_i_ij = ddot_i_ij + RemlState::weighted_cross(&x_tau_i_reduced, x_r, &dw_j);
        ddot_i_ij = ddot_i_ij + RemlState::weighted_cross(x_r, &x_tau_i_reduced, &dw_j);
        ddot_i_ij = ddot_i_ij + RemlState::weighted_cross(&x_tau_j_reduced, x_r, &dw_i);
        ddot_i_ij = ddot_i_ij + RemlState::weighted_cross(x_r, &x_tau_j_reduced, &dw_i);
        ddot_i_ij = ddot_i_ij + crate::faer_ndarray::fast_xt_diag_x(x_r, &ddw_ij);

        //  K̈_{r,ij} = −K Ï_ij K  +  K İ_i K İ_j K  +  K İ_j K İ_i K
        //  Using K İ_α K = −K̇_α (already cached in dot_k_α):
        //     K İ_i K İ_j K = (−K̇_i) İ_j K = −dot_k_i · dot_i_j · K
        //  so K İ_i K İ_j K + K İ_j K İ_i K = −dot_k_i·dot_i_j·K − dot_k_j·dot_i_i·K.
        let ddot_k_ij: Array2<f64> = -k.dot(&ddot_i_ij).dot(k)
            + (-&dot_k_i).dot(&dot_i_j_reduced).dot(k)
            + (-&dot_k_j).dot(&dot_i_i_reduced).dot(k);

        //  ḧ_{ij} = ∂²_{τ_iτ_j} diag(M)
        //         = 2 diag(X_{r,ij} K X_rᵀ)
        //         + diag(X_r K̈_{ij} X_rᵀ)
        //         + 2 diag(X_{r,i} K̇_j X_rᵀ)
        //         + 2 diag(X_{r,j} K̇_i X_rᵀ)
        //         + 2 diag(X_{r,i} K X_{r,j}ᵀ)
        let n = self.x_dense.nrows();
        let mut ddot_h_ij = Array1::<f64>::zeros(n);
        if let Some(x_rij) = x_tau_tau_reduced.as_ref() {
            let rij_k = x_rij.dot(k);
            ddot_h_ij = ddot_h_ij + 2.0 * Self::rowwise_dot(&rij_k, x_r);
        }
        let xr_kddot = x_r.dot(&ddot_k_ij);
        ddot_h_ij = ddot_h_ij + Self::rowwise_dot(&xr_kddot, x_r);
        let ri_kdot_j = x_tau_i_reduced.dot(&dot_k_j);
        ddot_h_ij = ddot_h_ij + 2.0 * Self::rowwise_dot(&ri_kdot_j, x_r);
        let rj_kdot_i = x_tau_j_reduced.dot(&dot_k_i);
        ddot_h_ij = ddot_h_ij + 2.0 * Self::rowwise_dot(&rj_kdot_i, x_r);
        let ri_k = x_tau_i_reduced.dot(k);
        ddot_h_ij = ddot_h_ij + 2.0 * Self::rowwise_dot(&ri_k, &x_tau_j_reduced);

        FirthTauTauPartialKernel {
            tau_i,
            tau_j,
            deta_i,
            deta_j,
            dot_i_i_reduced,
            dot_i_j_reduced,
            ddot_i_ij_reduced: ddot_i_ij,
            ddot_k_ij_reduced: ddot_k_ij,
            dot_h_i,
            dot_h_j,
            ddot_h_ij,
            x_tau_tau_reduced,
            deta_ij,
        }
    }

    /// Apply M̈_{ij} to each column of `mat`.
    ///
    /// M̈_{ij} · v = 2 (M ⊙ M_{τ_iτ_j}) · v  +  2 (M_i ⊙ M_j) · v.
    ///
    /// Both components expand into a sum of Hadamard-Gram applies with the
    /// reduced-basis pairs  (A_left, A_right)  chosen from
    ///   {(K, K), (K, K̇_i), (K, K̇_j), (K̇_i, K̇_j), ...}
    /// plus rowwise-bilinear evaluations analogous to
    /// `apply_mtau_to_matrix` for the X_{r,α} tails.
    ///
    /// The general trick is the identity
    ///   (YAWᵀ ⊙ ZB Zᵀ) · v
    ///     = rowwise_bilinear(Y,  A · (Wᵀ diag(v) Z) · B, Z)
    /// when Y, W, Z are reduced-basis matrices and A, B are r×r symmetric.
    #[allow(dead_code)]
    fn apply_p_ddot_ij_to_matrix(
        &self,
        kernel: &FirthTauTauPartialKernel,
        mat: &Array2<f64>,
    ) -> Array2<f64> {
        let n = self.x_dense.nrows();
        let m = mat.ncols();
        if mat.nrows() != n || m == 0 {
            return Array2::<f64>::zeros(mat.raw_dim());
        }
        let x_r = &self.x_reduced;
        let k = &self.k_reduced;
        let x_ri = &kernel.tau_i.x_tau_reduced;
        let x_rj = &kernel.tau_j.x_tau_reduced;
        let dot_k_i = &kernel.tau_i.dot_k_reduced;
        let dot_k_j = &kernel.tau_j.dot_k_reduced;
        let ddot_k_ij = &kernel.ddot_k_ij_reduced;
        let zero_xr = Array2::<f64>::zeros(x_r.raw_dim());
        let x_rij: &Array2<f64> = kernel.x_tau_tau_reduced.as_ref().unwrap_or(&zero_xr);
        let has_xij = kernel.x_tau_tau_reduced.is_some();

        let mut out = Array2::<f64>::zeros((n, m));
        for col in 0..m {
            let v = mat.column(col).to_owned();

            //  Component 1:  2 (M ⊙ M_{τ_iτ_j}) · v
            //  M_{τ_iτ_j} has 9 tails listed above.  Each summand
            //  (M ⊙ Y_α B_α W_αᵀ) · v yields
            //     rowwise_bilinear(Y_α, B_α · (W_αᵀ diag(v) X_r) · K, X_r)  [when
            //     one side is Z K Zᵀ]
            //  or the crossed  diag(X_r · (K · (X_rᵀ diag(v) W_α) · B_αᵀ) · Y_αᵀ)
            //  variant.  Collect all 9 tails.
            //
            //  The four X_{r,ij}, X_r pairs are handled via
            //  rowwise_bilinear and its transpose.
            let mut t1 = Array1::<f64>::zeros(n);

            //   tail A: Y=X_{r,ij}, A=K, W=X_r       (only if x_rij present)
            //   tail B: Y=X_r,     A=K, W=X_{r,ij}
            if has_xij {
                let szz = RemlState::reduced_crossweighted_gram(x_r, x_rij, &v);
                let mid = k.dot(&szz).dot(k);
                t1 = t1 + Self::rowwise_bilinear(x_rij, &mid, x_r);
                let szz2 = RemlState::reduced_crossweighted_gram(x_rij, x_r, &v);
                let mid2 = k.dot(&szz2).dot(k);
                t1 = t1 + Self::rowwise_bilinear(x_r, &mid2, x_r);
            }

            //   tail C: Y=X_{r,i}, A=K, W=X_{r,j}
            //   tail D: Y=X_{r,j}, A=K, W=X_{r,i}
            let sij = RemlState::reduced_crossweighted_gram(x_r, x_rj, &v);
            let mij = k.dot(&sij).dot(k);
            t1 = t1 + Self::rowwise_bilinear(x_ri, &mij, x_r);
            let sji = RemlState::reduced_crossweighted_gram(x_r, x_ri, &v);
            let mji = k.dot(&sji).dot(k);
            t1 = t1 + Self::rowwise_bilinear(x_rj, &mji, x_r);

            //   tail E: Y=X_{r,i}, A=K̇_j, W=X_r
            //   tail F: Y=X_r,    A=K̇_j, W=X_{r,i}
            let se_raw = RemlState::reducedweighted_gram(x_r, &v);
            let me = k.dot(&se_raw).dot(dot_k_j);
            t1 = t1 + Self::rowwise_bilinear(x_ri, &me, x_r);
            let sf_raw = RemlState::reduced_crossweighted_gram(x_r, x_ri, &v);
            let mf = k.dot(&sf_raw).dot(dot_k_j);
            t1 = t1 + Self::rowwise_bilinear(x_r, &mf, x_r);

            //   tail G: Y=X_{r,j}, A=K̇_i, W=X_r
            //   tail H: Y=X_r,    A=K̇_i, W=X_{r,j}
            let mg = k.dot(&se_raw).dot(dot_k_i);
            t1 = t1 + Self::rowwise_bilinear(x_rj, &mg, x_r);
            let sh_raw = RemlState::reduced_crossweighted_gram(x_r, x_rj, &v);
            let mh = k.dot(&sh_raw).dot(dot_k_i);
            t1 = t1 + Self::rowwise_bilinear(x_r, &mh, x_r);

            //   tail I: Y=X_r,   A=K̈_{ij}, W=X_r   (uses apply_hadamard_gram
            //   with K vs ddot_k_ij).
            t1 = t1 + RemlState::apply_hadamard_gram(x_r, k, ddot_k_ij, &v);

            //  Component 2:  2 (M_i ⊙ M_j) · v
            //  M_i = X_{r,i} K X_rᵀ + X_r K X_{r,i}ᵀ + X_r K̇_i X_rᵀ     and mirror for j.
            //  Each of the 9 cross-products (three tails of M_i × three of M_j)
            //  gives a Hadamard-Gram apply.  Write them in full.
            let mut t2 = Array1::<f64>::zeros(n);
            //   (X_{r,i} K X_rᵀ) ⊙ (X_{r,j} K X_rᵀ)
            let s11 = RemlState::reducedweighted_gram(x_r, &v);
            let m11 = k.dot(&s11).dot(k);
            t2 = t2 + Self::rowwise_bilinear(x_ri, &m11, x_rj);
            //   (X_{r,i} K X_rᵀ) ⊙ (X_r K X_{r,j}ᵀ)
            let s12 = RemlState::reduced_crossweighted_gram(x_r, x_rj, &v);
            let m12 = k.dot(&s12).dot(k);
            t2 = t2 + Self::rowwise_bilinear(x_ri, &m12, x_r);
            //   (X_{r,i} K X_rᵀ) ⊙ (X_r K̇_j X_rᵀ)
            let s13 = RemlState::reducedweighted_gram(x_r, &v);
            let m13 = k.dot(&s13).dot(dot_k_j);
            t2 = t2 + Self::rowwise_bilinear(x_ri, &m13, x_r);
            //   (X_r K X_{r,i}ᵀ) ⊙ (X_{r,j} K X_rᵀ)
            let s21 = RemlState::reduced_crossweighted_gram(x_ri, x_r, &v);
            let m21 = k.dot(&s21).dot(k);
            t2 = t2 + Self::rowwise_bilinear(x_r, &m21, x_rj);
            //   (X_r K X_{r,i}ᵀ) ⊙ (X_r K X_{r,j}ᵀ)
            let s22 = RemlState::reduced_crossweighted_gram(x_ri, x_rj, &v);
            let m22 = k.dot(&s22).dot(k);
            t2 = t2 + Self::rowwise_bilinear(x_r, &m22, x_r);
            //   (X_r K X_{r,i}ᵀ) ⊙ (X_r K̇_j X_rᵀ)
            let s23 = RemlState::reduced_crossweighted_gram(x_ri, x_r, &v);
            let m23 = k.dot(&s23).dot(dot_k_j);
            t2 = t2 + Self::rowwise_bilinear(x_r, &m23, x_r);
            //   (X_r K̇_i X_rᵀ) ⊙ (X_{r,j} K X_rᵀ)
            let s31 = RemlState::reducedweighted_gram(x_r, &v);
            let m31 = dot_k_i.dot(&s31).dot(k);
            t2 = t2 + Self::rowwise_bilinear(x_r, &m31, x_rj);
            //   (X_r K̇_i X_rᵀ) ⊙ (X_r K X_{r,j}ᵀ)
            let s32 = RemlState::reduced_crossweighted_gram(x_r, x_rj, &v);
            let m32 = dot_k_i.dot(&s32).dot(k);
            t2 = t2 + Self::rowwise_bilinear(x_r, &m32, x_r);
            //   (X_r K̇_i X_rᵀ) ⊙ (X_r K̇_j X_rᵀ) — via apply_hadamard_gram with
            //   (K̇_i, K̇_j) as the two symmetric factors.
            t2 = t2 + RemlState::apply_hadamard_gram(x_r, dot_k_i, dot_k_j, &v);

            let y = 2.0 * (t1 + t2);
            out.column_mut(col).assign(&y);
        }
        out
    }

    /// Primitive A body: compute `Hφ_{τ_iτ_j}|β · V` via the outer
    /// four-way formula.
    #[allow(dead_code)]
    pub(crate) fn hphi_tau_tau_partial_apply(
        &self,
        x_tau_i: &Array2<f64>,
        x_tau_j: &Array2<f64>,
        x_tau_tau: Option<&Array2<f64>>,
        kernel: &FirthTauTauPartialKernel,
        rhs: &Array2<f64>,
    ) -> Array2<f64> {
        let p = self.x_dense.ncols();
        if rhs.nrows() != p {
            return Array2::<f64>::zeros((p, rhs.ncols()));
        }
        if rhs.ncols() == 0 || p == 0 {
            return Array2::<f64>::zeros((p, rhs.ncols()));
        }
        let n = self.x_dense.nrows();
        let m = rhs.ncols();

        //  η-space blocks
        let etav = self.x_dense.dot(rhs); // n×m
        let etav_i = x_tau_i.dot(rhs); // n×m
        let etav_j = x_tau_j.dot(rhs); // n×m
        let etav_ij = match x_tau_tau {
            Some(xij) => xij.dot(rhs),
            None => Array2::<f64>::zeros((n, m)),
        };

        //  jet column views
        let w1_col = self.w1.view().insert_axis(Axis(1));
        let w2_col = self.w2.view().insert_axis(Axis(1));
        let h_col = self.h_diag.view().insert_axis(Axis(1));
        let dh_i_col = kernel.dot_h_i.view().insert_axis(Axis(1));
        let dh_j_col = kernel.dot_h_j.view().insert_axis(Axis(1));
        let ddh_ij_col = kernel.ddot_h_ij.view().insert_axis(Axis(1));

        //  ẇ1_α, ẇ2_α columns and pair second-jet columns
        let ddet_i = &self.w2 * &kernel.deta_i;
        let ddet_j = &self.w2 * &kernel.deta_j;
        let dw2_i = &self.w3 * &kernel.deta_i;
        let dw2_j = &self.w3 * &kernel.deta_j;
        let ddet_i_col = ddet_i.view().insert_axis(Axis(1));
        let ddet_j_col = ddet_j.view().insert_axis(Axis(1));
        let dw2_i_col = dw2_i.view().insert_axis(Axis(1));
        let dw2_j_col = dw2_j.view().insert_axis(Axis(1));
        //  cross second-jet on w: ∂_i ẇ2_j = w4·η̇_i·η̇_j + w3·η̈_ij
        //  and  ∂_i ẇ1_j = w3·η̇_i·η̇_j + w2·η̈_ij.
        let zeros_n = Array1::<f64>::zeros(n);
        let deta_ij_ref: &Array1<f64> = kernel.deta_ij.as_ref().unwrap_or(&zeros_n);
        let dcross_w2 = &(&self.w4 * &kernel.deta_i) * &kernel.deta_j;
        let dcross_w2 = dcross_w2 + &(&self.w3 * deta_ij_ref);
        let dcross_w1 = &(&self.w3 * &kernel.deta_i) * &kernel.deta_j;
        let dcross_w1 = dcross_w1 + &(&self.w2 * deta_ij_ref);
        let dcross_w2_col = dcross_w2.view().insert_axis(Axis(1));
        let dcross_w1_col = dcross_w1.view().insert_axis(Axis(1));

        //  qv pieces
        let qv = &etav * &w1_col;
        let dqv_i = &(&etav * &ddet_i_col) + &(&etav_i * &w1_col);
        let dqv_j = &(&etav * &ddet_j_col) + &(&etav_j * &w1_col);
        let dqv_ij = &(&etav * &dcross_w1_col)
            + &(&etav_j * &ddet_i_col)
            + &(&etav_i * &ddet_j_col)
            + &(&etav_ij * &w1_col);

        //  m_qv and τ_j-drift:
        //    m_qv        = P̄·qv
        //    m_qv_j      = Ṁ_j·qv + P̄·dqv_j
        //    ∂_i m_qv    = Ṁ_i·qv + P̄·dqv_i
        //    ∂_i m_qv_j  = M̈_ij·qv + Ṁ_j·dqv_i + Ṁ_i·dqv_j + P̄·dqv_ij
        let m_qv = self.apply_pbar_to_matrix(&qv);
        let m_qv_j = self.apply_mtau_to_matrix(&kernel.tau_j, &qv) + self.apply_pbar_to_matrix(&dqv_j);
        let di_m_qv =
            self.apply_mtau_to_matrix(&kernel.tau_i, &qv) + self.apply_pbar_to_matrix(&dqv_i);
        let pddot_qv = self.apply_p_ddot_ij_to_matrix(kernel, &qv);
        let di_m_qv_j = pddot_qv
            + self.apply_mtau_to_matrix(&kernel.tau_j, &dqv_i)
            + self.apply_mtau_to_matrix(&kernel.tau_i, &dqv_j)
            + self.apply_pbar_to_matrix(&dqv_ij);

        //  r_V = (w2 ⊙ η_V) ⊙ h  −  m_qv ⊙ w1
        let rv = &(&(&etav * &w2_col) * &h_col) - &(&m_qv * &w1_col);

        //  r_{V,τ_j} = (ẇ2_j ⊙ η_V + w2 ⊙ η_{V,τ_j}) ⊙ h
        //            + (w2 ⊙ η_V) ⊙ ḣ_j
        //            − ( m_qv ⊙ ẇ1_j + m_qv_j ⊙ w1 )
        let rvj = &(&(&(&etav * &dw2_j_col) + &(&etav_j * &w2_col)) * &h_col)
            + &(&(&etav * &w2_col) * &dh_j_col)
            - &(&(&m_qv * &ddet_j_col) + &(&m_qv_j * &w1_col));

        //  ∂_i r_V
        //    = ((w3·η̇_i) ⊙ η_V + w2 ⊙ η_{V,τ_i}) ⊙ h
        //    + (w2 ⊙ η_V) ⊙ ḣ_i
        //    − [ ∂_i m_qv ⊙ w1  +  m_qv ⊙ ẇ1_i ]
        let di_rv = &(&(&(&etav * &dw2_i_col) + &(&etav_i * &w2_col)) * &h_col)
            + &(&(&etav * &w2_col) * &dh_i_col)
            - &(&(&di_m_qv * &w1_col) + &(&m_qv * &ddet_i_col));

        //  ∂_i r_{V,τ_j} — assembled from the 4 pieces of r_{V,τ_j}.
        //
        //  (1) ∂_i[(ẇ2_j ⊙ η_V) ⊙ h]
        //      = (∂_i ẇ2_j ⊙ η_V + ẇ2_j ⊙ η_{V,τ_i}) ⊙ h + (ẇ2_j ⊙ η_V) ⊙ ḣ_i
        let p1 = &(&(&(&etav * &dcross_w2_col) + &(&etav_i * &dw2_j_col)) * &h_col)
            + &(&(&etav * &dw2_j_col) * &dh_i_col);

        //  (2) ∂_i[(w2 ⊙ η_{V,τ_j}) ⊙ h]
        //      = (ẇ1_i ⊙ η_{V,τ_j} + w2 ⊙ η_{V,τ_iτ_j}) ⊙ h + (w2 ⊙ η_{V,τ_j}) ⊙ ḣ_i
        //  Note ẇ1_i = w2 ⊙ η̇_i, and w3·η̇_i == dw2_i; BUT the dependency on
        //  w2 ⊙ η_{V,τ_j} through ∂_i w2 = w3·η̇_i gives w3·η̇_i = dw2_i
        //  multiplied by η_{V,τ_j}.  So:
        let p2 = &(&(&(&etav_j * &dw2_i_col) + &(&etav_ij * &w2_col)) * &h_col)
            + &(&(&etav_j * &w2_col) * &dh_i_col);

        //  (3) ∂_i[(w2 ⊙ η_V) ⊙ ḣ_j]
        //      = ((w3·η̇_i) ⊙ η_V + w2 ⊙ η_{V,τ_i}) ⊙ ḣ_j
        //      + (w2 ⊙ η_V) ⊙ ḧ_{ij}
        let p3 = &(&(&(&etav * &dw2_i_col) + &(&etav_i * &w2_col)) * &dh_j_col)
            + &(&(&etav * &w2_col) * &ddh_ij_col);

        //  (4) ∂_i[m_qv ⊙ ẇ1_j + m_qv_j ⊙ w1]
        //    = ∂_i m_qv ⊙ ẇ1_j
        //    + m_qv ⊙ ∂_i ẇ1_j                 (= m_qv ⊙ dcross_w1)
        //    + ∂_i m_qv_j ⊙ w1
        //    + m_qv_j ⊙ (w2 ⊙ η̇_i)              (= m_qv_j ⊙ ddet_i)
        let p4 = &(&di_m_qv * &ddet_j_col)
            + &(&m_qv * &dcross_w1_col)
            + &(&di_m_qv_j * &w1_col)
            + &(&m_qv_j * &ddet_i_col);

        let di_rvj = p1 + p2 + p3 - p4;

        //  Outer assembly:
        //     0.5 [ X_{ij}ᵀ·rv + X_jᵀ·di_rv + X_iᵀ·rvj + Xᵀ·di_rvj ].
        let mut out = if let Some(xij) = x_tau_tau {
            xij.t().dot(&rv)
        } else {
            Array2::<f64>::zeros((p, m))
        };
        out = out + x_tau_j.t().dot(&di_rv);
        out = out + x_tau_i.t().dot(&rvj);
        out = out + self.x_dense.t().dot(&di_rvj);
        0.5 * out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, array};

    fn logisticweight(eta: f64) -> f64 {
        logit_inverse_link_jet5(eta).d1
    }

    fn d1fd(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
        (f(x + h) - f(x - h)) / (2.0 * h)
    }

    fn d2fd(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
        (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h)
    }

    fn firthphivalue(x: &Array2<f64>, beta: &Array1<f64>) -> f64 {
        let eta = x.dot(beta);
        let op = FirthDenseOperator::build(x, &eta).expect("firth operator");
        op.jeffreys_logdet()
    }

    fn firthgradphi(x: &Array2<f64>, beta: &Array1<f64>) -> Array1<f64> {
        let eta = x.dot(beta);
        let op = FirthDenseOperator::build(x, &eta).expect("firth operator");
        op.jeffreys_beta_gradient()
    }

    fn weighted_firthphivalue(
        x: &Array2<f64>,
        beta: &Array1<f64>,
        observation_weights: &Array1<f64>,
    ) -> f64 {
        let eta = x.dot(beta);
        let op =
            FirthDenseOperator::build_with_observation_weights(x, &eta, observation_weights.view())
                .expect("weighted firth operator");
        op.jeffreys_logdet()
    }

    #[test]
    fn firth_logisticweight_derivatives_match_finite_difference() {
        let x = array![
            [1.0, -1.1, 0.2],
            [1.0, -0.5, -0.6],
            [1.0, 0.0, 0.3],
            [1.0, 0.8, -0.4],
            [1.0, 1.2, 0.7],
        ];
        let beta = array![0.15, -0.6, 0.35];
        let eta = x.dot(&beta);
        let op = FirthDenseOperator::build(&x, &eta).expect("firth operator");

        let h12 = 2e-5;
        let h34 = 1e-3;
        let w = |z: f64| logisticweight(z);
        let w2 = |z: f64| d2fd(w, z, h12);
        let w3 = |z: f64| d1fd(w2, z, h34);
        for i in 0..eta.len() {
            let z = eta[i];
            let wfd = w(z);
            let w1fd = d1fd(w, z, h12);
            let w2fd = d2fd(w, z, h12);
            let w3fd = d1fd(w2, z, h34);
            let w4fd = d1fd(w3, z, h34);

            assert!((op.w[i] - wfd).abs() < 1e-12);
            assert_eq!(op.w1[i].signum(), w1fd.signum());
            assert_eq!(op.w2[i].signum(), w2fd.signum());
            // w3/w4 sign checks omitted: nested central-difference at these
            // orders is too fragile to guarantee sign agreement.
            assert!((op.w1[i] - w1fd).abs() < 2e-7);
            assert!((op.w2[i] - w2fd).abs() < 2e-5);
            assert!((op.w3[i] - w3fd).abs() < 4e-4);
            assert!((op.w4[i] - w4fd).abs() < 2e-3);
        }
    }

    #[test]
    fn weighted_firth_jeffreys_gradient_matches_finite_difference() {
        let x = array![
            [1.0, -0.7, 0.3],
            [1.0, -0.2, -0.4],
            [1.0, 0.5, 0.1],
            [1.0, 1.1, -0.6],
            [1.0, 1.6, 0.8],
        ];
        let beta = array![0.2, -0.45, 0.25];
        let observation_weights = array![1.0, 0.5, 2.0, 1.5, 0.75];
        let eta = x.dot(&beta);
        let op = FirthDenseOperator::build_with_observation_weights(
            &x,
            &eta,
            observation_weights.view(),
        )
        .expect("weighted firth operator");
        let grad = op.jeffreys_beta_gradient();
        let h = 1e-6;

        for j in 0..beta.len() {
            let mut beta_plus = beta.clone();
            beta_plus[j] += h;
            let mut beta_minus = beta.clone();
            beta_minus[j] -= h;
            let fd = (weighted_firthphivalue(&x, &beta_plus, &observation_weights)
                - weighted_firthphivalue(&x, &beta_minus, &observation_weights))
                / (2.0 * h);
            assert!(
                (grad[j] - fd).abs() < 1e-5,
                "weighted Firth gradient mismatch at {}: analytic={}, fd={}",
                j,
                grad[j],
                fd
            );
        }
    }

    #[test]
    fn rank_deficient_and_explicit_reduced_designs_share_same_jeffreys_objective() {
        // Column 3 is exactly column 1 + column 2, so the original design is
        // rank-deficient but its identifiable subspace is represented exactly by
        // the explicit two-column reduced design below.
        let x_full = array![
            [1.0, -1.2, -0.2],
            [1.0, -0.4, 0.6],
            [1.0, 0.1, 1.1],
            [1.0, 0.7, 1.7],
            [1.0, 1.3, 2.3],
        ];
        let x_reduced = array![[1.0, -1.2], [1.0, -0.4], [1.0, 0.1], [1.0, 0.7], [1.0, 1.3],];
        let beta_full: ndarray::Array1<f64> = array![0.25, -0.5, 0.15];
        let beta_reduced = array![beta_full[0] + beta_full[2], beta_full[1] + beta_full[2]];
        let eta_full = x_full.dot(&beta_full);
        let eta_reduced = x_reduced.dot(&beta_reduced);
        let observation_weights = array![1.0, 0.5, 1.75, 0.9, 1.2];

        for i in 0..eta_full.len() {
            assert!(
                (eta_full[i] - eta_reduced[i]).abs() < 1e-12,
                "eta mismatch at row {i}: full={} reduced={}",
                eta_full[i],
                eta_reduced[i]
            );
        }

        let op_full = FirthDenseOperator::build_with_observation_weights(
            &x_full,
            &eta_full,
            observation_weights.view(),
        )
        .expect("full firth operator");
        let op_reduced = FirthDenseOperator::build_with_observation_weights(
            &x_reduced,
            &eta_reduced,
            observation_weights.view(),
        )
        .expect("reduced firth operator");

        assert!(
            (op_full.jeffreys_logdet() - op_reduced.jeffreys_logdet()).abs() < 1e-12,
            "Jeffreys logdet mismatch between rank-deficient full design and its explicit reduced identifiable basis: full={} reduced={}",
            op_full.jeffreys_logdet(),
            op_reduced.jeffreys_logdet()
        );

        let hat_full = op_full.pirls_hat_diag();
        let hat_reduced = op_reduced.pirls_hat_diag();
        for i in 0..hat_full.len() {
            assert!(
                (hat_full[i] - hat_reduced[i]).abs() < 1e-12,
                "PIRLS hat-diagonal mismatch at row {i}: full={} reduced={}",
                hat_full[i],
                hat_reduced[i]
            );
        }
    }

    #[test]
    fn full_rank_reparameterizations_share_same_jeffreys_objective() {
        let x = array![[1.0, -1.2], [1.0, -0.4], [1.0, 0.1], [1.0, 0.7], [1.0, 1.3],];
        let basis = array![[1.4, -0.3], [0.6, 1.1]];
        let x_reparameterized = x.dot(&basis);
        let beta = array![0.25, -0.5];
        let basis_det: f64 = basis[[0, 0]] * basis[[1, 1]] - basis[[0, 1]] * basis[[1, 0]];
        assert!(
            basis_det.abs() > 1e-12,
            "basis transform must be invertible"
        );
        let basis_inv = array![
            [basis[[1, 1]] / basis_det, -basis[[0, 1]] / basis_det],
            [-basis[[1, 0]] / basis_det, basis[[0, 0]] / basis_det],
        ];
        let beta_reparameterized = basis_inv.dot(&beta);
        let eta = x.dot(&beta);
        let eta_reparameterized = x_reparameterized.dot(&beta_reparameterized);
        let observation_weights = array![1.0, 0.5, 1.75, 0.9, 1.2];

        for i in 0..eta.len() {
            assert!(
                (eta[i] - eta_reparameterized[i]).abs() < 1e-12,
                "eta mismatch at row {i}: original={} reparameterized={}",
                eta[i],
                eta_reparameterized[i]
            );
        }

        let op = FirthDenseOperator::build_with_observation_weights(
            &x,
            &eta,
            observation_weights.view(),
        )
        .expect("original firth operator");
        let op_reparameterized = FirthDenseOperator::build_with_observation_weights(
            &x_reparameterized,
            &eta_reparameterized,
            observation_weights.view(),
        )
        .expect("reparameterized firth operator");

        assert!(
            (op.jeffreys_logdet() - op_reparameterized.jeffreys_logdet()).abs() < 1e-12,
            "Jeffreys logdet mismatch under invertible reparameterization: original={} reparameterized={}",
            op.jeffreys_logdet(),
            op_reparameterized.jeffreys_logdet()
        );

        let hat = op.pirls_hat_diag();
        let hat_reparameterized = op_reparameterized.pirls_hat_diag();
        for i in 0..hat.len() {
            assert!(
                (hat[i] - hat_reparameterized[i]).abs() < 1e-12,
                "PIRLS hat-diagonal mismatch at row {i}: original={} reparameterized={}",
                hat[i],
                hat_reparameterized[i]
            );
        }
    }

    #[test]
    fn full_rank_identifiable_basis_diagonalizes_design_metric() {
        let x = array![[1.0, -1.2], [1.0, -0.4], [1.0, 0.1], [1.0, 0.7], [1.0, 1.3],];
        let beta = array![0.25, -0.5];
        let eta = x.dot(&beta);
        let observation_weights = array![1.0, 0.5, 1.75, 0.9, 1.2];
        let op = FirthDenseOperator::build_with_observation_weights(
            &x,
            &eta,
            observation_weights.view(),
        )
        .expect("firth operator");

        let reduced_metric = fast_atb(&op.x_reduced, &op.x_reduced);
        for i in 0..reduced_metric.nrows() {
            for j in 0..reduced_metric.ncols() {
                if i == j {
                    continue;
                }
                assert!(
                    reduced_metric[[i, j]].abs() < 1e-10,
                    "full-rank identifiable basis should diagonalize X_r'X_r: metric[{i},{j}]={}",
                    reduced_metric[[i, j]]
                );
            }
        }
    }

    #[test]
    fn firth_mixedsecond_direction_apply_is_symmetric_in_direction_order() {
        let x = array![
            [1.0, -1.0, 0.2],
            [1.0, -0.6, -0.3],
            [1.0, -0.1, 0.5],
            [1.0, 0.3, -0.7],
            [1.0, 0.8, 0.1],
            [1.0, 1.2, -0.4],
        ];
        let beta = array![0.1, -0.25, 0.2];
        let eta = x.dot(&beta);
        let op = FirthDenseOperator::build(&x, &eta).expect("firth operator");

        let u = array![0.3, -0.2, 0.4];
        let v = array![-0.5, 0.1, 0.25];
        let du = op.direction_from_deta(x.dot(&u));
        let dv = op.direction_from_deta(x.dot(&v));

        let eye = Array2::<f64>::eye(x.ncols());
        let uv = op.hphisecond_direction_apply(&du, &dv, &eye);
        let vu = op.hphisecond_direction_apply(&dv, &du, &eye);

        for i in 0..uv.nrows() {
            for j in 0..uv.ncols() {
                let a = uv[[i, j]];
                let b = vu[[i, j]];
                assert_eq!(
                    a.signum(),
                    b.signum(),
                    "mixed direction sign mismatch at ({i},{j}): uv={a} vu={b}"
                );
                assert!(
                    (a - b).abs() < 2e-7,
                    "mixed direction mismatch at ({i},{j}): uv={a} vu={b}"
                );
            }
        }
    }

    #[test]
    fn firth_direction_matrix_form_matches_apply_identity_form() {
        let x = array![
            [1.0, -1.1, 0.2],
            [1.0, -0.6, -0.3],
            [1.0, -0.1, 0.5],
            [1.0, 0.3, -0.7],
            [1.0, 0.8, 0.1],
            [1.0, 1.2, -0.4],
        ];
        let beta = array![0.08, -0.22, 0.27];
        let eta = x.dot(&beta);
        let op = FirthDenseOperator::build(&x, &eta).expect("firth operator");
        let u = Array1::from_vec(vec![0.25, -0.4, 0.35]);
        let dir = op.direction_from_deta(x.dot(&u));

        let p = x.ncols();
        let eye = Array2::<f64>::eye(p);
        let mut via_apply = op.hphi_direction_apply(&dir, &eye);
        for i in 0..p {
            for j in 0..i {
                let sym = 0.5 * (via_apply[[i, j]] + via_apply[[j, i]]);
                via_apply[[i, j]] = sym;
                via_apply[[j, i]] = sym;
            }
        }
        let direct = op.hphi_direction(&dir);
        let diff = &direct - &via_apply;
        let err = diff.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(err < 1e-10, "direction/apply mismatch: {err:e}");
    }

    #[test]
    fn firthphi_tau_partial_matches_finite_difference_logdet() {
        let x = array![
            [1.0, -1.0, 0.2],
            [1.0, -0.6, -0.3],
            [1.0, -0.1, 0.5],
            [1.0, 0.3, -0.7],
            [1.0, 0.8, 0.1],
            [1.0, 1.2, -0.4],
        ];
        let x_tau = array![
            [0.0, 0.15, -0.05],
            [0.0, -0.10, 0.02],
            [0.0, 0.08, 0.04],
            [0.0, -0.06, -0.03],
            [0.0, 0.05, 0.01],
            [0.0, -0.12, 0.06],
        ];
        let beta = array![0.1, -0.25, 0.2];
        let eta = x.dot(&beta);
        let op = FirthDenseOperator::build(&x, &eta).expect("firth operator");
        let analytic = op.exact_tau_kernel(&x_tau, &beta, false).phi_tau_partial;

        let h = 1e-6;
        let x_plus = &x + &(h * &x_tau);
        let x_minus = &x - &(h * &x_tau);
        let fd = (firthphivalue(&x_plus, &beta) - firthphivalue(&x_minus, &beta)) / (2.0 * h);

        assert!(
            (analytic - fd).abs() < 1e-6,
            "Phi_tau mismatch: analytic={analytic:.12e}, fd={fd:.12e}"
        );
    }

    #[test]
    fn firth_gphi_tau_matches_finite_differencegradphi() {
        let x = array![
            [1.0, -1.0, 0.2],
            [1.0, -0.6, -0.3],
            [1.0, -0.1, 0.5],
            [1.0, 0.3, -0.7],
            [1.0, 0.8, 0.1],
            [1.0, 1.2, -0.4],
        ];
        let x_tau = array![
            [0.0, 0.15, -0.05],
            [0.0, -0.10, 0.02],
            [0.0, 0.08, 0.04],
            [0.0, -0.06, -0.03],
            [0.0, 0.05, 0.01],
            [0.0, -0.12, 0.06],
        ];
        let beta = array![0.1, -0.25, 0.2];
        let eta = x.dot(&beta);
        let op = FirthDenseOperator::build(&x, &eta).expect("firth operator");
        let analytic = op.exact_tau_kernel(&x_tau, &beta, false).gphi_tau;

        let h = 1e-6;
        let x_plus = &x + &(h * &x_tau);
        let x_minus = &x - &(h * &x_tau);
        let fd = (firthgradphi(&x_plus, &beta) - firthgradphi(&x_minus, &beta)) / (2.0 * h);

        let err = (&analytic - &fd).iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            err < 1e-6,
            "gphi_tau mismatch: analytic={analytic:?}, fd={fd:?}, err={err:e}"
        );
    }

    /// Verify Primitive A (pair τ_i × τ_j) by central-FD'ing the single-τ
    /// primitive along τ_j at fixed β.  Identity:
    ///     ∂/∂τ_j [Hφ_{τ_i}|β · V]  =  Hφ_{τ_iτ_j}|β · V.
    /// Tolerance 1e-7 relative max-abs.
    #[test]
    fn firthphi_tau_tau_partial_matches_finite_difference() {
        let x = array![
            [1.0, -1.0, 0.2],
            [1.0, -0.6, -0.3],
            [1.0, -0.1, 0.5],
            [1.0, 0.3, -0.7],
            [1.0, 0.8, 0.1],
            [1.0, 1.2, -0.4],
        ];
        let x_tau_i = array![
            [0.0, 0.15, -0.05],
            [0.0, -0.10, 0.02],
            [0.0, 0.08, 0.04],
            [0.0, -0.06, -0.03],
            [0.0, 0.05, 0.01],
            [0.0, -0.12, 0.06],
        ];
        let x_tau_j = array![
            [0.0, -0.04, 0.11],
            [0.0, 0.09, -0.02],
            [0.0, -0.06, 0.07],
            [0.0, 0.10, -0.05],
            [0.0, -0.03, 0.08],
            [0.0, 0.07, -0.09],
        ];
        let beta = array![0.1, -0.25, 0.2];
        let eta = x.dot(&beta);
        let op = FirthDenseOperator::build(&x, &eta).expect("firth operator");
        let p = x.ncols();

        let m = 3usize;
        let vals = [0.21, -0.44, 0.17, 0.38, 0.05, -0.22, -0.11, 0.27, 0.31];
        let mut rhs = Array2::<f64>::zeros((p, m));
        for r in 0..p {
            for c in 0..m {
                rhs[[r, c]] = vals[(r * m + c) % vals.len()];
            }
        }

        let x_tau_i_r = op.reduce_explicit_design(&x_tau_i);
        let x_tau_j_r = op.reduce_explicit_design(&x_tau_j);
        let deta_i = x_tau_i.dot(&beta);
        let deta_j = x_tau_j.dot(&beta);
        let (dot_i_i, dot_h_i) = op.dot_i_and_h_from_reduced(&x_tau_i_r, &deta_i);
        let (dot_i_j, dot_h_j) = op.dot_i_and_h_from_reduced(&x_tau_j_r, &deta_j);

        let kernel_ij = op.hphi_tau_tau_partial_prepare_from_partials(
            x_tau_i_r.clone(),
            x_tau_j_r.clone(),
            deta_i.clone(),
            deta_j.clone(),
            dot_i_i.clone(),
            dot_i_j.clone(),
            dot_h_i.clone(),
            dot_h_j.clone(),
            None,
            None,
        );
        let analytic_ij =
            op.hphi_tau_tau_partial_apply(&x_tau_i, &x_tau_j, None, &kernel_ij, &rhs);

        let kernel_ji = op.hphi_tau_tau_partial_prepare_from_partials(
            x_tau_j_r,
            x_tau_i_r,
            deta_j,
            deta_i,
            dot_i_j,
            dot_i_i,
            dot_h_j,
            dot_h_i,
            None,
            None,
        );
        let analytic_ji =
            op.hphi_tau_tau_partial_apply(&x_tau_j, &x_tau_i, None, &kernel_ji, &rhs);

        //  Symmetry cross-check (Clairaut).
        let sym_diff: f64 = (&analytic_ij - &analytic_ji)
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        let sym_scale: f64 = analytic_ij
            .iter()
            .chain(analytic_ji.iter())
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max)
            .max(1.0);
        assert!(
            sym_diff / sym_scale < 1e-10,
            "τ×τ primitive not symmetric in direction order: sym_diff={sym_diff:.3e}"
        );

        //  FD reference: central difference of single-τ primitive along τ_j.
        let h = 1e-4_f64;
        let fd_block_i = |x_eval: &Array2<f64>| -> Array2<f64> {
            let eta_e = x_eval.dot(&beta);
            let op_e = FirthDenseOperator::build(x_eval, &eta_e).expect("perturbed operator");
            let x_tau_i_r = op_e.reduce_explicit_design(&x_tau_i);
            let deta_i_e = x_tau_i.dot(&beta);
            let (dot_i_i_e, dot_h_i_e) = op_e.dot_i_and_h_from_reduced(&x_tau_i_r, &deta_i_e);
            let ker_i_e = op_e.hphi_tau_partial_prepare_from_partials(
                x_tau_i_r,
                &deta_i_e,
                dot_h_i_e,
                dot_i_i_e,
            );
            op_e.hphi_tau_partial_apply(&x_tau_i, &ker_i_e, &rhs)
        };
        let x_plus = &x + &(h * &x_tau_j);
        let x_minus = &x - &(h * &x_tau_j);
        let fd_ij = (&fd_block_i(&x_plus) - &fd_block_i(&x_minus)) / (2.0 * h);

        let rel_maxabs = |a: &Array2<f64>, b: &Array2<f64>| -> f64 {
            let scale = a
                .iter()
                .chain(b.iter())
                .map(|v| v.abs())
                .fold(0.0_f64, f64::max)
                .max(1.0);
            let max = (a - b).iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            max / scale
        };
        let err_ij = rel_maxabs(&analytic_ij, &fd_ij);
        assert!(
            err_ij < 1e-7,
            "Primitive A mismatch (i,j): rel_max_abs={err_ij:.3e} > 1e-7\n\
             analytic=\n{analytic_ij:?}\nfd=\n{fd_ij:?}"
        );
    }

    #[test]
    fn logisticweight_loses_positive_tail_mass() {
        let eta = 50.0_f64;
        let z = (-eta).exp();
        let stable = z / (1.0_f64 + z).powi(2);
        assert!(stable > 0.0);
        let got = logisticweight(eta);
        assert!(
            (got - stable).abs() < 1e-30,
            "Firth logisticweight should equal the stable tail formula z/(1+z)^2 at eta={eta}; got {} vs {}",
            got,
            stable
        );
    }
}
