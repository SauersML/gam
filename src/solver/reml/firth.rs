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
    pub(crate) fn reduce_explicit_design(&self, x: &Array2<f64>) -> Array2<f64> {
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

    pub(crate) fn dot_i_and_h_from_reduced(
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

    pub(crate) fn hphi_tau_partial_prepare_from_partials(
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

    // ═════════════════════════════════════════════════════════════════════════
    //  Pair-term primitives for the Firth outer Hessian (Task #13a / #17)
    // ═════════════════════════════════════════════════════════════════════════
    //
    // The REML outer Hessian at a ψ=(ρ,τ) pair needs two Firth contributions
    // that are NOT covered by the existing single-τ primitives:
    //
    //   A.  the pure τ×τ second partial of H_φ at fixed β (pair drift inside
    //       the fixed-β second-derivative trace of B_i,j used by
    //       build_tau_tau_pair_callback).  This is the Firth analog of the
    //       penalty-logdet pair term in the outer-derivative cookbook.
    //
    //   B.  the β-derivative of (H_φ)_τ|_β in direction v.  This is the
    //       fixed-drift-derivative M_i[v] = D_β B_i[v] that
    //       compute_drift_deriv_traces uses through the fixed_drift_deriv
    //       callback in build_tau_hyper_coords.  It is currently always None
    //       in the Firth+Logit path, which is what makes the outer Hessian
    //       approximate for Firth-reweighted models.
    //
    // Both primitives operate in the reduced identifiable subspace (X_r, K_r,
    // S_r, Z=X_r) of the dense Firth operator and are matrix-free in n.  They
    // do NOT introduce any dense n×n or p×p×p object; every contraction is
    // routed through reduced-space Hadamard-Gram applies and rowwise
    // bilinear forms, in the same spirit as hphi_tau_partial_apply and
    // hphisecond_direction_apply.
    //
    // Both are exact in the smooth-regime operating point assumed by this
    // module: X SPD-full-rank on its identifiable subspace, Q held fixed
    // within one outer REML step (active-subspace drift enters only between
    // outer iterates), w_i(η) > 0 strictly positive, and β is at the P-IRLS
    // solution for the current ψ so β_τ is supplied by the unified evaluator
    // via the IFT solve.
    //
    // Symbol conventions shared with the existing operator code:
    //   X_r   := X Q            (reduced identifiable design)
    //   W     := diag(w(η))     (Fisher weights), w', w'', w''', w''''
    //   I_r   := X_rᵀ W X_r,  K_r := I_r^{-1},  S_r := X_rᵀ X_r
    //   M     := X_r K_r X_rᵀ,  P := M ⊙ M,  B := diag(w') X
    //   h     := diag(M)
    //   X_i   := ∂X/∂τ_i,  X_{r,i} := X_i Q,  η̇_i := X_i β,  etc.
    //   δη_v  := X v           (β-direction v),  δη_{τ,v} := X_τ v
    //
    // H_φ structural form (reduced form of ∇²_β Φ_F):
    //   H_φ  =  ½ [ Xᵀ diag(w'' ⊙ h) X  −  Bᵀ P B ]
    // where the first term arises from differentiating the Jeffreys gradient
    // ½ Xᵀ (w' ⊙ h) once more in β, and the second collects the IFT-mediated
    // β-derivative of h = diag(X_r K_r X_rᵀ) through K_r = I_r^{-1}.  This is
    // the same form the existing hphi_direction code implements in directional
    // form (cf. firth.rs hphi_direction / hphisecond_direction_apply, the
    // 9-term D²J₂ expansion with J₂ = BᵀPB).
    //
    // ─────────────────────────────────────────────────────────────────────────
    //  Primitive A — ∂²H_φ/∂τ_i ∂τ_j |_β
    // ─────────────────────────────────────────────────────────────────────────
    //
    // WHAT IT COMPUTES
    //   Given a pair of τ-drift designs (X_τ_i, X_τ_j) and optional second
    //   design derivative X_{τ_i τ_j}, evaluates
    //
    //     ∂²H_φ/∂τ_i ∂τ_j |_β  =  ½ [ ∂²(Xᵀ Γ X)/∂τ_i ∂τ_j
    //                                − ∂²(Bᵀ P B)/∂τ_i ∂τ_j ],
    //   with Γ := diag(w'' ⊙ h).  Acts on a p×m rhs and returns a p×m block
    //   (exact same contract as hphi_tau_partial_apply, but for the *second*
    //   mixed τ-derivative at fixed β).
    //
    // WHY (REML callsite)
    //   In the outer Hessian entry Ḧ_{i,j} for τ×τ pair (i,j), the fixed-β
    //   second drift of B_i is exactly this primitive (with the Firth sign:
    //   B_i = −(H_φ)_τ_i|_β + other likelihood pieces, so ∂²B_i/∂τ_j|_β
    //   contributes −∂²H_φ/∂τ_i∂τ_j|_β to the outer Hessian trace).  The
    //   existing Firth pair callback at build_tau_tau_pair_callback currently
    //   carries zero for this Firth contribution; wiring this primitive into
    //   the TauTauPairHyperOperator is the remaining step to make the τ×τ
    //   outer Hessian exact in the Firth-reweighted Logit path.
    //
    // DERIVATION (full chain-rule expansion, at fixed β)
    //
    //   Building blocks at fixed β (single-τ):
    //     İ_i      := ∂I_r/∂τ_i |_β
    //                = X_{r,i}ᵀ W X_r + X_rᵀ W X_{r,i} + X_rᵀ Ẇ_i X_r,
    //       Ẇ_i    := diag(w' ⊙ η̇_i),   η̇_i := X_i β.
    //     K̇_i      := ∂K_r/∂τ_i = −K_r İ_i K_r.
    //     ḣ_i      := ∂h/∂τ_i |_β
    //                = 2·diag(X_{r,i} K_r X_rᵀ) + diag(X_r K̇_i X_rᵀ).
    //     Ṁ_i      := ∂M/∂τ_i |_β
    //                = X_{r,i} K_r X_rᵀ + X_r K̇_i X_rᵀ + X_r K_r X_{r,i}ᵀ.
    //     Ḃ_i      := ∂B/∂τ_i |_β
    //                = diag(w'' ⊙ η̇_i) X + diag(w') X_i.
    //     Ṗ_i      := ∂P/∂τ_i = 2 (M ⊙ Ṁ_i).
    //     Γ̇_i     := ∂Γ/∂τ_i |_β = diag(w''' ⊙ η̇_i ⊙ h + w'' ⊙ ḣ_i).
    //
    //   Second-order building blocks:
    //     η̈_{ij}  := X_{ij} β                   (0 for design linear in τ)
    //     Ẅ_{ij} := diag(w'' ⊙ η̇_i ⊙ η̇_j
    //                     + w' ⊙ η̈_{ij})
    //
    //     Ï_{ij}   := ∂²I_r/∂τ_i ∂τ_j |_β
    //                = X_{r,ij}ᵀ W X_r  +  X_rᵀ W X_{r,ij}
    //                 + X_{r,i}ᵀ W X_{r,j}  +  X_{r,j}ᵀ W X_{r,i}
    //                 + X_{r,i}ᵀ Ẇ_j X_r  +  X_rᵀ Ẇ_j X_{r,i}
    //                 + X_{r,j}ᵀ Ẇ_i X_r  +  X_rᵀ Ẇ_i X_{r,j}
    //                 + X_rᵀ Ẅ_{ij} X_r.
    //
    //     K̈_{ij}  := ∂²K_r/∂τ_i ∂τ_j
    //                = −K_r Ï_{ij} K_r
    //                  + K_r İ_i K_r İ_j K_r
    //                  + K_r İ_j K_r İ_i K_r.
    //
    //     M̈_{ij}  := X_{r,ij} K_r X_rᵀ + X_r K_r X_{r,ij}ᵀ
    //                 + X_{r,i} K̇_j X_rᵀ + X_r K̇_j X_{r,i}ᵀ
    //                 + X_{r,j} K̇_i X_rᵀ + X_r K̇_i X_{r,j}ᵀ
    //                 + X_{r,i} K_r X_{r,j}ᵀ + X_{r,j} K_r X_{r,i}ᵀ
    //                 + X_r K̈_{ij} X_rᵀ.
    //
    //     P̈_{ij}  := ∂²P/∂τ_i ∂τ_j
    //                = 2 (Ṁ_i ⊙ Ṁ_j) + 2 (Ṁ_j ⊙ Ṁ_i) + 2 (M ⊙ M̈_{ij})
    //                = 4 (Ṁ_i ⊙ Ṁ_j) + 2 (M ⊙ M̈_{ij}).
    //
    //     ḧ_{ij}  := ∂²h/∂τ_i ∂τ_j |_β
    //                = 2·diag(X_{r,ij} K_r X_rᵀ)
    //                 + diag(X_r K̈_{ij} X_rᵀ)
    //                 + 2·diag(X_{r,i} K̇_j X_rᵀ)
    //                 + 2·diag(X_{r,j} K̇_i X_rᵀ)
    //                 + 2·diag(X_{r,i} K_r X_{r,j}ᵀ).
    //
    //     B̈_{ij}  := ∂²B/∂τ_i ∂τ_j |_β
    //                = diag(w''' ⊙ η̇_i ⊙ η̇_j + w'' ⊙ η̈_{ij}) X
    //                 + diag(w'' ⊙ η̇_i) X_j
    //                 + diag(w'' ⊙ η̇_j) X_i
    //                 + diag(w') X_{ij}.
    //
    //     Γ̈_{ij} := ∂²Γ/∂τ_i ∂τ_j |_β
    //                = diag( w'''' ⊙ η̇_i ⊙ η̇_j ⊙ h
    //                       + w''' ⊙ η̈_{ij} ⊙ h
    //                       + w''' ⊙ η̇_i ⊙ ḣ_j
    //                       + w''' ⊙ η̇_j ⊙ ḣ_i
    //                       + w'' ⊙ ḧ_{ij} ).
    //
    //   Diagonal-term expansion (the Xᵀ Γ X branch):
    //
    //     ∂²(Xᵀ Γ X)/∂τ_i ∂τ_j  =
    //         X_{ij}ᵀ Γ X  + Xᵀ Γ X_{ij}
    //       + X_iᵀ Γ X_j  + X_jᵀ Γ X_i
    //       + X_iᵀ Γ̇_j X  + Xᵀ Γ̇_j X_i
    //       + X_jᵀ Γ̇_i X  + Xᵀ Γ̇_i X_j
    //       + Xᵀ Γ̈_{ij} X.
    //
    //   9-term expansion for the BᵀPB branch (structurally identical to
    //   the existing β×β D²J₂[u,v] at firth.rs:~820-830 with (u,v)
    //   substituted by (τ_i, τ_j) and the appropriate Ḃ, B̈, Ṗ, P̈):
    //
    //     D²(BᵀPB)[τ_i,τ_j]  =
    //         B̈_{ij}ᵀ  P    B      +  Bᵀ       P    B̈_{ij}
    //       + Ḃ_iᵀ    P    Ḃ_j   +  Ḃ_jᵀ    P    Ḃ_i
    //       + Ḃ_iᵀ    Ṗ_j  B      +  Bᵀ       Ṗ_j  Ḃ_i
    //       + Ḃ_jᵀ    Ṗ_i  B      +  Bᵀ       Ṗ_i  Ḃ_j
    //       + Bᵀ       P̈_{ij} B.
    //
    //   Combining,
    //
    //     ∂²H_φ/∂τ_i ∂τ_j |_β  =  ½ [
    //         ∂²(Xᵀ Γ X)/∂τ_i ∂τ_j  −  D²(BᵀPB)[τ_i, τ_j]
    //     ].
    //
    // IMPLEMENTATION SKETCH (for 13b)
    //   • Build per-direction reduced quantities for τ_i and τ_j:
    //       (x_tau_reduced, η̇, İ, K̇, Ṁ operator pieces, ḣ, b_uvec = w''⊙η̇).
    //     The existing `dot_i_and_h_from_reduced` yields İ and ḣ already;
    //     the per-direction "A_u" analog is A_τ = K_r İ K_r, matching the
    //     FirthDirection form used by hphisecond_direction_apply.
    //   • Use apply_hadamard_gram_to_matrix with
    //       (A_left, A_right) ∈ { (K_r, K_r), (K_r, A_τ_i), (K_r, A_τ_j),
    //                             (A_τ_i, A_τ_j) }
    //     to realize P-products, Ṗ_τ-products, and the (Ṁ_i ⊙ Ṁ_j) piece of
    //     P̈_{ij} without forming any n×n dense intermediate.
    //   • The pure-second piece `X_r K̈_{ij} X_rᵀ` decomposes into three
    //     reduced triple products (K_r Ï_{ij} K_r, K_r İ_i K_r İ_j K_r, and
    //     its transpose).  All are size-r×r in reduced coordinates.
    //   • For design-linear-in-τ smooths, X_{ij}=0 and η̈_{ij}=0, which
    //     prunes many sub-terms; callers who have X_{τ_i τ_j} available
    //     should pass it so the primitive remains exact on curved designs.
    //
    // ─────────────────────────────────────────────────────────────────────────
    //  Primitive B — D_β((H_φ)_τ|_β)[v]
    // ─────────────────────────────────────────────────────────────────────────
    //
    // WHAT IT COMPUTES
    //   Given a single τ-drift design X_τ, the β-fixed Firth partial
    //   (H_φ)_τ|_β encoded by FirthTauPartialKernel, and a β-direction
    //   vector v (of length p), returns the β-derivative of (H_φ)_τ|_β
    //   applied to an rhs block (so output is p×m, matching the pair's
    //   fixed_drift_deriv callback signature DriftDerivResult).  In
    //   symbols:
    //
    //     D_β((H_φ)_τ|_β)[v]  =  ½ [ D_β{(∂(XᵀΓX)/∂τ)|_β}[v]
    //                                 −  D_β{(∂(BᵀPB)/∂τ)|_β}[v] ].
    //
    // WHY (REML callsite)
    //   In the exact outer Hessian assembly (compute_drift_deriv_traces in
    //   unified.rs), the Ḧ_{ij} entry picks up
    //     tr(G_ε · D_β B_i[v_j])  +  tr(G_ε · D_β B_j[v_i]).
    //   For τ coordinates in the Firth+Logit path, B_τ = (penalty / design
    //   pieces) − (H_φ)_τ|_β, so the Firth share of D_β B_τ[v] is
    //     − D_β((H_φ)_τ|_β)[v].
    //   Hooking this primitive up through a FixedDriftDerivFn (returning
    //   DriftDerivResult::Dense of this p×p β-v action) is exactly what
    //   lets build_tau_hyper_coords pass a non-None fixed_drift_deriv
    //   closure into the unified evaluator, closing the approximation gap
    //   that firth_pair_terms_unavailable currently tracks.
    //
    // DERIVATION (β-derivative of each τ-partial term in direction v)
    //
    //   β enters only through η=Xβ, so designs X, X_τ, Q, X_r are all
    //   β-independent; D_β acts on w(η) and its derivatives, on I_r, K_r,
    //   M, h, and on η̇_τ = X_τ β.
    //
    //   Primary β-derivative building blocks (matches FirthDirection with
    //   deta := δη_v = X v):
    //     I'_v  := D_β I_r[v] = X_rᵀ diag(w' ⊙ δη_v) X_r      (g_u_reduced)
    //     A_v   := D_β K_r[v] = −K_r I'_v K_r                  (a_u_reduced)
    //     dh_v  := D_β h[v]    = −diag(X_r K_r I'_v K_r X_rᵀ)
    //                          = diag(X_r A_v X_rᵀ)            (dh)
    //     (w')_v  := D_β w'[v]  = w''  ⊙ δη_v
    //     (w'')_v := D_β w''[v] = w''' ⊙ δη_v
    //     (w''')_v:= D_β w'''[v]= w''''⊙ δη_v
    //     δη_{τ,v} := D_β(η̇_τ)[v] = X_τ v
    //
    //   Mixed τ-β pieces:
    //     D_β(İ_τ)[v]
    //       = X_{r,τ}ᵀ diag(w'' ⊙ δη_v) X_r
    //        + X_rᵀ diag(w'' ⊙ δη_v) X_{r,τ}
    //        + X_rᵀ diag(w'' ⊙ η̇_τ ⊙ δη_v
    //                     + w' ⊙ δη_{τ,v}) X_r.
    //     D_β(K̇_τ)[v]
    //       = −( A_v İ_τ K_r  +  K_r D_β(İ_τ)[v] K_r
    //             +  K_r İ_τ A_v ).
    //     D_β(Ṁ_τ)[v]
    //       = X_{r,τ} A_v X_rᵀ
    //        + X_r D_β(K̇_τ)[v] X_rᵀ
    //        + X_r A_v X_{r,τ}ᵀ.
    //     D_β(ḣ_τ)[v]
    //       = 2·diag(X_{r,τ} A_v X_rᵀ)
    //        + diag(X_r D_β(K̇_τ)[v] X_rᵀ).
    //
    //   Diagonal-term β-derivative ( (X_τᵀΓX + XᵀΓX_τ + XᵀΓ̇_τ X) branch ):
    //     D_β(X_τᵀ Γ X + Xᵀ Γ X_τ)[v]
    //       = X_τᵀ Γ_v X + Xᵀ Γ_v X_τ,
    //       Γ_v  := D_β Γ[v] = diag((w'')_v ⊙ h + w'' ⊙ dh_v)
    //                        = diag(w''' ⊙ δη_v ⊙ h + w'' ⊙ dh_v).
    //     D_β(Xᵀ Γ̇_τ X)[v]
    //       = Xᵀ Γ̇_{τ,v} X,
    //       Γ̇_{τ,v}
    //        := D_β Γ̇_τ[v]
    //         = diag( (w''')_v ⊙ η̇_τ ⊙ h
    //                 + w''' ⊙ δη_{τ,v} ⊙ h
    //                 + w''' ⊙ η̇_τ ⊙ dh_v
    //                 + (w'')_v ⊙ ḣ_τ
    //                 + w'' ⊙ D_β(ḣ_τ)[v] )
    //         = diag( w'''' ⊙ η̇_τ ⊙ δη_v ⊙ h
    //                 + w''' ⊙ δη_{τ,v} ⊙ h
    //                 + w''' ⊙ η̇_τ ⊙ dh_v
    //                 + w''' ⊙ δη_v ⊙ ḣ_τ
    //                 + w'' ⊙ D_β(ḣ_τ)[v] ).
    //
    //   Cross-coupling τ-β pieces for B:
    //     B_v  := D_β B[v]   = diag(w'' ⊙ δη_v) X               (b_uvec)
    //     B_τ  := ∂B/∂τ|_β   = diag(w'' ⊙ η̇_τ) X
    //                         + diag(w') X_τ.
    //     B_{τ,v}
    //         := D_β B_τ[v]  = diag( w''' ⊙ η̇_τ ⊙ δη_v
    //                                 + w'' ⊙ δη_{τ,v} ) X
    //                         + diag(w'' ⊙ δη_v) X_τ.
    //
    //   BᵀPB branch — 9 terms, obtained by applying the product rule to
    //   ∂(BᵀPB)/∂τ = Ḃ_τᵀ P B + Bᵀ Ṗ_τ B + Bᵀ P Ḃ_τ and then taking
    //   D_β(·)[v] of each factor:
    //
    //     D_β(Ḃ_τᵀ P B)[v]   = B_{τ,v}ᵀ P B + Ḃ_τᵀ P_v B + Ḃ_τᵀ P B_v,
    //     D_β(Bᵀ Ṗ_τ B)[v]   = B_vᵀ Ṗ_τ B  + Bᵀ P_{τ,v} B + Bᵀ Ṗ_τ B_v,
    //     D_β(Bᵀ P Ḃ_τ)[v]   = B_vᵀ P Ḃ_τ + Bᵀ P_v Ḃ_τ + Bᵀ P B_{τ,v}.
    //
    //   Here Ḃ_τ = B_τ above, and
    //     P_v := D_β P[v]         = 2 (M ⊙ M_v),   M_v = X_r A_v X_rᵀ.
    //     Ṗ_τ := ∂P/∂τ|_β         = 2 (M ⊙ M_τ),
    //       M_τ = X_{r,τ} K_r X_rᵀ + X_r K̇_τ X_rᵀ + X_r K_r X_{r,τ}ᵀ.
    //     P_{τ,v} := D_β(Ṗ_τ)[v]  = 2 (M_v ⊙ M_τ) + 2 (M ⊙ M_{τ,v}),
    //       M_{τ,v} = X_{r,τ} A_v X_rᵀ + X_r D_β(K̇_τ)[v] X_rᵀ + X_r A_v X_{r,τ}ᵀ.
    //
    //   Final primitive:
    //
    //     D_β((H_φ)_τ|_β)[v]  =  ½ [
    //           X_τᵀ Γ_v X  + Xᵀ Γ_v X_τ  + Xᵀ Γ̇_{τ,v} X
    //         −  (9-term BᵀPB β-τ expansion above)
    //     ].
    //
    //   Applied to an rhs block `R ∈ ℝ^{p × m}`, each Xᵀ(…) X R collapses
    //   to n-length row scalings of (X R) followed by Xᵀ; each Bᵀ P B
    //   variant uses apply_hadamard_gram_to_matrix with the correct
    //   (A_left, A_right) ∈ { (K_r, K_r), (K_r, A_v), (K_r, K̇_τ),
    //     (K_r, D_β(K̇_τ)[v]), (A_v, K̇_τ), (K_r, K̇_τ) } to realize
    //   P, P_v, Ṗ_τ, P_{τ,v} actions.  All operators are r×r in reduced
    //   coordinates, matching the existing apply cost profile.
    //
    // IMPLEMENTATION SKETCH (for 13c)
    //   • Build `FirthDirection` from deta = X v (reuses existing
    //     direction_from_deta, giving I'_v, A_v, dh_v, b_uvec).
    //   • Build β-derivatives of the τ-specific fields of
    //     FirthTauPartialKernel (dotw1, dotw2, dot_h_partial, dot_k_reduced,
    //     and the implicit M_τ reduced-coords operator).  These become a
    //     new FirthTauBetaPartialKernel attached to the prepared state.
    //   • The apply step is then algebraically identical to
    //     hphi_tau_partial_apply but with every W-tensor weight replaced by
    //     its β-derivative in v, and every (M, K_r)-Gram replaced by the
    //     appropriate β-derivative Gram above.  The structure is regular
    //     enough that a single helper, shared with Primitive A, can absorb
    //     both pair dispatches.
    //
    // NOTE ON DESIGN-LINEAR SMOOTHS
    //   For the common case of design-linear-in-τ smooths (scale-moving
    //   anisotropic bases), X_i and X_τ are constant in τ, so X_{ij}=0 and
    //   η̈_{ij}=0.  The primitives collapse to their W-reweighted cores but
    //   remain matrix-free; no special fast path is needed because the
    //   zeroed terms simply drop out of the Hadamard-Gram assembly.
    //
    // ═════════════════════════════════════════════════════════════════════════

    /// Primitive A — prepare step: assemble the τ_i × τ_j reduced kernel.
    ///
    /// Consumes the per-direction partial quantities produced by
    /// `dot_i_and_h_from_reduced` for τ_i and τ_j (plus an optional second
    /// design derivative X_{τ_i τ_j}), and returns a cached kernel carrying
    /// the M̈_{ij}, K̈_{ij}, ḧ_{ij}, Γ̈_{ij}, and B̈_{ij}-related reduced
    /// coordinates needed by `hphi_tau_tau_partial_apply`.
    ///
    /// This signature mirrors `hphi_tau_partial_prepare_from_partials` for
    /// consistency; the pair version needs both directions simultaneously
    /// (to realize the 9-term D² expansion) and therefore owns both
    /// `x_tau_{i,j}_reduced` and their η̇_i / η̇_j.
    ///
    /// Allow(dead_code): scaffold until pair-callback threading wires the
    /// operator into `HyperCoordPair::b_operator` additively.
    #[allow(dead_code)]
    pub(crate) fn hphi_tau_tau_partial_prepare_from_partials(
        &self,
        x_tau_i_reduced: Array2<f64>,
        x_tau_j_reduced: Array2<f64>,
        deta_i_partial: &Array1<f64>,
        deta_j_partial: &Array1<f64>,
        dot_h_i_partial: Array1<f64>,
        dot_h_j_partial: Array1<f64>,
        dot_i_i_partial: Array2<f64>,
        dot_i_j_partial: Array2<f64>,
        x_tau_tau_reduced: Option<Array2<f64>>,
        deta_ij_partial: Option<Array1<f64>>,
    ) -> FirthTauTauPartialKernel {
        // K̇_i = -K_r İ_i K_r;  K̇_j = -K_r İ_j K_r.
        let dot_k_i_reduced = -self.k_reduced.dot(&dot_i_i_partial).dot(&self.k_reduced);
        let dot_k_j_reduced = -self.k_reduced.dot(&dot_i_j_partial).dot(&self.k_reduced);
        FirthTauTauPartialKernel {
            x_tau_i_reduced,
            x_tau_j_reduced,
            deta_i_partial: deta_i_partial.clone(),
            deta_j_partial: deta_j_partial.clone(),
            dot_h_i_partial,
            dot_h_j_partial,
            dot_k_i_reduced,
            dot_k_j_reduced,
            dot_i_i_partial,
            dot_i_j_partial,
            x_tau_tau_reduced,
            deta_ij_partial,
        }
    }

    /// Primitive A — apply step: evaluate ∂²H_φ/∂τ_i ∂τ_j |_β against a p×m
    /// rhs block, returning a p×m block.
    ///
    /// Contract mirrors `hphi_tau_partial_apply`: the caller passes the two
    /// τ-drift designs and the prepared kernel, and receives the fixed-β
    /// second-τ Firth drift as a dense p×m action.  Matrix-free in n.
    ///
    /// Allow(dead_code): scaffold until pair-callback threading wires the
    /// operator into `HyperCoordPair::b_operator` additively.
    #[allow(dead_code)]
    pub(crate) fn hphi_tau_tau_partial_apply(
        &self,
        x_tau_i: &Array2<f64>,
        x_tau_j: &Array2<f64>,
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

        // Short aliases.
        let z = &self.z_reduced;
        let x_r = &self.x_reduced;
        let k = &self.k_reduced;
        let x_ri = &kernel.x_tau_i_reduced;
        let x_rj = &kernel.x_tau_j_reduced;
        let deta_i = &kernel.deta_i_partial;
        let deta_j = &kernel.deta_j_partial;
        let dh_i = &kernel.dot_h_i_partial;
        let dh_j = &kernel.dot_h_j_partial;
        let dot_k_i = &kernel.dot_k_i_reduced;
        let dot_k_j = &kernel.dot_k_j_reduced;
        let dot_i_i = &kernel.dot_i_i_partial;
        let dot_i_j = &kernel.dot_i_j_partial;

        // Optional second-design pieces: default to zero when the design is
        // τ-linear (η̈_{ij} = 0, X_{ij} = 0).
        let x_tau_tau_is_some = kernel.x_tau_tau_reduced.is_some();
        let x_rij_zero = Array2::<f64>::zeros(x_r.raw_dim());
        let x_rij: &Array2<f64> = kernel.x_tau_tau_reduced.as_ref().unwrap_or(&x_rij_zero);
        let zeros_n = Array1::<f64>::zeros(n);
        let deta_ij = kernel.deta_ij_partial.as_ref().unwrap_or(&zeros_n);

        // ─────────────────────────────────────────────────────────────────
        //  η̇ vectors in β-rhs space (η_V := X V, η_{i,V} := X_i V, etc.)
        // ─────────────────────────────────────────────────────────────────
        let eta_v = self.x_dense.dot(rhs); // n×m
        let eta_i_v = x_tau_i.dot(rhs); // n×m
        let eta_j_v = x_tau_j.dot(rhs); // n×m
        // X_{ij} V from the reduced second-derivative design:
        //   reduce_explicit_design: X_{r,τ} = diag(√a) X_τ Q,
        //   invert:  X_{ij} = diag(1/√a) X_{r,ij} Qᵀ.
        let eta_ij_v: Array2<f64> = if x_tau_tau_is_some {
            let qt_v = self.q_basis.t().dot(rhs); // r×m
            let mut out = x_rij.dot(&qt_v); // n×m in sqrt(a)-scaled space
            if let Some(scale) = self.observation_weight_sqrt.as_ref() {
                for row in 0..out.nrows() {
                    let s = scale[row];
                    if s > 0.0 {
                        let inv = s.recip();
                        for col in 0..out.ncols() {
                            out[[row, col]] *= inv;
                        }
                    } else {
                        for col in 0..out.ncols() {
                            out[[row, col]] = 0.0;
                        }
                    }
                }
            }
            out
        } else {
            Array2::<f64>::zeros((n, m))
        };

        // ─────────────────────────────────────────────────────────────────
        //  Shared per-direction reduced operators
        //    A_τ = K İ K   (reduced analog of T_τ = K I_τ K)
        //    K̇_τ = -A_τ  (already cached)
        // ─────────────────────────────────────────────────────────────────
        let a_i_reduced = -dot_k_i; // K İ_i K = -K̇_i
        let a_j_reduced = -dot_k_j;

        // ─────────────────────────────────────────────────────────────────
        //  Ï_{ij}  — second cross derivative of reduced Fisher
        // ─────────────────────────────────────────────────────────────────
        //   Ï_{ij} = X_{r,ij}ᵀ W X_r + X_rᵀ W X_{r,ij}
        //          + X_{r,i}ᵀ W X_{r,j} + X_{r,j}ᵀ W X_{r,i}
        //          + X_{r,i}ᵀ Ẇ_j X_r + X_rᵀ Ẇ_j X_{r,i}
        //          + X_{r,j}ᵀ Ẇ_i X_r + X_rᵀ Ẇ_i X_{r,j}
        //          + X_rᵀ Ẅ_{ij} X_r.
        //   Ẇ_α   = diag(w' ⊙ η̇_α),  Ẅ_{ij} = diag(w'' ⊙ η̇_i ⊙ η̇_j + w' ⊙ η̈_ij).
        let dw_i = &self.w1 * deta_i;
        let dw_j = &self.w1 * deta_j;
        let ddw_ij = &(&self.w2 * &(deta_i * deta_j)) + &(&self.w1 * deta_ij);
        let mut i_ddot = Array2::<f64>::zeros(k.raw_dim());
        if x_tau_tau_is_some {
            i_ddot = i_ddot + RemlState::weighted_cross(x_rij, x_r, &self.w);
            i_ddot = i_ddot + RemlState::weighted_cross(x_r, x_rij, &self.w);
        }
        i_ddot = i_ddot + RemlState::weighted_cross(x_ri, x_rj, &self.w);
        i_ddot = i_ddot + RemlState::weighted_cross(x_rj, x_ri, &self.w);
        i_ddot = i_ddot + RemlState::weighted_cross(x_ri, x_r, &dw_j);
        i_ddot = i_ddot + RemlState::weighted_cross(x_r, x_ri, &dw_j);
        i_ddot = i_ddot + RemlState::weighted_cross(x_rj, x_r, &dw_i);
        i_ddot = i_ddot + RemlState::weighted_cross(x_r, x_rj, &dw_i);
        i_ddot = i_ddot + crate::faer_ndarray::fast_xt_diag_x(x_r, &ddw_ij);

        // K̈_{ij} = −K Ï K + K İ_i K İ_j K + K İ_j K İ_i K.
        //   Using K İ_α K = −K̇_α = a_α_reduced, the two product terms collapse to
        //   K İ_i K İ_j K = a_i_reduced · İ_j · K,
        //   K İ_j K İ_i K = a_j_reduced · İ_i · K.
        let k_ddot: Array2<f64> = -k.dot(&i_ddot).dot(k)
            + a_i_reduced.dot(dot_i_j).dot(k)
            + a_j_reduced.dot(dot_i_i).dot(k);

        // ─────────────────────────────────────────────────────────────────
        //  ḧ_{ij}
        // ─────────────────────────────────────────────────────────────────
        //   ḧ_ij = 2 diag(X_{r,ij} K X_rᵀ)
        //        + diag(X_r K̈_ij X_rᵀ)
        //        + 2 diag(X_{r,i} K̇_j X_rᵀ)
        //        + 2 diag(X_{r,j} K̇_i X_rᵀ)
        //        + 2 diag(X_{r,i} K X_{r,j}ᵀ).
        // Using diag(A Bᵀ) = rowwise_dot(A, B):
        let dh_ij: Array1<f64> = {
            let mut acc = Array1::<f64>::zeros(n);
            if x_tau_tau_is_some {
                let rij_k = x_rij.dot(k);
                acc = acc + 2.0 * Self::rowwise_dot(&rij_k, x_r);
            }
            let xr_kddot = x_r.dot(&k_ddot);
            acc = acc + Self::rowwise_dot(&xr_kddot, x_r);
            let ri_kdot_j = x_ri.dot(dot_k_j);
            acc = acc + 2.0 * Self::rowwise_dot(&ri_kdot_j, x_r);
            let rj_kdot_i = x_rj.dot(dot_k_i);
            acc = acc + 2.0 * Self::rowwise_dot(&rj_kdot_i, x_r);
            let ri_k = x_ri.dot(k);
            acc = acc + 2.0 * Self::rowwise_dot(&ri_k, x_rj);
            acc
        };

        // ─────────────────────────────────────────────────────────────────
        //  Γ, Γ̇_i, Γ̇_j, Γ̈_ij  (diagonal row-weight n-vectors)
        // ─────────────────────────────────────────────────────────────────
        //   γ        = w'' ⊙ h
        //   γ̇_i     = w''' ⊙ η̇_i ⊙ h + w'' ⊙ ḣ_i
        //   γ̈_ij   = w'''' ⊙ η̇_i ⊙ η̇_j ⊙ h
        //            + w''' ⊙ η̈_ij ⊙ h
        //            + w''' ⊙ η̇_i ⊙ ḣ_j
        //            + w''' ⊙ η̇_j ⊙ ḣ_i
        //            + w'' ⊙ ḧ_ij
        let gamma = &self.w2 * &self.h_diag;
        let gamma_dot_i = &(&(&self.w3 * deta_i) * &self.h_diag) + &(&self.w2 * dh_i);
        let gamma_dot_j = &(&(&self.w3 * deta_j) * &self.h_diag) + &(&self.w2 * dh_j);
        let gamma_ddot = &(&(&(&self.w4 * deta_i) * deta_j) * &self.h_diag)
            + &(&(&(&self.w3 * deta_ij) * &self.h_diag)
                + &(&(&self.w3 * deta_i) * dh_j)
                + &(&(&self.w3 * deta_j) * dh_i)
                + &(&self.w2 * &dh_ij));

        // ─────────────────────────────────────────────────────────────────
        //  Diagonal-term β-rhs contributions:
        //    ∂²(XᵀΓX)/∂τ_i∂τ_j · V
        //  = X_{ij}ᵀ (γ ⊙ η_V)       + Xᵀ (γ ⊙ η_{ij,V})         [if X_ij]
        //    + X_iᵀ (γ ⊙ η_{j,V})    + X_jᵀ (γ ⊙ η_{i,V})
        //    + X_iᵀ (γ̇_j ⊙ η_V)     + Xᵀ (γ̇_j ⊙ η_{i,V})
        //    + X_jᵀ (γ̇_i ⊙ η_V)     + Xᵀ (γ̇_i ⊙ η_{j,V})
        //    + Xᵀ (γ̈_ij ⊙ η_V).
        // ─────────────────────────────────────────────────────────────────
        let xt_i = x_tau_i.t();
        let xt_j = x_tau_j.t();

        let mut diag_term = Array2::<f64>::zeros((p, m));
        let gamma_col = gamma.view().insert_axis(Axis(1));
        let gamma_i_col = gamma_dot_i.view().insert_axis(Axis(1));
        let gamma_j_col = gamma_dot_j.view().insert_axis(Axis(1));
        let gamma_ij_col = gamma_ddot.view().insert_axis(Axis(1));

        // X_iᵀ (γ ⊙ η_{j,V}) + X_jᵀ (γ ⊙ η_{i,V})
        diag_term = diag_term + xt_i.dot(&(&eta_j_v * &gamma_col));
        diag_term = diag_term + xt_j.dot(&(&eta_i_v * &gamma_col));
        // X_iᵀ (γ̇_j ⊙ η_V) + X_jᵀ (γ̇_i ⊙ η_V)
        diag_term = diag_term + xt_i.dot(&(&eta_v * &gamma_j_col));
        diag_term = diag_term + xt_j.dot(&(&eta_v * &gamma_i_col));
        // Xᵀ (γ̇_j ⊙ η_{i,V}) + Xᵀ (γ̇_i ⊙ η_{j,V})
        diag_term = diag_term + self.x_dense_t.dot(&(&eta_i_v * &gamma_j_col));
        diag_term = diag_term + self.x_dense_t.dot(&(&eta_j_v * &gamma_i_col));
        // Xᵀ (γ̈_ij ⊙ η_V)
        diag_term = diag_term + self.x_dense_t.dot(&(&eta_v * &gamma_ij_col));
        // X_{ij}ᵀ (γ ⊙ η_V) + Xᵀ (γ ⊙ η_{ij,V})
        if x_tau_tau_is_some {
            // X_{ij}ᵀ = Q X_{r,ij}ᵀ · diag(1/√a)  (inverse of the reduce shim),
            // but caller supplies the reduced second-derivative design.  We
            // form X_{ij}ᵀ Y as q_basis · (X_{r,ij}ᵀ · diag(1/√a)·Y) = Q · X_{r,ij}ᵀ (Y unscaled).
            // When no observation weights, X_{r,ij} = X_{ij} Q and
            //   X_{ij}ᵀ Y = Q X_{r,ij}ᵀ Y.
            let y = &eta_v * &gamma_col;
            let xt_ij_y: Array2<f64> = if let Some(scale) = self.observation_weight_sqrt.as_ref() {
                let mut y_scaled = y.clone();
                for row in 0..y_scaled.nrows() {
                    let s = scale[row];
                    if s > 0.0 {
                        let inv = s.recip();
                        for col in 0..y_scaled.ncols() {
                            y_scaled[[row, col]] *= inv;
                        }
                    } else {
                        for col in 0..y_scaled.ncols() {
                            y_scaled[[row, col]] = 0.0;
                        }
                    }
                }
                self.q_basis.dot(&x_rij.t().dot(&y_scaled))
            } else {
                self.q_basis.dot(&x_rij.t().dot(&y))
            };
            diag_term = diag_term + xt_ij_y;
            diag_term = diag_term + self.x_dense_t.dot(&(&eta_ij_v * &gamma_col));
        }

        // ─────────────────────────────────────────────────────────────────
        //  BᵀPB branch — 9-term expansion.
        //
        //  Represent each B-like operator as an "n-row scaling vector for the
        //  X part plus tails along X_τ and X_{ij}".  For rhs V, define the
        //  row-scaled η-space blocks R(B) = diag(scale) X V + tails.  Then
        //  Bᵀ (P action) R is assembled by row-scaling and left-multiplying
        //  the appropriate full designs.
        // ─────────────────────────────────────────────────────────────────

        // B V row-block (eta-space):  B V = diag(w') X V.
        let w1_col = self.w1.view().insert_axis(Axis(1));
        let b_v = &eta_v * &w1_col;

        // Ḃ_i V = diag(w'' ⊙ η̇_i) X V + diag(w') X_i V.
        let w2_deta_i = &self.w2 * deta_i;
        let w2_deta_j = &self.w2 * deta_j;
        let w2_deta_i_col = w2_deta_i.view().insert_axis(Axis(1));
        let w2_deta_j_col = w2_deta_j.view().insert_axis(Axis(1));
        let bdot_i_v = &(&eta_v * &w2_deta_i_col) + &(&eta_i_v * &w1_col);
        let bdot_j_v = &(&eta_v * &w2_deta_j_col) + &(&eta_j_v * &w1_col);

        // B̈_{ij} V =
        //   diag(w''' ⊙ η̇_i ⊙ η̇_j + w'' ⊙ η̈_ij) X V
        //   + diag(w'' ⊙ η̇_i) X_j V
        //   + diag(w'' ⊙ η̇_j) X_i V
        //   + diag(w') X_{ij} V.
        let w3_didj = &(&self.w3 * deta_i) * deta_j;
        let w2_dij = &self.w2 * deta_ij;
        let bddot_scale = &w3_didj + &w2_dij;
        let bddot_scale_col = bddot_scale.view().insert_axis(Axis(1));
        let mut bddot_ij_v = &eta_v * &bddot_scale_col;
        bddot_ij_v = bddot_ij_v + &(&eta_j_v * &w2_deta_i_col);
        bddot_ij_v = bddot_ij_v + &(&eta_i_v * &w2_deta_j_col);
        bddot_ij_v = bddot_ij_v + &(&eta_ij_v * &w1_col);

        // P V  (columnwise, using K ⊙ K Hadamard gram on Z = X_r).
        let p_bv = RemlState::apply_hadamard_gram_to_matrix(z, k, k, &b_v);
        let p_bddot_ij_v =
            RemlState::apply_hadamard_gram_to_matrix(z, k, k, &bddot_ij_v);

        // Ṗ_i, Ṗ_j applied to B V, Ḃ_j V, Ḃ_i V — use the existing
        // apply_mtau_to_matrix helper, which computes 2(M ⊙ Ṁ_τ) · mat.
        //
        // Construct a lightweight "FirthTauPartialKernel"-shaped tuple only for
        // apply_mtau_to_matrix; we mirror its input contract inline to avoid
        // owning a FirthTauPartialKernel copy here.
        let pdot_i_bv = self.apply_mtau_from_reduced(x_ri, dot_k_i, &b_v);
        let pdot_j_bv = self.apply_mtau_from_reduced(x_rj, dot_k_j, &b_v);
        let pdot_i_bdot_j_v = self.apply_mtau_from_reduced(x_ri, dot_k_i, &bdot_j_v);
        let pdot_j_bdot_i_v = self.apply_mtau_from_reduced(x_rj, dot_k_j, &bdot_i_v);

        // P Ḃ_j V and P Ḃ_i V.
        let p_bdot_j_v = RemlState::apply_hadamard_gram_to_matrix(z, k, k, &bdot_j_v);
        let p_bdot_i_v = RemlState::apply_hadamard_gram_to_matrix(z, k, k, &bdot_i_v);

        // P̈_{ij} V = 4 (Ṁ_i ⊙ Ṁ_j) V  + 2 (M ⊙ M̈_{ij}) V.
        let p_ddot_b_v = self.apply_p_ddot_ij(
            x_r, x_ri, x_rj, x_rij, k, dot_k_i, dot_k_j, &k_ddot, x_tau_tau_is_some, &b_v,
        );

        // Assemble 9 terms of D²(BᵀPB)[τ_i, τ_j] · V.
        //   term1 = B̈_ijᵀ P B V + Bᵀ P B̈_ij V
        //   term2 = Ḃ_iᵀ P Ḃ_j V + Ḃ_jᵀ P Ḃ_i V
        //   term3 = Ḃ_iᵀ Ṗ_j B V + Bᵀ Ṗ_j Ḃ_i V
        //   term4 = Ḃ_jᵀ Ṗ_i B V + Bᵀ Ṗ_i Ḃ_j V
        //   term5 = Bᵀ P̈_ij B V
        //
        // "Bᵀ Q V" with B = diag(w') X equals left_scaled_xt(w1, Q V).
        // "Ḃ_iᵀ Q V" = diag(w'' ⊙ η̇_i) X acting on the left, plus
        //              diag(w') X_i on the left.  In transpose:
        //   Ḃ_iᵀ Q V = Xᵀ (diag(w'' ⊙ η̇_i) Q V) + X_iᵀ (diag(w') Q V).
        // "B̈_ijᵀ Q V" mirrors B̈_ij above in transpose.

        let apply_bdot_tau_t = |scale_deta: &Array1<f64>,
                                x_tau_mat: &Array2<f64>,
                                q_v: &Array2<f64>| {
            let scale_col = scale_deta.view().insert_axis(Axis(1));
            self.x_dense_t.dot(&(q_v * &scale_col))
                + x_tau_mat.t().dot(&(q_v * &w1_col))
        };

        let apply_bddot_ij_t = |q_v: &Array2<f64>| -> Array2<f64> {
            let scale_col_full = bddot_scale.view().insert_axis(Axis(1));
            let mut out = self.x_dense_t.dot(&(q_v * &scale_col_full));
            out = out + x_tau_j.t().dot(&(q_v * &w2_deta_i_col));
            out = out + x_tau_i.t().dot(&(q_v * &w2_deta_j_col));
            if x_tau_tau_is_some {
                // X_{ij}ᵀ (w1 ⊙ Q V)
                let y = q_v * &w1_col;
                let contrib: Array2<f64> = if let Some(scale) =
                    self.observation_weight_sqrt.as_ref()
                {
                    let mut y_scaled = y.clone();
                    for row in 0..y_scaled.nrows() {
                        let s = scale[row];
                        if s > 0.0 {
                            let inv = s.recip();
                            for col in 0..y_scaled.ncols() {
                                y_scaled[[row, col]] *= inv;
                            }
                        } else {
                            for col in 0..y_scaled.ncols() {
                                y_scaled[[row, col]] = 0.0;
                            }
                        }
                    }
                    self.q_basis.dot(&x_rij.t().dot(&y_scaled))
                } else {
                    self.q_basis.dot(&x_rij.t().dot(&y))
                };
                out = out + contrib;
            }
            out
        };

        // term1
        let t1a = apply_bddot_ij_t(&p_bv);
        let t1b = self.left_scaled_xt(&self.w1, &p_bddot_ij_v);
        // term2
        let t2a = apply_bdot_tau_t(&w2_deta_i, x_tau_i, &p_bdot_j_v);
        let t2b = apply_bdot_tau_t(&w2_deta_j, x_tau_j, &p_bdot_i_v);
        // term3: Ḃ_iᵀ Ṗ_j B V + Bᵀ Ṗ_j Ḃ_i V
        let t3a = apply_bdot_tau_t(&w2_deta_i, x_tau_i, &pdot_j_bv);
        let t3b = self.left_scaled_xt(&self.w1, &pdot_j_bdot_i_v);
        // term4: Ḃ_jᵀ Ṗ_i B V + Bᵀ Ṗ_i Ḃ_j V
        let t4a = apply_bdot_tau_t(&w2_deta_j, x_tau_j, &pdot_i_bv);
        let t4b = self.left_scaled_xt(&self.w1, &pdot_i_bdot_j_v);
        // term5
        let t5 = self.left_scaled_xt(&self.w1, &p_ddot_b_v);

        let d2_bpb = t1a + t1b + t2a + t2b + t3a + t3b + t4a + t4b + t5;

        0.5 * (diag_term - d2_bpb)
    }

    /// Pair-level exact Firth kernel at fixed β for a (τ_i, τ_j) outer
    /// coordinate pair.
    ///
    /// Returns the two SCALAR- and P-VECTOR-valued second-derivative
    /// objects that the unified REML evaluator threads into
    /// `HyperCoordPair::{a,g}` as additive Firth contributions, plus an
    /// optional prepared Primitive-A `FirthTauTauPartialKernel` that the
    /// pair-callback can reuse for the `b_operator` action.
    ///
    /// ═════════════════════════════════════════════════════════════════════
    ///  DERIVATIONS (fixed β, reduced-basis identifiable coords).
    ///
    ///  Φ = 0.5 log|I_r| − 0.5 log|S_r|,   K_r = I_r⁻¹,   G_r = diag(S_r⁻¹)
    ///  Φ_{τ_i}|β = 0.5 tr(K_r İ_{r,i}) − 0.5 tr(G_r Ṡ_{r,i}).
    ///
    /// ┌── pair.a scalar Φ_{τ_i τ_j}|β ────────────────────────────────────┐
    ///  ∂/∂τ_j [0.5 tr(K_r İ_{r,i})]
    ///    = 0.5 tr(K̇_{r,j} İ_{r,i}) + 0.5 tr(K_r Ï_{r,ij})
    ///    = −0.5 tr(K_r İ_{r,j} K_r İ_{r,i}) + 0.5 tr(K_r Ï_{r,ij})
    ///
    ///  ∂/∂τ_j [−0.5 tr(G_r Ṡ_{r,i})]
    ///    = −0.5 tr(Ġ_{r,j} Ṡ_{r,i}) − 0.5 tr(G_r S̈_{r,ij})
    ///  (G_r diagonal in canonical basis →
    ///   Ġ_{r,j}_kk = −G_r_kk² · diag(Ṡ_{r,j})_kk.)
    ///
    ///  Ï_{r,ij} is the same 9-term Fisher cross used by Primitive A
    ///  (see `hphi_tau_tau_partial_apply`:i_ddot block).
    ///
    ///  S̈_{r,ij} = X_{r,ij}^T X_r + X_r^T X_{r,ij}
    ///            + X_{r,i}^T X_{r,j} + X_{r,j}^T X_{r,i}.
    /// └─────────────────────────────────────────────────────────────────┘
    ///
    /// ┌── pair.g p-vector (gΦ)_{τ_i τ_j}|β ───────────────────────────────┐
    ///  (gΦ)_{τ_i} = 0.5 X_{τ_i}^T (w1 ⊙ h)
    ///              + 0.5 X^T [ (w2 ⊙ η̇_i) ⊙ h + w1 ⊙ ḣ_i ]
    ///
    ///  Differentiating wrt τ_j at fixed β, using η̇_α = X_α β, η̈_{ij} =
    ///  X_{ij} β (when x_tau_tau is provided, else 0), and ḣ_α, ḧ_{ij}
    ///  from Primitive A:
    ///
    ///  term_A = 0.5 ∂/∂τ_j [X_{τ_i}^T (w1 ⊙ h)]
    ///        = 0.5 X_{τ_i τ_j}^T (w1 ⊙ h)          [if X_{ij} present]
    ///        + 0.5 X_{τ_i}^T [ (w2 ⊙ η̇_j) ⊙ h + w1 ⊙ ḣ_j ]
    ///
    ///  term_B = 0.5 ∂/∂τ_j [X^T · v_{τ_i}] with
    ///            v_{τ_i} = (w2 ⊙ η̇_i) ⊙ h + w1 ⊙ ḣ_i
    ///        = 0.5 X_{τ_j}^T v_{τ_i}
    ///        + 0.5 X^T · v̇_{τ_i,τ_j}
    ///
    ///  where the inner derivative
    ///  v̇_{τ_i,τ_j} = (w3 ⊙ η̇_j ⊙ η̇_i) ⊙ h   (from ∂w2 = w3 ⊙ η̇_j)
    ///              + (w2 ⊙ η̈_ij) ⊙ h          (from ∂η̇_i = η̈_{ij})
    ///              + (w2 ⊙ η̇_i) ⊙ ḣ_j        (from ∂h = ḣ_j)
    ///              + (w2 ⊙ η̇_j) ⊙ ḣ_i        (from ∂w1 = w2 ⊙ η̇_j, ⊙ ḣ_i)
    ///              +  w1 ⊙ ḧ_{ij}             (from ∂ḣ_i = ḧ_{ij}).
    /// └─────────────────────────────────────────────────────────────────┘
    ///
    /// ALL Ï_{r,ij}, η̈_{ij}, ḣ_i, ḧ_{ij} computations are identical to
    /// those already computed inside Primitive A's `hphi_tau_tau_partial_apply`.
    /// We replicate only the pieces needed to yield the scalar and p-vector
    /// outputs to avoid computing the full p×m action when unnecessary.
    ///
    /// Allow(dead_code): scaffold until pair-callback threading wires the
    /// scalar / p-vector additive contributions into `HyperCoordPair::{a,g}`.
    #[allow(dead_code)]
    pub(crate) fn exact_tau_tau_kernel(
        &self,
        x_tau_i: &Array2<f64>,
        x_tau_j: &Array2<f64>,
        x_tau_tau: Option<&Array2<f64>>,
        beta: &Array1<f64>,
        include_hphi_tau_tau_kernel: bool,
    ) -> FirthTauTauExactKernel {
        let deta_i = x_tau_i.dot(beta);
        let deta_j = x_tau_j.dot(beta);
        let deta_ij = x_tau_tau.as_ref().map(|xij| xij.dot(beta));

        let x_tau_i_reduced = self.reduce_explicit_design(x_tau_i);
        let x_tau_j_reduced = self.reduce_explicit_design(x_tau_j);
        let x_tau_tau_reduced = x_tau_tau.map(|xij| self.reduce_explicit_design(xij));

        let (dot_i_i, dot_h_i) =
            self.dot_i_and_h_from_reduced(&x_tau_i_reduced, &deta_i);
        let (dot_i_j, dot_h_j) =
            self.dot_i_and_h_from_reduced(&x_tau_j_reduced, &deta_j);

        // Ï_{r,ij} = X_{r,ij}^T W X_r + X_r^T W X_{r,ij}
        //            + X_{r,i}^T W X_{r,j} + X_{r,j}^T W X_{r,i}
        //            + X_{r,i}^T Ẇ_j X_r + X_r^T Ẇ_j X_{r,i}
        //            + X_{r,j}^T Ẇ_i X_r + X_r^T Ẇ_i X_{r,j}
        //            + X_r^T Ẅ_{ij} X_r
        // Ẇ_α = diag(w' ⊙ η̇_α);  Ẅ_{ij} = diag(w'' ⊙ η̇_i ⊙ η̇_j + w' ⊙ η̈_{ij}).
        let zeros_n = Array1::<f64>::zeros(self.x_dense.nrows());
        let deta_ij_ref: &Array1<f64> = deta_ij.as_ref().unwrap_or(&zeros_n);
        let dw_i = &self.w1 * &deta_i;
        let dw_j = &self.w1 * &deta_j;
        let ddw_ij = &(&self.w2 * &(&deta_i * &deta_j)) + &(&self.w1 * deta_ij_ref);

        let x_r = &self.x_reduced;
        let mut i_ddot = Array2::<f64>::zeros(self.k_reduced.raw_dim());
        if let Some(x_rij) = x_tau_tau_reduced.as_ref() {
            i_ddot = i_ddot + RemlState::weighted_cross(x_rij, x_r, &self.w);
            i_ddot = i_ddot + RemlState::weighted_cross(x_r, x_rij, &self.w);
        }
        i_ddot = i_ddot + RemlState::weighted_cross(&x_tau_i_reduced, &x_tau_j_reduced, &self.w);
        i_ddot = i_ddot + RemlState::weighted_cross(&x_tau_j_reduced, &x_tau_i_reduced, &self.w);
        i_ddot = i_ddot + RemlState::weighted_cross(&x_tau_i_reduced, x_r, &dw_j);
        i_ddot = i_ddot + RemlState::weighted_cross(x_r, &x_tau_i_reduced, &dw_j);
        i_ddot = i_ddot + RemlState::weighted_cross(&x_tau_j_reduced, x_r, &dw_i);
        i_ddot = i_ddot + RemlState::weighted_cross(x_r, &x_tau_j_reduced, &dw_i);
        i_ddot = i_ddot + crate::faer_ndarray::fast_xt_diag_x(x_r, &ddw_ij);

        // pair.a likelihood contribution:
        //   0.5 tr(K_r Ï_{r,ij}) − 0.5 tr(K_r İ_{r,j} K_r İ_{r,i}).
        // K_r İ_{r,α} — reuse inline dot().
        let k = &self.k_reduced;
        let k_dot_i_i = k.dot(&dot_i_i);
        let k_dot_i_j = k.dot(&dot_i_j);
        let a_lik = 0.5 * RemlState::trace_product(k, &i_ddot)
            - 0.5 * RemlState::trace_product(&k_dot_i_j, &k_dot_i_i);

        // pair.a penalty-basis contribution:
        //   Ṡ_{r,α} = X_{r,α}^T X_r + X_r^T X_{r,α}
        //   S̈_{r,ij} = X_{r,ij}^T X_r + X_r^T X_{r,ij}
        //            + X_{r,i}^T X_{r,j} + X_{r,j}^T X_{r,i}
        //   tr(G_r Ṡ_{r,i}) = Σ_k G_r_kk · diag(Ṡ_{r,i})_kk
        //   tr(Ġ_{r,j} Ṡ_{r,i}) = −Σ_k G_r_kk² · diag(Ṡ_{r,j})_kk · diag(Ṡ_{r,i})_kk
        let dot_s_i =
            fast_atb(&x_tau_i_reduced, x_r) + fast_atb(x_r, &x_tau_i_reduced);
        let dot_s_j =
            fast_atb(&x_tau_j_reduced, x_r) + fast_atb(x_r, &x_tau_j_reduced);
        let mut s_ddot = Array2::<f64>::zeros(k.raw_dim());
        if let Some(x_rij) = x_tau_tau_reduced.as_ref() {
            s_ddot = s_ddot + fast_atb(x_rij, x_r) + fast_atb(x_r, x_rij);
        }
        s_ddot = s_ddot
            + fast_atb(&x_tau_i_reduced, &x_tau_j_reduced)
            + fast_atb(&x_tau_j_reduced, &x_tau_i_reduced);
        // −tr(Ġ_j Ṡ_i) = +Σ G²·diag(Ṡ_j)_kk·diag(Ṡ_i)_kk;
        // −tr(G_r S̈_ij) = −Σ G·diag(S̈_ij)_kk.
        let g_inv = &self.x_metric_reduced_inv_diag;
        let rdim = k.nrows();
        let mut a_pen = 0.0_f64;
        for kk in 0..rdim {
            a_pen += 0.5 * g_inv[kk] * g_inv[kk] * dot_s_j[[kk, kk]] * dot_s_i[[kk, kk]];
            a_pen -= 0.5 * g_inv[kk] * s_ddot[[kk, kk]];
        }
        let phi_tau_tau_partial = a_lik + a_pen;

        // ─── pair.g p-vector: (gΦ)_{τ_i τ_j}|β ──────────────────────────
        //
        // Assemble ḧ_{ij} identically to Primitive A's body.  We need:
        //   K̇_{r,α} = −K_r İ_{r,α} K_r,
        //   K̈_{r,ij} = −K_r Ï_{r,ij} K_r + K_r İ_{r,i} K_r İ_{r,j} K_r
        //                                 + K_r İ_{r,j} K_r İ_{r,i} K_r.
        // (Same expression used by Primitive A; rebuilt here so we don't
        //  depend on a pre-built FirthTauTauPartialKernel.)
        let dot_k_i = -k.dot(&dot_i_i).dot(k);
        let dot_k_j = -k.dot(&dot_i_j).dot(k);
        let a_i_red = -&dot_k_i; // K İ_i K
        let a_j_red = -&dot_k_j; // K İ_j K
        let k_ddot: Array2<f64> = -k.dot(&i_ddot).dot(k)
            + a_i_red.dot(&dot_i_j).dot(k)
            + a_j_red.dot(&dot_i_i).dot(k);

        // ḧ_{ij} = 2 diag(X_{r,ij} K X_r^T)
        //        + diag(X_r K̈_{ij} X_r^T)
        //        + 2 diag(X_{r,i} K̇_j X_r^T)
        //        + 2 diag(X_{r,j} K̇_i X_r^T)
        //        + 2 diag(X_{r,i} K X_{r,j}^T).
        let n = self.x_dense.nrows();
        let mut dh_ij = Array1::<f64>::zeros(n);
        if let Some(x_rij) = x_tau_tau_reduced.as_ref() {
            let rij_k = x_rij.dot(k);
            dh_ij = dh_ij + 2.0 * Self::rowwise_dot(&rij_k, x_r);
        }
        let xr_kddot = x_r.dot(&k_ddot);
        dh_ij = dh_ij + Self::rowwise_dot(&xr_kddot, x_r);
        let ri_kdot_j = x_tau_i_reduced.dot(&dot_k_j);
        dh_ij = dh_ij + 2.0 * Self::rowwise_dot(&ri_kdot_j, x_r);
        let rj_kdot_i = x_tau_j_reduced.dot(&dot_k_i);
        dh_ij = dh_ij + 2.0 * Self::rowwise_dot(&rj_kdot_i, x_r);
        let ri_k = x_tau_i_reduced.dot(k);
        dh_ij = dh_ij + 2.0 * Self::rowwise_dot(&ri_k, &x_tau_j_reduced);

        // term_A = 0.5 X_{τ_i τ_j}^T (w1 ⊙ h)
        //        + 0.5 X_{τ_i}^T [ (w2 ⊙ η̇_j) ⊙ h + w1 ⊙ ḣ_j ]
        let w1_h = &self.w1 * &self.h_diag;
        let mut gphi_tau_tau = Array1::<f64>::zeros(self.x_dense.ncols());
        if let Some(x_ij) = x_tau_tau.as_ref() {
            gphi_tau_tau = gphi_tau_tau + 0.5 * x_ij.t().dot(&w1_h);
        }
        let inner_j = &(&(&self.w2 * &deta_j) * &self.h_diag) + &(&self.w1 * &dot_h_j);
        gphi_tau_tau = gphi_tau_tau + 0.5 * x_tau_i.t().dot(&inner_j);

        // term_B pieces:  v_{τ_i} = (w2 ⊙ η̇_i) ⊙ h + w1 ⊙ ḣ_i
        let v_tau_i = &(&(&self.w2 * &deta_i) * &self.h_diag) + &(&self.w1 * &dot_h_i);
        gphi_tau_tau = gphi_tau_tau + 0.5 * x_tau_j.t().dot(&v_tau_i);

        // v̇_{τ_i,τ_j} =
        //    (w3 ⊙ η̇_j ⊙ η̇_i) ⊙ h
        //  + (w2 ⊙ η̈_{ij}) ⊙ h
        //  + (w2 ⊙ η̇_i) ⊙ ḣ_j
        //  + (w2 ⊙ η̇_j) ⊙ ḣ_i
        //  +  w1 ⊙ ḧ_{ij}.
        let mut v_dot_ij = &(&(&self.w3 * &deta_j) * &deta_i) * &self.h_diag;
        v_dot_ij = v_dot_ij + &(&(&self.w2 * deta_ij_ref) * &self.h_diag);
        v_dot_ij = v_dot_ij + &(&(&self.w2 * &deta_i) * &dot_h_j);
        v_dot_ij = v_dot_ij + &(&(&self.w2 * &deta_j) * &dot_h_i);
        v_dot_ij = v_dot_ij + &(&self.w1 * &dh_ij);
        gphi_tau_tau = gphi_tau_tau + 0.5 * self.x_dense.t().dot(&v_dot_ij);

        let tau_tau_kernel = if include_hphi_tau_tau_kernel {
            Some(FirthTauTauPartialKernel {
                x_tau_i_reduced,
                x_tau_j_reduced,
                deta_i_partial: deta_i,
                deta_j_partial: deta_j,
                dot_h_i_partial: dot_h_i,
                dot_h_j_partial: dot_h_j,
                dot_k_i_reduced: dot_k_i,
                dot_k_j_reduced: dot_k_j,
                dot_i_i_partial: dot_i_i,
                dot_i_j_partial: dot_i_j,
                x_tau_tau_reduced,
                deta_ij_partial: deta_ij,
            })
        } else {
            None
        };

        FirthTauTauExactKernel {
            phi_tau_tau_partial,
            gphi_tau_tau,
            tau_tau_kernel,
        }
    }

    /// Apply `Ṗ_τ V = 2 (M ⊙ Ṁ_τ) V` given the reduced τ-drift design
    /// `x_tau_reduced` and the reduced Fisher-inverse drift `dot_k_reduced`.
    ///
    /// This mirrors the body of `apply_mtau_to_matrix` but accepts the
    /// x_tau/dot_k pieces directly, letting Primitive A reuse the same
    /// matrix-free Ṗ_τ applies without owning a `FirthTauPartialKernel`.
    #[allow(dead_code)]
    fn apply_mtau_from_reduced(
        &self,
        x_tau_reduced: &Array2<f64>,
        dot_k_reduced: &Array2<f64>,
        mat: &Array2<f64>,
    ) -> Array2<f64> {
        if mat.nrows() != self.x_dense.nrows() || mat.ncols() == 0 {
            return Array2::<f64>::zeros(mat.raw_dim());
        }
        let mut out = Array2::<f64>::zeros(mat.raw_dim());
        for col in 0..mat.ncols() {
            let v = mat.column(col).to_owned();
            let szz = RemlState::reducedweighted_gram(&self.z_reduced, &v);
            let mzz = self.k_reduced.dot(&szz).dot(&self.k_reduced);
            let t1 = Self::rowwise_bilinear(&self.z_reduced, &mzz, x_tau_reduced);

            let szt =
                RemlState::reduced_crossweighted_gram(&self.z_reduced, x_tau_reduced, &v);
            let mzt = self.k_reduced.dot(&szt).dot(&self.k_reduced);
            let t2 = RemlState::reduced_diag_gram(&self.z_reduced, &mzt);

            let t3 = RemlState::apply_hadamard_gram(
                &self.z_reduced,
                &self.k_reduced,
                dot_k_reduced,
                &v,
            );

            let y = 2.0 * (t1 + t2 + t3);
            out.column_mut(col).assign(&y);
        }
        out
    }

    /// Apply `P̈_{ij} V = 4 (Ṁ_i ⊙ Ṁ_j) V + 2 (M ⊙ M̈_{ij}) V` columnwise.
    ///
    /// `M̈_{ij}` expands into 9 pieces `Y_α C Y_βᵀ`; `Ṁ_i ⊙ Ṁ_j` into 9 cross
    /// pieces `(Y_{1,α} B_{1,α} W_{1,α}ᵀ) ⊙ (Y_{2,β} B_{2,β} W_{2,β}ᵀ)`.  Both
    /// are evaluated via the matrix-free identities:
    ///
    ///   [(ZAZᵀ) ⊙ (YBWᵀ) v]_i   = rowwise_bilinear(Y, B · (Wᵀdiag(v)Z) · A, Z)_i,
    ///   [(YBWᵀ) ⊙ (Y'B'W'ᵀ) v]_i= rowwise_bilinear(Y, B · (Wᵀdiag(v)W') · B'ᵀ, Y')_i,
    ///
    /// with S := row-wise reducedweighted Gram.
    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    fn apply_p_ddot_ij(
        &self,
        x_r: &Array2<f64>,
        x_ri: &Array2<f64>,
        x_rj: &Array2<f64>,
        x_rij: &Array2<f64>,
        k: &Array2<f64>,
        dot_k_i: &Array2<f64>,
        dot_k_j: &Array2<f64>,
        k_ddot: &Array2<f64>,
        x_tau_tau_is_some: bool,
        mat: &Array2<f64>,
    ) -> Array2<f64> {
        let n = self.x_dense.nrows();
        let m = mat.ncols();
        if mat.nrows() != n || m == 0 {
            return Array2::<f64>::zeros(mat.raw_dim());
        }
        let mut out = Array2::<f64>::zeros((n, m));
        for col in 0..m {
            let v = mat.column(col).to_owned();
            // Shared reducedweighted Grams for this column.  Only the Grams
            // actually appearing in the 18 pieces below are computed.
            let s_zz = RemlState::reducedweighted_gram(x_r, &v); // Z'diag(v)Z
            let s_zj = RemlState::reduced_crossweighted_gram(x_r, x_rj, &v); // Z'diag(v)Y_j
            let s_iz = RemlState::reduced_crossweighted_gram(x_ri, x_r, &v); // Y_i'diag(v)Z
            let s_jz = RemlState::reduced_crossweighted_gram(x_rj, x_r, &v); // Y_j'diag(v)Z
            let s_ij = RemlState::reduced_crossweighted_gram(x_ri, x_rj, &v); // Y_i'diag(v)Y_j

            // ── 4 (Ṁ_i ⊙ Ṁ_j) v ──
            // Ṁ_i has three pieces:
            //   P_i,a = Y_i K Zᵀ          — Y=Y_i, B=K, W=Z
            //   P_i,b = Z K̇_i Zᵀ         — Y=Z,   B=K̇_i, W=Z
            //   P_i,c = Z K Y_iᵀ          — Y=Z,   B=K,  W=Y_i
            // And symmetrically for Ṁ_j with (i→j).
            //
            // For each cross pair (α, β), compute
            //   core = B_α · (W_αᵀ diag(v) W_β) · B_βᵀ,
            //   y_piece = rowwise_bilinear(Y_α, core, Y_β),
            // then sum all 9 and scale by 4.
            let mut mdot_mdot = Array1::<f64>::zeros(n);
            // (a_i, a_j): Y_i, K, Z  ×  Y_j, K, Z  → W_α=Z, W_β=Z, S = s_zz
            {
                let core = k.dot(&s_zz).dot(&k.t());
                mdot_mdot = mdot_mdot + Self::rowwise_bilinear(x_ri, &core, x_rj);
            }
            // (a_i, b_j): Y_i, K, Z  ×  Z, K̇_j, Z  → S = s_zz; core = K · s_zz · K̇_jᵀ
            {
                let core = k.dot(&s_zz).dot(&dot_k_j.t());
                mdot_mdot = mdot_mdot + Self::rowwise_bilinear(x_ri, &core, x_r);
            }
            // (a_i, c_j): Y_i, K, Z  ×  Z, K, Y_j  → S = s_zj; core = K · s_zj · Kᵀ
            {
                let core = k.dot(&s_zj).dot(&k.t());
                mdot_mdot = mdot_mdot + Self::rowwise_bilinear(x_ri, &core, x_r);
            }
            // (b_i, a_j): Z, K̇_i, Z  ×  Y_j, K, Z  → S = s_zz; core = K̇_i · s_zz · Kᵀ
            {
                let core = dot_k_i.dot(&s_zz).dot(&k.t());
                mdot_mdot = mdot_mdot + Self::rowwise_bilinear(x_r, &core, x_rj);
            }
            // (b_i, b_j): Z, K̇_i, Z  ×  Z, K̇_j, Z  → S = s_zz; core = K̇_i · s_zz · K̇_jᵀ
            {
                let core = dot_k_i.dot(&s_zz).dot(&dot_k_j.t());
                mdot_mdot = mdot_mdot + Self::rowwise_bilinear(x_r, &core, x_r);
            }
            // (b_i, c_j): Z, K̇_i, Z  ×  Z, K, Y_j  → S = s_zj; core = K̇_i · s_zj · Kᵀ
            {
                let core = dot_k_i.dot(&s_zj).dot(&k.t());
                mdot_mdot = mdot_mdot + Self::rowwise_bilinear(x_r, &core, x_r);
            }
            // (c_i, a_j): Z, K, Y_i  ×  Y_j, K, Z  → S = Y_iᵀ diag(v) Z = s_iz;
            //   core = K · s_iz · Kᵀ
            {
                let core = k.dot(&s_iz).dot(&k.t());
                mdot_mdot = mdot_mdot + Self::rowwise_bilinear(x_r, &core, x_rj);
            }
            // (c_i, b_j): Z, K, Y_i  ×  Z, K̇_j, Z  → S = Y_iᵀ diag(v) Z = s_iz;
            //   core = K · s_iz · K̇_jᵀ
            {
                let core = k.dot(&s_iz).dot(&dot_k_j.t());
                mdot_mdot = mdot_mdot + Self::rowwise_bilinear(x_r, &core, x_r);
            }
            // (c_i, c_j): Z, K, Y_i  ×  Z, K, Y_j  → S = Y_iᵀ diag(v) Y_j = s_ij;
            //   core = K · s_ij · Kᵀ
            {
                let core = k.dot(&s_ij).dot(&k.t());
                mdot_mdot = mdot_mdot + Self::rowwise_bilinear(x_r, &core, x_r);
            }

            // ── 2 (M ⊙ M̈_{ij}) v ──
            // Each piece has the form Y_α C W_βᵀ; M = Z K Zᵀ with A=K.
            // Identity:  [(ZAZᵀ) ⊙ (Y_α C W_βᵀ) v]_i
            //          = rowwise_bilinear(Y_α, C · (W_βᵀ diag(v) Z) · A, Z).
            let mut m_mddot = Array1::<f64>::zeros(n);
            // (a) Y_α = X_{r,ij}, C = K, W_β = X_r  → W_βᵀ diag(v) Z = s_zz
            if x_tau_tau_is_some {
                let core = k.dot(&s_zz).dot(k);
                m_mddot = m_mddot + Self::rowwise_bilinear(x_rij, &core, x_r);
            }
            // (b) Y_α = X_r, C = K, W_β = X_{r,ij} → W_βᵀ diag(v) Z = X_{r,ij}ᵀ diag(v) Z
            if x_tau_tau_is_some {
                let s_ijz = RemlState::reduced_crossweighted_gram(x_rij, x_r, &v);
                let core = k.dot(&s_ijz).dot(k);
                m_mddot = m_mddot + Self::rowwise_bilinear(x_r, &core, x_r);
            }
            // (c) Y_α = X_{r,i}, C = K̇_j, W_β = X_r → S = s_zz
            {
                let core = dot_k_j.dot(&s_zz).dot(k);
                m_mddot = m_mddot + Self::rowwise_bilinear(x_ri, &core, x_r);
            }
            // (d) Y_α = X_r, C = K̇_j, W_β = X_{r,i} → W_βᵀ diag(v) Z = s_iz
            {
                let core = dot_k_j.dot(&s_iz).dot(k);
                m_mddot = m_mddot + Self::rowwise_bilinear(x_r, &core, x_r);
            }
            // (e) Y_α = X_{r,j}, C = K̇_i, W_β = X_r → S = s_zz
            {
                let core = dot_k_i.dot(&s_zz).dot(k);
                m_mddot = m_mddot + Self::rowwise_bilinear(x_rj, &core, x_r);
            }
            // (f) Y_α = X_r, C = K̇_i, W_β = X_{r,j} → W_βᵀ diag(v) Z = s_jz
            {
                let core = dot_k_i.dot(&s_jz).dot(k);
                m_mddot = m_mddot + Self::rowwise_bilinear(x_r, &core, x_r);
            }
            // (g) Y_α = X_{r,i}, C = K, W_β = X_{r,j} → W_βᵀ diag(v) Z = s_jz
            {
                let core = k.dot(&s_jz).dot(k);
                m_mddot = m_mddot + Self::rowwise_bilinear(x_ri, &core, x_r);
            }
            // (h) Y_α = X_{r,j}, C = K, W_β = X_{r,i} → W_βᵀ diag(v) Z = s_iz
            {
                let core = k.dot(&s_iz).dot(k);
                m_mddot = m_mddot + Self::rowwise_bilinear(x_rj, &core, x_r);
            }
            // (i) Y_α = X_r, C = K̈_ij, W_β = X_r → S = s_zz
            {
                let core = k_ddot.dot(&s_zz).dot(k);
                m_mddot = m_mddot + Self::rowwise_bilinear(x_r, &core, x_r);
            }

            let col_out = 4.0 * mdot_mdot + 2.0 * m_mddot;
            out.column_mut(col).assign(&col_out);
        }
        out
    }

    /// Primitive B — prepare step: assemble the reduced kernel for
    /// D_β((H_φ)_τ|_β)[v].
    ///
    /// Consumes the existing `FirthTauPartialKernel`, the τ-drift partials
    /// (`deta_partial = η̇_τ = X_τ β` and `dot_i_partial = İ_τ`), the
    /// β-direction `FirthDirection` built from `deta = X v`, and
    /// `x_tau_v = X_τ v`, and returns a cached kernel carrying the mixed
    /// β-τ reduced quantities A_v, dh_v, D_β(İ_τ)[v], D_β(K̇_τ)[v],
    /// D_β(ḣ_τ)[v], and the w-chain derivatives needed by
    /// `d_beta_hphi_tau_partial_apply`.
    pub(crate) fn d_beta_hphi_tau_partial_prepare_from_partials(
        &self,
        tau_kernel: &FirthTauPartialKernel,
        deta_partial: &Array1<f64>,
        dot_i_partial: &Array2<f64>,
        beta_direction: &FirthDirection,
        x_tau_v: &Array1<f64>,
    ) -> FirthTauBetaPartialKernel {
        // D_β(İ_τ)[v] — three-piece symmetric form from the product rule on
        //   İ_τ = X_{r,τ}ᵀ W X_r + X_rᵀ W X_{r,τ} + X_rᵀ diag(w' ⊙ η̇_τ) X_r,
        // where W = diag(w(η)) is the Fisher weight (not its derivative).
        // The β-differential hits w (through η=Xβ) and η̇_τ (through X_τ β):
        //   D_β(X_{r,τ}ᵀ W X_r)[v] = X_{r,τ}ᵀ diag(w' ⊙ δη_v) X_r,
        //   D_β(X_rᵀ diag(w' ⊙ η̇_τ) X_r)[v]
        //     = X_rᵀ diag(w'' ⊙ η̇_τ ⊙ δη_v + w' ⊙ δη_{τ,v}) X_r,
        // where δη_v = beta_direction.deta, δη_{τ,v} = x_tau_v.
        // s_v := w' ⊙ δη_v (same weight the FirthDirection uses to build
        // g_u_reduced); b_vvec := w'' ⊙ δη_v = beta_direction.b_uvec is the
        // weight for the third-term product-rule piece.
        let s_v = &self.w1 * &beta_direction.deta;
        let mixed_diag_weight =
            &(&tau_kernel.dotw1 * &beta_direction.deta) + &(&self.w1 * x_tau_v);
        let cross1 = RemlState::reduced_crossweighted_gram(
            &tau_kernel.x_tau_reduced,
            &self.x_reduced,
            &s_v,
        );
        let cross2 = RemlState::reduced_crossweighted_gram(
            &self.x_reduced,
            &tau_kernel.x_tau_reduced,
            &s_v,
        );
        let diag_piece = RemlState::reducedweighted_gram(&self.x_reduced, &mixed_diag_weight);
        let d_beta_dot_i = &cross1 + &cross2 + &diag_piece;

        // D_β(K̇_τ)[v] — direct Leibniz on K̇_τ = -K_r İ_τ K_r with
        //   D_β K_r[v] = -K_r I'_v K_r = -beta_direction.a_u_reduced.
        // Expanding yields
        //   D_β K̇_τ[v] = +A_v İ_τ K_r − K_r D_β(İ_τ)[v] K_r + K_r İ_τ A_v,
        // where A_v := beta_direction.a_u_reduced = +K_r I'_v K_r.  The
        // FirthDirection carries a_u with the opposite sign convention to the
        // derivation block's "A_v"; we keep the direction convention and
        // compose signs correctly here.
        let term_a = beta_direction
            .a_u_reduced
            .dot(dot_i_partial)
            .dot(&self.k_reduced);
        let term_b = self.k_reduced.dot(&d_beta_dot_i).dot(&self.k_reduced);
        let term_c = self
            .k_reduced
            .dot(dot_i_partial)
            .dot(&beta_direction.a_u_reduced);
        let d_beta_dot_k = &term_a - &term_b + &term_c;

        // D_β(ḣ_τ)[v] — β-differential of
        //   ḣ_τ = 2·diag(X_{r,τ} K_r X_rᵀ) + diag(X_r K̇_τ X_rᵀ):
        //   D_β ḣ_τ[v]
        //     = 2·diag(X_{r,τ} D_β K_r[v] X_rᵀ) + diag(X_r D_β K̇_τ[v] X_rᵀ)
        //     = -2·diag(X_{r,τ} A_v X_rᵀ) + diag(X_r (D_β K̇_τ[v]) X_rᵀ).
        let cross_diag = Self::rowwise_bilinear(
            &tau_kernel.x_tau_reduced,
            &beta_direction.a_u_reduced,
            &self.x_reduced,
        );
        let inner_diag = RemlState::reduced_diag_gram(&self.x_reduced, &d_beta_dot_k);
        let d_beta_dot_h = -2.0 * &cross_diag + &inner_diag;

        FirthTauBetaPartialKernel {
            x_tau_reduced: tau_kernel.x_tau_reduced.clone(),
            deta_partial: deta_partial.clone(),
            dot_h_partial: tau_kernel.dot_h_partial.clone(),
            dot_i_partial: dot_i_partial.clone(),
            dot_k_reduced: tau_kernel.dot_k_reduced.clone(),
            deta_v: beta_direction.deta.clone(),
            deta_tau_v: x_tau_v.clone(),
            g_v_reduced: beta_direction.g_u_reduced.clone(),
            a_v_reduced: beta_direction.a_u_reduced.clone(),
            dh_v: beta_direction.dh.clone(),
            b_vvec: beta_direction.b_uvec.clone(),
            d_beta_dot_i,
            d_beta_dot_k,
            d_beta_dot_h,
        }
    }

    /// Apply the mixed β-τ P-action `P_{τ,v} · mat` to an n×m column block.
    ///
    /// Expansion:
    ///   P_{τ,v} = 2 (M_v ⊙ M_τ) + 2 (M ⊙ M_{τ,v}),
    ///     M_v     = X_r K̇_v X_rᵀ,  K̇_v = -A_v (A_v = a_v_reduced),
    ///     M_τ     = X_{r,τ} K_r X_rᵀ + X_r K_r X_{r,τ}ᵀ + X_r K̇_τ X_rᵀ,
    ///     M_{τ,v} = X_{r,τ} K̇_v X_rᵀ + X_r K̇_v X_{r,τ}ᵀ + X_r D_β K̇_τ[v] X_rᵀ.
    /// Hadamard-Gram pieces are evaluated column-wise via
    ///   ((Z M_A Wᵀ) ⊙ (Y M_B Xᵀ)) v row-i
    ///       = z_iᵀ M_A (Wᵀ diag(v) X) M_Bᵀ y_i.
    fn apply_p_tau_v_to_matrix(
        &self,
        kernel: &FirthTauBetaPartialKernel,
        mat: &Array2<f64>,
    ) -> Array2<f64> {
        let n = self.x_dense.nrows();
        if mat.nrows() != n || mat.ncols() == 0 {
            return Array2::<f64>::zeros(mat.raw_dim());
        }
        let z = &self.z_reduced;
        let z_tau = &kernel.x_tau_reduced;
        let k_r = &self.k_reduced;
        let a_v = &kernel.a_v_reduced; // = +K_r I'_v K_r  (so K̇_v = -a_v)
        let dot_k_tau = &kernel.dot_k_reduced; // K̇_τ = -K_r İ_τ K_r
        let d_beta_dot_k = &kernel.d_beta_dot_k; // D_β K̇_τ[v]
        let mut out = Array2::<f64>::zeros(mat.raw_dim());
        for col in 0..mat.ncols() {
            let v = mat.column(col).to_owned();
            let s_zz = RemlState::reducedweighted_gram(z, &v);
            let s_z_ztau = RemlState::reduced_crossweighted_gram(z, z_tau, &v);

            // Piece 1: (X_r K̇_v X_rᵀ ⊙ X_{r,τ} K_r X_rᵀ) · v
            //   = -rowwise_bilinear(Z, a_v S_zz K_r, Z_τ).
            let mid_1 = a_v.dot(&s_zz).dot(k_r);
            let t1 = -Self::rowwise_bilinear(z, &mid_1, z_tau);
            // Piece 2: (X_r K̇_v X_rᵀ ⊙ X_r K_r X_{r,τ}ᵀ) · v
            //   = -reduced_diag_gram(Z, a_v S_z_ztau K_r).
            let mid_2 = a_v.dot(&s_z_ztau).dot(k_r);
            let t2 = -RemlState::reduced_diag_gram(z, &mid_2);
            // Piece 3: (X_r K̇_v X_rᵀ ⊙ X_r K̇_τ X_rᵀ) · v
            //   = -reduced_diag_gram(Z, a_v S_zz K̇_τ).
            let mid_3 = a_v.dot(&s_zz).dot(dot_k_tau);
            let t3 = -RemlState::reduced_diag_gram(z, &mid_3);
            // Piece 4: (M ⊙ X_{r,τ} K̇_v X_rᵀ) · v
            //   = -rowwise_bilinear(Z, K_r S_zz a_v, Z_τ).
            let mid_4 = k_r.dot(&s_zz).dot(a_v);
            let t4 = -Self::rowwise_bilinear(z, &mid_4, z_tau);
            // Piece 5: (M ⊙ X_r K̇_v X_{r,τ}ᵀ) · v
            //   = -reduced_diag_gram(Z, K_r S_z_ztau a_v).
            let mid_5 = k_r.dot(&s_z_ztau).dot(a_v);
            let t5 = -RemlState::reduced_diag_gram(z, &mid_5);
            // Piece 6: (M ⊙ X_r D_β K̇_τ[v] X_rᵀ) · v.
            let t6 = RemlState::apply_hadamard_gram(z, k_r, d_beta_dot_k, &v);

            // P_{τ,v} = 2·(pieces 1-3) + 2·(pieces 4-6); each group contributes
            // with the same outer factor 2.
            let y = 2.0 * (t1 + t2 + t3 + t4 + t5 + t6);
            out.column_mut(col).assign(&y);
        }
        out
    }

    pub(crate) fn d_beta_hphi_tau_partial_apply(
        &self,
        x_tau: &Array2<f64>,
        kernel: &FirthTauBetaPartialKernel,
        rhs: &Array2<f64>,
    ) -> Array2<f64> {
        let p = self.x_dense.ncols();
        if rhs.nrows() != p {
            return Array2::<f64>::zeros((p, rhs.ncols()));
        }
        if rhs.ncols() == 0 || p == 0 {
            return Array2::<f64>::zeros((p, rhs.ncols()));
        }
        // Matrix-free block apply of D_β((H_φ)_τ|_β)[v] evaluated on a rhs V.
        // Structure follows hphi_tau_partial_apply but replaces every weight
        // and every reduced Gram with its β-derivative in direction v:
        //
        //   (H_φ)_τ|_β (V) = 0.5 [X_τᵀ r(V) + Xᵀ r_τ(V)].
        //
        // D_β[v] leaves X, X_τ fixed and acts on r, r_τ:
        //   D_β((H_φ)_τ|_β)[v](V) = 0.5 [X_τᵀ D_β r(V)[v] + Xᵀ D_β r_τ(V)[v]].
        let etav = self.x_dense.dot(rhs);
        let etav_tau = x_tau.dot(rhs);
        let deta_v = &kernel.deta_v;
        let deta_tau_v = &kernel.deta_tau_v;
        let eta_tau = &kernel.deta_partial;
        let dot_h = &kernel.dot_h_partial;

        // Reuse τ-kernel weights.  dotw1 = w'' ⊙ η̇_τ, dotw2 = w''' ⊙ η̇_τ.
        let dotw1 = &self.w2 * eta_tau;
        let dotw2 = &self.w3 * eta_tau;

        // β-derivative scaling vectors in direction v:
        //   c_v              = D_β(w''·h)[v]    = w'''·δη_v·h + w''·dh_v
        //   b_vvec           = D_β(w')[v]       = w''·δη_v   (= kernel.b_vvec)
        //   d_beta_dotw1_vec = D_β(w''·η̇_τ)[v]  = w'''·δη_v·η̇_τ + w''·δη_{τ,v}
        //   d_beta_dotw2_vec = D_β(w'''·η̇_τ)[v] = w''''·δη_v·η̇_τ + w'''·δη_{τ,v}
        let c_v = &(&(&self.w3 * deta_v) * &self.h_diag) + &(&self.w2 * &kernel.dh_v);
        let b_vvec = &kernel.b_vvec;
        let d_beta_dotw1_vec =
            &(&(&self.w3 * deta_v) * eta_tau) + &(&self.w2 * deta_tau_v);
        let d_beta_dotw2_vec =
            &(&(&self.w4 * deta_v) * eta_tau) + &(&self.w3 * deta_tau_v);

        // Single-τ pieces (identical to hphi_tau_partial_apply).
        let qv = &etav * &self.w1.view().insert_axis(Axis(1));
        let qv_tau = &etav * &dotw1.view().insert_axis(Axis(1))
            + &etav_tau * &self.w1.view().insert_axis(Axis(1));
        let m_qv = self.apply_pbar_to_matrix(&qv);
        // apply_mtau_to_matrix only reads x_tau_reduced and dot_k_reduced off
        // the τ-kernel, but owning the full struct is cheap.
        let tau_kernel_view = FirthTauPartialKernel {
            dotw1: dotw1.clone(),
            dotw2: dotw2.clone(),
            dot_h_partial: dot_h.clone(),
            x_tau_reduced: kernel.x_tau_reduced.clone(),
            dot_k_reduced: kernel.dot_k_reduced.clone(),
        };
        let m_qv_tau = self.apply_mtau_to_matrix(&tau_kernel_view, &qv)
            + self.apply_pbar_to_matrix(&qv_tau);

        // β-derivatives of the single-τ pieces:
        //   D_β qv     = etav · D_β w'[v]       = etav · b_vvec
        //   D_β qv_tau = etav · D_β dotw1[v] + etav_tau · D_β w'[v]
        let d_beta_qv = &etav * &b_vvec.view().insert_axis(Axis(1));
        let d_beta_qv_tau = &etav * &d_beta_dotw1_vec.view().insert_axis(Axis(1))
            + &etav_tau * &b_vvec.view().insert_axis(Axis(1));

        //   D_β m_qv = P_v · qv + P · D_β qv
        let d_beta_m_qv = self.apply_p_u_to_matrix(&kernel.a_v_reduced, &qv)
            + self.apply_pbar_to_matrix(&d_beta_qv);

        //   D_β m_qv_tau = P_{τ,v}·qv + P_τ·D_β qv + P_v·qv_tau + P·D_β qv_tau
        let d_beta_m_qv_tau = self.apply_p_tau_v_to_matrix(kernel, &qv)
            + self.apply_mtau_to_matrix(&tau_kernel_view, &d_beta_qv)
            + self.apply_p_u_to_matrix(&kernel.a_v_reduced, &qv_tau)
            + self.apply_pbar_to_matrix(&d_beta_qv_tau);

        // D_β rv[v] where rv = etav·(w''·h) − w'·m_qv:
        //   D_β rv[v] = etav·c_v − b_vvec·m_qv − w'·D_β m_qv.
        let d_beta_rv = &etav * &c_v.view().insert_axis(Axis(1))
            - &m_qv * &b_vvec.view().insert_axis(Axis(1))
            - &d_beta_m_qv * &self.w1.view().insert_axis(Axis(1));

        // D_β rv_tau[v] where
        //   rv_tau = etav·dotw2·h + etav_tau·w''·h + etav·w''·dot_h
        //            − m_qv·dotw1 − m_qv_tau·w'.
        //
        //   D_β(dotw2·h)[v]   = (w''''·δη_v·η̇_τ + w'''·δη_{τ,v})·h
        //                        + dotw2·dh_v,
        //   D_β(w''·h)[v]     = c_v,
        //   D_β(w''·dot_h)[v] = w'''·δη_v·dot_h + w''·D_β dot_h[v],
        //   D_β dotw1[v]      = d_beta_dotw1_vec,
        //   D_β w'[v]         = b_vvec.
        let d_beta_dotw2_h = &(&d_beta_dotw2_vec * &self.h_diag) + &(&dotw2 * &kernel.dh_v);
        let d_beta_w2_doth =
            &(&(&self.w3 * deta_v) * dot_h) + &(&self.w2 * &kernel.d_beta_dot_h);

        let d_beta_rv_tau = &etav * &d_beta_dotw2_h.view().insert_axis(Axis(1))
            + &etav_tau * &c_v.view().insert_axis(Axis(1))
            + &etav * &d_beta_w2_doth.view().insert_axis(Axis(1))
            - &d_beta_m_qv * &dotw1.view().insert_axis(Axis(1))
            - &m_qv * &d_beta_dotw1_vec.view().insert_axis(Axis(1))
            - &d_beta_m_qv_tau * &self.w1.view().insert_axis(Axis(1))
            - &m_qv_tau * &b_vvec.view().insert_axis(Axis(1));

        0.5 * (x_tau.t().dot(&d_beta_rv) + self.x_dense.t().dot(&d_beta_rv_tau))
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
            println!("row {i}: z={z:.6} op.w4={} w4fd={} diff={}",
                op.w4[i], w4fd, (op.w4[i] - w4fd).abs());
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

    /// Verify the Primitive A body (`hphi_tau_tau_partial_apply`) against a
    /// finite-difference reference of the single-τ Primitive (
    /// `hphi_tau_partial_apply`).
    ///
    /// Identity under test:
    ///     ∂/∂τ_j  { (H_φ)_τ_i |_β · V }   =   ∂²H_φ/∂τ_i ∂τ_j |_β · V.
    ///
    /// Central-difference reference:
    ///   1. Evaluate the single-τ primitive at x, and at x ± h·X_τ_j
    ///      — rebuild the FirthDenseOperator (with fresh identifiable Q)
    ///      at each perturbed design; H_φ applied to a p-space rhs is
    ///      basis-invariant in unreduced β-coords, so Q rotation does not
    ///      contaminate the comparison.
    ///   2. FD_{i,j} = (T_{plus} − T_{minus}) / (2h) with T = hphi_tau_i_apply(V).
    ///   3. Contract both (i,j) and (j,i) directions and verify symmetry
    ///      of the analytic as a cross-check.
    ///
    /// Tolerance: 1e-7 relative max-abs (h chosen to balance truncation
    /// error at ~h² and evaluator roundoff at ~ε/h).
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

        // Reproducible small rhs block (p × m).
        let m = 3usize;
        let mut rhs = Array2::<f64>::zeros((p, m));
        let vals = [
            0.21, -0.44, 0.17, 0.38, 0.05, -0.22, -0.11, 0.27, 0.31,
        ];
        for r in 0..p {
            for c in 0..m {
                rhs[[r, c]] = vals[(r * m + c) % vals.len()];
            }
        }

        // ── Analytic τ×τ pair apply at base design (x_tau_tau = None,
        //    deta_ij = None, i.e. design is linear in τ).
        let x_tau_i_reduced = op.reduce_explicit_design(&x_tau_i);
        let x_tau_j_reduced = op.reduce_explicit_design(&x_tau_j);
        let deta_i = x_tau_i.dot(&beta);
        let deta_j = x_tau_j.dot(&beta);
        let (dot_i_i, dot_h_i) = op.dot_i_and_h_from_reduced(&x_tau_i_reduced, &deta_i);
        let (dot_i_j, dot_h_j) = op.dot_i_and_h_from_reduced(&x_tau_j_reduced, &deta_j);

        let kernel_ij = op.hphi_tau_tau_partial_prepare_from_partials(
            x_tau_i_reduced.clone(),
            x_tau_j_reduced.clone(),
            &deta_i,
            &deta_j,
            dot_h_i.clone(),
            dot_h_j.clone(),
            dot_i_i.clone(),
            dot_i_j.clone(),
            None,
            None,
        );
        let kernel_ji = op.hphi_tau_tau_partial_prepare_from_partials(
            x_tau_j_reduced,
            x_tau_i_reduced,
            &deta_j,
            &deta_i,
            dot_h_j,
            dot_h_i,
            dot_i_j,
            dot_i_i,
            None,
            None,
        );
        let analytic_ij = op.hphi_tau_tau_partial_apply(&x_tau_i, &x_tau_j, &kernel_ij, &rhs);
        let analytic_ji = op.hphi_tau_tau_partial_apply(&x_tau_j, &x_tau_i, &kernel_ji, &rhs);

        // Symmetry cross-check (Clairaut): ∂²H/∂τ_i∂τ_j = ∂²H/∂τ_j∂τ_i.
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

        // ── FD reference: central difference of single-τ primitive in
        //    τ_j direction, evaluated along τ_i.
        let h = 1e-5_f64;
        let fd_block = |x_eval: &Array2<f64>| -> Array2<f64> {
            let eta_e = x_eval.dot(&beta);
            let op_e =
                FirthDenseOperator::build(x_eval, &eta_e).expect("perturbed firth operator");
            let x_tau_i_r = op_e.reduce_explicit_design(&x_tau_i);
            let deta_i_e = x_tau_i.dot(&beta);
            let (dot_i_i_e, dot_h_i_e) =
                op_e.dot_i_and_h_from_reduced(&x_tau_i_r, &deta_i_e);
            let kernel_i_e = op_e.hphi_tau_partial_prepare_from_partials(
                x_tau_i_r,
                &deta_i_e,
                dot_h_i_e,
                dot_i_i_e,
            );
            op_e.hphi_tau_partial_apply(&x_tau_i, &kernel_i_e, &rhs)
        };
        let x_plus = &x + &(h * &x_tau_j);
        let x_minus = &x - &(h * &x_tau_j);
        let fd_ij = (&fd_block(&x_plus) - &fd_block(&x_minus)) / (2.0 * h);

        // ── Compare analytic_ij (contracted against V along τ_j→analytic's
        //    second index) to fd_ij (FD of T_i in τ_j direction).
        let rel_max_abs_diff = |a: &Array2<f64>, b: &Array2<f64>| -> f64 {
            let scale = a
                .iter()
                .chain(b.iter())
                .map(|v| v.abs())
                .fold(0.0_f64, f64::max)
                .max(1.0);
            let max_diff = (a - b).iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            max_diff / scale
        };
        let err_ij = rel_max_abs_diff(&analytic_ij, &fd_ij);

        // Also FD the other direction and compare to analytic_ji, to
        // double-cover the primitive.
        let fd_block_j = |x_eval: &Array2<f64>| -> Array2<f64> {
            let eta_e = x_eval.dot(&beta);
            let op_e =
                FirthDenseOperator::build(x_eval, &eta_e).expect("perturbed firth operator");
            let x_tau_j_r = op_e.reduce_explicit_design(&x_tau_j);
            let deta_j_e = x_tau_j.dot(&beta);
            let (dot_i_j_e, dot_h_j_e) =
                op_e.dot_i_and_h_from_reduced(&x_tau_j_r, &deta_j_e);
            let kernel_j_e = op_e.hphi_tau_partial_prepare_from_partials(
                x_tau_j_r,
                &deta_j_e,
                dot_h_j_e,
                dot_i_j_e,
            );
            op_e.hphi_tau_partial_apply(&x_tau_j, &kernel_j_e, &rhs)
        };
        let x_plus_i = &x + &(h * &x_tau_i);
        let x_minus_i = &x - &(h * &x_tau_i);
        let fd_ji = (&fd_block_j(&x_plus_i) - &fd_block_j(&x_minus_i)) / (2.0 * h);
        let err_ji = rel_max_abs_diff(&analytic_ji, &fd_ji);

        let tol = 1e-7_f64;
        assert!(
            err_ij < tol,
            "∂²H_φ/∂τ_i∂τ_j apply mismatch (i,j): rel_max_abs_diff={err_ij:.3e} > {tol:.1e}\n\
             analytic=\n{analytic_ij:?}\n\
             fd=\n{fd_ij:?}"
        );
        assert!(
            err_ji < tol,
            "∂²H_φ/∂τ_j∂τ_i apply mismatch (j,i): rel_max_abs_diff={err_ji:.3e} > {tol:.1e}\n\
             analytic=\n{analytic_ji:?}\n\
             fd=\n{fd_ji:?}"
        );
    }

    /// Verify the Primitive B body (`d_beta_hphi_tau_partial_apply`) against a
    /// finite-difference reference of the single-τ Primitive
    /// (`hphi_tau_partial_apply`).
    ///
    /// Identity under test (β held in the unreduced ambient; the design X is
    /// fixed so only w, η̇_τ = X_τ β, and their β-derivatives move):
    ///     D_β [ (H_φ)_τ|_β (β) · V ] [v]
    ///       = d_beta_hphi_tau_partial_apply(v, V).
    ///
    /// Central-difference reference:
    ///   1. Evaluate T(t) := hphi_tau_partial_apply(V) at β_t = β + t v,
    ///      rebuilding FirthDenseOperator at each β (so η = X β_t and the
    ///      w-chain are re-derived cleanly).  X is unchanged; Q is rebuilt
    ///      but H_φ applied to a p-space rhs is basis-invariant.
    ///   2. FD = (T(+h) − T(−h)) / (2h).
    ///   3. Tolerance 1e-7 relative max-abs (h chosen to balance truncation
    ///      error at ~h² and evaluator roundoff at ~ε/h).
    #[test]
    fn firth_d_beta_hphi_tau_partial_matches_finite_difference() {
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
        // β-direction v for the D_β[·][v] test.
        let v = array![0.3, 0.2, -0.15];

        let eta = x.dot(&beta);
        let op = FirthDenseOperator::build(&x, &eta).expect("firth operator");
        let p = x.ncols();

        // Reproducible small rhs block (p × m).
        let m = 3usize;
        let mut rhs = Array2::<f64>::zeros((p, m));
        let vals = [
            0.21, -0.44, 0.17, 0.38, 0.05, -0.22, -0.11, 0.27, 0.31,
        ];
        for r in 0..p {
            for c in 0..m {
                rhs[[r, c]] = vals[(r * m + c) % vals.len()];
            }
        }

        // ── Analytic apply at (x, β).
        let x_tau_reduced = op.reduce_explicit_design(&x_tau);
        let deta_partial = x_tau.dot(&beta);
        let (dot_i_partial, dot_h_partial) =
            op.dot_i_and_h_from_reduced(&x_tau_reduced, &deta_partial);
        let tau_kernel = op.hphi_tau_partial_prepare_from_partials(
            x_tau_reduced.clone(),
            &deta_partial,
            dot_h_partial.clone(),
            dot_i_partial.clone(),
        );

        let deta_v = x.dot(&v);
        let direction = op.direction_from_deta(deta_v);
        let x_tau_v = x_tau.dot(&v);
        let pair_kernel = op.d_beta_hphi_tau_partial_prepare_from_partials(
            &tau_kernel,
            &deta_partial,
            &dot_i_partial,
            &direction,
            &x_tau_v,
        );
        let analytic = op.d_beta_hphi_tau_partial_apply(&x_tau, &pair_kernel, &rhs);

        // ── FD reference: central difference of single-τ primitive under
        //    β → β ± h v.  X stays fixed; η, w, η̇_τ are re-derived.
        let h = 1e-5_f64;
        let single_tau_apply = |beta_eval: &Array1<f64>| -> Array2<f64> {
            let eta_e = x.dot(beta_eval);
            let op_e = FirthDenseOperator::build(&x, &eta_e).expect("perturbed firth operator");
            let x_tau_r = op_e.reduce_explicit_design(&x_tau);
            let deta_e = x_tau.dot(beta_eval);
            let (dot_i_e, dot_h_e) = op_e.dot_i_and_h_from_reduced(&x_tau_r, &deta_e);
            let ker_e = op_e.hphi_tau_partial_prepare_from_partials(
                x_tau_r, &deta_e, dot_h_e, dot_i_e,
            );
            op_e.hphi_tau_partial_apply(&x_tau, &ker_e, &rhs)
        };
        let beta_plus = &beta + &(h * &v);
        let beta_minus = &beta - &(h * &v);
        let fd = (&single_tau_apply(&beta_plus) - &single_tau_apply(&beta_minus)) / (2.0 * h);

        let rel_max_abs_diff = |a: &Array2<f64>, b: &Array2<f64>| -> f64 {
            let scale = a
                .iter()
                .chain(b.iter())
                .map(|v| v.abs())
                .fold(0.0_f64, f64::max)
                .max(1.0);
            let max_diff = (a - b).iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            max_diff / scale
        };
        let err = rel_max_abs_diff(&analytic, &fd);

        let tol = 1e-7_f64;
        assert!(
            err < tol,
            "D_β (H_φ)_τ|_β apply mismatch: rel_max_abs_diff={err:.3e} > {tol:.1e}\n\
             analytic=\n{analytic:?}\n\
             fd=\n{fd:?}"
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
