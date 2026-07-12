use super::*;

// ---------------------------------------------------------------------------
// Row-precision prior penalty
// ---------------------------------------------------------------------------

/// Fixed zero-mean Gaussian row-precision prior on the latent block.
///
/// Evaluates the row-wise precision energy `½ μ Σ_n t_nᵀ Λ_n t_n`, with the
/// `ρ`-dependent Gaussian precision normalizer when `μ` is learnable. Callers
/// pass one positive-definite precision matrix per row. This is not the iVAE
/// conditional-mean gauge `½ μ ||t - h(u)||²`; use `LatentIdMode::AuxPrior`
/// for the ridge/linear projection-residual gauge.
#[derive(Debug, Clone)]
pub struct RowPrecisionPriorPenalty {
    pub lambda_per_row: Array3<f64>,
    /// Base strength. If `learnable_weight` is true, the resolved strength is
    /// `weight * exp(rho[rho_index])`; otherwise it is fixed at `weight`.
    pub weight: f64,
    /// Number of rows in the row-major matrix-valued latent block.
    pub n_eff: usize,
    pub learnable_weight: bool,
    pub rho_index: usize,
    pub target: PsiSlice,
    pub weight_schedule: Option<ScalarWeightSchedule>,
}

impl RowPrecisionPriorPenalty {
    #[must_use = "build error must be handled"]
    pub fn new(
        target: PsiSlice,
        lambda_per_row: Array3<f64>,
        weight: f64,
        n_eff: usize,
        learnable_weight: bool,
    ) -> Result<Self, String> {
        if target.is_empty() {
            return Err("RowPrecisionPriorPenalty::new requires a non-empty target".to_string());
        }
        if !(weight.is_finite() && weight > 0.0) {
            return Err(format!(
                "RowPrecisionPriorPenalty::new requires finite weight > 0, got {weight}"
            ));
        }
        if n_eff == 0 {
            return Err("RowPrecisionPriorPenalty::new requires n_eff > 0".to_string());
        }
        if !target.len().is_multiple_of(n_eff) {
            return Err(format!(
                "RowPrecisionPriorPenalty::new target length {} is not divisible by n_eff {}",
                target.len(),
                n_eff
            ));
        }
        let latent_dim = target.len() / n_eff;
        if let Some(expected_dim) = target.latent_dim {
            let expected = n_eff.checked_mul(expected_dim).ok_or_else(|| {
                "RowPrecisionPriorPenalty::new target shape overflows usize".to_string()
            })?;
            if expected != target.len() {
                return Err(format!(
                    "RowPrecisionPriorPenalty::new target length {} does not match n_eff {} × latent_dim {}",
                    target.len(),
                    n_eff,
                    expected_dim
                ));
            }
            if expected_dim != latent_dim {
                return Err(format!(
                    "RowPrecisionPriorPenalty::new inferred latent_dim {latent_dim} does not match target latent_dim {expected_dim}"
                ));
            }
        }
        let (lambda_n, lambda_rows, lambda_cols) = lambda_per_row.dim();
        if lambda_n != n_eff || lambda_rows != latent_dim || lambda_cols != latent_dim {
            return Err(format!(
                "RowPrecisionPriorPenalty::new lambda_per_row shape must be ({n_eff}, {latent_dim}, {latent_dim}), got ({lambda_n}, {lambda_rows}, {lambda_cols})"
            ));
        }
        for n in 0..n_eff {
            let mut matrix = Array2::<f64>::zeros((latent_dim, latent_dim));
            for i in 0..latent_dim {
                for j in 0..latent_dim {
                    let value = lambda_per_row[[n, i, j]];
                    if !value.is_finite() {
                        return Err(format!(
                            "RowPrecisionPriorPenalty::new lambda_per_row[{n},{i},{j}] must be finite"
                        ));
                    }
                    let transpose = lambda_per_row[[n, j, i]];
                    if (value - transpose).abs() >= 1.0e-10 {
                        return Err(format!(
                            "RowPrecisionPriorPenalty::new lambda_per_row[{n}] must be symmetric; |Λ[{i},{j}] - Λ[{j},{i}]| = {:.3e}",
                            (value - transpose).abs()
                        ));
                    }
                    matrix[[i, j]] = value;
                }
            }
            let (evals, _) = matrix.eigh(Side::Lower).map_err(|err| {
                format!("RowPrecisionPriorPenalty::new lambda_per_row[{n}] eigendecomposition failed: {err}")
            })?;
            let min_eval = evals.iter().fold(f64::INFINITY, |acc, &v| acc.min(v));
            if !(min_eval.is_finite() && min_eval > 0.0) {
                return Err(format!(
                    "RowPrecisionPriorPenalty::new lambda_per_row[{n}] must be positive definite; minimum eigenvalue {min_eval:.3e}"
                ));
            }
        }
        Ok(Self {
            lambda_per_row,
            weight,
            n_eff,
            learnable_weight,
            rho_index: 0,
            target,
            weight_schedule: None,
        })
    }

    impl_with_weight_schedule!(weight);

    fn resolved_weight(&self, rho: ArrayView1<'_, f64>) -> f64 {
        if self.learnable_weight {
            validated_learnable_weight(self.weight, rho[self.rho_index])
        } else {
            self.weight
        }
    }

    fn latent_dim(&self, target_len: usize) -> Option<usize> {
        if self.n_eff == 0 || !target_len.is_multiple_of(self.n_eff) {
            assert_eq!(
                target_len % self.n_eff.max(1),
                0,
                "target length must be divisible by n_eff"
            );
            return None;
        }
        Some(target_len / self.n_eff)
    }

    fn target_matrix<'a>(&self, target: ArrayView1<'a, f64>) -> Option<ArrayView2<'a, f64>> {
        let d = self.latent_dim(target.len())?;
        target.into_shape_with_order((self.n_eff, d)).ok()
    }

    fn flatten_matrix(m: &Array2<f64>) -> Array1<f64> {
        let n_obs = m.nrows();
        let d = m.ncols();
        let mut out = Array1::<f64>::zeros(n_obs * d);
        for n in 0..n_obs {
            for a in 0..d {
                out[n * d + a] = m[[n, a]];
            }
        }
        out
    }

    pub fn diag_target(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        let Some(t) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(target.len());
        };
        let weight = self.resolved_weight(rho);
        let mut out = Array1::<f64>::zeros(target.len());
        for n in 0..t.nrows() {
            for i in 0..t.ncols() {
                out[n * t.ncols() + i] = weight * self.lambda_per_row[[n, i, i]];
            }
        }
        out
    }

    /// Materialize the row-block-diagonal Hessian for exact spectral paths.
    pub fn as_dense(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array2<f64> {
        let n_total = target.len();
        let Some(t) = self.target_matrix(target) else {
            return Array2::<f64>::zeros((n_total, n_total));
        };
        let d = t.ncols();
        let weight = self.resolved_weight(rho);
        let mut dense = Array2::<f64>::zeros((n_total, n_total));
        for n in 0..t.nrows() {
            for i in 0..d {
                let row = n * d + i;
                for j in 0..d {
                    dense[[row, n * d + j]] = weight * self.lambda_per_row[[n, i, j]];
                }
            }
        }
        dense
    }

    pub fn log_det_plus_lambda_i(
        &self,
        rho: ArrayView1<'_, f64>,
        lambda: f64,
    ) -> Result<f64, String> {
        if !(lambda.is_finite() && lambda > 0.0) {
            return Err(format!(
                "RowPrecisionPriorPenalty::log_det_plus_lambda_i requires finite λ > 0; got {lambda}"
            ));
        }
        let (n_obs, d, _) = self.lambda_per_row.dim();
        let weight = self.resolved_weight(rho);
        let mut sum = 0.0;
        for n in 0..n_obs {
            let mut matrix = Array2::<f64>::zeros((d, d));
            for i in 0..d {
                for j in 0..d {
                    matrix[[i, j]] = self.lambda_per_row[[n, i, j]];
                }
            }
            let (evals, _) = matrix.eigh(Side::Lower).map_err(|err| {
                format!("RowPrecisionPriorPenalty::log_det_plus_lambda_i lambda_per_row[{n}] eigendecomposition failed: {err}")
            })?;
            for &eval in evals.iter() {
                let shifted = weight * eval + lambda;
                if !(shifted.is_finite() && shifted > 0.0) {
                    return Err(format!(
                        "RowPrecisionPriorPenalty::log_det_plus_lambda_i non-positive shifted eigenvalue {shifted:.3e}"
                    ));
                }
                sum += shifted.ln();
            }
        }
        Ok(sum)
    }
}

impl AnalyticPenalty for RowPrecisionPriorPenalty {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Psi
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let Some(t) = self.target_matrix(target) else {
            return 0.0;
        };
        let mut acc = 0.0;
        for n in 0..t.nrows() {
            for i in 0..t.ncols() {
                let mut row_dot = 0.0;
                for j in 0..t.ncols() {
                    row_dot += self.lambda_per_row[[n, i, j]] * t[[n, j]];
                }
                acc += t[[n, i]] * row_dot;
            }
        }
        let weight = self.resolved_weight(rho);
        let log_weight_normalizer = -0.5 * target.len() as f64 * weight.ln();
        0.5 * weight * acc + log_weight_normalizer
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let Some(t) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(target.len());
        };
        let weight = self.resolved_weight(rho);
        let mut grad = Array2::<f64>::zeros(t.dim());
        for n in 0..t.nrows() {
            for i in 0..t.ncols() {
                let mut acc = 0.0;
                for j in 0..t.ncols() {
                    acc += self.lambda_per_row[[n, i, j]] * t[[n, j]];
                }
                grad[[n, i]] = weight * acc;
            }
        }
        Self::flatten_matrix(&grad)
    }

    fn hessian_diag(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Option<Array1<f64>> {
        let Some(t) = self.target_matrix(target) else {
            return Some(Array1::<f64>::zeros(target.len()));
        };
        for n in 0..t.nrows() {
            for i in 0..t.ncols() {
                for j in 0..t.ncols() {
                    if i != j && self.lambda_per_row[[n, i, j]] != 0.0 {
                        return None;
                    }
                }
            }
        }
        Some(self.diag_target(target, rho))
    }

    fn hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        assert_eq!(target.len(), v.len(), "hvp dimension mismatch");
        if target.len() != v.len() {
            return Array1::<f64>::zeros(target.len());
        }
        let Some(t) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(target.len());
        };
        let Some(v_mat) = self.target_matrix(v) else {
            return Array1::<f64>::zeros(target.len());
        };
        let weight = self.resolved_weight(rho);
        let mut out = Array2::<f64>::zeros(t.dim());
        for n in 0..v_mat.nrows() {
            for i in 0..v_mat.ncols() {
                let mut acc = 0.0;
                for j in 0..v_mat.ncols() {
                    acc += self.lambda_per_row[[n, i, j]] * v_mat[[n, j]];
                }
                out[[n, i]] = weight * acc;
            }
        }
        Self::flatten_matrix(&out)
    }

    fn grad_rho(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        if !self.learnable_weight {
            return Array1::<f64>::zeros(0);
        }
        let Some(t) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(1);
        };
        let mut quad = 0.0;
        for n in 0..t.nrows() {
            for i in 0..t.ncols() {
                let mut row_dot = 0.0;
                for j in 0..t.ncols() {
                    row_dot += self.lambda_per_row[[n, i, j]] * t[[n, j]];
                }
                quad += t[[n, i]] * row_dot;
            }
        }
        let weight = self.resolved_weight(rho);
        let mut out = Array1::<f64>::zeros(1);
        out[self.rho_index] = 0.5 * weight * quad - 0.5 * target.len() as f64;
        out
    }

    impl_learnable_weight_rho_count!();
    impl_learnable_weight_domain!(weight);

    fn name(&self) -> &str {
        "row_precision_prior"
    }

    impl_scalar_apply_schedule!(weight);
}

// ---------------------------------------------------------------------------
// iVAE ridge conditional-mean gauge penalty
// ---------------------------------------------------------------------------

/// iVAE conditional-mean gauge penalty on the latent block.
///
/// Khemakhem et al. (2020) identify nonlinear ICA/iVAE latent factors from
/// auxiliary-variable variation up to an affine transform under sufficient
/// variation in `u`. This penalty implements the conditional-mean side of that
/// signal as `0.5 * μ * ||t - U(UᵀU + εI)⁻¹Uᵀt||²`, penalizing only the
/// component of each latent axis not explained by a ridge linear fit to `u`.
#[derive(Debug, Clone)]
pub struct IvaeRidgeMeanGauge {
    pub aux: Array2<f64>,
    pub ridge_inv: Array2<f64>,
    pub ridge_eps: f64,
    /// Base strength. If `learnable_weight` is true, the resolved strength is
    /// `weight * exp(rho[rho_index])`; otherwise it is fixed at `weight`.
    pub weight: f64,
    /// Number of rows in the row-major matrix-valued latent block.
    pub n_eff: usize,
    pub learnable_weight: bool,
    pub rho_index: usize,
    pub target: PsiSlice,
    pub weight_schedule: Option<ScalarWeightSchedule>,
}

impl IvaeRidgeMeanGauge {
    #[must_use = "build error must be handled"]
    pub fn new(
        target: PsiSlice,
        aux: Array2<f64>,
        ridge_eps: f64,
        weight: f64,
        n_eff: usize,
        learnable_weight: bool,
    ) -> Result<Self, String> {
        if target.is_empty() {
            return Err("IvaeRidgeMeanGauge::new requires a non-empty target".to_string());
        }
        if !(weight.is_finite() && weight > 0.0) {
            return Err(format!(
                "IvaeRidgeMeanGauge::new requires finite weight > 0, got {weight}"
            ));
        }
        if !(ridge_eps.is_finite() && ridge_eps > 0.0) {
            return Err(format!(
                "IvaeRidgeMeanGauge::new requires finite ridge_eps > 0, got {ridge_eps}"
            ));
        }
        if n_eff == 0 {
            return Err("IvaeRidgeMeanGauge::new requires n_eff > 0".to_string());
        }
        if !target.len().is_multiple_of(n_eff) {
            return Err(format!(
                "IvaeRidgeMeanGauge::new target length {} is not divisible by n_eff {}",
                target.len(),
                n_eff
            ));
        }
        let latent_dim = target.len() / n_eff;
        if let Some(expected_dim) = target.latent_dim {
            let expected = n_eff.checked_mul(expected_dim).ok_or_else(|| {
                "IvaeRidgeMeanGauge::new target shape overflows usize".to_string()
            })?;
            if expected != target.len() {
                return Err(format!(
                    "IvaeRidgeMeanGauge::new target length {} does not match n_eff {} × latent_dim {}",
                    target.len(),
                    n_eff,
                    expected_dim
                ));
            }
            if expected_dim != latent_dim {
                return Err(format!(
                    "IvaeRidgeMeanGauge::new inferred latent_dim {latent_dim} does not match target latent_dim {expected_dim}"
                ));
            }
        }
        let (aux_n, aux_dim) = aux.dim();
        if aux_n != n_eff {
            return Err(format!(
                "IvaeRidgeMeanGauge::new aux rows must equal n_eff {n_eff}, got {aux_n}"
            ));
        }
        if aux_dim == 0 {
            return Err("IvaeRidgeMeanGauge::new requires aux dimension > 0".to_string());
        }
        for (idx, &value) in aux.iter().enumerate() {
            if !value.is_finite() {
                return Err(format!("IvaeRidgeMeanGauge::new aux[{idx}] must be finite"));
            }
        }
        let mut gram = Array2::<f64>::zeros((aux_dim, aux_dim));
        for n in 0..n_eff {
            for i in 0..aux_dim {
                for j in 0..aux_dim {
                    gram[[i, j]] += aux[[n, i]] * aux[[n, j]];
                }
            }
        }
        for i in 0..aux_dim {
            gram[[i, i]] += ridge_eps;
        }
        let ridge_inv = Self::invert_spd_gram(gram)?;
        Ok(Self {
            aux,
            ridge_inv,
            ridge_eps,
            weight,
            n_eff,
            learnable_weight,
            rho_index: 0,
            target,
            weight_schedule: None,
        })
    }

    impl_with_weight_schedule!(weight);

    fn invert_spd_gram(gram: Array2<f64>) -> Result<Array2<f64>, String> {
        let q = gram.nrows();
        let (evals, evecs) = gram.eigh(Side::Lower).map_err(|err| {
            format!("IvaeRidgeMeanGauge::new ridge Gram eigendecomposition failed: {err}")
        })?;
        let mut inv = Array2::<f64>::zeros((q, q));
        for k in 0..q {
            let eval = evals[k];
            if !(eval.is_finite() && eval > 0.0) {
                return Err(format!(
                    "IvaeRidgeMeanGauge::new ridge Gram must be positive definite; eigenvalue {k} is {eval:.3e}"
                ));
            }
            let inv_eval = 1.0 / eval;
            for i in 0..q {
                for j in 0..q {
                    inv[[i, j]] += evecs[[i, k]] * evecs[[j, k]] * inv_eval;
                }
            }
        }
        Ok(inv)
    }

    fn resolved_weight(&self, rho: ArrayView1<'_, f64>) -> f64 {
        if self.learnable_weight {
            validated_learnable_weight(self.weight, rho[self.rho_index])
        } else {
            self.weight
        }
    }

    fn latent_dim(&self, target_len: usize) -> Option<usize> {
        if self.n_eff == 0 || !target_len.is_multiple_of(self.n_eff) {
            assert_eq!(
                target_len % self.n_eff.max(1),
                0,
                "target length must be divisible by n_eff"
            );
            return None;
        }
        Some(target_len / self.n_eff)
    }

    fn target_matrix<'a>(&self, target: ArrayView1<'a, f64>) -> Option<ArrayView2<'a, f64>> {
        let d = self.latent_dim(target.len())?;
        target.into_shape_with_order((self.n_eff, d)).ok()
    }

    fn flatten_matrix(m: &Array2<f64>) -> Array1<f64> {
        let n_obs = m.nrows();
        let d = m.ncols();
        let mut out = Array1::<f64>::zeros(n_obs * d);
        for n in 0..n_obs {
            for a in 0..d {
                out[n * d + a] = m[[n, a]];
            }
        }
        out
    }

    fn projected_matrix(&self, x: ArrayView2<'_, f64>) -> Array2<f64> {
        let q = self.aux.ncols();
        let d = x.ncols();
        let mut u_t_x = Array2::<f64>::zeros((q, d));
        for n in 0..x.nrows() {
            for i in 0..q {
                let u_ni = self.aux[[n, i]];
                for a in 0..d {
                    u_t_x[[i, a]] += u_ni * x[[n, a]];
                }
            }
        }
        let mut coeff = Array2::<f64>::zeros((q, d));
        for i in 0..q {
            for j in 0..q {
                let inv_ij = self.ridge_inv[[i, j]];
                for a in 0..d {
                    coeff[[i, a]] += inv_ij * u_t_x[[j, a]];
                }
            }
        }
        let mut projected = Array2::<f64>::zeros(x.dim());
        for n in 0..x.nrows() {
            for i in 0..q {
                let u_ni = self.aux[[n, i]];
                for a in 0..d {
                    projected[[n, a]] += u_ni * coeff[[i, a]];
                }
            }
        }
        projected
    }

    fn residual_matrix(&self, x: ArrayView2<'_, f64>) -> Array2<f64> {
        let projected = self.projected_matrix(x);
        let mut residual = Array2::<f64>::zeros(x.dim());
        for n in 0..x.nrows() {
            for a in 0..x.ncols() {
                residual[[n, a]] = x[[n, a]] - projected[[n, a]];
            }
        }
        residual
    }

    pub fn diag_target(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        let Some(t) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(target.len());
        };
        let weight = self.resolved_weight(rho);
        let mut out = Array1::<f64>::zeros(target.len());
        for n in 0..t.nrows() {
            let mut p_nn = 0.0;
            for i in 0..self.aux.ncols() {
                for j in 0..self.aux.ncols() {
                    p_nn += self.aux[[n, i]] * self.ridge_inv[[i, j]] * self.aux[[n, j]];
                }
            }
            let diag = weight * (1.0 - p_nn);
            for a in 0..t.ncols() {
                out[n * t.ncols() + a] = diag;
            }
        }
        out
    }

    /// Materialize `μ(I - U(UᵀU + εI)⁻¹Uᵀ)` repeated per latent axis.
    pub fn as_dense(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array2<f64> {
        let n_total = target.len();
        let Some(t) = self.target_matrix(target) else {
            return Array2::<f64>::zeros((n_total, n_total));
        };
        let d = t.ncols();
        let weight = self.resolved_weight(rho);
        let mut dense = Array2::<f64>::zeros((n_total, n_total));
        for n in 0..t.nrows() {
            for m in 0..t.nrows() {
                let mut p_nm = 0.0;
                for i in 0..self.aux.ncols() {
                    for j in 0..self.aux.ncols() {
                        p_nm += self.aux[[n, i]] * self.ridge_inv[[i, j]] * self.aux[[m, j]];
                    }
                }
                let entry = weight * (if n == m { 1.0 } else { 0.0 } - p_nm);
                for a in 0..d {
                    dense[[n * d + a, m * d + a]] = entry;
                }
            }
        }
        dense
    }
}

impl AnalyticPenalty for IvaeRidgeMeanGauge {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Psi
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let Some(t) = self.target_matrix(target) else {
            return 0.0;
        };
        let residual = self.residual_matrix(t.view());
        let mut acc = 0.0;
        for n in 0..t.nrows() {
            for a in 0..t.ncols() {
                acc += t[[n, a]] * residual[[n, a]];
            }
        }
        let weight = self.resolved_weight(rho);
        let mut value = 0.5 * weight * acc;
        if self.learnable_weight {
            value -= 0.5 * target.len() as f64 * weight.ln();
        }
        value
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let Some(t) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(target.len());
        };
        let weight = self.resolved_weight(rho);
        let mut grad = self.residual_matrix(t.view());
        for value in grad.iter_mut() {
            *value *= weight;
        }
        Self::flatten_matrix(&grad)
    }

    fn hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        assert_eq!(target.len(), v.len(), "hvp dimension mismatch");
        if target.len() != v.len() {
            return Array1::<f64>::zeros(target.len());
        }
        let Some(v_mat) = self.target_matrix(v) else {
            return Array1::<f64>::zeros(target.len());
        };
        let weight = self.resolved_weight(rho);
        let mut hv = self.residual_matrix(v_mat.view());
        for value in hv.iter_mut() {
            *value *= weight;
        }
        Self::flatten_matrix(&hv)
    }

    fn grad_rho(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        if !self.learnable_weight {
            return Array1::<f64>::zeros(0);
        }
        if self.target_matrix(target).is_none() {
            return Array1::<f64>::zeros(1);
        }
        let mut out = Array1::<f64>::zeros(1);
        let weight = self.resolved_weight(rho);
        out[self.rho_index] =
            self.value(target, rho) + 0.5 * target.len() as f64 * (weight.ln() - 1.0);
        out
    }

    impl_learnable_weight_rho_count!();
    impl_learnable_weight_domain!(weight);

    fn name(&self) -> &str {
        "ivae_ridge_mean_gauge"
    }

    impl_scalar_apply_schedule!(weight);
}

// ---------------------------------------------------------------------------
// Parametric row-precision prior penalty
// ---------------------------------------------------------------------------

/// Parametric zero-mean Gaussian row-precision prior on the latent block.
///
/// Uses a diagonal precision
/// `λ_k(u_n) = exp(log_alpha_k) + softplus(raw_beta_k) ||u_n - μ_k||²`.
/// REML may learn that conditional precision map, including the Gaussian
/// precision normalizer derivatives. This is not a learnable conditional
/// mean map and does not implement the iVAE projection-residual gauge.
#[derive(Debug, Clone)]
pub struct ParametricRowPrecisionPriorPenalty {
    pub aux: Array2<f64>,
    pub log_alpha: Array1<f64>,
    pub raw_beta: Array1<f64>,
    pub mu: Array2<f64>,
    /// Base strength. If `learnable_weight` is true, the resolved strength is
    /// `weight * exp(rho[weight_rho_index])`; otherwise it is fixed at `weight`.
    pub weight: f64,
    /// Number of rows in the row-major matrix-valued latent block.
    pub n_eff: usize,
    pub learnable_weight: bool,
    pub target: PsiSlice,
    pub weight_schedule: Option<ScalarWeightSchedule>,
}

impl ParametricRowPrecisionPriorPenalty {
    #[must_use = "build error must be handled"]
    pub fn new(
        target: PsiSlice,
        aux: Array2<f64>,
        log_alpha: Array1<f64>,
        raw_beta: Array1<f64>,
        mu: Array2<f64>,
        weight: f64,
        n_eff: usize,
        learnable_weight: bool,
    ) -> Result<Self, String> {
        if target.is_empty() {
            return Err(
                "ParametricRowPrecisionPriorPenalty::new requires a non-empty target".to_string(),
            );
        }
        if !(weight.is_finite() && weight > 0.0) {
            return Err(format!(
                "ParametricRowPrecisionPriorPenalty::new requires finite weight > 0, got {weight}"
            ));
        }
        if n_eff == 0 {
            return Err("ParametricRowPrecisionPriorPenalty::new requires n_eff > 0".to_string());
        }
        if !target.len().is_multiple_of(n_eff) {
            return Err(format!(
                "ParametricRowPrecisionPriorPenalty::new target length {} is not divisible by n_eff {}",
                target.len(),
                n_eff
            ));
        }
        let latent_dim = target.len() / n_eff;
        if latent_dim == 0 {
            return Err(
                "ParametricRowPrecisionPriorPenalty::new requires latent_dim > 0".to_string(),
            );
        }
        if let Some(expected_dim) = target.latent_dim {
            let expected = n_eff.checked_mul(expected_dim).ok_or_else(|| {
                "ParametricRowPrecisionPriorPenalty::new target shape overflows usize".to_string()
            })?;
            if expected != target.len() {
                return Err(format!(
                    "ParametricRowPrecisionPriorPenalty::new target length {} does not match n_eff {} × latent_dim {}",
                    target.len(),
                    n_eff,
                    expected_dim
                ));
            }
            if expected_dim != latent_dim {
                return Err(format!(
                    "ParametricRowPrecisionPriorPenalty::new inferred latent_dim {latent_dim} does not match target latent_dim {expected_dim}"
                ));
            }
        }
        let (aux_n, aux_dim) = aux.dim();
        if aux_n != n_eff {
            return Err(format!(
                "ParametricRowPrecisionPriorPenalty::new aux rows must equal n_eff {n_eff}, got {aux_n}"
            ));
        }
        if aux_dim == 0 {
            return Err(
                "ParametricRowPrecisionPriorPenalty::new requires aux dimension > 0".to_string(),
            );
        }
        if log_alpha.len() != latent_dim {
            return Err(format!(
                "ParametricRowPrecisionPriorPenalty::new log_alpha length must equal latent_dim {latent_dim}, got {}",
                log_alpha.len()
            ));
        }
        if raw_beta.len() != latent_dim {
            return Err(format!(
                "ParametricRowPrecisionPriorPenalty::new raw_beta length must equal latent_dim {latent_dim}, got {}",
                raw_beta.len()
            ));
        }
        let (mu_rows, mu_cols) = mu.dim();
        if mu_rows != latent_dim || mu_cols != aux_dim {
            return Err(format!(
                "ParametricRowPrecisionPriorPenalty::new mu shape must be ({latent_dim}, {aux_dim}), got ({mu_rows}, {mu_cols})"
            ));
        }
        for (idx, &value) in aux.iter().enumerate() {
            if !value.is_finite() {
                return Err(format!(
                    "ParametricRowPrecisionPriorPenalty::new aux[{idx}] must be finite"
                ));
            }
        }
        for k in 0..latent_dim {
            let log_alpha_k = log_alpha[k];
            if !log_alpha_k.is_finite() {
                return Err(format!(
                    "ParametricRowPrecisionPriorPenalty::new log_alpha[{k}] must be finite"
                ));
            }
            let alpha_k = log_alpha_k.exp();
            if !(alpha_k.is_finite() && alpha_k > 0.0) {
                return Err(format!(
                    "ParametricRowPrecisionPriorPenalty::new exp(log_alpha[{k}]) must be finite and > 0"
                ));
            }
            let raw_beta_k = raw_beta[k];
            if !raw_beta_k.is_finite() {
                return Err(format!(
                    "ParametricRowPrecisionPriorPenalty::new raw_beta[{k}] must be finite"
                ));
            }
            let beta_k = gam_linalg::utils::stable_softplus(raw_beta_k);
            if !(beta_k.is_finite() && beta_k >= 0.0) {
                return Err(format!(
                    "ParametricRowPrecisionPriorPenalty::new softplus(raw_beta[{k}]) must be finite and >= 0"
                ));
            }
        }
        for (idx, &value) in mu.iter().enumerate() {
            if !value.is_finite() {
                return Err(format!(
                    "ParametricRowPrecisionPriorPenalty::new mu[{idx}] must be finite"
                ));
            }
        }
        Ok(Self {
            aux,
            log_alpha,
            raw_beta,
            mu,
            weight,
            n_eff,
            learnable_weight,
            target,
            weight_schedule: None,
        })
    }

    impl_with_weight_schedule!(weight);

    fn latent_dim(&self, target_len: usize) -> Option<usize> {
        if self.n_eff == 0 || !target_len.is_multiple_of(self.n_eff) {
            assert_eq!(
                target_len % self.n_eff.max(1),
                0,
                "target length must be divisible by n_eff"
            );
            return None;
        }
        Some(target_len / self.n_eff)
    }

    fn target_matrix<'a>(&self, target: ArrayView1<'a, f64>) -> Option<ArrayView2<'a, f64>> {
        let d = self.latent_dim(target.len())?;
        target.into_shape_with_order((self.n_eff, d)).ok()
    }

    fn flatten_matrix(m: &Array2<f64>) -> Array1<f64> {
        let n_obs = m.nrows();
        let d = m.ncols();
        let mut out = Array1::<f64>::zeros(n_obs * d);
        for n in 0..n_obs {
            for a in 0..d {
                out[n * d + a] = m[[n, a]];
            }
        }
        out
    }

    fn log_alpha_offset(&self) -> usize {
        0
    }

    fn raw_beta_offset(&self) -> usize {
        self.log_alpha.len()
    }

    fn mu_offset(&self) -> usize {
        self.log_alpha.len() + self.raw_beta.len()
    }

    fn weight_offset(&self) -> usize {
        self.mu_offset() + self.mu.len()
    }

    fn active_log_alpha(&self, k: usize, rho: ArrayView1<'_, f64>) -> f64 {
        self.log_alpha[k] + rho[self.log_alpha_offset() + k]
    }

    fn active_raw_beta(&self, k: usize, rho: ArrayView1<'_, f64>) -> f64 {
        self.raw_beta[k] + rho[self.raw_beta_offset() + k]
    }

    fn active_mu(&self, k: usize, a: usize, rho: ArrayView1<'_, f64>) -> f64 {
        self.mu[[k, a]] + rho[self.mu_offset() + k * self.aux.ncols() + a]
    }

    fn resolved_weight(&self, rho: ArrayView1<'_, f64>) -> f64 {
        if self.learnable_weight {
            validated_learnable_weight(self.weight, rho[self.weight_offset()])
        } else {
            self.weight
        }
    }

    fn lambda_at(&self, n: usize, k: usize, rho: ArrayView1<'_, f64>) -> f64 {
        let alpha = validated_exp_log_strength(self.active_log_alpha(k, rho));
        let beta = gam_linalg::utils::stable_softplus(self.active_raw_beta(k, rho));
        MIN_CONDITIONAL_PRECISION + alpha + beta * self.dist2(n, k, rho)
    }

    fn dist2(&self, n: usize, k: usize, rho: ArrayView1<'_, f64>) -> f64 {
        let mut r2 = 0.0;
        for a in 0..self.aux.ncols() {
            let delta = self.aux[[n, a]] - self.active_mu(k, a, rho);
            r2 += delta * delta;
        }
        r2
    }

    pub fn diag_target(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        let Some(t) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(target.len());
        };
        let weight = self.resolved_weight(rho);
        let mut out = Array1::<f64>::zeros(target.len());
        for n in 0..t.nrows() {
            for k in 0..t.ncols() {
                out[n * t.ncols() + k] = weight * self.lambda_at(n, k, rho);
            }
        }
        out
    }

    /// Materialize the row-block-diagonal Hessian for exact spectral paths.
    pub fn as_dense(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array2<f64> {
        let n_total = target.len();
        let diag = self.diag_target(target, rho);
        let mut dense = Array2::<f64>::zeros((n_total, n_total));
        for i in 0..n_total {
            dense[[i, i]] = diag[i];
        }
        dense
    }

    pub fn log_det_plus_lambda_i(
        &self,
        rho: ArrayView1<'_, f64>,
        lambda: f64,
    ) -> Result<f64, String> {
        if !(lambda.is_finite() && lambda > 0.0) {
            return Err(format!(
                "ParametricRowPrecisionPriorPenalty::log_det_plus_lambda_i requires finite λ > 0; got {lambda}"
            ));
        }
        let weight = self.resolved_weight(rho);
        let mut sum = 0.0;
        for n in 0..self.n_eff {
            for k in 0..self.log_alpha.len() {
                let shifted = lambda + weight * self.lambda_at(n, k, rho);
                if !(shifted.is_finite() && shifted > 0.0) {
                    return Err(format!(
                        "ParametricRowPrecisionPriorPenalty::log_det_plus_lambda_i non-positive shifted diagonal {shifted:.3e}"
                    ));
                }
                sum += shifted.ln();
            }
        }
        Ok(sum)
    }
}

impl AnalyticPenalty for ParametricRowPrecisionPriorPenalty {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Psi
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let Some(t) = self.target_matrix(target) else {
            return 0.0;
        };
        let weight = self.resolved_weight(rho);
        let mut quadratic = 0.0;
        let mut log_det = 0.0;
        for n in 0..t.nrows() {
            for k in 0..t.ncols() {
                let lambda = self.lambda_at(n, k, rho);
                quadratic += lambda * t[[n, k]] * t[[n, k]];
                log_det += (weight * lambda).ln();
            }
        }
        0.5 * weight * quadratic - 0.5 * log_det
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let Some(t) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(target.len());
        };
        let weight = self.resolved_weight(rho);
        let mut grad = Array2::<f64>::zeros(t.dim());
        for n in 0..t.nrows() {
            for k in 0..t.ncols() {
                grad[[n, k]] = weight * self.lambda_at(n, k, rho) * t[[n, k]];
            }
        }
        Self::flatten_matrix(&grad)
    }

    fn hessian_diag(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Option<Array1<f64>> {
        Some(self.diag_target(target, rho))
    }

    fn hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        assert_eq!(target.len(), v.len(), "hvp dimension mismatch");
        if target.len() != v.len() {
            return Array1::<f64>::zeros(target.len());
        }
        let diag = self.diag_target(target, rho);
        let mut out = Array1::<f64>::zeros(v.len());
        for i in 0..v.len() {
            out[i] = diag[i] * v[i];
        }
        out
    }

    fn grad_rho(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let Some(t) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(self.rho_count());
        };
        let weight = self.resolved_weight(rho);
        let mut out = Array1::<f64>::zeros(self.rho_count());
        let d = t.ncols();
        let du = self.aux.ncols();
        let mut grad_weight_direct = 0.0;
        for k in 0..d {
            let log_alpha = self.active_log_alpha(k, rho);
            let alpha = validated_exp_log_strength(log_alpha);
            let raw_beta = self.active_raw_beta(k, rho);
            let beta = gam_linalg::utils::stable_softplus(raw_beta);
            let beta_jac = gam_linalg::utils::stable_logistic(raw_beta);
            let mut grad_alpha_direct = 0.0;
            let mut grad_beta_direct = 0.0;
            let mut grad_mu_direct = vec![0.0_f64; du];
            for n in 0..t.nrows() {
                let tk = t[[n, k]];
                let sq = tk * tk;
                let r2 = self.dist2(n, k, rho);
                // Same floored λ as `lambda_at`/`value`, so this gradient is the
                // exact derivative of the evaluated energy (no value↔grad drift).
                let lambda = MIN_CONDITIONAL_PRECISION + alpha + beta * r2;
                let precision_score = 0.5 * weight * sq - 0.5 / lambda;
                grad_weight_direct += 0.5 * weight * lambda * sq;
                grad_alpha_direct += precision_score;
                grad_beta_direct += precision_score * r2;
                for a in 0..du {
                    let delta = self.aux[[n, a]] - self.active_mu(k, a, rho);
                    grad_mu_direct[a] += -2.0 * precision_score * beta * delta;
                }
            }
            out[self.log_alpha_offset() + k] = grad_alpha_direct * alpha;
            out[self.raw_beta_offset() + k] = grad_beta_direct * beta_jac;
            for a in 0..du {
                out[self.mu_offset() + k * du + a] = grad_mu_direct[a];
            }
        }
        if self.learnable_weight {
            out[self.weight_offset()] = grad_weight_direct - 0.5 * target.len() as f64;
        }
        out
    }

    fn rho_count(&self) -> usize {
        self.log_alpha.len()
            + self.raw_beta.len()
            + self.mu.len()
            + usize::from(self.learnable_weight)
    }

    fn name(&self) -> &str {
        "parametric_row_precision_prior"
    }

    impl_scalar_apply_schedule!(weight);
}
