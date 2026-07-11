use super::*;
use gam_problem::dispersion_cov::se_from_covariance;

pub(crate) const REML_SECOND_ORDER_RHO_CAP: usize = 4;
/// Continuation prewarm is a seed-polishing pass, not part of the REML
/// objective. It can be useful for tiny rho spaces where one or two warm
/// solves amortize, but it scales with the number of starts and runs full
/// inner solves before the real optimizer even begins. Moderate/high-rho
/// smooths (measure-jet spectral candidates are the motivating profile) start
/// directly from the seed lattice; the optimizer's own line search owns
/// globalization.
pub(crate) const REML_CONTINUATION_PREWARM_RHO_CAP: usize = 4;
/// Above this rho dimension, startup work must be linear in "one real solve",
/// not "rank a seed lattice with capped PIRLS solves". The heuristic seed is
/// deterministic and already centered on the current penalty scale; BFGS/ARC
/// globalizes from there. Low-dimensional classic smooths keep screening
/// because the extra probes are cheap and sometimes useful.
pub(crate) const REML_SEED_SCREENING_RHO_CAP: usize = 4;

const KAHAN_SWITCH_ELEMS: usize = 10_000;

pub(crate) fn faer_frob_inner(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> f64 {
    let (m, n) = (a.nrows(), a.ncols());
    let elem_count = m.saturating_mul(n);
    if elem_count < KAHAN_SWITCH_ELEMS {
        let mut sum = 0.0_f64;
        for j in 0..n {
            for i in 0..m {
                sum += a[(i, j)] * b[(i, j)];
            }
        }
        sum
    } else {
        let mut sum = KahanSum::default();
        for j in 0..n {
            for i in 0..m {
                sum.add(a[(i, j)] * b[(i, j)]);
            }
        }
        sum.sum()
    }
}

pub(crate) fn kahan_sum<I>(iter: I) -> f64
where
    I: IntoIterator<Item = f64>,
{
    let mut acc = KahanSum::default();
    for value in iter {
        acc.add(value);
    }
    acc.sum()
}

#[derive(Clone, Debug)]
pub(crate) struct ParametricColumnConditioning {
    pub(crate) intercept_idx: Option<usize>,
    pub(crate) columns: Vec<(usize, f64, f64)>,
}

impl ParametricColumnConditioning {
    /// Build conditioning from explicit unpenalized column indices.
    ///
    /// Reads only the specified columns from `x` (via `extract_column`) to
    /// compute per-column mean/variance — no full-design densification.
    pub(crate) fn from_column_indices(x: &DesignMatrix, unpenalized_cols: &[usize]) -> Self {
        const SCALE_EPS: f64 = 1e-12;
        let n = x.nrows();
        if n == 0 {
            return Self {
                intercept_idx: None,
                columns: Vec::new(),
            };
        }
        let mut intercept_idx = None;
        let mut columns = Vec::new();
        // Batched extract avoids per-column unit-vector dispatch when `x` is a
        // lazy operator (e.g. ReparamOperator): one GEMM versus
        // `unpenalized_cols.len()` separate matvecs.
        let block = x.extract_columns(unpenalized_cols);
        for (k, &j) in unpenalized_cols.iter().enumerate() {
            let col = block.column(k);
            let first = col[0];
            let is_constant = col.iter().all(|&v| (v - first).abs() <= 1e-12);
            if is_constant {
                if (first - 1.0).abs() <= 1e-12 && intercept_idx.is_none() {
                    intercept_idx = Some(j);
                }
                continue;
            }
            let mean = col.iter().copied().sum::<f64>() / n as f64;
            let var = col
                .iter()
                .map(|&v| {
                    let d = v - mean;
                    d * d
                })
                .sum::<f64>()
                / n as f64;
            if !var.is_finite() || var <= SCALE_EPS * SCALE_EPS {
                continue;
            }
            columns.push((j, mean, var.sqrt()));
        }
        if intercept_idx.is_none() {
            for (_, mean, _) in &mut columns {
                *mean = 0.0;
            }
        }
        Self {
            intercept_idx,
            columns,
        }
    }

    /// Infer unpenalized columns from `PenaltySpec` slices.
    pub(crate) fn infer_from_penalty_specs(x: &DesignMatrix, specs: &[PenaltySpec]) -> Self {
        let p = x.ncols();
        let mut penalized = vec![false; p];
        for spec in specs {
            let range = spec.col_range(p);
            for j in range {
                penalized[j] = true;
            }
        }
        let unpenalized: Vec<usize> = (0..p).filter(|&j| !penalized[j]).collect();
        Self::from_column_indices(x, &unpenalized)
    }

    pub(crate) fn is_active(&self) -> bool {
        !self.columns.is_empty()
    }

    /// Return a lazily-conditioned design matrix (no materialization).
    ///
    /// Wraps `x` in a `ConditionedDesign` operator that applies per-column
    /// centering and scaling through matvec algebra, avoiding densification.
    pub(crate) fn apply_to_design(&self, x: &DesignMatrix) -> DesignMatrix {
        if !self.is_active() {
            return x.clone();
        }
        DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(Arc::new(
            gam_linalg::matrix::ConditionedDesign::new(x.clone(), self.columns.clone()),
        )))
    }

    /// Map a constraint matrix from original (user-scale) coefficients to the
    /// internally-conditioned coordinates the solver actually optimizes.
    ///
    /// Constraints are authored on the *original* design-column coefficients:
    /// `A_orig · β_orig {≥,≤} b` (e.g. a `linear(x, min, max)` box pushes rows
    /// `β_col ≥ min` and `β_col ≤ max`). The inner solve works with the
    /// conditioned coefficients `β_int`, where the back-transform `β_orig = M·β_int`
    /// is exactly the one implemented by [`Self::backtransform_beta`]:
    ///
    /// ```text
    ///   β_orig[j]         = β_int[j] / scale_j                         (conditioned col j)
    ///   β_orig[intercept] = β_int[intercept] − Σ_j (mean_j / scale_j) · β_int[j]
    /// ```
    ///
    /// so `M[j][j] = 1/scale_j`, `M[intercept][j] = −mean_j/scale_j`, and `M` is
    /// the identity elsewhere. Substituting into `A_orig · β_orig` gives the
    /// equivalent internal constraint `A_int · β_int {≥,≤} b` with `A_int = A_orig·M`.
    /// Only the conditioned columns of `A_int` differ from `A_orig`:
    ///
    /// ```text
    ///   A_int[:, j] = (A_orig[:, j] − mean_j · A_orig[:, intercept]) / scale_j
    /// ```
    ///
    /// The RHS `b` is unchanged, so [`Self::transform_linear_constraints_to_internal`]
    /// carries it through verbatim. `A_orig · M` is precisely `M` applied to the
    /// columns of `A_orig`, which is the canonical column-conditioning primitive
    /// [`Self::transform_matrix_columnswith_a`] — so delegate to it rather than
    /// carry a second copy of the per-column algebra.
    pub(crate) fn transform_constraint_matrix_to_internal(
        &self,
        a_original: &Array2<f64>,
    ) -> Array2<f64> {
        self.transform_matrix_columnswith_a(a_original)
    }

    pub(crate) fn transform_linear_constraints_to_internal(
        &self,
        constraints: Option<crate::pirls::LinearInequalityConstraints>,
    ) -> Option<crate::pirls::LinearInequalityConstraints> {
        constraints.map(|constraints| crate::pirls::LinearInequalityConstraints {
            a: self.transform_constraint_matrix_to_internal(&constraints.a),
            b: constraints.b,
        })
    }

    pub(crate) fn backtransform_beta(&self, beta_internal: &Array1<f64>) -> Array1<f64> {
        let mut beta = beta_internal.clone();
        for &(j, mean, scale) in &self.columns {
            if let Some(intercept_idx) = self.intercept_idx {
                beta[intercept_idx] -= beta_internal[j] * mean / scale;
            }
            beta[j] = beta_internal[j] / scale;
        }
        beta
    }

    pub(crate) fn transform_matrix_columnswith_a(&self, mat: &Array2<f64>) -> Array2<f64> {
        let mut out = mat.clone();
        self.transform_matrix_columnswith_a_inplace(&mut out);
        out
    }

    pub(crate) fn transform_matrix_columnswith_a_inplace(&self, mat: &mut Array2<f64>) {
        if !self.is_active() {
            return;
        }
        let intercept_col = self.intercept_idx.map(|idx| mat.column(idx).to_owned());
        for &(j, mean, scale) in &self.columns {
            let mut target = mat.column_mut(j);
            if mean != 0.0
                && let Some(intercept_col) = intercept_col.as_ref()
            {
                target -= &(intercept_col * mean);
            }
            if scale != 1.0 {
                target.mapv_inplace(|v| v / scale);
            }
        }
    }

    /// Left-multiply `mat_internal` by `M`, where `M` is the coefficient
    /// back-transform: `β_orig = M · β_int` (the same map
    /// [`Self::backtransform_beta`] applies to a single vector).
    ///
    /// `M` has the structure
    /// ```text
    ///   M[intercept, intercept] = 1
    ///   M[intercept, j]        = −mean_j / scale_j     (conditioned column j)
    ///   M[j, j]                = 1 / scale_j           (conditioned column j)
    /// ```
    /// and is the identity elsewhere. Acts on each column of `mat_internal`
    /// the same way `backtransform_beta` acts on a single vector.
    pub(crate) fn left_multiply_by_m(&self, mat_internal: &Array2<f64>) -> Array2<f64> {
        let mut out = mat_internal.clone();
        if !self.is_active() {
            return out;
        }
        if let Some(intercept_idx) = self.intercept_idx {
            // (M·X)[intercept, :] = X[intercept, :] − Σ_j (mean_j/scale_j) · X[j, :]
            // Each conditioned column reads from the ORIGINAL `mat_internal`
            // row j (snapshot), so the contributions accumulate independently
            // — identical semantics to `backtransform_beta`'s use of
            // `beta_internal[j]` rather than the running `beta[j]`.
            for &(j, mean, scale) in &self.columns {
                if mean != 0.0 {
                    let factor = mean / scale;
                    let row_j_snapshot = mat_internal.row(j).to_owned();
                    let mut interceptrow = out.row_mut(intercept_idx);
                    interceptrow -= &(&row_j_snapshot * factor);
                }
            }
        }
        // (M·X)[j, :] = X[j, :] / scale_j
        for &(j, _mean, scale) in &self.columns {
            if scale != 1.0 {
                out.row_mut(j).mapv_inplace(|v| v / scale);
            }
        }
        out
    }

    /// Right-multiply `mat_internal` by `Mᵀ` (the transpose of the
    /// coefficient back-transform). Mirror of [`Self::left_multiply_by_m`]
    /// on columns.
    pub(crate) fn right_multiply_by_m_transpose(&self, mat_internal: &Array2<f64>) -> Array2<f64> {
        let mut out = mat_internal.clone();
        if !self.is_active() {
            return out;
        }
        if let Some(intercept_idx) = self.intercept_idx {
            // (X·Mᵀ)[:, intercept] = X[:, intercept] − Σ_j (mean_j/scale_j) · X[:, j]
            for &(j, mean, scale) in &self.columns {
                if mean != 0.0 {
                    let factor = mean / scale;
                    let col_j_snapshot = mat_internal.column(j).to_owned();
                    let mut intercept_col = out.column_mut(intercept_idx);
                    intercept_col -= &(&col_j_snapshot * factor);
                }
            }
        }
        // (X·Mᵀ)[:, j] = X[:, j] / scale_j
        for &(j, _mean, scale) in &self.columns {
            if scale != 1.0 {
                out.column_mut(j).mapv_inplace(|v| v / scale);
            }
        }
        out
    }

    /// Left-multiply `mat_internal` by `M⁻ᵀ`. The inverse basis map is
    /// ```text
    ///   M⁻¹[intercept, intercept] = 1
    ///   M⁻¹[intercept, j]         = mean_j     (conditioned column j)
    ///   M⁻¹[j, j]                 = scale_j    (conditioned column j)
    /// ```
    /// so `(M⁻ᵀ · X)[j, :] = scale_j · X[j, :] + mean_j · X[intercept, :]`
    /// and `(M⁻ᵀ · X)[intercept, :] = X[intercept, :]`.
    pub(crate) fn left_multiply_by_m_inv_transpose(
        &self,
        mat_internal: &Array2<f64>,
    ) -> Array2<f64> {
        let mut out = mat_internal.clone();
        if !self.is_active() {
            return out;
        }
        if let Some(intercept_idx) = self.intercept_idx {
            let interceptrow_snapshot = mat_internal.row(intercept_idx).to_owned();
            for &(j, mean, scale) in &self.columns {
                if scale != 1.0 {
                    out.row_mut(j).mapv_inplace(|v| v * scale);
                }
                if mean != 0.0 {
                    let mut row_j = out.row_mut(j);
                    row_j += &(&interceptrow_snapshot * mean);
                }
            }
        } else {
            for &(j, _mean, scale) in &self.columns {
                if scale != 1.0 {
                    out.row_mut(j).mapv_inplace(|v| v * scale);
                }
            }
        }
        out
    }

    /// Right-multiply `mat_internal` by `M⁻¹`. Mirror of
    /// [`Self::left_multiply_by_m_inv_transpose`] on columns.
    pub(crate) fn right_multiply_by_m_inv(&self, mat_internal: &Array2<f64>) -> Array2<f64> {
        let mut out = mat_internal.clone();
        if !self.is_active() {
            return out;
        }
        if let Some(intercept_idx) = self.intercept_idx {
            let intercept_col_snapshot = mat_internal.column(intercept_idx).to_owned();
            for &(j, mean, scale) in &self.columns {
                if scale != 1.0 {
                    out.column_mut(j).mapv_inplace(|v| v * scale);
                }
                if mean != 0.0 {
                    let mut col_j = out.column_mut(j);
                    col_j += &(&intercept_col_snapshot * mean);
                }
            }
        } else {
            for &(j, _mean, scale) in &self.columns {
                if scale != 1.0 {
                    out.column_mut(j).mapv_inplace(|v| v * scale);
                }
            }
        }
        out
    }

    /// `Cov(β_orig) = M · Cov(β_int) · Mᵀ`.
    ///
    /// Since `β_orig = M · β_int`, the covariance back-transform is the
    /// congruence `M · Σ · Mᵀ`, NOT `Mᵀ · Σ · M`. The latter (the prior
    /// implementation) silently swapped the variance of every conditioned
    /// parametric column with the variance of the intercept, off by exactly
    /// the basis change the intercept absorbs when columns are centered.
    pub(crate) fn backtransform_covariance(&self, cov_internal: &Array2<f64>) -> Array2<f64> {
        let right = self.right_multiply_by_m_transpose(cov_internal);
        self.left_multiply_by_m(&right)
    }

    /// `H_orig = M⁻ᵀ · H_int · M⁻¹`.
    ///
    /// Derived from `L_int(β_int) = L_orig(M · β_int)`: the chain rule gives
    /// `H_int = Mᵀ · H_orig · M`, so `H_orig = M⁻ᵀ · H_int · M⁻¹`. The prior
    /// implementation multiplied the intercept entry of `M⁻¹` by `scale_j`,
    /// silently scaling the Hessian by `scale_j²` along every conditioned
    /// column whenever scaling (not just centering) was active.
    pub(crate) fn backtransform_penalized_hessian(&self, h_internal: &Array2<f64>) -> Array2<f64> {
        let right = self.right_multiply_by_m_inv(h_internal);
        self.left_multiply_by_m_inv_transpose(&right)
    }

    pub(crate) fn backtransform_external_result(
        &self,
        mut result: ExternalOptimResult,
    ) -> ExternalOptimResult {
        if !self.is_active() {
            return result;
        }
        result.beta = self.backtransform_beta(&result.beta);
        if let Some(inf) = result.inference.as_mut() {
            inf.penalized_hessian = self
                .backtransform_penalized_hessian(inf.penalized_hessian.as_array())
                .into();
            inf.beta_covariance = inf
                .beta_covariance
                .take()
                .map(|cov| self.backtransform_covariance(cov.as_array()).into());
            inf.beta_standard_errors = inf
                .beta_covariance
                .as_ref()
                .map(|c| se_from_covariance(c.as_array()));
            inf.beta_covariance_corrected = inf
                .beta_covariance_corrected
                .take()
                .map(|cov| self.backtransform_covariance(&cov));
            inf.beta_standard_errors_corrected = inf
                .beta_covariance_corrected
                .as_ref()
                .map(se_from_covariance);
            inf.beta_covariance_frequentist = inf
                .beta_covariance_frequentist
                .take()
                .map(|cov| self.backtransform_covariance(&cov));
            // The influence matrix `F = H⁻¹·X'WX` is a mixed linear operator
            // (it transforms by SIMILARITY `F_orig = M·F_int·M⁻¹`, not
            // congruence). We do not carry the similarity primitive here, so
            // drop `F` rather than applying the wrong map; downstream code can
            // reconstruct it from the (now-preserved) original-basis `H` and
            // `X'WX` when it needs it.
            inf.coefficient_influence = None;
            // X'WX is a genuine congruence object under column-conditioning —
            // it transforms by EXACTLY the same map as the penalized Hessian
            // `H` (both are `Mᵀ·(·)_orig·M` internally, so `(·)_orig =
            // M⁻ᵀ·(·)_int·M⁻¹`): from `X_int = X_orig·M` we get
            // `X_intᵀ·W·X_int = Mᵀ·(X_orgᵀ·W·X_org)·M`. The Hessian is
            // back-transformed two lines above; back-transform the Gram with the
            // identical congruence so it survives in the original basis. This
            // keeps `X'WX = H − S(λ)` consistent (both factors mapped the same
            // way), restores the exact WPS corrected-EDF term `tr(X'WX·Σ_ρ)`
            // for every model carrying a parametric (non-intercept) term — that
            // trace is congruence-invariant, so it matches the internal-basis
            // value bit-for-bit — and lets the debiased-functional Riesz engine
            // recover `S(λ)·β` (issue #1622) instead of aborting on a missing
            // Gram. Previously this was unconditionally nulled, silently
            // degrading the corrected EDF to its conditional fallback and making
            // `debiased_functional` unavailable for the entire `y ~ x` /
            // `y ~ s(x) + z` class of Gaussian models.
            inf.weighted_gram = inf
                .weighted_gram
                .take()
                .map(|g| self.backtransform_penalized_hessian(&g));
            inf.bias_correction_beta = inf
                .bias_correction_beta
                .take()
                .map(|b| self.backtransform_beta(&b));
            inf.smoothing_correction = inf
                .smoothing_correction
                .take()
                .map(|cov| self.backtransform_covariance(&cov));
            inf.reparam_qs = None;
        }
        result.constraint_kkt = None;
        // `result.artifacts.pirls` is a self-consistent geometric bundle in the
        // PIRLS internal basis (`x_transformed`, `beta_transformed`,
        // `penalized_hessian_transformed`, and the per-observation
        // `final_eta`/`finalmu`/`solveworking_response`/weights, all paired in
        // that one frame). Observation-space quantities derived from it
        // — η̂_i, leverages a_ii, sandwich SEs — are invariant under the
        // invertible coefficient-space reparameterization that conditioning
        // introduces, so the bundle stays correct in its own coordinates and
        // we keep it instead of wiping `pirls: None`.
        result
    }
}

pub(crate) fn map_hessian_to_original_basis(
    pirls: &crate::pirls::PirlsResult,
) -> Result<Array2<f64>, EstimationError> {
    let qs = &pirls.reparam_result.qs;
    let h_t = &pirls.penalized_hessian_transformed;
    // H_original = Qs * H_transformed * Qs'
    // left_dot_matrix avoids densification for sparse Hessians.
    let tmp = h_t.left_dot_matrix(qs);
    let mut h = tmp.dot(&qs.t());
    // Two non-self-adjoint matmuls accumulate ~p · ε rounding noise that
    // breaks bitwise symmetry even though the analytic result `Q H Qᵀ` is
    // symmetric whenever `H_transformed` is.  Average opposite entries
    // explicitly so downstream `validate_dense_hessian_export` doesn't
    // reject otherwise-valid fits over rounding-noise asymmetry.
    gam_linalg::matrix::symmetrize_in_place(&mut h);
    Ok(h)
}

/// Scale a posterior covariance `H^{-1}` by the coefficient-covariance scale.
///
/// `Vb = H^{-1} * scale`. The multiplier is supplied by
/// `GlmLikelihoodSpec::coefficient_covariance_scale`: it is the profiled
/// residual variance `sigma^2` for the scale-free profiled Gaussian, and `1.0`
/// for every family whose IRLS working weight already carries the dispersion /
/// full Fisher information (Gamma, Tweedie, Beta, Negative-Binomial, and the
/// fixed-scale Poisson/Binomial). For the latter the stored `H = X'WX + S_λ`
/// is already the true penalized Hessian, so no further dispersion multiply is
/// applied — multiplying again would double-count the dispersion (#679).
/// Centralizing the scaling here keeps the contract visible at every covariance
/// construction site instead of being inlined as a bare `cov * scale`.
#[inline]
pub(crate) fn scaled_covariance(cov: Array2<f64>, phi: f64) -> Array2<f64> {
    if (phi - 1.0).abs() <= f64::EPSILON {
        cov
    } else {
        cov * phi
    }
}

#[cfg(test)]
mod weighted_gram_backtransform_tests {
    use super::*;
    use ndarray::{Array1, Array2};

    /// Build the conditioned (internal-basis) design `X_int` from an
    /// original-basis design `X_orig` by applying the same per-column
    /// centering/scaling that `ParametricColumnConditioning` derived from
    /// `X_orig`. `X_int = X_orig · M` (so `η = X_orig·β_orig = X_int·β_int`).
    fn condition_design(
        cond: &ParametricColumnConditioning,
        x_orig: &Array2<f64>,
    ) -> Array2<f64> {
        let mut x_int = x_orig.clone();
        let intercept = cond.intercept_idx.map(|idx| x_orig.column(idx).to_owned());
        for &(j, mean, scale) in &cond.columns {
            let mut col = x_int.column_mut(j);
            if mean != 0.0
                && let Some(ic) = intercept.as_ref()
            {
                col -= &(ic * mean);
            }
            if scale != 1.0 {
                col.mapv_inplace(|v| v / scale);
            }
        }
        x_int
    }

    fn weighted_gram(x: &Array2<f64>, w: &Array1<f64>) -> Array2<f64> {
        // XᵀWX with W = diag(w).
        let xw = x * &w.view().insert_axis(ndarray::Axis(1));
        x.t().dot(&xw)
    }

    /// The crux of issue #1622: the weighted Gram `X'WX` is a genuine congruence
    /// object under column-conditioning, transforming by the SAME map as the
    /// penalized Hessian. Back-transforming the internal-basis Gram with
    /// `backtransform_penalized_hessian` (`M⁻ᵀ·(·)·M⁻¹`) must reproduce the
    /// original-basis Gram `X_origᵀ W X_orig` exactly — which is what lets
    /// `debiased_functional` recover `S(λ)·β` and the WPS correction recover
    /// `tr(X'WX·Σ_ρ)` for models carrying a parametric term. Before the fix the
    /// Gram was nulled here, so this identity could never be exercised.
    #[test]
    fn backtransformed_internal_gram_equals_original_basis_gram() {
        // p = 3: intercept (col 0) + two non-constant parametric covariates that
        // both get centered AND scaled (distinct means / spreads).
        let n = 40usize;
        let mut x_orig = Array2::<f64>::ones((n, 3));
        for i in 0..n {
            let t = i as f64;
            x_orig[[i, 1]] = 3.0 + 0.5 * t; // mean ≈ 12.75, nonzero spread
            x_orig[[i, 2]] = -7.0 + (t * 0.31).sin() * 4.0;
        }
        // Heteroscedastic positive weights so the test is not secretly W = I.
        let w = Array1::from_shape_fn(n, |i| 0.25 + (i as f64 * 0.137).cos().abs());

        let design = DesignMatrix::from(x_orig.clone());
        let cond = ParametricColumnConditioning::from_column_indices(&design, &[0, 1, 2]);
        assert!(cond.is_active(), "parametric columns must trigger conditioning");
        assert_eq!(cond.intercept_idx, Some(0));
        assert_eq!(cond.columns.len(), 2, "cols 1 and 2 are conditioned");

        let x_int = condition_design(&cond, &x_orig);
        let gram_int = weighted_gram(&x_int, &w);
        let gram_orig_expected = weighted_gram(&x_orig, &w);

        let gram_orig_actual = cond.backtransform_penalized_hessian(&gram_int);

        let max_err = gram_orig_actual
            .iter()
            .zip(gram_orig_expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_err < 1e-9,
            "back-transformed internal Gram must equal X_origᵀWX_orig; max |Δ| = {max_err:e}\n\
             actual=\n{gram_orig_actual:?}\nexpected=\n{gram_orig_expected:?}"
        );
    }

    /// `tr(X'WX · Σ_ρ)` (the WPS corrected-EDF term) is congruence-invariant:
    /// computing it from the internal-basis Gram with the internal-basis
    /// covariance gives the identical value as from the back-transformed
    /// original-basis pair. This is why restoring the Gram cannot perturb the
    /// corrected EDF for pure-smooth models while finally making it correct for
    /// parametric ones.
    #[test]
    fn wps_trace_is_invariant_under_backtransform() {
        let n = 24usize;
        let mut x_orig = Array2::<f64>::ones((n, 3));
        for i in 0..n {
            let t = i as f64;
            x_orig[[i, 1]] = 1.0 + 0.7 * t;
            x_orig[[i, 2]] = (t * 0.21).cos() * 2.5 - 0.4 * t;
        }
        let w = Array1::from_shape_fn(n, |i| 0.5 + (i as f64 * 0.09).sin().abs());

        let design = DesignMatrix::from(x_orig.clone());
        let cond = ParametricColumnConditioning::from_column_indices(&design, &[0, 1, 2]);

        let x_int = condition_design(&cond, &x_orig);
        let gram_int = weighted_gram(&x_int, &w);

        // Arbitrary SPD smoothing-uncertainty covariance Σ in the internal
        // basis; back-transform as a COVARIANCE (M·Σ·Mᵀ) — the companion map to
        // the Gram's congruence — via left_multiply_by_m / right_multiply_by_m_transpose.
        let mut sigma_int = Array2::<f64>::eye(3) * 0.3;
        sigma_int[[1, 2]] = 0.05;
        sigma_int[[2, 1]] = 0.05;

        let gram_orig = cond.backtransform_penalized_hessian(&gram_int);
        let sigma_orig =
            cond.right_multiply_by_m_transpose(&cond.left_multiply_by_m(&sigma_int));

        let trace = |a: &Array2<f64>, b: &Array2<f64>| -> f64 {
            let k = a.nrows();
            (0..k)
                .map(|i| (0..k).map(|j| a[[i, j]] * b[[j, i]]).sum::<f64>())
                .sum()
        };
        let t_int = trace(&gram_int, &sigma_int);
        let t_orig = trace(&gram_orig, &sigma_orig);
        assert!(
            (t_int - t_orig).abs() < 1e-9,
            "tr(X'WX·Σ) must be congruence-invariant: internal={t_int} original={t_orig}"
        );
    }
}
