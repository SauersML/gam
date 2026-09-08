use super::*;
use gam_problem::dispersion_cov::se_from_covariance;

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
    ) -> Result<ExternalOptimResult, EstimationError> {
        if !self.is_active() {
            return Ok(result);
        }
        result.beta = self.backtransform_beta(&result.beta);
        if let Some(geometry) = result.geometry.as_mut() {
            geometry.penalized_hessian = self
                .backtransform_penalized_hessian(geometry.penalized_hessian.as_array())
                .into();
            if let Some(posterior) = geometry.constrained_posterior.as_mut() {
                posterior.constraints.a =
                    self.right_multiply_by_m_inv(&posterior.constraints.a);
                posterior.mode = self.backtransform_beta(&posterior.mode);
                if let Some((unconstrained_center, correction)) =
                    posterior.available_parts_mut()
                {
                    *unconstrained_center = self.backtransform_beta(unconstrained_center);
                    if let Some(correction) = correction {
                        correction.lift = self.left_multiply_by_m(&correction.lift);
                    }
                }
            }
        }
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
                .map(|c| se_from_covariance(c.as_array()))
                .transpose()
                .map_err(|err| {
                    EstimationError::InvalidInput(format!(
                        "back-transformed conditional covariance is invalid: {err}"
                    ))
                })?;
            inf.beta_covariance_corrected = inf
                .beta_covariance_corrected
                .take()
                .map(|cov| self.backtransform_covariance(&cov));
            inf.beta_standard_errors_corrected = inf
                .beta_covariance_corrected
                .as_ref()
                .map(se_from_covariance)
                .transpose()
                .map_err(|err| {
                    EstimationError::InvalidInput(format!(
                        "back-transformed corrected covariance is invalid: {err}"
                    ))
                })?;
            inf.beta_covariance_frequentist = inf
                .beta_covariance_frequentist
                .take()
                .map(|cov| self.backtransform_covariance(&cov));
            // The influence matrix `F = H⁻¹·X'WX` is a mixed linear operator: it
            // transforms by SIMILARITY, not congruence. From `X_int = X_orig·M`,
            //
            //     H_int    = Mᵀ·H_orig·M,      (X'WX)_int = Mᵀ·(X'WX)_orig·M
            //  ⇒  F_int    = H_int⁻¹(X'WX)_int = M⁻¹·F_orig·M
            //  ⇒  F_orig   = M·F_int·M⁻¹,
            //
            // which is exactly [`Self::left_multiply_by_m`] composed with
            // [`Self::right_multiply_by_m_inv`] — both primitives already exist on
            // this type (they are what `backtransform_covariance` and
            // `backtransform_penalized_hessian` are built from). This site used to
            // drop `F` with the note "we do not carry the similarity primitive
            // here"; that was not true, and the cost of the drop was invisible
            // because every fixture that reads `F` back is a `y ~ s(x)` model with
            // NO conditioned parametric column, i.e. `is_active() == false`, where
            // this whole function returns early (#2672).
            //
            // What the drop cost: `F` is the sole input to Wood's
            // smoothing-selection-corrected reference `edf1 = 2·tr(F_jj) −
            // tr(F_jj²)`, so `wood_reference_df` returned `None` for EVERY model
            // carrying a non-intercept parametric term and the smooth-term LR test
            // silently fell back to the raw conditional EDF — the anti-conservative
            // reference #1766 replaced, measured there at FPR ~0.15 against a
            // nominal 0.05. `per_term_edf` and the smooth summary lost their
            // primary channel the same way.
            //
            // The similarity map is exact, not an approximation: it preserves
            // `tr(F) = edf` and the consistency identity `H·F = X'WX` in the
            // original basis, because `H_orig·F_orig = M⁻ᵀH_int M⁻¹·M F_int M⁻¹ =
            // M⁻ᵀ(H_int F_int)M⁻¹ = M⁻ᵀ(X'WX)_int M⁻¹ = (X'WX)_orig`, the same
            // congruence the Gram gets below. `F` must NOT be symmetrized here for
            // the reasons `optimizer.rs` records at its assembly.
            inf.coefficient_influence = inf
                .coefficient_influence
                .take()
                .map(|f_int| self.left_multiply_by_m(&self.right_multiply_by_m_inv(&f_int)));
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
            // `bias_correction_jacobian` is `A = I + H⁻¹S = 2I − F`, the fixed-ρ
            // linearization of the bias-corrected estimator — the SAME kind of
            // object as `F` above and therefore the same similarity map.
            // `fit.rs`'s response-rescale path already treats them as one class:
            // it applies `rescale_influence_coordinates` to
            // `coefficient_influence` and `bias_correction_jacobian` together,
            // and `rescale_covariance_coordinates` to the covariances. Here the
            // Jacobian's own centre `b̂` is back-transformed one line above as a
            // coefficient vector while the Jacobian itself was left INTERNAL, so
            // prediction's `A·V·Aᵀ` band (#1870) paired an internal-basis `A`
            // with an original-basis `V` on every conditioned model. Same missing
            // primitive as `F`, same repair (#2672).
            inf.bias_correction_jacobian = inf
                .bias_correction_jacobian
                .take()
                .map(|a| self.left_multiply_by_m(&self.right_multiply_by_m_inv(&a)));
            inf.smoothing_correction = inf
                .smoothing_correction
                .take()
                .map(|cov| self.backtransform_covariance(&cov));
            // The RETAINED first-order correction `J·Var(ρ)·Jᵀ` is the same kind
            // of object as the primary one above — a coefficient-space covariance
            // — and takes the same congruence. It was left in the INTERNAL basis
            // while every matrix it is contracted against (`weighted_gram`, the
            // penalized Hessian) was carried to the original one, so
            // `tr(X'WX · C)` mixed two frames on any conditioned model. The two
            // corrections are read by different consumers — `model_comparison`'s
            // WPS corrected EDF and, since #2672, the smooth-term LR reference
            // d.f. — which is exactly why the primary being mapped did not imply
            // this one was: nothing pairs them, so nothing compared them (#2672).
            inf.smoothing_correction_first_order = inf
                .smoothing_correction_first_order
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
        Ok(result)
    }
}

pub(crate) fn map_hessian_to_original_basis(
    pirls: &crate::pirls::PirlsResult,
) -> Result<Array2<f64>, EstimationError> {
    let qs = &pirls.reparam_result.qs;
    // The accepted posterior precision is the stabilized Hessian. Any solver
    // ridge is part of the minted objective and its RidgePassport; exporting
    // the pre-stabilization matrix would make dense inference, factorized
    // prediction, and constrained-posterior moments describe different local
    // Gaussians.
    let h_t = &pirls.stabilizedhessian_transformed;
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
    fn condition_design(cond: &ParametricColumnConditioning, x_orig: &Array2<f64>) -> Array2<f64> {
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
        assert!(
            cond.is_active(),
            "parametric columns must trigger conditioning"
        );
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

    /// Dense inverse of a small square matrix by Gauss–Jordan with partial
    /// pivoting. Test-local so the identity below is checked against ordinary
    /// arithmetic rather than against the same factorization the production path
    /// uses.
    fn dense_inverse(a: &Array2<f64>) -> Array2<f64> {
        let n = a.nrows();
        let mut aug = Array2::<f64>::zeros((n, 2 * n));
        for i in 0..n {
            for j in 0..n {
                aug[[i, j]] = a[[i, j]];
            }
            aug[[i, n + i]] = 1.0;
        }
        for col in 0..n {
            let mut pivot = col;
            for row in (col + 1)..n {
                if aug[[row, col]].abs() > aug[[pivot, col]].abs() {
                    pivot = row;
                }
            }
            assert!(aug[[pivot, col]].abs() > 1e-12, "singular matrix in test");
            if pivot != col {
                for j in 0..(2 * n) {
                    let tmp = aug[[col, j]];
                    aug[[col, j]] = aug[[pivot, j]];
                    aug[[pivot, j]] = tmp;
                }
            }
            let d = aug[[col, col]];
            for j in 0..(2 * n) {
                aug[[col, j]] /= d;
            }
            for row in 0..n {
                if row == col {
                    continue;
                }
                let f = aug[[row, col]];
                if f == 0.0 {
                    continue;
                }
                for j in 0..(2 * n) {
                    aug[[row, j]] -= f * aug[[col, j]];
                }
            }
        }
        let mut inv = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                inv[[i, j]] = aug[[i, n + j]];
            }
        }
        inv
    }

    /// #2672: the coefficient influence `F = H⁻¹·X'WX` transforms by SIMILARITY
    /// under column-conditioning, `F_orig = M·F_int·M⁻¹`, and that map is exactly
    /// [`ParametricColumnConditioning::left_multiply_by_m`] composed with
    /// [`ParametricColumnConditioning::right_multiply_by_m_inv`].
    ///
    /// This site used to drop `F` on the stated ground that "we do not carry the
    /// similarity primitive here". Both halves of it are defined on this very
    /// type, and the drop silently disabled Wood's `edf1` reference — the entire
    /// #1766 smooth-term LR calibration — for every model carrying a conditioned
    /// parametric column. The identity is checked against a directly-assembled
    /// original-basis `F`, not against the production solve.
    #[test]
    fn backtransformed_influence_equals_original_basis_influence_2672() {
        let n = 32usize;
        let p = 3usize;
        let mut x_orig = Array2::<f64>::ones((n, p));
        for i in 0..n {
            let t = i as f64;
            x_orig[[i, 1]] = 4.0 + 0.9 * t;
            x_orig[[i, 2]] = -2.5 + (t * 0.27).sin() * 3.0;
        }
        let w = Array1::from_shape_fn(n, |i| 0.4 + (i as f64 * 0.11).cos().abs());

        let design = DesignMatrix::from(x_orig.clone());
        let cond = ParametricColumnConditioning::from_column_indices(&design, &[0, 1, 2]);
        assert!(cond.is_active(), "conditioning must be active");

        // `M` and `M⁻¹` materialized from the two primitives themselves, so the
        // composition below is checked rather than assumed to be inverse.
        let eye = Array2::<f64>::eye(p);
        let m = cond.left_multiply_by_m(&eye);
        let m_inv = cond.right_multiply_by_m_inv(&eye);
        let round_trip = m.dot(&m_inv);
        let round_trip_err = round_trip
            .iter()
            .zip(eye.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            round_trip_err < 1e-12,
            "left_multiply_by_m and right_multiply_by_m_inv must be mutually \
             inverse; max |M·M⁻¹ − I| = {round_trip_err:e}"
        );

        let x_int = condition_design(&cond, &x_orig);
        let gram_int = weighted_gram(&x_int, &w);
        let gram_orig = weighted_gram(&x_orig, &w);

        // A non-trivial symmetric PSD penalty in the ORIGINAL basis, pushed into
        // the internal basis by the congruence `S_int = Mᵀ·S_orig·M` that the
        // design change induces.
        let mut s_orig = Array2::<f64>::zeros((p, p));
        s_orig[[1, 1]] = 2.0;
        s_orig[[2, 2]] = 5.0;
        s_orig[[1, 2]] = -0.75;
        s_orig[[2, 1]] = -0.75;
        let s_int = m.t().dot(&s_orig).dot(&m);

        let h_orig = &gram_orig + &s_orig;
        let h_int = &gram_int + &s_int;
        let f_orig_expected = dense_inverse(&h_orig).dot(&gram_orig);
        let f_int = dense_inverse(&h_int).dot(&gram_int);

        let f_orig_actual = cond.left_multiply_by_m(&cond.right_multiply_by_m_inv(&f_int));

        let scale = f_orig_expected
            .iter()
            .copied()
            .map(f64::abs)
            .fold(0.0_f64, f64::max)
            .max(1.0);
        let max_err = f_orig_actual
            .iter()
            .zip(f_orig_expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_err < 1e-9 * scale,
            "M·F_int·M⁻¹ must equal H_orig⁻¹·(X'WX)_orig; max |Δ| = {max_err:e} \
             (scale {scale:e})\nactual=\n{f_orig_actual:?}\nexpected=\n{f_orig_expected:?}"
        );

        // The EDF channel every downstream consumer reads: tr(F) is a similarity
        // invariant, so the back-transform cannot move it. This is also the
        // property that makes the restored `F` agree with `edf_total`.
        let tr = |a: &Array2<f64>| (0..a.nrows()).map(|i| a[[i, i]]).sum::<f64>();
        assert!(
            (tr(&f_orig_actual) - tr(&f_int)).abs() < 1e-9,
            "tr(F) is similarity-invariant: internal {} vs back-transformed {}",
            tr(&f_int),
            tr(&f_orig_actual)
        );

        // And the consistency identity the whole inference block is tied to,
        // asserted in the ORIGINAL basis where consumers read it.
        let hf = h_orig.dot(&f_orig_actual);
        let identity_err = hf
            .iter()
            .zip(gram_orig.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let gram_scale = gram_orig
            .iter()
            .copied()
            .map(f64::abs)
            .fold(0.0_f64, f64::max)
            .max(1.0);
        assert!(
            identity_err < 1e-9 * gram_scale,
            "H·F = X'WX must survive the back-transform; max |Δ| = {identity_err:e} \
             (scale {gram_scale:e})"
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
        let sigma_orig = cond.right_multiply_by_m_transpose(&cond.left_multiply_by_m(&sigma_int));

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
