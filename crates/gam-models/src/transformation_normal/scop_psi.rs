use super::*;

impl TransformationNormalFamily {
    /// First derivative of the profiled objective pieces with respect to one
    /// covariate-basis deformation axis `psi`.
    ///
    /// Direct-α chart (gam#2306): `psi` differentiates the covariate row, not
    /// `beta`, and the transform is LINEAR in the coordinates, so
    /// `α_k,psi = beta_k·c_psi`, `h_psi = Σ_k r_k α_k,psi`, and the only mixed
    /// β-ψ second derivative is through the deformed design row itself:
    /// `∂²h/∂A_{kp}∂ψ = r_k · c_psi[p]`. The objective contribution is
    ///
    /// `L_psi = w [ h h_psi - hp_psi / hp ]`
    ///
    /// because the minimized criterion is `0.5 h^2 - log(hp)`. Differentiating
    /// once with respect to `beta_a` gives `score_psi`.
    pub(crate) fn scop_psi_terms(
        &self,
        beta: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
        op: &TensorKroneckerPsiOperator,
        op_arc: Arc<dyn CustomFamilyPsiDerivativeOperator>,
        axis: usize,
    ) -> Result<ExactNewtonJointPsiTerms, String> {
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if beta.len() != p_total {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi terms beta length {} != p_resp({p_resp}) * p_cov({p_cov})",
                    beta.len()
                ),
            }
            .into());
        }
        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP psi beta reshape failed: {e}"))?;
        let cov = self
            .covariate_dense_arc()
            .map_err(|e| format!("SCOP psi terms require cached covariate design: {e}"))?;
        let cov_psi = op
            .materialize_cov_first_axis(axis)
            .map_err(|e| format!("SCOP psi materialize_cov_first failed: {e}"))?;
        if cov_psi.nrows() != n || cov_psi.ncols() != p_cov {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi covariate derivative shape {}x{} != expected {}x{}",
                    cov_psi.nrows(),
                    cov_psi.ncols(),
                    n,
                    p_cov
                ),
            }
            .into());
        }

        let weights = self.effective_weights();
        let h = row_quantities.h.as_ref();
        let h_prime = row_quantities.h_prime.as_ref();
        let mut objective_psi = 0.0;
        let mut score_psi = Array1::<f64>::zeros(p_total);
        let endpoint_basis = [
            self.response_upper_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint upper basis is not contiguous".to_string())?,
            self.response_lower_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint lower basis is not contiguous".to_string())?,
        ];
        let mut alpha_psi = vec![0.0; p_resp];

        for i in 0..n {
            let cov_row = cov.row(i);
            let psi_row = cov_psi.row(i);
            let rv = self.response_val_basis.row(i);
            let rd = self.response_deriv_basis.row(i);
            let wi = weights[i];
            let hi = h[i];
            let hp = h_prime[i];
            let inv_hp = 1.0 / hp;
            let inv_hp_sq = inv_hp * inv_hp;
            let q = row_quantities.endpoint_q[i];

            for k in 0..p_resp {
                alpha_psi[k] = beta_mat.row(k).dot(&psi_row);
            }

            let mut h_psi = 0.0;
            let mut hp_psi = 0.0;
            let mut endpoint_psi = [0.0; 2];
            for k in 0..p_resp {
                h_psi += rv[k] * alpha_psi[k];
                hp_psi += rd[k] * alpha_psi[k];
                endpoint_psi[0] += endpoint_basis[0][k] * alpha_psi[k];
                endpoint_psi[1] += endpoint_basis[1][k] * alpha_psi[k];
            }

            objective_psi +=
                wi * (hi * h_psi - hp_psi * inv_hp + endpoint_chain_first(&q, endpoint_psi));

            for k in 0..p_resp {
                for c in 0..p_cov {
                    let idx = k * p_cov + c;
                    let h_a = rv[k] * cov_row[c];
                    let hp_a = rd[k] * cov_row[c];
                    let hpsi_a = rv[k] * psi_row[c];
                    let hppsi_a = rd[k] * psi_row[c];
                    let endpoint_a = [
                        endpoint_basis[0][k] * cov_row[c],
                        endpoint_basis[1][k] * cov_row[c],
                    ];
                    let endpoint_psi_a = [
                        endpoint_basis[0][k] * psi_row[c],
                        endpoint_basis[1][k] * psi_row[c],
                    ];
                    score_psi[idx] += wi
                        * (h_a * h_psi + hi * hpsi_a - hppsi_a * inv_hp
                            + hp_psi * hp_a * inv_hp_sq
                            + endpoint_chain_second(&q, endpoint_psi, endpoint_a, endpoint_psi_a));
                }
            }
        }

        let hessian_psi_operator: Arc<dyn HyperOperator> =
            Arc::new(TransformationNormalPsiHessianOperator::new(
                Arc::new(self.clone()),
                beta.clone(),
                Arc::clone(&op_arc),
                axis,
                Arc::clone(&row_quantities.alpha),
                Arc::clone(&row_quantities.h),
                Arc::clone(&row_quantities.h_prime),
                Arc::clone(&row_quantities.endpoint_q),
            ));

        Ok(ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi: Array2::zeros((0, 0)),
            hessian_psi_operator: Some(hessian_psi_operator),
        })
    }

    pub(crate) fn scop_psi_hessian_apply_from_operator(
        &self,
        beta: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
        op: &TensorKroneckerPsiOperator,
        axis: usize,
        direction: &Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        let cov = self
            .covariate_dense_arc()
            .map_err(|e| format!("SCOP psi Hessian apply requires cached covariate design: {e}"))?;
        let cov_psi = op
            .materialize_cov_first_axis(axis)
            .map_err(|e| format!("SCOP psi Hessian apply materialize_cov_first failed: {e}"))?;
        self.scop_psi_hessian_apply_from_operator_with_cov(
            beta,
            row_quantities,
            axis,
            &cov,
            &cov_psi,
            direction,
        )
    }

    pub(crate) fn scop_psi_hessian_apply_from_operator_with_cov(
        &self,
        beta: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
        axis: usize,
        cov: &Array2<f64>,
        cov_psi: &Array2<f64>,
        direction: &Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if cov.nrows() != n || cov.ncols() != p_cov {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi Hessian apply covariate shape {}x{} != expected {}x{}",
                    cov.nrows(),
                    cov.ncols(),
                    n,
                    p_cov
                ),
            }
            .into());
        }
        if beta.len() != p_total || direction.len() != p_total {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi Hessian apply length mismatch: beta={}, direction={}, expected={p_total}",
                beta.len(),
                direction.len()
            ) }.into());
        }
        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP psi Hessian apply beta reshape failed: {e}"))?;
        let dir_mat = direction
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP psi Hessian apply direction reshape failed: {e}"))?;
        if cov_psi.nrows() != n || cov_psi.ncols() != p_cov {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi Hessian apply covariate derivative shape {}x{} for axis {axis} != expected {}x{}",
                cov_psi.nrows(),
                cov_psi.ncols(),
                n,
                p_cov
            ) }.into());
        }

        let weights = self.effective_weights();
        let h_prime = row_quantities.h_prime.as_ref();
        let endpoint_basis = [
            self.response_upper_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint upper basis is not contiguous".to_string())?,
            self.response_lower_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint lower basis is not contiguous".to_string())?,
        ];
        let mut out = Array1::<f64>::zeros(p_total);
        let mut alpha_dir = vec![0.0; p_resp];
        let mut alpha_psi = vec![0.0; p_resp];
        let mut alpha_psi_dir = vec![0.0; p_resp];

        for i in 0..n {
            let cov_row = cov.row(i);
            let psi_row = cov_psi.row(i);
            let rv = self.response_val_basis.row(i);
            let rd = self.response_deriv_basis.row(i);
            let wi = weights[i];
            let hp = h_prime[i];
            let inv_hp = 1.0 / hp;
            let inv_hp_sq = inv_hp * inv_hp;
            let inv_hp_cu = inv_hp_sq * inv_hp;
            let q = row_quantities.endpoint_q[i];

            for k in 0..p_resp {
                alpha_dir[k] = dir_mat.row(k).dot(&cov_row);
                alpha_psi[k] = beta_mat.row(k).dot(&psi_row);
                alpha_psi_dir[k] = dir_mat.row(k).dot(&psi_row);
            }

            // Linear chart: every transform functional is a plain
            // basis-weighted sum, and the mixed β/ψ dependence flows entirely
            // through the deformed design row (`alpha_psi_dir`).
            let mut h_dir = 0.0;
            let mut hp_dir = 0.0;
            let mut h_psi = 0.0;
            let mut hp_psi = 0.0;
            let mut h_psi_dir = 0.0;
            let mut hp_psi_dir = 0.0;
            let mut endpoint_psi = [0.0; 2];
            let mut endpoint_dir = [0.0; 2];
            let mut endpoint_psi_dir = [0.0; 2];
            for k in 0..p_resp {
                h_dir += rv[k] * alpha_dir[k];
                hp_dir += rd[k] * alpha_dir[k];
                h_psi += rv[k] * alpha_psi[k];
                hp_psi += rd[k] * alpha_psi[k];
                h_psi_dir += rv[k] * alpha_psi_dir[k];
                hp_psi_dir += rd[k] * alpha_psi_dir[k];
                for e in 0..2 {
                    let basis = endpoint_basis[e];
                    endpoint_psi[e] += basis[k] * alpha_psi[k];
                    endpoint_dir[e] += basis[k] * alpha_dir[k];
                    endpoint_psi_dir[e] += basis[k] * alpha_psi_dir[k];
                }
            }
            let d_inv_hp = -hp_dir * inv_hp_sq;
            let d_inv_hp_sq = -2.0 * hp_dir * inv_hp_cu;

            for k in 0..p_resp {
                for c in 0..p_cov {
                    let idx = k * p_cov + c;
                    let h_a = rv[k] * cov_row[c];
                    let hp_a = rd[k] * cov_row[c];
                    let hpsi_a = rv[k] * psi_row[c];
                    let hppsi_a = rd[k] * psi_row[c];
                    let endpoint_a = [
                        endpoint_basis[0][k] * cov_row[c],
                        endpoint_basis[1][k] * cov_row[c],
                    ];
                    let endpoint_psi_a = [
                        endpoint_basis[0][k] * psi_row[c],
                        endpoint_basis[1][k] * psi_row[c],
                    ];
                    // β-directional derivatives of the (constant) coefficient
                    // factors vanish for the linear chart.
                    let endpoint_a_dir = [0.0, 0.0];
                    let endpoint_psi_a_dir = [0.0, 0.0];
                    let value = h_a * h_psi_dir + h_dir * hpsi_a
                        - hppsi_a * d_inv_hp
                        + hp_psi_dir * hp_a * inv_hp_sq
                        + hp_psi * hp_a * d_inv_hp_sq
                        + endpoint_chain_third(
                            &q,
                            endpoint_psi,
                            endpoint_a,
                            endpoint_dir,
                            endpoint_psi_a,
                            endpoint_psi_dir,
                            endpoint_a_dir,
                            endpoint_psi_a_dir,
                        );
                    out[idx] += wi * value;
                }
            }
        }

        Ok(out)
    }

    pub(crate) fn scop_psi_hessian_hvp_mat_from_cov(
        &self,
        beta: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
        axis: usize,
        cov: &Array2<f64>,
        cov_psi: &Array2<f64>,
        factor: ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        let rank = factor.ncols();
        if cov.nrows() != n || cov.ncols() != p_cov {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi Hessian batched apply covariate shape {}x{} != expected {}x{}",
                    cov.nrows(),
                    cov.ncols(),
                    n,
                    p_cov
                ),
            }
            .into());
        }
        if cov_psi.nrows() != n || cov_psi.ncols() != p_cov {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi Hessian batched apply covariate derivative shape {}x{} for axis {axis} != expected {}x{}",
                cov_psi.nrows(),
                cov_psi.ncols(),
                n,
                p_cov
            ) }.into());
        }
        if beta.len() != p_total || factor.nrows() != p_total {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi Hessian batched apply length mismatch: beta={}, factor_rows={}, expected={p_total}",
                beta.len(),
                factor.nrows()
            ) }.into());
        }
        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP psi Hessian batched apply beta reshape failed: {e}"))?;
        let endpoint_basis = [
            self.response_upper_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint upper basis is not contiguous".to_string())?,
            self.response_lower_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint lower basis is not contiguous".to_string())?,
        ];

        struct PsiBatchedAccum {
            pub(crate) hvp: Array2<f64>,
            pub(crate) alpha_psi: Vec<f64>,
            pub(crate) alpha_dir: Vec<f64>,
            pub(crate) alpha_psi_dir: Vec<f64>,
            pub(crate) h_dir: Vec<f64>,
            pub(crate) hp_dir: Vec<f64>,
            pub(crate) h_psi_dir: Vec<f64>,
            pub(crate) hp_psi_dir: Vec<f64>,
            pub(crate) endpoint_dir: Vec<[f64; 2]>,
            pub(crate) endpoint_psi_dir: Vec<[f64; 2]>,
        }

        impl PsiBatchedAccum {
            pub(crate) fn new(p_total: usize, p_resp: usize, rank: usize) -> Self {
                let projected_len = p_resp * rank;
                Self {
                    hvp: Array2::<f64>::zeros((p_total, rank)),
                    alpha_psi: vec![0.0; p_resp],
                    alpha_dir: vec![0.0; projected_len],
                    alpha_psi_dir: vec![0.0; projected_len],
                    h_dir: vec![0.0; rank],
                    hp_dir: vec![0.0; rank],
                    h_psi_dir: vec![0.0; rank],
                    hp_psi_dir: vec![0.0; rank],
                    endpoint_dir: vec![[0.0; 2]; rank],
                    endpoint_psi_dir: vec![[0.0; 2]; rank],
                }
            }

            pub(crate) fn merge(mut self, rhs: Self) -> Self {
                self.hvp += &rhs.hvp;
                self
            }
        }

        let weights = self.effective_weights();
        let h_prime = row_quantities.h_prime.as_ref();
        let accum = gam_linalg::pairwise_reduce::par_deterministic_block_fold(
            n,
            |range| {
                let mut acc = PsiBatchedAccum::new(p_total, p_resp, rank);
                for i in range {
                    let cov_row = cov.row(i);
                    let psi_row = cov_psi.row(i);
                    let rv = self.response_val_basis.row(i);
                    let rd = self.response_deriv_basis.row(i);
                    let wi = weights[i];
                    let hp = h_prime[i];
                    let inv_hp = 1.0 / hp;
                    let inv_hp_sq = inv_hp * inv_hp;
                    let inv_hp_cu = inv_hp_sq * inv_hp;
                    let q = row_quantities.endpoint_q[i];

                    for k in 0..p_resp {
                        acc.alpha_psi[k] = beta_mat.row(k).dot(&psi_row);
                    }

                    acc.alpha_dir.fill(0.0);
                    acc.alpha_psi_dir.fill(0.0);
                    for k in 0..p_resp {
                        let factor_row_base = k * p_cov;
                        let projected_base = k * rank;
                        for cidx in 0..p_cov {
                            let factor_row = factor_row_base + cidx;
                            let cov_v = cov_row[cidx];
                            let psi_v = psi_row[cidx];
                            for col in 0..rank {
                                let coeff = factor[[factor_row, col]];
                                let idx = projected_base + col;
                                acc.alpha_dir[idx] += coeff * cov_v;
                                acc.alpha_psi_dir[idx] += coeff * psi_v;
                            }
                        }
                    }

                    let (h_psi, hp_psi, endpoint_psi) =
                        scop_psi_marginal(rv, rd, p_resp, endpoint_basis, &acc.alpha_psi);

                    for col in 0..rank {
                        acc.h_dir[col] = 0.0;
                        acc.hp_dir[col] = 0.0;
                        acc.h_psi_dir[col] = 0.0;
                        acc.hp_psi_dir[col] = 0.0;
                        acc.endpoint_dir[col] = [0.0, 0.0];
                        acc.endpoint_psi_dir[col] = [0.0, 0.0];
                    }
                    for k in 0..p_resp {
                        for col in 0..rank {
                            let idx = k * rank + col;
                            let a_dir = acc.alpha_dir[idx];
                            let a_psi_dir = acc.alpha_psi_dir[idx];
                            acc.h_dir[col] += rv[k] * a_dir;
                            acc.hp_dir[col] += rd[k] * a_dir;
                            acc.h_psi_dir[col] += rv[k] * a_psi_dir;
                            acc.hp_psi_dir[col] += rd[k] * a_psi_dir;
                            for e in 0..2 {
                                let basis = endpoint_basis[e];
                                acc.endpoint_dir[col][e] += basis[k] * a_dir;
                                acc.endpoint_psi_dir[col][e] += basis[k] * a_psi_dir;
                            }
                        }
                    }

                    for k in 0..p_resp {
                        let offset = k * p_cov;
                        let rvk = rv[k];
                        let rdk = rd[k];
                        let e_k = [endpoint_basis[0][k], endpoint_basis[1][k]];
                        for cidx in 0..p_cov {
                            let c = cov_row[cidx];
                            let psi = psi_row[cidx];
                            let h_a = rvk * c;
                            let hp_a = rdk * c;
                            let hpsi_a = rvk * psi;
                            let hppsi_a = rdk * psi;
                            let endpoint_a = [e_k[0] * c, e_k[1] * c];
                            let endpoint_psi_a = [e_k[0] * psi, e_k[1] * psi];
                            let out_idx = offset + cidx;
                            for col in 0..rank {
                                let d_inv_hp = -acc.hp_dir[col] * inv_hp_sq;
                                let d_inv_hp_sq = -2.0 * acc.hp_dir[col] * inv_hp_cu;
                                let value = h_a * acc.h_psi_dir[col]
                                    + acc.h_dir[col] * hpsi_a
                                    - hppsi_a * d_inv_hp
                                    + acc.hp_psi_dir[col] * hp_a * inv_hp_sq
                                    + hp_psi * hp_a * d_inv_hp_sq
                                    + endpoint_chain_third(
                                        &q,
                                        endpoint_psi,
                                        endpoint_a,
                                        acc.endpoint_dir[col],
                                        endpoint_psi_a,
                                        acc.endpoint_psi_dir[col],
                                        [0.0, 0.0],
                                        [0.0, 0.0],
                                    );
                                acc.hvp[[out_idx, col]] += wi * value;
                            }
                        }
                    }
                }
                acc
            },
            |left, right| left.merge(right),
        )
        .unwrap_or_else(|| PsiBatchedAccum::new(p_total, p_resp, rank));

        Ok(accum.hvp)
    }

    pub(crate) fn scop_psi_hessian_trace_factor_from_cov(
        &self,
        beta: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
        axis: usize,
        cov: &Array2<f64>,
        cov_psi: &Array2<f64>,
        factor: ArrayView2<'_, f64>,
    ) -> Result<f64, String> {
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        let rank = factor.ncols();
        if cov.nrows() != n || cov.ncols() != p_cov {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi Hessian projected trace covariate shape {}x{} != expected {}x{}",
                    cov.nrows(),
                    cov.ncols(),
                    n,
                    p_cov
                ),
            }
            .into());
        }
        if cov_psi.nrows() != n || cov_psi.ncols() != p_cov {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi Hessian projected trace covariate derivative shape {}x{} for axis {axis} != expected {}x{}",
                cov_psi.nrows(),
                cov_psi.ncols(),
                n,
                p_cov
            ) }.into());
        }
        if beta.len() != p_total || factor.nrows() != p_total {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi Hessian projected trace length mismatch: beta={}, factor_rows={}, expected={p_total}",
                beta.len(),
                factor.nrows()
            ) }.into());
        }
        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP psi Hessian projected trace beta reshape failed: {e}"))?;
        let endpoint_basis = [
            self.response_upper_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint upper basis is not contiguous".to_string())?,
            self.response_lower_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint lower basis is not contiguous".to_string())?,
        ];

        struct PsiTraceAccum {
            pub(crate) value: f64,
            pub(crate) alpha_psi: Vec<f64>,
            pub(crate) alpha_dir: Vec<f64>,
            pub(crate) alpha_psi_dir: Vec<f64>,
            pub(crate) h_dir: Vec<f64>,
            pub(crate) hp_dir: Vec<f64>,
            pub(crate) h_psi_dir: Vec<f64>,
            pub(crate) hp_psi_dir: Vec<f64>,
            pub(crate) endpoint_dir: Vec<[f64; 2]>,
            pub(crate) endpoint_psi_dir: Vec<[f64; 2]>,
        }

        impl PsiTraceAccum {
            pub(crate) fn new(p_resp: usize, rank: usize) -> Self {
                let projected_len = p_resp * rank;
                Self {
                    value: 0.0,
                    alpha_psi: vec![0.0; p_resp],
                    alpha_dir: vec![0.0; projected_len],
                    alpha_psi_dir: vec![0.0; projected_len],
                    h_dir: vec![0.0; rank],
                    hp_dir: vec![0.0; rank],
                    h_psi_dir: vec![0.0; rank],
                    hp_psi_dir: vec![0.0; rank],
                    endpoint_dir: vec![[0.0; 2]; rank],
                    endpoint_psi_dir: vec![[0.0; 2]; rank],
                }
            }

            pub(crate) fn merge(mut self, rhs: Self) -> Self {
                self.value += rhs.value;
                self
            }
        }

        let weights = self.effective_weights();
        let h_prime = row_quantities.h_prime.as_ref();
        let accum = gam_linalg::pairwise_reduce::par_deterministic_block_fold(
            n,
            |range| {
                let mut acc = PsiTraceAccum::new(p_resp, rank);
                for i in range {
                    let cov_row = cov.row(i);
                    let psi_row = cov_psi.row(i);
                    let rv = self.response_val_basis.row(i);
                    let rd = self.response_deriv_basis.row(i);
                    let wi = weights[i];
                    let hp = h_prime[i];
                    let inv_hp = 1.0 / hp;
                    let inv_hp_sq = inv_hp * inv_hp;
                    let q = row_quantities.endpoint_q[i];

                    for k in 0..p_resp {
                        acc.alpha_psi[k] = beta_mat.row(k).dot(&psi_row);
                    }

                    acc.alpha_dir.fill(0.0);
                    acc.alpha_psi_dir.fill(0.0);
                    for k in 0..p_resp {
                        let factor_row_base = k * p_cov;
                        let projected_base = k * rank;
                        for cidx in 0..p_cov {
                            let factor_row = factor_row_base + cidx;
                            let cov_v = cov_row[cidx];
                            let psi_v = psi_row[cidx];
                            for col in 0..rank {
                                let coeff = factor[[factor_row, col]];
                                let idx = projected_base + col;
                                acc.alpha_dir[idx] += coeff * cov_v;
                                acc.alpha_psi_dir[idx] += coeff * psi_v;
                            }
                        }
                    }

                    // `h_psi` participates only through the (zero) chart
                    // second-derivative term for the linear map.
                    let (_h_psi, hp_psi, endpoint_psi) =
                        scop_psi_marginal(rv, rd, p_resp, endpoint_basis, &acc.alpha_psi);

                    for col in 0..rank {
                        acc.h_dir[col] = 0.0;
                        acc.hp_dir[col] = 0.0;
                        acc.h_psi_dir[col] = 0.0;
                        acc.hp_psi_dir[col] = 0.0;
                        acc.endpoint_dir[col] = [0.0, 0.0];
                        acc.endpoint_psi_dir[col] = [0.0, 0.0];
                    }
                    for k in 0..p_resp {
                        for col in 0..rank {
                            let idx = k * rank + col;
                            let a_dir = acc.alpha_dir[idx];
                            let a_psi_dir = acc.alpha_psi_dir[idx];
                            acc.h_dir[col] += rv[k] * a_dir;
                            acc.hp_dir[col] += rd[k] * a_dir;
                            acc.h_psi_dir[col] += rv[k] * a_psi_dir;
                            acc.hp_psi_dir[col] += rd[k] * a_psi_dir;
                            for e in 0..2 {
                                let basis = endpoint_basis[e];
                                acc.endpoint_dir[col][e] += basis[k] * a_dir;
                                acc.endpoint_psi_dir[col][e] += basis[k] * a_psi_dir;
                            }
                        }
                    }

                    for col in 0..rank {
                        let barrier = 2.0 * acc.hp_psi_dir[col] * acc.hp_dir[col] * inv_hp_sq
                            - 2.0
                                * hp_psi
                                * acc.hp_dir[col]
                                * acc.hp_dir[col]
                                * inv_hp_sq
                                * inv_hp;
                        acc.value += wi
                            * (2.0 * acc.h_dir[col] * acc.h_psi_dir[col]
                                + barrier
                                + endpoint_chain_third(
                                    &q,
                                    endpoint_psi,
                                    acc.endpoint_dir[col],
                                    acc.endpoint_dir[col],
                                    acc.endpoint_psi_dir[col],
                                    acc.endpoint_psi_dir[col],
                                    [0.0, 0.0],
                                    [0.0, 0.0],
                                ));
                    }
                }
                acc
            },
            |left, right| left.merge(right),
        )
        .unwrap_or_else(|| PsiTraceAccum::new(p_resp, rank));

        Ok(accum.value)
    }

    /// Single-pass fused projected-trace evaluation across every ψ axis.
    ///
    /// Computes the projected trace `tr(factor^T B_e factor)` for every axis
    /// `e` in `0..cov_psi_per_axis.len()` in ONE row-streaming parallel pass.
    /// Per-row state that is independent of the ψ axis (γ, γ_dir, h_dir,
    /// hp_dir, h_vv, hp_vv, endpoint_dir, endpoint_vv) is computed exactly
    /// once per row and reused across every axis; only the ψ-row-driven
    /// axis-DEP state (γ_psi, γ_psi_dir, h_psi, hp_psi, endpoint_psi, the
    /// `*_psi_dir` / `*_psi_vv` buffers and the per-axis trace contribution)
    /// is recomputed inside the per-row axis loop.
    ///
    /// The arithmetic is reorganised but bit-identical to running
    /// [`scop_psi_hessian_trace_factor_from_cov`] once per axis: the same
    /// scalar accumulations, evaluated in the same order per row, with the
    /// rayon row reduction summing axis-INDEP contributions exactly once per
    /// row and axis-DEP contributions exactly once per `(row, axis)`.
    ///
    /// Returns a `Vec<f64>` of length `cov_psi_per_axis.len()` with the trace
    /// of each axis's projected ψ-Hessian.
    pub(crate) fn scop_psi_hessian_trace_factor_all_axes_chunk_from_cov(
        &self,
        beta: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
        row_start: usize,
        cov: ArrayView2<'_, f64>,
        cov_psi_per_axis: &[ArrayView2<'_, f64>],
        factor: ArrayView2<'_, f64>,
    ) -> Result<Vec<f64>, String> {
        let total_n = self.response_val_basis.nrows();
        let n = cov.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        let rank = factor.ncols();
        let n_psi = cov_psi_per_axis.len();
        if n_psi == 0 {
            return Ok(Vec::new());
        }
        if row_start > total_n || row_start + n > total_n {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi Hessian projected trace row window [{row_start}, {}) exceeds n={total_n}",
                row_start + n
            ) }.into());
        }
        if cov.nrows() != n || cov.ncols() != p_cov {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi Hessian projected trace covariate chunk shape {}x{} != expected {}x{}",
                cov.nrows(),
                cov.ncols(),
                n,
                p_cov
            ) }.into());
        }
        for (axis, cov_psi) in cov_psi_per_axis.iter().enumerate() {
            if cov_psi.nrows() != n || cov_psi.ncols() != p_cov {
                return Err(TransformationNormalError::InvalidInput { reason: format!(
                    "SCOP psi Hessian projected trace covariate derivative chunk shape {}x{} for axis {axis} != expected {}x{}",
                    cov_psi.nrows(),
                    cov_psi.ncols(),
                    n,
                    p_cov
                ) }.into());
            }
        }
        if beta.len() != p_total || factor.nrows() != p_total {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi Hessian projected trace length mismatch: beta={}, factor_rows={}, expected={p_total}",
                beta.len(),
                factor.nrows()
            ) }.into());
        }
        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP psi Hessian projected trace beta reshape failed: {e}"))?;
        let endpoint_basis = [
            self.response_upper_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint upper basis is not contiguous".to_string())?,
            self.response_lower_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint lower basis is not contiguous".to_string())?,
        ];

        struct PsiAllAxesTraceAccum {
            pub(crate) values: Vec<f64>,
            pub(crate) alpha_dir: Vec<f64>,
            pub(crate) h_dir: Vec<f64>,
            pub(crate) hp_dir: Vec<f64>,
            pub(crate) endpoint_dir: Vec<[f64; 2]>,
            pub(crate) alpha_psi: Vec<f64>,
            pub(crate) alpha_psi_dir: Vec<f64>,
            pub(crate) h_psi_dir: Vec<f64>,
            pub(crate) hp_psi_dir: Vec<f64>,
            pub(crate) endpoint_psi_dir: Vec<[f64; 2]>,
        }

        impl PsiAllAxesTraceAccum {
            pub(crate) fn new(p_resp: usize, rank: usize, n_psi: usize) -> Self {
                let projected_len = p_resp * rank;
                Self {
                    values: vec![0.0; n_psi],
                    alpha_dir: vec![0.0; projected_len],
                    h_dir: vec![0.0; rank],
                    hp_dir: vec![0.0; rank],
                    endpoint_dir: vec![[0.0; 2]; rank],
                    alpha_psi: vec![0.0; p_resp],
                    alpha_psi_dir: vec![0.0; projected_len],
                    h_psi_dir: vec![0.0; rank],
                    hp_psi_dir: vec![0.0; rank],
                    endpoint_psi_dir: vec![[0.0; 2]; rank],
                }
            }

            pub(crate) fn merge(mut self, rhs: Self) -> Self {
                for (a, v) in rhs.values.into_iter().enumerate() {
                    self.values[a] += v;
                }
                self
            }
        }

        let weights = self.effective_weights();
        let h_prime = row_quantities.h_prime.as_ref();
        let accum = gam_linalg::pairwise_reduce::par_deterministic_block_fold(
            n,
            |range| {
                let mut acc = PsiAllAxesTraceAccum::new(p_resp, rank, n_psi);
                for local_i in range {
                    let i = row_start + local_i;
                    let cov_row = cov.row(local_i);
                    let rv = self.response_val_basis.row(i);
                    let rd = self.response_deriv_basis.row(i);
                    let wi = weights[i];
                    let hp = h_prime[i];
                    let inv_hp = 1.0 / hp;
                    let inv_hp_sq = inv_hp * inv_hp;
                    let q = row_quantities.endpoint_q[i];

                    // ---- Axis-INDEP per-row state (computed exactly once) ----
                    acc.alpha_dir.fill(0.0);
                    for k in 0..p_resp {
                        let factor_row_base = k * p_cov;
                        let projected_base = k * rank;
                        for cidx in 0..p_cov {
                            let factor_row = factor_row_base + cidx;
                            let cov_v = cov_row[cidx];
                            for col in 0..rank {
                                let coeff = factor[[factor_row, col]];
                                let idx = projected_base + col;
                                acc.alpha_dir[idx] += coeff * cov_v;
                            }
                        }
                    }

                    for col in 0..rank {
                        acc.h_dir[col] = 0.0;
                        acc.hp_dir[col] = 0.0;
                        acc.endpoint_dir[col] = [0.0, 0.0];
                    }
                    for k in 0..p_resp {
                        for col in 0..rank {
                            let idx = k * rank + col;
                            let a_dir = acc.alpha_dir[idx];
                            acc.h_dir[col] += rv[k] * a_dir;
                            acc.hp_dir[col] += rd[k] * a_dir;
                            for e in 0..2 {
                                acc.endpoint_dir[col][e] += endpoint_basis[e][k] * a_dir;
                            }
                        }
                    }

                    // ---- Axis-DEP per-(row,axis) state ----
                    for axis_idx in 0..n_psi {
                        let psi_row = cov_psi_per_axis[axis_idx].row(local_i);

                        for k in 0..p_resp {
                            acc.alpha_psi[k] = beta_mat.row(k).dot(&psi_row);
                        }

                        acc.alpha_psi_dir.fill(0.0);
                        for k in 0..p_resp {
                            let factor_row_base = k * p_cov;
                            let projected_base = k * rank;
                            for cidx in 0..p_cov {
                                let factor_row = factor_row_base + cidx;
                                let psi_v = psi_row[cidx];
                                for col in 0..rank {
                                    let coeff = factor[[factor_row, col]];
                                    let idx = projected_base + col;
                                    acc.alpha_psi_dir[idx] += coeff * psi_v;
                                }
                            }
                        }

                        let (_h_psi, hp_psi, endpoint_psi) =
                            scop_psi_marginal(rv, rd, p_resp, endpoint_basis, &acc.alpha_psi);

                        for col in 0..rank {
                            acc.h_psi_dir[col] = 0.0;
                            acc.hp_psi_dir[col] = 0.0;
                            acc.endpoint_psi_dir[col] = [0.0, 0.0];
                        }
                        for k in 0..p_resp {
                            for col in 0..rank {
                                let idx = k * rank + col;
                                let a_psi_dir = acc.alpha_psi_dir[idx];
                                acc.h_psi_dir[col] += rv[k] * a_psi_dir;
                                acc.hp_psi_dir[col] += rd[k] * a_psi_dir;
                                for e in 0..2 {
                                    acc.endpoint_psi_dir[col][e] +=
                                        endpoint_basis[e][k] * a_psi_dir;
                                }
                            }
                        }

                        let mut axis_value = 0.0;
                        for col in 0..rank {
                            let barrier = 2.0
                                * acc.hp_psi_dir[col]
                                * acc.hp_dir[col]
                                * inv_hp_sq
                                - 2.0
                                    * hp_psi
                                    * acc.hp_dir[col]
                                    * acc.hp_dir[col]
                                    * inv_hp_sq
                                    * inv_hp;
                            axis_value += wi
                                * (2.0 * acc.h_dir[col] * acc.h_psi_dir[col]
                                    + barrier
                                    + endpoint_chain_third(
                                        &q,
                                        endpoint_psi,
                                        acc.endpoint_dir[col],
                                        acc.endpoint_dir[col],
                                        acc.endpoint_psi_dir[col],
                                        acc.endpoint_psi_dir[col],
                                        [0.0, 0.0],
                                        [0.0, 0.0],
                                    ));
                        }
                        acc.values[axis_idx] += axis_value;
                    }
                }
                acc
            },
            |left, right| left.merge(right),
        )
        .unwrap_or_else(|| PsiAllAxesTraceAccum::new(p_resp, rank, n_psi));

        Ok(accum.values)
    }

    pub(crate) fn scop_psi_psi_value_score_hvp_from_cov(
        &self,
        beta: &Array1<f64>,
        cached_gamma: ArrayView2<'_, f64>,
        cached_h: ArrayView1<'_, f64>,
        cached_h_prime: ArrayView1<'_, f64>,
        cov: ArrayView2<'_, f64>,
        cov_i: ArrayView2<'_, f64>,
        cov_j: ArrayView2<'_, f64>,
        cov_ij: ArrayView2<'_, f64>,
        row_start: usize,
        endpoint_q: &[LogNormalCdfDiffDerivatives],
        direction: Option<&Array1<f64>>,
    ) -> Result<(f64, Array1<f64>, Option<Array1<f64>>), String> {
        let total_n = self.response_val_basis.nrows();
        let n = cov.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if row_start > total_n || row_start + n > total_n {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi-psi row window [{row_start}, {}) exceeds n={total_n}",
                    row_start + n
                ),
            }
            .into());
        }
        if beta.len() != p_total {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi-psi beta length {} != p_resp({p_resp}) * p_cov({p_cov})",
                    beta.len()
                ),
            }
            .into());
        }
        if endpoint_q.len() != n {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi-psi endpoint normalizer cache length {} != n={n}",
                    endpoint_q.len()
                ),
            }
            .into());
        }
        if cached_h.len() != n || cached_h_prime.len() != n {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi-psi row-quantity cache length mismatch: h={}, h_prime={}, expected={n}",
                cached_h.len(),
                cached_h_prime.len()
            ) }.into());
        }
        if cached_gamma.nrows() != n || cached_gamma.ncols() != p_resp {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi-psi gamma cache shape {}x{} != expected {}x{}",
                    cached_gamma.nrows(),
                    cached_gamma.ncols(),
                    n,
                    p_resp
                ),
            }
            .into());
        }
        for (name, mat) in [
            ("cov", cov),
            ("cov_i", cov_i),
            ("cov_j", cov_j),
            ("cov_ij", cov_ij),
        ] {
            if mat.nrows() != n || mat.ncols() != p_cov {
                return Err(TransformationNormalError::InvalidInput {
                    reason: format!(
                        "SCOP psi-psi {name} shape {}x{} != expected {}x{}",
                        mat.nrows(),
                        mat.ncols(),
                        n,
                        p_cov
                    ),
                }
                .into());
            }
        }
        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP psi-psi beta reshape failed: {e}"))?;
        let direction_mat = match direction {
            Some(v) => {
                if v.len() != p_total {
                    return Err(TransformationNormalError::InvalidInput {
                        reason: format!(
                            "SCOP psi-psi HVP direction length {} != p_total {p_total}",
                            v.len()
                        ),
                    }
                    .into());
                }
                Some(
                    v.view()
                        .into_shape_with_order((p_resp, p_cov))
                        .map_err(|e| format!("SCOP psi-psi direction reshape failed: {e}"))?,
                )
            }
            None => None,
        };
        let endpoint_basis = [
            self.response_upper_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint upper basis is not contiguous".to_string())?,
            self.response_lower_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint lower basis is not contiguous".to_string())?,
        ];

        if direction_mat.is_none() {
            let weights = self.effective_weights();

            struct PsiPairScoreAccum {
                pub(crate) objective: f64,
                pub(crate) score: Array1<f64>,
                pub(crate) alpha_i: Vec<f64>,
                pub(crate) alpha_j: Vec<f64>,
                pub(crate) alpha_ij: Vec<f64>,
            }

            impl PsiPairScoreAccum {
                pub(crate) fn new(p_total: usize, p_resp: usize) -> Self {
                    Self {
                        objective: 0.0,
                        score: Array1::<f64>::zeros(p_total),
                        alpha_i: vec![0.0; p_resp],
                        alpha_j: vec![0.0; p_resp],
                        alpha_ij: vec![0.0; p_resp],
                    }
                }

                pub(crate) fn merge(mut self, rhs: Self) -> Self {
                    self.objective += rhs.objective;
                    self.score.scaled_add(1.0, &rhs.score);
                    self
                }
            }

            let accum = gam_linalg::pairwise_reduce::par_deterministic_block_fold(
                n,
                |range| {
                    let mut acc = PsiPairScoreAccum::new(p_total, p_resp);
                    for row_idx in range {
                        let cov_row = cov.row(row_idx);
                        let cov_i_row = cov_i.row(row_idx);
                        let cov_j_row = cov_j.row(row_idx);
                        let cov_ij_row = cov_ij.row(row_idx);
                        let global_row = row_start + row_idx;
                        let rv = self.response_val_basis.row(global_row);
                        let rd = self.response_deriv_basis.row(global_row);

                        for k in 0..p_resp {
                            let beta_k = beta_mat.row(k);
                            acc.alpha_i[k] = beta_k.dot(&cov_i_row);
                            acc.alpha_j[k] = beta_k.dot(&cov_j_row);
                            acc.alpha_ij[k] = beta_k.dot(&cov_ij_row);
                        }

                        let h = cached_h[row_idx];
                        let hp = cached_h_prime[row_idx];
                        let [h_i, h_j, h_ij, hp_i, hp_j, hp_ij] = scop_second_order_h(
                            rv,
                            rd,
                            p_resp,
                            &acc.alpha_i,
                            &acc.alpha_j,
                            &acc.alpha_ij,
                        );

                        let inv_hp = 1.0 / hp;
                        let inv_hp_sq = inv_hp * inv_hp;
                        let inv_hp_cu = inv_hp_sq * inv_hp;
                        let q = endpoint_q[row_idx];
                        let (endpoint_i, endpoint_j, endpoint_ij) = scop_second_order_endpoints(
                            endpoint_basis,
                            p_resp,
                            &acc.alpha_i,
                            &acc.alpha_j,
                            &acc.alpha_ij,
                        );
                        let value = h_i * h_j + h * h_ij - hp_ij * inv_hp
                            + hp_i * hp_j * inv_hp_sq
                            + endpoint_chain_second(&q, endpoint_i, endpoint_j, endpoint_ij);
                        let wi = weights[global_row];
                        acc.objective += wi * value;

                        for k in 0..p_resp {
                            let offset = k * p_cov;
                            let (rvk, rdk) = (rv[k], rd[k]);
                            let e_k = [endpoint_basis[0][k], endpoint_basis[1][k]];
                            for cidx in 0..p_cov {
                                let c = cov_row[cidx];
                                let ci = cov_i_row[cidx];
                                let cj = cov_j_row[cidx];
                                let cij = cov_ij_row[cidx];
                                // Linear chart: every coefficient factor is the
                                // basis value times the matching (deformed)
                                // design entry; no coordinate chain terms.
                                let dh = rvk * c;
                                let dhp = rdk * c;
                                let dh_i = rvk * ci;
                                let dh_j = rvk * cj;
                                let dh_ij = rvk * cij;
                                let dhp_i = rdk * ci;
                                let dhp_j = rdk * cj;
                                let dhp_ij = rdk * cij;
                                let endpoint_a = [e_k[0] * c, e_k[1] * c];
                                let endpoint_i_a = [e_k[0] * ci, e_k[1] * ci];
                                let endpoint_j_a = [e_k[0] * cj, e_k[1] * cj];
                                let endpoint_ij_a = [e_k[0] * cij, e_k[1] * cij];
                                let grad = dh_i * h_j + h_i * dh_j + dh * h_ij + h * dh_ij
                                    - dhp_ij * inv_hp
                                    + hp_ij * dhp * inv_hp_sq
                                    + (dhp_i * hp_j + hp_i * dhp_j) * inv_hp_sq
                                    - 2.0 * hp_i * hp_j * dhp * inv_hp_cu
                                    + endpoint_chain_third(
                                        &q,
                                        endpoint_i,
                                        endpoint_j,
                                        endpoint_a,
                                        endpoint_ij,
                                        endpoint_i_a,
                                        endpoint_j_a,
                                        endpoint_ij_a,
                                    );
                                acc.score[offset + cidx] += wi * grad;
                            }
                        }
                    }
                    acc
                },
                |left, right| left.merge(right),
            )
            .unwrap_or_else(|| PsiPairScoreAccum::new(p_total, p_resp));

            return Ok((accum.objective, accum.score, None));
        }

        let weights = self.effective_weights();
        let direction_mat = direction_mat.expect("directional CTN psi-psi path requires direction");

        struct PsiPairDirectionalAccum {
            pub(crate) hvp: Array1<f64>,
            pub(crate) alpha_i: Vec<f64>,
            pub(crate) alpha_j: Vec<f64>,
            pub(crate) alpha_ij: Vec<f64>,
            pub(crate) alpha_dot: Vec<f64>,
            pub(crate) alpha_i_dot: Vec<f64>,
            pub(crate) alpha_j_dot: Vec<f64>,
            pub(crate) alpha_ij_dot: Vec<f64>,
        }

        impl PsiPairDirectionalAccum {
            pub(crate) fn new(p_total: usize, p_resp: usize) -> Self {
                Self {
                    hvp: Array1::<f64>::zeros(p_total),
                    alpha_i: vec![0.0; p_resp],
                    alpha_j: vec![0.0; p_resp],
                    alpha_ij: vec![0.0; p_resp],
                    alpha_dot: vec![0.0; p_resp],
                    alpha_i_dot: vec![0.0; p_resp],
                    alpha_j_dot: vec![0.0; p_resp],
                    alpha_ij_dot: vec![0.0; p_resp],
                }
            }

            pub(crate) fn merge(mut self, rhs: Self) -> Self {
                self.hvp.scaled_add(1.0, &rhs.hvp);
                self
            }
        }

        let accum = gam_linalg::pairwise_reduce::par_deterministic_block_fold(
            n,
            |range| {
                let mut acc = PsiPairDirectionalAccum::new(p_total, p_resp);
                for row_idx in range {
                    let cov_row = cov.row(row_idx);
                    let cov_i_row = cov_i.row(row_idx);
                    let cov_j_row = cov_j.row(row_idx);
                    let cov_ij_row = cov_ij.row(row_idx);
                    let global_row = row_start + row_idx;
                    let rv = self.response_val_basis.row(global_row);
                    let rd = self.response_deriv_basis.row(global_row);

                    for k in 0..p_resp {
                        let beta_k = beta_mat.row(k);
                        let dir_k = direction_mat.row(k);
                        acc.alpha_i[k] = beta_k.dot(&cov_i_row);
                        acc.alpha_j[k] = beta_k.dot(&cov_j_row);
                        acc.alpha_ij[k] = beta_k.dot(&cov_ij_row);
                        acc.alpha_dot[k] = dir_k.dot(&cov_row);
                        acc.alpha_i_dot[k] = dir_k.dot(&cov_i_row);
                        acc.alpha_j_dot[k] = dir_k.dot(&cov_j_row);
                        acc.alpha_ij_dot[k] = dir_k.dot(&cov_ij_row);
                    }

                    let hp = cached_h_prime[row_idx];
                    // Linear chart: every transform functional is a plain
                    // basis-weighted sum of the matching coordinate slots.
                    // The value-level `h`, `h_i`, `h_j`, `h_ij` drop out of
                    // the directional expression entirely because every
                    // second-β chart derivative is zero.
                    let mut hp_i = 0.0;
                    let mut hp_j = 0.0;
                    let mut hp_ij = 0.0;
                    let mut h_dot = 0.0;
                    let mut hp_dot = 0.0;
                    let mut h_i_dot = 0.0;
                    let mut h_j_dot = 0.0;
                    let mut h_ij_dot = 0.0;
                    let mut hp_i_dot = 0.0;
                    let mut hp_j_dot = 0.0;
                    let mut hp_ij_dot = 0.0;
                    for k in 0..p_resp {
                        hp_i += rd[k] * acc.alpha_i[k];
                        hp_j += rd[k] * acc.alpha_j[k];
                        hp_ij += rd[k] * acc.alpha_ij[k];
                        h_dot += rv[k] * acc.alpha_dot[k];
                        hp_dot += rd[k] * acc.alpha_dot[k];
                        h_i_dot += rv[k] * acc.alpha_i_dot[k];
                        h_j_dot += rv[k] * acc.alpha_j_dot[k];
                        h_ij_dot += rv[k] * acc.alpha_ij_dot[k];
                        hp_i_dot += rd[k] * acc.alpha_i_dot[k];
                        hp_j_dot += rd[k] * acc.alpha_j_dot[k];
                        hp_ij_dot += rd[k] * acc.alpha_ij_dot[k];
                    }

                    let inv_hp = 1.0 / hp;
                    let inv_hp_sq = inv_hp * inv_hp;
                    let inv_hp_cu = inv_hp_sq * inv_hp;
                    let inv_hp_qu = inv_hp_sq * inv_hp_sq;
                    let wi = weights[global_row];
                    let q = endpoint_q[row_idx];
                    let mut endpoint_i = [0.0; 2];
                    let mut endpoint_j = [0.0; 2];
                    let mut endpoint_ij = [0.0; 2];
                    let mut endpoint_d = [0.0; 2];
                    let mut endpoint_i_d = [0.0; 2];
                    let mut endpoint_j_d = [0.0; 2];
                    let mut endpoint_ij_d = [0.0; 2];
                    for e in 0..2 {
                        let basis = endpoint_basis[e];
                        for k in 0..p_resp {
                            endpoint_i[e] += basis[k] * acc.alpha_i[k];
                            endpoint_j[e] += basis[k] * acc.alpha_j[k];
                            endpoint_ij[e] += basis[k] * acc.alpha_ij[k];
                            endpoint_d[e] += basis[k] * acc.alpha_dot[k];
                            endpoint_i_d[e] += basis[k] * acc.alpha_i_dot[k];
                            endpoint_j_d[e] += basis[k] * acc.alpha_j_dot[k];
                            endpoint_ij_d[e] += basis[k] * acc.alpha_ij_dot[k];
                        }
                    }

                    for k in 0..p_resp {
                        let offset = k * p_cov;
                        let (rvk, rdk) = (rv[k], rd[k]);
                        let e_k = [endpoint_basis[0][k], endpoint_basis[1][k]];
                        for cidx in 0..p_cov {
                            let c = cov_row[cidx];
                            let ci = cov_i_row[cidx];
                            let cj = cov_j_row[cidx];
                            let cij = cov_ij_row[cidx];
                            // Constant coefficient factors; all second-β
                            // (`dd*`) factors vanish for the linear chart.
                            let dh = rvk * c;
                            let dhp = rdk * c;
                            let dh_i = rvk * ci;
                            let dh_j = rvk * cj;
                            let dh_ij = rvk * cij;
                            let dhp_i = rdk * ci;
                            let dhp_j = rdk * cj;
                            let dhp_ij = rdk * cij;
                            let endpoint_a = [e_k[0] * c, e_k[1] * c];
                            let endpoint_i_a = [e_k[0] * ci, e_k[1] * ci];
                            let endpoint_j_a = [e_k[0] * cj, e_k[1] * cj];
                            let endpoint_ij_a = [e_k[0] * cij, e_k[1] * cij];
                            let n1 = dhp_i * hp_j + hp_i * dhp_j;
                            let n1_dot = dhp_i * hp_j_dot + hp_i_dot * dhp_j;
                            let n2_dot =
                                hp_i_dot * hp_j * dhp + hp_i * hp_j_dot * dhp;
                            let hv = dh_i * h_j_dot
                                + h_i_dot * dh_j
                                + dh * h_ij_dot
                                + h_dot * dh_ij
                                + dhp_ij * hp_dot * inv_hp_sq
                                + hp_ij_dot * dhp * inv_hp_sq
                                - 2.0 * hp_ij * dhp * hp_dot * inv_hp_cu
                                + n1_dot * inv_hp_sq
                                - 2.0 * n1 * hp_dot * inv_hp_cu
                                - 2.0 * n2_dot * inv_hp_cu
                                + 6.0 * hp_i * hp_j * dhp * hp_dot * inv_hp_qu
                                + endpoint_chain_fourth(
                                    &q,
                                    endpoint_i,
                                    endpoint_j,
                                    endpoint_a,
                                    endpoint_d,
                                    endpoint_ij,
                                    endpoint_i_a,
                                    endpoint_i_d,
                                    endpoint_j_a,
                                    endpoint_j_d,
                                    [0.0; 2],
                                    endpoint_ij_a,
                                    endpoint_ij_d,
                                    [0.0; 2],
                                    [0.0; 2],
                                    [0.0; 2],
                                );
                            acc.hvp[offset + cidx] += wi * hv;
                        }
                    }
                }
                acc
            },
            |left, right| left.merge(right),
        )
        .unwrap_or_else(|| PsiPairDirectionalAccum::new(p_total, p_resp));

        Ok((0.0, Array1::<f64>::zeros(p_total), Some(accum.hvp)))
    }

    pub(crate) fn scop_psi_psi_hvp_mat_from_cov(
        &self,
        beta: &Array1<f64>,
        cached_gamma: ArrayView2<'_, f64>,
        cached_h: ArrayView1<'_, f64>,
        cached_h_prime: ArrayView1<'_, f64>,
        cov: ArrayView2<'_, f64>,
        cov_i: ArrayView2<'_, f64>,
        cov_j: ArrayView2<'_, f64>,
        cov_ij: ArrayView2<'_, f64>,
        row_start: usize,
        endpoint_q: &[LogNormalCdfDiffDerivatives],
        factor: ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        let total_n = self.response_val_basis.nrows();
        let n = cov.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        let rank = factor.ncols();
        if row_start > total_n || row_start + n > total_n {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi-psi batched HVP row window [{row_start}, {}) exceeds n={total_n}",
                    row_start + n
                ),
            }
            .into());
        }
        if beta.len() != p_total || factor.nrows() != p_total {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi-psi batched HVP length mismatch: beta={}, factor_rows={}, expected={p_total}",
                beta.len(),
                factor.nrows()
            ) }.into());
        }
        if endpoint_q.len() != n {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi-psi batched HVP endpoint normalizer cache length {} != n={n}",
                    endpoint_q.len()
                ),
            }
            .into());
        }
        if cached_h.len() != n || cached_h_prime.len() != n {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi-psi batched HVP row-quantity cache length mismatch: h={}, h_prime={}, expected={n}",
                cached_h.len(),
                cached_h_prime.len()
            ) }.into());
        }
        if cached_gamma.nrows() != n || cached_gamma.ncols() != p_resp {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi-psi batched HVP gamma cache shape {}x{} != expected {}x{}",
                    cached_gamma.nrows(),
                    cached_gamma.ncols(),
                    n,
                    p_resp
                ),
            }
            .into());
        }
        for (name, mat) in [
            ("cov", cov),
            ("cov_i", cov_i),
            ("cov_j", cov_j),
            ("cov_ij", cov_ij),
        ] {
            if mat.nrows() != n || mat.ncols() != p_cov {
                return Err(TransformationNormalError::InvalidInput {
                    reason: format!(
                        "SCOP psi-psi batched HVP {name} shape {}x{} != expected {}x{}",
                        mat.nrows(),
                        mat.ncols(),
                        n,
                        p_cov
                    ),
                }
                .into());
            }
        }

        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP psi-psi batched HVP beta reshape failed: {e}"))?;
        let endpoint_basis = [
            self.response_upper_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint upper basis is not contiguous".to_string())?,
            self.response_lower_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint lower basis is not contiguous".to_string())?,
        ];

        struct PsiPairBatchedAccum {
            pub(crate) hvp: Array2<f64>,
            pub(crate) gamma: Vec<f64>,
            pub(crate) gamma_i: Vec<f64>,
            pub(crate) gamma_j: Vec<f64>,
            pub(crate) gamma_ij: Vec<f64>,
            pub(crate) gamma_dot: Vec<f64>,
            pub(crate) gamma_i_dot: Vec<f64>,
            pub(crate) gamma_j_dot: Vec<f64>,
            pub(crate) gamma_ij_dot: Vec<f64>,
        }

        impl PsiPairBatchedAccum {
            pub(crate) fn new(p_total: usize, p_resp: usize, rank: usize) -> Self {
                let projected_len = p_resp * rank;
                Self {
                    hvp: Array2::<f64>::zeros((p_total, rank)),
                    gamma: vec![0.0; p_resp],
                    gamma_i: vec![0.0; p_resp],
                    gamma_j: vec![0.0; p_resp],
                    gamma_ij: vec![0.0; p_resp],
                    gamma_dot: vec![0.0; projected_len],
                    gamma_i_dot: vec![0.0; projected_len],
                    gamma_j_dot: vec![0.0; projected_len],
                    gamma_ij_dot: vec![0.0; projected_len],
                }
            }

            pub(crate) fn merge(mut self, rhs: Self) -> Self {
                self.hvp += &rhs.hvp;
                self
            }
        }

        let weights = self.effective_weights();
        let accum = gam_linalg::pairwise_reduce::par_deterministic_block_fold(
            n,
            |range| {
                let mut acc = PsiPairBatchedAccum::new(p_total, p_resp, rank);
                for row_idx in range {
                    let cov_row = cov.row(row_idx);
                    let cov_i_row = cov_i.row(row_idx);
                    let cov_j_row = cov_j.row(row_idx);
                    let cov_ij_row = cov_ij.row(row_idx);
                    let global_row = row_start + row_idx;
                    let rv = self.response_val_basis.row(global_row);
                    let rd = self.response_deriv_basis.row(global_row);
                    let gamma_row = cached_gamma.row(row_idx);

                    for k in 0..p_resp {
                        let beta_k = beta_mat.row(k);
                        acc.gamma[k] = gamma_row[k];
                        acc.gamma_i[k] = beta_k.dot(&cov_i_row);
                        acc.gamma_j[k] = beta_k.dot(&cov_j_row);
                        acc.gamma_ij[k] = beta_k.dot(&cov_ij_row);
                    }

                    let h = cached_h[row_idx];
                    let hp = cached_h_prime[row_idx];
                    let [h_i, h_j, h_ij, hp_i, hp_j, hp_ij] = scop_second_order_h(
                        rv,
                        rd,
                        p_resp,
                        &acc.gamma,
                        &acc.gamma_i,
                        &acc.gamma_j,
                        &acc.gamma_ij,
                    );

                    let inv_hp = 1.0 / hp;
                    let inv_hp_sq = inv_hp * inv_hp;
                    let inv_hp_cu = inv_hp_sq * inv_hp;
                    let inv_hp_qu = inv_hp_sq * inv_hp_sq;
                    let wi = weights[global_row];
                    let q = endpoint_q[row_idx];
                    let (endpoint_i, endpoint_j, endpoint_ij) = scop_second_order_endpoints(
                        endpoint_basis,
                        p_resp,
                        &acc.gamma,
                        &acc.gamma_i,
                        &acc.gamma_j,
                        &acc.gamma_ij,
                    );

                    acc.gamma_dot.fill(0.0);
                    acc.gamma_i_dot.fill(0.0);
                    acc.gamma_j_dot.fill(0.0);
                    acc.gamma_ij_dot.fill(0.0);
                    for k in 0..p_resp {
                        let factor_row_base = k * p_cov;
                        let projected_base = k * rank;
                        for cidx in 0..p_cov {
                            let factor_row = factor_row_base + cidx;
                            let cov_v = cov_row[cidx];
                            let cov_i_v = cov_i_row[cidx];
                            let cov_j_v = cov_j_row[cidx];
                            let cov_ij_v = cov_ij_row[cidx];
                            for col in 0..rank {
                                let coeff = factor[[factor_row, col]];
                                let idx = projected_base + col;
                                acc.gamma_dot[idx] += coeff * cov_v;
                                acc.gamma_i_dot[idx] += coeff * cov_i_v;
                                acc.gamma_j_dot[idx] += coeff * cov_j_v;
                                acc.gamma_ij_dot[idx] += coeff * cov_ij_v;
                            }
                        }
                    }

                    for col in 0..rank {
                        let mut h_dot = rv[0] * acc.gamma_dot[col];
                        let mut hp_dot = rd[0] * acc.gamma_dot[col];
                        let mut h_i_dot = rv[0] * acc.gamma_i_dot[col];
                        let mut h_j_dot = rv[0] * acc.gamma_j_dot[col];
                        let mut h_ij_dot = rv[0] * acc.gamma_ij_dot[col];
                        let mut hp_i_dot = rd[0] * acc.gamma_i_dot[col];
                        let mut hp_j_dot = rd[0] * acc.gamma_j_dot[col];
                        let mut hp_ij_dot = rd[0] * acc.gamma_ij_dot[col];
                        for k in 1..p_resp {
                            let idx = k * rank + col;
                            let g = acc.gamma[k];
                            let gi = acc.gamma_i[k];
                            let gj = acc.gamma_j[k];
                            let gij = acc.gamma_ij[k];
                            let u = acc.gamma_dot[idx];
                            let ui = acc.gamma_i_dot[idx];
                            let uj = acc.gamma_j_dot[idx];
                            let uij = acc.gamma_ij_dot[idx];
                            h_dot += 2.0 * rv[k] * g * u;
                            hp_dot += 2.0 * rd[k] * g * u;
                            h_i_dot += 2.0 * rv[k] * (u * gi + g * ui);
                            h_j_dot += 2.0 * rv[k] * (u * gj + g * uj);
                            h_ij_dot += 2.0 * rv[k] * (uj * gi + gj * ui + u * gij + g * uij);
                            hp_i_dot += 2.0 * rd[k] * (u * gi + g * ui);
                            hp_j_dot += 2.0 * rd[k] * (u * gj + g * uj);
                            hp_ij_dot += 2.0 * rd[k] * (uj * gi + gj * ui + u * gij + g * uij);
                        }

                        let mut endpoint_d = [0.0; 2];
                        let mut endpoint_i_d = [0.0; 2];
                        let mut endpoint_j_d = [0.0; 2];
                        let mut endpoint_ij_d = [0.0; 2];
                        for e in 0..2 {
                            let basis = endpoint_basis[e];
                            endpoint_d[e] = basis[0] * acc.gamma_dot[col];
                            endpoint_i_d[e] = basis[0] * acc.gamma_i_dot[col];
                            endpoint_j_d[e] = basis[0] * acc.gamma_j_dot[col];
                            endpoint_ij_d[e] = basis[0] * acc.gamma_ij_dot[col];
                            for k in 1..p_resp {
                                let idx = k * rank + col;
                                endpoint_d[e] += 2.0 * basis[k] * acc.gamma[k] * acc.gamma_dot[idx];
                                endpoint_i_d[e] += 2.0
                                    * basis[k]
                                    * (acc.gamma_dot[idx] * acc.gamma_i[k]
                                        + acc.gamma[k] * acc.gamma_i_dot[idx]);
                                endpoint_j_d[e] += 2.0
                                    * basis[k]
                                    * (acc.gamma_dot[idx] * acc.gamma_j[k]
                                        + acc.gamma[k] * acc.gamma_j_dot[idx]);
                                endpoint_ij_d[e] += 2.0
                                    * basis[k]
                                    * (acc.gamma_j_dot[idx] * acc.gamma_i[k]
                                        + acc.gamma_j[k] * acc.gamma_i_dot[idx]
                                        + acc.gamma_dot[idx] * acc.gamma_ij[k]
                                        + acc.gamma[k] * acc.gamma_ij_dot[idx]);
                            }
                        }

                        for k in 0..p_resp {
                            let offset = k * p_cov;
                            let (rvk, rdk) = (rv[k], rd[k]);
                            let (g, gi, gj, gij) = (
                                acc.gamma[k],
                                acc.gamma_i[k],
                                acc.gamma_j[k],
                                acc.gamma_ij[k],
                            );
                            let (u, ui, uj, uij) = (
                                acc.gamma_dot[k * rank + col],
                                acc.gamma_i_dot[k * rank + col],
                                acc.gamma_j_dot[k * rank + col],
                                acc.gamma_ij_dot[k * rank + col],
                            );
                            for cidx in 0..p_cov {
                                let c = cov_row[cidx];
                                let ci = cov_i_row[cidx];
                                let cj = cov_j_row[cidx];
                                let cij = cov_ij_row[cidx];
                                let (
                                    dh,
                                    dhp,
                                    dh_i,
                                    dh_j,
                                    dh_ij,
                                    dhp_i,
                                    dhp_j,
                                    dhp_ij,
                                    ddh,
                                    ddhp,
                                    ddh_i,
                                    ddh_j,
                                    ddh_ij,
                                    ddhp_i,
                                    ddhp_j,
                                    ddhp_ij,
                                ) = if k == 0 {
                                    (
                                        rvk * c,
                                        rdk * c,
                                        rvk * ci,
                                        rvk * cj,
                                        rvk * cij,
                                        rdk * ci,
                                        rdk * cj,
                                        rdk * cij,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                    )
                                } else {
                                    (
                                        2.0 * rvk * g * c,
                                        2.0 * rdk * g * c,
                                        2.0 * rvk * (gi * c + g * ci),
                                        2.0 * rvk * (gj * c + g * cj),
                                        2.0 * rvk * (gj * ci + gi * cj + gij * c + g * cij),
                                        2.0 * rdk * (gi * c + g * ci),
                                        2.0 * rdk * (gj * c + g * cj),
                                        2.0 * rdk * (gj * ci + gi * cj + gij * c + g * cij),
                                        2.0 * rvk * u * c,
                                        2.0 * rdk * u * c,
                                        2.0 * rvk * (ui * c + u * ci),
                                        2.0 * rvk * (uj * c + u * cj),
                                        2.0 * rvk * (uj * ci + ui * cj + uij * c + u * cij),
                                        2.0 * rdk * (ui * c + u * ci),
                                        2.0 * rdk * (uj * c + u * cj),
                                        2.0 * rdk * (uj * ci + ui * cj + uij * c + u * cij),
                                    )
                                };

                                let endpoint_a = if k == 0 {
                                    [endpoint_basis[0][k] * c, endpoint_basis[1][k] * c]
                                } else {
                                    [
                                        2.0 * endpoint_basis[0][k] * g * c,
                                        2.0 * endpoint_basis[1][k] * g * c,
                                    ]
                                };
                                let endpoint_i_a = if k == 0 {
                                    [endpoint_basis[0][k] * ci, endpoint_basis[1][k] * ci]
                                } else {
                                    [
                                        2.0 * endpoint_basis[0][k] * (gi * c + g * ci),
                                        2.0 * endpoint_basis[1][k] * (gi * c + g * ci),
                                    ]
                                };
                                let endpoint_j_a = if k == 0 {
                                    [endpoint_basis[0][k] * cj, endpoint_basis[1][k] * cj]
                                } else {
                                    [
                                        2.0 * endpoint_basis[0][k] * (gj * c + g * cj),
                                        2.0 * endpoint_basis[1][k] * (gj * c + g * cj),
                                    ]
                                };
                                let endpoint_ij_a = if k == 0 {
                                    [endpoint_basis[0][k] * cij, endpoint_basis[1][k] * cij]
                                } else {
                                    [
                                        2.0 * endpoint_basis[0][k]
                                            * (gj * ci + gi * cj + gij * c + g * cij),
                                        2.0 * endpoint_basis[1][k]
                                            * (gj * ci + gi * cj + gij * c + g * cij),
                                    ]
                                };
                                let endpoint_a_d = if k == 0 {
                                    [0.0; 2]
                                } else {
                                    [
                                        2.0 * endpoint_basis[0][k] * u * c,
                                        2.0 * endpoint_basis[1][k] * u * c,
                                    ]
                                };
                                let endpoint_i_a_d = if k == 0 {
                                    [0.0; 2]
                                } else {
                                    [
                                        2.0 * endpoint_basis[0][k] * (ui * c + u * ci),
                                        2.0 * endpoint_basis[1][k] * (ui * c + u * ci),
                                    ]
                                };
                                let endpoint_j_a_d = if k == 0 {
                                    [0.0; 2]
                                } else {
                                    [
                                        2.0 * endpoint_basis[0][k] * (uj * c + u * cj),
                                        2.0 * endpoint_basis[1][k] * (uj * c + u * cj),
                                    ]
                                };
                                let endpoint_ij_a_d = if k == 0 {
                                    [0.0; 2]
                                } else {
                                    [
                                        2.0 * endpoint_basis[0][k]
                                            * (uj * ci + ui * cj + uij * c + u * cij),
                                        2.0 * endpoint_basis[1][k]
                                            * (uj * ci + ui * cj + uij * c + u * cij),
                                    ]
                                };
                                let n1 = dhp_i * hp_j + hp_i * dhp_j;
                                let n1_dot = ddhp_i * hp_j
                                    + dhp_i * hp_j_dot
                                    + hp_i_dot * dhp_j
                                    + hp_i * ddhp_j;
                                let n2_dot = hp_i_dot * hp_j * dhp
                                    + hp_i * hp_j_dot * dhp
                                    + hp_i * hp_j * ddhp;
                                let hv = ddh_i * h_j
                                    + dh_i * h_j_dot
                                    + h_i_dot * dh_j
                                    + h_i * ddh_j
                                    + ddh * h_ij
                                    + dh * h_ij_dot
                                    + h_dot * dh_ij
                                    + h * ddh_ij
                                    - ddhp_ij * inv_hp
                                    + dhp_ij * hp_dot * inv_hp_sq
                                    + hp_ij_dot * dhp * inv_hp_sq
                                    + hp_ij * ddhp * inv_hp_sq
                                    - 2.0 * hp_ij * dhp * hp_dot * inv_hp_cu
                                    + n1_dot * inv_hp_sq
                                    - 2.0 * n1 * hp_dot * inv_hp_cu
                                    - 2.0 * n2_dot * inv_hp_cu
                                    + 6.0 * hp_i * hp_j * dhp * hp_dot * inv_hp_qu
                                    + endpoint_chain_fourth(
                                        &q,
                                        endpoint_i,
                                        endpoint_j,
                                        endpoint_a,
                                        endpoint_d,
                                        endpoint_ij,
                                        endpoint_i_a,
                                        endpoint_i_d,
                                        endpoint_j_a,
                                        endpoint_j_d,
                                        endpoint_a_d,
                                        endpoint_ij_a,
                                        endpoint_ij_d,
                                        endpoint_i_a_d,
                                        endpoint_j_a_d,
                                        endpoint_ij_a_d,
                                    );
                                acc.hvp[[offset + cidx, col]] += wi * hv;
                            }
                        }
                    }
                }
                acc
            },
            |left, right| left.merge(right),
        )
        .unwrap_or_else(|| PsiPairBatchedAccum::new(p_total, p_resp, rank));

        Ok(accum.hvp)
    }

    pub(crate) fn scop_psi_psi_bilinear_from_cov(
        &self,
        beta: &Array1<f64>,
        cached_gamma: ArrayView2<'_, f64>,
        cached_h: ArrayView1<'_, f64>,
        cached_h_prime: ArrayView1<'_, f64>,
        cov: ArrayView2<'_, f64>,
        cov_i: ArrayView2<'_, f64>,
        cov_j: ArrayView2<'_, f64>,
        cov_ij: ArrayView2<'_, f64>,
        row_start: usize,
        endpoint_q: &[LogNormalCdfDiffDerivatives],
        left: ArrayView1<'_, f64>,
        right: ArrayView1<'_, f64>,
    ) -> Result<f64, String> {
        let total_n = self.response_val_basis.nrows();
        let n = cov.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if row_start > total_n || row_start + n > total_n {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi-psi bilinear row window [{row_start}, {}) exceeds n={total_n}",
                    row_start + n
                ),
            }
            .into());
        }
        if beta.len() != p_total || left.len() != p_total || right.len() != p_total {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi-psi bilinear length mismatch: beta={}, left={}, right={}, expected={p_total}",
                beta.len(),
                left.len(),
                right.len()
            ) }.into());
        }
        if endpoint_q.len() != n {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi-psi bilinear endpoint normalizer cache length {} != n={n}",
                    endpoint_q.len()
                ),
            }
            .into());
        }
        if cached_h.len() != n || cached_h_prime.len() != n {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi-psi bilinear row-quantity cache length mismatch: h={}, h_prime={}, expected={n}",
                cached_h.len(),
                cached_h_prime.len()
            ) }.into());
        }
        if cached_gamma.nrows() != n || cached_gamma.ncols() != p_resp {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi-psi bilinear gamma cache shape {}x{} != expected {}x{}",
                    cached_gamma.nrows(),
                    cached_gamma.ncols(),
                    n,
                    p_resp
                ),
            }
            .into());
        }
        for (name, mat) in [
            ("cov", cov),
            ("cov_i", cov_i),
            ("cov_j", cov_j),
            ("cov_ij", cov_ij),
        ] {
            if mat.nrows() != n || mat.ncols() != p_cov {
                return Err(TransformationNormalError::InvalidInput {
                    reason: format!(
                        "SCOP psi-psi bilinear {name} shape {}x{} != expected {}x{}",
                        mat.nrows(),
                        mat.ncols(),
                        n,
                        p_cov
                    ),
                }
                .into());
            }
        }
        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP psi-psi bilinear beta reshape failed: {e}"))?;
        let left_mat = left
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP psi-psi bilinear left reshape failed: {e}"))?;
        let right_mat = right
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP psi-psi bilinear right reshape failed: {e}"))?;
        let endpoint_basis = [
            self.response_upper_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint upper basis is not contiguous".to_string())?,
            self.response_lower_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint lower basis is not contiguous".to_string())?,
        ];

        struct PsiPairBilinearAccum {
            pub(crate) value: f64,
            pub(crate) gamma: Vec<f64>,
            pub(crate) gamma_i: Vec<f64>,
            pub(crate) gamma_j: Vec<f64>,
            pub(crate) gamma_ij: Vec<f64>,
            pub(crate) left: Vec<f64>,
            pub(crate) left_i: Vec<f64>,
            pub(crate) left_j: Vec<f64>,
            pub(crate) left_ij: Vec<f64>,
            pub(crate) right: Vec<f64>,
            pub(crate) right_i: Vec<f64>,
            pub(crate) right_j: Vec<f64>,
            pub(crate) right_ij: Vec<f64>,
        }

        impl PsiPairBilinearAccum {
            pub(crate) fn new(p_resp: usize) -> Self {
                Self {
                    value: 0.0,
                    gamma: vec![0.0; p_resp],
                    gamma_i: vec![0.0; p_resp],
                    gamma_j: vec![0.0; p_resp],
                    gamma_ij: vec![0.0; p_resp],
                    left: vec![0.0; p_resp],
                    left_i: vec![0.0; p_resp],
                    left_j: vec![0.0; p_resp],
                    left_ij: vec![0.0; p_resp],
                    right: vec![0.0; p_resp],
                    right_i: vec![0.0; p_resp],
                    right_j: vec![0.0; p_resp],
                    right_ij: vec![0.0; p_resp],
                }
            }
        }

        let weights = self.effective_weights();
        let total = gam_linalg::pairwise_reduce::par_deterministic_block_fold(
            n,
            |range| {
                let mut acc = PsiPairBilinearAccum::new(p_resp);
                for row_idx in range {
                    let cov_row = cov.row(row_idx);
                    let cov_i_row = cov_i.row(row_idx);
                    let cov_j_row = cov_j.row(row_idx);
                    let cov_ij_row = cov_ij.row(row_idx);
                    let global_row = row_start + row_idx;
                    let rv = self.response_val_basis.row(global_row);
                    let rd = self.response_deriv_basis.row(global_row);
                    let gamma_row = cached_gamma.row(row_idx);

                    for k in 0..p_resp {
                        let beta_k = beta_mat.row(k);
                        let left_k = left_mat.row(k);
                        let right_k = right_mat.row(k);
                        acc.gamma[k] = gamma_row[k];
                        acc.gamma_i[k] = beta_k.dot(&cov_i_row);
                        acc.gamma_j[k] = beta_k.dot(&cov_j_row);
                        acc.gamma_ij[k] = beta_k.dot(&cov_ij_row);
                        acc.left[k] = left_k.dot(&cov_row);
                        acc.left_i[k] = left_k.dot(&cov_i_row);
                        acc.left_j[k] = left_k.dot(&cov_j_row);
                        acc.left_ij[k] = left_k.dot(&cov_ij_row);
                        acc.right[k] = right_k.dot(&cov_row);
                        acc.right_i[k] = right_k.dot(&cov_i_row);
                        acc.right_j[k] = right_k.dot(&cov_j_row);
                        acc.right_ij[k] = right_k.dot(&cov_ij_row);
                    }

                    let h = cached_h[row_idx];
                    let hp = cached_h_prime[row_idx];
                    let mut h_i = rv[0] * acc.gamma_i[0];
                    let mut h_j = rv[0] * acc.gamma_j[0];
                    let mut h_ij = rv[0] * acc.gamma_ij[0];
                    let mut hp_i = rd[0] * acc.gamma_i[0];
                    let mut hp_j = rd[0] * acc.gamma_j[0];
                    let mut hp_ij = rd[0] * acc.gamma_ij[0];

                    let mut h_l = rv[0] * acc.left[0];
                    let mut hp_l = rd[0] * acc.left[0];
                    let mut h_i_l = rv[0] * acc.left_i[0];
                    let mut h_j_l = rv[0] * acc.left_j[0];
                    let mut h_ij_l = rv[0] * acc.left_ij[0];
                    let mut hp_i_l = rd[0] * acc.left_i[0];
                    let mut hp_j_l = rd[0] * acc.left_j[0];
                    let mut hp_ij_l = rd[0] * acc.left_ij[0];

                    let mut h_r = rv[0] * acc.right[0];
                    let mut hp_r = rd[0] * acc.right[0];
                    let mut h_i_r = rv[0] * acc.right_i[0];
                    let mut h_j_r = rv[0] * acc.right_j[0];
                    let mut h_ij_r = rv[0] * acc.right_ij[0];
                    let mut hp_i_r = rd[0] * acc.right_i[0];
                    let mut hp_j_r = rd[0] * acc.right_j[0];
                    let mut hp_ij_r = rd[0] * acc.right_ij[0];

                    let mut h_lr = 0.0;
                    let mut hp_lr = 0.0;
                    let mut h_i_lr = 0.0;
                    let mut h_j_lr = 0.0;
                    let mut h_ij_lr = 0.0;
                    let mut hp_i_lr = 0.0;
                    let mut hp_j_lr = 0.0;
                    let mut hp_ij_lr = 0.0;

                    for k in 1..p_resp {
                        let g = acc.gamma[k];
                        let gi = acc.gamma_i[k];
                        let gj = acc.gamma_j[k];
                        let gij = acc.gamma_ij[k];
                        let l = acc.left[k];
                        let li = acc.left_i[k];
                        let lj = acc.left_j[k];
                        let lij = acc.left_ij[k];
                        let r = acc.right[k];
                        let ri = acc.right_i[k];
                        let rj = acc.right_j[k];
                        let rij = acc.right_ij[k];

                        h_i += 2.0 * rv[k] * g * gi;
                        h_j += 2.0 * rv[k] * g * gj;
                        h_ij += 2.0 * rv[k] * (gj * gi + g * gij);
                        hp_i += 2.0 * rd[k] * g * gi;
                        hp_j += 2.0 * rd[k] * g * gj;
                        hp_ij += 2.0 * rd[k] * (gj * gi + g * gij);

                        h_l += 2.0 * rv[k] * g * l;
                        hp_l += 2.0 * rd[k] * g * l;
                        h_i_l += 2.0 * rv[k] * (l * gi + g * li);
                        h_j_l += 2.0 * rv[k] * (l * gj + g * lj);
                        h_ij_l += 2.0 * rv[k] * (lj * gi + gj * li + l * gij + g * lij);
                        hp_i_l += 2.0 * rd[k] * (l * gi + g * li);
                        hp_j_l += 2.0 * rd[k] * (l * gj + g * lj);
                        hp_ij_l += 2.0 * rd[k] * (lj * gi + gj * li + l * gij + g * lij);

                        h_r += 2.0 * rv[k] * g * r;
                        hp_r += 2.0 * rd[k] * g * r;
                        h_i_r += 2.0 * rv[k] * (r * gi + g * ri);
                        h_j_r += 2.0 * rv[k] * (r * gj + g * rj);
                        h_ij_r += 2.0 * rv[k] * (rj * gi + gj * ri + r * gij + g * rij);
                        hp_i_r += 2.0 * rd[k] * (r * gi + g * ri);
                        hp_j_r += 2.0 * rd[k] * (r * gj + g * rj);
                        hp_ij_r += 2.0 * rd[k] * (rj * gi + gj * ri + r * gij + g * rij);

                        h_lr += 2.0 * rv[k] * l * r;
                        hp_lr += 2.0 * rd[k] * l * r;
                        h_i_lr += 2.0 * rv[k] * (l * ri + r * li);
                        h_j_lr += 2.0 * rv[k] * (l * rj + r * lj);
                        h_ij_lr += 2.0 * rv[k] * (lj * ri + rj * li + l * rij + r * lij);
                        hp_i_lr += 2.0 * rd[k] * (l * ri + r * li);
                        hp_j_lr += 2.0 * rd[k] * (l * rj + r * lj);
                        hp_ij_lr += 2.0 * rd[k] * (lj * ri + rj * li + l * rij + r * lij);
                    }

                    let q = endpoint_q[row_idx];
                    let mut endpoint_i = [0.0; 2];
                    let mut endpoint_j = [0.0; 2];
                    let mut endpoint_ij = [0.0; 2];
                    let mut endpoint_l = [0.0; 2];
                    let mut endpoint_r = [0.0; 2];
                    let mut endpoint_i_l = [0.0; 2];
                    let mut endpoint_j_l = [0.0; 2];
                    let mut endpoint_ij_l = [0.0; 2];
                    let mut endpoint_i_r = [0.0; 2];
                    let mut endpoint_j_r = [0.0; 2];
                    let mut endpoint_ij_r = [0.0; 2];
                    let mut endpoint_l_r = [0.0; 2];
                    let mut endpoint_i_l_r = [0.0; 2];
                    let mut endpoint_j_l_r = [0.0; 2];
                    let mut endpoint_ij_l_r = [0.0; 2];
                    for e in 0..2 {
                        let basis = endpoint_basis[e];
                        endpoint_i[e] = basis[0] * acc.gamma_i[0];
                        endpoint_j[e] = basis[0] * acc.gamma_j[0];
                        endpoint_ij[e] = basis[0] * acc.gamma_ij[0];
                        endpoint_l[e] = basis[0] * acc.left[0];
                        endpoint_r[e] = basis[0] * acc.right[0];
                        endpoint_i_l[e] = basis[0] * acc.left_i[0];
                        endpoint_j_l[e] = basis[0] * acc.left_j[0];
                        endpoint_ij_l[e] = basis[0] * acc.left_ij[0];
                        endpoint_i_r[e] = basis[0] * acc.right_i[0];
                        endpoint_j_r[e] = basis[0] * acc.right_j[0];
                        endpoint_ij_r[e] = basis[0] * acc.right_ij[0];
                        for k in 1..p_resp {
                            let basis_k = basis[k];
                            let g = acc.gamma[k];
                            let gi = acc.gamma_i[k];
                            let gj = acc.gamma_j[k];
                            let gij = acc.gamma_ij[k];
                            let l = acc.left[k];
                            let li = acc.left_i[k];
                            let lj = acc.left_j[k];
                            let lij = acc.left_ij[k];
                            let r = acc.right[k];
                            let ri = acc.right_i[k];
                            let rj = acc.right_j[k];
                            let rij = acc.right_ij[k];
                            endpoint_i[e] += 2.0 * basis_k * g * gi;
                            endpoint_j[e] += 2.0 * basis_k * g * gj;
                            endpoint_ij[e] += 2.0 * basis_k * (gj * gi + g * gij);
                            endpoint_l[e] += 2.0 * basis_k * g * l;
                            endpoint_r[e] += 2.0 * basis_k * g * r;
                            endpoint_i_l[e] += 2.0 * basis_k * (l * gi + g * li);
                            endpoint_j_l[e] += 2.0 * basis_k * (l * gj + g * lj);
                            endpoint_ij_l[e] +=
                                2.0 * basis_k * (lj * gi + gj * li + l * gij + g * lij);
                            endpoint_i_r[e] += 2.0 * basis_k * (r * gi + g * ri);
                            endpoint_j_r[e] += 2.0 * basis_k * (r * gj + g * rj);
                            endpoint_ij_r[e] +=
                                2.0 * basis_k * (rj * gi + gj * ri + r * gij + g * rij);
                            endpoint_l_r[e] += 2.0 * basis_k * l * r;
                            endpoint_i_l_r[e] += 2.0 * basis_k * (l * ri + r * li);
                            endpoint_j_l_r[e] += 2.0 * basis_k * (l * rj + r * lj);
                            endpoint_ij_l_r[e] +=
                                2.0 * basis_k * (lj * ri + rj * li + l * rij + r * lij);
                        }
                    }

                    let inv_hp = 1.0 / hp;
                    let inv_hp_sq = inv_hp * inv_hp;
                    let inv_hp_cu = inv_hp_sq * inv_hp;
                    let inv_hp_qu = inv_hp_sq * inv_hp_sq;
                    let numerator_l = hp_i_l * hp_j + hp_i * hp_j_l;
                    let numerator_r = hp_i_r * hp_j + hp_i * hp_j_r;
                    let numerator_lr =
                        hp_i_lr * hp_j + hp_i_l * hp_j_r + hp_i_r * hp_j_l + hp_i * hp_j_lr;
                    let value_lr = h_i_lr * h_j
                        + h_i_l * h_j_r
                        + h_i_r * h_j_l
                        + h_i * h_j_lr
                        + h_lr * h_ij
                        + h_l * h_ij_r
                        + h_r * h_ij_l
                        + h * h_ij_lr
                        - hp_ij_lr * inv_hp
                        + hp_ij_l * hp_r * inv_hp_sq
                        + hp_ij_r * hp_l * inv_hp_sq
                        + hp_ij * hp_lr * inv_hp_sq
                        - 2.0 * hp_ij * hp_l * hp_r * inv_hp_cu
                        + numerator_lr * inv_hp_sq
                        - 2.0 * numerator_l * hp_r * inv_hp_cu
                        - 2.0 * numerator_r * hp_l * inv_hp_cu
                        - 2.0 * hp_i * hp_j * hp_lr * inv_hp_cu
                        + 6.0 * hp_i * hp_j * hp_l * hp_r * inv_hp_qu
                        + endpoint_chain_fourth(
                            &q,
                            endpoint_i,
                            endpoint_j,
                            endpoint_l,
                            endpoint_r,
                            endpoint_ij,
                            endpoint_i_l,
                            endpoint_i_r,
                            endpoint_j_l,
                            endpoint_j_r,
                            endpoint_l_r,
                            endpoint_ij_l,
                            endpoint_ij_r,
                            endpoint_i_l_r,
                            endpoint_j_l_r,
                            endpoint_ij_l_r,
                        );
                    acc.value += weights[global_row] * value_lr;
                }
                acc
            },
            |mut left, right| {
                left.value += right.value;
                left
            },
        )
        .unwrap_or_else(|| PsiPairBilinearAccum::new(p_resp))
        .value;
        Ok(total)
    }

    pub(crate) fn scop_psi_psi_trace_factor_from_cov(
        &self,
        beta: &Array1<f64>,
        cached_gamma: ArrayView2<'_, f64>,
        cached_h: ArrayView1<'_, f64>,
        cached_h_prime: ArrayView1<'_, f64>,
        cov: ArrayView2<'_, f64>,
        cov_i: ArrayView2<'_, f64>,
        cov_j: ArrayView2<'_, f64>,
        cov_ij: ArrayView2<'_, f64>,
        row_start: usize,
        endpoint_q: &[LogNormalCdfDiffDerivatives],
        factor: ArrayView2<'_, f64>,
    ) -> Result<f64, String> {
        let total_n = self.response_val_basis.nrows();
        let n = cov.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        let rank = factor.ncols();
        if row_start > total_n || row_start + n > total_n {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi-psi projected trace row window [{row_start}, {}) exceeds n={total_n}",
                    row_start + n
                ),
            }
            .into());
        }
        if beta.len() != p_total || factor.nrows() != p_total {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi-psi projected trace length mismatch: beta={}, factor_rows={}, expected={p_total}",
                beta.len(),
                factor.nrows()
            ) }.into());
        }
        if cached_gamma.nrows() != n || cached_gamma.ncols() != p_resp {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi-psi projected trace gamma cache shape {}x{} != expected {}x{}",
                    cached_gamma.nrows(),
                    cached_gamma.ncols(),
                    n,
                    p_resp
                ),
            }
            .into());
        }
        let factor_data = factor.as_slice().ok_or_else(|| {
            "SCOP psi-psi projected trace factor matrix must be standard contiguous".to_string()
        })?;
        if endpoint_q.len() != n {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi-psi projected trace endpoint normalizer cache length {} != n={n}",
                    endpoint_q.len()
                ),
            }
            .into());
        }
        if cached_h.len() != n || cached_h_prime.len() != n {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi-psi projected trace row-quantity cache length mismatch: h={}, h_prime={}, expected={n}",
                cached_h.len(),
                cached_h_prime.len()
            ) }.into());
        }
        for (name, mat) in [
            ("cov", cov),
            ("cov_i", cov_i),
            ("cov_j", cov_j),
            ("cov_ij", cov_ij),
        ] {
            if mat.nrows() != n || mat.ncols() != p_cov {
                return Err(TransformationNormalError::InvalidInput {
                    reason: format!(
                        "SCOP psi-psi projected trace {name} shape {}x{} != expected {}x{}",
                        mat.nrows(),
                        mat.ncols(),
                        n,
                        p_cov
                    ),
                }
                .into());
            }
        }

        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP psi-psi projected trace beta reshape failed: {e}"))?;
        let endpoint_basis = [
            self.response_upper_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint upper basis is not contiguous".to_string())?,
            self.response_lower_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint lower basis is not contiguous".to_string())?,
        ];

        struct PsiPairTraceAccum {
            pub(crate) value: f64,
            pub(crate) gamma: Vec<f64>,
            pub(crate) gamma_i: Vec<f64>,
            pub(crate) gamma_j: Vec<f64>,
            pub(crate) gamma_ij: Vec<f64>,
            pub(crate) f: Vec<f64>,
            pub(crate) f_i: Vec<f64>,
            pub(crate) f_j: Vec<f64>,
            pub(crate) f_ij: Vec<f64>,
        }

        impl PsiPairTraceAccum {
            pub(crate) fn new(p_resp: usize, rank: usize) -> Self {
                let projected_len = p_resp * rank;
                Self {
                    value: 0.0,
                    gamma: vec![0.0; p_resp],
                    gamma_i: vec![0.0; p_resp],
                    gamma_j: vec![0.0; p_resp],
                    gamma_ij: vec![0.0; p_resp],
                    f: vec![0.0; projected_len],
                    f_i: vec![0.0; projected_len],
                    f_j: vec![0.0; projected_len],
                    f_ij: vec![0.0; projected_len],
                }
            }
        }

        let weights = self.effective_weights();
        let total = gam_linalg::pairwise_reduce::par_deterministic_block_fold(
            n,
            |range| {
                let mut acc = PsiPairTraceAccum::new(p_resp, rank);
                for row_idx in range {
                    let cov_row = cov.row(row_idx);
                    let cov_i_row = cov_i.row(row_idx);
                    let cov_j_row = cov_j.row(row_idx);
                    let cov_ij_row = cov_ij.row(row_idx);
                    let global_row = row_start + row_idx;
                    let rv = self.response_val_basis.row(global_row);
                    let rd = self.response_deriv_basis.row(global_row);
                    let gamma_row = cached_gamma.row(row_idx);

                    for k in 0..p_resp {
                        let beta_k = beta_mat.row(k);
                        acc.gamma[k] = gamma_row[k];
                        acc.gamma_i[k] = beta_k.dot(&cov_i_row);
                        acc.gamma_j[k] = beta_k.dot(&cov_j_row);
                        acc.gamma_ij[k] = beta_k.dot(&cov_ij_row);
                    }

                    let h = cached_h[row_idx];
                    let hp = cached_h_prime[row_idx];
                    let [h_i, h_j, h_ij, hp_i, hp_j, hp_ij] = scop_second_order_h(
                        rv,
                        rd,
                        p_resp,
                        &acc.gamma,
                        &acc.gamma_i,
                        &acc.gamma_j,
                        &acc.gamma_ij,
                    );

                    let q = endpoint_q[row_idx];
                    let (endpoint_i, endpoint_j, endpoint_ij) = scop_second_order_endpoints(
                        endpoint_basis,
                        p_resp,
                        &acc.gamma,
                        &acc.gamma_i,
                        &acc.gamma_j,
                        &acc.gamma_ij,
                    );

                    let inv_hp = 1.0 / hp;
                    let inv_hp_sq = inv_hp * inv_hp;
                    let inv_hp_cu = inv_hp_sq * inv_hp;
                    let inv_hp_qu = inv_hp_sq * inv_hp_sq;
                    let wi = weights[global_row];

                    acc.f.fill(0.0);
                    acc.f_i.fill(0.0);
                    acc.f_j.fill(0.0);
                    acc.f_ij.fill(0.0);
                    for k in 0..p_resp {
                        let factor_row_base = k * p_cov;
                        let projected_base = k * rank;
                        for cidx in 0..p_cov {
                            let coeff_base = (factor_row_base + cidx) * rank;
                            let cov_v = cov_row[cidx];
                            let cov_i_v = cov_i_row[cidx];
                            let cov_j_v = cov_j_row[cidx];
                            let cov_ij_v = cov_ij_row[cidx];
                            for col in 0..rank {
                                let coeff = factor_data[coeff_base + col];
                                let idx = projected_base + col;
                                acc.f[idx] += coeff * cov_v;
                                acc.f_i[idx] += coeff * cov_i_v;
                                acc.f_j[idx] += coeff * cov_j_v;
                                acc.f_ij[idx] += coeff * cov_ij_v;
                            }
                        }
                    }

                    for col in 0..rank {
                        let mut h_f = rv[0] * acc.f[col];
                        let mut hp_f = rd[0] * acc.f[col];
                        let mut h_i_f = rv[0] * acc.f_i[col];
                        let mut h_j_f = rv[0] * acc.f_j[col];
                        let mut h_ij_f = rv[0] * acc.f_ij[col];
                        let mut hp_i_f = rd[0] * acc.f_i[col];
                        let mut hp_j_f = rd[0] * acc.f_j[col];
                        let mut hp_ij_f = rd[0] * acc.f_ij[col];

                        let mut h_ff = 0.0;
                        let mut hp_ff = 0.0;
                        let mut h_i_ff = 0.0;
                        let mut h_j_ff = 0.0;
                        let mut h_ij_ff = 0.0;
                        let mut hp_i_ff = 0.0;
                        let mut hp_j_ff = 0.0;
                        let mut hp_ij_ff = 0.0;

                        for k in 1..p_resp {
                            let g = acc.gamma[k];
                            let gi = acc.gamma_i[k];
                            let gj = acc.gamma_j[k];
                            let gij = acc.gamma_ij[k];
                            let projected_idx = k * rank + col;
                            let f = acc.f[projected_idx];
                            let fi = acc.f_i[projected_idx];
                            let fj = acc.f_j[projected_idx];
                            let fij = acc.f_ij[projected_idx];

                            h_f += 2.0 * rv[k] * g * f;
                            hp_f += 2.0 * rd[k] * g * f;
                            h_i_f += 2.0 * rv[k] * (f * gi + g * fi);
                            h_j_f += 2.0 * rv[k] * (f * gj + g * fj);
                            h_ij_f += 2.0 * rv[k] * (fj * gi + gj * fi + f * gij + g * fij);
                            hp_i_f += 2.0 * rd[k] * (f * gi + g * fi);
                            hp_j_f += 2.0 * rd[k] * (f * gj + g * fj);
                            hp_ij_f += 2.0 * rd[k] * (fj * gi + gj * fi + f * gij + g * fij);

                            h_ff += 2.0 * rv[k] * f * f;
                            hp_ff += 2.0 * rd[k] * f * f;
                            h_i_ff += 4.0 * rv[k] * f * fi;
                            h_j_ff += 4.0 * rv[k] * f * fj;
                            h_ij_ff += 2.0 * rv[k] * (fj * fi + fj * fi + f * fij + f * fij);
                            hp_i_ff += 4.0 * rd[k] * f * fi;
                            hp_j_ff += 4.0 * rd[k] * f * fj;
                            hp_ij_ff += 2.0 * rd[k] * (fj * fi + fj * fi + f * fij + f * fij);
                        }

                        let mut endpoint_f = [0.0; 2];
                        let mut endpoint_i_f = [0.0; 2];
                        let mut endpoint_j_f = [0.0; 2];
                        let mut endpoint_ij_f = [0.0; 2];
                        let mut endpoint_ff = [0.0; 2];
                        let mut endpoint_i_ff = [0.0; 2];
                        let mut endpoint_j_ff = [0.0; 2];
                        let mut endpoint_ij_ff = [0.0; 2];
                        for e in 0..2 {
                            let basis = endpoint_basis[e];
                            endpoint_f[e] = basis[0] * acc.f[col];
                            endpoint_i_f[e] = basis[0] * acc.f_i[col];
                            endpoint_j_f[e] = basis[0] * acc.f_j[col];
                            endpoint_ij_f[e] = basis[0] * acc.f_ij[col];
                            for k in 1..p_resp {
                                let basis_k = basis[k];
                                let g = acc.gamma[k];
                                let gi = acc.gamma_i[k];
                                let gj = acc.gamma_j[k];
                                let gij = acc.gamma_ij[k];
                                let projected_idx = k * rank + col;
                                let f = acc.f[projected_idx];
                                let fi = acc.f_i[projected_idx];
                                let fj = acc.f_j[projected_idx];
                                let fij = acc.f_ij[projected_idx];
                                endpoint_f[e] += 2.0 * basis_k * g * f;
                                endpoint_i_f[e] += 2.0 * basis_k * (f * gi + g * fi);
                                endpoint_j_f[e] += 2.0 * basis_k * (f * gj + g * fj);
                                endpoint_ij_f[e] +=
                                    2.0 * basis_k * (fj * gi + gj * fi + f * gij + g * fij);
                                endpoint_ff[e] += 2.0 * basis_k * f * f;
                                endpoint_i_ff[e] += 4.0 * basis_k * f * fi;
                                endpoint_j_ff[e] += 4.0 * basis_k * f * fj;
                                endpoint_ij_ff[e] += 4.0 * basis_k * (fj * fi + f * fij);
                            }
                        }

                        let numerator_f = hp_i_f * hp_j + hp_i * hp_j_f;
                        let numerator_ff = hp_i_ff * hp_j + 2.0 * hp_i_f * hp_j_f + hp_i * hp_j_ff;
                        let value_ff = h_i_ff * h_j
                            + 2.0 * h_i_f * h_j_f
                            + h_i * h_j_ff
                            + h_ff * h_ij
                            + 2.0 * h_f * h_ij_f
                            + h * h_ij_ff
                            - hp_ij_ff * inv_hp
                            + 2.0 * hp_ij_f * hp_f * inv_hp_sq
                            + hp_ij * hp_ff * inv_hp_sq
                            - 2.0 * hp_ij * hp_f * hp_f * inv_hp_cu
                            + numerator_ff * inv_hp_sq
                            - 4.0 * numerator_f * hp_f * inv_hp_cu
                            - 2.0 * hp_i * hp_j * hp_ff * inv_hp_cu
                            + 6.0 * hp_i * hp_j * hp_f * hp_f * inv_hp_qu
                            + endpoint_chain_fourth(
                                &q,
                                endpoint_i,
                                endpoint_j,
                                endpoint_f,
                                endpoint_f,
                                endpoint_ij,
                                endpoint_i_f,
                                endpoint_i_f,
                                endpoint_j_f,
                                endpoint_j_f,
                                endpoint_ff,
                                endpoint_ij_f,
                                endpoint_ij_f,
                                endpoint_i_ff,
                                endpoint_j_ff,
                                endpoint_ij_ff,
                            );
                        acc.value += wi * value_ff;
                    }
                }
                acc
            },
            |mut left, right| {
                left.value += right.value;
                left
            },
        )
        .unwrap_or_else(|| PsiPairTraceAccum::new(p_resp, rank))
        .value;
        Ok(total)
    }

    pub(crate) fn scop_psi_pair_rows_per_chunk(&self, p_cov: usize) -> usize {
        let policy = ResourcePolicy::default_library();
        gam_runtime::resource::rows_for_target_bytes(
            policy.row_chunk_target_bytes,
            4 * p_cov.max(1),
        )
        .max(1)
    }

    pub(crate) fn scop_psi_pair_cov_chunks(
        &self,
        op: &TensorKroneckerPsiOperator,
        axis_i: usize,
        axis_j: usize,
        rows: std::ops::Range<usize>,
    ) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>), String> {
        let cov = self
            .covariate_dense_arc()?
            .slice(s![rows.clone(), ..])
            .to_owned();
        let cov_i = op
            .cov_first_axis_row_chunk(axis_i, rows.clone())
            .map_err(|e| format!("SCOP psi-psi covariate first-axis row chunk(i) failed: {e}"))?;
        let cov_j = op
            .cov_first_axis_row_chunk(axis_j, rows.clone())
            .map_err(|e| format!("SCOP psi-psi covariate first-axis row chunk(j) failed: {e}"))?;
        let cov_ij = op
            .cov_second_axis_row_chunk(axis_i, axis_j, rows)
            .map_err(|e| format!("SCOP psi-psi covariate second-axis row chunk failed: {e}"))?;
        Ok((cov, cov_i, cov_j, cov_ij))
    }

    pub(crate) fn scop_psi_psi_value_score_hvp_from_operator(
        &self,
        beta: &Array1<f64>,
        op: &TensorKroneckerPsiOperator,
        axis_i: usize,
        axis_j: usize,
        cached_gamma: ArrayView2<'_, f64>,
        cached_h: ArrayView1<'_, f64>,
        cached_h_prime: ArrayView1<'_, f64>,
        endpoint_q: &[LogNormalCdfDiffDerivatives],
        direction: Option<&Array1<f64>>,
    ) -> Result<(f64, Array1<f64>, Option<Array1<f64>>), String> {
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if endpoint_q.len() != n {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi-psi operator endpoint normalizer cache length {} != n={n}",
                    endpoint_q.len()
                ),
            }
            .into());
        }
        if cached_h.len() != n || cached_h_prime.len() != n {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi-psi operator row-quantity cache length mismatch: h={}, h_prime={}, expected={n}",
                cached_h.len(),
                cached_h_prime.len()
            ) }.into());
        }
        if cached_gamma.nrows() != n || cached_gamma.ncols() != p_resp {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi-psi operator gamma cache shape {}x{} != expected {}x{}",
                    cached_gamma.nrows(),
                    cached_gamma.ncols(),
                    n,
                    p_resp
                ),
            }
            .into());
        }
        let rows_per_chunk = self.scop_psi_pair_rows_per_chunk(p_cov).min(n.max(1));
        let mut objective = 0.0;
        let mut score = Array1::<f64>::zeros(p_total);
        let mut hvp = direction.map(|_| Array1::<f64>::zeros(p_total));

        for start in (0..n).step_by(rows_per_chunk) {
            let end = (start + rows_per_chunk).min(n);
            let rows = start..end;
            let (cov, cov_i, cov_j, cov_ij) =
                self.scop_psi_pair_cov_chunks(op, axis_i, axis_j, rows.clone())?;
            let (obj_chunk, score_chunk, hvp_chunk) = self.scop_psi_psi_value_score_hvp_from_cov(
                beta,
                cached_gamma.slice(s![start..end, ..]),
                cached_h.slice(s![start..end]),
                cached_h_prime.slice(s![start..end]),
                cov.view(),
                cov_i.view(),
                cov_j.view(),
                cov_ij.view(),
                start,
                &endpoint_q[start..end],
                direction,
            )?;
            objective += obj_chunk;
            score.scaled_add(1.0, &score_chunk);
            if let (Some(total), Some(chunk)) = (hvp.as_mut(), hvp_chunk.as_ref()) {
                total.scaled_add(1.0, chunk);
            }
        }

        Ok((objective, score, hvp))
    }

    pub(crate) fn scop_psi_psi_bilinear_from_operator(
        &self,
        beta: &Array1<f64>,
        op: &TensorKroneckerPsiOperator,
        axis_i: usize,
        axis_j: usize,
        cached_gamma: ArrayView2<'_, f64>,
        cached_h: ArrayView1<'_, f64>,
        cached_h_prime: ArrayView1<'_, f64>,
        endpoint_q: &[LogNormalCdfDiffDerivatives],
        left: ArrayView1<'_, f64>,
        right: ArrayView1<'_, f64>,
    ) -> Result<f64, String> {
        let n = self.response_val_basis.nrows();
        let p_cov = self.covariate_design.ncols();
        if endpoint_q.len() != n {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi-psi bilinear operator endpoint normalizer cache length {} != n={n}",
                    endpoint_q.len()
                ),
            }
            .into());
        }
        if cached_h.len() != n || cached_h_prime.len() != n {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi-psi bilinear operator row-quantity cache length mismatch: h={}, h_prime={}, expected={n}",
                cached_h.len(),
                cached_h_prime.len()
            ) }.into());
        }
        let p_resp = self.response_val_basis.ncols();
        if cached_gamma.nrows() != n || cached_gamma.ncols() != p_resp {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi-psi bilinear operator gamma cache shape {}x{} != expected {}x{}",
                    cached_gamma.nrows(),
                    cached_gamma.ncols(),
                    n,
                    p_resp
                ),
            }
            .into());
        }
        let rows_per_chunk = self.scop_psi_pair_rows_per_chunk(p_cov).min(n.max(1));
        let mut total = 0.0;
        for start in (0..n).step_by(rows_per_chunk) {
            let end = (start + rows_per_chunk).min(n);
            let rows = start..end;
            let (cov, cov_i, cov_j, cov_ij) =
                self.scop_psi_pair_cov_chunks(op, axis_i, axis_j, rows.clone())?;
            total += self.scop_psi_psi_bilinear_from_cov(
                beta,
                cached_gamma.slice(s![start..end, ..]),
                cached_h.slice(s![start..end]),
                cached_h_prime.slice(s![start..end]),
                cov.view(),
                cov_i.view(),
                cov_j.view(),
                cov_ij.view(),
                start,
                &endpoint_q[start..end],
                left,
                right,
            )?;
        }
        Ok(total)
    }

    pub(crate) fn scop_psi_hessian_directional_derivative(
        &self,
        beta: &Array1<f64>,
        direction: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
        op: &TensorKroneckerPsiOperator,
        axis: usize,
    ) -> Result<Array2<f64>, String> {
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if beta.len() != p_total || direction.len() != p_total {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi Hessian directional derivative length mismatch: beta={}, direction={}, expected={p_total}",
                beta.len(),
                direction.len()
            ) }.into());
        }
        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP psi hessian beta reshape failed: {e}"))?;
        let dir_mat = direction
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP psi hessian direction reshape failed: {e}"))?;
        let cov = self.covariate_dense_arc().map_err(|e| {
            format!("SCOP psi hessian direction requires cached covariate design: {e}")
        })?;
        let cov_psi_arc = op
            .materialize_cov_first_axis_arc(axis)
            .map_err(|e| format!("SCOP psi hessian materialize_cov_first failed: {e}"))?;
        let cov_psi = cov_psi_arc.view();
        if cov_psi.nrows() != n || cov_psi.ncols() != p_cov {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi hessian covariate derivative shape {}x{} != expected {}x{}",
                    cov_psi.nrows(),
                    cov_psi.ncols(),
                    n,
                    p_cov
                ),
            }
            .into());
        }

        let weights = self.effective_weights();
        let h_prime = row_quantities.h_prime.as_ref();
        let endpoint_basis = [
            self.response_upper_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint upper basis is not contiguous".to_string())?,
            self.response_lower_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint lower basis is not contiguous".to_string())?,
        ];
        // Parallelise the outer `for i in 0..n` row accumulation across
        // Rayon threads. The inner k×l×c×d loop dominates wall-clock at
        // large scale; per-row contributions to `out` are independent
        // (each row only contributes additively), so a thread-local
        // accumulator + reduction is safe and gives ~Nthreads× wall-clock
        // win. Per-thread scratch buffers are created once via
        // `fold(|| init, …)` and reused across all rows assigned to that
        // thread.
        struct Scratch {
            pub(crate) out: Array2<f64>,
            pub(crate) gamma: Vec<f64>,
            pub(crate) gamma_dir: Vec<f64>,
            pub(crate) gamma_psi: Vec<f64>,
            pub(crate) gamma_psi_dir: Vec<f64>,
            pub(crate) endpoint_factor: Vec<[f64; 2]>,
            pub(crate) endpoint_factor_dir: Vec<[f64; 2]>,
            pub(crate) endpoint_psi_cov_factor: Vec<[f64; 2]>,
            pub(crate) endpoint_psi_psi_factor: Vec<[f64; 2]>,
            pub(crate) endpoint_psi_cov_factor_dir: Vec<[f64; 2]>,
            pub(crate) endpoint_psi_psi_factor_dir: Vec<[f64; 2]>,
            pub(crate) h_factor: Vec<f64>,
            pub(crate) hp_factor: Vec<f64>,
            pub(crate) h_factor_dir: Vec<f64>,
            pub(crate) hp_factor_dir: Vec<f64>,
            pub(crate) hpsi_cov_factor: Vec<f64>,
            pub(crate) hppsi_cov_factor: Vec<f64>,
            pub(crate) hpsi_psi_factor: Vec<f64>,
            pub(crate) hppsi_psi_factor: Vec<f64>,
            pub(crate) hpsi_cov_factor_dir: Vec<f64>,
            pub(crate) hppsi_cov_factor_dir: Vec<f64>,
            pub(crate) hpsi_psi_factor_dir: Vec<f64>,
            pub(crate) hppsi_psi_factor_dir: Vec<f64>,
        }
        let init_scratch = || Scratch {
            out: Array2::<f64>::zeros((p_total, p_total)),
            gamma: vec![0.0; p_resp],
            gamma_dir: vec![0.0; p_resp],
            gamma_psi: vec![0.0; p_resp],
            gamma_psi_dir: vec![0.0; p_resp],
            endpoint_factor: vec![[0.0_f64; 2]; p_resp],
            endpoint_factor_dir: vec![[0.0_f64; 2]; p_resp],
            endpoint_psi_cov_factor: vec![[0.0_f64; 2]; p_resp],
            endpoint_psi_psi_factor: vec![[0.0_f64; 2]; p_resp],
            endpoint_psi_cov_factor_dir: vec![[0.0_f64; 2]; p_resp],
            endpoint_psi_psi_factor_dir: vec![[0.0_f64; 2]; p_resp],
            h_factor: vec![0.0; p_resp],
            hp_factor: vec![0.0; p_resp],
            h_factor_dir: vec![0.0; p_resp],
            hp_factor_dir: vec![0.0; p_resp],
            hpsi_cov_factor: vec![0.0; p_resp],
            hppsi_cov_factor: vec![0.0; p_resp],
            hpsi_psi_factor: vec![0.0; p_resp],
            hppsi_psi_factor: vec![0.0; p_resp],
            hpsi_cov_factor_dir: vec![0.0; p_resp],
            hppsi_cov_factor_dir: vec![0.0; p_resp],
            hpsi_psi_factor_dir: vec![0.0; p_resp],
            hppsi_psi_factor_dir: vec![0.0; p_resp],
        };

        let process_row = |scratch: &mut Scratch, i: usize| {
            let Scratch {
                out,
                gamma,
                gamma_dir,
                gamma_psi,
                gamma_psi_dir,
                endpoint_factor,
                endpoint_factor_dir,
                endpoint_psi_cov_factor,
                endpoint_psi_psi_factor,
                endpoint_psi_cov_factor_dir,
                endpoint_psi_psi_factor_dir,
                h_factor,
                hp_factor,
                h_factor_dir,
                hp_factor_dir,
                hpsi_cov_factor,
                hppsi_cov_factor,
                hpsi_psi_factor,
                hppsi_psi_factor,
                hpsi_cov_factor_dir,
                hppsi_cov_factor_dir,
                hpsi_psi_factor_dir,
                hppsi_psi_factor_dir,
            } = scratch;
            let cov_row = cov.row(i);
            let psi_row = cov_psi.row(i);
            let rv = self.response_val_basis.row(i);
            let rd = self.response_deriv_basis.row(i);
            let wi = weights[i];
            let hp = h_prime[i];
            let inv_hp = 1.0 / hp;
            let inv_hp_sq = inv_hp * inv_hp;
            let inv_hp_cu = inv_hp_sq * inv_hp;
            let inv_hp_qu = inv_hp_sq * inv_hp_sq;

            // Reset endpoint/factor buffers whose index 0 is never written
            // explicitly below — original code relied on `vec![0.0; …]`
            // zero-init for those slots. The four `gamma` Vecs are fully
            // overwritten in the loop just below, so they don't need filling.
            endpoint_factor.fill([0.0; 2]);
            endpoint_factor_dir.fill([0.0; 2]);
            endpoint_psi_cov_factor.fill([0.0; 2]);
            endpoint_psi_psi_factor.fill([0.0; 2]);
            endpoint_psi_cov_factor_dir.fill([0.0; 2]);
            endpoint_psi_psi_factor_dir.fill([0.0; 2]);
            h_factor.fill(0.0);
            hp_factor.fill(0.0);
            h_factor_dir.fill(0.0);
            hp_factor_dir.fill(0.0);
            hpsi_cov_factor.fill(0.0);
            hppsi_cov_factor.fill(0.0);
            hpsi_psi_factor.fill(0.0);
            hppsi_psi_factor.fill(0.0);
            hpsi_cov_factor_dir.fill(0.0);
            hppsi_cov_factor_dir.fill(0.0);
            hpsi_psi_factor_dir.fill(0.0);
            hppsi_psi_factor_dir.fill(0.0);
            for k in 0..p_resp {
                gamma[k] = beta_mat.row(k).dot(&cov_row);
                gamma_dir[k] = dir_mat.row(k).dot(&cov_row);
                gamma_psi[k] = beta_mat.row(k).dot(&psi_row);
                gamma_psi_dir[k] = dir_mat.row(k).dot(&psi_row);
            }

            let mut h_dir = rv[0] * gamma_dir[0];
            let mut hp_dir = rd[0] * gamma_dir[0];
            let mut hp_psi = rd[0] * gamma_psi[0];
            let mut h_psi_dir = rv[0] * gamma_psi_dir[0];
            let mut hp_psi_dir = rd[0] * gamma_psi_dir[0];
            for k in 1..p_resp {
                h_dir += 2.0 * rv[k] * gamma[k] * gamma_dir[k];
                hp_dir += 2.0 * rd[k] * gamma[k] * gamma_dir[k];
                hp_psi += 2.0 * rd[k] * gamma[k] * gamma_psi[k];
                h_psi_dir +=
                    2.0 * rv[k] * (gamma_dir[k] * gamma_psi[k] + gamma[k] * gamma_psi_dir[k]);
                hp_psi_dir +=
                    2.0 * rd[k] * (gamma_dir[k] * gamma_psi[k] + gamma[k] * gamma_psi_dir[k]);
            }
            let q = row_quantities.endpoint_q[i];
            let mut endpoint_psi = [0.0; 2];
            let mut endpoint_dir = [0.0; 2];
            let mut endpoint_psi_dir = [0.0; 2];
            for e in 0..2 {
                let basis = endpoint_basis[e];
                endpoint_psi[e] = basis[0] * gamma_psi[0];
                endpoint_dir[e] = basis[0] * gamma_dir[0];
                endpoint_psi_dir[e] = basis[0] * gamma_psi_dir[0];
                endpoint_factor[0][e] = basis[0];
                endpoint_psi_psi_factor[0][e] = basis[0];
                for k in 1..p_resp {
                    endpoint_psi[e] += 2.0 * basis[k] * gamma[k] * gamma_psi[k];
                    endpoint_dir[e] += 2.0 * basis[k] * gamma[k] * gamma_dir[k];
                    endpoint_psi_dir[e] += 2.0
                        * basis[k]
                        * (gamma_dir[k] * gamma_psi[k] + gamma[k] * gamma_psi_dir[k]);
                    endpoint_factor[k][e] = 2.0 * basis[k] * gamma[k];
                    endpoint_factor_dir[k][e] = 2.0 * basis[k] * gamma_dir[k];
                    endpoint_psi_cov_factor[k][e] = 2.0 * basis[k] * gamma_psi[k];
                    endpoint_psi_psi_factor[k][e] = 2.0 * basis[k] * gamma[k];
                    endpoint_psi_cov_factor_dir[k][e] = 2.0 * basis[k] * gamma_psi_dir[k];
                    endpoint_psi_psi_factor_dir[k][e] = 2.0 * basis[k] * gamma_dir[k];
                }
            }
            let d_inv_hp = -hp_dir * inv_hp_sq;
            let d_inv_hp_sq = -2.0 * hp_dir * inv_hp_cu;
            let d_inv_hp_cu = -3.0 * hp_dir * inv_hp_qu;

            h_factor[0] = rv[0];
            hp_factor[0] = rd[0];
            hpsi_psi_factor[0] = rv[0];
            hppsi_psi_factor[0] = rd[0];
            for k in 1..p_resp {
                h_factor[k] = 2.0 * rv[k] * gamma[k];
                hp_factor[k] = 2.0 * rd[k] * gamma[k];
                h_factor_dir[k] = 2.0 * rv[k] * gamma_dir[k];
                hp_factor_dir[k] = 2.0 * rd[k] * gamma_dir[k];
                hpsi_cov_factor[k] = 2.0 * rv[k] * gamma_psi[k];
                hppsi_cov_factor[k] = 2.0 * rd[k] * gamma_psi[k];
                hpsi_psi_factor[k] = 2.0 * rv[k] * gamma[k];
                hppsi_psi_factor[k] = 2.0 * rd[k] * gamma[k];
                hpsi_cov_factor_dir[k] = 2.0 * rv[k] * gamma_psi_dir[k];
                hppsi_cov_factor_dir[k] = 2.0 * rd[k] * gamma_psi_dir[k];
                hpsi_psi_factor_dir[k] = 2.0 * rv[k] * gamma_dir[k];
                hppsi_psi_factor_dir[k] = 2.0 * rd[k] * gamma_dir[k];
            }

            for k in 0..p_resp {
                for l in 0..p_resp {
                    let same_shape = k == l && k > 0;
                    for c in 0..p_cov {
                        let row_idx = k * p_cov + c;
                        let h_a = h_factor[k] * cov_row[c];
                        let hp_a = hp_factor[k] * cov_row[c];
                        let h_a_dir = h_factor_dir[k] * cov_row[c];
                        let hp_a_dir = hp_factor_dir[k] * cov_row[c];
                        let hpsi_a =
                            hpsi_cov_factor[k] * cov_row[c] + hpsi_psi_factor[k] * psi_row[c];
                        let hppsi_a =
                            hppsi_cov_factor[k] * cov_row[c] + hppsi_psi_factor[k] * psi_row[c];
                        let hpsi_a_dir = hpsi_cov_factor_dir[k] * cov_row[c]
                            + hpsi_psi_factor_dir[k] * psi_row[c];
                        let hppsi_a_dir = hppsi_cov_factor_dir[k] * cov_row[c]
                            + hppsi_psi_factor_dir[k] * psi_row[c];
                        for d in 0..p_cov {
                            let col_idx = l * p_cov + d;
                            let h_b = h_factor[l] * cov_row[d];
                            let hp_b = hp_factor[l] * cov_row[d];
                            let h_b_dir = h_factor_dir[l] * cov_row[d];
                            let hp_b_dir = hp_factor_dir[l] * cov_row[d];
                            let hpsi_b =
                                hpsi_cov_factor[l] * cov_row[d] + hpsi_psi_factor[l] * psi_row[d];
                            let hppsi_b =
                                hppsi_cov_factor[l] * cov_row[d] + hppsi_psi_factor[l] * psi_row[d];
                            let hpsi_b_dir = hpsi_cov_factor_dir[l] * cov_row[d]
                                + hpsi_psi_factor_dir[l] * psi_row[d];
                            let hppsi_b_dir = hppsi_cov_factor_dir[l] * cov_row[d]
                                + hppsi_psi_factor_dir[l] * psi_row[d];
                            let (h_ab, hp_ab, hpsi_ab, hppsi_ab) = if same_shape {
                                (
                                    2.0 * rv[k] * cov_row[c] * cov_row[d],
                                    2.0 * rd[k] * cov_row[c] * cov_row[d],
                                    2.0 * rv[k]
                                        * (psi_row[d] * cov_row[c] + psi_row[c] * cov_row[d]),
                                    2.0 * rd[k]
                                        * (psi_row[d] * cov_row[c] + psi_row[c] * cov_row[d]),
                                )
                            } else {
                                (0.0, 0.0, 0.0, 0.0)
                            };
                            let endpoint_a = [
                                endpoint_factor[k][0] * cov_row[c],
                                endpoint_factor[k][1] * cov_row[c],
                            ];
                            let endpoint_b = [
                                endpoint_factor[l][0] * cov_row[d],
                                endpoint_factor[l][1] * cov_row[d],
                            ];
                            let endpoint_psi_a = [
                                endpoint_psi_cov_factor[k][0] * cov_row[c]
                                    + endpoint_psi_psi_factor[k][0] * psi_row[c],
                                endpoint_psi_cov_factor[k][1] * cov_row[c]
                                    + endpoint_psi_psi_factor[k][1] * psi_row[c],
                            ];
                            let endpoint_psi_b = [
                                endpoint_psi_cov_factor[l][0] * cov_row[d]
                                    + endpoint_psi_psi_factor[l][0] * psi_row[d],
                                endpoint_psi_cov_factor[l][1] * cov_row[d]
                                    + endpoint_psi_psi_factor[l][1] * psi_row[d],
                            ];
                            let endpoint_a_dir = [
                                endpoint_factor_dir[k][0] * cov_row[c],
                                endpoint_factor_dir[k][1] * cov_row[c],
                            ];
                            let endpoint_b_dir = [
                                endpoint_factor_dir[l][0] * cov_row[d],
                                endpoint_factor_dir[l][1] * cov_row[d],
                            ];
                            let endpoint_psi_a_dir = [
                                endpoint_psi_cov_factor_dir[k][0] * cov_row[c]
                                    + endpoint_psi_psi_factor_dir[k][0] * psi_row[c],
                                endpoint_psi_cov_factor_dir[k][1] * cov_row[c]
                                    + endpoint_psi_psi_factor_dir[k][1] * psi_row[c],
                            ];
                            let endpoint_psi_b_dir = [
                                endpoint_psi_cov_factor_dir[l][0] * cov_row[d]
                                    + endpoint_psi_psi_factor_dir[l][0] * psi_row[d],
                                endpoint_psi_cov_factor_dir[l][1] * cov_row[d]
                                    + endpoint_psi_psi_factor_dir[l][1] * psi_row[d],
                            ];
                            let (endpoint_ab, endpoint_psi_ab) = if same_shape {
                                (
                                    [
                                        2.0 * endpoint_basis[0][k] * cov_row[c] * cov_row[d],
                                        2.0 * endpoint_basis[1][k] * cov_row[c] * cov_row[d],
                                    ],
                                    [
                                        2.0 * endpoint_basis[0][k]
                                            * (psi_row[d] * cov_row[c] + psi_row[c] * cov_row[d]),
                                        2.0 * endpoint_basis[1][k]
                                            * (psi_row[d] * cov_row[c] + psi_row[c] * cov_row[d]),
                                    ],
                                )
                            } else {
                                ([0.0; 2], [0.0; 2])
                            };
                            let numerator = hppsi_a * hp_b + hp_a * hppsi_b;
                            let numerator_dir = hppsi_a_dir * hp_b
                                + hppsi_a * hp_b_dir
                                + hp_a_dir * hppsi_b
                                + hp_a * hppsi_b_dir;
                            let barrier_product = hp_a * hp_b * hp_psi;
                            let barrier_product_dir = hp_a_dir * hp_b * hp_psi
                                + hp_a * hp_b_dir * hp_psi
                                + hp_a * hp_b * hp_psi_dir;
                            let value = hpsi_a_dir * h_b
                                + hpsi_a * h_b_dir
                                + h_a_dir * hpsi_b
                                + h_a * hpsi_b_dir
                                + h_psi_dir * h_ab
                                + h_dir * hpsi_ab
                                + numerator_dir * inv_hp_sq
                                + numerator * d_inv_hp_sq
                                - 2.0
                                    * (barrier_product_dir * inv_hp_cu
                                        + barrier_product * d_inv_hp_cu)
                                - hppsi_ab * d_inv_hp
                                + hp_ab * hp_psi_dir * inv_hp_sq
                                + hp_ab * hp_psi * d_inv_hp_sq
                                + endpoint_chain_fourth(
                                    &q,
                                    endpoint_psi,
                                    endpoint_a,
                                    endpoint_b,
                                    endpoint_dir,
                                    endpoint_psi_a,
                                    endpoint_psi_b,
                                    endpoint_psi_dir,
                                    endpoint_ab,
                                    endpoint_a_dir,
                                    endpoint_b_dir,
                                    endpoint_psi_ab,
                                    endpoint_psi_a_dir,
                                    endpoint_psi_b_dir,
                                    [0.0; 2],
                                    [0.0; 2],
                                );
                            out[[row_idx, col_idx]] += wi * value;
                        }
                    }
                }
            }
        };

        // Drive the per-row work in parallel. Each Rayon thread accumulates
        // into its own Scratch (out + scratch buffers); we then reduce the
        // per-thread `out` matrices by summation. Only `out` needs reducing —
        // the other scratch fields are temporaries.
        let mut out: Array2<f64> = gam_linalg::pairwise_reduce::par_deterministic_block_fold(
            n,
            |range| {
                let mut scratch = init_scratch();
                for i in range {
                    process_row(&mut scratch, i);
                }
                scratch.out
            },
            |a, b| a + b,
        )
        .unwrap_or_else(|| Array2::<f64>::zeros((p_total, p_total)));

        // In-place symmetrization: avoid allocating an extra `p_total × p_total`
        // f64 matrix for `&out + &out.t()`. Mathematically identical to
        // `0.5 * (out + out.T)`.
        for i in 0..p_total {
            for j in (i + 1)..p_total {
                let s = 0.5 * (out[[i, j]] + out[[j, i]]);
                out[[i, j]] = s;
                out[[j, i]] = s;
            }
        }
        Ok(out)
    }

    pub(crate) fn scop_psi_hessian_directional_trace_factor_chunk_from_cov(
        &self,
        beta: &Array1<f64>,
        direction: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
        row_start: usize,
        cov: ArrayView2<'_, f64>,
        cov_psi: ArrayView2<'_, f64>,
        factor: ArrayView2<'_, f64>,
        projected_cov_f: Option<ArrayView2<'_, f64>>,
        projected_psi_f: Option<ArrayView2<'_, f64>>,
    ) -> Result<f64, String> {
        let total_n = self.response_val_basis.nrows();
        let n = cov.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        let rank = factor.ncols();
        if row_start > total_n || row_start + n > total_n {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi Hessian directional projected trace row window [{row_start}, {}) exceeds n={total_n}",
                row_start + n
            ) }.into());
        }
        if cov.ncols() != p_cov || cov_psi.nrows() != n || cov_psi.ncols() != p_cov {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi Hessian directional projected trace chunk shape mismatch: cov={}x{}, cov_psi={}x{}, expected n={} p_cov={}",
                cov.nrows(),
                cov.ncols(),
                cov_psi.nrows(),
                cov_psi.ncols(),
                n,
                p_cov
            ) }.into());
        }
        if beta.len() != p_total || direction.len() != p_total || factor.nrows() != p_total {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi Hessian directional projected trace length mismatch: beta={}, direction={}, factor_rows={}, expected={p_total}",
                beta.len(),
                direction.len(),
                factor.nrows()
            ) }.into());
        }
        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP psi directional trace beta reshape failed: {e}"))?;
        let dir_mat = direction
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP psi directional trace direction reshape failed: {e}"))?;
        let endpoint_basis = [
            self.response_upper_basis.as_slice().ok_or_else(|| {
                "SCOP psi directional trace endpoint upper basis is not contiguous".to_string()
            })?,
            self.response_lower_basis.as_slice().ok_or_else(|| {
                "SCOP psi directional trace endpoint lower basis is not contiguous".to_string()
            })?,
        ];

        struct PsiDhTraceAccum {
            pub(crate) value: f64,
            pub(crate) gamma: Vec<f64>,
            pub(crate) gamma_dir: Vec<f64>,
            pub(crate) gamma_psi: Vec<f64>,
            pub(crate) gamma_psi_dir: Vec<f64>,
            pub(crate) gamma_f: Vec<f64>,
            pub(crate) gamma_psi_f: Vec<f64>,
            pub(crate) h_f: Vec<f64>,
            pub(crate) hp_f: Vec<f64>,
            pub(crate) h_f_dir: Vec<f64>,
            pub(crate) hp_f_dir: Vec<f64>,
            pub(crate) h_ff: Vec<f64>,
            pub(crate) hp_ff: Vec<f64>,
            pub(crate) hpsi_f: Vec<f64>,
            pub(crate) hppsi_f: Vec<f64>,
            pub(crate) hpsi_f_dir: Vec<f64>,
            pub(crate) hppsi_f_dir: Vec<f64>,
            pub(crate) hpsi_ff: Vec<f64>,
            pub(crate) hppsi_ff: Vec<f64>,
            pub(crate) endpoint_f: Vec<[f64; 2]>,
            pub(crate) endpoint_f_dir: Vec<[f64; 2]>,
            pub(crate) endpoint_ff: Vec<[f64; 2]>,
            pub(crate) endpoint_psi_f: Vec<[f64; 2]>,
            pub(crate) endpoint_psi_f_dir: Vec<[f64; 2]>,
            pub(crate) endpoint_psi_ff: Vec<[f64; 2]>,
        }

        impl PsiDhTraceAccum {
            pub(crate) fn new(p_resp: usize, rank: usize) -> Self {
                let projected_len = p_resp * rank;
                Self {
                    value: 0.0,
                    gamma: vec![0.0; p_resp],
                    gamma_dir: vec![0.0; p_resp],
                    gamma_psi: vec![0.0; p_resp],
                    gamma_psi_dir: vec![0.0; p_resp],
                    gamma_f: vec![0.0; projected_len],
                    gamma_psi_f: vec![0.0; projected_len],
                    h_f: vec![0.0; rank],
                    hp_f: vec![0.0; rank],
                    h_f_dir: vec![0.0; rank],
                    hp_f_dir: vec![0.0; rank],
                    h_ff: vec![0.0; rank],
                    hp_ff: vec![0.0; rank],
                    hpsi_f: vec![0.0; rank],
                    hppsi_f: vec![0.0; rank],
                    hpsi_f_dir: vec![0.0; rank],
                    hppsi_f_dir: vec![0.0; rank],
                    hpsi_ff: vec![0.0; rank],
                    hppsi_ff: vec![0.0; rank],
                    endpoint_f: vec![[0.0; 2]; rank],
                    endpoint_f_dir: vec![[0.0; 2]; rank],
                    endpoint_ff: vec![[0.0; 2]; rank],
                    endpoint_psi_f: vec![[0.0; 2]; rank],
                    endpoint_psi_f_dir: vec![[0.0; 2]; rank],
                    endpoint_psi_ff: vec![[0.0; 2]; rank],
                }
            }

            pub(crate) fn merge(mut self, rhs: Self) -> Self {
                self.value += rhs.value;
                self
            }
        }

        // Project the low-rank REML factor through each response-slice of the
        // covariate design with BLAS instead of recomputing
        // `sum_c F[(k,c), r] * x_ic` inside every row. This is an exact
        // algebraic refactor of the old row loop: for each response basis `k`,
        // `projected_cov_f[:, k, :] = cov · factor_k` and
        // `projected_psi_f[:, k, :] = cov_psi · factor_k`. Callers that sweep
        // multiple ψ-Hessian directional traces may supply cached projections;
        // otherwise we build the chunk-local projections here.
        let projected_len = p_resp * rank;
        let mut projected_cov_storage;
        let mut projected_psi_storage;
        let projected_cov_f = match projected_cov_f {
            Some(view) => {
                if view.nrows() != n || view.ncols() != projected_len {
                    return Err(format!(
                        "SCOP psi Hessian directional projected cov-factor shape {}x{} != expected {}x{}",
                        view.nrows(),
                        view.ncols(),
                        n,
                        projected_len
                    ));
                }
                view
            }
            None => {
                projected_cov_storage = Array2::<f64>::zeros((n, projected_len));
                if rank > 0 && n > 0 {
                    for k in 0..p_resp {
                        let factor_block = factor.slice(s![k * p_cov..(k + 1) * p_cov, ..]);
                        let cov_projection = fast_ab(&cov, &factor_block);
                        projected_cov_storage
                            .slice_mut(s![.., k * rank..(k + 1) * rank])
                            .assign(&cov_projection);
                    }
                }
                projected_cov_storage.view()
            }
        };
        let projected_psi_f = match projected_psi_f {
            Some(view) => {
                if view.nrows() != n || view.ncols() != projected_len {
                    return Err(format!(
                        "SCOP psi Hessian directional projected psi-factor shape {}x{} != expected {}x{}",
                        view.nrows(),
                        view.ncols(),
                        n,
                        projected_len
                    ));
                }
                view
            }
            None => {
                projected_psi_storage = Array2::<f64>::zeros((n, projected_len));
                if rank > 0 && n > 0 {
                    for k in 0..p_resp {
                        let factor_block = factor.slice(s![k * p_cov..(k + 1) * p_cov, ..]);
                        let psi_projection = fast_ab(&cov_psi, &factor_block);
                        projected_psi_storage
                            .slice_mut(s![.., k * rank..(k + 1) * rank])
                            .assign(&psi_projection);
                    }
                }
                projected_psi_storage.view()
            }
        };

        let weights = self.effective_weights();
        let h_prime = row_quantities.h_prime.as_ref();
        let accum = gam_linalg::pairwise_reduce::par_deterministic_block_fold(
            n,
            |range| {
                let mut acc = PsiDhTraceAccum::new(p_resp, rank);
                for local_i in range {
                    let i = row_start + local_i;
                    let cov_row = cov.row(local_i);
                    let psi_row = cov_psi.row(local_i);
                    let rv = self.response_val_basis.row(i);
                    let rd = self.response_deriv_basis.row(i);
                    let wi = weights[i];
                    let hp = h_prime[i];
                    let inv_hp = 1.0 / hp;
                    let inv_hp_sq = inv_hp * inv_hp;
                    let inv_hp_cu = inv_hp_sq * inv_hp;
                    let inv_hp_qu = inv_hp_sq * inv_hp_sq;
                    let q = row_quantities.endpoint_q[i];

                    for k in 0..p_resp {
                        acc.gamma[k] = beta_mat.row(k).dot(&cov_row);
                        acc.gamma_dir[k] = dir_mat.row(k).dot(&cov_row);
                        acc.gamma_psi[k] = beta_mat.row(k).dot(&psi_row);
                        acc.gamma_psi_dir[k] = dir_mat.row(k).dot(&psi_row);
                    }

                    let projected_cov_row = projected_cov_f.row(local_i);
                    let projected_psi_row = projected_psi_f.row(local_i);
                    acc.gamma_f.copy_from_slice(
                        projected_cov_row
                            .as_slice()
                            .expect("projected CTN covariate-factor row should be contiguous"),
                    );
                    acc.gamma_psi_f.copy_from_slice(
                        projected_psi_row
                            .as_slice()
                            .expect("projected CTN psi-factor row should be contiguous"),
                    );

                    let mut hp_psi = rd[0] * acc.gamma_psi[0];
                    let mut h_dir = rv[0] * acc.gamma_dir[0];
                    let mut hp_dir = rd[0] * acc.gamma_dir[0];
                    let mut h_psi_dir = rv[0] * acc.gamma_psi_dir[0];
                    let mut hp_psi_dir = rd[0] * acc.gamma_psi_dir[0];
                    for k in 1..p_resp {
                        hp_psi += 2.0 * rd[k] * acc.gamma[k] * acc.gamma_psi[k];
                        h_dir += 2.0 * rv[k] * acc.gamma[k] * acc.gamma_dir[k];
                        hp_dir += 2.0 * rd[k] * acc.gamma[k] * acc.gamma_dir[k];
                        h_psi_dir += 2.0
                            * rv[k]
                            * (acc.gamma_dir[k] * acc.gamma_psi[k]
                                + acc.gamma[k] * acc.gamma_psi_dir[k]);
                        hp_psi_dir += 2.0
                            * rd[k]
                            * (acc.gamma_dir[k] * acc.gamma_psi[k]
                                + acc.gamma[k] * acc.gamma_psi_dir[k]);
                    }
                    let d_inv_hp = -hp_dir * inv_hp_sq;
                    let d_inv_hp_sq = -2.0 * hp_dir * inv_hp_cu;
                    let d_inv_hp_cu = -3.0 * hp_dir * inv_hp_qu;

                    let mut endpoint_psi = [0.0_f64; 2];
                    let mut endpoint_dir = [0.0_f64; 2];
                    let mut endpoint_psi_dir = [0.0_f64; 2];
                    for e in 0..2 {
                        let basis = endpoint_basis[e];
                        endpoint_psi[e] = basis[0] * acc.gamma_psi[0];
                        endpoint_dir[e] = basis[0] * acc.gamma_dir[0];
                        endpoint_psi_dir[e] = basis[0] * acc.gamma_psi_dir[0];
                        for k in 1..p_resp {
                            endpoint_psi[e] += 2.0 * basis[k] * acc.gamma[k] * acc.gamma_psi[k];
                            endpoint_dir[e] += 2.0 * basis[k] * acc.gamma[k] * acc.gamma_dir[k];
                            endpoint_psi_dir[e] += 2.0
                                * basis[k]
                                * (acc.gamma_dir[k] * acc.gamma_psi[k]
                                    + acc.gamma[k] * acc.gamma_psi_dir[k]);
                        }
                    }

                    for col in 0..rank {
                        acc.h_f[col] = rv[0] * acc.gamma_f[col];
                        acc.hp_f[col] = rd[0] * acc.gamma_f[col];
                        acc.h_f_dir[col] = 0.0;
                        acc.hp_f_dir[col] = 0.0;
                        acc.h_ff[col] = 0.0;
                        acc.hp_ff[col] = 0.0;
                        acc.hpsi_f[col] = rv[0] * acc.gamma_psi_f[col];
                        acc.hppsi_f[col] = rd[0] * acc.gamma_psi_f[col];
                        acc.hpsi_f_dir[col] = 0.0;
                        acc.hppsi_f_dir[col] = 0.0;
                        acc.hpsi_ff[col] = 0.0;
                        acc.hppsi_ff[col] = 0.0;
                        acc.endpoint_f[col] = [
                            endpoint_basis[0][0] * acc.gamma_f[col],
                            endpoint_basis[1][0] * acc.gamma_f[col],
                        ];
                        acc.endpoint_f_dir[col] = [0.0; 2];
                        acc.endpoint_ff[col] = [0.0; 2];
                        acc.endpoint_psi_f[col] = [
                            endpoint_basis[0][0] * acc.gamma_psi_f[col],
                            endpoint_basis[1][0] * acc.gamma_psi_f[col],
                        ];
                        acc.endpoint_psi_f_dir[col] = [0.0; 2];
                        acc.endpoint_psi_ff[col] = [0.0; 2];
                    }
                    for k in 1..p_resp {
                        let g = acc.gamma[k];
                        let gd = acc.gamma_dir[k];
                        let gp = acc.gamma_psi[k];
                        let gpd = acc.gamma_psi_dir[k];
                        for col in 0..rank {
                            let idx = k * rank + col;
                            let gf = acc.gamma_f[idx];
                            let gpf = acc.gamma_psi_f[idx];
                            acc.h_f[col] += 2.0 * rv[k] * g * gf;
                            acc.hp_f[col] += 2.0 * rd[k] * g * gf;
                            acc.h_f_dir[col] += 2.0 * rv[k] * gd * gf;
                            acc.hp_f_dir[col] += 2.0 * rd[k] * gd * gf;
                            acc.h_ff[col] += 2.0 * rv[k] * gf * gf;
                            acc.hp_ff[col] += 2.0 * rd[k] * gf * gf;
                            acc.hpsi_f[col] += 2.0 * rv[k] * (gf * gp + g * gpf);
                            acc.hppsi_f[col] += 2.0 * rd[k] * (gf * gp + g * gpf);
                            acc.hpsi_f_dir[col] += 2.0 * rv[k] * (gf * gpd + gd * gpf);
                            acc.hppsi_f_dir[col] += 2.0 * rd[k] * (gf * gpd + gd * gpf);
                            acc.hpsi_ff[col] += 4.0 * rv[k] * gf * gpf;
                            acc.hppsi_ff[col] += 4.0 * rd[k] * gf * gpf;
                            for e in 0..2 {
                                let basis = endpoint_basis[e];
                                acc.endpoint_f[col][e] += 2.0 * basis[k] * g * gf;
                                acc.endpoint_f_dir[col][e] += 2.0 * basis[k] * gd * gf;
                                acc.endpoint_ff[col][e] += 2.0 * basis[k] * gf * gf;
                                acc.endpoint_psi_f[col][e] += 2.0 * basis[k] * (gf * gp + g * gpf);
                                acc.endpoint_psi_f_dir[col][e] +=
                                    2.0 * basis[k] * (gf * gpd + gd * gpf);
                                acc.endpoint_psi_ff[col][e] += 4.0 * basis[k] * gf * gpf;
                            }
                        }
                    }

                    for col in 0..rank {
                        let numerator = 2.0 * acc.hppsi_f[col] * acc.hp_f[col];
                        let numerator_dir = 2.0
                            * (acc.hppsi_f_dir[col] * acc.hp_f[col]
                                + acc.hppsi_f[col] * acc.hp_f_dir[col]);
                        let barrier_product = acc.hp_f[col] * acc.hp_f[col] * hp_psi;
                        let barrier_product_dir = 2.0 * acc.hp_f_dir[col] * acc.hp_f[col] * hp_psi
                            + acc.hp_f[col] * acc.hp_f[col] * hp_psi_dir;
                        let value = 2.0 * acc.hpsi_f_dir[col] * acc.h_f[col]
                            + 2.0 * acc.hpsi_f[col] * acc.h_f_dir[col]
                            + h_psi_dir * acc.h_ff[col]
                            + h_dir * acc.hpsi_ff[col]
                            + numerator_dir * inv_hp_sq
                            + numerator * d_inv_hp_sq
                            - 2.0
                                * (barrier_product_dir * inv_hp_cu + barrier_product * d_inv_hp_cu)
                            - acc.hppsi_ff[col] * d_inv_hp
                            + acc.hp_ff[col] * hp_psi_dir * inv_hp_sq
                            + acc.hp_ff[col] * hp_psi * d_inv_hp_sq
                            + endpoint_chain_fourth(
                                &q,
                                endpoint_psi,
                                acc.endpoint_f[col],
                                acc.endpoint_f[col],
                                endpoint_dir,
                                acc.endpoint_psi_f[col],
                                acc.endpoint_psi_f[col],
                                endpoint_psi_dir,
                                acc.endpoint_ff[col],
                                acc.endpoint_f_dir[col],
                                acc.endpoint_f_dir[col],
                                acc.endpoint_psi_ff[col],
                                acc.endpoint_psi_f_dir[col],
                                acc.endpoint_psi_f_dir[col],
                                [0.0; 2],
                                [0.0; 2],
                            );
                        acc.value += wi * value;
                    }
                }
                acc
            },
            |left, right| left.merge(right),
        )
        .unwrap_or_else(|| PsiDhTraceAccum::new(p_resp, rank));
        Ok(accum.value)
    }
}
