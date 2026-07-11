use super::*;

impl TransformationNormalFamily {
    pub(crate) fn scop_gradient_and_negative_hessian(
        &self,
        beta: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
    ) -> Result<(Array1<f64>, Array2<f64>), String> {
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if beta.len() != p_total {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP gradient beta length {} != p_resp({p_resp}) * p_cov({p_cov})",
                    beta.len()
                ),
            }
            .into());
        }
        if !row_quantities.matches_beta(beta) {
            return Err(
                "SCOP gradient/Hessian received row quantities for a different beta".to_string(),
            );
        }
        if !row_quantities.matches_beta(beta) {
            return Err(
                "SCOP gradient/Hessian received row quantities for a different beta".to_string(),
            );
        }
        let cov = self
            .covariate_dense_arc()
            .map_err(|e| format!("SCOP gradient requires cached covariate design: {e}"))?;
        let weights = self.effective_weights();
        let h = row_quantities.h.as_ref();
        let h_prime = row_quantities.h_prime.as_ref();
        let endpoint_q = row_quantities.endpoint_q.as_ref();
        let gamma_rows = row_quantities.gamma.as_ref();
        if gamma_rows.nrows() != n || gamma_rows.ncols() != p_resp {
            return Err(format!(
                "SCOP gradient/Hessian gamma cache shape mismatch: got {}x{}, expected {}x{}",
                gamma_rows.nrows(),
                gamma_rows.ncols(),
                n,
                p_resp
            ));
        }
        let response_val_basis = &self.response_val_basis;
        let response_deriv_basis = &self.response_deriv_basis;
        let response_lower_basis = &self.response_lower_basis;
        let response_upper_basis = &self.response_upper_basis;

        struct ScopAccum {
            pub(crate) gradient: Array1<f64>,
            pub(crate) hessian: Array2<f64>,
        }

        impl ScopAccum {
            pub(crate) fn new(p_total: usize) -> Self {
                Self {
                    gradient: Array1::<f64>::zeros(p_total),
                    hessian: Array2::<f64>::zeros((p_total, p_total)),
                }
            }
        }

        let policy = ResourcePolicy::default_library();
        let accum_bytes = p_total
            .saturating_mul(p_total.saturating_add(1))
            .saturating_mul(std::mem::size_of::<f64>())
            .max(1);
        let memory_bound_chunks = (policy.max_single_materialization_bytes / accum_bytes).max(1);
        let target_chunks = rayon::current_num_threads()
            .saturating_mul(4)
            .max(1)
            .min(memory_bound_chunks)
            .min(n.max(1));
        let chunk_rows = n.max(1).div_ceil(target_chunks);
        let row_chunks: Vec<(usize, usize)> = (0..n)
            .step_by(chunk_rows)
            .map(|start| (start, (start + chunk_rows).min(n)))
            .collect();

        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        // Rayon collects this indexed iterator into `Vec` in row-chunk order;
        // the final serial fold below preserves that order so results do not
        // depend on worker scheduling.
        let partials: Vec<ScopAccum> = row_chunks
            .into_par_iter()
            .map(|(start, end)| {
                let mut acc = ScopAccum::new(p_total);
                let mut dh_factor = vec![0.0; p_resp];
                let mut dhp_factor = vec![0.0; p_resp];
                let mut second_diag = vec![0.0; p_resp];
                let mut lower_factor = vec![0.0; p_resp];
                let mut upper_factor = vec![0.0; p_resp];

                for i in start..end {
                    let cov_row = cov.row(i);
                    let rv = response_val_basis.row(i);
                    let rd = response_deriv_basis.row(i);
                    let gamma = gamma_rows.row(i);
                    let wi = weights[i];
                    let hi = h[i];
                    let hp = h_prime[i];
                    let inv_hp = 1.0 / hp;
                    let inv_hp_sq = inv_hp * inv_hp;

                    let q = endpoint_q[i];
                    lower_factor[0] = response_lower_basis[0];
                    upper_factor[0] = response_upper_basis[0];
                    for k in 1..p_resp {
                        lower_factor[k] = 2.0 * response_lower_basis[k] * gamma[k];
                        upper_factor[k] = 2.0 * response_upper_basis[k] * gamma[k];
                    }

                    second_diag.fill(0.0);
                    dh_factor[0] = rv[0];
                    dhp_factor[0] = rd[0];
                    for k in 1..p_resp {
                        dh_factor[k] = 2.0 * rv[k] * gamma[k];
                        dhp_factor[k] = 2.0 * rd[k] * gamma[k];
                        second_diag[k] = 2.0 * (hi * rv[k] - rd[k] * inv_hp);
                    }

                    for k in 0..p_resp {
                        let normalizer_score_factor =
                            q.first[0] * upper_factor[k] + q.first[1] * lower_factor[k];
                        let score_factor = wi
                            * (-hi * dh_factor[k] + dhp_factor[k] * inv_hp
                                - normalizer_score_factor);
                        for c in 0..p_cov {
                            acc.gradient[k * p_cov + c] += score_factor * cov_row[c];
                        }
                    }

                    for k in 0..p_resp {
                        for l in 0..p_resp {
                            let mut block_factor = dh_factor[k] * dh_factor[l]
                                + dhp_factor[k] * dhp_factor[l] * inv_hp_sq;
                            if k == l {
                                block_factor += second_diag[k];
                            }
                            let upper_ab = if k == l && k > 0 {
                                2.0 * response_upper_basis[k]
                            } else {
                                0.0
                            };
                            let lower_ab = if k == l && k > 0 {
                                2.0 * response_lower_basis[k]
                            } else {
                                0.0
                            };
                            block_factor += q.first[0] * upper_ab
                                + q.first[1] * lower_ab
                                + q.second[0][0] * upper_factor[k] * upper_factor[l]
                                + q.second[0][1] * upper_factor[k] * lower_factor[l]
                                + q.second[1][0] * lower_factor[k] * upper_factor[l]
                                + q.second[1][1] * lower_factor[k] * lower_factor[l];
                            block_factor *= wi;
                            if block_factor == 0.0 {
                                continue;
                            }
                            for c in 0..p_cov {
                                let row_idx = k * p_cov + c;
                                let left = block_factor * cov_row[c];
                                for d in 0..p_cov {
                                    acc.hessian[[row_idx, l * p_cov + d]] += left * cov_row[d];
                                }
                            }
                        }
                    }
                }

                acc
            })
            .collect();

        let mut gradient = Array1::<f64>::zeros(p_total);
        let mut hessian = Array2::<f64>::zeros((p_total, p_total));
        for partial in partials {
            gradient.scaled_add(1.0, &partial.gradient);
            hessian.scaled_add(1.0, &partial.hessian);
        }

        Ok((gradient, hessian))
    }

    pub(crate) fn scop_gradient(
        &self,
        beta: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
    ) -> Result<Array1<f64>, String> {
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if beta.len() != p_total {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP gradient beta length {} != p_resp({p_resp}) * p_cov({p_cov})",
                    beta.len()
                ),
            }
            .into());
        }
        if !row_quantities.matches_beta(beta) {
            return Err(TransformationNormalError::InvalidInput {
                reason: "SCOP gradient received row quantities for a different beta".to_string(),
            }
            .into());
        }
        if !row_quantities.matches_beta(beta) {
            return Err("SCOP gradient received row quantities for a different beta".to_string());
        }
        let cov = self
            .covariate_dense_arc()
            .map_err(|e| format!("SCOP gradient requires cached covariate design: {e}"))?;
        let weights = self.effective_weights();
        let h = row_quantities.h.as_ref();
        let h_prime = row_quantities.h_prime.as_ref();
        let gamma_rows = row_quantities.gamma.as_ref();
        if gamma_rows.nrows() != n || gamma_rows.ncols() != p_resp {
            return Err(format!(
                "SCOP gradient gamma cache shape mismatch: got {}x{}, expected {}x{}",
                gamma_rows.nrows(),
                gamma_rows.ncols(),
                n,
                p_resp
            ));
        }
        let mut gradient = Array1::<f64>::zeros(p_total);
        let mut lower_factor = vec![0.0; p_resp];
        let mut upper_factor = vec![0.0; p_resp];

        for i in 0..n {
            let cov_row = cov.row(i);
            let rv = self.response_val_basis.row(i);
            let rd = self.response_deriv_basis.row(i);
            let gamma = gamma_rows.row(i);
            let wi = weights[i];
            let hi = h[i];
            let inv_hp = 1.0 / h_prime[i];

            let q = row_quantities.endpoint_q[i];
            lower_factor[0] = self.response_lower_basis[0];
            upper_factor[0] = self.response_upper_basis[0];
            for k in 1..p_resp {
                lower_factor[k] = 2.0 * self.response_lower_basis[k] * gamma[k];
                upper_factor[k] = 2.0 * self.response_upper_basis[k] * gamma[k];
            }

            let normalizer_score0 = q.first[0] * upper_factor[0] + q.first[1] * lower_factor[0];
            let score0 = wi * (-hi * rv[0] + rd[0] * inv_hp - normalizer_score0);
            for c in 0..p_cov {
                gradient[c] += score0 * cov_row[c];
            }
            for k in 1..p_resp {
                let normalizer_score = q.first[0] * upper_factor[k] + q.first[1] * lower_factor[k];
                let score_factor =
                    wi * (2.0 * gamma[k] * (-hi * rv[k] + rd[k] * inv_hp) - normalizer_score);
                let offset = k * p_cov;
                for c in 0..p_cov {
                    gradient[offset + c] += score_factor * cov_row[c];
                }
            }
        }

        Ok(gradient)
    }

    /// Directional derivative of the SCOP negative Hessian.
    ///
    /// For one observation with covariate row `c`, response value row `r`, and
    /// derivative row `m`, define `g_k = beta_k·c`, `u_k = direction_k·c`,
    ///
    /// `h  = r_0 g_0 + Σ_{k>0} r_k g_k^2`
    /// `hp = m_0 g_0 + Σ_{k>0} m_k g_k^2`
    /// `Hu  = r_0 u_0 + Σ_{k>0} 2 r_k g_k u_k`
    /// `HPu = m_0 u_0 + Σ_{k>0} 2 m_k g_k u_k`.
    ///
    /// For coefficient `a=(k,p)`:
    ///
    /// `h_a  = r_k c_p` for `k=0`, else `2 r_k g_k c_p`
    /// `hp_a = m_k c_p` for `k=0`, else `2 m_k g_k c_p`
    /// `D h_a[u]  = 0` for `k=0`, else `2 r_k u_k c_p`
    /// `D hp_a[u] = 0` for `k=0`, else `2 m_k u_k c_p`.
    ///
    /// The per-row negative Hessian block is
    ///
    /// `H_ab = w [ h_a h_b + h h_ab + hp_a hp_b / hp^2 - hp_ab / hp ]`,
    ///
    /// where `h_ab = 2 r_k c_p c_q` and `hp_ab = 2 m_k c_p c_q` only when
    /// `a` and `b` are in the same squared shape component `k>0`; otherwise
    /// both second derivatives are zero. Differentiating this expression in
    /// direction `u` gives the exact formula implemented below:
    ///
    /// `D H_ab[u] = w [ (D h_a)h_b + h_a(D h_b) + Hu h_ab
    ///              + ((D hp_a)hp_b + hp_a(D hp_b))/hp^2
    ///              - 2 hp_a hp_b HPu/hp^3 + hp_ab HPu/hp^2 ]`.
    pub(crate) fn scop_hessian_directional_derivative(
        &self,
        beta: &Array1<f64>,
        direction: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
    ) -> Result<Array2<f64>, String> {
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if beta.len() != p_total || direction.len() != p_total {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP Hessian directional derivative length mismatch: beta={}, direction={}, expected={p_total}",
                beta.len(),
                direction.len()
            ) }.into());
        }
        if !row_quantities.matches_beta(beta) {
            return Err(
                "SCOP Hessian directional derivative received row quantities for a different beta"
                    .to_string(),
            );
        }
        let dir_mat = direction
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP direction reshape failed: {e}"))?;
        let cov = self.covariate_dense_arc().map_err(|e| {
            format!("SCOP Hessian directional derivative requires cached covariate design: {e}")
        })?;
        let weights = self.effective_weights();
        let h_prime = row_quantities.h_prime.as_ref();
        let gamma_rows = row_quantities.gamma.as_ref();
        if gamma_rows.nrows() != n || gamma_rows.ncols() != p_resp {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP Hessian directional derivative gamma cache shape mismatch: got {}x{}, expected {}x{}",
                gamma_rows.nrows(),
                gamma_rows.ncols(),
                n,
                p_resp
            ) }.into());
        }
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        const TARGET_CHUNK_COUNT: usize = 32;
        let chunk_size = n.div_ceil(TARGET_CHUNK_COUNT).max(1);
        let n_chunks = n.div_ceil(chunk_size);

        // Fixed row chunks give each rayon worker a private matrix accumulator while
        // preserving chunk-index order for deterministic floating-point reduction.
        let chunk_outputs: Vec<Array2<f64>> = (0..n_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * chunk_size;
                let end = (start + chunk_size).min(n);
                let mut chunk_out = Array2::<f64>::zeros((p_total, p_total));

                // Hoist per-row scratch buffers above the row loop; each iteration
                // overwrites them after a `.fill(0.0)` of the slots whose `k=0`
                // entry stays at zero (the k>=1 entries are fully overwritten).
                let mut gamma_dir = vec![0.0; p_resp];
                let mut h_factor = vec![0.0; p_resp];
                let mut hp_factor = vec![0.0; p_resp];
                let mut h_factor_dir = vec![0.0; p_resp];
                let mut hp_factor_dir = vec![0.0; p_resp];
                let mut endpoint_factor_0 = vec![0.0; p_resp];
                let mut endpoint_factor_1 = vec![0.0; p_resp];
                let mut endpoint_factor_dir_0 = vec![0.0; p_resp];
                let mut endpoint_factor_dir_1 = vec![0.0; p_resp];

                for i in start..end {
                    let cov_row = cov.row(i);
                    let rv = self.response_val_basis.row(i);
                    let rd = self.response_deriv_basis.row(i);
                    let wi = weights[i];
                    let hp = h_prime[i];
                    let inv_hp = 1.0 / hp;
                    let inv_hp_sq = inv_hp * inv_hp;
                    let inv_hp_cu = inv_hp_sq * inv_hp;

                    let gamma = gamma_rows.row(i);
                    for k in 0..p_resp {
                        gamma_dir[k] = dir_mat.row(k).dot(&cov_row);
                    }

                    let mut h_dir = rv[0] * gamma_dir[0];
                    let mut hp_dir = rd[0] * gamma_dir[0];
                    let mut endpoint_dir = [
                        self.response_upper_basis[0] * gamma_dir[0],
                        self.response_lower_basis[0] * gamma_dir[0],
                    ];
                    for k in 1..p_resp {
                        h_dir += 2.0 * rv[k] * gamma[k] * gamma_dir[k];
                        hp_dir += 2.0 * rd[k] * gamma[k] * gamma_dir[k];
                        endpoint_dir[0] +=
                            2.0 * self.response_upper_basis[k] * gamma[k] * gamma_dir[k];
                        endpoint_dir[1] +=
                            2.0 * self.response_lower_basis[k] * gamma[k] * gamma_dir[k];
                    }
                    let q = row_quantities.endpoint_q[i];

                    // Reset the per-row scratch arrays. k=0 is set explicitly below;
                    // the *_dir factors only ever store k>=1 values so their k=0 slot
                    // must stay zero.
                    h_factor_dir[0] = 0.0;
                    hp_factor_dir[0] = 0.0;
                    endpoint_factor_dir_0[0] = 0.0;
                    endpoint_factor_dir_1[0] = 0.0;
                    h_factor[0] = rv[0];
                    hp_factor[0] = rd[0];
                    endpoint_factor_0[0] = self.response_upper_basis[0];
                    endpoint_factor_1[0] = self.response_lower_basis[0];
                    for k in 1..p_resp {
                        h_factor[k] = 2.0 * rv[k] * gamma[k];
                        hp_factor[k] = 2.0 * rd[k] * gamma[k];
                        h_factor_dir[k] = 2.0 * rv[k] * gamma_dir[k];
                        hp_factor_dir[k] = 2.0 * rd[k] * gamma_dir[k];
                        endpoint_factor_0[k] = 2.0 * self.response_upper_basis[k] * gamma[k];
                        endpoint_factor_1[k] = 2.0 * self.response_lower_basis[k] * gamma[k];
                        endpoint_factor_dir_0[k] =
                            2.0 * self.response_upper_basis[k] * gamma_dir[k];
                        endpoint_factor_dir_1[k] =
                            2.0 * self.response_lower_basis[k] * gamma_dir[k];
                    }
                    let endpoint_factor = [&endpoint_factor_0[..], &endpoint_factor_1[..]];
                    let endpoint_factor_dir =
                        [&endpoint_factor_dir_0[..], &endpoint_factor_dir_1[..]];

                    for k in 0..p_resp {
                        for l in 0..p_resp {
                            let same_shape = k == l && k > 0;
                            let mut normalizer_block = 0.0;
                            for a in 0..2 {
                                let h_a_ab = if same_shape {
                                    2.0 * if a == 0 {
                                        self.response_upper_basis[k]
                                    } else {
                                        self.response_lower_basis[k]
                                    }
                                } else {
                                    0.0
                                };
                                for b in 0..2 {
                                    normalizer_block += q.second[a][b] * endpoint_dir[b] * h_a_ab;
                                    normalizer_block += q.second[a][b]
                                        * (endpoint_factor_dir[a][k] * endpoint_factor[b][l]
                                            + endpoint_factor[a][k] * endpoint_factor_dir[b][l]);
                                    for c_ep in 0..2 {
                                        normalizer_block += q.third[a][b][c_ep]
                                            * endpoint_dir[c_ep]
                                            * endpoint_factor[a][k]
                                            * endpoint_factor[b][l];
                                    }
                                }
                            }
                            for c in 0..p_cov {
                                let row_idx = k * p_cov + c;
                                let h_a = h_factor[k] * cov_row[c];
                                let hp_a = hp_factor[k] * cov_row[c];
                                let dh_a = h_factor_dir[k] * cov_row[c];
                                let dhp_a = hp_factor_dir[k] * cov_row[c];
                                for d in 0..p_cov {
                                    let col_idx = l * p_cov + d;
                                    let h_b = h_factor[l] * cov_row[d];
                                    let hp_b = hp_factor[l] * cov_row[d];
                                    let dh_b = h_factor_dir[l] * cov_row[d];
                                    let dhp_b = hp_factor_dir[l] * cov_row[d];
                                    let (h_ab, hp_ab) = if same_shape {
                                        (
                                            2.0 * rv[k] * cov_row[c] * cov_row[d],
                                            2.0 * rd[k] * cov_row[c] * cov_row[d],
                                        )
                                    } else {
                                        (0.0, 0.0)
                                    };
                                    let value = dh_a * h_b
                                        + h_a * dh_b
                                        + h_dir * h_ab
                                        + (dhp_a * hp_b + hp_a * dhp_b) * inv_hp_sq
                                        - 2.0 * hp_a * hp_b * hp_dir * inv_hp_cu
                                        + hp_ab * hp_dir * inv_hp_sq
                                        + normalizer_block * cov_row[c] * cov_row[d];
                                    chunk_out[[row_idx, col_idx]] += wi * value;
                                }
                            }
                        }
                    }
                }

                chunk_out
            })
            .collect();

        let mut out = Array2::<f64>::zeros((p_total, p_total));
        for chunk in chunk_outputs {
            out.scaled_add(1.0, &chunk);
        }

        Ok(0.5 * (&out + &out.t()))
    }

    /// Second directional derivative of the SCOP negative Hessian.
    ///
    /// Reuse the notation from `scop_hessian_directional_derivative`, with
    /// two beta-space directions `u` and `v`. Since each shape component is
    /// quadratic in `g_k`, all third beta derivatives of `h` and `hp` are zero,
    /// while the mixed second directional derivatives are
    ///
    /// `Huv  = Σ_{k>0} 2 r_k u_k v_k`
    /// `HPuv = Σ_{k>0} 2 m_k u_k v_k`.
    ///
    /// Differentiating
    /// `D H_ab[u] = w[(D h_a)h_b + h_a(D h_b) + Hu h_ab
    ///              + ((D hp_a)hp_b + hp_a(D hp_b))/hp^2
    ///              - 2 hp_a hp_b HPu/hp^3 + hp_ab HPu/hp^2]`
    /// once more in direction `v` gives the expression implemented below:
    ///
    /// `D²H_ab[u,v] = w[
    ///      h_{a,u} h_{b,v} + h_{a,v} h_{b,u} + Huv h_ab
    ///    + (hp_{a,u} hp_{b,v} + hp_{a,v} hp_{b,u})/hp²
    ///    - 2(hp_{a,u} hp_b + hp_a hp_{b,u}) HPv/hp³
    ///    - 2(hp_{a,v} hp_b + hp_a hp_{b,v}) HPu/hp³
    ///    - 2 hp_a hp_b HPuv/hp³
    ///    + 6 hp_a hp_b HPu HPv/hp⁴
    ///    + hp_ab HPuv/hp²
    ///    - 2 hp_ab HPu HPv/hp³ ]`.
    pub(crate) fn scop_hessian_second_directional_derivative(
        &self,
        beta: &Array1<f64>,
        direction_u: &Array1<f64>,
        direction_v: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
    ) -> Result<Array2<f64>, String> {
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if beta.len() != p_total || direction_u.len() != p_total || direction_v.len() != p_total {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP Hessian second directional derivative length mismatch: beta={}, u={}, v={}, expected={p_total}",
                beta.len(),
                direction_u.len(),
                direction_v.len()
            ) }.into());
        }
        if !row_quantities.matches_beta(beta) {
            return Err(
                "SCOP Hessian second directional derivative received row quantities for a different beta"
                    .to_string(),
            );
        }
        let dir_u_mat = direction_u
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP u direction reshape failed: {e}"))?;
        let dir_v_mat = direction_v
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP v direction reshape failed: {e}"))?;
        let cov = self.covariate_dense_arc().map_err(|e| {
            format!(
                "SCOP Hessian second directional derivative requires cached covariate design: {e}"
            )
        })?;
        let weights = self.effective_weights();
        let h_prime = row_quantities.h_prime.as_ref();
        let gamma_rows = row_quantities.gamma.as_ref();
        if gamma_rows.nrows() != n || gamma_rows.ncols() != p_resp {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP Hessian second directional derivative gamma cache shape mismatch: got {}x{}, expected {}x{}",
                gamma_rows.nrows(),
                gamma_rows.ncols(),
                n,
                p_resp
            ) }.into());
        }
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        const TARGET_CHUNK_COUNT: usize = 32;
        let chunk_size = n.div_ceil(TARGET_CHUNK_COUNT).max(1);
        let n_chunks = n.div_ceil(chunk_size);

        // Fixed row chunks give each rayon worker a private matrix accumulator while
        // preserving chunk-index order for deterministic floating-point reduction.
        let chunk_outputs: Vec<Array2<f64>> = (0..n_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * chunk_size;
                let end = (start + chunk_size).min(n);
                let mut chunk_out = Array2::<f64>::zeros((p_total, p_total));

                // Hoist per-row scratch buffers above the row loop; each iteration
                // fully overwrites the entries it reads (k>=1 entries reset by
                // arithmetic, k=0 entries reassigned below).
                let mut gamma_u = vec![0.0; p_resp];
                let mut gamma_v = vec![0.0; p_resp];
                let mut h_factor = vec![0.0; p_resp];
                let mut hp_factor = vec![0.0; p_resp];
                let mut h_factor_u = vec![0.0; p_resp];
                let mut hp_factor_u = vec![0.0; p_resp];
                let mut h_factor_v = vec![0.0; p_resp];
                let mut hp_factor_v = vec![0.0; p_resp];
                let mut endpoint_factor_0 = vec![0.0; p_resp];
                let mut endpoint_factor_1 = vec![0.0; p_resp];
                let mut endpoint_factor_u_0 = vec![0.0; p_resp];
                let mut endpoint_factor_u_1 = vec![0.0; p_resp];
                let mut endpoint_factor_v_0 = vec![0.0; p_resp];
                let mut endpoint_factor_v_1 = vec![0.0; p_resp];

                for i in start..end {
                    let cov_row = cov.row(i);
                    let rv = self.response_val_basis.row(i);
                    let rd = self.response_deriv_basis.row(i);
                    let wi = weights[i];
                    let hp = h_prime[i];
                    let inv_hp = 1.0 / hp;
                    let inv_hp_sq = inv_hp * inv_hp;
                    let inv_hp_cu = inv_hp_sq * inv_hp;
                    let inv_hp_qu = inv_hp_sq * inv_hp_sq;

                    let gamma = gamma_rows.row(i);
                    for k in 0..p_resp {
                        gamma_u[k] = dir_u_mat.row(k).dot(&cov_row);
                        gamma_v[k] = dir_v_mat.row(k).dot(&cov_row);
                    }

                    let mut hp_u = rd[0] * gamma_u[0];
                    let mut hp_v = rd[0] * gamma_v[0];
                    let mut h_uv = 0.0;
                    let mut hp_uv = 0.0;
                    let mut endpoint_u = [
                        self.response_upper_basis[0] * gamma_u[0],
                        self.response_lower_basis[0] * gamma_u[0],
                    ];
                    let mut endpoint_v = [
                        self.response_upper_basis[0] * gamma_v[0],
                        self.response_lower_basis[0] * gamma_v[0],
                    ];
                    let mut endpoint_uv = [0.0, 0.0];
                    for k in 1..p_resp {
                        hp_u += 2.0 * rd[k] * gamma[k] * gamma_u[k];
                        hp_v += 2.0 * rd[k] * gamma[k] * gamma_v[k];
                        h_uv += 2.0 * rv[k] * gamma_u[k] * gamma_v[k];
                        hp_uv += 2.0 * rd[k] * gamma_u[k] * gamma_v[k];
                        endpoint_u[0] += 2.0 * self.response_upper_basis[k] * gamma[k] * gamma_u[k];
                        endpoint_u[1] += 2.0 * self.response_lower_basis[k] * gamma[k] * gamma_u[k];
                        endpoint_v[0] += 2.0 * self.response_upper_basis[k] * gamma[k] * gamma_v[k];
                        endpoint_v[1] += 2.0 * self.response_lower_basis[k] * gamma[k] * gamma_v[k];
                        endpoint_uv[0] +=
                            2.0 * self.response_upper_basis[k] * gamma_u[k] * gamma_v[k];
                        endpoint_uv[1] +=
                            2.0 * self.response_lower_basis[k] * gamma_u[k] * gamma_v[k];
                    }
                    let q = row_quantities.endpoint_q[i];

                    // Reset the per-row scratch arrays. k=0 entries are reassigned
                    // below; the *_u / *_v factors only ever hold k>=1 contributions,
                    // so their k=0 slot must be zeroed.
                    h_factor_u[0] = 0.0;
                    hp_factor_u[0] = 0.0;
                    h_factor_v[0] = 0.0;
                    hp_factor_v[0] = 0.0;
                    endpoint_factor_u_0[0] = 0.0;
                    endpoint_factor_u_1[0] = 0.0;
                    endpoint_factor_v_0[0] = 0.0;
                    endpoint_factor_v_1[0] = 0.0;
                    h_factor[0] = rv[0];
                    hp_factor[0] = rd[0];
                    endpoint_factor_0[0] = self.response_upper_basis[0];
                    endpoint_factor_1[0] = self.response_lower_basis[0];
                    for k in 1..p_resp {
                        h_factor[k] = 2.0 * rv[k] * gamma[k];
                        hp_factor[k] = 2.0 * rd[k] * gamma[k];
                        h_factor_u[k] = 2.0 * rv[k] * gamma_u[k];
                        hp_factor_u[k] = 2.0 * rd[k] * gamma_u[k];
                        h_factor_v[k] = 2.0 * rv[k] * gamma_v[k];
                        hp_factor_v[k] = 2.0 * rd[k] * gamma_v[k];
                        endpoint_factor_0[k] = 2.0 * self.response_upper_basis[k] * gamma[k];
                        endpoint_factor_1[k] = 2.0 * self.response_lower_basis[k] * gamma[k];
                        endpoint_factor_u_0[k] = 2.0 * self.response_upper_basis[k] * gamma_u[k];
                        endpoint_factor_u_1[k] = 2.0 * self.response_lower_basis[k] * gamma_u[k];
                        endpoint_factor_v_0[k] = 2.0 * self.response_upper_basis[k] * gamma_v[k];
                        endpoint_factor_v_1[k] = 2.0 * self.response_lower_basis[k] * gamma_v[k];
                    }
                    let endpoint_factor = [&endpoint_factor_0[..], &endpoint_factor_1[..]];
                    let endpoint_factor_u = [&endpoint_factor_u_0[..], &endpoint_factor_u_1[..]];
                    let endpoint_factor_v = [&endpoint_factor_v_0[..], &endpoint_factor_v_1[..]];

                    for k in 0..p_resp {
                        for l in 0..p_resp {
                            let same_shape = k == l && k > 0;
                            let mut normalizer_block = 0.0;
                            for a in 0..2 {
                                let h_a_ab = if same_shape {
                                    2.0 * if a == 0 {
                                        self.response_upper_basis[k]
                                    } else {
                                        self.response_lower_basis[k]
                                    }
                                } else {
                                    0.0
                                };
                                for b in 0..2 {
                                    normalizer_block += q.second[a][b] * endpoint_uv[b] * h_a_ab;
                                    for c_ep in 0..2 {
                                        normalizer_block += q.third[a][b][c_ep]
                                            * endpoint_v[c_ep]
                                            * endpoint_u[b]
                                            * h_a_ab;
                                        normalizer_block += q.third[a][b][c_ep]
                                            * endpoint_uv[c_ep]
                                            * endpoint_factor[a][k]
                                            * endpoint_factor[b][l];
                                        normalizer_block += q.third[a][b][c_ep]
                                            * endpoint_u[c_ep]
                                            * (endpoint_factor_v[a][k] * endpoint_factor[b][l]
                                                + endpoint_factor[a][k] * endpoint_factor_v[b][l]);
                                        normalizer_block += q.third[a][b][c_ep]
                                            * endpoint_v[c_ep]
                                            * endpoint_factor_u[a][k]
                                            * endpoint_factor[b][l];
                                        normalizer_block += q.third[a][b][c_ep]
                                            * endpoint_v[c_ep]
                                            * endpoint_factor[a][k]
                                            * endpoint_factor_u[b][l];
                                        for d_ep in 0..2 {
                                            normalizer_block += q.fourth[a][b][c_ep][d_ep]
                                                * endpoint_v[d_ep]
                                                * endpoint_u[c_ep]
                                                * endpoint_factor[a][k]
                                                * endpoint_factor[b][l];
                                        }
                                    }
                                    normalizer_block += q.second[a][b]
                                        * (endpoint_factor_u[a][k] * endpoint_factor_v[b][l]
                                            + endpoint_factor_v[a][k] * endpoint_factor_u[b][l]);
                                }
                            }
                            for c in 0..p_cov {
                                let row_idx = k * p_cov + c;
                                let hp_a = hp_factor[k] * cov_row[c];
                                let dh_a_u = h_factor_u[k] * cov_row[c];
                                let dhp_a_u = hp_factor_u[k] * cov_row[c];
                                let dh_a_v = h_factor_v[k] * cov_row[c];
                                let dhp_a_v = hp_factor_v[k] * cov_row[c];
                                for d in 0..p_cov {
                                    let col_idx = l * p_cov + d;
                                    let hp_b = hp_factor[l] * cov_row[d];
                                    let dh_b_u = h_factor_u[l] * cov_row[d];
                                    let dhp_b_u = hp_factor_u[l] * cov_row[d];
                                    let dh_b_v = h_factor_v[l] * cov_row[d];
                                    let dhp_b_v = hp_factor_v[l] * cov_row[d];
                                    let (h_ab, hp_ab) = if same_shape {
                                        (
                                            2.0 * rv[k] * cov_row[c] * cov_row[d],
                                            2.0 * rd[k] * cov_row[c] * cov_row[d],
                                        )
                                    } else {
                                        (0.0, 0.0)
                                    };
                                    let value = dh_a_u * dh_b_v
                                        + dh_a_v * dh_b_u
                                        + h_uv * h_ab
                                        + (dhp_a_u * dhp_b_v + dhp_a_v * dhp_b_u) * inv_hp_sq
                                        - 2.0
                                            * (dhp_a_u * hp_b + hp_a * dhp_b_u)
                                            * hp_v
                                            * inv_hp_cu
                                        - 2.0
                                            * (dhp_a_v * hp_b + hp_a * dhp_b_v)
                                            * hp_u
                                            * inv_hp_cu
                                        - 2.0 * hp_a * hp_b * hp_uv * inv_hp_cu
                                        + 6.0 * hp_a * hp_b * hp_u * hp_v * inv_hp_qu
                                        + hp_ab * hp_uv * inv_hp_sq
                                        - 2.0 * hp_ab * hp_u * hp_v * inv_hp_cu
                                        + normalizer_block * cov_row[c] * cov_row[d];
                                    chunk_out[[row_idx, col_idx]] += wi * value;
                                }
                            }
                        }
                    }
                }

                chunk_out
            })
            .collect();

        let mut out = Array2::<f64>::zeros((p_total, p_total));
        for chunk in chunk_outputs {
            out.scaled_add(1.0, &chunk);
        }

        Ok(0.5 * (&out + &out.t()))
    }

    pub(crate) fn scop_hessian_matvec_into(
        &self,
        beta: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
        probe: &Array1<f64>,
        out: &mut Array1<f64>,
    ) -> Result<(), String> {
        let stage_start = std::time::Instant::now();
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if beta.len() != p_total || probe.len() != p_total || out.len() != p_total {
            return Err(format!(
                "SCOP Hessian matvec length mismatch: beta={}, probe={}, out={}, expected={p_total}",
                beta.len(),
                probe.len(),
                out.len()
            ));
        }
        if !row_quantities.matches_beta(beta) {
            return Err(
                "SCOP Hessian matvec received row quantities for a different beta".to_string(),
            );
        }
        let probe_mat = probe
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP probe reshape failed: {e}"))?;
        let cov = self
            .covariate_dense_arc()
            .map_err(|e| format!("SCOP Hessian matvec requires cached covariate design: {e}"))?;
        let weights = self.weights.as_ref();
        let h = row_quantities.h.as_ref();
        let h_prime = row_quantities.h_prime.as_ref();
        let gamma_rows = row_quantities.gamma.as_ref();
        if gamma_rows.nrows() != n || gamma_rows.ncols() != p_resp {
            return Err(format!(
                "SCOP Hessian matvec gamma cache shape mismatch: got {}x{}, expected {}x{}",
                gamma_rows.nrows(),
                gamma_rows.ncols(),
                n,
                p_resp
            ));
        }

        out.fill(0.0);
        let mut probe_gamma = vec![0.0; p_resp];

        for i in 0..n {
            let cov_row = cov.row(i);
            let rv = self.response_val_basis.row(i);
            let rd = self.response_deriv_basis.row(i);
            let gamma = gamma_rows.row(i);
            let wi = weights[i];
            let hi = h[i];
            let hp = h_prime[i];
            let inv_hp = 1.0 / hp;
            let inv_hp_sq = inv_hp * inv_hp;

            for k in 0..p_resp {
                probe_gamma[k] = probe_mat.row(k).dot(&cov_row);
            }

            let mut h_probe = rv[0] * probe_gamma[0];
            let mut hp_probe = rd[0] * probe_gamma[0];
            let mut lower_probe = self.response_lower_basis[0] * probe_gamma[0];
            let mut upper_probe = self.response_upper_basis[0] * probe_gamma[0];
            for k in 1..p_resp {
                let pg = probe_gamma[k];
                let gamma_k = gamma[k];
                h_probe += 2.0 * rv[k] * gamma_k * pg;
                hp_probe += 2.0 * rd[k] * gamma_k * pg;
                lower_probe += 2.0 * self.response_lower_basis[k] * gamma_k * pg;
                upper_probe += 2.0 * self.response_upper_basis[k] * gamma_k * pg;
            }
            let q = row_quantities.endpoint_q[i];

            for k in 0..p_resp {
                let h_factor = if k == 0 {
                    rv[0]
                } else {
                    2.0 * rv[k] * gamma[k]
                };
                let hp_factor = if k == 0 {
                    rd[0]
                } else {
                    2.0 * rd[k] * gamma[k]
                };
                let lower_factor = if k == 0 {
                    self.response_lower_basis[0]
                } else {
                    2.0 * self.response_lower_basis[k] * gamma[k]
                };
                let upper_factor = if k == 0 {
                    self.response_upper_basis[0]
                } else {
                    2.0 * self.response_upper_basis[k] * gamma[k]
                };
                let pg = probe_gamma[k];
                let second_probe = if k == 0 {
                    0.0
                } else {
                    2.0 * (hi * rv[k] - rd[k] * inv_hp) * pg
                };
                let lower_factor_probe = if k == 0 {
                    0.0
                } else {
                    2.0 * self.response_lower_basis[k] * pg
                };
                let upper_factor_probe = if k == 0 {
                    0.0
                } else {
                    2.0 * self.response_upper_basis[k] * pg
                };
                let normalizer_probe = q.first[0] * upper_factor_probe
                    + q.first[1] * lower_factor_probe
                    + (q.second[0][0] * upper_factor + q.second[1][0] * lower_factor) * upper_probe
                    + (q.second[0][1] * upper_factor + q.second[1][1] * lower_factor) * lower_probe;
                let scalar = wi
                    * (h_factor * h_probe
                        + hp_factor * hp_probe * inv_hp_sq
                        + second_probe
                        + normalizer_probe);
                let row_offset = k * p_cov;
                for c in 0..p_cov {
                    out[row_offset + c] += scalar * cov_row[c];
                }
            }
        }

        log::info!(
            "[STAGE] CTN scop_hessian_matvec n={} p={} elapsed={:.3}s",
            n,
            p_total,
            stage_start.elapsed().as_secs_f64(),
        );
        Ok(())
    }

    pub(crate) fn scop_hessian_directional_matvec(
        &self,
        beta: &Array1<f64>,
        direction: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
        probe: &Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        let mut probes = Array2::<f64>::zeros((probe.len(), 1));
        probes.column_mut(0).assign(probe);
        let out = self.scop_hessian_directional_matmat(beta, direction, row_quantities, &probes)?;
        Ok(out.column(0).to_owned())
    }

    pub(crate) fn scop_hessian_directional_matmat(
        &self,
        beta: &Array1<f64>,
        direction: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
        probes: &Array2<f64>,
    ) -> Result<Array2<f64>, String> {
        let stage_start = std::time::Instant::now();
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        let n_probe = probes.ncols();
        if beta.len() != p_total || direction.len() != p_total || probes.nrows() != p_total {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP dH matmat length mismatch: beta={}, direction={}, probes rows={}, expected={p_total}",
                beta.len(),
                direction.len(),
                probes.nrows()
            ) }.into());
        }
        if !row_quantities.matches_beta(beta) {
            return Err("SCOP dH matmat received row quantities for a different beta".to_string());
        }
        let dir_mat = direction
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP direction reshape failed: {e}"))?;
        let cov = self
            .covariate_dense_arc()
            .map_err(|e| format!("SCOP dH matmat requires cached covariate design: {e}"))?;
        let weights = self.effective_weights();
        let h_prime = row_quantities.h_prime.as_ref();
        let gamma_rows = row_quantities.gamma.as_ref();
        if gamma_rows.nrows() != n || gamma_rows.ncols() != p_resp {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP dH matmat gamma cache shape mismatch: got {}x{}, expected {}x{}",
                    gamma_rows.nrows(),
                    gamma_rows.ncols(),
                    n,
                    p_resp
                ),
            }
            .into());
        }
        let mut out = Array2::<f64>::zeros((p_total, n_probe));
        let mut gamma_dir = vec![0.0; p_resp];
        let mut gamma_probe = vec![0.0; p_resp * n_probe];
        let mut h_probe = vec![0.0; n_probe];
        let mut hp_probe = vec![0.0; n_probe];
        let mut h_dir_probe = vec![0.0; n_probe];
        let mut hp_dir_probe = vec![0.0; n_probe];
        let mut endpoint_probe = [vec![0.0; n_probe], vec![0.0; n_probe]];
        let mut endpoint_dir_probe = [vec![0.0; n_probe], vec![0.0; n_probe]];

        for i in 0..n {
            let cov_row = cov.row(i);
            let rv = self.response_val_basis.row(i);
            let rd = self.response_deriv_basis.row(i);
            let wi = weights[i];
            let hp = h_prime[i];
            let inv_hp = 1.0 / hp;
            let inv_hp_sq = inv_hp * inv_hp;
            let inv_hp_cu = inv_hp_sq * inv_hp;

            let gamma = gamma_rows.row(i);
            for k in 0..p_resp {
                gamma_dir[k] = dir_mat.row(k).dot(&cov_row);
                let row_offset = k * p_cov;
                let probe_offset = k * n_probe;
                for j in 0..n_probe {
                    let mut value = 0.0;
                    for c in 0..p_cov {
                        value += probes[[row_offset + c, j]] * cov_row[c];
                    }
                    gamma_probe[probe_offset + j] = value;
                }
            }

            let mut h_dir = rv[0] * gamma_dir[0];
            let mut hp_dir = rd[0] * gamma_dir[0];
            let mut endpoint_dir = [
                self.response_upper_basis[0] * gamma_dir[0],
                self.response_lower_basis[0] * gamma_dir[0],
            ];
            for j in 0..n_probe {
                h_probe[j] = rv[0] * gamma_probe[j];
                hp_probe[j] = rd[0] * gamma_probe[j];
                h_dir_probe[j] = 0.0;
                hp_dir_probe[j] = 0.0;
                endpoint_probe[0][j] = self.response_upper_basis[0] * gamma_probe[j];
                endpoint_probe[1][j] = self.response_lower_basis[0] * gamma_probe[j];
                endpoint_dir_probe[0][j] = 0.0;
                endpoint_dir_probe[1][j] = 0.0;
            }
            for k in 1..p_resp {
                let probe_offset = k * n_probe;
                let gamma_k = gamma[k];
                let gamma_dir_k = gamma_dir[k];
                h_dir += 2.0 * rv[k] * gamma[k] * gamma_dir[k];
                hp_dir += 2.0 * rd[k] * gamma[k] * gamma_dir[k];
                endpoint_dir[0] += 2.0 * self.response_upper_basis[k] * gamma[k] * gamma_dir[k];
                endpoint_dir[1] += 2.0 * self.response_lower_basis[k] * gamma[k] * gamma_dir[k];
                for j in 0..n_probe {
                    let pg = gamma_probe[probe_offset + j];
                    h_probe[j] += 2.0 * rv[k] * gamma_k * pg;
                    hp_probe[j] += 2.0 * rd[k] * gamma_k * pg;
                    h_dir_probe[j] += 2.0 * rv[k] * gamma_dir_k * pg;
                    hp_dir_probe[j] += 2.0 * rd[k] * gamma_dir_k * pg;
                    endpoint_probe[0][j] += 2.0 * self.response_upper_basis[k] * gamma_k * pg;
                    endpoint_probe[1][j] += 2.0 * self.response_lower_basis[k] * gamma_k * pg;
                    endpoint_dir_probe[0][j] +=
                        2.0 * self.response_upper_basis[k] * gamma_dir_k * pg;
                    endpoint_dir_probe[1][j] +=
                        2.0 * self.response_lower_basis[k] * gamma_dir_k * pg;
                }
            }
            let q = row_quantities.endpoint_q[i];

            for k in 0..p_resp {
                let probe_offset = k * n_probe;
                let h_factor = if k == 0 {
                    rv[0]
                } else {
                    2.0 * rv[k] * gamma[k]
                };
                let hp_factor = if k == 0 {
                    rd[0]
                } else {
                    2.0 * rd[k] * gamma[k]
                };
                let h_factor_dir = if k == 0 {
                    0.0
                } else {
                    2.0 * rv[k] * gamma_dir[k]
                };
                let hp_factor_dir = if k == 0 {
                    0.0
                } else {
                    2.0 * rd[k] * gamma_dir[k]
                };
                let endpoint_factor = [
                    if k == 0 {
                        self.response_upper_basis[0]
                    } else {
                        2.0 * self.response_upper_basis[k] * gamma[k]
                    },
                    if k == 0 {
                        self.response_lower_basis[0]
                    } else {
                        2.0 * self.response_lower_basis[k] * gamma[k]
                    },
                ];
                let endpoint_factor_dir = [
                    if k == 0 {
                        0.0
                    } else {
                        2.0 * self.response_upper_basis[k] * gamma_dir[k]
                    },
                    if k == 0 {
                        0.0
                    } else {
                        2.0 * self.response_lower_basis[k] * gamma_dir[k]
                    },
                ];
                for j in 0..n_probe {
                    let pg = gamma_probe[probe_offset + j];
                    let h_second_probe = if k == 0 { 0.0 } else { 2.0 * rv[k] * pg };
                    let hp_second_probe = if k == 0 { 0.0 } else { 2.0 * rd[k] * pg };
                    let endpoint_factor_probe = [
                        if k == 0 {
                            0.0
                        } else {
                            2.0 * self.response_upper_basis[k] * pg
                        },
                        if k == 0 {
                            0.0
                        } else {
                            2.0 * self.response_lower_basis[k] * pg
                        },
                    ];
                    let mut normalizer_scalar = 0.0;
                    for a in 0..2 {
                        for b in 0..2 {
                            normalizer_scalar +=
                                q.second[a][b] * endpoint_dir[b] * endpoint_factor_probe[a];
                            normalizer_scalar += q.second[a][b]
                                * (endpoint_factor_dir[a] * endpoint_probe[b][j]
                                    + endpoint_factor[a] * endpoint_dir_probe[b][j]);
                            for c_ep in 0..2 {
                                normalizer_scalar += q.third[a][b][c_ep]
                                    * endpoint_dir[c_ep]
                                    * endpoint_factor[a]
                                    * endpoint_probe[b][j];
                            }
                        }
                    }
                    let scalar = wi
                        * (h_factor_dir * h_probe[j]
                            + h_factor * h_dir_probe[j]
                            + h_dir * h_second_probe
                            + (hp_factor_dir * hp_probe[j] + hp_factor * hp_dir_probe[j])
                                * inv_hp_sq
                            - 2.0 * hp_factor * hp_probe[j] * hp_dir * inv_hp_cu
                            + hp_second_probe * hp_dir * inv_hp_sq
                            + normalizer_scalar);
                    for c in 0..p_cov {
                        out[[k * p_cov + c, j]] += scalar * cov_row[c];
                    }
                }
            }
        }

        log::info!(
            "[STAGE] CTN scop_hessian_directional_matmat n={} p={} k={} elapsed={:.3}s",
            n,
            p_total,
            n_probe,
            stage_start.elapsed().as_secs_f64(),
        );
        Ok(out)
    }

    pub(crate) fn scop_projected_response_gram_table(
        &self,
        factor: ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        let rank = factor.ncols();
        if factor.nrows() != p_total {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP projected response Gram factor row mismatch: factor_rows={}, expected={p_total}",
                factor.nrows()
            ) }.into());
        }
        let cov = self.covariate_dense_arc().map_err(|e| {
            format!("SCOP projected response Gram requires cached covariate design: {e}")
        })?;
        let stride = p_resp * p_resp;
        let mut grams = vec![0.0_f64; n * stride];

        let fill_row = |i: usize, row_out: &mut [f64], projected: &mut [f64]| {
            let cov_row = cov.row(i);
            projected.fill(0.0);
            for k in 0..p_resp {
                let factor_row_base = k * p_cov;
                let projected_base = k * rank;
                for c in 0..p_cov {
                    let x_ic = cov_row[c];
                    if x_ic == 0.0 {
                        continue;
                    }
                    let factor_row = factor_row_base + c;
                    for col in 0..rank {
                        projected[projected_base + col] += x_ic * factor[[factor_row, col]];
                    }
                }
            }
            for k in 0..p_resp {
                let k_base = k * rank;
                for l in 0..p_resp {
                    let l_base = l * rank;
                    let mut value = 0.0;
                    for col in 0..rank {
                        value += projected[k_base + col] * projected[l_base + col];
                    }
                    row_out[k * p_resp + l] = value;
                }
            }
        };

        if rayon::current_thread_index().is_some() {
            let mut projected = vec![0.0_f64; p_resp * rank];
            for (i, row_out) in grams.chunks_mut(stride).enumerate() {
                fill_row(i, row_out, &mut projected);
            }
        } else {
            use rayon::iter::{IndexedParallelIterator, ParallelIterator};
            use rayon::slice::ParallelSliceMut;
            grams.par_chunks_mut(stride).enumerate().for_each_init(
                || vec![0.0_f64; p_resp * rank],
                |projected, (i, row_out)| fill_row(i, row_out, projected),
            );
        }

        Array2::from_shape_vec((n, stride), grams)
            .map_err(|e| format!("SCOP projected response Gram table shape failed: {e}"))
    }

    pub(crate) fn scop_hessian_directional_trace_from_response_grams(
        &self,
        beta: &Array1<f64>,
        direction: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
        row_grams: ArrayView2<'_, f64>,
    ) -> Result<f64, String> {
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if beta.len() != p_total || direction.len() != p_total {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP dH projected trace length mismatch: beta={}, direction={}, expected={p_total}",
                beta.len(),
                direction.len()
            ) }.into());
        }
        if row_grams.nrows() != n || row_grams.ncols() != p_resp * p_resp {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP dH projected trace Gram shape {}x{} != expected {}x{}",
                    row_grams.nrows(),
                    row_grams.ncols(),
                    n,
                    p_resp * p_resp
                ),
            }
            .into());
        }
        if !row_quantities.matches_beta(beta) {
            return Err(
                "SCOP dH projected trace received row quantities for a different beta".to_string(),
            );
        }
        let dir_mat = direction
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP dH projected trace direction reshape failed: {e}"))?;
        let cov = self.covariate_dense_arc().map_err(|e| {
            format!("SCOP dH projected trace requires cached covariate design: {e}")
        })?;
        let weights = self.effective_weights();
        let h_prime = row_quantities.h_prime.as_ref();
        let row_gamma = row_quantities.gamma.as_ref();

        struct DhTraceScratch {
            pub(crate) gamma: Vec<f64>,
            pub(crate) gamma_dir: Vec<f64>,
            pub(crate) h_factor: Vec<f64>,
            pub(crate) hp_factor: Vec<f64>,
            pub(crate) h_factor_dir: Vec<f64>,
            pub(crate) hp_factor_dir: Vec<f64>,
            pub(crate) endpoint_factor: [Vec<f64>; 2],
            pub(crate) endpoint_factor_dir: [Vec<f64>; 2],
        }

        impl DhTraceScratch {
            pub(crate) fn new(p_resp: usize) -> Self {
                Self {
                    gamma: vec![0.0; p_resp],
                    gamma_dir: vec![0.0; p_resp],
                    h_factor: vec![0.0; p_resp],
                    hp_factor: vec![0.0; p_resp],
                    h_factor_dir: vec![0.0; p_resp],
                    hp_factor_dir: vec![0.0; p_resp],
                    endpoint_factor: [vec![0.0; p_resp], vec![0.0; p_resp]],
                    endpoint_factor_dir: [vec![0.0; p_resp], vec![0.0; p_resp]],
                }
            }
        }

        let row_trace = |i: usize, scratch: &mut DhTraceScratch| {
            let cov_row = cov.row(i);
            let rv = self.response_val_basis.row(i);
            let rd = self.response_deriv_basis.row(i);
            let wi = weights[i];
            let hp = h_prime[i];
            let inv_hp = 1.0 / hp;
            let inv_hp_sq = inv_hp * inv_hp;
            let inv_hp_cu = inv_hp_sq * inv_hp;
            let gamma_row = row_gamma.row(i);
            for k in 0..p_resp {
                scratch.gamma[k] = gamma_row[k];
                scratch.gamma_dir[k] = dir_mat.row(k).dot(&cov_row);
            }

            let mut h_dir = rv[0] * scratch.gamma_dir[0];
            let mut hp_dir = rd[0] * scratch.gamma_dir[0];
            let mut endpoint_dir = [
                self.response_upper_basis[0] * scratch.gamma_dir[0],
                self.response_lower_basis[0] * scratch.gamma_dir[0],
            ];
            for k in 1..p_resp {
                h_dir += 2.0 * rv[k] * scratch.gamma[k] * scratch.gamma_dir[k];
                hp_dir += 2.0 * rd[k] * scratch.gamma[k] * scratch.gamma_dir[k];
                endpoint_dir[0] +=
                    2.0 * self.response_upper_basis[k] * scratch.gamma[k] * scratch.gamma_dir[k];
                endpoint_dir[1] +=
                    2.0 * self.response_lower_basis[k] * scratch.gamma[k] * scratch.gamma_dir[k];
            }
            let q = row_quantities.endpoint_q[i];

            scratch.h_factor[0] = rv[0];
            scratch.hp_factor[0] = rd[0];
            scratch.h_factor_dir[0] = 0.0;
            scratch.hp_factor_dir[0] = 0.0;
            scratch.endpoint_factor[0][0] = self.response_upper_basis[0];
            scratch.endpoint_factor[1][0] = self.response_lower_basis[0];
            scratch.endpoint_factor_dir[0][0] = 0.0;
            scratch.endpoint_factor_dir[1][0] = 0.0;
            for k in 1..p_resp {
                scratch.h_factor[k] = 2.0 * rv[k] * scratch.gamma[k];
                scratch.hp_factor[k] = 2.0 * rd[k] * scratch.gamma[k];
                scratch.h_factor_dir[k] = 2.0 * rv[k] * scratch.gamma_dir[k];
                scratch.hp_factor_dir[k] = 2.0 * rd[k] * scratch.gamma_dir[k];
                scratch.endpoint_factor[0][k] =
                    2.0 * self.response_upper_basis[k] * scratch.gamma[k];
                scratch.endpoint_factor[1][k] =
                    2.0 * self.response_lower_basis[k] * scratch.gamma[k];
                scratch.endpoint_factor_dir[0][k] =
                    2.0 * self.response_upper_basis[k] * scratch.gamma_dir[k];
                scratch.endpoint_factor_dir[1][k] =
                    2.0 * self.response_lower_basis[k] * scratch.gamma_dir[k];
            }

            let gram_row = row_grams.row(i);
            let mut total = 0.0;
            for k in 0..p_resp {
                for l in 0..p_resp {
                    let same_shape = k == l && k > 0;
                    let mut normalizer_block = 0.0;
                    for a in 0..2 {
                        let endpoint_second = if same_shape {
                            2.0 * if a == 0 {
                                self.response_upper_basis[k]
                            } else {
                                self.response_lower_basis[k]
                            }
                        } else {
                            0.0
                        };
                        for b in 0..2 {
                            normalizer_block += q.second[a][b] * endpoint_dir[b] * endpoint_second;
                            normalizer_block += q.second[a][b]
                                * (scratch.endpoint_factor_dir[a][k]
                                    * scratch.endpoint_factor[b][l]
                                    + scratch.endpoint_factor[a][k]
                                        * scratch.endpoint_factor_dir[b][l]);
                            for c_ep in 0..2 {
                                normalizer_block += q.third[a][b][c_ep]
                                    * endpoint_dir[c_ep]
                                    * scratch.endpoint_factor[a][k]
                                    * scratch.endpoint_factor[b][l];
                            }
                        }
                    }
                    let second_h = if same_shape { 2.0 * rv[k] } else { 0.0 };
                    let second_hp = if same_shape { 2.0 * rd[k] } else { 0.0 };
                    let q_kl = scratch.h_factor_dir[k] * scratch.h_factor[l]
                        + scratch.h_factor[k] * scratch.h_factor_dir[l]
                        + h_dir * second_h
                        + (scratch.hp_factor_dir[k] * scratch.hp_factor[l]
                            + scratch.hp_factor[k] * scratch.hp_factor_dir[l])
                            * inv_hp_sq
                        - 2.0 * scratch.hp_factor[k] * scratch.hp_factor[l] * hp_dir * inv_hp_cu
                        + second_hp * hp_dir * inv_hp_sq
                        + normalizer_block;
                    total += q_kl * gram_row[k * p_resp + l];
                }
            }
            wi * total
        };

        if rayon::current_thread_index().is_some() {
            let mut scratch = DhTraceScratch::new(p_resp);
            Ok((0..n).map(|i| row_trace(i, &mut scratch)).sum())
        } else {
            Ok(gam_linalg::pairwise_reduce::par_deterministic_block_fold(
                n,
                |range| {
                    let mut scratch = DhTraceScratch::new(p_resp);
                    let mut sum = 0.0;
                    for i in range {
                        sum += row_trace(i, &mut scratch);
                    }
                    sum
                },
                |a, b| a + b,
            )
            .unwrap_or(0.0))
        }
    }

    pub(crate) fn scop_hessian_second_directional_matvec(
        &self,
        beta: &Array1<f64>,
        direction_u: &Array1<f64>,
        direction_v: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
        probe: &Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        let mut probes = Array2::<f64>::zeros((probe.len(), 1));
        probes.column_mut(0).assign(probe);
        let out = self.scop_hessian_second_directional_matmat(
            beta,
            direction_u,
            direction_v,
            row_quantities,
            &probes,
        )?;
        Ok(out.column(0).to_owned())
    }

    pub(crate) fn scop_hessian_second_directional_matmat(
        &self,
        beta: &Array1<f64>,
        direction_u: &Array1<f64>,
        direction_v: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
        probes: &Array2<f64>,
    ) -> Result<Array2<f64>, String> {
        let stage_start = std::time::Instant::now();
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        let n_probe = probes.ncols();
        if beta.len() != p_total
            || direction_u.len() != p_total
            || direction_v.len() != p_total
            || probes.nrows() != p_total
        {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP d2H matmat length mismatch: beta={}, u={}, v={}, probes rows={}, expected={p_total}",
                beta.len(),
                direction_u.len(),
                direction_v.len(),
                probes.nrows()
            ) }.into());
        }
        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP beta reshape failed: {e}"))?;
        let dir_u_mat = direction_u
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP u direction reshape failed: {e}"))?;
        let dir_v_mat = direction_v
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP v direction reshape failed: {e}"))?;
        let cov = self
            .covariate_dense_arc()
            .map_err(|e| format!("SCOP d2H matmat requires cached covariate design: {e}"))?;
        let weights = self.effective_weights();
        let h_prime = row_quantities.h_prime.as_ref();
        let mut out = Array2::<f64>::zeros((p_total, n_probe));
        let mut gamma = vec![0.0; p_resp];
        let mut gamma_u = vec![0.0; p_resp];
        let mut gamma_v = vec![0.0; p_resp];
        let mut gamma_probe = vec![0.0; p_resp * n_probe];
        let mut hp_probe = vec![0.0; n_probe];
        let mut h_u_probe = vec![0.0; n_probe];
        let mut hp_u_probe = vec![0.0; n_probe];
        let mut h_v_probe = vec![0.0; n_probe];
        let mut hp_v_probe = vec![0.0; n_probe];
        let mut endpoint_probe = [vec![0.0; n_probe], vec![0.0; n_probe]];
        let mut endpoint_u_probe = [vec![0.0; n_probe], vec![0.0; n_probe]];
        let mut endpoint_v_probe = [vec![0.0; n_probe], vec![0.0; n_probe]];

        for i in 0..n {
            let cov_row = cov.row(i);
            let rv = self.response_val_basis.row(i);
            let rd = self.response_deriv_basis.row(i);
            let wi = weights[i];
            let hp = h_prime[i];
            let inv_hp = 1.0 / hp;
            let inv_hp_sq = inv_hp * inv_hp;
            let inv_hp_cu = inv_hp_sq * inv_hp;
            let inv_hp_qu = inv_hp_sq * inv_hp_sq;

            for k in 0..p_resp {
                gamma[k] = beta_mat.row(k).dot(&cov_row);
                gamma_u[k] = dir_u_mat.row(k).dot(&cov_row);
                gamma_v[k] = dir_v_mat.row(k).dot(&cov_row);
                let row_offset = k * p_cov;
                let probe_offset = k * n_probe;
                for j in 0..n_probe {
                    let mut value = 0.0;
                    for c in 0..p_cov {
                        value += probes[[row_offset + c, j]] * cov_row[c];
                    }
                    gamma_probe[probe_offset + j] = value;
                }
            }

            let mut hp_u = rd[0] * gamma_u[0];
            let mut hp_v = rd[0] * gamma_v[0];
            let mut h_uv = 0.0;
            let mut hp_uv = 0.0;
            let mut endpoint_u = [
                self.response_upper_basis[0] * gamma_u[0],
                self.response_lower_basis[0] * gamma_u[0],
            ];
            let mut endpoint_v = [
                self.response_upper_basis[0] * gamma_v[0],
                self.response_lower_basis[0] * gamma_v[0],
            ];
            let mut endpoint_uv = [0.0, 0.0];
            for j in 0..n_probe {
                hp_probe[j] = rd[0] * gamma_probe[j];
                h_u_probe[j] = 0.0;
                hp_u_probe[j] = 0.0;
                h_v_probe[j] = 0.0;
                hp_v_probe[j] = 0.0;
                endpoint_probe[0][j] = self.response_upper_basis[0] * gamma_probe[j];
                endpoint_probe[1][j] = self.response_lower_basis[0] * gamma_probe[j];
                endpoint_u_probe[0][j] = 0.0;
                endpoint_u_probe[1][j] = 0.0;
                endpoint_v_probe[0][j] = 0.0;
                endpoint_v_probe[1][j] = 0.0;
            }
            for k in 1..p_resp {
                let probe_offset = k * n_probe;
                let gamma_k = gamma[k];
                let gamma_u_k = gamma_u[k];
                let gamma_v_k = gamma_v[k];
                hp_u += 2.0 * rd[k] * gamma[k] * gamma_u[k];
                hp_v += 2.0 * rd[k] * gamma[k] * gamma_v[k];
                h_uv += 2.0 * rv[k] * gamma_u[k] * gamma_v[k];
                hp_uv += 2.0 * rd[k] * gamma_u[k] * gamma_v[k];
                endpoint_u[0] += 2.0 * self.response_upper_basis[k] * gamma[k] * gamma_u[k];
                endpoint_u[1] += 2.0 * self.response_lower_basis[k] * gamma[k] * gamma_u[k];
                endpoint_v[0] += 2.0 * self.response_upper_basis[k] * gamma[k] * gamma_v[k];
                endpoint_v[1] += 2.0 * self.response_lower_basis[k] * gamma[k] * gamma_v[k];
                endpoint_uv[0] += 2.0 * self.response_upper_basis[k] * gamma_u[k] * gamma_v[k];
                endpoint_uv[1] += 2.0 * self.response_lower_basis[k] * gamma_u[k] * gamma_v[k];
                for j in 0..n_probe {
                    let pg = gamma_probe[probe_offset + j];
                    hp_probe[j] += 2.0 * rd[k] * gamma_k * pg;
                    h_u_probe[j] += 2.0 * rv[k] * gamma_u_k * pg;
                    hp_u_probe[j] += 2.0 * rd[k] * gamma_u_k * pg;
                    h_v_probe[j] += 2.0 * rv[k] * gamma_v_k * pg;
                    hp_v_probe[j] += 2.0 * rd[k] * gamma_v_k * pg;
                    endpoint_probe[0][j] += 2.0 * self.response_upper_basis[k] * gamma_k * pg;
                    endpoint_probe[1][j] += 2.0 * self.response_lower_basis[k] * gamma_k * pg;
                    endpoint_u_probe[0][j] += 2.0 * self.response_upper_basis[k] * gamma_u_k * pg;
                    endpoint_u_probe[1][j] += 2.0 * self.response_lower_basis[k] * gamma_u_k * pg;
                    endpoint_v_probe[0][j] += 2.0 * self.response_upper_basis[k] * gamma_v_k * pg;
                    endpoint_v_probe[1][j] += 2.0 * self.response_lower_basis[k] * gamma_v_k * pg;
                }
            }
            let q = row_quantities.endpoint_q[i];

            for k in 0..p_resp {
                let probe_offset = k * n_probe;
                let hp_factor = if k == 0 {
                    rd[0]
                } else {
                    2.0 * rd[k] * gamma[k]
                };
                let h_factor_u = if k == 0 {
                    0.0
                } else {
                    2.0 * rv[k] * gamma_u[k]
                };
                let hp_factor_u = if k == 0 {
                    0.0
                } else {
                    2.0 * rd[k] * gamma_u[k]
                };
                let h_factor_v = if k == 0 {
                    0.0
                } else {
                    2.0 * rv[k] * gamma_v[k]
                };
                let hp_factor_v = if k == 0 {
                    0.0
                } else {
                    2.0 * rd[k] * gamma_v[k]
                };
                let endpoint_factor = [
                    if k == 0 {
                        self.response_upper_basis[0]
                    } else {
                        2.0 * self.response_upper_basis[k] * gamma[k]
                    },
                    if k == 0 {
                        self.response_lower_basis[0]
                    } else {
                        2.0 * self.response_lower_basis[k] * gamma[k]
                    },
                ];
                let endpoint_factor_u = [
                    if k == 0 {
                        0.0
                    } else {
                        2.0 * self.response_upper_basis[k] * gamma_u[k]
                    },
                    if k == 0 {
                        0.0
                    } else {
                        2.0 * self.response_lower_basis[k] * gamma_u[k]
                    },
                ];
                let endpoint_factor_v = [
                    if k == 0 {
                        0.0
                    } else {
                        2.0 * self.response_upper_basis[k] * gamma_v[k]
                    },
                    if k == 0 {
                        0.0
                    } else {
                        2.0 * self.response_lower_basis[k] * gamma_v[k]
                    },
                ];
                for j in 0..n_probe {
                    let pg = gamma_probe[probe_offset + j];
                    let h_second_probe = if k == 0 { 0.0 } else { 2.0 * rv[k] * pg };
                    let hp_second_probe = if k == 0 { 0.0 } else { 2.0 * rd[k] * pg };
                    let endpoint_factor_probe = [
                        if k == 0 {
                            0.0
                        } else {
                            2.0 * self.response_upper_basis[k] * pg
                        },
                        if k == 0 {
                            0.0
                        } else {
                            2.0 * self.response_lower_basis[k] * pg
                        },
                    ];
                    let mut normalizer_scalar = 0.0;
                    for a in 0..2 {
                        for b in 0..2 {
                            normalizer_scalar +=
                                q.second[a][b] * endpoint_uv[b] * endpoint_factor_probe[a];
                            for c_ep in 0..2 {
                                normalizer_scalar += q.third[a][b][c_ep]
                                    * endpoint_v[c_ep]
                                    * endpoint_u[b]
                                    * endpoint_factor_probe[a];
                                normalizer_scalar += q.third[a][b][c_ep]
                                    * endpoint_uv[c_ep]
                                    * endpoint_factor[a]
                                    * endpoint_probe[b][j];
                                normalizer_scalar += q.third[a][b][c_ep]
                                    * endpoint_u[c_ep]
                                    * (endpoint_factor_v[a] * endpoint_probe[b][j]
                                        + endpoint_factor[a] * endpoint_v_probe[b][j]);
                                normalizer_scalar += q.third[a][b][c_ep]
                                    * endpoint_v[c_ep]
                                    * endpoint_factor_u[a]
                                    * endpoint_probe[b][j];
                                normalizer_scalar += q.third[a][b][c_ep]
                                    * endpoint_v[c_ep]
                                    * endpoint_factor[a]
                                    * endpoint_u_probe[b][j];
                                for d_ep in 0..2 {
                                    normalizer_scalar += q.fourth[a][b][c_ep][d_ep]
                                        * endpoint_v[d_ep]
                                        * endpoint_u[c_ep]
                                        * endpoint_factor[a]
                                        * endpoint_probe[b][j];
                                }
                            }
                            normalizer_scalar += q.second[a][b]
                                * (endpoint_factor_u[a] * endpoint_v_probe[b][j]
                                    + endpoint_factor_v[a] * endpoint_u_probe[b][j]);
                        }
                    }
                    let scalar = wi
                        * (h_factor_u * h_v_probe[j]
                            + h_factor_v * h_u_probe[j]
                            + h_uv * h_second_probe
                            + (hp_factor_u * hp_v_probe[j] + hp_factor_v * hp_u_probe[j])
                                * inv_hp_sq
                            - 2.0
                                * (hp_factor_u * hp_probe[j] + hp_factor * hp_u_probe[j])
                                * hp_v
                                * inv_hp_cu
                            - 2.0
                                * (hp_factor_v * hp_probe[j] + hp_factor * hp_v_probe[j])
                                * hp_u
                                * inv_hp_cu
                            - 2.0 * hp_factor * hp_probe[j] * hp_uv * inv_hp_cu
                            + 6.0 * hp_factor * hp_probe[j] * hp_u * hp_v * inv_hp_qu
                            + hp_second_probe * hp_uv * inv_hp_sq
                            - 2.0 * hp_second_probe * hp_u * hp_v * inv_hp_cu
                            + normalizer_scalar);
                    for c in 0..p_cov {
                        out[[k * p_cov + c, j]] += scalar * cov_row[c];
                    }
                }
            }
        }

        log::info!(
            "[STAGE] CTN scop_hessian_second_directional_matmat n={} p={} k={} elapsed={:.3}s",
            n,
            p_total,
            n_probe,
            stage_start.elapsed().as_secs_f64(),
        );
        Ok(out)
    }

    pub(crate) fn scop_hessian_diagonal(
        &self,
        beta: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
    ) -> Result<Array1<f64>, String> {
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if beta.len() != p_total {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP Hessian diagonal beta length {} != expected {p_total}",
                    beta.len()
                ),
            }
            .into());
        }
        if !row_quantities.matches_beta(beta) {
            return Err(
                "SCOP Hessian diagonal received row quantities for a different beta".to_string(),
            );
        }
        if !row_quantities.matches_beta(beta) {
            return Err(
                "SCOP Hessian diagonal received row quantities for a different beta".to_string(),
            );
        }
        let cov = self
            .covariate_dense_arc()
            .map_err(|e| format!("SCOP Hessian diagonal requires cached covariate design: {e}"))?;
        let weights = self.effective_weights();
        let h = row_quantities.h.as_ref();
        let h_prime = row_quantities.h_prime.as_ref();
        let gamma_rows = row_quantities.gamma.as_ref();
        if gamma_rows.nrows() != n || gamma_rows.ncols() != p_resp {
            return Err(format!(
                "SCOP Hessian diagonal gamma cache shape mismatch: got {}x{}, expected {}x{}",
                gamma_rows.nrows(),
                gamma_rows.ncols(),
                n,
                p_resp
            ));
        }
        let mut diag = Array1::<f64>::zeros(p_total);
        for i in 0..n {
            let cov_row = cov.row(i);
            let rv = self.response_val_basis.row(i);
            let rd = self.response_deriv_basis.row(i);
            let gamma = gamma_rows.row(i);
            let wi = weights[i];
            let hi = h[i];
            let hp = h_prime[i];
            let inv_hp = 1.0 / hp;
            let inv_hp_sq = inv_hp * inv_hp;
            let q = row_quantities.endpoint_q[i];

            // k = 0 prologue: second / lower_second / upper_second vanish.
            {
                let h_factor = rv[0];
                let hp_factor = rd[0];
                let lower_factor = self.response_lower_basis[0];
                let upper_factor = self.response_upper_basis[0];
                let normalizer_second = q.second[0][0] * upper_factor * upper_factor
                    + (q.second[0][1] + q.second[1][0]) * upper_factor * lower_factor
                    + q.second[1][1] * lower_factor * lower_factor;
                let coeff = wi
                    * (h_factor * h_factor + hp_factor * hp_factor * inv_hp_sq + normalizer_second);
                for c in 0..p_cov {
                    let cc = cov_row[c] * cov_row[c];
                    diag[c] += coeff * cc;
                }
            }

            for k in 1..p_resp {
                let two_gamma_k = 2.0 * gamma[k];
                let h_factor = rv[k] * two_gamma_k;
                let hp_factor = rd[k] * two_gamma_k;
                let second = 2.0 * (hi * rv[k] - rd[k] * inv_hp);
                let lower_factor = self.response_lower_basis[k] * two_gamma_k;
                let upper_factor = self.response_upper_basis[k] * two_gamma_k;
                let lower_second = 2.0 * self.response_lower_basis[k];
                let upper_second = 2.0 * self.response_upper_basis[k];
                let normalizer_second = q.first[0] * upper_second
                    + q.first[1] * lower_second
                    + q.second[0][0] * upper_factor * upper_factor
                    + (q.second[0][1] + q.second[1][0]) * upper_factor * lower_factor
                    + q.second[1][1] * lower_factor * lower_factor;
                let coeff = wi
                    * (h_factor * h_factor
                        + hp_factor * hp_factor * inv_hp_sq
                        + second
                        + normalizer_second);
                let row_offset = k * p_cov;
                for c in 0..p_cov {
                    let cc = cov_row[c] * cov_row[c];
                    diag[row_offset + c] += coeff * cc;
                }
            }
        }
        Ok(diag)
    }
}
