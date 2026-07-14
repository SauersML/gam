//! Direct-α CTN curvature (gam#2306).
//!
//! With the identity chart the transformation is exactly LINEAR in the
//! coefficients: for one observation with covariate row `c`, response value
//! row `r`, derivative row `m`, upper/lower endpoint value rows `U`, `L`,
//! and coefficient `a = (k, p)`,
//!
//! ```text
//! h_a  = r_k c_p        hp_a = m_k c_p        (∂U)_a = U_k c_p   (∂L)_a = L_k c_p
//! h_ab = hp_ab = 0      (no chart second derivatives — the γ² terms are gone)
//! ```
//!
//! Writing `E_0 = U`, `E_1 = L` and `q` for the endpoint-normalizer
//! derivative tower (`LogNormalCdfDiffDerivatives`), the per-row NEGATIVE
//! log-likelihood Hessian and its β-directional derivatives reduce to
//! separable (k, l) block factors:
//!
//! ```text
//! H:      w [ r_k r_l + m_k m_l / hp² + Σ_{e1,e2} q.second[e1][e2] E_{e1,k} E_{e2,l} ]
//! DH[u]:  w [ −2 m_k m_l · hp_u / hp³ + Σ_{e1,e2} (Σ_e3 q.third[e1][e2][e3] ep_u[e3]) E_{e1,k} E_{e2,l} ]
//! D²H[u,v]: w [ 6 m_k m_l · hp_u hp_v / hp⁴
//!             + Σ_{e1,e2} (Σ_{e3,e4} q.fourth[e1][e2][e3][e4] ep_u[e3] ep_v[e4]) E_{e1,k} E_{e2,l} ]
//! ```
//!
//! because `hp_uv = 0` and every endpoint directional second derivative is
//! zero for a linear map. All β-dependence flows through `h`, `hp`, and the
//! endpoint arguments; the coefficient-side factors are constants of the
//! basis. Every (k, l) block is therefore `covᵀ diag(block_weights) cov`,
//! assembled by the same weighted-Gram kernel the value Hessian always used.

use super::*;

impl TransformationNormalFamily {
    /// Shared validation for the curvature entry points.
    fn scop_check(
        &self,
        beta: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
        context: &str,
    ) -> Result<(usize, usize, usize, usize), String> {
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if beta.len() != p_total {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "{context}: beta length {} != p_resp({p_resp}) * p_cov({p_cov})",
                    beta.len()
                ),
            }
            .into());
        }
        if !row_quantities.matches_beta(beta) {
            return Err(format!(
                "{context}: received row quantities for a different beta"
            ));
        }
        let alpha_rows = row_quantities.alpha.as_ref();
        if alpha_rows.nrows() != n || alpha_rows.ncols() != p_resp {
            return Err(format!(
                "{context}: alpha cache shape mismatch: got {}x{}, expected {}x{}",
                alpha_rows.nrows(),
                alpha_rows.ncols(),
                n,
                p_resp
            ));
        }
        Ok((n, p_resp, p_cov, p_total))
    }

    pub(crate) fn scop_gradient_and_negative_hessian(
        &self,
        beta: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
    ) -> Result<(Array1<f64>, Array2<f64>), String> {
        let (n, p_resp, p_cov, p_total) =
            self.scop_check(beta, row_quantities, "SCOP gradient/Hessian")?;
        let cov = self
            .covariate_dense_arc()
            .map_err(|e| format!("SCOP gradient requires cached covariate design: {e}"))?;
        let weights = self.effective_weights();
        let h_prime = row_quantities.h_prime.as_ref();
        let endpoint_q = row_quantities.endpoint_q.as_ref();
        let response_val_basis = &self.response_val_basis;
        let response_deriv_basis = &self.response_deriv_basis;
        let response_lower_basis = &self.response_lower_basis;
        let response_upper_basis = &self.response_upper_basis;
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let response_pairs: Vec<(usize, usize)> = (0..p_resp)
            .flat_map(|k| (k..p_resp).map(move |l| (k, l)))
            .collect();
        let blocks: Vec<(usize, usize, Array2<f64>)> = response_pairs
            .into_par_iter()
            .map(|(k, l)| {
                let mut block_weights = Array1::<f64>::zeros(n);
                for i in 0..n {
                    let rv = response_val_basis.row(i);
                    let rd = response_deriv_basis.row(i);
                    let inv_hp = 1.0 / h_prime[i];
                    let inv_hp_sq = inv_hp * inv_hp;
                    let q = endpoint_q[i];
                    let upper_k = response_upper_basis[k];
                    let lower_k = response_lower_basis[k];
                    let upper_l = response_upper_basis[l];
                    let lower_l = response_lower_basis[l];
                    let mut block_factor = rv[k] * rv[l] + rd[k] * rd[l] * inv_hp_sq;
                    block_factor += q.second[0][0] * upper_k * upper_l
                        + q.second[0][1] * upper_k * lower_l
                        + q.second[1][0] * lower_k * upper_l
                        + q.second[1][1] * lower_k * lower_l;
                    block_weights[i] = weights[i] * block_factor;
                }
                let block = gam_problem::with_nested_parallel(|| {
                    gam_linalg::faer_ndarray::fast_xt_diag_x_with_parallelism(
                        cov.as_ref(),
                        &block_weights,
                        faer::Par::Seq,
                    )
                });
                (k, l, block)
            })
            .collect();

        let gradient = self.scop_gradient(beta, row_quantities)?;
        let mut hessian = Array2::<f64>::zeros((p_total, p_total));
        for (k, l, block) in blocks {
            hessian
                .slice_mut(s![k * p_cov..(k + 1) * p_cov, l * p_cov..(l + 1) * p_cov])
                .assign(&block);
            if k != l {
                hessian
                    .slice_mut(s![l * p_cov..(l + 1) * p_cov, k * p_cov..(k + 1) * p_cov])
                    .assign(&block.t());
            }
        }

        Ok((gradient, hessian))
    }

    pub(crate) fn scop_gradient(
        &self,
        beta: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
    ) -> Result<Array1<f64>, String> {
        let (n, p_resp, p_cov, p_total) = self.scop_check(beta, row_quantities, "SCOP gradient")?;
        let cov = self
            .covariate_dense_arc()
            .map_err(|e| format!("SCOP gradient requires cached covariate design: {e}"))?;
        let weights = self.effective_weights();
        let h = row_quantities.h.as_ref();
        let h_prime = row_quantities.h_prime.as_ref();
        let mut gradient = Array1::<f64>::zeros(p_total);

        for i in 0..n {
            let cov_row = cov.row(i);
            let rv = self.response_val_basis.row(i);
            let rd = self.response_deriv_basis.row(i);
            let wi = weights[i];
            let hi = h[i];
            let inv_hp = 1.0 / h_prime[i];
            let q = row_quantities.endpoint_q[i];

            for k in 0..p_resp {
                let normalizer_score = q.first[0] * self.response_upper_basis[k]
                    + q.first[1] * self.response_lower_basis[k];
                let score_factor = wi * (-hi * rv[k] + rd[k] * inv_hp - normalizer_score);
                let offset = k * p_cov;
                for c in 0..p_cov {
                    gradient[offset + c] += score_factor * cov_row[c];
                }
            }
        }

        Ok(gradient)
    }

    /// Directional derivative of the negative Hessian: the only β-dependence
    /// is through `hp` and the endpoint arguments (see the module header).
    pub(crate) fn scop_hessian_directional_derivative(
        &self,
        beta: &Array1<f64>,
        direction: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
    ) -> Result<Array2<f64>, String> {
        let (n, p_resp, p_cov, p_total) =
            self.scop_check(beta, row_quantities, "SCOP dH directional")?;
        if direction.len() != p_total {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP dH directional: direction length {} != expected {p_total}",
                    direction.len()
                ),
            }
            .into());
        }
        let dir_mat = direction
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP direction reshape failed: {e}"))?;
        let cov = self
            .covariate_dense_arc()
            .map_err(|e| format!("SCOP dH directional requires cached covariate design: {e}"))?;
        let weights = self.effective_weights();
        let h_prime = row_quantities.h_prime.as_ref();
        let response_deriv_basis = &self.response_deriv_basis;
        let response_lower_basis = &self.response_lower_basis;
        let response_upper_basis = &self.response_upper_basis;

        // Per-row directional endpoint / derivative rates (linear in u).
        let mut hp_dir = Array1::<f64>::zeros(n);
        let mut endpoint_dir = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let cov_row = cov.row(i);
            let rd = response_deriv_basis.row(i);
            let mut hp_u = 0.0;
            let mut ep_u = [0.0_f64; 2];
            for k in 0..p_resp {
                let u_k = dir_mat.row(k).dot(&cov_row);
                hp_u += rd[k] * u_k;
                ep_u[0] += response_upper_basis[k] * u_k;
                ep_u[1] += response_lower_basis[k] * u_k;
            }
            hp_dir[i] = hp_u;
            endpoint_dir[[i, 0]] = ep_u[0];
            endpoint_dir[[i, 1]] = ep_u[1];
        }

        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let response_pairs: Vec<(usize, usize)> = (0..p_resp)
            .flat_map(|k| (k..p_resp).map(move |l| (k, l)))
            .collect();
        let blocks: Vec<(usize, usize, Array2<f64>)> = response_pairs
            .into_par_iter()
            .map(|(k, l)| {
                let mut block_weights = Array1::<f64>::zeros(n);
                for i in 0..n {
                    let rd = response_deriv_basis.row(i);
                    let inv_hp = 1.0 / h_prime[i];
                    let inv_hp_cu = inv_hp * inv_hp * inv_hp;
                    let q = row_quantities.endpoint_q[i];
                    let e_k = [response_upper_basis[k], response_lower_basis[k]];
                    let e_l = [response_upper_basis[l], response_lower_basis[l]];
                    let mut factor = -2.0 * rd[k] * rd[l] * hp_dir[i] * inv_hp_cu;
                    for e1 in 0..2 {
                        for e2 in 0..2 {
                            let mut third = 0.0;
                            for e3 in 0..2 {
                                third += q.third[e1][e2][e3] * endpoint_dir[[i, e3]];
                            }
                            factor += third * e_k[e1] * e_l[e2];
                        }
                    }
                    block_weights[i] = weights[i] * factor;
                }
                let block = gam_problem::with_nested_parallel(|| {
                    gam_linalg::faer_ndarray::fast_xt_diag_x_with_parallelism(
                        cov.as_ref(),
                        &block_weights,
                        faer::Par::Seq,
                    )
                });
                (k, l, block)
            })
            .collect();

        let mut out = Array2::<f64>::zeros((p_total, p_total));
        for (k, l, block) in blocks {
            out.slice_mut(s![k * p_cov..(k + 1) * p_cov, l * p_cov..(l + 1) * p_cov])
                .assign(&block);
            if k != l {
                out.slice_mut(s![l * p_cov..(l + 1) * p_cov, k * p_cov..(k + 1) * p_cov])
                    .assign(&block.t());
            }
        }
        // The (k,l) block factors are symmetric in (k,l) by construction and
        // each block is covᵀ D cov (symmetric), so `out` is exactly symmetric.
        Ok(out)
    }

    /// Second directional derivative of the negative Hessian (see header):
    /// `hp_uv = 0` for a linear map, so only the `6/hp⁴` term and the
    /// fourth-order endpoint tower survive.
    pub(crate) fn scop_hessian_second_directional_derivative(
        &self,
        beta: &Array1<f64>,
        direction_u: &Array1<f64>,
        direction_v: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
    ) -> Result<Array2<f64>, String> {
        let (n, p_resp, p_cov, p_total) =
            self.scop_check(beta, row_quantities, "SCOP d2H directional")?;
        if direction_u.len() != p_total || direction_v.len() != p_total {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP d2H directional: u={}, v={}, expected {p_total}",
                    direction_u.len(),
                    direction_v.len()
                ),
            }
            .into());
        }
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
            .map_err(|e| format!("SCOP d2H directional requires cached covariate design: {e}"))?;
        let weights = self.effective_weights();
        let h_prime = row_quantities.h_prime.as_ref();
        let response_deriv_basis = &self.response_deriv_basis;
        let response_lower_basis = &self.response_lower_basis;
        let response_upper_basis = &self.response_upper_basis;

        let mut hp_u = Array1::<f64>::zeros(n);
        let mut hp_v = Array1::<f64>::zeros(n);
        let mut ep_u = Array2::<f64>::zeros((n, 2));
        let mut ep_v = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let cov_row = cov.row(i);
            let rd = response_deriv_basis.row(i);
            let mut hpu = 0.0;
            let mut hpv = 0.0;
            let mut epu = [0.0_f64; 2];
            let mut epv = [0.0_f64; 2];
            for k in 0..p_resp {
                let u_k = dir_u_mat.row(k).dot(&cov_row);
                let v_k = dir_v_mat.row(k).dot(&cov_row);
                hpu += rd[k] * u_k;
                hpv += rd[k] * v_k;
                epu[0] += response_upper_basis[k] * u_k;
                epu[1] += response_lower_basis[k] * u_k;
                epv[0] += response_upper_basis[k] * v_k;
                epv[1] += response_lower_basis[k] * v_k;
            }
            hp_u[i] = hpu;
            hp_v[i] = hpv;
            ep_u[[i, 0]] = epu[0];
            ep_u[[i, 1]] = epu[1];
            ep_v[[i, 0]] = epv[0];
            ep_v[[i, 1]] = epv[1];
        }

        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let response_pairs: Vec<(usize, usize)> = (0..p_resp)
            .flat_map(|k| (k..p_resp).map(move |l| (k, l)))
            .collect();
        let blocks: Vec<(usize, usize, Array2<f64>)> = response_pairs
            .into_par_iter()
            .map(|(k, l)| {
                let mut block_weights = Array1::<f64>::zeros(n);
                for i in 0..n {
                    let rd = response_deriv_basis.row(i);
                    let inv_hp = 1.0 / h_prime[i];
                    let inv_hp_sq = inv_hp * inv_hp;
                    let inv_hp_qu = inv_hp_sq * inv_hp_sq;
                    let q = row_quantities.endpoint_q[i];
                    let e_k = [response_upper_basis[k], response_lower_basis[k]];
                    let e_l = [response_upper_basis[l], response_lower_basis[l]];
                    let mut factor = 6.0 * rd[k] * rd[l] * hp_u[i] * hp_v[i] * inv_hp_qu;
                    for e1 in 0..2 {
                        for e2 in 0..2 {
                            let mut fourth = 0.0;
                            for e3 in 0..2 {
                                for e4 in 0..2 {
                                    fourth += q.fourth[e1][e2][e3][e4]
                                        * ep_u[[i, e3]]
                                        * ep_v[[i, e4]];
                                }
                            }
                            factor += fourth * e_k[e1] * e_l[e2];
                        }
                    }
                    block_weights[i] = weights[i] * factor;
                }
                let block = gam_problem::with_nested_parallel(|| {
                    gam_linalg::faer_ndarray::fast_xt_diag_x_with_parallelism(
                        cov.as_ref(),
                        &block_weights,
                        faer::Par::Seq,
                    )
                });
                (k, l, block)
            })
            .collect();

        let mut out = Array2::<f64>::zeros((p_total, p_total));
        for (k, l, block) in blocks {
            out.slice_mut(s![k * p_cov..(k + 1) * p_cov, l * p_cov..(l + 1) * p_cov])
                .assign(&block);
            if k != l {
                out.slice_mut(s![l * p_cov..(l + 1) * p_cov, k * p_cov..(k + 1) * p_cov])
                    .assign(&block.t());
            }
        }
        Ok(out)
    }

    pub(crate) fn scop_hessian_matvec_into(
        &self,
        beta: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
        probe: &Array1<f64>,
        out: &mut Array1<f64>,
    ) -> Result<(), String> {
        let stage_start = std::time::Instant::now();
        let (n, p_resp, p_cov, p_total) = self.scop_check(beta, row_quantities, "SCOP H matvec")?;
        if probe.len() != p_total || out.len() != p_total {
            return Err(format!(
                "SCOP Hessian matvec length mismatch: probe={}, out={}, expected={p_total}",
                probe.len(),
                out.len()
            ));
        }
        let probe_mat = probe
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP probe reshape failed: {e}"))?;
        let cov = self
            .covariate_dense_arc()
            .map_err(|e| format!("SCOP Hessian matvec requires cached covariate design: {e}"))?;
        let weights = self.effective_weights();
        let h_prime = row_quantities.h_prime.as_ref();

        out.fill(0.0);
        for i in 0..n {
            let cov_row = cov.row(i);
            let rv = self.response_val_basis.row(i);
            let rd = self.response_deriv_basis.row(i);
            let wi = weights[i];
            let inv_hp = 1.0 / h_prime[i];
            let inv_hp_sq = inv_hp * inv_hp;
            let q = row_quantities.endpoint_q[i];

            let mut h_probe = 0.0;
            let mut hp_probe = 0.0;
            let mut upper_probe = 0.0;
            let mut lower_probe = 0.0;
            for k in 0..p_resp {
                let pg = probe_mat.row(k).dot(&cov_row);
                h_probe += rv[k] * pg;
                hp_probe += rd[k] * pg;
                upper_probe += self.response_upper_basis[k] * pg;
                lower_probe += self.response_lower_basis[k] * pg;
            }

            for k in 0..p_resp {
                let upper_factor = self.response_upper_basis[k];
                let lower_factor = self.response_lower_basis[k];
                let normalizer_probe = (q.second[0][0] * upper_factor
                    + q.second[1][0] * lower_factor)
                    * upper_probe
                    + (q.second[0][1] * upper_factor + q.second[1][1] * lower_factor)
                        * lower_probe;
                let scalar =
                    wi * (rv[k] * h_probe + rd[k] * hp_probe * inv_hp_sq + normalizer_probe);
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
        let (n, p_resp, p_cov, p_total) =
            self.scop_check(beta, row_quantities, "SCOP dH matmat")?;
        let n_probe = probes.ncols();
        if direction.len() != p_total || probes.nrows() != p_total {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP dH matmat length mismatch: direction={}, probes rows={}, expected={p_total}",
                    direction.len(),
                    probes.nrows()
                ),
            }
            .into());
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
        let mut out = Array2::<f64>::zeros((p_total, n_probe));
        let mut hp_probe = vec![0.0; n_probe];
        let mut upper_probe = vec![0.0; n_probe];
        let mut lower_probe = vec![0.0; n_probe];

        for i in 0..n {
            let cov_row = cov.row(i);
            let rd = self.response_deriv_basis.row(i);
            let wi = weights[i];
            let inv_hp = 1.0 / h_prime[i];
            let inv_hp_cu = inv_hp * inv_hp * inv_hp;
            let q = row_quantities.endpoint_q[i];

            let mut hp_dir = 0.0;
            let mut ep_dir = [0.0_f64; 2];
            hp_probe.iter_mut().for_each(|v| *v = 0.0);
            upper_probe.iter_mut().for_each(|v| *v = 0.0);
            lower_probe.iter_mut().for_each(|v| *v = 0.0);
            for k in 0..p_resp {
                let u_k = dir_mat.row(k).dot(&cov_row);
                hp_dir += rd[k] * u_k;
                ep_dir[0] += self.response_upper_basis[k] * u_k;
                ep_dir[1] += self.response_lower_basis[k] * u_k;
                let row_offset = k * p_cov;
                for j in 0..n_probe {
                    let mut pg = 0.0;
                    for c in 0..p_cov {
                        pg += probes[[row_offset + c, j]] * cov_row[c];
                    }
                    hp_probe[j] += rd[k] * pg;
                    upper_probe[j] += self.response_upper_basis[k] * pg;
                    lower_probe[j] += self.response_lower_basis[k] * pg;
                }
            }

            for k in 0..p_resp {
                let e_k = [self.response_upper_basis[k], self.response_lower_basis[k]];
                for j in 0..n_probe {
                    let mut normalizer_scalar = 0.0;
                    let ep_probe = [upper_probe[j], lower_probe[j]];
                    for e1 in 0..2 {
                        for e2 in 0..2 {
                            let mut third = 0.0;
                            for e3 in 0..2 {
                                third += q.third[e1][e2][e3] * ep_dir[e3];
                            }
                            normalizer_scalar += third * e_k[e1] * ep_probe[e2];
                        }
                    }
                    let scalar = wi
                        * (-2.0 * rd[k] * hp_probe[j] * hp_dir * inv_hp_cu + normalizer_scalar);
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
        let (n, p_resp, p_cov, p_total) =
            self.scop_check(beta, row_quantities, "SCOP dH projected trace")?;
        if direction.len() != p_total {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP dH projected trace direction length {} != expected {p_total}",
                    direction.len()
                ),
            }
            .into());
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
        let dir_mat = direction
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP dH projected trace direction reshape failed: {e}"))?;
        let cov = self.covariate_dense_arc().map_err(|e| {
            format!("SCOP dH projected trace requires cached covariate design: {e}")
        })?;
        let weights = self.effective_weights();
        let h_prime = row_quantities.h_prime.as_ref();

        let row_trace = |i: usize| {
            let cov_row = cov.row(i);
            let rd = self.response_deriv_basis.row(i);
            let wi = weights[i];
            let inv_hp = 1.0 / h_prime[i];
            let inv_hp_cu = inv_hp * inv_hp * inv_hp;
            let q = row_quantities.endpoint_q[i];

            let mut hp_dir = 0.0;
            let mut ep_dir = [0.0_f64; 2];
            for k in 0..p_resp {
                let u_k = dir_mat.row(k).dot(&cov_row);
                hp_dir += rd[k] * u_k;
                ep_dir[0] += self.response_upper_basis[k] * u_k;
                ep_dir[1] += self.response_lower_basis[k] * u_k;
            }

            let gram_row = row_grams.row(i);
            let mut total = 0.0;
            for k in 0..p_resp {
                let e_k = [self.response_upper_basis[k], self.response_lower_basis[k]];
                for l in 0..p_resp {
                    let e_l = [self.response_upper_basis[l], self.response_lower_basis[l]];
                    let mut normalizer_block = 0.0;
                    for e1 in 0..2 {
                        for e2 in 0..2 {
                            let mut third = 0.0;
                            for e3 in 0..2 {
                                third += q.third[e1][e2][e3] * ep_dir[e3];
                            }
                            normalizer_block += third * e_k[e1] * e_l[e2];
                        }
                    }
                    let q_kl =
                        -2.0 * rd[k] * rd[l] * hp_dir * inv_hp_cu + normalizer_block;
                    total += q_kl * gram_row[k * p_resp + l];
                }
            }
            wi * total
        };

        if rayon::current_thread_index().is_some() {
            Ok((0..n).map(row_trace).sum())
        } else {
            Ok(gam_linalg::pairwise_reduce::par_deterministic_block_fold(
                n,
                |range| {
                    let mut sum = 0.0;
                    for i in range {
                        sum += row_trace(i);
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
        let (n, p_resp, p_cov, p_total) =
            self.scop_check(beta, row_quantities, "SCOP d2H matmat")?;
        let n_probe = probes.ncols();
        if direction_u.len() != p_total
            || direction_v.len() != p_total
            || probes.nrows() != p_total
        {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP d2H matmat length mismatch: u={}, v={}, probes rows={}, expected={p_total}",
                direction_u.len(),
                direction_v.len(),
                probes.nrows()
            ) }.into());
        }
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
        let mut hp_probe = vec![0.0; n_probe];
        let mut upper_probe = vec![0.0; n_probe];
        let mut lower_probe = vec![0.0; n_probe];

        for i in 0..n {
            let cov_row = cov.row(i);
            let rd = self.response_deriv_basis.row(i);
            let wi = weights[i];
            let inv_hp = 1.0 / h_prime[i];
            let inv_hp_sq = inv_hp * inv_hp;
            let inv_hp_qu = inv_hp_sq * inv_hp_sq;
            let q = row_quantities.endpoint_q[i];

            let mut hp_u = 0.0;
            let mut hp_v = 0.0;
            let mut ep_u = [0.0_f64; 2];
            let mut ep_v = [0.0_f64; 2];
            hp_probe.iter_mut().for_each(|value| *value = 0.0);
            upper_probe.iter_mut().for_each(|value| *value = 0.0);
            lower_probe.iter_mut().for_each(|value| *value = 0.0);
            for k in 0..p_resp {
                let u_k = dir_u_mat.row(k).dot(&cov_row);
                let v_k = dir_v_mat.row(k).dot(&cov_row);
                hp_u += rd[k] * u_k;
                hp_v += rd[k] * v_k;
                ep_u[0] += self.response_upper_basis[k] * u_k;
                ep_u[1] += self.response_lower_basis[k] * u_k;
                ep_v[0] += self.response_upper_basis[k] * v_k;
                ep_v[1] += self.response_lower_basis[k] * v_k;
                let row_offset = k * p_cov;
                for j in 0..n_probe {
                    let mut pg = 0.0;
                    for c in 0..p_cov {
                        pg += probes[[row_offset + c, j]] * cov_row[c];
                    }
                    hp_probe[j] += rd[k] * pg;
                    upper_probe[j] += self.response_upper_basis[k] * pg;
                    lower_probe[j] += self.response_lower_basis[k] * pg;
                }
            }

            for k in 0..p_resp {
                let e_k = [self.response_upper_basis[k], self.response_lower_basis[k]];
                for j in 0..n_probe {
                    let ep_probe = [upper_probe[j], lower_probe[j]];
                    let mut normalizer_scalar = 0.0;
                    for e1 in 0..2 {
                        for e2 in 0..2 {
                            let mut fourth = 0.0;
                            for e3 in 0..2 {
                                for e4 in 0..2 {
                                    fourth += q.fourth[e1][e2][e3][e4] * ep_u[e3] * ep_v[e4];
                                }
                            }
                            normalizer_scalar += fourth * e_k[e1] * ep_probe[e2];
                        }
                    }
                    let scalar = wi
                        * (6.0 * rd[k] * hp_probe[j] * hp_u * hp_v * inv_hp_qu
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
        let (n, p_resp, p_cov, p_total) =
            self.scop_check(beta, row_quantities, "SCOP Hessian diagonal")?;
        let cov = self
            .covariate_dense_arc()
            .map_err(|e| format!("SCOP Hessian diagonal requires cached covariate design: {e}"))?;
        let weights = self.effective_weights();
        let h_prime = row_quantities.h_prime.as_ref();
        let mut diag = Array1::<f64>::zeros(p_total);
        for i in 0..n {
            let cov_row = cov.row(i);
            let rv = self.response_val_basis.row(i);
            let rd = self.response_deriv_basis.row(i);
            let wi = weights[i];
            let inv_hp = 1.0 / h_prime[i];
            let inv_hp_sq = inv_hp * inv_hp;
            let q = row_quantities.endpoint_q[i];

            for k in 0..p_resp {
                let upper_factor = self.response_upper_basis[k];
                let lower_factor = self.response_lower_basis[k];
                let normalizer_second = q.second[0][0] * upper_factor * upper_factor
                    + (q.second[0][1] + q.second[1][0]) * upper_factor * lower_factor
                    + q.second[1][1] * lower_factor * lower_factor;
                let coeff =
                    wi * (rv[k] * rv[k] + rd[k] * rd[k] * inv_hp_sq + normalizer_second);
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
