//! Direct-α CTN curvature (gam#2306).
//!
//! With the identity chart the transformation is exactly LINEAR in the
//! coefficients: for one observation with covariate row `c`, response value
//! row `r`, derivative row `m`, and coefficient `a = (k, p)`,
//!
//! ```text
//! h_a  = r_k c_p        hp_a = m_k c_p
//! h_ab = hp_ab = 0      (no chart second derivatives — the γ² terms are gone)
//! ```
//!
//! The per-row negative log-likelihood Hessian and its β-directional
//! derivatives therefore reduce to separable `(k, l)` block factors:
//!
//! ```text
//! H:          w [ r_k r_l + m_k m_l / hp² ]
//! DH[u]:      w [ −2 m_k m_l · hp_u / hp³ ]
//! D²H[u,v]:  w [ 6 m_k m_l · hp_u hp_v / hp⁴ ]
//! ```
//!
//! because `hp_uv = 0`. All β-dependence flows through `h` and `hp`; the
//! coefficient-side factors are constants of the basis. Every `(k, l)` block
//! is therefore `covᵀ diag(block_weights) cov`, assembled by the same
//! weighted-Gram kernel the value Hessian always used.

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
        let response_val_basis = &self.response_val_basis;
        let response_deriv_basis = &self.response_deriv_basis;
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
                    let block_factor = rv[k] * rv[l] + rd[k] * rd[l] * inv_hp_sq;
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

            for k in 0..p_resp {
                let score_factor = wi * (-hi * rv[k] + rd[k] * inv_hp);
                let offset = k * p_cov;
                for c in 0..p_cov {
                    gradient[offset + c] += score_factor * cov_row[c];
                }
            }
        }

        Ok(gradient)
    }

    /// Directional derivative of the negative Hessian: its only β-dependence
    /// is through `hp` (see the module header).
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

        // Per-row directional derivative rate (linear in u).
        let hp_dir = row_direction_rates(cov.as_ref(), response_deriv_basis, dir_mat);

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
                    let factor = -2.0 * rd[k] * rd[l] * hp_dir[i] * inv_hp_cu;
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
    /// `hp_uv = 0` for a linear map, so only the `6/hp⁴` term survives.
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

        let hp_u = row_direction_rates(cov.as_ref(), response_deriv_basis, dir_u_mat);
        let hp_v = row_direction_rates(cov.as_ref(), response_deriv_basis, dir_v_mat);

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
                    let factor = 6.0 * rd[k] * rd[l] * hp_u[i] * hp_v[i] * inv_hp_qu;
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

            let mut h_probe = 0.0;
            let mut hp_probe = 0.0;
            for k in 0..p_resp {
                let pg = probe_mat.row(k).dot(&cov_row);
                h_probe += rv[k] * pg;
                hp_probe += rd[k] * pg;
            }

            for k in 0..p_resp {
                let scalar = wi * (rv[k] * h_probe + rd[k] * hp_probe * inv_hp_sq);
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

            for k in 0..p_resp {
                let coeff = wi * (rv[k] * rv[k] + rd[k] * rd[k] * inv_hp_sq);
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

/// `hp_d[i] = Σ_k m_ik · (cov_i · d_k)`: the rate at which the row's `h'`
/// moves along a coefficient direction `d` reshaped to `(p_resp, p_cov)`.
///
/// `cov_i · d_k` over all `(i, k)` is `cov · dᵀ`, one GEMM, so the whole
/// vector is that product read against the response-derivative basis rather
/// than `n · p_resp` separate row dot products (gam#979).
fn row_direction_rates(
    cov: &Array2<f64>,
    response_deriv_basis: &Array2<f64>,
    direction: ArrayView2<'_, f64>,
) -> Array1<f64> {
    let projected = gam_linalg::faer_ndarray::fast_abt(cov, &direction);
    let n = cov.nrows();
    let p_resp = response_deriv_basis.ncols();
    assert_eq!(projected.shape(), [n, p_resp]);
    let mut rates = Array1::<f64>::zeros(n);
    for (i, rate) in rates.iter_mut().enumerate() {
        let mut accumulated = 0.0;
        for k in 0..p_resp {
            accumulated += response_deriv_basis[[i, k]] * projected[[i, k]];
        }
        *rate = accumulated;
    }
    rates
}
