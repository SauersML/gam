use super::*;
use crate::construction::{
    create_balanced_penalty_root_from_canonical, precompute_reparam_invariant_from_canonical,
};
use crate::linalg::sparse_exact::build_sparse_penalty_blocks_from_canonical;
use crate::linalg::utils::{boundary_hit_indices, symmetric_spectrum_condition_number};
use crate::pirls::PirlsWorkspace;
use crate::solver::estimate::reml::inner_strategy::HessianEvalStrategyKind;
use crate::solver::outer_strategy::{HessianResult, OuterEval};
use crate::types::{InverseLink, LinkFunction, SasLinkState};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

const TK_BLOCK_SIZE: usize = 128;
const TK_MAX_OBSERVATIONS: usize = 20_000;
const TK_MAX_COEFFICIENTS: usize = 2_000;
const TK_MAX_DENSE_WORK: usize = 5_000_000;

#[inline]
fn compute_gradient_for_tk(mode: super::unified::EvalMode) -> bool {
    mode != super::unified::EvalMode::ValueOnly
}

struct TkCorrectionTerms {
    value: f64,
    gradient: Option<Array1<f64>>,
}

struct TkSharedIntermediates {
    h_diag: Array1<f64>,
    x_m: Array1<f64>,
    y: Array1<f64>,
    active_blocks: Vec<TkActiveBlock>,
}

struct TkActiveBlock {
    start: usize,
    end: usize,
    entries: Vec<(usize, f64)>,
}

/// Family-dependent derivative context shared by all assembly builders.
///
/// Both `build_dense_derivative_context` and `build_sparse_derivative_context`
/// return this, eliminating the tuple-order mismatch that previously existed
/// between the two paths.
struct DerivativeContext {
    deriv_provider: Box<dyn super::unified::HessianDerivativeProvider>,
    dispersion: super::unified::DispersionHandling,
    log_likelihood: f64,
    firth_op: Option<std::sync::Arc<super::FirthDenseOperator>>,
    barrier_config: Option<super::unified::BarrierConfig>,
}

impl<'a> RemlState<'a> {
    pub(crate) fn analytic_outer_hessian_enabled(&self) -> bool {
        if self.config.link_function() == LinkFunction::Identity {
            return true;
        }

        let n_obs = self.x().nrows();
        let p_coeff = self.x().ncols();
        let enabled = super::exact_outer_hessian_problem_scale_allows(n_obs, p_coeff);
        if !enabled
            && !self
                .outer_hessian_downgrade_logged
                .swap(true, Ordering::Relaxed)
        {
            log::info!(
                "[OUTER] standard REML: disabling analytic outer Hessian for large corrected model (n={}, p={}, n*p={}, n*p^2={})",
                n_obs,
                p_coeff,
                n_obs.saturating_mul(p_coeff),
                n_obs.saturating_mul(p_coeff).saturating_mul(p_coeff),
            );
        }
        enabled
    }

    pub(super) fn sparse_exact_beta_original(&self, pirls_result: &PirlsResult) -> Array1<f64> {
        match pirls_result.coordinate_frame {
            pirls::PirlsCoordinateFrame::OriginalSparseNative => {
                pirls_result.beta_transformed.as_ref().clone()
            }
            pirls::PirlsCoordinateFrame::TransformedQs => pirls_result
                .reparam_result
                .qs
                .dot(pirls_result.beta_transformed.as_ref()),
        }
    }

    fn bundle_matrix_in_original_basis(
        &self,
        pirls_result: &PirlsResult,
        matrix: &Array2<f64>,
    ) -> Array2<f64> {
        match pirls_result.coordinate_frame {
            pirls::PirlsCoordinateFrame::OriginalSparseNative => matrix.clone(),
            pirls::PirlsCoordinateFrame::TransformedQs => {
                let qs = &pirls_result.reparam_result.qs;
                let tmp = qs.dot(matrix);
                tmp.dot(&qs.t())
            }
        }
    }

    pub(crate) fn last_ridge_used(&self) -> Option<f64> {
        self.cache_manager
            .current_eval_bundle
            .read()
            .unwrap()
            .as_ref()
            .map(|bundle| bundle.ridge_passport.delta)
    }

    fn dense_penalty_logdet_derivs(
        &self,
        rho: &Array1<f64>,
        e_for_logdet: &Array2<f64>,
        penalty_roots: &[Array2<f64>],
        ridge_passport: RidgePassport,
        mode: super::unified::EvalMode,
    ) -> Result<(usize, super::unified::PenaltyLogdetDerivs), EstimationError> {
        let (penalty_rank, log_det_s) =
            self.fixed_subspace_penalty_rank_and_logdet(e_for_logdet, ridge_passport)?;
        let lambdas = rho.mapv(f64::exp);

        // Use block-local path from canonical penalties (basis-invariant logdet).
        // The penalty logdet log|Σ λ_k S_k|₊ is invariant under orthogonal
        // transformation, so the original-basis canonical penalties give the
        // same result as transformed-basis roots.
        let (det1, det2_full) = if !self.canonical_penalties.is_empty()
            && self.canonical_penalties.len() == rho.len()
        {
            self.structural_penalty_logdet_derivatives_block_local(
                &lambdas,
                ridge_passport.penalty_logdet_ridge(),
            )?
        } else if !penalty_roots.is_empty() {
            self.structural_penalty_logdet_derivatives(
                penalty_roots,
                &lambdas,
                ridge_passport.penalty_logdet_ridge(),
            )?
        } else {
            (
                Array1::zeros(rho.len()),
                Array2::zeros((rho.len(), rho.len())),
            )
        };

        let det2 = if mode == super::unified::EvalMode::ValueGradientHessian {
            Some(det2_full)
        } else {
            None
        };
        Ok((
            penalty_rank,
            super::unified::PenaltyLogdetDerivs {
                value: log_det_s,
                first: det1,
                second: det2,
            },
        ))
    }

    fn tk_shared_intermediates(
        x_dense: &Array2<f64>,
        z: &Array2<f64>,
        c_array: &Array1<f64>,
        context: &str,
        h_inv_solve: &dyn Fn(&Array1<f64>) -> Result<Array1<f64>, EstimationError>,
    ) -> Result<TkSharedIntermediates, EstimationError> {
        let n = x_dense.nrows();
        let active_blocks = Self::tk_active_blocks(c_array);
        let mut h_diag = Array1::<f64>::zeros(n);
        for i in 0..n {
            let val = x_dense.row(i).dot(&z.column(i));
            if !val.is_finite() {
                return Err(EstimationError::InvalidInput(format!(
                    "{context} produced non-finite leverage at row {i}: {val}"
                )));
            }
            h_diag[i] = val;
        }
        let m_vec = c_array * &h_diag;
        let x_m = x_dense.t().dot(&m_vec);
        let y = h_inv_solve(&x_m)?;
        Ok(TkSharedIntermediates {
            h_diag,
            x_m,
            y,
            active_blocks,
        })
    }

    fn tk_scalar_from_shared(
        x_dense: &Array2<f64>,
        z: &Array2<f64>,
        d_array: &Array1<f64>,
        shared: &TkSharedIntermediates,
        gram: &mut Array2<f64>,
    ) -> Result<f64, EstimationError> {
        let q_term = -0.125
            * d_array
                .iter()
                .zip(shared.h_diag.iter())
                .map(|(&d_i, &h_i)| d_i * h_i * h_i)
                .sum::<f64>();
        let t2_term = 0.125 * shared.x_m.dot(&shared.y);

        let mut t1_sum = 0.0_f64;
        for (j_block_idx, j_block) in shared.active_blocks.iter().enumerate() {
            let j0 = j_block.start;
            let j1 = j_block.end;
            for i_block in &shared.active_blocks[..=j_block_idx] {
                let i0 = i_block.start;
                let i1 = i_block.end;
                Self::tk_fill_gram_block(x_dense, z, i0, i1, j0, j1, gram);
                let mut block_sum = 0.0_f64;
                for &(bi, ci) in &i_block.entries {
                    for &(bj, cj) in &j_block.entries {
                        let kij = gram[[bi, bj]];
                        block_sum += ci * cj * kij * kij * kij;
                    }
                }
                t1_sum += if i0 == j0 { block_sum } else { 2.0 * block_sum };
            }
        }

        let value = q_term + t1_sum / 12.0 + t2_term;
        if !value.is_finite() {
            return Err(EstimationError::InvalidInput(format!(
                "Tierney-Kadane correction produced non-finite value: {value}"
            )));
        }
        Ok(value)
    }

    fn tk_active_blocks(c_array: &Array1<f64>) -> Vec<TkActiveBlock> {
        let n = c_array.len();
        let mut blocks = Vec::with_capacity(n.div_ceil(TK_BLOCK_SIZE));
        for start in (0..n).step_by(TK_BLOCK_SIZE) {
            let end = (start + TK_BLOCK_SIZE).min(n);
            let entries = c_array
                .slice(s![start..end])
                .iter()
                .enumerate()
                .filter_map(|(offset, &value)| (value != 0.0).then_some((offset, value)))
                .collect::<Vec<_>>();
            if !entries.is_empty() {
                blocks.push(TkActiveBlock {
                    start,
                    end,
                    entries,
                });
            }
        }
        blocks
    }

    fn tk_active_weighted_trace(
        active_blocks: &[TkActiveBlock],
        x_vk: &Array1<f64>,
        lev_p: &Array1<f64>,
    ) -> f64 {
        let mut trace = 0.0;
        for block in active_blocks {
            for &(offset, c_i) in &block.entries {
                let i = block.start + offset;
                trace += c_i * x_vk[i] * lev_p[i];
            }
        }
        trace
    }

    fn tk_fill_gram_block(
        x_dense: &Array2<f64>,
        z: &Array2<f64>,
        i0: usize,
        i1: usize,
        j0: usize,
        j1: usize,
        gram: &mut Array2<f64>,
    ) {
        let rows = i1 - i0;
        let cols = j1 - j0;
        debug_assert!(rows <= gram.nrows());
        debug_assert!(cols <= gram.ncols());
        let x_block = x_dense.slice(s![i0..i1, ..]);
        let z_block = z.slice(s![.., j0..j1]);
        let mut target = gram.slice_mut(s![..rows, ..cols]);
        ndarray::linalg::general_mat_mul(1.0, &x_block, &z_block, 0.0, &mut target);
    }

    fn tk_gradient_from_shared(
        x_dense: &Array2<f64>,
        z: &Array2<f64>,
        c_array: &Array1<f64>,
        d_array: &Array1<f64>,
        e_array: &Array1<f64>,
        tk_penalties: &[crate::construction::CanonicalPenalty],
        lambdas: &[f64],
        ext_drifts: &[Array2<f64>],
        ext_eta_fixed: &[Option<Array1<f64>>],
        ext_x_fixed: &[Option<Array2<f64>>],
        x_vks: &[Array1<f64>],
        beta_dirs: &[Array1<f64>],
        firth_op: Option<&super::FirthDenseOperator>,
        shared: &TkSharedIntermediates,
        gram: &mut Array2<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        let n = x_dense.nrows();
        let p = x_dense.ncols();
        let k = tk_penalties.len();
        let total_k = k + ext_drifts.len();
        if x_vks.len() != total_k {
            return Err(EstimationError::InvalidInput(format!(
                "Tierney-Kadane correction internal gradient arity mismatch: {} response modes for {} coordinates",
                x_vks.len(),
                total_k
            )));
        }
        if beta_dirs.len() != total_k {
            return Err(EstimationError::InvalidInput(format!(
                "Tierney-Kadane correction internal beta-direction arity mismatch: {} beta directions for {} coordinates",
                beta_dirs.len(),
                total_k
            )));
        }
        let x_y = x_dense.dot(&shared.y);

        let mut diag_combined = Array1::<f64>::zeros(n);
        for i in 0..n {
            diag_combined[i] = d_array[i] * shared.h_diag[i] - c_array[i] * x_y[i];
        }
        let mut p_total = Array2::<f64>::zeros((p, p));
        for i in 0..n {
            let wi = diag_combined[i];
            if wi == 0.0 {
                continue;
            }
            let z_i = z.column(i);
            for a in 0..p {
                let wa = wi * z_i[a];
                for b in a..p {
                    let val = wa * z_i[b];
                    p_total[[a, b]] += val;
                    if a != b {
                        p_total[[b, a]] += val;
                    }
                }
            }
        }
        p_total.mapv_inplace(|v| 0.25 * v);
        for a in 0..p {
            for b in 0..p {
                p_total[[a, b]] -= 0.125 * shared.y[a] * shared.y[b];
            }
        }

        for (j_block_idx, j_block) in shared.active_blocks.iter().enumerate() {
            let j0 = j_block.start;
            let j1 = j_block.end;
            for i_block in &shared.active_blocks[..=j_block_idx] {
                let i0 = i_block.start;
                let i1 = i_block.end;
                Self::tk_fill_gram_block(x_dense, z, i0, i1, j0, j1, gram);
                for &(bi, ci) in &i_block.entries {
                    for &(bj, cj) in &j_block.entries {
                        let gij = gram[[bi, bj]];
                        let weight = ci * cj * gij * gij;
                        let ii = i0 + bi;
                        let jj = j0 + bj;
                        let sym_factor = if i0 == j0 { 1.0 } else { 2.0 };
                        let scale = -0.25 * weight * sym_factor;
                        if ii == jj {
                            let z_i = z.column(ii);
                            for a in 0..p {
                                for b in a..p {
                                    let val = scale * z_i[a] * z_i[b];
                                    p_total[[a, b]] += val;
                                    if a != b {
                                        p_total[[b, a]] += val;
                                    }
                                }
                            }
                        } else {
                            let z_ii = z.column(ii);
                            let z_jj = z.column(jj);
                            let half_scale = 0.5 * scale;
                            for a in 0..p {
                                for b in 0..p {
                                    p_total[[a, b]] +=
                                        half_scale * (z_ii[a] * z_jj[b] + z_jj[a] * z_ii[b]);
                                }
                            }
                        }
                    }
                }
            }
        }

        let xp = x_dense.dot(&p_total);
        let mut lev_p = Array1::<f64>::zeros(n);
        for i in 0..n {
            lev_p[i] = xp.row(i).dot(&x_dense.row(i));
        }

        let mut gradient = Array1::<f64>::zeros(total_k);
        for idx in 0..k {
            let cp = &tk_penalties[idx];
            let r = &cp.col_range;
            let p_block = p_total.slice(s![r.start..r.end, r.start..r.end]);
            let rk_p = cp.root.dot(&p_block);
            let trace_ak_p = lambdas[idx]
                * (0..cp.rank())
                    .map(|row| rk_p.row(row).dot(&cp.root.row(row)))
                    .sum::<f64>();
            let correction_trace =
                Self::tk_active_weighted_trace(&shared.active_blocks, &x_vks[idx], &lev_p);
            let firth_trace =
                Self::tk_firth_beta_hessian_trace(firth_op, &beta_dirs[idx], &p_total)?;
            let eta_total = x_vks[idx].mapv(|value| -value);
            let direct = Self::tk_direct_gradient_from_cd_and_design(
                x_dense, z, c_array, d_array, e_array, &eta_total, None, shared, gram,
            )?;
            gradient[idx] = trace_ak_p - correction_trace + firth_trace + direct;
        }
        for (extra_idx, drift) in ext_drifts.iter().enumerate() {
            if drift.raw_dim() != p_total.raw_dim() {
                return Err(EstimationError::InvalidInput(format!(
                    "Tierney-Kadane ext penalty drift shape mismatch: expected {}x{}, got {}x{}",
                    p,
                    p,
                    drift.nrows(),
                    drift.ncols()
                )));
            }
            let mut trace_ak_p = 0.0;
            for row in 0..p {
                for col in 0..p {
                    trace_ak_p += drift[[row, col]] * p_total[[col, row]];
                }
            }
            let x_vk_idx = k + extra_idx;
            let correction_trace =
                Self::tk_active_weighted_trace(&shared.active_blocks, &x_vks[x_vk_idx], &lev_p);
            let firth_trace =
                Self::tk_firth_beta_hessian_trace(firth_op, &beta_dirs[x_vk_idx], &p_total)?;
            let mut eta_total = x_vks[x_vk_idx].clone();
            if let Some(eta_fixed) = ext_eta_fixed
                .get(extra_idx)
                .and_then(|value| value.as_ref())
            {
                if eta_fixed.len() != n {
                    return Err(EstimationError::InvalidInput(format!(
                        "Tierney-Kadane ext fixed eta length mismatch: expected {}, got {}",
                        n,
                        eta_fixed.len()
                    )));
                }
                eta_total += eta_fixed;
            }
            let x_fixed = ext_x_fixed.get(extra_idx).and_then(|value| value.as_ref());
            if let Some(x_fixed) = x_fixed {
                if x_fixed.raw_dim() != x_dense.raw_dim() {
                    return Err(EstimationError::InvalidInput(format!(
                        "Tierney-Kadane ext fixed design shape mismatch: expected {}x{}, got {}x{}",
                        x_dense.nrows(),
                        x_dense.ncols(),
                        x_fixed.nrows(),
                        x_fixed.ncols()
                    )));
                }
            }
            let direct = Self::tk_direct_gradient_from_cd_and_design(
                x_dense, z, c_array, d_array, e_array, &eta_total, x_fixed, shared, gram,
            )?;
            gradient[x_vk_idx] = trace_ak_p + correction_trace + firth_trace + direct;
        }

        for g in gradient.iter_mut() {
            if !g.is_finite() {
                return Err(EstimationError::InvalidInput(
                    "Tierney-Kadane correction produced a non-finite gradient entry".to_string(),
                ));
            }
        }
        Ok(gradient)
    }

    fn tk_firth_beta_hessian_trace(
        firth_op: Option<&super::FirthDenseOperator>,
        beta_dir: &Array1<f64>,
        p_total: &Array2<f64>,
    ) -> Result<f64, EstimationError> {
        let Some(firth_op) = firth_op else {
            return Ok(0.0);
        };
        if beta_dir.len() != p_total.nrows() {
            return Err(EstimationError::InvalidInput(format!(
                "Tierney-Kadane Firth beta-direction length mismatch: expected {}, got {}",
                p_total.nrows(),
                beta_dir.len()
            )));
        }
        let deta = firth_op.x_dense.dot(beta_dir);
        let dir = firth_op.direction_from_deta(deta);
        let hphi = firth_op.hphi_direction(&dir);
        if hphi.raw_dim() != p_total.raw_dim() {
            return Err(EstimationError::InvalidInput(format!(
                "Tierney-Kadane Firth Hessian derivative shape mismatch: expected {}x{}, got {}x{}",
                p_total.nrows(),
                p_total.ncols(),
                hphi.nrows(),
                hphi.ncols()
            )));
        }

        let mut trace = 0.0;
        for row in 0..hphi.nrows() {
            for col in 0..hphi.ncols() {
                trace -= hphi[[row, col]] * p_total[[col, row]];
            }
        }
        Ok(trace)
    }

    /// Direct analytic derivative of the TK scalar through the per-row
    /// curvature carriers and design rows, with `H⁻¹` held fixed.
    ///
    /// For
    ///   V_TK = -1/8 Σ d_i h_i^2 + 1/12 Σ c_i c_j K_ij^3 + 1/8 qᵀH⁻¹q,
    /// where `h_i = K_ii`, `K_ij = x_iᵀH⁻¹x_j`, and `q = Xᵀ(c ⊙ h)`,
    /// the direct terms are:
    ///   c' = d ⊙ η',  d' = e ⊙ η',
    ///   h'_i = 2 x'ᵢᵀH⁻¹x_i,
    ///   K'_ij = x'ᵢᵀH⁻¹x_j + x'ⱼᵀH⁻¹x_i,
    ///   q' = Xᵀ(c'⊙h + c⊙h') + X'ᵀ(c⊙h).
    /// The remaining `H`-drift part of dV_TK/dθ is assembled by
    /// `tk_gradient_from_shared`.
    fn tk_direct_gradient_from_cd_and_design(
        x_dense: &Array2<f64>,
        z: &Array2<f64>,
        c_array: &Array1<f64>,
        d_array: &Array1<f64>,
        e_array: &Array1<f64>,
        eta_total: &Array1<f64>,
        x_fixed: Option<&Array2<f64>>,
        shared: &TkSharedIntermediates,
        gram: &mut Array2<f64>,
    ) -> Result<f64, EstimationError> {
        let n = x_dense.nrows();
        if eta_total.len() != n || e_array.len() != n {
            return Err(EstimationError::InvalidInput(format!(
                "Tierney-Kadane direct derivative length mismatch: n={}, eta={}, e={}",
                n,
                eta_total.len(),
                e_array.len()
            )));
        }

        let mut c_prime = Array1::<f64>::zeros(n);
        let mut d_prime = Array1::<f64>::zeros(n);
        ndarray::Zip::from(&mut c_prime)
            .and(d_array)
            .and(eta_total)
            .for_each(|out, &d, &eta| *out = d * eta);
        ndarray::Zip::from(&mut d_prime)
            .and(e_array)
            .and(eta_total)
            .for_each(|out, &e, &eta| *out = e * eta);

        let mut h_prime = Array1::<f64>::zeros(n);
        let mut design_q_prime = Array1::<f64>::zeros(x_dense.ncols());
        let has_design_deriv = x_fixed.is_some();
        if let Some(x_theta) = x_fixed {
            let ch = c_array * &shared.h_diag;
            design_q_prime += &x_theta.t().dot(&ch);
            for i in 0..n {
                h_prime[i] = 2.0 * x_theta.row(i).dot(&z.column(i));
            }
        }

        let q_weight_prime = &(&c_prime * &shared.h_diag) + &(c_array * &h_prime);
        let q_prime = x_dense.t().dot(&q_weight_prime) + design_q_prime;
        let q_term_prime = 0.25 * q_prime.dot(&shared.y);

        let d_term_prime = -0.125
            * d_prime
                .iter()
                .zip(shared.h_diag.iter())
                .map(|(&dp, &h)| dp * h * h)
                .sum::<f64>()
            - 0.25
                * d_array
                    .iter()
                    .zip(shared.h_diag.iter())
                    .zip(h_prime.iter())
                    .map(|((&d, &h), &hp)| d * h * hp)
                    .sum::<f64>();

        let direct_blocks = Self::tk_cd_direct_active_blocks(c_array, &c_prime);
        let mut c_term_prime = 0.0_f64;
        for (j_block_idx, j_block) in direct_blocks.iter().enumerate() {
            let j0 = j_block.start;
            let j1 = j_block.end;
            for i_block in &direct_blocks[..=j_block_idx] {
                let i0 = i_block.start;
                let i1 = i_block.end;
                Self::tk_fill_gram_block(x_dense, z, i0, i1, j0, j1, gram);

                let design_gram = if has_design_deriv {
                    let x_theta = x_fixed.expect("design derivative checked above");
                    let rows = i1 - i0;
                    let cols = j1 - j0;
                    let mut block = Array2::<f64>::zeros((rows, cols));
                    let x_theta_i = x_theta.slice(s![i0..i1, ..]);
                    let z_j = z.slice(s![.., j0..j1]);
                    ndarray::linalg::general_mat_mul(1.0, &x_theta_i, &z_j, 0.0, &mut block);
                    let x_theta_j = x_theta.slice(s![j0..j1, ..]);
                    let z_i = z.slice(s![.., i0..i1]);
                    let mut reverse = Array2::<f64>::zeros((cols, rows));
                    ndarray::linalg::general_mat_mul(1.0, &x_theta_j, &z_i, 0.0, &mut reverse);
                    Some((block, reverse))
                } else {
                    None
                };

                let mut block_sum = 0.0_f64;
                for &(bi, _) in &i_block.entries {
                    let ii = i0 + bi;
                    let ci = c_array[ii];
                    let cpi = c_prime[ii];
                    for &(bj, _) in &j_block.entries {
                        let jj = j0 + bj;
                        let cj = c_array[jj];
                        let gij = gram[[bi, bj]];
                        let cpj = c_prime[jj];
                        let c_direct = (cpi * cj + ci * cpj) * gij * gij * gij / 12.0;
                        let k_direct = if let Some((forward, reverse)) = design_gram.as_ref() {
                            let kp = forward[[bi, bj]] + reverse[[bj, bi]];
                            0.25 * ci * cj * gij * gij * kp
                        } else {
                            0.0
                        };
                        block_sum += c_direct + k_direct;
                    }
                }
                let sym_factor = if i0 == j0 { 1.0 } else { 2.0 };
                c_term_prime += sym_factor * block_sum;
            }
        }

        let value = d_term_prime + c_term_prime + q_term_prime;
        if !value.is_finite() {
            return Err(EstimationError::InvalidInput(format!(
                "Tierney-Kadane direct c/d derivative produced non-finite value: {value}"
            )));
        }
        Ok(value)
    }

    fn tk_cd_direct_active_blocks(
        c_array: &Array1<f64>,
        c_prime: &Array1<f64>,
    ) -> Vec<TkActiveBlock> {
        let n = c_array.len();
        let mut blocks = Vec::with_capacity(n.div_ceil(TK_BLOCK_SIZE));
        for start in (0..n).step_by(TK_BLOCK_SIZE) {
            let end = (start + TK_BLOCK_SIZE).min(n);
            let mut entries = Vec::new();
            for offset in 0..(end - start) {
                let idx = start + offset;
                if c_array[idx] != 0.0 || c_prime[idx] != 0.0 {
                    entries.push((offset, 0.0));
                }
            }
            if !entries.is_empty() {
                blocks.push(TkActiveBlock {
                    start,
                    end,
                    entries,
                });
            }
        }
        blocks
    }

    fn tierney_kadane_analytic_core(
        &self,
        x_dense: &Array2<f64>,
        z: &Array2<f64>,
        c_array: &Array1<f64>,
        d_array: &Array1<f64>,
        e_array: &Array1<f64>,
        tk_penalties: &[crate::construction::CanonicalPenalty],
        lambdas: &[f64],
        ext_coords: &[super::unified::HyperCoord],
        beta: &Array1<f64>,
        firth_op: Option<&super::FirthDenseOperator>,
        compute_gradient: bool,
        h_inv_solve: &dyn Fn(&Array1<f64>) -> Result<Array1<f64>, EstimationError>,
    ) -> Result<TkCorrectionTerms, EstimationError> {
        let p = x_dense.ncols();
        let k = tk_penalties.len();

        let shared = Self::tk_shared_intermediates(
            x_dense,
            z,
            c_array,
            "Tierney-Kadane correction",
            h_inv_solve,
        )?;
        let mut gram = Array2::<f64>::zeros((TK_BLOCK_SIZE, TK_BLOCK_SIZE));
        let value = Self::tk_scalar_from_shared(x_dense, z, d_array, &shared, &mut gram)?;
        if !compute_gradient {
            return Ok(TkCorrectionTerms {
                value,
                gradient: None,
            });
        }

        let mut x_vks: Vec<Array1<f64>> = Vec::with_capacity(k + ext_coords.len());
        let mut beta_dirs: Vec<Array1<f64>> = Vec::with_capacity(k + ext_coords.len());
        for idx in 0..k {
            let cp = &tk_penalties[idx];
            let r = &cp.col_range;
            let beta_block = beta.slice(s![r.start..r.end]);
            let r_beta = cp.root.dot(&beta_block);
            let mut s_k_beta = Array1::<f64>::zeros(p);
            for a in 0..cp.block_dim() {
                s_k_beta[r.start + a] = (0..cp.rank())
                    .map(|row| cp.root[[row, a]] * r_beta[row])
                    .sum::<f64>();
            }
            let a_k_beta = &s_k_beta * lambdas[idx];
            let v_k = h_inv_solve(&a_k_beta)?;
            x_vks.push(x_dense.dot(&v_k));
            beta_dirs.push(v_k.mapv(|value| -value));
        }
        let mut ext_drifts = Vec::with_capacity(ext_coords.len());
        let mut ext_eta_fixed = Vec::with_capacity(ext_coords.len());
        let mut ext_x_fixed = Vec::with_capacity(ext_coords.len());
        for coord in ext_coords {
            let drift = coord.drift.materialize();
            if drift.ncols() != beta.len() || drift.nrows() != beta.len() {
                return Err(EstimationError::InvalidInput(format!(
                    "Tierney-Kadane ext drift shape mismatch: expected {}x{}, got {}x{}",
                    beta.len(),
                    beta.len(),
                    drift.nrows(),
                    drift.ncols()
                )));
            }
            if coord.g.len() != beta.len() {
                return Err(EstimationError::InvalidInput(format!(
                    "Tierney-Kadane ext mode RHS length mismatch: expected {}, got {}",
                    beta.len(),
                    coord.g.len()
                )));
            }
            let beta_theta = h_inv_solve(&coord.g)?;
            x_vks.push(x_dense.dot(&beta_theta));
            beta_dirs.push(beta_theta);
            ext_drifts.push(drift);
            ext_eta_fixed.push(coord.tk_eta_fixed.clone());
            ext_x_fixed.push(coord.tk_x_fixed.clone());
        }

        let gradient = Self::tk_gradient_from_shared(
            x_dense,
            z,
            c_array,
            d_array,
            e_array,
            tk_penalties,
            lambdas,
            &ext_drifts,
            &ext_eta_fixed,
            &ext_x_fixed,
            &x_vks,
            &beta_dirs,
            firth_op,
            &shared,
            &mut gram,
        )?;
        Ok(TkCorrectionTerms {
            value,
            gradient: Some(gradient),
        })
    }

    fn tierney_kadane_terms(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        mode: super::unified::EvalMode,
        ext_coords: &[super::unified::HyperCoord],
    ) -> Result<TkCorrectionTerms, EstimationError> {
        if self.config.link_function() == LinkFunction::Identity {
            return Ok(TkCorrectionTerms {
                value: 0.0,
                gradient: None,
            });
        }
        if !self.config.firth_bias_reduction {
            return Ok(TkCorrectionTerms {
                value: 0.0,
                gradient: None,
            });
        }

        let pirls_result = bundle.pirls_result.as_ref();
        let (c_array, d_array, e_array) = self.hessian_cde_arrays(pirls_result)?;
        if let Some(idx) = c_array.iter().position(|v| !v.is_finite()) {
            return Err(EstimationError::InvalidInput(format!(
                "Tierney-Kadane correction received non-finite c derivative at row {idx}: {}",
                c_array[idx]
            )));
        }
        if let Some(idx) = d_array.iter().position(|v| !v.is_finite()) {
            return Err(EstimationError::InvalidInput(format!(
                "Tierney-Kadane correction received non-finite d derivative at row {idx}: {}",
                d_array[idx]
            )));
        }
        let compute_gradient = compute_gradient_for_tk(mode);
        if mode == super::unified::EvalMode::ValueGradientHessian {
            return Err(EstimationError::InvalidInput(
                "Tierney-Kadane outer Hessian requires analytic second derivatives; production finite-difference stencils are disabled".to_string(),
            ));
        }
        if c_array.is_empty() || d_array.is_empty() {
            return Ok(TkCorrectionTerms {
                value: 0.0,
                gradient: if compute_gradient {
                    Some(Array1::zeros(rho.len() + ext_coords.len()))
                } else {
                    None
                },
            });
        }

        let n_x = self.x().nrows();
        let p_x = self.x().ncols();
        let dense_work = n_x.saturating_mul(p_x);
        if n_x > TK_MAX_OBSERVATIONS || p_x > TK_MAX_COEFFICIENTS || dense_work > TK_MAX_DENSE_WORK
        {
            return Err(EstimationError::InvalidInput(format!(
                "Tierney-Kadane correction requires dense small-model calculus; refusing to silently disable it for n={n_x}, p={p_x}, n*p={dense_work}"
            )));
        }

        if let Some(sparse) = bundle.sparse_exact.as_ref() {
            let x_dense = self
                .x()
                .try_to_dense_arc("frozen-curvature TK correction requires dense design access")
                .map_err(EstimationError::InvalidInput)?;
            let xt = x_dense.t().to_owned();
            let z_mat =
                crate::linalg::sparse_exact::solve_sparse_spdmulti(sparse.factor.as_ref(), &xt)?;
            let factor_ref = sparse.factor.clone();
            let h_inv_solve = |rhs: &Array1<f64>| -> Result<Array1<f64>, EstimationError> {
                crate::linalg::sparse_exact::solve_sparse_spd(&factor_ref, rhs)
            };
            let lambdas: Vec<f64> = rho.iter().map(|r| r.exp()).collect();
            let beta = self.sparse_exact_beta_original(pirls_result);
            let firth_op = if self.config.firth_bias_reduction
                && matches!(self.config.link_function(), LinkFunction::Logit)
            {
                if let Some(cached) = bundle.firth_dense_operator_original.as_ref() {
                    Some(cached.clone())
                } else {
                    Some(std::sync::Arc::new(Self::build_firth_dense_operator(
                        x_dense.as_ref(),
                        &pirls_result.final_eta,
                        self.weights,
                    )?))
                }
            } else {
                None
            };
            return self.tierney_kadane_analytic_core(
                x_dense.as_ref(),
                &z_mat,
                &c_array,
                &d_array,
                &e_array,
                &self.canonical_penalties,
                &lambdas,
                ext_coords,
                &beta,
                firth_op.as_deref(),
                compute_gradient,
                &h_inv_solve,
            );
        }

        let free_basis_opt = self.active_constraint_free_basis(pirls_result);
        let use_original_basis = matches!(
            pirls_result.coordinate_frame,
            pirls::PirlsCoordinateFrame::TransformedQs
        ) && free_basis_opt.is_none();
        let h_tk_source = if self.config.firth_bias_reduction {
            bundle.h_total.as_ref()
        } else {
            bundle.h_eff.as_ref()
        };
        let h_tk_eval = if use_original_basis {
            self.bundle_matrix_in_original_basis(pirls_result, h_tk_source)
        } else if let Some(z) = free_basis_opt.as_ref() {
            Self::projectwith_basis(h_tk_source, z)
        } else {
            h_tk_source.clone()
        };
        let x_eff_dense = if use_original_basis {
            self.x()
                .try_to_dense_arc("Tierney-Kadane correction requires dense original design access")
                .map_err(EstimationError::InvalidInput)?
                .as_ref()
                .clone()
        } else if let Some(z) = free_basis_opt.as_ref() {
            pirls_result.x_transformed.to_dense().dot(z)
        } else {
            pirls_result.x_transformed.to_dense()
        };

        let xt = x_eff_dense.t().to_owned();
        let p = x_eff_dense.ncols();
        let n = x_eff_dense.nrows();
        enum HFactor {
            Cholesky(crate::linalg::faer_ndarray::FaerCholeskyFactor),
            Eigh {
                evals: Array1<f64>,
                evecs: Array2<f64>,
            },
        }
        let h_factor = if let Ok(chol) = h_tk_eval.cholesky(Side::Lower) {
            HFactor::Cholesky(chol)
        } else if let Ok((evals, evecs)) = h_tk_eval.eigh(Side::Lower) {
            if let Some((idx, ev)) = evals.iter().enumerate().find(|(_, ev)| **ev <= 1e-12) {
                return Err(EstimationError::InvalidInput(format!(
                    "Tierney-Kadane correction requires a positive definite Hessian; eigenvalue {idx} is {ev}"
                )));
            }
            HFactor::Eigh { evals, evecs }
        } else {
            return Err(EstimationError::InvalidInput(
                "Tierney-Kadane correction could not factor the effective Hessian".to_string(),
            ));
        };

        let z_mat = match &h_factor {
            HFactor::Cholesky(chol) => {
                let mut solved = xt.clone();
                chol.solve_mat_in_place(&mut solved);
                solved
            }
            HFactor::Eigh { evals, evecs } => {
                let mut solved = Array2::<f64>::zeros((p, n));
                for m in 0..evals.len() {
                    let ev = evals[m];
                    let u = evecs.column(m);
                    let coeffs = xt.t().dot(&u).mapv(|v| v / ev);
                    for row in 0..p {
                        let u_row = u[row];
                        for col in 0..n {
                            solved[[row, col]] += u_row * coeffs[col];
                        }
                    }
                }
                solved
            }
        };

        let h_inv_solve = move |rhs: &Array1<f64>| -> Result<Array1<f64>, EstimationError> {
            match &h_factor {
                HFactor::Cholesky(chol) => Ok(chol.solvevec(rhs)),
                HFactor::Eigh { evals, evecs } => {
                    let mut sol = Array1::<f64>::zeros(rhs.len());
                    for m in 0..evals.len() {
                        let u = evecs.column(m);
                        let coeff = u.dot(rhs) / evals[m];
                        for row in 0..sol.len() {
                            sol[row] += coeff * u[row];
                        }
                    }
                    Ok(sol)
                }
            }
        };

        let p_eff = x_eff_dense.ncols();
        let tk_penalties: Vec<crate::construction::CanonicalPenalty> = if use_original_basis {
            self.canonical_penalties.as_ref().clone()
        } else if let Some(z) = free_basis_opt.as_ref() {
            pirls_result
                .reparam_result
                .canonical_transformed
                .iter()
                .map(|cp| {
                    crate::construction::CanonicalPenalty::from_dense_root(cp.root.dot(z), p_eff)
                })
                .collect()
        } else {
            pirls_result.reparam_result.canonical_transformed.clone()
        };
        let lambdas: Vec<f64> = rho.iter().map(|r| r.exp()).collect();
        let beta = if use_original_basis {
            self.sparse_exact_beta_original(pirls_result)
        } else if let Some(z) = free_basis_opt.as_ref() {
            z.t()
                .dot(&pirls_result.beta_transformed.as_ref().to_owned())
        } else {
            pirls_result.beta_transformed.as_ref().clone()
        };
        let firth_op = if self.config.firth_bias_reduction
            && matches!(self.config.link_function(), LinkFunction::Logit)
        {
            Some(std::sync::Arc::new(Self::build_firth_dense_operator(
                &x_eff_dense,
                &pirls_result.final_eta,
                self.weights,
            )?))
        } else {
            None
        };

        self.tierney_kadane_analytic_core(
            &x_eff_dense,
            &z_mat,
            &c_array,
            &d_array,
            &e_array,
            &tk_penalties,
            &lambdas,
            ext_coords,
            &beta,
            firth_op.as_deref(),
            compute_gradient,
            &h_inv_solve,
        )
    }

    fn validate_tk_ext_coords(
        &self,
        mode: super::unified::EvalMode,
        ext_coords: &[super::unified::HyperCoord],
    ) -> Result<(), EstimationError> {
        if self.config.link_function() == LinkFunction::Identity
            || !self.config.firth_bias_reduction
            || !compute_gradient_for_tk(mode)
        {
            return Ok(());
        }
        for (idx, coord) in ext_coords.iter().enumerate() {
            if coord.tk_eta_fixed.is_none() || coord.tk_x_fixed.is_none() {
                return Err(EstimationError::InvalidInput(format!(
                    "Tierney-Kadane external gradient coordinate {idx} is missing analytic fixed-beta design/eta derivative carriers"
                )));
            }
        }
        Ok(())
    }

    fn apply_tk_to_result(
        &self,
        mut result: super::unified::RemlLamlResult,
        tk_terms: TkCorrectionTerms,
    ) -> Result<super::unified::RemlLamlResult, EstimationError> {
        result.cost += tk_terms.value;
        if let (Some(ref mut grad), Some(tk_grad)) = (result.gradient.as_mut(), tk_terms.gradient) {
            if tk_grad.len() == grad.len() {
                **grad += &tk_grad;
            } else {
                // The unified evaluator returns one gradient entry per
                // (ρ, ext_coord) coordinate; the analytic Tierney–Kadane
                // propagation in `tk_gradient_from_shared` produces exactly
                // the same per-coordinate layout (`tk_penalties.len() +
                // ext_drifts.len()`).  An arity mismatch means the two
                // sides were assembled against different coordinate sets,
                // which would yield a structurally inconsistent total
                // gradient and is therefore rejected outright instead of
                // silently zero-padding or truncating.
                return Err(EstimationError::InvalidInput(format!(
                    "Tierney-Kadane gradient coordinate count mismatch: evaluator produced {} entries, analytic c/d propagation produced {}; this indicates the TK term and the unified evaluator were assembled against different coordinate sets",
                    grad.len(),
                    tk_grad.len()
                )));
            }
        }
        Ok(result)
    }

    pub(super) fn should_compute_hot_diagnostics(&self, eval_idx: u64) -> bool {
        // Keep expensive diagnostics out of the hot path unless they can
        // be surfaced. This has zero effect on optimization math.
        (log::log_enabled!(log::Level::Info) || log::log_enabled!(log::Level::Warn))
            && (eval_idx == 1 || eval_idx % 200 == 0)
    }

    fn invalidate_link_dependent_state(&self) {
        self.cache_manager.clear_eval_and_factor_caches();
        self.cache_manager.pirls_cache.write().unwrap().clear();
        self.warm_start_beta.write().unwrap().take();
    }

    pub(crate) fn set_link_states(
        &mut self,
        mixture_link_state: Option<crate::types::MixtureLinkState>,
        sas_link_state: Option<SasLinkState>,
    ) {
        let changed = self.runtime_mixture_link_state != mixture_link_state
            || self.runtime_sas_link_state != sas_link_state;
        if !changed {
            return;
        }
        self.runtime_mixture_link_state = mixture_link_state;
        self.runtime_sas_link_state = sas_link_state;
        self.invalidate_link_dependent_state();
    }

    /// Returns the eta-derivative carriers (c = dW/deta, d = d^2W/deta^2) for
    /// the exact Hessian surface that PIRLS accepted at the mode.
    ///
    /// For canonical links these coincide with the Fisher arrays; for
    /// non-canonical links they carry the clamped **observed-information**
    /// curvature used on the PIRLS left-hand side, including the residual-
    /// dependent corrections:
    ///   c_obs = c_Fisher + h'*B - (y-mu)*B_eta
    ///   d_obs = d_Fisher + h''*B + 2*h'*B_eta - (y-mu)*B_etaeta
    ///
    /// These flow into the outer REML gradient's C[v] correction term and
    /// the outer Hessian's Q[v_k, v_l] correction term. Using the observed
    /// versions is required for the exact Laplace approximation; Fisher
    /// versions would yield a PQL-type surrogate. See response.md Section 3.
    fn hessian_cd_arrays(
        &self,
        pirls_result: &PirlsResult,
    ) -> Result<(Array1<f64>, Array1<f64>), EstimationError> {
        Ok((
            pirls_result.solve_c_array.clone(),
            pirls_result.solve_d_array.clone(),
        ))
    }

    /// Returns the (c, d, e) per-row mode-curvature carriers required for
    /// the analytic Tierney–Kadane c/d derivative propagation.
    ///
    /// `c` and `d` are the same observed-information arrays returned by
    /// [`hessian_cd_arrays`] (cᵢ = ∂Wᵢ/∂ηᵢ, dᵢ = ∂²Wᵢ/∂ηᵢ²); `e` is the
    /// next term in the η-Taylor expansion, eᵢ = ∂³Wᵢ/∂ηᵢ³, so that
    /// ∂c/∂θ |_β = d ⊙ ∂η/∂θ and ∂d/∂θ |_β = e ⊙ ∂η/∂θ along any chain
    /// rule that flows through η.  This is precisely what
    /// `tk_direct_gradient_from_cd_and_design` consumes.
    ///
    /// For canonical Logit on a binomial likelihood, W = h'(η)²/V(μ)
    /// reduces to W = h'(η) (because V(μ) = h'(η) for the canonical
    /// pairing μ(1−μ)), so ∂³W/∂η³ = h''''(η) and the closed-form 5-jet
    /// (`mixture_link::logit_inverse_link_jet5`) gives that exactly. We
    /// clamp to zero in saturated tails where the jet is dominated by
    /// floating-point noise (matching the existing `c_array`/`d_array`
    /// saturation handling).
    ///
    /// For non-canonical Bernoulli links (Probit, CLogLog, SAS,
    /// BetaLogistic, Mixture) and non-Bernoulli families with
    /// non-trivial observed corrections we use the analytic
    /// [`pirls::e_obs_from_jets`] formula. It expresses
    ///   ∂³W_obs/∂η³ = W_F''' + h₃ T₁ + 3 h₂ T₂ + 3 h₁ T₃ − (y−μ) T₄
    /// where T = h₁/(φV), T_k = ∂^k T/∂η^k, and W_F = h₁ T. Everything
    /// is closed-form in h₁..h₅ (inverse-link jet plus
    /// `inverse_link_pdf{third,fourth}_derivative_for_inverse_link`)
    /// and the variance jet V, V₁..V₄. No finite-difference stencil is
    /// involved.
    fn hessian_cde_arrays(
        &self,
        pirls_result: &PirlsResult,
    ) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), EstimationError> {
        let c_array = pirls_result.solve_c_array.clone();
        let d_array = pirls_result.solve_d_array.clone();
        let n = d_array.len();
        let mut e_array = Array1::<f64>::zeros(n);

        let link_function = self.config.link_function();
        let inverse_link = if let Some(state) = self.runtime_mixture_link_state.clone() {
            InverseLink::Mixture(state)
        } else if let Some(state) = self.runtime_sas_link_state {
            if matches!(link_function, LinkFunction::BetaLogistic) {
                InverseLink::BetaLogistic(state)
            } else {
                InverseLink::Sas(state)
            }
        } else {
            InverseLink::Standard(link_function)
        };

        // Saturation guard mirrors `eta_for_observed_hessian_jet` in pirls:
        // for non-Logit / non-Log links eta is clamped to ±30 to keep the
        // jets within numerically tractable territory; outside that window
        // we report 0 to match the c/d saturation policy.
        let (clamp_lo, clamp_hi) = match link_function {
            LinkFunction::Logit | LinkFunction::Log => (-700.0_f64, 700.0_f64),
            LinkFunction::Identity => (f64::NEG_INFINITY, f64::INFINITY),
            LinkFunction::Probit
            | LinkFunction::CLogLog
            | LinkFunction::Sas
            | LinkFunction::BetaLogistic => (-30.0_f64, 30.0_f64),
        };

        // Canonical-Logit fast path: W = h'(η), so ∂³W/∂η³ = h''''(η)
        // taken from the dedicated 5-jet (no variance-jet machinery).
        // Mixture links advertise `link_function() == Logit` but are
        // non-canonical; route them through the general path below.
        let canonical_logit = matches!(link_function, LinkFunction::Logit)
            && self.runtime_mixture_link_state.is_none();

        if canonical_logit {
            for i in 0..n {
                let eta_raw = pirls_result.final_eta[i];
                let eta_used = eta_raw.clamp(clamp_lo, clamp_hi);
                if eta_raw != eta_used {
                    e_array[i] = 0.0;
                } else {
                    let jet = crate::mixture_link::logit_inverse_link_jet5(eta_used);
                    e_array[i] = self.weights[i].max(0.0) * jet.d4;
                }
            }
            return Ok((c_array, d_array, e_array));
        }

        // General observed-information path for non-canonical Bernoulli
        // links and other GLM families that support the observed Hessian
        // surface (Probit, CLogLog, SAS, BetaLogistic, Mixture, GammaLog).
        let likelihood = pirls_result.likelihood;
        let weight_family = pirls::weight_family_for_glm_likelihood(likelihood);
        let phi = likelihood.fixed_phi().unwrap_or(1.0);
        let dmu_deta = &pirls_result.solve_dmu_deta;
        let d2mu_deta2 = &pirls_result.solve_d2mu_deta2;
        let d3mu_deta3 = &pirls_result.solve_d3mu_deta3;
        if dmu_deta.len() != n || d2mu_deta2.len() != n || d3mu_deta3.len() != n {
            return Err(EstimationError::InvalidInput(format!(
                "Tierney-Kadane e_obs requires populated solve_*mu_deta arrays (n={}, dmu={}, d2mu={}, d3mu={}); ensure PIRLS rehydration ran",
                n,
                dmu_deta.len(),
                d2mu_deta2.len(),
                d3mu_deta3.len(),
            )));
        }
        let mu = &pirls_result.solvemu;
        if mu.len() != n {
            return Err(EstimationError::InvalidInput(format!(
                "Tierney-Kadane e_obs requires solvemu populated (n={}, len={})",
                n,
                mu.len(),
            )));
        }
        for i in 0..n {
            let eta_raw = pirls_result.final_eta[i];
            let eta_used = eta_raw.clamp(clamp_lo, clamp_hi);
            if eta_raw != eta_used {
                e_array[i] = 0.0;
                continue;
            }
            let h1 = dmu_deta[i];
            let h2 = d2mu_deta2[i];
            let h3 = d3mu_deta3[i];
            let h4 = crate::mixture_link::inverse_link_pdfthird_derivative_for_inverse_link(
                &inverse_link,
                eta_used,
            )?;
            let h5 = crate::mixture_link::inverse_link_pdffourth_derivative_for_inverse_link(
                &inverse_link,
                eta_used,
            )?;
            if !h1.is_finite()
                || !h2.is_finite()
                || !h3.is_finite()
                || !h4.is_finite()
                || !h5.is_finite()
            {
                e_array[i] = 0.0;
                continue;
            }
            let mu_i = mu[i];
            let vj = pirls::variance_jet_for_weight_family(weight_family, mu_i);
            if !(vj.v.is_finite() && vj.v > 0.0) {
                e_array[i] = 0.0;
                continue;
            }
            let pw = self.weights[i].max(0.0);
            let y_i = self.y[i];
            let e_i = pirls::e_obs_from_jets(y_i, mu_i, h1, h2, h3, h4, h5, vj, phi, pw);
            e_array[i] = if e_i.is_finite() { e_i } else { 0.0 };
        }

        Ok((c_array, d_array, e_array))
    }

    /// Compute soft prior cost without needing workspace
    pub(super) fn compute_soft_priorcost(&self, rho: &Array1<f64>) -> f64 {
        let len = rho.len();
        if len == 0 || RHO_SOFT_PRIOR_WEIGHT == 0.0 {
            return 0.0;
        }

        let inv_bound = 1.0 / RHO_BOUND;
        let sharp = RHO_SOFT_PRIOR_SHARPNESS;
        let mut cost = 0.0;
        for &ri in rho.iter() {
            let scaled = sharp * ri * inv_bound;
            cost += scaled.cosh().ln();
        }

        cost * RHO_SOFT_PRIOR_WEIGHT
    }

    /// Compute soft prior gradient without workspace mutation.
    pub(super) fn compute_soft_priorgrad(&self, rho: &Array1<f64>) -> Array1<f64> {
        let len = rho.len();
        let mut grad = Array1::<f64>::zeros(len);
        if len == 0 || RHO_SOFT_PRIOR_WEIGHT == 0.0 {
            return grad;
        }
        let inv_bound = 1.0 / RHO_BOUND;
        let sharp = RHO_SOFT_PRIOR_SHARPNESS;
        for (g, &ri) in grad.iter_mut().zip(rho.iter()) {
            let scaled = sharp * ri * inv_bound;
            *g = sharp * inv_bound * scaled.tanh() * RHO_SOFT_PRIOR_WEIGHT;
        }
        grad
    }

    /// Add the exact Hessian of the soft rho prior in place.
    ///
    /// Prior definition per coordinate:
    ///   C_i(rho_i) = w * log(cosh(a * rho_i)),
    ///   a = rho_soft_prior_sharpness / rho_bound,
    ///   w = rho_soft_prior_weight.
    ///
    /// Then:
    ///   dC_i/drho_i   = w * a * tanh(a * rho_i),
    ///   d²C_i/drho_i² = w * a² * sech²(a * rho_i)
    ///                = w * a² * (1 - tanh²(a * rho_i)).
    ///
    /// The prior is separable across coordinates, so off-diagonals are zero.
    pub(super) fn add_soft_priorhessian_in_place(&self, rho: &Array1<f64>, hess: &mut Array2<f64>) {
        let len = rho.len();
        if len == 0 || RHO_SOFT_PRIOR_WEIGHT == 0.0 {
            return;
        }
        let a = RHO_SOFT_PRIOR_SHARPNESS / RHO_BOUND;
        let prefactor = RHO_SOFT_PRIOR_WEIGHT * a * a;
        for i in 0..len {
            let t = (a * rho[i]).tanh();
            hess[[i, i]] += prefactor * (1.0 - t * t);
        }
    }

    /// Compute the soft prior Hessian as a standalone matrix.
    ///
    /// This is the diagonal matrix of second derivatives of the soft prior penalty.
    /// Returns `None` when the prior contributes zero curvature (empty ρ or zero weight).
    pub(super) fn compute_soft_priorhess(&self, rho: &Array1<f64>) -> Option<Array2<f64>> {
        let len = rho.len();
        if len == 0 || RHO_SOFT_PRIOR_WEIGHT == 0.0 {
            return None;
        }
        let mut hess = Array2::<f64>::zeros((len, len));
        self.add_soft_priorhessian_in_place(rho, &mut hess);
        if hess.iter().any(|&v| v != 0.0) {
            Some(hess)
        } else {
            None
        }
    }

    /// Returns the effective Hessian and the ridge value used (if any).
    /// Uses the same Hessian matrix in both cost and gradient calculations.
    ///
    /// PIRLS folds any stabilization ridge directly into the penalized objective:
    ///   l_p(beta; rho) = l(beta) - 0.5 * beta^T (S_lambda + ridge I) beta.
    /// Therefore the curvature used in LAML is
    ///   H_eff = X' W_H X + S_lambda + ridge I,
    /// and adding another ridge here places the Laplace expansion on a different surface.
    ///
    /// IMPORTANT: W_H here is the **observed-information** weight for non-canonical
    /// links (or Fisher weight for canonical links, where the two coincide). The
    /// `stabilizedhessian_transformed` from PIRLS was assembled with the Hessian-side
    /// weights (`lasthessian_weights`), which are observed when PIRLS accepted that
    /// curvature. This ensures log|H_eff| in the outer REML uses the correct Laplace
    /// Hessian. See response.md Section 3 for the mathematical justification.
    ///
    pub(super) fn effectivehessian(
        &self,
        pr: &PirlsResult,
    ) -> Result<(Array2<f64>, RidgePassport), EstimationError> {
        // Verify PD via SymmetricMatrix::factorize() (sparse-aware),
        // then densify for the spectral evaluator which needs eigh().
        let h = &pr.stabilizedhessian_transformed;
        if h.factorize().is_ok() {
            return Ok((h.to_dense(), pr.ridge_passport));
        }

        Err(EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        })
    }

    pub(crate) fn newwith_offset<X>(
        y: ArrayView1<'a, f64>,
        x: X,
        weights: ArrayView1<'a, f64>,
        offset: ArrayView1<'_, f64>,
        canonical_penalties: Vec<crate::construction::CanonicalPenalty>,
        p: usize,
        config: &'a RemlConfig,
        nullspace_dims: Option<Vec<usize>>,
        coefficient_lower_bounds: Option<Array1<f64>>,
        linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,
    ) -> Result<Self, EstimationError>
    where
        X: Into<DesignMatrix>,
    {
        Self::newwith_offset_shared(
            y,
            x,
            weights,
            offset,
            Arc::new(canonical_penalties),
            p,
            Arc::new(config.clone()),
            nullspace_dims,
            coefficient_lower_bounds,
            linear_constraints,
        )
    }

    pub(crate) fn newwith_offset_shared<X>(
        y: ArrayView1<'a, f64>,
        x: X,
        weights: ArrayView1<'a, f64>,
        offset: ArrayView1<'_, f64>,
        canonical_penalties: Arc<Vec<crate::construction::CanonicalPenalty>>,
        p: usize,
        config: Arc<RemlConfig>,
        nullspace_dims: Option<Vec<usize>>,
        coefficient_lower_bounds: Option<Array1<f64>>,
        linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,
    ) -> Result<Self, EstimationError>
    where
        X: Into<DesignMatrix>,
    {
        let x = x.into();

        let expected_len = canonical_penalties.len();
        let nullspace_dims = match nullspace_dims {
            Some(dims) => {
                if dims.len() != expected_len {
                    return Err(EstimationError::InvalidInput(format!(
                        "nullspace_dims length {} does not match penalties {}",
                        dims.len(),
                        expected_len
                    )));
                }
                dims
            }
            None => vec![0; expected_len],
        };

        let balanced_penalty_root =
            create_balanced_penalty_root_from_canonical(&canonical_penalties, p)?;
        let reparam_invariant =
            precompute_reparam_invariant_from_canonical(&canonical_penalties, p)?;

        let sparse_penalty_blocks =
            build_sparse_penalty_blocks_from_canonical(canonical_penalties.as_ref(), p)?
                .map(Arc::new);

        let runtime_mixture_link_state = config.link_kind.mixture_state().cloned();
        let runtime_sas_link_state = config.link_kind.sas_state().copied();

        Ok(Self {
            y,
            x,
            weights,
            offset: offset.to_owned(),
            canonical_penalties,
            balanced_penalty_root,
            reparam_invariant,
            sparse_penalty_blocks,
            p,
            config,
            runtime_mixture_link_state,
            runtime_sas_link_state,
            nullspace_dims,
            coefficient_lower_bounds,
            linear_constraints,
            penalty_shrinkage_floor: None,
            cache_manager: EvalCacheManager::new(),
            arena: RemlArena::new(),
            warm_start_beta: RwLock::new(None),
            warm_start_enabled: AtomicBool::new(true),
            outer_hessian_downgrade_logged: AtomicBool::new(false),
            screening_max_inner_iterations: Arc::new(AtomicUsize::new(0)),
            kronecker_penalty_system: None,
            kronecker_factored: None,
        })
    }

    pub(crate) fn reset_surface<X>(
        &mut self,
        x: X,
        canonical_penalties: Arc<Vec<crate::construction::CanonicalPenalty>>,
        p: usize,
        nullspace_dims: Vec<usize>,
        coefficient_lower_bounds: Option<Array1<f64>>,
        linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,
        kronecker_penalty_system: Option<crate::smooth::KroneckerPenaltySystem>,
        kronecker_factored: Option<crate::basis::KroneckerFactoredBasis>,
    ) -> Result<(), EstimationError>
    where
        X: Into<DesignMatrix>,
    {
        let expected_len = canonical_penalties.len();
        if nullspace_dims.len() != expected_len {
            return Err(EstimationError::InvalidInput(format!(
                "nullspace_dims length {} does not match penalties {}",
                nullspace_dims.len(),
                expected_len
            )));
        }

        let balanced_penalty_root =
            create_balanced_penalty_root_from_canonical(&canonical_penalties, p)?;
        let reparam_invariant =
            precompute_reparam_invariant_from_canonical(&canonical_penalties, p)?;
        let sparse_penalty_blocks =
            build_sparse_penalty_blocks_from_canonical(canonical_penalties.as_ref(), p)?
                .map(Arc::new);

        self.x = x.into();
        self.canonical_penalties = canonical_penalties;
        self.balanced_penalty_root = balanced_penalty_root;
        self.reparam_invariant = reparam_invariant;
        self.sparse_penalty_blocks = sparse_penalty_blocks;
        self.p = p;
        self.nullspace_dims = nullspace_dims;
        self.coefficient_lower_bounds = coefficient_lower_bounds;
        self.linear_constraints = linear_constraints;
        self.kronecker_penalty_system = kronecker_penalty_system;
        self.kronecker_factored = kronecker_factored;
        self.cache_manager.clear_eval_and_factor_caches();
        self.cache_manager.pirls_cache.write().unwrap().clear();
        Ok(())
    }

    /// Inject Kronecker penalty system metadata for tensor-product smooth terms.
    ///
    /// When set, the REML evaluator will use O(∏q_j) logdet instead of O(p³)
    /// eigendecomposition.  Also stores the full factored basis so that P-IRLS
    /// can use factored reparameterization (Qs = U_1 ⊗ ... ⊗ U_d).
    pub(crate) fn set_kronecker_penalty_system(
        &mut self,
        system: crate::smooth::KroneckerPenaltySystem,
    ) {
        self.kronecker_penalty_system = Some(system);
    }

    /// Inject the full Kronecker factored basis for P-IRLS factored reparameterization.
    pub(crate) fn set_kronecker_factored(
        &mut self,
        factored: crate::basis::KroneckerFactoredBasis,
    ) {
        self.kronecker_factored = Some(factored);
    }

    /// Sets the shrinkage floor for penalized block eigenvalues.
    /// The ridge magnitude will be `epsilon * max_balanced_eigenvalue` (rho-independent).
    /// This prevents barely-penalized directions from causing pathological
    /// non-Gaussianity in the posterior. Typical value: `Some(1e-6)`.
    pub(crate) fn set_penalty_shrinkage_floor(&mut self, floor: Option<f64>) {
        self.penalty_shrinkage_floor = floor;
    }

    /// Creates a sanitized cache key from rho values.
    /// Returns None if any component is NaN, in which case caching is skipped.
    /// Maps -0.0 to 0.0 to ensure consistency in caching.
    pub(super) fn rhokey_sanitized(&self, rho: &Array1<f64>) -> Option<Vec<u64>> {
        EvalCacheManager::sanitized_rhokey(rho)
    }

    pub(super) fn prepare_eval_bundlewithkey(
        &self,
        rho: &Array1<f64>,
        key: Option<Vec<u64>>,
    ) -> Result<EvalShared, EstimationError> {
        let decision = self.select_reml_geometry(rho);
        match decision.geometry {
            RemlGeometry::SparseExactSpd => {
                match self.prepare_sparse_eval_bundlewithkey(rho, key.clone()) {
                    Ok(bundle) => {
                        log::info!(
                            "[reml-geometry] sparse_exact_spd reason={} p={} nnz_x={} nnz_h_est={} density_h_est={}",
                            decision.reason,
                            decision.p,
                            decision.nnz_x,
                            decision
                                .nnz_h_upper_est
                                .map(|v| v.to_string())
                                .unwrap_or_else(|| "na".to_string()),
                            decision
                                .density_h_upper_est
                                .map(|v| format!("{v:.4}"))
                                .unwrap_or_else(|| "na".to_string()),
                        );
                        Ok(bundle)
                    }
                    Err(err) => {
                        log::warn!(
                            "[reml-geometry] sparse_exact_spd failed ({}); falling back to dense spectral",
                            err
                        );
                        self.prepare_dense_eval_bundlewithkey(rho, key)
                    }
                }
            }
            RemlGeometry::DenseSpectral => self.prepare_dense_eval_bundlewithkey(rho, key),
        }
    }

    pub(crate) fn obtain_eval_bundle(
        &self,
        rho: &Array1<f64>,
    ) -> Result<EvalShared, EstimationError> {
        let key = self.rhokey_sanitized(rho);
        if let Some(existing) = self.cache_manager.cached_eval_bundle(&key) {
            return Ok(existing.clone());
        }
        let bundle = self.prepare_eval_bundlewithkey(rho, key)?;
        self.cache_manager.store_eval_bundle(bundle.clone());
        Ok(bundle)
    }

    pub(crate) fn objective_innerhessian(
        &self,
        rho: &Array1<f64>,
    ) -> Result<Array2<f64>, EstimationError> {
        let bundle = self.obtain_eval_bundle(rho)?;
        Ok(bundle.h_total.as_ref().clone())
    }

    pub(super) fn active_constraint_free_basis(&self, pr: &PirlsResult) -> Option<Array2<f64>> {
        let lin = pr.linear_constraints_transformed.as_ref()?;
        let active_tol = 1e-8;
        let beta_t = pr.beta_transformed.as_ref();
        let mut activerows: Vec<Array1<f64>> = Vec::new();
        for i in 0..lin.a.nrows() {
            let slack = lin.a.row(i).dot(beta_t) - lin.b[i];
            if slack <= active_tol {
                activerows.push(lin.a.row(i).to_owned());
            }
        }
        if activerows.is_empty() {
            return None;
        }

        let p_t = lin.a.ncols();
        let mut a_t = Array2::<f64>::zeros((p_t, activerows.len()));
        for (j, row) in activerows.iter().enumerate() {
            for k in 0..p_t {
                a_t[[k, j]] = row[k];
            }
        }

        let qrow = Self::orthonormalize_columns(&a_t, 1e-10); // basis for active row-space^T
        let rank = qrow.ncols();
        if rank == 0 {
            return None;
        }
        if rank >= p_t {
            return Some(Array2::<f64>::zeros((p_t, 0)));
        }

        // Build orthonormal basis for null(A_active) as complement of row-space.
        let mut z = Array2::<f64>::zeros((p_t, p_t - rank));
        let mut kept = 0usize;
        for j in 0..p_t {
            let mut v = Array1::<f64>::zeros(p_t);
            v[j] = 1.0;
            for t in 0..rank {
                let qt = qrow.column(t);
                let proj = qt.dot(&v);
                v -= &qt.mapv(|x| x * proj);
            }
            for t in 0..kept {
                let zt = z.column(t);
                let proj = zt.dot(&v);
                v -= &zt.mapv(|x| x * proj);
            }
            let nrm = v.dot(&v).sqrt();
            if nrm > 1e-10 {
                z.column_mut(kept).assign(&v.mapv(|x| x / nrm));
                kept += 1;
                if kept == p_t - rank {
                    break;
                }
            }
        }
        Some(z.slice(ndarray::s![.., 0..kept]).to_owned())
    }

    /// Construct a `BarrierConfig` from linear inequality constraints `A β ≥ b`
    /// by extracting rows that represent simple coordinate bounds (β_j ≥ b_i).
    ///
    /// Delegates to `BarrierConfig::from_constraints` and logs a diagnostic
    /// barrier-curvature check at a test point near the bounds.
    fn barrier_config_from_constraints(
        constraints: &crate::pirls::LinearInequalityConstraints,
    ) -> Option<super::unified::BarrierConfig> {
        let config = super::unified::BarrierConfig::from_constraints(Some(constraints))?;
        // Diagnostic: check curvature significance at a test point near bounds.
        {
            let max_idx = config
                .constrained_indices
                .iter()
                .max()
                .copied()
                .unwrap_or(0);
            let mut beta_test = Array1::<f64>::zeros(max_idx + 1);
            for (&idx, &lb) in config
                .constrained_indices
                .iter()
                .zip(config.lower_bounds.iter())
            {
                beta_test[idx] = lb + 0.01; // slightly above constraint
            }
            let significant = config.barrier_curvature_is_significant(&beta_test, 1.0, 0.05);
            log::trace!(
                "[barrier] curvature significant={significant} (tau={:.2e}, n_constrained={})",
                config.tau,
                config.constrained_indices.len(),
            );
        }
        Some(config)
    }

    pub(super) fn enforce_constraint_kkt(&self, pr: &PirlsResult) -> Result<(), EstimationError> {
        let Some(kkt) = pr.constraint_kkt.as_ref() else {
            return Ok(());
        };
        let tol_primal = 1e-7;
        let tol_dual = 1e-7;
        let tol_comp = 1e-7;
        let tol_stat = 5e-6;
        if kkt.primal_feasibility > tol_primal
            || kkt.dual_feasibility > tol_dual
            || kkt.complementarity > tol_comp
            || kkt.stationarity > tol_stat
        {
            let mut worstrow_msg = String::new();
            if let Some(lin) = pr.linear_constraints_transformed.as_ref() {
                let mut worst = 0.0_f64;
                let mut worstrow = 0usize;
                for i in 0..lin.a.nrows() {
                    let slack = lin.a.row(i).dot(&pr.beta_transformed.0) - lin.b[i];
                    let viol = (-slack).max(0.0);
                    if viol > worst {
                        worst = viol;
                        worstrow = i;
                    }
                }
                if worst > 0.0 {
                    worstrow_msg = format!("; worstrow={} worstviolation={:.3e}", worstrow, worst);
                }
            }
            return Err(EstimationError::ParameterConstraintViolation(format!(
                "KKT residuals exceed tolerance: primal={:.3e}, dual={:.3e}, comp={:.3e}, stat={:.3e}; active={}/{}{}",
                kkt.primal_feasibility,
                kkt.dual_feasibility,
                kkt.complementarity,
                kkt.stationarity,
                kkt.n_active,
                kkt.n_constraints,
                worstrow_msg
            )));
        }
        Ok(())
    }

    pub(super) fn projectwith_basis(matrix: &Array2<f64>, z: &Array2<f64>) -> Array2<f64> {
        let zt_m = z.t().dot(matrix);
        zt_m.dot(z)
    }

    /// Compute the structural penalty rank and exact pseudo-logdet log|S|₊.
    ///
    /// Uses the exact pseudo-logdet on the positive eigenspace:
    ///   L(S) = Σ_{σ_i > ε} log σ_i
    /// where ε is a relative threshold identifying the structural nullspace
    /// directly from the eigenspectrum. No δ-regularization or nullity metadata.
    ///
    /// For S(ρ) = Σ exp(ρ_k) S_k with S_k ⪰ 0, the nullspace N(S) = ∩_k N(S_k)
    /// is structurally fixed, so this is C∞ in ρ.
    pub(super) fn fixed_subspace_penalty_rank_and_logdet(
        &self,
        e_transformed: &Array2<f64>,
        ridge_passport: RidgePassport,
    ) -> Result<(usize, f64), EstimationError> {
        let structural_rank = e_transformed.nrows().min(e_transformed.ncols());
        if structural_rank == 0 {
            return Ok((0, 0.0));
        }

        let mut s_lambda = e_transformed.t().dot(e_transformed);
        let ridge = ridge_passport.penalty_logdet_ridge();
        if ridge > 0.0 {
            for i in 0..s_lambda.nrows() {
                s_lambda[[i, i]] += ridge;
            }
        }
        let (evals, _) = s_lambda
            .eigh(Side::Lower)
            .map_err(EstimationError::EigendecompositionFailed)?;

        // Exact pseudo-logdet on the positive eigenspace: L = Σ_{σ_i > ε} log σ_i.
        // The structural nullspace is identified directly from the eigenspectrum.
        let threshold = super::unified::positive_eigenvalue_threshold(evals.as_slice().unwrap());
        let log_det = super::unified::exact_pseudo_logdet(evals.as_slice().unwrap(), threshold);

        Ok((structural_rank, log_det))
    }

    /// Compute the projected logdet `log|U_Sᵀ H U_S|_+` together with the
    /// matching trace kernel `(U_S, (U_Sᵀ H U_S)⁻¹)` — the [`PenaltySubspaceTrace`]
    /// that pairs the rank-deficient LAML cost with a gradient-consistent
    /// `tr(U_S · (U_Sᵀ H U_S)⁻¹ · U_Sᵀ · Ḣ)` on the identified subspace.
    ///
    /// Computing both in one pass amortises the `U_S` + `H_proj` work that
    /// otherwise has to be redone separately in the cost path and the
    /// gradient trace path.  The kernel is `None` only when the penalty is
    /// structurally null (no identified subspace to project onto).
    ///
    /// See [`PenaltySubspaceTrace`] for the full Schur-complement derivation
    /// of why the full-space `G_ε(H)` kernel mismatches the projected cost
    /// for non-Gaussian families.
    pub(super) fn fixed_subspace_hessian_projected_parts(
        &self,
        h_total: &Array2<f64>,
        e_transformed: &Array2<f64>,
        ridge_passport: RidgePassport,
    ) -> Result<(f64, Option<super::unified::PenaltySubspaceTrace>), EstimationError> {
        let p = h_total.ncols();
        if p == 0 {
            return Ok((0.0, None));
        }
        if h_total.nrows() != p {
            return Err(EstimationError::InvalidInput(format!(
                "fixed_subspace_hessian_projected_parts: H must be square, got {}x{}",
                h_total.nrows(),
                p
            )));
        }
        if e_transformed.ncols() != p {
            return Err(EstimationError::InvalidInput(format!(
                "fixed_subspace_hessian_projected_parts: E cols {} do not match H dim {}",
                e_transformed.ncols(),
                p
            )));
        }

        let mut s_lambda = e_transformed.t().dot(e_transformed);
        let ridge = ridge_passport.penalty_logdet_ridge();
        if ridge > 0.0 {
            for i in 0..p {
                s_lambda[[i, i]] += ridge;
            }
        }
        let (s_evals, s_evecs) = s_lambda
            .eigh(Side::Lower)
            .map_err(EstimationError::EigendecompositionFailed)?;

        let s_thr = super::unified::positive_eigenvalue_threshold(s_evals.as_slice().unwrap());
        let positive_cols: Vec<usize> = (0..p).filter(|&j| s_evals[j] > s_thr).collect();
        let r = positive_cols.len();
        if r == 0 {
            // Penalty is structurally null → there is no identified
            // subspace to project onto.  Returning 0.0 lets the caller
            // leave the correction untouched; no penalty means there is
            // no pair to balance in the LAML ratio.
            return Ok((0.0, None));
        }

        // Build U_S: p × r basis of range(S_+).
        let mut u_s = Array2::<f64>::zeros((p, r));
        for (out_col, &src_col) in positive_cols.iter().enumerate() {
            for row in 0..p {
                u_s[[row, out_col]] = s_evecs[[row, src_col]];
            }
        }

        // H_proj = U_S^T · H · U_S is r × r, symmetric in exact
        // arithmetic.  Force exact symmetry before eigh — faer rejects
        // visibly non-symmetric input, and the two halves should match
        // up to roundoff.
        let h_times_u = h_total.dot(&u_s);
        let mut h_proj = u_s.t().dot(&h_times_u);
        for i in 0..r {
            for j in (i + 1)..r {
                let avg = 0.5 * (h_proj[[i, j]] + h_proj[[j, i]]);
                h_proj[[i, j]] = avg;
                h_proj[[j, i]] = avg;
            }
        }

        let (h_proj_evals, h_proj_evecs) = h_proj
            .eigh(Side::Lower)
            .map_err(EstimationError::EigendecompositionFailed)?;
        let h_thr = super::unified::positive_eigenvalue_threshold(h_proj_evals.as_slice().unwrap());
        let log_det = super::unified::exact_pseudo_logdet(h_proj_evals.as_slice().unwrap(), h_thr);

        // H_proj⁻¹ on the positive eigenspace.  Directions whose eigenvalue
        // sits at or below the pseudo-determinant threshold are excluded so
        // the kernel does not pick up divergent contributions from numerical
        // zeros — matching the filter applied to `log_det`.
        let mut h_proj_inverse = Array2::<f64>::zeros((r, r));
        for a in 0..r {
            let sigma = h_proj_evals[a];
            if sigma <= h_thr {
                continue;
            }
            let inv = 1.0 / sigma;
            for i in 0..r {
                for j in 0..r {
                    h_proj_inverse[[i, j]] += inv * h_proj_evecs[[i, a]] * h_proj_evecs[[j, a]];
                }
            }
        }

        Ok((
            log_det,
            Some(super::unified::PenaltySubspaceTrace {
                u_s,
                h_proj_inverse,
            }),
        ))
    }

    /// Compute tr(S⁺ S_direction) — the first derivative of log|S|₊
    /// in the direction S_direction.
    ///
    /// Uses the exact pseudoinverse S⁺ restricted to the positive eigenspace.
    /// Only eigenvectors with eigenvalues above the structural threshold
    /// participate.
    pub(super) fn fixed_subspace_penalty_trace(
        &self,
        e_transformed: &Array2<f64>,
        s_direction: &Array2<f64>,
        ridge_passport: RidgePassport,
    ) -> Result<f64, EstimationError> {
        let (structural_rank, _) =
            self.fixed_subspace_penalty_rank_and_logdet(e_transformed, ridge_passport)?;
        let p_dim = e_transformed.ncols();
        if structural_rank == 0 || p_dim == 0 {
            return Ok(0.0);
        }

        let mut s_lambda = e_transformed.t().dot(e_transformed);
        let ridge = ridge_passport.penalty_logdet_ridge();
        if ridge > 0.0 {
            for i in 0..p_dim {
                s_lambda[[i, i]] += ridge;
            }
        }
        let (evals, evecs) = s_lambda
            .eigh(Side::Lower)
            .map_err(EstimationError::EigendecompositionFailed)?;

        // Exact pseudoinverse trace: tr(S⁺ S_direction) on the positive eigenspace.
        let threshold = super::unified::positive_eigenvalue_threshold(evals.as_slice().unwrap());

        let mut trace = 0.0;
        for idx in 0..p_dim {
            let ev = evals[idx];
            if ev > threshold {
                let u = evecs.column(idx).to_owned();
                let spsi_u = s_direction.dot(&u);
                trace += u.dot(&spsi_u) / ev;
            }
        }
        Ok(trace)
    }

    pub(super) fn updatewarm_start_from(&self, pr: &PirlsResult) {
        if !self.warm_start_enabled.load(Ordering::Relaxed) {
            return;
        }
        match pr.status {
            pirls::PirlsStatus::Converged | pirls::PirlsStatus::StalledAtValidMinimum => {
                let beta_original = match pr.coordinate_frame {
                    pirls::PirlsCoordinateFrame::OriginalSparseNative => {
                        pr.beta_transformed.as_ref().clone()
                    }
                    pirls::PirlsCoordinateFrame::TransformedQs => {
                        pr.reparam_result.qs.dot(pr.beta_transformed.as_ref())
                    }
                };
                self.warm_start_beta
                    .write()
                    .unwrap()
                    .replace(Coefficients::new(beta_original));
            }
            _ => {
                self.warm_start_beta.write().unwrap().take();
            }
        }
    }

    pub(crate) fn setwarm_start_original_beta(&self, beta_original: Option<ArrayView1<'_, f64>>) {
        if !self.warm_start_enabled.load(Ordering::Relaxed) {
            return;
        }
        match beta_original {
            Some(beta) if beta.len() == self.p => {
                self.warm_start_beta
                    .write()
                    .unwrap()
                    .replace(Coefficients::new(beta.to_owned()));
            }
            _ => {
                self.warm_start_beta.write().unwrap().take();
            }
        }
    }

    pub(crate) fn reset_outer_seed_state(&self) {
        self.cache_manager.invalidate_eval_bundle();
        self.warm_start_beta.write().unwrap().take();
    }

    // Accessor methods for private fields
    pub(crate) fn x(&self) -> &DesignMatrix {
        &self.x
    }

    pub(crate) fn balanced_penalty_root(&self) -> &Array2<f64> {
        &self.balanced_penalty_root
    }

    pub(crate) fn canonical_penalties(&self) -> &[crate::construction::CanonicalPenalty] {
        &self.canonical_penalties
    }

    /// Compute the exact pseudo-logdet for the sparse penalty path.
    ///
    /// For each sparse penalty block k with precomputed positive eigenvalues
    /// {σ_1, ..., σ_r}, the exact pseudo-logdet of λ_k S_k is:
    ///
    ///   L(λ_k S_k) = Σ_i log(λ_k σ_i) = r ρ_k + Σ_i log σ_i
    ///
    /// The first derivative is:
    ///   ∂/∂ρ_k L = r  (the rank, i.e. number of positive eigenvalues)
    pub(super) fn sparse_penalty_logdet_runtime(
        &self,
        rho: &Array1<f64>,
        blocks: &[SparsePenaltyBlock],
    ) -> (f64, Array1<f64>) {
        let mut logdet = 0.0_f64;
        let mut det1 = Array1::<f64>::zeros(rho.len());
        for block in blocks {
            let lambda_k = rho[block.term_index].exp();

            // L(λ_k S_k) = Σ_{positive} log(λ_k σ_i).
            for &eig in block.positive_eigenvalues.iter() {
                logdet += (lambda_k * eig).ln();
            }

            // ∂/∂ρ_k L = rank (number of positive eigenvalues).
            // Since ∂/∂ρ_k log(λ_k σ_i) = ∂/∂ρ_k (ρ_k + log σ_i) = 1.
            if block.term_index < det1.len() {
                det1[block.term_index] = block.positive_eigenvalues.len() as f64;
            }
        }
        (logdet, det1)
    }

    pub(super) fn prepare_dense_eval_bundlewithkey(
        &self,
        rho: &Array1<f64>,
        key: Option<Vec<u64>>,
    ) -> Result<EvalShared, EstimationError> {
        let pirls_result = self.execute_pirls_if_needed(rho)?;
        let (h_eff, ridge_passport) = self.effectivehessian(pirls_result.as_ref())?;

        let mut h_total = h_eff.clone();
        let mut firth_dense_operator: Option<Arc<FirthDenseOperator>> = None;
        if self.config.firth_bias_reduction
            && matches!(self.config.link_function(), LinkFunction::Logit)
        {
            let firth_n = pirls_result.x_transformed.nrows();
            let firth_p = pirls_result.x_transformed.ncols();
            if !super::firth_problem_scale_allows(firth_n, firth_p) {
                log::info!(
                    "disabling Firth bias reduction for large model (n={}, p={}, n*p={}, n*p^2={}): \
                     exact Firth operator is small-model-only",
                    firth_n,
                    firth_p,
                    firth_n.saturating_mul(firth_p),
                    firth_n.saturating_mul(firth_p).saturating_mul(firth_p),
                );
            } else {
                let x_dense = pirls_result
                .x_transformed
                .try_to_dense_arc(
                    "dense REML eval bundle requires dense transformed design for Firth operator",
                )
                .map_err(EstimationError::InvalidInput)?;
                let firth_build_start = std::time::Instant::now();
                let firth_op = Arc::new(Self::build_firth_dense_operator(
                    x_dense.as_ref(),
                    &pirls_result.final_eta,
                    self.weights,
                )?);
                log::info!(
                    "[Firth-op] build n={} p={} r={} half_logdet={:.3e} elapsed={:.3}s",
                    firth_op.x_dense.nrows(),
                    firth_op.x_dense.ncols(),
                    firth_op.k_reduced.nrows(),
                    firth_op.half_log_det,
                    firth_build_start.elapsed().as_secs_f64(),
                );
                // Firth-adjusted inner Jacobian for implicit differentiation:
                //   H_total = Xᵀ W X + S - H_φ,
                //   H_φ     = ∇²_β Φ
                //          = 0.5 [ Xᵀ diag(w'' ⊙ h) X - Bᵀ P B ].
                // This keeps B_k/B_{kl} solves on the same objective surface as
                // the Firth-augmented stationarity system.
                //
                // Conceptually Φ is the identifiable-subspace Jeffreys term
                // obtained by evaluating W on a canonical orthonormal basis of
                // the transformed design column space. The hphi block below is
                // therefore the curvature of that basis-invariant penalty,
                // represented in the current transformed basis.
                let mut weighted_xtdx = Array2::<f64>::zeros((0, 0));
                let diag_term = Self::xt_diag_x_dense_into(
                    &firth_op.x_dense,
                    &(&firth_op.w2 * &firth_op.h_diag),
                    &mut weighted_xtdx,
                );
                let bpb = fast_ab(&firth_op.b_base.t().to_owned(), &firth_op.p_b_base);
                let mut hphi = 0.5 * (diag_term - bpb);
                // Numerical symmetry guard.
                for i in 0..hphi.nrows() {
                    for j in 0..i {
                        let avg = 0.5 * (hphi[[i, j]] + hphi[[j, i]]);
                        hphi[[i, j]] = avg;
                        hphi[[j, i]] = avg;
                    }
                }
                // Keep tiny numerical noise from making the solve surface less stable.
                if hphi.iter().all(|v| v.is_finite()) {
                    h_total -= &hphi;
                }
                firth_dense_operator = Some(firth_op);
            } // else (not too large for Firth)
        }

        // Add log-barrier Hessian diagonal for monotonicity-constrained coefficients.
        // This augments the penalized Hessian before the spectral decomposition so
        // that logdet, trace, and solve operations all reflect the barrier curvature.
        if let Some(ref lin) = pirls_result.linear_constraints_transformed {
            if let Some(barrier_cfg) = Self::barrier_config_from_constraints(lin) {
                let beta_t = pirls_result.beta_transformed.as_ref();
                if let Err(e) = barrier_cfg.add_barrier_hessian_diagonal(&mut h_total, beta_t) {
                    log::warn!("Barrier Hessian diagonal skipped: {e}");
                }
            }
        }

        Ok(EvalShared {
            key,
            pirls_result,
            ridge_passport,
            geometry: RemlGeometry::DenseSpectral,
            h_eff: Arc::new(h_eff),
            h_total: Arc::new(h_total),
            sparse_exact: None,
            firth_dense_operator,
            firth_dense_operator_original: None,
        })
    }

    pub(super) fn prepare_sparse_eval_bundlewithkey(
        &self,
        rho: &Array1<f64>,
        key: Option<Vec<u64>>,
    ) -> Result<EvalShared, EstimationError> {
        let pirls_result = self.execute_pirls_if_needed(rho)?;
        if !matches!(
            pirls_result.coordinate_frame,
            pirls::PirlsCoordinateFrame::OriginalSparseNative
        ) {
            return Err(EstimationError::InvalidInput(
                "sparse exact geometry requires sparse-native PIRLS coordinates".to_string(),
            ));
        }
        let ridge_passport = pirls_result.ridge_passport;
        let x_sparse = self.x().as_sparse().ok_or_else(|| {
            EstimationError::InvalidInput(
                "sparse exact geometry requires sparse original design".to_string(),
            )
        })?;
        let penalty_blocks = self
            .sparse_penalty_blocks
            .as_ref()
            .ok_or_else(|| {
                EstimationError::InvalidInput(
                    "sparse exact geometry requires block-separable penalties".to_string(),
                )
            })?
            .clone();

        let lambdas = rho.mapv(f64::exp);
        let mut s_lambda = Array2::<f64>::zeros((self.p, self.p));
        for (k, cp) in self.canonical_penalties.iter().enumerate() {
            if k < lambdas.len() && lambdas[k] != 0.0 {
                cp.accumulate_weighted(&mut s_lambda, lambdas[k]);
            }
        }
        // Add log-barrier Hessian diagonal for monotonicity-constrained
        // coefficients (sparse path uses original coordinates).
        if let Some(ref lin) = self.linear_constraints {
            if let Some(barrier_cfg) = Self::barrier_config_from_constraints(lin) {
                let beta_orig = self.sparse_exact_beta_original(pirls_result.as_ref());
                if let Err(e) = barrier_cfg.add_barrier_hessian_diagonal(&mut s_lambda, &beta_orig)
                {
                    log::warn!("Sparse barrier Hessian diagonal skipped: {e}");
                }
            }
        }

        let mut workspace = PirlsWorkspace::new(self.y.len(), self.p, 0, 0);
        let sparse_system = assemble_and_factor_sparse_penalized_system(
            &mut workspace,
            x_sparse,
            &pirls_result.finalweights,
            &s_lambda,
            ridge_passport.delta,
        )?;
        let (logdet_s_pos, det1_values) =
            self.sparse_penalty_logdet_runtime(rho, penalty_blocks.as_ref());
        let firth_dense_operator_original = if self.config.firth_bias_reduction
            && matches!(self.config.link_function(), LinkFunction::Logit)
        {
            let firth_n = self.x().nrows();
            let firth_p = self.x().ncols();
            if !super::firth_problem_scale_allows(firth_n, firth_p) {
                log::info!(
                    "disabling Firth bias reduction for large model (n={}, p={}, n*p={}, n*p^2={}): \
                     exact Firth operator is small-model-only",
                    firth_n,
                    firth_p,
                    firth_n.saturating_mul(firth_p),
                    firth_n.saturating_mul(firth_p).saturating_mul(firth_p),
                );
                None
            } else {
                let x_dense = self
                    .x()
                    .try_to_dense_arc(
                        "sparse exact REML runtime requires dense design for Firth operator",
                    )
                    .map_err(EstimationError::InvalidInput)?;
                Some(Arc::new(Self::build_firth_dense_operator(
                    x_dense.as_ref(),
                    &pirls_result.final_eta,
                    self.weights,
                )?))
            }
        } else {
            None
        };

        Ok(EvalShared {
            key,
            pirls_result,
            ridge_passport,
            geometry: RemlGeometry::SparseExactSpd,
            h_eff: Arc::new(Array2::zeros((0, 0))),
            h_total: Arc::new(Array2::zeros((0, 0))),
            sparse_exact: Some(Arc::new({
                let factor = Arc::new(sparse_system.factor);
                // Compute Takahashi selected inverse from simplicial factorization.
                // This precomputes H^{-1} entries on the filled pattern of L, enabling
                // O(nnz) trace computations instead of O(p) column solves.
                let sfactor =
                    crate::linalg::sparse_exact::factorize_simplicial(&sparse_system.h_sparse)?;
                let takahashi = Some(Arc::new(
                    crate::linalg::sparse_exact::TakahashiInverse::compute(&sfactor)?,
                ));
                SparseExactEvalData {
                    factor,
                    takahashi,
                    logdet_h: sparse_system.logdet_h,
                    logdet_s_pos,
                    det1_values: Arc::new(det1_values),
                }
            })),
            firth_dense_operator: None,
            firth_dense_operator_original,
        })
    }

    /// Runs the inner P-IRLS loop, caching the result.
    pub(super) fn execute_pirls_if_needed(
        &self,
        rho: &Array1<f64>,
    ) -> Result<Arc<PirlsResult>, EstimationError> {
        let use_cache = self
            .cache_manager
            .pirls_cache_enabled
            .load(Ordering::Relaxed);
        // Use sanitized key to handle NaN and -0.0 vs 0.0 issues
        let key_opt = self.rhokey_sanitized(rho);
        if use_cache
            && let Some(key) = &key_opt
            && let Some(cached) = self.cache_manager.pirls_cache.write().unwrap().get(key)
        {
            // Do not overwrite the current warm start from cache hits.
            // Line search / multi-eval outer loops revisit older rho keys and
            // replacing a recent nearby beta with an older cached mode can
            // materially slow subsequent PIRLS convergence.
            if cached.cache_compacted {
                let mut pirls_config = self.config.as_pirls_config();
                pirls_config.link_kind =
                    if let Some(state) = self.runtime_mixture_link_state.clone() {
                        InverseLink::Mixture(state)
                    } else if let Some(state) = self.runtime_sas_link_state {
                        if matches!(self.config.link_function(), LinkFunction::BetaLogistic) {
                            InverseLink::BetaLogistic(state)
                        } else {
                            InverseLink::Sas(state)
                        }
                    } else {
                        InverseLink::Standard(self.config.link_function())
                    };
                return Ok(Arc::new(cached.rehydrate_after_reml_cache(
                    self.x(),
                    self.y,
                    self.weights,
                    self.offset.view(),
                    &pirls_config.link_kind,
                )?));
            }
            return Ok(cached);
        }

        // Run P-IRLS with original matrices to perform fresh reparameterization
        // The returned result will include the transformation matrix qs
        let pirls_result = {
            let warm_start_holder = self.warm_start_beta.read().unwrap();
            let warm_start_ref = if self.warm_start_enabled.load(Ordering::Relaxed) {
                warm_start_holder.as_ref()
            } else {
                None
            };
            let mut pirls_config = self.config.as_pirls_config();
            let screening_cap = self.screening_max_inner_iterations.load(Ordering::Relaxed);
            if screening_cap > 0 {
                pirls_config.max_iterations = pirls_config.max_iterations.min(screening_cap);
            }
            pirls_config.link_kind = if let Some(state) = self.runtime_mixture_link_state.clone() {
                InverseLink::Mixture(state)
            } else if let Some(state) = self.runtime_sas_link_state {
                if matches!(self.config.link_function(), LinkFunction::BetaLogistic) {
                    InverseLink::BetaLogistic(state)
                } else {
                    InverseLink::Sas(state)
                }
            } else {
                InverseLink::Standard(self.config.link_function())
            };
            let problem = pirls::PirlsProblem {
                x: &self.x,
                offset: self.offset.view(),
                y: self.y,
                priorweights: self.weights,
                covariate_se: None,
            };
            let penalty = pirls::PenaltyConfig {
                canonical_penalties: &self.canonical_penalties,
                balanced_penalty_root: Some(&self.balanced_penalty_root),
                reparam_invariant: Some(&self.reparam_invariant),
                p: self.p,
                coefficient_lower_bounds: self.coefficient_lower_bounds.as_ref(),
                linear_constraints_original: self.linear_constraints.as_ref(),
                penalty_shrinkage_floor: self.penalty_shrinkage_floor,
                kronecker_factored: self.kronecker_factored.as_ref(),
            };
            let pirls_start = std::time::Instant::now();
            let result = pirls::fit_model_for_fixed_rho(
                LogSmoothingParamsView::new(rho.view()),
                problem,
                penalty,
                &pirls_config,
                warm_start_ref,
            );
            let pirls_elapsed = pirls_start.elapsed();
            if let Ok((ref res, ref wm)) = result {
                log::info!(
                    "[PIRLS-timing] iters={} status={:?} max_eta={:.1} jeffreys_logdet={} elapsed={:.3}s",
                    wm.iterations,
                    res.status,
                    res.max_abs_eta,
                    res.jeffreys_logdet()
                        .map(|v| format!("{v:.3e}"))
                        .unwrap_or_else(|| "none".to_string()),
                    pirls_elapsed.as_secs_f64(),
                );
            }
            result
        };

        if let Err(e) = &pirls_result {
            log::warn!("[GAM COST]   -> P-IRLS INNER LOOP FAILED. Error: {e:?}");
            // Keep the previous successful warm start even when a trial point
            // fails. Outer line search commonly probes unstable candidates and
            // then returns to nearby feasible rho values where the prior warm
            // beta remains the best initializer.
        }

        let (pirls_result, _) = pirls_result?; // Propagate error if it occurred
        let pirls_result = Arc::new(pirls_result);
        self.enforce_constraint_kkt(pirls_result.as_ref())?;

        // Check the status returned by the P-IRLS routine.
        match pirls_result.status {
            pirls::PirlsStatus::Converged | pirls::PirlsStatus::StalledAtValidMinimum => {
                self.updatewarm_start_from(pirls_result.as_ref());
                // This is a successful fit. Cache only if key is valid (not NaN).
                if use_cache && let Some(key) = key_opt {
                    self.cache_manager
                        .pirls_cache
                        .write()
                        .unwrap()
                        .insert(key, Arc::new(pirls_result.compact_for_reml_cache()));
                }
                Ok(pirls_result)
            }
            pirls::PirlsStatus::Unstable => {
                // The fit was unstable. This is where we throw our specific, user-friendly error.
                // Pass the diagnostic info into the error
                Err(EstimationError::PerfectSeparationDetected {
                    iteration: pirls_result.iteration,
                    max_abs_eta: pirls_result.max_abs_eta,
                })
            }
            pirls::PirlsStatus::MaxIterationsReached => {
                log::error!(
                    "P-IRLS reached max iterations without certifying a valid minimum (gradient norm {:.3e}, iter {})",
                    pirls_result.lastgradient_norm,
                    pirls_result.iteration
                );
                Err(EstimationError::PirlsDidNotConverge {
                    max_iterations: pirls_result.iteration,
                    last_change: pirls_result.lastgradient_norm,
                })
            }
        }
    }
}

impl<'a> RemlState<'a> {
    /// Compute the scalar outer objective value used by the planner-selected
    /// outer optimizer.
    ///
    /// Full objective reference.
    /// This function returns the scalar outer cost minimized over ρ.
    ///
    /// Non-Gaussian branch (negative LAML form used by optimizer):
    ///   V_LAML(ρ) =
    ///      -ℓ(β̂(ρ))
    ///      + 0.5 β̂(ρ)ᵀ S(ρ) β̂(ρ)
    ///      + 0.5 log|H(ρ)|
    ///      - 0.5 log|S(ρ)|_+
    ///      + const
    ///
    /// where:
    ///   S(ρ) = Σ_k exp(ρ_k) S_k + δI
    ///   H(ρ) = -∇²ℓ(β̂(ρ)) + S(ρ)
    ///
    /// Gaussian identity-link REML branch:
    ///   V_REML(ρ, φ) =
    ///      D_p(ρ)/(2φ)
    ///      + (n_r/2) log φ
    ///      + 0.5 log|H(ρ)|
    ///      - 0.5 log|S(ρ)|_+
    ///      + const
    ///
    /// with profiled φ:
    ///   φ̂(ρ) = D_p(ρ)/n_r
    ///   V_REML,prof(ρ) =
    ///      (n_r/2) log D_p(ρ)
    ///      + 0.5 log|H(ρ)|
    ///      - 0.5 log|S(ρ)|_+
    ///      + const.
    ///
    /// Consistency rule enforced throughout:
    ///   The same stabilized matrices/factorizations are used for
    ///   objective and gradient/Hessian terms. Mixing different H/S variants
    ///   causes deterministic gradient mismatch and unstable outer optimization.
    ///
    /// Determinant conventions:
    ///   - log|H| may use positive-part/stabilized spectrum conventions when needed.
    ///   - log|S|_+ follows fixed-rank pseudo-determinant conventions in the
    ///     transformed penalty basis, optionally including ridge policy.
    /// These conventions are mirrored in gradient code via corresponding trace terms.
    pub fn compute_cost(&self, p: &Array1<f64>) -> Result<f64, EstimationError> {
        let cost_call_idx = {
            let mut calls = self.arena.cost_eval_count.write().unwrap();
            *calls += 1;
            *calls
        };
        let t_eval_start = std::time::Instant::now();
        {
            let prefix: Vec<String> = p.iter().take(4).map(|v| format!("{:.3}", v)).collect();
            log::info!(
                "[REML] eval#{} begin cost-only | rho[..4]=[{}] | k={}",
                cost_call_idx,
                prefix.join(","),
                p.len()
            );
        }
        let rho_key = EvalCacheManager::sanitized_rhokey(p);
        if let Some(eval) = self.cache_manager.cached_outer_eval(&rho_key) {
            log::info!(
                "[REML] eval#{} cache hit | cost {:.6e} | elapsed {:.1}ms",
                cost_call_idx,
                eval.cost,
                t_eval_start.elapsed().as_secs_f64() * 1000.0
            );
            return Ok(eval.cost);
        }
        let t_pirls = std::time::Instant::now();
        let bundle = match self.obtain_eval_bundle(p) {
            Ok(bundle) => bundle,
            Err(EstimationError::ModelIsIllConditioned { .. }) => {
                self.cache_manager.invalidate_eval_bundle();
                // Inner linear algebra says "too singular" — treat as barrier.
                log::warn!(
                    "P-IRLS flagged ill-conditioning for current rho; returning +inf cost to retreat."
                );
                // Diagnostics: which rho are at bounds
                let (at_lower, at_upper) = boundary_hit_indices(p.view(), RHO_BOUND, 1e-8);
                if !(at_lower.is_empty() && at_upper.is_empty()) {
                    log::debug!(
                        "[Diag] rho bounds: lower={:?} upper={:?}",
                        at_lower,
                        at_upper
                    );
                }
                return Ok(f64::INFINITY);
            }
            Err(EstimationError::PerfectSeparationDetected { .. })
            | Err(EstimationError::PirlsDidNotConverge { .. }) => {
                self.cache_manager.invalidate_eval_bundle();
                log::warn!(
                    "P-IRLS separation/non-convergence at current rho; returning +inf cost to retreat."
                );
                return Ok(f64::INFINITY);
            }
            Err(e) => {
                self.cache_manager.invalidate_eval_bundle();
                let (at_lower, at_upper) = boundary_hit_indices(p.view(), RHO_BOUND, 1e-8);
                if !(at_lower.is_empty() && at_upper.is_empty()) {
                    log::debug!(
                        "[Diag] rho bounds: lower={:?} upper={:?}",
                        at_lower,
                        at_upper
                    );
                }
                return Err(e);
            }
        };
        let pirls_ms = t_pirls.elapsed().as_secs_f64() * 1000.0;
        log::info!(
            "[REML] eval#{} pirls done | elapsed {:.1}ms | backend {:?}",
            cost_call_idx,
            pirls_ms,
            bundle.backend_kind()
        );
        if bundle.backend_kind() == GeometryBackendKind::SparseExactSpd {
            let t_assemble = std::time::Instant::now();
            let result =
                self.evaluate_unified_sparse(p, &bundle, super::unified::EvalMode::ValueOnly)?;
            log::info!(
                "[REML] eval#{} sparse cost {:.6e} | assemble {:.1}ms | total {:.1}ms",
                cost_call_idx,
                result.cost,
                t_assemble.elapsed().as_secs_f64() * 1000.0,
                t_eval_start.elapsed().as_secs_f64() * 1000.0
            );
            return Ok(result.cost);
        }
        {
            // Validation and diagnostics (before delegating to unified evaluator).
            let pirls_result = bundle.pirls_result.as_ref();
            let ridge_used = bundle.ridge_passport.delta;

            if !p.is_empty() {
                let k_lambda = p.len();
                let k_r = pirls_result.reparam_result.canonical_transformed.len();
                let k_d = pirls_result.reparam_result.det1.len();
                if !(k_lambda == k_r && k_r == k_d) {
                    return Err(EstimationError::LayoutError(format!(
                        "Penalty dimension mismatch: lambdas={}, R={}, det1={}",
                        k_lambda, k_r, k_d
                    )));
                }
                if self.nullspace_dims.len() != k_lambda {
                    return Err(EstimationError::LayoutError(format!(
                        "Nullspace dimension mismatch: expected {} entries, got {}",
                        k_lambda,
                        self.nullspace_dims.len()
                    )));
                }
            }

            const MIN_ACCEPTABLE_HESSIAN_EIGENVALUE: f64 = 1e-12;
            let want_hot_diag = self.should_compute_hot_diagnostics(cost_call_idx);
            if ridge_used > 0.0 && want_hot_diag {
                // Eigenvalue diagnostics require dense; only pay the cost when
                // hot diagnostics are requested and ridge was applied.
                let pht_dense = pirls_result.penalized_hessian_transformed.to_dense();
                if let Ok((eigs, _)) = pht_dense.eigh(Side::Lower) {
                    if let Some(min_eig) = eigs.iter().cloned().reduce(f64::min) {
                        if should_emit_h_min_eig_diag(min_eig) {
                            log::debug!(
                                "[Diag] H min_eig={:.3e} (ridge={:.3e})",
                                min_eig,
                                ridge_used
                            );
                        }
                        if min_eig <= 0.0 {
                            log::warn!(
                                "Penalized Hessian not PD (min eig <= 0) before stabilization; proceeding with ridge {:.3e}.",
                                ridge_used
                            );
                        }
                        if !min_eig.is_finite() || min_eig <= MIN_ACCEPTABLE_HESSIAN_EIGENVALUE {
                            let condition_number = symmetric_spectrum_condition_number(&pht_dense);
                            log::warn!(
                                "Penalized Hessian extremely ill-conditioned (cond={:.3e}); continuing with stabilized Hessian.",
                                condition_number
                            );
                        }
                    }
                }
            }
        }
        // Delegate to the unified evaluator for the actual formula computation.
        // This ensures cost and gradient share the exact same formula.
        let t_assemble = std::time::Instant::now();
        let result = self.evaluate_unified(p, &bundle, super::unified::EvalMode::ValueOnly)?;
        log::info!(
            "[REML] eval#{} dense cost {:.6e} | assemble {:.1}ms | total {:.1}ms",
            cost_call_idx,
            result.cost,
            t_assemble.elapsed().as_secs_f64() * 1000.0,
            t_eval_start.elapsed().as_secs_f64() * 1000.0
        );
        Ok(result.cost)
    }

    ///
    /// Exact non-Laplace evidence identities (reference comments; not runtime path)
    /// We optimize a Laplace-style outer objective for scalability, but the exact
    /// marginal likelihood for non-Gaussian models can be written analytically as:
    ///
    ///   L(ρ) = ∫ exp(l(β) - 0.5 βᵀ S(ρ) β) dβ,   S(ρ)=Σ_k exp(ρ_k) S_k.
    ///
    /// Universal exact gradient identity (when differentiation under the integral
    /// is justified and L(ρ) < ∞):
    ///
    ///   ∂_{ρ_k} log L(ρ)
    ///   = -0.5 * exp(ρ_k) * E_{π(β|y,ρ)}[ βᵀ S_k β ].
    ///
    /// Laplace bridge to implemented terms:
    /// - If π(β|y,ρ) is approximated locally by N(β̂, H^{-1}), then
    ///     E[βᵀ S_k β] ≈ β̂ᵀ S_k β̂ + tr(H^{-1} S_k),
    ///   giving the familiar quadratic + trace structure.
    /// - In this code those appear as:
    ///     0.5 * β̂ᵀ S_k^ρ β̂,
    ///     -0.5 * tr(S^+ S_k^ρ),
    ///     +0.5 * tr(H^{-1} H_k).
    ///
    /// Why this does NOT collapse to only tr(H^{-1}S_k):
    /// - The exact identity differentiates the true integral measure.
    /// - LAML differentiates a moving approximation:
    ///     V_LAML(ρ) = -ℓ(β̂(ρ)) + 0.5 β̂(ρ)ᵀ S(ρ) β̂(ρ)
    ///                 + 0.5 log|H(ρ)| - 0.5 log|S(ρ)|_+.
    /// - Here both center β̂(ρ) and curvature H(ρ) move with ρ.
    /// - For non-Gaussian families, H_k includes the third-derivative tensor path
    ///   through β̂(ρ), i.e. H_k != S_k^ρ. These are the explicit dH/dρ_k terms
    ///   retained below to differentiate the Laplace objective exactly.
    ///
    /// For Bernoulli-logit, an exact Pólya-Gamma augmentation gives:
    ///
    ///   L(ρ) = 2^{-n} (2π)^{p/2}
    ///          E_{ω_i ~ PG(1,0)} [ |Q(ω,ρ)|^{-1/2} exp(0.5 bᵀ Q^{-1} b) ],
    ///   Q(ω,ρ)=S(ρ)+XᵀΩX, b=Xᵀ(y-1/2).
    ///
    /// and
    ///
    ///   ∂_{ρ_k} log L
    ///   = -0.5 * exp(ρ_k) *
    ///     E_{ω|y,ρ}[ tr(S_k Q^{-1}) + μᵀ S_k μ ],  μ=Q^{-1}b.
    /// Equivalently, since β|ω,y,ρ ~ N(μ,Q^{-1}):
    ///   E[βᵀS_kβ | ω,y,ρ] = tr(S_k Q^{-1}) + μᵀS_kμ.
    ///
    /// yielding exact (but high-dimensional) contour integrals / series after
    /// analytically integrating β.
    ///
    /// Practical note:
    /// - These are exact equalities but generally not polynomial-time tractable
    ///   for arbitrary dense (X, n, p).
    /// - This code therefore uses deterministic Laplace/implicit-differentiation
    ///   machinery for the main optimizer path, with exact tensor terms where
    ///   feasible (H_k, H_{kℓ}, c/d arrays), and scalable trace backends.
    ///
    /// Full outer-derivative reference (exact system, sign convention used here).
    /// This optimizer minimizes an outer cost V(ρ).
    ///
    /// Common definitions:
    ///   λ_k = exp(ρ_k)
    ///   S(ρ) = Σ_k λ_k S_k + δI
    ///   A_k = ∂S/∂ρ_k = λ_k S_k
    ///   A_{kℓ} = ∂²S/(∂ρ_k∂ρ_ℓ) = δ_{kℓ} A_k
    ///
    /// Inner mode (β̂):
    ///   ∇_β ℓ(β̂) - S(ρ) β̂ = 0
    ///
    /// Curvature:
    ///   H(ρ) = -∇²_β ℓ(β̂(ρ)) + S(ρ)
    ///
    ///   w_i = -∂²ℓ_i/∂η_i²
    ///   d_i = -∂³ℓ_i/∂η_i³
    ///   e_i = -∂⁴ℓ_i/∂η_i⁴
    ///
    /// Then:
    ///   H_k = A_k + Xᵀ diag(d ⊙ u_k) X,     u_k := X B_k
    ///   H_{kℓ} = δ_{kℓ}A_k + Xᵀ diag(e ⊙ u_k ⊙ u_ℓ + d ⊙ u_{kℓ}) X
    ///
    /// with implicit derivatives:
    ///   B_k := ∂β̂/∂ρ_k = -H^{-1}(A_k β̂)
    ///   H B_{kℓ} = -(H_ℓ B_k + δ_{kℓ}A_k β̂ + A_k B_ℓ)
    ///
    /// Non-Gaussian negative LAML cost:
    ///   V(ρ) = -ℓ(β̂) + 0.5 β̂ᵀSβ̂ + 0.5 log|H| - 0.5 log|S|_+
    ///
    /// Exact gradient:
    ///   g_k = 0.5 β̂ᵀA_kβ̂ + 0.5 tr(H^{-1}H_k) - 0.5 ∂_k log|S|_+
    ///
    /// Exact Hessian decomposition:
    ///   ∂²V/(∂ρ_k∂ρ_ℓ) = Q_{kℓ} + L_{kℓ} + P_{kℓ}
    ///
    ///   Q_{kℓ} = 0.5 δ_{kℓ} β̂ᵀA_kβ̂ - B_ℓᵀ H B_k
    ///
    ///   L_{kℓ} = 0.5 [ tr(H^{-1}H_{kℓ}) - tr(H^{-1}H_k H^{-1}H_ℓ) ]
    ///
    ///   P_{kℓ} = -0.5 ∂²_{kℓ} log|S|_+
    ///
    /// Here, this function computes the exact gradient terms (including dH/dρ_k via d_i).
    /// The full exact Hessian is not assembled in this loop because it requires B_{kℓ}
    /// solves and fourth-derivative terms for every (k,ℓ) pair.
    ///
    /// Gaussian REML note:
    ///   In identity-link Gaussian, d=e=0 so H_k=A_k and H_{kℓ}=δ_{kℓ}A_k.
    ///   With profiled φ, use either:
    ///   - explicit profiled objective derivatives, or
    ///   - Schur complement in (ρ, log φ):
    ///       H_prof = H_{ρρ} - H_{ρα} H_{αα}^{-1} H_{αρ}.
    ///
    /// Pseudo-determinant note:
    ///   The code uses fixed-rank/stabilized conventions for log|S|_+ to keep objective
    ///   derivatives smooth and consistent with the transformed penalty basis used by PIRLS.
    ///
    /// This is the core gradient evaluator for outer optimization. The
    /// centralized planner may feed it to ARC, BFGS, or an EFS variant,
    /// depending on available curvature information.
    /// The calculation differs significantly between the Gaussian (REML) and non-Gaussian (LAML) cases.
    ///
    /// # Mathematical Basis (Gaussian/REML Case)
    ///
    /// For Gaussian models (Identity link), we minimize the negative REML log-likelihood, which serves as our cost function.
    /// From Wood (2011, JRSSB, Eq. 4), the cost function to minimize is:
    ///
    ///   Cost(ρ) = -l_r(ρ) = D_p / (2φ) + (1/2)log|XᵀWX + S(ρ)| - (1/2)log|S(ρ)|_+
    ///
    /// where D_p is the penalized deviance, H = XᵀWX + S(ρ) is the penalized Hessian, S(ρ) is the total
    /// penalty matrix, and |S(ρ)|_+ is the pseudo-determinant.
    ///
    /// The gradient ∇Cost(ρ) is computed term-by-term. A key simplification for the Gaussian case is the
    /// **envelope theorem**: at the P-IRLS optimum for β̂, the derivative of the cost function with respect to β̂ is zero.
    /// This means we only need the *partial* derivatives with respect to ρ, and the complex indirect derivatives
    /// involving ∂β̂/∂ρ can be ignored.
    ///
    /// # Mathematical Basis (Non-Gaussian/LAML Case)
    ///
    /// For non-Gaussian models, the envelope theorem does not apply because the weight matrix W depends on β̂.
    /// The gradient requires calculating the full derivative, including the indirect term (∂V/∂β̂)ᵀ(∂β̂/∂ρ).
    /// This leads to a different final formula involving derivatives of the weight matrix, as detailed in
    /// Wood (2011, Appendix D).
    ///
    /// This method handles two distinct statistical criteria for marginal likelihood optimization:
    ///
    /// - For Gaussian models (Identity link), this calculates the exact REML gradient
    ///   (Restricted Maximum Likelihood).
    /// - For non-Gaussian GLMs, this calculates the LAML gradient (Laplace Approximate
    ///   Marginal Likelihood) as derived in Wood (2011, Appendix C & D).
    ///
    /// # Mathematical Theory
    ///
    /// The gradient calculation requires careful application of the chain rule and envelope theorem
    /// due to the nested optimization structure of GAMs:
    ///
    /// - The inner loop (P-IRLS) finds coefficients β̂ that maximize the penalized log-likelihood
    ///   for a fixed set of smoothing parameters ρ.
    /// - The outer loop finds smoothing parameters ρ that maximize the
    ///   marginal likelihood using the centralized outer-strategy planner.
    ///
    /// Since β̂ is an implicit function of ρ, the total derivative is:
    ///
    ///    dV_R/dρ_k = (∂V_R/∂β̂)ᵀ(∂β̂/∂ρ_k) + ∂V_R/∂ρ_k
    ///
    /// By the envelope theorem, (∂V_R/∂β̂) = 0 at the optimum β̂, so the first term vanishes.
    ///
    /// # Key Distinction Between REML and LAML Gradients
    ///
    /// - Gaussian (REML): by the envelope theorem the indirect β̂ terms vanish. The deviance
    ///   contribution reduces to the penalty-only derivative, yielding the familiar
    ///   (β̂ᵀS_kβ̂)/σ² piece in the gradient.
    /// - Non-Gaussian (LAML): there is no cancellation of the penalty derivative within the
    ///   deviance component. The derivative of the penalized deviance contains both
    ///   d(D)/dρ_k and d(βᵀSβ)/dρ_k. Our implementation follows mgcv’s gdi1: we add the penalty
    ///   derivative to the deviance derivative before applying the 1/2 factor.
    // Stage: Start with the chain rule for any λₖ,
    //     dV/dλₖ = ∂V/∂λₖ  (holding β̂ fixed)  +  (∂V/∂β̂)ᵀ · (∂β̂/∂λₖ).
    //     The first summand is called the direct part, the second the indirect part.
    //
    // Stage: Note the two outer criteria—Gaussian likelihood maximizes REML, while non-Gaussian likelihood
    //     maximizes a Laplace approximation to the marginal likelihood (LAML). These objectives respond differently to β̂.
    //
    //     2.1  Gaussian case, REML.
    //          The REML construction integrates the fixed effects out of the likelihood.  At the optimum
    //          the partial derivative ∂V/∂β̂ is exactly zero.  The indirect part therefore vanishes.
    //          What remains is the direct derivative of the penalty and determinant terms.  The penalty
    //          contribution is found by differentiating −½ β̂ᵀ S_λ β̂ / σ² with respect to λₖ; this yields
    //          −½ β̂ᵀ Sₖ β̂ / σ².  No opposing term exists, so the quantity stays in the REML gradient.
    //          The code path selected by LinkFunction::Identity therefore computes
    //          beta_term = β̂ᵀ Sₖ β̂ and places it inside
    //          gradient[k] = 0.5 * λₖ * (beta_term / σ² − trace_term).
    //
    //     2.2  Non-Gaussian case, LAML.
    //          The Laplace objective contains −½ log |H_p| with H_p = Xᵀ W(β̂) X + S_λ.  Because W
    //          depends on β̂, the total derivative includes dW/dλₖ via β̂.  Differentiating the
    //          optimality condition for β̂ gives
    //          ∂β̂/∂λₖ = −λₖ H_p⁻¹ Sₖ β̂.  The penalized log-likelihood L(β̂, λ) still obeys the
    //          envelope theorem, so dL/dλₖ = −½ β̂ᵀ Sₖ β̂ (no implicit term).
    //          The resulting cost gradient combines four pieces:
    //            +½ λₖ β̂ᵀ Sₖ β̂
    //            +½ λₖ tr(H_p⁻¹ Sₖ)
    //            +½ tr(H_p⁻¹ Xᵀ ∂W/∂λₖ X)
    //            −½ λₖ tr(S_λ⁺ Sₖ)
    //
    // Stage: Remember that the sign of ∂β̂/∂λₖ matters; from the implicit-function theorem the linear solve reads
    //     −H_p (∂β̂/∂λₖ) = λₖ Sₖ β̂, giving the minus sign used above.  With that sign the indirect and
    //     direct quadratic pieces are exact negatives, which is what the algebra requires.
    /// Build the exact Jeffreys operator in the same coefficient basis used by
    /// the outer Hessian operator and derivative provider.
    ///
    /// If the outer solve is projected to a free subspace `β = Z β_free`, the
    /// Jeffreys term must also be evaluated on `X_eff = X Z`. Reusing the
    /// cached full-basis operator in that situation would recreate the original
    /// value/derivative mismatch this code is meant to eliminate.
    fn build_dense_firth_operator_for_outer_basis(
        &self,
        pirls_result: &PirlsResult,
        bundle: &EvalShared,
        free_basis_opt: &Option<Array2<f64>>,
    ) -> Result<Option<std::sync::Arc<super::FirthDenseOperator>>, EstimationError> {
        if !(self.config.firth_bias_reduction
            && matches!(self.config.link_function(), LinkFunction::Logit))
        {
            return Ok(None);
        }

        if let Some(z) = free_basis_opt.as_ref() {
            let x_projected = pirls_result.x_transformed.to_dense().dot(z);
            if !super::firth_problem_scale_allows(x_projected.nrows(), x_projected.ncols()) {
                log::info!(
                    "disabling Firth bias reduction for projected outer basis (n={}, p={}, n*p={}, n*p^2={}): \
                     exact Firth operator is small-model-only",
                    x_projected.nrows(),
                    x_projected.ncols(),
                    x_projected.nrows().saturating_mul(x_projected.ncols()),
                    x_projected
                        .nrows()
                        .saturating_mul(x_projected.ncols())
                        .saturating_mul(x_projected.ncols()),
                );
                return Ok(None);
            }
            return Ok(Some(std::sync::Arc::new(Self::build_firth_dense_operator(
                &x_projected,
                &pirls_result.final_eta,
                self.weights,
            )?)));
        }

        if let Some(cached) = bundle.firth_dense_operator.clone() {
            return Ok(Some(cached));
        }

        let x_dense = pirls_result.x_transformed.to_dense();
        Ok(Some(std::sync::Arc::new(Self::build_firth_dense_operator(
            &x_dense,
            &pirls_result.final_eta,
            self.weights,
        )?)))
    }

    /// Build the derivative provider, dispersion handling, zeroed TK placeholders,
    /// exact Firth operator, log-likelihood, and barrier config
    /// that are common to all dense evaluation paths.
    ///
    /// # Arguments
    /// - `pirls_result`: the inner-loop solution
    /// - `bundle`: the evaluation bundle (carries h_eff, firth operator, etc.)
    /// - `free_basis_opt`: optional constraint-projection basis
    /// - `include_firth_derivs`: whether to wrap the derivative provider with
    ///   Firth-aware derivatives
    ///
    /// The actual Tierney-Kadane correction is added outside the unified
    /// evaluator so value, gradient, and Hessian can all come from the same
    /// invariant scalar correction.
    fn build_dense_derivative_context(
        &self,
        pirls_result: &PirlsResult,
        bundle: &EvalShared,
        free_basis_opt: &Option<Array2<f64>>,
        include_firth_derivs: bool,
    ) -> Result<DerivativeContext, EstimationError> {
        use super::unified::{
            DispersionHandling, GaussianDerivatives, SinglePredictorGlmDerivatives,
        };

        let is_gaussian_identity = self.config.link_function() == LinkFunction::Identity;
        let firth_op =
            self.build_dense_firth_operator_for_outer_basis(pirls_result, bundle, free_basis_opt)?;

        // Derivative provider.
        let firth_active_for_derivs = include_firth_derivs && firth_op.is_some();
        let deriv_provider: Box<dyn super::unified::HessianDerivativeProvider> =
            if is_gaussian_identity {
                Box::new(GaussianDerivatives)
            } else {
                // Differentiate the same Hessian-side curvature arrays that the
                // inner PIRLS solve accepted at the mode.
                let (c_array, d_array) = self.hessian_cd_arrays(pirls_result)?;
                let x_transformed = if let Some(z) = free_basis_opt.as_ref() {
                    // Project the design: X_proj = X Z
                    let x_dense = pirls_result.x_transformed.to_dense();
                    crate::linalg::matrix::DesignMatrix::Dense(
                        crate::matrix::DenseDesignMatrix::from(x_dense.dot(z)),
                    )
                } else {
                    pirls_result.x_transformed.clone()
                };
                let base = SinglePredictorGlmDerivatives {
                    c_array,
                    d_array: Some(d_array),
                    x_transformed,
                };
                if firth_active_for_derivs {
                    if let Some(firth_op) = firth_op.clone() {
                        Box::new(super::unified::FirthAwareGlmDerivatives { base, firth_op })
                    } else {
                        Box::new(base)
                    }
                } else {
                    Box::new(base)
                }
            };

        // Dispersion handling.
        let dispersion = if is_gaussian_identity {
            DispersionHandling::ProfiledGaussian
        } else {
            DispersionHandling::Fixed {
                phi: pirls_result.likelihood.fixed_phi().unwrap_or(1.0),
                include_logdet_h: true,
                include_logdet_s: true,
            }
        };

        let log_likelihood = crate::pirls::calculate_loglikelihood_omitting_constants(
            self.y,
            &pirls_result.finalmu,
            pirls_result.likelihood,
            self.weights,
        );

        // Construct barrier config for monotonicity constraints when no
        // active-set projection is in effect (barrier indices are in the
        // full transformed-coefficient space).
        let barrier_config = if free_basis_opt.is_none() {
            pirls_result
                .linear_constraints_transformed
                .as_ref()
                .and_then(Self::barrier_config_from_constraints)
        } else {
            None
        };

        Ok(DerivativeContext {
            deriv_provider,
            dispersion,
            log_likelihood,
            firth_op,
            barrier_config,
        })
    }

    /// Build the dispersion handling, derivative provider, log-likelihood,
    /// exact Firth operator, and barrier config that are common
    /// to both sparse evaluation paths (`evaluate_unified_sparse` and
    /// `evaluate_efs` when sparse-dispatched).
    ///
    /// TK correction is NOT included here because the sparse paths compute
    /// it via sparse Cholesky solves with completely different logic.
    fn build_sparse_derivative_context(
        &self,
        pirls_result: &PirlsResult,
        bundle: &EvalShared,
    ) -> Result<DerivativeContext, EstimationError> {
        use super::unified::{
            DispersionHandling, FirthAwareGlmDerivatives, GaussianDerivatives,
            SinglePredictorGlmDerivatives,
        };

        let is_gaussian_identity = self.config.link_function() == LinkFunction::Identity;

        // Sparse exact still uses the same dense Jeffreys operator; only the
        // H^{-1} applications move to the sparse Cholesky operator.
        let firth_op = if self.config.firth_bias_reduction
            && matches!(self.config.link_function(), LinkFunction::Logit)
        {
            if let Some(cached) = bundle.firth_dense_operator_original.clone() {
                Some(cached)
            } else {
                let x_dense = self
                    .x()
                    .try_to_dense_arc(
                        "sparse exact REML runtime requires dense design for Firth operator",
                    )
                    .map_err(EstimationError::InvalidInput)?;
                Some(std::sync::Arc::new(Self::build_firth_dense_operator(
                    x_dense.as_ref(),
                    &pirls_result.final_eta,
                    self.weights,
                )?))
            }
        } else {
            None
        };

        // Dispersion and derivative provider depend on family.
        let (dispersion, deriv_provider): (_, Box<dyn super::unified::HessianDerivativeProvider>) =
            if is_gaussian_identity {
                (
                    DispersionHandling::ProfiledGaussian,
                    Box::new(GaussianDerivatives),
                )
            } else {
                // Differentiate the same Hessian-side curvature arrays that the
                // inner PIRLS solve accepted at the mode.
                let (c_array, d_array) = self.hessian_cd_arrays(pirls_result)?;
                (
                    DispersionHandling::Fixed {
                        phi: pirls_result.likelihood.fixed_phi().unwrap_or(1.0),
                        include_logdet_h: true,
                        include_logdet_s: true,
                    },
                    {
                        let base = SinglePredictorGlmDerivatives {
                            c_array,
                            d_array: Some(d_array),
                            x_transformed: self.x().clone(),
                        };
                        // Match the dense exact path: when Firth-logit is
                        // active, the directional log|H_total| drift includes
                        // -D(H_phi)[B_k]. The sparse backend differs only in how
                        // B_k = H^{-1} rhs is solved.
                        if let Some(firth_op) = firth_op.clone() {
                            Box::new(FirthAwareGlmDerivatives { base, firth_op })
                        } else {
                            Box::new(base)
                        }
                    },
                )
            };

        let log_likelihood = crate::pirls::calculate_loglikelihood_omitting_constants(
            self.y,
            &pirls_result.finalmu,
            pirls_result.likelihood,
            self.weights,
        );

        // Construct barrier config for monotonicity constraints.
        // The sparse path operates in the original (non-reparameterized)
        // coordinate system, so use the original linear constraints.
        let barrier_config = self
            .linear_constraints
            .as_ref()
            .and_then(Self::barrier_config_from_constraints);

        Ok(DerivativeContext {
            deriv_provider,
            dispersion,
            log_likelihood,
            firth_op,
            barrier_config,
        })
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  Unified inner-solution assembly
    // ═══════════════════════════════════════════════════════════════════════════

    /// Build penalty coordinates from canonical penalties, with Kronecker
    /// fast-path when available and active.
    fn build_penalty_coords(&self) -> Vec<super::unified::PenaltyCoordinate> {
        if let Some(ref kron) = self.kronecker_penalty_system {
            if self.kronecker_factored.is_some() {
                let d = kron.ndim();
                let total_dim = kron.p_total();
                let eigenvalues: Vec<ndarray::Array1<f64>> = kron
                    .marginal_eigensystems
                    .iter()
                    .map(|(evals, _)| evals.clone())
                    .collect();
                let mut coords = Vec::with_capacity(kron.num_penalties());
                for k in 0..d {
                    coords.push(super::unified::PenaltyCoordinate::KroneckerMarginal {
                        eigenvalues: eigenvalues.clone(),
                        dim_index: k,
                        marginal_dims: kron.marginal_dims.clone(),
                        total_dim,
                    });
                }
                if kron.has_double_penalty {
                    let identity_root = ndarray::Array2::<f64>::eye(total_dim);
                    coords.push(super::unified::PenaltyCoordinate::from_dense_root(
                        identity_root,
                    ));
                }
                return coords;
            }
        }
        self.canonical_penalties
            .iter()
            .map(|cp| cp.to_penalty_coordinate())
            .collect()
    }

    /// Pack a `DerivativeContext` plus backend-specific pieces into an
    /// `InnerAssembly`. All three assembly builders (`build_dense_assembly`,
    /// `build_sparse_assembly`, `build_dense_original_assembly`) delegate
    /// here to avoid repeating the 18-field struct literal.
    fn finish_assembly(
        &self,
        pirls_result: &PirlsResult,
        ctx: DerivativeContext,
        hessian_op: std::sync::Arc<dyn super::unified::HessianOperator>,
        beta: Array1<f64>,
        penalty_logdet: super::unified::PenaltyLogdetDerivs,
        nullspace_dim: f64,
        hessian_logdet_correction: f64,
        penalty_subspace_trace: Option<std::sync::Arc<super::unified::PenaltySubspaceTrace>>,
    ) -> super::assembly::InnerAssembly<'static> {
        super::assembly::InnerAssembly {
            log_likelihood: ctx.log_likelihood,
            penalty_quadratic: pirls_result.stable_penalty_term,
            beta,
            n_observations: self.y.len(),
            hessian_op,
            penalty_coords: self.build_penalty_coords(),
            penalty_logdet,
            dispersion: ctx.dispersion,
            rho_curvature_scale: 1.0,
            hessian_logdet_correction,
            penalty_subspace_trace,
            deriv_provider: Some(ctx.deriv_provider),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: ctx.firth_op,
            nullspace_dim: Some(nullspace_dim),
            barrier_config: ctx.barrier_config,
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
        }
    }

    /// Build an `InnerAssembly` using the dense-transformed backend.
    ///
    /// Shared ingredient builder for all dense-spectral evaluation paths
    /// (normal, ext-coord, and EFS). Callers may set `ext_coords` and pair
    /// functions on the returned assembly before passing it to
    /// `assemble_and_evaluate` or `assemble_and_evaluate_efs`.
    fn build_dense_assembly(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        mode: super::unified::EvalMode,
    ) -> Result<super::assembly::InnerAssembly<'static>, EstimationError> {
        use super::unified::{DenseSpectralOperator, PseudoLogdetMode};

        let pirls_result = bundle.pirls_result.as_ref();
        let ridge_passport = pirls_result.ridge_passport;

        let free_basis_opt = self.active_constraint_free_basis(pirls_result);
        let (h_for_operator, e_for_logdet) = if let Some(z) = free_basis_opt.as_ref() {
            (
                Self::projectwith_basis(bundle.h_total.as_ref(), z),
                pirls_result.reparam_result.e_transformed.dot(z),
            )
        } else {
            (
                bundle.h_total.as_ref().clone(),
                pirls_result.reparam_result.e_transformed.clone(),
            )
        };

        // When Firth bias reduction is active, the design column space may be
        // rank-deficient (e.g. aliased covariates).  The PIRLS penalized Hessian
        // H_total = X'WX + S − H_φ then inherits that rank deficiency on any
        // direction that is simultaneously null for X'WX, S, AND H_φ.  Under
        // `Smooth`, the small eigenvalues of H_total get a soft ε-floor that
        // leaks a spurious non-zero gradient contribution through the null
        // direction; both `log|H|` and its derivatives then count a direction
        // that carries no actual curvature.  Switching to `HardPseudo` masks
        // those `σ_j ≤ ε` eigenpairs consistently across `logdet`,
        // `trace_logdet_*`, and `H⁻¹` solves, so log|H|, its ρ-gradient, and
        // its ρ-Hessian all see the same reduced active subspace.
        //
        // Rationale: the Firth/Jeffreys penalty, the penalized score equation,
        // and therefore dβ̂/dρ all live on the identifiable subspace (matching
        // `firth_op.q_basis`); extending the logdet kernel to the null space
        // is both unphysical and FD-inconsistent.
        let hessian_mode = if bundle.firth_dense_operator.is_some() {
            PseudoLogdetMode::HardPseudo
        } else {
            PseudoLogdetMode::Smooth
        };

        let hessian_op = std::sync::Arc::new(
            DenseSpectralOperator::from_symmetric_with_mode(&h_for_operator, hessian_mode)
                .map_err(|e| {
                    EstimationError::InvalidInput(format!(
                        "DenseSpectralOperator from PIRLS Hessian: {e}"
                    ))
                })?,
        );

        let (penalty_rank, penalty_logdet) =
            self.dense_penalty_logdet_derivs(rho, &e_for_logdet, &[], ridge_passport, mode)?;

        let beta = if let Some(z) = free_basis_opt.as_ref() {
            z.t()
                .dot(&pirls_result.beta_transformed.as_ref().to_owned())
        } else {
            pirls_result.beta_transformed.as_ref().clone()
        };

        let nullspace_dim = h_for_operator.ncols().saturating_sub(penalty_rank) as f64;

        // Rank-deficient LAML fix: under `Smooth` the spectral operator's
        // cached logdet sums `ln r_ε(σ_j(H))` over ALL eigenvalues, so
        // genuinely null directions contribute `(p − rank) · ln ε` — very
        // negative — and make `½(log|H| − log|S|_+)` diverge to −∞
        // whenever `rank(X'WX) + rank(S) < p`.  Project H onto
        // `range(S_+)` and pass the scalar correction through
        // `hessian_logdet_correction`; `reml_laml_evaluate` already sums
        // `hop.logdet() + correction`, so the evaluator sees
        // `log|H_proj|_+` paired with `log|S|_+` on the same rank-r
        // identified subspace.
        //
        // The gradient trace kernel must pair with the **projected** logdet,
        // not the full-space `G_ε(H)`: for non-Gaussian families the IFT
        // correction `D_β H[v] = X' diag(c ⊙ X v) X` does NOT vanish on
        // `null(S)` (the intercept column of X is all-ones, typically in
        // `null(S)`), so `tr(G_ε · Ḣ)` picks up a spurious null-direction
        // contribution that is absent from `d log|U_Sᵀ H U_S|/dτ`.  Carry
        // the `PenaltySubspaceTrace { U_S, (U_Sᵀ H U_S)⁻¹ }` alongside the
        // scalar correction so `reml_laml_evaluate` can trace through the
        // identified subspace via `tr(U_S · (U_Sᵀ H U_S)⁻¹ · U_Sᵀ · Ḣ)`.
        // For Gaussian identity `c = 0`, so `D_β H = 0` and the leakage
        // does not arise — setting this kernel for Gaussian is still
        // correct (the traces agree) but strictly unnecessary.
        //
        // Under `HardPseudo` the operator already masks null eigenpairs
        // consistently in `logdet`, `trace_logdet_*`, and `solve`, so the
        // correction is zero there and no projected-trace kernel is needed.
        let (hessian_logdet_correction, penalty_subspace_trace) =
            if matches!(hessian_mode, PseudoLogdetMode::Smooth)
                && penalty_rank < h_for_operator.ncols()
            {
                use super::unified::HessianOperator;
                let (log_det_h_proj, kernel) = self.fixed_subspace_hessian_projected_parts(
                    &h_for_operator,
                    &e_for_logdet,
                    ridge_passport,
                )?;
                (
                    log_det_h_proj - hessian_op.logdet(),
                    kernel.map(std::sync::Arc::new),
                )
            } else {
                (0.0, None)
            };

        let ctx =
            self.build_dense_derivative_context(pirls_result, bundle, &free_basis_opt, true)?;
        Ok(self.finish_assembly(
            pirls_result,
            ctx,
            hessian_op,
            beta,
            penalty_logdet,
            nullspace_dim,
            hessian_logdet_correction,
            penalty_subspace_trace,
        ))
    }

    /// Build an `InnerAssembly` using the sparse-exact backend.
    ///
    /// Shared ingredient builder for all sparse Cholesky evaluation paths.
    fn build_sparse_assembly(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        mode: super::unified::EvalMode,
    ) -> Result<super::assembly::InnerAssembly<'static>, EstimationError> {
        use super::unified::{HessianOperator, PenaltyLogdetDerivs, SparseCholeskyOperator};

        let sparse = bundle.sparse_exact.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput("missing sparse exact evaluation payload".to_string())
        })?;
        let pirls_result = bundle.pirls_result.as_ref();

        let beta = self.sparse_exact_beta_original(pirls_result);
        let p_dim = beta.len();
        let hessian_op: std::sync::Arc<dyn HessianOperator> = {
            let mut op = SparseCholeskyOperator::new(sparse.factor.clone(), sparse.logdet_h, p_dim);
            if let Some(ref taka) = sparse.takahashi {
                op = op.with_takahashi(taka.clone());
            }
            std::sync::Arc::new(op)
        };

        log::trace!(
            "SparseCholeskyOperator: dim={}, active_rank={}",
            hessian_op.dim(),
            hessian_op.active_rank()
        );

        let mp = self.nullspace_dims.iter().copied().sum::<usize>() as f64;
        let det2 = if mode == super::unified::EvalMode::ValueGradientHessian {
            let lambdas = rho.mapv(f64::exp);
            let (_, det2) = self.structural_penalty_logdet_derivatives_block_local(
                &lambdas,
                bundle.ridge_passport.penalty_logdet_ridge(),
            )?;
            Some(det2)
        } else {
            None
        };
        let penalty_logdet = PenaltyLogdetDerivs {
            value: sparse.logdet_s_pos,
            first: sparse.det1_values.as_ref().clone(),
            second: det2,
        };

        let ctx = self.build_sparse_derivative_context(pirls_result, bundle)?;
        // Sparse-exact `log|H|` is Cholesky-derived from `X'WX + S_λ + δI`,
        // not a `PseudoLogdetMode::Smooth` eigenvalue sum, so the
        // `(p − rank) · ln ε` floor leakage that motivated e33c06be on
        // dense cannot occur here: Cholesky either succeeds (H is SPD,
        // logdet sums true log-diagonals) or fails (ridge escalates in
        // powers of 10 via `ensure_sparse_positive_definitewithridge`).
        //
        // The analogous rank-deficiency concern — that a nonzero ridge
        // δ would contribute `(p − rank(X'WX + S)) · log(δ)` to
        // `logdet_h` while `sparse_penalty_logdet_runtime` excludes that
        // subspace from `log|S|_+` — is structurally out of reach on the
        // backends that select sparse-exact:
        //
        //   1. `estimate_sparse_native_decision` gates on H density, which
        //      implies large p and typically n ≫ p (biobank-scale) — i.e.
        //      `X'WX` is rank-full, so `X'WX + S` is rank-full, so
        //      `ridge_used = 0` and no leak term exists.
        //   2. The small-n/high-dim regime that motivated e33c06be (n=120,
        //      k=6 Duchon) is dense territory; it never chooses
        //      `SparseNative` because the sparse density heuristic rejects
        //      near-dense systems.
        //
        // No projection correction is emitted here; pass 0.0 through to
        // `finish_assembly`.  If a future test exercise ever finds a
        // sparse-native fit with `ridge_used > 0` AND genuine null
        // directions of `X'WX + S`, extend `sparse_penalty_logdet_runtime`
        // to include the ridge-matched null contributions (keeping both
        // sides of the LAML ratio on the same p-dim space) rather than
        // reintroducing the range(S_+) projection; Cholesky can't cheaply
        // compute `U_S^T H U_S` without densifying H.
        Ok(self.finish_assembly(
            pirls_result,
            ctx,
            hessian_op,
            beta,
            penalty_logdet,
            mp,
            0.0,
            None,
        ))
    }

    /// Build an `InnerAssembly` using the dense original-basis backend.
    ///
    /// Used when QS-transformed coordinates have no active constraints,
    /// allowing us to work in the original basis for better numerical
    /// stability with extended ψ coordinates. TK correction is zeroed
    /// in the assembly and applied post-hoc by `assemble_and_evaluate`.
    fn build_dense_original_assembly(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        mode: super::unified::EvalMode,
    ) -> Result<super::assembly::InnerAssembly<'static>, EstimationError> {
        use super::unified::{DenseSpectralOperator, PseudoLogdetMode};

        let pirls_result = bundle.pirls_result.as_ref();
        let ridge_passport = pirls_result.ridge_passport;

        let mut h_total_original =
            self.bundle_matrix_in_original_basis(pirls_result, bundle.h_total.as_ref());

        let beta = self.sparse_exact_beta_original(pirls_result);

        // Barrier Hessian diagonal: restore ORIGINAL-basis barrier curvature
        // on `h_total_original`.
        //
        // `prepare_dense_eval_bundle` (runtime.rs:2192-2202) adds
        // `τ · diag(1/Δ_j²)` to `bundle.h_total` IN THE TRANSFORMED BASIS,
        // gated on
        // `barrier_config_from_constraints(pirls_result
        //  .linear_constraints_transformed)`.  That gate only accepts
        // simple-bound rows (exactly one nonzero = 1.0 in the transformed
        // constraint matrix).  `build_transformed_lower_bound_constraints`
        // / `build_transformed_linear_constraints` rotate each original
        // simple bound `e_jᵀ β_o ≥ b` into `qs.row(j)ᵀ β_t ≥ b`; for
        // non-trivial `Qs` that row is dense, so the transformed-basis
        // gate typically returns `None` and no barrier is added to
        // `bundle.h_total`.  The edge case where it DOES fire is when
        // `qs.row(j)` happens to equal some standard basis vector `e_kᵀ`
        // (Qs permutation-or-identity on that row), and the bundle then
        // carries `τ / (β_t[k] − b)²` at position `[k, k]` in the
        // transformed Hessian.
        //
        // After `bundle_matrix_in_original_basis` rotates by `Qs · ... ·
        //  Qsᵀ`, any such transformed diagonal increment becomes `Qs ·
        //  diag(τ/Δ_t²) · Qsᵀ` in the original basis — which is a
        // DIFFERENT matrix than the correct original-basis barrier
        // `τ · diag(1/Δ_o²)` at the original simple-bound indices.  Even
        // in the identity-preserving edge case, double-adding is wrong.
        //
        // Clean invariant: strip the rotated transformed-basis barrier
        // first, then apply the ORIGINAL-basis barrier directly.  The
        // transformed-basis barrier is reproducible: we can rebuild its
        // `BarrierConfig` from `pirls_result.linear_constraints_
        //  transformed` and its β from `pirls_result.beta_transformed`,
        // rotate the resulting diagonal up into the original basis via
        // `Qs · D · Qsᵀ`, and subtract it.  Then add the correct
        // `τ · diag_o(1/Δ_o²)` using `self.linear_constraints` and
        // `β_original`.
        //
        // This makes `log|H|`, its ρ-gradient (`trace_logdet_gradient(
        //  dH/dρ)` in unified.rs), and the `barrier_cost` added at
        // unified.rs:3106-3112 all derive from the same penalized
        // Hessian surface `X'WX + S − H_φ + τ·diag_o(1/Δ_o²)`, which
        // the `BarrierDerivativeProvider` at unified.rs:3133-3150
        // differentiates in the original basis via
        // `self.linear_constraints`-derived config.
        let transformed_barrier_to_strip = pirls_result
            .linear_constraints_transformed
            .as_ref()
            .and_then(Self::barrier_config_from_constraints);
        if let Some(ref barrier_cfg_t) = transformed_barrier_to_strip {
            let mut diag_trans = Array2::<f64>::zeros((beta.len(), beta.len()));
            let beta_t = pirls_result.beta_transformed.as_ref();
            match barrier_cfg_t.add_barrier_hessian_diagonal(&mut diag_trans, beta_t) {
                Ok(()) => {
                    let qs = &pirls_result.reparam_result.qs;
                    // diag_orig_form = Qs · diag_trans · Qsᵀ — the rotated
                    // form of whatever transformed barrier already sits in
                    // bundle.h_total.
                    let tmp = qs.dot(&diag_trans);
                    let rotated = tmp.dot(&qs.t());
                    h_total_original -= &rotated;
                }
                Err(e) => {
                    log::warn!(
                        "Transformed-basis barrier-diagonal reconstruction failed ({e}); \
                         leaving bundle Hessian unchanged before adding original-basis barrier"
                    );
                }
            }
        }
        if let Some(barrier_cfg) = self
            .linear_constraints
            .as_ref()
            .and_then(Self::barrier_config_from_constraints)
        {
            if let Err(e) = barrier_cfg.add_barrier_hessian_diagonal(&mut h_total_original, &beta) {
                log::warn!(
                    "Original-basis barrier Hessian diagonal skipped: {e}; \
                     cost/gradient/logdet consistency may regress on infeasible \
                     candidates (slack ≤ 0).  BFGS line search must maintain \
                     feasibility for the barrier-aware outer objective."
                );
            }
        }

        // Match `build_dense_assembly`: under Firth bias-reduction the penalized
        // Hessian `X'WX + S − H_φ` inherits the Jeffreys-identifiable subspace,
        // and `Smooth` would leak a spurious gradient contribution through the
        // null directions.  Keep `log|H|`, its gradient, its cross-traces, and
        // `H⁻¹` solves consistent on the same reduced active subspace by using
        // `HardPseudo` whenever a Firth operator is in play.
        let hessian_mode = if bundle.firth_dense_operator.is_some()
            || bundle.firth_dense_operator_original.is_some()
        {
            PseudoLogdetMode::HardPseudo
        } else {
            PseudoLogdetMode::Smooth
        };
        let hessian_op = std::sync::Arc::new(
            DenseSpectralOperator::from_symmetric_with_mode(&h_total_original, hessian_mode)
                .map_err(|e| {
                    EstimationError::InvalidInput(format!(
                        "DenseSpectralOperator from original-basis PIRLS Hessian: {e}"
                    ))
                })?,
        );

        let e_for_logdet = pirls_result.reparam_result.e_transformed.clone();
        let (penalty_rank, penalty_logdet) =
            self.dense_penalty_logdet_derivs(rho, &e_for_logdet, &[], ridge_passport, mode)?;

        let nullspace_dim = beta.len().saturating_sub(penalty_rank) as f64;

        // Same rank-deficient LAML fix as `build_dense_assembly`, adapted
        // to the original-basis Hessian.  Under `Smooth` the spectral
        // operator's cached logdet sums `ln r_ε(σ_j(H))` over ALL
        // eigenvalues; projecting H onto `range(S_+)` and passing the
        // scalar difference through `hessian_logdet_correction` restores
        // pairing with `log|S|_+`.  Under `HardPseudo` null eigenpairs
        // are already masked consistently in logdet / solves / traces,
        // so the correction is zero.
        //
        // `h_total_original` lives in the original basis; `e_for_logdet`
        // lives in the transformed basis (carries the reparameterization
        // that defines `range(S_+)`).  Rotate H into the transformed
        // basis via `Qs` before projecting so both inputs agree on which
        // subspace is "range(S_+)"; the scalar log-determinant is
        // invariant under the orthogonal Qs rotation, and comparing with
        // `hessian_op.logdet()` (original basis) is valid because the
        // two logdets agree to roundoff for the same matrix under
        // orthogonal change of basis.
        // As in `build_dense_assembly`: the gradient trace kernel must pair
        // with the projected cost `log|U_Sᵀ H U_S|_+`, not the full-space
        // `G_ε(H)`, on non-Gaussian families where the IFT correction
        // `D_β H[v] = X' diag(c ⊙ X v) X` leaks onto `null(S)`.  Build the
        // projected kernel in the transformed basis (where `e_for_logdet`
        // lives) and rotate `U_S` back into the original basis via `Qs`
        // since the ψ/τ drift matrices consumed by the trace are produced
        // in the original basis by this assembly path.
        let (hessian_logdet_correction, penalty_subspace_trace) =
            if matches!(hessian_mode, PseudoLogdetMode::Smooth)
                && penalty_rank < h_total_original.ncols()
            {
                use super::unified::HessianOperator;
                let qs = &pirls_result.reparam_result.qs;
                let h_transformed = qs.t().dot(&h_total_original).dot(qs);
                let (log_det_h_proj, kernel_trans) = self.fixed_subspace_hessian_projected_parts(
                    &h_transformed,
                    &e_for_logdet,
                    ridge_passport,
                )?;
                let kernel_orig = kernel_trans.map(|kernel_trans| {
                    let u_s_orig = qs.dot(&kernel_trans.u_s);
                    std::sync::Arc::new(super::unified::PenaltySubspaceTrace {
                        u_s: u_s_orig,
                        h_proj_inverse: kernel_trans.h_proj_inverse,
                    })
                });
                (log_det_h_proj - hessian_op.logdet(), kernel_orig)
            } else {
                (0.0, None)
            };

        let ctx = self.build_sparse_derivative_context(pirls_result, bundle)?;
        Ok(self.finish_assembly(
            pirls_result,
            ctx,
            hessian_op,
            beta,
            penalty_logdet,
            nullspace_dim,
            hessian_logdet_correction,
            penalty_subspace_trace,
        ))
    }

    /// Build an `InnerAssembly` by auto-detecting the backend.
    ///
    /// - Sparse-exact SPD  → `build_sparse_assembly`
    /// - QS-transformed + unconstrained (with or without ext_coords)
    ///     → `build_dense_original_assembly`
    /// - Otherwise → `build_dense_assembly` (used only when active
    ///     inequality constraints force evaluation in the transformed frame)
    ///
    /// Coordinate-frame correctness.
    /// `self.canonical_penalties` — which `build_penalty_coords()` feeds into
    /// the outer gradient as `PenaltyCoordinate::{DenseRoot, BlockRoot}` — is
    /// stored in the ORIGINAL (pre-`Qs`) coefficient basis. PIRLS, however,
    /// runs in the QS-transformed frame and reports `β_transformed =
    /// Qsᵀ · β_original`. If we hand the assembly `β_transformed` together
    /// with original-basis penalty roots, then `coord.apply_penalty(β, λ)`
    /// silently returns `λ · S_orig · β_t`, which is **not** a valid
    /// `∂(½ βᵀ Sβ)/∂ρₖ` in any single basis (the scalar `βᵀSβ` is invariant,
    /// but `Sβ` is not). That inconsistency is what starved BFGS of descent
    /// steps on large-n binomial+logit — only 2/200 line searches accepted —
    /// because the analytic gradient disagreed with the cost by a term that
    /// grew with the off-diagonal structure of `Qs`.
    ///
    /// Dispatching the unconstrained dense path through
    /// `build_dense_original_assembly` keeps `β`, the Hessian operator, the
    /// GLM derivative design, and the penalty coordinates all in the same
    /// (original) basis, restoring cost–gradient consistency. The only path
    /// that still needs `build_dense_assembly` is the monotonicity/active-set
    /// branch, where `free_basis_opt` already rotates everything into a
    /// reduced QS subspace (and the original-basis penalty-coord fast path
    /// isn't wired through the projected `Z` matrix yet).
    fn build_auto_assembly(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        mode: super::unified::EvalMode,
        _has_ext: bool,
    ) -> Result<super::assembly::InnerAssembly<'static>, EstimationError> {
        if bundle.backend_kind() == GeometryBackendKind::SparseExactSpd {
            self.build_sparse_assembly(rho, bundle, mode)
        } else if matches!(
            bundle.pirls_result.coordinate_frame,
            pirls::PirlsCoordinateFrame::TransformedQs
        ) && self
            .active_constraint_free_basis(bundle.pirls_result.as_ref())
            .is_none()
        {
            self.build_dense_original_assembly(rho, bundle, mode)
        } else {
            self.build_dense_assembly(rho, bundle, mode)
        }
    }

    /// Build the soft prior tuple for the given mode.
    fn build_prior(
        &self,
        rho: &Array1<f64>,
        mode: super::unified::EvalMode,
    ) -> Option<(f64, Array1<f64>, Option<Array2<f64>>)> {
        super::assembly::soft_prior_for_mode(
            rho,
            mode,
            |r| self.compute_soft_priorcost(r),
            |r| self.compute_soft_priorgrad(r),
            |r| self.compute_soft_priorhess(r),
        )
    }

    /// Single assembly point: evaluate an `InnerAssembly`, compute prior, and
    /// apply TK correction.
    ///
    /// All `evaluate_unified*` methods funnel through here.
    fn assemble_and_evaluate(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        mode: super::unified::EvalMode,
        assembly: super::assembly::InnerAssembly<'static>,
    ) -> Result<super::unified::RemlLamlResult, EstimationError> {
        let prior = self.build_prior(rho, mode);
        self.validate_tk_ext_coords(mode, &assembly.ext_coords)?;
        let tk_terms = self.tierney_kadane_terms(rho, bundle, mode, &assembly.ext_coords)?;
        let result = assembly
            .evaluate(rho.as_slice().unwrap(), mode, prior)
            .map_err(EstimationError::InvalidInput)?;
        self.apply_tk_to_result(result, tk_terms)
    }

    fn assemble_and_evaluate_efs(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        assembly: super::assembly::InnerAssembly<'static>,
    ) -> Result<crate::solver::outer_strategy::EfsEval, EstimationError> {
        use super::unified::{compute_efs_update, compute_hybrid_efs_update};

        let beta_for_barrier = assembly.beta.clone();
        let has_psi = assembly.ext_coords.iter().any(|c| !c.is_penalty_like);
        let eval_mode = if has_psi {
            super::unified::EvalMode::ValueAndGradient
        } else {
            super::unified::EvalMode::ValueOnly
        };
        self.validate_tk_ext_coords(eval_mode, &assembly.ext_coords)?;
        let tk_terms = self.tierney_kadane_terms(rho, bundle, eval_mode, &assembly.ext_coords)?;
        let inner_solution = assembly.build();

        let prior = self.build_prior(rho, eval_mode);
        let cost_result = super::assembly::evaluate_solution(
            &inner_solution,
            rho.as_slice().unwrap(),
            eval_mode,
            prior,
        )
        .map_err(EstimationError::InvalidInput)?;
        let cost_result = self.apply_tk_to_result(cost_result, tk_terms)?;

        let efs_eval = if has_psi {
            let gradient = cost_result.gradient.as_ref().expect(
                "hybrid EFS requires gradient (ValueAndGradient mode should have computed it)",
            );
            let hybrid = compute_hybrid_efs_update(
                &inner_solution,
                rho.as_slice().unwrap(),
                gradient.as_slice().unwrap(),
            );
            let psi_gradient = if hybrid.psi_indices.is_empty() {
                None
            } else {
                Some(ndarray::Array1::from_vec(hybrid.psi_gradient))
            };
            let psi_indices = if hybrid.psi_indices.is_empty() {
                None
            } else {
                Some(hybrid.psi_indices)
            };
            crate::solver::outer_strategy::EfsEval {
                cost: cost_result.cost,
                steps: hybrid.steps,
                beta: Some(beta_for_barrier),
                psi_gradient,
                psi_indices,
            }
        } else {
            let steps = compute_efs_update(&inner_solution, rho.as_slice().unwrap());
            crate::solver::outer_strategy::EfsEval {
                cost: cost_result.cost,
                steps,
                beta: Some(beta_for_barrier),
                psi_gradient: None,
                psi_indices: None,
            }
        };

        Ok(efs_eval)
    }

    /// Evaluate the REML/LAML objective (and optionally gradient) through the
    /// unified evaluator, using a pre-obtained eval bundle.
    ///
    /// This is the single bridge from the existing PIRLS infrastructure to the
    /// unified `reml_laml_evaluate` function. Both `compute_cost` and
    /// `compute_gradient` ultimately delegate here, ensuring that cost and
    /// gradient share the exact same formula.
    pub fn evaluate_unified(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        mode: super::unified::EvalMode,
    ) -> Result<super::unified::RemlLamlResult, EstimationError> {
        // Dispatch through `build_auto_assembly` rather than hard-wiring
        // `build_dense_assembly`.
        //
        // `build_dense_assembly` keeps β and H in PIRLS's QS-transformed frame
        // while `self.canonical_penalties` (consumed by the outer gradient via
        // `build_penalty_coords()`) stays in the ORIGINAL basis. The scalar
        // cost is invariant to that rotation, but `coord.apply_penalty(β, λ)`
        // is not: it returns `λ S_orig β_t`, which is neither `λ S_orig β_orig`
        // nor `λ S_t β_t`. That fed a miscalibrated `½ βᵀ Sβ` first partial
        // into BFGS on unconstrained dense problems (n=320k, p=71, duchon),
        // which accepted only 2/200 line-search steps before stalling while
        // an independent numeric check on the same cost showed descent
        // progress. `build_auto_assembly` routes unconstrained
        // `TransformedQs` fits through `build_dense_original_assembly`, which
        // maps β and H back to the original basis so every ingredient agrees
        // with the canonical-penalty coordinate frame.
        let assembly = self.build_auto_assembly(rho, bundle, mode, false)?;
        self.assemble_and_evaluate(rho, bundle, mode, assembly)
    }

    /// Sparse-exact bridge: delegates to `build_sparse_assembly` + `assemble_and_evaluate`.
    pub fn evaluate_unified_sparse(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        mode: super::unified::EvalMode,
    ) -> Result<super::unified::RemlLamlResult, EstimationError> {
        let assembly = self.build_sparse_assembly(rho, bundle, mode)?;
        self.assemble_and_evaluate(rho, bundle, mode, assembly)
    }
    /// Evaluate the unified REML/LAML objective with anisotropic ψ ext_coords
    /// injected into the `InnerSolution`.
    ///
    /// This is the primary bridge for joint [ρ, ψ] optimization of anisotropic
    /// spatial smooths. It:
    /// 1. Obtains the PIRLS bundle at the current ρ (with the design already
    ///    rebuilt for the current ψ).
    /// 2. Builds `HyperCoord` ext_coords from the supplied `DirectionalHyperParam`
    ///    list via the dense/transformed or sparse/native τ builders, depending
    ///    on the active backend.
    /// 3. Builds the full `InnerSolution` (mirroring `evaluate_unified`), injects
    ///    the ext_coords, and calls `reml_laml_evaluate`.
    ///
    /// The caller is responsible for rebuilding the `RemlState` with the design
    /// X(ψ) and penalties S(ψ) corresponding to the current ψ values before
    /// calling this method. The unified evaluator then preserves the backend:
    /// dense spectral paths stay dense, and sparse exact paths stay sparse.
    pub fn evaluate_unified_with_psi_ext(
        &self,
        rho: &Array1<f64>,
        mode: super::unified::EvalMode,
        hyper_dirs: &[crate::estimate::reml::DirectionalHyperParam],
    ) -> Result<super::unified::RemlLamlResult, EstimationError> {
        let t0 = std::time::Instant::now();
        let bundle = self.obtain_eval_bundle(rho)?;
        let pirls_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = std::time::Instant::now();
        let (ext_coords, ext_pair_fn, rho_ext_pair_fn, fixed_drift_deriv) =
            if !hyper_dirs.is_empty() {
                if mode == super::unified::EvalMode::ValueGradientHessian {
                    let (coords, epf, repf, fixed_drift_deriv) =
                        self.build_tau_unified_objects_from_bundle(rho, &bundle, hyper_dirs)?;
                    (coords, Some(epf), Some(repf), fixed_drift_deriv)
                } else if bundle.backend_kind() == GeometryBackendKind::SparseExactSpd {
                    (
                        self.build_tau_hyper_coords_sparse_exact(rho, &bundle, hyper_dirs)?,
                        None,
                        None,
                        None,
                    )
                } else if matches!(
                    bundle.pirls_result.coordinate_frame,
                    pirls::PirlsCoordinateFrame::TransformedQs
                ) && self
                    .active_constraint_free_basis(bundle.pirls_result.as_ref())
                    .is_none()
                {
                    (
                        self.build_tau_hyper_coords_original_basis(rho, &bundle, hyper_dirs)?,
                        None,
                        None,
                        None,
                    )
                } else {
                    (
                        self.build_tau_hyper_coords(rho, &bundle, hyper_dirs)?,
                        None,
                        None,
                        None,
                    )
                }
            } else {
                (Vec::new(), None, None, None)
            };
        let tau_build_ms = t1.elapsed().as_secs_f64() * 1000.0;
        let t2 = std::time::Instant::now();
        let mut assembly = self.build_auto_assembly(rho, &bundle, mode, !ext_coords.is_empty())?;
        assembly.ext_coords = ext_coords;
        assembly.ext_coord_pair_fn = ext_pair_fn;
        assembly.rho_ext_pair_fn = rho_ext_pair_fn;
        assembly.fixed_drift_deriv = fixed_drift_deriv;
        let result = self.assemble_and_evaluate(rho, &bundle, mode, assembly);
        let reml_eval_ms = t2.elapsed().as_secs_f64() * 1000.0;

        log::info!(
            "[outer-timing] evaluate_unified_with_psi_ext: PIRLS={:.1}ms  tau_build={:.1}ms  reml_eval={:.1}ms  total={:.1}ms",
            pirls_ms,
            tau_build_ms,
            reml_eval_ms,
            t0.elapsed().as_secs_f64() * 1000.0,
        );
        result
    }

    /// Compute EFS (Extended Fellner-Schall) evaluation: runs the inner solve
    /// at the given rho, computes the REML/LAML cost, and returns the EFS
    /// step vector from `compute_efs_update`.
    ///
    /// This is the entry point called by the EFS branch of `run_outer` via
    /// the `OuterObjective::eval_efs` method.
    pub fn compute_efs_steps(
        &self,
        p: &Array1<f64>,
    ) -> Result<crate::solver::outer_strategy::EfsEval, EstimationError> {
        let bundle = match self.obtain_eval_bundle(p) {
            Ok(bundle) => bundle,
            Err(EstimationError::ModelIsIllConditioned { .. }) => {
                self.cache_manager.invalidate_eval_bundle();
                return Err(EstimationError::RemlOptimizationFailed(
                    "inner solve ill-conditioned during EFS evaluation".to_string(),
                ));
            }
            Err(e) => {
                self.cache_manager.invalidate_eval_bundle();
                return Err(e);
            }
        };

        self.evaluate_efs(p, &bundle, Vec::new())
    }

    /// EFS evaluation with anisotropic or isotropic ψ ext_coords injected.
    pub fn compute_efs_steps_with_psi_ext(
        &self,
        rho: &Array1<f64>,
        hyper_dirs: &[crate::estimate::reml::DirectionalHyperParam],
    ) -> Result<crate::solver::outer_strategy::EfsEval, EstimationError> {
        let bundle = match self.obtain_eval_bundle(rho) {
            Ok(bundle) => bundle,
            Err(EstimationError::ModelIsIllConditioned { .. }) => {
                self.cache_manager.invalidate_eval_bundle();
                return Err(EstimationError::RemlOptimizationFailed(
                    "inner solve ill-conditioned during psi-ext EFS evaluation".to_string(),
                ));
            }
            Err(e) => {
                self.cache_manager.invalidate_eval_bundle();
                return Err(e);
            }
        };

        let ext_coords = if !hyper_dirs.is_empty() {
            if bundle.backend_kind() == GeometryBackendKind::SparseExactSpd {
                self.build_tau_hyper_coords_sparse_exact(rho, &bundle, hyper_dirs)?
            } else if matches!(
                bundle.pirls_result.coordinate_frame,
                pirls::PirlsCoordinateFrame::TransformedQs
            ) && self
                .active_constraint_free_basis(bundle.pirls_result.as_ref())
                .is_none()
            {
                self.build_tau_hyper_coords_original_basis(rho, &bundle, hyper_dirs)?
            } else {
                self.build_tau_hyper_coords(rho, &bundle, hyper_dirs)?
            }
        } else {
            Vec::new()
        };

        if ext_coords.is_empty() {
            return self.compute_efs_steps(rho);
        }

        self.evaluate_efs(rho, &bundle, ext_coords)
    }

    pub fn compute_gradient(&self, p: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
        self.arena
            .lastgradient_used_stochastic_fallback
            .store(false, Ordering::Relaxed);
        let t_eval_start = std::time::Instant::now();
        {
            let prefix: Vec<String> = p.iter().take(4).map(|v| format!("{:.3}", v)).collect();
            log::info!(
                "[REML] grad-only begin | rho[..4]=[{}] | k={}",
                prefix.join(","),
                p.len()
            );
        }
        let rho_key = EvalCacheManager::sanitized_rhokey(p);
        if let Some(eval) = self.cache_manager.cached_outer_eval(&rho_key) {
            let gnorm = eval.gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
            log::info!(
                "[REML] grad-only cache hit | |g| {:.3e} | elapsed {:.1}ms",
                gnorm,
                t_eval_start.elapsed().as_secs_f64() * 1000.0
            );
            return Ok(eval.gradient);
        }
        let t_pirls = std::time::Instant::now();
        let bundle = match self.obtain_eval_bundle(p) {
            Ok(bundle) => bundle,
            Err(err @ EstimationError::ModelIsIllConditioned { .. }) => {
                self.cache_manager.invalidate_eval_bundle();
                return Err(err);
            }
            Err(e) => {
                self.cache_manager.invalidate_eval_bundle();
                return Err(e);
            }
        };
        let pirls_ms = t_pirls.elapsed().as_secs_f64() * 1000.0;
        log::info!(
            "[REML] grad-only pirls done | elapsed {:.1}ms | backend {:?}",
            pirls_ms,
            bundle.backend_kind()
        );
        let t_assemble = std::time::Instant::now();
        if bundle.backend_kind() == GeometryBackendKind::SparseExactSpd {
            let result = self.evaluate_unified_sparse(
                p,
                &bundle,
                super::unified::EvalMode::ValueAndGradient,
            )?;
            let grad = result.gradient.unwrap();
            let gnorm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
            log::info!(
                "[REML] grad-only sparse done | |g| {:.3e} | assemble {:.1}ms | total {:.1}ms",
                gnorm,
                t_assemble.elapsed().as_secs_f64() * 1000.0,
                t_eval_start.elapsed().as_secs_f64() * 1000.0
            );
            return Ok(grad);
        }
        let result =
            self.evaluate_unified(p, &bundle, super::unified::EvalMode::ValueAndGradient)?;
        let grad = result.gradient.unwrap();
        let gnorm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        log::info!(
            "[REML] grad-only dense done | |g| {:.3e} | assemble {:.1}ms | total {:.1}ms",
            gnorm,
            t_assemble.elapsed().as_secs_f64() * 1000.0,
            t_eval_start.elapsed().as_secs_f64() * 1000.0
        );
        Ok(grad)
    }

    pub fn compute_outer_eval_with_order(
        &self,
        p: &Array1<f64>,
        order: crate::solver::outer_strategy::OuterEvalOrder,
    ) -> Result<OuterEval, EstimationError> {
        self.arena
            .lastgradient_used_stochastic_fallback
            .store(false, Ordering::Relaxed);
        let allow_second_order = matches!(
            order,
            crate::solver::outer_strategy::OuterEvalOrder::ValueGradientHessian
        ) && self.analytic_outer_hessian_enabled();
        let t_eval_start = std::time::Instant::now();
        {
            let prefix: Vec<String> = p.iter().take(4).map(|v| format!("{:.3}", v)).collect();
            log::info!(
                "[REML] outer-eval begin {:?} | rho[..4]=[{}] | k={} | 2nd-order={}",
                order,
                prefix.join(","),
                p.len(),
                allow_second_order
            );
        }
        let rho_key = EvalCacheManager::sanitized_rhokey(p);
        if let Some(eval) = self.cache_manager.cached_outer_eval(&rho_key) {
            let cache_satisfies_request = !allow_second_order || eval.hessian.is_analytic();
            if cache_satisfies_request {
                let gnorm = eval.gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
                log::info!(
                    "[REML] outer-eval cache hit | cost {:.6e} | |g| {:.3e} | elapsed {:.1}ms",
                    eval.cost,
                    gnorm,
                    t_eval_start.elapsed().as_secs_f64() * 1000.0
                );
                return Ok(eval);
            }
        }

        let t_pirls = std::time::Instant::now();
        let bundle = match self.obtain_eval_bundle(p) {
            Ok(bundle) => bundle,
            Err(EstimationError::ModelIsIllConditioned { .. }) => {
                self.cache_manager.invalidate_eval_bundle();
                log::warn!(
                    "P-IRLS flagged ill-conditioning for current rho; returning infeasible outer eval to retreat."
                );
                return Ok(OuterEval::infeasible(p.len()));
            }
            Err(EstimationError::PerfectSeparationDetected { .. })
            | Err(EstimationError::PirlsDidNotConverge { .. }) => {
                self.cache_manager.invalidate_eval_bundle();
                log::warn!(
                    "P-IRLS separation/non-convergence at current rho; returning infeasible outer eval to retreat."
                );
                return Ok(OuterEval::infeasible(p.len()));
            }
            Err(err) => {
                self.cache_manager.invalidate_eval_bundle();
                return Err(err);
            }
        };

        let decision = match order {
            crate::solver::outer_strategy::OuterEvalOrder::ValueAndGradient => None,
            crate::solver::outer_strategy::OuterEvalOrder::ValueGradientHessian => {
                if allow_second_order {
                    Some(self.selecthessian_strategy_policy(p, &bundle))
                } else {
                    None
                }
            }
        };
        let eval_mode = match decision.as_ref().map(|decision| decision.strategy) {
            Some(HessianEvalStrategyKind::SpectralExact) => {
                super::unified::EvalMode::ValueGradientHessian
            }
            _ => super::unified::EvalMode::ValueAndGradient,
        };

        let pirls_ms = t_pirls.elapsed().as_secs_f64() * 1000.0;
        log::info!(
            "[REML] outer-eval pirls done | elapsed {:.1}ms | backend {:?} | mode {:?}",
            pirls_ms,
            bundle.backend_kind(),
            eval_mode
        );

        let t_assemble = std::time::Instant::now();
        let result = if bundle.backend_kind() == GeometryBackendKind::SparseExactSpd {
            self.evaluate_unified_sparse(p, &bundle, eval_mode)?
        } else {
            self.evaluate_unified(p, &bundle, eval_mode)?
        };
        let assemble_ms = t_assemble.elapsed().as_secs_f64() * 1000.0;

        let gradient = result.gradient.ok_or_else(|| {
            EstimationError::InvalidInput(format!(
                "unified evaluator returned no gradient in {:?} mode",
                eval_mode
            ))
        })?;

        let hessian = match decision.map(|decision| decision.strategy) {
            Some(HessianEvalStrategyKind::SpectralExact) => result.hessian,
            None => HessianResult::Unavailable,
        };

        let eval = OuterEval {
            cost: result.cost,
            gradient,
            hessian,
        };
        {
            let gnorm = eval.gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
            log::info!(
                "[REML] outer-eval done | cost {:.6e} | |g| {:.3e} | assemble {:.1}ms | total {:.1}ms",
                eval.cost,
                gnorm,
                assemble_ms,
                t_eval_start.elapsed().as_secs_f64() * 1000.0
            );
        }
        self.cache_manager.store_outer_eval(&rho_key, &eval);
        Ok(eval)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  Unified REML evaluation with link-parameter ext_coords
    // ═══════════════════════════════════════════════════════════════════════════

    /// Evaluate the unified REML/LAML objective with SAS or mixture link
    /// parameters injected as ext_coords into the `InnerSolution`.
    ///
    /// This replaces the manual gradient computation in the outer optimizer's
    /// eval_fn closure. The link parameter derivatives (du/dθ, dW/dθ, dℓ/dθ)
    /// are wrapped as `HyperCoord` objects and passed through the same unified
    /// evaluator that handles ρ coordinates, giving a single Newton/BFGS step
    /// over `[ρ, θ_link]` jointly.
    ///
    /// # Parameters
    ///
    /// - `rho`: log-smoothing parameters (penalty coordinates only, length k)
    /// - `mode`: evaluation mode (cost-only, cost+gradient, cost+gradient+Hessian)
    ///
    /// # Returns
    ///
    /// `RemlLamlResult` whose gradient (when requested) has length `k + aux_dim`,
    /// with the link parameters appended after the ρ coordinates. The caller is
    /// responsible for:
    /// - Applying SAS epsilon reparameterization chain rule (`grad[k] *= d_eps/d_raw`)
    /// - Adding SAS ridge/barrier gradient contributions to `grad[k+1]`
    /// - Adding SAS ridge/barrier cost contributions to `result.cost`
    /// Build link ext_coords from the current runtime link state.
    fn build_link_ext_coords(
        &self,
        bundle: &EvalShared,
    ) -> Result<Vec<super::unified::HyperCoord>, EstimationError> {
        if let Some(sas_state) = &self.runtime_sas_link_state {
            let is_beta_logistic = matches!(
                self.config.link_function(),
                crate::types::LinkFunction::BetaLogistic
            );
            self.build_sas_link_ext_coords(bundle, sas_state, is_beta_logistic)
        } else if let Some(mix_state) = &self.runtime_mixture_link_state {
            self.build_mixture_link_ext_coords(bundle, mix_state)
        } else {
            Ok(Vec::new())
        }
    }

    /// Check that Firth is not active (incompatible with link ext_coords).
    fn reject_firth_link_ext(&self) -> Result<(), EstimationError> {
        if self.config.firth_bias_reduction {
            return Err(EstimationError::InvalidInput(
                "link-parameter ext_coord optimization is incompatible with \
                 Firth-adjusted outer gradients"
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// Rotate link-parameter `HyperCoord`s from the PIRLS transformed basis
    /// into the original coefficient basis, gated on the same predicate
    /// `build_auto_assembly` uses to select `build_dense_original_assembly`.
    ///
    /// The SAS / mixture / beta-logistic link-ext builders
    /// (`build_sas_link_ext_coords`, `build_mixture_link_ext_coords`) form
    /// `g_j = −X_tᵀ·du/dθ_j` and `B_j = X_tᵀ·diag(dW/dθ_j)·X_t` from
    /// `pirls_result.x_transformed`, i.e. in the transformed basis.  When
    /// PIRLS is in `TransformedQs` with no active constraints,
    /// `build_auto_assembly` routes through `build_dense_original_assembly`
    /// — which reports `β` and `H` in the ORIGINAL basis.  Attaching
    /// transformed-basis `(g_j, B_j)` to an original-basis assembly would
    /// evaluate `coord.g·β` and `coord.drift·v` across mismatched frames.
    ///
    /// Under an orthogonal rotation `Qs` with `β_original = Qs·β_transformed`
    /// (matching `sparse_exact_beta_original` /
    /// `bundle_matrix_in_original_basis`), the covariant / bilinear
    /// transformation laws give
    ///
    ///   g_original = Qs · g_transformed                   (covariant vector)
    ///   B_original = Qs · B_transformed · Qsᵀ            (bilinear form)
    ///
    /// derived from scalar invariance `⟨g_o, β_o⟩ = ⟨g_t, β_t⟩` and
    /// `β_oᵀ B_o β_o = β_tᵀ B_t β_t`, both under `β_o = Qs β_t`.
    ///
    /// Scalars (`a`, `ld_s`) and semantic flags (`b_depends_on_beta`,
    /// `is_penalty_like`) are basis-invariant by construction.  The
    /// `HyperCoordDrift::block_local` and `operator` variants are not
    /// handled by this helper — the link-ext builders must emit only
    /// the `dense` variant (`HyperCoordDrift::from_dense`), and the
    /// helper returns an error if that invariant is broken so a future
    /// builder change trips it immediately rather than silently
    /// regressing SAS/mixture optimisation.
    fn rotate_link_ext_coords_to_original(
        &self,
        bundle: &EvalShared,
        coords: &mut [super::unified::HyperCoord],
    ) -> Result<(), EstimationError> {
        if coords.is_empty() {
            return Ok(());
        }
        let pirls_result = bundle.pirls_result.as_ref();
        // Sparse-native (OriginalSparseNative) path is structurally
        // identity-rotated: `make_reparam_operator` sets
        // `x_transformed = x_original` and `sparse_exact_beta_original`
        // returns β verbatim, so link-ext coords built from
        // `pirls_result.x_transformed` are already in the original basis
        // that `build_sparse_assembly` reports `β` and `H` in.  The gate
        // below therefore returns Ok(()) without touching coords whenever
        // the frame is OriginalSparseNative — the analogue of 4ec0208f's
        // dense rotation is a no-op on sparse, not a missing fix.
        let needs_rotation = matches!(
            pirls_result.coordinate_frame,
            pirls::PirlsCoordinateFrame::TransformedQs
        ) && self.active_constraint_free_basis(pirls_result).is_none();
        if !needs_rotation {
            return Ok(());
        }
        let qs = &pirls_result.reparam_result.qs;
        let qs_t = qs.t();
        for (coord_idx, coord) in coords.iter_mut().enumerate() {
            if coord.g.len() == qs.nrows() {
                coord.g = qs.dot(&coord.g);
            }
            // Invariant guard: link-ext builders must emit ONLY the
            // `dense` variant of `HyperCoordDrift`.  `block_local` and
            // `operator` variants would require distinct rotations
            // (block-local would lose its block structure under
            // arbitrary `Qs`; operator-backed drifts would need a
            // wrapping `RotatedHyperOperator`).  If a future link-ext
            // builder ever populates either variant, this guard
            // refuses rather than silently running a partial rotation
            // that regresses SAS/mixture optimisation.
            if coord.drift.block_local.is_some() || coord.drift.operator.is_some() {
                return Err(EstimationError::InvalidInput(format!(
                    "link-ext HyperCoord[{coord_idx}] carries a non-dense drift \
                     variant (block_local={} operator={}); the coord-frame rotation \
                     helper only handles `HyperCoordDrift::from_dense`.  Update the \
                     link-ext builder (or extend `rotate_link_ext_coords_to_original`) \
                     before attaching non-dense drifts to original-basis assemblies.",
                    coord.drift.block_local.is_some(),
                    coord.drift.operator.is_some(),
                )));
            }
            if let Some(b) = coord.drift.dense.as_mut() {
                if b.nrows() == qs.nrows() && b.ncols() == qs.nrows() {
                    // B_original = Qs · B_transformed · Qsᵀ
                    let tmp = qs.dot(&*b);
                    *b = tmp.dot(&qs_t);
                }
            }
        }
        Ok(())
    }

    pub fn evaluate_unified_with_link_ext(
        &self,
        rho: &Array1<f64>,
        mode: super::unified::EvalMode,
    ) -> Result<super::unified::RemlLamlResult, EstimationError> {
        self.reject_firth_link_ext()?;
        let bundle = self.obtain_eval_bundle(rho)?;
        let mut ext_coords = self.build_link_ext_coords(&bundle)?;

        if ext_coords.is_empty() {
            return self.evaluate_unified(rho, &bundle, mode);
        }

        // Rotate link-ext coords to match the basis `build_auto_assembly`
        // will pick for the assembly (original basis under TransformedQs +
        // no active constraints).  See `rotate_link_ext_coords_to_original`.
        self.rotate_link_ext_coords_to_original(&bundle, &mut ext_coords)?;

        let mut assembly = self.build_auto_assembly(rho, &bundle, mode, true)?;
        assembly.ext_coords = ext_coords;
        self.assemble_and_evaluate(rho, &bundle, mode, assembly)
    }

    /// EFS evaluation with link-parameter ext_coords injected.
    pub fn compute_efs_steps_with_link_ext(
        &self,
        rho: &Array1<f64>,
    ) -> Result<crate::solver::outer_strategy::EfsEval, EstimationError> {
        self.reject_firth_link_ext()?;
        let bundle = self.obtain_eval_bundle(rho)?;
        let mut ext_coords = self.build_link_ext_coords(&bundle)?;

        if ext_coords.is_empty() {
            return self.compute_efs_steps(rho);
        }

        // Same rotation gate as `evaluate_unified_with_link_ext`: the EFS
        // path eventually builds its assembly via `build_auto_assembly`
        // inside `evaluate_efs`, which picks original basis under the same
        // predicate.
        self.rotate_link_ext_coords_to_original(&bundle, &mut ext_coords)?;

        self.evaluate_efs(rho, &bundle, ext_coords)
    }

    /// Unified EFS bridge: auto-dispatches backend, injects ext_coords,
    /// and evaluates via `assemble_and_evaluate_efs`.
    fn evaluate_efs(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        ext_coords: Vec<super::unified::HyperCoord>,
    ) -> Result<crate::solver::outer_strategy::EfsEval, EstimationError> {
        let mut assembly = self.build_auto_assembly(
            rho,
            bundle,
            super::unified::EvalMode::ValueOnly,
            !ext_coords.is_empty(),
        )?;
        assembly.tk_gradient = None;
        assembly.ext_coords = ext_coords;
        self.assemble_and_evaluate_efs(rho, bundle, assembly)
    }
}

#[cfg(test)]
mod tk_math_tests {
    use super::*;
    use crate::faer_ndarray::FaerCholesky;
    use faer::Side;
    use ndarray::array;
    use num_dual::{Dual64, DualNum};

    fn solve_vec(h: &Array2<f64>, rhs: &Array1<f64>) -> Array1<f64> {
        h.cholesky(Side::Lower).expect("chol(H)").solvevec(rhs)
    }

    fn solve_xt(h: &Array2<f64>, x: &Array2<f64>) -> Array2<f64> {
        let mut xt = x.t().to_owned();
        h.cholesky(Side::Lower)
            .expect("chol(H)")
            .solve_mat_in_place(&mut xt);
        xt
    }

    /// Closed-form TK scalar V_TK in any DualNum-compatible field, for a
    /// scalar (1×1) Hessian H.  Restricting to p = 1 keeps the matrix
    /// solve to a single division, which lets us evaluate the entire
    /// V_TK formula in Dual64 and read the analytic derivative directly
    /// from the dual part — no finite differences anywhere.
    ///
    /// All quantities derive from the same closed forms used by
    /// `tk_scalar_from_shared`, just expanded for the n×1 case:
    ///   K_ij = (x_i * x_j) / h
    ///   h_ii = K_ii = x_i² / h
    ///   m_i  = c_i * h_ii
    ///   y    = (Σ x_i m_i) / h          (since H⁻¹ is 1/h)
    ///   q    = Σ x_i m_i
    ///   V_TK = −¹⁄₈ Σ d_i h_ii²
    ///          + ¹⁄₁₂ Σᵢⱼ c_i c_j K_ij³
    ///          + ¹⁄₈ q · y
    fn tk_scalar_dual<D: DualNum<f64> + Copy>(h: D, x: &[D], c: &[D], d_arr: &[D]) -> D {
        let n = x.len();
        let inv_h = D::one() / h;
        let mut h_diag = vec![D::from(0.0); n];
        for i in 0..n {
            h_diag[i] = x[i] * x[i] * inv_h;
        }
        let mut d_term = D::from(0.0);
        for i in 0..n {
            d_term += d_arr[i] * h_diag[i] * h_diag[i];
        }
        d_term *= D::from(-0.125);

        let mut c_term = D::from(0.0);
        for i in 0..n {
            for j in 0..n {
                let k_ij = x[i] * x[j] * inv_h;
                c_term += c[i] * c[j] * k_ij * k_ij * k_ij;
            }
        }
        c_term *= D::from(1.0 / 12.0);

        let mut q = D::from(0.0);
        for i in 0..n {
            q += x[i] * c[i] * h_diag[i];
        }
        let q_term = D::from(0.125) * q * q * inv_h;

        d_term + c_term + q_term
    }

    /// Verify that `tk_gradient_from_shared` computes the exact analytic
    /// directional derivative ∂V_TK/∂θ along a hand-chosen direction
    /// (h_dot, x_dot, c_dot = d⊙η_dot, d_dot = e⊙η_dot).
    ///
    /// Reference is computed via dual-number AD on the closed-form
    /// scalar in [`tk_scalar_dual`] — there is no finite-difference
    /// stencil involved.  The setup uses p = 1 so the matrix solve
    /// inside H reduces to a scalar division, which Dual64 handles
    /// natively.
    #[test]
    fn tierney_kadane_gradient_matches_dual_ad_reference_scalarhessian() {
        // p = 1, n = 4 toy problem.
        let x_vec = vec![1.3_f64, -0.4, 0.7, -0.9];
        let h_val = 2.5_f64;
        let c = array![0.21_f64, -0.13, 0.18, 0.07];
        let d = array![-0.05_f64, 0.09, 0.04, -0.07];
        let e = array![0.03_f64, -0.04, 0.02, 0.018];
        let eta_dot = array![0.11_f64, -0.06, 0.07, -0.085];
        let x_dot_vec = vec![0.025_f64, -0.018, 0.022, -0.014];
        let h_dot_val = 0.07_f64;

        // Build dual versions of (h, x, c, d) at θ = 0 with derivatives
        // (h_dot, x_dot, c_dot = d ⊙ η_dot, d_dot = e ⊙ η_dot).  This is
        // the same chain rule the production code applies internally.
        let h_dual = Dual64::new(h_val, h_dot_val);
        let x_dual: Vec<Dual64> = x_vec
            .iter()
            .zip(x_dot_vec.iter())
            .map(|(&v, &dv)| Dual64::new(v, dv))
            .collect();
        let c_dual: Vec<Dual64> = c
            .iter()
            .zip(d.iter().zip(eta_dot.iter()))
            .map(|(&cv, (&dv, &edot))| Dual64::new(cv, dv * edot))
            .collect();
        let d_dual: Vec<Dual64> = d
            .iter()
            .zip(e.iter().zip(eta_dot.iter()))
            .map(|(&dv, (&ev, &edot))| Dual64::new(dv, ev * edot))
            .collect();

        let v_tk_dual = tk_scalar_dual(h_dual, &x_dual, &c_dual, &d_dual);
        let dv_tk_ad = v_tk_dual.eps;
        let v_tk_value = v_tk_dual.re;

        // Sanity-check the closed-form scalar against the production
        // `tk_scalar_from_shared` at θ = 0.
        let x_mat = Array2::from_shape_vec((4, 1), x_vec.clone()).unwrap();
        let h_mat = Array2::from_shape_vec((1, 1), vec![h_val]).unwrap();
        let z_mat = solve_xt(&h_mat, &x_mat);
        let solve = |rhs: &Array1<f64>| -> Result<Array1<f64>, EstimationError> {
            Ok(solve_vec(&h_mat, rhs))
        };
        let shared = RemlState::tk_shared_intermediates(
            &x_mat,
            &z_mat,
            &c,
            "tk scalar dual baseline",
            &solve,
        )
        .expect("shared TK intermediates");
        let mut gram = Array2::<f64>::zeros((TK_BLOCK_SIZE, TK_BLOCK_SIZE));
        let v_tk_prod = RemlState::tk_scalar_from_shared(&x_mat, &z_mat, &d, &shared, &mut gram)
            .expect("TK scalar (production)");
        let scalar_rel =
            (v_tk_value - v_tk_prod).abs() / v_tk_value.abs().max(v_tk_prod.abs()).max(1.0e-14);
        assert!(
            scalar_rel < 1.0e-12,
            "TK scalar mismatch (production vs closed-form): prod={v_tk_prod:.12e}, dual_re={v_tk_value:.12e}, rel={scalar_rel:.3e}"
        );

        // Production analytic derivative through the same
        // `tk_gradient_from_shared` path that runs in REML evaluation.
        // We treat the derivative as a single ext coordinate so the
        // returned gradient[0] is precisely ∂V_TK/∂θ.
        let h_dot_mat = Array2::from_shape_vec((1, 1), vec![h_dot_val]).unwrap();
        let x_dot_mat = Array2::from_shape_vec((4, 1), x_dot_vec.clone()).unwrap();
        let gradient = RemlState::tk_gradient_from_shared(
            &x_mat,
            &z_mat,
            &c,
            &d,
            &e,
            &[],
            &[],
            &[h_dot_mat],
            &[Some(eta_dot.clone())],
            &[Some(x_dot_mat)],
            &[Array1::<f64>::zeros(x_mat.nrows())],
            &[Array1::<f64>::zeros(x_mat.ncols())],
            None,
            &shared,
            &mut gram,
        )
        .expect("analytic TK derivative");
        let analytic = gradient[0];

        let rel = (analytic - dv_tk_ad).abs() / analytic.abs().max(dv_tk_ad.abs()).max(1.0e-14);
        assert!(
            rel < 1.0e-12,
            "Tierney-Kadane analytic c/d propagation does not match Dual64 AD reference: analytic={analytic:.12e}, ad={dv_tk_ad:.12e}, rel={rel:.3e}"
        );
    }
}
