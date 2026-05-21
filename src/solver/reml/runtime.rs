use super::*;
use crate::construction::{
    create_balanced_penalty_root_from_canonical, precompute_reparam_invariant_from_canonical,
};
use crate::linalg::sparse_exact::build_sparse_penalty_blocks_from_canonical;
use crate::linalg::utils::{
    boundary_hit_indices, enforce_symmetry, symmetric_spectrum_condition_number,
};
use crate::pirls::PirlsWorkspace;
use crate::solver::estimate::reml::inner_strategy::HessianEvalStrategyKind;
use crate::solver::outer_strategy::{HessianResult, OuterEval};
use crate::solver::persistent_warm_start::{
    PersistentWarmStartRecord, StableHasher, load_record, store_record,
};
use crate::types::{InverseLink, LinkFunction, RhoPrior, SasLinkState};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

const TK_BLOCK_SIZE: usize = 128;
const TK_MAX_OBSERVATIONS: usize = 20_000;
const TK_MAX_COEFFICIENTS: usize = 2_000;
const TK_MAX_DENSE_WORK: usize = 5_000_000;

#[inline]
fn compute_gradient_for_tk(mode: super::unified::EvalMode) -> bool {
    mode != super::unified::EvalMode::ValueOnly
}

/// Apply the screening residual penalty to a cost.
///
/// Under multi-start seed screening (a ranking pass over candidate ρ
/// vectors), the inner P-IRLS is intentionally capped at a few iterations.
/// Partial modes that do not certify stationarity are accepted for ranking
/// in `execute_pirls_if_needed`; this helper turns the partial cost into a
/// finite ranking score
///
/// ```text
/// C_screen(s) = C_approx(s) + ½ · r_g(s)² ,
/// ```
///
/// where r_g = ‖g‖ / (1 + ‖score‖ + ‖Sβ‖ + ridge·‖β‖) is the scale-invariant
/// relative gradient residual. Using r_g rather than the absolute ‖g‖ keeps
/// the penalty meaningful at biobank n: the absolute score grows as O(√n),
/// so an absolute residual term would swamp the actual REML cost differences
/// across seeds and reduce the screen to a √n-scaled tie-break. r_g is
/// dimensionless and bounded above by 1 for any well-defined PIRLS state, so
/// the penalty stays comparable to the cost differences that actually
/// distinguish good seeds from bad. The penalty vanishes at the true inner
/// mode (r_g → 0), so converged screening fits incur no penalty.
///
/// Outside of screening, partial fits never reach this helper:
/// `execute_pirls_if_needed` surfaces `MaxIterationsReached` and
/// `LmStepSearchExhausted` as `EstimationError::PirlsDidNotConverge`, so
/// production REML evaluations always operate on certified inner modes
/// and this helper is a strict no-op for them.
#[inline]
fn screening_residual_penalty(cost: f64, pr: &PirlsResult) -> f64 {
    if !cost.is_finite() || !pr.status.is_failed_max_iterations() {
        return cost;
    }
    let r_g = pr.relative_gradient_norm();
    if r_g.is_finite() {
        cost + 0.5 * r_g * r_g
    } else {
        f64::INFINITY
    }
}

fn hash_array_view(hasher: &mut StableHasher, values: ndarray::ArrayView1<'_, f64>) {
    hasher.write_usize(values.len());
    for &value in values {
        hasher.write_f64(value);
    }
}

fn hash_array2(hasher: &mut StableHasher, values: &Array2<f64>) {
    hasher.write_usize(values.nrows());
    hasher.write_usize(values.ncols());
    for &value in values {
        hasher.write_f64(value);
    }
}

fn hash_design_matrix(hasher: &mut StableHasher, design: &DesignMatrix) -> Result<(), String> {
    let n = design.nrows();
    let p = design.ncols();
    hasher.write_usize(n);
    hasher.write_usize(p);
    let bytes_per_row = p.saturating_mul(std::mem::size_of::<f64>()).max(1);
    let chunk_rows = ((8 * 1024 * 1024) / bytes_per_row).clamp(1, 4096);
    for start in (0..n).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(n);
        let chunk = design
            .try_row_chunk(start..end)
            .map_err(|e| format!("persistent warm-start design hash failed: {e}"))?;
        hash_array2(hasher, &chunk);
    }
    Ok(())
}

fn hash_canonical_penalties(
    hasher: &mut StableHasher,
    penalties: &[crate::construction::CanonicalPenalty],
) {
    hasher.write_usize(penalties.len());
    for penalty in penalties {
        hasher.write_usize(penalty.col_range.start);
        hasher.write_usize(penalty.col_range.end);
        hasher.write_usize(penalty.total_dim);
        hasher.write_usize(penalty.nullity);
        hash_array2(hasher, &penalty.root);
        hash_array2(hasher, &penalty.local);
        hash_array_view(hasher, penalty.prior_mean.view());
        hasher.write_usize(penalty.positive_eigenvalues.len());
        for &value in &penalty.positive_eigenvalues {
            hasher.write_f64(value);
        }
        hasher.write_bool(penalty.op.is_some());
    }
}

fn finite_positive_from_bits(bits: u64) -> Option<f64> {
    if bits == 0 {
        return None;
    }
    let value = f64::from_bits(bits);
    if value.is_finite() && value > 0.0 {
        Some(value)
    } else {
        None
    }
}

fn finite_nonnegative_from_bits(bits: u64) -> Option<f64> {
    let value = f64::from_bits(bits);
    if value.is_finite() && value >= 0.0 {
        Some(value)
    } else {
        None
    }
}

fn finite_nonnegative_bits_or_no_signal(value: Option<f64>) -> u64 {
    value
        .filter(|v| v.is_finite() && *v >= 0.0)
        .map(f64::to_bits)
        .unwrap_or(IFT_RESIDUAL_NO_SIGNAL_BITS)
}

struct TkCorrectionTerms {
    value: f64,
    gradient: Option<Array1<f64>>,
    hessian: Option<Array2<f64>>,
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
        // The Tierney-Kadane fallback gate is no longer needed: the analytic
        // TK value, first ρ-derivative, AND second ρ-derivative paths are
        // implemented in `tierney_kadane_terms`, which now populates the
        // `hessian` field whenever the caller requests `ValueGradientHessian`.
        // The earlier gate (Firth + non-identity link → return false) was
        // kept during the manual conflict merge that landed the TK Hessian
        // implementation; it is now stale and was suppressing the analytic
        // path that was actually in place.
        //
        // Biobank-scale fallback: analytic outer Hessian is only safe when
        // the unified evaluator can express it as a matrix-free Hv operator
        // (`prefer_outer_hessian_operator`). Whenever that path is
        // unavailable at biobank scale, the dense `O(K²·n·p²)` LAML pairwise
        // assembly would run instead — route to BFGS.
        let n_obs = self.x.nrows();
        let p_dim = self.x.ncols();
        let k_outer = self.canonical_penalties.len();
        let operator_path_available =
            super::unified::prefer_outer_hessian_operator(n_obs, p_dim, k_outer);
        if n_obs > 50_000 && !operator_path_available {
            log::info!(
                "[standard-GAM] declining analytic outer Hessian for \
                 n={n_obs} p={p_dim} k={k_outer} (matrix-free operator \
                 path unavailable, dense LAML pairwise assembly is \
                 O(k²·n·p²)); routing to BFGS"
            );
            return false;
        }
        true
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
                let tmp = crate::faer_ndarray::fast_ab(qs, matrix);
                crate::faer_ndarray::fast_abt(&tmp, qs)
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
        let logdet_s_start = std::time::Instant::now();
        let lambdas = rho.mapv(f64::exp);
        let ridge = ridge_passport.penalty_logdet_ridge();
        let kron_logdet = self
            .kronecker_penalty_system
            .as_ref()
            .filter(|kron| self.kronecker_factored.is_some() && kron.num_penalties() == rho.len())
            .map(|kron| kron.logdet_rank_and_derivatives(lambdas.as_slice().unwrap(), ridge));
        let (penalty_rank, log_det_s) = if let Some((logdet, rank, _, _)) = kron_logdet.as_ref() {
            (*rank, *logdet)
        } else {
            self.fixed_subspace_penalty_rank_and_logdet(e_for_logdet, ridge_passport)?
        };
        log::info!(
            "[STAGE] logdet S rho_dim={} penalty_rank={} elapsed={:.3}s",
            rho.len(),
            penalty_rank,
            logdet_s_start.elapsed().as_secs_f64(),
        );

        // Use block-local path from canonical penalties (basis-invariant logdet).
        // The penalty logdet log|Σ λ_k S_k|₊ is invariant under orthogonal
        // transformation, so the original-basis canonical penalties give the
        // same result as transformed-basis roots.
        let (det1, det2_full) = if let Some((_, _, det1, det2)) = kron_logdet {
            (det1, det2)
        } else if !self.canonical_penalties.is_empty()
            && self.canonical_penalties.len() == rho.len()
        {
            self.structural_penalty_logdet_derivatives_block_local(&lambdas, ridge)?
        } else if !penalty_roots.is_empty() {
            self.structural_penalty_logdet_derivatives(penalty_roots, &lambdas, ridge)?
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
        use rayon::prelude::*;
        let h_diag_vec: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|i| {
                let val = x_dense.row(i).dot(&z.column(i));
                if !val.is_finite() {
                    return Err(EstimationError::InvalidInput(format!(
                        "{context} produced non-finite leverage at row {i}: {val}"
                    )));
                }
                Ok(val)
            })
            .collect::<Result<_, _>>()?;
        let h_diag = Array1::from(h_diag_vec);
        let m_vec = c_array * &h_diag;
        let x_m = crate::faer_ndarray::fast_atv(x_dense, &m_vec);
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

    fn tk_fill_gram_block_entries_scalar(
        x_dense: &Array2<f64>,
        z: &Array2<f64>,
        i_block: &TkActiveBlock,
        j_block: &TkActiveBlock,
        gram: &mut Array2<f64>,
    ) {
        let p = x_dense.ncols();
        for &(bi, _) in &i_block.entries {
            let ii = i_block.start + bi;
            for &(bj, _) in &j_block.entries {
                let jj = j_block.start + bj;
                gram[[bi, bj]] = (0..p)
                    .map(|col| x_dense[[ii, col]] * z[[col, jj]])
                    .sum::<f64>();
            }
        }
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
        let x_y = crate::faer_ndarray::fast_av(x_dense, &shared.y);

        let mut diag_combined = Array1::<f64>::zeros(n);
        ndarray::Zip::from(&mut diag_combined)
            .and(d_array)
            .and(&shared.h_diag)
            .and(c_array)
            .and(&x_y)
            .par_for_each(|o, &d, &h, &c, &xy| *o = d * h - c * xy);
        let chunk_len = (n / (rayon::current_num_threads().saturating_mul(4).max(1)))
            .clamp(TK_BLOCK_SIZE, 2048);
        let diag_chunks: Vec<usize> = (0..n).step_by(chunk_len).collect();
        let mut p_total = diag_chunks
            .par_iter()
            .fold(
                || Array2::<f64>::zeros((p, p)),
                |mut local, &i0| {
                    let i1 = (i0 + chunk_len).min(n);
                    for i in i0..i1 {
                        let wi = diag_combined[i];
                        if wi == 0.0 {
                            continue;
                        }
                        for a in 0..p {
                            let wa = wi * z[[a, i]];
                            for b in a..p {
                                let val = wa * z[[b, i]];
                                local[[a, b]] += val;
                                if a != b {
                                    local[[b, a]] += val;
                                }
                            }
                        }
                    }
                    local
                },
            )
            .reduce(
                || Array2::<f64>::zeros((p, p)),
                |mut left, right| {
                    left += &right;
                    left
                },
            );
        p_total.mapv_inplace(|v| 0.25 * v);
        for a in 0..p {
            for b in 0..p {
                p_total[[a, b]] -= 0.125 * shared.y[a] * shared.y[b];
            }
        }

        let active_pairs: Vec<(usize, usize)> = (0..shared.active_blocks.len())
            .flat_map(|j_block_idx| {
                (0..=j_block_idx).map(move |i_block_idx| (i_block_idx, j_block_idx))
            })
            .collect();
        let active_total = active_pairs
            .par_iter()
            .fold(
                || Array2::<f64>::zeros((p, p)),
                |mut local, &(i_block_idx, j_block_idx)| {
                    let i_block = &shared.active_blocks[i_block_idx];
                    let j_block = &shared.active_blocks[j_block_idx];
                    let sym_factor = if i_block_idx == j_block_idx { 1.0 } else { 2.0 };
                    for &(bi, ci) in &i_block.entries {
                        let ii = i_block.start + bi;
                        for &(bj, cj) in &j_block.entries {
                            let jj = j_block.start + bj;
                            let gij = (0..p)
                                .map(|col| x_dense[[ii, col]] * z[[col, jj]])
                                .sum::<f64>();
                            let weight = ci * cj * gij * gij;
                            let scale = -0.25 * weight * sym_factor;
                            if ii == jj {
                                for a in 0..p {
                                    let za = z[[a, ii]];
                                    for b in a..p {
                                        let val = scale * za * z[[b, ii]];
                                        local[[a, b]] += val;
                                        if a != b {
                                            local[[b, a]] += val;
                                        }
                                    }
                                }
                            } else {
                                let half_scale = 0.5 * scale;
                                for a in 0..p {
                                    let z_ii_a = z[[a, ii]];
                                    let z_jj_a = z[[a, jj]];
                                    for b in 0..p {
                                        local[[a, b]] += half_scale
                                            * (z_ii_a * z[[b, jj]] + z_jj_a * z[[b, ii]]);
                                    }
                                }
                            }
                        }
                    }
                    local
                },
            )
            .reduce(
                || Array2::<f64>::zeros((p, p)),
                |mut left, right| {
                    left += &right;
                    left
                },
            );
        p_total += &active_total;

        let xp = crate::faer_ndarray::fast_ab(x_dense, &p_total);
        let mut lev_p = Array1::<f64>::zeros(n);
        ndarray::Zip::from(&mut lev_p)
            .and(xp.rows())
            .and(x_dense.rows())
            .par_for_each(|o, xp_row, x_row| *o = xp_row.dot(&x_row));

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
                x_dense, z, c_array, d_array, e_array, &eta_total, None, shared, gram, true,
            )?;
            gradient[idx] = trace_ak_p - correction_trace + firth_trace + direct;
        }
        let ext_values = (0..ext_drifts.len())
            .into_par_iter()
            .map(|extra_idx| -> Result<(usize, f64), EstimationError> {
                let drift = &ext_drifts[extra_idx];
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
                let mut eta_total = x_vks[x_vk_idx].mapv(|value| -value);
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
                let mut local_gram = Array2::<f64>::zeros((TK_BLOCK_SIZE, TK_BLOCK_SIZE));
                let direct = Self::tk_direct_gradient_from_cd_and_design(
                    x_dense,
                    z,
                    c_array,
                    d_array,
                    e_array,
                    &eta_total,
                    x_fixed,
                    shared,
                    &mut local_gram,
                    false,
                )?;
                Ok((x_vk_idx, trace_ak_p - correction_trace + firth_trace + direct))
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (idx, value) in ext_values {
            gradient[idx] = value;
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
        let deta = crate::faer_ndarray::fast_av(&firth_op.x_dense, beta_dir);
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
        use_dense_kernels: bool,
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
            .par_for_each(|out, &d, &eta| *out = d * eta);
        ndarray::Zip::from(&mut d_prime)
            .and(e_array)
            .and(eta_total)
            .par_for_each(|out, &e, &eta| *out = e * eta);

        let mut h_prime = Array1::<f64>::zeros(n);
        let mut design_q_prime = Array1::<f64>::zeros(x_dense.ncols());
        let has_design_deriv = x_fixed.is_some();
        if let Some(x_theta) = x_fixed {
            let ch = c_array * &shared.h_diag;
            design_q_prime += &crate::faer_ndarray::fast_atv(x_theta, &ch);
            ndarray::Zip::from(&mut h_prime)
                .and(x_theta.rows())
                .and(z.columns())
                .par_for_each(|o, xr, zc| *o = 2.0 * xr.dot(&zc));
        }

        let q_weight_prime = &(&c_prime * &shared.h_diag) + &(c_array * &h_prime);
        let q_prime = crate::faer_ndarray::fast_atv(x_dense, &q_weight_prime) + design_q_prime;
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
                if use_dense_kernels {
                    Self::tk_fill_gram_block(x_dense, z, i0, i1, j0, j1, gram);
                } else {
                    Self::tk_fill_gram_block_entries_scalar(x_dense, z, i_block, j_block, gram);
                }

                let design_gram = if has_design_deriv && use_dense_kernels {
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
                        } else if let Some(x_theta) = x_fixed {
                            let kp = (0..x_dense.ncols())
                                .map(|col| {
                                    x_theta[[ii, col]] * z[[col, jj]]
                                        + x_theta[[jj, col]] * z[[col, ii]]
                                })
                                .sum::<f64>();
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

    fn tk_penalty_dense(
        cp: &crate::construction::CanonicalPenalty,
        lambda: f64,
        p: usize,
    ) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((p, p));
        let r = &cp.col_range;
        for a in 0..cp.block_dim() {
            for b in 0..cp.block_dim() {
                let mut val = 0.0;
                for row in 0..cp.rank() {
                    val += cp.root[[row, a]] * cp.root[[row, b]];
                }
                out[[r.start + a, r.start + b]] = lambda * val;
            }
        }
        out
    }

    fn tk_xt_diag_x(x_dense: &Array2<f64>, diag: &Array1<f64>) -> Array2<f64> {
        let mut weighted = Array2::<f64>::zeros(x_dense.raw_dim());
        Self::xt_diag_x_dense_into(x_dense, diag, &mut weighted)
    }

    fn tk_hessian_rho_canonical_logit(
        x_dense: &Array2<f64>,
        c_array: &Array1<f64>,
        d_array: &Array1<f64>,
        e_array: &Array1<f64>,
        f_array: &Array1<f64>,
        tk_penalties: &[crate::construction::CanonicalPenalty],
        lambdas: &[f64],
        beta: &Array1<f64>,
        firth_op: Option<&super::FirthDenseOperator>,
        h_inv_solve: &dyn Fn(&Array1<f64>) -> Result<Array1<f64>, EstimationError>,
    ) -> Result<Array2<f64>, EstimationError> {
        let n = x_dense.nrows();
        let p = x_dense.ncols();
        let k = tk_penalties.len();
        if k == 0 {
            return Ok(Array2::zeros((0, 0)));
        }
        if c_array.len() != n || d_array.len() != n || e_array.len() != n || f_array.len() != n {
            return Err(EstimationError::InvalidInput(
                "Tierney-Kadane Hessian derivative arrays have inconsistent lengths".to_string(),
            ));
        }

        let mut k_mat = Array2::<f64>::zeros((p, p));
        for col in 0..p {
            let mut rhs = Array1::<f64>::zeros(p);
            rhs[col] = 1.0;
            let sol = h_inv_solve(&rhs)?;
            k_mat.column_mut(col).assign(&sol);
        }
        enforce_symmetry(&mut k_mat);

        let mut a_mats = Vec::with_capacity(k);
        let mut v = Vec::with_capacity(k);
        let mut eta_i = Vec::with_capacity(k);
        for idx in 0..k {
            let a = Self::tk_penalty_dense(&tk_penalties[idx], lambdas[idx], p);
            let rhs = a.dot(beta);
            let vi = h_inv_solve(&rhs)?;
            let bi = vi.mapv(|value| -value);
            let ei = crate::faer_ndarray::fast_av(x_dense, &bi);
            a_mats.push(a);
            v.push(vi);
            eta_i.push(ei);
        }

        let mut h_i = Vec::with_capacity(k);
        for idx in 0..k {
            let diag = c_array * &eta_i[idx];
            let mut h = &a_mats[idx] + &Self::tk_xt_diag_x(x_dense, &diag);
            if let Some(op) = firth_op {
                let dir = op.direction_from_deta(eta_i[idx].clone());
                h -= &op.hphi_direction(&dir);
            }
            enforce_symmetry(&mut h);
            h_i.push(h);
        }

        let mut beta_ij: Vec<Vec<Array1<f64>>> = (0..k)
            .map(|_| (0..k).map(|_| Array1::<f64>::zeros(p)).collect())
            .collect();
        let mut h_ij: Vec<Vec<Array2<f64>>> = (0..k)
            .map(|_| (0..k).map(|_| Array2::<f64>::zeros((p, p))).collect())
            .collect();
        for i in 0..k {
            for j in 0..=i {
                let mut rhs = h_i[j].dot(&v[i]);
                rhs += &a_mats[i].dot(&v[j]);
                if i == j {
                    rhs -= &a_mats[i].dot(beta);
                }
                let bij = h_inv_solve(&rhs)?;
                beta_ij[i][j] = bij.clone();
                beta_ij[j][i] = bij;
            }
        }
        for i in 0..k {
            for j in 0..=i {
                let eta_ij = crate::faer_ndarray::fast_av(x_dense, &beta_ij[i][j]);
                let diag = c_array * &eta_ij + &(d_array * &(&eta_i[i] * &eta_i[j]));
                let mut h = Self::tk_xt_diag_x(x_dense, &diag);
                if i == j {
                    h += &a_mats[i];
                }
                if let Some(op) = firth_op {
                    let dir_ij = op.direction_from_deta(eta_ij);
                    h -= &op.hphi_direction(&dir_ij);
                    let dir_i = op.direction_from_deta(eta_i[i].clone());
                    let dir_j = op.direction_from_deta(eta_i[j].clone());
                    let eye = Array2::<f64>::eye(p);
                    h -= &op.hphisecond_direction_apply(&dir_i, &dir_j, &eye);
                }
                enforce_symmetry(&mut h);
                h_ij[i][j] = h.clone();
                h_ij[j][i] = h;
            }
        }

        let mut k_i = Vec::with_capacity(k);
        for i in 0..k {
            k_i.push(-k_mat.dot(&h_i[i]).dot(&k_mat));
        }
        let mut k_ij: Vec<Vec<Array2<f64>>> = (0..k)
            .map(|_| (0..k).map(|_| Array2::<f64>::zeros((p, p))).collect())
            .collect();
        for i in 0..k {
            for j in 0..=i {
                let kij = k_mat.dot(&h_i[j]).dot(&k_mat).dot(&h_i[i]).dot(&k_mat)
                    + k_mat.dot(&h_i[i]).dot(&k_mat).dot(&h_i[j]).dot(&k_mat)
                    - k_mat.dot(&h_ij[i][j]).dot(&k_mat);
                k_ij[i][j] = kij.clone();
                k_ij[j][i] = kij;
            }
        }

        #[derive(Clone)]
        struct Jet {
            v: f64,
            g: Array1<f64>,
            h: Array2<f64>,
        }
        impl Jet {
            fn constant(v: f64, k: usize) -> Self {
                Self {
                    v,
                    g: Array1::zeros(k),
                    h: Array2::zeros((k, k)),
                }
            }
            fn add(&self, other: &Self) -> Self {
                Self {
                    v: self.v + other.v,
                    g: &self.g + &other.g,
                    h: &self.h + &other.h,
                }
            }
            fn scale(&self, a: f64) -> Self {
                Self {
                    v: self.v * a,
                    g: self.g.mapv(|x| x * a),
                    h: self.h.mapv(|x| x * a),
                }
            }
            fn mul(&self, other: &Self) -> Self {
                let k = self.g.len();
                let mut h = &self.h * other.v + &other.h * self.v;
                for i in 0..k {
                    for j in 0..k {
                        h[[i, j]] += self.g[i] * other.g[j] + other.g[i] * self.g[j];
                    }
                }
                Self {
                    v: self.v * other.v,
                    g: &self.g * other.v + &other.g * self.v,
                    h,
                }
            }
            fn square(&self) -> Self {
                self.mul(self)
            }
            fn cube(&self) -> Self {
                self.mul(self).mul(self)
            }
        }

        let mut hdiag = Vec::with_capacity(n);
        for row in 0..n {
            let x = x_dense.row(row);
            let mut jet = Jet::constant(x.dot(&k_mat.dot(&x.to_owned())), k);
            for a in 0..k {
                jet.g[a] = x.dot(&k_i[a].dot(&x.to_owned()));
            }
            for a in 0..k {
                for b in 0..k {
                    jet.h[[a, b]] = x.dot(&k_ij[a][b].dot(&x.to_owned()));
                }
            }
            hdiag.push(jet);
        }
        let mut cjet = Vec::with_capacity(n);
        let mut djet = Vec::with_capacity(n);
        for row in 0..n {
            let mut eta = Jet::constant(0.0, k);
            for a in 0..k {
                eta.g[a] = eta_i[a][row];
                for b in 0..k {
                    eta.h[[a, b]] = x_dense.row(row).dot(&beta_ij[a][b]);
                }
            }
            let mut c = Jet::constant(c_array[row], k);
            let mut d = Jet::constant(d_array[row], k);
            for a in 0..k {
                c.g[a] = d_array[row] * eta.g[a];
                d.g[a] = e_array[row] * eta.g[a];
                for b in 0..k {
                    c.h[[a, b]] = d_array[row] * eta.h[[a, b]] + e_array[row] * eta.g[a] * eta.g[b];
                    d.h[[a, b]] = e_array[row] * eta.h[[a, b]] + f_array[row] * eta.g[a] * eta.g[b];
                }
            }
            cjet.push(c);
            djet.push(d);
        }

        let mut total = Jet::constant(0.0, k);
        for row in 0..n {
            total = total.add(&djet[row].mul(&hdiag[row].square()).scale(-0.125));
        }
        for irow in 0..n {
            let xi = x_dense.row(irow).to_owned();
            for jrow in 0..n {
                let xj = x_dense.row(jrow).to_owned();
                let mut kg = Jet::constant(xi.dot(&k_mat.dot(&xj)), k);
                for a in 0..k {
                    kg.g[a] = xi.dot(&k_i[a].dot(&xj));
                }
                for a in 0..k {
                    for b in 0..k {
                        kg.h[[a, b]] = xi.dot(&k_ij[a][b].dot(&xj));
                    }
                }
                let term = cjet[irow]
                    .mul(&cjet[jrow])
                    .mul(&kg.cube())
                    .scale(1.0 / 12.0);
                total = total.add(&term);
            }
        }
        let mut qjets: Vec<Jet> = (0..p).map(|_| Jet::constant(0.0, k)).collect();
        for row in 0..n {
            let wh = cjet[row].mul(&hdiag[row]);
            for col in 0..p {
                qjets[col] = qjets[col].add(&wh.scale(x_dense[[row, col]]));
            }
        }
        for a in 0..p {
            for b in 0..p {
                let mut kj = Jet::constant(k_mat[[a, b]], k);
                for i in 0..k {
                    kj.g[i] = k_i[i][[a, b]];
                }
                for i in 0..k {
                    for j in 0..k {
                        kj.h[[i, j]] = k_ij[i][j][[a, b]];
                    }
                }
                total = total.add(&qjets[a].mul(&kj).mul(&qjets[b]).scale(0.125));
            }
        }
        if total.h.iter().any(|v| !v.is_finite()) {
            return Err(EstimationError::InvalidInput(
                "Tierney-Kadane analytic Hessian produced a non-finite entry".to_string(),
            ));
        }
        Ok(total.h)
    }

    fn tierney_kadane_analytic_core(
        &self,
        x_dense: &Array2<f64>,
        z: &Array2<f64>,
        c_array: &Array1<f64>,
        d_array: &Array1<f64>,
        e_array: &Array1<f64>,
        f_array: &Array1<f64>,
        tk_penalties: &[crate::construction::CanonicalPenalty],
        lambdas: &[f64],
        ext_coords: &[super::unified::HyperCoord],
        beta: &Array1<f64>,
        firth_op: Option<&super::FirthDenseOperator>,
        compute_gradient: bool,
        compute_hessian: bool,
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
                hessian: None,
            });
        }

        let mut x_vks: Vec<Array1<f64>> = Vec::with_capacity(k + ext_coords.len());
        let mut beta_dirs: Vec<Array1<f64>> = Vec::with_capacity(k + ext_coords.len());
        for idx in 0..k {
            let cp = &tk_penalties[idx];
            let r = &cp.col_range;
            let beta_block = beta.slice(s![r.start..r.end]);
            let centered = &beta_block - &cp.prior_mean;
            let r_beta = cp.root.dot(&centered);
            let mut s_k_beta = Array1::<f64>::zeros(p);
            for a in 0..cp.block_dim() {
                s_k_beta[r.start + a] = (0..cp.rank())
                    .map(|row| cp.root[[row, a]] * r_beta[row])
                    .sum::<f64>();
            }
            let a_k_beta = &s_k_beta * lambdas[idx];
            let v_k = h_inv_solve(&a_k_beta)?;
            x_vks.push(crate::faer_ndarray::fast_av(x_dense, &v_k));
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
            x_vks.push(crate::faer_ndarray::fast_av(x_dense, &beta_theta));
            beta_dirs.push(beta_theta.mapv(|value| -value));
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
        let hessian = if compute_hessian {
            Some(Self::tk_hessian_rho_canonical_logit(
                x_dense,
                c_array,
                d_array,
                e_array,
                f_array,
                tk_penalties,
                lambdas,
                beta,
                firth_op,
                h_inv_solve,
            )?)
        } else {
            None
        };
        Ok(TkCorrectionTerms {
            value,
            gradient: Some(gradient),
            hessian,
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
                hessian: None,
            });
        }
        if !self.config.firth_bias_reduction {
            return Ok(TkCorrectionTerms {
                value: 0.0,
                gradient: None,
                hessian: None,
            });
        }

        let pirls_result = bundle.pirls_result.as_ref();
        let (c_array, d_array, e_array, f_array) = self.hessian_cdef_arrays(pirls_result)?;
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
        if c_array.is_empty() || d_array.is_empty() {
            return Ok(TkCorrectionTerms {
                value: 0.0,
                gradient: if compute_gradient {
                    Some(Array1::zeros(rho.len() + ext_coords.len()))
                } else {
                    None
                },
                hessian: if mode == super::unified::EvalMode::ValueGradientHessian {
                    Some(Array2::zeros((
                        rho.len() + ext_coords.len(),
                        rho.len() + ext_coords.len(),
                    )))
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
                &f_array,
                &self.canonical_penalties,
                &lambdas,
                ext_coords,
                &beta,
                firth_op.as_deref(),
                compute_gradient,
                mode == super::unified::EvalMode::ValueGradientHessian,
                &h_inv_solve,
            );
        }

        let free_basis_opt = self.active_constraint_free_basis(pirls_result);
        let use_original_basis = matches!(
            pirls_result.coordinate_frame,
            pirls::PirlsCoordinateFrame::TransformedQs
        ) && free_basis_opt.is_none();
        // `firth_bias_reduction == true` here: the early return above bails
        // out otherwise, so h_total is the only TK-relevant operator.
        let h_tk_source = bundle.h_total.as_ref();
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
            &f_array,
            &tk_penalties,
            &lambdas,
            ext_coords,
            &beta,
            firth_op.as_deref(),
            compute_gradient,
            mode == super::unified::EvalMode::ValueGradientHessian,
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
        if let Some(tk_hess) = tk_terms.hessian.as_ref() {
            result
                .hessian
                .add_rho_block_dense(tk_hess)
                .map_err(EstimationError::InvalidInput)?;
        }
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
        // Under a link change the previous link's working-weight
        // curvature differs from the new link's, so the previous
        // solve's β / H_pen / Cholesky factor / damping / iter-count
        // / IFT residual / tangent-line history are all calibrated to
        // the wrong geometry. Wipe in lockstep — see
        // `clear_warm_start_predictor_state` and
        // `clear_warm_start_adaptive_signals` for the per-slot
        // staleness arguments.
        self.clear_warm_start_predictor_state();
        self.clear_warm_start_adaptive_signals();
    }

    /// Wipe every RwLock-guarded slot used by the warm-start
    /// predictors (tangent-line and IFT). Called on link change, on
    /// surface reset, on outer-seed restart, on failed PIRLS solve,
    /// and any other event where the cached `(β, ρ, H_pen, qs,
    /// factor)` is no longer calibrated to the geometry the next
    /// solve will see.
    ///
    /// Single source of truth: any new warm-start RwLock added to
    /// `RemlObjectiveState` MUST be wiped here, otherwise the
    /// warm-start machinery can leak stale state across the
    /// invalidation boundary and produce β-drift regressions
    /// (caught by `tests/warm_start_quality_regression.rs`).
    fn clear_warm_start_predictor_state(&self) {
        self.warm_start_beta.write().unwrap().take();
        self.warm_start_rho.write().unwrap().take();
        self.prev_warm_start_beta.write().unwrap().take();
        self.prev_warm_start_rho.write().unwrap().take();
        self.ift_warm_start_cache.write().unwrap().take();
        self.ift_cached_factor.write().unwrap().take();
    }

    /// Wipe every atomic-bit-packed signal used by the adaptive
    /// policies (inner-cap schedule margin, IFT |Δρ| cap, LM-λ
    /// hint clamp). Called from the same invalidation paths as
    /// `clear_warm_start_predictor_state`, plus from
    /// `execute_pirls_if_needed`'s failure branch.
    ///
    /// Single source of truth: any new adaptive-policy atomic added
    /// to `RemlObjectiveState` MUST be wiped here. The β-drift
    /// regression tests will catch correctness bugs from a stale
    /// β / H, but stale POLICY signals only surface as performance
    /// regressions (the inner solve takes longer than it should),
    /// which the bench runner's [PHASE summary] line catches via
    /// the `pirls_conv_*` and `ift_*` fields.
    fn clear_warm_start_adaptive_signals(&self) {
        self.last_inner_iters.store(0, Ordering::Relaxed);
        self.last_inner_converged.store(false, Ordering::Relaxed);
        self.last_pirls_lm_lambda.store(0, Ordering::Relaxed);
        // Use the NaN sentinel (not literal 0) so a residual of exactly
        // 0.0 — possible if β_predicted matched β_converged element-wise
        // — is not confused with "no signal yet".
        self.last_ift_prediction_residual
            .store(IFT_RESIDUAL_NO_SIGNAL_BITS, Ordering::Relaxed);
        // Same NaN-sentinel discipline as last_ift_prediction_residual:
        // a recorded gain ratio of exactly 0 (degenerate but possible
        // for noise-floor steps) must not collide with "no signal yet".
        self.last_pirls_accept_rho
            .store(IFT_RESIDUAL_NO_SIGNAL_BITS, Ordering::Relaxed);
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
        *self.persistent_warm_start_key.write().unwrap() = None;
        self.persistent_warm_start_loaded
            .store(false, Ordering::Relaxed);
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
    /// and the variance jet V, V₁..V₄. The production path is fully
    /// analytic.
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
            // Canonical Logit fast path: per-row 5-jet evaluation, no
            // cross-row dependency. At biobank n the 5-jet exp/log work
            // is the dominant cost.
            use rayon::prelude::*;
            let final_eta = &pirls_result.final_eta;
            let weights = &self.weights;
            let e_s = e_array.as_slice_mut().expect("e_array must be contiguous");
            e_s.par_iter_mut().enumerate().for_each(|(i, e_o)| {
                let eta_raw = final_eta[i];
                let eta_used = eta_raw.clamp(clamp_lo, clamp_hi);
                if eta_raw != eta_used {
                    *e_o = 0.0;
                } else {
                    let jet = crate::mixture_link::logit_inverse_link_jet5(eta_used);
                    *e_o = weights[i].max(0.0) * jet.d4;
                }
            });
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
        // Noncanonical / GammaLog observed-information path: each row's
        // e_i depends only on (eta[i], mu[i], priorweights[i], y[i], dmu/d2/d3
        // jets at row i, and the inverse-link's higher-order pdf derivatives
        // evaluated at eta_used). No carrier across rows. The h4/h5 lookup
        // calls return Result, so we propagate first error via try_for_each.
        use rayon::prelude::*;
        let final_eta = &pirls_result.final_eta;
        let weights = &self.weights;
        let y_view = &self.y;
        let inverse_link_ref = &inverse_link;
        let e_s = e_array.as_slice_mut().expect("e_array must be contiguous");
        e_s.par_iter_mut()
            .enumerate()
            .try_for_each(|(i, e_o)| -> Result<(), EstimationError> {
                let eta_raw = final_eta[i];
                let eta_used = eta_raw.clamp(clamp_lo, clamp_hi);
                if eta_raw != eta_used {
                    *e_o = 0.0;
                    return Ok(());
                }
                let h1 = dmu_deta[i];
                let h2 = d2mu_deta2[i];
                let h3 = d3mu_deta3[i];
                let h4 = crate::mixture_link::inverse_link_pdfthird_derivative_for_inverse_link(
                    inverse_link_ref,
                    eta_used,
                )?;
                let h5 = crate::mixture_link::inverse_link_pdffourth_derivative_for_inverse_link(
                    inverse_link_ref,
                    eta_used,
                )?;
                if !h1.is_finite()
                    || !h2.is_finite()
                    || !h3.is_finite()
                    || !h4.is_finite()
                    || !h5.is_finite()
                {
                    *e_o = 0.0;
                    return Ok(());
                }
                let mu_i = mu[i];
                let vj = pirls::variance_jet_for_weight_family(weight_family, mu_i);
                if !(vj.v.is_finite() && vj.v > 0.0) {
                    *e_o = 0.0;
                    return Ok(());
                }
                let pw = weights[i].max(0.0);
                let y_i = y_view[i];
                let e_i = pirls::e_obs_from_jets(y_i, mu_i, h1, h2, h3, h4, h5, vj, phi, pw);
                *e_o = if e_i.is_finite() { e_i } else { 0.0 };
                Ok(())
            })?;

        Ok((c_array, d_array, e_array))
    }

    fn hessian_cdef_arrays(
        &self,
        pirls_result: &PirlsResult,
    ) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>), EstimationError> {
        let (c_array, d_array, e_array) = self.hessian_cde_arrays(pirls_result)?;
        let link_function = self.config.link_function();
        let canonical_logit = matches!(link_function, LinkFunction::Logit)
            && self.runtime_mixture_link_state.is_none();
        if !canonical_logit {
            return Err(EstimationError::InvalidInput(
                "Tierney-Kadane outer Hessian is implemented for canonical Binomial Logit Firth fits only".to_string(),
            ));
        }
        let mut f_array = Array1::<f64>::zeros(e_array.len());
        use rayon::prelude::*;
        let final_eta = &pirls_result.final_eta;
        let weights = &self.weights;
        let f_s = f_array.as_slice_mut().expect("f_array must be contiguous");
        f_s.par_iter_mut().enumerate().for_each(|(i, f_o)| {
            let eta_raw = final_eta[i];
            let eta_used = eta_raw.clamp(-700.0_f64, 700.0_f64);
            if eta_raw != eta_used {
                *f_o = 0.0;
            } else {
                let jet = crate::mixture_link::logit_inverse_link_jet5(eta_used);
                *f_o = weights[i].max(0.0) * jet.d5;
            }
        });
        Ok((c_array, d_array, e_array, f_array))
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

    // Gamma(a, b) precision hyperprior identity.
    //
    // For one penalty block, lambda = exp(rho) and
    // p(lambda) proportional to lambda^(a-1) exp(-b lambda). REML/LAML
    // smoothing-parameter optimization targets the mode in lambda, expressed
    // on the rho coordinate for numerical stability. Therefore no
    // change-of-variables Jacobian is added here; the hyperprior contributes
    //
    //     -log p(exp(rho)) = b exp(rho) - (a - 1) rho + constant.
    //
    // The likelihood/penalty part already contributes the usual
    // nu_p/2 log(lambda) - lambda (beta-mu_p)' S_p (beta-mu_p)/2
    // terms through the assembled
    // penalty coordinates. Adding the term below gives the conditional MAP
    // score
    //
    //     d/dlambda: (a - 1 + nu_p/2)/lambda
    //         - (b + (beta-mu_p)' S_p (beta-mu_p)/2) = 0,
    //
    // hence lambda* = (a - 1 + nu_p/2)
    //     / (b + (beta-mu_p)' S_p (beta-mu_p)/2), reducing to the existing
    // MacKay/Tipping fixed point when (a, b) = (1, 0).
    fn compute_configured_rho_prior_cost(&self, rho: &Array1<f64>) -> f64 {
        match &self.rho_prior {
            RhoPrior::Flat => 0.0,
            RhoPrior::Normal { mean, sd } => {
                if *sd <= 0.0 || !sd.is_finite() || !mean.is_finite() {
                    return f64::INFINITY;
                }
                let inv_var = 1.0 / (*sd * *sd);
                0.5 * inv_var
                    * rho
                        .iter()
                        .map(|&r| {
                            let d = r - *mean;
                            d * d
                        })
                        .sum::<f64>()
            }
            RhoPrior::GammaPrecision { shape, rate } => {
                if *shape <= 0.0 || *rate < 0.0 || !shape.is_finite() || !rate.is_finite() {
                    return f64::INFINITY;
                }
                rho.iter()
                    .map(|&r| *rate * r.exp() - (*shape - 1.0) * r)
                    .sum()
            }
            RhoPrior::Independent(priors) => {
                if priors.len() != rho.len() {
                    return f64::INFINITY;
                }
                priors
                    .iter()
                    .zip(rho.iter())
                    .map(|(prior, &r)| match prior {
                        RhoPrior::Flat => 0.0,
                        RhoPrior::Normal { mean, sd } => {
                            if *sd <= 0.0 || !sd.is_finite() || !mean.is_finite() {
                                f64::INFINITY
                            } else {
                                let d = r - *mean;
                                0.5 * d * d / (*sd * *sd)
                            }
                        }
                        RhoPrior::GammaPrecision { shape, rate } => {
                            if *shape <= 0.0
                                || *rate < 0.0
                                || !shape.is_finite()
                                || !rate.is_finite()
                            {
                                f64::INFINITY
                            } else {
                                *rate * r.exp() - (*shape - 1.0) * r
                            }
                        }
                        RhoPrior::Independent(_) => f64::INFINITY,
                    })
                    .sum()
            }
        }
    }

    fn compute_configured_rho_prior_grad(&self, rho: &Array1<f64>) -> Array1<f64> {
        match &self.rho_prior {
            RhoPrior::Flat => Array1::zeros(rho.len()),
            RhoPrior::Normal { mean, sd } => {
                if *sd <= 0.0 || !sd.is_finite() || !mean.is_finite() {
                    return Array1::from_elem(rho.len(), f64::NAN);
                }
                let inv_var = 1.0 / (*sd * *sd);
                rho.mapv(|r| (r - *mean) * inv_var)
            }
            RhoPrior::GammaPrecision { shape, rate } => {
                if *shape <= 0.0 || *rate < 0.0 || !shape.is_finite() || !rate.is_finite() {
                    return Array1::from_elem(rho.len(), f64::NAN);
                }
                rho.mapv(|r| *rate * r.exp() - (*shape - 1.0))
            }
            RhoPrior::Independent(priors) => {
                if priors.len() != rho.len() {
                    return Array1::from_elem(rho.len(), f64::NAN);
                }
                let mut grad = Array1::<f64>::zeros(rho.len());
                for (i, (prior, &r)) in priors.iter().zip(rho.iter()).enumerate() {
                    grad[i] = match prior {
                        RhoPrior::Flat => 0.0,
                        RhoPrior::Normal { mean, sd } => {
                            if *sd <= 0.0 || !sd.is_finite() || !mean.is_finite() {
                                f64::NAN
                            } else {
                                (r - *mean) / (*sd * *sd)
                            }
                        }
                        RhoPrior::GammaPrecision { shape, rate } => {
                            if *shape <= 0.0
                                || *rate < 0.0
                                || !shape.is_finite()
                                || !rate.is_finite()
                            {
                                f64::NAN
                            } else {
                                *rate * r.exp() - (*shape - 1.0)
                            }
                        }
                        RhoPrior::Independent(_) => f64::NAN,
                    };
                }
                grad
            }
        }
    }

    fn compute_configured_rho_prior_hess(&self, rho: &Array1<f64>) -> Option<Array2<f64>> {
        match &self.rho_prior {
            RhoPrior::Flat => None,
            RhoPrior::Normal { mean, sd } => {
                if *sd <= 0.0 || !sd.is_finite() || !mean.is_finite() {
                    return Some(Array2::from_elem((rho.len(), rho.len()), f64::NAN));
                }
                let inv_var = 1.0 / (*sd * *sd);
                let mut hess = Array2::<f64>::zeros((rho.len(), rho.len()));
                for i in 0..rho.len() {
                    hess[[i, i]] = inv_var;
                }
                Some(hess)
            }
            RhoPrior::GammaPrecision { shape, rate } => {
                if *shape <= 0.0 || *rate < 0.0 || !shape.is_finite() || !rate.is_finite() {
                    return Some(Array2::from_elem((rho.len(), rho.len()), f64::NAN));
                }
                let mut hess = Array2::<f64>::zeros((rho.len(), rho.len()));
                for i in 0..rho.len() {
                    hess[[i, i]] = *rate * rho[i].exp();
                }
                Some(hess)
            }
            RhoPrior::Independent(priors) => {
                if priors.len() != rho.len() {
                    return Some(Array2::from_elem((rho.len(), rho.len()), f64::NAN));
                }
                let mut hess = Array2::<f64>::zeros((rho.len(), rho.len()));
                let mut any = false;
                for (i, (prior, &r)) in priors.iter().zip(rho.iter()).enumerate() {
                    let value = match prior {
                        RhoPrior::Flat => 0.0,
                        RhoPrior::Normal { mean, sd } => {
                            if *sd <= 0.0 || !sd.is_finite() || !mean.is_finite() {
                                f64::NAN
                            } else {
                                1.0 / (*sd * *sd)
                            }
                        }
                        RhoPrior::GammaPrecision { shape, rate } => {
                            if *shape <= 0.0
                                || *rate < 0.0
                                || !shape.is_finite()
                                || !rate.is_finite()
                            {
                                f64::NAN
                            } else {
                                *rate * r.exp()
                            }
                        }
                        RhoPrior::Independent(_) => f64::NAN,
                    };
                    hess[[i, i]] = value;
                    any |= value != 0.0;
                }
                if any { Some(hess) } else { None }
            }
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
    /// IMPORTANT: W_H here is the **final observed-information** weight for
    /// non-canonical links (or Fisher weight for canonical links, where the two
    /// coincide). PIRLS may rehydrate final Laplace curvature arrays after compacting
    /// the accepted state; reusing `stabilizedhessian_transformed` would then evaluate
    /// log|H_eff| on the accepted-step weights while the hyper-gradient differentiates
    /// the final curvature arrays. Rebuilding from `finalweights` keeps the REML value
    /// and hyper-derivative on the same Hessian surface.
    ///
    pub(super) fn effectivehessian(
        &self,
        pr: &PirlsResult,
    ) -> Result<(Array2<f64>, RidgePassport), EstimationError> {
        // Use the same stabilized H = X' W X + S + δI that PIRLS built,
        // where W = `solver_weights = max(W_obs, floor(W_F))` keeps H PD.
        // The OUTER analytic operator constructs ∂H/∂ψ from the same
        // floored W (via `outer_hessian_curvature_arrays`), so log|H| and
        // its trace gradient live on a single, consistent surface.
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

        // Single-shot structural redundancy diagnostic. Fires exactly once
        // per RemlState construction — see
        // `crate::construction::report_penalty_pair_redundancy` for the
        // logging policy. O(k² · block_dim²); negligible at fit-prep.
        crate::construction::report_penalty_pair_redundancy(canonical_penalties.as_ref());

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
            rho_prior: RhoPrior::Flat,
            cache_manager: EvalCacheManager::new(),
            arena: RemlArena::new(),
            warm_start_beta: RwLock::new(None),
            warm_start_rho: RwLock::new(None),
            prev_warm_start_beta: RwLock::new(None),
            prev_warm_start_rho: RwLock::new(None),
            warm_start_enabled: AtomicBool::new(true),
            screening_max_inner_iterations: Arc::new(AtomicUsize::new(0)),
            outer_inner_cap: Arc::new(AtomicUsize::new(0)),
            last_inner_iters: Arc::new(AtomicUsize::new(0)),
            last_inner_converged: Arc::new(AtomicBool::new(false)),
            ift_warm_start_cache: RwLock::new(None),
            last_pirls_lm_lambda: Arc::new(AtomicU64::new(0)),
            last_ift_prediction_residual: Arc::new(AtomicU64::new(IFT_RESIDUAL_NO_SIGNAL_BITS)),
            last_pirls_accept_rho: Arc::new(AtomicU64::new(IFT_RESIDUAL_NO_SIGNAL_BITS)),
            ift_cached_factor: RwLock::new(None),
            kronecker_penalty_system: None,
            kronecker_factored: None,
            penalty_block_structural_nullities: RwLock::new(None),
            gaussian_fixed_cache: RwLock::new(None),
            persistent_warm_start_key: RwLock::new(None),
            persistent_warm_start_loaded: AtomicBool::new(false),
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
        *self.penalty_block_structural_nullities.write().unwrap() = None;
        // The Gaussian-fixed cache is keyed to (X, y, w, offset); replacing the
        // design invalidates it. The new surface will repopulate it on demand.
        *self.gaussian_fixed_cache.write().unwrap() = None;
        *self.persistent_warm_start_key.write().unwrap() = None;
        self.persistent_warm_start_loaded
            .store(false, Ordering::Relaxed);
        self.cache_manager.clear_eval_and_factor_caches();
        self.cache_manager.pirls_cache.write().unwrap().clear();
        // The new surface has a different design / penalty system /
        // working-weight curvature, so every warm-start signal calibrated
        // to the previous surface is now stale. Wipe in lockstep — see
        // the helpers' doc-comments for the per-slot staleness arguments.
        self.clear_warm_start_predictor_state();
        self.clear_warm_start_adaptive_signals();
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
        *self.persistent_warm_start_key.write().unwrap() = None;
        self.persistent_warm_start_loaded
            .store(false, Ordering::Relaxed);
    }

    pub(crate) fn set_rho_prior(&mut self, prior: RhoPrior) {
        self.rho_prior = prior;
        *self.persistent_warm_start_key.write().unwrap() = None;
        self.persistent_warm_start_loaded
            .store(false, Ordering::Relaxed);
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

    /// Debug-only: return `log|U_Sᵀ H U_S|_+` — the *projected* Hessian
    /// log-determinant on the identified `range(S_+)` subspace, evaluated
    /// at the PIRLS state driven to convergence at this `rho`.
    ///
    /// Production REML/LAML cost reads this quantity via
    /// `hop.logdet() + hessian_logdet_correction` (the latter carries the
    /// `log|H_proj| − hop.logdet()` correction from the dense_assembly
    /// rank-deficiency fix).  Returning the same sum here gives tests a
    /// matching finite-difference reference: centered-differencing the
    /// projected logdet across nearby `theta` values produces the analytic
    /// `d/dψ log|H_proj|` that production's trace formula computes — *not*
    /// the full-space `d/dψ log|H_full|`, which differs by the `null(S)`
    /// directions' contribution and is the wrong finite-difference reference
    /// for the projected LAML cost identity.  Used by
    /// `iso_kappa_duchon_penalty_subspace_projection_pins_trace`'s
    /// INVARIANT 2.
    #[cfg(test)]
    pub(crate) fn objective_logdet_h_proj(
        &self,
        rho: &Array1<f64>,
    ) -> Result<f64, EstimationError> {
        let bundle = self.obtain_eval_bundle(rho)?;
        let assembly =
            self.build_auto_assembly(rho, &bundle, super::unified::EvalMode::ValueOnly, false)?;
        let logdet = super::unified::HessianOperator::logdet(assembly.hessian_op.as_ref())
            + assembly.hessian_logdet_correction;
        Ok(logdet)
    }

    /// Debug-only: return `(final_eta, finalweights, solve_c_array)` at the
    /// PIRLS state produced by driving the solver to convergence at this `rho`.
    /// Used by the iso-κ Duchon FD probe to test whether the analytic
    /// `c · dη/dψ_total` per-row diagonal matches FD `c · (η_+ − η_−) / 2h`.
    #[cfg(test)]
    pub(crate) fn debug_eta_w_c(
        &self,
        rho: &Array1<f64>,
    ) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), EstimationError> {
        let pr = self.execute_pirls_if_needed(rho)?;
        Ok((
            pr.final_eta.clone(),
            pr.finalweights.clone(),
            pr.solve_c_array.clone(),
        ))
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
        let zt_m = crate::faer_ndarray::fast_atb(z, matrix);
        crate::faer_ndarray::fast_ab(&zt_m, z)
    }

    /// Compute the structural penalty rank and exact pseudo-logdet log|S|₊.
    ///
    /// Uses the exact pseudo-logdet on the positive eigenspace:
    ///   L(S) = Σ_{σ_i > ε} log σ_i
    /// where ε is a relative threshold identifying the structural nullspace
    /// directly from the eigenspectrum. No δ-regularization or nullity metadata.
    ///
    /// For S(ρ) = Σ exp(ρ_k) S_k with S_k ⪰ 0, the nullspace N(S) = ∩_k N(S_k)
    /// is structurally fixed, so this is C∞ in ρ. The rank must come from the
    /// same positive eigenspace as the pseudo-logdet; root row count overstates
    /// rank whenever penalty blocks overlap or the root carries redundant rows.
    pub(super) fn fixed_subspace_penalty_rank_and_logdet(
        &self,
        e_transformed: &Array2<f64>,
        ridge_passport: RidgePassport,
    ) -> Result<(usize, f64), EstimationError> {
        if e_transformed.nrows() == 0 || e_transformed.ncols() == 0 {
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

        // Use the STRUCTURAL penalty rank from the canonical-penalty
        // decomposition rather than a numerical-threshold rank assessment.
        //
        // Why: the threshold path (`positive_penalty_rank_and_logdet`) uses
        // `100·p·ε_machine·max_ev` as its cutoff. With the ridge floor
        // added to `s_lambda` above, the null-direction eigenvalues sit at
        // `ridge` (a fixed number) while `max_ev` scales linearly with λ,
        // so `λ × ε_machine` crossing the ridge floor flips the rank
        // assessment between p and the true structural rank as ρ varies.
        // The result was a step change in `penalty_rank`, which in turn
        // toggled the rank-deficient LAML projection in the cost — a
        // discrete cliff that biased the outer optimizer toward
        // over-smoothed local minima (see fuzz seed=92: ~18-unit cost drop
        // between ρ=9 and ρ=12 with no underlying smooth gradient).
        //
        // The structural rank is invariant in λ by construction (the
        // basis fixes the null space), so the cost surface stays C∞ in ρ.
        // We still take the eigh log-sum across the top
        // `structural_rank` eigenvalues so that any column-rank-deficient
        // basis (e.g. user-supplied data triggering a non-trivial QR
        // pivot at construction) gets the same numerical log|S|+ as
        // before — the only change is which eigenvalues count as
        // "active", and that count is now anchored to the basis.
        let p = e_transformed.ncols();
        let structural_rank = if self.canonical_penalties.is_empty() {
            let (rank, log_det) = positive_penalty_rank_and_logdet(evals.as_slice().unwrap());
            return Ok((rank, log_det));
        } else {
            self.canonical_penalties
                .iter()
                .map(crate::construction::CanonicalPenalty::rank)
                .sum::<usize>()
                .min(p)
        };

        // eigh returns eigenvalues sorted ascending; the structural null
        // directions live at the bottom of the spectrum. Sum the top
        // `structural_rank` log-eigenvalues for log|S|+.
        let evals_slice = evals.as_slice().unwrap();
        let log_det: f64 = evals_slice
            .iter()
            .skip(p.saturating_sub(structural_rank))
            .filter_map(|&ev| if ev > 0.0 { Some(ev.ln()) } else { None })
            .sum();

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
        let h_times_u = crate::faer_ndarray::fast_ab(&h_total, &u_s);
        let mut h_proj = crate::faer_ndarray::fast_atb(&u_s, &h_times_u);
        enforce_symmetry(&mut h_proj);

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
                let frame_was_original = matches!(
                    pr.coordinate_frame,
                    pirls::PirlsCoordinateFrame::OriginalSparseNative
                );
                let beta_original = match pr.coordinate_frame {
                    pirls::PirlsCoordinateFrame::OriginalSparseNative => {
                        pr.beta_transformed.as_ref().clone()
                    }
                    pirls::PirlsCoordinateFrame::TransformedQs => {
                        pr.reparam_result.qs.dot(pr.beta_transformed.as_ref())
                    }
                };
                // Shift the (ρ, β) history one step before storing the new
                // pair, so the previous-pair slot always holds the
                // pre-most-recent solve. Used by
                // `predict_warm_start_beta_with_source` for tangent-line
                // extrapolation across outer iterations.
                {
                    let mut prev_beta_w = self.prev_warm_start_beta.write().unwrap();
                    let mut prev_rho_w = self.prev_warm_start_rho.write().unwrap();
                    let mut cur_beta_w = self.warm_start_beta.write().unwrap();
                    let mut cur_rho_w = self.warm_start_rho.write().unwrap();
                    *prev_beta_w = cur_beta_w.take();
                    *prev_rho_w = cur_rho_w.take();
                    cur_beta_w.replace(Coefficients::new(beta_original.clone()));
                }
                // IFT warm-start cache: stash β / H_pen / qs from this
                // solve so the next outer iter's predictor can apply
                // `dβ/dρ_k = -H^{-1}(e^{ρ_k} S_k(β-μ_k))` directly. ρ is
                // populated by `record_warm_start_rho` immediately
                // after this call returns; until then the slot holds
                // an empty `rho` placeholder that the predictor
                // detects and skips.
                // Precompute the per-penalty `S_k · (β_cur-μ_k)` block-local
                // mat-vecs once at cache-write time so the IFT
                // predictor's per-call rhs construction skips the
                // `O(block²)` mat-vec on every penalty. At biobank-scale
                // CTN this saves a few ms per predict call; combined
                // with the H_pen factor cache (commit ec18559d) the
                // predictor's per-call work is now dominated by the
                // back-solve plus the `O(p)` rhs accumulation.
                let lambda_s_beta_blocks: Option<Vec<ndarray::Array1<f64>>> = {
                    // Parallelize across penalties: each `S_k · (β_cur-μ_k)`
                    // mat-vec is independent of the others. At biobank-
                    // scale CTN with p ≈ several thousand and ~10
                    // penalties, the serial precompute is ~250M flops
                    // per cache write — repeated 5-10× per fit as the
                    // outer optimizer accepts new ρ. Parallelizing
                    // across rayon's thread pool brings this down to
                    // (~250M / cores) flops per write, eliminating
                    // the precompute as a meaningful biobank-scale
                    // serial cost.
                    use rayon::prelude::*;
                    let blocks: Vec<ndarray::Array1<f64>> = self
                        .canonical_penalties
                        .par_iter()
                        .map(|cp| {
                            let r = &cp.col_range;
                            let beta_block = beta_original.slice(s![r.start..r.end]);
                            let centered = &beta_block - &cp.prior_mean;
                            cp.local.dot(&centered)
                        })
                        .collect();
                    if blocks.is_empty() {
                        None
                    } else {
                        Some(blocks)
                    }
                };
                {
                    let mut cache_w = self.ift_warm_start_cache.write().unwrap();
                    cache_w.replace(super::IftWarmStartCache {
                        beta_original,
                        rho: ndarray::Array1::zeros(0),
                        penalized_hessian_transformed: pr.penalized_hessian_transformed.clone(),
                        qs: pr.reparam_result.qs.clone(),
                        frame_was_original,
                        lambda_s_beta_blocks,
                    });
                }
                // The factor cache holds the Cholesky of the PREVIOUS
                // H_pen; replacing the IFT cache invalidates it. Drop
                // here so the next predict call lazily refactors the
                // new H_pen and stashes the new factor.
                self.ift_cached_factor.write().unwrap().take();
            }
            _ => {
                // On a failed solve, drop both the current pair AND the
                // history — the tangent prediction would be misleading.
                // Adaptive signals are wiped by `execute_pirls_if_needed`'s
                // failure branch (where `pirls_result.iteration` is
                // available for the schedule's geometric backoff
                // accounting), so we only clear predictor state here.
                self.clear_warm_start_predictor_state();
            }
        }
    }

    /// Record the ρ at which the most recent successful warm-start β was
    /// obtained. Called immediately after `updatewarm_start_from` succeeds
    /// (the caller knows the ρ that produced the inner solve).
    pub(super) fn record_warm_start_rho(&self, rho: &Array1<f64>) {
        if !self.warm_start_enabled.load(Ordering::Relaxed) {
            return;
        }
        self.warm_start_rho.write().unwrap().replace(rho.to_owned());
        // Stamp the IFT cache's ρ slot so the predictor can compute
        // Δρ = ρ_new − ρ_cache. Skipped if the slot was cleared in
        // between (e.g. a concurrent failure path); the predictor will
        // see an empty `rho` (length 0) and fall back to the
        // tangent-line / flat warm-start path.
        if let Some(cache) = self.ift_warm_start_cache.write().unwrap().as_mut() {
            cache.rho = rho.to_owned();
        }
    }

    /// Outer-loop [`crate::cache::Session`] for this fit, derived from the
    /// same realized-fit-context key as the inner beta record. Disjoint
    /// keyspace, so inner and outer payloads don't collide.
    ///
    /// Returns `None` if no platform cache directory is discoverable.
    pub(crate) fn outer_cache_session(&self) -> Option<std::sync::Arc<crate::cache::Session>> {
        if !self.warm_start_enabled.load(Ordering::Relaxed) {
            return None;
        }
        let key = self.persistent_warm_start_cache_key()?;
        crate::solver::persistent_warm_start::open_outer_session(&key)
    }

    fn persistent_warm_start_cache_key(&self) -> Option<String> {
        if let Some(key) = self.persistent_warm_start_key.read().unwrap().clone() {
            return Some(key);
        }
        let mut hasher = StableHasher::new();
        hasher.write_str("gamfit-persistent-warm-start");
        // Use the cache schema tag (NOT CARGO_PKG_VERSION) so routine
        // library version bumps don't invalidate users' on-disk
        // warm-start caches.
        hasher.write_str(&crate::solver::persistent_warm_start::cache_schema_tag());
        hasher.write_usize(self.y.len());
        hasher.write_usize(self.p);
        hasher.write_str(&format!("{:?}", self.config.likelihood));
        hasher.write_str(&format!("{:?}", self.config.link_kind));
        hasher.write_f64(self.config.pirls_convergence_tolerance);
        hasher.write_f64(self.config.reml_convergence_tolerance);
        hasher.write_usize(self.config.max_iterations);
        hasher.write_bool(self.config.firth_bias_reduction);
        hasher.write_str(&format!("{:?}", self.runtime_mixture_link_state));
        hasher.write_str(&format!("{:?}", self.runtime_sas_link_state));
        match self.penalty_shrinkage_floor {
            Some(value) => {
                hasher.write_bool(true);
                hasher.write_f64(value);
            }
            None => hasher.write_bool(false),
        }
        hasher.write_str(&format!("{:?}", self.rho_prior));

        hash_array_view(&mut hasher, self.y);
        hash_array_view(&mut hasher, self.weights);
        hash_array_view(&mut hasher, self.offset.view());
        if hash_design_matrix(&mut hasher, &self.x).is_err() {
            return None;
        }
        hash_canonical_penalties(&mut hasher, self.canonical_penalties.as_ref());
        hasher.write_usize(self.nullspace_dims.len());
        for &dim in &self.nullspace_dims {
            hasher.write_usize(dim);
        }
        match self.coefficient_lower_bounds.as_ref() {
            Some(bounds) => {
                hasher.write_bool(true);
                hash_array_view(&mut hasher, bounds.view());
            }
            None => hasher.write_bool(false),
        }
        match self.linear_constraints.as_ref() {
            Some(constraints) => {
                hasher.write_bool(true);
                hash_array2(&mut hasher, &constraints.a);
                hash_array_view(&mut hasher, constraints.b.view());
            }
            None => hasher.write_bool(false),
        }
        hasher.write_bool(self.kronecker_penalty_system.is_some());
        hasher.write_bool(self.kronecker_factored.is_some());
        let key = hasher.finish_hex();
        self.persistent_warm_start_key
            .write()
            .unwrap()
            .replace(key.clone());
        Some(key)
    }

    fn load_persistent_warm_start_once(&self) {
        if !self.warm_start_enabled.load(Ordering::Relaxed) {
            return;
        }
        if self
            .persistent_warm_start_loaded
            .swap(true, Ordering::Relaxed)
        {
            return;
        }
        if self.warm_start_beta.read().unwrap().is_some() {
            return;
        }
        let Some(key) = self.persistent_warm_start_cache_key() else {
            return;
        };
        let Some(record) = load_record(&key) else {
            return;
        };
        if !record.is_compatible(&key, self.y.len(), self.p) {
            return;
        }
        let rho_len = record.rho.len();
        if rho_len != self.canonical_penalties.len() {
            return;
        }
        {
            self.warm_start_beta
                .write()
                .unwrap()
                .replace(Coefficients::new(Array1::from_vec(record.beta)));
            self.warm_start_rho
                .write()
                .unwrap()
                .replace(Array1::from_vec(record.rho));
            *self.prev_warm_start_beta.write().unwrap() = record
                .prev_beta
                .map(|beta| Coefficients::new(Array1::from_vec(beta)));
            *self.prev_warm_start_rho.write().unwrap() = record.prev_rho.map(Array1::from_vec);
        }
        self.last_inner_iters
            .store(record.last_inner_iters, Ordering::Relaxed);
        self.last_inner_converged
            .store(record.last_inner_converged, Ordering::Relaxed);
        self.last_pirls_lm_lambda.store(
            record
                .last_pirls_lm_lambda
                .filter(|v| v.is_finite() && *v > 0.0)
                .map(f64::to_bits)
                .unwrap_or(0),
            Ordering::Relaxed,
        );
        self.last_ift_prediction_residual.store(
            finite_nonnegative_bits_or_no_signal(record.last_ift_prediction_residual),
            Ordering::Relaxed,
        );
        self.last_pirls_accept_rho.store(
            finite_nonnegative_bits_or_no_signal(record.last_pirls_accept_rho),
            Ordering::Relaxed,
        );
        log::info!("[warm-start-cache] restored persistent warm start key={key}");
    }

    fn store_persistent_warm_start(&self) {
        if !self.warm_start_enabled.load(Ordering::Relaxed) {
            return;
        }
        let Some(key) = self.persistent_warm_start_cache_key() else {
            return;
        };
        let Some(beta) = self.warm_start_beta.read().unwrap().as_ref().cloned() else {
            return;
        };
        let Some(rho) = self.warm_start_rho.read().unwrap().clone() else {
            return;
        };
        if beta.0.len() != self.p || rho.len() != self.canonical_penalties.len() {
            return;
        }
        let mut record = PersistentWarmStartRecord::new(key, self.y.len(), self.p);
        record.updated_unix_secs = record.created_unix_secs;
        record.rho = rho.to_vec();
        record.beta = beta.0.to_vec();
        record.prev_rho = self
            .prev_warm_start_rho
            .read()
            .unwrap()
            .as_ref()
            .map(|rho| rho.to_vec());
        record.prev_beta = self
            .prev_warm_start_beta
            .read()
            .unwrap()
            .as_ref()
            .map(|coefficients| coefficients.0.to_vec());
        record.last_inner_iters = self.last_inner_iters.load(Ordering::Relaxed);
        record.last_inner_converged = self.last_inner_converged.load(Ordering::Relaxed);
        record.last_pirls_lm_lambda =
            finite_positive_from_bits(self.last_pirls_lm_lambda.load(Ordering::Relaxed));
        record.last_ift_prediction_residual =
            finite_nonnegative_from_bits(self.last_ift_prediction_residual.load(Ordering::Relaxed));
        record.last_pirls_accept_rho =
            finite_nonnegative_from_bits(self.last_pirls_accept_rho.load(Ordering::Relaxed));
        if let Err(err) = store_record(&record) {
            log::warn!("[warm-start-cache] failed to persist warm start: {err}");
        }
    }

    /// Predict β at `new_rho` via the implicit-function-theorem first-order
    /// expansion
    /// `β_predict = β_cur − Σ_k Δρ_k · H_pen^{-1} · (e^{ρ_cur_k} S_k(β_cur-μ_k))`,
    /// surfacing the outcome (Predicted vs Noop) so callers can map directly
    /// onto `WarmStartPredictionSource` without re-deriving noop-ness via an
    /// O(p) array comparison against the cached β.
    ///
    /// Returns `None` (caller falls back to tangent-line / flat warm-start) when:
    /// * the IFT cache is empty (no prior converged solve, or one was invalidated),
    /// * ρ has been stamped yet (length 0 — see `record_warm_start_rho`),
    /// * Δρ is too aggressive for the linearized predictor to trust
    ///   (max |Δρ_k| > 2.0 — i.e. a single penalty has moved by more than e²
    ///   in λ-space, well outside the regime where the local linear Jacobian
    ///   is descriptive),
    /// * factorization or back-solve fails / produces non-finite output.
    ///
    /// The factor is cached at `self.ift_cached_factor` and reused across
    /// successive predict calls until the IFT cache itself is replaced (a
    /// new `updatewarm_start_from`) or invalidated (failed solve, link
    /// change, surface reset). At biobank scale (p ≈ several thousand)
    /// the dense Cholesky is O(p³)/3 — multiple seconds per refactor —
    /// so caching saves real wall time across the typical 5-10 IFT
    /// predict calls per outer fit.
    pub(crate) fn predict_warm_start_beta_ift_with_outcome(
        &self,
        new_rho: &Array1<f64>,
    ) -> Option<(Coefficients, IftPredictionOutcome)> {
        if !self.warm_start_enabled.load(Ordering::Relaxed) {
            return None;
        }
        let cache_guard = self.ift_warm_start_cache.read().unwrap();
        let cache = cache_guard.as_ref()?;
        // The NaN sentinel + the is_finite() check together cover three
        // cases in one expression: "no signal yet" (sentinel decodes to
        // NaN, which fails is_finite), "corrupted state" (any non-finite
        // or negative residual stored by mistake), and "real signal"
        // (finite non-negative residual → Some).
        let last_residual_bits = self.last_ift_prediction_residual.load(Ordering::Relaxed);
        let r = f64::from_bits(last_residual_bits);
        let last_residual = if r.is_finite() && r >= 0.0 {
            Some(r)
        } else {
            None
        };
        // Early short-circuit: detect both the no-op case (every
        // |Δρ_k| below the numerical-noise floor → predictor reduces
        // to identity) AND the large-Δρ rejection case (|Δρ| exceeds
        // the adaptive cap → predictor would reject) BEFORE acquiring
        // the H_pen factor. The inner function detects both cases and
        // returns the same outcome, but only AFTER the factor cache
        // lookup — which on a miss pays a fresh O(p³)/3 Cholesky
        // (multiple seconds at biobank-scale p) for a prediction
        // that's about to be discarded.
        //
        // The dim-check guards mirror the inner function's so an
        // ambiguous case (rho-not-stamped, dim mismatch) falls through
        // and lets the inner function emit its precise rejection
        // marker, preserving the single-source-of-truth contract.
        if !cache.rho.is_empty() && cache.rho.len() == new_rho.len() {
            let mut max_abs_drho = 0.0_f64;
            let mut any_non_finite = false;
            for i in 0..cache.rho.len() {
                let d = new_rho[i] - cache.rho[i];
                if !d.is_finite() {
                    any_non_finite = true;
                    max_abs_drho = f64::INFINITY;
                    break;
                }
                if d.abs() > max_abs_drho {
                    max_abs_drho = d.abs();
                }
            }
            // No-op case: every |Δρ_k| below the eps floor.
            if !any_non_finite && max_abs_drho <= IFT_WARM_START_DRHO_EPS {
                // Same NOOP marker the inner function would emit, so the
                // bench runner aggregator's count is preserved across
                // both the early-out path and the post-factor path.
                log::info!(
                    "[IFT-NOOP] reason=all_drho_below_eps max_drho={:.3e} drho_dim={}",
                    max_abs_drho,
                    cache.rho.len(),
                );
                return Some((
                    Coefficients::new(cache.beta_original.clone()),
                    IftPredictionOutcome::Noop,
                ));
            }
            // Large-Δρ rejection: |Δρ| exceeds the adaptive cap.
            // `adaptive_ift_max_drho` reads the same `last_residual`
            // signal we already loaded above. Same marker as the inner
            // function emits so the rejection-rate aggregator
            // (`_IFT_REJECTED_PATTERN` in runner.py) is preserved
            // across both paths.
            let max_drho_cap = adaptive_ift_max_drho(last_residual);
            if !max_abs_drho.is_finite() || max_abs_drho > max_drho_cap {
                log::info!(
                    "[IFT-REJECTED] reason=large_drho max_drho={:.3e} cap={:.3e} drho_dim={}",
                    max_abs_drho,
                    max_drho_cap,
                    cache.rho.len(),
                );
                return None;
            }
        }
        // Get the cached factor or lazily compute it. Holding the
        // ift_warm_start_cache read lock while we acquire the factor's
        // lock is safe: cache invalidation (writer) takes the cache
        // write lock first, then the factor write lock, so a reader
        // holding the cache lock prevents the writer from advancing
        // and we never observe a factor that doesn't match the cached
        // matrix.
        let factor_arc: Arc<dyn crate::linalg::matrix::FactorizedSystem> = {
            let read_guard = self.ift_cached_factor.read().unwrap();
            if let Some(arc) = read_guard.as_ref() {
                // Cache hit: H_pen factor was already computed by an
                // earlier predict call at the same surface. The
                // [IFT-CACHE] marker validates that the factor cache
                // (commit ec18559d) is paying off at biobank scale —
                // every cache hit avoids a fresh O(p³)/3 Cholesky,
                // which is multiple seconds at p ≈ several thousand.
                log::info!(
                    "[IFT-CACHE] outcome=hit drho_dim={} p={}",
                    new_rho.len(),
                    self.p,
                );
                Arc::clone(arc)
            } else {
                drop(read_guard);
                let factorize_start = std::time::Instant::now();
                let new_factor = match cache.penalized_hessian_transformed.factorize() {
                    Ok(f) => f,
                    Err(_) => {
                        log::info!(
                            "[IFT-REJECTED] reason=hessian_factorize_failed_cached drho_dim={}",
                            new_rho.len(),
                        );
                        return None;
                    }
                };
                // Cache miss: paid the Cholesky once. Subsequent predict
                // calls at the same surface will hit the cache.
                log::info!(
                    "[IFT-CACHE] outcome=miss drho_dim={} p={} elapsed={:.3}s",
                    new_rho.len(),
                    self.p,
                    factorize_start.elapsed().as_secs_f64(),
                );
                let arc: Arc<dyn crate::linalg::matrix::FactorizedSystem> = Arc::from(new_factor);
                let mut write_guard = self.ift_cached_factor.write().unwrap();
                // Race window: another reader may have populated the
                // slot between our drop(read_guard) and the write lock
                // acquisition. Prefer the existing entry to keep the
                // single-factor invariant and avoid tearing.
                if let Some(existing) = write_guard.as_ref() {
                    Arc::clone(existing)
                } else {
                    *write_guard = Some(Arc::clone(&arc));
                    arc
                }
            }
        };
        predict_warm_start_beta_ift_inner_with_outcome(
            cache,
            self.canonical_penalties.as_ref(),
            new_rho,
            self.p,
            last_residual,
            Some(factor_arc.as_ref()),
        )
    }

    /// Predict β at `new_rho` together with a tag identifying which
    /// predictor produced it (IFT first-order Jacobian or tangent-line
    /// extrapolation across the last two (ρ, β) pairs). Returns `None`
    /// when no predictor can be applied — the warm-start machinery is
    /// disabled, neither cache is populated, the ρ-step direction is
    /// degenerate, or the extrapolation step `α` exceeds the adaptive
    /// safety cap (default 1.5 — see `adaptive_tangent_alpha_cap` for
    /// the residual-driven policy that loosens to 2.0 when prior IFT
    /// predictions were excellent and tightens to 0.5 when the local
    /// linear approximation has been shown to collapse toward flat
    /// warm-start).
    ///
    /// Callers fall back to the stored `warm_start_beta` (`β(ρ_k)` —
    /// the standard "use last β as-is" warm start) on `None`. So this
    /// is strictly an improvement: when prediction is safe, we use it;
    /// otherwise we use the existing flat warm-start. The source tag
    /// drives per-predictor quality markers (the [IFT-QUALITY] /
    /// [TANGENT-QUALITY] chain in `execute_pirls_if_needed`) so each
    /// marker is attributed correctly to the predictor that actually
    /// produced the β.
    pub(crate) fn predict_warm_start_beta_with_source(
        &self,
        new_rho: &Array1<f64>,
    ) -> Option<(Coefficients, WarmStartPredictionSource)> {
        if !self.warm_start_enabled.load(Ordering::Relaxed) {
            return None;
        }
        // Try the IFT-based predictor first. It uses the exact first-order
        // Jacobian of β(ρ) at the cached solve and beats the tangent-line
        // predictor whenever a converged H_pen is available — which is
        // every call after the first successful PIRLS solve.
        //
        // Source disambiguation: IFT returns Some(β_cur) on noop (all
        // Δρ_k below the per-component eps floor). The tangent-line
        // path makes the same distinction explicit by tagging Flat
        // (commit 1b77588b). The IFT inner predictor exposes the
        // outcome directly via `predict_warm_start_beta_ift_with_outcome`,
        // eliminating the O(p) array-comparison-against-cached-β that
        // the prior implementation needed to re-derive noop-ness here.
        // Map outcome → source enum without further work.
        if let Some((predicted, outcome)) = self.predict_warm_start_beta_ift_with_outcome(new_rho) {
            log::debug!("[warm-start] IFT prediction accepted");
            let source = match outcome {
                IftPredictionOutcome::Predicted => WarmStartPredictionSource::Ift,
                IftPredictionOutcome::Noop => WarmStartPredictionSource::Flat,
            };
            return Some((predicted, source));
        }
        let cur_beta = self.warm_start_beta.read().unwrap().clone()?;
        let cur_rho = self.warm_start_rho.read().unwrap().clone();
        let prev_beta = self.prev_warm_start_beta.read().unwrap().clone();
        let prev_rho = self.prev_warm_start_rho.read().unwrap().clone();
        let (cur_rho, prev_beta, prev_rho) = match (cur_rho, prev_beta, prev_rho) {
            (Some(cr), Some(pb), Some(pr)) => (cr, pb, pr),
            // No history yet — first call after a fresh successful
            // solve (only one (ρ, β) pair stashed). Silent fallback
            // is the right behavior; emitting a marker here would
            // just be noise on every first call.
            _ => return Some((cur_beta, WarmStartPredictionSource::Flat)),
        };
        // Dimension mismatch paths below are bug signals: state-machine
        // inconsistency between (β, ρ) cache and current state. Emit
        // structured markers so the bench runner can detect non-zero
        // counts as a regression signal. The wipe helpers from commit
        // 6f7cbfc8 should have cleared these slots before the layout
        // shifted; non-zero count indicates a missed invalidation.
        if cur_rho.len() != new_rho.len() || cur_rho.len() != prev_rho.len() {
            log::info!(
                "[TANGENT-REJECTED] reason=rho_dim_mismatch new_rho_dim={} cur_rho_dim={} prev_rho_dim={}",
                new_rho.len(),
                cur_rho.len(),
                prev_rho.len(),
            );
            return Some((cur_beta, WarmStartPredictionSource::Flat));
        }
        if cur_beta.0.len() != prev_beta.0.len() {
            log::info!(
                "[TANGENT-REJECTED] reason=beta_dim_mismatch cur_beta_dim={} prev_beta_dim={}",
                cur_beta.0.len(),
                prev_beta.0.len(),
            );
            return Some((cur_beta, WarmStartPredictionSource::Flat));
        }
        // d_rho = ρ_k − ρ_{k-1}; step_rho = ρ_new − ρ_k.
        let d_rho_norm_sq: f64 = cur_rho
            .iter()
            .zip(prev_rho.iter())
            .map(|(c, p)| (c - p) * (c - p))
            .sum();
        if !d_rho_norm_sq.is_finite() || d_rho_norm_sq <= 1e-24 {
            // Degenerate Δρ direction (the previous ρ-step had zero or
            // unfinite length). Diagnostic rather than bug: this fires
            // when the outer optimizer landed on a flat region or the
            // ρ history collapsed. Surfacing the case lets the bench
            // runner see flat-region traces.
            log::info!(
                "[TANGENT-REJECTED] reason=degenerate_drho d_rho_norm_sq={:.3e}",
                d_rho_norm_sq,
            );
            return Some((cur_beta, WarmStartPredictionSource::Flat));
        }
        let step_dot_d: f64 = new_rho
            .iter()
            .zip(cur_rho.iter())
            .zip(prev_rho.iter())
            .map(|((n, c), p)| (n - c) * (c - p))
            .sum();
        let alpha = step_dot_d / d_rho_norm_sq;
        if !alpha.is_finite() {
            // Non-finite α (NaN or Inf). Real bug signal — the
            // numerator or denominator overflowed. Should never happen
            // if d_rho_norm_sq passed the prior finiteness check.
            log::info!(
                "[TANGENT-REJECTED] reason=nonfinite_alpha step_dot_d={:.3e} d_rho_norm_sq={:.3e}",
                step_dot_d,
                d_rho_norm_sq,
            );
            return Some((cur_beta, WarmStartPredictionSource::Flat));
        }
        // Don't extrapolate against the previous step direction (α<0).
        // The upper cap is adaptive: the same IFT residual signal that
        // drives `adaptive_ift_max_drho` (commit 06888a1e) and the cap-
        // schedule margin (commit 4eb3686a) is the most direct proxy
        // for "how trustworthy is the local linear approximation" —
        // and the tangent-line predictor IS a local linear
        // approximation (just along the previous ρ-step direction
        // rather than the IFT Jacobian's full direction). When the
        // most recent IFT prediction was excellent, the tangent
        // approximation can be trusted for slightly larger α; when
        // it was poor, tighten.
        // Same NaN-sentinel discipline as the IFT predictor's reader
        // — see the comment there for the encoding rationale.
        let last_residual_bits = self.last_ift_prediction_residual.load(Ordering::Relaxed);
        let r = f64::from_bits(last_residual_bits);
        let last_residual = if r.is_finite() && r >= 0.0 {
            Some(r)
        } else {
            None
        };
        let alpha_cap = adaptive_tangent_alpha_cap(last_residual);
        if alpha <= 0.0 || alpha > alpha_cap {
            // Emit a structured reject marker so the bench runner can
            // count tangent-line rejections alongside IFT ones.
            // Tangent-line only fires when IFT returned None for
            // non-cache reasons (large Δρ, factor failed, etc.), so
            // this represents the "linear predictor stack failed
            // entirely → fall back to flat warm-start" case. Counting
            // the rate at biobank scale tells us how often the warm-
            // start is degenerating to flat after IFT rejects.
            let reason = if alpha <= 0.0 {
                "alpha_negative"
            } else {
                "alpha_above_cap"
            };
            log::info!(
                "[TANGENT-REJECTED] reason={} alpha={:.3e} cap={:.3e}",
                reason,
                alpha,
                alpha_cap,
            );
            return Some((cur_beta, WarmStartPredictionSource::Flat));
        }
        // Tangent noop short-circuit: when α is below the
        // numerical-noise floor, `c + α · (c − pp) ≈ c` to machine
        // precision and the per-coefficient mat-add is wasted work.
        // Symmetric to the IFT-NOOP path (commit d437aed1 / 52372fd5)
        // and tagged with the same `[TANGENT-NOOP]` marker so the
        // bench runner can distinguish "tangent-line ran with
        // negligible step" from "tangent-line produced a real
        // prediction" without inferring it from the alpha
        // distribution. Returns Flat source so the
        // [TANGENT-QUALITY] block in execute_pirls_if_needed
        // doesn't fire on a near-identity prediction (would
        // contaminate the residual percentile distribution with
        // ~zero residuals — same bug class as commit 52372fd5).
        const TANGENT_ALPHA_NOOP_EPS: f64 = 1e-12;
        if alpha.abs() <= TANGENT_ALPHA_NOOP_EPS {
            log::info!(
                "[TANGENT-NOOP] reason=alpha_below_eps alpha={:.3e} eps={:.3e}",
                alpha,
                TANGENT_ALPHA_NOOP_EPS,
            );
            return Some((cur_beta, WarmStartPredictionSource::Flat));
        }
        let mut predicted = cur_beta.0.clone();
        for ((p, c), pp) in predicted
            .iter_mut()
            .zip(cur_beta.0.iter())
            .zip(prev_beta.0.iter())
        {
            *p = c + alpha * (c - pp);
        }
        if !predicted.iter().all(|v: &f64| v.is_finite()) {
            log::info!(
                "[TANGENT-REJECTED] reason=non_finite_predicted alpha={:.3e} cap={:.3e}",
                alpha,
                alpha_cap,
            );
            return Some((cur_beta, WarmStartPredictionSource::Flat));
        }
        log::info!(
            "[TANGENT-PREDICT] alpha={:.3e} cap={:.3e} drho_step_norm_sq={:.3e} drho_prev_norm_sq={:.3e}",
            alpha,
            alpha_cap,
            step_dot_d.abs(),
            d_rho_norm_sq,
        );
        Some((
            Coefficients::new(predicted),
            WarmStartPredictionSource::TangentLine,
        ))
    }

    pub(crate) fn setwarm_start_original_beta(&self, beta_original: Option<ArrayView1<'_, f64>>) {
        if !self.warm_start_enabled.load(Ordering::Relaxed) {
            return;
        }
        // Whenever an external caller substitutes the warm-start β
        // (e.g. a multi-start seed selector), every other piece of
        // warm-start state was calibrated to the OLD β at the OLD ρ
        // and is now stale: the IFT cache's H_pen and qs were assembled
        // against the old β; the cached factor is the Cholesky of the
        // OLD H_pen; the ρ pair (warm_start_rho, prev_*) was the OLD β's
        // ρ-trajectory; the LM-λ hint was the damping that worked at
        // the OLD geometry; the inner-PIRLS feedback signals reflect
        // the OLD solve's convergence behavior; the IFT residual was
        // measured against the OLD predictor.
        //
        // Without this wipe, the IFT predictor would seed PIRLS with
        // `β_new − H_old^{-1} · (e^{ρ_old_k} S_k · (β_new-μ_k))` — using the
        // new β as a substitute for the old β in a Jacobian
        // calibrated against the old β. That is mathematically wrong
        // and would produce arbitrary predictions, not just degraded
        // ones. Similarly the LM-λ hint would clamp to a value
        // calibrated to a curvature that no longer applies.
        //
        // Wipe everything EXCEPT `warm_start_beta` (which we are
        // about to replace), then write the new β. The order matters:
        // `clear_warm_start_predictor_state` clears `warm_start_beta`
        // too, so we replace afterwards.
        self.clear_warm_start_predictor_state();
        self.clear_warm_start_adaptive_signals();
        match beta_original {
            Some(beta) if beta.len() == self.p => {
                if !beta.iter().all(|v: &f64| v.is_finite()) {
                    // Caller supplied a β with NaN / Inf entries — would
                    // poison the next PIRLS solve immediately. Refuse
                    // (slot remains cleared) and log so the source of
                    // the bad seed is debuggable.
                    log::warn!(
                        "[warm-start] external β setter rejected non-finite seed (len={}); slot left empty",
                        beta.len(),
                    );
                    return;
                }
                self.warm_start_beta
                    .write()
                    .unwrap()
                    .replace(Coefficients::new(beta.to_owned()));
            }
            Some(beta) => {
                // Length mismatch — common bug when a caller forgets to
                // re-derive β under a basis transformation. Surface it
                // rather than silently dropping; the slot is already
                // cleared.
                log::warn!(
                    "[warm-start] external β setter rejected length mismatch: got {}, expected {}",
                    beta.len(),
                    self.p,
                );
            }
            None => {
                // Caller asked for an explicit clear — already done by
                // `clear_warm_start_predictor_state`; nothing more to do.
            }
        }
    }

    pub(crate) fn reset_outer_seed_state(&self) {
        self.cache_manager.invalidate_eval_bundle();
        // Drop cross-call PIRLS LRU entries: cached β may have been computed under a coarsened inner cap, so reusing them on retry skips real work and bit-replays the prior attempt.
        self.cache_manager.pirls_cache.write().unwrap().clear();
        // The outer is restarting from a fresh seed — the previous
        // trajectory's warm-start signals are calibrated to a different
        // ρ-path and would mislead both predictors and the adaptive cap
        // policies. Wipe in lockstep so the first solve at the new
        // seed starts fully cold.
        self.clear_warm_start_predictor_state();
        self.clear_warm_start_adaptive_signals();
        // Inner-PIRLS iteration caps are cross-trajectory state: the
        // previous outer's first-order bridge writes `outer_inner_cap`
        // on every accepted gradient eval, and seed-screening writes
        // `screening_max_inner_iterations` during cascade probes. A
        // value left over from the prior trajectory would silently
        // shape the first PIRLS solve at the new seed under a cap the
        // new trajectory never chose. The next outer is responsible
        // for re-establishing any cap it wants via its own bridge or
        // screening hook; zeroing here is the safe baseline.
        self.outer_inner_cap.store(0, Ordering::Relaxed);
        self.screening_max_inner_iterations
            .store(0, Ordering::Relaxed);
    }

    // Accessor methods for private fields
    pub(crate) fn x(&self) -> &DesignMatrix {
        &self.x
    }

    pub(crate) fn balanced_penalty_root(&self) -> &Array2<f64> {
        &self.balanced_penalty_root
    }

    /// Return a Gaussian-Identity `XᵀWX` / `XᵀW(y−offset)` cache when the
    /// outer-loop preconditions for the Identity short-circuit hold, building
    /// it lazily on the first call.  Returns `None` otherwise — callers must
    /// then route through the non-cached path.
    ///
    /// Preconditions: family `GaussianIdentity` + standard `Identity` link
    /// + no Firth bias reduction + no coefficient lower bounds + no linear
    /// inequality constraints.  These exactly match the gate in
    /// `solve_penalized_least_squares_implicit` so the cache is only ever
    /// consulted where it is mathematically equivalent to streaming the
    /// dense `XᵀWX` from scratch.
    pub(crate) fn gaussian_fixed_cache_if_eligible(
        &self,
    ) -> Option<Arc<crate::pirls::GaussianFixedCache>> {
        // Static eligibility — these only depend on data the outer loop
        // never mutates, so the gate is correct once and stays correct.
        let family_ok = matches!(
            self.config.likelihood.family,
            crate::types::GlmLikelihoodFamily::GaussianIdentity
        );
        let link_ok = matches!(
            self.config.link_kind,
            crate::types::InverseLink::Standard(LinkFunction::Identity)
        );
        if !family_ok
            || !link_ok
            || self.config.firth_bias_reduction
            || self.coefficient_lower_bounds.is_some()
            || self.linear_constraints.is_some()
        {
            return None;
        }
        // Fast path — already populated.
        {
            let guard = self.gaussian_fixed_cache.read().unwrap();
            if let Some(cache) = guard.as_ref() {
                return Some(Arc::clone(cache));
            }
        }
        // First-call construction. Re-check under the write lock so two
        // concurrent callers cannot both pay the O(N·p²) bill.
        let mut guard = self.gaussian_fixed_cache.write().unwrap();
        if let Some(cache) = guard.as_ref() {
            return Some(Arc::clone(cache));
        }
        let build_start = std::time::Instant::now();
        let weights_owned = self.weights.to_owned();
        // wz_minus_offset = (y - offset) elementwise; cache stores XᵀW(y−offset).
        let mut wz = self.y.to_owned();
        wz -= &self.offset;
        wz *= &weights_owned;
        let centered_weighted_y_sq = self
            .y
            .iter()
            .zip(self.offset.iter())
            .zip(weights_owned.iter())
            .map(|((&y, &offset), &w)| {
                let centered = y - offset;
                w * centered * centered
            })
            .sum::<f64>();
        let xtwx = match self.x.compute_xtwx(&weights_owned) {
            Ok(m) => m,
            Err(e) => {
                log::warn!("[gaussian-fixed-cache] disabling cache: failed to build XᵀWX: {e}");
                return None;
            }
        };
        let xtwy = self.x.transpose_vector_multiply(&wz);
        // Build the matching sparse-path XᵀWX as well, when the design
        // exposes a sparse form. The sparse REML path rebuilds
        // H = XᵀWX + Sλ + δI per outer eval, so caching the constant
        // XᵀWX contribution lets the inner assemble skip the SpGEMM.
        let xtwx_sparse_orig = if let Some(sparse_design) = self.x.as_sparse() {
            let sparse_start = std::time::Instant::now();
            match crate::pirls::SparseXtwxPrecomputed::build(sparse_design.as_ref(), &weights_owned)
            {
                Ok(precomp) => {
                    log::info!(
                        "[gaussian-fixed-cache] sparse XᵀWX nnz={} built in {:.3} ms",
                        precomp.xtwxvalues.len(),
                        sparse_start.elapsed().as_secs_f64() * 1e3
                    );
                    Some(Arc::new(precomp))
                }
                Err(e) => {
                    log::warn!(
                        "[gaussian-fixed-cache] sparse XᵀWX build failed; falling back: {e}"
                    );
                    None
                }
            }
        } else {
            None
        };
        let cache = Arc::new(crate::pirls::GaussianFixedCache {
            xtwx_orig: xtwx,
            xtwy_orig: xtwy,
            centered_weighted_y_sq,
            xtwx_sparse_orig,
        });
        log::info!(
            "[gaussian-fixed-cache] built p={} n={} in {:.3} ms",
            self.p,
            self.y.len(),
            build_start.elapsed().as_secs_f64() * 1e3
        );
        *guard = Some(Arc::clone(&cache));
        Some(cache)
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
        let (mut h_total, ridge_passport) = self.effectivehessian(pirls_result.as_ref())?;
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
                log::debug!(
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
                enforce_symmetry(&mut hphi);
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
        // Gaussian-Identity fast path: reuse the per-RemlState `XᵀWX` cache
        // built once from constant weights. The outer REML loop never
        // mutates W for Gaussian, so the inner sparse assemble can scatter
        // these values directly instead of re-running the SpGEMM.
        let gaussian_cache = self.gaussian_fixed_cache_if_eligible();
        let precomputed_xtwx = gaussian_cache
            .as_ref()
            .and_then(|c| c.xtwx_sparse_orig.as_ref().map(|arc| arc.as_ref()));
        let sparse_system = assemble_and_factor_sparse_penalized_system(
            &mut workspace,
            x_sparse,
            &pirls_result.finalweights,
            &s_lambda,
            ridge_passport.delta,
            precomputed_xtwx,
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

        // Detect whether we are running under outer-loop seed screening. While
        // screening is active the planner caps inner iterations to a small
        // value (~3); trial seeds are NOT expected to certify a stationary
        // mode under that cap. Partial fits whose objective (deviance +
        // penalty), β, Hessian proxy, and residual are all finite are
        // surfaced as `Ok` so the caller can rank them by an approximate
        // cost (`screening_residual_penalty`); KKT enforcement, the
        // pirls_cache LRU write, and the warm-start update are all
        // suppressed to keep screening-mode results out of cross-call
        // state. The single atomic load below feeds both the bool and the
        // iteration cap from one observation, avoiding a stale-value race
        // between the two derivations.
        let screening_cap = self.screening_max_inner_iterations.load(Ordering::Relaxed);
        let in_screening = screening_cap > 0;
        // Outer-aware cap: a sibling atomic that only caps the inner Newton
        // iteration count. Unlike `screening_cap`, it does NOT suppress
        // cache writes / warm-start updates / KKT enforcement — it is purely
        // a budget. Driven by the outer optimizer to coarsen early-iter
        // inner solves when ρ is far from converged. Both caps are honored
        // jointly via `min` when both are nonzero.
        let outer_cap = self.outer_inner_cap.load(Ordering::Relaxed);

        // Run P-IRLS with original matrices to perform fresh reparameterization
        // The returned result will include the transformation matrix qs.
        //
        // Warm-start: try the tangent-line prediction first (uses last
        // two (ρ, β) pairs to extrapolate β at the new ρ; falls back to
        // the flat last-β when no history). Either path produces a
        // `Coefficients` value used as the inner Newton's seed.
        if !in_screening {
            self.load_persistent_warm_start_once();
        }
        let predicted_warm_start_with_source = if self.warm_start_enabled.load(Ordering::Relaxed) {
            self.predict_warm_start_beta_with_source(rho)
        } else {
            None
        };
        let predicted_warm_start = predicted_warm_start_with_source
            .as_ref()
            .map(|(c, _)| c.clone());
        let prediction_source = predicted_warm_start_with_source.as_ref().map(|(_, s)| *s);
        let pirls_result = {
            let warm_start_holder = self.warm_start_beta.read().unwrap();
            let fallback_warm_start_ref = if self.warm_start_enabled.load(Ordering::Relaxed) {
                warm_start_holder.as_ref()
            } else {
                None
            };
            let warm_start_ref = predicted_warm_start.as_ref().or(fallback_warm_start_ref);
            let mut pirls_config = self.config.as_pirls_config();
            let original_cap = pirls_config.max_iterations;
            if in_screening {
                pirls_config.max_iterations = pirls_config.max_iterations.min(screening_cap);
            }
            if outer_cap > 0 {
                pirls_config.max_iterations = pirls_config.max_iterations.min(outer_cap);
            }
            if pirls_config.max_iterations != original_cap {
                log::debug!(
                    "[PIRLS cap] inner_max_iterations={} (full={} screening={} outer={})",
                    pirls_config.max_iterations,
                    original_cap,
                    if in_screening {
                        screening_cap as i64
                    } else {
                        -1
                    },
                    if outer_cap > 0 { outer_cap as i64 } else { -1 },
                );
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
            // Levenberg-Marquardt damping warm-start. Read the cached
            // λ from the previous successful PIRLS solve at this
            // surface (0 = no hint), and seed the inner solver. The
            // PIRLS layer applies a final safety clamp (defense in
            // depth); this layer pre-clamps adaptively based on the
            // previous solve's halving history, plus a quality signal
            // that the static clamp couldn't see.
            //
            // The principle: the cached λ encodes the curvature regime
            // the previous fit settled into. A Newton-friendly fit
            // (converged in ≤2 iters, no halving) leaves λ near the
            // 1e-9 floor; the next fit at a nearby ρ is likely
            // Newton-friendly too, so we allow the hint down to that
            // floor. A hard fit (many iters, possibly hit cap) leaves
            // λ in the heavy-damping regime; the next fit needs to
            // preserve that signal, so we allow the hint up to ~1.0.
            // The default (1-9 iters, converged) matches the historical
            // static clamp [1e-6, 1e-3].
            //
            // Each regime keeps the cached value's geometry information;
            // none of them throw it away. The Madsen rejection
            // trajectory (commit d37626e6) handles "wrong starting λ"
            // gracefully, so the cost of a slightly mis-tuned regime
            // is just a few extra rejections — far less than the cost
            // of throwing away the cache entirely.
            let cached_lambda_bits = self.last_pirls_lm_lambda.load(Ordering::Relaxed);
            if cached_lambda_bits != 0 {
                let cached_lambda = f64::from_bits(cached_lambda_bits);
                let last_iters = self.last_inner_iters.load(Ordering::Relaxed);
                let last_converged = self.last_inner_converged.load(Ordering::Relaxed);
                pirls_config.initial_lm_lambda =
                    adaptive_lm_lambda_hint(cached_lambda, last_iters, last_converged);
            }
            // Gaussian + Identity outer REML reuses a precomputed XᵀWX and
            // XᵀW(y − offset) across every inner solve; for other families /
            // links this returns None and the inner solver falls back to the
            // streaming GEMM.
            let cache_handle = self.gaussian_fixed_cache_if_eligible();
            let problem = pirls::PirlsProblem {
                x: &self.x,
                offset: self.offset.view(),
                y: self.y,
                priorweights: self.weights,
                covariate_se: None,
                gaussian_fixed_cache: cache_handle.as_deref(),
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
                // Surface every non-screening PIRLS call at INFO so CI logs
                // reveal exactly which inner solves dominate wall-clock at
                // biobank scale — the main signal for "what's the slow path"
                // when an outer BFGS / line-search step blows past the
                // 2400 s job budget.
                log::info!(
                    "[STAGE] inner pirls solve iters={} status={:?} max_eta={:.1} jeffreys_logdet={} elapsed={:.3}s",
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
            if in_screening {
                // Seed-screening intentionally caps inner iterations very low,
                // so trial-point failures here are routine and ranked — not
                // bugs.
                log::debug!("[seed-screen] P-IRLS rejected candidate: {e:?}");
            } else {
                log::warn!("[GAM COST]   -> P-IRLS INNER LOOP FAILED. Error: {e:?}");
            }
            // Keep the previous successful warm start even when a trial point
            // fails. Outer line search commonly probes unstable candidates and
            // then returns to nearby feasible rho values where the prior warm
            // beta remains the best initializer.
        }

        let (pirls_result, _) = pirls_result?; // Propagate error if it occurred
        let pirls_result = Arc::new(pirls_result);
        // Under seed screening the inner solver is intentionally given a tiny
        // iteration budget, so KKT stationarity will not be satisfied at the
        // partial mode. Skip the certificate so the seed can still be ranked
        // by an approximate cost; the actual fit (full inner budget) will
        // certify KKT later.
        if !in_screening {
            self.enforce_constraint_kkt(pirls_result.as_ref())?;
        }

        // Check the status returned by the P-IRLS routine.
        match pirls_result.status {
            pirls::PirlsStatus::Converged | pirls::PirlsStatus::StalledAtValidMinimum => {
                // IFT-quality probe: when the predictor produced a β
                // (any non-screening solve after the first cache fill
                // typically does), measure how close the prediction was
                // to the converged β. Logged as
                // `[IFT-QUALITY] residual=X converged_norm=Y predicted_norm=Z iters=K`
                // so the bench runner / scaling-law analyzer can
                // empirically validate the IFT linearization at biobank
                // scale. A residual ≪ 1 means the linear predictor is
                // faithful (warm-start is paying off); a residual ≈ 1
                // means the prediction was no better than flat (and
                // the adaptive cap / next outer iter should treat the
                // hint as unreliable). Only emitted outside screening
                // because screening's intentionally-truncated PIRLS
                // does not reflect production solve geometry.
                if !in_screening {
                    if let Some(predicted) = predicted_warm_start.as_ref() {
                        // Detect noop predictions (predictor returned the
                        // cached β unchanged because all Δρ were below the
                        // numerical floor). On noop the residual measures
                        // inner-Newton movement after a zero-step seed,
                        // NOT predictor faithfulness, and including it in
                        // the IFT-QUALITY residual distribution biases
                        // the percentiles toward whatever β-shift PIRLS
                        // produced from a free warm-start. The [IFT-NOOP]
                        // marker (commit d437aed1) already counts these
                        // calls separately, so we skip the [IFT-QUALITY]
                        // emission here for cleanliness.
                        // Noop detection: the predictor's source enum
                        // already marks Flat returns unambiguously
                        // (commit 1b77588b for tangent-line + this
                        // commit's IFT alignment). When source=Flat,
                        // the predictor returned the cached β unchanged
                        // — the QUALITY block would emit a residual
                        // that measures inner-Newton β movement at a
                        // zero-step seed, NOT predictor faithfulness.
                        // Skip the emission entirely on Flat. Older
                        // array-comparison `was_noop` check is now
                        // redundant; the source enum is the single
                        // source of truth.
                        let was_noop =
                            matches!(prediction_source, Some(WarmStartPredictionSource::Flat),);
                        let converged_original = match pirls_result.coordinate_frame {
                            pirls::PirlsCoordinateFrame::OriginalSparseNative => {
                                pirls_result.beta_transformed.as_ref().clone()
                            }
                            pirls::PirlsCoordinateFrame::TransformedQs => pirls_result
                                .reparam_result
                                .qs
                                .dot(pirls_result.beta_transformed.as_ref()),
                        };
                        if !was_noop && predicted.0.len() == converged_original.len() {
                            let mut diff_sq = 0.0_f64;
                            let mut conv_sq = 0.0_f64;
                            let mut pred_sq = 0.0_f64;
                            for (p_val, c_val) in predicted.0.iter().zip(converged_original.iter())
                            {
                                let d = c_val - p_val;
                                diff_sq += d * d;
                                conv_sq += c_val * c_val;
                                pred_sq += p_val * p_val;
                            }
                            let conv_norm = conv_sq.sqrt();
                            let pred_norm = pred_sq.sqrt();
                            let residual = if conv_norm > 0.0 {
                                diff_sq.sqrt() / conv_norm
                            } else {
                                f64::NAN
                            };
                            // Δρ-step magnitude the predictor handled
                            // here. Pulled from the OLD IFT cache (the
                            // one the prediction was made against) BEFORE
                            // updatewarm_start_from replaces it. Lets the
                            // bench runner correlate residual quality
                            // with step size — a small step that still
                            // produced a large residual is a stronger
                            // indictment of the linearization than a
                            // large step at the same residual.
                            let drho_norm = match self.ift_warm_start_cache.read() {
                                Ok(guard) => match guard.as_ref() {
                                    Some(c) if c.rho.len() == rho.len() => {
                                        let mut sq = 0.0_f64;
                                        for (n, o) in rho.iter().zip(c.rho.iter()) {
                                            let d = n - o;
                                            sq += d * d;
                                        }
                                        sq.sqrt()
                                    }
                                    _ => f64::NAN,
                                },
                                Err(_) => f64::NAN,
                            };
                            // Hessian conditioning indicator. log|H_pen|
                            // tracks how the penalized curvature matrix
                            // shifts across solves; sudden jumps signal
                            // a flat direction opening up or a near-
                            // singular geometry (the kind that often
                            // precedes a PIRLS failure or a large IFT
                            // residual). Read from the cached factor
                            // when present (commit ec18559d), or NaN
                            // when it hasn't been populated yet (first
                            // solve at this surface).
                            let h_pen_logdet = match self.ift_cached_factor.read() {
                                Ok(guard) => guard.as_ref().map(|f| f.logdet()).unwrap_or(f64::NAN),
                                Err(_) => f64::NAN,
                            };
                            // Route the marker by predictor source so the
                            // bench runner's residual percentile aggregation
                            // is correctly attributed: tangent-line predictions
                            // get [TANGENT-QUALITY] rather than mistakenly
                            // landing in [IFT-QUALITY] stats. The Flat
                            // case is already short-circuited by the
                            // was_noop check above (the source enum is
                            // the single source of truth), so we
                            // shouldn't reach this dispatch with
                            // source=Flat. The Flat arm here is
                            // defensive — the marker name match is
                            // exhaustive because the enum is closed,
                            // and emitting under [IFT-QUALITY] for an
                            // unexpected Flat would surface as a
                            // residual entry that the bench runner
                            // could investigate.
                            let marker = match prediction_source {
                                Some(WarmStartPredictionSource::Ift) => "[IFT-QUALITY]",
                                Some(WarmStartPredictionSource::TangentLine) => "[TANGENT-QUALITY]",
                                Some(WarmStartPredictionSource::Flat) | None => "[IFT-QUALITY]",
                            };
                            log::info!(
                                "{} residual={:.3e} converged_norm={:.3e} predicted_norm={:.3e} drho_norm={:.3e} h_pen_logdet={:.3e} iters={}",
                                marker,
                                residual,
                                conv_norm,
                                pred_norm,
                                drho_norm,
                                h_pen_logdet,
                                pirls_result.iteration,
                            );
                            // Feed the residual back into the adaptive
                            // |Δρ| cap so the next predict call reflects
                            // the empirical faithfulness of the linearization
                            // at this surface's scale. Only stash finite
                            // non-negative values; anything else (NaN
                            // from divide-by-zero, e.g. trivial β) leaves
                            // the cache untouched so the predictor falls
                            // back to its default 2.0 cap.
                            if residual.is_finite() && residual >= 0.0 {
                                self.last_ift_prediction_residual
                                    .store(residual.to_bits(), Ordering::Relaxed);
                            }
                        }
                    }
                    self.updatewarm_start_from(pirls_result.as_ref());
                    // Record the ρ that produced this β so the next call
                    // can extrapolate. Only meaningful after a successful
                    // solve (updatewarm_start_from clears history on
                    // failure); recording here is harmless on failure
                    // because the warm_start_beta will be None.
                    self.record_warm_start_rho(rho);
                    // Inner-PIRLS feedback signal for the adaptive cap
                    // schedule: record actual iteration count and
                    // converged flag. Only written for non-screening
                    // solves so the screening 3-iter cap doesn't
                    // poison the schedule's history.
                    self.last_inner_iters
                        .store(pirls_result.iteration, Ordering::Relaxed);
                    self.last_inner_converged.store(true, Ordering::Relaxed);
                    // Persist the converged λ_LM so the next PIRLS call
                    // at this surface can seed near the right damping.
                    // Only stash finite-positive values; failure modes
                    // (NaN, ≤0) fall through to the cold default.
                    if pirls_result.final_lm_lambda.is_finite()
                        && pirls_result.final_lm_lambda > 0.0
                    {
                        self.last_pirls_lm_lambda
                            .store(pirls_result.final_lm_lambda.to_bits(), Ordering::Relaxed);
                    }
                    // Persist the accepted gain ratio so the cap
                    // schedule can adapt to inner Newton model fidelity
                    // alongside `last_iters` and `last_converged`. The
                    // LM accept-branch invariant (rho > 0 is necessary
                    // for acceptance) means a finite-positive value is
                    // the only "meaningful signal" outcome; anything
                    // else (None, NaN) leaves the NaN sentinel in
                    // place so the schedule falls back to the
                    // default margin.
                    if let Some(rho) = pirls_result.final_accept_rho
                        && rho.is_finite()
                        && rho >= 0.0
                    {
                        self.last_pirls_accept_rho
                            .store(rho.to_bits(), Ordering::Relaxed);
                    }
                    self.store_persistent_warm_start();
                    // Cache only if key is valid (not NaN).
                    if use_cache && let Some(key) = key_opt {
                        self.cache_manager
                            .pirls_cache
                            .write()
                            .unwrap()
                            .insert(key, Arc::new(pirls_result.compact_for_reml_cache()));
                    }
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
            pirls::PirlsStatus::MaxIterationsReached
            | pirls::PirlsStatus::LmStepSearchExhausted => {
                let kind = match pirls_result.status {
                    pirls::PirlsStatus::LmStepSearchExhausted => "LM step search exhausted",
                    _ => "max iterations reached",
                };
                // Seed screening's purpose is to rank candidate seeds by an
                // approximate cost. Requiring full KKT-style convergence under
                // a 3-iteration cap would discard all informative seeds, so
                // when the partial state is finite — objective (deviance +
                // penalty), β coordinates, residual gradient norm, and the
                // gradient-natural-scale (‖score‖ + ‖Sβ‖ + ridge·‖β‖, which
                // covers the penalty and ridge contributions to H and
                // detects state-blowup pathologies upstream of an explicit
                // O(p²) Hessian-eigenspectrum walk) — we surface the result
                // as Ok and let downstream cost computation derive a finite
                // approximate score. The result is not cached, the warm
                // start is not updated, and KKT is not enforced (the actual
                // fit at full inner budget will).
                // Order the predicates so the O(1) scalar finiteness checks
                // run before the O(p) β-vector walk: short-circuit means a
                // pathological partial fit (NaN deviance, NaN
                // gradient_natural_scale, etc.) is rejected without paying
                // the linear scan over coefficients.
                if in_screening
                    && pirls_result.deviance.is_finite()
                    && pirls_result.stable_penalty_term.is_finite()
                    && pirls_result.gradient_natural_scale.is_finite()
                    && pirls_result.lastgradient_norm.is_finite()
                    && pirls_result
                        .beta_transformed
                        .0
                        .iter()
                        .all(|v| v.is_finite())
                {
                    log::debug!(
                        "[seed-screen] partial-fit accepted for ranking: {kind} (|g| {:.3e}, r_g {:.3e}, iter {})",
                        pirls_result.lastgradient_norm,
                        pirls_result.relative_gradient_norm(),
                        pirls_result.iteration
                    );
                    return Ok(pirls_result);
                }
                if in_screening {
                    log::debug!(
                        "[seed-screen] P-IRLS rejected: {kind} (gradient norm {:.3e}, iter {})",
                        pirls_result.lastgradient_norm,
                        pirls_result.iteration
                    );
                } else {
                    log::error!(
                        "P-IRLS could not certify a valid minimum: {kind} (gradient norm {:.3e}, iter {})",
                        pirls_result.lastgradient_norm,
                        pirls_result.iteration
                    );
                    // Adaptive-cap feedback: cap was hit. Geometric
                    // backoff on the next outer iter's cap. Only write
                    // outside screening so the 3-iter screening cap
                    // does not corrupt the production schedule history.
                    self.last_inner_iters
                        .store(pirls_result.iteration, Ordering::Relaxed);
                    self.last_inner_converged.store(false, Ordering::Relaxed);
                    // A failed solve says nothing useful about the
                    // right damping for the next call; reset the LM
                    // hint to "no signal" so the next solve cold-starts.
                    self.last_pirls_lm_lambda.store(0, Ordering::Relaxed);
                    // A failed solve also invalidates the IFT residual
                    // signal — there's no meaningful "predicted vs
                    // converged" datum to feed back. Reset so the next
                    // predict call uses the default 2.0 cap. NaN
                    // sentinel rather than literal 0 — see
                    // `IFT_RESIDUAL_NO_SIGNAL_BITS`.
                    self.last_ift_prediction_residual
                        .store(IFT_RESIDUAL_NO_SIGNAL_BITS, Ordering::Relaxed);
                }
                Err(EstimationError::PirlsDidNotConverge {
                    max_iterations: pirls_result.iteration,
                    last_change: pirls_result.lastgradient_norm,
                })
            }
        }
    }
}

/// Default cap on |Δρ_k| beyond which the IFT linear predictor rejects.
/// Δρ = log(λ_new / λ_old); 2.0 corresponds to a 7.4× change in λ along
/// any single penalty direction — well outside the regime where the local
/// first-order Jacobian dβ/dρ is faithful. The cap is now ADAPTIVE: see
/// `adaptive_ift_max_drho` for the empirical-quality-driven loosening /
/// tightening policy. This default is used when no IFT-quality history
/// is available yet (the first PIRLS solve at a fresh surface).
const IFT_WARM_START_DEFAULT_MAX_DRHO: f64 = 2.0;

/// Adaptive |Δρ| cap for the IFT predictor, driven by the residual of
/// the previous IFT prediction (see `last_ift_prediction_residual`).
///
/// `last_residual = ‖β_converged − β_predicted‖ / ‖β_converged‖`:
/// - `r < 0.01` → linearization was excellent; allow Δρ up to **4.0**
///   (54× λ-step). At this regime the local Jacobian is faithful and
///   tightening to 2.0 leaves performance on the table at biobank Δρ
///   magnitudes (where outer optimizers commonly take ρ-jumps of ~1).
/// - `0.01 ≤ r < 0.05` → very good; modest expansion to **3.0**.
/// - `0.05 ≤ r < 0.20` → ok; default **2.0** (the original constant).
/// - `0.20 ≤ r < 0.50` → marginal; tighten to **1.0** (2.7× λ-step).
/// - `r ≥ 0.50` → poor; tighten to **0.5** (1.6× λ-step). The IFT
///   prediction is not paying off at biobank Δρ; fall back to flat
///   warm-start (β_cur) for any non-trivial outer step.
/// - `None` → no signal yet (first PIRLS solve at this surface) → default 2.0.
///
/// The thresholds are deliberately one-step-per-decade-of-residual so
/// the policy remains stable under noise; small fluctuations in
/// `last_residual` don't whipsaw the cap.
fn adaptive_ift_max_drho(last_residual: Option<f64>) -> f64 {
    let Some(r) = last_residual else {
        return IFT_WARM_START_DEFAULT_MAX_DRHO;
    };
    // Defensive: NaN or negative residuals indicate corrupted state
    // (bit-pattern unpacking from the atomic encountered an unwritten
    // slot, or norm-ratio division produced a non-physical value).
    // Fall back to the documented default rather than committing to
    // a tier based on garbage. Note: INFINITY is not rejected here —
    // it represents a catastrophic prediction (predicted_norm finite
    // but ‖Δβ‖ overflowed), so falling through to the catch-all 0.5
    // (tightest cap) is the right policy.
    if r.is_nan() || r < 0.0 {
        return IFT_WARM_START_DEFAULT_MAX_DRHO;
    }
    match r {
        r if r < 0.01 => 4.0,
        r if r < 0.05 => 3.0,
        r if r < 0.20 => 2.0,
        r if r < 0.50 => 1.0,
        _ => 0.5,
    }
}

/// Default upper cap on the tangent-line predictor's α (extrapolation
/// fraction beyond the previous ρ-step). 1.5 means "we permit at most
/// 50% extrapolation past the last step length"; the original hardcoded
/// constant from commit dcacf9ee. Used when no IFT-quality history is
/// available (typically right after the first successful PIRLS solve
/// at a fresh surface — IFT cache exists but tangent-line history
/// pair is being assembled). Adaptive policy in
/// `adaptive_tangent_alpha_cap` adjusts this based on the IFT
/// residual signal when present.
const TANGENT_ALPHA_DEFAULT_CAP: f64 = 1.5;

/// Adaptive α-cap for the tangent-line predictor, sharing the IFT
/// residual signal as a proxy for "how trustworthy is the local linear
/// approximation". The tangent line IS a local linear approximation
/// (just along the previous ρ-step direction rather than the IFT
/// Jacobian's full direction), so the same residual that gates
/// `adaptive_ift_max_drho` informs the right cap here too:
///
/// - `r < 0.01`  → linearization excellent → α_cap = 2.0 (1.5×
///                  the default; permits a full step beyond the
///                  previous one)
/// - `r < 0.05`  → very good → α_cap = 1.75
/// - `r < 0.20`  → ok → α_cap = 1.5 (the original constant)
/// - `r < 0.50`  → marginal → α_cap = 1.0 (no extrapolation past
///                  the previous step length)
/// - `r ≥ 0.50`  → poor → α_cap = 0.5 (only HALF the previous step;
///                  the linear approximation has been shown to
///                  collapse toward flat warm-start at this surface)
/// - `None` (no signal yet) → default 1.5
///
/// Same tier-stable, monotone-non-increasing-in-residual shape as
/// `adaptive_ift_max_drho`; the two predictors share a single quality
/// signal so their caps move together.
fn adaptive_tangent_alpha_cap(last_residual: Option<f64>) -> f64 {
    let Some(r) = last_residual else {
        return TANGENT_ALPHA_DEFAULT_CAP;
    };
    if r.is_nan() || r < 0.0 {
        return TANGENT_ALPHA_DEFAULT_CAP;
    }
    match r {
        r if r < 0.01 => 2.0,
        r if r < 0.05 => 1.75,
        r if r < 0.20 => 1.5,
        r if r < 0.50 => 1.0,
        _ => 0.5,
    }
}

/// What the IFT predictor's inner computation actually did with the
/// β it returned. Surfaced by the inner predictor so callers don't
/// need to re-derive noop-ness via O(p) array comparison against
/// the cached β. Production callers map this onto the higher-level
/// `WarmStartPredictionSource` enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IftPredictionOutcome {
    /// The predictor computed a real Δβ (`any_active=true`) and the
    /// returned Coefficients differs from the cached β.
    Predicted,
    /// All Δρ_k were below the per-component eps floor, so the
    /// predictor returned the cached β unchanged. The
    /// [IFT-NOOP] marker has already been emitted at the inner
    /// predictor's noop branch; callers should NOT re-emit a
    /// quality probe on the returned β.
    Noop,
}

/// Tag identifying which branch of the warm-start linear-predictor
/// stack actually produced the β returned by
/// `predict_warm_start_beta_with_source`. Used by the caller in
/// `execute_pirls_if_needed` to emit per-predictor quality markers
/// (`[IFT-QUALITY]` vs `[TANGENT-QUALITY]`) so the bench runner's
/// residual percentile aggregations are correctly attributed instead
/// of mistakenly bucketing tangent-line predictions into IFT stats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WarmStartPredictionSource {
    /// The IFT predictor produced a non-identity β (the cache was
    /// populated and Δρ stayed within the adaptive |Δρ| cap).
    Ift,
    /// The tangent-line predictor produced a non-identity β (the IFT
    /// predictor returned None for non-cache reasons, so the
    /// fallback chain reached tangent-line and it accepted).
    TangentLine,
    /// Both predictors fell through (no history, dim mismatch,
    /// large Δρ, non-finite intermediate, etc.) — the returned
    /// β is the cached `warm_start_beta` unchanged. Same outcome
    /// as the original "use last β as-is" warm start.
    Flat,
}

/// Below this magnitude in any Δρ_k, the per-component IFT contribution is
/// numerically negligible and skipped (saves one back-solve per inactive
/// component).
const IFT_WARM_START_DRHO_EPS: f64 = 1e-12;

/// Bit-pattern sentinel used by `RemlObjectiveState::last_ift_prediction_residual`
/// to encode "no signal yet" unambiguously. Decodes via `f64::from_bits`
/// to a quiet NaN — readers detect the sentinel via NaN's
/// self-inequality (NaN.is_finite() == false), so no value-comparison
/// trickery is needed.
///
/// The original `0` sentinel collided with `f64::to_bits(0.0) == 0`,
/// which would have made a true residual of exactly 0 (mathematically
/// possible if β_predicted matched β_converged element-wise) silently
/// indistinguishable from "predictor never reported". Any standard
/// quiet-NaN bit pattern works; we use the one Rust's `f64::NAN`
/// canonicalizes to (mantissa MSB = 1, sign = 0, exponent = all 1s).
///
/// `pub(crate)` so the bridge's `InnerProgressFeedback::snapshot` (in
/// `crate::solver::outer_strategy`) can reference the same constant —
/// both ends of this atomic must use the same sentinel discipline,
/// otherwise a residual-of-zero round-trips correctly through the
/// writer but the reader treats it as "no signal" and silently
/// drops the adaptive cap-margin signal.
pub(crate) const IFT_RESIDUAL_NO_SIGNAL_BITS: u64 = 0x7ff8_0000_0000_0000;

/// Adaptive clamp for the `initial_lm_lambda` warm-start hint passed
/// from `execute_pirls_if_needed` into PIRLS. Selects one of three
/// principled regimes based on the previous PIRLS solve's halving
/// history:
///
/// * Newton-friendly  (last_converged AND last_iters in 1..=2):
///   well-conditioned local geometry; allow the cached λ down to the
///   LM-internal floor. Range: [1e-9, 1e-3].
/// * Hard fit  (NOT last_converged OR last_iters ≥ 10):
///   previous solve hit the cap or needed many iters; preserve the
///   heavy-damping signal up to gradient-descent. Range: [1e-3, 1.0].
/// * Default  (everything else, incl. unset feedback):
///   matches the historical static `[1e-6, 1e-3]` clamp.
///
/// Returns `None` for non-finite or non-positive `cached_lambda`,
/// matching the historical contract that pathological cache entries
/// fall through to the cold default. Also returns `None` if the
/// caller hasn't recorded any feedback yet (`last_iters == 0` AND
/// `!last_converged`) — that combination signals "no signal" via
/// `clear_warm_start_adaptive_signals`, and the cold default is the
/// safer choice than seeding from a stale-but-finite cache slot.
pub(crate) fn adaptive_lm_lambda_hint(
    cached_lambda: f64,
    last_iters: usize,
    last_converged: bool,
) -> Option<f64> {
    if !cached_lambda.is_finite() || cached_lambda <= 0.0 {
        return None;
    }
    if last_iters == 0 && !last_converged {
        return None;
    }
    let (floor, ceiling) = if last_converged && (1..=2).contains(&last_iters) {
        (1e-9_f64, 1e-3_f64)
    } else if !last_converged || last_iters >= 10 {
        (1e-3_f64, 1.0_f64)
    } else {
        (1e-6_f64, 1e-3_f64)
    };
    Some(cached_lambda.clamp(floor, ceiling))
}

/// Variant taking a pre-built `&dyn FactorizedSystem` so the caller can
/// reuse a cached Cholesky factor across successive predict calls. See
/// `RemlState::predict_warm_start_beta_ift_with_outcome` for the cache
/// invalidation invariants. Test-only after the production wrapper was
/// refactored to call `_inner_with_outcome` directly; retained because
/// the negative-test at line ~6911 still verifies the factor argument
/// is actually consumed (vs silently ignored).
#[cfg(test)]
pub(crate) fn predict_warm_start_beta_ift_with_factor(
    cache: &super::IftWarmStartCache,
    canonical_penalties: &[crate::construction::CanonicalPenalty],
    new_rho: &Array1<f64>,
    p: usize,
    last_ift_residual: Option<f64>,
    factor_override: &dyn crate::linalg::matrix::FactorizedSystem,
) -> Option<Coefficients> {
    predict_warm_start_beta_ift_inner_with_outcome(
        cache,
        canonical_penalties,
        new_rho,
        p,
        last_ift_residual,
        Some(factor_override),
    )
    .map(|(coef, _outcome)| coef)
}

fn predict_warm_start_beta_ift_inner_with_outcome(
    cache: &super::IftWarmStartCache,
    canonical_penalties: &[crate::construction::CanonicalPenalty],
    new_rho: &Array1<f64>,
    p: usize,
    last_ift_residual: Option<f64>,
    factor_override: Option<&dyn crate::linalg::matrix::FactorizedSystem>,
) -> Option<(Coefficients, IftPredictionOutcome)> {
    // Cache populated but ρ not yet stamped (happens between
    // updatewarm_start_from and record_warm_start_rho on the first solve).
    // This is the *expected* race window during normal operation — silent
    // None is correct; emitting a marker would just be noise on every
    // first call.
    if cache.rho.is_empty() {
        return None;
    }
    let k = cache.rho.len();
    // Dimension consistency check. A mismatch here is a state-machine
    // bug signal — the cache was populated under a different penalty
    // layout than the predictor is being asked to use. Emit a structured
    // [IFT-REJECTED] marker so the bench runner counts these. If the
    // count is ever non-zero in production, we have a real inconsistency
    // in the cache invalidation chain; reset_surface, link change, and
    // similar should have wiped the cache before the layout shifted.
    if new_rho.len() != k {
        log::info!(
            "[IFT-REJECTED] reason=rho_dim_mismatch new_rho_dim={} cache_rho_dim={}",
            new_rho.len(),
            k,
        );
        return None;
    }
    if canonical_penalties.len() != k {
        log::info!(
            "[IFT-REJECTED] reason=penalty_dim_mismatch penalties_dim={} cache_rho_dim={}",
            canonical_penalties.len(),
            k,
        );
        return None;
    }
    if cache.beta_original.len() != p {
        log::info!(
            "[IFT-REJECTED] reason=beta_dim_mismatch cache_beta_dim={} expected_p={}",
            cache.beta_original.len(),
            p,
        );
        return None;
    }

    // Δρ guard: reject in toto if any single component exceeds the cap.
    // We do not partially clip: the directions we drop matter just as much
    // as the directions we keep, and a clipped predictor is harder to
    // reason about than a clean fallback.
    let mut max_abs_drho = 0.0_f64;
    let drho: Array1<f64> = (0..k)
        .map(|i| {
            let d = new_rho[i] - cache.rho[i];
            if !d.is_finite() {
                return f64::INFINITY;
            }
            if d.abs() > max_abs_drho {
                max_abs_drho = d.abs();
            }
            d
        })
        .collect();
    let max_drho_cap = adaptive_ift_max_drho(last_ift_residual);
    if !max_abs_drho.is_finite() || max_abs_drho > max_drho_cap {
        // Emit a structured reject marker so the bench runner can
        // count predictor rejections alongside accepts (the
        // `[IFT-QUALITY]` markers count only the accepts). The
        // accept/reject ratio at biobank scale tells us how often
        // the warm-start machinery is actually delivering, vs
        // falling through to flat warm-start at every accepted
        // outer iter.
        log::info!(
            "[IFT-REJECTED] reason=large_drho max_drho={:.3e} cap={:.3e} drho_dim={}",
            max_abs_drho,
            max_drho_cap,
            k,
        );
        return None;
    }

    // Build Σ_k Δρ_k · e^{ρ_cur_k} · S_k · (β_cur-μ_k) in the ORIGINAL basis.
    // Aggregating into a single rhs lets us factor H once and back-solve
    // exactly once, instead of k times. Mathematically:
    //   Σ_k Δρ_k H^{-1} v_k = H^{-1} (Σ_k Δρ_k v_k)
    // since H^{-1} is linear in its rhs.
    let beta_cur = &cache.beta_original;
    let mut rhs_original = Array1::<f64>::zeros(p);
    let mut any_active = false;
    // Use the precomputed `S_k · (β_cur-μ_k)` blocks (cache write-time
    // hook) when available. Falls back to recomputing the local
    // mat-vec when the cache predates this commit's writer hook
    // (None) or the precomputation length disagrees with the
    // current canonical_penalties (defensive: guards against
    // racing replace events).
    let precomputed_ok = match &cache.lambda_s_beta_blocks {
        Some(blocks) => blocks.len() == canonical_penalties.len(),
        None => false,
    };
    for (idx, cp) in canonical_penalties.iter().enumerate() {
        let dr = drho[idx];
        if dr.abs() <= IFT_WARM_START_DRHO_EPS {
            continue;
        }
        any_active = true;
        let r = &cp.col_range;
        // S_k · (β_cur-μ_k) is block-local: only β_cur[r] and μ_k contribute,
        // and the result lives in r as well.
        let scale = dr * cache.rho[idx].exp();
        let mut rhs_slice = rhs_original.slice_mut(s![r.start..r.end]);
        if precomputed_ok {
            // SAFE: precomputed_ok guarantees Some + length match.
            let blocks = cache.lambda_s_beta_blocks.as_ref().unwrap();
            let sb_block = &blocks[idx];
            // Defensive: a fresh canonical_penalty may have a different
            // col_range size than the cache's precomputation captured.
            // If so, fall through to the recompute path for this
            // index. (Should not happen in practice — the cache is
            // invalidated whenever the penalty layout changes.)
            if sb_block.len() == rhs_slice.len() {
                for (target, src) in rhs_slice.iter_mut().zip(sb_block.iter()) {
                    *target += scale * *src;
                }
                continue;
            }
        }
        let beta_block = beta_cur.slice(s![r.start..r.end]);
        let centered = &beta_block - &cp.prior_mean;
        let sb_block = cp.local.dot(&centered);
        for (target, src) in rhs_slice.iter_mut().zip(sb_block.iter()) {
            *target += scale * *src;
        }
    }

    if !any_active {
        // All Δρ are below the eps floor — predictor reduces to identity.
        // Returning the cached β here is equivalent to the flat warm-start
        // and slightly better than the tangent-line predictor (which the
        // caller would otherwise try), since the IFT cache reflects a
        // genuinely converged solve.
        //
        // Emit a structured no-op marker so the bench runner's accept /
        // reject ratio (commit fec27c97) can distinguish:
        //   accept  → predictor produced a non-identity β prediction
        //   reject  → predictor fell through to caller's tangent-line / flat
        //   noop    → predictor returned identity (this branch). The outer
        //             made an effectively-zero ρ-step; warm-start was free.
        // Without this marker, "noop" calls would inflate the accept count
        // because the predictor returns Some(β) — masking how often the
        // outer is actually exercising the linearization.
        log::info!(
            "[IFT-NOOP] reason=all_drho_below_eps max_drho={:.3e} drho_dim={}",
            max_abs_drho,
            k,
        );
        return Some((
            Coefficients::new(beta_cur.clone()),
            IftPredictionOutcome::Noop,
        ));
    }

    if !rhs_original.iter().all(|v| v.is_finite()) {
        log::info!(
            "[IFT-REJECTED] reason=non_finite_rhs max_drho={:.3e} drho_dim={}",
            max_abs_drho,
            k,
        );
        return None;
    }

    // Convert rhs to the basis H_pen lives in, solve, convert back.
    let solve_in_original = cache.frame_was_original;
    let rhs_in_h_basis = if solve_in_original {
        rhs_original
    } else {
        // rhs_tfd = qs^T · rhs_original. Dimension check: qs is p×p.
        if cache.qs.nrows() != p || cache.qs.ncols() != p {
            log::info!(
                "[IFT-REJECTED] reason=qs_dim_mismatch qs_dim={}x{} expected_p={}",
                cache.qs.nrows(),
                cache.qs.ncols(),
                p,
            );
            return None;
        }
        cache.qs.t().dot(&rhs_original)
    };

    // Use the caller's pre-built factor when present (the method-level
    // wrapper threads in a cached Cholesky); otherwise factorize inline.
    let owned_factor;
    let factor_ref: &dyn crate::linalg::matrix::FactorizedSystem = match factor_override {
        Some(f) => f,
        None => {
            owned_factor = match cache.penalized_hessian_transformed.factorize() {
                Ok(f) => f,
                Err(_) => {
                    log::info!(
                        "[IFT-REJECTED] reason=hessian_factorize_failed max_drho={:.3e} drho_dim={}",
                        max_abs_drho,
                        k,
                    );
                    return None;
                }
            };
            owned_factor.as_ref()
        }
    };
    let solution_in_h_basis = match factor_ref.solve(&rhs_in_h_basis) {
        Ok(u) => u,
        Err(_) => {
            log::info!(
                "[IFT-REJECTED] reason=hessian_solve_failed max_drho={:.3e} drho_dim={}",
                max_abs_drho,
                k,
            );
            return None;
        }
    };
    let solution_original = if solve_in_original {
        solution_in_h_basis
    } else {
        cache.qs.dot(&solution_in_h_basis)
    };

    if !solution_original.iter().all(|v| v.is_finite()) {
        log::info!(
            "[IFT-REJECTED] reason=non_finite_solution max_drho={:.3e} drho_dim={}",
            max_abs_drho,
            k,
        );
        return None;
    }

    // β_predict = β_cur − H^{-1} · (Σ_k Δρ_k e^{ρ_k} S_k(β_cur-μ_k)).
    // (The sign convention: dβ/dρ_k = −H^{-1}(e^{ρ_k}S_k(β-μ_k)), so
    //  Δβ = −Σ_k Δρ_k H^{-1}(e^{ρ_k}S_k(β-μ_k)) = −solution_original.)
    let mut predicted = beta_cur.clone();
    for (target, &correction) in predicted.iter_mut().zip(solution_original.iter()) {
        *target -= correction;
    }
    if !predicted.iter().all(|v| v.is_finite()) {
        log::info!(
            "[IFT-REJECTED] reason=non_finite_predicted max_drho={:.3e} drho_dim={}",
            max_abs_drho,
            k,
        );
        return None;
    }
    log::debug!(
        "[warm-start] IFT prediction: max|Δρ|={:.3e}, ‖rhs‖={:.3e}, ‖Δβ‖={:.3e}",
        max_abs_drho,
        rhs_in_h_basis.dot(&rhs_in_h_basis).sqrt(),
        solution_original.dot(&solution_original).sqrt(),
    );
    Some((
        Coefficients::new(predicted),
        IftPredictionOutcome::Predicted,
    ))
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
            log::debug!(
                "[REML] eval#{} begin cost-only | rho[..4]=[{}] | k={}",
                cost_call_idx,
                prefix.join(","),
                p.len()
            );
        }
        let rho_key = EvalCacheManager::sanitized_rhokey(p);
        if let Some(eval) = self.cache_manager.cached_outer_eval(&rho_key) {
            log::debug!(
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
                log::debug!(
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
                log::debug!(
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
        log::debug!(
            "[REML] eval#{} pirls done | elapsed {:.1}ms | backend {:?}",
            cost_call_idx,
            pirls_ms,
            bundle.backend_kind()
        );
        if bundle.backend_kind() == GeometryBackendKind::SparseExactSpd {
            let t_assemble = std::time::Instant::now();
            let result =
                self.evaluate_unified_sparse(p, &bundle, super::unified::EvalMode::ValueOnly)?;
            let cost = screening_residual_penalty(result.cost, bundle.pirls_result.as_ref());
            log::debug!(
                "[REML] eval#{} sparse cost {:.6e} | assemble {:.1}ms | total {:.1}ms",
                cost_call_idx,
                cost,
                t_assemble.elapsed().as_secs_f64() * 1000.0,
                t_eval_start.elapsed().as_secs_f64() * 1000.0
            );
            return Ok(cost);
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
            // Hot diagnostics walk the Hessian eigenspectrum and emit
            // ill-conditioning warnings. They are only meaningful for fully
            // converged inner modes — partial fits accepted from seed
            // screening (cap=3) routinely yield indefinite Hessians, and
            // surfacing those as `Penalized Hessian not PD` warnings would
            // confuse log readers and mask real production-fit issues.
            let want_hot_diag = !pirls_result.status.is_failed_max_iterations()
                && self.should_compute_hot_diagnostics(cost_call_idx);
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
        let cost = screening_residual_penalty(result.cost, bundle.pirls_result.as_ref());
        log::debug!(
            "[REML] eval#{} dense cost {:.6e} | assemble {:.1}ms | total {:.1}ms",
            cost_call_idx,
            cost,
            t_assemble.elapsed().as_secs_f64() * 1000.0,
            t_eval_start.elapsed().as_secs_f64() * 1000.0
        );
        Ok(cost)
    }

    /// Seed-screening ranking proxy.
    ///
    /// In screening mode (`screening_max_inner_iterations > 0`), runs the
    /// inner P-IRLS solve under the active iteration cap and returns the
    /// minimum penalized deviance (`-2·log L + βᵀSβ`, plus the structural
    /// ridge contribution carried in `stable_penalty_term`) observed across
    /// all inner iterations. Penalized deviance descends monotonically along
    /// any inner descent path P-IRLS takes, so this minimum is a meaningful
    /// quality signal even when the inner solver was capped before reaching
    /// the mode — strictly better than ranking by the partial-fit V_LAML
    /// criterion, whose `0.5·log|H|` term is dominated by noise at a
    /// poorly-conditioned partial β̂.
    ///
    /// Outside of screening mode this delegates to [`Self::compute_cost`] so
    /// the optimization objective itself is never changed by this method's
    /// presence.
    pub fn compute_screening_proxy(&self, p: &Array1<f64>) -> Result<f64, EstimationError> {
        let in_screening = self.screening_max_inner_iterations.load(Ordering::Relaxed) > 0;
        if !in_screening {
            return self.compute_cost(p);
        }
        // Use the same bundle pipeline as `compute_cost`, but read the proxy
        // directly from `pirls_result.min_penalized_deviance` instead of
        // assembling the LAML criterion. Bundle assembly already runs
        // P-IRLS under the screening cap; reusing it keeps the screening
        // path's behavioural invariants (no warm-start update, no LRU write,
        // no KKT enforcement at partial fits).
        let bundle = match self.obtain_eval_bundle(p) {
            Ok(bundle) => bundle,
            Err(EstimationError::ModelIsIllConditioned { .. })
            | Err(EstimationError::PerfectSeparationDetected { .. })
            | Err(EstimationError::PirlsDidNotConverge { .. }) => {
                self.cache_manager.invalidate_eval_bundle();
                return Ok(f64::INFINITY);
            }
            Err(e) => {
                self.cache_manager.invalidate_eval_bundle();
                return Err(e);
            }
        };
        let proxy = bundle.pirls_result.min_penalized_deviance;
        if proxy.is_finite() {
            Ok(proxy)
        } else {
            Ok(f64::INFINITY)
        }
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
    ///   = -0.5 * exp(ρ_k) * E_{π(β|y,ρ)}[ (β-μ_k)ᵀ S_k (β-μ_k) ].
    ///
    /// Laplace bridge to implemented terms:
    /// - If π(β|y,ρ) is approximated locally by N(β̂, H^{-1}), then
    ///     E[(β-μ_k)ᵀ S_k(β-μ_k)] ≈ (β̂-μ_k)ᵀ S_k(β̂-μ_k) + tr(H^{-1} S_k),
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
    /// - `bundle`: the evaluation bundle (carries h_total, firth operator, etc.)
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
                    hessian_weights: pirls_result.finalweights.clone(),
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
                            hessian_weights: pirls_result.finalweights.clone(),
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
            rho_prior: self.rho_prior.clone(),
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
            kkt_residual: None,
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
        // is both unphysical and inconsistent with the analytic subspace
        // derivatives.
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
        //
        // Previously this kernel was only constructed when
        // `penalty_rank < h_for_operator.ncols()` — i.e. only when the
        // penalty matrix itself was structurally rank-deficient. That
        // condition is wrong for non-Gaussian families: the leakage
        // described above is driven by the third-derivative correction
        // `D_β H[v]`, not by the rank of `S`. Even when `S_λ` has full
        // rank (penalty_rank == p_total), the correction term still spills
        // onto `null(S_λ_term)` for individual penalty components `S_k`
        // whose ranges don't cover all of `range(S_λ)`. The result is a
        // spurious nonzero outer gradient at large `λ_k` (where the
        // unbalanced trace term dominates the `λ_k tr(S_λ⁺ S_k)` and
        // `λ_k(β-μ_k)'S_k(β-μ_k)` terms that should cancel to zero), which makes
        // ARC's deterministic-replay detector fire on a flat REML surface
        // and surfaces "SurvivalLocationScaleFamily expects 3 blocks,
        // got 0" panics in the downstream refit path.
        //
        // Gate on `c_nontrivial`: only build the projected kernel when
        // the IRLS cubic coefficient `c = w'/(2η')` has nonzero support,
        // i.e. when `D_β H[v] = X' diag(c ⊙ X v) X` can leak onto
        // `null(S)`. For canonical Gaussian (Identity link) the IRLS
        // weights `W = 1/σ²` are η-independent so `c ≡ 0` and
        // `D_β H ≡ 0` — there is no leakage to project away. Activating
        // the projection there would silently switch the cost identity
        // from `log|H|` to `log|H_proj|`, putting the analytic ψ-gradient
        // on a different cost surface than a centered finite difference
        // of the cost (the missing `dU_S/dψ` contribution shows up as a
        // ~6e-3 rel error in the Gaussian identity projection test).
        // Skipping the projection
        // when `c ≡ 0` preserves the classical Gaussian REML cost identity
        // (`log|H|`) and keeps that test on its analytic match.  For
        // every c-nontrivial family (Probit / Logit / cloglog / Poisson /
        // Gamma / SAS / GAMLSS noise blocks / etc.) the projection is
        // still built unconditionally — that is the rank-deficient LAML
        // fix the comment above motivates.
        let c_nontrivial = pirls_result.solve_c_array.iter().any(|&c| c != 0.0);
        let (hessian_logdet_correction, penalty_subspace_trace) =
            if matches!(hessian_mode, PseudoLogdetMode::Smooth) && c_nontrivial {
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
        //
        // The kernel is built when (a) the spectral mode is `Smooth` and
        // (b) the IRLS cubic coefficient `c = w'/(2η')` has nontrivial
        // support — i.e. the family's `D_β H[v] = X' diag(c ⊙ X v) X` can
        // actually leak onto `null(S)`.  An earlier version of this path
        // gated only on `penalty_rank < h_total_original.ncols()`, which
        // missed `c`-nontrivial-but-full-rank designs.  A subsequent
        // iteration removed the gate entirely, which over-projected
        // canonical Gaussian (Identity link, `c ≡ 0`): the cost identity
        // then silently shifts from `log|H|` to `log|H_proj|`, and the
        // analytic ψ-gradient (still computed via `K · op_total`) no
        // longer matches a centered finite difference of the cost within
        // the projection's numerical roundoff — surfacing as a ~6e-3 rel error in
        // the Gaussian identity projection test. The `c_nontrivial`
        // gate keeps the rank-deficient LAML fix active for every
        // non-Gaussian-Identity family (where the leakage is real and
        // the projected kernel is the only formula that matches the
        // cost identity — pinned by
        // `iso_kappa_duchon_penalty_subspace_projection_pins_trace`)
        // while keeping the classical `log|H|` cost identity for
        // canonical Gaussian, where the leakage is identically zero.
        let c_nontrivial = pirls_result.solve_c_array.iter().any(|&c| c != 0.0);
        let (hessian_logdet_correction, penalty_subspace_trace) =
            if matches!(hessian_mode, PseudoLogdetMode::Smooth) && c_nontrivial {
                use super::unified::HessianOperator;
                let qs = &pirls_result.reparam_result.qs;
                let h_transformed = crate::faer_ndarray::fast_ab(
                    &crate::faer_ndarray::fast_atb(qs, &h_total_original),
                    qs,
                );
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
            |r| self.compute_soft_priorcost(r) + self.compute_configured_rho_prior_cost(r),
            |r| self.compute_soft_priorgrad(r) + &self.compute_configured_rho_prior_grad(r),
            |r| {
                let mut hess = self
                    .compute_soft_priorhess(r)
                    .unwrap_or_else(|| Array2::<f64>::zeros((r.len(), r.len())));
                if let Some(configured) = self.compute_configured_rho_prior_hess(r) {
                    hess += &configured;
                }
                if hess.iter().any(|&v| v != 0.0) {
                    Some(hess)
                } else {
                    None
                }
            },
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
        // Always evaluate cost + gradient: the EFS update uses the
        // universal-form `Δρ = log(1 − 2·g_full / q_eff)` for ρ and τ
        // coordinates, which folds Tierney–Kadane corrections, smoothing-
        // parameter priors, Firth bias-reduction, monotonicity barriers,
        // and SAS log-δ ridge contributions into the multiplicative
        // target through their gradient channel. Without the gradient
        // the EFS step targets the wrong stationarity equation whenever
        // any of those terms are active.
        let eval_mode = super::unified::EvalMode::ValueAndGradient;
        self.validate_tk_ext_coords(eval_mode, &assembly.ext_coords)?;
        let tk_terms = self.tierney_kadane_terms(rho, bundle, eval_mode, &assembly.ext_coords)?;
        let inner_solution = assembly.build();
        let inner_hessian_scale =
            super::unified::hessian_operator_geometric_scale(inner_solution.hessian_op.as_ref());

        let prior = self.build_prior(rho, eval_mode);
        let cost_result = super::assembly::evaluate_solution(
            &inner_solution,
            rho.as_slice().unwrap(),
            eval_mode,
            prior,
        )
        .map_err(EstimationError::InvalidInput)?;
        let cost_result = self.apply_tk_to_result(cost_result, tk_terms)?;
        let gradient = cost_result
            .gradient
            .as_ref()
            .expect("EFS requires gradient (ValueAndGradient mode should have computed it)");

        let efs_eval = if has_psi {
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
                inner_hessian_scale,
            }
        } else {
            let steps = compute_efs_update(
                &inner_solution,
                rho.as_slice().unwrap(),
                gradient.as_slice().unwrap(),
            );
            crate::solver::outer_strategy::EfsEval {
                cost: cost_result.cost,
                steps,
                beta: Some(beta_for_barrier),
                psi_gradient: None,
                psi_indices: None,
                inner_hessian_scale,
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

        log::debug!(
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
            log::debug!(
                "[REML] grad-only begin | rho[..4]=[{}] | k={}",
                prefix.join(","),
                p.len()
            );
        }
        let rho_key = EvalCacheManager::sanitized_rhokey(p);
        if let Some(eval) = self.cache_manager.cached_outer_eval(&rho_key) {
            let gnorm = eval.gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
            log::debug!(
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
        log::debug!(
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
            log::debug!(
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
        log::debug!(
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
            log::debug!(
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
                log::debug!(
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
                log::debug!(
                    "P-IRLS flagged ill-conditioning for current rho; returning infeasible outer eval to retreat."
                );
                return Ok(OuterEval::infeasible(p.len()));
            }
            Err(EstimationError::PerfectSeparationDetected { .. })
            | Err(EstimationError::PirlsDidNotConverge { .. }) => {
                self.cache_manager.invalidate_eval_bundle();
                log::debug!(
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
        log::debug!(
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
            inner_beta_hint: None,
        };
        {
            let gnorm = eval.gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
            log::debug!(
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
        let ext_dim = ext_coords.len();
        let p_dim = ext_coords.first().map(|coord| coord.g.len()).unwrap_or(0);
        assembly.ext_coords = ext_coords;
        if mode == super::unified::EvalMode::ValueGradientHessian {
            // Link-shape parameters do not directly differentiate the penalty
            // matrices, so their explicit rho-link second partials are zero.
            // Installing this callback is still required: the unified Hessian
            // builder then evaluates the nonzero profiled cross terms from the
            // first-order link drifts and IFT mode responses.
            assembly.rho_ext_pair_fn = Some(Box::new(move |_, _| super::unified::HyperCoordPair {
                a: 0.0,
                g: Array1::zeros(p_dim),
                b_mat: Array2::zeros((p_dim, p_dim)),
                b_operator: None,
                ld_s: 0.0,
            }));
            assembly.ext_coord_pair_fn =
                Some(Box::new(move |_, _| super::unified::HyperCoordPair {
                    a: 0.0,
                    g: Array1::zeros(p_dim),
                    b_mat: Array2::zeros((p_dim, p_dim)),
                    b_operator: None,
                    ld_s: 0.0,
                }));
            debug_assert!(ext_dim > 0);
        }
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

fn positive_penalty_rank_and_logdet(eigenvalues: &[f64]) -> (usize, f64) {
    let threshold = super::unified::positive_eigenvalue_threshold(eigenvalues);
    let rank = eigenvalues.iter().filter(|&&ev| ev > threshold).count();
    let log_det = super::unified::exact_pseudo_logdet(eigenvalues, threshold);
    (rank, log_det)
}

#[cfg(test)]
mod tk_math_tests {
    use super::*;
    use crate::faer_ndarray::FaerCholesky;
    use crate::mixture_link::{
        beta_logistic_inverse_link_jet, beta_logistic_inverse_link_pdffourth_derivative,
        beta_logistic_inverse_link_pdfthird_derivative, inverse_link_jet_for_inverse_link,
        inverse_link_pdffourth_derivative_for_inverse_link,
        inverse_link_pdfthird_derivative_for_inverse_link, mixture_inverse_link_jet,
        sas_inverse_link_jet, sas_inverse_link_pdffourth_derivative,
        sas_inverse_link_pdfthird_derivative, state_fromspec,
    };
    use crate::pirls::{VarianceJet, e_obs_from_jets};
    use crate::types::{LinkComponent, MixtureLinkSpec};
    use faer::Side;
    use ndarray::array;
    use num_dual::{Dual3_64, Dual64, DualNum, third_derivative};

    #[test]
    fn penalty_rank_uses_actual_positive_eigenspace_not_root_rows() {
        let e = array![[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0],];
        let s = e.t().dot(&e);
        let (evals, _) = s.eigh(Side::Lower).expect("penalty eigensystem");
        let (rank, log_det) = positive_penalty_rank_and_logdet(evals.as_slice().unwrap());

        assert_eq!(rank, 1);
        assert!(
            (log_det - 2.0_f64.ln()).abs() < 1e-12,
            "logdet should use the single positive eigenvalue, got {log_det}"
        );
    }

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
    /// from the dual part.
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
    /// scalar in [`tk_scalar_dual`].  The setup uses p = 1 so the matrix solve
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

    // ---------------------------------------------------------------------
    //  Analytic e_i = ∂³W_obs/∂η³ vs. Dual3 AD across non-Logit links.
    //
    //  These tests verify the general observed-Hessian path that
    //  `RemlState::hessian_cde_arrays` invokes for every non-canonical link
    //  (Probit, CLogLog, SAS, BetaLogistic, Mixture).  The reference is
    //  `num_dual::third_derivative`, which evaluates W_obs(η₀+δ) in
    //  third-order dual arithmetic and extracts ∂³W_obs/∂δ³ at δ=0
    //  symbolically.
    //
    //  The trick is that Dual3 only carries (re, v1, v2, v3) so any
    //  Taylor truncation of μ to order δ³ is propagated *exactly* through
    //  Dual3 arithmetic.  We therefore parametrise μ(δ), μ'(δ), μ''(δ)
    //  as length-(4..6) Taylor polynomials whose coefficients come from
    //  the *independently derived* closed-form helpers in
    //  `mixture_link.rs` (μ⁽ᵏ⁾(η₀) for k ∈ 0..5).  This makes the test
    //  link-agnostic: the same AD machinery validates the polynomial
    //  formula in `pirls::e_obs_from_jets` for every link.
    // ---------------------------------------------------------------------

    /// Build a Dual3 polynomial in δ representing
    ///   μ(η₀ + δ) ≈ Σ_{k=0..5} μ⁽ᵏ⁾(η₀)·δᵏ/k!
    /// truncated at δ³ – Dual3's arithmetic discards higher-order terms
    /// anyway, so the propagated (re, v1, v2, v3) match the analytic
    /// values μ⁽⁰⁾..μ⁽³⁾(η₀) exactly.
    fn taylor_mu_dual3(delta: Dual3_64, h0: f64, h1: f64, h2: f64, h3: f64) -> Dual3_64 {
        let inv2 = 0.5_f64;
        let inv6 = 1.0_f64 / 6.0_f64;
        let d2 = delta * delta;
        let d3 = d2 * delta;
        Dual3_64::from_re(h0) + delta * h1 + d2 * (h2 * inv2) + d3 * (h3 * inv6)
    }

    /// Same Taylor expansion as [`taylor_mu_dual3`] but starting one order
    /// later in the η-jet, so that the resulting Dual3 represents
    ///   ∂μ/∂η at η₀+δ ≈ Σ_{k=0..4} μ⁽ᵏ⁺¹⁾(η₀)·δᵏ/k!
    /// (used as the closed-form `h1(δ)` carrier inside W_obs).
    fn taylor_mu_eta_dual3(delta: Dual3_64, h1: f64, h2: f64, h3: f64, h4: f64) -> Dual3_64 {
        let inv2 = 0.5_f64;
        let inv6 = 1.0_f64 / 6.0_f64;
        let d2 = delta * delta;
        let d3 = d2 * delta;
        Dual3_64::from_re(h1) + delta * h2 + d2 * (h3 * inv2) + d3 * (h4 * inv6)
    }

    /// Closed-form `h2(δ)` carrier (= ∂²μ/∂η² at η₀+δ) as a Dual3 Taylor
    /// polynomial – picks up `h5` in its δ³ coefficient.
    fn taylor_mu_etaeta_dual3(delta: Dual3_64, h2: f64, h3: f64, h4: f64, h5: f64) -> Dual3_64 {
        let inv2 = 0.5_f64;
        let inv6 = 1.0_f64 / 6.0_f64;
        let d2 = delta * delta;
        let d3 = d2 * delta;
        Dual3_64::from_re(h2) + delta * h3 + d2 * (h4 * inv2) + d3 * (h5 * inv6)
    }

    /// Closed-form W_obs for a binomial likelihood with arbitrary link:
    ///   W_obs(η) = h₁²/(φV) − (y−μ)·(h₂·V − h₁²·V_μ)/(φV²)
    /// where V(μ)=μ(1−μ) and V_μ(μ)=1−2μ.  Built entirely from the three
    /// Taylor carriers (mu, mu_eta, mu_etaeta) using only num_dual ops.
    fn binomial_w_obs_dual3(
        mu: Dual3_64,
        mu_eta: Dual3_64,
        mu_etaeta: Dual3_64,
        y: f64,
        phi: f64,
    ) -> Dual3_64 {
        let one = Dual3_64::from_re(1.0);
        let two = Dual3_64::from_re(2.0);
        let v = mu * (one - mu);
        let v_mu = one - two * mu;
        let h1sq = mu_eta * mu_eta;
        // φ enters as a constant scalar — fold it into the existing
        // Dual3 multiplications via `Mul<F>` instead of allocating a new
        // Dual3 carrier (mathematically identical, just lighter).
        let phi_v = v * phi;
        let fisher = h1sq / phi_v;
        let resid = Dual3_64::from_re(y) - mu;
        let t_prime = (mu_etaeta * v - h1sq * v_mu) / (phi_v * v);
        fisher - resid * t_prime
    }

    /// Drive `pirls::e_obs_from_jets` and Dual3 AD for the same scalar
    /// fixture and assert the analytic 3rd-η-derivative of W_obs matches
    /// the AD reference to floating-point tolerance.
    fn assert_e_obs_matches_dual3_ad(
        link_label: &str,
        eta0: f64,
        y: f64,
        h0: f64,
        h1: f64,
        h2: f64,
        h3: f64,
        h4: f64,
        h5: f64,
        phi: f64,
        tol: f64,
    ) {
        let v = h0 * (1.0 - h0);
        let vj = VarianceJet::bernoulli(h0);
        assert!(
            v.is_finite() && v > 0.0,
            "fixture {link_label} (eta={eta0}) has degenerate V(μ)={v}; pick a non-saturated point"
        );
        let analytic = e_obs_from_jets(y, h0, h1, h2, h3, h4, h5, vj, phi, 1.0);

        let (_, _, _, d3_w) = third_derivative(
            |delta| {
                let mu = taylor_mu_dual3(delta, h0, h1, h2, h3);
                let mu_eta = taylor_mu_eta_dual3(delta, h1, h2, h3, h4);
                let mu_etaeta = taylor_mu_etaeta_dual3(delta, h2, h3, h4, h5);
                binomial_w_obs_dual3(mu, mu_eta, mu_etaeta, y, phi)
            },
            0.0_f64,
        );

        let scale = analytic.abs().max(d3_w.abs()).max(1.0e-12);
        let rel = (analytic - d3_w).abs() / scale;
        assert!(
            rel < tol,
            "{link_label}: analytic e_obs ({analytic:.12e}) does not match Dual3 AD ({d3_w:.12e}) at eta={eta0}, y={y}; rel={rel:.3e}"
        );
    }

    /// Probit + Bernoulli: closed-form e_obs (via `pirls::e_obs_from_jets`)
    /// matches the Dual3-AD third η-derivative of W_obs at every fixture
    /// point.  This is the production link for biobank-scale GAMLSS
    /// marginal-slope inference, so we cover both shoulders and the
    /// origin to exercise the (η³,η²,η) polynomial structure of the
    /// Probit jets.
    #[test]
    fn e_obs_from_jets_matches_dual3_ad_probit_bernoulli() {
        let etas = [-1.4_f64, -0.4, 0.0, 0.6, 1.3];
        let ys = [0.0_f64, 1.0];
        let probit = InverseLink::Standard(LinkFunction::Probit);
        for &eta in &etas {
            let jet = inverse_link_jet_for_inverse_link(&probit, eta).expect("probit jet");
            let h4 =
                inverse_link_pdfthird_derivative_for_inverse_link(&probit, eta).expect("probit h4");
            let h5 = inverse_link_pdffourth_derivative_for_inverse_link(&probit, eta)
                .expect("probit h5");
            for &y in &ys {
                assert_e_obs_matches_dual3_ad(
                    "probit", eta, y, jet.mu, jet.d1, jet.d2, jet.d3, h4, h5, 1.0, 1.0e-9,
                );
            }
        }
    }

    /// CLogLog + Bernoulli: same AD vs analytic check.  CLogLog is the
    /// canonical hazard-link for survival models so its W_obs has a
    /// non-trivial residual term that exercises the full (h₁..h₅,
    /// V₀..V₂) polynomial composition inside `e_obs_from_jets`.
    #[test]
    fn e_obs_from_jets_matches_dual3_ad_cloglog_bernoulli() {
        let etas = [-1.6_f64, -0.5, 0.0, 0.4, 1.2];
        let ys = [0.0_f64, 1.0];
        let cloglog = InverseLink::Standard(LinkFunction::CLogLog);
        for &eta in &etas {
            let jet = inverse_link_jet_for_inverse_link(&cloglog, eta).expect("cloglog jet");
            let h4 = inverse_link_pdfthird_derivative_for_inverse_link(&cloglog, eta)
                .expect("cloglog h4");
            let h5 = inverse_link_pdffourth_derivative_for_inverse_link(&cloglog, eta)
                .expect("cloglog h5");
            for &y in &ys {
                assert_e_obs_matches_dual3_ad(
                    "cloglog", eta, y, jet.mu, jet.d1, jet.d2, jet.d3, h4, h5, 1.0, 1.0e-9,
                );
            }
        }
    }

    /// Logit + Bernoulli (canonical fast-path consistency check):
    /// `hessian_cde_arrays` ships canonical Logit through a dedicated
    /// fast path that returns `pw·μ⁽⁴⁾(η)` (since W = h₁ for the
    /// canonical pairing).  Independently, the *general* observed-Hessian
    /// path in `pirls::e_obs_from_jets` must produce the identical value
    /// when evaluated on the same Logit jets, otherwise the two paths
    /// would disagree on outer-gradient propagation.  We additionally
    /// sanity-check both quantities against Dual3 AD.
    #[test]
    fn e_obs_from_jets_matches_dual3_ad_logit_bernoulli_canonical_consistency() {
        let etas = [-1.7_f64, -0.6, 0.0, 0.5, 1.4];
        let ys = [0.0_f64, 1.0];
        let logit = InverseLink::Standard(LinkFunction::Logit);
        for &eta in &etas {
            let jet = inverse_link_jet_for_inverse_link(&logit, eta).expect("logit jet");
            let h4 =
                inverse_link_pdfthird_derivative_for_inverse_link(&logit, eta).expect("logit h4");
            let h5 =
                inverse_link_pdffourth_derivative_for_inverse_link(&logit, eta).expect("logit h5");

            // The (y−μ) residual term in the general formula must drop
            // out when y is replaced by μ; pick y = μ to isolate
            // W_F = h₁²/V = h₁ on the Logit canonical pairing and confirm
            // that path returns h⁽⁴⁾, which is exactly the value the
            // canonical fast-path stores in `e_array`.
            let canonical_fast_path = h4; // W = h₁ ⇒ ∂³W/∂η³ = h⁽⁴⁾.
            let general_at_y_eq_mu = e_obs_from_jets(
                jet.mu,
                jet.mu,
                jet.d1,
                jet.d2,
                jet.d3,
                h4,
                h5,
                VarianceJet::bernoulli(jet.mu),
                1.0,
                1.0,
            );
            let rel_fast = (general_at_y_eq_mu - canonical_fast_path).abs()
                / canonical_fast_path.abs().max(1.0e-12);
            assert!(
                rel_fast < 1.0e-10,
                "Logit canonical fast path ({canonical_fast_path:.12e}) does not match general e_obs at y=μ ({general_at_y_eq_mu:.12e}) at eta={eta}; rel={rel_fast:.3e}"
            );

            for &y in &ys {
                assert_e_obs_matches_dual3_ad(
                    "logit", eta, y, jet.mu, jet.d1, jet.d2, jet.d3, h4, h5, 1.0, 1.0e-9,
                );
            }
        }
    }

    /// SAS + Bernoulli: validate `e_obs_from_jets` against Dual3 AD using
    /// the closed-form SAS jets `(μ, h₁..h₃)` from `sas_inverse_link_jet`
    /// plus the analytic `h₄, h₅` helpers.  Two SAS shape parameters are
    /// covered to exercise both the symmetric (ε≈0) and skewed regimes.
    #[test]
    fn e_obs_from_jets_matches_dual3_ad_sas_bernoulli() {
        let etas = [-1.1_f64, -0.3, 0.0, 0.5, 1.0];
        let ys = [0.0_f64, 1.0];
        let configs = [(-0.25_f64, 0.35_f64), (0.4_f64, -0.2_f64)];
        for &(epsilon, log_delta) in &configs {
            let state = SasLinkState::new(epsilon, log_delta).expect("sas state");
            let link = InverseLink::Sas(state);
            for &eta in &etas {
                let jet = sas_inverse_link_jet(eta, state.epsilon, state.log_delta);
                let h4 = sas_inverse_link_pdfthird_derivative(eta, state.epsilon, state.log_delta);
                let h5 = sas_inverse_link_pdffourth_derivative(eta, state.epsilon, state.log_delta);
                // Cross-check the Result-returning dispatch path used by
                // `hessian_cde_arrays` against the direct closed-form
                // helpers — both must return the same h₄, h₅.
                let h4_dispatch = inverse_link_pdfthird_derivative_for_inverse_link(&link, eta)
                    .expect("sas h4 via dispatch");
                let h5_dispatch = inverse_link_pdffourth_derivative_for_inverse_link(&link, eta)
                    .expect("sas h5 via dispatch");
                assert!(
                    (h4 - h4_dispatch).abs() <= 1.0e-12 * h4.abs().max(1.0),
                    "sas h4 dispatch mismatch at eta={eta}, eps={epsilon}, log_delta={log_delta}: direct={h4} dispatch={h4_dispatch}"
                );
                assert!(
                    (h5 - h5_dispatch).abs() <= 1.0e-12 * h5.abs().max(1.0),
                    "sas h5 dispatch mismatch at eta={eta}, eps={epsilon}, log_delta={log_delta}: direct={h5} dispatch={h5_dispatch}"
                );
                for &y in &ys {
                    assert_e_obs_matches_dual3_ad(
                        "sas", eta, y, jet.mu, jet.d1, jet.d2, jet.d3, h4, h5, 1.0, 1.0e-9,
                    );
                }
            }
        }
    }

    /// BetaLogistic + Bernoulli: the third production-grade flexible link
    /// shipped by the GAMLSS marginal-slope family.  Same Dual3 AD test
    /// against `e_obs_from_jets`, exercised at two (delta, ε) shape pairs.
    #[test]
    fn e_obs_from_jets_matches_dual3_ad_beta_logistic_bernoulli() {
        let etas = [-1.0_f64, -0.25, 0.0, 0.4, 0.9];
        let ys = [0.0_f64, 1.0];
        let configs = [(0.18_f64, -0.22_f64), (-0.3_f64, 0.4_f64)];
        for &(epsilon, log_delta) in &configs {
            let delta = log_delta.exp();
            let state = SasLinkState {
                epsilon,
                log_delta,
                delta,
            };
            let link = InverseLink::BetaLogistic(state);
            for &eta in &etas {
                let jet = beta_logistic_inverse_link_jet(eta, state.log_delta, state.epsilon);
                let h4 = beta_logistic_inverse_link_pdfthird_derivative(
                    eta,
                    state.log_delta,
                    state.epsilon,
                );
                let h5 = beta_logistic_inverse_link_pdffourth_derivative(
                    eta,
                    state.log_delta,
                    state.epsilon,
                );
                let h4_dispatch = inverse_link_pdfthird_derivative_for_inverse_link(&link, eta)
                    .expect("beta-logistic h4 dispatch");
                let h5_dispatch = inverse_link_pdffourth_derivative_for_inverse_link(&link, eta)
                    .expect("beta-logistic h5 dispatch");
                assert!(
                    (h4 - h4_dispatch).abs() <= 1.0e-12 * h4.abs().max(1.0),
                    "beta-logistic h4 dispatch mismatch at eta={eta}: direct={h4} dispatch={h4_dispatch}"
                );
                assert!(
                    (h5 - h5_dispatch).abs() <= 1.0e-12 * h5.abs().max(1.0),
                    "beta-logistic h5 dispatch mismatch at eta={eta}: direct={h5} dispatch={h5_dispatch}"
                );
                for &y in &ys {
                    assert_e_obs_matches_dual3_ad(
                        "beta-logistic",
                        eta,
                        y,
                        jet.mu,
                        jet.d1,
                        jet.d2,
                        jet.d3,
                        h4,
                        h5,
                        1.0,
                        1.0e-9,
                    );
                }
            }
        }
    }

    /// Mixture-of-links + Bernoulli: μ is a convex combination of standard
    /// component inverse links, so every η-derivative is the matching
    /// convex combination of the per-component derivatives.  We cover a
    /// 4-component Probit/Logit/CLogLog/Cauchit mixture to exercise the
    /// dispatch path inside `inverse_link_pdf{third,fourth}_derivative_for_inverse_link`.
    #[test]
    fn e_obs_from_jets_matches_dual3_ad_mixture_bernoulli() {
        let spec = MixtureLinkSpec {
            components: vec![
                LinkComponent::Probit,
                LinkComponent::Logit,
                LinkComponent::CLogLog,
                LinkComponent::Cauchit,
            ],
            initial_rho: Array1::from_vec(vec![0.35, -0.45, 0.2]),
        };
        let state = state_fromspec(&spec).expect("mixture state");
        let link = InverseLink::Mixture(state.clone());
        let etas = [-1.2_f64, -0.4, 0.0, 0.5, 1.1];
        let ys = [0.0_f64, 1.0];
        for &eta in &etas {
            let jet = mixture_inverse_link_jet(&state, eta);
            let h4 = inverse_link_pdfthird_derivative_for_inverse_link(&link, eta)
                .expect("mixture h4 dispatch");
            let h5 = inverse_link_pdffourth_derivative_for_inverse_link(&link, eta)
                .expect("mixture h5 dispatch");
            for &y in &ys {
                assert_e_obs_matches_dual3_ad(
                    "mixture", eta, y, jet.mu, jet.d1, jet.d2, jet.d3, h4, h5, 1.0, 1.0e-9,
                );
            }
        }
    }
}

#[cfg(test)]
mod adaptive_lm_lambda_tests {
    use super::adaptive_lm_lambda_hint;

    #[test]
    fn pathological_cached_lambdas_fall_through_to_cold_default() {
        // Non-finite or non-positive cached values must return None so
        // the cold default `1e-6` is used. Locks the historical
        // contract that pathological cache slots can't poison the warm
        // start.
        assert_eq!(adaptive_lm_lambda_hint(f64::NAN, 5, true), None);
        assert_eq!(adaptive_lm_lambda_hint(f64::INFINITY, 5, true), None);
        assert_eq!(adaptive_lm_lambda_hint(0.0, 5, true), None);
        assert_eq!(adaptive_lm_lambda_hint(-1e-3, 5, true), None);
    }

    #[test]
    fn no_feedback_yet_falls_through_to_cold_default() {
        // (last_iters == 0, last_converged == false) is the sentinel
        // `clear_warm_start_adaptive_signals` writes — there's no
        // feedback to drive an adaptive regime, so the cold default
        // `1e-6` is preferred over a possibly-stale cache slot.
        assert_eq!(adaptive_lm_lambda_hint(1e-5, 0, false), None);
    }

    #[test]
    fn newton_friendly_regime_admits_floor_to_1e_minus_9() {
        // Previous fit converged in 1 iter — well-conditioned local
        // geometry. The cached λ is allowed down to 1e-9 (the LM-
        // internal floor) so the next fit can leverage the previous
        // Newton-like trajectory.
        assert_eq!(
            adaptive_lm_lambda_hint(1e-9, 1, true),
            Some(1e-9),
            "cached λ at the LM floor must pass through unchanged"
        );
        assert_eq!(
            adaptive_lm_lambda_hint(1e-12, 1, true),
            Some(1e-9),
            "below-floor cached λ clamped up to 1e-9"
        );
        assert_eq!(
            adaptive_lm_lambda_hint(1e-2, 1, true),
            Some(1e-3),
            "above-ceiling cached λ clamped down to 1e-3 even in Newton-friendly regime"
        );
        // last_iters == 2 still counts as Newton-friendly.
        assert_eq!(adaptive_lm_lambda_hint(1e-9, 2, true), Some(1e-9));
    }

    #[test]
    fn hard_fit_regime_preserves_heavy_damping_signal() {
        // Previous fit didn't converge OR took many iters — the cached
        // λ is in the heavy-damping regime, and we want to preserve
        // that signal up to gradient-descent (1.0).
        assert_eq!(
            adaptive_lm_lambda_hint(0.5, 12, true),
            Some(0.5),
            "heavy-damping cached λ passes through unchanged"
        );
        assert_eq!(
            adaptive_lm_lambda_hint(2.0, 12, true),
            Some(1.0),
            "above-ceiling cached λ clamped to 1.0"
        );
        assert_eq!(
            adaptive_lm_lambda_hint(1e-6, 12, true),
            Some(1e-3),
            "below-floor cached λ clamped up to 1e-3 in hard-fit regime"
        );
        // Non-converged (cap exhausted) path takes the same regime.
        assert_eq!(adaptive_lm_lambda_hint(0.5, 5, false), Some(0.5));
    }

    #[test]
    fn default_regime_matches_historical_static_clamp() {
        // 1-9 iters AND converged — a "moderate" fit. The clamp is
        // [1e-6, 1e-3], matching the historical static clamp before
        // the adaptive layer was introduced. Locks behavior so a
        // future commit can't silently widen the default range.
        assert_eq!(adaptive_lm_lambda_hint(1e-5, 5, true), Some(1e-5));
        assert_eq!(adaptive_lm_lambda_hint(1e-9, 5, true), Some(1e-6));
        assert_eq!(adaptive_lm_lambda_hint(1e-1, 5, true), Some(1e-3));
        // Boundary: last_iters=3 (above Newton-friendly cap of 2,
        // below hard-fit floor of 10) goes to default.
        assert_eq!(adaptive_lm_lambda_hint(1e-5, 3, true), Some(1e-5));
        assert_eq!(adaptive_lm_lambda_hint(1e-5, 9, true), Some(1e-5));
    }
}

#[cfg(test)]
mod ift_warm_start_tests {
    use super::*;
    use crate::construction::CanonicalPenalty;
    use crate::linalg::matrix::SymmetricMatrix;
    use ndarray::Array2;

    /// Test-only entry into `predict_warm_start_beta_ift_inner_with_outcome`
    /// that drops the outcome — the production
    /// `RemlState::predict_warm_start_beta_ift_with_outcome` path uses the
    /// outcome to bookkeep the IFT cache, but the regression tests below
    /// only assert on `Coefficients`.
    fn predict_warm_start_beta_ift_inner(
        cache: &super::IftWarmStartCache,
        canonical_penalties: &[CanonicalPenalty],
        new_rho: &Array1<f64>,
        p: usize,
        last_ift_residual: Option<f64>,
    ) -> Option<Coefficients> {
        super::predict_warm_start_beta_ift_inner_with_outcome(
            cache,
            canonical_penalties,
            new_rho,
            p,
            last_ift_residual,
            None,
        )
        .map(|(coef, _outcome)| coef)
    }

    /// Build a CanonicalPenalty from a dense p×p SPD-ish penalty matrix by
    /// taking its eigendecomposition and packing the positive-eigenvalue
    /// components into the `rank × p` root.
    fn dense_canonical_from_local(local: Array2<f64>, p: usize) -> CanonicalPenalty {
        use crate::faer_ndarray::FaerEigh;
        use faer::Side;
        let (evals, evecs) = local.eigh(Side::Lower).expect("eigh penalty");
        let mut rows: Vec<Array1<f64>> = Vec::new();
        let mut positive_eigenvalues: Vec<f64> = Vec::new();
        for (idx, &lam) in evals.iter().enumerate() {
            if lam > 1e-12 {
                let scale = lam.sqrt();
                let v = evecs.column(idx).to_owned();
                rows.push(v.mapv(|x| x * scale));
                positive_eigenvalues.push(lam);
            }
        }
        let rank = rows.len();
        let mut root = Array2::<f64>::zeros((rank, p));
        for (i, row) in rows.iter().enumerate() {
            for j in 0..p {
                root[[i, j]] = row[j];
            }
        }
        let nullity = p - rank;
        CanonicalPenalty {
            root,
            col_range: 0..p,
            total_dim: p,
            nullity,
            local,
            prior_mean: Array1::zeros(p),
            positive_eigenvalues,
            op: None,
        }
    }

    /// Verify the IFT predictor satisfies the linearized FOC:
    /// `H_pen · (β_predict − β_cur) ≈ −Σ_k Δρ_k · e^{ρ_k} · S_k · (β_cur-μ_k)`.
    /// Tested in the original-basis path (`frame_was_original = true`).
    #[test]
    fn ift_predictor_satisfies_linearized_foc_original_basis() {
        let p = 5usize;
        // Hand-built S_1, S_2 — diagonal-dominant SPD blocks so the
        // local matrices have full rank and the FOC is non-trivial.
        let s1 = Array2::from_shape_vec(
            (p, p),
            vec![
                2.0, 0.3, 0.0, 0.0, 0.0, 0.3, 2.5, 0.4, 0.0, 0.0, 0.0, 0.4, 1.8, 0.2, 0.0, 0.0,
                0.0, 0.2, 1.2, 0.1, 0.0, 0.0, 0.0, 0.1, 1.5,
            ],
        )
        .unwrap();
        let s2 = Array2::from_shape_vec(
            (p, p),
            vec![
                1.1, 0.0, 0.2, 0.0, 0.1, 0.0, 0.9, 0.0, 0.3, 0.0, 0.2, 0.0, 1.4, 0.0, 0.2, 0.0,
                0.3, 0.0, 1.7, 0.0, 0.1, 0.0, 0.2, 0.0, 2.1,
            ],
        )
        .unwrap();
        let cp1 = dense_canonical_from_local(s1.clone(), p);
        let cp2 = dense_canonical_from_local(s2.clone(), p);
        let canonical = vec![cp1, cp2];

        // Cached β at converged solve.
        let beta_cur = ndarray::array![0.4, -0.7, 0.2, 0.9, -0.3];
        // Cached ρ.
        let rho_cur = ndarray::array![0.2_f64, -0.1];
        // H_pen at the cached solve. Doesn't have to literally equal
        // ∇²D + Σ e^ρ_k S_k for the linear-FOC check; it just has to
        // be SPD and consistent with what the predictor uses. Keep it
        // simple: use ∇²D = identity and add the penalties.
        let mut h_pen = Array2::<f64>::eye(p);
        for (k_idx, cp) in canonical.iter().enumerate() {
            let lam = rho_cur[k_idx].exp();
            for i in 0..p {
                for j in 0..p {
                    h_pen[[i, j]] += lam * cp.local[[i, j]];
                }
            }
        }

        let cache = super::super::IftWarmStartCache {
            beta_original: beta_cur.clone(),
            rho: rho_cur.clone(),
            penalized_hessian_transformed: SymmetricMatrix::Dense(h_pen.clone()),
            qs: Array2::eye(p),
            frame_was_original: true,
            lambda_s_beta_blocks: None,
        };

        // Small Δρ — well within the 2.0 cap.
        let new_rho = ndarray::array![0.25_f64, -0.05];
        let predicted = predict_warm_start_beta_ift_inner(&cache, &canonical, &new_rho, p, None)
            .expect("IFT predictor should accept small Δρ");

        // Check the linearized FOC residual:
        //   H_pen · (β_pred − β_cur) + Σ_k Δρ_k · e^{ρ_k} · S_k · (β_cur-μ_k) ≈ 0
        let dbeta = &predicted.0 - &beta_cur;
        let lhs = h_pen.dot(&dbeta);
        let mut rhs = Array1::<f64>::zeros(p);
        for (k_idx, cp) in canonical.iter().enumerate() {
            let drho = new_rho[k_idx] - rho_cur[k_idx];
            let scale = drho * rho_cur[k_idx].exp();
            let sb = cp.local.dot(&beta_cur);
            for i in 0..p {
                rhs[i] += scale * sb[i];
            }
        }
        // residual should be tiny (linear-FOC is exactly satisfied by the IFT
        // predictor up to factorization round-off).
        let residual: f64 = lhs
            .iter()
            .zip(rhs.iter())
            .map(|(&a, &b)| (a + b) * (a + b))
            .sum::<f64>()
            .sqrt();
        let scale: f64 = rhs.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
        assert!(
            residual / scale < 1e-9,
            "IFT predictor violates linearized FOC: residual {residual:.3e}, scale {scale:.3e}"
        );

        // β_predict should be different from β_cur (otherwise the predictor
        // is a no-op and the test is meaningless).
        let dbeta_norm: f64 = dbeta.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            dbeta_norm > 1e-6,
            "IFT predictor produced zero β-update; check that Δρ propagated"
        );
    }

    /// The lambda_s_beta_blocks precompute path (commit 2f2670b7) must
    /// produce BIT-EQUIVALENT output to the inline-mat-vec fallback.
    /// The 4 existing IFT tests pass `None` for the precompute field,
    /// leaving the fast path untested. This test populates the field
    /// from the same penalties the fallback would use and asserts both
    /// β_predict are identical to ~floating-point round-off.
    #[test]
    fn ift_predictor_precompute_path_matches_inline_path() {
        let p = 5usize;
        let s1 = Array2::from_shape_vec(
            (p, p),
            vec![
                1.0, 0.2, 0.0, 0.0, 0.0, 0.2, 1.5, 0.1, 0.0, 0.0, 0.0, 0.1, 1.3, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        let s2 = Array2::from_shape_vec(
            (p, p),
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1, 0.05, 0.0, 0.0,
                0.0, 0.05, 1.7, 0.0, 0.0, 0.0, 0.0, 0.0, 2.1,
            ],
        )
        .unwrap();
        let cp1 = dense_canonical_from_local(s1, p);
        let cp2 = dense_canonical_from_local(s2, p);
        let canonical = vec![cp1.clone(), cp2.clone()];

        let beta_cur = ndarray::array![0.4_f64, -0.7, 0.2, 0.9, -0.3];
        let rho_cur = ndarray::array![0.2_f64, -0.1];

        // Build H_pen consistently with the cached solve.
        let mut h_pen = Array2::<f64>::eye(p);
        for (k_idx, cp) in canonical.iter().enumerate() {
            let lam = rho_cur[k_idx].exp();
            for i in 0..p {
                for j in 0..p {
                    h_pen[[i, j]] += lam * cp.local[[i, j]];
                }
            }
        }

        // Precompute the per-penalty `S_k · (β_cur-μ_k)` blocks the same way
        // updatewarm_start_from does at cache-write time.
        let lambda_s_beta_blocks: Vec<ndarray::Array1<f64>> = canonical
            .iter()
            .map(|cp| {
                let r = &cp.col_range;
                let beta_block = beta_cur.slice(s![r.start..r.end]);
                let centered = &beta_block - &cp.prior_mean;
                cp.local.dot(&centered)
            })
            .collect();

        let cache_inline = super::super::IftWarmStartCache {
            beta_original: beta_cur.clone(),
            rho: rho_cur.clone(),
            penalized_hessian_transformed: SymmetricMatrix::Dense(h_pen.clone()),
            qs: Array2::eye(p),
            frame_was_original: true,
            lambda_s_beta_blocks: None,
        };
        let cache_precompute = super::super::IftWarmStartCache {
            beta_original: beta_cur.clone(),
            rho: rho_cur.clone(),
            penalized_hessian_transformed: SymmetricMatrix::Dense(h_pen.clone()),
            qs: Array2::eye(p),
            frame_was_original: true,
            lambda_s_beta_blocks: Some(lambda_s_beta_blocks),
        };

        let new_rho = ndarray::array![0.25_f64, -0.05];
        let predicted_inline =
            predict_warm_start_beta_ift_inner(&cache_inline, &canonical, &new_rho, p, None)
                .expect("inline-path predict");
        let predicted_precompute =
            predict_warm_start_beta_ift_inner(&cache_precompute, &canonical, &new_rho, p, None)
                .expect("precompute-path predict");

        // Bit-equivalent: the precompute path multiplies and accumulates
        // in the same order as the inline path, so the two should agree
        // to round-off at most a few ULPs.
        for i in 0..p {
            let diff = (predicted_inline.0[i] - predicted_precompute.0[i]).abs();
            assert!(
                diff < 1e-12,
                "precompute and inline IFT paths diverged at index {i}: \
                 inline={} precompute={} diff={:.3e}",
                predicted_inline.0[i],
                predicted_precompute.0[i],
                diff,
            );
        }

        // Defensive: if the precompute length disagrees with
        // canonical_penalties.len(), the predictor must fall back to
        // inline-mat-vec rather than mis-indexing.
        let cache_wrong_len = super::super::IftWarmStartCache {
            beta_original: beta_cur.clone(),
            rho: rho_cur.clone(),
            penalized_hessian_transformed: SymmetricMatrix::Dense(h_pen.clone()),
            qs: Array2::eye(p),
            frame_was_original: true,
            lambda_s_beta_blocks: Some(vec![ndarray::Array1::zeros(0)]), // length 1, expected 2
        };
        let predicted_fallback =
            predict_warm_start_beta_ift_inner(&cache_wrong_len, &canonical, &new_rho, p, None)
                .expect("wrong-length precompute should still predict via inline fallback");
        for i in 0..p {
            let diff = (predicted_inline.0[i] - predicted_fallback.0[i]).abs();
            assert!(
                diff < 1e-12,
                "wrong-length precompute fell back incorrectly at index {i}",
            );
        }

        // Exercise the `_with_factor` production path: build the
        // factor manually, pass it in, verify the output matches the
        // inline-factorize path. This covers the H_pen factor cache
        // (commit ec18559d) which the integration tests only touch
        // transitively.
        let pre_built_factor = SymmetricMatrix::Dense(h_pen.clone())
            .factorize()
            .expect("factorize for _with_factor test");
        let predicted_via_factor = predict_warm_start_beta_ift_with_factor(
            &cache_inline,
            &canonical,
            &new_rho,
            p,
            None,
            pre_built_factor.as_ref(),
        )
        .expect("with_factor predict");
        for i in 0..p {
            let diff = (predicted_inline.0[i] - predicted_via_factor.0[i]).abs();
            assert!(
                diff < 1e-12,
                "_with_factor and inline paths diverged at index {i}: \
                 inline={} factor={} diff={:.3e}",
                predicted_inline.0[i],
                predicted_via_factor.0[i],
                diff,
            );
        }

        // Negative test: pass a DIFFERENT factor than the one matching
        // cache.penalized_hessian_transformed, and verify the predictor
        // actually USES it (vs silently ignoring and re-factorizing
        // internally). With a perturbed H, the prediction must
        // differ measurably from the inline-path prediction.
        let mut h_perturbed = h_pen.clone();
        for i in 0..p {
            // Add 5× to diagonal — produces a factor that maps
            // back-solves to noticeably smaller magnitudes.
            h_perturbed[[i, i]] *= 5.0;
        }
        let perturbed_factor = SymmetricMatrix::Dense(h_perturbed)
            .factorize()
            .expect("factorize perturbed H");
        let predicted_with_wrong_factor = predict_warm_start_beta_ift_with_factor(
            &cache_inline,
            &canonical,
            &new_rho,
            p,
            None,
            perturbed_factor.as_ref(),
        )
        .expect("with_factor predict (wrong factor)");
        let mut max_diff = 0.0_f64;
        for i in 0..p {
            let diff = (predicted_inline.0[i] - predicted_with_wrong_factor.0[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        assert!(
            max_diff > 1e-3,
            "wrong factor produced bit-identical output (max_diff={:.3e}); \
             _with_factor path is silently ignoring its factor argument",
            max_diff,
        );
    }

    /// Verify the basis-conversion path: when frame_was_original=false,
    /// solving with H in transformed basis and round-tripping through qs
    /// (which is orthogonal) must produce the SAME prediction as solving
    /// in original basis with H_orig = qs · H_tfd · qs^T.
    #[test]
    fn ift_predictor_basis_conversion_matches_original() {
        let p = 4usize;
        let s1 = Array2::from_shape_vec(
            (p, p),
            vec![
                1.5, 0.2, 0.0, 0.0, 0.2, 1.8, 0.1, 0.0, 0.0, 0.1, 1.2, 0.05, 0.0, 0.0, 0.05, 1.4,
            ],
        )
        .unwrap();
        let cp1 = dense_canonical_from_local(s1.clone(), p);
        let canonical = vec![cp1];

        let beta_cur = ndarray::array![0.5, -0.3, 0.8, 0.2];
        let rho_cur = ndarray::array![0.1_f64];

        // H_orig = I + e^ρ · S
        let mut h_orig = Array2::<f64>::eye(p);
        let lam = rho_cur[0].exp();
        for i in 0..p {
            for j in 0..p {
                h_orig[[i, j]] += lam * canonical[0].local[[i, j]];
            }
        }

        // Build a non-trivial orthogonal qs (Gram-Schmidt of a random-ish basis).
        let raw = Array2::from_shape_vec(
            (p, p),
            vec![
                1.0, 0.5, 0.0, 0.1, 0.0, 1.0, 0.3, 0.0, 0.2, 0.0, 1.0, 0.4, 0.0, 0.1, 0.0, 1.0,
            ],
        )
        .unwrap();
        let (q, _r) = {
            use crate::faer_ndarray::FaerQr;
            raw.qr().expect("QR")
        };
        let qs: Array2<f64> = q;
        // Sanity: qs is orthogonal.
        let qq = qs.t().dot(&qs);
        for i in 0..p {
            for j in 0..p {
                let target = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (qq[[i, j]] - target).abs() < 1e-10,
                    "qs not orthogonal at [{},{}]: {}",
                    i,
                    j,
                    qq[[i, j]]
                );
            }
        }

        let h_tfd = qs.t().dot(&h_orig).dot(&qs);

        let cache_tfd = super::super::IftWarmStartCache {
            beta_original: beta_cur.clone(),
            rho: rho_cur.clone(),
            penalized_hessian_transformed: SymmetricMatrix::Dense(h_tfd),
            qs: qs.clone(),
            frame_was_original: false,
            lambda_s_beta_blocks: None,
        };
        let cache_orig = super::super::IftWarmStartCache {
            beta_original: beta_cur.clone(),
            rho: rho_cur.clone(),
            penalized_hessian_transformed: SymmetricMatrix::Dense(h_orig.clone()),
            qs: Array2::eye(p),
            frame_was_original: true,
            lambda_s_beta_blocks: None,
        };

        let new_rho = ndarray::array![0.3_f64];
        let predicted_tfd =
            predict_warm_start_beta_ift_inner(&cache_tfd, &canonical, &new_rho, p, None)
                .expect("tfd predict");
        let predicted_orig =
            predict_warm_start_beta_ift_inner(&cache_orig, &canonical, &new_rho, p, None)
                .expect("orig predict");

        for i in 0..p {
            assert!(
                (predicted_tfd.0[i] - predicted_orig.0[i]).abs() < 1e-10,
                "basis-conversion path mismatch at index {i}: tfd={}, orig={}",
                predicted_tfd.0[i],
                predicted_orig.0[i],
            );
        }
    }

    /// Δρ above the safety cap must reject (return None), so the caller
    /// falls through to the tangent-line / flat warm-start.
    #[test]
    fn ift_predictor_rejects_large_drho() {
        let p = 3usize;
        let s1 = Array2::from_shape_vec((p, p), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
            .unwrap();
        let cp1 = dense_canonical_from_local(s1, p);
        let canonical = vec![cp1];
        let beta_cur = ndarray::array![1.0, 1.0, 1.0];
        let rho_cur = ndarray::array![0.0_f64];
        let h_pen = Array2::<f64>::eye(p) * 2.0;
        let cache = super::super::IftWarmStartCache {
            beta_original: beta_cur,
            rho: rho_cur,
            penalized_hessian_transformed: SymmetricMatrix::Dense(h_pen),
            qs: Array2::eye(p),
            frame_was_original: true,
            lambda_s_beta_blocks: None,
        };
        // |Δρ| = 3 > IFT_WARM_START_DEFAULT_MAX_DRHO = 2.0 → reject under
        // default cap (no quality history).
        let new_rho = ndarray::array![3.0_f64];
        let predicted = predict_warm_start_beta_ift_inner(&cache, &canonical, &new_rho, p, None);
        assert!(
            predicted.is_none(),
            "predictor should reject Δρ above default cap, got {:?}",
            predicted
        );
        // With excellent prior quality (residual=0.005) the adaptive cap
        // expands to 4.0, so |Δρ|=3 is now ACCEPTED.
        let predicted_good_history =
            predict_warm_start_beta_ift_inner(&cache, &canonical, &new_rho, p, Some(0.005));
        assert!(
            predicted_good_history.is_some(),
            "predictor should accept Δρ=3 under expanded cap (good prior quality)",
        );
        // With poor prior quality (residual=0.6) the adaptive cap
        // tightens to 0.5, so even modest |Δρ|=1 is REJECTED.
        let modest_rho = ndarray::array![1.0_f64];
        let predicted_bad_history =
            predict_warm_start_beta_ift_inner(&cache, &canonical, &modest_rho, p, Some(0.6));
        assert!(
            predicted_bad_history.is_none(),
            "predictor should reject Δρ=1 under tightened cap (poor prior quality)",
        );
    }

    /// All Δρ below the eps floor (effectively-zero outer step) must
    /// return Some(β_cur, Noop) — preserving the cached β unchanged.
    /// The inner predictor short-circuits identity predictions before
    /// paying a fresh O(p³)/3 Cholesky; this test exercises that
    /// contract directly. The production
    /// `RemlState::predict_warm_start_beta_ift_with_outcome` wraps the
    /// same check in front of the H_pen factor lookup so the cache miss
    /// is also avoided.
    #[test]
    fn ift_predictor_returns_noop_when_all_drho_below_eps() {
        use crate::estimate::reml::IftWarmStartCache;

        let p = 3usize;
        let s1 = Array2::from_shape_vec((p, p), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
            .unwrap();
        let cp1 = dense_canonical_from_local(s1, p);
        let canonical = vec![cp1];
        let beta_cur = ndarray::array![0.5_f64, -0.3, 1.7];
        let rho_cur = ndarray::array![0.0_f64];
        let h_pen = Array2::<f64>::eye(p) * 2.0;
        let cache = IftWarmStartCache {
            beta_original: beta_cur.clone(),
            rho: rho_cur,
            penalized_hessian_transformed: SymmetricMatrix::Dense(h_pen),
            qs: Array2::eye(p),
            frame_was_original: true,
            lambda_s_beta_blocks: None,
        };
        // Δρ = 1e-15 is well below IFT_WARM_START_DRHO_EPS (1e-12);
        // every component of the new ρ is essentially the cached ρ.
        let new_rho = ndarray::array![1e-15_f64];
        let predicted = predict_warm_start_beta_ift_inner(&cache, &canonical, &new_rho, p, None)
            .expect("noop must return Some(β_cur)");
        for i in 0..p {
            assert_eq!(
                predicted.0[i], beta_cur[i],
                "noop must return cached β bit-equal at idx {i}: got {} vs cached {}",
                predicted.0[i], beta_cur[i]
            );
        }
    }

    /// Empty ρ in the cache (the in-between state where
    /// updatewarm_start_from has populated β/H but record_warm_start_rho
    /// has not stamped ρ yet) must return None.
    #[test]
    fn ift_predictor_rejects_unstamped_cache() {
        let p = 2usize;
        let cache = super::super::IftWarmStartCache {
            beta_original: ndarray::array![1.0, 2.0],
            rho: Array1::zeros(0),
            penalized_hessian_transformed: SymmetricMatrix::Dense(Array2::eye(p)),
            qs: Array2::eye(p),
            frame_was_original: true,
            lambda_s_beta_blocks: None,
        };
        let new_rho = ndarray::array![0.1_f64];
        let predicted = predict_warm_start_beta_ift_inner(&cache, &[], &new_rho, p, None);
        assert!(predicted.is_none());
    }

    /// `adaptive_ift_max_drho` must follow the documented piecewise
    /// policy: small residual → looser cap; large residual → tighter
    /// cap; non-finite or missing residual → default 2.0.
    #[test]
    fn adaptive_ift_max_drho_follows_quality_tiers() {
        use super::adaptive_ift_max_drho;
        // No history (first solve at this surface) → default 2.0.
        assert_eq!(adaptive_ift_max_drho(None), 2.0);
        // Pathological residuals (NaN, negative) → default 2.0.
        assert_eq!(adaptive_ift_max_drho(Some(f64::NAN)), 2.0);
        assert_eq!(adaptive_ift_max_drho(Some(-1.0)), 2.0);
        assert_eq!(adaptive_ift_max_drho(Some(f64::INFINITY)), 0.5);
        // Tier boundaries.
        assert_eq!(adaptive_ift_max_drho(Some(0.0)), 4.0);
        assert_eq!(adaptive_ift_max_drho(Some(0.005)), 4.0);
        assert_eq!(adaptive_ift_max_drho(Some(0.01)), 3.0);
        assert_eq!(adaptive_ift_max_drho(Some(0.04)), 3.0);
        assert_eq!(adaptive_ift_max_drho(Some(0.05)), 2.0);
        assert_eq!(adaptive_ift_max_drho(Some(0.10)), 2.0);
        assert_eq!(adaptive_ift_max_drho(Some(0.20)), 1.0);
        assert_eq!(adaptive_ift_max_drho(Some(0.30)), 1.0);
        assert_eq!(adaptive_ift_max_drho(Some(0.50)), 0.5);
        assert_eq!(adaptive_ift_max_drho(Some(1.5)), 0.5);
        // Policy is monotone non-increasing in residual.
        let residuals = [0.001, 0.02, 0.10, 0.30, 0.80];
        let caps: Vec<f64> = residuals
            .iter()
            .map(|&r| adaptive_ift_max_drho(Some(r)))
            .collect();
        for w in caps.windows(2) {
            assert!(
                w[0] >= w[1],
                "adaptive cap is not monotone non-increasing in residual: {caps:?}"
            );
        }
    }

    /// Parallel `S_k · (β-μ_k)` mat-vec across penalties (the rayon par_iter
    /// pattern used at IFT cache-write time in updatewarm_start_from)
    /// must produce bit-equivalent output to the serial version. The
    /// parallelization is across penalties (each penalty's mat-vec is
    /// independent and writes to its own Vec slot — no shared state),
    /// so reordering doesn't change the floating-point result. This
    /// test pins that invariant down: 8 penalties × 50-coefficient
    /// blocks, par_iter vs serial iter, must agree to bit-equality
    /// (not just within-tolerance).
    #[test]
    fn parallel_lambda_s_beta_blocks_matches_serial() {
        use rayon::prelude::*;
        let p = 50usize;
        let n_penalties = 8usize;
        let mut canonical = Vec::with_capacity(n_penalties);
        for k in 0..n_penalties {
            let scale = (k as f64 + 1.0) * 0.5;
            let mut s = Array2::<f64>::zeros((p, p));
            for i in 0..p {
                s[[i, i]] = scale;
                if i + 1 < p {
                    s[[i, i + 1]] = scale * 0.1;
                    s[[i + 1, i]] = scale * 0.1;
                }
            }
            canonical.push(dense_canonical_from_local(s, p));
        }
        let beta_cur =
            ndarray::Array1::from_shape_fn(p, |i| (i as f64 * 0.1).sin() + (i as f64 * 0.05).cos());
        // Serial reference.
        let serial: Vec<ndarray::Array1<f64>> = canonical
            .iter()
            .map(|cp| {
                let r = &cp.col_range;
                let beta_block = beta_cur.slice(s![r.start..r.end]);
                let centered = &beta_block - &cp.prior_mean;
                cp.local.dot(&centered)
            })
            .collect();
        // Parallel (matches the writer at line ~2272).
        let parallel: Vec<ndarray::Array1<f64>> = canonical
            .par_iter()
            .map(|cp| {
                let r = &cp.col_range;
                let beta_block = beta_cur.slice(s![r.start..r.end]);
                let centered = &beta_block - &cp.prior_mean;
                cp.local.dot(&centered)
            })
            .collect();
        assert_eq!(serial.len(), parallel.len());
        for (s_block, p_block) in serial.iter().zip(parallel.iter()) {
            assert_eq!(s_block.len(), p_block.len());
            for (a, b) in s_block.iter().zip(p_block.iter()) {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "parallel mat-vec diverged from serial: {a} vs {b}",
                );
            }
        }
    }

    /// Pin down the NaN-sentinel discipline for the IFT residual
    /// atomic. The `0` sentinel previously used would have collided
    /// with `f64::to_bits(0.0) == 0`, making a residual of exactly 0
    /// indistinguishable from "no signal yet". The new sentinel
    /// (`IFT_RESIDUAL_NO_SIGNAL_BITS`, a quiet-NaN bit pattern)
    /// encodes "no signal" via NaN's self-inequality, so any finite
    /// non-negative value is unambiguously genuine signal.
    #[test]
    fn ift_residual_sentinel_is_distinguishable_from_zero() {
        use super::IFT_RESIDUAL_NO_SIGNAL_BITS;
        // Sentinel decodes to NaN.
        assert!(
            f64::from_bits(IFT_RESIDUAL_NO_SIGNAL_BITS).is_nan(),
            "sentinel must decode to NaN so reads can detect 'no signal yet'",
        );
        // 0.0 round-trips as 0 bits — distinct from the sentinel, so
        // a stored residual of 0.0 would not be confused with the
        // no-signal state.
        assert_eq!(0.0_f64.to_bits(), 0, "f64::to_bits(0.0) is 0 by IEEE 754");
        assert_ne!(
            IFT_RESIDUAL_NO_SIGNAL_BITS, 0,
            "sentinel must not collide with f64::to_bits(0.0)",
        );
        // The reader's `r.is_finite() && r >= 0.0` predicate accepts
        // 0.0 (a real signal) and rejects the NaN sentinel.
        let r_zero = f64::from_bits(0.0_f64.to_bits());
        assert!(r_zero.is_finite() && r_zero >= 0.0, "0.0 is genuine signal");
        let r_sentinel = f64::from_bits(IFT_RESIDUAL_NO_SIGNAL_BITS);
        assert!(
            !(r_sentinel.is_finite() && r_sentinel >= 0.0),
            "sentinel must fail the reader's accept predicate",
        );
        // Any finite non-negative residual round-trips through
        // `to_bits` / `from_bits` losslessly.
        for &val in &[0.0_f64, 1e-10, 0.05, 0.5, 4.0] {
            let bits = val.to_bits();
            let back = f64::from_bits(bits);
            assert_eq!(back, val, "round-trip failed for {val}");
        }
    }

    /// Dim-mismatch rejection paths must return None (so the caller
    /// falls through to tangent-line / flat warm-start) rather than
    /// dereferencing into mismatched arrays. The new structured
    /// markers added in this commit make these silent failure modes
    /// visible in production logs as bug signals.
    #[test]
    fn ift_predictor_rejects_dim_mismatches() {
        let p = 3usize;
        let s1 = Array2::from_shape_vec((p, p), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
            .unwrap();
        let cp1 = dense_canonical_from_local(s1, p);
        let canonical = vec![cp1];
        let beta_cur = ndarray::array![1.0_f64, 1.0, 1.0];
        let rho_cur = ndarray::array![0.0_f64];
        let cache = super::super::IftWarmStartCache {
            beta_original: beta_cur,
            rho: rho_cur,
            penalized_hessian_transformed: SymmetricMatrix::Dense(Array2::<f64>::eye(p) * 2.0),
            qs: Array2::eye(p),
            frame_was_original: true,
            lambda_s_beta_blocks: None,
        };
        // new_rho dim mismatch: cache has 1 ρ, caller passes 2.
        let bad_new_rho = ndarray::array![0.1_f64, 0.2];
        assert!(
            predict_warm_start_beta_ift_inner(&cache, &canonical, &bad_new_rho, p, None).is_none(),
            "new_rho dim mismatch must reject",
        );
        // penalty_penalties dim mismatch: cache has 1 ρ, caller passes 0 penalties.
        let new_rho = ndarray::array![0.1_f64];
        assert!(
            predict_warm_start_beta_ift_inner(&cache, &[], &new_rho, p, None).is_none(),
            "penalty dim mismatch must reject",
        );
        // beta dim mismatch: cache has p=3 β, caller passes p=4.
        assert!(
            predict_warm_start_beta_ift_inner(&cache, &canonical, &new_rho, 4, None).is_none(),
            "beta dim mismatch must reject",
        );
    }

    /// `adaptive_tangent_alpha_cap` follows the same quality-tier
    /// pattern as `adaptive_ift_max_drho`, but with values calibrated
    /// to the tangent-line predictor's α scale (multiples of the
    /// previous ρ-step). Pin down each tier transition + defensive
    /// handling.
    #[test]
    fn adaptive_tangent_alpha_cap_follows_quality_tiers() {
        use super::adaptive_tangent_alpha_cap;
        // No history → default 1.5 (the original hardcoded constant).
        assert_eq!(adaptive_tangent_alpha_cap(None), 1.5);
        // NaN / negative → default (defensive against torn atomics).
        assert_eq!(adaptive_tangent_alpha_cap(Some(f64::NAN)), 1.5);
        assert_eq!(adaptive_tangent_alpha_cap(Some(-1.0)), 1.5);
        // INFINITY → tightest tier (catastrophic prediction).
        assert_eq!(adaptive_tangent_alpha_cap(Some(f64::INFINITY)), 0.5);
        // Tier boundaries.
        assert_eq!(adaptive_tangent_alpha_cap(Some(0.0)), 2.0);
        assert_eq!(adaptive_tangent_alpha_cap(Some(0.005)), 2.0);
        assert_eq!(adaptive_tangent_alpha_cap(Some(0.01)), 1.75);
        assert_eq!(adaptive_tangent_alpha_cap(Some(0.04)), 1.75);
        assert_eq!(adaptive_tangent_alpha_cap(Some(0.05)), 1.5);
        assert_eq!(adaptive_tangent_alpha_cap(Some(0.10)), 1.5);
        assert_eq!(adaptive_tangent_alpha_cap(Some(0.20)), 1.0);
        assert_eq!(adaptive_tangent_alpha_cap(Some(0.49)), 1.0);
        assert_eq!(adaptive_tangent_alpha_cap(Some(0.50)), 0.5);
        assert_eq!(adaptive_tangent_alpha_cap(Some(2.0)), 0.5);
        // Monotone non-increasing in residual: same shape as
        // adaptive_ift_max_drho. The two predictors share the
        // residual signal so their caps move together.
        let residuals = [0.001, 0.02, 0.10, 0.30, 0.80];
        let caps: Vec<f64> = residuals
            .iter()
            .map(|&r| adaptive_tangent_alpha_cap(Some(r)))
            .collect();
        for w in caps.windows(2) {
            assert!(
                w[0] >= w[1],
                "tangent α cap is not monotone non-increasing in residual: {caps:?}"
            );
        }
    }
}
