//! KKT-residual ρ corrections and the shared REML derivative workspace.
//!
//! Holds the precomputed gradient-pass intermediates threaded into the dense
//! Hessian assembler, the active-upper-ρ box mask, the residual solve kernel,
//! and the exact derivatives of the Newton/IFT residual correction that the
//! cost uses (so the additive block vanishes at exact KKT).

use super::*;
use crate::estimate::reml::outer_eval;

/// Shared precomputed REML derivative intermediates threaded from the
/// gradient pass into the dense Hessian assembler so the per-coordinate
/// `penalty_a_k_beta` / `hop.solve` / drift-correction work is not repeated.
pub(crate) struct RemlDerivativeWorkspace<'a> {
    pub curvature_lambdas: &'a [f64],
    pub rho_penalty_a_k_betas: &'a [Array1<f64>],
    pub rho_curvature_a_k_betas: &'a [Array1<f64>],
    pub rho_v_ks: Option<&'a [Array1<f64>]>,
    pub ext_v_is: Option<&'a [Array1<f64>]>,
    pub coord_corrections: &'a [Option<DriftDerivResult>],
}

pub(crate) struct KktRhoCorrections {
    pub(crate) gradient: Array1<f64>,
    pub(crate) hessian: Option<Array2<f64>>,
}

pub(crate) fn solve_kkt_residual_kernel(
    hop: &dyn HessianOperator,
    subspace: Option<&PenaltySubspaceTrace>,
    rhs: &Array1<f64>,
) -> Array1<f64> {
    if let Some(kernel) = subspace {
        let projected = crate::faer_ndarray::fast_atv(&kernel.u_s, rhs);
        let solved_projected = kernel.h_proj_inverse.dot(&projected);
        crate::faer_ndarray::fast_av(&kernel.u_s, &solved_projected)
    } else {
        hop.solve(rhs)
    }
}

pub(crate) fn active_upper_rho_mask(rho: &[f64]) -> Vec<bool> {
    let latest_theta = outer_eval::latest_outer_theta_for_ift();
    let matching_outer_theta = latest_theta.as_ref().is_some_and(|theta| {
        theta.len() >= rho.len()
            && theta
                .iter()
                .take(rho.len())
                .zip(rho.iter())
                .all(|(&recorded, &current)| recorded.to_bits() == current.to_bits())
    });
    let upper_bounds = matching_outer_theta
        .then(outer_eval::latest_outer_rho_upper_bounds_for_ift)
        .flatten();
    rho.iter()
        .enumerate()
        .map(|(idx, &value)| {
            let upper = upper_bounds
                .as_ref()
                .and_then(|bounds| bounds.get(idx))
                .copied()
                .unwrap_or(crate::solver::estimate::RHO_BOUND);
            upper.is_finite() && value >= upper - 1.0e-8
        })
        .collect()
}

/// Derivatives of the same Newton/IFT residual correction used by the cost:
///
///   C(ρ) = -½ r(ρ)^T K(ρ) r(ρ),   K = H^{-1}
///
/// for fixed-dispersion LAML. At fixed β̂, `r_i = A_i β̂` and
/// `H_i = A_i`, so
///
///   C_i  = -a_i^T q + ½ q^T A_i q,
///   q    = K r,
///   q_j  = K(a_j - A_j q),
///   C_ij = -δ_ij a_i^T q - a_i^T q_j
///          + q_j^T A_i q + ½δ_ij q^T A_i q.
///
/// The dense outer Hessian already contains the exact-KKT profile term
/// `-a_i^T K a_j`. That term is valid only when `r = 0`; the residual
/// correction is therefore added as `a_i^T K a_j + C_ij`. This guarantees
/// the additive block vanishes at exact KKT and is exact for the Gaussian
/// quadratic reproduction.
pub(crate) fn compute_kkt_residual_rho_corrections(
    solution: &InnerSolution<'_>,
    hop: &dyn HessianOperator,
    lambdas: &[f64],
    penalty_a_k_betas: &[Array1<f64>],
    residual: &Array1<f64>,
    include_hessian: bool,
    upper_active_rho: &[bool],
) -> Result<KktRhoCorrections, String> {
    let k = penalty_a_k_betas.len();
    if k == 0 {
        return Ok(KktRhoCorrections {
            gradient: Array1::zeros(0),
            hessian: include_hessian.then(|| Array2::zeros((0, 0))),
        });
    }
    if lambdas.len() != k || solution.penalty_coords.len() != k {
        return Err(RemlError::DimensionMismatch {
            reason: format!(
                "KKT rho correction dimension mismatch: lambdas={} coords={} rhs={}",
                lambdas.len(),
                solution.penalty_coords.len(),
                k
            ),
        }
        .into());
    }
    if upper_active_rho.len() != k {
        return Err(RemlError::DimensionMismatch {
            reason: format!(
                "KKT rho correction active-bound mask mismatch: mask={} rhs={}",
                upper_active_rho.len(),
                k
            ),
        }
        .into());
    }
    if residual.len() != hop.dim() {
        return Err(RemlError::DimensionMismatch {
            reason: format!(
                "KKT residual dimension mismatch: residual={} Hessian dim={}",
                residual.len(),
                hop.dim()
            ),
        }
        .into());
    }

    let subspace = solution.penalty_subspace_trace.as_deref();
    let q = solve_kkt_residual_kernel(hop, subspace, residual);
    let mut a_i_qs = Vec::with_capacity(k);
    let mut a_i_dot_q = Vec::with_capacity(k);
    let mut q_a_i_q = Vec::with_capacity(k);

    for idx in 0..k {
        if upper_active_rho[idx] {
            a_i_dot_q.push(0.0);
            q_a_i_q.push(0.0);
            a_i_qs.push(Array1::<f64>::zeros(hop.dim()));
            continue;
        }
        let a_i_q = solution.penalty_coords[idx].scaled_matvec(&q, lambdas[idx]);
        let linear = penalty_a_k_betas[idx].dot(&q);
        let quadratic = q.dot(&a_i_q);
        if !linear.is_finite() || !quadratic.is_finite() {
            return Err(RemlError::NonFiniteValue {
                reason: format!(
                    "KKT rho correction produced non-finite gradient ingredients at coord {idx}: \
                     linear={linear} quadratic={quadratic}"
                ),
            }
            .into());
        }
        a_i_dot_q.push(linear);
        q_a_i_q.push(quadratic);
        a_i_qs.push(a_i_q);
    }

    let mut gradient = Array1::<f64>::zeros(k);
    for idx in 0..k {
        if !upper_active_rho[idx] {
            gradient[idx] = -a_i_dot_q[idx] + 0.5 * q_a_i_q[idx];
        }
    }

    let hessian = if include_hessian {
        let mut a_solutions = Vec::with_capacity(k);
        let mut q_derivs = Vec::with_capacity(k);
        for idx in 0..k {
            if upper_active_rho[idx] {
                a_solutions.push(Array1::<f64>::zeros(hop.dim()));
                q_derivs.push(Array1::<f64>::zeros(hop.dim()));
                continue;
            }
            a_solutions.push(solve_kkt_residual_kernel(
                hop,
                subspace,
                &penalty_a_k_betas[idx],
            ));
            let mut rhs = penalty_a_k_betas[idx].clone();
            rhs -= &a_i_qs[idx];
            q_derivs.push(solve_kkt_residual_kernel(hop, subspace, &rhs));
        }

        let entry = |i: usize, j: usize| -> f64 {
            if upper_active_rho[i] || upper_active_rho[j] {
                return 0.0;
            }
            let delta = if i == j { 1.0 } else { 0.0 };
            let cancel_exact_kkt_profile_term = penalty_a_k_betas[i].dot(&a_solutions[j]);
            cancel_exact_kkt_profile_term
                - delta * a_i_dot_q[i]
                - penalty_a_k_betas[i].dot(&q_derivs[j])
                + q_derivs[j].dot(&a_i_qs[i])
                + 0.5 * delta * q_a_i_q[i]
        };

        let mut h = Array2::<f64>::zeros((k, k));
        for i in 0..k {
            for j in i..k {
                let raw = if i == j {
                    entry(i, j)
                } else {
                    0.5 * (entry(i, j) + entry(j, i))
                };
                if !raw.is_finite() {
                    return Err(RemlError::NonFiniteValue {
                        reason: format!(
                            "KKT rho correction produced non-finite Hessian entry ({i}, {j}): {raw}"
                        ),
                    }
                    .into());
                }
                h[[i, j]] = raw;
                if i != j {
                    h[[j, i]] = raw;
                }
            }
        }
        Some(h)
    } else {
        None
    };

    Ok(KktRhoCorrections { gradient, hessian })
}
