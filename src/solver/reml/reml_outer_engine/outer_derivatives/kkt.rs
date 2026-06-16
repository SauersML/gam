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
/// beta-Gaussian prior atom emission / `hop.solve` / drift-correction work is
/// not repeated.
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

/// The KKT-residual correction over the FULL θ = (ρ ‖ ψ) coordinate set.
///
/// `gradient` is `k_outer = k + ext_dim` long; `hessian`, when present, is the
/// full `k_outer × k_outer` block covering ρρ, the cross ρψ, and ψψ. The
/// caller adds it to the whole outer Hessian (the operator's `dim` equals
/// `k_outer`), not just the top-left ρ block.
pub(crate) struct KktThetaCorrections {
    pub(crate) gradient: Array1<f64>,
    pub(crate) hessian: Option<Array2<f64>>,
}

/// Exact derivatives of the Newton/IFT residual correction `C(θ) = −½ r(θ)ᵀ K
/// r(θ)`, `K = H⁻¹`, over the FULL θ = (ρ ‖ ψ) coordinate set — the cross-ρψ
/// and ψψ generalization of [`compute_kkt_residual_rho_corrections`].
///
/// At fixed β̂ each coordinate `i` contributes a score derivative `r_i = ∂_θᵢ r`
/// and a frozen Hessian drift `A_i = ∂_θᵢ H|_β̂`:
///   * ρ coordinates: `r_i = a_i = λ_i S_i β̂`, `A_i[v] = λ_i S_i v`;
///   * ψ/ext coordinates: `r_i = coord.g` (= score_ψ + S_ψ β̂),
///     `A_i[v] = B_i v` (the FROZEN ψ Hessian drift, no IFT correction).
/// Both are supplied uniformly by the caller through `score_derivs` and the
/// `drift_apply(i, v) = A_i v` closure, so the SAME algebra produces every
/// block — there is no separate ρ-only and ψ-only code path to drift apart.
///
/// With `q = K r`:
///   C_i  = −r_iᵀ q + ½ qᵀ A_i q,
///   q_j  = K(r_j − A_j q),
///   C_ij = −δ_ij r_iᵀ q − r_iᵀ q_j + q_jᵀ A_i q + ½ δ_ij qᵀ A_i q.
/// The dense/operator outer Hessian already carries the exact-KKT profile term
/// `−r_iᵀ K r_j` for EVERY block (ρρ via the penalty profile, ρψ/ψψ via the
/// IFT profile `a_iᵀ K a_j` in `outer_hessian_entry`), valid only at `r = 0`,
/// so the correction adds back `r_iᵀ K r_j + C_ij`; the additive block vanishes
/// identically at exact KKT (`r = 0 ⇒ q = 0`).
///
/// `active[i]` masks a coordinate at an active upper box bound (only ρ
/// coordinates can be masked; the ψ entries are always `false`): a masked
/// coordinate freezes (its θ does not move), so its row/column correction is
/// zero, exactly as in the ρ-only path.
pub(crate) fn compute_kkt_residual_theta_corrections<F>(
    hop: &dyn HessianOperator,
    subspace: Option<&PenaltySubspaceTrace>,
    score_derivs: &[Array1<f64>],
    drift_apply: F,
    residual: &Array1<f64>,
    include_hessian: bool,
    active: &[bool],
) -> Result<KktThetaCorrections, String>
where
    F: Fn(usize, &Array1<f64>) -> Array1<f64>,
{
    let m = score_derivs.len();
    if m == 0 {
        return Ok(KktThetaCorrections {
            gradient: Array1::zeros(0),
            hessian: include_hessian.then(|| Array2::zeros((0, 0))),
        });
    }
    if active.len() != m {
        return Err(RemlError::DimensionMismatch {
            reason: format!(
                "KKT theta correction active-bound mask mismatch: mask={} coords={}",
                active.len(),
                m
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

    let q = solve_kkt_residual_kernel(hop, subspace, residual);
    let mut a_i_qs = Vec::with_capacity(m);
    let mut r_i_dot_q = Vec::with_capacity(m);
    let mut q_a_i_q = Vec::with_capacity(m);

    for idx in 0..m {
        if active[idx] {
            r_i_dot_q.push(0.0);
            q_a_i_q.push(0.0);
            a_i_qs.push(Array1::<f64>::zeros(hop.dim()));
            continue;
        }
        let a_i_q = drift_apply(idx, &q);
        let linear = score_derivs[idx].dot(&q);
        let quadratic = q.dot(&a_i_q);
        if !linear.is_finite() || !quadratic.is_finite() {
            return Err(RemlError::NonFiniteValue {
                reason: format!(
                    "KKT theta correction produced non-finite gradient ingredients at coord \
                     {idx}: linear={linear} quadratic={quadratic}"
                ),
            }
            .into());
        }
        r_i_dot_q.push(linear);
        q_a_i_q.push(quadratic);
        a_i_qs.push(a_i_q);
    }

    let mut gradient = Array1::<f64>::zeros(m);
    for idx in 0..m {
        if !active[idx] {
            gradient[idx] = -r_i_dot_q[idx] + 0.5 * q_a_i_q[idx];
        }
    }

    let hessian = if include_hessian {
        let mut a_solutions = Vec::with_capacity(m);
        let mut q_derivs = Vec::with_capacity(m);
        for idx in 0..m {
            if active[idx] {
                a_solutions.push(Array1::<f64>::zeros(hop.dim()));
                q_derivs.push(Array1::<f64>::zeros(hop.dim()));
                continue;
            }
            a_solutions.push(solve_kkt_residual_kernel(hop, subspace, &score_derivs[idx]));
            let mut rhs = score_derivs[idx].clone();
            rhs -= &a_i_qs[idx];
            q_derivs.push(solve_kkt_residual_kernel(hop, subspace, &rhs));
        }

        // C_ij + (exact-KKT profile term r_iᵀ K r_j that the dense/operator
        // outer Hessian already subtracted). `a_solutions[j] = K r_j`, so the
        // profile term is `r_iᵀ a_solutions[j]`.
        let entry = |i: usize, j: usize| -> f64 {
            if active[i] || active[j] {
                return 0.0;
            }
            let delta = if i == j { 1.0 } else { 0.0 };
            let cancel_exact_kkt_profile_term = score_derivs[i].dot(&a_solutions[j]);
            cancel_exact_kkt_profile_term - delta * r_i_dot_q[i] - score_derivs[i].dot(&q_derivs[j])
                + q_derivs[j].dot(&a_i_qs[i])
                + 0.5 * delta * q_a_i_q[i]
        };

        let mut h = Array2::<f64>::zeros((m, m));
        for i in 0..m {
            for j in i..m {
                let raw = if i == j {
                    entry(i, j)
                } else {
                    0.5 * (entry(i, j) + entry(j, i))
                };
                if !raw.is_finite() {
                    return Err(RemlError::NonFiniteValue {
                        reason: format!(
                            "KKT theta correction produced non-finite Hessian entry ({i}, {j}): \
                             {raw}"
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

    Ok(KktThetaCorrections { gradient, hessian })
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

/// ρ-only entry point for the Newton/IFT residual correction
/// `C(ρ) = −½ r(ρ)ᵀ K r(ρ)`, `K = H⁻¹` — now a thin marshalling shim over the
/// full-θ [`compute_kkt_residual_theta_corrections`].
///
/// At fixed β̂ the ρ ingredients are `r_i = a_i = λ_iS_iβ̂` and the frozen drift
/// `A_i[v] = λ_iS_i v`; this builds them and forwards to the generalized
/// calculus, so there is exactly ONE implementation of the `C_i`/`C_ij`
/// algebra (the no-parallel-math rule). Retained so the ρ-isolation pins
/// (`ift_gradient_correction_with_zero_projected_residual_is_zero`,
/// `ift_rho_upper_bound_masks_residual_correction_direction`) keep guarding the
/// ρ contract directly while the production path uses the full-θ form.
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
    let subspace = solution.penalty_subspace_trace.as_deref();
    let drift_apply = |idx: usize, v: &Array1<f64>| -> Array1<f64> {
        solution.penalty_coords[idx].scaled_matvec(v, lambdas[idx])
    };
    let theta = compute_kkt_residual_theta_corrections(
        hop,
        subspace,
        penalty_a_k_betas,
        drift_apply,
        residual,
        include_hessian,
        upper_active_rho,
    )?;
    Ok(KktRhoCorrections {
        gradient: theta.gradient,
        hessian: theta.hessian,
    })
}
