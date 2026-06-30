//! KKT-residual θ corrections and the shared REML derivative workspace.
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
/// r(θ)`, `K = H⁻¹`, over the FULL θ = (ρ ‖ ψ) coordinate set covering the ρρ,
/// cross-ρψ, and ψψ blocks uniformly.
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
/// With `q = K r` and the FIRST derivatives of the correction layer:
///   C_i  = −r_iᵀ q + ½ qᵀ A_i q,
///   q_j  = ∂_θⱼ q = K(r_j − A_j q),
///   C_ij = −r_iᵀ q_j + q_jᵀ A_i q  +  δ_ij·[−(∂_ir_i)ᵀq + ½qᵀ(∂_iA_i)q].
/// The bracketed diagonal term is the SECOND self-derivative of the correction
/// (`∂²_θᵢ r` and `∂²_θᵢ H`), and it is non-zero precisely for coordinates whose
/// score-derivative and drift scale MULTIPLICATIVELY with their own coordinate.
/// The ρ coordinates are exactly such: `λᵢ = exp(ρᵢ)`, so `rᵢ = λᵢSᵢβ̂` and
/// `Aᵢ = λᵢSᵢ` both obey `∂_ρᵢ rᵢ = rᵢ` and `∂_ρᵢ Aᵢ = Aᵢ`, which collapses the
/// bracket to exactly `Cᵢ` (the gradient correction already computed at this
/// coordinate). `exponential_self_coupling[i]` flags those coordinates; for an
/// affine/frozen coordinate (`r`, `A` linear in θ ⇒ `∂²r = ∂²H = 0`, e.g. the
/// ψ/ext frozen-drift block whose genuine second order lives elsewhere) the
/// bracket is zero and the flag is `false`.
///
/// This self-derivative term is NOT carried by the outer pair assembly: with no
/// KKT residual there is no correction `C(θ)` at all, so its curvature can only
/// enter through this block. The dense/operator outer Hessian carries the
/// exact-KKT profile term `−r_iᵀ K r_j` for EVERY block (ρρ via the penalty
/// profile, ρψ/ψψ via the IFT profile `a_iᵀ K a_j` in `outer_hessian_entry`),
/// valid only at `r = 0`, so the correction adds back `r_iᵀ K r_j + C_ij`; the
/// additive block vanishes identically at exact KKT (`r = 0 ⇒ q = 0`, and the
/// diagonal self-term is `Cᵢ = 0` there too).
///
/// `active[i]` masks a coordinate at an active upper box bound (only ρ
/// coordinates can be masked; the ψ entries are always `false`): a masked
/// coordinate freezes (its θ does not move), so its row/column correction is
/// zero.
pub(crate) fn compute_kkt_residual_theta_corrections<F>(
    hop: &dyn HessianOperator,
    subspace: Option<&PenaltySubspaceTrace>,
    score_derivs: &[Array1<f64>],
    drift_apply: F,
    residual: &Array1<f64>,
    include_hessian: bool,
    active: &[bool],
    exponential_self_coupling: &[bool],
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
    if exponential_self_coupling.len() != m {
        return Err(RemlError::DimensionMismatch {
            reason: format!(
                "KKT theta correction self-coupling mask mismatch: mask={} coords={}",
                exponential_self_coupling.len(),
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
        // profile term is `r_iᵀ a_solutions[j]`. Coordinate second-derivative
        // terms are intentionally absent here; the outer pair assembly owns
        // them, and duplicating them in this residual-only correction inflates
        // the diagonal.
        let entry = |i: usize, j: usize| -> f64 {
            if active[i] || active[j] {
                return 0.0;
            }
            let cancel_exact_kkt_profile_term = score_derivs[i].dot(&a_solutions[j]);
            cancel_exact_kkt_profile_term - score_derivs[i].dot(&q_derivs[j])
                + q_derivs[j].dot(&a_i_qs[i])
        };

        let mut h = Array2::<f64>::zeros((m, m));
        for i in 0..m {
            for j in i..m {
                let raw = if i == j {
                    // Diagonal: add the second self-derivative `δ_ij·C_i` for
                    // coordinates whose r_i and A_i scale multiplicatively with
                    // their own coordinate (λ = exp(ρ)); `gradient[i]` already
                    // equals `C_i = −r_iᵀq + ½qᵀA_iq`, and `∂_ρᵢrᵢ = rᵢ`,
                    // `∂_ρᵢAᵢ = Aᵢ` collapse the bracket to exactly C_i. Affine/
                    // frozen coordinates (`∂²r = ∂²H = 0`) get no such term.
                    let self_term = if exponential_self_coupling[i] && !active[i] {
                        gradient[i]
                    } else {
                        0.0
                    };
                    entry(i, j) + self_term
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
        let projected = gam_linalg::faer_ndarray::fast_atv(&kernel.u_s, rhs);
        let solved_projected = kernel.h_proj_inverse.dot(&projected);
        gam_linalg::faer_ndarray::fast_av(&kernel.u_s, &solved_projected)
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
                .unwrap_or(crate::estimate::RHO_BOUND);
            upper.is_finite() && value >= upper - 1.0e-8
        })
        .collect()
}
