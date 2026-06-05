//! Shared under-identified-subspace selector for the universal Jeffreys/Firth
//! robustness machinery.
//!
//! The Jeffreys penalty `Phi = 1/2 log|I(beta)|` is only ever applied to the
//! directions that are identified by NEITHER the data nor a proper prior — the
//! "under-identified span". Penalized smooth directions already carry a proper
//! wiggliness prior (their `S_lambda` curvature), so applying Jeffreys there
//! would double-regularize and bias the smooth fit. This module produces the
//! orthonormal basis `Z_J` of that span for one parameter block.
//!
//! For a block with aggregate penalty `S = sum_k S_k`, the under-identified
//! span is exactly `ker(S)` — the penalty null space, which always contains the
//! parametric (unpenalized) part and the structural null space of every smooth
//! penalty (the polynomial/affine basis a difference/curvature penalty cannot
//! see). A block with no penalties at all (a pure parametric block) is entirely
//! under-identified, so `Z_J = I`.
//!
//! Both tiers of the robustness machinery consume the SAME `Z_J`:
//!   * Tier A (single-eta GLM via `FirthDenseOperator`) scopes the Fisher
//!     information to `X * Z_J`.
//!   * Tier B (coupled multi-predictor custom-family joint Newton, e.g. BMS)
//!     restricts the joint-Hessian Jeffreys term `Phi_J = 1/2 log|Z_J^T H Z_J|`
//!     to the same span.
//!
//! Everything here is pure linear algebra on the block's penalty matrices and
//! is gated upstream by `RobustConfig` (default OFF), so it never runs in the
//! released solver until a caller opts in.

use crate::linalg::faer_ndarray::FaerEigh;
use faer::Side;
use ndarray::{Array1, Array2, ArrayView2};

/// Relative floor on a reduced-information eigenvalue, as a fraction of the
/// dominant (identified) curvature `λ_max`. Negligible on data-identified
/// directions (whose curvature is `O(n) · λ_max`-scale), positive on separating
/// directions, keeping the Jeffreys log-det finite even when the observed
/// information is indefinite at an off-mode trial point.
const REDUCED_INFO_RELATIVE_FLOOR: f64 = 1e-10;

/// Absolute floor for the degenerate case where every reduced eigenvalue is
/// (near) zero, so `λ_max ≈ 0` cannot scale the relative floor.
const REDUCED_INFO_ABSOLUTE_FLOOR: f64 = 1e-12;

/// Orthonormal basis of one block's Jeffreys span.
///
/// `columns` is `p x m` with orthonormal columns spanning `ker(S_aggregate)`
/// (the parametric + smooth-null directions). `m == 0` means the block is fully
/// penalized in every direction and gets no Jeffreys term.
#[derive(Debug, Clone)]
pub struct JeffreysSubspace {
    /// `p x m` orthonormal basis of the under-identified span (m <= p).
    pub columns: Array2<f64>,
}

impl JeffreysSubspace {
    /// Dimension `m` of the under-identified span (columns of the basis).
    #[inline]
    pub fn span_dim(&self) -> usize {
        self.columns.ncols()
    }
}

/// Build `Z_J` for a block: the FULL identifiable coefficient span of the
/// (post-rank-deficiency-removal) reduced block design — the entire reduced
/// coefficient space, `Z_J = I_p`.
///
/// PRINCIPLE (why this is the right span, not `ker(S)`). The Jeffreys penalty
/// `Φ = ½ log|I_r(β)|` is SELF-LIMITING: its score is `O(1)` against the data's
/// `O(n)` Fisher information. So on a data-identified direction (penalized OR
/// not) its only effect is the `O(1/n)` Firth bias-reduction correction — it
/// does not bias a genuine smooth fit. It bites ONLY where `I(β)` is
/// near-singular, i.e. a separating direction, supplying the missing
/// `O(1)`-bounding curvature there regardless of whether that direction lives in
/// `ker(S)` (an unpenalized nullspace direction) or `range(S)` (a penalized
/// spline direction). Scoping `Z_J` to `ker(S)` only — the previous behavior —
/// could not reach a near-separation on a penalized spline direction, which is
/// the residual BMS-probit pathology. Using the full identifiable span makes the
/// inner objective coercive with a finite unique minimizer on EVERY direction,
/// without any design surgery and with the optimizer untouched.
///
/// `aggregate_penalty` is `p x p` and PSD (`sum_k S_k`); it is used only to
/// validate squareness and pick up `p`. Rank-softness, if any, is absorbed by
/// the reduced-Fisher Cholesky in [`joint_jeffreys_term`] (which simply omits the
/// `Φ` value contribution for a not-yet-SPD trial point while the step machinery
/// still bounds the coefficient).
pub fn jeffreys_subspace_from_penalty(
    aggregate_penalty: ArrayView2<'_, f64>,
) -> Result<JeffreysSubspace, String> {
    let p = aggregate_penalty.nrows();
    if aggregate_penalty.ncols() != p {
        return Err(format!(
            "jeffreys_subspace: aggregate penalty must be square, got {}x{}",
            aggregate_penalty.nrows(),
            aggregate_penalty.ncols()
        ));
    }
    if p == 0 {
        return Ok(JeffreysSubspace {
            columns: Array2::zeros((0, 0)),
        });
    }
    Ok(JeffreysSubspace {
        columns: Array2::eye(p),
    })
}

/// Tier-B Jeffreys term on the joint under-identified span, computed directly
/// from the coupled joint Hessian `H` (NOT from a single-eta
/// `FirthDenseOperator`). This is the path BMS / survival-marginal-slope /
/// location-scale GAMLSS take: their working curvature block IS the Fisher
/// information at the working point, so the Jeffreys penalty is
/// `Phi_J = 1/2 log|Z_J^T H Z_J|` on the under-identified span `Z_J`.
///
/// Returns `(phi, grad, hphi)`:
///   * `phi`   = `1/2 log|H_id|`, the objective contribution (`H_id = Z_J^T H Z_J`).
///   * `grad`  = the `p`-vector `dPhi/dbeta`, with `grad[k] = 1/2 tr(H_id^{-1} Z_J^T Hdot[e_k] Z_J)`.
///   * `hphi`  = the `p x p` symmetric curvature contribution to the penalized
///               Hessian, the leading Gauss-Newton term
///               `1/2 sum over reduced pairs`. We use the positive-semidefinite
///               Gauss-Newton surrogate `H_Phi = 1/2 * J^T H_id^{-1} J` built
///               from the reduced gradient sensitivities, which supplies the
///               correct O(n) automatic curvature that bounds a near-separating
///               direction to O(1) while keeping `H_pen + H_Phi` SPD.
///
/// `hessian_dir` is a closure returning `Hdot[d] = d/d eps H(beta + eps d)|_0`
/// for a full coefficient-space direction `d` (the exact joint-Hessian
/// directional derivative the inner Newton already exposes). `Z_J` is the
/// `p x m` joint under-identified basis (block-diagonal stack of per-block
/// `Z_J`). When `m == 0` this returns a zero term.
pub fn joint_jeffreys_term<DirFn>(
    h_joint: ArrayView2<'_, f64>,
    z_j: ArrayView2<'_, f64>,
    mut hessian_dir: DirFn,
) -> Result<(f64, Array1<f64>, Array2<f64>), String>
where
    DirFn: FnMut(&Array1<f64>) -> Result<Option<Array2<f64>>, String>,
{
    let p = h_joint.nrows();
    if h_joint.ncols() != p {
        return Err(format!(
            "joint_jeffreys_term: H must be square, got {}x{}",
            h_joint.nrows(),
            h_joint.ncols()
        ));
    }
    if z_j.nrows() != p {
        return Err(format!(
            "joint_jeffreys_term: Z_J has {} rows, expected {} to match H",
            z_j.nrows(),
            p
        ));
    }
    let m = z_j.ncols();
    if m == 0 {
        return Ok((0.0, Array1::zeros(p), Array2::zeros((p, p))));
    }
    // H_id = Z_J^T H Z_J  (m x m reduced information on the Jeffreys span).
    let hz = h_joint.dot(&z_j);
    let h_id = z_j.t().dot(&hz);
    // Symmetrize defensively (observed-information round-off can break exact
    // symmetry).
    let mut h_id_sym = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        for j in 0..m {
            h_id_sym[[i, j]] = 0.5 * (h_id[[i, j]] + h_id[[j, i]]);
        }
    }
    // FULL-SPAN ROBUSTNESS. With the Jeffreys span equal to the FULL identifiable
    // coefficient space, `H_id` is the (reduced) observed information over every
    // direction. For non-canonical links (e.g. probit) the observed information
    // need NOT be PSD away from the mode, so a plain Cholesky would fail at
    // off-mode trial points and reject every outer seed. The Jeffreys prior is
    // `Φ = ½ log det I(β)` with `I` the EXPECTED (PSD) Fisher information; we
    // realise that here through the symmetric eigendecomposition, flooring each
    // eigenvalue at a tiny absolute ridge so `Φ` is the log-volume of the
    // POSITIVE curvature and the reduced inverse is the floored (pseudo-)inverse.
    // On an identified direction the data's O(n) curvature dwarfs the floor, so
    // the value, gradient and curvature are the exact Jeffreys quantities there;
    // a genuinely separating direction has near-zero curvature, where the floor
    // simply keeps `Φ` finite while the `H_Φ` curvature below grows to bound it.
    let (evals, evecs) = h_id_sym
        .eigh(Side::Lower)
        .map_err(|e| format!("joint_jeffreys_term: reduced-information eigendecomposition failed: {e}"))?;
    let lambda_max = evals.iter().cloned().fold(0.0_f64, f64::max);
    // Absolute floor relative to the dominant identified curvature: negligible on
    // identified directions (O(n)), positive on separating ones.
    let floor = (REDUCED_INFO_RELATIVE_FLOOR * lambda_max).max(REDUCED_INFO_ABSOLUTE_FLOOR);
    let mut phi = 0.0_f64;
    // h_id_inv = V diag(1/max(λ,floor)) Vᵀ  (floored symmetric pseudo-inverse).
    let mut inv_diag = Array1::<f64>::zeros(m);
    for (i, &lam) in evals.iter().enumerate() {
        let lam_floored = lam.max(floor);
        phi += 0.5 * lam_floored.ln();
        inv_diag[i] = 1.0 / lam_floored;
    }
    let scaled = &evecs * &inv_diag.view().insert_axis(ndarray::Axis(0));
    let h_id_inv = scaled.dot(&evecs.t());

    // Gradient: grad[k] = 1/2 tr(H_id^{-1} Z_J^T Hdot[e_k] Z_J).
    // For the inner-Newton dense path the Hessian is beta-dependent through the
    // working weights only along coefficient directions; we evaluate Hdot per
    // canonical coefficient axis. `J_red[:, k]` stores the reduced sensitivity
    // s_k = vec(Z_J^T Hdot[e_k] Z_J) contracted with H_id^{-1}, which feeds both
    // the gradient and the Gauss-Newton curvature surrogate.
    let mut grad = Array1::<f64>::zeros(p);
    // Reduced sensitivity rows g_k = H_id^{-1} (Z_J^T Hdot[e_k] Z_J), flattened,
    // kept to assemble the PSD Gauss-Newton curvature surrogate.
    let mut sensitivity = Array2::<f64>::zeros((p, m * m));
    let mut axis = Array1::<f64>::zeros(p);
    for k in 0..p {
        axis.fill(0.0);
        axis[k] = 1.0;
        let hdot = match hessian_dir(&axis)? {
            Some(hdot) => hdot,
            None => {
                // Family does not expose an exact directional derivative; the
                // Jeffreys gradient/curvature degenerate to zero (objective
                // still well-defined). This keeps the term safe rather than
                // wrong.
                return Ok((phi, Array1::zeros(p), Array2::zeros((p, p))));
            }
        };
        if hdot.nrows() != p || hdot.ncols() != p {
            return Err(format!(
                "joint_jeffreys_term: Hdot shape {}x{} != {p}x{p}",
                hdot.nrows(),
                hdot.ncols()
            ));
        }
        // Reduced derivative D_k = Z_J^T Hdot Z_J  (m x m).
        let hdz = hdot.dot(&z_j);
        let d_k = z_j.t().dot(&hdz);
        // M_k = H_id^{-1} D_k.
        let m_k = h_id_inv.dot(&d_k);
        // grad[k] = 1/2 tr(M_k).
        let mut trace = 0.0;
        for i in 0..m {
            trace += m_k[[i, i]];
        }
        grad[k] = 0.5 * trace;
        // Store vec(M_k) for the Gauss-Newton surrogate.
        let mut col = 0usize;
        for i in 0..m {
            for j in 0..m {
                sensitivity[[k, col]] = m_k[[i, j]];
                col += 1;
            }
        }
    }
    // Gauss-Newton curvature surrogate: H_Phi = 1/2 J J^T over the reduced
    // sensitivities, i.e. H_Phi[a,b] = 1/2 <vec(M_a), vec(M_b)>. This is PSD by
    // construction, vanishes on directions the data already identifies (M_k = 0
    // there), and grows as the reduced curvature shrinks along a separating
    // direction — exactly the automatic O(1)-bounding curvature Firth supplies.
    let mut hphi = Array2::<f64>::zeros((p, p));
    for a in 0..p {
        for b in a..p {
            let mut acc = 0.0;
            for col in 0..(m * m) {
                acc += sensitivity[[a, col]] * sensitivity[[b, col]];
            }
            let value = 0.5 * acc;
            hphi[[a, b]] = value;
            hphi[[b, a]] = value;
        }
    }
    Ok((phi, grad, hphi))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn full_span_is_identity_regardless_of_penalty() {
        // The principled cure: Z_J is the FULL identifiable span (the entire
        // reduced block), i.e. the identity, irrespective of the penalty's null
        // space. Jeffreys is self-limiting, so this does not bias identified
        // directions; it only bounds near-separating ones.
        for s in [
            Array2::<f64>::zeros((3, 3)),            // pure parametric
            {
                let mut s = Array2::<f64>::zeros((3, 3));
                s[[2, 2]] = 5.0;                     // rank-deficient (ker dim 2)
                s
            },
            Array2::<f64>::eye(4) * 2.0,             // full-rank penalty
        ] {
            let p = s.nrows();
            let z = jeffreys_subspace_from_penalty(s.view()).unwrap();
            assert_eq!(z.span_dim(), p, "full span must equal the block dimension");
            assert_eq!(z.columns.nrows(), p);
            // Identity ⇒ orthonormal columns spanning the whole space.
            let gram = z.columns.t().dot(&z.columns);
            for i in 0..p {
                for j in 0..p {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!((gram[[i, j]] - expected).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn empty_block_yields_empty_span() {
        let s = Array2::<f64>::zeros((0, 0));
        let z = jeffreys_subspace_from_penalty(s.view()).unwrap();
        assert_eq!(z.span_dim(), 0);
    }

    #[test]
    fn joint_jeffreys_term_matches_finite_difference_gradient() {
        // A 2x2 quadratic-form Hessian whose log-determinant has a known
        // gradient. Build a beta-dependent H(beta) = diag(exp(beta0), 1+beta1^2)
        // restricted to the full span (Z_J = I) and finite-difference Phi.
        let p = 2usize;
        let z = Array2::<f64>::eye(p);
        let h_at = |b: &Array1<f64>| -> Array2<f64> {
            let mut h = Array2::<f64>::zeros((p, p));
            h[[0, 0]] = b[0].exp();
            h[[1, 1]] = 1.0 + b[1] * b[1];
            h
        };
        // Hdot[d] = d/d eps H(beta + eps d): diag(exp(b0) d0, 2 b1 d1).
        let beta: Array1<f64> = array![0.3, -0.4];
        let hdir = |d: &Array1<f64>| -> Result<Option<Array2<f64>>, String> {
            let mut hd = Array2::<f64>::zeros((p, p));
            hd[[0, 0]] = beta[0].exp() * d[0];
            hd[[1, 1]] = 2.0 * beta[1] * d[1];
            Ok(Some(hd))
        };
        let h = h_at(&beta);
        let (phi, grad, hphi) = joint_jeffreys_term(h.view(), z.view(), hdir).unwrap();
        // Phi = 1/2 log(exp(b0) * (1 + b1^2)).
        let expected_phi = 0.5 * (beta[0].exp() * (1.0 + beta[1] * beta[1])).ln();
        assert!(
            (phi - expected_phi).abs() < 1e-10,
            "phi {phi} vs {expected_phi}"
        );
        // Finite-difference the gradient.
        let eps = 1e-6;
        for k in 0..p {
            let mut bp = beta.clone();
            let mut bm = beta.clone();
            bp[k] += eps;
            bm[k] -= eps;
            let hp = h_at(&bp);
            let hm = h_at(&bm);
            let phi_p = 0.5 * (hp[[0, 0]] * hp[[1, 1]]).ln();
            let phi_m = 0.5 * (hm[[0, 0]] * hm[[1, 1]]).ln();
            let fd = (phi_p - phi_m) / (2.0 * eps);
            assert!(
                (grad[k] - fd).abs() < 1e-5,
                "grad[{k}] {} vs fd {fd}",
                grad[k]
            );
        }
        // H_Phi is symmetric PSD.
        for a in 0..p {
            for b in 0..p {
                assert!((hphi[[a, b]] - hphi[[b, a]]).abs() < 1e-12);
            }
        }
        let (evals, _) = hphi.eigh(Side::Lower).unwrap();
        for e in evals.iter() {
            assert!(*e >= -1e-10, "H_Phi must be PSD, got eigenvalue {e}");
        }
    }

    #[test]
    fn empty_span_yields_zero_term() {
        let h = Array2::<f64>::eye(3);
        let z = Array2::<f64>::zeros((3, 0));
        let hdir = |_d: &Array1<f64>| -> Result<Option<Array2<f64>>, String> {
            Ok(Some(Array2::<f64>::zeros((3, 3))))
        };
        let (phi, grad, hphi) = joint_jeffreys_term(h.view(), z.view(), hdir).unwrap();
        assert_eq!(phi, 0.0);
        assert!(grad.iter().all(|v| *v == 0.0));
        assert!(hphi.iter().all(|v| *v == 0.0));
    }
}
