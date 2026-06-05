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
//! The under-identified span is the FULL identifiable coefficient span of the
//! (post-rank-deficiency-removal) reduced block — `Z_J = I_p` — NOT the penalty
//! null space `ker(S)`. The Jeffreys penalty is self-limiting (its `O(1)` score
//! is dominated by the data's `O(n)` Fisher information), so on a data-identified
//! direction (penalized OR not) its only effect is the `O(1/n)` Firth bias
//! correction; it bites only where the information is near-singular. Using the
//! full span — rather than scoping to `ker(S)` — lets it reach a near-separation
//! on a penalized spline direction too (the residual BMS-probit pathology). The
//! aggregate penalty is consulted only to pick up the block dimension `p`.
//!
//! Both tiers of the robustness machinery consume the SAME `Z_J`:
//!   * Tier A (single-eta GLM via `FirthDenseOperator`) scopes the Fisher
//!     information to `X * Z_J`.
//!   * Tier B (coupled multi-predictor custom-family joint Newton, e.g. BMS)
//!     restricts the joint-Hessian Jeffreys term `Phi_J = 1/2 log|Z_J^T H Z_J|`
//!     to the same span.
//!
//! Everything here is pure linear algebra on the block's penalty matrices.
//! Robustness is the unconditional default; the conditioning gate in
//! [`joint_jeffreys_term`] (self-limiting, returns the exact zero contribution on
//! a well-conditioned fit) is the only "apply where needed" mechanism.

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

/// Conditioning gate. When the reduced information `H_id = Z_J^T H Z_J` is
/// well-conditioned — every direction's curvature is within this relative
/// factor of the dominant `λ_max` — the data identifies the WHOLE span at
/// `O(n)` strength and the self-limiting `O(1)` Jeffreys term is negligible
/// there (its only effect would be the `O(1/n)` Firth bias correction, which is
/// not what this machinery exists to supply). We therefore SKIP the term
/// entirely and return the zero contribution, so a clean/easy fit pays no cost
/// and stays byte-identical to the un-penalized inner Newton. The gate fires
/// only on the OTHER side: an ill-conditioned / near-separating reduced
/// information (`λ_min/λ_max` below this threshold), where the floored log-det
/// curvature below is exactly the `O(1)`-bounding term Firth supplies.
///
/// The threshold sits far from machine precision: at `1e-8` the worst-
/// conditioned direction is still 8 orders of magnitude from the absolute floor
/// (`REDUCED_INFO_RELATIVE_FLOOR = 1e-10`), i.e. comfortably identified rather
/// than separating, so nothing the term would actually bound is gated out.
///
/// NOTE: a relative ratio is SCALE-FREE in `n` — it cannot, on its own, tell a
/// near-separating direction (absolute curvature `O(1)`) from a well-identified
/// one (absolute curvature `O(n)`). At small `n` an absolutely-near-separating
/// direction can still clear this relative gate (if `λ_max` is also small), so it
/// is paired with the ABSOLUTE gate below; the term fires when EITHER gate
/// reports under-identification (see [`conditioning_gate_skips`]).
const CONDITIONING_GATE_RELATIVE: f64 = 1e-8;

/// Absolute-curvature conditioning gate (the `n`-aware half of the gate).
///
/// Separation is an ABSOLUTE statement about curvature, not a relative one: a
/// direction is near-separating when the data place `O(1)` Fisher information on
/// it — a handful of near-boundary observations — REGARDLESS of the sample size
/// `n`. A well-identified direction instead accumulates `O(n)` information (each
/// of `n` observations contributes `O(1)` curvature). The reduced information
/// `H_id = Z_Jᵀ H Z_J` here IS that observed/expected Fisher information (an
/// un-normalised sum over observations, NOT a per-observation average), so its
/// smallest eigenvalue `λ_min` is `O(n)` on an identified direction and `O(1)`
/// on a separating one — the two regimes are separated by an absolute `O(1)`
/// scale that does not move with `n`.
///
/// We therefore ALSO fire the Jeffreys term whenever `λ_min` falls below this
/// absolute scale, independent of the relative ratio. This catches the
/// small-`n` admixture-cline / near-separation regime the relative gate misses,
/// where `λ_max` is itself modest so a near-zero `λ_min` can still satisfy
/// `λ_min/λ_max ≥ 1e-8`.
///
/// THRESHOLD CHOICE. One observation contributes at most `O(1)` curvature to a
/// unit-scale direction (e.g. a binomial Fisher weight `p(1−p) ≤ 1/4`, a
/// Gaussian unit weight `1`), so a direction carrying less than a single
/// observation's worth of information is, by construction, not identified by the
/// data and is the regime Firth exists to stabilise. We set the gate at `1.0`:
/// `λ_min < 1` ⇒ the direction holds under one observation-equivalent of
/// curvature ⇒ treat as near-separating and fire the term. This is conservative
/// (it never fires on a genuinely well-conditioned large-`n` fit, whose
/// `λ_min = O(n) ≫ 1`, so the byte-identical clean-fit guarantee is preserved)
/// while catching absolute near-separation at any `n`. The design is assumed to
/// be on a standardized/O(1)-column scale, which the upstream reduction already
/// enforces; the floor below (`REDUCED_INFO_ABSOLUTE_FLOOR = 1e-12`) keeps the
/// log-det finite once the term fires.
const CONDITIONING_GATE_ABSOLUTE: f64 = 1.0;

/// Shared conditioning-gate predicate for the Jeffreys term, evaluated from the
/// reduced-information spectrum (`λ_min`, `λ_max`). Returns `true` when the term
/// should be SKIPPED (zero contribution) because the reduced information is
/// well-conditioned — both relatively (`λ_min/λ_max ≥ CONDITIONING_GATE_RELATIVE`)
/// AND absolutely (`λ_min ≥ CONDITIONING_GATE_ABSOLUTE`). If EITHER test reports
/// under-identification the gate does NOT skip and the floored log-det term
/// fires. Centralised so every call site (objective value, gradient/curvature,
/// and the `H_Φ` directional derivative) uses byte-identical gating — any
/// divergence would reintroduce the value/derivative mismatch the term removes.
#[inline]
fn conditioning_gate_skips(lambda_min: f64, lambda_max: f64) -> bool {
    if lambda_max <= 0.0 {
        // Degenerate / non-positive spectrum: not well-conditioned, never skip.
        return false;
    }
    if !lambda_min.is_finite() {
        return false;
    }
    let relative_ok = lambda_min / lambda_max >= CONDITIONING_GATE_RELATIVE;
    let absolute_ok = lambda_min >= CONDITIONING_GATE_ABSOLUTE;
    relative_ok && absolute_ok
}

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
    // CONDITIONING GATE ("no cost on easy fits"). The eigendecomposition we just
    // computed gives the full reduced spectrum; the worst-conditioned direction
    // is `λ_min`. We skip the term (zero value, gradient and curvature) only when
    // the reduced information is well-conditioned BOTH relatively
    // (`λ_min/λ_max ≥ CONDITIONING_GATE_RELATIVE`) AND absolutely
    // (`λ_min ≥ CONDITIONING_GATE_ABSOLUTE`, the `n`-aware criterion): every
    // direction is then identified by the data at `O(n)` strength, the
    // self-limiting Jeffreys term is negligible, and a clean/well-conditioned fit
    // stays byte-identical to the un-penalized inner Newton. If EITHER gate
    // reports under-identification — including an absolutely-near-separating
    // direction at small `n` that the scale-free relative ratio alone would miss
    // — we fall through to the floored log-det term below, the `O(1)`-bounding
    // curvature this machinery exists to supply.
    {
        let lambda_min = evals.iter().cloned().fold(f64::INFINITY, f64::min);
        if conditioning_gate_skips(lambda_min, lambda_max) {
            return Ok((0.0, Array1::zeros(p), Array2::zeros((p, p))));
        }
    }
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

/// Exact directional derivative `D_β H_Φ[δ]` of the Tier-B Gauss-Newton Jeffreys
/// curvature surrogate along a coefficient-space direction `δ` (`delta`).
///
/// CONTEXT (the outer-REML drift this exists to supply). The Tier-B outer LAML
/// score folds the joint Jeffreys curvature `H_Φ` into the joint Hessian logdet:
/// `½ log|H + S_λ + H_Φ|`. Its exact ρ-gradient is
///   `½ tr[(H+S_λ+H_Φ)⁻¹ (∂_ρ S_λ + D_β H[v_k] + D_β H_Φ[v_k])]`,
/// where `v_k = dβ̂/dρ_k` is the mode response and `D_β·[v_k]` is the total
/// (through β̂) derivative of the curvature along the mode response. The
/// likelihood-Hessian drift `D_β H[v_k]` is already supplied by the family's
/// joint directional-derivative provider; `H_Φ` ALSO moves with β̂ (it is built
/// from `H_id = Z_Jᵀ H Z_J` and `D_a = Z_Jᵀ ∂_a H Z_J`, both β-dependent), so its
/// drift `D_β H_Φ[δ]` is a real, non-zero term whenever the Jeffreys term is
/// active (near-separation). This function returns exactly that `p×p` term so the
/// outer gradient matches the objective the inner Newton converged on.
///
/// DERIVATION. With `K = H_id⁻¹` (the floored symmetric pseudo-inverse used as
/// the analytic inverse on the floored spectrum), `M_a = K D_a`,
/// `H_Φ[a,b] = ½⟨vec(M_a), vec(M_b)⟩`, and `δ` the direction:
///   * `δ_δ H_id = Ḋ := Z_Jᵀ Hdot[δ] Z_J`,   so `δ_δ K = −K Ḋ K`.
///   * `δ_δ D_a = Z_Jᵀ H²dot[δ, e_a] Z_J =: D_a^δ` (the second directional
///     derivative of the joint Hessian along `(δ, e_a)`).
///   * `δ_δ M_a = (δ_δ K) D_a + K (δ_δ D_a) = −K Ḋ M_a + K D_a^δ`.
///   * `δ_δ H_Φ[a,b] = ½[⟨vec(δ_δ M_a), vec(M_b)⟩ + ⟨vec(M_a), vec(δ_δ M_b)⟩]`.
///
/// `hessian_dir` returns `Hdot[d] = ∂_d H` and `hessian_second_dir` returns
/// `H²dot[u, v] = ∂_u ∂_v H`. When EITHER is unavailable (the family does not
/// expose the needed exact derivatives) or the conditioning gate skips the term
/// (so `H_Φ ≡ 0` in a neighborhood, hence `D_β H_Φ ≡ 0`), this returns the zero
/// matrix — the safe value that leaves the existing `D_β H[v_k]`-only gradient
/// unchanged rather than wrong.
pub fn joint_jeffreys_hphi_directional_derivative<DirFn, Dir2Fn>(
    h_joint: ArrayView2<'_, f64>,
    z_j: ArrayView2<'_, f64>,
    delta: &Array1<f64>,
    mut hessian_dir: DirFn,
    mut hessian_second_dir: Dir2Fn,
) -> Result<Array2<f64>, String>
where
    DirFn: FnMut(&Array1<f64>) -> Result<Option<Array2<f64>>, String>,
    Dir2Fn: FnMut(&Array1<f64>, &Array1<f64>) -> Result<Option<Array2<f64>>, String>,
{
    let p = h_joint.nrows();
    if h_joint.ncols() != p {
        return Err(format!(
            "joint_jeffreys_hphi_directional_derivative: H must be square, got {}x{}",
            h_joint.nrows(),
            h_joint.ncols()
        ));
    }
    if z_j.nrows() != p {
        return Err(format!(
            "joint_jeffreys_hphi_directional_derivative: Z_J has {} rows, expected {p}",
            z_j.nrows()
        ));
    }
    if delta.len() != p {
        return Err(format!(
            "joint_jeffreys_hphi_directional_derivative: delta has {} entries, expected {p}",
            delta.len()
        ));
    }
    let m = z_j.ncols();
    if m == 0 || p == 0 {
        return Ok(Array2::zeros((p, p)));
    }

    // Reproduce EXACTLY the value-path reduced information, conditioning gate, and
    // floored pseudo-inverse so the derivative is consistent with the `H_Φ` the
    // objective uses. Any divergence here would reintroduce the value/gradient
    // mismatch this whole term exists to remove.
    let hz0 = h_joint.dot(&z_j);
    let h_id = z_j.t().dot(&hz0);
    let mut h_id_sym = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        for j in 0..m {
            h_id_sym[[i, j]] = 0.5 * (h_id[[i, j]] + h_id[[j, i]]);
        }
    }
    let (evals, evecs) = h_id_sym.eigh(Side::Lower).map_err(|e| {
        format!("joint_jeffreys_hphi_directional_derivative: eigendecomposition failed: {e}")
    })?;
    let lambda_max = evals.iter().cloned().fold(0.0_f64, f64::max);
    {
        // Same combined relative+absolute gate as the value/gradient path, via the
        // shared predicate, so the derivative is byte-identically consistent with
        // the `H_Φ` the objective uses (any divergence reintroduces the
        // value/derivative mismatch this term exists to remove).
        let lambda_min = evals.iter().cloned().fold(f64::INFINITY, f64::min);
        if conditioning_gate_skips(lambda_min, lambda_max) {
            // Gated out ⇒ H_Φ ≡ 0 in a neighborhood ⇒ its drift vanishes.
            return Ok(Array2::zeros((p, p)));
        }
    }
    let floor = (REDUCED_INFO_RELATIVE_FLOOR * lambda_max).max(REDUCED_INFO_ABSOLUTE_FLOOR);
    let mut inv_diag = Array1::<f64>::zeros(m);
    for (i, &lam) in evals.iter().enumerate() {
        inv_diag[i] = 1.0 / lam.max(floor);
    }
    let scaled = &evecs * &inv_diag.view().insert_axis(ndarray::Axis(0));
    let h_id_inv = scaled.dot(&evecs.t());

    // Ḋ = Z_Jᵀ Hdot[δ] Z_J, the directional derivative of the reduced information
    // along the mode-response direction δ. δ_δ K = −K Ḋ K.
    let hdot_delta = match hessian_dir(delta)? {
        Some(hd) => hd,
        // No exact first directional derivative ⇒ drift undefined ⇒ safe zero.
        None => return Ok(Array2::zeros((p, p))),
    };
    if hdot_delta.nrows() != p || hdot_delta.ncols() != p {
        return Err(format!(
            "joint_jeffreys_hphi_directional_derivative: Hdot[δ] shape {}x{} != {p}x{p}",
            hdot_delta.nrows(),
            hdot_delta.ncols()
        ));
    }
    let dbar = z_j.t().dot(&hdot_delta.dot(&z_j)); // m x m
    let k_dbar = h_id_inv.dot(&dbar); // K Ḋ

    // For each canonical axis e_a: M_a = K D_a and its drift δM_a.
    // We assemble flattened vec(M_a) and vec(δM_a) so the final contraction is a
    // pair of m·m inner products per (a,b).
    let mut m_rows = Array2::<f64>::zeros((p, m * m)); // vec(M_a)
    let mut dm_rows = Array2::<f64>::zeros((p, m * m)); // vec(δM_a)
    let mut axis = Array1::<f64>::zeros(p);
    for a in 0..p {
        axis.fill(0.0);
        axis[a] = 1.0;
        let hdot_a = match hessian_dir(&axis)? {
            Some(hd) => hd,
            None => return Ok(Array2::zeros((p, p))),
        };
        if hdot_a.nrows() != p || hdot_a.ncols() != p {
            return Err(format!(
                "joint_jeffreys_hphi_directional_derivative: Hdot[e_a] shape {}x{} != {p}x{p}",
                hdot_a.nrows(),
                hdot_a.ncols()
            ));
        }
        let d_a = z_j.t().dot(&hdot_a.dot(&z_j)); // Z_Jᵀ ∂_a H Z_J
        let m_a = h_id_inv.dot(&d_a); // K D_a

        // D_a^δ = Z_Jᵀ H²dot[δ, e_a] Z_J  (second directional derivative).
        let h2dot = match hessian_second_dir(delta, &axis)? {
            Some(h2) => h2,
            // No exact second directional derivative ⇒ drift undefined ⇒ safe zero.
            None => return Ok(Array2::zeros((p, p))),
        };
        if h2dot.nrows() != p || h2dot.ncols() != p {
            return Err(format!(
                "joint_jeffreys_hphi_directional_derivative: H²dot[δ,e_a] shape {}x{} != {p}x{p}",
                h2dot.nrows(),
                h2dot.ncols()
            ));
        }
        let d_a_delta = z_j.t().dot(&h2dot.dot(&z_j)); // Z_Jᵀ ∂_δ∂_a H Z_J

        // δM_a = −K Ḋ M_a + K D_a^δ.
        let dm_a = &h_id_inv.dot(&d_a_delta) - &k_dbar.dot(&m_a);

        let mut col = 0usize;
        for i in 0..m {
            for j in 0..m {
                m_rows[[a, col]] = m_a[[i, j]];
                dm_rows[[a, col]] = dm_a[[i, j]];
                col += 1;
            }
        }
    }

    // D_β H_Φ[δ][a,b] = ½ (⟨vec δM_a, vec M_b⟩ + ⟨vec M_a, vec δM_b⟩). Symmetric.
    let mut out = Array2::<f64>::zeros((p, p));
    for a in 0..p {
        for b in a..p {
            let mut acc = 0.0;
            for col in 0..(m * m) {
                acc += dm_rows[[a, col]] * m_rows[[b, col]] + m_rows[[a, col]] * dm_rows[[b, col]];
            }
            let value = 0.5 * acc;
            out[[a, b]] = value;
            out[[b, a]] = value;
        }
    }
    Ok(out)
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
        // gradient. The SECOND direction is scaled by `ill` so the reduced
        // information is ILL-conditioned (`λ_min/λ_max ≈ 8.6e-10`, below the
        // conditioning gate) — this exercises the active Jeffreys path rather
        // than the gate, while both eigenvalues stay comfortably above the
        // floored ridge so `Φ` and `grad` are the exact log-det quantities.
        // H(beta) = diag(exp(beta0), ill*(1+beta1^2)), Z_J = I.
        let p = 2usize;
        let ill = 1e-9_f64;
        let z = Array2::<f64>::eye(p);
        let h_at = |b: &Array1<f64>| -> Array2<f64> {
            let mut h = Array2::<f64>::zeros((p, p));
            h[[0, 0]] = b[0].exp();
            h[[1, 1]] = ill * (1.0 + b[1] * b[1]);
            h
        };
        // Hdot[d] = d/d eps H(beta + eps d): diag(exp(b0) d0, ill*2 b1 d1).
        let beta: Array1<f64> = array![0.3, -0.4];
        let hdir = |d: &Array1<f64>| -> Result<Option<Array2<f64>>, String> {
            let mut hd = Array2::<f64>::zeros((p, p));
            hd[[0, 0]] = beta[0].exp() * d[0];
            hd[[1, 1]] = ill * 2.0 * beta[1] * d[1];
            Ok(Some(hd))
        };
        let h = h_at(&beta);
        let (phi, grad, hphi) = joint_jeffreys_term(h.view(), z.view(), hdir).unwrap();
        // Phi = 1/2 log(exp(b0) * ill*(1 + b1^2)). The reduced-information
        // eigendecomposition resolves a spectrum spanning ~9 orders of magnitude
        // (λ_max ≈ 1.35, λ_min ≈ 1.16e-9), so the small eigenvalue — and hence Φ
        // — carries the eigensolver's relative round-off (~1e-7 abs on a Φ ≈ -10
        // log-volume). That is expected on a deliberately ill-conditioned design
        // exercising the active (un-gated) path; the load-bearing correctness
        // check is the gradient FD below, which is insensitive to the constant
        // `ill` scale.
        let expected_phi = 0.5 * (beta[0].exp() * ill * (1.0 + beta[1] * beta[1])).ln();
        assert!(
            (phi - expected_phi).abs() < 1e-6,
            "phi {phi} vs {expected_phi}"
        );
        // Finite-difference the gradient. Note ∂/∂β of log|H| is scale-free in
        // the constant `ill` factor (it differentiates the log), so the gradient
        // matches the un-scaled form exactly.
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
    fn conditioning_gate_skips_well_conditioned_information() {
        // A WELL-conditioned reduced information (`λ_min/λ_max = 0.5`, far above
        // the gate) must skip the Jeffreys term entirely: zero value, gradient
        // and curvature, so an easy fit pays no cost. The directional-derivative
        // closure here is deliberately NONZERO; the gate must short-circuit
        // before it would otherwise produce a nonzero gradient.
        let p = 2usize;
        let z = Array2::<f64>::eye(p);
        let mut h = Array2::<f64>::zeros((p, p));
        h[[0, 0]] = 2.0;
        h[[1, 1]] = 1.0; // ratio 0.5 ≫ 1e-8 ⇒ gated
        let hdir = |d: &Array1<f64>| -> Result<Option<Array2<f64>>, String> {
            // Nonzero derivative; would yield a nonzero gradient if not gated.
            let mut hd = Array2::<f64>::zeros((p, p));
            hd[[0, 0]] = 3.0 * d[0];
            hd[[1, 1]] = 5.0 * d[1];
            Ok(Some(hd))
        };
        let (phi, grad, hphi) = joint_jeffreys_term(h.view(), z.view(), hdir).unwrap();
        assert_eq!(phi, 0.0, "well-conditioned ⇒ no Jeffreys value");
        assert!(grad.iter().all(|v| *v == 0.0), "well-conditioned ⇒ zero grad");
        assert!(hphi.iter().all(|v| *v == 0.0), "well-conditioned ⇒ zero curvature");
    }

    #[test]
    fn conditioning_gate_fires_only_below_threshold() {
        // Bracket the gate: a ratio just ABOVE the threshold is skipped (zero
        // term), a ratio just BELOW it falls through to the active path and
        // produces a NONZERO value/curvature. This pins the "no cost on easy
        // fits, full term on hard ones" boundary.
        let p = 2usize;
        let z = Array2::<f64>::eye(p);
        let hdir = |d: &Array1<f64>| -> Result<Option<Array2<f64>>, String> {
            let mut hd = Array2::<f64>::zeros((p, p));
            hd[[0, 0]] = d[0];
            hd[[1, 1]] = d[1];
            Ok(Some(hd))
        };
        // lambda_max = 1.0; choose lambda_min on either side of CONDITIONING_GATE_RELATIVE.
        let mk = |lmin: f64| {
            let mut h = Array2::<f64>::zeros((p, p));
            h[[0, 0]] = 1.0;
            h[[1, 1]] = lmin;
            h
        };
        // Above threshold (well-conditioned) ⇒ gated.
        let above = mk(CONDITIONING_GATE_RELATIVE * 10.0);
        let (phi_a, grad_a, _) = joint_jeffreys_term(above.view(), z.view(), hdir).unwrap();
        assert_eq!(phi_a, 0.0);
        assert!(grad_a.iter().all(|v| *v == 0.0));
        // Below threshold (ill-conditioned) ⇒ active, nonzero contribution.
        let below = mk(CONDITIONING_GATE_RELATIVE * 0.1);
        let (phi_b, _grad_b, hphi_b) = joint_jeffreys_term(below.view(), z.view(), hdir).unwrap();
        assert!(phi_b != 0.0, "below-gate must produce a nonzero Jeffreys value");
        assert!(
            hphi_b.iter().any(|v| v.abs() > 0.0),
            "below-gate must produce nonzero bounding curvature"
        );
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
