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

use crate::linalg::faer_ndarray::{FaerEigh, fast_abt};
use crate::linalg::lanczos::{SymmetricLanczosOptions, symmetric_lanczos_eigenpairs};
use faer::Side;
use ndarray::{Array1, Array2, ArrayView2};

#[inline]
fn norm2_slice(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

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
/// reports under-identification (see [`conditioning_gate_weight`]).
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

/// Upper knot of the SMOOTH absolute conditioning ramp. Below
/// `CONDITIONING_GATE_ABSOLUTE` (one observation-equivalent of curvature) the
/// direction is near-separating and the Jeffreys weight is `1` (full term);
/// above this value it is comfortably identified and the weight is `0` (skip).
/// Between the two knots the weight is a C¹ cubic-smoothstep blend, so the outer
/// LAML objective `Φ(ρ)` is CONTINUOUS as `β̂(ρ)` carries the spectrum across the
/// boundary. A BINARY gate makes `Φ(ρ)` jump there, which the gradient-based
/// outer smoother (BFGS) cannot optimize across — the root cause of the #787
/// "outer smoothing did not converge" regression. `16` ≈ a handful of
/// observation-equivalents: comfortably past "identified by the data".
const CONDITIONING_GATE_ABSOLUTE_CLEAR: f64 = 16.0;

/// Upper knot of the SMOOTH relative conditioning ramp (ramped in `log10` space
/// since conditioning ratios span orders of magnitude). At
/// `λ_min/λ_max ≥ CONDITIONING_GATE_RELATIVE_CLEAR` the relative sub-weight is
/// `0`; at `≤ CONDITIONING_GATE_RELATIVE` it is `1`; smooth in between.
const CONDITIONING_GATE_RELATIVE_CLEAR: f64 = 1e-6;

/// Shared conditioning-gate predicate for the Jeffreys term, evaluated from the
/// reduced-information spectrum (`λ_min`, `λ_max`). Returns `true` when the term
/// should be SKIPPED (zero contribution) because the reduced information is
/// well-conditioned — both relatively (`λ_min/λ_max ≥ CONDITIONING_GATE_RELATIVE`)
/// AND absolutely (`λ_min ≥ CONDITIONING_GATE_ABSOLUTE`). If EITHER test reports
/// under-identification the gate does NOT skip and the floored log-det term
/// fires. Centralised so every call site (objective value, gradient/curvature,
/// and the `H_Φ` directional derivative) uses byte-identical gating — any
/// divergence would reintroduce the value/derivative mismatch the term removes.
///
/// Returns a SMOOTH (C¹) weight in `[0, 1]`: `1` when the reduced information is
/// under-identified (the term is fully active and supplies the `O(1)`-bounding
/// curvature), `0` when comfortably well-conditioned (the term is skipped,
/// preserving the clean-fit guarantee), and a cubic-smoothstep blend across each
/// transition band so `Φ(ρ)`, `∇Φ` and `H_Φ` are continuous in the smoothing
/// parameters. The term fires when EITHER criterion reports under-identification,
/// so the weight is the MAX of the absolute and relative sub-weights.
#[inline]
fn conditioning_gate_weight(lambda_min: f64, lambda_max: f64) -> f64 {
    if lambda_max <= 0.0 {
        // Degenerate / non-positive spectrum: not well-conditioned, fully active.
        return 1.0;
    }
    if !lambda_min.is_finite() {
        return 1.0;
    }
    // `ramp_down(x, under, clear)`: the still-active weight, `1` for `x ≤ under`,
    // `0` for `x ≥ clear`, C¹ cubic smoothstep `1 − (3t² − 2t³)` between.
    #[inline]
    fn ramp_down(x: f64, under: f64, clear: f64) -> f64 {
        if x <= under {
            return 1.0;
        }
        if x >= clear {
            return 0.0;
        }
        let t = (x - under) / (clear - under);
        1.0 - t * t * (3.0 - 2.0 * t)
    }
    let w_abs = ramp_down(
        lambda_min,
        CONDITIONING_GATE_ABSOLUTE,
        CONDITIONING_GATE_ABSOLUTE_CLEAR,
    );
    let ratio = (lambda_min / lambda_max).max(f64::MIN_POSITIVE);
    let w_rel = ramp_down(
        ratio.log10(),
        CONDITIONING_GATE_RELATIVE.log10(),
        CONDITIONING_GATE_RELATIVE_CLEAR.log10(),
    );
    w_abs.max(w_rel)
}

/// Partial derivatives `(∂G/∂λ_min, ∂G/∂λ_max)` of the conditioning gate weight
/// `G = max(w_abs, w_rel)` (see [`conditioning_gate_weight`]).
///
/// The gate scales the Jeffreys curvature the LAML value folds into
/// `½ log|H + S_λ + G·H_Φ_raw|`; because `λ_min, λ_max` move with β through the
/// inner mode response, the gate's own mode-response variation is part of the
/// EXACT outer hypergradient. Dropping it (treating `G` as locally constant in the
/// drift) desyncs the analytic outer gradient from its own value precisely when
/// the gate sits in its smooth transition band — the residual tension-axis
/// mismatch in gam#854, even when no eigenvalue is floored. Returns `(0, 0)` on the
/// saturated / degenerate branches where `G` is locally constant (so the outer
/// drift is byte-unchanged on every fully-active or well-conditioned fit).
fn conditioning_gate_weight_grad(lambda_min: f64, lambda_max: f64) -> (f64, f64) {
    if lambda_max <= 0.0 || !lambda_min.is_finite() {
        // Matches `conditioning_gate_weight`'s constant-`1.0` early returns.
        return (0.0, 0.0);
    }
    // `ramp_down`'s value and derivative: `d/dx [1 − (3t² − 2t³)] = −6 t (1−t) / (clear − under)`
    // on the open band (`under < x < clear`), `0` at/outside both knots (C¹).
    #[inline]
    fn ramp_down_value_and_deriv(x: f64, under: f64, clear: f64) -> (f64, f64) {
        if x <= under {
            return (1.0, 0.0);
        }
        if x >= clear {
            return (0.0, 0.0);
        }
        let span = clear - under;
        let t = (x - under) / span;
        let value = 1.0 - t * t * (3.0 - 2.0 * t);
        let deriv = -6.0 * t * (1.0 - t) / span;
        (value, deriv)
    }
    let (w_abs, dw_abs_dlmin) = ramp_down_value_and_deriv(
        lambda_min,
        CONDITIONING_GATE_ABSOLUTE,
        CONDITIONING_GATE_ABSOLUTE_CLEAR,
    );
    let ratio = (lambda_min / lambda_max).max(f64::MIN_POSITIVE);
    let (w_rel, dw_rel_dlogratio) = ramp_down_value_and_deriv(
        ratio.log10(),
        CONDITIONING_GATE_RELATIVE.log10(),
        CONDITIONING_GATE_RELATIVE_CLEAR.log10(),
    );
    // `G = w_abs.max(w_rel)`: only the active branch varies the max. A tie is a
    // measure-zero kink the smooth band stays away from; resolve it to `w_abs`
    // (consistent, and the dominant branch in the small-`n` absolute regime).
    if w_abs >= w_rel {
        (dw_abs_dlmin, 0.0)
    } else {
        // `∂ log₁₀(λ_min/λ_max)/∂λ_min = 1/(λ_min ln10)`,
        // `∂ log₁₀(λ_min/λ_max)/∂λ_max = −1/(λ_max ln10)`.
        let ln10 = std::f64::consts::LN_10;
        (
            dw_rel_dlogratio / (lambda_min * ln10),
            -dw_rel_dlogratio / (lambda_max * ln10),
        )
    }
}

/// Below this joint dimension the dense reduced eigendecomposition in
/// [`joint_jeffreys_term`] is itself cheap (`O(p³)` with `p` in the tens — e.g.
/// the BMS-probit `p≈51` fit), so the matrix-free pre-check below would only add
/// `O(p·k)` matvecs for no asymptotic win and a (tiny) chance of a conservative
/// false-fall-through. We therefore run the exact path directly for small joint
/// systems and reserve the cheap pre-check for the wide systems whose `O(p³)`
/// eigendecomposition (and the dense `H_id` it needs) is the cost we want to
/// avoid on a well-conditioned fit. This threshold matches the matrix-free joint
/// path's `JOINT_MATRIX_FREE_MIN_DIM_AT_LARGE_N` so the pre-check exists exactly
/// where the dense formation is the regression.
pub const CHEAP_CONDITIONING_PRECHECK_MIN_DIM: usize = 128;

/// Safety factor by which the CONSERVATIVE spectral bounds must clear each
/// conditioning gate before the cheap pre-check is allowed to declare the term
/// skippable. The Lanczos bounds below are already one-sided-conservative
/// (`λ_min` is bounded from BELOW, `λ_max` from ABOVE — see
/// [`cheap_conditioning_bounds`]), so a clearance factor of `1` would already be
/// correct; we demand an extra `8×` margin purely as defense-in-depth against
/// round-off in the bound arithmetic and against a `k`-step subspace that has not
/// yet resolved the extreme Ritz pair. The cost of being wrong is asymmetric:
/// a false SKIP omits the curvature that prevents non-convergence (unacceptable),
/// whereas a false FALL-THROUGH merely pays the exact dense path we would have
/// paid anyway. The margin is therefore set firmly on the safe side.
const CHEAP_PRECHECK_SAFETY_MARGIN: f64 = 8.0;

/// Number of Lanczos steps for the cheap conditioning pre-check. A handful of
/// steps with full reorthogonalization resolves the extreme Ritz pair to far
/// better than the `8×` safety margin on the realistic spectra here (a
/// well-conditioned joint information whose `λ_min` we only need to certify is
/// `≳ 8` and whose ratio we only need to certify is `≳ 8e-8`), while keeping the
/// pre-check at `O(p·k)` matvecs — negligible against the `O(p³)`/dense-`H_id`
/// path it guards. Capped at `p` for tiny systems (which the size gate already
/// routes to the exact path anyway).
const CHEAP_PRECHECK_LANCZOS_STEPS: usize = 12;

/// Relative residual below which an extreme Ritz pair counts as "converged" and
/// its residual-augmented eigenvalue bound may be trusted. Measured against the
/// spectral scale `θ_max` (floored at 1): once `‖H y − θ y‖ ≤ 1e-3·scale`, the
/// extreme eigenvalue is resolved to three digits, far tighter than the `8×`
/// safety margin the skip decision then applies. An unresolved residual returns
/// `None` so the caller falls through to the exact dense path.
const CHEAP_PRECHECK_RITZ_REL_TOL: f64 = 1e-3;

/// Conservative extreme-eigenvalue bounds `(λ_min_lower, λ_max_upper)` for the
/// reduced information `H_id = Z_Jᵀ H Z_J` on the FULL span (`Z_J = I`, so
/// `H_id = H`), computed MATRIX-FREE from a Hessian-vector product `hv` and the
/// dimension `p`, WITHOUT ever forming the dense `H_id` or its eigendecomposition.
///
/// METHOD. `k`-step Lanczos with FULL reorthogonalization builds a symmetric
/// tridiagonal `T_k` from a dense aperiodic start vector. With full reorth the
/// factorization `H Q_k = Q_k T_k + β_k q_{k+1} e_kᵀ` holds exactly, so each Ritz
/// pair `(θ_i, y_i)` has the SHARP residual `‖H(Q_k y_i) − θ_i (Q_k y_i)‖ =
/// β_k·|e_kᵀ y_i| =: res_i` (Saad, *Numerical Methods for Large Eigenvalue
/// Problems*, §6; Parlett, *The Symmetric Eigenvalue Problem*). The residual
/// eigenvalue bound then guarantees a true eigenvalue of `H` within `res_i` of
/// `θ_i`.
///
/// CONSERVATIVE one-sided bounds. The EXTREME Ritz pairs converge FIRST under
/// Lanczos (Kaniel–Paige); the TRUST GATE below requires the extreme residuals to
/// be small relative to the spectral scale `θ_max` before trusting the estimate,
/// which (with full reorth, so spurious/ghost eigenvalues cannot arise) means the
/// Krylov space has resolved BOTH ends of the spectrum — hence `θ_min`/`θ_max`
/// ARE `λ_min`/`λ_max` to within `res_min`/`res_max`. We therefore return:
///   * `λ_min(H) ≥ θ_min − res_min`  (LOWER bound on the smallest eigenvalue),
///   * `λ_max(H) ≤ θ_max + res_max`  (UPPER bound on the largest eigenvalue).
/// These bias the conditioning estimate toward "looks WORSE-conditioned than it
/// is" — the direction that makes a SKIP decision safe: if even these pessimistic
/// bounds clear the gate (with the caller's extra `8×` margin), the true spectrum
/// clears it by more.
///
/// Returns `None` — forcing the caller to fall through to the EXACT dense path —
/// whenever the estimate cannot be trusted: a non-finite/degenerate Krylov space
/// (zero-start collapse) OR an UNCONVERGED extreme Ritz pair (`res_min`/`res_max`
/// not small). The latter is the critical safety valve: if the cheap iteration
/// has not resolved the bottom of the spectrum it NEVER authorises a skip, so a
/// hidden small eigenvalue cannot be missed — the term is then formed exactly.
fn cheap_conditioning_bounds<HvFn>(mut hv: HvFn, p: usize) -> Result<Option<(f64, f64)>, String>
where
    HvFn: FnMut(&Array1<f64>) -> Result<Array1<f64>, String>,
{
    if p == 0 {
        return Ok(None);
    }
    let steps = CHEAP_PRECHECK_LANCZOS_STEPS.min(p);
    // Deterministic dense start vector with a NON-UNIFORM, aperiodic pattern. A
    // plain all-ones vector is orthogonal to any eigenvector whose entries sum to
    // zero — a symmetry an exactly-balanced Fisher information could exhibit,
    // which would hide that eigenvalue from the Krylov space. Seeding each entry
    // from an irrational-rotation sequence (`frac(i·φ) − ½`, φ the golden ratio)
    // gives a deterministic, reproducible (no RNG in the hot path) start that has
    // a non-zero component along every eigenvector for any realistic operator, so
    // the extreme Ritz pairs are resolved and the trust gate below is meaningful.
    let mut q0 = vec![0.0_f64; p];
    let golden = 0.618_033_988_749_894_8_f64; // frac(golden ratio)
    for (i, qi) in q0.iter_mut().enumerate() {
        let frac = ((i as f64 + 1.0) * golden).fract();
        *qi = frac - 0.5;
    }
    let q_norm = norm2_slice(&q0);
    if !(q_norm.is_finite() && q_norm > 0.0) {
        return Ok(None);
    }
    // Single shared matrix-free Lanczos primitive with FULL reorthogonalization,
    // so `H Q_k = Q_k T_k + β_k q_{k+1} e_kᵀ` holds exactly and the per-Ritz-pair
    // residual `β_k·|e_kᵀ y_i|` (read from `residual_norm` and the last row of the
    // Ritz vectors below) is a SHARP eigenvalue bound. A matvec/HVP failure or a
    // non-finite iterate surfaces as `None` (caller falls through to the exact
    // dense path), preserving the conservative "never authorise a skip on an
    // unconverged check" contract.
    let mut hv_failed: Option<String> = None;
    let eigen = match symmetric_lanczos_eigenpairs(
        p,
        &q0,
        SymmetricLanczosOptions {
            max_steps: steps,
            // Lucky-breakdown floor: stop (β_k reported as 0 ⇒ EXACT Ritz
            // spectrum, tight bounds) once the next Lanczos vector is at the
            // machine-precision noise floor of a unit-norm start, before any
            // divide-by-≈0 can pollute the basis. Conservative: a non-breakdown
            // small residual still propagates as a faithful (small) bound below.
            residual_tol: f64::EPSILON,
            local_reorthogonalize: false,
            full_reorthogonalize: true,
        },
        |q, out| {
            let qv = Array1::from_vec(q.to_vec());
            let w = match hv(&qv) {
                Ok(w) => w,
                Err(e) => {
                    hv_failed = Some(e);
                    return Err("cheap_conditioning_bounds: HVP failed".to_string());
                }
            };
            if w.len() != p || w.iter().any(|x| !x.is_finite()) {
                return Err(
                    "cheap_conditioning_bounds: HVP produced non-finite/ill-sized output"
                        .to_string(),
                );
            }
            out.copy_from_slice(w.as_slice().ok_or_else(|| {
                "cheap_conditioning_bounds: HVP output not contiguous".to_string()
            })?);
            Ok(())
        },
    ) {
        Ok(eigen) => eigen,
        Err(_) => {
            // Propagate a genuine HVP error to the caller; treat a Lanczos
            // numerical degeneracy (zero-start collapse, non-finite iterate) as
            // an untrusted estimate and fall through to the exact path.
            if let Some(e) = hv_failed {
                return Err(e);
            }
            return Ok(None);
        }
    };
    let ritz = eigen.eigenvalues;
    let ritz_vecs = eigen.eigenvectors;
    let last_residual_norm = eigen.residual_norm;
    let k = ritz.len();
    if k == 0 {
        return Ok(None);
    }
    // Index of the smallest / largest Ritz value and the eigenvector last
    // components, which give the SHARP per-pair residual `β_k·|e_kᵀ y_i|` ≤ β_k.
    let mut idx_min = 0usize;
    let mut idx_max = 0usize;
    for i in 1..k {
        if ritz[i] < ritz[idx_min] {
            idx_min = i;
        }
        if ritz[i] > ritz[idx_max] {
            idx_max = i;
        }
    }
    let theta_min = ritz[idx_min];
    let theta_max = ritz[idx_max];
    if !theta_min.is_finite() || !theta_max.is_finite() {
        return Ok(None);
    }
    // Sharp Ritz residuals: `‖H y_i − θ_i y_i‖ = β_k·|e_kᵀ y_i|` (β_k =
    // last_residual_norm, the norm of the unnormalised next Lanczos vector). The
    // last row of `ritz_vecs` holds `e_kᵀ y_i` for every Ritz pair.
    let last_row = k - 1;
    let res_min = last_residual_norm * ritz_vecs[[last_row, idx_min]].abs();
    let res_max = last_residual_norm * ritz_vecs[[last_row, idx_max]].abs();
    // TRUST GATE (Krylov near-invariance). With FULL reorthogonalization the
    // Lanczos factorization satisfies `H Q_k = Q_k T_k + β_k q_{k+1} e_kᵀ` exactly,
    // and the EXTREME Ritz pairs converge FIRST (Kaniel–Paige). When the extreme
    // residuals are small relative to the spectral scale θ_max, the Krylov space
    // has resolved the spectrum's ENDS, so `θ_min`/`θ_max` are faithful estimates
    // of `λ_min`/`λ_max` and the residual-augmented one-sided bounds below are
    // sound. When a residual is NOT small (the bottom of the spectrum is
    // unresolved — e.g. a start vector poorly aligned with the extreme
    // eigenspace), we return `None` so the caller CONSERVATIVELY falls through to
    // the exact dense path rather than trusting an unconverged estimate. This is
    // the safety valve: an unconverged cheap check never authorises a skip.
    let scale = theta_max.abs().max(1.0);
    let converged_tol = CHEAP_PRECHECK_RITZ_REL_TOL * scale;
    if res_min > converged_tol || res_max > converged_tol {
        return Ok(None);
    }
    // Conservative one-sided bounds. Subtract/ add the SHARP residual: there is a
    // true eigenvalue of `H` within `res_min` of `θ_min` and within `res_max` of
    // `θ_max` (residual eigenvalue bound), and with the extreme pairs converged
    // these ARE the extreme eigenvalues, so:
    //   λ_min(H) ≥ θ_min − res_min,   λ_max(H) ≤ θ_max + res_max.
    let lambda_min_lb = theta_min - res_min;
    let lambda_max_ub = theta_max + res_max;
    Ok(Some((lambda_min_lb, lambda_max_ub)))
}

/// Cheap MATRIX-FREE pre-check answering "is the Jeffreys term provably
/// SKIPPABLE here?" using only Hessian-vector products against the full-span
/// reduced information `H_id = H` — WITHOUT forming the dense `H_id` or running
/// the `O(p³)` eigendecomposition in [`joint_jeffreys_term`].
///
/// Returns `true` ONLY when the CONSERVATIVE spectral bounds from
/// [`cheap_conditioning_bounds`] clear BOTH conditioning gates with the
/// [`CHEAP_PRECHECK_SAFETY_MARGIN`] safety factor, i.e. the fit is so clearly
/// well-conditioned that the exact gate is certain to skip too. In that case the
/// caller may return the EXACT-ZERO term (byte-identical to the gated-off dense
/// path) without forming anything dense. Returns `false` (the conservative
/// default) whenever the cheap bounds are unresolved, non-positive, or merely
/// close to the gate — the caller then falls through to the exact dense
/// formation + gate, so the term is still computed exactly wherever it might be
/// needed.
///
/// CORRECTNESS. `cheap_conditioning_bounds` returns `λ_min_lb ≤ λ_min(H)` and
/// `λ_max_ub ≥ λ_max(H)`. Hence
///   `λ_min(H) ≥ λ_min_lb` and `λ_min(H)/λ_max(H) ≥ λ_min_lb/λ_max_ub`,
/// so when `λ_min_lb ≥ MARGIN·CONDITIONING_GATE_ABSOLUTE` and
/// `λ_min_lb/λ_max_ub ≥ MARGIN·CONDITIONING_GATE_RELATIVE` the TRUE spectrum
/// satisfies the exact `conditioning_gate_skips` predicate by at least `MARGIN×`
/// — the exact path would skip, so skipping cheaply is byte-identical. The
/// converse cases never skip, preserving exactness where the term bites.
pub fn jeffreys_term_skippable_via_matvec<HvFn>(hv: HvFn, p: usize) -> Result<bool, String>
where
    HvFn: FnMut(&Array1<f64>) -> Result<Array1<f64>, String>,
{
    if p < CHEAP_CONDITIONING_PRECHECK_MIN_DIM {
        // Small systems: the exact dense eigh is already cheap; do not pre-check.
        return Ok(false);
    }
    let (lambda_min_lb, lambda_max_ub) = match cheap_conditioning_bounds(hv, p)? {
        Some(bounds) => bounds,
        None => return Ok(false),
    };
    if !(lambda_min_lb.is_finite() && lambda_max_ub.is_finite()) {
        return Ok(false);
    }
    // The conservative lower bound must itself be positive (and large) — a
    // non-positive `λ_min_lb` cannot certify SPD/well-conditioned, so never skip.
    if lambda_min_lb <= 0.0 || lambda_max_ub <= 0.0 {
        return Ok(false);
    }
    // A full skip now requires the SMOOTH weight to be exactly 0, i.e. the
    // spectrum must clear the UPPER (`*_CLEAR`) knots of both ramps — not merely
    // the lower (firing) thresholds. The conservative bounds must clear those
    // upper knots by the safety margin.
    let absolute_clears =
        lambda_min_lb >= CHEAP_PRECHECK_SAFETY_MARGIN * CONDITIONING_GATE_ABSOLUTE_CLEAR;
    let relative_clears = lambda_min_lb / lambda_max_ub
        >= CHEAP_PRECHECK_SAFETY_MARGIN * CONDITIONING_GATE_RELATIVE_CLEAR;
    Ok(absolute_clears && relative_clears)
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
    let (evals, evecs) = h_id_sym.eigh(Side::Lower).map_err(|e| {
        format!("joint_jeffreys_term: reduced-information eigendecomposition failed: {e}")
    })?;
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
    let gate_weight = {
        let lambda_min = evals.iter().cloned().fold(f64::INFINITY, f64::min);
        conditioning_gate_weight(lambda_min, lambda_max)
    };
    if gate_weight == 0.0 {
        return Ok((0.0, Array1::zeros(p), Array2::zeros((p, p))));
    }
    // Absolute floor relative to the dominant identified curvature: negligible on
    // identified directions (O(n)), positive on separating ones.
    let floor = (REDUCED_INFO_RELATIVE_FLOOR * lambda_max).max(REDUCED_INFO_ABSOLUTE_FLOOR);
    // FLOOR β-DEPENDENCE (root cause of gam#826 / the below-floor value↔gradient
    // mismatch). The regularization floor is `max(REL·λ_max, ABS)`; in the active
    // RELATIVE regime it scales with `λ_max(β)`, which is itself a function of β.
    // The below-floor antiderivative is `g(λ; floor) = λ/floor + ln(floor) − 1`, so
    // for a below-floor eigenvalue the TOTAL derivative `dΦ/dβ_k` carries, beyond
    // the eigenvalue term `(1/floor) ∂λ/∂β_k`, the floor-response term
    //   `∂g/∂floor · ∂floor/∂β_k = (1/floor − λ/floor²) · ∂floor/∂β_k`.
    // The earlier gradient differentiated only the eigenvalue and treated the floor
    // as constant, so on any fit with an eigenvalue parked below the floor the
    // analytic gradient did not equal `d/dβ Φ` — the inner joint-Newton KKT residual
    // could not reach zero (a contributor to the coupled location-scale
    // non-convergence). We restore the exact pair below by adding the floor-response
    // term, using `∂λ_max/∂β_k = v_maxᵀ D_k v_max` (first-order eigenvalue
    // perturbation; `D_k = Z_Jᵀ Hdot[e_k] Z_J` is already formed in the gradient
    // loop). When the floor is in the ABSOLUTE regime (`REL·λ_max ≤ ABS`, including
    // a non-positive `λ_max`) the floor is β-independent, so the term is exactly
    // zero and nothing is added — preserving the PSD-fit and indefinite fast paths
    // byte-for-byte. The eigenvalue-perturbation formula is exact only at a SIMPLE
    // dominant eigenvalue; a tied `λ_max` is a measure-zero kink the smooth-gate
    // band keeps away from (and the floor-response term is itself O(λ/floor)-tiny
    // there), so no special-casing is warranted.
    let floor_in_relative_regime =
        lambda_max > 0.0 && REDUCED_INFO_RELATIVE_FLOOR * lambda_max >= REDUCED_INFO_ABSOLUTE_FLOOR;
    // Eigenvector of the dominant eigenvalue `λ_max` (the one the relative floor
    // tracks), needed for `∂λ_max/∂β_k = v_maxᵀ D_k v_max`. Only consulted in the
    // relative regime.
    let lambda_max_evec: Option<Array1<f64>> = if floor_in_relative_regime {
        let mut idx_max = 0usize;
        for i in 1..m {
            if evals[i] > evals[idx_max] {
                idx_max = i;
            }
        }
        Some(evecs.column(idx_max).to_owned())
    } else {
        None
    };
    // `Σ_{i: |λ_i| < floor} ∂g(λ_i; floor)/∂floor = Σ (1/floor − λ_i/floor²)`, the
    // sensitivity of the below-floor value contributions to the floor. Zero when no
    // eigenvalue sits below the floor (so the floor-response term vanishes on every
    // well-conditioned / indefinite fit), making this fix inert outside the
    // genuinely below-floor regime it targets.
    let mut floor_value_sensitivity = 0.0_f64;
    let mut phi = 0.0_f64;
    // h_id_inv = V diag(1/max(λ,floor)) Vᵀ  (floored symmetric pseudo-inverse).
    let mut inv_diag = Array1::<f64>::zeros(m);
    for (i, &lam) in evals.iter().enumerate() {
        // VALUE / GRADIENT CONSISTENCY (was a stall — gam#787/#785) AND SIGN
        // HANDLING ON INDEFINITE REDUCED INFO (gam#814). The gradient below is
        // `½ tr(H_id⁻¹ D_k) = ½ Σ_i inv_diag_i ∂λ_i/∂β`, so for the value/gradient
        // pair to be consistent (∇Φ = d/dβ Φ) the value `Φ = ½ Σ_i g(λ_i)` must use
        // the antiderivative `g` of `λ ↦ inv_diag(λ)`.
        //
        // We floor on the MAGNITUDE `|λ|` and keep the SIGN of the inverse:
        //   • |λ| ≥ floor: g(λ) = ln|λ|               (g'(λ) = 1/λ),
        //   • |λ| < floor: g(λ) = λ/floor + ln(floor) − 1  (g'(λ) = 1/floor),
        // the #787 linear continuation, C¹ at λ = +floor (g(floor) = ln(floor),
        // g'(floor⁻) = g'(floor⁺) = 1/floor).
        //
        // WHY THE FLOOR IS ON `|λ|`, NOT `λ` (gam#814). `H_id = Z_Jᵀ H Z_J` is the
        // reduced OBSERVED information; for a non-canonical link (e.g. the
        // marginal-slope survival family) it need NOT be PSD away from the mode, so
        // at off-mode trial points it carries MODERATE NEGATIVE eigenvalues
        // (measured: λ ≈ −0.05 … −0.36, |λ| ≫ floor = 1e-10·λ_max). The previous
        // `1/max(λ, floor)` flooring keyed on the SIGNED eigenvalue, so EVERY
        // negative direction — however moderate — was treated as near-zero and
        // pinned to `1/floor ≈ 1.7e6`: a phantom Firth score (∇Φ ≈ 1.4e6,
        // H_Φ ≈ 1e12) that no Newton step could satisfy, so the inner joint-Newton
        // crawled its whole cycle budget and the outer LAML never received a
        // stationary mode (gam#814 survival clustered-PC marginal-slope timeout).
        // Flooring on `|λ|` with the SIGNED true inverse `1/λ` gives the moderate
        // negatives their genuine `1/λ ≈ −20 … −2.8` instead of `+1.7e6`, so the
        // Jeffreys log-volume uses `½ ln|λ|` (the determinant magnitude, the
        // PSD-realisation of `½ log det I` the Jeffreys prior intends) and the term
        // self-limits rather than exploding. On a genuinely PSD fit every λ ≥ floor
        // and this is BYTE-IDENTICAL to the prior `1/λ`, `½ ln λ`; the #787
        // near-separation branch (`0 ≤ λ < floor`) is UNCHANGED, preserving the
        // 1/floor separation bound. There is a harmless C⁰ kink in inv_diag at
        // λ = ±floor (it jumps `+1/floor ↔ −1/floor`), but no eigenvalue sits near
        // ∓floor in practice and the value `½ ln|λ|` stays continuous there.
        let lam_mag = lam.abs();
        if lam_mag >= floor {
            phi += 0.5 * lam_mag.ln();
            inv_diag[i] = 1.0 / lam;
        } else {
            phi += 0.5 * (lam / floor + floor.ln() - 1.0);
            inv_diag[i] = 1.0 / floor;
            // ∂g(λ; floor)/∂floor = 1/floor − λ/floor², accumulated so the gradient
            // below can add the floor-response term `½ · this · ∂floor/∂β_k`.
            floor_value_sensitivity += 1.0 / floor - lam / (floor * floor);
        }
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
                return Ok((gate_weight * phi, Array1::zeros(p), Array2::zeros((p, p))));
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
        // grad[k] = 1/2 tr(M_k) (the eigenvalue term `½ Σ_i inv_diag_i ∂λ_i/∂β_k`).
        let mut trace = 0.0;
        for i in 0..m {
            trace += m_k[[i, i]];
        }
        grad[k] = 0.5 * trace;
        // FLOOR-RESPONSE term (see the `floor` block above). For below-floor
        // eigenvalues the floor moves with `λ_max(β)`, so `dΦ/dβ_k` carries
        // `½ · floor_value_sensitivity · ∂floor/∂β_k`, with
        // `∂floor/∂β_k = REL · ∂λ_max/∂β_k = REL · v_maxᵀ D_k v_max`. This is the
        // exact antiderivative partner of the below-floor value branch; it is
        // identically zero (and skipped) whenever no eigenvalue is below the floor
        // or the floor is in the β-independent absolute regime, so the well-
        // conditioned, indefinite, and above-floor paths are unchanged.
        if let Some(v_max) = lambda_max_evec.as_ref() {
            if floor_value_sensitivity != 0.0 {
                let dvmax = d_k.dot(v_max);
                let dlambda_max = v_max.dot(&dvmax); // v_maxᵀ D_k v_max
                let dfloor = REDUCED_INFO_RELATIVE_FLOOR * dlambda_max;
                grad[k] += 0.5 * floor_value_sensitivity * dfloor;
            }
        }
        // Store vec(M_k) for the Gauss-Newton surrogate.
        let mut col = 0usize;
        for i in 0..m {
            for j in 0..m {
                sensitivity[[k, col]] = m_k[[i, j]];
                col += 1;
            }
        }
    }
    // Gauss-Newton curvature surrogate: H_Phi = ½ S Sᵀ over the reduced
    // sensitivities `S = sensitivity` (p × m²), i.e. H_Phi[a,b] = ½ <vec(M_a),
    // vec(M_b)>. PSD by construction, vanishes on directions the data already
    // identifies (M_k = 0 there), and grows as the reduced curvature shrinks
    // along a separating direction — the automatic O(1)-bounding Firth curvature.
    //
    // PERF (gam#729/#826/#808): assemble it as one BLAS-3 GEMM `S·Sᵀ` instead of
    // the p²·m² scalar triple loop. For a K-block coupled family (Dirichlet/
    // multinomial) the joint width p and reduced dimension m make the triple loop
    // the dominant per-inner-cycle cost (it is rebuilt every inner Newton cycle
    // and every outer continuation eval); routing through faer's matmul makes it
    // cache-blocked and parallel, the same arithmetic with no accuracy change.
    let mut hphi = fast_abt(&sensitivity, &sensitivity);
    hphi.mapv_inplace(|v| 0.5 * v);
    // Scale the (value, gradient, curvature) triple by the smooth gate weight.
    // `gate_weight == 1` in the fully-active (under-identified) regime, so this is
    // identity there (byte-identical to the binary-gate term); it only tapers the
    // term to 0 across the transition band, making Φ/∇Φ/H_Φ continuous in ρ.
    Ok((gate_weight * phi, grad * gate_weight, hphi * gate_weight))
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
    if delta.len() != p {
        return Err(format!(
            "joint_jeffreys_hphi_directional_derivative: delta has {} entries, expected {p}",
            delta.len()
        ));
    }
    // The mode-response perturbation acts on `H_joint` through `Hdot[δ] = D_β H[δ]`
    // and on each axis derivative `D_a` through `H²dot[δ, e_a] = D²_β H[δ, e_a]`.
    let pert_h = match hessian_dir(delta)? {
        Some(hd) => hd,
        // No exact first directional derivative ⇒ drift undefined ⇒ safe zero.
        None => return Ok(Array2::zeros((p, p))),
    };
    if pert_h.nrows() != p || pert_h.ncols() != p {
        return Err(format!(
            "joint_jeffreys_hphi_directional_derivative: Hdot[δ] shape {}x{} != {p}x{p}",
            pert_h.nrows(),
            pert_h.ncols()
        ));
    }
    joint_jeffreys_hphi_perturbation_derivative(
        h_joint,
        z_j,
        |axis| hessian_dir(axis),
        &pert_h,
        |axis| hessian_second_dir(delta, axis),
    )
}

/// Explicit (β-frozen) derivative `∂_ρ H_Φ|_β` of the gated joint-Jeffreys
/// curvature along an OUTER hyperparameter `ρ` (e.g. a log-penalty `log λ_m` or a
/// family log-scale `log ε_m`), for the augmented-LAML hypergradient.
///
/// THE GAP THIS CLOSES (gam#854). `H_Φ` is built from the JOINT Hessian
/// `H_joint(β, ρ) = H_data + Σ_m λ_m H_m^pen(β; ε_m)` (value path
/// [`joint_jeffreys_term`]), so it depends on ρ BOTH through β̂ — the mode response,
/// supplied by [`joint_jeffreys_hphi_directional_derivative`] — AND EXPLICITLY
/// through the `λ_m`/`ε_m` that scale and shape the penalty blocks INSIDE
/// `H_joint`. The outer score
///   `½ tr[(H+S_λ+H_Φ)⁻¹ ∂_ρ(H+S_λ+H_Φ)]`
/// therefore needs the explicit term `½ tr[(·)⁻¹ ∂_ρ H_Φ|_β]`; omitting it leaves
/// the analytic hypergradient short on exactly the most-active penalty axis (the
/// residual spatial-adaptive tension-axis miss).
///
/// The arithmetic is IDENTICAL to the mode-response drift, with the perturbation
/// sourced from the explicit ρ-derivatives instead of `Hdot[δ]`/`H²dot[δ,e_a]`:
///   * `pert_h = ∂_ρ H_joint|_β`              perturbs `H_id` and hence `K`,
///   * `pert_hessian_dir(e_a) = ∂_ρ Hdot[e_a]|_β`  perturbs each axis derivative `D_a`,
/// while the BASE `M_a = K D_a` uses `base_hessian_dir` (= the real `Hdot[e_a]` at
/// the current point). Returns the zero matrix when the family lacks the exact
/// derivatives or the conditioning gate skips the term, so a clean fit is
/// byte-unchanged.
pub fn joint_jeffreys_hphi_explicit_param_derivative<BaseFn, PertFn>(
    h_joint: ArrayView2<'_, f64>,
    z_j: ArrayView2<'_, f64>,
    pert_h: &Array2<f64>,
    base_hessian_dir: BaseFn,
    pert_hessian_dir: PertFn,
) -> Result<Array2<f64>, String>
where
    BaseFn: FnMut(&Array1<f64>) -> Result<Option<Array2<f64>>, String>,
    PertFn: FnMut(&Array1<f64>) -> Result<Option<Array2<f64>>, String>,
{
    joint_jeffreys_hphi_perturbation_derivative(
        h_joint,
        z_j,
        base_hessian_dir,
        pert_h,
        pert_hessian_dir,
    )
}

/// Shared core for the joint-Jeffreys curvature perturbation derivative
/// `D[gate·H_Φ_raw]`, given a perturbation that acts on `H_joint` through `pert_h`
/// (a `p×p` matrix `∂H_joint`) and on each axis derivative `D_a` through
/// `pert_hessian_dir(e_a)` (a `p×p` matrix `∂Hdot[e_a]`). `base_hessian_dir(e_a)`
/// supplies the unperturbed `Hdot[e_a]` that forms the base `M_a = K D_a`.
///
/// Both the mode-response drift (perturbation `Hdot[δ]`, `H²dot[δ,e_a]`) and the
/// explicit ρ-derivative (perturbation `∂_ρ H_joint`, `∂_ρ Hdot[e_a]`) are
/// instances. It reproduces the value path's reduced information, smooth
/// conditioning gate, and floored pseudo-inverse, and differentiates the gate
/// (`(D gate)·H_Φ_raw`) so the result is consistent with the gated `H_Φ` the
/// objective folds into `½ log|H+S_λ+H_Φ|`. With `K = H_id⁻¹` the floored
/// pseudo-inverse, `M_a = K D_a`, `H_Φ_raw[a,b] = ½⟨vec M_a, vec M_b⟩`:
///   * `δK = −K Ḋ K`, `Ḋ = Z_Jᵀ (∂H_joint) Z_J` (exact on the unfloored spectrum,
///     matching the value path; the floored-spectrum divided-difference correction
///     is the same residual the value/gradient pair already carries),
///   * `δM_a = −K Ḋ M_a + K (Z_Jᵀ ∂Hdot[e_a] Z_J)`,
///   * `δH_Φ_raw[a,b] = ½(⟨δM_a, M_b⟩ + ⟨M_a, δM_b⟩)`.
fn joint_jeffreys_hphi_perturbation_derivative<BaseFn, PertFn>(
    h_joint: ArrayView2<'_, f64>,
    z_j: ArrayView2<'_, f64>,
    mut base_hessian_dir: BaseFn,
    pert_h: &Array2<f64>,
    mut pert_hessian_dir: PertFn,
) -> Result<Array2<f64>, String>
where
    BaseFn: FnMut(&Array1<f64>) -> Result<Option<Array2<f64>>, String>,
    PertFn: FnMut(&Array1<f64>) -> Result<Option<Array2<f64>>, String>,
{
    let p = h_joint.nrows();
    if h_joint.ncols() != p {
        return Err(format!(
            "joint_jeffreys_hphi_perturbation_derivative: H must be square, got {}x{}",
            h_joint.nrows(),
            h_joint.ncols()
        ));
    }
    if z_j.nrows() != p {
        return Err(format!(
            "joint_jeffreys_hphi_perturbation_derivative: Z_J has {} rows, expected {p}",
            z_j.nrows()
        ));
    }
    if pert_h.nrows() != p || pert_h.ncols() != p {
        return Err(format!(
            "joint_jeffreys_hphi_perturbation_derivative: pert_h shape {}x{} != {p}x{p}",
            pert_h.nrows(),
            pert_h.ncols()
        ));
    }
    let m = z_j.ncols();
    if m == 0 || p == 0 {
        return Ok(Array2::zeros((p, p)));
    }

    // Reproduce EXACTLY the value-path reduced information, conditioning gate, and
    // floored pseudo-inverse so the derivative is consistent with the `H_Φ` the
    // objective uses.
    let hz0 = h_joint.dot(&z_j);
    let h_id = z_j.t().dot(&hz0);
    let mut h_id_sym = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        for j in 0..m {
            h_id_sym[[i, j]] = 0.5 * (h_id[[i, j]] + h_id[[j, i]]);
        }
    }
    let (evals, evecs) = h_id_sym.eigh(Side::Lower).map_err(|e| {
        format!("joint_jeffreys_hphi_perturbation_derivative: eigendecomposition failed: {e}")
    })?;
    let lambda_max = evals.iter().cloned().fold(0.0_f64, f64::max);
    let lambda_min = evals.iter().cloned().fold(f64::INFINITY, f64::min);
    let gate_weight = conditioning_gate_weight(lambda_min, lambda_max);
    if gate_weight == 0.0 {
        // Fully gated out ⇒ H_Φ ≡ 0 in a neighborhood ⇒ its derivative vanishes.
        return Ok(Array2::zeros((p, p)));
    }
    let floor = (REDUCED_INFO_RELATIVE_FLOOR * lambda_max).max(REDUCED_INFO_ABSOLUTE_FLOOR);
    let mut inv_diag = Array1::<f64>::zeros(m);
    for (i, &lam) in evals.iter().enumerate() {
        // Floor on |λ| with the SIGNED true inverse, identical to the value/gradient
        // path in `joint_jeffreys_term` (gam#814).
        if lam.abs() >= floor {
            inv_diag[i] = 1.0 / lam;
        } else {
            inv_diag[i] = 1.0 / floor;
        }
    }
    let scaled = &evecs * &inv_diag.view().insert_axis(ndarray::Axis(0));
    let h_id_inv = scaled.dot(&evecs.t());

    // Ḋ = Z_Jᵀ (∂H_joint) Z_J, the reduced perturbation of the reduced information.
    let dbar = z_j.t().dot(&pert_h.dot(&z_j)); // m x m
    let k_dbar = h_id_inv.dot(&dbar); // K Ḋ

    // For each canonical axis e_a: base M_a = K D_a and its perturbation δM_a. We
    // assemble flattened vec(M_a) and vec(δM_a) so the final contraction is a pair
    // of m·m inner products per (a,b).
    let mut m_rows = Array2::<f64>::zeros((p, m * m)); // vec(M_a)
    let mut dm_rows = Array2::<f64>::zeros((p, m * m)); // vec(δM_a)
    let mut axis = Array1::<f64>::zeros(p);
    for a in 0..p {
        axis.fill(0.0);
        axis[a] = 1.0;
        let hdot_a = match base_hessian_dir(&axis)? {
            Some(hd) => hd,
            None => return Ok(Array2::zeros((p, p))),
        };
        if hdot_a.nrows() != p || hdot_a.ncols() != p {
            return Err(format!(
                "joint_jeffreys_hphi_perturbation_derivative: Hdot[e_a] shape {}x{} != {p}x{p}",
                hdot_a.nrows(),
                hdot_a.ncols()
            ));
        }
        let d_a = z_j.t().dot(&hdot_a.dot(&z_j)); // Z_Jᵀ ∂_a H Z_J
        let m_a = h_id_inv.dot(&d_a); // K D_a

        let pert_hdot_a = match pert_hessian_dir(&axis)? {
            Some(h2) => h2,
            None => return Ok(Array2::zeros((p, p))),
        };
        if pert_hdot_a.nrows() != p || pert_hdot_a.ncols() != p {
            return Err(format!(
                "joint_jeffreys_hphi_perturbation_derivative: ∂Hdot[e_a] shape {}x{} != {p}x{p}",
                pert_hdot_a.nrows(),
                pert_hdot_a.ncols()
            ));
        }
        let d_a_pert = z_j.t().dot(&pert_hdot_a.dot(&z_j)); // Z_Jᵀ (∂Hdot[e_a]) Z_J

        // δM_a = (∂K) D_a + K (∂D_a) = −K Ḋ M_a + K D_a^pert.
        let dm_a = &h_id_inv.dot(&d_a_pert) - &k_dbar.dot(&m_a);

        let mut col = 0usize;
        for i in 0..m {
            for j in 0..m {
                m_rows[[a, col]] = m_a[[i, j]];
                dm_rows[[a, col]] = dm_a[[i, j]];
                col += 1;
            }
        }
    }

    // δ(½Gram)[a,b] = ½ (⟨vec δM_a, vec M_b⟩ + ⟨vec M_a, vec δM_b⟩). Symmetric.
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

    // GATE DERIVATIVE (value↔gradient consistency, gam#854). `H_Φ = G(λ_min,λ_max)·H_Φ_raw`,
    // and the gate moves with the perturbation through the reduced eigenvalues, so
    //   `D[G·H_Φ_raw] = (D G)·H_Φ_raw + G·(D H_Φ_raw)`,
    // `D G = G_λmin·δλ_min + G_λmax·δλ_max`, `δλ = vᵀ Ḋ v`. Identically zero on a
    // saturated gate, so fully-active / well-conditioned fits are byte-unchanged.
    let mut result = out * gate_weight;
    let (g_dlmin, g_dlmax) = conditioning_gate_weight_grad(lambda_min, lambda_max);
    if g_dlmin != 0.0 || g_dlmax != 0.0 {
        let mut idx_min = 0usize;
        let mut idx_max = 0usize;
        for i in 1..m {
            if evals[i] < evals[idx_min] {
                idx_min = i;
            }
            if evals[i] > evals[idx_max] {
                idx_max = i;
            }
        }
        let extreme_perturbation = |idx: usize| -> f64 {
            let v = evecs.column(idx);
            v.dot(&dbar.dot(&v))
        };
        let d_gate =
            g_dlmin * extreme_perturbation(idx_min) + g_dlmax * extreme_perturbation(idx_max);
        if d_gate != 0.0 {
            let hphi_raw = m_rows.dot(&m_rows.t()).mapv(|x| 0.5 * x);
            result.scaled_add(d_gate, &hphi_raw);
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// `joint_jeffreys_hphi_explicit_param_derivative` must equal the central
    /// finite difference of the value-path gated curvature `H_Φ` w.r.t. a scalar
    /// outer parameter `s`, on a synthetic β-frozen family where
    /// `H_joint(s) = H0 + s·P` and `Hdot[e_a](s) = G_a + s·Q_a` (so `∂_s H_joint = P`,
    /// `∂_s Hdot[e_a] = Q_a`). H0's smallest reduced eigenvalue sits in the ABSOLUTE
    /// gate transition band (exercising the gate derivative) with no floored
    /// eigenvalue, the regime of the gam#854 tension-axis miss.
    #[test]
    fn explicit_param_derivative_matches_finite_difference() {
        let p = 4usize;
        let z = Array2::<f64>::eye(p);
        let h0 = array![
            [30.0, 1.0, 0.5, 0.2],
            [1.0, 12.0, 0.3, 0.1],
            [0.5, 0.3, 5.0, 0.4],
            [0.2, 0.1, 0.4, 1.5],
        ];
        let pmat = array![
            [2.0, 0.3, 0.1, 0.05],
            [0.3, 1.5, 0.2, 0.1],
            [0.1, 0.2, 1.0, 0.15],
            [0.05, 0.1, 0.15, 0.7],
        ];
        let make_sym = |seed: f64| -> Array2<f64> {
            let mut a = Array2::<f64>::zeros((p, p));
            for i in 0..p {
                for j in 0..p {
                    a[[i, j]] = (seed + 0.37 * (i as f64) - 0.19 * (j as f64)).sin()
                        + 0.5 * ((i + j) as f64 * seed).cos();
                }
            }
            let at = a.t().to_owned();
            (&a + &at).mapv(|v| 0.5 * v)
        };
        let g: Vec<Array2<f64>> = (0..p).map(|a| make_sym(1.0 + a as f64)).collect();
        let q: Vec<Array2<f64>> = (0..p).map(|a| make_sym(7.0 + 2.0 * a as f64)).collect();
        let axis_index = |axis: &Array1<f64>| -> usize {
            axis.iter().position(|&x| x != 0.0).expect("one-hot axis")
        };

        let hphi_at = |s: f64| -> Array2<f64> {
            let h = &h0 + &pmat.mapv(|v| s * v);
            joint_jeffreys_term(h.view(), z.view(), |axis: &Array1<f64>| {
                let a = axis_index(axis);
                Ok(Some(&g[a] + &q[a].mapv(|v| s * v)))
            })
            .expect("value-path H_Φ")
            .2
        };

        let s0 = 0.0_f64;
        let hh = 1e-5;
        let fd = (&hphi_at(s0 + hh) - &hphi_at(s0 - hh)).mapv(|v| v / (2.0 * hh));

        let h_s0 = &h0 + &pmat.mapv(|v| s0 * v);
        let analytic = joint_jeffreys_hphi_explicit_param_derivative(
            h_s0.view(),
            z.view(),
            &pmat,
            |axis: &Array1<f64>| {
                let a = axis_index(axis);
                Ok(Some(&g[a] + &q[a].mapv(|v| s0 * v)))
            },
            |axis: &Array1<f64>| {
                let a = axis_index(axis);
                Ok(Some(q[a].clone()))
            },
        )
        .expect("explicit ∂_s H_Φ");

        let mut max_err = 0.0_f64;
        for i in 0..p {
            for j in 0..p {
                max_err = max_err.max((analytic[[i, j]] - fd[[i, j]]).abs());
            }
        }
        assert!(
            max_err < 1e-5,
            "explicit ∂_s H_Φ mismatch vs FD: max_err={max_err}"
        );
    }

    /// Test-only convenience predicate: `true` when the smooth gate weight is exactly
    /// `0` (the term is fully skippable). Non-test code uses `conditioning_gate_weight`
    /// directly so the transition band stays continuous; the cheap matrix-free
    /// pre-check certifies a full skip by clearing the UPPER (`*_CLEAR`) knots.
    fn conditioning_gate_skips(lambda_min: f64, lambda_max: f64) -> bool {
        conditioning_gate_weight(lambda_min, lambda_max) == 0.0
    }

    #[test]
    fn full_span_is_identity_regardless_of_penalty() {
        // The principled cure: Z_J is the FULL identifiable span (the entire
        // reduced block), i.e. the identity, irrespective of the penalty's null
        // space. Jeffreys is self-limiting, so this does not bias identified
        // directions; it only bounds near-separating ones.
        for s in [
            Array2::<f64>::zeros((3, 3)), // pure parametric
            {
                let mut s = Array2::<f64>::zeros((3, 3));
                s[[2, 2]] = 5.0; // rank-deficient (ker dim 2)
                s
            },
            Array2::<f64>::eye(4) * 2.0, // full-rank penalty
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
    fn joint_jeffreys_term_value_gradient_consistent_below_floor() {
        // Regression for the bernoulli-MS outer-non-convergence stall
        // (gam#787/#785): a separating direction whose reduced-information
        // eigenvalue sits BELOW the floored ridge. The released code computed the
        // value as the CONSTANT `½ ln(floor)` there (derivative 0) while the
        // gradient used the floored inverse `½ (1/floor) ∂λ/∂β` (derivative
        // nonzero), so ∇Φ ≠ d/dβ Φ exactly where Firth arms. The inner KKT
        // residual then floored at that mismatch and the joint-Newton could never
        // certify. The existing above-floor FD test could not catch this (its
        // λ_min stays above the floor). Here the second eigenvalue is genuinely
        // below the floor, so the FD MUST match the analytic gradient only with
        // the C¹ linear continuation of the value below the floor.
        let p = 2usize;
        // λ_max ≈ exp(0.3) ≈ 1.35 ⇒ floor = 1e-10·λ_max ≈ 1.35e-10. With
        // ill = 1e-12 the second eigenvalue λ_min ≈ 1.16e-12 < floor.
        let ill = 1e-12_f64;
        let z = Array2::<f64>::eye(p);
        let h_at = |b: &Array1<f64>| -> Array2<f64> {
            let mut h = Array2::<f64>::zeros((p, p));
            h[[0, 0]] = b[0].exp();
            h[[1, 1]] = ill * (1.0 + b[1] * b[1]);
            h
        };
        let beta: Array1<f64> = array![0.3, -0.4];
        let hdir = |d: &Array1<f64>| -> Result<Option<Array2<f64>>, String> {
            let mut hd = Array2::<f64>::zeros((p, p));
            hd[[0, 0]] = beta[0].exp() * d[0];
            hd[[1, 1]] = ill * 2.0 * beta[1] * d[1];
            Ok(Some(hd))
        };
        let h = h_at(&beta);
        let (_phi, grad, _hphi) = joint_jeffreys_term(h.view(), z.view(), hdir).unwrap();
        // Re-derive the floor exactly as the term does, and finite-difference the
        // value the term ACTUALLY accumulates (C¹ floored-inverse antiderivative):
        //   g(λ) = ln(λ)                    for λ ≥ floor,
        //   g(λ) = λ/floor + ln(floor) − 1  for λ < floor.
        let value_at = |b: &Array1<f64>| -> f64 {
            let hh = h_at(b);
            let lam0 = hh[[0, 0]];
            let lam1 = hh[[1, 1]];
            let lambda_max = lam0.max(lam1);
            let floor = (1e-10_f64 * lambda_max).max(1e-12_f64);
            let g = |lam: f64| -> f64 {
                if lam >= floor {
                    lam.ln()
                } else {
                    lam / floor + floor.ln() - 1.0
                }
            };
            0.5 * (g(lam0) + g(lam1))
        };
        let eps = 1e-7;
        for k in 0..p {
            let mut bp = beta.clone();
            let mut bm = beta.clone();
            bp[k] += eps;
            bm[k] -= eps;
            let fd = (value_at(&bp) - value_at(&bm)) / (2.0 * eps);
            assert!(
                (grad[k] - fd).abs() <= 1e-5 * (1.0 + fd.abs()),
                "below-floor grad[{k}] {} vs fd {fd}; the Jeffreys value must be the \
                 exact antiderivative of the floored-inverse gradient",
                grad[k]
            );
        }
    }

    #[test]
    fn joint_jeffreys_term_indefinite_value_gradient_consistent() {
        // Regression for the survival clustered-PC marginal-slope inner-solve
        // crawl (gam#814). The reduced OBSERVED information `H_id = Z_Jᵀ H Z_J` is
        // NOT PSD away from the mode for a non-canonical link, so it carries a
        // MODERATE NEGATIVE eigenvalue (|λ| ≫ floor). The released code floored on
        // the SIGNED eigenvalue (`1/max(λ, floor)`), pinning that moderate negative
        // to `+1/floor ≈ 1.7e6` — a phantom Firth score that no Newton step could
        // satisfy. The fix floors on `|λ|` and keeps the sign of the inverse
        // (`1/λ`), with the value `½ ln|λ|`. This test exercises that branch: the
        // second eigenvalue is genuinely NEGATIVE and well above the floor in
        // magnitude, so the FD gradient must match the analytic gradient ONLY with
        // the signed `1/λ` inverse and the `½ ln|λ|` value. The existing PSD-only
        // FD test cannot catch a sign error here.
        //
        // H(beta) = diag(exp(beta0), -(1 + beta1^2)), Z_J = I. λ_min < 0 < 1 so the
        // conditioning gate fires fully (weight 1) and the active path runs.
        let p = 2usize;
        let z = Array2::<f64>::eye(p);
        let h_at = |b: &Array1<f64>| -> Array2<f64> {
            let mut h = Array2::<f64>::zeros((p, p));
            h[[0, 0]] = b[0].exp();
            h[[1, 1]] = -(1.0 + b[1] * b[1]);
            h
        };
        let beta: Array1<f64> = array![0.3, -0.4];
        // Hdot[d] = diag(exp(b0) d0, -2 b1 d1).
        let hdir = |d: &Array1<f64>| -> Result<Option<Array2<f64>>, String> {
            let mut hd = Array2::<f64>::zeros((p, p));
            hd[[0, 0]] = beta[0].exp() * d[0];
            hd[[1, 1]] = -2.0 * beta[1] * d[1];
            Ok(Some(hd))
        };
        let h = h_at(&beta);
        let (phi, grad, hphi) = joint_jeffreys_term(h.view(), z.view(), hdir).unwrap();
        // Sanity: the negative direction must NOT have been pinned to the floor.
        // With the buggy signed floor, |grad| would be ~1/floor ≈ 1.7e9 here.
        assert!(
            grad.iter().all(|g| g.abs() < 1e3),
            "indefinite direction must use the signed 1/λ inverse, not 1/floor; grad={grad:?}"
        );
        // Φ = ½(ln|λ0| + ln|λ1|) = ½ ln(exp(b0) · (1 + b1^2)), both |λ| ≫ floor.
        let expected_phi = 0.5 * (beta[0].exp() * (1.0 + beta[1] * beta[1])).ln();
        assert!(
            (phi - expected_phi).abs() < 1e-9,
            "phi {phi} vs {expected_phi}"
        );
        // Finite-difference the value the term accumulates (½ Σ ln|λ_i| since both
        // magnitudes are above the floor) and compare to the analytic gradient.
        let value_at = |b: &Array1<f64>| -> f64 {
            let hh = h_at(b);
            0.5 * (hh[[0, 0]].abs().ln() + hh[[1, 1]].abs().ln())
        };
        let eps = 1e-7;
        for k in 0..p {
            let mut bp = beta.clone();
            let mut bm = beta.clone();
            bp[k] += eps;
            bm[k] -= eps;
            let fd = (value_at(&bp) - value_at(&bm)) / (2.0 * eps);
            assert!(
                (grad[k] - fd).abs() <= 1e-5 * (1.0 + fd.abs()),
                "indefinite grad[{k}] {} vs fd {fd}; value must be ½Σln|λ| (signed 1/λ inverse)",
                grad[k]
            );
        }
        // H_Phi is a Gram (½ S Sᵀ) so it stays symmetric PSD even with a negative
        // eigenvalue in H_id.
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
        h[[0, 0]] = 200.0;
        h[[1, 1]] = 100.0; // λ_min=100 ≫ 16 (upper knot), ratio 0.5 ⇒ fully skipped
        let hdir = |d: &Array1<f64>| -> Result<Option<Array2<f64>>, String> {
            // Nonzero derivative; would yield a nonzero gradient if not gated.
            let mut hd = Array2::<f64>::zeros((p, p));
            hd[[0, 0]] = 3.0 * d[0];
            hd[[1, 1]] = 5.0 * d[1];
            Ok(Some(hd))
        };
        let (phi, grad, hphi) = joint_jeffreys_term(h.view(), z.view(), hdir).unwrap();
        assert_eq!(phi, 0.0, "well-conditioned ⇒ no Jeffreys value");
        assert!(
            grad.iter().all(|v| *v == 0.0),
            "well-conditioned ⇒ zero grad"
        );
        assert!(
            hphi.iter().all(|v| *v == 0.0),
            "well-conditioned ⇒ zero curvature"
        );
    }

    #[test]
    fn conditioning_gate_fires_only_below_threshold() {
        // Bracket the COMBINED relative+absolute gate. To be SKIPPED a fit must be
        // well-conditioned both relatively (ratio ≥ 1e-8) AND absolutely
        // (λ_min ≥ 1); if EITHER fails the term fires. This pins the "no cost on a
        // genuinely well-conditioned large-n fit, full term on a (relatively OR
        // absolutely) near-separating one" boundary.
        let p = 2usize;
        let z = Array2::<f64>::eye(p);
        let hdir = |d: &Array1<f64>| -> Result<Option<Array2<f64>>, String> {
            let mut hd = Array2::<f64>::zeros((p, p));
            hd[[0, 0]] = d[0];
            hd[[1, 1]] = d[1];
            Ok(Some(hd))
        };
        // λ_max = 1.0 (h[[0,0]]); λ_min = the closure argument (h[[1,1]]).
        let mk = |lmin: f64| {
            let mut h = Array2::<f64>::zeros((p, p));
            h[[0, 0]] = 1.0;
            h[[1, 1]] = lmin;
            h
        };
        // Genuinely well-conditioned (large-n): ratio 0.5 ≥ 1e-8 AND λ_min = 50 ≫ 1
        // ⇒ gated. NOTE the second arg of `mk` is λ_min while `h[[0,0]]` is fixed
        // at 1.0 in the closure above; we override it here to a large λ_max.
        let mut above = mk(50.0);
        above[[0, 0]] = 100.0;
        let (phi_a, grad_a, _) = joint_jeffreys_term(above.view(), z.view(), hdir).unwrap();
        assert_eq!(phi_a, 0.0);
        assert!(grad_a.iter().all(|v| *v == 0.0));
        // Relatively near-separating (ratio < 1e-8, λ_max = 1.0) ⇒ fires.
        let below_rel = mk(CONDITIONING_GATE_RELATIVE * 0.1);
        let (phi_r, _g, hphi_r) = joint_jeffreys_term(below_rel.view(), z.view(), hdir).unwrap();
        assert!(phi_r != 0.0, "relatively near-separating must fire");
        assert!(hphi_r.iter().any(|v| v.abs() > 0.0));
        // ABSOLUTELY near-separating at SMALL n: λ_max = 1.0, λ_min = 0.05 ⇒ ratio
        // 0.05 ≥ 1e-8 (the relative gate alone would WRONGLY skip), but λ_min < 1
        // ⇒ the n-aware ABSOLUTE gate fires the stabilising term. This is exactly
        // the FIX-C small-n admixture-cline regime the relative-only gate missed.
        let below_abs = mk(0.05);
        let (phi_b, _grad_b, hphi_b) =
            joint_jeffreys_term(below_abs.view(), z.view(), hdir).unwrap();
        assert!(
            phi_b != 0.0,
            "absolutely near-separating (small-n) must fire even though the relative ratio clears the gate",
        );
        assert!(
            hphi_b.iter().any(|v| v.abs() > 0.0),
            "absolute-gate firing must produce nonzero bounding curvature",
        );
    }

    #[test]
    fn conditioning_gate_predicate_relative_and_absolute() {
        // Unit coverage of the shared predicate's two-sided logic.
        // Well-conditioned (both gates pass) ⇒ skip.
        assert!(conditioning_gate_skips(50.0, 100.0));
        // Relatively ill-conditioned ⇒ do not skip.
        assert!(!conditioning_gate_skips(
            CONDITIONING_GATE_RELATIVE * 0.1,
            1.0
        ));
        // Absolutely near-separating at small n (ratio fine, λ_min < 1) ⇒ do not skip.
        assert!(!conditioning_gate_skips(0.05, 1.0));
        // SMOOTH boundary: λ_min at the lower (firing) knot is still fully active,
        // and anywhere inside the ramp band is only partially tapered — NOT a full
        // skip. A full skip requires clearing the UPPER (`*_CLEAR`) knot.
        assert!(!conditioning_gate_skips(
            CONDITIONING_GATE_ABSOLUTE,
            CONDITIONING_GATE_ABSOLUTE
        ));
        assert!(!conditioning_gate_skips(4.0, 100.0));
        // Comfortably identified (λ_min past the upper knot, fine ratio) ⇒ skip.
        assert!(conditioning_gate_skips(
            CONDITIONING_GATE_ABSOLUTE_CLEAR,
            CONDITIONING_GATE_ABSOLUTE_CLEAR
        ));
        // Non-positive / non-finite spectra ⇒ never skip (fully active).
        assert!(!conditioning_gate_skips(0.0, 0.0));
        assert!(!conditioning_gate_skips(f64::NAN, 100.0));
    }

    #[test]
    fn conditioning_gate_weight_is_continuous_and_monotone() {
        // The whole point of the smooth gate (#787): the weight is C⁰/C¹ across the
        // absolute transition band [1, 16], so the outer LAML objective does not
        // jump as β̂(ρ) carries λ_min across the boundary. Sweep λ_min upward with a
        // fixed large λ_max (relative sub-weight pinned to 0 throughout) and assert
        // the weight is 1 at/below the lower knot, 0 at/above the upper knot,
        // strictly decreasing inside, and never jumps by more than a small step.
        let lambda_max = 1.0e6; // ratio ≪ knots, so the absolute ramp dominates
        let w = |lmin: f64| conditioning_gate_weight(lmin, lambda_max);
        assert_eq!(w(CONDITIONING_GATE_ABSOLUTE), 1.0);
        assert_eq!(w(0.1), 1.0);
        assert_eq!(w(CONDITIONING_GATE_ABSOLUTE_CLEAR), 0.0);
        assert_eq!(w(100.0), 0.0);
        let mut prev = 1.0;
        let n = 200usize;
        for k in 0..=n {
            let lmin = CONDITIONING_GATE_ABSOLUTE
                + (CONDITIONING_GATE_ABSOLUTE_CLEAR - CONDITIONING_GATE_ABSOLUTE)
                    * (k as f64 / n as f64);
            let cur = w(lmin);
            assert!((0.0..=1.0).contains(&cur));
            assert!(cur <= prev + 1e-12, "weight must be non-increasing");
            assert!(
                (prev - cur).abs() < 0.1,
                "no large jumps across the smooth band (continuity)"
            );
            prev = cur;
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

    // ---- cheap matrix-free conditioning pre-check ----

    /// Build a diagonal-matvec closure for a synthetic spectrum (the joint
    /// information on the full span is `H` itself, so a diagonal `H` exercises the
    /// Lanczos bound against a known `[λ_min, λ_max]`).
    fn diag_hv(diag: Vec<f64>) -> impl FnMut(&Array1<f64>) -> Result<Array1<f64>, String> {
        move |v: &Array1<f64>| {
            let mut out = Array1::<f64>::zeros(v.len());
            for (i, &d) in diag.iter().enumerate() {
                out[i] = d * v[i];
            }
            Ok(out)
        }
    }

    #[test]
    fn cheap_precheck_skips_clearly_well_conditioned_large_p() {
        // A wide (p ≥ threshold) well-conditioned spectrum: every eigenvalue in
        // [200, 250], so λ_min = 200 clears the 8x margin on the smooth absolute
        // clear knot (16) and the ratio 0.8 clears the relative margin. The
        // conservative Lanczos bounds must still clear both gates ⇒ skippable.
        // This is the common large-p fast path the pre-check exists to make
        // matrix-free-free of any dense formation.
        let p = 200usize;
        let mut diag = vec![220.0; p];
        diag[0] = 200.0; // λ_min
        diag[1] = 250.0; // λ_max
        let skippable = jeffreys_term_skippable_via_matvec(diag_hv(diag), p).unwrap();
        assert!(
            skippable,
            "clearly well-conditioned wide fit must be skippable"
        );
    }

    #[test]
    fn cheap_precheck_does_not_skip_near_separating() {
        // One near-zero eigenvalue (λ_min = 1e-3) below the absolute gate (1.0):
        // the term is genuinely needed here. The conservative bounds must NOT
        // certify skippable, so the caller falls through to the exact formation.
        let p = 200usize;
        let mut diag = vec![50.0; p];
        diag[7] = 1e-3; // near-separating direction
        let skippable = jeffreys_term_skippable_via_matvec(diag_hv(diag), p).unwrap();
        assert!(
            !skippable,
            "a near-separating direction must NOT be skipped (term is needed)"
        );
    }

    #[test]
    fn cheap_precheck_does_not_skip_below_size_threshold() {
        // Small p: even a perfectly-conditioned spectrum is never pre-checked
        // (the exact dense eigh is already cheap there). Guarantees the small-p
        // BMS-style fits keep running the exact dense path unchanged.
        let p = CHEAP_CONDITIONING_PRECHECK_MIN_DIM - 1;
        let diag = vec![100.0; p];
        let skippable = jeffreys_term_skippable_via_matvec(diag_hv(diag), p).unwrap();
        assert!(
            !skippable,
            "below the size threshold the pre-check never skips"
        );
    }

    #[test]
    fn cheap_precheck_does_not_skip_marginal_absolute() {
        // Absolutely marginal: λ_min = 2.0 clears the bare absolute gate (1.0) but
        // NOT the 8× safety margin (needs ≥ 8). The conservative pre-check must
        // refuse to skip even though the EXACT gate would skip — the asymmetric
        // safety bias (false fall-through is cheap, false skip is fatal).
        let p = 200usize;
        let mut diag = vec![50.0; p];
        diag[3] = 2.0;
        let skippable = jeffreys_term_skippable_via_matvec(diag_hv(diag), p).unwrap();
        assert!(
            !skippable,
            "λ_min within the 8× absolute margin must conservatively fall through"
        );
    }

    #[test]
    fn cheap_precheck_skip_implies_exact_gate_skips() {
        // CONSISTENCY: wherever the cheap pre-check declares skippable, the EXACT
        // conditioning gate on the same spectrum must also skip (return the zero
        // term). This is the byte-identical-on-skip guarantee. Sweep a range of
        // well-conditioned spectra and assert the implication.
        let p = 150usize;
        let z = Array2::<f64>::eye(p);
        let hdir = |_d: &Array1<f64>| -> Result<Option<Array2<f64>>, String> {
            Ok(Some(Array2::<f64>::zeros((p, p))))
        };
        for &lmin in &[10.0_f64, 25.0, 80.0, 200.0] {
            let mut diag = vec![lmin * 4.0; p];
            diag[0] = lmin;
            let cheap_skip = jeffreys_term_skippable_via_matvec(diag_hv(diag.clone()), p).unwrap();
            if cheap_skip {
                // The exact path on the same dense H must produce the zero term.
                let mut h = Array2::<f64>::zeros((p, p));
                for (i, &d) in diag.iter().enumerate() {
                    h[[i, i]] = d;
                }
                let (phi, grad, hphi) = joint_jeffreys_term(h.view(), z.view(), hdir).unwrap();
                assert_eq!(
                    phi, 0.0,
                    "cheap-skip ⇒ exact phi must be zero (byte-identical)"
                );
                assert!(grad.iter().all(|v| *v == 0.0));
                assert!(hphi.iter().all(|v| *v == 0.0));
            }
        }
    }

    #[test]
    fn cheap_precheck_bails_on_nonfinite_matvec() {
        // A matvec that returns non-finite values cannot certify conditioning ⇒
        // the pre-check must return false (never skip on an unresolved estimate).
        let p = 200usize;
        let hv = |v: &Array1<f64>| -> Result<Array1<f64>, String> {
            Ok(Array1::from_elem(v.len(), f64::NAN))
        };
        assert!(!jeffreys_term_skippable_via_matvec(hv, p).unwrap());
    }
}
