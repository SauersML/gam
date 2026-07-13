//! The Jeffreys-prior contribution to the joint objective: subspace construction,
//! skippability gating, conditioning-gate weights, the value/term assembly, the
//! second-order completion, and the outer Jeffreys H_phi (+drift) terms.

use super::*;

pub(crate) fn block_param_ranges(specs: &[ParameterBlockSpec]) -> Vec<(usize, usize)> {
    block_offsets_from_specs(specs)
        .iter()
        .map(|r| (r.start, r.end))
        .collect()
}

/// Build the joint Jeffreys/Firth basis `Z_J` (block-diagonal stack of each
/// block's per-block span) for the universal robustness term.
///
/// Each block contributes its FULL reduced coefficient span (`I_p` per block) —
/// the principled cure. Because the Jeffreys score is `O(1)` against the data's
/// `O(n)` Fisher information, applying it on the full span is the `O(1/n)` Firth
/// bias correction on data-identified directions (no bias on genuine smooth
/// fits) and the missing `O(1)`-bounding curvature on ANY near-separating
/// direction — penalized (`range(S)`) or not (`ker(S)`) — so the inner objective
/// becomes coercive with a finite unique minimizer. The previous `ker(S)`-only
/// scoping could not reach a near-separation on a penalized spline direction,
/// which was the residual BMS-probit pathology.
///
/// The per-block bases are embedded block-diagonally into the joint
/// `total_p x m_total` matrix. Returns `None` only for an empty system.
///
/// The Jeffreys conditioning gate, not the smoothing penalty null space,
/// decides whether this basis contributes at the current iterate.
pub(crate) fn build_joint_jeffreys_subspace(
    specs: &[ParameterBlockSpec],
    ranges: &[(usize, usize)],
) -> Result<Option<Array2<f64>>, String> {
    let total_p = ranges.last().map(|(_, e)| *e).unwrap_or(0);
    if total_p == 0 {
        return Ok(None);
    }
    let mut per_block: Vec<Array2<f64>> = Vec::with_capacity(specs.len());
    let mut m_total = 0usize;
    for (b, _spec) in specs.iter().enumerate() {
        let (start, end) = ranges[b];
        let p_block = end - start;
        // Full identifiable-span Jeffreys: `Z_J = I_{p_block}` over the entire
        // reduced block coefficient space. The aggregate penalty only fixes the
        // block dimension; the span no longer depends on `ker(S)`.
        let aggregate = Array2::<f64>::zeros((p_block, p_block));
        let subspace =
            gam_solve::estimate::reml::jeffreys_subspace::jeffreys_subspace_from_penalty(
                aggregate.view(),
            )?;
        m_total += subspace.span_dim();
        per_block.push(subspace.columns);
    }
    if m_total == 0 {
        return Ok(None);
    }
    let mut z_joint = Array2::<f64>::zeros((total_p, m_total));
    let mut col_cursor = 0usize;
    for (b, columns) in per_block.iter().enumerate() {
        let (start, _) = ranges[b];
        let m_block = columns.ncols();
        let p_block = columns.nrows();
        for j in 0..m_block {
            for i in 0..p_block {
                z_joint[[start + i, col_cursor + j]] = columns[[i, j]];
            }
        }
        col_cursor += m_block;
    }
    Ok(Some(z_joint))
}

/// CHEAP, matrix-free conditioning pre-check: can the always-on Jeffreys term be
/// PROVABLY skipped at this working point WITHOUT forming the dense joint Hessian
/// `H` or running the `O(p³)` reduced eigendecomposition?
///
/// This is the perf gate in front of the expensive `custom_family_joint_jeffreys_*`
/// formation. On the FULL span (`Z_J = I`) the reduced information is `H_id = H`,
/// so the conditioning gate only needs `H`'s extreme eigenvalues — and those can
/// be bounded conservatively from a few Hessian-vector products against the SAME
/// `joint_hessian_source` operator the inner Newton already built (matrix-free on
/// the large-`p` path, dense otherwise). When the conservative bounds clear both
/// gates with a safe margin (see `jeffreys_term_skippable_via_matvec`), the exact
/// gate is CERTAIN to return the zero term, so the caller skips the dense `H`
/// materialization, the `Z_JᵀHZ_J` build, the eigendecomposition, the `∇Φ`/`H_Φ`
/// assembly, and the Q1 outer drift entirely — returning the EXACT-ZERO term,
/// byte-identical to the gated-off dense path. Returns `false` (never skip)
/// whenever the cheap bounds are unresolved or merely near the gate, so any fit
/// where the term might bite still flows to the exact formation.
///
/// Matrix-free preservation: the pre-check issues only `O(p·k)` (`k≤12`) matvecs
/// through `source` and forms nothing dense at `p`-scale; on a well-conditioned
/// large-`p` matrix-free fit (the common case) it returns `true` and NOTHING
/// dense is ever built — preserving the matrix-free path the dense `H_id`
/// formation was defeating. Only on a genuinely near-separating large-`p` fit
/// (rare) does it return `false` and fall through to the inherent `O(p²)` dense
/// `H_id`/`H_Φ` formation, where that cost is justified.
pub(crate) fn jeffreys_term_skippable_for_source(
    source: &JointHessianSource,
    total_p: usize,
) -> Result<bool, String> {
    // Small joint system: the dense reduced eigendecomposition is itself cheap
    // (`O(p³)` with `p` in the tens), so run the EXACT conditioning gate directly
    // instead of forcing the always-on Jeffreys term on every cycle. The previous
    // unconditional `false` here meant a small fit ALWAYS paid the full
    // `O(p·n·special-fn)` all-axes Jeffreys directional-derivative sweep (and its
    // per-row allocations) on EVERY inner-Newton cycle and EVERY outer LAML eval —
    // the constant-scale survival location-scale #1389 non-termination, where a
    // bounded `n=300` fit ran past the 600s per-test CI cap. Form the
    // `total_p × total_p` H once (a clone for the dense source, `total_p` matvecs
    // for the operator — both cheap below the threshold) and apply the SAME
    // `conditioning_gate_weight` the term assembly uses, so a well-conditioned
    // cycle skips a provably-zero term (byte-identical to forming it) while a
    // near-separating cycle still falls through to the exact term and keeps the
    // Firth bound exactly where the ridge needs it.
    if total_p < gam_solve::estimate::reml::jeffreys_subspace::CHEAP_CONDITIONING_PRECHECK_MIN_DIM {
        let h_dense = match source {
            JointHessianSource::Dense(matrix) => matrix.clone(),
            JointHessianSource::Operator { apply, .. } => {
                let mut h = Array2::<f64>::zeros((total_p, total_p));
                let mut e_a = Array1::<f64>::zeros(total_p);
                for a in 0..total_p {
                    e_a[a] = 1.0;
                    let col = apply(&e_a)?;
                    e_a[a] = 0.0;
                    if col.len() != total_p {
                        // Operator returned an unexpected shape: fall through to the
                        // exact term rather than risk a wrong skip.
                        return Ok(false);
                    }
                    for r in 0..total_p {
                        h[[r, a]] = col[r];
                    }
                }
                h
            }
        };
        return gam_solve::estimate::reml::jeffreys_subspace::jeffreys_term_skippable_dense(
            h_dense.view(),
        );
    }
    // Matrix-free Hessian-vector product against the OBSERVED joint information.
    // For families whose Jeffreys information IS the observed Hessian (the trait
    // default), `joint_jeffreys_term`'s reduced information is `Z_JᵀHZ_J` with
    // `Z_J = I`, i.e. exactly the UNRIDGED likelihood joint Hessian `H` that
    // `exact_newton_joint_hessian_with_specs` materializes; the `Operator::apply`
    // / `Dense` here is that SAME `H` (the workspace's `hessian_matvec`, which the
    // dense source also reconstructs). So the pre-check estimates the spectrum of
    // precisely the matrix the dense path eigendecomposes — the skip decision and
    // the exact gate are consistent by construction, with no ridge discrepancy
    // (the solver's separate ridged solve operator is not involved here).
    //
    // EXPECTED-INFORMATION CAVEAT (gam#1020): when the family overrides
    // `joint_jeffreys_information_with_specs` with the expected Fisher
    // information, the gate eigendecomposes a DIFFERENT matrix than this matvec
    // probes, and the certificate does not transfer (observed information grows
    // on saturated misclassified rows where the expected information decays).
    // Callers must gate this pre-check on
    // `family.joint_jeffreys_information_matches_observed_hessian()`.
    let hv = |v: &Array1<f64>| -> Result<Array1<f64>, String> {
        match source {
            JointHessianSource::Dense(matrix) => Ok(matrix.dot(v)),
            JointHessianSource::Operator { apply, .. } => apply(v),
        }
    };
    gam_solve::estimate::reml::jeffreys_subspace::jeffreys_term_skippable_via_matvec(hv, total_p)
}

/// Evaluate ONLY the Jeffreys objective value `Phi = 1/2 log|Z_J^T H Z_J|` at
/// the current working point. Cheaper than the full term (no directional
/// derivatives), used to keep the trust-region accept/reject objective
/// consistent with the Jeffreys-modified Newton step. Returns `0.0` when there
/// is no coefficient system, the family exposes no exact joint Hessian,
/// or the reduced Fisher information is not yet SPD (the value contribution is
/// then simply omitted for that trial point — the step machinery still bounds
/// the coefficient, and the next accepted cycle re-folds a finite value).
pub(crate) fn custom_family_joint_jeffreys_value<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    ranges: &[(usize, usize)],
    z_joint: &Array2<f64>,
) -> f64 {
    let total_p = ranges.last().map(|(_, e)| *e).unwrap_or(0);
    if total_p == 0 || z_joint.ncols() == 0 {
        return 0.0;
    }
    let h_joint = match family.joint_jeffreys_information_with_specs(states, specs) {
        Ok(Some(h)) if h.nrows() == total_p && h.ncols() == total_p => h,
        _ => return 0.0,
    };
    match gam_solve::estimate::reml::jeffreys_subspace::joint_jeffreys_term(
        h_joint.view(),
        z_joint.view(),
        |_direction: &Array1<f64>| Ok(None),
    ) {
        Ok((phi, _grad, _hphi)) => phi,
        Err(_) => 0.0,
    }
}

/// Evaluate the family-general Jeffreys term `(Phi, grad, H_Phi)` at the current
/// working point from the coupled joint Hessian (Tier-B path). Returns `None`
/// when there is no coefficient system or the family does not expose an
/// exact joint Hessian (in which case the term is inapplicable and the caller
/// proceeds unchanged).
pub(crate) fn custom_family_joint_jeffreys_term<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    ranges: &[(usize, usize)],
    z_joint: &Array2<f64>,
) -> Result<Option<(f64, Array1<f64>, Array2<f64>)>, String> {
    let total_p = ranges.last().map(|(_, e)| *e).unwrap_or(0);
    if total_p == 0 || z_joint.ncols() == 0 {
        return Ok(None);
    }
    let h_joint = match family.joint_jeffreys_information_with_specs(states, specs)? {
        Some(h) => h,
        None => return Ok(None),
    };
    if h_joint.nrows() != total_p || h_joint.ncols() != total_p {
        return Ok(None);
    }
    // The reduced information and its conditioning gate are authoritative and
    // are prepared before this lazy provider can run.  A gated-off term therefore
    // performs ZERO all-axes builds.  When active, the provider is called once and
    // returns the same canonical `{Hdot[e_a]}` batch the prior eager path used.
    let term = gam_solve::estimate::reml::jeffreys_subspace::joint_jeffreys_term_batched(
        h_joint.view(),
        z_joint.view(),
        || {
            family.joint_jeffreys_information_directional_derivative_all_axes_with_specs(
                states, specs,
            )
        },
    )?;
    Ok(Some(term))
}

pub(crate) const JEFFREYS_REDUCED_INFO_RELATIVE_FLOOR: f64 = 1e-10;

pub(crate) const JEFFREYS_REDUCED_INFO_ABSOLUTE_FLOOR: f64 = 1e-12;

pub(crate) const JEFFREYS_CONDITIONING_GATE_RELATIVE: f64 = 1e-8;

pub(crate) const JEFFREYS_CONDITIONING_GATE_ABSOLUTE: f64 = 1.0;

pub(crate) const JEFFREYS_CONDITIONING_GATE_ABSOLUTE_CLEAR: f64 = 16.0;

pub(crate) const JEFFREYS_CONDITIONING_GATE_RELATIVE_CLEAR: f64 = 1e-6;

/// Trust-region tolerance for folding the second-order completion into the
/// mode-response operator. The completion is the true-Hessian remainder of the
/// Φ-augmented inner objective and is accurate ONLY where the second-order
/// expansion is valid. The divided-difference base `H_Φ` is bounded and PSD; the
/// completed curvature `H_Φ + completion` must remain PSD (a legitimate Hessian
/// of the convex Firth-augmented penalty over the under-identified span) for the
/// expansion to be trusted. We tolerate negative excursions only up to this
/// fraction of `H_Φ`'s own curvature scale — benign float rounding, never the
/// `O(λ_max)` sign-flip the near-separable cancellation produces (measured:
/// `H_Φ` spectrum `8e-9 … 1e10` but `H_Φ + completion` spectrum `−3.3e9 … 9e-3`,
/// a 33%-of-`λ_max` negative excursion AND a collapse of the curvature scale).
pub(crate) const JEFFREYS_COMPLETION_PSD_REL_TOL: f64 = 1e-8;

/// Decide whether the second-order completion may be folded into the
/// mode-response operator without destroying its positive semidefiniteness.
///
/// `H_Φ` (PSD, bounded) is the divided-difference curvature the value/logdet/trace
/// path uses; `completion` refines it into the TRUE Hessian of the Φ-augmented
/// inner objective, but only inside the second-order expansion's trust region. In
/// the near-degenerate regime the completion's `−½ tr(K·D_ab)` remainder explodes
/// negative and cancels `H_Φ`, leaving `H_Φ + completion` strongly indefinite. As
/// the mode-response operator `M = H + S_λ + H_Φ + completion`, that indefinite
/// curvature is not a legitimate Hessian: under the smooth pseudo-logdet
/// regularization a large negative eigenvalue regularizes to a near-zero pivot, so
/// the IFT solve `v_k = −M⁻¹ Ṡ_k β̂` amplifies by `~1/ε²` and the outer gradient
/// explodes (then the envelope tripwire suppresses the Hessian → `Unavailable`).
///
/// Returns `true` only when the completed curvature `H_Φ + completion` stays
/// positive semidefinite to within [`JEFFREYS_COMPLETION_PSD_REL_TOL`] of `H_Φ`'s
/// own eigenvalue scale — i.e. the completion is a small correction, not a
/// sign-flipping cancellation. When `false`, the caller keeps the bounded PSD
/// `H_Φ`, which is exactly the curvature the criterion's value and traces use, so
/// the operator and the criterion agree. PSD-PROJECTING the indefinite sum is the
/// wrong fix: it collapses the `O(1e10)` curvature scale to the surviving positive
/// dregs (`9e-3`), re-singularizing the operator. The trust decision is therefore
/// all-or-nothing per evaluation.
pub(crate) fn custom_family_jeffreys_completion_preserves_psd(
    hphi: &Array2<f64>,
    completion: &Array2<f64>,
) -> bool {
    if hphi.dim() != completion.dim() {
        return false;
    }
    let p = hphi.nrows();
    if p == 0 {
        return true;
    }
    // Symmetrize both before any spectral query: the divided-difference assembly
    // and the contracted-trace completion are each symmetric up to rounding, and
    // `eigh` reads only one triangle, so a tiny asymmetry must not leak in.
    let sym = |m: &Array2<f64>| -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            for j in 0..p {
                out[[i, j]] = 0.5 * (m[[i, j]] + m[[j, i]]);
            }
        }
        out
    };
    let hphi_sym = sym(hphi);
    let combined = sym(&(hphi + completion));
    let Ok((hphi_evals, _)) = hphi_sym.eigh(Side::Lower) else {
        // Cannot certify PSD ⇒ refuse the completion (keep the bounded base).
        return false;
    };
    let Ok((combined_evals, _)) = combined.eigh(Side::Lower) else {
        return false;
    };
    let hphi_lambda_max = hphi_evals.iter().copied().fold(0.0_f64, f64::max);
    if !hphi_lambda_max.is_finite() || hphi_lambda_max <= 0.0 {
        // Degenerate / zero base curvature: any completion is untrustworthy here.
        return false;
    }
    let combined_lambda_min = combined_evals.iter().copied().fold(f64::INFINITY, f64::min);
    if !combined_lambda_min.is_finite() {
        return false;
    }
    combined_lambda_min >= -JEFFREYS_COMPLETION_PSD_REL_TOL * hphi_lambda_max
}

#[inline]
pub(crate) fn custom_family_jeffreys_cap(floor: f64) -> f64 {
    JEFFREYS_CONDITIONING_GATE_ABSOLUTE_CLEAR.max(floor)
}

#[inline]
pub(crate) fn custom_family_jeffreys_floored_inverse(lam: f64, floor: f64) -> f64 {
    let cap = custom_family_jeffreys_cap(floor);
    if lam >= cap {
        cap / (lam * lam)
    } else if lam >= floor {
        1.0 / lam
    } else if lam >= 0.0 {
        1.0 / floor
    } else {
        let denom = floor - lam;
        floor / (denom * denom)
    }
}

#[inline]
pub(crate) fn custom_family_jeffreys_conditioning_gate_weight(
    lambda_min: f64,
    lambda_max: f64,
) -> f64 {
    if lambda_max <= 0.0 || !lambda_min.is_finite() {
        return 1.0;
    }
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
        JEFFREYS_CONDITIONING_GATE_ABSOLUTE,
        JEFFREYS_CONDITIONING_GATE_ABSOLUTE_CLEAR,
    );
    let ratio = (lambda_min / lambda_max).max(f64::MIN_POSITIVE);
    let w_rel = ramp_down(
        ratio.log10(),
        JEFFREYS_CONDITIONING_GATE_RELATIVE.log10(),
        JEFFREYS_CONDITIONING_GATE_RELATIVE_CLEAR.log10(),
    );
    w_abs.max(w_rel)
}

pub(crate) fn custom_family_joint_jeffreys_contract_weight(
    h_joint: ndarray::ArrayView2<'_, f64>,
    z_joint: ndarray::ArrayView2<'_, f64>,
) -> Result<Option<(f64, Array2<f64>)>, String> {
    let p = h_joint.nrows();
    if h_joint.ncols() != p {
        return Err(format!(
            "custom_family_joint_jeffreys_contract_weight: H must be square, got {}x{}",
            h_joint.nrows(),
            h_joint.ncols()
        ));
    }
    if z_joint.nrows() != p {
        return Err(format!(
            "custom_family_joint_jeffreys_contract_weight: Z_J has {} rows, expected {p}",
            z_joint.nrows()
        ));
    }
    let m = z_joint.ncols();
    if m == 0 {
        return Ok(None);
    }

    let hz = h_joint.dot(&z_joint);
    let h_id = z_joint.t().dot(&hz);
    let mut h_id_sym = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        for j in 0..m {
            h_id_sym[[i, j]] = 0.5 * (h_id[[i, j]] + h_id[[j, i]]);
        }
    }
    let (evals, evecs) = h_id_sym.eigh(Side::Lower).map_err(|e| {
        format!(
            "custom_family_joint_jeffreys_contract_weight: reduced-information eigendecomposition failed: {e}"
        )
    })?;
    let lambda_max = evals.iter().copied().fold(0.0_f64, f64::max);
    let lambda_min = evals.iter().copied().fold(f64::INFINITY, f64::min);
    let gate_weight = custom_family_jeffreys_conditioning_gate_weight(lambda_min, lambda_max);
    if gate_weight == 0.0 {
        return Ok(None);
    }
    let floor = (JEFFREYS_REDUCED_INFO_RELATIVE_FLOOR * lambda_max)
        .max(JEFFREYS_REDUCED_INFO_ABSOLUTE_FLOOR);
    let mut k_reduced = Array2::<f64>::zeros((m, m));
    for eig in 0..m {
        let weight = custom_family_jeffreys_floored_inverse(evals[eig], floor);
        if weight == 0.0 {
            continue;
        }
        for row in 0..m {
            let wr = weight * evecs[[row, eig]];
            for col in 0..m {
                k_reduced[[row, col]] += wr * evecs[[col, eig]];
            }
        }
    }
    let weight_full = z_joint.dot(&k_reduced).dot(&z_joint.t());
    Ok(Some((gate_weight, weight_full)))
}

pub(crate) fn custom_family_joint_jeffreys_second_order_completion<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    h_joint: &Array2<f64>,
    z_joint: &Array2<f64>,
    allow_pairwise_fallback: bool,
) -> Result<Option<Array2<f64>>, String> {
    let p = h_joint.nrows();
    let Some((gate_weight, trace_weight)) =
        custom_family_joint_jeffreys_contract_weight(h_joint.view(), z_joint.view())?
    else {
        return if allow_pairwise_fallback {
            Ok(Some(Array2::zeros((p, p))))
        } else {
            Ok(None)
        };
    };
    match family.joint_jeffreys_information_contracted_trace_hessian_with_specs(
        states,
        specs,
        &trace_weight,
    )? {
        Some(mut contracted) => {
            if contracted.dim() != (p, p) {
                return Err(format!(
                    "custom_family_joint_jeffreys_second_order_completion: contracted shape {:?} != ({p}, {p})",
                    contracted.dim()
                ));
            }
            contracted.mapv_inplace(|value| -0.5 * gate_weight * value);
            Ok(Some(contracted))
        }
        None if allow_pairwise_fallback => {
            gam_solve::estimate::reml::jeffreys_subspace::joint_jeffreys_second_order_completion(
                h_joint.view(),
                z_joint.view(),
                |u: &Array1<f64>, v: &Array1<f64>| {
                    family.joint_jeffreys_information_second_directional_derivative_with_specs(
                        states, specs, u, v,
                    )
                },
            )
        }
        None => Ok(None),
    }
}

/// Outer-REML full-span Jeffreys curvature `H_Φ` for the coupled joint Hessian.
/// Returns `None` when there is no coefficient system or the family exposes no
/// exact joint Hessian.
///
/// This is the OUTER-path companion to the inner-Newton wiring: the LAML score
/// uses `log|H + S_λ + H_Φ|` and its analytic ρ-derivatives
/// `tr((H+S_λ+H_Φ)⁻¹ ∂_ρ(H+S_λ+H_Φ))`.
///
/// CORRECTNESS NOTE (was a bug — see `custom_family_outer_jeffreys_hphi_drift_batched`).
/// `H_Φ` has no EXPLICIT ρ-dependence, but it DOES depend on ρ implicitly through
/// the mode β̂(ρ): `H_Φ = H_Φ(β̂(ρ))` because it is built from `H_id = Z_Jᵀ H Z_J`
/// and `D_a = Z_Jᵀ ∂_a H Z_J`, both functions of β̂. So the exact outer gradient
/// of `½ log|H+S_λ+H_Φ|` carries a `½ tr[(·)⁻¹ D_β H_Φ[v_k]]` drift term ALONGSIDE
/// the likelihood drift `D_β H[v_k]`. Folding `H_Φ` into the `HessianFactorization`
/// (the `(·)⁻¹` kernel and `logdet()`) is necessary but NOT sufficient: the
/// trace contraction must ALSO include `D_β H_Φ[v_k]`, supplied by the companion
/// drift wrapper. Without it the analytic gradient describes a DIFFERENT objective
/// than the value, breaking the line search / KKT certification exactly in the
/// near-separating regime where the Jeffreys term is active.
pub(crate) fn custom_family_outer_jeffreys_hphi<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    ranges: &[(usize, usize)],
) -> Result<Option<(f64, Array2<f64>, Option<Array2<f64>>)>, String> {
    if !family.joint_jeffreys_term_required() {
        return Ok(None);
    }
    let z_joint = match build_joint_jeffreys_subspace(specs, ranges)? {
        Some(z) => z,
        None => return Ok(None),
    };
    // Return the gated VALUE alongside the curvature: the outer LAML must fold
    // `−Φ(β̂)` into its cost (the inner mode is Φ-augmented-stationary, so the
    // envelope identity only holds for the Φ-folded criterion — gam#979), and
    // value/curvature must come from the SAME term evaluation.
    let phi_and_hphi = custom_family_joint_jeffreys_term(family, states, specs, ranges, &z_joint)?
        .map(|(phi, _grad, hphi)| (phi, hphi));
    let Some((phi, hphi)) = phi_and_hphi else {
        return Ok(None);
    };
    // SECOND-ORDER COMPLETION AT THE MODE (gam#979), returned SEPARATELY. The
    // divided-difference `H_Φ` omits the second-directional-Hessian remainder
    // `½ tr(K·D_ab)`, so the TRUE Hessian of the Φ-augmented inner objective
    // is `M_true = H + S_λ + H_Φ + completion`. The chain rule fixes where
    // each belongs in the outer gradient of `V = f(β̂) + ½log|M_DD|₊ − ½log|S|₊`:
    //   * the logdet VALUE and its trace kernel must share ONE object
    //     (`M_DD = H + S_λ + H_Φ`), whose drift `D_β H_Φ[v]` the wrapper
    //     supplies exactly — folding the completion THERE would desync value
    //     from drift (the completion's own β-motion needs third directional
    //     derivatives no family exposes; measured: ~38% gradient / ~70%
    //     Hessian FD bias when tried);
    //   * the mode response `v_k = ∂β̂/∂ρ_k = −(∇²f)⁻¹ Ṡ_k β̂` must be solved
    //     on `M_true` — it is a property of the inner stationarity system,
    //     not of the criterion (measured: ~10% uniform FD bias when solved
    //     on `M_DD`).
    // Callers therefore fold this term into the mode-response OPERATOR only.
    // The contracted trace hook may supply it in one family pass. The generic
    // pairwise `p(p+1)/2` fallback is intentionally not selected here: in
    // production large-n fits a "small" p still means hundreds of row-streamed
    // second-directional Hessian passes. `None` degrades to the
    // divided-difference solve, preserving the value/gradient contract.
    let total_p = ranges.last().map(|(_, e)| *e).unwrap_or(0);
    let mut completion: Option<Array2<f64>> = None;
    let completion_requested =
        family.joint_jeffreys_information_contracted_trace_hessian_available();
    if completion_requested
        && let Some(h_joint) = family.joint_jeffreys_information_with_specs(states, specs)?
        && h_joint.nrows() == total_p
        && h_joint.ncols() == total_p
    {
        completion = custom_family_joint_jeffreys_second_order_completion(
            family, states, specs, &h_joint, &z_joint, false,
        )?;
    }
    Ok(Some((phi, hphi, completion)))
}

pub(crate) fn batched_outer_gradient_contract_allows_override(
    robust_jeffreys_hphi: Option<&Array2<f64>>,
) -> bool {
    match robust_jeffreys_hphi {
        None => true,
        Some(hphi) => hphi.iter().all(|value| *value == 0.0),
    }
}

/// Build the Tier-B Jeffreys-curvature drift over ALL `k` mode-response
/// directions of one outer gradient eval, preparing the β-fixed `H_Φ` drift base
/// ONCE ([`JeffreysHphiDriftBase`]) and reusing it across every direction.
///
/// The base's `p` first-directional-derivative row-streams `Hdot[e_a]` (the
/// dominant `O(n·p)` cost) and the reduced-information eigendecomposition are
/// β-fixed across the eval, so they are computed once instead of `k` times. Each
/// direction then pays only its own `Hdot[δ]` (one row-stream) and `p`
/// second-directional `H²dot[δ,e_a]` row-streams. Per-direction output is
/// byte-identical to the per-direction divided-difference drift
/// (`gam_solve::estimate::reml::jeffreys_subspace`'s test-only
/// `joint_jeffreys_hphi_directional_derivative` oracle),
/// which the outer LAML gradient folds via `JeffreysHphiAwareJointDerivatives`.
///
/// Returns `None` exactly when there is no coefficient system, the family exposes
/// no exact joint Hessian, or the term is not required (clean / gated fit) — the
/// same condition as `custom_family_outer_jeffreys_hphi`.
pub(crate) fn custom_family_outer_jeffreys_hphi_drift_batched<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    ranges: &[(usize, usize)],
) -> Result<Option<JeffreysHphiDriftBatchFn>, String> {
    if !family.joint_jeffreys_term_required() {
        return Ok(None);
    }
    let z_joint = match build_joint_jeffreys_subspace(specs, ranges)? {
        Some(z) => z,
        None => return Ok(None),
    };
    let total_p = ranges.last().map(|(_, e)| *e).unwrap_or(0);
    if total_p == 0 || z_joint.ncols() == 0 {
        return Ok(None);
    }
    let h_joint = match family.joint_jeffreys_information_with_specs(states, specs)? {
        Some(h) => h,
        None => return Ok(None),
    };
    if h_joint.nrows() != total_p || h_joint.ncols() != total_p {
        return Ok(None);
    }
    let family_owned = family.clone();
    let states_owned: Vec<ParameterBlockState> = states.to_vec();
    let specs_owned: Vec<ParameterBlockSpec> = specs.to_vec();
    let z_columns = z_joint.clone();
    let batch: JeffreysHphiDriftBatchFn = Arc::new(move |deltas: &[Array1<f64>]| {
        // Prepare the β-fixed base ONCE: the reduced-information eigendecomposition
        // plus the `p` first directional derivatives `Hdot[e_a]` (the dominant
        // cost). Acquire the WHOLE canonical-axis set in ONE batched hook call —
        // the same path the value-path `joint_jeffreys_term` uses — so a family
        // that assembles every axis in one shared softmax/Gram pass (multinomial)
        // pays a SINGLE sweep instead of the `p` concurrent cache-miss sweeps the
        // per-axis fan-out triggered on a fresh β (#1082/#979). `None` batch ⇒ some
        // axis lacks the exact derivative ⇒ fall back to the per-axis closure,
        // whose first `None` collapses the base to the zero drift everywhere
        // (matching the singular hook).
        let all_axes = family_owned
            .joint_jeffreys_information_directional_derivative_all_axes_with_specs(
                &states_owned,
                &specs_owned,
            )?;
        let base = match all_axes {
            Some(hdots) => {
                gam_solve::estimate::reml::jeffreys_subspace::JeffreysHphiDriftBase::prepare_with_axes(
                    h_joint.view(),
                    z_columns.view(),
                    hdots,
                )?
            }
            None => gam_solve::estimate::reml::jeffreys_subspace::JeffreysHphiDriftBase::prepare(
                h_joint.view(),
                z_columns.view(),
                |direction: &Array1<f64>| {
                    family_owned.joint_jeffreys_information_directional_derivative_with_specs(
                        &states_owned,
                        &specs_owned,
                        direction,
                    )
                },
            )?,
        };
        let Some(base) = base else {
            let zeros = vec![Some(Array2::<f64>::zeros((total_p, total_p))); deltas.len()];
            return Ok(zeros);
        };
        // Per direction: the only δ-dependent work — `pert_h = Hdot[δ]` and the
        // `p` second-directional derivatives `H²dot[δ,e_a]` — reusing the base.
        deltas
            .iter()
            .map(|delta| {
                let pert_h = match family_owned
                    .joint_jeffreys_information_directional_derivative_with_specs(
                        &states_owned,
                        &specs_owned,
                        delta,
                    )? {
                    Some(hd) => hd,
                    // No exact first derivative ⇒ drift undefined ⇒ safe zero
                    // (matching `joint_jeffreys_hphi_directional_derivative`).
                    None => return Ok(Some(Array2::<f64>::zeros((total_p, total_p)))),
                };
                // Batched all-axes second-directional object `{H²dot[δ,e_a]}` in
                // ONE pass (BLAS-3 for the rigid family; per-axis fallback for the
                // rest). This collapses the dominant `p` independent full-data
                // second-directional sweeps the per-axis closure used to run.
                let pert_axis_matrices = family_owned
                    .joint_jeffreys_information_second_directional_all_axes_with_specs(
                        &states_owned,
                        &specs_owned,
                        delta,
                    )?;
                base.perturbation_derivative_batched_axes(&pert_h, pert_axis_matrices)
                    .map(Some)
            })
            .collect()
    });
    Ok(Some(batch))
}
