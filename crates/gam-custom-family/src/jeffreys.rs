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

/// Build the joint Jeffreys/Firth basis `Z_J` for the universal robustness
/// term — the directions the term is allowed to act on.
///
/// The term exists to supply the `O(1)`-bounding curvature a near-separating
/// direction has none of, so the span it acts on is the set of directions the
/// model does not ALREADY bound. A smoothing penalty bounds exactly
/// `range(S)`, because `(H + S_λ)v = Hv + λSv`; on `range(S)` the model already
/// carries a proper prior and a second one is not a bias correction there but a
/// duplicate. So `Z_J = ker(S)`.
///
/// Two routes reach that kernel, and they agree wherever both apply:
///
/// * a family whose smoothing rides on a JOINT penalty bundle (gam#1587) leaves
///   every per-block `penalties` list empty, so it states the aggregate whose
///   kernel is the span through
///   [`CustomFamily::jeffreys_span_aggregate_penalty`] (gam#2612);
/// * otherwise each block contributes its own per-block span and the bases are
///   embedded block-diagonally into the joint `total_p x m_total` matrix.
///
/// A family with NO penalized component at all therefore keeps the full
/// identifiable span, which is the `S_λ = 0` case of the same statement.
/// Returns `None` for an empty system, and for a system every smoothing
/// parameter reaches (nothing is left for the term to bound).
///
/// The Jeffreys conditioning gate then decides whether this basis contributes
/// at the current iterate.
pub(crate) fn build_joint_jeffreys_subspace<F: CustomFamily + ?Sized>(
    family: &F,
    specs: &[ParameterBlockSpec],
    ranges: &[(usize, usize)],
) -> Result<Option<Array2<f64>>, CustomFamilyError> {
    let total_p = ranges.last().map(|(_, e)| *e).unwrap_or(0);
    if total_p == 0 {
        return Ok(None);
    }
    // gam#2612: a family that has MEASURED the directions it fails to bound
    // states them directly, and that statement wins over anything derived from a
    // penalty's kernel — see `CustomFamily::jeffreys_span_basis` for why a kernel
    // is the wrong object to derive the span from, and for the constancy contract
    // the family is promising by returning `Some` here.
    if let Some(basis) = family
        .jeffreys_span_basis()
        .map_err(|reason| CustomFamilyError::DimensionMismatch { reason })?
    {
        if basis.nrows() != total_p {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "jeffreys span basis has {} rows, expected the raw joint width {total_p}",
                    basis.nrows()
                ),
            });
        }
        if basis.ncols() == 0 {
            // The model bounds every direction at the smoothing it selected, so
            // there is nothing for this term to bound — the same exit the
            // aggregate route takes when every direction is penalised.
            return Ok(None);
        }
        return Ok(Some(basis));
    }
    // gam#2612: a family whose smoothing rides on a JOINT penalty bundle
    // (gam#1587) leaves every per-block `penalties` list empty, so the
    // block-diagonal assembly below sees no penalty and hands back the full
    // identifiable span — and the term then acts on every direction the joint
    // penalty already bounds. `jeffreys_span_aggregate_penalty` is that family's
    // own statement of which directions its penalty reaches. It is λ-free (any
    // positive combination of the joint specs has the same kernel), hence
    // constant in `β` and `ρ`, so every `Φ` derivative formula below and
    // downstream is unchanged in form: they all differentiate through `H` with
    // `Z_J` held fixed.
    if let Some(aggregate) = family
        .jeffreys_span_aggregate_penalty()
        .map_err(|reason| CustomFamilyError::DimensionMismatch { reason })?
    {
        if aggregate.dim() != (total_p, total_p) {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "jeffreys span aggregate penalty is {:?}, expected ({total_p}, {total_p})",
                    aggregate.dim()
                ),
            });
        }
        let subspace =
            gam_solve::estimate::reml::jeffreys_subspace::jeffreys_subspace_from_penalty(
                aggregate.view(),
            )?;
        if subspace.span_dim() == 0 {
            // Every direction is reached by some smoothing parameter, so the
            // model already carries a proper prior everywhere and there is
            // nothing for this term to bound.
            return Ok(None);
        }
        return Ok(Some(subspace.columns));
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
) -> Result<bool, CustomFamilyError> {
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
        )
        .map_err(CustomFamilyError::trial_point);
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
    // Display boundary (gam#2689): `jeffreys_term_skippable_via_matvec` is a
    // gam-solve entry point declared over `Result<_, String>`, so the typed
    // operator error is rendered here rather than by a silent blanket `From`.
    let hv = |v: &Array1<f64>| -> Result<Array1<f64>, String> {
        match source {
            JointHessianSource::Dense(matrix) => Ok(matrix.dot(v)),
            JointHessianSource::Operator { apply, .. } => {
                apply(v).map_err(|error| error.to_string())
            }
        }
    };
    gam_solve::estimate::reml::jeffreys_subspace::jeffreys_term_skippable_via_matvec(hv, total_p)
        .map_err(CustomFamilyError::trial_point)
}

/// The Jeffreys objective contribution at one working point together with the
/// rounding one evaluation of it carries (gam#2718). The trust-region ratio
/// test compares two of these; the second field is what lets the solve's
/// objective-resolution ceiling admit a discrepancy the log-determinant's own
/// arithmetic produced instead of refusing it as model error.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct JointJeffreysValue {
    /// `strength · Φ`, the value folded into the inner objective.
    pub(crate) phi: f64,
    /// `strength · ` the plan's `value_roundoff_bound`.
    pub(crate) roundoff: f64,
}

/// Evaluate ONLY the Jeffreys objective value `Phi = 1/2 log|Z_J^T H Z_J|` at
/// the current working point, with its round-off bound. Cheaper than the full
/// term (no directional derivatives), used to keep the trust-region
/// accept/reject objective consistent with the Jeffreys-modified Newton step.
/// Returns a zero value when there is no coefficient system or the family
/// exposes no exact joint Hessian (the term is inapplicable, exactly as
/// [`custom_family_joint_jeffreys_term`] reports `None`).
///
/// A point where the family CANNOT form its information (an infeasible trial
/// point — on the survival marginal-slope families a row whose transformed
/// time derivative is not positive — or a reduced spectrum the eigensolver
/// refuses) is an error, not a zero: the objective `−ℓ + ½βᵀSβ − Φ` does not
/// exist there. Returning `0.0` instead evaluated a DIFFERENT objective at that
/// trial point, off from the incumbent's by `|Φ|`, and the #2765 replays show
/// that arm firing thousands of times per solve. The trial loop routes this
/// error through the same refusal a likelihood failure takes.
pub(crate) fn custom_family_joint_jeffreys_value<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    ranges: &[(usize, usize)],
    z_joint: &Array2<f64>,
) -> Result<JointJeffreysValue, CustomFamilyError> {
    let total_p = ranges.last().map(|(_, e)| *e).unwrap_or(0);
    if total_p == 0 || z_joint.ncols() == 0 {
        return Ok(JointJeffreysValue::default());
    }
    let h_joint = match family.joint_jeffreys_information_with_specs(states, specs)? {
        None => return Ok(JointJeffreysValue::default()),
        Some(h) if h.nrows() == total_p && h.ncols() == total_p => h,
        Some(h) => {
            return Err(CustomFamilyError::trial_point(format!(
                "joint Jeffreys information has shape {:?}, expected ({total_p},{total_p})",
                h.dim()
            )));
        }
    };
    let plan = gam_solve::estimate::reml::jeffreys_subspace::JointJeffreysPlan::prepare(
        h_joint.view(),
        z_joint.view(),
    )
    .map_err(|error| {
        CustomFamilyError::trial_point(format!("Jeffreys value unavailable at this point: {error}"))
    })?;
    let strength = family.joint_jeffreys_term_strength();
    Ok(JointJeffreysValue {
        phi: plan.value() * strength,
        roundoff: plan.value_roundoff_bound() * strength.abs(),
    })
}

fn scale_jeffreys_triple(
    mut term: (f64, Array1<f64>, Array2<f64>),
    strength: f64,
) -> (f64, Array1<f64>, Array2<f64>) {
    if strength == 1.0 {
        return term;
    }
    term.0 *= strength;
    term.1 *= strength;
    term.2 *= strength;
    term
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
) -> Result<Option<(f64, Array1<f64>, Array2<f64>)>, CustomFamilyError> {
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
    custom_family_joint_jeffreys_term_from_information(family, states, specs, &h_joint, z_joint)
        .map(Some)
}

fn custom_family_joint_jeffreys_term_from_information<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    h_joint: &Array2<f64>,
    z_joint: &Array2<f64>,
) -> Result<(f64, Array1<f64>, Array2<f64>), CustomFamilyError> {
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
    Ok(scale_jeffreys_triple(
        term,
        family.joint_jeffreys_term_strength(),
    ))
}

/// Evaluate the accepted-mode Jeffreys triple directly from the exact Newton
/// workspace that was built at that same beta.
///
/// The generic family route reconstructs an information source and its
/// all-axes derivatives from `(family, states)`. After an accepted fused trial,
/// however, the retained workspace already owns the authoritative current-beta
/// row cache and Hessian. Using it avoids a second cache/Hessian construction
/// and lets row-kernel workspaces dispatch their build-once all-axes hook.
/// `None` is a typed capability result: the caller may use the family route
/// only when the concrete workspace does not expose a batched derivative.
pub(crate) fn custom_family_joint_jeffreys_term_from_workspace(
    workspace: &dyn ExactNewtonJointHessianWorkspace,
    total_p: usize,
    z_joint: &Array2<f64>,
    strength: f64,
) -> Result<Option<(f64, Array1<f64>, Array2<f64>)>, CustomFamilyError> {
    if total_p == 0 || z_joint.ncols() == 0 {
        return Ok(None);
    }
    let Some(h_joint) = workspace.hessian_dense_forced()? else {
        return Ok(None);
    };
    if h_joint.dim() != (total_p, total_p) {
        return Err(CustomFamilyError::trial_point(format!(
            "accepted workspace Jeffreys Hessian shape {:?}, expected ({total_p}, {total_p})",
            h_joint.dim(),
        )));
    }
    let Some(directional_derivatives) = workspace.directional_derivative_all_axes()? else {
        return Ok(None);
    };
    if directional_derivatives.len() != total_p {
        return Err(CustomFamilyError::trial_point(format!(
            "accepted workspace Jeffreys derivative count {}, expected {total_p}",
            directional_derivatives.len(),
        )));
    }
    gam_solve::estimate::reml::jeffreys_subspace::joint_jeffreys_term_batched(
        h_joint.view(),
        z_joint.view(),
        || Ok(Some(directional_derivatives)),
    )
    .map_err(CustomFamilyError::trial_point)
    .map(|term| Some(scale_jeffreys_triple(term, strength)))
}

/// Evaluate the Jeffreys term and the exact remainder of its coefficient
/// Hessian from one information-matrix snapshot.
///
/// The divided-difference matrix returned by
/// [`custom_family_joint_jeffreys_term`] is only the first-derivative part of
/// `-∇²Φ`. The true inner-objective precision also contains
/// `-½ tr(K H''[e_a,e_b])`. Consumers that invert the coefficient objective
/// (terminal covariance and returned-mode certification) must use both pieces;
/// omitting the completion silently widens the posterior. Keeping this
/// operation here ensures the term and completion share the same information
/// matrix even for a stateful family.
pub(crate) fn custom_family_joint_jeffreys_term_with_exact_completion<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    ranges: &[(usize, usize)],
    z_joint: &Array2<f64>,
) -> Result<Option<(f64, Array1<f64>, Array2<f64>, Array2<f64>)>, CustomFamilyError> {
    let total_p = ranges.last().map(|(_, end)| *end).unwrap_or(0);
    if total_p == 0 || z_joint.ncols() == 0 {
        return Ok(None);
    }
    let Some(h_joint) = family.joint_jeffreys_information_with_specs(states, specs)? else {
        return Ok(None);
    };
    if h_joint.dim() != (total_p, total_p) {
        return Ok(None);
    }
    let (phi, gradient, hphi) = custom_family_joint_jeffreys_term_from_information(
        family, states, specs, &h_joint, z_joint,
    )?;
    let completion = custom_family_joint_jeffreys_second_order_completion(
        family,
        states,
        specs,
        &h_joint,
        z_joint,
        JeffreysCompletionAssembly::Exact,
    )?
    .ok_or_else(|| {
        "active Jeffreys term did not supply its exact second-order completion".to_string()
    })?;
    if completion.dim() != (total_p, total_p) || completion.iter().any(|value| !value.is_finite()) {
        return Err(CustomFamilyError::trial_point(format!(
            "exact Jeffreys completion is non-finite or has shape {:?}, expected ({total_p}, {total_p})",
            completion.dim(),
        )));
    }
    Ok(Some((phi, gradient, hphi, completion)))
}

pub(crate) const JEFFREYS_REDUCED_INFO_RELATIVE_FLOOR: f64 = 1e-10;

pub(crate) const JEFFREYS_REDUCED_INFO_ABSOLUTE_FLOOR: f64 = 1e-12;

pub(crate) const JEFFREYS_CONDITIONING_GATE_RELATIVE: f64 = 1e-8;

pub(crate) const JEFFREYS_CONDITIONING_GATE_ABSOLUTE: f64 = 1.0;

pub(crate) const JEFFREYS_CONDITIONING_GATE_ABSOLUTE_CLEAR: f64 = 16.0;

pub(crate) const JEFFREYS_CONDITIONING_GATE_RELATIVE_CLEAR: f64 = 1e-6;

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
) -> Result<Option<(f64, Array2<f64>)>, CustomFamilyError> {
    let p = h_joint.nrows();
    if h_joint.ncols() != p {
        return Err(CustomFamilyError::trial_point(format!(
            "custom_family_joint_jeffreys_contract_weight: H must be square, got {}x{}",
            h_joint.nrows(),
            h_joint.ncols()
        )));
    }
    if z_joint.nrows() != p {
        return Err(CustomFamilyError::trial_point(format!(
            "custom_family_joint_jeffreys_contract_weight: Z_J has {} rows, expected {p}",
            z_joint.nrows()
        )));
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum JeffreysCompletionAssembly {
    /// Use only the family's fused contracted-trace implementation. This is the
    /// outer-profile route, where the completion is an optional response
    /// acceleration and a pairwise row stream would change the cost class.
    Contracted,
    /// Produce the exact objective Hessian. If the family has no fused
    /// contracted implementation, assemble the mathematically identical
    /// pairwise second-directional form. Returned-mode certification and the
    /// inner Newton endgame use this authority and may not omit a term.
    Exact,
}

pub(crate) fn custom_family_joint_jeffreys_second_order_completion<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    h_joint: &Array2<f64>,
    z_joint: &Array2<f64>,
    assembly: JeffreysCompletionAssembly,
) -> Result<Option<Array2<f64>>, CustomFamilyError> {
    let p = h_joint.nrows();
    let Some((gate_weight, trace_weight)) =
        custom_family_joint_jeffreys_contract_weight(h_joint.view(), z_joint.view())?
    else {
        return if assembly == JeffreysCompletionAssembly::Exact {
            Ok(Some(Array2::zeros((p, p))))
        } else {
            Ok(None)
        };
    };
    let completion = match family.joint_jeffreys_information_contracted_trace_hessian_with_specs(
        states,
        specs,
        &trace_weight,
    )? {
        Some(mut contracted) => {
            if contracted.dim() != (p, p) {
                return Err(CustomFamilyError::trial_point(format!(
                    "custom_family_joint_jeffreys_second_order_completion: contracted shape {:?} != ({p}, {p})",
                    contracted.dim()
                )));
            }
            contracted.mapv_inplace(|value| -0.5 * gate_weight * value);
            Some(contracted)
        }
        None if assembly == JeffreysCompletionAssembly::Exact => {
            gam_solve::estimate::reml::jeffreys_subspace::joint_jeffreys_second_order_completion(
                h_joint.view(),
                z_joint.view(),
                |u: &Array1<f64>, v: &Array1<f64>| {
                    family.joint_jeffreys_information_second_directional_derivative_with_specs(
                        states, specs, u, v,
                    )
                },
            )?
        }
        None => None,
    };
    Ok(completion.map(|mut matrix| {
        let strength = family.joint_jeffreys_term_strength();
        if strength != 1.0 {
            matrix *= strength;
        }
        matrix
    }))
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
) -> Result<Option<(f64, Array2<f64>, Option<Array2<f64>>)>, CustomFamilyError> {
    if !family.joint_jeffreys_term_required() {
        return Ok(None);
    }
    let z_joint = match build_joint_jeffreys_subspace(family, specs, ranges)? {
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
    // pairwise `p(p+1)/2` assembly is intentionally not selected here: in
    // production large-n fits a "small" p still means hundreds of row-streamed
    // second-directional Hessian passes. `None` degrades to the
    // divided-difference solve, preserving the value/gradient contract.
    let total_p = ranges.last().map(|(_, e)| *e).unwrap_or(0);
    let mut completion: Option<Array2<f64>> = None;
    // Objective geometry is independent of the requested derivative order.
    // The one-step profile correction consumes the mode-response operator, and
    // whether that correction is numerically visible is governed by the solved
    // displacement H⁻¹r—not by a raw-residual threshold known at this layer.
    // Always form the available true-Hessian completion for an active Jeffreys
    // profile so value screening, finite differences, and analytic derivatives
    // price one canonical objective (#2460).
    let completion_requested =
        family.joint_jeffreys_information_contracted_trace_hessian_available();
    if completion_requested
        && let Some(h_joint) = family.joint_jeffreys_information_with_specs(states, specs)?
        && h_joint.nrows() == total_p
        && h_joint.ncols() == total_p
    {
        completion = custom_family_joint_jeffreys_second_order_completion(
            family,
            states,
            specs,
            &h_joint,
            &z_joint,
            JeffreysCompletionAssembly::Contracted,
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
) -> Result<Option<JeffreysHphiDriftBatchFn>, CustomFamilyError> {
    // Install one canonical lazy drift provider for every active Jeffreys
    // profile. Value-only evaluation can consume it through the moving-Hessian
    // KKT correction just as derivative evaluation consumes it through trace
    // derivatives. The expensive row-streamed base remains lazy inside the
    // returned closure, so an evaluation that never requests a correction does
    // not perform that work.
    if !family.joint_jeffreys_term_required() {
        return Ok(None);
    }
    let z_joint = match build_joint_jeffreys_subspace(family, specs, ranges)? {
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
    let plan = gam_solve::estimate::reml::jeffreys_subspace::JointJeffreysPlan::prepare(
        h_joint.view(),
        z_joint.view(),
    )?;
    if !plan.is_active() {
        return Ok(None);
    }
    let family_owned = family.clone();
    let strength = family.joint_jeffreys_term_strength();
    let states_owned: Vec<ParameterBlockState> = states.to_vec();
    let specs_owned: Vec<ParameterBlockSpec> = specs.to_vec();
    let batch: JeffreysHphiDriftBatchFn = Arc::new(move |deltas: &[Array1<f64>]| {
        // The exact reduced-information plan authorized this closure before it
        // was returned, so an inactive outer gate performs zero derivative work.
        // Acquire the WHOLE canonical-axis set in ONE batched hook call —
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
                gam_solve::estimate::reml::jeffreys_subspace::JeffreysHphiDriftBase::prepare_with_plan_axes(
                    plan.clone(),
                    hdots,
                )?
            }
            None => gam_solve::estimate::reml::jeffreys_subspace::JeffreysHphiDriftBase::prepare_from_plan(
                plan.clone(),
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
                // ONE pass (BLAS-3 for the rigid family; the defining per-axis
                // implementation for the rest). This collapses the dominant
                // `p` independent full-data second-directional sweeps the
                // per-axis closure used to run.
                let pert_axis_matrices = family_owned
                    .joint_jeffreys_information_second_directional_all_axes_with_specs(
                        &states_owned,
                        &specs_owned,
                        delta,
                    )?;
                base.perturbation_derivative_batched_axes(&pert_h, pert_axis_matrices)
                    .map(|mut derivative| {
                        if strength != 1.0 {
                            derivative *= strength;
                        }
                        Some(derivative)
                    })
            })
            .collect::<Result<Vec<_>, String>>()
            .map_err(CustomFamilyError::trial_point)
    });
    Ok(Some(batch))
}
