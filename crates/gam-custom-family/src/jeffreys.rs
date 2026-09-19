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
/// Three routes, in precedence order:
///
/// * a family that has MEASURED the directions it fails to bound states them
///   directly through [`CustomFamily::jeffreys_span_basis`] (gam#2612);
/// * a family whose smoothing rides on a JOINT penalty bundle (gam#1587) leaves
///   every per-block `penalties` list empty, so it states the aggregate whose
///   kernel is the span through
///   [`CustomFamily::jeffreys_span_aggregate_penalty`] (gam#2612);
/// * otherwise each block contributes its FULL identifiable span, `Z_J = I_p`,
///   and the bases are embedded block-diagonally into the joint
///   `total_p x m_total` matrix.
///
/// The last route is not `ker(S)`. The Jeffreys penalty is self-limiting: its
/// score is `O(1)` against the data's `O(n)` Fisher information, so on a
/// data-identified direction its only effect is the `O(1/n)` Firth correction.
/// It bites only where `I(β)` is near-singular, whether that direction lies in
/// `ker(S)` or in `range(S)`, and a `ker(S)` span could not reach a
/// near-separation along a penalized spline direction (see
/// `jeffreys_subspace_from_penalty` in gam-solve).
///
/// Returns `None` for an empty system, and when a family's measured or aggregate
/// route leaves nothing for the term to bound.
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

/// Conditioning pre-check: can the always-on Jeffreys term be PROVABLY skipped
/// at this working point, before the `custom_family_joint_jeffreys_*` formation?
///
/// On the FULL span (`Z_J = I`) the reduced information is `H_id = H`, so the
/// conditioning gate only needs `H`'s extreme eigenvalues. They are first bounded
/// conservatively from a few Hessian-vector products against the SAME
/// `joint_hessian_source` operator the inner Newton already built (matrix-free on
/// the large-`p` path, dense otherwise). When the bounds clear both gates with a
/// safe margin (see `jeffreys_term_skippable_via_matvec`), the exact gate is
/// CERTAIN to return the zero term, so the caller skips the dense `H`
/// materialization, the `Z_JᵀHZ_J` build, the eigendecomposition, the `∇Φ`/`H_Φ`
/// assembly, and the Q1 outer drift entirely — returning the EXACT-ZERO term,
/// byte-identical to the gated-off dense path.
///
/// When the bounds do not certify, `H` is formed once (a clone for the dense
/// source, `total_p` products for the operator) and its exact spectrum decides,
/// so a well-conditioned cycle the bounds' margin refuses still skips the
/// formation and all-axes sweep a non-skipped cycle pays (#1389). No joint
/// dimension selects between the two (#2900): a `false` from both flows to the
/// exact formation, which forms and eigendecomposes the same information anyway.
///
/// Matrix-free preservation: the bounds issue only `O(p·k)` (`k≤12`) matvecs
/// through `source` and form nothing dense at `p`-scale; on a well-conditioned
/// large-`p` matrix-free fit (the common case) they certify and NOTHING dense is
/// ever built.
pub(crate) fn jeffreys_term_skippable_for_source(
    source: &JointHessianSource,
    total_p: usize,
) -> Result<bool, CustomFamilyError> {
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
    // Display boundary (gam#2689): the gam-solve pre-check is declared over
    // `Result<_, String>`, so the typed operator error is rendered here rather
    // than by a silent blanket `From`.
    let hv = |v: &Array1<f64>| -> Result<Array1<f64>, String> {
        match source {
            JointHessianSource::Dense(matrix) => Ok(matrix.dot(v)),
            JointHessianSource::Operator { apply, .. } => {
                apply(v).map_err(|error| error.to_string())
            }
        }
    };
    // The exact spectrum's `H`, built as the dense Newton step builds it: a clone
    // for the dense source, and for the operator the workspace's structural dense
    // build or one batched multi-RHS sweep, never `total_p` single products that
    // each re-walk every row. A formation the dense budget or a non-finite entry
    // refuses cannot certify, so the exact term runs.
    let dense = || -> Result<Option<Array2<f64>>, String> {
        crate::joint_newton::materialize_joint_hessian_source(
            source,
            total_p,
            "Jeffreys exact conditioning pre-check",
        )
        .map(Some)
        .map_err(|error| error.to_string())
    };
    gam_solve::estimate::reml::jeffreys_subspace::jeffreys_term_skippable(hv, total_p, dense)
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
    // are prepared before either lazy provider can run.  A gated-off term therefore
    // performs ZERO all-axes builds.  When active, a family that forms the rotated
    // rows supplies them and no `p × p` axis matrix is built (#1082); otherwise the
    // dense provider is called once and returns the canonical `{Hdot[e_a]}` batch.
    let term = gam_solve::estimate::reml::jeffreys_subspace::joint_jeffreys_term_batched_rotated(
        h_joint.view(),
        z_joint.view(),
        |basis| {
            family
                .jeffreys_rotated_first_derivative()
                .map(|rotated| rotated.first_directional_rotated_all_axes(states, specs, basis))
                .transpose()
        },
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum JeffreysCompletionAssembly {
    /// Use only the family's fused contracted-trace implementation. This is the
    /// outer-profile route, where the completion is an optional response
    /// acceleration and a pairwise row stream would change the cost class.
    Contracted,
    /// Produce the exact objective Hessian. If the family has no fused
    /// contracted implementation, assemble the mathematically identical
    /// second-directional form: one pass per span direction when the information
    /// is the observed Hessian, one per coefficient pair otherwise. Returned-mode
    /// certification and the inner Newton endgame use this authority and may not
    /// omit a term.
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
    // The one reduced-information spectrum (gam-solve's plan) gives the gate, the trace
    // weight `Z_J K Z_Jᵀ` and the gate/floor motion below, so the completion gates and
    // floors exactly as the value does.
    let plan = gam_solve::estimate::reml::jeffreys_subspace::JointJeffreysPlan::prepare(
        h_joint.view(),
        z_joint.view(),
    )?;
    if !plan.is_active() {
        return if assembly == JeffreysCompletionAssembly::Exact {
            Ok(Some(Array2::zeros((p, p))))
        } else {
            Ok(None)
        };
    }
    let gate_weight = plan.conditioning_gate_weight();
    let trace_weight = plan.contracted_trace_weight();
    // Gate and floor motion (gam#1082). `−½·G·⟨Z_J K Z_Jᵀ, H''⟩` is the whole
    // second-directional remainder of `−∇²Φ` only while the conditioning gate is
    // saturated and no eigenvalue feels the relative floor move. Inside the
    // gate's transition band, or under a moving floor, `Φ = G·U` also carries
    // `∇G⊗∇U + ∇U⊗∇G + U·∇²G` and the floor's own motion. The mode response is
    // solved on this completion, so leaving those terms out makes `∂β̂/∂ρ` the
    // response of a different objective. Measured on the quasi-separated
    // multinomial repro (`multinomial_outer_derivatives_1082`): analytic-vs-FD
    // outer gradient gaps of 3e-4 to 1.5e-2, independent of the FD step and of
    // the inner tolerance, while the value-path Jeffreys gradient and the
    // divided-difference drift were both finite-difference exact.
    let motion = if assembly == JeffreysCompletionAssembly::Exact
        || family.joint_jeffreys_information_contracted_trace_hessian_available()
    {
        if plan.hessian_motion_active() {
            // A family that forms the rotated rows hands them over, and no `p × p` axis
            // matrix is built (#1082).
            match family.jeffreys_rotated_first_derivative() {
                Some(rotated) => {
                    let rows = rotated.first_directional_rotated_all_axes(
                        states,
                        specs,
                        plan.ambient_eigenbasis().view(),
                    )?;
                    Some(plan.hessian_motion_from_rotated_rows(&rows)?)
                }
                None => {
                    let axes = family
                        .joint_jeffreys_information_directional_derivative_all_axes_with_specs(
                            states, specs,
                        )?
                        .ok_or_else(|| {
                            CustomFamilyError::trial_point(
                                "active Jeffreys gate/floor motion requires exact first information \
                                 derivatives"
                                    .to_string(),
                            )
                        })?;
                    Some(plan.hessian_motion(&axes)?)
                }
            }
        } else {
            None
        }
    } else {
        None
    };
    // The contracted hook is linear in its weight and the completion keeps the
    // hook's `−½·G` convention, so the motion's H''-linear part rides in the same
    // single pass as `(2/G)·extra_trace_weight`.
    let hook_weight = match motion.as_ref() {
        Some(motion) => {
            let mut weight = trace_weight.clone();
            weight.scaled_add(2.0 / gate_weight, &motion.extra_trace_weight);
            weight
        }
        None => trace_weight,
    };
    let completion = match family.joint_jeffreys_information_contracted_trace_hessian_with_specs(
        states,
        specs,
        &hook_weight,
    )? {
        Some(mut contracted) => {
            if contracted.dim() != (p, p) {
                return Err(CustomFamilyError::trial_point(format!(
                    "custom_family_joint_jeffreys_second_order_completion: contracted shape {:?} != ({p}, {p})",
                    contracted.dim()
                )));
            }
            contracted.mapv_inplace(|value| -0.5 * gate_weight * value);
            if let Some(motion) = motion.as_ref() {
                contracted -= &motion.remainder;
            }
            Some(contracted)
        }
        None if assembly == JeffreysCompletionAssembly::Exact => {
            let second_direction = |u: &Array1<f64>, v: &Array1<f64>| {
                family.joint_jeffreys_information_second_directional_derivative_with_specs(
                    states, specs, u, v,
                )
            };
            if family.joint_jeffreys_information_matches_observed_hessian() {
                // The observed Hessian's fourth derivative is fully symmetric, so one pass per
                // span direction replaces one per coefficient pair (#2893).
                gam_solve::estimate::reml::jeffreys_subspace::joint_jeffreys_observed_hessian_completion(
                    h_joint.view(),
                    z_joint.view(),
                    second_direction,
                    motion.as_ref(),
                )?
            } else {
                match motion.as_ref() {
                    Some(motion) => {
                        gam_solve::estimate::reml::jeffreys_subspace::joint_jeffreys_second_order_completion_with_motion(
                            h_joint.view(),
                            z_joint.view(),
                            second_direction,
                            motion,
                        )?
                    }
                    None => {
                        gam_solve::estimate::reml::jeffreys_subspace::joint_jeffreys_second_order_completion(
                            h_joint.view(),
                            z_joint.view(),
                            second_direction,
                        )?
                    }
                }
            }
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
    // is `M_true = H + S_λ + H_Φ + completion`, and the Laplace normalizer is
    // `½log|M_true|`. The chain rule fixes where each piece belongs:
    //   * the logdet VALUE and its trace kernel must share ONE object, and that
    //     object's β-drift must be supplied exactly. A family exposing
    //     `jeffreys_completion_outer_derivatives` supplies the
    //     completion's β-drifts, so the projected criterion prices `M_true`
    //     (gam#2894). Every other family keeps `M_DD = H + S_λ + H_Φ`, whose drift
    //     `D_β H_Φ[v]` the wrapper supplies (folding the completion without its
    //     drift measured ~38% gradient / ~70% Hessian FD bias);
    //   * the mode response `v_k = ∂β̂/∂ρ_k = −(∇²f)⁻¹ Ṡ_k β̂` is always solved
    //     on `M_true`, since it is a property of the inner stationarity system
    //     (measured: ~10% uniform FD bias when solved on `M_DD`).
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
    let family_owned = Arc::new(family.clone());
    let strength = family.joint_jeffreys_term_strength();
    let states_owned = Arc::new(states.to_vec());
    let specs_owned = Arc::new(specs.to_vec());
    let family_second = Arc::clone(&family_owned);
    let states_second = Arc::clone(&states_owned);
    let specs_second = Arc::clone(&specs_owned);
    let prepare_base = {
        let family = Arc::clone(&family_owned);
        let states = Arc::clone(&states_owned);
        let specs = Arc::clone(&specs_owned);
        let cached = OnceLock::<Result<
            Arc<gam_solve::estimate::reml::jeffreys_subspace::JeffreysHphiDriftBase>,
            CustomFamilyError,
        >>::new();
        Arc::new(move || cached.get_or_init(|| {
        // The exact reduced-information plan authorized this closure before it
        // was returned, so an inactive outer gate performs zero derivative work.
        // Acquire the WHOLE canonical-axis set in ONE batched hook call —
        // the same path the value-path `joint_jeffreys_term` uses — so a family
        // that assembles every axis in one shared softmax/Gram pass (multinomial)
        // pays a SINGLE sweep instead of the `p` concurrent cache-miss sweeps the
        // per-axis fan-out triggered on a fresh β (#1082/#979). An active
        // curvature requires every exact derivative; an absent batch is an
        // error, since substituting zero would change the outer objective's
        // derivative.
        // A family that forms the rotated rows hands them over, and no `p × p` axis
        // matrix is built (#1082).
        let base = match family.jeffreys_rotated_first_derivative() {
            Some(rotated) => {
                let rows = rotated.first_directional_rotated_all_axes(
                    &states,
                    &specs,
                    plan.ambient_eigenbasis().view(),
                )?;
                gam_solve::estimate::reml::jeffreys_subspace::JeffreysHphiDriftBase::prepare_with_plan_rotated_rows(
                    plan.clone(), rows,
                )?
            }
            None => {
                let all_axes = family
                    .joint_jeffreys_information_directional_derivative_all_axes_with_specs(
                        &states,
                        &specs,
                    )?;
                let axes = all_axes.ok_or_else(|| {
                    CustomFamilyError::trial_point(
                        "active Jeffreys drift requires exact first information derivatives".to_string(),
                    )
                })?;
                gam_solve::estimate::reml::jeffreys_subspace::JeffreysHphiDriftBase::prepare_with_plan_axes(
                    plan.clone(), axes,
                )?
            }
        };
        base.map(Arc::new).ok_or_else(|| CustomFamilyError::trial_point(
            "active Jeffreys drift could not prepare its information derivative base".to_string()
        ))
        }).clone())
    };
    let prepare_first = Arc::clone(&prepare_base);
    // gam#2894: a family that exposes the completion's contracted-trace derivatives can
    // have its criterion priced on the complete curvature. The drifts read this snapshot.
    let completion_derivatives = family
        .jeffreys_completion_outer_derivatives()
        .is_some()
        .then(|| {
            (
                Arc::clone(&prepare_base),
                Arc::clone(&family_owned),
                Arc::clone(&states_owned),
                Arc::clone(&specs_owned),
            )
        });
    let completion_beta = {
        let prepare = Arc::clone(&prepare_base);
        let family = Arc::clone(&family_owned);
        let states = Arc::clone(&states_owned);
        let specs = Arc::clone(&specs_owned);
        // #979: the completion is only ever read along pairs of coordinate mode
        // responses, `(−v_l, v_k)`, over every ρ pair of the dense outer Hessian and
        // every column of the operator route's pair right-hand side. Its information
        // derivative `H[u]` and all-axes `H²[v,·]` therefore repeat across pairs and
        // across operator applications. Both are kept for this coefficient snapshot,
        // one entry per coordinate response, keyed by exact bit pattern; the third
        // derivative stays per pair.
        let kept_first = Arc::new(std::sync::Mutex::new(Vec::new()));
        let kept_second = Arc::new(std::sync::Mutex::new(Vec::new()));
        Arc::new(move |u: &Array1<f64>, v: &Array1<f64>| -> Result<Array1<f64>, CustomFamilyError> {
            let missing = || CustomFamilyError::trial_point("active Jeffreys mode response requires exact completion derivatives");
            let base = prepare()?;
            let h = kept_along_response(&kept_first, u, || {
                family
                    .joint_jeffreys_information_directional_derivative_with_specs(&states, &specs, u)?
                    .ok_or_else(missing)
            })?;
            // The rotation into the base's eigenbasis is kept with the derivative, so
            // a pair only rotates its own third derivative.
            let rotated = |direction: &Array1<f64>| {
                kept_along_response(&kept_second, direction, || {
                    // Only the rotation of `H²[d, ·]` is read, so the family forms it (#1082).
                    let mut rows = None;
                    let complete = family
                        .joint_jeffreys_information_second_directional_rotated_all_axes_each_with_specs(
                            &states,
                            &specs,
                            std::slice::from_ref(direction),
                            base.ambient_eigenbasis(),
                            &mut |_, axis_rows| {
                                rows = Some(axis_rows);
                                Ok(())
                            },
                        )?;
                    let rows = rows.filter(|_| complete).ok_or_else(missing)?;
                    Ok(base.rotated_axes_from_rows(rows)?)
                })
            };
            let axes_v = rotated(v)?;
            // #1082: where the conditioning gate or the relative floor moves, the
            // completion carries their motion, and its drift also reads `H²[u,·]`.
            let axes_u = if base.hessian_motion_active() { Some(rotated(u)?) } else { None };
            // Only the rotation of `H³[u, v, ·]` is read, so the family forms it (#1082).
            let moving = family
                .jeffreys_third_information_derivative()
                .ok_or_else(missing)?
                .third_directional_rotated_all_axes(&states, &specs, u, v, base.ambient_eigenbasis())?
                .ok_or_else(missing)?;
            let moving = base.rotated_axes_from_rows(moving)?;
            Ok(base.completion_drift_action_from_rotated(v, &h, &axes_v, axes_u.as_deref(), &moving)? * strength)
        })
    };
    let first = Arc::new(move |deltas: &[Array1<f64>]| {
        if deltas.is_empty() {
            return Ok(Vec::new());
        }
        // The coefficient snapshot is immutable. Share its base across first
        // and second derivatives and across every subsequent operator apply,
        // not merely across directions in a single callback invocation.
        let base = prepare_first()?;
        // Per direction, only `pert_h = Hdot[δ]` and the all-axes
        // `{H²dot[δ,e_a]}` depend on δ. The all-axes objects of the whole batch come
        // from ONE family request, so a family whose all-axes object contracts a
        // per-snapshot row object builds that object once for the batch (#979).
        let pert_hs = deltas
            .iter()
            .map(|delta| {
                family_owned
                    .joint_jeffreys_information_directional_derivative_with_specs(
                        &states_owned,
                        &specs_owned,
                        delta,
                    )?
                    .ok_or_else(|| {
                        "active Jeffreys drift requires an exact first information derivative"
                            .to_string()
                    })
            })
            .collect::<Result<Vec<_>, String>>()
            .map_err(CustomFamilyError::trial_point)?;
        let mut derivatives: Vec<Option<Array2<f64>>> = vec![None; deltas.len()];
        // A family with a contraction pass hands over `⟨H²dot[δ, e_a], K_b⟩` against the
        // base's ambient kernels. Those close the drift exactly as the rotated rows do,
        // without forming the `p` axis matrices of any direction (#2668).
        if let Some(contractor) = family_owned.jeffreys_axis_contractions() {
            contractor
                .second_directional_axis_contractions_each(
                    &states_owned,
                    &specs_owned,
                    deltas,
                    &|| base.ambient_axis_kernels(),
                    &mut |index, contractions| {
                        let mut derivative = base.perturbation_derivative_from_axis_contractions(
                            &pert_hs[index],
                            &contractions,
                        )?;
                        if strength != 1.0 {
                            derivative *= strength;
                        }
                        derivatives[index] = Some(derivative);
                        Ok(())
                    },
                )
                .map_err(CustomFamilyError::trial_point)?;
            if derivatives.iter().any(Option::is_none) {
                return Err(CustomFamilyError::trial_point(
                    "active Jeffreys drift contraction pass skipped a direction".to_string(),
                ));
            }
            return Ok(derivatives);
        }
        // Only the rotations of `{H²dot[δ, e_a]}` are read, so the family forms them (#1082).
        let complete = family_owned
            .joint_jeffreys_information_second_directional_rotated_all_axes_each_with_specs(
                &states_owned,
                &specs_owned,
                deltas,
                base.ambient_eigenbasis(),
                &mut |index, rows| {
                    let axes = base.rotated_axes_from_rows(rows)?;
                    let mut derivative =
                        base.perturbation_derivative_from_rotated_axes(&pert_hs[index], &axes)?;
                    if strength != 1.0 {
                        derivative *= strength;
                    }
                    derivatives[index] = Some(derivative);
                    Ok(())
                },
            )
            .map_err(CustomFamilyError::trial_point)?;
        if !complete {
            return Err(CustomFamilyError::trial_point(
                "active Jeffreys drift requires exact second information derivatives".to_string(),
            ));
        }
        Ok(derivatives)
    });
    let second = Arc::new(move |pairs: &[(Array1<f64>, Array1<f64>)]| {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }
        let missing = |derivative: &str| {
            CustomFamilyError::trial_point(format!(
                "active Jeffreys outer Hessian requires exact {derivative}; family does not expose it"
            ))
        };
        let base = prepare_base()?;
        // #979: an outer Hessian over `k` coordinates batches `k(k+1)/2` pairs
        // drawn from `k` mode responses. Each pair used to re-derive `H[u]`,
        // `{H²[u,e_a]}` and their spectral rows for both of its directions, so
        // the dense survival marginal-slope Hessian ran seven family sweeps where
        // two suffice. Directions are matched by exact bit pattern (the batch
        // carries clones of the same response vectors), never by tolerance, and
        // the mixed third derivative stays per pair.
        let mut distinct: Vec<&Array1<f64>> = Vec::new();
        let pair_frames: Vec<(usize, usize)> = pairs
            .iter()
            .map(|(u, v)| {
                (
                    distinct_direction_slot(&mut distinct, u),
                    distinct_direction_slot(&mut distinct, v),
                )
            })
            .collect();
        let frame_perturbations = distinct
            .iter()
            .map(|direction| {
                family_second
                    .joint_jeffreys_information_directional_derivative_with_specs(
                        &states_second,
                        &specs_second,
                        direction,
                    )?
                    .ok_or_else(|| missing("first information derivative"))
            })
            .collect::<Result<Vec<_>, CustomFamilyError>>()?;
        // The distinct directions' all-axes objects come from one family request.
        let frame_directions: Vec<Array1<f64>> =
            distinct.iter().map(|direction| (*direction).clone()).collect();
        let mut frame_slots = Vec::with_capacity(frame_directions.len());
        frame_slots.resize_with(frame_directions.len(), || None);
        let complete = family_second
            .joint_jeffreys_information_second_directional_rotated_all_axes_each_with_specs(
                &states_second,
                &specs_second,
                &frame_directions,
                base.ambient_eigenbasis(),
                &mut |index, rows| {
                    frame_slots[index] = Some(base.direction_frame_from_rotated(
                        &frame_perturbations[index],
                        base.rotated_axes_from_rows(rows)?,
                    )?);
                    Ok(())
                },
            )
            .map_err(CustomFamilyError::trial_point)?;
        if !complete {
            return Err(missing("second information derivatives"));
        }
        let frames = frame_slots
            .into_iter()
            .map(|slot| slot.ok_or_else(|| missing("second information derivatives")))
            .collect::<Result<Vec<_>, CustomFamilyError>>()?;
        // Each pair reads only its own two directions and the immutable snapshot,
        // so the `k(k+1)/2` pairs of a dense outer Hessian run concurrently. Every
        // pair is evaluated and the first refusal in pair order is returned, so the
        // drift and any refusal are the serial ones (#1082).
        use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
        let evaluated: Vec<Result<Array2<f64>, CustomFamilyError>> = pairs
            .par_iter()
            .zip(pair_frames.par_iter())
            .map(|((u, v), &(frame_u, frame_v))| {
                let huv = family_second
                    .joint_jeffreys_information_second_directional_derivative_with_specs(
                        &states_second,
                        &specs_second,
                        u,
                        v,
                    )?
                    .ok_or_else(|| missing("second information derivative"))?;
                let axes_uv = family_second
                    .jeffreys_third_information_derivative()
                    .ok_or_else(|| {
                        missing("third information derivatives (fifth likelihood derivatives)")
                    })?
                    .third_directional_rotated_all_axes(
                        &states_second,
                        &specs_second,
                        u,
                        v,
                        base.ambient_eigenbasis(),
                    )?
                    .ok_or_else(|| {
                        missing("third information derivatives (fifth likelihood derivatives)")
                    })?;
                let axes_uv = base
                    .rotated_axes_from_rows(axes_uv)
                    .map_err(CustomFamilyError::trial_point)?;
                let mut derivative = base
                    .mixed_perturbation_derivative_from_frames(
                        &frames[frame_u],
                        &frames[frame_v],
                        &huv,
                        &axes_uv,
                    )
                    .map_err(CustomFamilyError::trial_point)?;
                derivative *= strength;
                Ok(derivative)
            })
            .collect();
        evaluated
            .into_iter()
            .collect::<Result<Vec<_>, CustomFamilyError>>()
    });
    // gam#2894: the criterion's completion drifts, `D_β completion[δ]` and `D² completion[u, w]`,
    // formed on the same drift base as `H_Φ`'s.
    let (completion_first, completion_second): (
        Option<CompletionDriftFn>,
        Option<CompletionSecondDriftFn>,
    ) = match completion_derivatives {
        Some((prepare, family, states, specs)) => {
            let missing = |derivative: &str| {
                CustomFamilyError::trial_point(format!(
                    "a criterion priced on the complete Jeffreys curvature requires exact \
                     {derivative} (gam#2894)"
                ))
            };
            let completion_first: CompletionDriftFn = {
                let (prepare, family, states, specs) = (
                    Arc::clone(&prepare),
                    Arc::clone(&family),
                    Arc::clone(&states),
                    Arc::clone(&specs),
                );
                Arc::new(move |deltas: &[Array1<f64>]| {
                    let base = prepare()?;
                    // The rotated `{H²[δ, e_a]}` feeds only the gate and floor motion, so it is
                    // formed only where they move, and then every direction's axes come from one
                    // family request (gam#2905).
                    let rotated_seconds = if base.hessian_motion_active() {
                        let mut slots = Vec::with_capacity(deltas.len());
                        slots.resize_with(deltas.len(), || None);
                        let complete = family
                            .joint_jeffreys_information_second_directional_rotated_all_axes_each_with_specs(
                                &states,
                                &specs,
                                deltas,
                                base.ambient_eigenbasis(),
                                &mut |index, rows| {
                                    slots[index] = Some(base.rotated_axes_from_rows(rows)?);
                                    Ok(())
                                },
                            )
                            .map_err(CustomFamilyError::trial_point)?;
                        if !complete {
                            return Err(missing("second information derivatives"));
                        }
                        slots
                            .into_iter()
                            .map(|slot| slot.ok_or_else(|| missing("second information derivatives")))
                            .collect::<Result<Vec<_>, CustomFamilyError>>()?
                    } else {
                        Vec::new()
                    };
                    deltas
                        .iter()
                        .enumerate()
                        .map(|(index, delta)| -> Result<Array2<f64>, CustomFamilyError> {
                            let information = family
                                .joint_jeffreys_information_directional_derivative_with_specs(
                                    &states, &specs, delta,
                                )?
                                .ok_or_else(|| missing("first information derivatives"))?;
                            let contracted =
                                |weight: &Array2<f64>| -> Result<Array2<f64>, String> {
                                    family
                                        .joint_jeffreys_information_contracted_trace_hessian_with_specs(
                                            &states, &specs, weight,
                                        )?
                                        .ok_or_else(|| {
                                            "priced Jeffreys completion requires the contracted \
                                             trace Hessian"
                                                .to_string()
                                        })
                                };
                            let along = |weight: &Array2<f64>| -> Result<Array2<f64>, String> {
                                family
                                    .jeffreys_completion_outer_derivatives()
                                    .ok_or_else(|| {
                                        "priced Jeffreys completion requires the completion outer derivatives"
                                            .to_string()
                                    })?
                                    .contracted_trace_hessian_directional(
                                        &states, &specs, weight, delta,
                                    )?
                                    .ok_or_else(|| {
                                        "priced Jeffreys completion requires the directional \
                                         contracted trace Hessian"
                                            .to_string()
                                    })
                            };
                            let mut drift = base.completion_drift_matrix(
                                &information,
                                rotated_seconds.get(index),
                                &contracted,
                                &along,
                            )?;
                            if strength != 1.0 {
                                drift *= strength;
                            }
                            Ok(drift)
                        })
                        .collect::<Result<Vec<_>, CustomFamilyError>>()
                })
            };
            let completion_second: CompletionSecondDriftFn =
                Arc::new(move |pairs: &[(Array1<f64>, Array1<f64>)]| {
                    let base = prepare()?;
                    // Where the gate or the floor moves, the complete second drift also reads
                    // `H²[u,·]`, `H²[w,·]` and `H³[u,w,·]` (gam#2905).
                    let motion = base.hessian_motion_active();
                    // An outer Hessian over `k` coordinates asks for `k(k+1)/2` pairs drawn from `k`
                    // mode responses. `H[x]` and, under motion, the rotated `{H²[x, e_a]}` read one
                    // direction each, so they are formed once per distinct direction (matched by
                    // exact bit pattern) instead of twice per pair; the mixed objects stay per pair.
                    let mut distinct: Vec<&Array1<f64>> = Vec::new();
                    let pair_frames: Vec<(usize, usize)> = pairs
                        .iter()
                        .map(|(u, w)| {
                            (
                                distinct_direction_slot(&mut distinct, u),
                                distinct_direction_slot(&mut distinct, w),
                            )
                        })
                        .collect();
                    let perturbations = distinct
                        .iter()
                        .map(|direction| {
                            family
                                .joint_jeffreys_information_directional_derivative_with_specs(
                                    &states, &specs, direction,
                                )?
                                .ok_or_else(|| missing("first information derivatives"))
                        })
                        .collect::<Result<Vec<_>, CustomFamilyError>>()?;
                    let rotated_seconds = if motion {
                        let directions: Vec<Array1<f64>> =
                            distinct.iter().map(|direction| (*direction).clone()).collect();
                        let mut slots = Vec::with_capacity(directions.len());
                        slots.resize_with(directions.len(), || None);
                        let complete = family
                            .joint_jeffreys_information_second_directional_rotated_all_axes_each_with_specs(
                                &states,
                                &specs,
                                &directions,
                                base.ambient_eigenbasis(),
                                &mut |index, rows| {
                                    slots[index] = Some(base.rotated_axes_from_rows(rows)?);
                                    Ok(())
                                },
                            )
                            .map_err(CustomFamilyError::trial_point)?;
                        if !complete {
                            return Err(missing("second information derivatives"));
                        }
                        slots
                            .into_iter()
                            .map(|slot| slot.ok_or_else(|| missing("second information derivatives")))
                            .collect::<Result<Vec<_>, CustomFamilyError>>()?
                    } else {
                        Vec::new()
                    };
                    pairs
                        .iter()
                        .zip(&pair_frames)
                        .map(|((u, w), &(frame_u, frame_w))| -> Result<Array2<f64>, CustomFamilyError> {
                            let pert_uw = family
                                .joint_jeffreys_information_second_directional_derivative_with_specs(
                                    &states, &specs, u, w,
                                )?
                                .ok_or_else(|| missing("second information derivatives"))?;
                            let contracted =
                                |weight: &Array2<f64>| -> Result<Array2<f64>, String> {
                                    family
                                        .joint_jeffreys_information_contracted_trace_hessian_with_specs(
                                            &states, &specs, weight,
                                        )?
                                        .ok_or_else(|| {
                                            "priced Jeffreys completion requires the contracted \
                                             trace Hessian"
                                                .to_string()
                                        })
                                };
                            let along = |weight: &Array2<f64>,
                                         direction: &Array1<f64>|
                             -> Result<Array2<f64>, String> {
                                family
                                    .jeffreys_completion_outer_derivatives()
                                    .ok_or_else(|| {
                                        "priced Jeffreys completion requires the completion outer derivatives"
                                            .to_string()
                                    })?
                                    .contracted_trace_hessian_directional(
                                        &states, &specs, weight, direction,
                                    )?
                                    .ok_or_else(|| {
                                        "priced Jeffreys completion requires the directional \
                                         contracted trace Hessian"
                                            .to_string()
                                    })
                            };
                            let along_u = |weight: &Array2<f64>| along(weight, u);
                            let along_w = |weight: &Array2<f64>| along(weight, w);
                            let along_uw = |weight: &Array2<f64>| -> Result<Array2<f64>, String> {
                                family
                                    .jeffreys_completion_outer_derivatives()
                                    .ok_or_else(|| {
                                        "priced Jeffreys completion requires the completion outer derivatives"
                                            .to_string()
                                    })?
                                    .contracted_trace_hessian_second_directional(
                                        &states, &specs, weight, u, w,
                                    )?
                                    .ok_or_else(|| {
                                        "priced Jeffreys completion requires the second \
                                         directional contracted trace Hessian"
                                            .to_string()
                                    })
                            };
                            let third_uw = if motion {
                                let rows = family
                                    .jeffreys_third_information_derivative()
                                    .ok_or_else(|| missing("third information derivatives"))?
                                    .third_directional_rotated_all_axes(
                                        &states,
                                        &specs,
                                        u,
                                        w,
                                        base.ambient_eigenbasis(),
                                    )?
                                    .ok_or_else(|| missing("third information derivatives"))?;
                                Some(base.rotated_axes_from_rows(rows)?)
                            } else {
                                None
                            };
                            let mut drift = base.completion_second_drift_matrix(
                                &perturbations[frame_u],
                                &perturbations[frame_w],
                                &pert_uw,
                                rotated_seconds.get(frame_u),
                                rotated_seconds.get(frame_w),
                                third_uw.as_ref(),
                                &contracted,
                                &along_u,
                                &along_w,
                                &along_uw,
                            )?;
                            if strength != 1.0 {
                                drift *= strength;
                            }
                            Ok(drift)
                        })
                        .collect::<Result<Vec<_>, CustomFamilyError>>()
                });
            (Some(completion_first), Some(completion_second))
        }
        None => (None, None),
    };
    Ok(Some(JeffreysHphiDriftBatchFn {
        first,
        second,
        completion_beta,
        completion_psi: None,
        response_scale: 1.0,
        completion_first,
        completion_second,
        // The assembly that installs the mode response knows whether its operator carries a
        // completion (gam#2765).
        completion_present: false,
        completion_derivatives_supplied: family.jeffreys_third_information_derivative().is_some(),
    }))
}

/// The value kept for `direction`, matched by exact bit pattern, or `compute`'s
/// result, kept for the next request along the same direction.
fn kept_along_response<T>(
    kept: &std::sync::Mutex<Vec<(Vec<u64>, Arc<T>)>>,
    direction: &Array1<f64>,
    compute: impl FnOnce() -> Result<T, CustomFamilyError>,
) -> Result<Arc<T>, CustomFamilyError> {
    let key: Vec<u64> = direction.iter().map(|value| value.to_bits()).collect();
    let find = |entries: &[(Vec<u64>, Arc<T>)]| {
        entries
            .iter()
            .find(|(seen, _)| *seen == key)
            .map(|(_, value)| Arc::clone(value))
    };
    if let Some(value) = find(&kept.lock().unwrap_or_else(std::sync::PoisonError::into_inner)) {
        return Ok(value);
    }
    let value = Arc::new(compute()?);
    let mut entries = kept.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(existing) = find(&entries) {
        return Ok(existing);
    }
    entries.push((key, Arc::clone(&value)));
    Ok(value)
}

/// Index of `direction` among the distinct directions seen so far, by exact bit
/// pattern; a new direction is appended.
fn distinct_direction_slot<'a>(
    distinct: &mut Vec<&'a Array1<f64>>,
    direction: &'a Array1<f64>,
) -> usize {
    let same = |seen: &&Array1<f64>| {
        seen.len() == direction.len()
            && seen
                .iter()
                .zip(direction.iter())
                .all(|(left, right)| left.to_bits() == right.to_bits())
    };
    match distinct.iter().position(same) {
        Some(slot) => slot,
        None => {
            distinct.push(direction);
            distinct.len() - 1
        }
    }
}
